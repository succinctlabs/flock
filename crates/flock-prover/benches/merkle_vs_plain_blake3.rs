//! **Merkle-path table vs plain BLAKE3 table, at matched hash counts.**
//!
//! What does the composite Merkle-path encoding (one table row = one whole
//! depth-26 path, `r1cs_hashes::merkle_r1cs`) cost relative to simply proving
//! the same compressions as independent blocks (`r1cs_hashes::blake3`)?
//!
//! A depth-26 path is 26 BLAKE3 compressions, so `n` paths is compared against
//! `26n` loose compressions — the two sides do the **same hashing work** and
//! differ only in how it is expressed as R1CS. The Merkle side additionally
//! proves the level-to-level dataflow (each level's digest IS the next level's
//! input, plus a conditional swap per level); the plain side proves 26n
//! unrelated compressions and nothing about how they connect. So the plain side
//! is a *lower bound* on a real Merkle-verification circuit, not an equivalent
//! statement — it is the "what does the structure cost" baseline.
//!
//! Both sides run through the SAME union entry
//! (`prove_fast_ligerito_union` /
//! `verify_ligerito_union`) as a single-type registry, so the
//! only difference is the table type. Comparing Merkle-via-union against
//! BLAKE3-via-direct-path would confound the encoding with the proving entry.
//!
//! ## Threading
//!
//! `multi` is this project's calibrated prover pool (`init_perf_thread_pool`,
//! = physical P-cores); `1thr` runs the same code inside a 1-thread pool.
//!
//! The verifier is normally single-threaded by construction (`flock_core::
//! verifier`'s dedicated pool), and that is the honest production number. To
//! also report a parallel verify, re-run with `FLOCK_VERIFY_THREADS=<n>` — the
//! pool is process-wide and read once, so each run measures one verify
//! configuration. Prove numbers are unaffected by that variable.
//!
//! Run:
//! ```text
//! cargo bench -p flock-prover --bench merkle_vs_plain_blake3
//! FLOCK_VERIFY_THREADS=4 cargo bench -p flock-prover --bench merkle_vs_plain_blake3
//! ```

use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
use std::array::from_fn;
use std::env::var;
use std::hint::black_box;
use std::time::Instant;

use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::schedule::{Registry, TableType};
use flock_core::union::UnionInstance;
use flock_core::verifier;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::merkle_r1cs::{MerkleTreeLayout, PathInput, blake3_spec};
use flock_prover::r1cs_hashes::{blake3, merkle_r1cs};

const DOMAIN: &[u8] = b"flock-merkle-vs-blake3-bench-v0";

/// Merkle depth; override with `MVB_DEPTH`. Drives the composite block width
/// `k_log = 14 + log2(next_pow2(depth))`, hence the jagged column count
/// `2^(k_log-7)` that the Frobenius assist scales with — so it is the knob that
/// moves the assist, unlike the path count.
fn depth() -> usize {
    var("MVB_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&d| d >= 1)
        .unwrap_or(26)
}
/// Repetitions per timed phase (after a warm-up). Median is reported. 7 rather
/// than 3: at these sizes prove is 15–350 ms and verify 4–100 ms, small enough
/// that 3 reps left the BLAKE3 column visibly non-monotonic in size.
/// Override with `MVB_REPS` (`MVB_REPS=1` for a quick `VERIFY_TRACE` run).
fn reps() -> usize {
    var("MVB_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(7)
}

/// Path counts to sweep; override with e.g. `MVB_PATHS=8,16`. Each must be a
/// power of two — it is the registry's `2^nu` row capacity.
fn path_counts() -> Vec<usize> {
    match var("MVB_PATHS") {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .inspect(|n| {
                assert!(
                    n.is_power_of_two(),
                    "MVB_PATHS entries must be powers of two"
                )
            })
            .collect(),
        Err(_) => vec![8, 16, 32, 64],
    }
}

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
    fn digest(&mut self) -> [u32; merkle_r1cs::SLOT_WORDS] {
        from_fn(|_| self.next_u32())
    }
    fn path(&mut self, depth: usize) -> PathInput {
        PathInput {
            leaf: self.digest(),
            index: ((self.next_u32() as u64) << 32) | self.next_u32() as u64,
            siblings: (0..depth).map(|_| self.digest()).collect(),
        }
    }
    fn compression(&mut self) -> blake3::Compression {
        let cv: [u32; 8] = from_fn(|_| self.next_u32());
        let m: [u32; 16] = from_fn(|_| self.next_u32());
        let counter = ((self.next_u32() as u64) << 32) | self.next_u32() as u64;
        (cv, m, counter, 64u32, 11u32)
    }
}

/// Median of `REPS` runs of `f`, in seconds, after one warm-up.
fn median<F: FnMut() -> R + Send, R: Send>(pool: Option<&rayon::ThreadPool>, mut f: F) -> f64 {
    let mut run = || {
        let t = Instant::now();
        let r = match pool {
            Some(p) => p.install(&mut f),
            None => f(),
        };
        black_box(&r);
        t.elapsed().as_secs_f64()
    };
    let _ = run(); // warm-up: first touch of caches / lazily-built tables
    let n = reps();
    let mut v: Vec<f64> = (0..n).map(|_| run()).collect();
    v.sort_by(f64::total_cmp);
    v[n / 2]
}

/// One measured configuration.
struct Row {
    label: String,
    /// Hash compressions actually proven.
    hashes: usize,
    /// Table rows and their capacity, `nu` = log2(capacity).
    rows: usize,
    nu: usize,
    /// Committed stack size.
    dense_m: usize,
    /// `k_log` of the table type: the block is `2^k_log` columns wide.
    k_log: usize,
    /// Useful (non-padding) witness bits over all declared rows.
    useful_bits_total: usize,
    /// Nonzeros of the BASE block — what ONE `fold_alpha_batched` walks, and
    /// all that is stored, since `A = I_rows ⊗ A_0`.
    base_nnz: usize,
    /// Nonzeros of the whole constraint system, `rows × base_nnz`: the actual
    /// size of the R1CS being proven.
    system_nnz: usize,
    proof_bytes: usize,
    witness_multi: f64,
    prove_multi: f64,
    prove_solo: f64,
    verify: f64,
}

/// Time one single-type union slot end to end. `make_witness` is re-run per
/// rep, since the prover consumes it.
fn measure(
    label: String,
    hashes: usize,
    rows: usize,
    nu: usize,
    registry: &Registry,
    k_log: usize,
    useful_bits: usize,
    base_nnz: usize,
    circuit: &dyn LincheckCircuit,
    make_witness: &(dyn Fn() -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) + Sync),
    solo: &rayon::ThreadPool,
) -> Row {
    let union = UnionInstance::new(registry, vec![rows]);
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };

    let prove = || {
        let mut ch = FsChallenger::new(DOMAIN);
        prover::prove_fast_ligerito_union(
            &union,
            &pcs_params,
            vec![UnionSlotProverInput::new(make_witness(), circuit)],
            &mut ch,
        )
    };

    // Prove once outside the timing loops to get the artifacts to verify, and
    // to confirm the pair actually round-trips before reporting any number.
    let (proof, commitment, claim) = prove();
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim_v = verifier::verify_ligerito_union(
        &union,
        &[circuit],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("{label}: proof did not verify: {e:?}"));
    assert_eq!(claim.ab.value, claim_v.ab.value, "{label}: ab claim");
    assert_eq!(claim.c.value, claim_v.c.value, "{label}: c claim");

    // Witness generation is inside `prove` (the prover consumes it), so time it
    // alone too — otherwise a gap between the two sides could just be hashing
    // rather than proving.
    let witness_multi = median(None, make_witness);
    let prove_multi = median(None, prove);
    let prove_solo = median(Some(solo), prove);
    let verify = median(None, || {
        let mut ch = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union(
            &union,
            &[circuit],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .expect("verify")
    });

    Row {
        label,
        hashes,
        rows,
        nu,
        dense_m: pcs_params.m,
        k_log,
        useful_bits_total: useful_bits * rows,
        base_nnz,
        system_nnz: base_nnz * rows,
        proof_bytes: bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0),
        witness_multi,
        prove_multi,
        prove_solo,
        verify,
    }
}

fn merkle_row(n_paths: usize, solo: &rayon::ThreadPool) -> Row {
    let d = depth();
    let nu = n_paths.trailing_zeros() as usize;
    let layout = MerkleTreeLayout::new(d, blake3_spec());
    let stub = layout.build_block_r1cs_stub(nu);
    let registry = Registry::new(vec![TableType::from_block_r1cs(&stub)], nu);
    let walker = layout.build_walker();
    let mut rng = Rng(0x_2600_A711);
    let paths: Vec<PathInput> = (0..n_paths).map(|_| rng.path(d)).collect();
    // The composite is `depth` copies of the BLAKE3 block plus the swap gadget
    // and globals, so the base block IS one whole path. `effective_nnz` is what
    // the materialized composite would hold; the walker stores one base copy
    // and the factored fold touches ~1/depth of it.
    let base_nnz = walker.effective_nnz();
    measure(
        format!("merkle d{d} x{n_paths}"),
        n_paths * d,
        n_paths,
        nu,
        &registry,
        layout.k_log,
        layout.useful_bits,
        base_nnz,
        &walker,
        &|| layout.generate_witness_batch_major_partial(&paths, nu),
        solo,
    )
}

fn blake3_row(n_hashes: usize, solo: &rayon::ThreadPool) -> Row {
    // The lincheck floor is n_outer >= 8, so nu >= 3.
    let nu = blake3::min_n_blocks_log(n_hashes).max(3);
    let r1cs = blake3::build_block_r1cs(nu);
    let registry = Registry::new(vec![TableType::from_block_r1cs(&r1cs)], nu);
    let mut rng = Rng(0x_B3_2600_A711);
    let blocks: Vec<blake3::Compression> = (0..n_hashes).map(|_| rng.compression()).collect();
    let base_nnz: usize = r1cs
        .a_0
        .rows
        .iter()
        .chain(r1cs.b_0.rows.iter())
        .map(|r| r.len())
        .sum();
    measure(
        format!("blake3 loose x{n_hashes}"),
        n_hashes,
        n_hashes,
        nu,
        &registry,
        r1cs.k_log,
        r1cs.useful_bits,
        base_nnz,
        r1cs.csc_lincheck_circuit(),
        &|| blake3::generate_witness_batch_major_partial(&blocks, nu),
        solo,
    )
}

fn main() {
    let threads = flock_prover::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let solo = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("1-thread pool");

    println!("Merkle-path table vs plain BLAKE3 table, matched hash counts");
    println!("  prover pool     : {threads} threads (physical P-cores)");
    println!(
        "  verify pool     : {} thread(s){}",
        var("FLOCK_VERIFY_THREADS").unwrap_or_else(|_| "1".into()),
        if var("FLOCK_VERIFY_THREADS").is_ok() {
            " (FLOCK_VERIFY_THREADS override)"
        } else {
            " (production default)"
        }
    );
    println!(
        "  depth           : {}  =>  1 path = {} compressions, k_log = {}",
        depth(),
        depth(),
        MerkleTreeLayout::new(depth(), blake3_spec()).k_log
    );
    println!("  reps            : median of {} after warm-up\n", reps());

    let mut rows = Vec::new();
    for n_paths in path_counts() {
        rows.push(merkle_row(n_paths, &solo));
        rows.push(blake3_row(n_paths * depth(), &solo));
    }

    println!(
        "{:<22} {:>7} {:>6} {:>4} {:>8} {:>9} {:>9} {:>10} {:>10} {:>10}",
        "config",
        "hashes",
        "rows",
        "nu",
        "dense_m",
        "proof KiB",
        "wit/mt",
        "prove/mt",
        "prove/1t",
        "verify"
    );
    for r in &rows {
        println!(
            "{:<22} {:>7} {:>6} {:>4} {:>8} {:>9.1} {:>6.0} ms {:>8.0} ms {:>8.0} ms {:>7.1} ms",
            r.label,
            r.hashes,
            r.rows,
            r.nu,
            r.dense_m,
            r.proof_bytes as f64 / 1024.0,
            r.witness_multi * 1e3,
            r.prove_multi * 1e3,
            r.prove_solo * 1e3,
            r.verify * 1e3,
        );
    }

    // Is the Merkle circuit actually bigger? `system_nnz` is the size of the
    // R1CS proven; `base_nnz` is what one fold walks and all that is stored.
    println!("\ncircuit size (is the Merkle system bigger?):");
    println!(
        "{:<22} {:>6} {:>13} {:>9} {:>13} {:>13} {:>11}",
        "config", "k_log", "useful bits", "bits/hash", "base nnz", "system nnz", "nnz/hash"
    );
    for r in &rows {
        println!(
            "{:<22} {:>6} {:>13} {:>9} {:>13} {:>13} {:>11}",
            r.label,
            r.k_log,
            r.useful_bits_total,
            r.useful_bits_total / r.hashes,
            r.base_nnz,
            r.system_nnz,
            r.system_nnz / r.hashes,
        );
    }

    println!("\nper-compression cost (lower is better):");
    println!(
        "{:<22} {:>12} {:>12} {:>12}",
        "config", "prove/mt", "prove/1t", "verify"
    );
    for r in &rows {
        println!(
            "{:<22} {:>9.1} us {:>9.1} us {:>9.1} us",
            r.label,
            r.prove_multi * 1e6 / r.hashes as f64,
            r.prove_solo * 1e6 / r.hashes as f64,
            r.verify * 1e6 / r.hashes as f64,
        );
    }

    println!("\nmerkle / blake3 ratio at equal hash count:");
    for pair in rows.chunks(2) {
        let (m, b) = (&pair[0], &pair[1]);
        assert_eq!(m.hashes, b.hashes, "pair must match on hash count");
        println!(
            "  {:>6} compressions:  prove/mt {:>5.2}x   prove/1t {:>5.2}x   verify {:>5.2}x",
            m.hashes,
            m.prove_multi / b.prove_multi,
            m.prove_solo / b.prove_solo,
            m.verify / b.verify,
        );
    }
}
