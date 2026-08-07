//! **The recursion target: proving the verifier's L0 openings.**
//!
//! A Ligerito verifier at dense m = 25 checks 218 L0 Merkle openings: each
//! is a 1 KiB codeword leaf hashed as a BLAKE3 chunk (16 compressions) and a
//! depth-13 path of PARENT-node compressions — 29 compressions in one
//! dependent chain, exactly what `MerkleTreeLayout::with_blake3_chunk_leaf`
//! encodes as ONE table row (bit-compatible with `flock_core::merkle`'s
//! BLAKE3 mode, see `chunk_root_matches_flock_core_blake3_tree`).
//!
//! This bench proves that workload through the union entry and compares it
//! against proving the same compressions as loose BLAKE3 blocks — the
//! structure-free lower bound, as in `merkle_vs_plain_blake3`.
//!
//! Knobs: `MLO_PATHS` (default 218 — need not be a power of two; capacity is
//! the next power of two), `MLO_DEPTH` (default 13), `MLO_LEAF` (leaf bytes,
//! default 1024), `MLO_REPS` (default 5), `FLOCK_VERIFY_THREADS` as in the
//! sibling bench.
//!
//! Run:
//! ```text
//! cargo bench -p flock-prover --bench merkle_l0_opening
//! MLO_PATHS=106,218 MLO_DEPTH=11 cargo bench -p flock-prover --bench merkle_l0_opening
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
use flock_prover::r1cs_hashes::blake3;
use flock_prover::r1cs_hashes::merkle_r1cs::{
    ChunkPathInput, MerkleTreeLayout, SLOT_WORDS, blake3_spec,
};

const DOMAIN: &[u8] = b"flock-merkle-l0-opening-bench-v0";

fn env_usize(name: &str, default: usize) -> usize {
    var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn depth() -> usize {
    env_usize("MLO_DEPTH", 13)
}

fn leaf_bytes() -> usize {
    env_usize("MLO_LEAF", 1024)
}

fn reps() -> usize {
    env_usize("MLO_REPS", 5)
}

/// Path counts to sweep; any positive count (capacity rounds up).
fn path_counts() -> Vec<usize> {
    match var("MLO_PATHS") {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect(),
        Err(_) => vec![218],
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
    fn digest(&mut self) -> [u32; SLOT_WORDS] {
        from_fn(|_| self.next_u32())
    }
    fn opening(&mut self, d: usize, leaf: usize) -> ChunkPathInput {
        ChunkPathInput {
            leaf_data: (0..leaf).map(|_| self.next_u32() as u8).collect(),
            index: ((self.next_u32() as u128) << 32 | self.next_u32() as u128) & ((1u128 << d) - 1),
            siblings: (0..d).map(|_| self.digest()).collect(),
        }
    }
    fn compression(&mut self) -> blake3::Compression {
        let cv: [u32; 8] = from_fn(|_| self.next_u32());
        let m: [u32; 16] = from_fn(|_| self.next_u32());
        let counter = ((self.next_u32() as u64) << 32) | self.next_u32() as u64;
        (cv, m, counter, 64u32, 11u32)
    }
}

/// Median of `reps()` runs of `f`, in seconds, after one warm-up.
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
    let _ = run();
    let n = reps();
    let mut v: Vec<f64> = (0..n).map(|_| run()).collect();
    v.sort_by(f64::total_cmp);
    v[n / 2]
}

struct Row {
    label: String,
    hashes: usize,
    rows: usize,
    nu: usize,
    dense_m: usize,
    k_log: usize,
    useful_bits_total: usize,
    base_nnz: usize,
    system_nnz: usize,
    proof_bytes: usize,
    witness_multi: f64,
    prove_multi: f64,
    prove_solo: f64,
    verify: f64,
}

#[allow(clippy::too_many_arguments)]
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

fn l0_row(n_paths: usize, solo: &rayon::ThreadPool) -> Row {
    let (d, leaf) = (depth(), leaf_bytes());
    let nu = n_paths.next_power_of_two().trailing_zeros().max(3) as usize;
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(d, leaf, blake3_spec());
    let stub = layout.build_block_r1cs_stub(nu);
    let registry = Registry::new(vec![TableType::from_block_r1cs(&stub)], nu);
    let walker = layout.build_walker();
    let mut rng = Rng(0x_10_2600_A711);
    let paths: Vec<ChunkPathInput> = (0..n_paths).map(|_| rng.opening(d, leaf)).collect();
    let base_nnz = walker.effective_nnz();
    measure(
        format!("L0 open d{d} L{leaf} x{n_paths}"),
        n_paths * layout.total_blocks(),
        n_paths,
        nu,
        &registry,
        layout.k_log,
        layout.useful_bits,
        base_nnz,
        &walker,
        &|| layout.generate_witness_batch_major_partial_chunk(&paths, nu),
        solo,
    )
}

fn blake3_row(n_hashes: usize, solo: &rayon::ThreadPool) -> Row {
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

    let (d, leaf) = (depth(), leaf_bytes());
    let blocks_per_row = leaf / 64 + d;
    println!("L0-opening table (BLAKE3 chunk leaf + PARENT path) vs loose BLAKE3");
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
        "  shape           : {leaf} B leaf ({} chunk blocks) + depth {d}  =>  \
         1 opening = {blocks_per_row} compressions, k_log = {}",
        leaf / 64,
        MerkleTreeLayout::with_blake3_chunk_leaf(d, leaf, blake3_spec()).k_log
    );
    println!("  reps            : median of {} after warm-up\n", reps());

    let mut rows = Vec::new();
    for n_paths in path_counts() {
        rows.push(l0_row(n_paths, &solo));
        rows.push(blake3_row(n_paths * blocks_per_row, &solo));
    }

    println!(
        "{:<24} {:>7} {:>6} {:>4} {:>8} {:>9} {:>9} {:>10} {:>10} {:>10}",
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
            "{:<24} {:>7} {:>6} {:>4} {:>8} {:>9.1} {:>6.0} ms {:>8.0} ms {:>8.0} ms {:>7.1} ms",
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

    println!("\ncircuit size:");
    println!(
        "{:<24} {:>6} {:>13} {:>9} {:>13} {:>13} {:>11}",
        "config", "k_log", "useful bits", "bits/hash", "base nnz", "system nnz", "nnz/hash"
    );
    for r in &rows {
        println!(
            "{:<24} {:>6} {:>13} {:>9} {:>13} {:>13} {:>11}",
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
        "{:<24} {:>12} {:>12} {:>12}",
        "config", "prove/mt", "prove/1t", "verify"
    );
    for r in &rows {
        println!(
            "{:<24} {:>9.1} us {:>9.1} us {:>9.1} us",
            r.label,
            r.prove_multi * 1e6 / r.hashes as f64,
            r.prove_solo * 1e6 / r.hashes as f64,
            r.verify * 1e6 / r.hashes as f64,
        );
    }

    println!("\nL0-table / loose-blake3 ratio at equal hash count:");
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
