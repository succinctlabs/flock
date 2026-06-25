//! PCS commit phase benchmark — pack + parallel RS encode + Merkle.
//!
//! Measures the full `pcs::commit` flow at typical witness sizes:
//! - pack bits into F_{2^128} elements
//! - allocate codeword buffer, zero-pad
//! - forward additive NTT (uses NEON + rayon when available)
//! - Merkle tree over the codeword
//!
//! Run: `cargo bench --bench pcs_commit`

use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::merkle;
use flock_prover::ntt::AdditiveNttF128;
use flock_prover::pcs::{LOG_PACKING, PcsParams, commit, pack_witness};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn bits(&mut self, n: usize) -> Vec<bool> {
        // Each next_u64 gives 64 bits; cheaper than 1 call per bit.
        let mut v = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mut w = self.next_u64();
            for _ in 0..64 {
                if i == n {
                    break;
                }
                v.push((w & 1) == 1);
                w >>= 1;
                i += 1;
            }
        }
        v
    }
}

fn header(name: &str) {
    println!("\n===== {name} =====");
}

fn fmt_secs(s: f64) -> String {
    if s < 1e-3 {
        format!("{:>6.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:>6.2} ms", s * 1e3)
    } else {
        format!("{:>6.2} s ", s)
    }
}

fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b < 1024.0 {
        format!("{:>6.0} B", b)
    } else if b < 1024.0 * 1024.0 {
        format!("{:>5.1} KB", b / 1024.0)
    } else if b < 1024.0 * 1024.0 * 1024.0 {
        format!("{:>5.1} MB", b / 1024.0 / 1024.0)
    } else {
        format!("{:>5.2} GB", b / 1024.0 / 1024.0 / 1024.0)
    }
}

/// Measure each step of the commit separately. Reproduces the body of
/// `pcs::commit` but with `Instant` calls between phases.
fn bench_commit_breakdown(m: usize) {
    let params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    let n_bits = 1usize << m;
    let n_packed = 1usize << params.log_msg_len();
    let n_code = params.codeword_len_f128();
    let bytes_code = (n_code as u64) * 16;

    println!(
        "  m={m}: witness {} bits, packed {} × F128 ({}), code {} × F128 ({})",
        n_bits,
        n_packed,
        fmt_bytes((n_packed as u64) * 16),
        n_code,
        fmt_bytes(bytes_code),
    );

    let mut rng = Rng::new(0xC0FFEE ^ (m as u64));
    let z = rng.bits(n_bits);

    // ---- 1. Pack (now once at the boundary; not counted as commit-internal).
    let t0 = Instant::now();
    let packed_witness = pack_witness(&z, m);
    let secs_pack = t0.elapsed().as_secs_f64();

    // ---- 2. Allocate + copy + zero-pad. (Commit-internal from here onward.)
    let t0 = Instant::now();
    let mut codeword = vec![F128::ZERO; n_code];
    codeword[..packed_witness.len()].copy_from_slice(&packed_witness);
    let secs_alloc = t0.elapsed().as_secs_f64();

    // ---- 3. Interleaved NTT.
    let ntt = AdditiveNttF128::standard(params.k_code());
    let t0 = Instant::now();
    ntt.forward_transform_interleaved(&mut codeword, params.num_ntts());
    let secs_ntt = t0.elapsed().as_secs_f64();

    // ---- 4. Serialize codeword to bytes.
    let t0 = Instant::now();
    let codeword_bytes: Vec<u8> = codeword
        .iter()
        .flat_map(|f| {
            let mut b = [0u8; 16];
            b[0..8].copy_from_slice(&f.lo.to_le_bytes());
            b[8..16].copy_from_slice(&f.hi.to_le_bytes());
            b
        })
        .collect();
    let secs_ser = t0.elapsed().as_secs_f64();

    // ---- 5. Merkle tree (leaves of num_ntts × 16 bytes).
    let t0 = Instant::now();
    let merkle_tree = merkle::merkle_tree(&codeword_bytes, params.n_positions());
    let secs_merkle = t0.elapsed().as_secs_f64();
    let _root = merkle_tree.last().unwrap();
    let leaf_bytes = params.leaf_size_bytes();
    let n_leaves = params.n_positions();
    println!(
        "    leaves: {} × {}  (vs single-NTT: {} × 16 B)",
        n_leaves,
        fmt_bytes(leaf_bytes as u64),
        n_leaves * params.num_ntts(),
    );

    let total = secs_pack + secs_alloc + secs_ntt + secs_ser + secs_merkle;

    println!(
        "    pack    {}    alloc/pad {}",
        fmt_secs(secs_pack),
        fmt_secs(secs_alloc)
    );
    println!(
        "    NTT     {}    serialize {}    merkle  {}",
        fmt_secs(secs_ntt),
        fmt_secs(secs_ser),
        fmt_secs(secs_merkle)
    );
    println!("    -------------------------------------",);
    println!("    total   {}", fmt_secs(total));

    // Sanity: ensure the breakdown matches the integrated `commit` function
    // (called on the already-packed witness, no internal pack step).
    let z_packed = pack_witness(&z, m);
    let t0 = Instant::now();
    let _ = commit(&z_packed, &params);
    let secs_integrated = t0.elapsed().as_secs_f64();
    println!(
        "    (full pcs::commit on packed: {})",
        fmt_secs(secs_integrated)
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes — NEON + parallel NTT path active)");
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    println!("(target: software fallback path)");

    println!("\nLOG_PACKING = {}", LOG_PACKING);

    header("PCS commit at typical R1CS witness sizes (rate-1/2)");
    for &m in &[13usize, 15, 20, 24, 26, 28] {
        bench_commit_breakdown(m);
    }

    header("PCS commit at the production target");
    // m=29 = 2^29 bits = 64 MB witness. Packed = 4M F128 = 64 MB. Code = 8M F128 = 128 MB.
    bench_commit_breakdown(29);

    header("PCS commit at BLAKE3 m=30 / matched-codeword m=31");
    // m=30: codeword = 2^24 F128 = 256 MB (Flock blake3_proof n=65536 size).
    bench_commit_breakdown(30);
    // m=31: codeword = 2^25 F128 = 512 MB (matches binius64 blake3 m=30 commit size).
    // Witness 256 MB packed; bool-vec would be 2 GB so use packed-only path.
    bench_commit_packed_breakdown(31);

    header("PCS commit at LARGE m (witness born packed, no bool step)");
    // At m≥34 the bool witness vec alone is ≥16 GB, so we build the packed
    // witness directly and just measure the commit phase.
    // m=33: packed = 1 GB, code = 2 GB, tree = ~4 GB.
    // m=34: packed = 2 GB, code = 4 GB, tree = ~8 GB.   ← "4 GB codeword"
    // m=35: packed = 4 GB, code = 8 GB, tree = ~16 GB.  ← "4 GB raw witness"
    for &m in &[33usize, 34, 35] {
        bench_commit_packed_only(m);
    }
}

/// Build the packed witness directly (no bool intermediate). For m≥34 the
/// bool vec wouldn't fit in RAM.
fn bench_commit_packed_only(m: usize) {
    let params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    let n_packed = 1usize << params.log_msg_len();
    let n_code = params.codeword_len_f128();
    let bytes_packed = (n_packed as u64) * 16;
    let bytes_code = (n_code as u64) * 16;

    println!(
        "  m={m}: packed {} × F128 ({}), code {} × F128 ({})",
        n_packed,
        fmt_bytes(bytes_packed),
        n_code,
        fmt_bytes(bytes_code),
    );

    // Build packed witness with deterministic random F128 contents.
    let t0 = Instant::now();
    let mut rng = Rng::new(0xC0FFEE ^ (m as u64));
    let mut packed_witness: Vec<F128> = Vec::with_capacity(n_packed);
    for _ in 0..n_packed {
        packed_witness.push(F128 {
            lo: rng.next_u64(),
            hi: rng.next_u64(),
        });
    }
    let secs_build = t0.elapsed().as_secs_f64();
    println!("    build packed: {}", fmt_secs(secs_build));

    // Call the real commit() and time it. The witness lives outside commit
    // (commit takes by reference and doesn't retain).
    let t0 = Instant::now();
    let (_commitment, prover_data) = commit(&packed_witness, &params);
    let secs_commit = t0.elapsed().as_secs_f64();

    let gb_out = bytes_code as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "    pcs::commit: {} ({:.2} GB/s on codeword bytes)",
        fmt_secs(secs_commit),
        gb_out / secs_commit
    );

    // Drop everything explicitly so the next m iteration starts clean.
    drop(prover_data);
    drop(packed_witness);
}

/// Packed-input variant of `bench_commit_breakdown` for sizes where the bool
/// witness wouldn't fit (≥ m=31). Times alloc + NTT + Merkle separately.
fn bench_commit_packed_breakdown(m: usize) {
    let params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
    };
    let n_packed = 1usize << params.log_msg_len();
    let n_code = params.codeword_len_f128();
    let bytes_packed = (n_packed as u64) * 16;
    let bytes_code = (n_code as u64) * 16;

    println!(
        "  m={m}: packed {} × F128 ({}), code {} × F128 ({})",
        n_packed,
        fmt_bytes(bytes_packed),
        n_code,
        fmt_bytes(bytes_code),
    );

    // Build packed witness directly.
    let mut rng = Rng::new(0xC0FFEE ^ (m as u64));
    let mut packed_witness: Vec<F128> = Vec::with_capacity(n_packed);
    for _ in 0..n_packed {
        packed_witness.push(F128 {
            lo: rng.next_u64(),
            hi: rng.next_u64(),
        });
    }

    // ---- 1. Allocate codeword + copy witness + zero-pad upper half.
    let t0 = Instant::now();
    let mut codeword = vec![F128::ZERO; n_code];
    codeword[..packed_witness.len()].copy_from_slice(&packed_witness);
    let secs_alloc = t0.elapsed().as_secs_f64();

    // ---- 2. Interleaved NTT.
    let ntt = AdditiveNttF128::standard(params.k_code());
    let t0 = Instant::now();
    ntt.forward_transform_interleaved(&mut codeword, params.num_ntts());
    let secs_ntt = t0.elapsed().as_secs_f64();

    // ---- 3. Cast codeword bytes (zero-copy — same as pcs::commit).
    let codeword_bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(
            codeword.as_ptr() as *const u8,
            codeword.len() * core::mem::size_of::<F128>(),
        )
    };

    // ---- 4. Merkle tree.
    let t0 = Instant::now();
    let merkle_tree = merkle::merkle_tree(codeword_bytes, params.n_positions());
    let secs_merkle = t0.elapsed().as_secs_f64();
    let _root = merkle_tree.last().unwrap();

    let leaf_bytes = params.leaf_size_bytes();
    let n_leaves = params.n_positions();
    println!(
        "    leaves: {} × {}",
        n_leaves,
        fmt_bytes(leaf_bytes as u64),
    );
    println!(
        "    alloc/pad {}    NTT  {}    merkle  {}",
        fmt_secs(secs_alloc),
        fmt_secs(secs_ntt),
        fmt_secs(secs_merkle),
    );
    let total = secs_alloc + secs_ntt + secs_merkle;
    println!("    -------------------------------------");
    println!("    total   {}", fmt_secs(total));

    let mb_code = bytes_code as f64 / (1024.0 * 1024.0);
    println!(
        "    per-codeword-MB:  NTT {:.3} ms/MB    merkle {:.3} ms/MB    total {:.3} ms/MB",
        secs_ntt * 1000.0 / mb_code,
        secs_merkle * 1000.0 / mb_code,
        total * 1000.0 / mb_code,
    );

    drop(merkle_tree);
    drop(codeword);
    drop(packed_witness);
}
