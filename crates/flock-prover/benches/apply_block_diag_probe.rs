//! Probe: the generic block-diagonal apply `(I ⊗ M₀)·z` on packed witnesses
//! — the kernel behind the generic (non-fused) `prove()` path's `a = A·z`,
//! `b = B·z`, `c = C·z` materialization.
//!
//! Compares the library implementation (CSR + 8-block strips: the matrix is
//! streamed once per strip and each nonzero's word/shift decode is shared by
//! 8 instances) against a self-contained copy of the previous per-block
//! algorithm (re-walks `Vec<Vec<usize>>` rows for every block).
//!
//! Run MT (default) and ST (`RAYON_NUM_THREADS=1`). Real matrices:
//! sha2 hybrid (k=2^15, ~937k+362k nnz) at several batch sizes, and blake3
//! (k=2^14, ~21M nnz) at a small batch (its generic path is quadratic-ish in
//! density — exactly why the fused generators exist — but the kernel ratio
//! is still meaningful).

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::r1cs::{SparseBinaryMatrix, apply_block_diag_packed};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

// ---------------------------------------------------------------------------
// Reference: the previous library algorithm (per-block, Vec<Vec> row walk),
// copied verbatim so old-vs-new runs in one binary.
// ---------------------------------------------------------------------------

fn apply_block_diag_packed_old(
    m_0: &SparseBinaryMatrix,
    z_packed: &[F128],
    m: usize,
    k_log: usize,
) -> Vec<F128> {
    use rayon::prelude::*;
    let k = 1usize << k_log;
    let n_packed = 1usize << (m - 7);
    assert_eq!(z_packed.len(), n_packed);
    let mut out = vec![F128::ZERO; n_packed];
    let f128_per_block = k / 128;
    out.par_chunks_mut(f128_per_block)
        .zip(z_packed.par_chunks(f128_per_block))
        .for_each(|(out_block, z_block)| {
            apply_one_block_aligned_old(m_0, z_block, out_block, k);
        });
    out
}

fn apply_one_block_aligned_old(
    m_0: &SparseBinaryMatrix,
    z_block: &[F128],
    out_block: &mut [F128],
    k: usize,
) {
    let z_u128: &[u128] =
        unsafe { std::slice::from_raw_parts(z_block.as_ptr() as *const u128, z_block.len()) };
    let f128_per_block = k / 128;
    let mut row_iter = m_0.rows.iter();
    for out_idx in 0..f128_per_block {
        let mut acc: u128 = 0;
        for offset in 0..128 {
            let row = row_iter.next().expect("row count == k");
            let mut row_acc: u128 = 0;
            for &j in row.iter() {
                row_acc ^= (z_u128[j >> 7] >> (j & 127)) & 1;
            }
            acc |= row_acc << offset;
        }
        if acc != 0 {
            let cur = out_block[out_idx];
            out_block[out_idx] = F128 {
                lo: cur.lo | (acc as u64),
                hi: cur.hi | ((acc >> 64) as u64),
            };
        }
    }
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Prototype: STRIP=64 with u64 column bits. Input AND output are
// bit-transposed per strip (64×64 transpose, Hacker's Delight), so each
// matrix nonzero costs one u64 load + XOR covering 64 blocks at once.
// ---------------------------------------------------------------------------

fn transpose_64x64(a: &mut [u64; 64]) {
    let mut j: usize = 32;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k: usize = 0;
        while k < 64 {
            // LSB-first delta swap: high half of a[k] <-> low half of
            // a[k+j] (Hacker's Delight transpose32 is MSB-first; mirrored).
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

fn flatten_csr_local(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    let mut row_ptr = Vec::with_capacity(m.num_rows + 1);
    let mut cols = Vec::new();
    row_ptr.push(0u32);
    for row in &m.rows {
        for &c in row {
            cols.push(c as u32);
        }
        row_ptr.push(cols.len() as u32);
    }
    (row_ptr, cols)
}

fn apply_block_diag_packed_strip64(
    m_0: &SparseBinaryMatrix,
    z_packed: &[F128],
    m: usize,
    k_log: usize,
) -> Vec<F128> {
    use rayon::prelude::*;
    let k = 1usize << k_log;
    let n_packed = 1usize << (m - 7);
    let f128_per_block = k / 128;
    let u64_per_block = k / 64;
    let (row_ptr, cols) = flatten_csr_local(m_0);
    let mut out = vec![F128::ZERO; n_packed];

    const S: usize = 64;
    out.par_chunks_mut(S * f128_per_block)
        .zip(z_packed.par_chunks(S * f128_per_block))
        .for_each(|(out_strip, z_strip)| {
            let n_blocks = z_strip.len() / f128_per_block;
            assert_eq!(n_blocks, S, "probe requires n_outer % 64 == 0");
            let z_u64: &[u64] = unsafe {
                std::slice::from_raw_parts(z_strip.as_ptr() as *const u64, z_strip.len() * 2)
            };
            let out_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(
                    out_strip.as_mut_ptr() as *mut u64,
                    out_strip.len() * 2,
                )
            };

            // Transpose in: colbits[j] = bit j of all 64 blocks.
            let mut colbits = vec![0u64; k];
            let mut lanes = [0u64; 64];
            for w in 0..u64_per_block {
                for s in 0..S {
                    lanes[s] = z_u64[s * u64_per_block + w];
                }
                transpose_64x64(&mut lanes);
                colbits[w * 64..w * 64 + 64].copy_from_slice(&lanes);
            }

            // One matrix pass: rowbits[r] = XOR of colbits over row r's cols.
            let mut rowbits = vec![0u64; k];
            for r in 0..k {
                let lo = row_ptr[r] as usize;
                let hi = row_ptr[r + 1] as usize;
                let mut x = 0u64;
                for &j in &cols[lo..hi] {
                    x ^= colbits[j as usize];
                }
                rowbits[r] = x;
            }

            // Transpose out: block-major.
            for w in 0..u64_per_block {
                lanes.copy_from_slice(&rowbits[w * 64..w * 64 + 64]);
                transpose_64x64(&mut lanes);
                for s in 0..S {
                    out_u64[s * u64_per_block + w] = lanes[s];
                }
            }
        });
    out
}

fn best_of<F: FnMut() -> Vec<F128>>(n: usize, mut f: F) -> (f64, Vec<F128>) {
    let mut best = f64::INFINITY;
    let mut out = Vec::new();
    for _ in 0..n {
        let t = Instant::now();
        out = f();
        best = best.min(t.elapsed().as_secs_f64());
        black_box(&out);
    }
    (best, out)
}

fn probe(name: &str, a_0: &SparseBinaryMatrix, k_log: usize, n_log: usize, runs: usize) {
    let m = k_log + n_log;
    let nnz: usize = a_0.rows.iter().map(|r| r.len()).sum();
    let n_outer = 1usize << n_log;
    let mut rng = Rng(0xD1A6 ^ m as u64);
    let z: Vec<F128> = (0..(1usize << (m - 7))).map(|_| rng.f128()).collect();

    let (t_old, out_old) = best_of(runs, || apply_block_diag_packed_old(a_0, &z, m, k_log));
    let (t_new, out_new) = best_of(runs, || apply_block_diag_packed(a_0, &z, m, k_log));
    assert_eq!(out_old, out_new, "{name}: old/new mismatch");
    if (1usize << (m - k_log)).is_multiple_of(64) {
        let (t_64, out_64) = best_of(runs, || apply_block_diag_packed_strip64(a_0, &z, m, k_log));
        assert_eq!(out_old, out_64, "{name}: strip64 mismatch");
        println!(
            "  strip64 prototype: {:>9.2} ms ({:>6.2} Gop/s)   vs strip8 {:>5.2}x",
            t_64 * 1e3,
            (a_0.rows.iter().map(|r| r.len()).sum::<usize>() as f64)
                * ((1usize << (m - k_log)) as f64)
                / t_64
                / 1e9,
            t_new / t_64,
        );
    }

    // "Useful work" rate: nnz bit-ops per block × blocks.
    let ops = (nnz as f64) * (n_outer as f64);
    println!(
        "{name}: m={m} (k=2^{k_log}, {n_outer} blocks, {nnz} nnz)\n  \
         old {:>9.2} ms ({:>6.2} Gop/s)   new {:>9.2} ms ({:>6.2} Gop/s)   speedup {:>5.2}x",
        t_old * 1e3,
        ops / t_old / 1e9,
        t_new * 1e3,
        ops / t_new / 1e9,
        t_old / t_new,
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    println!(
        "generic apply_block_diag_packed: old (per-block Vec<Vec>) vs new (CSR + 8-block strips)"
    );
    println!("threads: {}\n", rayon::current_num_threads());

    let (sha2_a, sha2_b) = flock_prover::r1cs_hashes::sha2::build_matrices();
    for n_log in [10usize, 12, 14] {
        probe("sha2 A0", &sha2_a, 15, n_log, 3);
    }
    probe("sha2 B0", &sha2_b, 15, 12, 3);

    let (bl_a, _) = flock_prover::r1cs_hashes::blake3::build_matrices();
    probe("blake3 A0", &bl_a, 14, 6, 3);

    // e2e context: the generic (non-fused) sha2 prove materializes a = A·z
    // and b = B·z through this kernel (c = z since C = I). Everything else
    // in the prove (commit/zerocheck/lincheck/open) is unchanged.
    {
        use flock_prover::challenger::FsChallenger;
        use flock_prover::r1cs_hashes::sha2::Sha256HybridSetup;
        let n = 1usize << 10; // m = 25
        let setup = Sha256HybridSetup::new(n);
        let mut rng = Rng(0xE2E);
        let comps: Vec<([u32; 8], [u32; 16])> = (0..n)
            .map(|_| {
                (
                    std::array::from_fn(|_| rng.next_u64() as u32),
                    std::array::from_fn(|_| rng.next_u64() as u32),
                )
            })
            .collect();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let mut ch = FsChallenger::new(b"probe-e2e");
            let t = Instant::now();
            let p = setup.prove(&comps, &mut ch);
            best = best.min(t.elapsed().as_secs_f64());
            black_box(&p);
        }
        println!(
            "\ne2e generic (non-fused) sha2 prove, m=25, 1024 compressions: {:.2} ms",
            best * 1e3
        );
    }
}
