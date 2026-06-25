//! `lincheck::prove` end-to-end benchmark. Inputs (witness packed, random
//! sparse base matrices, random claim points, dummy claim values) are
//! hoisted outside the timed section — the prover doesn't check honesty
//! against `v, v', v''` so dummy values are fine for benchmarking.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::field::F128;
use flock_prover::lincheck::{QuirkyPoint, SparseMatrixCircuit, prove};
use flock_prover::r1cs::SparseBinaryMatrix;

const K_LOG: usize = 11; // k = 2048
const K_SKIP: usize = 6; // matches zerocheck's univariate-skip dim

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
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
    fn f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.f128()).collect()
    }
    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let len = buf.len();
        let mut i = 0;
        while i + 8 <= len {
            let v = self.next_u64();
            buf[i..i + 8].copy_from_slice(&v.to_le_bytes());
            i += 8;
        }
        if i < len {
            let v = self.next_u64().to_le_bytes();
            buf[i..].copy_from_slice(&v[..len - i]);
        }
    }
}

/// Sparse matrix with ~`nnz` random nonzeros across `k × k` slots.
fn random_sparse_matrix(k: usize, nnz: usize, rng: &mut Rng) -> SparseBinaryMatrix {
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut seen = std::collections::HashSet::new();
    let mut count = 0;
    while count < nnz {
        let r = (rng.next_u64() as usize) % k;
        let c = (rng.next_u64() as usize) % k;
        if seen.insert((r, c)) {
            rows[r].push(c);
            count += 1;
        }
    }
    for row in &mut rows {
        row.sort();
    }
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows,
    }
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("k_log = {K_LOG}, k = {}", 1usize << K_LOG);

    for &m in &[16usize, 20, 24, 26, 28, 29] {
        if m < K_LOG {
            continue;
        }
        let n_log = m - K_LOG;
        let n_bits = 1usize << m;
        let n_bytes = n_bits / 8;
        let k = 1usize << K_LOG;
        println!(
            "\n=== m = {m}, n_log = {n_log} ({n_bits} constraints, {} MB packed) ===",
            n_bytes >> 20
        );

        let mut rng = Rng::new(0xBEEFCAFE + m as u64);

        // Sparse base matrices: ~few thousand nonzeros each, in the target range.
        let nnz_per_mat = 3 * k; // ~6K nonzeros per matrix
        let a_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);
        let b_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);
        // C dropped: in circuit R1CS we assume C = I, which makes the c-claim
        // a direct z-claim that bypasses lincheck.

        let n_runs = if m >= 24 { 3 } else { 1 };

        // Pre-generate n_runs + 1 distinct (witness, claim point) sets so each
        // run hits a fresh FS transcript. The first is the warm-up; the rest
        // are timed.
        let mut witnesses: Vec<(Vec<u8>, QuirkyPoint)> = Vec::with_capacity(n_runs + 1);
        for _ in 0..=n_runs {
            let mut z_packed = vec![0u8; n_bytes];
            rng.fill_bytes(&mut z_packed);
            let x_ab = QuirkyPoint {
                z_skip: rng.f128(),
                x_inner_rest: rng.f128_vec(K_LOG - K_SKIP),
                x_outer: rng.f128_vec(m - K_LOG),
            };
            witnesses.push((z_packed, x_ab));
        }

        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);

        // Warm-up.
        {
            let (z_packed, x_ab) = &witnesses[0];
            let mut ch = FsChallenger::new(b"flock-bench-v0");
            let _ = prove(z_packed, m, K_LOG, K_SKIP, &circuit, x_ab, &mut ch);
        }

        let mut best_ms = f64::INFINITY;
        let mut cs = 0u64;
        for run in 0..n_runs {
            let label = if n_runs == 1 {
                String::from("lincheck::prove")
            } else {
                format!("lincheck::prove (run {})", run + 1)
            };
            let (z_packed, x_ab) = &witnesses[run + 1];
            let mut ch = FsChallenger::new(b"flock-bench-v0");
            let t0 = Instant::now();
            let (proof, claim) = prove(
                black_box(z_packed),
                m,
                K_LOG,
                K_SKIP,
                &circuit,
                black_box(x_ab),
                &mut ch,
            );
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  {:<40} {:>10.2} ms", label, elapsed);
            best_ms = best_ms.min(elapsed);
            cs ^= proof.z_partial[0].lo ^ claim.w.lo;
        }
        if n_runs > 1 {
            println!("  {:<40} {:>10.2} ms", "  (best)", best_ms);
        }
        println!("  checksum: {cs:016x}");
    }
}
