//! Generic (matrix-driven) prove vs the specialized `prove_fast` path —
//! quantifies what the hand-written fused witness generators buy now that
//! the generic apply kernel is fast.
//!
//! Per hash/size, times the two pipelines end-to-end plus the phases that
//! differ:
//!
//!   specialized:  fused gen (z, a, b, z_lincheck emitted inline) ─► prove core
//!   generic:      bool trace gen ─► pack ─► a = A·z, b = B·z (CSR strip
//!                 kernel) ─► prove core
//!
//! Everything downstream (commit, zerocheck, lincheck, open) is shared.
//! Keccak has no generic path at all — its BlockR1cs carries empty matrix
//! stubs by design, so the matrices the generic path needs don't exist.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

fn fmt_ms(s: f64) -> String {
    if s * 1e3 < 1000.0 {
        format!("{:>8.2} ms", s * 1e3)
    } else {
        format!("{:>8.2} s ", s)
    }
}

fn best_of<T, F: FnMut() -> T>(n: usize, mut f: F) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..n {
        let t = Instant::now();
        let v = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(v);
    }
    (best, out.unwrap())
}

fn bench_sha2(n: usize) {
    use flock_prover::r1cs_hashes::sha2::{
        Compression, Sha256HybridSetup, generate_witness_with_ab_packed_and_lincheck,
    };
    let setup = Sha256HybridSetup::new(n);
    let m = setup.m();
    let mut rng = Rng(0x5A2 ^ n as u64);
    let comps: Vec<Compression> = (0..n)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.u32()),
                std::array::from_fn(|_| rng.u32()),
            )
        })
        .collect();

    println!("\n=== sha2, {n} compressions (m = {m}) ===");

    // ---- specialized (fused) path ----
    let (t_gen_fused, _) = best_of(2, || {
        generate_witness_with_ab_packed_and_lincheck(&comps, setup.n_blocks_log())
    });
    let (t_fast, _) = best_of(2, || {
        let mut ch = FsChallenger::new(b"bench-gvf");
        black_box(setup.prove_fast(&comps, &mut ch))
    });

    // ---- generic (matrix-driven) path: phases ----
    let (t_trace, z_packed) = best_of(2, || setup.generate_witness_packed(&comps));
    let t_pack = 0.0f64; // packed trace — no separate pack step
    let (t_apply_a, _a) = best_of(2, || setup.r1cs.apply_a_packed(&z_packed));
    let (t_apply_b, _b) = best_of(2, || setup.r1cs.apply_b_packed(&z_packed));
    let (t_generic, _) = best_of(2, || {
        let mut ch = FsChallenger::new(b"bench-gvf");
        black_box(setup.prove_ligerito(&comps, &mut ch))
    });
    println!(
        "  fast: {}  generic: {}  ratio {:>5.2}x   ({:.0} vs {:.0} H/s)",
        fmt_ms(t_fast),
        fmt_ms(t_generic),
        t_generic / t_fast,
        n as f64 / t_fast,
        n as f64 / t_generic,
    );
    println!(
        "  generic phases:  trace {} | pack {} | A·z {} | B·z {}   (fused gen: {})",
        fmt_ms(t_trace),
        fmt_ms(t_pack),
        fmt_ms(t_apply_a),
        fmt_ms(t_apply_b),
        fmt_ms(t_gen_fused)
    );
}

fn bench_blake3(n: usize) {
    use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};
    let setup = Blake3Setup::new(n);
    let mut rng = Rng(0xB1A ^ n as u64);
    let blocks: Vec<Compression> = (0..n)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.u32()),
                std::array::from_fn(|_| rng.u32()),
                rng.next_u64(),
                rng.u32(),
                64u32,
            )
        })
        .collect();

    println!("\n=== blake3, {n} compressions (m = {}) ===", setup.m());

    let (t_fast, _) = best_of(2, || {
        let mut ch = FsChallenger::new(b"bench-gvf");
        black_box(setup.prove_fast(&blocks, &mut ch))
    });
    let (t_trace, z_packed) = best_of(2, || setup.generate_witness_packed(&blocks));
    let t_pack = 0.0f64;
    let (t_apply_a, _a) = best_of(2, || setup.r1cs.apply_a_packed(&z_packed));
    let (t_generic, _) = best_of(2, || {
        let mut ch = FsChallenger::new(b"bench-gvf");
        black_box(setup.prove_ligerito(&blocks, &mut ch))
    });

    println!("  specialized prove_fast: {}", fmt_ms(t_fast));
    println!(
        "  generic prove:          {}   (trace: {} | pack: {} | A·z: {})",
        fmt_ms(t_generic),
        fmt_ms(t_trace),
        fmt_ms(t_pack),
        fmt_ms(t_apply_a)
    );
    println!(
        "  generic / specialized:  {:>5.2}x   (21M-nnz dense encoding — the generic\n  \
         path's cost is nnz-driven; blake3's substituted encoding is the worst case)",
        t_generic / t_fast
    );
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    println!(
        "generic (matrix-driven) prove vs specialized prove_fast — {} threads",
        rayon::current_num_threads()
    );

    // Sizes with registered default Ligerito configs (m = 22, 29).
    bench_sha2(128); // m = 22
    bench_sha2(16384); // m = 29
    bench_blake3(256); // m = 22
}
