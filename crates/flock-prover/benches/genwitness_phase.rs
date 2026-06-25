//! Focused micro-benchmark for `generate_witness_with_ab_packed_and_lincheck`
//! (the "gen_witness_ab + lincheck" phase). Best-of-N to isolate it from the
//! rest of the prove pipeline's thermal load. Honors RAYON_NUM_THREADS.

use std::hint::black_box;
use std::time::Instant;

use flock_prover::r1cs_hashes::blake3::{
    Blake3Setup, Compression, generate_witness_with_ab_packed_and_lincheck, min_n_blocks_log,
};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as u32
    }
}

fn random_compression(rng: &mut Rng) -> Compression {
    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    for &n_blocks in &[32768usize, 65536] {
        let n_log = min_n_blocks_log(n_blocks);
        let _setup = Blake3Setup::new(n_blocks);
        let mut rng = Rng::new(0xC0FFEE ^ n_blocks as u64);
        let blocks: Vec<Compression> = (0..n_blocks)
            .map(|_| random_compression(&mut rng))
            .collect();

        // Warm up.
        for _ in 0..3 {
            let r = generate_witness_with_ab_packed_and_lincheck(&blocks, n_log);
            black_box(&r);
        }

        let n_runs = 12;
        let mut best = f64::INFINITY;
        let mut sum = 0.0;
        for _ in 0..n_runs {
            let t = Instant::now();
            let r = generate_witness_with_ab_packed_and_lincheck(&blocks, n_log);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            best = best.min(ms);
            sum += ms;
            black_box(&r);
        }
        println!(
            "n={n_blocks:>6} (m={})  gen_witness: best {:7.2} ms   avg {:7.2} ms",
            n_log + 14,
            best,
            sum / n_runs as f64
        );
    }
}
