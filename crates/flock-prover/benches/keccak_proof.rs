//! End-to-end Keccak-f proof benchmark using the monolithic R1CS, with
//! per-phase timing breakdown. Times the fast prover path
//! (`KeccakSetup::prove_fast`); the slow `prove` path is exercised by
//! unit tests in `src/keccak.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::lincheck::{self, QuirkyPoint};
use flock_prover::pcs;
use flock_prover::r1cs_hashes::keccak::{K_LOG, KeccakSetup, STATE_BITS, State, min_n_keccaks_log};
use flock_prover::zerocheck;

// Peak-heap tracker (wraps System): records the high-water mark of currently
// outstanding bytes, same notion as binius64's peakmem-alloc. Negligible
// overhead (one relaxed atomic op per alloc/dealloc).
struct PeakAlloc;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                let c = CUR.fetch_add(new - l.size(), Ordering::Relaxed) + (new - l.size());
                PEAK.fetch_max(c, Ordering::Relaxed);
            } else {
                CUR.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}
#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;
fn reset_peak() {
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak_mb() -> f64 {
    PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
}

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
}

fn random_state(rng: &mut Rng) -> State {
    let mut s = [false; STATE_BITS];
    let mut i = 0;
    while i < STATE_BITS {
        let w = rng.next_u64();
        for b in 0..64 {
            if i + b < STATE_BITS {
                s[i + b] = (w >> b) & 1 == 1;
            }
        }
        i += 64;
    }
    s
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1.0 {
        format!("{:>8.2} µs", s * 1e6)
    } else if ms < 1000.0 {
        format!("{:>8.2} ms", ms)
    } else {
        format!("{:>8.2} s ", s)
    }
}

fn bench_one(n_keccaks: usize, n_runs: usize) {
    let n_log = min_n_keccaks_log(n_keccaks);
    let m = K_LOG + n_log;
    let n_slots = 1usize << n_log;
    let witness_bytes = (1usize << m) / 8;

    println!(
        "\n=== K = {n_keccaks:>5} Keccaks  (m = {m}, slots = {n_slots}, witness = {} MB) ===",
        witness_bytes >> 20
    );

    let setup = KeccakSetup::new(n_keccaks);
    let mk_states = |seed: u64| {
        let mut rng = Rng::new(seed);
        (0..n_keccaks)
            .map(|_| random_state(&mut rng))
            .collect::<Vec<State>>()
    };
    let state_sets: Vec<Vec<State>> = (0..=n_runs)
        .map(|run| mk_states(0xC0FFEE_BEEF ^ (n_keccaks as u64) ^ (run as u64)))
        .collect();

    // Warm-up (prove_fast).
    {
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (p, _, _) = setup.prove_fast_basefold(&state_sets[0], &mut ch_p);
        black_box(&p);
    }

    // Best-of-n_runs prove_fast. Each run uses a distinct initial-state set so
    // the FS transcript varies across iterations.
    let mut best_fast = f64::INFINITY;
    for run in 0..n_runs {
        let initial_states = &state_sets[run + 1];
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let t0 = Instant::now();
        let (proof, _, _) = setup.prove_fast_basefold(initial_states, &mut ch);
        let elapsed = t0.elapsed().as_secs_f64();
        best_fast = best_fast.min(elapsed);
        black_box(&proof);
        println!(
            "  [run {}/{}] prove_fast: {}",
            run + 1,
            n_runs,
            fmt_ms(elapsed)
        );
    }
    println!("  best prove_fast: {}", fmt_ms(best_fast));

    // Peak memory (heap high-water mark over a single prove) + verify time.
    {
        let states_v = &state_sets[0];
        reset_peak();
        let mut ch_p = FsChallenger::new(b"flock-bench-v0");
        let (proof, commitment, _) = setup.prove_fast_basefold(states_v, &mut ch_p);
        let peak = peak_mb();
        println!("  peak memory: {:>8.2} MB", peak);

        let mut ch_v = FsChallenger::new(b"flock-bench-v0");
        let t = Instant::now();
        let _ = setup
            .verify_basefold(&commitment, &proof, &mut ch_v)
            .expect("verify failed");
        println!("  verify: {}", fmt_ms(t.elapsed().as_secs_f64()));

        // Serialized proof size (commitment + proof, bincode w/ 7-byte header).
        let bundle = flock_prover::proof_io::R1csProofBundle { commitment, proof };
        let proof_size = bundle.to_bytes().len();
        println!(
            "  proof size: {} bytes ({:.2} KiB)",
            proof_size,
            proof_size as f64 / 1024.0
        );
        black_box(&bundle);
    }

    // Inline per-phase breakdown — uses the warm-up state set.
    println!("  [prove_fast breakdown]");
    let initial_states = &state_sets[0];
    let mut ch = FsChallenger::new(b"flock-bench-v0");
    let r1cs = &setup.r1cs;
    let pcs_params = &setup.pcs_params;

    let t = Instant::now();
    let (z_p, a_p, b_p, z_lc) =
        flock_prover::r1cs_hashes::keccak::generate_witness_with_ab_packed_and_lincheck(
            initial_states,
            setup.n_keccaks_log(),
        );
    println!(
        "    {:32} {}",
        "gen_witness_ab + lincheck",
        fmt_ms(t.elapsed().as_secs_f64())
    );

    let t = Instant::now();
    let (commitment, prover_data) = pcs::commit(&z_p, pcs_params);
    println!(
        "    {:32} {}",
        "pcs::commit",
        fmt_ms(t.elapsed().as_secs_f64())
    );

    let a_zc: &[u8] =
        unsafe { std::slice::from_raw_parts(a_p.as_ptr() as *const u8, a_p.len() * 16) };
    let b_zc: &[u8] =
        unsafe { std::slice::from_raw_parts(b_p.as_ptr() as *const u8, b_p.len() * 16) };
    // c == z (C = I).
    let c_zc: &[u8] =
        unsafe { std::slice::from_raw_parts(z_p.as_ptr() as *const u8, z_p.len() * 16) };

    let t = Instant::now();
    let (_zc_proof_dense, _) = zerocheck::prove_packed(
        a_zc,
        b_zc,
        c_zc,
        r1cs.m,
        &mut FsChallenger::new(b"flock-bench-v0"),
    );
    println!(
        "    {:32} {}",
        "zerocheck::prove_packed (dense)",
        fmt_ms(t.elapsed().as_secs_f64())
    );

    let padding = zerocheck::PaddingSpec {
        k_log: r1cs.k_log,
        useful_bits_per_block: r1cs.useful_bits,
    };
    let t = Instant::now();
    let (_zc_proof, zc_claim) =
        zerocheck::prove_packed_padded(a_zc, b_zc, c_zc, r1cs.m, &padding, &mut ch);
    println!(
        "    {:32} {}",
        "zerocheck::prove_packed (padded)",
        fmt_ms(t.elapsed().as_secs_f64())
    );

    let inner_rest_len = r1cs.k_log - r1cs.k_skip;
    let x_ab = QuirkyPoint {
        z_skip: zc_claim.z,
        x_inner_rest: zc_claim.mlv_challenges[..inner_rest_len].to_vec(),
        x_outer: zc_claim.mlv_challenges[inner_rest_len..].to_vec(),
    };

    let lc_circuit = flock_prover::r1cs_hashes::keccak::KeccakLincheckCircuit;
    let t = Instant::now();
    let (_lc_proof, lc_claim) = lincheck::prove(
        &z_lc,
        r1cs.m,
        r1cs.k_log,
        r1cs.k_skip,
        &lc_circuit,
        &x_ab,
        &mut ch,
    );
    println!(
        "    {:32} {}",
        "lincheck::prove",
        fmt_ms(t.elapsed().as_secs_f64())
    );

    let mut x_ab_full = lc_claim.r_inner_rest.clone();
    x_ab_full.extend_from_slice(&x_ab.x_outer);
    let mut x_c_full = zc_claim.r_rest[..inner_rest_len].to_vec();
    x_c_full.extend_from_slice(&zc_claim.r_rest[inner_rest_len..]);

    let t = Instant::now();
    let _open = pcs::open_batch(
        &z_p,
        &prover_data,
        &commitment,
        &[&x_ab_full, &x_c_full],
        &mut ch,
    );
    println!(
        "    {:32} {}",
        "pcs::open_batch",
        fmt_ms(t.elapsed().as_secs_f64())
    );
    black_box(&zc_claim);
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    println!("(target: aarch64 + aes)");
    println!("Full Keccak-f[1600] R1CS proof timings (prove_fast).");

    // Sizes to bench. By default: 1 and 64 give small-scale context, 4096 = 2^12
    // and 16384 = 2^14 are the competitor-comparison points. Override with
    // KECCAK_LOG2S (space/comma-separated log2 keccak counts, e.g. "16 18") —
    // used by benchmarks/bench_keccak.sh to sweep Flock at the same sizes as
    // the competitors; each listed size is benched best-of-3.
    let specs: Vec<(usize, usize)> = match std::env::var("KECCAK_LOG2S") {
        Ok(s) => s
            .split([',', ' '])
            .filter(|t| !t.is_empty())
            .map(|t| {
                let h: u32 = t
                    .parse()
                    .expect("KECCAK_LOG2S: space/comma-separated integer log2 values");
                (1usize << h, 3usize)
            })
            .collect(),
        Err(_) => vec![(1usize, 3), (64, 2), (4096, 2), (16384, 2)],
    };
    for &(k, n_runs) in &specs {
        bench_one(k, n_runs);
    }
}
