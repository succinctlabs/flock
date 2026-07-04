//! End-to-end hash-chain proof benchmark: `KeccakSetup::prove_chain` vs the
//! base `prove_fast` (no chain), to isolate the chain overhead at scale, plus
//! the standalone region-fold + shift-sumcheck costs.
//!
//! Run: `cargo run --release --example keccak_chain_bench`

use std::hint::black_box;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::field::F128;
use flock_prover::r1cs_hashes::chain_common::{ChainFold, fold_in_out};
use flock_prover::r1cs_hashes::keccak::{
    CHAIN_LAYOUT, KeccakSetup, STATE_BITS, State, generate_witness_with_ab_packed_and_lincheck,
    keccak_f,
};

struct Rng(u64);
impl Rng {
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn state(&mut self) -> State {
        let mut s = [false; STATE_BITS];
        for b in s.iter_mut() {
            *b = self.nx() & 1 == 1;
        }
        s
    }
}

fn fmt_ms(s: f64) -> String {
    let ms = s * 1000.0;
    if ms < 1.0 {
        format!("{:>9.2} µs", s * 1e6)
    } else if ms < 1000.0 {
        format!("{:>9.2} ms", ms)
    } else {
        format!("{:>9.3} s ", s)
    }
}

/// Honest chain: inputs[i] = keccak_f^i(x_0); x_last = keccak_f(inputs[N-1]).
fn honest_chain(n: usize, seed: u64) -> (Vec<State>, State, State) {
    let mut rng = Rng(seed);
    let x0 = rng.state();
    let mut inputs = Vec::with_capacity(n);
    let mut cur = x0;
    for _ in 0..n {
        inputs.push(cur);
        keccak_f(&mut cur);
    }
    (inputs, x0, cur)
}

fn bench(n_keccaks: usize, n_runs: usize) {
    let setup = KeccakSetup::new(n_keccaks);
    let n_slots = setup.n_keccak_slots();
    let m = setup.m();
    let (inputs, x0, x_last) = honest_chain(n_slots, 0xC0FFEE ^ n_keccaks as u64);

    println!("\n=== K = {n_keccaks} (m = {m}, slots = {n_slots}) ===");

    // --- warm up both paths + correctness check.
    {
        let mut ch = FsChallenger::new(b"chain-bench");
        let (p, _, _) = setup.prove_fast(&inputs, &mut ch);
        black_box(&p);
        let mut ch = FsChallenger::new(b"chain-bench");
        let (cp, comm) = setup.prove_chain(&inputs, &mut ch);
        let mut chv = FsChallenger::new(b"chain-bench");
        setup
            .verify_chain(&comm, &cp, &x0, &x_last, &mut chv)
            .expect("warmup chain must verify");
        black_box(&cp);
    }

    // --- base prove_fast (no chain).
    let mut best_base = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"chain-bench");
        let t = Instant::now();
        let (p, _, _) = setup.prove_fast(&inputs, &mut ch);
        best_base = best_base.min(t.elapsed().as_secs_f64());
        black_box(&p);
    }

    // --- full prove_chain.
    let mut best_chain = f64::INFINITY;
    for _ in 0..n_runs {
        let mut ch = FsChallenger::new(b"chain-bench");
        let t = Instant::now();
        let (p, _) = setup.prove_chain(&inputs, &mut ch);
        best_chain = best_chain.min(t.elapsed().as_secs_f64());
        black_box(&p);
    }

    // --- isolate the chain-only pieces (region fold + shift sumcheck), using
    //     the prover's witness + a throwaway transcript.
    let (z_packed, _a, _b, _lc) =
        generate_witness_with_ab_packed_and_lincheck(&inputs, setup.n_keccaks_log());
    let mut ch = FsChallenger::new(b"chain-bench-iso");
    let tau_pos: Vec<F128> = {
        use flock_prover::challenger::Challenger;
        ch.sample_f128_vec(CHAIN_LAYOUT.tau_pos_len())
    };
    let fold = ChainFold::new(&CHAIN_LAYOUT, tau_pos);

    let mut best_fold = f64::INFINITY;
    let mut io = (Vec::new(), Vec::new());
    for _ in 0..n_runs {
        let t = Instant::now();
        io = fold_in_out(
            &CHAIN_LAYOUT,
            flock_prover::r1cs::WitnessLayout::RowMajor,
            &z_packed,
            &fold,
        );
        best_fold = best_fold.min(t.elapsed().as_secs_f64());
    }
    let (in_vals, out_vals) = io;

    let mut best_shift = f64::INFINITY;
    for _ in 0..n_runs {
        use flock_prover::challenger::Challenger;
        let mut ch = FsChallenger::new(b"chain-bench-shift");
        let _ = ch.sample_f128(); // keep transcript nondegenerate
        let t = Instant::now();
        let (p, _) = flock_prover::chain::prove_chain_shift(&in_vals, &out_vals, &mut ch);
        best_shift = best_shift.min(t.elapsed().as_secs_f64());
        black_box(&p);
    }

    // --- verify timing.
    let mut ch = FsChallenger::new(b"chain-bench");
    let (cp, comm) = setup.prove_chain(&inputs, &mut ch);
    let mut best_verify = f64::INFINITY;
    for _ in 0..n_runs {
        let mut chv = FsChallenger::new(b"chain-bench");
        let t = Instant::now();
        setup
            .verify_chain(&comm, &cp, &x0, &x_last, &mut chv)
            .unwrap();
        best_verify = best_verify.min(t.elapsed().as_secs_f64());
    }

    let overhead = best_chain - best_base;
    println!("  prove_fast (base):    {}", fmt_ms(best_base));
    println!("  prove_chain (full):   {}", fmt_ms(best_chain));
    println!(
        "  chain overhead:       {}  ({:.1}% of base)",
        fmt_ms(overhead),
        100.0 * overhead / best_base
    );
    println!("    ├─ region fold:     {}", fmt_ms(best_fold));
    println!("    └─ shift sumcheck:  {}", fmt_ms(best_shift));
    println!("  verify_chain:         {}", fmt_ms(best_verify));
}

fn main() {
    println!("Keccak hash-chain proof — prove_fast (base) vs prove_chain (full)");
    // K chosen to step m up toward 29 (m = 17 + n_log).
    for &(k, runs) in &[(8usize, 5), (64, 4), (512, 3), (4096, 3), (16384, 3)] {
        bench(k, runs);
    }
}
