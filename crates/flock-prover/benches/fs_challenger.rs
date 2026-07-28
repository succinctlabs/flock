//! Fiat-Shamir transcript cost, per hash.
//!
//! End-to-end prove timings cannot resolve the FS hash choice: the transcript
//! is a low-single-digit percentage of a proof, which is the same order as
//! run-to-run noise on this machine (and smaller than the position bias you
//! get from benchmarking configurations in a fixed order). This measures the
//! challenger directly instead, where the work is deterministic and the signal
//! is clean.
//!
//! Three measurements per hash:
//! 1. **Absorption** — `observe_f128`, the transcript's bulk input.
//! 2. **Squeezing** — `sample_f128` and `sample_f128_vec`. This is where the
//!    two genuinely differ: SHA-256 is not an XOF so it re-finalizes per
//!    32 bytes (`SHA256(state ‖ ctr)`), while BLAKE3 finalizes once into an
//!    XOF reader. The gap should widen with the squeeze length.
//! 3. **PoW inner loop** — one hash of the 40-byte `state ‖ nonce` pre-image,
//!    the operation `grind_pow` repeats ~2^bits times.
//!
//! Run: `cargo bench --bench fs_challenger`

use std::hint::black_box;
use std::time::Instant;

use flock_core::challenger::{Challenger, FsChallenger};
use flock_core::field::F128;
use flock_core::hash::HashKind;
use sha2::{Digest, Sha256};

const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

/// Repeats per measurement; reported as best-of, which is the run least
/// perturbed by other work on the machine.
const REPS: usize = 7;

fn best<T>(mut f: impl FnMut() -> T) -> f64 {
    // Warm-up.
    black_box(f());
    let mut b = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        black_box(f());
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn report(label: &str, secs: f64, ops: u64) {
    println!(
        "  {:<44}  {:>9.3} ms  {:>9.1} ns/op  {:>10.2} Mop/s",
        label,
        secs * 1e3,
        secs * 1e9 / ops as f64,
        ops as f64 / secs / 1e6,
    );
}

fn main() {
    println!("Fiat-Shamir transcript cost by hash (best of {REPS}).");

    // ---- 1. Absorption.
    println!("\n===== observe_f128 (transcript absorption) =====");
    const N_OBSERVE: usize = 1 << 16;
    for kind in KINDS {
        let secs = best(|| {
            let mut ch = FsChallenger::with_hash(b"fs-bench", kind);
            for i in 0..N_OBSERVE {
                ch.observe_f128(F128 {
                    lo: i as u64,
                    hi: !(i as u64),
                });
            }
            ch.sample_f128()
        });
        report(
            &format!("{kind}: {N_OBSERVE} × observe_f128"),
            secs,
            N_OBSERVE as u64,
        );
    }

    // ---- 2. Squeezing, scalar.
    println!("\n===== sample_f128 (16-byte squeeze each) =====");
    const N_SAMPLE: usize = 1 << 14;
    for kind in KINDS {
        let secs = best(|| {
            let mut ch = FsChallenger::with_hash(b"fs-bench", kind);
            let mut acc = F128::ZERO;
            for _ in 0..N_SAMPLE {
                acc += ch.sample_f128();
            }
            acc
        });
        report(
            &format!("{kind}: {N_SAMPLE} × sample_f128"),
            secs,
            N_SAMPLE as u64,
        );
    }

    // ---- 2b. Squeezing, vector. SHA-256 pays one finalize per 32 output
    // bytes here; BLAKE3 pays one finalize total, so the gap should grow with
    // the request size.
    println!("\n===== sample_f128_vec (one squeeze of 16n bytes) =====");
    for n in [4usize, 16, 64, 256] {
        for kind in KINDS {
            const CALLS: usize = 1 << 10;
            let secs = best(|| {
                let mut ch = FsChallenger::with_hash(b"fs-bench", kind);
                let mut acc = F128::ZERO;
                for _ in 0..CALLS {
                    acc += ch.sample_f128_vec(n)[0];
                }
                acc
            });
            report(
                &format!(
                    "{kind}: {CALLS} × sample_f128_vec({n})  [{} B/squeeze]",
                    n * 16
                ),
                secs,
                CALLS as u64,
            );
        }
    }

    // ---- 3. PoW inner loop, one nonce at a time. The pre-image length is
    // per-hash: SHA-256 uses 40 bytes (one compression), BLAKE3 uses a padded
    // 64-byte whole block so the search can be SIMD-batched.
    println!("\n===== PoW inner loop, scalar (one nonce at a time) =====");
    const N_POW: usize = 1 << 20;
    let state = [0xA5u8; 32];
    for kind in KINDS {
        let secs = best(|| {
            let mut acc = 0u8;
            for nonce in 0..N_POW as u64 {
                let h: [u8; 32] = match kind {
                    HashKind::Sha256 => {
                        let mut pre = [0u8; 40];
                        pre[..32].copy_from_slice(&state);
                        pre[32..].copy_from_slice(&nonce.to_le_bytes());
                        Sha256::digest(pre).into()
                    }
                    HashKind::Blake3 => {
                        let mut pre = [0u8; 64];
                        pre[..32].copy_from_slice(&state);
                        pre[32..40].copy_from_slice(&nonce.to_le_bytes());
                        *blake3::hash(&pre).as_bytes()
                    }
                };
                acc ^= h[0];
            }
            acc
        });
        report(
            &format!("{kind}: {N_POW} × H(pre-image)"),
            secs,
            N_POW as u64,
        );
    }

    // ---- 3b. The real thing: `grind_pow` at the bit levels the m=30 fast
    // profile actually uses (fold_grinding_bits [17,12,9,6,4]). Deterministic
    // per (domain, script, bits): the same transcript finds the same nonce and
    // therefore does the same work every run.
    println!("\n===== grind_pow (search for a satisfying nonce) =====");
    for bits in [9u32, 12, 15, 17] {
        for kind in KINDS {
            let secs = best(|| {
                let mut ch = FsChallenger::with_hash(b"fs-bench-grind", kind);
                ch.observe_bytes(b"root");
                ch.grind_pow(bits)
            });
            // Report against expected attempts (2^bits) rather than 1 op, so
            // ns/op is the effective per-nonce cost including parallelism.
            let expected = 1u64 << bits;
            report(&format!("{kind}: grind_pow({bits})"), secs, expected);
        }
    }

    // ---- 4. A transcript shaped like a real proof: interleaved observes and
    // samples, which is what the sumcheck rounds actually do.
    println!("\n===== mixed script (2 observes + 1 sample, ×N) =====");
    const N_MIXED: usize = 1 << 14;
    for kind in KINDS {
        let secs = best(|| {
            let mut ch = FsChallenger::with_hash(b"fs-bench", kind);
            let mut acc = F128::ZERO;
            for i in 0..N_MIXED {
                ch.observe_f128(F128 {
                    lo: i as u64,
                    hi: 0,
                });
                ch.observe_f128(F128 {
                    lo: 0,
                    hi: i as u64,
                });
                acc += ch.sample_f128();
            }
            acc
        });
        report(&format!("{kind}: {N_MIXED} rounds"), secs, N_MIXED as u64);
    }
}
