//! In-process A/B oracle for the F256 ladder's round-1 lookahead skip on the
//! MERGED transport — the union / `prove_fast` route, where the seeded
//! EqPoint combine emits BLOCKED coefficients and the ladder consumes them
//! through the virtual-basis double fold. The skip is an exact polynomial
//! identity, so proofs must be byte-identical with it forced on vs forced
//! off (`FOLD_LOOKAHEAD_OVERRIDE`), across the L0 OOD β-glues the blocked
//! coefficient corrections keep it live through.
//!
//! (The compression-proof / materialized path's identity is pinned by
//! `f256_round1_lookahead_is_byte_identical` in flock-core; the historical
//! byte anchors are the union m6 fixtures + the mixed-class pins.)

use std::sync::atomic::Ordering;

use flock_prover::challenger::FsChallenger;
use flock_prover::pcs::ligerito::FOLD_LOOKAHEAD_OVERRIDE;
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, Compression};

use flock_core::test_rng::Rng;

fn blocks(n: usize, seed: u64) -> Vec<Compression> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            (cv, m, rng.next_u32() as u64, 64u32, 11u32)
        })
        .collect()
}

fn assert_byte_identical(n_blocks: usize, seed: u64) {
    // The override is PROCESS-GLOBAL and the harness runs tests on parallel
    // threads: without serialization, a concurrent store can land between
    // this store and the prover's read, both proves take the same arm, and
    // the byte-equality assert passes vacuously (an A/A). One lock per
    // A/B, released only after the override is reset.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let setup = Blake3Setup::new(n_blocks);
    let inputs = blocks(n_blocks, seed);
    let prove = |force: u8| {
        FOLD_LOOKAHEAD_OVERRIDE.store(force, Ordering::Relaxed);
        let mut ch = FsChallenger::new(b"la-union-test");
        let out = setup.prove_fast(&inputs, &mut ch);
        FOLD_LOOKAHEAD_OVERRIDE.store(0, Ordering::Relaxed);
        out
    };
    let (p_on, c_on, _) = prove(1);
    let (p_off, c_off, _) = prove(2);
    assert_eq!(
        c_on.cap, c_off.cap,
        "commitment moved (n_blocks={n_blocks})"
    );
    assert_eq!(
        p_on, p_off,
        "lookahead skip moved a byte on the merged transport (n_blocks={n_blocks})"
    );
    let mut ch = FsChallenger::new(b"la-union-test");
    setup
        .verify(&c_on, &p_on, &mut ch)
        .expect("lookahead-path proof must verify");
}

/// Registered m22 fast config: L0 OOD samples > 0, so the blocked per-OOD
/// coefficient corrections are on the line.
#[test]
fn union_prove_fast_lookahead_is_byte_identical_m22() {
    assert_byte_identical(256, 0xAB);
}

/// A second registered shape (m23) and seed.
#[test]
fn union_prove_fast_lookahead_is_byte_identical_m23() {
    assert_byte_identical(512, 0xCD);
}

/// m27: the lane-major block hits the ≥2^16 superblock kernels (the KC-split
/// branches) AND the degenerate pre-switch drain whose array is smaller than
/// a superblock — the shape the small arms never reach.
#[test]
fn union_prove_fast_lookahead_is_byte_identical_m27() {
    assert_byte_identical(8192, 0xEF);
}
