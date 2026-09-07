//! Timing probe for the jagged assist at the shapes the union actually
//! commits: `k_t` consecutive columns of height `n_t` per table type, zero
//! gaps between type regions (`UnionInstance::jagged_heights`).
//!
//! The batched Frobenius assist is the merged jagged/ring-switch transport's
//! dominant cost, and the single-statement assist is the unmerged path's. Both
//! are driven here through public entry points only, so the same probe runs
//! against any revision of `pcs::jagged`.
//!
//! Informational — run with
//! `cargo test -p flock-core --release --test assist_blocked -- --ignored --nocapture`.

use std::{hint::black_box, time::Instant};

use flock_core::{
    challenger::FsChallenger,
    field::F128,
    pcs::jagged::{
        FrobeniusClaim, JaggedParams, f_hat_t, prove_assist, prove_frobenius_assist, verify_assist,
        verify_frobenius_assist,
    },
};

/// xorshift — the probe only needs points that aren't structurally special.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f128(&mut self) -> F128 {
        F128::new(self.next_u64(), self.next_u64())
    }
    fn vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.f128()).collect()
    }
}

/// Registry-shaped heights: `regions` of `(run, height)`, zero-padded to `2^k`.
fn registry_heights(regions: &[(usize, u64)], k: usize) -> Vec<u64> {
    let mut h = vec![0u64; 1usize << k];
    let mut at = 0usize;
    for &(run, height) in regions {
        for slot in h.iter_mut().skip(at).take(run) {
            *slot = height;
        }
        // A zero column between regions, as the slot offsets produce.
        at += run + 1;
    }
    h
}

fn min_ms(iters: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
#[ignore] // Timing probe — run explicitly with --ignored --nocapture
fn assist_shapes_probe() {
    // (label, regions, k, n, m, iters). The merged transport's own shape is the
    // first: two types at a few hundred rows over tens of chunk-columns. The
    // 4096-column cases are the scale `multipoint_twisted_bench` guards the
    // batched assist off at (`run_assist = false`) — its per-statement suffix
    // state, dense, is ~6.5 MB, so ~1.7 GB across the 256 statements. Fewer
    // iterations there: each one allocates that state 256 times over.
    let cases: [(&str, Vec<(usize, u64)>, usize, usize, usize, usize); 7] = [
        (
            "2 types x 32 cols, n_t = 300/700",
            vec![(32, 300), (32, 700)],
            9,
            10,
            16,
            8,
        ),
        (
            "4 types x 48 cols, n_t = 1024..8192",
            vec![(48, 1024), (48, 2048), (48, 4096), (48, 8192)],
            10,
            13,
            21,
            8,
        ),
        (
            "2 types x 64 cols, n_t = 65536 (m = 24)",
            vec![(64, 65536), (64, 65536)],
            9,
            17,
            24,
            8,
        ),
        (
            "odd stride: 96 cols of n_t = 1373",
            vec![(96, 1373)],
            7,
            11,
            18,
            8,
        ),
        (
            "full height: 368 cols of 2^14 (the old bench shape)",
            vec![(368, 1 << 14)],
            9,
            14,
            23,
            8,
        ),
        (
            "4096 cols of 2^11 (the multipoint bench's 4K shape)",
            vec![(4096, 1 << 11)],
            13,
            11,
            23,
            3,
        ),
        (
            "4096 cols of n_t = 300 (odd stride at 4K)",
            vec![(4096, 300)],
            13,
            9,
            21,
            3,
        ),
    ];

    for (label, regions, k, n, m, iters) in cases {
        let heights = registry_heights(&regions, k);
        let params = JaggedParams::from_heights(&heights, n, m);
        let mut rng = Rng(0xA551_5700 ^ (m as u64) << 8);

        // ---- Batched Frobenius assist (the merged transport): K = 2 claims,
        // 128 Frobenius powers each = 256 statements.
        let (zr_a, zc_a, ca) = (rng.vec(n), rng.vec(k), rng.vec(128));
        let (zr_b, zc_b, cb) = (rng.vec(n), rng.vec(k), rng.vec(128));
        let claims = [
            FrobeniusClaim {
                z_row: &zr_a,
                z_col: &zc_a,
                coeffs: &ca,
            },
            FrobeniusClaim {
                z_row: &zr_b,
                z_col: &zc_b,
                coeffs: &cb,
            },
        ];
        let rho = rng.vec(m);
        let mut fp = FsChallenger::new(b"assist-blocked-probe");
        let fproof = prove_frobenius_assist(&params, &claims, &[], &rho, &mut fp);
        let fprove = min_ms(iters, || {
            let mut ch = FsChallenger::new(b"assist-blocked-probe");
            black_box(prove_frobenius_assist(&params, &claims, &[], &rho, &mut ch));
        });
        let fverify = min_ms(iters, || {
            let mut ch = FsChallenger::new(b"assist-blocked-probe");
            black_box(verify_frobenius_assist(
                &params,
                &claims,
                &[],
                &rho,
                &fproof,
                &mut ch,
            ));
        });
        let mut vch = FsChallenger::new(b"assist-blocked-probe");
        assert!(
            verify_frobenius_assist(&params, &claims, &[], &rho, &fproof, &mut vch).is_some(),
            "frobenius assist must verify [{label}]"
        );

        // ---- Single-statement assist (spec-level `prove_assist`; its
        // transport caller was removed with the jagged transport, the
        // function remains the Lemma 4.6 reference).
        let (zr, zc, zi) = (rng.vec(n), rng.vec(k), rng.vec(m));
        let mut ap = FsChallenger::new(b"assist-blocked-probe");
        let aproof = prove_assist(&params, &zr, &zc, &zi, &mut ap);
        let aprove = min_ms(iters, || {
            let mut ch = FsChallenger::new(b"assist-blocked-probe");
            black_box(prove_assist(&params, &zr, &zc, &zi, &mut ch));
        });
        let averify = min_ms(iters, || {
            let mut ch = FsChallenger::new(b"assist-blocked-probe");
            black_box(verify_assist(&params, &zr, &zc, &zi, &aproof, &mut ch));
        });
        let mut vch = FsChallenger::new(b"assist-blocked-probe");
        assert_eq!(
            verify_assist(&params, &zr, &zc, &zi, &aproof, &mut vch),
            Some(aproof.beta),
            "assist must verify [{label}]"
        );
        // The value is shape-independent, so it also pins the collapse.
        assert_eq!(aproof.beta, f_hat_t(&params, &zr, &zc, &zi));

        println!(
            "{label:<46} m={m:<3} frobenius(256 stmt) prove {fprove:6.2} verify {fverify:5.2} |  \
             single prove {aprove:5.2} verify {averify:5.2}  (ms)"
        );
    }
}
