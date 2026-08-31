//! Step-wise e2e comparison of the AG-skip zerocheck vs the RS univariate-skip
//! zerocheck, at m up to 32. Both reduce `a·b = c` over `{0,1}^m` to evaluation
//! claims; this times the prover phases of each from the same thermal state.
//!
//! Phases (comparable rows):
//!   - Round-1 URM:   AG `round1_slp_packed` (Paar SLP)        vs  RS additive-NTT URM.
//!   - skip→mlv fold: AG `fold_and_first_round` (fused, γ-Horner) vs RS fused fold+round2.
//!   - r1 sample:     AG-only (rejection sample of a curve point).
//!   - mlv tail:      AG fused `fold_and_compute_round_pair_into` vs RS fused tail
//!     (shared `multilinear.rs` — same path both sides).
//!
//! The tail is timed under three fold schedules (see [`Sched`]): Classic, the
//! production Lookahead, and the prototype Uniform 4->1. Cross-schedule parity is
//! asserted, so all three provably compute the same thing.
//!
//!   AG_E2E_MS=30,32 cargo bench --bench ag_e2e_zerocheck

// The AG round-1 kernel (and the ag_skip prover entry points this bench
// drives) are aarch64-only; on other arches the bench is a no-op stub.
#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("ag_e2e_zerocheck: aarch64-only bench (NEON AG round-1 kernel)");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    aarch64_only::run()
}

#[cfg(target_arch = "aarch64")]
mod aarch64_only {
    use std::hint::black_box;
    use std::time::Instant;

    use flock_prover::challenger::{Challenger, FsChallenger};
    use flock_prover::field::{F8, F128};
    use flock_prover::genus95_curve_code::round1::round1_slp_packed;
    use flock_prover::genus95_curve_code::{
        Sha256Rng, base_evaluation_functional, sample_random_evaluation_point,
    };
    use flock_prover::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
    use flock_prover::zerocheck::ag_skip::{d_inv, fold_and_first_round, friendly_challenges};
    use flock_prover::zerocheck::multilinear::{
        UniSkipFoldTable, fold_and_compute_round_pair_into, fold_in_place_pair,
        fold1_lookahead_into, fold2_lookahead_into, lookahead_msg_first, lookahead_msg_second,
        round_pair_naive, uni_skip_fold_and_round_pair_optimized_packed,
    };
    use flock_prover::zerocheck::univariate_skip::build_eq;
    use flock_prover::zerocheck::univariate_skip_optimized::{
        c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed,
        small_challenges_ghash,
    };

    const K_SKIP: usize = 6;
    const N_INNER: usize = 7;

    use flock_core::test_rng::Rng;

    /// Site-specific draws kept verbatim from this file's former local `Rng`.
    trait RngExt {
        fn fill_bytes_words(&mut self, buf: &mut [u8]);
    }
    impl RngExt for Rng {
        fn fill_bytes_words(&mut self, buf: &mut [u8]) {
            let mut i = 0;
            while i + 8 <= buf.len() {
                buf[i..i + 8].copy_from_slice(&self.next_u64().to_le_bytes());
                i += 8;
            }
        }
    }

    fn time_ms<R>(f: impl FnOnce() -> R) -> (f64, R) {
        let t0 = Instant::now();
        let r = f();
        (t0.elapsed().as_secs_f64() * 1000.0, r)
    }

    /// `AG_E2E_TAILS=classic` times only the Classic tail — the shipping schedule.
    /// Keeps a paper-facing run lean (no extra fold regenerations, less thermal load)
    /// at the cost of the cross-schedule parity checks. Default times all three.
    fn tails_all() -> bool {
        !matches!(std::env::var("AG_E2E_TAILS").as_deref(), Ok("classic"))
    }

    /// Cross-schedule parity. All three bind `rho_at(0), rho_at(1), …` in the same
    /// order, so the final value must agree; Uniform's message stream is Classic's
    /// minus the first entry, rounds 0/1 having moved into the skip->mlv fold.
    fn check_tails(
        tag: &str,
        msgs_c: &[(F128, F128)],
        fin_c: (F128, F128),
        msgs_la: &[(F128, F128)],
        fin_la: (F128, F128),
        msgs_un: &[(F128, F128)],
        fin_un: (F128, F128),
    ) {
        assert_eq!(
            msgs_c, msgs_la,
            "{tag}: lookahead tail messages differ from classic"
        );
        assert_eq!(
            fin_c, fin_la,
            "{tag}: lookahead final binding differs from classic"
        );
        assert_eq!(
            msgs_un,
            &msgs_c[1..],
            "{tag}: uniform tail messages differ from classic rounds 2.."
        );
        assert_eq!(
            fin_un, fin_c,
            "{tag}: uniform final binding differs from classic"
        );
    }

    /// Which fold schedule the tail runs.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Sched {
        /// One pass per round: read 2s, write s. Sizes n, n/2, n/4, …
        Classic,
        /// Production lookahead. Entry `fold1` (read 2n, write n — no traffic saving,
        /// it only emits two messages) then steady `fold2`. Sizes n, n/2, n/8, n/32, …
        Lookahead,
        /// Every pass 4->1 (read 2s, write s/2). Requires TWO challenges already
        /// pending at tail entry, i.e. the skip->mlv fold must emit the bivariate Q
        /// for rounds 0 and 1 instead of the univariate for round 0. Modelled here by
        /// starting the same loop with `pending2` set. Sizes n, n/4, n/16, …
        Uniform,
    }

    /// Shared multilinear tail, mirroring the production AG loop in
    /// `zerocheck::ag_skip::prove_capture_s_hat_v_c`. Both the RS and AG sides reach
    /// this with `a_mlv.len() == 2^r_rest.len()` and the same challenge indexing
    /// (`r_rest[i+1..]` for round `i`), so it is literally the same code on both.
    ///
    /// `lookahead_on` selects the one-pass-per-TWO-rounds path (`fold1/fold2_lookahead_into`,
    /// second message derived from the bivariate Q); `false` reproduces the classic
    /// one-pass-per-round loop. Challenges are synthetic (no transcript here), which
    /// does not change the work done — only the values.
    /// Returns `(elapsed_ms, round messages, final bound (a,b))`. Classic and lookahead
    /// bind the identical challenge sequence (`rho_at(0), rho_at(1), …`), so both must
    /// return identical messages and final values — asserted by the callers.
    fn mlv_tail(
        mut a_mlv: Vec<F128>,
        mut b_mlv: Vec<F128>,
        r_rest: &[F128],
        sched: Sched,
    ) -> (f64, Vec<(F128, F128)>, (F128, F128)) {
        let n_mlv = r_rest.len();
        let n_in = a_mlv.len();
        // Scratch allocated (and zero-filled) OUTSIDE the timed region, as before.
        let (mut a_nxt, mut b_nxt) = if n_in >= 1024 {
            (vec![F128::ZERO; n_in / 2], vec![F128::ZERO; n_in / 2])
        } else {
            (Vec::new(), Vec::new())
        };
        let rho_at = |i: usize| F128 {
            lo: 0xC0DE + i as u64,
            hi: 0xFACE,
        };
        let mut rounds: Vec<(F128, F128)> = Vec::with_capacity(n_mlv);
        let (t_tail, _) = time_ms(|| {
            let mut rho_prev = rho_at(0);
            // `pending2`: a sampled challenge whose fold is deferred to the next
            // lookahead pass (which folds it with `rho_prev`, 4 -> 1). Uniform starts
            // with one already pending — rounds 0/1 having been emitted upstream —
            // so its very first pass is a 4->1 fold at full size.
            let (mut pending2, mut i) = match sched {
                Sched::Uniform => (Some(rho_at(1)), 2usize),
                _ => (None, 1usize),
            };
            while i < n_mlv {
                let len = a_mlv.len();
                // Same gate as production: needs a round after this one, and a big
                // enough array for the fused path.
                if sched != Sched::Classic && i + 1 < n_mlv && len >= 1024 {
                    let out_len = if pending2.is_some() { len / 4 } else { len / 2 };
                    let (ao, bo) = (&mut a_nxt[..out_len], &mut b_nxt[..out_len]);
                    let q = if let Some(r2) = pending2 {
                        fold2_lookahead_into(
                            &a_mlv,
                            &b_mlv,
                            ao,
                            bo,
                            (rho_prev, r2),
                            &r_rest[i + 2..],
                        )
                    } else {
                        fold1_lookahead_into(
                            &a_mlv,
                            &b_mlv,
                            ao,
                            bo,
                            (rho_prev, F128::ZERO),
                            &r_rest[i + 2..],
                        )
                    };
                    std::mem::swap(&mut a_mlv, &mut a_nxt);
                    std::mem::swap(&mut b_mlv, &mut b_nxt);
                    a_mlv.truncate(out_len);
                    b_mlv.truncate(out_len);
                    let rho_a = rho_at(i);
                    rounds.push(lookahead_msg_first(&q, r_rest[i + 1]));
                    rounds.push(lookahead_msg_second(&q, rho_a));
                    rho_prev = rho_a;
                    pending2 = Some(rho_at(i + 1));
                    i += 2;
                    continue;
                }
                // Leaving lookahead mode: resolve the deferred fold first.
                if let Some(r2) = pending2.take() {
                    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
                    rho_prev = r2;
                    continue;
                }
                let log_before = a_mlv.len().trailing_zeros() as usize;
                let mut r_next = vec![F128::ONE; log_before - 1];
                r_next[1..].copy_from_slice(&r_rest[i + 1..]);
                let pair = if log_before >= 10 {
                    let half = a_mlv.len() / 2;
                    let pair = fold_and_compute_round_pair_into(
                        &a_mlv,
                        &b_mlv,
                        &mut a_nxt[..half],
                        &mut b_nxt[..half],
                        rho_prev,
                        &r_next,
                    );
                    std::mem::swap(&mut a_mlv, &mut a_nxt);
                    std::mem::swap(&mut b_mlv, &mut b_nxt);
                    a_mlv.truncate(half);
                    b_mlv.truncate(half);
                    pair
                } else {
                    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
                    round_pair_naive(&a_mlv, &b_mlv, &r_next)
                };
                rounds.push(pair);
                rho_prev = rho_at(i);
                i += 1;
            }
            // Final binding: one or (after a trailing lookahead pass) two deferred
            // challenges remain.
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            if let Some(r2) = pending2 {
                fold_in_place_pair(&mut a_mlv, &mut b_mlv, r2);
            }
            black_box(&a_mlv);
        });
        assert_eq!(a_mlv.len(), 1, "tail must bind every variable");
        (t_tail, rounds, (a_mlv[0], b_mlv[0]))
    }

    /// RS univariate-skip prover phases (non-padded path), mirroring `prove_packed`.
    fn rs_phases(a: &[u8], b: &[u8], c: &[u8], m: usize) -> (f64, f64, f64, f64, f64) {
        let mut ch = FsChallenger::new(b"flock-bench-v0");
        let n_mlv = m - K_SKIP;
        ch.observe_label(b"flock-zerocheck-v0");
        let r_skip = ch.sample_f128_vec(K_SKIP);
        let r_outer = ch.sample_f128_vec(m - K_SKIP - N_INNER);
        let mut r = vec![F128::ZERO; m];
        r[..K_SKIP].copy_from_slice(&r_skip);
        for (i, v) in small_challenges_ghash().iter().enumerate() {
            r[K_SKIP + i] = *v;
        }
        for (i, v) in medium_challenges_ghash().iter().enumerate() {
            r[K_SKIP + 3 + i] = *v;
        }
        r[K_SKIP + N_INNER..].copy_from_slice(&r_outer);

        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        let inv = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);

        let (t_round1, (r1ab, r1c)) = time_ms(|| {
            round1_shift_reduce_extract_c_packed(
                black_box(a),
                black_box(b),
                black_box(c),
                m,
                K_SKIP,
                &r,
                &inv,
            )
        });
        let cs = c_s_f128();
        let r1ab: Vec<F128> = r1ab.iter().map(|x| cs * *x).collect();
        let r1c: Vec<F128> = r1c.iter().map(|x| cs * *x).collect();
        ch.observe_f128_slice(&r1ab);
        ch.observe_f128_slice(&r1c);
        let z = ch.sample_f128();

        let fold_table = UniSkipFoldTable::new(K_SKIP, z);
        let mut mlv_arg = vec![F128::ONE; n_mlv];
        mlv_arg[1..].copy_from_slice(&r[K_SKIP + 1..]);
        let (t_fold, (a_mlv, b_mlv, _m1, _mi)) = time_ms(|| {
            uni_skip_fold_and_round_pair_optimized_packed(
                black_box(a),
                black_box(b),
                m,
                K_SKIP,
                &fold_table,
                &mlv_arg,
            )
        });

        // Tail twice: classic, then lookahead. The second fold is untimed — it only
        // regenerates the inputs the first tail consumed (cheaper than a 2 GB clone).
        debug_assert_eq!(a_mlv.len(), 1usize << n_mlv);
        let r_rest = &r[K_SKIP..];
        let refold = || {
            uni_skip_fold_and_round_pair_optimized_packed(a, b, m, K_SKIP, &fold_table, &mlv_arg)
        };
        let (t_tail_classic, msgs_c, fin_c) = mlv_tail(a_mlv, b_mlv, r_rest, Sched::Classic);
        if !tails_all() {
            return (t_round1, t_fold, t_tail_classic, 0.0, 0.0);
        }
        let (a2, b2, _, _) = refold();
        let (t_tail_la, msgs_la, fin_la) = mlv_tail(a2, b2, r_rest, Sched::Lookahead);
        let (a3, b3, _, _) = refold();
        let (t_tail_un, msgs_un, fin_un) = mlv_tail(a3, b3, r_rest, Sched::Uniform);
        check_tails("RS", &msgs_c, fin_c, &msgs_la, fin_la, &msgs_un, fin_un);
        (t_round1, t_fold, t_tail_classic, t_tail_la, t_tail_un)
    }

    /// AG-skip prover phases.
    fn ag_phases(a: &[u8], b: &[u8], c: &[u8], m: usize) -> (f64, f64, f64, f64, f64, f64) {
        let mut ch = FsChallenger::new(b"flock-bench-ag-v0");
        ch.observe_label(b"flock-ag-skip-v0");
        let r_outer = ch.sample_f128_vec(m - K_SKIP - N_INNER);
        let eq = build_eq(&r_outer);

        let (t_round1, (res_ab, wbar)) =
            time_ms(|| round1_slp_packed(black_box(a), black_box(b), black_box(c), &eq));
        let di = d_inv();
        let ab_fresh: Vec<F128> = (0..158).map(|s| di * res_ab[s]).collect();
        let c_msg: Vec<F128> = (0..64).map(|i| di * wbar[i]).collect();
        ch.observe_f128_slice(&ab_fresh);
        ch.observe_f128_slice(&c_msg);

        let (t_r1, r1) = time_ms(|| {
            ch.observe_label(b"flock-ag-skip-r1-point");
            let s0 = ch.sample_f128();
            let s1 = ch.sample_f128();
            let mut seed = [0u8; 32];
            seed[0..8].copy_from_slice(&s0.lo.to_le_bytes());
            seed[8..16].copy_from_slice(&s0.hi.to_le_bytes());
            seed[16..24].copy_from_slice(&s1.lo.to_le_bytes());
            seed[24..32].copy_from_slice(&s1.hi.to_le_bytes());
            sample_random_evaluation_point(&mut Sha256Rng::new(seed)).expect("point")
        });

        let mut r_rest = friendly_challenges().to_vec();
        r_rest.extend_from_slice(&r_outer);
        let bf = base_evaluation_functional(&r1).expect("bf");
        let w: Vec<F128> = bf.iter().copied().collect();
        // Fold + first multilinear message, fused (like RS's uni_skip_fold).
        let (t_fold, (a_mlv, b_mlv)) = time_ms(|| {
            let (am, bm, g1, ginf) = fold_and_first_round(black_box(a), black_box(b), &w, &r_rest);
            black_box((g1, ginf));
            (am, bm)
        });

        // Tail: rounds 1..n_mlv, classic then lookahead (same shared loop as RS).
        debug_assert_eq!(a_mlv.len(), 1usize << r_rest.len());
        let (t_tail_classic, msgs_c, fin_c) = mlv_tail(a_mlv, b_mlv, &r_rest, Sched::Classic);
        if !tails_all() {
            return (t_round1, t_r1, t_fold, t_tail_classic, 0.0, 0.0);
        }
        let (a2, b2, _, _) = fold_and_first_round(a, b, &w, &r_rest);
        let (t_tail_la, msgs_la, fin_la) = mlv_tail(a2, b2, &r_rest, Sched::Lookahead);
        let (a3, b3, _, _) = fold_and_first_round(a, b, &w, &r_rest);
        let (t_tail_un, msgs_un, fin_un) = mlv_tail(a3, b3, &r_rest, Sched::Uniform);
        check_tails("AG", &msgs_c, fin_c, &msgs_la, fin_la, &msgs_un, fin_un);
        (t_round1, t_r1, t_fold, t_tail_classic, t_tail_la, t_tail_un)
    }

    pub(super) fn run() {
        let _ = flock_prover::init_perf_thread_pool();
        // `AG_E2E_MS=31,32` overrides the default sweep (comma/space-separated).
        let ms: Vec<usize> = match std::env::var("AG_E2E_MS") {
            Ok(s) => s
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().expect("AG_E2E_MS: integer m"))
                .collect(),
            Err(_) => vec![24, 28, 30],
        };
        for &m in &ms {
            let n_bytes = (1usize << m) / 8;
            println!(
                "\n=== m = {m}  ({} MB / witness, {} MB total) ===",
                n_bytes >> 20,
                (3 * n_bytes) >> 20
            );
            let mut rng = Rng::new(0xABCD_0000 + m as u64);
            let mut a = vec![0u8; n_bytes];
            rng.fill_bytes_words(&mut a);
            let mut b = vec![0u8; n_bytes];
            rng.fill_bytes_words(&mut b);
            let c: Vec<u8> = a.iter().zip(&b).map(|(x, y)| x & y).collect();

            // warm caches (LUT/SLP/eval OnceLocks, NTT tables).
            let _ = rs_phases(&a, &b, &c, m);
            let _ = ag_phases(&a, &b, &c, m);

            let (rs_r1, rs_fold, rs_tail, rs_tail_la, rs_tail_un) = rs_phases(&a, &b, &c, m);
            let (ag_r1, ag_sample, ag_fold, ag_tail, ag_tail_la, ag_tail_un) =
                ag_phases(&a, &b, &c, m);
            let rs_head = rs_r1 + rs_fold;
            let ag_head = ag_r1 + ag_sample + ag_fold;

            println!(
                "  {:<26} {:>10} {:>10} {:>9}",
                "phase", "RS (ms)", "AG (ms)", "RS/AG"
            );
            let row = |name: &str, rs: f64, ag: f64| {
                let spd = if ag > 0.0 {
                    format!("{:.2}x", rs / ag)
                } else {
                    "—".into()
                };
                println!("  {:<26} {:>10.1} {:>10.1} {:>9}", name, rs, ag, spd);
            };
            row("round-1 URM", rs_r1, ag_r1);
            row("skip->mlv fold", rs_fold, ag_fold);
            println!(
                "  {:<26} {:>10} {:>10.1} {:>9}",
                "r1 sample (AG only)", "-", ag_sample, ""
            );
            row("mlv tail", rs_tail, ag_tail);
            row("TOTAL (prove phases)", rs_head + rs_tail, ag_head + ag_tail);
            if tails_all() {
                row("mlv tail (lookahead)", rs_tail_la, ag_tail_la);
                row("mlv tail (uniform 4->1)", rs_tail_un, ag_tail_un);
                row(
                    "TOTAL (lookahead tail)",
                    rs_head + rs_tail_la,
                    ag_head + ag_tail_la,
                );
                row(
                    "TOTAL (uniform tail)",
                    rs_head + rs_tail_un,
                    ag_head + ag_tail_un,
                );
                println!(
                    "  tail vs classic: lookahead RS {:.2}x AG {:.2}x | uniform RS {:.2}x AG {:.2}x",
                    rs_tail / rs_tail_la,
                    ag_tail / ag_tail_la,
                    rs_tail / rs_tail_un,
                    ag_tail / ag_tail_un
                );
                println!(
                    "  (uniform excludes the Q the skip->mlv fold would have to emit — traffic-free there, but not free)"
                );
            }
            println!(
                "  note: AG fold = unchecked byte-dot skip-collapse + γ² wide-shift first message (deferred reduction); round-1 SLP carries the overall win."
            );
        }
    }
}
