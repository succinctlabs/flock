//! Hash-chain "shift" argument.
//!
//! Given `2^n` independent hash instances committed in one witness `ẑ`, where
//! instance `i` enforces `output_i = h(input_i)` internally, this protocol
//! glues them into a **sequential chain** `x_{i+1} = h(x_i)` with public
//! endpoints `x_0`, `x_{2^n}`.
//!
//! ## What it proves
//!
//! `In`/`Out` are two-variable MLEs over instance index `i ∈ {0,1}^n` and
//! state-bit index `b ∈ {0,1}^{11}`, restrictions of `ẑ`:
//! `In(i,b) = ẑ(i, sel=state_0, b)`, `Out(i,b) = ẑ(i, sel=state_24, b)`.
//!
//! **Step 0 (region fold).** The verifier picks a random `r` and evaluates the
//! bit variable at `r`, collapsing each 1600-bit region to one scalar per
//! instance: `In(i) := In(i,r)`, `Out(i) := Out(i,r)`. By Schwartz–Zippel over
//! the bit dimension (same `r` for both), the per-bit chain reduces to the
//! scalar chain. The obligations are then
//!
//! ```text
//!   Out(i) = In(i+1)   for i = 0 .. 2^n − 2     (internal glue)
//!   In(0)  = x_0(r)                              (input endpoint, public)
//!   Out(2^n − 1) = x_{2^n}(r)                    (output endpoint, public)
//! ```
//!
//! ## The reduction to ONE `ẑ` query
//!
//! `shift(a,b)` is the MLE of the successor relation `b = a+1` ([`shift_mle`]);
//! for boolean `y`, `shift(τ, y) = eq(τ, y−1)`. The char-2 identity (top term
//! cancels) gives `Out(τ) + eq(τ,1ⁿ)·x_last = Σ_y shift(τ,y)·In(y)`. Expand
//! `Out(τ) = Σ_y eq(τ,y)·Out(y)` into the sum:
//!
//! ```text
//!   Σ_y [ shift(τ,y)·In(y) + eq(τ,y)·Out(y) ]  =  eq(τ,1ⁿ)·x_last      (*)
//! ```
//!
//! **Step A — batch the input endpoint (α).** `In(0ⁿ) = Σ_y eq(y,0ⁿ)·In(y)`, a
//! sum over the same `In` table, so fold it in with a random `α`: add
//! `α·eq(y,0ⁿ)` to the `In` weight and `α·x_0(r)` to the claim.
//!
//! **Step B — merge In/Out via the selector bit (s₀).** `state_0`/`state_24`
//! sit in adjacent slots (`sel = 000000`/`000001`), differing in one selector
//! bit `s₀`. Define `g(y,s₀) = ẑ(y, (0,0,0,0,0,s₀), r)`, so `g(y,0)=In(y)`,
//! `g(y,1)=Out(y)`. Sumcheck over `(y, s₀) ∈ {0,1}^{n+1}` with weight
//!
//! ```text
//!   W(y,s₀) = shift(τ,y)·(1+s₀) + eq(τ,y)·s₀ + α·eq(y,0ⁿ)·(1+s₀)
//! ```
//!
//! and public claim `C = eq(τ,1ⁿ)·x_last + α·x_0(r)`.
//!
//! ## Output: a SINGLE MLE-evaluation claim on `ẑ`
//!
//! The combined sumcheck (`n+1` rounds, degree 2) reduces to one opening
//! `g(τ', s₀*) = ẑ(τ', (0⁵,s₀*), r)` at the sumcheck point `(τ', s₀*)`. The
//! verifier computes `W(τ', s₀*)` itself from `shift(τ,τ')`, `eq(τ,τ')`,
//! `eq(τ',0ⁿ)`. Soundness rests on the PCS binding `g(τ',s₀*)` to the committed
//! `ẑ`; the sumcheck (random `τ`, `α`) proves the glue + both endpoints at once.

use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::lincheck::build_eq_table;
use flock_core::zerocheck::multilinear::eq_eval;
use serde::{Deserialize, Serialize};

/// Multilinear extension of the successor relation `b = a + 1` (integer
/// increment on `n` bits, LSB-first), evaluated at `(a, b) ∈ Fⁿ × Fⁿ`.
///
/// Closed form (`eq(u,v) = 1 + u + v` in characteristic 2): `b = a+1` means some
/// bit `j` flips `0→1`, all lower bits flip `1→0` (the carry chain), and all
/// higher bits are unchanged:
///
/// ```text
///   shift(a,b) = Σ_j [Π_{l<j} a_l(1+b_l)] · [(1+a_j)b_j] · [Π_{l>j} eq(a_l,b_l)]
/// ```
///
/// Evaluated in `O(n)` via prefix/suffix products. `shift(1ⁿ, ·) = 0`.
pub fn shift_mle(a: &[F128], b: &[F128]) -> F128 {
    let n = a.len();
    assert_eq!(b.len(), n, "shift_mle: arity mismatch");

    // pre[j] = Π_{l<j} a_l·(1 + b_l)
    let mut pre = vec![F128::ONE; n + 1];
    for j in 0..n {
        pre[j + 1] = pre[j] * (a[j] * (F128::ONE + b[j]));
    }
    // eqsuf[j] = Π_{l=j}^{n-1} eq(a_l, b_l)
    let mut eqsuf = vec![F128::ONE; n + 1];
    for j in (0..n).rev() {
        let eq_l = F128::ONE + a[j] + b[j];
        eqsuf[j] = eqsuf[j + 1] * eq_l;
    }

    let mut acc = F128::ZERO;
    for j in 0..n {
        let mid = (F128::ONE + a[j]) * b[j]; // bit j flips 0 → 1
        acc += pre[j] * mid * eqsuf[j + 1]; // eqsuf[j+1] = Π_{l>j} eq
    }
    acc
}

/// Proof for the hash-chain shift argument: one combined sumcheck over the
/// `n+1` variables `(y, s₀)`, reducing to a **single** `ẑ` opening.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainShiftProof {
    /// Per-round sumcheck messages `(q(1), q(∞))` (length `n+1`) for the
    /// combined summand `W(y,s₀)·g(y,s₀)`. `q(0)` is recovered from the running
    /// claim via the sum rule. The initial claim `C = eq(τ,1ⁿ)·x_last +
    /// α·x_0(r)` is a *public* scalar the verifier forms itself.
    pub rounds: Vec<(F128, F128)>,
    /// `g(τ', s₀*) = ẑ(τ', (0⁵,s₀*), r)` — the single folded opening value.
    pub g_at_point: F128,
}

/// The single `ẑ`-evaluation claim the shift argument reduces to, for the PCS
/// layer to verify. The full inner point is `(selector = (0,0,0,0,0,sel0),
/// bits = r)`; `r` and the five zero selector bits come from the region-fold
/// layer, while this struct supplies the instance coordinate `τ'`, the merged
/// selector bit `s₀*`, and the value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainClaims {
    /// Instance coordinate `τ'` (length `n`).
    pub instance_point: Vec<F128>,
    /// Merged selector-bit coordinate `s₀*` (picks `state_0` ↔ `state_24`).
    pub sel0: F128,
    /// `g(τ', s₀*) = ẑ(τ', (0⁵,s₀*), r)`.
    pub value: F128,
}

/// Errors the chain-shift verifier can raise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainError {
    /// Round count doesn't match `n+1`.
    MalformedProof,
    /// Final sumcheck claim `≠ W(τ',s₀*)·g(τ',s₀*)` (covers the glue and both
    /// endpoints, since they are batched into the single claim `C`).
    SumcheckFinal,
}

/// Inner product `Σ eq[i]·vals[i]` — used to spot-check claims in tests.
#[cfg(test)]
fn dot(eq: &[F128], vals: &[F128]) -> F128 {
    debug_assert_eq!(eq.len(), vals.len());
    let mut acc = F128::ZERO;
    for i in 0..eq.len() {
        acc += eq[i] * vals[i];
    }
    acc
}

/// Prove the chain-shift relation for instance-indexed MLE value-vectors
/// `in_vals[i] = In(i)` and `out_vals[i] = Out(i)`, each of length `2^n`
/// (already folded over the per-instance bit index by the verifier's `r`).
/// The transcript drives `τ`, `α`, and the sumcheck challenges (Fiat–Shamir).
pub fn prove_chain_shift<Ch: Challenger>(
    in_vals: &[F128],
    out_vals: &[F128],
    challenger: &mut Ch,
) -> (ChainShiftProof, ChainClaims) {
    let n_total = in_vals.len();
    assert!(n_total.is_power_of_two(), "n_total must be a power of two");
    assert_eq!(out_vals.len(), n_total, "In/Out length mismatch");
    let n = n_total.trailing_zeros() as usize;

    // τ ∈ Fⁿ, then α — both before the sumcheck (mirrored by the verifier).
    let tau = challenger.sample_f128_vec(n);
    let alpha = challenger.sample_f128();
    let eqtau = build_eq_table(&tau); // eqtau[y] = eq(τ, y)

    // Weight table over (y, s₀), s₀ the HIGH bit: index y + s₀·N.
    //   s₀ = 0 (In side):  W(y,0) = shift(τ,y) + α·eq(y,0ⁿ)
    //                              = eq(τ,y−1) (y≥1) + α·[y==0]
    //   s₀ = 1 (Out side): W(y,1) = eq(τ,y)
    let mut wt = vec![F128::ZERO; 2 * n_total];
    for y in 1..n_total {
        wt[y] = eqtau[y - 1]; // shift weight
    }
    wt[0] += alpha; // α·eq(0,0ⁿ) = α; eq(y,0ⁿ)=0 for y≠0
    wt[n_total..].copy_from_slice(&eqtau); // s₀ = 1 half

    // g table over (y, s₀): g(y,0)=In(y), g(y,1)=Out(y)  →  [In ‖ Out].
    let mut g = Vec::with_capacity(2 * n_total);
    g.extend_from_slice(in_vals);
    g.extend_from_slice(out_vals);

    // Product-sumcheck on Σ_{y,s₀} W(y,s₀)·g(y,s₀), n+1 variables.
    let d = n + 1;
    let mut rounds = Vec::with_capacity(d);
    let mut r_pts = Vec::with_capacity(d);
    for _ in 0..d {
        let half = g.len() / 2;
        let mut e1 = F128::ZERO; // q(1)  = Σ W_hi·g_hi
        let mut einf = F128::ZERO; // q(∞) = Σ ΔW·Δg
        for i in 0..half {
            let (wlo, whi) = (wt[i], wt[i + half]);
            let (glo, ghi) = (g[i], g[i + half]);
            e1 += whi * ghi;
            einf += (whi + wlo) * (ghi + glo);
        }
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = challenger.sample_f128();
        // Fold (bind the top remaining variable): lo + r·(hi+lo).
        for i in 0..half {
            wt[i] = wt[i] + r * (wt[i + half] + wt[i]);
            g[i] = g[i] + r * (g[i + half] + g[i]);
        }
        wt.truncate(half);
        g.truncate(half);
        rounds.push((e1, einf));
        r_pts.push(r);
    }

    // After n+1 folds, g[0] = g(τ', s₀*) — the single opening value. Build the
    // claim point identically to the verifier: full[d-1-k] = r_pts[k] (bit d-1
    // = s₀, the HIGH bit); τ' = full[..n], s₀* = full[n].
    let mut full = vec![F128::ZERO; d];
    for (k, &r) in r_pts.iter().enumerate() {
        full[d - 1 - k] = r;
    }
    let claims = ChainClaims {
        instance_point: full[..n].to_vec(),
        sel0: full[n],
        value: g[0],
    };
    (
        ChainShiftProof {
            rounds,
            g_at_point: g[0],
        },
        claims,
    )
}

/// Verify the chain-shift proof. `x0_r` and `xlast_r` are the public endpoints
/// `x_0`, `x_{2^n}` folded by the verifier's `r` (the same fold producing
/// `In`/`Out`). Returns the single `ẑ`-evaluation claim for the PCS layer.
pub fn verify_chain_shift<Ch: Challenger>(
    proof: &ChainShiftProof,
    x0_r: F128,
    xlast_r: F128,
    n: usize,
    challenger: &mut Ch,
) -> Result<ChainClaims, ChainError> {
    let d = n + 1;
    if proof.rounds.len() != d {
        return Err(ChainError::MalformedProof);
    }

    // Resample τ, α. The initial claim is the *public* scalar
    //   C = eq(τ,1ⁿ)·x_last + α·x_0(r),     eq(τ,1ⁿ) = Π_j τ_j.
    let tau = challenger.sample_f128_vec(n);
    let alpha = challenger.sample_f128();
    let eq_tau_ones = tau.iter().copied().fold(F128::ONE, |acc, t| acc * t);
    let mut claim = eq_tau_ones * xlast_r + alpha * x0_r;

    // Replay the combined sumcheck (n+1 rounds).
    let mut r_pts = Vec::with_capacity(d);
    for &(e1, einf) in &proof.rounds {
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = challenger.sample_f128();
        // q(0) = claim − q(1) = claim + e1 (char 2). Degree-2 poly through
        // (0,e0),(1,e1),(∞→einf): q(X) = einf·X² + c1·X + e0, c1 = e0+e1+einf.
        let e0 = claim + e1;
        let c1 = e0 + e1 + einf;
        claim = einf * r * r + c1 * r + e0;
        r_pts.push(r);
    }

    // Full point LSB-first (bit d−1 = s₀, the HIGH bit): r_pts[k] bound bit d−1−k.
    let mut full = vec![F128::ZERO; d];
    for (k, &r) in r_pts.iter().enumerate() {
        full[d - 1 - k] = r;
    }
    let taup: Vec<F128> = full[..n].to_vec(); // τ' (instance coords)
    let s0 = full[n]; // s₀*

    // Final weight W(τ', s₀*) (verifier-computed):
    //   shift(τ,τ')·(1+s₀) + eq(τ,τ')·s₀ + α·eq(τ',0ⁿ)·(1+s₀).
    let s = shift_mle(&tau, &taup);
    let eq_tt = eq_eval(&tau, &taup);
    let zero_n = vec![F128::ZERO; n];
    let eq_t0 = eq_eval(&taup, &zero_n); // eq(τ', 0ⁿ) = Π_j (1+τ'_j)
    let one_plus_s0 = F128::ONE + s0;
    let w_final = s * one_plus_s0 + eq_tt * s0 + alpha * eq_t0 * one_plus_s0;

    if claim != w_final * proof.g_at_point {
        return Err(ChainError::SumcheckFinal);
    }

    Ok(ChainClaims {
        instance_point: taup,
        sel0: s0,
        value: proof.g_at_point,
    })
}

// ---------------------------------------------------------------------------
// Region fold (Step 0): collapse a per-instance region of ẑ to one F128 each.
// ---------------------------------------------------------------------------

/// Read logical bit `g` of the packed witness. Convention (see `pcs::pack`):
/// bit `i_skip` of `packed[i_rest]` is global bit `i_rest·128 + i_skip`.
#[inline]
pub fn read_packed_bit(packed: &[F128], g: usize) -> bool {
    let elem = packed[g >> 7];
    let i_skip = g & 127;
    if i_skip < 64 {
        (elem.lo >> i_skip) & 1 == 1
    } else {
        (elem.hi >> (i_skip - 64)) & 1 == 1
    }
}

/// **Naive region fold.** Collapse one per-instance region of the committed
/// witness `ẑ` to a single `F128` per instance.
///
/// `packed` is `ẑ` in PCS-packed form (length `2^(m−7)`). `k_log` is the number
/// of inner (within-block) variables, so instance `i` occupies global bits
/// `[i·2^k_log, (i+1)·2^k_log)` and there are `2^n = 2^(m−k_log)` instances.
/// `taps[t] = (pos, w)` says region bit `t` lives at within-block position `pos`
/// with fold weight `w`.
///
/// Returns `out` of length `2^n` with
/// `out[i] = Σ_t w_t · ẑ_bit(i·2^k_log + pos_t)`.
///
/// This is the correctness oracle; an optimized lane-batched version (mirroring
/// the zerocheck `c`-extraction) will replace it on the hot path.
pub fn fold_region_naive(packed: &[F128], k_log: usize, taps: &[(usize, F128)]) -> Vec<F128> {
    let total_bits = packed.len() << 7;
    let block = 1usize << k_log;
    assert!(
        total_bits.is_multiple_of(block),
        "packed witness not a whole number of blocks"
    );
    let n_inst = total_bits >> k_log;

    (0..n_inst)
        .map(|i| {
            let block_base = i * block;
            let mut acc = F128::ZERO;
            for &(pos, w) in taps {
                if read_packed_bit(packed, block_base + pos) {
                    acc += w;
                }
            }
            acc
        })
        .collect()
}

/// **Optimized region fold** for *byte-contiguous* regions, via method-of-four-
/// Russians (mirrors the zerocheck `c`/`fold_1b` byte-table trick).
///
/// Use when each per-instance region is a run of contiguous physical bits whose
/// fold weight is a fixed `region_weights[p]` (per within-region bit `p`), the
/// **same** for every region and every instance — e.g. the keccak `state_0` /
/// `state_24` slots. `region_byte_offsets[g]` is region `g`'s start, in bytes,
/// within a block; `region_weights.len()` (bits) must be a multiple of 8.
///
/// For each byte-offset `bo` we precompute a 256-entry subset-sum table
/// `tab[bo][v] = Σ_{r: bit r of v set} region_weights[8·bo + r]`. Then each
/// region fold is `Σ_bo tab[bo][byte]` — `region_len/8` table lookups instead of
/// `region_len` branchy bit reads. Returns one length-`2^n` vector per region.
///
/// Parallel over instances. Result is identical to [`fold_region_naive`] with
/// taps `(byte_off·8 + p, region_weights[p])`.
pub fn fold_contiguous_regions(
    packed: &[F128],
    k_log: usize,
    region_byte_offsets: &[usize],
    region_weights: &[F128],
) -> Vec<Vec<F128>> {
    use rayon::prelude::*;

    let region_bits = region_weights.len();
    assert!(
        region_bits.is_multiple_of(8),
        "region weights must be a whole number of bytes"
    );
    let n_bytes = region_bits / 8;

    let block = 1usize << k_log;
    assert!(block.is_multiple_of(8), "block must be byte-aligned");
    let block_bytes = block / 8;
    let total_bits = packed.len() << 7;
    assert!(
        total_bits.is_multiple_of(block),
        "packed witness not a whole number of blocks"
    );
    let n_inst = total_bits >> k_log;

    // Subset-sum byte tables: tab[bo][v] = Σ weights at set bits of v.
    let mut tab = vec![[F128::ZERO; 256]; n_bytes];
    for bo in 0..n_bytes {
        let t = &mut tab[bo];
        for v in 1usize..256 {
            let lsb = v & v.wrapping_neg(); // lowest set bit
            let bit = lsb.trailing_zeros() as usize;
            t[v] = t[v ^ lsb] + region_weights[8 * bo + bit];
        }
    }

    // SAFETY: F128 is repr(C, align(16)) = two LE u64s, so byte B of this view
    // holds logical bits [8B, 8B+8); bit (8B+r) = (byte >> r) & 1.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 16) };

    // Single par_iter over instances producing one length-`n_regions` row per
    // instance — fuses what was previously N sequential `(par_iter).collect()`
    // passes into one rayon dispatch, and lets the per-byte `tab[bo][..]` reads
    // stay hot in L1 across all regions of the same instance.
    let n_regions = region_byte_offsets.len();
    let flat: Vec<F128> = (0..n_inst)
        .into_par_iter()
        .flat_map_iter(|i| {
            let instance_base = i * block_bytes;
            let mut row = [F128::ZERO; 8]; // supports up to 8 regions; matches realistic chain layouts
            assert!(
                n_regions <= row.len(),
                "fold_contiguous_regions: too many regions"
            );
            for bo in 0..n_bytes {
                let t_bo = &tab[bo];
                for (r_idx, &off) in region_byte_offsets.iter().enumerate() {
                    row[r_idx] += t_bo[bytes[instance_base + off + bo] as usize];
                }
            }
            (0..n_regions).map(move |r_idx| row[r_idx])
        })
        .collect();

    // De-interleave: flat is [(inst 0 r0), (inst 0 r1), ..., (inst 1 r0), ...].
    // Transpose to one Vec per region.
    (0..n_regions)
        .map(|r_idx| (0..n_inst).map(|i| flat[i * n_regions + r_idx]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::challenger::RandomChallenger;

    /// SplitMix64-ish RNG for test data.
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
    }

    /// Build an LSB-first boolean point as F128 (0/1) of length `n` from index.
    fn bool_point(idx: usize, n: usize) -> Vec<F128> {
        (0..n)
            .map(|j| {
                if (idx >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                }
            })
            .collect()
    }

    /// Pack a bool witness the way `pcs::pack` does (bit i_skip of out[i_rest] =
    /// z[i_rest·128 + i_skip]).
    fn pack(z: &[bool]) -> Vec<F128> {
        assert!(z.len().is_multiple_of(128));
        (0..z.len() / 128)
            .map(|i_rest| {
                let base = i_rest * 128;
                let mut lo = 0u64;
                let mut hi = 0u64;
                for r in 0..64 {
                    if z[base + r] {
                        lo |= 1 << r;
                    }
                    if z[base + 64 + r] {
                        hi |= 1 << r;
                    }
                }
                F128 { lo, hi }
            })
            .collect()
    }

    /// `fold_region_naive` reads the right bits and weights them: compare its
    /// output against a direct fold over the bool witness.
    #[test]
    fn fold_region_naive_matches_direct() {
        let mut rng = Rng::new(0xF01D);
        // k_log = 5 (block = 32 bits), n = 3 (8 instances) → m = 8, 256 bits.
        let k_log = 5usize;
        let n = 3usize;
        let block = 1usize << k_log;
        let total = (1usize << n) * block;
        let z: Vec<bool> = (0..total).map(|_| rng.next_u64() & 1 == 1).collect();
        let packed = pack(&z);

        // Random taps: 10 region bits at distinct in-block positions, random w.
        let taps: Vec<(usize, F128)> = (0..10).map(|t| (3 * t % block, rng.f128())).collect();

        let got = fold_region_naive(&packed, k_log, &taps);
        assert_eq!(got.len(), 1 << n);
        for i in 0..(1 << n) {
            let mut want = F128::ZERO;
            for &(pos, w) in &taps {
                if z[i * block + pos] {
                    want += w;
                }
            }
            assert_eq!(got[i], want, "instance {i}");
        }
    }

    /// `fold_contiguous_regions` (fused multi-region pass) matches calling
    /// `fold_region_naive` once per region. Exercises 1, 2, and 3 regions.
    #[test]
    fn fold_contiguous_regions_matches_per_region() {
        let mut rng = Rng::new(0xC0DE_F00D);
        let k_log = 6usize; // block = 64 bits = 8 bytes
        let n = 4usize; // 16 instances
        let block = 1usize << k_log;
        let total = (1usize << n) * block;
        let z: Vec<bool> = (0..total).map(|_| rng.next_u64() & 1 == 1).collect();
        let packed = pack(&z);

        // Region: 16 contiguous bits = 2 bytes, with random weights.
        let region_bits = 16;
        let weights: Vec<F128> = (0..region_bits).map(|_| rng.f128()).collect();

        // Test with N regions at distinct byte-aligned offsets.
        for &offs in &[&[0usize] as &[usize], &[0, 2], &[0, 2, 4]] {
            let got = fold_contiguous_regions(&packed, k_log, offs, &weights);
            assert_eq!(got.len(), offs.len());
            for (r_idx, &off) in offs.iter().enumerate() {
                let taps: Vec<(usize, F128)> = (0..region_bits)
                    .map(|p| (off * 8 + p, weights[p]))
                    .collect();
                let want = fold_region_naive(&packed, k_log, &taps);
                assert_eq!(got[r_idx], want, "region {r_idx} (offset {off})");
            }
        }
    }

    /// `shift_mle` on boolean inputs is exactly the successor indicator.
    #[test]
    fn shift_mle_boolean_is_successor() {
        for n in 1..=6 {
            let n_total = 1usize << n;
            for a in 0..n_total {
                for b in 0..n_total {
                    let av = bool_point(a, n);
                    let bv = bool_point(b, n);
                    let got = shift_mle(&av, &bv);
                    let want = if b == a + 1 { F128::ONE } else { F128::ZERO };
                    assert_eq!(got, want, "shift({a},{b}) n={n}");
                }
            }
        }
    }

    /// `shift(1ⁿ, ·) = 0` (no successor in range).
    #[test]
    fn shift_mle_top_has_no_successor() {
        let mut rng = Rng::new(7);
        for n in 1..=5 {
            let a = bool_point((1 << n) - 1, n);
            let b = rng.f128_vec(n);
            assert_eq!(shift_mle(&a, &b), F128::ZERO);
        }
    }

    /// `shift(τ, y) = eq(τ, y−1)` for boolean `y ≥ 1` and field `τ`.
    #[test]
    fn shift_equals_shifted_eq() {
        let mut rng = Rng::new(11);
        for n in 1..=5 {
            let n_total = 1usize << n;
            let tau = rng.f128_vec(n);
            let eqtau = build_eq_table(&tau);
            for y in 0..n_total {
                let yv = bool_point(y, n);
                let got = shift_mle(&tau, &yv);
                let want = if y == 0 { F128::ZERO } else { eqtau[y - 1] };
                assert_eq!(got, want, "y={y} n={n}");
            }
        }
    }

    /// Honest chained data: `In[i]=x_i`, `Out[i]=x_{i+1}`. Prove + verify must
    /// accept, and the single returned claim must be the true merged MLE
    /// `g(τ',s₀*) = (1+s₀*)·In(τ') + s₀*·Out(τ')` (what the PCS would enforce).
    #[test]
    fn honest_roundtrip_accepts() {
        for n in 3..=8 {
            let n_total = 1usize << n;
            let mut rng = Rng::new(100 + n as u64);
            // x_0 .. x_N  (N+1 chain values); In[i]=x_i, Out[i]=x_{i+1}.
            let chain: Vec<F128> = rng.f128_vec(n_total + 1);
            let in_vals: Vec<F128> = chain[..n_total].to_vec();
            let out_vals: Vec<F128> = chain[1..].to_vec();
            let x0_r = chain[0];
            let xlast_r = chain[n_total];

            let mut chp = RandomChallenger::new(42);
            let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);

            let mut chv = RandomChallenger::new(42);
            let claims = verify_chain_shift(&proof, x0_r, xlast_r, n, &mut chv)
                .expect("honest proof should verify");

            let eq_taup = build_eq_table(&claims.instance_point);
            let in_true = dot(&eq_taup, &in_vals);
            let out_true = dot(&eq_taup, &out_vals);
            let g_true = (F128::ONE + claims.sel0) * in_true + claims.sel0 * out_true;
            assert_eq!(claims.value, g_true, "merged claim n={n}");
        }
    }

    /// Breaking the chain at one index makes the sumcheck reject.
    #[test]
    fn broken_chain_rejects() {
        let n = 6;
        let n_total = 1usize << n;
        let mut rng = Rng::new(2024);
        let chain: Vec<F128> = rng.f128_vec(n_total + 1);
        let in_vals: Vec<F128> = chain[..n_total].to_vec();
        let mut out_vals: Vec<F128> = chain[1..].to_vec();
        let x0_r = chain[0];
        let xlast_r = chain[n_total];

        // Break the glue: Out[3] no longer equals In[4].
        out_vals[3] += F128::ONE;

        let mut chp = RandomChallenger::new(9);
        let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        let mut chv = RandomChallenger::new(9);
        let res = verify_chain_shift(&proof, x0_r, xlast_r, n, &mut chv);
        assert_eq!(res, Err(ChainError::SumcheckFinal));
    }

    /// A wrong public input endpoint is caught. It is batched (via α) into the
    /// single claim, so the failure surfaces as a final-sumcheck mismatch.
    #[test]
    fn wrong_input_endpoint_rejects() {
        let n = 5;
        let n_total = 1usize << n;
        let mut rng = Rng::new(555);
        let chain: Vec<F128> = rng.f128_vec(n_total + 1);
        let in_vals: Vec<F128> = chain[..n_total].to_vec();
        let out_vals: Vec<F128> = chain[1..].to_vec();
        let xlast_r = chain[n_total];
        let wrong_x0 = chain[0] + F128::ONE;

        let mut chp = RandomChallenger::new(3);
        let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        let mut chv = RandomChallenger::new(3);
        let res = verify_chain_shift(&proof, wrong_x0, xlast_r, n, &mut chv);
        assert_eq!(res, Err(ChainError::SumcheckFinal));
    }

    /// A wrong public output endpoint is caught.
    #[test]
    fn wrong_output_endpoint_rejects() {
        let n = 5;
        let n_total = 1usize << n;
        let mut rng = Rng::new(777);
        let chain: Vec<F128> = rng.f128_vec(n_total + 1);
        let in_vals: Vec<F128> = chain[..n_total].to_vec();
        let out_vals: Vec<F128> = chain[1..].to_vec();
        let x0_r = chain[0];
        let wrong_xlast = chain[n_total] + F128::ONE;

        let mut chp = RandomChallenger::new(1);
        let (proof, _claims) = prove_chain_shift(&in_vals, &out_vals, &mut chp);
        let mut chv = RandomChallenger::new(1);
        let res = verify_chain_shift(&proof, x0_r, wrong_xlast, n, &mut chv);
        assert_eq!(res, Err(ChainError::SumcheckFinal));
    }
}
