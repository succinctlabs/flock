// Copyright 2024-2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The algorithm skeleton (iterative LCH NTT, neighbors-last ordering) is
// derived from binius64's `NeighborsLastReference`
// (https://github.com/binius-zk/binius64, `crates/math/src/ntt/reference.rs`).
// The interleaved SoA layout, fused 2-layer butterfly, and parallelization
// strategy are original to Flock.

//! Additive NTT over F_{2^128} using the LCH novel polynomial basis.
//!
//! Iterative LCH NTT skeleton derived from binius64's `NeighborsLastReference`,
//! with an interleaved SoA layout, a fused 2-layer butterfly, and rayon-based
//! parallelization added on top. The forward transform maps polynomial
//! coefficients (in the novel polynomial basis) to evaluations over an
//! F_2-affine subspace; the inverse reverses this. Used by the PCS commit and
//! by FRI folding.
//!
//! ## Convention
//!
//! Given a basis `{β_0, …, β_{ℓ-1}}` of an F_2-subspace V ⊂ F_{2^128}, define
//! the subspace polynomials W_i recursively:
//! ```text
//!     W_0(z) = z
//!     W_i(z) = W_{i-1}(z) · (W_{i-1}(z) + W_{i-1}(β_{i-1}))     (for i ≥ 1)
//! ```
//! and the *normalized* forms `Ŵ_i(z) = W_i(z) / W_i(β_i)` so that
//! `Ŵ_i(β_i) = 1`. The "twiddle" at layer `l` and block `b` is then
//! `Ŵ_{ℓ-l-1}(z)` evaluated at the `b`-th element of the F_2-span of
//! `{β_{ℓ-l}, β_{ℓ-l+1}, …, β_{ℓ-1}}`.
//!
//! At forward-transform layer `l` (`l = 0, …, log_d − 1`):
//! - There are `2^l` blocks, each of size `2^(log_d − l)`.
//! - Within each block, pairs `(idx0, idx0 | block_size_half)` are
//!   butterflied with the block's twiddle.
//! - **Pairing at layer `l`**: positions differ by `block_size_half =
//!   2^(log_d − l − 1)`. So at layer 0 pairs are far (N/2 apart), and at the
//!   deepest layer pairs are adjacent (1 apart) — this is "neighbors-last."
//!
//! FRI fold processes layers in **reverse** (deepest first), at which level
//! pairs are adjacent — matching the standard `fold_pair` formula in DP24.

use crate::all_core_pool;
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use core::arch::aarch64::*;
use rayon::current_num_threads;
use rayon::prelude::*;
use std::env::var;
use std::env::var_os;
use std::mem::size_of_val;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::field::F128;

use self::kernels::{
    butterfly_fused_2layer, butterfly_fused_3layer, butterfly_fused_4layer_row, butterfly_row_pair,
};
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use self::kernels::{
    butterfly_neon_block, butterfly_neon_block_pair, butterfly_neon_block_pair_chunk,
};
mod kernels;

/// A/B toggle: when set, the deep (cache-resident) pass of the interleaved
/// parallel NTT stays on the caller's (P-core) pool instead of hopping to
/// [`crate::all_core_pool`] for large transforms. `NTT_DEEP_PCORES_ONLY=1`
/// in the environment forces the same fallback (production kill-switch); the
/// AtomicBool exists for paired within-process A/B.
pub static NTT_DEEP_PCORES_ONLY: AtomicBool = AtomicBool::new(false);

/// A/B toggle: when set, the deep pass runs every layer as its own sweep
/// instead of fusing general-width layer pairs. `NTT_DEEP_NOFUSE=1` env is
/// the production kill-switch; the AtomicBool is for paired within-process
/// A/B.
pub static NTT_DEEP_NOFUSE: AtomicBool = AtomicBool::new(false);

/// Compute the normalized subspace-polynomial evaluation table.
///
/// Returns `evals` where `evals[i] = [Ŵ_i(β_i), Ŵ_i(β_{i+1}), …, Ŵ_i(β_{ℓ-1})]`.
/// The 0-th element of each row is always `1` (by normalization).
fn generate_evals_from_subspace(basis: &[F128]) -> Vec<Vec<F128>> {
    let l = basis.len();
    let mut evals: Vec<Vec<F128>> = Vec::with_capacity(l);

    // evals[0] = [W_0(β_0), W_0(β_1), …, W_0(β_{ℓ-1})] = basis.
    evals.push(basis.to_vec());

    // evals[i][k] = W_i(β_{i+k}) computed from evals[i-1].
    // evals[i-1] = [W_{i-1}(β_{i-1}), W_{i-1}(β_i), W_{i-1}(β_{i+1}), …]
    // We want W_i(β_{i+k}) = W_{i-1}(β_{i+k}) · (W_{i-1}(β_{i+k}) + W_{i-1}(β_{i-1}))
    //                     = evals[i-1][k+1] · (evals[i-1][k+1] + evals[i-1][0])
    for i in 1..l {
        let mut row = Vec::with_capacity(l - i);
        for k in 1..evals[i - 1].len() {
            let val = evals[i - 1][k] * (evals[i - 1][k] + evals[i - 1][0]);
            row.push(val);
        }
        evals.push(row);
    }

    // Normalize each row by its 0-th element (= W_i(β_i)).
    for row in evals.iter_mut() {
        let inv = row[0].inv();
        for v in row.iter_mut() {
            *v *= inv;
        }
    }

    evals
}

/// Compute `Σ_j bit_j(idx) · basis[j]` — the `idx`-th element of the F_2-span
/// of `basis`.
#[inline]
fn span_get(basis: &[F128], idx: usize) -> F128 {
    let mut acc = F128::ZERO;
    for (j, &b) in basis.iter().enumerate() {
        if (idx >> j) & 1 == 1 {
            acc += b;
        }
    }
    acc
}

/// Additive NTT over F_{2^128} with the standard polynomial-basis subspace.
///
/// The basis is `{1, x, x², …, x^(ℓ-1)}` in F_{2^128} = F_2[x]/(GHASH-poly).
/// This makes the F_2-subspace V = `{0, 1, …, 2^ℓ-1}` (under the natural
/// integer encoding of F_{2^128} elements).
#[derive(Clone, Debug)]
pub struct AdditiveNttF128 {
    /// `evals[i]` of length `ℓ − i`, the normalized subspace polynomial values.
    evals: Vec<Vec<F128>>,
}

impl AdditiveNttF128 {
    /// Construct an NTT from an explicit F_2-basis.
    pub fn new(basis: &[F128]) -> Self {
        Self {
            evals: generate_evals_from_subspace(basis),
        }
    }

    /// Standard NTT with basis `{1, x, x², …, x^(dim-1)}`. Requires `dim ≤ 64`
    /// (the low 64 bits of F_{2^128} hold these basis vectors).
    pub fn standard(dim: usize) -> Self {
        assert!(dim <= 64, "standard NTT requires dim ≤ 64");
        let basis: Vec<F128> = (0..dim).map(|i| F128::new(1u64 << i, 0)).collect();
        Self::new(&basis)
    }

    pub fn log_domain_size(&self) -> usize {
        self.evals.len()
    }

    /// Twiddle at `(layer, block)` for the forward NTT and FRI fold.
    ///
    /// At layer `l` ∈ `[0, ℓ)`, block index `b` ∈ `[0, 2^l)`:
    /// `twiddle(l, b) = Σ_j bit_j(b) · Ŵ_{ℓ-l-1}(β_{ℓ-l+j})`
    ///
    /// (The 0-th element of the row corresponds to `Ŵ_{ℓ-l-1}(β_{ℓ-l-1}) = 1`,
    /// which is "absorbed" into the butterfly and not in the twiddle.)
    pub fn twiddle(&self, layer: usize, block: usize) -> F128 {
        let v = &self.evals[self.log_domain_size() - layer - 1];
        span_get(&v[1..], block)
    }

    /// Forward additive NTT in place. `data.len()` must be `2^log_d` for some
    /// `log_d ≤ log_domain_size()`. Layer `l ∈ [0, log_d)` is processed in
    /// order (neighbors-last: top layer first).
    ///
    /// Dispatches to the cache-blocked batched implementation when available
    /// and the buffer is large enough to benefit; otherwise falls back to the
    /// per-layer parallel path or scalar.
    pub fn forward_transform(&self, data: &mut [F128]) {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            self.forward_transform_batched(data);
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.forward_transform_scalar(data);
        }
    }

    /// Interleaved forward NTT: process `num_ntts` independent NTTs in
    /// position-major SoA layout.
    ///
    /// `data` layout: `data[pos * num_ntts + lane]` for `pos ∈ 0..2^log_d`,
    /// `lane ∈ 0..num_ntts`. Each "lane" is an independent NTT instance over
    /// the same domain; all `num_ntts` instances share the twiddle structure
    /// (same `self.twiddle(layer, block)` is applied to every lane at the
    /// corresponding butterfly).
    ///
    /// `num_ntts` must be a positive integer (need NOT be a power of two — the
    /// integer-lane commit path passes an arbitrary `t`). `data.len()` must
    /// equal `(1 << log_d) * num_ntts` for some `log_d ≤ log_domain_size()`.
    ///
    /// This produces the SAME RS code per lane as `forward_transform`, with
    /// FRI-compatible twiddles. The SoA layout is what makes each Merkle leaf
    /// = one position across all `num_ntts` lanes (= contiguous slice of
    /// `num_ntts` F_{2^128} elements).
    pub fn forward_transform_interleaved(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_from_layer(data, num_ntts, 0);
    }

    /// Forward interleaved NTT starting at `start_layer`, assuming the first
    /// `start_layer` layers have already been applied to `data`.
    ///
    /// The RS-encoding use case: with `log_inv_rate = r` the upper
    /// `(2^r − 1)/2^r` of the coefficient buffer is zero, so each of the first
    /// `r` layers degenerates to a copy (butterfly with `v = 0` gives
    /// `(u, u)`). The caller replicates the message into all `2^r` sub-blocks
    /// — which IS the exact post-layer-`r` state — and skips those layers'
    /// reads and multiplies here.
    pub fn forward_transform_interleaved_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        self.forward_transform_interleaved_live_from_layer(data, num_ntts, num_ntts, start_layer);
    }

    /// [`Self::forward_transform_interleaved_from_layer`] transforming only
    /// the FIRST `live` lanes of every position; lanes `live..num_ntts` must
    /// be IDENTICALLY ZERO on entry. Zero is a fixed point of the butterfly
    /// (`u = 0, v = 0 → (0, 0)`), so the bounded transform leaves the buffer
    /// byte-identical to what the full transform would produce — it skips
    /// the arithmetic and memory traffic of lanes whose codeword is already
    /// in place. This is the integer-lane commit's DEAD-LANE SKIP: a pinned
    /// lane count above the content's (the envelope's `lanes*`) commits
    /// trailing all-zero lanes, and their encode is free.
    pub fn forward_transform_interleaved_live_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        live: usize,
        start_layer: usize,
    ) {
        // `num_ntts` may be any positive integer (integer-lane commit): the
        // per-lane butterfly kernels iterate over the lane stride, and the
        // SIMD kernels handle the `num_ntts % SIMD_WIDTH` tail lanes with the
        // portable path. Only `n_total / num_ntts` (the per-lane length) must
        // be a power of two.
        assert!(num_ntts > 0);
        assert!(live <= num_ntts, "live lanes are a prefix of the stride");
        let n_total = data.len();
        assert_eq!(n_total % num_ntts, 0);
        let log_d = log2_pow2(n_total / num_ntts);
        assert!(log_d <= self.log_domain_size());
        assert!(start_layer <= log_d);
        debug_assert!(
            live == num_ntts
                || data
                    .chunks_exact(num_ntts)
                    .all(|row| row[live..].iter().all(|w| w.is_zero())),
            "dead lanes must be identically zero"
        );
        if live == 0 {
            return; // an all-zero message's codeword is already in place
        }

        // Scalar; SIMD/parallel variants below dispatch from `forward_transform_interleaved`
        // on supported targets.
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        ))]
        {
            self.interleaved_parallel_live_from_layer(data, num_ntts, live, start_layer);
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        {
            self.interleaved_scalar_live_from_layer(data, num_ntts, live, start_layer);
        }
    }

    /// Scalar reference for the interleaved forward NTT.
    pub fn forward_transform_interleaved_scalar(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, 0);
    }

    /// Scalar interleaved forward NTT from `start_layer` (see
    /// [`Self::forward_transform_interleaved_from_layer`]).
    pub fn forward_transform_interleaved_scalar_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        self.interleaved_scalar_live_from_layer(data, num_ntts, num_ntts, start_layer);
    }

    /// Scalar interleaved forward NTT over the first `live` lanes of the
    /// `num_ntts`-lane stride (dead lanes stay untouched — they must be zero,
    /// see [`Self::forward_transform_interleaved_live_from_layer`]).
    fn interleaved_scalar_live_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        live: usize,
        start_layer: usize,
    ) {
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        for layer in start_layer..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;
            let block_size_bytes = block_size * num_ntts;
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block * block_size_bytes;
                // Butterfly pairs (top, bot) at positions (row, row + block_size_half)
                // within the block. Each "position" holds num_ntts lanes side-by-side.
                for row in 0..block_size_half {
                    let off_top = block_start + row * num_ntts;
                    let off_bot = off_top + block_size_half * num_ntts;
                    for lane in 0..live {
                        let v = data[off_bot + lane];
                        let new_u = data[off_top + lane] + v * twiddle;
                        data[off_top + lane] = new_u;
                        data[off_bot + lane] = v + new_u;
                    }
                }
            }
        }
    }

    /// Parallel + NEON interleaved forward NTT. Cache-blocks the same way as
    /// `forward_transform_batched`: top layers process the full SoA buffer with
    /// per-block parallelism; deep layers process each sub-NTT-group in cache.
    ///
    /// Internally calls [`forward_transform_interleaved_scalar`] for very small
    /// inputs to avoid rayon overhead; for large inputs it uses an in-place
    /// scalar butterfly per lane (per-lane vectorization is future work — the
    /// big win at large `m` is cache locality + thread parallelism).
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    pub fn forward_transform_interleaved_parallel(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_parallel_from_layer(data, num_ntts, 0);
    }

    /// Parallel interleaved forward NTT from `start_layer` (see
    /// [`Self::forward_transform_interleaved_from_layer`]).
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    pub fn forward_transform_interleaved_parallel_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        self.interleaved_parallel_live_from_layer(data, num_ntts, num_ntts, start_layer);
    }

    /// Parallel interleaved forward NTT over the first `live` lanes (dead
    /// lanes stay untouched — they must be zero, see
    /// [`Self::forward_transform_interleaved_live_from_layer`]).
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    fn interleaved_parallel_live_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        live: usize,
        start_layer: usize,
    ) {
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        // Target sub-group size = 2 MB total bytes. Each position is
        // `num_ntts × 16` bytes, so positions per sub-group =
        // 2^21 / (num_ntts · 16). With num_ntts=1: 2^17 positions. With
        // num_ntts=32: 2^12 positions. (Without this scaling, sub-groups at
        // num_ntts=32 would be 64 MB and overflow L2 cache.)
        const TARGET_SUBGROUP_LOG_BYTES: usize = 21;
        // `num_ntts` need not be a power of two (integer-lane commit). Round the
        // lane count UP to a power of two for the cache-blocking heuristic so an
        // integer `t` blocks exactly like the padded `2^ceil(log2 t)` (measured:
        // this recovers the full per-lane efficiency — a floor-log2 here left
        // t=46 with oversized 3 MB sub-groups and ~15% slower than ideal). Only
        // affects the sub-group SIZE (a tuning knob), never correctness.
        let log_bytes_per_position = 4 + ceil_log2(num_ntts);
        let target_log_positions = TARGET_SUBGROUP_LOG_BYTES.saturating_sub(log_bytes_per_position);
        let cache_n_top = log_d.saturating_sub(target_log_positions);

        // Parallelism floor. The cache heuristic keeps each sub-NTT ~2 MB, but
        // for a mid-size transform whose whole codeword already fits that
        // budget it yields `cache_n_top == 0` and the transform runs fully
        // serial — e.g. the recursive Ligerito commits (~1 ms of NTT each,
        // previously 1.0× across threads). When the transform is big enough to
        // amortize rayon overhead, raise `n_top` so the deep-layer split
        // produces ~one sub-NTT per worker thread (capped to keep each sub-NTT
        // ≥ 2^MIN_SUB_LOG positions). The large initial PCS commit is unaffected:
        // its `cache_n_top` already exceeds this floor.
        //
        // The floor (log_d ≥ 12) is the measured dispatch-vs-compute crossover
        // for num_ntts≈8 recursive commits: at log_d=12 parallelizing cuts the
        // NTT ~0.22 → ~0.08 ms, but at log_d=10 the rayon dispatch costs more
        // than the ~0.04 ms of work, so those stay scalar.
        const PARALLEL_FLOOR_LOG_D: usize = 12;
        const MIN_SUB_LOG: usize = 8;
        let n_top = if log_d >= PARALLEL_FLOOR_LOG_D {
            let want_subs_log = log2_pow2(current_num_threads().next_power_of_two());
            let max_n_top = log_d.saturating_sub(MIN_SUB_LOG);
            cache_n_top.max(want_subs_log.min(max_n_top))
        } else {
            cache_n_top
        };
        if n_top == 0 || log_d < 8 {
            self.interleaved_scalar_live_from_layer(data, num_ntts, live, start_layer);
            return;
        }

        // Top layers: full-buffer sweep. Parallelize **rows within each
        // block** so even layer 0 (1 huge block) gets rayon parallelism.
        //
        // Layer fusion: at top layers each layer is a separate full-buffer
        // sweep (read 512 MB + write 512 MB at m=31). Fusing two consecutive
        // layers in one pass loads each row once, applies both butterflies
        // in registers, stores once — halving memory traffic on the fused
        // layers. Each "outer block" at layer L has 4 contributing rows per
        // quarter-row; layer L butterflies (a,c) and (b,d) (distance =
        // block_size/2), layer L+1 butterflies (a,b) and (c,d) (distance =
        // block_size/4).
        // Fuse FOUR layers per pass only where a SIMD fused-4 kernel exists
        // (x86 AVX-512). On other targets the 16-point kernel falls back to
        // scalar, which is slower than the NEON fused-2 path — so keep fused-2
        // there. NEON fused-4 is a future addition.
        // The fused-4 kernel walks rows through raw pointers with its own
        // offset math; the dead-lane skip keeps to the slice-based kernels,
        // so a bounded transform takes the fused-3/2/block route instead.
        let fused4_ok = live == num_ntts
            && cfg!(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ));
        // Fused-3 (8-point): the aarch64 middle tier — a third fewer full-
        // buffer passes than fused-2 on the layers it covers, with 8 values
        // + 7 twiddles in flight (the 16-point kernel's register pressure is
        // what lost on this target). `FLOCK_NTT_NO_FUSED3=1` disables — the
        // A/B knob.
        let fused3_ok = var_os("FLOCK_NTT_NO_FUSED3").is_none();
        let mut layer = start_layer.min(n_top);
        while layer < n_top {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_bytes = block_size * num_ntts;

            if fused4_ok && layer + 3 < n_top && block_size >= 16 {
                // Fuse four layers (layer..layer+4): one read+write per block
                // instead of four. Each block contributes a 16-point butterfly.
                let sixteenth = block_size >> 4;
                for block in 0..num_blocks {
                    let mut tw = [F128 { lo: 0, hi: 0 }; 15];
                    tw[0] = self.twiddle(layer, block);
                    for s in 0..2 {
                        tw[1 + s] = self.twiddle(layer + 1, 2 * block + s);
                    }
                    for s in 0..4 {
                        tw[3 + s] = self.twiddle(layer + 2, 4 * block + s);
                    }
                    for s in 0..8 {
                        tw[7 + s] = self.twiddle(layer + 3, 8 * block + s);
                    }
                    let start = block * block_bytes;
                    butterfly_interleaved_fused_4layer_par_rows(
                        &mut data[start..start + block_bytes],
                        &tw,
                        sixteenth,
                        num_ntts,
                    );
                }
                layer += 4;
            } else if fused3_ok && layer + 2 < n_top && block_size >= 8 {
                // Fuse three layers (layer..layer+3): 8-point butterflies,
                // one read+write pass where fused-2 alternation needs 1.5.
                let eighth = block_size >> 3;
                for block in 0..num_blocks {
                    let t0 = self.twiddle(layer, block);
                    let t1 = [
                        self.twiddle(layer + 1, 2 * block),
                        self.twiddle(layer + 1, 2 * block + 1),
                    ];
                    let t2 = [
                        self.twiddle(layer + 2, 4 * block),
                        self.twiddle(layer + 2, 4 * block + 1),
                        self.twiddle(layer + 2, 4 * block + 2),
                        self.twiddle(layer + 2, 4 * block + 3),
                    ];
                    let start = block * block_bytes;
                    butterfly_interleaved_fused_3layer_par_rows(
                        &mut data[start..start + block_bytes],
                        t0,
                        t1,
                        t2,
                        eighth,
                        num_ntts,
                        live,
                    );
                }
                layer += 3;
            } else if layer + 1 < n_top && block_size >= 4 {
                // Fuse layers (layer, layer+1).
                let quarter = block_size >> 2;
                for block in 0..num_blocks {
                    let t_outer = self.twiddle(layer, block);
                    let t_inner_a = self.twiddle(layer + 1, 2 * block);
                    let t_inner_b = self.twiddle(layer + 1, 2 * block + 1);
                    let start = block * block_bytes;
                    butterfly_interleaved_fused_2layer_par_rows(
                        &mut data[start..start + block_bytes],
                        t_outer,
                        t_inner_a,
                        t_inner_b,
                        quarter,
                        num_ntts,
                        live,
                    );
                }
                layer += 2;
            } else {
                let block_size_half = block_size >> 1;
                for block in 0..num_blocks {
                    let t = self.twiddle(layer, block);
                    let start = block * block_bytes;
                    butterfly_interleaved_block_par_rows(
                        &mut data[start..start + block_bytes],
                        t,
                        block_size_half,
                        num_ntts,
                        live,
                    );
                }
                layer += 1;
            }
        }

        // Deep layers: process each sub-NTT-group cache-resident.
        let sub_size_positions = 1usize << (log_d - n_top);
        let sub_bytes = sub_size_positions * num_ntts;

        // Within each sub-group, fuse consecutive GENERAL-width layer pairs
        // into one sweep (halves the sub's L1/L2 traffic and the butterfly
        // load/store count for those layers — the deep pass is compute-bound,
        // not DRAM-bound, so in-cache traffic and instruction count are what
        // matter). The three deepest layers stay single-layer: their twiddles
        // are all half-width (see `mul_small_twiddle`) and the fast path in
        // `butterfly_interleaved_block` beats fusion's general muls.
        // `NTT_DEEP_NOFUSE` restores per-layer sweeps (A/B).
        let fuse = !NTT_DEEP_NOFUSE.load(Ordering::Relaxed) && var("NTT_DEEP_NOFUSE").is_err();
        let halfwidth_start = log_d.saturating_sub(3);
        let deep = |data: &mut [F128]| {
            data.par_chunks_mut(sub_bytes)
                .enumerate()
                .for_each(|(sub_idx, sub_data)| {
                    let mut layer = n_top.max(start_layer);
                    while layer < log_d {
                        let layer_in_sub = layer - n_top;
                        let num_blocks_in_sub = 1usize << layer_in_sub;
                        let block_size = 1usize << (log_d - layer);
                        let block_bytes = block_size * num_ntts;

                        if fuse && layer + 2 <= halfwidth_start && block_size >= 4 {
                            let quarter = block_size >> 2;
                            for block_in_sub in 0..num_blocks_in_sub {
                                let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                                let t_outer = self.twiddle(layer, global_block);
                                let t_inner_a = self.twiddle(layer + 1, 2 * global_block);
                                let t_inner_b = self.twiddle(layer + 1, 2 * global_block + 1);
                                let block_start = block_in_sub * block_bytes;
                                let block = &mut sub_data[block_start..block_start + block_bytes];
                                butterfly_interleaved_fused_2layer_serial(
                                    block, t_outer, t_inner_a, t_inner_b, quarter, num_ntts, live,
                                );
                            }
                            layer += 2;
                        } else {
                            let block_size_half = block_size >> 1;
                            for block_in_sub in 0..num_blocks_in_sub {
                                let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                                let twiddle = self.twiddle(layer, global_block);
                                let block_start = block_in_sub * block_bytes;
                                let block = &mut sub_data[block_start..block_start + block_bytes];
                                butterfly_interleaved_block(
                                    block,
                                    twiddle,
                                    block_size_half,
                                    num_ntts,
                                    live,
                                );
                            }
                            layer += 1;
                        }
                    }
                });
        };
        // The deep pass is a flat parallel-for over independent ~2 MB
        // sub-groups with a single join — measured ~80% PMULL-compute-bound at
        // m=30 (one streaming pass of traffic, 11 in-cache layers). Unlike the
        // barrier-per-block top passes, it drains cleanly around slow cores,
        // and the E-cores have PMULL too — so large transforms hop to the
        // all-core (P+E) pool. Pool choice cannot change output bits (each
        // sub-group is written deterministically). Gate: enough sub-groups to
        // drain (≥ 4× workers) and ≥ 64 MB of data so the pool switch and
        // E-core L2 pressure can't hurt small/recursive commits.
        // `NTT_DEEP_PCORES_ONLY` (atomic or env) restores the caller's pool.
        let n_subs = data.len() / sub_bytes;
        let use_all_cores = size_of_val(data) >= (64 << 20)
            && !NTT_DEEP_PCORES_ONLY.load(Ordering::Relaxed)
            && var("NTT_DEEP_PCORES_ONLY").is_err()
            && {
                let pool = all_core_pool();
                pool.current_num_threads() > current_num_threads()
                    && n_subs >= 4 * pool.current_num_threads()
            };
        if use_all_cores {
            all_core_pool().install(|| deep(data));
        } else {
            deep(data);
        }
    }

    /// Scalar reference implementation. Used as the test oracle and on
    /// platforms without NEON+PMULL.
    pub fn forward_transform_scalar(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;
                    // Forward butterfly: u += v·twiddle; v += u.
                    let v = data[idx1];
                    let new_u = data[idx0] + v * twiddle;
                    data[idx0] = new_u;
                    data[idx1] = v + new_u;
                }
            }
        }
    }

    /// Inverse of [`Self::forward_transform_scalar`]: evaluations back to
    /// LCH-basis coefficients. Layers run in reverse with the same twiddles;
    /// each butterfly inverts as `v = v' + u'; u = u' + v·twiddle`. In char 2
    /// the additive butterfly is involutive up to this reordering, so no
    /// scaling pass is needed.
    pub fn inverse_transform_scalar(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in (0..log_d).rev() {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;
                    let u = data[idx0];
                    let v = data[idx1] + u;
                    data[idx0] = u + v * twiddle;
                    data[idx1] = v;
                }
            }
        }
    }

    /// Single-threaded NEON forward transform (uses `ghash_mul_vec2_neon` to
    /// batch 2 butterflies per PMULL pair).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_neon(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;
            // SAFETY: target_feature = "aes" enabled at compile time.
            unsafe {
                if block_size_half >= 2 {
                    // Within-block: batch 2 pairs with shared twiddle.
                    for block in 0..num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        let chunk = &mut data[block_start..block_start + block_size];
                        butterfly_neon_block(chunk, twiddle, block_size_half);
                    }
                } else {
                    // Deepest layer (half = 1): batch across 2 adjacent blocks
                    // (different twiddles). Handle odd tail with scalar when
                    // num_blocks = 1 (only happens at log_d = 1).
                    debug_assert_eq!(block_size_half, 1);
                    let mut block = 0;
                    while block + 1 < num_blocks {
                        let t_a = self.twiddle(layer, block);
                        let t_b = self.twiddle(layer, block + 1);
                        butterfly_neon_block_pair(data, block * 2, t_a, t_b);
                        block += 2;
                    }
                    // Scalar tail (num_blocks odd — only when num_blocks = 1).
                    while block < num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let idx0 = block * 2;
                        let idx1 = idx0 + 1;
                        let v = data[idx1];
                        let new_u = data[idx0] + v * twiddle;
                        data[idx0] = new_u;
                        data[idx1] = v + new_u;
                        block += 1;
                    }
                }
            }
        }
    }

    /// Rayon-parallel + NEON forward transform.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_parallel(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        // For small data (or shallow layers with few large blocks), the rayon
        // overhead exceeds the gain — fall back to the NEON single-thread path.
        const PARALLEL_THRESHOLD_LOG: usize = 14; // 2^14 = 16K elements (256 KB)
        if log_d <= PARALLEL_THRESHOLD_LOG {
            self.forward_transform_neon(data);
            return;
        }

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;

            // Parallelize across blocks when there are enough; otherwise process
            // sequentially with NEON (still fast for small block counts).
            if num_blocks >= 4 && block_size_half >= 2 {
                let twiddles: Vec<F128> = (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                data.par_chunks_mut(block_size)
                    .zip(twiddles.par_iter())
                    .for_each(|(chunk, &twiddle)| {
                        // SAFETY: aes target feature enabled.
                        unsafe { butterfly_neon_block(chunk, twiddle, block_size_half) };
                    });
            } else if block_size_half >= 2 {
                // Few large blocks — process sequentially with NEON.
                // SAFETY: aes target feature enabled.
                unsafe {
                    for block in 0..num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        butterfly_neon_block(
                            &mut data[block_start..block_start + block_size],
                            twiddle,
                            block_size_half,
                        );
                    }
                }
            } else {
                // Deepest layer (half = 1): need num_blocks ≥ 2 to batch
                // pairs; if there are at least 2 blocks, batch across them.
                // (When num_blocks < 2, fall back to NEON-single-thread which
                // handles the trivial cases.)
                debug_assert_eq!(block_size_half, 1);
                if num_blocks >= 2 {
                    let twiddles: Vec<F128> =
                        (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                    data.par_chunks_mut(4).zip(twiddles.par_chunks(2)).for_each(
                        |(chunk, twiddle_pair)| {
                            // SAFETY: aes target feature enabled.
                            unsafe {
                                butterfly_neon_block_pair_chunk(
                                    chunk,
                                    twiddle_pair[0],
                                    twiddle_pair[1],
                                )
                            };
                        },
                    );
                } else {
                    let twiddle = self.twiddle(layer, 0);
                    let v = data[1];
                    let new_u = data[0] + v * twiddle;
                    data[0] = new_u;
                    data[1] = v + new_u;
                }
            }
        }
    }

    /// Cache-blocked + parallel + NEON forward transform.
    ///
    /// **Strategy**: decompose the NTT into two stages so the deep layers
    /// (which dominate work) operate on sub-buffers small enough to fit in L2
    /// cache, avoiding the DRAM round-trip per layer.
    ///
    /// 1. **Top layers** (layers `0..n_top`): each layer touches the full buffer
    ///    in one sweep. Bandwidth-bound; parallelize across blocks.
    /// 2. **Deep layers** (layers `n_top..log_d`): treat the data as `2^n_top`
    ///    independent sub-NTTs, each of size `2^(log_d − n_top)`. For each
    ///    sub-NTT, process ALL remaining layers in one cache-resident pass.
    ///    Parallelize across sub-NTTs via rayon.
    ///
    /// `n_top` is chosen so each sub-NTT is `≈ 2 MB` (= `2^17` F_{2^128} ≈ 2 MB).
    /// For `log_d ≤ 17` the whole NTT fits in cache and we fall back to the
    /// per-layer parallel path.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_batched(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        // Target sub-NTT size: 2^17 F_{2^128} = 2 MB. Tunable.
        const TARGET_SUB_NTT_LOG: usize = 17;
        if log_d <= TARGET_SUB_NTT_LOG {
            self.forward_transform_parallel(data);
            return;
        }
        let n_top = log_d - TARGET_SUB_NTT_LOG;
        let sub_ntt_size = 1usize << (log_d - n_top);

        // ---- Stage 1: top layers (full-buffer, bandwidth-bound).
        for layer in 0..n_top {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;

            if num_blocks >= 4 {
                let twiddles: Vec<F128> = (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                data.par_chunks_mut(block_size)
                    .zip(twiddles.par_iter())
                    .for_each(|(chunk, &t)| {
                        // SAFETY: aes target feature enabled.
                        unsafe { butterfly_neon_block(chunk, t, block_size_half) };
                    });
            } else {
                // Few large blocks at very top layers: sequential NEON.
                unsafe {
                    for block in 0..num_blocks {
                        let t = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        butterfly_neon_block(
                            &mut data[block_start..block_start + block_size],
                            t,
                            block_size_half,
                        );
                    }
                }
            }
        }

        // ---- Stage 2: deep layers as parallel cache-resident sub-NTTs.
        data.par_chunks_mut(sub_ntt_size)
            .enumerate()
            .for_each(|(sub_idx, sub_data)| {
                for layer in n_top..log_d {
                    let layer_in_sub = layer - n_top;
                    let num_blocks_in_sub = 1usize << layer_in_sub;
                    let block_size = 1usize << (log_d - layer);
                    let block_size_half = block_size >> 1;

                    for block_in_sub in 0..num_blocks_in_sub {
                        let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                        let twiddle = self.twiddle(layer, global_block);
                        let block_start = block_in_sub * block_size;
                        let block = &mut sub_data[block_start..block_start + block_size];
                        if block_size_half >= 2 {
                            // SAFETY: aes target feature enabled.
                            unsafe { butterfly_neon_block(block, twiddle, block_size_half) };
                        } else {
                            // Deepest layer: 1 pair per block, scalar.
                            let v = block[1];
                            let new_u = block[0] + v * twiddle;
                            block[0] = new_u;
                            block[1] = v + new_u;
                        }
                    }
                }
            });
    }

    /// Inverse additive NTT in place. Exact inverse of `forward_transform`.
    pub fn inverse_transform(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in (0..log_d).rev() {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;
                    // Inverse butterfly: v += u; u += v·twiddle.
                    let u = data[idx0];
                    let new_v = data[idx1] + u;
                    data[idx1] = new_v;
                    data[idx0] = u + new_v * twiddle;
                }
            }
        }
    }
}

/// Like [`butterfly_interleaved_block`] but parallelizes across rows via
/// rayon. Used at top layers where the block is large (≥ 1024 rows) and only
/// 1-2 blocks exist (so block-level parallelism would be too coarse).
///
/// Falls back to sequential when the row count is small.
#[inline]
fn butterfly_interleaved_block_par_rows(
    block: &mut [F128],
    twiddle: F128,
    block_size_half: usize,
    num_ntts: usize,
    live: usize,
) {
    const PARALLEL_ROW_THRESHOLD: usize = 512;
    if block_size_half < PARALLEL_ROW_THRESHOLD {
        butterfly_interleaved_block(block, twiddle, block_size_half, num_ntts, live);
        return;
    }
    let half_offset = block_size_half * num_ntts;
    let (top, bot) = block.split_at_mut(half_offset);
    // Zero-twiddle fast path (see `butterfly_interleaved_block`).
    if twiddle == F128::ZERO {
        top.par_chunks_mut(num_ntts)
            .zip(bot.par_chunks_mut(num_ntts))
            .for_each(|(top_row, bot_row)| {
                for lane in 0..num_ntts {
                    bot_row[lane] += top_row[lane];
                }
            });
        return;
    }
    top.par_chunks_mut(num_ntts)
        .zip(bot.par_chunks_mut(num_ntts))
        .for_each(|(top_row, bot_row)| {
            butterfly_row_pair(&mut top_row[..live], &mut bot_row[..live], twiddle);
        });
}

/// Fused 3-layer butterfly over one layer-L block: 8-point sub-butterflies,
/// rows `r + k·eighth` for `k ∈ 0..8` per group. Layer L pairs at distance
/// `4·eighth` (@ `t0`), L+1 at `2·eighth` (@ `t1[half]`), L+2 at `eighth`
/// (@ `t2[quarter]`). One read+write pass over the block for three layers.
#[allow(clippy::too_many_arguments)]
#[inline]
fn butterfly_interleaved_fused_3layer_par_rows(
    block: &mut [F128],
    t0: F128,
    t1: [F128; 2],
    t2: [F128; 4],
    eighth: usize,
    num_ntts: usize,
    live: usize,
) {
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    let stride = eighth * num_ntts;
    debug_assert_eq!(block.len(), 8 * stride);

    // Split the block into eight eighths, then zip row-wise: each task is
    // one row-group index = 8 logical rows of work.
    let (q0, rest) = block.split_at_mut(stride);
    let (q1, rest) = rest.split_at_mut(stride);
    let (q2, rest) = rest.split_at_mut(stride);
    let (q3, rest) = rest.split_at_mut(stride);
    let (q4, rest) = rest.split_at_mut(stride);
    let (q5, rest) = rest.split_at_mut(stride);
    let (q6, q7) = rest.split_at_mut(stride);

    if eighth < PARALLEL_ROW_THRESHOLD {
        for r in 0..eighth {
            let o = r * num_ntts;
            butterfly_fused_3layer(
                [
                    &mut q0[o..o + live],
                    &mut q1[o..o + live],
                    &mut q2[o..o + live],
                    &mut q3[o..o + live],
                    &mut q4[o..o + live],
                    &mut q5[o..o + live],
                    &mut q6[o..o + live],
                    &mut q7[o..o + live],
                ],
                t0,
                &t1,
                &t2,
            );
        }
    } else {
        q0.par_chunks_mut(num_ntts)
            .zip(q1.par_chunks_mut(num_ntts))
            .zip(q2.par_chunks_mut(num_ntts))
            .zip(q3.par_chunks_mut(num_ntts))
            .zip(q4.par_chunks_mut(num_ntts))
            .zip(q5.par_chunks_mut(num_ntts))
            .zip(q6.par_chunks_mut(num_ntts))
            .zip(q7.par_chunks_mut(num_ntts))
            .for_each(|(((((((r0, r1), r2), r3), r4), r5), r6), r7)| {
                butterfly_fused_3layer(
                    [
                        &mut r0[..live],
                        &mut r1[..live],
                        &mut r2[..live],
                        &mut r3[..live],
                        &mut r4[..live],
                        &mut r5[..live],
                        &mut r6[..live],
                        &mut r7[..live],
                    ],
                    t0,
                    &t1,
                    &t2,
                );
            });
    }
}

/// One quarter-row of the fused 2-layer butterfly, with the zero-twiddle
/// short-circuit for outer block 0 (`t_outer = t_inner_a = 0` — only the
/// (c,d) inner butterfly multiplies; the branch is per-row, not per-lane).
/// The general case delegates to the arch-dispatched
/// [`butterfly_fused_2layer`].
#[inline(always)]
fn fused_2layer_row_op(
    row_a: &mut [F128],
    row_b: &mut [F128],
    row_c: &mut [F128],
    row_d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    zero_block: bool,
) {
    if zero_block {
        for lane in 0..row_a.len() {
            let a = row_a[lane];
            let b = row_b[lane];
            // Layer L with t_outer=0: a,b unchanged; c += a; d += b.
            let c = row_c[lane] + a;
            let d = row_d[lane] + b;
            // Layer L+1: (a,b) with t_inner_a=0: a unchanged; b += a.
            row_b[lane] = b + a;
            // (c,d) with the real t_inner_b.
            let new_c2 = c + d * t_inner_b;
            row_c[lane] = new_c2;
            row_d[lane] = d + new_c2;
        }
        return;
    }
    butterfly_fused_2layer(row_a, row_b, row_c, row_d, t_outer, t_inner_a, t_inner_b);
}

/// Forced-serial fused 2-layer butterfly for use INSIDE the deep pass's
/// per-sub-group workers (already running one task per pool worker — nested
/// row-parallelism would only add dispatch overhead). Same math as
/// [`butterfly_interleaved_fused_2layer_par_rows`].
fn butterfly_interleaved_fused_2layer_serial(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
    live: usize,
) {
    let stride = quarter * num_ntts;
    debug_assert_eq!(block.len(), 4 * stride);
    let zero_block = t_outer == F128::ZERO && t_inner_a == F128::ZERO;
    let (top_half, bot_half) = block.split_at_mut(2 * stride);
    let (q1, q2) = top_half.split_at_mut(stride);
    let (q3, q4) = bot_half.split_at_mut(stride);
    for r in 0..quarter {
        let off = r * num_ntts;
        let (q1r, _) = q1[off..].split_at_mut(num_ntts);
        let (q2r, _) = q2[off..].split_at_mut(num_ntts);
        let (q3r, _) = q3[off..].split_at_mut(num_ntts);
        let (q4r, _) = q4[off..].split_at_mut(num_ntts);
        fused_2layer_row_op(
            &mut q1r[..live],
            &mut q2r[..live],
            &mut q3r[..live],
            &mut q4r[..live],
            t_outer,
            t_inner_a,
            t_inner_b,
            zero_block,
        );
    }
}

/// Fused 2-layer butterfly: combines layer L (twiddle `t_outer`, shared by
/// the whole outer block) with layer L+1 (twiddles `t_inner_a` for the top
/// half, `t_inner_b` for the bottom half). Reads each row of the outer
/// block once and writes once — halving memory traffic vs running the two
/// layers as separate sweeps.
///
/// `block` has length `4 * quarter * num_ntts` (= one layer-L block of
/// `4*quarter` rows). For each `r ∈ 0..quarter`, four rows participate:
/// `a=r`, `b=r+quarter`, `c=r+2*quarter`, `d=r+3*quarter`. Layer L
/// butterflies `(a,c)` and `(b,d)`; layer L+1 then butterflies `(a,b)` (in
/// the new top sub-block) and `(c,d)` (in the new bottom sub-block).
#[inline]
fn butterfly_interleaved_fused_2layer_par_rows(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
    live: usize,
) {
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    let stride = quarter * num_ntts;
    debug_assert_eq!(block.len(), 4 * stride);

    // Block 0 of the outer layer has t_outer = 0 AND t_inner_a =
    // twiddle(L+1, 0) = 0 — only the (c,d) inner butterfly multiplies. The
    // branch is per-row, not per-lane.
    let zero_block = t_outer == F128::ZERO && t_inner_a == F128::ZERO;
    let do_one =
        |row_a: &mut [F128], row_b: &mut [F128], row_c: &mut [F128], row_d: &mut [F128]| {
            fused_2layer_row_op(
                row_a, row_b, row_c, row_d, t_outer, t_inner_a, t_inner_b, zero_block,
            );
        };

    // Split the block into four quarters, then zip row-wise. Each rayon task
    // processes one quarter-row index = 4 logical rows of work.
    let (top_half, bot_half) = block.split_at_mut(2 * stride);
    let (q1, q2) = top_half.split_at_mut(stride);
    let (q3, q4) = bot_half.split_at_mut(stride);

    if quarter < PARALLEL_ROW_THRESHOLD {
        for r in 0..quarter {
            let off = r * num_ntts;
            let (q1r, q1_rest) = q1[off..].split_at_mut(num_ntts);
            let _ = q1_rest;
            let (q2r, _) = q2[off..].split_at_mut(num_ntts);
            let (q3r, _) = q3[off..].split_at_mut(num_ntts);
            let (q4r, _) = q4[off..].split_at_mut(num_ntts);
            do_one(
                &mut q1r[..live],
                &mut q2r[..live],
                &mut q3r[..live],
                &mut q4r[..live],
            );
        }
    } else {
        q1.par_chunks_mut(num_ntts)
            .zip(q2.par_chunks_mut(num_ntts))
            .zip(q3.par_chunks_mut(num_ntts))
            .zip(q4.par_chunks_mut(num_ntts))
            .for_each(|(((row_a, row_b), row_c), row_d)| {
                do_one(
                    &mut row_a[..live],
                    &mut row_b[..live],
                    &mut row_c[..live],
                    &mut row_d[..live],
                );
            });
    }
}

/// Butterfly one block of an interleaved (SoA) buffer with shared twiddle.
///
/// `block` has length `(2 * block_size_half) * num_ntts` and is laid out as
/// `num_ntts` lanes interleaved per row, `2 * block_size_half` rows total.
/// Pairs row `r` with row `r + block_size_half` for `r ∈ 0..block_size_half`.
///
/// **Note**: This is scalar-per-lane on purpose. With `num_ntts = 32` and
/// shared twiddle, the inner loop has 32 independent F_{2^128} muls per row
/// that the compiler ILPs effectively (each mul uses NEON via the field's
/// `binius_mul` already). An explicit 2-lane `ghash_mul_vec2_neon` variant was
/// tried but **regressed** by ~10-30% because the explicit batching prevented
/// ILP across more than 2 muls and added load/store overhead.
#[inline]
/// `v · t` for a HALF-WIDTH twiddle (`t.hi == 0`, i.e. deg(t) ≤ 63): the
/// schoolbook shrinks to 2 PMULL and the 192-bit product needs only a single
/// reduction fold (overflow deg ≤ 62+7 < 128) — half the cost of the general
/// multiply. In the polynomial basis `{1, x, x², …}` the THREE DEEPEST layers'
/// twiddles are all half-width (they are spans/short combinations of low
/// basis powers): 3/17 ≈ 18% of all NTT mults.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
fn mul_small_twiddle(v: F128, t_lo: u64) -> F128 {
    unsafe {
        let d0 = vreinterpretq_u64_p128(vmull_p64(v.lo, t_lo));
        let d1 = vreinterpretq_u64_p128(vmull_p64(v.hi, t_lo));
        // 192-bit product: r0 = d0.lo, r1 = d0.hi ^ d1.lo, r2 = d1.hi (r3 = 0).
        let r2 = vgetq_lane_u64::<1>(d1);
        // One fold: r2 · (x^7+x^2+x+1) lands entirely within 128 bits.
        let h = vreinterpretq_u64_p128(vmull_p64(r2, 0x87));
        F128 {
            lo: vgetq_lane_u64::<0>(d0) ^ vgetq_lane_u64::<0>(h),
            hi: vgetq_lane_u64::<1>(d0) ^ vgetq_lane_u64::<0>(d1) ^ vgetq_lane_u64::<1>(h),
        }
    }
}

fn butterfly_interleaved_block(
    block: &mut [F128],
    twiddle: F128,
    block_size_half: usize,
    num_ntts: usize,
    live: usize,
) {
    let off_bot = block_size_half * num_ntts;
    // Zero-twiddle fast path: block 0 of EVERY layer has twiddle 0
    // (`twiddle(l, 0) = span_get(_, 0) = 0`), so its butterfly degenerates to
    // (u, v + u) — no multiply. Σ_l 2^-l ≈ 1 layer-equivalent ≈ 6% of all NTT
    // mults, and the NTT is mult-throughput-bound.
    if twiddle == F128::ZERO {
        for r in 0..block_size_half {
            let off_top = r * num_ntts;
            let off_bot_r = off_top + off_bot;
            for lane in 0..num_ntts {
                let u = block[off_top + lane];
                block[off_bot_r + lane] += u;
            }
        }
        return;
    }
    // Half-width-twiddle fast path (see `mul_small_twiddle`): the deep layers
    // this kernel serves are exactly where all twiddles are half-width.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    if twiddle.hi == 0 {
        for r in 0..block_size_half {
            let off_top = r * num_ntts;
            let off_bot_r = off_top + off_bot;
            for lane in 0..num_ntts {
                let v = block[off_bot_r + lane];
                let new_u = block[off_top + lane] + mul_small_twiddle(v, twiddle.lo);
                block[off_top + lane] = new_u;
                block[off_bot_r + lane] = v + new_u;
            }
        }
        return;
    }
    let (top, bot) = block.split_at_mut(off_bot);
    for r in 0..block_size_half {
        let o = r * num_ntts;
        butterfly_row_pair(&mut top[o..o + live], &mut bot[o..o + live], twiddle);
    }
}

/// Butterfly one top-layer block, fusing four layers `(L..L+4)`. `block` holds
/// `16 * sixteenth` rows of `num_ntts` lanes; `t` carries the 15 twiddles for
/// the sub-butterflies (see module comment above). Parallel over row groups.
#[inline]
fn butterfly_interleaved_fused_4layer_par_rows(
    block: &mut [F128],
    t: &[F128; 15],
    sixteenth: usize,
    num_ntts: usize,
) {
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    debug_assert_eq!(block.len(), 16 * sixteenth * num_ntts);
    // Carry the base as `usize` (Send+Sync) so rayon's per-`r` closure can hold
    // it without a raw-pointer `Sync` shim. Each `r` writes the disjoint rows
    // `{i*sixteenth + r : i ∈ 0..16}`, so concurrent writes never alias.
    let base = block.as_mut_ptr() as usize;
    if sixteenth < PARALLEL_ROW_THRESHOLD {
        for r in 0..sixteenth {
            // SAFETY: row group r writes disjoint rows of this block.
            unsafe { butterfly_fused_4layer_row(base as *mut F128, sixteenth, num_ntts, r, t) };
        }
    } else {
        (0..sixteenth).into_par_iter().for_each(|r| {
            // SAFETY: distinct r → disjoint row groups → no aliasing.
            unsafe { butterfly_fused_4layer_row(base as *mut F128, sixteenth, num_ntts, r, t) };
        });
    }
}

#[inline]
fn log2_pow2(n: usize) -> usize {
    assert!(
        n.is_power_of_two() && n > 0,
        "length must be a positive power of 2"
    );
    n.trailing_zeros() as usize
}

/// Ceil of `log2(n)` for any `n ≥ 1` (`ceil_log2(1) = 0`). Used only by the
/// cache-blocking heuristic for the interleaved transform, where `num_ntts`
/// may be a non-power-of-two integer lane count — rounding up makes an integer
/// `t` block exactly like the padded `2^ceil(log2 t)` power-of-two width.
#[inline]
fn ceil_log2(n: usize) -> usize {
    debug_assert!(n >= 1);
    n.next_power_of_two().trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_rng::Rng;

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<F128> {
        (0..n).map(|_| rng.f128()).collect()
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let mut rng = Rng::new(0xAB1);
        for log_d in [1usize, 2, 3, 4, 6, 8] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v = original.clone();
            ntt.forward_transform(&mut v);
            ntt.inverse_transform(&mut v);
            assert_eq!(v, original, "roundtrip failed at log_d={log_d}");
        }
    }

    #[test]
    fn inverse_forward_roundtrip() {
        let mut rng = Rng::new(0xAB2);
        for log_d in [1usize, 2, 3, 4, 6, 8] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v = original.clone();
            ntt.inverse_transform(&mut v);
            ntt.forward_transform(&mut v);
            assert_eq!(
                v, original,
                "inverse∘forward roundtrip failed at log_d={log_d}"
            );
        }
    }

    #[test]
    fn forward_is_linear() {
        let mut rng = Rng::new(0xAB3);
        for log_d in [1usize, 2, 3, 5] {
            let ntt = AdditiveNttF128::standard(log_d);
            let n = 1 << log_d;
            let a = rand_vec(&mut rng, n);
            let b = rand_vec(&mut rng, n);
            let ab: Vec<F128> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();

            let mut fa = a.clone();
            ntt.forward_transform(&mut fa);
            let mut fb = b.clone();
            ntt.forward_transform(&mut fb);
            let mut fab = ab.clone();
            ntt.forward_transform(&mut fab);

            for i in 0..n {
                assert_eq!(
                    fa[i] + fb[i],
                    fab[i],
                    "linearity fails at i={i}, log_d={log_d}"
                );
            }
        }
    }

    #[test]
    fn ntt_of_zero_is_zero() {
        for log_d in [1usize, 2, 3, 6] {
            let ntt = AdditiveNttF128::standard(log_d);
            let mut v = vec![F128::ZERO; 1 << log_d];
            ntt.forward_transform(&mut v);
            assert!(v.iter().all(|&x| x == F128::ZERO));
        }
    }

    #[test]
    fn twiddle_at_layer_0_uses_full_basis_minus_one() {
        // At layer 0 (topmost forward butterfly), there's 1 block.
        // twiddle(0, 0) = 0 (no bits set in block index 0).
        let ntt = AdditiveNttF128::standard(4);
        assert_eq!(ntt.twiddle(0, 0), F128::ZERO);
    }

    /// At layer log_d - 1 (deepest, where FRI starts), pairs are adjacent.
    /// twiddle should match the "domain points" indexing.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_matches_scalar() {
        let mut rng = Rng::new(0xBB1);
        for log_d in 1..=10 {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_neon = original.clone();
            ntt.forward_transform_neon(&mut v_neon);
            assert_eq!(
                v_neon, v_scalar,
                "NEON disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[test]
    fn interleaved_matches_per_lane() {
        let mut rng = Rng::new(0xCC1);
        // For several log_d and num_ntts, verify the interleaved transform
        // matches running the per-lane scalar transform on each sub-NTT.
        for log_d in [3usize, 4, 8] {
            for num_ntts in [1usize, 2, 4, 8] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n_total = (1 << log_d) * num_ntts;
                let original = rand_vec(&mut rng, n_total);

                // Interleaved.
                let mut v_inter = original.clone();
                ntt.forward_transform_interleaved_scalar(&mut v_inter, num_ntts);

                // Reference: per-lane, gather + scalar transform + scatter.
                let mut v_ref = original.clone();
                for lane in 0..num_ntts {
                    let mut sub: Vec<F128> = (0..(1 << log_d))
                        .map(|pos| v_ref[pos * num_ntts + lane])
                        .collect();
                    ntt.forward_transform_scalar(&mut sub);
                    for pos in 0..(1 << log_d) {
                        v_ref[pos * num_ntts + lane] = sub[pos];
                    }
                }

                assert_eq!(
                    v_inter, v_ref,
                    "interleaved mismatch at log_d={log_d}, num_ntts={num_ntts}"
                );
            }
        }
    }

    // Runs on both SIMD backends so the x86 PCLMUL and aarch64 NEON parallel
    // paths are each validated against the scalar oracle. AVX-512 builds also
    // exercise the fused-4 top-layer kernel in the larger cases.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    #[test]
    fn interleaved_parallel_matches_scalar() {
        let mut rng = Rng::new(0xCC2);
        for log_d in [4usize, 10, 14, 17, 19] {
            for &num_ntts in &[2usize, 8, 32] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n_total = (1 << log_d) * num_ntts;
                let original = rand_vec(&mut rng, n_total);
                let mut v_scalar = original.clone();
                ntt.forward_transform_interleaved_scalar(&mut v_scalar, num_ntts);
                let mut v_par = original.clone();
                ntt.forward_transform_interleaved_parallel(&mut v_par, num_ntts);
                assert_eq!(
                    v_par, v_scalar,
                    "interleaved parallel mismatch at log_d={log_d}, num_ntts={num_ntts}"
                );
            }
        }
    }

    /// Oracle 2 (integer-lane encode correctness): the `t`-lane interleaved
    /// encode of a dense buffer `D` (SoA stride `t`) is byte-identical, per
    /// real lane, to the `2^k`-lane encode of the SAME data zero-padded in the
    /// lane dimension (SoA stride `2^k`, top lanes zero). Covers the scalar and
    /// the parallel/SIMD paths, and several non-power-of-two `t`.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    #[test]
    fn interleaved_integer_lanes_match_padded() {
        let mut rng = Rng::new(0x1A6E_2026);
        for log_d in [4usize, 8, 12, 14] {
            let positions = 1usize << log_d;
            for &t in &[1usize, 3, 5, 7, 13, 46, 63] {
                let padded_lanes = t.next_power_of_two();
                let ntt = AdditiveNttF128::standard(log_d);

                // Dense t-lane buffer D[pos*t + lane].
                let dense = rand_vec(&mut rng, positions * t);
                // Zero-pad the lane dimension to `padded_lanes`.
                let mut padded = vec![F128::ZERO; positions * padded_lanes];
                for pos in 0..positions {
                    for lane in 0..t {
                        padded[pos * padded_lanes + lane] = dense[pos * t + lane];
                    }
                }

                // Encode both (parallel path — exercises the SIMD tail-lane
                // handling for non-power-of-two t).
                let mut enc_t = dense.clone();
                ntt.forward_transform_interleaved_parallel(&mut enc_t, t);
                let mut enc_padded = padded.clone();
                ntt.forward_transform_interleaved_parallel(&mut enc_padded, padded_lanes);

                for pos in 0..positions {
                    for lane in 0..t {
                        assert_eq!(
                            enc_t[pos * t + lane],
                            enc_padded[pos * padded_lanes + lane],
                            "lane {lane} pos {pos} diverged (log_d={log_d}, t={t})"
                        );
                    }
                }

                // The scalar path must agree with the parallel one too.
                let mut enc_t_scalar = dense.clone();
                ntt.forward_transform_interleaved_scalar(&mut enc_t_scalar, t);
                assert_eq!(enc_t_scalar, enc_t, "scalar vs parallel (t={t})");
            }
        }
    }

    /// The DEAD-LANE SKIP oracle: a buffer whose trailing lanes are zero,
    /// transformed with `live` bounded, is byte-identical to the full
    /// transform (zeros are butterfly fixed points). Sizes above and below
    /// the parallel thresholds; (24, 18) is the envelope internal's shape;
    /// live = 0 is the all-zero degenerate; a nonzero start layer is the
    /// commit path's replicate-fill entry.
    #[test]
    fn interleaved_live_skip_byte_identical() {
        let mut rng = Rng::new(0x11FE_5C1B);
        for log_d in [4usize, 8, 12, 14] {
            let positions = 1usize << log_d;
            for &(t, live) in &[(7usize, 4usize), (24, 18), (24, 23), (5, 0), (13, 13)] {
                let ntt = AdditiveNttF128::standard(log_d);
                let mut data = vec![F128::ZERO; positions * t];
                for pos in 0..positions {
                    for lane in 0..live {
                        data[pos * t + lane] = rng.f128();
                    }
                }
                let mut full = data.clone();
                ntt.forward_transform_interleaved(&mut full, t);
                let mut skip = data.clone();
                ntt.forward_transform_interleaved_live_from_layer(&mut skip, t, live, 0);
                assert_eq!(skip, full, "live skip (log_d={log_d}, t={t}, live={live})");
                // Both sides from the same nonzero start layer: the bounded
                // and full transforms must stay in lockstep there too.
                let mut full2 = data.clone();
                ntt.forward_transform_interleaved_from_layer(&mut full2, t, 2);
                let mut skip2 = data.clone();
                ntt.forward_transform_interleaved_live_from_layer(&mut skip2, t, live, 2);
                assert_eq!(
                    skip2, full2,
                    "live skip from layer 2 (log_d={log_d}, t={t}, live={live})"
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn batched_matches_scalar() {
        let mut rng = Rng::new(0xBB4);
        // Include sizes above the TARGET_SUB_NTT_LOG threshold (17) so we
        // exercise the cache-blocked path.
        for log_d in [4usize, 8, 12, 17, 18, 19, 20] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_batched = original.clone();
            ntt.forward_transform_batched(&mut v_batched);
            assert_eq!(
                v_batched, v_scalar,
                "batched disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn parallel_matches_scalar() {
        let mut rng = Rng::new(0xBB2);
        for log_d in [4usize, 8, 12, 15, 16] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_par = original.clone();
            ntt.forward_transform_parallel(&mut v_par);
            assert_eq!(
                v_par, v_scalar,
                "parallel disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[test]
    fn deepest_layer_twiddle_count() {
        let log_d = 4;
        let ntt = AdditiveNttF128::standard(log_d);
        // At layer log_d - 1 = 3, there are 2^3 = 8 blocks. twiddle(3, b) for b ∈ 0..8.
        for b in 0..8 {
            let _t = ntt.twiddle(log_d - 1, b);
        }
    }
}
