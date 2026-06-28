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

use crate::field::F128;

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
    /// `num_ntts` must be a positive power of 2. `data.len()` must equal
    /// `(1 << log_d) * num_ntts` for some `log_d ≤ log_domain_size()`.
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
        assert!(num_ntts.is_power_of_two() && num_ntts > 0);
        let n_total = data.len();
        assert_eq!(n_total % num_ntts, 0);
        let log_d = log2_pow2(n_total / num_ntts);
        assert!(log_d <= self.log_domain_size());
        assert!(start_layer <= log_d);

        // Scalar; SIMD/parallel variants below dispatch from `forward_transform_interleaved`
        // on supported targets.
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            self.forward_transform_interleaved_parallel_from_layer(data, num_ntts, start_layer);
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        {
            self.forward_transform_interleaved_x86_from_layer(data, num_ntts, start_layer);
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )
        )))]
        {
            self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, start_layer);
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
                    for lane in 0..num_ntts {
                        let v = data[off_bot + lane];
                        let new_u = data[off_top + lane] + v * twiddle;
                        data[off_top + lane] = new_u;
                        data[off_bot + lane] = v + new_u;
                    }
                }
            }
        }
    }

    /// x86-64 (AVX-512 VPCLMULQDQ) interleaved forward NTT — the x86 sibling of
    /// the NEON path. Identical butterfly to the scalar reference, but exploits
    /// that **all butterflies in a block share one twiddle**: within a block the
    /// `top` half `[0, half_elems)` and `bot` half `[half_elems, 2·half_elems)`
    /// are contiguous and pair index-for-index, so the inner loop flattens to a
    /// run of `half_elems` butterflies multiplied **4 F128 at a time** by the
    /// broadcast twiddle via [`x86_64::ghash_mul_zmm`]. `num_ntts`-agnostic;
    /// the scalar tail handles deep layers where `half_elems < 4`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    pub fn forward_transform_interleaved_x86_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        use crate::field::gf2_128::x86_64::ghash_mul_zmm;
        use core::arch::x86_64::*;
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        for layer in start_layer..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let half_elems = (block_size >> 1) * num_ntts;
            let block_elems = block_size * num_ntts;
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block * block_elems;
                // SAFETY: avx512f/avx512bw/vpclmulqdq are statically enabled
                // (target-cpu=native on Granite Rapids). All accesses stay within
                // data[block_start .. block_start + block_elems]; the top range
                // [0, half_elems) and bot range [half_elems, 2·half_elems) are
                // disjoint, so the raw-pointer reads/writes do not alias.
                unsafe {
                    let tw = _mm512_broadcast_i32x4(_mm_loadu_si128(
                        &twiddle as *const F128 as *const __m128i,
                    ));
                    let base = data.as_mut_ptr().add(block_start);
                    let mut i = 0usize;
                    while i + 4 <= half_elems {
                        let top_p = base.add(i) as *mut __m512i;
                        let bot_p = base.add(half_elems + i) as *mut __m512i;
                        let vbot = _mm512_loadu_si512(bot_p as *const __m512i);
                        let vtop = _mm512_loadu_si512(top_p as *const __m512i);
                        let new_u = _mm512_xor_si512(vtop, ghash_mul_zmm(vbot, tw));
                        _mm512_storeu_si512(top_p, new_u);
                        _mm512_storeu_si512(bot_p, _mm512_xor_si512(vbot, new_u));
                        i += 4;
                    }
                    while i < half_elems {
                        let v = *base.add(half_elems + i);
                        let new_u = *base.add(i) + v * twiddle;
                        *base.add(i) = new_u;
                        *base.add(half_elems + i) = v + new_u;
                        i += 1;
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
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_interleaved_parallel(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_parallel_from_layer(data, num_ntts, 0);
    }

    /// Parallel interleaved forward NTT from `start_layer` (see
    /// [`Self::forward_transform_interleaved_from_layer`]).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_interleaved_parallel_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        use rayon::prelude::*;
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        // Target sub-group size = 2 MB total bytes. Each position is
        // `num_ntts × 16` bytes, so positions per sub-group =
        // 2^21 / (num_ntts · 16). With num_ntts=1: 2^17 positions. With
        // num_ntts=32: 2^12 positions. (Without this scaling, sub-groups at
        // num_ntts=32 would be 64 MB and overflow L2 cache.)
        const TARGET_SUBGROUP_LOG_BYTES: usize = 21;
        let log_bytes_per_position = 4 + log2_pow2(num_ntts);
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
            let want_subs_log = log2_pow2(rayon::current_num_threads().next_power_of_two());
            let max_n_top = log_d.saturating_sub(MIN_SUB_LOG);
            cache_n_top.max(want_subs_log.min(max_n_top))
        } else {
            cache_n_top
        };
        if n_top == 0 || log_d < 8 {
            self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, start_layer);
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
        let mut layer = start_layer.min(n_top);
        while layer < n_top {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_bytes = block_size * num_ntts;

            if layer + 1 < n_top && block_size >= 4 {
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
                    );
                }
                layer += 1;
            }
        }

        // Deep layers: process each sub-NTT-group cache-resident.
        let sub_size_positions = 1usize << (log_d - n_top);
        let sub_bytes = sub_size_positions * num_ntts;

        data.par_chunks_mut(sub_bytes)
            .enumerate()
            .for_each(|(sub_idx, sub_data)| {
                for layer in n_top.max(start_layer)..log_d {
                    let layer_in_sub = layer - n_top;
                    let num_blocks_in_sub = 1usize << layer_in_sub;
                    let block_size = 1usize << (log_d - layer);
                    let block_size_half = block_size >> 1;
                    let block_bytes = block_size * num_ntts;

                    for block_in_sub in 0..num_blocks_in_sub {
                        let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                        let twiddle = self.twiddle(layer, global_block);
                        let block_start = block_in_sub * block_bytes;
                        let block = &mut sub_data[block_start..block_start + block_bytes];
                        butterfly_interleaved_block(block, twiddle, block_size_half, num_ntts);
                    }
                }
            });
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
                        butterfly_block_neon(chunk, twiddle, block_size_half);
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
                        butterfly_across_blocks_neon(data, block * 2, t_a, t_b);
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
        use rayon::prelude::*;
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
                        unsafe { butterfly_block_neon(chunk, twiddle, block_size_half) };
                    });
            } else if block_size_half >= 2 {
                // Few large blocks — process sequentially with NEON.
                // SAFETY: aes target feature enabled.
                unsafe {
                    for block in 0..num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        butterfly_block_neon(
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
                                butterfly_across_blocks_neon_in_chunk(
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
        use rayon::prelude::*;
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
                        unsafe { butterfly_block_neon(chunk, t, block_size_half) };
                    });
            } else {
                // Few large blocks at very top layers: sequential NEON.
                unsafe {
                    for block in 0..num_blocks {
                        let t = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        butterfly_block_neon(
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
                            unsafe { butterfly_block_neon(block, twiddle, block_size_half) };
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
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 512;
    if block_size_half < PARALLEL_ROW_THRESHOLD {
        butterfly_interleaved_block(block, twiddle, block_size_half, num_ntts);
        return;
    }
    let half_offset = block_size_half * num_ntts;
    let (top, bot) = block.split_at_mut(half_offset);
    top.par_chunks_mut(num_ntts)
        .zip(bot.par_chunks_mut(num_ntts))
        .for_each(|(top_row, bot_row)| {
            for lane in 0..num_ntts {
                let v = bot_row[lane];
                let new_u = top_row[lane] + v * twiddle;
                top_row[lane] = new_u;
                bot_row[lane] = v + new_u;
            }
        });
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
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    let stride = quarter * num_ntts;
    debug_assert_eq!(block.len(), 4 * stride);

    let do_one =
        |row_a: &mut [F128], row_b: &mut [F128], row_c: &mut [F128], row_d: &mut [F128]| {
            for lane in 0..num_ntts {
                let mut a = row_a[lane];
                let mut b = row_b[lane];
                let mut c = row_c[lane];
                let mut d = row_d[lane];
                // Layer L (matches `butterfly_interleaved_block`'s formula:
                // new_u = u + v*twiddle; new_v = v + new_u).
                let new_a = a + c * t_outer;
                c += new_a;
                a = new_a;
                let new_b = b + d * t_outer;
                d += new_b;
                b = new_b;
                // Layer L+1: (a, b) with t_inner_a (top sub-block);
                //            (c, d) with t_inner_b (bottom sub-block).
                let new_a2 = a + b * t_inner_a;
                b += new_a2;
                a = new_a2;
                let new_c2 = c + d * t_inner_b;
                d += new_c2;
                c = new_c2;
                row_a[lane] = a;
                row_b[lane] = b;
                row_c[lane] = c;
                row_d[lane] = d;
            }
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
            do_one(q1r, q2r, q3r, q4r);
        }
    } else {
        q1.par_chunks_mut(num_ntts)
            .zip(q2.par_chunks_mut(num_ntts))
            .zip(q3.par_chunks_mut(num_ntts))
            .zip(q4.par_chunks_mut(num_ntts))
            .for_each(|(((row_a, row_b), row_c), row_d)| {
                do_one(row_a, row_b, row_c, row_d);
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
fn butterfly_interleaved_block(
    block: &mut [F128],
    twiddle: F128,
    block_size_half: usize,
    num_ntts: usize,
) {
    let off_bot = block_size_half * num_ntts;
    for r in 0..block_size_half {
        let off_top = r * num_ntts;
        let off_bot_r = off_top + off_bot;
        for lane in 0..num_ntts {
            let v = block[off_bot_r + lane];
            let new_u = block[off_top + lane] + v * twiddle;
            block[off_top + lane] = new_u;
            block[off_bot_r + lane] = v + new_u;
        }
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

// ---------------------------------------------------------------------------
// NEON butterfly helpers — batch 2 F128 butterflies per `ghash_mul_vec2_neon`.
// ---------------------------------------------------------------------------

/// Two butterflies within a single block (shared twiddle).
///
/// `chunk` is one block of length `2 * half`. Pairs (idx0, idx0 + half) for
/// idx0 = 0..half are butterflied. We process two consecutive idx0's at once
/// to share the twiddle across the `ghash_mul_vec2_neon` call.
///
/// Precondition: `half >= 2`. (At deepest layer where half=1, use
/// [`butterfly_across_blocks_neon`] instead.)
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn butterfly_block_neon(chunk: &mut [F128], twiddle: F128, half: usize) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;
    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        // SAFETY: aes target feature enabled.
        let prod = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };

        let new_u_a = F128 {
            lo: u_a.lo ^ prod[0].lo,
            hi: u_a.hi ^ prod[0].hi,
        };
        let new_u_b = F128 {
            lo: u_b.lo ^ prod[1].lo,
            hi: u_b.hi ^ prod[1].hi,
        };
        let new_v_a = F128 {
            lo: v_a.lo ^ new_u_a.lo,
            hi: v_a.hi ^ new_u_a.hi,
        };
        let new_v_b = F128 {
            lo: v_b.lo ^ new_u_b.lo,
            hi: v_b.hi ^ new_u_b.hi,
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

/// Two butterflies across 2 adjacent blocks at the deepest layer (each block
/// has just 1 pair, i.e., block_size_half = 1). The two pairs have DIFFERENT
/// twiddles.
///
/// Operates on `data[base..base+4]` = (block0_lo, block0_hi, block1_lo, block1_hi).
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn butterfly_across_blocks_neon(data: &mut [F128], base: usize, t_a: F128, t_b: F128) {
    // SAFETY: caller's `aes` target-feature attribute covers this call.
    unsafe { butterfly_across_blocks_neon_in_chunk(&mut data[base..base + 4], t_a, t_b) };
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn butterfly_across_blocks_neon_in_chunk(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;
    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    // SAFETY: aes target feature enabled.
    let prod = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };

    let new_u_a = F128 {
        lo: u_a.lo ^ prod[0].lo,
        hi: u_a.hi ^ prod[0].hi,
    };
    let new_u_b = F128 {
        lo: u_b.lo ^ prod[1].lo,
        hi: u_b.hi ^ prod[1].hi,
    };
    let new_v_a = F128 {
        lo: v_a.lo ^ new_u_a.lo,
        hi: v_a.hi ^ new_u_a.hi,
    };
    let new_v_b = F128 {
        lo: v_b.lo ^ new_u_b.lo,
        hi: v_b.hi ^ new_u_b.hi,
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

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

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
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

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn interleaved_x86_matches_scalar() {
        // The AVX-512 interleaved transform must agree with the scalar reference.
        // Covers num_ntts=1 (across-element flattening), the vectorized path
        // (num_ntts≥4), and small log_d where the scalar tail (half_elems<4) runs.
        let mut rng = Rng::new(0xCC3);
        for log_d in [1usize, 4, 10, 14, 17] {
            for &num_ntts in &[1usize, 2, 4, 8, 32] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n_total = (1 << log_d) * num_ntts;
                let original = rand_vec(&mut rng, n_total);
                let mut v_scalar = original.clone();
                ntt.forward_transform_interleaved_scalar(&mut v_scalar, num_ntts);
                let mut v_x86 = original.clone();
                ntt.forward_transform_interleaved_x86_from_layer(&mut v_x86, num_ntts, 0);
                assert_eq!(
                    v_x86, v_scalar,
                    "interleaved x86 mismatch at log_d={log_d}, num_ntts={num_ntts}"
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
