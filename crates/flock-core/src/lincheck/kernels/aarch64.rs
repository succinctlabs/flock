use super::super::{F128, build_sum_table};

const NEON_TILE_T: usize = 8;

/// Single-matrix partial fold with **tiled + NEON-register accumulators**.
/// Keeps `BLOCK_K = 8` accumulators in NEON registers across a `NEON_TILE_T`
/// stripe sweep — no per-byte accumulator LD/ST. Hand-rolled aarch64
/// intrinsics force the F128 XOR to a single `EOR.16B` and pin the 8 accs
/// in Q registers.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_single(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let k = 1usize << k_log;
    partial_fold_packed_z_neon_single_padded(z_packed, m, k_log, k, eq_outer)
}

/// Padding-aware variant of [`partial_fold_packed_z_neon_single`]. Rounds
/// `useful_bits` up to a multiple of `BLOCK_K = 8` and processes only the
/// covered blocks; the trailing blocks (entirely padding) stay zero in the
/// accumulator. Any partially-useful boundary block is processed in full —
/// its padding bytes are zero, table[0] = 0, so they contribute nothing.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_single_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;
    use std::arch::aarch64::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;
    let n_blocks_full = k / BLOCK_K;
    // Cover only the blocks that touch useful bits. The boundary block
    // contains padding bytes which are 0 — table[0] = 0 → they contribute
    // nothing to the per-block XOR chain.
    let n_blocks = useful_bits.div_ceil(BLOCK_K).min(n_blocks_full);

    let tiles_per_chunk = (n_tiles / 256).max(1);
    let bytes_per_chunk = tiles_per_chunk * TILE_T * k;

    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut out, (chunk_idx, chunk_bytes)| {
                let tile_start = chunk_idx * tiles_per_chunk;
                // TILE_T × 256 F128 = 32 KB tables. L1 resident.
                let mut tables = vec![F128::ZERO; TILE_T * 256];

                let n_tiles_in_chunk = chunk_bytes.len() / (TILE_T * k);
                for tile_rel in 0..n_tiles_in_chunk {
                    let tile_idx = tile_start + tile_rel;
                    let stripe_base = tile_idx * TILE_T;
                    let tile_bytes_ptr = unsafe { chunk_bytes.as_ptr().add(tile_rel * TILE_T * k) };

                    for t in 0..TILE_T {
                        let byte_idx = stripe_base + t;
                        let eq_off = 8 * byte_idx;
                        build_sum_table(
                            &eq_outer[eq_off..eq_off + 8],
                            &mut tables[t * 256..(t + 1) * 256],
                        );
                    }

                    let tables_ptr = tables.as_ptr() as *const u8;

                    for block_idx in 0..n_blocks {
                        let bs = block_idx * BLOCK_K;
                        unsafe {
                            process_block_neon_single(
                                tile_bytes_ptr,
                                k,
                                bs,
                                tables_ptr,
                                out.as_mut_ptr().add(bs),
                            );
                        }
                    }
                }
                // Suppress unused variable warning when not aarch64
                let _ = unsafe { vdupq_n_u8(0) };
                out
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Single-matrix NEON inner kernel — sweep TILE_T=8 stripes of a stripe-tile
/// for one BLOCK_K=8 block of i_inner positions, keeping all 8 accumulators
/// in NEON Q-registers.
///
/// # Safety
/// - `tile_bytes_ptr` must point to at least `TILE_T * k` bytes.
/// - `tables_ptr` must point to at least `TILE_T * 256 * 16` bytes.
/// - `out_ptr` must point to at least 8 F128 (128 bytes) of mutable storage.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn process_block_neon_single(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const u8,
    out_ptr: *mut F128,
) {
    use std::arch::aarch64::*;
    const TILE_T: usize = NEON_TILE_T;

    let o = out_ptr as *mut u8;

    let mut a0 = vld1q_u8(o);
    let mut a1 = vld1q_u8(o.add(16));
    let mut a2 = vld1q_u8(o.add(32));
    let mut a3 = vld1q_u8(o.add(48));
    let mut a4 = vld1q_u8(o.add(64));
    let mut a5 = vld1q_u8(o.add(80));
    let mut a6 = vld1q_u8(o.add(96));
    let mut a7 = vld1q_u8(o.add(112));

    for t in 0..TILE_T {
        let stripe_ptr = tile_bytes_ptr.add(t * k + bs);
        let ta = tables_ptr.add(t * 256 * 16);

        let i0 = *stripe_ptr as usize;
        let i1 = *stripe_ptr.add(1) as usize;
        let i2 = *stripe_ptr.add(2) as usize;
        let i3 = *stripe_ptr.add(3) as usize;
        let i4 = *stripe_ptr.add(4) as usize;
        let i5 = *stripe_ptr.add(5) as usize;
        let i6 = *stripe_ptr.add(6) as usize;
        let i7 = *stripe_ptr.add(7) as usize;

        a0 = veorq_u8(a0, vld1q_u8(ta.add(i0 * 16)));
        a1 = veorq_u8(a1, vld1q_u8(ta.add(i1 * 16)));
        a2 = veorq_u8(a2, vld1q_u8(ta.add(i2 * 16)));
        a3 = veorq_u8(a3, vld1q_u8(ta.add(i3 * 16)));
        a4 = veorq_u8(a4, vld1q_u8(ta.add(i4 * 16)));
        a5 = veorq_u8(a5, vld1q_u8(ta.add(i5 * 16)));
        a6 = veorq_u8(a6, vld1q_u8(ta.add(i6 * 16)));
        a7 = veorq_u8(a7, vld1q_u8(ta.add(i7 * 16)));
    }

    vst1q_u8(o, a0);
    vst1q_u8(o.add(16), a1);
    vst1q_u8(o.add(32), a2);
    vst1q_u8(o.add(48), a3);
    vst1q_u8(o.add(64), a4);
    vst1q_u8(o.add(80), a5);
    vst1q_u8(o.add(96), a6);
    vst1q_u8(o.add(112), a7);
}

/// **i_inner-partitioned** NEON partial fold. Same result as
/// [`partial_fold_packed_z_neon_single_padded`] but parallelizes over the
/// **output** (`i_inner`) instead of over z stripes.
///
/// Why: the stripe-parallel kernel gives every worker its own full length-`k`
/// accumulator (2 MB at k = 2¹⁷). With P workers that's `P · 2 MB` of live
/// accumulators — past ~3 workers it exceeds L2, so each worker's accumulator
/// spills and gets re-streamed from **main memory** once per stripe-tile
/// (≈ `n_tiles · 2·k` F128 of memory traffic). Measured: scaling saturates at
/// ~5× on 10 cores (memory-bound), not ~10×.
///
/// Here the workers own **disjoint** slices of a single shared `out`, so the
/// total live accumulator is just `k` F128 = 2 MB — it stays L2-resident, never
/// re-streamed from memory, and there is **no final reduction**. Main-memory
/// traffic drops to one pass over z plus one write of `out`. Each worker still
/// uses the register-tiled inner kernel (8 accumulators across `TILE_T`
/// stripes); it just rebuilds the per-tile sum tables for its own slice (a few
/// % of redundant table-build XORs, far cheaper than the memory re-streaming).
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_iblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;

    // Only i_inner < useful_bits can be nonzero (padded rows fold to 0). Round
    // up to BLOCK_K; the boundary block's padding bytes are 0 ⇒ table[0] = 0 ⇒
    // contribute nothing. Rows [useful, k) stay zero from the vec init.
    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);

    let mut out = vec![F128::ZERO; k];
    if useful == 0 {
        return out;
    }

    // Partition the useful i_inner range across workers. Each chunk independently
    // rebuilds the per-tile sum tables, so chunk count drives redundant table
    // work — work that does NOT scale with cores and dominates the residual at
    // m=30 (≈3.3 ms/core at 3 chunks/worker). On the homogeneous pinned P-core
    // pool, 1 chunk/worker is perfectly balanced (par_chunks_mut → exactly `p`
    // equal chunks) and cuts that residual ~3×: partial-fold MT 6.2 → 4.5 ms,
    // no ST change. Oversubscribe (3/worker) only when the pool is larger than
    // the P-core count — i.e. likely includes slower E-cores — so rayon can
    // steal from a straggler. Each chunk is a BLOCK_K multiple.
    let p = rayon::current_num_threads().max(1);
    let chunks_per_worker = if p <= crate::perf_core_count_cached() {
        1
    } else {
        3
    };
    let i_chunk = (useful / (p * chunks_per_worker))
        .max(BLOCK_K)
        .next_multiple_of(BLOCK_K);

    out[..useful]
        .par_chunks_mut(i_chunk)
        .enumerate()
        .for_each(|(ci, out_slice)| {
            let i_base = ci * i_chunk;
            let n_block = out_slice.len() / BLOCK_K;
            // TILE_T × 256 F128 = 32 KB tables, L1-resident, rebuilt per tile.
            let mut tables = vec![F128::ZERO; TILE_T * 256];
            for tile in 0..n_tiles {
                let stripe_base = tile * TILE_T;
                for t in 0..TILE_T {
                    let eq_off = 8 * (stripe_base + t);
                    build_sum_table(
                        &eq_outer[eq_off..eq_off + 8],
                        &mut tables[t * 256..(t + 1) * 256],
                    );
                }
                let tables_ptr = tables.as_ptr() as *const u8;
                // Base of this (tile, i_base): process_block reads
                // z_base[t·k + bs] = z[(stripe_base+t)·k + i_base + bs].
                let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k + i_base) };
                for b in 0..n_block {
                    let i = b * BLOCK_K;
                    unsafe {
                        process_block_neon_single(
                            z_base,
                            k,
                            i,
                            tables_ptr,
                            out_slice.as_mut_ptr().add(i),
                        );
                    }
                }
            }
        });
    out
}

/// Outer(tile)-partitioned sibling of [`partial_fold_packed_z_neon_iblock_padded`]
/// — same result, parallelized to remove the redundant per-worker sum-table
/// rebuilds that cap iblock's multicore scaling. **This is the default fold**
/// (`partial_fold_packed_z_best`); set [`FOLD_IBLOCK`] to fall back to iblock.
///
/// iblock partitions the length-k **output** across workers, so every worker
/// rebuilds **all** `n_stripes` tile tables — table work is done `p`× and does not
/// shrink with cores (≈44 % of the MT wall at m=32). Here we partition the **tiles**
/// (outer/stripe dim): each worker owns a contiguous tile band, builds each of its
/// tile tables exactly **once**, folds them into a private length-k partial, and the
/// `p` partials are XOR-reduced at the end. The partial is the full length-k
/// (256 KB at k_log=14 ⇒ spills L1 to L2), but the register-tiled inner kernel keeps
/// 8 F128 accumulators in NEON registers, so the L2 traffic is mild — measured ≈2 %
/// ST cost at m=32, none at m=30 — and far cheaper than iblock's redundant tables:
/// the fold scales ~8.5× vs iblock's ~6.5× on 10 P-cores at m=32, and the margin
/// grows with the outer dim (the redundant-table cost it removes is ∝ `n_stripes`).
///
/// # Safety / preconditions: identical to the iblock kernel.
#[cfg(target_arch = "aarch64")]
pub fn partial_fold_packed_z_neon_oblock_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = NEON_TILE_T;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(
        n_log >= 3 + TILE_T.trailing_zeros() as usize,
        "need n_outer ≥ 8·TILE_T stripes"
    );
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_tiles = n_stripes / TILE_T;

    // Only i_inner < useful_bits can be nonzero (padded rows fold to 0). Rounded
    // up to BLOCK_K; columns [useful, k) stay zero from the partial init.
    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);
    if useful == 0 {
        return vec![F128::ZERO; k];
    }

    // One private length-k partial per worker; workers own contiguous tile bands,
    // so each tile's sum-tables are built exactly once (not once per worker).
    let p = rayon::current_num_threads().max(1);
    let tiles_per_worker = n_tiles.div_ceil(p);
    let n_workers = n_tiles.div_ceil(tiles_per_worker); // ≤ p, every band non-empty

    let mut partials = vec![F128::ZERO; n_workers * k];
    partials
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(w, partial)| {
            let tile_lo = w * tiles_per_worker;
            let tile_hi = ((w + 1) * tiles_per_worker).min(n_tiles);
            // TILE_T × 256 F128 = 32 KB tables, L1-resident, built once per tile.
            let mut tables = vec![F128::ZERO; TILE_T * 256];
            for tile in tile_lo..tile_hi {
                let stripe_base = tile * TILE_T;
                for t in 0..TILE_T {
                    let eq_off = 8 * (stripe_base + t);
                    build_sum_table(
                        &eq_outer[eq_off..eq_off + 8],
                        &mut tables[t * 256..(t + 1) * 256],
                    );
                }
                let tables_ptr = tables.as_ptr() as *const u8;
                let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k) };
                let mut bs = 0usize;
                while bs < useful {
                    unsafe {
                        process_block_neon_single(
                            z_base,
                            k,
                            bs,
                            tables_ptr,
                            partial.as_mut_ptr().add(bs),
                        );
                    }
                    bs += BLOCK_K;
                }
            }
        });

    // XOR-reduce the per-worker partials: parallel over columns, sequential over
    // workers so each 256 KB partial is streamed once (cache-friendly).
    let (first, rest) = partials.split_at(k);
    let mut out = first.to_vec();
    for chunk in rest.chunks(k) {
        out.par_iter_mut()
            .zip(chunk.par_iter())
            .for_each(|(o, s)| *o += *s);
    }
    out
}
