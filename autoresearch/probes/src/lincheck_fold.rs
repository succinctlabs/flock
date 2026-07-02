//! E3 — lincheck partial fold **directly from the L1′ witness**, no byte
//! stripe.
//!
//! Production computes `z_vec[i_inner] = Σ_o eq_outer[o] · z[i_inner, o]`
//! from a dedicated byte-stripe copy of z (built during witness gen), using
//! 256-entry subset-sum tables: one byte lookup + XOR per (8-outer-group,
//! i_inner).
//!
//! Under L1′ the stripe is unnecessary: for chunk-column `c`, the words of
//! all instances are contiguous, so a 64-instance tile read is one
//! sequential 1 KB load; the 8×64 bit-transposes production ran at
//! witness-gen time happen here in-register, and the same sum-table lookups
//! follow. Tiling 8 groups per accumulator touch amortizes the z_vec RMW
//! traffic 8× (the same trick as lincheck's NEON oblock kernel).
//!
//! Output is byte-identical to `lincheck::partial_fold_packed_z_fast_padded`
//! on the corresponding stripe (tested).

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;
use rayon::prelude::*;

/// Groups (of 8 instances) per accumulator sweep. 8 → 64-instance tiles:
/// 1 KB contiguous column reads, 8 × 4 KB sum tables (L1-resident), z_vec
/// touched once per 64 instances.
const TILE_G: usize = 8;

/// `table[b] = Σ_{r: bit r of b} eq8[r]` (256-entry subset-sum).
fn build_sum_table(eq8: &[F128], table: &mut [F128; 256]) {
    table[0] = F128::ZERO;
    for r in 0..8 {
        let w = eq8[r];
        let half = 1usize << r;
        for prev in 0..half {
            table[half | prev] = table[prev] + w;
        }
    }
}

/// Partial fold over the batch dims, reading the L1′ witness directly.
///
/// - `z_l1`: L1′ u64 buffer (`2^m` bits; word index `(c << n_log) | o`).
/// - `eq_outer`: `2^n_log` weights.
/// - `useful_bits`: per-instance useful prefix (rows ≥ useful fold to zero
///   and are skipped; matches `partial_fold_packed_z_fast_padded`).
///
/// Returns `z_vec` of length `2^k_log`.
pub fn partial_fold_l1(
    z_l1: &[u64],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n = 1usize << n_log;
    assert_eq!(z_l1.len(), (1usize << m) / 64);
    assert_eq!(eq_outer.len(), n);
    assert!(n >= 8, "need n_outer >= 8");
    assert!(useful_bits <= k);
    let useful_chunks = useful_bits.div_ceil(128);
    let n_groups = n / 8;
    let groups_per_tile = TILE_G.min(n_groups);
    let n_tiles = n_groups.div_ceil(groups_per_tile);

    // Parallel over tiles; per-worker k-sized accumulator, XOR-reduced (same
    // shape as lincheck's fast_padded).
    (0..n_tiles)
        .into_par_iter()
        .fold(
            || vec![F128::ZERO; k],
            |mut acc, tile| {
                let g0 = tile * groups_per_tile;
                let gs = groups_per_tile.min(n_groups - g0);
                // Sum tables for this tile's groups.
                let mut tables = vec![[F128::ZERO; 256]; gs];
                for (t, table) in tables.iter_mut().enumerate() {
                    build_sum_table(&eq_outer[8 * (g0 + t)..8 * (g0 + t) + 8], table);
                }
                let mut bytes = vec![[0u8; 128]; gs];
                for c in 0..useful_chunks {
                    // Contiguous column segment: instances 8·g0 .. 8·(g0+gs)
                    // at chunk c — gs·8 words = gs·128 bytes sequential.
                    let base = ((c << n_log) + 8 * g0) * 2;
                    for t in 0..gs {
                        // 8 instances' words: (lo, hi) u64 pairs.
                        let wb = base + t * 16;
                        let lo: [u64; 8] = std::array::from_fn(|j| z_l1[wb + 2 * j]);
                        let hi: [u64; 8] = std::array::from_fn(|j| z_l1[wb + 2 * j + 1]);
                        transpose_8_u64s_to_64_bytes(&lo, &mut bytes[t][..64]);
                        transpose_8_u64s_to_64_bytes(&hi, &mut bytes[t][64..]);
                    }
                    let out = &mut acc[c * 128..(c + 1) * 128];
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        // NEON: one 16-B table gather + EOR per (group, i);
                        // accumulator touched once per i. Matches the
                        // register discipline of lincheck's oblock kernel.
                        use core::arch::aarch64::*;
                        let out_ptr = out.as_mut_ptr() as *mut u8;
                        for i in 0..128 {
                            let mut s = vld1q_u8(
                                (tables[0].as_ptr() as *const u8)
                                    .add((bytes[0][i] as usize) * 16),
                            );
                            for t in 1..gs {
                                let e = vld1q_u8(
                                    (tables[t].as_ptr() as *const u8)
                                        .add((bytes[t][i] as usize) * 16),
                                );
                                s = veorq_u8(s, e);
                            }
                            let dst = out_ptr.add(i * 16);
                            vst1q_u8(dst, veorq_u8(vld1q_u8(dst), s));
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    for (i, o) in out.iter_mut().enumerate() {
                        let mut s = tables[0][bytes[0][i] as usize];
                        for t in 1..gs {
                            s += tables[t][bytes[t][i] as usize];
                        }
                        *o += s;
                    }
                }
                acc
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
