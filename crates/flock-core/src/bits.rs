//! Small bit-manipulation primitives shared across modules.

use crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes;

/// Lowest set bit of `x`, or 0 if `x == 0` — i.e. `usize::isolate_lowest_one`.
///
/// Written out rather than calling the standard-library method on purpose.
/// `isolate_lowest_one` only stabilized in Rust **1.97.0**, and the crate's
/// `edition = "2024"` otherwise builds on anything from 1.85, so using it would
/// silently raise the minimum toolchain by twelve releases for a two-instruction
/// idiom. (It previously sat behind `#![feature(...)]`, which made the tree
/// nightly-only, then a hard error on stable once the feature landed.) Do not
/// "simplify" this back — clippy 1.98's `manual_isolate_lowest_one` tries to,
/// and the Blackwell CI runner's older toolchain cannot compile the result.
#[inline(always)]
#[allow(clippy::manual_isolate_lowest_one)]
pub fn lowest_one(x: usize) -> usize {
    x & x.wrapping_neg()
}

/// Hacker's Delight (Sec. 7-3) 8×8 bit-matrix transpose stored in a `u64`.
///
/// The input holds 8 bytes representing 8 rows of 8 bits each; the output holds
/// the transposed matrix (bit `r·8 + c` of input → bit `c·8 + r` of output).
///
/// Shared by the lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`)
/// and the PCS ring-switch `fold_1b` kernels ([`crate::pcs::ring_switch`]).
#[inline(always)]
pub(crate) fn transpose_8x8_bits(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AAu64;
    x = x ^ t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCCu64;
    x = x ^ t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0u64;
    x = x ^ t ^ (t << 28);
    x
}

/// Bit-transpose 8 little-endian `u64` lanes (the 64-byte block they form) into
/// a 64-byte output stripe.
///
/// The 8 LE u64s viewed as 64 bytes are exactly the input shape of the NEON
/// [`bit_transpose_64bytes`] kernel (input byte `r·8 + c` = byte `c` of lane
/// `r`; output byte `c·8 + t` bit `r` = that byte's bit `t`), so this delegates
/// to it — ~5× fewer ops than the scalar per-column loop. Shared by the
/// lincheck byte-stripe builder (`flock_prover::r1cs_hashes::common`) and the
/// core R1CS matrix-apply ([`crate::r1cs`]).
///
/// [`bit_transpose_64bytes`]: crate::zerocheck::univariate_skip_optimized::bit_transpose_64bytes
#[inline(always)]
pub fn transpose_8_u64s_to_64_bytes(lanes: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    // SAFETY: [u64; 8] is 64 bytes with no padding; u8 has weaker alignment.
    let input: &[u8; 64] = unsafe { &*(lanes.as_ptr() as *const [u8; 64]) };
    let out64: &mut [u8; 64] = out.try_into().expect("64-byte stripe slice");
    bit_transpose_64bytes(input, out64);
}

#[cfg(test)]
mod tests {
    use std::array::from_fn;

    use crate::bits::{lowest_one, transpose_8_u64s_to_64_bytes, transpose_8x8_bits};

    /// [`lowest_one`] against an independent `trailing_zeros` reference. Not
    /// against `usize::isolate_lowest_one`: the whole point of the helper is
    /// that the tree must build without that method, tests included.
    #[test]
    fn lowest_one_matches_trailing_zeros() {
        assert_eq!(lowest_one(0), 0, "no set bit to isolate");
        // The four call sites all work over 1..256; check that exhaustively,
        // then the bit-width edges.
        for x in 1usize..1024 {
            assert_eq!(lowest_one(x), 1usize << x.trailing_zeros(), "x={x}");
        }
        for b in 0..usize::BITS {
            let bit = 1usize << b;
            assert_eq!(lowest_one(bit), bit, "single bit {b}");
            // Higher bits set alongside must not change the answer.
            assert_eq!(lowest_one(usize::MAX << b), bit, "MAX<<{b}");
        }
    }

    /// Scalar reference for [`transpose_8_u64s_to_64_bytes`] — test oracle only.
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn transpose_8_u64s_to_64_bytes_scalar(lanes: &[u64; 8], out: &mut [u8]) {
        debug_assert_eq!(out.len(), 64);
        for c in 0..8 {
            let shift = c * 8;
            let mut packed: u64 = 0;
            packed |= ((lanes[0] >> shift) & 0xFF) << (0 * 8);
            packed |= ((lanes[1] >> shift) & 0xFF) << (1 * 8);
            packed |= ((lanes[2] >> shift) & 0xFF) << (2 * 8);
            packed |= ((lanes[3] >> shift) & 0xFF) << (3 * 8);
            packed |= ((lanes[4] >> shift) & 0xFF) << (4 * 8);
            packed |= ((lanes[5] >> shift) & 0xFF) << (5 * 8);
            packed |= ((lanes[6] >> shift) & 0xFF) << (6 * 8);
            packed |= ((lanes[7] >> shift) & 0xFF) << (7 * 8);
            let transposed = transpose_8x8_bits(packed);
            out[c * 8..c * 8 + 8].copy_from_slice(&transposed.to_le_bytes());
        }
    }

    /// The NEON-delegating transpose must match the scalar per-column oracle
    /// bit-for-bit on varied inputs.
    #[test]
    fn transpose_8_u64s_matches_scalar() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        for _ in 0..100 {
            let lanes: [u64; 8] = from_fn(|_| next());
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle);
        }
        // Edge patterns.
        for lanes in [[0u64; 8], [u64::MAX; 8], from_fn(|i| 1u64 << i)] {
            let mut fast = [0u8; 64];
            let mut oracle = [0u8; 64];
            transpose_8_u64s_to_64_bytes(&lanes, &mut fast);
            transpose_8_u64s_to_64_bytes_scalar(&lanes, &mut oracle);
            assert_eq!(fast, oracle, "lanes={lanes:?}");
        }
    }

    /// Transposing twice is the identity.
    #[test]
    fn transpose_is_involution() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(31);
            assert_eq!(transpose_8x8_bits(transpose_8x8_bits(state)), state);
        }
    }

    /// Cross-check against a naive bit-by-bit transpose of the 8×8 matrix.
    #[test]
    fn matches_naive() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..256 {
            state = state.wrapping_mul(0x2545_F491_4F6C_DD1D).rotate_left(17);
            let got = transpose_8x8_bits(state);
            let mut want = 0u64;
            for r in 0..8 {
                for c in 0..8 {
                    if (state >> (r * 8 + c)) & 1 == 1 {
                        want |= 1u64 << (c * 8 + r);
                    }
                }
            }
            assert_eq!(got, want, "input={state:016x}");
        }
    }
}
