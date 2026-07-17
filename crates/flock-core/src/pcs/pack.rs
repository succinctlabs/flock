//! Bit-witness packing into F_{2^128} for the PCS commitment phase.
//!
//! The witness `z : {0,1}^m → {0,1}` is laid out as a flat 2^m-length bool
//! array. Packing groups the **first** `LOG_PACKING = 7` boolean coordinates
//! into one F_{2^128} element, leaving an array of 2^(m−7) packed elements
//! indexed by the remaining m−7 outer coords.
//!
//! Layout convention: for packed index `i_rest ∈ {0..2^(m−7)}` and bit position
//! `i_skip ∈ {0..128}`,
//! ```text
//!     bit i_skip of out[i_rest]  ==  z[i_rest * 128 + i_skip]
//! ```
//! where "bit i_skip of an F_{2^128} element" means the i_skip-th coordinate of
//! its natural polynomial basis decomposition (i.e. the i_skip-th bit of the
//! u128 representation, little-endian).
//!
//! This matches the convention used in the [DP24] ring-switching reduction:
//! `s_hat_v[i_skip] = ẑ_{i_skip}(x_mlv)`, the multilinear extension of the
//! `i_skip`-th bit-slice of the witness.
//!
//! [DP24]: https://eprint.iacr.org/2024/504

use crate::field::F128;

/// `log_2` of the packing width. F_{2^128} holds 128 bits = 2^7.
pub const LOG_PACKING: usize = 7;

/// Packing width (number of bits per F_{2^128} element).
pub const PACKING_WIDTH: usize = 1 << LOG_PACKING;

/// Pack a Boolean witness `z` of length `2^m` into `2^(m − LOG_PACKING)`
/// F_{2^128} elements.
///
/// See module docs for the layout convention.
///
/// # Panics
///
/// - if `z.len() != 1 << m`
/// - if `m < LOG_PACKING`
pub fn pack_witness(z: &[bool], m: usize) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(z.len(), 1usize << m, "z length must be 2^m");
    assert!(
        m >= LOG_PACKING,
        "witness too small to pack: m = {m} < LOG_PACKING = {LOG_PACKING}",
    );
    let n_packed = 1usize << (m - LOG_PACKING);

    // `bool` is guaranteed 1 byte holding 0x00/0x01, so 8 bools read as one
    // little-endian u64 pack to an LSB-first byte with one multiply:
    // byte 7 of `x * 0x0102040810204080` is Σ_r b_r·2^r (each lower product
    // byte sums distinct powers of two ≤ 0xFE — no carry into byte 7).
    // SAFETY: same length, and any &[bool] is a valid &[u8].
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(z.as_ptr() as *const u8, z.len()) };
    #[inline]
    fn pack64(b: &[u8]) -> u64 {
        let mut w = 0u64;
        for (i, ch) in b.as_chunks::<8>().0.iter().enumerate() {
            let x = u64::from_le_bytes(*ch);
            w |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * i);
        }
        w
    }
    let one = |i_rest: usize| {
        let base = i_rest << LOG_PACKING;
        F128 {
            lo: pack64(&bytes[base..base + 64]),
            hi: pack64(&bytes[base + 64..base + 128]),
        }
    };
    // Parallel for real witnesses; sequential below the dispatch-overhead
    // floor (tiny test instances).
    if n_packed >= (1 << 12) {
        (0..n_packed).into_par_iter().map(one).collect()
    } else {
        (0..n_packed).map(one).collect()
    }
}

/// Inverse of [`pack_witness`]: unpack F_{2^128} elements back to a Boolean
/// witness of length `2^m`.
///
/// Round-trips with [`pack_witness`] by construction.
pub fn unpack_witness(packed: &[F128], m: usize) -> Vec<bool> {
    let n_packed = 1usize << (m - LOG_PACKING);
    assert_eq!(
        packed.len(),
        n_packed,
        "packed length must be 2^(m - LOG_PACKING)"
    );
    let mut out = vec![false; 1usize << m];
    for (i_rest, elem) in packed.iter().enumerate() {
        let base = i_rest << LOG_PACKING;
        for r in 0..64 {
            out[base | r] = (elem.lo >> r) & 1 == 1;
        }
        for r in 0..64 {
            out[base | 64 | r] = (elem.hi >> r) & 1 == 1;
        }
    }
    out
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
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let mut rng = Rng::new(0xC0FFEE);
        for m in [7usize, 8, 10, 12, 14] {
            let z = rng.bits(1 << m);
            let packed = pack_witness(&z, m);
            assert_eq!(packed.len(), 1 << (m - LOG_PACKING));
            let z_back = unpack_witness(&packed, m);
            assert_eq!(z, z_back, "roundtrip failed at m={m}");
        }
    }

    #[test]
    fn pack_layout_matches_natural_bit_order() {
        // For m = LOG_PACKING (= 7): exactly one packed element, holding the
        // entire 128-bit witness in natural u128 bit order.
        let mut z = vec![false; 128];
        // Set a known bit pattern: bits at positions 0, 1, 5, 63, 64, 127.
        for &i in &[0usize, 1, 5, 63, 64, 127] {
            z[i] = true;
        }
        let packed = pack_witness(&z, LOG_PACKING);
        assert_eq!(packed.len(), 1);
        let expected = F128 {
            lo: (1u64 << 0) | (1u64 << 1) | (1u64 << 5) | (1u64 << 63),
            hi: (1u64 << 0) | (1u64 << 63),
        };
        assert_eq!(packed[0], expected);
    }

    #[test]
    fn pack_independent_chunks() {
        // Two adjacent 128-bit chunks should pack independently — flipping a
        // bit in one chunk affects only that chunk.
        let z = vec![true; 256];
        let packed = pack_witness(&z, 8);
        assert_eq!(packed.len(), 2);
        assert_eq!(
            packed[0],
            F128 {
                lo: u64::MAX,
                hi: u64::MAX
            }
        );
        assert_eq!(
            packed[1],
            F128 {
                lo: u64::MAX,
                hi: u64::MAX
            }
        );
    }

    #[test]
    #[should_panic(expected = "witness too small")]
    fn rejects_undersized_witness() {
        let z = vec![false; 64]; // m = 6 < LOG_PACKING = 7
        let _ = pack_witness(&z, 6);
    }
}
