//! §2.1 single-table collapse of the LDE matrix `M = fwd_NTT_V₈ ∘ inv_NTT_S`,
//! for the **mismatched-k** case used by degree-4 zerocheck round-1.
//!
//! Companion to [`super::inv_table::InvNttTableByteSingleGf8`] (the same-k
//! variant for the degree-2 path). The §2.1 collapse trick generalizes
//! whenever the output domain is **closed under XOR with the input domain**:
//!
//!   `M[i', 8b + t]  =  T_0[bit-t-mask(8b+t)][i' ⊕ 8b]`
//!
//! For deg-4 the input domain S has `|S| = 64 = 2^6` and the output domain
//! V₈ has `|V₈| = 256 = 2^8 = |F₈|`. Since `V₈ ⊃ S` and V₈ is a vector
//! subspace, V₈ is closed under XOR with S — so the same XOR-shift relation
//! holds for the wider output. (Same proof: LCH NTT preserves F₂-affine
//! structure; `(j + d)`-translation in input becomes `(i + d)`-translation in
//! output for any `d ∈ S`.)
//!
//! ## Storage
//!
//! 256 rows × 256 bytes/row = **64 KB**. Just barely L1-resident on M-series
//! P-cores (128 KB L1d), or one half. Computed once via the constructor;
//! cached for the session.
//!
//! ## Apply
//!
//! Input: 8 bytes `bytes[0..8]` (one per b_chunk; the 64-bit S row).
//! Output: 256 F₈ bytes — evaluations on V₈ in canonical order.
//! The first 64 (positions on S) reproduce the original input bits; the last
//! 192 (positions on Λ₄ = V₈ \ S) are the fresh extension that the zerocheck
//! round-1 message uses.

use crate::field::F8;
use crate::ntt::AdditiveNttGf8;

/// `M = fwd_NTT_V₈ ∘ inv_NTT_S` collapsed into a single 256×256-byte table.
///
/// Construct via [`new`]; apply via [`apply`]. Cheap to clone (Arc-wraps
/// the underlying data via Vec, so cloning is O(1) by Vec semantics).
#[derive(Clone, Debug)]
pub struct InvNttTableSToV8Gf8 {
    /// log₂ of the input domain size (= 6 for the deg-4 zerocheck).
    pub k_in: usize,
    /// log₂ of the output domain size (= 8 for the deg-4 zerocheck).
    pub k_out: usize,
    /// |S| = 2^k_in.
    pub ell_in: usize,
    /// |V₈| = 2^k_out.
    pub ell_out: usize,
    /// Number of input bytes per row (= ell_in / 8 = 8).
    pub n_chunks: usize,
    /// `data[w * ell_out .. (w+1) * ell_out]` = T_0[w], the XOR-sum of the
    /// columns of `M` indexed by the set bits of `w` (taken from cols[0..8]).
    data: Vec<F8>,
}

impl InvNttTableSToV8Gf8 {
    /// Build the table given the input NTT `ntt_s` (k=`k_in`) and the output
    /// NTT `ntt_v8` (k=`k_out`).
    ///
    /// Requires `k_in <= k_out` and `k_in >= 3` (so n_chunks ≥ 1 and the §2.1
    /// chunk-XOR encoding fits in a byte).
    pub fn new(ntt_s: &AdditiveNttGf8, ntt_v8: &AdditiveNttGf8) -> Self {
        let k_in = ntt_s.k();
        let k_out = ntt_v8.k();
        assert!(k_in <= k_out, "input k must be ≤ output k");
        assert!(k_in >= 3, "k_in must be ≥ 3 so n_chunks ≥ 1");
        let ell_in = 1usize << k_in;
        let ell_out = 1usize << k_out;
        let n_chunks = ell_in / 8;
        assert!(
            n_chunks <= 16,
            "n_chunks must fit the i'/chunk XOR encoding"
        );

        let mut data = vec![F8::ZERO; 256 * ell_out];

        // Compute 8 unit-column images cols[t] = fwd_NTT_V₈ ∘ inv_NTT_S (e_t)
        // for t ∈ 0..8. Each col has length ell_out.
        // Procedure per t:
        //   - Build length-ell_in vector with 1 at position t.
        //   - inv_NTT_S → ell_in coefficients in the k_in-dim novel basis.
        //   - Zero-pad to length ell_out (extend coefficients to the k_out
        //     novel basis, which extends the k_in basis as a prefix).
        //   - fwd_NTT_V₈ → ell_out evaluations on V₈.
        let mut tmp_in = vec![F8::ZERO; ell_in];
        let mut tmp_out = vec![F8::ZERO; ell_out];
        let mut cols: Vec<Vec<F8>> = Vec::with_capacity(8);
        for t in 0..8 {
            tmp_in.iter_mut().for_each(|x| *x = F8::ZERO);
            tmp_in[t] = F8::ONE;
            ntt_s.inverse(&mut tmp_in);
            // Pad to ell_out by copying coefficients to a longer buffer (rest 0).
            tmp_out.iter_mut().for_each(|x| *x = F8::ZERO);
            tmp_out[..ell_in].copy_from_slice(&tmp_in);
            ntt_v8.forward(&mut tmp_out);
            cols.push(tmp_out.clone());
        }

        // T_0[0] = 0 (already zero).
        // T_0[2^t] = cols[t] for t = 0..8.
        // T_0[w] for non-pow-of-2 w: XOR of T_0[parent] and T_0[lo_bit], built
        // up by w in 3..256.
        for t in 0..8 {
            let entry_start = (1usize << t) * ell_out;
            data[entry_start..entry_start + ell_out].copy_from_slice(&cols[t]);
        }
        for w in 3usize..256 {
            if (w & (w - 1)) == 0 {
                continue; // skip powers of 2
            }
            let lo_bit = w & w.wrapping_neg();
            let parent = w ^ lo_bit;
            let (parent_off, bit_off, entry_off) =
                (parent * ell_out, lo_bit * ell_out, w * ell_out);
            for i in 0..ell_out {
                let v = data[parent_off + i] + data[bit_off + i];
                data[entry_off + i] = v;
            }
        }

        Self {
            k_in,
            k_out,
            ell_in,
            ell_out,
            n_chunks,
            data,
        }
    }

    /// Raw pointer to the table data — for NEON kernels that can't take a slice.
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr() as *const u8
    }

    /// Apply M to a single byte-packed S row.
    ///
    /// - `bytes`: length `n_chunks` (= 8 for the deg-4 case), each byte is
    ///   8 bits of the S input row in LSB-first order.
    /// - `out`: length `ell_out` (= 256 for the deg-4 case). Written entirely.
    ///   The first `ell_in` lanes reproduce the input bits on S; the rest are
    ///   fresh evaluations on V₈ \ S.
    ///
    /// The §2.1 collapse: for chunk index b ∈ 0..n_chunks, contribution is
    /// `T_0[bytes[b]]` permuted by the XOR-shift `i' → i' ⊕ 8b`.
    #[inline]
    pub fn apply_scalar(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell_out);
        out.iter_mut().for_each(|x| *x = F8::ZERO);
        for (b, &byte_b) in bytes.iter().enumerate() {
            let row_off = byte_b as usize * self.ell_out;
            let row = &self.data[row_off..row_off + self.ell_out];
            let shift = 8 * b;
            for i in 0..self.ell_out {
                out[i] += row[i ^ shift];
            }
        }
    }

    /// Dispatch helper — uses NEON/SSE2 when available, scalar otherwise.
    #[inline]
    pub fn apply(&self, bytes: &[u8], out: &mut [F8]) {
        #[cfg(target_arch = "aarch64")]
        if self.ell_out >= 16 {
            unsafe { self.apply_neon_unchecked(bytes, out) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.ell_out >= 16 {
            unsafe { self.apply_sse2(bytes, out) };
            return;
        }
        self.apply_scalar(bytes, out);
    }

    /// SSE2 variant — same algorithm as NEON, 128-bit load/store/XOR.
    #[cfg(target_arch = "x86_64")]
    unsafe fn apply_sse2(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::x86_64::*;
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell_out);
        let n128 = self.ell_out / 16;
        let base = self.data.as_ptr() as *const u8;
        let out_ptr = out.as_mut_ptr() as *mut u8;
        unsafe {
            let row0 = base.add(bytes[0] as usize * self.ell_out);
            for c in 0..n128 {
                let v = _mm_loadu_si128(row0.add(c * 16) as *const __m128i);
                _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, v);
            }
            for b in 1..self.n_chunks {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(bytes[b] as usize * self.ell_out);
                if b_odd {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = _mm_loadu_si128(row_b.add(sc * 16) as *const __m128i);
                        let v_swapped = _mm_shuffle_epi32(v, 0x4E);
                        let dst = out_ptr.add(c * 16) as *mut __m128i;
                        _mm_storeu_si128(dst, _mm_xor_si128(_mm_loadu_si128(dst), v_swapped));
                    }
                } else {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = _mm_loadu_si128(row_b.add(sc * 16) as *const __m128i);
                        let dst = out_ptr.add(c * 16) as *mut __m128i;
                        _mm_storeu_si128(dst, _mm_xor_si128(_mm_loadu_si128(dst), v));
                    }
                }
            }
        }
    }

    /// NEON variant of `apply` — operates in 16-byte chunks. Mirrors the
    /// same-k version in `inv_table.rs`. The chunk-XOR encoding `(b >> 1)`
    /// plus odd/even within-chunk half-swap implements `π_b(i') = i' ⊕ 8b`.
    ///
    /// # Safety
    /// Uses `core::arch::aarch64` NEON intrinsics; only call on `aarch64`.
    /// `bytes.len()` and `out.len()` must match the table shape (asserted).
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn apply_neon_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::aarch64::*;
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell_out);
        let n128 = self.ell_out / 16; // 16 for ell_out = 256
        let base = self.data.as_ptr() as *const u8;
        let out_ptr = out.as_mut_ptr() as *mut u8;

        unsafe {
            // b = 0: straight copy from row 0.
            let row0 = base.add(bytes[0] as usize * self.ell_out);
            for c in 0..n128 {
                vst1q_u8(out_ptr.add(c * 16), vld1q_u8(row0.add(c * 16)));
            }

            // b ≥ 1: XOR with table row[bytes[b]], permuted per (b >> 1, b & 1).
            for b in 1..self.n_chunks {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(bytes[b] as usize * self.ell_out);
                if b_odd {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = vld1q_u8(row_b.add(sc * 16));
                        let v_swapped = vextq_u8::<8>(v, v);
                        let dst = out_ptr.add(c * 16);
                        vst1q_u8(dst, veorq_u8(vld1q_u8(dst), v_swapped));
                    }
                } else {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = vld1q_u8(row_b.add(sc * 16));
                        let dst = out_ptr.add(c * 16);
                        vst1q_u8(dst, veorq_u8(vld1q_u8(dst), v));
                    }
                }
            }
        }
    }
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
    }

    /// Direct NTT path: same math as the table but without the table.
    /// inv_NTT_S on length-64, zero-pad to 256, fwd_NTT_V₈ on length-256.
    fn naive_extend(bytes: &[u8], ntt_s: &AdditiveNttGf8, ntt_v8: &AdditiveNttGf8) -> Vec<F8> {
        let ell_in = 1usize << ntt_s.k();
        let ell_out = 1usize << ntt_v8.k();
        let mut buf = vec![F8::ZERO; ell_out];
        for s in 0..ell_in {
            let bit = (bytes[s / 8] >> (s % 8)) & 1;
            buf[s] = F8(bit);
        }
        ntt_s.inverse(&mut buf[..ell_in]);
        // Coefficients at positions ell_in..ell_out are already zero (W_64..W_255).
        ntt_v8.forward(&mut buf);
        buf
    }

    /// Table.apply equals the direct NTT extension on random inputs.
    #[test]
    fn apply_matches_naive_random() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(8, F8::ZERO);
        let table = InvNttTableSToV8Gf8::new(&ntt_s, &ntt_v8);

        let mut rng = Rng::new(0xC0FFEE);
        for _trial in 0..16 {
            let bytes: [u8; 8] = [
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
            ];
            let naive = naive_extend(&bytes, &ntt_s, &ntt_v8);
            let mut got = vec![F8::ZERO; 256];
            table.apply(&bytes, &mut got);
            assert_eq!(got, naive, "table.apply ≠ direct NTT on bytes {bytes:?}");
        }
    }

    /// `apply` and `apply_scalar` agree (validates the NEON path against the
    /// scalar reference).
    #[test]
    fn apply_matches_apply_scalar() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(8, F8::ZERO);
        let table = InvNttTableSToV8Gf8::new(&ntt_s, &ntt_v8);

        let mut rng = Rng::new(0xBADCAFE);
        for _trial in 0..16 {
            let bytes: [u8; 8] = [
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
            ];
            let mut got_scalar = vec![F8::ZERO; 256];
            let mut got_dispatch = vec![F8::ZERO; 256];
            table.apply_scalar(&bytes, &mut got_scalar);
            table.apply(&bytes, &mut got_dispatch);
            assert_eq!(
                got_scalar, got_dispatch,
                "apply (NEON or scalar) ≠ apply_scalar on bytes {bytes:?}"
            );
        }
    }

    /// First 64 lanes of the output reproduce the input bits exactly.
    #[test]
    fn first_64_lanes_match_input_bits() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(8, F8::ZERO);
        let table = InvNttTableSToV8Gf8::new(&ntt_s, &ntt_v8);

        let mut rng = Rng::new(0xA17);
        for _trial in 0..8 {
            let bytes: [u8; 8] = [
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
            ];
            let mut got = vec![F8::ZERO; 256];
            table.apply(&bytes, &mut got);
            for s in 0..64 {
                let expected_bit = (bytes[s / 8] >> (s % 8)) & 1;
                assert_eq!(got[s], F8(expected_bit), "input bit mismatch at s={s}");
            }
        }
    }
}
