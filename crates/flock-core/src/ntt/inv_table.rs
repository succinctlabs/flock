//! §2.1 single-table collapse of the LDE matrix `M = fwd_NTT_Λ ∘ inv_NTT_S`.
//!
//! Background: the URM round-1 needs to map each `ell`-bit row of the boolean
//! witness (packed as `n_chunks = ell/8` bytes) to `ell` evaluations on the
//! NTT domain `Λ`. The naive way computes inv_NTT on S then fwd_NTT on Λ for
//! every row — too slow.
//!
//! The optimization (§2.1 of the paper): `M = α · M̃` with `M̃` Cauchy and `α`
//! a scalar. The columns of `M` satisfy a XOR-shift relation, so the `n_chunks`
//! per-byte sub-tables collapse to a single 256-row base table `T_0`:
//!
//!   M[i', 8b + t]  =  T_0[bit-t-mask(8b+t)][i' ⊕ 8b]
//!
//! Per-byte-chunk b contributes `π_b(T_0[byte_b])` to the output, where
//! `π_b(i') = i' ⊕ 8b`.
//!
//! Storage: 256 × ell bytes (16 KB at k=6, 32 KB at k=7) — fits in L1.
//! Lookups per row: n_chunks (= ell/8), each load is `ell` contiguous bytes.
//!
//! Scalar/correctness-first implementation; NEON `apply_triple` and the
//! unrolled `ntt_and_accum` can be added if the URM hot path needs them.

use crate::field::F8;
use crate::ntt::AdditiveNttGf8;

#[derive(Clone, Debug)]
pub struct InvNttTableByteSingleGf8 {
    pub k: usize,
    pub ell: usize,
    pub n_chunks: usize,
    /// Backing allocation for the table plus at most 63 bytes of alignment
    /// padding. The logical table starts at `data_offset`.
    data: Vec<F8>,
    /// Start of the logical table inside `data`, chosen so every 64-byte row
    /// in the production `ell = 64` table starts on a cache-line boundary.
    data_offset: usize,
}

impl InvNttTableByteSingleGf8 {
    /// Build the table given the two NTT instances: `ntt_S` over the input
    /// domain, `ntt_L` over the output (extension) domain. Both must have the
    /// same `k`.
    pub fn new(ntt_s: &AdditiveNttGf8, ntt_l: &AdditiveNttGf8) -> Self {
        assert_eq!(ntt_s.k(), ntt_l.k(), "ntt_S and ntt_L must share k");
        let k = ntt_s.k();
        let ell = 1usize << k;
        assert!(ell >= 8, "ell must be ≥ 8 so n_chunks ≥ 1");
        let n_chunks = ell / 8;
        assert!(
            n_chunks <= 16,
            "n_chunks must fit the i'/chunk XOR encoding"
        );

        // Vec<F8> only promises byte alignment. Over-allocate and select a
        // cache-line-aligned logical start so an AVX-512 row load never
        // straddles two cache lines.
        const TABLE_ALIGNMENT: usize = 64;
        let table_len = 256 * ell;
        let mut data = vec![F8::ZERO; table_len + TABLE_ALIGNMENT - 1];
        let data_offset = (TABLE_ALIGNMENT - (data.as_ptr() as usize & (TABLE_ALIGNMENT - 1)))
            & (TABLE_ALIGNMENT - 1);
        let table = &mut data[data_offset..data_offset + table_len];

        // Compute the 8 unit-column images cols[t] = fwd_NTT_Λ ∘ inv_NTT_S (e_t)
        // for t ∈ 0..8. The remaining columns of M are XOR-shifted versions.
        let mut tmp = vec![F8::ZERO; ell];
        let mut cols: Vec<Vec<F8>> = Vec::with_capacity(8);
        for t in 0..8 {
            tmp.iter_mut().for_each(|x| *x = F8::ZERO);
            tmp[t] = F8::ONE;
            ntt_s.inverse(&mut tmp);
            ntt_l.forward(&mut tmp);
            cols.push(tmp.clone());
        }

        // T_0[0] already zero. T_0[2^t] = cols[t]. Then for non-power-of-two w,
        // T_0[w] = T_0[w ^ lo_bit] ⊕ T_0[lo_bit]; this builds all 256 entries
        // with one XOR per entry.
        for t in 0..8 {
            let entry_start = (1usize << t) * ell;
            table[entry_start..entry_start + ell].copy_from_slice(&cols[t]);
        }
        for w in 3usize..256 {
            if (w & (w - 1)) == 0 {
                continue; // skip powers of 2 (already written)
            }
            let lo_bit = w.isolate_lowest_one();
            let parent = w ^ lo_bit;
            // Borrow-checker friendly: read parent + bit_v slices, then write entry.
            let (parent_off, bit_off, entry_off) = (parent * ell, lo_bit * ell, w * ell);
            for i in 0..ell {
                let v = table[parent_off + i] + table[bit_off + i];
                table[entry_off + i] = v;
            }
        }

        Self {
            k,
            ell,
            n_chunks,
            data,
            data_offset,
        }
    }

    /// Raw pointer to the table data (`256 × ell` bytes, row-major). Used by
    /// the URM fused inner kernel, which can't go through the safe slice API
    /// without losing the register-fused layout.
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        // SAFETY: `data_offset` selects a position within the over-allocated
        // buffer and the allocation cannot move while borrowed through self.
        unsafe { self.data.as_ptr().add(self.data_offset) as *const u8 }
    }

    #[inline]
    fn table(&self) -> &[F8] {
        &self.data[self.data_offset..self.data_offset + 256 * self.ell]
    }

    /// Apply M to a single byte-packed row, in place.
    /// `bytes` is `n_chunks` bytes (the LCH-coefficient bits of the row);
    /// `out` will be filled with the `ell` evaluations on Λ.
    ///
    /// Dispatches: NEON on aarch64 when `ell ≥ 16` (true for the protocol
    /// path k_skip=6 ⇒ ell=64), scalar otherwise.
    #[inline]
    pub fn apply(&self, bytes: &[u8], out: &mut [F8]) {
        #[cfg(target_arch = "aarch64")]
        if self.ell >= 16 {
            // SAFETY: aarch64 statically guarantees NEON; ell ≥ 16 ⇒ at least
            // one 128-bit chunk; method validates slice lengths.
            unsafe { self.apply_neon_unchecked(bytes, out) };
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        if self.ell == 64 {
            // SAFETY: feature-gated avx512f; ell == 64 as required.
            unsafe { self.apply_x86_avx512_unchecked(bytes, out) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.ell >= 16 {
            // SAFETY: x86_64 statically guarantees SSE2; ell ≥ 16 ⇒ at least
            // one 128-bit chunk; method validates slice lengths.
            unsafe { self.apply_x86_unchecked(bytes, out) };
            return;
        }
        self.apply_scalar(bytes, out);
    }

    /// Scalar reference. Kept public so tests can use it as the cross-check
    /// oracle for the NEON variant.
    pub fn apply_scalar(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        out.iter_mut().for_each(|x| *x = F8::ZERO);
        let table = self.table();
        for (b, &byte_b) in bytes.iter().enumerate() {
            let row_off = byte_b as usize * self.ell;
            let row = &table[row_off..row_off + self.ell];
            let shift = 8 * b;
            for i in 0..self.ell {
                out[i] += row[i ^ shift];
            }
        }
    }

    /// NEON variant of `apply` — operates in 16-byte chunks.
    ///
    /// For each output chunk `c ∈ 0..ell/16`:
    ///   * `b = 0`: straight 16-byte copy from `row0[c]`
    ///   * `b ≥ 1`: load `row_b[c ⊕ (b>>1)]`, half-swap if `b` is odd, XOR
    ///
    /// The `b>>1` chunk-XOR and the `8 · b` within-chunk shift together
    /// implement the `π_b(i') = i' ⊕ 8b` permutation that the §2.1 collapse
    /// requires.
    ///
    /// # Safety
    /// Caller must be on aarch64 (statically true at the dispatch site). The
    /// method validates slice lengths.
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn apply_neon_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::aarch64::*;
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        let n128 = self.ell / 16; // 4 for ell = 64
        let base = self.data_ptr();
        let out_ptr = out.as_mut_ptr() as *mut u8;

        unsafe {
            // b = 0: identity permutation — straight copy from row 0.
            let row0 = base.add(bytes[0] as usize * self.ell);
            for c in 0..n128 {
                vst1q_u8(out_ptr.add(c * 16), vld1q_u8(row0.add(c * 16)));
            }

            // b ≥ 1: XOR with table row[bytes[b]], permuted.
            for b in 1..self.n_chunks {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(bytes[b] as usize * self.ell);
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

    /// x86_64 variant of `apply` — operates in 16-byte chunks. Direct port of
    /// [`apply_neon_unchecked`]: SSE2 `_mm_loadu_si128` / `_mm_xor_si128` /
    /// `_mm_storeu_si128`, with the odd-`b` 8-byte half-swap (`vextq_u8::<8>`)
    /// realized as `_mm_shuffle_epi32::<0x4E>` (swap the two 64-bit halves).
    ///
    /// Replaces the scalar inner loop whose `row[i ^ shift]` gather defeats
    /// auto-vectorization — this is the single hottest function in the prover.
    ///
    /// # Safety
    /// Caller must be on x86_64 (statically true at the dispatch site). The
    /// method validates slice lengths.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    pub unsafe fn apply_x86_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::x86_64::*;
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        let n128 = self.ell / 16; // 4 for ell = 64
        let base = self.data_ptr();
        let out_ptr = out.as_mut_ptr() as *mut u8;

        unsafe {
            // b = 0: identity permutation — straight copy from row 0.
            let row0 = base.add(bytes[0] as usize * self.ell);
            for c in 0..n128 {
                let v = _mm_loadu_si128(row0.add(c * 16) as *const __m128i);
                _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, v);
            }

            // b ≥ 1: XOR with table row[bytes[b]], permuted.
            for b in 1..self.n_chunks {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(bytes[b] as usize * self.ell);
                if b_odd {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = _mm_loadu_si128(row_b.add(sc * 16) as *const __m128i);
                        // Swap the two 64-bit halves (NEON vextq_u8::<8>(v, v)).
                        let v_swapped = _mm_shuffle_epi32::<0x4E>(v);
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

    /// AVX-512 variant for the production `ell = 64` case: the whole output is
    /// one `__m512i` (the four 16-byte chunks of [`apply_x86_unchecked`]). The
    /// per-`b` chunk permutation `sc = c ^ (b>>1)` becomes a single
    /// `shuffle_i64x2` (128-bit-lane permute), the odd-`b` 64-bit half-swap a
    /// `shuffle_epi32::<0x4E>`, and the running XOR stays in a register —
    /// stored once at the end. 4× fewer load/store/XOR ops than the SSE path.
    ///
    /// # Safety
    /// Caller must be on x86_64 with `avx512f`, and `self.ell == 64`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn apply_x86_avx512_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        debug_assert_eq!(self.ell, 64, "avx512 apply is specialized for ell = 64");
        // SAFETY: `bytes` contains `n_chunks == 8` readable bytes and `out`
        // contains `ell == 64` writable bytes.
        unsafe {
            use core::arch::x86_64::*;
            let acc = self.apply_x86_avx512_register_unchecked(bytes.as_ptr());
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
        }
    }

    /// Register-returning AVX-512 inverse-NTT apply for the production
    /// `ell = 64`, `n_chunks = 8` case. This is the computation behind
    /// [`Self::apply_x86_avx512_unchecked`] without the final store, allowing
    /// fused consumers to keep the 64 F_8 outputs in a ZMM register.
    ///
    /// # Safety
    /// Caller must have `avx512f`, `self.ell == 64`, `self.n_chunks == 8`, and
    /// `bytes` must point to eight readable bytes.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn apply_x86_avx512_register_unchecked(
        &self,
        bytes: *const u8,
    ) -> core::arch::x86_64::__m512i {
        use core::arch::x86_64::*;
        debug_assert_eq!(self.ell, 64, "avx512 apply is specialized for ell = 64");
        debug_assert_eq!(self.n_chunks, 8, "avx512 apply expects eight input bytes");
        let base = self.data_ptr();
        // SAFETY: the caller guarantees eight readable input bytes. Every table
        // index is a u8 and each table row contains exactly 64 readable bytes.
        unsafe {
            let row0 = base.add(*bytes as usize * self.ell);
            let mut acc = _mm512_loadu_si512(row0 as *const __m512i);
            for b in 1..8 {
                let row_b = base.add(*bytes.add(b) as usize * self.ell);
                let v = _mm512_loadu_si512(row_b as *const __m512i);
                // Output lane c reads source lane c ^ (b >> 1).
                let perm = match b >> 1 {
                    0 => v,
                    1 => _mm512_shuffle_i64x2::<0xB1>(v, v),
                    2 => _mm512_shuffle_i64x2::<0x4E>(v, v),
                    _ => _mm512_shuffle_i64x2::<0x1B>(v, v),
                };
                // Odd b swaps the two 64-bit halves within every 128-bit lane.
                let perm = if b & 1 != 0 {
                    _mm512_shuffle_epi32::<0x4E>(perm)
                } else {
                    perm
                };
                acc = _mm512_xor_si512(acc, perm);
            }
            acc
        }
    }

    /// Apply M to three byte-packed rows (a, b, c) — matches the C++ hot-path
    /// signature. Identical math to three `apply` calls; kept separate so the
    /// future NEON port can batch loads across the three rows.
    pub fn apply_triple(
        &self,
        a_bytes: &[u8],
        a_out: &mut [F8],
        b_bytes: &[u8],
        b_out: &mut [F8],
        c_bytes: &[u8],
        c_out: &mut [F8],
    ) {
        self.apply(a_bytes, a_out);
        self.apply(b_bytes, b_out);
        self.apply(c_bytes, c_out);
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

    /// Naive reference: unpack `bytes` into `ell` GF(2)-valued F8 elements
    /// (one per coefficient bit), apply inv_NTT_S, then fwd_NTT_Λ.
    fn naive_apply(ntt_s: &AdditiveNttGf8, ntt_l: &AdditiveNttGf8, bytes: &[u8]) -> Vec<F8> {
        let ell = 1usize << ntt_s.k();
        assert_eq!(bytes.len(), ell / 8);
        let mut v = vec![F8::ZERO; ell];
        for (b, &byte) in bytes.iter().enumerate() {
            for t in 0..8 {
                if (byte >> t) & 1 != 0 {
                    v[8 * b + t] = F8::ONE;
                }
            }
        }
        ntt_s.inverse(&mut v);
        ntt_l.forward(&mut v);
        v
    }

    #[test]
    fn table_storage_is_cache_line_aligned() {
        for k in 3..=7 {
            let ntt_s = AdditiveNttGf8::new(k, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(k, F8(1u8 << k));
            let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            assert_eq!(table.data_ptr() as usize % 64, 0, "k={k}");
        }
    }

    #[test]
    fn matches_naive_k3() {
        let ntt_s = AdditiveNttGf8::new(3, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(3, F8(1 << 3));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert_eq!(table.ell, 8);
        assert_eq!(table.n_chunks, 1);

        let mut rng = Rng::new(100);
        let mut out = vec![F8::ZERO; 8];
        for _ in 0..64 {
            let bytes = [(rng.next_u64() & 0xff) as u8];
            table.apply(&bytes, &mut out);
            let expected = naive_apply(&ntt_s, &ntt_l, &bytes);
            assert_eq!(out, expected, "byte={:02x}", bytes[0]);
        }
    }

    #[test]
    fn matches_naive_k4() {
        let ntt_s = AdditiveNttGf8::new(4, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(4, F8(1 << 4));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert_eq!(table.ell, 16);
        assert_eq!(table.n_chunks, 2);

        let mut rng = Rng::new(101);
        let mut out = vec![F8::ZERO; 16];
        for _ in 0..64 {
            let bytes: [u8; 2] = [(rng.next_u64() & 0xff) as u8, (rng.next_u64() & 0xff) as u8];
            table.apply(&bytes, &mut out);
            let expected = naive_apply(&ntt_s, &ntt_l, &bytes);
            assert_eq!(out, expected, "bytes={:02x?}", bytes);
        }
    }

    #[test]
    fn matches_naive_k6_protocol_size() {
        // k_skip = 6 is the headline parameter for the m=29 workload.
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(6, F8(1 << 6));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert_eq!(table.ell, 64);
        assert_eq!(table.n_chunks, 8);

        let mut rng = Rng::new(102);
        let mut out = vec![F8::ZERO; 64];
        for _ in 0..16 {
            let bytes: Vec<u8> = (0..8).map(|_| (rng.next_u64() & 0xff) as u8).collect();
            table.apply(&bytes, &mut out);
            let expected = naive_apply(&ntt_s, &ntt_l, &bytes);
            assert_eq!(out, expected, "bytes={:02x?}", bytes);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn apply_neon_matches_apply_scalar() {
        // Cover both k=4 (n_chunks=2, n128=1) and k=6 (n_chunks=8, n128=4 —
        // the headline protocol size).
        for &k in &[4usize, 5, 6] {
            let ntt_s = AdditiveNttGf8::new(k, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(k, F8(1u8 << k));
            let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let n_chunks = table.n_chunks;
            let ell = table.ell;

            let mut rng = Rng::new(100 + k as u64);
            for _ in 0..32 {
                let bytes: Vec<u8> = (0..n_chunks)
                    .map(|_| (rng.next_u64() & 0xff) as u8)
                    .collect();
                let mut out_scalar = vec![F8::ZERO; ell];
                let mut out_neon = vec![F8::ZERO; ell];
                table.apply_scalar(&bytes, &mut out_scalar);
                // SAFETY: on aarch64.
                unsafe { table.apply_neon_unchecked(&bytes, &mut out_neon) };
                assert_eq!(
                    out_scalar, out_neon,
                    "scalar/neon apply disagree at k={k}, bytes={:02x?}",
                    bytes
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_x86_matches_apply_scalar() {
        // Cover k=4 (n_chunks=2, n128=1), k=5, and k=6 (n_chunks=8, n128=4 —
        // the headline protocol size).
        for &k in &[4usize, 5, 6] {
            let ntt_s = AdditiveNttGf8::new(k, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(k, F8(1u8 << k));
            let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let n_chunks = table.n_chunks;
            let ell = table.ell;

            let mut rng = Rng::new(100 + k as u64);
            for _ in 0..32 {
                let bytes: Vec<u8> = (0..n_chunks)
                    .map(|_| (rng.next_u64() & 0xff) as u8)
                    .collect();
                let mut out_scalar = vec![F8::ZERO; ell];
                let mut out_x86 = vec![F8::ZERO; ell];
                table.apply_scalar(&bytes, &mut out_scalar);
                // SAFETY: on x86_64.
                unsafe { table.apply_x86_unchecked(&bytes, &mut out_x86) };
                assert_eq!(
                    out_scalar, out_x86,
                    "scalar/x86 apply disagree at k={k}, bytes={:02x?}",
                    bytes
                );
                // AVX-512 path is specialized for ell == 64 (k = 6).
                #[cfg(target_feature = "avx512f")]
                if ell == 64 {
                    let mut out_avx = vec![F8::ZERO; ell];
                    // SAFETY: build carries avx512f; ell == 64.
                    unsafe { table.apply_x86_avx512_unchecked(&bytes, &mut out_avx) };
                    assert_eq!(
                        out_scalar, out_avx,
                        "scalar/avx512 apply disagree at k={k}, bytes={:02x?}",
                        bytes
                    );
                }
            }
        }
    }

    #[test]
    fn apply_triple_matches_three_singles() {
        let ntt_s = AdditiveNttGf8::new(5, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(5, F8(1 << 5));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);

        let mut rng = Rng::new(103);
        let nc = table.n_chunks;
        let ell = table.ell;
        let ab: Vec<u8> = (0..nc).map(|_| (rng.next_u64() & 0xff) as u8).collect();
        let bb: Vec<u8> = (0..nc).map(|_| (rng.next_u64() & 0xff) as u8).collect();
        let cb: Vec<u8> = (0..nc).map(|_| (rng.next_u64() & 0xff) as u8).collect();

        let mut a1 = vec![F8::ZERO; ell];
        let mut b1 = vec![F8::ZERO; ell];
        let mut c1 = vec![F8::ZERO; ell];
        table.apply(&ab, &mut a1);
        table.apply(&bb, &mut b1);
        table.apply(&cb, &mut c1);

        let mut a2 = vec![F8::ZERO; ell];
        let mut b2 = vec![F8::ZERO; ell];
        let mut c2 = vec![F8::ZERO; ell];
        table.apply_triple(&ab, &mut a2, &bb, &mut b2, &cb, &mut c2);

        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);
    }
}
