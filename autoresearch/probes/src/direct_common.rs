//! Shared plumbing for the direct-write (C2) producers of the bit-packed
//! hashes (sha2, blake3), whose witness layouts are NOT u64-aligned.
//!
//! Recipe: simulate V = 8 instances in lockstep; OR fields into a group-local
//! **interleaved row buffer** `rows[w] = [u64; V]` (word-index-major,
//! instance-minor — i.e. already the L1′ order, 16–32 KB, L1-resident); then
//! NT-flush the useful chunks as contiguous 128-byte runs and transpose the
//! stripe straight out of the hot rows.

pub const V: usize = 8;
pub type Row = [u64; V];

#[derive(Copy, Clone)]
pub struct SendPtr(pub *mut u64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// Method (not field) access so `move` closures capture the wrapper.
    pub fn get(self) -> *mut u64 {
        self.0
    }
}

/// V-wide `or_u32_at_bit`: OR the V instances' 32-bit values into row `w`
/// (and `w+1` on straddle) at bit offset `bit`.
#[inline(always)]
pub fn or_u32_row(rows: &mut [Row], bit: usize, vals: &[u32; V]) {
    let w = bit >> 6;
    let s = bit & 63;
    for j in 0..V {
        rows[w][j] |= (vals[j] as u64) << s;
    }
    if s > 32 {
        for j in 0..V {
            rows[w + 1][j] |= (vals[j] as u64) >> (64 - s);
        }
    }
}

/// V-wide `or_bit_at`: set bit `bit` in every instance's row.
#[inline(always)]
pub fn or_bit_row(rows: &mut [Row], bit: usize) {
    let w = bit >> 6;
    let s = bit & 63;
    for j in 0..V {
        rows[w][j] |= 1u64 << s;
    }
}

/// Non-temporal store of one interleaved 128-byte chunk-row.
#[inline(always)]
pub unsafe fn nt_store_row(src: *const u64, dst: *mut u64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "ldp {t0:q}, {t1:q}, [{s}]",
            "stnp {t0:q}, {t1:q}, [{d}]",
            "ldp {t0:q}, {t1:q}, [{s}, #32]",
            "stnp {t0:q}, {t1:q}, [{d}, #32]",
            "ldp {t0:q}, {t1:q}, [{s}, #64]",
            "stnp {t0:q}, {t1:q}, [{d}, #64]",
            "ldp {t0:q}, {t1:q}, [{s}, #96]",
            "stnp {t0:q}, {t1:q}, [{d}, #96]",
            s = in(reg) src, d = in(reg) dst,
            t0 = out(vreg) _, t1 = out(vreg) _,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, 2 * V);
    }
}

/// NT-flush `useful_chunks` chunk-rows of an interleaved row buffer to the
/// L1′ destination (`dest` word index `(c << n_log) + o0`).
///
/// SAFETY: caller guarantees dest sizing and per-group disjointness.
#[inline]
pub unsafe fn flush_rows_nt(
    rows: &[Row],
    dest: *mut u64,
    o0: usize,
    n_log: usize,
    useful_chunks: usize,
) {
    debug_assert!(2 * useful_chunks <= rows.len());
    for c in 0..useful_chunks {
        let even = &rows[2 * c];
        let odd = &rows[2 * c + 1];
        let mut buf = [0u64; 2 * V];
        for j in 0..V {
            buf[2 * j] = even[j];
            buf[2 * j + 1] = odd[j];
        }
        unsafe {
            nt_store_row(buf.as_ptr(), dest.add(((c << n_log) + o0) * 2));
        }
    }
}

/// Transpose the z rows into the lincheck byte-stripe for this V = 8 group.
/// Only `useful_words` rows are written (the rest of the stripe is zero and
/// must be pre-zeroed by the caller, once).
#[inline]
pub unsafe fn stripe_from_rows(
    rows: &[Row],
    stripe: *mut u8,
    o0: usize,
    u64_per_block: usize,
    useful_words: usize,
) {
    use flock_core::bits::transpose_8_u64s_to_64_bytes;
    let base = (o0 / 8) * u64_per_block * 64;
    for (w, row) in rows.iter().enumerate().take(useful_words) {
        let out = unsafe { std::slice::from_raw_parts_mut(stripe.add(base + w * 64), 64) };
        transpose_8_u64s_to_64_bytes(row, out);
    }
}

/// V-wide `add_carry_parts`: per-instance `(sum, left, right, carry_aux)`.
#[inline(always)]
pub fn add_carry_parts_v(
    x: &[u32; V],
    y: &[u32; V],
) -> ([u32; V], [u32; V], [u32; V], [u32; V]) {
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let mut sum = [0u32; V];
    let mut left = [0u32; V];
    let mut right = [0u32; V];
    let mut carry = [0u32; V];
    for j in 0..V {
        let s = x[j].wrapping_add(y[j]);
        let cin = s ^ x[j] ^ y[j];
        let l = (x[j] ^ cin) & MASK_LO31;
        let r = (y[j] ^ cin) & MASK_LO31;
        sum[j] = s;
        left[j] = l;
        right[j] = r;
        carry[j] = l & r;
    }
    (sum, left, right, carry)
}
