//! Eight-message BLAKE3 compression: two transposed 4-wide NEON states in
//! flight.
//!
//! The blake3 crate's NEON backend is 4-wide (one `uint32x4_t` per state
//! word, four messages in lockstep). A single 4-wide state is latency-bound
//! on Apple silicon: the G function is a serial add–xor–rotate chain, so the
//! four NEON pipes sit half idle waiting on the dependency chain. Running a
//! SECOND independent 4-wide state interleaved G-for-G fills them — the same
//! independent-chains pattern as the prover's other kernels. (The challenge
//! tree reaches the same shape with 2.6k lines of generated assembly; this
//! is the idea re-derived as intrinsics, with LLVM doing the allocation.)
//!
//! Semantics match `blake3::platform::Platform::hash_many` with the IV key,
//! counter 0, `IncrementCounter::No`, and whole 64-byte blocks — exactly the
//! slice of the API `blake3_hash_many` uses. Byte-identical by test
//! (`blake3_neon8_matches_crate`).

use core::arch::aarch64::*;

use super::BLAKE3_IV;

/// Per-round message-word schedule, fixed by the BLAKE3 spec.
const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// Byte-shuffle index for a 32-bit rotate right by 8 (little-endian lanes).
const ROT8_TABLE: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];

#[inline(always)]
unsafe fn ror16(x: uint32x4_t) -> uint32x4_t {
    vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)))
}

#[inline(always)]
unsafe fn ror12(x: uint32x4_t) -> uint32x4_t {
    vsriq_n_u32::<12>(vshlq_n_u32::<20>(x), x)
}

#[inline(always)]
unsafe fn ror8(x: uint32x4_t, tbl: uint8x16_t) -> uint32x4_t {
    vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), tbl))
}

#[inline(always)]
unsafe fn ror7(x: uint32x4_t) -> uint32x4_t {
    vsriq_n_u32::<7>(vshlq_n_u32::<25>(x), x)
}

/// 4×4 u32 transpose: rows in, columns out.
#[inline(always)]
unsafe fn transpose4(
    r0: uint32x4_t,
    r1: uint32x4_t,
    r2: uint32x4_t,
    r3: uint32x4_t,
) -> [uint32x4_t; 4] {
    let t0 = vtrn1q_u32(r0, r1);
    let t1 = vtrn2q_u32(r0, r1);
    let t2 = vtrn1q_u32(r2, r3);
    let t3 = vtrn2q_u32(r2, r3);
    [
        vreinterpretq_u32_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(t0),
            vreinterpretq_u64_u32(t2),
        )),
        vreinterpretq_u32_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(t1),
            vreinterpretq_u64_u32(t3),
        )),
        vreinterpretq_u32_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(t0),
            vreinterpretq_u64_u32(t2),
        )),
        vreinterpretq_u32_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(t1),
            vreinterpretq_u64_u32(t3),
        )),
    ]
}

/// Load block `b` of four consecutive `stride`-byte messages starting at
/// `msgs[first]`, transposed: `out[w]` holds message word `w` across the
/// four messages.
///
/// # Safety
/// Caller guarantees `base + (first + i) * stride + (b + 1) * 64 <= data end`
/// for `i < 4`.
#[inline(always)]
unsafe fn load_block_transposed(
    base: *const u8,
    stride: usize,
    first: usize,
    b: usize,
) -> [uint32x4_t; 16] {
    let mut m = [vdupq_n_u32(0); 16];
    let p0 = base.add(first * stride + b * 64) as *const u32;
    let p1 = base.add((first + 1) * stride + b * 64) as *const u32;
    let p2 = base.add((first + 2) * stride + b * 64) as *const u32;
    let p3 = base.add((first + 3) * stride + b * 64) as *const u32;
    for g in 0..4 {
        let t = transpose4(
            vld1q_u32(p0.add(4 * g)),
            vld1q_u32(p1.add(4 * g)),
            vld1q_u32(p2.add(4 * g)),
            vld1q_u32(p3.add(4 * g)),
        );
        m[4 * g] = t[0];
        m[4 * g + 1] = t[1];
        m[4 * g + 2] = t[2];
        m[4 * g + 3] = t[3];
    }
    m
}

/// Compress one 64-byte block for both 4-wide halves, interleaved G-for-G.
#[inline(always)]
unsafe fn compress2(
    cva: &mut [uint32x4_t; 8],
    cvb: &mut [uint32x4_t; 8],
    ma: &[uint32x4_t; 16],
    mb: &[uint32x4_t; 16],
    flags: uint32x4_t,
    blen: uint32x4_t,
    tbl: uint8x16_t,
) {
    let iv0 = vdupq_n_u32(BLAKE3_IV[0]);
    let iv1 = vdupq_n_u32(BLAKE3_IV[1]);
    let iv2 = vdupq_n_u32(BLAKE3_IV[2]);
    let iv3 = vdupq_n_u32(BLAKE3_IV[3]);
    let zero = vdupq_n_u32(0);

    let mut va = [
        cva[0], cva[1], cva[2], cva[3], cva[4], cva[5], cva[6], cva[7], iv0, iv1, iv2, iv3, zero,
        zero, blen, flags,
    ];
    let mut vb = [
        cvb[0], cvb[1], cvb[2], cvb[3], cvb[4], cvb[5], cvb[6], cvb[7], iv0, iv1, iv2, iv3, zero,
        zero, blen, flags,
    ];

    macro_rules! g2 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $y:expr) => {{
            let (a, b, c, d, x, y) = ($a, $b, $c, $d, $x, $y);
            va[a] = vaddq_u32(vaddq_u32(va[a], va[b]), ma[x]);
            vb[a] = vaddq_u32(vaddq_u32(vb[a], vb[b]), mb[x]);
            va[d] = ror16(veorq_u32(va[d], va[a]));
            vb[d] = ror16(veorq_u32(vb[d], vb[a]));
            va[c] = vaddq_u32(va[c], va[d]);
            vb[c] = vaddq_u32(vb[c], vb[d]);
            va[b] = ror12(veorq_u32(va[b], va[c]));
            vb[b] = ror12(veorq_u32(vb[b], vb[c]));
            va[a] = vaddq_u32(vaddq_u32(va[a], va[b]), ma[y]);
            vb[a] = vaddq_u32(vaddq_u32(vb[a], vb[b]), mb[y]);
            va[d] = ror8(veorq_u32(va[d], va[a]), tbl);
            vb[d] = ror8(veorq_u32(vb[d], vb[a]), tbl);
            va[c] = vaddq_u32(va[c], va[d]);
            vb[c] = vaddq_u32(vb[c], vb[d]);
            va[b] = ror7(veorq_u32(va[b], va[c]));
            vb[b] = ror7(veorq_u32(vb[b], vb[c]));
        }};
    }

    for s in &MSG_SCHEDULE {
        g2!(0, 4, 8, 12, s[0], s[1]);
        g2!(1, 5, 9, 13, s[2], s[3]);
        g2!(2, 6, 10, 14, s[4], s[5]);
        g2!(3, 7, 11, 15, s[6], s[7]);
        g2!(0, 5, 10, 15, s[8], s[9]);
        g2!(1, 6, 11, 12, s[10], s[11]);
        g2!(2, 7, 8, 13, s[12], s[13]);
        g2!(3, 4, 9, 14, s[14], s[15]);
    }

    for i in 0..8 {
        cva[i] = veorq_u32(va[i], va[i + 8]);
        cvb[i] = veorq_u32(vb[i], vb[i + 8]);
    }
}

/// Store one 4-wide half's chaining values: message `first + j`'s 8 CV words
/// to `out[(first + j) * 32..]`, little-endian — the `hash_many` layout.
#[inline(always)]
unsafe fn store_cvs(cv: &[uint32x4_t; 8], out: *mut u8, first: usize) {
    let lo = transpose4(cv[0], cv[1], cv[2], cv[3]);
    let hi = transpose4(cv[4], cv[5], cv[6], cv[7]);
    for j in 0..4 {
        let p = out.add((first + j) * 32) as *mut u32;
        vst1q_u32(p, lo[j]);
        vst1q_u32(p.add(4), hi[j]);
    }
}

/// [`load_block_transposed`] for a PARTIAL final block: each lane's `len`
/// bytes (`1..64`) are copied into a zeroed 64-byte block first, exactly
/// the zero padding BLAKE3 specifies for a short last block.
#[inline(always)]
unsafe fn load_block_transposed_partial(
    base: *const u8,
    stride: usize,
    first: usize,
    b: usize,
    len: usize,
) -> [uint32x4_t; 16] {
    let mut bufs = [[0u8; 64]; 4];
    for (lane, buf) in bufs.iter_mut().enumerate() {
        core::ptr::copy_nonoverlapping(
            base.add((first + lane) * stride + b * 64),
            buf.as_mut_ptr(),
            len,
        );
    }
    let mut m = [vdupq_n_u32(0); 16];
    for g in 0..4 {
        let t = transpose4(
            vld1q_u32(bufs[0].as_ptr().add(16 * g) as *const u32),
            vld1q_u32(bufs[1].as_ptr().add(16 * g) as *const u32),
            vld1q_u32(bufs[2].as_ptr().add(16 * g) as *const u32),
            vld1q_u32(bufs[3].as_ptr().add(16 * g) as *const u32),
        );
        m[4 * g] = t[0];
        m[4 * g + 1] = t[1];
        m[4 * g + 2] = t[2];
        m[4 * g + 3] = t[3];
    }
    m
}

/// Hash eight contiguous `stride`-byte messages (`n_blocks` 64-byte blocks
/// each, the last one `last_block_len` bytes long, `1..=64`) into eight
/// 32-byte chaining values at `out`.
///
/// Equivalent to `hash_many` with the IV key, counter 0 and no increment:
/// block 0 carries `flags | flags_start`, the last block `flags | flags_end`
/// and its true byte length (a short last block is zero-padded, as BLAKE3
/// specifies) — so any message of `1..=1024` bytes gets its exact chunk CV.
///
/// # Safety
/// `data` must hold at least `7 * stride + (n_blocks - 1) * 64 +
/// last_block_len` readable bytes (`stride` at least that per message);
/// `out` must hold 256 writable bytes. NEON must be available (aarch64).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn hash8(
    data: *const u8,
    stride: usize,
    n_blocks: usize,
    last_block_len: usize,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
) {
    debug_assert!(n_blocks >= 1 && (1..=64).contains(&last_block_len));
    debug_assert!(stride >= (n_blocks - 1) * 64 + last_block_len);
    let tbl = vld1q_u8(ROT8_TABLE.as_ptr());
    let full = vdupq_n_u32(64);
    let mut cva = [vdupq_n_u32(0); 8];
    let mut cvb = [vdupq_n_u32(0); 8];
    for i in 0..8 {
        cva[i] = vdupq_n_u32(BLAKE3_IV[i]);
        cvb[i] = vdupq_n_u32(BLAKE3_IV[i]);
    }
    for b in 0..n_blocks {
        let mut fl = flags;
        if b == 0 {
            fl |= flags_start;
        }
        let last = b == n_blocks - 1;
        if last {
            fl |= flags_end;
        }
        let flv = vdupq_n_u32(fl as u32);
        if last && last_block_len < 64 {
            let ma = load_block_transposed_partial(data, stride, 0, b, last_block_len);
            let mb = load_block_transposed_partial(data, stride, 4, b, last_block_len);
            let blen = vdupq_n_u32(last_block_len as u32);
            compress2(&mut cva, &mut cvb, &ma, &mb, flv, blen, tbl);
        } else {
            let ma = load_block_transposed(data, stride, 0, b);
            let mb = load_block_transposed(data, stride, 4, b);
            compress2(&mut cva, &mut cvb, &ma, &mb, flv, full, tbl);
        }
    }
    store_cvs(&cva, out, 0);
    store_cvs(&cvb, out, 4);
}
