//! C2 — the no-compromise keccak producer: V = 8 instances simulated in
//! lockstep, witness words written **directly** at their L1′ addresses.
//!
//! No staging buffer, no second pass:
//! - the simulation state is `[[u64; V]; 25]` (lane-major, instance-minor),
//!   so every θ/ρπ/χ/ι op is a V-wide SIMD-friendly array op;
//! - a witness "word-row" (the same block-u64 index `w` across the V
//!   instances) is exactly the data L1′ wants contiguous: chunk
//!   `c = w >> 1`, dest words `(c << n_log) + o0 .. + V`. The [`RowWriter`]
//!   pairs the even/odd u64 halves of each 128-bit chunk and emits one
//!   contiguous `16·V`-byte non-temporal store run (128 B = exactly one
//!   cache line at V = 8);
//! - the lincheck stripe consumes the same in-flight row (`V = 8` = one
//!   stripe group) via the 8×64 bit-transpose — zero extra reads.
//!
//! Keccak's block layout is u64-aligned (state/t/const regions at word
//! boundaries), which is what makes the direct path this clean. The word
//! emission order is: state_0 (words 0..25), const (64), t_0..t_23
//! (65..665, sequential), state_24 (32..57); every chunk's even half arrives
//! before its odd half, and boundary flushes write zero odd-halves only into
//! genuine padding words.
//!
//! **Destination buffers must be zeroed once before first use** (padding
//! words are never written); reuse across proves keeps them valid since
//! every useful word is rewritten by assignment.

use flock_prover::r1cs_hashes::keccak::{State, state_to_lanes};
use rayon::prelude::*;

/// Instances per lockstep group. 8 = one lincheck-stripe group, and one
/// chunk-row emission = 128 B = one Apple cache line.
pub const V: usize = 8;

/// u64 words per block (k = 2^16 bits).
const U64_PER_BLOCK: usize = 1024;
/// Block-word bases (see keccak.rs layout): state_0, state_24, const, t_r.
const S0_W: usize = 0;
const S24_W: usize = 32;
const CONST_W: usize = 64;
const T_W: usize = 65;
const N_LANES: usize = 25;
const N_ROUNDS: usize = 24;

/// ρ offsets, `RHO_OFFSETS[a][b]` with `a = (x + 3y) % 5`, `b = x`
/// (copy of keccak.rs's private table; guarded by the lockstep test).
const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

type VLane = [u64; V];
type VLanes = [[u64; V]; N_LANES];

#[inline(always)]
fn theta_v(s: &mut VLanes) {
    let mut c = [[0u64; V]; 5];
    for x in 0..5 {
        for y in 0..5 {
            for j in 0..V {
                c[x][j] ^= s[x + 5 * y][j];
            }
        }
    }
    let mut d = [[0u64; V]; 5];
    for x in 0..5 {
        for j in 0..V {
            d[x][j] = c[(x + 4) % 5][j] ^ c[(x + 1) % 5][j].rotate_left(1);
        }
    }
    for i in 0..N_LANES {
        let x = i % 5;
        for j in 0..V {
            s[i][j] ^= d[x][j];
        }
    }
}

#[inline(always)]
fn rho_pi_v(s_in: &VLanes) -> VLanes {
    let mut out = [[0u64; V]; N_LANES];
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = RHO_OFFSETS[a][b] % 64;
            for j in 0..V {
                out[x + 5 * y][j] = s_in[a + 5 * b][j].rotate_left(r);
            }
        }
    }
    out
}

/// Non-temporal copy of one interleaved chunk-row (16·V = 128 bytes).
#[inline(always)]
unsafe fn nt_store_row(src: *const u64, dst: *mut u64) {
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

/// Pairs even/odd u64 halves of each 128-bit chunk and emits contiguous
/// NT-store runs at the L1′ destination.
struct RowWriter {
    dest: *mut u64,
    /// `(base-instance offset in words) * 2` — dest u64 index of chunk c is
    /// `(c << n_log) * 2 + o0_x2`.
    o0_x2: usize,
    n_log: usize,
    pending_w: usize, // block-u64 index of a held even half; usize::MAX = none
    pending: VLane,
}

impl RowWriter {
    fn new(dest: *mut u64, o0: usize, n_log: usize) -> Self {
        Self {
            dest,
            o0_x2: o0 * 2,
            n_log,
            pending_w: usize::MAX,
            pending: [0u64; V],
        }
    }

    #[inline(always)]
    unsafe fn emit(&mut self, c: usize, even: &VLane, odd: &VLane) {
        let mut buf = [0u64; 2 * V];
        for j in 0..V {
            buf[2 * j] = even[j];
            buf[2 * j + 1] = odd[j];
        }
        unsafe {
            let dst = self.dest.add((c << self.n_log) * 2 + self.o0_x2);
            nt_store_row(buf.as_ptr(), dst);
        }
    }

    /// Push the word-row for block-u64 index `w`. Within a run, `w` must
    /// arrive in order and each odd `w` must directly follow its even
    /// partner (keccak's natural emission order guarantees this).
    #[inline(always)]
    unsafe fn push(&mut self, w: usize, vals: &VLane) {
        debug_assert!(w < U64_PER_BLOCK);
        if w & 1 == 1 {
            debug_assert_eq!(self.pending_w, w - 1, "odd half without its even partner");
            let pending = self.pending;
            self.pending_w = usize::MAX;
            unsafe { self.emit(w >> 1, &pending, vals) };
        } else {
            unsafe { self.flush() };
            self.pending_w = w;
            self.pending = *vals;
        }
    }

    /// Complete a held even half with a zero odd half (the odd word is
    /// genuine padding at every keccak region boundary).
    #[inline(always)]
    unsafe fn flush(&mut self) {
        if self.pending_w != usize::MAX {
            let (w, pending) = (self.pending_w, self.pending);
            self.pending_w = usize::MAX;
            unsafe { self.emit(w >> 1, &pending, &[0u64; V]) };
        }
    }
}

#[derive(Copy, Clone)]
struct SendPtr(*mut u64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// Method (not field) access so `move` closures capture the whole
    /// wrapper — field capture would move the bare `*mut u64`, which is
    /// `!Send`.
    fn get(self) -> *mut u64 {
        self.0
    }
}

/// Build one group of V instances directly into L1′ dest buffers.
///
/// SAFETY: caller guarantees dest buffers are `2^(n_log) · U64_PER_BLOCK`
/// u64s, pre-zeroed in padding words, and that distinct groups (`o0`) are
/// never written concurrently with overlapping instance ranges.
unsafe fn build_group(
    states: &[State],
    o0: usize,
    n_log: usize,
    z: *mut u64,
    a: *mut u64,
    b: *mut u64,
    stripe: Option<*mut u8>,
) {
    use flock_core::bits::transpose_8_u64s_to_64_bytes;

    let mut wz = RowWriter::new(z, o0, n_log);
    let mut wa = RowWriter::new(a, o0, n_log);
    let mut wb = RowWriter::new(b, o0, n_log);
    let ones: VLane = [u64::MAX; V];
    let one: VLane = [1u64; V];

    // Stripe base for this group (V = 8 = one stripe group of 2^16 bytes).
    let stripe_base = stripe.map(|p| unsafe { p.add((o0 / 8) * U64_PER_BLOCK * 64) });
    // z word-row hook: stripe gets the 8×64 transpose of every z row.
    macro_rules! push_z {
        ($w:expr, $vals:expr) => {{
            let w: usize = $w;
            let vals: &VLane = $vals;
            if let Some(sb) = stripe_base {
                let out = unsafe { std::slice::from_raw_parts_mut(sb.add(w * 64), 64) };
                transpose_8_u64s_to_64_bytes(vals, out);
            }
            unsafe { wz.push(w, vals) };
        }};
    }

    // Initial lanes, instance-minor.
    let mut lanes: VLanes = [[0u64; V]; N_LANES];
    for (j, s) in states[o0..o0 + V].iter().enumerate() {
        let l = state_to_lanes(s);
        for i in 0..N_LANES {
            lanes[i][j] = l[i];
        }
    }

    unsafe {
        // state_0 self-loops: z = a = v, b = 1.
        for i in 0..N_LANES {
            push_z!(S0_W + i, &lanes[i]);
            wa.push(S0_W + i, &lanes[i]);
            wb.push(S0_W + i, &ones);
        }
        wz.flush();
        wa.flush();
        wb.flush();

        // Constant word (bit 0 of word 64): z = a = b = 1. Word 64 is even;
        // its odd partner is t_0's first row, pushed next.
        push_z!(CONST_W, &one);
        wa.push(CONST_W, &one);
        wb.push(CONST_W, &one);

        // 24 rounds; t rows are word-sequential 65..665.
        for r in 0..N_ROUNDS {
            let mut b_state = lanes;
            theta_v(&mut b_state);
            let b_state = rho_pi_v(&b_state);

            let mut t = [[0u64; V]; N_LANES];
            let mut next = [[0u64; V]; N_LANES];
            for y in 0..5 {
                for x in 0..5 {
                    let i = x + 5 * y;
                    let i1 = (x + 1) % 5 + 5 * y;
                    let i2 = (x + 2) % 5 + 5 * y;
                    for j in 0..V {
                        t[i][j] = (!b_state[i1][j]) & b_state[i2][j];
                        next[i][j] = b_state[i][j] ^ t[i][j];
                    }
                }
            }
            for j in 0..V {
                next[0][j] ^= ROUND_CONSTANTS[r];
            }

            let t_base = T_W + r * N_LANES;
            for y in 0..5 {
                for x in 0..5 {
                    let i = x + 5 * y;
                    let w = t_base + i;
                    push_z!(w, &t[i]);
                    let mut av = [0u64; V];
                    let i1 = (x + 1) % 5 + 5 * y;
                    for j in 0..V {
                        av[j] = !b_state[i1][j];
                    }
                    wa.push(w, &av);
                    wb.push(w, &b_state[(x + 2) % 5 + 5 * y]);
                }
            }

            lanes = next;
        }
        wz.flush();
        wa.flush();
        wb.flush();

        // state_24 pin rows: z = a = state_24, b = 1.
        for i in 0..N_LANES {
            push_z!(S24_W + i, &lanes[i]);
            wa.push(S24_W + i, &lanes[i]);
            wb.push(S24_W + i, &ones);
        }
        wz.flush();
        wa.flush();
        wb.flush();
    }
}

/// The direct L1′ keccak producer: parallel over V-instance groups.
///
/// `z`/`a`/`b` are L1′-layout u64 buffers (`2^n_log · 1024` u64s each) that
/// MUST be zeroed before first use (padding words are never written; reuse
/// across calls is fine). `stripe`, when `Some`, must be
/// `(2^n_log / 8) · 2^16` bytes, likewise pre-zeroed.
pub fn build_l1_direct(
    states: &[State],
    n_log: usize,
    stripe: Option<&mut [u8]>,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    let n = 1usize << n_log;
    assert_eq!(states.len(), n, "need exactly 2^n_log states");
    assert!(n >= V);
    let total = n * U64_PER_BLOCK;
    assert_eq!(z.len(), total);
    assert_eq!(a.len(), total);
    assert_eq!(b.len(), total);
    let (zp, ap, bp) = (
        SendPtr(z.as_mut_ptr()),
        SendPtr(a.as_mut_ptr()),
        SendPtr(b.as_mut_ptr()),
    );
    let sp = stripe.map(|s| {
        assert_eq!(s.len(), (n / 8) * U64_PER_BLOCK * 64);
        SendPtr(s.as_mut_ptr() as *mut u64)
    });

    let states_ref = &states[..];
    (0..n / V).into_par_iter().for_each(move |g| {
        // SAFETY: group g writes only instance words o0..o0+V (disjoint per
        // g) and its own stripe region; see build_group.
        unsafe {
            build_group(
                states_ref,
                g * V,
                n_log,
                zp.get(),
                ap.get(),
                bp.get(),
                sp.map(|p| p.get() as *mut u8),
            )
        }
    });
}
