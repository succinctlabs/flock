//! Copies of `flock_prover::r1cs_hashes::common`'s pub(crate) bit-packing
//! helpers (not reachable from outside the crate). Kept in lockstep by the
//! per-hash builder tests in `tests/e1_builder_lockstep.rs` — if these drift
//! from common.rs the byte-equality against the public drivers fails.

/// OR the low 32 bits of `val` into `buf` starting at bit-offset `bit_off`.
#[inline(always)]
pub fn or_u32_at_bit(buf: &mut [u64], bit_off: usize, val: u32) {
    let u64_idx = bit_off >> 6;
    let shift = bit_off & 63;
    buf[u64_idx] |= (val as u64) << shift;
    if shift > 32 {
        buf[u64_idx + 1] |= (val as u64) >> (64 - shift);
    }
}

/// Set bit `bit_off` of `buf`.
#[inline(always)]
pub fn or_bit_at(buf: &mut [u64], bit_off: usize) {
    buf[bit_off >> 6] |= 1u64 << (bit_off & 63);
}

/// A `64·NW`-bit record composed in registers and OR-flushed once.
pub struct BitRecord<const NW: usize> {
    w: [u64; NW],
}

impl<const NW: usize> BitRecord<NW> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { w: [0u64; NW] }
    }

    #[inline(always)]
    pub fn push<const POS: usize>(&mut self, val: u32) {
        let v = val as u64;
        let idx = POS >> 6;
        let s = POS & 63;
        self.w[idx] |= v << s;
        if s > 32 {
            self.w[idx + 1] |= v >> (64 - s);
        }
    }

    #[inline(always)]
    pub fn flush(&self, buf: &mut [u64], base_bit: usize) {
        let bi = base_bit >> 6;
        let s = base_bit & 63;
        let mut spill = 0u64;
        for j in 0..NW {
            buf[bi + j] |= (self.w[j] << s) | spill;
            spill = (self.w[j] >> 1) >> (63 - s);
        }
        buf[bi + NW] |= spill;
    }
}

impl<const NW: usize> Default for BitRecord<NW> {
    fn default() -> Self {
        Self::new()
    }
}

/// One 32-bit ADD's witness parts: `(sum, left, right, carry_aux)`.
#[inline(always)]
pub fn add_carry_parts(x: u32, y: u32) -> (u32, u32, u32, u32) {
    let sum = x.wrapping_add(y);
    let cin = sum ^ x ^ y;
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let left = (x ^ cin) & MASK_LO31;
    let right = (y ^ cin) & MASK_LO31;
    let carry_aux = left & right;
    (sum, left, right, carry_aux)
}
