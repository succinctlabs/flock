//! Bit-packing and R1CS-row helpers shared by the monolithic hash R1CS
//! modules (`sha2`, `blake3`, `keccak`). The shared `prove_fast`
//! orchestration lives in [`crate::prover::prove_fast_from_witness`].

use std::sync::OnceLock;

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

/// OR the low 32 bits of `val` into `buf` starting at bit-offset `bit_off`.
/// Handles u64 straddling when `bit_off % 64 > 32`.
#[inline(always)]
pub(crate) fn or_u32_at_bit(buf: &mut [u64], bit_off: usize, val: u32) {
    let u64_idx = bit_off >> 6;
    let shift = bit_off & 63;
    buf[u64_idx] |= (val as u64) << shift;
    if shift > 32 {
        buf[u64_idx + 1] |= (val as u64) >> (64 - shift);
    }
}

/// Set bit `bit_off` of `buf` (low-bit-first within each u64).
#[inline(always)]
pub(crate) fn or_bit_at(buf: &mut [u64], bit_off: usize) {
    buf[bit_off >> 6] |= 1u64 << (bit_off & 63);
}

/// A `64·NW`-bit record composed in registers and OR-flushed into the block
/// buffer once.
///
/// Hash witness builders write groups of adjacent sub-word fields (e.g.
/// 31-bit carry slots) with `or_u32_at_bit`; back-to-back fields hit the
/// same u64 word, serializing on store-to-load forwarding, with a straddle
/// branch per call. Composing the group in registers (const positions,
/// branchless) and flushing with one `NW + 1`-word shifted OR pass turns
/// ~2 read-modify-writes per field into `NW + 1` per group.
pub(crate) struct BitRecord<const NW: usize> {
    w: [u64; NW],
}

impl<const NW: usize> BitRecord<NW> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self { w: [0u64; NW] }
    }

    /// OR a (pre-masked) value into record bits `[POS, POS + width)`.
    /// `POS` is const so the straddle branch and shifts fold at compile time.
    #[inline(always)]
    pub(crate) fn push<const POS: usize>(&mut self, val: u32) {
        let v = val as u64;
        let idx = POS >> 6;
        let s = POS & 63;
        self.w[idx] |= v << s;
        if s > 32 {
            self.w[idx + 1] |= v >> (64 - s);
        }
    }

    /// OR the record into `buf` starting at bit `base_bit`.
    #[inline(always)]
    pub(crate) fn flush(&self, buf: &mut [u64], base_bit: usize) {
        let bi = base_bit >> 6;
        let s = base_bit & 63;
        let mut spill = 0u64;
        for j in 0..NW {
            buf[bi + j] |= (self.w[j] << s) | spill;
            // `(x >> 1) >> (63 - s)` = `x >> (64 - s)` without the s = 0 UB.
            spill = (self.w[j] >> 1) >> (63 - s);
        }
        buf[bi + NW] |= spill;
    }
}

/// One 32-bit ADD's witness parts: `(sum, left, right, carry_aux)` with
/// `left/right/carry_aux` masked to the low 31 bits (bit 31 is the discarded
/// mod-2³² carry-out; the carry slot is 31 bits wide).
#[inline(always)]
pub(crate) fn add_carry_parts(x: u32, y: u32) -> (u32, u32, u32, u32) {
    let sum = x.wrapping_add(y);
    let cin = sum ^ x ^ y;
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let left = (x ^ cin) & MASK_LO31;
    let right = (y ^ cin) & MASK_LO31;
    let carry_aux = left & right;
    (sum, left, right, carry_aux)
}

// ---------------------------------------------------------------------------
// Shared R1CS helpers: empty matrix, identity, BlockR1cs stub builder.
//
// The K_LOG=16 hash encoders all use empty A_0/B_0 matrices (constraint
// definition lives in their LincheckCircuit walkers) and C_0 = I_K. These
// three helpers were duplicated across keccak.rs, blake3.rs, sha2.rs.
// ---------------------------------------------------------------------------

/// K × K sparse matrix with no nonzero entries. Used as an `a_0`/`b_0` stub
/// when the constraint definition lives in a `LincheckCircuit` walker.
pub(crate) fn empty_matrix(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: vec![Vec::new(); k],
    }
}

/// K × K identity sparse matrix.
pub(crate) fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k).map(|i| vec![i]).collect(),
    }
}

/// Build a `BlockR1cs` shell with empty A_0, B_0 stubs and C_0 = I_K. The
/// constraint definition lives in a per-hash `LincheckCircuit` walker. Used
/// by Keccak.
pub(crate) fn build_block_r1cs_empty_stub(
    n_blocks_log: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
) -> BlockR1cs {
    let k = 1usize << k_log;
    // Empty-stub R1CS carry their constraints (and constant-wire pin) on a
    // per-hash `LincheckCircuit` walker, so no R1CS-level `const_pin` here.
    build_block_r1cs_with_matrices(
        n_blocks_log,
        k_log,
        k_skip,
        useful_bits,
        empty_matrix(k),
        empty_matrix(k),
        None,
    )
}

/// Build a `BlockR1cs` with caller-supplied A_0, B_0 sparse matrices and
/// C_0 = I_K. Used by BLAKE3 and SHA-2 (they materialize real A_0/B_0 via
/// their `build_matrices`).
///
/// `useful_bits ≤ 2^k_log` declares how many rows of each block carry real
/// data; the remainder is zero padding (URM can skip work over those).
///
/// `const_pin` is the column of the constant-one wire to pin to 1 across all
/// blocks (closing the all-zero soundness gap — see `docs/const-wire-pin.md`),
/// or `None`. It is propagated into the CSC / sparse `LincheckCircuit` this
/// R1CS builds. Encoders that set it MUST fill padding blocks with valid
/// (constant = 1) computations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_r1cs_with_matrices(
    n_blocks_log: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    a_0: SparseBinaryMatrix,
    b_0: SparseBinaryMatrix,
    const_pin: Option<usize>,
) -> BlockR1cs {
    assert!(
        n_blocks_log >= 3,
        "lincheck needs n_outer ≥ 8 — pick n_blocks_log ≥ 3"
    );
    let k = 1usize << k_log;
    assert!(
        useful_bits <= k,
        "useful_bits ({useful_bits}) must be ≤ 2^k_log ({k})"
    );
    BlockR1cs {
        m: k_log + n_blocks_log,
        k_log,
        k_skip,
        useful_bits,
        a_0,
        b_0,
        c_0: identity(k),
        const_pin,
        digest_cache: OnceLock::new(),
        csc_cache: OnceLock::new(),
    }
}

// ---------------------------------------------------------------------------
// Generic witness packing driver.
//
// All three hash encoders (keccak, blake3, sha2) had identical chunked
// parallel iteration + bit-transpose-to-stripe boilerplate around their
// per-block witness builder. This driver captures that shape; each hash
// passes its `per_block` closure that fills 3 length-`U64_PER_BLOCK`
// buffers (z, a, b) from one input.
// ---------------------------------------------------------------------------

/// Drive the parallel chunked witness build for `n_blocks` instances padded
/// to `2^n_blocks_log` slots. Returns `(z, a, b, z_lincheck)` packed in
/// F128 form (z/a/b) and byte-stripe form (z_lincheck).
///
/// `per_block(initial, z_u64, a_u64, b_u64)` populates one block's worth of
/// `(z, a, b)` data — 3 zero-initialized `u64`-buffers of length `K / 64`.
/// `K` is derived from `k_log`. `initial_states.len()` may be less than
/// `2^n_blocks_log`.
///
/// `padding` controls what fills the trailing `2^n_blocks_log −
/// initial_states.len()` slots:
/// - `None` — leave them all-zero (trivial constraint satisfaction).
/// - `Some(p)` — build a real block from `p` in every padding slot. Encoders
///   that pin a constant wire need this so the constant column is all-ones
///   across *every* batched instance (see `docs/const-wire-pin.md`); for keccak
///   the padding input is the all-zero state, whose witness is `keccak_f(0)`.
pub(crate) fn drive_witness_packed_and_lincheck<S: Sync, F>(
    initial_states: &[S],
    padding: Option<&S>,
    n_blocks_log: usize,
    k_log: usize,
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    use rayon::prelude::*;

    let k = 1usize << k_log;
    let f128_per_block = k / 128;
    let u64_per_block = k / 64;
    let n_total = 1usize << n_blocks_log;
    let n_blocks = initial_states.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} blocks > 2^{n_blocks_log} = {n_total} slots"
    );
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "lincheck stripe layout requires n_total ≥ 8 and divisible by 8"
    );

    let total_f128 = n_total * f128_per_block;
    // z/a/b are allocated uninitialized and zeroed *inside* the parallel loop
    // (one memset per 8-block group), so the ~192 MB zero-fill scales with the
    // thread count instead of running serially on the main thread before the
    // parallel build. The per-block builders OR 1-bits into pre-zeroed words,
    // so each group must be zeroed before its `per_block` calls. `z_lincheck`
    // stays `vec![0u8; _]` (lazy `alloc_zeroed`/mmap — no eager memset).
    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut z_lincheck = vec![0u8; (n_total / 8) * k];

    z.par_chunks_mut(8 * f128_per_block)
        .zip(a.par_chunks_mut(8 * f128_per_block))
        .zip(b.par_chunks_mut(8 * f128_per_block))
        .zip(z_lincheck.par_chunks_mut(k))
        .enumerate()
        .for_each(|(g, (((z_grp, a_grp), b_grp), stripe))| {
            // Zero this group's z/a/b up front (parallel memset — the buffers
            // were uninit-allocated). The per-block builder ORs 1-bits into
            // pre-zeroed words; any slot left unbuilt (no padding block) stays
            // zero, which the lincheck transpose below reads correctly.
            // SAFETY: F128 is `Copy` (no Drop) and the all-zero bit pattern is
            // the valid `F128::ZERO`, so a byte memset is a correct init.
            unsafe {
                std::ptr::write_bytes(z_grp.as_mut_ptr(), 0, z_grp.len());
                std::ptr::write_bytes(a_grp.as_mut_ptr(), 0, a_grp.len());
                std::ptr::write_bytes(b_grp.as_mut_ptr(), 0, b_grp.len());
            }
            for k_in in 0..8 {
                let global_idx = 8 * g + k_in;
                let init: &S = if global_idx < n_blocks {
                    &initial_states[global_idx]
                } else if let Some(p) = padding {
                    // Fill the padding slot with a real block so its constant
                    // wire is set (see `padding` docs above).
                    p
                } else {
                    // No padding block — leave this slot zero.
                    continue;
                };
                let z_chunk = &mut z_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                let a_chunk = &mut a_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                let b_chunk = &mut b_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                // SAFETY: F128 is `repr(C, align(16))` with two `u64` fields in
                // LE order — same byte layout as a u64 pair.
                let z_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        z_chunk.as_mut_ptr() as *mut u64,
                        z_chunk.len() * 2,
                    )
                };
                let a_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        a_chunk.as_mut_ptr() as *mut u64,
                        a_chunk.len() * 2,
                    )
                };
                let b_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        b_chunk.as_mut_ptr() as *mut u64,
                        b_chunk.len() * 2,
                    )
                };
                per_block(init, z_u64, a_u64, b_u64);
            }

            // Bit-transpose 8 z chunks into the lincheck stripe.
            let z_u64_all: &[u64] = unsafe {
                std::slice::from_raw_parts(z_grp.as_ptr() as *const u64, z_grp.len() * 2)
            };
            for i in 0..u64_per_block {
                let lanes: [u64; 8] = [
                    z_u64_all[0 * u64_per_block + i],
                    z_u64_all[u64_per_block + i],
                    z_u64_all[2 * u64_per_block + i],
                    z_u64_all[3 * u64_per_block + i],
                    z_u64_all[4 * u64_per_block + i],
                    z_u64_all[5 * u64_per_block + i],
                    z_u64_all[6 * u64_per_block + i],
                    z_u64_all[7 * u64_per_block + i],
                ];
                transpose_8_u64s_to_64_bytes(&lanes, &mut stripe[i * 64..i * 64 + 64]);
            }
        });

    (z, a, b, z_lincheck)
}

/// Sort `v` and remove pairs of duplicates (GF(2) cancellation). Keeps R1CS
/// rows in canonical (sorted, square-free) form.
pub(crate) fn xor_dedup(mut v: Vec<usize>) -> Vec<usize> {
    v.sort();
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        let val = v[i];
        let mut count = 0;
        while i < v.len() && v[i] == val {
            count += 1;
            i += 1;
        }
        if count % 2 == 1 {
            out.push(val);
        }
    }
    out
}

