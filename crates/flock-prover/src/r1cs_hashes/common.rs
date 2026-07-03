//! Bit-packing and R1CS-row helpers shared by the monolithic hash R1CS
//! modules (`sha2`, `blake3`, `keccak`). The shared `prove_fast`
//! orchestration lives in [`crate::prover::prove_fast_from_witness`].

use std::sync::OnceLock;

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};

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
        layout: WitnessLayout::RowMajor,
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


// ---------------------------------------------------------------------------
// Batch-major (WitnessLayout::BatchMajor) witness-producer plumbing.
//
// The batch-major producers simulate V = 8 instances in lockstep and write
// witness words directly at their batch-major addresses: the word-row for
// block-u64 index `w` across the 8 instances is exactly one 128-byte
// chunk-row (= one cache line) at dest word `((w >> 1) << n_log) + o0`,
// stored non-temporally (dest lines are fully overwritten and not re-read
// soon, so write-allocate reads are pure waste). V = 8 also equals the
// lincheck stripe group, so the byte-stripe is transposed from the in-flight
// rows at zero extra reads.
//
// Producer contract: chunk-columns `[0, useful_chunks)` are FULLY written
// every call; the padding suffix `[useful_chunks, k/128)` columns are never
// written (the generators zero that contiguous buffer suffix themselves, so
// recycled scratch buffers stay valid).
// ---------------------------------------------------------------------------

/// Instances per lockstep group (= one lincheck-stripe group; one chunk-row
/// emission = 128 B).
pub(crate) const BM_V: usize = 8;
/// One interleaved word-row: the same block-u64 index across BM_V instances.
pub(crate) type BmRow = [u64; BM_V];

/// Raw-pointer wrapper for the disjoint per-group strided writes.
#[derive(Copy, Clone)]
pub(crate) struct SendPtr(pub *mut u64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// Method (not field) access so `move` closures capture the whole
    /// wrapper — field capture would move the bare `*mut u64` (`!Send`).
    pub(crate) fn get(self) -> *mut u64 {
        self.0
    }
}

/// V-wide `or_u32_at_bit`: OR the V instances' 32-bit values into row `w`
/// (and `w + 1` on straddle) at bit offset `bit`.
#[inline(always)]
pub(crate) fn or_u32_row(rows: &mut [BmRow], bit: usize, vals: &[u32; BM_V]) {
    let w = bit >> 6;
    let s = bit & 63;
    for j in 0..BM_V {
        rows[w][j] |= (vals[j] as u64) << s;
    }
    if s > 32 {
        for j in 0..BM_V {
            rows[w + 1][j] |= (vals[j] as u64) >> (64 - s);
        }
    }
}

/// V-wide `or_bit_at`: set bit `bit` in every instance's row.
#[inline(always)]
pub(crate) fn or_bit_row(rows: &mut [BmRow], bit: usize) {
    let w = bit >> 6;
    let s = bit & 63;
    for j in 0..BM_V {
        rows[w][j] |= 1u64 << s;
    }
}

/// Non-temporal store of one interleaved 128-byte chunk-row.
#[inline(always)]
pub(crate) unsafe fn nt_store_row(src: *const u64, dst: *mut u64) {
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
        std::ptr::copy_nonoverlapping(src, dst, 2 * BM_V);
    }
}

/// NT-flush `useful_chunks` chunk-rows of an interleaved row buffer to the
/// batch-major destination (dest word index `(c << n_log) + o0`).
///
/// SAFETY: caller guarantees dest sizing and per-group disjointness.
#[inline]
pub(crate) unsafe fn flush_rows_nt(
    rows: &[BmRow],
    dest: *mut u64,
    o0: usize,
    n_log: usize,
    useful_chunks: usize,
) {
    debug_assert!(2 * useful_chunks <= rows.len());
    for c in 0..useful_chunks {
        let even = &rows[2 * c];
        let odd = &rows[2 * c + 1];
        let mut buf = [0u64; 2 * BM_V];
        for j in 0..BM_V {
            buf[2 * j] = even[j];
            buf[2 * j + 1] = odd[j];
        }
        unsafe {
            nt_store_row(buf.as_ptr(), dest.add(((c << n_log) + o0) * 2));
        }
    }
}

/// Transpose the z rows into the lincheck byte-stripe for one V = 8 group.
/// Only `useful_words` rows are written (the stripe tail stays zero).
#[inline]
pub(crate) unsafe fn stripe_from_rows(
    rows: &[BmRow],
    stripe: *mut u8,
    o0: usize,
    u64_per_block: usize,
    useful_words: usize,
) {
    let base = (o0 / 8) * u64_per_block * 64;
    for (w, row) in rows.iter().enumerate().take(useful_words) {
        let out = unsafe { std::slice::from_raw_parts_mut(stripe.add(base + w * 64), 64) };
        transpose_8_u64s_to_64_bytes(row, out);
    }
}

/// V-wide [`add_carry_parts`]: per-instance `(sum, left, right, carry_aux)`.
#[inline(always)]
pub(crate) fn add_carry_parts_v(
    x: &[u32; BM_V],
    y: &[u32; BM_V],
) -> ([u32; BM_V], [u32; BM_V], [u32; BM_V], [u32; BM_V]) {
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let mut sum = [0u32; BM_V];
    let mut left = [0u32; BM_V];
    let mut right = [0u32; BM_V];
    let mut carry = [0u32; BM_V];
    for j in 0..BM_V {
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

/// Shared driver for the interleaved-row batch-major producers (sha2,
/// blake3 — the bit-packed encoders): parallel over V-instance groups, each
/// group builds its rows via `per_group(group_inputs, rows)` then NT-flushes
/// the useful chunks and transposes the stripe. Padding slots use `padding`
/// (required, matching the row-major driver's const-wire-pin behavior).
///
/// Returns `(z, a, b, stripe)`; z/a/b come from the scratch pool with the
/// padding suffix zeroed (the producers fully write the useful prefix).
pub(crate) fn drive_witness_batch_major<S: Sync, F>(
    inputs: &[S],
    padding: &S,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_group: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn([&S; BM_V], &mut [BmRow], &mut [BmRow], &mut [BmRow]) + Sync + Send,
{
    use rayon::prelude::*;

    let n_total = 1usize << n_blocks_log;
    assert!(inputs.len() <= n_total);
    assert!(n_total >= BM_V);
    let u64_per_block = (1usize << k_log) / 64;
    let useful_chunks = useful_bits.div_ceil(128);
    let useful_words = useful_bits.div_ceil(64);
    let total_f128 = n_total * (u64_per_block / 2);

    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let stripe = vec![0u8; n_total * u64_per_block * 8];
    // Zero the padding suffix (contiguous chunk-columns >= useful_chunks);
    // the producers fully rewrite the useful prefix every call.
    let tail = useful_chunks << n_blocks_log;
    for buf in [&mut z, &mut a, &mut b] {
        buf[tail..]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
    }

    let (zp, ap, bp) = (
        SendPtr(z.as_mut_ptr() as *mut u64),
        SendPtr(a.as_mut_ptr() as *mut u64),
        SendPtr(b.as_mut_ptr() as *mut u64),
    );
    let sp = SendPtr(stripe.as_ptr() as *mut u64);
    let inputs_ref = &inputs[..];

    (0..n_total / BM_V).into_par_iter().for_each_init(
        || {
            (
                vec![[0u64; BM_V]; u64_per_block],
                vec![[0u64; BM_V]; u64_per_block],
                vec![[0u64; BM_V]; u64_per_block],
            )
        },
        move |(rz, ra, rb), g| {
            rz[..useful_words].fill([0u64; BM_V]);
            ra[..useful_words].fill([0u64; BM_V]);
            rb[..useful_words].fill([0u64; BM_V]);
            let o0 = g * BM_V;
            let group: [&S; BM_V] = std::array::from_fn(|j| {
                inputs_ref.get(o0 + j).unwrap_or(padding)
            });
            per_group(group, rz, ra, rb);
            // SAFETY: disjoint instance ranges per group; suffix pre-zeroed.
            unsafe {
                flush_rows_nt(rz, zp.get(), o0, n_blocks_log, useful_chunks);
                flush_rows_nt(ra, ap.get(), o0, n_blocks_log, useful_chunks);
                flush_rows_nt(rb, bp.get(), o0, n_blocks_log, useful_chunks);
                stripe_from_rows(rz, sp.get() as *mut u8, o0, u64_per_block, useful_words);
            }
        },
    );

    (z, a, b, stripe)
}
