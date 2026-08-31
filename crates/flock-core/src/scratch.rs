//! Process-global pool for the prover's large transient `F128` buffers.
//!
//! Each prove allocates, faults in, and frees several 64–128 MB vectors
//! (the RS codeword, the round-2 fold outputs, the multilinear tail's
//! ping-pong scratch). The allocator returns such allocations to the OS on
//! free (`munmap`), so every prove re-pays soft page faults on first touch
//! and a single-threaded unmap on drop — a few ms per prove at m = 29 that
//! no kernel tuning can parallelize away.
//!
//! The pool recycles those buffers across phases and across proves: `take`
//! hands out a previously-used buffer when one with enough capacity exists,
//! `give` returns a buffer for later reuse. Contents are NOT cleared —
//! `take` has the same write-before-read contract as
//! [`crate::alloc_uninit_vec`].
//!
//! Steady-state retention is bounded by [`MAX_POOLED`] buffers (~640 MB for
//! the m = 29 prove set). Call [`clear`] to release everything to the OS,
//! e.g. after the last prove of a batch.

use crate::alloc_uninit_vec;
use crate::alloc_zeroed_vec;
use core::mem::size_of;
use core::ops::Range;
use rayon::prelude::*;
use std::env::var_os;
use std::mem::ManuallyDrop;
use std::ptr::write_bytes;

use crate::field::{F128, F256};
use std::sync::Mutex;

static POOL: Mutex<Vec<Vec<F128>>> = Mutex::new(Vec::new());

/// Max buffers retained. The m=29 prove cycle gives ~18 distinct buffers:
/// witness z/a/b, the L0 codeword, zerocheck's 2 fold outputs + 2 ping-pong
/// halves, ring-switch's per-claim rs_eq_ind vectors, b_combined, and
/// the PCS open's working buffers. Pooling ALL of the
/// open stage's transients matters beyond their own reuse: if they were
/// left to malloc while the earlier phases' buffers sat in the pool, the
/// open stage would fault fresh pages every prove (the pool denies malloc
/// the page reuse it would otherwise get from the freed early-phase
/// buffers) — measured as a +24% open_batch regression on M4 before this.
const MAX_POOLED: usize = 24;

/// Take a length-`n` `F128` vector, preferring a pooled buffer (smallest
/// capacity ≥ `n`); falls back to a fresh uninitialized allocation.
///
/// Contents are UNINITIALIZED in both cases — recycled buffers hold stale
/// data from a previous use. Caller MUST write every slot before reading it
/// (same contract as [`crate::alloc_uninit_vec`]).
pub fn take_f128(n: usize) -> Vec<F128> {
    if let Some(v) = try_take_f128(n) {
        return v;
    }
    alloc_uninit_vec(n)
}

/// Pool-only variant of [`take_f128`]: returns `None` instead of falling
/// back to a fresh allocation. Lets callers branch on warm-vs-cold (e.g.
/// the commit prefault skips its page-touch thread when the pool can
/// supply an already-resident buffer).
pub(crate) fn try_take_f128(n: usize) -> Option<Vec<F128>> {
    let mut pool = POOL.lock().unwrap();
    // Prefer a buffer within a 4x capacity window; fall back to any fitting
    // buffer. Per-take this is never worse than smallest-fitting alone: the
    // fallback IS the old policy. Measured (attribution probe, controlled
    // pairs): far-oversized idle buffers served small dense-domain requests
    // several times slower than right-class cycling ones (the nu14 combine
    // and nu18 Ligerito anomalies), while large requests still want the
    // oversized-but-resident fallback over a fresh allocation.
    let mut best: Option<usize> = None;
    let mut best_windowed: Option<usize> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() < n {
            continue;
        }
        if best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
            best = Some(i);
        }
        if v.capacity() < 4 * n.max(1)
            && best_windowed.is_none_or(|b| v.capacity() < pool[b].capacity())
        {
            best_windowed = Some(i);
        }
    }
    let best = best_windowed.or(best);
    if let Some(i) = best {
        let mut v = pool.swap_remove(i);
        drop(pool);
        if var_os("FLOCK_POOL_TRACE").is_some() {
            eprintln!(
                "      [pool] take_f128 n=2^{:.1} cap=2^{:.1} ({}x)",
                (n as f64).log2(),
                (v.capacity() as f64).log2(),
                v.capacity() / n.max(1),
            );
        }
        v.clear();
        // SAFETY: capacity ≥ n was checked above; F128: Copy (no Drop), so
        // exposing uninit/stale elements is sound to *hold* — the caller
        // upholds write-before-read per this function's contract.
        unsafe { v.set_len(n) };
        return Some(v);
    }
    if var_os("FLOCK_POOL_TRACE").is_some() {
        eprintln!(
            "      [pool] take_f128 n=2^{:.1} MISS (fresh)",
            (n as f64).log2()
        );
    }
    None
}

/// Return a buffer to the pool for reuse. When the pool is full, the
/// smallest-capacity buffer is evicted (large buffers are the expensive ones
/// to re-fault; a run that ramps problem sizes upward must not get its big
/// buffers crowded out by stale small ones).
///
/// TRIED AND REJECTED (2026-07-28): byte-budgeted eviction, largest-first, to
/// stop zerocheck's capacity-scaled giants (12.9 GB at nu = 18) from crowding
/// out the dense set the open cycles. A single-session A/B over budgets
/// 4 / 12 / 24 / 512 GB moved the nu = 18 steady total by NOTHING
/// (132.5 / 132.6 / 134.6 / 133.7; nu = 14 113.5 / 114.3 / 113.0 / 113.2).
/// Per-phase it is a zero-sum dial — shedding the giants takes the open's
/// oversized takes from 32-256x down to right-sized (`ligerito` -6 ms) and
/// hands the same back to the zerocheck tail, which then re-faults them.
/// Cross-session measurement made it look like a ~3 ms win; it is not one.
/// The residual is the capacity-scaled buffers EXISTING, which no eviction
/// policy can address — see the live-span note on zerocheck's fold output.
pub fn give_f128(v: Vec<F128>) {
    if v.capacity() == 0 {
        return;
    }
    let mut pool = POOL.lock().unwrap();
    pool.push(v);
    if pool.len() > MAX_POOLED {
        // Evict from the most-populated log2 size class (tie: the smallest
        // buffer in it). Always-evict-smallest let a prewarmed set of large
        // buffers permanently starve the actively-cycling smaller class —
        // every give of the hot size evicted the buffer just given.
        let class_of = |c: usize| usize::BITS - c.leading_zeros();
        let mut counts = [0u32; 65];
        for b in pool.iter() {
            counts[class_of(b.capacity()) as usize] += 1;
        }
        let crowded = (0..counts.len())
            .max_by_key(|&k| counts[k])
            .expect("non-empty");
        let victim = pool
            .iter()
            .enumerate()
            .filter(|(_, b)| class_of(b.capacity()) as usize == crowded)
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i)
            .expect("crowded class non-empty");
        pool.swap_remove(victim);
    }
}

/// [`take_f128`] for `F256` buffers, SHARING the F128 pool: an F256 array is
/// layout-identical to twice-as-many F128s (`repr(C)`, two `F128` fields,
/// align 16 — `Layout::array::<F256>(n) == Layout::array::<F128>(2n)`), so
/// the F256 fold chain recycles the same physical buffers the F128 phases
/// cycle (witness, codeword, ping-pong) instead of faulting fresh pages
/// every prove. A split pool was the shape of the measured "+24% open_batch"
/// trap in [`give_f128`]'s doc — do not separate them.
///
/// Same write-before-read contract as [`take_f128`].
pub fn take_f256(n: usize) -> Vec<F256> {
    if let Some(v) = try_take_f128(2 * n) {
        // Reinterpreting requires an even F128 capacity so the F256 vec's
        // drop layout (`Layout::array::<F256>(cap/2)`) matches the
        // allocation. Practically every pooled buffer is a power-of-two
        // size; an odd-capacity stray just goes back untouched.
        if v.capacity().is_multiple_of(2) {
            let mut v = ManuallyDrop::new(v);
            let (ptr, cap) = (v.as_mut_ptr(), v.capacity());
            // SAFETY: identical allocation layout (asserted even capacity);
            // both types are Copy PODs valid for every bit pattern; len n
            // ≤ cap/2 holds because take gave len 2n ≤ cap. Contents stay
            // uninitialized per the take contract.
            return unsafe { Vec::from_raw_parts(ptr as *mut F256, n, cap / 2) };
        }
        give_f128(v);
    }
    alloc_uninit_vec(n)
}

/// Return an `F256` buffer to the shared pool (see [`take_f256`]).
pub fn give_f256(v: Vec<F256>) {
    let mut v = ManuallyDrop::new(v);
    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
    // SAFETY: exact inverse of the reinterpretation in [`take_f256`] —
    // identical allocation layout, POD contents, doubled len/cap in F128
    // units.
    give_f128(unsafe { Vec::from_raw_parts(ptr as *mut F128, 2 * len, 2 * cap) });
}

// ---------------------------------------------------------------------------
// Zero pool: buffers KNOWN to be all-zero, for the padding-dominant witness
// shapes (every node-shaped circuit prove).
//
// `take_witness_buffers`' FreshZeroed branch used to `alloc_zeroed` fresh
// multi-GiB buffers per prove and never pool them. Early in a process that is
// lazy zero pages and nearly free; once the process has churned tens of GiB,
// `alloc_zeroed` gets recycled arena memory back from the allocator and
// memsets the whole request for real — measured as level-2 witgen going
// 35 → 590 ms in the SAME buffer mode. This pool keeps such buffers alive
// and ZERO: `give_zeroed_f128` re-zeros only the ranges the caller declares
// dirty (untouched lazy pages are never faulted), so a pooled buffer's
// zeroness is an invariant, not a per-take memset.
//
// Keyed by EXACT length: the prove shapes are stable per level, and exact
// matching sidesteps every question about zeroness beyond `len`.
// ---------------------------------------------------------------------------

static ZERO_POOL: Mutex<Vec<Vec<F128>>> = Mutex::new(Vec::new());

/// Max zero-pool buffers retained. A mixed prove cycles 5 per shape (witness
/// z/a/b + the element region pa/pb); the recursion tower runs three shapes
/// (leaf, level-1, level-2), but only the shape being proven cycles — the cap
/// covers two shapes' sets so a level's children and the level itself coexist.
const MAX_POOLED_ZERO: usize = 10;

/// Take a length-`n` all-zero `F128` vector: a pooled buffer of EXACTLY this
/// length when one exists (already zero — no memset, no faults), else a fresh
/// [`crate::alloc_zeroed_vec`] (lazy OS zero pages).
///
/// Return it with [`give_zeroed_f128`], declaring every range that may have
/// been written; the pool's invariant is that stored buffers are all-zero.
pub fn take_zeroed_f128(n: usize) -> Vec<F128> {
    let mut pool = ZERO_POOL.lock().unwrap();
    if let Some(i) = pool.iter().position(|v| v.len() == n) {
        let v = pool.swap_remove(i);
        drop(pool);
        if var_os("FLOCK_POOL_TRACE").is_some() {
            eprintln!("      [pool] take_zeroed n=2^{:.1} HIT", (n as f64).log2());
        }
        return v;
    }
    drop(pool);
    if var_os("FLOCK_POOL_TRACE").is_some() {
        eprintln!(
            "      [pool] take_zeroed n=2^{:.1} MISS (fresh lazy-zero)",
            (n as f64).log2()
        );
    }
    alloc_zeroed_vec(n)
}

/// Return an all-zero-except-`dirty` buffer to the zero pool: re-zero exactly
/// the declared `dirty` ranges (parallel memset — resident pages only, since
/// a range that was written is resident and one that was not needs no write),
/// then store the buffer with its zeroness remembered.
///
/// The caller CERTIFIES that every write since [`take_zeroed_f128`] fell
/// inside `dirty`; a stray write outside them would poison the pool. The two
/// callers make this structural rather than promised: the union witness
/// buffers only ever hand out their slot blocks (`slot_dests` carves exactly
/// those, borrow-checked), and `copy_live_region` writes exactly the live
/// spans its give-back re-zeros. Debug builds verify the invariant outright.
pub fn give_zeroed_f128(mut v: Vec<F128>, dirty: &[Range<usize>]) {
    if v.capacity() == 0 {
        return;
    }
    for r in dirty {
        v[r.start..r.end]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
    }
    debug_assert!(
        v.iter().all(|w| w.is_zero()),
        "a zero-pool give-back held nonzero words outside its declared dirty \
         ranges — the caller's dirty accounting is unsound"
    );
    let mut pool = ZERO_POOL.lock().unwrap();
    pool.push(v);
    if pool.len() > MAX_POOLED_ZERO {
        // Evict the smallest buffer: large ones are the expensive ones to
        // re-fault and re-zero (same rationale as `give_f128`).
        let victim = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i)
            .expect("pool non-empty");
        pool.swap_remove(victim);
    }
}

// ---------------------------------------------------------------------------
// Byte pool, for the lincheck stripe.
//
// The stripe is the drivers' fourth output and is as large as the packed
// witness itself (134 MB at m = 30). `vec![0u8; n]` gets zero pages from the
// OS cheaply, but every page still soft-faults on first touch during the
// transpose — measured at ~0.8 ms per 134 MB, paid on every prove. Recycling
// resident buffers removes that; callers zero only the region they do not
// write (the stripe's per-group tail rows), which is a few percent.
// ---------------------------------------------------------------------------

static U8_POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Max byte buffers retained. One stripe per slot in flight, plus headroom.
const MAX_POOLED_U8: usize = 4;

/// Take a length-`n` byte vector, preferring a pooled buffer (smallest
/// capacity ≥ `n`); falls back to a fresh zeroed allocation.
///
/// Contents are UNSPECIFIED when a pooled buffer is returned — stale bytes
/// from a previous use. The caller MUST write or explicitly zero every byte it
/// later reads (same contract as [`take_f128`]).
pub fn take_u8(n: usize) -> Vec<u8> {
    let mut pool = U8_POOL.lock().unwrap();
    let mut best: Option<usize> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() >= n && best.is_none_or(|b| v.capacity() < pool[b].capacity()) {
            best = Some(i);
        }
    }
    if let Some(i) = best {
        let mut v = pool.swap_remove(i);
        drop(pool);
        v.clear();
        // SAFETY: capacity ≥ n checked above; u8 has no Drop and every bit
        // pattern is valid, so exposing stale bytes is sound to hold — the
        // caller upholds write-or-zero-before-read per this contract.
        unsafe { v.set_len(n) };
        return v;
    }
    drop(pool);
    vec![0u8; n]
}

/// Return a byte buffer for reuse. Smallest-capacity eviction, as
/// [`give_f128`].
pub fn give_u8(v: Vec<u8>) {
    if v.capacity() == 0 {
        return;
    }
    let mut pool = U8_POOL.lock().unwrap();
    pool.push(v);
    if pool.len() > MAX_POOLED_U8 {
        let smallest = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.capacity())
            .map(|(i, _)| i)
            .expect("pool non-empty");
        pool.swap_remove(smallest);
    }
}

/// Pre-warm the pool for proves at witness size `2^m`: allocate and
/// first-touch the full prove-cycle buffer set once, in parallel, then park
/// it in the pool. Called from the per-hash Setup constructors, this moves
/// ALL page-fault cost off the prove path — including the first prove — so
/// proving performs no memory-management syscalls on any machine. (This is
/// the machine-independent alternative to overlapping the faults with other
/// work: a race between fault cost and the hiding window flips sign across
/// machines; eliminated work doesn't.)
///
/// The set (sizes in F128s): 2^(m-6)-class — L0 codeword, zerocheck round-2
/// a/b, open-stage codeword ping-pong ×2 → 5 buffers; 2^(m-7)-class — witness
/// z/a/b, zerocheck tail ping-pong ×2, open-stage transients, rs_eq_ind ×2,
/// b_combined → 11 buffers. ~1.1 GB resident at m = 29; release with
/// [`clear`].
///
/// Sized for a UNIFORM buffer set, i.e. the single-table paths where the
/// padded `m` drives every class. The merged/union path is not uniform in
/// `m` (its commit/open stack is count-derived, not capacity-derived); a
/// union-shaped prewarm would list per-class sets via `prewarm_sets`. The one
/// that existed was deleted with the union probes that called it.
pub fn prewarm_prover(m: usize) {
    prewarm_sets(&[(m, 5, 11)]);
}

/// Byte budget for first-touched prewarm pages. Prewarming is only ever a
/// win while the touched set stays comfortably resident: past that the OS
/// starts compressing and reclaiming, and the prewarm costs more than the
/// faults it was meant to remove. MEASURED on the m30 load (one-shot prove,
/// median of 3): touching the full padded-`m` set reads 10.6 GB / 399 ms at
/// nu = 16, 21 GB / 477 ms at nu = 17 and 45 GB / 604 ms at nu = 18 — against
/// 450-460 ms for no prewarm at all. So the win inverts between 10 and 21 GB
/// on a 36 GB box; 8 GB keeps a margin and preserves every tier that was
/// already winning.
const PREWARM_BUDGET_BYTES: usize = 8 << 30;

/// Allocate and first-touch `(m, n_large, n_small)` buffer sets in order,
/// stopping once the cumulative touched bytes would exceed
/// [`PREWARM_BUDGET_BYTES`]. Earlier sets have priority, so callers list the
/// always-worth-it ones first. Buffers allocated past the budget are skipped
/// entirely rather than allocated-but-untouched: an untouched pooled buffer
/// would still be handed out by `take` and fault during the prove, which is
/// exactly the cost this is trying to place off the prove path.
fn prewarm_sets(sets: &[(usize, usize, usize)]) {
    let mut bufs: Vec<Vec<F128>> = Vec::new();
    let mut budget = PREWARM_BUDGET_BYTES;
    let w = size_of::<F128>();
    for &(m, n_large, n_small) in sets {
        if m < 7 {
            continue;
        }
        for (count, len) in [(n_large, 1usize << (m - 6)), (n_small, 1usize << (m - 7))] {
            for _ in 0..count {
                let bytes = len * w;
                if bytes > budget {
                    break;
                }
                budget -= bytes;
                bufs.push(take_f128(len));
            }
        }
    }
    // First-touch every page of every buffer, all cores. Already-resident
    // (re-warmed) buffers cost a fast memset; fresh ones fault here, once.
    bufs.par_iter_mut().for_each(|b| {
        b.par_chunks_mut(1 << 16).for_each(|chunk| {
            // SAFETY: F128 is plain bytes (no Drop); zero is a valid pattern.
            unsafe { write_bytes(chunk.as_mut_ptr(), 0u8, chunk.len()) }
        });
    });
    for b in bufs {
        give_f128(b);
    }
}

/// Release every pooled buffer back to the OS.
pub fn clear() {
    POOL.lock().unwrap().clear();
    U8_POOL.lock().unwrap().clear();
    ZERO_POOL.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_reuses_given_buffer() {
        clear();
        let mut v = take_f128(1024);
        for slot in v.iter_mut() {
            *slot = F128 { lo: 7, hi: 9 };
        }
        let ptr = v.as_ptr();
        give_f128(v);
        // Same capacity request gets the same allocation back.
        let v2 = take_f128(512);
        assert_eq!(v2.as_ptr(), ptr);
        assert_eq!(v2.len(), 512);
        clear();
    }

    #[test]
    fn zero_pool_round_trips_rezeroed_buffers() {
        clear();
        let mut v = take_zeroed_f128(1024);
        assert!(v.iter().all(|w| w.is_zero()), "fresh take is zero");
        // Dirty two disjoint ranges, declare them, and take the buffer back.
        for i in 100..200 {
            v[i] = F128 { lo: 1, hi: 2 };
        }
        for i in 700..1024 {
            v[i] = F128 { lo: 3, hi: 4 };
        }
        let ptr = v.as_ptr();
        give_zeroed_f128(v, &[100..200, 700..1024]);
        let v2 = take_zeroed_f128(1024);
        assert_eq!(v2.as_ptr(), ptr, "exact-length key returns the buffer");
        assert!(v2.iter().all(|w| w.is_zero()), "give-back re-zeroed it");
        // A different length misses and allocates fresh.
        let v3 = take_zeroed_f128(512);
        assert!(v3.iter().all(|w| w.is_zero()));
        assert_ne!(v3.as_ptr(), ptr);
        clear();
    }

    #[test]
    fn pool_is_bounded() {
        clear();
        for _ in 0..(MAX_POOLED + 4) {
            give_f128(take_f128(16));
        }
        assert!(POOL.lock().unwrap().len() <= MAX_POOLED);
        clear();
    }
}
