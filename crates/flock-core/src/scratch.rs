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

use crate::field::F128;
use core::mem::size_of;
use std::env::var_os;
use std::ptr::write_bytes;
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
    crate::alloc_uninit_vec(n)
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
/// `m` — use [`prewarm_prover_union`].
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

/// Prewarm for a MERGED/UNION prove, whose buffer set is NOT uniform in `m`.
///
/// Only witgen (3 small) and zerocheck (2 large + 2 small) scale with the
/// padded capacity; commit, compaction and the whole open work on the
/// count-derived DENSE stack, which is the same size at every capacity. Both
/// sets are prewarmed — the dense one first, since it is small and always
/// pays — and the capacity set is included only while it fits
/// [`PREWARM_BUDGET_BYTES`].
///
/// Sizing the WHOLE set off the padded `m` (what [`prewarm_prover`] does, and
/// what the union probes used to call) inverts the point of prewarming at high
/// capacity: it force-touches the padding this pipeline exists to skip. The
/// capacity-scaled buffers are `alloc_uninit` and written under the sparse
/// contract, so their padded pages are otherwise NEVER faulted in — at
/// nu = 18 the prove touches ~3 GB of the ~19 GB it allocates.
pub fn prewarm_prover_union(m_capacity: usize, m_dense: usize) {
    // Dense classes always; capacity classes in the counts actually taken
    // (zerocheck 2 large + 2 small, witgen 3 small) when they fit the budget.
    if m_capacity > m_dense {
        prewarm_sets(&[(m_dense, 5, 11), (m_capacity, 2, 5)]);
    } else {
        prewarm_sets(&[(m_dense, 5, 11)]);
    }
}

/// Allocate and first-touch `(m, n_large, n_small)` buffer sets in order,
/// stopping once the cumulative touched bytes would exceed
/// [`PREWARM_BUDGET_BYTES`]. Earlier sets have priority, so callers list the
/// always-worth-it ones first. Buffers allocated past the budget are skipped
/// entirely rather than allocated-but-untouched: an untouched pooled buffer
/// would still be handed out by `take` and fault during the prove, which is
/// exactly the cost this is trying to place off the prove path.
fn prewarm_sets(sets: &[(usize, usize, usize)]) {
    use rayon::prelude::*;
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
    fn pool_is_bounded() {
        clear();
        for _ in 0..(MAX_POOLED + 4) {
            give_f128(take_f128(16));
        }
        assert!(POOL.lock().unwrap().len() <= MAX_POOLED);
        clear();
    }
}
