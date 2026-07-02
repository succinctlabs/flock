//! E1 — witness producers: row-major baseline vs staged L1′ scatter.
//! Hash-generic: parameterized by `k_log` and a per-block builder closure
//! (same seam as `drive_witness_packed_and_lincheck`).
//!
//! Variants (identical output modulo layout; only the write pattern differs):
//!
//! - [`build_row_major`]: the production pattern — parallel groups, each
//!   instance one contiguous `2^k_log`-bit run.
//! - [`build_l1_staged_opts_nt`]: the L1′ producer — stage G instances in a
//!   worker-local row-major buffer, flush per 128-bit chunk index as
//!   contiguous `16·G`-byte runs at dest word `(c << n_log) | instance`.
//!   Knobs: `useful_chunks` (skip the padding suffix), fused lincheck
//!   stripe, non-temporal stores. No global transpose is ever materialized.
//! - [`build_compute_only`]: staging without flushing — the compute floor.

use rayon::prelude::*;

/// Per-block builder closure shape (same as the production driver's seam).
pub trait PerBlock<S>: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync {}
impl<S, F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync> PerBlock<S> for F {}

/// Raw-pointer wrapper for the disjoint strided writes of the L1′ flush.
#[derive(Copy, Clone)]
struct SendPtr(*mut u64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Worker-local staging for one group of `group` instances (row-major).
struct Staging {
    z: Vec<u64>,
    a: Vec<u64>,
    b: Vec<u64>,
    u64_per_block: usize,
}

impl Staging {
    fn new(group: usize, u64_per_block: usize) -> Self {
        Self {
            z: vec![0u64; group * u64_per_block],
            a: vec![0u64; group * u64_per_block],
            b: vec![0u64; group * u64_per_block],
            u64_per_block,
        }
    }
    fn zero(&mut self) {
        self.z.fill(0);
        self.a.fill(0);
        self.b.fill(0);
    }
    fn build<S>(&mut self, inputs: &[S], base: usize, group: usize, per_block: &impl PerBlock<S>) {
        let w = self.u64_per_block;
        for g in 0..group {
            per_block(
                &inputs[base + g],
                &mut self.z[g * w..(g + 1) * w],
                &mut self.a[g * w..(g + 1) * w],
                &mut self.b[g * w..(g + 1) * w],
            );
        }
    }
}

fn check_sizes<S>(inputs: &[S], k_log: usize, n_log: usize, group: usize, bufs: [&[u64]; 3]) {
    assert_eq!(inputs.len(), 1usize << n_log, "need exactly 2^n_log inputs");
    assert!(group.is_power_of_two() && group <= inputs.len());
    let total = (1usize << n_log) * ((1usize << k_log) / 64);
    for b in bufs {
        assert_eq!(b.len(), total);
    }
}

/// Baseline: row-major output, contiguous per-instance writes, parallel over
/// groups of `group` instances (production uses 8).
pub fn build_row_major<S: Sync>(
    inputs: &[S],
    k_log: usize,
    n_log: usize,
    group: usize,
    per_block: &impl PerBlock<S>,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    build_row_major_with_stripe(inputs, k_log, n_log, group, per_block, None, z, a, b)
}

/// Row-major baseline with the lincheck stripe built exactly the way the
/// production driver does (per 8-instance group, from the just-written L1-hot
/// z), but with **caller-recycled buffers** — the fairest possible version of
/// today's layout, with the driver's per-call stripe allocation removed.
pub fn build_row_major_with_stripe<S: Sync>(
    inputs: &[S],
    k_log: usize,
    n_log: usize,
    group: usize,
    per_block: &impl PerBlock<S>,
    stripe: Option<&mut [u8]>,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    use flock_core::bits::transpose_8_u64s_to_64_bytes;

    check_sizes(inputs, k_log, n_log, group, [z, a, b]);
    let w = (1usize << k_log) / 64;
    let gw = group * w;
    assert!(stripe.is_none() || group.is_multiple_of(8));
    let stripe_group_bytes = (group / 8) * w * 64;
    let mut stripe_chunks: Vec<Option<&mut [u8]>> = match stripe {
        Some(s) => {
            assert_eq!(s.len(), (inputs.len() / 8) * w * 64);
            s.chunks_mut(stripe_group_bytes).map(Some).collect()
        }
        None => (0..inputs.len() / group).map(|_| None).collect(),
    };
    z.par_chunks_mut(gw)
        .zip(a.par_chunks_mut(gw))
        .zip(b.par_chunks_mut(gw))
        .zip(stripe_chunks.par_iter_mut())
        .enumerate()
        .for_each(|(gi, (((zg, ag), bg), sg))| {
            zg.fill(0);
            ag.fill(0);
            bg.fill(0);
            for g in 0..group {
                per_block(
                    &inputs[gi * group + g],
                    &mut zg[g * w..(g + 1) * w],
                    &mut ag[g * w..(g + 1) * w],
                    &mut bg[g * w..(g + 1) * w],
                );
            }
            if let Some(sg) = sg {
                // Same transpose as drive_witness_packed_and_lincheck: per
                // 8-instance subgroup, lanes = z word i of each instance.
                for g8 in 0..group / 8 {
                    let base = g8 * 8 * w;
                    let sbase = g8 * w * 64;
                    for i in 0..w {
                        let lanes: [u64; 8] = std::array::from_fn(|s| zg[base + s * w + i]);
                        transpose_8_u64s_to_64_bytes(&lanes, &mut sg[sbase + i * 64..sbase + i * 64 + 64]);
                    }
                }
            }
        });
}

/// L1′ producer with all knobs. Output layout: word index `(c << n_log) | o`
/// for chunk `c` of instance `o` (each word = 2 consecutive u64s).
pub fn build_l1_staged_opts_nt<S: Sync>(
    inputs: &[S],
    k_log: usize,
    n_log: usize,
    group: usize,
    useful_chunks: usize,
    stripe: Option<&mut [u8]>,
    nt_stores: bool,
    per_block: &impl PerBlock<S>,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    use flock_core::bits::transpose_8_u64s_to_64_bytes;

    check_sizes(inputs, k_log, n_log, group, [z, a, b]);
    let u64_per_block = (1usize << k_log) / 64;
    let chunks_per_block = u64_per_block / 2;
    assert!(useful_chunks <= chunks_per_block);
    assert!(group >= 8 || stripe.is_none(), "stripe build needs group >= 8");
    let (zp, ap, bp) = (
        SendPtr(z.as_mut_ptr()),
        SendPtr(a.as_mut_ptr()),
        SendPtr(b.as_mut_ptr()),
    );
    let stripe_ptr = stripe.map(|s| {
        // One stripe per 8 instances, 2^k_log bytes each (matches
        // drive_witness_packed_and_lincheck's z_lincheck).
        assert_eq!(s.len(), (inputs.len() / 8) * u64_per_block * 64);
        SendPtr(s.as_mut_ptr() as *mut u64)
    });
    let n_groups = inputs.len() / group;
    (0..n_groups)
        .into_par_iter()
        .for_each_init(
            || Staging::new(group, u64_per_block),
            |st, gi| {
                st.zero();
                st.build(inputs, gi * group, group, per_block);
                let g0 = gi * group;
                // Flush: for each chunk index, one contiguous 2·group-u64 run
                // at dest word (c << n_log) + g0. Staging reads are strided
                // but cache-resident. SAFETY: distinct groups write disjoint
                // u64 ranges — for any chunk c the run
                // [((c << n_log) + g0)·2, +2·group) only overlaps runs with
                // the same g0.
                unsafe {
                    for (src, dstp) in [(&st.z, zp), (&st.a, ap), (&st.b, bp)] {
                        let s = src.as_ptr();
                        for c in 0..useful_chunks {
                            let dst = dstp.0.add(((c << n_log) + g0) * 2);
                            #[cfg(target_arch = "aarch64")]
                            if nt_stores {
                                // 32 B per stnp: two instances' 16 B chunks.
                                for g in (0..group).step_by(2) {
                                    let s0 = s.add(g * u64_per_block + 2 * c);
                                    let s1 = s.add((g + 1) * u64_per_block + 2 * c);
                                    let d = dst.add(2 * g);
                                    std::arch::asm!(
                                        "ldr {q0:q}, [{s0}]",
                                        "ldr {q1:q}, [{s1}]",
                                        "stnp {q0:q}, {q1:q}, [{d}]",
                                        s0 = in(reg) s0,
                                        s1 = in(reg) s1,
                                        d = in(reg) d,
                                        q0 = out(vreg) _,
                                        q1 = out(vreg) _,
                                        options(nostack),
                                    );
                                }
                                continue;
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            let _ = nt_stores;
                            for g in 0..group {
                                *dst.add(2 * g) = *s.add(g * u64_per_block + 2 * c);
                                *dst.add(2 * g + 1) = *s.add(g * u64_per_block + 2 * c + 1);
                            }
                        }
                    }
                    // Lincheck byte-stripe from staging (same transpose the
                    // production driver runs; stripes are group-contiguous so
                    // writes are disjoint per group).
                    if let Some(sp) = stripe_ptr {
                        let sp = sp.0 as *mut u8;
                        for g8 in 0..group / 8 {
                            let stripe_base = (g0 / 8 + g8) * u64_per_block * 64;
                            for i in 0..u64_per_block {
                                let lanes: [u64; 8] = std::array::from_fn(|s| {
                                    st.z[(g8 * 8 + s) * u64_per_block + i]
                                });
                                let out = std::slice::from_raw_parts_mut(
                                    sp.add(stripe_base + i * 64),
                                    64,
                                );
                                transpose_8_u64s_to_64_bytes(&lanes, out);
                            }
                        }
                    }
                }
            },
        );
}

/// Compute floor: staged build with no flush.
pub fn build_compute_only<S: Sync>(
    inputs: &[S],
    k_log: usize,
    n_log: usize,
    group: usize,
    per_block: &impl PerBlock<S>,
) {
    assert_eq!(inputs.len(), 1usize << n_log);
    let u64_per_block = (1usize << k_log) / 64;
    let n_groups = inputs.len() / group;
    (0..n_groups).into_par_iter().for_each_init(
        || Staging::new(group, u64_per_block),
        |st, gi| {
            st.zero();
            st.build(inputs, gi * group, group, per_block);
            std::hint::black_box(&st.z[gi % st.z.len()]);
        },
    );
}

/// Mechanical group-width rule (stated in the plan's methodology): wide
/// enough for long flush runs, small enough to keep every worker fed.
pub fn auto_group(n_instances: usize, threads: usize) -> usize {
    let feed = (n_instances / (2 * threads.max(1))).max(8);
    feed.next_power_of_two().min(64).min(n_instances)
}
