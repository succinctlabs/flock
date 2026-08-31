#[allow(unused_imports)] // used only under cfg(test)
use super::*;
#[cfg(test)]
use bincode::serialize;
#[cfg(test)]
use flock_core::{
    pcs::{MergedOpenProof, ligerito::RecursiveProof},
    proof::R1csProofCircuitMerged,
};
#[cfg(test)]
use serde::Serialize;

// ---------------------------------------------------------------------------
// THE BENCHMARK CONTRACT: what a proof costs ONLINE.
//
// ONLINE is per-STATEMENT work — everything a prover pays again for the
// next segment of the chain, the next pair of children:
//
//   walk    the circuit's evaluation over this statement (for a chain leaf
//           this IS the sequential hashing; reported apart from proving)
//   tapes   the child tape sources: recorded DEFERRED child verifies, the
//           production statement work (the pin/locate/replica scaffolding
//           around them in these tests is not this — it is per shape)
//   witgen  witness/trace generation and packing into the union's blocks
//   prove
//
// SETUP is per-SHAPE and cacheable, so it is timed separately and never
// folded into a per-proof number: the circuit emit+finish, the R1CS tables,
// the union and PCS params, the fill plan, the tape pins. A shape is
// statement-independent (the digest pins say so), so a production prover
// pays it once per level and then never again.
// Populated by the pub builders; READ only by the in-file `#[test]` benches
// (`tower_online_bench` and friends), so the lib unit sees the fields unread.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Default)]
pub(super) struct Online {
    pub(super) setup_ms: f64,
    pub(super) walk_ms: f64,
    pub(super) tapes_ms: f64,
    pub(super) witgen_ms: f64,
    pub(super) prove_ms: f64,
    pub(super) verify_ms: f64,
    /// ONE timer around the whole online span (walk through prove), not the
    /// phase sum: everything between the phases — the union/PCS param
    /// construction, buffer drops, allocator work — lands here and nowhere
    /// else. `0.0` where a stage has not been wired for it.
    pub(super) wall_ms: f64,
}

#[cfg(test)]
impl Online {
    /// The per-proof online total. The MEASURED wall where a stage supplies
    /// it, the phase sum otherwise — a sum can only be a lower bound.
    pub(super) fn total(&self) -> f64 {
        if self.wall_ms > 0.0 {
            self.wall_ms
        } else {
            self.walk_ms + self.tapes_ms + self.witgen_ms + self.prove_ms
        }
    }

    /// What the phases add up to — printed beside the wall so the gap between
    /// them is visible rather than assumed away.
    fn summed(&self) -> f64 {
        self.walk_ms + self.tapes_ms + self.witgen_ms + self.prove_ms
    }
}

#[cfg(test)]
pub(super) fn median_of(runs: &[Online], f: impl Fn(&Online) -> f64) -> f64 {
    let mut v: Vec<f64> = runs.iter().map(&f).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[cfg(test)]
pub(super) fn median_total(runs: &[Online]) -> f64 {
    median_of(runs, |o| o.total())
}

/// One stage's ONLINE line: per-phase medians, the total's median and
/// range, then the per-SHAPE setup for reference. Medians, not means —
/// the first run of any stage pays first-touch allocator costs that are
/// warmup, not marginal cost (the recorded L2 lesson).
/// **WHERE A PROOF'S BYTES ARE.** Serialized size per component, so that any
/// shrinking effort is steered by the census rather than by intuition about
/// which piece looks big. Sizes are `bincode` lengths of the sub-structures;
/// they sum to slightly less than the whole (the outer struct's own tags).
///
/// The interesting ratio is proof bytes vs what they cost a PARENT: the
/// parent replays the child's transcript through its b3 slot at one
/// compression per 64 bytes, so at the measured ~6.1 µs per b3 row a KiB of
/// child proof is ~0.1 ms of parent per child.
#[cfg(test)]
pub(super) fn census_kib<T: Serialize + ?Sized>(v: &T) -> f64 {
    serialize(v).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0
}

#[cfg(test)]
pub(super) fn proof_census(label: &str, p: &R1csProofCircuitMerged, pcs: &PcsParams) {
    proof_census_parts(
        label,
        census_kib(p),
        census_kib(&p.boolean),
        census_kib(&p.element),
        census_kib(&p.wiring),
        &p.pcs_open,
        pcs,
    )
}

/// [`proof_census`] over the leaf's flavor enum — the AG flavor differs
/// only in the boolean PIOP struct; every other section is shape-shared.
#[cfg(test)]
pub(super) fn proof_census_mixed(label: &str, p: &MixedProof, pcs: &PcsParams) {
    match p {
        MixedProof::Rs(p) => proof_census(label, p, pcs),
        MixedProof::Ag(p) => proof_census_parts(
            label,
            census_kib(p),
            census_kib(&p.boolean),
            census_kib(&p.element),
            census_kib(&p.wiring),
            &p.pcs_open,
            pcs,
        ),
    }
}

#[cfg(test)]
pub(super) fn proof_census_parts(
    label: &str,
    total: f64,
    boolean_kib: f64,
    element_kib: f64,
    wiring_kib: f64,
    pcs_open: &MergedOpenProof,
    pcs: &PcsParams,
) {
    // Per-level stratified schedules, and the siblings a path emits ABOVE
    // the cap layer. The cap is the whole layer at the DEEPEST summand's
    // depth c1, so every query can be checked with d - c1 siblings — since
    // truncation, that is all the prover emits. MEASURED from the proof
    // (not recomputed from the schedule), so `redundant` certifies the
    // truncation stays landed: anything past q·(d − c1) is waste.
    if let Ok(cfg) = pcs.ligerito_prover_config() {
        let lig = &pcs_open.inner.ligerito;
        let r = lig.recursive_caps.len();
        assert_eq!(cfg.stratified.len(), r + 1, "one schedule per open level");
        let level_paths = |lvl: usize| -> usize {
            if lvl == 0 {
                lig.initial_proof.merkle_proof.len()
            } else if lvl < r {
                lig.recursive_proofs[lvl - 1].merkle_proof.len()
            } else {
                lig.final_proof.merkle_proof.len()
            }
        };
        let (mut waste, mut emitted) = (0usize, 0usize);
        let mut per_level: Vec<String> = Vec::new();
        for (lvl, sch) in cfg.stratified.iter().enumerate() {
            let c1 = sch.cap_depth();
            let e = level_paths(lvl);
            let w = e - sch.queries() * (sch.log_block_len - c1);
            per_level.push(format!(
                "L{lvl}: q={} depths={:?} cap={c1} sibs={e} redundant={w}",
                sch.queries(),
                sch.summand_depths,
            ));
            waste += w;
            emitted += e;
        }
        println!(
            "\n  STRATIFIED PATHS — {label}\n    {}\n    \
             emitted {emitted} siblings, {waste} redundant above the cap \
             ({:.1} KiB of {:.1} KiB, {:.0}%)",
            per_level.join("\n    "),
            waste as f64 * 32.0 / 1024.0,
            emitted as f64 * 32.0 / 1024.0,
            100.0 * waste as f64 / emitted as f64,
        );
    }
    let sz = |b: Result<Vec<u8>, _>| b.map(|v| v.len()).unwrap_or(0) as f64 / 1024.0;
    let lig = &pcs_open.inner.ligerito;
    let rows = |v: &Vec<RecursiveProof>| -> (f64, f64) {
        (
            v.iter().map(|r| sz(serialize(&r.opened_rows))).sum(),
            v.iter().map(|r| sz(serialize(&r.merkle_proof))).sum(),
        )
    };
    let (rec_rows, rec_paths) = rows(&lig.recursive_proofs);
    let l0_rows = sz(serialize(&lig.initial_proof.opened_rows));
    let l0_paths = sz(serialize(&lig.initial_proof.merkle_proof));
    println!(
        "\n  PROOF CENSUS — {label}: {total:.1} KiB\n\
         \x20   boolean PIOP        {:6.1}\n\
         \x20   element PIOP        {:6.1}\n\
         \x20   wiring              {:6.1}\n\
         \x20   merged rounds       {:6.1}\n\
         \x20   ring switches       {:6.1}\n\
         \x20   multipoint values   {:6.1}   (128 per rs claim)\n\
         \x20   multipoint rounds   {:6.1}\n\
         \x20   multipoint anchor   {:6.1}\n\
         \x20   inner: L0 rows      {:6.1}\n\
         \x20   inner: L0 paths     {:6.1}\n\
         \x20   inner: rec rows     {:6.1}\n\
         \x20   inner: rec paths    {:6.1}\n\
         \x20   inner: caps         {:6.1}   (L0 {:.1} + rec {:.1})\n\
         \x20   inner: final block  {:6.1}\n\
         \x20   inner: sumcheck     {:6.1}",
        boolean_kib,
        element_kib,
        wiring_kib,
        sz(serialize(&pcs_open.merged_rounds)),
        sz(serialize(&pcs_open.ring_switches)),
        sz(serialize(&pcs_open.frobenius.values)),
        sz(serialize(&pcs_open.frobenius.rounds)),
        sz(serialize(&pcs_open.frobenius.anchor)),
        l0_rows,
        l0_paths,
        rec_rows,
        rec_paths,
        sz(serialize(&lig.initial_cap)) + sz(serialize(&lig.recursive_caps)),
        sz(serialize(&lig.initial_cap)),
        sz(serialize(&lig.recursive_caps)),
        sz(serialize(&lig.final_proof)),
        sz(serialize(&lig.sumcheck_transcript)),
    );
}

#[cfg(test)]
pub(super) fn report_stage(name: &str, runs: &[Online]) {
    let mut tot: Vec<f64> = runs.iter().map(|o| o.total()).collect();
    tot.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "    {name:9} walk {:6.1} + tapes {:5.1} + witgen {:5.1} + prove {:7.1} \
         = {:7.1} ms [{:.1}-{:.1}] | verify {:4.1} | (setup {:.0})",
        median_of(runs, |o| o.walk_ms),
        median_of(runs, |o| o.tapes_ms),
        median_of(runs, |o| o.witgen_ms),
        median_of(runs, |o| o.prove_ms),
        tot[tot.len() / 2],
        tot[0],
        tot[tot.len() - 1],
        median_of(runs, |o| o.verify_ms),
        median_of(runs, |o| o.setup_ms),
    );
    // Where a stage measures its wall directly, print what the phases add up
    // to beside it: the difference is real per-proof cost that no phase timer
    // owns, and quoting the sum alone hides it.
    if runs.iter().any(|o| o.wall_ms > 0.0) {
        let (wall, summed) = (
            median_of(runs, |o| o.wall_ms),
            median_of(runs, |o| o.summed()),
        );
        println!(
            "    {:9} MEASURED wall {:7.1} ms vs phase sum {:7.1} ({:+.1} unaccounted)",
            "",
            wall,
            summed,
            wall - summed,
        );
    }
}
