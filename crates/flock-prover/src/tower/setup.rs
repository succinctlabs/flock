//! **THE SETUP PHASE**: what a tower build needs that depends only on the
//! SHAPE of what it builds, never on the statement.
//!
//! Every FL has one circuit, and so do the base node and every spine node
//! (the driver asserts the spine's convergence). The first build of a kind
//! finishes its circuit (`ShapeBuilder::finish`: registry, cell space,
//! wiring, digest) and compiles its fill plan. Every later build of that
//! kind still emits its circuit, because the emission computes the
//! statement's input values, but takes the finished shape and the plan from
//! here. The fill runner checks the inputs and hints against the shape, so
//! a stale shape fails loudly. Under [`SelfCheck::Full`] the builder also
//! finishes its own circuit and asserts the digests agree.
//!
//! [`SelfCheck::Full`]: crate::tower::SelfCheck::Full

use std::sync::{Arc, OnceLock};

use flock_core::circuit::builder::{CircuitShape, FillPlan, ShapeBuilder};

use crate::tower::SelfCheck;

/// One kind's finished shape and fill plan.
pub(super) struct OuterShape {
    pub(super) shape: Arc<CircuitShape>,
    pub(super) fill_plan: Arc<FillPlan>,
}

/// The cache slot for one kind of build. A fresh slot makes the builder
/// behave as if there were no setup phase.
#[derive(Default)]
pub struct ShapeSlot(OnceLock<OuterShape>);

impl ShapeSlot {
    /// The finished shape for `sb`: the cached one when this kind has been
    /// built before, else `sb`'s own (which then fills the slot).
    pub(super) fn finish(
        &self,
        sb: ShapeBuilder,
        checks: SelfCheck,
        what: &str,
    ) -> (Arc<CircuitShape>, Arc<FillPlan>) {
        if let Some(c) = self.0.get() {
            if checks.on() {
                let own = sb.finish().expect("the circuit builds");
                assert_eq!(
                    own.circuit.digest(),
                    c.shape.circuit.digest(),
                    "{what}: the cached shape is this build's shape"
                );
            }
            return (c.shape.clone(), c.fill_plan.clone());
        }
        let shape = Arc::new(sb.finish().expect("the circuit builds"));
        let fill_plan = Arc::new(shape.fill_plan());
        let c = self.0.get_or_init(|| OuterShape { shape, fill_plan });
        (c.shape.clone(), c.fill_plan.clone())
    }
}

/// The per-kind slots a tower run fills: FL, base, spine.
#[derive(Default)]
pub struct TowerShapes {
    pub(super) fl: ShapeSlot,
    pub(super) base: ShapeSlot,
    pub(super) spine: ShapeSlot,
}
