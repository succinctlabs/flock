//! Compact forward and transposed circuit walks.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

use crate::field::F128;

use super::{
    BooleanCircuit, CircuitId, ExpressionNode, LayoutError, LinearExprId, PhysicalLayout, RowId,
    RowKind,
};

/// A compiled, layout-specific structural circuit walk.
///
/// The plan owns no witness data and can therefore be shared by prover and
/// verifier code. Virtual XOR results use preallocated, reusable temporary
/// slots; materialized values remain boundaries in both directions.
#[derive(Clone, Debug)]
pub struct WalkPlan {
    circuit: CircuitId,
    actions: Vec<Action>,
    input_positions: Vec<usize>,
    one_position: usize,
    useful_bits: usize,
    c_is_identity: bool,
    occupied_positions: Vec<usize>,
    expression_fanout: Vec<usize>,
    stats: WalkStats,
}

/// Structural cost and temporary-memory bounds for a [`WalkPlan`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkStats {
    /// XOR nodes retained by the compiled plan.
    pub xor_nodes: usize,
    /// Authored XOR term edges plus the three expression edges of each row.
    pub structural_edges: usize,
    /// Number of compact forward/reverse actions.
    pub actions: usize,
    /// Reusable slots needed for all simultaneously live virtual values.
    pub max_live_temporaries: usize,
}

/// Values produced by one forward walk, in physical R1CS order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwardTrace {
    pub z: Vec<bool>,
    pub a_z: Vec<bool>,
    pub b_z: Vec<bool>,
    pub c_z: Vec<bool>,
}

/// A forward or transposed walk request that does not match its compiled plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WalkError {
    InputCount { expected: usize, actual: usize },
    InvalidKLog(usize),
    Capacity { required: usize, actual: usize },
    UnsatisfiedRow(RowId),
    WeightCount { a: usize, b: usize, c: usize },
    InvalidTransposeSize(usize),
    IdentityCRequired,
}

impl fmt::Display for WalkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCount { expected, actual } => {
                write!(f, "expected {expected} inputs, received {actual}")
            }
            Self::InvalidKLog(k_log) => write!(f, "2^{k_log} does not fit in usize"),
            Self::Capacity { required, actual } => write!(
                f,
                "walk needs capacity {required}, but capacity is {actual}"
            ),
            Self::UnsatisfiedRow(row) => {
                write!(f, "general constraint row {} is not satisfied", row.index())
            }
            Self::WeightCount { a, b, c } => write!(
                f,
                "transpose weights must have one common length (A={a}, B={b}, C={c})"
            ),
            Self::InvalidTransposeSize(size) => {
                write!(f, "transpose weight count {size} is not a power of two")
            }
            Self::IdentityCRequired => {
                f.write_str("identity-C walk requested for a non-identity C relation")
            }
        }
    }
}

impl std::error::Error for WalkError {}

#[derive(Clone, Copy, Debug)]
enum Source {
    Zero,
    Value(usize),
    Temporary(usize),
}

#[derive(Clone, Debug)]
enum Action {
    Xor {
        output: usize,
        terms: Box<[Source]>,
    },
    Row {
        id: RowId,
        physical_row: usize,
        kind: RowKind,
        lhs: Source,
        rhs: Source,
        result: Source,
        output: Option<usize>,
    },
}

#[derive(Clone, Debug)]
enum RawAction {
    Xor {
        expression: LinearExprId,
        terms: Vec<LinearExprId>,
    },
    Row(usize),
}

impl BooleanCircuit {
    /// Compile a source-order structural walk.
    pub fn walk_plan(&self) -> WalkPlan {
        WalkPlan::compile(self, &PhysicalLayout::source_order(self))
            .expect("source-order layout must be valid")
    }

    /// Compile a structural walk for an explicit physical layout.
    pub fn walk_plan_with_layout(&self, layout: &PhysicalLayout) -> Result<WalkPlan, LayoutError> {
        WalkPlan::compile(self, layout)
    }
}

impl WalkPlan {
    fn compile(circuit: &BooleanCircuit, layout: &PhysicalLayout) -> Result<Self, LayoutError> {
        layout.validate_for(circuit)?;

        let mut raw_actions = Vec::new();
        let mut scheduled = vec![false; circuit.expressions.len()];
        for (row_index, row) in circuit.rows.iter().enumerate() {
            for expression in [row.lhs, row.rhs, row.result] {
                schedule_expression(circuit, expression, &mut scheduled, &mut raw_actions);
            }
            raw_actions.push(RawAction::Row(row_index));
        }

        let mut fanout = vec![0usize; circuit.expressions.len()];
        let mut last_use = vec![None; circuit.expressions.len()];
        for (action_index, action) in raw_actions.iter().enumerate() {
            match action {
                RawAction::Xor { terms, .. } => {
                    for &term in terms {
                        fanout[term.index] += 1;
                        last_use[term.index] = Some(action_index);
                    }
                }
                RawAction::Row(row_index) => {
                    let row = &circuit.rows[*row_index];
                    for expression in [row.lhs, row.rhs, row.result] {
                        fanout[expression.index] += 1;
                        last_use[expression.index] = Some(action_index);
                    }
                }
            }
        }

        let mut expression_slots = vec![None; circuit.expressions.len()];
        let mut free_slots = BinaryHeap::<Reverse<usize>>::new();
        let mut slot_count = 0;
        let mut live_temporaries = 0;
        let mut max_live_temporaries = 0;
        let mut actions = Vec::with_capacity(raw_actions.len());
        let mut xor_nodes = 0;
        let mut xor_edges = 0;

        for (action_index, raw) in raw_actions.into_iter().enumerate() {
            match raw {
                RawAction::Xor { expression, terms } => {
                    let sources: Vec<Source> = terms
                        .iter()
                        .map(|&term| source(circuit, layout, &expression_slots, term))
                        .collect();
                    // Sources are captured first. The action folds them into a
                    // local value before writing, so its output may reuse a
                    // source slot whose last use is this action.
                    release_last_uses(
                        action_index,
                        &terms,
                        &last_use,
                        &expression_slots,
                        &mut free_slots,
                        &mut live_temporaries,
                    );
                    let output = if let Some(Reverse(slot)) = free_slots.pop() {
                        slot
                    } else {
                        let slot = slot_count;
                        slot_count += 1;
                        slot
                    };
                    expression_slots[expression.index] = Some(output);
                    live_temporaries += 1;
                    max_live_temporaries = max_live_temporaries.max(live_temporaries);
                    xor_nodes += 1;
                    xor_edges += sources.len();
                    actions.push(Action::Xor {
                        output,
                        terms: sources.into_boxed_slice(),
                    });
                }
                RawAction::Row(row_index) => {
                    let row = &circuit.rows[row_index];
                    let expressions = [row.lhs, row.rhs, row.result];
                    let [lhs, rhs, result] =
                        expressions.map(|id| source(circuit, layout, &expression_slots, id));
                    release_last_uses(
                        action_index,
                        &expressions,
                        &last_use,
                        &expression_slots,
                        &mut free_slots,
                        &mut live_temporaries,
                    );
                    actions.push(Action::Row {
                        id: row.id,
                        physical_row: layout.row_positions[row.id.index],
                        kind: row.kind,
                        lhs,
                        rhs,
                        result,
                        output: row
                            .defined_value
                            .map(|value| layout.value_positions[value.index]),
                    });
                }
            }
        }
        debug_assert_eq!(live_temporaries, 0);

        let mut row_positions = layout.row_positions.clone();
        let mut value_positions = layout.value_positions.clone();
        row_positions.sort_unstable();
        value_positions.sort_unstable();
        let mut occupied_positions = row_positions.clone();
        occupied_positions.extend_from_slice(&value_positions);
        occupied_positions.sort_unstable();
        occupied_positions.dedup();
        let c_is_identity = circuit.rows.iter().all(|row| {
            let position = layout.row_positions[row.id.index];
            let support = &circuit.expressions[row.result.index].support;
            support.len() == 1 && layout.value_positions[support[0].index] == position
        }) && row_positions == value_positions;

        Ok(Self {
            circuit: circuit.id,
            actions,
            input_positions: circuit
                .input_values
                .iter()
                .map(|value| layout.value_positions[value.index])
                .collect(),
            one_position: layout.value_positions[circuit.one.index],
            useful_bits: layout.useful_bits,
            c_is_identity,
            occupied_positions,
            expression_fanout: fanout,
            stats: WalkStats {
                xor_nodes,
                structural_edges: xor_edges + circuit.rows.len() * 3,
                actions: circuit.rows.len() + xor_nodes,
                max_live_temporaries,
            },
        })
    }

    /// Physical prefix containing every real row and materialized value.
    pub const fn useful_bits(&self) -> usize {
        self.useful_bits
    }

    /// Whether this layout's complete C matrix is the identity, including its
    /// padding rows.
    pub const fn c_is_identity(&self) -> bool {
        self.c_is_identity
    }

    pub const fn stats(&self) -> WalkStats {
        self.stats
    }

    /// Number of structural consumers of an expression, including row sides.
    pub fn expression_fanout(&self, expression: LinearExprId) -> Option<usize> {
        if expression.circuit != self.circuit {
            return None;
        }
        self.expression_fanout.get(expression.index).copied()
    }

    /// Generate `z`, `A z`, `B z`, and `C z` without sparse matrix products.
    pub fn forward(&self, inputs: &[bool], k_log: usize) -> Result<ForwardTrace, WalkError> {
        if inputs.len() != self.input_positions.len() {
            return Err(WalkError::InputCount {
                expected: self.input_positions.len(),
                actual: inputs.len(),
            });
        }
        let capacity = checked_capacity(k_log).ok_or(WalkError::InvalidKLog(k_log))?;
        if capacity < self.useful_bits {
            return Err(WalkError::Capacity {
                required: self.useful_bits,
                actual: capacity,
            });
        }

        let mut z = vec![false; capacity];
        let mut a_z = vec![false; capacity];
        let mut b_z = vec![false; capacity];
        let mut c_z = vec![false; capacity];
        let mut temporaries = vec![false; self.stats.max_live_temporaries];
        z[self.one_position] = true;
        for (&position, &input) in self.input_positions.iter().zip(inputs) {
            z[position] = input;
        }

        for action in &self.actions {
            match action {
                Action::Xor { output, terms } => {
                    let value = terms.iter().fold(false, |acc, source| {
                        acc ^ read_bool(*source, &z, &temporaries)
                    });
                    temporaries[*output] = value;
                }
                Action::Row {
                    id,
                    physical_row,
                    kind,
                    lhs,
                    rhs,
                    result,
                    output,
                } => {
                    let lhs = read_bool(*lhs, &z, &temporaries);
                    let rhs = read_bool(*rhs, &z, &temporaries);
                    match kind {
                        RowKind::And | RowKind::Materialize => {
                            z[output.expect("definitional row must have an output")] = lhs & rhs;
                        }
                        RowKind::One | RowKind::Input | RowKind::Constraint => {}
                    }
                    let result = read_bool(*result, &z, &temporaries);
                    if lhs & rhs != result {
                        return Err(WalkError::UnsatisfiedRow(*id));
                    }
                    a_z[*physical_row] = lhs;
                    b_z[*physical_row] = rhs;
                    c_z[*physical_row] = result;
                }
            }
        }

        Ok(ForwardTrace { z, a_z, b_z, c_z })
    }

    /// Compute `A^T e_a + B^T e_b + C^T e_c` by reversing the structural DAG.
    pub fn transpose(
        &self,
        e_a: &[F128],
        e_b: &[F128],
        e_c: &[F128],
    ) -> Result<Vec<F128>, WalkError> {
        self.transpose_inner(e_a, e_b, e_c, false)
    }

    /// Identity-C specialization of [`Self::transpose`]. C contributes by a
    /// direct vector copy instead of traversing its structural row expressions.
    pub fn transpose_identity_c(
        &self,
        e_a: &[F128],
        e_b: &[F128],
        e_c: &[F128],
    ) -> Result<Vec<F128>, WalkError> {
        if !self.c_is_identity {
            return Err(WalkError::IdentityCRequired);
        }
        self.transpose_inner(e_a, e_b, e_c, true)
    }

    fn transpose_inner(
        &self,
        e_a: &[F128],
        e_b: &[F128],
        e_c: &[F128],
        identity_c: bool,
    ) -> Result<Vec<F128>, WalkError> {
        if e_a.len() != e_b.len() || e_a.len() != e_c.len() {
            return Err(WalkError::WeightCount {
                a: e_a.len(),
                b: e_b.len(),
                c: e_c.len(),
            });
        }
        if !e_a.len().is_power_of_two() {
            return Err(WalkError::InvalidTransposeSize(e_a.len()));
        }
        if e_a.len() < self.useful_bits {
            return Err(WalkError::Capacity {
                required: self.useful_bits,
                actual: e_a.len(),
            });
        }

        // Lowering gives C an identity row only at padding positions. Begin
        // with that diagonal contribution, clear every real row or witness
        // column, then add the real C rows during the reverse walk.
        let mut z = e_c.to_vec();
        if !identity_c {
            for &position in &self.occupied_positions {
                z[position] = F128::ZERO;
            }
        }
        let mut temporaries = vec![F128::ZERO; self.stats.max_live_temporaries];

        for action in self.actions.iter().rev() {
            match action {
                Action::Row {
                    physical_row,
                    lhs,
                    rhs,
                    result,
                    ..
                } => {
                    add_f128(*lhs, e_a[*physical_row], &mut z, &mut temporaries);
                    add_f128(*rhs, e_b[*physical_row], &mut z, &mut temporaries);
                    if !identity_c {
                        add_f128(*result, e_c[*physical_row], &mut z, &mut temporaries);
                    }
                }
                Action::Xor { output, terms } => {
                    let adjoint = std::mem::replace(&mut temporaries[*output], F128::ZERO);
                    for &term in terms.iter() {
                        add_f128(term, adjoint, &mut z, &mut temporaries);
                    }
                }
            }
        }
        debug_assert!(temporaries.iter().all(|value| value.is_zero()));
        Ok(z)
    }
}

fn schedule_expression(
    circuit: &BooleanCircuit,
    root: LinearExprId,
    scheduled: &mut [bool],
    actions: &mut Vec<RawAction>,
) {
    let mut stack = vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        if scheduled[id.index] {
            continue;
        }
        match &circuit.expressions[id.index].node {
            ExpressionNode::Xor(terms) if expanded => {
                scheduled[id.index] = true;
                actions.push(RawAction::Xor {
                    expression: id,
                    terms: terms.clone(),
                });
            }
            ExpressionNode::Xor(terms) => {
                stack.push((id, true));
                stack.extend(terms.iter().rev().map(|&term| (term, false)));
            }
            ExpressionNode::Zero | ExpressionNode::Value(_) => {
                scheduled[id.index] = true;
            }
        }
    }
}

fn source(
    circuit: &BooleanCircuit,
    layout: &PhysicalLayout,
    expression_slots: &[Option<usize>],
    id: LinearExprId,
) -> Source {
    match circuit.expressions[id.index].node {
        ExpressionNode::Zero => Source::Zero,
        ExpressionNode::Value(value) => Source::Value(layout.value_positions[value.index]),
        ExpressionNode::Xor(_) => Source::Temporary(
            expression_slots[id.index].expect("XOR source must already be scheduled"),
        ),
    }
}

fn release_last_uses(
    action_index: usize,
    expressions: &[LinearExprId],
    last_use: &[Option<usize>],
    expression_slots: &[Option<usize>],
    free_slots: &mut BinaryHeap<Reverse<usize>>,
    live_temporaries: &mut usize,
) {
    let mut released: Vec<usize> = expressions
        .iter()
        .filter(|expression| last_use[expression.index] == Some(action_index))
        .filter_map(|expression| expression_slots[expression.index])
        .collect();
    released.sort_unstable();
    released.dedup();
    for slot in released {
        free_slots.push(Reverse(slot));
        *live_temporaries -= 1;
    }
}

#[inline]
fn read_bool(source: Source, values: &[bool], temporaries: &[bool]) -> bool {
    match source {
        Source::Zero => false,
        Source::Value(position) => values[position],
        Source::Temporary(slot) => temporaries[slot],
    }
}

#[inline]
fn add_f128(source: Source, value: F128, values: &mut [F128], temporaries: &mut [F128]) {
    match source {
        Source::Zero => {}
        Source::Value(position) => values[position] += value,
        Source::Temporary(slot) => temporaries[slot] += value,
    }
}

fn checked_capacity(k_log: usize) -> Option<usize> {
    u32::try_from(k_log)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
}

#[cfg(test)]
#[path = "walk/tests.rs"]
mod tests;
