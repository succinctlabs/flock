//! Deferred acceptance-checker reference path; requires an outer accept binding.
//! The optional no-accept lowering lives in `lowering/two_variable`.

use std::ops::Range;

use super::{BooleanCircuit, CircuitBuilder, CircuitId, ExpressionNode, RowId, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringPurpose {
    RowProduct,
    AllChecks,
}

/// One generated value and defining row, with its source-row provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweringAux {
    pub value: ValueId,
    pub row: RowId,
    pub source_rows: Range<usize>,
    pub purpose: LoweringPurpose,
}

/// A Boolean relation checker whose raw circuit requires an outer accept binding.
///
/// This intentionally simple conversion makes the entire source witness input.
/// Source schema, ports, operations, and interactions stay on the source circuit;
/// use [`Self::mapped_value`] to translate their bindings. This is not a prover
/// adapter or an optimized lowering.
pub struct IdentityChecker {
    circuit: BooleanCircuit,
    source: CircuitId,
    source_one: usize,
    projection: Vec<ValueId>,
    auxiliaries: Vec<LoweringAux>,
    accept: ValueId,
}

impl BooleanCircuit {
    /// Convert each source row into a satisfaction bit and AND all checks.
    ///
    /// For Boolean witnesses with ONE pinned, the source relation holds iff
    /// the checker relation holds and its terminal accept bit is pinned to one.
    pub fn identity_checker(&self) -> IdentityChecker {
        let mut b = CircuitBuilder::new();
        let values: Vec<_> = (0..self.value_count())
            .map(|i| {
                if i == self.one.index {
                    b.one()
                } else {
                    b.input()
                }
            })
            .collect();
        let mut expressions = Vec::with_capacity(self.expression_count());
        for node in self.expressions() {
            expressions.push(match node {
                ExpressionNode::Zero => b.zero(),
                ExpressionNode::Value(value) => values[value.index].expr(),
                ExpressionNode::Xor(terms) => b.xor(terms.iter().map(|id| expressions[id.index])),
            });
        }

        let mut generated = Vec::new();
        let mut checks = Vec::with_capacity(self.row_count());
        for row in self.rows() {
            let product = b.and(expressions[row.lhs.index], expressions[row.rhs.index]);
            let rows = row.id.index..row.id.index + 1;
            generated.push((product, rows.clone(), LoweringPurpose::RowProduct));
            let ok = b.xor3(b.one(), product, expressions[row.result.index]);
            checks.push((ok, rows));
        }

        let mut accept = None;
        while checks.len() > 1 {
            checks = checks
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 1 {
                        return pair[0].clone();
                    }
                    let product = b.and(pair[0].0, pair[1].0);
                    let rows = pair[0].1.start..pair[1].1.end;
                    generated.push((product, rows.clone(), LoweringPurpose::AllChecks));
                    accept = Some(product);
                    (product.expr(), rows)
                })
                .collect();
        }
        // Even the empty authored computation has its distinguished ONE row.
        let accept = accept.unwrap_or_else(|| {
            let bit = b.materialize(checks[0].0);
            generated.push((bit, checks[0].1.clone(), LoweringPurpose::AllChecks));
            bit
        });
        b.output("identity.accept", [accept]);
        let circuit = b.finish();
        let auxiliaries = generated
            .into_iter()
            .map(|(bit, source_rows, purpose)| LoweringAux {
                value: bit.value_id(),
                row: circuit
                    .definition_row(bit.value_id())
                    .expect("generated definition"),
                source_rows,
                purpose,
            })
            .collect();
        IdentityChecker {
            circuit,
            source: self.id,
            source_one: self.one.index,
            projection: values.into_iter().map(|bit| bit.value_id()).collect(),
            auxiliaries,
            accept: accept.value_id(),
        }
    }
}

impl IdentityChecker {
    /// Local identity-C computation only: its rows do NOT enforce accept = 1.
    /// A prover must bind [`Self::accept`] through its checked outer statement.
    pub fn unbound_circuit(&self) -> &BooleanCircuit {
        &self.circuit
    }

    pub fn accept(&self) -> ValueId {
        self.accept
    }

    pub fn auxiliaries(&self) -> &[LoweringAux] {
        &self.auxiliaries
    }

    /// Translate a source column, including ONE, into the checker relation.
    pub fn mapped_value(&self, source: ValueId) -> Option<ValueId> {
        if source.circuit != self.source {
            return None;
        }
        self.projection.get(source.index).copied()
    }

    /// Extend a logical source witness, including invalid ones. Only shape and ONE
    /// are prerequisites; [`Self::accepts`] checks whether all source rows hold.
    pub fn extend(&self, source: &[bool]) -> Option<Vec<bool>> {
        if source.len() != self.projection.len() || !source[self.source_one] {
            return None;
        }
        let inputs: Vec<_> = source
            .iter()
            .enumerate()
            .filter_map(|(i, &bit)| (i != self.source_one).then_some(bit))
            .collect();
        Some(
            self.circuit
                .evaluate(&inputs)
                .expect("checker has only definitions"),
        )
    }

    /// Recover logical source coordinates; this does not check satisfaction.
    pub fn project(&self, checker: &[bool]) -> Option<Vec<bool>> {
        if checker.len() != self.circuit.value_count() {
            return None;
        }
        Some(
            self.projection
                .iter()
                .map(|value| checker[value.index])
                .collect(),
        )
    }

    /// Reference membership check on an exact logical witness. Enforces both pins
    /// and all checker rows; unlike an output declaration, this rejects accept = 0.
    pub fn accepts(&self, values: &[bool]) -> bool {
        values.len() == self.circuit.value_count()
            && values[self.circuit.one.index]
            && values[self.accept.index]
            && self.circuit.rows().iter().all(|row| {
                (self.circuit.eval_expression(row.lhs, values)
                    & self.circuit.eval_expression(row.rhs, values))
                    == self.circuit.eval_expression(row.result, values)
            })
    }
}

#[cfg(test)]
mod tests;
