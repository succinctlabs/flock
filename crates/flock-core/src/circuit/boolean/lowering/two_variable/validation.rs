use super::*;
use crate::circuit::boolean::builder::symmetric_difference;
use crate::circuit::boolean::{RowKind, ValueIndex};

impl LoweredCircuit {
    /// Check initialized boundaries separately from row-defined values.
    pub(super) fn validate(&self) {
        assert_eq!(
            self.rows.len(),
            self.source_rows.len() + self.auxiliaries.len()
        );
        let mut boundaries = vec![None; self.value_count()];
        for (index, expr) in self.expressions.iter().enumerate() {
            let expected = match &expr.node {
                ExpressionNode::Zero => Vec::new(),
                ExpressionNode::Value(value) => {
                    assert_eq!(value.circuit, self.id);
                    assert!(boundaries[value.index].replace(index).is_none());
                    vec![ValueIndex::new(value.index)]
                }
                ExpressionNode::Xor(terms) => terms.iter().fold(Vec::new(), |support, term| {
                    assert_eq!(term.circuit, self.id);
                    assert!(term.index < index, "expression DAG must be acyclic");
                    symmetric_difference(&support, &self.expressions[term.index].support)
                }),
            };
            assert_eq!(expr.support, expected);
        }
        assert!(boundaries.iter().all(Option::is_some));

        // Zero means initialized before execution; computed values use row index + 1.
        let mut available = vec![None; self.value_count()];
        for value in std::iter::once(self.one)
            .chain(self.input_values.iter().copied())
            .chain(self.auxiliaries.iter().map(|aux| aux.cancellation))
        {
            assert_eq!(value.circuit, self.id);
            assert!(available[value.index].replace(0).is_none());
        }
        let mut defined = vec![false; self.value_count()];
        for (index, row) in self.rows.iter().enumerate() {
            assert_eq!(
                row.id,
                RowId {
                    circuit: self.id,
                    index
                }
            );
            for expr in [row.lhs, row.rhs, row.result] {
                assert_eq!(expr.circuit, self.id);
                assert!(expr.index < self.expressions.len());
            }
            if let Some(value) = row.defined_value {
                assert_eq!(value.circuit, self.id);
                assert!(!std::mem::replace(&mut defined[value.index], true));
                assert_eq!(boundaries[value.index], Some(row.result.index));
                match row.kind {
                    RowKind::One | RowKind::Input => {
                        assert_eq!(available[value.index], Some(0));
                        assert_eq!(row.lhs, row.result);
                        assert_eq!(
                            self.expressions[row.rhs.index].node,
                            ExpressionNode::Value(self.one)
                        );
                        assert_eq!(row.kind == RowKind::One, value == self.one);
                    }
                    RowKind::And | RowKind::Materialize => {
                        assert!(available[value.index].replace(index + 1).is_none());
                        if row.kind == RowKind::Materialize {
                            assert_eq!(
                                self.expressions[row.rhs.index].node,
                                ExpressionNode::Value(self.one)
                            );
                        }
                    }
                    RowKind::Constraint => panic!("a cancellation check must not define t"),
                }
            } else {
                assert_eq!(row.kind, RowKind::Constraint);
            }
        }
        assert!(available.iter().all(Option::is_some));
        for aux in &self.auxiliaries {
            assert!(!defined[aux.cancellation.index]);
            assert_eq!(
                self.rows[aux.product_row.index].defined_value,
                Some(aux.product)
            );
            let check = &self.rows[aux.cancellation_row.index];
            assert_eq!(check.kind, RowKind::Constraint);
            assert_eq!(boundaries[aux.cancellation.index], Some(check.result.index));
        }
        assert_eq!(
            defined.iter().filter(|&&value| !value).count(),
            self.auxiliaries.len()
        );

        // Inspect structural dependencies, including XOR terms that cancel in support.
        let mut latest = Vec::with_capacity(self.expressions.len());
        for expr in &self.expressions {
            latest.push(match &expr.node {
                ExpressionNode::Zero => 0,
                ExpressionNode::Value(value) => available[value.index].unwrap(),
                ExpressionNode::Xor(terms) => terms
                    .iter()
                    .map(|term| latest[term.index])
                    .max()
                    .unwrap_or(0),
            });
        }
        for row in &self.rows {
            for (expr, is_result) in [(row.lhs, false), (row.rhs, false), (row.result, true)] {
                let is_definition = is_result && row.defined_value.is_some();
                assert!(
                    is_definition || latest[expr.index] <= row.id.index,
                    "structural read before definition"
                );
            }
        }
    }
}
