//! Internal consistency checks run when a circuit is finished.

use super::{
    CircuitBuilder, ExpressionNode, LinearExprId, RowId, RowKind, ValueId, ValueIndex,
    symmetric_difference,
};

impl CircuitBuilder {
    pub(super) fn validate(&self) -> Vec<RowId> {
        assert_eq!(self.zero.id.index, 0, "ZERO must be the first expression");
        assert!(matches!(self.expressions[0].node, ExpressionNode::Zero));
        assert_eq!(self.one.value.index, 0, "ONE must be the first value");
        assert_eq!(self.one.expression.index, 1);
        assert!(matches!(self.rows[0].kind, RowKind::One));

        let mut value_expressions = vec![None; self.value_count];
        for (index, expression) in self.expressions.iter().enumerate() {
            let expected = match &expression.node {
                ExpressionNode::Zero => Vec::new(),
                ExpressionNode::Value(value) => {
                    assert!(
                        value.circuit == self.id && value.index < self.value_count,
                        "expression has invalid value id"
                    );
                    assert!(
                        value_expressions[value.index]
                            .replace(LinearExprId {
                                circuit: self.id,
                                index,
                            })
                            .is_none(),
                        "materialized value has multiple boundary expressions"
                    );
                    vec![ValueIndex::new(value.index)]
                }
                ExpressionNode::Xor(terms) => {
                    let mut support = Vec::new();
                    for term in terms {
                        assert_eq!(
                            term.circuit, self.id,
                            "expression belongs to another circuit"
                        );
                        assert!(
                            term.index < index,
                            "structural expression DAG contains a cycle"
                        );
                        // Earlier nodes have already passed this check, so
                        // their stored supports are valid by induction.
                        support =
                            symmetric_difference(&support, &self.expressions[term.index].support);
                    }
                    support
                }
            };
            assert_eq!(
                expression.support, expected,
                "stored normalized support disagrees with structural DAG"
            );
        }
        assert!(
            value_expressions.iter().all(Option::is_some),
            "materialized value is missing its boundary expression"
        );

        let mut definitions = vec![None; self.value_count];
        for (index, row) in self.rows.iter().enumerate() {
            assert_eq!(row.id.circuit, self.id, "row belongs to another circuit");
            assert_eq!(row.id.index, index, "non-canonical row order");
            assert!(
                row.lhs.circuit == self.id && row.lhs.index < self.expressions.len(),
                "invalid lhs expression"
            );
            assert!(
                row.rhs.circuit == self.id && row.rhs.index < self.expressions.len(),
                "invalid rhs expression"
            );
            assert!(
                row.result.circuit == self.id && row.result.index < self.expressions.len(),
                "invalid result expression"
            );
            match row.kind {
                RowKind::One => {
                    assert_eq!(index, 0, "ONE must be the first row");
                    assert_eq!(row.defined_value, Some(self.one.value));
                    assert_eq!(row.lhs, self.one.expression);
                    assert_eq!(row.rhs, self.one.expression);
                    assert_eq!(row.result, self.one.expression);
                }
                RowKind::Input => {
                    assert_eq!(row.lhs, row.result, "input row must be a tautology");
                    assert_eq!(row.rhs, self.one.expression);
                    assert!(row.defined_value.is_some());
                }
                RowKind::And => assert!(row.defined_value.is_some()),
                RowKind::Materialize => {
                    assert_eq!(row.rhs, self.one.expression);
                    assert!(row.defined_value.is_some());
                }
                RowKind::Constraint => assert!(row.defined_value.is_none()),
            }
            if let Some(value) = row.defined_value {
                assert!(
                    value.circuit == self.id && value.index < self.value_count,
                    "row defines an invalid value id"
                );
                assert!(
                    definitions[value.index].replace(row.id).is_none(),
                    "materialized value has multiple defining rows"
                );
                assert_eq!(
                    Some(row.result),
                    value_expressions[value.index],
                    "defining row result is not its value boundary"
                );
            }
        }
        assert!(
            definitions.iter().all(Option::is_some),
            "materialized value is missing its defining row"
        );
        let definition_rows: Vec<RowId> = definitions.into_iter().map(Option::unwrap).collect();
        let row_inputs: Vec<ValueId> = self
            .rows
            .iter()
            .filter(|row| row.kind == RowKind::Input)
            .map(|row| row.defined_value.unwrap())
            .collect();
        assert_eq!(
            self.input_values, row_inputs,
            "input list disagrees with rows"
        );

        // Structural dependencies do not cancel when their normalized supports do.
        let mut latest_definition: Vec<Option<usize>> = Vec::with_capacity(self.expressions.len());
        for expression in &self.expressions {
            let latest = match &expression.node {
                ExpressionNode::Zero => None,
                ExpressionNode::Value(value) => Some(definition_rows[value.index].index),
                ExpressionNode::Xor(terms) => terms
                    .iter()
                    .filter_map(|term| latest_definition[term.index])
                    .max(),
            };
            latest_definition.push(latest);
        }
        for row in &self.rows {
            for (expression, is_result) in [(row.lhs, false), (row.rhs, false), (row.result, true)]
            {
                if let Some(definition) = latest_definition[expression.index] {
                    let self_reference = matches!(self.expressions[expression.index].node,
                        ExpressionNode::Value(value) if row.defined_value == Some(value))
                        && (matches!(row.kind, RowKind::One | RowKind::Input) || is_result);
                    assert!(
                        definition < row.id.index || self_reference,
                        "row reads a value before it is defined"
                    );
                }
            }
        }
        self.validate_interactions();
        definition_rows
    }
}
