//! Named operation columns, arguments, results, and dependency checks.

use std::collections::BTreeSet;

use super::{CircuitBuilder, ExpressionNode, Interaction, LinearExpr, Row, ValueId};
use crate::circuit::boolean::{OperationWord, SchemaOperation, Selector, Var};

impl CircuitBuilder {
    /// Use a supplied or defined schema column as a selector.
    pub fn column_selector(&self, var: Var) -> Selector {
        self.assert_available(var.into());
        self.selector(var.0)
    }

    /// Record an operation over explicit columns and a word result.
    /// Calls may share a kind; column names do not determine ownership.
    pub fn operation<R: AsRef<[LinearExpr]>>(
        &mut self,
        kind: &str,
        columns: &[Var],
        inputs: impl IntoIterator<Item = (&'static str, Vec<LinearExpr>)>,
        eval: impl FnOnce(&mut Self) -> R,
    ) -> R {
        assert!(!kind.is_empty(), "operation kind must be nonempty");
        assert!(!columns.is_empty(), "operation must own columns");
        let mut owned = BTreeSet::new();
        let values: Vec<ValueId> = columns
            .iter()
            .map(|var| {
                let value = var.0.value;
                self.assert_circuit(value.circuit);
                assert!(owned.insert(value), "duplicate operation column");
                assert!(
                    !self.schema.defined[value.index],
                    "operation columns already defined"
                );
                value
            })
            .collect();
        let inputs: Vec<OperationWord> = inputs
            .into_iter()
            .map(|(name, expressions)| {
                assert!(
                    !name.is_empty() && !expressions.is_empty(),
                    "empty operation argument"
                );
                for &expression in &expressions {
                    self.assert_available(expression);
                }
                OperationWord {
                    name: name.into(),
                    expressions: expressions.iter().map(|expression| expression.id).collect(),
                }
            })
            .collect();
        for (index, input) in inputs.iter().enumerate() {
            assert!(
                inputs[..index].iter().all(|other| other.name != input.name),
                "duplicate operation argument"
            );
        }
        let row_start = self.rows.len();
        let expression_start = self.expressions.len();
        let interaction_start = self.interactions.len();
        let result = eval(self);
        let word = result.as_ref();
        assert!(!word.is_empty(), "operation result must be nonempty");
        for &expression in word {
            self.assert_available(expression);
        }
        assert!(
            values.iter().all(|value| self.schema.defined[value.index]),
            "operation left columns undefined"
        );
        assert!(
            self.rows[row_start..]
                .iter()
                .filter_map(|row| row.defined_value)
                .all(|value| owned.contains(&value)),
            "operation defined a column it does not own"
        );
        self.validate_operation_dependencies(
            columns,
            &inputs,
            word,
            &self.rows[row_start..],
            &self.interactions[interaction_start..],
        );
        self.schema.operations.push(SchemaOperation {
            kind: kind.into(),
            columns: values,
            inputs,
            output: OperationWord {
                name: "result".into(),
                expressions: word.iter().map(|expression| expression.id).collect(),
            },
            rows: row_start..self.rows.len(),
            expressions: expression_start..self.expressions.len(),
            interactions: interaction_start..self.interactions.len(),
        });
        result
    }

    fn validate_operation_dependencies(
        &self,
        columns: &[Var],
        inputs: &[OperationWord],
        result: &[LinearExpr],
        rows: &[Row],
        interactions: &[Interaction],
    ) {
        // Every external dependency must be accounted for by a named argument.
        let mut boundaries: BTreeSet<_> = inputs
            .iter()
            .flat_map(|word| word.expressions.iter().map(|expression| expression.index))
            .collect();
        boundaries.extend(columns.iter().map(|var| var.0.expression.index));
        boundaries.extend([self.zero.id.index, self.one.expression.index]);
        let mut pending: Vec<_> = result.iter().map(|expression| expression.id).collect();
        for row in rows {
            pending.extend([row.lhs, row.rhs, row.result]);
        }
        for interaction in interactions {
            // Message references are materialized; recover their unique boundary nodes.
            for value in std::iter::once(&interaction.selector)
                .chain(&interaction.multiplicity)
                .chain(interaction.message.iter().flat_map(|field| &field.values))
            {
                let declared = columns.iter().any(|var| var.0.value == *value)
                    || *value == self.one.value
                    || inputs
                        .iter()
                        .flat_map(|word| &word.expressions)
                        .any(|expression| {
                            matches!(
                                self.expressions[expression.index].node,
                                ExpressionNode::Value(v) if v == *value
                            )
                        });
                assert!(
                    declared,
                    "interaction reads an undeclared operation argument"
                );
            }
        }
        while let Some(expression) = pending.pop() {
            if !boundaries.insert(expression.index) {
                continue;
            }
            match &self.expressions[expression.index].node {
                ExpressionNode::Xor(terms) => pending.extend(terms),
                _ => panic!("operation reads an undeclared argument"),
            }
        }
    }
}

#[cfg(test)]
mod tests;
