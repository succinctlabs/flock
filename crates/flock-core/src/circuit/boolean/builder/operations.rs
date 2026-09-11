//! Operation boundaries for the typed authoring prototype.

use super::*;
use crate::circuit::boolean::{OperationWord, SchemaOperation, Selector, Var};

impl CircuitBuilder {
    /// Use a supplied or defined schema column as a selector.
    pub fn column_selector(&self, var: Var) -> Selector {
        self.assert_available(var.into());
        self.selector(var.0)
    }

    /// Record an operation over one complete schema instance and a virtual word result.
    /// The instance path is inferred from its owned columns, not supplied by the caller.
    pub fn operation<const N: usize>(
        &mut self,
        kind: &str,
        columns: &[Var],
        inputs: impl IntoIterator<Item = (&'static str, Vec<LinearExpr>)>,
        eval: impl FnOnce(&mut Self) -> [LinearExpr; N],
    ) -> [LinearExpr; N] {
        assert!(
            !kind.is_empty() && N > 0,
            "operation kind and result must be nonempty"
        );
        let values: Vec<ValueId> = columns
            .iter()
            .map(|var| {
                self.assert_circuit(var.0.value.circuit);
                var.0.value
            })
            .collect();
        let name = self.operation_instance(&values);
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
        for &expression in &result {
            self.assert_available(expression);
        }
        self.validate_operation_dependencies(
            columns,
            &inputs,
            &result,
            &self.rows[row_start..],
            &self.interactions[interaction_start..],
        );
        let schema = self.schema.as_mut().unwrap();
        assert!(
            values.iter().all(|value| schema.defined[value.index]),
            "operation left columns undefined"
        );
        assert!(
            self.rows[row_start..]
                .iter()
                .filter_map(|row| row.defined_value)
                .all(|value| values.contains(&value)),
            "operation defined another instance's column"
        );
        schema.operations.push(SchemaOperation {
            name,
            kind: kind.into(),
            columns: values,
            inputs,
            output: OperationWord {
                name: "result".into(),
                expressions: result.iter().map(|expression| expression.id).collect(),
            },
            rows: row_start..self.rows.len(),
            expressions: expression_start..self.expressions.len(),
            interactions: interaction_start..self.interactions.len(),
        });
        result
    }

    fn operation_instance(&self, values: &[ValueId]) -> String {
        let schema = self.schema.as_ref().expect("operation requires a schema");
        let first = schema
            .columns
            .iter()
            .find(|field| field.values.first() == values.first())
            .expect("operation must own complete schema fields");
        let (mut name, _) = first
            .name
            .rsplit_once('.')
            .expect("operation fields need an instance path");
        // Find the deepest enclosing scope, including fields in child operations.
        let expected = loop {
            let prefix = format!("{name}.");
            let expected: Vec<_> = schema
                .columns
                .iter()
                .filter(|field| field.name.starts_with(&prefix))
                .flat_map(|field| field.values.iter().copied())
                .collect();
            if values.iter().all(|value| expected.contains(value)) {
                break expected;
            }
            (name, _) = name
                .rsplit_once('.')
                .expect("operation must own its exact schema instance");
        };
        assert_eq!(
            values, expected,
            "operation must own its exact schema instance"
        );
        assert!(
            values.iter().all(|value| !schema.defined[value.index]),
            "operation columns already defined"
        );
        name.to_owned()
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
        let mut boundaries: std::collections::BTreeSet<_> = inputs
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
