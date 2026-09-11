//! Reference execution shared by authored and lowered equations.

use super::RelationRef;
use crate::circuit::boolean::{EvaluationError, LinearExprId, RowKind};

impl RelationRef<'_> {
    pub(in crate::circuit::boolean) fn evaluate(
        &self,
        inputs: &[bool],
    ) -> Result<Vec<bool>, EvaluationError> {
        if inputs.len() != self.input_values.len() {
            return Err(EvaluationError::InputCount {
                expected: self.input_values.len(),
                actual: inputs.len(),
            });
        }

        let mut values = vec![false; self.value_count];
        values[self.one.index] = true;
        for (&value, &input) in self.input_values.iter().zip(inputs) {
            values[value.index] = input;
        }

        for row in self.rows {
            match row.kind {
                RowKind::One | RowKind::Input => {}
                RowKind::And => {
                    let output = row.defined_value.expect("AND row must define a value");
                    values[output.index] = self.eval_expression(row.lhs, &values)
                        & self.eval_expression(row.rhs, &values);
                }
                RowKind::Materialize => {
                    let output = row
                        .defined_value
                        .expect("materialization row must define a value");
                    values[output.index] = self.eval_expression(row.lhs, &values);
                }
                RowKind::Constraint => {
                    let lhs = self.eval_expression(row.lhs, &values);
                    let rhs = self.eval_expression(row.rhs, &values);
                    let result = self.eval_expression(row.result, &values);
                    if (lhs & rhs) != result {
                        return Err(EvaluationError::UnsatisfiedRow(row.id));
                    }
                }
            }
        }
        Ok(values)
    }

    pub(in crate::circuit::boolean) fn eval_expression(
        &self,
        id: LinearExprId,
        values: &[bool],
    ) -> bool {
        self.expressions[id.index]
            .support
            .iter()
            .fold(false, |sum, value| sum ^ values[value.index()])
    }
}
