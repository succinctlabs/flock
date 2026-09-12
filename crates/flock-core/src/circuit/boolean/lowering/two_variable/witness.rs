use super::LoweredCircuit;
use crate::circuit::boolean::EvaluationError;

impl LoweredCircuit {
    /// Extend any complete candidate witness without repairing its derived values.
    /// Only shape is checked here; use `accepts` to check ONE and every equation.
    pub fn extend(&self, source: &[bool]) -> Option<Vec<bool>> {
        if source.len() != self.source_value_count {
            return None;
        }
        let mut values = source.to_vec();
        values.resize(self.value_count(), false);
        for aux in &self.auxiliaries {
            let row = &self.rows[aux.product_row.index];
            values[aux.product.index] = self.relation().eval_expression(row.lhs, &values)
                & self.relation().eval_expression(row.rhs, &values);
        }
        Some(values)
    }

    /// Extract source coordinates without certifying satisfaction.
    pub fn project(&self, values: &[bool]) -> Option<Vec<bool>> {
        (values.len() == self.value_count()).then(|| values[..self.source_value_count].to_vec())
    }

    /// Check an exact logical witness, including arbitrary cancellation values.
    pub fn accepts(&self, values: &[bool]) -> bool {
        values.len() == self.value_count()
            && values[self.one.index]
            && self.rows.iter().all(|row| {
                (self.relation().eval_expression(row.lhs, values)
                    & self.relation().eval_expression(row.rhs, values))
                    == self.relation().eval_expression(row.result, values)
            })
    }

    /// Supply the original inputs/advice; canonical evaluation initializes each t to zero.
    /// Failures report source row IDs. No acceptance-output binding is required.
    pub fn evaluate(&self, inputs: &[bool]) -> Result<Vec<bool>, EvaluationError> {
        self.relation()
            .evaluate(inputs)
            .map_err(|error| match error {
                EvaluationError::UnsatisfiedRow(row) => EvaluationError::UnsatisfiedRow(
                    self.source_row(row).expect("lowered row provenance"),
                ),
                other => other,
            })
    }
}
