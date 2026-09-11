//! Allocation-reusing forward execution of compiled circuit walks.

use super::{Action, WalkError, WalkPlan, checked_capacity, read_bool};
use crate::circuit::boolean::RowKind;

/// Values produced by forward walks, in contiguous physical R1CS block order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForwardTrace {
    pub z: Vec<bool>,
    pub a_z: Vec<bool>,
    pub b_z: Vec<bool>,
    pub c_z: Vec<bool>,
}

impl WalkPlan {
    /// Generate one block without sparse matrix products.
    pub fn forward(&self, inputs: &[bool], k_log: usize) -> Result<ForwardTrace, WalkError> {
        let mut trace = ForwardTrace::default();
        self.forward_into(inputs, k_log, &mut trace)?;
        Ok(trace)
    }

    /// Generate one block into reusable destination vectors.
    ///
    /// All four vectors are resized and completely overwritten. Their
    /// allocations are reused when their capacities are sufficient. Preflight
    /// errors preserve `trace`; a failed constraint may partially overwrite it,
    /// so callers must use its contents only after `Ok(())`.
    pub fn forward_into(
        &self,
        inputs: &[bool],
        k_log: usize,
        trace: &mut ForwardTrace,
    ) -> Result<(), WalkError> {
        let capacity = self.validate_forward(inputs, k_log)?;
        trace.prepare(capacity);
        let mut temporaries = vec![false; self.stats.max_live_temporaries];
        self.execute_block(
            inputs,
            &mut trace.z,
            &mut trace.a_z,
            &mut trace.b_z,
            &mut trace.c_z,
            &mut temporaries,
        )
    }

    /// Generate a batch into newly allocated contiguous block vectors.
    pub fn forward_batch<I: AsRef<[bool]>>(
        &self,
        inputs: &[I],
        k_log: usize,
    ) -> Result<ForwardTrace, WalkError> {
        let mut trace = ForwardTrace::default();
        self.forward_batch_into(inputs, k_log, &mut trace)?;
        Ok(trace)
    }

    /// Generate a batch into reusable contiguous block vectors.
    ///
    /// The batch may contain any number of supplied blocks; a higher-level
    /// table assembler remains responsible for its padding policy. Inputs are
    /// validated before any destination is modified. A failed constraint may
    /// leave complete earlier blocks and a partial failing block in `trace`;
    /// callers must use its contents only after `Ok(())`.
    pub fn forward_batch_into<I: AsRef<[bool]>>(
        &self,
        inputs: &[I],
        k_log: usize,
        trace: &mut ForwardTrace,
    ) -> Result<(), WalkError> {
        let (capacity, total) = self.validate_batch(inputs, k_log)?;
        trace.prepare(total);
        let mut temporaries = vec![false; self.stats.max_live_temporaries];
        for (block, input) in inputs.iter().enumerate() {
            temporaries.fill(false);
            let range = block * capacity..(block + 1) * capacity;
            match self.execute_block(
                input.as_ref(),
                &mut trace.z[range.clone()],
                &mut trace.a_z[range.clone()],
                &mut trace.b_z[range.clone()],
                &mut trace.c_z[range],
                &mut temporaries,
            ) {
                Err(WalkError::UnsatisfiedRow(row)) => {
                    return Err(WalkError::UnsatisfiedBatchRow { block, row });
                }
                result => result?,
            }
        }
        Ok(())
    }

    /// Generate a batch in parallel into newly allocated block vectors.
    pub fn forward_batch_parallel<I: AsRef<[bool]> + Sync>(
        &self,
        inputs: &[I],
        k_log: usize,
    ) -> Result<ForwardTrace, WalkError> {
        let mut trace = ForwardTrace::default();
        self.forward_batch_into_parallel(inputs, k_log, &mut trace)?;
        Ok(trace)
    }

    /// Parallel counterpart of [`Self::forward_batch_into`].
    ///
    /// Each worker owns one reusable temporary arena. Output blocks retain
    /// input order, and if several blocks reject, the lowest failing block is
    /// reported deterministically. Preflight errors preserve `trace`, but a
    /// constraint failure may leave any block partially or fully written;
    /// callers must use its contents only after `Ok(())`.
    pub fn forward_batch_into_parallel<I: AsRef<[bool]> + Sync>(
        &self,
        inputs: &[I],
        k_log: usize,
        trace: &mut ForwardTrace,
    ) -> Result<(), WalkError> {
        use rayon::prelude::*;

        let (capacity, total) = self.validate_batch(inputs, k_log)?;
        trace.prepare(total);
        let results: Vec<Result<(), WalkError>> = trace
            .z
            .par_chunks_mut(capacity)
            .zip(trace.a_z.par_chunks_mut(capacity))
            .zip(trace.b_z.par_chunks_mut(capacity))
            .zip(trace.c_z.par_chunks_mut(capacity))
            .zip(inputs.par_iter())
            .map_init(
                || vec![false; self.stats.max_live_temporaries],
                |temporaries, ((((z, a_z), b_z), c_z), input)| {
                    temporaries.fill(false);
                    self.execute_block(input.as_ref(), z, a_z, b_z, c_z, temporaries)
                },
            )
            .collect();

        for (block, result) in results.into_iter().enumerate() {
            match result {
                Err(WalkError::UnsatisfiedRow(row)) => {
                    return Err(WalkError::UnsatisfiedBatchRow { block, row });
                }
                result => result?,
            }
        }
        Ok(())
    }

    fn validate_forward(&self, inputs: &[bool], k_log: usize) -> Result<usize, WalkError> {
        self.validate_input(inputs)?;
        self.validate_capacity(k_log)
    }

    pub(super) fn validate_input(&self, inputs: &[bool]) -> Result<(), WalkError> {
        if inputs.len() != self.input_positions.len() {
            return Err(WalkError::InputCount {
                expected: self.input_positions.len(),
                actual: inputs.len(),
            });
        }
        Ok(())
    }

    fn validate_batch<I: AsRef<[bool]>>(
        &self,
        inputs: &[I],
        k_log: usize,
    ) -> Result<(usize, usize), WalkError> {
        let capacity = self.validate_capacity(k_log)?;
        for input in inputs {
            self.validate_input(input.as_ref())?;
        }
        let total = capacity
            .checked_mul(inputs.len())
            .ok_or(WalkError::BatchSizeOverflow {
                blocks: inputs.len(),
                block_capacity: capacity,
            })?;
        Ok((capacity, total))
    }

    pub(super) fn validate_capacity(&self, k_log: usize) -> Result<usize, WalkError> {
        let capacity = checked_capacity(k_log).ok_or(WalkError::InvalidKLog(k_log))?;
        if capacity < self.useful_bits {
            return Err(WalkError::Capacity {
                required: self.useful_bits,
                actual: capacity,
            });
        }
        Ok(capacity)
    }

    fn execute_block(
        &self,
        inputs: &[bool],
        z: &mut [bool],
        a_z: &mut [bool],
        b_z: &mut [bool],
        c_z: &mut [bool],
        temporaries: &mut [bool],
    ) -> Result<(), WalkError> {
        z[self.one_position] = true;
        for (&position, &input) in self.input_positions.iter().zip(inputs) {
            z[position] = input;
        }
        // Honest initialization, not constant folding: transpose still sees these boundaries.
        for &position in &self.initialized_zero_positions {
            z[position] = false;
        }

        for action in &self.actions {
            match action {
                Action::Xor { output, terms } => {
                    let value = terms.iter().fold(false, |acc, source| {
                        acc ^ read_bool(*source, z, temporaries)
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
                    let lhs = read_bool(*lhs, z, temporaries);
                    let rhs = read_bool(*rhs, z, temporaries);
                    match kind {
                        RowKind::And | RowKind::Materialize => {
                            z[output.expect("definitional row must have an output")] = lhs & rhs;
                        }
                        RowKind::One | RowKind::Input | RowKind::Constraint => {}
                    }
                    let result = read_bool(*result, z, temporaries);
                    if lhs & rhs != result {
                        return Err(WalkError::UnsatisfiedRow(*id));
                    }
                    a_z[*physical_row] = lhs;
                    b_z[*physical_row] = rhs;
                    c_z[*physical_row] = result;
                }
            }
        }
        Ok(())
    }
}

impl ForwardTrace {
    fn prepare(&mut self, len: usize) {
        for values in [&mut self.z, &mut self.a_z, &mut self.b_z, &mut self.c_z] {
            values.resize(len, false);
            values.fill(false);
        }
    }
}

#[cfg(test)]
mod tests;
