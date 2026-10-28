//! The bit-sliced forward walk as one flat instruction stream.
//!
//! [`WalkPlan::forward_sliced_into`] interprets the plan's actions as they
//! were compiled: an enum per action, a boxed term list per XOR and separate
//! value and temporary arrays. That is fine for a block at a time but
//! dominates a packer that walks every row of a slot. Here the same actions
//! become `u32` words over one array of 64-lane words: the block's positions
//! (rounded up to whole 128-bit columns), then the temporaries, then a word
//! that stays zero. A row whose `C` side is its own freshly written output
//! cannot fail, so it carries no check; any other row is checked exactly as
//! the action walk checks it.

use super::{Action, Source, WalkError, WalkPlan};
use crate::circuit::boolean::{RowId, RowKind};

const XOR2: u32 = 0;
const XOR3: u32 = 1;
/// `XORN | n << 3`: an XOR of `n` terms.
const XORN: u32 = 2;
/// `out, lhs, rhs, row`: `out = lhs & rhs` and the row's `A·z`, `B·z`.
const DEFINE: u32 = 3;
/// `lhs, rhs, result, row, check`: records the row, then checks it.
const CHECK: u32 = 4;
/// `out, lhs, rhs, result, row, check`: defines `out`, then checks the row.
const DEFINE_CHECK: u32 = 5;

#[derive(Clone, Debug)]
pub(super) struct SlicedProgram {
    code: Vec<u32>,
    /// Words of the position region: the useful bits rounded up to 128.
    positions: usize,
    words: usize,
    /// The row of each checked instruction, for errors.
    checked: Vec<RowId>,
}

impl SlicedProgram {
    pub(super) fn new(plan: &WalkPlan) -> Self {
        let positions = plan.useful_bits.div_ceil(128) * 128;
        let temporaries = plan.stats.max_live_temporaries;
        let zero = positions + temporaries;
        let index = |source: Source| -> u32 {
            let i = match source {
                Source::Zero => zero,
                Source::Value(position) => position,
                Source::Temporary(slot) => positions + slot,
            };
            u32::try_from(i).expect("walk positions fit in u32")
        };
        let mut code = Vec::new();
        let mut checked = Vec::new();
        for action in &plan.actions {
            match action {
                Action::Xor { output, terms } => {
                    let op = match terms.len() {
                        2 => XOR2,
                        3 => XOR3,
                        n => XORN | (n as u32) << 3,
                    };
                    code.push(op);
                    code.push(index(Source::Temporary(*output)));
                    code.extend(terms.iter().map(|&t| index(t)));
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
                    let row = *physical_row as u32;
                    let defines = matches!(kind, RowKind::And | RowKind::Materialize);
                    let output = output.filter(|_| defines);
                    match output {
                        Some(out) if matches!(result, Source::Value(p) if *p == out) => {
                            code.extend([DEFINE, out as u32, index(*lhs), index(*rhs), row]);
                        }
                        _ => {
                            let check = checked.len() as u32;
                            checked.push(*id);
                            if let Some(out) = output {
                                code.extend([DEFINE_CHECK, out as u32]);
                            } else {
                                code.push(CHECK);
                            }
                            code.extend([index(*lhs), index(*rhs), index(*result), row, check]);
                        }
                    }
                }
            }
        }
        Self {
            code,
            positions,
            words: zero + 1,
            checked,
        }
    }
}

impl WalkPlan {
    /// Words of the position region [`Self::forward_sliced_compact`] fills:
    /// the useful bits rounded up to whole 128-bit columns.
    pub fn sliced_positions(&self) -> usize {
        self.sliced().positions
    }

    fn sliced(&self) -> &SlicedProgram {
        self.sliced.get_or_init(|| SlicedProgram::new(self))
    }

    /// [`Self::forward_sliced_into`] over the flat instruction stream. On
    /// return `z[..sliced_positions()]` holds the block positions (zero where
    /// nothing is placed) and `a_z`, `b_z` (resized to `sliced_positions()`)
    /// the rows' sides; the rest of `z` is scratch.
    pub fn forward_sliced_compact(
        &self,
        inputs: &[u64],
        live: u64,
        z: &mut Vec<u64>,
        a_z: &mut Vec<u64>,
        b_z: &mut Vec<u64>,
    ) -> Result<(), WalkError> {
        if inputs.len() != self.input_positions.len() {
            return Err(WalkError::InputCount {
                expected: self.input_positions.len(),
                actual: inputs.len(),
            });
        }
        let program = self.sliced();
        z.clear();
        z.resize(program.words, 0);
        for v in [&mut *a_z, &mut *b_z] {
            v.clear();
            v.resize(program.positions, 0);
        }
        z[self.one_position] = live;
        for (&position, &input) in self.input_positions.iter().zip(inputs) {
            z[position] = input & live;
        }
        let (z, a_z, b_z) = (&mut z[..], &mut a_z[..], &mut b_z[..]);
        let code = &program.code[..];
        let at = |i: usize| code[i] as usize;
        let mut pc = 0;
        while pc < code.len() {
            let op = code[pc];
            match op & 7 {
                XOR2 => {
                    z[at(pc + 1)] = z[at(pc + 2)] ^ z[at(pc + 3)];
                    pc += 4;
                }
                XOR3 => {
                    z[at(pc + 1)] = z[at(pc + 2)] ^ z[at(pc + 3)] ^ z[at(pc + 4)];
                    pc += 5;
                }
                XORN => {
                    let n = (op >> 3) as usize;
                    let value = code[pc + 2..pc + 2 + n]
                        .iter()
                        .fold(0, |acc, &t| acc ^ z[t as usize]);
                    z[at(pc + 1)] = value;
                    pc += 2 + n;
                }
                DEFINE => {
                    let (lhs, rhs) = (z[at(pc + 2)], z[at(pc + 3)]);
                    z[at(pc + 1)] = lhs & rhs;
                    a_z[at(pc + 4)] = lhs;
                    b_z[at(pc + 4)] = rhs;
                    pc += 5;
                }
                CHECK | DEFINE_CHECK => {
                    let define = op & 7 == DEFINE_CHECK;
                    let base = pc + 1 + define as usize;
                    let (lhs, rhs) = (z[at(base)], z[at(base + 1)]);
                    if define {
                        z[at(pc + 1)] = lhs & rhs;
                    }
                    let bad = ((lhs & rhs) ^ z[at(base + 2)]) & live;
                    if bad != 0 {
                        return Err(WalkError::UnsatisfiedBatchRow {
                            block: bad.trailing_zeros() as usize,
                            row: program.checked[at(base + 4)],
                        });
                    }
                    a_z[at(base + 3)] = lhs;
                    b_z[at(base + 3)] = rhs;
                    pc = base + 5;
                }
                _ => unreachable!("unknown walk instruction"),
            }
        }
        Ok(())
    }
}
