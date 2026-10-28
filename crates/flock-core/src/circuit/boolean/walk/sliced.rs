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
//!
//! A program may also leave whole operations to native kernels
//! ([`super::super::OperationKernel`]): their rows are dropped, together
//! with every XOR only they needed, and one instruction at the operation's
//! place hands control to the caller, which fills the operation's result
//! positions before the walk reads them.

use std::ops::Range;

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
/// `site`: the caller fills kernel site `site`'s result positions.
const KERNEL: u32 = 6;

/// Positions of one operation left to a kernel: the block positions its
/// columns (and their definitional rows) occupy, and the positions of its
/// materialized result, inside that range.
#[derive(Clone, Debug)]
pub(crate) struct KernelSite {
    pub region: Range<usize>,
    pub results: Vec<usize>,
}

#[derive(Clone, Debug)]
enum Instr {
    Xor {
        out: u32,
        terms: Vec<u32>,
    },
    Define {
        out: u32,
        lhs: u32,
        rhs: u32,
        row: u32,
    },
    Check {
        out: Option<u32>,
        lhs: u32,
        rhs: u32,
        result: u32,
        row: u32,
        id: RowId,
    },
    Kernel(u32),
}

impl Instr {
    fn reads(&self) -> Vec<u32> {
        match self {
            Instr::Xor { terms, .. } => terms.clone(),
            Instr::Define { lhs, rhs, .. } => vec![*lhs, *rhs],
            Instr::Check {
                lhs, rhs, result, ..
            } => vec![*lhs, *rhs, *result],
            Instr::Kernel(_) => Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SlicedProgram {
    code: Vec<u32>,
    /// Words of the position region: the useful bits rounded up to 128.
    positions: usize,
    words: usize,
    /// The row of each checked instruction, for errors.
    checked: Vec<RowId>,
    one_position: usize,
    input_positions: Vec<usize>,
}

impl SlicedProgram {
    pub(crate) fn new(plan: &WalkPlan) -> Self {
        Self::with_kernels(plan, &[]).expect("a program without kernels always compiles")
    }

    /// The program with each site's rows left to its kernel. Fails if a row
    /// in a site is not definitional, or if anything outside a site reads a
    /// site position other than its results, or reads a result before the
    /// site's kernel runs.
    pub(crate) fn with_kernels(plan: &WalkPlan, sites: &[KernelSite]) -> Result<Self, String> {
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
        let site_of = |position: usize| sites.iter().position(|s| s.region.contains(&position));
        let mut program = Vec::with_capacity(plan.actions.len() + sites.len());
        let mut placed = vec![false; sites.len()];
        for action in &plan.actions {
            match action {
                Action::Xor { output, terms } => program.push(Instr::Xor {
                    out: index(Source::Temporary(*output)),
                    terms: terms.iter().map(|&t| index(t)).collect(),
                }),
                Action::Row {
                    id,
                    physical_row,
                    kind,
                    lhs,
                    rhs,
                    result,
                    output,
                } => {
                    let defines = matches!(kind, RowKind::And | RowKind::Materialize);
                    let output = output.filter(|_| defines);
                    let definitional =
                        matches!((output, result), (Some(out), Source::Value(p)) if *p == out);
                    if let Some(site) = site_of(*physical_row) {
                        if !definitional || output != Some(*physical_row) {
                            return Err(format!(
                                "row {} in a kernel site is not a definition of its own position",
                                id.index()
                            ));
                        }
                        if !std::mem::replace(&mut placed[site], true) {
                            program.push(Instr::Kernel(site as u32));
                        }
                        continue;
                    }
                    let row = *physical_row as u32;
                    program.push(match output {
                        Some(out) if definitional => Instr::Define {
                            out: out as u32,
                            lhs: index(*lhs),
                            rhs: index(*rhs),
                            row,
                        },
                        _ => Instr::Check {
                            out: output.map(|o| o as u32),
                            lhs: index(*lhs),
                            rhs: index(*rhs),
                            result: index(*result),
                            row,
                            id: *id,
                        },
                    });
                }
            }
        }
        if let Some(site) = placed.iter().position(|&p| !p) {
            return Err(format!("kernel site {site} has no rows"));
        }
        // Drop the XORs nothing kept reads: walk backwards with the set of
        // temporaries still to be read.
        let mut live = vec![false; temporaries];
        let mut kept = vec![true; program.len()];
        for (i, instr) in program.iter().enumerate().rev() {
            if let Instr::Xor { out, .. } = instr {
                let slot = *out as usize - positions;
                if !std::mem::replace(&mut live[slot], false) {
                    kept[i] = false;
                    continue;
                }
            }
            for read in instr.reads() {
                if (positions..zero).contains(&(read as usize)) {
                    live[read as usize - positions] = true;
                }
            }
        }
        let program: Vec<Instr> = program
            .into_iter()
            .zip(kept)
            .filter_map(|(instr, keep)| keep.then_some(instr))
            .collect();
        // Nothing kept may read a site's internals, or its results early.
        let mut ran = vec![false; sites.len()];
        for instr in &program {
            if let Instr::Kernel(site) = instr {
                ran[*site as usize] = true;
            }
            for read in instr.reads() {
                if let Some(site) = site_of(read as usize) {
                    if !sites[site].results.contains(&(read as usize)) {
                        return Err(format!(
                            "position {read} inside kernel site {site} is read outside it"
                        ));
                    }
                    if !ran[site] {
                        return Err(format!(
                            "result position {read} of kernel site {site} is read before it runs"
                        ));
                    }
                }
            }
        }
        let mut code = Vec::new();
        let mut checked = Vec::new();
        for instr in program {
            match instr {
                Instr::Xor { out, terms } => {
                    code.push(match terms.len() {
                        2 => XOR2,
                        3 => XOR3,
                        n => XORN | (n as u32) << 3,
                    });
                    code.push(out);
                    code.extend(terms);
                }
                Instr::Define { out, lhs, rhs, row } => {
                    code.extend([DEFINE, out, lhs, rhs, row]);
                }
                Instr::Check {
                    out,
                    lhs,
                    rhs,
                    result,
                    row,
                    id,
                } => {
                    let check = checked.len() as u32;
                    checked.push(id);
                    match out {
                        Some(out) => code.extend([DEFINE_CHECK, out]),
                        None => code.push(CHECK),
                    }
                    code.extend([lhs, rhs, result, row, check]);
                }
                Instr::Kernel(site) => code.extend([KERNEL, site]),
            }
        }
        Ok(Self {
            code,
            positions,
            words: zero + 1,
            checked,
            one_position: plan.one_position,
            input_positions: plan.input_positions.clone(),
        })
    }

    /// Runs the program over 64 lanes. On return `z[..positions()]` holds
    /// the block positions (zero where nothing is placed, and at every site
    /// position but those `kernel` writes) and `a_z`, `b_z` (resized to
    /// `positions()`) the walked rows' sides; the rest of `z` is scratch.
    /// `kernel(site, z)` must write the site's result positions of `z`.
    pub(crate) fn run(
        &self,
        inputs: &[u64],
        live: u64,
        z: &mut Vec<u64>,
        a_z: &mut Vec<u64>,
        b_z: &mut Vec<u64>,
        kernel: &mut dyn FnMut(usize, &mut [u64]),
    ) -> Result<(), WalkError> {
        if inputs.len() != self.input_positions.len() {
            return Err(WalkError::InputCount {
                expected: self.input_positions.len(),
                actual: inputs.len(),
            });
        }
        z.clear();
        z.resize(self.words, 0);
        for v in [&mut *a_z, &mut *b_z] {
            v.clear();
            v.resize(self.positions, 0);
        }
        z[self.one_position] = live;
        for (&position, &input) in self.input_positions.iter().zip(inputs) {
            z[position] = input & live;
        }
        let (z, a_z, b_z) = (&mut z[..], &mut a_z[..], &mut b_z[..]);
        let code = &self.code[..];
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
                            row: self.checked[at(base + 4)],
                        });
                    }
                    a_z[at(base + 3)] = lhs;
                    b_z[at(base + 3)] = rhs;
                    pc = base + 5;
                }
                KERNEL => {
                    kernel(at(pc + 1), &mut z[..self.positions]);
                    pc += 2;
                }
                _ => unreachable!("unknown walk instruction"),
            }
        }
        Ok(())
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
        self.sliced()
            .run(inputs, live, z, a_z, b_z, &mut |_, _| unreachable!())
    }
}
