use crate::genus95_curve_code::field::{F128, F128Ext};

#[derive(Clone)]
pub(crate) struct ArtinSchreierSolver {
    solution_tables: [[u128; 256]; 16],
    /// The absolute-trace functional in the bit basis. `rhs` is solvable iff
    /// `parity_u128(rhs_bits & trace_mask) == 0`, i.e. it lies in the image of
    /// `x -> x^2 + x`. Equivalent to the old per-byte obstruction tables but a
    /// single AND + popcount instead of 16 lookups.
    trace_mask: u128,
}

impl ArtinSchreierSolver {
    pub(crate) fn new() -> Self {
        let mut rows = [0u128; 128];
        for column in 0..128 {
            let basis = F128::from_bits(1u128 << column);
            let image = (basis.square() + basis).to_bits();
            for (row, row_mask) in rows.iter_mut().enumerate() {
                if ((image >> row) & 1) != 0 {
                    *row_mask |= 1u128 << column;
                }
            }
        }

        let mut transforms = [0u128; 128];
        for (i, transform) in transforms.iter_mut().enumerate() {
            *transform = 1u128 << i;
        }

        let mut pivot_columns = [0usize; 128];
        let mut rank = 0usize;
        for column in 0..128 {
            let Some(pivot) = (rank..128).find(|&row| ((rows[row] >> column) & 1) != 0) else {
                continue;
            };
            rows.swap(rank, pivot);
            transforms.swap(rank, pivot);

            for row in 0..128 {
                if row != rank && ((rows[row] >> column) & 1) != 0 {
                    rows[row] ^= rows[rank];
                    transforms[row] ^= transforms[rank];
                }
            }

            pivot_columns[rank] = column;
            rank += 1;
        }

        debug_assert_eq!(rank, 127);
        let mut solution_tables = [[0u128; 256]; 16];
        for byte_index in 0..16 {
            let shift = 8 * byte_index;
            for byte in 0..256 {
                let rhs_part = (byte as u128) << shift;
                let mut solution = 0u128;
                for row in 0..rank {
                    if parity_u128(transforms[row] & rhs_part) {
                        solution |= 1u128 << pivot_columns[row];
                    }
                }
                solution_tables[byte_index][byte] = solution;
            }
        }

        Self {
            solution_tables,
            trace_mask: transforms[rank],
        }
    }

    pub(crate) fn solve(&self, rhs: F128) -> Option<F128> {
        let rhs_bits = rhs.to_bits();
        if parity_u128(rhs_bits & self.trace_mask) {
            return None;
        }

        let mut out = 0u128;
        for byte_index in 0..16 {
            let byte = ((rhs_bits >> (8 * byte_index)) & 0xff) as usize;
            out ^= self.solution_tables[byte_index][byte];
        }
        Some(F128::from_bits(out))
    }
}

#[inline(always)]
pub(crate) fn parity_u128(x: u128) -> bool {
    (x.count_ones() & 1) != 0
}
