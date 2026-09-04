//! BLAKE3 compression expressed through the Boolean circuit DSL.
//!
//! State words stay virtual across all seven rounds. Only carry products and
//! the two output halves are materialized, matching the legacy relation.

use std::sync::{Arc, OnceLock};

use flock_core::circuit::boolean::{
    Bit, BooleanCircuit, CircuitBuilder, LinearExpr, PhysicalLayout, WalkPlan,
};
use flock_core::r1cs::BlockR1cs;

use super::{ProjectionCache, blake3};

type Word = [LinearExpr; blake3::WORD_BITS];
type MaterializedWord = [Bit; blake3::WORD_BITS];

const CV_PORT: &str = "cv";
const MESSAGE_PORT: &str = "message";
const COUNTER_PORT: &str = "counter";
const BLOCK_LEN_PORT: &str = "block_len";
const FLAGS_PORT: &str = "flags";
const OUT_LO_PORT: &str = "out_lo";
const OUT_HI_PORT: &str = "out_hi";

/// One reusable BLAKE3 DSL artifact and its legacy-compatible placement.
#[derive(Debug)]
pub struct Blake3DslCircuit {
    circuit: BooleanCircuit,
    layout: PhysicalLayout,
}

impl Blake3DslCircuit {
    pub fn circuit(&self) -> &BooleanCircuit {
        &self.circuit
    }

    pub fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    /// Lower to the same block-diagonal shape as the legacy relation.
    pub fn to_block_r1cs(&self, n_blocks_log: usize) -> BlockR1cs {
        assert!(
            n_blocks_log >= 3,
            "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
        );
        self.circuit
            .to_block_r1cs_with_layout(blake3::K_LOG, blake3::K_SKIP, n_blocks_log, &self.layout)
            .expect("the fixed BLAKE3 compatibility layout must be valid")
    }

    /// Evaluate one compression in legacy physical witness order.
    pub fn evaluate_block(
        &self,
        cv: &[u32; 8],
        message: &[u32; 16],
        counter: u64,
        block_len: u32,
        flags: u32,
    ) -> Vec<bool> {
        let inputs = block_inputs(cv, message, counter, block_len, flags);
        self.circuit
            .evaluate_r1cs_with_layout(&inputs, blake3::K_LOG, &self.layout)
            .expect("BLAKE3 has no rejecting general constraints")
    }

    /// Compile the structural execution plan at the compatibility layout.
    pub fn walk_plan(&self) -> WalkPlan {
        self.circuit
            .walk_plan_with_layout(&self.layout)
            .expect("the fixed BLAKE3 compatibility layout must be valid")
    }
}

/// Build the BLAKE3 DSL artifact once per process.
pub fn blake3_circuit() -> &'static Blake3DslCircuit {
    static CIRCUIT: OnceLock<Blake3DslCircuit> = OnceLock::new();
    CIRCUIT.get_or_init(build_blake3_circuit)
}

/// Cache only the compiled walk, dropping the construction artifact and its
/// normalized supports after compilation.
pub fn blake3_walk_projection() -> &'static WalkPlan {
    static WALK: OnceLock<WalkPlan> = OnceLock::new();
    WALK.get_or_init(|| build_blake3_circuit().walk_plan())
}

/// Return the cached relation for one batch shape without retaining the
/// construction artifact or structural walk.
pub fn blake3_relation_projection(n_blocks_log: usize) -> Arc<BlockR1cs> {
    assert!(
        n_blocks_log >= 3,
        "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
    );
    static RELATION: ProjectionCache<BlockR1cs> = ProjectionCache::new();
    RELATION.get_or_init(n_blocks_log, || {
        build_blake3_circuit().to_block_r1cs(n_blocks_log)
    })
}

fn block_inputs(
    cv: &[u32; 8],
    message: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> Vec<bool> {
    let mut inputs = Vec::with_capacity(28 * blake3::WORD_BITS);
    for &word in cv.iter().chain(message) {
        inputs.extend((0..blake3::WORD_BITS).map(|bit| word >> bit & 1 == 1));
    }
    inputs.extend((0..64).map(|bit| counter >> bit & 1 == 1));
    for word in [block_len, flags] {
        inputs.extend((0..blake3::WORD_BITS).map(|bit| word >> bit & 1 == 1));
    }
    inputs
}

/// Adds physical placement to ordinary DSL operations.
struct Blake3Builder {
    builder: CircuitBuilder,
    placements: Vec<(Bit, usize)>,
}

impl Blake3Builder {
    fn new() -> Self {
        Self {
            builder: CircuitBuilder::new(),
            placements: Vec::with_capacity(blake3::USEFUL_BITS),
        }
    }

    fn zero(&self) -> LinearExpr {
        self.builder.zero()
    }

    fn one(&self) -> LinearExpr {
        self.builder.one().expr()
    }

    fn zero_word(&self) -> Word {
        [self.zero(); blake3::WORD_BITS]
    }

    fn xor2(&mut self, lhs: LinearExpr, rhs: LinearExpr) -> LinearExpr {
        self.builder.xor2(lhs, rhs)
    }

    fn xor3(&mut self, first: LinearExpr, second: LinearExpr, third: LinearExpr) -> LinearExpr {
        self.builder.xor3(first, second, third)
    }

    fn xor_words(&mut self, lhs: &Word, rhs: &Word) -> Word {
        self.builder.xor2_words(*lhs, *rhs)
    }

    fn rotate_right(&self, word: &Word, amount: usize) -> Word {
        self.builder.rotate_right(*word, amount)
    }

    fn and_at(&mut self, lhs: LinearExpr, rhs: LinearExpr, position: usize) -> Bit {
        let bit = self.builder.and(lhs, rhs);
        self.placements.push((bit, position));
        bit
    }

    fn materialize_at(&mut self, value: LinearExpr, position: usize) -> Bit {
        let bit = self.builder.materialize(value);
        self.placements.push((bit, position));
        bit
    }

    fn finish(
        mut self,
        out_lo: [MaterializedWord; 8],
        out_hi: [MaterializedWord; 8],
    ) -> Blake3DslCircuit {
        let out_lo: [Bit; 8 * blake3::WORD_BITS] =
            std::array::from_fn(|i| out_lo[i / blake3::WORD_BITS][i % blake3::WORD_BITS]);
        let out_hi: [Bit; 8 * blake3::WORD_BITS] =
            std::array::from_fn(|i| out_hi[i / blake3::WORD_BITS][i % blake3::WORD_BITS]);
        self.builder
            .output_word_aligned(OUT_LO_PORT, blake3::SLOT_BITS, out_lo);
        self.builder
            .output_word_aligned(OUT_HI_PORT, blake3::SLOT_BITS / 2, out_hi);

        assert_eq!(self.builder.value_count(), blake3::USEFUL_BITS);
        assert_eq!(self.builder.row_count(), blake3::USEFUL_BITS);
        let one = self.builder.one().value_id();
        let circuit = self.builder.finish();

        let mut layout = circuit.layout();
        for (name, start) in [
            (CV_PORT, blake3::CV_BASE),
            (OUT_LO_PORT, blake3::OUT_LO_BASE),
            (MESSAGE_PORT, blake3::M_BASE),
            (COUNTER_PORT, blake3::T_LO_BASE),
            (BLOCK_LEN_PORT, blake3::BLEN_BASE),
            (FLAGS_PORT, blake3::FLAGS_BASE),
            (OUT_HI_PORT, blake3::OUT_HI_BASE),
        ] {
            layout
                .place_port(name, start)
                .expect("BLAKE3 port layout must be aligned and unique");
        }
        layout
            .place_definition(one, blake3::Z_CONST_POS)
            .expect("constant slot is unique");
        for (bit, position) in self.placements {
            layout
                .place_definition(bit.value_id(), position)
                .expect("BLAKE3 intermediate slots must be unique");
        }
        let layout = layout.finish().expect("BLAKE3 layout must be complete");
        assert_eq!(layout.useful_bits(), blake3::USEFUL_BITS);

        Blake3DslCircuit { circuit, layout }
    }
}

fn expressions(bits: &[Bit]) -> Word {
    assert_eq!(bits.len(), blake3::WORD_BITS);
    std::array::from_fn(|i| bits[i].expr())
}

fn constant_word(dsl: &Blake3Builder, value: u32) -> Word {
    std::array::from_fn(|bit| {
        if value >> bit & 1 == 1 {
            dsl.one()
        } else {
            dsl.zero()
        }
    })
}

fn xor_rotate(dsl: &mut Blake3Builder, lhs: &Word, rhs: &Word, amount: usize) -> Word {
    let mixed = dsl.xor_words(lhs, rhs);
    dsl.rotate_right(&mixed, amount)
}

fn add32_inline(dsl: &mut Blake3Builder, x: &Word, y: &Word, base: usize) -> Word {
    let mut sum = dsl.zero_word();
    let mut carry = dsl.zero();
    for bit in 0..blake3::WORD_BITS {
        sum[bit] = dsl.xor3(x[bit], y[bit], carry);
        if bit < blake3::CARRY_BITS_PER_ADD {
            let lhs = dsl.xor2(x[bit], carry);
            let rhs = dsl.xor2(y[bit], carry);
            let product = dsl.and_at(lhs, rhs, base + bit);
            carry = dsl.xor2(carry, product.expr());
        }
    }
    sum
}

/// Add a constant while omitting carries that remain affine.
fn add_const_inline(dsl: &mut Blake3Builder, constant: u32, y: &Word, base: usize) -> Word {
    let seed_bit = constant.trailing_zeros() as usize + 1;
    let mut sum = dsl.zero_word();
    let mut carry = dsl.zero();
    for bit in 0..blake3::WORD_BITS {
        if bit == seed_bit {
            carry = y[seed_bit - 1];
        }
        let constant_bit = if constant >> bit & 1 == 1 {
            dsl.one()
        } else {
            dsl.zero()
        };
        sum[bit] = dsl.xor3(constant_bit, y[bit], carry);
        if (seed_bit..blake3::CARRY_BITS_PER_ADD).contains(&bit) {
            let lhs = dsl.xor2(constant_bit, carry);
            let rhs = dsl.xor2(y[bit], carry);
            let product = dsl.and_at(lhs, rhs, base + bit - seed_bit);
            carry = dsl.xor2(carry, product.expr());
        }
    }
    sum
}

fn carry_save_layer(
    dsl: &mut Blake3Builder,
    x: &Word,
    y: &Word,
    shared: &Word,
    majority_base: usize,
) -> (Word, Word) {
    let mut parity = dsl.zero_word();
    let mut shifted_majority = dsl.zero_word();
    for bit in 0..blake3::WORD_BITS {
        parity[bit] = dsl.xor3(x[bit], y[bit], shared[bit]);
        if bit < blake3::CARRY_BITS_PER_ADD {
            let lhs = dsl.xor2(x[bit], shared[bit]);
            let rhs = dsl.xor2(y[bit], shared[bit]);
            let product = dsl.and_at(lhs, rhs, majority_base + bit);
            shifted_majority[bit + 1] = dsl.xor2(product.expr(), shared[bit]);
        }
    }
    (parity, shifted_majority)
}

fn carry_save_ripple(
    dsl: &mut Blake3Builder,
    parity: &Word,
    shifted_majority: &Word,
    ripple_base: usize,
) -> Word {
    let mut sum = dsl.zero_word();
    let mut carry = dsl.zero();
    for bit in 0..blake3::WORD_BITS {
        sum[bit] = dsl.xor3(parity[bit], shifted_majority[bit], carry);
        if (1..=blake3::RIPPLE_BITS_PER_FADD).contains(&bit) {
            let lhs = dsl.xor2(parity[bit], carry);
            let rhs = dsl.xor2(shifted_majority[bit], carry);
            let product = dsl.and_at(lhs, rhs, ripple_base + bit - 1);
            carry = dsl.xor2(carry, product.expr());
        }
    }
    sum
}

fn fused_add3_inline(
    dsl: &mut Blake3Builder,
    x: &Word,
    y: &Word,
    shared: &Word,
    majority_base: usize,
    ripple_base: usize,
) -> Word {
    let (parity, shifted_majority) = carry_save_layer(dsl, x, y, shared, majority_base);
    carry_save_ripple(dsl, &parity, &shifted_majority, ripple_base)
}

fn materialize_word(dsl: &mut Blake3Builder, word: &Word, base: usize) -> MaterializedWord {
    std::array::from_fn(|bit| dsl.materialize_at(word[bit], base + bit))
}

fn build_blake3_circuit() -> Blake3DslCircuit {
    let mut dsl = Blake3Builder::new();
    let cv_bits = dsl
        .builder
        .input_word_aligned::<{ 8 * blake3::WORD_BITS }>(CV_PORT, blake3::SLOT_BITS);
    let message_bits = dsl
        .builder
        .input_word_aligned::<{ 16 * blake3::WORD_BITS }>(MESSAGE_PORT, blake3::SLOT_BITS);
    let counter_bits = dsl
        .builder
        .input_word_aligned::<64>(COUNTER_PORT, blake3::WORD_BITS);
    let block_len_bits = dsl
        .builder
        .input_word_aligned::<{ blake3::WORD_BITS }>(BLOCK_LEN_PORT, blake3::WORD_BITS);
    let flags_bits = dsl
        .builder
        .input_word_aligned::<{ blake3::WORD_BITS }>(FLAGS_PORT, blake3::WORD_BITS);

    let cv: [Word; 8] = std::array::from_fn(|word| {
        expressions(&cv_bits[word * blake3::WORD_BITS..(word + 1) * blake3::WORD_BITS])
    });
    let mut message: [Word; 16] = std::array::from_fn(|word| {
        expressions(&message_bits[word * blake3::WORD_BITS..(word + 1) * blake3::WORD_BITS])
    });
    let counter_lo = expressions(&counter_bits[..blake3::WORD_BITS]);
    let counter_hi = expressions(&counter_bits[blake3::WORD_BITS..]);
    let block_len = expressions(&block_len_bits);
    let flags = expressions(&flags_bits);

    let mut state: [Word; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        constant_word(&dsl, blake3::BLAKE3_IV[0]),
        constant_word(&dsl, blake3::BLAKE3_IV[1]),
        constant_word(&dsl, blake3::BLAKE3_IV[2]),
        constant_word(&dsl, blake3::BLAKE3_IV[3]),
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];

    for round in 0..blake3::N_ROUNDS {
        for g_in_round in 0..blake3::N_G_PER_ROUND {
            let g = round * blake3::N_G_PER_ROUND + g_in_round;
            let [a_lane, b_lane, c_lane, d_lane] = blake3::G_LANES[g_in_round];
            let [mx, my] = blake3::G_MSG_IDX[g_in_round];
            let [a, b, c, d] = [state[a_lane], state[b_lane], state[c_lane], state[d_lane]];

            let a_1 = fused_add3_inline(
                &mut dsl,
                &a,
                &b,
                &message[mx],
                blake3::g_bit(g, blake3::OFF_MAJ1),
                blake3::g_bit(g, blake3::OFF_RIP1),
            );
            let d_1 = xor_rotate(&mut dsl, &d, &a_1, 16);
            // These are the only G calls whose c lane is still a known IV
            // word, so their first affine carries need no product rows.
            let c_1 = if g < 4 {
                add_const_inline(
                    &mut dsl,
                    blake3::BLAKE3_IV[g],
                    &d_1,
                    blake3::g_bit(g, blake3::OFF_C1),
                )
            } else {
                add32_inline(&mut dsl, &c, &d_1, blake3::g_bit(g, blake3::OFF_C1))
            };
            let b_1 = xor_rotate(&mut dsl, &b, &c_1, 12);
            let a_2 = fused_add3_inline(
                &mut dsl,
                &a_1,
                &b_1,
                &message[my],
                blake3::g_bit(g, blake3::off_maj2(g)),
                blake3::g_bit(g, blake3::off_rip2(g)),
            );
            let d_2 = xor_rotate(&mut dsl, &d_1, &a_2, 8);
            let c_2 = add32_inline(&mut dsl, &c_1, &d_2, blake3::g_bit(g, blake3::off_c2(g)));
            let b_new = xor_rotate(&mut dsl, &b_1, &c_2, 7);

            state[a_lane] = a_2;
            state[b_lane] = b_new;
            state[c_lane] = c_2;
            state[d_lane] = d_2;
        }
        if round + 1 < blake3::N_ROUNDS {
            message = std::array::from_fn(|i| message[blake3::MSG_PERMUTATION[i]]);
        }
    }

    let out_lo: [Word; 8] =
        std::array::from_fn(|word| dsl.xor_words(&state[word], &state[word + 8]));
    let out_hi: [Word; 8] = std::array::from_fn(|word| dsl.xor_words(&state[word + 8], &cv[word]));
    let out_lo = std::array::from_fn(|word| {
        materialize_word(
            &mut dsl,
            &out_lo[word],
            blake3::OUT_LO_BASE + word * blake3::WORD_BITS,
        )
    });
    let out_hi = std::array::from_fn(|word| {
        materialize_word(
            &mut dsl,
            &out_hi[word],
            blake3::OUT_HI_BASE + word * blake3::WORD_BITS,
        )
    });
    dsl.finish(out_lo, out_hi)
}

#[cfg(test)]
#[path = "blake3_dsl/tests.rs"]
mod tests;
