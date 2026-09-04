//! SHA-256 compression expressed through the Boolean circuit DSL.
//!
//! The arithmetic below uses only typed bits, virtual linear expressions,
//! ANDs, and explicit materialization. Legacy physical indices appear only as
//! arguments to `and_at` and `materialize_at`, keeping compatibility layout
//! separate from the computation.

use std::sync::{Arc, OnceLock};

use flock_core::circuit::boolean::{
    Bit, BooleanCircuit, CircuitBuilder, LinearExpr, PhysicalLayout, WalkPlan,
};
use flock_core::r1cs::BlockR1cs;

use super::{ProjectionCache, sha2};

type Word = [LinearExpr; sha2::WORD_BITS];
type MaterializedWord = [Bit; sha2::WORD_BITS];

const H_PORT: &str = "h_in";
const MESSAGE_PORT: &str = "message";
const H_OUT_PORT: &str = "h_out";

/// One reusable SHA-256 DSL artifact and its legacy-compatible placement.
#[derive(Debug)]
pub struct Sha256DslCircuit {
    circuit: BooleanCircuit,
    layout: PhysicalLayout,
}

impl Sha256DslCircuit {
    pub fn circuit(&self) -> &BooleanCircuit {
        &self.circuit
    }

    pub fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    /// Lower the DSL artifact to the same block-diagonal shape as the legacy
    /// SHA-256 relation.
    pub fn to_block_r1cs(&self, n_blocks_log: usize) -> BlockR1cs {
        assert!(
            n_blocks_log >= 3,
            "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
        );
        self.circuit
            .to_block_r1cs_with_layout(sha2::K_LOG, sha2::K_SKIP, n_blocks_log, &self.layout)
            .expect("the fixed SHA-256 compatibility layout must be valid")
    }

    /// Reference evaluation of one compression, returned in legacy physical
    /// witness order and padded to `K` bits.
    pub fn evaluate_block(&self, h_in: &[u32; 8], message: &[u32; 16]) -> Vec<bool> {
        let inputs = block_inputs(h_in, message);
        self.circuit
            .evaluate_r1cs_with_layout(&inputs, sha2::K_LOG, &self.layout)
            .expect("SHA-256 has no rejecting general constraints")
    }

    /// Compile the structural forward/reverse execution plan at the legacy
    /// physical layout.
    pub fn walk_plan(&self) -> WalkPlan {
        self.circuit
            .walk_plan_with_layout(&self.layout)
            .expect("the fixed SHA-256 compatibility layout must be valid")
    }
}

/// Build the SHA-256 DSL artifact once per process.
pub fn sha256_circuit() -> &'static Sha256DslCircuit {
    static CIRCUIT: OnceLock<Sha256DslCircuit> = OnceLock::new();
    CIRCUIT.get_or_init(build_sha256_circuit)
}

/// Cache only the compiled walk, dropping the construction artifact and its
/// normalized supports after compilation.
pub fn sha256_walk_projection() -> &'static WalkPlan {
    static WALK: OnceLock<WalkPlan> = OnceLock::new();
    WALK.get_or_init(|| build_sha256_circuit().walk_plan())
}

/// Return the cached relation for one batch shape without retaining the
/// construction artifact or structural walk.
pub fn sha256_relation_projection(n_blocks_log: usize) -> Arc<BlockR1cs> {
    assert!(
        n_blocks_log >= 3,
        "lincheck needs n_outer >= 8; pick n_blocks_log >= 3"
    );
    static RELATION: ProjectionCache<BlockR1cs> = ProjectionCache::new();
    RELATION.get_or_init(n_blocks_log, || {
        build_sha256_circuit().to_block_r1cs(n_blocks_log)
    })
}

fn block_inputs(h_in: &[u32; 8], message: &[u32; 16]) -> Vec<bool> {
    let mut inputs = Vec::with_capacity((sha2::H_WORDS + sha2::M_WORDS) * sha2::WORD_BITS);
    for &word in h_in.iter().chain(message) {
        inputs.extend((0..sha2::WORD_BITS).map(|bit| word >> bit & 1 == 1));
    }
    inputs
}

/// Adds legacy placement to ordinary DSL operations without exposing raw
/// positions to the SHA arithmetic itself.
struct ShaBuilder {
    builder: CircuitBuilder,
    placements: Vec<(Bit, usize)>,
}

impl ShaBuilder {
    fn new() -> Self {
        Self {
            builder: CircuitBuilder::new(),
            placements: Vec::with_capacity(sha2::USEFUL_BITS),
        }
    }

    fn zero(&self) -> LinearExpr {
        self.builder.zero()
    }

    fn one(&self) -> LinearExpr {
        self.builder.one().expr()
    }

    fn zero_word(&self) -> Word {
        [self.zero(); sha2::WORD_BITS]
    }

    fn xor2(&mut self, a: LinearExpr, b: LinearExpr) -> LinearExpr {
        self.builder.xor2(a, b)
    }

    fn xor3(&mut self, a: LinearExpr, b: LinearExpr, c: LinearExpr) -> LinearExpr {
        self.builder.xor3(a, b, c)
    }

    fn xor_words(&mut self, a: &Word, b: &Word) -> Word {
        self.builder.xor2_words(*a, *b)
    }

    fn rotate_right(&self, word: &Word, amount: usize) -> Word {
        self.builder.rotate_right(*word, amount)
    }

    fn shift_right(&self, word: &Word, amount: usize) -> Word {
        self.builder.shift_right(*word, amount)
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

    fn finish(mut self, output: [MaterializedWord; sha2::N_OUT_WORDS]) -> Sha256DslCircuit {
        let output: [Bit; sha2::N_OUT_WORDS * sha2::WORD_BITS] =
            std::array::from_fn(|i| output[i / sha2::WORD_BITS][i % sha2::WORD_BITS]);
        self.builder
            .output_word_aligned(H_OUT_PORT, sha2::SLOT_BITS, output);

        assert_eq!(self.builder.value_count(), sha2::USEFUL_BITS);
        assert_eq!(self.builder.row_count(), sha2::USEFUL_BITS);
        let one = self.builder.one().value_id();
        let circuit = self.builder.finish();

        let mut layout = circuit.layout();
        layout
            .place_port(H_PORT, sha2::H_BASE)
            .expect("H input slot is aligned");
        layout
            .place_port(MESSAGE_PORT, sha2::M_BASE)
            .expect("message input slot is aligned");
        layout
            .place_port(H_OUT_PORT, sha2::H_OUT_BASE)
            .expect("H output slot is aligned");
        layout
            .place_definition(one, sha2::Z_CONST_POS)
            .expect("constant slot is unique");
        for (bit, position) in self.placements {
            layout
                .place_definition(bit.value_id(), position)
                .expect("SHA intermediate slots must be unique");
        }
        let layout = layout.finish().expect("SHA layout must be complete");
        assert_eq!(layout.useful_bits(), sha2::USEFUL_BITS);

        Sha256DslCircuit { circuit, layout }
    }
}

fn expressions(bits: &[Bit]) -> Word {
    assert_eq!(bits.len(), sha2::WORD_BITS);
    std::array::from_fn(|i| bits[i].expr())
}

fn rotate_xor3(dsl: &mut ShaBuilder, word: &Word, a: usize, b: usize, c: usize) -> Word {
    let a = dsl.rotate_right(word, a);
    let b = dsl.rotate_right(word, b);
    let c = dsl.rotate_right(word, c);
    dsl.builder.xor3_words(a, b, c)
}

fn sigma_xor(dsl: &mut ShaBuilder, word: &Word, a: usize, b: usize, shift: usize) -> Word {
    let a = dsl.rotate_right(word, a);
    let b = dsl.rotate_right(word, b);
    let shifted = dsl.shift_right(word, shift);
    dsl.builder.xor3_words(a, b, shifted)
}

fn small_sigma0(dsl: &mut ShaBuilder, word: &Word) -> Word {
    sigma_xor(dsl, word, 7, 18, 3)
}

fn small_sigma1(dsl: &mut ShaBuilder, word: &Word) -> Word {
    sigma_xor(dsl, word, 17, 19, 10)
}

fn big_sigma0(dsl: &mut ShaBuilder, word: &Word) -> Word {
    rotate_xor3(dsl, word, 2, 13, 22)
}

fn big_sigma1(dsl: &mut ShaBuilder, word: &Word) -> Word {
    rotate_xor3(dsl, word, 6, 11, 25)
}

/// A 32-bit add with only its 31 carry products materialized.
fn add32_inline(
    dsl: &mut ShaBuilder,
    x: &Word,
    y: &Word,
    carry_position: impl Fn(usize) -> usize,
) -> Word {
    let mut sum = dsl.zero_word();
    let mut carry = dsl.zero();
    for bit in 0..sha2::WORD_BITS {
        sum[bit] = dsl.xor3(x[bit], y[bit], carry);
        if bit < sha2::CARRIES_PER_ADD {
            let lhs = dsl.xor2(x[bit], carry);
            let rhs = dsl.xor2(y[bit], carry);
            let product = dsl.and_at(lhs, rhs, carry_position(bit));
            carry = dsl.xor2(carry, product.expr());
        }
    }
    sum
}

/// Add a compile-time constant, omitting carries below its first possible
/// nonlinear carry exactly as the legacy SHA relation does.
fn add_const_inline(dsl: &mut ShaBuilder, constant: u32, y: &Word, base: usize) -> Word {
    let seed_bit = constant.trailing_zeros() as usize + 1;
    let mut sum = dsl.zero_word();
    let mut carry = dsl.zero();
    for bit in 0..sha2::WORD_BITS {
        if bit == seed_bit {
            carry = y[seed_bit - 1];
        }
        let constant_bit = if constant >> bit & 1 == 1 {
            dsl.one()
        } else {
            dsl.zero()
        };
        sum[bit] = dsl.xor3(constant_bit, y[bit], carry);
        if (seed_bit..sha2::CARRIES_PER_ADD).contains(&bit) {
            let lhs = dsl.xor2(constant_bit, carry);
            let rhs = dsl.xor2(y[bit], carry);
            let product = dsl.and_at(lhs, rhs, base + bit - seed_bit);
            carry = dsl.xor2(carry, product.expr());
        }
    }
    sum
}

/// One carry-save layer. The majority products are materialized; both output
/// words remain virtual.
fn carry_save_layer(
    dsl: &mut ShaBuilder,
    x: &Word,
    y: &Word,
    shared: &Word,
    majority_base: usize,
) -> (Word, Word) {
    let mut parity = dsl.zero_word();
    let mut shifted_majority = dsl.zero_word();
    for bit in 0..sha2::WORD_BITS {
        parity[bit] = dsl.xor3(x[bit], y[bit], shared[bit]);
        if bit < sha2::CARRIES_PER_ADD {
            let lhs = dsl.xor2(x[bit], shared[bit]);
            let rhs = dsl.xor2(y[bit], shared[bit]);
            let product = dsl.and_at(lhs, rhs, majority_base + bit);
            shifted_majority[bit + 1] = dsl.xor2(product.expr(), shared[bit]);
        }
    }
    (parity, shifted_majority)
}

fn carry_save_ripple(
    dsl: &mut ShaBuilder,
    parity: &Word,
    shifted_majority: &Word,
    ripple_base: usize,
) -> Word {
    let mut sum = dsl.zero_word();
    let mut carry = dsl.zero();
    for bit in 0..sha2::WORD_BITS {
        sum[bit] = dsl.xor3(parity[bit], shifted_majority[bit], carry);
        if (1..=sha2::RIPPLE_BITS).contains(&bit) {
            let lhs = dsl.xor2(parity[bit], carry);
            let rhs = dsl.xor2(shifted_majority[bit], carry);
            let product = dsl.and_at(lhs, rhs, ripple_base + bit - 1);
            carry = dsl.xor2(carry, product.expr());
        }
    }
    sum
}

fn fused_add3_inline(
    dsl: &mut ShaBuilder,
    x: &Word,
    y: &Word,
    shared: &Word,
    majority_base: usize,
    ripple_base: usize,
) -> Word {
    let (parity, shifted_majority) = carry_save_layer(dsl, x, y, shared, majority_base);
    carry_save_ripple(dsl, &parity, &shifted_majority, ripple_base)
}

#[allow(clippy::too_many_arguments)]
fn fused_add4_inline(
    dsl: &mut ShaBuilder,
    x: &Word,
    y: &Word,
    shared_first: &Word,
    shared_second: &Word,
    majority_first_base: usize,
    majority_second_base: usize,
    ripple_base: usize,
) -> Word {
    let (parity_first, majority_first) =
        carry_save_layer(dsl, x, y, shared_first, majority_first_base);
    let (parity_second, majority_second) = carry_save_layer(
        dsl,
        &parity_first,
        &majority_first,
        shared_second,
        majority_second_base,
    );
    carry_save_ripple(dsl, &parity_second, &majority_second, ripple_base)
}

fn materialize_word(
    dsl: &mut ShaBuilder,
    word: &Word,
    position: impl Fn(usize) -> usize,
) -> MaterializedWord {
    std::array::from_fn(|bit| dsl.materialize_at(word[bit], position(bit)))
}

fn add32_materialized(
    dsl: &mut ShaBuilder,
    x: &Word,
    y: &Word,
    carry_position: impl Fn(usize) -> usize,
    sum_position: impl Fn(usize) -> usize,
) -> MaterializedWord {
    let sum = add32_inline(dsl, x, y, carry_position);
    materialize_word(dsl, &sum, sum_position)
}

fn build_sha256_circuit() -> Sha256DslCircuit {
    let mut dsl = ShaBuilder::new();
    let h_bits = dsl
        .builder
        .input_word_aligned::<{ sha2::H_WORDS * sha2::WORD_BITS }>(H_PORT, sha2::SLOT_BITS);
    let message_bits = dsl
        .builder
        .input_word_aligned::<{ sha2::M_WORDS * sha2::WORD_BITS }>(MESSAGE_PORT, sha2::SLOT_BITS);
    let h_in: [Word; sha2::H_WORDS] = std::array::from_fn(|word| {
        expressions(&h_bits[word * sha2::WORD_BITS..(word + 1) * sha2::WORD_BITS])
    });
    let mut schedule: Vec<Word> = (0..sha2::M_WORDS)
        .map(|word| {
            expressions(&message_bits[word * sha2::WORD_BITS..(word + 1) * sha2::WORD_BITS])
        })
        .collect();

    for t in 16..64 {
        let sigma1 = small_sigma1(&mut dsl, &schedule[t - 2]);
        let sigma0 = small_sigma0(&mut dsl, &schedule[t - 15]);
        let word = fused_add4_inline(
            &mut dsl,
            &sigma1,
            &sigma0,
            &schedule[t - 7],
            &schedule[t - 16],
            sha2::sched_bit(t, sha2::SC_MAJ1),
            sha2::sched_bit(t, sha2::SC_MAJ2),
            sha2::sched_bit(t, sha2::SC_RIP),
        );
        schedule.push(word);
    }

    let mut state = h_in;
    for round in 0..sha2::N_ROUNDS {
        let [a, b, c, d, e, f, g, h] = state;

        let choose_products: Word = std::array::from_fn(|bit| {
            let f_xor_g = dsl.xor2(f[bit], g[bit]);
            dsl.and_at(e[bit], f_xor_g, sha2::ch_and_bit(round, bit))
                .expr()
        });
        let majority_products: Word = std::array::from_fn(|bit| {
            let a_xor_b = dsl.xor2(a[bit], b[bit]);
            let a_xor_c = dsl.xor2(a[bit], c[bit]);
            dsl.and_at(a_xor_b, a_xor_c, sha2::maj_and_bit(round, bit))
                .expr()
        });
        let choose = dsl.xor_words(&choose_products, &g);
        let majority = dsl.xor_words(&majority_products, &a);

        let h_plus_k = add_const_inline(
            &mut dsl,
            sha2::SHA256_K[round],
            &h,
            sha2::round_bit(round, sha2::RC_ADDK),
        );
        let sigma1_e = big_sigma1(&mut dsl, &e);
        let t1 = fused_add4_inline(
            &mut dsl,
            &h_plus_k,
            &sigma1_e,
            &choose,
            &schedule[round],
            sha2::round_bit(round, sha2::RC_MAJ1),
            sha2::round_bit(round, sha2::RC_MAJ2),
            sha2::round_bit(round, sha2::RC_RIP),
        );
        let sigma0_a = big_sigma0(&mut dsl, &a);
        let a_raw = fused_add3_inline(
            &mut dsl,
            &t1,
            &sigma0_a,
            &majority,
            sha2::round_bit(round, sha2::RC_AMAJ),
            sha2::round_bit(round, sha2::RC_ARIP),
        );
        let e_raw = add32_inline(&mut dsl, &d, &t1, |bit| {
            sha2::round_bit(round, sha2::RC_ENEW + bit)
        });

        let materialize_state = round % sha2::EA_PERIOD == sha2::EA_PERIOD - 1;
        let a_new = if materialize_state {
            materialize_word(&mut dsl, &a_raw, |bit| sha2::a_new_bit(round, bit)).map(Bit::expr)
        } else {
            a_raw
        };
        let e_new = if materialize_state {
            materialize_word(&mut dsl, &e_raw, |bit| sha2::e_new_bit(round, bit)).map(Bit::expr)
        } else {
            e_raw
        };
        state = [a_new, a, b, c, e_new, e, f, g];
    }

    let output: [MaterializedWord; sha2::N_OUT_WORDS] = std::array::from_fn(|word| {
        add32_materialized(
            &mut dsl,
            &state[word],
            &h_in[word],
            |bit| sha2::out_carry_bit(word, bit),
            |bit| sha2::h_out_bit(word, bit),
        )
    });
    dsl.finish(output)
}

#[cfg(test)]
#[path = "sha2_dsl/tests.rs"]
mod tests;
