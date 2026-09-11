use super::super::dsl::{Word, constant, materialize};
use super::columns::FirstC;
use super::{Blake3Cols, blake3};
use flock_core::circuit::boolean::{CircuitBuilder, Var};

pub(super) fn eval(builder: &mut CircuitBuilder, cols: &Blake3Cols<Var>) {
    let cv = cols.cv.map(|word| word.map(Into::into));
    let mut message = cols.message.map(|word| word.map(Into::into));
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        constant(builder, blake3::BLAKE3_IV[0]),
        constant(builder, blake3::BLAKE3_IV[1]),
        constant(builder, blake3::BLAKE3_IV[2]),
        constant(builder, blake3::BLAKE3_IV[3]),
        cols.counter[0].map(Into::into),
        cols.counter[1].map(Into::into),
        cols.block_len.map(Into::into),
        cols.flags.map(Into::into),
    ];
    for (round, gs) in cols.rounds.iter().enumerate() {
        for (i, g) in gs.iter().enumerate() {
            let [a_lane, b_lane, c_lane, d_lane] = blake3::G_LANES[i];
            let [mx, my] = blake3::G_MSG_IDX[i];
            let [a, b, c, d] = [state[a_lane], state[b_lane], state[c_lane], state[d_lane]];
            let a1 = g.a_first.eval(builder, a, b, message[mx]);
            let d1 = xor_rotate(builder, d, a1, 16);
            let c1 = match &g.c_first {
                FirstC::Constant(add) => add.eval(builder, blake3::BLAKE3_IV[i], d1),
                FirstC::Variable(add) => add.eval(builder, c, d1),
            };
            let b1 = xor_rotate(builder, b, c1, 12);
            let a2 = g.a_second.eval(builder, a1, b1, message[my]);
            let d2 = xor_rotate(builder, d1, a2, 8);
            let c2 = g.c_second.eval(builder, c1, d2);
            let b_new = xor_rotate(builder, b1, c2, 7);
            state[a_lane] = a2;
            state[b_lane] = b_new;
            state[c_lane] = c2;
            state[d_lane] = d2;
        }
        if round + 1 < cols.rounds.len() {
            message = std::array::from_fn(|i| message[blake3::MSG_PERMUTATION[i]]);
        }
    }
    // Keep expression and materialization order identical to the compatibility relation.
    let out_lo: [Word; 8] = std::array::from_fn(|i| builder.xor2_words(state[i], state[i + 8]));
    let out_hi: [Word; 8] = std::array::from_fn(|i| builder.xor2_words(state[i + 8], cv[i]));
    for (cols, word) in cols.out_lo.iter().zip(out_lo) {
        materialize(builder, cols, word);
    }
    for (cols, word) in cols.out_hi.iter().zip(out_hi) {
        materialize(builder, cols, word);
    }
}

fn xor_rotate(builder: &mut CircuitBuilder, x: Word, y: Word, amount: usize) -> Word {
    let mixed = builder.xor2_words(x, y);
    builder.rotate_right(mixed, amount)
}
