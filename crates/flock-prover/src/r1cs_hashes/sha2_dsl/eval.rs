use super::super::dsl::{Word, materialize};
use super::{Sha256Cols, sha2};
use flock_core::circuit::boolean::{CircuitBuilder, Var};

pub(super) fn eval(builder: &mut CircuitBuilder, cols: &Sha256Cols<Var>) {
    let h_in = cols.h_in.map(|word| word.map(Into::into));
    let mut schedule: Vec<Word> = cols.message.map(|word| word.map(Into::into)).to_vec();
    for (i, addition) in cols.schedule.iter().enumerate() {
        let t = i + 16;
        let sigma1 = small_sigma(builder, schedule[t - 2], 17, 19, 10);
        let sigma0 = small_sigma(builder, schedule[t - 15], 7, 18, 3);
        schedule.push(addition.eval(builder, sigma1, sigma0, schedule[t - 7], schedule[t - 16]));
    }
    let mut state = h_in;
    for (round, cols) in cols.rounds.iter().enumerate() {
        let [a, b, c, d, e, f, g, h] = state;
        for bit in 0..32 {
            let difference = builder.xor2(f[bit], g[bit]);
            builder.define_and(cols.choose_product[bit], e[bit], difference);
        }
        for bit in 0..32 {
            let ab = builder.xor2(a[bit], b[bit]);
            let ac = builder.xor2(a[bit], c[bit]);
            builder.define_and(cols.majority_product[bit], ab, ac);
        }
        let choose = builder.xor2_words(cols.choose_product, g);
        let majority = builder.xor2_words(cols.majority_product, a);
        let h_plus_k = cols.add_constant.eval(builder, sha2::SHA256_K[round], h);
        let sigma_e = big_sigma(builder, e, 6, 11, 25);
        let t1 = cols
            .t1
            .eval(builder, h_plus_k, sigma_e, choose, schedule[round]);
        let sigma_a = big_sigma(builder, a, 2, 13, 22);
        let mut a_new = cols.a_new.eval(builder, t1, sigma_a, majority);
        let mut e_new = cols.e_new.eval(builder, d, t1);
        if let Some(boundary) = &cols.state {
            a_new = materialize(builder, &boundary.a, a_new);
            e_new = materialize(builder, &boundary.e, e_new);
        }
        state = [a_new, a, b, c, e_new, e, f, g];
    }
    for i in 0..8 {
        let sum = cols.output_add[i].eval(builder, state[i], h_in[i]);
        materialize(builder, &cols.h_out[i], sum);
    }
}

fn big_sigma(builder: &mut CircuitBuilder, word: Word, x: usize, y: usize, z: usize) -> Word {
    builder.xor3_words(
        builder.rotate_right(word, x),
        builder.rotate_right(word, y),
        builder.rotate_right(word, z),
    )
}

fn small_sigma(builder: &mut CircuitBuilder, word: Word, x: usize, y: usize, shift: usize) -> Word {
    builder.xor3_words(
        builder.rotate_right(word, x),
        builder.rotate_right(word, y),
        builder.shift_right(word, shift),
    )
}

// TODO(rkm)
// think about whether or not we need to specify every column
// or just the "input" columns that are required for computing the circuit
// (potentially, later, we could be overwhelmed by the number of "witness"-like columns)
