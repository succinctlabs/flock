use flock_core::circuit::{boolean::BooleanGate, builder::GateType};
use flock_field::F128;

use super::*;
use crate::{
    leanxmss::{
        circuit::{keyless_batch, word},
        reference::{
            HEIGHT, XmssHash, chain_address, codeword, layer_root, le128, leaf_address,
            msg_address, th,
        },
    },
    r1cs_hashes::blake3::K_LOG,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn word(&mut self) -> F128 {
        F128::new(self.next(), self.next())
    }
}

fn walk(gate: &BooleanGate, inputs: &[F128]) -> Vec<F128> {
    let mut out = Vec::new();
    gate.eval_walk(inputs, &mut out);
    out
}

fn run_native(inputs: &[F128]) -> Vec<F128> {
    let mut out = Vec::new();
    native(inputs, &mut out);
    out
}

/// One layer of one signature: its digits, chain values, `P`, chain address
/// at step 0, leaf address and the leaf the reference computes.
struct Layer {
    digits: [u8; V],
    y: [Value; V],
    pp: Value,
    a0: Value,
    a_leaf: Value,
    leaf: Value,
}

fn layers(n: usize, seed: u64) -> Vec<Layer> {
    let hash = XmssHash::Blake3;
    let (statements, sigs) = keyless_batch(n, seed);
    let mut out = Vec::new();
    for (st, sig) in statements.iter().zip(&sigs) {
        let pp = st.pk.pp;
        let mut msg = st.msg.to_vec();
        for layer in 0..2 {
            let e = if layer == 0 {
                sig.idx
            } else {
                sig.idx >> HEIGHT
            };
            let ls = &sig.layers[layer];
            let digest = th(hash, &pp, &msg_address(layer, e), &[&msg, &ls.rho]);
            let digits = codeword(&digest).unwrap();
            let ends: Vec<u8> = (0..V)
                .flat_map(|i| {
                    let mut c = ls.y[i];
                    for step in digits[i] as usize..CHAIN_END {
                        c = th(hash, &pp, &chain_address(layer, e, step), &[&c, &le128(i)]);
                    }
                    c
                })
                .collect();
            let leaf = th(hash, &pp, &leaf_address(layer, e), &[&ends]);
            out.push(Layer {
                digits,
                y: ls.y,
                pp,
                a0: chain_address(layer, e, 0),
                a_leaf: leaf_address(layer, e),
                leaf,
            });
            msg = layer_root(hash, &pp, layer, e, &msg, ls, &mut None)
                .unwrap()
                .to_vec();
        }
    }
    out
}

/// Runs a layer's rows with `eval`, returning every row's inputs and the
/// final state.
fn run_layer(
    layer: &Layer,
    mut eval: impl FnMut(&[F128]) -> Vec<F128>,
) -> (Vec<Vec<F128>>, Vec<F128>) {
    let digest = layer
        .digits
        .iter()
        .enumerate()
        .fold(0u128, |d, (i, &x)| d | (x as u128) << (W * i));
    let fresh = layer_fresh(&layer.digits, &layer.y);
    let mut state = vec![
        F128::ZERO,
        word(&layer.pp),
        word(&layer.a_leaf),
        fresh.b2,
        f(iv_word(0)),
        f(iv_word(1)),
        f(digest),
    ];
    let mut all = Vec::new();
    for row in &fresh.rows {
        let mut inputs = state.clone();
        inputs.extend([word(&layer.pp), word(&layer.a0)]);
        inputs.extend(row);
        state = eval(&inputs);
        all.push(inputs);
    }
    (all, state)
}

#[test]
fn a_layer_is_exactly_its_rows() {
    assert_eq!((STEPS, LEAF_BLOCKS, ROWS), (68, 15, 83));
    for layer in layers(64, 7) {
        let (_, state) = run_layer(&layer, run_native);
        assert_eq!(state[6], final_ctl());
        assert_eq!(state[4], word(&layer.leaf));
    }
}

#[test]
fn the_row_fits_a_blake3_block() {
    let gate = machine_gate(3).unwrap();
    eprintln!(
        "machine row: {} useful bits, k_log {}",
        gate.useful_bits(),
        gate.k_log()
    );
    assert_eq!(gate.k_log(), K_LOG);
}

#[test]
fn native_matches_the_circuit() {
    let gate = machine_gate(3).unwrap();
    // Reachable rows: every row of a few layers.
    for layer in layers(3, 11) {
        let (rows, _) = run_layer(&layer, |i| {
            let want = walk(&gate, i);
            assert_eq!(run_native(i), want);
            want
        });
        assert_eq!(rows.len(), ROWS);
    }
    // Arbitrary inputs, including unreachable control states.
    let mut rng = Rng(5);
    for _ in 0..256 {
        let inputs: Vec<F128> = (0..13).map(|_| rng.word()).collect();
        assert_eq!(run_native(&inputs), walk(&gate, &inputs));
        let mut out = Vec::new();
        gate.eval(&inputs, &(), &mut out);
    }
}

/// Witness packing cost: one full slot of machine rows through the generic
/// packer, against the hand-written BLAKE3 generator on as many plain
/// compressions; and the two relations' sizes.
#[test]
#[ignore = "timing; run explicitly"]
fn packing_cost() {
    use crate::r1cs_hashes::blake3::{
        Blake3Gate, build_block_r1cs, generate_witness_batch_major_partial_into,
    };
    use flock_core::union::SlotWitnessDest;
    use std::time::Instant;
    let nu = 14;
    let gate = machine_gate(nu).unwrap();
    let nnz =
        |m: &flock_core::r1cs::SparseBinaryMatrix| -> usize { m.rows.iter().map(Vec::len).sum() };
    let hand = build_block_r1cs(nu);
    eprintln!(
        "machine: useful {} nnz A {} B {}; blake3 gate: useful {} nnz A {} B {}",
        gate.useful_bits(),
        nnz(&gate.r1cs().a_0),
        nnz(&gate.r1cs().b_0),
        hand.useful_bits,
        nnz(&hand.a_0),
        nnz(&hand.b_0)
    );
    let mut rows: Vec<Box<[F128]>> = Vec::new();
    for layer in layers(4, 3) {
        run_layer(&layer, |i| {
            rows.push(i.into());
            run_native(i)
        });
    }
    let n = 1 << nu;
    let rows: Vec<Box<[F128]>> = rows.iter().cycle().take(n).cloned().collect();
    let len = (gate.useful_bits().div_ceil(128)) << nu;
    let len = len.max(128 << nu);
    let (mut z, mut a, mut b) = (
        vec![F128::ZERO; len],
        vec![F128::ZERO; len],
        vec![F128::ZERO; len],
    );
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for _ in 0..2 {
            let t = Instant::now();
            pool.install(|| {
                gate.pack_into(
                    &rows,
                    SlotWitnessDest {
                        z: &mut z,
                        a: &mut a,
                        b: &mut b,
                        elide_padding_writes: true,
                        dead_padding_unread: false,
                    },
                )
            });
            eprintln!(
                "machine pack_into, {threads} threads: {:.1} ms ({:.2} µs/row)",
                t.elapsed().as_secs_f64() * 1e3,
                t.elapsed().as_secs_f64() * 1e6 / n as f64
            );
        }
        let blocks: Vec<_> = rows
            .iter()
            .map(|r| {
                let mut out = Vec::new();
                Blake3Gate { nu }.eval(&r[..7], &(), &mut out)
            })
            .collect();
        for _ in 0..2 {
            let t = Instant::now();
            pool.install(|| {
                generate_witness_batch_major_partial_into(
                    &blocks,
                    nu,
                    SlotWitnessDest {
                        z: &mut z,
                        a: &mut a,
                        b: &mut b,
                        elide_padding_writes: true,
                        dead_padding_unread: false,
                    },
                )
            });
            eprintln!(
                "blake3 generator, {threads} threads: {:.1} ms ({:.2} µs/row)",
                t.elapsed().as_secs_f64() * 1e3,
                t.elapsed().as_secs_f64() * 1e6 / n as f64
            );
        }
    }
}

/// A plain row is one compression of `(h0, h1)` and `b0 ‖ b1 ‖ b2 ‖ v`.
#[test]
fn plain_rows_compress() {
    let gate = machine_gate(3).unwrap();
    let mut rng = Rng(9);
    for (flags, block_len) in [(1 | 2 | 8, 64), (1, 64), (2 | 8, 16)] {
        for _ in 0..8 {
            let mut inputs: Vec<F128> = (0..13).map(|_| rng.word()).collect();
            inputs[6] = plain_ctl(flags, block_len);
            let cv: [u32; 8] = std::array::from_fn(|i| u32s(u(inputs[4 + i / 4]))[i % 4]);
            let block = [inputs[1], inputs[2], inputs[3], inputs[0]];
            let m: [u32; 16] = std::array::from_fn(|i| u32s(u(block[i / 4]))[i % 4]);
            let out = flock_hash::blake3_compress(&cv, &m, 0, block_len, flags);
            let want = [f(word_of(&out[..4])), f(word_of(&out[4..8]))];
            let got = walk(&gate, &inputs);
            assert_eq!(run_native(&inputs), got);
            assert_eq!([got[4], got[5]], want, "flags {flags:#x}");
        }
    }
}
