use flock_core::{
    circuit::{boolean::BooleanGate, builder::GateType},
    union::SlotWitnessDest,
};
use flock_field::F128;

use super::*;
use crate::leanxmss::reference::{
    chain_address, codeword, leaf_address, msg_address, node_address,
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

/// The outputs as the circuit computes them (the walk, not the native
/// evaluator the online phase uses).
fn eval(gate: &BooleanGate, inputs: &[F128]) -> Vec<F128> {
    let mut out = Vec::new();
    gate.eval_walk(inputs, &mut out);
    out
}

/// What the online phase computes: the native evaluator.
fn eval_native(gate: &BooleanGate, inputs: &[F128]) -> Vec<F128> {
    let mut out = Vec::new();
    gate.eval(inputs, &(), &mut out);
    out
}

fn word_of(v: &[u8; 16]) -> F128 {
    let x = u128::from_le_bytes(*v);
    F128::new(x as u64, (x >> 64) as u64)
}

fn bit(v: F128, j: usize) -> bool {
    (if j < 64 { v.lo >> j } else { v.hi >> (j - 64) }) & 1 == 1
}

#[test]
fn select_and_swap_match_their_semantics() {
    let mut rng = Rng(1);
    let gates = [0, 1, 2].map(|step| select_gate(step, 3).unwrap());
    let swap = swap_gate(3).unwrap();
    for _ in 0..64 {
        let (sel, x, h) = (rng.word(), rng.word(), rng.word());
        for (step, gate) in gates.iter().enumerate() {
            let want = if bit(sel, step) { h } else { x };
            assert_eq!(eval(gate, &[sel, x, h]), [want], "select{step}");
        }
        let (dir, node, sibling) = (rng.word(), rng.word(), rng.word());
        let want = if bit(dir, 0) {
            [sibling, node]
        } else {
            [node, sibling]
        };
        assert_eq!(eval(&swap, &[dir, node, sibling]), want);
    }
}

/// A random digest with the given digits (digits 58..63 zero).
fn digest_from_digits(digits: &[u8; V]) -> [u8; 16] {
    let d = digits
        .iter()
        .enumerate()
        .fold(0u128, |d, (i, &x)| d | (x as u128) << (W * i));
    d.to_le_bytes()
}

/// Random digits, nudged to sum to [`TARGET_SUM`].
fn random_codeword_digits(rng: &mut Rng) -> [u8; V] {
    let mut digits: [u8; V] = std::array::from_fn(|_| (rng.next() & 3) as u8);
    let mut sum: usize = digits.iter().map(|&x| x as usize).sum();
    for x in digits.iter_mut() {
        while sum < TARGET_SUM && *x < 3 {
            *x += 1;
            sum += 1;
        }
        while sum > TARGET_SUM && *x > 0 {
            *x -= 1;
            sum -= 1;
        }
    }
    digits
}

/// The spread chain over a codeword digest gives each chain's selectors, and
/// the check accepts exactly the codewords.
#[test]
fn codeword_check_and_selectors() {
    let check = codeword_gate(3).unwrap();
    let spread = spread_gate(3).unwrap();
    let mut rng = Rng(2);
    for _ in 0..32 {
        let digits = random_codeword_digits(&mut rng);
        let digest = digest_from_digits(&digits);
        assert_eq!(codeword(&digest), Some(digits));
        assert!(eval(&check, &[word_of(&digest)]).is_empty());
        let mut r = word_of(&digest);
        let mut sel = Vec::new();
        for _ in 0..SPREADS {
            let out = eval(&spread, &[r]);
            assert_eq!(out.len(), SPREAD_DIGITS + 1);
            sel.extend_from_slice(&out[..SPREAD_DIGITS]);
            r = out[SPREAD_DIGITS];
        }
        assert_eq!(r, F128::ZERO);
        for (i, &x) in digits.iter().enumerate() {
            for step in 0..CHAIN_END {
                assert_eq!(
                    bit(sel[i], step),
                    x as usize <= step,
                    "chain {i} step {step}"
                );
            }
            assert!((CHAIN_END..128).all(|j| !bit(sel[i], j)));
        }
    }
    // Wrong digit sum, and a nonzero high digit, are both rejected.
    let mut digits = random_codeword_digits(&mut rng);
    let i = digits.iter().position(|&x| x < 3).unwrap();
    digits[i] += 1;
    let off_by_one = digest_from_digits(&digits);
    let mut high = digest_from_digits(&random_codeword_digits(&mut rng));
    high[15] = 0x10;
    for digest in [off_by_one, high] {
        assert_eq!(codeword(&digest), None);
        for run in [eval, eval_native] {
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&check, &[word_of(&digest)])
            }));
            assert!(
                rejected.is_err(),
                "the codeword check accepted a non-codeword"
            );
        }
    }
}

/// The shift chain and the address gadgets give every address of both layers
/// and the Merkle directions, whatever the index word's high bits hold.
#[test]
fn addresses_match_the_reference() {
    let address = address_gate(3).unwrap();
    let step = address_step_gate(3).unwrap();
    let mut rng = Rng(3);
    for idx in [
        0u32,
        1,
        0xFFFF,
        0x1_0000,
        u32::MAX,
        rng.next() as u32,
        rng.next() as u32,
    ] {
        let mut idx_word = rng.word();
        idx_word.lo = (idx_word.lo & !0xFFFF_FFFF) | idx as u64;
        let mut shifted = vec![idx_word];
        let mut nodes = [Vec::new(), Vec::new()];
        for layer in 0..2 {
            let constants = layer_address_constants(layer);
            for level in 1..=HEIGHT {
                let out = eval(
                    &step,
                    &[*shifted.last().unwrap(), word_of(&constants[4 + level])],
                );
                shifted.push(out[0]);
                nodes[layer].push(out[1]);
            }
        }
        for j in 1..=2 * HEIGHT {
            assert_eq!(shifted[j], F128::new((idx as u64) >> j, 0), "R_{j}");
        }
        for j in 0..2 * HEIGHT {
            assert_eq!(bit(shifted[j], 0), (idx >> j) & 1 == 1, "direction {j}");
        }
        for layer in 0..2 {
            let e = if layer == 0 { idx } else { idx >> HEIGHT };
            let constants = layer_address_constants(layer);
            let a = |pos: usize| {
                eval(
                    &address,
                    &[shifted[HEIGHT * layer], word_of(&constants[pos])],
                )[0]
            };
            assert_eq!(a(0), word_of(&msg_address(layer, e)));
            for s in 0..CHAIN_END {
                assert_eq!(a(1 + s), word_of(&chain_address(layer, e, s)));
            }
            assert_eq!(a(4), word_of(&leaf_address(layer, e)));
            for level in 1..=HEIGHT {
                assert_eq!(
                    nodes[layer][level - 1],
                    word_of(&node_address(layer, e, level))
                );
            }
        }
    }
}

/// Every gadget keeps a small block: a slot's address space is 2^nu times
/// its block, whatever its row count.
#[test]
fn gadget_blocks_are_small() {
    let gates = [
        select_gate(0, 3).unwrap(),
        swap_gate(3).unwrap(),
        codeword_gate(3).unwrap(),
        spread_gate(3).unwrap(),
        address_gate(3).unwrap(),
        address_step_gate(3).unwrap(),
        lookup_gate(3).unwrap(),
    ];
    // The routing blocks carry eight positions each.
    let routing = [
        (benes_gate(false, 3).unwrap(), 13),
        (benes_gate(true, 3).unwrap(), 12),
    ];
    for (g, max) in gates
        .iter()
        .map(|g| (g, 11))
        .chain(routing.iter().map(|(g, m)| (g, *m)))
    {
        eprintln!(
            "{}: k_log {} ({} useful bits)",
            g.name(),
            g.k_log(),
            g.useful_bits()
        );
        assert!(g.k_log() <= max, "{} has a 2^{} block", g.name(), g.k_log());
    }
}

/// The packed slot holds, for each row, a block that satisfies the gadget's
/// relation with `A·z` and `B·z` alongside, dummy rows are zero, and the
/// stripe transposes `z`.
#[test]
fn packed_rows_satisfy_the_relation() {
    let nu = 4;
    let gate = select_gate(1, nu).unwrap();
    let single = select_gate(1, 0).unwrap();
    let r1cs = single.r1cs();
    let k_log = gate.k_log();
    let k = 1usize << k_log;
    let cols = k >> 7;
    let mut rng = Rng(4);
    let rows: Vec<Box<[F128]>> = (0..11)
        .map(|_| vec![rng.word(), rng.word(), rng.word()].into_boxed_slice())
        .collect();
    let len = cols << nu;
    let poison = F128::new(0xDEAD, 0xBEEF);
    let (mut z, mut a, mut b) = (vec![poison; len], vec![poison; len], vec![poison; len]);
    let stripe = gate.pack_into(
        &rows,
        SlotWitnessDest {
            z: &mut z,
            a: &mut a,
            b: &mut b,
            elide_padding_writes: false,
            dead_padding_unread: false,
        },
    );
    assert_eq!(stripe.len(), (1 << nu) / 8 * k);
    let bits_of = |buf: &[F128], row: usize| -> Vec<bool> {
        (0..k)
            .map(|c| bit(buf[((c / 128) << nu) + row], c % 128))
            .collect()
    };
    for row in 0..1 << nu {
        let zr = bits_of(&z, row);
        if row < rows.len() {
            assert!(r1cs.satisfies(&zr), "row {row}");
            assert_eq!(bits_of(&a, row), r1cs.apply_a(&zr), "row {row} A·z");
            assert_eq!(bits_of(&b, row), r1cs.apply_b(&zr), "row {row} B·z");
            let mut y = Vec::new();
            gate.eval(&rows[row], &(), &mut y);
            let out_col = gate.table().io_schema.last().unwrap().word_col;
            assert_eq!(z[(out_col << nu) + row], y[0], "row {row} output word");
        } else {
            assert!(zr.iter().all(|&x| !x), "dummy row {row} is not zero");
        }
        for (c, &zc) in zr.iter().enumerate().take(gate.useful_bits()) {
            let byte = stripe[(row / 8) * k + c];
            assert_eq!((byte >> (row % 8)) & 1 == 1, zc, "stripe row {row} bit {c}");
        }
    }
}

/// The forward block permutes values and index words exactly as the native
/// network does, and the reverse block with the same control word undoes it.
#[test]
fn benes_blocks_match_the_network() {
    use crate::leanxmss::routing::pair;
    let forward = benes_gate(false, 3).unwrap();
    let reverse = benes_gate(true, 3).unwrap();
    let mut rng = Rng(5);
    for _ in 0..64 {
        let ctrl = rng.next() & 0xFFF;
        let values: Vec<F128> = (0..8).map(|_| rng.word()).collect();
        let indices: Vec<u64> = (0..8).map(|_| rng.next() % 64).collect();
        let mut inputs = vec![F128::new(ctrl, 0)];
        inputs.extend(&values);
        inputs.extend(indices.iter().map(|&i| F128::new(i, 0)));
        let out = eval(&forward, &inputs);
        let mut items: Vec<(F128, u64)> = values
            .iter()
            .copied()
            .zip(indices.iter().copied())
            .collect();
        for s in 0..3 {
            for q in 0..4 {
                if (ctrl >> (4 * s + q)) & 1 == 1 {
                    let lo = pair(s, q);
                    items.swap(lo, lo | 1 << s);
                }
            }
        }
        for e in 0..8 {
            assert_eq!(out[e], items[e].0, "value {e}");
            assert_eq!(out[8 + e], F128::new(items[e].1, 0), "index {e}");
        }
        let mut back_in = vec![F128::new(ctrl, 0)];
        back_in.extend(&out[..8]);
        assert_eq!(eval(&reverse, &back_in), values, "reverse undoes forward");
    }
}

/// The lookup reads each index's digit (dummies as 3) and enforces its class.
#[test]
fn lookup_reads_digits_and_enforces_classes() {
    let gate = lookup_gate(3).unwrap();
    let mut rng = Rng(6);
    let digits = random_codeword_digits(&mut rng);
    let digest = word_of(&digest_from_digits(&digits));
    for index in 0..64u64 {
        let digit = digits.get(index as usize).copied().unwrap_or(3);
        let sel = eval(&gate, &[digest, F128::new(index, 0), F128::ZERO])[0];
        assert_eq!(
            sel,
            F128::new((digit == 0) as u64 | ((digit <= 1) as u64) << 1, 0)
        );
        for (class, allowed) in [(1u64, digit >= 1), (2, digit >= 2)] {
            for run in [eval, eval_native] {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&gate, &[digest, F128::new(index, 0), F128::new(class, 0)])
                }));
                assert_eq!(
                    r.is_ok(),
                    allowed,
                    "index {index} digit {digit} class {class}"
                );
            }
        }
    }
}

/// Every gadget's native evaluator agrees with its circuit on random inputs
/// (valid ones where the circuit constrains them).
#[test]
fn native_evaluation_matches_the_circuit() {
    let mut rng = Rng(7);
    let mut gates: Vec<(BooleanGate, Box<dyn Fn(&mut Rng) -> Vec<F128>>)> = Vec::new();
    for step in 0..CHAIN_END {
        gates.push((
            select_gate(step, 3).unwrap(),
            Box::new(|r| vec![r.word(), r.word(), r.word()]),
        ));
    }
    gates.push((
        swap_gate(3).unwrap(),
        Box::new(|r| vec![r.word(), r.word(), r.word()]),
    ));
    gates.push((
        codeword_gate(3).unwrap(),
        Box::new(|r| vec![word_of(&digest_from_digits(&random_codeword_digits(r)))]),
    ));
    gates.push((spread_gate(3).unwrap(), Box::new(|r| vec![r.word()])));
    gates.push((
        address_gate(3).unwrap(),
        Box::new(|r| vec![r.word(), r.word()]),
    ));
    gates.push((
        address_step_gate(3).unwrap(),
        Box::new(|r| vec![r.word(), r.word()]),
    ));
    for reverse in [false, true] {
        gates.push((
            benes_gate(reverse, 3).unwrap(),
            Box::new(move |r| {
                let mut v = vec![F128::new(r.next() & 0xFFF, 0)];
                v.extend((0..8).map(|_| r.word()));
                if !reverse {
                    v.extend((0..8).map(|_| F128::new(r.next() % 64, 0)));
                }
                v
            }),
        ));
    }
    gates.push((
        lookup_gate(3).unwrap(),
        Box::new(|r| {
            let d = word_of(&digest_from_digits(&random_codeword_digits(r)));
            vec![d, F128::new(r.next() % 64, 0), F128::ZERO]
        }),
    ));
    for (gate, inputs) in &gates {
        assert!(gate.has_native(), "{} has no native evaluator", gate.name());
        for _ in 0..32 {
            let i = inputs(&mut rng);
            assert_eq!(eval_native(gate, &i), eval(gate, &i), "{}", gate.name());
        }
    }
}
