use flock_core::{circuit::boolean::BooleanGate, union::SlotWitnessDest};
use flock_field::F128;

use super::*;
use crate::leanxmss::reference::{XmssHash, chain_address, le128, th};

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

fn word(v: &[u8; 16]) -> F128 {
    let x = u128::from_le_bytes(*v);
    F128::new(x as u64, (x >> 64) as u64)
}

const ONE_BLOCK: u32 = 1 | 2 | 8;

fn inputs(rng: &mut Rng, flags: u32, block_len: u32) -> Vec<F128> {
    let mut i: Vec<F128> = (0..7).map(|_| rng.word()).collect();
    i[6] = ctl(flags, block_len);
    i
}

/// The row is one word-aligned run: the rounds end where the output begins.
#[test]
fn the_row_fits_a_blake3_block() {
    let gate = compress_gate(3).unwrap();
    eprintln!(
        "compress row: {} useful bits, k_log {}",
        gate.useful_bits(),
        gate.k_log()
    );
    assert_eq!(gate.k_log(), 14);
    assert!(gate.useful_bits() <= 92 * 128);
}

/// The walk, the native evaluation and BLAKE3 agree, for every kind of
/// block the circuit uses.
#[test]
fn native_matches_the_circuit() {
    let gate = compress_gate(3).unwrap();
    let mut rng = Rng(3);
    for (flags, block_len) in [(ONE_BLOCK, 64), (1, 64), (0, 64), (2 | 8, 64), (2 | 8, 16)] {
        for _ in 0..8 {
            let i = inputs(&mut rng, flags, block_len);
            let mut native_out = Vec::new();
            native(&i, &mut native_out);
            assert_eq!(walk(&gate, &i), native_out, "flags {flags:#x}");
        }
    }
}

/// A chain step row computes `Th(P, A(µ), v ‖ LE128(c))` in `o0` and
/// `A(µ + 1)` in `next`.
#[test]
fn chain_step_row() {
    let gate = compress_gate(3).unwrap();
    let iv = |h: usize| {
        let w = crate::r1cs_hashes::blake3::BLAKE3_IV;
        F128::new(
            w[4 * h] as u64 | (w[4 * h + 1] as u64) << 32,
            w[4 * h + 2] as u64 | (w[4 * h + 3] as u64) << 32,
        )
    };
    let pp = [7u8; 16];
    let v = [0x5Au8; 16];
    for (layer, e, c) in [(0usize, 0x1234_5678u32, 0usize), (1, 0xFFFF, 57)] {
        for step in 0..3 {
            let a = chain_address(layer, e, step);
            let i = [
                iv(0),
                iv(1),
                word(&pp),
                word(&a),
                word(&v),
                word(&le128(c)),
                ctl(ONE_BLOCK, 64),
            ];
            let out = walk(&gate, &i);
            let want = th(XmssHash::Blake3, &pp, &a, &[&v, &le128(c)]);
            assert_eq!(out[0], word(&want));
            assert_eq!(out[2], word(&chain_address(layer, e, step + 1)));
        }
    }
}

/// Packing with the BLAKE3 kernel writes exactly the slot the walk writes:
/// `z`, `A·z`, `B·z` and the lincheck stripe, batched or not.
#[test]
fn kernel_packing_matches_the_walk() {
    let nu = 7;
    let (circuit, _) = compress_circuit();
    let walked = BooleanGate::new("walked", circuit, &INPUTS, &OUTPUTS, nu).unwrap();
    let kernel = compress_gate(nu).unwrap();
    let mut rng = Rng(13);
    let all: Vec<Box<[F128]>> = (0..1 << nu)
        .map(|r| {
            inputs(
                &mut rng,
                [ONE_BLOCK, 1, 0, 2 | 8][r % 4],
                [64, 64, 64, 16][r % 4],
            )
            .into()
        })
        .collect();
    let len = 1usize << (nu + walked.k_log() - 7);
    for n in [1 << nu, 77, 64, 61, 8, 3] {
        let pack = |gate: &BooleanGate, batched: bool| {
            let poison = F128::new(0xDEAD, 0xBEEF);
            let (mut z, mut a, mut b) = (vec![poison; len], vec![poison; len], vec![poison; len]);
            let dst = SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: false,
                dead_padding_unread: false,
            };
            let stripe = if batched {
                gate.pack_into(&all[..n], dst)
            } else {
                gate.pack_into_unbatched(&all[..n], dst)
            };
            (z, a, b, stripe)
        };
        let want = pack(&walked, false);
        for (what, got) in [
            ("walk, batched", pack(&walked, true)),
            ("kernel", pack(&kernel, false)),
            ("kernel, batched", pack(&kernel, true)),
        ] {
            assert!(want.0 == got.0, "{what}, {n} rows: z differs");
            assert!(want.1 == got.1, "{what}, {n} rows: A·z differs");
            assert!(want.2 == got.2, "{what}, {n} rows: B·z differs");
            assert!(want.3 == got.3, "{what}, {n} rows: stripe differs");
        }
    }
}
