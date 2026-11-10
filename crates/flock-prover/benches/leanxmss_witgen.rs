//! Witness generation for one slot of BLAKE3 compressions: the hand-written
//! batch-major generator against the leanXMSS compression gate's packer
//! (the DSL walk with the hand generator as its kernel).
//!
//! `cargo bench -p flock-prover --bench leanxmss_witgen`; honors
//! `RAYON_NUM_THREADS`, `WITGEN_NU` (default 15) and `WITGEN_RUNS`.

use std::{env::var, hint::black_box, time::Instant};

use flock_core::{test_rng::Rng, union::SlotWitnessDest};
use flock_field::F128;
use flock_prover::{
    leanxmss::compress::{compress_gate, ctl, next_step},
    r1cs_hashes::blake3::{Compression, generate_witness_batch_major_partial_into},
};

fn word(w: &[u32]) -> F128 {
    F128::new(
        w[0] as u64 | (w[1] as u64) << 32,
        w[2] as u64 | (w[3] as u64) << 32,
    )
}

fn best_of(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let nu: usize = var("WITGEN_NU").map_or(15, |v| v.parse().unwrap());
    let runs: usize = var("WITGEN_RUNS").map_or(10, |v| v.parse().unwrap());
    let n = 1usize << nu;
    let mut rng = Rng::new(0x5EED);
    let blocks: Vec<Compression> = (0..n)
        .map(|_| {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            (cv, m, 0, 64, 11)
        })
        .collect();
    let rows: Vec<Box<[F128]>> = blocks
        .iter()
        .map(|(cv, m, _, len, flags)| {
            let out = flock_hash::blake3_compress(cv, m, 0, *len, *flags);
            let inputs = [
                word(&cv[..4]),
                word(&cv[4..]),
                word(&m[..4]),
                word(&m[4..8]),
                word(&m[8..12]),
                word(&m[12..]),
                ctl(*flags, *len),
            ];
            let outputs = [word(&out[..4]), word(&out[4..8]), next_step(inputs[3])];
            inputs.into_iter().chain(outputs).collect()
        })
        .collect();
    let gate = compress_gate(nu).unwrap();
    let words = 1usize << (nu + gate.k_log() - 7);
    let (mut z, mut a, mut b) = (
        vec![F128::ZERO; words],
        vec![F128::ZERO; words],
        vec![F128::ZERO; words],
    );
    macro_rules! dst {
        () => {
            SlotWitnessDest {
                z: &mut z,
                a: &mut a,
                b: &mut b,
                elide_padding_writes: true,
                dead_padding_unread: true,
            }
        };
    }
    let threads = rayon::current_num_threads();
    let hand = best_of(runs, || {
        black_box(generate_witness_batch_major_partial_into(
            &blocks,
            nu,
            dst!(),
        ));
    });
    let packer = best_of(runs, || {
        black_box(gate.pack_into(&rows, dst!()));
    });
    let unbatched = best_of(runs, || {
        black_box(gate.pack_into_unbatched(&rows, dst!()));
    });
    println!(
        "2^{nu} compressions, {threads} threads: hand {hand:6.2} ms, gate packer {packer:6.2} ms \
         ({:.2}x), unbatched {unbatched:6.2} ms",
        packer / hand
    );
}
