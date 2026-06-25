//! Native single-threaded throughput of the hash primitives Flock proves —
//! keccak-f[1600] permutations, SHA-256 compressions, BLAKE3 compressions — to
//! contextualize the prover's per-core overhead.
//!
//! Constraint: NO dedicated hardware crypto-extension instructions (ARMv8-SHA2 /
//! ARMv8.2-SHA3), since a SNARK can't use those — it has to evaluate the bit/word
//! operations.
//!
//!   - keccak-f[1600]:   `keccak` crate — scalar (no SHA3 extension).
//!   - SHA-256 compress: Flock's scalar `sha256_compress` (no SHA2 extension; the
//!                       `sha2` crate's "asm"/intrinsic path WOULD use it).
//!   - BLAKE3 compress:  Flock's scalar `blake3_compress`.
//!
//! Single-threaded (rayon unused). Compare against Flock's single-threaded prover
//! throughput (bench_keccak.sh / bench_sha256.sh, RAYON_NUM_THREADS=1).
//!
//! Run: `cargo bench --bench native_hash`

use std::hint::black_box;
use std::time::Instant;

use flock_prover::r1cs_hashes::blake3::blake3_compress;
use flock_prover::r1cs_hashes::sha2::sha256_compress;

/// Warm up ~0.2 s, then run `f` for ~`secs` and report ops/sec + ns/op.
fn bench<F: FnMut()>(name: &str, secs: f64, mut f: F) {
    let w = Instant::now();
    while w.elapsed().as_secs_f64() < 0.2 {
        f();
    }
    let mut ops: u64 = 0;
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < secs {
        for _ in 0..256 {
            f();
        }
        ops += 256;
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "  {:38} {:>13.0} ops/s   ({:>7.1} ns/op)",
        name,
        ops as f64 / s,
        s * 1e9 / ops as f64
    );
}

fn main() {
    println!(
        "Native single-threaded primitive throughput (scalar software, NO crypto-\n\
         extension instructions). cf. Flock's single-threaded prover.\n"
    );

    // keccak-f[1600] — scalar software permutation (no ARMv8.2-SHA3).
    {
        let mut st = [0u64; 25];
        for (i, w) in st.iter_mut().enumerate() {
            *w = 0x0123_4567_89ab_cdefu64.wrapping_mul(i as u64 + 1);
        }
        bench("keccak-f[1600]   (scalar)", 1.0, || {
            keccak::f1600(&mut st);
            black_box(&st);
        });
    }

    // SHA-256 compression — Flock's scalar software (no ARMv8-SHA2).
    {
        let mut h = [
            0x6a09_e667u32,
            0xbb67_ae85,
            0x3c6e_f372,
            0xa54f_f53a,
            0x510e_527f,
            0x9b05_688c,
            0x1f83_d9ab,
            0x5be0_cd19,
        ];
        let m = [
            0x6162_6380u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0x18,
        ];
        bench("SHA-256 compress (scalar)", 1.0, || {
            h = sha256_compress(&h, &m);
            black_box(&h);
        });
    }

    // BLAKE3 compression — Flock's scalar software.
    {
        let mut cv = [0u32; 8];
        let m = [0x1122_3344u32; 16];
        bench("BLAKE3 compress  (scalar)", 1.0, || {
            let out = blake3_compress(&cv, &m, 0, 64, 0);
            cv.copy_from_slice(&out[..8]);
            black_box(&cv);
        });
    }
}
