//! Dump GF(2^128) GHASH test vectors from the *real* `flock` implementation so
//! the CUDA port (`cuda-ghash/`) can be checked bit-for-bit against it.
//!
//! On this x86_64 host `F128::mul` uses the PCLMULQDQ binius path — the same
//! algorithm the CUDA `ghash_mul_binius` mirrors with `clmad`.
//!
//! Output: little-endian binary to the path in argv[1] (default vectors.bin):
//!   magic  u32 = 0x47483132 ("GH12")
//!   count  u32
//!   count * { a.lo, a.hi, b.lo, b.hi, prod.lo, prod.hi } : u64 each
//!
//! Run:  cargo run --release --bin dump_ghash_vectors -- cuda-ghash/vectors.bin

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use flock_prover::field::F128;

use flock_core::test_rng::Rng;

fn main() -> std::io::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "vectors.bin".to_string());
    let count: u32 = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let mut rng = Rng::new(0xC0FFEE);
    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(&0x4748_3132u32.to_le_bytes())?; // "GH12"
    w.write_all(&count.to_le_bytes())?;

    // Include a few structured edge cases first, then random.
    let edges = [
        (F128::ZERO, F128::ZERO),
        (
            F128::ONE,
            F128 {
                lo: 0xDEAD_BEEF,
                hi: 0x1234,
            },
        ),
        (
            F128::generator(),
            F128 {
                lo: 0,
                hi: 1u64 << 63,
            },
        ), // x · x^127 = 0x87
        (F128 { lo: 0, hi: 1 }, F128 { lo: 0, hi: 1 }), // x^64 · x^64
        (
            F128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
            F128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
        ),
    ];

    for i in 0..count as usize {
        let (a, b) = if i < edges.len() {
            edges[i]
        } else {
            (rng.f128(), rng.f128())
        };
        let p = a * b; // the canonical flock product
        for v in [a.lo, a.hi, b.lo, b.hi, p.lo, p.hi] {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    eprintln!("wrote {count} vectors to {path}");
    Ok(())
}
