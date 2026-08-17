//! SHA-256 counter-mode DRBG shared across the genus-95 code: an [`RngCore`]
//! whose stream is `SHA256(seed ‖ counter)` for `counter = 0, 1, …` — the same
//! construction the SHA-256 [`crate::challenger::FsChallenger`] uses for its own
//! squeezes. Using it everywhere — the AG-skip `r₁` sampler in production and the
//! genus-95 tests — keeps the only cryptographic assumption SHA-256 (no separate
//! stream-cipher RNG).

use rand_core::RngCore;
use sha2::{Digest, Sha256};

/// SHA-256 counter-mode DRBG.
pub struct Sha256Rng {
    seed: [u8; 32],
    counter: u64,
    buf: [u8; 32],
    pos: usize,
}

impl Sha256Rng {
    /// Seed from 32 bytes (e.g. a transcript squeeze). The seed should be
    /// domain-separated from other squeezes when derived from a transcript.
    pub fn new(seed: [u8; 32]) -> Self {
        Self {
            seed,
            counter: 0,
            buf: [0u8; 32],
            pos: 32,
        }
    }

    /// Seed from a `u64`, for reproducible tests. The value goes in the low 8
    /// seed bytes; the SHA-256 in `refill` supplies the diffusion.
    pub fn seed_from_u64(n: u64) -> Self {
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&n.to_le_bytes());
        Self::new(seed)
    }

    fn refill(&mut self) {
        let mut h = Sha256::new();
        h.update(self.seed);
        h.update(self.counter.to_le_bytes());
        let out = h.finalize();
        self.buf.copy_from_slice(&out);
        self.counter += 1;
        self.pos = 0;
    }
}

impl RngCore for Sha256Rng {
    fn next_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.fill_bytes(&mut b);
        u32::from_le_bytes(b)
    }

    fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.fill_bytes(&mut b);
        u64::from_le_bytes(b)
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut i = 0;
        while i < dst.len() {
            if self.pos == 32 {
                self.refill();
            }
            let n = (32 - self.pos).min(dst.len() - i);
            dst[i..i + n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
            self.pos += n;
            i += n;
        }
    }
}
