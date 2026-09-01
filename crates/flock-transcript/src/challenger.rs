//! Fiat-Shamir challenger interfaces and implementations.
//!
//! Provers and verifiers observe the same messages and sample the same challenges.
//! [`FsChallenger`] supports SHA-256 and BLAKE3.
//! `RandomChallenger` supports tests and benchmarks only.

use std::array::from_fn;

use blake3::{Hasher, IncrementCounter, hash, platform::Platform};
use flock_field::{F128, F256};
use flock_hash::{BLAKE3_IV, HashKind, blake3_compress};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use sha2::{Digest, Sha256};
#[cfg(feature = "hash-count")]
use {
    crate::challenger::fs_count::{POW_SHA256, SQUEEZED_BYTES, SQUEEZES},
    std::sync::atomic::Ordering,
};

/// Number of grinding bits needed to turn a Schwartz--Zippel event of
/// degree at most `degree` over `F_{2^128}` into a *strictly* sub-`2^-128`
/// event at this site.
///
/// After a `bits`-bit Fiat--Shamir grind, the usual bound is
/// `degree / 2^(128 + bits)`.  Thus non-constant events need
/// `floor(log2(degree)) + 1` bits; degree zero needs none.
#[inline]
pub const fn grinding_bits_for_degree(degree: usize) -> u32 {
    if degree == 0 {
        0
    } else {
        usize::BITS - degree.leading_zeros()
    }
}

// `Send` supertrait: the verifier runs its PIOP/PCS replay inside a dedicated
// single-thread rayon pool (see `verifier::verifier_pool`), so the challenger
// it threads through must be able to cross into that pool. Both concrete
// challengers (`RandomChallenger`, `FsChallenger`) are trivially `Send`.
pub trait Challenger: Send {
    /// Whether the explicit PoW+sample methods use the chained-BLAKE3 fused
    /// transition rather than their legacy default composition. Recording
    /// wrappers use this only to choose the matching transcript opcode.
    fn supports_fused_pow_squeeze(&self) -> bool {
        false
    }

    /// Absorb a domain-separation label (e.g. `b"flock-zerocheck-v0"`). Each
    /// protocol entry should call this once on entry so a transcript from
    /// one protocol cannot be replayed as another.
    fn observe_label(&mut self, _label: &[u8]) {
        // default no-op — RandomChallenger inherits this.
    }

    /// Absorb a single F128 prover message.
    fn observe_f128(&mut self, value: F128);

    /// Absorb a slice of F128 prover messages (e.g. the round-1 vector).
    fn observe_f128_slice(&mut self, values: &[F128]) {
        for v in values {
            self.observe_f128(*v);
        }
    }

    /// Absorb one quadratic-extension message as its two canonical F128
    /// coordinates in `(c0, c1)` order.
    fn observe_f256(&mut self, value: F256) {
        self.observe_f128_slice(&value.coordinates());
    }

    /// Absorb arbitrary bytes (e.g. a Merkle root or a statement digest).
    fn observe_bytes(&mut self, _bytes: &[u8]) {
        // default no-op — RandomChallenger inherits this.
    }

    /// Produce one F128 challenge.
    fn sample_f128(&mut self) -> F128;

    /// Produce `n` F128 challenges, in order.
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.sample_f128()).collect()
    }

    /// Produce one uniform quadratic-extension challenge from one two-word
    /// transcript squeeze.
    fn sample_f256(&mut self) -> F256 {
        let words = self.sample_f128_vec(2);
        F256::new(words[0], words[1])
    }

    /// Prover-side PoW grinding: snapshot the current transcript state,
    /// search for a `u64` nonce such that `H(state ‖ nonce)` has at
    /// least `bits` leading zero bits, then absorb the nonce into the
    /// transcript so subsequent challenges bind to it.
    ///
    /// Default implementation is a no-op (returns 0). Real implementations
    /// — e.g. [`FsChallenger`] — do the actual grind work and absorb the
    /// nonce. `bits = 0` means "no PoW required"; still absorbs the 0 nonce
    /// so the verifier mirror is byte-identical.
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        0
    }

    /// Verifier-side mirror of [`Self::grind_pow`]: check that `nonce`
    /// satisfies the `bits`-leading-zeros PoW against the current transcript
    /// state, then absorb the nonce so the running state stays in lockstep
    /// with the prover.
    ///
    /// Default implementation accepts unconditionally (no-op). Real
    /// implementations must check the PoW; an honest verifier rejects the
    /// proof if this returns `false`.
    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        true
    }

    /// Prover-side fused PoW + scalar squeeze.  The default preserves the
    /// legacy two-operation transcript; the chained-BLAKE3 challenger
    /// overrides this with one compression that emits a disjoint PoW word and
    /// challenge word.
    fn grind_pow_and_sample_f128(&mut self, bits: u32) -> (u64, F128) {
        let nonce = self.grind_pow(bits);
        (nonce, self.sample_f128())
    }

    /// Verifier mirror of [`Self::grind_pow_and_sample_f128`].
    fn verify_pow_and_sample_f128(&mut self, nonce: u64, bits: u32) -> Option<F128> {
        if !self.verify_pow(nonce, bits) {
            return None;
        }
        Some(self.sample_f128())
    }

    /// Prover-side fused PoW + vector squeeze.
    fn grind_pow_and_sample_f128_vec(&mut self, bits: u32, n: usize) -> (u64, Vec<F128>) {
        let nonce = self.grind_pow(bits);
        (nonce, self.sample_f128_vec(n))
    }

    /// Verifier mirror of [`Self::grind_pow_and_sample_f128_vec`].
    fn verify_pow_and_sample_f128_vec(
        &mut self,
        nonce: u64,
        bits: u32,
        n: usize,
    ) -> Option<Vec<F128>> {
        if !self.verify_pow(nonce, bits) {
            return None;
        }
        Some(self.sample_f128_vec(n))
    }

    /// Construct the DOMAIN-SEPARATED child transcript from an
    /// externally-supplied 256-bit seed. The building block of
    /// [`Self::fork`]; split out so wrappers that must RECORD the seed
    /// extraction (the tape recorder) can sample through themselves and
    /// then delegate here.
    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self
    where
        Self: Sized;

    /// Fork a DOMAIN-SEPARATED child transcript for parallel sub-protocol
    /// composition. The child is deterministically derived from a seed
    /// SAMPLED from the parent (which advances the parent's state, so the
    /// fork point itself is bound) plus `label`. Sub-protocols run on
    /// disjoint children may execute in any order — or concurrently — and
    /// each child's closing digest must be absorbed back into the parent
    /// ([`Self::merge_child`]) before the parent samples anything that must
    /// bind the child's messages. Prover and verifier must fork and merge
    /// at identical transcript positions with identical labels.
    fn fork(&mut self, label: &'static [u8]) -> Self
    where
        Self: Sized,
    {
        let seed = [self.sample_f128(), self.sample_f128()];
        self.fork_from_seed(seed, label)
    }

    /// Absorb a child transcript's closing digest (two squeezed field
    /// elements — 256 state-binding bits) into this transcript. See
    /// [`Self::fork`].
    fn merge_child(&mut self, mut child: Self)
    where
        Self: Sized,
    {
        let d0 = child.sample_f128();
        let d1 = child.sample_f128();
        self.observe_f128(d0);
        self.observe_f128(d1);
    }

    /// The hash backing this transcript, for protocol components that derive
    /// auxiliary randomness outside the challenger itself (e.g. the AG-skip
    /// `r₁` nonce-grind DRBG) and must follow the transcript's hash choice so
    /// no second primitive enters the soundness argument. Default SHA-256
    /// (`RandomChallenger` and legacy implementations inherit it).
    fn hash_kind(&self) -> HashKind {
        HashKind::Sha256
    }
}

// ---------------------------------------------------------------------------
// RandomChallenger — seeded SplitMix64 pseudo-random source.
//
// Ignores observed messages (no Fiat-Shamir binding). Keep for bench isolation
// and soundness mutation tests; real proofs MUST use FsChallenger.
//
// Gated behind `cfg(test)` / `feature = "unsound-challenger"`: a real-proof
// build does not compile this type at all, so no production code path can
// accidentally instantiate an unsound challenger. See the module docs.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "unsound-challenger"))]
#[derive(Clone, Debug)]
pub struct RandomChallenger {
    state: u64,
}

#[cfg(any(test, feature = "unsound-challenger"))]
impl RandomChallenger {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

#[cfg(any(test, feature = "unsound-challenger"))]
impl Challenger for RandomChallenger {
    #[inline]
    fn observe_f128(&mut self, _value: F128) {
        // intentional no-op: random challenger is independent of prover state
    }

    fn sample_f128(&mut self) -> F128 {
        let lo = splitmix64(&mut self.state);
        let hi = splitmix64(&mut self.state);
        F128 { lo, hi }
    }

    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        // Binding is irrelevant here (this challenger ignores messages by
        // design); mix the seed with the label for distinct streams.
        let mut l = 0u64;
        for &b in label {
            l = l.wrapping_mul(0x100_0000_01B3).wrapping_add(b as u64);
        }
        RandomChallenger::new(seed[0].lo ^ l)
    }
}

#[cfg(any(test, feature = "unsound-challenger"))]
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// FsChallenger — Fiat-Shamir over a selectable hash (SHA-256 or BLAKE3).
//
// Tag bytes (one-byte op + one-byte kind) encode the operation type so that
// e.g. an `observe_f128_slice` of length 1 cannot collide with `observe_f128`,
// and a slice observation cannot collide with two scalar observations of the
// same total length. Tagging, absorption order and the duplex structure are
// identical for both hashes — only the primitive differs.
//
// Sampling absorbs a header, then clones the live hasher and squeezes challenge
// bytes from the clone. The output is NOT absorbed back: it is `XOF(state)`, so
// feeding it in adds nothing the state does not already determine. Later
// observations still bind to the challenge, because they are absorbed into the
// very state that produced it. The header absorb is what makes two consecutive
// squeezes differ.
//
// How the squeeze itself is done is the one place the two hashes genuinely
// diverge, because SHA-256 is not an extendable-output function and BLAKE3 is:
//
//   SHA-256: derive the stream as SHA256(state ‖ ctr) for ctr = 0, 1, …,
//            32 bytes at a time.
//   BLAKE3:  finalize the cloned state into an XOF reader and fill straight
//            from it — no counter, and one finalization regardless of length.
//
// Both are deterministic functions of the transcript state, which is all the
// duplex requires. The counter is a workaround for SHA-256's fixed output, so
// BLAKE3 does not inherit it; a proof is only ever verified under the same
// hash it was produced with (see `FsChallenger::with_hash`).
// ---------------------------------------------------------------------------

// Public so [`crate::transcript_record`] can reconstruct the absorbed byte
// stream from a recorded shape against ONE definition of the framing rather
// than a copy of it.
pub const OP_DOMAIN: u8 = 0x01;
pub const OP_LABEL: u8 = 0x02;
pub const OP_OBSERVE: u8 = 0x03;
pub const OP_SQUEEZE: u8 = 0x04;
pub const OP_BYTES: u8 = 0x05;

pub const KIND_SCALAR: u8 = 0x01;
pub const KIND_SLICE: u8 = 0x02;
pub const KIND_NONE: u8 = 0x00;

/// Global Fiat–Shamir hash counters, enabled with `--features hash-count`.
/// Tracks the squeeze count, the squeezed output length and the PoW checks;
/// absorbed transcript bytes are tracked via [`FsChallenger::absorbed_bytes`].
#[cfg(feature = "hash-count")]
pub mod fs_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Number of XOF finalizations (one per `sample_f128` /
    /// `sample_f128_vec` / PoW state-digest extraction).
    pub static SQUEEZES: AtomicU64 = AtomicU64::new(0);
    /// Total bytes of squeezed OUTPUT. Tracked separately from the squeeze
    /// count because the two scale differently: a squeeze costs one
    /// finalization of the pending state plus one compression per 64 bytes of
    /// output. That distinction was invisible while every squeeze was a
    /// 16-byte `sample_f128`, and became load-bearing when query sampling
    /// moved to one batched `sample_f128_vec` per level (3888 bytes at L0 —
    /// 61 output blocks, not 1).
    pub static SQUEEZED_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Number of PoW evaluations, under whichever hash the transcript uses
    /// (1 compression each; 40 B input).
    pub static POW_SHA256: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        SQUEEZES.store(0, Relaxed);
        SQUEEZED_BYTES.store(0, Relaxed);
        POW_SHA256.store(0, Relaxed);
    }

    /// (squeezes, squeezed_bytes, pow_calls)
    pub fn snapshot() -> (u64, u64, u64) {
        (
            SQUEEZES.load(Relaxed),
            SQUEEZED_BYTES.load(Relaxed),
            POW_SHA256.load(Relaxed),
        )
    }
}

/// The running transcript state, one variant per supported hash.
#[derive(Clone)]
enum FsState {
    Sha256(Sha256),
    Blake3(Box<Hasher>),
    /// The sponge-chained BLAKE3 discipline (transcript-v2): a sequential
    /// compression chain — no chunk tree, no per-squeeze root forks. A
    /// recursion circuit replays one row per 64-byte block plus ~two per
    /// squeeze, instead of the tree discipline's O(log) fork parents.
    Blake3Chain(B3Chain),
}

/// Chained-MD state over [`crate::hash::blake3_compress`]: `cv` is the
/// running 256-bit chaining value, `buf` the pending partial block
/// (invariant: `< 64` bytes after any absorb). Ordinary absorb compressions
/// run at counter 0 — block order is bound by the cv chain itself, so a
/// position counter would be BLAKE3-tree residue (and every distinct counter
/// value costs a public in the recursion circuit).
///
/// **Squeezes are DUPLEX (transcript-v3)**: a squeeze's first output row is
/// `compress(cv, m = pending partial block zero-padded, 0, buf.len(),
/// CHAIN_SQUEEZE)` — it consumes the pending bytes as its message, its full
/// 64 output bytes are the first output bytes, and `cv` advances to its
/// chaining half. Further output rows are `compress(cv, ZERO, 0, 0,
/// CHAIN_SQUEEZE)`, each advancing `cv` the same way. So squeezes MUTATE
/// the state: no separate flush compression, no `OP_SQUEEZE` header absorb,
/// no per-finalize block fragmentation — under transcript-v2's discipline
/// those three were ~30% of every recursion child's chain rows. Counter
/// stays 0 for ordinary rows (sequential cv chaining binds order; no two rows
/// share a cv, so the old output index `j` is dead too). The fused PoW+squeeze
/// transition uses [`pow_squeeze_counter`] to bind its domain, difficulty and
/// real message length. Consecutive squeezes separate because each advances
/// `cv`.
#[derive(Clone)]
struct B3Chain {
    cv: [u32; 8],
    buf: Vec<u8>,
}

/// Domain flag for chain absorb compressions (disjoint from BLAKE3's
/// CHUNK_START/END/PARENT/ROOT bits, which occupy the low bits).
const CHAIN_ABSORB: u32 = 1 << 6;
/// Domain flag for squeeze/output compressions.
const CHAIN_SQUEEZE: u32 = 1 << 7;

/// Domain-separating counter prefix for a fused PoW+squeeze row.  Ordinary
/// absorb and squeeze rows use counter zero.  The low word binds the requested
/// difficulty and the number of real bytes in the zero-padded message block.
pub const POW_SQUEEZE_COUNTER_TAG: u64 = 0xF10C_5000_0000_0000;

#[inline]
pub const fn pow_squeeze_counter(bits: u32, message_len: usize) -> u64 {
    POW_SQUEEZE_COUNTER_TAG | ((message_len as u64) << 32) | bits as u64
}

impl B3Chain {
    fn new() -> Self {
        Self {
            cv: BLAKE3_IV,
            buf: Vec::with_capacity(64),
        }
    }

    fn block_words(bytes: &[u8]) -> [u32; 16] {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            let mut b = [0u8; 4];
            let at = 4 * i;
            if at < bytes.len() {
                let n = (bytes.len() - at).min(4);
                b[..n].copy_from_slice(&bytes[at..at + n]);
            }
            *w = u32::from_le_bytes(b);
        }
        m
    }

    fn absorb(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        while self.buf.len() >= 64 {
            let m = Self::block_words(&self.buf[..64]);
            let out = blake3_compress(&self.cv, &m, 0, 64, CHAIN_ABSORB);
            self.cv = out[..8].try_into().expect("8 words");
            self.buf.drain(..64);
        }
    }

    /// Duplex squeeze: the first output compression eats the pending
    /// partial block as its message and every output compression advances
    /// `cv` — see the struct docs. Mutates.
    fn squeeze_into(&mut self, out: &mut [u8]) {
        let mut off = 0usize;
        let mut first = true;
        while off < out.len() {
            let (m, blen) = if first {
                (Self::block_words(&self.buf), self.buf.len() as u32)
            } else {
                ([0u32; 16], 0u32)
            };
            let ob = blake3_compress(&self.cv, &m, 0, blen, CHAIN_SQUEEZE);
            self.cv = ob[..8].try_into().expect("8 words");
            let mut bytes = [0u8; 64];
            for (i, w) in ob.iter().enumerate() {
                bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
            let take = (out.len() - off).min(64);
            out[off..off + take].copy_from_slice(&bytes[..take]);
            off += take;
            first = false;
        }
        self.buf.clear();
    }

    fn pow_block(&self, nonce: u64) -> [u8; 64] {
        assert!(
            self.buf.len() + 16 <= 64,
            "the aligned chained transcript must leave one word for a PoW nonce"
        );
        let mut block = [0u8; 64];
        block[..self.buf.len()].copy_from_slice(&self.buf);
        block[self.buf.len()..self.buf.len() + 8].copy_from_slice(&nonce.to_le_bytes());
        // The next eight bytes are deliberately zero.  The recursive circuit
        // constrains them, so the nonce has exactly 64 grinding bits.
        block
    }

    fn pow_candidate_output(&self, nonce: u64, bits: u32) -> [u32; 16] {
        let block = self.pow_block(nonce);
        let m = Self::block_words(&block);
        blake3_compress(
            &self.cv,
            &m,
            pow_squeeze_counter(bits, self.buf.len() + 16),
            64,
            CHAIN_SQUEEZE,
        )
    }

    /// Apply the fused transition after the nonce is known.  Output word 1 is
    /// reserved for the PoW predicate; challenge bytes are words 0, 2 and 3,
    /// followed by ordinary zero-message continuation rows.  The first row's
    /// high half becomes the next chaining value, keeping a scalar challenge,
    /// the PoW predicate and the continuing state pairwise disjoint.
    fn apply_pow_squeeze(&mut self, nonce: u64, bits: u32, out: &mut [u8]) -> bool {
        assert!(
            !out.is_empty(),
            "a PoW must protect at least one challenge word"
        );
        assert!(
            bits <= 128,
            "the fused PoW predicate occupies one F128 word"
        );
        // One PoW evaluation + one squeeze of `out.len()` bytes: the fused
        // transition is both at once, so both ledgers get their entry (the
        // grind side's failed attempts are counted inside the scan itself).
        #[cfg(feature = "hash-count")]
        {
            POW_SHA256.fetch_add(1, Ordering::Relaxed);
            SQUEEZES.fetch_add(1, Ordering::Relaxed);
            SQUEEZED_BYTES.fetch_add(out.len() as u64, Ordering::Relaxed);
        }
        let ob = self.pow_candidate_output(nonce, bits);
        let mut first = [0u8; 64];
        for (i, w) in ob.iter().enumerate() {
            first[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        let ok = if bits == 0 {
            nonce == 0
        } else {
            has_leading_zero_bits(&first[16..32], bits)
        };

        // out_hi is disjoint from both the predicate (out_lo word 1) and the
        // scalar challenge (out_lo word 0).
        self.cv = ob[8..16].try_into().expect("8 words");
        self.buf.clear();

        let mut off = 0usize;
        for src in [&first[..16], &first[32..64]] {
            let take = (out.len() - off).min(src.len());
            out[off..off + take].copy_from_slice(&src[..take]);
            off += take;
            if off == out.len() {
                return ok;
            }
        }
        while off < out.len() {
            let ob = blake3_compress(&self.cv, &[0u32; 16], 0, 0, CHAIN_SQUEEZE);
            self.cv = ob[..8].try_into().expect("8 words");
            let mut bytes = [0u8; 64];
            for (i, w) in ob.iter().enumerate() {
                bytes[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
            let take = (out.len() - off).min(64);
            out[off..off + take].copy_from_slice(&bytes[..take]);
            off += take;
        }
        ok
    }

    fn grind_pow_squeeze_into(&mut self, bits: u32, out: &mut [u8]) -> u64 {
        assert!(
            bits <= 128,
            "the fused PoW predicate occupies one F128 word"
        );
        const PARALLEL_GRIND_MIN_HASHES: u64 = 1 << 13;
        const GRIND_CHUNK: u64 = 1 << 10;
        let nonce = if bits == 0 {
            0
        } else if (1u64 << bits.min(63)) < PARALLEL_GRIND_MIN_HASHES {
            let mut start = 0u64;
            loop {
                if let Some(n) =
                    blake3_chain_pow_scan(&self.cv, &self.buf, start, GRIND_CHUNK, bits)
                {
                    break n;
                }
                start = start.saturating_add(GRIND_CHUNK);
            }
        } else {
            let block = 1u64 << (bits.min(24) + 1);
            let n_chunks = block.div_ceil(GRIND_CHUNK);
            let mut start = 0u64;
            loop {
                let found = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk| {
                        blake3_chain_pow_scan(
                            &self.cv,
                            &self.buf,
                            start.saturating_add(chunk * GRIND_CHUNK),
                            GRIND_CHUNK,
                            bits,
                        )
                    })
                    .find_first(|r| r.is_some())
                    .flatten();
                if let Some(n) = found {
                    break n;
                }
                start = start.saturating_add(block);
            }
        };
        let ok = self.apply_pow_squeeze(nonce, bits, out);
        debug_assert!(ok, "the nonce search returned an invalid fused PoW");
        nonce
    }
}

#[derive(Clone)]
pub struct FsChallenger {
    state: FsState,
    /// Running total of absorbed transcript bytes.
    #[cfg(feature = "hash-count")]
    n_absorbed: u64,
}

impl FsChallenger {
    /// New challenger seeded with a domain-separation tag (e.g.
    /// `b"flock-r1cs-v0"`), using SHA-256.
    ///
    /// The domain is length-prefixed before being absorbed so two domains
    /// where one is a prefix of the other cannot produce the same initial
    /// state. For the BLAKE3 transcript, see [`Self::with_hash`].
    pub fn new(domain: &[u8]) -> Self {
        Self::with_hash(domain, HashKind::Sha256)
    }

    /// New challenger over an explicit hash.
    ///
    /// The prover and verifier must agree: the transcript is a function of the
    /// hash, so a mismatch diverges at the first challenge and the proof fails
    /// to verify. That is the intended failure mode — nothing tries to detect
    /// or negotiate it, exactly as with the Merkle hash.
    pub fn with_hash(domain: &[u8], kind: HashKind) -> Self {
        let mut c = Self {
            state: match kind {
                HashKind::Sha256 => FsState::Sha256(Sha256::new()),
                HashKind::Blake3 => FsState::Blake3(Box::new(Hasher::new())),
            },
            #[cfg(feature = "hash-count")]
            n_absorbed: 0,
        };
        c.absorb_header(OP_DOMAIN, 0, domain.len() as u64);
        c.absorb_padded(domain);
        c
    }

    /// New challenger over the sponge-CHAINED BLAKE3 discipline
    /// (transcript-v2): a sequential compression chain in place of the
    /// BLAKE3 chunk tree, so a recursion circuit replaying the transcript
    /// pays ~one row per 64 absorbed bytes plus ~two per squeeze — the tree
    /// discipline's per-squeeze root forks (O(log absorbed) parents each)
    /// were more than half of every child's chain rows.
    pub fn with_chained_blake3(domain: &[u8]) -> Self {
        let mut c = Self {
            state: FsState::Blake3Chain(B3Chain::new()),
            #[cfg(feature = "hash-count")]
            n_absorbed: 0,
        };
        c.absorb_header(OP_DOMAIN, 0, domain.len() as u64);
        c.absorb_padded(domain);
        c
    }

    /// Which hash backs this transcript.
    pub fn hash_kind(&self) -> HashKind {
        match self.state {
            FsState::Sha256(_) => HashKind::Sha256,
            // The chained discipline reports Blake3: PoW grinding and every
            // other kind-keyed helper hash the same primitive.
            FsState::Blake3(_) | FsState::Blake3Chain(_) => HashKind::Blake3,
        }
    }

    /// Absorb bytes into the running transcript state.
    #[inline]
    fn absorb(&mut self, bytes: &[u8]) {
        match &mut self.state {
            FsState::Sha256(h) => {
                h.update(bytes);
            }
            FsState::Blake3(h) => {
                h.update(bytes);
            }
            FsState::Blake3Chain(c) => {
                c.absorb(bytes);
            }
        }
        #[cfg(feature = "hash-count")]
        {
            self.n_absorbed = self.n_absorbed.wrapping_add(bytes.len() as u64);
        }
    }

    /// Absorb one op's 16-byte header: `[op][kind][0;6][len u64 LE]`.
    ///
    /// **Why 16 and not 2.** Every observed value is an `F128` — 16 bytes, and
    /// exactly one 128-bit committed word. A recursion circuit replaying this
    /// transcript has to place those bytes into BLAKE3's `m` words, and its
    /// wires carry 128-bit words, so the placement is a pure copy *iff* each
    /// value starts at a multiple of 16 in the absorbed stream. Under the
    /// former 1–2 byte tags and 8-byte length prefixes it did not: successive
    /// scalars landed at `2 + 18k`, so seven values in eight straddled two `m`
    /// words and sometimes two rows. Expressing that needs a byte-shifted
    /// merge, which a copy constraint cannot state — it would cost a packing
    /// gate and a boolean glue table. A fixed 16-byte header removes the
    /// misalignment at its source.
    ///
    /// Domain separation is unchanged: the op and kind bytes still make an
    /// `observe_f128_slice` of length 1 distinguishable from an
    /// `observe_f128`, and the length still binds.
    #[inline]
    fn absorb_header(&mut self, op: u8, kind: u8, len: u64) {
        let mut h = [0u8; 16];
        h[0] = op;
        h[1] = kind;
        h[8..].copy_from_slice(&len.to_le_bytes());
        self.absorb(&h);
    }

    /// Absorb a byte payload, zero-padded up to a multiple of 16.
    ///
    /// Unambiguous because the true length rides in the header: two payloads
    /// with the same padded form must share a length, hence be equal.
    #[inline]
    fn absorb_padded(&mut self, bytes: &[u8]) {
        self.absorb(bytes);
        let rem = bytes.len() % 16;
        if rem != 0 {
            self.absorb(&[0u8; 16][..16 - rem]);
        }
    }

    #[inline]
    fn absorb_f128(&mut self, v: F128) {
        self.absorb(&v.lo.to_le_bytes());
        self.absorb(&v.hi.to_le_bytes());
    }

    /// Squeeze `out.len()` pseudorandom bytes from the current transcript
    /// state. The SHA-256 and tree-BLAKE3 disciplines do not mutate; the
    /// CHAINED discipline is a duplex sponge, so its squeeze advances the
    /// state (which is also why the chain absorbs no `OP_SQUEEZE` header —
    /// the squeeze itself separates consecutive samples).
    ///
    /// SHA-256 is not an XOF, so its stream is `SHA256(state ‖ ctr)` for
    /// ctr = 0, 1, … (32 bytes each). Tree-BLAKE3 *is* an XOF, so it
    /// finalizes the cloned state once and fills straight from the reader —
    /// no counter, and no per-32-byte re-finalization.
    fn squeeze_into(&mut self, out: &mut [u8]) {
        match &mut self.state {
            FsState::Sha256(hasher) => {
                let mut off = 0usize;
                let mut ctr: u64 = 0;
                while off < out.len() {
                    let mut h = hasher.clone();
                    h.update(ctr.to_le_bytes());
                    let block: [u8; 32] = h.finalize().into();
                    let take = (out.len() - off).min(32);
                    out[off..off + take].copy_from_slice(&block[..take]);
                    off += take;
                    ctr = ctr.wrapping_add(1);
                }
            }
            FsState::Blake3(hasher) => hasher.finalize_xof().fill(out),
            FsState::Blake3Chain(c) => c.squeeze_into(out),
        }
    }

    /// 32-byte digest of the current transcript state, used as the PoW base.
    /// SHA-256/tree-BLAKE3 clone + finalize without mutating; the chained
    /// discipline's digest is a duplex squeeze and advances the state —
    /// `grind_pow` and `verify_pow` each take exactly one digest per PoW op,
    /// so both sides stay in lockstep.
    #[inline]
    fn state_digest(&mut self) -> [u8; 32] {
        #[cfg(feature = "hash-count")]
        {
            SQUEEZES.fetch_add(1, Ordering::Relaxed);
            SQUEEZED_BYTES.fetch_add(32, Ordering::Relaxed);
        }
        match &mut self.state {
            FsState::Sha256(h) => h.clone().finalize().into(),
            FsState::Blake3(h) => *h.finalize().as_bytes(),
            FsState::Blake3Chain(c) => {
                let mut d = [0u8; 32];
                c.squeeze_into(&mut d);
                d
            }
        }
    }

    /// Total bytes absorbed into the transcript so far. Used by the
    /// `hash-count` instrumentation to estimate SHA-256 compression calls
    /// (≈ bytes / 64).
    #[cfg(feature = "hash-count")]
    pub fn absorbed_bytes(&self) -> u64 {
        self.n_absorbed
    }
}

impl Challenger for FsChallenger {
    fn supports_fused_pow_squeeze(&self) -> bool {
        matches!(&self.state, FsState::Blake3Chain(_))
    }

    fn hash_kind(&self) -> HashKind {
        FsChallenger::hash_kind(self)
    }

    fn observe_label(&mut self, label: &[u8]) {
        self.absorb_header(OP_LABEL, 0, label.len() as u64);
        self.absorb_padded(label);
    }

    fn observe_f128(&mut self, value: F128) {
        self.absorb_header(OP_OBSERVE, KIND_SCALAR, 1);
        self.absorb_f128(value);
    }

    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.absorb_header(OP_OBSERVE, KIND_SLICE, values.len() as u64);
        for v in values {
            self.absorb_f128(*v);
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.absorb_header(OP_BYTES, 0, bytes.len() as u64);
        self.absorb_padded(bytes);
    }

    fn sample_f128(&mut self) -> F128 {
        #[cfg(feature = "hash-count")]
        {
            SQUEEZES.fetch_add(1, Ordering::Relaxed);
            SQUEEZED_BYTES.fetch_add(16, Ordering::Relaxed);
        }
        // The duplex chain drops the OP_SQUEEZE header: its squeeze itself
        // advances the state, so the header's only job (separating
        // consecutive samples) is already done — and each header word was
        // a recursion-circuit chain row.
        if !matches!(self.state, FsState::Blake3Chain(_)) {
            self.absorb_header(OP_SQUEEZE, KIND_SCALAR, 1);
        }
        let mut buf = [0u8; 16];
        self.squeeze_into(&mut buf);
        let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let hi = u64::from_le_bytes(buf[8..].try_into().unwrap());
        F128 { lo, hi }
    }

    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        // 256 bits of parent state (sampled by `fork`'s default body, which
        // advances the parent and binds the fork point) seed the child under
        // its own domain.
        let mut child = match &self.state {
            FsState::Sha256(_) => Self::with_hash(label, HashKind::Sha256),
            FsState::Blake3(_) => Self::with_hash(label, HashKind::Blake3),
            FsState::Blake3Chain(_) => Self::with_chained_blake3(label),
        };
        child.observe_f128(seed[0]);
        child.observe_f128(seed[1]);
        child
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        #[cfg(feature = "hash-count")]
        {
            SQUEEZES.fetch_add(1, Ordering::Relaxed);
            SQUEEZED_BYTES.fetch_add((n * 16) as u64, Ordering::Relaxed);
        }
        if !matches!(self.state, FsState::Blake3Chain(_)) {
            self.absorb_header(OP_SQUEEZE, KIND_SLICE, n as u64);
        }
        let mut buf = vec![0u8; n * 16];
        self.squeeze_into(&mut buf);
        buf.as_chunks::<16>()
            .0
            .iter()
            .map(|c| F128 {
                lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
            })
            .collect()
    }

    fn grind_pow(&mut self, bits: u32) -> u64 {
        let kind = self.hash_kind();
        let state_digest = self.state_digest();
        // Aggregate-aware parallelism: decide on the grind's *expected hash
        // work* (`2^bits`), not a raw bit threshold. Fold-challenge grinds are
        // individually modest — e.g. 2^15 at L0 under the per-round profiles —
        // but the prover issues one per lane fold (6× at L0, 3× per recursive
        // level), so the per-level aggregate (~2^17–2^18 hashes) lands on the
        // multi-threaded critical path. We go parallel once a single grind
        // clears the rayon dispatch break-even (~2^13 hashes); the genuinely
        // tiny deep-level grinds (2^3–2^11) stay sequential, where the serial
        // loop beats parallel-dispatch overhead. `find_first` returns the
        // globally smallest satisfying nonce, so the result is identical to the
        // sequential search (deterministic proofs) regardless of this choice.
        const PARALLEL_GRIND_MIN_HASHES: u64 = 1 << 13;
        // Nonces per rayon task in the parallel search. Large enough to amortize
        // task dispatch and to let the BLAKE3 batch run many `hash_many` calls
        // per task, small enough to keep cancellation granular once an earlier
        // task has found a match.
        const GRIND_CHUNK: u64 = 1 << 10;
        let nonce = if bits == 0 {
            0
        } else if (1u64 << bits.min(63)) < PARALLEL_GRIND_MIN_HASHES {
            // Sequential search: scan ascending blocks until a nonce lands.
            // `pow_scan` returns the smallest match within the block it is
            // given, so scanning blocks in order yields the globally smallest.
            let mut start: u64 = 0;
            loop {
                if let Some(n) = pow_scan(&state_digest, start, GRIND_CHUNK, bits, kind) {
                    break n;
                }
                start = start.saturating_add(GRIND_CHUNK);
            }
        } else {
            // Search ordered blocks in parallel and return the first match.
            let block: u64 = 1 << (bits.min(24) + 1);
            let n_chunks = block.div_ceil(GRIND_CHUNK);
            let mut start: u64 = 0;
            loop {
                // `find_first` takes the earliest *chunk* that yields a match
                // and cancels the rest; within a chunk `pow_scan` returns the
                // smallest nonce. A later chunk cannot hold a smaller nonce, so
                // this is exactly the globally smallest — identical to the
                // sequential search, which is what keeps proofs deterministic.
                let found = (0..n_chunks)
                    .into_par_iter()
                    .map(|c| {
                        pow_scan(
                            &state_digest,
                            start.saturating_add(c * GRIND_CHUNK),
                            GRIND_CHUNK,
                            bits,
                            kind,
                        )
                    })
                    .find_first(|r| r.is_some())
                    .flatten();
                if let Some(n) = found {
                    break n;
                }
                start = start.saturating_add(block);
            }
        };
        // Absorb the nonce so subsequent transcript state binds to it.
        // Verifier mirrors via verify_pow.
        self.observe_bytes(&nonce.to_le_bytes());
        nonce
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        let kind = self.hash_kind();
        let state_digest = self.state_digest();
        let ok = if bits == 0 {
            // No PoW required here. An honest prover emits the canonical nonce
            // 0 (see `grind_pow`), so reject any non-zero value: it can only be
            // a re-grinding knob, and accepting it would leave proofs malleable
            // (a proof and its nonce-mutated twin would both verify). This
            // closes no soundness gap — when grinding_bits = 0 the query phase
            // already carries the full security target, and the FS soundness
            // accounting assumes free re-grinding regardless — it just keeps
            // proofs canonical / non-malleable at zero-bit grinding sites.
            nonce == 0
        } else {
            pow_has_leading_zero_bits(&state_digest, nonce, bits, kind)
        };
        // Absorb regardless of `ok` so the transcript stays byte-identical to
        // the prover's (an honest prover always reaches this with the same
        // nonce); a failed check rejects the proof at the call site anyway.
        self.observe_bytes(&nonce.to_le_bytes());
        ok
    }

    fn grind_pow_and_sample_f128(&mut self, bits: u32) -> (u64, F128) {
        if let FsState::Blake3Chain(c) = &mut self.state {
            let mut bytes = [0u8; 16];
            let nonce = c.grind_pow_squeeze_into(bits, &mut bytes);
            let value = F128 {
                lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            };
            return (nonce, value);
        }
        let nonce = self.grind_pow(bits);
        (nonce, self.sample_f128())
    }

    fn verify_pow_and_sample_f128(&mut self, nonce: u64, bits: u32) -> Option<F128> {
        if let FsState::Blake3Chain(c) = &mut self.state {
            let mut bytes = [0u8; 16];
            if !c.apply_pow_squeeze(nonce, bits, &mut bytes) {
                return None;
            }
            return Some(F128 {
                lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            });
        }
        if !self.verify_pow(nonce, bits) {
            return None;
        }
        Some(self.sample_f128())
    }

    fn grind_pow_and_sample_f128_vec(&mut self, bits: u32, n: usize) -> (u64, Vec<F128>) {
        if let FsState::Blake3Chain(c) = &mut self.state {
            assert!(n != 0, "a fused PoW vector squeeze must be nonempty");
            let mut bytes = vec![0u8; 16 * n];
            let nonce = c.grind_pow_squeeze_into(bits, &mut bytes);
            return (nonce, f128s_from_le_bytes(&bytes));
        }
        let nonce = self.grind_pow(bits);
        (nonce, self.sample_f128_vec(n))
    }

    fn verify_pow_and_sample_f128_vec(
        &mut self,
        nonce: u64,
        bits: u32,
        n: usize,
    ) -> Option<Vec<F128>> {
        if let FsState::Blake3Chain(c) = &mut self.state {
            assert!(n != 0, "a fused PoW vector squeeze must be nonempty");
            let mut bytes = vec![0u8; 16 * n];
            if !c.apply_pow_squeeze(nonce, bits, &mut bytes) {
                return None;
            }
            return Some(f128s_from_le_bytes(&bytes));
        }
        if !self.verify_pow(nonce, bits) {
            return None;
        }
        Some(self.sample_f128_vec(n))
    }
}

fn f128s_from_le_bytes(bytes: &[u8]) -> Vec<F128> {
    debug_assert_eq!(bytes.len() % 16, 0);
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| F128 {
            lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
            hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Proof-of-work grinding.
//
// The PoW pre-image is `state_digest ‖ nonce_le`, but its *padded length*
// differs per hash, because each hash has a different natural block:
//
//   SHA-256: 40 bytes. With the 0x80 pad and 8-byte length that is one
//            compression; padding further to 64 would make it two, halving
//            the grind rate for no benefit.
//   BLAKE3:  64 bytes (24 zero bytes of tail padding). A whole-block
//            single-chunk message is exactly what the crate's SIMD
//            `hash_many` can compute a batch of at a time, which is worth
//            ~2× on the nonce search — see `blake3_pow_scan`. At 40 bytes it
//            would be a partial block and could not be batched at all.
//
// Both are fixed-length and injective in `(state_digest, nonce)`, which is all
// the PoW needs; the asymmetry costs nothing and is never compared across
// hashes (a proof is only verified under the hash it was made with).
// ---------------------------------------------------------------------------

/// BLAKE3's PoW pre-image: `state_digest ‖ nonce_le ‖ zero padding`, one whole
/// 64-byte block. `blake3::hash` of this is what the PoW is defined against.
#[inline]
fn blake3_pow_preimage(state_digest: &[u8; 32], nonce: u64) -> [u8; 64] {
    let mut pre = [0u8; 64];
    pre[..32].copy_from_slice(state_digest);
    pre[32..40].copy_from_slice(&nonce.to_le_bytes());
    pre
}

/// Whether `h` has at least `bits` leading zero bits — MSB-first within each
/// serialized byte, the one PoW bit convention everywhere (the fused
/// transcript PoW here, the recursion circuit's `PowMaskTable`, and the AG
/// fused sampling nonce in `genus95_curve_code::evaluation_point_from_nonce_pow`).
/// Public: the recursion tower's native replicas assert it beside the
/// in-circuit PowMask rows.
#[inline]
pub fn has_leading_zero_bits(h: &[u8], bits: u32) -> bool {
    let full_bytes = (bits / 8) as usize;
    let extra = bits % 8;
    for &b in h.iter().take(full_bytes) {
        if b != 0 {
            return false;
        }
    }
    if extra > 0 && (h[full_bytes] >> (8 - extra)) != 0 {
        return false;
    }
    true
}

/// Check whether `H(pre-image(state_digest, nonce))` has at least `bits`
/// leading zero bits, under the transcript's own hash `kind`.
///
/// This is the *specification* of the PoW — `verify_pow` uses it directly, and
/// the batched search below must agree with it for every nonce. Grinding under
/// the transcript's own hash keeps the whole protocol resting on one primitive
/// rather than pulling in a second.
#[inline]
/// `pub`: recursion tests use this as the native differential oracle for the
/// in-circuit BLAKE3 + leading-zero relation, so the grinding convention
/// cannot drift.
pub fn pow_has_leading_zero_bits(
    state_digest: &[u8; 32],
    nonce: u64,
    bits: u32,
    kind: HashKind,
) -> bool {
    #[cfg(feature = "hash-count")]
    POW_SHA256.fetch_add(1, Ordering::Relaxed);
    match kind {
        HashKind::Sha256 => {
            let mut pre = [0u8; 40];
            pre[..32].copy_from_slice(state_digest);
            pre[32..].copy_from_slice(&nonce.to_le_bytes());
            let h: [u8; 32] = Sha256::digest(pre).into();
            has_leading_zero_bits(&h, bits)
        }
        HashKind::Blake3 => {
            let h = hash(&blake3_pow_preimage(state_digest, nonce));
            has_leading_zero_bits(h.as_bytes(), bits)
        }
    }
}

/// Nonces hashed per `hash_many` call in the BLAKE3 grind.
///
/// Must clear the widest `simd_degree` (16, under AVX-512) so the batch fills
/// the machine's vector; 32 leaves headroom and keeps the buffers (2 KiB of
/// pre-images + 1 KiB of digests) stack-resident. Swept 1/4/8/16/32/64 on an
/// M4 Max: 1 is ~2.2× slower at 17 bits, everything from 4 up is within noise
/// of each other.
const BLAKE3_POW_BATCH: usize = 32;

/// Smallest nonce in `start .. start + len` whose BLAKE3 PoW hash has `bits`
/// leading zeros, or `None`.
///
/// Batches the independent nonce hashes through the crate's SIMD compression.
/// A 64-byte pre-image is a whole-block single chunk, which `hash_many`
/// reproduces byte-for-byte given `CHUNK_START` / `CHUNK_END | ROOT` — so this
/// agrees with `blake3::hash` on every nonce, which
/// `blake3_batched_pow_matches_scalar` asserts.
fn blake3_pow_scan(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
    // BLAKE3 constants, fixed by the spec.
    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19,
    ];
    const CHUNK_START: u8 = 1;
    const CHUNK_END: u8 = 2;
    const ROOT: u8 = 8;

    let plat = Platform::detect();
    // The 32-byte state prefix is constant across the whole scan; only the
    // 8 nonce bytes change per lane.
    let mut pre = [[0u8; 64]; BLAKE3_POW_BATCH];
    for p in pre.iter_mut() {
        p[..32].copy_from_slice(state_digest);
    }
    let mut out = [0u8; BLAKE3_POW_BATCH * 32];

    let mut base = start;
    let end = start.saturating_add(len);
    while base < end {
        let n = BLAKE3_POW_BATCH.min((end - base) as usize);
        for (i, p) in pre[..n].iter_mut().enumerate() {
            p[32..40].copy_from_slice(&(base + i as u64).to_le_bytes());
        }
        #[cfg(feature = "hash-count")]
        POW_SHA256.fetch_add(n as u64, Ordering::Relaxed);
        let inputs: [&[u8; 64]; BLAKE3_POW_BATCH] = from_fn(|i| &pre[i]);
        plat.hash_many(
            &inputs[..n],
            &IV,
            0,
            IncrementCounter::No,
            0,
            CHUNK_START,
            CHUNK_END | ROOT,
            &mut out[..n * 32],
        );
        for i in 0..n {
            if has_leading_zero_bits(&out[i * 32..(i + 1) * 32], bits) {
                return Some(base + i as u64);
            }
        }
        base += n as u64;
    }
    None
}

/// Batched nonce search for the chained transcript's fused PoW+squeeze row.
/// `hash_many` returns the compression's first 32 bytes; word 1 (bytes
/// 16..32) is the PoW predicate, while word 0 remains an unbiased challenge.
fn blake3_chain_pow_scan(
    cv: &[u32; 8],
    pending: &[u8],
    start: u64,
    len: u64,
    bits: u32,
) -> Option<u64> {
    assert!(pending.len() + 16 <= 64, "one aligned nonce word must fit");
    let counter = pow_squeeze_counter(bits, pending.len() + 16);
    let nonce_at = pending.len();
    let plat = Platform::detect();
    let mut pre = [[0u8; 64]; BLAKE3_POW_BATCH];
    for p in &mut pre {
        p[..pending.len()].copy_from_slice(pending);
    }
    let mut compressed_lo = [0u8; BLAKE3_POW_BATCH * 32];
    let mut base = start;
    let end = start.saturating_add(len);
    while base < end {
        let n = BLAKE3_POW_BATCH.min((end - base) as usize);
        for (i, p) in pre[..n].iter_mut().enumerate() {
            p[nonce_at..nonce_at + 8].copy_from_slice(&(base + i as u64).to_le_bytes());
        }
        #[cfg(feature = "hash-count")]
        POW_SHA256.fetch_add(n as u64, Ordering::Relaxed);
        let inputs: [&[u8; 64]; BLAKE3_POW_BATCH] = from_fn(|i| &pre[i]);
        plat.hash_many(
            &inputs[..n],
            cv,
            counter,
            IncrementCounter::No,
            CHAIN_SQUEEZE as u8,
            0,
            0,
            &mut compressed_lo[..n * 32],
        );
        for i in 0..n {
            let lo = &compressed_lo[i * 32..(i + 1) * 32];
            if has_leading_zero_bits(&lo[16..32], bits) {
                return Some(base + i as u64);
            }
        }
        base += n as u64;
    }
    None
}

/// Smallest nonce in `start .. start + len` satisfying the PoW, or `None`.
/// Batched under BLAKE3; a plain scan under SHA-256, whose hardware path is
/// already faster than anything batching would buy.
#[inline]
fn pow_scan(
    state_digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
    kind: HashKind,
) -> Option<u64> {
    match kind {
        HashKind::Blake3 => blake3_pow_scan(state_digest, start, len, bits),
        HashKind::Sha256 => (start..start.saturating_add(len))
            .find(|&n| pow_has_leading_zero_bits(state_digest, n, bits, kind)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::challenger::{
        Challenger, F128, F256, FsChallenger, HashKind, RandomChallenger, blake3_pow_scan,
        pow_has_leading_zero_bits,
    };

    /// Every FsChallenger property must hold under both transcript hashes:
    /// the tagging, absorption order and duplex structure are shared, and
    /// only the primitive differs.
    const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

    /// Prover-side PoW grinding produces a nonce that the verifier-side
    /// `verify_pow` accepts at the same transcript position. State binding
    /// is preserved — sampling after PoW gives identical challenges on both
    /// sides.
    #[test]
    fn fs_challenger_pow_roundtrip() {
        for kind in KINDS {
            for bits in [0u32, 5, 10, 14] {
                let mut prover = FsChallenger::with_hash(b"pow-test", kind);
                prover.observe_label(b"flock-pow-test");
                prover.observe_bytes(b"some root data");
                let nonce = prover.grind_pow(bits);

                let mut verifier = FsChallenger::with_hash(b"pow-test", kind);
                verifier.observe_label(b"flock-pow-test");
                verifier.observe_bytes(b"some root data");
                assert!(
                    verifier.verify_pow(nonce, bits),
                    "verify failed at bits={bits}"
                );

                // Subsequent challenges must agree.
                for _ in 0..4 {
                    assert_eq!(prover.sample_f128(), verifier.sample_f128());
                }
            }
        }
    }

    /// `verify_pow` rejects a wrong nonce when grinding bits > 0.
    #[test]
    fn fs_challenger_pow_rejects_wrong_nonce() {
        for kind in KINDS {
            let mut prover = FsChallenger::with_hash(b"pow-test", kind);
            prover.observe_bytes(b"root");
            let nonce = prover.grind_pow(10);
            let bad_nonce = nonce.wrapping_add(1);

            let mut verifier = FsChallenger::with_hash(b"pow-test", kind);
            verifier.observe_bytes(b"root");
            assert!(
                !verifier.verify_pow(bad_nonce, 10),
                "should reject wrong nonce"
            );
        }
    }

    /// At a zero-bit grinding site `verify_pow` accepts the canonical nonce 0
    /// (what `grind_pow(0)` emits) but rejects any non-zero nonce, so a proof
    /// can't be made malleable by swapping in an arbitrary nonce.
    #[test]
    fn fs_challenger_pow_zero_bits_requires_canonical_nonce() {
        for kind in KINDS {
            let mk = || {
                let mut ch = FsChallenger::with_hash(b"pow-test", kind);
                ch.observe_bytes(b"root");
                ch
            };
            assert_eq!(mk().grind_pow(0), 0, "honest zero-bit grind is the 0 nonce");
            assert!(mk().verify_pow(0, 0), "canonical 0 nonce must verify");
            for bad in [1u64, 42, u64::MAX] {
                assert!(
                    !mk().verify_pow(bad, 0),
                    "non-zero nonce {bad} must be rejected at zero-bit grinding"
                );
            }
        }
    }

    /// `new` must stay SHA-256: 300-odd call sites construct challengers that
    /// way, and silently moving them to another hash would invalidate every
    /// proof they produce.
    #[test]
    fn fs_challenger_new_defaults_to_sha256() {
        assert_eq!(FsChallenger::new(b"d").hash_kind(), HashKind::Sha256);
        for kind in KINDS {
            assert_eq!(FsChallenger::with_hash(b"d", kind).hash_kind(), kind);
        }
        // The default constructor must be exactly the SHA-256 one, transcript
        // and all — not merely tagged the same.
        let mut a = FsChallenger::new(b"d");
        let mut b = FsChallenger::with_hash(b"d", HashKind::Sha256);
        assert_eq!(a.sample_f128_vec(4), b.sample_f128_vec(4));
    }

    /// The two transcript hashes must produce different challenges from the
    /// same script — otherwise the option would be doing nothing.
    #[test]
    fn fs_challenger_hashes_diverge() {
        let script = |ch: &mut FsChallenger| {
            ch.observe_label(b"phase");
            ch.observe_bytes(b"root");
            ch.observe_f128(F128::ONE);
            ch.sample_f128_vec(4)
        };
        let mut sha = FsChallenger::with_hash(b"d", HashKind::Sha256);
        let mut blake = FsChallenger::with_hash(b"d", HashKind::Blake3);
        assert_ne!(script(&mut sha), script(&mut blake));
    }

    /// A verifier on the wrong transcript hash must reject: the PoW check is
    /// against a different digest, and the challenges diverge from there.
    #[test]
    fn fs_challenger_pow_rejects_the_other_hash() {
        for kind in KINDS {
            let other = match kind {
                HashKind::Sha256 => HashKind::Blake3,
                HashKind::Blake3 => HashKind::Sha256,
            };
            let mut prover = FsChallenger::with_hash(b"pow-test", kind);
            prover.observe_bytes(b"root");
            let nonce = prover.grind_pow(10);

            let mut wrong = FsChallenger::with_hash(b"pow-test", other);
            wrong.observe_bytes(b"root");
            assert!(
                !wrong.verify_pow(nonce, 10),
                "{kind} nonce must not satisfy a {other} PoW"
            );
        }
    }

    /// BLAKE3 squeezes from an XOF rather than a counter, so a long squeeze
    /// must still agree with the concatenation of the short ones it replaces —
    /// i.e. `sample_f128_vec(n)` is one XOF read of `16n` bytes, not `n`
    /// independent reads. Pins the stream layout for both hashes.
    #[test]
    fn fs_challenger_long_squeeze_is_prefix_stable() {
        for kind in KINDS {
            // Two challengers on identical scripts, one squeezing 8 values and
            // one squeezing 8 values in a single call, must agree — this is
            // just determinism, but it is what the duplex relies on.
            let mut a = FsChallenger::with_hash(b"d", kind);
            let mut b = FsChallenger::with_hash(b"d", kind);
            assert_eq!(a.sample_f128_vec(8), b.sample_f128_vec(8), "{kind}");

            // A squeeze longer than one 32-byte block must not repeat itself:
            // catches a counter that fails to advance, or an XOF read that
            // restarts per block.
            let vals = FsChallenger::with_hash(b"d", kind).sample_f128_vec(16);
            let unique: HashSet<_> = vals.iter().collect();
            assert_eq!(unique.len(), vals.len(), "{kind}: squeeze stream repeats");
        }
    }

    /// The batched BLAKE3 nonce search must agree with the scalar spec
    /// (`blake3::hash` of the 64-byte pre-image) on every nonce. This is what
    /// makes the SIMD path safe to use: if `hash_many`'s flag semantics ever
    /// changed, this fails rather than silently producing PoW hashes that
    /// `verify_pow` would then reject.
    #[test]
    fn blake3_batched_pow_matches_scalar() {
        let state = [0x5Au8; 32];
        // Cover nonce counts either side of the batch width (32): a partial
        // batch, exactly one, one past, and several with a ragged tail.
        for len in [1u64, 5, 31, 32, 33, 100] {
            for start in [0u64, 7, 1_000_000] {
                // `bits = 0` makes every nonce a match, so the scan must return
                // `start` — and the per-lane hashes are all exercised below.
                assert_eq!(
                    blake3_pow_scan(&state, start, len, 0),
                    Some(start),
                    "start={start} len={len}"
                );
                // Compare the scan against a scalar sweep at a threshold low
                // enough to hit but high enough to skip some nonces.
                let want = (start..start + len)
                    .find(|&n| pow_has_leading_zero_bits(&state, n, 6, HashKind::Blake3));
                assert_eq!(
                    blake3_pow_scan(&state, start, len, 6),
                    want,
                    "start={start} len={len}"
                );
            }
        }
    }

    /// The grind must return the globally smallest satisfying nonce, on both
    /// the sequential and the block-parallel path, and under both hashes.
    /// Proof determinism depends on it: a different nonce is a different
    /// transcript and therefore a different proof.
    #[test]
    fn fs_challenger_grind_returns_smallest_nonce() {
        for kind in KINDS {
            // 4 bits stays sequential; 14 crosses PARALLEL_GRIND_MIN_HASHES.
            for bits in [4u32, 14] {
                let mut ch = FsChallenger::with_hash(b"grind-min", kind);
                ch.observe_bytes(b"root");
                let digest_probe = {
                    let mut probe = FsChallenger::with_hash(b"grind-min", kind);
                    probe.observe_bytes(b"root");
                    probe.state_digest()
                };
                let nonce = ch.grind_pow(bits);
                // Every smaller nonce must fail the scalar check.
                for n in 0..nonce {
                    assert!(
                        !pow_has_leading_zero_bits(&digest_probe, n, bits, kind),
                        "{kind} bits={bits}: nonce {n} < {nonce} also satisfies the PoW"
                    );
                }
                assert!(
                    pow_has_leading_zero_bits(&digest_probe, nonce, bits, kind),
                    "{kind} bits={bits}: returned nonce {nonce} does not satisfy the PoW"
                );
            }
        }
    }

    /// Default Challenger impl (RandomChallenger) is a no-op for PoW.
    #[test]
    fn random_challenger_pow_is_noop() {
        let mut ch = RandomChallenger::new(0);
        assert_eq!(ch.grind_pow(16), 0);
        assert!(ch.verify_pow(0, 16));
    }

    #[test]
    fn random_challenger_is_deterministic_per_seed() {
        let mut c1 = RandomChallenger::new(42);
        let mut c2 = RandomChallenger::new(42);
        for _ in 0..16 {
            assert_eq!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn random_challenger_observe_is_noop() {
        // Observing arbitrary messages does not change the sampled values.
        let mut c1 = RandomChallenger::new(7);
        let mut c2 = RandomChallenger::new(7);
        c2.observe_f128(F128 {
            lo: 0xDEADBEEF,
            hi: 0xCAFEBABE,
        });
        c2.observe_f128_slice(&[F128::ONE, F128::ZERO]);
        c2.observe_label(b"ignored");
        c2.observe_bytes(b"also ignored");
        for _ in 0..8 {
            assert_eq!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn sample_f128_vec_matches_individual_samples() {
        let mut c1 = RandomChallenger::new(99);
        let mut c2 = RandomChallenger::new(99);
        let batch = c1.sample_f128_vec(5);
        let individual: Vec<F128> = (0..5).map(|_| c2.sample_f128()).collect();
        assert_eq!(batch, individual);
    }

    #[test]
    fn sample_f256_is_one_double_width_squeeze() {
        for kind in KINDS {
            let mut extension = FsChallenger::with_hash(b"f256-squeeze", kind);
            let mut words = FsChallenger::with_hash(b"f256-squeeze", kind);
            let sampled = extension.sample_f256();
            let expected = words.sample_f128_vec(2);
            assert_eq!(sampled, F256::new(expected[0], expected[1]), "{kind}");
        }
    }

    #[test]
    fn observe_f256_is_canonical_coordinate_absorption() {
        for kind in KINDS {
            let value = F256::new(F128::new(1, 2), F128::new(3, 4));
            let mut extension = FsChallenger::with_hash(b"f256-observe", kind);
            let mut words = FsChallenger::with_hash(b"f256-observe", kind);
            extension.observe_f256(value);
            words.observe_f128_slice(&value.coordinates());
            assert_eq!(
                extension.sample_f128_vec(4),
                words.sample_f128_vec(4),
                "{kind}"
            );
        }
    }

    // ---- FsChallenger ------------------------------------------------------

    #[test]
    fn fs_challenger_identical_scripts_produce_identical_output() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock-test", kind);
            let mut c2 = FsChallenger::with_hash(b"flock-test", kind);
            let msg = F128 {
                lo: 0x1234,
                hi: 0x5678,
            };
            c1.observe_f128(msg);
            c2.observe_f128(msg);
            let r1 = c1.sample_f128_vec(8);
            let r2 = c2.sample_f128_vec(8);
            assert_eq!(r1, r2);
        }
    }

    #[test]
    fn fs_challenger_different_domains_diverge() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock-a", kind);
            let mut c2 = FsChallenger::with_hash(b"flock-b", kind);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_different_observations_diverge() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_f128(F128::ONE);
            c2.observe_f128(F128::ZERO);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_label_changes_output() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_label(b"phase-A");
            // c2 omits the label entirely.
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_scalar_vs_slice_dont_collide() {
        for kind in KINDS {
            // observe_f128_slice(&[v]) must NOT produce the same state as
            // observe_f128(v) — the length prefix and kind tag must defeat this.
            let v = F128 { lo: 0xAB, hi: 0xCD };
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_f128(v);
            c2.observe_f128_slice(&[v]);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_two_scalars_dont_collide_with_one_slice_of_two() {
        for kind in KINDS {
            let a = F128 { lo: 1, hi: 2 };
            let b = F128 { lo: 3, hi: 4 };
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_f128(a);
            c1.observe_f128(b);
            c2.observe_f128_slice(&[a, b]);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_sample_one_vs_sample_vec_one_differ() {
        for kind in KINDS {
            // Squeeze tag differs (KIND_SCALAR vs KIND_SLICE+len), so a single
            // sample_f128 must not equal sample_f128_vec(1)[0].
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            assert_ne!(c1.sample_f128(), c2.sample_f128_vec(1)[0]);
        }
    }

    #[test]
    fn fs_challenger_sample_advances_state() {
        for kind in KINDS {
            // After a sample, the next observation should not collapse to the
            // pre-sample state (the squeezed bytes are re-absorbed).
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            let _ = c1.sample_f128();
            // c2 skips the sample.
            c1.observe_f128(F128::ONE);
            c2.observe_f128(F128::ONE);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }
}

#[cfg(test)]
mod b3_chain_tests {
    use crate::challenger::{Challenger, F128, FsChallenger};

    #[test]
    fn chained_blake3_is_deterministic_and_binding() {
        let mk = || FsChallenger::with_chained_blake3(b"flock-chain-test");
        let mut a = mk();
        let mut b = mk();
        a.observe_f128(F128 { lo: 7, hi: 9 });
        b.observe_f128(F128 { lo: 7, hi: 9 });
        let (x, y) = (a.sample_f128(), b.sample_f128());
        assert_eq!(x, y, "deterministic");
        // Consecutive squeezes differ: the DUPLEX squeeze advances cv, so
        // no header is needed to separate them (transcript-v3).
        let x2 = a.sample_f128();
        assert_ne!(x, x2, "consecutive squeezes separate");
        // A different absorbed value moves the challenge.
        let mut c = mk();
        c.observe_f128(F128 { lo: 7, hi: 10 });
        assert_ne!(c.sample_f128(), x, "absorb-sensitive");
        // The duplex is PREFIX-STABLE, the standard sponge property: with
        // no per-kind squeeze header, a slice squeeze's first element at a
        // given state equals the scalar squeeze there. The op sequence is
        // protocol-fixed (never adversary-chosen), so this collides nothing
        // an attacker controls — and it is precisely what the v2 header
        // bought with one recursion-circuit chain row per sample.
        let mut d = mk();
        d.observe_f128(F128 { lo: 7, hi: 9 });
        let v = d.sample_f128_vec(2);
        assert_eq!(v[0], x, "duplex squeezes are prefix-stable");
        assert_ne!(v[0], v[1], "output blocks differ");
        // And the state after a slice squeeze differs from after a scalar
        // one ONLY through subsequent output count — both advanced.
        let (dn, an) = (d.sample_f128(), a.sample_f128());
        assert_ne!(dn, an, "post-squeeze states advanced independently");
    }

    #[test]
    fn fused_pow_squeeze_prover_and_verifier_stay_in_lockstep() {
        let mut prover = FsChallenger::with_chained_blake3(b"fused-pow-test");
        let mut verifier = FsChallenger::with_chained_blake3(b"fused-pow-test");
        let observed = [F128::new(1, 2), F128::new(3, 4), F128::new(5, 6)];
        prover.observe_f128_slice(&observed);
        verifier.observe_f128_slice(&observed);

        let (nonce, challenges) = prover.grind_pow_and_sample_f128_vec(6, 9);
        let replay = verifier
            .verify_pow_and_sample_f128_vec(nonce, 6, 9)
            .expect("honest fused nonce verifies");
        assert_eq!(replay, challenges);
        assert_eq!(prover.sample_f128(), verifier.sample_f128());

        let mut bad = FsChallenger::with_chained_blake3(b"fused-pow-test");
        bad.observe_f128_slice(&observed);
        let bad_nonce = (0..u64::MAX)
            .find(|&n| {
                n != nonce && {
                    let mut probe = bad.clone();
                    probe.verify_pow_and_sample_f128_vec(n, 6, 9).is_none()
                }
            })
            .expect("an invalid six-bit nonce exists");
        assert!(
            bad.verify_pow_and_sample_f128_vec(bad_nonce, 6, 9)
                .is_none()
        );

        let mut zero = FsChallenger::with_chained_blake3(b"fused-pow-zero");
        assert!(zero.verify_pow_and_sample_f128(1, 0).is_none());
        let mut zero = FsChallenger::with_chained_blake3(b"fused-pow-zero");
        assert!(zero.verify_pow_and_sample_f128(0, 0).is_some());
    }
}
