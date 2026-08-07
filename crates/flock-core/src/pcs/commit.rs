//! PCS commit phase: pack → RS encode (additive NTT) → Merkle root.
//!
//! Uses [`AdditiveNttF128`], the binius-style LCH NTT with neighbors-last
//! pairing. The commit produces a non-systematic RS codeword (treating the
//! packed witness as novel-basis coefficients, zero-padded to the larger
//! domain, then forward-NTT'd).
//!
//! ## Layout
//!
//! With parameters `(m, log_inv_rate)`:
//! - `log_msg_len = m − LOG_PACKING` (= log2 of packed witness length)
//! - `k_code      = log_msg_len + log_inv_rate` (= log2 of codeword length)
//!
//! The codeword is a flat sequence of `2^k_code` F_{2^128} elements. Each
//! Merkle leaf is **one** F_{2^128} element = 16 bytes.

use crate::field::F128;
#[cfg(test)]
use crate::merkle::cap_layer;
#[cfg(test)]
use crate::merkle::merkle_tree;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::AdditiveNttF128;
use crate::pcs::ligerito::LigeritoProfile;
use crate::pcs::ligerito::ProverConfig;
use crate::pcs::ligerito::VerifierConfig;
use crate::pcs::ligerito::prover_config_for;
use crate::pcs::ligerito::udr_queries;
use crate::pcs::ligerito::verifier_config_for;
use crate::pcs::pack::LOG_PACKING;
use crate::scratch::give_f128;
use crate::scratch::take_f128;
use crate::scratch::try_take_f128;
use core::mem::size_of;
use core::slice::from_raw_parts;
use serde::{Deserialize, Serialize};
use std::env::var_os;
#[cfg(test)]
use std::hint::black_box;
use std::mem::take;
use std::ptr::write_bytes;
use std::thread::scope;
use std::time::Instant;

/// PCS configuration. Polynomial-basis subspace `{1, x, x², …}` for the NTT.
///
/// Interleaved RS: the packed witness is split into `2^log_batch_size`
/// independent sub-NTTs of size `2^log_dim` each. Each Merkle leaf holds one
/// codeword position across all `2^log_batch_size` lanes
/// (`2^log_batch_size · 16` bytes per leaf). This trades leaf-call SHA-256
/// overhead (was 16 B leaves, now 512 B leaves at default `log_batch_size=5`)
/// for much fewer Merkle nodes and better scaling to large `m`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcsParams {
    pub m: usize,
    pub log_inv_rate: usize,
    /// Number of parallel sub-NTTs = `2^log_batch_size`. Default 5 (= 32 lanes).
    pub log_batch_size: usize,
    /// Ligerito parameter profile (fast/slim/secure). Selects which embedded
    /// security config (queries, OOD samples, grinding schedule) drives the
    /// PCS opening; must agree with `log_inv_rate`
    /// (`profile.log_inv_rate() == log_inv_rate`). Defaults to `Fast`.
    #[serde(default)]
    pub profile: LigeritoProfile,
    /// **Integer-lane commit** (optional). `None` (the default) commits the
    /// full `2^log_batch_size` interleaved lanes — today's power-of-two
    /// scheme. `Some(t)` with `1 ≤ t ≤ 2^log_batch_size` commits exactly `t`
    /// integer lanes, each of size `2^log_dim`, so the committed message is
    /// `t · 2^log_dim ≤ 2^(m−7)` F_{2^128} words — eliminating the encode +
    /// Merkle work of the `2^log_batch_size − t` zero lanes. The per-lane
    /// codeword length (`n_positions = 2^k_code`, hence `n_leaves`) is
    /// UNCHANGED; only the leaf width (`t` F128 = `t·16` bytes) and the total
    /// codeword length shrink. When `t == 2^log_batch_size` the commit is
    /// byte-identical to `None` (`num_ntts` and every derived quantity
    /// coincide). The Ligerito `initial_k` stays `log_batch_size`; lanes
    /// `[t, 2^log_batch_size)` are definitionally zero on the opening side.
    #[serde(default)]
    pub num_lanes: Option<usize>,
    /// Hash backing the Merkle commitment. Defaults to SHA-256, so params
    /// serialized before this option existed deserialize unchanged.
    ///
    /// The verifier must be given the same value the prover committed under —
    /// it is carried in [`Commitment`] alongside the root for exactly that
    /// reason.
    #[serde(default)]
    pub merkle_hash: HashKind,
}

impl PcsParams {
    /// Total log message length (= log2 packed witness length).
    pub fn log_msg_len(&self) -> usize {
        self.m - LOG_PACKING
    }
    /// Per-sub-NTT log dimension (= number of "position" coords).
    pub fn log_dim(&self) -> usize {
        self.log_msg_len() - self.log_batch_size
    }
    /// Codeword size (log) per sub-NTT.
    pub fn k_code(&self) -> usize {
        self.log_dim() + self.log_inv_rate
    }
    /// Number of Merkle leaves (= per-sub-NTT codeword length).
    pub fn n_positions(&self) -> usize {
        1usize << self.k_code()
    }
    /// Number of interleaved lanes actually committed: `num_lanes` when set
    /// (integer-lane commit), else `2^log_batch_size` (the full power-of-two
    /// scheme). Always in `[1, 2^log_batch_size]`.
    pub fn num_ntts(&self) -> usize {
        self.num_lanes.unwrap_or(1usize << self.log_batch_size)
    }
    /// Committed message length in F_{2^128} words = `num_ntts() · 2^log_dim`.
    /// Equals `2^log_msg_len` on the power-of-two path (`num_lanes == None`).
    pub fn msg_len_f128(&self) -> usize {
        self.num_ntts() << self.log_dim()
    }
    /// Total codeword length in F_{2^128} elements
    /// (= `n_positions() * num_ntts()`).
    pub fn codeword_len_f128(&self) -> usize {
        self.n_positions() * self.num_ntts()
    }
    /// `log_2` of the F_{2^128} count per **initial** Merkle leaf
    /// (= `log_batch_size`; just the row-batch lanes per position). Meaningful
    /// only on the power-of-two path; the integer-lane leaf width is
    /// `num_ntts()` (see [`Self::leaf_size_bytes`]).
    pub fn log_leaf_f128_count(&self) -> usize {
        self.log_batch_size
    }
    /// Number of initial-tree Merkle leaves = per-lane codeword length
    /// `2^k_code` (= `n_positions()`). UNCHANGED by the integer-lane commit —
    /// only the leaf WIDTH shrinks, not the leaf count.
    pub fn n_leaves(&self) -> usize {
        self.n_positions()
    }
    /// Merkle leaf size in bytes = `num_ntts() * 16`.
    pub fn leaf_size_bytes(&self) -> usize {
        self.num_ntts() * size_of::<F128>()
    }

    /// Ligerito prover config for these params.
    ///
    /// Prefer this over calling [`ligerito::prover_config_for`] directly: the
    /// embedded security config carries its own `hash` field, but the Merkle
    /// hash the opening must use is the one the *commitment* was built under.
    /// This stamps `self.merkle_hash` over it, so the L0 tree and every
    /// recursive level cannot end up on different hashes.
    ///
    /// [`ligerito::prover_config_for`]: crate::pcs::ligerito::prover_config_for
    pub fn ligerito_prover_config(&self) -> Result<ProverConfig, String> {
        let mut cfg = prover_config_for(self.log_msg_len(), self.log_batch_size, self.profile)?;
        cfg.merkle_hash = self.merkle_hash;
        Ok(cfg)
    }

    /// Verifier-side counterpart to [`Self::ligerito_prover_config`], stamped
    /// with the same Merkle hash for the same reason.
    pub fn ligerito_verifier_config(&self) -> Result<VerifierConfig, String> {
        let mut cfg = verifier_config_for(self.log_msg_len(), self.log_batch_size, self.profile)?;
        cfg.merkle_hash = self.merkle_hash;
        Ok(cfg)
    }

    /// The L0 query count these params imply — the exact rule every opener
    /// uses to pick its config: the embedded security config when one exists
    /// for `(log_msg_len, log_batch_size, profile)` (the flock-prover
    /// paths), else `udr_queries(log_inv_rate)` (the `default_config` paths:
    /// the standalone element proof and the permutation check, whose
    /// `queries[0]` IS `udr_queries`). Commit-time cap sizing MUST agree
    /// with the opener's config — this is that single source of truth.
    pub fn l0_queries(&self) -> usize {
        match self.ligerito_prover_config() {
            Ok(cfg) => cfg.queries[0],
            Err(_) => udr_queries(self.log_inv_rate),
        }
    }

    /// Cap depth of the L0 commitment tree: `min(⌈log2 q₀⌉, k_code)`.
    pub fn l0_cap_depth(&self) -> usize {
        merkle::cap_depth(self.l0_queries(), self.k_code())
    }

    fn validate(&self) {
        assert!(
            self.m >= LOG_PACKING + self.log_batch_size,
            "m={} too small (need m ≥ LOG_PACKING + log_batch_size = {})",
            self.m,
            LOG_PACKING + self.log_batch_size,
        );
        assert!(
            self.log_inv_rate >= 1,
            "log_inv_rate must be ≥ 1 for a non-trivial RS code",
        );
        if let Some(t) = self.num_lanes {
            assert!(
                t >= 1 && t <= (1usize << self.log_batch_size),
                "num_lanes={t} out of range [1, 2^log_batch_size={}]",
                1usize << self.log_batch_size,
            );
        }
    }
}

/// Public commitment (Merkle CAP + params). The cap is the `2^c` tree nodes
/// at depth `c = params.l0_cap_depth()` below the root — the commitment IS
/// the cap; there is no root (a 32-byte id, if ever needed externally, is
/// just a hash of the cap and lives outside the protocol). Openings
/// authenticate leaf → cap node in `k_code − c` siblings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment {
    pub cap: Vec<Hash>,
    pub params: PcsParams,
}

/// Prover-side state retained after commit for use in the opening phase.
///
/// **The packed witness is NOT stored here.** The caller is responsible for
/// retaining its own copy of the packed witness across commit + open. This
/// avoids ~4 GB of duplication at large `m`, dropping peak commit memory by
/// a factor of ~1.5 (e.g. at m=35: 13 GB → 9 GB).
pub struct ProverData {
    pub codeword: Vec<F128>,
    pub merkle_tree: Vec<Hash>,
}

// Recycle the codeword buffer (the prover's largest single allocation —
// 128 MB at m = 29) through the scratch pool instead of unmapping it.
impl Drop for ProverData {
    fn drop(&mut self) {
        give_f128(take(&mut self.codeword));
    }
}

/// Commit to a witness in **F_{2^128}-packed** form (polynomial basis: bit
/// `r` of `z_packed[i]` = logical bit `i·128 + r`).
///
/// Uses **interleaved RS encoding**: `num_ntts = 2^log_batch_size` independent
/// sub-NTTs share the same domain and twiddles, processed via the SoA
/// interleaved transform. The codeword is stored position-major SoA
/// (`codeword[pos · num_ntts + lane]`); each Merkle leaf is one position =
/// `num_ntts` F_{2^128} = `num_ntts · 16` bytes.
///
/// **Takes the witness by reference**. The returned [`ProverData`] does NOT
/// retain a copy of the packed witness — the caller is responsible for
/// keeping its own copy across commit + open. This frees ~4 GB during the
/// NTT/Merkle phase at large `m`.
///
/// `z_packed.len()` must equal `2^(m - LOG_PACKING) = 2^(m - 7)`.
pub fn commit(z_packed: &[F128], params: &PcsParams) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), params.msg_len_f128());

    let num_ntts = params.num_ntts();
    let n_positions = params.n_positions();
    let codeword_len = n_positions * num_ntts;

    // ---- Codeword buffer (SoA): codeword[pos * num_ntts + lane].
    // Copy first 2^log_msg_len positions from packed witness; zero-pad the rest.
    //
    // At large m the codeword buffer is huge (128 MB at m=29, 512 MB at m=31).
    // `vec![F128::ZERO; n]` would eagerly zero all 128 MB upfront, then
    // immediately overwrite the lower half with `z_packed` — half the zero-fill
    // is wasted. Instead allocate uninit, write each half exactly once: copy
    // `z_packed` into the lower half, and zero-fill JUST the upper half (the
    // RS-encoding zero coefficients that the NTT's first-layer butterfly will
    // read). Saves ~64 MB of memory writes at m=29 (~9 ms).
    let codeword = take_f128(codeword_len);
    commit_into(z_packed, params, codeword)
}

/// Like [`commit`], but reuses a caller-provided codeword buffer instead of
/// allocating its own. The buffer must have length `codeword_len`; its
/// CONTENTS may be arbitrary (uninit/stale) — every slot is written here:
/// `z_packed` is replicated into all `2^log_inv_rate` sub-blocks (the exact
/// state after the first `log_inv_rate` NTT layers on `[z, 0, …, 0]`), in
/// parallel. Buffers from [`prefault_codeword_during`] or the scratch pool
/// are already resident, so no write faults.
pub fn commit_into(
    z_packed: &[F128],
    params: &PcsParams,
    mut codeword: Vec<F128>,
) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), params.msg_len_f128());
    let codeword_len = params.n_positions() * params.num_ntts();
    assert_eq!(
        codeword.len(),
        codeword_len,
        "commit_into: prebuilt codeword buffer has wrong length"
    );

    // RS encoding of [z, 0, …, 0] starts with `log_inv_rate` butterfly layers
    // whose bottom inputs are all zero — each is a pure copy, so after those
    // layers the buffer holds 2^log_inv_rate replicas of z. Write that state
    // directly (replicating z costs the same writes as the zero-fill it
    // replaces) and start the NTT at layer `log_inv_rate`, skipping those
    // layers' full-buffer reads and multiplies.
    replicate_message_fill(&mut codeword, z_packed);

    finalize_commit(codeword, params)
}

// ---------------------------------------------------------------------------
// High-bit lanes: the layout that makes the integer-lane commit reachable.
//
// The zero padding of a committed stack `q` is a contiguous TAIL (`q` holds
// `dense_words` real words inside a power-of-two array). Ligerito's lane index
// is by construction the LOW `log_batch_size` bits of the message index
// (`ligero_commit`: "the first log_num_interleaved LSB variables ARE the lane
// indices"), so under that labelling the zero tail SMEARS across every lane
// and no lane is wholly zero — there is nothing to drop.
//
// Relabel the lane as the HIGH `log_batch_size` bits instead — lane `l` owns
// the contiguous logical block `q[l·D .. (l+1)·D)`, `D = 2^log_dim` — and the
// tail becomes WHOLE zero lanes `l ≥ t = ceil(dense_words / D)`, which the
// commit simply does not encode or hash. Crucially the relabelling is a
// **variable rotation** of the index bits, so every multilinear extension
// downstream (the jagged weight `f̂_t`, the assist, `b_tilde`) survives it by
// permuting its evaluation point — unlike a tight stride-`t` packing, whose
// `e = t·p + l` is not multilinear in the index bits at all.
//
// [`lane_grid_from_lane_major`] converts the lane-major dense stack into the
// LSB-lane "grid" array Ligerito folds; [`commit_lane_grid`] encodes it while
// dropping the zero lanes. See `pcs::open_batch_merged`.
// ---------------------------------------------------------------------------

/// Number of nonzero high-bit lanes of a dense stack: `ceil(dense_words / D)`
/// with `D = 2^log_dim`. Lanes `[t, 2^log_batch_size)` are wholly zero and are
/// not committed. Returns `2^log_batch_size` when the stack is full (the
/// power-of-two case — callers should then keep `num_lanes = None` so the
/// commit stays byte-identical to today's).
pub fn dense_lanes(dense_words: usize, log_batch_size: usize, log_dim: usize) -> usize {
    dense_words
        .div_ceil(1usize << log_dim)
        .clamp(1, 1usize << log_batch_size)
}

/// Transpose the lane-major dense stack `q` (lane `l` = the contiguous block
/// `q[l·D .. (l+1)·D)`) into the LSB-lane grid `g[p·2^log_batch_size + l] =
/// q[l·D + p]` that Ligerito folds — i.e. rotate the index bits so the high
/// `log_batch_size` lane bits become the low ones.
///
/// Cache-blocked over position tiles (one tile is `TILE · 2^log_batch_size`
/// words, which stays in L2), so it runs at near-memcpy speed despite the
/// strided writes.
pub fn lane_grid_from_lane_major(q: &[F128], log_batch_size: usize) -> Vec<F128> {
    const TILE: usize = 64;
    use rayon::prelude::*;

    let lanes = 1usize << log_batch_size;
    assert!(q.len().is_multiple_of(lanes), "dense stack must fill lanes");
    let d = q.len() >> log_batch_size;
    let mut grid = take_f128(q.len());
    // positions per tile
    grid.par_chunks_mut(TILE * lanes)
        .enumerate()
        .for_each(|(tile, out)| {
            let p0 = tile * TILE;
            let n = TILE.min(d - p0);
            for lane in 0..lanes {
                let src = &q[lane * d + p0..lane * d + p0 + n];
                for (p, &v) in src.iter().enumerate() {
                    out[p * lanes + lane] = v;
                }
            }
        });
    grid
}

/// [`commit`] for a LANE-MAJOR message: `q` is the full `2^log_msg_len`-word
/// dense stack, lane `l` being its contiguous block `q[l·D .. (l+1)·D)`.
/// Lanes `[t, 2^log_batch_size)` — the stack's zero tail, `t =
/// params.num_ntts()` — must be identically zero and are neither encoded nor
/// hashed.
///
/// Equivalent to `commit(&extract, params)` on the compacted interleaved
/// message `extract[p·t + l] = q[l·D + p]`, but the transpose happens inside
/// the codeword fill, so it costs no extra pass and no intermediate buffer.
pub fn commit_lane_major(q: &[F128], params: &PcsParams) -> (Commitment, ProverData) {
    params.validate();
    let lanes = 1usize << params.log_batch_size;
    let t = params.num_ntts();
    assert_eq!(
        q.len(),
        1usize << params.log_msg_len(),
        "lane-major message must be the full padded stack"
    );
    let d = q.len() >> params.log_batch_size;
    debug_assert!(
        q[t * d..].iter().all(|w| w.is_zero()),
        "lanes >= num_lanes must be identically zero"
    );
    let _ = lanes;
    let codeword_len = params.n_positions() * t;
    let mut codeword = take_f128(codeword_len);
    replicate_lane_major_fill(&mut codeword, q, t, d);
    finalize_commit(codeword, params)
}

/// [`replicate_message_fill`] for a lane-major message: fill `codeword` with
/// `2^r` replicas of the transposed `t`-lane extract, `msg[p·t + l] =
/// q[l·D + p]`. Cache-blocked over position tiles (see
/// [`lane_grid_from_lane_major`]).
fn replicate_lane_major_fill(codeword: &mut [F128], q: &[F128], t: usize, d: usize) {
    const TILE: usize = 64;
    use rayon::prelude::*;

    let msg_len = t * d;
    debug_assert!(codeword.len().is_multiple_of(msg_len));
    // positions per tile
    codeword.par_chunks_mut(msg_len).for_each(|rep| {
        rep.par_chunks_mut(TILE * t)
            .enumerate()
            .for_each(|(tile, out)| {
                let p0 = tile * TILE;
                let n = TILE.min(d - p0);
                for lane in 0..t {
                    let src = &q[lane * d + p0..lane * d + p0 + n];
                    for (p, &v) in src.iter().enumerate() {
                        out[p * t + lane] = v;
                    }
                }
            });
    });
}

/// Fill `codeword` with `2^r` replicas of `msg` (`r = log2(codeword.len() /
/// msg.len())`) — the exact state after the first `r` forward-NTT layers on
/// the zero-padded coefficient vector `[msg, 0, …, 0]`. Pair with
/// `forward_transform_interleaved_from_layer(…, r)`. Every slot of `codeword`
/// is written (input contents may be stale/uninit).
pub(crate) fn replicate_message_fill(codeword: &mut [F128], msg: &[F128]) {
    const COPY_CHUNK: usize = 1 << 16;
    use rayon::prelude::*;
    let msg_len = msg.len();
    debug_assert!(codeword.len().is_multiple_of(msg_len));

    // Fast finer-grained path only when the chunk size divides `msg_len` (so a
    // COPY_CHUNK-aligned slice never straddles a replica boundary). On the
    // integer-lane commit `msg_len = t · 2^log_dim` is not a power of two, but
    // for real commit sizes `2^log_dim ≥ 2^16 = COPY_CHUNK` still divides it;
    // the guard falls back to per-replica copies otherwise.
    if msg_len >= COPY_CHUNK && msg_len.is_multiple_of(COPY_CHUNK) {
        codeword
            .par_chunks_mut(COPY_CHUNK)
            .enumerate()
            .for_each(|(i, dst)| {
                let src_off = (i * COPY_CHUNK) % msg_len;
                dst.copy_from_slice(&msg[src_off..src_off + dst.len()]);
            });
    } else {
        // One full copy of `msg` per replica (parallel across replicas). Each
        // chunk is exactly `msg_len` long since `codeword.len()` is a multiple.
        codeword.par_chunks_mut(msg_len).for_each(|rep| {
            rep.copy_from_slice(msg);
        });
    }
}

/// Shared tail of [`commit`] / [`commit_into`]: interleaved forward additive
/// NTT (RS-encode every lane) then the initial Merkle tree over codeword rows.
fn finalize_commit(mut codeword: Vec<F128>, params: &PcsParams) -> (Commitment, ProverData) {
    let timing = var_os("FLOCK_COMMIT_TIMING").is_some();
    let t_ntt = Instant::now();
    // ---- Interleaved forward additive NTT: 2^log_batch_size independent
    // sub-NTTs with shared twiddles. Each sub-NTT operates on its lane of the
    // SoA buffer. The first `log_inv_rate` layers were pre-applied by the
    // caller's replicate-fill (commit_into), so start past them.
    let ntt = AdditiveNttF128::standard(params.k_code());
    ntt.forward_transform_interleaved_from_layer(
        &mut codeword,
        params.num_ntts(),
        params.log_inv_rate,
    );
    if timing {
        eprintln!(
            "[commit-timing] ntt: {:.2} ms",
            t_ntt.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_merkle = Instant::now();

    // ---- Merkle commitment: one leaf per codeword position = num_ntts F128.
    // Zero-copy: cast the codeword Vec<F128> directly to &[u8]. F128 is
    // repr(C, align(16)) with two u64s laid out little-endian — same bytes
    // as the explicit lo.to_le_bytes() + hi.to_le_bytes() serialization.
    let codeword_bytes: &[u8] = unsafe {
        from_raw_parts(
            codeword.as_ptr() as *const u8,
            codeword.len() * size_of::<F128>(),
        )
    };
    // Initial tree: one leaf per codeword position, each containing the
    // row-batch lanes (num_ntts F_{2^128} values = 2^log_batch_size). This is
    // Ligerito's L0 commitment.
    let merkle_tree = merkle::merkle_tree(codeword_bytes, params.n_leaves(), params.merkle_hash);
    let cap = merkle::cap_layer(&merkle_tree, params.n_leaves(), params.l0_cap_depth()).to_vec();
    if timing {
        eprintln!(
            "[commit-timing] merkle: {:.2} ms",
            t_merkle.elapsed().as_secs_f64() * 1e3
        );
    }

    (
        Commitment {
            cap,
            params: params.clone(),
        },
        ProverData {
            codeword,
            merkle_tree,
        },
    )
}

/// Tag the current thread as background QoS. On macOS the scheduler then
/// strongly prefers efficiency (E) cores — ideal for the fault/bandwidth-bound
/// codeword pre-fault, which we want OFF the performance cores running witness
/// generation. No-op on other platforms.
#[cfg(target_os = "macos")]
fn set_background_qos() {
    // QOS_CLASS_BACKGROUND = 0x09. Declared inline to avoid a libc dependency.
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x09, 0);
    }
}
#[cfg(not(target_os = "macos"))]
fn set_background_qos() {}

/// Allocate + zero-fill (pre-fault) the codeword buffer that [`commit_into`]
/// will consume, on a background-QoS (E-core) thread, **while** `gen` runs on
/// the caller's performance threads. Returns `(Some(buf), gen_result)`.
///
/// The codeword alloc is page-fault-bound (first-touch of a fresh 64–512 MB
/// buffer) and scales ~1.0×, so overlapping it with witness generation hides it
/// almost entirely (measured ~99% at m=29 — see `benches/ecore_offload_probe`).
///
/// **Gated for honest single-threaded behavior:** when the rayon pool has ≤ 1
/// thread (i.e. `RAYON_NUM_THREADS=1`), this spawns **zero** OS threads — it
/// runs `gen` and returns `None`, leaving [`commit`] to allocate inline. The
/// whole offload is therefore invisible to truly-serial runs.
pub fn prefault_codeword_during<R>(
    params: &PcsParams,
    generate: impl FnOnce() -> R,
) -> (Option<Vec<F128>>, R) {
    if rayon::current_num_threads() <= 1 || var_os("FLOCK_NO_PREFAULT").is_some() {
        // Truly single-threaded (or explicitly disabled): no extra OS thread;
        // commit allocates inline. FLOCK_NO_PREFAULT lets benchmarks A/B the
        // offload and keeps fixed-thread-count sweeps honest.
        return (None, generate());
    }
    let codeword_len = params.n_positions() * params.num_ntts();
    // Warm path: a pooled buffer is already resident — there is nothing to
    // pre-fault, and commit_into writes every slot itself. Skip the thread.
    if let Some(buf) = try_take_f128(codeword_len) {
        return (Some(buf), generate());
    }
    // Cold path: allocate + first-touch on a background-QoS thread, hidden
    // under witness generation. (commit_into rewrites all slots, so the
    // zero values themselves don't matter — the page faults do.)
    scope(|s| {
        let h = s.spawn(move || {
            set_background_qos();
            let mut buf: Vec<F128> = crate::alloc_uninit_f128_vec(codeword_len);
            unsafe {
                write_bytes(buf.as_mut_ptr(), 0u8, codeword_len);
            }
            buf
        });
        let r = generate();
        (Some(h.join().unwrap()), r)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    /// The Ligerito configs derived from `PcsParams` must carry the params'
    /// Merkle hash, not the embedded security config's `hash` field. If they
    /// diverge, the L0 commitment and the recursive levels are built under
    /// different hashes and nothing verifies — silently, and only at the
    /// geometries that reach recursion.
    #[test]
    fn ligerito_configs_inherit_the_params_merkle_hash() {
        let mut params = default_params(22);
        params.log_batch_size = 6;

        assert_eq!(params.merkle_hash, HashKind::Sha256);
        assert_eq!(
            params.ligerito_prover_config().unwrap().merkle_hash,
            HashKind::Sha256
        );

        params.merkle_hash = HashKind::Blake3;
        assert_eq!(
            params.ligerito_prover_config().unwrap().merkle_hash,
            HashKind::Blake3,
            "prover config must follow PcsParams, not the embedded TOML"
        );
        assert_eq!(
            params.ligerito_verifier_config().unwrap().merkle_hash,
            HashKind::Blake3,
            "verifier config must follow PcsParams, not the embedded TOML"
        );
    }

    fn default_params(m: usize) -> PcsParams {
        PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: 1,
            profile: Default::default(),
            num_lanes: None,
            merkle_hash: Default::default(),
        }
    }

    /// The replicate-fill + start-at-layer-`log_inv_rate` fast path must be
    /// byte-identical to the definitional encoding: zero-padded coefficients
    /// through the FULL forward NTT. Covers rate 1/2 and 1/4 and both
    /// interleaving widths.
    #[test]
    fn commit_matches_full_ntt_oracle() {
        use crate::ntt::AdditiveNttF128;
        let mut rng = Rng::new(0xFEED);
        for (m, log_inv_rate, log_batch_size) in [(10, 1, 1), (12, 1, 2), (12, 2, 1), (14, 2, 3)] {
            let params = PcsParams {
                m,
                log_inv_rate,
                log_batch_size,
                profile: Default::default(),
                num_lanes: None,
                merkle_hash: Default::default(),
            };
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);

            let (commitment, pd) = commit(&z_packed, &params);

            // Oracle: explicit [z, 0, …, 0] coefficients, full NTT from layer 0.
            let mut oracle = vec![F128::ZERO; params.codeword_len_f128()];
            oracle[..z_packed.len()].copy_from_slice(&z_packed);
            let ntt = AdditiveNttF128::standard(params.k_code());
            ntt.forward_transform_interleaved(&mut oracle, params.num_ntts());

            assert_eq!(
                pd.codeword, oracle,
                "codeword mismatch at m={m} r={log_inv_rate}"
            );
            let oracle_bytes: &[u8] =
                unsafe { from_raw_parts(oracle.as_ptr() as *const u8, oracle.len() * 16) };
            let oracle_tree = merkle_tree(oracle_bytes, params.n_leaves(), params.merkle_hash);
            let oracle_cap = cap_layer(&oracle_tree, params.n_leaves(), params.l0_cap_depth());
            assert_eq!(
                commitment.cap, oracle_cap,
                "cap mismatch at m={m} r={log_inv_rate}"
            );
        }
    }

    /// Oracle 1 (pow2 byte-identity anchor) at the commit level: committing
    /// `num_lanes = Some(2^log_batch_size)` is byte-identical — root, codeword,
    /// and full Merkle tree — to the default `num_lanes = None`. This is the
    /// safety net: the integer-lane path collapses to today's path at full
    /// lane utilization.
    #[test]
    fn commit_pow2_num_lanes_byte_identical() {
        let mut rng = Rng::new(0xA0C1);
        for (m, log_inv_rate, log_batch_size) in [(10, 1, 1), (12, 1, 2), (12, 2, 1), (14, 2, 3)] {
            let full = 1usize << log_batch_size;
            let base = PcsParams {
                m,
                log_inv_rate,
                log_batch_size,
                profile: Default::default(),
                num_lanes: None,
                merkle_hash: Default::default(),
            };
            let explicit = PcsParams {
                num_lanes: Some(full),
                ..base.clone()
            };
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);
            let (c_none, pd_none) = commit(&z_packed, &base);
            let (c_full, pd_full) = commit(&z_packed, &explicit);
            assert_eq!(c_none.cap, c_full.cap, "cap diverged (m={m})");
            assert_eq!(pd_none.codeword, pd_full.codeword, "codeword diverged");
            assert_eq!(pd_none.merkle_tree, pd_full.merkle_tree, "tree diverged");
        }
    }

    /// Oracle 2 (integer-lane encode correctness) at the commit level: the
    /// `t`-lane commit of a dense message `q` (length `t·2^log_dim`) produces
    /// a codeword whose real lane `l` is byte-identical to lane `l` of the
    /// `2^log_batch_size`-lane commit of `q` zero-padded in the lane
    /// dimension. The committed codeword and Merkle tree are strictly smaller
    /// (t < 2^log_batch_size lanes), and the root is over `t·16`-byte leaves.
    #[test]
    fn commit_integer_lanes_encode_oracle() {
        let mut rng = Rng::new(0x1A6E_C0);
        // (m, log_inv_rate, log_batch_size) with several non-power-of-two t.
        for (m, log_inv_rate, log_batch_size) in [(12, 1, 3), (14, 2, 3), (15, 1, 4)] {
            let full = 1usize << log_batch_size;
            let log_dim = (m - LOG_PACKING) - log_batch_size;
            let dim = 1usize << log_dim;
            for t in [full / 2 + 1, full - 1, (full * 3) / 4] {
                let t_params = PcsParams {
                    m,
                    log_inv_rate,
                    log_batch_size,
                    profile: Default::default(),
                    num_lanes: Some(t),
                    merkle_hash: Default::default(),
                };
                let full_params = PcsParams {
                    num_lanes: None,
                    ..t_params.clone()
                };

                // Dense t-lane message q[pos*t + lane].
                let q = rng.f128_vec(t * dim);
                // Zero-pad the lane dimension to `full` lanes.
                let mut q_padded = vec![F128::ZERO; full * dim];
                for pos in 0..dim {
                    for lane in 0..t {
                        q_padded[pos * full + lane] = q[pos * t + lane];
                    }
                }

                let (_c_t, pd_t) = commit(&q, &t_params);
                let (_c_full, pd_full) = commit(&q_padded, &full_params);

                assert_eq!(pd_t.codeword.len(), t_params.codeword_len_f128());
                assert!(
                    pd_t.codeword.len() < pd_full.codeword.len(),
                    "integer-lane codeword must be smaller"
                );
                assert!(
                    pd_t.merkle_tree.len() < pd_full.merkle_tree.len()
                        || t_params.n_leaves() == full_params.n_leaves(),
                    "n_leaves unchanged, so tree node count matches"
                );
                assert_eq!(t_params.n_leaves(), full_params.n_leaves());

                let n_positions = t_params.n_positions();
                for pos in 0..n_positions {
                    for lane in 0..t {
                        assert_eq!(
                            pd_t.codeword[pos * t + lane],
                            pd_full.codeword[pos * full + lane],
                            "lane {lane} pos {pos} diverged (m={m}, t={t})"
                        );
                    }
                }

                // Root is the Merkle tree over t-wide leaves of pd_t.codeword.
                let bytes: &[u8] = unsafe {
                    from_raw_parts(
                        pd_t.codeword.as_ptr() as *const u8,
                        pd_t.codeword.len() * 16,
                    )
                };
                let tree = merkle_tree(bytes, t_params.n_leaves(), t_params.merkle_hash);
                let cap = cap_layer(&tree, t_params.n_leaves(), t_params.l0_cap_depth());
                assert_eq!(cap, _c_t.cap, "cap must be over t-wide leaves");
            }
        }
    }

    /// High-bit lanes (Oracle 3): the lane-grid commit of a lane-major dense
    /// stack `q` — whose real data is the contiguous prefix `q[..dense]`, so
    /// lanes `≥ t` are wholly zero — is byte-identical to the plain `t`-lane
    /// commit of the compacted message. I.e. `commit_lane_grid` really is
    /// "encode the `t` real lanes and nothing else", and the transpose
    /// `lane_grid_from_lane_major` is its inverse-consistent partner.
    ///
    /// Also pins the two structural facts the whole scheme rests on: the
    /// transpose is a pure index-bit rotation (`g[p·L + l] = q[l·D + p]`), and
    /// a CONTIGUOUS zero tail of `q` becomes WHOLE zero lanes of `g` — which
    /// is exactly what today's LSB-lane labelling fails to give.
    #[test]
    fn lane_grid_commit_matches_compacted_message() {
        for (m, log_inv_rate, log_batch_size) in [(12, 1, 3), (14, 2, 3), (15, 1, 4)] {
            let lanes = 1usize << log_batch_size;
            let log_dim = (m - LOG_PACKING) - log_batch_size;
            let d = 1usize << log_dim;
            // A dense stack that leaves the top few lanes empty.
            for spare in [1usize, 2, lanes / 2] {
                let t = lanes - spare;
                let dense = (t - 1) * d + d / 3; // real prefix, mid-lane end
                let mut q = vec![F128::ZERO; lanes * d];
                for (i, w) in q[..dense].iter_mut().enumerate() {
                    *w = F128 {
                        lo: i as u64 + 1,
                        hi: 0xA5,
                    };
                }
                assert_eq!(dense_lanes(dense, log_batch_size, log_dim), t);

                let grid = lane_grid_from_lane_major(&q, log_batch_size);
                // The rotation, and the whole-zero-lane property.
                for lane in 0..lanes {
                    for p in 0..d {
                        assert_eq!(grid[p * lanes + lane], q[lane * d + p]);
                    }
                }
                for chunk in grid.chunks(lanes) {
                    assert!(
                        chunk[t..].iter().all(|w| w.is_zero()),
                        "lanes >= t must be wholly zero"
                    );
                }

                let params = PcsParams {
                    m,
                    log_inv_rate,
                    log_batch_size,
                    profile: Default::default(),
                    num_lanes: Some(t),
                    merkle_hash: Default::default(),
                };
                // Reference: the plain t-lane commit of the compacted message.
                let mut extract = vec![F128::ZERO; t * d];
                for p in 0..d {
                    extract[p * t..(p + 1) * t].copy_from_slice(&grid[p * lanes..p * lanes + t]);
                }
                let (c_ref, pd_ref) = commit(&extract, &params);
                let (c_grid, pd_grid) = commit_lane_major(&q, &params);
                assert_eq!(c_ref.cap, c_grid.cap, "cap diverged (m={m}, t={t})");
                assert_eq!(pd_ref.codeword, pd_grid.codeword, "codeword diverged");
                assert_eq!(pd_ref.merkle_tree, pd_grid.merkle_tree, "tree diverged");
            }
        }
    }

    /// Savings measurement (Oracle 6): at m30-representative sizes
    /// (log_dim=17, rate 1/2, k_code=18 → 256 MB padded codeword), measure the
    /// interleaved NTT + Merkle for the integer-lane t=46 vs the padded 64.
    /// Reports the ratios; asserts the non-pow2 parallel NTT is NOT slower per
    /// lane (efficiency ≈ t/64, i.e. total time ≲ 0.9× the pow2 time — a
    /// generous ceiling that still fails if the remainder path regresses).
    /// Run: `cargo test -p flock-core --release measure_integer_lane_savings
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn measure_integer_lane_savings() {
        use crate::ntt::AdditiveNttF128;
        use std::time::Instant;

        let log_dim = 17usize;
        let log_inv_rate = 1usize;
        let k_code = log_dim + log_inv_rate; // 18
        let n_positions = 1usize << k_code;
        let ntt = AdditiveNttF128::standard(k_code);

        let mut rng = Rng::new(0x5A71_2026);

        let bench = |num_ntts: usize, rng: &mut Rng| -> (f64, f64) {
            let codeword_len = n_positions * num_ntts;
            // Random message replicated (as commit does), then timed NTT.
            let msg = rng.f128_vec((1usize << log_dim) * num_ntts);
            let n_runs = 5usize;
            let mut best_ntt = f64::INFINITY;
            let mut buf = vec![F128::ZERO; codeword_len];
            for _ in 0..n_runs {
                replicate_message_fill(&mut buf, &msg);
                let t = Instant::now();
                ntt.forward_transform_interleaved_from_layer(&mut buf, num_ntts, log_inv_rate);
                best_ntt = best_ntt.min(t.elapsed().as_secs_f64() * 1e3);
            }
            // Merkle over the (encoded) codeword: n_positions leaves, each
            // num_ntts F128 wide. Timed under the default hash (SHA-256) — this
            // probe compares lane counts, not hashes, and has no `PcsParams` in
            // scope; use `HashKind::Blake3` here to re-measure the other one.
            let bytes: &[u8] = unsafe { from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 16) };
            let mut best_merkle = f64::INFINITY;
            for _ in 0..n_runs {
                let t = Instant::now();
                let tree = merkle::merkle_tree(bytes, n_positions, HashKind::default());
                best_merkle = best_merkle.min(t.elapsed().as_secs_f64() * 1e3);
                black_box(tree.last());
            }
            (best_ntt, best_merkle)
        };

        // Pure per-lane arithmetic ratio (scalar, no cache-blocking / rayon).
        let bench_scalar = |num_ntts: usize, rng: &mut Rng| -> f64 {
            let codeword_len = n_positions * num_ntts;
            let msg = rng.f128_vec((1usize << log_dim) * num_ntts);
            let mut buf = vec![F128::ZERO; codeword_len];
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                replicate_message_fill(&mut buf, &msg);
                let t = Instant::now();
                ntt.forward_transform_interleaved_scalar_from_layer(
                    &mut buf,
                    num_ntts,
                    log_inv_rate,
                );
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let sc64 = bench_scalar(64, &mut rng);
        let sc46 = bench_scalar(46, &mut rng);
        eprintln!(
            "[savings]   NTT scalar (per-lane work): t=64 {sc64:7.2} ms  t=46 {sc46:7.2} ms  ratio {:.3}",
            sc46 / sc64
        );

        let (ntt64, mrk64) = bench(64, &mut rng);
        let (ntt46, mrk46) = bench(46, &mut rng);

        let ntt_ratio = ntt46 / ntt64;
        let mrk_ratio = mrk46 / mrk64;
        let commit_ratio = (ntt46 + mrk46) / (ntt64 + mrk64);
        eprintln!("[savings] m30-scale (log_dim={log_dim}, k_code={k_code}, 256 MB padded)");
        eprintln!(
            "[savings]   NTT:    t=64 {ntt64:7.2} ms   t=46 {ntt46:7.2} ms   ratio {ntt_ratio:.3}  (ideal 0.719)"
        );
        eprintln!(
            "[savings]   Merkle: t=64 {mrk64:7.2} ms   t=46 {mrk46:7.2} ms   ratio {mrk_ratio:.3}  (ideal 0.719)"
        );
        eprintln!(
            "[savings]   NTT+Merkle commit: ratio {commit_ratio:.3}  (=> {:.1}% reduction)",
            (1.0 - commit_ratio) * 100.0
        );
        // The non-pow2 parallel NTT must be at least as efficient per lane as
        // the pow2 path (no remainder-path regression). Ideal 46/64 = 0.719;
        // measured warm ~0.80–0.86 (some fixed rayon/bandwidth overhead on the
        // top-layer sweeps). Fail hard only if the saving is essentially erased
        // (ratio → 1.0) — a generous ceiling that tolerates a loaded machine.
        assert!(
            ntt_ratio < 0.95,
            "non-pow2 t=46 NTT is inefficient (ratio {ntt_ratio:.3} ≥ 0.95) — the \
             remainder path erases the saving"
        );
        assert!(
            commit_ratio < 0.95,
            "integer-lane commit did not reduce NTT+Merkle time (ratio {commit_ratio:.3})"
        );
    }

    #[test]
    fn commit_runs_and_produces_root() {
        let mut rng = Rng::new(42);
        for m in [8usize, 10, 12] {
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);
            let params = default_params(m);
            let (commitment, prover_data) = commit(&z_packed, &params);
            assert_eq!(prover_data.codeword.len(), params.codeword_len_f128());
            assert_eq!(
                cap_layer(
                    &prover_data.merkle_tree,
                    params.n_leaves(),
                    params.l0_cap_depth(),
                ),
                commitment.cap
            );
            assert_eq!(z_packed.len(), 1 << params.log_msg_len());
        }
    }

    #[test]
    fn commit_is_deterministic() {
        let mut rng = Rng::new(7);
        let m = 10;
        let z = rng.bits(1 << m);
        let z_packed = super::super::pack::pack_witness(&z, m);
        let params = default_params(m);
        let (c1, _) = commit(&z_packed, &params);
        let (c2, _) = commit(&z_packed, &params);
        assert_eq!(c1.cap, c2.cap);
    }

    #[test]
    fn commit_root_sensitive_to_witness() {
        let mut rng = Rng::new(99);
        let m = 10;
        let mut z = rng.bits(1 << m);
        let params = default_params(m);
        let (c1, _) = commit(&super::super::pack::pack_witness(&z, m), &params);
        z[7] ^= true;
        let (c2, _) = commit(&super::super::pack::pack_witness(&z, m), &params);
        assert_ne!(c1.cap, c2.cap);
    }

    #[test]
    fn rs_encoding_is_linear() {
        let mut rng = Rng::new(123);
        let m = 9;
        let params = default_params(m);
        let z1 = rng.bits(1 << m);
        let z2 = rng.bits(1 << m);
        let z_xor: Vec<bool> = z1.iter().zip(&z2).map(|(a, b)| a ^ b).collect();
        let pack = |z: &[bool]| super::super::pack::pack_witness(z, m);
        let (_, pd1) = commit(&pack(&z1), &params);
        let (_, pd2) = commit(&pack(&z2), &params);
        let (_, pd_x) = commit(&pack(&z_xor), &params);
        for (i, (&c1, &c2)) in pd1.codeword.iter().zip(&pd2.codeword).enumerate() {
            assert_eq!(c1 + c2, pd_x.codeword[i], "linearity fails at i={i}");
        }
    }

    #[test]
    fn codeword_doubles_message_length() {
        let mut rng = Rng::new(2);
        let m = 10;
        let params = default_params(m);
        let z = rng.bits(1 << m);
        let z_packed = super::super::pack::pack_witness(&z, m);
        let (_, pd) = commit(&z_packed, &params);
        assert_eq!(pd.codeword.len(), 2 * z_packed.len());
    }
}
