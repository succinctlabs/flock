//! **MVP-3: PCS query openings as circuit gates, and the arithmetic on them.**
//!
//! A recursive verifier's dominant cost is checking PCS query openings — 218
//! depth-13 Merkle paths over 1 KiB leaves, per the L0 measurement. This file
//! makes one such opening a [`GateType`], so a circuit can instantiate as many
//! as it needs and the builder produces the wiring and the witness together.
//!
//! Three things are being validated, and they are the three that were open:
//!
//! - **The sibling path travels as a [`GateType::Hint`].** It is not a schema
//!   word and never can be: a sibling sits at
//!   [`sibling_bit`](MerkleTreeLayout::sibling_bit), inside its level's
//!   base-block padding at an unaligned offset, and a wire carries a whole
//!   128-bit word. It needs no wire — no other gate reads it, and the relation
//!   binds it through the root the schema does export.
//!
//! - **The index word is wireable, and is wired to a hash output.** This is
//!   the Fiat–Shamir query binding in miniature: a BLAKE3 gate produces a
//!   challenge word, that word is connected to the opening's index, and the
//!   masking `sample_queries` does natively (`& (block_len − 1)`) costs no
//!   gadget at all — it is which columns the Merkle relation reads. The word
//!   alignment this needs is why the index moved.
//!
//! - **A boolean slot with a hint proves and verifies** against the real
//!   union prover, over the walker lincheck circuit rather than materialized
//!   matrices.
//!
//! MVP-3b adds the other half — `LeafEvalGate`, an ELEMENT gate consuming the
//! same leaf words to compute `Σ_i α_i · ⟨row_i, eq(v, ·)⟩`, the `enforced_sum`
//! the Ligerito verifier checks. The 64 leaf words are one wire class each with
//! cells in both slots, so the copy constraint is what makes "the leaf that is
//! in the tree" and "the leaf the arithmetic ran on" the same leaf — across the
//! boolean/element class boundary.
//!
//! Small shapes where the geometry allows; `l0_shape_circuit_cost` runs the
//! real one (218 openings, depth 13, 1 KiB leaves).

use flock_core::challenger::pow_has_leading_zero_bits;
use flock_core::circuit::builder::SlotId;
use flock_core::circuit::builder::{CircuitBuilder, GateType, ShapeBuilder, SlotWitness, Wire};
use flock_core::circuit::wire_cells;
use flock_core::element_r1cs::ElementTableType;
use flock_core::field::F128;
use flock_core::lincheck::LincheckCircuit;
#[cfg(test)]
use flock_core::lincheck::build_eq_table;
use flock_core::merkle::{self as core_merkle, HashKind};
use flock_core::pcs::PcsParams;
use flock_core::pcs::jagged::JaggedParams;
use flock_core::pcs::jagged::assist_boundaries;
use flock_core::pcs::jagged::assist_sparse_transitions;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::r1cs::BlockR1cs;
use flock_core::transcript_record::Stream;
use flock_core::transcript_record::StreamWord;
use flock_core::transcript_record::TranscriptOp;
use flock_core::verifier;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::blake3;
use flock_prover::r1cs_hashes::fs_chain::FsChainTrace;
use flock_prover::r1cs_hashes::fs_chain::IV as FS_CHAIN_IV;
use flock_prover::r1cs_hashes::merkle_r1cs::{
    ChunkPathInput, MerkleTreeLayout, SLOT_WORDS, blake3_spec,
};
use flock_prover::schedule::Registry;
use flock_prover::schedule::TableType;
use flock_prover::union::UnionInstance;
use std::any::Any;
use std::array::from_fn;
use std::env::var;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;
use std::hint::black_box;
use std::iter::repeat_n;
use std::sync::Arc;
use std::time::Instant;

const DOMAIN: &[u8] = b"flock-circuit-merkle-v0";

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;

const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

fn pack4(w: [u32; 4]) -> F128 {
    F128::new(
        w[0] as u64 | ((w[1] as u64) << 32),
        w[2] as u64 | ((w[3] as u64) << 32),
    )
}

fn unpack4(v: F128) -> [u32; 4] {
    [
        v.lo as u32,
        (v.lo >> 32) as u32,
        v.hi as u32,
        (v.hi >> 32) as u32,
    ]
}

fn pack8(w: &[u32; 8]) -> [F128; 2] {
    [
        pack4([w[0], w[1], w[2], w[3]]),
        pack4([w[4], w[5], w[6], w[7]]),
    ]
}

fn pack_params(counter: u64, block_len: u32, flags: u32) -> F128 {
    F128::new(counter, block_len as u64 | ((flags as u64) << 32))
}

fn unpack_params(v: F128) -> (u64, u32, u32) {
    (v.lo, v.hi as u32, (v.hi >> 32) as u32)
}

/// A 128-bit word of leaf data: bytes `[o, o+16)` little-endian, which is
/// exactly how the message region reads them (`leaf_msg_words` is LE `u32`s,
/// and committed bit `t` of a word is bit `t` of `lo`).
fn leaf_word(data: &[u8], o: usize) -> F128 {
    F128::new(
        u64::from_le_bytes(data[o..o + 8].try_into().unwrap()),
        u64::from_le_bytes(data[o + 8..o + 16].try_into().unwrap()),
    )
}

fn unpack8(a: F128, b: F128) -> [u32; SLOT_WORDS] {
    let (x, y) = (unpack4(a), unpack4(b));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}

fn digest_words(d: &[u32; SLOT_WORDS]) -> [F128; 2] {
    [
        pack4([d[0], d[1], d[2], d[3]]),
        pack4([d[4], d[5], d[6], d[7]]),
    ]
}

fn hash_to_digest(h: &[u8; 32]) -> [u32; SLOT_WORDS] {
    from_fn(|w| u32::from_le_bytes(h[4 * w..4 * w + 4].try_into().unwrap()))
}

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
}

// ---------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------

/// One BLAKE3 compression, the challenge source. (Same gate as
/// `circuit_builder.rs`; duplicated rather than shared because these are
/// separate test binaries.)
struct Blake3Gate {
    nu: usize,
}

impl GateType for Blake3Gate {
    type Row = blake3::Compression;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&blake3::build_block_r1cs(self.nu))
            .with_io_schema(blake3::io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        let cv: [u32; 8] = {
            let (a, b) = (unpack4(inputs[0]), unpack4(inputs[1]));
            [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        };
        let mut m = [0u32; 16];
        for i in 0..4 {
            m[4 * i..4 * i + 4].copy_from_slice(&unpack4(inputs[2 + i]));
        }
        let (counter, block_len, flags) = unpack_params(inputs[6]);
        let out = blake3::blake3_compress(&cv, &m, counter, block_len, flags);
        let lo = pack8(&out[0..8].try_into().unwrap());
        let hi = pack8(&out[8..16].try_into().unwrap());
        (
            vec![lo[0], lo[1], hi[0], hi[1]],
            (cv, m, counter, block_len, flags),
        )
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// One chunk-leaf Merkle opening: leaf data and an index word in, the root
/// out, the sibling path as a hint.
struct MerklePathGate {
    layout: MerkleTreeLayout,
    nu: usize,
}

impl MerklePathGate {
    /// `block_len` is the PCS block length the opening's index will be
    /// sampled against — `sample_queries` masks a challenge with
    /// `block_len − 1`, and the relation reads the index word's low `depth`
    /// bits, so the two agree only when `depth = log2(block_len)`. Asserting
    /// it here means a circuit cannot silently wire a challenge that the
    /// relation truncates differently than the sampler did.
    ///
    /// Real-protocol paths are CAPPED since Merkle capping landed: they are
    /// `d − c` deep and the index's high `c` bits select a node of the
    /// absorbed cap layer rather than folding to a root. The COLLAPSED path
    /// models this (`emit_opening` + the boundary select in `mvp6`): the
    /// select is done by the checker on published words, so no mux gadget
    /// exists in-circuit. This COMPOSITE gate stays full-depth against its
    /// synthetic trees, where the assert above IS the index-binding
    /// argument; it is the uncapped differential oracle, not the protocol.
    fn new(depth: usize, leaf_bytes: usize, nu: usize, block_len: usize) -> Self {
        assert!(
            block_len.is_power_of_two() && block_len.trailing_zeros() as usize == depth,
            "tree depth {depth} does not match block_len {block_len}: the index \
             word's low {depth} bits are not the query sample_queries picked"
        );
        Self {
            layout: MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec()),
            nu,
        }
    }
}

impl GateType for MerklePathGate {
    type Row = ChunkPathInput;
    /// The sibling path, level 0 closest to the leaf. Unwireable by
    /// construction — see the module docs.
    type Hint = Vec<[u32; SLOT_WORDS]>;

    fn table(&self) -> TableType {
        // Stub matrices: the constraints live in the walker, as on the
        // production path. The digest binds `k_log`, `useful_bits` and the
        // const pin — hence the depth — and the verifier builds the matching
        // walker out of band.
        TableType::from_block_r1cs(&self.layout.build_block_r1cs_stub(self.nu))
            .with_io_schema(self.layout.io_schema())
    }

    fn eval(&self, inputs: &[F128], hint: &Self::Hint) -> (Vec<F128>, Self::Row) {
        assert_eq!(hint.len(), self.layout.depth, "one sibling per level");
        // Schema In-order: 4 words per chunk block, then the index word.
        let mut leaf_data = Vec::with_capacity(64 * self.layout.leaf_blocks);
        for w in &inputs[..4 * self.layout.leaf_blocks] {
            leaf_data.extend_from_slice(&w.lo.to_le_bytes());
            leaf_data.extend_from_slice(&w.hi.to_le_bytes());
        }
        let index_word = inputs[4 * self.layout.leaf_blocks];
        let row = ChunkPathInput {
            leaf_data,
            // The WHOLE word. The low `depth` bits are the position; the rest
            // ride along committed, pinned by the copy constraint, and read by
            // no row of the relation.
            index: (index_word.lo as u128) | ((index_word.hi as u128) << 64),
            siblings: hint.clone(),
        };
        let root = self.layout.root_chunk(&row);
        (digest_words(&root).to_vec(), row)
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

// ---------------------------------------------------------------------------

/// A tree, and one opening's siblings out of it.
struct Tree {
    data: Vec<u8>,
    flat: Vec<[u8; 32]>,
    root: [u8; 32],
    depth: usize,
    leaf_bytes: usize,
}

impl Tree {
    fn new(depth: usize, leaf_bytes: usize, rng: &mut Rng) -> Self {
        let n_leaves = 1usize << depth;
        let data: Vec<u8> = (0..n_leaves * leaf_bytes)
            .map(|_| rng.next_u32() as u8)
            .collect();
        let flat = core_merkle::merkle_tree(&data, n_leaves, HashKind::Blake3);
        let root = flat[flat.len() - 1];
        Self {
            data,
            flat,
            root,
            depth,
            leaf_bytes,
        }
    }

    fn leaf(&self, pos: usize) -> &[u8] {
        &self.data[pos * self.leaf_bytes..(pos + 1) * self.leaf_bytes]
    }

    fn siblings(&self, pos: usize) -> Vec<[u32; SLOT_WORDS]> {
        let mut out = Vec::with_capacity(self.depth);
        let (mut seg, mut width, mut idx) = (0usize, 1usize << self.depth, pos);
        for _ in 0..self.depth {
            out.push(hash_to_digest(&self.flat[seg + (idx ^ 1)]));
            seg += width;
            width /= 2;
            idx >>= 1;
        }
        out
    }
}

/// The table's index and the tree position are THE SAME NUMBER (`0776f64`
/// flipped the swap gadget's polarity to make it so). Kept as a named function
/// because it is the identity that has to hold for a Fiat-Shamir challenge to
/// be wireable straight into the index: `sample_queries` masks the challenge
/// to a position, so the circuit must open that position and not its
/// complement.
fn table_index(pos: usize, _depth: usize) -> u128 {
    pos as u128
}

/// **The gate, standalone**: the builder's rows reproduce the openings, the
/// computed roots are the real tree root, and the sibling path reached `eval`
/// as a hint without ever being a wire.
#[test]
fn merkle_openings_through_the_builder() {
    let (depth, leaf_bytes, nu) = (2usize, 128usize, 6usize);
    let mut rng = Rng(0x_3E_11_5E_ED);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    let mut b = CircuitBuilder::new(nu);
    let g = b.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));

    // Every opening's inputs first, then every root — `public_value` and
    // `publish` share one public vector, so keeping the publishes last is what
    // makes the roots the trailing `2n` entries.
    let positions: Vec<usize> = (0..1usize << depth).collect();
    let roots: Vec<Vec<Wire>> = positions
        .iter()
        .map(|&pos| {
            let leaf = tree.leaf(pos);
            let mut inputs: Vec<Wire> = (0..leaf_bytes / 16)
                .map(|w| b.public_value(leaf_word(leaf, 16 * w)))
                .collect();
            inputs.push(b.public_value(F128::new(table_index(pos, depth) as u64, 0)));
            b.gate_with_hint(g, &inputs, tree.siblings(pos))
        })
        .collect();
    for root in &roots {
        b.publish(root[0]);
        b.publish(root[1]);
    }

    let built = b.finish().expect("builder produces a valid circuit");
    assert_eq!(built.shape.counts, vec![positions.len()]);

    // Every opening's row is the opening we asked for, and every published
    // root is the tree's.
    let rows = built.rows::<MerklePathGate>(g);
    assert_eq!(rows.len(), positions.len());
    let want = digest_words(&hash_to_digest(&tree.root));
    for (i, &pos) in positions.iter().enumerate() {
        assert_eq!(rows[i].leaf_data, tree.leaf(pos), "row {i} leaf");
        assert_eq!(rows[i].index, table_index(pos, depth), "row {i} index");
        assert_eq!(rows[i].siblings, tree.siblings(pos), "row {i} siblings");
    }
    let published = &built.witness.public[built.witness.public.len() - 2 * positions.len()..];
    for i in 0..positions.len() {
        assert_eq!(
            [published[2 * i], published[2 * i + 1]],
            want,
            "opening {i} did not fold to the tree root"
        );
    }
}

/// **The Fiat–Shamir query binding.** A BLAKE3 gate emits a challenge word;
/// that word is wired straight into a Merkle opening's index. Nothing masks
/// it — the relation reads the low `depth` columns and the other 115 ride
/// along, pinned by the copy constraint and read by nothing.
///
/// Then the whole thing proves and verifies: two slot types, one boolean
/// each, joined by a wire that crosses from a hash output to a Merkle input.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn merkle_index_wired_to_a_challenge() {
    let (depth, leaf_bytes) = (2usize, 128usize);
    // The Merkle composite is 4 base blocks ⇒ k_log = 16, so nu = 6 puts the
    // union at the M = 22 Ligerito floor.
    let nu = 6usize;
    let mut rng = Rng(0x_C0FF_EE01);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    let mut b = CircuitBuilder::new(nu);
    let hash = b.slot(Blake3Gate { nu });
    let merkle = b.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));

    // One compression over a public message: its `out_lo0` is the challenge.
    let iv = pack8(&IV);
    let m: [u32; 16] = from_fn(|_| rng.next_u32());
    let mut hash_in = vec![b.public_value(iv[0]), b.public_value(iv[1])];
    for j in 0..4 {
        hash_in.push(b.public_value(pack4(m[4 * j..4 * j + 4].try_into().unwrap())));
    }
    hash_in.push(b.public_value(pack_params(0, 64, CHUNK_START | CHUNK_END)));
    let out = b.gate(hash, &hash_in);

    // What the challenge word selects, natively. This is `sample_queries`'
    // own arithmetic — `challenge.lo & (block_len - 1)` — and the circuit must
    // open exactly this position. It does, with no masking gadget and no
    // complement: the relation reads the index word's low `depth` columns, and
    // the gadget's polarity makes the index the tree position.
    let compressed = blake3::blake3_compress(&IV, &m, 0, 64, CHUNK_START | CHUNK_END);
    let challenge = pack8(&compressed[0..8].try_into().unwrap())[0];
    let index = (challenge.lo as u128) | ((challenge.hi as u128) << 64);
    let pos = (index & ((1u128 << depth) - 1)) as usize;
    assert_eq!(
        pos,
        (challenge.lo as usize) & ((1 << depth) - 1),
        "sample_queries' mask"
    );
    assert_ne!(index >> depth, 0, "the challenge must have high bits set");

    // The opening: leaf data public, index NOT public — it comes off the wire.
    let leaf = tree.leaf(pos);
    let mut inputs: Vec<Wire> = (0..leaf_bytes / 16)
        .map(|w| b.public_value(leaf_word(leaf, 16 * w)))
        .collect();
    inputs.push(out[0]); // ← the binding: challenge word IS the index word
    let root = b.gate_with_hint(merkle, &inputs, tree.siblings(pos));
    b.publish(root[0]);
    b.publish(root[1]);

    let built = b.finish().expect("builder produces a valid circuit");

    // The row really did receive the full challenge word, not a masked copy —
    // and it arrived by WIRE: the public segment is the 7 compression inputs,
    // the 8 leaf words and the 2 published root halves, with no index among
    // them. Nothing but the copy constraint put it there.
    assert_eq!(built.witness.public.len(), 7 + leaf_bytes / 16 + 2);
    let rows = built.rows::<MerklePathGate>(merkle);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].index, index,
        "the whole challenge word reached the row"
    );
    let published = &built.witness.public[built.witness.public.len() - 2..];
    assert_eq!(
        [published[0], published[1]],
        digest_words(&hash_to_digest(&tree.root)),
        "the wired query did not fold to the tree root"
    );

    // ---- prove / verify ----
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };

    let blake_r1cs = blake3::build_block_r1cs(nu);
    let blake_lc = blake_r1cs.csc_lincheck_circuit();
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let walker = layout.build_walker();

    // Slot order is the registry's, not the declaration's.
    let (hash_slot, merkle_slot) = (built.registry_slot(hash), built.registry_slot(merkle));
    let mut slots: Vec<(usize, UnionSlotProverInput)> = vec![
        (
            hash_slot,
            UnionSlotProverInput::new(
                blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu),
                blake_lc,
            ),
        ),
        (
            merkle_slot,
            UnionSlotProverInput::new(
                layout.generate_witness_batch_major_partial_chunk(rows, nu),
                &walker,
            ),
        ),
    ];
    slots.sort_by_key(|(i, _)| *i);
    let mut lcs: Vec<(usize, &dyn LincheckCircuit)> =
        vec![(hash_slot, blake_lc), (merkle_slot, &walker)];
    lcs.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn LincheckCircuit> = lcs.into_iter().map(|(_, c)| c).collect();

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &pcs_params,
        slots.into_iter().map(|(_, s)| s).collect(),
        Vec::new(),
        &mut ch,
    );

    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("a challenge-derived Merkle opening verifies");

    // A wrong claimed root breaks the wiring — the opening is doing work.
    let mut bad = built.witness.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            &bad,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a tampered root must be rejected"
    );
}

/// **The real recursion shape, timed**: 218 depth-13 openings over 1 KiB
/// leaves — the L0 workload a Ligerito verifier at dense m = 25 checks — as a
/// circuit rather than a bare table.
///
/// The point of the measurement is the delta against `merkle_l0_opening`,
/// which proves the identical 218 rows with no wiring at all. Everything the
/// circuit adds is the copy-constraint layer: `product_gkr` over the cell
/// space, whose size is `2^(nu + c)` with `c = ceil(log2(67 schema words))`,
/// so mu = 8 + 7 = 15 here. That is tiny next to the k_log-19 slot, and the
/// numbers should say so.
#[test]
#[ignore] // The real shape: ~8 MiB tree, minutes of proving. `-- --ignored`.
fn l0_shape_circuit_cost() {
    use std::time::Instant;

    // Pin rayon to the physical P-cores, as `merkle_l0_opening` does. Without
    // it the default pool spreads across efficiency cores and `prove` swings
    // 53-91 ms run to run, which swamps everything being measured.
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);

    let (depth, leaf_bytes, n_paths) = (13usize, 1024usize, 218usize);
    let nu = 8usize; // 218 rows ⇒ capacity 256
    let mut rng = Rng(0x_10_5A_4E_11);
    let t = Instant::now();
    let tree = Tree::new(depth, leaf_bytes, &mut rng);
    let tree_ms = t.elapsed().as_secs_f64() * 1e3;

    // How expensive is just materializing the TableType at k_log 19? It
    // carries a 2^19 identity `c_0`, and `finish` handles one per slot.
    black_box(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth).table());
    let t = Instant::now(); // second call: the first pays cold-allocator faults
    black_box(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth).table());
    let table_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- SETUP: no values, no field arithmetic, paid once ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let g = sb.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));
    let roots: Vec<Vec<Wire>> = (0..n_paths)
        .map(|_| {
            // Leaf data is free witness here — bound by the root, not pinned.
            // In the full verifier it is wired to the arithmetic gate instead.
            let mut ins: Vec<Wire> = (0..leaf_bytes / 16).map(|_| sb.input()).collect();
            ins.push(sb.public_input());
            sb.gate_hinted(g, &ins)
        })
        .collect();
    for root in &roots {
        sb.publish(root[0]);
        sb.publish(root[1]);
    }
    let shape = sb.finish().expect("builder produces a valid circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- ONLINE: this proof's values ----
    let positions: Vec<usize> = (0..n_paths).map(|i| (i * 37 + 11) % (1 << depth)).collect();
    let gather = || {
        let mut vals = Vec::with_capacity(shape.num_inputs());
        let mut hints = Vec::with_capacity(n_paths);
        for &pos in &positions {
            let leaf = tree.leaf(pos);
            vals.extend((0..leaf_bytes / 16).map(|w| leaf_word(leaf, 16 * w)));
            vals.push(F128::new(table_index(pos, depth) as u64, 0));
            hints.push(tree.siblings(pos));
        }
        (vals, hints)
    };
    let (vals, hints) = gather();
    let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();

    black_box(shape.run(&vals, &hint_refs)); // warm
    let t = Instant::now();
    let built = shape.run(&vals, &hint_refs);
    let online_ms = t.elapsed().as_secs_f64() * 1e3;

    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let walker = layout.build_walker();
    let rows = built.rows::<MerklePathGate>(g);

    let witgen = || layout.generate_witness_batch_major_partial_chunk(rows, nu);
    let prove = |witness| {
        let mut ch = FsChallenger::new(DOMAIN);
        prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            vec![UnionSlotProverInput::new(witness, &walker)],
            Vec::new(),
            &mut ch,
        )
    };

    // WARM-UP. A first prove in a fresh process pays lazy initialization —
    // twiddle tables, `OnceLock` caches, thread-pool spin-up, cold allocator —
    // and came out 20-40% high. `merkle_l0_opening` reports a median after
    // warm-up, so timing a cold single shot against it compares two different
    // statistics. Discard one round, then measure.
    black_box(prove(witgen()));

    let t = Instant::now();
    let witness = witgen();
    let wit_ms = t.elapsed().as_secs_f64() * 1e3;

    // How much of `build` is the gate re-executing BLAKE3 natively?
    let t = Instant::now();
    for row in rows {
        black_box(layout.root_chunk(row));
    }
    let eval_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let (proof, commitment, _) = prove(witness);
    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    let lcs: Vec<&dyn LincheckCircuit> = vec![&walker];
    // No thread-pool wrapper: `flock_core::verifier` pins its own 1-thread
    // `verifier_pool` around the verify cores, and the bench calls verify on
    // the default pool exactly like this. Wrapping would change the regime,
    // not match it.
    let mut ch = FsChallenger::new(DOMAIN);
    let t = Instant::now();
    verifier::verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the L0-shape circuit verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    let proof_kib = bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0;

    // Split by what recurs. `circuit_structure_does_not_depend_on_the_witness`
    // licenses the setup line: the statement is identical every proof, so
    // `ShapeBuilder::finish` is paid once and `CircuitShape::run` per proof.
    let per_proof = online_ms + wit_ms + prove_ms;
    println!(
        "\nL0 shape as a CIRCUIT: {n_paths} openings, depth {depth}, {leaf_bytes} B leaves\n\
           k_log {}  nu {nu}  dense_m {}  public {}  wires {}  proof {proof_kib:.1} KiB  \
         {threads} threads\n\
         \n\
           PER PROOF     {per_proof:6.0} ms = online {online_ms:.0} (of which eval \
         {eval_ms:.0}) + witgen {wit_ms:.0} + prove {prove_ms:.0}\n\
           verifier side {verify_ms:6.1} ms\n\
           SETUP         {setup_ms:6.0} ms = shape + finish, of which TableType \
         {table_ms:.0}   (tree gen {tree_ms:.0} ms is the test's own fixture)\n\
         \n\
        compare `cargo bench --bench merkle_l0_opening`: 48 ms prove for the \
         same rows unwired.\n",
        layout.k_log,
        union.dense_m(),
        built.public.len(),
        shape.circuit.wires().len(),
    );
}

/// **The circuit structure is witness-independent** — which is what makes
/// `finish`'s cost amortizable across proofs.
///
/// Two circuits over different trees, opening different positions with
/// different siblings, must produce the same `Circuit::digest`: the statement
/// binds the registry, the cell space and sigma, none of which depend on a
/// value. Only the witness and the public values differ.
///
/// The consequence for the timing in `l0_shape_circuit_cost`: `finish`'s work
/// could be done once per recursion shape and reused, while the gate phase —
/// `eval` computing each root — is genuinely per-proof, because that IS the
/// witness.
#[test]
fn circuit_structure_does_not_depend_on_the_witness() {
    let (depth, leaf_bytes, nu) = (2usize, 128usize, 6usize);

    let build = |seed: u64, shift: usize| {
        let mut rng = Rng(seed);
        let tree = Tree::new(depth, leaf_bytes, &mut rng);
        let mut b = CircuitBuilder::new(nu);
        let g = b.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));
        let roots: Vec<Vec<Wire>> = (0..1usize << depth)
            .map(|i| {
                let pos = (i + shift) % (1 << depth);
                let leaf = tree.leaf(pos);
                let mut inputs: Vec<Wire> = (0..leaf_bytes / 16)
                    .map(|w| b.value(leaf_word(leaf, 16 * w)))
                    .collect();
                inputs.push(b.public_value(F128::new(table_index(pos, depth) as u64, 0)));
                b.gate_with_hint(g, &inputs, tree.siblings(pos))
            })
            .collect();
        for root in &roots {
            b.publish(root[0]);
            b.publish(root[1]);
        }
        (b.finish().expect("valid circuit"), g)
    };

    let (a, ga) = build(0x_A1_11_00_01, 0);
    let (c, gc) = build(0x_B2_22_00_02, 3);

    assert_eq!(
        a.shape.circuit.digest(),
        c.shape.circuit.digest(),
        "the statement moved when only the witness did"
    );
    assert_eq!(a.shape.counts, c.shape.counts);
    assert_eq!(a.witness.public.len(), c.witness.public.len());
    // ...and the witnesses really are different, so the check is not vacuous.
    let (ra, rc) = (a.rows::<MerklePathGate>(ga), c.rows::<MerklePathGate>(gc));
    assert_ne!(ra[0].leaf_data, rc[0].leaf_data, "same leaf data");
    assert_ne!(a.witness.public, c.witness.public, "same public values");
}

// ---------------------------------------------------------------------------
// MVP-3b: the leaf arithmetic
// ---------------------------------------------------------------------------

/// What the verifier actually computes from the opened leaves, at one level:
///
/// ```text
/// enforced_sum = Σ_i  α_i · ⟨row_i, eq(v_challenges, ·)⟩
/// ```
///
/// (`ligerito::induce_sumcheck_enforced_sum`.) The inner product against an
/// `eq` table IS the multilinear evaluation of the leaf's 64 lanes at the
/// 6-dimensional point `v`, so this gate evaluates it by folding, one variable
/// per level, and folds the α-weighted result into a running accumulator so
/// the whole level's sum falls out of the last row.
///
/// Layout (`kappa = 8`, 256 columns, 200 real):
///
/// ```text
///   0  .. 64   leaf lanes         (In — the SAME wires the Merkle gate reads)
///  64  .. 70   v challenges       (In)
///  70          alpha_i            (In)
///  71          prev accumulator   (In)
///  72  ..198   fold tree          (2 columns per node: d, then the folded value)
/// 198          alpha_i · y
/// 199          accumulator out    (Out)
/// ```
///
/// **The fold, and why 2 columns per node.** `build_eq_table` is LSB-first, so
/// variable `j` is bit `j` of the lane index and folding pairs `(2i, 2i+1)`:
///
/// ```text
///   new[i] = (1+v)·f[2i] + v·f[2i+1] = f[2i] + v·(f[2i] + f[2i+1])
/// ```
///
/// A row's left-hand side is a product of two linear forms, so `v·(f+f)` is
/// one `mult_lin` — the addition rides the multiplication's `A_0` row for
/// free — but the trailing `+ f[2i]` is outside the product and costs a
/// `linear` row of its own. Hence 2 rows per node, 126 for the 63 nodes.
///
/// That is not the floor. Materializing only `d[i] = v·(f[2i]+f[2i+1])` and
/// leaving `new[i]` as the *linear form* `f[2i] + d[i]` would let the next
/// level's `mult_lin` absorb it, giving 1 row per node — at the price of `A_0`
/// rows that grow to 127 terms at the last level. Left for later on purpose:
/// this is the MVP.
struct LeafEvalGate {
    ty: Arc<ElementTableType>,
    lay: LeafLayout,
}

/// The column layout of a [`LeafEvalGate`] over `lanes` leaf words.
///
/// Parameterised because the levels differ: L0's leaves are 1 KiB (64 lanes
/// at `log_batch_size = 6`) and every recursive level's are 128 B (8 lanes).
/// Same shape, different width — and two levels with the same lane count
/// share one table type, hence one slot.
#[derive(Clone, Copy)]
struct LeafLayout {
    lanes: usize,
    vars: usize,
    v: usize,
    alpha: usize,
    prev: usize,
    fold: usize,
    n_in: usize,
    t: usize,
    acc: usize,
    k: usize,
    kappa: usize,
}

impl LeafLayout {
    fn new(lanes: usize) -> Self {
        assert!(lanes.is_power_of_two() && lanes >= 2);
        let vars = lanes.trailing_zeros() as usize;
        let (v, alpha) = (lanes, lanes + vars);
        let (prev, fold) = (alpha + 1, alpha + 2);
        let t = fold + 2 * (lanes - 1);
        let k = t + 2;
        Self {
            lanes,
            vars,
            v,
            alpha,
            prev,
            fold,
            n_in: fold,
            t,
            acc: t + 1,
            k,
            kappa: k.next_power_of_two().trailing_zeros().max(2) as usize,
        }
    }

    /// First column of fold level `l` (`1..=vars`); level `l` has
    /// `lanes >> l` nodes and each node owns two columns.
    fn base(&self, l: usize) -> usize {
        (1..l).fold(self.fold, |acc, k| acc + 2 * (self.lanes >> k))
    }

    /// The column holding entry `j` of the array entering fold level `l`.
    fn prev_col(&self, l: usize, j: usize) -> usize {
        if l == 1 {
            j
        } else {
            self.base(l - 1) + 2 * j + 1
        }
    }

    /// The fully folded value: the last level's single node.
    fn y(&self) -> usize {
        self.base(self.vars) + 1
    }
}

/// L0's lane count, and the width the single-level tests use.
const LE_LANES: usize = 64;
const LE_VARS: usize = 6;

impl LeafEvalGate {
    fn new(lanes: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let lay = LeafLayout::new(lanes);
        let mut b = ElementTableBuilder::new(lay.kappa);
        for c in 0..lay.n_in {
            b.free_wire(c);
        }
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let (p0, p1) = (lay.prev_col(l, 2 * i), lay.prev_col(l, 2 * i + 1));
                let d = lay.base(l) + 2 * i;
                b.mult_lin(d, &[(p0, one), (p1, one)], &[(lay.v + l - 1, one)]);
                b.linear(d + 1, &[(p0, one), (d, one)]);
            }
        }
        b.mult(lay.t, lay.alpha, lay.y());
        b.linear(lay.acc, &[(lay.prev, one), (lay.t, one)]);
        Self {
            ty: Arc::new(b.build().expect("leaf-eval block is valid")),
            lay,
        }
    }
}

impl GateType for LeafEvalGate {
    /// The row's committed columns, verbatim.
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.lay.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.lay.acc));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        let lay = self.lay;
        let mut z = vec![F128::ZERO; lay.k];
        z[..lay.n_in].copy_from_slice(&inputs[..lay.n_in]);
        for l in 1..=lay.vars {
            for i in 0..(lay.lanes >> l) {
                let (p0, p1) = (z[lay.prev_col(l, 2 * i)], z[lay.prev_col(l, 2 * i + 1)]);
                let d = lay.base(l) + 2 * i;
                z[d] = (p0 + p1) * z[lay.v + l - 1];
                z[d + 1] = p0 + z[d];
            }
        }
        z[lay.t] = z[lay.alpha] * z[lay.y()];
        z[lay.acc] = z[lay.prev] + z[lay.t];
        (vec![z[lay.acc]], z)
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                z[(c << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// **MVP-3b: the opened leaf reaches the arithmetic.**
///
/// The two halves of checking a PCS query, in one circuit and one proof:
///
/// - a boolean [`MerklePathGate`] binds each leaf to the committed root, and
/// - an element [`LeafEvalGate`] consumes the SAME leaf words and computes
///   `α_i · ⟨row_i, eq(v, ·)⟩`, accumulating across openings.
///
/// The 64 leaf words are one wire class each with cells in BOTH slots, so the
/// copy constraint is what makes "the leaf that is in the tree" and "the leaf
/// the arithmetic ran on" the same leaf. That is the join the whole wiring
/// layer exists for, and it crosses the class boundary: a `k_log`-19 boolean
/// slot and a `kappa`-8 element slot in one union.
///
/// The published accumulator is checked against `enforced_sum` computed
/// natively the way `ligerito::induce_sumcheck_enforced_sum` computes it.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn leaf_arithmetic_joins_the_merkle_openings() {
    #[cfg(test)]
    use flock_core::lincheck::build_eq_table;
    use flock_prover::prover::UnionElementSlotInput;

    // 1 KiB leaves: the L0 shape, and what makes a leaf exactly LE_LANES words.
    let (depth, leaf_bytes, n_open) = (2usize, 1024usize, 4usize);
    let nu = 3usize; // Merkle k_log is 19 here, so M = 22 at the Ligerito floor
    let mut rng = Rng(0x_3B_1EA_F00);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    let v: Vec<F128> = (0..LE_VARS)
        .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
        .collect();
    let alpha: Vec<F128> = (0..n_open)
        .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
        .collect();
    let positions: Vec<usize> = (0..n_open).map(|i| i % (1 << depth)).collect();

    // ---- setup ----
    let mut sb = ShapeBuilder::new(nu);
    let merkle = sb.slot(MerklePathGate::new(depth, leaf_bytes, nu, 1 << depth));
    let leafeval = sb.slot(LeafEvalGate::new(LE_LANES));

    let v_w: Vec<Wire> = (0..LE_VARS).map(|_| sb.public_input()).collect();
    let mut acc = sb.public_input(); // the accumulator's seed, published as zero
    let mut roots = Vec::new();
    for _ in 0..n_open {
        let leaf_w: Vec<Wire> = (0..LE_LANES).map(|_| sb.input()).collect();
        let idx_w = sb.public_input();

        // The join: `leaf_w` feeds the Merkle gate AND the arithmetic gate.
        let mut m_in = leaf_w.clone();
        m_in.push(idx_w);
        roots.push(sb.gate_hinted(merkle, &m_in));

        let mut a_in = leaf_w;
        a_in.extend_from_slice(&v_w);
        a_in.push(sb.public_input()); // alpha_i
        a_in.push(acc);
        acc = sb.gate(leafeval, &a_in)[0];
    }
    for r in &roots {
        sb.publish(r[0]);
        sb.publish(r[1]);
    }
    sb.publish(acc);
    let shape = sb.finish().expect("valid circuit");

    // ---- online ----
    let mut vals: Vec<F128> = v.clone();
    vals.push(F128::ZERO); // accumulator seed
    let mut hints: Vec<Vec<[u32; SLOT_WORDS]>> = Vec::new();
    for (i, &pos) in positions.iter().enumerate() {
        let leaf = tree.leaf(pos);
        vals.extend((0..LE_LANES).map(|w| leaf_word(leaf, 16 * w)));
        vals.push(F128::new(table_index(pos, depth) as u64, 0));
        vals.push(alpha[i]);
        hints.push(tree.siblings(pos));
    }
    let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();
    let built = shape.run(&vals, &hint_refs);

    // ---- the join is structural, not incidental ----
    // Every leaf word must be ONE wire class holding a cell in the Merkle
    // slot's leaf region and a cell in the leaf-eval slot's. Without this the
    // test would pass just as well on two unrelated circuits that happen to
    // agree numerically, which is exactly what the wiring layer is for.
    let iota_base = |reg: usize| -> usize {
        (0..reg)
            .map(|i| shape.registry.types()[i].io_schema.len())
            .sum()
    };
    let (m_leaf, l_leaf) = (
        iota_base(shape.registry_slot(merkle)),
        iota_base(shape.registry_slot(leafeval)),
    );
    let joined = wire_cells(&shape.circuit)
        .iter()
        .filter(|cls| {
            let has = |lo: usize| cls.iter().any(|c| (lo..lo + LE_LANES).contains(&c.slot));
            has(m_leaf) && has(l_leaf)
        })
        .count();
    assert_eq!(
        joined,
        LE_LANES * n_open,
        "leaf words are not shared between the Merkle and arithmetic slots"
    );

    // ---- the accumulator IS the verifier's enforced_sum ----
    let eq = build_eq_table(&v);
    let want = positions
        .iter()
        .enumerate()
        .fold(F128::ZERO, |s, (i, &pos)| {
            let leaf = tree.leaf(pos);
            let dot = (0..LE_LANES)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            s + alpha[i] * dot
        });
    assert_eq!(
        *built.public.last().unwrap(),
        want,
        "the circuit's accumulator disagrees with enforced_sum"
    );
    // ...and every opening still folds to the tree root.
    let root_want = digest_words(&hash_to_digest(&tree.root));
    let base = built.public.len() - 1 - 2 * n_open;
    for i in 0..n_open {
        assert_eq!(
            [built.public[base + 2 * i], built.public[base + 2 * i + 1]],
            root_want,
            "opening {i} root"
        );
    }

    // ---- prove / verify, both classes ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let walker = layout.build_walker();
    let el = match &built.witnesses[shape.registry_slot(leafeval)] {
        SlotWitness::Element(z) => z.clone(),
        other => panic!("leaf-eval slot produced {other:?}"),
    };
    // The fold is pinned by the relation, not merely consistent with it:
    // an honest witness satisfies, and perturbing any one intermediate — a
    // `mult_lin` product, its `linear` partner, or the alpha product — does
    // not. Otherwise the accumulator could be reached without doing the
    // arithmetic.
    {
        let ty = &LeafEvalGate::new(LE_LANES).ty;
        assert!(ty.satisfies(&el, nu, n_open), "honest leaf-eval witness");
        for (what, col) in [
            ("fold product", LeafLayout::new(LE_LANES).base(1)),
            ("fold sum", LeafLayout::new(LE_LANES).base(4) + 1),
            ("alpha product", LeafLayout::new(LE_LANES).t),
        ] {
            let mut bad = el.clone();
            bad[col << nu] += F128::ONE;
            assert!(
                !ty.satisfies(&bad, nu, n_open),
                "{what} (column {col}) is not constrained"
            );
        }
    }

    let m_witness =
        layout.generate_witness_batch_major_partial_chunk(built.rows::<MerklePathGate>(merkle), nu);

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(m_witness, &walker)],
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el)
        })],
        &mut ch,
    );

    let lcs: Vec<&dyn LincheckCircuit> = vec![&walker];
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("the joined Merkle + leaf-arithmetic circuit verifies");

    // Tampering with the claimed sum breaks it — the accumulator is wired to
    // the same leaf words the Merkle openings bind.
    let mut bad = built.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &bad,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a tampered enforced_sum must be rejected"
    );
}

// ---------------------------------------------------------------------------
// MVP-4: the vertical slice
// ---------------------------------------------------------------------------

/// **The query phase of a PCS verification, entirely in-circuit.**
///
/// Every earlier MVP proved one link. This joins them, and the join is that
/// *nothing about which leaves get opened is an input*:
///
/// ```text
///   transcript bytes ──(FS chain, BLAKE3)──▶ challenge words
///                                                │  copy constraint
///                                                ▼
///                                          index word of a Merkle opening
///                                                │  copy constraint (leaf words)
///                                                ▼
///                                          leaf-eval ──▶ enforced_sum
/// ```
///
/// The commitment is Ligerito-shaped: `block_len` codeword rows of 64 `F128`
/// lanes each — a 1 KiB leaf under `log_batch_size = 6` — hashed into a BLAKE3
/// Merkle tree by `flock_core::merkle`, which is exactly what `ligero_commit`
/// builds and what `verify_level_opens` checks against
/// (`chunk_root_matches_flock_core_blake3_tree` pins the table to it).
///
/// The query rule is the protocol's: `sample_queries` is
/// `challenger.sample_f128_vec(count)` masked with `block_len - 1`, and the
/// circuit reproduces it with no gadget at all — the challenge word is wired
/// into the index and the relation reads its low `depth` columns.
///
/// Six queries on purpose: a squeeze spans 64-byte XOF blocks and challenge
/// `k` is output `k % 4` of block `k / 4`, so six crosses a block boundary.
#[test]
#[ignore] // Heavy — run with `-- --ignored`.
fn mvp4_query_phase_end_to_end() {
    mvp4_slice(4, 6, 3);
}

/// The same slice at the **real L0 shape** — 218 queries over a 2^13 x 1 KiB
/// commitment, what a Ligerito verifier at dense m = 25 actually checks — with
/// the phases timed. Compare `l0_shape_circuit_cost`, which proves the Merkle
/// openings alone at this shape: the delta is what deriving the queries and
/// running the arithmetic on them costs.
#[test]
#[ignore] // The real shape. `-- --ignored`.
fn mvp4_l0_shape_cost() {
    mvp4_slice(13, 218, 8);
}

fn mvp4_slice(depth: usize, n_queries: usize, nu: usize) {
    use flock_core::challenger::Challenger as _;
    #[cfg(test)]
    use flock_core::lincheck::build_eq_table;
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp};
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::{CvSource, FsChain};

    use std::time::Instant;

    const SLICE: &[u8] = b"flock-mvp4-query-phase-v0";
    // Pin the P-cores and measure warm, as `l0_shape_circuit_cost` does — the
    // default pool and a cold first prove each move the numbers ~2x.
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let block_len = 1usize << depth;
    let leaf_bytes = 16 * LE_LANES; // 1 KiB: 64 F128 lanes

    // ---- a Ligerito-shaped L0 commitment ----
    let mut rng = Rng(0x_4E_C0_DE_01);
    let tree = Tree::new(depth, leaf_bytes, &mut rng);

    // ---- the transcript, and the queries it determines ----
    let mut ch = FsChallenger::with_hash(SLICE, HashKind::Blake3);
    ch.observe_bytes(&tree.root);
    let want_positions: Vec<usize> = ch
        .sample_f128_vec(n_queries)
        .iter()
        .map(|v| (v.lo as usize) & (block_len - 1))
        .collect();

    // Record the same transcript: the shape drives the circuit.
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(SLICE, HashKind::Blake3));
    rec.observe_bytes(&tree.root);
    let derived = rec.sample_f128_vec(n_queries);
    assert_eq!(
        derived
            .iter()
            .map(|v| (v.lo as usize) & (block_len - 1))
            .collect::<Vec<_>>(),
        want_positions,
        "recording the transcript changed it"
    );
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(SLICE);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let challenges = rec.challenges().to_vec();
    assert_eq!(challenges.len(), n_queries);

    // ---- replay it through the FS chain ----
    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&TranscriptOp> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();
    assert_eq!(trace.squeezes.len(), 1, "one squeeze: the query draw");

    // ---- setup ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let t0 = Instant::now();
    let hash = sb.slot(Blake3Gate { nu });
    let slot_hash_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let merkle = sb.slot(MerklePathGate::new(depth, leaf_bytes, nu, block_len));
    let slot_merkle_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let leafeval = sb.slot(LeafEvalGate::new(LE_LANES));
    let slot_leaf_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t_wiring = Instant::now();

    // The FS chain, verbatim from MVP-1: every row's cv and message come from
    // an earlier row's output or from a transcript word.
    let iv = [sb.public_input(), sb.public_input()];
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    // Input values in declaration order. Every FS wire is a `public_input`,
    // declared and published in the same order, so a value's index here IS its
    // index in the public segment — which the tamper check below relies on.
    let mut fs_values: Vec<F128> = Vec::new();
    let mut first_msg_pub: Option<usize> = None;
    let iv_w = pack8(&FS_CHAIN_IV);
    fs_values.extend_from_slice(&iv_w);

    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = sb.public_input();
        fs_values.push(pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(hash, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        fs_values.push(F128::ZERO);
                        sb.public_input()
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                fs_values.push(F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                ));
                                let w = sb.public_input();
                                first_msg_pub.get_or_insert(fs_values.len() - 1);
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(hash, &g_in));
    }

    // **The binding.** Challenge `k` is output `k % 4` of the squeeze's block
    // `k / 4` — so this is the wire, and there is no other route from the
    // transcript to a query.
    let sq = &trace.squeezes[0];
    let challenge_w: Vec<Wire> = (0..n_queries).map(|k| outs[sq[k / 4]][k % 4]).collect();

    // The openings, and the arithmetic on them.
    let v_w: Vec<Wire> = (0..LE_VARS).map(|_| sb.public_input()).collect();
    let mut acc = sb.public_input();
    let mut roots = Vec::new();
    for (k, &cw) in challenge_w.iter().enumerate() {
        let leaf_w: Vec<Wire> = (0..LE_LANES).map(|_| sb.input()).collect();
        let mut m_in = leaf_w.clone();
        m_in.push(cw); // ← the challenge word IS the index word
        roots.push(sb.gate_hinted(merkle, &m_in));

        let mut a_in = leaf_w;
        a_in.extend_from_slice(&v_w);
        a_in.push(sb.public_input()); // alpha_k
        a_in.push(acc);
        acc = sb.gate(leafeval, &a_in)[0];
        let _ = k;
    }
    for r in &roots {
        sb.publish(r[0]);
        sb.publish(r[1]);
    }
    sb.publish(acc);
    let wiring_ms = t_wiring.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let shape = sb.finish().expect("valid circuit");
    let finish_ms = t0.elapsed().as_secs_f64() * 1e3;
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- online ----
    let v: Vec<F128> = (0..LE_VARS)
        .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
        .collect();
    let alpha: Vec<F128> = (0..n_queries)
        .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
        .collect();

    let mut vals = fs_values;
    vals.extend_from_slice(&v);
    vals.push(F128::ZERO); // accumulator seed
    let mut hints: Vec<Vec<[u32; SLOT_WORDS]>> = Vec::new();
    for (k, &pos) in want_positions.iter().enumerate() {
        let leaf = tree.leaf(pos);
        vals.extend((0..LE_LANES).map(|w| leaf_word(leaf, 16 * w)));
        vals.push(alpha[k]);
        hints.push(tree.siblings(pos));
    }
    let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();
    black_box(shape.run(&vals, &hint_refs)); // warm
    let t = Instant::now();
    let built = shape.run(&vals, &hint_refs);
    let online_ms = t.elapsed().as_secs_f64() * 1e3;

    // **The chain is structural.** Each query's wire class must hold a cell in
    // the BLAKE3 slot's output region and a cell at the Merkle slot's index
    // word, and each leaf word a cell in the Merkle slot and one in the
    // leaf-eval slot. Values agreeing is not enough: without these classes the
    // circuit would be three unrelated computations that happen to line up.
    {
        let iota = |reg: usize| -> usize {
            (0..reg)
                .map(|i| shape.registry.types()[i].io_schema.len())
                .sum::<usize>()
        };
        let (h, m, l) = (
            iota(shape.registry_slot(hash)),
            iota(shape.registry_slot(merkle)),
            iota(shape.registry_slot(leafeval)),
        );
        let classes = wire_cells(&shape.circuit);
        let spans = |lo_a: usize, n_a: usize, lo_b: usize, n_b: usize| {
            classes
                .iter()
                .filter(|cls| {
                    let has =
                        |lo: usize, n: usize| cls.iter().any(|c| (lo..lo + n).contains(&c.slot));
                    has(lo_a, n_a) && has(lo_b, n_b)
                })
                .count()
        };
        // blake3 outputs are schema words 7..11; the Merkle index is word 64.
        assert_eq!(
            spans(h + blake3::IO_OUT_LO0, 4, m + 4 * (leaf_bytes / 64), 1),
            n_queries,
            "challenge words are not wired to the Merkle index words"
        );
        assert_eq!(
            spans(m, LE_LANES, l, LE_LANES),
            LE_LANES * n_queries,
            "leaf words are not shared with the arithmetic slot"
        );
    }

    // The circuit opened the positions the TRANSCRIPT chose. If the wiring
    // from challenge to index were wrong, `run`'s own equality check on the
    // Merkle rows' index word would already have fired; assert it anyway,
    // because this is the sentence the whole slice exists to make true.
    let rows = built.rows::<MerklePathGate>(merkle);
    for (k, &pos) in want_positions.iter().enumerate() {
        assert_eq!(
            (rows[k].index & (block_len as u128 - 1)) as usize,
            pos,
            "opening {k} did not open the query the transcript derived"
        );
        assert_eq!(
            rows[k].index,
            (challenges[k].lo as u128) | ((challenges[k].hi as u128) << 64),
            "opening {k}'s index is not the whole challenge word"
        );
        assert_eq!(rows[k].leaf_data, tree.leaf(pos), "opening {k} leaf");
    }
    // ...every opening folds to the committed root...
    let root_want = digest_words(&hash_to_digest(&tree.root));
    let base = built.public.len() - 1 - 2 * n_queries;
    for k in 0..n_queries {
        assert_eq!(
            [built.public[base + 2 * k], built.public[base + 2 * k + 1]],
            root_want,
            "opening {k} root"
        );
    }
    // ...and the accumulator is the verifier's enforced_sum over them.
    let eq = build_eq_table(&v);
    let want_sum = want_positions
        .iter()
        .enumerate()
        .fold(F128::ZERO, |s, (k, &pos)| {
            let leaf = tree.leaf(pos);
            let dot = (0..LE_LANES)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            s + alpha[k] * dot
        });
    assert_eq!(
        *built.public.last().unwrap(),
        want_sum,
        "enforced_sum disagrees with the native computation"
    );

    // ---- prove / verify: three slots, both classes ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
    let walker = layout.build_walker();
    let b3 = blake3::build_block_r1cs(nu);
    let b3_lc = b3.csc_lincheck_circuit();
    let el = match &built.witnesses[shape.registry_slot(leafeval)] {
        SlotWitness::Element(z) => z.clone(),
        other => panic!("leaf-eval slot produced {other:?}"),
    };

    // Boolean slots go in REGISTRY order.
    let t = Instant::now();
    let m_wit = layout.generate_witness_batch_major_partial_chunk(rows, nu);
    let h_wit = blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu);
    let wit_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut bool_slots = vec![
        (
            shape.registry_slot(merkle),
            UnionSlotProverInput::new(m_wit, &walker),
        ),
        (
            shape.registry_slot(hash),
            UnionSlotProverInput::new(h_wit, b3_lc),
        ),
    ];
    bool_slots.sort_by_key(|(i, _)| *i);
    let mut lcs_ord: Vec<(usize, &dyn LincheckCircuit)> = vec![
        (shape.registry_slot(merkle), &walker),
        (shape.registry_slot(hash), b3_lc),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn LincheckCircuit> = lcs_ord.into_iter().map(|(_, c)| c).collect();

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &pcs_params,
        bool_slots.into_iter().map(|(_, s)| s).collect(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el)
        })],
        &mut c,
    );

    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union,
        &shape.circuit,
        &built.public,
        &lcs,
        &commitment,
        &proof,
        &pcs_params,
        &mut c,
    )
    .expect("the query phase verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    // Tampering with a TRANSCRIPT word must break it. This is the sharp one:
    // that word is hashed into the challenge, the challenge is the index, and
    // the index selects the leaf — so a transcript the prover did not commit to
    // cannot be made to justify the openings it already produced.
    let msg_pub = first_msg_pub.expect("the transcript has message words");
    let mut bad = built.public.clone();
    bad[msg_pub] += F128::ONE;
    let mut c = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &bad,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut c,
        )
        .is_err(),
        "a tampered transcript word must be rejected"
    );

    // ---- trace size ----
    // Two different sizes, and they behave differently:
    //  * COMMITTED is jagged — `dense_words` sums `used_cols(ty) * n_t` over
    //    real row counts, then rounds to a power of two. This is what the PCS
    //    commits and opens.
    //  * ADDRESS SPACE is capacity-based — `m_bool` is
    //    `next_pow2(sum 2^(nu + k_log))`, so every slot pays a full `2^nu`
    //    rows whether it uses them or not. This is what the zerocheck and
    //    lincheck run over, and it is where the padding bites.
    {
        let mut hdr = format!(
            "\nTRACE SIZE\n  {:<10} {:>6} {:>12} {:>8} {:>6} {:>12} {:>7}\n",
            "slot", "k_log", "useful bits", "words", "rows", "dense words", "used"
        );
        let mut addr_cells = 0usize;
        for (i, ty) in shape.registry.types().iter().enumerate() {
            let words = ty.useful_bits.div_ceil(128);
            let n_t = shape.counts[i];
            let cells = 1usize << (nu + ty.k_log);
            addr_cells += cells;
            let name = if i == shape.registry_slot(hash) {
                "blake3"
            } else if i == shape.registry_slot(merkle) {
                "merkle"
            } else {
                "leaf-eval"
            };
            hdr += &format!(
                "  {:<10} {:>6} {:>12} {:>8} {:>6} {:>12} {:>6.1}%\n",
                name,
                ty.k_log,
                ty.useful_bits,
                words,
                format!("{}/{}", n_t, 1usize << nu),
                words * n_t,
                100.0 * (words * n_t) as f64 / (cells / 128) as f64,
            );
        }
        hdr += &format!(
            "  {:<10} {:>6} {:>12} {:>8} {:>6} {:>12}\n",
            "TOTAL",
            "",
            "",
            "",
            "",
            union.dense_words()
        );
        // What the LINCHECK actually sweeps — O(nnz), and unrelated to trace
        // size. The Merkle walker keeps ONE copy of the base CSC and walks it
        // per block; BLAKE3's own slot sweeps the full materialized block.
        let (ba, bb) = {
            let (a, b) = blake3::build_matrices();
            (
                a.rows.iter().map(|r| r.len()).sum::<usize>(),
                b.rows.iter().map(|r| r.len()).sum::<usize>(),
            )
        };
        hdr += &format!(
            "  lincheck nnz: merkle walker {} (effective) | blake3 block {}\n",
            walker.effective_nnz(),
            ba + bb,
        );
        hdr += &format!(
            "  committed {} words (2^{}) | address space 2^{} cells = {} words \
             ({:.1}% used)\n  M_bool {} | M_elem {} | M_total {}\n",
            union.committed_words(),
            union.dense_m() - 7,
            (addr_cells as f64).log2().ceil() as usize,
            addr_cells / 128,
            100.0 * union.dense_words() as f64 / (addr_cells / 128) as f64,
            union.m_bool(),
            union.m_elem(),
            union.m_total(),
        );
        println!("{hdr}");
    }

    println!(
        "\nMVP-4 query phase: {n_queries} queries over a 2^{depth} x 1 KiB commitment\n\
           slots: blake3 {} rows, merkle {} rows, leaf-eval {} rows | public {} | \
         dense_m {} | proof {:.1} KiB | {threads} threads\n\
         \n\
           PER PROOF     {:6.0} ms = online {online_ms:.0} + witgen {wit_ms:.0} + \
         prove {prove_ms:.0}\n\
           verifier side {verify_ms:6.1} ms\n\
           SETUP         {setup_ms:6.0} ms = slot(blake3) {slot_hash_ms:.0} + \
         slot(merkle) {slot_merkle_ms:.0} + slot(leaf) {slot_leaf_ms:.0} + \
         wiring {wiring_ms:.0} + finish {finish_ms:.0}\n",
        shape.counts[shape.registry_slot(hash)],
        shape.counts[shape.registry_slot(merkle)],
        shape.counts[shape.registry_slot(leafeval)],
        built.public.len(),
        union.dense_m(),
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        online_ms + wit_ms + prove_ms,
    );
}

// ---------------------------------------------------------------------------
// MVP-5: every level's query phase
// ---------------------------------------------------------------------------

/// Median wall-clock of `reps` timed runs after one discarded warm-up, plus
/// the spread.
///
/// Single-shot timings on these tests vary by 15-20% — more than several of
/// the effects being compared — so a lone number is not evidence. Every
/// figure quoted from `mvp5`/`mvp6` should come from here.
#[derive(Clone, Copy)]
struct Timing {
    median: f64,
    min: f64,
    max: f64,
}

impl Display for Timing {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:.0} [{:.0}-{:.0}]", self.median, self.min, self.max)
    }
}

/// Timed repetitions per phase. Five is enough to see the spread without
/// making an `#[ignore]`d test tedious.
const REPS: usize = 5;

fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, Timing) {
    let mut out = f(); // warm-up, discarded
    let mut ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        out = f();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        out,
        Timing {
            median: ms[ms.len() / 2],
            min: ms[0],
            max: ms[ms.len() - 1],
        },
    )
}

/// One Ligerito level's commitment shape.
#[derive(Clone, Copy)]
struct Level {
    /// `log2(block_len)` = `log_msg_cols + log_inv_rate`; the tree's depth.
    depth: usize,
    /// Interleaved lanes per codeword row; the leaf is `16 * lanes` bytes.
    lanes: usize,
    queries: usize,
}

/// **The whole query phase**: all four levels of the m=26 Fast ladder, in one
/// circuit and one proof.
///
/// The ladder (`docs/local/recursion-verifier-map.md` §2.5c, from
/// `configs/ligerito/m26_fast.toml`) is
/// `(log_inv_rate, log_msg_cols, lanes, queries)` per level; `block_len` is
/// `msg_cols << log_inv_rate`, so the tree depth is their sum:
///
/// ```text
///   L0  rate 1  cols 13  lanes 64  218 queries  ⇒  depth 14, 1 KiB leaves
///   L1  rate 2  cols 10  lanes  8  106 queries  ⇒  depth 12, 128 B leaves
///   L2  rate 3  cols  7  lanes  8   71 queries  ⇒  depth 10, 128 B leaves
///   L3  rate 4  cols  4  lanes  8   53 queries  ⇒  depth  8, 128 B leaves
/// ```
///
/// **What this measures.** Each level's tree shape yields a DIFFERENT
/// `MerkleTreeLayout` — different `useful_bits`, different matrices — so each
/// is its own table type, and each one's walker stores its own copy of
/// BLAKE3's base CSC. The lincheck sweeps once per slot, so the prediction is
/// ~5 x 21M nonzeros against MVP-4's ~2 x. This is the measurement that
/// decides whether collapsing the composites into one plain BLAKE3 table is
/// worth the conditional-swap and bit-decomposition glue it would need.
///
/// L1..L3 share one 8-lane leaf-eval slot: same lane count, same table type.
#[test]
#[ignore] // The full shape. `-- --ignored`.
fn mvp5_all_levels_query_phase() {
    use flock_core::challenger::Challenger as _;
    #[cfg(test)]
    use flock_core::lincheck::build_eq_table;
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp};
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::{CvSource, FsChain};
    use std::time::Instant;

    const SLICE: &[u8] = b"flock-mvp5-all-levels-v0";
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let levels = [
        Level {
            depth: 14,
            lanes: 64,
            queries: 218,
        },
        Level {
            depth: 12,
            lanes: 8,
            queries: 106,
        },
        Level {
            depth: 10,
            lanes: 8,
            queries: 71,
        },
        Level {
            depth: 8,
            lanes: 8,
            queries: 53,
        },
    ];
    let nu = 8usize; // 218 is the largest row count

    // ---- one commitment per level ----
    let mut rng = Rng(0x_5EED_0005);
    let trees: Vec<Tree> = levels
        .iter()
        .map(|l| Tree::new(l.depth, 16 * l.lanes, &mut rng))
        .collect();

    // ---- the transcript: absorb each root, then draw that level's queries ----
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(SLICE, HashKind::Blake3));
    let mut want: Vec<Vec<usize>> = Vec::new();
    for (l, tree) in levels.iter().zip(&trees) {
        rec.observe_bytes(&tree.root);
        want.push(
            rec.sample_f128_vec(l.queries)
                .iter()
                .map(|v| (v.lo as usize) & ((1usize << l.depth) - 1))
                .collect(),
        );
    }
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(SLICE);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let challenges = rec.challenges().to_vec();
    assert_eq!(
        challenges.len(),
        levels.iter().map(|l| l.queries).sum::<usize>()
    );

    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&TranscriptOp> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();
    assert_eq!(trace.squeezes.len(), levels.len(), "one squeeze per level");

    // ---- setup ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let hash = sb.slot(Blake3Gate { nu });
    let merkle: Vec<_> = levels
        .iter()
        .map(|l| sb.slot(MerklePathGate::new(l.depth, 16 * l.lanes, nu, 1 << l.depth)))
        .collect();
    // One leaf-eval slot per distinct lane count; L1..L3 share.
    let mut leaf_slot: Vec<(usize, SlotId)> = Vec::new();
    let leafeval: Vec<_> = levels
        .iter()
        .map(|l| match leaf_slot.iter().find(|(n, _)| *n == l.lanes) {
            Some((_, s)) => *s,
            None => {
                let s = sb.slot(LeafEvalGate::new(l.lanes));
                leaf_slot.push((l.lanes, s));
                s
            }
        })
        .collect();

    let iv = [sb.public_input(), sb.public_input()];
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    let mut fs_values: Vec<F128> = Vec::new();
    let iv_w = pack8(&FS_CHAIN_IV);
    fs_values.extend_from_slice(&iv_w);

    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = sb.public_input();
        fs_values.push(pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(hash, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        fs_values.push(F128::ZERO);
                        sb.public_input()
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                fs_values.push(F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                ));
                                let w = sb.public_input();
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(hash, &g_in));
    }

    // Per level: challenge words → indices, openings, arithmetic.
    let mut v_w: Vec<Vec<Wire>> = Vec::new();
    let mut acc = sb.public_input();
    let mut all_roots: Vec<Vec<Vec<Wire>>> = Vec::new();
    for (li, l) in levels.iter().enumerate() {
        let sq = &trace.squeezes[li];
        let vars = l.lanes.trailing_zeros() as usize;
        let vs: Vec<Wire> = (0..vars).map(|_| sb.public_input()).collect();
        let mut roots = Vec::with_capacity(l.queries);
        for k in 0..l.queries {
            let cw = outs[sq[k / 4]][k % 4];
            let leaf_w: Vec<Wire> = (0..l.lanes).map(|_| sb.input()).collect();
            let mut m_in = leaf_w.clone();
            m_in.push(cw);
            roots.push(sb.gate_hinted(merkle[li], &m_in));

            let mut a_in = leaf_w;
            a_in.extend_from_slice(&vs);
            a_in.push(sb.public_input()); // alpha
            a_in.push(acc);
            acc = sb.gate(leafeval[li], &a_in)[0];
        }
        v_w.push(vs);
        all_roots.push(roots);
    }
    for roots in &all_roots {
        for r in roots {
            sb.publish(r[0]);
            sb.publish(r[1]);
        }
    }
    sb.publish(acc);
    let shape = sb.finish().expect("valid circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- online ----
    let vees: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.lanes.trailing_zeros() as usize)
                .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
                .collect()
        })
        .collect();
    let alphas: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.queries)
                .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
                .collect()
        })
        .collect();

    let mut vals = fs_values;
    vals.push(F128::ZERO); // accumulator seed
    let mut hints: Vec<Vec<[u32; SLOT_WORDS]>> = Vec::new();
    for (li, l) in levels.iter().enumerate() {
        vals.extend_from_slice(&vees[li]);
        for k in 0..l.queries {
            let pos = want[li][k];
            let leaf = trees[li].leaf(pos);
            vals.extend((0..l.lanes).map(|w| leaf_word(leaf, 16 * w)));
            vals.push(alphas[li][k]);
            hints.push(trees[li].siblings(pos));
        }
    }
    let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();
    let (built, online_t) = timed(REPS, || shape.run(&vals, &hint_refs));

    // Every level opened the queries its own squeeze determined, and every
    // opening folds to that level's root.
    for (li, l) in levels.iter().enumerate() {
        let rows = built.rows::<MerklePathGate>(merkle[li]);
        assert_eq!(rows.len(), l.queries);
        for k in 0..l.queries {
            assert_eq!(
                (rows[k].index & ((1u128 << l.depth) - 1)) as usize,
                want[li][k],
                "L{li} opening {k} is not the query the transcript derived"
            );
        }
    }
    let mut at = built.public.len() - 1 - 2 * levels.iter().map(|l| l.queries).sum::<usize>();
    for (li, l) in levels.iter().enumerate() {
        let rw = digest_words(&hash_to_digest(&trees[li].root));
        for k in 0..l.queries {
            assert_eq!(
                [built.public[at], built.public[at + 1]],
                rw,
                "L{li} root {k}"
            );
            at += 2;
        }
    }
    // ...and the accumulator is enforced_sum over ALL levels.
    let mut want_sum = F128::ZERO;
    for (li, l) in levels.iter().enumerate() {
        let eq = build_eq_table(&vees[li]);
        for k in 0..l.queries {
            let leaf = trees[li].leaf(want[li][k]);
            let dot = (0..l.lanes)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            want_sum += alphas[li][k] * dot;
        }
    }
    assert_eq!(*built.public.last().unwrap(), want_sum, "enforced_sum");

    // ---- prove / verify ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let layouts: Vec<MerkleTreeLayout> = levels
        .iter()
        .map(|l| MerkleTreeLayout::with_blake3_chunk_leaf(l.depth, 16 * l.lanes, blake3_spec()))
        .collect();
    let walkers: Vec<_> = layouts.iter().map(|l| l.build_walker()).collect();
    let b3 = blake3::build_block_r1cs(nu);
    let b3_lc = b3.csc_lincheck_circuit();

    // Witnesses once; each prove rep rebuilds its slot inputs from CLONES,
    // outside the timer, because `UnionSlotProverInput::new` consumes them.
    let (hash_wit, wit_t) = timed(3, || {
        blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(hash), nu)
    });
    let merkle_wit: Vec<_> = (0..levels.len())
        .map(|li| {
            layouts[li].generate_witness_batch_major_partial_chunk(
                built.rows::<MerklePathGate>(merkle[li]),
                nu,
            )
        })
        .collect();
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();

    let mut lcs_ord: Vec<(usize, &dyn LincheckCircuit)> = vec![(shape.registry_slot(hash), b3_lc)];
    for (li, _) in levels.iter().enumerate() {
        lcs_ord.push((shape.registry_slot(merkle[li]), &walkers[li]));
    }
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn LincheckCircuit> = lcs_ord.into_iter().map(|(_, c)| c).collect();

    let ((proof, commitment), prove_t) = timed(REPS, || {
        let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![(
            shape.registry_slot(hash),
            UnionSlotProverInput::new(hash_wit.clone(), b3_lc),
        )];
        for (li, _) in levels.iter().enumerate() {
            bool_slots.push((
                shape.registry_slot(merkle[li]),
                UnionSlotProverInput::new(merkle_wit[li].clone(), &walkers[li]),
            ));
        }
        bool_slots.sort_by_key(|(i, _)| *i);
        let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
            .iter()
            .zip(els.clone())
            .map(|((_, s), z)| (shape.registry_slot(*s), z))
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(_, z)| {
                UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z))
            })
            .collect();
        let mut c = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            bool_slots.into_iter().map(|(_, s)| s).collect(),
            el_inputs,
            &mut c,
        );
        (proof, commitment)
    });

    let (_, verify_t) = timed(REPS, || {
        let mut c = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut c,
        )
        .expect("the full query phase verifies")
    });

    // ---- what it cost ----
    let mut nnz_total = 0usize;
    let mut report = String::from("\nMVP-5 FULL QUERY PHASE (m=26 Fast ladder)\n");
    report += &format!(
        "  {:<12} {:>6} {:>8} {:>7} {:>14}\n",
        "slot", "k_log", "rows", "words", "lincheck nnz"
    );
    for (i, ty) in shape.registry.types().iter().enumerate() {
        let words = ty.useful_bits.div_ceil(128);
        let nnz = if i == shape.registry_slot(hash) {
            let (a, b) = blake3::build_matrices();
            let n = a.rows.iter().map(|r| r.len()).sum::<usize>()
                + b.rows.iter().map(|r| r.len()).sum::<usize>();
            nnz_total += n;
            n
        } else if let Some(li) = (0..levels.len()).find(|&li| shape.registry_slot(merkle[li]) == i)
        {
            let n = walkers[li].effective_nnz() / layouts[li].total_blocks();
            nnz_total += n;
            n
        } else {
            0
        };
        report += &format!(
            "  {:<12} {:>6} {:>8} {:>7} {:>14}\n",
            format!("slot{i}"),
            ty.k_log,
            shape.counts[i],
            words * shape.counts[i],
            nnz
        );
    }
    report += &format!(
        "  stored lincheck nnz total {nnz_total} | dense {} words | dense_m {} | \
         M_bool {} | M_total {}\n\n  \
         medians of {REPS} runs, spread in brackets\n  \
         PER PROOF     online {online_t} + witgen {wit_t} + prove {prove_t} ms\n  \
         verifier side {verify_t} ms | proof {:.1} KiB | {threads} threads\n  \
         MERGED OPEN   frobenius {:.1} KiB, {} rounds | {} gather claims\n  \
         SETUP         {setup_ms:6.0} ms\n",
        union.dense_words(),
        union.dense_m(),
        union.m_bool(),
        union.m_total(),
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        bincode::serialize(&proof.pcs_open.frobenius)
            .map(|b| b.len())
            .unwrap_or(0) as f64
            / 1024.0,
        proof.pcs_open.merged_rounds.len(),
        shape.circuit.cells().num_gate_slots(),
    );
    println!("{report}");
}

// ---------------------------------------------------------------------------
// The collapsed opening: wiring over ONE BLAKE3 table
// ---------------------------------------------------------------------------
// MVP-7: the query phase of a REAL inner proof
// ---------------------------------------------------------------------------

/// Replay a recorded transcript's FS chain into the blake3 slot: stream words
/// become public inputs, squeeze rows chain off prior outputs. Returns the
/// per-row output wires; `trace.squeezes[fin]` indexes into them. (The same
/// block lives inline in `mvp6`; factored here for the real-transcript path.)
fn emit_fs_chain(
    sb: &mut ShapeBuilder,
    b3: SlotId,
    iv: [Wire; 2],
    trace: &FsChainTrace,
    stream: &Stream,
    bytes: &[u8],
    vals: &mut Vec<F128>,
) -> (Vec<Vec<Wire>>, Vec<Option<Wire>>) {
    use flock_prover::r1cs_hashes::fs_chain::CvSource;
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        vals.push(pack_params(counter, blen, flags));
        let params = sb.public_input();
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(b3, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        vals.push(F128::ZERO);
                        sb.public_input()
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                vals.push(F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                ));
                                let w = sb.public_input();
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(b3, &g_in));
    }
    (outs, word_wire)
}

/// The sumcheck-spine gate: one fold-and-eval step of the verifier's running
/// quadratic, `RoundQuad` in circuit form (char-2, so `u1 = t + u0` is the
/// linear coefficient trick):
///
///   c' = c + beta*u0     b' = b + beta*(y + u2)     a' = a + beta*u2
///   tr' = tr + beta*y    t' = c' + r*b' + r^2*a'
///
/// Three degenerate uses cover every verifier step with ONE table type:
/// BUILD `from_msg` (zero quad in, beta = 1, y = the running target),
/// EVAL a held quad (beta = 0; only t' consumed), and INTRO-FOLD an OOD or
/// enforced-sum claim (consume c', b', a', tr'; t' unwired).
struct SpineGate {
    ty: Arc<ElementTableType>,
}

const SP_IN: usize = 9; // c b a tr u0 u2 y beta r
const SP_K: usize = 21;

impl SpineGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let (c, b, a, tr, u0, u2, y, beta, r) = (0, 1, 2, 3, 4, 5, 6, 7, 8);
        let (pc, pb, pa, pt, co, bo, ao, tro, r2, m1, m2, to) =
            (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
        let mut bld = ElementTableBuilder::new(5);
        for w in 0..SP_IN {
            bld.free_wire(w);
        }
        bld.mult(pc, beta, u0)
            .mult_lin(pb, &[(y, one), (u2, one)], &[(beta, one)])
            .mult(pa, beta, u2)
            .mult(pt, beta, y)
            .linear(co, &[(c, one), (pc, one)])
            .linear(bo, &[(b, one), (pb, one)])
            .linear(ao, &[(a, one), (pa, one)])
            .linear(tro, &[(tr, one), (pt, one)])
            .mult(r2, r, r)
            .mult(m1, r, bo)
            .mult(m2, r2, ao)
            .linear(to, &[(co, one), (m1, one), (m2, one)]);
        Self {
            ty: Arc::new(bld.build().expect("spine gate is valid")),
        }
    }
}

impl GateType for SpineGate {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..SP_IN).map(IoWord::input).collect();
        for o in [13, 14, 15, 16, 20] {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; SP_K];
        z[..SP_IN].copy_from_slice(&inputs[..SP_IN]);
        let (c, b, a, tr, u0, u2, y, beta, r) =
            (z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8]);
        z[9] = beta * u0;
        z[10] = (y + u2) * beta;
        z[11] = beta * u2;
        z[12] = beta * y;
        z[13] = c + z[9];
        z[14] = b + z[10];
        z[15] = a + z[11];
        z[16] = tr + z[12];
        z[17] = r * r;
        z[18] = r * z[14];
        z[19] = z[17] * z[15];
        z[20] = z[13] + z[18] + z[19];
        (vec![z[13], z[14], z[15], z[16], z[20]], z)
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// The residual-basis gate (step 2b): one query's contribution to a level's
/// `induce_sumcheck_evaluate_at_residual`, at every residual position `y`.
///
/// From `q_field` the novel-basis chain runs `s_{k+1} = s_k (s_k + c_k)`
/// (`c_k = s_k(v_k)`, a constant; the `1/s_k(v_k)` normalizations fold into
/// downstream weights). The level's post-intro fold challenges `ris` build
/// `prefix = prod_k (1 + ris_k (1 + W_k))`, the suffix `W`s form subset
/// products over the `2^yr` residual positions (`1 + p_j(1+w) = w` iff the
/// bit is set), and `aw * prefix * subset(y)` accumulates into `2^yr` running
/// sums. One gate row per (level, query); the accumulators chain across
/// queries like `LeafEvalGate`'s.
///
/// `q_field` is a public input, bound at the boundary: the checker masks the
/// (already published) challenge word natively — same pattern as the cap
/// select.
struct ResidualGate {
    ty: Arc<ElementTableType>,
    sks_vks: Vec<F128>,
    acc_out: Vec<usize>,
    lmc: usize,
    pl: usize,
    yr: usize,
    n_in: usize,
    k: usize,
}

impl ResidualGate {
    /// Column layout, in declaration order:
    ///   q(0), ris(1..=pl), aw, one, acc_in[yr]          — inputs (n_in)
    ///   s_1..s_{lmc-1}                                   — the chain
    ///   pk_0..pk_{pl-1}, pr_1..pr_{pl-1}                 — prefix terms/products
    ///   w_0..w_{yl-1}                                    — normalized suffix
    ///   sp for each y with >=2 bits                      — subset products
    ///   t = aw*prefix, c_y (y>0), acc_out[yr]            — the contributions
    fn new(log_msg_cols: usize, prefix_len: usize, yr_log_n: usize, sks_vks: &[F128]) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one_w = F128::ONE;
        let (lmc, pl, yl) = (log_msg_cols, prefix_len, yr_log_n);
        assert_eq!(pl + yl, lmc);
        let yr = 1usize << yl;
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let n_in = 1 + pl + 1 + 1 + yr;
        let (q, aw, one, acc0) = (0usize, 1 + pl, 2 + pl, 3 + pl);
        let mut c = n_in; // next free column
        let mut b = ElementTableBuilder::new(6);
        for wcol in 0..n_in {
            b.free_wire(wcol);
        }
        // s columns: s_col[k] holds s_k(q); s_0 IS the q input column.
        let mut s_col = vec![q];
        for k in 1..lmc {
            b.mult_lin(
                c,
                &[(s_col[k - 1], one_w)],
                &[(s_col[k - 1], one_w), (one, sks_vks[k - 1])],
            );
            s_col.push(c);
            c += 1;
        }
        // prefix: pk = ris_k * (1 + W_k), pr = running product of (1 + pk).
        let mut pr = one; // empty product
        for k in 0..pl {
            let ivk = inv(sks_vks[k]);
            b.mult_lin(c, &[(1 + k, one_w)], &[(one, one_w), (s_col[k], ivk)]);
            let pk = c;
            c += 1;
            b.mult_lin(c, &[(pr, one_w)], &[(one, one_w), (pk, one_w)]);
            pr = c;
            c += 1;
        }
        // suffix W columns, normalized.
        let mut w = Vec::with_capacity(yl);
        for j in 0..yl {
            b.linear(c, &[(s_col[pl + j], inv(sks_vks[pl + j]))]);
            w.push(c);
            c += 1;
        }
        // subset products; sp[y]: None = 1 (y=0), single bit = w[j].
        let mut sp: Vec<Option<usize>> = vec![None; yr];
        for (j, &wc) in w.iter().enumerate() {
            sp[1 << j] = Some(wc);
        }
        for y in 1..yr {
            if sp[y].is_none() {
                let low = y & y.wrapping_neg();
                b.mult(c, sp[y ^ low].unwrap(), sp[low].unwrap());
                sp[y] = Some(c);
                c += 1;
            }
        }
        // t = aw * prefix; contributions and accumulators.
        b.mult(c, aw, pr);
        let t = c;
        c += 1;
        let mut acc_out = Vec::with_capacity(yr);
        for (y, spy) in sp.iter().enumerate() {
            let cy = match spy {
                None => t,
                Some(spc) => {
                    b.mult(c, t, *spc);
                    c += 1;
                    c - 1
                }
            };
            b.linear(c, &[(acc0 + y, one_w), (cy, one_w)]);
            acc_out.push(c);
            c += 1;
        }
        assert!(c <= 64, "residual gate spills kappa=6 ({c} cols)");
        Self {
            ty: Arc::new(b.build().expect("residual gate is valid")),
            sks_vks: sks_vks.to_vec(),
            acc_out,
            lmc,
            pl,
            yr,
            n_in,
            k: c,
        }
    }
}

impl GateType for ResidualGate {
    type Row = Vec<F128>;
    type Hint = ();

    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        for &o in &self.acc_out {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, Self::Row) {
        // A structural mirror of `new()`: same column cursor, same order.
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let (lmc, pl) = (self.lmc, self.pl);
        let yl = lmc - pl;
        let acc0 = 3 + pl;
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut s_col = vec![0usize];
        for k in 1..lmc {
            z[c] = z[s_col[k - 1]] * (z[s_col[k - 1]] + self.sks_vks[k - 1]);
            s_col.push(c);
            c += 1;
        }
        let mut pr_v = F128::ONE;
        for k in 0..pl {
            z[c] = z[1 + k] * (F128::ONE + z[s_col[k]] * inv(self.sks_vks[k]));
            let pk = z[c];
            c += 1;
            z[c] = pr_v * (F128::ONE + pk);
            pr_v = z[c];
            c += 1;
        }
        let mut w = Vec::with_capacity(yl);
        for j in 0..yl {
            z[c] = z[s_col[pl + j]] * inv(self.sks_vks[pl + j]);
            w.push(c);
            c += 1;
        }
        let mut sp: Vec<Option<usize>> = vec![None; self.yr];
        for (j, &wc) in w.iter().enumerate() {
            sp[1 << j] = Some(wc);
        }
        for y in 1..self.yr {
            if sp[y].is_none() {
                let low = y & y.wrapping_neg();
                z[c] = z[sp[y ^ low].unwrap()] * z[sp[low].unwrap()];
                sp[y] = Some(c);
                c += 1;
            }
        }
        z[c] = z[1 + pl] * pr_v;
        let t = c;
        c += 1;
        let mut outs = Vec::with_capacity(self.yr);
        for (y, spy) in sp.iter().enumerate() {
            let cy = match spy {
                None => z[t],
                Some(spc) => {
                    z[c] = z[t] * z[*spc];
                    c += 1;
                    z[c - 1]
                }
            };
            z[c] = z[acc0 + y] + cy;
            outs.push(z[c]);
            c += 1;
        }
        (outs, z)
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// `s_{k+1}(x) = s_k(x)^2 + s_k(v_k) s_k(x)` — the subspace-polynomial chain,
/// and its `s_k(v_k)` constants (a replica of ligerito's pub(crate)
/// `eval_sk_at_vks`, pinned by the residual boundary check below).
fn sk_at_vks(log_n: usize) -> Vec<F128> {
    let next = |s: F128, c: F128| s * s + c * s;
    let mut sks = vec![F128::ZERO; log_n + 1];
    sks[0] = F128::ONE;
    if log_n == 0 {
        return sks;
    }
    let mut layer: Vec<F128> = (1..=log_n).map(|i| F128::new(1u64 << i, 0)).collect();
    let mut cur = log_n;
    for i in 0..log_n {
        for j in 0..cur {
            let v = next(layer[j], sks[i]);
            if j == 0 {
                sks[i + 1] = v;
            } else {
                layer[j - 1] = v;
            }
        }
        cur -= 1;
    }
    sks
}

/// 2b stage 2, split into kappa<=6 gates (kappa=7 breaks the union's
/// column-split): PrefixGate computes `seed * prod_j (1 + a_j + b_j)` — the
/// char-2 eq prefix of a packed-direct claim (seed = gamma, a = point,
/// b = fold challenges) or an OOD claim (seed = beta, a = z). SuffixGate
/// tensors the point's tail over the 2^yr binary positions and accumulates;
/// PartialCombineGate folds `beta * resid` into the running combined vector;
/// FinalDotGate dots against the absorbed yr words.
struct PrefixGate {
    ty: Arc<ElementTableType>,
    pl: usize,
    n_in: usize,
    k: usize,
}

impl PrefixGate {
    fn new(pl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let n_in = 2 + 2 * pl; // seed, a[pl], b[pl], one
        let one = n_in - 1;
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(6);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut pr = 0;
        for j in 0..pl {
            bl.linear(c, &[(one, o), (1 + j, o), (1 + pl + j, o)]);
            c += 1;
            bl.mult(c, pr, c - 1);
            pr = c;
            c += 1;
        }
        assert!(c <= 64, "prefix gate spills ({c})");
        Self {
            ty: Arc::new(bl.build().expect("prefix gate")),
            pl,
            n_in,
            k: c,
        }
    }
}

impl GateType for PrefixGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.k - 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut pr = z[0];
        for j in 0..self.pl {
            z[c] = F128::ONE + z[1 + j] + z[1 + self.pl + j];
            c += 1;
            z[c] = pr * z[c - 1];
            pr = z[c];
            c += 1;
        }
        (vec![pr], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

struct SuffixGate {
    ty: Arc<ElementTableType>,
    acc_out: Vec<usize>,
    yl: usize,
    n_in: usize,
    k: usize,
}

impl SuffixGate {
    fn new(yl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let yr = 1usize << yl;
        let n_in = 2 + yl + yr; // p, ptS[yl], one, acc[yr]
        let (pt0, one, acc0) = (1, 1 + yl, 2 + yl);
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(6);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut e = vec![one];
        for j in 0..yl {
            bl.linear(c, &[(one, o), (pt0 + j, o)]);
            let neg = c;
            c += 1;
            let mut nx = Vec::new();
            for &pv in &e {
                bl.mult(c, pv, neg);
                nx.push(c);
                c += 1;
            }
            for &pv in &e {
                bl.mult(c, pv, pt0 + j);
                nx.push(c);
                c += 1;
            }
            e = nx;
        }
        let mut acc_out = Vec::new();
        for (y, &ey) in e.iter().enumerate() {
            bl.mult(c, 0, ey);
            c += 1;
            bl.linear(c, &[(acc0 + y, o), (c - 1, o)]);
            acc_out.push(c);
            c += 1;
        }
        assert!(c <= 64, "suffix gate spills ({c})");
        Self {
            ty: Arc::new(bl.build().expect("suffix gate")),
            acc_out,
            yl,
            n_in,
            k: c,
        }
    }
}

impl GateType for SuffixGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        for &oc in &self.acc_out {
            schema.push(IoWord::output(oc));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let yl = self.yl;
        let (pt0, acc0) = (1, 2 + yl);
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut e = vec![F128::ONE];
        for j in 0..yl {
            z[c] = F128::ONE + z[pt0 + j];
            let neg = z[c];
            c += 1;
            let mut nx = Vec::new();
            for &pv in &e {
                z[c] = pv * neg;
                nx.push(z[c]);
                c += 1;
            }
            for &pv in &e {
                z[c] = pv * z[pt0 + j];
                nx.push(z[c]);
                c += 1;
            }
            e = nx;
        }
        let mut outs = Vec::new();
        for (y, &ey) in e.iter().enumerate() {
            z[c] = z[0] * ey;
            c += 1;
            z[c] = z[acc0 + y] + z[c - 1];
            outs.push(z[c]);
            c += 1;
        }
        (outs, z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

struct PartialCombineGate {
    ty: Arc<ElementTableType>,
    acc_out: Vec<usize>,
    yr: usize,
    n_in: usize,
    k: usize,
}

impl PartialCombineGate {
    fn new(yl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let yr = 1usize << yl;
        let n_in = 1 + 2 * yr; // beta, acc[yr], resid[yr]
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(6);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut acc_out = Vec::new();
        for y in 0..yr {
            bl.mult(c, 0, 1 + yr + y);
            c += 1;
            bl.linear(c, &[(1 + y, o), (c - 1, o)]);
            acc_out.push(c);
            c += 1;
        }
        Self {
            ty: Arc::new(bl.build().expect("partial combine")),
            acc_out,
            yr,
            n_in,
            k: c,
        }
    }
}

impl GateType for PartialCombineGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        for &oc in &self.acc_out {
            schema.push(IoWord::output(oc));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let yr = self.yr;
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut outs = Vec::new();
        for y in 0..yr {
            z[c] = z[0] * z[1 + yr + y];
            c += 1;
            z[c] = z[1 + y] + z[c - 1];
            outs.push(z[c]);
            c += 1;
        }
        (outs, z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

struct FinalDotGate {
    ty: Arc<ElementTableType>,
    yr: usize,
    n_in: usize,
    k: usize,
}

impl FinalDotGate {
    fn new(yl: usize) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        let yr = 1usize << yl;
        let n_in = 2 * yr; // yr words, combined
        let mut c = n_in;
        let mut bl = ElementTableBuilder::new(6);
        for w in 0..n_in {
            bl.free_wire(w);
        }
        let mut terms = Vec::new();
        for y in 0..yr {
            bl.mult(c, y, yr + y);
            terms.push((c, o));
            c += 1;
        }
        bl.linear(c, &terms);
        c += 1;
        Self {
            ty: Arc::new(bl.build().expect("final dot")),
            yr,
            n_in,
            k: c,
        }
    }
}

impl GateType for FinalDotGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..self.n_in).map(IoWord::input).collect();
        schema.push(IoWord::output(self.k - 1));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let yr = self.yr;
        let mut z = vec![F128::ZERO; self.k];
        z[..self.n_in].copy_from_slice(&inputs[..self.n_in]);
        let mut c = self.n_in;
        let mut inner = F128::ZERO;
        for y in 0..yr {
            z[c] = z[y] * z[yr + y];
            inner += z[c];
            c += 1;
        }
        z[c] = inner;
        (vec![inner], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One merged W-round of the verifier (`jagged::fold_round_claim`):
/// `t' = (t + g1) + (t + gi) r + gi r^2` — messages `(G(1), G(inf))` wire
/// from the absorbed stream, `r` from the chain squeeze; the chain of these
/// binds rho and carries the outer gamma-combination down to `running`.
struct MergedRoundGate {
    ty: Arc<ElementTableType>,
}

impl MergedRoundGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: t(0), g1(1), gi(2), r(3)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..4 {
            b.free_wire(w);
        }
        b.mult_lin(4, &[(0, o), (2, o)], &[(3, o)]); // (t+gi) r
        b.mult(5, 3, 3); // r^2
        b.mult(6, 5, 2); // gi r^2
        b.linear(7, &[(0, o), (1, o), (4, o), (6, o)]);
        Self {
            ty: Arc::new(b.build().expect("merged round gate")),
        }
    }
}

impl GateType for MergedRoundGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..4).map(IoWord::input).collect();
        schema.push(IoWord::output(7));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 8];
        z[..4].copy_from_slice(&inputs[..4]);
        z[4] = (z[0] + z[2]) * z[3];
        z[5] = z[3] * z[3];
        z[6] = z[5] * z[2];
        z[7] = z[0] + z[1] + z[4] + z[6];
        (vec![z[7]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// Multiply-accumulate: `out = acc + x·y` — the workhorse of the multipoint
/// intake (gamma-power chains, the `T0`/`V` sums, and zero-delta joins:
/// `mac(a, b, one) = a + b` is the char-2 equality delta).
struct MacGate {
    ty: Arc<ElementTableType>,
}

impl MacGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: acc(0), x(1), y(2)
        let mut b = ElementTableBuilder::new(3);
        for w in 0..3 {
            b.free_wire(w);
        }
        b.mult(3, 1, 2);
        b.linear(4, &[(0, o), (3, o)]);
        Self {
            ty: Arc::new(b.build().expect("mac gate")),
        }
    }
}

impl GateType for MacGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..3).map(IoWord::input).collect();
        schema.push(IoWord::output(4));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 5];
        z[..3].copy_from_slice(&inputs[..3]);
        z[3] = z[1] * z[2];
        z[4] = z[0] + z[3];
        (vec![z[4]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One layer of the multipoint anchor's 4-state boundary DP (MVP-8 step 3;
/// oracle-tested standalone in `circuit_assist.rs` — this is the same gate,
/// both sourcing the transition table from `flock_core::pcs::jagged`).
struct AssistLayerGate {
    ty: Arc<ElementTableType>,
}

const AL_IN: usize = 9; // g0..g3, za, rb, rc, rd, one
const AL_OUT0: usize = 49;

impl AssistLayerGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let one = F128::ONE;
        let sparse = assist_sparse_transitions();
        let mut b = ElementTableBuilder::new(6);
        for w in 0..AL_IN {
            b.free_wire(w);
        }
        b.mult(9, 4, 5)
            .linear(10, &[(8, one), (4, one), (5, one), (9, one)])
            .linear(11, &[(4, one), (9, one)])
            .linear(12, &[(5, one), (9, one)]);
        let eq4 = [10usize, 11, 12, 9];
        b.mult(13, 6, 7)
            .linear(14, &[(8, one), (6, one), (7, one), (13, one)])
            .linear(15, &[(6, one), (13, one)])
            .linear(16, &[(7, one), (13, one)]);
        let e = [14usize, 15, 16, 13];
        let p = |i: usize, o: usize| 17 + 4 * i + o;
        for i in 0..4 {
            for o in 0..4 {
                b.mult(p(i, o), eq4[i], o);
            }
        }
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                b.mult_lin(
                    33 + 4 * cd + s,
                    &[(p(i0, o0), one), (p(i1, o1), one)],
                    &[(e[cd], one)],
                );
            }
        }
        for s in 0..4 {
            b.linear(
                AL_OUT0 + s,
                &[(33 + s, one), (37 + s, one), (41 + s, one), (45 + s, one)],
            );
        }
        Self {
            ty: Arc::new(b.build().expect("assist layer gate")),
        }
    }
}

impl GateType for AssistLayerGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..AL_IN).map(IoWord::input).collect();
        for s in 0..4 {
            schema.push(IoWord::output(AL_OUT0 + s));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let sparse = assist_sparse_transitions();
        let mut z = vec![F128::ZERO; 53];
        z[..AL_IN].copy_from_slice(&inputs[..AL_IN]);
        let one = F128::ONE;
        z[9] = z[4] * z[5];
        z[10] = one + z[4] + z[5] + z[9];
        z[11] = z[4] + z[9];
        z[12] = z[5] + z[9];
        let eq4 = [10usize, 11, 12, 9];
        z[13] = z[6] * z[7];
        z[14] = one + z[6] + z[7] + z[13];
        z[15] = z[6] + z[13];
        z[16] = z[7] + z[13];
        let e = [14usize, 15, 16, 13];
        let p = |i: usize, o: usize| 17 + 4 * i + o;
        for i in 0..4 {
            for o in 0..4 {
                z[p(i, o)] = z[eq4[i]] * z[o];
            }
        }
        for (cd, rows) in sparse.iter().enumerate() {
            for (s, row) in rows.iter().enumerate() {
                let [(i0, o0), (i1, o1)] = *row;
                z[33 + 4 * cd + s] = z[e[cd]] * (z[p(i0, o0)] + z[p(i1, o1)]);
            }
        }
        for s in 0..4 {
            z[AL_OUT0 + s] = z[33 + s] + z[37 + s] + z[41 + s] + z[45 + s];
        }
        (z[AL_OUT0..AL_OUT0 + 4].to_vec(), z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One element-zerocheck round (degree-3, convention A): `g0` rides as
/// ADVICE (a public input) and the gate enforces its defining identity as a
/// published-zero delta — the family-I pattern, no in-circuit inversion:
///
///   delta = g0 (1+t) + running + t g1          (must be 0)
///   running' = g0 (1+rho) + g1 rho + g_inf rho (1+rho)
struct ZcRoundGate {
    ty: Arc<ElementTableType>,
}

impl ZcRoundGate {
    fn new() -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        // in: running(0) g1(1) gi(2) t(3) rho(4) g0(5) one(6)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..7 {
            b.free_wire(w);
        }
        b.mult_lin(7, &[(5, o)], &[(6, o), (3, o)]); // g0(1+t)
        b.mult(8, 3, 1); // t g1
        b.linear(9, &[(7, o), (0, o), (8, o)]); // delta
        b.mult_lin(10, &[(5, o)], &[(6, o), (4, o)]); // g0(1+rho)
        b.mult(11, 1, 4); // g1 rho
        b.mult_lin(12, &[(4, o)], &[(6, o), (4, o)]); // rho(1+rho)
        b.mult(13, 2, 12); // gi rho(1+rho)
        b.linear(14, &[(10, o), (11, o), (13, o)]);
        Self {
            ty: Arc::new(b.build().expect("zc round gate")),
        }
    }
}

impl GateType for ZcRoundGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..7).map(IoWord::input).collect();
        schema.push(IoWord::output(9));
        schema.push(IoWord::output(14));
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 15];
        z[..7].copy_from_slice(&inputs[..7]);
        z[7] = z[5] * (z[6] + z[3]);
        z[8] = z[3] * z[1];
        z[9] = z[7] + z[0] + z[8];
        z[10] = z[5] * (z[6] + z[4]);
        z[11] = z[1] * z[4];
        z[12] = z[4] * (z[6] + z[4]);
        z[13] = z[2] * z[12];
        z[14] = z[10] + z[11] + z[13];
        (vec![z[9], z[14]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// The zerocheck's closing identity `running = ea eb + ec` (published-zero
/// delta) and the constant-strip + lincheck entry: with the inner shape's
/// single kappa=2 slot at full prefix, `va = ea + <eq(r_con), a_const>` and
/// likewise `vb` — the four eq-tensor weights over the last two zerocheck
/// challenges, with the table's affine constants baked as weights.
struct ZcJoinGate {
    ty: Arc<ElementTableType>,
    ac: Vec<F128>,
    bc: Vec<F128>,
}

impl ZcJoinGate {
    fn new(a_const: &[F128], b_const: &[F128]) -> Self {
        use flock_core::element_r1cs::ElementTableBuilder;
        let o = F128::ONE;
        assert_eq!(a_const.len(), 4, "kappa=2 slot");
        // in: running(0) ea(1) eb(2) ec(3) r0(4) r1(5) one(6)
        let mut b = ElementTableBuilder::new(4);
        for w in 0..7 {
            b.free_wire(w);
        }
        b.mult(7, 1, 2); // ea eb
        b.linear(8, &[(0, o), (7, o), (3, o)]); // delta
        b.mult_lin(9, &[(6, o), (4, o)], &[(6, o), (5, o)]); // (1+r0)(1+r1)
        b.mult_lin(10, &[(4, o)], &[(6, o), (5, o)]); // r0(1+r1)
        b.mult_lin(11, &[(6, o), (4, o)], &[(5, o)]); // (1+r0)r1
        b.mult(12, 4, 5); // r0 r1
        // The builder rejects zero coefficients — free-wire columns have
        // zero affine constants, so filter.
        let mut ta = vec![(1usize, o)];
        let mut tb = vec![(2usize, o)];
        for (c, (&wa, &wb)) in a_const.iter().zip(b_const).enumerate() {
            if wa != F128::ZERO {
                ta.push((9 + c, wa));
            }
            if wb != F128::ZERO {
                tb.push((9 + c, wb));
            }
        }
        b.linear(13, &ta); // va
        b.linear(14, &tb); // vb
        Self {
            ty: Arc::new(b.build().expect("zc join gate")),
            ac: a_const.to_vec(),
            bc: b_const.to_vec(),
        }
    }
}

impl GateType for ZcJoinGate {
    type Row = Vec<F128>;
    type Hint = ();
    fn table(&self) -> TableType {
        use flock_core::schedule::IoWord;
        let mut schema: Vec<IoWord> = (0..7).map(IoWord::input).collect();
        for o in [8, 13, 14] {
            schema.push(IoWord::output(o));
        }
        TableType::element(self.ty.clone()).with_io_schema(schema)
    }
    fn eval(&self, inputs: &[F128], _h: &()) -> (Vec<F128>, Self::Row) {
        let mut z = vec![F128::ZERO; 15];
        z[..7].copy_from_slice(&inputs[..7]);
        z[7] = z[1] * z[2];
        z[8] = z[0] + z[7] + z[3];
        z[9] = (z[6] + z[4]) * (z[6] + z[5]);
        z[10] = z[4] * (z[6] + z[5]);
        z[11] = (z[6] + z[4]) * z[5];
        z[12] = z[4] * z[5];
        z[13] =
            z[1] + z[9] * self.ac[0] + z[10] * self.ac[1] + z[11] * self.ac[2] + z[12] * self.ac[3];
        z[14] =
            z[2] + z[9] * self.bc[0] + z[10] * self.bc[1] + z[11] * self.bc[2] + z[12] * self.bc[3];
        (vec![z[8], z[13], z[14]], z)
    }
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, row) in rows.iter().enumerate() {
            for (col, &v) in row.iter().enumerate() {
                z[(col << nu) + j] = v;
            }
        }
        SlotWitness::Element(z)
    }
}

/// One packed-direct claim on the tape: its absorbed point/value and gamma.
struct PdRec {
    pt_v: usize,
    pt_len: usize,
    val_v: usize,
    fin: usize,
    ch: usize,
}

/// One merged W-round: the (G(1), G(inf)) value index and the rho squeeze.
struct RoundRec {
    g_v: usize,
    fin: usize,
    ch: usize,
}

/// The inner ligerito intake's single claim: q_eval's value index + gamma'.
struct InnerPd {
    q_v: usize,
    fin: usize,
    ch: usize,
}

/// The element PIOP on the tape: tau, the zerocheck rounds, (ea, eb, ec),
/// alpha, and the lincheck rounds.
struct PiopRec {
    tau_fin: usize,
    tau_ch: usize,
    tau_len: usize,
    zc_rounds: Vec<RoundRec>,
    eab_v: usize,
    alpha_fin: usize,
    alpha_ch: usize,
    lc_rounds: Vec<RoundRec>,
}

/// One OOD claim on the tape: where its `z` squeezed, its `y`/intro-msg
/// values, and its beta.
struct OodRec {
    z_fin: usize,
    z_ch: usize,
    z_len: usize,
    y_v: usize,
    intro_v: usize,
    beta_fin: usize,
    beta_ch: usize,
}

/// The multipoint region of the merged open, located on the tape (MVP-8):
/// the group values' absorb, the batching gamma, the two-product sumcheck
/// rounds, and the anchor assist's `v` + rounds. For a pure-element inner
/// (R = 0) the RS values are absent and the sumcheck is the single
/// untwisted product — see docs/multipoint-twisted-assist.tex.
struct MpRec {
    /// Value indices of the P group values `B_k` (stream-wireable).
    val_vs: Vec<usize>,
    /// The batching gamma squeeze: `(fin, ch)`.
    gamma_fin: usize,
    gamma_ch: usize,
    /// The m two-product sumcheck rounds.
    rounds: Vec<RoundRec>,
    /// The anchor's claimed twisted evaluation `v` (value index).
    anchor_v: usize,
    /// The anchor's `2(m + 1)` rounds.
    anchor_rounds: Vec<RoundRec>,
}

/// One open-phase level, located on a recorded op tape. `*_fin` are finalize
/// ordinals (indices into `FsChainTrace::squeezes`); `*_ch` index into
/// `RecordingChallenger::challenges()`.
struct OpenLevel {
    fold_fins: Vec<usize>,
    fold_chs: Vec<usize>,
    /// Value index of each fold round's message `u_0` (`u_2` is `+1`).
    fold_msg_vs: Vec<usize>,
    /// OOD claims folded before this level's queries.
    ood: Vec<OodRec>,
    /// The level's intro message value idx (unused for the final level).
    intro_v: usize,
    /// The intro/final beta: `(fin, ch)`.
    beta_fin: usize,
    beta_ch: usize,
    q_fin: usize,
    q_ch: usize,
    q_count: usize,
    a_fin: usize,
    a_ch: usize,
    a_count: usize,
}

/// Walk the recorded transcript ops and locate every open-phase squeeze the
/// circuit needs. The walk MIRRORS the succinct verifier's structure (folds,
/// cap absorbs, OOD groups, PoW, queries, alpha, beta per level), asserting
/// each op kind — a config change that moves the shape fails here, loudly,
/// not as a wrong wire.
#[allow(clippy::type_complexity)]
fn parse_open_levels(
    ops: &[TranscriptOp],
    cap0_bytes: usize,
    r: usize,
) -> (
    usize,
    Option<PiopRec>,
    Vec<PdRec>,
    Vec<RoundRec>,
    MpRec,
    InnerPd,
    usize,
    Vec<OpenLevel>,
) {
    use flock_core::transcript_record::TranscriptOp as Op;
    struct Cur<'a> {
        ops: &'a [Op],
        i: usize,
        fin: usize,
        ch: usize,
        v: usize,
    }
    impl Cur<'_> {
        fn bump(&mut self) {
            let op = &self.ops[self.i];
            if op.finalizes() {
                self.fin += 1;
            }
            match op {
                Op::SqueezeScalar => self.ch += 1,
                Op::SqueezeSlice(n) => self.ch += n,
                Op::ObserveScalar => self.v += 1,
                Op::ObserveSlice(n) => self.v += n,
                _ => {}
            }
            self.i += 1;
        }
        fn expect_obs_scalar(&mut self) {
            assert!(
                matches!(self.ops[self.i], Op::ObserveScalar),
                "op {}: expected ObserveScalar, got {:?}",
                self.i,
                self.ops[self.i]
            );
            self.bump();
        }
    }

    // Open start: the LAST ObserveBytes of the L0 cap's size —
    // `bind_statement` absorbs the same cap once, earlier.
    let starts: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, o)| matches!(o, Op::ObserveBytes(n) if *n == cap0_bytes))
        .map(|(i, _)| i)
        .collect();
    assert!(starts.len() >= 2, "expected bind + open cap absorbs");
    let start = *starts.last().unwrap();
    // The merged intake absorbs, per packed-direct claim, [point slice,
    // value, gamma squeeze] right after the merged-open label — so the claim
    // POINTS are stream words (wireable), and each gamma is a scalar squeeze.
    let mut cur = Cur {
        ops,
        i: 0,
        fin: 0,
        ch: 0,
        v: 0,
    };
    let mut gammas: Vec<PdRec> = Vec::new();
    let mut rounds: Vec<RoundRec> = Vec::new();
    let mut mp: Option<MpRec> = None;
    let mut inner_pd: Option<InnerPd> = None;
    let mut piop: Option<PiopRec> = None;
    let mut in_pd = false;
    while cur.i < start {
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-element-union-zc-v0") {
            cur.bump();
            let (tau_fin, tau_ch, tau_len) = match ops[cur.i] {
                Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
                ref o => panic!("tau, got {o:?}"),
            };
            cur.bump();
            let mut zc_rounds = Vec::with_capacity(tau_len);
            for _ in 0..tau_len {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "zc rho");
                zc_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            let eab_v = cur.v;
            cur.expect_obs_scalar(); // ea
            cur.expect_obs_scalar(); // eb
            cur.expect_obs_scalar(); // ec
            assert!(
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-element-union-lc-v0"),
                "lc label"
            );
            cur.bump();
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "alpha");
            let (alpha_fin, alpha_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut lc_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "lc rho");
                lc_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            piop = Some(PiopRec {
                tau_fin,
                tau_ch,
                tau_len,
                zc_rounds,
                eab_v,
                alpha_fin,
                alpha_ch,
                lc_rounds,
            });
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-merged-open-v0") {
            in_pd = true;
            cur.bump();
            continue;
        }
        if in_pd {
            // Ring-switched claims front the intake on boolean-bearing
            // tapes: [label, s_hat_v slice, r_dprime slice] each, then the
            // bare gamma squeezes — walk over them (mvp9 pins them
            // separately).
            if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0") {
                cur.bump(); // label
                cur.bump(); // s_hat_v slice
                cur.bump(); // r_dprime slice
                continue;
            }
            if matches!(ops[cur.i], Op::SqueezeScalar) {
                // an rs gamma — bare squeeze, no absorb
                cur.bump();
                continue;
            }
            if let Op::ObserveSlice(n) = ops[cur.i] {
                let pt_v = cur.v;
                cur.bump();
                let val_v = cur.v;
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "pd gamma");
                gammas.push(PdRec {
                    pt_v,
                    pt_len: n,
                    val_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
                continue;
            }
            in_pd = false;
            // The merged W-rounds follow the intake immediately: one
            // [ObserveScalar x2, SqueezeScalar] triplet per dense variable,
            // running until the multipoint label — count-free, so boolean
            // tapes (no packed-direct claims) parse identically.
            while matches!(ops[cur.i], Op::ObserveScalar)
                && matches!(ops[cur.i + 1], Op::ObserveScalar)
                && matches!(ops[cur.i + 2], Op::SqueezeScalar)
            {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1") {
            // The multipoint region: P group-value absorbs, the batching
            // gamma, m two-product rounds, then the anchor's label + v +
            // 2(m + 1) rounds. Each loop terminates on the next label /
            // squeeze, so a shape change fails loudly here.
            cur.bump();
            let mut val_vs = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                val_vs.push(cur.v);
                cur.bump();
            }
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "multipoint gamma");
            let (gamma_fin, gamma_ch) = (cur.fin, cur.ch);
            cur.bump();
            let mut mp_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "multipoint round");
                mp_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            assert!(
                matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0"),
                "op {}: expected the anchor label, got {:?}",
                cur.i,
                ops[cur.i]
            );
            cur.bump();
            let anchor_v = cur.v;
            cur.expect_obs_scalar();
            let mut anchor_rounds = Vec::new();
            while matches!(ops[cur.i], Op::ObserveScalar) {
                let g_v = cur.v;
                cur.expect_obs_scalar();
                cur.expect_obs_scalar();
                assert!(matches!(ops[cur.i], Op::SqueezeScalar), "anchor round");
                anchor_rounds.push(RoundRec {
                    g_v,
                    fin: cur.fin,
                    ch: cur.ch,
                });
                cur.bump();
            }
            mp = Some(MpRec {
                val_vs,
                gamma_fin,
                gamma_ch,
                rounds: mp_rounds,
                anchor_v,
                anchor_rounds,
            });
            continue;
        }
        if matches!(&ops[cur.i], Op::Label(l) if l.as_slice() == b"flock-pcs-packed-direct-v0") {
            cur.bump();
            let q_v = cur.v;
            cur.expect_obs_scalar(); // q_eval
            assert!(matches!(ops[cur.i], Op::SqueezeScalar), "inner gamma");
            inner_pd = Some(InnerPd {
                q_v,
                fin: cur.fin,
                ch: cur.ch,
            });
            cur.bump();
            continue;
        }
        cur.bump();
    }
    let inner_pd = inner_pd.expect("the inner ligerito intake");
    let mp = mp.expect("the multipoint region");
    cur.bump(); // the open-phase initial cap absorb
    let start_v = cur.v;
    cur.expect_obs_scalar(); // sumcheck start msg u_0
    cur.expect_obs_scalar(); // ... u_2

    let mut levels = Vec::new();
    let mut yr_v = 0usize;
    for li in 0..=r {
        // Fold batch: [Pow?] SqueezeScalar + ObserveScalar x2 per round. Only
        // consume a Pow that fronts a fold — the query-grinding Pow follows
        // this loop and must survive it.
        let mut fold_fins = Vec::new();
        let mut fold_chs = Vec::new();
        let mut fold_msg_vs = Vec::new();
        loop {
            match cur.ops[cur.i] {
                Op::Pow { .. } if matches!(cur.ops.get(cur.i + 1), Some(Op::SqueezeScalar)) => {
                    cur.bump()
                }
                Op::SqueezeScalar => {
                    fold_fins.push(cur.fin);
                    fold_chs.push(cur.ch);
                    cur.bump();
                    fold_msg_vs.push(cur.v);
                    cur.expect_obs_scalar();
                    cur.expect_obs_scalar();
                }
                _ => break,
            }
        }
        let mut ood = Vec::new();
        if li < r {
            // The NEXT commitment's cap, then its OOD groups.
            assert!(
                matches!(cur.ops[cur.i], Op::ObserveBytes(_)),
                "op {}: expected the next cap absorb, got {:?}",
                cur.i,
                cur.ops[cur.i]
            );
            cur.bump();
            while !matches!(cur.ops[cur.i], Op::Pow { .. }) {
                let z_len = match cur.ops[cur.i] {
                    Op::SqueezeSlice(n) => n,
                    ref o => panic!("OOD z, got {o:?}"),
                };
                let (z_fin, z_ch) = (cur.fin, cur.ch);
                cur.bump();
                let y_v = cur.v;
                cur.expect_obs_scalar(); // y
                let intro_v = cur.v;
                cur.expect_obs_scalar(); // intro u_0
                cur.expect_obs_scalar(); // intro u_2
                assert!(matches!(cur.ops[cur.i], Op::SqueezeScalar), "OOD beta");
                ood.push(OodRec {
                    z_fin,
                    z_ch,
                    z_len,
                    y_v,
                    intro_v,
                    beta_fin: cur.fin,
                    beta_ch: cur.ch,
                });
                cur.bump();
            }
        } else {
            // Final level: the yr observes.
            yr_v = cur.v;
            while matches!(cur.ops[cur.i], Op::ObserveScalar) {
                cur.bump();
            }
        }
        assert!(
            matches!(cur.ops[cur.i], Op::Pow { .. }),
            "op {}: expected query-grinding Pow, got {:?}",
            cur.i,
            cur.ops[cur.i]
        );
        cur.bump();
        let (q_fin, q_ch, q_count) = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
            ref o => panic!("op {}: expected queries squeeze, got {o:?}", cur.i),
        };
        cur.bump();
        let (a_fin, a_ch, a_count) = match cur.ops[cur.i] {
            Op::SqueezeSlice(n) => (cur.fin, cur.ch, n),
            ref o => panic!("op {}: expected alpha squeeze, got {o:?}", cur.i),
        };
        cur.bump();
        let intro_v = cur.v;
        if li < r {
            cur.expect_obs_scalar(); // intro u_0
            cur.expect_obs_scalar(); // intro u_2
        }
        assert!(matches!(cur.ops[cur.i], Op::SqueezeScalar), "beta");
        let (beta_fin, beta_ch) = (cur.fin, cur.ch);
        cur.bump();
        levels.push(OpenLevel {
            fold_fins,
            fold_chs,
            fold_msg_vs,
            ood,
            intro_v,
            beta_fin,
            beta_ch,
            q_fin,
            q_ch,
            q_count,
            a_fin,
            a_ch,
            a_count,
        });
    }
    (start_v, piop, gammas, rounds, mp, inner_pd, yr_v, levels)
}

// ---------------------------------------------------------------------------
/// **MVP-7: the query phase of a REAL inner proof.** Everything MVP-6 models,
/// against an actual proof: a pure-element union proof (no ring-switch, no
/// Frobenius assist) is proven and verified natively, its verifier transcript
/// recorded, and the circuit then replays the REAL FS chain, opens the REAL
/// leaf rows out of the proof against the REAL cap layers, and computes each
/// level's enforced sum with FS-DERIVED weights:
///
/// - the query index words, the lane-fold challenges `v`, and the basis
///   challenges `alpha` are all chain squeeze outputs, WIRED — not public
///   inputs. `v` feeds `LeafEvalGate` directly; `alpha`'s eq-tensor expansion
///   happens at the boundary (the words are published, the checker expands
///   `eq(alpha, k)` natively and supplies the weights as public inputs).
/// - the circuit's witness IS the proof: opened rows are the leaf words,
///   `merkle_proof` supplies the per-query sibling hints, the caps are the
///   absorbed commitment. Nothing is plumbed out of the prover.
/// - per query the boundary select publishes (challenge word, terminal
///   digest); per level the alpha words and the enforced-sum accumulator.
///   The checker masks, indexes the cap, and compares the sums against
///   natively recomputed `enforced_sum` values.
///
/// **Step 2a, the sumcheck spine, is IN**: one `SpineGate` element type
/// replays the verifier's running quadratic across all levels — build from
/// each message, eval at each chain challenge, intro-fold every OOD claim and
/// enforced sum (the LeafEval accumulators, consumed in-circuit rather than
/// boundary-checked) — and publishes the final `t_r`, checked against a
/// native replay. The start target is a fixed public input until the merged
/// intake lands (2c — now DONE, see below). **2b stage 1**: per-level `ResidualGate`s evaluate the
/// induced-basis residuals (`next_s` chain, prefix over later fold
/// challenges, suffix subset products) with `q_field` boundary-bound like
/// the cap select; the `2^yr` accumulators publish and check against a
/// native replica. **2c (the merged intake)**: the outer target is the
/// gamma-combination of the ABSORBED claim values (SpineGate tr-rows), the
/// W-rounds fold it through `MergedRoundGate` binding rho, and the ligerito
/// spine starts from gamma' * q_eval — every scalar in the statement is
/// transcript-bound, and `inner == t_r` closes BETWEEN CIRCUIT OUTPUTS.
/// **The PIOP spine extension**: the element zerocheck replays with `g0` as
/// advice (published-zero identity deltas — family I, no in-circuit
/// inversion), closes `running = ea eb + ec`, strips the slot's affine
/// constants (ZcJoinGate), enters the lincheck at `va + alpha vb`, and
/// reuses MergedRoundGate for the lincheck rounds; the published target IS
/// the deferred ElementAssertion's target, and the ec join forces the merged
/// intake's first absorbed value to be the zerocheck's output. **MVP-8
/// (2026-08-03) closed the rest**: the multipoint region is fully
/// in-circuit (MacGate T0/V chains, mrslot rounds, the AssistLayerGate
/// anchor DP with claim-POINT joins load-bearing, three tail zero-deltas —
/// the native `v` is gone), the PoW bit predicate binds through published
/// digest/nonce wires, and the ElementAssertion exits as bound publics.
/// Nothing in the pure-element verifier is native. The inner proof commits with
/// the lane grid at full utilization — count 2^13 x 4 cols = 2^15 words =
/// exactly 2^22 dense bits, t = 64 — so L0 is the real 64-lane / 1 KiB-leaf
/// shape with zero padding.
#[test]
#[ignore] // Proves a real m=22 inner proof first. `-- --ignored`.
fn mvp7_real_query_phase() {
    // ---- PoW grinding ops, located on the tape ----
    // Each Pow op finalizes the chain (the state digest — output wires the
    // replay already computes) and absorbs one aligned 8-byte nonce word
    // (a Bytes payload). Record (finalize ordinal, payload ordinal, bits).
    struct PowRec {
        fin: usize,
        pay: usize,
        bits: u32,
    }
    // Per level: q, c, path depth d-c, lanes — and the native cross-checks
    // that pin every piece of the plumbing before the circuit exists: each
    // opened row verifies against its cap under the recorded challenge, and
    // the recorded weights reproduce `induce_sumcheck_enforced_sum`.
    struct Lvl {
        q: usize,
        c: usize,
        path: usize,
        depth: usize,
        lanes: usize,
    }
    use Arc;
    use flock_core::element_r1cs::ElementTableBuilder;
    #[cfg(test)]
    use flock_core::lincheck::build_eq_table;
    use flock_core::transcript_record::RecordingChallenger;
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::FsChain;
    use flock_prover::schedule::Registry;
    use std::time::Instant;

    const DOMAIN7: &[u8] = b"flock-mvp7-real-v0";
    const INNER_NU: usize = 13;
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);

    // ---- the inner proof: pure element, m=22 dense, BLAKE3 everywhere ----
    let mut rng = Rng(0x_5EED_0007);
    let rf = |rng: &mut Rng| {
        F128::new(
            ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
            ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
        )
    };
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));
    let inner_ty = {
        let mut b = ElementTableBuilder::new(2);
        b.free_wire(0)
            .free_wire(1)
            .mult(2, 0, 1)
            .linear(3, &[(0, w0), (1, w1)]);
        Arc::new(b.build().expect("gate block is valid"))
    };
    let registry = Registry::new(vec![TableType::element(inner_ty.clone())], INNER_NU);
    let n_rows = 1usize << INNER_NU; // full utilization: dense = 4 * 2^20 = 2^22
    let witness: Vec<F128> = {
        let at = |c: usize, j: usize| (c << INNER_NU) + j;
        let mut z = vec![F128::ZERO; inner_ty.width() << INNER_NU];
        for j in 0..n_rows {
            let (a, b) = (rf(&mut rng), rf(&mut rng));
            z[at(0, j)] = a;
            z[at(1, j)] = b;
            z[at(2, j)] = a * b;
            z[at(3, j)] = w0 * a + w1 * b;
        }
        z
    };
    let inner_union = UnionInstance::new(&registry, vec![n_rows]);
    assert_eq!(
        inner_union.dense_m(),
        22,
        "the inner commit is exactly m=22"
    );
    let inner_params = PcsParams {
        m: inner_union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: inner_union.commit_lanes(6), // = 64 at full utilization
        merkle_hash: HashKind::Blake3,
    };
    let t = Instant::now();
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_mixed_class(
        &inner_union,
        &inner_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(|dst: &mut [F128]| {
            dst.copy_from_slice(&witness)
        })],
        &mut FsChallenger::with_hash(DOMAIN7, HashKind::Blake3),
    );
    let inner_prove_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- record the REAL verifier transcript ----
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(DOMAIN7, HashKind::Blake3));
    let claims_v = verifier::verify_ligerito_union_mixed_class(
        &inner_union,
        &[],
        &commitment,
        &proof,
        &inner_params,
        &mut rec,
    )
    .expect("the inner proof verifies natively");
    let el_claims = claims_v.element.as_ref().expect("element claims");
    // The packed-direct intake order (verifier.rs): [(c_point, c_value),
    // (lc_point, lc_value)].
    let pd_pts: Vec<Vec<F128>> = vec![el_claims.c_point.clone(), el_claims.lc_point.clone()];
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(DOMAIN7);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let chals: Vec<F128> = rec.challenges().to_vec();

    // ---- level geometry, from the proof alone ----
    let lig = &proof.pcs_open.inner.ligerito;
    assert_eq!(commitment.cap, lig.initial_cap, "commitment IS the L0 cap");
    let r = lig.recursive_caps.len();
    assert_eq!(
        lig.recursive_proofs.len(),
        r - 1,
        "levels with opens = r + 1"
    );
    let lvl_src: Vec<(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)> = (0..=r)
        .map(|li| {
            if li == 0 {
                (
                    lig.initial_cap.as_slice(),
                    &lig.initial_proof.opened_rows,
                    &lig.initial_proof.merkle_proof,
                )
            } else if li < r {
                (
                    lig.recursive_caps[li - 1].as_slice(),
                    &lig.recursive_proofs[li - 1].opened_rows,
                    &lig.recursive_proofs[li - 1].merkle_proof,
                )
            } else {
                (
                    lig.recursive_caps[r - 1].as_slice(),
                    &lig.final_proof.opened_rows,
                    &lig.final_proof.merkle_proof,
                )
            }
        })
        .collect();
    let (start_v, piop, gammas, w_rounds, mp, inner_pd, yr_v, levels) =
        parse_open_levels(t_shape.ops(), 32 * lig.initial_cap.len(), r);
    let piop = piop.expect("the element PIOP");
    assert_eq!(levels.len(), r + 1);

    // The multipoint region, validated field-for-field against the proof:
    // the located stream words ARE the proof's group values / round
    // messages / anchor transcript, in order — so the wires the MVP-8
    // assembly reads are the scalars the native verifier consumed. R = 0
    // for this pure-element inner: no RS values exist.
    {
        let fro = &proof.pcs_open.frobenius;
        let vals_rec = rec.values();
        assert!(fro.values.is_empty(), "pure-element inner has R = 0");
        assert_eq!(mp.val_vs.len(), fro.group_values.len(), "group value count");
        for (vi, want) in mp.val_vs.iter().zip(&fro.group_values) {
            assert_eq!(vals_rec[*vi], *want, "group value stream word");
        }
        assert_eq!(mp.rounds.len(), fro.rounds.len(), "two-product round count");
        for (rr, want) in mp.rounds.iter().zip(&fro.rounds) {
            assert_eq!(
                (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]),
                *want,
                "mp round msg"
            );
        }
        assert_eq!(vals_rec[mp.anchor_v], fro.anchor.v, "anchor v stream word");
        assert_eq!(
            mp.anchor_rounds.len(),
            fro.anchor.rounds.len(),
            "anchor round count"
        );
        for (rr, want) in mp.anchor_rounds.iter().zip(&fro.anchor.rounds) {
            assert_eq!(
                (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]),
                *want,
                "anchor round msg"
            );
        }
        // The native accept relations, replayed from the located pieces:
        // T0 = sum gamma^k B_k folds through the rounds to T_m == anchor.v,
        // and the anchor.v folds through its rounds to the expect the DP
        // gates will compute. This is the exact statement the in-circuit
        // assembly must publish as zero-deltas.
        let gamma = chals[mp.gamma_ch];
        let mut t = F128::ZERO;
        let mut pw = F128::ONE;
        for &vi in &mp.val_vs {
            t += pw * vals_rec[vi];
            pw *= gamma;
        }
        for rr in &mp.rounds {
            let (g1, gi) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1]);
            let r = chals[rr.ch];
            let g0 = t + g1;
            t = g0 + (g1 + g0 + gi) * r + gi * r * r;
        }
        assert_eq!(t, fro.anchor.v, "T_m must equal the anchor's claimed v");
    }

    let pows: Vec<PowRec> = {
        use flock_core::transcript_record::TranscriptOp as Op;
        let mut out = Vec::new();
        let (mut fin, mut pay) = (0usize, 0usize);
        for op in t_shape.ops() {
            if let Op::Pow { bits } = op {
                out.push(PowRec {
                    fin,
                    pay,
                    bits: *bits,
                });
            }
            if op.finalizes() {
                fin += 1;
            }
            match op {
                Op::ObserveBytes(_) | Op::Pow { .. } => pay += 1,
                _ => {}
            }
        }
        out
    };
    assert!(!pows.is_empty(), "the Fast profile grinds");

    let mut geo: Vec<Lvl> = Vec::new();
    let mut native_sums: Vec<F128> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let (cap, rows, paths) = lvl_src[li];
        let q = lvl.q_count;
        assert_eq!(rows.len(), q, "L{li}: one opened row per query");
        let c = cap.len().trailing_zeros() as usize;
        assert_eq!(cap.len(), 1 << c, "L{li}: cap is a power of two");
        let path = paths.len() / q;
        assert_eq!(paths.len(), q * path, "L{li}: flat paths divide evenly");
        let depth = path + c;
        let lanes = rows[0].len();
        assert!(
            lanes.is_power_of_two() && lanes >= 4,
            "L{li}: lanes {lanes}"
        );
        assert_eq!(
            lanes,
            1 << lvl.fold_fins.len(),
            "L{li}: one fold challenge per lane bit"
        );
        let fold_vals: Vec<F128> = lvl.fold_chs.iter().map(|&i| chals[i]).collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let eqv = build_eq_table(&fold_vals);
        let aw = build_eq_table(&alpha_vals);
        let mut sum = F128::ZERO;
        for (k, row) in rows.iter().enumerate() {
            let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << depth) - 1);
            let mut leaf_bytes = Vec::with_capacity(16 * lanes);
            for f in row {
                leaf_bytes.extend_from_slice(&f.lo.to_le_bytes());
                leaf_bytes.extend_from_slice(&f.hi.to_le_bytes());
            }
            let lh = core_merkle::hash_leaf(&leaf_bytes, HashKind::Blake3);
            assert!(
                core_merkle::verify_merkle_proof_capped(
                    cap,
                    1 << depth,
                    &lh,
                    pos,
                    &paths[k * path..(k + 1) * path],
                    HashKind::Blake3,
                ),
                "L{li} query {k}: capped path verifies natively"
            );
            let dot = row
                .iter()
                .zip(eqv.iter())
                .map(|(&x, &e)| x * e)
                .fold(F128::ZERO, |a, v| a + v);
            sum += aw[k] * dot;
        }
        native_sums.push(sum);
        geo.push(Lvl {
            q,
            c,
            path,
            depth,
            lanes,
        });
    }

    // ---- the FS chain over the real byte stream ----
    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&TranscriptOp> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();
    let b3_rows: usize = trace.rows.len()
        + geo
            .iter()
            .map(|g| (g.lanes / 4 + g.path) * g.q)
            .sum::<usize>();
    let nu = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
    let max_path = geo.iter().map(|g| g.path).max().unwrap().max(1);

    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let slots = CollapsedSlots {
        b3: sb.slot(Blake3Gate { nu }),
        swap: sb.slot(SwapGate { nu }),
        spread: sb.slot(BitSpreadGate {
            ty: BitSpreadTable::new(max_path),
            nu,
        }),
    };
    let mut leaf_slot: Vec<(usize, SlotId)> = Vec::new();
    // ONE 8-lane leaf-eval type serves every level: a 64-lane leaf is 8
    // CHAINED rows, row h taking lanes [8h, 8h+8) with alpha input
    // `alpha_k·eq(v[3..], h)` — the same boundary-expanded-public pattern
    // alpha itself rides — because `y_64 = Σ_h eq(v[3..], h)·y_8(group h)`
    // (split the lane index at bit 3; build_eq_table is LSB-first). This
    // deletes the kappa-8 slot: 2^21 element-region words (27% of the
    // region) and a 73-word schema from the cell space. The high v wires
    // stay bound through the residual/prefix gates, which consume every
    // level's fold wires.
    let leafeval: Vec<_> = geo
        .iter()
        .map(|g| {
            let lanes = g.lanes.min(8);
            match leaf_slot.iter().find(|(n, _)| *n == lanes) {
                Some((_, s)) => *s,
                None => {
                    let s = sb.slot(LeafEvalGate::new(lanes));
                    leaf_slot.push((lanes, s));
                    s
                }
            }
        })
        .collect();

    let spine = sb.slot(SpineGate::new());
    leaf_slot.push((0, spine));

    let mut vals: Vec<F128> = Vec::new();
    let iv_w = pack8(&FS_CHAIN_IV);
    vals.extend_from_slice(&iv_w);
    let iv = [sb.public_input(), sb.public_input()];
    let (outs, word_wire) =
        emit_fs_chain(&mut sb, slots.b3, iv, &trace, &stream, &bytes, &mut vals);
    // Observed-value index -> absorbed-stream word index, for wiring the
    // sumcheck messages (they are absorbed proof scalars, so their wires
    // already exist as the chain's block inputs).
    let mut vmap: Vec<Option<usize>> = Vec::new();
    for (wi, w) in stream.words.iter().enumerate() {
        if let StreamWord::Value(vi) = *w {
            if vmap.len() <= vi {
                vmap.resize(vi + 1, None);
            }
            vmap[vi] = Some(wi);
        }
    }

    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    // (alpha wires, per-query (cw, cv), acc) per level — published AFTER the
    // loop: `built.public` lists entries in DECLARATION order, so publishing
    // inside the loop would interleave with the next level's public inputs
    // and break the tail walk below.
    let mut to_publish: Vec<(Vec<Wire>, Vec<(Wire, [Wire; 2])>)> = Vec::new();
    let mut level_accs: Vec<Wire> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let g = &geo[li];
        let (_, rows, paths) = lvl_src[li];
        let sqq = &trace.squeezes[lvl.q_fin];
        let sqa = &trace.squeezes[lvl.a_fin];
        // alpha words: chain outputs, PUBLISHED for the checker's expansion.
        let a_wires: Vec<Wire> = (0..lvl.a_count).map(|j| outs[sqa[j / 4]][j % 4]).collect();
        // v: this level's fold challenges, chain outputs, wired straight in.
        let v_wires: Vec<Wire> = lvl
            .fold_fins
            .iter()
            .map(|&f| outs[trace.squeezes[f][0]][0])
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        // The hi-group weights of the leaf-eval split: eq over the native
        // values of the fold challenges past the 8-lane gate's three.
        let le_vars = g.lanes.min(8).trailing_zeros() as usize;
        let le_groups = g.lanes >> le_vars;
        let hw = {
            let v_hi: Vec<F128> = lvl.fold_chs[le_vars..].iter().map(|&i| chals[i]).collect();
            build_eq_table(&v_hi)
        };
        vals.push(F128::ZERO);
        let mut acc = sb.public_input();
        let mut opens: Vec<(Wire, [Wire; 2])> = Vec::with_capacity(g.q);
        for k in 0..g.q {
            vals.extend_from_slice(&rows[k]);
            let leaf_w: Vec<Wire> = (0..g.lanes).map(|_| sb.input()).collect();
            let cw = outs[sqq[k / 4]][k % 4];
            let cv = emit_opening(&mut sb, slots, iv, &leaf_w, cw, g.depth, g.c, &mut vals);
            opens.push((cw, cv));
            hints.extend(
                paths[k * g.path..(k + 1) * g.path]
                    .iter()
                    .map(hash_to_digest),
            );
            let lanes = g.lanes.min(8);
            for h in 0..le_groups {
                let mut a_in: Vec<Wire> = leaf_w[lanes * h..lanes * (h + 1)].to_vec();
                a_in.extend_from_slice(&v_wires[..le_vars]);
                vals.push(aw[k] * hw[h]);
                a_in.push(sb.public_input());
                a_in.push(acc);
                acc = sb.gate(leafeval[li], &a_in)[0];
            }
        }
        to_publish.push((a_wires, opens));
        level_accs.push(acc);
    }

    // ---- the merged intake + W-rounds (2c): binding the start target ----
    // The outer target is the gamma-combination of the element claims'
    // absorbed values; each merged round folds it through the quadratic
    // (t+g1) + (t+gi)r + gi r^2, binding rho; the boundary check below
    // closes `running == q_eval * v` with the assist's v native. The
    // ligerito spine then starts from gamma' * q_eval — every scalar in the
    // statement is now transcript-bound.
    vals.push(F128::ZERO);
    let zw = sb.public_input();
    vals.push(F128::ONE);
    let ow = sb.public_input();
    let chw = |outs: &Vec<Vec<Wire>>, trace_sq: &Vec<Vec<usize>>, fin: usize| -> Wire {
        outs[trace_sq[fin][0]][0]
    };
    let wv = |vi: usize| -> Wire { word_wire[vmap[vi].expect("stream word")].expect("wired") };
    // ---- the element PIOP (spine extension): zerocheck + lincheck ----
    // g0 rides as advice per zerocheck round; its defining identity and the
    // closing `running = ea eb + ec` publish as zero-deltas. The strip gate
    // folds the slot's affine constants at the last two zerocheck challenges,
    // and the lincheck's two rounds reuse MergedRoundGate. The `ec` join
    // (published-zero) forces the merged intake's first absorbed value to BE
    // the zerocheck's output.
    let zc_natives: Vec<F128> = {
        // native g0 chain, needed as the advice values
        let mut out = Vec::with_capacity(piop.tau_len);
        let mut running = F128::ZERO;
        for (i, rr) in piop.zc_rounds.iter().enumerate() {
            let (g1, gi) = (rec.values()[rr.g_v], rec.values()[rr.g_v + 1]);
            let t = chals[piop.tau_ch + i];
            let rho = chals[rr.ch];
            let g0 = (running + t * g1) * (F128::ONE + t).inv();
            out.push(g0);
            running = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        out
    };
    let zslot = sb.slot(ZcRoundGate::new());
    leaf_slot.push((500, zslot));
    let mut zr = zw;
    let mut zc_deltas: Vec<Wire> = Vec::new();
    for (i, rr) in piop.zc_rounds.iter().enumerate() {
        let sqt = &trace.squeezes[piop.tau_fin];
        let t_w = outs[sqt[i / 4]][i % 4];
        let rho_w = chw(&outs, &trace.squeezes, rr.fin);
        vals.push(zc_natives[i]);
        let g0w = sb.public_input();
        let g = sb.gate(
            zslot,
            &[zr, wv(rr.g_v), wv(rr.g_v + 1), t_w, rho_w, g0w, ow],
        );
        zc_deltas.push(g[0]);
        zr = g[1];
    }
    let jslot = sb.slot(ZcJoinGate::new(inner_ty.a_const(), inner_ty.b_const()));
    leaf_slot.push((501, jslot));
    let nzc = piop.zc_rounds.len();
    let jr = sb.gate(
        jslot,
        &[
            zr,
            wv(piop.eab_v),
            wv(piop.eab_v + 1),
            wv(piop.eab_v + 2),
            chw(&outs, &trace.squeezes, piop.zc_rounds[nzc - 2].fin),
            chw(&outs, &trace.squeezes, piop.zc_rounds[nzc - 1].fin),
            ow,
        ],
    );
    let (zc_fin_delta, va_w, vb_w) = (jr[0], jr[1], jr[2]);
    let alpha_w = chw(&outs, &trace.squeezes, piop.alpha_fin);
    let lc0 = sb.gate(spine, &[zw, zw, zw, va_w, zw, zw, vb_w, alpha_w, zw])[3];
    let mut lr = lc0;
    // (MergedRoundGate is declared just below for the W-rounds; the lincheck
    // rounds share it, so hoist the slot.)
    let mrslot = sb.slot(MergedRoundGate::new());
    leaf_slot.push((400, mrslot));
    for rr in &piop.lc_rounds {
        lr = sb.gate(
            mrslot,
            &[
                lr,
                wv(rr.g_v),
                wv(rr.g_v + 1),
                chw(&outs, &trace.squeezes, rr.fin),
            ],
        )[0];
    }
    let lc_target = lr;
    // ec join: the intake's first absorbed value == the zerocheck's ec.
    let ec_join = sb.gate(
        spine,
        &[
            zw,
            zw,
            zw,
            wv(piop.eab_v + 2),
            zw,
            zw,
            wv(gammas[0].val_v),
            ow,
            zw,
        ],
    )[3];

    // Outer target: SpineGate tr-rows accumulate gamma_k * value_k.
    let mut mt = zw;
    for pd in &gammas {
        let gw = chw(&outs, &trace.squeezes, pd.fin);
        let f = sb.gate(spine, &[zw, zw, zw, mt, zw, zw, wv(pd.val_v), gw, zw]);
        mt = f[3];
    }
    // The W-rounds, binding rho.
    for rr in &w_rounds {
        let rw = chw(&outs, &trace.squeezes, rr.fin);
        mt = sb.gate(mrslot, &[mt, wv(rr.g_v), wv(rr.g_v + 1), rw])[0];
    }
    let running_w = mt;
    // The ligerito start target: gamma' * q_eval.
    let gpw = chw(&outs, &trace.squeezes, inner_pd.fin);
    let tw0 = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, wv(inner_pd.q_v), gpw, zw]);
    let mut tw = tw0[3];
    let st = sb.gate(
        spine,
        &[zw, zw, zw, zw, wv(start_v), wv(start_v + 1), tw, ow, zw],
    );
    let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            let rw = chw(&outs, &trace.squeezes, lvl.fold_fins[j]);
            let ev = sb.gate(spine, &[qc, qb, qa, zw, zw, zw, zw, zw, rw]);
            tw = ev[4];
            let bld = sb.gate(spine, &[zw, zw, zw, zw, wv(mv), wv(mv + 1), tw, ow, zw]);
            (qc, qb, qa) = (bld[0], bld[1], bld[2]);
        }
        if li < r {
            for od in &lvl.ood {
                let bw = chw(&outs, &trace.squeezes, od.beta_fin);
                let f = sb.gate(
                    spine,
                    &[
                        qc,
                        qb,
                        qa,
                        tw,
                        wv(od.intro_v),
                        wv(od.intro_v + 1),
                        wv(od.y_v),
                        bw,
                        zw,
                    ],
                );
                (qc, qb, qa, tw) = (f[0], f[1], f[2], f[3]);
            }
            let bw = chw(&outs, &trace.squeezes, lvl.beta_fin);
            let f = sb.gate(
                spine,
                &[
                    qc,
                    qb,
                    qa,
                    tw,
                    wv(lvl.intro_v),
                    wv(lvl.intro_v + 1),
                    level_accs[li],
                    bw,
                    zw,
                ],
            );
            (qc, qb, qa, tw) = (f[0], f[1], f[2], f[3]);
        } else {
            let bw = chw(&outs, &trace.squeezes, lvl.beta_fin);
            let f = sb.gate(spine, &[zw, zw, zw, tw, zw, zw, level_accs[li], bw, zw]);
            tw = f[3];
        }
    }
    let t_final = tw;

    // ---- the residual basis (2b, stage 1): induce_..._at_residual in-circuit ----
    // Level li's basis prefix folds over the LATER levels' fold challenges;
    // yr comes from the proof. eval_b / OOD residuals / the final inner==t_r
    // stay native for now.
    let yr_log = {
        let yl = proof.pcs_open.inner.ligerito.final_proof.yr.len();
        assert!(yl.is_power_of_two());
        yl.trailing_zeros() as usize
    };
    let yr_len = 1usize << yr_log;
    let mut resid_pub: Vec<Vec<Wire>> = Vec::new();
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len()).sum();
        let lmc = pl + yr_log;
        let sks = sk_at_vks(lmc);
        let rslot = sb.slot(ResidualGate::new(lmc, pl, yr_log, &sks));
        leaf_slot.push((100 + li, rslot));
        let ris_w: Vec<Wire> = levels[li + 1..]
            .iter()
            .flat_map(|l| l.fold_fins.iter().map(|&f| chw(&outs, &trace.squeezes, f)))
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        let mut accs: Vec<Wire> = (0..yr_len).map(|_| zw).collect();
        for k in 0..geo[li].q {
            let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << geo[li].depth) - 1);
            vals.push(F128::new(pos as u64, 0));
            let qf = sb.public_input();
            vals.push(aw[k]);
            let awp = sb.public_input();
            let mut g_in = vec![qf];
            g_in.extend_from_slice(&ris_w);
            g_in.push(awp);
            g_in.push(ow);
            g_in.extend_from_slice(&accs);
            accs = sb.gate(rslot, &g_in);
        }
        resid_pub.push(accs);
    }

    // ---- eval_b + the close-out (2b stage 2) ----
    // (These gates once needed MVP7_CLOSEOUT=1: the extra element types
    // pushed the outer union's boolean RS claims off the DeferredDense
    // shape — small/sparse suffixes routed to forms the merged open
    // rejects. ring_switch now defers every claim, so the close-out is
    // unconditional.)
    let closeout = true;
    assert_eq!(gammas.len(), pd_pts.len(), "one gamma per claim");
    for (k, pd) in gammas.iter().enumerate() {
        for j in 0..pd.pt_len {
            assert_eq!(
                rec.values()[pd.pt_v + j],
                pd_pts[k][j],
                "pt {k}:{j} on tape"
            );
        }
    }
    let pl_full: usize = levels.iter().map(|l| l.fold_fins.len()).sum();
    let inner_w = if !closeout {
        None
    } else {
        let ris_full: Vec<Wire> = levels
            .iter()
            .flat_map(|l| l.fold_fins.iter().map(|&f| chw(&outs, &trace.squeezes, f)))
            .collect();
        let sxslot = sb.slot(SuffixGate::new(yr_log));
        leaf_slot.push((300, sxslot));
        // ONE prefix slot at pl_full serves every prefix length: shorter calls
        // pad their (a, b) blocks with zero pairs — each padded factor is
        // 1 + 0 + 0 = 1, so the wide gate is exact. (Was one slot per distinct
        // pl: two extra kappa-6 types whose schemas alone were ~200K cells.)
        let mut pf_slots: Vec<(usize, SlotId)> = Vec::new();
        let mut pf_slot = |sb: &mut ShapeBuilder,
                           leaf_slot: &mut Vec<(usize, SlotId)>,
                           _pl: usize| {
            match pf_slots.first() {
                Some((_, sl)) => *sl,
                None => {
                    let sl = sb.slot(PrefixGate::new(pl_full));
                    leaf_slot.push((310 + pl_full, sl));
                    pf_slots.push((pl_full, sl));
                    sl
                }
            }
        };
        let mut evb_accs: Vec<Wire> = (0..yr_len).map(|_| zw).collect();
        // The ligerito layer sees ONE packed-direct claim: (rho, q_eval) with
        // gamma'. rho's coords are the merged-round squeezes — chain wires.
        {
            assert_eq!(
                w_rounds.len(),
                pl_full + yr_log,
                "rho spans the dense domain"
            );
            let sl = pf_slot(&mut sb, &mut leaf_slot, pl_full);
            let mut g_in = vec![chw(&outs, &trace.squeezes, inner_pd.fin)];
            for rr in &w_rounds[..pl_full] {
                g_in.push(chw(&outs, &trace.squeezes, rr.fin));
            }
            g_in.extend_from_slice(&ris_full);
            g_in.push(ow);
            let p = sb.gate(sl, &g_in)[0];
            let mut s_in = vec![p];
            for rr in &w_rounds[pl_full..] {
                s_in.push(chw(&outs, &trace.squeezes, rr.fin));
            }
            s_in.push(ow);
            s_in.extend_from_slice(&evb_accs);
            evb_accs = sb.gate(sxslot, &s_in);
        }
        // OOD claims: same shape, seed = beta, point = the squeezed z.
        for (li, lvl) in levels.iter().enumerate() {
            for od in &lvl.ood {
                let folded = od.z_len - yr_log;
                let later: Vec<Wire> = levels[li + 1..]
                    .iter()
                    .flat_map(|l| l.fold_fins.iter().map(|&f| chw(&outs, &trace.squeezes, f)))
                    .collect();
                assert_eq!(later.len(), folded, "OOD prefix = later folds");
                let sl = pf_slot(&mut sb, &mut leaf_slot, folded);
                let sq = &trace.squeezes[od.z_fin];
                let mut g_in = vec![chw(&outs, &trace.squeezes, od.beta_fin)];
                for j in 0..folded {
                    g_in.push(outs[sq[j / 4]][j % 4]);
                }
                g_in.extend(repeat_n(zw, pl_full - folded));
                g_in.extend_from_slice(&later);
                g_in.extend(repeat_n(zw, pl_full - folded));
                g_in.push(ow);
                let p = sb.gate(sl, &g_in)[0];
                let mut s_in = vec![p];
                for j in 0..yr_log {
                    let jj = folded + j;
                    s_in.push(outs[sq[jj / 4]][jj % 4]);
                }
                s_in.push(ow);
                s_in.extend_from_slice(&evb_accs);
                evb_accs = sb.gate(sxslot, &s_in);
            }
        }
        // beta-weighted residuals fold in per level, then the yr dot.
        let pcslot = sb.slot(PartialCombineGate::new(yr_log));
        leaf_slot.push((301, pcslot));
        let mut comb = evb_accs;
        for (li, lvl) in levels.iter().enumerate() {
            let mut g_in = vec![chw(&outs, &trace.squeezes, lvl.beta_fin)];
            g_in.extend_from_slice(&comb);
            g_in.extend_from_slice(&resid_pub[li]);
            comb = sb.gate(pcslot, &g_in);
        }
        let fdslot = sb.slot(FinalDotGate::new(yr_log));
        leaf_slot.push((302, fdslot));
        let mut g_in: Vec<Wire> = (0..yr_len).map(|y| wv(yr_v + y)).collect();
        g_in.extend_from_slice(&comb);
        Some(sb.gate(fdslot, &g_in)[0])
    };

    // ---- MVP-8 step 2: the multipoint intake in-circuit ----
    // T0 = Σ gamma^k·B_k (Mac chain over the absorbed group values), the m
    // two-product rounds through the SAME MergedRoundGate slot the W-rounds
    // use, and two zero-delta joins: T_m == anchor.v binds the round chain
    // to the anchor's claimed evaluation, and running_W == q_eval·V with
    // V = Σ B_k IN-CIRCUIT replaces the boundary's native v. The anchor's
    // own rounds + expect (the AssistLayerGate chains) are step 3.
    let macslot = sb.slot(MacGate::new());
    leaf_slot.push((600, macslot));
    let gamma_w = chw(&outs, &trace.squeezes, mp.gamma_fin);
    let mut t0 = zw;
    let mut vsum = zw;
    let mut pws: Vec<Wire> = vec![ow];
    for (k, &vi) in mp.val_vs.iter().enumerate() {
        t0 = sb.gate(macslot, &[t0, pws[k], wv(vi)])[0];
        vsum = sb.gate(macslot, &[vsum, wv(vi), ow])[0];
        if k + 1 < mp.val_vs.len() {
            let p = sb.gate(macslot, &[zw, pws[k], gamma_w])[0];
            pws.push(p);
        }
    }
    let mut tm = t0;
    let mut rho2_w: Vec<Wire> = Vec::new();
    for rr in &mp.rounds {
        let r_w = chw(&outs, &trace.squeezes, rr.fin);
        rho2_w.push(r_w);
        tm = sb.gate(mrslot, &[tm, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
    }
    let delta_tm = sb.gate(macslot, &[tm, wv(mp.anchor_v), ow])[0];
    let qv = sb.gate(macslot, &[zw, wv(inner_pd.q_v), vsum])[0];
    let delta_rq = sb.gate(macslot, &[running_w, qv, ow])[0];

    // ---- MVP-8 step 3: the anchor in-circuit ----
    // 3a: the anchor's claimed v folds through its 2(m+1) rounds (mrslot
    // reuse); the squeezes are the sigma wires the expect consumes.
    let mut acl = wv(mp.anchor_v);
    let mut sig_w: Vec<Wire> = Vec::new();
    for rr in &mp.anchor_rounds {
        let r_w = chw(&outs, &trace.squeezes, rr.fin);
        sig_w.push(r_w);
        acl = sb.gate(mrslot, &[acl, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
    }
    let m_mp = mp.rounds.len();
    assert_eq!(sig_w.len(), 2 * (m_mp + 1), "sigma spans the anchor layers");

    // The pl_full prefix slot (created in the close-out above); a chunked
    // product of (1 + a + b) factors, seed-chained across rows, padded
    // factors (zw, zw) = 1.
    let pfslot = leaf_slot
        .iter()
        .find(|(n, _)| *n == 310 + pl_full)
        .expect("the close-out created the prefix slot")
        .1;
    let prefix_product = |sb: &mut ShapeBuilder, factors: &[(Wire, Wire)]| -> Wire {
        let mut seed = ow;
        for chunk in factors.chunks(pl_full) {
            let mut g_in = vec![seed];
            for (a, _) in chunk {
                g_in.push(*a);
            }
            g_in.extend(repeat_n(zw, pl_full - chunk.len()));
            for (_, b) in chunk {
                g_in.push(*b);
            }
            g_in.extend(repeat_n(zw, pl_full - chunk.len()));
            g_in.push(ow);
            seed = sb.gate(pfslot, &g_in)[0];
        }
        seed
    };

    // 3b: e_at = eq(rho, rho'') — rho is the W-round challenge wires.
    let rho_w: Vec<Wire> = w_rounds
        .iter()
        .map(|rr| chw(&outs, &trace.squeezes, rr.fin))
        .collect();
    let factors: Vec<(Wire, Wire)> = rho_w.iter().copied().zip(rho2_w.iter().copied()).collect();
    let e_at_w = prefix_product(&mut sb, &factors);

    // 3c: the expect. Groups by shared row part (structural for this fixed
    // shape — asserted one dual value per group), run structure from the
    // inner's jagged boundaries (pub, same source as the verifier).
    let n_log_i = INNER_NU;
    let k_cols_i = gammas[0].pt_len - n_log_i;
    let mut groups_ix: Vec<Vec<usize>> = Vec::new();
    for (i, pt) in pd_pts.iter().enumerate() {
        match groups_ix
            .iter_mut()
            .find(|g| pd_pts[g[0]][..n_log_i] == pt[..n_log_i])
        {
            Some(g) => g.push(i),
            None => groups_ix.push(vec![i]),
        }
    }
    assert_eq!(groups_ix.len(), mp.val_vs.len(), "one dual value per group");
    let params_i = JaggedParams::from_heights(&inner_union.jagged_heights(), n_log_i, m_mp);
    let bounds = assist_boundaries(&params_i);
    // Shape assumption: singleton used-column runs, plus AT MOST one
    // zero-height tail run (absent at full utilization — this inner).
    let n_runs = bounds.len();
    let has_tail = bounds[n_runs - 1].0 == bounds[n_runs - 1].1;
    let n_single = if has_tail { n_runs - 1 } else { n_runs };
    for &(_, _, len) in &bounds[..n_single] {
        assert_eq!(len, 1, "used columns are singleton runs");
    }

    // Per-run boundary eq products at sigma (statement-independent).
    let eqc: Vec<Wire> = bounds
        .iter()
        .map(|&(t_c, t_next, _)| {
            let mut factors = Vec::with_capacity(2 * (m_mp + 1));
            for l in 0..=m_mp {
                factors.push((sig_w[2 * l], if (t_c >> l) & 1 == 1 { ow } else { zw }));
                factors.push((
                    sig_w[2 * l + 1],
                    if (t_next >> l) & 1 == 1 { ow } else { zw },
                ));
            }
            prefix_product(&mut sb, &factors)
        })
        .collect();

    // Per (claim, singleton run) column-eq sums, the tail by char-2
    // complement (total eq mass is 1), then the gamma_pd combination and
    // the per-statement dot + DP + coefficient.
    let alslot = sb.slot(AssistLayerGate::new());
    leaf_slot.push((601, alslot));
    let mut expect = zw;
    for (g_ix, members) in groups_ix.iter().enumerate() {
        let mut run_w: Vec<Wire> = vec![zw; n_runs];
        for &i in members {
            let pd = &gammas[i];
            let gpd_w = chw(&outs, &trace.squeezes, pd.fin);
            let mut tail = ow;
            for r in 0..n_single {
                let y = r as u64;
                let factors: Vec<(Wire, Wire)> = (0..k_cols_i)
                    .map(|j| {
                        (
                            wv(pd.pt_v + n_log_i + j),
                            if (y >> j) & 1 == 1 { ow } else { zw },
                        )
                    })
                    .collect();
                let s = prefix_product(&mut sb, &factors);
                tail = sb.gate(macslot, &[tail, s, ow])[0];
                run_w[r] = sb.gate(macslot, &[run_w[r], gpd_w, s])[0];
            }
            if has_tail {
                run_w[n_runs - 1] = sb.gate(macslot, &[run_w[n_runs - 1], gpd_w, tail])[0];
            }
        }
        let mut w_st = zw;
        for (r, &rw) in run_w.iter().enumerate() {
            w_st = sb.gate(macslot, &[w_st, rw, eqc[r]])[0];
        }
        let mut g = [zw, zw, ow, zw]; // STATE_SUCCESS seed
        let row0 = pd_pts[members[0]].len(); // silence: row wires below
        let _ = row0;
        for layer in (0..=m_mp).rev() {
            let za = if layer < n_log_i {
                wv(gammas[members[0]].pt_v + layer)
            } else {
                zw
            };
            let rb = if layer < m_mp { rho2_w[layer] } else { zw };
            let mut a_in = g.to_vec();
            a_in.extend_from_slice(&[za, rb, sig_w[2 * layer], sig_w[2 * layer + 1], ow]);
            let o = sb.gate(alslot, &a_in);
            g = [o[0], o[1], o[2], o[3]];
        }
        let coeff = sb.gate(macslot, &[zw, pws[g_ix], e_at_w])[0];
        let wd = sb.gate(macslot, &[zw, w_st, g[0]])[0];
        expect = sb.gate(macslot, &[expect, coeff, wd])[0];
    }
    // 3d: the join — the anchor's folded claim equals the expect.
    let delta_anchor = sb.gate(macslot, &[acl, expect, ow])[0];

    // ---- the PoW bit predicate (boundary pattern) ----
    // Per Pow op, publish (state-digest words, nonce word): the digest is
    // the chain finalize's first two output words, the nonce its aligned
    // stream word. The checker recomputes H(digest ‖ nonce) natively and
    // applies the leading-zero predicate — the same trust structure as the
    // alpha expansion and the cap select.
    let pow_pub: Vec<[Wire; 3]> = pows
        .iter()
        .map(|pr| {
            let sq = &trace.squeezes[pr.fin];
            let wi = stream
                .words
                .iter()
                .position(|w| matches!(w, StreamWord::Bytes { payload, .. } if *payload == pr.pay))
                .expect("pow nonce stream word");
            let nw = word_wire[wi].expect("pow nonce wired");
            [outs[sq[0]][0], outs[sq[0]][1], nw]
        })
        .collect();

    // ---- ASSERTION EMISSION: the ElementAssertion exits as bound publics.
    // For this one-slot inner every field is already a wire: alpha (chain),
    // r_con = the zerocheck challenges, r_col = the lincheck challenges
    // (chain squeezes), evals = the ZcJoin-derived (va, vb) — the
    // strip_constants values the native assertion carries — z_eval = the
    // element c claim's absorbed value (ec-joined to the zerocheck), and
    // target = lc_target (published below with the PIOP tail). The
    // accumulator can reconstruct the assertion from the public segment
    // alone. Multi-slot inners absorb per-slot eval pairs — a shape
    // extension for the mixed phase.
    let mut assertion_pub: Vec<Wire> = vec![alpha_w];
    for rr in &piop.zc_rounds {
        assertion_pub.push(chw(&outs, &trace.squeezes, rr.fin));
    }
    for rr in &piop.lc_rounds {
        assertion_pub.push(chw(&outs, &trace.squeezes, rr.fin));
    }
    assertion_pub.extend_from_slice(&[va_w, vb_w, wv(gammas[0].val_v)]);

    for (a_wires, opens) in &to_publish {
        for w in a_wires {
            sb.publish(*w);
        }
        for (cw, cv) in opens {
            sb.publish(*cw);
            sb.publish(cv[0]);
            sb.publish(cv[1]);
        }
    }
    sb.publish(t_final);
    sb.publish(running_w);
    for d in &zc_deltas {
        sb.publish(*d);
    }
    sb.publish(zc_fin_delta);
    sb.publish(ec_join);
    sb.publish(lc_target);
    for accs in &resid_pub {
        for w in accs {
            sb.publish(*w);
        }
    }
    if let Some(w) = inner_w {
        sb.publish(w);
    }
    sb.publish(delta_tm);
    sb.publish(delta_rq);
    sb.publish(delta_anchor);
    for p in &pow_pub {
        for w in p {
            sb.publish(*w);
        }
    }
    for w in &assertion_pub {
        sb.publish(*w);
    }
    let shape = sb.finish().expect("valid real-query circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();
    let (built, online_t) = timed(REPS, || shape.run(&vals, &hint_refs));

    // ---- the boundary checks ----
    // The tail publics: three multipoint zero-deltas (T_m == anchor.v,
    // running_W == q_eval·V, claim == expect), per-Pow (digest word0,
    // digest word1, nonce word) triples the checker validates natively,
    // then the emitted ElementAssertion fields.
    let n_assert = 1 + piop.zc_rounds.len() + piop.lc_rounds.len() + 3;
    let assert_base = built.public.len() - n_assert;
    {
        let vals_rec = rec.values();
        let mut at = assert_base;
        assert_eq!(built.public[at], chals[piop.alpha_ch], "assertion alpha");
        at += 1;
        for rr in &piop.zc_rounds {
            assert_eq!(built.public[at], chals[rr.ch], "assertion r_con");
            at += 1;
        }
        for rr in &piop.lc_rounds {
            assert_eq!(built.public[at], chals[rr.ch], "assertion r_col");
            at += 1;
        }
        // evals = (va, vb) are strip_constants derivations — validated
        // transitively by the lincheck target equality; z_eval is the
        // absorbed c-claim value.
        assert_eq!(
            built.public[at + 2],
            vals_rec[gammas[0].val_v],
            "assertion z_eval"
        );
    }
    let pow_base = assert_base - 3 * pows.len();
    for (i, off) in [3, 2, 1].into_iter().enumerate() {
        assert_eq!(
            built.public[pow_base - off],
            F128::ZERO,
            "multipoint zero-delta {i}"
        );
    }
    for (i, pr) in pows.iter().enumerate() {
        let d0 = built.public[pow_base + 3 * i];
        let d1 = built.public[pow_base + 3 * i + 1];
        let nn = built.public[pow_base + 3 * i + 2];
        let mut digest = [0u8; 32];
        digest[..8].copy_from_slice(&d0.lo.to_le_bytes());
        digest[8..16].copy_from_slice(&d0.hi.to_le_bytes());
        digest[16..24].copy_from_slice(&d1.lo.to_le_bytes());
        digest[24..].copy_from_slice(&d1.hi.to_le_bytes());
        assert_eq!(nn.hi, 0, "pow {i}: nonce word is 8 bytes zero-padded");
        if pr.bits == 0 {
            assert_eq!(nn.lo, 0, "pow {i}: canonical zero nonce");
        } else {
            assert!(
                pow_has_leading_zero_bits(&digest, nn.lo, pr.bits, HashKind::Blake3,),
                "pow {i}: grinding predicate on the published wires"
            );
        }
    }
    let yr_pub = levels.len() * yr_len
        + usize::from(closeout)
        + 1
        + piop.zc_rounds.len()
        + 3
        + 3
        + 3 * pows.len()
        + n_assert;
    let total_pub: usize = 1
        + yr_pub
        + levels
            .iter()
            .zip(&geo)
            .map(|(l, g)| l.a_count + 3 * g.q)
            .sum::<usize>();
    let mut at = built.public.len() - total_pub;
    for (li, lvl) in levels.iter().enumerate() {
        let g = &geo[li];
        let (cap, _, _) = lvl_src[li];
        for j in 0..lvl.a_count {
            assert_eq!(built.public[at + j], chals[lvl.a_ch + j], "L{li} alpha {j}");
        }
        at += lvl.a_count;
        for k in 0..g.q {
            let chal = chals[lvl.q_ch + k];
            assert_eq!(built.public[at], chal, "L{li} challenge {k}");
            let pos = (chal.lo as usize) & ((1usize << g.depth) - 1);
            let node = digest_words(&hash_to_digest(&cap[pos >> g.path]));
            assert_eq!(
                [built.public[at + 1], built.public[at + 2]],
                node,
                "L{li} cap node {k}"
            );
            at += 3;
        }
    }
    // The spine, replayed natively over the recorded transcript: same quad
    // math, same start target, native enforced sums. Equality transitively
    // validates every eval/build/fold gate AND the LeafEval accumulators.
    let vals_rec = rec.values();
    let quad = |u0: F128, u2: F128, t: F128| (u0, t + u2, u2);
    let evalq = |q: (F128, F128, F128), x: F128| q.0 + x * q.1 + x * x * q.2;
    // The bound start target: gamma' * q_eval.
    let mut nt = chals[inner_pd.ch] * vals_rec[inner_pd.q_v];
    let mut nq = quad(vals_rec[start_v], vals_rec[start_v + 1], nt);
    for (li, lvl) in levels.iter().enumerate() {
        for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
            nt = evalq(nq, chals[lvl.fold_chs[j]]);
            nq = quad(vals_rec[mv], vals_rec[mv + 1], nt);
        }
        if li < levels.len() - 1 {
            for od in &lvl.ood {
                let b = chals[od.beta_ch];
                let iq = quad(
                    vals_rec[od.intro_v],
                    vals_rec[od.intro_v + 1],
                    vals_rec[od.y_v],
                );
                nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                nt += b * vals_rec[od.y_v];
            }
            let b = chals[lvl.beta_ch];
            let iq = quad(
                vals_rec[lvl.intro_v],
                vals_rec[lvl.intro_v + 1],
                native_sums[li],
            );
            nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
            nt += b * native_sums[li];
        } else {
            nt += chals[lvl.beta_ch] * native_sums[li];
        }
    }
    assert_eq!(built.public[at], nt, "the spine's final t_r");
    at += 1;
    // The merged rounds, natively: outer gamma-combination through
    // fold_round_claim. The native verify already enforced
    // `running == q_eval * v` (the assist stays native), so the published
    // running matching this replica closes the outer chain.
    {
        let mut mt = F128::ZERO;
        for pd in &gammas {
            mt += chals[pd.ch] * vals_rec[pd.val_v];
        }
        for rr in &w_rounds {
            let (g1, gi, rch) = (vals_rec[rr.g_v], vals_rec[rr.g_v + 1], chals[rr.ch]);
            mt = (mt + g1) + (mt + gi) * rch + gi * (rch * rch);
        }
        assert_eq!(built.public[at], mt, "the merged running claim");
        at += 1;
    }
    // The PIOP: every zero-delta must BE zero, and the lincheck target must
    // match a native replay (which is the deferred assertion's target).
    {
        for i in 0..piop.zc_rounds.len() {
            assert_eq!(built.public[at + i], F128::ZERO, "zc delta {i}");
        }
        at += piop.zc_rounds.len();
        assert_eq!(built.public[at], F128::ZERO, "zc final delta");
        at += 1;
        assert_eq!(built.public[at], F128::ZERO, "ec join");
        at += 1;
        // native lincheck replay
        let mut running = F128::ZERO;
        for (i, rr) in piop.zc_rounds.iter().enumerate() {
            let (g1, gi) = (rec.values()[rr.g_v], rec.values()[rr.g_v + 1]);
            let (t, rho) = (chals[piop.tau_ch + i], chals[rr.ch]);
            let g0 = (running + t * g1) * (F128::ONE + t).inv();
            running = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        let (ea, eb, ec) = (
            rec.values()[piop.eab_v],
            rec.values()[piop.eab_v + 1],
            rec.values()[piop.eab_v + 2],
        );
        assert_eq!(running, ea * eb + ec, "native zc closes");
        let nzc = piop.zc_rounds.len();
        let (r0, r1) = (
            chals[piop.zc_rounds[nzc - 2].ch],
            chals[piop.zc_rounds[nzc - 1].ch],
        );
        let eqt = [
            (F128::ONE + r0) * (F128::ONE + r1),
            r0 * (F128::ONE + r1),
            (F128::ONE + r0) * r1,
            r0 * r1,
        ];
        let (mut va, mut vb) = (ea, eb);
        for c in 0..4 {
            va += eqt[c] * inner_ty.a_const()[c];
            vb += eqt[c] * inner_ty.b_const()[c];
        }
        let mut lrn = va + chals[piop.alpha_ch] * vb;
        for rr in &piop.lc_rounds {
            let (e1, ei) = (rec.values()[rr.g_v], rec.values()[rr.g_v + 1]);
            let rho = chals[rr.ch];
            let e0 = lrn + e1;
            lrn = ei * rho * rho + (e0 + e1 + ei) * rho + e0;
        }
        assert_eq!(built.public[at], lrn, "the deferred assertion's target");
        at += 1;
    }
    // Native replica of induce_sumcheck_evaluate_at_residual, per level.
    let mut resid_native: Vec<Vec<F128>> = vec![vec![F128::ZERO; yr_len]; levels.len()];
    for (li, lvl) in levels.iter().enumerate() {
        let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len()).sum();
        let lmc = pl + yr_log;
        let sks = sk_at_vks(lmc);
        let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
        let ris: Vec<F128> = levels[li + 1..]
            .iter()
            .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
            .collect();
        let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
        let aw = build_eq_table(&alpha_vals);
        for y in 0..yr_len {
            let mut sum = F128::ZERO;
            for k in 0..geo[li].q {
                let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << geo[li].depth) - 1);
                let mut sk = Vec::with_capacity(lmc);
                if lmc > 0 {
                    sk.push(F128::new(pos as u64, 0));
                    for j in 1..lmc {
                        sk.push(sk[j - 1] * sk[j - 1] + sks[j - 1] * sk[j - 1]);
                    }
                }
                let mut prod = F128::ONE;
                for j in 0..pl {
                    prod *= F128::ONE + ris[j] * (F128::ONE + sk[j] * inv(sks[j]));
                }
                for j in 0..yr_log {
                    if (y >> j) & 1 == 1 {
                        prod *= sk[pl + j] * inv(sks[pl + j]);
                    }
                }
                sum += aw[k] * prod;
            }
            assert_eq!(built.public[at], sum, "L{li} residual y={y}");
            resid_native[li][y] = sum;
            at += 1;
        }
    }
    // evb + combine, natively: gamma-weighted char-2 eq products, then the
    // yr dot. The published inner must match; it equals the TRUE t_r of the
    // native verify (which accepted), while the spine's t_final still starts
    // from the unbound T0 — the merged intake (2c) closes that gap.
    if closeout {
        let ris_v: Vec<F128> = levels
            .iter()
            .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
            .collect();
        let pl_full = ris_v.len();
        let mut inner_n = F128::ZERO;
        for y in 0..yr_len {
            let mut evb = chals[inner_pd.ch];
            for j in 0..pl_full {
                evb *= F128::ONE + chals[w_rounds[j].ch] + ris_v[j];
            }
            for j in 0..yr_log {
                evb *= if (y >> j) & 1 == 1 {
                    chals[w_rounds[pl_full + j].ch]
                } else {
                    F128::ONE + chals[w_rounds[pl_full + j].ch]
                };
            }
            let mut comb = evb;
            for (li, lvl) in levels.iter().enumerate() {
                comb += chals[lvl.beta_ch] * resid_native[li][y];
                for od in &lvl.ood {
                    let folded = od.z_len - yr_log;
                    let later: Vec<F128> = levels[li + 1..]
                        .iter()
                        .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
                        .collect();
                    let mut t = chals[od.beta_ch];
                    for j in 0..folded {
                        t *= F128::ONE + chals[od.z_ch + j] + later[j];
                    }
                    for j in 0..yr_log {
                        t *= if (y >> j) & 1 == 1 {
                            chals[od.z_ch + folded + j]
                        } else {
                            F128::ONE + chals[od.z_ch + folded + j]
                        };
                    }
                    comb += t;
                }
            }
            inner_n += rec.values()[yr_v + y] * comb;
        }
        assert_eq!(built.public[at], inner_n, "the close-out inner");
        // THE CLOSURE: with the start target bound, the spine's t_r and the
        // residual side's inner are the same statement scalar — the native
        // verifier's final check, now enforced between two circuit outputs.
        assert_eq!(built.public[at], nt, "inner == t_r: the statement closes");
    }

    // ---- prove / verify the circuit itself ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let b3 = blake3::build_block_r1cs(nu);
    let b3_lc = b3.csc_lincheck_circuit();
    let swap_r1cs = SwapTable::build_block_r1cs(nu);
    let swap_lc = swap_r1cs.csc_lincheck_circuit();
    let spread_ty = BitSpreadTable::new(max_path);
    let spread_r1cs = spread_ty.build_block_r1cs(nu);
    let spread_lc = spread_r1cs.csc_lincheck_circuit();

    let (b3_wit, wit_t) = timed(3, || {
        blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(slots.b3), nu)
    });
    let swap_wit = SwapTable::generate_witness_batch_major(built.rows::<SwapGate>(slots.swap), nu);
    let spread_wit =
        spread_ty.generate_witness_batch_major(built.rows::<BitSpreadGate>(slots.spread), nu);
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();

    let mut lcs_ord: Vec<(usize, &dyn LincheckCircuit)> = vec![
        (shape.registry_slot(slots.b3), b3_lc),
        (shape.registry_slot(slots.swap), swap_lc),
        (shape.registry_slot(slots.spread), spread_lc),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn LincheckCircuit> = lcs_ord.into_iter().map(|(_, c)| c).collect();

    let ((cproof, ccommitment), prove_t) = timed(REPS, || {
        let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape.registry_slot(slots.b3),
                UnionSlotProverInput::new(b3_wit.clone(), b3_lc),
            ),
            (
                shape.registry_slot(slots.swap),
                UnionSlotProverInput::new(swap_wit.clone(), swap_lc),
            ),
            (
                shape.registry_slot(slots.spread),
                UnionSlotProverInput::new(spread_wit.clone(), spread_lc),
            ),
        ];
        bool_slots.sort_by_key(|(i, _)| *i);
        let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
            .iter()
            .zip(els.clone())
            .map(|((_, s), z)| (shape.registry_slot(*s), z))
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(_, z)| {
                UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z))
            })
            .collect();
        let mut c = FsChallenger::new(DOMAIN);
        let (cproof, ccommitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            bool_slots.into_iter().map(|(_, s)| s).collect(),
            el_inputs,
            &mut c,
        );
        (cproof, ccommitment)
    });

    let (_, verify_t) = timed(REPS, || {
        let mut c = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &ccommitment,
            &cproof,
            &pcs_params,
            &mut c,
        )
        .expect("the real query phase verifies")
    });

    println!(
        "\nMVP-7 REAL QUERY PHASE (inner: pure-element m=22, BLAKE3)\n  \
         inner prove {inner_prove_ms:.0} ms | levels {:?} (q, depth, cap, path)\n  \
         blake3 {} rows ({} chain) | swap {} | spread {} | leaf-eval slots {:?}\n  \
         nu {nu} | dense_m {} | mu {}\n\n  \
         medians of {REPS} runs, spread in brackets\n  \
         PER PROOF     online {online_t} + witgen {wit_t} + prove {prove_t} ms\n  \
         verifier side {verify_t} ms | proof {:.1} KiB | {threads} threads\n  \
         SETUP         {setup_ms:6.0} ms\n",
        geo.iter()
            .map(|g| (g.q, g.depth, g.c, g.path))
            .collect::<Vec<_>>(),
        shape.counts[shape.registry_slot(slots.b3)],
        trace.rows.len(),
        shape.counts[shape.registry_slot(slots.swap)],
        shape.counts[shape.registry_slot(slots.spread)],
        leaf_slot.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        union.dense_m(),
        shape.circuit.cells().mu(),
        bincode::serialize(&cproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
    );
}

// ---------------------------------------------------------------------------

use flock_prover::r1cs_hashes::merkle_glue::{BitSpreadTable, SwapInput, SwapTable};

/// One Merkle level's conditional swap. The sibling is a [`GateType::Hint`] —
/// it is not word-aligned-wireable in the composite and nothing else reads it
/// here either, so it stays free witness.
struct SwapGate {
    nu: usize,
}

impl GateType for SwapGate {
    type Row = SwapInput;
    type Hint = [u32; SLOT_WORDS];

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&SwapTable::build_block_r1cs(self.nu))
            .with_io_schema(SwapTable::io_schema())
    }

    fn eval(&self, inputs: &[F128], hint: &Self::Hint) -> (Vec<F128>, Self::Row) {
        let row = SwapInput {
            bit_word: (inputs[0].lo as u128) | ((inputs[0].hi as u128) << 64),
            prev: unpack8(inputs[1], inputs[2]),
            sib: *hint,
        };
        let (left, right) = SwapTable::outputs(&row);
        let (lw, rw) = (digest_words(&left), digest_words(&right));
        (vec![lw[0], lw[1], rw[0], rw[1]], row)
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// Relocate each of the index word's low `depth` bits into its own word, so a
/// per-level swap row can read it at the one column its uniform relation is
/// allowed to look at.
struct BitSpreadGate {
    ty: BitSpreadTable,
    nu: usize,
}

impl GateType for BitSpreadGate {
    type Row = u128;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&self.ty.build_block_r1cs(self.nu))
            .with_io_schema(self.ty.io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &()) -> (Vec<F128>, u128) {
        let idx = (inputs[0].lo as u128) | ((inputs[0].hi as u128) << 64);
        let outs = (0..self.ty.depth)
            .map(|l| F128::new(((idx >> l) & 1) as u64, 0))
            .collect();
        (outs, idx)
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// The three slots a collapsed opening writes into.
#[derive(Clone, Copy)]
struct CollapsedSlots {
    b3: SlotId,
    swap: SlotId,
    spread: SlotId,
}

/// Emit one Merkle opening as rows of the shipped BLAKE3 table plus glue,
/// wired together. Returns the two words of the root.
///
/// This is what replaces a composite row. The dataflow, per level:
///
/// ```text
///   index word ─▶ BitSpread ─bit_l─▶ Swap ─left‖right─▶ BLAKE3 ─out_lo─▶ next Swap.prev
/// ```
///
/// and before it, the chunk chain: `leaf_blocks` BLAKE3 rows whose out_lo
/// threads row to row, seeded by the IV, `CHUNK_START` on the first and
/// `CHUNK_END` on the last. Every arrow is a copy constraint on whole words.
#[allow(clippy::too_many_arguments)]
fn emit_opening(
    sb: &mut ShapeBuilder,
    s: CollapsedSlots,
    iv: [Wire; 2],
    leaf_w: &[Wire],
    index_w: Wire,
    depth: usize,
    cap_depth: usize,
    pubs: &mut Vec<F128>,
) -> [Wire; 2] {
    let blocks = leaf_w.len() / 4;
    assert_eq!(leaf_w.len(), 4 * blocks, "leaf is whole 64-byte blocks");

    // The index word's bits, one per level.
    let bits = sb.gate(s.spread, &[index_w]);

    // Chunk chain: the leaf hashed as a BLAKE3 chunk.
    let mut cv = iv;
    for i in 0..blocks {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i + 1 == blocks {
            flags |= CHUNK_END;
        }
        pubs.push(pack_params(0, 64, flags));
        let params = sb.public_input();
        let out = sb.gate(
            s.b3,
            &[
                cv[0],
                cv[1],
                leaf_w[4 * i],
                leaf_w[4 * i + 1],
                leaf_w[4 * i + 2],
                leaf_w[4 * i + 3],
                params,
            ],
        );
        cv = [out[0], out[1]];
    }

    // Node levels: swap, then a PARENT compression over the swapped pair.
    // The sibling is the swap's hint, supplied at `run` time in this call
    // order — setup has no values. Under capping the fold stops `cap_depth`
    // levels below the root: the returned digest is the depth-`cap_depth`
    // ancestor, which the CHECKER compares against the absorbed cap layer
    // (the boundary select — the circuit never touches the cap words).
    // `cap_depth = 0` is the uncapped statement, terminal = root.
    for l in 0..(depth - cap_depth) {
        let sw = sb.gate_hinted(s.swap, &[bits[l], cv[0], cv[1]]);
        pubs.push(pack_params(0, 64, PARENT));
        let params = sb.public_input();
        let out = sb.gate(s.b3, &[iv[0], iv[1], sw[0], sw[1], sw[2], sw[3], params]);
        cv = [out[0], out[1]];
    }
    cv
}

/// **The collapse, one opening at a time.** Emit an opening as BLAKE3 rows
/// plus glue and check the root it computes is the one the composite computes
/// — for the real L0 shape and a small one, at every index polarity that
/// exercises both swap directions.
///
/// If this holds, a composite row and `leaf_blocks + depth` wired rows are
/// interchangeable, and the whole collapse is a matter of scale.
#[test]
#[ignore] // Builds real BLAKE3 matrices. `-- --ignored`.
fn collapsed_opening_matches_the_composite() {
    for (depth, leaf_bytes, n_open) in [(3usize, 128usize, 8usize), (14, 1024, 3)] {
        let blocks = leaf_bytes / 64;
        let mut rng = Rng(0x_C0_11_09_5E ^ depth as u64);
        let tree = Tree::new(depth, leaf_bytes, &mut rng);
        // Rows: one bit-spread + `blocks + depth` BLAKE3 + `depth` swaps each.
        let nu = (((blocks + depth) * n_open)
            .next_power_of_two()
            .trailing_zeros() as usize)
            .max(3);

        let mut sb = ShapeBuilder::new(nu);
        let slots = CollapsedSlots {
            b3: sb.slot(Blake3Gate { nu }),
            swap: sb.slot(SwapGate { nu }),
            spread: sb.slot(BitSpreadGate {
                ty: BitSpreadTable::new(depth),
                nu,
            }),
        };

        let mut pubs: Vec<F128> = Vec::new();
        let iv_w = pack8(&IV);
        pubs.push(iv_w[0]);
        pubs.push(iv_w[1]);
        let iv = [sb.public_input(), sb.public_input()];

        let positions: Vec<usize> = (0..n_open).map(|i| (i * 5 + 1) % (1 << depth)).collect();
        let mut leaf_vals: Vec<F128> = Vec::new();
        let mut idx_vals: Vec<F128> = Vec::new();
        let mut roots = Vec::new();
        for &pos in &positions {
            let leaf_w: Vec<Wire> = (0..4 * blocks).map(|_| sb.input()).collect();
            let index_w = sb.input();
            roots.push(emit_opening(
                &mut sb, slots, iv, &leaf_w, index_w, depth, 0, &mut pubs,
            ));
            let leaf = tree.leaf(pos);
            leaf_vals.extend((0..4 * blocks).map(|w| leaf_word(leaf, 16 * w)));
            idx_vals.push(F128::new(pos as u64, 0));
        }
        for r in &roots {
            sb.publish(r[0]);
            sb.publish(r[1]);
        }
        let shape = sb.finish().expect("valid collapsed circuit");

        // Inputs in declaration order: iv, then per opening (params are
        // public_inputs emitted INSIDE emit_opening, already in `pubs`), and
        // the leaf/index free wires. Rebuild that order exactly.
        let mut vals: Vec<F128> = vec![iv_w[0], iv_w[1]];
        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        for (i, &pos) in positions.iter().enumerate() {
            let leaf = tree.leaf(pos);
            vals.extend((0..4 * blocks).map(|w| leaf_word(leaf, 16 * w)));
            vals.push(idx_vals[i]);
            // Then `emit_opening`'s own public params, in its call order.
            for b in 0..blocks {
                let mut f = 0u32;
                if b == 0 {
                    f |= CHUNK_START;
                }
                if b + 1 == blocks {
                    f |= CHUNK_END;
                }
                vals.push(pack_params(0, 64, f));
            }
            for _ in 0..depth {
                vals.push(pack_params(0, 64, PARENT));
            }
            hints.extend(tree.siblings(pos));
        }
        let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();
        let built = shape.run(&vals, &hint_refs);

        // Every opening's root is the tree's — i.e. the collapsed rows fold
        // exactly as `root_chunk` does.
        let layout = MerkleTreeLayout::with_blake3_chunk_leaf(depth, leaf_bytes, blake3_spec());
        let want = digest_words(&hash_to_digest(&tree.root));
        let base = built.public.len() - 2 * n_open;
        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(
                [built.public[base + 2 * i], built.public[base + 2 * i + 1]],
                want,
                "depth {depth} opening {i} (pos {pos}) root"
            );
            // ...and it agrees with the composite on the same input.
            let composite = layout.root_chunk(&ChunkPathInput {
                leaf_data: tree.leaf(pos).to_vec(),
                index: pos as u128,
                siblings: tree.siblings(pos),
            });
            assert_eq!(
                digest_words(&composite),
                want,
                "depth {depth} opening {i}: composite disagrees with the tree"
            );
        }
        println!(
            "  depth {depth} leaf {leaf_bytes}: {n_open} openings = {} blake3 + {} swap + \
             {} spread rows, nu {nu}, mu {}",
            shape.counts[shape.registry_slot(slots.b3)],
            shape.counts[shape.registry_slot(slots.swap)],
            shape.counts[shape.registry_slot(slots.spread)],
            shape.circuit.cells().mu(),
        );
    }
}

/// **MVP-6: the full query phase, collapsed.** The same four levels as
/// [`mvp5_all_levels_query_phase`], with every Merkle composite replaced by
/// wiring over ONE BLAKE3 table.
///
/// The prediction from `wiring_scaling.rs` and the glue tables' measured nnz:
/// lincheck ~105.1M → ~21.0M, wiring ~1.7 → ~18 ms, prove ~174 → ~119 ms.
/// Note the FS chain's compressions and the openings' share the SAME slot —
/// that is the point.
///
/// ONE `BitSpreadTable`, sized for the deepest level; shallower levels leave
/// its extra outputs unwired, and an unwired schema word is sigma-fixed and
/// costs nothing. Four depth-specific spread tables would have been four more
/// types, which is the thing being removed.
///
/// **Capped openings (the real protocol since Merkle capping)**: each level
/// absorbs its depth-c cap layer (`c = cap_depth(q, d)`) instead of a root,
/// paths fold only `d − c` levels, and the terminal digest is bound by the
/// BOUNDARY SELECT — the circuit publishes `(challenge, digest)` per query
/// and the checker compares the digest against `cap[pos >> (d − c)]`
/// natively. The c-bit select costs the circuit nothing; the price is one
/// extra published word per query and ~18 KiB more FS absorb (the caps),
/// against ~3.3k fewer BLAKE3 rows and ~3.3k fewer swaps.
#[test]
#[ignore] // The full shape. `-- --ignored`.
fn mvp6_all_levels_collapsed() {
    use flock_core::challenger::Challenger as _;
    #[cfg(test)]
    use flock_core::lincheck::build_eq_table;
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp};
    use flock_prover::prover::UnionElementSlotInput;
    use flock_prover::r1cs_hashes::fs_chain::{CvSource, FsChain};
    use std::time::Instant;

    const SLICE: &[u8] = b"flock-mvp6-collapsed-v0";
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let levels = [
        Level {
            depth: 14,
            lanes: 64,
            queries: 218,
        },
        Level {
            depth: 12,
            lanes: 8,
            queries: 106,
        },
        Level {
            depth: 10,
            lanes: 8,
            queries: 71,
        },
        Level {
            depth: 8,
            lanes: 8,
            queries: 53,
        },
    ];
    let max_depth = levels.iter().map(|l| l.depth).max().unwrap();

    let mut rng = Rng(0x_5EED_0006);
    let trees: Vec<Tree> = levels
        .iter()
        .map(|l| Tree::new(l.depth, 16 * l.lanes, &mut rng))
        .collect();

    // ---- transcript ----
    // Capping: the commitment is the depth-c cap layer, absorbed flattened
    // where the root used to be. The synthetic trees are core merkle trees,
    // so `cap_layer` reads the caps straight out of them.
    let cap_depths: Vec<usize> = levels
        .iter()
        .map(|l| core_merkle::cap_depth(l.queries, l.depth))
        .collect();
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(SLICE, HashKind::Blake3));
    let mut chals: Vec<Vec<F128>> = Vec::new();
    let mut want: Vec<Vec<usize>> = Vec::new();
    for (li, (l, tree)) in levels.iter().zip(&trees).enumerate() {
        let cap = core_merkle::cap_layer(&tree.flat, 1 << l.depth, cap_depths[li]);
        rec.observe_bytes(cap.as_flattened());
        let cs = rec.sample_f128_vec(l.queries);
        want.push(
            cs.iter()
                .map(|v| (v.lo as usize) & ((1usize << l.depth) - 1))
                .collect(),
        );
        chals.push(cs);
    }
    let t_shape = rec.shape();
    let stream = t_shape.stream_words(SLICE);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());

    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin_ops: Vec<&TranscriptOp> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin_ops[i].squeezed_bytes());
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();

    // Rows in the ONE blake3 slot: the FS chain plus every opening's
    // compressions.
    let b3_rows: usize = trace.rows.len()
        + levels
            .iter()
            .zip(&cap_depths)
            .map(|(l, &c)| (16 * l.lanes / 64 + l.depth - c) * l.queries)
            .sum::<usize>();
    let nu = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);

    // ---- setup ----
    let t = Instant::now();
    let mut sb = ShapeBuilder::new(nu);
    let slots = CollapsedSlots {
        b3: sb.slot(Blake3Gate { nu }),
        swap: sb.slot(SwapGate { nu }),
        spread: sb.slot(BitSpreadGate {
            ty: BitSpreadTable::new(max_depth),
            nu,
        }),
    };
    let mut leaf_slot: Vec<(usize, SlotId)> = Vec::new();
    let leafeval: Vec<_> = levels
        .iter()
        .map(|l| match leaf_slot.iter().find(|(n, _)| *n == l.lanes) {
            Some((_, s)) => *s,
            None => {
                let s = sb.slot(LeafEvalGate::new(l.lanes));
                leaf_slot.push((l.lanes, s));
                s
            }
        })
        .collect();

    // Values are pushed in DECLARATION order throughout; `emit_opening` pushes
    // its own params into the same vector as it declares them.
    let spine = sb.slot(SpineGate::new());
    leaf_slot.push((0, spine));

    let mut vals: Vec<F128> = Vec::new();
    let iv_w = pack8(&FS_CHAIN_IV);
    vals.extend_from_slice(&iv_w);
    let iv = [sb.public_input(), sb.public_input()];

    // The FS chain, into the same blake3 slot.
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        vals.push(pack_params(counter, blen, flags));
        let params = sb.public_input();
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(slots.b3, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        vals.push(F128::ZERO);
                        sb.public_input()
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                vals.push(F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                ));
                                let w = sb.public_input();
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(slots.b3, &g_in));
    }

    // The openings.
    let vees: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.lanes.trailing_zeros() as usize)
                .map(|_| F128::new(rng.next_u32() as u64 | 1, rng.next_u32() as u64))
                .collect()
        })
        .collect();
    let alphas: Vec<Vec<F128>> = levels
        .iter()
        .map(|l| {
            (0..l.queries)
                .map(|_| F128::new(rng.next_u32() as u64, rng.next_u32() as u64 | 1))
                .collect()
        })
        .collect();

    vals.push(F128::ZERO);
    let mut acc = sb.public_input();
    let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
    let mut all_opens: Vec<Vec<(Wire, [Wire; 2])>> = Vec::new();
    for (li, l) in levels.iter().enumerate() {
        let sq = &trace.squeezes[li];
        let c = cap_depths[li];
        let vars = l.lanes.trailing_zeros() as usize;
        vals.extend_from_slice(&vees[li]);
        let vs: Vec<Wire> = (0..vars).map(|_| sb.public_input()).collect();
        let blocks = 16 * l.lanes / 64;
        let mut opens = Vec::with_capacity(l.queries);
        for k in 0..l.queries {
            let pos = want[li][k];
            let leaf = trees[li].leaf(pos);
            vals.extend((0..4 * blocks).map(|w| leaf_word(leaf, 16 * w)));
            let leaf_w: Vec<Wire> = (0..4 * blocks).map(|_| sb.input()).collect();

            // The challenge word IS the index word — no masking gadget.
            let cw = outs[sq[k / 4]][k % 4];
            let cv = emit_opening(&mut sb, slots, iv, &leaf_w, cw, l.depth, c, &mut vals);
            opens.push((cw, cv));
            hints.extend(trees[li].siblings(pos).into_iter().take(l.depth - c));

            // The same leaf words feed the arithmetic.
            let mut a_in = leaf_w;
            a_in.extend_from_slice(&vs);
            vals.push(alphas[li][k]);
            a_in.push(sb.public_input());
            a_in.push(acc);
            acc = sb.gate(leafeval[li], &a_in)[0];
        }
        all_opens.push(opens);
    }
    // The boundary select: each opening publishes its challenge word and its
    // terminal digest. The checker derives the cap index from the challenge
    // natively and compares the digest against the absorbed cap — the c-bit
    // select never enters the circuit. Publishing `cw` (the same wire the
    // spread gate consumes) is what binds the high bits: no index bit floats.
    for opens in &all_opens {
        for (cw, cv) in opens {
            sb.publish(*cw);
            sb.publish(cv[0]);
            sb.publish(cv[1]);
        }
    }
    sb.publish(acc);
    let shape = sb.finish().expect("valid collapsed circuit");
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;

    // ---- online ----
    let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();
    let (built, online_t) = timed(REPS, || shape.run(&vals, &hint_refs));

    // Every opening folds to its cap node, and the accumulator is
    // enforced_sum — the root equality of the uncapped statement became a
    // per-query cap-node equality, checked HERE, natively: read the
    // published challenge word, mask, take the high c bits, index the cap.
    let mut at = built.public.len() - 1 - 3 * levels.iter().map(|l| l.queries).sum::<usize>();
    for (li, l) in levels.iter().enumerate() {
        let c = cap_depths[li];
        let cap = core_merkle::cap_layer(&trees[li].flat, 1 << l.depth, c);
        for k in 0..l.queries {
            assert_eq!(built.public[at], chals[li][k], "L{li} challenge {k}");
            let pos = (chals[li][k].lo as usize) & ((1usize << l.depth) - 1);
            assert_eq!(pos, want[li][k], "L{li} position {k}");
            let node = digest_words(&hash_to_digest(&cap[pos >> (l.depth - c)]));
            assert_eq!(
                [built.public[at + 1], built.public[at + 2]],
                node,
                "L{li} cap node {k}"
            );
            at += 3;
        }
    }
    let mut want_sum = F128::ZERO;
    for (li, l) in levels.iter().enumerate() {
        let eq = build_eq_table(&vees[li]);
        for k in 0..l.queries {
            let leaf = trees[li].leaf(want[li][k]);
            let dot = (0..l.lanes)
                .map(|w| leaf_word(leaf, 16 * w) * eq[w])
                .fold(F128::ZERO, |a, x| a + x);
            want_sum += alphas[li][k] * dot;
        }
    }
    assert_eq!(*built.public.last().unwrap(), want_sum, "enforced_sum");

    // ---- prove / verify ----
    let union = UnionInstance::new(&shape.registry, shape.counts.clone());
    let pcs_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let b3 = blake3::build_block_r1cs(nu);
    let b3_lc = b3.csc_lincheck_circuit();
    let swap_r1cs = SwapTable::build_block_r1cs(nu);
    let swap_lc = swap_r1cs.csc_lincheck_circuit();
    let spread_ty = BitSpreadTable::new(max_depth);
    let spread_r1cs = spread_ty.build_block_r1cs(nu);
    let spread_lc = spread_r1cs.csc_lincheck_circuit();

    // Witnesses once; each prove rep rebuilds its inputs from CLONES outside
    // the timer (`UnionSlotProverInput::new` consumes them).
    let (b3_wit, wit_t) = timed(3, || {
        blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(slots.b3), nu)
    });
    let swap_wit = SwapTable::generate_witness_batch_major(built.rows::<SwapGate>(slots.swap), nu);
    let spread_wit =
        spread_ty.generate_witness_batch_major(built.rows::<BitSpreadGate>(slots.spread), nu);
    let els: Vec<Vec<F128>> = leaf_slot
        .iter()
        .map(|(_, s)| match &built.witnesses[shape.registry_slot(*s)] {
            SlotWitness::Element(z) => z.clone(),
            other => panic!("leaf-eval slot produced {other:?}"),
        })
        .collect();

    let mut lcs_ord: Vec<(usize, &dyn LincheckCircuit)> = vec![
        (shape.registry_slot(slots.b3), b3_lc),
        (shape.registry_slot(slots.swap), swap_lc),
        (shape.registry_slot(slots.spread), spread_lc),
    ];
    lcs_ord.sort_by_key(|(i, _)| *i);
    let lcs: Vec<&dyn LincheckCircuit> = lcs_ord.into_iter().map(|(_, c)| c).collect();

    let ((proof, commitment), prove_t) = timed(REPS, || {
        let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![
            (
                shape.registry_slot(slots.b3),
                UnionSlotProverInput::new(b3_wit.clone(), b3_lc),
            ),
            (
                shape.registry_slot(slots.swap),
                UnionSlotProverInput::new(swap_wit.clone(), swap_lc),
            ),
            (
                shape.registry_slot(slots.spread),
                UnionSlotProverInput::new(spread_wit.clone(), spread_lc),
            ),
        ];
        bool_slots.sort_by_key(|(i, _)| *i);
        let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
            .iter()
            .zip(els.clone())
            .map(|((_, s), z)| (shape.registry_slot(*s), z))
            .collect();
        el_ord.sort_by_key(|(i, _)| *i);
        let el_inputs: Vec<UnionElementSlotInput> = el_ord
            .into_iter()
            .map(|(_, z)| {
                UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z))
            })
            .collect();
        let mut c = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &pcs_params,
            bool_slots.into_iter().map(|(_, s)| s).collect(),
            el_inputs,
            &mut c,
        );
        (proof, commitment)
    });

    let (_, verify_t) = timed(REPS, || {
        let mut c = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &shape.circuit,
            &built.public,
            &lcs,
            &commitment,
            &proof,
            &pcs_params,
            &mut c,
        )
        .expect("the collapsed query phase verifies")
    });

    let nnz = |r: &BlockR1cs| {
        r.a_0.rows.iter().map(|x| x.len()).sum::<usize>()
            + r.b_0.rows.iter().map(|x| x.len()).sum::<usize>()
    };
    println!(
        "\nMVP-6 FULL QUERY PHASE, COLLAPSED (m=26 Fast ladder)\n  \
         blake3 {} rows | swap {} | spread {} | leaf-eval {}+{}\n  \
         lincheck nnz {} (MVP-5: 105145720) | dense {} words | dense_m {} | \
         M_bool {} | mu {}\n\n  \
         medians of {REPS} runs, spread in brackets\n  \
         PER PROOF     online {online_t} + witgen {wit_t} + prove {prove_t} ms\n  \
         verifier side {verify_t} ms | proof {:.1} KiB | {threads} threads\n  \
         MERGED OPEN   frobenius {:.1} KiB, {} rounds | {} gather claims\n  \
         SETUP         {setup_ms:6.0} ms\n",
        shape.counts[shape.registry_slot(slots.b3)],
        shape.counts[shape.registry_slot(slots.swap)],
        shape.counts[shape.registry_slot(slots.spread)],
        shape.counts[shape.registry_slot(leaf_slot[0].1)],
        shape.counts[shape.registry_slot(leaf_slot[1].1)],
        nnz(&b3) + nnz(&swap_r1cs) + nnz(&spread_r1cs),
        union.dense_words(),
        union.dense_m(),
        union.m_bool(),
        shape.circuit.cells().mu(),
        bincode::serialize(&proof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
        bincode::serialize(&proof.pcs_open.frobenius)
            .map(|b| b.len())
            .unwrap_or(0) as f64
            / 1024.0,
        proof.pcs_open.merged_rounds.len(),
        shape.circuit.cells().num_gate_slots(),
    );
}

/// **MVP-9: the boolean LEAF — the recursion tree's real leaf shape
/// (rs×2, pd = 0).** A real blake3 workload proof (blake3 for BOTH the
/// FS chain and the Merkle trees — each default diverges silently) is
/// natively verified under a RecordingChallenger, and the outer circuit
/// grows over the recorded tape in the mvp7 pattern:
///
/// - TAPE PINS, all field-for-field vs the proof: the R = 2 multipoint
///   region (2×128 RS dual values, γ^{128i+j} schedule, T0 → rounds →
///   T_m == anchor.v), the boolean PIOP (zerocheck tau/skip slices/
///   rounds/finals, lincheck rounds + z_partial; matrix_evals are NOT
///   absorbed — deferred proof-side by design), and the ring-switch
///   regions (s_hat_v slices, r_dprime/gamma ordinals) with the whole
///   R = 2 merged boundary replayed: succinct outputs, W-fold,
///   linearized coefficients, running == q_eval·V.
/// - IN-CIRCUIT: the full FS chain; the full query phase (collapsed
///   openings against the absorbed caps, FS-derived v, boundary-
///   expanded alpha, per-level enforced sums == native replicas); the
///   PoW bit predicate (published digest/nonce wires, checker-applied);
///   and the intake W-rounds (target as checker-validated advice — the
///   sc dots are family-H, the bit-matrix transpose — with rho BOUND
///   in-circuit and running published). The outer proves and verifies
///   over the circuit path.
///
/// Remaining for the leaf node: the ligerito spine, the boolean PIOP
/// gates (skip round checker-native first), the R = 2 T0/anchor deltas,
/// MatrixAssertion emission, then the family-H upgrade batch.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp9_boolean_leaf_tape() {
    use flock_core::transcript_record::{RecordingChallenger, TranscriptOp as Op};
    // Pin the perf pool BEFORE any rayon touch (the native leaf prove would
    // otherwise auto-initialize the global pool at all cores).
    let threads = flock_core::init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);

    let n_blocks = 256usize;
    let setup = blake3::Blake3Setup::new_batch_major(n_blocks);
    let mut rng = Rng(0x4D50_9B00);
    let inputs: Vec<blake3::Compression> = (0..n_blocks)
        .map(|_| {
            let cv: [u32; 8] = from_fn(|_| rng.next_u32());
            let m: [u32; 16] = from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
            (cv, m, counter, 64u32, 11u32)
        })
        .collect();
    let circuit = setup.r1cs.csc_lincheck_circuit();
    let registry = Registry::new(
        vec![TableType::from_block_r1cs(&setup.r1cs)],
        setup.r1cs.n_log(),
    );
    let union = UnionInstance::new(&registry, vec![n_blocks]);
    let slot = UnionSlotProverInput::new(
        blake3::generate_witness_batch_major(&inputs, setup.n_blocks_log()),
        circuit,
    );
    // BLAKE3 for BOTH the Merkle trees and the FS chain — the circuit's
    // opening gates and chain rows model blake3; the setup's defaults
    // (SHA-256 Merkle) diverge silently otherwise.
    let mut leaf_pcs = setup.pcs_params.clone();
    leaf_pcs.merkle_hash = HashKind::Blake3;
    let mut ch = FsChallenger::with_hash(DOMAIN, HashKind::Blake3);
    let (proof, commitment, _claim) =
        prover::prove_fast_ligerito_union(&union, &leaf_pcs, vec![slot], &mut ch);

    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(DOMAIN, HashKind::Blake3));
    verifier::verify_ligerito_union(&union, &[circuit], &commitment, &proof, &leaf_pcs, &mut rec)
        .expect("the leaf workload proof verifies");
    let t_shape = rec.shape();
    let chals: Vec<F128> = rec.challenges().to_vec();
    let vals_rec = rec.values();

    if var("MVP9_DUMP").is_ok() {
        for (i, op) in t_shape.ops().iter().enumerate().take(160) {
            eprintln!("op {i:>3}: {op:?}");
        }
    }

    // ---- the boolean PIOP region, located and pinned (phase 2a) ----
    // bind -> zerocheck (skip slices + rounds + finals) -> lincheck
    // (rounds + z_partial + matrix_evals). Every absorbed value is
    // identified against the proof field it carries, so the assembly's
    // wires have named indices — the MVP-8-step-1 pattern.
    {
        use flock_core::transcript_record::TranscriptOp as Op2;
        let ops = t_shape.ops();
        let (mut v, mut c, mut i) = (0usize, 0usize, 0usize);
        let bump = |op: &Op2, v: &mut usize, c: &mut usize| match op {
            Op2::SqueezeScalar => *c += 1,
            Op2::SqueezeSlice(n) => *c += n,
            Op2::ObserveScalar => *v += 1,
            Op2::ObserveSlice(n) => *v += n,
            _ => {}
        };
        while !matches!(&ops[i], Op2::Label(l) if l.as_slice() == b"flock-zerocheck-v0") {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        i += 1;
        assert!(matches!(ops[i], Op2::SqueezeSlice(_)), "zc tau lo");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        assert!(matches!(ops[i], Op2::SqueezeSlice(_)), "zc tau hi");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        let r1ab_v = v;
        assert!(matches!(ops[i], Op2::ObserveSlice(64)), "round1_ab");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        let r1c_v = v;
        assert!(matches!(ops[i], Op2::ObserveSlice(64)), "round1_c");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        assert!(matches!(ops[i], Op2::SqueezeScalar), "z_skip");
        bump(&ops[i], &mut v, &mut c);
        i += 1;
        assert_eq!(
            &vals_rec[r1ab_v..r1ab_v + 64],
            &proof.zerocheck.round1_ab[..],
            "round1_ab words"
        );
        assert_eq!(
            &vals_rec[r1c_v..r1c_v + 64],
            &proof.zerocheck.round1_c[..],
            "round1_c words"
        );
        let mut zc_rounds = Vec::new();
        loop {
            // rounds are [obs, obs, squeeze]; the finals are obs NOT
            // followed by a squeeze.
            if matches!(ops[i], Op2::ObserveScalar)
                && matches!(ops[i + 1], Op2::ObserveScalar)
                && matches!(ops[i + 2], Op2::SqueezeScalar)
            {
                zc_rounds.push((v, c + 1));
                for _ in 0..3 {
                    bump(&ops[i], &mut v, &mut c);
                    i += 1;
                }
            } else {
                break;
            }
        }
        assert_eq!(
            zc_rounds.len(),
            proof.zerocheck.multilinear_rounds.len(),
            "zc rounds"
        );
        for ((g_v, _), want) in zc_rounds.iter().zip(&proof.zerocheck.multilinear_rounds) {
            assert_eq!((vals_rec[*g_v], vals_rec[*g_v + 1]), *want, "zc round msg");
        }
        let mut zc_finals = Vec::new();
        while matches!(ops[i], Op2::ObserveScalar) {
            zc_finals.push(vals_rec[v]);
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let want_finals = [
            proof.zerocheck.final_a_eval,
            proof.zerocheck.final_b_eval,
            proof.zerocheck.final_c_eval,
        ];
        assert_eq!(&want_finals[..zc_finals.len()], &zc_finals[..], "zc finals");
        assert!(
            matches!(&ops[i], Op2::Label(l) if l.as_slice() == b"flock-lincheck-v0"),
            "lincheck label, got {:?}",
            ops[i]
        );
        i += 1;
        let mut lc_pre_squeezes = 0usize;
        while matches!(ops[i], Op2::SqueezeScalar) {
            lc_pre_squeezes += 1;
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        assert!(lc_pre_squeezes >= 1, "lc alpha");
        let mut lc_rounds = Vec::new();
        while matches!(ops[i], Op2::ObserveScalar)
            && matches!(ops[i + 1], Op2::ObserveScalar)
            && matches!(ops[i + 2], Op2::SqueezeScalar)
        {
            lc_rounds.push((v, c + 1));
            for _ in 0..3 {
                bump(&ops[i], &mut v, &mut c);
                i += 1;
            }
        }
        assert_eq!(lc_rounds.len(), proof.lincheck.rounds.len(), "lc rounds");
        for ((g_v, _), want) in lc_rounds.iter().zip(&proof.lincheck.rounds) {
            assert_eq!((vals_rec[*g_v], vals_rec[*g_v + 1]), *want, "lc round msg");
        }
        // The lincheck tail: z_partial (a 64-slice) and the matrix_evals
        // pairs, in whatever op order — located by value identification.
        let mut zp_v = None;
        let mut tail_scalars = Vec::new();
        while !matches!(&ops[i], Op2::Label(l) if l.as_slice() == b"flock-merged-open-v0") {
            match ops[i] {
                Op2::ObserveSlice(64) if zp_v.is_none() => zp_v = Some(v),
                Op2::ObserveScalar => tail_scalars.push(vals_rec[v]),
                Op2::ObserveSlice(n) => {
                    tail_scalars.extend_from_slice(&vals_rec[v..v + n]);
                }
                _ => {}
            }
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let zp_v = zp_v.expect("z_partial slice on the tape");
        assert_eq!(
            &vals_rec[zp_v..zp_v + 64],
            &proof.lincheck.z_partial[..],
            "z_partial words"
        );
        // matrix_evals are NOT on the tape — the deferral leaves them
        // proof-side, pinned only by the lincheck's final one-equation
        // check and the accumulator's root discharge. In the outer
        // circuit they enter as published advice bound by that equation
        // (the assertion-emission shape for the leaf).
        assert!(
            tail_scalars.is_empty(),
            "the lincheck tail carries only z_partial"
        );
        assert!(
            !proof.lincheck.matrix_evals.is_empty(),
            "the deferred matrix work"
        );
    }

    // The leaf's opening shape: rs×2, pd = 0 — R = 2, P = 0.
    let fro = &proof.pcs_open.frobenius;
    assert_eq!(fro.values.len(), 2, "the boolean class contributes rs×2");
    assert!(
        fro.values.iter().all(|v| v.len() == 128),
        "128 values per RS claim"
    );
    assert!(
        fro.group_values.is_empty(),
        "no packed-direct claims at the leaf"
    );

    // Locate the multipoint region with a minimal cursor (value/challenge
    // ordinals), exactly as parse_open_levels does for the element inner.
    let ops = t_shape.ops();
    let (mut v, mut c, mut i) = (0usize, 0usize, 0usize);
    let bump = |op: &Op, v: &mut usize, c: &mut usize| match op {
        Op::SqueezeScalar => *c += 1,
        Op::SqueezeSlice(n) => *c += n,
        Op::ObserveScalar => *v += 1,
        Op::ObserveSlice(n) => *v += n,
        _ => {}
    };
    while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1") {
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    i += 1;
    let mut val_vs = Vec::new();
    while matches!(ops[i], Op::ObserveScalar) {
        val_vs.push(v);
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    assert_eq!(val_vs.len(), 256, "2×128 RS dual values absorbed");
    assert!(matches!(ops[i], Op::SqueezeScalar), "multipoint gamma");
    let gamma = chals[c];
    bump(&ops[i], &mut v, &mut c);
    i += 1;
    let mut rounds = Vec::new();
    while matches!(ops[i], Op::ObserveScalar) {
        let g_v = v;
        for _ in 0..2 {
            assert!(matches!(ops[i], Op::ObserveScalar));
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        assert!(matches!(ops[i], Op::SqueezeScalar), "mp round");
        rounds.push((g_v, c));
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    assert!(
        matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0"),
        "anchor label"
    );
    i += 1;
    let anchor_v = v;
    assert!(matches!(ops[i], Op::ObserveScalar));
    bump(&ops[i], &mut v, &mut c);
    i += 1;
    let mut anchor_rounds = 0usize;
    while matches!(ops[i], Op::ObserveScalar) {
        for _ in 0..2 {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        assert!(matches!(ops[i], Op::SqueezeScalar), "anchor round");
        anchor_rounds += 1;
        bump(&ops[i], &mut v, &mut c);
        i += 1;
    }
    assert_eq!(rounds.len(), fro.rounds.len(), "two-product round count");
    assert_eq!(anchor_rounds, fro.anchor.rounds.len(), "anchor round count");
    assert_eq!(vals_rec[anchor_v], fro.anchor.v, "anchor v on the tape");

    // The located stream words ARE the proof's RS dual values, in order.
    for (k, &vi) in val_vs.iter().enumerate() {
        assert_eq!(vals_rec[vi], fro.values[k / 128][k % 128], "RS value {k}");
    }

    // The accept chain with the R = 2 schedule: T0 = Σ γ^{128 i + j}·A_ij
    // folds through the rounds to T_m == anchor.v.
    let mut t = F128::ZERO;
    let mut pw = F128::ONE;
    for &vi in &val_vs {
        t += pw * vals_rec[vi];
        pw *= gamma;
    }
    for &(g_v, ch_ix) in &rounds {
        let (g1, gi) = (vals_rec[g_v], vals_rec[g_v + 1]);
        let r = chals[ch_ix];
        let g0 = t + g1;
        t = g0 + (g1 + g0 + gi) * r + gi * r * r;
    }
    assert_eq!(
        t, fro.anchor.v,
        "T_m must equal the anchor's claimed v (R = 2)"
    );

    // ---- phase 2b step 1: the ring-switch regions pinned, and the R = 2
    // merged boundary replayed from located pieces: the two s_hat_v slices
    // ARE the proof's, r_dprime/gamma ordinals named, the succinct outputs
    // (claim transpose dot) rebuilt, the W-rounds folded to `running`, the
    // linearized coefficients derived from the SAME pub helpers, and
    // running == q_eval·V closed with the R = 2 recombination. This is the
    // exact relation chain the leaf circuit will publish as zero-deltas.
    let (native_target, native_running) = {
        use flock_core::pcs::ring_switch as rs;
        use flock_core::zerocheck::univariate_skip::build_eq;
        let (mut v, mut c, mut i) = (0usize, 0usize, 0usize);
        let bump = |op: &Op, v: &mut usize, c: &mut usize| match op {
            Op::SqueezeScalar => *c += 1,
            Op::SqueezeSlice(n) => *c += n,
            Op::ObserveScalar => *v += 1,
            Op::ObserveSlice(n) => *v += n,
            _ => {}
        };
        while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-merged-open-v0") {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        i += 1;
        let mut rs_recs: Vec<(usize, usize)> = Vec::new();
        while matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-ring-switch-v0") {
            i += 1;
            assert!(matches!(ops[i], Op::ObserveSlice(128)), "s_hat_v slice");
            let sv = v;
            bump(&ops[i], &mut v, &mut c);
            i += 1;
            assert!(matches!(ops[i], Op::SqueezeSlice(7)), "r_dprime");
            let rc = c;
            bump(&ops[i], &mut v, &mut c);
            i += 1;
            rs_recs.push((sv, rc));
        }
        assert_eq!(rs_recs.len(), 2, "rs×2 at the leaf");
        let mut gs = Vec::new();
        for _ in 0..2 {
            assert!(matches!(ops[i], Op::SqueezeScalar), "rs gamma");
            gs.push(chals[c]);
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let mut target = F128::ZERO;
        let mut coeffs: Vec<Vec<F128>> = Vec::new();
        for (k, &(sv, rc)) in rs_recs.iter().enumerate() {
            let shv = &vals_rec[sv..sv + 128];
            assert_eq!(
                shv,
                &proof.pcs_open.ring_switches[k].s_hat_v[..],
                "s_hat_v {k} on the stream"
            );
            let rdp: Vec<F128> = (0..7).map(|j| chals[rc + j]).collect();
            let eq = build_eq(&rdp);
            target += gs[k] * rs::inner_product(&rs::tensor_algebra_transpose(shv), &eq);
            let scaled: Vec<F128> = eq.iter().map(|x| gs[k] * *x).collect();
            coeffs.push(rs::linearized_coefficients(&rs::build_fold_byte_table(
                &scaled,
            )));
        }
        let mut running = target;
        while matches!(ops[i], Op::ObserveScalar)
            && matches!(ops[i + 1], Op::ObserveScalar)
            && matches!(ops[i + 2], Op::SqueezeScalar)
        {
            let (g1, gi) = (vals_rec[v], vals_rec[v + 1]);
            for _ in 0..2 {
                bump(&ops[i], &mut v, &mut c);
                i += 1;
            }
            let r = chals[c];
            bump(&ops[i], &mut v, &mut c);
            i += 1;
            let g0 = running + g1;
            running = g0 + (g1 + g0 + gi) * r + gi * r * r;
        }
        while !matches!(&ops[i], Op::Label(l) if l.as_slice() == b"flock-pcs-packed-direct-v0") {
            bump(&ops[i], &mut v, &mut c);
            i += 1;
        }
        let q_eval = vals_rec[v];
        let mut big_v = F128::ZERO;
        for (k, cs) in coeffs.iter().enumerate() {
            for (j, &cj) in cs.iter().enumerate() {
                if cj.is_zero() {
                    continue;
                }
                let mut x = fro.values[k][j];
                for _ in 0..j {
                    x = x * x;
                }
                big_v += cj * x;
            }
        }
        assert_eq!(running, q_eval * big_v, "the R = 2 merged boundary replays");
        (target, running)
    };

    // ---- phase 2a step 2 + the query-phase port: the leaf's chain AND
    // query phase in-circuit — Merkle openings against the absorbed caps,
    // FS-derived v, boundary-expanded alpha, per-level enforced sums
    // published and checked against native replicas. The mvp7 machinery,
    // instantiated on the boolean leaf tape.
    {
        struct Lvl {
            q: usize,
            c: usize,
            path: usize,
            depth: usize,
            lanes: usize,
        }
        // The PoW grinding ops, located and bound (the mvp7 machinery).
        struct PowRec {
            fin: usize,
            pay: usize,
            bits: u32,
        }
        // Native seed + g0/running chain.
        #[cfg(test)]
        use flock_core::lincheck::build_eq_table;
        use flock_core::zerocheck::multilinear::{
            interpolate_at_z_combined, interpolate_at_z_on_lambda,
        };
        use flock_core::zerocheck::univariate_skip_optimized::{
            medium_challenges_ghash, small_challenges_ghash,
        };
        use flock_prover::prover::UnionElementSlotInput;
        use flock_prover::r1cs_hashes::fs_chain::FsChain;

        let lig = &proof.pcs_open.inner.ligerito;
        assert_eq!(commitment.cap, lig.initial_cap, "commitment IS the L0 cap");
        let r = lig.recursive_caps.len();
        let lvl_src: Vec<(&[[u8; 32]], &Vec<Vec<F128>>, &Vec<[u8; 32]>)> = (0..=r)
            .map(|li| {
                if li == 0 {
                    (
                        lig.initial_cap.as_slice(),
                        &lig.initial_proof.opened_rows,
                        &lig.initial_proof.merkle_proof,
                    )
                } else if li < r {
                    (
                        lig.recursive_caps[li - 1].as_slice(),
                        &lig.recursive_proofs[li - 1].opened_rows,
                        &lig.recursive_proofs[li - 1].merkle_proof,
                    )
                } else {
                    (
                        lig.recursive_caps[r - 1].as_slice(),
                        &lig.final_proof.opened_rows,
                        &lig.final_proof.merkle_proof,
                    )
                }
            })
            .collect();
        let (start_v, piop_o, gammas_o, w_rounds, _mp2, inner_pd2, yr_v2, levels) =
            parse_open_levels(t_shape.ops(), 32 * lig.initial_cap.len(), r);
        assert!(piop_o.is_none(), "a boolean tape has no element PIOP");
        assert!(gammas_o.is_empty(), "no packed-direct claims at the leaf");
        assert_eq!(levels.len(), r + 1);

        let mut geo: Vec<Lvl> = Vec::new();
        let mut native_sums: Vec<F128> = Vec::new();
        for (li, lvl) in levels.iter().enumerate() {
            let (cap, rows, paths) = lvl_src[li];
            let q = lvl.q_count;
            assert_eq!(rows.len(), q, "L{li}: one opened row per query");
            let c = cap.len().trailing_zeros() as usize;
            let path = paths.len() / q;
            let depth = path + c;
            let lanes = rows[0].len();
            assert_eq!(lanes, 1 << lvl.fold_fins.len(), "L{li}: lanes vs folds");
            let fold_vals: Vec<F128> = lvl.fold_chs.iter().map(|&i| chals[i]).collect();
            let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
            let eqv = build_eq_table(&fold_vals);
            let aw = build_eq_table(&alpha_vals);
            let mut sum = F128::ZERO;
            for (k, row) in rows.iter().enumerate() {
                let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << depth) - 1);
                let mut leaf_bytes = Vec::with_capacity(16 * lanes);
                for f in row {
                    leaf_bytes.extend_from_slice(&f.lo.to_le_bytes());
                    leaf_bytes.extend_from_slice(&f.hi.to_le_bytes());
                }
                let lh = core_merkle::hash_leaf(&leaf_bytes, HashKind::Blake3);
                assert!(
                    core_merkle::verify_merkle_proof_capped(
                        cap,
                        1 << depth,
                        &lh,
                        pos,
                        &paths[k * path..(k + 1) * path],
                        HashKind::Blake3,
                    ),
                    "L{li} query {k}: capped path verifies natively"
                );
                let dot = row
                    .iter()
                    .zip(eqv.iter())
                    .map(|(&x, &e)| x * e)
                    .fold(F128::ZERO, |a, vv| a + vv);
                sum += aw[k] * dot;
            }
            native_sums.push(sum);
            geo.push(Lvl {
                q,
                c,
                path,
                depth,
                lanes,
            });
        }

        let stream = t_shape.stream_words(DOMAIN);
        let bytes = stream.to_bytes(rec.values(), rec.payloads());
        let mut chain = FsChain::new();
        let mut at = 0usize;
        let fin_ops: Vec<_> = t_shape.ops().iter().filter(|o| o.finalizes()).collect();
        assert_eq!(
            stream.finalize_after.len(),
            fin_ops.len(),
            "finalize alignment"
        );
        for (k, &upto) in stream.finalize_after.iter().enumerate() {
            chain.absorb(&bytes[at * 16..upto * 16]);
            at = upto;
            chain.finalize(fin_ops[k].squeezed_bytes());
        }
        chain.absorb(&bytes[at * 16..]);
        let trace = chain.finish();
        let b3_rows: usize = trace.rows.len()
            + geo
                .iter()
                .map(|g| (g.lanes / 4 + g.path) * g.q)
                .sum::<usize>();
        let nu = (b3_rows.next_power_of_two().trailing_zeros() as usize).max(3);
        let max_path = geo.iter().map(|g| g.path).max().unwrap().max(1);

        let mut sb = ShapeBuilder::new(nu);
        let slots = CollapsedSlots {
            b3: sb.slot(Blake3Gate { nu }),
            swap: sb.slot(SwapGate { nu }),
            spread: sb.slot(BitSpreadGate {
                ty: BitSpreadTable::new(max_path),
                nu,
            }),
        };
        let mut leaf_slot: Vec<(usize, SlotId)> = Vec::new();
        let leafeval: Vec<_> = geo
            .iter()
            .map(|g| {
                let lanes = g.lanes.min(8);
                match leaf_slot.iter().find(|(n, _)| *n == lanes) {
                    Some((_, sl)) => *sl,
                    None => {
                        let sl = sb.slot(LeafEvalGate::new(lanes));
                        leaf_slot.push((lanes, sl));
                        sl
                    }
                }
            })
            .collect();
        let mut vals: Vec<F128> = Vec::new();
        let iv_w = pack8(&FS_CHAIN_IV);
        vals.extend_from_slice(&iv_w);
        let iv = [sb.public_input(), sb.public_input()];
        let (outs, ww) = emit_fs_chain(&mut sb, slots.b3, iv, &trace, &stream, &bytes, &mut vals);

        let pows: Vec<PowRec> = {
            use flock_core::transcript_record::TranscriptOp as Op3;
            let mut out = Vec::new();
            let (mut fin, mut pay) = (0usize, 0usize);
            for op in t_shape.ops() {
                if let Op3::Pow { bits } = op {
                    out.push(PowRec {
                        fin,
                        pay,
                        bits: *bits,
                    });
                }
                if op.finalizes() {
                    fin += 1;
                }
                match op {
                    Op3::ObserveBytes(_) | Op3::Pow { .. } => pay += 1,
                    _ => {}
                }
            }
            out
        };
        assert!(!pows.is_empty(), "the Fast profile grinds");
        let pow_pub: Vec<[Wire; 3]> = pows
            .iter()
            .map(|pr| {
                let sq = &trace.squeezes[pr.fin];
                let wi = stream
                    .words
                    .iter()
                    .position(
                        |w| matches!(w, StreamWord::Bytes { payload, .. } if *payload == pr.pay),
                    )
                    .expect("pow nonce stream word");
                let nw = ww[wi].expect("pow nonce wired");
                [outs[sq[0]][0], outs[sq[0]][1], nw]
            })
            .collect();

        let mut hints: Vec<[u32; SLOT_WORDS]> = Vec::new();
        let mut to_publish: Vec<(Vec<Wire>, Vec<(Wire, [Wire; 2])>)> = Vec::new();
        let mut level_accs: Vec<Wire> = Vec::new();
        for (li, lvl) in levels.iter().enumerate() {
            let g = &geo[li];
            let (_, rows, paths) = lvl_src[li];
            let sqq = &trace.squeezes[lvl.q_fin];
            let sqa = &trace.squeezes[lvl.a_fin];
            let a_wires: Vec<Wire> = (0..lvl.a_count).map(|j| outs[sqa[j / 4]][j % 4]).collect();
            let v_wires: Vec<Wire> = lvl
                .fold_fins
                .iter()
                .map(|&f| outs[trace.squeezes[f][0]][0])
                .collect();
            let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
            let aw = build_eq_table(&alpha_vals);
            let le_vars = g.lanes.min(8).trailing_zeros() as usize;
            let le_groups = g.lanes >> le_vars;
            let hw = {
                let v_hi: Vec<F128> = lvl.fold_chs[le_vars..].iter().map(|&i| chals[i]).collect();
                build_eq_table(&v_hi)
            };
            vals.push(F128::ZERO);
            let mut acc = sb.public_input();
            let mut opens: Vec<(Wire, [Wire; 2])> = Vec::with_capacity(g.q);
            for k in 0..g.q {
                vals.extend_from_slice(&rows[k]);
                let leaf_w: Vec<Wire> = (0..g.lanes).map(|_| sb.input()).collect();
                let cw = outs[sqq[k / 4]][k % 4];
                let cv = emit_opening(&mut sb, slots, iv, &leaf_w, cw, g.depth, g.c, &mut vals);
                opens.push((cw, cv));
                hints.extend(
                    paths[k * g.path..(k + 1) * g.path]
                        .iter()
                        .map(hash_to_digest),
                );
                let lanes = g.lanes.min(8);
                for h in 0..le_groups {
                    let mut a_in: Vec<Wire> = leaf_w[lanes * h..lanes * (h + 1)].to_vec();
                    a_in.extend_from_slice(&v_wires[..le_vars]);
                    vals.push(aw[k] * hw[h]);
                    a_in.push(sb.public_input());
                    a_in.push(acc);
                    acc = sb.gate(leafeval[li], &a_in)[0];
                }
            }
            to_publish.push((a_wires, opens));
            level_accs.push(acc);
        }
        // ---- the intake W-rounds in-circuit: the RS target enters as
        // CHECKER-VALIDATED advice (its sc dots are family-H — the bit
        // transpose — deferred to the boundary batch), the W-rounds fold
        // it through mrslot binding rho, and `running` publishes; the
        // checker closes target == Σ γ·sc and running == q_eval·V
        // natively (the 2b replay above is the reference).
        let mut vmap: Vec<Option<usize>> = Vec::new();
        for (wi, w) in stream.words.iter().enumerate() {
            if let StreamWord::Value(vi) = *w {
                if vmap.len() <= vi {
                    vmap.resize(vi + 1, None);
                }
                vmap[vi] = Some(wi);
            }
        }
        let wv = |vi: usize| -> Wire { ww[vmap[vi].expect("stream word")].expect("wired") };
        let mrslot = sb.slot(MergedRoundGate::new());
        leaf_slot.push((400, mrslot));
        vals.push(native_target);
        let tw = sb.public_input();
        let mut runw = tw;
        for rr in &w_rounds {
            let r_w = outs[trace.squeezes[rr.fin][0]][0];
            runw = sb.gate(mrslot, &[runw, wv(rr.g_v), wv(rr.g_v + 1), r_w])[0];
        }

        // ---- the ligerito spine (2a on the leaf): start = gamma'·q_eval,
        // eval/build per fold round, intro-folds for OODs and levels with
        // the LeafEval accumulators consumed IN-CIRCUIT; the final t_r
        // publishes and is checked against a native replay — equality
        // transitively validates every eval/build/fold gate AND the accs.
        vals.push(F128::ZERO);
        let zw = sb.public_input();
        vals.push(F128::ONE);
        let ow = sb.public_input();
        let spine = sb.slot(SpineGate::new());
        leaf_slot.push((0, spine));
        let gpw = outs[trace.squeezes[inner_pd2.fin][0]][0];
        let tw0 = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, wv(inner_pd2.q_v), gpw, zw]);
        let mut tsp = tw0[3];
        let st = sb.gate(
            spine,
            &[zw, zw, zw, zw, wv(start_v), wv(start_v + 1), tsp, ow, zw],
        );
        let (mut qc, mut qb, mut qa) = (st[0], st[1], st[2]);
        for (li, lvl) in levels.iter().enumerate() {
            for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
                let rw = outs[trace.squeezes[lvl.fold_fins[j]][0]][0];
                let ev = sb.gate(spine, &[qc, qb, qa, zw, zw, zw, zw, zw, rw]);
                tsp = ev[4];
                let bld = sb.gate(spine, &[zw, zw, zw, zw, wv(mv), wv(mv + 1), tsp, ow, zw]);
                (qc, qb, qa) = (bld[0], bld[1], bld[2]);
            }
            if li < r {
                for od in &lvl.ood {
                    let bw = outs[trace.squeezes[od.beta_fin][0]][0];
                    let f = sb.gate(
                        spine,
                        &[
                            qc,
                            qb,
                            qa,
                            tsp,
                            wv(od.intro_v),
                            wv(od.intro_v + 1),
                            wv(od.y_v),
                            bw,
                            zw,
                        ],
                    );
                    (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
                }
                let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
                let f = sb.gate(
                    spine,
                    &[
                        qc,
                        qb,
                        qa,
                        tsp,
                        wv(lvl.intro_v),
                        wv(lvl.intro_v + 1),
                        level_accs[li],
                        bw,
                        zw,
                    ],
                );
                (qc, qb, qa, tsp) = (f[0], f[1], f[2], f[3]);
            } else {
                let bw = outs[trace.squeezes[lvl.beta_fin][0]][0];
                let f = sb.gate(spine, &[zw, zw, zw, tsp, zw, zw, level_accs[li], bw, zw]);
                tsp = f[3];
            }
        }
        let t_final = tsp;

        // ---- the boolean zerocheck's multilinear rounds in-circuit ----
        // The skip round (round1 interpolation at z) stays checker-native;
        // its output seeds the chain as CHECKER-VALIDATED advice. The 16
        // rounds are ZcRoundGate rows: eq weights t_i are the 7 baked
        // ghash constants then the 9 r_outer squeeze wires, rho from the
        // chain, msgs from the stream, g0 as advice with published-zero
        // deltas — family I, no in-circuit inversion.
        let zc0 = {
            use flock_core::transcript_record::TranscriptOp as Op4;
            let ops2 = t_shape.ops();
            let (mut v2, mut c2, mut f2, mut i2) = (0usize, 0usize, 0usize, 0usize);
            let bump2 = |op: &Op4, v: &mut usize, c: &mut usize, f: &mut usize| {
                if op.finalizes() {
                    *f += 1;
                }
                match op {
                    Op4::SqueezeScalar => *c += 1,
                    Op4::SqueezeSlice(n) => *c += n,
                    Op4::ObserveScalar => *v += 1,
                    Op4::ObserveSlice(n) => *v += n,
                    _ => {}
                }
            };
            while !matches!(&ops2[i2], Op4::Label(l) if l.as_slice() == b"flock-zerocheck-v0") {
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                i2 += 1;
            }
            i2 += 1;
            // r_skip: SqueezeSlice(6)
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            // r_outer: SqueezeSlice(9)
            let (outer_ch, outer_fin) = (c2, f2);
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let r1ab_v = v2;
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let r1c_v = v2;
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let z_ch = c2;
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let mut rounds2 = Vec::new();
            while matches!(ops2[i2], Op4::ObserveScalar)
                && matches!(ops2[i2 + 1], Op4::ObserveScalar)
                && matches!(ops2[i2 + 2], Op4::SqueezeScalar)
            {
                let g_v = v2;
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                bump2(&ops2[i2 + 1], &mut v2, &mut c2, &mut f2);
                let (ch, fin) = (c2, f2);
                bump2(&ops2[i2 + 2], &mut v2, &mut c2, &mut f2);
                rounds2.push((g_v, ch, fin));
                i2 += 3;
            }
            // the zc finals (v_a, v_b — the lincheck's entry values)
            let mut finals_v = Vec::new();
            while matches!(ops2[i2], Op4::ObserveScalar) {
                finals_v.push(v2);
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                i2 += 1;
            }
            assert!(
                matches!(&ops2[i2], Op4::Label(l) if l.as_slice() == b"flock-lincheck-v0"),
                "lincheck label"
            );
            i2 += 1;
            let (alpha_ch2, alpha_fin2) = (c2, f2);
            assert!(matches!(ops2[i2], Op4::SqueezeScalar), "lc alpha");
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let (beta_ch2, beta_fin2) = (c2, f2);
            assert!(
                matches!(ops2[i2], Op4::SqueezeScalar),
                "lc beta (const pin)"
            );
            bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
            i2 += 1;
            let mut lc_rounds2 = Vec::new();
            while matches!(ops2[i2], Op4::ObserveScalar)
                && matches!(ops2[i2 + 1], Op4::ObserveScalar)
                && matches!(ops2[i2 + 2], Op4::SqueezeScalar)
            {
                let g_v = v2;
                bump2(&ops2[i2], &mut v2, &mut c2, &mut f2);
                bump2(&ops2[i2 + 1], &mut v2, &mut c2, &mut f2);
                let (ch, fin) = (c2, f2);
                bump2(&ops2[i2 + 2], &mut v2, &mut c2, &mut f2);
                lc_rounds2.push((g_v, ch, fin));
                i2 += 3;
            }
            (
                (outer_ch, outer_fin, r1ab_v, r1c_v, z_ch),
                rounds2,
                finals_v,
                (alpha_ch2, alpha_fin2, beta_ch2, beta_fin2),
                lc_rounds2,
            )
        };
        let (zc0, zc_rounds2, zc_finals_v, lc_chs, lc_rounds2) = zc0;
        let (outer_ch, outer_fin, r1ab_v, r1c_v, z_ch) = zc0;
        let (alpha_ch2, alpha_fin2, beta_ch2, beta_fin2) = lc_chs;
        assert!(zc_finals_v.len() >= 2, "zc finals (v_a, v_b) on the tape");

        let zval = chals[z_ch];
        let c_eval = interpolate_at_z_on_lambda(&vals_rec[r1c_v..r1c_v + 64], 6, zval);
        let combined: Vec<F128> = vals_rec[r1ab_v..r1ab_v + 64]
            .iter()
            .zip(&vals_rec[r1c_v..r1c_v + 64])
            .map(|(a, b)| *a + *b)
            .collect();
        let zc_seed = interpolate_at_z_combined(&combined, 6, zval) + c_eval;
        let mut t_vals: Vec<F128> = Vec::new();
        t_vals.extend_from_slice(&small_challenges_ghash());
        t_vals.extend_from_slice(&medium_challenges_ghash());
        for j in 0..zc_rounds2.len() - 7 {
            t_vals.push(chals[outer_ch + j]);
        }
        let mut g0_native = Vec::new();
        let mut zc_run = zc_seed;
        for (k2, &(g_v, ch, _)) in zc_rounds2.iter().enumerate() {
            let (g1, gi) = (vals_rec[g_v], vals_rec[g_v + 1]);
            let t = t_vals[k2];
            let g0 = (zc_run + t * g1) * (F128::ONE + t).inv();
            g0_native.push(g0);
            let rho = chals[ch];
            zc_run = g0 * (F128::ONE + rho) + g1 * rho + gi * rho * (F128::ONE + rho);
        }
        let zc_end_native = zc_run;
        // The circuit chain.
        let zslot = sb.slot(ZcRoundGate::new());
        leaf_slot.push((500, zslot));
        vals.push(zc_seed);
        let zc_seed_w = sb.public_input();
        let mut zrw = zc_seed_w;
        let mut zc_deltas2: Vec<Wire> = Vec::new();
        for (k2, &(g_v, _, fin)) in zc_rounds2.iter().enumerate() {
            let t_w = if k2 < 7 {
                vals.push(t_vals[k2]);
                sb.public_input()
            } else {
                let j = k2 - 7;
                let sq = &trace.squeezes[outer_fin];
                outs[sq[j / 4]][j % 4]
            };
            let rho_w = outs[trace.squeezes[fin][0]][0];
            vals.push(g0_native[k2]);
            let g0w = sb.public_input();
            let g = sb.gate(zslot, &[zrw, wv(g_v), wv(g_v + 1), t_w, rho_w, g0w, ow]);
            zc_deltas2.push(g[0]);
            zrw = g[1];
        }

        // ---- the lincheck rounds in-circuit ----
        // The entry is FULLY BOUND: target = alpha·v_a + v_b + beta with
        // alpha/beta chain squeezes and v_a/v_b the zerocheck finals on the
        // stream (SpineGate tr-rows). The rounds are MergedRoundGate; the
        // end publishes and equals the native chain (the comb-side final
        // consistency and the deferred matrix equation stay checker-native).
        let alpha_w2 = outs[trace.squeezes[alpha_fin2][0]][0];
        let beta_w2 = outs[trace.squeezes[beta_fin2][0]][0];
        let s1 = sb.gate(
            spine,
            &[zw, zw, zw, zw, zw, zw, wv(zc_finals_v[0]), alpha_w2, zw],
        )[3];
        let s2 = sb.gate(spine, &[zw, zw, zw, s1, zw, zw, wv(zc_finals_v[1]), ow, zw])[3];
        let mut lcw = sb.gate(spine, &[zw, zw, zw, s2, zw, zw, beta_w2, ow, zw])[3];
        for &(g_v, _, fin) in &lc_rounds2 {
            let rho_w = outs[trace.squeezes[fin][0]][0];
            lcw = sb.gate(mrslot, &[lcw, wv(g_v), wv(g_v + 1), rho_w])[0];
        }
        // Native replica of the same chain.
        let lc_end_native = {
            let mut lrn = chals[alpha_ch2] * vals_rec[zc_finals_v[0]]
                + vals_rec[zc_finals_v[1]]
                + chals[beta_ch2];
            for &(g_v, ch, _) in &lc_rounds2 {
                let (e1, ei) = (vals_rec[g_v], vals_rec[g_v + 1]);
                let rho = chals[ch];
                let e0 = lrn + e1;
                lrn = ei * rho * rho + (e0 + e1 + ei) * rho + e0;
            }
            lrn
        };

        // ---- the residual region (mvp7's 2b, ported to the leaf) ----
        // Per-level ResidualGates compute the induced-basis residuals
        // (next_s chain from a boundary-bound q_field, prefix over LATER
        // levels' fold wires, suffix subset products), and the close-out
        // (prefix/suffix/partial-combine/final-dot) assembles eval_b and
        // dots the absorbed yr words — `inner == t_r` then closes between
        // circuit outputs, exactly as on mvp7.
        let yr_log = {
            let yl = proof.pcs_open.inner.ligerito.final_proof.yr.len();
            assert!(yl.is_power_of_two());
            yl.trailing_zeros() as usize
        };
        let yr_len = 1usize << yr_log;
        let mut resid_pub: Vec<Vec<Wire>> = Vec::new();
        for (li, lvl) in levels.iter().enumerate() {
            let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len()).sum();
            let lmc = pl + yr_log;
            let sks = sk_at_vks(lmc);
            let rslot = sb.slot(ResidualGate::new(lmc, pl, yr_log, &sks));
            leaf_slot.push((100 + li, rslot));
            let ris_w: Vec<Wire> = levels[li + 1..]
                .iter()
                .flat_map(|l| l.fold_fins.iter().map(|&f| outs[trace.squeezes[f][0]][0]))
                .collect();
            let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
            let aw = build_eq_table(&alpha_vals);
            let mut accs: Vec<Wire> = (0..yr_len).map(|_| zw).collect();
            for k in 0..geo[li].q {
                let pos = (chals[lvl.q_ch + k].lo as usize) & ((1usize << geo[li].depth) - 1);
                vals.push(F128::new(pos as u64, 0));
                let qf = sb.public_input();
                vals.push(aw[k]);
                let awp = sb.public_input();
                let mut g_in = vec![qf];
                g_in.extend_from_slice(&ris_w);
                g_in.push(awp);
                g_in.push(ow);
                g_in.extend_from_slice(&accs);
                accs = sb.gate(rslot, &g_in);
            }
            resid_pub.push(accs);
        }
        let pl_full: usize = levels.iter().map(|l| l.fold_fins.len()).sum();
        let ris_full: Vec<Wire> = levels
            .iter()
            .flat_map(|l| l.fold_fins.iter().map(|&f| outs[trace.squeezes[f][0]][0]))
            .collect();
        let sxslot = sb.slot(SuffixGate::new(yr_log));
        leaf_slot.push((300, sxslot));
        let pfslot2 = sb.slot(PrefixGate::new(pl_full));
        leaf_slot.push((310 + pl_full, pfslot2));
        let mut evb_accs: Vec<Wire> = (0..yr_len).map(|_| zw).collect();
        {
            assert_eq!(
                w_rounds.len(),
                pl_full + yr_log,
                "rho spans the dense domain"
            );
            let mut g_in = vec![outs[trace.squeezes[inner_pd2.fin][0]][0]];
            for rr in &w_rounds[..pl_full] {
                g_in.push(outs[trace.squeezes[rr.fin][0]][0]);
            }
            g_in.extend_from_slice(&ris_full);
            g_in.push(ow);
            let pw2 = sb.gate(pfslot2, &g_in)[0];
            let mut s_in = vec![pw2];
            for rr in &w_rounds[pl_full..] {
                s_in.push(outs[trace.squeezes[rr.fin][0]][0]);
            }
            s_in.push(ow);
            s_in.extend_from_slice(&evb_accs);
            evb_accs = sb.gate(sxslot, &s_in);
        }
        for (li, lvl) in levels.iter().enumerate() {
            for od in &lvl.ood {
                let folded = od.z_len - yr_log;
                let later: Vec<Wire> = levels[li + 1..]
                    .iter()
                    .flat_map(|l| l.fold_fins.iter().map(|&f| outs[trace.squeezes[f][0]][0]))
                    .collect();
                assert_eq!(later.len(), folded, "OOD prefix = later folds");
                let sq = &trace.squeezes[od.z_fin];
                let mut g_in = vec![outs[trace.squeezes[od.beta_fin][0]][0]];
                for j in 0..folded {
                    g_in.push(outs[sq[j / 4]][j % 4]);
                }
                g_in.extend(repeat_n(zw, pl_full - folded));
                g_in.extend_from_slice(&later);
                g_in.extend(repeat_n(zw, pl_full - folded));
                g_in.push(ow);
                let pw2 = sb.gate(pfslot2, &g_in)[0];
                let mut s_in = vec![pw2];
                for j in 0..yr_log {
                    let jj = folded + j;
                    s_in.push(outs[sq[jj / 4]][jj % 4]);
                }
                s_in.push(ow);
                s_in.extend_from_slice(&evb_accs);
                evb_accs = sb.gate(sxslot, &s_in);
            }
        }
        let pcslot = sb.slot(PartialCombineGate::new(yr_log));
        leaf_slot.push((301, pcslot));
        let mut comb = evb_accs;
        for (li, lvl) in levels.iter().enumerate() {
            let mut g_in = vec![outs[trace.squeezes[lvl.beta_fin][0]][0]];
            g_in.extend_from_slice(&comb);
            g_in.extend_from_slice(&resid_pub[li]);
            comb = sb.gate(pcslot, &g_in);
        }
        let fdslot = sb.slot(FinalDotGate::new(yr_log));
        leaf_slot.push((302, fdslot));
        let mut g_in: Vec<Wire> = (0..yr_len).map(|y| wv(yr_v2 + y)).collect();
        g_in.extend_from_slice(&comb);
        let inner_w = sb.gate(fdslot, &g_in)[0];

        // ---- the R = 2 multipoint chains ----
        // T0 = Σ gamma^{128i+j}·A_ij over the 256 absorbed dual values
        // (gamma-power chain + tr-row MACs), the rounds via mrslot,
        // T_m + anchor.v published as a zero-delta, and the anchor's own
        // rounds folded to an endpoint checked against the native replay.
        // The anchor EXPECT (RS statements, ĝ closed form) stays
        // checker-native with the family-H batch.
        let (mp_gamma_fin, mp_rounds3, mp_anchor_v, mp_anchor_rounds3) = {
            use flock_core::transcript_record::TranscriptOp as Op5;
            let ops3 = t_shape.ops();
            let (mut v3, mut c3, mut f3, mut i3) = (0usize, 0usize, 0usize, 0usize);
            let bump3 = |op: &Op5, v: &mut usize, c: &mut usize, f: &mut usize| {
                if op.finalizes() {
                    *f += 1;
                }
                match op {
                    Op5::SqueezeScalar => *c += 1,
                    Op5::SqueezeSlice(n) => *c += n,
                    Op5::ObserveScalar => *v += 1,
                    Op5::ObserveSlice(n) => *v += n,
                    _ => {}
                }
            };
            while !matches!(&ops3[i3], Op5::Label(l) if l.as_slice() == b"flock-multipoint-twisted-v1")
            {
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                i3 += 1;
            }
            i3 += 1;
            while matches!(ops3[i3], Op5::ObserveScalar) {
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                i3 += 1;
            }
            let gfin = f3;
            bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
            i3 += 1;
            let mut rds = Vec::new();
            while matches!(ops3[i3], Op5::ObserveScalar) && !matches!(ops3[i3], Op5::Label(_)) {
                if !matches!(ops3[i3 + 2], Op5::SqueezeScalar) {
                    break;
                }
                let g_v = v3;
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                bump3(&ops3[i3 + 1], &mut v3, &mut c3, &mut f3);
                let (ch, fin) = (c3, f3);
                bump3(&ops3[i3 + 2], &mut v3, &mut c3, &mut f3);
                rds.push((g_v, ch, fin));
                i3 += 3;
            }
            assert!(
                matches!(&ops3[i3], Op5::Label(l) if l.as_slice() == b"flock-frobenius-assist-v0")
            );
            i3 += 1;
            let av = v3;
            bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
            i3 += 1;
            let mut ards = Vec::new();
            while matches!(ops3[i3], Op5::ObserveScalar) {
                let g_v = v3;
                bump3(&ops3[i3], &mut v3, &mut c3, &mut f3);
                bump3(&ops3[i3 + 1], &mut v3, &mut c3, &mut f3);
                let (ch, fin) = (c3, f3);
                bump3(&ops3[i3 + 2], &mut v3, &mut c3, &mut f3);
                ards.push((g_v, ch, fin));
                i3 += 3;
            }
            (gfin, rds, av, ards)
        };
        let mp_gamma_w = outs[trace.squeezes[mp_gamma_fin][0]][0];
        let mut mp_t0 = zw;
        let mut mp_pw = ow;
        for (k3, &vi) in val_vs.iter().enumerate() {
            mp_t0 = sb.gate(spine, &[zw, zw, zw, mp_t0, zw, zw, wv(vi), mp_pw, zw])[3];
            if k3 + 1 < val_vs.len() {
                mp_pw = sb.gate(spine, &[zw, zw, zw, zw, zw, zw, mp_pw, mp_gamma_w, zw])[3];
            }
        }
        let mut mp_tm = mp_t0;
        for &(g_v, _, fin) in &mp_rounds3 {
            let r_w = outs[trace.squeezes[fin][0]][0];
            mp_tm = sb.gate(mrslot, &[mp_tm, wv(g_v), wv(g_v + 1), r_w])[0];
        }
        let mp_delta = sb.gate(spine, &[zw, zw, zw, mp_tm, zw, zw, wv(mp_anchor_v), ow, zw])[3];
        let mut anc = wv(mp_anchor_v);
        for &(g_v, _, fin) in &mp_anchor_rounds3 {
            let r_w = outs[trace.squeezes[fin][0]][0];
            anc = sb.gate(mrslot, &[anc, wv(g_v), wv(g_v + 1), r_w])[0];
        }
        let anc_end_native = {
            let mut t = vals_rec[mp_anchor_v];
            for &(g_v, ch, _) in &mp_anchor_rounds3 {
                let (g1, gi) = (vals_rec[g_v], vals_rec[g_v + 1]);
                let r = chals[ch];
                let g0 = t + g1;
                t = g0 + (g1 + g0 + gi) * r + gi * r * r;
            }
            t
        };

        // ---- MatrixAssertion emission ----
        // The deferred matrix work exits as bound publics: alpha and the
        // lincheck round challenges (chain wires — the assertion's point),
        // plus the matrix_evals as advice publics (they are NOT absorbed —
        // deferral leaves them proof-side, pinned by the one-equation final
        // check and the root discharge, both checker-native for now).
        let mut assert_pub: Vec<Wire> = vec![alpha_w2];
        for &(_, _, fin) in &lc_rounds2 {
            assert_pub.push(outs[trace.squeezes[fin][0]][0]);
        }
        for &(a, b) in &proof.lincheck.matrix_evals {
            vals.push(a);
            assert_pub.push(sb.public_input());
            vals.push(b);
            assert_pub.push(sb.public_input());
        }

        for (a_wires, opens) in &to_publish {
            for w in a_wires {
                sb.publish(*w);
            }
            for (cw, cv) in opens {
                sb.publish(*cw);
                sb.publish(cv[0]);
                sb.publish(cv[1]);
            }
        }
        for w in &level_accs {
            sb.publish(*w);
        }
        for pp in &pow_pub {
            for w in pp {
                sb.publish(*w);
            }
        }
        for accs in &resid_pub {
            for w in accs {
                sb.publish(*w);
            }
        }
        sb.publish(inner_w);
        sb.publish(tw);
        sb.publish(runw);
        sb.publish(t_final);
        sb.publish(zc_seed_w);
        for d in &zc_deltas2 {
            sb.publish(*d);
        }
        sb.publish(zrw);
        sb.publish(lcw);
        sb.publish(mp_delta);
        sb.publish(anc);
        for w in &assert_pub {
            sb.publish(*w);
        }
        let shape = sb.finish().expect("valid leaf query-phase circuit");
        let hint_refs: Vec<&dyn Any> = hints.iter().map(|h| h as &dyn Any).collect();
        let built = shape.run(&vals, &hint_refs);

        // ---- boundary checks: alphas, cap selects, and the enforced sums.
        let n_assert_pub = 1 + lc_rounds2.len() + 2 * proof.lincheck.matrix_evals.len();
        let total_pub: usize = levels.len()
            + levels.len() * yr_len
            + 1
            + 3
            + 5
            + n_assert_pub
            + zc_rounds2.len()
            + 3 * pows.len()
            + levels
                .iter()
                .zip(&geo)
                .map(|(l, g)| l.a_count + 3 * g.q)
                .sum::<usize>();
        let mut at2 = built.public.len() - total_pub;
        for (li, lvl) in levels.iter().enumerate() {
            let g = &geo[li];
            let (cap, _, _) = lvl_src[li];
            for j in 0..lvl.a_count {
                assert_eq!(
                    built.public[at2 + j],
                    chals[lvl.a_ch + j],
                    "L{li} alpha {j}"
                );
            }
            at2 += lvl.a_count;
            for k in 0..g.q {
                let chal = chals[lvl.q_ch + k];
                assert_eq!(built.public[at2], chal, "L{li} challenge {k}");
                let pos = (chal.lo as usize) & ((1usize << g.depth) - 1);
                let node = digest_words(&hash_to_digest(&cap[pos >> g.path]));
                assert_eq!(
                    [built.public[at2 + 1], built.public[at2 + 2]],
                    node,
                    "L{li} cap node {k}"
                );
                at2 += 3;
            }
        }
        for (li, want) in native_sums.iter().enumerate() {
            assert_eq!(
                built.public[at2 + li],
                *want,
                "L{li} enforced sum matches the native replica"
            );
        }
        // The residual accumulators, against the native induced-basis
        // replica (sks replicated via sk_at_vks — the mvp7 discipline).
        let resid_base = at2 + native_sums.len() + 3 * pows.len();
        let mut resid_native: Vec<Vec<F128>> = vec![vec![F128::ZERO; yr_len]; levels.len()];
        {
            let mut at3 = resid_base;
            for (li, lvl) in levels.iter().enumerate() {
                let pl: usize = levels[li + 1..].iter().map(|l| l.fold_fins.len()).sum();
                let lmc = pl + yr_log;
                let sks = sk_at_vks(lmc);
                let inv = |v: F128| if v == F128::ZERO { F128::ZERO } else { v.inv() };
                let ris: Vec<F128> = levels[li + 1..]
                    .iter()
                    .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
                    .collect();
                let alpha_vals: Vec<F128> = (0..lvl.a_count).map(|j| chals[lvl.a_ch + j]).collect();
                let aw = build_eq_table(&alpha_vals);
                for y in 0..yr_len {
                    let mut sum = F128::ZERO;
                    for k in 0..geo[li].q {
                        let pos =
                            (chals[lvl.q_ch + k].lo as usize) & ((1usize << geo[li].depth) - 1);
                        let mut sk = Vec::with_capacity(lmc);
                        if lmc > 0 {
                            sk.push(F128::new(pos as u64, 0));
                            for j in 1..lmc {
                                sk.push(sk[j - 1] * sk[j - 1] + sks[j - 1] * sk[j - 1]);
                            }
                        }
                        let mut prod = F128::ONE;
                        for j in 0..pl {
                            prod *= F128::ONE + ris[j] * (F128::ONE + sk[j] * inv(sks[j]));
                        }
                        for j in 0..yr_log {
                            if (y >> j) & 1 == 1 {
                                prod *= sk[pl + j] * inv(sks[pl + j]);
                            }
                        }
                        sum += aw[k] * prod;
                    }
                    assert_eq!(built.public[at3], sum, "L{li} residual y={y}");
                    resid_native[li][y] = sum;
                    at3 += 1;
                }
            }
            // The close-out inner, natively — and THE CLOSURE: it must
            // equal the spine's native t_r, so `inner == t_r` holds
            // between two circuit outputs.
            let ris_v: Vec<F128> = levels
                .iter()
                .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
                .collect();
            let plf = ris_v.len();
            let mut inner_n = F128::ZERO;
            for y in 0..yr_len {
                let mut evb = chals[inner_pd2.ch];
                for j in 0..plf {
                    evb *= F128::ONE + chals[w_rounds[j].ch] + ris_v[j];
                }
                for j in 0..yr_log {
                    evb *= if (y >> j) & 1 == 1 {
                        chals[w_rounds[plf + j].ch]
                    } else {
                        F128::ONE + chals[w_rounds[plf + j].ch]
                    };
                }
                let mut combn = evb;
                for (li, lvl) in levels.iter().enumerate() {
                    combn += chals[lvl.beta_ch] * resid_native[li][y];
                    for od in &lvl.ood {
                        let folded = od.z_len - yr_log;
                        let later: Vec<F128> = levels[li + 1..]
                            .iter()
                            .flat_map(|l| l.fold_chs.iter().map(|&i| chals[i]))
                            .collect();
                        let mut t = chals[od.beta_ch];
                        for j in 0..folded {
                            t *= F128::ONE + chals[od.z_ch + j] + later[j];
                        }
                        for j in 0..yr_log {
                            t *= if (y >> j) & 1 == 1 {
                                chals[od.z_ch + folded + j]
                            } else {
                                F128::ONE + chals[od.z_ch + folded + j]
                            };
                        }
                        combn += t;
                    }
                }
                inner_n += vals_rec[yr_v2 + y] * combn;
            }
            assert_eq!(built.public[at3], inner_n, "the close-out inner");
            // THE CLOSURE, between circuit outputs: the residual side's
            // inner and the spine's t_r are the same statement scalar.
            let zc_tail2 = n_assert_pub + 5 + zc_rounds2.len();
            assert_eq!(
                built.public[at3],
                built.public[built.public.len() - zc_tail2 - 1],
                "inner == t_r: the leaf statement closes"
            );
        }
        let pow_base = at2 + native_sums.len();
        for (k, pr) in pows.iter().enumerate() {
            let d0 = built.public[pow_base + 3 * k];
            let d1 = built.public[pow_base + 3 * k + 1];
            let nn = built.public[pow_base + 3 * k + 2];
            let mut digest = [0u8; 32];
            digest[..8].copy_from_slice(&d0.lo.to_le_bytes());
            digest[8..16].copy_from_slice(&d0.hi.to_le_bytes());
            digest[16..24].copy_from_slice(&d1.lo.to_le_bytes());
            digest[24..].copy_from_slice(&d1.hi.to_le_bytes());
            assert_eq!(nn.hi, 0, "pow {k}: nonce word zero-padded");
            if pr.bits == 0 {
                assert_eq!(nn.lo, 0, "pow {k}: canonical zero nonce");
            } else {
                assert!(
                    pow_has_leading_zero_bits(&digest, nn.lo, pr.bits, HashKind::Blake3,),
                    "pow {k}: grinding predicate on the published wires"
                );
            }
        }
        // Tail order: [.., zc_end, lc_end, mp_delta, anc, assertion fields].
        {
            let base = built.public.len() - n_assert_pub;
            assert_eq!(built.public[base], chals[alpha_ch2], "assertion alpha");
            for (k4, &(_, ch, _)) in lc_rounds2.iter().enumerate() {
                assert_eq!(built.public[base + 1 + k4], chals[ch], "assertion r {k4}");
            }
            let me_base = base + 1 + lc_rounds2.len();
            for (k4, &(a, b)) in proof.lincheck.matrix_evals.iter().enumerate() {
                assert_eq!(built.public[me_base + 2 * k4], a, "matrix eval a {k4}");
                assert_eq!(built.public[me_base + 2 * k4 + 1], b, "matrix eval b {k4}");
            }
        }
        let tail0 = built.public.len() - n_assert_pub;
        assert_eq!(
            built.public[tail0 - 2],
            F128::ZERO,
            "T_m + anchor.v is the multipoint zero-delta"
        );
        assert_eq!(
            built.public[tail0 - 1],
            anc_end_native,
            "the anchor rounds end at the native claim"
        );
        assert_eq!(
            built.public[tail0 - 3],
            lc_end_native,
            "the lincheck chain ends at the native running claim"
        );
        let zc_tail = n_assert_pub + 5 + zc_rounds2.len();
        assert_eq!(
            built.public[built.public.len() - zc_tail],
            zc_seed,
            "the zc seed advice is the native skip output"
        );
        for k2 in 0..zc_rounds2.len() {
            assert_eq!(
                built.public[built.public.len() - zc_tail + 1 + k2],
                F128::ZERO,
                "zc delta {k2}"
            );
        }
        assert_eq!(
            built.public[tail0 - 4],
            zc_end_native,
            "the zc chain ends at the native running claim"
        );
        // The intake boundary: the advice target and the in-circuit
        // running, both checker-validated against the native replay.
        assert_eq!(
            built.public[built.public.len() - zc_tail - 3],
            native_target,
            "the RS target advice is the native gamma-combination"
        );
        assert_eq!(
            built.public[built.public.len() - zc_tail - 2],
            native_running,
            "the W-rounds fold the target to the native running claim"
        );
        // The spine's t_r, against the native quad replay.
        {
            let quad = |u0: F128, u2: F128, t: F128| (u0, t + u2, u2);
            let evalq = |q: (F128, F128, F128), x: F128| q.0 + x * q.1 + x * x * q.2;
            let mut nt = chals[inner_pd2.ch] * vals_rec[inner_pd2.q_v];
            let mut nq = quad(vals_rec[start_v], vals_rec[start_v + 1], nt);
            for (li, lvl) in levels.iter().enumerate() {
                for (j, &mv) in lvl.fold_msg_vs.iter().enumerate() {
                    nt = evalq(nq, chals[lvl.fold_chs[j]]);
                    nq = quad(vals_rec[mv], vals_rec[mv + 1], nt);
                }
                if li < levels.len() - 1 {
                    for od in &lvl.ood {
                        let b = chals[od.beta_ch];
                        let iq = quad(
                            vals_rec[od.intro_v],
                            vals_rec[od.intro_v + 1],
                            vals_rec[od.y_v],
                        );
                        nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                        nt += b * vals_rec[od.y_v];
                    }
                    let b = chals[lvl.beta_ch];
                    let iq = quad(
                        vals_rec[lvl.intro_v],
                        vals_rec[lvl.intro_v + 1],
                        native_sums[li],
                    );
                    nq = (nq.0 + b * iq.0, nq.1 + b * iq.1, nq.2 + b * iq.2);
                    nt += b * native_sums[li];
                } else {
                    nt += chals[lvl.beta_ch] * native_sums[li];
                }
            }
            assert_eq!(
                built.public[built.public.len() - zc_tail - 1],
                nt,
                "the spine's final t_r matches the native replay"
            );
        }

        // ---- prove / verify the leaf query-phase circuit ----
        let union_o = UnionInstance::new(&shape.registry, shape.counts.clone());
        let pcs_o = PcsParams {
            m: union_o.dense_m(),
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: LigeritoProfile::Fast,
            num_lanes: union_o.commit_lanes(6),
            merkle_hash: Default::default(),
        };
        let b3_r1cs = blake3::build_block_r1cs(nu);
        let b3_lc = b3_r1cs.csc_lincheck_circuit();
        let swap_r1cs = SwapTable::build_block_r1cs(nu);
        let swap_lc = swap_r1cs.csc_lincheck_circuit();
        let spread_ty = BitSpreadTable::new(max_path);
        let spread_r1cs = spread_ty.build_block_r1cs(nu);
        let spread_lc = spread_r1cs.csc_lincheck_circuit();
        let b3_wit =
            blake3::generate_witness_batch_major_partial(built.rows::<Blake3Gate>(slots.b3), nu);
        let swap_wit =
            SwapTable::generate_witness_batch_major(built.rows::<SwapGate>(slots.swap), nu);
        let spread_wit =
            spread_ty.generate_witness_batch_major(built.rows::<BitSpreadGate>(slots.spread), nu);
        let els: Vec<Vec<F128>> = leaf_slot
            .iter()
            .map(|(_, sl)| match &built.witnesses[shape.registry_slot(*sl)] {
                SlotWitness::Element(z) => z.clone(),
                other => panic!("leaf-eval slot produced {other:?}"),
            })
            .collect();
        let mut lcs_ord: Vec<(usize, &dyn LincheckCircuit)> = vec![
            (shape.registry_slot(slots.b3), b3_lc),
            (shape.registry_slot(slots.swap), swap_lc),
            (shape.registry_slot(slots.spread), spread_lc),
        ];
        lcs_ord.sort_by_key(|(i, _)| *i);
        let lcs: Vec<&dyn LincheckCircuit> = lcs_ord.into_iter().map(|(_, cc)| cc).collect();
        let ((oproof, ocommit), prove_t) = timed(REPS, || {
            let mut bool_slots: Vec<(usize, UnionSlotProverInput)> = vec![
                (
                    shape.registry_slot(slots.b3),
                    UnionSlotProverInput::new(b3_wit.clone(), b3_lc),
                ),
                (
                    shape.registry_slot(slots.swap),
                    UnionSlotProverInput::new(swap_wit.clone(), swap_lc),
                ),
                (
                    shape.registry_slot(slots.spread),
                    UnionSlotProverInput::new(spread_wit.clone(), spread_lc),
                ),
            ];
            bool_slots.sort_by_key(|(i, _)| *i);
            let mut el_ord: Vec<(usize, Vec<F128>)> = leaf_slot
                .iter()
                .zip(els.clone())
                .map(|((_, sl), z)| (shape.registry_slot(*sl), z))
                .collect();
            el_ord.sort_by_key(|(i, _)| *i);
            let el_inputs: Vec<UnionElementSlotInput> = el_ord
                .into_iter()
                .map(|(_, z)| {
                    UnionElementSlotInput::new(move |dst: &mut [F128]| dst.copy_from_slice(&z))
                })
                .collect();
            let mut ch = FsChallenger::new(DOMAIN);
            let (p, c, _) = prover::prove_fast_ligerito_union_circuit(
                &union_o,
                &shape.circuit,
                &built.public,
                &pcs_o,
                bool_slots.into_iter().map(|(_, x)| x).collect(),
                el_inputs,
                &mut ch,
            );
            (p, c)
        });
        let (_, verify_t) = timed(REPS, || {
            let mut ch = FsChallenger::new(DOMAIN);
            verifier::verify_ligerito_union_circuit(
                &union_o,
                &shape.circuit,
                &built.public,
                &lcs,
                &ocommit,
                &oproof,
                &pcs_o,
                &mut ch,
            )
            .expect("the leaf query-phase circuit verifies")
        });
        println!(
            "\nMVP-9 BOOLEAN LEAF (inner: blake3 workload m=22, rs×2 pd=0)\n  \
             nu {} | dense_m {} | mu {} | b3 rows {}\n\n  \
             medians of {REPS} runs, spread in brackets\n  \
             PER PROOF     prove {prove_t} ms\n  \
             verifier side {verify_t} ms | proof {:.1} KiB | {} threads\n",
            nu,
            union_o.dense_m(),
            shape.circuit.cells().mu(),
            b3_rows,
            bincode::serialize(&oproof).map(|b| b.len()).unwrap_or(0) as f64 / 1024.0,
            threads,
        );
    }
}
