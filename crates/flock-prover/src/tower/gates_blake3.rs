use std::array::from_fn;

use flock_hash::blake3_compress;
#[cfg(test)]
use {
    crate::r1cs_hashes::merkle_r1cs::{ChunkPathInput, MerkleTreeLayout, blake3_spec},
    flock_core::circuit::builder::{CircuitBuilder, Wire},
    flock_merkle::{HashKind, merkle_tree},
};

use crate::{
    r1cs_hashes::blake3::{Compression, build_block_r1cs, io_schema},
    tower::{F128, GateType, SLOT_WORDS, SlotWitness, TableType},
};

pub(super) const DOMAIN: &[u8] = b"flock-circuit-merkle-v0";

pub(super) const CHUNK_START: u32 = 1 << 0;
pub(super) const CHUNK_END: u32 = 1 << 1;
pub(super) const PARENT: u32 = 1 << 2;
pub(super) const ROOT: u32 = 1 << 3;

pub(super) const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

pub(super) fn pack4(w: [u32; 4]) -> F128 {
    F128::new(
        w[0] as u64 | ((w[1] as u64) << 32),
        w[2] as u64 | ((w[3] as u64) << 32),
    )
}

pub(super) fn unpack4(v: F128) -> [u32; 4] {
    [
        v.lo as u32,
        (v.lo >> 32) as u32,
        v.hi as u32,
        (v.hi >> 32) as u32,
    ]
}

pub(super) fn pack8(w: &[u32; 8]) -> [F128; 2] {
    [
        pack4([w[0], w[1], w[2], w[3]]),
        pack4([w[4], w[5], w[6], w[7]]),
    ]
}

pub(super) fn pack_params(counter: u64, block_len: u32, flags: u32) -> F128 {
    F128::new(counter, block_len as u64 | ((flags as u64) << 32))
}

pub(super) fn unpack_params(v: F128) -> (u64, u32, u32) {
    (v.lo, v.hi as u32, (v.hi >> 32) as u32)
}

/// A 128-bit word of leaf data: bytes `[o, o+16)` little-endian, which is
/// exactly how the message region reads them (`leaf_msg_words` is LE `u32`s,
/// and committed bit `t` of a word is bit `t` of `lo`).
#[cfg(test)]
pub(super) fn leaf_word(data: &[u8], o: usize) -> F128 {
    F128::new(
        u64::from_le_bytes(data[o..o + 8].try_into().unwrap()),
        u64::from_le_bytes(data[o + 8..o + 16].try_into().unwrap()),
    )
}

pub(super) fn unpack8(a: F128, b: F128) -> [u32; SLOT_WORDS] {
    let (x, y) = (unpack4(a), unpack4(b));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}

pub(super) fn digest_words(d: &[u32; SLOT_WORDS]) -> [F128; 2] {
    [
        pack4([d[0], d[1], d[2], d[3]]),
        pack4([d[4], d[5], d[6], d[7]]),
    ]
}

pub(super) fn hash_to_digest(h: &[u8; 32]) -> [u32; SLOT_WORDS] {
    from_fn(|w| u32::from_le_bytes(h[4 * w..4 * w + 4].try_into().unwrap()))
}

#[cfg(test)]
pub(super) struct Rng(pub(super) u64);
#[cfg(test)]
impl Rng {
    pub(super) fn next_u32(&mut self) -> u32 {
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
/// `tests/circuit_builder.rs`; duplicated rather than shared because a lib
/// module cannot import from the crate's `tests/` binaries.)
pub(super) struct Blake3Gate {
    pub(super) nu: usize,
}

impl GateType for Blake3Gate {
    type Row = Compression;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&build_block_r1cs(self.nu)).with_io_schema(io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let cv: [u32; 8] = {
            let (a, b) = (unpack4(inputs[0]), unpack4(inputs[1]));
            [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        };
        let mut m = [0u32; 16];
        for i in 0..4 {
            m[4 * i..4 * i + 4].copy_from_slice(&unpack4(inputs[2 + i]));
        }
        let (counter, block_len, flags) = unpack_params(inputs[6]);
        let out = blake3_compress(&cv, &m, counter, block_len, flags);
        let lo = pack8(&out[0..8].try_into().unwrap());
        let hi = pack8(&out[8..16].try_into().unwrap());
        outputs.extend_from_slice(&[lo[0], lo[1], hi[0], hi[1]]);
        (cv, m, counter, block_len, flags)
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// One chunk-leaf Merkle opening: leaf data and an index word in, the root
/// out, the sibling path as a hint.
#[cfg(test)]
pub(super) struct MerklePathGate {
    pub(super) layout: MerkleTreeLayout,
    pub(super) nu: usize,
}

#[cfg(test)]
impl MerklePathGate {
    /// `block_len` is the PCS block length the opening's index will be
    /// sampled against — `sample_queries` masks a challenge with
    /// `block_len − 1`, and the relation reads the index word's low `depth`
    /// bits, so the two agree only when `depth = log2(block_len)`. Asserting
    /// it here means a circuit cannot silently wire a challenge that the
    /// relation truncates differently than the sampler did.
    ///
    /// Protocol paths are capped. They are
    /// `d − c` deep and the index's high `c` bits select a node of the
    /// absorbed cap layer rather than folding to a root. The COLLAPSED path
    /// models this with `emit_opening` and the boundary select. The
    /// select is done by the checker on published words, so no mux gadget
    /// exists in-circuit. This COMPOSITE gate stays full-depth against its
    /// synthetic trees, where the assert above IS the index-binding
    /// argument; it is the uncapped differential oracle, not the protocol.
    pub(super) fn new(depth: usize, leaf_bytes: usize, nu: usize, block_len: usize) -> Self {
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

#[cfg(test)]
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

    fn eval(&self, inputs: &[F128], hint: &Self::Hint, outputs: &mut Vec<F128>) -> Self::Row {
        let (o, row) = {
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
        };
        outputs.extend_from_slice(&o);
        row
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

// ---------------------------------------------------------------------------

/// A tree, and one opening's siblings out of it.
#[cfg(test)]
pub(super) struct Tree {
    pub(super) data: Vec<u8>,
    pub(super) flat: Vec<[u8; 32]>,
    pub(super) root: [u8; 32],
    pub(super) depth: usize,
    pub(super) leaf_bytes: usize,
}

#[cfg(test)]
impl Tree {
    pub(super) fn new(depth: usize, leaf_bytes: usize, rng: &mut Rng) -> Self {
        let n_leaves = 1usize << depth;
        let data: Vec<u8> = (0..n_leaves * leaf_bytes)
            .map(|_| rng.next_u32() as u8)
            .collect();
        let flat = merkle_tree(&data, n_leaves, HashKind::Blake3);
        let root = flat[flat.len() - 1];
        Self {
            data,
            flat,
            root,
            depth,
            leaf_bytes,
        }
    }

    pub(super) fn leaf(&self, pos: usize) -> &[u8] {
        &self.data[pos * self.leaf_bytes..(pos + 1) * self.leaf_bytes]
    }

    pub(super) fn siblings(&self, pos: usize) -> Vec<[u32; SLOT_WORDS]> {
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
#[cfg(test)]
pub(super) fn table_index(pos: usize, _depth: usize) -> u128 {
    pos as u128
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
pub(super) fn circuit_structure_does_not_depend_on_the_witness() {
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
