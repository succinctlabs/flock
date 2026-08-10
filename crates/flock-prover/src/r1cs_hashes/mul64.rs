//! u64 × u64 → u128 batch multiplication R1CS, 15 independent products per
//! K_LOG=17 block (94.5% fill, keccak3-style packing).
//!
//! Per-product statement: the committed sub-block holds `p`, `q` (64 bits
//! each, free wires via identity rows) together with the full schoolbook
//! multiplication transcript: all 4096 partial-product bits `pp[i][j] =
//! p_i ∧ q_j` and one committed AND output per carry-save adder (4032 of
//! them). The 128 product bits are *linear* combinations of committed bits
//! (never committed themselves); in any satisfying assignment they equal the
//! bits of `p·q` because every adder preserves the integer identity
//! `x + y + c = sum + 2·carry` exactly.
//!
//! Sub-block layout (`SUB_BITS = 8256` committed slots per product):
//! - `[0, 64)`    p bits (identity rows: `z∧z = z`, free)
//! - `[64, 128)`  q bits (identity rows)
//! - `[128, 4224)` partial products, slot `128 + 64·i + j` = `p_i ∧ q_j`
//! - `[4224, 8256)` adder aux bits, one per full/half adder, in schedule
//!   order. A full adder on pool expressions `(x, y, c)` commits
//!   `t = (x⊕c)∧(y⊕c)`; its sum `x⊕y⊕c` and carry `t⊕c` stay linear. A
//!   half adder on `(u, v)` commits `t = u∧v` (= its carry); sum `u⊕v`.
//!
//! The adder count is exact: column populations follow `L_{k+1} = c_{k+1} +
//! ⌊L_k/2⌋` with `⌊L_k/2⌋` adders per column, totalling 4032 (the ~64
//! half-adders at even-population columns are why one product misses a
//! K_LOG=13 block by 64 bits — hence the 15-wide packing). Column 127 needs
//! no adders: since `p·q < 2^128`, the leftover weight-127 pool sums to 0 or
//! 1 over ℤ, so bit 127 is its plain XOR.
//!
//! The adder schedule is column-serial (weight 0 → 127), Huffman-style: each
//! step combines the three smallest-support expressions, which keeps the
//! substituted linear combinations (and hence A₀/B₀ nnz) small.

use std::sync::OnceLock;

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

pub const K_LOG: usize = 17;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;

/// Independent u64×u64 products packed into one block.
pub const MULS_PER_BLOCK: usize = 15;

/// Committed slots per product (128 inputs + 4096 partial products + 4032
/// adder aux bits). Asserted against the built circuit.
pub const SUB_BITS: usize = 8256;

pub const P_BASE: usize = 0;
pub const Q_BASE: usize = 64;
pub const PP_BASE: usize = 128;
pub const T_BASE: usize = PP_BASE + 64 * 64; // 4224

pub const USEFUL_BITS: usize = MULS_PER_BLOCK * SUB_BITS; // 123,840 of 131,072

/// Ligerito configs ship for m ≥ 22, so the batch floor is 2^(22 − K_LOG)
/// blocks = 32 blocks = 480 products.
pub const MIN_N_BLOCKS_LOG: usize = 22 - K_LOG; // 5

// ───────────────────────────────────────────────────────────────────────────
// Base circuit description (one product; built once, shared)
// ───────────────────────────────────────────────────────────────────────────

/// Evaluation-order node of the multiplication circuit. Node ids `[0, 64)`
/// are the p bits and `[64, 128)` the q bits (`Input`); every other node
/// references strictly earlier ids.
#[derive(Clone, Copy, Debug)]
pub enum Node {
    Input,
    /// `val = v[a] ^ v[b]` — free linear node (never committed).
    Xor(u32, u32),
    /// `val = v[a] & v[b]` — committed at base slot `slot` (partial products
    /// and adder aux bits). The R1CS row at `slot` has A/B-side supports
    /// equal to the supports of nodes `a`/`b`.
    And { a: u32, b: u32, slot: u32 },
}

pub struct MulCircuit {
    /// All nodes in evaluation order (ids `[0,128)` are `Input`).
    pub nodes: Vec<Node>,
    /// For base slot `s < SUB_BITS`: node whose value is `z[s]`.
    pub slot_nodes: Vec<u32>,
    /// For base slot `s`: nodes whose values are the A-side / B-side of row
    /// `s` (for the fused (z, a, b) witness builder).
    pub row_ab_nodes: Vec<(u32, u32)>,
    /// For base slot `s`: the A/B row supports (sorted base-slot indices).
    pub row_supports: Vec<(Vec<usize>, Vec<usize>)>,
    /// Node ids of the 128 product bits (linear; for tests and future IO).
    pub product_bit_nodes: [u32; 128],
    pub n_adders: usize,
    /// Total nonzeros across one product's A₀+B₀ rows (diagnostic).
    pub nnz: usize,
}

/// Sorted-symmetric-difference of two sorted slot lists (GF(2) support xor).
fn symdiff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

struct PoolEntry {
    node: u32,
    supp: Vec<usize>,
}

fn build_circuit() -> MulCircuit {
    let mut nodes: Vec<Node> = Vec::with_capacity(26_000);
    for _ in 0..128 {
        nodes.push(Node::Input);
    }

    let mut slot_nodes: Vec<u32> = Vec::with_capacity(SUB_BITS);
    let mut row_ab_nodes: Vec<(u32, u32)> = Vec::with_capacity(SUB_BITS);
    let mut row_supports: Vec<(Vec<usize>, Vec<usize>)> = Vec::with_capacity(SUB_BITS);

    // Identity rows for the free input wires.
    for s in 0..128u32 {
        slot_nodes.push(s);
        row_ab_nodes.push((s, s));
        row_supports.push((vec![s as usize], vec![s as usize]));
    }

    // Partial products, and the weight-k pools they seed.
    let mut pools: Vec<Vec<PoolEntry>> = (0..129).map(|_| Vec::new()).collect();
    for i in 0..64u32 {
        for j in 0..64u32 {
            let slot = (PP_BASE as u32) + 64 * i + j;
            let id = nodes.len() as u32;
            nodes.push(Node::And {
                a: P_BASE as u32 + i,
                b: Q_BASE as u32 + j,
                slot,
            });
            slot_nodes.push(id);
            row_ab_nodes.push((P_BASE as u32 + i, Q_BASE as u32 + j));
            row_supports.push((vec![P_BASE + i as usize], vec![Q_BASE + j as usize]));
            pools[(i + j) as usize].push(PoolEntry {
                node: id,
                supp: vec![slot as usize],
            });
        }
    }

    // Keep each pool sorted by descending support size; the smallest entries
    // (Huffman-style) live at the tail. Ties break on insertion order, which
    // is deterministic.
    let sort_pool = |pool: &mut Vec<PoolEntry>| {
        pool.sort_by(|x, y| y.supp.len().cmp(&x.supp.len()));
    };
    let insert_sorted = |pool: &mut Vec<PoolEntry>, e: PoolEntry| {
        let pos = pool.partition_point(|x| x.supp.len() > e.supp.len());
        pool.insert(pos, e);
    };

    let mut n_adders = 0usize;
    let mut next_t_slot = T_BASE as u32;
    let mut product_bit_nodes = [0u32; 128];

    for k in 0..127 {
        let mut pool = std::mem::take(&mut pools[k]);
        sort_pool(&mut pool);

        while pool.len() >= 3 {
            // c = smallest support: it appears on both AND sides and in the
            // carry, so keeping it small keeps three supports small.
            let c = pool.pop().unwrap();
            let x = pool.pop().unwrap();
            let y = pool.pop().unwrap();

            let a_node = nodes.len() as u32;
            nodes.push(Node::Xor(x.node, c.node));
            let a_supp = symdiff(&x.supp, &c.supp);
            let b_node = nodes.len() as u32;
            nodes.push(Node::Xor(y.node, c.node));
            let b_supp = symdiff(&y.supp, &c.supp);

            let t_slot = next_t_slot;
            next_t_slot += 1;
            n_adders += 1;
            let t_node = nodes.len() as u32;
            nodes.push(Node::And {
                a: a_node,
                b: b_node,
                slot: t_slot,
            });
            slot_nodes.push(t_node);
            row_ab_nodes.push((a_node, b_node));

            // sum = x ⊕ y ⊕ c stays in this column.
            let sum_node = nodes.len() as u32;
            nodes.push(Node::Xor(a_node, y.node));
            let sum_supp = symdiff(&a_supp, &y.supp);
            // carry = t ⊕ c moves to the next column.
            let carry_node = nodes.len() as u32;
            nodes.push(Node::Xor(t_node, c.node));
            let carry_supp = symdiff(&[t_slot as usize], &c.supp);

            row_supports.push((a_supp, b_supp));
            insert_sorted(
                &mut pool,
                PoolEntry {
                    node: sum_node,
                    supp: sum_supp,
                },
            );
            pools[k + 1].push(PoolEntry {
                node: carry_node,
                supp: carry_supp,
            });
        }

        if pool.len() == 2 {
            // Half adder: commits t = u ∧ v (= the carry); sum is bit k.
            let v = pool.pop().unwrap();
            let u = pool.pop().unwrap();
            let t_slot = next_t_slot;
            next_t_slot += 1;
            n_adders += 1;
            let t_node = nodes.len() as u32;
            nodes.push(Node::And {
                a: u.node,
                b: v.node,
                slot: t_slot,
            });
            slot_nodes.push(t_node);
            row_ab_nodes.push((u.node, v.node));
            row_supports.push((u.supp, v.supp));

            let sum_node = nodes.len() as u32;
            nodes.push(Node::Xor(u.node, v.node));
            product_bit_nodes[k] = sum_node;
            pools[k + 1].push(PoolEntry {
                node: t_node,
                supp: vec![t_slot as usize],
            });
        } else {
            assert_eq!(pool.len(), 1, "column {k} ended with {} exprs", pool.len());
            product_bit_nodes[k] = pool.pop().unwrap().node;
        }
    }

    // Column 127: p·q < 2^128 forces the leftover weight-127 pool to sum to
    // 0 or 1 over ℤ, so bit 127 is the plain XOR of the pool — no adders.
    let top = std::mem::take(&mut pools[127]);
    assert!(!top.is_empty(), "weight-127 pool unexpectedly empty");
    let mut acc = top[0].node;
    for e in &top[1..] {
        let id = nodes.len() as u32;
        nodes.push(Node::Xor(acc, e.node));
        acc = id;
    }
    product_bit_nodes[127] = acc;

    assert_eq!(
        T_BASE + n_adders,
        SUB_BITS,
        "adder count drifted from the documented layout"
    );
    let nnz: usize = row_supports.iter().map(|(a, b)| a.len() + b.len()).sum();

    MulCircuit {
        nodes,
        slot_nodes,
        row_ab_nodes,
        row_supports,
        product_bit_nodes,
        n_adders,
        nnz,
    }
}

pub fn circuit() -> &'static MulCircuit {
    static CIRCUIT: OnceLock<MulCircuit> = OnceLock::new();
    CIRCUIT.get_or_init(build_circuit)
}

// ───────────────────────────────────────────────────────────────────────────
// R1CS: 15 shifted copies of the base rows per block
// ───────────────────────────────────────────────────────────────────────────

pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let circ = circuit();
    let mut a_rows: Vec<Vec<usize>> = Vec::with_capacity(K);
    let mut b_rows: Vec<Vec<usize>> = Vec::with_capacity(K);
    for sub in 0..MULS_PER_BLOCK {
        let off = sub * SUB_BITS;
        for (a, b) in &circ.row_supports {
            a_rows.push(a.iter().map(|&c| c + off).collect());
            b_rows.push(b.iter().map(|&c| c + off).collect());
        }
    }
    a_rows.resize(K, Vec::new());
    b_rows.resize(K, Vec::new());
    (
        SparseBinaryMatrix {
            num_rows: K,
            num_cols: K,
            rows: a_rows,
        },
        SparseBinaryMatrix {
            num_rows: K,
            num_cols: K,
            rows: b_rows,
        },
    )
}

/// `BlockR1cs` for `2^n_blocks_log` blocks of [`MULS_PER_BLOCK`] u64
/// multiplications. All-zero padding (sub-)blocks are trivially satisfying
/// (0·0 = 0), so no constant-wire pin is needed.
pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices();
    super::common::build_block_r1cs_with_matrices(
        n_blocks_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        a_0,
        b_0,
        None,
    )
}

// ───────────────────────────────────────────────────────────────────────────
// Witness generation — 64 blocks per lane group, bit-sliced over u64
// ───────────────────────────────────────────────────────────────────────────

/// 64×64 bit-matrix transpose, LSB-first (output word t bit s = input word s
/// bit t). Hacker's Delight 7-3 delta-swap network (same as the private copy
/// in `flock_core::r1cs`). Shared with the Karatsuba encoder.
#[inline]
pub(crate) fn transpose_64x64(a: &mut [u64; 64]) {
    let mut j: usize = 32;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k: usize = 0;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// Evaluate all circuit nodes for 64 instances at once. `p_lanes[b]` /
/// `q_lanes[b]` hold bit `b` of p/q across the 64 lanes; `vals` is scratch
/// of length `nodes.len()`.
fn eval_lanes(circ: &MulCircuit, p_lanes: &[u64; 64], q_lanes: &[u64; 64], vals: &mut [u64]) {
    vals[..64].copy_from_slice(p_lanes);
    vals[64..128].copy_from_slice(q_lanes);
    for (id, node) in circ.nodes.iter().enumerate().skip(128) {
        vals[id] = match *node {
            Node::Input => unreachable!("inputs are the first 128 nodes"),
            Node::Xor(x, y) => vals[x as usize] ^ vals[y as usize],
            Node::And { a, b, .. } => vals[a as usize] & vals[b as usize],
        };
    }
}

/// Fused witness builder: returns `(z, a, b, z_lincheck)` packed exactly like
/// `common::drive_witness_packed_and_lincheck` (RowMajor layout), but
/// bit-sliced: each of the 15 sub-products is evaluated for 64 blocks at a
/// time in u64 lanes, then bit-transposed into the packed per-block rows.
///
/// `pairs[b * MULS_PER_BLOCK + s]` is block `b`'s sub-product `s`; missing
/// tail entries are all-zero products.
pub fn generate_witness_with_ab_packed_and_lincheck(
    pairs: &[(u64, u64)],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    use rayon::prelude::*;

    let circ = circuit();
    let n_total = 1usize << n_blocks_log;
    assert!(pairs.len() <= n_total * MULS_PER_BLOCK);
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "lincheck stripe layout requires n_total ≥ 8 and divisible by 8"
    );

    let f128_per_block = K / 128;
    let u64_per_block = K / 64;
    let sub_words = SUB_BITS / 64; // 129
    let total_f128 = n_total * f128_per_block;

    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut stripe = vec![0u8; (n_total / 8) * K];

    // One parallel work item = up to 64 blocks (u64 lanes).
    let group_f128 = 64 * f128_per_block;
    let group_stripe = 8 * K;

    z.par_chunks_mut(group_f128)
        .zip(a.par_chunks_mut(group_f128))
        .zip(b.par_chunks_mut(group_f128))
        .zip(stripe.par_chunks_mut(group_stripe))
        .enumerate()
        .for_each_init(
            || vec![0u64; circ.nodes.len()],
            |vals, (g, (((z_grp, a_grp), b_grp), stripe_grp))| {
                // SAFETY: F128 = repr(C, align(16)) two LE u64s; zero bytes
                // are F128::ZERO (buffers come from the uninit scratch pool).
                unsafe {
                    std::ptr::write_bytes(z_grp.as_mut_ptr(), 0, z_grp.len());
                    std::ptr::write_bytes(a_grp.as_mut_ptr(), 0, a_grp.len());
                    std::ptr::write_bytes(b_grp.as_mut_ptr(), 0, b_grp.len());
                }
                let base_block = g * 64;
                let blocks_here = z_grp.len() / f128_per_block;
                if base_block * MULS_PER_BLOCK >= pairs.len() {
                    return; // all-padding group stays zero
                }

                let z_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(z_grp.as_mut_ptr() as *mut u64, z_grp.len() * 2)
                };
                let a_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(a_grp.as_mut_ptr() as *mut u64, a_grp.len() * 2)
                };
                let b_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(b_grp.as_mut_ptr() as *mut u64, b_grp.len() * 2)
                };

                for sub in 0..MULS_PER_BLOCK {
                    // Lane l = block (base_block + l), product index
                    // (base_block + l)·15 + sub.
                    let mut p_lanes = [0u64; 64];
                    let mut q_lanes = [0u64; 64];
                    let mut any = false;
                    for l in 0..blocks_here {
                        let idx = (base_block + l) * MULS_PER_BLOCK + sub;
                        if let Some(&(p, q)) = pairs.get(idx) {
                            p_lanes[l] = p;
                            q_lanes[l] = q;
                            any = true;
                        }
                    }
                    if !any {
                        continue;
                    }
                    transpose_64x64(&mut p_lanes);
                    transpose_64x64(&mut q_lanes);
                    eval_lanes(circ, &p_lanes, &q_lanes, vals);

                    // Emit z/a/b: per 64-slot chunk, transpose lane-major →
                    // block-major and scatter into the packed rows.
                    let word_off = sub * sub_words;
                    let mut zc = [0u64; 64];
                    let mut ac = [0u64; 64];
                    let mut bc = [0u64; 64];
                    for c in 0..sub_words {
                        let s0 = c * 64;
                        for o in 0..64 {
                            let s = s0 + o;
                            zc[o] = vals[circ.slot_nodes[s] as usize];
                            let (an, bn) = circ.row_ab_nodes[s];
                            ac[o] = vals[an as usize];
                            bc[o] = vals[bn as usize];
                        }
                        transpose_64x64(&mut zc);
                        transpose_64x64(&mut ac);
                        transpose_64x64(&mut bc);
                        for l in 0..blocks_here {
                            z_u64[l * u64_per_block + word_off + c] = zc[l];
                            a_u64[l * u64_per_block + word_off + c] = ac[l];
                            b_u64[l * u64_per_block + word_off + c] = bc[l];
                        }
                    }
                }

                // Lincheck byte-stripe: transpose 8 blocks' z words.
                for sub8 in 0..blocks_here / 8 {
                    let out = &mut stripe_grp[sub8 * K..(sub8 + 1) * K];
                    for w in 0..u64_per_block {
                        let lanes: [u64; 8] =
                            std::array::from_fn(|s| z_u64[(sub8 * 8 + s) * u64_per_block + w]);
                        transpose_8_u64s_to_64_bytes(&lanes, &mut out[w * 64..w * 64 + 64]);
                    }
                }
            },
        );

    (z, a, b, stripe)
}

// ───────────────────────────────────────────────────────────────────────────
// Setup
// ───────────────────────────────────────────────────────────────────────────

/// Minimum `n_blocks_log` for `n_muls` products at 15 per block, subject to
/// the Ligerito config floor (m ≥ 22 ⇒ ≥ 2^5 blocks).
pub fn min_n_blocks_log(n_muls: usize) -> usize {
    assert!(n_muls >= 1);
    let blocks = n_muls.div_ceil(MULS_PER_BLOCK).max(1 << MIN_N_BLOCKS_LOG);
    blocks.next_power_of_two().trailing_zeros() as usize
}

#[derive(Clone, Debug)]
pub struct Mul64Setup {
    pub n_muls: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: flock_core::pcs::PcsParams,
}

impl Mul64Setup {
    pub fn new(n_muls: usize) -> Self {
        Self::with_profile(n_muls, flock_core::pcs::ligerito::LigeritoProfile::Fast)
    }

    pub fn with_profile(
        n_muls: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        let n_log = min_n_blocks_log(n_muls);
        let r1cs = build_block_r1cs(n_log);
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let pcs_params = flock_core::pcs::PcsParams {
            m: r1cs.m,
            log_inv_rate: profile.log_inv_rate(),
            log_batch_size: 6,
            profile,
            merkle_hash: Default::default(),
        };
        Self {
            n_muls,
            r1cs,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    /// Product capacity of the batch (15 per block slot).
    pub fn capacity(&self) -> usize {
        MULS_PER_BLOCK << self.n_blocks_log()
    }

    fn generate_witness_ab(
        &self,
        pairs: &[(u64, u64)],
    ) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        generate_witness_with_ab_packed_and_lincheck(pairs, self.n_blocks_log())
    }

    /// Packed witness for the generic matrix-driven prover.
    pub fn generate_witness_packed(&self, pairs: &[(u64, u64)]) -> Vec<F128> {
        let (z, _a, _b, _stripe) = self.generate_witness_ab(pairs);
        z
    }

    /// Generic (matrix-driven) prover; byte-identical proofs to
    /// [`Self::prove_fast`].
    pub fn prove_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        pairs: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        let z = self.generate_witness_packed(pairs);
        crate::prover::prove_ligerito(&self.r1cs, z, &self.pcs_params, challenger)
    }

    /// Fast prover: fused (z, a, b, z_lincheck) emission, CSC lincheck.
    pub fn prove_fast<Ch: flock_core::challenger::Challenger>(
        &self,
        pairs: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        assert_eq!(pairs.len(), self.n_muls);
        let (codeword, (z, a, b, stripe)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                self.generate_witness_ab(pairs)
            });
        crate::prover::prove_fast_ligerito_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z,
            a,
            b,
            stripe,
            self.r1cs.csc_lincheck_circuit(),
            codeword,
            challenger,
        )
    }

    /// [`Self::prove_fast`] with the per-phase timing breakdown.
    pub fn prove_fast_timed<Ch: flock_core::challenger::Challenger>(
        &self,
        pairs: &[(u64, u64)],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(pairs.len(), self.n_muls);
        let t0 = std::time::Instant::now();
        let (z, a, b, stripe) = self.generate_witness_ab(pairs);
        let witness_s = t0.elapsed().as_secs_f64();
        let (proof, commitment, claim, mut timings) = crate::prover::prove_fast_ligerito_timed(
            &self.r1cs,
            &self.pcs_params,
            z,
            a,
            b,
            stripe,
            self.r1cs.csc_lincheck_circuit(),
            None,
            challenger,
        );
        timings.witness_s = witness_s;
        (proof, commitment, claim, timings)
    }

    pub fn verify<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::VerifyError> {
        flock_core::verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    #[test]
    fn circuit_stats() {
        let c = circuit();
        println!(
            "mul64 base circuit: adders = {}, sub_bits = {SUB_BITS}, nodes = {}, nnz = {} \
             | block: {MULS_PER_BLOCK} products, useful = {USEFUL_BITS} / {K} ({:.1}%)",
            c.n_adders,
            c.nodes.len(),
            c.nnz,
            100.0 * USEFUL_BITS as f64 / K as f64,
        );
        assert_eq!(c.slot_nodes.len(), SUB_BITS);
        assert_eq!(c.row_ab_nodes.len(), SUB_BITS);
        assert_eq!(c.row_supports.len(), SUB_BITS);
    }

    #[test]
    fn product_bits_match_native() {
        let circ = circuit();
        let mut rng = Rng(0xDEADBEEF);
        let mut p_lanes = [0u64; 64];
        let mut q_lanes = [0u64; 64];
        let pairs: Vec<(u64, u64)> = (0..64)
            .map(|i| match i {
                0 => (0, 0),
                1 => (u64::MAX, u64::MAX),
                2 => (u64::MAX, 1),
                3 => (1 << 63, 2),
                _ => (rng.next_u64(), rng.next_u64()),
            })
            .collect();
        for (l, &(p, q)) in pairs.iter().enumerate() {
            p_lanes[l] = p;
            q_lanes[l] = q;
        }
        transpose_64x64(&mut p_lanes);
        transpose_64x64(&mut q_lanes);
        let mut vals = vec![0u64; circ.nodes.len()];
        eval_lanes(circ, &p_lanes, &q_lanes, &mut vals);
        for (l, &(p, q)) in pairs.iter().enumerate() {
            let want = (p as u128) * (q as u128);
            for k in 0..128 {
                let got = (vals[circ.product_bit_nodes[k] as usize] >> l) & 1;
                assert_eq!(
                    got,
                    ((want >> k) & 1) as u64,
                    "product bit {k} of {p:#x} * {q:#x}"
                );
            }
        }
    }

    /// The fused (z, a, b) emission must agree with the matrices: a = A·z,
    /// b = B·z, and the R1CS must be satisfied. Small non-Ligerito size
    /// (n_log = 3 → m = 20), partial last block included.
    #[test]
    fn fused_ab_matches_matrices() {
        let n_log = 3;
        let r1cs = build_block_r1cs(n_log);
        let mut rng = Rng(0xC0FFEE);
        let n_muls = 5 * MULS_PER_BLOCK + 7; // 82 of 120 slots
        let pairs: Vec<(u64, u64)> = (0..n_muls)
            .map(|_| (rng.next_u64(), rng.next_u64()))
            .collect();
        let (z, a, b, _stripe) = generate_witness_with_ab_packed_and_lincheck(&pairs, n_log);
        assert!(r1cs.satisfies_packed(&z), "R1CS unsatisfied");
        let a_ref = r1cs.apply_a_packed(&z);
        let b_ref = r1cs.apply_b_packed(&z);
        assert_eq!(a, a_ref, "fused A-side disagrees with A₀·z");
        assert_eq!(b, b_ref, "fused B-side disagrees with B₀·z");
    }

    /// End-to-end prove + verify at the smallest Ligerito size (m = 22).
    #[test]
    fn prove_verify_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let n_muls = MULS_PER_BLOCK << MIN_N_BLOCKS_LOG; // 480
        let setup = Mul64Setup::new(n_muls);
        assert_eq!(setup.m(), 22);
        let mut rng = Rng(0x5EED);
        let pairs: Vec<(u64, u64)> = (0..n_muls)
            .map(|_| (rng.next_u64(), rng.next_u64()))
            .collect();
        let mut ch_p = FsChallenger::new(b"mul64-test-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&pairs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"mul64-test-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .expect("verification failed");
        assert_eq!(claim_p, claim_v);
    }
}
