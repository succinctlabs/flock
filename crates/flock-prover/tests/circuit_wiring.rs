//! Circuit proofs end to end: one proof attesting per-gate relations AND the
//! circuit's wiring equalities AND the public IO.
//!
//! `docs/circuit-wiring-design.tex` §5–§8 is the spec; `flock_core::circuit`
//! carries the mechanism. The driving workloads here are the ones the wiring
//! milestone is for:
//!
//! 1. **SHA-256 binary tree** (the primary): `2^k` public leaf messages → root,
//!    every gate's `h` wired to the public IV cell (one LONG fan-out cycle),
//!    each internal node's message wired from its children's outputs (fan-in),
//!    the root's output to public output cells — checked against a native
//!    SHA-256 tree computation.
//! 2. **Element chain**: a product chain through the large-field class, public
//!    in and out.
//! 3. **Cross-class**: a hash gate's output words feed an element `mult` gate —
//!    the recursion plan's class boundary in miniature. The crossing is
//!    ordinary wiring.
//! 4. The **tamper matrix**: a broken wiring equality on a per-gate-satisfying
//!    witness, wrong public words, wrong gate counts, swapped wire
//!    connections, bit-flipped transcript and gather values — plus the three
//!    tests that pin the binding obligations one layer at a time:
//!    `g_side_forgery_is_rejected` (the `g = w∘σ` attack that only
//!    `f_eval == g_eval` stops), `fabricated_witness_fails_recombination`
//!    (jointly consistent evals over the wrong vector), and
//!    `gather_claims_are_bound_by_the_opening` (a self-consistent wiring proof
//!    of a fabricated witness, rejected by the PCS opening).
//! 5. A **differential oracle**: a brute-force wire-class checker over the
//!    committed words, agreeing with accept/reject on randomized circuits with
//!    randomly generated wirings.
//!
//! Geometry: SHA-256 (κ = 15) at ν = 7 gives `M = 22`, the smallest embedded
//! Ligerito config (`union::MIN_DENSE_M`), with 128 gate rows. The element-only
//! registry uses ν = 12, κ = 3 → `M = 22` likewise.
//!
//! Run with `cargo test --release -p flock-prover --test circuit_wiring --
//! --ignored`. A DEBUG run needs `--test-threads=1` (the repo's known
//! pre-existing rayon stack hazard in the Ligerito recursion).

use flock_core::circuit::{Cell, Circuit, CircuitError, WiringError};
use flock_core::element_r1cs::{ElementTableBuilder, ElementTableType};
use flock_core::field::F128;
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_core::product_gkr;
use flock_core::proof::R1csProofCircuitMerged;
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionElementSlotInput, UnionSlotProverInput};
use flock_prover::r1cs_hashes::sha2;
use flock_prover::schedule::{IoWord, Registry, TableType};
use flock_prover::union::UnionInstance;
use flock_prover::verifier::{self, FlockVerifyError};
use std::sync::Arc;

const DOMAIN: &[u8] = b"flock-circuit-wiring-v0";

use flock_core::test_rng::Rng;

fn union_pcs_params(union: &UnionInstance<'_>) -> PcsParams {
    PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: flock_core::pcs::ligerito::embedded_initial_k_or_default(
            union.dense_m(),
            LigeritoProfile::Fast,
        ),
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(flock_core::pcs::ligerito::embedded_initial_k_or_default(
            union.dense_m(),
            LigeritoProfile::Fast,
        )),
        merkle_hash: Default::default(),
    }
}

/// Pack 32-bit words into the 128-bit committed words the boolean trace holds:
/// word `i` of the result is `u32s[4i..4i+4]`, LSB-first inside the word — the
/// BatchMajor bit layout (`sha2::h_bit(w, b) = 32w + b`).
fn pack_u32_words(u32s: &[u32]) -> Vec<F128> {
    assert!(u32s.len().is_multiple_of(4));
    u32s.chunks(4)
        .map(|c| {
            F128::new(
                c[0] as u64 | ((c[1] as u64) << 32),
                c[2] as u64 | ((c[3] as u64) << 32),
            )
        })
        .collect()
}

// ===========================================================================
// 1. The SHA-256 binary tree
// ===========================================================================

/// SHA-256's chaining-gate IO schema, in 128-bit words of the row:
/// `h0,h1` (the input chaining value, wireable so multi-compression chains
/// work), `m0..m3` (the 512-bit message), `o0,o1` (the output chaining value).
/// Word `w` covers block bits `[128w, 128w+128)`, so `H_in` is words 0–1,
/// `H_out` words 2–3 (`H_OUT_BASE = 256`) and `M` words 4–7 (`M_BASE = 512`).
fn sha2_schema() -> Vec<IoWord> {
    vec![
        IoWord::input(0),
        IoWord::input(1),
        IoWord::input(4),
        IoWord::input(5),
        IoWord::input(6),
        IoWord::input(7),
        IoWord::output(2),
        IoWord::output(3),
    ]
}

// Cell-slot indices into the schema above (the enumeration order IS the
// schema order).
const SHA_H0: usize = 0;
const SHA_H1: usize = 1;
const SHA_M0: usize = 2;
const SHA_O0: usize = 6;
const SHA_O1: usize = 7;

/// A complete binary tree of SHA-256 compressions over `1 << k_leaves` public
/// leaf messages: gate `i < L` hashes leaf message `i` under the IV; internal
/// gates hash the concatenation of their two children's 256-bit outputs.
///
/// Gate rows: leaves `0..L`, then each level in order, root last. The message
/// of an internal gate is exactly `[left.o0, left.o1, right.o0, right.o1]` —
/// the 128-bit word granularity lines up with the tree, which is why the
/// wiring is pure copy constraints and no packing gate is needed.
struct Tree {
    /// `(h_in, m)` per gate row, in row order.
    compressions: Vec<([u32; 8], [u32; 16])>,
    /// Public words: `iv0, iv1`, then each leaf's 4 message words, then the
    /// root's 2 output words.
    public: Vec<F128>,
    wires: Vec<Vec<Cell>>,
    n_gates: usize,
    /// The native root, for the differential against the circuit's public out.
    root: [u32; 8],
}

fn build_tree(k_leaves: usize, nu: usize, rng: &mut Rng) -> Tree {
    let leaves = 1usize << k_leaves;
    let n_gates = 2 * leaves - 1;
    let mut compressions: Vec<([u32; 8], [u32; 16])> = Vec::with_capacity(n_gates);
    let mut out: Vec<[u32; 8]> = Vec::with_capacity(n_gates);
    let mut public: Vec<F128> = pack_u32_words(&sha2::SHA256_IV); // iv0, iv1
    let iv_cells = 2;

    // Leaves: public messages under the IV.
    for _ in 0..leaves {
        let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
        public.extend(pack_u32_words(&m));
        out.push(sha2::sha256_compress(&sha2::SHA256_IV, &m));
        compressions.push((sha2::SHA256_IV, m));
    }
    // Internal levels: children (2i, 2i+1) of the previous level.
    let mut level_start = 0usize;
    let mut level_len = leaves;
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(leaves - 1);
    while level_len > 1 {
        for i in 0..level_len / 2 {
            let (l, r) = (level_start + 2 * i, level_start + 2 * i + 1);
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&out[l]);
            m[8..].copy_from_slice(&out[r]);
            out.push(sha2::sha256_compress(&sha2::SHA256_IV, &m));
            compressions.push((sha2::SHA256_IV, m));
            children.push((l, r));
        }
        level_start += level_len;
        level_len /= 2;
    }
    assert_eq!(compressions.len(), n_gates);
    let root = out[n_gates - 1];
    public.extend(pack_u32_words(&root));

    // ---- Wiring.
    let mut wires: Vec<Vec<Cell>> = Vec::new();
    // The IV: two long cycles, one per IV word, each holding the public cell
    // and EVERY gate's corresponding h word (fan-out over the whole tree).
    for (w, slot) in [(0usize, SHA_H0), (1, SHA_H1)] {
        let mut class = vec![pub_cell(w, nu)];
        class.extend((0..n_gates).map(|g| Cell::new(slot, g)));
        wires.push(class);
    }
    // Leaf messages from public cells: leaf `i`'s word `w` ← public word
    // `iv_cells + 4i + w`.
    for i in 0..leaves {
        for w in 0..4 {
            wires.push(vec![
                pub_cell(iv_cells + 4 * i + w, nu),
                Cell::new(SHA_M0 + w, i),
            ]);
        }
    }
    // Internal messages from the children's outputs (fan-in).
    for (p, &(l, r)) in children.iter().enumerate() {
        let gate = leaves + p;
        wires.push(vec![Cell::new(SHA_O0, l), Cell::new(SHA_M0, gate)]);
        wires.push(vec![Cell::new(SHA_O1, l), Cell::new(SHA_M0 + 1, gate)]);
        wires.push(vec![Cell::new(SHA_O0, r), Cell::new(SHA_M0 + 2, gate)]);
        wires.push(vec![Cell::new(SHA_O1, r), Cell::new(SHA_M0 + 3, gate)]);
    }
    // The root's output to the public output cells.
    let out_base = iv_cells + 4 * leaves;
    wires.push(vec![Cell::new(SHA_O0, n_gates - 1), pub_cell(out_base, nu)]);
    wires.push(vec![
        Cell::new(SHA_O1, n_gates - 1),
        pub_cell(out_base + 1, nu),
    ]);

    Tree {
        compressions,
        public,
        wires,
        n_gates,
        root,
    }
}

/// The tree registry's public cell-slots start right after the 8 SHA-256 gate
/// slots; public word `p` is `(PUB_SLOT + (p >> ν), p mod 2^ν)`, so a segment
/// wider than the row capacity simply spills into the next slot.
const PUB_SLOT: usize = 8;

fn pub_cell(p: usize, nu: usize) -> Cell {
    Cell::new(PUB_SLOT + (p >> nu), p & ((1 << nu) - 1))
}

fn sha2_registry(nu: usize) -> (Registry, flock_core::r1cs::BlockR1cs) {
    let r1cs = sha2::build_block_r1cs(nu);
    let registry = Registry::new(
        vec![TableType::from_block_r1cs(&r1cs).with_io_schema(sha2_schema())],
        nu,
    );
    (registry, r1cs)
}

/// **THE PRIMARY DRIVING WORKLOAD.** A SHA-256 binary tree as one circuit
/// proof: gate relations + wiring + public IO, verified against a native tree
/// computation. Plus the tamper matrix.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn sha256_binary_tree_circuit() {
    let (nu, k_leaves) = (7usize, 3usize); // 8 leaves → 15 gates, 128 rows
    let (registry, r1cs) = sha2_registry(nu);
    assert_eq!(registry.m_total(), 22);
    let mut rng = Rng::new(0x5A20_0001);
    let tree = build_tree(k_leaves, nu, &mut rng);

    let union = UnionInstance::new(&registry, vec![tree.n_gates]);
    let pcs_params = union_pcs_params(&union);
    let circuit = Circuit::new(
        &registry,
        vec![tree.n_gates],
        tree.public.len(),
        tree.wires.clone(),
    )
    .expect("the tree circuit is valid");
    assert_eq!(circuit.cells().num_gate_slots(), 8);
    assert_eq!(circuit.cells().num_public_slots(), 1);
    assert_eq!(circuit.cells().mu(), nu + 4);
    // The IV class is one cycle over every gate plus the public cell.
    assert!(circuit.wires().iter().any(|c| c.len() == tree.n_gates + 1));

    let circuit_lc = r1cs.csc_lincheck_circuit();
    let prove = |compressions: &[([u32; 8], [u32; 16])], public: &[F128]| {
        let mut ch = FsChallenger::new(DOMAIN);
        prover::prove_fast_ligerito_union_circuit(
            &union,
            &circuit,
            public,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(compressions, nu),
                circuit_lc,
            )],
            Vec::new(),
            &mut ch,
        )
    };
    let verify = |public: &[F128],
                  commitment: &flock_prover::pcs::Commitment,
                  proof: &R1csProofCircuitMerged| {
        let mut ch = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &circuit,
            public,
            &[circuit_lc],
            &commitment.clone(),
            proof,
            &pcs_params,
            &mut ch,
        )
    };

    let (proof, commitment, _) = prove(&tree.compressions, &tree.public);
    verify(&tree.public, &commitment, &proof).expect("honest tree circuit verifies");

    // The proof's public output words ARE the native root.
    let out_base = 2 + 4 * (1 << k_leaves);
    assert_eq!(
        &tree.public[out_base..out_base + 2],
        &pack_u32_words(&tree.root)[..],
        "public output must be the native SHA-256 tree root"
    );

    // ---- Tamper matrix -----------------------------------------------------

    // (a) A satisfying-per-gate witness with ONE copied value changed: rewire
    //     the tree so a leaf's output no longer reaches its parent — i.e. hash
    //     a DIFFERENT message at one internal node. Every gate still satisfies
    //     the SHA-256 relation; only the wiring equality breaks.
    {
        let mut bad = tree.compressions.clone();
        let leaves = 1usize << k_leaves;
        bad[leaves].1[0] ^= 1; // parent of leaves 0,1: one message word off
        let (p, cm, _) = prove(&bad, &tree.public);
        assert!(
            matches!(
                verify(&tree.public, &cm, &p),
                Err(FlockVerifyError::Wiring(WiringError::Gkr(
                    product_gkr::ProductGkrError::ProductMismatch
                )))
            ),
            "a broken wiring equality must be rejected by the product identity"
        );
    }

    // (b) A wrong PUBLIC word — the statement changes, so the transcript does.
    for i in [0usize, 3, out_base] {
        let mut bad = tree.public.clone();
        bad[i] += F128::ONE;
        assert!(
            verify(&bad, &commitment, &proof).is_err(),
            "public word {i} must be bound"
        );
        // …and a prover honestly proving the tampered statement is rejected
        // too: the wiring equality it asserts no longer holds.
        let (p, cm, _) = prove(&tree.compressions, &bad);
        assert!(
            verify(&bad, &cm, &p).is_err(),
            "public word {i} must be enforced by the wiring"
        );
    }

    // (c) A wrong gate count: the circuit and the union disagree, and the
    //     count binds in the statement, the heights and the digest.
    {
        let bad_union = UnionInstance::new(&registry, vec![tree.n_gates - 1]);
        let bad_params = union_pcs_params(&bad_union);
        let mut ch = FsChallenger::new(DOMAIN);
        assert_eq!(
            verifier::verify_ligerito_union_circuit(
                &bad_union,
                &circuit,
                &tree.public,
                &[circuit_lc],
                &commitment,
                &proof,
                &bad_params,
                &mut ch,
            ),
            Err(FlockVerifyError::CircuitMismatch)
        );
        // A circuit rebuilt at the wrong count is a different statement.
        let bad_circuit = Circuit::new(
            &registry,
            vec![tree.n_gates - 1],
            tree.public.len(),
            tree.wires
                .iter()
                .map(|c| {
                    c.iter()
                        .filter(|cell| cell.slot >= PUB_SLOT || cell.row < tree.n_gates - 1)
                        .copied()
                        .collect()
                })
                .collect(),
        )
        .expect("still a valid circuit");
        let mut ch = FsChallenger::new(DOMAIN);
        assert!(
            verifier::verify_ligerito_union_circuit(
                &bad_union,
                &bad_circuit,
                &tree.public,
                &[circuit_lc],
                &commitment,
                &proof,
                &bad_params,
                &mut ch,
            )
            .is_err()
        );
    }

    // (d) SWAPPED WIRES: exchange two wire connections. Same cells, same
    //     multiset of classes sizes — only σ moves, and the verifier rebuilds
    //     σ from its own circuit, so the input check fails.
    {
        let leaves = 1usize << k_leaves;
        let mut swapped = tree.wires.clone();
        // Leaf 0's o0 goes to its parent's m0 and leaf 0's o1 to m1; swap
        // those two connections.
        let i0 = swapped
            .iter()
            .position(|c| {
                c.contains(&Cell::new(SHA_O0, 0)) && c.contains(&Cell::new(SHA_M0, leaves))
            })
            .expect("leaf 0 → parent m0");
        let i1 = swapped
            .iter()
            .position(|c| {
                c.contains(&Cell::new(SHA_O1, 0)) && c.contains(&Cell::new(SHA_M0 + 1, leaves))
            })
            .expect("leaf 0 → parent m1");
        swapped[i0] = vec![Cell::new(SHA_O0, 0), Cell::new(SHA_M0 + 1, leaves)];
        swapped[i1] = vec![Cell::new(SHA_O1, 0), Cell::new(SHA_M0, leaves)];
        let swapped_circuit =
            Circuit::new(&registry, vec![tree.n_gates], tree.public.len(), swapped)
                .expect("valid circuit, different wiring");
        assert_ne!(circuit.digest(), swapped_circuit.digest());
        let mut ch = FsChallenger::new(DOMAIN);
        assert!(
            verifier::verify_ligerito_union_circuit(
                &union,
                &swapped_circuit,
                &tree.public,
                &[circuit_lc],
                &commitment,
                &proof,
                &pcs_params,
                &mut ch,
            )
            .is_err(),
            "a swapped wiring must be rejected"
        );
    }

    // (e) Tampered wiring transcript / gather values, and proof bytes.
    {
        let mut bad = proof.clone();
        bad.wiring.gkr.top_lhs += F128::ONE;
        assert!(verify(&tree.public, &commitment, &bad).is_err());
        let mut bad = proof.clone();
        bad.wiring.gkr.layers[2].vl0 += F128::ONE;
        assert!(verify(&tree.public, &commitment, &bad).is_err());
        let mut bad = proof.clone();
        bad.wiring.gather[0] += F128::ONE;
        assert_eq!(
            verify(&tree.public, &commitment, &bad),
            Err(FlockVerifyError::Wiring(WiringError::Recombination))
        );

        let bytes = bincode::serialize(&proof).expect("serialize");
        let decoded: R1csProofCircuitMerged = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(decoded, proof);
        assert!(verify(&tree.public, &commitment, &decoded).is_ok());
        let n_flips = 16usize;
        for i in 0..n_flips {
            let pos = i * (bytes.len() / n_flips);
            let mut b = bytes.clone();
            b[pos] ^= 1 << (i % 8);
            match bincode::deserialize::<R1csProofCircuitMerged>(&b) {
                Err(_) => {}
                Ok(p) => assert!(
                    verify(&tree.public, &commitment, &p).is_err(),
                    "bit flip at byte {pos} verified"
                ),
            }
        }
    }
}

/// **THE `g`-SIDE FORGERY.** The attack the `f_eval == g_eval` check exists to
/// stop, run for real.
///
/// The grand-product identity `∏(f_i + α·i + β) = ∏(g_i + α·s_σ(i) + β)` says
/// the multisets `{(f_i, i)}` and `{(g_i, σ(i))}` agree, which holds iff
/// `g_x = f_{σ(x)}` — for ANY `f`, not just a σ-invariant one. So a cheating
/// prover takes the true (wiring-VIOLATING) witness as `f` and runs the RHS
/// product on `g = f ∘ σ`: the two products match, every layer check passes,
/// and `f_eval` still recombines to the committed gather values. Only
/// `f_eval ≠ g_eval` gives it away.
///
/// The test asserts exactly that: the GKR itself accepts, the recombination
/// accepts, and the wiring layer rejects with [`WiringError::GEvalMismatch`].
#[test]
fn g_side_forgery_is_rejected() {
    use flock_core::zerocheck::univariate_skip::build_eq;

    // A small element-only registry keeps this to the wiring layer alone.
    let (nu, kappa) = (4usize, 2usize);
    let registry = element_registry(nu, kappa);
    let n = 3usize;
    let n_public = 2usize;
    let circuit = Circuit::new(
        &registry,
        vec![n],
        n_public,
        vec![
            vec![Cell::new(EL_C, 0), Cell::new(EL_B, 1)],
            vec![Cell::new(EL_C, 1), Cell::new(EL_B, 2)],
        ],
    )
    .expect("valid");
    let cells = circuit.cells();
    let sigma = circuit.sigma();
    let mut rng = Rng::new(0x6516_0001);

    // A committed buffer that VIOLATES the wiring (every cell independent) —
    // the statement the forger wants to pass.
    let mut packed = vec![F128::ZERO; 1usize << (registry.m_total() - 7)];
    for iota in 0..cells.num_gate_slots() {
        for row in 0..n {
            packed[cells.gate_word_addr(iota, row)] = rng.f128();
        }
    }
    let public: Vec<F128> = (0..n_public).map(|_| rng.f128()).collect();

    // `w` over the cell space, exactly as an honest prover would build it.
    let mu = cells.mu();
    let mut w = vec![F128::ZERO; 1 << mu];
    for (iota, slot) in cells.slots().iter().enumerate() {
        for row in 0..1usize << nu {
            w[(iota << nu) | row] = match *slot {
                flock_core::circuit::CellSlot::Gate { .. } => {
                    packed[cells.gate_word_addr(iota, row)]
                }
                flock_core::circuit::CellSlot::Public { s } => {
                    public.get((s << nu) + row).copied().unwrap_or(F128::ZERO)
                }
                flock_core::circuit::CellSlot::Pad => F128::ZERO,
            };
        }
    }
    // The forgery: g = w ∘ σ. Equal grand products, different vector.
    let g: Vec<F128> = (0..1 << mu).map(|x| w[sigma[x]]).collect();
    assert_ne!(g, w, "the witness must not already satisfy the wiring");

    let mut ch = FsChallenger::new(DOMAIN);
    let mask = circuit.live_mask();
    let (gkr, claim) = product_gkr::prove_batched(&w, &g, sigma, Some(&mask), &mut ch);
    assert_eq!(
        gkr.top_lhs, gkr.top_rhs,
        "the forged products DO match — the multiset identity holds for g = w∘σ"
    );
    assert_ne!(
        claim.f_eval, claim.g_eval,
        "…but the two vectors differ at ρ"
    );

    // Honest gather values for the honest `f = w` side, so the recombination
    // check passes and ONLY the g-side binding is left to catch it.
    let eq_row = build_eq(&claim.rho[..nu]);
    let gather: Vec<F128> = (0..cells.num_gate_slots())
        .map(|iota| {
            let base = cells.gate_word_addr(iota, 0);
            eq_row
                .iter()
                .enumerate()
                .map(|(j, &e)| e * packed[base + j])
                .fold(F128::ZERO, |a, b| a + b)
        })
        .collect();

    let proof = flock_core::circuit::WiringProof { gkr, gather };
    let mut ch = FsChallenger::new(DOMAIN);
    assert_eq!(
        flock_core::circuit::verify_wiring(&circuit, &public, &proof, &mut ch),
        Err(WiringError::GEvalMismatch),
        "the g-side check must be what stops this — the GKR and the \
         recombination both accept"
    );

    // The control: the SAME prover on a σ-invariant witness (the wiring
    // honestly satisfied) has g = w and verifies.
    for class in circuit.wires() {
        let v = rng.f128();
        for &idx in class {
            let (iota, row) = (idx >> nu, idx & ((1 << nu) - 1));
            if let flock_core::circuit::CellSlot::Gate { .. } = cells.slots()[iota] {
                packed[cells.gate_word_addr(iota, row)] = v;
            }
        }
    }
    // Public cells are not wired in this circuit, so the repair above is
    // complete.
    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, _) = flock_core::circuit::prove_wiring(&circuit, &packed, &public, &mut ch);
    let mut ch = FsChallenger::new(DOMAIN);
    flock_core::circuit::verify_wiring(&circuit, &public, &proof, &mut ch)
        .expect("the honest control must verify");
}

/// **A fabricated witness**: the GKR run honestly on a DIFFERENT (but
/// wiring-consistent) vector, so `f_eval == g_eval` holds jointly and every
/// layer check passes — and the recombination is what fails, because the
/// gather values are the ones the commitment carries.
///
/// This is the other half of the two-eval binding: `f_eval == g_eval` alone
/// says the two products ran on the same vector, not that the vector is the
/// committed one. Only the recombination ties it to the witness and the public
/// words.
#[test]
fn fabricated_witness_fails_recombination() {
    use flock_core::zerocheck::univariate_skip::build_eq;

    let (nu, kappa) = (4usize, 2usize);
    let registry = element_registry(nu, kappa);
    let n = 3usize;
    let n_public = 2usize;
    let circuit = Circuit::new(
        &registry,
        vec![n],
        n_public,
        vec![
            vec![Cell::new(EL_C, 0), Cell::new(EL_B, 1)],
            vec![Cell::new(EL_C, 1), Cell::new(EL_B, 2)],
        ],
    )
    .expect("valid");
    let cells = circuit.cells();
    let mut rng = Rng::new(0xFAB0_0001);
    let public: Vec<F128> = (0..n_public).map(|_| rng.f128()).collect();

    // Two committed buffers, both satisfying the wiring — the real one and the
    // one the forger runs the argument on.
    let mut buffers: Vec<Vec<F128>> = Vec::new();
    for _ in 0..2 {
        let mut packed = vec![F128::ZERO; 1usize << (registry.m_total() - 7)];
        for iota in 0..cells.num_gate_slots() {
            for row in 0..n {
                packed[cells.gate_word_addr(iota, row)] = rng.f128();
            }
        }
        for class in circuit.wires() {
            let v = rng.f128();
            for &idx in class {
                let (iota, row) = (idx >> nu, idx & ((1 << nu) - 1));
                packed[cells.gate_word_addr(iota, row)] = v;
            }
        }
        buffers.push(packed);
    }
    let (real, fake) = (&buffers[0], &buffers[1]);

    // The forger's transcript: an honest wiring proof of the FAKE buffer.
    let mut ch = FsChallenger::new(DOMAIN);
    let (fake_proof, _) = flock_core::circuit::prove_wiring(&circuit, fake, &public, &mut ch);

    // Recover ρ by replaying, then compute the REAL buffer's gather values —
    // the ones the PCS opening would accept — and splice them in.
    let mut ch = FsChallenger::new(DOMAIN);
    let rho = product_gkr::verify_batched_with_sigma(
        cells.mu(),
        &fake_proof.gkr,
        circuit.sigma(),
        Some(&circuit.live_mask()),
        &mut ch,
    )
    .expect("the fabricated GKR is internally honest")
    .rho;
    let eq_row = build_eq(&rho[..nu]);
    let real_gather: Vec<F128> = (0..cells.num_gate_slots())
        .map(|iota| {
            let base = cells.gate_word_addr(iota, 0);
            eq_row
                .iter()
                .enumerate()
                .map(|(j, &e)| e * real[base + j])
                .fold(F128::ZERO, |a, b| a + b)
        })
        .collect();
    let spliced = flock_core::circuit::WiringProof {
        gkr: fake_proof.gkr.clone(),
        gather: real_gather,
    };
    let mut ch = FsChallenger::new(DOMAIN);
    assert_eq!(
        flock_core::circuit::verify_wiring(&circuit, &public, &spliced, &mut ch),
        Err(WiringError::Recombination),
        "jointly consistent evals over a fabricated vector must fail the \
         recombination against the committed gather values"
    );
    // The control: the same GKR with its OWN gather values is internally
    // consistent (it is an honest proof — of the wrong witness), which is
    // precisely why the gather claims must ride the PCS opening.
    let mut ch = FsChallenger::new(DOMAIN);
    flock_core::circuit::verify_wiring(&circuit, &public, &fake_proof, &mut ch)
        .expect("the fabricated proof is self-consistent — the opening is what rejects it");
}

// ===========================================================================
// 2. The element chain, and 3. the cross-class circuit
// ===========================================================================

/// The element `mult` gate: columns `0,1` free wires in, column `2 = z0·z1`
/// out. Cell-slots follow the schema order below.
const EL_A: usize = 0;
const EL_B: usize = 1;
const EL_C: usize = 2;

fn mult_ty(kappa: usize) -> TableType {
    let mut b = ElementTableBuilder::new(kappa);
    b.free_wire(0).free_wire(1).mult(2, 0, 1);
    TableType::element(Arc::new(b.build().expect("mult block is valid"))).with_io_schema(vec![
        IoWord::input(0),
        IoWord::input(1),
        IoWord::output(2),
    ])
}

fn element_registry(nu: usize, kappa: usize) -> Registry {
    Registry::new(vec![mult_ty(kappa)], nu)
}

/// A satisfying `mult`-table witness for the chain `b_0 = seed`,
/// `c_i = a_i · b_i`, `b_{i+1} = c_i`, in the BatchMajor rows-low layout.
fn chain_witness(ty: &ElementTableType, nu: usize, a: &[F128], seed: F128) -> (Vec<F128>, F128) {
    let at = |c: usize, j: usize| (c << nu) + j;
    let mut z = vec![F128::ZERO; ty.width() << nu];
    let mut b = seed;
    for (j, &aj) in a.iter().enumerate() {
        z[at(0, j)] = aj;
        z[at(1, j)] = b;
        z[at(2, j)] = aj * b;
        b = aj * b;
    }
    assert!(
        ty.satisfies(&z, nu, a.len()),
        "generated witness must satisfy"
    );
    (z, b)
}

/// **The element-class driving workload**: a product chain, public seed and
/// multipliers in, public product out.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn element_chain_circuit() {
    let (nu, kappa, n) = (12usize, 3usize, 20usize);
    let registry = element_registry(nu, kappa);
    assert_eq!(registry.m_total(), 22);
    let ty = registry.types()[0].element_type().expect("element type");

    let mut rng = Rng::new(0xC4A1_0001);
    let seed = rng.f128();
    let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let (z, result) = chain_witness(ty, nu, &a, seed);

    // Public segment: seed, a_0..a_{n-1}, result.
    let mut public = vec![seed];
    public.extend_from_slice(&a);
    public.push(result);
    const PUB: usize = 3; // 3 gate slots, then the public slot

    let mut wires = vec![vec![Cell::new(PUB, 0), Cell::new(EL_B, 0)]];
    for i in 0..n {
        wires.push(vec![Cell::new(PUB, 1 + i), Cell::new(EL_A, i)]);
    }
    for i in 0..n - 1 {
        wires.push(vec![Cell::new(EL_C, i), Cell::new(EL_B, i + 1)]);
    }
    wires.push(vec![Cell::new(EL_C, n - 1), Cell::new(PUB, 1 + n)]);

    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = union_pcs_params(&union);
    let circuit = Circuit::new(&registry, vec![n], public.len(), wires).expect("valid");

    let prove = |z: &[F128], public: &[F128]| {
        let z = z.to_vec();
        let mut ch = FsChallenger::new(DOMAIN);
        prover::prove_fast_ligerito_union_circuit(
            &union,
            &circuit,
            public,
            &pcs_params,
            Vec::new(),
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z)
            })],
            &mut ch,
        )
    };
    let (proof, commitment, _) = prove(&z, &public);
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &[],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("honest element chain verifies");

    // A wrong claimed result: the last wire equality breaks.
    let mut bad_public = public.clone();
    bad_public[1 + n] += F128::ONE;
    let (p, cm, _) = prove(&z, &bad_public);
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verifier::verify_ligerito_union_circuit(
            &union,
            &circuit,
            &bad_public,
            &[],
            &cm,
            &p,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a wrong public result must be rejected"
    );
}

/// **The cross-class circuit**: a SHA-256 gate's two output words feed an
/// element `mult` gate, proving `t = o0 · o1` in `F128` for the hash output.
/// The class crossing is ordinary wiring — the recursion plan's boundary in
/// miniature.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn cross_class_hash_into_mult() {
    let (nu, kappa) = (7usize, 2usize);
    let r1cs = sha2::build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            mult_ty(kappa),
            TableType::from_block_r1cs(&r1cs).with_io_schema(sha2_schema()),
        ],
        nu,
    );
    assert_eq!(
        registry.num_boolean(),
        1,
        "SHA-256 sorts before the element"
    );
    assert!(registry.types()[1].is_element());
    assert_eq!(registry.m_total(), 23);

    let mut rng = Rng::new(0xC205_0001);
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let h_out = sha2::sha256_compress(&sha2::SHA256_IV, &m);
    let out_words = pack_u32_words(&h_out);
    let (o0, o1) = (out_words[0], out_words[1]);

    // Public: iv0, iv1, the 4 message words, and the product t.
    let mut public = pack_u32_words(&sha2::SHA256_IV);
    public.extend(pack_u32_words(&m));
    public.push(o0 * o1);

    // Cell-slots: 8 SHA-256 gate slots, then 3 element slots, then public.
    const EL: usize = 8;
    const PUB: usize = 11;
    let wires = vec![
        vec![Cell::new(PUB, 0), Cell::new(SHA_H0, 0)],
        vec![Cell::new(PUB, 1), Cell::new(SHA_H1, 0)],
        vec![Cell::new(PUB, 2), Cell::new(SHA_M0, 0)],
        vec![Cell::new(PUB, 3), Cell::new(SHA_M0 + 1, 0)],
        vec![Cell::new(PUB, 4), Cell::new(SHA_M0 + 2, 0)],
        vec![Cell::new(PUB, 5), Cell::new(SHA_M0 + 3, 0)],
        // The crossing: hash output words → element operands.
        vec![Cell::new(SHA_O0, 0), Cell::new(EL + EL_A, 0)],
        vec![Cell::new(SHA_O1, 0), Cell::new(EL + EL_B, 0)],
        vec![Cell::new(EL + EL_C, 0), Cell::new(PUB, 6)],
    ];

    let counts = vec![1usize, 1];
    let union = UnionInstance::new(&registry, counts.clone());
    let pcs_params = union_pcs_params(&union);
    let circuit = Circuit::new(&registry, counts, public.len(), wires).expect("valid");

    let el_ty = registry.types()[1].element_type().expect("element");
    let mut z = vec![F128::ZERO; el_ty.width() << nu];
    z[0 << nu] = o0;
    z[1 << nu] = o1;
    z[2 << nu] = o0 * o1;
    let circuit_lc = r1cs.csc_lincheck_circuit();

    let prove = |z: Vec<F128>, public: &[F128]| {
        let mut ch = FsChallenger::new(DOMAIN);
        prover::prove_fast_ligerito_union_circuit(
            &union,
            &circuit,
            public,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(&[(sha2::SHA256_IV, m)], nu),
                circuit_lc,
            )],
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z)
            })],
            &mut ch,
        )
    };
    let (proof, commitment, claims) = prove(z.clone(), &public);
    assert!(claims.boolean.is_some() && claims.element.is_some());
    let mut ch = FsChallenger::new(DOMAIN);
    verifier::verify_ligerito_union_circuit(
        &union,
        &circuit,
        &public,
        &[circuit_lc],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("honest cross-class circuit verifies");

    // Break the crossing: the element gate multiplies the RIGHT values but the
    // hash output is wired to different cells — i.e. feed the mult gate an
    // unrelated (but self-consistent) pair.
    let mut bad = z.clone();
    let (x, y) = (rng.f128(), rng.f128());
    bad[0 << nu] = x;
    bad[1 << nu] = y;
    bad[2 << nu] = x * y;
    let mut bad_public = public.clone();
    bad_public[6] = x * y;
    let (p, cm, _) = prove(bad, &bad_public);
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        matches!(
            verifier::verify_ligerito_union_circuit(
                &union,
                &circuit,
                &bad_public,
                &[circuit_lc],
                &cm,
                &p,
                &pcs_params,
                &mut ch,
            ),
            Err(FlockVerifyError::Wiring(_))
        ),
        "a self-consistent element gate on the WRONG operands must be rejected"
    );
}

/// **The gather claims are bound by the OPENING.** A wiring proof that is
/// internally honest — over a FABRICATED witness satisfying the same wiring —
/// spliced in at the right transcript position: the wiring layer has nothing to
/// object to (matching products, `f_eval == g_eval`, gather values that
/// recombine), and the PCS opening is what rejects it, because those values are
/// not the ones the commitment carries.
///
/// The circuit here wires only the chain's internal edges, leaving the seed
/// free — a circuit whose public segment pinned every input would leave nothing
/// to fabricate (which is itself the point of public IO).
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn gather_claims_are_bound_by_the_opening() {
    let (nu, kappa, n) = (12usize, 3usize, 6usize);
    let registry = element_registry(nu, kappa);
    let ty = registry.types()[0].element_type().expect("element type");
    let mut rng = Rng::new(0x09E4_0001);
    let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let (z, _) = chain_witness(ty, nu, &a, rng.f128());
    let (fake, _) = chain_witness(ty, nu, &a, rng.f128());
    assert_ne!(z, fake, "the fabricated witness must differ");

    let wires: Vec<Vec<Cell>> = (0..n - 1)
        .map(|i| vec![Cell::new(EL_C, i), Cell::new(EL_B, i + 1)])
        .collect();
    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = union_pcs_params(&union);
    let circuit = Circuit::new(&registry, vec![n], 0, wires).expect("valid");

    let z_gen = z.clone();
    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &circuit,
        &[],
        &pcs_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&z_gen)
        })],
        &mut ch,
    );
    let verify = |p: &R1csProofCircuitMerged| {
        let mut ch = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit(
            &union,
            &circuit,
            &[],
            &[],
            &commitment,
            p,
            &pcs_params,
            &mut ch,
        )
    };
    verify(&proof).expect("the honest proof verifies");

    // Replay the verifier's prefix to reach the wiring position — statement
    // binding, then the element region's PIOP — and prove the wiring there over
    // the fabricated witness.
    let mut ch = FsChallenger::new(DOMAIN);
    union.bind_statement_circuit(&mut ch, &commitment, &circuit.digest(), &[]);
    flock_core::element_r1cs::union::verify(
        &union,
        proof.element.as_ref().expect("element half"),
        &mut ch,
    )
    .expect("the honest element PIOP replays");
    let (fake_wiring, _) = flock_core::circuit::prove_wiring(&circuit, &fake, &[], &mut ch);

    let mut spliced = proof.clone();
    spliced.wiring = fake_wiring;
    let err = verify(&spliced).expect_err("a fabricated-witness wiring proof must be rejected");
    assert!(
        matches!(err, FlockVerifyError::PcsOpen(_)),
        "the OPENING must be what rejects it, not the wiring layer — got {err:?}"
    );
}

// ===========================================================================
// 4. The differential oracle
// ===========================================================================

/// Brute force: does every wire class hold ONE value across the committed
/// words and the public segment? This is the wiring statement, read directly
/// off the witness — the oracle the proof must agree with.
fn oracle_accepts(circuit: &Circuit, z: &[F128], nu: usize, public: &[F128]) -> bool {
    let cells = circuit.cells();
    // The element-only registry has ONE slot at offset 0, so its own witness
    // buffer IS the padded union buffer and the cell → word map applies to `z`
    // directly.
    let at = |slot: usize, row: usize| -> F128 {
        match cells.slots()[slot] {
            flock_core::circuit::CellSlot::Gate { .. } => z[cells.gate_word_addr(slot, row)],
            flock_core::circuit::CellSlot::Public { s } => public[(s << nu) + row],
            flock_core::circuit::CellSlot::Pad => F128::ZERO,
        }
    };
    circuit.wires().iter().all(|class| {
        let first = at(class[0] >> nu, class[0] & ((1 << nu) - 1));
        class
            .iter()
            .all(|&idx| at(idx >> nu, idx & ((1 << nu) - 1)) == first)
    })
}

/// **The differential oracle**: randomized circuits with randomly generated
/// wirings, proved and verified, against the brute-force checker. Accept and
/// reject cases both appear, and the two must agree on every one.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn randomized_wirings_agree_with_the_oracle() {
    let (nu, kappa, n) = (12usize, 3usize, 8usize);
    let registry = element_registry(nu, kappa);
    let ty = registry.types()[0].element_type().expect("element");
    let union = UnionInstance::new(&registry, vec![n]);
    let pcs_params = union_pcs_params(&union);
    const PUB: usize = 3;
    let n_public = 6usize;

    let mut rng = Rng::new(0x0AC1_E000);
    let (mut accepts, mut rejects, mut classes_seen) = (0usize, 0usize, 0usize);
    for case in 0..6usize {
        // ---- A random FORWARD wiring: producers only feed later rows, and a
        // class holds at most one producer, so validation passes by
        // construction and the space of wirings is still broad.
        // `classes[k] = (source cell, its class)`; the source is what the
        // repair pass below propagates from (a class's canonical FIRST cell is
        // its least index, which need not be the source).
        let mut used: Vec<Cell> = Vec::new();
        let mut classes: Vec<(Cell, Vec<Cell>)> = Vec::new();
        for row in 0..n {
            for slot in [EL_A, EL_B] {
                if rng.below(3) == 0 {
                    continue; // leave this input free
                }
                let cell = Cell::new(slot, row);
                if used.contains(&cell) {
                    continue;
                }
                // Wire it either to an earlier gate's output or to a public
                // cell.
                let src = if row > 0 && rng.below(2) == 0 {
                    Cell::new(EL_C, rng.below(row))
                } else {
                    Cell::new(PUB, rng.below(n_public))
                };
                match classes.iter_mut().find(|(s, _)| *s == src) {
                    Some((_, class)) => class.push(cell),
                    None => {
                        if used.contains(&src) {
                            continue; // that source already feeds another class
                        }
                        classes.push((src, vec![src, cell]));
                        used.push(src);
                    }
                }
                used.push(cell);
            }
        }
        let wires: Vec<Vec<Cell>> = classes.iter().map(|(_, c)| c.clone()).collect();
        let circuit = match Circuit::new(&registry, vec![n], n_public, wires) {
            Ok(c) => c,
            Err(CircuitError::EmptyClass) => continue,
            Err(e) => panic!("random forward wiring must validate, got {e:?}"),
        };

        // ---- A witness that satisfies the RELATION always, and the wiring in
        // half the cases.
        let repair = case % 2 == 0;
        let public: Vec<F128> = (0..n_public).map(|_| rng.f128()).collect();
        let at = |c: usize, j: usize| (c << nu) + j;
        let mut z = vec![F128::ZERO; ty.width() << nu];
        for row in 0..n {
            let (mut a, mut b) = (rng.f128(), rng.f128());
            if repair {
                // Pull each wired input from its class's SOURCE value. Sources
                // are public cells or strictly earlier rows, so one forward
                // pass suffices.
                for (slot, v) in [(EL_A, &mut a), (EL_B, &mut b)] {
                    let cell = Cell::new(slot, row);
                    if let Some((src, _)) = classes.iter().find(|(_, c)| c.contains(&cell)) {
                        *v = if src.slot == PUB {
                            public[src.row]
                        } else {
                            z[at(src.slot, src.row)]
                        };
                    }
                }
            }
            z[at(0, row)] = a;
            z[at(1, row)] = b;
            z[at(2, row)] = a * b;
        }
        assert!(ty.satisfies(&z, nu, n), "the relation always holds");

        let expected = oracle_accepts(&circuit, &z, nu, &public);
        let z_gen = z.clone();
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &circuit,
            &public,
            &pcs_params,
            Vec::new(),
            vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
                dst.copy_from_slice(&z_gen)
            })],
            &mut ch,
        );
        let mut ch = FsChallenger::new(DOMAIN);
        let got = verifier::verify_ligerito_union_circuit(
            &union,
            &circuit,
            &public,
            &[],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_ok();
        assert_eq!(
            got,
            expected,
            "case {case}: proof and brute-force oracle disagree \
             ({} classes, repaired: {repair})",
            circuit.wires().len()
        );
        classes_seen += circuit.wires().len();
        if got { accepts += 1 } else { rejects += 1 }
    }
    // The differential is only worth anything if both verdicts occurred over
    // non-trivial wirings.
    assert!(
        accepts > 0 && rejects > 0,
        "{accepts} accepts, {rejects} rejects"
    );
    assert!(
        classes_seen >= 12,
        "wirings were too sparse to test anything"
    );
}

// ===========================================================================
// 5. The smoke number
// ===========================================================================

/// Circuit proofs over the MERGED transport — the production shape (and,
/// since the jagged transport's removal, the only one).
///
/// The wiring argument's gather claims are packed-direct, so until the merged
/// transport grew an intake for those, circuit proofs were stuck on the
/// unmerged jagged path and its padded-domain auxiliaries. The jagged A/B
/// arm this test carried was run for the last time at the removal's B1 gate
/// (claims/root/wiring equality all held); what remains is the merged
/// tamper matrix over the gather claims.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn circuit_proofs_verify_over_the_merged_transport() {
    let (nu, k_leaves) = (7usize, 3usize);
    let (registry, r1cs) = sha2_registry(nu);
    let mut rng = Rng::new(0x_4D_47_C1_02);
    let tree = build_tree(k_leaves, nu, &mut rng);

    let union = UnionInstance::new(&registry, vec![tree.n_gates]);
    let pcs_params = union_pcs_params(&union);
    let circuit = Circuit::new(
        &registry,
        vec![tree.n_gates],
        tree.public.len(),
        tree.wires.clone(),
    )
    .expect("the tree circuit is valid");
    let circuit_lc = r1cs.csc_lincheck_circuit();
    let slot = || {
        UnionSlotProverInput::new(
            sha2::generate_witness_batch_major_partial(&tree.compressions, nu),
            circuit_lc,
        )
    };

    // Merged: the production path.
    let mut ch = FsChallenger::new(DOMAIN);
    let (merged, commitment, claims_m) = prover::prove_fast_ligerito_union_circuit(
        &union,
        &circuit,
        &tree.public,
        &pcs_params,
        vec![slot()],
        Vec::new(),
        &mut ch,
    );
    let mut ch_v = FsChallenger::new(DOMAIN);
    let got = verifier::verify_ligerito_union_circuit(
        &union,
        &circuit,
        &tree.public,
        &[circuit_lc],
        &commitment,
        &merged,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("merged rejected an honest circuit proof: {e:?}"));
    assert_eq!(got, claims_m);

    // Tampering must still be caught on the merged path — otherwise the
    // gather claims would be riding along unchecked.
    for (what, bad) in [
        ("opening", {
            let mut b = merged.clone();
            b.pcs_open.q_eval += F128::ONE;
            b
        }),
        ("gather value", {
            let mut b = merged.clone();
            b.wiring.gather[0] += F128::ONE;
            b
        }),
        ("boolean claim", {
            let mut b = merged.clone();
            b.boolean.as_mut().unwrap().lincheck.z_partial[0] += F128::ONE;
            b
        }),
    ] {
        let mut ch_v = FsChallenger::new(DOMAIN);
        assert!(
            verifier::verify_ligerito_union_circuit(
                &union,
                &circuit,
                &tree.public,
                &[circuit_lc],
                &commitment,
                &bad,
                &pcs_params,
                &mut ch_v,
            )
            .is_err(),
            "tampered {what} must be rejected by the merged transport"
        );
    }
}

/// A native merge node, end to end: verify two circuit proofs with their
/// matrix work DEFERRED, fold both children's claims into one accumulator,
/// discharge once.
///
/// This is the operation a recursion merge circuit performs, built natively
/// first so its Fiat–Shamir order, accumulator shape and discharge API are
/// settled before any of it is arithmetised. The deferred verify reads no
/// base matrix — that is what makes it circuit-shaped — and the O(nnz) work
/// happens only in the fold's prover and the final discharge, neither of
/// which a circuit would contain.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn a_merge_node_folds_two_circuit_proofs() {
    use flock_core::aggregate;

    let (nu, k_leaves) = (7usize, 3usize);
    let (registry, r1cs) = sha2_registry(nu);
    let circuit_lc = r1cs.csc_lincheck_circuit();

    // Two independent proofs of the SAME circuit — different witnesses, so
    // their claims land at unrelated points, which is what a merge sees.
    let mut proofs = Vec::new();
    let mut trees = Vec::new();
    for seed in [0x_4D_47_01u64, 0x_4D_47_02] {
        let mut rng = Rng::new(seed);
        let tree = build_tree(k_leaves, nu, &mut rng);
        let union = UnionInstance::new(&registry, vec![tree.n_gates]);
        let pcs_params = union_pcs_params(&union);
        let circuit = Circuit::new(
            &registry,
            vec![tree.n_gates],
            tree.public.len(),
            tree.wires.clone(),
        )
        .expect("valid circuit");
        let mut ch = FsChallenger::new(DOMAIN);
        let (proof, commitment, _) = prover::prove_fast_ligerito_union_circuit(
            &union,
            &circuit,
            &tree.public,
            &pcs_params,
            vec![UnionSlotProverInput::new(
                sha2::generate_witness_batch_major_partial(&tree.compressions, nu),
                circuit_lc,
            )],
            Vec::new(),
            &mut ch,
        );
        proofs.push((proof, commitment, pcs_params, circuit));
        trees.push(tree);
    }

    // The merge node's first job: verify each child succinctly, keeping its
    // matrix work AND its sigma claim (route B: the wiring's s_sigma(rho)
    // evaluation rides out as a foldable claim instead of the O(2^mu)
    // discharge).
    let mut assertions = Vec::new();
    let mut sigmas = Vec::new();
    let mut jaggeds = Vec::new();
    let mut jagged_params = None;
    for ((proof, commitment, pcs_params, circuit), tree) in proofs.iter().zip(&trees) {
        let union = UnionInstance::new(&registry, vec![tree.n_gates]);
        let mut ch = FsChallenger::new(DOMAIN);
        let (_, work, sigma) = verifier::verify_ligerito_union_circuit_deferred(
            &union,
            circuit,
            &tree.public,
            &[circuit_lc],
            commitment,
            proof,
            pcs_params,
            &mut ch,
        )
        .expect("deferred verify accepts an honest child");
        assert!(
            work.element.is_none(),
            "this circuit is boolean-only, so no element work"
        );
        assertions.push(work.boolean.expect("a boolean PIOP ran"));
        sigmas.push(sigma);
        // The layout's W-claims (the count win). The layout is a shape
        // constant — same circuit, same heights — so both children's claims
        // name ONE table, rebuilt here exactly as the opening verifier did.
        let params = flock_core::pcs::jagged::JaggedParams::from_heights(
            &union.jagged_heights(),
            union.n_log(),
            commitment.params.m - flock_core::pcs::LOG_PACKING,
        );
        assert!(
            work.jagged.check(&params),
            "the exported jagged claims discharge against the child's own layout"
        );
        jaggeds.push(work.jagged);
        jagged_params.get_or_insert(params);
    }
    let jagged_params = jagged_params.expect("two children ran");
    // Both children prove the SAME circuit, which is what makes their sigma
    // claims foldable (the accumulator is digest-keyed).
    assert_eq!(
        proofs[0].3.digest(),
        proofs[1].3.digest(),
        "the children share one circuit"
    );

    // Its second job: fold both children's claims into ONE accumulator —
    // matrix work and sigma together (sigma never travels alone).
    let mats = [(&r1cs.a_0, &r1cs.b_0)];
    let circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![circuit_lc];
    let circuit0 = &proofs[0].3;
    let digest = circuit0.digest();
    let jagged_p: Vec<aggregate::JaggedKeyProve<'_>> =
        vec![(digest, &jagged_params, jaggeds.iter().collect())];
    let jagged_v: Vec<aggregate::JaggedKeyVerify<'_>> = vec![(digest, jaggeds.iter().collect())];
    let mut chp = FsChallenger::new(b"merge");
    let (agg, acc) = aggregate::prove_aggregate_classes(
        &registry,
        &mats,
        &circs,
        &assertions,
        &[],
        &[],
        &[(circuit0, sigmas.iter().collect())],
        &jagged_p,
        &[],
        &mut chp,
    )
    .expect("the fold proves");
    let mut chv = FsChallenger::new(b"merge");
    let acc_v = aggregate::verify_aggregate_classes(
        &registry,
        &assertions,
        &[],
        &[(circuit0, sigmas.iter().collect())],
        &jagged_v,
        &[],
        &agg,
        &mut chv,
    )
    .expect("the fold verifies");
    assert_eq!(acc, acc_v, "prover and verifier accumulators must agree");

    // The accumulator summarises BOTH children's matrix obligations, and
    // discharging it once retroactively validates both linchecks — and the
    // folded sigma claim discharges against the circuit's own table: the
    // root's single O(2^mu) evaluation, once for the whole tree.
    assert!(acc.discharge(&mats), "the merged accumulator must be true");
    assert!(
        !acc.sigma.is_empty(),
        "the sigma group carries the folded claim"
    );
    assert!(
        acc.discharge_sigma(&[circuit0]),
        "the folded sigma claim discharges at the root"
    );
    // The jagged group: both children's W-claims folded to ONE claim on the
    // shared layout, discharged once at the root — the count win's native
    // chain, end to end.
    assert_eq!(
        acc.jagged.len(),
        1,
        "one folded jagged claim per child shape"
    );
    assert!(
        acc.discharge_jagged(&[(digest, &jagged_params)]),
        "the folded jagged claim discharges at the root"
    );
    let mut bad = acc.clone();
    bad.jagged[0].1.value += flock_core::field::F128::ONE;
    assert!(
        !bad.discharge_jagged(&[(digest, &jagged_params)]),
        "a tampered jagged fold fails the root discharge"
    );
    assert!(
        !acc.discharge_jagged(&[]),
        "a jagged entry with no table to discharge against must fail, not skip"
    );
    assert_eq!(acc.registry_digest, registry.digest());
    // A tampered folded sigma value must fail the root discharge.
    {
        let mut bad_acc = acc.clone();
        if let Some((_, claim)) = bad_acc.sigma.first_mut() {
            claim.value += F128::ONE;
        }
        assert!(
            !bad_acc.discharge_sigma(&[circuit0]),
            "a tampered sigma claim fails the root discharge"
        );
    }

    // A corrupted child must poison the accumulator — the deferred verify
    // cannot see it, so this is the only thing standing between a bad child
    // and an accepted merge.
    // A corrupted child must poison the accumulator — the deferred verify
    // cannot see it, so this is the only thing standing between a bad child
    // and an accepted merge.
    //
    // Note WHICH accumulator: the PROVER computes its output from the real
    // matrix, so its claim is honest no matter what the inputs said. It is
    // the VERIFIER's accumulator that is conditional on them, and the one a
    // merge node would carry forward.
    let mut bad = assertions.clone();
    bad[1].evals[0].0 += F128::ONE;
    let mut chp = FsChallenger::new(b"merge");
    let (bad_agg, _) = aggregate::prove_aggregate(&registry, &mats, &circs, &bad, &[], &mut chp)
        .expect("the prover will happily fold false claims");
    let mut chv = FsChallenger::new(b"merge");
    match aggregate::verify_aggregate(&registry, &bad, &[], &bad_agg, &mut chv) {
        Err(_) => {}
        Ok(a) => assert!(
            !a.discharge(&mats),
            "a corrupted child produced a true accumulator"
        ),
    }

    // The RECURSIVE merge shape: each child folded alone first (its own
    // leaf accumulator, sigma included), then a merge over the TWO priors —
    // the inherited claims fold 2 -> 1 per group with no fresh assertions,
    // and the sigma group rides along keyed by the shared circuit digest.
    let fold_one = |i: usize| -> aggregate::Accumulator {
        let mut chp = FsChallenger::new(b"child-leaf");
        let (agg1, _) = aggregate::prove_aggregate_classes(
            &registry,
            &mats,
            &circs,
            &assertions[i..i + 1],
            &[],
            &[],
            &[(circuit0, sigmas[i..i + 1].iter().collect())],
            &[],
            &[],
            &mut chp,
        )
        .expect("the single-child fold proves");
        let mut chv = FsChallenger::new(b"child-leaf");
        aggregate::verify_aggregate_classes(
            &registry,
            &assertions[i..i + 1],
            &[],
            &[(circuit0, sigmas[i..i + 1].iter().collect())],
            &[],
            &[],
            &agg1,
            &mut chv,
        )
        .expect("the single-child fold verifies")
    };
    let (acc_a, acc_b) = (fold_one(0), fold_one(1));
    let mut chp = FsChallenger::new(b"merge-two-priors");
    let (agg2, acc2_p) = aggregate::prove_aggregate_classes(
        &registry,
        &mats,
        &circs,
        &[],
        &[],
        &[],
        &[(circuit0, Vec::new())],
        &[],
        &[&acc_a, &acc_b],
        &mut chp,
    )
    .expect("the two-prior merge proves");
    let mut chv = FsChallenger::new(b"merge-two-priors");
    let acc2_v = aggregate::verify_aggregate_classes(
        &registry,
        &[],
        &[],
        &[(circuit0, Vec::new())],
        &[],
        &[&acc_a, &acc_b],
        &agg2,
        &mut chv,
    )
    .expect("the two-prior merge verifies");
    assert_eq!(
        acc2_p, acc2_v,
        "prover and verifier accumulators must agree"
    );
    assert!(
        acc2_v.discharge(&mats),
        "the inherited-only merge discharges"
    );
    assert!(
        acc2_v.discharge_sigma(&[circuit0]),
        "both children's sigma claims fold through the merge to one root discharge"
    );
}

/// AG-flavor twin of [`cross_class_hash_into_mult`]: the SAME mixed
/// boolean+element circuit (SHA-256 gate crossing into an element mult, one
/// public product) proven and verified through the **AG-skip** circuit
/// entries — the element class and wiring are flavor-independent, only the
/// boolean zerocheck's round 1 changes. Plus the AG-specific tamper arms:
/// the fused r₁ nonce and the fused lincheck skip nonce must reject.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn cross_class_circuit_ag_roundtrip() {
    let (nu, kappa) = (7usize, 2usize);
    let r1cs = sha2::build_block_r1cs(nu);
    let registry = Registry::new(
        vec![
            mult_ty(kappa),
            TableType::from_block_r1cs(&r1cs).with_io_schema(sha2_schema()),
        ],
        nu,
    );

    let mut rng = Rng::new(0xA6C2_0001);
    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
    let h_out = sha2::sha256_compress(&sha2::SHA256_IV, &m);
    let out_words = pack_u32_words(&h_out);
    let (o0, o1) = (out_words[0], out_words[1]);

    let mut public = pack_u32_words(&sha2::SHA256_IV);
    public.extend(pack_u32_words(&m));
    public.push(o0 * o1);

    const EL: usize = 8;
    const PUB: usize = 11;
    let wires = vec![
        vec![Cell::new(PUB, 0), Cell::new(SHA_H0, 0)],
        vec![Cell::new(PUB, 1), Cell::new(SHA_H1, 0)],
        vec![Cell::new(PUB, 2), Cell::new(SHA_M0, 0)],
        vec![Cell::new(PUB, 3), Cell::new(SHA_M0 + 1, 0)],
        vec![Cell::new(PUB, 4), Cell::new(SHA_M0 + 2, 0)],
        vec![Cell::new(PUB, 5), Cell::new(SHA_M0 + 3, 0)],
        vec![Cell::new(SHA_O0, 0), Cell::new(EL + EL_A, 0)],
        vec![Cell::new(SHA_O1, 0), Cell::new(EL + EL_B, 0)],
        vec![Cell::new(EL + EL_C, 0), Cell::new(PUB, 6)],
    ];

    let counts = vec![1usize, 1];
    let union = UnionInstance::new(&registry, counts.clone());
    let pcs_params = union_pcs_params(&union);
    let circuit = Circuit::new(&registry, counts, public.len(), wires).expect("valid");

    let el_ty = registry.types()[1].element_type().expect("element");
    let mut z = vec![F128::ZERO; el_ty.width() << nu];
    z[0 << nu] = o0;
    z[1 << nu] = o1;
    z[2 << nu] = o0 * o1;
    let circuit_lc = r1cs.csc_lincheck_circuit();

    let z_gen = z.clone();
    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, claims) = prover::prove_fast_ligerito_union_circuit_ag(
        &union,
        &circuit,
        &public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            sha2::generate_witness_batch_major_partial(&[(sha2::SHA256_IV, m)], nu),
            circuit_lc,
        )],
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&z_gen)
        })],
        &mut ch,
    );
    assert!(claims.boolean.is_some() && claims.element.is_some());

    let verify = |p: &flock_core::proof::R1csProofCircuitMergedAg| {
        let mut ch = FsChallenger::new(DOMAIN);
        verifier::verify_ligerito_union_circuit_ag(
            &union,
            &circuit,
            &public,
            &[circuit_lc],
            &commitment,
            p,
            &pcs_params,
            &mut ch,
        )
    };
    verify(&proof).expect("honest AG cross-class circuit verifies");

    // The DEFERRED twin accepts and returns dischargeable work.
    let mut ch = FsChallenger::new(DOMAIN);
    let (_, work, _sigma) = verifier::verify_ligerito_union_circuit_ag_deferred(
        &union,
        &circuit,
        &public,
        &[circuit_lc],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("deferred AG verify accepts");
    work.boolean
        .expect("boolean assertion present")
        .check(&union, &[circuit_lc])
        .expect("deferred boolean matrix work discharges");

    // AG round-1 message tamper.
    let mut bad = proof.clone();
    bad.boolean.as_mut().expect("boolean").ag.round1_ab[0] += F128::ONE;
    assert!(verify(&bad).is_err(), "tampered AG round-1 accepted");

    // Fused r₁ nonce tamper.
    let mut bad = proof.clone();
    let n = &mut bad.boolean.as_mut().expect("boolean").ag.r1_nonce;
    *n = n.wrapping_add(1);
    assert!(verify(&bad).is_err(), "tampered fused r1 nonce accepted");

    // Fused lincheck skip nonce tamper (the last lincheck nonce).
    let mut bad = proof.clone();
    *bad.boolean
        .as_mut()
        .expect("boolean")
        .lincheck
        .grinding_nonces
        .last_mut()
        .expect("AG arm carries a fused skip nonce") ^= 1;
    assert!(verify(&bad).is_err(), "tampered fused skip nonce accepted");
}
