//! The circuit verifying `n` leanXMSS\[BLAKE3\] signatures in one proof.
//!
//! Every `Th` call is BLAKE3 compressions ([`Blake3Gate`]): one block for a
//! chain step, a Merkle node and the top digest, two for the bottom digest,
//! fifteen chained for a one-time key's leaf. The wiring does not depend on
//! the signature: each step has a fixed number of hash slots, and a select
//! per step keeps the hash only when the codeword's digit says the step is
//! taken. A routing network ([`super::routing`]) first sorts a layer's chains
//! by digit, so the steps leaving positions 0 and 1 need only 22 and 34 slots
//! ([`STEP_SLOTS`]) instead of 58. That is 2·(22 + 34 + 58 + 15 + 16) + 3 =
//! 293 compressions per signature ([`COMPRESSIONS_PER_SIGNATURE`]) against
//! the 201 native verification spends.
//!
//! Per signer the statement is public: `P`, the root and the 32-byte message.
//! The leaf index, randomizers, chain values and Merkle siblings are private.

use flock_core::{
    challenger::FsChallenger,
    circuit::{
        boolean::BooleanGate,
        builder::{CircuitShape, CircuitWitness, FillPlan, ShapeBuilder, SlotId, Wire},
    },
    lincheck::LincheckCircuit,
    merkle::HashKind,
    pcs::{
        Commitment, PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    r1cs::BlockR1cs,
    union::UnionInstance,
    verifier::FlockVerifyError,
};
use flock_field::F128;

use super::{
    gadgets::{
        SPREAD_DIGITS, SPREADS, address_gate, address_step_gate, benes_gate, codeword_gate,
        layer_address_constants, lookup_gate, select_gate, spread_gate, swap_gate,
    },
    reference::{
        CHAIN_END, HEIGHT, PublicKey, Signature, TARGET_SUM, V, Value, XmssHash, codeword,
        layer_root, le128, msg_address, th,
    },
    routing::{BLOCKS, GROUPS, WIDTH, block_controls, position, route, sort_by_digit},
};
use crate::{
    prover::UnionSlotProverInput,
    r1cs_hashes::blake3::{BLAKE3_IV, Blake3Gate, build_block_r1cs, params_word},
};

/// Hash slots for the step leaving each position, after the chains are
/// sorted by digit ([`super::routing`]). A codeword's digits sum to
/// [`TARGET_SUM`], so at most `(3·58 − 106) / (3 − µ)` chains take step µ:
/// 22, 34, and all 58.
pub const STEP_SLOTS: [usize; CHAIN_END] = {
    let mut slots = [0; CHAIN_END];
    let mut step = 0;
    while step < CHAIN_END {
        let bound = (V * CHAIN_END - TARGET_SUM) / (CHAIN_END - step);
        slots[step] = if bound < V { bound } else { V };
        step += 1;
    }
    slots
};

/// BLAKE3 compressions per signature: per layer the step slots, the leaf (15)
/// and the path (16); then the digests (2 + 1).
pub const COMPRESSIONS_PER_SIGNATURE: usize =
    2 * (STEP_SLOTS[0] + STEP_SLOTS[1] + STEP_SLOTS[2] + 15 + HEIGHT) + 3;

const FIAT_SHAMIR_DOMAIN: &[u8] = b"flock-leanxmss-blake3-v1";

const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const ROOT: u32 = 8;

/// One signer's public statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Statement {
    pub pk: PublicKey,
    pub msg: [u8; 32],
}

/// Where one online input (or one public word) comes from.
#[derive(Clone, Copy, Debug)]
enum Source {
    Const(F128),
    Pp(usize),
    Root(usize),
    /// Half `h` (bytes `16h..16h+16`) of signer `s`'s message.
    Msg(usize, usize),
    Idx(usize),
    Rho(usize, usize),
    Chain(usize, usize, usize),
    Sibling(usize, usize, usize),
    /// Signer, layer, group, block: one routing control word.
    Control(usize, usize, usize, usize),
}

/// Each signature's routing control words, per layer, group and block: the
/// network sorts the layer's chains by digit.
type SignatureControls = [[[u16; BLOCKS]; GROUPS]; 2];

fn signature_controls(statement: &Statement, sig: &Signature) -> SignatureControls {
    let hash = XmssHash::Blake3;
    let pp = statement.pk.pp;
    let mut msg = statement.msg.to_vec();
    let mut out = [[[0; BLOCKS]; GROUPS]; 2];
    for (layer, controls) in out.iter_mut().enumerate() {
        let e = if layer == 0 {
            sig.idx
        } else {
            sig.idx >> HEIGHT
        };
        let ls = &sig.layers[layer];
        let digest = th(hash, &pp, &msg_address(layer, e), &[&msg, &ls.rho]);
        let digits = codeword(&digest).expect("the signature does not verify");
        let network = route(&sort_by_digit(&digits));
        *controls = std::array::from_fn(|g| block_controls(&network, g));
        if layer == 0 {
            msg = layer_root(hash, &pp, 0, e, &msg, ls, &mut None)
                .unwrap()
                .to_vec();
        }
    }
    out
}

/// `n` signers, each with its own `P`, message and leaf index, and a
/// key-free signature ([`super::reference::sign_keyless`]) of the message.
/// Deterministic in `seed`; signers are signed in parallel.
pub fn keyless_batch(n: usize, seed: u64) -> (Vec<Statement>, Vec<Signature>) {
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    (0..n)
        .into_par_iter()
        .map(|s| {
            let k = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(s as u64);
            let pp: Value = std::array::from_fn(|j| (k.rotate_left(8 * (j as u32 % 8)) >> j) as u8);
            let msg: [u8; 32] = std::array::from_fn(|j| (k >> ((j % 8) * 8)) as u8 ^ j as u8);
            let idx = (k >> 17) as u32;
            let (pk, sig) = super::reference::sign_keyless(
                super::reference::XmssHash::Blake3,
                &pp,
                idx,
                &msg,
                k,
                false,
            );
            (Statement { pk, msg }, sig)
        })
        .unzip()
}

/// A 16-byte value as a wire word: little-endian, four `u32`s per word,
/// exactly as BLAKE3 reads a message block.
pub fn word(v: &Value) -> F128 {
    let x = u128::from_le_bytes(*v);
    F128::new(x as u64, (x >> 64) as u64)
}

fn iv_words() -> [F128; 2] {
    let w = |i: usize| {
        F128::new(
            BLAKE3_IV[i] as u64 | (BLAKE3_IV[i + 1] as u64) << 32,
            BLAKE3_IV[i + 2] as u64 | (BLAKE3_IV[i + 3] as u64) << 32,
        )
    };
    [w(0), w(4)]
}

/// The `n`-signature circuit: its shape, the online-phase plan and how the
/// prover packs each slot.
pub struct XmssCircuit {
    n: usize,
    nu: usize,
    shape: CircuitShape,
    plan: FillPlan,
    blake3: Vec<SlotId>,
    gadgets: Vec<(SlotId, BooleanGate)>,
    blake3_r1cs: BlockR1cs,
    inputs: Vec<Source>,
    publics: Vec<Source>,
}

/// Records each declared input's source alongside the shape.
struct Builder {
    sb: ShapeBuilder,
    inputs: Vec<Source>,
    publics: Vec<Source>,
}

impl Builder {
    fn input(&mut self, source: Source) -> Wire {
        self.inputs.push(source);
        self.sb.input()
    }

    fn public(&mut self, source: Source) -> Wire {
        self.inputs.push(source);
        self.publics.push(source);
        self.sb.public_input()
    }

    fn constant(&mut self, v: F128) -> Wire {
        self.inputs.push(Source::Const(v));
        self.publics.push(Source::Const(v));
        self.sb.fixed_public_input(v)
    }
}

/// The constant wires every signature shares.
struct Constants {
    iv: [Wire; 2],
    zero: Wire,
    /// `LE128(i)` for each network position (chains, then dummies).
    chain_index: Vec<Wire>,
    /// Lookup classes: no bound, digit at least 1, digit at least 2.
    class: [Wire; 3],
    single: Wire,
    first: Wire,
    middle: Wire,
    last: Wire,
    last16: Wire,
    /// Per layer, the constant parts of its addresses
    /// ([`layer_address_constants`]).
    address: [Vec<Wire>; 2],
}

/// Positions within [`layer_address_constants`].
const MSG: usize = 0;
const fn chain_step(step: usize) -> usize {
    1 + step
}
const LEAF: usize = 4;
const fn node(level: usize) -> usize {
    4 + level
}

/// One gate kind, declared as as many slots of `2^nu` rows as its count
/// needs; gates fill the slots in order.
struct Kind {
    slots: Vec<SlotId>,
    capacity: usize,
    used: usize,
}

impl Kind {
    fn next(&mut self) -> SlotId {
        let slot = self.slots[self.used / self.capacity];
        self.used += 1;
        slot
    }
}

/// The declared gate kinds.
struct Slots {
    blake3: Kind,
    select: [Kind; CHAIN_END],
    swap: Kind,
    codeword: Kind,
    spread: Kind,
    address: Kind,
    address_step: Kind,
    benes: Kind,
    benes_reverse: Kind,
    lookup: Kind,
}

/// Gates of each kind per signature, in [`Slots`] order after BLAKE3.
const GADGETS_PER_SIGNATURE: [usize; CHAIN_END + 8] = [
    2 * STEP_SLOTS[0],
    2 * STEP_SLOTS[1],
    2 * STEP_SLOTS[2],
    2 * HEIGHT,
    2,
    2 * SPREADS,
    2 * 5,
    2 * HEIGHT,
    2 * GROUPS * BLOCKS,
    2 * GROUPS * BLOCKS,
    2 * WIDTH,
];

/// The `nu` whose layout has the smallest Boolean address space,
/// `⌈log2 Σ_kind ⌈count/2^nu⌉·2^(nu + k_log)⌉`, preferring the larger `nu`
/// (fewer slots) on a tie. Every slot has `2^nu` rows whatever its count, so
/// a large `nu` inflates the small gadget kinds, and a small `nu` declares a
/// heavy kind many times over.
pub fn plan_nu(kinds: &[(usize, usize)]) -> usize {
    let space = |nu: usize| -> u32 {
        let total: u128 = kinds
            .iter()
            .map(|&(count, k_log)| count.div_ceil(1 << nu).max(1) as u128 * (1u128 << (nu + k_log)))
            .sum();
        total.next_power_of_two().trailing_zeros()
    };
    (3..=24)
        .min_by_key(|&nu| (space(nu), std::cmp::Reverse(nu)))
        .unwrap()
}

/// The gadget kinds' constructors, in [`Slots`] order after BLAKE3.
fn gadget_constructors() -> [fn(usize) -> BooleanGate; CHAIN_END + 8] {
    [
        |nu| select_gate(0, nu).unwrap(),
        |nu| select_gate(1, nu).unwrap(),
        |nu| select_gate(2, nu).unwrap(),
        |nu| swap_gate(nu).unwrap(),
        |nu| codeword_gate(nu).unwrap(),
        |nu| spread_gate(nu).unwrap(),
        |nu| address_gate(nu).unwrap(),
        |nu| address_step_gate(nu).unwrap(),
        |nu| benes_gate(false, nu).unwrap(),
        |nu| benes_gate(true, nu).unwrap(),
        |nu| lookup_gate(nu).unwrap(),
    ]
}

/// Declares every gate kind: `counts[0]` BLAKE3 compressions, then the
/// gadget counts in [`Slots`] order, each over `⌈count/2^nu⌉` slots (at
/// least one). Returns the kinds and every gadget slot with its gate.
fn declare_slots(
    b: &mut Builder,
    nu: usize,
    counts: &[usize],
) -> (Slots, Vec<(SlotId, BooleanGate)>) {
    let slots_for = |count: usize| count.div_ceil(1 << nu).max(1);
    let blake3 = Kind {
        slots: (0..slots_for(counts[0]))
            .map(|_| b.sb.slot(Blake3Gate { nu }))
            .collect(),
        capacity: 1 << nu,
        used: 0,
    };
    let mut gadgets = Vec::new();
    let mut declared: Vec<Kind> = gadget_constructors()
        .iter()
        .zip(&counts[1..])
        .map(|(f, &count)| {
            let gate = f(nu);
            let slots = (0..slots_for(count))
                .map(|_| {
                    let id = b.sb.slot(gate.clone());
                    gadgets.push((id, gate.clone()));
                    id
                })
                .collect();
            Kind {
                slots,
                capacity: 1 << nu,
                used: 0,
            }
        })
        .collect();
    let mut take = || declared.remove(0);
    let slots = Slots {
        blake3,
        select: std::array::from_fn(|_| take()),
        swap: take(),
        codeword: take(),
        spread: take(),
        address: take(),
        address_step: take(),
        benes: take(),
        benes_reverse: take(),
        lookup: take(),
    };
    (slots, gadgets)
}

fn declare_constants(b: &mut Builder) -> Constants {
    Constants {
        iv: iv_words().map(|w| b.constant(w)),
        zero: b.constant(F128::ZERO),
        chain_index: (0..WIDTH).map(|i| b.constant(word(&le128(i)))).collect(),
        class: [0, 1, 2].map(|c| b.constant(F128::new(c, 0))),
        single: b.constant(params_word(0, 64, CHUNK_START | CHUNK_END | ROOT)),
        first: b.constant(params_word(0, 64, CHUNK_START)),
        middle: b.constant(params_word(0, 64, 0)),
        last: b.constant(params_word(0, 64, CHUNK_END | ROOT)),
        last16: b.constant(params_word(0, 16, CHUNK_END | ROOT)),
        address: [0, 1].map(|layer| {
            layer_address_constants(layer)
                .iter()
                .map(|c| b.constant(word(c)))
                .collect()
        }),
    }
}

impl XmssCircuit {
    /// The circuit for `n ≥ 1` signatures.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1);
        let gadget_fns = gadget_constructors();
        let mut kinds = vec![(
            n * COMPRESSIONS_PER_SIGNATURE,
            crate::r1cs_hashes::blake3::K_LOG,
        )];
        for (f, per_sig) in gadget_fns.iter().zip(GADGETS_PER_SIGNATURE) {
            kinds.push((n * per_sig, f(3).k_log()));
        }
        let nu = plan_nu(&kinds);
        let mut b = Builder {
            sb: ShapeBuilder::new(nu),
            inputs: Vec::new(),
            publics: Vec::new(),
        };
        let counts: Vec<usize> = kinds.iter().map(|&(count, _)| count).collect();
        let (mut slots, gadgets) = declare_slots(&mut b, nu, &counts);
        let constants = declare_constants(&mut b);
        for s in 0..n {
            let island = b.sb.begin_island();
            signature(&mut b, &mut slots, &constants, s);
            b.sb.end_island(island);
        }
        let shape = b.sb.finish().expect("the leanXMSS circuit is well formed");
        let plan = shape.fill_plan();
        XmssCircuit {
            n,
            nu,
            shape,
            plan,
            blake3: slots.blake3.slots,
            gadgets,
            blake3_r1cs: build_block_r1cs(nu),
            inputs: b.inputs,
            publics: b.publics,
        }
    }

    pub fn signatures(&self) -> usize {
        self.n
    }

    /// Each slot has `2^nu` rows.
    pub fn nu(&self) -> usize {
        self.nu
    }

    pub fn shape(&self) -> &CircuitShape {
        &self.shape
    }

    fn resolve(
        source: Source,
        statements: &[Statement],
        sigs: Option<(&[Signature], &[SignatureControls])>,
    ) -> F128 {
        let private = || sigs.expect("private inputs need the signatures");
        let sig = |s: usize| &private().0[s];
        match source {
            Source::Const(v) => v,
            Source::Pp(s) => word(&statements[s].pk.pp),
            Source::Root(s) => word(&statements[s].pk.root),
            Source::Msg(s, h) => word(&statements[s].msg[16 * h..16 * h + 16].try_into().unwrap()),
            Source::Idx(s) => F128::new(sig(s).idx as u64, 0),
            Source::Rho(s, l) => word(&sig(s).layers[l].rho),
            Source::Chain(s, l, i) => word(&sig(s).layers[l].y[i]),
            Source::Sibling(s, l, k) => word(&sig(s).layers[l].path[k]),
            Source::Control(s, l, g, block) => F128::new(private().1[s][l][g][block] as u64, 0),
        }
    }

    /// The public words, in publication order: the shared constants, then
    /// per signer `P`, the root and the message.
    pub fn public(&self, statements: &[Statement]) -> Vec<F128> {
        assert_eq!(statements.len(), self.n);
        self.publics
            .iter()
            .map(|&src| Self::resolve(src, statements, None))
            .collect()
    }

    /// The online phase: every gate evaluated, rows collected per slot. Each
    /// signature must verify natively; a gadget panics on one that does not.
    pub fn witness(&self, statements: &[Statement], sigs: &[Signature]) -> CircuitWitness {
        assert_eq!((statements.len(), sigs.len()), (self.n, self.n));
        use rayon::prelude::{IntoParallelIterator, ParallelIterator};
        let controls: Vec<SignatureControls> = (0..self.n)
            .into_par_iter()
            .map(|s| signature_controls(&statements[s], &sigs[s]))
            .collect();
        let values: Vec<F128> = self
            .inputs
            .iter()
            .map(|&src| Self::resolve(src, statements, Some((sigs, &controls))))
            .collect();
        self.shape.run_filled_deferred(&self.plan, &values, &[])
    }

    /// The union instance. Every used column is committed at full height
    /// (dummy rows as zeros), so the opening takes the block-first transport
    /// instead of materializing the jagged weight.
    fn union(&self) -> UnionInstance<'_> {
        let mut union = UnionInstance::new(&self.shape.registry, self.shape.counts.clone());
        union.set_full_height_commit();
        union
    }

    fn pcs_params(union: &UnionInstance<'_>) -> PcsParams {
        let log_batch_size = embedded_initial_k_or_default(union.dense_m(), LigeritoProfile::Fast);
        PcsParams {
            m: union.dense_m(),
            log_inv_rate: 1,
            log_batch_size,
            profile: LigeritoProfile::Fast,
            num_lanes: union.commit_lanes(log_batch_size),
            merkle_hash: HashKind::Blake3,
        }
    }

    /// Lincheck circuits in registry order.
    fn lincheck_circuits(&self) -> Vec<&dyn LincheckCircuit> {
        let mut by_slot: Vec<(usize, &dyn LincheckCircuit)> = self
            .gadgets
            .iter()
            .map(|(id, g)| {
                (
                    self.shape.registry_slot(*id),
                    g.r1cs().csc_lincheck_circuit() as _,
                )
            })
            .collect();
        for &id in &self.blake3 {
            by_slot.push((
                self.shape.registry_slot(id),
                self.blake3_r1cs.csc_lincheck_circuit() as _,
            ));
        }
        by_slot.sort_by_key(|&(slot, _)| slot);
        by_slot.into_iter().map(|(_, c)| c).collect()
    }

    /// Each slot's in-place witness driver, in registry order.
    fn slot_inputs<'a>(&'a self, witness: &'a CircuitWitness) -> Vec<UnionSlotProverInput<'a>> {
        let nu = self.nu;
        let mut by_slot: Vec<(usize, UnionSlotProverInput<'a>)> = Vec::new();
        for &id in &self.blake3 {
            let rows = witness.rows::<Blake3Gate>(id);
            by_slot.push((
                self.shape.registry_slot(id),
                UnionSlotProverInput::in_place(
                    move |dst| {
                        crate::r1cs_hashes::blake3::generate_witness_batch_major_partial_into(
                            rows, nu, dst,
                        )
                    },
                    self.blake3_r1cs.csc_lincheck_circuit(),
                ),
            ));
        }
        for (id, gate) in &self.gadgets {
            let rows = witness.rows::<BooleanGate>(*id);
            by_slot.push((
                self.shape.registry_slot(*id),
                UnionSlotProverInput::in_place(
                    move |dst| gate.pack_into(rows, dst),
                    gate.r1cs().csc_lincheck_circuit(),
                ),
            ));
        }
        by_slot.sort_by_key(|&(slot, _)| slot);
        by_slot.into_iter().map(|(_, input)| input).collect()
    }

    pub fn prove(&self, witness: &CircuitWitness) -> XmssProof {
        let union = self.union();
        let pcs = Self::pcs_params(&union);
        let mut challenger = FsChallenger::new(FIAT_SHAMIR_DOMAIN);
        let (proof, commitment, _claims) = prove_circuit(
            &union,
            &self.shape.circuit,
            &witness.public,
            &pcs,
            self.slot_inputs(witness),
            Vec::new(),
            &mut challenger,
        );
        XmssProof { proof, commitment }
    }

    pub fn verify(
        &self,
        statements: &[Statement],
        proof: &XmssProof,
    ) -> Result<(), FlockVerifyError> {
        let union = self.union();
        let pcs = Self::pcs_params(&union);
        let mut challenger = FsChallenger::new(FIAT_SHAMIR_DOMAIN);
        verify_circuit(
            &union,
            &self.shape.circuit,
            &self.public(statements),
            &self.lincheck_circuits(),
            &proof.commitment,
            &proof.proof,
            &pcs,
            &mut challenger,
        )
        .map(|_| ())
    }
}

#[cfg(target_arch = "aarch64")]
type CircuitProof = flock_core::proof::R1csProofCircuitMergedAg;
#[cfg(not(target_arch = "aarch64"))]
type CircuitProof = flock_core::proof::R1csProofCircuitMerged;

#[cfg(not(target_arch = "aarch64"))]
use {
    crate::prover::prove_fast_ligerito_union_circuit as prove_circuit,
    flock_core::verifier::verify_ligerito_union_circuit as verify_circuit,
};
#[cfg(target_arch = "aarch64")]
use {
    crate::prover::prove_fast_ligerito_union_circuit_ag as prove_circuit,
    flock_core::verifier::verify_ligerito_union_circuit_ag as verify_circuit,
};

/// A proof that `n` signatures verify (the AG zerocheck on aarch64, RS
/// elsewhere).
pub struct XmssProof {
    pub proof: CircuitProof,
    pub commitment: Commitment,
}

/// One compression's outputs: `[out_lo0, out_lo1, out_hi0, out_hi1]`.
fn compress(
    b: &mut Builder,
    slots: &mut Slots,
    cv: [Wire; 2],
    m: [Wire; 4],
    params: Wire,
) -> Vec<Wire> {
    let slot = slots.blake3.next();
    b.sb.gate(slot, &[cv[0], cv[1], m[0], m[1], m[2], m[3], params])
}

/// `Th` of a 64-byte input: one root compression from the IV; `H0` is the
/// first output word.
fn th64(b: &mut Builder, slots: &mut Slots, k: &Constants, m: [Wire; 4]) -> Wire {
    compress(b, slots, k.iv, m, k.single)[0]
}

/// One layer's chain ends: each chain advanced from its revealed value to
/// position 3. The chains (and six dummies) are sorted by digit through the
/// routing network so the steps leaving positions 0 and 1 hash only
/// [`STEP_SLOTS`] positions; then they are routed back and every chain takes
/// the step leaving position 2 if its digit allows. `sel` holds the
/// canonical selectors (bit 2: digit ≤ 2), `ctrl` the control words per group
/// and block.
#[allow(clippy::too_many_arguments)]
fn chain_ends(
    b: &mut Builder,
    slots: &mut Slots,
    k: &Constants,
    pp: Wire,
    digest: Wire,
    a_chain: &[Wire; CHAIN_END],
    sel: &[Wire],
    chains: &[Wire],
    ctrl: &[Vec<Wire>],
) -> Vec<Wire> {
    // Sort the chains (and six dummies) by digit through the network, so
    // the steps leaving positions 0 and 1 need only STEP_SLOTS hashes.
    let mut value: Vec<Wire> = (0..WIDTH)
        .map(|i| if i < V { chains[i] } else { k.zero })
        .collect();
    let mut index = k.chain_index.clone();
    for g in 0..GROUPS {
        for blk in 0..BLOCKS {
            let pos: Vec<usize> = (0..8).map(|e| position(g, blk, e)).collect();
            let mut ins = vec![ctrl[g][blk]];
            ins.extend(pos.iter().map(|&p| value[p]));
            ins.extend(pos.iter().map(|&p| index[p]));
            let slot = slots.benes.next();
            let out = b.sb.gate(slot, &ins);
            for (e, &p) in pos.iter().enumerate() {
                value[p] = out[e];
                index[p] = out[8 + e];
            }
        }
    }
    // Each routed position's digit. A position past a step's slots must
    // not take that step, which with the codeword check puts every chain
    // that does within the slots.
    let mut routed_sel = Vec::with_capacity(WIDTH);
    for (j, &index_j) in index.iter().enumerate() {
        let class = if j < STEP_SLOTS[0] {
            k.class[0]
        } else if j < STEP_SLOTS[1] {
            k.class[1]
        } else {
            k.class[2]
        };
        let slot = slots.lookup.next();
        routed_sel.push(b.sb.gate(slot, &[digest, index_j, class])[0]);
    }
    for (step, &a) in a_chain[..2].iter().enumerate() {
        for j in 0..STEP_SLOTS[step] {
            let h = th64(b, slots, k, [pp, a, value[j], index[j]]);
            let slot = slots.select[step].next();
            value[j] = b.sb.gate(slot, &[routed_sel[j], value[j], h])[0];
        }
    }
    // Back to chain order (the same control words, stages reversed), then
    // the step leaving position 2 for every chain.
    for g in (0..GROUPS).rev() {
        for blk in 0..BLOCKS {
            let pos: Vec<usize> = (0..8).map(|e| position(g, blk, e)).collect();
            let mut ins = vec![ctrl[g][blk]];
            ins.extend(pos.iter().map(|&p| value[p]));
            let slot = slots.benes_reverse.next();
            let out = b.sb.gate(slot, &ins);
            for (e, &p) in pos.iter().enumerate() {
                value[p] = out[e];
            }
        }
    }
    let mut ends = Vec::with_capacity(V);
    for (i, &sel_i) in sel[..V].iter().enumerate() {
        let h = th64(b, slots, k, [pp, a_chain[2], value[i], k.chain_index[i]]);
        let slot = slots.select[2].next();
        ends.push(b.sb.gate(slot, &[sel_i, value[i], h])[0]);
    }
    ends
}

/// Signature `s`: wires its inputs and checks both layers, ending at the
/// public root.
fn signature(b: &mut Builder, slots: &mut Slots, k: &Constants, s: usize) {
    let pp = b.public(Source::Pp(s));
    let root = b.public(Source::Root(s));
    let msg = [b.public(Source::Msg(s, 0)), b.public(Source::Msg(s, 1))];
    // The shift chain R_j = idx >> j (j = 0..32): R_16 is the top layer's
    // key index, and bit 0 of R_j is the Merkle direction at bit j of idx.
    let mut shifted = vec![b.input(Source::Idx(s))];
    let mut node_address = [Vec::new(), Vec::new()];
    for layer in 0..2 {
        for level in 1..=HEIGHT {
            let slot = slots.address_step.next();
            let out = b.sb.gate(
                slot,
                &[*shifted.last().unwrap(), k.address[layer][node(level)]],
            );
            shifted.push(out[0]);
            node_address[layer].push(out[1]);
        }
    }
    let mut bottom_root = None;
    for layer in 0..2 {
        let e = shifted[HEIGHT * layer];
        let address = |b: &mut Builder, slots: &mut Slots, pos: usize| {
            let slot = slots.address.next();
            b.sb.gate(slot, &[e, k.address[layer][pos]])[0]
        };
        let a_msg = address(b, slots, MSG);
        let a_chain: [Wire; CHAIN_END] =
            std::array::from_fn(|step| address(b, slots, chain_step(step)));
        let a_leaf = address(b, slots, LEAF);
        let rho = b.input(Source::Rho(s, layer));
        // The message digest H0(P ‖ A_msg ‖ M ‖ ρ): 80 bytes at the bottom,
        // 64 at the top (M is the bottom root).
        let digest = match bottom_root {
            None => {
                let first = compress(b, slots, k.iv, [pp, a_msg, msg[0], msg[1]], k.first);
                compress(
                    b,
                    slots,
                    [first[0], first[1]],
                    [rho, k.zero, k.zero, k.zero],
                    k.last16,
                )[0]
            }
            Some(x) => th64(b, slots, k, [pp, a_msg, x, rho]),
        };
        let slot = slots.codeword.next();
        b.sb.gate(slot, &[digest]);
        let mut sel = Vec::with_capacity(SPREADS * SPREAD_DIGITS);
        let mut r = digest;
        for _ in 0..SPREADS {
            let slot = slots.spread.next();
            let out = b.sb.gate(slot, &[r]);
            sel.extend_from_slice(&out[..SPREAD_DIGITS]);
            r = out[SPREAD_DIGITS];
        }
        let chains: Vec<Wire> = (0..V)
            .map(|i| b.input(Source::Chain(s, layer, i)))
            .collect();
        let ctrl: Vec<Vec<Wire>> = (0..GROUPS)
            .map(|g| {
                (0..BLOCKS)
                    .map(|blk| b.input(Source::Control(s, layer, g, blk)))
                    .collect()
            })
            .collect();
        let ends = chain_ends(b, slots, k, pp, digest, &a_chain, &sel, &chains, &ctrl);
        // The one-time key's leaf: H0(P ‖ A_leaf ‖ end_0 ‖ … ‖ end_57), 960
        // bytes in 15 chained compressions.
        let words: Vec<Wire> = [pp, a_leaf].into_iter().chain(ends).collect();
        let mut cv = k.iv;
        for (blk, m) in words.as_chunks::<4>().0.iter().enumerate() {
            let params = match blk {
                0 => k.first,
                14 => k.last,
                _ => k.middle,
            };
            let out = compress(b, slots, cv, [m[0], m[1], m[2], m[3]], params);
            cv = [out[0], out[1]];
        }
        let mut node = cv[0];
        for level in 0..HEIGHT {
            let sibling = b.input(Source::Sibling(s, layer, level));
            let dir = shifted[HEIGHT * layer + level];
            let slot = slots.swap.next();
            let lr = b.sb.gate(slot, &[dir, node, sibling]);
            node = th64(b, slots, k, [pp, node_address[layer][level], lr[0], lr[1]]);
        }
        bottom_root = Some(node);
    }
    b.sb.connect(bottom_root.unwrap(), root);
}

#[cfg(test)]
mod tests;
