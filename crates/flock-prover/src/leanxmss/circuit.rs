//! The circuit verifying `n` leanXMSS\[BLAKE3\] signatures in one proof.
//!
//! Every `Th` call is BLAKE3 compressions, one per row of the compression
//! gate ([`super::compress`]): 201 per signature
//! ([`COMPRESSIONS_PER_SIGNATURE`]), exactly what native verification
//! spends. Most are fixed: the digests, the 15 blocks of each one-time
//! key's leaf and the Merkle nodes are wired like any circuit. The 68 chain
//! steps of a layer are not — which chain a step advances depends on the
//! digits — so they move their values over the tuple bus
//! ([`flock_core::circuit::bus`], `docs/tuple-bus-design.md`).
//!
//! A bus tuple is `(S, c, A, v)`: the layer `S` (a static row tag: signer
//! and layer), the chain `c`, the chain address `A(µ)` (which carries the
//! position `µ` in its step field) and the value there. Per chain, a chain
//! row sends its start `(S, c, A(d_c), x_c)` and receives its end
//! `(S, c, A(3), e_c)`; the end is wired into the leaf. Each step row
//! receives `(S, m3, m1, m2)`, hashes `P ‖ m1 ‖ m2 ‖ m3` and sends
//! `(S, m3, next(m1), H0)`. Every layer has exactly 68 step rows, so with
//! the digit sum checked, the only way to balance is to walk each chain
//! from its digit to the end.
//!
//! Per signer the statement is public: `P`, the root and the 32-byte message.
//! The leaf index, randomizers, chain values and Merkle siblings are private.

use flock_core::{
    challenger::FsChallenger,
    circuit::{
        boolean::BooleanGate,
        builder::{CircuitShape, CircuitWitness, FillPlan, ShapeBuilder, SlotId, Wire},
        bus::{BusDirection, BusTerm, RowTag},
    },
    lincheck::LincheckCircuit,
    merkle::HashKind,
    pcs::{
        Commitment, PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    union::UnionInstance,
    verifier::FlockVerifyError,
};
use flock_field::F128;

use super::{
    compress::{self, CompressPacker, M, NEXT, O0, compress_gate, ctl},
    gadgets::{
        address_gate, address_step_gate, chain_gate, chain_io, codeword_gate,
        layer_address_constants, swap_gate,
    },
    reference::{
        CHAIN_END, HEIGHT, PublicKey, Signature, TARGET_SUM, V, Value, XmssHash, chain_address,
        codeword, layer_root, le128, msg_address, th,
    },
};
use crate::{prover::UnionSlotProverInput, r1cs_hashes::blake3::BLAKE3_IV};

/// Chain steps per layer: every codeword takes exactly this many.
pub const STEPS: usize = V * CHAIN_END - TARGET_SUM;
/// Leaf blocks: `P ‖ A_leaf` and the 58 ends, four 16-byte words a block.
pub const LEAF_BLOCKS: usize = (2 + V).div_ceil(4);
/// Compressions per signature: both layers' chain steps, leaves and paths,
/// and the digests (2 + 1).
pub const COMPRESSIONS_PER_SIGNATURE: usize = 2 * (STEPS + LEAF_BLOCKS + HEIGHT) + 3;

pub(crate) const FIAT_SHAMIR_DOMAIN: &[u8] = b"flock-leanxmss-blake3-v1";

/// The proof's transcript and its optional optimisations.
///
/// [`Self::DEFAULT`] is the shipped proof. [`Self::RECURSION_LEAF`] is what
/// the chain tower's walkers can replay in-circuit today
/// (`docs/leanxmss-recursion-plan.md`): the chained BLAKE3 transcript and
/// none of the optimisations, each of which the walkers refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XmssOptions {
    /// The chained BLAKE3 transcript ([`FsChallenger::with_chained_blake3`])
    /// instead of the tree transcript.
    pub chained_transcript: bool,
    /// Each gate's degree-one words leave the zerocheck
    /// ([`BooleanGate::with_zeroed_linear_words`]).
    pub zero_linear_words: bool,
    /// Cell-slots nobody wires leave the wiring
    /// ([`ShapeBuilder::prune_unwired`]).
    pub prune_unwired: bool,
    /// Columns that take only public words leave the grand product
    /// ([`ShapeBuilder::open_public_columns`]).
    pub public_columns: bool,
    /// The grand product's slots in order of live count
    /// ([`ShapeBuilder::pack_live_slots`]).
    pub pack_live_slots: bool,
    /// The signers' words (`P`, root, message) are private and hashed
    /// in-circuit; the statement digest ([`XmssCircuit::statement_digest`])
    /// is published in their place, so a verifier of this proof reads two
    /// words instead of four per signer.
    pub hashed_statement: bool,
}

impl XmssOptions {
    pub const DEFAULT: Self = Self {
        chained_transcript: false,
        zero_linear_words: true,
        prune_unwired: true,
        public_columns: true,
        pack_live_slots: true,
        hashed_statement: false,
    };

    pub const RECURSION_LEAF: Self = Self {
        chained_transcript: true,
        zero_linear_words: false,
        prune_unwired: false,
        public_columns: false,
        pack_live_slots: false,
        hashed_statement: true,
    };
}

impl Default for XmssOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// BLAKE3 flags.
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
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
    Sibling(usize, usize, usize),
    /// Signer, layer, chain: the revealed value and the end.
    Start(usize, usize, usize),
    End(usize, usize, usize),
    /// Signer, layer, step row, word (`[A, v, c]`).
    Step(usize, usize, usize, usize),
    /// Word `k` of the statement digest ([`XmssOptions::hashed_statement`]).
    Digest(usize),
}

/// One layer's chain values: the revealed values, the ends, and the step
/// rows' `[A(µ), v, c]` in chain order.
struct LayerPlan {
    starts: [F128; V],
    ends: [F128; V],
    steps: Vec<[F128; 3]>,
}

/// Walks both layers natively: every chain from its digit to the end.
fn signature_plan(statement: &Statement, sig: &Signature) -> [LayerPlan; 2] {
    let hash = XmssHash::Blake3;
    let pp = statement.pk.pp;
    let mut msg = statement.msg.to_vec();
    std::array::from_fn(|layer| {
        let e = if layer == 0 {
            sig.idx
        } else {
            sig.idx >> HEIGHT
        };
        let ls = &sig.layers[layer];
        let digest = th(hash, &pp, &msg_address(layer, e), &[&msg, &ls.rho]);
        let digits = codeword(&digest).expect("the signature does not verify");
        let mut ends = [F128::ZERO; V];
        let mut steps = Vec::with_capacity(STEPS);
        for (c, end) in ends.iter_mut().enumerate() {
            let mut v = ls.y[c];
            for step in digits[c] as usize..CHAIN_END {
                let a = chain_address(layer, e, step);
                steps.push([word(&a), word(&v), word(&le128(c))]);
                v = th(hash, &pp, &a, &[&v, &le128(c)]);
            }
            *end = word(&v);
        }
        assert_eq!(steps.len(), STEPS, "the digits are not a codeword");
        if layer == 0 {
            msg = layer_root(hash, &pp, 0, e, &msg, ls, &mut None)
                .unwrap()
                .to_vec();
        }
        LayerPlan {
            starts: ls.y.each_ref().map(word),
            ends,
            steps,
        }
    })
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
    gadgets: Vec<(SlotId, BooleanGate)>,
    /// The compression slots' driver, in place of the gate's packer.
    compress: CompressPacker,
    inputs: Vec<Source>,
    publics: Vec<Source>,
    options: XmssOptions,
}

/// Records each declared input's source alongside the shape.
struct Builder {
    sb: ShapeBuilder,
    inputs: Vec<Source>,
    publics: Vec<Source>,
    /// The signers' words, when they are hashed rather than public.
    statement: Option<Vec<Wire>>,
}

impl Builder {
    /// A signer's word: public, or private and queued for the statement
    /// hash.
    fn statement_word(&mut self, source: Source) -> Wire {
        match self.statement.as_mut() {
            Some(words) => {
                self.inputs.push(source);
                let w = self.sb.input();
                words.push(w);
                w
            }
            None => self.public(source),
        }
    }

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
    /// Control words ([`ctl`]): a lone 64-byte block, a first block, a
    /// middle block, a 64-byte last block, and the 16-byte last block of the
    /// bottom digest.
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
const CHAIN: usize = 1;
const LEAF: usize = 2;
const fn node(level: usize) -> usize {
    2 + level
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

/// The declared gate kinds. Chain steps and the other compressions are the
/// same gate in separate slots: a bus port covers whole slots.
struct Slots {
    step: Kind,
    hash: Kind,
    chain: Kind,
    swap: Kind,
    codeword: Kind,
    address: Kind,
    address_step: Kind,
}

/// Gates of each kind per signature, in [`Slots`] order.
const GADGETS_PER_SIGNATURE: [usize; 7] = [
    2 * STEPS,
    COMPRESSIONS_PER_SIGNATURE - 2 * STEPS,
    2 * V,
    2 * HEIGHT,
    2,
    2 * 3,
    2 * HEIGHT,
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

/// The gate kinds at `nu`, in [`Slots`] order. The two compression kinds
/// share one gate: building it is most of the circuit's setup. Every kind
/// drops its degree-one words (inputs, copied or XORed outputs, the pin's
/// word) from the zerocheck; chain, address and address-step have no other
/// word.
fn gates(nu: usize) -> [BooleanGate; 7] {
    gates_keeping_every_word(nu).map(BooleanGate::with_zeroed_linear_words)
}

/// [`gates`] without dropping degree-one words from the zerocheck.
fn gates_keeping_every_word(nu: usize) -> [BooleanGate; 7] {
    let compress = compress_gate(nu).unwrap();
    [
        compress.clone(),
        compress,
        chain_gate(nu).unwrap(),
        swap_gate(nu).unwrap(),
        codeword_gate(nu).unwrap(),
        address_gate(nu).unwrap(),
        address_step_gate(nu).unwrap(),
    ]
}

/// Declares every gate kind, the counts in [`Slots`] order, each over
/// `⌈count/2^nu⌉` slots (at least one), and the bus ports of the step and
/// chain slots. Returns the kinds and every slot with its gate.
fn declare_slots(
    b: &mut Builder,
    nu: usize,
    counts: &[usize],
    zero_linear_words: bool,
) -> (Slots, Vec<(SlotId, BooleanGate)>) {
    let mut gadgets = Vec::new();
    let kinds = if zero_linear_words {
        gates(nu)
    } else {
        gates_keeping_every_word(nu)
    };
    let mut declared: Vec<Kind> = kinds
        .into_iter()
        .zip(counts)
        .map(|(gate, &count)| {
            let slots = (0..count.div_ceil(1 << nu).max(1))
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
        step: take(),
        hash: take(),
        chain: take(),
        swap: take(),
        codeword: take(),
        address: take(),
        address_step: take(),
    };
    // Rows are emitted layer by layer, so the `k`-th row of a kind (over its
    // slots) belongs to layer `k / per_layer`: that index is the tuple's `S`.
    let tag = |slot: usize, div: usize, modulo: usize| {
        BusTerm::row(RowTag {
            offset: (slot << nu) as u64,
            div: div as u64,
            modulo: modulo as u64,
        })
    };
    for (i, &id) in slots.step.slots.iter().enumerate() {
        let layer = tag(i, STEPS, 0);
        let m = |k: usize| BusTerm::word(M + k);
        b.sb.bus_port(id, BusDirection::Recv, vec![layer, m(3), m(1), m(2)]);
        b.sb.bus_port(
            id,
            BusDirection::Send,
            vec![layer, m(3), BusTerm::word(NEXT), BusTerm::word(O0)],
        );
    }
    let step3 = F128::new(3 << 61, 0);
    for (i, &id) in slots.chain.slots.iter().enumerate() {
        let (layer, chain) = (tag(i, V, 0), tag(i, 1, V));
        b.sb.bus_port(
            id,
            BusDirection::Send,
            vec![
                layer,
                chain,
                BusTerm::word(chain_io::S),
                BusTerm::word(chain_io::X),
            ],
        );
        b.sb.bus_port(
            id,
            BusDirection::Recv,
            vec![
                layer,
                chain,
                BusTerm::word(chain_io::A).plus(step3),
                BusTerm::word(chain_io::E),
            ],
        );
    }
    (slots, gadgets)
}

/// The signers' words in statement order: per signer `P`, the root and
/// the message's two halves.
fn statement_words(statements: &[Statement]) -> Vec<F128> {
    statements
        .iter()
        .flat_map(|st| {
            [
                word(&st.pk.pp),
                word(&st.pk.root),
                word(&st.msg[..16].try_into().unwrap()),
                word(&st.msg[16..].try_into().unwrap()),
            ]
        })
        .collect()
}

/// The statement digest: [`flock_core::union::publics_digest`] of the
/// signers' words — 1 KiB BLAKE3 chunks (16 signers each), left-folded
/// through parents, never root-flagged.
pub fn statement_digest(statements: &[Statement]) -> [u8; 32] {
    flock_core::union::publics_digest(&statement_words(statements))
}

/// A 32-byte digest as two words, little-endian.
fn digest_words(d: &[u8; 32]) -> [F128; 2] {
    std::array::from_fn(|k| word(&d[16 * k..16 * k + 16].try_into().unwrap()))
}

/// Compressions the statement hash adds for `n` signers: one block per
/// signer and a parent per chunk after the first.
fn statement_hash_compressions(n: usize) -> usize {
    n + n.div_ceil(16) - 1
}

/// Hash the queued signer words ([`statement_digest`], gate for gate) and
/// publish the digest in their place.
fn hash_statement(b: &mut Builder, slots: &mut Slots, k: &Constants) {
    let words = b.statement.take().expect("hashed statement words");
    let [first_last, last, parent] =
        [CHUNK_START | CHUNK_END, CHUNK_END, PARENT].map(|f| b.constant(ctl(f, 64)));
    let mut cv: Option<[Wire; 2]> = None;
    for chunk in words.chunks(64) {
        let blocks = chunk.len() / 4;
        let mut lcv = k.iv;
        for (i, m) in chunk.chunks(4).enumerate() {
            let ctl = match (i == 0, i + 1 == blocks) {
                (true, true) => first_last,
                (true, false) => k.first,
                (false, true) => last,
                (false, false) => k.middle,
            };
            lcv = compress(b, &mut slots.hash, lcv, [m[0], m[1], m[2], m[3]], ctl);
        }
        cv = Some(match cv {
            None => lcv,
            Some(prev) => compress(
                b,
                &mut slots.hash,
                k.iv,
                [prev[0], prev[1], lcv[0], lcv[1]],
                parent,
            ),
        });
    }
    for (j, w) in cv.expect("at least one signer").into_iter().enumerate() {
        b.sb.publish(w);
        b.publics.push(Source::Digest(j));
    }
}

fn declare_constants(b: &mut Builder) -> Constants {
    Constants {
        iv: iv_words().map(|w| b.constant(w)),
        zero: b.constant(F128::ZERO),
        single: b.constant(ctl(CHUNK_START | CHUNK_END | ROOT, 64)),
        first: b.constant(ctl(CHUNK_START, 64)),
        middle: b.constant(ctl(0, 64)),
        last: b.constant(ctl(CHUNK_END | ROOT, 64)),
        last16: b.constant(ctl(CHUNK_END | ROOT, 16)),
        address: [0, 1].map(|layer| {
            layer_address_constants(layer)
                .iter()
                .map(|c| b.constant(word(c)))
                .collect()
        }),
    }
}

impl XmssCircuit {
    /// The circuit for `n ≥ 1` signatures, its slot height from [`plan_nu`].
    pub fn new(n: usize) -> Self {
        Self::with_nu(n, None)
    }

    /// The circuit for `n ≥ 1` signatures with `2^nu` rows per slot, or
    /// [`plan_nu`]'s choice.
    pub fn with_nu(n: usize, nu: Option<usize>) -> Self {
        Self::with_options(n, nu, XmssOptions::DEFAULT)
    }

    /// [`Self::with_nu`], proving with `options`.
    pub fn with_options(n: usize, nu: Option<usize>, options: XmssOptions) -> Self {
        assert!(n >= 1);
        let mut kinds: Vec<(usize, usize)> = gates(3)
            .iter()
            .zip(GADGETS_PER_SIGNATURE)
            .map(|(g, per_sig)| (n * per_sig, g.k_log()))
            .collect();
        if options.hashed_statement {
            kinds[1].0 += statement_hash_compressions(n);
        }
        let nu = nu.unwrap_or_else(|| plan_nu(&kinds));
        let mut b = Builder {
            sb: ShapeBuilder::new(nu),
            inputs: Vec::new(),
            publics: Vec::new(),
            statement: options.hashed_statement.then(Vec::new),
        };
        // The step rows' bus words and every row's unused outputs need no
        // copy constraint, and the columns that only take constants or the
        // signer's public words (IV, control words, address constants, P) can
        // be checked by the opening. That leaves dead cell-slots among live
        // ones, so the grand product packs the live ones together. The
        // recursion leaf keeps all of them in the wiring.
        if options.prune_unwired {
            b.sb.prune_unwired();
        }
        if options.public_columns {
            b.sb.open_public_columns();
        }
        if options.pack_live_slots {
            b.sb.pack_live_slots();
        }
        let counts: Vec<usize> = kinds.iter().map(|&(count, _)| count).collect();
        let (mut slots, gadgets) = declare_slots(&mut b, nu, &counts, options.zero_linear_words);
        let constants = declare_constants(&mut b);
        for s in 0..n {
            let island = b.sb.begin_island();
            signature(&mut b, &mut slots, &constants, s);
            b.sb.end_island(island);
        }
        if options.hashed_statement {
            hash_statement(&mut b, &mut slots, &constants);
        }
        let shape = b.sb.finish().expect("the leanXMSS circuit is well formed");
        let plan = shape.fill_plan();
        let compress = gadgets
            .iter()
            .find(|(_, gate)| gate.name() == compress::NAME)
            .map(|(_, gate)| CompressPacker::new(gate))
            .unwrap();
        XmssCircuit {
            n,
            nu,
            shape,
            plan,
            gadgets,
            compress,
            inputs: b.inputs,
            publics: b.publics,
            options,
        }
    }

    pub fn signatures(&self) -> usize {
        self.n
    }

    pub fn options(&self) -> XmssOptions {
        self.options
    }

    pub(crate) fn challenger(&self) -> FsChallenger {
        if self.options.chained_transcript {
            FsChallenger::with_chained_blake3(FIAT_SHAMIR_DOMAIN)
        } else {
            FsChallenger::new(FIAT_SHAMIR_DOMAIN)
        }
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
        sigs: Option<(&[Signature], &[[LayerPlan; 2]])>,
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
            Source::Sibling(s, l, k) => word(&sig(s).layers[l].path[k]),
            Source::Start(s, l, c) => private().1[s][l].starts[c],
            Source::End(s, l, c) => private().1[s][l].ends[c],
            Source::Step(s, l, i, k) => private().1[s][l].steps[i][k],
            Source::Digest(_) => unreachable!("the digest is an output"),
        }
    }

    /// The public words, in publication order: the shared constants, then
    /// per signer `P`, the root and the message — or, with
    /// [`XmssOptions::hashed_statement`], the statement digest's two words.
    pub fn public(&self, statements: &[Statement]) -> Vec<F128> {
        assert_eq!(statements.len(), self.n);
        let digest = self
            .options
            .hashed_statement
            .then(|| digest_words(&statement_digest(statements)));
        self.publics
            .iter()
            .map(|&src| match src {
                Source::Digest(k) => digest.expect("a hashed statement")[k],
                _ => Self::resolve(src, statements, None),
            })
            .collect()
    }

    /// The statement digest a [`XmssOptions::hashed_statement`] proof
    /// publishes: [`statement_digest`].
    pub fn statement_digest(&self, statements: &[Statement]) -> [u8; 32] {
        assert_eq!(statements.len(), self.n);
        statement_digest(statements)
    }

    /// The online phase: every gate evaluated, rows collected per slot. Each
    /// signature must verify natively; a gadget panics on one that does not.
    pub fn witness(&self, statements: &[Statement], sigs: &[Signature]) -> CircuitWitness {
        assert_eq!((statements.len(), sigs.len()), (self.n, self.n));
        use rayon::prelude::{IntoParallelIterator, ParallelIterator};
        let plans: Vec<[LayerPlan; 2]> = (0..self.n)
            .into_par_iter()
            .map(|s| signature_plan(&statements[s], &sigs[s]))
            .collect();
        let values: Vec<F128> = self
            .inputs
            .iter()
            .map(|&src| Self::resolve(src, statements, Some((sigs, &plans))))
            .collect();
        self.shape.run_filled_deferred(&self.plan, &values, &[])
    }

    /// The union instance. Every used column is committed at full height
    /// (dummy rows as zeros), so the opening takes the block-first transport
    /// instead of materializing the jagged weight.
    pub(crate) fn union(&self) -> UnionInstance<'_> {
        let mut union = UnionInstance::new(&self.shape.registry, self.shape.counts.clone());
        union.set_full_height_commit();
        union
    }

    pub(crate) fn pcs_params(union: &UnionInstance<'_>) -> PcsParams {
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
    pub(crate) fn lincheck_circuits(&self) -> Vec<&dyn LincheckCircuit> {
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
        by_slot.sort_by_key(|&(slot, _)| slot);
        by_slot.into_iter().map(|(_, c)| c).collect()
    }

    /// Each slot's in-place witness driver, in registry order.
    fn slot_inputs<'a>(&'a self, witness: &'a CircuitWitness) -> Vec<UnionSlotProverInput<'a>> {
        let mut by_slot: Vec<(usize, UnionSlotProverInput<'a>)> = self
            .gadgets
            .iter()
            .map(|(id, gate)| {
                let rows = witness.rows::<BooleanGate>(*id);
                let compress = (gate.name() == compress::NAME).then_some(&self.compress);
                (
                    self.shape.registry_slot(*id),
                    UnionSlotProverInput::in_place(
                        move |dst| match compress {
                            Some(packer) => packer.pack_into(rows, dst),
                            None => gate.pack_into(rows, dst),
                        },
                        gate.r1cs().csc_lincheck_circuit(),
                    ),
                )
            })
            .collect();
        by_slot.sort_by_key(|&(slot, _)| slot);
        by_slot.into_iter().map(|(_, input)| input).collect()
    }

    pub fn prove(&self, witness: &CircuitWitness) -> XmssProof {
        let union = self.union();
        let pcs = Self::pcs_params(&union);
        let mut challenger = self.challenger();
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
        let mut challenger = self.challenger();
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

/// One compression of `cv` and the block `m` in `kind`'s next row, its
/// flags and block length set by `ctl` ([`ctl`]): the output chaining value.
fn compress(b: &mut Builder, kind: &mut Kind, cv: [Wire; 2], m: [Wire; 4], ctl: Wire) -> [Wire; 2] {
    let slot = kind.next();
    let out =
        b.sb.gate(slot, &[cv[0], cv[1], m[0], m[1], m[2], m[3], ctl]);
    [out[0], out[1]]
}

/// `Th` of a 64-byte input: one root compression from the IV; `H0` is the
/// first output word.
fn th64(b: &mut Builder, slots: &mut Slots, k: &Constants, m: [Wire; 4]) -> Wire {
    compress(b, &mut slots.hash, k.iv, m, k.single)[0]
}

/// Signature `s`: wires its inputs and checks both layers, ending at the
/// public root.
fn signature(b: &mut Builder, slots: &mut Slots, k: &Constants, s: usize) {
    let pp = b.statement_word(Source::Pp(s));
    let root = b.statement_word(Source::Root(s));
    let msg = [
        b.statement_word(Source::Msg(s, 0)),
        b.statement_word(Source::Msg(s, 1)),
    ];
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
        let a_chain = address(b, slots, CHAIN);
        let a_leaf = address(b, slots, LEAF);
        let rho = b.input(Source::Rho(s, layer));
        // The message digest H0(P ‖ A_msg ‖ M ‖ ρ): 80 bytes at the bottom,
        // 64 at the top (M is the bottom root).
        let digest = match bottom_root {
            None => {
                let first = compress(
                    b,
                    &mut slots.hash,
                    k.iv,
                    [pp, a_msg, msg[0], msg[1]],
                    k.first,
                );
                compress(
                    b,
                    &mut slots.hash,
                    first,
                    [rho, k.zero, k.zero, k.zero],
                    k.last16,
                )[0]
            }
            Some(x) => th64(b, slots, k, [pp, a_msg, x, rho]),
        };
        let slot = slots.codeword.next();
        b.sb.gate(slot, &[digest]);
        // One chain row per chain: its start goes out on the bus, its end
        // comes back and is wired into the leaf.
        let mut digits = digest;
        let mut ends = Vec::with_capacity(V);
        for c in 0..V {
            let x = b.input(Source::Start(s, layer, c));
            let end = b.input(Source::End(s, layer, c));
            let slot = slots.chain.next();
            digits = b.sb.gate(slot, &[digits, a_chain, x, end])[0];
            ends.push(end);
        }
        // The layer's chain steps, in whatever order the digits give.
        for i in 0..STEPS {
            let [a, v, c] = std::array::from_fn(|w| b.input(Source::Step(s, layer, i, w)));
            compress(b, &mut slots.step, k.iv, [pp, a, v, c], k.single);
        }
        // The leaf H0(P ‖ A_leaf ‖ end_0 ‖ … ‖ end_57): 15 blocks of one chunk.
        let words: Vec<Wire> = [pp, a_leaf].into_iter().chain(ends).collect();
        let mut cv = k.iv;
        for (i, block) in words.chunks(4).enumerate() {
            let flags = match i {
                0 => k.first,
                i if i + 1 == LEAF_BLOCKS => k.last,
                _ => k.middle,
            };
            cv = compress(b, &mut slots.hash, cv, block.try_into().unwrap(), flags);
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
