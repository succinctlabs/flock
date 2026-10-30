//! The tuple bus: a multiset equality between tuples that gate rows SEND and
//! tuples they RECEIVE, proven by a grand product next to the wiring argument.
//!
//! The wiring argument proves copy constraints along a permutation σ fixed
//! when the circuit is built. Some dataflow is not fixed: which row of a
//! hash-chain table steps which chain depends on the witness. A bus moves
//! such values by CONTENT instead of by position. Each [`BusPort`] is one
//! gate type's rows, every live row contributing one tuple of
//! [`Bus::width`] coordinates, and the argument proves
//!
//! ```text
//! { tuple(p, j) : p a Send port, j < count(p) }
//!   = { tuple(p, j) : p a Recv port, j < count(p) }        (as multisets)
//! ```
//!
//! A coordinate ([`BusTerm`]) is a committed word of the row (one of its
//! type's gate cell-slots), plus a constant, plus a static [`RowTag`] — the
//! part that ties a tuple to a region of rows (a signature, say) without
//! committing anything. `docs/tuple-bus-design.md` has the soundness
//! argument and how leanXMSS uses it.
//!
//! ## Protocol
//!
//! After the wiring GKR, on the same transcript: sample `β` (ground like the
//! wiring's `α`) and `γ_1..γ_w`; a tuple `t` becomes the leaf
//! `β + Σ_k γ_k·t_k`. Send ports fill the left product's groups and receive
//! ports the right's, one group of `2^ν` rows per port, dead rows and unused
//! groups being 1. [`product_gkr::prove_leaves`] proves the products equal
//! and reduces both sides to one point `ρ = (ρ_row, ρ_port)`. Each side's
//! leaf MLE there is affine in the committed words' MLEs at `ρ_row`:
//!
//! ```text
//! L̂(ρ) = 1 + Σ_p eq(ρ_port, p)·[ (β + 1)·E_p + Σ_k γ_k·(ŵ_{p,k}(ρ_row)
//!                                   + c_{p,k}·E_p + T_{p,k}(ρ_row)) ]
//! ```
//!
//! with `E_p = Σ_{j<count} eq(ρ_row, j)` and `T` the row tag's MLE over the
//! live rows. The prover sends one gather value per committed word used —
//! the same packed-direct claims the wiring's gather produces, bound to the
//! commitment by the opening — and the verifier rebuilds both sides.

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, ParallelSliceMut,
};
use serde::{Deserialize, Serialize};

use crate::{
    challenger::Challenger,
    circuit::{CellSlot, Circuit, CircuitError, WiringError},
    field::F128,
    pcs::{DirectEqInd, PackedDirectClaim},
    product_gkr::{
        BatchedGrinding, LiveMask, ProductGkrLeavesProof, ProductLeaves, prove_leaves,
        verify_leaves,
    },
    scratch::take_f128,
    zerocheck::univariate_skip::build_eq,
};

const DOMAIN_BUS: &[u8] = b"flock-tuple-bus-v1";

/// Which product a port's tuples join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusDirection {
    Send,
    Recv,
}

/// A static per-row value: row `j` of the port gets the integer
/// `((offset + j) / div) mod modulo` (no reduction when `modulo = 0`), as the
/// field element with that bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowTag {
    pub offset: u64,
    pub div: u64,
    pub modulo: u64,
}

impl RowTag {
    #[inline]
    pub fn value(&self, row: usize) -> F128 {
        let q = (self.offset + row as u64) / self.div;
        F128::new(if self.modulo == 0 { q } else { q % self.modulo }, 0)
    }
}

/// One tuple coordinate: `word + constant + row tag`, each part optional. In
/// a circuit, `word` is a gate cell-slot of the port's type; at the builder
/// (`ShapeBuilder::bus_port`), it is an index into the gate's io schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusTerm {
    pub word: Option<usize>,
    pub constant: F128,
    pub row: Option<RowTag>,
}

impl BusTerm {
    /// A committed word of the row.
    pub const fn word(word: usize) -> Self {
        Self {
            word: Some(word),
            constant: F128::ZERO,
            row: None,
        }
    }

    /// A per-row static value.
    pub const fn row(tag: RowTag) -> Self {
        Self {
            word: None,
            constant: F128::ZERO,
            row: Some(tag),
        }
    }

    /// The same term plus a constant.
    pub const fn plus(self, constant: F128) -> Self {
        Self { constant, ..self }
    }
}

/// One gate type's live rows, each contributing a tuple to one side.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusPort {
    /// Registry slot index.
    pub ty: usize,
    pub dir: BusDirection,
    pub terms: Vec<BusTerm>,
}

/// A circuit's bus: every port's tuple has `width` coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bus {
    pub width: usize,
    pub ports: Vec<BusPort>,
}

/// The bus half of a wiring proof: the grand product's transcript, one
/// gather value per committed word the ports read (ascending cell-slot), and
/// the fingerprint challenge's PoW nonce when grinding is on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusProof {
    pub gkr: ProductGkrLeavesProof,
    pub gather: Vec<F128>,
    pub fingerprint_nonce: Vec<u64>,
}

impl Bus {
    /// Check the bus against the circuit it rides on: every port names a
    /// registry type, carries `width` terms, and reads only that type's own
    /// cell-slots.
    pub(crate) fn validate(&self, circuit: &Circuit) -> Result<(), CircuitError> {
        let bad = |why: &'static str| Err(CircuitError::InvalidBus(why));
        if self.width == 0 {
            return bad("a bus tuple needs a coordinate");
        }
        if self.ports.is_empty() {
            return bad("a bus needs a port");
        }
        let slots = circuit.cells().slots();
        for port in &self.ports {
            if port.ty >= circuit.counts().len() {
                return bad("a port names no registry type");
            }
            if port.terms.len() != self.width {
                return bad("a port's tuple has the wrong width");
            }
            for term in &port.terms {
                if term.row.is_some_and(|t| t.div == 0) {
                    return bad("a row tag divides by zero");
                }
                if let Some(iota) = term.word {
                    match slots.get(iota) {
                        Some(CellSlot::Gate { ty, .. }) if *ty == port.ty => {}
                        _ => return bad("a term reads a word outside its port's type"),
                    }
                }
            }
        }
        Ok(())
    }

    /// The digest encoding: fixed-width fields, length-prefixed lists.
    pub(crate) fn absorb_into(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(DOMAIN_BUS);
        buf.extend_from_slice(&(self.width as u32).to_le_bytes());
        buf.extend_from_slice(&(self.ports.len() as u32).to_le_bytes());
        for port in &self.ports {
            buf.extend_from_slice(&(port.ty as u32).to_le_bytes());
            buf.push(matches!(port.dir, BusDirection::Recv) as u8);
            for term in &port.terms {
                match term.word {
                    Some(iota) => {
                        buf.push(1);
                        buf.extend_from_slice(&(iota as u32).to_le_bytes());
                    }
                    None => buf.push(0),
                }
                buf.extend_from_slice(&term.constant.lo.to_le_bytes());
                buf.extend_from_slice(&term.constant.hi.to_le_bytes());
                match term.row {
                    Some(t) => {
                        buf.push(1);
                        for x in [t.offset, t.div, t.modulo] {
                            buf.extend_from_slice(&x.to_le_bytes());
                        }
                    }
                    None => buf.push(0),
                }
            }
        }
    }

    /// The cell-slots the ports read, ascending — the gather order.
    fn words(&self) -> Vec<usize> {
        let mut words: Vec<usize> = self
            .ports
            .iter()
            .flat_map(|p| p.terms.iter().filter_map(|t| t.word))
            .collect();
        words.sort_unstable();
        words.dedup();
        words
    }

    /// Ports per side, in declaration order.
    fn side(&self, dir: BusDirection) -> Vec<&BusPort> {
        self.ports.iter().filter(|p| p.dir == dir).collect()
    }

    /// Port-group bits: both sides share a `2^c` grid of port groups.
    fn c_bits(&self) -> usize {
        let most = self
            .side(BusDirection::Send)
            .len()
            .max(self.side(BusDirection::Recv).len())
            .max(1);
        most.next_power_of_two().trailing_zeros() as usize
    }
}

/// The bus's challenges: `β` and one `γ_k` per coordinate.
struct Challenges {
    beta: F128,
    gamma: Vec<F128>,
}

fn live_entries(bus: &Bus, circuit: &Circuit) -> usize {
    let rows = 1usize << circuit.cells().nu();
    bus.ports
        .iter()
        .map(|p| circuit.counts()[p.ty].min(rows))
        .sum()
}

/// Prove the bus balances on the committed buffer `packed` (the same padded
/// union buffer [`super::prove_wiring`] reads). Returns the proof and one
/// packed-direct claim per committed word read, for the opening.
pub(crate) fn prove_bus<C: Challenger>(
    circuit: &Circuit,
    bus: &Bus,
    packed: &[F128],
    grinding: BatchedGrinding,
    ch: &mut C,
) -> (BusProof, Vec<PackedDirectClaim>) {
    let cells = circuit.cells();
    let nu = cells.nu();
    let rows = 1usize << nu;
    ch.observe_label(DOMAIN_BUS);
    let bits = grinding.fingerprint_bits_for(live_entries(bus, circuit));
    let mut fingerprint_nonce = Vec::new();
    let beta = if bits != 0 {
        let (nonce, beta) = ch.grind_pow_and_sample_f128(bits);
        fingerprint_nonce.push(nonce);
        beta
    } else {
        ch.sample_f128()
    };
    let gamma: Vec<F128> = (0..bus.width).map(|_| ch.sample_f128()).collect();
    let challenges = Challenges { beta, gamma };

    let groups = 1usize << bus.c_bits();
    let leaves = |dir: BusDirection| -> ProductLeaves {
        let ports = bus.side(dir);
        let mut buf = take_f128(groups * rows);
        let lens: Vec<usize> = (0..groups)
            .map(|g| ports.get(g).map_or(0, |p| circuit.counts()[p.ty].min(rows)))
            .collect();
        // Parallel over ports and, within a port, chunks of rows.
        const CHUNK: usize = 1 << 13;
        buf[..groups * rows]
            .par_chunks_mut(rows)
            .zip(ports.par_iter())
            .zip(lens.par_iter())
            .for_each(|((group, &port), &live)| {
                group[..live]
                    .par_chunks_mut(CHUNK)
                    .enumerate()
                    .for_each(|(c, out)| {
                        for (i, leaf) in out.iter_mut().enumerate() {
                            *leaf = fingerprint(port, c * CHUNK + i, &challenges, |iota, j| {
                                packed[cells.gate_word_addr(iota, j)]
                            });
                        }
                    });
            });
        ProductLeaves { buf, lens, rows }
    };
    let lhs = leaves(BusDirection::Send);
    let rhs = leaves(BusDirection::Recv);
    let (gkr, claim) = prove_leaves(lhs, rhs, grinding, ch);

    let rho_row = &claim.rho[..nu];
    let eq_row = build_eq(rho_row);
    let words = bus.words();
    let gather: Vec<F128> = words
        .par_iter()
        .map(|&iota| {
            let CellSlot::Gate { ty, .. } = cells.slots()[iota] else {
                unreachable!("validated: bus words are gate cell-slots")
            };
            let live = circuit.counts()[ty].min(rows);
            let base = cells.gate_word_addr(iota, 0);
            eq_row[..live]
                .iter()
                .zip(&packed[base..base + live])
                .fold(F128::ZERO, |acc, (&e, &w)| acc + e * w)
        })
        .collect();
    let claims = words
        .iter()
        .zip(&gather)
        .map(|(&iota, &value)| {
            let point = cells.gate_claim_point(iota, rho_row);
            PackedDirectClaim {
                value,
                eq_ind: DirectEqInd::EqPoint(point.clone()),
                point,
            }
        })
        .collect();
    (
        BusProof {
            gkr,
            gather,
            fingerprint_nonce,
        },
        claims,
    )
}

/// Replay the bus and check both products' leaf claims against the gather
/// values. Returns the gather claims `(point, value)` in proof order.
pub(crate) fn verify_bus<C: Challenger>(
    circuit: &Circuit,
    bus: &Bus,
    proof: &BusProof,
    grinding: BatchedGrinding,
    ch: &mut C,
) -> Result<Vec<(Vec<F128>, F128)>, WiringError> {
    let cells = circuit.cells();
    let nu = cells.nu();
    let rows = 1usize << nu;
    let words = bus.words();
    if proof.gather.len() != words.len() {
        return Err(WiringError::MalformedProof);
    }
    ch.observe_label(DOMAIN_BUS);
    let bits = grinding.fingerprint_bits_for(live_entries(bus, circuit));
    let beta = if bits != 0 {
        let [nonce] = proof.fingerprint_nonce[..] else {
            return Err(WiringError::BusGkr(
                crate::product_gkr::ProductGkrError::InvalidGrinding,
            ));
        };
        ch.verify_pow_and_sample_f128(nonce, bits)
            .ok_or(WiringError::BusGkr(
                crate::product_gkr::ProductGkrError::InvalidGrinding,
            ))?
    } else {
        if !proof.fingerprint_nonce.is_empty() {
            return Err(WiringError::MalformedProof);
        }
        ch.sample_f128()
    };
    let gamma: Vec<F128> = (0..bus.width).map(|_| ch.sample_f128()).collect();

    let c = bus.c_bits();
    let claim = verify_leaves(nu + c, &proof.gkr, grinding, ch).map_err(WiringError::BusGkr)?;
    let (rho_row, rho_port) = claim.rho.split_at(nu);
    let eq_row = build_eq(rho_row);
    let eq_port = build_eq(rho_port);
    let gathered = |iota: usize| proof.gather[words.binary_search(&iota).expect("a bus word")];

    // Each side: 1 + Σ_p eq(ρ_port, p)·[(β + 1)·E_p + Σ_k γ_k·(coordinate MLE)].
    let side = |dir: BusDirection| -> F128 {
        let mut acc = F128::ONE;
        for (g, port) in bus.side(dir).into_iter().enumerate() {
            let live = circuit.counts()[port.ty].min(rows);
            let mass = LiveMask::eq_prefix_sum(rho_row, live);
            let mut inner = (beta + F128::ONE) * mass;
            for (term, &gk) in port.terms.iter().zip(&gamma) {
                let mut value = term.constant * mass;
                if let Some(iota) = term.word {
                    value += gathered(iota);
                }
                if let Some(tag) = term.row {
                    value += eq_row[..live]
                        .iter()
                        .enumerate()
                        .fold(F128::ZERO, |a, (j, &e)| a + e * tag.value(j));
                }
                inner += gk * value;
            }
            acc += eq_port[g] * inner;
        }
        acc
    };
    if side(BusDirection::Send) != claim.lhs || side(BusDirection::Recv) != claim.rhs {
        return Err(WiringError::BusRecombination);
    }
    Ok(words
        .iter()
        .zip(&proof.gather)
        .map(|(&iota, &value)| (cells.gate_claim_point(iota, rho_row), value))
        .collect())
}

/// Row `j`'s leaf in `port`: `β + Σ_k γ_k·t_k`, reading committed words
/// through `word(cell_slot, row)`.
#[inline]
fn fingerprint(
    port: &BusPort,
    j: usize,
    ch: &Challenges,
    word: impl Fn(usize, usize) -> F128,
) -> F128 {
    let mut acc = ch.beta;
    for (term, &g) in port.terms.iter().zip(&ch.gamma) {
        let mut t = term.constant;
        if let Some(iota) = term.word {
            t += word(iota, j);
        }
        if let Some(tag) = term.row {
            t += tag.value(j);
        }
        acc += g * t;
    }
    acc
}

/// The tuple of `port`'s row `j` — what the fingerprint encodes. For
/// witness-side balance checks.
pub fn tuple(port: &BusPort, j: usize, word: impl Fn(usize, usize) -> F128) -> Vec<F128> {
    port.terms
        .iter()
        .map(|term| {
            let mut t = term.constant;
            if let Some(iota) = term.word {
                t += word(iota, j);
            }
            if let Some(tag) = term.row {
                t += tag.value(j);
            }
            t
        })
        .collect()
}
