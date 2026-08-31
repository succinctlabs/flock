//! The circuit builder driving a **BLAKE3 chunk chain** end to end.
//!
//! This is the Fiat–Shamir chain's core structure in miniature: 16 BLAKE3
//! compressions whose chaining values thread row to row, `CHUNK_START` on the
//! first and `CHUNK_END` on the last, counter fixed — i.e. exactly one 1 KiB
//! BLAKE3 chunk. What the FS chain adds on top is the byte-packing glue that
//! places transcript bytes into `m` at arbitrary offsets; the chaining, the
//! flag pinning and the row layout are all here.
//!
//! It exercises, together and against a real prove/verify:
//!
//! - [`CircuitBuilder`] on a **boolean** slot (the element chain unit test
//!   covers the other class),
//! - `blake3::io_schema()`, including the packed `counter|block_len|flags`
//!   word being *wired to public cells* — which is what pins the flags per row
//!   position and lets one BLAKE3 table serve every compression flavour,
//! - `BuiltCircuit::rows::<G>()`, the read-back that hands a boolean slot's
//!   `&[Compression]` to `generate_witness_batch_major_partial`.
//!
//! BLAKE3 rather than SHA-256 on purpose: it is the settled hash for this
//! work, so validating the SHA path would validate one we do not use.

use bincode::serialize;
use blake3::Compression;
use blake3::build_block_r1cs;
use blake3::generate_witness_batch_major_partial;
use blake3::io_schema;
use flock_core::circuit::builder::{CircuitBuilder, GateType, SlotWitness, Wire};
use flock_core::field::F128;
use flock_core::pcs::PcsParams;
use flock_core::pcs::ligerito::LigeritoProfile;
use flock_hash::{HashKind, blake3_compress};
use flock_prover::challenger::FsChallenger;
use flock_prover::prover::{self, UnionSlotProverInput};
use flock_prover::r1cs_hashes::blake3;
use flock_prover::r1cs_hashes::fs_chain::IV as FS_CHAIN_IV;
use flock_prover::schedule::TableType;
use flock_prover::union::UnionInstance;
use flock_prover::verifier;
use prover::prove_fast_ligerito_union_circuit;
use prover::prove_fast_ligerito_union_mixed_class;
use std::array::from_fn;
use std::iter::once;
use std::time::Instant;
use verifier::verify_ligerito_union_circuit;
use verifier::verify_ligerito_union_mixed_class;
use zerocheck::prove_with_label;
use zerocheck::verify_with_label;

use flock_core::challenger::Challenger as _;
use flock_core::element_r1cs::{ElementTableBuilder, ElementTableType, zerocheck};
use flock_core::test_rng::Rng;
use flock_core::transcript_record::{RecordingChallenger, StreamWord, TranscriptOp};
use flock_prover::prover::UnionElementSlotInput;
use flock_prover::r1cs_hashes::fs_chain::{CvSource, FsChain};
use flock_prover::schedule::IoWord;
use flock_prover::schedule::Registry;
use std::sync::Arc;
const DOMAIN: &[u8] = b"flock-circuit-builder-v0";

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;

/// BLAKE3's IV, the chaining value a chunk starts from.
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

/// Four `u32`s into one committed 128-bit word: `lo` holds bits `[0,64)`, so
/// words 0,1 land in `lo` and 2,3 in `hi`.
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

fn unpack8(a: F128, b: F128) -> [u32; 8] {
    let (x, y) = (unpack4(a), unpack4(b));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}

/// The params word: `counter_lo | counter_hi | block_len | flags`, in that bit
/// order — so `lo` IS the 64-bit counter and `hi` carries `block_len` low,
/// `flags` high.
fn pack_params(counter: u64, block_len: u32, flags: u32) -> F128 {
    F128::new(counter, block_len as u64 | ((flags as u64) << 32))
}

fn unpack_params(v: F128) -> (u64, u32, u32) {
    (v.lo, v.hi as u32, (v.hi >> 32) as u32)
}

/// One BLAKE3 compression as a circuit gate.
struct Blake3Gate {
    nu: usize,
}

impl GateType for Blake3Gate {
    type Row = Compression;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&build_block_r1cs(self.nu)).with_io_schema(io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let (o, row) = {
            // Schema In-order: cv0, cv1, m0..m3, params.
            let cv = unpack8(inputs[0], inputs[1]);
            let mut m = [0u32; 16];
            for i in 0..4 {
                m[4 * i..4 * i + 4].copy_from_slice(&unpack4(inputs[2 + i]));
            }
            let (counter, block_len, flags) = unpack_params(inputs[6]);

            let out = blake3_compress(&cv, &m, counter, block_len, flags);
            let out_lo: [u32; 8] = out[0..8].try_into().unwrap();
            let out_hi: [u32; 8] = out[8..16].try_into().unwrap();
            let (lo, hi) = (pack8(&out_lo), pack8(&out_hi));

            (
                vec![lo[0], lo[1], hi[0], hi[1]],
                (cv, m, counter, block_len, flags),
            )
        };
        outputs.extend_from_slice(&o);
        row
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        // Boolean slots are bit-packed by `generate_witness_batch_major_partial`
        // in this crate, above the one the builder lives in.
        SlotWitness::DeferredToRows
    }
}

/// One BLAKE3 chunk (16 chained blocks) as a circuit: the IV and every message
/// block are public, the chunk's chaining value out is public, and every
/// intermediate CV is wired row to row.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn blake3_chunk_chain_through_the_builder() {
    let nu = 8usize; // BLAKE3 kappa = 14 ⇒ M = 22, the Ligerito floor
    let n_blocks = 16usize; // one 1 KiB chunk
    let mut rng = Rng(0xB1A3_0001);

    let messages: Vec<[u32; 16]> = (0..n_blocks).map(|_| from_fn(|_| rng.next_u32())).collect();

    let mut b = CircuitBuilder::new(nu);
    let g = b.slot(Blake3Gate { nu });

    // The chunk's starting chaining value is public.
    let iv = pack8(&IV);
    let mut cv: [Wire; 2] = [b.public_value(iv[0]), b.public_value(iv[1])];

    for (i, m) in messages.iter().enumerate() {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i == n_blocks - 1 {
            flags |= CHUNK_END;
        }
        // Message words and the params word are public: pinning params per row
        // is what fixes the flags and the counter at that position.
        let m_w: Vec<Wire> = (0..4)
            .map(|j| b.public_value(pack4(m[4 * j..4 * j + 4].try_into().unwrap())))
            .collect();
        let params = b.public_value(pack_params(0, 64, flags));

        let outs = b.gate(g, &[cv[0], cv[1], m_w[0], m_w[1], m_w[2], m_w[3], params]);
        // Schema Out-order: out_lo0, out_lo1, out_hi0, out_hi1. Chaining takes
        // out_lo; out_hi is unwired here (σ-fixed) — it only matters at a root.
        cv = [outs[0], outs[1]];
    }

    // The chunk's output chaining value is the circuit's public result.
    b.publish(cv[0]);
    b.publish(cv[1]);

    let built = b.finish().expect("builder produces a valid circuit");
    assert_eq!(built.shape.counts, vec![n_blocks]);

    // The builder's rows must reproduce a plain native BLAKE3 chunk.
    let rows = built.rows::<Blake3Gate>(g);
    assert_eq!(rows.len(), n_blocks);
    let mut want_cv = IV;
    for (i, m) in messages.iter().enumerate() {
        let mut flags = 0u32;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i == n_blocks - 1 {
            flags |= CHUNK_END;
        }
        assert_eq!(rows[i], (want_cv, *m, 0u64, 64u32, flags), "row {i}");
        let out = blake3_compress(&want_cv, m, 0, 64, flags);
        want_cv = out[0..8].try_into().unwrap();
    }
    // ...and the published result is that chunk's chaining value.
    let published = &built.witness.public[built.witness.public.len() - 2..];
    assert_eq!(unpack8(published[0], published[1]), want_cv);

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
    let r1cs = build_block_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();

    let mut ch = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(rows, nu),
            lc,
        )],
        Vec::new(),
        &mut ch,
    );

    let mut ch = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &[lc],
        &commitment,
        &proof,
        &pcs_params,
        &mut ch,
    )
    .expect("a builder-produced BLAKE3 chunk chain verifies");

    // A wrong claimed chunk output breaks the last wire equality — the wiring
    // is doing real work, not just decorating a satisfiable trace.
    let mut bad = built.witness.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(
        verify_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            &bad,
            &[lc],
            &commitment,
            &proof,
            &pcs_params,
            &mut ch,
        )
        .is_err(),
        "a tampered public output must be rejected"
    );
}

/// **MVP-1**: the Fiat–Shamir chain as a circuit — the challenges are DERIVED,
/// not asserted.
///
/// Given a public transcript, the circuit proves that a stated set of
/// challenges is the correct BLAKE3 Fiat–Shamir derivation of it. That is the
/// piece with no fallback: everything else a recursive verifier does checks an
/// arithmetic relation a circuit can state directly, but if the challenge words
/// were free witness a prover would choose challenges that make a false inner
/// proof pass, and every other constraint would still be satisfied.
///
/// The load-bearing wiring is the **derived challenge**. Squeezed output is
/// never re-absorbed into the transcript; challenge `k` IS the `out_lo` of
/// the ROOT row that finalizes it, and downstream consumers take that output
/// wire directly — a pure copy, which is what the 16-byte-aligned framing
/// bought. Take the challenge as a public constant instead and the circuit
/// asserts the challenges rather than deriving them, which is the entire
/// content of Fiat–Shamir.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn fs_chain_circuit_derives_the_challenges() {
    const D: &[u8] = b"flock-fs-chain-mvp";
    let nu = 8usize; // BLAKE3 kappa = 14 ⇒ M = 22; 256 rows of capacity

    // ---- drive a real challenger, capturing values and challenges ----
    let mut rng = Rng(0xF5C4_0001);
    let mut f = || F128::new(rng.next_u32() as u64, rng.next_u32() as u64);
    // A slice long enough to cross the 1 KiB chunk boundary, so the parent
    // tree and a non-empty chunk stack are actually exercised.
    let slice: Vec<F128> = (0..70).map(|_| f()).collect();
    let scalars: [F128; 3] = [f(), f(), f()];

    let mut ch = RecordingChallenger::new(FsChallenger::with_hash(D, HashKind::Blake3));
    ch.observe_label(b"mvp-phase");
    ch.observe_f128(scalars[0]);
    ch.observe_f128_slice(&slice);
    let c0 = ch.sample_f128();
    ch.observe_f128(scalars[1]);
    let c1 = ch.sample_f128();
    ch.observe_f128(scalars[2]);
    let c2 = ch.sample_f128();
    let shape = ch.shape();

    let values: Vec<F128> = once(scalars[0])
        .chain(slice.iter().copied())
        .chain([scalars[1], scalars[2]])
        .collect();
    let challenges = [c0, c1, c2];

    // ---- resolve the stream, and replay it through the chain ----
    let stream = shape.stream_words(D);
    let words = &stream.words;
    let resolve = |w: &StreamWord| match *w {
        StreamWord::Const(c) => c,
        StreamWord::Value(i) => values[i],
        StreamWord::Bytes { .. } => unreachable!("this script observes no raw bytes"),
    };
    // Squeezed output is not absorbed, so nothing in the stream marks a
    // squeeze — `finalize_after` says how many words precede each one.
    let mut chain = FsChain::new();
    let mut at = 0usize;
    for (k, &upto) in stream.finalize_after.iter().enumerate() {
        let mut pending: Vec<u8> = Vec::new();
        for w in &words[at..upto] {
            let v = resolve(w);
            pending.extend_from_slice(&v.lo.to_le_bytes());
            pending.extend_from_slice(&v.hi.to_le_bytes());
        }
        chain.absorb(&pending);
        at = upto;
        let out = chain.finalize(16);
        assert_eq!(
            F128::new(
                u64::from_le_bytes(out[..8].try_into().unwrap()),
                u64::from_le_bytes(out[8..].try_into().unwrap())
            ),
            challenges[k],
            "chain reproduced a different challenge than the challenger"
        );
    }
    let mut tail: Vec<u8> = Vec::new();
    for w in &words[at..] {
        let v = resolve(w);
        tail.extend_from_slice(&v.lo.to_le_bytes());
        tail.extend_from_slice(&v.hi.to_le_bytes());
    }
    chain.absorb(&tail);
    let trace = chain.finish();

    // ---- build the circuit ----
    let mut b = CircuitBuilder::new(nu);
    let g = b.slot(Blake3Gate { nu });
    let iv_w = pack8(&FS_CHAIN_IV);
    let iv = [b.public_value(iv_w[0]), b.public_value(iv_w[1])];

    // Stream words become public cells, memoized in `word_wire` so every
    // consumer of a stream word shares one wire. (Squeezed output never
    // appears in the stream; challenges are read off the producing rows'
    // output wires instead — see `rho_w` below.)
    let mut word_wire: Vec<Option<[Wire; 1]>> = vec![None; words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());

    for (i, row) in trace.rows.iter().enumerate() {
        let (cv, m, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = b.public_value(pack_params(counter, blen, flags));

        let (cv_in, m_in): ([Wire; 2], [Wire; 4]) = match link.right {
            // PARENT: cv is the IV; the message is left‖right chaining values.
            Some(right) => {
                let l = &outs[match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::RowHi(r) => r,
                    CvSource::Iv => unreachable!("a parent's left input is a row"),
                }];
                let r = &outs[right];
                (iv, [l[0], l[1], r[0], r[1]])
            }
            // A transcript block: cv chains, the message is stream words.
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let base = trace.block_offsets[i].expect("a stream block has an offset") / 16;
                // `block_len` bounds how much of this block is real stream. A
                // finalize's pending block is usually SHORT — its remaining
                // words are BLAKE3's zero padding, not the next transcript
                // bytes, and in particular not the challenge this very finalize
                // is about to produce.
                let real_words = (blen as usize) / 16;
                let mut m_in = [iv[0]; 4];
                for j in 0..4 {
                    let wi = base + j;
                    let w = match words.get(wi).filter(|_| j < real_words) {
                        // Zero padding past `block_len`.
                        None => b.public_value(F128::ZERO),
                        Some(sw) => match word_wire[wi] {
                            Some([w]) => w,
                            None => {
                                let w = b.public_value(resolve(sw));
                                word_wire[wi] = Some([w]);
                                w
                            }
                        },
                    };
                    m_in[j] = w;
                }
                let _ = (cv, m);
                (cv_in, m_in)
            }
        };

        outs.push(b.gate(
            g,
            &[
                cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
            ],
        ));
    }

    // The derived challenges are the circuit's public output.
    for k in 0..challenges.len() {
        b.publish(outs[trace.squeezes[k][0]][0]);
    }

    let built = b.finish().expect("builder produces a valid circuit");
    assert_eq!(built.shape.counts, vec![trace.rows.len()]);
    let pub_out = &built.witness.public[built.witness.public.len() - challenges.len()..];
    assert_eq!(
        pub_out, &challenges,
        "published challenges must be the real ones"
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
    let r1cs = build_block_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();
    let rows = built.rows::<Blake3Gate>(g);

    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &pcs_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(rows, nu),
            lc,
        )],
        Vec::new(),
        &mut c,
    );
    let mut c = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &[lc],
        &commitment,
        &proof,
        &pcs_params,
        &mut c,
    )
    .expect("the FS chain circuit verifies");

    // A wrong claimed challenge breaks the wiring: it is derived, not asserted.
    let mut bad = built.witness.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut c = FsChallenger::new(DOMAIN);
    assert!(
        verify_ligerito_union_circuit(
            &union,
            &built.shape.circuit,
            &bad,
            &[lc],
            &commitment,
            &proof,
            &pcs_params,
            &mut c,
        )
        .is_err(),
        "a tampered challenge must be rejected"
    );

    println!(
        "FS chain circuit: {} rows, {} public words, {} challenges derived",
        trace.rows.len(),
        built.witness.public.len(),
        challenges.len()
    );
}

/// **The MVP proof, on a REAL transcript**: record an actual element-only
/// Flock proof, then prove in-circuit that its Fiat–Shamir challenges are the
/// correct BLAKE3 derivation of its transcript.
///
/// The scripted test above shows the mechanism; this shows it at the shape and
/// scale the recursive verifier will actually meet — every op kind the protocol
/// emits, a multi-chunk transcript, and finalizes deep enough that the chunk
/// stack has real depth.
#[test]
#[ignore] // Heavy — run with `-- --ignored`.
fn mvp_fs_chain_of_a_real_proof() {
    const INNER: &[u8] = b"flock-union-element-v0";
    let (inner_nu, kappa, count) = (12usize, 3usize, 1usize << 12);

    // ---- an ordinary element-only proof, recorded ----
    let (w0, w1) = (F128::new(7, 0), F128::new(0, 3));
    let ty: Arc<ElementTableType> = {
        let mut b = ElementTableBuilder::new(kappa);
        b.free_wire(0)
            .free_wire(1)
            .mult(2, 0, 1)
            .linear(3, &[(0, w0), (1, w1)]);
        Arc::new(b.build().expect("gate block"))
    };
    let registry = Registry::new(vec![TableType::element(ty.clone())], inner_nu);
    let union = UnionInstance::new(&registry, vec![count]);
    let inner_params = PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: union.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let mut rng = Rng(0x11FE_0001);
    let z = {
        let at = |c: usize, j: usize| (c << inner_nu) + j;
        let mut z = vec![F128::ZERO; ty.width() << inner_nu];
        for j in 0..count {
            let (a, b) = (
                F128::new(rng.next_u32() as u64, rng.next_u32() as u64),
                F128::new(rng.next_u32() as u64, rng.next_u32() as u64),
            );
            z[at(0, j)] = a;
            z[at(1, j)] = b;
            z[at(2, j)] = a * b;
            z[at(3, j)] = w0 * a + w1 * b;
        }
        z
    };
    // BLAKE3 transcript: the FS chain is a BLAKE3 circuit, and BLAKE3 is the
    // settled Merkle/FS hash for this work. `FsChallenger::new` defaults to
    // SHA-256, which would be a different chain entirely.
    let mut ch_p = FsChallenger::with_hash(INNER, HashKind::Blake3);
    let (inner_proof, inner_commit, _) = prove_fast_ligerito_union_mixed_class(
        &union,
        &inner_params,
        Vec::new(),
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&z)
        })],
        &mut ch_p,
    );

    // Record the VERIFIER's transcript — that is what a recursive verifier replays.
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(INNER, HashKind::Blake3));
    verify_ligerito_union_mixed_class(
        &union,
        &[],
        &inner_commit,
        &inner_proof,
        &inner_params,
        &mut rec,
    )
    .expect("inner proof verifies");
    let shape = rec.shape();
    let stream = shape.stream_words(INNER);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let challenges = rec.challenges().to_vec();

    // ---- replay it through the FS chain ----
    let mut chain = FsChain::new();
    let mut at = 0usize;
    let mut produced: Vec<F128> = Vec::new();
    // Every finalizing op needs its rows, but a PoW's state digest is the
    // grinding base, NOT a challenge — the circuit still computes it, and it
    // still binds, but it is not part of `challenges`.
    let fin_ops: Vec<&TranscriptOp> = shape.ops().iter().filter(|o| o.finalizes()).collect();
    for (i, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        let op = fin_ops[i];
        let out = chain.finalize(op.squeezed_bytes());
        if !matches!(op, TranscriptOp::Pow { .. }) {
            for c in out.chunks(16) {
                produced.push(F128::new(
                    u64::from_le_bytes(c[..8].try_into().unwrap()),
                    u64::from_le_bytes(c[8..].try_into().unwrap()),
                ));
            }
        }
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();
    let first = produced.iter().zip(&challenges).position(|(a, b)| a != b);
    assert_eq!(
        (produced.len(), first),
        (challenges.len(), None),
        "chain vs verifier: {} produced, {} expected, first mismatch at {:?}",
        produced.len(),
        challenges.len(),
        first
    );

    // ---- the circuit ----
    let nu = (trace.rows.len().next_power_of_two().trailing_zeros() as usize).max(1);
    let mut b = CircuitBuilder::new(nu);
    let g = b.slot(Blake3Gate { nu });
    let iv_w = pack8(&FS_CHAIN_IV);
    let iv = [b.public_value(iv_w[0]), b.public_value(iv_w[1])];
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut inputs: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());

    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = b.public_value(pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            // An XOF output block: same cv and message as the ROOT row, only
            // the counter differs, so wire the identical sources.
            let src = inputs[root];
            let g_in = [src[0], src[1], src[2], src[3], src[4], src[5], params];
            inputs.push(g_in);
            outs.push(b.gate(g, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::RowHi(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        b.public_value(F128::ZERO)
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                let v = F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                );
                                let w = b.public_value(v);
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
        inputs.push(g_in);
        outs.push(b.gate(g, &g_in));
    }
    for s in &trace.squeezes {
        b.publish(outs[s[0]][0]);
    }
    let built = b.finish().expect("valid circuit");

    let outer = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let outer_params = PcsParams {
        m: outer.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: outer.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let r1cs = build_block_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();
    let rows = built.rows::<Blake3Gate>(g);

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &outer,
        &built.shape.circuit,
        &built.witness.public,
        &outer_params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(rows, nu),
            lc,
        )],
        Vec::new(),
        &mut c,
    );
    let prove_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &outer,
        &built.shape.circuit,
        &built.witness.public,
        &[lc],
        &commitment,
        &proof,
        &outer_params,
        &mut c,
    )
    .expect("the MVP proof verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    println!(
        "\n=== MVP: FS chain of a real element-only proof ===\n  \
         inner: {} transcript ops, {} absorbed bytes, {} challenges\n  \
         outer: {} BLAKE3 rows (nu={nu}, M={}), {} public words, {} proof bytes\n  \
         prove {prove_ms:.0} ms | verify {verify_ms:.1} ms",
        shape.len(),
        bytes.len(),
        challenges.len(),
        trace.rows.len(),
        outer.m_total(),
        built.witness.public.len(),
        serialize(&proof).unwrap().len(),
    );
}

/// The element arithmetic gate: four inputs, and both a **fused product** and
/// a four-way sum out.
///
/// ```text
///   prod = (a0 + a1) · (b0 + b1)
///   sum  =  a0 + a1  +  b0 + b1
/// ```
///
/// The fusion is the point. In the element class an addition is NOT free —
/// every committed column is the output of exactly one R1CS row, `linear` ones
/// included — but `A_0` and `B_0` are matrix *rows*, so a sum on either side of
/// a product rides the multiplication's own row instead of costing a column.
/// A naive one-op-per-row gate spends 12 rows on a sumcheck round; this spends
/// 8, and the saving is per round of the real replay.
///
/// Every operation the round needs is one call: `x·y` is `(x+0)·(y+0)`,
/// `(x+y)·z` is direct, and `x+y+z` is the sum output.
struct ArithGate {
    ty: Arc<ElementTableType>,
}

impl ArithGate {
    fn new() -> Self {
        let one = F128::ONE;
        let mut b = ElementTableBuilder::new(3); // 8 columns; 6,7 self-pinned zero
        b.free_wire(0)
            .free_wire(1)
            .free_wire(2)
            .free_wire(3)
            .mult_lin(4, &[(0, one), (1, one)], &[(2, one), (3, one)])
            .linear(5, &[(0, one), (1, one), (2, one), (3, one)]);
        Self {
            ty: Arc::new(b.build().expect("arith block")),
        }
    }
}

impl GateType for ArithGate {
    type Row = [F128; 4];
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::element(self.ty.clone()).with_io_schema(vec![
            IoWord::input(0),
            IoWord::input(1),
            IoWord::input(2),
            IoWord::input(3),
            IoWord::output(4), // (a0+a1)·(b0+b1)
            IoWord::output(5), // a0+a1+b0+b1
        ])
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let (o, row) = {
            let r: [F128; 4] = [inputs[0], inputs[1], inputs[2], inputs[3]];
            let prod = (r[0] + r[1]) * (r[2] + r[3]);
            let sum = r[0] + r[1] + r[2] + r[3];
            (vec![prod, sum], r)
        };
        outputs.extend_from_slice(&o);
        row
    }

    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
        let at = |c: usize, j: usize| (c << nu) + j;
        let mut z = vec![F128::ZERO; self.ty.width() << nu];
        for (j, r) in rows.iter().enumerate() {
            for (c, &v) in r.iter().enumerate() {
                z[at(c, j)] = v;
            }
            z[at(4, j)] = (r[0] + r[1]) * (r[2] + r[3]);
            z[at(5, j)] = r[0] + r[1] + r[2] + r[3];
        }
        SlotWitness::Element(z)
    }
}

/// **MVP-2**: a zerocheck sumcheck round replayed in-circuit, against a
/// challenge the FS chain *derived* rather than one handed in.
///
/// This is the handoff's vertical slice — "FS replay plus one sumcheck-round
/// replay against a pure-element inner proof". It is the first circuit where
/// the two classes share a union: BLAKE3 rows deriving the challenge, element
/// rows consuming it, joined by copy constraints across the class boundary.
///
/// The round is `zerocheck.rs`'s, verbatim:
/// ```text
///   g0       = (c_running + r_eq·g1) · (1 + r_eq)⁻¹
///   c_next   = g0·(1+ρ) + g1·ρ + g∞·ρ·(1+ρ)
/// ```
/// The inverse needs no gate: witness `d⁻¹`, emit `d·d⁻¹`, and connect that
/// product to a public cell holding 1.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp2_sumcheck_round_consumes_a_derived_challenge() {
    const D: &[u8] = b"flock-mvp2";
    let nu = 8usize;

    // ---- the transcript: observe a round message, squeeze the fold challenge ----
    let mut rng = Rng(0x27C4_0001);
    let mut f = || F128::new(rng.next_u32() as u64, rng.next_u32() as u64);
    let (c_running, r_eq, g1, g_inf) = (f(), f(), f(), f());

    let mut ch = RecordingChallenger::new(FsChallenger::with_hash(D, HashKind::Blake3));
    ch.observe_label(b"flock-zerocheck-v0");
    ch.observe_f128(g1);
    ch.observe_f128(g_inf);
    let rho = ch.sample_f128(); // THE fold challenge, derived below in-circuit
    let shape = ch.shape();
    let stream = shape.stream_words(D);
    let bytes = stream.to_bytes(ch.values(), ch.payloads());

    // Native round update — what the circuit must reproduce.
    let g0 = (c_running + r_eq * g1) * (F128::ONE + r_eq).inv();
    let one_plus_rho = F128::ONE + rho;
    let c_next = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;

    // ---- FS chain over the transcript ----
    let mut chain = FsChain::new();
    let upto = stream.finalize_after[0];
    chain.absorb(&bytes[..upto * 16]);
    let out = chain.finalize(16);
    assert_eq!(
        F128::new(
            u64::from_le_bytes(out[..8].try_into().unwrap()),
            u64::from_le_bytes(out[8..].try_into().unwrap())
        ),
        rho
    );
    chain.absorb(&bytes[upto * 16..]);
    let trace = chain.finish();

    // ---- circuit: BLAKE3 slot derives rho, element slot consumes it ----
    let mut b = CircuitBuilder::new(nu);
    let hash = b.slot(Blake3Gate { nu });
    let arith = b.slot(ArithGate::new());

    let one = b.public_value(F128::ONE);
    let iv_w = pack8(&FS_CHAIN_IV);
    let iv = [b.public_value(iv_w[0]), b.public_value(iv_w[1])];
    let mut outs: Vec<Vec<Wire>> = Vec::new();
    let mut inputs: Vec<[Wire; 7]> = Vec::new();
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];

    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = b.public_value(pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = inputs[root];
            let gi = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            inputs.push(gi);
            outs.push(b.gate(hash, &gi));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::RowHi(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        b.public_value(F128::ZERO)
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                let w = match stream.words[wi] {
                                    // The round message is WITNESS: the same
                                    // wire feeds the hash and the arithmetic,
                                    // and that shared wire is the binding.
                                    StreamWord::Value(k) => b.value(ch.values()[k]),
                                    other => b.public_value(match other {
                                        StreamWord::Const(c) => c,
                                        _ => unreachable!("no raw-byte ops here"),
                                    }),
                                };
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let gi = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        inputs.push(gi);
        outs.push(b.gate(hash, &gi));
    }

    // rho, derived — the ROOT row's out_lo low word.
    let rho_w = outs[trace.squeezes[0][0]][0];
    // g1 and g_inf are the SAME wires the hash consumed (stream words 2 and 3
    // of the observe ops); recover them by their stream index.
    let value_wire = |k: usize| -> Wire {
        let wi = stream
            .words
            .iter()
            .position(|w| matches!(*w, StreamWord::Value(i) if i == k))
            .expect("observed value is in the stream");
        word_wire[wi].expect("its word was wired")
    };
    let g1_w = value_wire(0);
    let ginf_w = value_wire(1);

    let zero = b.public_value(F128::ZERO);
    let c_run_w = b.public_value(c_running);
    let r_eq_w = b.public_value(r_eq);

    // Eight rows for the round. Each additive term rides the row of the
    // multiplication consuming it, so no addition costs a constraint of its
    // own except the final three-way sum.
    //   1. t1     = r_eq·g1
    let t1 = b.gate(arith, &[r_eq_w, zero, g1_w, zero])[0];
    //   2. dprod  = (1 + r_eq)·d⁻¹, pinned to 1
    let dinv = b.value((F128::ONE + r_eq).inv());
    let dprod = b.gate(arith, &[one, r_eq_w, dinv, zero])[0];
    // A pin cell used for NOTHING else: connecting to the shared `one` would
    // be cyclic, since `one` feeds this very gate.
    let one_pin = b.public_value(F128::ONE);
    b.connect(dprod, one_pin);
    //   3. g0     = (c_running + t1)·d⁻¹   ← the addition is free here
    let g0_w = b.gate(arith, &[c_run_w, t1, dinv, zero])[0];
    //   4. p1     = g0·(1 + ρ)             ← and here
    let p1 = b.gate(arith, &[g0_w, zero, one, rho_w])[0];
    //   5. p2     = g1·ρ
    let p2 = b.gate(arith, &[g1_w, zero, rho_w, zero])[0];
    //   6. p3     = g∞·ρ
    let p3 = b.gate(arith, &[ginf_w, zero, rho_w, zero])[0];
    //   7. p4     = p3·(1 + ρ)
    let p4 = b.gate(arith, &[p3, zero, one, rho_w])[0];
    //   8. c_next = p1 + p2 + p4           ← three-way sum in one row
    let c_next_w = b.gate(arith, &[p1, p2, p4, zero])[1];
    b.publish(c_next_w);

    let built = b.finish().expect("valid circuit");
    assert_eq!(
        *built.witness.public.last().unwrap(),
        c_next,
        "the circuit's round output must equal the native one"
    );

    // ---- prove / verify ----
    let outer = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let params = PcsParams {
        m: outer.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: outer.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let r1cs = build_block_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();
    let hash_rows = built.rows::<Blake3Gate>(hash);
    let el = match &built.witness.witnesses[built.registry_slot(arith)] {
        SlotWitness::Element(z) => z.clone(),
        _ => panic!("arith slot is element-class"),
    };

    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &outer,
        &built.shape.circuit,
        &built.witness.public,
        &params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(hash_rows, nu),
            lc,
        )],
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el)
        })],
        &mut c,
    );
    let mut c = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &outer,
        &built.shape.circuit,
        &built.witness.public,
        &[lc],
        &commitment,
        &proof,
        &params,
        &mut c,
    )
    .expect("MVP-2 verifies");

    // A wrong claimed round output is rejected: the challenge is derived and
    // the arithmetic consumes it, so the two halves cannot be decoupled.
    let mut bad = built.witness.public.clone();
    let last = bad.len() - 1;
    bad[last] += F128::ONE;
    let mut c = FsChallenger::new(DOMAIN);
    assert!(
        verify_ligerito_union_circuit(
            &outer,
            &built.shape.circuit,
            &bad,
            &[lc],
            &commitment,
            &proof,
            &params,
            &mut c,
        )
        .is_err(),
        "a tampered round output must be rejected"
    );

    println!(
        "\n=== MVP-2: FS chain + one sumcheck round ===\n  \
         {} BLAKE3 rows + {} element rows, M={}, {} public words",
        built.shape.counts[built.registry_slot(hash)],
        built.shape.counts[built.registry_slot(arith)],
        outer.m_total(),
        built.witness.public.len(),
    );
}

/// **The whole element zerocheck replayed in-circuit**, against challenges the
/// FS chain derives — a real proof, every round, and the final consistency
/// check.
///
/// A mirror of `element_r1cs::zerocheck::verify_with_label`, statement by
/// statement. That is a deliberate, recorded choice (wiring doc §"Mirroring the
/// verifier, for now"): at ~150 element rows a hand-written replay is hours and
/// prejudges nothing, and the decision gets revisited when the opening's ~47k
/// arrives.
///
/// What it proves: *there exist round messages such that hashing them in the
/// verifier's order yields these challenges, and the Convention-A chain run on
/// them reaches a running claim equal to `ea·eb + ec`* — the zerocheck's own
/// accept condition. The round messages are witness, and the same wires feed
/// both the hash and the arithmetic, so the prover cannot shop for challenges.
#[test]
#[ignore] // Heavier — run with `-- --ignored`.
fn mvp2b_full_element_zerocheck_replayed() {
    const D: &[u8] = b"flock-mvp2b";
    const LABEL: &[u8] = b"flock-element-union-zc-v0";
    let nu = 9usize;

    // ---- a real element zerocheck over a small satisfying witness ----
    let (kappa, n_log) = (2usize, 3usize); // 4 columns x 8 rows
    let m_words = kappa + n_log;
    let ety = {
        let mut b = ElementTableBuilder::new(kappa);
        b.free_wire(0).free_wire(1).mult(2, 0, 1);
        b.build().expect("mult block")
    };
    let mut rng = Rng(0x2B00_0001);
    let mut f = || F128::new(rng.next_u32() as u64, rng.next_u32() as u64);
    let z = {
        let at = |c: usize, j: usize| (c << n_log) + j;
        let mut z = vec![F128::ZERO; ety.width() << n_log];
        for j in 0..(1usize << n_log) {
            let (a, bb) = (f(), f());
            z[at(0, j)] = a;
            z[at(1, j)] = bb;
            z[at(2, j)] = a * bb;
        }
        z
    };
    assert!(ety.satisfies(&z, n_log, 1 << n_log));
    let (mut pa, mut pb) = (vec![F128::ZERO; z.len()], vec![F128::ZERO; z.len()]);
    ety.affine_products_into(&z, n_log, None, &mut pa, &mut pb);

    let mut ch_p = FsChallenger::with_hash(D, HashKind::Blake3);
    let (zc_proof, _) = prove_with_label(LABEL, pa, pb, &z, m_words, &mut ch_p);

    // ---- record the verifier's transcript for it ----
    let mut rec = RecordingChallenger::new(FsChallenger::with_hash(D, HashKind::Blake3));
    verify_with_label(LABEL, m_words, &zc_proof, &mut rec).expect("zerocheck verifies");

    // Phase 2 continues on the SAME transcript — the recorder is a challenger,
    // so the script just carries on. The round messages here are synthetic (a
    // standalone lincheck would need the union's comb and g vectors); what is
    // real is the round algebra and the transcript order.
    const LC_LABEL: &[u8] = b"flock-element-union-lc-v0";
    const LC_ROUNDS: usize = 3;
    rec.observe_label(LC_LABEL);
    let alpha_v = rec.sample_f128();
    let lc_msgs: Vec<(F128, F128)> = (0..LC_ROUNDS).map(|_| (f(), f())).collect();
    let mut lc_rho_v = Vec::with_capacity(LC_ROUNDS);
    for &(e1, einf) in &lc_msgs {
        rec.observe_f128(e1);
        rec.observe_f128(einf);
        lc_rho_v.push(rec.sample_f128());
    }
    let shape = rec.shape();
    let stream = shape.stream_words(D);
    let bytes = stream.to_bytes(rec.values(), rec.payloads());
    let challenges = rec.challenges().to_vec();
    assert_eq!(
        challenges.len(),
        m_words * 2 + 1 + LC_ROUNDS,
        "tau slice + a rho per zerocheck round + alpha + a rho per lincheck round"
    );

    // ---- FS chain over it ----
    let mut chain = FsChain::new();
    let mut at = 0usize;
    let fin: Vec<usize> = shape
        .ops()
        .iter()
        .filter(|o| o.finalizes())
        .map(|o| o.squeezed_bytes())
        .collect();
    for (k, &upto) in stream.finalize_after.iter().enumerate() {
        chain.absorb(&bytes[at * 16..upto * 16]);
        at = upto;
        chain.finalize(fin[k]);
    }
    chain.absorb(&bytes[at * 16..]);
    let trace = chain.finish();

    // ---- circuit ----
    let mut b = CircuitBuilder::new(nu);
    let hash = b.slot(Blake3Gate { nu });
    let arith = b.slot(ArithGate::new());
    let zero = b.public_value(F128::ZERO);
    let one = b.public_value(F128::ONE);
    let iv_w = pack8(&FS_CHAIN_IV);
    let iv = [b.public_value(iv_w[0]), b.public_value(iv_w[1])];

    let mut outs: Vec<Vec<Wire>> = Vec::new();
    let mut ins: Vec<[Wire; 7]> = Vec::new();
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    // Every observed value gets its wire up front. Most are consumed by the
    // hash as well, and that shared wire is the binding — but the transcript's
    // TAIL is absorbed after the last squeeze and never compressed, so no row
    // covers it. Here that is `ea`, `eb`, `ec`: in a standalone zerocheck they
    // bind nothing, and in the real protocol the lincheck's alpha binds them.
    for (wi, w) in stream.words.iter().enumerate() {
        if let StreamWord::Value(k) = *w {
            word_wire[wi] = Some(b.value(rec.values()[k]));
        }
    }
    for (i, row) in trace.rows.iter().enumerate() {
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = b.public_value(pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = ins[root];
            let gi = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            ins.push(gi);
            outs.push(b.gate(hash, &gi));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::RowHi(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                (iv, [outs[l][0], outs[l][1], outs[right][0], outs[right][1]])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = (blen as usize) / 16;
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        zero
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            None => {
                                let w = match stream.words[wi] {
                                    StreamWord::Const(c) => b.public_value(c),
                                    _ => unreachable!("values are pre-wired"),
                                };
                                word_wire[wi] = Some(w);
                                w
                            }
                        }
                    };
                }
                (cv_in, m)
            }
        };
        let gi = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        ins.push(gi);
        outs.push(b.gate(hash, &gi));
    }

    // Challenge k's wire: the ROOT row of finalize k gives its first 16 bytes.
    // The tau slice is one squeeze of `m_words` words, so its later words come
    // from the XOF rows that follow that root.
    let challenge_wire = |k: usize| -> Wire {
        // finalize 0 is the tau slice (m_words challenges), then one per round.
        if k < m_words {
            // A 64-byte XOF block IS the four schema outputs, in order:
            // out_lo0, out_lo1, out_hi0, out_hi1. So challenge k sits at
            // output k%4 of block k/4.
            outs[trace.squeezes[0][k / 4]][k % 4]
        } else {
            // Every later squeeze is a single word: finalize `1 + (k - m_words)`,
            // first output.
            outs[trace.squeezes[1 + (k - m_words)][0]][0]
        }
    };
    let value_wire = |k: usize| -> Wire {
        let wi = stream
            .words
            .iter()
            .position(|w| matches!(*w, StreamWord::Value(i) if i == k))
            .expect("observed value is in the stream");
        word_wire[wi].expect("wired")
    };

    // ---- the replay, mirroring zerocheck::verify_with_label ----
    let mut running = zero; // a zerocheck starts at target 0
    let mut running_v = F128::ZERO;
    for i in 0..m_words {
        let t = challenge_wire(i);
        let t_v = challenges[i];
        let (g1, g_inf) = (value_wire(2 * i), value_wire(2 * i + 1));
        let (g1_v, g_inf_v) = (zc_proof.rounds[i].0, zc_proof.rounds[i].1);
        let rho = challenge_wire(m_words + i);
        let rho_v = challenges[m_words + i];

        // g0 = (running + t·g1) · (1+t)⁻¹
        let t_g1 = b.gate(arith, &[t, zero, g1, zero])[0];
        let dinv_v = (F128::ONE + t_v).inv();
        let dinv = b.value(dinv_v);
        let dprod = b.gate(arith, &[one, t, dinv, zero])[0];
        let pin = b.public_value(F128::ONE);
        b.connect(dprod, pin);
        let g0 = b.gate(arith, &[running, t_g1, dinv, zero])[0];
        let g0_v = (running_v + t_v * g1_v) * dinv_v;

        // running = g0·(1+ρ) + g1·ρ + g∞·ρ·(1+ρ)
        let p1 = b.gate(arith, &[g0, zero, one, rho])[0];
        let p2 = b.gate(arith, &[g1, zero, rho, zero])[0];
        let p3 = b.gate(arith, &[g_inf, zero, rho, zero])[0];
        let p4 = b.gate(arith, &[p3, zero, one, rho])[0];
        running = b.gate(arith, &[p1, p2, p4, zero])[1];
        let opr = F128::ONE + rho_v;
        running_v = g0_v * opr + g1_v * rho_v + g_inf_v * rho_v * opr;
    }

    // Final consistency: running == ea·eb + ec.
    // Phase 2's target. The real `verify_deferred` first strips the affine
    // constants (`strip_constants`, an eq-table dot per slot) to get (va, vb);
    // that needs the union's slot layout, so this standalone mirror starts from
    // the zerocheck's own (ea, eb).
    let alpha_w = challenge_wire(m_words * 2);
    let ea = value_wire(2 * m_words);
    let eb = value_wire(2 * m_words + 1);
    let ec = value_wire(2 * m_words + 2);
    let mut lc_running = b.gate(arith, &[alpha_w, zero, eb, zero])[0]; // α·eb
    let mut lc_running_v = alpha_v * zc_proof.eb;
    lc_running = b.gate(arith, &[lc_running, ea, zero, zero])[1]; // + ea
    lc_running_v += zc_proof.ea;

    let eaeb = b.gate(arith, &[ea, zero, eb, zero])[0];
    let rhs = b.gate(arith, &[eaeb, ec, zero, zero])[1];
    // THE accept condition. Both sides are COMPUTED, so connecting them
    // directly would give the class two producers; in characteristic 2 the
    // equality is `running + rhs == 0`, pinned against a fresh public zero
    // (fresh because `zero` feeds these very gates, and reusing it is cyclic).
    let diff = b.gate(arith, &[running, rhs, zero, zero])[1];
    let zero_pin = b.public_value(F128::ZERO);
    b.connect(diff, zero_pin);

    // ---- Phase 2: the lincheck's column sumcheck ----
    //
    // `column_sumcheck_replay`'s round is
    //     e0 = running + e1;  c1 = e0 + e1 + einf;  running = einf·ρ² + c1·ρ + e0
    // and in characteristic 2 the two `e1` terms CANCEL, so `c1 = running +
    // einf`. That is what makes a lincheck round four gates against the
    // zerocheck's eight: no inversion, and the `c1` sum rides its product.
    for (i, &(e1_v, einf_v)) in lc_msgs.iter().enumerate() {
        let e0_v = lc_running_v + e1_v;
        assert_eq!(
            e0_v + e1_v + einf_v,
            lc_running_v + einf_v,
            "the char-2 cancellation this fusion relies on"
        );
        let rho_v = lc_rho_v[i];
        let rho = challenge_wire(m_words * 2 + 1 + i);
        let e1 = value_wire(2 * m_words + 3 + 2 * i);
        let einf = value_wire(2 * m_words + 3 + 2 * i + 1);

        //   1. ρ²          2. einf·ρ²      3. (running + einf)·ρ
        //   4. sum: einf·ρ² + c1·ρ + running + e1     (= …+ e0)
        let rho2 = b.gate(arith, &[rho, zero, rho, zero])[0];
        let p = b.gate(arith, &[einf, zero, rho2, zero])[0];
        let q = b.gate(arith, &[lc_running, einf, rho, zero])[0];
        lc_running = b.gate(arith, &[p, q, lc_running, e1])[1];
        lc_running_v = einf_v * rho_v * rho_v + (lc_running_v + einf_v) * rho_v + e0_v;
    }
    b.publish(lc_running);
    assert_eq!(
        running_v,
        zc_proof.ea * zc_proof.eb + zc_proof.ec,
        "native mirror must agree with the real verifier"
    );
    b.publish(running);

    let built = b.finish().expect("valid circuit");
    let outer = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let params = PcsParams {
        m: outer.dense_m(),
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: LigeritoProfile::Fast,
        num_lanes: outer.commit_lanes(6),
        merkle_hash: Default::default(),
    };
    let r1cs = build_block_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();
    let hrows = built.rows::<Blake3Gate>(hash);
    let el = match &built.witness.witnesses[built.registry_slot(arith)] {
        SlotWitness::Element(z) => z.clone(),
        _ => unreachable!(),
    };
    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    let (proof, commitment, _) = prove_fast_ligerito_union_circuit(
        &outer,
        &built.shape.circuit,
        &built.witness.public,
        &params,
        vec![UnionSlotProverInput::new(
            generate_witness_batch_major_partial(hrows, nu),
            lc,
        )],
        vec![UnionElementSlotInput::new(move |dst: &mut [F128]| {
            dst.copy_from_slice(&el)
        })],
        &mut c,
    );
    let prove_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let mut c = FsChallenger::new(DOMAIN);
    verify_ligerito_union_circuit(
        &outer,
        &built.shape.circuit,
        &built.witness.public,
        &[lc],
        &commitment,
        &proof,
        &params,
        &mut c,
    )
    .expect("the replayed zerocheck verifies");
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;

    println!(
        "\n=== MVP-2b: the whole element zerocheck, in-circuit ===\n  \
         zerocheck {m_words} rounds + lincheck {LC_ROUNDS} rounds | \
         {} BLAKE3 rows + {} element rows | M={} | {} public words | {} proof bytes\n  \
         prove {prove_ms:.1} ms | verify {verify_ms:.2} ms",
        built.shape.counts[built.registry_slot(hash)],
        built.shape.counts[built.registry_slot(arith)],
        outer.m_total(),
        built.witness.public.len(),
        serialize(&proof).unwrap().len(),
    );
}
