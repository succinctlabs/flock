use super::*;
use crate::r1cs_hashes::fs_chain::{CvSource, Link, trace_duplex_forked};
use flock_hash::blake3_compress;
use flock_transcript::transcript_record::{StreamWord, TranscriptOp as Op};

/// A shared-constant public: one public input PER DISTINCT VALUE, wired to
/// every use through copy constraints — the `zw`/`ow` pattern generalized.
/// The per-row structural words (params, zero pads) collapse from one
/// public per ROW to one per VALUE; being few and public they are also the
/// auditable surface the checker contract pins (the fixed-shape statement).
pub(super) fn cw(
    sb: &mut ShapeBuilder,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    v: F128,
) -> Wire {
    match consts.iter().find(|&&(x, _)| x == v) {
        Some(&(_, w)) => w,
        None => {
            vals.push(v);
            let w = sb.fixed_public_input(v);
            consts.push((v, w));
            w
        }
    }
}

/// Which byte payloads of a tape stay PUBLIC under the witness/public
/// split: every `observe_bytes` payload — the STATEMENT surfaces (registry
/// digest, counts, caps, a child's circuit digest + public words) and
/// nothing else. PoW nonces share the payload counter but remain private
/// witnesses constrained by the fused BLAKE3 and bit-spread rows.
pub(super) fn bytes_payload_mask(
    ops: &[flock_transcript::transcript_record::TranscriptOp],
) -> Vec<bool> {
    let mut v = Vec::new();
    for op in ops {
        match op {
            Op::ObserveBytes(_) => v.push(true),
            Op::Pow { .. } | Op::LegacyPow { .. } => v.push(false),
            _ => {}
        }
    }
    v
}

/// The 32-byte AG sampling seed from its two transcript squeezes — the
/// exact layout `ag_skip::r1_seed` writes: s0.lo, s0.hi, s1.lo, s1.hi,
/// each LE.
pub(super) fn ag_seed_bytes(s0: F128, s1: F128) -> [u8; 32] {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&s0.lo.to_le_bytes());
    seed[8..16].copy_from_slice(&s0.hi.to_le_bytes());
    seed[16..24].copy_from_slice(&s1.lo.to_le_bytes());
    seed[24..32].copy_from_slice(&s1.hi.to_le_bytes());
    seed
}

/// Re-derive an AG skip point from (seed, nonce) under the child's grinding
/// schedule — ONE hash + one attempt, the verifier's own fused-nonce
/// derivation (`H(seed ‖ nonce)` must clear the PoW target AND decode).
pub(super) fn decode_ag_point(
    seed: &[u8; 32],
    nonce: u32,
    pow_bits: Option<u32>,
) -> flock_core::genus95_curve_code::EvaluationPoint {
    match pow_bits {
        Some(bits) => flock_core::genus95_curve_code::evaluation_point_from_nonce_pow(
            seed,
            nonce,
            HashKind::Blake3,
            bits,
        ),
        None => flock_core::genus95_curve_code::evaluation_point_from_nonce(
            seed,
            nonce,
            HashKind::Blake3,
        ),
    }
    .expect("the AG nonce decodes to a valid cover point")
}

/// Replay a recorded transcript's FS chain into the blake3 slot; squeeze
/// rows chain off prior outputs. Returns the per-row output wires
/// (`trace.squeezes[fin]` indexes into them) and the per-stream-word wires.
///
/// **The witness/public split** (the recursion-composition fix): the child
/// PROOF BODY is existentially quantified — its stream words enter as
/// WITNESS inputs, bound in-circuit by the chain compressions and the
/// region gates that consume them, never read natively. What stays public:
/// the byte payloads `pub_payloads` selects (the STATEMENT: digests,
/// counts, caps — the caps' wires also feed the in-circuit cap trees the
/// openings connect to), domain constants, and the shared structural
/// constants through `consts`.
///
/// **The FORKED transcript needs nothing from this function.** A circuit-bound
/// union proof always forks (the wiring argument runs on its own chain), but
/// [`merge_chain`] presents both chains as ONE — rows spliced at the fork
/// point, indices remapped — so this loop only ever sees a linear trace. The
/// fork's whole in-circuit footprint arrives through `cross`: four words that
/// are ALIASES of earlier squeeze outputs rather than declared inputs.
///
/// Still open, and cheaper still: the child chain could CONTINUE from the
/// fork-point CV under a domain byte instead of seed-squeeze-then-absorb.
/// That drops the two seed rows and takes the fork's cost to ~one row.
pub(super) fn emit_fs_chain(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    iv: [Wire; 2],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: &flock_transcript::transcript_record::Stream,
    bytes: &[u8],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    pub_payloads: &[bool],
    cross: &[Option<(usize, usize)>],
) -> (Vec<Vec<Wire>>, Vec<Option<Wire>>) {
    emit_fs_chain_partitioned(
        sb,
        b3,
        None,
        iv,
        trace,
        stream,
        bytes,
        vals,
        consts,
        pub_payloads,
        cross,
    )
}

/// As [`emit_fs_chain`], with rows at and after `primary_rows` emitted into
/// a second slot carrying the same BLAKE3 relation. Wires may cross the slot
/// boundary normally; the circuit's copy constraints preserve the chain.
pub(super) fn emit_fs_chain_partitioned(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    alternate: Option<(flock_core::circuit::builder::SlotId, usize)>,
    iv: [Wire; 2],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    stream: &flock_transcript::transcript_record::Stream,
    bytes: &[u8],
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    pub_payloads: &[bool],
    cross: &[Option<(usize, usize)>],
) -> (Vec<Vec<Wire>>, Vec<Option<Wire>>) {
    let mut word_wire: Vec<Option<Wire>> = vec![None; stream.words.len()];
    let mut outs: Vec<Vec<Wire>> = Vec::with_capacity(trace.rows.len());
    let mut gate_in: Vec<[Wire; 7]> = Vec::with_capacity(trace.rows.len());
    for (i, row) in trace.rows.iter().enumerate() {
        let b3_row = match alternate {
            Some((slot, primary_rows)) if i >= primary_rows => slot,
            _ => b3,
        };
        let (_, _, counter, blen, flags) = *row;
        let link = trace.links[i];
        let params = cw(sb, vals, consts, pack_params(counter, blen, flags));
        if let Some(root) = link.repeats {
            let s = gate_in[root];
            let g_in = [s[0], s[1], s[2], s[3], s[4], s[5], params];
            gate_in.push(g_in);
            outs.push(sb.gate(b3_row, &g_in));
            continue;
        }
        let (cv_in, m_in) = match link.right {
            Some(right) => {
                let l = match link.cv {
                    CvSource::Row(r) => r,
                    CvSource::RowHi(r) => r,
                    CvSource::Iv => unreachable!(),
                };
                let left = match link.cv {
                    CvSource::Row(_) => [outs[l][0], outs[l][1]],
                    CvSource::RowHi(_) => [outs[l][2], outs[l][3]],
                    CvSource::Iv => unreachable!(),
                };
                (iv, [left[0], left[1], outs[right][0], outs[right][1]])
            }
            None if trace.block_offsets[i].is_none() => {
                // A sponge-chain SQUEEZE output row (transcript-v2): zero
                // message block via the shared constant, chaining value
                // from the link.
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let z4 = cw(sb, vals, consts, F128::ZERO);
                (cv_in, [z4, z4, z4, z4])
            }
            None => {
                let cv_in = match link.cv {
                    CvSource::Iv => iv,
                    CvSource::Row(r) => [outs[r][0], outs[r][1]],
                    CvSource::RowHi(r) => [outs[r][2], outs[r][3]],
                };
                let base = trace.block_offsets[i].expect("stream block") / 16;
                let real = trace.block_word_counts[i];
                let mut m = [iv[0]; 4];
                for (j, slot) in m.iter_mut().enumerate() {
                    let wi = base + j;
                    *slot = if j >= real || wi >= stream.words.len() {
                        cw(sb, vals, consts, F128::ZERO)
                    } else {
                        match word_wire[wi] {
                            Some(w) => w,
                            // A CROSS-LINK word is not an input at all: it IS
                            // an earlier squeeze's output (the fork's seed on
                            // the child side, the child's closing digest on
                            // the parent's). Aliasing the wire is the whole
                            // in-circuit cost of the fork — zero extra rows,
                            // and the link is unforgeable because the row that
                            // produced it is the same row the challenge came
                            // from.
                            None if cross.get(wi).copied().flatten().is_some() => {
                                let (row, half) = cross[wi].unwrap();
                                assert!(row < i, "cross-link word {wi} reads row {row} >= {i}");
                                let w = outs[row][half];
                                word_wire[wi] = Some(w);
                                w
                            }
                            None => {
                                let v = F128::new(
                                    u64::from_le_bytes(
                                        bytes[wi * 16..wi * 16 + 8].try_into().unwrap(),
                                    ),
                                    u64::from_le_bytes(
                                        bytes[wi * 16 + 8..wi * 16 + 16].try_into().unwrap(),
                                    ),
                                );
                                let w = match &stream.words[wi] {
                                    // Domain labels are statement CONSTANTS
                                    // that repeat with every region — one
                                    // public per VALUE via the shared cache,
                                    // not one per occurrence (the census
                                    // found ~2.3k of the latter per child).
                                    StreamWord::Const(_) => cw(sb, vals, consts, v),
                                    StreamWord::Bytes { payload, .. }
                                        if pub_payloads.get(*payload).copied().unwrap_or(true) =>
                                    {
                                        vals.push(v);
                                        sb.public_input()
                                    }
                                    _ => {
                                        vals.push(v);
                                        sb.input()
                                    }
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
        let g_in = [
            cv_in[0], cv_in[1], m_in[0], m_in[1], m_in[2], m_in[3], params,
        ];
        gate_in.push(g_in);
        outs.push(sb.gate(b3_row, &g_in));
    }
    (outs, word_wire)
}

/// Splice every fork's ops inline at its fork position and drop the `Merge`
/// markers — the FLAT view of a forked transcript.
///
/// This is the whole locator story. Every region walker in this file resolves
/// its position by `find(label)` over a flat op list and by counting
/// value/challenge/finalize ops up to an index; a fork's ops sit inline at the
/// fork slot (the recorder splices values, payloads and challenges at the
/// fork-time bases for exactly this reason), so on the flattened view every
/// label is found and every ordinal is the GLOBAL one. No walker changes, no
/// chain index anywhere.
pub(super) fn flatten_ops(
    ops: &[flock_transcript::transcript_record::TranscriptOp],
) -> Vec<flock_transcript::transcript_record::TranscriptOp> {
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            Op::Forked { ops: child, .. } => out.extend(flatten_ops(child)),
            Op::Merge { .. } => {}
            other => out.push(other.clone()),
        }
    }
    out
}

/// A forked transcript presented to the circuit as ONE chain.
///
/// The fork is two independent BLAKE3 chains, but the emitter, the region
/// locators and every `trace.squeezes[fin]` site want a single linear
/// numbering. So the child's rows are SPLICED into the parent's at the fork
/// point — after the two seed squeezes, before the parent's post-fork absorbs
/// — and every row index is remapped. The result is indistinguishable from an
/// unforked trace except for four wires:
///
/// - the child's two opening seed words ARE the parent's two seed-squeeze
///   outputs, and
/// - the parent's two merge words ARE the child's two closing-squeeze outputs.
///
/// [`MergedChain::cross`] carries those four as (row, half) aliases, so they
/// cost no gates and no rows: the emitter wires them instead of declaring
/// inputs. Splicing at the fork point is what makes that possible — both
/// sources are already emitted when their consumer's row comes up.
pub(super) struct MergedChain {
    /// Parent words then child words, in the same order as `trace`'s rows.
    pub(super) stream: flock_transcript::transcript_record::Stream,
    pub(super) bytes: Vec<u8>,
    pub(super) trace: crate::r1cs_hashes::fs_chain::FsChainTrace,
    /// Per merged word: `Some((row, half))` iff the word is a cross-link.
    pub(super) cross: Vec<Option<(usize, usize)>>,
}

/// Build the merged view from a recorded shape's ops and its parent stream.
/// With no fork this is the identity (the trace the sites built by hand).
pub(super) fn merge_chain(
    ops: &[flock_transcript::transcript_record::TranscriptOp],
    stream: &flock_transcript::transcript_record::Stream,
    values: &[F128],
    payloads: &[Vec<u8>],
) -> MergedChain {
    let chains = trace_duplex_forked(ops, stream, values, payloads);
    let parent_bytes = stream.to_bytes(values, payloads);
    if chains.children.is_empty() {
        let cross = vec![None; stream.words.len()];
        return MergedChain {
            stream: stream.clone(),
            bytes: parent_bytes,
            trace: chains.parent,
            cross,
        };
    }
    let p = &chains.parent;
    let n_ch = chains.children.len();
    // Every fork's splice point, in parent-row order. The forks a proof takes
    // are SEQUENTIAL (the wiring's opens and closes before the opening phase
    // starts its own), so the splits are strictly increasing and each child
    // occupies one contiguous run of merged rows. `last_seed` is the second
    // of the pair — `seed_squeeze` names the first — which is exactly where
    // the flattened ops put the child.
    let last_seed: Vec<usize> = chains.children.iter().map(|c| c.seed_squeeze + 1).collect();
    let splits: Vec<usize> = last_seed
        .iter()
        .map(|&k| p.squeezes[k].iter().copied().max().unwrap() + 1)
        .collect();
    assert!(
        splits.windows(2).all(|w| w[0] <= w[1]) && last_seed.windows(2).all(|w| w[0] < w[1]),
        "forks must be sequential — nested or interleaved forks are not supported"
    );
    let ncs: Vec<usize> = chains.children.iter().map(|c| c.trace.rows.len()).collect();
    // A parent row shifts by every child spliced at or before it; child `i`
    // starts after its split plus every earlier child's rows.
    let pmap = |r: usize| {
        r + (0..n_ch)
            .filter(|&j| splits[j] <= r)
            .map(|j| ncs[j])
            .sum::<usize>()
    };
    let child_base: Vec<usize> = (0..n_ch)
        .map(|i| splits[i] + ncs[..i].iter().sum::<usize>())
        .collect();
    let cmap = |i: usize| {
        let base = child_base[i];
        move |r: usize| base + r
    };
    let remap = |l: &Link, f: &dyn Fn(usize) -> usize| Link {
        cv: match l.cv {
            CvSource::Iv => CvSource::Iv,
            CvSource::Row(r) => CvSource::Row(f(r)),
            CvSource::RowHi(r) => CvSource::RowHi(f(r)),
        },
        right: l.right.map(f),
        repeats: l.repeats.map(f),
    };

    // Child streams and their word/byte offsets in the merged view.
    let cstreams: Vec<&flock_transcript::transcript_record::Stream> =
        stream.forks.iter().map(|f| &f.stream).collect();
    let woffs: Vec<usize> = (0..n_ch)
        .map(|i| stream.words.len() + cstreams[..i].iter().map(|s| s.words.len()).sum::<usize>())
        .collect();
    debug_assert_eq!(
        parent_bytes.len(),
        stream.words.len() * 16,
        "words are 16 bytes"
    );

    // Splice: parent up to split 0, child 0, parent to split 1, child 1, ...
    let mut rows = Vec::new();
    let mut links: Vec<Link> = Vec::new();
    let mut block_offsets = Vec::new();
    let mut block_word_counts = Vec::new();
    let mut at = 0usize;
    for i in 0..n_ch {
        let c = &chains.children[i];
        let cm = cmap(i);
        rows.extend_from_slice(&p.rows[at..splits[i]]);
        links.extend(p.links[at..splits[i]].iter().map(|l| remap(l, &pmap)));
        block_offsets.extend_from_slice(&p.block_offsets[at..splits[i]]);
        block_word_counts.extend_from_slice(&p.block_word_counts[at..splits[i]]);
        rows.extend_from_slice(&c.trace.rows);
        links.extend(c.trace.links.iter().map(|l| remap(l, &cm)));
        block_offsets.extend(
            c.trace
                .block_offsets
                .iter()
                .map(|o| o.map(|b| b + woffs[i] * 16)),
        );
        block_word_counts.extend_from_slice(&c.trace.block_word_counts);
        at = splits[i];
    }
    rows.extend_from_slice(&p.rows[at..]);
    links.extend(p.links[at..].iter().map(|l| remap(l, &pmap)));
    block_offsets.extend_from_slice(&p.block_offsets[at..]);
    block_word_counts.extend_from_slice(&p.block_word_counts[at..]);

    // The same splice on the squeeze list, the words and the finalize points.
    let mut squeezes: Vec<Vec<usize>> = Vec::new();
    let mut squeeze_words: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut words = stream.words.clone();
    let mut finalize_after: Vec<usize> = Vec::new();
    let mut bytes = parent_bytes;
    let mut at = 0usize;
    for i in 0..n_ch {
        let c = &chains.children[i];
        let cm = cmap(i);
        squeezes.extend(
            p.squeezes[at..=last_seed[i]]
                .iter()
                .map(|s| s.iter().copied().map(pmap).collect::<Vec<_>>()),
        );
        squeeze_words.extend(
            p.squeeze_words[at..=last_seed[i]]
                .iter()
                .map(|s| s.iter().map(|&(r, w)| (pmap(r), w)).collect::<Vec<_>>()),
        );
        squeezes.extend(
            c.trace
                .squeezes
                .iter()
                .map(|s| s.iter().copied().map(&cm).collect::<Vec<_>>()),
        );
        squeeze_words.extend(
            c.trace
                .squeeze_words
                .iter()
                .map(|s| s.iter().map(|&(r, w)| (cm(r), w)).collect::<Vec<_>>()),
        );
        finalize_after.extend_from_slice(&stream.finalize_after[at..=last_seed[i]]);
        finalize_after.extend(cstreams[i].finalize_after.iter().map(|w| w + woffs[i]));
        words.extend(cstreams[i].words.iter().cloned());
        bytes.extend_from_slice(&cstreams[i].to_bytes(values, payloads));
        at = last_seed[i] + 1;
    }
    squeezes.extend(
        p.squeezes[at..]
            .iter()
            .map(|s| s.iter().copied().map(pmap).collect::<Vec<_>>()),
    );
    squeeze_words.extend(
        p.squeeze_words[at..]
            .iter()
            .map(|s| s.iter().map(|&(r, w)| (pmap(r), w)).collect::<Vec<_>>()),
    );
    finalize_after.extend_from_slice(&stream.finalize_after[at..]);

    // The four cross-links. Each side's pair is two CONSECUTIVE
    // `ObserveScalar`s, and the walk emits [header, value] per observe — so
    // the second word sits two after the first.
    //
    // Each link is CHECKED, not just placed. Getting a cross index wrong is
    // the one mistake nothing downstream would notice: challenges replay
    // identically whether a word is aliased to its squeeze or declared as a
    // witness input that happens to hold the same value — and the second is a
    // soundness hole, because it leaves the wiring's chain unbound. So the
    // row is compressed here and matched against the recorded value.
    let mut cross = vec![None; words.len()];
    let mut link_word = |wi: usize, row: usize| {
        let StreamWord::Value(vi) = words[wi] else {
            panic!("cross-link word {wi} is not an observed value");
        };
        let (cv, m, counter, blen, flags) = rows[row];
        let out = blake3_compress(&cv, &m, counter, blen, flags);
        let mut b = [0u8; 16];
        for (i, w) in out[..4].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        assert_eq!(
            F128::new(
                u64::from_le_bytes(b[..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..].try_into().unwrap()),
            ),
            values[vi],
            "cross-link word {wi} does not carry row {row}'s squeeze output"
        );
        cross[wi] = Some((row, 0));
    };
    for i in 0..n_ch {
        let c = &chains.children[i];
        let cm = cmap(i);
        for (k, half) in [(c.seed_squeeze, 0usize), (c.seed_squeeze + 1, 1)] {
            link_word(
                woffs[i] + c.child_seed_word + 2 * half,
                pmap(p.squeezes[k][0]),
            );
        }
        for (k, half) in [(c.digest_squeeze, 0usize), (c.digest_squeeze + 1, 1)] {
            link_word(c.parent_digest_word + 2 * half, cm(c.trace.squeezes[k][0]));
        }
    }

    assert_eq!(
        cross.iter().filter(|c| c.is_some()).count(),
        4 * n_ch,
        "each fork contributes exactly four cross-link words"
    );
    MergedChain {
        stream: flock_transcript::transcript_record::Stream {
            words,
            finalize_after,
            forks: Vec::new(),
        },
        bytes,
        trace: crate::r1cs_hashes::fs_chain::FsChainTrace {
            rows,
            links,
            squeezes,
            squeeze_words,
            block_offsets,
            block_word_counts,
        },
        cross,
    }
}

/// THE SPLICE DIFFERENTIAL: every challenge the recorder produced must
/// fall back out of the merged chain at its flattened finalize ordinal.
///
/// This is what makes [`merge_chain`] trustworthy rather than merely
/// plausible. A single match requires the row order, the index remapping, the
/// squeeze ordering AND the byte-offset shift to all be right at once, and the
/// child's own challenges sit in the middle of the sequence — the run walks
/// straight through the fork. Cheap enough to leave on at every real shape.
pub(super) fn assert_chain_replays(
    ops: &[flock_transcript::transcript_record::TranscriptOp],
    trace: &crate::r1cs_hashes::fs_chain::FsChainTrace,
    chals: &[F128],
) {
    let (mut fin, mut ch, mut checked) = (0usize, 0usize, 0usize);
    for op in ops {
        let n = match op {
            Op::SqueezeScalar => 1,
            Op::SqueezeSlice(n) => *n,
            _ => 0,
        };
        for j in 0..n {
            let (row, word) = trace.squeeze_words[fin][j];
            let (cv, m, counter, blen, flags) = trace.rows[row];
            let out = blake3_compress(&cv, &m, counter, blen, flags);
            let mut b = [0u8; 16];
            for (i, w) in out[word * 4..word * 4 + 4].iter().enumerate() {
                b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            assert_eq!(
                F128::new(
                    u64::from_le_bytes(b[..8].try_into().unwrap()),
                    u64::from_le_bytes(b[8..].try_into().unwrap()),
                ),
                chals[ch + j],
                "the merged chain diverges at finalize {fin}, word {j} (challenge {})",
                ch + j,
            );
            checked += 1;
        }
        if op.finalizes() {
            fin += 1;
        }
        match op {
            Op::SqueezeScalar => ch += 1,
            Op::SqueezeSlice(n) => ch += n,
            _ => {}
        }
    }
    assert!(checked > 0, "no scalar squeezes to replay");
    assert_eq!(fin, trace.squeezes.len(), "finalize count vs squeeze rows");
    assert_eq!(ch, chals.len(), "challenge count vs the recorded list");
}

/// Independent row count for the duplex transcript, including every fork as
/// its own IV-rooted chain.  It deliberately derives absorption from the
/// serialized stream rather than from [`FsChainTrace`].
pub(super) fn duplex_row_count_model(
    ops: &[flock_transcript::transcript_record::TranscriptOp],
    stream: &flock_transcript::transcript_record::Stream,
) -> usize {
    let mut pending_pow = None;
    let mut finals: Vec<(&Op, Option<u32>)> = Vec::new();
    for op in ops {
        match op {
            Op::Pow { bits } => {
                assert!(
                    pending_pow.replace(*bits).is_none(),
                    "nested fused PoW markers"
                );
            }
            op if op.finalizes() => finals.push((op, pending_pow.take())),
            Op::Forked { .. } => {}
            _ => assert!(pending_pow.is_none(), "fused PoW must precede its squeeze"),
        }
    }
    assert!(pending_pow.is_none(), "fused PoW marker without a squeeze");
    assert_eq!(finals.len(), stream.finalize_after.len());

    let (mut rows, mut at, mut pending) = (0usize, 0usize, 0usize);
    for (k, &upto) in stream.finalize_after.iter().enumerate() {
        pending += 16 * (upto - at);
        at = upto;
        let (op, pow_bits) = finals[k];
        let words = op.squeezed_bytes() / 16;
        if pow_bits.is_some() {
            // Retain the final (possibly full) block for the fused row.
            rows += pending.saturating_sub(1) / 64;
            rows += 1 + words.saturating_sub(3).div_ceil(4);
        } else {
            // Ordinary absorb drains every full block before the squeeze.
            rows += pending / 64;
            rows += 1 + words.saturating_sub(4).div_ceil(4);
        }
        pending = 0;
    }
    pending += 16 * (stream.words.len() - at);
    rows += pending / 64;

    let children: Vec<_> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Forked { label, ops } => Some((label, ops)),
            _ => None,
        })
        .collect();
    assert_eq!(children.len(), stream.forks.len());
    for ((label, child_ops), child_stream) in children.into_iter().zip(&stream.forks) {
        assert_eq!(label, &child_stream.label);
        rows += duplex_row_count_model(child_ops, &child_stream.stream);
    }
    rows
}
