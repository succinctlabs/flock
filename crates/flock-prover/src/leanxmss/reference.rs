//! Native leanXMSS: verification, key generation and signing, generic over the
//! hash `H`.
//!
//! Written from leanVM's `leanXMSS.pdf` and its `crates/signatures` reference,
//! which agree byte for byte. leanVM instantiates `H` with Keccak-256 (since
//! 2026-10-23, `4af2cdd`) and before that with BLAKE2s-256 (last at `4242200`);
//! only `H` changed between them. BLAKE3 is not a leanVM instantiation: it is
//! the same scheme with the same byte strings and `H` swapped.
//!
//! Every value, `P` and every address is 16 bytes, which is one Flock word.
//! `Th(P, A, M)` is the first 16 bytes of `H(P ‖ A ‖ M)`.

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// The hash `H` the scheme is instantiated with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum XmssHash {
    /// Unkeyed BLAKE3 with a 32-byte output. Every leanXMSS input is at most
    /// 1024 bytes, so `H` is a single chunk: plain chained compressions.
    Blake3,
    /// Unkeyed BLAKE2s-256 (RFC 7693): no key, salt or personalization.
    Blake2s,
}

impl XmssHash {
    pub fn hash(self, data: &[u8]) -> [u8; 32] {
        match self {
            XmssHash::Blake3 => *blake3::hash(data).as_bytes(),
            XmssHash::Blake2s => blake2s256(data),
        }
    }

    /// Compressions for a `len`-byte input (`len ≤ 1024` for BLAKE3).
    pub fn compressions(self, len: usize) -> usize {
        len.div_ceil(64).max(1)
    }
}

const B2_IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

const B2_SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// One BLAKE2s compression: `t` counts the bytes absorbed including this
/// block, and `last` sets the final-block flag `f0`.
fn blake2s_compress(h: &mut [u32; 8], m: &[u32; 16], t: u64, last: bool) {
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&B2_IV);
    v[12] ^= t as u32;
    v[13] ^= (t >> 32) as u32;
    if last {
        v[14] = !v[14];
    }
    let g = |v: &mut [u32; 16], [a, b, c, d]: [usize; 4], x: u32, y: u32| {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    };
    for s in &B2_SIGMA {
        g(&mut v, [0, 4, 8, 12], m[s[0]], m[s[1]]);
        g(&mut v, [1, 5, 9, 13], m[s[2]], m[s[3]]);
        g(&mut v, [2, 6, 10, 14], m[s[4]], m[s[5]]);
        g(&mut v, [3, 7, 11, 15], m[s[6]], m[s[7]]);
        g(&mut v, [0, 5, 10, 15], m[s[8]], m[s[9]]);
        g(&mut v, [1, 6, 11, 12], m[s[10]], m[s[11]]);
        g(&mut v, [2, 7, 8, 13], m[s[12]], m[s[13]]);
        g(&mut v, [3, 4, 9, 14], m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Unkeyed BLAKE2s-256: the zero-padded last block carries the final flag and
/// the total length.
pub fn blake2s256(data: &[u8]) -> [u8; 32] {
    let mut h = B2_IV;
    h[0] ^= 0x0101_0000 ^ 32; // digest length 32, no key, fanout 1, depth 1
    let blocks = data.len().div_ceil(64).max(1);
    for b in 0..blocks {
        let chunk = &data[(64 * b).min(data.len())..(64 * (b + 1)).min(data.len())];
        let mut block = [0u8; 64];
        block[..chunk.len()].copy_from_slice(chunk);
        let m = std::array::from_fn(|i| {
            u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
        });
        let last = b + 1 == blocks;
        let t = if last {
            data.len() as u64
        } else {
            64 * (b as u64 + 1)
        };
        blake2s_compress(&mut h, &m, t, last);
    }
    let mut out = [0u8; 32];
    for (o, w) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        *o = w.to_le_bytes();
    }
    out
}

/// Bytes of every value, of `P` and of every address.
pub const N: usize = 16;
/// Winternitz chains per one-time key.
pub const V: usize = 58;
/// Bits per digit: chain positions are `0..=3`.
pub const W: usize = 2;
/// The top chain position.
pub const CHAIN_END: usize = 3;
/// The digit sum every codeword has.
pub const TARGET_SUM: usize = 106;
/// Each layer's tree height.
pub const HEIGHT: usize = 16;
/// One layer's signature: `ρ ‖ y_0..y_57 ‖ π_0..π_15`.
pub const LAYER_SIG_LEN: usize = N * (1 + V + HEIGHT);
/// `LE32(idx) ‖ bottom layer ‖ top layer`.
pub const SIG_LEN: usize = 4 + 2 * LAYER_SIG_LEN;
/// `root ‖ P`.
pub const PK_LEN: usize = 2 * N;

/// A 16-byte value.
pub type Value = [u8; N];

/// Address type tags (byte 7, low 5 bits).
pub const TAG_PRF: u8 = 0;
pub const TAG_CHAIN: u8 = 1;
pub const TAG_LEAF: u8 = 2;
pub const TAG_NODE: u8 = 3;
pub const TAG_PARAM: u8 = 5;
pub const TAG_RANDOMIZER: u8 = 7;
pub const TAG_MSG: u8 = 12;
/// Mock keys only: never hashed by verification.
pub const TAG_FILLER: u8 = 13;

/// `enc(τ, µ, hi, lo) = LE32(lo) ‖ LE24(hi) ‖ LE8(τ + 32µ) ‖ 0^64`.
pub fn address(tag: u8, step: u8, hi: u32, lo: u32) -> Value {
    assert!(tag < 32 && step < 8 && hi < (1 << 24));
    let mut a = [0u8; N];
    a[..4].copy_from_slice(&lo.to_le_bytes());
    a[4..7].copy_from_slice(&hi.to_le_bytes()[..3]);
    a[7] = tag | step << 5;
    a
}

/// `hi_ℓ(a) = a + 2^23·ℓ`: the top layer sets bit 23 of `hi`.
pub fn layer_hi(layer: usize, a: u32) -> u32 {
    a + ((layer as u32) << 23)
}

/// `Th(P, A, M)`: the first 16 bytes of `H(P ‖ A ‖ M)`.
pub fn th(hash: XmssHash, pp: &Value, addr: &Value, parts: &[&[u8]]) -> Value {
    let mut buf = Vec::with_capacity(2 * N + parts.iter().map(|p| p.len()).sum::<usize>());
    buf.extend_from_slice(pp);
    buf.extend_from_slice(addr);
    for p in parts {
        buf.extend_from_slice(p);
    }
    hash.hash(&buf)[..N].try_into().unwrap()
}

/// `LE128(i)`, the chain index suffix of a chain step.
pub fn le128(i: usize) -> Value {
    (i as u128).to_le_bytes()
}

/// One layer of a signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerSig {
    pub rho: Value,
    pub y: [Value; V],
    pub path: [Value; HEIGHT],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub idx: u32,
    /// Bottom, then top.
    pub layers: [LayerSig; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicKey {
    pub root: Value,
    pub pp: Value,
}

impl PublicKey {
    pub fn to_bytes(&self) -> [u8; PK_LEN] {
        let mut out = [0u8; PK_LEN];
        out[..N].copy_from_slice(&self.root);
        out[N..].copy_from_slice(&self.pp);
        out
    }

    pub fn from_bytes(bytes: &[u8; PK_LEN]) -> Self {
        PublicKey {
            root: bytes[..N].try_into().unwrap(),
            pp: bytes[N..].try_into().unwrap(),
        }
    }
}

impl Signature {
    /// leanVM's SSZ encoding: `LE32(idx)`, then each layer's `ρ ‖ y ‖ π`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SIG_LEN);
        out.extend_from_slice(&self.idx.to_le_bytes());
        for layer in &self.layers {
            out.extend_from_slice(&layer.rho);
            for v in layer.y.iter().chain(&layer.path) {
                out.extend_from_slice(v);
            }
        }
        out
    }

    /// `None` unless `bytes` is exactly [`SIG_LEN`] long; every such string
    /// parses.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != SIG_LEN {
            return None;
        }
        let value = |off: usize| -> Value { bytes[off..off + N].try_into().unwrap() };
        let layer = |base: usize| LayerSig {
            rho: value(base),
            y: std::array::from_fn(|i| value(base + N * (1 + i))),
            path: std::array::from_fn(|l| value(base + N * (1 + V + l))),
        };
        Some(Signature {
            idx: u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            layers: [layer(4), layer(4 + LAYER_SIG_LEN)],
        })
    }
}

/// The digits of a digest if it is a codeword: `x_i = (LE⁻¹(D) >> 2i) & 3`,
/// digits 58..63 zero, and the first 58 summing to [`TARGET_SUM`].
pub fn codeword(digest: &Value) -> Option<[u8; V]> {
    let d = u128::from_le_bytes(*digest);
    if d >> (W * V) != 0 {
        return None;
    }
    let digits: [u8; V] = std::array::from_fn(|i| ((d >> (W * i)) & 3) as u8);
    (digits.iter().map(|&x| x as usize).sum::<usize>() == TARGET_SUM).then_some(digits)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// The signature is not [`SIG_LEN`] bytes.
    BadLength,
    /// That layer's message digest is not a codeword.
    InvalidEncoding(usize),
    /// The top layer reached a root other than the public key's.
    RootMismatch,
}

/// Observes every `Th` call of a verification: a label, the full `H` input
/// and the 16-byte output.
pub type Trace<'a> = &'a mut dyn FnMut(&str, &[u8], &Value);

fn th_traced(
    hash: XmssHash,
    pp: &Value,
    addr: &Value,
    parts: &[&[u8]],
    label: impl FnOnce() -> String,
    trace: &mut Option<Trace>,
) -> Value {
    let out = th(hash, pp, addr, parts);
    if let Some(f) = trace {
        let mut input = [&pp[..], &addr[..]].concat();
        for p in parts {
            input.extend_from_slice(p);
        }
        f(&label(), &input, &out);
    }
    out
}

/// The address of the message digest, of a chain step leaving position
/// `step`, of the one-time key's leaf, and of the node at `level` (1..=16)
/// above key `e`.
pub fn msg_address(layer: usize, e: u32) -> Value {
    address(TAG_MSG, 0, layer_hi(layer, 0), e)
}
pub fn chain_address(layer: usize, e: u32, step: usize) -> Value {
    address(TAG_CHAIN, step as u8, layer_hi(layer, 0), e)
}
pub fn leaf_address(layer: usize, e: u32) -> Value {
    address(TAG_LEAF, 0, layer_hi(layer, 0), e)
}
pub fn node_address(layer: usize, e: u32, level: usize) -> Value {
    address(TAG_NODE, 0, layer_hi(layer, level as u32), e >> level)
}

/// One-time-key leaf then tree fold for key `e` of `layer`: the root reached,
/// or `None` if the message digest is not a codeword.
pub fn layer_root(
    hash: XmssHash,
    pp: &Value,
    layer: usize,
    e: u32,
    msg: &[u8],
    sig: &LayerSig,
    trace: &mut Option<Trace>,
) -> Option<Value> {
    let digest = th_traced(
        hash,
        pp,
        &msg_address(layer, e),
        &[msg, &sig.rho],
        || format!("L{layer} msg"),
        trace,
    );
    let digits = codeword(&digest)?;
    let mut ends = [[0u8; N]; V];
    for (i, end) in ends.iter_mut().enumerate() {
        let mut c = sig.y[i];
        for step in digits[i] as usize..CHAIN_END {
            c = th_traced(
                hash,
                pp,
                &chain_address(layer, e, step),
                &[&c, &le128(i)],
                || format!("L{layer} chain {i} step {step}"),
                trace,
            );
        }
        *end = c;
    }
    let mut node = th_traced(
        hash,
        pp,
        &leaf_address(layer, e),
        &[ends.as_flattened()],
        || format!("L{layer} leaf"),
        trace,
    );
    for level in 0..HEIGHT {
        let sibling = &sig.path[level];
        let (l, r) = if (e >> level) & 1 == 0 {
            (&node, sibling)
        } else {
            (sibling, &node)
        };
        node = th_traced(
            hash,
            pp,
            &node_address(layer, e, level + 1),
            &[l, r],
            || format!("L{layer} node {}", level + 1),
            trace,
        );
    }
    Some(node)
}

/// leanXMSS verification: the bottom one-time key `idx` signs `msg`, the top
/// one-time key `idx >> 16` signs the bottom tree's root, and the top tree
/// must reach `pk.root`.
pub fn verify(
    hash: XmssHash,
    pk: &PublicKey,
    msg: &[u8; 32],
    sig: &Signature,
    mut trace: Option<Trace>,
) -> Result<(), VerifyError> {
    let bottom = layer_root(hash, &pk.pp, 0, sig.idx, msg, &sig.layers[0], &mut trace)
        .ok_or(VerifyError::InvalidEncoding(0))?;
    let top = layer_root(
        hash,
        &pk.pp,
        1,
        sig.idx >> HEIGHT,
        &bottom,
        &sig.layers[1],
        &mut trace,
    )
    .ok_or(VerifyError::InvalidEncoding(1))?;
    if top == pk.root {
        Ok(())
    } else {
        Err(VerifyError::RootMismatch)
    }
}

pub fn verify_bytes(
    hash: XmssHash,
    pk: &[u8; PK_LEN],
    msg: &[u8; 32],
    sig: &[u8],
) -> Result<(), VerifyError> {
    let sig = Signature::from_bytes(sig).ok_or(VerifyError::BadLength)?;
    verify(hash, &PublicKey::from_bytes(pk), msg, &sig, None)
}

/// The least `ρ = R0 + c` whose message digest is a codeword, with its digits.
/// Expected `2^20` trials. `parallel` searches 2^16-trial batches on the rayon
/// pool and still returns the least `c`.
pub fn grind(
    hash: XmssHash,
    pp: &Value,
    layer: usize,
    e: u32,
    msg: &[u8],
    r0: &Value,
    parallel: bool,
) -> (Value, [u8; V]) {
    let base = u128::from_le_bytes(*r0);
    let addr = msg_address(layer, e);
    let try_c = |c: u64| {
        let rho = base.wrapping_add(c as u128).to_le_bytes();
        codeword(&th(hash, pp, &addr, &[msg, &rho])).map(|digits| (rho, digits))
    };
    const BATCH: u64 = 1 << 16;
    for start in (0..1u64 << 32).step_by(BATCH as usize) {
        let found = if parallel {
            (start..start + BATCH)
                .into_par_iter()
                .find_first(|&c| try_c(c).is_some())
        } else {
            (start..start + BATCH).find(|&c| try_c(c).is_some())
        };
        if let Some(c) = found {
            return try_c(c).unwrap();
        }
    }
    panic!("no codeword among 2^32 randomizers");
}

/// A key's secret seed `S = H("leanXMSS" ‖ S0)` and its public parameter `P`.
pub struct Keys {
    hash: XmssHash,
    seed: [u8; 32],
    pub pp: Value,
}

impl Keys {
    pub fn new(hash: XmssHash, s0: &[u8; 32]) -> Self {
        let seed = hash.hash(&[&b"leanXMSS"[..], s0].concat());
        let pp = th(hash, &[0u8; N], &address(TAG_PARAM, 0, 0, 0), &[&seed]);
        Keys { hash, seed, pp }
    }

    /// `C_{i,0}`: chains `2t` and `2t+1` are the halves of `H(P ‖ A_prf(ℓ,e,t) ‖ S)`.
    fn chain_starts(&self, layer: usize, e: u32) -> [Value; V] {
        let mut out = [[0u8; N]; V];
        for t in 0..V / 2 {
            let addr = address(TAG_PRF, 0, layer_hi(layer, t as u32), e);
            let h = self.hash.hash(&[&self.pp[..], &addr, &self.seed].concat());
            out[2 * t] = h[..N].try_into().unwrap();
            out[2 * t + 1] = h[N..].try_into().unwrap();
        }
        out
    }

    fn chain(&self, layer: usize, e: u32, i: usize, from: usize, to: usize, mut c: Value) -> Value {
        for step in from..to {
            c = th(
                self.hash,
                &self.pp,
                &chain_address(layer, e, step),
                &[&c, &le128(i)],
            );
        }
        c
    }

    fn one_time_leaf(&self, layer: usize, e: u32) -> Value {
        let starts = self.chain_starts(layer, e);
        let ends: [Value; V] =
            std::array::from_fn(|i| self.chain(layer, e, i, 0, CHAIN_END, starts[i]));
        th(
            self.hash,
            &self.pp,
            &leaf_address(layer, e),
            &[ends.as_flattened()],
        )
    }

    fn filler(&self, layer: usize, level: usize, j: u32) -> Value {
        let addr = address(TAG_FILLER, 0, layer_hi(layer, level as u32), j);
        th(self.hash, &self.pp, &addr, &[&self.seed])
    }

    /// One-time signature of `msg` by key `e`: the least admissible `ρ`, and
    /// each chain advanced to its digit.
    pub fn sign_one_time(&self, layer: usize, e: u32, msg: &[u8]) -> (Value, [Value; V]) {
        let addr = address(TAG_RANDOMIZER, 0, layer_hi(layer, 0), e);
        let r0 = th(self.hash, &self.pp, &addr, &[&self.seed, msg]);
        let (rho, digits) = grind(self.hash, &self.pp, layer, e, msg, &r0, true);
        let starts = self.chain_starts(layer, e);
        (
            rho,
            std::array::from_fn(|i| self.chain(layer, e, i, 0, digits[i] as usize, starts[i])),
        )
    }
}

/// One layer's tree over leaves `first..=last` of the 2^16 under `base`;
/// nodes over no leaf in range are fillers (leanVM's mock keys; a full key
/// covers every leaf and uses none).
struct Tree {
    layer: usize,
    base: u32,
    first: u32,
    last: u32,
    levels: Vec<Vec<Value>>,
}

impl Tree {
    fn build(keys: &Keys, layer: usize, base: u32, first: u32, last: u32) -> Self {
        assert!(first <= last && last < (1 << HEIGHT));
        let leaves = (first..=last)
            .into_par_iter()
            .map(|j| keys.one_time_leaf(layer, base + j))
            .collect();
        let mut tree = Tree {
            layer,
            base,
            first,
            last,
            levels: vec![leaves],
        };
        for level in 1..=HEIGHT {
            let up = ((first >> level)..=(last >> level))
                .into_par_iter()
                .map(|j| {
                    let l = tree.at(keys, level - 1, 2 * j);
                    let r = tree.at(keys, level - 1, 2 * j + 1);
                    let addr = address(
                        TAG_NODE,
                        0,
                        layer_hi(layer, level as u32),
                        (base >> level) + j,
                    );
                    th(keys.hash, &keys.pp, &addr, &[&l, &r])
                })
                .collect();
            tree.levels.push(up);
        }
        tree
    }

    fn at(&self, keys: &Keys, level: usize, j: u32) -> Value {
        if ((self.first >> level)..=(self.last >> level)).contains(&j) {
            self.levels[level][(j - (self.first >> level)) as usize]
        } else {
            keys.filler(self.layer, level, (self.base >> level) + j)
        }
    }

    fn root(&self) -> Value {
        self.levels[HEIGHT][0]
    }

    fn path(&self, keys: &Keys, leaf: u32) -> [Value; HEIGHT] {
        std::array::from_fn(|level| self.at(keys, level, (leaf >> level) ^ 1))
    }
}

/// A key real over leaf indices `first..=last` (the full key is
/// `0..=u32::MAX`), exactly as leanVM's `key_gen_from_seed` and
/// `mock::key_gen_from_seed`, and its signature of `msg` at `idx`.
pub fn keygen_and_sign(
    hash: XmssHash,
    s0: &[u8; 32],
    first: u32,
    last: u32,
    msg: &[u8; 32],
    idx: u32,
) -> (PublicKey, Signature) {
    assert!((first..=last).contains(&idx));
    let keys = Keys::new(hash, s0);
    let top = Tree::build(&keys, 1, 0, first >> HEIGHT, last >> HEIGHT);
    let b = idx >> HEIGHT;
    let base = b << HEIGHT;
    let bottom = Tree::build(
        &keys,
        0,
        base,
        first.max(base) - base,
        last.min(base | 0xFFFF) - base,
    );
    let (rho0, y0) = keys.sign_one_time(0, idx, msg);
    let (rho1, y1) = keys.sign_one_time(1, b, &bottom.root());
    let sig = Signature {
        idx,
        layers: [
            LayerSig {
                rho: rho0,
                y: y0,
                path: bottom.path(&keys, idx & 0xFFFF),
            },
            LayerSig {
                rho: rho1,
                y: y1,
                path: top.path(&keys, b),
            },
        ],
    };
    (
        PublicKey {
            root: top.root(),
            pp: keys.pp,
        },
        sig,
    )
}

/// A valid `(public key, signature)` for `msg` at `idx` without building any
/// tree: the revealed chain values and Merkle siblings are drawn from `seed`,
/// each layer's `ρ` is ground until its digest is a codeword, and the public
/// root is whatever the top layer folds to. Verification cannot tell this
/// from a real key's signature; it costs two grinds (about 2^21 hashes).
pub fn sign_keyless(
    hash: XmssHash,
    pp: &Value,
    idx: u32,
    msg: &[u8; 32],
    seed: u64,
    parallel: bool,
) -> (PublicKey, Signature) {
    let mut state = seed ^ 0x6c65_616e_584d_5353;
    let mut value = || -> Value {
        let mut v = [0u8; N];
        for half in v.as_chunks_mut::<8>().0 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *half = (z ^ (z >> 31)).to_le_bytes();
        }
        v
    };
    let mut layers = Vec::with_capacity(2);
    let mut msg_bytes = msg.to_vec();
    let mut root = [0u8; N];
    for layer in 0..2 {
        let e = if layer == 0 { idx } else { idx >> HEIGHT };
        let r0 = value();
        let (rho, _) = grind(hash, pp, layer, e, &msg_bytes, &r0, parallel);
        let sig = LayerSig {
            rho,
            y: std::array::from_fn(|_| value()),
            path: std::array::from_fn(|_| value()),
        };
        root = layer_root(hash, pp, layer, e, &msg_bytes, &sig, &mut None)
            .expect("the ground digest is a codeword");
        msg_bytes = root.to_vec();
        layers.push(sig);
    }
    let layers: [LayerSig; 2] = layers.try_into().unwrap();
    (PublicKey { root, pp: *pp }, Signature { idx, layers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn hashes_match_published_vectors() {
        assert_eq!(
            hex(&blake2s256(b"")),
            "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9"
        );
        assert_eq!(
            hex(&blake2s256(b"abc")),
            "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982"
        );
        // RFC 7693 Appendix-style multi-block check against the 2-block path.
        let long: Vec<u8> = (0..200u32).map(|i| i as u8).collect();
        assert_ne!(blake2s256(&long[..64]), blake2s256(&long[..65]));
    }

    #[test]
    fn address_layout_matches_leanvm() {
        // leanVM `address_layout`: enc(3, 5, hi_1(0x12), 0x89abcdef).
        let a = address(TAG_NODE, 5, layer_hi(1, 0x12), 0x89ab_cdef);
        assert_eq!(a[..8], [0xef, 0xcd, 0xab, 0x89, 0x12, 0x00, 0x80, 0xa3]);
        assert_eq!(a[8..], [0; 8]);
    }

    /// leanVM's `known_answer()` at the last BLAKE2s commit (`4242200`): a full
    /// key from `seed[i] = 37i + 11`, `message[i] = 7i + 3`, leaf index
    /// `0xA5C37E19`. Pins the public key and `H(signature)`.
    #[test]
    fn blake2s_matches_leanvm_known_answer() {
        let s0: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let msg: [u8; 32] = std::array::from_fn(|i| (i * 7 + 3) as u8);
        let (pk, sig) = keygen_and_sign(XmssHash::Blake2s, &s0, 0, u32::MAX, &msg, 0xA5C3_7E19);
        assert_eq!(
            hex(&pk.to_bytes()),
            "b60df5557dd72654da9b26eb5aee454fbb8b8839d80d6d0ab29a4226fc2c910f"
        );
        assert_eq!(
            hex(&blake2s256(&sig.to_bytes())),
            "76fe28fc5609099765d1289f0a75b397d8698e9abfe74d3f83bf01118e2db831"
        );
        assert_eq!(verify(XmssHash::Blake2s, &pk, &msg, &sig, None), Ok(()));
    }

    /// The same key with `H` = BLAKE3. Not a leanVM vector: it pins this
    /// implementation (checked independently by a second implementation).
    #[test]
    fn blake3_known_answer() {
        let s0: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let msg: [u8; 32] = std::array::from_fn(|i| (i * 7 + 3) as u8);
        let (pk, sig) = keygen_and_sign(XmssHash::Blake3, &s0, 0, u32::MAX, &msg, 0xA5C3_7E19);
        assert_eq!(
            hex(&pk.to_bytes()),
            "42cb7c0519c5d57e93dbff7d0cf0bffaf557c42a90739b68b944c4ed7aee60fa"
        );
        assert_eq!(
            hex(blake3::hash(&sig.to_bytes()).as_bytes()),
            "53566b9c33802a77e3ebe0d26a3d6dcda4f0180cd16bfe49986e282a1068365e"
        );
        assert_eq!(verify(XmssHash::Blake3, &pk, &msg, &sig, None), Ok(()));
    }

    #[test]
    fn verification_counts_and_rejections() {
        let pp = [7u8; N];
        let msg = [9u8; 32];
        for hash in [XmssHash::Blake3, XmssHash::Blake2s] {
            let (pk, sig) = sign_keyless(hash, &pp, 0xA5C3_7E19, &msg, 1, true);
            let (mut calls, mut compressions) = (0, 0);
            let mut count = |_: &str, input: &[u8], _: &Value| {
                calls += 1;
                compressions += hash.compressions(input.len());
            };
            assert_eq!(verify(hash, &pk, &msg, &sig, Some(&mut count)), Ok(()));
            assert_eq!((calls, compressions), (172, 201));

            let bytes = sig.to_bytes();
            assert_eq!(bytes.len(), SIG_LEN);
            assert_eq!(Signature::from_bytes(&bytes).as_ref(), Some(&sig));
            let pk_bytes = pk.to_bytes();
            let flip = |off: usize| {
                let mut b = bytes.clone();
                b[off] ^= 1;
                b
            };
            let mut other_msg = msg;
            other_msg[0] ^= 1;
            assert_eq!(
                verify_bytes(hash, &pk_bytes, &other_msg, &bytes),
                Err(VerifyError::InvalidEncoding(0))
            );
            assert_eq!(
                verify_bytes(hash, &pk_bytes, &msg, &flip(4)),
                Err(VerifyError::InvalidEncoding(0))
            );
            assert_eq!(
                verify_bytes(hash, &pk_bytes, &msg, &flip(4 + N)),
                Err(VerifyError::InvalidEncoding(1))
            );
            assert_eq!(
                verify_bytes(
                    hash,
                    &pk_bytes,
                    &msg,
                    &flip(4 + LAYER_SIG_LEN + N * (V + HEIGHT))
                ),
                Err(VerifyError::RootMismatch)
            );
            assert_eq!(
                verify_bytes(hash, &pk_bytes, &msg, &bytes[..SIG_LEN - 1]),
                Err(VerifyError::BadLength)
            );
        }
    }
}
