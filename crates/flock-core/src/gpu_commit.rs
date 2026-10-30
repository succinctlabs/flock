//! Metal GPU commit: the PCS's level-0 Reed–Solomon encoding and BLAKE3
//! Merkle tree on the Apple GPU.
//!
//! Ported from the flock-challenge tree (`gpu_commit.rs` at `c35c1c4`), where
//! the GPU commit was the largest single win of that prover. The kernels and
//! their dispatch geometry are the challenge's. Two things differ:
//! - **Any lane count.** The challenge's kernels hard-code 64 interleaved
//!   lanes and 1 KiB leaves. The union commit trims empty lanes (BLAKE3 at
//!   2^18 commits 61), so every kernel here takes the lane count `t ≤ 64`:
//!   element `(pos, lane)` sits at `pos·t + lane`, and a leaf is `16·t` bytes,
//!   one BLAKE3 chunk whose last block may be short.
//! - **Buffers.** The CPU writes the message straight into a persistent
//!   shared Metal buffer, and the GPU leaves the codeword and the tree in two
//!   more. The prover reads them in place during the open (unified memory),
//!   through [`GpuCodeword`] and [`GpuMerkleTree`]; no 1 GiB copy-out.
//!
//! The GPU graph (`log_d` = log2 positions, rate 1/2):
//! 1. a from-message pass: layers 0..4, reading the message once and
//!    synthesizing the zero half of `[z, 0]`;
//! 2. fused 4-layer passes over the codeword, the last one shorter when
//!    `log_d − 4` is not a multiple of 4;
//! 3. leaf hashing, then the parent levels, three per dispatch while they
//!    hold whole 256-node groups.
//!
//! **Hybrid (the default from `log_d = 12`).** On this M4 (10-core GPU) the
//! GPU alone is slower than the 10-thread CPU commit (about 190 against
//! 145 ms at BLAKE3 2^18). So, like the challenge tree, the commit runs on
//! both, in two phases over the same shared buffers:
//! 1. the first pass, split by tile: the GPU takes `FLOCK_METAL_GPU_FIRST16`
//!    sixteenths of the tiles (default 6), the CPU the rest;
//! 2. the rest, split by layer-4 position block (after layer 4 the 16 blocks
//!    are independent transforms with independent subtrees): the GPU takes
//!    `FLOCK_METAL_GPU_BLOCKS` of them (default 7; 16 runs the whole graph on
//!    the GPU), the CPU the rest. The CPU runs main's tuned NTT on each of
//!    its blocks through [`crate::ntt::AdditiveNttF128::sub_block`] and
//!    hashes finished sub-groups from the deep-pass hook.
//!
//! The CPU hashes the last 7 levels of the tree after the join. Every
//! boundary between what the CPU and the GPU write is a whole 256-byte span,
//! so the two never share a cache line.
//!
//! **Switches.** Build with `--features metal` (macOS/aarch64; elsewhere the
//! feature is inert). At runtime the GPU takes every eligible commit unless
//! `FLOCK_METAL=0`. Eligible: BLAKE3 Merkle, rate 1/2, at most 64 lanes, at
//! least [`MIN_CODEWORD_LOG`] codeword words, and the buffers not held by an
//! earlier commit's prover data.
//!
//! **Correctness.** The first GPU commit of each shape in a process also runs
//! the CPU commit and compares codeword and tree byte for byte. A mismatch
//! turns the GPU off for the rest of the process and the CPU result is used.
//! A GPU error falls back to the CPU from the message buffer, which the GPU
//! never writes.

use crate::field::F128;
use crate::merkle::Hash;
use crate::pcs::commit::PcsParams;

/// Runtime switch: `FLOCK_METAL=0` keeps every commit on the CPU in a build
/// with the `metal` feature.
pub const ENV_METAL: &str = "FLOCK_METAL";

/// Smallest codeword (log2 of F128 words) the GPU takes. Below it the
/// dispatch and synchronization overheads are not worth paying.
pub const MIN_CODEWORD_LOG: usize = 20;

/// Whether this build has the Metal commit and the runtime switch allows it.
pub fn metal_enabled() -> bool {
    cfg!(all(
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    )) && std::env::var_os(ENV_METAL).is_none_or(|v| v != "0")
}

/// The shapes the GPU graph covers: BLAKE3 leaves, rate 1/2, at most 64
/// lanes (one BLAKE3 chunk per leaf), and a domain with a full from-message
/// pass plus grouped tiles (`log_d ≥ 6`).
pub(crate) fn gpu_shape(params: &PcsParams) -> bool {
    let t = params.num_ntts();
    params.merkle_hash == crate::merkle::HashKind::Blake3
        && params.log_inv_rate == 1
        && (1..=64).contains(&t)
        && params.k_code() >= 6
}

/// Flat breadth-first twiddle table for `log_d` layers: layer `l` occupies
/// `[2^l − 1, 2^(l+1) − 1)`, block `b` of layer `l` at `2^l − 1 + b`.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) fn flat_twiddle_table(log_d: usize) -> Vec<F128> {
    let ntt = crate::ntt::AdditiveNttF128::standard(log_d);
    let mut out = Vec::with_capacity((1usize << log_d) - 1);
    for layer in 0..log_d {
        for block in 0..1usize << layer {
            out.push(ntt.twiddle(layer, block));
        }
    }
    out
}

/// Group the layers `[start_layer, log_d)` into fused passes of at most 4
/// layers. Each pass reads and writes the codeword once.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) fn plan_passes(log_d: usize, start_layer: usize) -> Vec<(usize, usize)> {
    let mut passes = Vec::new();
    let mut l = start_layer;
    while l < log_d {
        let f = (log_d - l).min(4);
        passes.push((l, f));
        l += f;
    }
    passes
}

/// Upper bound on the bit length of any twiddle at `layer` of a size-`2^log_d`
/// additive NTT in the standard basis.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) fn max_twiddle_bits(log_d: usize, layer: usize) -> u32 {
    if layer == 0 || layer >= log_d {
        return 0;
    }
    let shift = log_d - layer - 1;
    if shift >= 32 {
        return u32::MAX;
    }
    match (layer as u64).checked_mul(1u64 << shift) {
        Some(d) if d < u32::MAX as u64 - 1 => d as u32 + 1,
        _ => u32::MAX,
    }
}

/// The mixed final-pass kernel multiplies its last two sub-layers by
/// short twiddles with fixed 40- and 20-bit loops; this is its correctness
/// gate.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) fn pass5_mixed_ok(log_d: usize, l: usize, f: usize) -> bool {
    f == 4
        && l + 4 == log_d
        && max_twiddle_bits(log_d, l + 2) <= 40
        && max_twiddle_bits(log_d, l + 3) <= 20
}

/// The outcome of a commit the GPU took.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) enum GpuCommit {
    /// Codeword and tree in the GPU's buffers.
    Gpu(GpuCodeword, GpuMerkleTree),
    /// The GPU failed or disagreed with the CPU; the CPU result.
    Cpu(Vec<F128>, Vec<Hash>),
}

/// Commit on the GPU, or `None` to leave the commit to the CPU (GPU off or
/// unavailable, shape not covered, codeword below [`MIN_CODEWORD_LOG`], or
/// the buffers still held by an earlier commit's prover data).
///
/// `fill(msg, positions)` writes those positions of the message
/// (`t·2^(log_d−1)` words, element `(pos, lane)` at `pos·t + lane`, the CPU
/// commit's replicate-fill source); the commit fills the GPU's share first
/// and the rest while the GPU runs. `cpu` computes the CPU codeword and tree
/// from the whole message; it runs on the first commit of each shape, and as
/// the fallback after a GPU error.
pub(crate) fn try_commit(
    params: &PcsParams,
    fill: impl Fn(&mut [F128], core::ops::Range<usize>),
    cpu: impl FnOnce(&[F128]) -> (Vec<F128>, Vec<Hash>),
) -> Option<GpuCommit> {
    if !metal_enabled() || !gpu_shape(params) {
        return None;
    }
    let words = params.n_positions() * params.num_ntts();
    if words < 1usize << MIN_CODEWORD_LOG {
        return None;
    }
    imp::try_commit(params.k_code(), params.num_ntts(), fill, cpu)
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use imp::{GpuCodeword, GpuMerkleTree};

/// The codeword of a GPU commit. Uninhabited in builds without Metal.
#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
pub enum GpuCodeword {}

/// The Merkle tree of a GPU commit. Uninhabited in builds without Metal.
#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
pub enum GpuMerkleTree {}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
impl core::ops::Deref for GpuCodeword {
    type Target = [F128];
    fn deref(&self) -> &[F128] {
        match *self {}
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
impl core::ops::Deref for GpuMerkleTree {
    type Target = [Hash];
    fn deref(&self) -> &[Hash] {
        match *self {}
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
mod imp {
    use super::{F128, GpuCommit, Hash};

    pub(super) fn try_commit(
        _log_d: usize,
        _t: usize,
        _fill: impl Fn(&mut [F128], core::ops::Range<usize>),
        _cpu: impl FnOnce(&[F128]) -> (Vec<F128>, Vec<Hash>),
    ) -> Option<GpuCommit> {
        None
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use super::{F128, GpuCommit, Hash, flat_twiddle_table, pass5_mixed_ok, plan_passes};
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex, OnceLock};

    // -----------------------------------------------------------------------
    // Minimal Objective-C / Metal FFI: dlopen + objc_msgSend, no crate deps.
    // -----------------------------------------------------------------------

    pub(crate) type Id = *mut c_void;
    type Sel = *mut c_void;

    unsafe extern "C" {
        fn dlopen(path: *const i8, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
    }
    const RTLD_NOW: i32 = 2;

    pub(crate) const NIL: Id = std::ptr::null_mut();

    /// Function pointers resolved from libobjc and Metal at init.
    pub(crate) struct Api {
        msg_send: *const c_void,
        get_class: unsafe extern "C" fn(*const i8) -> Id,
        sel_register: unsafe extern "C" fn(*const i8) -> Sel,
        pool_push: unsafe extern "C" fn() -> *mut c_void,
        pool_pop: unsafe extern "C" fn(*mut c_void),
        create_system_default_device: unsafe extern "C" fn() -> Id,
        copy_all_devices: unsafe extern "C" fn() -> Id,
    }
    // SAFETY: all fields are process-global immutable function pointers.
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}

    /// `objc_msgSend` cast to a concrete signature per call site.
    macro_rules! send {
        ($api:expr, $ty:ty, $obj:expr, $sel:expr $(, $a:expr)* $(,)?) => {{
            let f: $ty = core::mem::transmute($api.msg_send);
            f($obj, ($api.sel_register)($sel.as_ptr()) $(, $a)*)
        }};
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MtlSize {
        width: u64,
        height: u64,
        depth: u64,
    }

    impl Api {
        fn load() -> Result<Api, String> {
            unsafe {
                let objc = dlopen(c"/usr/lib/libobjc.A.dylib".as_ptr().cast(), RTLD_NOW);
                if objc.is_null() {
                    return Err("dlopen libobjc failed".into());
                }
                // Foundation first (registers NSString), then Metal.
                let foundation = dlopen(
                    c"/System/Library/Frameworks/Foundation.framework/Foundation"
                        .as_ptr()
                        .cast(),
                    RTLD_NOW,
                );
                if foundation.is_null() {
                    return Err("dlopen Foundation failed".into());
                }
                let metal = dlopen(
                    c"/System/Library/Frameworks/Metal.framework/Metal"
                        .as_ptr()
                        .cast(),
                    RTLD_NOW,
                );
                if metal.is_null() {
                    return Err("dlopen Metal failed".into());
                }
                let sym = |h: *mut c_void, name: &core::ffi::CStr| -> Result<*mut c_void, String> {
                    let p = dlsym(h, name.as_ptr());
                    if p.is_null() {
                        Err(format!("dlsym {name:?} failed"))
                    } else {
                        Ok(p)
                    }
                };
                type ClassFn = unsafe extern "C" fn(*const i8) -> Id;
                type SelFn = unsafe extern "C" fn(*const i8) -> Sel;
                type PushFn = unsafe extern "C" fn() -> *mut c_void;
                type PopFn = unsafe extern "C" fn(*mut c_void);
                type DeviceFn = unsafe extern "C" fn() -> Id;
                Ok(Api {
                    msg_send: sym(objc, c"objc_msgSend")?,
                    get_class: core::mem::transmute::<*mut c_void, ClassFn>(sym(
                        objc,
                        c"objc_getClass",
                    )?),
                    sel_register: core::mem::transmute::<*mut c_void, SelFn>(sym(
                        objc,
                        c"sel_registerName",
                    )?),
                    pool_push: core::mem::transmute::<*mut c_void, PushFn>(sym(
                        objc,
                        c"objc_autoreleasePoolPush",
                    )?),
                    pool_pop: core::mem::transmute::<*mut c_void, PopFn>(sym(
                        objc,
                        c"objc_autoreleasePoolPop",
                    )?),
                    create_system_default_device: core::mem::transmute::<*mut c_void, DeviceFn>(
                        sym(metal, c"MTLCreateSystemDefaultDevice")?,
                    ),
                    copy_all_devices: core::mem::transmute::<*mut c_void, DeviceFn>(sym(
                        metal,
                        c"MTLCopyAllDevices",
                    )?),
                })
            }
        }

        unsafe fn nsstring(&self, s: &str) -> Result<Id, String> {
            // NSString stringWithUTF8String: (autoreleased).
            unsafe {
                let cls = (self.get_class)(c"NSString".as_ptr().cast());
                if cls.is_null() {
                    return Err("NSString class not found".into());
                }
                let mut buf = Vec::with_capacity(s.len() + 1);
                buf.extend_from_slice(s.as_bytes());
                buf.push(0);
                let ns: Id = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel, *const u8) -> Id,
                    cls,
                    c"stringWithUTF8String:",
                    buf.as_ptr()
                );
                if ns.is_null() {
                    Err("NSString creation failed".into())
                } else {
                    Ok(ns)
                }
            }
        }

        unsafe fn error_string(&self, err: Id) -> String {
            if err.is_null() {
                return "unknown error (nil NSError)".into();
            }
            unsafe {
                let desc: Id = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    err,
                    c"localizedDescription"
                );
                if desc.is_null() {
                    return "unknown error (nil description)".into();
                }
                let cstr: *const u8 = send!(
                    self,
                    unsafe extern "C" fn(Id, Sel) -> *const u8,
                    desc,
                    c"UTF8String"
                );
                if cstr.is_null() {
                    return "unknown error (nil UTF8String)".into();
                }
                std::ffi::CStr::from_ptr(cstr.cast())
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    // -----------------------------------------------------------------------
    // Metal Shading Language kernels.
    // -----------------------------------------------------------------------

    const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// GF(2^128), GHASH polynomial P = x^128 + x^7 + x^2 + x + 1.
//
// F128 memory layout (little-endian struct { uint64 lo; uint64 hi; }):
// uint4 v = (lo31..0, lo63..32, hi31..0, hi63..32); bit i of the field
// element is bit (i mod 32) of word i/32.
// ===========================================================================

// v * x mod P.
static inline uint4 gf_mulx(uint4 v) {
    uint carry = v.w >> 31;
    uint4 r;
    r.w = (v.w << 1) | (v.z >> 31);
    r.z = (v.z << 1) | (v.y >> 31);
    r.y = (v.y << 1) | (v.x >> 31);
    r.x = (v.x << 1) ^ (carry * 0x87u);
    return r;
}

// a * x^8 mod P. The 8 bits shifted out (h) fold back as h * (x^7+x^2+x+1),
// which spans at most bit 14 and lands entirely in the low word.
static inline uint4 gf_shl8(uint4 a) {
    uint h = a.w >> 24;
    uint4 r;
    r.w = (a.w << 8) | (a.z >> 24);
    r.z = (a.z << 8) | (a.y >> 24);
    r.y = (a.y << 8) | (a.x >> 24);
    r.x = (a.x << 8) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h);
    return r;
}

// v * tw mod P via byte-wise Horner over v, using the twiddle's reduced
// nibble-multiple tables: tab[n] = n*tw, tab[16+n] = (n*x^4)*tw (n = 0..15).
static inline uint4 gf_mul_tab(uint4 v, threadgroup const uint4* tab) {
    uint4 acc = uint4(0u);
    for (int i = 15; i >= 0; i--) {
        acc = gf_shl8(acc);
        uint b = (v[i >> 2] >> ((i & 3) * 8)) & 0xffu;
        acc ^= tab[b & 15u] ^ tab[16u + (b >> 4)];
    }
    return acc;
}

// a * x^16 mod P. The 16 bits shifted out fold back as h * 0x87 (<= bit 22).
static inline uint4 gf_shl16(uint4 a) {
    uint h = a.w >> 16;
    uint4 r;
    r.w = (a.w << 16) | (a.z >> 16);
    r.z = (a.z << 16) | (a.y >> 16);
    r.y = (a.y << 16) | (a.x >> 16);
    r.x = (a.x << 16) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h);
    return r;
}

// v * tw mod P, 16 bits of v per Horner step, using four reduced nibble
// tables: tab[16k + n] = (n * x^(4k)) * tw for k = 0..3, n = 0..15.
static inline uint4 gf_mul_tab4(uint4 v, threadgroup const uint4* tab) {
    uint4 acc = uint4(0u);
    for (int i = 7; i >= 0; i--) {
        acc = gf_shl16(acc);
        uint h = (v[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
        acc ^= tab[h & 15u]
             ^ tab[16u + ((h >> 4) & 15u)]
             ^ tab[32u + ((h >> 8) & 15u)]
             ^ tab[48u + (h >> 12)];
    }
    return acc;
}

// v * tw mod P via two 256-entry byte tables, tab[b] = b*tw and
// tab[256 + b] = (b * x^8) * tw: 2 threadgroup loads per Horner step.
static inline uint4 gf_mul_byte16(uint4 v, threadgroup const uint4* tab) {
    uint4 acc = uint4(0u);
    for (int i = 7; i >= 0; i--) {
        acc = gf_shl16(acc);
        uint h = (v[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
        acc ^= tab[h & 255u] ^ tab[256u + (h >> 8)];
    }
    return acc;
}

// Rebuild `tabB` for the twiddle at `twiddles[idx]`, in registers. 64 threads:
// `lid` owns a 4-lo x 2-hi block of one byte table (t = lid >> 5 selects tw or
// tw * x^8). The leading barrier retires the previous table's readers; the
// trailing one publishes the new one.
static inline void gf_build_byte16(
    threadgroup uint4* tabB, device const uint4* twiddles, uint idx, uint lid)
{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t  = lid >> 5;
    const uint u  = lid & 31u;
    const uint aa = u & 3u;
    const uint cc = u >> 2;
    uint4 g = twiddles[idx];
    if (t != 0u) {
        for (uint m = 0; m < 8u; m++) { g = gf_mulx(g); }
    }
    uint4 e0 = g;
    uint4 e1 = gf_mulx(e0), e2 = gf_mulx(e1), e3 = gf_mulx(e2);
    uint4 f0 = gf_mulx(e3), f1 = gf_mulx(f0), f2 = gf_mulx(f1), f3 = gf_mulx(f2);
    uint4 A = uint4(0u);
    if (aa & 1u) { A ^= e2; }
    if (aa & 2u) { A ^= e3; }
    const uint4 L0 = A, L1 = A ^ e0, L2 = A ^ e1, L3 = A ^ e0 ^ e1;
    uint4 C = uint4(0u);
    if (cc & 1u) { C ^= f1; }
    if (cc & 2u) { C ^= f2; }
    if (cc & 4u) { C ^= f3; }
    const uint4 H0 = C, H1 = C ^ f0;
    threadgroup uint4* dst = &tabB[t << 8];
    const uint o0 = (cc << 5) + (aa << 2);
    const uint o1 = o0 + 16u;
    dst[o0 + 0u] = L0 ^ H0;
    dst[o0 + 1u] = L1 ^ H0;
    dst[o0 + 2u] = L2 ^ H0;
    dst[o0 + 3u] = L3 ^ H0;
    dst[o1 + 0u] = L0 ^ H1;
    dst[o1 + 1u] = L1 ^ H1;
    dst[o1 + 2u] = L2 ^ H1;
    dst[o1 + 3u] = L3 ^ H1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Rebuild `tabB` for tile-local twiddle `t` from its four precomputed bases
// (phase A: tab1[16k + n] = (n * x^(4k)) * tw; phase B: the byte tables).
static inline void gf_build_byte16_from_bases(
    threadgroup uint4* tabB,
    threadgroup uint4* tab1,
    threadgroup const uint4* bases,
    uint t,
    uint lid)
{
    const uint sub = lid & 63u;
    {
        uint4 p = bases[(t << 2) | (sub >> 4)];
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4u; k++) {
            if (((sub & 15u) >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tab1[sub] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint T  = lid >> 5;
    const uint u  = lid & 31u;
    const uint aa = u & 3u;
    const uint cc = u >> 2;
    threadgroup const uint4* src = &tab1[T << 5];
    const uint4 L0 = src[(aa << 2) + 0u];
    const uint4 L1 = src[(aa << 2) + 1u];
    const uint4 L2 = src[(aa << 2) + 2u];
    const uint4 L3 = src[(aa << 2) + 3u];
    const uint4 H0 = src[16u + (cc << 1) + 0u];
    const uint4 H1 = src[16u + (cc << 1) + 1u];
    threadgroup uint4* dst = &tabB[T << 8];
    const uint o0 = (cc << 5) + (aa << 2);
    const uint o1 = o0 + 16u;
    dst[o0 + 0u] = L0 ^ H0;
    dst[o0 + 1u] = L1 ^ H0;
    dst[o0 + 2u] = L2 ^ H0;
    dst[o0 + 3u] = L3 ^ H0;
    dst[o1 + 0u] = L0 ^ H1;
    dst[o1 + 1u] = L1 ^ H1;
    dst[o1 + 2u] = L2 ^ H1;
    dst[o1 + 3u] = L3 ^ H1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ===========================================================================
// Fused multi-layer interleaved additive-NTT butterfly passes.
//
// Layout matches AdditiveNttF128::forward_transform_interleaved with
// `P.lanes` SoA lanes: element (pos, lane) at flat index pos*lanes + lane.
// At global layer L (log_d total layers), butterflies pair positions that
// differ in position bit (log_d - L - 1); the pair's twiddle is
// twiddles[(1<<L)-1 + (pos >> (log_d - L))], shared by all lanes.
//
// One pass applies f consecutive layers l..l+f-1 to tiles of 2^f positions.
// A tile's positions share every position bit except the f pair bits, so they
// are strided by 2^s, s = log_d-l-f:
//     pos(e) = (B << (log_d-l)) + (e << s) + r,  tgid = B*2^s + r.
// Threadgroups are 64 threads, one per lane; threads with lane >= P.lanes
// help build the twiddle tables and skip every global load and store.
// ===========================================================================

struct NttParams {
    uint log_d;   // log2 of positions
    uint l;       // first fused layer
    uint f;       // number of fused layers (1..=4)
    uint s;       // log_d - l - f
    uint lanes;   // interleaved lanes, 1..=64
};

#define NTT_MAX_F 4u
#define AT(pos, lane) ((pos) * P.lanes + (lane))

// Generic staged pass for f = 1, 2: 2^(f-1) threads per lane.
kernel void ntt_fused(device uint4* data                [[buffer(0)]],
                      device const uint4* twiddles      [[buffer(1)]],
                      constant NttParams& P             [[buffer(2)]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint lid  [[thread_index_in_threadgroup]])
{
    threadgroup uint4 tile[(1u << NTT_MAX_F) * 64u];
    threadgroup uint4 tabs[((1u << NTT_MAX_F) - 1u) * 32u];

    const uint lane = lid & 63u;
    const bool live = lane < P.lanes;
    const uint tid  = lid >> 6;
    const uint nf   = 1u << P.f;
    const uint nhalf = nf >> 1;
    const uint B    = tgid >> P.s;
    const uint r    = tgid & ((1u << P.s) - 1u);
    const uint pos_base = (B << (P.log_d - P.l)) + r;

    for (uint e = tid; e < nf; e += nhalf) {
        tile[(e << 6) + lane] = live ? data[AT(pos_base + (e << P.s), lane)] : uint4(0u);
    }

    // Reduced nibble tables for the tile's 2^f - 1 twiddles. Tile-local
    // twiddle t (heap order) = sub-layer j = floor(log2(t+1)), in-layer index
    // c = t+1-2^j, global twiddle twiddles[(1 << (l+j)) - 1 + (B << j) + c].
    const uint n_entries = (nf - 1u) * 32u;
    for (uint ei = lid; ei < n_entries; ei += nhalf << 6) {
        uint t   = ei >> 5;
        uint sub = ei & 31u;
        uint hi  = sub >> 4;
        uint n   = sub & 15u;
        uint j   = 31u - clz(t + 1u);
        uint c   = t + 1u - (1u << j);
        uint4 tw = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];
        uint4 p = tw;
        if (hi != 0u) {
            p = gf_mulx(gf_mulx(gf_mulx(gf_mulx(p))));
        }
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4; k++) {
            if ((n >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tabs[ei] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < P.f; j++) {
        uint bpos = P.f - 1u - j;
        uint low  = tid & ((1u << bpos) - 1u);
        uint eu   = ((tid >> bpos) << (bpos + 1u)) | low;
        uint ev   = eu | (1u << bpos);
        uint tsel = ((1u << j) - 1u) + (eu >> (P.f - j));
        uint4 u = tile[(eu << 6) + lane];
        uint4 v = tile[(ev << 6) + lane];
        uint4 nu = u ^ gf_mul_tab(v, &tabs[tsel << 5]);
        tile[(eu << 6) + lane] = nu;
        tile[(ev << 6) + lane] = nu ^ v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (live) {
        for (uint e = tid; e < nf; e += nhalf) {
            data[AT(pos_base + (e << P.s), lane)] = tile[(e << 6) + lane];
        }
    }
}

// Register-resident f = 4 and f = 3 passes: one thread owns all 2^f tile
// positions of one lane in registers; the twiddles' four reduced nibble
// tables (gf_mul_tab4) are built cooperatively. The f loops have
// compile-time bounds, so elems[] stays in registers.
#define DEF_NTT_FUSED_REG(NAME, F_CONST)                                       \
kernel void NAME(device uint4* data                [[buffer(0)]],              \
                 device const uint4* twiddles      [[buffer(1)]],              \
                 constant NttParams& P             [[buffer(2)]],              \
                 uint tgid [[threadgroup_position_in_grid]],                   \
                 uint lid  [[thread_index_in_threadgroup]])                    \
{                                                                              \
    constexpr uint F   = F_CONST;                                              \
    constexpr uint NF  = 1u << F;                                              \
    constexpr uint NTW = NF - 1u;                                              \
    threadgroup uint4 bases[NTW * 4u];                                         \
    threadgroup uint4 tabs[NTW * 64u];                                         \
                                                                               \
    const uint lane = lid;                                                     \
    const bool live = lane < P.lanes;                                          \
    const uint B = tgid >> P.s;                                                \
    const uint r = tgid & ((1u << P.s) - 1u);                                  \
                                                                               \
    if (lid < NTW * 4u) {                                                      \
        uint t = lid >> 2;                                                     \
        uint k = lid & 3u;                                                     \
        uint j = 31u - clz(t + 1u);                                            \
        uint c = t + 1u - (1u << j);                                           \
        uint4 p = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];             \
        for (uint m = 0; m < k * 4u; m++) { p = gf_mulx(p); }                  \
        bases[lid] = p;                                                        \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint ei = lid; ei < NTW * 64u; ei += 64u) {                           \
        uint t   = ei >> 6;                                                    \
        uint sub = ei & 63u;                                                   \
        uint n   = sub & 15u;                                                  \
        uint4 p  = bases[(t << 2) | (sub >> 4)];                               \
        uint4 val = uint4(0u);                                                 \
        for (uint k = 0; k < 4u; k++) {                                        \
            if ((n >> k) & 1u) { val ^= p; }                                   \
            p = gf_mulx(p);                                                    \
        }                                                                      \
        tabs[ei] = val;                                                        \
    }                                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    if (!live) { return; }                                                     \
                                                                               \
    const uint pos_base = (B << (P.log_d - P.l)) + r;                          \
    uint4 elems[NF];                                                           \
    for (uint e = 0; e < NF; e++) {                                            \
        elems[e] = data[AT(pos_base + (e << P.s), lane)];                      \
    }                                                                          \
    for (uint j = 0; j < F; j++) {                                             \
        uint bpos = F - 1u - j;                                                \
        for (uint b = 0; b < (NF >> 1); b++) {                                 \
            uint low = b & ((1u << bpos) - 1u);                                \
            uint eu  = ((b >> bpos) << (bpos + 1u)) | low;                     \
            uint ev  = eu | (1u << bpos);                                      \
            uint tsel = ((1u << j) - 1u) + (eu >> (F - j));                    \
            uint4 nu = elems[eu] ^ gf_mul_tab4(elems[ev], &tabs[tsel << 6]);   \
            elems[eu] = nu;                                                    \
            elems[ev] ^= nu;                                                   \
        }                                                                      \
    }                                                                          \
    for (uint e = 0; e < NF; e++) {                                            \
        data[AT(pos_base + (e << P.s), lane)] = elems[e];                      \
    }                                                                          \
}

DEF_NTT_FUSED_REG(ntt_fused_reg4, 4u)
DEF_NTT_FUSED_REG(ntt_fused_reg3, 3u)

// ===========================================================================
// The production f = 4, s >= 2 pass: four same-B tiles per 64-thread group
// share one twiddle set (table reuse), and the multiply is gf_mul_byte16 over
// byte tables rebuilt per twiddle (8 KiB each, so only the live one is
// resident). `pad` is never read; it sizes the group's threadgroup
// allocation for the residency this pass wants.
// ===========================================================================

#define NTT_G4_B16_SUBLAYER(J)                                                 \
    {                                                                          \
        constexpr uint j    = (J);                                             \
        constexpr uint bpos = F - 1u - j;                                      \
        constexpr uint NC   = 1u << j;                                         \
        constexpr uint BPER = (NF >> 1) >> j;                                  \
        for (uint c = 0; c < NC; c++) {                                        \
            gf_build_byte16(tabB, twiddles,                                    \
                            (1u << (P.l + j)) - 1u + (B << j) + c, lid);       \
            for (uint bb = 0; bb < BPER; bb++) {                               \
                const uint b   = c * BPER + bb;                                \
                const uint low = b & ((1u << bpos) - 1u);                      \
                const uint eu  = ((b >> bpos) << (bpos + 1u)) | low;           \
                const uint ev  = eu | (1u << bpos);                            \
                uint4 nu = elems[eu] ^ gf_mul_byte16(elems[ev], &tabB[0]);     \
                elems[eu] = nu;                                                \
                elems[ev] ^= nu;                                               \
            }                                                                  \
        }                                                                      \
    }

kernel void ntt_fused_reg4g4(device uint4* data                [[buffer(0)]],
                             device const uint4* twiddles      [[buffer(1)]],
                             constant NttParams& P             [[buffer(2)]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F     = 4u;
    constexpr uint NF    = 1u << F;
    constexpr uint LOG_G = 2u;
    threadgroup uint4 tabB[512u];
    threadgroup uint4 pad[124u];

    const uint B = tgid >> (P.s - LOG_G);
    const uint r_base = (tgid & ((1u << (P.s - LOG_G)) - 1u)) << LOG_G;

    // The group's 4 same-B tiles x `lanes` lanes are flat work items, 64 per
    // round (3 rounds instead of 4 at 46 lanes). The round count is the same
    // for every thread, since the table rebuilds inside need all 64 at their
    // barriers; past the last item a thread runs the network on zeros and
    // skips the loads and stores.
    const uint items = P.lanes << LOG_G;
    for (uint it = 0; it < (items + 63u) >> 6; it++) {
        const uint item = (it << 6) + lid;
        const bool live = item < items;
        const uint rr = live ? item / P.lanes : 0u;
        const uint lane = live ? item - rr * P.lanes : 0u;
        const uint r = r_base + rr;
        const uint pos_base = (B << (P.log_d - P.l)) + r;
        uint4 elems[NF];
        for (uint e = 0; e < NF; e++) {
            elems[e] = live ? data[AT(pos_base + (e << P.s), lane)] : uint4(0u);
        }
        NTT_G4_B16_SUBLAYER(0u)
        NTT_G4_B16_SUBLAYER(1u)
        NTT_G4_B16_SUBLAYER(2u)
        NTT_G4_B16_SUBLAYER(3u)
        if (P.log_d == 77u) {          /* unreachable; keeps `pad` allocated */
            pad[lid % 124u] = elems[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            elems[0] ^= pad[(lid + 1u) % 124u];
        }
        if (live) {
            for (uint e = 0; e < NF; e++) {
                data[AT(pos_base + (e << P.s), lane)] = elems[e];
            }
        }
    }
}

// Final pass with s < 2 (every tile its own block, no table reuse): 32-entry
// byte-Horner tables (gf_mul_tab), half the threadgroup footprint of the
// 16-bit-Horner ones, so twice the tiles fit a core.
kernel void ntt_fused_reg4h8(device uint4* data                [[buffer(0)]],
                             device const uint4* twiddles      [[buffer(1)]],
                             constant NttParams& P             [[buffer(2)]],
                             uint tgid [[threadgroup_position_in_grid]],
                             uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F   = 4u;
    constexpr uint NF  = 1u << F;
    constexpr uint NTW = NF - 1u;
    threadgroup uint4 tabs[NTW * 32u];

    const uint lane = lid & 63u;
    const bool live = lane < P.lanes;
    const uint B = tgid >> P.s;
    const uint r = tgid & ((1u << P.s) - 1u);
    const uint pos_base = (B << (P.log_d - P.l)) + r;

    for (uint ei = lid; ei < NTW * 32u; ei += 64u) {
        uint t   = ei >> 5;
        uint sub = ei & 31u;
        uint hi  = sub >> 4;
        uint n   = sub & 15u;
        uint j   = 31u - clz(t + 1u);
        uint c   = t + 1u - (1u << j);
        uint4 p = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];
        if (hi != 0u) {
            p = gf_mulx(gf_mulx(gf_mulx(gf_mulx(p))));
        }
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4u; k++) {
            if ((n >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tabs[ei] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!live) { return; }

    uint4 elems[NF];
    for (uint e = 0; e < NF; e++) {
        elems[e] = data[AT(pos_base + (e << P.s), lane)];
    }
    for (uint j = 0; j < F; j++) {
        uint bpos = F - 1u - j;
        for (uint b = 0; b < (NF >> 1); b++) {
            uint low = b & ((1u << bpos) - 1u);
            uint eu  = ((b >> bpos) << (bpos + 1u)) | low;
            uint ev  = eu | (1u << bpos);
            uint tsel = ((1u << j) - 1u) + (eu >> (F - j));
            uint4 nu = elems[eu] ^ gf_mul_tab(elems[ev], &tabs[tsel << 5]);
            elems[eu] = nu;
            elems[ev] ^= nu;
        }
    }
    for (uint e = 0; e < NF; e++) {
        data[AT(pos_base + (e << P.s), lane)] = elems[e];
    }
}

// a * x^4 mod P.
static inline uint4 gf_shl4(uint4 a) {
    uint h = a.w >> 28;
    uint4 r;
    r.w = (a.w << 4) | (a.z >> 28);
    r.z = (a.z << 4) | (a.y >> 28);
    r.y = (a.y << 4) | (a.x >> 28);
    r.x = (a.x << 4) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h);
    return r;
}

#define NTT_MIX_B16_SUBLAYER(J, C)                                             \
    {                                                                          \
        constexpr uint j    = (J);                                             \
        constexpr uint c    = (C);                                             \
        constexpr uint bpos = F - 1u - j;                                      \
        constexpr uint BPER = (NF >> 1) >> j;                                  \
        gf_build_byte16_from_bases(tabB, tab1, bases,                          \
                                   ((1u << j) - 1u) + c, lid);                 \
        for (uint bb = 0; bb < BPER; bb++) {                                   \
            const uint b   = c * BPER + bb;                                    \
            const uint low = b & ((1u << bpos) - 1u);                          \
            const uint eu  = ((b >> bpos) << (bpos + 1u)) | low;               \
            const uint ev  = eu | (1u << bpos);                                \
            uint4 nu = elems[eu] ^ gf_mul_byte16(elems[ev], &tabB[0]);         \
            elems[eu] = nu;                                                    \
            elems[ev] ^= nu;                                                   \
        }                                                                      \
    }

// Mixed final pass (l = log_d - 4): the shallow sub-layers multiply by byte
// tables rebuilt per twiddle; the deep two Horner over the short twiddle
// (< 2^40 and < 2^20 bits, checked by pass5_mixed_ok) and keep no table.
kernel void ntt_pass5_mixed(device uint4* data                [[buffer(0)]],
                            device const uint4* twiddles      [[buffer(1)]],
                            constant NttParams& P             [[buffer(2)]],
                            uint tgid [[threadgroup_position_in_grid]],
                            uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F = 4u, NF = 1u << F;
    constexpr uint NNIB_A = 10u;   // sub-layer 2: twiddle < 2^40
    constexpr uint NNIB_B = 5u;    // sub-layer 3: twiddle < 2^20
    threadgroup uint4 bases[3u * 4u];
    threadgroup uint4 tab1[64u];
    threadgroup uint4 tabB[512u];
    threadgroup uint  nibA[4u * NNIB_A];
    threadgroup uint  nibB[8u * NNIB_B];

    const uint lane = lid & 63u;
    const bool live = lane < P.lanes;
    const uint B = tgid >> P.s;
    const uint r = tgid & ((1u << P.s) - 1u);
    const uint pos_base = (B << (P.log_d - P.l)) + r;

    if (lid < 12u) {
        uint t = lid >> 2, k = lid & 3u;
        uint j = 31u - clz(t + 1u), c = t + 1u - (1u << j);
        uint4 p = twiddles[(1u << (P.l + j)) - 1u + (B << j) + c];
        for (uint m = 0; m < k * 4u; m++) p = gf_mulx(p);
        bases[lid] = p;
    }
    if (lid < 40u) {
        uint cA = lid / NNIB_A, qA = lid % NNIB_A;
        uint4 twA = twiddles[(1u << (P.l + 2u)) - 1u + (B << 2) + cA];
        nibA[lid] = (twA[qA >> 3] >> ((qA & 7u) * 4u)) & 15u;
        uint cB = lid / NNIB_B, qB = lid % NNIB_B;
        uint4 twB = twiddles[(1u << (P.l + 3u)) - 1u + (B << 3) + cB];
        nibB[lid] = (twB[qB >> 3] >> ((qB & 7u) * 4u)) & 15u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dead lanes take part in the per-twiddle table rebuilds (barriers).
    uint4 elems[NF];
    for (uint e = 0; e < NF; e++) {
        elems[e] = live ? data[AT(pos_base + (e << P.s), lane)] : uint4(0u);
    }
    NTT_MIX_B16_SUBLAYER(0u, 0u)
    NTT_MIX_B16_SUBLAYER(1u, 0u)
    NTT_MIX_B16_SUBLAYER(1u, 1u)
    if (!live) { return; }
    for (uint j = 2u; j < F; j++) {
        const uint bpos = F - 1u - j;
        for (uint b = 0; b < (NF >> 1); b++) {
            uint low = b & ((1u << bpos) - 1u);
            uint eu = ((b >> bpos) << (bpos + 1u)) | low;
            uint ev = eu | (1u << bpos);
            uint c = eu >> (F - j);
            const uint NN = (j == 2u) ? NNIB_A : NNIB_B;
            threadgroup const uint* nb =
                (j == 2u) ? &nibA[c * NNIB_A] : &nibB[c * NNIB_B];
            uint4 V0 = elems[ev];
            uint4 V1 = gf_mulx(V0), V2 = gf_mulx(V1), V3 = gf_mulx(V2);
            uint4 acc = uint4(0u);
            for (int q = (int)NN - 1; q >= 0; q--) {
                acc = gf_shl4(acc);
                uint n = nb[q];
                if (n & 1u) acc ^= V0;
                if (n & 2u) acc ^= V1;
                if (n & 4u) acc ^= V2;
                if (n & 8u) acc ^= V3;
            }
            uint4 nu = elems[eu] ^ acc;
            elems[eu] = nu;
            elems[ev] ^= nu;
        }
    }
    for (uint e = 0; e < NF; e++) {
        data[AT(pos_base + (e << P.s), lane)] = elems[e];
    }
}

// ===========================================================================
// From-message first pass: layers 0..3 of the rate-1/2 encoding of [z, 0].
// With l = 0 and f = 4 a tile's top e-bit is the codeword's top position
// bit, so the tile's upper half is the zero region and its lower half is the
// message. The pass reads z once, writes the post-layer-3 codeword out of
// place (z is never written), and skips sub-layer 0, which pairs (e, e+8)
// across the zero half and is a copy. One group shares the (block-0)
// twiddle tables across four consecutive tiles.
// ===========================================================================
kernel void ntt_fused_reg4_from_zg4(device uint4* data                [[buffer(0)]],
                                    device const uint4* twiddles      [[buffer(1)]],
                                    constant NttParams& P             [[buffer(2)]],
                                    device const uint4* z             [[buffer(3)]],
                                    uint tgid [[threadgroup_position_in_grid]],
                                    uint lid  [[thread_index_in_threadgroup]])
{
    constexpr uint F   = 4u;
    constexpr uint NF  = 1u << F;
    constexpr uint NTW = NF - 1u;
    constexpr uint LOG_G = 2u;
    threadgroup uint4 bases[NTW * 4u];
    threadgroup uint4 tabs[NTW * 64u];

    const uint r_base = tgid << LOG_G;

    if (lid < NTW * 4u) {
        uint t = lid >> 2;
        uint k = lid & 3u;
        uint j = 31u - clz(t + 1u);
        uint c = t + 1u - (1u << j);
        uint4 p = twiddles[(1u << j) - 1u + c];
        for (uint m = 0; m < k * 4u; m++) { p = gf_mulx(p); }
        bases[lid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ei = lid; ei < NTW * 64u; ei += 64u) {
        uint t   = ei >> 6;
        uint sub = ei & 63u;
        uint n   = sub & 15u;
        uint4 p  = bases[(t << 2) | (sub >> 4)];
        uint4 val = uint4(0u);
        for (uint k = 0; k < 4u; k++) {
            if ((n >> k) & 1u) { val ^= p; }
            p = gf_mulx(p);
        }
        tabs[ei] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The group's 4 tiles x `lanes` lanes are flat work items, so no thread
    // idles on a dead lane (with no barrier below, the loop count may differ
    // per thread). Consecutive items stay consecutive in memory.
    for (uint item = lid; item < (P.lanes << LOG_G); item += 64u) {
        const uint rr = item / P.lanes;
        const uint lane = item - rr * P.lanes;
        const uint r = r_base + rr;
        uint4 elems[NF];
        for (uint e = 0; e < NF / 2u; e++) {
            elems[e] = z[AT((e << P.s) + r, lane)];
            elems[e + NF / 2u] = elems[e];
        }
        for (uint j = 1; j < F; j++) {
            uint bpos = F - 1u - j;
            for (uint b = 0; b < (NF >> 1); b++) {
                uint low = b & ((1u << bpos) - 1u);
                uint eu  = ((b >> bpos) << (bpos + 1u)) | low;
                uint ev  = eu | (1u << bpos);
                uint tsel = ((1u << j) - 1u) + (eu >> (F - j));
                uint4 nu = elems[eu] ^ gf_mul_tab4(elems[ev], &tabs[tsel << 6]);
                elems[eu] = nu;
                elems[ev] ^= nu;
            }
        }
        for (uint e = 0; e < NF; e++) {
            data[AT(r + (e << P.s), lane)] = elems[e];
        }
    }
}

// ===========================================================================
// BLAKE3 tree kernels.
//
// Leaf   = BLAKE3 non-root chaining value of one leaf of 16*lanes bytes: a
//          single chunk (lanes <= 64), counter 0, CHUNK_START on its first
//          block and CHUNK_END on its last, which is short and zero-padded
//          when lanes is not a multiple of 4. Matches
//          Hasher::update().finalize_non_root.
// Parent = one compression: cv = IV, block = left||right, counter 0,
//          block_len 64, flags PARENT. Matches merge_subtrees_non_root.
// Flat tree: leaves, then each parent level, root last.
// ===========================================================================

constant uint B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};
constant uchar B3_PERM[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};

#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_PARENT      4u

static void b3_compress(thread uint* cv, thread const uint* m_in,
                        uint block_len, uint flags) {
    uint v[16];
    uint m[16];
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    for (int i = 0; i < 4; i++) v[8 + i] = B3_IV[i];
    v[12] = 0u;
    v[13] = 0u;
    v[14] = block_len;
    v[15] = flags;
    for (int i = 0; i < 16; i++) m[i] = m_in[i];
    for (int r = 0; r < 7; r++) {
        #define G(a,b,c,d,x,y) \
            v[a] = v[a] + v[b] + x; v[d] = ((v[d]^v[a])>>16)|((v[d]^v[a])<<16); \
            v[c] = v[c] + v[d];     v[b] = ((v[b]^v[c])>>12)|((v[b]^v[c])<<20); \
            v[a] = v[a] + v[b] + y; v[d] = ((v[d]^v[a])>>8) |((v[d]^v[a])<<24); \
            v[c] = v[c] + v[d];     v[b] = ((v[b]^v[c])>>7) |((v[b]^v[c])<<25);
        G(0,4,8,12,  m[0], m[1]);  G(1,5,9,13,  m[2], m[3]);
        G(2,6,10,14, m[4], m[5]);  G(3,7,11,15, m[6], m[7]);
        G(0,5,10,15, m[8], m[9]);  G(1,6,11,12, m[10],m[11]);
        G(2,7,8,13,  m[12],m[13]); G(3,4,9,14,  m[14],m[15]);
        #undef G
        if (r < 6) {
            uint t[16];
            for (int i = 0; i < 16; i++) t[i] = m[B3_PERM[i]];
            for (int i = 0; i < 16; i++) m[i] = t[i];
        }
    }
    for (int i = 0; i < 8; i++) cv[i] = v[i] ^ v[8 + i];
}

kernel void leaf_hash(device const uint* codeword [[buffer(0)]],
                      device uint* out            [[buffer(1)]],
                      constant uint& lanes        [[buffer(2)]],
                      uint id [[thread_position_in_grid]])
{
    const uint words = lanes * 4u;              // 32-bit words per leaf
    const uint nblocks = (words + 15u) >> 4;    // 64-byte blocks
    device const uint* leaf = codeword + id * words;
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    for (uint b = 0; b < nblocks; b++) {
        const uint have = min(16u, words - b * 16u);
        uint block[16];
        for (uint i = 0; i < 16u; i++) block[i] = i < have ? leaf[b * 16u + i] : 0u;
        uint flags = (b == 0u ? B3_CHUNK_START : 0u) | (b + 1u == nblocks ? B3_CHUNK_END : 0u);
        b3_compress(cv, block, have * 4u, flags);
    }
    for (int i = 0; i < 8; i++) out[id * 8u + i] = cv[i];
}

kernel void parent_hash(device const uint* children [[buffer(0)]],
                        device uint* parents        [[buffer(1)]],
                        uint id [[thread_position_in_grid]])
{
    uint block[16];
    for (uint i = 0; i < 16u; i++) block[i] = children[id * 16u + i];
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    b3_compress(cv, block, 64u, B3_PARENT);
    for (int i = 0; i < 8; i++) parents[id * 8u + i] = cv[i];
}

// Three adjacent parent levels in one dispatch: a 128-thread group consumes
// 256 children and emits 128 / 64 / 32 parents into their flat-tree levels,
// keeping the intermediate levels in threadgroup memory.
kernel void parent_hash3(device const uint* children [[buffer(0)]],
                         device uint* parents1      [[buffer(1)]],
                         device uint* parents2      [[buffer(2)]],
                         device uint* parents3      [[buffer(3)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]])
{
    threadgroup uint level1[128u * 8u];
    threadgroup uint level2[64u * 8u];
    {
        uint block[16];
        const uint id = tgid * 128u + lid;
        for (uint i = 0u; i < 16u; i++) block[i] = children[id * 16u + i];
        uint cv[8];
        for (uint i = 0u; i < 8u; i++) cv[i] = B3_IV[i];
        b3_compress(cv, block, 64u, B3_PARENT);
        for (uint i = 0u; i < 8u; i++) {
            parents1[id * 8u + i] = cv[i];
            level1[lid * 8u + i] = cv[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 64u) {
        uint block[16];
        for (uint i = 0u; i < 8u; i++) {
            block[i] = level1[(2u * lid) * 8u + i];
            block[8u + i] = level1[(2u * lid + 1u) * 8u + i];
        }
        uint cv[8];
        for (uint i = 0u; i < 8u; i++) cv[i] = B3_IV[i];
        b3_compress(cv, block, 64u, B3_PARENT);
        const uint id = tgid * 64u + lid;
        for (uint i = 0u; i < 8u; i++) {
            parents2[id * 8u + i] = cv[i];
            level2[lid * 8u + i] = cv[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32u) {
        uint block[16];
        for (uint i = 0u; i < 8u; i++) {
            block[i] = level2[(2u * lid) * 8u + i];
            block[8u + i] = level2[(2u * lid + 1u) * 8u + i];
        }
        uint cv[8];
        for (uint i = 0u; i < 8u; i++) cv[i] = B3_IV[i];
        b3_compress(cv, block, 64u, B3_PARENT);
        const uint id = tgid * 32u + lid;
        for (uint i = 0u; i < 8u; i++) parents3[id * 8u + i] = cv[i];
    }
}
"#;

    // -----------------------------------------------------------------------
    // Device, queue and pipelines.
    // -----------------------------------------------------------------------

    struct Gpu {
        api: Api,
        device: Id,
        queue: Id,
        pso_ntt: Id,
        pso_ntt4: Id,
        pso_ntt3: Id,
        pso_ntt4g4: Id,
        pso_ntt4h8: Id,
        pso_ntt5mix: Id,
        pso_from_zg4: Id,
        pso_leaf: Id,
        pso_parent: Id,
        pso_parent3: Id,
    }
    // SAFETY: MTLDevice, MTLCommandQueue and MTLComputePipelineState are
    // documented thread-safe; command buffers and encoders are created and
    // used within one call.
    unsafe impl Send for Gpu {}
    unsafe impl Sync for Gpu {}

    static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();

    fn gpu() -> Result<&'static Gpu, String> {
        GPU.get_or_init(init_gpu).as_ref().map_err(|e| e.clone())
    }

    fn init_gpu() -> Result<Gpu, String> {
        let api = Api::load()?;
        unsafe {
            let pool = (api.pool_push)();
            let result = (|| -> Result<(Id, Id, [Id; 10]), String> {
                let mut device = (api.create_system_default_device)();
                if device.is_null() {
                    // Sessions without a WindowServer bootstrap (ssh, CI) get
                    // no default device; MTLCopyAllDevices still lists the
                    // built-in GPU.
                    let all = (api.copy_all_devices)();
                    if !all.is_null() {
                        device = send!(
                            api,
                            unsafe extern "C" fn(Id, Sel) -> Id,
                            all,
                            c"firstObject"
                        );
                    }
                }
                if device.is_null() {
                    return Err("no Metal device".into());
                }
                let queue: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    device,
                    c"newCommandQueue"
                );
                if queue.is_null() {
                    return Err("newCommandQueue failed".into());
                }
                let src = api.nsstring(MSL_SOURCE)?;
                let mut err: Id = NIL;
                let library: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id,
                    device,
                    c"newLibraryWithSource:options:error:",
                    src,
                    NIL,
                    &mut err
                );
                if library.is_null() {
                    return Err(format!("shader compile failed: {}", api.error_string(err)));
                }
                let pso = |name: &str| -> Result<Id, String> {
                    let ns = api.nsstring(name)?;
                    let f: Id = send!(
                        api,
                        unsafe extern "C" fn(Id, Sel, Id) -> Id,
                        library,
                        c"newFunctionWithName:",
                        ns
                    );
                    if f.is_null() {
                        return Err(format!("kernel {name} not found"));
                    }
                    let mut err: Id = NIL;
                    let p: Id = send!(
                        api,
                        unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id,
                        device,
                        c"newComputePipelineStateWithFunction:error:",
                        f,
                        &mut err
                    );
                    send!(api, unsafe extern "C" fn(Id, Sel) -> Id, f, c"release");
                    if p.is_null() {
                        return Err(format!("pipeline {name}: {}", api.error_string(err)));
                    }
                    Ok(p)
                };
                let mut psos = [NIL; 10];
                for (slot, name) in psos.iter_mut().zip(KERNELS) {
                    *slot = pso(name)?;
                }
                send!(
                    api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    library,
                    c"release"
                );
                Ok((device, queue, psos))
            })();
            (api.pool_pop)(pool);
            let (device, queue, psos) = result?;
            let [
                pso_ntt,
                pso_ntt4,
                pso_ntt3,
                pso_ntt4g4,
                pso_ntt4h8,
                pso_ntt5mix,
                pso_from_zg4,
                pso_leaf,
                pso_parent,
                pso_parent3,
            ] = psos;
            Ok(Gpu {
                api,
                device,
                queue,
                pso_ntt,
                pso_ntt4,
                pso_ntt3,
                pso_ntt4g4,
                pso_ntt4h8,
                pso_ntt5mix,
                pso_from_zg4,
                pso_leaf,
                pso_parent,
                pso_parent3,
            })
        }
    }

    /// The pipelines, in `Gpu` field order.
    const KERNELS: [&str; 10] = [
        "ntt_fused",
        "ntt_fused_reg4",
        "ntt_fused_reg3",
        "ntt_fused_reg4g4",
        "ntt_fused_reg4h8",
        "ntt_pass5_mixed",
        "ntt_fused_reg4_from_zg4",
        "leaf_hash",
        "parent_hash",
        "parent_hash3",
    ];

    impl Gpu {
        /// `newBufferWithLength:options:`, shared storage.
        unsafe fn new_buffer(&self, len: usize) -> Result<Id, String> {
            unsafe {
                let b: Id = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, u64, u64) -> Id,
                    self.device,
                    c"newBufferWithLength:options:",
                    len as u64,
                    0u64 // MTLResourceStorageModeShared
                );
                if b.is_null() {
                    Err(format!("newBufferWithLength {len} failed"))
                } else {
                    Ok(b)
                }
            }
        }

        unsafe fn contents(&self, buf: Id) -> *mut u8 {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> *mut u8,
                    buf,
                    c"contents"
                )
            }
        }

        unsafe fn release(&self, obj: Id) {
            if !obj.is_null() {
                unsafe {
                    send!(
                        self.api,
                        unsafe extern "C" fn(Id, Sel) -> Id,
                        obj,
                        c"release"
                    );
                }
            }
        }

        unsafe fn command_buffer(&self) -> Result<Id, String> {
            unsafe {
                let cb: Id = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    self.queue,
                    c"commandBuffer"
                );
                if cb.is_null() {
                    Err("commandBuffer failed".into())
                } else {
                    Ok(cb)
                }
            }
        }

        unsafe fn compute_encoder(&self, cb: Id) -> Result<Id, String> {
            unsafe {
                let e: Id = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> Id,
                    cb,
                    c"computeCommandEncoder"
                );
                if e.is_null() {
                    Err("computeCommandEncoder failed".into())
                } else {
                    Ok(e)
                }
            }
        }

        unsafe fn set_pipeline(&self, enc: Id, pso: Id) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, Id),
                    enc,
                    c"setComputePipelineState:",
                    pso
                );
            }
        }

        unsafe fn set_buffer(&self, enc: Id, buf: Id, offset: usize, index: usize) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, Id, u64, u64),
                    enc,
                    c"setBuffer:offset:atIndex:",
                    buf,
                    offset as u64,
                    index as u64
                );
            }
        }

        unsafe fn set_bytes<T>(&self, enc: Id, value: &T, index: usize) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64),
                    enc,
                    c"setBytes:length:atIndex:",
                    (value as *const T).cast(),
                    core::mem::size_of::<T>() as u64,
                    index as u64
                );
            }
        }

        unsafe fn dispatch(&self, enc: Id, groups: u64, threads_per_group: u64) {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize),
                    enc,
                    c"dispatchThreadgroups:threadsPerThreadgroup:",
                    MtlSize {
                        width: groups,
                        height: 1,
                        depth: 1
                    },
                    MtlSize {
                        width: threads_per_group,
                        height: 1,
                        depth: 1
                    }
                );
            }
        }

        unsafe fn end_encoding(&self, enc: Id) {
            unsafe {
                send!(self.api, unsafe extern "C" fn(Id, Sel), enc, c"endEncoding");
            }
        }

        unsafe fn commit_async(&self, cb: Id) {
            unsafe {
                send!(self.api, unsafe extern "C" fn(Id, Sel), cb, c"commit");
            }
        }

        /// Commit and block until completion; Err unless the status is
        /// `MTLCommandBufferStatusCompleted`.
        unsafe fn commit_and_wait(&self, cb: Id) -> Result<(), String> {
            unsafe {
                self.commit_async(cb);
                self.wait(cb)
            }
        }

        /// Block until a committed buffer completes; Err unless completed.
        unsafe fn wait(&self, cb: Id) -> Result<(), String> {
            unsafe {
                send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel),
                    cb,
                    c"waitUntilCompleted"
                );
                let status: u64 = send!(
                    self.api,
                    unsafe extern "C" fn(Id, Sel) -> u64,
                    cb,
                    c"status"
                );
                if status == 4 {
                    Ok(())
                } else {
                    let err: Id =
                        send!(self.api, unsafe extern "C" fn(Id, Sel) -> Id, cb, c"error");
                    Err(format!(
                        "command buffer status {status}: {}",
                        self.api.error_string(err)
                    ))
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // The commit graph.
    // -----------------------------------------------------------------------

    #[repr(C)]
    struct NttParams {
        log_d: u32,
        l: u32,
        f: u32,
        s: u32,
        lanes: u32,
    }

    /// Encode the from-message pass (layers 0..4) reading `msg_buf` into
    /// `code_buf`, then the fused passes for layers `[4, log_d)` in place.
    /// `mixed` allows the mixed final pass (tests turn it off to reach the
    /// half-footprint one, which production uses only from `log_d = 21`).
    unsafe fn encode_ntt(gpu: &Gpu, enc: Id, bufs: &Buffers, mixed: bool) {
        unsafe {
            encode_pass(gpu, enc, bufs, 0, 4, mixed, 16);
            for (l, f) in plan_passes(bufs.log_d, 4) {
                encode_pass(gpu, enc, bufs, l, f, mixed, 16);
            }
        }
    }

    /// Encode one fused pass of layers `[l, l + f)`; `l = 0` is the
    /// from-message pass. For `l >= 4`, only the first `blocks` of the 16
    /// layer-4 position blocks: every kernel takes its block index from the
    /// high bits of `tgid`, so a prefix of the threadgroups is exactly a
    /// prefix of the blocks.
    unsafe fn encode_pass(
        gpu: &Gpu,
        enc: Id,
        bufs: &Buffers,
        l: usize,
        f: usize,
        mixed: bool,
        blocks: u64,
    ) {
        let (log_d, lanes) = (bufs.log_d, bufs.lanes as u32);
        let s = log_d - l - f;
        let p = NttParams {
            log_d: log_d as u32,
            l: l as u32,
            f: f as u32,
            s: s as u32,
            lanes,
        };
        unsafe {
            if l == 0 {
                debug_assert_eq!(f, 4);
                gpu.set_pipeline(enc, gpu.pso_from_zg4);
                gpu.set_buffer(enc, bufs.code_buf, 0, 0);
                gpu.set_buffer(enc, bufs.tw_buf, 0, 1);
                gpu.set_bytes(enc, &p, 2);
                gpu.set_buffer(enc, bufs.msg_buf, 0, 3);
                gpu.dispatch(enc, 1u64 << (log_d - 4 - 2), 64);
                return;
            }
            // Four same-B tiles share one twiddle set while s >= 2; the final
            // 4-layer pass takes the mixed kernel where its twiddle bounds
            // hold and the half-footprint one otherwise.
            let (pso, tpg, groups) = match f {
                4 if s >= 2 => (gpu.pso_ntt4g4, 64u64, 1u64 << (log_d - f - 2)),
                4 if mixed && pass5_mixed_ok(log_d, l, f) => {
                    (gpu.pso_ntt5mix, 64, 1 << (log_d - f))
                }
                4 if s == 0 => (gpu.pso_ntt4h8, 64, 1 << (log_d - f)),
                4 => (gpu.pso_ntt4, 64, 1 << (log_d - f)),
                3 => (gpu.pso_ntt3, 64, 1 << (log_d - f)),
                _ => (gpu.pso_ntt, 1u64 << (f + 5), 1u64 << (log_d - f)),
            };
            debug_assert!(blocks == 16 || groups.is_multiple_of(16));
            gpu.set_pipeline(enc, pso);
            gpu.set_buffer(enc, bufs.code_buf, 0, 0);
            gpu.set_buffer(enc, bufs.tw_buf, 0, 1);
            gpu.set_bytes(enc, &p, 2);
            gpu.dispatch(enc, groups / 16 * blocks, tpg);
        }
    }

    /// Encode the from-message pass (layers 0..4) for the tiles `[0, r1)`
    /// only (`r1` a multiple of 4): the grouped kernel's `tgid` is `r >> 2`.
    unsafe fn encode_first_pass(gpu: &Gpu, enc: Id, bufs: &Buffers, r1: usize) {
        let log_d = bufs.log_d;
        let p = NttParams {
            log_d: log_d as u32,
            l: 0,
            f: 4,
            s: (log_d - 4) as u32,
            lanes: bufs.lanes as u32,
        };
        unsafe {
            gpu.set_pipeline(enc, gpu.pso_from_zg4);
            gpu.set_buffer(enc, bufs.code_buf, 0, 0);
            gpu.set_buffer(enc, bufs.tw_buf, 0, 1);
            gpu.set_bytes(enc, &p, 2);
            gpu.set_buffer(enc, bufs.msg_buf, 0, 3);
            gpu.dispatch(enc, (r1 / 4) as u64, 64);
        }
    }

    /// Encode the leaves `[0, n_leaves)` and their parents up to `levels`
    /// levels above the leaves, into the flat tree's slots.
    unsafe fn encode_merkle_range(
        gpu: &Gpu,
        enc: Id,
        bufs: &Buffers,
        n_leaves: usize,
        levels: usize,
    ) {
        let total = 1usize << bufs.log_d;
        // Offset of level j in the flat tree.
        let level_off = |j: usize| (0..j).map(|i| total >> i).sum::<usize>();
        // Largest power-of-two group size <= 256 that divides n.
        let tpg = |n: usize| 256u64.min(n.isolate_lowest_one() as u64);
        unsafe {
            gpu.set_pipeline(enc, gpu.pso_leaf);
            gpu.set_buffer(enc, bufs.code_buf, 0, 0);
            gpu.set_buffer(enc, bufs.tree_buf, 0, 1);
            gpu.set_bytes(enc, &(bufs.lanes as u32), 2);
            gpu.dispatch(enc, n_leaves as u64 / tpg(n_leaves), tpg(n_leaves));
            let (mut j, mut count) = (0usize, n_leaves);
            while j < levels {
                if j + 3 <= levels && count.is_multiple_of(256) {
                    gpu.set_pipeline(enc, gpu.pso_parent3);
                    for k in 0..4 {
                        gpu.set_buffer(enc, bufs.tree_buf, level_off(j + k) * 32, k);
                    }
                    gpu.dispatch(enc, (count / 256) as u64, 128);
                    j += 3;
                    count /= 8;
                } else {
                    let n_out = count / 2;
                    gpu.set_pipeline(enc, gpu.pso_parent);
                    gpu.set_buffer(enc, bufs.tree_buf, level_off(j) * 32, 0);
                    gpu.set_buffer(enc, bufs.tree_buf, level_off(j + 1) * 32, 1);
                    gpu.dispatch(enc, n_out as u64 / tpg(n_out), tpg(n_out));
                    j += 1;
                    count = n_out;
                }
            }
        }
    }

    /// Encode leaf hashing and every parent level into the flat tree.
    unsafe fn encode_merkle(gpu: &Gpu, enc: Id, bufs: &Buffers) {
        let n_leaves = 1usize << bufs.log_d;
        unsafe {
            gpu.set_pipeline(enc, gpu.pso_leaf);
            gpu.set_buffer(enc, bufs.code_buf, 0, 0);
            gpu.set_buffer(enc, bufs.tree_buf, 0, 1);
            gpu.set_bytes(enc, &(bufs.lanes as u32), 2);
            let tpg = 256u64.min(n_leaves as u64);
            gpu.dispatch(enc, n_leaves as u64 / tpg, tpg);

            let (mut level_start, mut level_len) = (0usize, n_leaves);
            gpu.set_pipeline(enc, gpu.pso_parent3);
            while level_len >= 256 {
                let l1 = level_start + level_len;
                let l2 = l1 + level_len / 2;
                let l3 = l2 + level_len / 4;
                gpu.set_buffer(enc, bufs.tree_buf, level_start * 32, 0);
                gpu.set_buffer(enc, bufs.tree_buf, l1 * 32, 1);
                gpu.set_buffer(enc, bufs.tree_buf, l2 * 32, 2);
                gpu.set_buffer(enc, bufs.tree_buf, l3 * 32, 3);
                gpu.dispatch(enc, (level_len / 256) as u64, 128);
                level_start = l3;
                level_len /= 8;
            }
            gpu.set_pipeline(enc, gpu.pso_parent);
            while level_len > 1 {
                let next = level_start + level_len;
                let n_out = level_len / 2;
                gpu.set_buffer(enc, bufs.tree_buf, level_start * 32, 0);
                gpu.set_buffer(enc, bufs.tree_buf, next * 32, 1);
                let tpg = 256u64.min(n_out as u64);
                gpu.dispatch(enc, n_out as u64 / tpg, tpg);
                level_start = next;
                level_len = n_out;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Persistent buffers and their lease.
    // -----------------------------------------------------------------------

    /// The shared buffers of one commit shape: message, codeword, tree and
    /// the flat twiddle table. Kept across commits; reallocated when the
    /// shape changes while no prover data holds them.
    #[derive(Clone, Copy)]
    struct Buffers {
        log_d: usize,
        lanes: usize,
        msg_buf: Id,
        code_buf: Id,
        tree_buf: Id,
        tw_buf: Id,
    }
    // SAFETY: the buffers are MTLBuffer objects (thread-safe); their contents
    // are only touched under the state lock or by the lease holder.
    unsafe impl Send for Buffers {}

    impl Buffers {
        fn msg_words(&self) -> usize {
            self.lanes << (self.log_d - 1)
        }
        fn code_words(&self) -> usize {
            self.lanes << self.log_d
        }
        fn tree_nodes(&self) -> usize {
            (2usize << self.log_d) - 1
        }

        unsafe fn new(gpu: &Gpu, log_d: usize, lanes: usize) -> Result<Buffers, String> {
            let mut b = Buffers {
                log_d,
                lanes,
                msg_buf: NIL,
                code_buf: NIL,
                tree_buf: NIL,
                tw_buf: NIL,
            };
            unsafe {
                let made = (|| -> Result<(), String> {
                    b.msg_buf = gpu.new_buffer(b.msg_words() * 16)?;
                    b.code_buf = gpu.new_buffer(b.code_words() * 16)?;
                    b.tree_buf = gpu.new_buffer(b.tree_nodes() * 32)?;
                    let tw = flat_twiddle_table(log_d);
                    b.tw_buf = gpu.new_buffer(tw.len().max(1) * 16)?;
                    core::ptr::copy_nonoverlapping(
                        tw.as_ptr(),
                        gpu.contents(b.tw_buf).cast(),
                        tw.len(),
                    );
                    Ok(())
                })();
                if let Err(e) = made {
                    b.release(gpu);
                    return Err(e);
                }
            }
            Ok(b)
        }

        unsafe fn release(&mut self, gpu: &Gpu) {
            unsafe {
                for buf in [self.msg_buf, self.code_buf, self.tree_buf, self.tw_buf] {
                    gpu.release(buf);
                }
            }
            self.msg_buf = NIL;
            self.code_buf = NIL;
            self.tree_buf = NIL;
            self.tw_buf = NIL;
        }
    }

    struct State {
        bufs: Option<Buffers>,
        /// Prover data from the last GPU commit still reads the buffers.
        in_use: bool,
        /// Shapes `(log_d, lanes)` whose GPU commit matched the CPU's.
        verified: Vec<(usize, usize)>,
        /// A GPU result disagreed with the CPU: no more GPU commits.
        disabled: bool,
        /// The hybrid split per shape `(log_d, lanes)`, tuned commit by commit.
        tunes: Vec<((usize, usize), Tune)>,
    }

    static STATE: Mutex<State> = Mutex::new(State {
        bufs: None,
        in_use: false,
        verified: Vec::new(),
        disabled: false,
        tunes: Vec::new(),
    });

    /// One shape's hybrid split and the per-unit costs it is balanced on: ms
    /// per sixteenth of the first pass's tiles (`g1` GPU, `c1` CPU including
    /// its share of the fill) and per layer-4 block of the rest (`g2`, `c2`).
    /// Each commit measures both sides and moves the split toward equal
    /// finish times; output is identical for every split.
    #[derive(Clone, Copy, Debug)]
    struct Tune {
        first16: usize,
        blocks: usize,
        g1: f64,
        c1: f64,
        g2: f64,
        c2: f64,
    }

    impl Tune {
        fn initial() -> Tune {
            Tune {
                first16: hybrid_gpu_first16().unwrap_or(6),
                blocks: hybrid_gpu_blocks().unwrap_or(7),
                g1: 0.0,
                c1: 0.0,
                g2: 0.0,
                c2: 0.0,
            }
        }

        /// Fold one commit's timings in (exponential average) and re-balance:
        /// `x` units on the GPU finish together with `16 − x` on the CPU at
        /// `x = 16·c / (g + c)`. A pinned (env) split stays put.
        fn update(&mut self, t: &HybridTimes) {
            let ema = |old: f64, new: f64| if old == 0.0 { new } else { 0.5 * (old + new) };
            if self.first16 > 0 && self.first16 < 16 {
                self.g1 = ema(self.g1, t.gpu1 / self.first16 as f64);
                self.c1 = ema(self.c1, t.cpu1 / (16 - self.first16) as f64);
            }
            self.g2 = ema(self.g2, t.gpu2 / self.blocks as f64);
            self.c2 = ema(self.c2, t.cpu2 / (16 - self.blocks) as f64);
            let balance = |g: f64, c: f64, lo: usize| -> usize {
                ((16.0 * c / (g + c)).round() as usize).clamp(lo, 15)
            };
            // At least one sixteenth, so the GPU side stays measured.
            if hybrid_gpu_first16().is_none() && self.g1 > 0.0 {
                self.first16 = balance(self.g1, self.c1, 1);
            }
            if hybrid_gpu_blocks().is_none() {
                self.blocks = balance(self.g2, self.c2, 1);
            }
        }
    }

    /// One hybrid commit's measured sides, ms: GPU execution time of each
    /// phase's command buffer (its own timestamps) and the CPU's wall time
    /// for its share.
    struct HybridTimes {
        gpu1: f64,
        cpu1: f64,
        gpu2: f64,
        cpu2: f64,
    }

    /// Held by the codeword and tree views of a GPU commit; the buffers are
    /// free for the next commit once both are dropped.
    struct Lease;

    impl Drop for Lease {
        fn drop(&mut self) {
            STATE.lock().unwrap_or_else(|e| e.into_inner()).in_use = false;
        }
    }

    /// The codeword of a GPU commit: a read-only view of the persistent
    /// shared codeword buffer (unified memory; CPU reads are ordinary cached
    /// reads).
    pub struct GpuCodeword {
        ptr: *const F128,
        len: usize,
        _lease: Arc<Lease>,
    }

    /// The Merkle tree of a GPU commit, in the persistent shared tree buffer.
    pub struct GpuMerkleTree {
        ptr: *const Hash,
        len: usize,
        _lease: Arc<Lease>,
    }

    // SAFETY: the views point into process-lifetime shared buffers that the
    // GPU writes only while no lease exists.
    unsafe impl Send for GpuCodeword {}
    unsafe impl Sync for GpuCodeword {}
    unsafe impl Send for GpuMerkleTree {}
    unsafe impl Sync for GpuMerkleTree {}

    impl core::ops::Deref for GpuCodeword {
        type Target = [F128];
        fn deref(&self) -> &[F128] {
            // SAFETY: `len` initialized words, unmodified while the lease lives.
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    impl core::ops::Deref for GpuMerkleTree {
        type Target = [Hash];
        fn deref(&self) -> &[Hash] {
            // SAFETY: as for the codeword.
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Run the graph over `bufs`.
    unsafe fn run_graph(gpu: &Gpu, bufs: &Buffers, mixed: bool) -> Result<(), String> {
        unsafe {
            let pool = (gpu.api.pool_push)();
            let result = (|| {
                let cb = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb)?;
                encode_ntt(gpu, enc, bufs, mixed);
                encode_merkle(gpu, enc, bufs);
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb)
            })();
            (gpu.api.pool_pop)(pool);
            result
        }
    }

    /// `FLOCK_METAL_GPU_BLOCKS`: pin the layer-4 blocks (of 16) the GPU
    /// takes in the hybrid commit; 16 runs the whole graph on the GPU. Unset,
    /// the split is tuned per shape ([`Tune`]).
    fn hybrid_gpu_blocks() -> Option<usize> {
        static K: OnceLock<Option<usize>> = OnceLock::new();
        *K.get_or_init(|| {
            std::env::var("FLOCK_METAL_GPU_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|k| k.clamp(1, 16))
        })
    }

    /// `FLOCK_METAL_GPU_FIRST16`: pin the sixteenths of the first pass's
    /// tiles the GPU takes. Unset, tuned per shape ([`Tune`]).
    fn hybrid_gpu_first16() -> Option<usize> {
        static K: OnceLock<Option<usize>> = OnceLock::new();
        *K.get_or_init(|| {
            std::env::var("FLOCK_METAL_GPU_FIRST16")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map(|k| k.clamp(0, 16))
        })
    }

    /// GPU execution time of a completed command buffer, ms.
    unsafe fn gpu_exec_ms(gpu: &Gpu, cb: Id) -> f64 {
        unsafe {
            let t0: f64 = send!(
                gpu.api,
                unsafe extern "C" fn(Id, Sel) -> f64,
                cb,
                c"GPUStartTime"
            );
            let t1: f64 = send!(
                gpu.api,
                unsafe extern "C" fn(Id, Sel) -> f64,
                cb,
                c"GPUEndTime"
            );
            (t1 - t0) * 1e3
        }
    }

    /// The message positions of first-pass tiles `[r0, r1)`: the 8 runs
    /// `(e << s) + [r0, r1)`, `s = log_d − 4`.
    fn tile_positions(
        log_d: usize,
        r0: usize,
        r1: usize,
    ) -> impl Iterator<Item = core::ops::Range<usize>> {
        let s = log_d - 4;
        (0..8usize)
            .map(move |e| (e << s) + r0..(e << s) + r1)
            .filter(|r| !r.is_empty())
    }

    /// Levels below the root where both sides of the hybrid stop hashing:
    /// each layer-4 block then owns 8 nodes of that level, so the CPU/GPU
    /// boundary stays a whole 256-byte span at every level either side
    /// writes. The CPU hashes the last `HYBRID_TOP` levels after the join.
    const HYBRID_TOP: usize = 7;

    /// A raw pointer the CPU suffix's tasks share; each writes a disjoint
    /// range.
    #[derive(Clone, Copy)]
    struct SyncPtr<T>(*mut T);
    // SAFETY: only used for disjoint per-task ranges.
    unsafe impl<T> Send for SyncPtr<T> {}
    unsafe impl<T> Sync for SyncPtr<T> {}

    impl<T> SyncPtr<T> {
        /// The pointer, through a method so closures capture the whole
        /// (Send) wrapper rather than its raw field.
        fn get(self) -> *mut T {
            self.0
        }
    }

    /// The hybrid commit, in two phases over the same shared buffers:
    /// 1. the first pass (layers 0..4), the GPU on tiles `[0, r1)` and the
    ///    CPU on the rest, `r1 = first16/16` of them;
    /// 2. the rest, the GPU on layer-4 blocks `[0, k)` (NTT and their
    ///    subtrees) and the CPU on `[k, 16)`.
    ///
    /// The CPU hashes the top [`HYBRID_TOP`] levels last.
    unsafe fn run_hybrid(
        gpu: &Gpu,
        bufs: &Buffers,
        first16: usize,
        k: usize,
        fill: &dyn Fn(&mut [F128], core::ops::Range<usize>),
        msg: &mut [F128],
    ) -> Result<HybridTimes, String> {
        let (log_d, lanes) = (bufs.log_d, bufs.lanes);
        let n = 1usize << log_d;
        let stop = log_d - HYBRID_TOP;
        let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
        unsafe {
            let t0 = std::time::Instant::now();
            let tiles = 1usize << (log_d - 4);
            let r1 = tiles / 16 * first16;
            let tw = core::slice::from_raw_parts(gpu.contents(bufs.tw_buf).cast::<F128>(), n - 1);
            // The GPU's message rows first, so it can start; the CPU fills its
            // own while the GPU runs.
            for pos in tile_positions(log_d, 0, r1) {
                fill(msg, pos);
            }
            let t_fill_gpu = t0.elapsed();
            let pool = (gpu.api.pool_push)();
            let first = (|| {
                let cb = gpu.command_buffer()?;
                if r1 > 0 {
                    let enc = gpu.compute_encoder(cb)?;
                    encode_first_pass(gpu, enc, bufs, r1);
                    gpu.end_encoding(enc);
                }
                gpu.commit_async(cb);
                Ok::<Id, String>(cb)
            })();
            let mut cpu1 = 0.0;
            let mut gpu1 = 0.0;
            let first = match first {
                Ok(cb) => {
                    let t_cpu1 = std::time::Instant::now();
                    for pos in tile_positions(log_d, r1, tiles) {
                        fill(msg, pos);
                    }
                    crate::ntt::AdditiveNttF128::first_pass_from_message_tiles(
                        gpu.contents(bufs.code_buf).cast(),
                        msg,
                        lanes,
                        log_d,
                        r1,
                        tiles,
                        tw,
                    );
                    cpu1 = t_cpu1.elapsed().as_secs_f64() * 1e3;
                    let waited = gpu.wait(cb);
                    if waited.is_ok() && r1 > 0 {
                        gpu1 = gpu_exec_ms(gpu, cb);
                    }
                    waited
                }
                Err(e) => Err(e),
            };
            (gpu.api.pool_pop)(pool);
            first?;
            let t_first = t0.elapsed();

            let pool = (gpu.api.pool_push)();
            let cb2 = (|| {
                let cb = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb)?;
                for (l, f) in plan_passes(log_d, 4) {
                    encode_pass(gpu, enc, bufs, l, f, true, k as u64);
                }
                encode_merkle_range(gpu, enc, bufs, k << (log_d - 4), stop);
                gpu.end_encoding(enc);
                gpu.commit_async(cb);
                Ok::<Id, String>(cb)
            })();
            let cb2 = match cb2 {
                Ok(cb) => cb,
                Err(e) => {
                    (gpu.api.pool_pop)(pool);
                    return Err(e);
                }
            };

            // The CPU suffix, in the same buffers.
            let t_cpu = std::time::Instant::now();
            let block_elems = lanes << (log_d - 4);
            let code = core::slice::from_raw_parts_mut(
                gpu.contents(bufs.code_buf)
                    .cast::<F128>()
                    .add(k * block_elems),
                (16 - k) * block_elems,
            );
            let tree = SyncPtr(gpu.contents(bufs.tree_buf).cast::<Hash>());
            let level_off = |j: usize| (0..j).map(|i| n >> i).sum::<usize>();
            let base_pos = k << (log_d - 4);
            let leaf_bytes = lanes * 16;
            // The lowest sub-group root level any hook reached: levels above
            // it are hashed after the transforms.
            let sub_top = std::sync::atomic::AtomicUsize::new(stop);
            let on_sub = |block_pos: usize, elem_off: usize, sub: &[F128]| {
                let p0 = block_pos + elem_off / lanes;
                let np = sub.len() / lanes;
                debug_assert!(np.is_power_of_two() && p0.is_multiple_of(np));
                let bytes = core::slice::from_raw_parts(sub.as_ptr().cast::<u8>(), sub.len() * 16);
                let leaves = core::slice::from_raw_parts_mut(tree.get().add(p0), np);
                crate::merkle::hash_leaves_serial(
                    bytes,
                    leaf_bytes,
                    leaves,
                    crate::merkle::HashKind::Blake3,
                );
                let top = (np.trailing_zeros() as usize).min(stop);
                for j in 1..=top {
                    let read = core::slice::from_raw_parts(
                        tree.get().add(level_off(j - 1) + (p0 >> (j - 1))),
                        np >> (j - 1),
                    );
                    let out = core::slice::from_raw_parts_mut(
                        tree.get().add(level_off(j) + (p0 >> j)),
                        np >> j,
                    );
                    crate::merkle::hash_parents_serial(read, out);
                }
                sub_top.fetch_min(top, std::sync::atomic::Ordering::Relaxed);
            };
            // Each CPU block is a transform of its own (`sub_block`), run on
            // the tuned path; its deep pass hands finished sub-groups to
            // `on_sub` while they are cache-hot.
            let ntt = crate::ntt::AdditiveNttF128::standard(log_d);
            for (i, block) in code.chunks_exact_mut(block_elems).enumerate() {
                let b = k + i;
                let block_pos = b << (log_d - 4);
                let hashed = |elem_off: usize, sub: &[F128]| on_sub(block_pos, elem_off, sub);
                ntt.sub_block(4, b)
                    .forward_transform_interleaved_live_from_layer_with(
                        block,
                        lanes,
                        lanes,
                        0,
                        Some(crate::ntt::DeepPassHook {
                            on_sub: &hashed,
                            caller_pool_only: false,
                        }),
                    );
            }
            // Levels from the sub-groups' roots up to `stop`, over the CPU's
            // blocks (none when the sub-groups reach `stop` themselves).
            let done = sub_top.load(std::sync::atomic::Ordering::Relaxed);
            for j in done + 1..=stop {
                let lo = base_pos >> j;
                let count = (n - base_pos) >> j;
                let read = core::slice::from_raw_parts(
                    tree.get().add(level_off(j - 1) + 2 * lo),
                    2 * count,
                );
                let out = core::slice::from_raw_parts_mut(tree.get().add(level_off(j) + lo), count);
                use rayon::prelude::{
                    IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut,
                };
                out.par_chunks_mut(1024)
                    .zip(read.par_chunks(2048))
                    .for_each(|(o, r)| {
                        crate::merkle::hash_parents_serial(r, o);
                    });
            }
            let t_cpu = t_cpu.elapsed();
            let t_wait = std::time::Instant::now();
            let waited = gpu.wait(cb2);
            let gpu2 = if waited.is_ok() {
                gpu_exec_ms(gpu, cb2)
            } else {
                0.0
            };
            (gpu.api.pool_pop)(pool);
            waited?;
            let t_wait = t_wait.elapsed();

            // The top levels, over both halves.
            for j in stop + 1..=log_d {
                let read =
                    core::slice::from_raw_parts(tree.get().add(level_off(j - 1)), n >> (j - 1));
                let out = core::slice::from_raw_parts_mut(tree.get().add(level_off(j)), n >> j);
                crate::merkle::hash_parents_serial(read, out);
            }
            if timing {
                eprintln!(
                    "[commit-timing] metal hybrid (gpu first {first16}/16, blocks {k}/16): gpu fill {:.2} | first pass {:.2} (cpu {cpu1:.2}, gpu {gpu1:.2}) | cpu suffix {:.2} (gpu {gpu2:.2}) | gpu wait after {:.2} ms",
                    t_fill_gpu.as_secs_f64() * 1e3,
                    t_first.as_secs_f64() * 1e3,
                    t_cpu.as_secs_f64() * 1e3,
                    t_wait.as_secs_f64() * 1e3,
                );
            }
            Ok(HybridTimes {
                gpu1,
                cpu1,
                gpu2,
                cpu2: t_cpu.as_secs_f64() * 1e3,
            })
        }
    }

    pub(super) fn try_commit(
        log_d: usize,
        lanes: usize,
        fill: impl Fn(&mut [F128], core::ops::Range<usize>),
        cpu: impl FnOnce(&[F128]) -> (Vec<F128>, Vec<Hash>),
    ) -> Option<GpuCommit> {
        let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let gpu = match gpu() {
            Ok(g) => g,
            Err(e) => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| eprintln!("[metal] GPU commit unavailable: {e}"));
                return None;
            }
        };
        // Take the buffers for this shape, or leave the commit to the CPU.
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if state.disabled || state.in_use {
            return None;
        }
        if state
            .bufs
            .as_ref()
            .is_none_or(|b| (b.log_d, b.lanes) != (log_d, lanes))
        {
            if let Some(mut old) = state.bufs.take() {
                unsafe { old.release(gpu) };
            }
            match unsafe { Buffers::new(gpu, log_d, lanes) } {
                Ok(b) => state.bufs = Some(b),
                Err(e) => {
                    eprintln!("[metal] buffer allocation failed ({e}); CPU commit");
                    return None;
                }
            }
        }
        state.in_use = true;
        let verify = !state.verified.contains(&(log_d, lanes));
        let tune = match state
            .tunes
            .iter()
            .find(|(shape, _)| *shape == (log_d, lanes))
        {
            Some((_, t)) => *t,
            None => Tune::initial(),
        };
        // The graph runs with the lock dropped: `in_use` keeps other commits
        // off the buffers, so they are neither reallocated nor written.
        let bufs = state.bufs.expect("buffers just ensured");
        drop(state);
        let lease = Arc::new(Lease);

        // SAFETY: we hold the buffers exclusively (`in_use`) until the lease
        // drops; the GPU never writes the message buffer.
        let msg = unsafe {
            core::slice::from_raw_parts_mut(
                gpu.contents(bufs.msg_buf).cast::<F128>(),
                bufs.msg_words(),
            )
        };
        let half = 1usize << (log_d - 1);
        let ran = if tune.blocks < 16 && log_d >= 12 {
            let ran = unsafe { run_hybrid(gpu, &bufs, tune.first16, tune.blocks, &fill, msg) };
            // The first commit of a shape runs cold (fresh buffers, pipeline
            // warm-up) and is not a fair sample.
            if let (Ok(times), false) = (&ran, verify) {
                let mut tune = tune;
                tune.update(times);
                let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
                match state
                    .tunes
                    .iter_mut()
                    .find(|(shape, _)| *shape == (log_d, lanes))
                {
                    Some((_, t)) => *t = tune,
                    None => state.tunes.push(((log_d, lanes), tune)),
                }
            }
            ran.map(|_| ())
        } else {
            fill(msg, 0..half);
            unsafe { run_graph(gpu, &bufs, true) }
        };
        if timing {
            eprintln!(
                "[commit-timing] metal: total {:.2} ms (2^{log_d} x {lanes})",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        let codeword = GpuCodeword {
            ptr: unsafe { gpu.contents(bufs.code_buf).cast() },
            len: bufs.code_words(),
            _lease: lease.clone(),
        };
        let tree = GpuMerkleTree {
            ptr: unsafe { gpu.contents(bufs.tree_buf).cast() },
            len: bufs.tree_nodes(),
            _lease: lease,
        };
        if let Err(e) = ran {
            eprintln!("[metal] GPU commit failed ({e}); CPU commit");
            // The hybrid may have stopped before the CPU's rows were filled.
            fill(msg, 0..half);
            let (cw, tr) = cpu(msg);
            return Some(GpuCommit::Cpu(cw, tr));
        }
        if verify {
            let (cw, tr) = cpu(msg);
            let same = *cw == *codeword && *tr == *tree;
            let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
            if !same {
                state.disabled = true;
                drop(state);
                eprintln!(
                    "[metal] GPU commit disagrees with the CPU at 2^{log_d} x {lanes}; Metal off for this process"
                );
                return Some(GpuCommit::Cpu(cw, tr));
            }
            state.verified.push((log_d, lanes));
            drop(state);
            crate::scratch::give_f128(cw);
        }
        Some(GpuCommit::Gpu(codeword, tree))
    }

    // -----------------------------------------------------------------------
    // AG zerocheck round 1 on the GPU: a prototype, test-only until it is
    // wired into the prover (see `ag_round1_timing`).
    // -----------------------------------------------------------------------

    /// The prototype's library source: the commit kernels' field helpers,
    /// the generated encode, and the kernel.
    #[cfg(test)]
    fn ag_round1_source() -> String {
        format!(
            "{MSL_SOURCE}\n{}\n{}",
            include_str!("gpu_ag_round1_gen.msl"),
            include_str!("gpu_ag_round1.msl")
        )
    }

    /// The `ag_round1` pipeline, compiled on first use (kept as an address:
    /// the pipeline object is thread-safe).
    #[cfg(test)]
    fn ag_round1_pso(gpu: &Gpu) -> Result<Id, String> {
        static PSO: OnceLock<Result<usize, String>> = OnceLock::new();
        PSO.get_or_init(|| unsafe {
            let api = &gpu.api;
            let pool = (api.pool_push)();
            let made = (|| {
                let src = api.nsstring(&ag_round1_source())?;
                let mut err: Id = NIL;
                let library: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id,
                    gpu.device,
                    c"newLibraryWithSource:options:error:",
                    src,
                    NIL,
                    &mut err
                );
                if library.is_null() {
                    return Err(format!(
                        "ag_round1 compile failed: {}",
                        api.error_string(err)
                    ));
                }
                let name = api.nsstring("ag_round1")?;
                let f: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel, Id) -> Id,
                    library,
                    c"newFunctionWithName:",
                    name
                );
                if f.is_null() {
                    return Err("ag_round1 not found".into());
                }
                let mut err: Id = NIL;
                let pso: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id,
                    gpu.device,
                    c"newComputePipelineStateWithFunction:error:",
                    f,
                    &mut err
                );
                if pso.is_null() {
                    return Err(format!("ag_round1 pipeline: {}", api.error_string(err)));
                }
                Ok(pso as usize)
            })();
            (api.pool_pop)(pool);
            made
        })
        .clone()
        .map(|p| p as Id)
    }

    /// The prototype's sums over `eq.len()` Full blocks (a multiple of 8):
    /// `(res, bank0, bank1)` as the CPU kernel returns them, plus the GPU
    /// execution time in ms. Copies its inputs into fresh shared buffers.
    #[cfg(test)]
    pub(crate) fn ag_round1_on_gpu(
        a: &[u8],
        b: &[u8],
        c: &[u8],
        eq: &[F128],
        groups: usize,
    ) -> Result<([F128; 160], [F128; 64], [F128; 64], f64), String> {
        let n = eq.len();
        assert!(
            n.is_multiple_of(8) && a.len() == n * 1024 && b.len() == a.len() && c.len() == a.len()
        );
        let gpu = gpu()?;
        let pso = ag_round1_pso(gpu)?;
        let groups = groups.clamp(1, n / 8);
        unsafe {
            let upload = |data: &[u8]| -> Result<Id, String> {
                let buf = gpu.new_buffer(data.len().max(16))?;
                core::ptr::copy_nonoverlapping(data.as_ptr(), gpu.contents(buf), data.len());
                Ok(buf)
            };
            let eq_bytes = core::slice::from_raw_parts(eq.as_ptr().cast::<u8>(), n * 16);
            let bufs = [
                upload(a)?,
                upload(b)?,
                upload(c)?,
                upload(eq_bytes)?,
                gpu.new_buffer(groups * 288 * 16)?,
            ];
            let pool = (gpu.api.pool_push)();
            let ran = (|| {
                let cb = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb)?;
                gpu.set_pipeline(enc, pso);
                for (i, &buf) in bufs.iter().enumerate() {
                    gpu.set_buffer(enc, buf, 0, i);
                }
                gpu.set_bytes(enc, &(n as u32), 5);
                gpu.dispatch(enc, groups as u64, 32);
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb)?;
                Ok::<f64, String>(gpu_exec_ms(gpu, cb))
            })();
            (gpu.api.pool_pop)(pool);
            let ms = ran;
            let partial =
                core::slice::from_raw_parts(gpu.contents(bufs[4]).cast::<F128>(), groups * 288);
            let mut sums = [F128::ZERO; 288];
            for g in partial.as_chunks::<288>().0 {
                for (s, &v) in sums.iter_mut().zip(g) {
                    *s += v;
                }
            }
            for buf in bufs {
                gpu.release(buf);
            }
            let ms = ms?;
            let mut res = [F128::ZERO; 160];
            let mut b0 = [F128::ZERO; 64];
            let mut b1 = [F128::ZERO; 64];
            res.copy_from_slice(&sums[..160]);
            b0.copy_from_slice(&sums[160..224]);
            b1.copy_from_slice(&sums[224..]);
            Ok((res, b0, b1, ms))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::test_rng::Rng;

        /// The GPU AG round-1 prototype equals the CPU kernel on random Full
        /// blocks: the 160 product sums and both c banks.
        #[test]
        fn ag_round1_matches_cpu() {
            if let Err(e) = gpu() {
                eprintln!("no Metal device ({e}); skipping");
                return;
            }
            let mut rng = Rng::new(53);
            let n = 64usize;
            let bytes =
                |rng: &mut Rng| -> Vec<u8> { (0..n * 1024).map(|_| rng.f128().lo as u8).collect() };
            let (a, b, c) = (bytes(&mut rng), bytes(&mut rng), bytes(&mut rng));
            let eq: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let coverage = vec![crate::zerocheck::BlockCoverage::Full; n];
            let (res, b0, b1) = crate::genus95_curve_code::round1::round1_straight_line_program_packed_banks_fused_padded(
                &a, &b, &c, &eq, &coverage,
            );
            for groups in [1usize, 3, 8] {
                let (gres, gb0, gb1, _) = ag_round1_on_gpu(&a, &b, &c, &eq, groups).unwrap();
                assert!(gb0 == b0, "bank0, {groups} groups");
                assert!(gb1 == b1, "bank1, {groups} groups");
                assert!(gres == res, "res, {groups} groups");
            }
        }

        /// GPU vs CPU time of round 1 at production-like size: run with
        /// `--ignored --nocapture` (`AG_BLOCKS` = log2 blocks, default 17).
        #[test]
        #[ignore]
        fn ag_round1_timing() {
            let lg: usize = std::env::var("AG_BLOCKS").map_or(17, |v| v.parse().unwrap());
            let n = 1usize << lg;
            let mut rng = Rng::new(7);
            let bytes = |rng: &mut Rng| -> Vec<u8> {
                let words: Vec<F128> = (0..n * 64).map(|_| rng.f128()).collect();
                unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), n * 1024) }
                    .to_vec()
            };
            let (a, b, c) = (bytes(&mut rng), bytes(&mut rng), bytes(&mut rng));
            let eq: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let coverage = vec![crate::zerocheck::BlockCoverage::Full; n];
            for rep in 0..3 {
                let t = std::time::Instant::now();
                let cpu = crate::genus95_curve_code::round1::round1_straight_line_program_packed_banks_fused_padded(
                    &a, &b, &c, &eq, &coverage,
                );
                let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
                for groups in [256usize, 1024, 4096] {
                    let t = std::time::Instant::now();
                    let (gres, ..) = ag_round1_on_gpu(&a, &b, &c, &eq, groups).unwrap();
                    let wall = t.elapsed().as_secs_f64() * 1e3;
                    let (_, _, _, gpu_ms) = ag_round1_on_gpu(&a, &b, &c, &eq, groups).unwrap();
                    assert!(gres == cpu.0);
                    eprintln!(
                        "2^{lg} blocks rep {rep}: cpu {cpu_ms:.1} ms | gpu {groups} groups: {gpu_ms:.1} ms exec ({wall:.1} wall incl. copies)"
                    );
                }
            }
        }

        /// GPU execution time of one command buffer (`GPUEndTime −
        /// GPUStartTime`), in ms.
        unsafe fn gpu_ms(gpu: &Gpu, cb: Id) -> f64 {
            unsafe {
                let t0: f64 = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> f64,
                    cb,
                    c"GPUStartTime"
                );
                let t1: f64 = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, Sel) -> f64,
                    cb,
                    c"GPUEndTime"
                );
                (t1 - t0) * 1e3
            }
        }

        /// The hybrid commit (GPU prefix blocks, CPU suffix blocks, CPU top
        /// levels) equals the CPU commit for every split.
        #[test]
        fn hybrid_commit_matches_cpu() {
            let gpu = match gpu() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("no Metal device ({e}); skipping");
                    return;
                }
            };
            let mut rng = Rng::new(29);
            for log_d in [12usize, 13] {
                for lanes in [64usize, 61, 6] {
                    let bufs = unsafe { Buffers::new(gpu, log_d, lanes) }.unwrap();
                    let msg: Vec<F128> = (0..bufs.msg_words()).map(|_| rng.f128()).collect();
                    let (cw_ref, tree_ref) = cpu_commit(&msg, log_d, lanes);
                    // The hybrid fills the message by ranges; every position
                    // must be covered exactly once across the two phases.
                    let fill = |m: &mut [F128], pos: core::ops::Range<usize>| {
                        let r = pos.start * lanes..pos.end * lanes;
                        m[r.clone()].copy_from_slice(&msg[r]);
                    };
                    for (first16, k) in [(0usize, 1usize), (6, 5), (16, 8), (3, 15)] {
                        unsafe {
                            let msg_buf = core::slice::from_raw_parts_mut(
                                gpu.contents(bufs.msg_buf).cast::<F128>(),
                                bufs.msg_words(),
                            );
                            msg_buf.fill(F128::new(0xBAD, 0xBAD));
                            // Stale bytes must not survive.
                            core::ptr::write_bytes(
                                gpu.contents(bufs.code_buf),
                                0xA5,
                                bufs.code_words() * 16,
                            );
                            core::ptr::write_bytes(
                                gpu.contents(bufs.tree_buf),
                                0x5A,
                                bufs.tree_nodes() * 32,
                            );
                            run_hybrid(gpu, &bufs, first16, k, &fill, msg_buf).unwrap();
                            let cw = core::slice::from_raw_parts(
                                gpu.contents(bufs.code_buf).cast::<F128>(),
                                bufs.code_words(),
                            );
                            let tree = core::slice::from_raw_parts(
                                gpu.contents(bufs.tree_buf).cast::<Hash>(),
                                bufs.tree_nodes(),
                            );
                            assert!(
                                cw == &cw_ref[..],
                                "codeword, 2^{log_d} x {lanes}, {first16}/16, {k}/16"
                            );
                            assert!(
                                tree == &tree_ref[..],
                                "tree, 2^{log_d} x {lanes}, {first16}/16, {k}/16"
                            );
                        }
                    }
                    let mut bufs = bufs;
                    unsafe { bufs.release(gpu) };
                }
            }
        }

        /// Per-stage GPU time of the commit graph at production size: run
        /// with `--ignored --nocapture`. Each stage gets its own command
        /// buffer, timed by the GPU's own timestamps, next to the wall clock
        /// of the whole graph in one buffer.
        #[test]
        #[ignore]
        fn gpu_graph_timing() {
            let gpu = gpu().expect("Metal device");
            let log_d: usize = std::env::var("GPU_LOG_D").map_or(20, |v| v.parse().unwrap());
            for lanes in [64usize, 46] {
                let bufs = unsafe { Buffers::new(gpu, log_d, lanes) }.unwrap();
                unsafe {
                    core::ptr::write_bytes(gpu.contents(bufs.msg_buf), 0x5a, bufs.msg_words() * 16);
                    for rep in 0..4 {
                        let t = std::time::Instant::now();
                        run_graph(gpu, &bufs, true).unwrap();
                        let wall = t.elapsed().as_secs_f64() * 1e3;
                        // Stage by stage.
                        let stage = |name: &str, f: &dyn Fn(Id)| {
                            let pool = (gpu.api.pool_push)();
                            let cb = gpu.command_buffer().unwrap();
                            let enc = gpu.compute_encoder(cb).unwrap();
                            f(enc);
                            gpu.end_encoding(enc);
                            gpu.commit_and_wait(cb).unwrap();
                            let ms = gpu_ms(gpu, cb);
                            (gpu.api.pool_pop)(pool);
                            (name.to_string(), ms)
                        };
                        let mut stages = vec![stage("ntt (all passes)", &|enc| {
                            encode_ntt(gpu, enc, &bufs, true)
                        })];
                        stages.push(stage("merkle", &|enc| encode_merkle(gpu, enc, &bufs)));
                        let total: f64 = stages.iter().map(|s| s.1).sum();
                        eprintln!(
                            "2^{log_d} x {lanes} rep {rep}: one buffer {wall:.1} ms wall | {} | sum {total:.1}",
                            stages
                                .iter()
                                .map(|(n, ms)| format!("{n} {ms:.1}"))
                                .collect::<Vec<_>>()
                                .join(" | ")
                        );
                    }
                    // Each NTT pass alone.
                    let mut line = String::new();
                    for (l, f) in std::iter::once((0usize, 4usize)).chain(plan_passes(log_d, 4)) {
                        let pool = (gpu.api.pool_push)();
                        let cb = gpu.command_buffer().unwrap();
                        let enc = gpu.compute_encoder(cb).unwrap();
                        encode_pass(gpu, enc, &bufs, l, f, true, 16);
                        gpu.end_encoding(enc);
                        gpu.commit_and_wait(cb).unwrap();
                        line += &format!(" ({l},{f}) {:.1}", gpu_ms(gpu, cb));
                        (gpu.api.pool_pop)(pool);
                    }
                    eprintln!("2^{log_d} x {lanes} passes:{line}");
                    let mut b = bufs;
                    b.release(gpu);
                }
            }
        }

        /// The CPU reference: replicate the message, NTT from layer 1, hash.
        fn cpu_commit(msg: &[F128], log_d: usize, lanes: usize) -> (Vec<F128>, Vec<Hash>) {
            let mut cw = vec![F128::ZERO; 2 * msg.len()];
            cw[..msg.len()].copy_from_slice(msg);
            cw[msg.len()..].copy_from_slice(msg);
            crate::ntt::AdditiveNttF128::standard(log_d)
                .forward_transform_interleaved_from_layer(&mut cw, lanes, 1);
            let bytes =
                unsafe { core::slice::from_raw_parts(cw.as_ptr().cast::<u8>(), cw.len() * 16) };
            let tree =
                crate::merkle::merkle_tree(bytes, 1 << log_d, crate::merkle::HashKind::Blake3);
            (cw, tree)
        }

        /// The GPU graph equals the CPU commit, codeword and tree, for full
        /// and trimmed lane counts (short final BLAKE3 blocks) and every
        /// pass kernel: log_d 8 and 12 end in the mixed pass, or in the
        /// half-footprint one with `mixed` off; 9 runs the plain f = 4 pass
        /// (s = 1) and the generic f = 1; 10 the generic f = 2; 11 the f = 3
        /// pass. From log_d 10 the f = 4 passes share tables (s >= 2).
        #[test]
        fn gpu_commit_matches_cpu() {
            let gpu = match gpu() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("no Metal device ({e}); skipping");
                    return;
                }
            };
            let mut rng = Rng::new(17);
            for (log_d, mixed) in [
                (8usize, true),
                (8, false),
                (9, true),
                (10, true),
                (11, true),
                (12, true),
                (12, false),
            ] {
                for lanes in [64usize, 61, 6, 1] {
                    let bufs = unsafe { Buffers::new(gpu, log_d, lanes) }.unwrap();
                    let msg: Vec<F128> = (0..bufs.msg_words()).map(|_| rng.f128()).collect();
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            msg.as_ptr(),
                            gpu.contents(bufs.msg_buf).cast(),
                            msg.len(),
                        );
                        run_graph(gpu, &bufs, mixed).unwrap();
                        let cw = core::slice::from_raw_parts(
                            gpu.contents(bufs.code_buf).cast::<F128>(),
                            bufs.code_words(),
                        );
                        let tree = core::slice::from_raw_parts(
                            gpu.contents(bufs.tree_buf).cast::<Hash>(),
                            bufs.tree_nodes(),
                        );
                        let (cw_ref, tree_ref) = cpu_commit(&msg, log_d, lanes);
                        assert!(cw == &cw_ref[..], "codeword, 2^{log_d} x {lanes}");
                        assert!(tree == &tree_ref[..], "tree, 2^{log_d} x {lanes}");
                    }
                    let mut bufs = bufs;
                    unsafe { bufs.release(gpu) };
                }
            }
        }
    }
}
