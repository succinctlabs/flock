//! Sixteen-message BLAKE3 compression on Apple SME2.
//!
//! Streaming SVE at a 512-bit vector length gives sixteen 32-bit lanes, so one
//! group hashes sixteen messages in lockstep, against eight for
//! `blake3_neon::hash8`. The kernel (`blake3_sme2.S`) keeps the state and the
//! message in the 32 Z registers, does every xor-rotate with one SVE2 `XAR`, and
//! transposes the blocks in and the chaining values out through the ZA tile; its
//! header has the details.
//!
//! Provenance: written for Flock from the BLAKE3 specification and the Arm
//! A-profile ISA. It is not derived from any other SME or SVE BLAKE3
//! implementation.
//!
//! Semantics match [`blake3_neon::hash8`](super::blake3_neon::hash8): the IV
//! key, counter 0, block 0 carrying `flags | flags_start`, the last block
//! `flags | flags_end` and its true length, zero-padded. Byte-identical by test
//! (`blake3_sme2_matches_crate`).
//!
//! Apple M4 has one SME unit per CPU cluster, shared by the cluster's cores, so
//! the callers engage this kernel only from a single-threaded pool.

use std::{env::var_os, sync::OnceLock};

core::arch::global_asm!(include_str!("blake3_sme2.S"), options(raw));

unsafe extern "C" {
    fn flock_blake3_sme2_hash16(
        data: *const u8,
        stride: usize,
        n_blocks: usize,
        last_len: usize,
        flags: u32,
        out: *mut u8,
        groups: usize,
    ) -> u64;
    fn flock_blake3_sme2_svl_bytes() -> u64;
}

/// Messages per kernel group.
pub(super) const LANES: usize = 16;

/// Whether the kernel can run here: FEAT_SME2 present, a 512-bit streaming
/// vector length (the kernel's lane count), and no `FLOCK_NO_SME2=1`.
pub(super) fn available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        if var_os("FLOCK_NO_SME2").is_some_and(|v| v == *"1") {
            return false;
        }
        // SAFETY: RDSVL is only reached once FEAT_SME2 (which implies
        // FEAT_SME) is reported; it executes outside streaming mode.
        apple_feature(c"hw.optional.arm.FEAT_SME2")
            && unsafe { flock_blake3_sme2_svl_bytes() } == 64
    })
}

/// Hash `groups * 16` messages, message `i` at `data + i * stride`
/// (`n_blocks` 64-byte blocks, the last `last_len` bytes long), into 32-byte
/// chaining values at `out + i * 32`. Returns `false`, having written
/// nothing, if the kernel refuses because the caller's ZA state is dormant.
///
/// # Safety
/// [`available`] must have returned `true`. `data` must hold
/// `(groups * 16 - 1) * stride + (n_blocks - 1) * 64 + last_len` readable
/// bytes and `out` `groups * 16 * 32` writable bytes.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn hash16(
    data: *const u8,
    stride: usize,
    n_blocks: usize,
    last_len: usize,
    flags: u8,
    flags_start: u8,
    flags_end: u8,
    out: *mut u8,
    groups: usize,
) -> bool {
    debug_assert!((1..=16).contains(&n_blocks));
    debug_assert!((4..=64).contains(&last_len) && last_len.is_multiple_of(4));
    let packed = flags as u32 | (flags_start as u32) << 8 | (flags_end as u32) << 16;
    // SAFETY: forwarded from the caller.
    unsafe { flock_blake3_sme2_hash16(data, stride, n_blocks, last_len, packed, out, groups) == 1 }
}

/// Whether the kernel takes messages of this shape: whole 32-bit words in a
/// single chunk (the kernel loads the last block word by word).
#[inline]
pub(super) fn takes(n_blocks: usize, last_len: usize) -> bool {
    (1..=16).contains(&n_blocks) && last_len.is_multiple_of(4) && last_len >= 4
}

/// One integer `sysctl` flag, read in-process; absent or unreadable is false.
fn apple_feature(key: &std::ffi::CStr) -> bool {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    let mut value: u32 = 0;
    let mut len = size_of::<u32>();
    // SAFETY: `value` is a writable 4-byte buffer and `len` says so.
    let rc = unsafe {
        sysctlbyname(
            key.as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    rc == 0 && len == size_of::<u32>() && value == 1
}
