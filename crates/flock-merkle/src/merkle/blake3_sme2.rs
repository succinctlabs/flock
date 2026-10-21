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
//! Apple M4 has one SME unit per CPU cluster, shared by the cluster's cores:
//! two threads streaming on one unit split its throughput while each still
//! occupies a core. [`try_claim`] hands out at most one claim per unit, keyed
//! by the cluster the caller is running on, so any number of hashing threads
//! put one SME2 stream on each cluster and the rest on their own NEON pipes.

use std::{
    env::var_os,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

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

/// Most SME units a claim can name; CPUs beyond the last share its claim,
/// which only costs concurrency.
const MAX_UNITS: usize = 8;

static UNIT_CLAIMED: [AtomicBool; MAX_UNITS] = [const { AtomicBool::new(false) }; MAX_UNITS];

/// Exclusive use of one cluster's SME unit, released on drop.
pub(super) struct Claim(usize);

impl Drop for Claim {
    fn drop(&mut self) {
        UNIT_CLAIMED[self.0].store(false, Ordering::Release);
    }
}

/// Claim the SME unit of the cluster the calling thread is running on, or
/// `None` if another thread holds it. A thread that migrates mid-call keeps a
/// claim on the cluster it left, which is harmless: at worst two streams share
/// a unit for the rest of that call.
pub(super) fn try_claim() -> Option<Claim> {
    let unit = current_unit();
    UNIT_CLAIMED[unit]
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .ok()
        .map(|_| Claim(unit))
}

/// The SME unit (CPU cluster) of the CPU this thread is running on.
///
/// Apple Silicon numbers the efficiency cores first, then the performance
/// cores (M4: CPUs 0-5 E, 6-9 P), and every `cpusperl2` consecutive CPUs of a
/// perf level share an L2 and an SME unit.
fn current_unit() -> usize {
    // (efficiency CPUs, efficiency per cluster, performance per cluster).
    static GEOMETRY: OnceLock<(usize, usize, usize)> = OnceLock::new();
    let &(e_cpus, e_per, p_per) = GEOMETRY.get_or_init(|| {
        let get = |key: &std::ffi::CStr| sysctl_u32(key).map_or(0, |v| v as usize);
        match get(c"hw.nperflevels") {
            2 => (
                get(c"hw.perflevel1.logicalcpu"),
                get(c"hw.perflevel1.cpusperl2").max(1),
                get(c"hw.perflevel0.cpusperl2").max(1),
            ),
            _ => (0, 1, get(c"hw.perflevel0.cpusperl2").max(1)),
        }
    });
    unsafe extern "C" {
        fn pthread_cpu_number_np(cpu_number_out: *mut usize) -> i32;
    }
    let mut cpu = 0usize;
    // SAFETY: `cpu` is a writable `usize`.
    if unsafe { pthread_cpu_number_np(&raw mut cpu) } != 0 {
        return 0;
    }
    let unit = if cpu < e_cpus {
        cpu / e_per
    } else {
        e_cpus.div_ceil(e_per) + (cpu - e_cpus) / p_per
    };
    unit.min(MAX_UNITS - 1)
}

/// One integer `sysctl` flag, read in-process; absent or unreadable is false.
fn apple_feature(key: &std::ffi::CStr) -> bool {
    sysctl_u32(key) == Some(1)
}

/// One 32-bit integer `sysctl`, read in-process.
fn sysctl_u32(key: &std::ffi::CStr) -> Option<u32> {
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
    (rc == 0 && len == size_of::<u32>()).then_some(value)
}
