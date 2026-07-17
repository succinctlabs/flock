#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(super) mod x86_64;
