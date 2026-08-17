//! Runtime sentinel for known-suboptimal code paths.
//!
//! The crate keeps superseded kernels, A/B baselines, and probe-record
//! variants in-tree (they anchor bit-identity tests and future re-measures),
//! which makes it easy to land on one by accident. Instrumented paths call
//! [`suboptimal_path!`]; with the flag set, each such call SITE logs one
//! warning to stderr the first time it executes:
//!
//! ```text
//! FLOCK_WARN_SUBOPTIMAL=1 cargo run/bench/test ...
//! ```
//!
//! Off by default (a single cached-bool check per instrumented call), so A/B
//! benches that exercise the old paths on purpose stay silent unless asked.

use std::sync::OnceLock;

static ENABLED: OnceLock<bool> = OnceLock::new();

/// True when `FLOCK_WARN_SUBOPTIMAL` is set to anything but `0`.
#[inline]
pub fn warnings_enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("FLOCK_WARN_SUBOPTIMAL").is_some_and(|v| v != *"0"))
}

/// Emit the warning line. Not called directly — use [`suboptimal_path!`],
/// which adds the once-per-site dedup and the source location.
#[doc(hidden)]
pub fn warn(site: &str, what: &str, instead: &str) {
    eprintln!("[flock] suboptimal path at {site}: {what} — production path: {instead}");
}

/// Mark the enclosing code path as known-suboptimal. `$what` names the path
/// being executed, `$instead` the production alternative. Warns at most once
/// per call site, and only when `FLOCK_WARN_SUBOPTIMAL` is set.
#[macro_export]
macro_rules! suboptimal_path {
    ($what:expr, $instead:expr) => {{
        if $crate::suboptimal::warnings_enabled() {
            static ONCE: ::std::sync::Once = ::std::sync::Once::new();
            ONCE.call_once(|| {
                $crate::suboptimal::warn(concat!(file!(), ":", line!()), $what, $instead);
            });
        }
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn macro_compiles_and_dedups() {
        // Flag-off smoke: repeated invocation must be silent and cheap.
        for _ in 0..3 {
            crate::suboptimal_path!("test path", "the good path");
        }
    }
}
