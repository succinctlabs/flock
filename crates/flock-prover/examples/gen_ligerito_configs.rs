//! Regenerator for the Ligerito security configs.
//!
//! For each `m` in 22..=35 it mechanically derives the security config of
//! all three named profiles via `LigeritoSecurityConfig::derive_profile`:
//!
//! - `fast`:   JohnsonOod, rate 1/2, η = 0.02, 100-bit overall soundness.
//! - `slim`:   JohnsonOod, rate 1/4, η = 0.02, 16-bit query grinding,
//!             100-bit overall.
//! - `secure`: Udr, rate 1/2, ε* = 1e-3, 120-bit overall.
//!
//! Each derived config is validated (including the whole-protocol union
//! bound), serialized, round-trip checked, and written to
//! `crates/flock-core/configs/ligerito/m<m>_<profile>.toml` (these are the
//! configs `flock-core` embeds via `include_str!`).
//!
//! Run: `cargo run --release --example gen_ligerito_configs`

use std::path::Path;

use flock_prover::pcs::ligerito::{LigeritoProfile, LigeritoSecurityConfig};

fn main() {
    let profiles = [
        LigeritoProfile::Fast,
        LigeritoProfile::Slim,
        LigeritoProfile::Secure,
    ];
    // Configs live in the flock-core crate (which embeds them via include_str!).
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../flock-core/configs/ligerito");
    let mut failures = 0usize;

    for m in 22..=35usize {
        for &profile in &profiles {
            let path = dir.join(format!("m{m}_{}.toml", profile.as_str()));
            match LigeritoSecurityConfig::derive_profile(m, profile) {
                Ok(cfg) => {
                    let toml = cfg.to_toml_string().expect("serialize");
                    // Round-trip to be sure the written form re-validates.
                    LigeritoSecurityConfig::from_toml_str(&toml)
                        .unwrap_or_else(|e| panic!("m={m}: written toml fails reload: {e}"));
                    std::fs::write(&path, &toml).expect("write toml");
                    let queries: usize = cfg.levels.iter().map(|l| l.queries).sum();
                    let ood: usize = cfg.levels.iter().map(|l| l.ood_samples).sum();
                    let max_fold_grind = cfg
                        .levels
                        .iter()
                        .map(|l| l.fold_grinding_bits)
                        .max()
                        .unwrap_or(0);
                    println!(
                        "write m={m} {:<6} -> {} (levels={}, Σqueries={queries}, Σood={ood}, max fold grind=2^{max_fold_grind})",
                        profile.as_str(),
                        path.file_name().unwrap().to_string_lossy(),
                        cfg.levels.len(),
                    );
                }
                Err(e) => {
                    eprintln!("FAIL  m={m} {}: derive failed: {e}", profile.as_str());
                    failures += 1;
                }
            }
        }
    }
    if failures > 0 {
        std::process::exit(1);
    }
}
