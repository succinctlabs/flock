//! Builds the CUDA prover static library when the `gpu` feature is enabled.
//!
//! Without `--features gpu` this script does nothing, so the workspace builds
//! on machines without nvcc. With it, `cuda-ghash/prove_ffi.cu` is compiled
//! for sm_120 (the inline-PTX clmad kernels are Blackwell-only) into a static
//! archive the test binary links against.

use env::var;
use env::var_os;
use std::env;
use std::fs::read_dir;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest.parent().unwrap().parent().unwrap();
    let cuda_dir = repo.join("cuda-ghash");
    let src = cuda_dir.join("prove_ffi.cu");
    let out_dir = PathBuf::from(var("OUT_DIR").unwrap());
    let lib = out_dir.join("libflock_cuda_prover.a");

    // Rebuild when the FFI TU or any header it includes changes.
    println!("cargo:rerun-if-changed={}", src.display());
    for entry in read_dir(&cuda_dir).unwrap().flatten() {
        let p = entry.path();
        if p.extension()
            .is_some_and(|e| e == "cuh" || e == "hpp" || e == "h")
        {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
    println!("cargo:rerun-if-env-changed=NVCC");

    let nvcc = var("NVCC").unwrap_or_else(|_| {
        for cand in ["/usr/local/cuda/bin/nvcc", "nvcc"] {
            if Command::new(cand).arg("--version").output().is_ok() {
                return cand.to_string();
            }
        }
        panic!("nvcc not found: set NVCC or install CUDA at /usr/local/cuda")
    });

    let status = Command::new(&nvcc)
        .args(["-O3", "-std=c++17", "-lineinfo"])
        .args(["-gencode", "arch=compute_120,code=sm_120"])
        .args(["-Xcompiler", "-fPIC"])
        .arg("-lib")
        .arg(&src)
        .arg("-o")
        .arg(&lib)
        .status()
        .expect("failed to spawn nvcc");
    assert!(status.success(), "nvcc failed on {}", src.display());

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=flock_cuda_prover");
    // CUDA runtime + C++ host runtime for the nvcc-generated host code.
    for dir in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs"] {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
