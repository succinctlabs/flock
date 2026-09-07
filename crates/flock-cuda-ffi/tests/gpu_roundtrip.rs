//! GPU prove -> Rust verify roundtrips. Needs a Blackwell GPU (sm_120):
//!   cargo test -p flock-cuda-ffi --release --features gpu -- --ignored --nocapture
//!
//! The CUDA prover (cuda-ghash/prove_ffi.cu) returns a flat little-endian
//! stream; `gpu_prove` mirrors its FfiWriter layout exactly, rebuilds the
//! typed `R1csProofLigerito` (F256 ladder: capped Merkle commitments,
//! stratified queries, the full 128-bit grinding schedule), and runs the
//! ordinary Rust verifier. One roundtrip per proof size: m = 14 +
//! n_blocks_log, gated on a Ligerito config existing for that m.
#![cfg(feature = "gpu")]

use std::{
    ffi::{CString, c_char},
    fmt::Debug,
    fs::read,
    ptr::{null, null_mut},
    slice::from_raw_parts,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use b3::build_block_r1cs;
use flock_cuda_ffi::gpu::device_count;
use flock_prover::{
    challenger::FsChallenger,
    field::{F8, F128, F256},
    lincheck::{self, LincheckProof},
    ntt::AdditiveNttGf8,
    pcs::{
        BatchOpeningProofLigerito, Commitment, PcsParams,
        ligerito::{FinalProof, LigeritoProof, RecursiveProof, SumcheckMessage256},
        ring_switch::RingSwitchProof,
    },
    proof::R1csProofLigerito,
    prover::prove_ligerito,
    r1cs::{BlockR1cs, SparseBinaryMatrix},
    r1cs_hashes::blake3 as b3,
    verifier,
    zerocheck::{K_SKIP, ZerocheckProof},
};
use lincheck::SparseMatrixCircuit;
use verifier::verify_ligerito;

const DOMAIN: &[u8] = b"flock-lig-r1cs-v0";
static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
static BLAKE3_CSC_MATRICES: OnceLock<CscMatrices> = OnceLock::new();

struct CscMatrices {
    a_col_ptr: Vec<u32>,
    a_rows: Vec<u32>,
    b_col_ptr: Vec<u32>,
    b_rows: Vec<u32>,
}

#[repr(C)]
struct ProveParams {
    m: i32,
    statement_digest: *const u8,
    domain: *const u8,
    domain_len: u32,
    a_col_ptr: *const u32,
    a_rows: *const u32,
    a_nnz: u32,
    b_col_ptr: *const u32,
    b_rows: *const u32,
    b_nnz: u32,
    const_pin_col: i32,
    useful_bits: i32,
    k_log: i32,
    zc_mcol: *const u8,
    zc_f8mul: *const u8,
    initial_k: i32,
    num_levels: i32,
    log_inv_rates: *const i32,
    recursive_ks: *const i32,
    queries: *const i32,
    grinding_bits: *const i32,
    claim_batch_grinding_bits: *const i32,
    consistency_batch_grinding_bits: *const i32,
    ood_samples: *const i32,
    recursive_steps: i32,
    zc_initial_bits: i32,
    zc_skip_bits: i32,
    zc_round_bits: i32,
    lc_alpha_bits: i32,
    lc_beta_bits: i32,
    lc_round_bits: i32,
    lc_skip_bits: i32,
    rs_bits: i32,
    gamma_bits: i32,
    dump_z_path: *const c_char,
}

unsafe extern "C" {
    fn flock_cuda_prove_blake3(
        p: *const ProveParams,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> i32;
    fn flock_cuda_free(p: *mut u8);
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_link_smoke() {
    let n = device_count();
    assert!(n > 0, "no CUDA device visible (got {n})");
}

// `lincheck.rs::csc_from_rows` twin (same as dump_lincheck_vectors).
fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

// Zerocheck round-1 kernel tables (same as dump_zerocheck_full_vectors).
fn zc_tables() -> (Vec<u8>, Vec<u8>) {
    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
    let mut mcol = vec![0u8; 64 * 64];
    for s in 0..64 {
        let mut col = vec![F8::ZERO; 64];
        col[s] = F8(1);
        ntt_s.inverse(&mut col);
        ntt_l.forward(&mut col);
        for i in 0..64 {
            mcol[s * 64 + i] = col[i].0;
        }
    }
    let mut f8mul = vec![0u8; 256 * 256];
    for x in 0..256usize {
        for y in 0..256usize {
            f8mul[x * 256 + y] = (F8(x as u8) * F8(y as u8)).0;
        }
    }
    (mcol, f8mul)
}

struct Reader<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> Reader<'a> {
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.o..self.o + 8].try_into().unwrap());
        self.o += 8;
        v
    }
    fn u64s(&mut self) -> Vec<u64> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.u64()).collect()
    }
    fn f128(&mut self) -> F128 {
        let lo = self.u64();
        let hi = self.u64();
        F128 { lo, hi }
    }
    fn f256(&mut self) -> F256 {
        let c0 = self.f128();
        let c1 = self.f128();
        F256::new(c0, c1)
    }
    fn f128s(&mut self) -> Vec<F128> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.f128()).collect()
    }
    fn hash(&mut self) -> [u8; 32] {
        let h: [u8; 32] = self.b[self.o..self.o + 32].try_into().unwrap();
        self.o += 32;
        h
    }
    fn hashes(&mut self) -> Vec<[u8; 32]> {
        let n = self.u64() as usize;
        (0..n).map(|_| self.hash()).collect()
    }
    fn rows(&mut self) -> Vec<Vec<F128>> {
        let n_rows = self.u64() as usize;
        let row_len = self.u64() as usize;
        (0..n_rows)
            .map(|_| (0..row_len).map(|_| self.f128()).collect())
            .collect()
    }
}

struct GpuArtifacts {
    r1cs: BlockR1cs,
    pcs_params: PcsParams,
    proof: R1csProofLigerito,
    commitment: Commitment,
    prove_secs: f64,
}

/// Prove on the GPU at `m = 14 + n_blocks_log`, parse the flat stream into the
/// typed proof. `dump_z` optionally writes the packed witness for host replay.
fn gpu_prove(n_blocks_log: usize, dump_z: Option<&str>) -> GpuArtifacts {
    let r1cs = build_block_r1cs(n_blocks_log);
    let m = r1cs.m;
    let pcs_params = PcsParams {
        m,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash: Default::default(),
    };
    let cfg = pcs_params
        .ligerito_prover_config()
        .unwrap_or_else(|_| panic!("no fast ligerito config for m={m}"));
    assert!(
        cfg.fold_grinding_bits.iter().all(|&b| b == 0),
        "the F256 ladder never fold-grinds"
    );

    let digest = r1cs.statement_digest();
    let matrices = BLAKE3_CSC_MATRICES.get_or_init(|| {
        let (a_col_ptr, a_rows) = csc_from_rows(&r1cs.a_0);
        let (b_col_ptr, b_rows) = csc_from_rows(&r1cs.b_0);
        CscMatrices {
            a_col_ptr,
            a_rows,
            b_col_ptr,
            b_rows,
        }
    });
    let (mcol, f8mul) = zc_tables();

    let to_i32 = |v: &[usize]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
    let log_inv_rates = to_i32(&cfg.log_inv_rates);
    let recursive_ks = to_i32(&cfg.recursive_ks);
    let queries = to_i32(&cfg.queries);
    let grinding_bits = to_i32(&cfg.grinding_bits);
    let claim_batch = to_i32(&cfg.claim_batch_grinding_bits);
    let consistency_batch = to_i32(&cfg.consistency_batch_grinding_bits);
    let ood_samples = to_i32(&cfg.ood_samples);
    let r_steps = cfg.recursive_steps;

    // The 128-bit FS grinding schedule the profile selects (0 = site absent).
    let zc = pcs_params.zerocheck_grinding();
    let lc = pcs_params.lincheck_grinding();
    let og = pcs_params.opening_grinding();
    let opt = |b: Option<u32>| b.map_or(0, |x| x as i32);

    let dump_c = dump_z.map(|p| CString::new(p).unwrap());
    let params = ProveParams {
        m: m as i32,
        statement_digest: digest.as_ptr(),
        domain: DOMAIN.as_ptr(),
        domain_len: DOMAIN.len() as u32,
        a_col_ptr: matrices.a_col_ptr.as_ptr(),
        a_rows: matrices.a_rows.as_ptr(),
        a_nnz: matrices.a_rows.len() as u32,
        b_col_ptr: matrices.b_col_ptr.as_ptr(),
        b_rows: matrices.b_rows.as_ptr(),
        b_nnz: matrices.b_rows.len() as u32,
        const_pin_col: r1cs.const_pin.map_or(-1, |c| c as i32),
        useful_bits: r1cs.useful_bits as i32,
        k_log: r1cs.k_log as i32,
        zc_mcol: mcol.as_ptr(),
        zc_f8mul: f8mul.as_ptr(),
        initial_k: cfg.initial_k as i32,
        num_levels: log_inv_rates.len() as i32,
        log_inv_rates: log_inv_rates.as_ptr(),
        recursive_ks: recursive_ks.as_ptr(),
        queries: queries.as_ptr(),
        grinding_bits: grinding_bits.as_ptr(),
        claim_batch_grinding_bits: claim_batch.as_ptr(),
        consistency_batch_grinding_bits: consistency_batch.as_ptr(),
        ood_samples: ood_samples.as_ptr(),
        recursive_steps: r_steps as i32,
        zc_initial_bits: opt(zc.initial_bits(m)),
        zc_skip_bits: opt(zc.skip_bits()),
        zc_round_bits: opt(zc.multilinear_round_bits()),
        lc_alpha_bits: opt(lc.alpha_bits()),
        lc_beta_bits: opt(lc.beta_bits()),
        lc_round_bits: opt(lc.multilinear_round_bits()),
        lc_skip_bits: opt(lc.skip_bits(K_SKIP)),
        rs_bits: og.ring_switch_bits as i32,
        gamma_bits: og.claim_batch_bits as i32, // 2 claims > 1 → the batch grinds
        dump_z_path: dump_c.as_ref().map_or(null(), |c| c.as_ptr()),
    };

    let t0 = Instant::now();
    let mut out: *mut u8 = null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe { flock_cuda_prove_blake3(&params, &mut out, &mut out_len) };
    assert_eq!(rc, 0, "CUDA prover returned error {rc} at m={m}");
    let bytes = unsafe { from_raw_parts(out, out_len) }.to_vec();
    unsafe { flock_cuda_free(out) };
    let t_prove = t0.elapsed();

    // ---- parse the flat stream (must mirror prove_ffi.cu::FfiWriter) ----
    let mut r = Reader { b: &bytes, o: 0 };
    let cap = r.hashes();
    let round1_ab = r.f128s();
    let round1_c = r.f128s();
    let n_mlv = r.u64() as usize;
    let multilinear_rounds: Vec<(F128, F128)> = (0..n_mlv).map(|_| (r.f128(), r.f128())).collect();
    let final_a_eval = r.f128();
    let final_b_eval = r.f128();
    let final_c_eval = r.f128();
    let zc_nonces = r.u64s();
    let n_lc = r.u64() as usize;
    let lc_rounds: Vec<(F128, F128)> = (0..n_lc).map(|_| (r.f128(), r.f128())).collect();
    let z_partial = r.f128s();
    let lc_nonces = r.u64s();
    let shat_ab = r.f128s();
    let rs_nonce_ab = r.u64();
    let shat_c = r.f128s();
    let rs_nonce_c = r.u64();
    let batching_nonces = r.u64s();
    let n_rcaps = r.u64() as usize;
    let recursive_caps: Vec<Vec<[u8; 32]>> = (0..n_rcaps).map(|_| r.hashes()).collect();
    let n_opens = r.u64() as usize;
    assert_eq!(n_opens, r_steps + 1, "level opens = r+1");
    let mut opens: Vec<(Vec<Vec<F128>>, Vec<[u8; 32]>)> = (0..n_opens)
        .map(|_| {
            let rows = r.rows();
            let proof = r.hashes();
            (rows, proof)
        })
        .collect();
    let yr = r.f128s();
    let n_sc = r.u64() as usize;
    let sumcheck_transcript_f256: Vec<SumcheckMessage256> = (0..n_sc)
        .map(|_| SumcheckMessage256 {
            u_0: r.f256(),
            u_2: r.f256(),
        })
        .collect();
    let ood_values = r.f128s();
    let grinding_nonces = r.u64s();
    let claim_batch_grinding_nonces = r.u64s();
    let consistency_batch_grinding_nonces = r.u64s();
    assert_eq!(r.o, bytes.len(), "trailing bytes in FFI stream");

    let (initial_rows, initial_mp) = opens.remove(0);
    let (final_rows, final_mp) = opens.pop().expect("final open");
    let recursive_proofs: Vec<RecursiveProof> = opens
        .into_iter()
        .map(|(rows, mp)| RecursiveProof {
            opened_rows: rows,
            merkle_proof: mp,
        })
        .collect();

    let proof = R1csProofLigerito {
        zerocheck: ZerocheckProof {
            round1_ab,
            round1_c,
            multilinear_rounds,
            final_a_eval,
            final_b_eval,
            final_c_eval,
            grinding_nonces: zc_nonces,
        },
        lincheck: LincheckProof {
            rounds: lc_rounds,
            z_partial,
            matrix_evals: Vec::new(),
            grinding_nonces: lc_nonces,
        },
        pcs_open: BatchOpeningProofLigerito {
            ring_switches: vec![
                RingSwitchProof {
                    s_hat_v: shat_ab,
                    grinding_nonce: rs_nonce_ab,
                },
                RingSwitchProof {
                    s_hat_v: shat_c,
                    grinding_nonce: rs_nonce_c,
                },
            ],
            batching_nonces,
            ligerito: LigeritoProof {
                initial_cap: cap.clone(),
                initial_proof: RecursiveProof {
                    opened_rows: initial_rows,
                    merkle_proof: initial_mp,
                },
                recursive_caps,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: final_rows,
                    merkle_proof: final_mp,
                },
                sumcheck_transcript: Vec::new(),
                sumcheck_transcript_f256,
                grinding_nonces,
                ood_values,
                fold_grinding_nonces: Vec::new(),
                claim_batch_grinding_nonces,
                consistency_batch_grinding_nonces,
            },
        },
    };
    GpuArtifacts {
        r1cs,
        pcs_params: pcs_params.clone(),
        proof,
        commitment: Commitment {
            cap,
            params: pcs_params,
        },
        prove_secs: t_prove.as_secs_f64(),
    }
}

/// Full roundtrip: GPU prove, Rust verify; with `tamper`, also check that two
/// corrupted variants are rejected.
fn roundtrip<const N_BLOCKS_LOG: usize, const TAMPER: bool>() {
    let _test_guard = GPU_TEST_LOCK.lock().expect("GPU test lock poisoned");
    let warmup_secs = gpu_prove(N_BLOCKS_LOG, None).prove_secs;
    let GpuArtifacts {
        r1cs,
        pcs_params,
        proof,
        commitment,
        prove_secs,
    } = gpu_prove(N_BLOCKS_LOG, None);
    let m = r1cs.m;
    let lc_circuit = SparseMatrixCircuit::new(&r1cs.a_0, &r1cs.b_0).with_const_pin(r1cs.const_pin);
    let t1 = Instant::now();
    let mut ch_v = FsChallenger::new(DOMAIN);
    let claim = verify_ligerito(
        &r1cs,
        &commitment,
        &proof,
        &lc_circuit,
        &pcs_params,
        &mut ch_v,
    )
    .unwrap_or_else(|e| panic!("Rust verifier rejected the GPU proof at m={m}: {e:?}"));
    println!(
        "GPU proof verified: m={m}, warmup {:.4}s, steady prove(+glue) {:.4}s, verify {:.4}s, ab claim {:016x}:{:016x}",
        warmup_secs,
        prove_secs,
        t1.elapsed().as_secs_f64(),
        claim.ab.value.hi,
        claim.ab.value.lo
    );

    if TAMPER {
        // Flip one bit of the final-level clear polynomial -> reject.
        let mut bad = proof.clone();
        bad.pcs_open.ligerito.final_proof.yr[0].lo ^= 1;
        let mut ch_t = FsChallenger::new(DOMAIN);
        assert!(
            verify_ligerito(
                &r1cs,
                &commitment,
                &bad,
                &lc_circuit,
                &pcs_params,
                &mut ch_t
            )
            .is_err(),
            "verifier accepted a tampered GPU proof"
        );
        // Corrupt one zerocheck round message -> transcript replay rejects.
        let mut bad = proof.clone();
        bad.zerocheck.multilinear_rounds[0].0.hi ^= 1;
        let mut ch_t = FsChallenger::new(DOMAIN);
        assert!(
            verify_ligerito(
                &r1cs,
                &commitment,
                &bad,
                &lc_circuit,
                &pcs_params,
                &mut ch_t
            )
            .is_err(),
            "verifier accepted a zerocheck-tampered GPU proof"
        );
    }
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_roundtrip_m22() {
    roundtrip::<8, true>();
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_roundtrip_m32() {
    roundtrip::<18, false>();
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_roundtrip_m33() {
    roundtrip::<19, false>();
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_roundtrip_m34() {
    roundtrip::<20, false>();

    for _ in 0..5 {
        roundtrip::<20, false>();
    }
}

/// Debug harness: prove the SAME witness on the GPU and in Rust, then report
/// the first divergent proof field.
#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_debug_diff_m22() {
    gpu_debug_diff::<22>();
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_debug_diff_m33() {
    gpu_debug_diff::<33>();
}

#[test]
#[ignore] // needs an sm_120 GPU; run explicitly with --ignored
fn gpu_debug_diff_m34() {
    gpu_debug_diff::<34>();
}

fn gpu_debug_diff<const M: usize>() {
    const { assert!(M >= 17, "need m >= 17 (n_blocks_log >= 3)") };
    let m = M;
    let zpath = format!("/tmp/ffi_z_m{m}.bin");
    let art = gpu_prove(m - 14, Some(&zpath));
    let zb = read(&zpath).expect("witness dump missing");
    assert_eq!(zb.len(), (1usize << (m - 7)) * 16, "witness dump size");
    let z: Vec<F128> = zb
        .chunks_exact(16)
        .map(|c| F128 {
            lo: u64::from_le_bytes(c[0..8].try_into().unwrap()),
            hi: u64::from_le_bytes(c[8..16].try_into().unwrap()),
        })
        .collect();

    let mut ch = FsChallenger::new(DOMAIN);
    let (rp, rcomm, _claim) = prove_ligerito(&art.r1cs, z, &art.pcs_params, &mut ch);
    let gp = &art.proof;

    fn first_diff<T: PartialEq + Debug>(name: &str, r: &[T], g: &[T]) {
        if r.len() != g.len() {
            panic!("{name}: len rust {} vs gpu {}", r.len(), g.len());
        }
        for (i, (a, b)) in r.iter().zip(g.iter()).enumerate() {
            if a != b {
                panic!("{name}[{i}]: rust {a:?} vs gpu {b:?}");
            }
        }
        println!("  {name}: OK ({} entries)", r.len());
    }

    first_diff("commitment.cap", &rcomm.cap, &art.commitment.cap);
    first_diff(
        "zc.round1_ab",
        &rp.zerocheck.round1_ab,
        &gp.zerocheck.round1_ab,
    );
    first_diff(
        "zc.round1_c",
        &rp.zerocheck.round1_c,
        &gp.zerocheck.round1_c,
    );
    first_diff(
        "zc.multilinear_rounds",
        &rp.zerocheck.multilinear_rounds,
        &gp.zerocheck.multilinear_rounds,
    );
    first_diff(
        "zc.grinding_nonces",
        &rp.zerocheck.grinding_nonces,
        &gp.zerocheck.grinding_nonces,
    );
    assert_eq!(
        rp.zerocheck.final_a_eval, gp.zerocheck.final_a_eval,
        "zc.final_a"
    );
    assert_eq!(
        rp.zerocheck.final_b_eval, gp.zerocheck.final_b_eval,
        "zc.final_b"
    );
    assert_eq!(
        rp.zerocheck.final_c_eval, gp.zerocheck.final_c_eval,
        "zc.final_c"
    );
    first_diff("lc.rounds", &rp.lincheck.rounds, &gp.lincheck.rounds);
    first_diff(
        "lc.z_partial",
        &rp.lincheck.z_partial,
        &gp.lincheck.z_partial,
    );
    first_diff(
        "lc.grinding_nonces",
        &rp.lincheck.grinding_nonces,
        &gp.lincheck.grinding_nonces,
    );
    for (i, (r, g)) in rp
        .pcs_open
        .ring_switches
        .iter()
        .zip(gp.pcs_open.ring_switches.iter())
        .enumerate()
    {
        first_diff(&format!("rs[{i}].s_hat_v"), &r.s_hat_v, &g.s_hat_v);
        assert_eq!(r.grinding_nonce, g.grinding_nonce, "rs[{i}].grinding_nonce");
    }
    first_diff(
        "batching_nonces",
        &rp.pcs_open.batching_nonces,
        &gp.pcs_open.batching_nonces,
    );
    let (rl, gl) = (&rp.pcs_open.ligerito, &gp.pcs_open.ligerito);
    first_diff("lig.initial_cap", &rl.initial_cap, &gl.initial_cap);
    first_diff(
        "lig.sumcheck_transcript_f256",
        &rl.sumcheck_transcript_f256,
        &gl.sumcheck_transcript_f256,
    );
    first_diff("lig.ood_values", &rl.ood_values, &gl.ood_values);
    first_diff(
        "lig.grinding_nonces",
        &rl.grinding_nonces,
        &gl.grinding_nonces,
    );
    first_diff(
        "lig.claim_batch_grinding_nonces",
        &rl.claim_batch_grinding_nonces,
        &gl.claim_batch_grinding_nonces,
    );
    first_diff(
        "lig.consistency_batch_grinding_nonces",
        &rl.consistency_batch_grinding_nonces,
        &gl.consistency_batch_grinding_nonces,
    );
    assert_eq!(
        rl.recursive_caps.len(),
        gl.recursive_caps.len(),
        "recursive_caps count"
    );
    for (i, (r, g)) in rl
        .recursive_caps
        .iter()
        .zip(gl.recursive_caps.iter())
        .enumerate()
    {
        first_diff(&format!("lig.rcap[{i}]"), r, g);
    }
    first_diff(
        "lig.initial.rows",
        &rl.initial_proof.opened_rows,
        &gl.initial_proof.opened_rows,
    );
    first_diff(
        "lig.initial.mp",
        &rl.initial_proof.merkle_proof,
        &gl.initial_proof.merkle_proof,
    );
    assert_eq!(
        rl.recursive_proofs.len(),
        gl.recursive_proofs.len(),
        "recursive_proofs count"
    );
    for (i, (r, g)) in rl
        .recursive_proofs
        .iter()
        .zip(gl.recursive_proofs.iter())
        .enumerate()
    {
        first_diff(
            &format!("lig.rec[{i}].rows"),
            &r.opened_rows,
            &g.opened_rows,
        );
        first_diff(
            &format!("lig.rec[{i}].mp"),
            &r.merkle_proof,
            &g.merkle_proof,
        );
    }
    first_diff("lig.final.yr", &rl.final_proof.yr, &gl.final_proof.yr);
    first_diff(
        "lig.final.rows",
        &rl.final_proof.opened_rows,
        &gl.final_proof.opened_rows,
    );
    first_diff(
        "lig.final.mp",
        &rl.final_proof.merkle_proof,
        &gl.final_proof.merkle_proof,
    );
    println!("no divergence found (proofs identical at m={m})");
}
