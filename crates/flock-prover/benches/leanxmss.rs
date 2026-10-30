//! leanXMSS[BLAKE3] aggregation: `LEANXMSS_N` signatures (default 1000) in one
//! proof, `LEANXMSS_RUNS` proves (default 5). The scratch pools are emptied
//! after every prove unless `LEANXMSS_KEEP_SCRATCH` is set: at 1000
//! signatures a warm pool holds about one more prove's worth of buffers,
//! which on a 24 GB machine pushes the next proves into memory compression.
//!
//! ```text
//! LEANXMSS_N=1000 cargo bench -p flock-prover --bench leanxmss
//! ```
//!
//! `LEANXMSS_NU` overrides the slot height `2^nu` the layout planner picks.
//!
//! Signatures are key-free (each costs two grinds of about 2^20 hashes) and
//! cached under `target/`.

use std::{
    env::var,
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use flock_prover::leanxmss::{
    circuit::{Statement, XmssCircuit, keyless_batch},
    reference::{PK_LEN, PublicKey, SIG_LEN, Signature},
};

/// Resident set size in bytes (macOS `task_info`), 0 elsewhere.
fn rss() -> u64 {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Info {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            times: [i32; 4],
            policy: i32,
            suspend_count: i32,
        }
        unsafe extern "C" {
            static mach_task_self_: u32;
            fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
        }
        let mut info = Info::default();
        let mut count = (size_of::<Info>() / 4) as u32;
        // SAFETY: MACH_TASK_BASIC_INFO (20) fills exactly `Info`.
        unsafe { task_info(mach_task_self_, 20, (&raw mut info).cast(), &mut count) };
        info.resident_size
    }
    #[cfg(not(target_os = "macos"))]
    0
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn cached(n: usize) -> (Vec<Statement>, Vec<Signature>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("leanxmss-blake3-keyless-v2-{n}.bin"));
    let record = PK_LEN + 32 + SIG_LEN;
    if let Ok(bytes) = std::fs::read(&path)
        && bytes.len() == n * record
    {
        return bytes
            .chunks_exact(record)
            .map(|r| {
                let pk = PublicKey::from_bytes(r[..PK_LEN].try_into().unwrap());
                let msg = r[PK_LEN..PK_LEN + 32].try_into().unwrap();
                let sig = Signature::from_bytes(&r[PK_LEN + 32..]).unwrap();
                (Statement { pk, msg }, sig)
            })
            .unzip();
    }
    let t = Instant::now();
    let (statements, sigs) = keyless_batch(n, 1);
    eprintln!("signed {n} in {:.1} s", t.elapsed().as_secs_f64());
    let mut bytes = Vec::with_capacity(n * record);
    for (st, sig) in statements.iter().zip(&sigs) {
        bytes.extend_from_slice(&st.pk.to_bytes());
        bytes.extend_from_slice(&st.msg);
        bytes.extend_from_slice(&sig.to_bytes());
    }
    let _ = std::fs::write(&path, bytes);
    (statements, sigs)
}

fn main() {
    // `LEANXMSS_POOL=perf`: P-core pool plus all-core hops (blake3_proof's
    // setup); default: rayon's pool over every core.
    let threads = (var("LEANXMSS_POOL").as_deref() == Ok("perf"))
        .then(flock_prover::init_perf_thread_pool)
        .flatten();
    let n: usize = var("LEANXMSS_N").map_or(1000, |v| v.parse().unwrap());
    let runs: usize = var("LEANXMSS_RUNS").map_or(5, |v| v.parse().unwrap());
    let clear = var("LEANXMSS_KEEP_SCRATCH").is_err();
    let (statements, sigs) = cached(n);
    let t = Instant::now();
    // `LEANXMSS_NU`: rows per slot as a power of two (default: `plan_nu`).
    let nu = var("LEANXMSS_NU").ok().map(|v| v.parse().unwrap());
    let circuit = XmssCircuit::with_nu(n, nu);
    eprintln!(
        "leanXMSS[BLAKE3] n = {n} (pool {threads:?}): nu = {}, shape {:.0} ms",
        circuit.nu(),
        t.elapsed().as_secs_f64() * 1e3
    );
    eprintln!("rss after shape: {:.2} GB", gb(rss()));
    // Samples RSS every 2 ms while a prove runs.
    static PEAK: AtomicU64 = AtomicU64::new(0);
    static STOP: AtomicBool = AtomicBool::new(false);
    let sampler = thread::spawn(|| {
        while !STOP.load(Ordering::Relaxed) {
            PEAK.fetch_max(rss(), Ordering::Relaxed);
            thread::sleep(Duration::from_millis(2));
        }
    });
    let mut best = f64::MAX;
    for run in 0..runs {
        let t = Instant::now();
        let witness = circuit.witness(&statements, &sigs);
        let online = t.elapsed().as_secs_f64();
        let before = rss();
        PEAK.store(before, Ordering::Relaxed);
        let t = Instant::now();
        let proof = circuit.prove(&witness);
        let prove = t.elapsed().as_secs_f64();
        let peak = PEAK.load(Ordering::Relaxed);
        drop(witness);
        let t = Instant::now();
        circuit.verify(&statements, &proof).expect("verifies");
        let verify = t.elapsed().as_secs_f64();
        best = best.min(online + prove);
        eprintln!(
            "run {run}: online {:.0} ms, prove {:.0} ms, verify {:.1} ms; rss {:.2} GB before prove, peak {:.2} GB",
            online * 1e3,
            prove * 1e3,
            verify * 1e3,
            gb(before),
            gb(peak),
        );
        if clear {
            flock_prover::scratch::clear();
        }
    }
    STOP.store(true, Ordering::Relaxed);
    sampler.join().unwrap();
    eprintln!(
        "best online+prove: {:.0} ms ({:.0} signatures/s)",
        best * 1e3,
        n as f64 / best
    );
}
