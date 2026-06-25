//! Streaming-fusion probe: skip materializing per-claim γ_k·B_k buffers,
//! generate them inline and XOR-accumulate into b_combined chunk-by-chunk
//! with prime accumulation in the same pass.
//!
//! Path A (current): per-claim fold (write 128 MB × 2) + combine (read 256 MB,
//! write 128 MB) + prime read of 128 MB. ~640-768 MB memory traffic.
//!
//! Path B (fused): per chunk, generate claim 0's contribution, XOR claim 1's
//! contribution, accumulate prime, write chunk to b_combined. Only b_combined
//! is written (128 MB total). Each chunk stays in L1 across all 3 stages.
//!
//! Run with: RAYON_NUM_THREADS=1 cargo bench --bench streaming_fusion_probe

use std::hint::black_box;
use std::time::Instant;

use flock_prover::field::F128;
use flock_prover::pcs::ring_switch::fold_b128_elems_split;

const N_BYTES: usize = 16;
const TABLE_SIZE: usize = 256;

fn build_phi_byte_table(eq_r_dprime: &[F128]) -> Vec<F128> {
    assert_eq!(eq_r_dprime.len(), 128);
    let mut tables = vec![F128 { lo: 0, hi: 0 }; N_BYTES * TABLE_SIZE];
    for byte_idx in 0..N_BYTES {
        let bit_base = byte_idx * 8;
        for value in 0..TABLE_SIZE {
            let mut acc = F128 { lo: 0, hi: 0 };
            for bit_in_byte in 0..8 {
                if (value >> bit_in_byte) & 1 == 1 {
                    acc += eq_r_dprime[bit_base + bit_in_byte];
                }
            }
            tables[byte_idx * TABLE_SIZE + value] = acc;
        }
    }
    tables
}

#[inline(always)]
fn apply_phi_byte_table(tables: &[F128], elem: F128) -> F128 {
    let lo_bytes = elem.lo.to_le_bytes();
    let hi_bytes = elem.hi.to_le_bytes();
    let tables_ptr = tables.as_ptr();
    let (l0, l1, l2, l3, l4, l5, l6, l7, h0, h1, h2, h3, h4, h5, h6, h7) = unsafe {
        (
            *tables_ptr.add(lo_bytes[0] as usize),
            *tables_ptr.add(TABLE_SIZE + lo_bytes[1] as usize),
            *tables_ptr.add(2 * TABLE_SIZE + lo_bytes[2] as usize),
            *tables_ptr.add(3 * TABLE_SIZE + lo_bytes[3] as usize),
            *tables_ptr.add(4 * TABLE_SIZE + lo_bytes[4] as usize),
            *tables_ptr.add(5 * TABLE_SIZE + lo_bytes[5] as usize),
            *tables_ptr.add(6 * TABLE_SIZE + lo_bytes[6] as usize),
            *tables_ptr.add(7 * TABLE_SIZE + lo_bytes[7] as usize),
            *tables_ptr.add(8 * TABLE_SIZE + hi_bytes[0] as usize),
            *tables_ptr.add(9 * TABLE_SIZE + hi_bytes[1] as usize),
            *tables_ptr.add(10 * TABLE_SIZE + hi_bytes[2] as usize),
            *tables_ptr.add(11 * TABLE_SIZE + hi_bytes[3] as usize),
            *tables_ptr.add(12 * TABLE_SIZE + hi_bytes[4] as usize),
            *tables_ptr.add(13 * TABLE_SIZE + hi_bytes[5] as usize),
            *tables_ptr.add(14 * TABLE_SIZE + hi_bytes[6] as usize),
            *tables_ptr.add(15 * TABLE_SIZE + hi_bytes[7] as usize),
        )
    };
    let p0 = l0 + l1;
    let p1 = l2 + l3;
    let p2 = l4 + l5;
    let p3 = l6 + l7;
    let p4 = h0 + h1;
    let p5 = h2 + h3;
    let p6 = h4 + h5;
    let p7 = h6 + h7;
    let q0 = p0 + p1;
    let q1 = p2 + p3;
    let q2 = p4 + p5;
    let q3 = p6 + p7;
    let r0 = q0 + q1;
    let r1 = q2 + q3;
    r0 + r1
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        F128 {
            lo: self.next_u64(),
            hi: self.next_u64(),
        }
    }
}

fn fmt_ms(s: f64) -> String {
    format!("{:>8.2} ms", s * 1000.0)
}

fn build_eq(r: &[F128]) -> Vec<F128> {
    let mut acc = vec![F128 { lo: 1, hi: 0 }];
    for &ri in r {
        let mut next = Vec::with_capacity(acc.len() * 2);
        let one = F128 { lo: 1, hi: 0 };
        for &a in &acc {
            next.push(a * (one + ri));
            next.push(a * ri);
        }
        acc = next;
    }
    acc
}

fn main() {
    let _ = flock_prover::init_perf_thread_pool();
    let threads = rayon::current_num_threads();
    println!("streaming-fusion probe ({threads} thread(s))\n");

    const M: usize = 30;
    let l = 1usize << (M - 7);
    let n_lo = (M - 7) / 2;
    let n_hi = (M - 7) - n_lo;
    let b_lo = 1usize << n_lo;
    let b_hi = 1usize << n_hi;
    assert_eq!(b_lo * b_hi, l);

    println!(
        "m = {M}, L = 2^{} = {l}, n_lo = {n_lo}, n_hi = {n_hi}",
        M - 7
    );
    println!(
        "  eq_lo size = {} KB, eq_hi size = {} KB",
        b_lo * 16 / 1024,
        b_hi * 16 / 1024
    );

    let mut rng = Rng::new(0xFEEDFACE);

    let eq_lo_0: Vec<F128> = (0..b_lo).map(|_| rng.f128()).collect();
    let eq_hi_0: Vec<F128> = (0..b_hi).map(|_| rng.f128()).collect();
    let eq_lo_1: Vec<F128> = (0..b_lo).map(|_| rng.f128()).collect();
    let eq_hi_1: Vec<F128> = (0..b_hi).map(|_| rng.f128()).collect();
    let r_dprime_0: Vec<F128> = (0..7).map(|_| rng.f128()).collect();
    let r_dprime_1: Vec<F128> = (0..7).map(|_| rng.f128()).collect();
    let eq_rd_0 = build_eq(&r_dprime_0);
    let eq_rd_1 = build_eq(&r_dprime_1);
    let g0 = rng.f128();
    let g1 = rng.f128();
    // a_init is the packed witness — random for the probe.
    let a_init: Vec<F128> = (0..l).map(|_| rng.f128()).collect();

    // γ-scale eq_r_dprime per claim.
    let scaled_0: Vec<F128> = eq_rd_0.iter().map(|x| g0 * *x).collect();
    let scaled_1: Vec<F128> = eq_rd_1.iter().map(|x| g1 * *x).collect();

    // Warm caches with one discard run.
    {
        let _b0 = fold_b128_elems_split(&eq_lo_0, &eq_hi_0, &scaled_0);
        let _b1 = fold_b128_elems_split(&eq_lo_1, &eq_hi_1, &scaled_1);
    }

    // ============================================================
    // Path A: current — materialize γ_k·B_k per claim, then combine
    // (par_chunks_mut(2) writes b_combined and accumulates prime).
    // ============================================================
    println!("\n[PATH A] current (materialize per-claim + fused combine + prime)");
    const RUNS: usize = 5;
    let mut a_fold0_total = 0.0;
    let mut a_fold1_total = 0.0;
    let mut a_combine_total = 0.0;
    let mut a_total = 0.0;
    let mut path_a_b = Vec::new();
    let mut path_a_u0 = F128::ZERO;
    let mut path_a_u2 = F128::ZERO;
    for run in 0..RUNS {
        let t_all = Instant::now();
        let t0 = Instant::now();
        let b0 = fold_b128_elems_split(&eq_lo_0, &eq_hi_0, &scaled_0);
        a_fold0_total += t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let b1 = fold_b128_elems_split(&eq_lo_1, &eq_hi_1, &scaled_1);
        a_fold1_total += t1.elapsed().as_secs_f64();

        // Combine + prime in one par_chunks_mut(2) pass (mirrors current pcs::open_batch_mixed).
        let tc = Instant::now();
        use rayon::prelude::*;
        let mut b_combined: Vec<F128> = vec![F128 { lo: 0, hi: 0 }; l];
        let (u_0, u_2) = b_combined
            .par_chunks_mut(2)
            .enumerate()
            .map(|(i, chunk)| {
                let v_a = b0[2 * i] + b1[2 * i];
                let v_b = b0[2 * i + 1] + b1[2 * i + 1];
                chunk[0] = v_a;
                chunk[1] = v_b;
                let a0 = a_init[2 * i];
                let a1 = a_init[2 * i + 1];
                (a0 * v_a, (a0 + a1) * (v_a + v_b))
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
            );
        a_combine_total += tc.elapsed().as_secs_f64();
        a_total += t_all.elapsed().as_secs_f64();

        if run == RUNS - 1 {
            path_a_b = b_combined;
            path_a_u0 = u_0;
            path_a_u2 = u_2;
        }
        black_box(&b0);
        black_box(&b1);
    }
    println!(
        "  fold[0]       avg {}",
        fmt_ms(a_fold0_total / RUNS as f64)
    );
    println!(
        "  fold[1]       avg {}",
        fmt_ms(a_fold1_total / RUNS as f64)
    );
    println!(
        "  combine+prime avg {}",
        fmt_ms(a_combine_total / RUNS as f64)
    );
    println!("  TOTAL         avg {}", fmt_ms(a_total / RUNS as f64));

    // ============================================================
    // Path B: streaming fusion — per chunk, write claim 0, XOR claim 1,
    // accumulate prime. Each chunk stays in L1 across all stages.
    // Byte tables (2 × 64 KB γ-baked) are built once up-front.
    // ============================================================
    println!("\n[PATH B] streaming fusion (per-chunk: write c0, XOR c1, prime)");
    let mut b_total = 0.0;
    let mut b_table_build_total = 0.0;
    let mut b_fused_pass_total = 0.0;
    let mut path_b_b = Vec::new();
    let mut path_b_u0 = F128::ZERO;
    let mut path_b_u2 = F128::ZERO;
    for run in 0..RUNS {
        let t_all = Instant::now();
        // Build γ-baked byte tables.
        let tb = Instant::now();
        let table_0 = build_phi_byte_table(&scaled_0);
        let table_1 = build_phi_byte_table(&scaled_1);
        b_table_build_total += tb.elapsed().as_secs_f64();

        // Streaming fusion. Chunk size = b_lo (the eq_lo span); each rayon
        // task processes one i_hi worth of slots (= one row of the
        // logical (i_hi, i_lo) grid).
        let tp = Instant::now();
        use rayon::prelude::*;
        let mut b_combined: Vec<F128> = vec![F128 { lo: 0, hi: 0 }; l];
        let (u_0, u_2) = b_combined
            .par_chunks_mut(b_lo)
            .enumerate()
            .map(|(i_hi, chunk)| {
                let e_hi_0 = eq_hi_0[i_hi];
                let e_hi_1 = eq_hi_1[i_hi];
                // Claim 0: write into chunk.
                for (i_lo, slot) in chunk.iter_mut().enumerate() {
                    let elem = eq_lo_0[i_lo] * e_hi_0;
                    *slot = apply_phi_byte_table(&table_0, elem);
                }
                // Claim 1: XOR-add into chunk.
                for (i_lo, slot) in chunk.iter_mut().enumerate() {
                    let elem = eq_lo_1[i_lo] * e_hi_1;
                    *slot += apply_phi_byte_table(&table_1, elem);
                }
                // Prime accumulation (pairs within chunk).
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
                let a_chunk_base = i_hi * b_lo;
                let mut i = 0;
                while i + 1 < chunk.len() {
                    let a0 = a_init[a_chunk_base + i];
                    let a1 = a_init[a_chunk_base + i + 1];
                    let v_a = chunk[i];
                    let v_b = chunk[i + 1];
                    u0 += a0 * v_a;
                    u2 += (a0 + a1) * (v_a + v_b);
                    i += 2;
                }
                (u0, u2)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
            );
        b_fused_pass_total += tp.elapsed().as_secs_f64();
        b_total += t_all.elapsed().as_secs_f64();

        if run == RUNS - 1 {
            path_b_b = b_combined;
            path_b_u0 = u_0;
            path_b_u2 = u_2;
        }
    }
    println!(
        "  build byte tables  avg {}",
        fmt_ms(b_table_build_total / RUNS as f64)
    );
    println!(
        "  fused chunk pass   avg {}",
        fmt_ms(b_fused_pass_total / RUNS as f64)
    );
    println!("  TOTAL              avg {}", fmt_ms(b_total / RUNS as f64));

    // ============================================================
    // Path C: micro-tiled streaming fusion. Process the chunk in
    // small (256-element = 4 KB) micro-tiles. Per micro-tile:
    //   Sub-pass 1: write scratchpad from Φ_0 (only table_0 active)
    //   Sub-pass 2: XOR Φ_1 into scratchpad (only table_1 active)
    //   Sub-pass 3: prime + write chunk[micro range]
    // Each sub-pass keeps only ONE byte table hot, reducing per-set
    // L1 cache pressure (12-way × 256 sets = 12 lines/set capacity;
    // two simultaneously-hot 64 KB tables = 8 lines/set, plus chunk
    // and eq_lo lines pushed it over).
    // ============================================================
    println!("\n[PATH C] micro-tiled fusion (256-element scratchpad, 1 active table per sub-pass)");
    const MICRO: usize = 256;
    let mut c_total = 0.0;
    let mut c_table_build_total = 0.0;
    let mut c_fused_pass_total = 0.0;
    let mut path_c_b = Vec::new();
    let mut path_c_u0 = F128::ZERO;
    let mut path_c_u2 = F128::ZERO;
    for run in 0..RUNS {
        let t_all = Instant::now();
        let tb = Instant::now();
        let table_0 = build_phi_byte_table(&scaled_0);
        let table_1 = build_phi_byte_table(&scaled_1);
        c_table_build_total += tb.elapsed().as_secs_f64();

        let tp = Instant::now();
        use rayon::prelude::*;
        let mut b_combined: Vec<F128> = vec![F128 { lo: 0, hi: 0 }; l];
        let (u_0, u_2) = b_combined
            .par_chunks_mut(b_lo)
            .enumerate()
            .map(|(i_hi, chunk)| {
                let e_hi_0 = eq_hi_0[i_hi];
                let e_hi_1 = eq_hi_1[i_hi];
                let mut u0 = F128::ZERO;
                let mut u2 = F128::ZERO;
                let mut scratch = [F128 { lo: 0, hi: 0 }; MICRO];
                let a_chunk_base = i_hi * b_lo;
                let chunk_len = chunk.len();
                let mut i_lo_base = 0;
                while i_lo_base < chunk_len {
                    let micro_end = (i_lo_base + MICRO).min(chunk_len);
                    let micro_len = micro_end - i_lo_base;

                    // Sub-pass 1: only table_0 hot.
                    for i in 0..micro_len {
                        let i_lo = i_lo_base + i;
                        let elem = eq_lo_0[i_lo] * e_hi_0;
                        scratch[i] = apply_phi_byte_table(&table_0, elem);
                    }
                    // Sub-pass 2: only table_1 hot (table_0 may evict).
                    for i in 0..micro_len {
                        let i_lo = i_lo_base + i;
                        let elem = eq_lo_1[i_lo] * e_hi_1;
                        scratch[i] += apply_phi_byte_table(&table_1, elem);
                    }
                    // Sub-pass 3: prime + write chunk slice.
                    let mut i = 0;
                    while i + 1 < micro_len {
                        let a0 = a_init[a_chunk_base + i_lo_base + i];
                        let a1 = a_init[a_chunk_base + i_lo_base + i + 1];
                        let v_a = scratch[i];
                        let v_b = scratch[i + 1];
                        chunk[i_lo_base + i] = v_a;
                        chunk[i_lo_base + i + 1] = v_b;
                        u0 += a0 * v_a;
                        u2 += (a0 + a1) * (v_a + v_b);
                        i += 2;
                    }
                    // Handle odd tail of micro-batch (rare).
                    if i < micro_len {
                        chunk[i_lo_base + i] = scratch[i];
                    }
                    i_lo_base = micro_end;
                }
                (u0, u2)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(x0, x2), (y0, y2)| (x0 + y0, x2 + y2),
            );
        c_fused_pass_total += tp.elapsed().as_secs_f64();
        c_total += t_all.elapsed().as_secs_f64();

        if run == RUNS - 1 {
            path_c_b = b_combined;
            path_c_u0 = u_0;
            path_c_u2 = u_2;
        }
    }
    println!(
        "  build byte tables  avg {}",
        fmt_ms(c_table_build_total / RUNS as f64)
    );
    println!(
        "  micro-tiled pass   avg {}",
        fmt_ms(c_fused_pass_total / RUNS as f64)
    );
    println!("  TOTAL              avg {}", fmt_ms(c_total / RUNS as f64));

    let mismatches_c: usize = path_a_b
        .iter()
        .zip(path_c_b.iter())
        .filter(|(a, c)| a != c)
        .count();
    println!("\nbyte-identical (A vs C):");
    println!(
        "  b_combined: {} / {} mismatches",
        mismatches_c,
        path_a_b.len()
    );
    println!("  u_0 match: {}", path_a_u0 == path_c_u0);
    println!("  u_2 match: {}", path_a_u2 == path_c_u2);
    assert_eq!(mismatches_c, 0);
    assert_eq!(path_a_u0, path_c_u0);
    assert_eq!(path_a_u2, path_c_u2);

    // Verify byte-identical: b_combined, u_0, u_2 must match.
    let mismatches: usize = path_a_b
        .iter()
        .zip(path_b_b.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!("\nbyte-identical check:");
    println!(
        "  b_combined: {} / {} mismatches",
        mismatches,
        path_a_b.len()
    );
    println!(
        "  u_0: A={:?} B={:?} match={}",
        path_a_u0,
        path_b_u0,
        path_a_u0 == path_b_u0
    );
    println!(
        "  u_2: A={:?} B={:?} match={}",
        path_a_u2,
        path_b_u2,
        path_a_u2 == path_b_u2
    );
    assert_eq!(mismatches, 0);
    assert_eq!(path_a_u0, path_b_u0);
    assert_eq!(path_a_u2, path_b_u2);

    println!(
        "\nDelta (A - B): {:.2} ms",
        (a_total - b_total) / RUNS as f64 * 1000.0
    );
}
