// End-to-end Ligerito open prover benchmark (GPU pcs::open, step 7) — runs the
// full prove (no oracle, no validation): host FsChallenger derives every
// challenge, device kernels do the compute, host does multi-proof + induce
// setup. Times each phase via wall clock (device synced at boundaries).
//
// Mirrors test_ligerito_l0.cu's prove exactly, minus the byte-for-byte checks.
//
// Build:  make bench_ligerito
// Run:    ./bench_ligerito log_n initial_k log_inv_rate_0 num_queries_0 \
//                          log_inv_rate_1 ood r k_rec rate_rec ood_rec nq_rec [iters]
//   default: 22 5 1 148 1 1 2 3 1 1 148
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include "ntt_f128.cuh"
#include "merkle.cuh"
#include "merkle_open.hpp"
#include "merkle_open_device.cuh"
#include "induce_sumcheck.cuh"
#include "ntt_transpose.cuh"
#include "introduce_glue.cuh"
#include "sumcheck_ab.cuh"
#include "lincheck.cuh"
#include "blake3_witness.cuh"
#include "zerocheck_round1.cuh"
#include "zerocheck_round1_cpustyle.cuh"
#include "zerocheck_round2.cuh"
#include "zerocheck_tail.cuh"
#include "phi8_table.cuh"
#include "challenger.hpp"
#include "zc_challenger_device.cuh"   // resident on-device challenger for the tail
static ZcSha zc_pack(const Sha256& s){ ZcSha z; for(int i=0;i<8;i++)z.h[i]=s.h[i]; z.total_len=s.total_len;
    for(int i=0;i<64;i++)z.buf[i]=s.buf[i]; z.buf_len=(unsigned)s.buf_len; return z; }
static void zc_unpack(Sha256& s, const ZcSha& z){ for(int i=0;i<8;i++)s.h[i]=z.h[i]; s.total_len=z.total_len;
    for(int i=0;i<64;i++)s.buf[i]=z.buf[i]; s.buf_len=z.buf_len; }

#define CK(x) do { cudaError_t e=(x); if(e){ printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)
static const uint8_t PROVER_LABEL[] = "flock-ligerito-basis-v0";
using Clock = std::chrono::steady_clock;
static double ms_wall(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
static double ms_since(Clock::time_point t) { CK(cudaDeviceSynchronize()); return ms_wall(t); }

// Per-prove wall-clock accounting. Every millisecond of prove() lands in exactly
// one bucket: a Charge scope subtracts the device alloc/free/H2D time recorded
// inside its window, and that time lands in cuda_malloc/cuda_free/h_to_d
// instead. main() checks the identity `total() == prove wall clock`; a gap means
// real work nobody is measuring (the reason this accounting exists — the phase
// breakdown used to omit ~100 ms of host-side eq build and H2D per prove).
// Reconstruct a quadratic from its values at 0, 1 and infinity (the leading
// coefficient) and evaluate at rho. Used by the two-round lookahead: the second
// round's message components are quadratics in the first round's challenge.
static F128 zt_interp3(F128 h0, F128 h1, F128 hinf, F128 rho) {
    F128 c1 = f128_add_hd(f128_add_hd(h0, h1), hinf);        // char 2: h(1) = c0+c1+c2
    return f128_add_hd(h0, f128_mul_hd(rho, f128_add_hd(c1, f128_mul_hd(rho, hinf))));
}

struct Phase {
    double commit=0, fold=0, ood=0, open=0, induce=0, intro=0, lincheck=0,
           witness=0, zerocheck=0, l0commit=0, eq_build=0;
    double cuda_malloc=0, cuda_free=0, h_to_d=0;
    // Bench scaffolding: pseudo-random inputs standing in for real upstream data.
    double bench_fill=0;
    double overhead() const { return cuda_malloc + cuda_free + h_to_d; }
    double compute() const { return commit + fold + ood + open + induce + intro
                                  + lincheck + witness + zerocheck + l0commit + eq_build; }
    double open_phase() const { return commit + fold + ood + open + induce + intro; }
    double total() const { return compute() + overhead() + bench_fill; }
};

// Exactly one prove() is in flight at a time and it owns the current Phase. A
// file-scope pointer beats threading Phase& through cached_tt and
// commit_dev, none of which otherwise know anything about timing.
static Phase* g_ph = nullptr;
static Phase& cur_phase() {
    if (!g_ph) { printf("FATAL: timed CUDA call outside prove()\n"); exit(1); }
    return *g_ph;
}
// Stream-ordered allocation on the default stream. The driver's memory pool
// recycles freed blocks instead of unmapping them, so after the warm-up a prove
// reuses the previous prove's buffers and never pays for a real cudaMalloc —
// and cudaFreeAsync does not carry cudaFree's implicit device synchronize. This
// replaces a hand-rolled exact-size free list that did the same job worse.
//
// These deliberately do NOT sync first: an added sync would break the
// side-stream overlap under the l0 commit and bill pending kernel time to the
// allocator. Buffers handed to the s_pre side stream are safe because the phase
// timers device-synchronize between the allocation and that stream's work.
static void cuda_pool_setup() {
    cudaMemPool_t pool; CK(cudaDeviceGetDefaultMemPool(&pool, 0));
    uint64_t keep = UINT64_MAX;    // hold freed blocks for the next prove
    CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep));
}
static void* timed_malloc(size_t bytes) {
    auto t = Clock::now();
    void* p; cudaError_t e = cudaMallocAsync(&p, bytes, 0);
    if (e) {
        // The pool holds every retired block, so at m>=34 its reservation can
        // crowd out a new size class. Hand the idle blocks back to the driver and
        // retry once; the sync is needed for pending stream-ordered frees to land.
        cudaGetLastError();
        CK(cudaDeviceSynchronize());
        cudaMemPool_t pool; CK(cudaDeviceGetDefaultMemPool(&pool, 0));
        CK(cudaMemPoolTrimTo(pool, 0));
        e = cudaMallocAsync(&p, bytes, 0);
    }
    cur_phase().cuda_malloc += ms_wall(t);
    if (e) {
        size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
        printf("CUDA err %s: cudaMallocAsync %.2f GiB, device free %.2f/%.2f GiB\n",
               cudaGetErrorString(e), bytes / 1073741824.0, freeb / 1073741824.0, totb / 1073741824.0);
        exit(1);
    }
    return p;
}
static void timed_free(void* p) {
    auto t = Clock::now(); CK(cudaFreeAsync(p, 0)); cur_phase().cuda_free += ms_wall(t);
}
static void timed_h2d(void* dst, const void* src, size_t bytes) {
    auto t = Clock::now();
    CK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    cur_phase().h_to_d += ms_wall(t);
}
// Charges its scope's wall clock to `slot`, less any alloc/free/H2D billed to
// its own bucket inside the scope, so the buckets stay disjoint.
struct Charge {
    Phase& ph; double& slot; Clock::time_point t0; double ovh0;
    Charge(Phase& p, double& s) : ph(p), slot(s), t0(Clock::now()), ovh0(p.overhead()) {}
    ~Charge() { slot += ms_since(t0) - (ph.overhead() - ovh0); }
};

__global__ void fill_benchmark_polynomials(F128* A, F128* B, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
    u64 x = (u64)i * 0x9E3779B97F4A7C15ull + 1; A[i] = F128{x, x*0xBF58476D1CE4E5B9ull};
    B[i] = F128{x ^ 0x55, x*0x2545F4914F6CDD1Dull};
}
__global__ void replicate_fill(const F128* __restrict__ m, F128* __restrict__ cw, long long cw_len, long long ml) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= cw_len) return; cw[i] = m[i % ml];
}
static ChF128 to_ch(F128 x){ return ChF128{x.lo,x.hi}; }

// Twiddle tables are data-independent static data (the CPU ligero_commit takes
// a precomputed AdditiveNttF128) — build + upload once per k_code, reuse.
static std::map<int, TwiddleTable> g_tt;
static std::map<int, F128*> g_dtw;
static const TwiddleTable& cached_tt(int k_code, F128*& d_tw) {
    auto it = g_tt.find(k_code);
    if (it == g_tt.end()) {
        g_tt[k_code] = build_twiddle_table(k_code);
        size_t bytes = g_tt[k_code].data.size()*sizeof(F128);
        F128* dtw = (F128*)timed_malloc(bytes);
        timed_h2d(dtw, g_tt[k_code].data.data(), bytes);
        g_dtw[k_code] = dtw;
    }
    d_tw = g_dtw[k_code];
    return g_tt[k_code];
}

// device ligero_commit; returns root (host), leaves codeword+tree on device.
static void commit_dev(const F128* d_src, int msg_log, int log_msg_cols, int log_ni, int log_inv_rate,
                       F128*& d_cw, uint8_t*& d_tree, long long& block_len, int& num_ntts, uint8_t root[32],
                       bool detail = false) {
    int k_code = log_msg_cols + log_inv_rate; num_ntts = 1 << log_ni; block_len = 1LL << k_code;
    long long cw_len = block_len * num_ntts, msg_len = 1LL << msg_log;
    F128* d_tw; const TwiddleTable& tt = cached_tt(k_code, d_tw);
    d_cw = (F128*)timed_malloc(cw_len*sizeof(F128));
    d_tree = (uint8_t*)timed_malloc((size_t)(2*block_len-1)*32);
    cudaEvent_t e0, e1, e2; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
    cudaEventRecord(e0);
    // Rate-extend fusion: the pre-NTT codeword is cw[e]=msg[e & (msg_len-1)], so
    // when the first NTT pass is a shared-memory chunk it reads the message directly —
    // no replicate_fill pass, and pass-1 reads msg_len instead of cw_len elems.
    if (ntt_can_fuse_source(k_code - log_inv_rate)) {
        launch_ntt(d_cw,d_tw,tt,log_inv_rate,k_code,num_ntts,256,d_src,msg_len-1);
    } else {
        int tpb=256; replicate_fill<<<(unsigned)((cw_len+tpb-1)/tpb),tpb>>>(d_src,d_cw,cw_len,msg_len);
        launch_ntt(d_cw,d_tw,tt,log_inv_rate,k_code,num_ntts);
    }
    cudaEventRecord(e1);
    launch_merkle((const uint8_t*)d_cw,d_tree,block_len,num_ntts*16,256,4);   // kway=4 ILP
    cudaEventRecord(e2);
    CK(cudaDeviceSynchronize());
    static int l0_seen = 0;
    if (detail && ++l0_seen == 2) {   // first post-warmup l0 call
        float t_ntt, t_mk; cudaEventElapsedTime(&t_ntt, e0, e1); cudaEventElapsedTime(&t_mk, e1, e2);
        printf("  [l0-detail] ntt(fused rate-extend, 2^%d x %d lanes) %.3f | merkle(2^%d leaves x %d B) %.3f ms\n",
               k_code, num_ntts, t_ntt, k_code, num_ntts*16, t_mk);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaEventDestroy(e2));
    CK(cudaMemcpy(root, d_tree+(size_t)(2*block_len-2)*32, 32, cudaMemcpyDeviceToHost));
}
static void msg(const F128*A,const F128*B,long long len,F128*p0,F128*p2,F128*du0,F128*du2,F128&u0,F128&u2){
    launch_sumcheck_message(A,B,len/2,p0,p2,du0,du2);
    F128 u[2]; CK(cudaMemcpy(u,du0,2*sizeof(F128),cudaMemcpyDeviceToHost)); u0=u[0]; u2=u[1];   // du2 = du0+1
}

// Full zerocheck prove_packed orchestration, resident on the witness products
// a=A·z, b=B·z (c=z=df), threading the shared challenger `ch`. Times into
// ph.zerocheck and returns the x_ab quirky point (z_skip + mlv challenges) that
// lincheck consumes — closing the zerocheck→lincheck hand-off on-GPU. Tables M/
// f8mul are data-independent; arbitrary patterns here (this is a timing/dataflow
// bench — correctness is in test_zerocheck_full). m = log_n + 7, k_skip = 6.
static void zerocheck_phase(F128* da, F128* db, F128* dc, int m, FsChallenger& ch, Phase& ph,
                            F128& z_skip_out, std::vector<F128>& x_inner_rest,
                            std::vector<F128>& x_outer, int lc_k_log) {
    const int k_skip = 6;
    long long rows = 1LL << (m - 6);          // round-1 rows / a_mlv length

    static bool tables_done = false;
    if (!tables_done) {
        std::vector<uint8_t> mcol(64 * 64), f8mul((size_t)256 * 256);
        for (size_t i = 0; i < mcol.size(); i++) mcol[i] = (uint8_t)(i * 7 + 1);
        for (size_t i = 0; i < f8mul.size(); i++) f8mul[i] = (uint8_t)(i ^ (i >> 8));
        upload_zerocheck_first_round_tables(mcol.data(), f8mul.data(), PHI_8_TABLE);
        tables_done = true;
    }

    F128 *d_eq, *d_r1ab, *d_r1c, *d_ft, *d_am, *d_bm, *d_amn, *d_bmn, *d_p1, *d_pinf, *d_m1, *d_minf;
    d_eq = (F128*)timed_malloc(rows * sizeof(F128));
    d_r1ab = (F128*)timed_malloc(64 * sizeof(F128)); d_r1c = (F128*)timed_malloc(64 * sizeof(F128));
    d_ft = (F128*)timed_malloc(8 * 256 * sizeof(F128));
    d_am = (F128*)timed_malloc(rows * sizeof(F128)); d_bm = (F128*)timed_malloc(rows * sizeof(F128));
    d_amn = (F128*)timed_malloc(rows * sizeof(F128)); d_bmn = (F128*)timed_malloc(rows * sizeof(F128));
    d_p1 = (F128*)timed_malloc(ZT_MAX_BLOCKS * sizeof(F128)); d_pinf = (F128*)timed_malloc(ZT_MAX_BLOCKS * sizeof(F128));
    d_m1 = (F128*)timed_malloc(sizeof(F128)); d_minf = (F128*)timed_malloc(sizeof(F128));
    ZcSha* d_state = (ZcSha*)timed_malloc(sizeof(ZcSha));
    F128* d_rhos = (F128*)timed_malloc((m - 6) * sizeof(F128));
    F128* d_scales = (F128*)timed_malloc((m - 6) * sizeof(F128));
    F128* d_part8 = (F128*)timed_malloc(8 * ZT_MAX_BLOCKS * sizeof(F128));   // lookahead partials
    F128* d_out8 = (F128*)timed_malloc(8 * sizeof(F128));
    // split-eq tables (see zerocheck_tail.cuh): lo = (m-7)-7 vars, hi = 7 vars.
    const int zt_dfull = m - 7, zt_lobits = zt_dfull > 7 ? zt_dfull - 7 : 0;
    F128* d_eqlo = (F128*)timed_malloc((1LL << zt_lobits) * sizeof(F128));
    F128* d_eqhi = (F128*)timed_malloc((1LL << (zt_dfull - zt_lobits)) * sizeof(F128));
    const F128 ONE{1, 0};

    Charge zc_charge(ph, ph.zerocheck);   // runs to function scope end: covers the frees + hand-off
    ch.observe_label((const uint8_t*)"flock-zerocheck-v0", 18);
    // r = [r_skip(6) | small(3) | medium(4) | r_outer(m-13)].
    std::vector<ChF128> rs(6); ch.sample_f128_vec(rs.data(), 6);
    std::vector<ChF128> ro(m - 13); ch.sample_f128_vec(ro.data(), m - 13);
    std::vector<F128> r(m);
    for (int i = 0; i < 6; i++) r[i] = F128{rs[i].lo, rs[i].hi};
    int sm[3] = {0xF7, 0x53, 0xB5};
    for (int i = 0; i < 3; i++) r[6 + i] = PHI_8_TABLE[sm[i]];
    F128 gm[4] = {F128{2, 0}, F128{4, 0}, F128{16, 0}, F128{256, 0}};
    for (int i = 0; i < 4; i++) r[9 + i] = f128_mul_hd(gm[i], f128_inv_host(f128_add_hd(ONE, gm[i])));
    for (int i = 0; i < m - 13; i++) r[13 + i] = F128{ro[i].lo, ro[i].hi};

    double t_r1=0,t_r1eq=0,t_r2=0,t_msg1=0,t_tail=0,t_fin=0; auto _s=Clock::now();
    static int zc_call = 0; zc_call++; const bool zc_det = (zc_call == 2);
    cudaEvent_t ev0, ev1, evk; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1)); CK(cudaEventCreate(&evk));
    float det_r1k=0, det_r2k=0, det_eqb=0;
    double det_r2host=0, tr_wall[40]={0}; float tr_gpu[40]={0}; int tr_n=0, tr_round[40]={0};
    long long tr_op[40]={0};
    // round-1 univariate round message. CPU-structured only needs eq_out = eq(r[13..m]) (the stride-128 subsample),
    // Build eq(r[13..m]) and apply the fixed-round scale separately.
    { std::vector<F128> ro13(r.begin() + 13, r.end()); build_eq_device(d_eq, ro13.data(), m - 13); }
    F128 r1scale = ONE; for (int i = 6; i < 13; i++) r1scale = f128_mul_hd(r1scale, f128_add_hd(ONE, r[i]));
    t_r1eq=ms_since(_s); _s=Clock::now();
    cudaEventRecord(ev0);
    launch_zerocheck_first_round_cpu_structured((const uint8_t*)da, (const uint8_t*)db, (const uint8_t*)dc,
                               d_eq, 1LL << (m - 13), r1scale, d_r1ab, d_r1c);
    cudaEventRecord(ev1);
    std::vector<F128> r1ab(64), r1c(64);
    CK(cudaMemcpy(r1ab.data(), d_r1ab, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r1c.data(), d_r1c, 64 * sizeof(F128), cudaMemcpyDeviceToHost));
    { std::vector<ChF128> s(64); for (int i = 0; i < 64; i++) s[i] = ChF128{r1ab[i].lo, r1ab[i].hi};
      ch.observe_f128_slice(s.data(), 64);
      for (int i = 0; i < 64; i++) s[i] = ChF128{r1c[i].lo, r1c[i].hi}; ch.observe_f128_slice(s.data(), 64); }
    ChF128 zc = ch.sample_f128(); F128 z{zc.lo, zc.hi};
    t_r1=ms_since(_s); _s=Clock::now();
    cudaEventElapsedTime(&det_r1k, ev0, ev1);

    // round-2 fold-at-z + first message.
    auto _h2 = Clock::now();
    std::vector<F128> ws = lagrange_weights_host(6, z);
    std::vector<F128> ft(8 * 256, F128{0, 0});
    for (int j = 0; j < 8; j++) for (int v = 0; v < 256; v++) { F128 acc{0, 0};
        for (int bb = 0; bb < 8; bb++) if ((v >> bb) & 1) acc = f128_add_hd(acc, ws[8 * j + bb]); ft[j * 256 + v] = acc; }
    det_r2host = ms_wall(_h2);
    timed_h2d(d_ft, ft.data(), 8 * 256 * sizeof(F128));

    F128 *cA = d_am, *cB = d_bm, *nA = d_amn, *nB = d_bmn;
    long long len = rows;
    std::vector<F128> mlv_rhos;
    int n_mlv = m - 6;
    // SPLIT-EQ tail (host challenger): eqlo/eqhi built ONCE, each round's eq is an
    // index shift into them plus a scalar S_k = prod_{j=7}^{6+k}(1+r[j])^{-1} applied
    // to the two message sums in the combine kernel (see zerocheck_tail.cuh). No
    // per-round halve+scale pass, no full-size eq table ever built or streamed.
    // (A fully-resident on-device-challenger variant was tried — byte-exact but ~0.1 ms slower:
    // single-thread GPU SHA > CPU SHA; the tail was never round-trip-bound. See zc_challenger_device.cuh.)
    // Built BEFORE the fold: it depends only on r, and the fold now consumes it.
    cudaEventRecord(ev0);
    build_eq_device(d_eqlo, &r[7], zt_lobits);
    build_eq_device(d_eqhi, &r[7 + zt_lobits], zt_dfull - zt_lobits);
    cudaEventRecord(ev1);
    cudaEventElapsedTime(&det_eqb, ev0, ev1);

    // Fold-at-z plus the first TWO multilinear messages in ONE pass over the
    // packed witness: the fold already holds a whole output quad in registers, so
    // both message #0 and message #1's lookahead quadratic come for free. That
    // second message is what lets every tail pass below fold twice.

    int n_tail = n_mlv - 1;                     // m >= 13 here, so n_tail >= 6
    std::vector<F128> sc(n_tail);                                        // S_i = prod_{j=7}^{7+i}(1+r[j])^{-1}
    { std::vector<F128> v(n_tail), pre(n_tail); F128 acc = ONE;
      for (int i = 0; i < n_tail; i++) v[i] = f128_add_hd(ONE, r[7 + i]);
      for (int i = 0; i < n_tail; i++) { pre[i] = acc; acc = f128_mul_hd(acc, v[i]); }
      F128 inv = f128_inv_host(acc);
      for (int i = n_tail - 1; i >= 0; i--) { sc[i] = f128_mul_hd(pre[i], inv); inv = f128_mul_hd(inv, v[i]); }
      for (int i = 1; i < n_tail; i++) sc[i] = f128_mul_hd(sc[i - 1], sc[i]); }   // prefix products

    cudaEventRecord(ev0);
    launch_zerocheck_second_round_fold_with_lookahead((const uint8_t*)da, (const uint8_t*)db, d_ft,
                               d_eqlo, d_eqhi, zt_lobits, rows, d_am, d_bm,
                               sc[0], d_part8, d_out8);
    cudaEventRecord(ev1);
    t_r2=ms_since(_s); _s=Clock::now();
    cudaEventElapsedTime(&det_r2k, ev0, ev1);
    {   F128 h[8]; CK(cudaMemcpy(h, d_out8, 8 * sizeof(F128), cudaMemcpyDeviceToHost));
        ch.observe_f128(ChF128{h[0].lo, h[0].hi}); ch.observe_f128(ChF128{h[1].lo, h[1].hi});
        ChF128 r0 = ch.sample_f128(); F128 rho_0{r0.lo, r0.hi};
        mlv_rhos.push_back(rho_0);
        F128 g1 = zt_interp3(h[2], h[3], h[4], rho_0);
        F128 gi = zt_interp3(h[5], h[6], h[7], rho_0);
        ch.observe_f128(ChF128{g1.lo, g1.hi}); ch.observe_f128(ChF128{gi.lo, gi.hi});
        ChF128 r1c = ch.sample_f128(); mlv_rhos.push_back(F128{r1c.lo, r1c.hi});
    }
    t_msg1=ms_since(_s); _s=Clock::now();
    double fin_wall = 0; float fin_gpu = 0; int fin_rem = 0; long long fin_op0 = 0;
    int look_passes = 0; double look_wall = 0; long long look_op0 = 0;
    { long long L = len;
      int i = 0;
      // ---- two-round lookahead (see zerocheck_tail.cuh) ----
      // Invariant after a pass with index k: the array is folded through
      // rho_{k-1} (length rows/2^k), messages #0..#(k+1) are on the transcript,
      // and rho_k, rho_{k+1} are pending. The round-2 kernel above is the k=0
      // pass, so this always starts with two rhos in hand and always folds twice.
      int k = 2;
      while (k + 1 <= n_tail) {
          long long out_quads = L / 16;
          if (out_quads <= ZT_FINISH_OP) break;
          auto _lw = Clock::now();
          launch_zerocheck_tail_lookahead(cA, cB, nA, nB, d_eqlo, d_eqhi, k, zt_lobits, out_quads,
                              mlv_rhos[mlv_rhos.size() - 2], mlv_rhos.back(),
                              sc[k - 1], sc[k], d_part8, d_out8);
          { F128* t2; t2 = cA; cA = nA; nA = t2; t2 = cB; cB = nB; nB = t2; }
          L = L / 4; len = L;
          F128 h[8]; CK(cudaMemcpy(h, d_out8, 8 * sizeof(F128), cudaMemcpyDeviceToHost));
          ch.observe_f128(ChF128{h[0].lo, h[0].hi}); ch.observe_f128(ChF128{h[1].lo, h[1].hi});
          ChF128 rk = ch.sample_f128(); F128 rho_k{rk.lo, rk.hi};
          mlv_rhos.push_back(rho_k);
          F128 g1 = zt_interp3(h[2], h[3], h[4], rho_k);
          F128 gi = zt_interp3(h[5], h[6], h[7], rho_k);
          ch.observe_f128(ChF128{g1.lo, g1.hi}); ch.observe_f128(ChF128{gi.lo, gi.hi});
          ChF128 rk1 = ch.sample_f128(); mlv_rhos.push_back(F128{rk1.lo, rk1.hi});
          if (!look_passes) look_op0 = out_quads;
          look_passes++; look_wall += ms_wall(_lw);
          k += 2;
      }
      {   // Two rhos are pending but the single-round loop below takes one, so
          // spend rho_{k-2} on a plain fold; rho_{k-1} then drives its first round.
          long long half = L / 2;
          launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos[mlv_rhos.size() - 2]);
          { F128* t2; t2 = cA; cA = nA; nA = t2; t2 = cB; cB = nB; nB = t2; }
          L = half; len = half;
          i = k - 1;   // messages #0..#(k-1) are on the transcript; the loop makes #(i+1)
      }
      // ...then one round per pass while still bandwidth-bound...
      for (; i < n_tail && L / 4 > ZT_FINISH_OP; i++) {
          long long op = L / 4;
          auto _rw = Clock::now();
          cudaEventRecord(ev0);
          launch_zerocheck_tail_fold_and_message(cA, cB, nA, nB, d_eqlo, d_eqhi, i + 1, zt_lobits, op,
                                   mlv_rhos.back(), sc[i], d_p1, d_pinf, d_m1, d_minf);
          cudaEventRecord(evk);
          { F128* t2; t2 = cA; cA = nA; nA = t2; t2 = cB; cB = nB; nB = t2; } len = len / 2;
          F128 m1, mi; CK(cudaMemcpy(&m1, d_m1, sizeof(F128), cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(&mi, d_minf, sizeof(F128), cudaMemcpyDeviceToHost));
          ch.observe_f128(ChF128{m1.lo, m1.hi}); ch.observe_f128(ChF128{mi.lo, mi.hi});
          ChF128 rr = ch.sample_f128();
          tr_round[tr_n] = i;
          tr_wall[tr_n] = ms_wall(_rw);
          cudaEventElapsedTime(&tr_gpu[tr_n], ev0, evk); tr_op[tr_n] = op; tr_n++;
          mlv_rhos.push_back(F128{rr.lo, rr.hi});
          L /= 2; }
      // ...then ONE fused finisher launch for the latency-floor rounds: fold +
      // message + on-device challenger every remaining round, no host round-trips.
      if (i < n_tail) {
          int rem = n_tail - i;
          auto _rw = Clock::now();
          ZcSha zs = zc_pack(ch.hasher);
          timed_h2d(d_state, &zs, sizeof(ZcSha));
          timed_h2d(d_scales, sc.data() + i, rem * sizeof(F128));
          cudaEventRecord(ev0);
          finish_zerocheck_tail<<<1, ZT_FIN_TPB>>>(cA, cB, nA, nB, d_eqlo, d_eqhi, zt_lobits, i + 1,
                                              d_scales, mlv_rhos.back(), rem, L,
                                              d_state, d_rhos, nullptr, nullptr);
          cudaEventRecord(evk);
          CK(cudaMemcpy(&zs, d_state, sizeof(ZcSha), cudaMemcpyDeviceToHost));
          zc_unpack(ch.hasher, zs);
          std::vector<F128> rh(rem);
          CK(cudaMemcpy(rh.data(), d_rhos, rem * sizeof(F128), cudaMemcpyDeviceToHost));
          for (int t = 0; t < rem; t++) mlv_rhos.push_back(rh[t]);
          if (rem & 1) { F128* t2 = cA; cA = nA; nA = t2; t2 = cB; cB = nB; nB = t2; }
          fin_op0 = L / 4; len >>= rem; L >>= rem;
          fin_wall = ms_wall(_rw);
          cudaEventElapsedTime(&fin_gpu, ev0, evk); fin_rem = rem;
      } }
    t_tail=ms_since(_s); _s=Clock::now();
    { long long half = len / 2; launch_sumcheck_fold(cA, cB, nA, nB, half, mlv_rhos.back()); len = half; }  // final binding
    t_fin=ms_since(_s);
    static bool _pr=false; if(!_pr){_pr=true; printf("  [zc] r1-eqbuild %.2f  round1-kernel %.2f  round2(lag+fold) %.2f  msg1 %.2f  tail(%d) %.2f  final %.2f ms\n", t_r1eq,t_r1,t_r2,t_msg1,n_tail,t_tail,t_fin);}
    if (zc_det) {
        printf("  [zc-detail] (iter 1, post-warmup)\n");
        printf("  [zc-detail] round1: kernel %.3f | d2h+fiat-shamir %.3f ms  (reads a/b/c bit-packed: 3 x %lld MB)\n",
               det_r1k, t_r1 - det_r1k, (1LL << m) / 8 / (1 << 20));
        printf("  [zc-detail] round2: host lagrange/table %.3f | fold kernel %.3f | other %.3f ms\n",
               det_r2host, det_r2k, t_r2 - det_r2host - det_r2k);
        printf("  [zc-detail] msg1:   eqlo/eqhi build(%d+%d vars, once) %.3f | (msg fused into round-2 fold) | d2h+fs %.3f ms\n",
               zt_lobits, zt_dfull - zt_lobits, det_eqb, t_msg1);
        double wsum = 0, gsum = 0;
        printf("  [zc-detail] tail: %d two-round lookahead pass(es) (first out_quads=2^%d) wall %.3f ms\n",
               look_passes, look_passes ? (int)(63 - __builtin_clzll(look_op0)) : 0, look_wall);
        printf("  [zc-detail] tail per-round (single-round remainder):\n");
        for (int i = 0; i < tr_n; i++) {
            wsum += tr_wall[i]; gsum += tr_gpu[i];
            printf("    r%-2d op=2^%-2d wall %6.3f  gpu %6.3f  host/launch %6.3f ms\n",
                   tr_round[i], (int)(63 - __builtin_clzll(tr_op[i])), tr_wall[i], tr_gpu[i],
                   tr_wall[i] - tr_gpu[i]);
        }
        if (fin_rem) {
            wsum += fin_wall; gsum += fin_gpu;
            printf("    finisher: %d rounds (op 2^%d..2^0) in ONE launch: wall %6.3f  gpu %6.3f  state-ship %6.3f ms\n",
                   fin_rem, (int)(63 - __builtin_clzll(fin_op0)), fin_wall, fin_gpu, fin_wall - fin_gpu);
        }
        printf("  [zc-detail] tail totals: wall %.3f = gpu-kernels %.3f + host+latency %.3f ms\n", wsum, gsum, wsum - gsum);
    }
    CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1)); CK(cudaEventDestroy(evk));

    // x_ab = QuirkyPoint{ z_skip=z, x_inner_rest=mlv_rhos[..inner_rest_len], x_outer=mlv_rhos[inner_rest_len..] }.
    int inner_rest_len = lc_k_log - k_skip;
    z_skip_out = z;
    x_inner_rest.assign(mlv_rhos.begin(), mlv_rhos.begin() + inner_rest_len);
    x_outer.assign(mlv_rhos.begin() + inner_rest_len, mlv_rhos.end());

    timed_free(d_eq); timed_free(d_r1ab); timed_free(d_r1c); timed_free(d_ft);
    timed_free(d_am); timed_free(d_bm); timed_free(d_amn); timed_free(d_bmn);
    timed_free(d_p1); timed_free(d_pinf); timed_free(d_m1); timed_free(d_minf);
    timed_free(d_eqlo); timed_free(d_eqhi);
    timed_free(d_state); timed_free(d_rhos); timed_free(d_scales);
    timed_free(d_part8); timed_free(d_out8);
}

// Fill SoA BLAKE3 Compression inputs with deterministic pseudo-random values.
__global__ void fill_compressions(uint32_t* cv, uint32_t* m, b3u64* ctr, uint32_t* blen,
                                  uint32_t* flags, int n_blocks) {
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= n_blocks) return;
    b3u64 s = (b3u64)blk * 0x9E3779B97F4A7C15ull + 1;
#define NXT (s = s * 6364136223846793005ull + 1, (uint32_t)(s >> 33))
    for (int w = 0; w < 8; w++) cv[blk * 8 + w] = NXT;
    for (int i = 0; i < 16; i++) m[blk * 16 + i] = NXT;
    ctr[blk] = ((b3u64)NXT << 32) | NXT; blen[blk] = NXT; flags[blk] = NXT;
#undef NXT
}

// Resident BLAKE3 witness generation (S4 GPU target). Produces the real witness
// z (= df, the commit input) plus a = A·z, b = B·z (resident outputs for the
// future zerocheck) and the lincheck stripe-packed z (d_zlin), all on-device —
// the only H2D in the full prover would be the Compression inputs themselves.
// n_blocks_log = log_n - 7 (K_LOG=14), fully populated so no padding/memset.
static void witness_phase(F128* df, F128* da, F128* db, int log_n, Phase& ph) {
    int n_blocks_log = log_n - 7;
    long long n_total = 1LL << n_blocks_log;
    int n_blocks = (int)n_total;
    uint32_t* d_cv    = (uint32_t*)timed_malloc((size_t)n_blocks * 8 * 4);
    uint32_t* d_m     = (uint32_t*)timed_malloc((size_t)n_blocks * 16 * 4);
    uint32_t* d_blen  = (uint32_t*)timed_malloc((size_t)n_blocks * 4);
    uint32_t* d_flags = (uint32_t*)timed_malloc((size_t)n_blocks * 4);
    b3u64*    d_ctr   = (b3u64*)timed_malloc((size_t)n_blocks * 8);
    {   // pseudo-random Compression inputs: bench scaffolding, not prove work
        Charge fill(ph, ph.bench_fill);
        fill_compressions<<<(unsigned)((n_blocks + 127) / 128), 128>>>(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks);
    }
    {   Charge c(ph, ph.witness);
        launch_blake3_witness_blocks(d_cv, d_m, d_ctr, d_blen, d_flags, n_blocks, n_total,
                                     (b3u64*)df, (b3u64*)da, (b3u64*)db);
        // The lincheck stripe transpose is NOT here: it depends only on the final z
        // (df) and is consumed only by lincheck, so prove() runs it on the side
        // stream under the l0 commit (thin grid), off the critical path.
    }
    timed_free(d_cv); timed_free(d_m); timed_free(d_blen); timed_free(d_flags); timed_free(d_ctr);
}

// Lincheck phase (src/lincheck.rs) run resident on the committed witness. In the
// full prover lincheck sits between zerocheck and PCS-open, reducing zerocheck's
// (â, b̂) claims to one z-claim. GPU zerocheck isn't ported yet, so here we drive
// the three lincheck kernels (CSC fold → comb_vec, partial fold → z_vec, top-bit
// product-sumcheck) over the RESIDENT L0 witness to measure their cost in place.
//
// Now fed the REAL resident stripe-packed witness `d_zlin` (from the GPU
// witness-gen transpose) AND the REAL x_ab (z_skip + mlv challenges) from the
// resident GPU zerocheck — closing the zerocheck→lincheck hand-off on-GPU.
// The CSC base matrices are the REAL BLAKE3 R1CS (A_0, B_0) loaded from
// blake3_lincheck_matrices.bin (dump_blake3_lincheck_matrices); α is arbitrary
// (timing bench, fold_alpha_batched is α-linear so timing is α-independent).

// Real BLAKE3 R1CS lincheck CSC matrices (GF(2), implicit ones). They are fixed
// circuit data — independent of the witness and of every challenge — so they are
// read and uploaded ONCE at startup and stay device-resident for every prove.
// Re-uploading them per prove cost ~3.5 ms of H2D, most of it a_rows.
// File: dump_blake3_lincheck_matrices.bin (magic "BL3M").
struct B3LincheckMatrices {
    int n_cols = 0, useful_bits = 0;            // useful_bits = 15409 for BLAKE3 (rest is padding)
    uint32_t *d_a_col_ptr = nullptr, *d_a_rows = nullptr, *d_b_col_ptr = nullptr, *d_b_rows = nullptr;
};
static B3LincheckMatrices g_b3;
// Setup-time, so raw cudaMalloc/cudaMemcpy: this is not prove work and must not
// be billed to a Phase (no prove is in flight, so the timed wrappers would fatal).
static void upload_b3_lincheck_matrices() {
    const char* path = "blake3_lincheck_matrices.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) { printf("FATAL: cannot open %s (run: make blake3_lincheck_matrices)\n", path); exit(1); }
    std::vector<uint32_t> a_col_ptr, a_rows, b_col_ptr, b_rows;
    auto ru32 = [&](uint32_t& v){ f.read((char*)&v, 4); };
    auto rvec = [&](std::vector<uint32_t>& v, size_t n){ v.resize(n); f.read((char*)v.data(), n*4); };
    uint32_t magic; ru32(magic);
    if (magic != 0x424C334Du) { printf("FATAL: %s bad magic 0x%08X\n", path, magic); exit(1); }
    uint32_t ncols, ub, annz, bnnz;
    ru32(ncols); ru32(ub); g_b3.n_cols = (int)ncols; g_b3.useful_bits = (int)ub;
    ru32(annz); rvec(a_col_ptr, ncols + 1); rvec(a_rows, annz);
    ru32(bnnz); rvec(b_col_ptr, ncols + 1); rvec(b_rows, bnnz);
    if (!f) { printf("FATAL: %s truncated\n", path); exit(1); }
    auto up = [](const std::vector<uint32_t>& v) {
        uint32_t* d; size_t bytes = v.size() * sizeof(uint32_t);
        CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, v.data(), bytes, cudaMemcpyHostToDevice));
        return d;
    };
    g_b3.d_a_col_ptr = up(a_col_ptr); g_b3.d_a_rows = up(a_rows);
    g_b3.d_b_col_ptr = up(b_col_ptr); g_b3.d_b_rows = up(b_rows);
}

static void lincheck_phase(const uint8_t* d_zlin, int m, int k_log, int k_skip,
                           F128 z_skip, const std::vector<F128>& x_inner_rest,
                           const std::vector<F128>& x_outer, Phase& ph) {
    int n_log = m - k_log;
    if (n_log < 3) return;                       // too small for byte stripes
    int k = 1 << k_log;
    int inner_rest_len = k_log - k_skip;
    long long n_outer = 1LL << n_log, n_stripes = n_outer / 8;

    F128 alpha{0x9abc, 0xdef0};
    if (g_b3.n_cols != k) { printf("FATAL: matrix n_cols %d != k %d (k_log mismatch)\n", g_b3.n_cols, k); exit(1); }

    F128* d_eq_inner = (F128*)timed_malloc(k*sizeof(F128));
    F128* d_comb     = (F128*)timed_malloc(k*sizeof(F128));
    F128* d_zvec     = (F128*)timed_malloc(k*sizeof(F128));
    F128* d_nC       = (F128*)timed_malloc(k*sizeof(F128));
    F128* d_nZ       = (F128*)timed_malloc(k*sizeof(F128));
    F128* d_eq_outer = (F128*)timed_malloc(n_outer*sizeof(F128));
    F128* d_p1       = (F128*)timed_malloc(LC_MAX_BLOCKS*sizeof(F128));
    F128* d_pinf     = (F128*)timed_malloc(LC_MAX_BLOCKS*sizeof(F128));
    F128* d_e1       = (F128*)timed_malloc(sizeof(F128));
    F128* d_einf     = (F128*)timed_malloc(sizeof(F128));
    {   // Both eq tables built on device — bit-identical to the host builders,
        // which cost ~99 ms at m=32 plus the H2D of the result.
        Charge c(ph, ph.eq_build);
        build_quirky_eq_device(d_eq_inner, z_skip, x_inner_rest, k_skip);
        build_eq_device(d_eq_outer, x_outer.data(), n_log);
    }

    std::vector<F128> chal(inner_rest_len);
    for (int r = 0; r < inner_rest_len; r++) chal[r] = F128{(u64)(r*2654435761ull+1), (u64)(r*40503+7)};

    {   Charge c(ph, ph.lincheck);
        cudaEvent_t l0, l1, l2, l3;
        CK(cudaEventCreate(&l0)); CK(cudaEventCreate(&l1));
        CK(cudaEventCreate(&l2)); CK(cudaEventCreate(&l3));
        cudaEventRecord(l0);
        launch_linear_check_compressed_column_fold(d_eq_inner, g_b3.d_a_col_ptr, g_b3.d_a_rows,
                                 g_b3.d_b_col_ptr, g_b3.d_b_rows, alpha, k, d_comb);
        cudaEventRecord(l1);
        launch_linear_check_partial_fold(d_zlin, d_eq_outer, n_stripes, k, g_b3.useful_bits, d_zvec);
        cudaEventRecord(l2);
        F128 *cC=d_comb,*cZ=d_zvec,*nC=d_nC,*nZ=d_nZ; long long len=k;
        for (int r = 0; r < inner_rest_len; r++) {
            long long half = len/2;
            launch_linear_check_message(cC, cZ, half, d_p1, d_pinf, d_e1, d_einf);
            launch_linear_check_fold_pair(cC, cZ, nC, nZ, half, chal[r]);
            F128* z; z=cC;cC=nC;nC=z; z=cZ;cZ=nZ;nZ=z; len=half;
        }
        cudaEventRecord(l3);
        CK(cudaDeviceSynchronize());
        static bool _lp=false; if(!_lp){_lp=true;
            float a1,a2,a3; cudaEventElapsedTime(&a1,l0,l1); cudaEventElapsedTime(&a2,l1,l2);
            cudaEventElapsedTime(&a3,l2,l3);
            printf("  [lc-detail] csc_fold(k=%d) %.3f | partial_fold(zlin, %lld stripes) %.3f | %d sumcheck rounds %.3f ms\n",
                   k, a1, n_stripes, a2, inner_rest_len, a3); }
        CK(cudaEventDestroy(l0)); CK(cudaEventDestroy(l1));
        CK(cudaEventDestroy(l2)); CK(cudaEventDestroy(l3));
    }

    timed_free(d_eq_inner); timed_free(d_comb); timed_free(d_zvec); timed_free(d_nC); timed_free(d_nZ);
    timed_free(d_eq_outer);
    timed_free(d_p1); timed_free(d_pinf); timed_free(d_e1); timed_free(d_einf);
}

static double prove(int log_n,int initial_k,int log_inv_rate_0,int num_queries_0,int log_inv_rate_1,
                    int ood1,int r,int k_rec,int ood_rec,
                    const std::vector<int>& rec_rates,const std::vector<int>& rec_queries, Phase& ph) {
    long long len = 1LL << log_n; int n1 = log_n - initial_k; long long n1_len = 1LL << n1;
    int log_ni1 = k_rec;
    // Sumcheck state allocated up front; the witness is filled directly into
    // (df, dcb) — no separate d_f/d_b1 (saves 2 full-size buffers, matters at m≥34).
    g_ph = &ph;                       // the timed CUDA wrappers charge into this Phase
    auto t_prove = Clock::now();      // end-to-end wall clock; ph.total() must match it
    F128 *du2;
    F128* df   = (F128*)timed_malloc(len*sizeof(F128));
    F128* dcb  = (F128*)timed_malloc(len*sizeof(F128));
    F128* df2  = (F128*)timed_malloc(len*sizeof(F128));
    F128* dcb2 = (F128*)timed_malloc(len*sizeof(F128));
    F128* p0   = (F128*)timed_malloc(SMC_MAX_BLOCKS*sizeof(F128));
    F128* p2   = (F128*)timed_malloc(SMC_MAX_BLOCKS*sizeof(F128));
    F128* du0  = (F128*)timed_malloc(2*sizeof(F128)); du2 = du0 + 1;   // adjacent: one 32 B D2H per round
    F128* sc_part = (F128*)timed_malloc(8*SMC_MAX_BLOCKS*sizeof(F128));   // fold lookahead partials
    F128* sc_out  = (F128*)timed_malloc(8*sizeof(F128));
    {   // random (df, dcb): bench scaffolding for the sumcheck basis, not prove work
        Charge fill(ph, ph.bench_fill);
        int tpb=256; fill_benchmark_polynomials<<<(unsigned)((len+tpb-1)/tpb),tpb>>>(df,dcb,len);
    }

    // ---- GPU witness generation (S4): produce the REAL witness z into `df`
    // (overwriting the random fill — `dcb` keeps its random basis), plus a/b and
    // the lincheck stripe `d_zlin`, all resident. `df` then feeds commit + the
    // open with no H2D. Requires n_blocks_log = log_n-7 >= 3.
    F128 *d_a=nullptr,*d_b=nullptr; uint8_t* d_zlin=nullptr;
    bool do_witness = (log_n - 7) >= 3;
    if (do_witness) {
        d_a = (F128*)timed_malloc(len*sizeof(F128)); d_b = (F128*)timed_malloc(len*sizeof(F128));
        d_zlin = (uint8_t*)timed_malloc((size_t)len*16);   // 2^m/8 bytes = len*16
        witness_phase(df, d_a, d_b, log_n, ph);
        // a/b stay resident — consumed by zerocheck below, then freed before the open.
    }

    // L0 commit is the UPSTREAM commit phase (pcs::commit), NOT the open — the
    // open receives l0_codeword + l0_tree as borrowed inputs. Committed from the
    // witness (df, before any fold), before timing starts, excluded from the open.
    F128 *d_prev_cw; uint8_t *d_tree0; long long l0bl; int l0lanes; uint8_t l0root[32];
    // Precompute the open's FIRST sumcheck message on a non-blocking side stream,
    // overlapped with the l0 commit: the message depends only on (df, dcb) — both
    // final here — so Fiat-Shamir only constrains when it is OBSERVED (open start),
    // not when it is computed. The grid is capped WELL below machine fill: fat
    // co-runs serialize (full-grid msg owned every SM; zerocheck round-1's 128-reg
    // kernel owns the register file) — a thin grid-strided kernel trickles through
    // the commit's DRAM headroom instead. Results wait in du0/du2.
    static cudaStream_t s_pre = nullptr;
    if (!s_pre) CK(cudaStreamCreateWithFlags(&s_pre, cudaStreamNonBlocking));
    constexpr int pre_cap = 24;
    // Lincheck stripe transpose first (consumed earliest, by lincheck): reads only
    // the final z (df), so it trickles under the commit exactly like the message.
    // ---- Shared Fiat-Shamir challenger, threaded through the whole chain:
    //   observe commitment → zerocheck → lincheck → open. This is the residency
    //   assembly: the resident witness products a/b feed zerocheck, whose x_ab
    //   feeds lincheck, all on-GPU with one transcript; the open continues on it.
    FsChallenger ch(PROVER_LABEL+0, 0); // domain unimportant for timing
    uint8_t* d_prev_tree; long long prev_bl; int prev_ni;
    {   // The window spans the side-stream launches, the commit itself, and
        // observing the resulting root: all of it is the l0 commit step.
        Charge c(ph, ph.l0commit);
        if (do_witness)
            launch_blake3_lincheck_transpose((const b3u64*)df, 1LL << (log_n - 7), d_zlin, s_pre, pre_cap);
            { long long quads = len/4;
          int pblocks = sumcheck_blocks(quads) < pre_cap ? sumcheck_blocks(quads) : pre_cap;
          sumcheck_lookahead_message_partial<<<pblocks, SMC_TPB, 0, s_pre>>>(df, dcb, quads, sc_part);
          combine_sumcheck_lookahead_message<<<8, SMC_TPB, 0, s_pre>>>(sc_part, pblocks, sc_out); }
        commit_dev(df, log_n, log_n-initial_k, initial_k, log_inv_rate_0, d_prev_cw,d_tree0,l0bl,l0lanes,l0root, true);
        F128 target{0x1234,0x5678};
        ch.observe_label(PROVER_LABEL,sizeof(PROVER_LABEL)-1); ch.observe_f128(to_ch(target));
        ch.observe_bytes(l0root,32);
    }
    d_prev_tree = d_tree0; prev_bl=l0bl; prev_ni=l0lanes;
    F128* d_l0_cw=d_prev_cw; uint8_t* d_l0_tree=d_prev_tree;  // borrowed input — freed after timing

    if (do_witness) {
        // Zerocheck resident on a=A·z, b=B·z, c=z(=df) → x_ab quirky point.
        F128 z_skip; std::vector<F128> x_inner_rest, x_outer;
        zerocheck_phase(d_a, d_b, df, log_n + 7, ch, ph, z_skip, x_inner_rest, x_outer, B3_K_LOG);
        timed_free(d_a); d_a = nullptr; timed_free(d_b); d_b = nullptr;   // consumed by zerocheck
        // Lincheck on the resident stripe witness with the REAL x_ab.
        lincheck_phase(d_zlin, log_n + 7, B3_K_LOG, 6, z_skip, x_inner_rest, x_outer, ph);
        timed_free(d_zlin); d_zlin = nullptr;   // free before the open's codeword allocs
    }

    // The OPEN (commit + zerocheck + lincheck already done). Its cost is the sum
    // of the sub-phase buckets below; ph.open_phase() rolls them up.
    F128 *cf=df,*ccb=dcb,*nf=df2,*ncb=dcb2; long long slen=len;
    F128 u0,u2;
    // First message was precomputed on s_pre during the l0 commit (df/dcb unchanged
    // since); the phase syncs since then guarantee it is complete. Just fetch it.
    std::vector<F128> r_lane;
    {   Charge c(ph, ph.fold);
        // The chunk is a strict alternation: observe #0, sample, observe #1,
        // sample, ... observe #initial_k (no sample after the last). Each
        // lookahead pass folds by the two pending challenges and returns the next
        // two messages, so `obs` owns the sampling rule in one place.
        int j = 0;                                  // index of the message being observed
        auto obs = [&](F128 a, F128 b) {
            ch.observe_f128(to_ch(a)); ch.observe_f128(to_ch(b));
            if (j < initial_k) { ChF128 rc = ch.sample_f128(); r_lane.push_back(F128{rc.lo, rc.hi}); }
            j++;
        };
        F128 h[8]; CK(cudaMemcpy(h, sc_out, 8*sizeof(F128), cudaMemcpyDeviceToHost));
        obs(h[0], h[1]);                            // #0, precomputed under the l0 commit
        if (j <= initial_k) { F128 rr = r_lane.back();
            obs(zt_interp3(h[2],h[3],h[4],rr), zt_interp3(h[5],h[6],h[7],rr)); }
        int folds = 0;
        while (folds + 2 <= initial_k) {
            launch_sumcheck_lookahead(cf,ccb,nf,ncb, slen/16, r_lane[folds], r_lane[folds+1],
                                       sc_part, sc_out);
            {F128*z;z=cf;cf=nf;nf=z;z=ccb;ccb=ncb;ncb=z;} slen/=4; folds+=2;
            CK(cudaMemcpy(h, sc_out, 8*sizeof(F128), cudaMemcpyDeviceToHost));
            obs(h[0], h[1]);
            if (j <= initial_k) { F128 rr = r_lane.back();
                obs(zt_interp3(h[2],h[3],h[4],rr), zt_interp3(h[5],h[6],h[7],rr)); }
        }
        // Odd initial_k leaves one challenge unfolded; all messages are already on
        // the transcript, so this is a plain fold.
        for (; folds < initial_k; folds++) {
            long long half = slen/2;
            launch_sumcheck_fold(cf,ccb,nf,ncb,half,r_lane[folds]);
            {F128*z;z=cf;cf=nf;nf=z;z=ccb;ccb=ncb;ncb=z;} slen=half;
        }
    }

    // commit f1
    F128 *d_cw1; uint8_t *d_tree1; long long bl1; int lanes1; uint8_t root1[32];
    {   Charge c(ph, ph.commit);   // keep d_cw1 + d_tree1 on device
        commit_dev(cf,n1,n1-log_ni1,log_ni1,log_inv_rate_1,d_cw1,d_tree1,bl1,lanes1,root1); ch.observe_bytes(root1,32);
    }

    // OOD scratch
    F128* d_bnew = (F128*)timed_malloc(n1_len*sizeof(F128));
    F128* ep0    = (F128*)timed_malloc(IGL_MAX_BLOCKS*sizeof(F128));
    F128* ep2    = (F128*)timed_malloc(IGL_MAX_BLOCKS*sizeof(F128));
    F128* epodd  = (F128*)timed_malloc(IGL_MAX_BLOCKS*sizeof(F128));
    F128* eu0    = (F128*)timed_malloc(sizeof(F128));
    F128* eu2    = (F128*)timed_malloc(sizeof(F128));
    F128* ehnew  = (F128*)timed_malloc(sizeof(F128));

    auto ood_loop=[&](int cnt,int nn){ long long nl=1LL<<nn; for(int o=0;o<cnt;o++){
        std::vector<ChF128> z(nn); ch.sample_f128_vec(z.data(),nn); std::vector<F128> zf(nn);
        for(int j=0;j<nn;j++) zf[j]=F128{z[j].lo,z[j].hi};
        build_eq_device(d_bnew, zf.data(), nn);   // device eq table (hardware clmad)
        launch_basis_message_evaluation(cf,d_bnew,nl/2,ep0,ep2,epodd,eu0,eu2,ehnew);
        F128 y,iu0,iu2; CK(cudaMemcpy(&iu0,eu0,sizeof(F128),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&iu2,eu2,sizeof(F128),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&y,ehnew,sizeof(F128),cudaMemcpyDeviceToHost));
        ch.observe_f128(to_ch(y));ch.observe_f128(to_ch(iu0));ch.observe_f128(to_ch(iu2));
        ChF128 bc=ch.sample_f128(); launch_glue(ccb,d_bnew,F128{bc.lo,bc.hi},nl); } };

    auto query_open_induce=[&](int nn,int nq,const F128* d_pcw,const uint8_t* d_ptree,long long pbl,int pni,std::vector<F128>&lvl_rs){
        long long nl=1LL<<nn;
        (void)pni; (void)d_pcw; (void)lvl_rs;
        int al=0;{int m=nq-1;while(m){al++;m>>=1;}} if(nq<=1)al=0;
        std::vector<size_t> q; std::vector<F128> af(al);
        {   // Deriving the queries is part of the open, not free: sampling 218
            // distinct indices is host SHA256 and shows up at this scale.
            Charge c(ph, ph.open);
            ch.grind_pow(0);
            q=ch.sample_distinct_queries((size_t)pbl,nq);
            std::vector<ChF128> alpha(al); ch.sample_f128_vec(alpha.data(),al);
            for(int i=0;i<al;i++) af[i]=F128{alpha[i].lo,alpha[i].hi};
            merkle_multi_proof_device(d_ptree,(size_t)pbl,q);
        }
        // ---- transpose-NTT induce: scatter alpha_pows over the queried codeword
        // domain (pbl), Fᵀ-NTT, truncate to 2^nn = basis. (enforced_sum is not
        // transcript-affecting, so the prove bench omits it.) ----
        // Pooled grow-only induce scratch (d_c is pbl-sized = 128MB at m=35 L0):
        // reused across levels, no per-level malloc/free.
        static F128* d_ap=nullptr; static F128* d_c=nullptr; static unsigned long long* d_q=nullptr;
        static long long ap_cap=0, c_cap=0; static int q_cap=0;
        {   Charge c(ph, ph.induce);
            int log_block=0; { long long b=pbl; while(b>1){ b>>=1; log_block++; } }
            long long ap_len = 1LL<<al;
            if(ap_len>ap_cap){ if(d_ap)timed_free(d_ap); d_ap=(F128*)timed_malloc(ap_len*sizeof(F128)); ap_cap=ap_len; }
            if(pbl>c_cap){ if(d_c)timed_free(d_c); d_c=(F128*)timed_malloc(pbl*sizeof(F128)); c_cap=pbl; }
            if(nq>q_cap){ if(d_q)timed_free(d_q); d_q=(unsigned long long*)timed_malloc(nq*sizeof(unsigned long long)); q_cap=nq; }
            build_eq_device(d_ap, af.data(), al);
            std::vector<unsigned long long> qh(nq); for(int i=0;i<nq;i++) qh[i]=q[i];
            timed_h2d(d_q,qh.data(),nq*sizeof(unsigned long long));
            F128* d_tw; const TwiddleTable& tt=cached_tt(log_block,d_tw);
            int tpb2=256;
            clear_field_elements<<<(unsigned)((pbl+tpb2-1)/tpb2),tpb2>>>(d_c,pbl);
            scatter_query_weights<<<(unsigned)((nq+tpb2-1)/tpb2),tpb2>>>(d_c,d_q,d_ap,nq);
            launch_transpose_ntt(d_c,d_tw,tt,log_block);
        }
        F128* dbasis=d_c;   // first nl elements are the truncated basis
        {   Charge c(ph, ph.intro);
            msg(cf,dbasis,nl,p0,p2,du0,du2,u0,u2); ch.observe_f128(to_ch(u0));ch.observe_f128(to_ch(u2));
            ChF128 bi=ch.sample_f128(); launch_glue(ccb,dbasis,F128{bi.lo,bi.hi},nl);
        }
        // pooled scratch — not freed per level
    };

    // L0 OOD + query/open/induce/introduce (query wtns_0)
    {   Charge c(ph, ph.ood); ood_loop(ood1,n1); }
    query_open_induce(n1,num_queries_0,d_prev_cw,d_prev_tree,prev_bl,prev_ni,r_lane);
    // prev = wtns_1. wtns_0 (L0) is the BORROWED INPUT — a real open doesn't free
    // it (the caller owns it); freeing 8GB here would wrongly inflate the open. So
    // just adopt wtns_1; d_l0_cw/tree are released after the timer.
    d_prev_cw=d_cw1; d_prev_tree=d_tree1; prev_bl=bl1; prev_ni=lanes1;

    // recursive levels
    for(int lvl=0;lvl<r;lvl++){
        std::vector<F128> lvl_rs;
        {   Charge c(ph, ph.fold);
            for(int k=0;k<k_rec;k++){ ChF128 rc=ch.sample_f128(); F128 rr{rc.lo,rc.hi};
                long long half=slen/2; launch_sumcheck_fold_and_message(cf,ccb,nf,ncb,half,rr,p0,p2,du0,du2); // fused fold + next msg (1 pass)
                {F128*z;z=cf;cf=nf;nf=z;z=ccb;ccb=ncb;ncb=z;} slen=half;
                { F128 u[2]; CK(cudaMemcpy(u,du0,2*sizeof(F128),cudaMemcpyDeviceToHost)); u0=u[0]; u2=u[1]; }
                ch.observe_f128(to_ch(u0));ch.observe_f128(to_ch(u2)); lvl_rs.push_back(rr);}
        }
        if(lvl==r-1){   // final claim: ship the tail, observe it, then open
            Charge c(ph, ph.open);
            std::vector<F128> yr(slen); CK(cudaMemcpy(yr.data(),cf,(size_t)slen*sizeof(F128),cudaMemcpyDeviceToHost));
            for(long long i=0;i<slen;i++)ch.observe_f128(to_ch(yr[i]));
            ch.grind_pow(0);
            std::vector<size_t> q=ch.sample_distinct_queries((size_t)prev_bl,rec_queries[lvl]);
            merkle_multi_proof_device(d_prev_tree,(size_t)prev_bl,q);
        } else {
            int nn=0;{long long s=slen;while(s>1){s>>=1;nn++;}}
            F128*dcwn;uint8_t*dtn;long long bln;int ln;uint8_t rn[32];
            {   Charge c(ph, ph.commit);   // keep dcwn + dtn on device
                commit_dev(cf,nn,nn-k_rec,k_rec,rec_rates[lvl],dcwn,dtn,bln,ln,rn); ch.observe_bytes(rn,32); }
            {   Charge c(ph, ph.ood); ood_loop(ood_rec,nn); }
            query_open_induce(nn,rec_queries[lvl],d_prev_cw,d_prev_tree,prev_bl,prev_ni,lvl_rs);
            timed_free(d_prev_cw); timed_free(d_prev_tree); d_prev_cw=dcwn; d_prev_tree=dtn; prev_bl=bln; prev_ni=ln;
        }
    }
    timed_free(d_prev_cw); timed_free(d_prev_tree);
    timed_free(d_l0_cw); timed_free(d_l0_tree);   // borrowed input, owned by the caller
    // NOT d_cw1/d_tree1: ownership moved to d_prev_cw/d_prev_tree above, which
    // the recursion releases at lvl 0 (or the line above when r == 0).
    if (d_a) timed_free(d_a); if (d_b) timed_free(d_b); if (d_zlin) timed_free(d_zlin);
    timed_free(df);timed_free(dcb);timed_free(df2);timed_free(dcb2);
    timed_free(p0);timed_free(p2);timed_free(du0);timed_free(sc_part);timed_free(sc_out);
    timed_free(d_bnew);timed_free(ep0);timed_free(ep2);timed_free(epodd);
    timed_free(eu0);timed_free(eu2);timed_free(ehnew);
    double wall = ms_since(t_prove);
    g_ph = nullptr;
    return wall;
}

int main(int argc,char**argv){
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("Device: %s | %d SMs\n",p.name,p.multiProcessorCount);
    int log_n, ik, r0, nq0, r1, ood1, r, k, oodr, iters;
    std::vector<int> rec_rates, rec_queries;

    if (argc > 1 && std::string(argv[1]) == "fast29") {
        // configs/ligerito/m29_fast.toml — grinding excluded (separate concern).
        log_n=22; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=4; k=3; oodr=1;
        rec_rates  = {3,4,5,5};        // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43};   // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):7;
        printf("Ligerito open [m29_fast config, grinding OFF]: log_n=22 initial_k=6 r=4 k_rec=3 "
               "rates=1,2,3,4,5  queries=218,106,71,53,43  ood=0,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast32") {
        // configs/ligerito/m32_fast.toml — grinding excluded.
        log_n=25; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=5; k=3; oodr=1;
        rec_rates  = {3,4,5,6,6};               // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43,36};         // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):5;
        printf("Ligerito open [m32_fast config, grinding OFF]: log_n=25 initial_k=6 r=5 k_rec=3 "
               "rates=1..6  queries=218,106,71,53,43,36  ood=0,1,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast33") {
        // configs/ligerito/m33_fast.toml — grinding excluded.
        log_n=26; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=5; k=3; oodr=1;
        rec_rates  = {3,4,5,6,6};               // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43,36};         // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):5;
        printf("Ligerito open [m33_fast config, grinding OFF]: log_n=26 initial_k=6 r=5 k_rec=3 "
               "rates=1..6  queries=218,106,71,53,43,36  ood=0,1,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast34") {
        // configs/ligerito/m34_fast.toml — grinding excluded.
        log_n=27; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=6; k=3; oodr=1;
        rec_rates  = {3,4,5,6,7,7};              // log_inv_rates[lvl+2] (last unused)
        rec_queries= {106,71,53,43,36,32};       // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):3;
        printf("Ligerito open [m34_fast config, grinding OFF]: log_n=27 initial_k=6 r=6 k_rec=3 "
               "rates=1..7  queries=218,106,71,53,43,36,32  ood=0,1,1,1,1,1,1\n");
    } else if (argc > 1 && std::string(argv[1]) == "fast35") {
        // configs/ligerito/m35_fast.toml — grinding excluded.
        log_n=28; ik=6; r0=1; nq0=218; r1=2; ood1=1; r=6; k=3; oodr=1;
        rec_rates  = {3,4,5,6,7,7};               // log_inv_rates[lvl+2]
        rec_queries= {106,71,53,43,36,32};        // queries[lvl+1]
        iters = argc>2?atoi(argv[2]):3;
        printf("Ligerito open [m35_fast config, grinding OFF]: log_n=28 initial_k=6 r=6 k_rec=3 "
               "rates=1..7  queries=218,106,71,53,43,36,32  ood=0,1,1,1,1,1,1\n");
    } else {
        auto A=[&](int i,int d){ return argc>i?atoi(argv[i]):d; };
        log_n=A(1,22);ik=A(2,5);r0=A(3,1);nq0=A(4,148);r1=A(5,1);ood1=A(6,1);r=A(7,2);k=A(8,3);
        int rr=A(9,1);oodr=A(10,1);int nqr=A(11,148);iters=A(12,5);
        rec_rates.assign(r, rr); rec_queries.assign(r, nqr);
        printf("Ligerito open: log_n=%d initial_k=%d r=%d k_rec=%d rate0=1/%d rate_rec=1/%d nq=%d/%d ood=%d/%d\n",
               log_n,ik,r,k,1<<r0,1<<rr,nq0,nqr,ood1,oodr);
    }

    // Setup: witness-independent data goes to the device once, before any prove.
    cuda_pool_setup();
    upload_b3_lincheck_matrices();

    Phase warm; prove(log_n,ik,r0,nq0,r1,ood1,r,k,oodr,rec_rates,rec_queries,warm); // warm-up
    Phase ph; double best=1e30;   // fastest end-to-end prove, with its own breakdown
    for(int it=0;it<iters;it++){ Phase p2;
        double wall=prove(log_n,ik,r0,nq0,r1,ood1,r,k,oodr,rec_rates,rec_queries,p2);
        if(wall<best){best=wall;ph=p2;} }
    printf("  open %.2f ms | commit %.2f  fold %.2f  ood %.2f  multiproof %.2f  induce %.2f  introduce/glue %.2f\n"
           "  resident chain: witness-gen %.2f  l0-commit %.2f  zerocheck %.2f  lincheck %.2f  eq-build %.2f ms\n"
           "  device overhead: cudaMalloc %.2f  cudaFree %.2f  H2D %.2f ms | bench-only input fill %.2f ms\n",
           ph.open_phase(),ph.commit,ph.fold,ph.ood,ph.open,ph.induce,ph.intro,
           ph.witness,ph.l0commit,ph.zerocheck,ph.lincheck,ph.eq_build,
           ph.cuda_malloc,ph.cuda_free,ph.h_to_d,ph.bench_fill);
    // The phase buckets partition prove()'s wall clock by construction, so a gap
    // here is work no phase is measuring, and optimizing off the breakdown above
    // would be misleading. Measured floor: ~0.2 ms of host glue spread over the
    // ~40 phase boundaries (5-20 us each), flat in the problem size — so judge
    // the ABSOLUTE gap, not just the percentage, on short proves.
    double sum = ph.total(), gap = best - sum, gap_pct = 100.0 * gap / best;
    printf("  >>> prove wall %.2f ms (%.2f excl. bench fill) | phase total %.2f ms | unattributed %.2f ms (%+.2f%%)\n",
           best, best - ph.bench_fill, sum, gap, gap_pct);
    if (fabs(gap_pct) > 1.0)
        printf("  !! unattributed %.2f ms is %+.2f%% of the prove wall (>1%%). Above ~0.2 ms that is a\n"
               "     region no phase timer covers; at or below it, it is the per-boundary measuring cost.\n",
               gap, gap_pct);
    return 0;
}
