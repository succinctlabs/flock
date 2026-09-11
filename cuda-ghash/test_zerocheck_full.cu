// Full zerocheck prove_packed orchestration on GPU, replayed byte-for-byte
// against the flock transcript (dump_zerocheck_full_vectors.rs, ZCFV). Drives
// the host FsChallenger through: round-1 univariate round message → c-interp → round-2 fold+msg →
// sumcheck tail → final binding, wiring the validated kernels together.
//
// Build:  make test_zerocheck_full
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "zerocheck_round1.cuh"
#include "zerocheck_round1_cpustyle.cuh"
#include "ntt_host.hpp"
#include "zerocheck_round2.cuh"
#include "zerocheck_tail.cuh"      // launch_zerocheck_tail_message + sumcheck_ab fold
#include "phi8_table.cuh"
#include "challenger.hpp"
#include "zc_challenger_device.cuh"   // resident on-device challenger for the tail
static ZcSha zc_pack(const Sha256& s){ ZcSha z; for(int i=0;i<8;i++)z.h[i]=s.h[i]; z.total_len=s.total_len;
    for(int i=0;i<64;i++)z.buf[i]=s.buf[i]; z.buf_len=(unsigned)s.buf_len; return z; }
static void zc_unpack(Sha256& s, const ZcSha& z){ for(int i=0;i<8;i++)s.h[i]=z.h[i]; s.total_len=z.total_len;
    for(int i=0;i<64;i++)s.buf[i]=z.buf[i]; s.buf_len=z.buf_len; }

#define CK(x) do { cudaError_t e=(x); if(e){ printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

static uint32_t rd_u32(FILE* f){ uint32_t v=0; if(fread(&v,4,1,f)!=1){printf("short u32\n");exit(1);} return v; }
static F128 rd_f128(FILE* f){ u64 v[2]; if(fread(v,8,2,f)!=2){printf("short f128\n");exit(1);} return F128{v[0],v[1]}; }
static bool eqf(F128 a, F128 b){ return a.lo==b.lo && a.hi==b.hi; }
static ChF128 toch(F128 x){ return ChF128{x.lo,x.hi}; }
static F128 frch(ChF128 x){ return F128{x.lo,x.hi}; }
static F128 ADD(F128 a,F128 b){ return f128_add_hd(a,b); }
static F128 MUL(F128 a,F128 b){ return f128_mul_hd(a,b); }
static const F128 ONE{1,0};

static std::vector<F128> build_eq(const std::vector<F128>& r){
    std::vector<F128> t; t.reserve((size_t)1<<r.size()); t.push_back(ONE);
    for(size_t j=0;j<r.size();j++){ F128 rj=r[j], omr=ADD(ONE,rj); size_t len=(size_t)1<<j; t.resize(2*len);
        for(size_t x=0;x<len;x++){ F128 v=t[x]; t[x+len]=MUL(v,rj); t[x]=MUL(v,omr);} }
    return t;
}
// Lagrange weights at z over the first ell = 2^k nodes (offset `off` into PHI):
//   off=0 → S domain (foldtable); off=64 → Λ domain (c-interp).
static std::vector<F128> lagrange(int k, F128 z, int off){
    int ell = 1<<k; std::vector<F128> w(ell);
    for(int i=0;i<ell;i++){ F128 si=PHI_8_TABLE[off+i], num=ONE, den=ONE;
        for(int j=0;j<ell;j++){ if(j==i) continue; F128 sj=PHI_8_TABLE[off+j];
            num=MUL(num,ADD(z,sj)); den=MUL(den,ADD(si,sj)); }
        w[i]=MUL(num,f128_inv_host(den)); }
    return w;
}
static int fail(const char* what){ printf("%s FAIL\n", what); return 1; }

int main(int argc, char** argv){
    const char* path = argc>1?argv[1]:"zerocheck_full_vectors.bin";
    FILE* f = fopen(path,"rb"); if(!f){ printf("cannot open %s\n", path); return 1; }
    if(rd_u32(f)!=0x5A434656u){ printf("bad magic (want ZCFV)\n"); return 1; }
    int m=(int)rd_u32(f), k_skip=6;
    uint32_t dlen=rd_u32(f); std::vector<uint8_t> domain(dlen);
    if(fread(domain.data(),1,dlen,f)!=dlen){printf("short domain\n");return 1;}
    std::vector<uint8_t> mcol(64*64), f8mul((size_t)256*256);
    if(fread(mcol.data(),1,mcol.size(),f)!=mcol.size()){return fail("M");}
    if(fread(f8mul.data(),1,f8mul.size(),f)!=f8mul.size()){return fail("f8mul");}
    size_t pb=(size_t)1<<(m-3);
    std::vector<uint8_t> a(pb),b(pb),c(pb);
    if(fread(a.data(),1,pb,f)!=pb||fread(b.data(),1,pb,f)!=pb||fread(c.data(),1,pb,f)!=pb){return fail("abc");}
    std::vector<F128> g_r1ab(64),g_r1c(64);
    for(auto&v:g_r1ab) v=rd_f128(f);
    for(auto&v:g_r1c) v=rd_f128(f);
    int n_mlv=(int)rd_u32(f);
    std::vector<F128> g_m1(n_mlv), g_mi(n_mlv);
    for(int i=0;i<n_mlv;i++){ g_m1[i]=rd_f128(f); g_mi[i]=rd_f128(f); }
    F128 g_fa=rd_f128(f), g_fb=rd_f128(f), g_fc=rd_f128(f);
    fclose(f);
    printf("ZCFV: m=%d n_mlv=%d\n", m, n_mlv);

    long long n_out = 1LL<<(m-6);      // a_mlv length after round-2
    upload_zerocheck_first_round_tables(mcol.data(), f8mul.data(), PHI_8_TABLE);

    // device buffers
    uint8_t *d_a,*d_b,*d_c; F128 *d_eq,*d_r1ab,*d_r1c,*d_ft,*d_am,*d_bm,*d_amn,*d_bmn,*d_p1,*d_pinf,*d_m1d,*d_mid;
    CK(cudaMalloc(&d_a,pb)); CK(cudaMalloc(&d_b,pb)); CK(cudaMalloc(&d_c,pb));
    CK(cudaMalloc(&d_eq,(1LL<<(m-13))*sizeof(F128)));
    CK(cudaMalloc(&d_r1ab,64*sizeof(F128))); CK(cudaMalloc(&d_r1c,64*sizeof(F128)));
    CK(cudaMalloc(&d_ft,8*256*sizeof(F128)));
    CK(cudaMalloc(&d_am,n_out*sizeof(F128))); CK(cudaMalloc(&d_bm,n_out*sizeof(F128)));
    CK(cudaMalloc(&d_amn,n_out*sizeof(F128))); CK(cudaMalloc(&d_bmn,n_out*sizeof(F128)));
    CK(cudaMalloc(&d_p1,ZT_MAX_BLOCKS*sizeof(F128))); CK(cudaMalloc(&d_pinf,ZT_MAX_BLOCKS*sizeof(F128)));
    CK(cudaMalloc(&d_m1d,sizeof(F128))); CK(cudaMalloc(&d_mid,sizeof(F128)));
    ZcSha* d_state; F128 *d_rho,*d_rhos,*d_m1log,*d_milog,*d_eqlo,*d_eqhi;
    int n_mlv_alloc = m - 6;
    CK(cudaMalloc(&d_state,sizeof(ZcSha))); CK(cudaMalloc(&d_rho,sizeof(F128)));
    CK(cudaMalloc(&d_rhos,n_mlv_alloc*sizeof(F128)));
    CK(cudaMalloc(&d_m1log,n_mlv_alloc*sizeof(F128))); CK(cudaMalloc(&d_milog,n_mlv_alloc*sizeof(F128)));
    // split-eq tables (see zerocheck_tail.cuh): lo = (m-7)-7 vars, hi = 7 vars.
    const int zt_dfull = m - 7, zt_lobits = zt_dfull > 7 ? zt_dfull - 7 : 0;
    CK(cudaMalloc(&d_eqlo,(1LL<<zt_lobits)*sizeof(F128)));
    CK(cudaMalloc(&d_eqhi,(1LL<<(zt_dfull-zt_lobits))*sizeof(F128)));
    CK(cudaMemcpy(d_a,a.data(),pb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b,b.data(),pb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_c,c.data(),pb,cudaMemcpyHostToDevice));

    FsChallenger ch(domain.data(), dlen);
    ch.observe_label((const uint8_t*)"flock-zerocheck-v0", 18);

    // ---- 1. r = [r_skip(6) | small(3) | medium(4) | r_outer(m-13)] ----
    std::vector<ChF128> rs(6); ch.sample_f128_vec(rs.data(), 6);
    std::vector<ChF128> ro(m-13); ch.sample_f128_vec(ro.data(), m-13);
    std::vector<F128> r(m);
    for(int i=0;i<6;i++) r[i]=frch(rs[i]);
    int sm[3]={0xF7,0x53,0xB5};
    for(int i=0;i<3;i++) r[6+i]=PHI_8_TABLE[sm[i]];
    F128 gm[4]={F128{2,0},F128{4,0},F128{16,0},F128{256,0}};
    for(int i=0;i<4;i++) r[9+i]=MUL(gm[i], f128_inv_host(ADD(ONE,gm[i])));
    for(int i=0;i<m-13;i++) r[13+i]=frch(ro[i]);

    // ---- 2. round-1 univariate round message ----
    std::vector<F128> r_outer(r.begin()+13, r.end());
    std::vector<F128> eq_outer=build_eq(r_outer);
    CK(cudaMemcpy(d_eq,eq_outer.data(),eq_outer.size()*sizeof(F128),cudaMemcpyHostToDevice));
    F128 round1_scale=ONE;
    for(int i=6;i<13;i++) round1_scale=MUL(round1_scale,ADD(ONE,r[i]));
    launch_zerocheck_first_round_cpu_structured(d_a,d_b,d_c,d_eq,eq_outer.size(),round1_scale,d_r1ab,d_r1c);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<F128> r1ab(64), r1c(64);
    CK(cudaMemcpy(r1ab.data(),d_r1ab,64*sizeof(F128),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r1c.data(),d_r1c,64*sizeof(F128),cudaMemcpyDeviceToHost));
    for(int i=0;i<64;i++){ if(!eqf(r1ab[i],g_r1ab[i])) return fail("round1_ab"); if(!eqf(r1c[i],g_r1c[i])) return fail("round1_c"); }
    { std::vector<ChF128> s(64); for(int i=0;i<64;i++) s[i]=toch(r1ab[i]); ch.observe_f128_slice(s.data(),64);
      for(int i=0;i<64;i++) s[i]=toch(r1c[i]); ch.observe_f128_slice(s.data(),64); }
    F128 z=frch(ch.sample_f128());
    printf("  round-1 OK\n");

    // ---- 3. c-interp at z over Λ ----
    std::vector<F128> wl=lagrange(6, z, 64);
    F128 final_c{0,0}; for(int i=0;i<64;i++) final_c=ADD(final_c, MUL(wl[i], r1c[i]));
    if(!eqf(final_c,g_fc)) return fail("final_c");
    printf("  c-interp OK\n");

    // ---- 4. round-2: fold-at-z + first message ----
    std::vector<F128> ws=lagrange(6, z, 0);     // S-domain weights
    std::vector<F128> ft(8*256, F128{0,0});
    for(int j=0;j<8;j++){ for(int v=0;v<256;v++){ F128 acc{0,0};
        for(int bb=0;bb<8;bb++) if((v>>bb)&1) acc=ADD(acc, ws[8*j+bb]); ft[j*256+v]=acc; } }
    CK(cudaMemcpy(d_ft, ft.data(), 8*256*sizeof(F128), cudaMemcpyHostToDevice));
    launch_zerocheck_second_round_fold(d_a,d_b,d_ft,n_out,d_am,d_bm);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    F128 *cA=d_am,*cB=d_bm,*nA=d_amn,*nB=d_bmn;
    long long len=n_out;
    // SPLIT-EQ (see zerocheck_tail.cuh): eqlo/eqhi built once on host, uploaded; each
    // round's eq is an index shift + a scalar S_k = prod_{j=7}^{6+k}(1+r[j])^{-1}.
    { std::vector<F128> eqlo=build_eq(std::vector<F128>(r.begin()+7, r.begin()+7+zt_lobits));
      std::vector<F128> eqhi=build_eq(std::vector<F128>(r.begin()+7+zt_lobits, r.end()));
      CK(cudaMemcpy(d_eqlo, eqlo.data(), eqlo.size()*sizeof(F128), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_eqhi, eqhi.data(), eqhi.size()*sizeof(F128), cudaMemcpyHostToDevice)); }
    // round-2 message: shift 0, scale ONE (eq = eq(r[7..m]) exactly).
    F128 m1,mi;
    { long long half=len/2;
      launch_zerocheck_tail_message(cA,cB,d_eqlo,d_eqhi,0,zt_lobits,half,ONE,d_p1,d_pinf,d_m1d,d_mid);
      CK(cudaMemcpy(&m1,d_m1d,sizeof(F128),cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&mi,d_mid,sizeof(F128),cudaMemcpyDeviceToHost)); }
    if(!eqf(m1,g_m1[0])||!eqf(mi,g_mi[0])) return fail("round2 msg");
    ch.observe_f128(toch(m1)); ch.observe_f128(toch(mi));
    F128 rho=frch(ch.sample_f128());
    printf("  round-2 OK\n");

    // ---- 5. tail rounds 3..(n_mlv+1) — RESIDENT: fused fold+msg + on-device challenger,
    //         all 22 rounds issued on one stream with NO host round-trip per round. ----
    int n_tail = n_mlv - 1;
    // Per-round scales S_i = prod_{j=7}^{7+i}(1+r[j])^{-1} (prefix products).
    std::vector<F128> S(n_tail);
    { F128 acc=ONE;
      for(int i=0;i<n_tail;i++){ acc=MUL(acc, f128_inv_host(ADD(ONE, r[7+i]))); S[i]=acc; } }
    { ZcSha zs=zc_pack(ch.hasher); CK(cudaMemcpy(d_state,&zs,sizeof(ZcSha),cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(d_rho,&rho,sizeof(F128),cudaMemcpyHostToDevice));
    // HYBRID tail: per-round fused fold+msg while op > ZT_FINISH_OP, then ONE
    // finish_zerocheck_tail launch runs all remaining small rounds internally.
    F128* d_S; CK(cudaMalloc(&d_S, n_tail*sizeof(F128)));
    CK(cudaMemcpy(d_S, S.data(), n_tail*sizeof(F128), cudaMemcpyHostToDevice));
    { long long L=len; int i=0;
      for(; i<n_tail && L/4 > ZT_FINISH_OP; i++){ long long op=L/4;
        launch_zerocheck_tail_fold_and_message_device_challenge(cA,cB,nA,nB,d_eqlo,d_eqhi,i+1,zt_lobits,op,d_rho,S[i],d_p1,d_pinf,d_m1d,d_mid);
        advance_zerocheck_tail_challenger<<<1,1>>>(d_state,d_m1d,d_mid,d_rho,d_rhos+i,d_m1log+i,d_milog+i);
        { F128* t; t=cA;cA=nA;nA=t; t=cB;cB=nB;nB=t; } L/=2; }
      if (i < n_tail) {
        F128 rho_h; CK(cudaMemcpy(&rho_h, d_rho, sizeof(F128), cudaMemcpyDeviceToHost));
        int rem = n_tail - i;
        finish_zerocheck_tail<<<1, ZT_FIN_TPB>>>(cA,cB,nA,nB,d_eqlo,d_eqhi,zt_lobits,i+1,d_S+i,
                                            rho_h, rem, L, d_state, d_rhos+i, d_m1log+i, d_milog+i);
        if (rem & 1) { F128* t; t=cA;cA=nA;nA=t; t=cB;cB=nB;nB=t; }
        L >>= rem;
        CK(cudaMemcpy(d_rho, d_rhos+(n_tail-1), sizeof(F128), cudaMemcpyDeviceToDevice));
      }
      len=L; }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    // validate the device-computed messages against the golden, restore host challenger.
    std::vector<F128> m1log(n_tail), milog(n_tail);
    CK(cudaMemcpy(m1log.data(),d_m1log,n_tail*sizeof(F128),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(milog.data(),d_milog,n_tail*sizeof(F128),cudaMemcpyDeviceToHost));
    for(int i=0;i<n_tail;i++) if(!eqf(m1log[i],g_m1[i+1])||!eqf(milog[i],g_mi[i+1])){ printf("tail round %d ",i); return fail("msg"); }
    { ZcSha zs; CK(cudaMemcpy(&zs,d_state,sizeof(ZcSha),cudaMemcpyDeviceToHost)); zc_unpack(ch.hasher,zs); }
    CK(cudaMemcpy(&rho,d_rho,sizeof(F128),cudaMemcpyDeviceToHost));   // last rho for final fold
    // ---- 6. final binding ----
    { long long half=len/2; launch_sumcheck_fold(cA,cB,nA,nB,half,rho);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      F128* t; t=cA;cA=nA;nA=t; t=cB;cB=nB;nB=t; len=half; }
    F128 fa,fb; CK(cudaMemcpy(&fa,cA,sizeof(F128),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&fb,cB,sizeof(F128),cudaMemcpyDeviceToHost));
    if(!eqf(fa,g_fa)||!eqf(fb,g_fb)) return fail("final a/b");
    ch.observe_f128(toch(fa)); ch.observe_f128(toch(fb));
    printf("  tail + final OK\n");

    printf("ZEROCHECK FULL OK: round1 + c-interp + round2 + %d tail rounds + final binding match flock bit-for-bit\n", n_mlv-1);
    return 0;
}
