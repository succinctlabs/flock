// Zerocheck round-2 fold-at-z on GPU — port of the fold half of
// src/zerocheck/multilinear.rs::uni_skip_fold_and_round_pair_optimized_packed.
// Folds the packed witness a/b at the univariate round message challenge z (over the skip domain)
// into a_mlv/b_mlv (F128, length 2^(m-6)):
//   a_mlv[row] = Σ_{j=0..8} foldtable[j*256 + a_packed[row*8 + j]]   (UniSkipFoldTable)
// The first multilinear message is then the eq-weighted deg-2 message over
// (a_mlv, b_mlv) — reuse zerocheck_tail.cuh::launch_zerocheck_tail_message.
#pragma once
#include "f128.cuh"
#include "zerocheck_tail.cuh"   // evaluate_zerocheck_split_equality / combine_zerocheck_tail_message (fused fold+message)

#ifndef ZR2_TPB
#define ZR2_TPB 256
#endif

// One thread per output row: 8 byte-lookups into the 8×256 F128 fold table.
__global__ void zerocheck_second_round_fold(const uint8_t* __restrict__ a_packed,
                               const uint8_t* __restrict__ b_packed,
                               const F128* __restrict__ foldtable, long long n_out,
                               F128* __restrict__ a_mlv, F128* __restrict__ b_mlv) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_out) return;
    const uint8_t* ar = a_packed + row * 8;
    const uint8_t* br = b_packed + row * 8;
    F128 av{0, 0}, bv{0, 0};
#pragma unroll
    for (int j = 0; j < 8; j++) {
        av = f128_add(av, foldtable[j * 256 + ar[j]]);
        bv = f128_add(bv, foldtable[j * 256 + br[j]]);
    }
    a_mlv[row] = av;
    b_mlv[row] = bv;
}

inline void launch_zerocheck_second_round_fold(const uint8_t* d_a, const uint8_t* d_b, const F128* d_foldtable,
                                  long long n_out, F128* d_a_mlv, F128* d_b_mlv) {
    long long blocks = (n_out + ZR2_TPB - 1) / ZR2_TPB;
    zerocheck_second_round_fold<<<(unsigned)blocks, ZR2_TPB>>>(d_a, d_b, d_foldtable, n_out, d_a_mlv, d_b_mlv);
}

// FUSED fold-at-z + the first TWO multilinear messages, in one pass over the
// packed witness. Each thread owns a QUAD of output rows (4z..4z+3), which is
// exactly the window the two-round lookahead needs: message #0 over the quad's
// two pairs, and message #1 as a quadratic in rho_0 (see zt_accum_quad).
//
// Emitting #1's quadratic here is what lets EVERY tail pass fold twice. Fusing
// only #0 would leave the first tail pass with a single rho in hand, so it would
// fold once and write back the full half-length array — 3L of the tail's 4.67L.
// With both rhos in hand up front the tail series is 3.33L instead.
//
// Message #0 has eq shift 0 and scale ONE, so pairs 2z and 2z+1 index the
// split-eq table directly; message #1 has shift 1 and scale S_1, and its pair z
// lands on the same entry as pair 2z. eqlo/eqhi must be built before this launch.
__global__ void zerocheck_second_round_fold_with_lookahead(const uint8_t* __restrict__ a_packed,
                                    const uint8_t* __restrict__ b_packed,
                                    const F128* __restrict__ foldtable,
                                    const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                    int lobits, long long out_quads,
                                    F128* __restrict__ a_mlv, F128* __restrict__ b_mlv,
                                    F128* __restrict__ part) {
    __shared__ F128 sh[ZT_TPB];
    F256 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = F256{0, 0, 0, 0};

    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long z = t; z < out_quads; z += stride) {
        const uint8_t* ar = a_packed + z * 32;      // 4 rows x 8 bytes
        const uint8_t* br = b_packed + z * 32;
        long long o = 4 * z;
        F128 w[4], x[4];
#pragma unroll
        for (int t4 = 0; t4 < 4; t4++) {
            F128 av{0, 0}, bv{0, 0};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                av = f128_add(av, foldtable[j * 256 + ar[t4 * 8 + j]]);
                bv = f128_add(bv, foldtable[j * 256 + br[t4 * 8 + j]]);
            }
            w[t4] = av; x[t4] = bv;
            a_mlv[o + t4] = av; b_mlv[o + t4] = bv;
        }
        zt_accum_quad(acc, w, x,
                      evaluate_zerocheck_split_equality(eqlo, eqhi, z << 1, lobits),
                      evaluate_zerocheck_split_equality(eqlo, eqhi, (z << 1) + 1, lobits));
    }
    zt_reduce8(acc, sh, part, F128{1, 0});
}

// Hi-factored variant: when a block's whole quad range maps to ONE eqhi entry,
// the per-quad eq lookups need eqlo only and the single eqhi multiply moves to
// the end of the block reduction. That removes the two eqlo*eqhi products per
// quad (2 of ~20 multiplies) from a kernel whose fold is otherwise pure table
// lookups, i.e. the one place where added multiply work costs real time.
// Distributivity over the XOR-sum is exact in GF(2^128), so this is bit-identical.
//
// Blocks own contiguous chunks here rather than grid-striding, which is what
// makes the single-eqhi property hold. Quad z reads eq indices 2z and 2z+1, so
// the block spans 2*chunk eq indices — that is plan_zerocheck_high_bits with shift 1. (2z is
// even and a segment boundary is odd, so 2z and 2z+1 never straddle one.)
__global__ void zerocheck_second_round_fold_with_lookahead_high_bits(const uint8_t* __restrict__ a_packed,
                                        const uint8_t* __restrict__ b_packed,
                                        const F128* __restrict__ foldtable,
                                        const F128* __restrict__ eqlo, const F128* __restrict__ eqhi,
                                        int lobits, long long out_quads, long long chunk,
                                        F128* __restrict__ a_mlv, F128* __restrict__ b_mlv,
                                        F128* __restrict__ part) {
    __shared__ F128 sh[ZT_TPB];
    F256 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = F256{0, 0, 0, 0};

    long long start = (long long)blockIdx.x * chunk;
    long long end = start + chunk < out_quads ? start + chunk : out_quads;
    const long long mask = (1LL << lobits) - 1;
    for (long long z = start + threadIdx.x; z < end; z += blockDim.x) {
        const uint8_t* ar = a_packed + z * 32;
        const uint8_t* br = b_packed + z * 32;
        long long o = 4 * z;
        F128 w[4], x[4];
#pragma unroll
        for (int t4 = 0; t4 < 4; t4++) {
            F128 av{0, 0}, bv{0, 0};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                av = f128_add(av, foldtable[j * 256 + ar[t4 * 8 + j]]);
                bv = f128_add(bv, foldtable[j * 256 + br[t4 * 8 + j]]);
            }
            w[t4] = av; x[t4] = bv;
            a_mlv[o + t4] = av; b_mlv[o + t4] = bv;
        }
        long long ez = z << 1;
        zt_accum_quad(acc, w, x, eqlo[ez & mask], eqlo[(ez + 1) & mask]);
    }
    // Empty blocks hold a zero accumulator, so the multiplier is immaterial; the
    // guard is only to keep the eqhi read in range.
    F128 eh = start < out_quads ? eqhi[(start << 1) >> lobits] : F128{0, 0};
    zt_reduce8(acc, sh, part, eh);
}

// n_out = number of a_mlv/b_mlv rows produced (2^(m-6), a multiple of 4).
// d_out receives the 8 lookahead slots; d_part needs 8 * ZT_MAX_BLOCKS entries.
inline void launch_zerocheck_second_round_fold_with_lookahead(const uint8_t* d_a, const uint8_t* d_b, const F128* d_foldtable,
                                       const F128* d_eqlo, const F128* d_eqhi, int lobits,
                                       long long n_out, F128* d_a_mlv, F128* d_b_mlv,
                                       F128 scale_1, F128* d_part, F128* d_out) {
    long long out_quads = n_out / 4;
    int blocks = zt_blocks(out_quads);
    long long chunk;
    if (plan_zerocheck_high_bits(out_quads, 1, lobits, blocks, chunk))
        zerocheck_second_round_fold_with_lookahead_high_bits<<<blocks, ZT_TPB>>>(d_a, d_b, d_foldtable, d_eqlo, d_eqhi, lobits,
                                                    out_quads, chunk, d_a_mlv, d_b_mlv, d_part);
    else
        zerocheck_second_round_fold_with_lookahead<<<blocks, ZT_TPB>>>(d_a, d_b, d_foldtable, d_eqlo, d_eqhi, lobits,
                                                out_quads, d_a_mlv, d_b_mlv, d_part);
    combine_zerocheck_tail_lookahead_message<<<8, ZT_TPB>>>(d_part, blocks, F128{1, 0}, scale_1, d_out);
}
