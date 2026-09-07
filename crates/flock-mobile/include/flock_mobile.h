/* C interface to the Flock prover for iOS. Mirrors crates/flock-mobile/src/lib.rs. */
#ifndef FLOCK_MOBILE_H
#define FLOCK_MOBILE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t n_blocks;        /* total compressions proved */
    double   prove_ms;        /* Tower::prove wall time */
    double   discharge_ms;    /* Tower::discharge_root wall time */
    double   root_proof_kib;  /* root proof size, KiB */
    uint64_t peak_rss_bytes;  /* process peak resident set, bytes */
    uint32_t threads;         /* rayon pool size */
    uint32_t h_end_0;         /* first word of the final chain state */
} FlockMobileReport;

/* Prove + discharge one 128-bit BLAKE3-chain tower. n_leaves: even, >= 4.
   Returns 0 on success, 1 on an internal panic, 2 if out is NULL. */
int32_t flock_mobile_tower_demo(uint32_t n_leaves, uint32_t blocks_per_leaf,
                                FlockMobileReport *out);

/* Prove ONE chain leaf of n_blocks compressions, no folding. Fills n_blocks,
   prove_ms, peak_rss_bytes, threads. Same return codes. */
int32_t flock_mobile_leaf_demo(uint32_t n_blocks, FlockMobileReport *out);

uint32_t flock_mobile_thread_count(void);
uint64_t flock_mobile_peak_rss_bytes(void);

#ifdef __cplusplus
}
#endif
#endif /* FLOCK_MOBILE_H */
