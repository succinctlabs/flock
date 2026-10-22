// Sixteen-message BLAKE3 compression on Apple SME2 (streaming SVE, SVL = 512).
//
// Provenance: written for Flock from the BLAKE3 specification and the Arm
// A-profile ISA (SME, SME2, SVE2). It is not derived from any other SME or SVE
// BLAKE3 implementation.
//
// Layout. A 512-bit streaming Z register holds sixteen 32-bit lanes, one per
// message, so a group hashes sixteen messages in lockstep:
//
//   z0-z15   the compression state v[0..16], word-major (z_i lane l = v_i of
//            message l). After the feed-forward, z0-z7 hold the chaining value.
//   z16-z31  the current block's message words m[0..16], word-major.
//
// Each round is eight G calls, run as two halves of four G's interleaved step
// by step (G4). SVE2 XAR (exclusive-or, then rotate right by an immediate) does
// each xor-rotate in one destructive instruction, so G needs no temporaries and
// all 32 Z registers go to state and message.
//
// The ZA tiles do the transposes. Message l's 64-byte block is loaded as
// horizontal slice l of ZA0.S; vertical slice w is then message word w across
// the sixteen messages. The loads run one block ahead: as soon as a block's
// words are in z16-z31, the next block's rows (or the next group's first
// block) start loading into ZA0 underneath the rounds. The chaining values go
// out through ZA1.S, so they never disturb those rows: z0-z7 become vertical
// slices 0-7, and horizontal slice l (first eight words) is message l's output.
//
// Memory. The SME unit reads through the L2 cache, and a group's sixteen
// messages are sixteen separate streams that the hardware prefetcher does not
// follow well; each block therefore prefetches the same block of the next
// group into L2 (on M4 this takes 736-byte leaves from ~187 to ~160 ps/B when
// streaming from DRAM, the same rate as from cache).
//
// ABI. These are ordinary private-ZA AAPCS64 functions called with PSTATE.SM = 0.
// The hash entry point refuses (returns 0) if TPIDR2_EL0 is non-null, i.e. if a
// caller's ZA contents are dormant awaiting a lazy save, so it never clobbers
// live ZA. It saves d8-d15 (smstart and smstop zero the vector registers) and
// leaves streaming mode and ZA off on return. Callers must block signals around
// the hash entry point: XNU runs a handler that interrupts streaming mode in
// streaming mode and returns with the Z registers cut to 128 bits.

.arch_extension sme2

// G on four independent (a, b, c, d, mx, my) column or diagonal tuples. The
// arguments are Z register numbers; BLAKE3 rotates right.
.macro G4 a0,b0,c0,d0,x0,y0, a1,b1,c1,d1,x1,y1, a2,b2,c2,d2,x2,y2, a3,b3,c3,d3,x3,y3
    add z\a0\().s, z\a0\().s, z\b0\().s
    add z\a1\().s, z\a1\().s, z\b1\().s
    add z\a2\().s, z\a2\().s, z\b2\().s
    add z\a3\().s, z\a3\().s, z\b3\().s
    add z\a0\().s, z\a0\().s, z\x0\().s
    add z\a1\().s, z\a1\().s, z\x1\().s
    add z\a2\().s, z\a2\().s, z\x2\().s
    add z\a3\().s, z\a3\().s, z\x3\().s
    xar z\d0\().s, z\d0\().s, z\a0\().s, #16
    xar z\d1\().s, z\d1\().s, z\a1\().s, #16
    xar z\d2\().s, z\d2\().s, z\a2\().s, #16
    xar z\d3\().s, z\d3\().s, z\a3\().s, #16
    add z\c0\().s, z\c0\().s, z\d0\().s
    add z\c1\().s, z\c1\().s, z\d1\().s
    add z\c2\().s, z\c2\().s, z\d2\().s
    add z\c3\().s, z\c3\().s, z\d3\().s
    xar z\b0\().s, z\b0\().s, z\c0\().s, #12
    xar z\b1\().s, z\b1\().s, z\c1\().s, #12
    xar z\b2\().s, z\b2\().s, z\c2\().s, #12
    xar z\b3\().s, z\b3\().s, z\c3\().s, #12
    add z\a0\().s, z\a0\().s, z\b0\().s
    add z\a1\().s, z\a1\().s, z\b1\().s
    add z\a2\().s, z\a2\().s, z\b2\().s
    add z\a3\().s, z\a3\().s, z\b3\().s
    add z\a0\().s, z\a0\().s, z\y0\().s
    add z\a1\().s, z\a1\().s, z\y1\().s
    add z\a2\().s, z\a2\().s, z\y2\().s
    add z\a3\().s, z\a3\().s, z\y3\().s
    xar z\d0\().s, z\d0\().s, z\a0\().s, #8
    xar z\d1\().s, z\d1\().s, z\a1\().s, #8
    xar z\d2\().s, z\d2\().s, z\a2\().s, #8
    xar z\d3\().s, z\d3\().s, z\a3\().s, #8
    add z\c0\().s, z\c0\().s, z\d0\().s
    add z\c1\().s, z\c1\().s, z\d1\().s
    add z\c2\().s, z\c2\().s, z\d2\().s
    add z\c3\().s, z\c3\().s, z\d3\().s
    xar z\b0\().s, z\b0\().s, z\c0\().s, #7
    xar z\b1\().s, z\b1\().s, z\c1\().s, #7
    xar z\b2\().s, z\b2\().s, z\c2\().s, #7
    xar z\b3\().s, z\b3\().s, z\c3\().s, #7
.endm

// One round: the column half, then the diagonal half. s0-s15 are the Z
// registers holding the round's permuted message words (16 + schedule[k]).
.macro ROUND s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15
    G4 0,4,8,12,\s0,\s1,     1,5,9,13,\s2,\s3,     2,6,10,14,\s4,\s5,    3,7,11,15,\s6,\s7
    G4 0,5,10,15,\s8,\s9,    1,6,11,12,\s10,\s11,  2,7,8,13,\s12,\s13,   3,4,9,14,\s14,\s15
.endm

// Load sixteen rows (messages) of one block into ZA0.S, row l from
// x8 + l * x1, under predicate p5. Advances x8.
.macro LOAD_ROWS
    ld1w    {za0h.s[w12, 0]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w12, 1]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w12, 2]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w12, 3]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w13, 0]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w13, 1]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w13, 2]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w13, 3]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w14, 0]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w14, 1]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w14, 2]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w14, 3]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w15, 0]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w15, 1]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w15, 2]}, p5/z, [x8]
    add     x8, x8, x1
    ld1w    {za0h.s[w15, 3]}, p5/z, [x8]
.endm

// Prefetch one 64-byte block of sixteen rows, row l at x8 + l * x1, into L2.
.macro PREFETCH_ROWS
    .rept 16
    prfm    pldl2keep, [x8]
    add     x8, x8, x1
    .endr
.endm

// Store output rows 4k..4k+3 from ZA1.S slice base \w (32 bytes each).
.macro STORE4 w
    st1w    {za1h.s[\w, 0]}, p3, [x5]
    st1w    {za1h.s[\w, 1]}, p3, [x5, x16, lsl #2]
    add     x5, x5, #64
    st1w    {za1h.s[\w, 2]}, p3, [x5]
    st1w    {za1h.s[\w, 3]}, p3, [x5, x16, lsl #2]
    add     x5, x5, #64
.endm

.text

// u64 flock_blake3_sme2_svl_bytes(void): the streaming vector length in bytes.
// Executes in non-streaming mode; call only when FEAT_SME is present.
.p2align 4
.globl _flock_blake3_sme2_svl_bytes
_flock_blake3_sme2_svl_bytes:
    rdsvl   x0, #1
    ret

// u64 flock_blake3_sme2_hash16(const u8 *data,   x0  message 0 of group 0
//                              usize stride,     x1  bytes between messages
//                              usize n_blocks,   x2  1..=16 blocks per message
//                              usize last_len,   x3  4..=64, a multiple of 4
//                              u32 flags,        w4  base | start << 8 | end << 16
//                              u8 *out,          x5  16 * groups * 32 bytes
//                              usize groups)     x6
//
// Hashes 16 * groups messages, message i at data + i * stride, into 32-byte
// chaining values at out + i * 32: the IV key, counter 0, block b carrying
// `base`, plus `start` on the first block and `end` on the last, which is
// last_len bytes long and zero-padded. Returns 1, or 0 without touching
// anything if TPIDR2_EL0 is non-null.
//
// Register use: x0 current group's message 0, x2 index of the last block,
// x6 groups left, x7 block index, x8 row pointer, x9 block-b pointer of
// message 0, w10 block flags, x11 block length, w12-w15 ZA slice bases
// 0/4/8/12, x16 scratch, x17 IV table.
.p2align 4
.globl _flock_blake3_sme2_hash16
_flock_blake3_sme2_hash16:
    mrs     x16, TPIDR2_EL0
    cbnz    x16, Lrefuse
    cbz     x6, Ldone
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    smstart
    ptrue   p0.s
    ptrue   p3.s, vl8
    mov     w12, #0
    mov     w13, #4
    mov     w14, #8
    mov     w15, #12
    adr     x17, Liv
    sub     x2, x2, #1

    // Rows of group 0, block 0 (last_len long if it is the only block).
    mov     x11, #64
    cmp     x2, #0
    csel    x11, x3, x11, eq
    lsr     x16, x11, #2
    whilelo p5.s, xzr, x16
    mov     x8, x0
    LOAD_ROWS

Lgroup:
    ld1rw   {z0.s}, p0/z, [x17]
    ld1rw   {z1.s}, p0/z, [x17, #4]
    ld1rw   {z2.s}, p0/z, [x17, #8]
    ld1rw   {z3.s}, p0/z, [x17, #12]
    ld1rw   {z4.s}, p0/z, [x17, #16]
    ld1rw   {z5.s}, p0/z, [x17, #20]
    ld1rw   {z6.s}, p0/z, [x17, #24]
    ld1rw   {z7.s}, p0/z, [x17, #28]
    mov     x7, #0
    mov     x9, x0

Lblock:
    // v[8..16] = IV[0..4], counter 0 (low, high), block length, flags. The
    // flags are base, | start on block 0, | end on the last block; the length
    // is last_len on the last block.
    and     w10, w4, #0xff
    ubfx    w16, w4, #8, #8
    cmp     x7, #0
    csel    w16, w16, wzr, eq
    orr     w10, w10, w16
    ubfx    w16, w4, #16, #8
    cmp     x7, x2
    csel    w16, w16, wzr, eq
    orr     w10, w10, w16
    mov     x11, #64
    csel    x11, x3, x11, eq
    ld1rw   {z8.s}, p0/z, [x17]
    ld1rw   {z9.s}, p0/z, [x17, #4]
    ld1rw   {z10.s}, p0/z, [x17, #8]
    ld1rw   {z11.s}, p0/z, [x17, #12]
    mov     z12.s, #0
    mov     z13.s, #0
    dup     z14.s, w11
    dup     z15.s, w10

    // Vertical slice w of the loaded rows -> message word w of the 16 messages.
    mova    {z16.s - z19.s}, za0v.s[w12, 0:3]
    mova    {z20.s - z23.s}, za0v.s[w13, 0:3]
    mova    {z24.s - z27.s}, za0v.s[w14, 0:3]
    mova    {z28.s - z31.s}, za0v.s[w15, 0:3]

    // Start loading the next rows under the rounds: block b + 1, or the next
    // group's block 0 (none after the last group).
    cmp     x7, x2
    b.eq    Lnext_group
    add     x8, x9, #64
    add     x16, x7, #1
    mov     x11, #64
    cmp     x16, x2
    csel    x11, x3, x11, eq
    b       Lload_next
Lnext_group:
    cmp     x6, #1
    b.eq    Lrounds
    add     x8, x0, x1, lsl #4
    mov     x11, #64
    cmp     x2, #0
    csel    x11, x3, x11, eq
Lload_next:
    lsr     x16, x11, #2
    whilelo p5.s, xzr, x16
    LOAD_ROWS
    // And bring this block of the next group into L2.
    add     x8, x9, x1, lsl #4
    PREFETCH_ROWS

Lrounds:
    // Seven rounds. The message registers are 16 + BLAKE3's per-round
    // permuted word indices.
    ROUND 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ROUND 18, 22, 19, 26, 23, 16, 20, 29, 17, 27, 28, 21, 25, 30, 31, 24
    ROUND 19, 20, 26, 28, 29, 18, 23, 30, 22, 21, 25, 16, 27, 31, 24, 17
    ROUND 26, 23, 28, 25, 30, 19, 29, 31, 20, 16, 27, 18, 21, 24, 17, 22
    ROUND 28, 29, 25, 27, 31, 26, 30, 24, 23, 18, 21, 19, 16, 17, 22, 20
    ROUND 25, 30, 27, 21, 24, 28, 31, 17, 29, 19, 16, 26, 18, 22, 20, 23
    ROUND 27, 31, 21, 16, 17, 25, 24, 22, 30, 26, 18, 28, 19, 20, 23, 29

    // Feed-forward: the chaining value is v[i] ^ v[i + 8].
    eor     z0.d, z0.d, z8.d
    eor     z1.d, z1.d, z9.d
    eor     z2.d, z2.d, z10.d
    eor     z3.d, z3.d, z11.d
    eor     z4.d, z4.d, z12.d
    eor     z5.d, z5.d, z13.d
    eor     z6.d, z6.d, z14.d
    eor     z7.d, z7.d, z15.d

    add     x9, x9, #64
    add     x7, x7, #1
    cmp     x7, x2
    b.ls    Lblock

    // Chaining value word i -> vertical slice i of ZA1.S; row l is message l's
    // output (odd rows land 8 words, 32 bytes, after even ones).
    mova    za1v.s[w12, 0:3], {z0.s - z3.s}
    mova    za1v.s[w13, 0:3], {z4.s - z7.s}
    mov     x16, #8
    STORE4  w12
    STORE4  w13
    STORE4  w14
    STORE4  w15

    add     x0, x0, x1, lsl #4          // next 16 messages
    subs    x6, x6, #1
    b.ne    Lgroup

    smstop
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
Ldone:
    mov     x0, #1
    ret
Lrefuse:
    mov     x0, #0
    ret

.p2align 2
Liv:
    .word 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A
    .word 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
