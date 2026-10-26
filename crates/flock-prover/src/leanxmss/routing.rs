//! The permutation network that sorts a layer's chains by digit.
//!
//! A fixed circuit has to provision every chain for all three steps (174
//! hashes per layer). But in a codeword at most 22 chains take the step
//! leaving position 0 and at most 34 the step leaving position 1, so if the
//! chains are first permuted with the small digits in front, steps 0 and 1
//! need only 22 and 34 hash slots. This module is the network that does the
//! permuting: an in-place Beneš network on 64 positions (the 58 chains and 6
//! dummies), and the prover's routing of it.
//!
//! **In place.** Stage `t` acts on one bit `b` of the position: for each pair
//! of positions `(p, p | 2^b)` with bit `b` of `p` clear, a control bit says
//! whether the two swap. The stage bits are `0,1,2,3,4,5,4,3,2,1,0` (a Beneš
//! network, rearrangeable) and a twelfth stage on bit 2 that the router leaves
//! as the identity, so the twelve stages split into four groups of three
//! acting on three distinct bits each: one circuit block per group per eight
//! positions sharing the other three bits.
//!
//! Any control bits give a permutation, so the circuit need not check the
//! routing: it checks the routed digits instead. Running the same blocks
//! backwards with the same control bits is the inverse permutation.

/// Positions in the network: the chains, padded to a power of two.
pub const WIDTH: usize = 64;
/// Bits of a position.
const LOG_WIDTH: usize = 6;
/// Switches per stage.
pub const PAIRS: usize = WIDTH / 2;
/// Stages, in groups of three.
pub const STAGES: usize = 12;
/// Global position bit of each stage.
pub const STAGE_BITS: [usize; STAGES] = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 2];
/// Groups of three stages; each group's blocks hold the eight positions that
/// share the three bits the group does not touch.
pub const GROUPS: usize = STAGES / 3;
/// Blocks per group.
pub const BLOCKS: usize = WIDTH / 8;

/// The three global bits a group's stages act on, in stage order: local bit
/// `l` of a block is global bit `group_bits(g)[l]`.
pub fn group_bits(g: usize) -> [usize; 3] {
    [
        STAGE_BITS[3 * g],
        STAGE_BITS[3 * g + 1],
        STAGE_BITS[3 * g + 2],
    ]
}

/// The global position of local element `e` of block `block` in group `g`:
/// the group's three bits come from `e`, the other three from `block` (low
/// to high in both).
pub fn position(g: usize, block: usize, e: usize) -> usize {
    let bits = group_bits(g);
    let mut p = 0;
    let mut rest = block;
    for b in 0..LOG_WIDTH {
        if let Some(l) = bits.iter().position(|&x| x == b) {
            p |= ((e >> l) & 1) << b;
        } else {
            p |= (rest & 1) << b;
            rest >>= 1;
        }
    }
    p
}

/// Within a block, local stage `s` pairs elements differing in local bit
/// `s`; `pair(s, q)` is the pair's lower element (bit `s` clear) for pair
/// index `q`, whose two bits are the other two local bits, low first.
pub fn pair(s: usize, q: usize) -> usize {
    let (o0, o1) = match s {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    ((q & 1) << o0) | (((q >> 1) & 1) << o1)
}

/// Control bits of every stage, by pair base position: `controls[t][p]` for
/// `p` with bit `STAGE_BITS[t]` clear (other entries unused).
pub type Controls = [[bool; WIDTH]; STAGES];

/// Apply the network to `items` in place (stage order), or its inverse.
pub fn apply<T: Copy>(controls: &Controls, items: &mut [T; WIDTH], inverse: bool) {
    let mut stages: Vec<usize> = (0..STAGES).collect();
    if inverse {
        stages.reverse();
    }
    for t in stages {
        let bit = 1 << STAGE_BITS[t];
        for p in (0..WIDTH).filter(|p| p & bit == 0) {
            if controls[t][p] {
                items.swap(p, p | bit);
            }
        }
    }
}

/// Control bits that move the item at position `p` to `dest[p]`, for a
/// permutation `dest`. The looping algorithm on the in-place Beneš stages;
/// the twelfth stage is the identity.
pub fn route(dest: &[usize; WIDTH]) -> Controls {
    let mut seen = [false; WIDTH];
    assert!(
        dest.iter()
            .all(|&d| d < WIDTH && !std::mem::replace(&mut seen[d], true))
    );
    let mut controls = [[false; WIDTH]; STAGES];
    let positions: Vec<usize> = (0..WIDTH).collect();
    route_level(&mut controls, 0, &positions, dest);
    controls
}

/// Route the subnetwork of `positions` (those sharing the bits fixed by the
/// outer levels) whose outer stages are `level` and `10 - level`;
/// `dest[p]` is where the item now at `p` must leave this subnetwork.
fn route_level(controls: &mut Controls, level: usize, positions: &[usize], dest: &[usize; WIDTH]) {
    let bit = 1 << STAGE_BITS[level];
    let (first, last) = (level, 2 * (LOG_WIDTH - 1) - level);
    if first == last {
        for &p in positions.iter().filter(|&&p| p & bit == 0) {
            controls[first][p] = dest[p] & bit != 0;
        }
        return;
    }
    let mut src = [usize::MAX; WIDTH];
    for &p in positions {
        src[dest[p]] = p;
    }
    // side[p]: the half (bit `bit` after the first stage) the item at `p`
    // crosses in. Paired inputs take opposite halves, and so must the
    // sources of paired outputs; the looping algorithm 2-colors that.
    let mut side = [u8::MAX; WIDTH];
    for &start in positions {
        if side[start] != u8::MAX {
            continue;
        }
        let mut p = start;
        let s = 0u8;
        while side[p] == u8::MAX {
            side[p] = s;
            side[p ^ bit] = 1 - s;
            // The partner input's output is paired with another output,
            // whose source must take side `s` again.
            p = src[dest[p ^ bit] ^ bit];
        }
    }
    let mut inner = [0usize; WIDTH];
    let mut halves: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for &p in positions {
        let s = side[p] as usize;
        let here = (p & !bit) | (s * bit);
        if p & bit == 0 {
            controls[first][p] = (s * bit) != (p & bit);
        }
        inner[here] = (dest[p] & !bit) | (s * bit);
        halves[s].push(here);
    }
    // Last stage: the item leaving half `s` at `q` is destined for `dest`;
    // the switch at the pair swaps when that differs from `q`.
    for &p in positions {
        let s = side[p] as usize;
        let q = (dest[p] & !bit) | (s * bit);
        if q & bit == 0 {
            controls[last][q] = dest[p] != q;
        }
    }
    for half in &halves {
        route_level(controls, level + 1, half, &inner);
    }
}

/// Pack one group's control bits into its blocks' control words: block `b`'s
/// bit `4s + q` is local stage `s`'s pair `q`.
pub fn block_controls(controls: &Controls, g: usize) -> [u16; BLOCKS] {
    std::array::from_fn(|block| {
        let mut word = 0u16;
        for s in 0..3 {
            for q in 0..4 {
                let p = position(g, block, pair(s, q));
                if controls[3 * g + s][p] {
                    word |= 1 << (4 * s + q);
                }
            }
        }
        word
    })
}

/// The routing that puts the chains in order of digit (dummies, as digit 3,
/// last), stable within a digit: `dest[i]` is chain `i`'s routed position.
pub fn sort_by_digit(digits: &[u8]) -> [usize; WIDTH] {
    let key = |i: usize| digits.get(i).copied().unwrap_or(3);
    let mut order: Vec<usize> = (0..WIDTH).collect();
    order.sort_by_key(|&i| key(i));
    let mut dest = [0; WIDTH];
    for (rank, &i) in order.iter().enumerate() {
        dest[i] = rank;
    }
    dest
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n as u64) as usize
        }
    }

    #[test]
    fn routes_every_permutation_it_is_given() {
        let mut rng = Rng(9);
        for trial in 0..500 {
            let mut dest: [usize; WIDTH] = std::array::from_fn(|i| i);
            if trial > 0 {
                for i in (1..WIDTH).rev() {
                    dest.swap(i, rng.below(i + 1));
                }
            }
            let controls = route(&dest);
            assert!(
                controls[STAGES - 1].iter().all(|&c| !c),
                "the extra stage is the identity"
            );
            let mut items: [usize; WIDTH] = std::array::from_fn(|i| i);
            apply(&controls, &mut items, false);
            for (i, &d) in dest.iter().enumerate() {
                assert_eq!(items[d], i, "trial {trial}: item {i}");
            }
            apply(&controls, &mut items, true);
            assert_eq!(items, std::array::from_fn(|i| i), "the inverse undoes it");
        }
    }

    #[test]
    fn blocks_partition_each_group() {
        for g in 0..GROUPS {
            let mut seen = [false; WIDTH];
            for block in 0..BLOCKS {
                for e in 0..8 {
                    let p = position(g, block, e);
                    assert!(!std::mem::replace(&mut seen[p], true));
                }
                for s in 0..3 {
                    for q in 0..4 {
                        let lo = pair(s, q);
                        assert_eq!(lo & (1 << s), 0);
                        let (a, b) = (position(g, block, lo), position(g, block, lo | 1 << s));
                        assert_eq!(a ^ b, 1 << group_bits(g)[s]);
                    }
                }
            }
        }
    }

    #[test]
    fn sorting_by_digit_fronts_the_small_digits() {
        let digits: Vec<u8> = (0..58).map(|i| ((i * 7 + 3) % 4) as u8).collect();
        let dest = sort_by_digit(&digits);
        let mut routed = [3u8; WIDTH];
        for (i, &d) in dest.iter().enumerate() {
            routed[d] = digits.get(i).copied().unwrap_or(3);
        }
        assert!(routed.windows(2).all(|w| w[0] <= w[1]));
    }
}
