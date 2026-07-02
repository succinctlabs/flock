//! E0 — the relabeling soundness oracle.
//!
//! The whole layout change is a permutation of MLE variables; these tests pin
//! that down: `MLE_row(z, x) == MLE_l1(permute(z), relabel(x))`, plus
//! consistency of the bit/word/point maps with each other.

use flock_autoresearch_probes::layout::{C_LOG, Layout};
use flock_core::field::F128;
use flock_core::pcs::pack_witness;

struct Rng(u64);
impl Rng {
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
    fn bits(&mut self, n: usize) -> Vec<bool> {
        (0..n).map(|_| self.next_u64() & 1 == 1).collect()
    }
}

/// Naive MLE evaluation by successive LSB folds: `point[0]` binds address
/// bit 0. In char 2: fold(lo, hi, x) = lo + x·(lo + hi).
fn mle_eval(z: &[bool], point: &[F128]) -> F128 {
    assert_eq!(z.len(), 1 << point.len());
    let mut table: Vec<F128> = z
        .iter()
        .map(|&b| if b { F128::ONE } else { F128::ZERO })
        .collect();
    for &x in point {
        let half = table.len() / 2;
        for i in 0..half {
            let lo = table[2 * i];
            let hi = table[2 * i + 1];
            table[i] = lo + x * (lo + hi);
        }
        table.truncate(half);
    }
    table[0]
}

/// perm_bit is a bijection and consistent with the (o, j) address maps.
#[test]
fn addr_maps_consistent() {
    let mut rng = Rng(0xE0_01);
    for &(k_log, n_log) in &[(7usize, 3usize), (9, 4), (10, 6), (16, 7)] {
        let l = Layout::new(k_log, n_log);
        let m = l.m();
        // Bijection.
        let mut seen = vec![false; m];
        for i in 0..m {
            let p = l.perm_bit(i);
            assert!(!seen[p], "perm_bit not injective at k={k_log},n={n_log}");
            seen[p] = true;
        }
        // Address maps agree with the bit permutation: bit i of addr_row ==
        // bit perm(i) of addr_l1, for random (o, j).
        for _ in 0..200 {
            let o = (rng.next_u64() as usize) & ((1 << n_log) - 1);
            let j = (rng.next_u64() as usize) & ((1 << k_log) - 1);
            let ar = l.addr_row(o, j);
            let al = l.addr_l1(o, j);
            for i in 0..m {
                assert_eq!(
                    (ar >> i) & 1,
                    (al >> l.perm_bit(i)) & 1,
                    "bit {i} mismatch (k={k_log},n={n_log},o={o},j={j})"
                );
            }
        }
    }
}

/// The core soundness statement: MLE evaluation is invariant under the
/// permutation-with-relabeled-point.
#[test]
fn mle_relabel_invariance() {
    let mut rng = Rng(0xE0_02);
    for &(k_log, n_log) in &[(7usize, 3usize), (8, 4), (9, 5)] {
        let l = Layout::new(k_log, n_log);
        let m = l.m();
        let z = rng.bits(1 << m);
        let z_l1 = l.permute_bits_row_to_l1(&z);
        for _ in 0..4 {
            let x: Vec<F128> = (0..m).map(|_| rng.f128()).collect();
            let y = l.relabel_point(&x);
            assert_eq!(
                mle_eval(&z, &x),
                mle_eval(&z_l1, &y),
                "MLE not invariant at k={k_log},n={n_log}"
            );
        }
    }
}

/// Word-level permutation (the one producers implement) matches the bit-level
/// reference through pack_witness, and the u64 variant matches the F128 one.
#[test]
fn word_permutation_matches_bits() {
    let mut rng = Rng(0xE0_03);
    for &(k_log, n_log) in &[(7usize, 4usize), (9, 4), (10, 5)] {
        let l = Layout::new(k_log, n_log);
        let m = l.m();
        let z = rng.bits(1 << m);

        let packed_row = pack_witness(&z, m);
        let by_words = l.permute_words_row_to_l1(&packed_row);
        let by_bits = pack_witness(&l.permute_bits_row_to_l1(&z), m);
        assert_eq!(by_words, by_bits, "word/bit permute mismatch k={k_log},n={n_log}");

        // u64 variant agrees (F128 = two LE u64s).
        let row_u64: Vec<u64> = packed_row.iter().flat_map(|w| [w.lo, w.hi]).collect();
        let l1_u64 = l.permute_words_u64_row_to_l1(&row_u64);
        let expect: Vec<u64> = by_words.iter().flat_map(|w| [w.lo, w.hi]).collect();
        assert_eq!(l1_u64, expect);
    }
}

/// Sanity: under L1′ the in-word dims stay put — the ring-switch prefix
/// (address dims 0..7) is untouched by the relabeling.
#[test]
fn ring_prefix_untouched() {
    for &(k_log, n_log) in &[(7usize, 3usize), (16, 13)] {
        let l = Layout::new(k_log, n_log);
        for i in 0..C_LOG {
            assert_eq!(l.perm_bit(i), i);
        }
        // Batch dims land immediately above the word: k_log.. -> 7..7+n_log.
        for i in 0..n_log {
            assert_eq!(l.perm_bit(k_log + i), C_LOG + i);
        }
    }
}
