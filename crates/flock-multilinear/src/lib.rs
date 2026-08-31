//! Multilinear polynomial primitives for binary extension fields.

use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::mem::forget;
use std::ops::{Add, Mul};

/// Maps point coordinates to bits in a multilinear table index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexOrder {
    /// `point[0]` maps to the least significant index bit.
    LowToHigh,
    /// `point[0]` maps to the most significant index bit.
    HighToLow,
}

/// Builds `eq(point, ·)` in the selected index order.
pub fn eq_table<T>(point: &[T], one: T, order: IndexOrder) -> Vec<T>
where
    T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T>,
{
    eq_table_scaled(point, one, one, order)
}

/// Builds `seed * eq(point, ·)` in the selected index order.
pub fn eq_table_scaled<T>(point: &[T], one: T, seed: T, order: IndexOrder) -> Vec<T>
where
    T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T>,
{
    match order {
        IndexOrder::LowToHigh => build_eq_table(point.iter().copied(), point.len(), one, seed),
        IndexOrder::HighToLow => {
            build_eq_table(point.iter().rev().copied(), point.len(), one, seed)
        }
    }
}

/// Evaluates `eq(left, right)` over a binary extension field.
pub fn eq_eval<T>(left: &[T], right: &[T], one: T) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(left.len(), right.len(), "equality point shape");
    left.iter()
        .zip(right)
        .fold(one, |value, (&left, &right)| value * (one + left + right))
}

fn build_eq_table<T, I>(coordinates: I, num_variables: usize, one: T, seed: T) -> Vec<T>
where
    T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T>,
    I: ExactSizeIterator<Item = T>,
{
    assert_eq!(coordinates.len(), num_variables, "equality point shape");
    let len = 1usize
        .checked_shl(num_variables.try_into().expect("point is too large"))
        .expect("equality table is too large");
    let mut table: Vec<MaybeUninit<T>> = Vec::with_capacity(len);
    table.resize_with(len, MaybeUninit::uninit);
    table[0].write(seed);

    const PARALLEL_THRESHOLD: usize = 1 << 12;
    for (variable, coordinate) in coordinates.enumerate() {
        let half = 1usize << variable;
        let (low, high) = table.split_at_mut(half);
        let high = &mut high[..half];
        let zero_weight = one + coordinate;

        if half < PARALLEL_THRESHOLD {
            low.iter_mut().zip(high).for_each(|(low, high)| {
                // SAFETY: Each completed level initialized the low half.
                let value = unsafe { *low.assume_init_ref() };
                high.write(value * coordinate);
                low.write(value * zero_weight);
            });
        } else {
            low.par_iter_mut().zip(high).for_each(|(low, high)| {
                // SAFETY: Each completed level initialized the low half.
                let value = unsafe { *low.assume_init_ref() };
                high.write(value * coordinate);
                low.write(value * zero_weight);
            });
        }
    }
    // SAFETY: The final level initializes all items. `T: Copy` has no destructor.
    unsafe {
        let pointer = table.as_mut_ptr().cast::<T>();
        let len = table.len();
        let capacity = table.capacity();
        forget(table);
        Vec::from_raw_parts(pointer, len, capacity)
    }
}

/// Evaluates a multilinear table at `point` in the selected index order.
pub fn evaluate<T>(table: &[T], point: &[T], order: IndexOrder) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    let expected_len = 1usize
        .checked_shl(point.len().try_into().expect("point is too large"))
        .expect("multilinear table is too large");
    assert_eq!(table.len(), expected_len, "multilinear table shape");

    let mut values = table.to_vec();
    match order {
        IndexOrder::LowToHigh => {
            for &coordinate in point {
                fold_low(&mut values, coordinate);
            }
        }
        IndexOrder::HighToLow => {
            for &coordinate in point.iter().rev() {
                fold_low(&mut values, coordinate);
            }
        }
    }
    values[0]
}

/// Binds the variable stored in the least significant index bit.
pub fn fold_low<T>(table: &mut Vec<T>, coordinate: T)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert!(table.len() >= 2 && table.len().is_power_of_two());
    let half = table.len() / 2;
    for index in 0..half {
        let low = table[2 * index];
        let high = table[2 * index + 1];
        table[index] = low + coordinate * (high + low);
    }
    table.truncate(half);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::array::from_fn;
    use std::ops::{Add, Mul};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct F8(u8);

    #[allow(clippy::suspicious_arithmetic_impl)]
    impl Add for F8 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 ^ rhs.0)
        }
    }

    impl Mul for F8 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            let mut left = self.0;
            let mut right = rhs.0;
            let mut product = 0;
            for _ in 0..8 {
                if right & 1 == 1 {
                    product ^= left;
                }
                let carry = left & 0x80;
                left <<= 1;
                if carry != 0 {
                    left ^= 0x1b;
                }
                right >>= 1;
            }
            Self(product)
        }
    }

    const ONE: F8 = F8(1);

    #[test]
    fn equality_tables_follow_index_order() {
        let point = [F8(2), F8(3), F8(5)];
        for order in [IndexOrder::LowToHigh, IndexOrder::HighToLow] {
            let table = eq_table(&point, ONE, order);
            for (index, &value) in table.iter().enumerate() {
                let expected = point.iter().enumerate().fold(ONE, |value, (variable, &r)| {
                    let bit = match order {
                        IndexOrder::LowToHigh => variable,
                        IndexOrder::HighToLow => point.len() - variable - 1,
                    };
                    value * if index >> bit & 1 == 1 { r } else { ONE + r }
                });
                assert_eq!(value, expected);
            }
        }
    }

    #[test]
    #[should_panic(expected = "equality point shape")]
    fn equality_table_builder_rejects_a_short_iterator() {
        let coordinates = [F8(2), F8(3)];
        let _ = build_eq_table(coordinates.into_iter().take(1), 2, ONE, ONE);
    }

    #[test]
    fn evaluation_supports_both_index_orders() {
        let point = [F8(7), F8(11), F8(13)];
        let values = [
            F8(19),
            F8(23),
            F8(29),
            F8(31),
            F8(37),
            F8(41),
            F8(43),
            F8(47),
        ];
        for order in [IndexOrder::LowToHigh, IndexOrder::HighToLow] {
            let weights = eq_table(&point, ONE, order);
            let expected = values
                .iter()
                .zip(weights)
                .fold(F8(0), |sum, (&value, weight)| sum + value * weight);
            assert_eq!(evaluate(&values, &point, order), expected);
        }
    }

    #[test]
    fn scaled_table_applies_seed_during_the_build() {
        let point = [F8(2), F8(3)];
        let seed = F8(17);
        let scaled = eq_table_scaled(&point, ONE, seed, IndexOrder::LowToHigh);
        let expected: Vec<_> = eq_table(&point, ONE, IndexOrder::LowToHigh)
            .into_iter()
            .map(|value| seed * value)
            .collect();
        assert_eq!(scaled, expected);
    }

    #[test]
    fn equality_evaluation_matches_a_table_entry() {
        let point = [F8(2), F8(3), F8(5)];
        let table = eq_table(&point, ONE, IndexOrder::LowToHigh);
        for index in 0..table.len() {
            let index_point: [F8; 3] = from_fn(|bit| F8(((index >> bit) & 1) as u8));
            assert_eq!(eq_eval(&point, &index_point, ONE), table[index]);
        }
    }
}
