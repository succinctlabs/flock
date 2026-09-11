use super::*;

pub struct Add32<T> {
    pub carry_product: [T; 31],
}

impl Add32<Var> {
    pub fn columns<V: ColumnVisitor>(v: &mut V, p: &str) -> Add32<V::Value> {
        Add32 {
            carry_product: v.word(&format!("{p}.carry_product"), ColumnRole::Witness),
        }
    }

    pub fn eval(&self, b: &mut CircuitBuilder, x: Word, y: Word) -> Word {
        b.operation(
            "Add32",
            &self.carry_product,
            [("x", x.to_vec()), ("y", y.to_vec())],
            |b| {
                let mut carry = b.zero();
                std::array::from_fn(|bit| {
                    let sum = b.xor3(x[bit], y[bit], carry);
                    if bit < 31 {
                        let lhs = b.xor2(x[bit], carry);
                        let rhs = b.xor2(y[bit], carry);
                        b.define_and(self.carry_product[bit], lhs, rhs);
                        carry = b.xor2(carry, self.carry_product[bit]);
                    }
                    sum
                })
            },
        )
    }
}

pub struct AddConst32<T> {
    pub carry_product: Vec<T>,
}

impl AddConst32<Var> {
    pub fn columns<V: ColumnVisitor>(v: &mut V, p: &str, constant: u32) -> AddConst32<V::Value> {
        let count = 31usize.saturating_sub(constant.trailing_zeros() as usize + 1);
        AddConst32 {
            carry_product: if count == 0 {
                Vec::new()
            } else {
                v.bits(&format!("{p}.carry_product"), ColumnRole::Witness, count, 1)
            },
        }
    }

    pub fn eval(&self, b: &mut CircuitBuilder, k: u32, y: Word) -> Word {
        let seed = k.trailing_zeros() as usize + 1;
        assert_eq!(self.carry_product.len(), 31usize.saturating_sub(seed));
        let k = constant(b, k);
        let run = |b: &mut CircuitBuilder| {
            let mut carry = b.zero();
            std::array::from_fn(|bit| {
                if bit == seed {
                    carry = y[seed - 1];
                }
                let sum = b.xor3(k[bit], y[bit], carry);
                if (seed..31).contains(&bit) {
                    let lhs = b.xor2(k[bit], carry);
                    let rhs = b.xor2(y[bit], carry);
                    b.define_and(self.carry_product[bit - seed], lhs, rhs);
                    carry = b.xor2(carry, self.carry_product[bit - seed]);
                }
                sum
            })
        };
        if self.carry_product.is_empty() {
            run(b)
        } else {
            b.operation(
                "AddConst32",
                &self.carry_product,
                [("constant", k.to_vec()), ("y", y.to_vec())],
                run,
            )
        }
    }
}

pub struct Add3<T> {
    pub majority_product: [T; 31],
    pub ripple_product: [T; 30],
}
pub struct Add4<T> {
    pub majority_first_product: [T; 31],
    pub majority_second_product: [T; 31],
    pub ripple_product: [T; 30],
}

impl Add3<Var> {
    pub fn columns<V: ColumnVisitor>(v: &mut V, p: &str) -> Add3<V::Value> {
        Add3 {
            majority_product: v.word(&format!("{p}.majority_product"), ColumnRole::Witness),
            ripple_product: v.word(&format!("{p}.ripple_product"), ColumnRole::Witness),
        }
    }
    pub fn eval(&self, b: &mut CircuitBuilder, x: Word, y: Word, shared: Word) -> Word {
        let owned: Vec<_> = self
            .majority_product
            .into_iter()
            .chain(self.ripple_product)
            .collect();
        b.operation(
            "Add3x32",
            &owned,
            [
                ("x", x.to_vec()),
                ("y", y.to_vec()),
                ("shared", shared.to_vec()),
            ],
            |b| {
                let (parity, majority) = carry_save(b, &self.majority_product, x, y, shared);
                ripple(b, &self.ripple_product, parity, majority)
            },
        )
    }
}

impl Add4<Var> {
    pub fn columns<V: ColumnVisitor>(v: &mut V, p: &str) -> Add4<V::Value> {
        Add4 {
            majority_first_product: v
                .word(&format!("{p}.majority_first_product"), ColumnRole::Witness),
            majority_second_product: v
                .word(&format!("{p}.majority_second_product"), ColumnRole::Witness),
            ripple_product: v.word(&format!("{p}.ripple_product"), ColumnRole::Witness),
        }
    }
    pub fn eval(
        &self,
        b: &mut CircuitBuilder,
        x: Word,
        y: Word,
        first: Word,
        second: Word,
    ) -> Word {
        let owned: Vec<_> = self
            .majority_first_product
            .into_iter()
            .chain(self.majority_second_product)
            .chain(self.ripple_product)
            .collect();
        b.operation(
            "Add4x32",
            &owned,
            [
                ("x", x.to_vec()),
                ("y", y.to_vec()),
                ("first", first.to_vec()),
                ("second", second.to_vec()),
            ],
            |b| {
                let (parity, majority) = carry_save(b, &self.majority_first_product, x, y, first);
                let (parity, majority) =
                    carry_save(b, &self.majority_second_product, parity, majority, second);
                ripple(b, &self.ripple_product, parity, majority)
            },
        )
    }
}

fn carry_save(
    b: &mut CircuitBuilder,
    cols: &[Var; 31],
    x: Word,
    y: Word,
    shared: Word,
) -> (Word, Word) {
    let mut parity = [b.zero(); 32];
    let mut majority = [b.zero(); 32];
    for bit in 0..32 {
        parity[bit] = b.xor3(x[bit], y[bit], shared[bit]);
        if bit < 31 {
            let lhs = b.xor2(x[bit], shared[bit]);
            let rhs = b.xor2(y[bit], shared[bit]);
            b.define_and(cols[bit], lhs, rhs);
            majority[bit + 1] = b.xor2(cols[bit], shared[bit]);
        }
    }
    (parity, majority)
}

fn ripple(b: &mut CircuitBuilder, cols: &[Var; 30], parity: Word, majority: Word) -> Word {
    let mut carry = b.zero();
    std::array::from_fn(|bit| {
        let sum = b.xor3(parity[bit], majority[bit], carry);
        if (1..=30).contains(&bit) {
            let lhs = b.xor2(parity[bit], carry);
            let rhs = b.xor2(majority[bit], carry);
            b.define_and(cols[bit - 1], lhs, rhs);
            carry = b.xor2(carry, cols[bit - 1]);
        }
        sum
    })
}
