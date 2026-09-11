//! Reusable arithmetic with explicit owned witnesses and virtual XORs.

use flock_core::circuit::boolean::{
    CircuitBuilder, ColumnRole, ColumnVisitor, LinearExpr as Expr, Var,
};

/// Retains the pilot's carry materialization, including the discarded final carry.
pub struct Add64<T> {
    pub value: [T; 64],
    pub generated: [T; 64],
    pub propagated: [T; 64],
}

impl Add64<Var> {
    pub fn columns<V: ColumnVisitor>(v: &mut V, p: &str) -> Add64<V::Value> {
        Add64 {
            value: v.word(&format!("{p}.value"), ColumnRole::Witness),
            generated: v.word(&format!("{p}.generated"), ColumnRole::Witness),
            propagated: v.word(&format!("{p}.propagated"), ColumnRole::Witness),
        }
    }

    pub fn eval(&self, b: &mut CircuitBuilder, lhs: [Expr; 64], rhs: [Expr; 64]) -> [Expr; 64] {
        let owned: Vec<_> = self
            .value
            .into_iter()
            .chain(self.generated)
            .chain(self.propagated)
            .collect();
        b.operation(
            "WrappingAdd64",
            &owned,
            [("lhs", lhs.to_vec()), ("rhs", rhs.to_vec())],
            |b| {
                let mut carry = b.zero();
                for i in 0..64 {
                    let propagate = b.xor2(lhs[i], rhs[i]);
                    let sum = b.xor2(propagate, carry);
                    b.define_linear(self.value[i], sum);
                    b.define_and(self.generated[i], lhs[i], rhs[i]);
                    b.define_and(self.propagated[i], propagate, carry);
                    carry = b.xor2(self.generated[i], self.propagated[i]);
                }
                self.value.map(Into::into)
            },
        )
    }
}

pub struct Or32<T> {
    pub overlap: [T; 31],
}

impl Or32<Var> {
    pub fn columns<V: ColumnVisitor>(v: &mut V, p: &str) -> Or32<V::Value> {
        Or32 {
            overlap: v.word(&format!("{p}.overlap"), ColumnRole::Witness),
        }
    }

    pub fn eval(&self, b: &mut CircuitBuilder, bits: [Expr; 32]) -> Expr {
        let [result] = b.operation("Or32", &self.overlap, [("bits", bits.to_vec())], |b| {
            let mut any = bits[0];
            for (overlap, bit) in self.overlap.into_iter().zip(&bits[1..]) {
                b.define_and(overlap, any, *bit);
                any = b.xor3(any, *bit, overlap);
            }
            [any]
        });
        result
    }
}

pub struct SelectByte<T> {
    /// Four pair selections, then two, then one; eight products per selection.
    pub products: [[T; 8]; 7],
}

impl SelectByte<Var> {
    pub fn columns<V: ColumnVisitor>(v: &mut V, p: &str) -> SelectByte<V::Value> {
        SelectByte {
            products: std::array::from_fn(|i| {
                v.word(&format!("{p}.products-{i}"), ColumnRole::Witness)
            }),
        }
    }

    pub fn eval(&self, b: &mut CircuitBuilder, memory: [Expr; 64], offset: [Expr; 3]) -> [Expr; 8] {
        let owned: Vec<_> = self.products.into_iter().flatten().collect();
        b.operation(
            "SelectByte64",
            &owned,
            [("memory", memory.to_vec()), ("offset", offset.to_vec())],
            |b| {
                let mut products = self.products.iter();
                let mut candidates: Vec<[Expr; 8]> = (0..8)
                    .map(|byte| std::array::from_fn(|bit| memory[8 * byte + bit]))
                    .collect();
                for selector in offset {
                    candidates = candidates
                        .chunks_exact(2)
                        .map(|pair| {
                            let products = products.next().unwrap();
                            std::array::from_fn(|bit| {
                                let difference = b.xor2(pair[0][bit], pair[1][bit]);
                                b.define_and(products[bit], selector, difference);
                                b.xor2(pair[0][bit], products[bit])
                            })
                        })
                        .collect();
                }
                candidates[0]
            },
        )
    }
}
