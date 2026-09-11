use super::super::dsl::{Add3, Add32, AddConst32, words};
use super::{
    BLOCK_LEN_PORT, COUNTER_PORT, CV_PORT, FLAGS_PORT, MESSAGE_PORT, OUT_HI_PORT, OUT_LO_PORT,
    blake3,
};
use flock_core::circuit::boolean::{ColumnRole, ColumnSchema, ColumnVisitor};

pub struct Blake3Cols<T> {
    pub cv: [[T; 32]; 8],
    pub message: [[T; 32]; 16],
    pub counter: [[T; 32]; 2],
    pub block_len: [T; 32],
    pub flags: [T; 32],
    pub rounds: Vec<[GCols<T>; 8]>,
    pub out_lo: [[T; 32]; 8],
    pub out_hi: [[T; 32]; 8],
}

pub struct GCols<T> {
    pub a_first: Add3<T>,
    pub c_first: FirstC<T>,
    pub a_second: Add3<T>,
    pub c_second: Add32<T>,
}

pub enum FirstC<T> {
    Constant(AddConst32<T>),
    Variable(Add32<T>),
}

#[derive(Debug)]
pub struct Blake3Schema;

impl ColumnSchema for Blake3Schema {
    type Cols<T> = Blake3Cols<T>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Self::Cols<V::Value> {
        Blake3Cols {
            cv: words(v, CV_PORT, ColumnRole::Input, blake3::SLOT_BITS),
            message: words(v, MESSAGE_PORT, ColumnRole::Input, blake3::SLOT_BITS),
            counter: words(v, COUNTER_PORT, ColumnRole::Input, 32),
            block_len: v.word_aligned(BLOCK_LEN_PORT, ColumnRole::Input, 32),
            flags: v.word_aligned(FLAGS_PORT, ColumnRole::Input, 32),
            rounds: (0..7)
                .map(|r| {
                    std::array::from_fn(|i| {
                        let p = format!("rounds.{r}.g-{i}");
                        GCols {
                            a_first: Add3::columns(v, &format!("{p}.a-first")),
                            // Only these four C lanes still contain a fixed IV word.
                            c_first: if r == 0 && i < 4 {
                                FirstC::Constant(AddConst32::columns(
                                    v,
                                    &format!("{p}.c-first"),
                                    blake3::BLAKE3_IV[i],
                                ))
                            } else {
                                FirstC::Variable(Add32::columns(v, &format!("{p}.c-first")))
                            },
                            a_second: Add3::columns(v, &format!("{p}.a-second")),
                            c_second: Add32::columns(v, &format!("{p}.c-second")),
                        }
                    })
                })
                .collect(),
            out_lo: words(v, OUT_LO_PORT, ColumnRole::Output, blake3::SLOT_BITS),
            out_hi: words(v, OUT_HI_PORT, ColumnRole::Output, blake3::SLOT_BITS / 2),
        }
    }
}
