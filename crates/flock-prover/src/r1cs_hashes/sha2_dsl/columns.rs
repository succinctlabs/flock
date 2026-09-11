use super::super::dsl::{Add3, Add4, Add32, AddConst32, words};
use super::{H_OUT_PORT, H_PORT, MESSAGE_PORT, sha2};
use flock_core::circuit::boolean::{ColumnRole, ColumnSchema, ColumnVisitor};

pub struct Sha256Cols<T> {
    pub h_in: [[T; 32]; 8],
    pub message: [[T; 32]; 16],
    pub schedule: Vec<Add4<T>>,
    pub rounds: Vec<RoundCols<T>>,
    pub output_add: [Add32<T>; 8],
    pub h_out: [[T; 32]; 8],
}

pub struct RoundCols<T> {
    pub choose_product: [T; 32],
    pub majority_product: [T; 32],
    pub add_constant: AddConst32<T>,
    pub t1: Add4<T>,
    pub a_new: Add3<T>,
    pub e_new: Add32<T>,
    pub state: Option<StateCols<T>>,
}

pub struct StateCols<T> {
    pub a: [T; 32],
    pub e: [T; 32],
}

#[derive(Debug)]
pub struct Sha256Schema;

impl ColumnSchema for Sha256Schema {
    type Cols<T> = Sha256Cols<T>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> Self::Cols<V::Value> {
        Sha256Cols {
            h_in: words(v, H_PORT, ColumnRole::Input, sha2::SLOT_BITS),
            message: words(v, MESSAGE_PORT, ColumnRole::Input, sha2::SLOT_BITS),
            schedule: (16..64)
                .map(|t| Add4::columns(v, &format!("schedule.{t}")))
                .collect(),
            rounds: (0..64)
                .map(|r| {
                    let p = format!("rounds.{r}");
                    RoundCols {
                        choose_product: v.word(&format!("{p}.choose_product"), ColumnRole::Witness),
                        majority_product: v
                            .word(&format!("{p}.majority_product"), ColumnRole::Witness),
                        add_constant: AddConst32::columns(
                            v,
                            &format!("{p}.add-constant"),
                            sha2::SHA256_K[r],
                        ),
                        t1: Add4::columns(v, &format!("{p}.t1")),
                        a_new: Add3::columns(v, &format!("{p}.a-new")),
                        e_new: Add32::columns(v, &format!("{p}.e-new")),
                        state: (r % sha2::EA_PERIOD == sha2::EA_PERIOD - 1).then(|| StateCols {
                            a: v.word(&format!("{p}.state.a"), ColumnRole::Witness),
                            e: v.word(&format!("{p}.state.e"), ColumnRole::Witness),
                        }),
                    }
                })
                .collect(),
            output_add: std::array::from_fn(|i| Add32::columns(v, &format!("output-add.{i}"))),
            h_out: words(v, H_OUT_PORT, ColumnRole::Output, sha2::SLOT_BITS),
        }
    }
}
