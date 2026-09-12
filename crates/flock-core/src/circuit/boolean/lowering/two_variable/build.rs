use super::*;
use crate::circuit::boolean::builder::{fresh_circuit_id, symmetric_difference};
use crate::circuit::boolean::{RowKind, ValueIndex};

impl LoweredCircuit {
    pub(super) fn build(
        source: &BooleanCircuit,
        placement: &PhysicalLayout,
        mode: LoweringMode,
    ) -> Result<Self, LayoutError> {
        placement.validate_for(source)?;
        let require_identity = mode == LoweringMode::RequireIdentityC;
        let assertions = source
            .rows
            .iter()
            .filter(|row| require_identity && row.kind == RowKind::Constraint)
            .count();
        let extra = assertions
            .checked_mul(2)
            .ok_or(LayoutError::PositionOverflow)?;
        let value_count = source
            .value_count
            .checked_add(extra)
            .ok_or(LayoutError::PositionOverflow)?;
        u32::try_from(value_count - 1).map_err(|_| LayoutError::PositionOverflow)?;
        let useful_bits = placement
            .useful_bits
            .checked_add(extra)
            .ok_or(LayoutError::PositionOverflow)?;
        let id = fresh_circuit_id();
        let map_value = |value: ValueId| ValueId {
            circuit: id,
            index: value.index,
        };
        let map_expr = |expr: LinearExprId| LinearExprId {
            circuit: id,
            index: expr.index,
        };
        let mut expressions = source.expressions.clone();
        for expr in &mut expressions {
            match &mut expr.node {
                ExpressionNode::Zero => {}
                ExpressionNode::Value(value) => *value = map_value(*value),
                ExpressionNode::Xor(terms) => {
                    for term in terms {
                        *term = map_expr(*term);
                    }
                }
            }
        }
        let mut columns = source.columns.clone();
        for column in &mut columns {
            column
                .values
                .iter_mut()
                .for_each(|value| *value = map_value(*value));
        }
        let mut interactions = source.interactions.clone();
        for interaction in &mut interactions {
            interaction.selector = map_value(interaction.selector);
            for value in interaction.multiplicity.iter_mut().chain(
                interaction
                    .message
                    .iter_mut()
                    .flat_map(|field| &mut field.values),
            ) {
                *value = map_value(*value);
            }
        }
        let mut lowered = Self {
            id,
            source: source.id,
            source_value_count: source.value_count,
            source_expression_count: source.expressions.len(),
            expressions,
            rows: Vec::with_capacity(source.rows.len() + assertions),
            input_values: source.input_values.iter().copied().map(map_value).collect(),
            one: map_value(source.one),
            columns,
            interactions,
            source_rows: Vec::with_capacity(source.rows.len()),
            auxiliaries: Vec::with_capacity(assertions),
            layout: PhysicalLayout {
                circuit: id,
                value_positions: placement
                    .value_positions
                    .iter()
                    .copied()
                    .chain(placement.useful_bits..useful_bits)
                    .collect(),
                row_positions: Vec::with_capacity(value_count),
                useful_bits,
            },
        };
        let one_expr = map_expr(source.rows[0].result);
        for row in &source.rows {
            let start = lowered.rows.len();
            let lhs = map_expr(row.lhs);
            let rhs = map_expr(row.rhs);
            let result = map_expr(row.result);
            if require_identity && row.kind == RowKind::Constraint {
                let product = ValueId {
                    circuit: id,
                    index: source.value_count + 2 * lowered.auxiliaries.len(),
                };
                let cancellation = ValueId {
                    circuit: id,
                    index: product.index + 1,
                };
                let y = lowered.boundary(product);
                let t = lowered.boundary(cancellation);
                let check = lowered.xor(vec![t, y, result]);
                let product_row = lowered.push_row(RowKind::And, lhs, rhs, y, Some(product));
                let cancellation_row =
                    lowered.push_row(RowKind::Constraint, one_expr, check, t, None);
                lowered.auxiliaries.push(AssertionAux {
                    source_row: row.id,
                    product,
                    cancellation,
                    product_row,
                    cancellation_row,
                });
            } else {
                lowered.push_row(row.kind, lhs, rhs, result, row.defined_value.map(map_value));
            }
            lowered.source_rows.push(start..lowered.rows.len());
        }
        lowered.layout.row_positions = if require_identity {
            lowered
                .rows
                .iter()
                .map(|row| {
                    let support = &lowered.expressions[row.result.index].support;
                    assert_eq!(support.len(), 1, "lowered C side must be one coordinate");
                    lowered.layout.value_positions[support[0].index()]
                })
                .collect()
        } else {
            placement.row_positions.clone()
        };
        lowered.validate();
        lowered.layout.validate_relation(&lowered.relation())?;
        assert!(!require_identity || lowered.c_is_identity());
        Ok(lowered)
    }

    fn boundary(&mut self, value: ValueId) -> LinearExprId {
        let id = LinearExprId {
            circuit: self.id,
            index: self.expressions.len(),
        };
        self.expressions.push(Expression {
            node: ExpressionNode::Value(value),
            support: vec![ValueIndex::new(value.index)],
        });
        id
    }

    fn xor(&mut self, terms: Vec<LinearExprId>) -> LinearExprId {
        let support = terms.iter().fold(Vec::new(), |support, term| {
            symmetric_difference(&support, &self.expressions[term.index].support)
        });
        let id = LinearExprId {
            circuit: self.id,
            index: self.expressions.len(),
        };
        self.expressions.push(Expression {
            node: ExpressionNode::Xor(terms),
            support,
        });
        id
    }

    fn push_row(
        &mut self,
        kind: RowKind,
        lhs: LinearExprId,
        rhs: LinearExprId,
        result: LinearExprId,
        defined_value: Option<ValueId>,
    ) -> RowId {
        let id = RowId {
            circuit: self.id,
            index: self.rows.len(),
        };
        self.rows.push(Row {
            id,
            kind,
            lhs,
            rhs,
            result,
            defined_value,
        });
        id
    }
}
