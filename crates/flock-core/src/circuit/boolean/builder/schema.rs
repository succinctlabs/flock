//! Schema-only construction. Legacy allocating calls are rejected in this mode.

use super::*;
use crate::circuit::boolean::schema::{
    ColumnRole, ColumnSchema, ColumnVisitor, CompiledColumns, SchemaColumn, SchemaState, Var,
};

impl CircuitBuilder {
    pub(super) fn assert_legacy_allocation(&self) {
        assert!(
            self.schema.is_none(),
            "cannot mix schema and legacy allocation"
        );
    }

    pub(super) fn assert_available(&self, expression: LinearExpr) {
        self.assert_circuit(expression.id.circuit);
        if let Some(schema) = &self.schema
            && let ExpressionNode::Value(value) = self.expressions[expression.id.index].node
        {
            assert!(
                schema.defined[value.index],
                "structural read before definition"
            );
        }
        // XOR operands were checked when their structural node was recorded.
    }

    /// Reserve all columns on a fresh builder; schema and legacy allocation cannot mix.
    pub fn reserve_columns<S: ColumnSchema>(&mut self, schema: &S) -> S::Cols<Var> {
        assert!(self.schema.is_none(), "schema already reserved");
        assert!(
            self.value_count == 1 && self.rows.len() == 1 && self.expressions.len() == 2,
            "cannot mix schema and legacy allocation"
        );
        self.schema = Some(SchemaState {
            columns: Vec::new(),
            defined: vec![true],
            operations: Vec::new(),
        });
        schema.columns(&mut ReservingVisitor(self))
    }

    /// Define a reserved column as an unconditional product.
    pub fn define_and(
        &mut self,
        output: Var,
        lhs: impl Into<LinearExpr>,
        rhs: impl Into<LinearExpr>,
    ) {
        self.define_column(output, RowKind::And, lhs.into(), rhs.into());
    }

    /// Explicitly materialize a linear expression into a reserved column.
    pub fn define_linear(&mut self, output: Var, value: impl Into<LinearExpr>) {
        self.define_column(output, RowKind::Materialize, value.into(), self.one.expr());
    }

    fn define_column(&mut self, output: Var, kind: RowKind, lhs: LinearExpr, rhs: LinearExpr) {
        self.assert_circuit(output.0.value.circuit);
        self.assert_available(lhs);
        self.assert_available(rhs);
        let schema = self
            .schema
            .as_mut()
            .expect("definition requires reserved columns");
        assert!(
            !schema.defined[output.0.value.index],
            "column defined twice"
        );
        schema.defined[output.0.value.index] = true;
        self.rows.push(Row {
            id: RowId {
                circuit: self.id,
                index: self.rows.len(),
            },
            kind,
            lhs: lhs.id,
            rhs: rhs.id,
            result: output.0.expression,
            defined_value: Some(output.0.value),
        });
    }

    /// Canonicalize definition order and retain a resolver for construction handles.
    pub fn finish_columns<S: ColumnSchema>(mut self, shape: S) -> CompiledColumns<S> {
        self.validate();
        let schema = self.schema.take().expect("no schema reserved");
        let construction_id = self.id;
        // Reject any construction-era IDs leaked through compatibility metadata APIs.
        let finalized_id = fresh_circuit_id();
        let mut resolved = vec![
            ValueId {
                circuit: finalized_id,
                index: 0
            };
            self.value_count
        ];
        for (index, value) in self
            .rows
            .iter()
            .filter_map(|row| row.defined_value)
            .enumerate()
        {
            resolved[value.index].index = index;
        }
        for expression in &mut self.expressions {
            match &mut expression.node {
                ExpressionNode::Zero => {}
                ExpressionNode::Value(value) => *value = resolved[value.index],
                ExpressionNode::Xor(terms) => {
                    for term in terms {
                        term.circuit = finalized_id;
                    }
                }
            }
        }
        for index in 0..self.expressions.len() {
            let support = match &self.expressions[index].node {
                ExpressionNode::Zero => Vec::new(),
                ExpressionNode::Value(value) => vec![ValueIndex::new(value.index)],
                ExpressionNode::Xor(terms) => terms.iter().fold(Vec::new(), |support, term| {
                    symmetric_difference(&support, &self.expressions[term.index].support)
                }),
            };
            self.expressions[index].support = support;
        }
        for row in &mut self.rows {
            row.id.circuit = finalized_id;
            row.lhs.circuit = finalized_id;
            row.rhs.circuit = finalized_id;
            row.result.circuit = finalized_id;
            row.defined_value = row.defined_value.map(|value| resolved[value.index]);
        }
        for value in &mut self.input_values {
            *value = resolved[value.index];
        }
        for port in &mut self.ports {
            for value in &mut port.values {
                *value = resolved[value.index];
            }
        }
        for interaction in &mut self.interactions {
            interaction.selector = resolved[interaction.selector.index];
            for value in &mut interaction.multiplicity {
                *value = resolved[value.index];
            }
            for field in &mut interaction.message {
                for value in &mut field.values {
                    *value = resolved[value.index];
                }
            }
        }
        self.id = finalized_id;
        self.one.value = resolved[0];
        self.one.expression.circuit = finalized_id;
        self.zero.id.circuit = finalized_id;
        let mut columns = schema.columns;
        for column in &mut columns {
            for value in &mut column.values {
                *value = resolved[value.index];
            }
        }
        let mut operations = schema.operations;
        for operation in &mut operations {
            for value in &mut operation.columns {
                *value = resolved[value.index];
            }
            for word in operation
                .inputs
                .iter_mut()
                .chain(std::iter::once(&mut operation.output))
            {
                for expression in &mut word.expressions {
                    expression.circuit = finalized_id;
                }
            }
        }
        let compiled = CompiledColumns {
            construction_id,
            circuit: self.finish(),
            columns,
            resolved,
            operations,
            schema: shape,
        };
        // Detect a mismatched or witness-dependent schema traversal immediately.
        compiled.columns();
        compiled
    }
}

struct ReservingVisitor<'a>(&'a mut CircuitBuilder);

impl ColumnVisitor for ReservingVisitor<'_> {
    type Value = Var;

    fn bits(
        &mut self,
        name: &str,
        role: ColumnRole,
        width: usize,
        alignment_bits: usize,
    ) -> Vec<Var> {
        let builder = &mut self.0;
        assert!(width > 0, "schema field must not be empty");
        assert!(
            alignment_bits.is_power_of_two(),
            "schema alignment must be a power of two"
        );
        assert!(
            role != ColumnRole::Witness || alignment_bits == 1,
            "internal columns have no port alignment"
        );
        assert!(!name.is_empty(), "schema field name must not be empty");
        assert!(
            builder
                .schema
                .as_ref()
                .unwrap()
                .columns
                .iter()
                .all(|column| column.name != name),
            "duplicate schema field"
        );
        if let ColumnRole::Advice(kind) = &role {
            assert!(!kind.is_empty(), "empty advice type");
        }
        let vars: Vec<Var> = (0..width)
            .map(|_| {
                let value = ValueId {
                    circuit: builder.id,
                    index: builder.value_count,
                };
                builder.value_count += 1;
                let expression = builder.push_value_expression(value);
                builder.schema.as_mut().unwrap().defined.push(false);
                let var = Var(Bit { value, expression });
                if role.is_input() {
                    builder.schema.as_mut().unwrap().defined[value.index] = true;
                    builder.rows.push(Row {
                        id: RowId {
                            circuit: builder.id,
                            index: builder.rows.len(),
                        },
                        kind: RowKind::Input,
                        lhs: expression,
                        rhs: builder.one.expression,
                        result: expression,
                        defined_value: Some(value),
                    });
                    builder.input_values.push(value);
                } else if let ColumnRole::Fixed(bit) = role {
                    builder.define_linear(
                        var,
                        if bit {
                            builder.one.expr()
                        } else {
                            builder.zero
                        },
                    );
                }
                var
            })
            .collect();
        let values: Vec<ValueId> = vars.iter().map(|var| var.0.value).collect();
        let port_role = match &role {
            ColumnRole::Input => Some((PortDirection::Input, PortOrigin::Witness)),
            ColumnRole::Advice(kind) => Some((
                PortDirection::Input,
                PortOrigin::Advice {
                    advice_type: (*kind).into(),
                },
            )),
            ColumnRole::Output => Some((PortDirection::Output, PortOrigin::Derived)),
            ColumnRole::Fixed(_) => Some((PortDirection::Fixed, PortOrigin::Fixed)),
            ColumnRole::Witness => None,
        };
        if let Some((direction, origin)) = port_role {
            builder.ports.push(Port {
                name: name.into(),
                direction,
                origin,
                encoding: PortEncoding::LittleEndianWord { alignment_bits },
                values: values.clone(),
            });
        }
        builder.schema.as_mut().unwrap().columns.push(SchemaColumn {
            name: name.into(),
            role,
            values,
            alignment_bits,
        });
        vars
    }
}
