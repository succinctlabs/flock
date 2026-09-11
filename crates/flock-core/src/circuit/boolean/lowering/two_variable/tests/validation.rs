use super::*;

fn fixture() -> LoweredCircuit {
    let mut b = CircuitBuilder::new();
    let [x] = b.input_bits("input");
    b.assert_zero(x);
    b.materialize(x);
    let source = b.finish();
    source
        .lower_identity_c(&source.layout().finish().unwrap())
        .unwrap()
}

#[test]
#[should_panic]
fn cancellation_cannot_be_presented_as_a_definition() {
    let mut lowered = fixture();
    let aux = &lowered.auxiliaries[0];
    lowered.rows[aux.cancellation_row.index].defined_value = Some(aux.cancellation);
    lowered.validate();
}

#[test]
#[should_panic(expected = "structural read before definition")]
fn canceled_terms_cannot_hide_a_future_definition() {
    let mut lowered = fixture();
    let future = lowered.rows.last().unwrap().result;
    let expr = LinearExprId {
        circuit: lowered.id,
        index: lowered.expressions.len(),
    };
    lowered.expressions.push(Expression {
        node: ExpressionNode::Xor(vec![future, future]),
        support: Vec::new(),
    });
    let check = lowered.auxiliaries[0].cancellation_row.index;
    lowered.rows[check].rhs = expr;
    lowered.validate();
}

#[test]
#[should_panic(expected = "expression DAG must be acyclic")]
fn canceled_terms_cannot_hide_an_expression_cycle() {
    let mut lowered = fixture();
    let check = lowered.auxiliaries[0].cancellation_row.index;
    let expr = lowered.rows[check].rhs;
    lowered.expressions[expr.index] = Expression {
        node: ExpressionNode::Xor(vec![expr, expr]),
        support: Vec::new(),
    };
    lowered.validate();
}
