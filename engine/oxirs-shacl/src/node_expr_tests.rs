//! Tests for SHACL node expression evaluation.
//!
//! Split out from `node_expressions.rs` to keep individual source files below
//! the workspace 2000-line refactor threshold.

#![cfg(test)]

use crate::node_expr_evaluator::NodeExprEvaluator;
use crate::node_expr_types::{
    EvalConfig, ExprBuilder, InMemoryGraph, NodeExprError, NodeExpression, NodeTerm, PropertyPath,
};

fn sample_graph() -> InMemoryGraph {
    let mut g = InMemoryGraph::new();
    // Alice knows Bob, Charlie; Bob knows Diana
    g.add_triple(
        NodeTerm::iri("http://ex.org/alice"),
        NodeTerm::iri("http://ex.org/knows"),
        NodeTerm::iri("http://ex.org/bob"),
    );
    g.add_triple(
        NodeTerm::iri("http://ex.org/alice"),
        NodeTerm::iri("http://ex.org/knows"),
        NodeTerm::iri("http://ex.org/charlie"),
    );
    g.add_triple(
        NodeTerm::iri("http://ex.org/bob"),
        NodeTerm::iri("http://ex.org/knows"),
        NodeTerm::iri("http://ex.org/diana"),
    );
    // Names
    g.add_triple(
        NodeTerm::iri("http://ex.org/alice"),
        NodeTerm::iri("http://ex.org/name"),
        NodeTerm::literal("Alice"),
    );
    g.add_triple(
        NodeTerm::iri("http://ex.org/bob"),
        NodeTerm::iri("http://ex.org/name"),
        NodeTerm::literal("Bob"),
    );
    g.add_triple(
        NodeTerm::iri("http://ex.org/charlie"),
        NodeTerm::iri("http://ex.org/name"),
        NodeTerm::literal("Charlie"),
    );
    g.add_triple(
        NodeTerm::iri("http://ex.org/diana"),
        NodeTerm::iri("http://ex.org/name"),
        NodeTerm::literal("Diana"),
    );
    // Ages
    g.add_triple(
        NodeTerm::iri("http://ex.org/alice"),
        NodeTerm::iri("http://ex.org/age"),
        NodeTerm::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
    );
    g.add_triple(
        NodeTerm::iri("http://ex.org/bob"),
        NodeTerm::iri("http://ex.org/age"),
        NodeTerm::typed_literal("25", "http://www.w3.org/2001/XMLSchema#integer"),
    );
    g
}

fn alice() -> NodeTerm {
    NodeTerm::iri("http://ex.org/alice")
}

fn bob() -> NodeTerm {
    NodeTerm::iri("http://ex.org/bob")
}

// ── FocusNode ─────────────────────────────────────────────────────────

#[test]
fn test_focus_node() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let result = eval
        .evaluate(&NodeExpression::FocusNode, &alice(), &g)
        .expect("eval");
    assert_eq!(result, vec![alice()]);
}

// ── Constant ──────────────────────────────────────────────────────────

#[test]
fn test_constant() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let c = NodeTerm::literal("hello");
    let result = eval
        .evaluate(&NodeExpression::Constant(c.clone()), &alice(), &g)
        .expect("eval");
    assert_eq!(result, vec![c]);
}

// ── Path ──────────────────────────────────────────────────────────────

#[test]
fn test_path_simple_predicate() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = ExprBuilder::path("http://ex.org/knows");
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 2);
    assert!(result.contains(&bob()));
    assert!(result.contains(&NodeTerm::iri("http://ex.org/charlie")));
}

#[test]
fn test_path_no_results() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = ExprBuilder::path("http://ex.org/nonexistent");
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert!(result.is_empty());
}

#[test]
fn test_path_inverse() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::Path(PropertyPath::Inverse(Box::new(PropertyPath::Predicate(
        "http://ex.org/knows".to_string(),
    ))));
    let result = eval.evaluate(&expr, &bob(), &g).expect("eval");
    assert_eq!(result, vec![alice()]);
}

#[test]
fn test_path_sequence() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    // Alice -> knows -> knows (should reach Diana via Bob)
    let expr = ExprBuilder::sequence_path(&["http://ex.org/knows", "http://ex.org/knows"]);
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert!(result.contains(&NodeTerm::iri("http://ex.org/diana")));
}

#[test]
fn test_path_alternative() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::Path(PropertyPath::Alternative(vec![
        PropertyPath::Predicate("http://ex.org/knows".to_string()),
        PropertyPath::Predicate("http://ex.org/name".to_string()),
    ]));
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    // Should find bob, charlie (from knows) and "Alice" (from name)
    assert!(result.len() >= 3);
}

#[test]
fn test_path_zero_or_more() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::Path(PropertyPath::ZeroOrMore(Box::new(PropertyPath::Predicate(
        "http://ex.org/knows".to_string(),
    ))));
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    // alice (start) + bob + charlie + diana (via bob)
    assert!(result.contains(&alice())); // zero steps
    assert!(result.contains(&bob()));
    assert!(result.contains(&NodeTerm::iri("http://ex.org/diana")));
}

#[test]
fn test_path_one_or_more() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::Path(PropertyPath::OneOrMore(Box::new(PropertyPath::Predicate(
        "http://ex.org/knows".to_string(),
    ))));
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    // Should NOT include alice (requires at least one step)
    assert!(!result.contains(&alice()));
    assert!(result.contains(&bob()));
    assert!(result.contains(&NodeTerm::iri("http://ex.org/diana")));
}

#[test]
fn test_path_zero_or_one() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::Path(PropertyPath::ZeroOrOne(Box::new(PropertyPath::Predicate(
        "http://ex.org/knows".to_string(),
    ))));
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    // alice (zero) + bob + charlie (one)
    assert!(result.contains(&alice()));
    assert!(result.contains(&bob()));
}

// ── Intersection ──────────────────────────────────────────────────────

#[test]
fn test_intersection() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = ExprBuilder::intersection(vec![
        ExprBuilder::path("http://ex.org/knows"),
        NodeExpression::Union(vec![
            NodeExpression::Constant(bob()),
            NodeExpression::Constant(NodeTerm::iri("http://ex.org/unknown")),
        ]),
    ]);
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result, vec![bob()]);
}

#[test]
fn test_intersection_empty() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::Intersection(vec![]);
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert!(result.is_empty());
}

// ── Union ─────────────────────────────────────────────────────────────

#[test]
fn test_union() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = ExprBuilder::union(vec![
        ExprBuilder::path("http://ex.org/knows"),
        ExprBuilder::path("http://ex.org/name"),
    ]);
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert!(result.len() >= 3); // bob, charlie, "Alice"
}

// ── Minus ─────────────────────────────────────────────────────────────

#[test]
fn test_minus() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    // Alice knows bob and charlie, minus bob
    let expr = ExprBuilder::minus(
        ExprBuilder::path("http://ex.org/knows"),
        NodeExpression::Constant(bob()),
    );
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], NodeTerm::iri("http://ex.org/charlie"));
}

// ── IfThenElse ────────────────────────────────────────────────────────

#[test]
fn test_if_then_else_true() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    // Alice has knows relations -> condition is non-empty -> then
    let expr = ExprBuilder::if_then_else(
        ExprBuilder::path("http://ex.org/knows"),
        NodeExpression::Constant(NodeTerm::literal("has friends")),
        NodeExpression::Constant(NodeTerm::literal("no friends")),
    );
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result[0], NodeTerm::literal("has friends"));
}

#[test]
fn test_if_then_else_false() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    // Diana has no knows relations -> condition is empty -> else
    let expr = ExprBuilder::if_then_else(
        ExprBuilder::path("http://ex.org/knows"),
        NodeExpression::Constant(NodeTerm::literal("has friends")),
        NodeExpression::Constant(NodeTerm::literal("no friends")),
    );
    let diana = NodeTerm::iri("http://ex.org/diana");
    let result = eval.evaluate(&expr, &diana, &g).expect("eval");
    assert_eq!(result[0], NodeTerm::literal("no friends"));
}

// ── FilterShape ───────────────────────────────────────────────────────

#[test]
fn test_filter_shape() {
    let mut g = sample_graph();
    g.set_conforms("http://ex.org/bob", "http://ex.org/AdultShape", true);
    g.set_conforms("http://ex.org/charlie", "http://ex.org/AdultShape", false);

    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::FilterShape {
        nodes: Box::new(ExprBuilder::path("http://ex.org/knows")),
        shape: "http://ex.org/AdultShape".to_string(),
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result, vec![bob()]);
}

// ── Count ─────────────────────────────────────────────────────────────

#[test]
fn test_count() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = ExprBuilder::count(ExprBuilder::path("http://ex.org/knows"));
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].as_integer(), Some(2));
}

#[test]
fn test_count_zero() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let diana = NodeTerm::iri("http://ex.org/diana");
    let expr = ExprBuilder::count(ExprBuilder::path("http://ex.org/knows"));
    let result = eval.evaluate(&expr, &diana, &g).expect("eval");
    assert_eq!(result[0].as_integer(), Some(0));
}

// ── Distinct ──────────────────────────────────────────────────────────

#[test]
fn test_distinct() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    // Union of same expression twice -> distinct should deduplicate
    let expr = ExprBuilder::distinct(ExprBuilder::union(vec![
        ExprBuilder::path("http://ex.org/knows"),
        ExprBuilder::path("http://ex.org/knows"),
    ]));
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 2); // bob and charlie, deduplicated
}

// ── Limit/Offset ─────────────────────────────────────────────────────

#[test]
fn test_limit() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = ExprBuilder::limit(ExprBuilder::path("http://ex.org/knows"), 1);
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_offset() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = ExprBuilder::offset(ExprBuilder::path("http://ex.org/knows"), 1);
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 1); // 2 results minus 1 offset
}

#[test]
fn test_limit_and_offset() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    // Get knows results, skip 0, take 1
    let expr = ExprBuilder::limit(
        ExprBuilder::offset(ExprBuilder::path("http://ex.org/knows"), 0),
        1,
    );
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 1);
}

// ── OrderBy ───────────────────────────────────────────────────────────

#[test]
fn test_order_by_ascending() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::OrderBy {
        expr: Box::new(ExprBuilder::path("http://ex.org/knows")),
        sort_path: PropertyPath::Predicate("http://ex.org/name".to_string()),
        ascending: true,
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    // Bob and Charlie, sorted by name ascending
    assert_eq!(result.len(), 2);
    // "Bob" < "Charlie" alphabetically
    assert_eq!(result[0], bob());
}

#[test]
fn test_order_by_descending() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::OrderBy {
        expr: Box::new(ExprBuilder::path("http://ex.org/knows")),
        sort_path: PropertyPath::Predicate("http://ex.org/name".to_string()),
        ascending: false,
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result[0], NodeTerm::iri("http://ex.org/charlie"));
}

// ── GroupConcat ───────────────────────────────────────────────────────

#[test]
fn test_group_concat() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::GroupConcat {
        expr: Box::new(ExprBuilder::path("http://ex.org/name")),
        separator: ", ".to_string(),
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].as_str(), "Alice");
}

// ── Function calls ────────────────────────────────────────────────────

#[test]
fn test_function_strlen() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::FunctionCall {
        function: "sh:strlen".to_string(),
        args: vec![ExprBuilder::path("http://ex.org/name")],
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    // "Alice" -> strlen = 5
    assert_eq!(result[0].as_integer(), Some(5));
}

#[test]
fn test_function_concat() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::FunctionCall {
        function: "sh:concat".to_string(),
        args: vec![
            ExprBuilder::path("http://ex.org/name"),
            NodeExpression::Constant(NodeTerm::literal("!")),
        ],
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result[0].as_str(), "Alice!");
}

#[test]
fn test_function_unknown() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::FunctionCall {
        function: "sh:unknown_fn".to_string(),
        args: vec![],
    };
    let result = eval.evaluate(&expr, &alice(), &g);
    assert!(matches!(result, Err(NodeExprError::UnknownFunction(_))));
}

#[test]
fn test_function_wrong_arg_count() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::FunctionCall {
        function: "sh:strlen".to_string(),
        args: vec![
            NodeExpression::Constant(NodeTerm::literal("a")),
            NodeExpression::Constant(NodeTerm::literal("b")),
        ],
    };
    let result = eval.evaluate(&expr, &alice(), &g);
    assert!(matches!(result, Err(NodeExprError::InvalidArgCount { .. })));
}

// ── Custom function registration ──────────────────────────────────────

#[test]
fn test_register_custom_function() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();

    fn my_upper(args: &[Vec<NodeTerm>]) -> Result<Vec<NodeTerm>, NodeExprError> {
        Ok(args[0]
            .iter()
            .map(|t| NodeTerm::literal(t.as_str().to_uppercase()))
            .collect())
    }

    eval.register_function("ex:upper", 1, my_upper);
    let expr = NodeExpression::FunctionCall {
        function: "ex:upper".to_string(),
        args: vec![ExprBuilder::path("http://ex.org/name")],
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result[0].as_str(), "ALICE");
}

// ── NodeTerm tests ────────────────────────────────────────────────────

#[test]
fn test_node_term_is_iri() {
    assert!(NodeTerm::iri("http://ex.org/x").is_iri());
    assert!(!NodeTerm::literal("hello").is_iri());
}

#[test]
fn test_node_term_is_literal() {
    assert!(NodeTerm::literal("hello").is_literal());
    assert!(!NodeTerm::iri("http://ex.org/x").is_literal());
}

#[test]
fn test_node_term_as_integer() {
    assert_eq!(
        NodeTerm::typed_literal("42", "xsd:int").as_integer(),
        Some(42)
    );
    assert_eq!(NodeTerm::literal("not_a_number").as_integer(), None);
}

#[test]
fn test_node_term_as_float() {
    assert!(
        (NodeTerm::typed_literal("3.125", "xsd:double")
            .as_float()
            .expect("ok")
            - 3.125)
            .abs()
            < 0.001
    );
}

#[test]
fn test_node_term_display() {
    assert_eq!(
        format!("{}", NodeTerm::iri("http://ex.org/x")),
        "<http://ex.org/x>"
    );
    assert_eq!(format!("{}", NodeTerm::literal("hello")), "\"hello\"");
    assert_eq!(
        format!("{}", NodeTerm::typed_literal("42", "xsd:int")),
        "\"42\"^^<xsd:int>"
    );
    assert_eq!(
        format!("{}", NodeTerm::lang_literal("hello", "en")),
        "\"hello\"@en"
    );
    assert_eq!(format!("{}", NodeTerm::BlankNode("b0".to_string())), "_:b0");
}

// ── PropertyPath display ──────────────────────────────────────────────

#[test]
fn test_property_path_display() {
    let p = PropertyPath::Predicate("http://ex.org/p".to_string());
    assert_eq!(format!("{p}"), "<http://ex.org/p>");

    let inv = PropertyPath::Inverse(Box::new(p.clone()));
    assert_eq!(format!("{inv}"), "^<http://ex.org/p>");

    let seq = PropertyPath::Sequence(vec![p.clone(), p.clone()]);
    assert!(format!("{seq}").contains(" / "));

    let alt = PropertyPath::Alternative(vec![p.clone(), p.clone()]);
    assert!(format!("{alt}").contains(" | "));

    let star = PropertyPath::ZeroOrMore(Box::new(p.clone()));
    assert!(format!("{star}").ends_with('*'));

    let plus = PropertyPath::OneOrMore(Box::new(p.clone()));
    assert!(format!("{plus}").ends_with('+'));

    let opt = PropertyPath::ZeroOrOne(Box::new(p));
    assert!(format!("{opt}").ends_with('?'));
}

// ── Error display ─────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    let e = NodeExprError::PathDepthExceeded {
        depth: 101,
        max: 100,
    };
    assert!(format!("{e}").contains("101"));

    let e = NodeExprError::ResultLimitExceeded {
        count: 10001,
        max: 10000,
    };
    assert!(format!("{e}").contains("10001"));

    let e = NodeExprError::UnknownFunction("foo".to_string());
    assert!(format!("{e}").contains("foo"));

    let e = NodeExprError::InvalidArgCount {
        function: "bar".to_string(),
        expected: 2,
        got: 1,
    };
    assert!(format!("{e}").contains("bar"));

    let e = NodeExprError::TypeError("bad type".to_string());
    assert!(format!("{e}").contains("bad type"));

    let e = NodeExprError::EmptySequence("test".to_string());
    assert!(format!("{e}").contains("test"));
}

// ── InMemoryGraph ─────────────────────────────────────────────────────

#[test]
fn test_in_memory_graph_len() {
    let g = sample_graph();
    assert_eq!(g.len(), 9);
    assert!(!g.is_empty());
}

#[test]
fn test_in_memory_graph_empty() {
    let g = InMemoryGraph::new();
    assert_eq!(g.len(), 0);
    assert!(g.is_empty());
}

#[test]
fn test_in_memory_graph_all_triples() {
    use crate::node_expr_types::NodeExprGraph;
    let g = sample_graph();
    assert_eq!(g.all_triples().len(), 9);
}

// ── EvalConfig ────────────────────────────────────────────────────────

#[test]
fn test_eval_config_default() {
    let config = EvalConfig::default();
    assert_eq!(config.max_path_depth, 100);
    assert_eq!(config.max_results, 10000);
    assert!(!config.deduplicate);
}

// ── Stats ─────────────────────────────────────────────────────────────

#[test]
fn test_stats_tracking() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let _ = eval.evaluate(&ExprBuilder::path("http://ex.org/knows"), &alice(), &g);
    assert!(eval.stats().path_traversals > 0);
    assert!(eval.stats().nodes_evaluated > 0);
}

#[test]
fn test_stats_reset() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let _ = eval.evaluate(&ExprBuilder::path("http://ex.org/knows"), &alice(), &g);
    eval.reset_stats();
    assert_eq!(eval.stats().path_traversals, 0);
}

// ── ExprBuilder ───────────────────────────────────────────────────────

#[test]
fn test_expr_builder_focus() {
    assert_eq!(ExprBuilder::focus(), NodeExpression::FocusNode);
}

#[test]
fn test_expr_builder_constant() {
    let c = NodeTerm::literal("x");
    assert_eq!(
        ExprBuilder::constant(c.clone()),
        NodeExpression::Constant(c)
    );
}

#[test]
fn test_expr_builder_path() {
    let expr = ExprBuilder::path("http://ex.org/p");
    assert!(matches!(
        expr,
        NodeExpression::Path(PropertyPath::Predicate(_))
    ));
}

#[test]
fn test_expr_builder_sequence_path() {
    let expr = ExprBuilder::sequence_path(&["http://ex.org/p1", "http://ex.org/p2"]);
    assert!(matches!(
        expr,
        NodeExpression::Path(PropertyPath::Sequence(_))
    ));
}

// ── Depth limit ───────────────────────────────────────────────────────

#[test]
fn test_max_path_depth_exceeded() {
    let g = sample_graph();
    let config = EvalConfig {
        max_path_depth: 1,
        ..Default::default()
    };
    let mut eval = NodeExprEvaluator::with_config(config);
    // Deep nesting: count(count(count(...)))
    let expr = ExprBuilder::count(ExprBuilder::count(ExprBuilder::path("http://ex.org/knows")));
    let result = eval.evaluate(&expr, &alice(), &g);
    assert!(matches!(
        result,
        Err(NodeExprError::PathDepthExceeded { .. })
    ));
}

// ── Function: sum ─────────────────────────────────────────────────────

#[test]
fn test_function_sum() {
    let g = sample_graph();
    let mut eval = NodeExprEvaluator::new();
    let expr = NodeExpression::FunctionCall {
        function: "sh:sum".to_string(),
        args: vec![ExprBuilder::path("http://ex.org/age")],
    };
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    let val = result[0].as_float().expect("should be float");
    assert!((val - 30.0).abs() < 0.01);
}

// ── Complex expression ────────────────────────────────────────────────

#[test]
fn test_complex_expression() {
    let mut g = sample_graph();
    g.set_conforms("http://ex.org/bob", "http://ex.org/FriendShape", true);
    g.set_conforms("http://ex.org/charlie", "http://ex.org/FriendShape", true);

    let mut eval = NodeExprEvaluator::new();
    // Count friends that conform to FriendShape
    let expr = ExprBuilder::count(NodeExpression::FilterShape {
        nodes: Box::new(ExprBuilder::path("http://ex.org/knows")),
        shape: "http://ex.org/FriendShape".to_string(),
    });
    let result = eval.evaluate(&expr, &alice(), &g).expect("eval");
    assert_eq!(result[0].as_integer(), Some(2));
}
