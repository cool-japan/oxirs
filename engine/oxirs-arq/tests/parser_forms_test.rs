//! Parser coverage for the SPARQL query forms completed in the 0.4.0 wave:
//!
//!   * `DESCRIBE` target retention (IRIs, prefixed names, variables, `*`).
//!   * `CONSTRUCT WHERE { … }` shorthand template population and its
//!     BGP-only restriction.
//!   * Parenthesized SELECT projections `( Expression AS ?var )`, including the
//!     seven SPARQL 1.1 set aggregates.
//!
//! These are pure parser assertions: each case checks the `Query` AST the
//! parser produces, not query execution.

use oxirs_arq::algebra::{Aggregate, Algebra, BinaryOperator, Expression, Term};
use oxirs_arq::query::{DescribeTarget, ProjectionItem, QueryParser, QueryType};

fn parse(query: &str) -> oxirs_arq::query::Query {
    let mut parser = QueryParser::new();
    parser.parse(query).expect("query must parse")
}

fn parse_err(query: &str) -> String {
    let mut parser = QueryParser::new();
    match parser.parse(query) {
        Ok(_) => panic!("expected a parse error for: {query}"),
        Err(e) => e.to_string(),
    }
}

// ── Task 1: DESCRIBE target retention ────────────────────────────────────────

#[test]
fn describe_single_iri_retains_target() {
    let q = parse("DESCRIBE <http://example.org/alice>");
    assert_eq!(q.query_type, QueryType::Describe);
    assert!(!q.describe_all);
    assert_eq!(q.describe_targets.len(), 1);
    match &q.describe_targets[0] {
        DescribeTarget::Iri(iri) => assert_eq!(iri.as_str(), "http://example.org/alice"),
        other => panic!("expected an IRI target, got {other:?}"),
    }
}

#[test]
fn describe_two_iris_retains_both() {
    let q = parse("DESCRIBE <http://example.org/a> <http://example.org/b>");
    assert_eq!(q.describe_targets.len(), 2);
    let iris: Vec<&str> = q
        .describe_targets
        .iter()
        .map(|t| match t {
            DescribeTarget::Iri(iri) => iri.as_str(),
            other => panic!("expected IRI target, got {other:?}"),
        })
        .collect();
    assert_eq!(iris, vec!["http://example.org/a", "http://example.org/b"]);
}

#[test]
fn describe_prefixed_name_is_expanded() {
    let q = parse("PREFIX ex: <http://example.org/> DESCRIBE ex:alice");
    assert_eq!(q.describe_targets.len(), 1);
    match &q.describe_targets[0] {
        DescribeTarget::Iri(iri) => assert_eq!(iri.as_str(), "http://example.org/alice"),
        other => panic!("expected an expanded IRI target, got {other:?}"),
    }
}

#[test]
fn describe_mixed_iri_and_variable() {
    let q = parse(
        "PREFIX ex: <http://example.org/> \
         DESCRIBE ex:alice ?friend WHERE { ex:alice ex:knows ?friend }",
    );
    assert_eq!(q.describe_targets.len(), 2);
    match &q.describe_targets[0] {
        DescribeTarget::Iri(iri) => assert_eq!(iri.as_str(), "http://example.org/alice"),
        other => panic!("expected IRI first, got {other:?}"),
    }
    match &q.describe_targets[1] {
        DescribeTarget::Variable(v) => assert_eq!(v.as_str(), "friend"),
        other => panic!("expected a variable second, got {other:?}"),
    }
    // WHERE clause is retained.
    assert!(matches!(q.where_clause, Algebra::Bgp(_)));
}

#[test]
fn describe_variable_with_where() {
    let q = parse("DESCRIBE ?x WHERE { ?x <http://example.org/p> ?y }");
    assert_eq!(q.describe_targets.len(), 1);
    assert!(matches!(
        &q.describe_targets[0],
        DescribeTarget::Variable(v) if v.as_str() == "x"
    ));
    assert!(matches!(q.where_clause, Algebra::Bgp(_)));
}

#[test]
fn describe_star_sets_flag_and_no_explicit_targets() {
    let q = parse("DESCRIBE *");
    assert!(q.describe_all);
    assert!(q.describe_targets.is_empty());
}

#[test]
fn describe_with_no_target_is_error() {
    let msg = parse_err("DESCRIBE");
    assert!(
        msg.contains("DESCRIBE requires"),
        "unexpected error message: {msg}"
    );
    // Also rejected when a WHERE clause immediately follows with no targets.
    let _ = parse_err("DESCRIBE WHERE { ?s ?p ?o }");
}

// ── Task 2: CONSTRUCT WHERE shorthand ────────────────────────────────────────

#[test]
fn construct_where_shorthand_template_equals_where_bgp() {
    let q = parse("CONSTRUCT WHERE { ?s ?p ?o }");
    assert_eq!(q.query_type, QueryType::Construct);
    let where_triples = match &q.where_clause {
        Algebra::Bgp(triples) => triples.clone(),
        other => panic!("expected a BGP WHERE clause, got {other:?}"),
    };
    assert_eq!(where_triples.len(), 1);
    // The shorthand copies the WHERE BGP verbatim into the template.
    assert_eq!(q.construct_template, where_triples);
}

#[test]
fn construct_where_shorthand_multi_triple() {
    let q = parse("CONSTRUCT WHERE { ?s ?p ?o . ?o ?p2 ?o2 }");
    let where_triples = match &q.where_clause {
        Algebra::Bgp(triples) => triples.clone(),
        other => panic!("expected a BGP WHERE clause, got {other:?}"),
    };
    assert_eq!(where_triples.len(), 2);
    assert_eq!(q.construct_template, where_triples);
}

#[test]
fn construct_where_shorthand_with_filter_is_error() {
    let msg = parse_err("CONSTRUCT WHERE { ?s ?p ?o FILTER(?o > 1) }");
    assert!(
        msg.contains("CONSTRUCT WHERE shorthand"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn construct_explicit_template_unchanged() {
    let q = parse("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }");
    assert_eq!(q.construct_template.len(), 1);
    let t = &q.construct_template[0];
    assert!(matches!(t.subject, Term::Variable(ref v) if v.as_str() == "s"));
    assert!(matches!(t.predicate, Term::Variable(ref v) if v.as_str() == "p"));
    assert!(matches!(t.object, Term::Variable(ref v) if v.as_str() == "o"));
    assert!(matches!(q.where_clause, Algebra::Bgp(_)));
}

#[test]
fn construct_explicit_template_differs_from_where() {
    // The explicit form must NOT overwrite the template from WHERE.
    let q = parse("CONSTRUCT { ?s <http://example.org/type> ?o } WHERE { ?s ?p ?o }");
    assert_eq!(q.construct_template.len(), 1);
    // Template predicate is the fixed IRI/path, not the WHERE's ?p variable.
    assert!(!matches!(
        q.construct_template[0].predicate,
        Term::Variable(_)
    ));
}

// ── Task 3: aggregate / expression projection parsing ────────────────────────

fn single_projection(query: &str) -> ProjectionItem {
    let q = parse(query);
    assert_eq!(
        q.projection_items.len(),
        1,
        "expected exactly one projection item"
    );
    q.projection_items.into_iter().next().expect("one item")
}

#[test]
fn projection_count_star() {
    let item = single_projection("SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }");
    match item {
        ProjectionItem::Aggregate { aggregate, alias } => {
            assert_eq!(alias.as_str(), "n");
            assert_eq!(
                aggregate,
                Aggregate::Count {
                    distinct: false,
                    expr: None
                }
            );
        }
        other => panic!("expected an aggregate projection, got {other:?}"),
    }
}

#[test]
fn projection_count_distinct_var() {
    let item = single_projection("SELECT (COUNT(DISTINCT ?x) AS ?c) WHERE { ?s ?p ?x }");
    match item {
        ProjectionItem::Aggregate { aggregate, alias } => {
            assert_eq!(alias.as_str(), "c");
            match aggregate {
                Aggregate::Count {
                    distinct: true,
                    expr: Some(Expression::Variable(v)),
                } => assert_eq!(v.as_str(), "x"),
                other => panic!("expected COUNT(DISTINCT ?x), got {other:?}"),
            }
        }
        other => panic!("expected an aggregate projection, got {other:?}"),
    }
}

#[test]
fn projection_sum_of_product_expr_shape() {
    let item = single_projection("SELECT (SUM(?a * ?b) AS ?t) WHERE { ?s ?a ?b }");
    match item {
        ProjectionItem::Aggregate { aggregate, alias } => {
            assert_eq!(alias.as_str(), "t");
            match aggregate {
                Aggregate::Sum {
                    distinct: false,
                    expr:
                        Expression::Binary {
                            op: BinaryOperator::Multiply,
                            left,
                            right,
                        },
                } => {
                    assert!(matches!(*left, Expression::Variable(ref v) if v.as_str() == "a"));
                    assert!(matches!(*right, Expression::Variable(ref v) if v.as_str() == "b"));
                }
                other => panic!("expected SUM(?a * ?b), got {other:?}"),
            }
        }
        other => panic!("expected an aggregate projection, got {other:?}"),
    }
}

#[test]
fn projection_avg_of_sum_expr_shape() {
    let item = single_projection("SELECT (AVG(?x + 1) AS ?m) WHERE { ?s ?p ?x }");
    match item {
        ProjectionItem::Aggregate { aggregate, alias } => {
            assert_eq!(alias.as_str(), "m");
            match aggregate {
                Aggregate::Avg {
                    distinct: false,
                    expr:
                        Expression::Binary {
                            op: BinaryOperator::Add,
                            left,
                            right,
                        },
                } => {
                    assert!(matches!(*left, Expression::Variable(ref v) if v.as_str() == "x"));
                    assert!(matches!(*right, Expression::Literal(ref l) if l.value == "1"));
                }
                other => panic!("expected AVG(?x + 1), got {other:?}"),
            }
        }
        other => panic!("expected an aggregate projection, got {other:?}"),
    }
}

#[test]
fn projection_min_max_sample() {
    let min = single_projection("SELECT (MIN(?x) AS ?lo) WHERE { ?s ?p ?x }");
    assert!(matches!(
        min,
        ProjectionItem::Aggregate {
            aggregate: Aggregate::Min {
                distinct: false,
                ..
            },
            ..
        }
    ));
    let max = single_projection("SELECT (MAX(?x) AS ?hi) WHERE { ?s ?p ?x }");
    assert!(matches!(
        max,
        ProjectionItem::Aggregate {
            aggregate: Aggregate::Max {
                distinct: false,
                ..
            },
            ..
        }
    ));
    let sample = single_projection("SELECT (SAMPLE(?x) AS ?any) WHERE { ?s ?p ?x }");
    assert!(matches!(
        sample,
        ProjectionItem::Aggregate {
            aggregate: Aggregate::Sample {
                distinct: false,
                ..
            },
            ..
        }
    ));
}

#[test]
fn projection_group_concat_plain_and_with_separator() {
    let plain = single_projection("SELECT (GROUP_CONCAT(?x) AS ?g) WHERE { ?s ?p ?x }");
    match plain {
        ProjectionItem::Aggregate {
            aggregate:
                Aggregate::GroupConcat {
                    distinct: false,
                    separator,
                    ..
                },
            ..
        } => assert!(separator.is_none()),
        other => panic!("expected GROUP_CONCAT, got {other:?}"),
    }

    let with_sep = single_projection(
        "SELECT (GROUP_CONCAT(?x ; SEPARATOR = \", \") AS ?g) WHERE { ?s ?p ?x }",
    );
    match with_sep {
        ProjectionItem::Aggregate {
            aggregate:
                Aggregate::GroupConcat {
                    distinct: false,
                    separator: Some(sep),
                    ..
                },
            ..
        } => assert_eq!(sep, ", "),
        other => panic!("expected GROUP_CONCAT with separator, got {other:?}"),
    }
}

#[test]
fn projection_non_aggregate_expression() {
    let item = single_projection("SELECT (?a + 1 AS ?b) WHERE { ?s ?p ?a }");
    match item {
        ProjectionItem::Expression { expr, alias } => {
            assert_eq!(alias.as_str(), "b");
            assert!(matches!(
                expr,
                Expression::Binary {
                    op: BinaryOperator::Add,
                    ..
                }
            ));
        }
        other => panic!("expected a plain expression projection, got {other:?}"),
    }
}

#[test]
fn projection_mixed_var_and_aggregate_with_group_by_and_having() {
    let q =
        parse("SELECT ?g (COUNT(*) AS ?n) WHERE { ?s ?p ?g } GROUP BY ?g HAVING (COUNT(?s) > 1)");
    // Two projection items: the plain grouping variable then the aggregate.
    assert_eq!(q.projection_items.len(), 2);
    assert!(matches!(
        &q.projection_items[0],
        ProjectionItem::Variable(v) if v.as_str() == "g"
    ));
    match &q.projection_items[1] {
        ProjectionItem::Aggregate { aggregate, alias } => {
            assert_eq!(alias.as_str(), "n");
            assert_eq!(
                *aggregate,
                Aggregate::Count {
                    distinct: false,
                    expr: None
                }
            );
        }
        other => panic!("expected an aggregate second item, got {other:?}"),
    }
    // Output column set carries both.
    let names: Vec<&str> = q.select_variables.iter().map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["g", "n"]);
    // GROUP BY and HAVING are populated.
    assert_eq!(q.group_by.len(), 1, "GROUP BY ?g must be captured");
    assert!(matches!(
        &q.group_by[0].expr,
        Expression::Variable(v) if v.as_str() == "g"
    ));
    assert!(q.having.is_some(), "HAVING must be captured");
}

#[test]
fn projection_paren_item_without_as_is_error() {
    let msg = parse_err("SELECT (?x + 1) WHERE { ?s ?p ?x }");
    assert!(
        msg.contains("AS ?var") || msg.to_lowercase().contains("as"),
        "unexpected error message: {msg}"
    );
}

// ── HAVING aggregate arity validated at parse time ───────────────────────────

#[test]
fn having_zero_arg_sum_is_parse_error() {
    // `SUM()` parses via the generic HAVING expression grammar but is a
    // wrong-arity aggregate; it must be rejected at parse time (a 400), not left
    // to blow up at execution (a 500).
    let msg =
        parse_err("SELECT ?g (COUNT(*) AS ?n) WHERE { ?s ?p ?g } GROUP BY ?g HAVING (SUM() > 1)");
    assert!(
        msg.contains("SUM in HAVING expects exactly one argument"),
        "zero-arg SUM in HAVING must be a parse error, got: {msg}"
    );
}

#[test]
fn having_two_arg_count_is_parse_error() {
    let msg = parse_err(
        "SELECT ?g (COUNT(*) AS ?n) WHERE { ?s ?p ?g } GROUP BY ?g HAVING (COUNT(?s, ?p) > 1)",
    );
    assert!(
        msg.contains("COUNT in HAVING expects at most one argument"),
        "two-arg COUNT in HAVING must be a parse error, got: {msg}"
    );
}

#[test]
fn having_correct_arity_aggregate_parses() {
    // Correct arity must still parse cleanly through the generic HAVING grammar.
    let q = parse("SELECT ?g (COUNT(*) AS ?n) WHERE { ?s ?p ?g } GROUP BY ?g HAVING (SUM(?g) > 1)");
    assert!(q.having.is_some(), "correct-arity HAVING must be captured");
}

// ── Regression: plain projections unchanged ──────────────────────────────────

#[test]
fn plain_select_vars_unchanged() {
    let q = parse("SELECT ?x ?y WHERE { ?x ?p ?y }");
    let names: Vec<&str> = q.select_variables.iter().map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["x", "y"]);
    assert_eq!(q.projection_items.len(), 2);
    assert!(matches!(
        &q.projection_items[0],
        ProjectionItem::Variable(v) if v.as_str() == "x"
    ));
    assert!(matches!(
        &q.projection_items[1],
        ProjectionItem::Variable(v) if v.as_str() == "y"
    ));
}

#[test]
fn select_star_leaves_projection_empty() {
    let q = parse("SELECT * WHERE { ?s ?p ?o }");
    assert!(q.select_variables.is_empty());
    assert!(q.projection_items.is_empty());
    assert!(matches!(q.where_clause, Algebra::Bgp(_)));
}
