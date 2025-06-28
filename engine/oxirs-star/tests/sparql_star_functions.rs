//! Integration tests for SPARQL-star built-in functions

use oxirs_star::{
    functions::{Expression, ExpressionEvaluator, FunctionEvaluator, StarFunction},
    query::{BasicGraphPattern, QueryExecutor, TermPattern, TriplePattern},
    StarGraph, StarStore, StarTerm, StarTriple,
};
use std::collections::HashMap;

#[test]
fn test_triple_function_integration() {
    // Test TRIPLE(s, p, o) function in a real query context
    let store = StarStore::new();

    // Add base facts
    let fact1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    let fact2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    store.insert(&fact1).unwrap();
    store.insert(&fact2).unwrap();

    // Create metadata about these facts
    let meta1 = StarTriple::new(
        StarTerm::quoted_triple(fact1.clone()),
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::literal("census2020").unwrap(),
    );

    store.insert(&meta1).unwrap();

    let mut executor = QueryExecutor::new(store);

    // Query: Find facts about Alice from census2020
    let mut bgp = BasicGraphPattern::new();
    bgp.add_pattern(TriplePattern::new(
        TermPattern::Variable("fact".to_string()),
        TermPattern::Term(StarTerm::iri("http://example.org/source").unwrap()),
        TermPattern::Term(StarTerm::literal("census2020").unwrap()),
    ));

    // Add filter to check if the fact is about Alice
    bgp.add_filter(Expression::equal(
        Expression::subject(Expression::var("fact")),
        Expression::term(StarTerm::iri("http://example.org/alice").unwrap()),
    ));

    let bindings = executor.execute_bgp(&bgp).unwrap();
    assert_eq!(bindings.len(), 1);
}

#[test]
fn test_subject_predicate_object_functions() {
    let store = StarStore::new();

    // Create a quoted triple
    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/paris").unwrap(),
        StarTerm::iri("http://example.org/capitalOf").unwrap(),
        StarTerm::iri("http://example.org/france").unwrap(),
    );

    let quoted = StarTriple::new(
        StarTerm::quoted_triple(triple.clone()),
        StarTerm::iri("http://example.org/since").unwrap(),
        StarTerm::literal("987").unwrap(),
    );

    store.insert(&quoted).unwrap();

    // Test extracting components of quoted triples
    let mut bindings = HashMap::new();
    let quoted_term = StarTerm::quoted_triple(triple);
    bindings.insert("fact".to_string(), quoted_term.clone());

    // Test SUBJECT function
    let subject_expr = Expression::subject(Expression::var("fact"));
    let subject_result = ExpressionEvaluator::evaluate(&subject_expr, &bindings).unwrap();
    assert_eq!(
        subject_result,
        StarTerm::iri("http://example.org/paris").unwrap()
    );

    // Test PREDICATE function
    let predicate_expr = Expression::predicate(Expression::var("fact"));
    let predicate_result = ExpressionEvaluator::evaluate(&predicate_expr, &bindings).unwrap();
    assert_eq!(
        predicate_result,
        StarTerm::iri("http://example.org/capitalOf").unwrap()
    );

    // Test OBJECT function
    let object_expr = Expression::object(Expression::var("fact"));
    let object_result = ExpressionEvaluator::evaluate(&object_expr, &bindings).unwrap();
    assert_eq!(
        object_result,
        StarTerm::iri("http://example.org/france").unwrap()
    );
}

#[test]
fn test_is_triple_function() {
    let mut bindings = HashMap::new();

    // Test with a quoted triple
    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/s").unwrap(),
        StarTerm::iri("http://example.org/p").unwrap(),
        StarTerm::iri("http://example.org/o").unwrap(),
    );
    let quoted = StarTerm::quoted_triple(triple);
    bindings.insert("x".to_string(), quoted);

    let expr = Expression::is_triple(Expression::var("x"));
    let result = ExpressionEvaluator::evaluate(&expr, &bindings).unwrap();

    assert_eq!(
        result,
        StarTerm::literal_with_datatype("true", "http://www.w3.org/2001/XMLSchema#boolean")
            .unwrap()
    );

    // Test with a non-triple
    bindings.insert(
        "y".to_string(),
        StarTerm::iri("http://example.org/not_a_triple").unwrap(),
    );

    let expr2 = Expression::is_triple(Expression::var("y"));
    let result2 = ExpressionEvaluator::evaluate(&expr2, &bindings).unwrap();

    assert_eq!(
        result2,
        StarTerm::literal_with_datatype("false", "http://www.w3.org/2001/XMLSchema#boolean")
            .unwrap()
    );
}

#[test]
fn test_nested_quoted_triple_functions() {
    // Test functions on nested quoted triples
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/knows").unwrap(),
        StarTerm::iri("http://example.org/bob").unwrap(),
    );

    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );

    let double_nested = StarTriple::new(
        StarTerm::quoted_triple(outer.clone()),
        StarTerm::iri("http://example.org/assertedBy").unwrap(),
        StarTerm::iri("http://example.org/system").unwrap(),
    );

    // Test extracting from nested structure
    let double_quoted = StarTerm::quoted_triple(double_nested);

    // SUBJECT of outer quoted triple should be the inner quoted triple
    let subject =
        FunctionEvaluator::evaluate(StarFunction::Subject, &[double_quoted.clone()]).unwrap();

    assert!(subject.is_quoted_triple());

    // SUBJECT of SUBJECT should give us the innermost quoted triple
    let inner_subject = FunctionEvaluator::evaluate(StarFunction::Subject, &[subject]).unwrap();

    assert!(inner_subject.is_quoted_triple());

    // SUBJECT of that should finally give us Alice
    let alice = FunctionEvaluator::evaluate(StarFunction::Subject, &[inner_subject]).unwrap();

    assert_eq!(alice, StarTerm::iri("http://example.org/alice").unwrap());
}

#[test]
fn test_triple_construction_and_comparison() {
    let mut bindings = HashMap::new();
    bindings.insert(
        "s".to_string(),
        StarTerm::iri("http://example.org/subject").unwrap(),
    );
    bindings.insert(
        "p".to_string(),
        StarTerm::iri("http://example.org/predicate").unwrap(),
    );
    bindings.insert("o".to_string(), StarTerm::literal("object").unwrap());

    // Construct a triple using TRIPLE function
    let construct_expr = Expression::triple(
        Expression::var("s"),
        Expression::var("p"),
        Expression::var("o"),
    );

    let constructed = ExpressionEvaluator::evaluate(&construct_expr, &bindings).unwrap();
    assert!(constructed.is_quoted_triple());

    // Create the same triple manually
    let manual_triple = StarTriple::new(
        StarTerm::iri("http://example.org/subject").unwrap(),
        StarTerm::iri("http://example.org/predicate").unwrap(),
        StarTerm::literal("object").unwrap(),
    );
    let manual_quoted = StarTerm::quoted_triple(manual_triple);

    // They should be equal
    assert_eq!(constructed, manual_quoted);
}

#[test]
fn test_complex_filter_with_functions() {
    let store = StarStore::new();

    // Add various types of statements
    let fact1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    let fact2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    // Add metadata about these facts
    let meta1 = StarTriple::new(
        StarTerm::quoted_triple(fact1.clone()),
        StarTerm::iri("http://example.org/confidence").unwrap(),
        StarTerm::literal("0.95").unwrap(),
    );

    let meta2 = StarTriple::new(
        StarTerm::quoted_triple(fact2.clone()),
        StarTerm::iri("http://example.org/confidence").unwrap(),
        StarTerm::literal("0.80").unwrap(),
    );

    // Also add a non-quoted statement
    let regular = StarTriple::new(
        StarTerm::iri("http://example.org/charlie").unwrap(),
        StarTerm::iri("http://example.org/confidence").unwrap(),
        StarTerm::literal("0.90").unwrap(),
    );

    store.insert(&fact1).unwrap();
    store.insert(&fact2).unwrap();
    store.insert(&meta1).unwrap();
    store.insert(&meta2).unwrap();
    store.insert(&regular).unwrap();

    let mut executor = QueryExecutor::new(store);

    // Query: Find all confidence statements where the subject is a quoted triple
    let mut bgp = BasicGraphPattern::new();
    bgp.add_pattern(TriplePattern::new(
        TermPattern::Variable("statement".to_string()),
        TermPattern::Term(StarTerm::iri("http://example.org/confidence").unwrap()),
        TermPattern::Variable("conf".to_string()),
    ));

    // Filter: isTRIPLE(?statement)
    bgp.add_filter(Expression::is_triple(Expression::var("statement")));

    let bindings = executor.execute_bgp(&bgp).unwrap();

    // Should find only the two metadata statements, not the regular one
    assert_eq!(bindings.len(), 2);

    // All results should have quoted triples as subjects
    for binding in &bindings {
        if let Some(statement) = binding.get("statement") {
            assert!(statement.is_quoted_triple());
        }
    }
}

#[test]
fn test_function_error_handling() {
    // Test error cases for functions

    // SUBJECT on non-triple
    let result = FunctionEvaluator::evaluate(
        StarFunction::Subject,
        &[StarTerm::iri("http://example.org/not_a_triple").unwrap()],
    );
    assert!(result.is_err());

    // Wrong number of arguments
    let result = FunctionEvaluator::evaluate(
        StarFunction::Triple,
        &[StarTerm::iri("http://example.org/only_one_arg").unwrap()],
    );
    assert!(result.is_err());

    // Invalid subject for TRIPLE
    let result = FunctionEvaluator::evaluate(
        StarFunction::Triple,
        &[
            StarTerm::literal("literals_cannot_be_subjects").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal("o").unwrap(),
        ],
    );
    assert!(result.is_err());
}

#[test]
fn test_sparql_star_end_to_end() {
    // Comprehensive end-to-end test
    let store = StarStore::new();

    // Create a knowledge graph with provenance
    let facts = vec![
        ("alice", "knows", "bob", "0.9", "social_network"),
        ("bob", "knows", "charlie", "0.8", "social_network"),
        ("alice", "worksFor", "acme", "1.0", "hr_database"),
        ("bob", "worksFor", "acme", "0.95", "linkedin"),
    ];

    for (subj, pred, obj, conf, source) in facts {
        let fact = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/{}", subj)).unwrap(),
            StarTerm::iri(&format!("http://example.org/{}", pred)).unwrap(),
            StarTerm::iri(&format!("http://example.org/{}", obj)).unwrap(),
        );

        store.insert(&fact).unwrap();

        let meta = StarTriple::new(
            StarTerm::quoted_triple(fact),
            StarTerm::iri("http://example.org/confidence").unwrap(),
            StarTerm::literal(conf).unwrap(),
        );

        store.insert(&meta).unwrap();

        let source_meta = StarTriple::new(
            StarTerm::quoted_triple(meta.subject.as_quoted_triple().unwrap().clone()),
            StarTerm::iri("http://example.org/source").unwrap(),
            StarTerm::literal(source).unwrap(),
        );

        store.insert(&source_meta).unwrap();
    }

    let mut executor = QueryExecutor::new(store);

    // Query: Find all facts about Alice with high confidence (> 0.9)
    let mut bgp = BasicGraphPattern::new();

    // Pattern: ?fact <confidence> ?conf
    bgp.add_pattern(TriplePattern::new(
        TermPattern::Variable("fact".to_string()),
        TermPattern::Term(StarTerm::iri("http://example.org/confidence").unwrap()),
        TermPattern::Variable("conf".to_string()),
    ));

    // Pattern: ?fact <source> ?source
    bgp.add_pattern(TriplePattern::new(
        TermPattern::Variable("fact".to_string()),
        TermPattern::Term(StarTerm::iri("http://example.org/source").unwrap()),
        TermPattern::Variable("source".to_string()),
    ));

    // Filter: SUBJECT(?fact) = <alice> AND isTRIPLE(?fact)
    bgp.add_filter(Expression::is_triple(Expression::var("fact")));
    bgp.add_filter(Expression::equal(
        Expression::subject(Expression::var("fact")),
        Expression::term(StarTerm::iri("http://example.org/alice").unwrap()),
    ));

    let bindings = executor.execute_bgp(&bgp).unwrap();

    // Should find facts about Alice
    assert!(bindings.len() > 0);

    for binding in &bindings {
        if let Some(fact) = binding.get("fact") {
            assert!(fact.is_quoted_triple());
            let triple = fact.as_quoted_triple().unwrap();
            assert_eq!(
                triple.subject,
                StarTerm::iri("http://example.org/alice").unwrap()
            );
        }
    }
}
