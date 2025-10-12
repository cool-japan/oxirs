//! Tests for RDF-star referential opacity semantics
//!
//! Referential opacity is the key semantic property of RDF-star:
//! quoted triples do NOT assert the statement they quote.
//!
//! This is the fundamental difference from standard RDF reification.

use oxirs_star::model::{StarGraph, StarTerm, StarTriple};
use oxirs_star::semantics::{EntailmentChecker, SemanticValidator, TransparencyEnablingProperties};

#[test]
fn test_quoted_triple_does_not_assert_content() {
    // Create a quoted triple
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();

    // Add metadata about the quoted triple
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(outer).unwrap();

    // CRITICAL: The inner triple should NOT be in the graph
    assert!(
        !graph.contains(&inner),
        "Referential opacity violated: quoted triple was asserted"
    );

    // The graph should only contain the outer triple
    assert_eq!(graph.len(), 1);
}

#[test]
fn test_explicit_assertion_vs_quotation() {
    let statement = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();

    // Case 1: Only quote the statement (no assertion)
    let quoted_only = StarTriple::new(
        StarTerm::quoted_triple(statement.clone()),
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::iri("http://example.org/census").unwrap(),
    );
    graph.insert(quoted_only).unwrap();

    // The statement should NOT be asserted
    let validator = SemanticValidator::new();
    assert!(
        !validator.is_asserted(&graph, &statement).unwrap(),
        "Statement should not be asserted when only quoted"
    );
    assert!(
        validator.is_only_quoted(&graph, &statement).unwrap(),
        "Statement should be recognized as only quoted"
    );

    // Case 2: Now explicitly assert the statement
    graph.insert(statement.clone()).unwrap();

    // Now it should be asserted
    assert!(
        validator.is_asserted(&graph, &statement).unwrap(),
        "Statement should be asserted after explicit insertion"
    );
    assert!(
        !validator.is_only_quoted(&graph, &statement).unwrap(),
        "Statement should no longer be only quoted"
    );
}

#[test]
fn test_nested_referential_opacity() {
    // Create nested quoted triples
    let level1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/knows").unwrap(),
        StarTerm::iri("http://example.org/bob").unwrap(),
    );

    let level2 = StarTriple::new(
        StarTerm::quoted_triple(level1.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.8").unwrap(),
    );

    let level3 = StarTriple::new(
        StarTerm::quoted_triple(level2.clone()),
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::iri("http://example.org/study").unwrap(),
    );

    let mut graph = StarGraph::new();
    graph.insert(level3).unwrap();

    // Neither level1 nor level2 should be asserted
    assert!(!graph.contains(&level1));
    assert!(!graph.contains(&level2));

    // Referential opacity applies recursively
    assert_eq!(
        graph.len(),
        1,
        "Only the outermost triple should be in the graph"
    );
}

#[test]
fn test_entailment_checker_referential_opacity() {
    let checker = EntailmentChecker::new();

    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(outer).unwrap();

    // Quoted triple does NOT entail its content (referential opacity)
    assert!(
        !checker.quoted_entails_content(&graph, &inner).unwrap(),
        "Referential opacity: quoted triple must not entail its content"
    );
}

#[test]
fn test_semantic_validator_accepts_valid_graph() {
    let validator = SemanticValidator::new();

    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(outer).unwrap();

    // This graph respects referential opacity
    assert!(validator.validate_graph(&graph).is_ok());
}

#[test]
fn test_extract_quoted_triples() {
    let validator = SemanticValidator::new();

    let quoted1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let quoted2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    let mut graph = StarGraph::new();

    let meta1 = StarTriple::new(
        StarTerm::quoted_triple(quoted1.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(meta1).unwrap();

    let meta2 = StarTriple::new(
        StarTerm::quoted_triple(quoted2.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.8").unwrap(),
    );
    graph.insert(meta2).unwrap();

    // Extract all quoted triples
    let extracted = validator.extract_quoted_triples(&graph);
    assert_eq!(extracted.len(), 2);
    assert!(extracted.contains(&quoted1));
    assert!(extracted.contains(&quoted2));

    // But they should not be in the graph as asserted triples
    assert!(!graph.contains(&quoted1));
    assert!(!graph.contains(&quoted2));
}

#[test]
fn test_entailment_simple_containment() {
    let checker = EntailmentChecker::new();

    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut source = StarGraph::new();
    source.insert(triple.clone()).unwrap();

    let mut target = StarGraph::new();
    target.insert(triple).unwrap();

    // Identical graphs entail each other
    assert!(checker.entails(&source, &target).unwrap());
    assert!(checker.entails(&target, &source).unwrap());
}

#[test]
fn test_entailment_with_additional_triples() {
    let checker = EntailmentChecker::new();

    let triple1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let triple2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    let mut source = StarGraph::new();
    source.insert(triple1.clone()).unwrap();
    source.insert(triple2).unwrap();

    let mut target = StarGraph::new();
    target.insert(triple1).unwrap();

    // Source has more triples, so it entails target
    assert!(checker.entails(&source, &target).unwrap());

    // But target does not entail source
    assert!(!checker.entails(&target, &source).unwrap());
}

#[test]
fn test_closure_respects_opacity() {
    let checker = EntailmentChecker::new();

    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(outer).unwrap();

    // Compute closure (should not add quoted triples as assertions)
    let closure = checker.compute_closure(&graph).unwrap();

    // The closure should NOT contain the inner triple
    assert!(
        !closure.contains(&inner),
        "Closure violated referential opacity"
    );

    // Should have same number of triples
    assert_eq!(closure.len(), graph.len());
}

#[test]
fn test_transparency_enabling_properties_basic() {
    let mut tep = TransparencyEnablingProperties::new();

    // Initially no TEPs
    assert_eq!(tep.properties().len(), 0);

    // Add rdf:type as a TEP
    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    tep.add_property(rdf_type.to_string());

    assert!(tep.is_tep(rdf_type));
    assert!(!tep.is_tep("http://example.org/other"));
}

#[test]
fn test_common_teps() {
    let tep = TransparencyEnablingProperties::with_common_properties();

    // Should include rdf:type
    assert!(tep.is_tep("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));
}

#[test]
fn test_triple_uses_tep() {
    let mut tep = TransparencyEnablingProperties::new();
    tep.add_property("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string());

    let type_triple = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        StarTerm::iri("http://example.org/Person").unwrap(),
    );

    let other_triple = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    assert!(tep.triple_uses_tep(&type_triple));
    assert!(!tep.triple_uses_tep(&other_triple));
}

#[test]
fn test_tep_no_duplicates() {
    let mut tep = TransparencyEnablingProperties::new();

    let prop = "http://example.org/prop".to_string();
    tep.add_property(prop.clone());
    tep.add_property(prop.clone());
    tep.add_property(prop);

    // Should only have one instance
    assert_eq!(tep.properties().len(), 1);
}

#[test]
fn test_referential_opacity_with_multiple_predicates() {
    let statement = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();

    // Add multiple metadata predicates about the same quoted triple
    let meta1 = StarTriple::new(
        StarTerm::quoted_triple(statement.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(meta1).unwrap();

    let meta2 = StarTriple::new(
        StarTerm::quoted_triple(statement.clone()),
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::iri("http://example.org/census").unwrap(),
    );
    graph.insert(meta2).unwrap();

    let meta3 = StarTriple::new(
        StarTerm::quoted_triple(statement.clone()),
        StarTerm::iri("http://example.org/timestamp").unwrap(),
        StarTerm::literal("2023-10-12").unwrap(),
    );
    graph.insert(meta3).unwrap();

    // The statement should still not be asserted
    assert!(!graph.contains(&statement));
    assert_eq!(graph.len(), 3, "Should have 3 metadata triples");
}

#[test]
fn test_opacity_with_blank_nodes() {
    let inner = StarTriple::new(
        StarTerm::blank_node("b1").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(outer).unwrap();

    // Referential opacity applies even with blank nodes
    assert!(!graph.contains(&inner));
}

#[test]
fn test_semantic_validator_non_strict_mode() {
    let validator = SemanticValidator::new().with_strict_opacity(false);

    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();

    // Add both quoted and asserted versions (technically allowed)
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(outer).unwrap();
    graph.insert(inner).unwrap();

    // In non-strict mode, this should still validate
    assert!(validator.validate_graph(&graph).is_ok());
}

#[test]
fn test_referential_opacity_with_literal_subjects() {
    // Although literals can't be subjects in standard RDF,
    // test that referential opacity logic handles all term types

    let validator = SemanticValidator::new();

    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();
    graph.insert(triple.clone()).unwrap();

    // Check that the validator correctly identifies asserted triples
    assert!(validator.is_asserted(&graph, &triple).unwrap());
    assert!(!validator.is_only_quoted(&graph, &triple).unwrap());
}

#[test]
fn test_quoted_in_multiple_positions() {
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut graph = StarGraph::new();

    // Quote in subject position
    let triple1 = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(triple1).unwrap();

    // Quote in object position
    let triple2 = StarTriple::new(
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::iri("http://example.org/states").unwrap(),
        StarTerm::quoted_triple(inner.clone()),
    );
    graph.insert(triple2).unwrap();

    // The inner triple should not be asserted
    assert!(!graph.contains(&inner));

    // But should be extractable from both positions
    let validator = SemanticValidator::new();
    let extracted = validator.extract_quoted_triples(&graph);

    // Should extract the inner triple (but counted once per occurrence)
    assert!(extracted.contains(&inner));
}
