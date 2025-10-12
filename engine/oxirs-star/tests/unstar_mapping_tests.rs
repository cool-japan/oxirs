//! Tests for W3C RDF-star unstar mapping
//!
//! The unstar mapping is the official W3C standard for translating RDF-star
//! to standard RDF for compatibility with non-RDF-star-aware reasoners.
//!
//! # W3C Unstar Mapping
//!
//! Each quoted triple in RDF-star is mapped to standard RDF reification:
//!
//! ```turtle
//! # RDF-star
//! <<:alice :age 30>> :certainty 0.9 .
//!
//! # Unstar mapping (standard RDF)
//! _:stmt rdf:subject :alice .
//! _:stmt rdf:predicate :age .
//! _:stmt rdf:object 30 .
//! _:stmt :certainty 0.9 .
//! ```

use oxirs_star::compatibility::{CompatibilityConfig, CompatibilityMode};
use oxirs_star::model::{StarGraph, StarTerm, StarTriple};

#[test]
fn test_unstar_basic_quoted_triple() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create RDF-star graph with quoted triple
    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Should have more triples (reification pattern)
    assert!(
        unstarred.len() > star_graph.len(),
        "Unstar should expand quoted triple to reification triples"
    );

    // Should not contain quoted triples anymore
    assert!(
        !CompatibilityMode::has_quoted_triples(&unstarred),
        "Unstarred graph should not contain quoted triples"
    );

    // Should contain reification patterns
    assert!(
        CompatibilityMode::has_reifications(&unstarred),
        "Unstarred graph should contain reification patterns"
    );
}

#[test]
fn test_rdfstar_reverse_mapping() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create RDF-star graph
    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Apply rdfstar (reverse) mapping
    let recovered = compat.rdfstar(&unstarred).unwrap();

    // Should recover the original structure
    assert_eq!(
        recovered.len(),
        star_graph.len(),
        "Rdfstar should recover original triple count"
    );

    // Should have quoted triples again
    assert!(
        CompatibilityMode::has_quoted_triples(&recovered),
        "Rdfstar should restore quoted triples"
    );
}

#[test]
fn test_unstar_roundtrip() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create RDF-star graph
    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta).unwrap();

    // Test round-trip
    let success = compat.test_unstar_roundtrip(&star_graph).unwrap();
    assert!(success, "Unstar round-trip should preserve graph structure");
}

#[test]
fn test_unstar_multiple_quoted_triples() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create RDF-star graph with multiple quoted triples
    let mut star_graph = StarGraph::new();

    let quoted1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta1 = StarTriple::new(
        StarTerm::quoted_triple(quoted1),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta1).unwrap();

    let quoted2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );
    let meta2 = StarTriple::new(
        StarTerm::quoted_triple(quoted2),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.8").unwrap(),
    );
    star_graph.insert(meta2).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Should have reification patterns for both quoted triples
    let reification_count = CompatibilityMode::count_reifications(&unstarred);
    assert_eq!(
        reification_count, 2,
        "Should have 2 reification patterns for 2 quoted triples"
    );

    // Round-trip should work
    let recovered = compat.rdfstar(&unstarred).unwrap();
    assert_eq!(recovered.len(), star_graph.len());
}

#[test]
fn test_unstar_nested_quoted_triples() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create nested quoted triples
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/knows").unwrap(),
        StarTerm::iri("http://example.org/bob").unwrap(),
    );

    let middle = StarTriple::new(
        StarTerm::quoted_triple(inner),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );

    let mut star_graph = StarGraph::new();
    let outer = StarTriple::new(
        StarTerm::quoted_triple(middle),
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::iri("http://example.org/study").unwrap(),
    );
    star_graph.insert(outer).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Should handle nested structure
    assert!(unstarred.len() > star_graph.len());
    assert!(!CompatibilityMode::has_quoted_triples(&unstarred));

    // Round-trip should work
    let recovered = compat.rdfstar(&unstarred).unwrap();
    assert_eq!(recovered.len(), star_graph.len());
}

#[test]
fn test_unstar_with_regular_triples() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create mixed graph (regular + quoted triples)
    let mut star_graph = StarGraph::new();

    // Regular triple
    let regular = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/name").unwrap(),
        StarTerm::literal("Alice").unwrap(),
    );
    star_graph.insert(regular.clone()).unwrap();

    // Quoted triple
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Regular triples should be preserved
    assert!(
        unstarred.contains(&regular),
        "Regular triples should be preserved during unstar mapping"
    );

    // Round-trip should work
    let recovered = compat.rdfstar(&unstarred).unwrap();
    assert_eq!(recovered.len(), star_graph.len());
    assert!(recovered.contains(&regular));
}

#[test]
fn test_unstar_with_blank_nodes() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create quoted triple with blank node
    let quoted = StarTriple::new(
        StarTerm::blank_node("b1").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut star_graph = StarGraph::new();
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Should handle blank nodes correctly
    assert!(unstarred.len() > star_graph.len());
    assert!(CompatibilityMode::has_reifications(&unstarred));
}

#[test]
fn test_unstar_preserves_metadata() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create quoted triple with multiple metadata properties
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let mut star_graph = StarGraph::new();

    let meta1 = StarTriple::new(
        StarTerm::quoted_triple(quoted.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta1).unwrap();

    let meta2 = StarTriple::new(
        StarTerm::quoted_triple(quoted.clone()),
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::iri("http://example.org/census").unwrap(),
    );
    star_graph.insert(meta2).unwrap();

    let meta3 = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/timestamp").unwrap(),
        StarTerm::literal("2023-10-12").unwrap(),
    );
    star_graph.insert(meta3).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Round-trip should preserve all metadata
    let recovered = compat.rdfstar(&unstarred).unwrap();
    assert_eq!(
        recovered.len(),
        star_graph.len(),
        "All metadata should be preserved"
    );
}

#[test]
fn test_unstar_empty_graph() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    let star_graph = StarGraph::new();

    // Apply unstar mapping on empty graph
    let unstarred = compat.unstar(&star_graph).unwrap();

    assert!(unstarred.is_empty());
}

#[test]
fn test_unstar_only_regular_triples() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create graph with only regular triples (no quoted triples)
    let mut star_graph = StarGraph::new();

    let triple1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/name").unwrap(),
        StarTerm::literal("Alice").unwrap(),
    );
    star_graph.insert(triple1.clone()).unwrap();

    let triple2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/name").unwrap(),
        StarTerm::literal("Bob").unwrap(),
    );
    star_graph.insert(triple2.clone()).unwrap();

    // Apply unstar mapping
    let unstarred = compat.unstar(&star_graph).unwrap();

    // Should be unchanged (no quoted triples to convert)
    assert_eq!(unstarred.len(), star_graph.len());
    assert!(unstarred.contains(&triple1));
    assert!(unstarred.contains(&triple2));
}

#[test]
fn test_unstar_with_different_strategies() {
    // Test that unstar always uses standard reification regardless of config

    // Start with blank nodes strategy
    let config = CompatibilityConfig::blank_nodes();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta).unwrap();

    // unstar() should use standard reification, not blank nodes
    let unstarred = compat.unstar(&star_graph).unwrap();

    // After unstar, should have reification patterns
    assert!(CompatibilityMode::has_reifications(&unstarred));

    // Original strategy should be restored
    let recovered = compat.rdfstar(&unstarred).unwrap();
    assert_eq!(recovered.len(), star_graph.len());
}

#[test]
fn test_unstar_idempotence() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta).unwrap();

    // Apply unstar once
    let unstarred1 = compat.unstar(&star_graph).unwrap();

    // Apply unstar again (should be idempotent)
    let unstarred2 = compat.unstar(&unstarred1).unwrap();

    // Should be the same (no quoted triples to convert)
    assert_eq!(unstarred1.len(), unstarred2.len());
}

#[test]
fn test_complex_unstar_roundtrip() {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create complex graph with various patterns
    let mut star_graph = StarGraph::new();

    // Regular triple
    let regular = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/name").unwrap(),
        StarTerm::literal("Alice").unwrap(),
    );
    star_graph.insert(regular).unwrap();

    // Simple quoted triple
    let quoted1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let meta1 = StarTriple::new(
        StarTerm::quoted_triple(quoted1),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    star_graph.insert(meta1).unwrap();

    // Another quoted triple
    let quoted2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );
    let meta2 = StarTriple::new(
        StarTerm::quoted_triple(quoted2),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.8").unwrap(),
    );
    star_graph.insert(meta2).unwrap();

    // Test complete round-trip
    let success = compat.test_unstar_roundtrip(&star_graph).unwrap();
    assert!(success, "Complex graph should round-trip successfully");
}
