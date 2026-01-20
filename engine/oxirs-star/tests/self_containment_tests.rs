//! Tests for RDF-star self-containment validation
//!
//! The W3C RDF-star specification explicitly forbids triples from containing
//! themselves (directly or transitively). This module tests the validation
//! logic that prevents such invalid structures.

use oxirs_star::model::{StarTerm, StarTriple};

#[test]
fn test_simple_triple_not_self_contained() {
    // Regular triple with no quoted triples
    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    assert!(!triple.is_self_contained());
    assert!(triple.validate().is_ok());
}

#[test]
fn test_quoted_triple_not_self_contained() {
    // Inner triple
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/bob").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    // Outer triple with quoted inner triple
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );

    assert!(!outer.is_self_contained());
    assert!(outer.validate().is_ok());
}

#[test]
fn test_deeply_nested_not_self_contained() {
    // Create a deeply nested structure without cycles
    // Level 1
    let level1 = StarTriple::new(
        StarTerm::iri("http://example.org/s1").unwrap(),
        StarTerm::iri("http://example.org/p1").unwrap(),
        StarTerm::literal("o1").unwrap(),
    );

    // Level 2
    let level2 = StarTriple::new(
        StarTerm::quoted_triple(level1),
        StarTerm::iri("http://example.org/p2").unwrap(),
        StarTerm::literal("o2").unwrap(),
    );

    // Level 3
    let level3 = StarTriple::new(
        StarTerm::quoted_triple(level2),
        StarTerm::iri("http://example.org/p3").unwrap(),
        StarTerm::literal("o3").unwrap(),
    );

    assert!(!level3.is_self_contained());
    assert!(level3.validate().is_ok());
}

#[test]
fn test_contains_triple_basic() {
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

    // Outer contains inner
    assert!(outer.contains_triple(&inner));

    // But not the other way around
    assert!(!inner.contains_triple(&outer));

    // Neither contains itself (no cycles)
    assert!(!outer.contains_triple(&outer));
    assert!(!inner.contains_triple(&inner));
}

#[test]
fn test_contains_triple_in_object_position() {
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let outer = StarTriple::new(
        StarTerm::iri("http://example.org/source").unwrap(),
        StarTerm::iri("http://example.org/states").unwrap(),
        StarTerm::quoted_triple(inner.clone()),
    );

    assert!(outer.contains_triple(&inner));
    assert!(!inner.contains_triple(&outer));
}

#[test]
fn test_contains_triple_multiple_levels() {
    // Create a 3-level nesting
    let level1 = StarTriple::new(
        StarTerm::iri("http://example.org/s").unwrap(),
        StarTerm::iri("http://example.org/p").unwrap(),
        StarTerm::literal("o").unwrap(),
    );

    let level2 = StarTriple::new(
        StarTerm::quoted_triple(level1.clone()),
        StarTerm::iri("http://example.org/meta1").unwrap(),
        StarTerm::literal("m1").unwrap(),
    );

    let level3 = StarTriple::new(
        StarTerm::quoted_triple(level2.clone()),
        StarTerm::iri("http://example.org/meta2").unwrap(),
        StarTerm::literal("m2").unwrap(),
    );

    // Level3 contains both level2 and level1 (transitively)
    assert!(level3.contains_triple(&level2));
    assert!(level3.contains_triple(&level1));

    // Level2 contains level1
    assert!(level2.contains_triple(&level1));

    // But not the reverse
    assert!(!level1.contains_triple(&level2));
    assert!(!level1.contains_triple(&level3));
    assert!(!level2.contains_triple(&level3));
}

#[test]
fn test_multiple_quoted_triples_same_level() {
    let quoted1 = StarTriple::new(
        StarTerm::iri("http://example.org/a").unwrap(),
        StarTerm::iri("http://example.org/p1").unwrap(),
        StarTerm::literal("v1").unwrap(),
    );

    let quoted2 = StarTriple::new(
        StarTerm::iri("http://example.org/b").unwrap(),
        StarTerm::iri("http://example.org/p2").unwrap(),
        StarTerm::literal("v2").unwrap(),
    );

    // Create a triple with two different quoted triples
    // Subject: quoted1, Object: quoted2
    let container = StarTriple::new(
        StarTerm::quoted_triple(quoted1.clone()),
        StarTerm::iri("http://example.org/relates").unwrap(),
        StarTerm::quoted_triple(quoted2.clone()),
    );

    assert!(container.contains_triple(&quoted1));
    assert!(container.contains_triple(&quoted2));
    assert!(!container.is_self_contained());
}

#[test]
fn test_max_depth_protection() {
    // Create a very deeply nested structure (beyond max depth of 100)
    let mut current = StarTriple::new(
        StarTerm::iri("http://example.org/s").unwrap(),
        StarTerm::iri("http://example.org/p").unwrap(),
        StarTerm::literal("o").unwrap(),
    );

    // Nest 150 levels (exceeds max_depth of 100)
    for i in 0..150 {
        current = StarTriple::new(
            StarTerm::quoted_triple(current),
            StarTerm::iri(&format!("http://example.org/p{i}")).unwrap(),
            StarTerm::literal(&format!("o{i}")).unwrap(),
        );
    }

    // Should still validate (no cycle, just deep)
    assert!(!current.is_self_contained());
    assert!(current.validate().is_ok());
}

#[test]
fn test_quoted_triple_in_predicate_position() {
    // Although rare, RDF-star allows quoted triples in predicate position
    let predicate_triple = StarTriple::new(
        StarTerm::iri("http://example.org/action").unwrap(),
        StarTerm::iri("http://example.org/type").unwrap(),
        StarTerm::literal("relationship").unwrap(),
    );

    let triple_with_quoted_predicate = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::quoted_triple(predicate_triple.clone()),
        StarTerm::iri("http://example.org/bob").unwrap(),
    );

    assert!(triple_with_quoted_predicate.contains_triple(&predicate_triple));
    assert!(!triple_with_quoted_predicate.is_self_contained());
}

#[test]
fn test_identical_but_separate_triples() {
    // Create two identical triples (same content, different instances)
    let triple1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    let triple2 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );

    // They should be equal
    assert_eq!(triple1, triple2);

    // If we nest triple2 inside triple1's structure
    let outer = StarTriple::new(
        StarTerm::quoted_triple(triple2.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );

    // outer contains triple1 (because triple1 == triple2)
    assert!(outer.contains_triple(&triple1));
}

#[test]
fn test_validation_catches_self_containment() {
    // Create a triple
    let base = StarTriple::new(
        StarTerm::iri("http://example.org/s").unwrap(),
        StarTerm::iri("http://example.org/p").unwrap(),
        StarTerm::literal("o").unwrap(),
    );

    // Try to create a quoted version of itself inside it
    // (Note: In practice, this is hard to construct due to Rust's ownership,
    // but we test the validation logic)

    // We can't actually create a true cycle in Rust, but we can test
    // the detection logic by checking that normal nested structures
    // don't trigger false positives

    let nested = StarTriple::new(
        StarTerm::quoted_triple(base.clone()),
        StarTerm::iri("http://example.org/meta").unwrap(),
        StarTerm::literal("metadata").unwrap(),
    );

    // This should NOT be self-contained (nested contains base, not itself)
    assert!(!nested.is_self_contained());
    assert!(nested.validate().is_ok());
}

#[test]
fn test_complex_nesting_without_cycles() {
    // Create a complex structure with multiple levels
    let t1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/knows").unwrap(),
        StarTerm::iri("http://example.org/bob").unwrap(),
    );

    let t2 = StarTriple::new(
        StarTerm::iri("http://example.org/charlie").unwrap(),
        StarTerm::iri("http://example.org/knows").unwrap(),
        StarTerm::iri("http://example.org/david").unwrap(),
    );

    let t3 = StarTriple::new(
        StarTerm::quoted_triple(t1.clone()),
        StarTerm::iri("http://example.org/and").unwrap(),
        StarTerm::quoted_triple(t2.clone()),
    );

    let t4 = StarTriple::new(
        StarTerm::quoted_triple(t3.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.95").unwrap(),
    );

    // Check containment relationships
    assert!(t3.contains_triple(&t1));
    assert!(t3.contains_triple(&t2));
    assert!(t4.contains_triple(&t3));
    assert!(t4.contains_triple(&t1)); // Transitive
    assert!(t4.contains_triple(&t2)); // Transitive

    // No cycles
    assert!(!t1.is_self_contained());
    assert!(!t2.is_self_contained());
    assert!(!t3.is_self_contained());
    assert!(!t4.is_self_contained());

    // All should validate
    assert!(t1.validate().is_ok());
    assert!(t2.validate().is_ok());
    assert!(t3.validate().is_ok());
    assert!(t4.validate().is_ok());
}

#[test]
fn test_contains_triple_with_blank_nodes() {
    let inner = StarTriple::new(
        StarTerm::blank_node("b1").unwrap(),
        StarTerm::iri("http://example.org/p").unwrap(),
        StarTerm::literal("value").unwrap(),
    );

    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/meta").unwrap(),
        StarTerm::blank_node("b2").unwrap(),
    );

    assert!(outer.contains_triple(&inner));
    assert!(!outer.is_self_contained());
}

#[test]
fn test_contains_triple_with_variables() {
    // Variables are used in SPARQL-star queries
    let triple_with_var = StarTriple::new(
        StarTerm::variable("x").unwrap(),
        StarTerm::iri("http://example.org/p").unwrap(),
        StarTerm::literal("value").unwrap(),
    );

    let outer = StarTriple::new(
        StarTerm::quoted_triple(triple_with_var.clone()),
        StarTerm::iri("http://example.org/pattern").unwrap(),
        StarTerm::variable("y").unwrap(),
    );

    assert!(outer.contains_triple(&triple_with_var));
    assert!(!outer.is_self_contained());
}

#[test]
fn test_self_containment_documentation() {
    // This test documents what self-containment means

    // Case 1: No self-containment - normal nesting
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/a").unwrap(),
        StarTerm::iri("http://example.org/p").unwrap(),
        StarTerm::literal("x").unwrap(),
    );
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner),
        StarTerm::iri("http://example.org/meta").unwrap(),
        StarTerm::literal("y").unwrap(),
    );
    assert!(
        !outer.is_self_contained(),
        "Normal nesting is not self-containment"
    );

    // Case 2: What would be self-containment (can't actually construct in Rust)
    // Conceptually: << A >> p o, where A = << A >> p o
    // This is impossible to construct due to Rust's ownership rules,
    // which is a good thing! The validation is for when parsing
    // potentially malformed data from external sources.
}
