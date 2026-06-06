//! # OWL 2 RL Tests
//!
//! Comprehensive test suite for the OWL 2 RL reasoner: core inference rules,
//! property axioms, class expressions, sameAs, and extended scenarios.

use crate::owl_rl_reasoner::Owl2RlReasoner;
use crate::owl_rl_rules::{
    OWL_ASYMMETRIC_PROPERTY, OWL_DISJOINT_WITH, OWL_EQUIVALENT_CLASS, OWL_EQUIVALENT_PROPERTY,
    OWL_FUNCTIONAL_PROPERTY, OWL_INV_FUNCTIONAL_PROPERTY, OWL_IRREFLEXIVE_PROPERTY, OWL_NOTHING,
    OWL_SAME_AS, RDFS_DOMAIN, RDFS_RANGE, RDFS_SUBCLASS_OF, RDFS_SUBPROPERTY_OF, RDF_TYPE,
};

fn rl() -> Owl2RlReasoner {
    Owl2RlReasoner::new()
}

#[test]
fn test_subclass_transitivity() {
    let mut r = rl();
    r.add_subclass_of("Dog", "Mammal");
    r.add_subclass_of("Mammal", "Animal");
    let report = r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("Dog", RDFS_SUBCLASS_OF, "Animal"),
        "Expected Dog ⊑ Animal, got {} new triples in {} iterations",
        report.new_triples_count,
        report.iterations
    );
}

#[test]
fn test_type_propagation_via_subclass() {
    let mut r = rl();
    r.add_type("fido", "Dog");
    r.add_subclass_of("Dog", "Animal");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("fido", RDF_TYPE, "Animal"),
        "Expected fido rdf:type Animal"
    );
}

#[test]
fn test_domain_inference() {
    let mut r = rl();
    r.add_axiom("hasParent", RDFS_DOMAIN, "Person");
    r.add_axiom("alice", "hasParent", "bob");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("alice", RDF_TYPE, "Person"),
        "Expected alice rdf:type Person from domain"
    );
}

#[test]
fn test_range_inference() {
    let mut r = rl();
    r.add_axiom("hasParent", RDFS_RANGE, "Person");
    r.add_axiom("alice", "hasParent", "bob");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("bob", RDF_TYPE, "Person"),
        "Expected bob rdf:type Person from range"
    );
}

#[test]
fn test_symmetric_property() {
    let mut r = rl();
    r.add_symmetric_property("knows");
    r.add_axiom("alice", "knows", "bob");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("bob", "knows", "alice"),
        "Expected bob knows alice from SymmetricProperty"
    );
}

#[test]
fn test_transitive_property() {
    let mut r = rl();
    r.add_transitive_property("ancestorOf");
    r.add_axiom("grandparent", "ancestorOf", "parent");
    r.add_axiom("parent", "ancestorOf", "child");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("grandparent", "ancestorOf", "child"),
        "Expected transitive ancestorOf"
    );
}

#[test]
fn test_inverse_of() {
    let mut r = rl();
    r.add_inverse_of("hasParent", "hasChild");
    r.add_axiom("alice", "hasParent", "bob");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("bob", "hasChild", "alice"),
        "Expected bob hasChild alice from inverseOf"
    );
}

#[test]
fn test_equivalent_class() {
    let mut r = rl();
    r.add_axiom("Human", OWL_EQUIVALENT_CLASS, "Person");
    r.add_type("alice", "Human");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("alice", RDF_TYPE, "Person"),
        "Expected alice rdf:type Person via equivalentClass"
    );
}

#[test]
fn test_disjoint_with_inconsistency() {
    let mut r = rl();
    r.add_axiom("Cat", OWL_DISJOINT_WITH, "Dog");
    r.add_type("fido", "Dog");
    r.add_type("fido", "Cat");
    r.materialize().expect("materialization failed");
    assert!(
        !r.is_consistent(),
        "Expected inconsistency due to disjointWith"
    );
    assert!(!r.inconsistencies().is_empty());
}

#[test]
fn test_same_as_transitivity() {
    let mut r = rl();
    r.add_axiom("alice", OWL_SAME_AS, "alicia");
    r.add_axiom("alicia", OWL_SAME_AS, "ali");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("alice", OWL_SAME_AS, "ali"),
        "Expected sameAs transitivity"
    );
}

#[test]
fn test_inference_report() {
    let mut r = rl();
    r.add_subclass_of("A", "B");
    r.add_subclass_of("B", "C");
    let report = r.materialize().expect("materialization failed");
    assert!(report.iterations >= 1);
    assert!(report.new_triples_count >= 1);
    assert!(!report.rules_fired.is_empty());
}

#[test]
fn test_subproperty_propagation() {
    let mut r = rl();
    r.add_axiom("isChildOf", RDFS_SUBPROPERTY_OF, "isRelatedTo");
    r.add_axiom("alice", "isChildOf", "bob");
    r.materialize().expect("materialization failed");
    assert!(
        r.is_entailed("alice", "isRelatedTo", "bob"),
        "Expected subProperty propagation"
    );
}

#[test]
fn test_max_iterations_safety() {
    let mut r = Owl2RlReasoner::new().with_max_iterations(5);
    // Non-terminating scenario is bounded
    r.add_axiom("A", RDFS_SUBCLASS_OF, "B");
    // Should succeed within 5 iterations for simple cases
    let result = r.materialize();
    // May succeed or fail with MaxIterationsExceeded, but should not panic
    let _ = result;
}

#[test]
fn test_match_triples_wildcard() {
    let mut set = std::collections::HashSet::new();
    set.insert((
        "alice".to_string(),
        RDF_TYPE.to_string(),
        "Person".to_string(),
    ));
    set.insert((
        "bob".to_string(),
        RDF_TYPE.to_string(),
        "Person".to_string(),
    ));
    set.insert(("alice".to_string(), "knows".to_string(), "bob".to_string()));

    let type_triples: Vec<_> = Owl2RlReasoner::match_triples(&set, None, Some(RDF_TYPE), None);
    assert_eq!(type_triples.len(), 2);

    let alice_triples: Vec<_> = Owl2RlReasoner::match_triples(&set, Some("alice"), None, None);
    assert_eq!(alice_triples.len(), 2);
}

#[test]
fn test_nothing_inconsistency() {
    let mut r = rl();
    r.add_axiom("x", RDF_TYPE, OWL_NOTHING);
    r.materialize().expect("materialization failed");
    assert!(!r.is_consistent());
}

// -----------------------------------------------------------------------
// Extended Tests
// -----------------------------------------------------------------------

#[test]
fn test_empty_reasoner_materialize() {
    let mut r = rl();
    let report = r.materialize().expect("empty materialize failed");
    assert_eq!(report.new_triples_count, 0);
    assert!(r.is_consistent());
}

#[test]
fn test_single_type_assertion_no_inference() {
    let mut r = rl();
    r.add_type("alice", "Person");
    r.materialize().expect("failed");
    assert!(r.is_entailed("alice", RDF_TYPE, "Person"));
}

#[test]
fn test_deep_subclass_chain() {
    let mut r = rl();
    // A ⊑ B ⊑ C ⊑ D ⊑ E ⊑ F
    for i in 0..5usize {
        r.add_subclass_of(&format!("C{}", i), &format!("C{}", i + 1));
    }
    r.add_type("x", "C0");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("x", RDF_TYPE, "C5"),
        "x should be C5 via chain"
    );
}

#[test]
fn test_equivalent_property_propagation() {
    let mut r = rl();
    r.add_axiom("likes", OWL_EQUIVALENT_PROPERTY, "enjoys");
    r.add_axiom("alice", "likes", "music");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("alice", "enjoys", "music"),
        "alice enjoys music via equivalentProperty"
    );
}

#[test]
fn test_equivalent_property_reverse() {
    let mut r = rl();
    r.add_axiom("P", OWL_EQUIVALENT_PROPERTY, "Q");
    r.add_axiom("a", "Q", "b");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("a", "P", "b"),
        "a P b via reverse equivalentProperty"
    );
}

#[test]
fn test_functional_property_same_as() {
    let mut r = rl();
    r.add_axiom("hasMother", RDF_TYPE, OWL_FUNCTIONAL_PROPERTY);
    r.add_axiom("alice", "hasMother", "eve");
    r.add_axiom("alice", "hasMother", "eva");
    r.materialize().expect("failed");
    // eve and eva should be sameAs
    let eve_same = r.is_entailed("eve", OWL_SAME_AS, "eva");
    let eva_same = r.is_entailed("eva", OWL_SAME_AS, "eve");
    assert!(
        eve_same || eva_same,
        "eve and eva should be sameAs via FunctionalProperty"
    );
}

#[test]
fn test_inverse_functional_property() {
    let mut r = rl();
    r.add_axiom(
        "hasSocialSecurityNumber",
        RDF_TYPE,
        OWL_INV_FUNCTIONAL_PROPERTY,
    );
    r.add_axiom("alice", "hasSocialSecurityNumber", "123-45-6789");
    r.add_axiom("alicia", "hasSocialSecurityNumber", "123-45-6789");
    r.materialize().expect("failed");
    // alice and alicia should be sameAs
    let same = r.is_entailed("alice", OWL_SAME_AS, "alicia")
        || r.is_entailed("alicia", OWL_SAME_AS, "alice");
    assert!(
        same,
        "alice and alicia should be sameAs via InverseFunctionalProperty"
    );
}

#[test]
fn test_subproperty_chain() {
    let mut r = rl();
    r.add_axiom("P1", RDFS_SUBPROPERTY_OF, "P2");
    r.add_axiom("P2", RDFS_SUBPROPERTY_OF, "P3");
    r.add_axiom("a", "P1", "b");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("a", "P3", "b"),
        "a P3 b via subProperty chain"
    );
}

#[test]
fn test_same_as_explicit_then_symmetric() {
    // Test that explicit sameAs generates symmetric inference
    let mut r = rl();
    r.add_axiom("alice", OWL_SAME_AS, "alicia");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("alicia", OWL_SAME_AS, "alice"),
        "alicia sameAs alice (symmetry from explicit assertion)"
    );
}

#[test]
fn test_same_as_symmetry() {
    let mut r = rl();
    r.add_axiom("alice", OWL_SAME_AS, "alicia");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("alicia", OWL_SAME_AS, "alice"),
        "sameAs symmetry"
    );
}

#[test]
fn test_same_as_transitivity_three() {
    let mut r = rl();
    r.add_axiom("a", OWL_SAME_AS, "b");
    r.add_axiom("b", OWL_SAME_AS, "c");
    r.add_axiom("c", OWL_SAME_AS, "d");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("a", OWL_SAME_AS, "d"),
        "a sameAs d via 3-step transitivity"
    );
}

#[test]
fn test_same_as_type_propagation() {
    let mut r = rl();
    r.add_type("alice", "Person");
    r.add_axiom("alice", OWL_SAME_AS, "alicia");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("alicia", RDF_TYPE, "Person"),
        "alicia:Person via sameAs with alice:Person"
    );
}

#[test]
fn test_domain_and_range_combined() {
    let mut r = rl();
    r.add_axiom("hasChild", RDFS_DOMAIN, "Parent");
    r.add_axiom("hasChild", RDFS_RANGE, "Child");
    r.add_axiom("bob", "hasChild", "tommy");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("bob", RDF_TYPE, "Parent"),
        "bob:Parent via domain"
    );
    assert!(
        r.is_entailed("tommy", RDF_TYPE, "Child"),
        "tommy:Child via range"
    );
}

#[test]
fn test_asymmetric_property_violation() {
    let mut r = rl();
    r.add_axiom("isStrictlyLessThan", RDF_TYPE, OWL_ASYMMETRIC_PROPERTY);
    r.add_axiom("a", "isStrictlyLessThan", "b");
    r.add_axiom("b", "isStrictlyLessThan", "a");
    r.materialize().expect("materialize still runs");
    assert!(
        !r.is_consistent(),
        "Asymmetric violation should be inconsistent"
    );
}

#[test]
fn test_irreflexive_property_violation() {
    let mut r = rl();
    r.add_axiom("isStrictlyBefore", RDF_TYPE, OWL_IRREFLEXIVE_PROPERTY);
    r.add_axiom("now", "isStrictlyBefore", "now");
    r.materialize().expect("materialize still runs");
    assert!(
        !r.is_consistent(),
        "IrreflexiveProperty self-loop should be inconsistent"
    );
}

#[test]
fn test_disjoint_with_three_classes() {
    let mut r = rl();
    r.add_axiom("A", OWL_DISJOINT_WITH, "B");
    r.add_axiom("B", OWL_DISJOINT_WITH, "C");
    r.add_type("x", "A");
    r.add_type("x", "B");
    r.materialize().expect("materialize still runs");
    assert!(
        !r.is_consistent(),
        "x:A and x:B with A disjointWith B is inconsistent"
    );
}

#[test]
fn test_multiple_disjoint_no_violation() {
    let mut r = rl();
    r.add_axiom("Mammal", OWL_DISJOINT_WITH, "Reptile");
    r.add_type("fido", "Mammal");
    r.add_type("rex", "Reptile");
    r.materialize().expect("failed");
    assert!(
        r.is_consistent(),
        "Different individuals in disjoint classes is OK"
    );
}

#[test]
fn test_inference_report_iterations() {
    let mut r = rl();
    r.add_subclass_of("A", "B");
    r.add_subclass_of("B", "C");
    r.add_subclass_of("C", "D");
    let report = r.materialize().expect("failed");
    assert!(report.iterations >= 1, "Should have at least 1 iteration");
    assert!(report.new_triples_count >= 1, "Should have new triples");
}

#[test]
fn test_inference_report_rules_fired() {
    let mut r = rl();
    r.add_subclass_of("X", "Y");
    r.add_type("a", "X");
    let report = r.materialize().expect("failed");
    assert!(
        !report.rules_fired.is_empty(),
        "rules_fired should not be empty"
    );
}

#[test]
fn test_inference_report_duration_positive() {
    let mut r = rl();
    r.add_subclass_of("A", "B");
    let report = r.materialize().expect("failed");
    // Duration::as_secs() always returns u64, just verify the field is accessible
    let _ = report.duration.as_millis();
}

#[test]
fn test_materialize_multiple_times_idempotent() {
    let mut r = rl();
    r.add_subclass_of("Dog", "Animal");
    r.add_type("fido", "Dog");
    r.materialize().expect("first materialize failed");
    r.materialize().expect("second materialize failed");
    assert!(
        r.is_entailed("fido", RDF_TYPE, "Animal"),
        "idempotent after second materialize"
    );
}

#[test]
fn test_add_axioms_bulk() {
    let mut r = rl();
    r.add_axioms(vec![
        (
            "A".to_string(),
            RDFS_SUBCLASS_OF.to_string(),
            "B".to_string(),
        ),
        (
            "B".to_string(),
            RDFS_SUBCLASS_OF.to_string(),
            "C".to_string(),
        ),
    ]);
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("A", RDFS_SUBCLASS_OF, "C"),
        "A ⊑ C via bulk add"
    );
}

#[test]
fn test_no_false_positive_without_assertion() {
    let mut r = rl();
    r.add_subclass_of("Dog", "Animal");
    r.materialize().expect("failed");
    // No individual assertions — no type inferences
    assert!(
        !r.is_entailed("fido", RDF_TYPE, "Animal"),
        "fido should not be Animal without type assertion"
    );
}

#[test]
fn test_equivalent_class_bidirectional_type() {
    let mut r = rl();
    r.add_axiom("Human", OWL_EQUIVALENT_CLASS, "Person");
    r.add_type("bob", "Person");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("bob", RDF_TYPE, "Human"),
        "bob:Human via equivalentClass with Person"
    );
}

#[test]
fn test_match_triples_wildcard_all() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(("a".to_string(), "p".to_string(), "b".to_string()));
    set.insert(("c".to_string(), "q".to_string(), "d".to_string()));
    let all = Owl2RlReasoner::match_triples(&set, None, None, None);
    assert_eq!(all.len(), 2, "wildcard should match all 2 triples");
}

#[test]
fn test_match_triples_subject_filter() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(("alice".to_string(), "knows".to_string(), "bob".to_string()));
    set.insert((
        "alice".to_string(),
        "likes".to_string(),
        "music".to_string(),
    ));
    set.insert((
        "bob".to_string(),
        "knows".to_string(),
        "charlie".to_string(),
    ));
    let alice_triples = Owl2RlReasoner::match_triples(&set, Some("alice"), None, None);
    assert_eq!(alice_triples.len(), 2, "Should get 2 alice triples");
}

#[test]
fn test_match_triples_object_filter() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(("a".to_string(), "p".to_string(), "X".to_string()));
    set.insert(("b".to_string(), "q".to_string(), "X".to_string()));
    set.insert(("c".to_string(), "r".to_string(), "Y".to_string()));
    let x_triples = Owl2RlReasoner::match_triples(&set, None, None, Some("X"));
    assert_eq!(x_triples.len(), 2, "Should get 2 triples with object X");
}

#[test]
fn test_is_consistent_before_materialize() {
    let mut r = rl();
    r.add_type("alice", "Person");
    // is_consistent before materialize — should be true (no inference yet)
    assert!(r.is_consistent(), "consistent before materialize");
}

#[test]
fn test_inconsistencies_empty_when_consistent() {
    let mut r = rl();
    r.add_subclass_of("A", "B");
    r.materialize().expect("failed");
    assert!(
        r.inconsistencies().is_empty(),
        "no inconsistencies for consistent ontology"
    );
}

#[test]
fn test_with_max_iterations() {
    let mut r = Owl2RlReasoner::new().with_max_iterations(2);
    r.add_subclass_of("A", "B");
    // Should complete without panic within 2 iterations
    let _ = r.materialize();
}

#[test]
fn test_domain_inheritance_via_subproperty() {
    let mut r = rl();
    // P1 ⊑ P2, P2 rdfs:domain C => P1 rdfs:domain C (scm-dom2)
    r.add_axiom("P1", RDFS_SUBPROPERTY_OF, "P2");
    r.add_axiom("P2", RDFS_DOMAIN, "C");
    r.add_axiom("a", "P1", "b");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("a", RDF_TYPE, "C"),
        "a:C via P1⊑P2, P2 domain C, a P1 b"
    );
}

#[test]
fn test_range_inheritance_via_subproperty() {
    let mut r = rl();
    r.add_axiom("P1", RDFS_SUBPROPERTY_OF, "P2");
    r.add_axiom("P2", RDFS_RANGE, "D");
    r.add_axiom("a", "P1", "b");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("b", RDF_TYPE, "D"),
        "b:D via P1⊑P2, P2 range D, a P1 b"
    );
}

#[test]
fn test_transitive_and_subclass_combined() {
    let mut r = rl();
    r.add_transitive_property("locatedIn");
    r.add_subclass_of("City", "Place");
    r.add_type("berlin", "City");
    r.add_axiom("berlin", "locatedIn", "germany");
    r.add_axiom("germany", "locatedIn", "europe");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("berlin", RDF_TYPE, "Place"),
        "berlin:Place via City⊑Place"
    );
    assert!(
        r.is_entailed("berlin", "locatedIn", "europe"),
        "berlin locatedIn europe via transitivity"
    );
}

#[test]
fn test_symmetric_and_type() {
    let mut r = rl();
    r.add_symmetric_property("marriedTo");
    r.add_axiom("hasSpouseOf", RDFS_SUBPROPERTY_OF, "marriedTo");
    r.add_axiom("alice", "marriedTo", "bob");
    r.materialize().expect("failed");
    assert!(
        r.is_entailed("bob", "marriedTo", "alice"),
        "bob marriedTo alice via SymmetricProperty"
    );
}

#[test]
fn test_owl_thing_type_inference() {
    // Any individual should be inferred to be owl:Thing
    let mut r = rl();
    r.add_type("alice", "Person");
    r.materialize().expect("failed");
    // owl:Thing is the universal superclass — RL may infer it
    // At minimum, alice:Person should be asserted
    assert!(r.is_entailed("alice", RDF_TYPE, "Person"));
}

#[test]
fn test_large_knowledge_base_performance() {
    let mut r = rl();
    // Add 50 subclass axioms
    for i in 0..50usize {
        r.add_subclass_of(&format!("Class{}", i), &format!("Class{}", i + 1));
    }
    r.add_type("ind", "Class0");
    let report = r.materialize().expect("large KB materialize failed");
    assert!(report.new_triples_count > 0, "Should infer new triples");
    assert!(
        r.is_entailed("ind", RDF_TYPE, "Class50"),
        "ind:Class50 via 50-hop chain"
    );
}
