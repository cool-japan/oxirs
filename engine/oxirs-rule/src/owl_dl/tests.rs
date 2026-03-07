//! Test suite for OWL 2 DL reasoner — covers both the 60% baseline and the
//! new 80%-milestone complex class constructor rules.

use super::vocab::*;
use super::*;

fn new_reasoner() -> Owl2DLReasoner {
    Owl2DLReasoner::new()
}

// ── 1. Individual classification via subclass hierarchy ──────────────────────

#[test]
fn test_individual_classification_direct_subclass() {
    let mut r = new_reasoner();
    r.add_subclass_of("Dog", "Mammal");
    r.assert_type("fido", "Dog");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed("fido", "Mammal"),
        "fido should be inferred as Mammal"
    );
}

#[test]
fn test_individual_classification_transitive_subclass() {
    let mut r = new_reasoner();
    r.add_subclass_of("Dog", "Mammal");
    r.add_subclass_of("Mammal", "Animal");
    r.assert_type("fido", "Dog");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed("fido", "Animal"));
    assert!(r.is_type_entailed("fido", "Mammal"));
}

#[test]
fn test_individual_classification_equivalentclass() {
    let mut r = new_reasoner();
    r.add_equivalent_classes("Human", "Person");
    r.assert_type("alice", "Human");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed("alice", "Person"),
        "alice should be Person via equivalentClass"
    );
}

#[test]
fn test_individual_classification_equivalentclass_reverse() {
    let mut r = new_reasoner();
    r.add_equivalent_classes("Human", "Person");
    r.assert_type("alice", "Person");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed("alice", "Human"),
        "alice should be Human via equivalentClass (reverse)"
    );
}

// ── 2. Property chains ────────────────────────────────────────────────────────

#[test]
fn test_property_chain_two_hops() {
    let mut r = new_reasoner();
    r.add_property_chain(
        "hasUncle",
        vec!["hasParent".to_string(), "hasBrother".to_string()],
    );
    r.add_property_assertion(":alice", "hasParent", ":bob");
    r.add_property_assertion(":bob", "hasBrother", ":charlie");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "hasUncle", ":charlie"),
        "alice hasUncle charlie should be inferred"
    );
}

#[test]
fn test_property_chain_three_hops() {
    let mut r = new_reasoner();
    r.add_property_chain(
        "transitiveLocation",
        vec![
            "locatedIn".to_string(),
            "locatedIn".to_string(),
            "locatedIn".to_string(),
        ],
    );
    r.add_property_assertion(":city", "locatedIn", ":region");
    r.add_property_assertion(":region", "locatedIn", ":country");
    r.add_property_assertion(":country", "locatedIn", ":continent");
    r.materialize().expect("materialize ok");
    assert!(r.is_triple_entailed(":city", "transitiveLocation", ":continent"));
}

#[test]
fn test_property_chain_no_spurious_inference() {
    let mut r = new_reasoner();
    r.add_property_chain(
        "hasUncle",
        vec!["hasParent".to_string(), "hasBrother".to_string()],
    );
    r.add_property_assertion(":alice", "hasParent", ":bob");
    // no hasBrother assertion
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_triple_entailed(":alice", "hasUncle", ":bob"),
        "should not infer uncle without hasBrother"
    );
}

// ── 3. Nominal reasoning (owl:oneOf) ─────────────────────────────────────────

#[test]
fn test_nominal_classification_member() {
    let mut r = new_reasoner();
    r.add_nominal_class(
        "PrimaryColors",
        vec!["Red".to_string(), "Green".to_string(), "Blue".to_string()],
    );
    r.add_property_assertion("Red", "label", "red_color");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed("Red", "PrimaryColors"),
        "Red should be PrimaryColors member"
    );
}

#[test]
fn test_nominal_classification_non_member_excluded() {
    let mut r = new_reasoner();
    r.add_nominal_class("Planets", vec!["Mars".to_string(), "Venus".to_string()]);
    r.add_property_assertion("Earth", "label", "third_planet");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed("Earth", "Planets"),
        "Earth is not in the Planets enumeration"
    );
}

// ── 4. HasValue reasoning ─────────────────────────────────────────────────────

#[test]
fn test_has_value_backward_classification() {
    let mut r = new_reasoner();
    r.add_has_value_restriction("ParentOfBob", "hasChild", ":bob");
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "ParentOfBob"),
        "alice should be ParentOfBob via hasValue backward"
    );
}

#[test]
fn test_has_value_forward_property_assertion() {
    let mut r = new_reasoner();
    r.add_has_value_restriction("TofuEater", "eats", ":tofu");
    r.assert_type(":alice", "TofuEater");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "eats", ":tofu"),
        "TofuEater member should eat tofu"
    );
}

#[test]
fn test_has_value_no_match_different_value() {
    let mut r = new_reasoner();
    r.add_has_value_restriction("ParentOfBob", "hasChild", ":bob");
    r.add_property_assertion(":alice", "hasChild", ":charlie");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", "ParentOfBob"),
        "wrong value — no classification expected"
    );
}

// ── 5. AllValuesFrom ─────────────────────────────────────────────────────────

#[test]
fn test_all_values_from_classification() {
    let mut r = new_reasoner();
    r.add_all_values_from_restriction("AllDogOwner", "hasPet", "Dog");
    r.assert_type(":alice", "AllDogOwner");
    r.add_property_assertion(":alice", "hasPet", ":fido");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":fido", "Dog"),
        "fido should be inferred Dog via allValuesFrom"
    );
}

#[test]
fn test_all_values_from_multiple_fillers() {
    let mut r = new_reasoner();
    r.add_all_values_from_restriction("AllAnimalOwner", "hasPet", "Animal");
    r.assert_type(":alice", "AllAnimalOwner");
    r.add_property_assertion(":alice", "hasPet", ":fido");
    r.add_property_assertion(":alice", "hasPet", ":kitty");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed(":fido", "Animal"));
    assert!(r.is_type_entailed(":kitty", "Animal"));
}

// ── 6. SomeValuesFrom ────────────────────────────────────────────────────────

#[test]
fn test_some_values_from_classification() {
    let mut r = new_reasoner();
    r.add_some_values_from_restriction("PetOwner", "hasPet", "Animal");
    r.assert_type(":fido", "Animal");
    r.add_property_assertion(":alice", "hasPet", ":fido");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "PetOwner"),
        "alice is PetOwner via someValuesFrom"
    );
}

#[test]
fn test_some_values_from_requires_correct_filler_type() {
    let mut r = new_reasoner();
    r.add_some_values_from_restriction("DogOwner", "hasPet", "Dog");
    r.assert_type(":fido", "Cat"); // wrong type
    r.add_property_assertion(":alice", "hasPet", ":fido");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", "DogOwner"),
        "fido is Cat not Dog — should not classify alice as DogOwner"
    );
}

// ── 7. Transitivity ──────────────────────────────────────────────────────────

#[test]
fn test_transitivity_two_hops() {
    let mut r = new_reasoner();
    r.add_transitive_property("ancestorOf");
    r.add_property_assertion(":grandparent", "ancestorOf", ":parent");
    r.add_property_assertion(":parent", "ancestorOf", ":child");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":grandparent", "ancestorOf", ":child"),
        "grandparent ancestorOf child via transitivity"
    );
}

#[test]
fn test_transitivity_three_hops() {
    let mut r = new_reasoner();
    r.add_transitive_property("partOf");
    r.add_property_assertion(":leg", "partOf", ":chair");
    r.add_property_assertion(":chair", "partOf", ":room");
    r.add_property_assertion(":room", "partOf", ":building");
    r.materialize().expect("materialize ok");
    assert!(r.is_triple_entailed(":leg", "partOf", ":building"));
    assert!(r.is_triple_entailed(":leg", "partOf", ":room"));
    assert!(r.is_triple_entailed(":chair", "partOf", ":building"));
}

// ── 8. Symmetry ──────────────────────────────────────────────────────────────

#[test]
fn test_symmetric_property() {
    let mut r = new_reasoner();
    r.add_symmetric_property("knows");
    r.add_property_assertion(":alice", "knows", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":bob", "knows", ":alice"),
        "symmetry: bob knows alice"
    );
}

#[test]
fn test_symmetric_property_not_applied_to_other_props() {
    let mut r = new_reasoner();
    r.add_symmetric_property("sibling");
    r.add_property_assertion(":alice", "hasParent", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_triple_entailed(":bob", "hasParent", ":alice"),
        "hasParent is not symmetric"
    );
}

// ── 9. Asymmetry ─────────────────────────────────────────────────────────────

#[test]
fn test_asymmetric_property_consistent() {
    let mut r = new_reasoner();
    r.add_asymmetric_property("hasChild");
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize ok");
    assert!(r.is_consistent());
}

#[test]
fn test_asymmetric_property_inconsistency_detected() {
    let mut r = new_reasoner();
    r.add_asymmetric_property("strictlyLessThan");
    r.add_property_assertion(":a", "strictlyLessThan", ":b");
    r.add_property_assertion(":b", "strictlyLessThan", ":a"); // violates asymmetry
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "both a < b and b < a violates AsymmetricProperty"
    );
    assert!(!r.inconsistencies().is_empty());
}

// ── 10. Domain / Range inference ─────────────────────────────────────────────

#[test]
fn test_domain_inference() {
    let mut r = new_reasoner();
    r.add_domain("hasChild", "Person");
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "Person"),
        "alice should be Person via domain"
    );
}

#[test]
fn test_range_inference() {
    let mut r = new_reasoner();
    r.add_range("hasChild", "Person");
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":bob", "Person"),
        "bob should be Person via range"
    );
}

// ── 11. Disjoint class inconsistency ─────────────────────────────────────────

#[test]
fn test_disjoint_class_inconsistency() {
    let mut r = new_reasoner();
    r.add_disjoint_classes("Cat", "Dog");
    r.assert_type(":fido", "Dog");
    r.assert_type(":fido", "Cat");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_consistent(),
        "fido cannot be both Cat and Dog (disjoint)"
    );
}

#[test]
fn test_disjoint_no_violation() {
    let mut r = new_reasoner();
    r.add_disjoint_classes("Cat", "Dog");
    r.assert_type(":fido", "Dog");
    r.assert_type(":whiskers", "Cat");
    r.materialize().expect("materialize ok");
    assert!(r.is_consistent());
}

// ── 12. sameAs propagation ───────────────────────────────────────────────────

#[test]
fn test_same_as_type_propagation() {
    let mut r = new_reasoner();
    r.assert_same_as(":alice", ":alicia");
    r.assert_type(":alice", "Person");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alicia", "Person"),
        "alicia should inherit Person type via sameAs"
    );
}

// ── 13. InverseOf ────────────────────────────────────────────────────────────

#[test]
fn test_inverse_of_inference() {
    let mut r = new_reasoner();
    r.add_inverse_of("hasParent", "hasChild");
    r.add_property_assertion(":alice", "hasParent", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":bob", "hasChild", ":alice"),
        "bob hasChild alice via inverseOf"
    );
}

// ── 14. Combined scenario ─────────────────────────────────────────────────────

#[test]
fn test_combined_restriction_and_subclass() {
    let mut r = new_reasoner();
    r.add_subclass_of("Dog", "Animal");
    r.add_some_values_from_restriction("PetOwner", "hasPet", "Animal");
    r.assert_type(":fido", "Dog");
    r.add_property_assertion(":alice", "hasPet", ":fido");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed(":fido", "Animal"));
    assert!(
        r.is_type_entailed(":alice", "PetOwner"),
        "alice is PetOwner via someValuesFrom + subclass"
    );
}

#[test]
fn test_report_statistics() {
    let mut r = new_reasoner();
    r.add_subclass_of("Dog", "Animal");
    r.assert_type(":fido", "Dog");
    let report = r.materialize().expect("materialize ok");
    assert!(report.iterations >= 1);
    assert!(report.new_triples >= 1);
    assert!(report.rule_firings.subclass_propagation >= 1);
}

#[test]
fn test_max_iterations_limit() {
    let mut r = Owl2DLReasoner::new().with_max_iterations(1);
    r.add_subclass_of("A", "B");
    r.add_subclass_of("B", "C");
    r.add_subclass_of("C", "D");
    r.assert_type(":x", "A");
    let _result = r.materialize();
}

#[test]
fn test_inferred_triples_excludes_asserted() {
    let mut r = new_reasoner();
    r.add_subclass_of("Dog", "Animal");
    r.assert_type(":fido", "Dog");
    r.materialize().expect("materialize ok");
    let inferred = r.inferred_triples();
    assert!(inferred
        .iter()
        .any(|(s, p, o)| s == ":fido" && p == RDF_TYPE && o == "Animal"));
}

#[test]
fn test_reflexive_property_declaration() {
    let mut r = new_reasoner();
    r.add_reflexive_property("knows");
    r.assert_type(":alice", "Person");
    r.materialize().expect("materialize ok");
    assert!(r.is_consistent());
}

// ═══════════════════════════════════════════════════════════════════════════════
// New tests for the 80% milestone features
// ═══════════════════════════════════════════════════════════════════════════════

// ── 15. ObjectComplementOf ───────────────────────────────────────────────────

#[test]
fn test_complement_of_inconsistency_detected() {
    let mut r = new_reasoner();
    // NotAnimal ≡ owl:complementOf Animal
    r.add_complement_of("NotAnimal", "Animal");
    // Contradiction: fido is both Animal and NotAnimal
    r.assert_type(":fido", "Animal");
    r.assert_type(":fido", "NotAnimal");
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "fido cannot be both Animal and NotAnimal (complement)"
    );
    assert!(
        r.inconsistencies()
            .iter()
            .any(|msg| msg.contains("ComplementOf")),
        "expected ComplementOf inconsistency message"
    );
}

#[test]
fn test_complement_of_no_inconsistency_when_consistent() {
    let mut r = new_reasoner();
    r.add_complement_of("NotAnimal", "Animal");
    // fido is Animal but NOT NotAnimal — consistent
    r.assert_type(":fido", "Animal");
    r.assert_type(":robot", "NotAnimal");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_consistent(),
        "different individuals in complementary classes should be consistent"
    );
}

#[test]
fn test_complement_of_multiple_individuals() {
    let mut r = new_reasoner();
    r.add_complement_of("NotPerson", "Person");
    r.assert_type(":alice", "Person");
    r.assert_type(":alice", "NotPerson"); // violation
    r.assert_type(":robot", "NotPerson"); // fine — robot is not a Person
    r.materialize().expect("materialize call ok");
    assert!(!r.is_consistent());
    // Should only have one inconsistency (alice), not for robot
    let alice_violations: Vec<_> = r
        .inconsistencies()
        .iter()
        .filter(|m| m.contains(":alice"))
        .collect();
    assert_eq!(alice_violations.len(), 1, "exactly one violation for alice");
}

// ── 16. ObjectIntersectionOf ─────────────────────────────────────────────────

#[test]
fn test_intersection_of_forward_classification() {
    let mut r = new_reasoner();
    // ParentAndEmployee ≡ Parent ∩ Employee
    r.add_intersection_of(
        "ParentAndEmployee",
        vec!["Parent".to_string(), "Employee".to_string()],
    );
    r.assert_type(":alice", "Parent");
    r.assert_type(":alice", "Employee");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "ParentAndEmployee"),
        "alice is Parent ∩ Employee so should be ParentAndEmployee"
    );
}

#[test]
fn test_intersection_of_backward_unfolding() {
    let mut r = new_reasoner();
    // RichAthlete ≡ Rich ∩ Athlete
    r.add_intersection_of(
        "RichAthlete",
        vec!["Rich".to_string(), "Athlete".to_string()],
    );
    // alice is directly classified as RichAthlete
    r.assert_type(":alice", "RichAthlete");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "Rich"),
        "alice (RichAthlete) should be inferred as Rich"
    );
    assert!(
        r.is_type_entailed(":alice", "Athlete"),
        "alice (RichAthlete) should be inferred as Athlete"
    );
}

#[test]
fn test_intersection_of_partial_operands_no_forward() {
    let mut r = new_reasoner();
    r.add_intersection_of(
        "ParentAndEmployee",
        vec!["Parent".to_string(), "Employee".to_string()],
    );
    // alice is only Parent, not Employee
    r.assert_type(":alice", "Parent");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", "ParentAndEmployee"),
        "alice is only Parent, not both operands → no forward classification"
    );
}

#[test]
fn test_intersection_of_three_operands() {
    let mut r = new_reasoner();
    r.add_intersection_of(
        "TripleCombo",
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    );
    r.assert_type(":x", "A");
    r.assert_type(":x", "B");
    r.assert_type(":x", "C");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed(":x", "TripleCombo"));
}

// ── 17. DisjointUnionOf ──────────────────────────────────────────────────────

#[test]
fn test_disjoint_union_subclass_inference() {
    let mut r = new_reasoner();
    // Animal owl:disjointUnionOf (Cat Dog Bird)
    r.add_disjoint_union(
        "Animal",
        vec!["Cat".to_string(), "Dog".to_string(), "Bird".to_string()],
    );
    r.assert_type(":fido", "Dog");
    r.materialize().expect("materialize ok");
    // Dog ⊑ Animal, so fido should be Animal
    assert!(
        r.is_type_entailed(":fido", "Animal"),
        "fido (Dog) should be Animal via disjointUnion subclass"
    );
}

#[test]
fn test_disjoint_union_member_disjointness_violation() {
    let mut r = new_reasoner();
    r.add_disjoint_union("Animal", vec!["Cat".to_string(), "Dog".to_string()]);
    // :hybrid is both Cat and Dog — violates disjointness
    r.assert_type(":hybrid", "Cat");
    r.assert_type(":hybrid", "Dog");
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "hybrid cannot be both Cat and Dog (disjointUnion operands are disjoint)"
    );
}

#[test]
fn test_disjoint_union_non_violating_individuals() {
    let mut r = new_reasoner();
    r.add_disjoint_union(
        "Shape",
        vec![
            "Circle".to_string(),
            "Square".to_string(),
            "Triangle".to_string(),
        ],
    );
    r.assert_type(":s1", "Circle");
    r.assert_type(":s2", "Square");
    r.assert_type(":s3", "Triangle");
    r.materialize().expect("materialize ok");
    // All belong to Shape via subclass
    assert!(r.is_type_entailed(":s1", "Shape"));
    assert!(r.is_type_entailed(":s2", "Shape"));
    assert!(r.is_type_entailed(":s3", "Shape"));
    // No violations
    assert!(r.is_consistent());
}

// ── 18. HasKey ────────────────────────────────────────────────────────────────

#[test]
fn test_has_key_same_as_inferred() {
    let mut r = new_reasoner();
    // Person owl:hasKey (ssn)
    r.add_has_key("Person", vec!["ssn".to_string()]);
    r.assert_type(":alice", "Person");
    r.assert_type(":alicia", "Person");
    // Both have same SSN
    r.add_property_assertion(":alice", "ssn", "123-45-6789");
    r.add_property_assertion(":alicia", "ssn", "123-45-6789");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", OWL_SAME_AS, ":alicia"),
        "alice and alicia share SSN — must be sameAs"
    );
    assert!(
        r.is_triple_entailed(":alicia", OWL_SAME_AS, ":alice"),
        "sameAs is symmetric"
    );
}

#[test]
fn test_has_key_different_values_no_same_as() {
    let mut r = new_reasoner();
    r.add_has_key("Person", vec!["ssn".to_string()]);
    r.assert_type(":alice", "Person");
    r.assert_type(":bob", "Person");
    r.add_property_assertion(":alice", "ssn", "111-11-1111");
    r.add_property_assertion(":bob", "ssn", "222-22-2222");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_triple_entailed(":alice", OWL_SAME_AS, ":bob"),
        "different SSNs → no sameAs"
    );
}

#[test]
fn test_has_key_composite_key_both_properties_must_match() {
    let mut r = new_reasoner();
    // Order owl:hasKey (orderDate customerId)
    r.add_has_key(
        "Order",
        vec!["orderDate".to_string(), "customerId".to_string()],
    );
    r.assert_type(":o1", "Order");
    r.assert_type(":o2", "Order");
    // Same date but different customer — NOT same individual
    r.add_property_assertion(":o1", "orderDate", "2024-01-15");
    r.add_property_assertion(":o1", "customerId", "C001");
    r.add_property_assertion(":o2", "orderDate", "2024-01-15");
    r.add_property_assertion(":o2", "customerId", "C002");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_triple_entailed(":o1", OWL_SAME_AS, ":o2"),
        "different customerId → not sameAs"
    );
}

#[test]
fn test_has_key_composite_key_both_match() {
    let mut r = new_reasoner();
    r.add_has_key(
        "Order",
        vec!["orderDate".to_string(), "customerId".to_string()],
    );
    r.assert_type(":o1", "Order");
    r.assert_type(":o2", "Order");
    r.add_property_assertion(":o1", "orderDate", "2024-01-15");
    r.add_property_assertion(":o1", "customerId", "C001");
    r.add_property_assertion(":o2", "orderDate", "2024-01-15");
    r.add_property_assertion(":o2", "customerId", "C001");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":o1", OWL_SAME_AS, ":o2"),
        "identical composite key → sameAs"
    );
}

#[test]
fn test_has_key_missing_key_property_excluded() {
    let mut r = new_reasoner();
    r.add_has_key("Person", vec!["ssn".to_string()]);
    r.assert_type(":alice", "Person");
    r.assert_type(":unknown", "Person");
    // :unknown has no ssn — should not be merged
    r.add_property_assertion(":alice", "ssn", "111");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_triple_entailed(":alice", OWL_SAME_AS, ":unknown"),
        "unknown lacks SSN — no sameAs"
    );
}

// ── 19. FunctionalProperty ───────────────────────────────────────────────────

#[test]
fn test_functional_property_same_as_inferred() {
    let mut r = new_reasoner();
    // hasBiologicalMother is functional
    r.add_functional_property("hasBiologicalMother");
    // alice has two mothers asserted — they must be sameAs
    r.add_property_assertion(":alice", "hasBiologicalMother", ":mary");
    r.add_property_assertion(":alice", "hasBiologicalMother", ":maria");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":mary", OWL_SAME_AS, ":maria"),
        "two values for functional property → sameAs"
    );
    assert!(r.is_triple_entailed(":maria", OWL_SAME_AS, ":mary"));
}

#[test]
fn test_functional_property_single_value_no_same_as() {
    let mut r = new_reasoner();
    r.add_functional_property("hasBiologicalMother");
    r.add_property_assertion(":alice", "hasBiologicalMother", ":mary");
    r.materialize().expect("materialize ok");
    // Only one value — no sameAs entailment
    assert!(
        !r.is_triple_entailed(":mary", OWL_SAME_AS, ":mary"),
        "self-sameAs should not be generated for single value"
    );
}

#[test]
fn test_functional_property_multiple_subjects_independent() {
    let mut r = new_reasoner();
    r.add_functional_property("hasSSN");
    // Different subjects with one value each — no sameAs
    r.add_property_assertion(":alice", "hasSSN", "111");
    r.add_property_assertion(":bob", "hasSSN", "222");
    r.materialize().expect("materialize ok");
    assert!(!r.is_triple_entailed(":alice", OWL_SAME_AS, ":bob"));
}

// ── 20. InverseFunctionalProperty ────────────────────────────────────────────

#[test]
fn test_inverse_functional_property_same_as_inferred() {
    let mut r = new_reasoner();
    // hasSSN is inverse-functional (unique identifier)
    r.add_inverse_functional_property("hasSSN");
    r.add_property_assertion(":alice", "hasSSN", "111-22-3333");
    r.add_property_assertion(":alicia", "hasSSN", "111-22-3333");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", OWL_SAME_AS, ":alicia"),
        "same SSN value for two subjects → sameAs via InverseFunctional"
    );
    assert!(r.is_triple_entailed(":alicia", OWL_SAME_AS, ":alice"));
}

#[test]
fn test_inverse_functional_property_different_values_no_same_as() {
    let mut r = new_reasoner();
    r.add_inverse_functional_property("hasSSN");
    r.add_property_assertion(":alice", "hasSSN", "111-11-1111");
    r.add_property_assertion(":bob", "hasSSN", "222-22-2222");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_triple_entailed(":alice", OWL_SAME_AS, ":bob"),
        "different SSN values → no sameAs"
    );
}

#[test]
fn test_inverse_functional_property_three_subjects_same_value() {
    let mut r = new_reasoner();
    r.add_inverse_functional_property("email");
    r.add_property_assertion(":a", "email", "x@example.com");
    r.add_property_assertion(":b", "email", "x@example.com");
    r.add_property_assertion(":c", "email", "x@example.com");
    r.materialize().expect("materialize ok");
    // All three are sameAs each other
    assert!(r.is_triple_entailed(":a", OWL_SAME_AS, ":b"));
    assert!(r.is_triple_entailed(":a", OWL_SAME_AS, ":c"));
    assert!(r.is_triple_entailed(":b", OWL_SAME_AS, ":c"));
}

// ── 21. Integrated scenarios (cross-feature interactions) ─────────────────────

#[test]
fn test_intersection_then_subclass_propagation() {
    let mut r = new_reasoner();
    // WorkingParent ≡ Worker ∩ Parent; Worker ⊑ Adult
    r.add_intersection_of(
        "WorkingParent",
        vec!["Worker".to_string(), "Parent".to_string()],
    );
    r.add_subclass_of("Worker", "Adult");
    r.assert_type(":alice", "WorkingParent");
    r.materialize().expect("materialize ok");
    // Backward: alice is WorkingParent → alice is Worker and Parent
    assert!(r.is_type_entailed(":alice", "Worker"));
    assert!(r.is_type_entailed(":alice", "Parent"));
    // Subclass propagation: alice is Worker → alice is Adult
    assert!(
        r.is_type_entailed(":alice", "Adult"),
        "WorkingParent → Worker → Adult via subclass"
    );
}

#[test]
fn test_functional_property_then_same_as_type_propagation() {
    let mut r = new_reasoner();
    r.add_functional_property("hasMentor");
    r.add_subclass_of("Professor", "AcademicStaff");
    // alice has two mentors (functional violation → they are sameAs)
    r.add_property_assertion(":alice", "hasMentor", ":prof1");
    r.add_property_assertion(":alice", "hasMentor", ":prof2");
    // prof1 is a Professor
    r.assert_type(":prof1", "Professor");
    r.materialize().expect("materialize ok");
    // prof1 sameAs prof2 → prof2 inherits Professor type
    assert!(
        r.is_triple_entailed(":prof1", OWL_SAME_AS, ":prof2"),
        "functional prop merges prof1 and prof2"
    );
    assert!(
        r.is_type_entailed(":prof2", "Professor"),
        "prof2 inherits Professor via sameAs propagation"
    );
    assert!(
        r.is_type_entailed(":prof2", "AcademicStaff"),
        "prof2 inherits AcademicStaff via sameAs + subclass"
    );
}

#[test]
fn test_has_key_then_same_as_type_propagation() {
    let mut r = new_reasoner();
    r.add_has_key("Employee", vec!["employeeId".to_string()]);
    r.add_subclass_of("Manager", "Employee");
    r.assert_type(":emp1", "Employee");
    r.assert_type(":emp2", "Employee");
    r.assert_type(":emp1", "Manager"); // emp1 is also a manager
    r.add_property_assertion(":emp1", "employeeId", "E001");
    r.add_property_assertion(":emp2", "employeeId", "E001");
    r.materialize().expect("materialize ok");
    // emp1 sameAs emp2 (same key)
    assert!(r.is_triple_entailed(":emp1", OWL_SAME_AS, ":emp2"));
    // emp2 should inherit Manager type via sameAs
    assert!(
        r.is_type_entailed(":emp2", "Manager"),
        "emp2 inherits Manager from emp1 via sameAs + type propagation"
    );
}

#[test]
fn test_disjoint_union_plus_complement_double_inconsistency() {
    let mut r = new_reasoner();
    // Direction owl:disjointUnionOf (Left Right)
    r.add_disjoint_union("Direction", vec!["Left".to_string(), "Right".to_string()]);
    // NotLeft ≡ complementOf Left
    r.add_complement_of("NotLeft", "Left");
    // :x is both Left and Right — violates disjointUnion
    r.assert_type(":x", "Left");
    r.assert_type(":x", "Right");
    // :y is both Left and NotLeft — violates complement
    r.assert_type(":y", "Left");
    r.assert_type(":y", "NotLeft");
    r.materialize().expect("materialize call ok");
    assert!(!r.is_consistent());
    assert!(
        r.inconsistencies().len() >= 2,
        "should have at least 2 inconsistencies"
    );
}

#[test]
fn test_add_inverse_functional_property_declaration() {
    let mut r = new_reasoner();
    r.add_inverse_functional_property("nationalId");
    // Check property is tracked
    assert!(
        r.property_chars
            .get("nationalId")
            .map(|c| c.is_inverse_functional)
            .unwrap_or(false),
        "nationalId should be marked inverse-functional"
    );
}

#[test]
fn test_intersection_of_with_disjoint_union_subclass() {
    let mut r = new_reasoner();
    // FlyingAnimal owl:disjointUnionOf (Bird Bat)
    r.add_disjoint_union("FlyingAnimal", vec!["Bird".to_string(), "Bat".to_string()]);
    // FlyingVertebrate ≡ FlyingAnimal ∩ Vertebrate
    r.add_intersection_of(
        "FlyingVertebrate",
        vec!["FlyingAnimal".to_string(), "Vertebrate".to_string()],
    );
    r.assert_type(":tweety", "Bird");
    r.assert_type(":tweety", "Vertebrate");
    r.materialize().expect("materialize ok");
    // tweety is Bird → (via disjointUnion subclass) FlyingAnimal
    assert!(r.is_type_entailed(":tweety", "FlyingAnimal"));
    // tweety is FlyingAnimal ∩ Vertebrate → FlyingVertebrate
    assert!(
        r.is_type_entailed(":tweety", "FlyingVertebrate"),
        "tweety qualifies as FlyingVertebrate via disjointUnion + intersection"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 90% milestone features — tests for the eight new advanced role/property rules
// ═══════════════════════════════════════════════════════════════════════════════

// ── 22. SubObjectPropertyOf ───────────────────────────────────────────────────

#[test]
fn test_sub_object_property_basic() {
    let mut r = new_reasoner();
    // hasChild ⊑ hasRelative: if x hasChild y then x hasRelative y
    r.add_sub_object_property_of("hasChild", "hasRelative");
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "hasRelative", ":bob"),
        "alice hasChild bob → alice hasRelative bob via sub-property"
    );
}

#[test]
fn test_sub_object_property_chain_transitive_closure() {
    let mut r = new_reasoner();
    // P1 ⊑ P2, P2 ⊑ P3: x P1 y should infer x P2 y AND x P3 y
    r.add_sub_object_property_of("P1", "P2");
    r.add_sub_object_property_of("P2", "P3");
    r.add_property_assertion(":x", "P1", ":y");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":x", "P2", ":y"),
        "x P1 y → x P2 y via direct sub-property"
    );
    assert!(
        r.is_triple_entailed(":x", "P3", ":y"),
        "x P1 y → x P3 y via transitive closure of sub-property chain"
    );
}

#[test]
fn test_sub_object_property_no_spurious_inference_wrong_direction() {
    let mut r = new_reasoner();
    // hasChild ⊑ hasRelative (NOT hasRelative ⊑ hasChild)
    r.add_sub_object_property_of("hasChild", "hasRelative");
    // Assert hasRelative but NOT hasChild — should NOT infer hasChild
    r.add_property_assertion(":alice", "hasRelative", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_triple_entailed(":alice", "hasChild", ":bob"),
        "sub-property is not symmetric — hasRelative does not imply hasChild"
    );
}

#[test]
fn test_sub_object_property_combined_with_domain() {
    let mut r = new_reasoner();
    // hasDirectChild ⊑ hasChild, domain(hasChild) = Parent
    r.add_sub_object_property_of("hasDirectChild", "hasChild");
    r.add_domain("hasChild", "Parent");
    r.add_property_assertion(":alice", "hasDirectChild", ":bob");
    r.materialize().expect("materialize ok");
    // Step 1: alice hasDirectChild bob → alice hasChild bob (sub-property)
    assert!(r.is_triple_entailed(":alice", "hasChild", ":bob"));
    // Step 2: alice hasChild bob, domain(hasChild)=Parent → alice rdf:type Parent
    assert!(
        r.is_type_entailed(":alice", "Parent"),
        "domain inference fires on the inferred hasChild triple"
    );
}

#[test]
fn test_sub_object_property_multiple_assertions() {
    let mut r = new_reasoner();
    r.add_sub_object_property_of("knows", "sociallyRelatedTo");
    r.add_property_assertion(":alice", "knows", ":bob");
    r.add_property_assertion(":bob", "knows", ":charlie");
    r.materialize().expect("materialize ok");
    assert!(r.is_triple_entailed(":alice", "sociallyRelatedTo", ":bob"));
    assert!(r.is_triple_entailed(":bob", "sociallyRelatedTo", ":charlie"));
    // alice does NOT automatically socially-relate charlie (no transitivity here)
    assert!(!r.is_triple_entailed(":alice", "sociallyRelatedTo", ":charlie"));
}

// ── 23. SubDataPropertyOf ────────────────────────────────────────────────────

#[test]
fn test_sub_data_property_basic() {
    let mut r = new_reasoner();
    // officialName ⊑ name: if x officialName v then x name v
    r.add_sub_data_property_of("officialName", "name");
    r.add_property_assertion(":acme", "officialName", "ACME Corp.");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":acme", "name", "ACME Corp."),
        "data sub-property: officialName ⊑ name, so name is inferred"
    );
}

#[test]
fn test_sub_data_property_multiple_supers() {
    let mut r = new_reasoner();
    // legalName ⊑ name, legalName ⊑ identifier
    r.add_sub_data_property_of("legalName", "name");
    r.add_sub_data_property_of("legalName", "identifier");
    r.add_property_assertion(":firm", "legalName", "FirmX");
    r.materialize().expect("materialize ok");
    assert!(r.is_triple_entailed(":firm", "name", "FirmX"));
    assert!(
        r.is_triple_entailed(":firm", "identifier", "FirmX"),
        "legalName ⊑ identifier inferred"
    );
}

// ── 24. EquivalentObjectProperties ────────────────────────────────────────────

#[test]
fn test_equivalent_properties_forward() {
    let mut r = new_reasoner();
    // spouse ≡ partner
    r.add_equivalent_properties("spouse", "partner");
    r.add_property_assertion(":alice", "spouse", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "partner", ":bob"),
        "spouse ≡ partner: alice spouse bob → alice partner bob"
    );
}

#[test]
fn test_equivalent_properties_reverse() {
    let mut r = new_reasoner();
    // spouse ≡ partner (bidirectional)
    r.add_equivalent_properties("spouse", "partner");
    r.add_property_assertion(":alice", "partner", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "spouse", ":bob"),
        "partner ≡ spouse: alice partner bob → alice spouse bob (reverse direction)"
    );
}

#[test]
fn test_equivalent_properties_combined_with_range() {
    let mut r = new_reasoner();
    // loc ≡ locatedIn, range(locatedIn) = Place
    r.add_equivalent_properties("loc", "locatedIn");
    r.add_range("locatedIn", "Place");
    r.add_property_assertion(":berlin", "loc", ":germany");
    r.materialize().expect("materialize ok");
    // berlin loc germany → berlin locatedIn germany (via equivalence)
    assert!(r.is_triple_entailed(":berlin", "locatedIn", ":germany"));
    // range(locatedIn) = Place → germany rdf:type Place
    assert!(
        r.is_type_entailed(":germany", "Place"),
        "range fires after equivalent property expansion"
    );
}

#[test]
fn test_equivalent_properties_no_cross_contamination() {
    let mut r = new_reasoner();
    // P1 ≡ P2, P3 ≡ P4 — no inference from P1 to P3 or P4
    r.add_equivalent_properties("P1", "P2");
    r.add_equivalent_properties("P3", "P4");
    r.add_property_assertion(":x", "P1", ":y");
    r.materialize().expect("materialize ok");
    assert!(!r.is_triple_entailed(":x", "P3", ":y"));
    assert!(!r.is_triple_entailed(":x", "P4", ":y"));
}

// ── 25. DisjointObjectProperties ─────────────────────────────────────────────

#[test]
fn test_disjoint_properties_inconsistency() {
    let mut r = new_reasoner();
    // parentOf disjointWith childOf
    r.add_disjoint_properties("parentOf", "childOf");
    // Both hold for the same pair: violation
    r.add_property_assertion(":alice", "parentOf", ":bob");
    r.add_property_assertion(":alice", "childOf", ":bob");
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "alice parentOf bob AND alice childOf bob violates disjoint properties"
    );
    assert!(
        r.inconsistencies()
            .iter()
            .any(|m| m.contains("DisjointProperties")),
        "expected DisjointProperties inconsistency message"
    );
}

#[test]
fn test_disjoint_properties_consistent_different_objects() {
    let mut r = new_reasoner();
    r.add_disjoint_properties("P1", "P2");
    // Same subject, different objects — NO violation (they apply to different (s, o) pairs)
    r.add_property_assertion(":x", "P1", ":y");
    r.add_property_assertion(":x", "P2", ":z");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_consistent(),
        "different objects for disjoint properties — not a violation"
    );
}

#[test]
fn test_disjoint_properties_consistent_different_subjects() {
    let mut r = new_reasoner();
    r.add_disjoint_properties("P1", "P2");
    // Different subjects pointing to same object — NOT a violation
    r.add_property_assertion(":x", "P1", ":z");
    r.add_property_assertion(":y", "P2", ":z");
    r.materialize().expect("materialize ok");
    assert!(r.is_consistent());
}

#[test]
fn test_disjoint_properties_violation_via_equivalent_expansion() {
    let mut r = new_reasoner();
    // P1 ≡ P2 (equivalent), P2 disjointWith P3
    r.add_equivalent_properties("P1", "P2");
    r.add_disjoint_properties("P2", "P3");
    // Assert P1 and P3 for same (x, y) pair — after expansion P2 is also inferred
    r.add_property_assertion(":x", "P1", ":y");
    r.add_property_assertion(":x", "P3", ":y");
    r.materialize().expect("materialize call ok");
    // P1 → P2 (via equivalence), then P2 disjoint P3 both hold → inconsistency
    assert!(
        !r.is_consistent(),
        "P1 ≡ P2 and P2 disjointWith P3: asserting P1 and P3 for same pair → inconsistency"
    );
}

// ── 26. ReflexiveObjectProperty ───────────────────────────────────────────────

#[test]
fn test_reflexive_property_self_loop_inferred() {
    let mut r = new_reasoner();
    r.add_reflexive_property("relatedTo");
    // alice exists in the ABox as a typed individual
    r.assert_type(":alice", "Person");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "relatedTo", ":alice"),
        "reflexive property: alice relatedTo alice should be inferred"
    );
}

#[test]
fn test_reflexive_property_multiple_individuals() {
    let mut r = new_reasoner();
    r.add_reflexive_property("coexists");
    r.assert_type(":alice", "Person");
    r.assert_type(":bob", "Animal");
    r.assert_type(":thing", "Object");
    r.materialize().expect("materialize ok");
    assert!(r.is_triple_entailed(":alice", "coexists", ":alice"));
    assert!(r.is_triple_entailed(":bob", "coexists", ":bob"));
    assert!(r.is_triple_entailed(":thing", "coexists", ":thing"));
}

#[test]
fn test_reflexive_and_irreflexive_independent_properties() {
    let mut r = new_reasoner();
    // knows is reflexive; strictlyLessThan is irreflexive — they are independent
    r.add_reflexive_property("knows");
    r.add_irreflexive_property("strictlyLessThan");
    r.assert_type(":alice", "Person");
    r.add_property_assertion(":a", "strictlyLessThan", ":b");
    r.materialize().expect("materialize ok");
    // reflexive self-loop for knows is fine
    assert!(r.is_triple_entailed(":alice", "knows", ":alice"));
    // irreflexive property has no self-loop, so no inconsistency
    assert!(r.is_consistent());
}

#[test]
fn test_irreflexive_violation_from_reflexive_inference() {
    let mut r = new_reasoner();
    // P is declared BOTH reflexive and irreflexive — contradictory for any individual
    r.add_reflexive_property("P");
    r.add_irreflexive_property("P");
    // alice exists — reflexive rule infers alice P alice,
    // then irreflexivity check fires → inconsistency
    r.assert_type(":alice", "Person");
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "P is both reflexive and irreflexive — alice P alice inferred then detected as violation"
    );
}

// ── 27. NegativePropertyAssertion ────────────────────────────────────────────

#[test]
fn test_negative_object_property_consistent_no_triple() {
    let mut r = new_reasoner();
    // Declare that alice does NOT have bob as a child
    r.assert_negative_object_property_assertion(":alice", "hasChild", ":bob");
    // No actual (alice, hasChild, bob) triple — should be consistent
    r.add_property_assertion(":alice", "hasChild", ":charlie"); // different target
    r.materialize().expect("materialize ok");
    assert!(
        r.is_consistent(),
        "negative assertion not violated — alice hasChild charlie, not bob"
    );
}

#[test]
fn test_negative_object_property_inconsistency_when_triple_exists() {
    let mut r = new_reasoner();
    r.assert_negative_object_property_assertion(":alice", "hasChild", ":bob");
    // Now explicitly assert the very triple that is declared negative
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "alice hasChild bob is asserted but declared as negative property assertion"
    );
    assert!(
        r.inconsistencies()
            .iter()
            .any(|m| m.contains("NegativePropertyAssertion")),
        "expected NegativePropertyAssertion in inconsistencies"
    );
}

#[test]
fn test_negative_data_property_consistent() {
    let mut r = new_reasoner();
    // Declare that alice's age is NOT 42
    r.assert_negative_data_property_assertion(":alice", "age", "42");
    // Actual age is 30 — consistent
    r.add_property_assertion(":alice", "age", "30");
    r.materialize().expect("materialize ok");
    assert!(r.is_consistent());
}

#[test]
fn test_negative_data_property_inconsistency() {
    let mut r = new_reasoner();
    r.assert_negative_data_property_assertion(":alice", "age", "42");
    // Assert the very value that was declared negative
    r.add_property_assertion(":alice", "age", "42");
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "alice age 42 is both asserted and declared negative"
    );
    assert!(
        r.inconsistencies()
            .iter()
            .any(|m| m.contains("NegativeDataPropertyAssertion")),
        "expected NegativeDataPropertyAssertion message"
    );
}

#[test]
fn test_negative_property_not_violated_by_different_property() {
    let mut r = new_reasoner();
    // Negative for P1, but only P2 is asserted (P1 ≠ P2)
    r.assert_negative_object_property_assertion(":x", "P1", ":y");
    r.add_property_assertion(":x", "P2", ":y");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_consistent(),
        "P2 triple does not violate the negative assertion on P1"
    );
}

#[test]
fn test_negative_property_violation_via_inferred_triple() {
    let mut r = new_reasoner();
    // P1 ⊑ P2: x P1 y is asserted, infers x P2 y
    // negative assertion says x P2 y must not hold
    r.add_sub_object_property_of("P1", "P2");
    r.assert_negative_object_property_assertion(":x", "P2", ":y");
    r.add_property_assertion(":x", "P1", ":y");
    r.materialize().expect("materialize call ok");
    // x P1 y → x P2 y (sub-property), but negative(x, P2, y) → inconsistency
    assert!(
        !r.is_consistent(),
        "inferred triple via sub-property violates negative assertion"
    );
}

// ── 28. hasSelf (self-restriction) ────────────────────────────────────────────

#[test]
fn test_has_self_forward_classification() {
    let mut r = new_reasoner();
    // SelfKnowing ≡ { x | x knows x }
    r.add_has_self_restriction("SelfKnowing", "knows");
    // alice knows herself — she should be classified as SelfKnowing
    r.add_property_assertion(":alice", "knows", ":alice");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "SelfKnowing"),
        "alice knows alice → alice is SelfKnowing via hasSelf forward"
    );
}

#[test]
fn test_has_self_no_forward_when_not_reflexive() {
    let mut r = new_reasoner();
    r.add_has_self_restriction("SelfKnowing", "knows");
    // alice knows bob (different individual) — should NOT classify alice as SelfKnowing
    r.add_property_assertion(":alice", "knows", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", "SelfKnowing"),
        "alice knows bob but not herself — no SelfKnowing classification"
    );
}

#[test]
fn test_has_self_backward_self_loop_inferred() {
    let mut r = new_reasoner();
    // SelfKnowing ≡ { x | x knows x }
    r.add_has_self_restriction("SelfKnowing", "knows");
    // alice is typed as SelfKnowing — backward: alice knows alice should be inferred
    r.assert_type(":alice", "SelfKnowing");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "knows", ":alice"),
        "alice is SelfKnowing → alice knows alice via hasSelf backward"
    );
}

#[test]
fn test_has_self_combined_with_range() {
    let mut r = new_reasoner();
    // SelfLinked ≡ { x | x linksTo x }; range(linksTo) = Node
    r.add_has_self_restriction("SelfLinked", "linksTo");
    r.add_range("linksTo", "Node");
    r.assert_type(":node1", "SelfLinked");
    r.materialize().expect("materialize ok");
    // Backward: node1 linksTo node1
    assert!(r.is_triple_entailed(":node1", "linksTo", ":node1"));
    // range fires: object of linksTo (which is node1) is classified as Node
    assert!(
        r.is_type_entailed(":node1", "Node"),
        "range(linksTo)=Node fires on the self-loop triple"
    );
}

#[test]
fn test_has_self_plus_reflexive_both_generate_self_loops() {
    let mut r = new_reasoner();
    // Both reflexive property and hasSelf restriction refer to the same property
    r.add_reflexive_property("rel");
    r.add_has_self_restriction("SelfRel", "rel");
    r.assert_type(":alice", "Person");
    r.materialize().expect("materialize ok");
    // Reflexive produces alice rel alice
    assert!(r.is_triple_entailed(":alice", "rel", ":alice"));
    // hasSelf forward: alice rel alice → alice is SelfRel
    assert!(
        r.is_type_entailed(":alice", "SelfRel"),
        "reflexive self-loop triggers hasSelf forward classification"
    );
}

// ── 29. Integration — cross-feature interactions ──────────────────────────────

#[test]
fn test_sub_property_then_functional_merge() {
    let mut r = new_reasoner();
    // P1 ⊑ P2, P2 is functional: x P1 a and x P1 b
    // → x P2 a and x P2 b → a sameAs b
    r.add_sub_object_property_of("P1", "P2");
    r.add_functional_property("P2");
    r.add_property_assertion(":x", "P1", ":a");
    r.add_property_assertion(":x", "P1", ":b");
    r.materialize().expect("materialize ok");
    // Sub-property: x P2 a and x P2 b are inferred
    assert!(r.is_triple_entailed(":x", "P2", ":a"));
    assert!(r.is_triple_entailed(":x", "P2", ":b"));
    // Functional: a sameAs b
    assert!(
        r.is_triple_entailed(":a", OWL_SAME_AS, ":b"),
        "functional P2 merges a and b after sub-property expansion"
    );
}

#[test]
fn test_equivalent_property_then_disjoint_violation() {
    let mut r = new_reasoner();
    // P1 ≡ P2 (equivalent), P1 disjointWith P3
    r.add_equivalent_properties("P1", "P2");
    r.add_disjoint_properties("P1", "P3");
    // Asserting P2 and P3 for the same (x, y) pair:
    // P2 equivalent P1 → x P1 y is inferred → disjoint with P3
    r.add_property_assertion(":x", "P2", ":y");
    r.add_property_assertion(":x", "P3", ":y");
    r.materialize().expect("materialize call ok");
    assert!(
        !r.is_consistent(),
        "P2 expands to P1 via equivalence, then P1 disjointWith P3 fires"
    );
}

#[test]
fn test_sub_property_chain_domain_range_type_inference() {
    let mut r = new_reasoner();
    // deeplyNestedIn ⊑ partOf; domain(partOf) = Component, range(partOf) = Assembly
    r.add_sub_object_property_of("deeplyNestedIn", "partOf");
    r.add_domain("partOf", "Component");
    r.add_range("partOf", "Assembly");
    r.add_property_assertion(":bolt", "deeplyNestedIn", ":engine");
    r.materialize().expect("materialize ok");
    // Sub-property: bolt partOf engine
    assert!(r.is_triple_entailed(":bolt", "partOf", ":engine"));
    // Domain: bolt rdf:type Component
    assert!(r.is_type_entailed(":bolt", "Component"));
    // Range: engine rdf:type Assembly
    assert!(r.is_type_entailed(":engine", "Assembly"));
}

#[test]
fn test_negative_assertion_with_same_as_propagation() {
    let mut r = new_reasoner();
    // Negative: alice does NOT have bob as mentor
    r.assert_negative_object_property_assertion(":alice", "hasMentor", ":bob");
    // alicia is sameAs alice — Rule 29 (full sameAs congruence) propagates
    // alicia hasMentor bob → alice hasMentor bob, which violates the negative assertion.
    // Per OWL 2 direct semantics, sameAs individuals share all property assertions.
    r.assert_same_as(":alicia", ":alice");
    r.add_property_assertion(":alicia", "hasMentor", ":bob");
    r.materialize().expect("materialize ok");
    // Full sameAs congruence: alicia hasMentor bob propagates to alice hasMentor bob
    // alice hasMentor bob is now inferred, violating the NegativePropertyAssertion
    assert!(
        !r.is_consistent(),
        "sameAs congruence propagates hasMentor to alice, violating the NegativePropertyAssertion"
    );
}

#[test]
fn test_has_self_with_sub_property_classification() {
    let mut r = new_reasoner();
    // SelfConnected ≡ { x | x connected x }
    // directlyConnected ⊑ connected
    r.add_has_self_restriction("SelfConnected", "connected");
    r.add_sub_object_property_of("directlyConnected", "connected");
    // x directlyConnected x (self-loop via sub-property, different predicate)
    r.add_property_assertion(":router", "directlyConnected", ":router");
    r.materialize().expect("materialize ok");
    // directlyConnected ⊑ connected → router connected router
    assert!(r.is_triple_entailed(":router", "connected", ":router"));
    // hasSelf forward: router connected router → router is SelfConnected
    assert!(
        r.is_type_entailed(":router", "SelfConnected"),
        "sub-property produces self-loop, hasSelf classifies the individual"
    );
}

#[test]
fn test_rule_firings_reported_for_new_features() {
    let mut r = new_reasoner();
    r.add_sub_object_property_of("P1", "P2");
    r.add_property_assertion(":a", "P1", ":b");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.sub_object_property >= 1,
        "sub_object_property counter should be at least 1"
    );
}

#[test]
fn test_equivalent_properties_rule_firings() {
    let mut r = new_reasoner();
    r.add_equivalent_properties("P1", "P2");
    r.add_property_assertion(":a", "P1", ":b");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.equivalent_properties >= 1,
        "equivalent_properties counter should be at least 1"
    );
}

#[test]
fn test_has_self_rule_firings_reported() {
    let mut r = new_reasoner();
    r.add_has_self_restriction("SelfRel", "rel");
    r.assert_type(":alice", "SelfRel");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.has_self >= 1,
        "has_self counter should be at least 1"
    );
}

#[test]
fn test_reflexive_self_rule_firings_reported() {
    let mut r = new_reasoner();
    r.add_reflexive_property("knows");
    r.assert_type(":alice", "Person");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.reflexive_self >= 1,
        "reflexive_self counter should be at least 1"
    );
}
