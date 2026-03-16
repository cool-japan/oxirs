//! Additional test suite — OWL 2 DL 100% milestone (Rules 22–29).
//!
//! Covers: MaxCardinality, MinCardinality, ExactCardinality, ObjectUnionOf,
//! DataSomeValuesFrom, DataAllValuesFrom, AllDifferent, and full sameAs congruence.

use super::vocab::*;
use super::*;

fn new_reasoner() -> Owl2DLReasoner {
    Owl2DLReasoner::new()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 100% milestone tests — Rules 22–29 (cardinality, union, data props, AllDifferent, sameAs)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Rule 22: MaxCardinality ───────────────────────────────────────────────────

#[test]
fn test_max_cardinality_zero_with_filler_is_nothing() {
    // MaxCardinality(0) on hasChild: AtMostZeroChildren members with any child → owl:Nothing
    let mut r = new_reasoner();
    r.add_max_cardinality("AtMostZeroChildren", "hasChild", 0);
    r.assert_type(":alice", "AtMostZeroChildren");
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", OWL_NOTHING),
        "maxCardinality(0) violation: alice has a child but restriction disallows it"
    );
    assert!(
        !r.is_consistent(),
        "ABox should be inconsistent (owl:Nothing member)"
    );
}

#[test]
fn test_max_cardinality_zero_no_filler_no_violation() {
    let mut r = new_reasoner();
    r.add_max_cardinality("AtMostZeroChildren", "hasChild", 0);
    r.assert_type(":alice", "AtMostZeroChildren");
    // No hasChild assertion — no violation
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", OWL_NOTHING),
        "no filler → no violation for maxCardinality(0)"
    );
    assert!(r.is_consistent());
}

#[test]
fn test_max_cardinality_one_single_value_ok() {
    let mut r = new_reasoner();
    r.add_max_cardinality("AtMostOneSupervisor", "hasSupervisor", 1);
    r.assert_type(":alice", "AtMostOneSupervisor");
    r.add_property_assertion(":alice", "hasSupervisor", ":bob");
    r.materialize().expect("materialize ok");
    // Exactly one filler — within the bound
    assert!(
        !r.is_type_entailed(":alice", OWL_NOTHING),
        "single filler within maxCardinality(1) — no violation"
    );
}

#[test]
fn test_max_cardinality_one_two_values_violation() {
    let mut r = new_reasoner();
    r.add_max_cardinality("AtMostOneSupervisor", "hasSupervisor", 1);
    r.assert_type(":alice", "AtMostOneSupervisor");
    r.add_property_assertion(":alice", "hasSupervisor", ":bob");
    r.add_property_assertion(":alice", "hasSupervisor", ":carol");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", OWL_NOTHING),
        "two fillers for maxCardinality(1) — owl:Nothing should be inferred"
    );
}

#[test]
fn test_max_cardinality_post_loop_inconsistency() -> anyhow::Result<()> {
    // The post-loop check should register an inconsistency message
    let mut r = new_reasoner();
    r.add_max_cardinality("AtMostOne", "hasFriend", 1);
    r.assert_type(":x", "AtMostOne");
    r.add_property_assertion(":x", "hasFriend", ":a");
    r.add_property_assertion(":x", "hasFriend", ":b");
    r.materialize().expect("materialize ok");
    let msgs = r.inconsistencies();
    assert!(
        msgs.iter()
            .any(|m| m.contains("maxCardinality") || m.contains("Nothing")),
        "expected maxCardinality violation in inconsistencies: {msgs:?}"
    );
    Ok(())
}

#[test]
fn test_max_qualified_cardinality_filters_by_class() {
    // maxQualifiedCardinality(1, Cat) on hasPet: at most 1 Cat pet allowed
    let mut r = new_reasoner();
    r.add_max_qualified_cardinality("AtMostOneCatOwner", "hasPet", 1, "Cat");
    r.assert_type(":alice", "AtMostOneCatOwner");
    // Two cats
    r.add_property_assertion(":alice", "hasPet", ":whiskers");
    r.add_property_assertion(":alice", "hasPet", ":mittens");
    r.assert_type(":whiskers", "Cat");
    r.assert_type(":mittens", "Cat");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", OWL_NOTHING),
        "two Cat pets for maxQualifiedCardinality(1, Cat) — violation"
    );
}

#[test]
fn test_max_qualified_cardinality_non_qualifying_ignored() {
    // maxQualifiedCardinality(1, Cat) on hasPet: dogs don't count
    let mut r = new_reasoner();
    r.add_max_qualified_cardinality("AtMostOneCatOwner", "hasPet", 1, "Cat");
    r.assert_type(":alice", "AtMostOneCatOwner");
    r.add_property_assertion(":alice", "hasPet", ":rex");
    r.assert_type(":rex", "Dog");
    // Rex is Dog, not Cat — should not count
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", OWL_NOTHING),
        "Dog pet doesn't violate maxQualifiedCardinality(1, Cat)"
    );
}

// ── Rule 23: MinCardinality ───────────────────────────────────────────────────

#[test]
fn test_min_cardinality_zero_classifies_all_individuals() {
    // minCardinality(0) — trivially satisfied by every individual
    let mut r = new_reasoner();
    r.add_min_cardinality("Anything", "someRelation", 0);
    r.assert_type(":alice", "Person");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "Anything"),
        "minCardinality(0) satisfied trivially — alice is Anything"
    );
}

#[test]
fn test_min_cardinality_one_with_filler_classifies() {
    // minCardinality(1) on hasEmail: any individual with an email is EmailOwner
    let mut r = new_reasoner();
    r.add_min_cardinality("EmailOwner", "hasEmail", 1);
    r.add_property_assertion(":alice", "hasEmail", "alice@example.com");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "EmailOwner"),
        "alice has email → alice is EmailOwner (minCardinality(1))"
    );
}

#[test]
fn test_min_cardinality_one_without_filler_no_classification() {
    let mut r = new_reasoner();
    r.add_min_cardinality("EmailOwner", "hasEmail", 1);
    r.assert_type(":bob", "Person");
    // No hasEmail assertion for bob
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":bob", "EmailOwner"),
        "bob has no email → not classified as EmailOwner"
    );
}

#[test]
fn test_min_qualified_cardinality_one_with_qualified_filler() {
    // minQualifiedCardinality(1, Mammal) on hasPet: having a Mammal pet → PetOwnerOfMammal
    let mut r = new_reasoner();
    r.add_min_qualified_cardinality("PetOwnerOfMammal", "hasPet", 1, "Mammal");
    r.add_property_assertion(":alice", "hasPet", ":fido");
    r.assert_type(":fido", "Mammal");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "PetOwnerOfMammal"),
        "fido is Mammal → alice is PetOwnerOfMammal via minQualifiedCardinality(1)"
    );
}

#[test]
fn test_min_qualified_cardinality_non_qualifying_does_not_classify() {
    let mut r = new_reasoner();
    r.add_min_qualified_cardinality("PetOwnerOfMammal", "hasPet", 1, "Mammal");
    r.add_property_assertion(":alice", "hasPet", ":parrot");
    r.assert_type(":parrot", "Bird");
    // Parrot is Bird, not Mammal — should not qualify
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", "PetOwnerOfMammal"),
        "Bird pet doesn't satisfy minQualifiedCardinality(1, Mammal)"
    );
}

// ── Rule 24: ExactCardinality ─────────────────────────────────────────────────

#[test]
fn test_exact_cardinality_zero_no_filler_ok() {
    let mut r = new_reasoner();
    r.add_exact_cardinality("NoPetOwner", "hasPet", 0);
    r.assert_type(":alice", "NoPetOwner");
    // No pet — satisfies exactCardinality(0)
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":alice", OWL_NOTHING),
        "exactCardinality(0) with no filler is satisfied — no violation"
    );
}

#[test]
fn test_exact_cardinality_zero_with_filler_violation() {
    let mut r = new_reasoner();
    r.add_exact_cardinality("NoPetOwner", "hasPet", 0);
    r.assert_type(":alice", "NoPetOwner");
    r.add_property_assertion(":alice", "hasPet", ":fido");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", OWL_NOTHING),
        "exactCardinality(0) with filler → owl:Nothing (violation)"
    );
}

#[test]
fn test_exact_cardinality_one_classifies_and_enforces_max() {
    // exactCardinality(1) on hasSupervisor:
    // - x hasSupervisor y → x rdf:type restriction (min side)
    // - x hasSupervisor a AND x hasSupervisor b → violation (max side)
    let mut r = new_reasoner();
    r.add_exact_cardinality("ExactlyOneSupervisee", "hasSupervisor", 1);
    r.add_property_assertion(":alice", "hasSupervisor", ":mgr");
    r.materialize().expect("materialize ok");
    // Min side: alice is classified
    assert!(
        r.is_type_entailed(":alice", "ExactlyOneSupervisee"),
        "alice with 1 supervisor → classified as ExactlyOneSupervisee"
    );
    // No violation for exactly 1 filler
    assert!(
        !r.is_type_entailed(":alice", OWL_NOTHING),
        "exactly 1 filler — no violation"
    );
}

#[test]
fn test_exact_cardinality_one_two_fillers_violation() {
    let mut r = new_reasoner();
    r.add_exact_cardinality("ExactlyOneSupervisee", "hasSupervisor", 1);
    r.assert_type(":alice", "ExactlyOneSupervisee");
    r.add_property_assertion(":alice", "hasSupervisor", ":mgr1");
    r.add_property_assertion(":alice", "hasSupervisor", ":mgr2");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", OWL_NOTHING),
        "exactCardinality(1) with two fillers → owl:Nothing"
    );
}

// ── Rule 25: ObjectUnionOf ────────────────────────────────────────────────────

#[test]
fn test_union_of_member_in_first_operand() {
    let mut r = new_reasoner();
    r.add_union_of(
        "AnimalOrPlant",
        vec!["Animal".to_string(), "Plant".to_string()],
    );
    r.assert_type(":fido", "Animal");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":fido", "AnimalOrPlant"),
        "fido is Animal → fido is in union class AnimalOrPlant"
    );
}

#[test]
fn test_union_of_member_in_second_operand() {
    let mut r = new_reasoner();
    r.add_union_of(
        "AnimalOrPlant",
        vec!["Animal".to_string(), "Plant".to_string()],
    );
    r.assert_type(":oak", "Plant");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":oak", "AnimalOrPlant"),
        "oak is Plant → oak is in union class AnimalOrPlant"
    );
}

#[test]
fn test_union_of_non_member_not_classified() {
    let mut r = new_reasoner();
    r.add_union_of(
        "AnimalOrPlant",
        vec!["Animal".to_string(), "Plant".to_string()],
    );
    r.assert_type(":rock", "Mineral");
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":rock", "AnimalOrPlant"),
        "rock is Mineral — not in any operand of the union"
    );
}

#[test]
fn test_union_of_with_subclass_propagation() {
    // Dog ⊑ Animal, AnimalOrPlant = Animal ∪ Plant
    // fido: Dog → Animal (subclass) → AnimalOrPlant (union)
    let mut r = new_reasoner();
    r.add_subclass_of("Dog", "Animal");
    r.add_union_of(
        "AnimalOrPlant",
        vec!["Animal".to_string(), "Plant".to_string()],
    );
    r.assert_type(":fido", "Dog");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed(":fido", "Animal"));
    assert!(
        r.is_type_entailed(":fido", "AnimalOrPlant"),
        "Dog ⊑ Animal ⊑ AnimalOrPlant via subclass + union"
    );
}

#[test]
fn test_union_of_three_operands() {
    let mut r = new_reasoner();
    r.add_union_of(
        "AnyPrimaryThing",
        vec![
            "Rock".to_string(),
            "Plant".to_string(),
            "Animal".to_string(),
        ],
    );
    r.assert_type(":stone", "Rock");
    r.assert_type(":leaf", "Plant");
    r.assert_type(":bird", "Animal");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed(":stone", "AnyPrimaryThing"));
    assert!(r.is_type_entailed(":leaf", "AnyPrimaryThing"));
    assert!(r.is_type_entailed(":bird", "AnyPrimaryThing"));
}

#[test]
fn test_union_of_rule_firings_counter() {
    let mut r = new_reasoner();
    r.add_union_of("U", vec!["A".to_string(), "B".to_string()]);
    r.assert_type(":x", "A");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.union_of >= 1,
        "union_of firing counter should be at least 1"
    );
}

// ── Rule 26: DataSomeValuesFrom ───────────────────────────────────────────────

#[test]
fn test_data_some_values_from_backward_classification() {
    let mut r = new_reasoner();
    r.add_data_some_values_from("HasName", "foaf_name", None);
    r.add_property_assertion(":alice", "foaf_name", "\"Alice\"");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_type_entailed(":alice", "HasName"),
        "alice has a name literal → classified as HasName via DataSomeValuesFrom"
    );
}

#[test]
fn test_data_some_values_from_multiple_properties() {
    let mut r = new_reasoner();
    r.add_data_some_values_from("HasAge", "xsd_age", None);
    r.add_data_some_values_from("HasEmail", "foaf_mbox", None);
    r.add_property_assertion(":alice", "xsd_age", "\"30\"^^xsd:integer");
    r.add_property_assertion(":alice", "foaf_mbox", "\"alice@example.com\"");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed(":alice", "HasAge"));
    assert!(r.is_type_entailed(":alice", "HasEmail"));
}

#[test]
fn test_data_some_values_from_no_assertion_no_classification() {
    let mut r = new_reasoner();
    r.add_data_some_values_from("HasName", "foaf_name", None);
    r.assert_type(":bob", "Person");
    // No foaf_name assertion
    r.materialize().expect("materialize ok");
    assert!(
        !r.is_type_entailed(":bob", "HasName"),
        "bob has no foaf_name → not HasName"
    );
}

#[test]
fn test_data_some_values_from_rule_firings() {
    let mut r = new_reasoner();
    r.add_data_some_values_from("HasId", "identifier", None);
    r.add_property_assertion(":thing1", "identifier", "\"id-001\"");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.data_some_values_from >= 1,
        "data_some_values_from firing counter should be >= 1"
    );
}

// ── Rule 27: DataAllValuesFrom ────────────────────────────────────────────────

#[test]
fn test_data_all_values_from_classifies_filler() {
    let mut r = new_reasoner();
    // AllDataValuesAreStrings: members with stringProp values → values typed as StringType
    r.add_data_all_values_from("AllStringData", "stringProp", "xsd_string");
    r.assert_type(":x", "AllStringData");
    r.add_property_assertion(":x", "stringProp", "\"hello\"");
    r.materialize().expect("materialize ok");
    // The literal value should be classified as xsd_string
    assert!(
        r.is_type_entailed("\"hello\"", "xsd_string"),
        "literal value classified as xsd_string via DataAllValuesFrom"
    );
}

#[test]
fn test_data_all_values_from_no_filler_no_classification() {
    let mut r = new_reasoner();
    r.add_data_all_values_from("AllIntData", "intProp", "xsd_integer");
    r.assert_type(":y", "AllIntData");
    // No intProp assertion
    r.materialize().expect("materialize ok");
    // No literal to classify
    assert!(
        !r.is_type_entailed("\"42\"", "xsd_integer"),
        "no filler — nothing to classify"
    );
}

#[test]
fn test_data_all_values_from_rule_firings() {
    let mut r = new_reasoner();
    r.add_data_all_values_from("AllDateData", "dateProp", "xsd_date");
    r.assert_type(":event", "AllDateData");
    r.add_property_assertion(":event", "dateProp", "\"2024-01-01\"");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.data_all_values_from >= 1,
        "data_all_values_from firing counter should be >= 1"
    );
}

// ── Rule 28: AllDifferent ─────────────────────────────────────────────────────

#[test]
fn test_all_different_materialises_different_from() {
    let mut r = new_reasoner();
    r.add_all_different(vec![
        ":alice".to_string(),
        ":bob".to_string(),
        ":carol".to_string(),
    ]);
    r.materialize().expect("materialize ok");
    // All pairwise differentFrom triples should be inferred
    assert!(
        r.is_triple_entailed(":alice", OWL_DIFFERENT_FROM, ":bob"),
        "alice differentFrom bob"
    );
    assert!(
        r.is_triple_entailed(":bob", OWL_DIFFERENT_FROM, ":alice"),
        "bob differentFrom alice (symmetric)"
    );
    assert!(
        r.is_triple_entailed(":alice", OWL_DIFFERENT_FROM, ":carol"),
        "alice differentFrom carol"
    );
    assert!(
        r.is_triple_entailed(":carol", OWL_DIFFERENT_FROM, ":alice"),
        "carol differentFrom alice"
    );
    assert!(
        r.is_triple_entailed(":bob", OWL_DIFFERENT_FROM, ":carol"),
        "bob differentFrom carol"
    );
}

#[test]
fn test_all_different_violation_detected_when_same_as() -> anyhow::Result<()> {
    let mut r = new_reasoner();
    r.add_all_different(vec![":a".to_string(), ":b".to_string()]);
    // Contradictory: also assert a sameAs b
    r.assert_same_as(":a", ":b");
    r.materialize().expect("materialize ok");
    let msgs = r.inconsistencies();
    assert!(
        msgs.iter().any(|m| m.contains("AllDifferent")),
        "expected AllDifferent violation in inconsistencies: {msgs:?}"
    );
    assert!(!r.is_consistent());
    Ok(())
}

#[test]
fn test_all_different_two_members() {
    let mut r = new_reasoner();
    r.add_all_different(vec![":x".to_string(), ":y".to_string()]);
    r.materialize().expect("materialize ok");
    assert!(r.is_triple_entailed(":x", OWL_DIFFERENT_FROM, ":y"));
    assert!(r.is_triple_entailed(":y", OWL_DIFFERENT_FROM, ":x"));
    assert!(r.is_consistent());
}

#[test]
fn test_all_different_rule_firings_counter() {
    let mut r = new_reasoner();
    r.add_all_different(vec![":p".to_string(), ":q".to_string(), ":r".to_string()]);
    let report = r.materialize().expect("materialize ok");
    // 3 individuals → 3 pairs × 2 directions = 6 firings
    assert!(
        report.rule_firings.all_different >= 6,
        "expected >= 6 all_different firings for 3 individuals, got {}",
        report.rule_firings.all_different
    );
}

#[test]
fn test_all_different_min_members_not_applied_for_one() {
    // add_all_different with < 2 members is a no-op
    let mut r = new_reasoner();
    r.add_all_different(vec![":a".to_string()]);
    r.materialize().expect("materialize ok");
    // No differentFrom triples should exist
    assert!(!r.is_triple_entailed(":a", OWL_DIFFERENT_FROM, ":a"));
}

// ── Rule 29: Full sameAs congruence ──────────────────────────────────────────

#[test]
fn test_same_as_congruence_property_value_forward() {
    // x sameAs y, x P z → y P z
    let mut r = new_reasoner();
    r.assert_same_as(":alice", ":alicia");
    r.add_property_assertion(":alice", "hasFriend", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alicia", "hasFriend", ":bob"),
        "alice sameAs alicia + alice hasFriend bob → alicia hasFriend bob"
    );
}

#[test]
fn test_same_as_congruence_property_value_backward() {
    // x sameAs y, z P x → z P y
    let mut r = new_reasoner();
    r.assert_same_as(":bob", ":robert");
    r.add_property_assertion(":alice", "hasFriend", ":bob");
    r.materialize().expect("materialize ok");
    assert!(
        r.is_triple_entailed(":alice", "hasFriend", ":robert"),
        "bob sameAs robert + alice hasFriend bob → alice hasFriend robert"
    );
}

#[test]
fn test_same_as_congruence_type_propagation() {
    // Existing: Rule 13 propagates rdf:type
    // New rule 29 also propagates non-type properties
    let mut r = new_reasoner();
    r.assert_same_as(":carol", ":carrie");
    r.assert_type(":carol", "Person");
    r.add_property_assertion(":carol", "age", "30");
    r.materialize().expect("materialize ok");
    // Type should propagate
    assert!(r.is_type_entailed(":carrie", "Person"));
    // Property assertion should also propagate
    assert!(
        r.is_triple_entailed(":carrie", "age", "30"),
        "carrie should inherit age=30 from carol via sameAs congruence"
    );
}

#[test]
fn test_same_as_congruence_chain_x_y_z() {
    // x sameAs y, y sameAs z → all properties propagate across the chain
    let mut r = new_reasoner();
    r.assert_same_as(":a", ":b");
    r.assert_same_as(":b", ":c");
    r.add_property_assertion(":a", "rel", ":target");
    r.materialize().expect("materialize ok");
    // a→b via sameAs
    assert!(r.is_triple_entailed(":b", "rel", ":target"));
    // b→c via sameAs (iterated)
    assert!(
        r.is_triple_entailed(":c", "rel", ":target"),
        "property propagates through sameAs chain a=b=c"
    );
}

#[test]
fn test_same_as_congruence_symmetric() {
    // sameAs is symmetric: a sameAs b stored both ways
    let mut r = new_reasoner();
    r.assert_same_as(":x", ":y");
    r.add_property_assertion(":y", "knows", ":z");
    r.materialize().expect("materialize ok");
    // y knows z, and x=y, so x knows z
    assert!(
        r.is_triple_entailed(":x", "knows", ":z"),
        "x=y, y knows z → x knows z"
    );
}

#[test]
fn test_same_as_congruence_rule_firings_counter() {
    let mut r = new_reasoner();
    r.assert_same_as(":p", ":q");
    r.add_property_assertion(":p", "hasValue", ":v");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.same_as_congruence >= 1,
        "same_as_congruence firing counter should be >= 1"
    );
}

// ── Integration tests for the 100% milestone ─────────────────────────────────

#[test]
fn test_cardinality_plus_functional_property_merging() {
    // MaxCardinality(1) on hasSupervisor + FunctionalProperty(hasSupervisor)
    // alice hasSupervisor bob AND alice hasSupervisor carol
    // → FunctionalProperty: bob sameAs carol
    // → MaxCardinality violation: owl:Nothing for alice
    let mut r = new_reasoner();
    r.add_max_cardinality("AtMostOneSupervisedEmployee", "hasSupervisor", 1);
    r.add_functional_property("hasSupervisor");
    r.assert_type(":alice", "AtMostOneSupervisedEmployee");
    r.add_property_assertion(":alice", "hasSupervisor", ":bob");
    r.add_property_assertion(":alice", "hasSupervisor", ":carol");
    r.materialize().expect("materialize ok");
    // Functional: bob sameAs carol
    assert!(r.is_triple_entailed(":bob", OWL_SAME_AS, ":carol"));
    // MaxCardinality violated → owl:Nothing
    assert!(r.is_type_entailed(":alice", OWL_NOTHING));
}

#[test]
fn test_union_of_with_cardinality_restriction() {
    // PersonOrOrg is union of Person ∪ Organization
    // AtMostOneOwner has maxCardinality(1, hasSupervisor)
    // Member of AtMostOneOwner that belongs to PersonOrOrg via subclass
    let mut r = new_reasoner();
    r.add_union_of(
        "PersonOrOrg",
        vec!["Person".to_string(), "Organization".to_string()],
    );
    r.add_subclass_of("Employee", "Person");
    r.add_max_cardinality("AtMostOneSupervisedEmployee", "hasSupervisor", 1);
    r.assert_type(":alice", "Employee");
    r.assert_type(":alice", "AtMostOneSupervisedEmployee");
    r.add_property_assertion(":alice", "hasSupervisor", ":mgr");
    r.materialize().expect("materialize ok");
    // Union: alice is Employee ⊑ Person → alice is PersonOrOrg
    assert!(r.is_type_entailed(":alice", "PersonOrOrg"));
    // MaxCardinality OK (1 filler)
    assert!(!r.is_type_entailed(":alice", OWL_NOTHING));
}

#[test]
fn test_all_different_combined_with_same_as_violation() {
    // Alice and Bob declared different; then functional property merges them
    let mut r = new_reasoner();
    r.add_all_different(vec![":alice".to_string(), ":bob".to_string()]);
    r.add_functional_property("hasSupervisor");
    // charlie hasSupervisor alice AND hasSupervisor bob → alice sameAs bob
    r.add_property_assertion(":charlie", "hasSupervisor", ":alice");
    r.add_property_assertion(":charlie", "hasSupervisor", ":bob");
    r.materialize().expect("materialize ok");
    // alice sameAs bob inferred
    assert!(r.is_triple_entailed(":alice", OWL_SAME_AS, ":bob"));
    // AllDifferent violation: alice differentFrom bob but also sameAs
    assert!(!r.is_consistent());
}

#[test]
fn test_data_some_values_from_then_subclass_propagation() {
    // DataSomeValuesFrom(HasAge, age) → HasAge
    // HasAge ⊑ IdentifiablePerson
    let mut r = new_reasoner();
    r.add_data_some_values_from("HasAge", "age", None);
    r.add_subclass_of("HasAge", "IdentifiablePerson");
    r.add_property_assertion(":alice", "age", "\"30\"");
    r.materialize().expect("materialize ok");
    assert!(r.is_type_entailed(":alice", "HasAge"));
    assert!(
        r.is_type_entailed(":alice", "IdentifiablePerson"),
        "HasAge ⊑ IdentifiablePerson → alice is IdentifiablePerson"
    );
}

#[test]
fn test_min_cardinality_then_domain_type_inference() {
    // minCardinality(1, hasChild) → ParentWithChild
    // domain(hasChild) = Parent
    let mut r = new_reasoner();
    r.add_min_cardinality("ParentWithChild", "hasChild", 1);
    r.add_domain("hasChild", "Parent");
    r.add_property_assertion(":alice", "hasChild", ":bob");
    r.materialize().expect("materialize ok");
    // minCardinality(1): alice is ParentWithChild
    assert!(r.is_type_entailed(":alice", "ParentWithChild"));
    // domain: alice is Parent
    assert!(r.is_type_entailed(":alice", "Parent"));
}

#[test]
fn test_exact_cardinality_firings_counter() {
    let mut r = new_reasoner();
    r.add_exact_cardinality("ExactlyOneChild", "hasChild", 1);
    r.add_property_assertion(":alice", "hasChild", ":bob");
    let report = r.materialize().expect("materialize ok");
    assert!(
        report.rule_firings.exact_cardinality >= 1,
        "exact_cardinality firing counter should be >= 1"
    );
}
