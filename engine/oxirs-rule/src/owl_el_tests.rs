//! # OWL 2 EL Reasoner Tests
//!
//! All unit tests for the OWL 2 EL reasoner, split into `tests` (core) and
//! `tests_extended` (comprehensive edge cases).

#[cfg(test)]
mod tests {
    use crate::owl_el_axioms::{ElAxiom, ElConcept};
    use crate::owl_el_reasoner::Owl2ElReasoner;

    #[test]
    fn test_simple_subclass_classification() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "Animal");

        let cls = r.classify().expect("classification failed");
        assert!(cls.is_subclass_of("Dog", "Mammal"), "Dog ⊑ Mammal");
        assert!(
            cls.is_subclass_of("Dog", "Animal"),
            "Dog ⊑ Animal (transitive)"
        );
        assert!(cls.is_subclass_of("Mammal", "Animal"), "Mammal ⊑ Animal");
        assert!(!cls.is_subclass_of("Animal", "Dog"), "Animal not ⊑ Dog");
    }

    #[test]
    fn test_equivalent_classes() {
        let mut r = Owl2ElReasoner::new();
        r.add_equivalent_classes("Human", "Person");

        let cls = r.classify().expect("classification failed");
        assert!(cls.is_subclass_of("Human", "Person"), "Human ⊑ Person");
        assert!(cls.is_subclass_of("Person", "Human"), "Person ⊑ Human");

        let equivs = &cls.equivalent_classes;
        let found = equivs
            .iter()
            .any(|g| g.contains(&"Human".to_string()) && g.contains(&"Person".to_string()));
        assert!(
            found,
            "Human and Person should be in the same equivalence group"
        );
    }

    #[test]
    fn test_intersection_on_left() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // Doctor ⊓ HaematologySpecialist ⊑ Haematologist
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::intersection(vec![
                ElConcept::named("Doctor"),
                ElConcept::named("HaematologySpecialist"),
            ]),
            sup: ElConcept::named("Haematologist"),
        });
        r.add_concept_assertion("alice", "Doctor");
        r.add_concept_assertion("alice", "HaematologySpecialist");

        let cls = r.classify().expect("classification failed");
        let alice_types = cls.get_individual_types("alice");
        assert!(
            alice_types.contains(&"Haematologist".to_string()),
            "Expected alice to be Haematologist via intersection. Got: {:?}",
            alice_types
        );
        Ok(())
    }

    #[test]
    fn test_existential_some_values_from_right() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // Person ⊑ ∃hasParent.Human
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::named("Person"),
            sup: ElConcept::some_values("hasParent", ElConcept::named("Human")),
        });
        // ∃hasParent.Human ⊑ OffspringOfHuman
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("hasParent", ElConcept::named("Human")),
            sup: ElConcept::named("OffspringOfHuman"),
        });
        r.add_concept_assertion("alice", "Person");

        let cls = r.classify().expect("classification failed");
        let alice_types = cls.get_individual_types("alice");
        assert!(
            alice_types.contains(&"OffspringOfHuman".to_string()),
            "Expected alice OffspringOfHuman via existential chain. Got: {:?}",
            alice_types
        );
        Ok(())
    }

    #[test]
    fn test_transitive_role() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_transitive_role("partOf");
        r.add_role_assertion("lug", "partOf", "wheel");
        r.add_role_assertion("wheel", "partOf", "car");

        let cls = r.classify().expect("classification failed");
        let lug_succs = cls
            .role_successors
            .get(&("lug".to_string(), "partOf".to_string()));
        assert!(
            lug_succs.map(|s| s.contains("car")).unwrap_or(false),
            "Expected lug partOf car via transitivity. Got: {:?}",
            lug_succs
        );
        Ok(())
    }

    #[test]
    fn test_property_chain() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // hasParent o hasParent ⊑ hasGrandParent
        r.add_property_chain(
            vec!["hasParent".to_string(), "hasParent".to_string()],
            "hasGrandParent",
        );
        r.add_role_assertion("child", "hasParent", "parent");
        r.add_role_assertion("parent", "hasParent", "grandparent");

        let cls = r.classify().expect("classification failed");
        let child_grand = cls
            .role_successors
            .get(&("child".to_string(), "hasGrandParent".to_string()));
        assert!(
            child_grand
                .map(|s| s.contains("grandparent"))
                .unwrap_or(false),
            "Expected child hasGrandParent grandparent via chain. Got: {:?}",
            child_grand
        );
        Ok(())
    }

    #[test]
    fn test_abox_type_propagation() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_concept_assertion("fido", "Dog");

        let cls = r.classify().expect("classification failed");
        let fido_types = cls.get_individual_types("fido");
        assert!(
            fido_types.contains(&"Animal".to_string()),
            "Expected fido to be Animal via subclass. Got: {:?}",
            fido_types
        );
        Ok(())
    }

    #[test]
    fn test_get_superclasses() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Labrador", "Dog");
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "Animal");

        let supers = r.get_superclasses("Labrador").expect("failed");
        assert!(supers.contains(&"Dog".to_string()), "Missing Dog");
        assert!(supers.contains(&"Mammal".to_string()), "Missing Mammal");
        assert!(supers.contains(&"Animal".to_string()), "Missing Animal");
    }

    #[test]
    fn test_get_subclasses() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_subclass_of("Cat", "Animal");

        let subs = r.get_subclasses("Animal").expect("failed");
        assert!(subs.contains(&"Dog".to_string()), "Missing Dog");
        assert!(subs.contains(&"Cat".to_string()), "Missing Cat");
    }

    #[test]
    fn test_is_subclass_of() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Poodle", "Dog");

        assert!(r.is_subclass_of("Poodle", "Dog").expect("failed"));
        assert!(!r.is_subclass_of("Dog", "Poodle").expect("failed"));
    }

    #[test]
    fn test_long_chain_classification() {
        let mut r = Owl2ElReasoner::new();
        for i in 0..9usize {
            r.add_subclass_of(&format!("C{}", i), &format!("C{}", i + 1));
        }

        let cls = r.classify().expect("classification failed");
        for i in 1..10usize {
            assert!(
                cls.is_subclass_of("C0", &format!("C{}", i)),
                "C0 should be ⊑ C{}",
                i
            );
        }
    }

    #[test]
    fn test_sub_role() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_axiom(ElAxiom::SubRole {
            sub: "isChildOf".to_string(),
            sup: "isRelatedTo".to_string(),
        });
        r.add_role_assertion("alice", "isChildOf", "bob");

        let cls = r.classify().expect("classification failed");
        let alice_related = cls
            .role_successors
            .get(&("alice".to_string(), "isRelatedTo".to_string()));
        assert!(
            alice_related.map(|s| s.contains("bob")).unwrap_or(false),
            "Expected alice isRelatedTo bob via subRole. Got: {:?}",
            alice_related
        );
        Ok(())
    }

    #[test]
    fn test_some_sub_atom() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // ∃worksIn.Organization ⊑ Employee
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("worksIn", ElConcept::named("Organization")),
            sup: ElConcept::named("Employee"),
        });
        r.add_role_assertion("alice", "worksIn", "acme");
        r.add_concept_assertion("acme", "Organization");

        let cls = r.classify().expect("classification failed");
        let alice_types = cls.get_individual_types("alice");
        assert!(
            alice_types.contains(&"Employee".to_string()),
            "Expected alice to be Employee via ∃worksIn.Organization. Got: {:?}",
            alice_types
        );
        Ok(())
    }

    #[test]
    fn test_long_property_chain() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // r1 o r2 o r3 ⊑ rResult
        r.add_property_chain(
            vec!["r1".to_string(), "r2".to_string(), "r3".to_string()],
            "rResult",
        );
        r.add_role_assertion("a", "r1", "b");
        r.add_role_assertion("b", "r2", "c");
        r.add_role_assertion("c", "r3", "d");

        let cls = r.classify().expect("classification failed");
        let a_result = cls
            .role_successors
            .get(&("a".to_string(), "rResult".to_string()));
        assert!(
            a_result.map(|s| s.contains("d")).unwrap_or(false),
            "Expected a rResult d via 3-chain. Got: {:?}",
            a_result
        );
        Ok(())
    }
}

// -----------------------------------------------------------------------
// Extended Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests_extended {
    use crate::owl_el_axioms::{ElAxiom, ElConcept};
    use crate::owl_el_reasoner::Owl2ElReasoner;

    // ---- Subclass hierarchy tests ----

    #[test]
    fn test_empty_reasoner_classify_succeeds() {
        let r = Owl2ElReasoner::new();
        let cls = r.classify().expect("empty classify failed");
        // No individuals should be present in an empty ontology
        assert_eq!(cls.individual_types.len(), 0);
        // No individual types for any name
        assert!(cls.get_individual_types("nobody").is_empty());
    }

    #[test]
    fn test_single_subclass_axiom() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Cat", "Animal");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("Cat", "Animal"), "Cat ⊑ Animal");
        assert!(!cls.is_subclass_of("Animal", "Cat"), "Animal not ⊑ Cat");
    }

    #[test]
    fn test_no_accidental_subsumption() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("C", "D");
        let cls = r.classify().expect("classify failed");
        assert!(!cls.is_subclass_of("A", "D"), "A should not ⊑ D");
        assert!(!cls.is_subclass_of("C", "B"), "C should not ⊑ B");
    }

    #[test]
    fn test_self_subsumption() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("X", "Y");
        let cls = r.classify().expect("classify failed");
        // Every class is subclass of itself
        assert!(cls.is_subclass_of("X", "X"), "X ⊑ X (reflexivity)");
        assert!(cls.is_subclass_of("Y", "Y"), "Y ⊑ Y (reflexivity)");
    }

    #[test]
    fn test_five_level_chain() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("B", "C");
        r.add_subclass_of("C", "D");
        r.add_subclass_of("D", "E");
        r.add_subclass_of("E", "F");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("A", "F"), "A ⊑ F (5-hop transitive)");
        assert!(cls.is_subclass_of("B", "F"), "B ⊑ F");
        assert!(!cls.is_subclass_of("F", "A"), "F not ⊑ A");
    }

    #[test]
    fn test_get_superclasses_method() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Poodle", "Dog");
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "Animal");
        let supers = r
            .get_superclasses("Poodle")
            .expect("get_superclasses failed");
        assert!(supers.contains(&"Dog".to_string()), "Missing Dog");
        assert!(supers.contains(&"Mammal".to_string()), "Missing Mammal");
        assert!(supers.contains(&"Animal".to_string()), "Missing Animal");
    }

    #[test]
    fn test_get_subclasses_method() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_subclass_of("Cat", "Animal");
        r.add_subclass_of("Bird", "Animal");
        let subs = r.get_subclasses("Animal").expect("get_subclasses failed");
        assert!(subs.contains(&"Dog".to_string()), "Missing Dog");
        assert!(subs.contains(&"Cat".to_string()), "Missing Cat");
        assert!(subs.contains(&"Bird".to_string()), "Missing Bird");
    }

    #[test]
    fn test_is_subclass_of_method_false() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        assert!(r.is_subclass_of("A", "B").expect("failed"));
        assert!(!r.is_subclass_of("B", "A").expect("failed"));
        assert!(!r.is_subclass_of("A", "C").expect("failed"));
    }

    #[test]
    fn test_equivalent_classes_chain() {
        // A ≡ B, B ≡ C => A ⊑ C, C ⊑ A
        let mut r = Owl2ElReasoner::new();
        r.add_equivalent_classes("A", "B");
        r.add_equivalent_classes("B", "C");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("A", "B"), "A ⊑ B");
        assert!(cls.is_subclass_of("B", "A"), "B ⊑ A");
        assert!(cls.is_subclass_of("B", "C"), "B ⊑ C");
        assert!(cls.is_subclass_of("C", "B"), "C ⊑ B");
        // Transitive: A ⊑ C
        assert!(cls.is_subclass_of("A", "C"), "A ⊑ C (equiv chain)");
    }

    // ---- Intersection tests ----

    #[test]
    fn test_three_way_intersection_on_left() {
        let mut r = Owl2ElReasoner::new();
        // A ⊓ B ⊓ C ⊑ D
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::intersection(vec![
                ElConcept::named("A"),
                ElConcept::named("B"),
                ElConcept::named("C"),
            ]),
            sup: ElConcept::named("D"),
        });
        r.add_concept_assertion("x", "A");
        r.add_concept_assertion("x", "B");
        r.add_concept_assertion("x", "C");

        let cls = r.classify().expect("classify failed");
        assert!(
            cls.get_individual_types("x").contains(&"D".to_string()),
            "Expected x:D via 3-way intersection"
        );
    }

    #[test]
    fn test_intersection_missing_one_class() {
        let mut r = Owl2ElReasoner::new();
        // A ⊓ B ⊑ C
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::intersection(vec![ElConcept::named("A"), ElConcept::named("B")]),
            sup: ElConcept::named("C"),
        });
        r.add_concept_assertion("x", "A");
        // Missing B — x should NOT be classified as C

        let cls = r.classify().expect("classify failed");
        assert!(
            !cls.get_individual_types("x").contains(&"C".to_string()),
            "x should not be C without B"
        );
    }

    // ---- Existential restriction tests ----

    #[test]
    fn test_existential_chain_three_hops() {
        let mut r = Owl2ElReasoner::new();
        // ∃r.A ⊑ B, ∃s.B ⊑ C
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("r", ElConcept::named("A")),
            sup: ElConcept::named("B"),
        });
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("s", ElConcept::named("B")),
            sup: ElConcept::named("C"),
        });
        r.add_role_assertion("x", "r", "y");
        r.add_concept_assertion("y", "A");
        r.add_role_assertion("z", "s", "x");

        let cls = r.classify().expect("classify failed");
        // x: ∃r.A ⊑ B, so x:B
        assert!(
            cls.get_individual_types("x").contains(&"B".to_string()),
            "x should be B"
        );
        // z: ∃s.B (via x:B) ⊑ C, so z:C
        assert!(
            cls.get_individual_types("z").contains(&"C".to_string()),
            "z should be C"
        );
    }

    #[test]
    fn test_some_values_named_filler() {
        let mut r = Owl2ElReasoner::new();
        // ∃worksAt.Company ⊑ Employee
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("worksAt", ElConcept::named("Company")),
            sup: ElConcept::named("Employee"),
        });
        r.add_role_assertion("alice", "worksAt", "acme");
        r.add_concept_assertion("acme", "Company");

        let cls = r.classify().expect("classify failed");
        assert!(
            cls.get_individual_types("alice")
                .contains(&"Employee".to_string()),
            "alice should be Employee via ∃worksAt.Company"
        );
    }

    // ---- Role / property chain tests ----

    #[test]
    fn test_transitive_role_three_steps() {
        let mut r = Owl2ElReasoner::new();
        r.add_transitive_role("ancestorOf");
        r.add_role_assertion("great_grandparent", "ancestorOf", "grandparent");
        r.add_role_assertion("grandparent", "ancestorOf", "parent");
        r.add_role_assertion("parent", "ancestorOf", "child");

        let cls = r.classify().expect("classify failed");
        let ggp = cls
            .role_successors
            .get(&("great_grandparent".to_string(), "ancestorOf".to_string()));
        assert!(
            ggp.map(|s| s.contains("child")).unwrap_or(false),
            "great_grandparent should ancestorOf child (3-step)"
        );
    }

    #[test]
    fn test_property_chain_two_roles() {
        let mut r = Owl2ElReasoner::new();
        // uncle ≡ hasParent o hasBrother
        r.add_property_chain(
            vec!["hasParent".to_string(), "hasBrother".to_string()],
            "hasUncle",
        );
        r.add_role_assertion("alice", "hasParent", "bob");
        r.add_role_assertion("bob", "hasBrother", "charlie");

        let cls = r.classify().expect("classify failed");
        let alice_uncles = cls
            .role_successors
            .get(&("alice".to_string(), "hasUncle".to_string()));
        assert!(
            alice_uncles.map(|s| s.contains("charlie")).unwrap_or(false),
            "alice hasUncle charlie via chain"
        );
    }

    #[test]
    fn test_sub_role_propagation() {
        let mut r = Owl2ElReasoner::new();
        r.add_axiom(ElAxiom::SubRole {
            sub: "worksFor".to_string(),
            sup: "associatedWith".to_string(),
        });
        r.add_role_assertion("emp", "worksFor", "company");

        let cls = r.classify().expect("classify failed");
        let emp_assoc = cls
            .role_successors
            .get(&("emp".to_string(), "associatedWith".to_string()));
        assert!(
            emp_assoc.map(|s| s.contains("company")).unwrap_or(false),
            "emp should associatedWith company via subRole"
        );
    }

    // ---- Individual classification tests ----

    #[test]
    fn test_individual_inherits_via_subclass_chain() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Labrador", "Dog");
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "LivingBeing");
        r.add_concept_assertion("rex", "Labrador");

        let cls = r.classify().expect("classify failed");
        let types = cls.get_individual_types("rex");
        assert!(types.contains(&"Dog".to_string()), "rex should be Dog");
        assert!(
            types.contains(&"Mammal".to_string()),
            "rex should be Mammal"
        );
        assert!(
            types.contains(&"LivingBeing".to_string()),
            "rex should be LivingBeing"
        );
    }

    #[test]
    fn test_multiple_individuals_multiple_classes() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Truck", "Vehicle");
        r.add_subclass_of("Car", "Vehicle");
        r.add_concept_assertion("t1", "Truck");
        r.add_concept_assertion("c1", "Car");

        let cls = r.classify().expect("classify failed");
        assert!(
            cls.get_individual_types("t1")
                .contains(&"Vehicle".to_string()),
            "t1 should be Vehicle"
        );
        assert!(
            cls.get_individual_types("c1")
                .contains(&"Vehicle".to_string()),
            "c1 should be Vehicle"
        );
    }

    #[test]
    fn test_individual_in_multiple_classes() {
        let mut r = Owl2ElReasoner::new();
        r.add_concept_assertion("alice", "Professor");
        r.add_concept_assertion("alice", "Researcher");
        r.add_concept_assertion("alice", "Person");

        let cls = r.classify().expect("classify failed");
        let types = cls.get_individual_types("alice");
        assert!(types.contains(&"Professor".to_string()));
        assert!(types.contains(&"Researcher".to_string()));
        assert!(types.contains(&"Person".to_string()));
    }

    #[test]
    fn test_individual_unknown_is_empty() {
        let r = Owl2ElReasoner::new();
        let cls = r.classify().expect("classify failed");
        let types = cls.get_individual_types("nonexistent");
        assert!(types.is_empty(), "unknown individual should have no types");
    }

    #[test]
    fn test_classify_idempotent() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("B", "C");
        let cls1 = r.classify().expect("first classify failed");
        let cls2 = r.classify().expect("second classify failed");
        assert_eq!(
            cls1.is_subclass_of("A", "C"),
            cls2.is_subclass_of("A", "C"),
            "classify should be idempotent"
        );
    }

    // ---- Edge-case / structural tests ----

    #[test]
    fn test_diamond_inheritance() {
        // A ⊑ B, A ⊑ C, B ⊑ D, C ⊑ D
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("A", "C");
        r.add_subclass_of("B", "D");
        r.add_subclass_of("C", "D");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("A", "D"), "A ⊑ D (diamond)");
        assert!(cls.is_subclass_of("B", "D"), "B ⊑ D");
        assert!(cls.is_subclass_of("C", "D"), "C ⊑ D");
    }

    #[test]
    fn test_sibling_classes_not_related() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Cat", "Animal");
        r.add_subclass_of("Dog", "Animal");
        let cls = r.classify().expect("classify failed");
        assert!(!cls.is_subclass_of("Cat", "Dog"), "Cat should not ⊑ Dog");
        assert!(!cls.is_subclass_of("Dog", "Cat"), "Dog should not ⊑ Cat");
    }

    #[test]
    fn test_intersection_empty_becomes_top() {
        let result = ElConcept::intersection(vec![]);
        assert!(
            matches!(result, ElConcept::Top),
            "empty intersection is Top"
        );
    }

    #[test]
    fn test_intersection_single_becomes_concept() {
        let result = ElConcept::intersection(vec![ElConcept::named("A")]);
        assert!(
            matches!(result, ElConcept::Named(ref n) if n == "A"),
            "singleton intersection is the concept"
        );
    }

    #[test]
    fn test_add_axioms_batch() {
        let mut r = Owl2ElReasoner::new();
        r.add_axioms(vec![
            ElAxiom::SubConceptOf {
                sub: ElConcept::named("X"),
                sup: ElConcept::named("Y"),
            },
            ElAxiom::SubConceptOf {
                sub: ElConcept::named("Y"),
                sup: ElConcept::named("Z"),
            },
        ]);
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("X", "Z"), "X ⊑ Z via batch axioms");
    }

    #[test]
    fn test_named_concept_as_named() {
        let c = ElConcept::named("MyClass");
        assert_eq!(c.as_named(), Some("MyClass"));
    }

    #[test]
    fn test_top_concept_as_named_none() {
        let c = ElConcept::Top;
        assert_eq!(c.as_named(), None);
    }

    #[test]
    fn test_bottom_concept_as_named_none() {
        let c = ElConcept::Bottom;
        assert_eq!(c.as_named(), None);
    }

    #[test]
    fn test_property_chain_two_roles_simple() {
        // father o father ⊑ grandfather
        let mut r = Owl2ElReasoner::new();
        r.add_property_chain(
            vec!["father".to_string(), "father".to_string()],
            "grandfather",
        );
        r.add_role_assertion("x", "father", "y");
        r.add_role_assertion("y", "father", "z");

        let cls = r.classify().expect("classify failed");
        let x_grfa = cls
            .role_successors
            .get(&("x".to_string(), "grandfather".to_string()));
        assert!(
            x_grfa.map(|s| s.contains("z")).unwrap_or(false),
            "x should have grandfather z via chain"
        );
    }

    #[test]
    fn test_multiple_property_chains() {
        let mut r = Owl2ElReasoner::new();
        r.add_property_chain(vec!["p".to_string(), "q".to_string()], "pq");
        r.add_property_chain(vec!["q".to_string(), "r".to_string()], "qr");
        r.add_role_assertion("a", "p", "b");
        r.add_role_assertion("b", "q", "c");
        r.add_role_assertion("c", "r", "d");

        let cls = r.classify().expect("classify failed");
        let a_pq = cls
            .role_successors
            .get(&("a".to_string(), "pq".to_string()));
        let b_qr = cls
            .role_successors
            .get(&("b".to_string(), "qr".to_string()));
        assert!(a_pq.map(|s| s.contains("c")).unwrap_or(false), "a pq c");
        assert!(b_qr.map(|s| s.contains("d")).unwrap_or(false), "b qr d");
    }

    #[test]
    fn test_subsumption_hierarchy_not_symmetric() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Child", "Parent");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("Child", "Parent"));
        assert!(!cls.is_subclass_of("Parent", "Child"));
    }

    #[test]
    fn test_role_successors_absent_for_unknown_role() {
        let mut r = Owl2ElReasoner::new();
        r.add_role_assertion("x", "knows", "y");
        let cls = r.classify().expect("classify failed");
        let unknown = cls
            .role_successors
            .get(&("x".to_string(), "unknown".to_string()));
        assert!(unknown.is_none() || unknown.map(|s| s.is_empty()).unwrap_or(true));
    }

    #[test]
    fn test_with_max_work_items() {
        let mut r = Owl2ElReasoner::new().with_max_work_items(10_000);
        r.add_subclass_of("A", "B");
        let cls = r.classify().expect("classify with custom limit failed");
        assert!(cls.is_subclass_of("A", "B"));
    }

    #[test]
    fn test_get_subclasses_empty_for_leaf() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        let subs = r.get_subclasses("Dog").expect("get_subclasses failed");
        // Dog has no explicit subclasses
        assert!(
            subs.is_empty(),
            "Dog should have no subclasses, got {:?}",
            subs
        );
        Ok(())
    }

    #[test]
    fn test_get_superclasses_root_does_not_include_subclasses() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_subclass_of("Cat", "Animal");
        let supers = r
            .get_superclasses("Animal")
            .expect("get_superclasses failed");
        // Animal's superclasses should not include Dog or Cat
        assert!(
            !supers.contains(&"Dog".to_string()),
            "Animal superclasses should not include Dog"
        );
        assert!(
            !supers.contains(&"Cat".to_string()),
            "Animal superclasses should not include Cat"
        );
    }
}
