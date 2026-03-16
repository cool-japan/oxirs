#[test]
fn test_combined_class_and_property_rewriting() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("Cat"), sup: QlConcept::named("Pet") },
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("ownedBy"),
            sup: QlRole::named("relatedTo"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![
        type_atom("x", "Pet"),
        prop_atom("x", "relatedTo", "y"),
    ]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(!result.is_empty());
    assert!(result.len() >= 2);
}

#[test]
fn test_head_variables_preserved() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::new(
        vec![type_atom("x", "Person")],
        vec!["x".to_string()],
    );
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    for cq in &result.queries {
        assert!(cq.head_variables.contains(&"x".to_string()));
    }
}

#[test]
fn test_constant_in_type_atom() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Student"),
        sup: QlConcept::named("Person"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = QueryAtom::TypeAtom {
        individual: QueryTerm::constant("alice"),
        class: "Person".to_string(),
    };
    let cq = ConjunctiveQuery::with_atoms(vec![atom]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Should include alice:Student
    let has_student = result.queries.iter().any(|cq| {
        cq.atoms.iter().any(|a| {
            matches!(a, QueryAtom::TypeAtom { individual: QueryTerm::Constant(c), class, .. }
                if c == "alice" && class == "Student")
        })
    });
    assert!(has_student);
}

#[test]
fn test_classify_idempotent() {
    let mut tbox = Owl2QLTBox::new();
    tbox.add_axiom(QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("B"),
    });
    tbox.classify().expect("first classify failed");
    // Calling classify again should not panic
    tbox.classify().expect("second classify failed");
    assert!(tbox.superclasses("A").contains("B"));
}

#[test]
fn test_tbox_clone() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("B"),
    }]);
    let tbox2 = tbox.clone();
    assert!(tbox2.superclasses("A").contains("B"));
}

#[test]
fn test_rewriter_tbox_accessor() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let _ = rewriter.tbox();
}

// ======================================================================
// 8. Edge cases and regression tests
// ======================================================================

#[test]
fn test_no_self_loop_superclass() {
    // A ⊑ A is possible but should not cause infinite loops
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("A"),
    }]);
    // Just verify no panic
    let _ = tbox.superclasses("A");
}

#[test]
fn test_rewrite_empty_query() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Empty query has one trivial rewriting (itself)
    assert_eq!(result.len(), 1);
}

#[test]
fn test_property_chain_three_deep() {
    let tbox = make_tbox(vec![
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("P1"),
            sup: QlRole::named("P2"),
        },
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("P2"),
            sup: QlRole::named("P3"),
        },
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("P3"),
            sup: QlRole::named("P4"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "P4", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_property(&result, "x", "P1", "y"));
    assert!(contains_property(&result, "x", "P2", "y"));
    assert!(contains_property(&result, "x", "P3", "y"));
    assert!(contains_property(&result, "x", "P4", "y"));
}

#[test]
fn test_inverse_property_chain() {
    let tbox = make_tbox(vec![
        QlAxiom::InverseObjectProperties("hasParent".to_string(), "hasChild".to_string()),
        QlAxiom::InverseObjectProperties("hasMother".to_string(), "hasDaughter".to_string()),
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("hasMother"),
            sup: QlRole::named("hasParent"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "hasParent", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_property(&result, "x", "hasParent", "y"));
    // hasMother ⊑ hasParent, so ?x hasMother ?y
    assert!(contains_property(&result, "x", "hasMother", "y"));
    // inverseOf(hasParent, hasChild): ?y hasChild ?x
    assert!(contains_property(&result, "y", "hasChild", "x"));
}

#[test]
fn test_multiple_inverse_declarations() {
    let tbox = make_tbox(vec![
        QlAxiom::InverseObjectProperties("P".to_string(), "Q".to_string()),
        QlAxiom::InverseObjectProperties("P".to_string(), "R".to_string()),
    ]);
    let invs = tbox.inverse_properties("P");
    assert!(invs.contains("Q"));
    assert!(invs.contains("R"));
}

#[test]
fn test_domain_multiple_properties() {
    let tbox = make_tbox(vec![
        QlAxiom::ObjectPropertyDomain {
            property: "P1".to_string(),
            domain: "A".to_string(),
        },
        QlAxiom::ObjectPropertyDomain {
            property: "P2".to_string(),
            domain: "A".to_string(),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "A")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    let has_p1 = result.queries.iter().any(|cq| {
        cq.atoms.iter().any(|a| {
            matches!(a, QueryAtom::PropertyAtom { property, .. } if property == "P1")
        })
    });
    let has_p2 = result.queries.iter().any(|cq| {
        cq.atoms.iter().any(|a| {
            matches!(a, QueryAtom::PropertyAtom { property, .. } if property == "P2")
        })
    });
    assert!(has_p1);
    assert!(has_p2);
}

#[test]
fn test_range_multiple_properties() {
    let tbox = make_tbox(vec![
        QlAxiom::ObjectPropertyRange {
            property: "P1".to_string(),
            range: "B".to_string(),
        },
        QlAxiom::ObjectPropertyRange {
            property: "P2".to_string(),
            range: "B".to_string(),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "B")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(!result.is_empty());
}

#[test]
fn test_axiom_display_error() {
    let err = QlError::InvalidAxiom("test error".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("test error"));
}

#[test]
fn test_rewriting_limit_error() {
    let err = QlError::RewritingLimitExceeded(100);
    let msg = format!("{err}");
    assert!(msg.contains("100"));
}

#[test]
fn test_inconsistency_error() {
    let err = QlError::Inconsistency("disjoint".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("disjoint"));
}

#[test]
fn test_unsupported_construct_error() {
    let err = QlError::UnsupportedConstruct("owl:allValuesFrom".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("owl:allValuesFrom"));
}

#[test]
fn test_conjunctive_query_new() {
    let cq = ConjunctiveQuery::new(
        vec![type_atom("x", "A")],
        vec!["x".to_string()],
    );
    assert_eq!(cq.head_variables, vec!["x".to_string()]);
    assert_eq!(cq.atoms.len(), 1);
}

#[test]
fn test_conjunctive_query_with_atoms_deduplicates_vars() {
    let cq = ConjunctiveQuery::with_atoms(vec![
        type_atom("x", "A"),
        prop_atom("x", "P", "y"),
    ]);
    // x should appear once in head_variables
    let x_count = cq.head_variables.iter().filter(|v| *v == "x").count();
    assert_eq!(x_count, 1);
}

#[test]
fn test_subclass_does_not_infer_sibling() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("Dog"), sup: QlConcept::named("Animal") },
        QlAxiom::SubClassOf { sub: QlConcept::named("Cat"), sup: QlConcept::named("Animal") },
    ]);
    // Dog should not be superclass of Cat
    assert!(!tbox.superclasses("Cat").contains("Dog"));
    assert!(!tbox.superclasses("Dog").contains("Cat"));
}

#[test]
fn test_superclasses_does_not_include_self() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Dog"),
        sup: QlConcept::named("Animal"),
    }]);
    assert!(!tbox.superclasses("Dog").contains("Dog"));
    assert!(!tbox.superclasses("Animal").contains("Animal"));
}

#[test]
fn test_subclasses_does_not_include_self() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Dog"),
        sup: QlConcept::named("Animal"),
    }]);
    assert!(!tbox.subclasses("Animal").contains("Animal"));
}

#[test]
fn test_default_tbox() {
    let tbox = Owl2QLTBox::default();
    assert!(tbox.superclasses("A").is_empty());
}

#[test]
fn test_add_axioms_bulk() {
    let mut tbox = Owl2QLTBox::new();
    tbox.add_axioms(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("A"), sup: QlConcept::named("B") },
        QlAxiom::SubClassOf { sub: QlConcept::named("B"), sup: QlConcept::named("C") },
    ]);
    tbox.classify().expect("classify failed");
    assert!(tbox.superclasses("A").contains("C"));
}

#[test]
fn test_owl2ql_real_world_scenario_foaf() {
    // FOAF-like ontology: Agent ⊑ Thing, Person ⊑ Agent, Organization ⊑ Agent
    // knows subPropertyOf knows (self), isPrimaryTopicOf inverseOf primaryTopic
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("Person"), sup: QlConcept::named("Agent") },
        QlAxiom::SubClassOf { sub: QlConcept::named("Organization"), sup: QlConcept::named("Agent") },
        QlAxiom::SubClassOf { sub: QlConcept::named("Agent"), sup: QlConcept::named("Thing") },
        QlAxiom::InverseObjectProperties(
            "isPrimaryTopicOf".to_string(),
            "primaryTopic".to_string(),
        ),
    ]);
    let rewriter = QueryRewriter::new(tbox);

    // Query: ?x:Agent → should include Person and Organization
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Agent")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Agent"));
    assert!(contains_type(&result, "x", "Person"));
    assert!(contains_type(&result, "x", "Organization"));

    // Query: ?x:Thing → should include Agent, Person, Organization
    let cq2 = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Thing")]);
    let result2 = rewriter.rewrite_query(&cq2).expect("rewrite failed");
    assert!(contains_type(&result2, "x", "Thing"));
    assert!(contains_type(&result2, "x", "Agent"));
    assert!(contains_type(&result2, "x", "Person"));
    assert!(contains_type(&result2, "x", "Organization"));
}

#[test]
fn test_owl2ql_real_world_property_inverse() {
    let tbox = make_tbox(vec![
        QlAxiom::InverseObjectProperties("parent".to_string(), "child".to_string()),
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("mother"),
            sup: QlRole::named("parent"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "parent", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // ?x parent ?y, ?x mother ?y, ?y child ?x
    assert!(contains_property(&result, "x", "parent", "y"));
    assert!(contains_property(&result, "x", "mother", "y"));
    assert!(contains_property(&result, "y", "child", "x"));
}

#[test]
fn test_ql_axiom_clone() {
    let ax = QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("B"),
    };
    let ax2 = ax.clone();
    assert_eq!(ax, ax2);
}

#[test]
fn test_ql_concept_eq() {
    assert_eq!(QlConcept::named("A"), QlConcept::named("A"));
    assert_ne!(QlConcept::named("A"), QlConcept::named("B"));
    assert_ne!(QlConcept::Top, QlConcept::Bottom);
}

#[test]
fn test_ql_role_eq() {
    assert_eq!(QlRole::named("P"), QlRole::named("P"));
    assert_ne!(QlRole::named("P"), QlRole::inverse("P"));
}

#[test]
fn test_query_term_eq() {
    assert_eq!(QueryTerm::var("x"), QueryTerm::var("x"));
    assert_ne!(QueryTerm::var("x"), QueryTerm::constant("x"));
}

#[test]
fn test_query_atom_eq() {
    let a1 = type_atom("x", "A");
    let a2 = type_atom("x", "A");
    assert_eq!(a1, a2);
}

#[test]
fn test_query_atom_prop_eq() {
    let a1 = prop_atom("x", "P", "y");
    let a2 = prop_atom("x", "P", "y");
    assert_eq!(a1, a2);
    let a3 = prop_atom("x", "Q", "y");
    assert_ne!(a1, a3);
}

#[test]
fn test_ql_error_clone() {
    let e = QlError::InvalidAxiom("test".to_string());
    let e2 = e.clone();
    assert_eq!(e, e2);
}

#[test]
fn test_superproperties_no_tbox() {
    let tbox = make_tbox(vec![]);
    assert!(tbox.superproperties("P").is_empty());
}

#[test]
fn test_inverse_properties_no_tbox() {
    let tbox = make_tbox(vec![]);
    assert!(tbox.inverse_properties("P").is_empty());
}

#[test]
fn test_domain_of_no_tbox() {
    let tbox = make_tbox(vec![]);
    assert!(tbox.domain_of("P").is_empty());
}

#[test]
fn test_range_of_no_tbox() {
    let tbox = make_tbox(vec![]);
    assert!(tbox.range_of("P").is_empty());
}

#[test]
fn test_all_subproperties_includes_self() {
    let tbox = make_tbox(vec![]);
    let subs = tbox.all_subproperties_of("P", false);
    assert!(subs.contains(&QlRole::Named("P".to_string())));
}

#[test]
fn test_all_subproperties_inverse_includes_self() {
    let tbox = make_tbox(vec![]);
    let subs = tbox.all_subproperties_of("P", true);
    assert!(subs.contains(&QlRole::Inverse("P".to_string())));
}

#[test]
fn test_unfold_property_with_inverse_subproperty() -> anyhow::Result<()> {
    // Q⁻ ⊑ P means if (y,x):Q then (x,y):P
    // So querying ?x P ?y should also yield ?y Q ?x
    let tbox = make_tbox(vec![QlAxiom::SubObjectPropertyOf {
        sub: QlRole::Inverse("Q".to_string()),
        sup: QlRole::Named("P".to_string()),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = prop_atom("x", "P", "y");
    let unfolded = rewriter.unfold_atom(&atom);
    let has_q_inv = unfolded.iter().any(|a| {
        if let QueryAtom::PropertyAtom { subject, property, object } = a {
            property == "Q" && subject == &QueryTerm::var("y") && object == &QueryTerm::var("x")
        } else {
            false
        }
    });
    assert!(has_q_inv, "should contain ?y Q ?x from inverse subproperty");
    Ok(())
}

#[test]
fn test_three_class_equivalence_chain() {
    // A ≡ B, B ≡ C → A ⊑ B, B ⊑ A, B ⊑ C, C ⊑ B → A ⊑ C (transitivity)
    let tbox = make_tbox(vec![
        QlAxiom::EquivalentClasses(QlConcept::named("A"), QlConcept::named("B")),
        QlAxiom::EquivalentClasses(QlConcept::named("B"), QlConcept::named("C")),
    ]);
    let rewriter = QueryRewriter::new(tbox.clone());
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "C")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "C"));
    assert!(contains_type(&result, "x", "B"));
    assert!(contains_type(&result, "x", "A"));
}

#[test]
fn test_full_pipeline_classify_then_rewrite() {
    let mut tbox = Owl2QLTBox::new();
    tbox.add_axiom(QlAxiom::SubClassOf {
        sub: QlConcept::named("Employee"),
        sup: QlConcept::named("Person"),
    });
    tbox.add_axiom(QlAxiom::ObjectPropertyDomain {
        property: "manages".to_string(),
        domain: "Manager".to_string(),
    });
    tbox.add_axiom(QlAxiom::SubClassOf {
        sub: QlConcept::named("Manager"),
        sup: QlConcept::named("Employee"),
    });
    tbox.classify().expect("classify failed");

    let rewriter = QueryRewriter::new(tbox.clone());
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Person")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");

    // Should include Employee, Manager, and Person
    assert!(contains_type(&result, "x", "Person"));
    assert!(contains_type(&result, "x", "Employee"));
    assert!(contains_type(&result, "x", "Manager"));
}

#[test]
fn test_rewrite_preserves_constants() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Dog"),
        sup: QlConcept::named("Animal"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![QueryAtom::PropertyAtom {
        subject: QueryTerm::constant("alice"),
        property: "knows".to_string(),
        object: QueryTerm::constant("bob"),
    }]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Constants should be preserved
    let has_alice = result.queries.iter().any(|cq| {
        cq.atoms.iter().any(|a| {
            matches!(a, QueryAtom::PropertyAtom { subject: QueryTerm::Constant(c), .. } if c == "alice")
        })
    });
    assert!(has_alice);
}

#[test]
fn test_subproperty_of_inverse_role() {
    // P ⊑ Q⁻ means if (x,y):P then (y,x):Q
    let tbox = make_tbox(vec![QlAxiom::SubObjectPropertyOf {
        sub: QlRole::Named("P".to_string()),
        sup: QlRole::Inverse("Q".to_string()),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    // Query ?x Q ?y → includes ?y P ?x (since P ⊑ Q⁻)
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "Q", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite");
    assert!(!result.is_empty());
}

#[test]
fn test_rewrite_query_multiple_atoms_independent() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("A1"), sup: QlConcept::named("A") },
        QlAxiom::SubClassOf { sub: QlConcept::named("B1"), sup: QlConcept::named("B") },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![
        type_atom("x", "A"),
        type_atom("y", "B"),
    ]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Should include combos of (A or A1) x (B or B1)
    assert!(result.len() >= 4);
}

#[test]
fn test_tbox_with_all_axiom_types() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("A"), sup: QlConcept::named("B") },
        QlAxiom::EquivalentClasses(QlConcept::named("C"), QlConcept::named("D")),
        QlAxiom::SubObjectPropertyOf { sub: QlRole::named("P"), sup: QlRole::named("Q") },
        QlAxiom::EquivalentObjectProperties(QlRole::named("R"), QlRole::named("S")),
        QlAxiom::InverseObjectProperties("T".to_string(), "U".to_string()),
        QlAxiom::ObjectPropertyDomain { property: "V".to_string(), domain: "W".to_string() },
        QlAxiom::ObjectPropertyRange { property: "X".to_string(), range: "Y".to_string() },
        QlAxiom::DisjointClasses(QlConcept::named("E"), QlConcept::named("F")),
        QlAxiom::DisjointObjectProperties(QlRole::named("G"), QlRole::named("H")),
    ]);
    // Should not panic
    assert!(tbox.superclasses("A").contains("B"));
    assert!(tbox.are_disjoint("E", "F"));
}

#[test]
fn test_large_subclass_hierarchy() {
    // 10-level chain
    let mut axioms = Vec::new();
    for i in 0..9 {
        axioms.push(QlAxiom::SubClassOf {
            sub: QlConcept::named(format!("C{i}")),
            sup: QlConcept::named(format!("C{}", i + 1)),
        });
    }
    let tbox = make_tbox(axioms);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "C9")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert_eq!(result.len(), 10); // C0..C9
}

#[test]
fn test_rewrite_non_existent_class() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "NonExistent")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert_eq!(result.len(), 1);
    assert!(contains_type(&result, "x", "NonExistent"));
}

#[test]
fn test_all_subsumed_by_transitive() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("A"), sup: QlConcept::named("B") },
        QlAxiom::SubClassOf { sub: QlConcept::named("B"), sup: QlConcept::named("C") },
    ]);
    let subsumed = tbox.all_subsumed_by("C");
    assert!(subsumed.contains("C"));
    assert!(subsumed.contains("B"));
    assert!(subsumed.contains("A"));
}

#[test]
fn test_ql_role_ordering() {
    let mut roles = [
        QlRole::named("Z"),
        QlRole::inverse("A"),
        QlRole::named("A"),
    ];
    roles.sort();
    // Just check no panic (ordering is well-defined)
    assert_eq!(roles.len(), 3);
}

#[test]
fn test_ql_concept_ordering() {
    let mut concepts = [
        QlConcept::named("Z"),
        QlConcept::Top,
        QlConcept::named("A"),
        QlConcept::Bottom,
    ];
    concepts.sort();
    assert_eq!(concepts.len(), 4);
}

#[test]
fn test_rewritten_query_len_matches_queries() {
    let mut r = RewrittenQuery::new();
    assert_eq!(r.len(), 0);
    r.add(ConjunctiveQuery::with_atoms(vec![type_atom("x", "A")]));
    r.add(ConjunctiveQuery::with_atoms(vec![type_atom("x", "B")]));
    assert_eq!(r.len(), 2);
    assert!(!r.is_empty());
}

#[test]
fn test_domain_from_subclass_of_some_values_top() {
    // ∃P.⊤ ⊑ A: using P implies being of type A
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::SomeValuesTop { property: "P".to_string() },
        sup: QlConcept::Named("A".to_string()),
    }]);
    assert!(tbox.domain_of("P").contains("A"));
}

#[test]
fn test_range_from_subclass_of_some_values_top_inverse() {
    // ∃P⁻.⊤ ⊑ B: being used as object of P implies being of type B
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::SomeValuesTopInverse { property: "P".to_string() },
        sup: QlConcept::Named("B".to_string()),
    }]);
    assert!(tbox.range_of("P").contains("B"));
}

// ======================================================================
// 9. Union query rewriting tests (NEW — ObjectUnionOf support)
// ======================================================================

#[test]
fn test_union_axiom_basic_indexing() {
    // Dog ⊑ Animal ⊔ Pet
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Dog"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Animal"),
            ConceptExpr::named("Pet"),
        ]),
    }]);
    // Dog should be in the union_axioms index
    let disjuncts = tbox.union_axiom_disjuncts("Dog");
    assert!(!disjuncts.is_empty(), "Dog should have union axiom disjuncts");
    let flat: Vec<String> = disjuncts.into_iter().flatten().collect();
    assert!(flat.contains(&"Animal".to_string()));
    assert!(flat.contains(&"Pet".to_string()));
}

#[test]
fn test_union_rev_index() {
    // Dog ⊑ Animal ⊔ Pet
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Dog"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Animal"),
            ConceptExpr::named("Pet"),
        ]),
    }]);
    // Animal should be in the reverse index pointing to Dog
    let sources = tbox.classes_with_union_disjunct("Animal");
    assert!(sources.contains("Dog"), "Animal should point back to Dog");
    let sources_pet = tbox.classes_with_union_disjunct("Pet");
    assert!(sources_pet.contains("Dog"), "Pet should point back to Dog");
}

#[test]
fn test_has_union_axioms_true() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("C"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("A"),
            ConceptExpr::named("B"),
        ]),
    }]);
    assert!(tbox.has_union_axioms());
}

#[test]
fn test_has_union_axioms_false() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("B"),
    }]);
    assert!(!tbox.has_union_axioms());
}

#[test]
fn test_union_rewriting_type_atom_gets_source_class() {
    // Dog ⊑ Animal ⊔ Pet
    // Query: ?x:Animal
    // Should include ?x:Dog (since Dog individuals may be Animals in the union)
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Dog"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Animal"),
            ConceptExpr::named("Pet"),
        ]),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Animal")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");

    assert!(contains_type(&result, "x", "Animal"), "original atom must be present");
    assert!(contains_type(&result, "x", "Dog"), "Dog must be in union rewriting of Animal");
}

#[test]
fn test_union_rewriting_both_disjuncts_get_source() {
    // Cat ⊑ Mammal ⊔ Vertebrate
    // Querying ?x:Mammal should include ?x:Cat
    // Querying ?x:Vertebrate should also include ?x:Cat
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Cat"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Mammal"),
            ConceptExpr::named("Vertebrate"),
        ]),
    }]);
    let rewriter = QueryRewriter::new(tbox);

    // Query Mammal
    let cq1 = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Mammal")]);
    let r1 = rewriter.rewrite_query(&cq1).expect("rewrite failed");
    assert!(contains_type(&r1, "x", "Cat"), "Cat should appear in Mammal rewriting");

    // Query Vertebrate
    let cq2 = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Vertebrate")]);
    let r2 = rewriter.rewrite_query(&cq2).expect("rewrite failed");
    assert!(contains_type(&r2, "x", "Cat"), "Cat should appear in Vertebrate rewriting");
}

#[test]
fn test_union_with_subclass_combined() {
    // TBox:
    //   Poodle ⊑ Dog
    //   Dog ⊑ Animal ⊔ Pet
    // Query: ?x:Animal
    // Should include: ?x:Animal, ?x:Dog, ?x:Poodle
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Poodle"),
            sup: QlConcept::named("Dog"),
        },
        QlAxiom::SubClassOfUnion {
            sub: QlConcept::named("Dog"),
            sup_union: ConceptExpr::union_of(vec![
                ConceptExpr::named("Animal"),
                ConceptExpr::named("Pet"),
            ]),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Animal")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");

    assert!(contains_type(&result, "x", "Animal"));
    assert!(contains_type(&result, "x", "Dog"), "Dog must appear via union");
    // Poodle ⊑ Dog, and Dog is in union for Animal, so Poodle should also appear
    assert!(contains_type(&result, "x", "Poodle"), "Poodle must appear via Dog union");
}

#[test]
fn test_union_multiple_classes_same_disjunct() {
    // Lizard ⊑ Reptile ⊔ Animal
    // Snake ⊑ Reptile ⊔ Animal
    // Query: ?x:Reptile should include both Lizard and Snake
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOfUnion {
            sub: QlConcept::named("Lizard"),
            sup_union: ConceptExpr::union_of(vec![
                ConceptExpr::named("Reptile"),
                ConceptExpr::named("Animal"),
            ]),
        },
        QlAxiom::SubClassOfUnion {
            sub: QlConcept::named("Snake"),
            sup_union: ConceptExpr::union_of(vec![
                ConceptExpr::named("Reptile"),
                ConceptExpr::named("Animal"),
            ]),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Reptile")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");

    assert!(contains_type(&result, "x", "Reptile"));
    assert!(contains_type(&result, "x", "Lizard"), "Lizard must appear");
    assert!(contains_type(&result, "x", "Snake"), "Snake must appear");
}

#[test]
fn test_union_three_way_disjunction() {
    // Vehicle ⊑ Car ⊔ Truck ⊔ Motorcycle
    // Query: ?x:Car should include ?x:Vehicle
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Vehicle"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Car"),
            ConceptExpr::named("Truck"),
            ConceptExpr::named("Motorcycle"),
        ]),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Car")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");

    assert!(contains_type(&result, "x", "Car"));
    assert!(contains_type(&result, "x", "Vehicle"), "Vehicle must appear via union");
}

#[test]
fn test_union_query_rewrite_convenience_fn() {
    // Using the rewrite_query_union convenience function
    let tbox = build_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Dog"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Animal"),
            ConceptExpr::named("Pet"),
        ]),
    }])
    .expect("build_tbox failed");

    let result = rewrite_query_union(
        vec![type_atom("x", "Animal")],
        &tbox,
    )
    .expect("rewrite failed");

    assert!(contains_type(&result, "x", "Animal"));
    assert!(contains_type(&result, "x", "Dog"));
}

#[test]
fn test_union_siblings_for() {
    // A ⊑ B ⊔ C
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("A"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("B"),
            ConceptExpr::named("C"),
        ]),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let siblings = rewriter.union_siblings_for("B");
    assert!(!siblings.is_empty(), "B should have union siblings");
    let flat: Vec<String> = siblings.into_iter().flatten().collect();
    assert!(flat.contains(&"B".to_string()));
    assert!(flat.contains(&"C".to_string()));
}

#[test]
fn test_are_union_siblings_true() {
    // A ⊑ B ⊔ C → B and C are siblings
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("A"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("B"),
            ConceptExpr::named("C"),
        ]),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    assert!(rewriter.are_union_siblings("B", "C"), "B and C should be union siblings");
    assert!(rewriter.are_union_siblings("C", "B"), "union siblings are symmetric");
}

#[test]
fn test_are_union_siblings_false() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("B"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    assert!(!rewriter.are_union_siblings("A", "B"), "no union axiom, not siblings");
}

#[test]
fn test_conjunction_satisfiable_no_disjointness() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    assert!(rewriter.conjunction_satisfiable(&["A", "B", "C"]));
}

#[test]
fn test_conjunction_satisfiable_with_disjointness() {
    let tbox = make_tbox(vec![QlAxiom::DisjointClasses(
        QlConcept::named("Cat"),
        QlConcept::named("Dog"),
    )]);
    let rewriter = QueryRewriter::new(tbox);
    // Cat ∩ Dog is unsatisfiable
    assert!(!rewriter.conjunction_satisfiable(&["Cat", "Dog"]));
    // Cat alone is satisfiable
    assert!(rewriter.conjunction_satisfiable(&["Cat"]));
}

#[test]
fn test_union_axiom_no_union_in_empty_tbox() {
    let tbox = make_tbox(vec![]);
    assert!(tbox.union_axiom_disjuncts("A").is_empty());
    assert!(tbox.classes_with_union_disjunct("A").is_empty());
}

#[test]
fn test_concept_expr_union_branches() {
    // ConceptExpr::Union(A, B, C).union_branches() returns ONE branch with ALL disjuncts.
    // This is the correct semantics: a single union axiom has all its disjuncts together.
    let expr = ConceptExpr::union_of(vec![
        ConceptExpr::named("A"),
        ConceptExpr::named("B"),
        ConceptExpr::named("C"),
    ]);
    let branches = expr.union_branches();
    // One branch entry containing all three disjuncts
    assert_eq!(branches.len(), 1, "union expression should produce one branch with all disjuncts");
    let disjuncts = &branches[0];
    assert_eq!(disjuncts.len(), 3);
    assert!(disjuncts.contains(&"A".to_string()));
    assert!(disjuncts.contains(&"B".to_string()));
    assert!(disjuncts.contains(&"C".to_string()));
}

#[test]
fn test_concept_expr_atomic_names() {
    let expr = ConceptExpr::intersection_of(vec![
        ConceptExpr::named("A"),
        ConceptExpr::named("B"),
    ]);
    let names = expr.atomic_names();
    assert!(names.contains(&"A".to_string()));
    assert!(names.contains(&"B".to_string()));
}

#[test]
fn test_concept_expr_has_union() {
    let union_expr = ConceptExpr::union_of(vec![
        ConceptExpr::named("A"),
        ConceptExpr::named("B"),
    ]);
    assert!(union_expr.has_union());

    let atomic_expr = ConceptExpr::named("A");
    assert!(!atomic_expr.has_union());
}

#[test]
fn test_union_rewriting_union_aware_method() {
    // Dog ⊑ Animal ⊔ Pet
    // rewrite_query_union_aware should produce the same result as rewrite_query
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Dog"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Animal"),
            ConceptExpr::named("Pet"),
        ]),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Animal")]);
    let result = rewriter.rewrite_query_union_aware(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Animal"));
    assert!(contains_type(&result, "x", "Dog"));
}

#[test]
fn test_union_combined_with_property_rewriting() {
    // TBox:
    //   Dog ⊑ Animal ⊔ Pet
    //   dog_of ⊑ pet_of
    // Query: ?x:Animal ∧ ?x pet_of ?y
    // Should include variants with Dog substituted for Animal and hasMother for hasParent
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOfUnion {
            sub: QlConcept::named("Dog"),
            sup_union: ConceptExpr::union_of(vec![
                ConceptExpr::named("Animal"),
                ConceptExpr::named("Pet"),
            ]),
        },
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("dog_of"),
            sup: QlRole::named("pet_of"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![
        type_atom("x", "Animal"),
        prop_atom("x", "pet_of", "y"),
    ]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(!result.is_empty());
    // Should contain original + subproperty expansions + union expansions
    assert!(result.len() >= 2, "should have at least 2 rewritings");
}

#[test]
fn test_union_axiom_clone_in_tbox() {
    // Verify that union axiom data is preserved through TBox clone
    let tbox = make_tbox(vec![QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Dog"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Animal"),
            ConceptExpr::named("Pet"),
        ]),
    }]);
    let tbox2 = tbox.clone();
    assert!(tbox2.has_union_axioms(), "cloned TBox should preserve union axioms");
    let disjuncts = tbox2.union_axiom_disjuncts("Dog");
    assert!(!disjuncts.is_empty());
}

#[test]
fn test_union_axiom_variant_clone() {
    let ax = QlAxiom::SubClassOfUnion {
        sub: QlConcept::named("Dog"),
        sup_union: ConceptExpr::union_of(vec![
            ConceptExpr::named("Animal"),
            ConceptExpr::named("Pet"),
        ]),
    };
    let ax2 = ax.clone();
    assert_eq!(ax, ax2);
}
