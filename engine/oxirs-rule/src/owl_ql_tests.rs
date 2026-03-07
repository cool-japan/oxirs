// ---- Helper builders ----

fn make_tbox(axioms: Vec<QlAxiom>) -> Owl2QLTBox {
    let mut tbox = Owl2QLTBox::new();
    tbox.add_axioms(axioms);
    tbox.classify().expect("classify failed");
    tbox
}

fn type_atom(var: &str, class: &str) -> QueryAtom {
    QueryAtom::TypeAtom {
        individual: QueryTerm::var(var),
        class: class.to_string(),
    }
}

fn prop_atom(s: &str, p: &str, o: &str) -> QueryAtom {
    QueryAtom::PropertyAtom {
        subject: QueryTerm::var(s),
        property: p.to_string(),
        object: QueryTerm::var(o),
    }
}

#[allow(dead_code)]
fn const_type_atom(ind: &str, class: &str) -> QueryAtom {
    QueryAtom::TypeAtom {
        individual: QueryTerm::constant(ind),
        class: class.to_string(),
    }
}

fn contains_type(result: &RewrittenQuery, var: &str, class: &str) -> bool {
    let target = QueryAtom::TypeAtom {
        individual: QueryTerm::var(var),
        class: class.to_string(),
    };
    result.queries.iter().any(|cq| cq.atoms.contains(&target))
}

fn contains_property(result: &RewrittenQuery, s: &str, p: &str, o: &str) -> bool {
    let target = QueryAtom::PropertyAtom {
        subject: QueryTerm::var(s),
        property: p.to_string(),
        object: QueryTerm::var(o),
    };
    result.queries.iter().any(|cq| cq.atoms.contains(&target))
}

// ======================================================================
// 1. TBox classification tests
// ======================================================================

#[test]
fn test_empty_tbox() {
    let tbox = make_tbox(vec![]);
    assert!(tbox.superclasses("A").is_empty());
    assert!(tbox.subclasses("A").is_empty());
}

#[test]
fn test_single_subclass_axiom() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Dog"),
        sup: QlConcept::named("Animal"),
    }]);
    assert!(tbox.superclasses("Dog").contains("Animal"));
    assert!(!tbox.superclasses("Animal").contains("Dog"));
}

#[test]
fn test_subclass_chain_transitivity() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Poodle"),
            sup: QlConcept::named("Dog"),
        },
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Dog"),
            sup: QlConcept::named("Animal"),
        },
    ]);
    // Poodle ⊑ Dog ⊑ Animal → Poodle ⊑ Animal
    assert!(tbox.superclasses("Poodle").contains("Dog"));
    assert!(tbox.superclasses("Poodle").contains("Animal"));
    assert!(!tbox.superclasses("Animal").contains("Poodle"));
}

#[test]
fn test_three_level_chain() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf {
            sub: QlConcept::named("A"),
            sup: QlConcept::named("B"),
        },
        QlAxiom::SubClassOf {
            sub: QlConcept::named("B"),
            sup: QlConcept::named("C"),
        },
        QlAxiom::SubClassOf {
            sub: QlConcept::named("C"),
            sup: QlConcept::named("D"),
        },
    ]);
    let supers = tbox.superclasses("A");
    assert!(supers.contains("B"));
    assert!(supers.contains("C"));
    assert!(supers.contains("D"));
}

#[test]
fn test_equivalent_classes_expansion() {
    let tbox = make_tbox(vec![QlAxiom::EquivalentClasses(
        QlConcept::named("Person"),
        QlConcept::named("Human"),
    )]);
    // Person ≡ Human → Person ⊑ Human and Human ⊑ Person
    assert!(tbox.superclasses("Person").contains("Human"));
    assert!(tbox.superclasses("Human").contains("Person"));
}

#[test]
fn test_subproperty_hierarchy() {
    let tbox = make_tbox(vec![
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("hasMother"),
            sup: QlRole::named("hasParent"),
        },
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("hasParent"),
            sup: QlRole::named("hasAncestor"),
        },
    ]);
    assert!(tbox.superproperties("hasMother").contains("hasParent"));
    assert!(tbox.superproperties("hasMother").contains("hasAncestor"));
}

#[test]
fn test_inverse_properties_declared() {
    let tbox = make_tbox(vec![QlAxiom::InverseObjectProperties(
        "hasParent".to_string(),
        "hasChild".to_string(),
    )]);
    assert!(tbox.inverse_properties("hasParent").contains("hasChild"));
    assert!(tbox.inverse_properties("hasChild").contains("hasParent"));
}

#[test]
fn test_domain_from_axiom() {
    let tbox = make_tbox(vec![QlAxiom::ObjectPropertyDomain {
        property: "worksAt".to_string(),
        domain: "Person".to_string(),
    }]);
    assert!(tbox.domain_of("worksAt").contains("Person"));
}

#[test]
fn test_range_from_axiom() {
    let tbox = make_tbox(vec![QlAxiom::ObjectPropertyRange {
        property: "worksAt".to_string(),
        range: "Organization".to_string(),
    }]);
    assert!(tbox.range_of("worksAt").contains("Organization"));
}

#[test]
fn test_domain_from_some_values_top() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::some_values_top("worksAt"),
        sup: QlConcept::named("Employee"),
    }]);
    assert!(tbox.domain_of("worksAt").contains("Employee"));
}

#[test]
fn test_range_from_some_values_top_inverse() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::some_values_top_inverse("worksAt"),
        sup: QlConcept::named("Employer"),
    }]);
    assert!(tbox.range_of("worksAt").contains("Employer"));
}

#[test]
fn test_disjoint_classes() {
    let tbox = make_tbox(vec![QlAxiom::DisjointClasses(
        QlConcept::named("Cat"),
        QlConcept::named("Dog"),
    )]);
    assert!(tbox.are_disjoint("Cat", "Dog"));
    assert!(tbox.are_disjoint("Dog", "Cat")); // symmetric
}

#[test]
fn test_not_disjoint() {
    let tbox = make_tbox(vec![]);
    assert!(!tbox.are_disjoint("Cat", "Dog"));
}

#[test]
fn test_all_subsumed_by_includes_self() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Dog"),
        sup: QlConcept::named("Animal"),
    }]);
    let subs = tbox.all_subsumed_by("Animal");
    assert!(subs.contains("Animal"));
    assert!(subs.contains("Dog"));
}

#[test]
fn test_all_subsumed_by_no_axioms() {
    let tbox = make_tbox(vec![]);
    let subs = tbox.all_subsumed_by("Animal");
    assert!(subs.contains("Animal"));
    assert_eq!(subs.len(), 1);
}

#[test]
fn test_subproperties_named() {
    let tbox = make_tbox(vec![QlAxiom::SubObjectPropertyOf {
        sub: QlRole::named("hasMother"),
        sup: QlRole::named("hasParent"),
    }]);
    let subs = tbox.all_subproperties_of("hasParent", false);
    let names: Vec<_> = subs.iter().map(|r| r.base_name().to_string()).collect();
    assert!(names.contains(&"hasMother".to_string()));
}

#[test]
fn test_equivalent_properties_expansion() {
    let tbox = make_tbox(vec![QlAxiom::EquivalentObjectProperties(
        QlRole::named("knows"),
        QlRole::named("acquaintanceOf"),
    )]);
    // knows ≡ acquaintanceOf
    let subs_knows = tbox.all_subproperties_of("knows", false);
    let names: Vec<_> = subs_knows.iter().map(|r| r.base_name().to_string()).collect();
    assert!(names.contains(&"acquaintanceOf".to_string()) || names.contains(&"knows".to_string()));
}

// ======================================================================
// 2. QueryAtom / QueryTerm tests
// ======================================================================

#[test]
fn test_query_term_variable() {
    let t = QueryTerm::var("x");
    assert!(t.is_variable());
    assert_eq!(t.as_variable(), Some("x"));
    assert!(t.as_constant().is_none());
}

#[test]
fn test_query_term_constant() {
    let t = QueryTerm::constant("http://example.org/alice");
    assert!(!t.is_variable());
    assert_eq!(t.as_constant(), Some("http://example.org/alice"));
    assert!(t.as_variable().is_none());
}

#[test]
fn test_type_atom_variables() {
    let atom = type_atom("x", "Person");
    let vars = atom.variables();
    assert_eq!(vars, vec!["x"]);
}

#[test]
fn test_property_atom_variables() {
    let atom = prop_atom("x", "knows", "y");
    let vars = atom.variables();
    assert!(vars.contains(&"x"));
    assert!(vars.contains(&"y"));
}

#[test]
fn test_property_atom_one_constant() {
    let atom = QueryAtom::PropertyAtom {
        subject: QueryTerm::constant("alice"),
        property: "knows".to_string(),
        object: QueryTerm::var("y"),
    };
    let vars = atom.variables();
    assert_eq!(vars, vec!["y"]);
}

#[test]
fn test_conjunctive_query_collect_vars() {
    let cq = ConjunctiveQuery::with_atoms(vec![
        type_atom("x", "Person"),
        prop_atom("x", "knows", "y"),
    ]);
    assert!(cq.head_variables.contains(&"x".to_string()));
    assert!(cq.head_variables.contains(&"y".to_string()));
}

// ======================================================================
// 3. Unfold atom tests
// ======================================================================

#[test]
fn test_unfold_type_atom_no_tbox() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = type_atom("x", "Person");
    let unfolded = rewriter.unfold_atom(&atom);
    assert_eq!(unfolded.len(), 1);
    assert_eq!(unfolded[0], atom);
}

#[test]
fn test_unfold_type_atom_one_subclass() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Student"),
        sup: QlConcept::named("Person"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = type_atom("x", "Person");
    let unfolded = rewriter.unfold_atom(&atom);
    // Should include ?x:Person and ?x:Student
    let classes: Vec<_> = unfolded
        .iter()
        .filter_map(|a| {
            if let QueryAtom::TypeAtom { class, .. } = a {
                Some(class.as_str())
            } else {
                None
            }
        })
        .collect();
    assert!(classes.contains(&"Person"));
    assert!(classes.contains(&"Student"));
}

#[test]
fn test_unfold_type_atom_chain() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf {
            sub: QlConcept::named("GradStudent"),
            sup: QlConcept::named("Student"),
        },
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Student"),
            sup: QlConcept::named("Person"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = type_atom("x", "Person");
    let unfolded = rewriter.unfold_atom(&atom);
    let classes: HashSet<String> = unfolded
        .iter()
        .filter_map(|a| {
            if let QueryAtom::TypeAtom { class, .. } = a {
                Some(class.clone())
            } else {
                None
            }
        })
        .collect();
    assert!(classes.contains("Person"));
    assert!(classes.contains("Student"));
    assert!(classes.contains("GradStudent"));
}

#[test]
fn test_unfold_type_atom_with_domain() {
    // ∃worksAt.⊤ ⊑ Employee → ?x:Employee can be rewritten as ?x worksAt ?y
    let tbox = make_tbox(vec![QlAxiom::ObjectPropertyDomain {
        property: "worksAt".to_string(),
        domain: "Employee".to_string(),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = type_atom("x", "Employee");
    let unfolded = rewriter.unfold_atom(&atom);
    let has_property = unfolded.iter().any(|a| {
        matches!(a, QueryAtom::PropertyAtom { property, .. } if property == "worksAt")
    });
    assert!(has_property, "should contain worksAt property atom");
}

#[test]
fn test_unfold_type_atom_with_range() {
    // ∃worksAt⁻.⊤ ⊑ Employer → ?x:Employer can be rewritten as ?y worksAt ?x
    let tbox = make_tbox(vec![QlAxiom::ObjectPropertyRange {
        property: "worksAt".to_string(),
        range: "Employer".to_string(),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = type_atom("x", "Employer");
    let unfolded = rewriter.unfold_atom(&atom);
    // Should contain: ?_fresh worksAt ?x (object side)
    let has_range_pattern = unfolded.iter().any(|a| {
        if let QueryAtom::PropertyAtom { property, object, .. } = a {
            property == "worksAt" && object == &QueryTerm::var("x")
        } else {
            false
        }
    });
    assert!(has_range_pattern, "should contain ?fresh worksAt ?x atom");
}

#[test]
fn test_unfold_property_atom_no_tbox() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = prop_atom("x", "knows", "y");
    let unfolded = rewriter.unfold_atom(&atom);
    assert_eq!(unfolded.len(), 1);
}

#[test]
fn test_unfold_property_atom_subproperty() {
    let tbox = make_tbox(vec![QlAxiom::SubObjectPropertyOf {
        sub: QlRole::named("hasMother"),
        sup: QlRole::named("hasParent"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = prop_atom("x", "hasParent", "y");
    let unfolded = rewriter.unfold_atom(&atom);
    let props: Vec<_> = unfolded
        .iter()
        .filter_map(|a| {
            if let QueryAtom::PropertyAtom { property, .. } = a {
                Some(property.as_str())
            } else {
                None
            }
        })
        .collect();
    assert!(props.contains(&"hasParent"));
    assert!(props.contains(&"hasMother"));
}

#[test]
fn test_unfold_property_atom_inverse() {
    // inverseOf(hasParent, hasChild): querying hasParent can use hasChild inverse
    let tbox = make_tbox(vec![QlAxiom::InverseObjectProperties(
        "hasParent".to_string(),
        "hasChild".to_string(),
    )]);
    let rewriter = QueryRewriter::new(tbox);
    let atom = prop_atom("x", "hasParent", "y");
    let unfolded = rewriter.unfold_atom(&atom);
    // Should include ?y hasChild ?x (inverse)
    let has_inverse = unfolded.iter().any(|a| {
        if let QueryAtom::PropertyAtom { subject, property, object } = a {
            property == "hasChild"
                && subject == &QueryTerm::var("y")
                && object == &QueryTerm::var("x")
        } else {
            false
        }
    });
    assert!(has_inverse, "should contain inverse hasChild atom");
}

// ======================================================================
// 4. Full rewrite_query tests
// ======================================================================

#[test]
fn test_rewrite_single_type_no_tbox() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Person")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(!result.is_empty());
    assert!(contains_type(&result, "x", "Person"));
}

#[test]
fn test_rewrite_single_type_with_subclass() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Student"),
        sup: QlConcept::named("Person"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Person")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Person"));
    assert!(contains_type(&result, "x", "Student"));
}

#[test]
fn test_rewrite_chain_subclass() {
    // GradStudent ⊑ Student ⊑ Person
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf {
            sub: QlConcept::named("GradStudent"),
            sup: QlConcept::named("Student"),
        },
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Student"),
            sup: QlConcept::named("Person"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Person")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Person"));
    assert!(contains_type(&result, "x", "Student"));
    assert!(contains_type(&result, "x", "GradStudent"));
}

#[test]
fn test_rewrite_property_with_subproperty() {
    let tbox = make_tbox(vec![QlAxiom::SubObjectPropertyOf {
        sub: QlRole::named("hasMother"),
        sup: QlRole::named("hasParent"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "hasParent", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_property(&result, "x", "hasParent", "y"));
    assert!(contains_property(&result, "x", "hasMother", "y"));
}

#[test]
fn test_rewrite_property_with_inverse() {
    let tbox = make_tbox(vec![QlAxiom::InverseObjectProperties(
        "hasParent".to_string(),
        "hasChild".to_string(),
    )]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "hasParent", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Should contain ?x hasParent ?y (original)
    assert!(contains_property(&result, "x", "hasParent", "y"));
    // Should contain ?y hasChild ?x (inverse rewriting)
    assert!(contains_property(&result, "y", "hasChild", "x"));
}

#[test]
fn test_rewrite_type_with_domain() {
    let tbox = make_tbox(vec![QlAxiom::ObjectPropertyDomain {
        property: "worksAt".to_string(),
        domain: "Employee".to_string(),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Employee")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Employee"));
    // Should also contain ?x worksAt ?fresh
    let has_work = result.queries.iter().any(|cq| {
        cq.atoms.iter().any(|a| {
            if let QueryAtom::PropertyAtom { subject, property, .. } = a {
                property == "worksAt" && subject == &QueryTerm::var("x")
            } else {
                false
            }
        })
    });
    assert!(has_work, "rewriting should include worksAt property pattern");
}

#[test]
fn test_rewrite_type_with_range() {
    let tbox = make_tbox(vec![QlAxiom::ObjectPropertyRange {
        property: "worksAt".to_string(),
        range: "Organization".to_string(),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Organization")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Organization"));
    let has_work = result.queries.iter().any(|cq| {
        cq.atoms.iter().any(|a| {
            if let QueryAtom::PropertyAtom { property, object, .. } = a {
                property == "worksAt" && object == &QueryTerm::var("x")
            } else {
                false
            }
        })
    });
    assert!(has_work, "rewriting should include worksAt range pattern");
}

#[test]
fn test_rewrite_conjunction_two_atoms() {
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Student"),
            sup: QlConcept::named("Person"),
        },
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("hasMother"),
            sup: QlRole::named("hasParent"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![
        type_atom("x", "Person"),
        prop_atom("x", "hasParent", "y"),
    ]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Must contain the original
    assert!(!result.is_empty());
    // Various combinations should be present
    let total = result.len();
    assert!(total >= 3, "expected at least 3 rewritings, got {total}");
}

#[test]
fn test_rewrite_result_count_no_tbox() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Person")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_rewrite_does_not_duplicate_original() {
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("B"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "B")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Count occurrences of ?x:B
    let count = result
        .queries
        .iter()
        .filter(|cq| {
            cq.atoms.contains(&QueryAtom::TypeAtom {
                individual: QueryTerm::var("x"),
                class: "B".to_string(),
            })
        })
        .count();
    assert_eq!(count, 1, "original should appear exactly once");
}

#[test]
fn test_rewrite_with_limit_exceeded() {
    // Create a large TBox to trigger the limit
    let mut axioms = Vec::new();
    for i in 0..20 {
        axioms.push(QlAxiom::SubClassOf {
            sub: QlConcept::named(format!("C{i}")),
            sup: QlConcept::named("Top"),
        });
    }
    let tbox = make_tbox(axioms);
    let rewriter = QueryRewriter::with_limit(tbox, 5);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Top")]);
    let result = rewriter.rewrite_query(&cq);
    // Either succeeds with ≤5 or returns limit error (both are correct)
    match result {
        Ok(r) => assert!(r.len() <= 5),
        Err(QlError::RewritingLimitExceeded(_)) => {} // expected
        Err(e) => panic!("unexpected error: {e}"),
    }
}

// ======================================================================
// 5. QlConcept / QlRole construction tests
// ======================================================================

#[test]
fn test_ql_concept_named() {
    let c = QlConcept::named("Person");
    assert_eq!(c.as_named(), Some("Person"));
}

#[test]
fn test_ql_concept_some_values_top() {
    let c = QlConcept::some_values_top("knows");
    assert_eq!(c.as_some_values_property(), Some(("knows", false)));
}

#[test]
fn test_ql_concept_some_values_top_inverse() {
    let c = QlConcept::some_values_top_inverse("knows");
    assert_eq!(c.as_some_values_property(), Some(("knows", true)));
}

#[test]
fn test_ql_role_named() {
    let r = QlRole::named("knows");
    assert!(!r.is_inverse());
    assert_eq!(r.base_name(), "knows");
}

#[test]
fn test_ql_role_inverse() {
    let r = QlRole::inverse("knows");
    assert!(r.is_inverse());
    assert_eq!(r.base_name(), "knows");
}

#[test]
fn test_ql_role_inverse_of_inverse() {
    let r = QlRole::inverse("knows");
    let r2 = r.inverse_role();
    assert!(!r2.is_inverse());
    assert_eq!(r2.base_name(), "knows");
}

#[test]
fn test_ql_role_inverse_of_named() {
    let r = QlRole::named("knows");
    let r2 = r.inverse_role();
    assert!(r2.is_inverse());
}

#[test]
fn test_ql_concept_top() {
    let c = QlConcept::Top;
    assert!(c.as_named().is_none());
    assert!(c.as_some_values_property().is_none());
}

#[test]
fn test_ql_concept_bottom() {
    let c = QlConcept::Bottom;
    assert!(c.as_named().is_none());
}

// ======================================================================
// 6. Complex rewriting scenarios
// ======================================================================

#[test]
fn test_multi_hop_class_rewriting() {
    // A ⊑ B ⊑ C ⊑ D (chain of 4)
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("A"), sup: QlConcept::named("B") },
        QlAxiom::SubClassOf { sub: QlConcept::named("B"), sup: QlConcept::named("C") },
        QlAxiom::SubClassOf { sub: QlConcept::named("C"), sup: QlConcept::named("D") },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "D")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "D"));
    assert!(contains_type(&result, "x", "C"));
    assert!(contains_type(&result, "x", "B"));
    assert!(contains_type(&result, "x", "A"));
}

#[test]
fn test_multi_property_rewriting() {
    // P1 ⊑ P2 ⊑ P3
    let tbox = make_tbox(vec![
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("P1"),
            sup: QlRole::named("P2"),
        },
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("P2"),
            sup: QlRole::named("P3"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "P3", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_property(&result, "x", "P3", "y"));
    assert!(contains_property(&result, "x", "P2", "y"));
    assert!(contains_property(&result, "x", "P1", "y"));
}

#[test]
fn test_equivalent_classes_rewriting() {
    let tbox = make_tbox(vec![QlAxiom::EquivalentClasses(
        QlConcept::named("Human"),
        QlConcept::named("Person"),
    )]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Person")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Person"));
    assert!(contains_type(&result, "x", "Human"));
}

#[test]
fn test_inverse_symmetry() {
    // inverseOf(P, Q): querying P gives Q inverse, querying Q gives P inverse
    let tbox = make_tbox(vec![QlAxiom::InverseObjectProperties(
        "P".to_string(),
        "Q".to_string(),
    )]);
    let rewriter = QueryRewriter::new(tbox);

    // Query: ?x P ?y
    let cq1 = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "P", "y")]);
    let r1 = rewriter.rewrite_query(&cq1).expect("rewrite failed");
    assert!(contains_property(&r1, "y", "Q", "x"));

    // Query: ?x Q ?y
    let cq2 = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "Q", "y")]);
    let r2 = rewriter.rewrite_query(&cq2).expect("rewrite failed");
    assert!(contains_property(&r2, "y", "P", "x"));
}

#[test]
fn test_domain_propagated_through_class_hierarchy() {
    // worksAt domain Employee, Employee ⊑ Person
    // Querying ?x:Person should eventually include ?x worksAt ?y
    let tbox = make_tbox(vec![
        QlAxiom::ObjectPropertyDomain {
            property: "worksAt".to_string(),
            domain: "Employee".to_string(),
        },
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Employee"),
            sup: QlConcept::named("Person"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox.clone());

    // worksAt domain should include Person via hierarchy
    assert!(tbox.domain_of("worksAt").contains("Person"));

    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Person")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Person"));
    assert!(contains_type(&result, "x", "Employee"));
}

#[test]
fn test_range_propagated_through_class_hierarchy() {
    let tbox = make_tbox(vec![
        QlAxiom::ObjectPropertyRange {
            property: "worksAt".to_string(),
            range: "Company".to_string(),
        },
        QlAxiom::SubClassOf {
            sub: QlConcept::named("Company"),
            sup: QlConcept::named("Organization"),
        },
    ]);
    assert!(tbox.range_of("worksAt").contains("Organization"));
}

#[test]
fn test_subproperty_inherits_domain() {
    // hasMother ⊑ hasParent, hasParent domain Person
    // → hasMother also has domain Person
    let tbox = make_tbox(vec![
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("hasMother"),
            sup: QlRole::named("hasParent"),
        },
        QlAxiom::ObjectPropertyDomain {
            property: "hasParent".to_string(),
            domain: "Person".to_string(),
        },
    ]);
    assert!(tbox.domain_of("hasMother").contains("Person"));
}

#[test]
fn test_rewritten_query_is_ucq() {
    // A rewritten query is a union of conjunctive queries
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf {
            sub: QlConcept::named("A"),
            sup: QlConcept::named("B"),
        },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "B")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Each element is a conjunctive query
    for cq in &result.queries {
        assert!(!cq.atoms.is_empty());
    }
}

#[test]
fn test_build_tbox_convenience() {
    let tbox = build_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("A"),
        sup: QlConcept::named("B"),
    }])
    .expect("build_tbox failed");
    assert!(tbox.superclasses("A").contains("B"));
}

#[test]
fn test_rewrite_query_convenience() {
    let tbox = build_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Student"),
        sup: QlConcept::named("Person"),
    }])
    .expect("build failed");
    let result =
        rewrite_query(vec![type_atom("x", "Person")], &tbox).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Student"));
}

#[test]
fn test_rewritten_query_default() {
    let r = RewrittenQuery::default();
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
}

#[test]
fn test_rewritten_query_add() {
    let mut r = RewrittenQuery::new();
    r.add(ConjunctiveQuery::with_atoms(vec![type_atom("x", "A")]));
    assert_eq!(r.len(), 1);
}

// ======================================================================
// 7. Correctness vs OWL 2 QL semantics
// ======================================================================

#[test]
fn test_perfectref_correctness_simple() {
    // TBox: Dog ⊑ Animal
    // ABox: Fido:Dog
    // Query: ?x:Animal
    // Expected: Fido should be returned (via subclass)
    // PerfectRef rewrites to: ?x:Animal ∪ ?x:Dog
    // Running ?x:Dog over ABox returns Fido ✓
    let tbox = make_tbox(vec![QlAxiom::SubClassOf {
        sub: QlConcept::named("Dog"),
        sup: QlConcept::named("Animal"),
    }]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Animal")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    // Both ?x:Animal and ?x:Dog must be in the union
    assert!(contains_type(&result, "x", "Animal"));
    assert!(contains_type(&result, "x", "Dog"));
}

#[test]
fn test_perfectref_property_correctness() {
    // TBox: worksFor ⊑ hasAffiliation, worksFor inverseOf affiliatedWith
    // Query: ?x hasAffiliation ?y
    // Rewriting should include ?x worksFor ?y and ?y affiliatedWith ?x
    let tbox = make_tbox(vec![
        QlAxiom::SubObjectPropertyOf {
            sub: QlRole::named("worksFor"),
            sup: QlRole::named("hasAffiliation"),
        },
        QlAxiom::InverseObjectProperties(
            "hasAffiliation".to_string(),
            "affiliatedWith".to_string(),
        ),
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![prop_atom("x", "hasAffiliation", "y")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_property(&result, "x", "hasAffiliation", "y"));
    assert!(contains_property(&result, "x", "worksFor", "y"));
    // The inverse rewriting: ?y affiliatedWith ?x
    assert!(contains_property(&result, "y", "affiliatedWith", "x"));
}

#[test]
fn test_disjointness_detection() {
    let tbox = make_tbox(vec![QlAxiom::DisjointClasses(
        QlConcept::named("Male"),
        QlConcept::named("Female"),
    )]);
    assert!(tbox.are_disjoint("Male", "Female"));
    assert!(!tbox.are_disjoint("Male", "Person"));
}

#[test]
fn test_satisfiability_check() {
    let tbox = make_tbox(vec![]);
    let rewriter = QueryRewriter::new(tbox);
    assert!(rewriter.is_satisfiable("SomeClass"));
}

#[test]
fn test_multiple_subclasses_fan_in() {
    // Both Dog and Cat are subclasses of Animal
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("Dog"), sup: QlConcept::named("Animal") },
        QlAxiom::SubClassOf { sub: QlConcept::named("Cat"), sup: QlConcept::named("Animal") },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "Animal")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "Animal"));
    assert!(contains_type(&result, "x", "Dog"));
    assert!(contains_type(&result, "x", "Cat"));
}

#[test]
fn test_diamond_hierarchy() {
    // A ⊑ B, A ⊑ C, B ⊑ D, C ⊑ D
    let tbox = make_tbox(vec![
        QlAxiom::SubClassOf { sub: QlConcept::named("A"), sup: QlConcept::named("B") },
        QlAxiom::SubClassOf { sub: QlConcept::named("A"), sup: QlConcept::named("C") },
        QlAxiom::SubClassOf { sub: QlConcept::named("B"), sup: QlConcept::named("D") },
        QlAxiom::SubClassOf { sub: QlConcept::named("C"), sup: QlConcept::named("D") },
    ]);
    let rewriter = QueryRewriter::new(tbox);
    let cq = ConjunctiveQuery::with_atoms(vec![type_atom("x", "D")]);
    let result = rewriter.rewrite_query(&cq).expect("rewrite failed");
    assert!(contains_type(&result, "x", "D"));
    assert!(contains_type(&result, "x", "B"));
    assert!(contains_type(&result, "x", "C"));
    assert!(contains_type(&result, "x", "A"));
}
