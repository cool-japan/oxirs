//! Tests for the OWL 2 QL profile checker.
#![cfg(test)]

use super::checker::*;
use super::profile::*;

fn class(iri: &str) -> ClassExpr {
    ClassExpr::Named(iri.into())
}

#[test]
fn test_compliant_subclass() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: class("B"),
    };
    let report = Owl2QlProfileChecker::new().check(&[axiom]);
    assert!(report.is_ql_compliant());
}

#[test]
fn test_compliant_some_values_thing() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: ClassExpr::SomeValuesFrom {
            property: "P".into(),
            filler: Box::new(ClassExpr::Thing),
        },
        sup: class("B"),
    };
    let report = Owl2QlProfileChecker::new().check(&[axiom]);
    assert!(report.is_ql_compliant());
}

#[test]
fn test_violation_all_values_from() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: ClassExpr::AllValuesFrom {
            property: "P".into(),
            filler: Box::new(class("B")),
        },
    };
    let report = Owl2QlProfileChecker::new().check(&[axiom]);
    assert!(!report.is_ql_compliant());
}

#[test]
fn test_violation_has_value() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: ClassExpr::HasValue {
            property: "P".into(),
            individual: "i".into(),
        },
    };
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_violation_min_cardinality() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: ClassExpr::MinCardinality {
            n: 1,
            property: "P".into(),
            filler: None,
        },
    };
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_violation_max_cardinality() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: ClassExpr::MaxCardinality {
            n: 1,
            property: "P".into(),
            filler: None,
        },
    };
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_violation_one_of() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: ClassExpr::OneOf(vec!["a".into(), "b".into()]),
    };
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_violation_property_chain() {
    let axiom = OntologyAxiom::SubObjectPropertyChain {
        chain: vec!["P".into(), "Q".into()],
        sup: "R".into(),
    };
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_violation_transitive_property() {
    let axiom = OntologyAxiom::TransitiveObjectProperty("P".into());
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_violation_functional_property() {
    let axiom = OntologyAxiom::FunctionalObjectProperty("P".into());
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_violation_disjoint_union() {
    let axiom = OntologyAxiom::DisjointUnion {
        class: "A".into(),
        classes: vec![class("B"), class("C")],
    };
    assert!(!Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_compliant_inverse_properties() {
    let axiom = OntologyAxiom::InverseObjectProperties {
        p1: "P".into(),
        p2: "Q".into(),
    };
    assert!(Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_compliant_disjoint_properties() {
    let axiom = OntologyAxiom::DisjointObjectProperties(vec!["P".into(), "Q".into()]);
    assert!(Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_compliant_class_assertion() {
    let axiom = OntologyAxiom::ClassAssertion {
        class: class("A"),
        individual: "i".into(),
    };
    assert!(Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_report_violation_count() {
    let axioms = vec![
        OntologyAxiom::SubClassOf {
            sub: class("A"),
            sup: class("B"),
        },
        OntologyAxiom::FunctionalObjectProperty("P".into()),
        OntologyAxiom::TransitiveObjectProperty("Q".into()),
    ];
    let report = Owl2QlProfileChecker::new().check(&axioms);
    assert_eq!(report.violation_count(), 2);
    assert_eq!(report.axioms_checked, 3);
}

#[test]
fn test_empty_ontology() {
    let report = Owl2QlProfileChecker::new().check(&[]);
    assert_eq!(report.axioms_checked, 0);
    assert!(report.is_ql_compliant());
}

#[test]
fn test_complement_in_super_class() {
    // ComplementOf(C) where C is QL sub-class-expression -- compliant.
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: ClassExpr::ComplementOf(Box::new(class("B"))),
    };
    assert!(Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_intersection_of_atomic_super() {
    let axiom = OntologyAxiom::SubClassOf {
        sub: class("A"),
        sup: ClassExpr::IntersectionOf(vec![class("B"), class("C")]),
    };
    assert!(Owl2QlProfileChecker::new()
        .check(&[axiom])
        .is_ql_compliant());
}

#[test]
fn test_summary_string() {
    let report = ProfileReport {
        axioms_checked: 5,
        violations: vec![],
    };
    assert!(report.summary().contains("compliant"));
}
