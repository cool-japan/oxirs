//! SPARQL-star (RDF-star) completeness tests.

#![cfg(test)]

use crate::algebra::{Literal, Term, TriplePattern};
use crate::rdf_star::rdf_star_binding::sparql_star_builtins::*;
use crate::rdf_star::{
    bind_pattern, instantiate_quoted_triple, Annotation, QuotedTriple, RdfStarStore, StarBinding,
    StarObject, StarOperator, StarPattern, StarPredicate, StarSubject,
};
use oxirs_core::model::{NamedNode, Variable};

// ── Helpers ───────────────────────────────────────────────────────────

fn iri(s: &str) -> NamedNode {
    NamedNode::new(s).unwrap()
}

fn var(name: &str) -> Variable {
    Variable::new(name).unwrap()
}

fn qt(s: &str, p: &str, o: &str) -> QuotedTriple {
    QuotedTriple::new(
        StarSubject::NamedNode(iri(s)),
        StarPredicate::NamedNode(iri(p)),
        StarObject::NamedNode(iri(o)),
    )
}

fn qt_lit(s: &str, p: &str, o: &str) -> QuotedTriple {
    QuotedTriple::new(
        StarSubject::NamedNode(iri(s)),
        StarPredicate::NamedNode(iri(p)),
        StarObject::Literal(Literal::new(o.to_string(), None, None)),
    )
}

fn ann(p: &str, o: &str) -> Annotation {
    Annotation::new(iri(p), StarObject::NamedNode(iri(o)))
}

// ── QuotedTriple construction ─────────────────────────────────────────

#[test]
fn test_quoted_triple_new() {
    let qt = qt("http://s", "http://p", "http://o");
    assert_eq!(qt.to_string(), "<< <http://s> <http://p> <http://o> >>");
}

#[test]
fn test_quoted_triple_from_iris() {
    let qt = QuotedTriple::from_iris("http://s", "http://p", "http://o").unwrap();
    assert!(matches!(&qt.subject, StarSubject::NamedNode(_)));
    assert!(matches!(&qt.predicate, StarPredicate::NamedNode(_)));
    assert!(matches!(&qt.object, StarObject::NamedNode(_)));
}

#[test]
fn test_quoted_triple_nesting_depth_simple() {
    let qt = qt("http://s", "http://p", "http://o");
    assert_eq!(qt.nesting_depth(), 1);
}

#[test]
fn test_quoted_triple_nesting_depth_nested() {
    let inner = qt("http://s", "http://p", "http://o");
    let outer = QuotedTriple::new(
        StarSubject::Quoted(Box::new(inner)),
        StarPredicate::NamedNode(iri("http://certainty")),
        StarObject::Literal(Literal::new("0.9".into(), None, None)),
    );
    assert_eq!(outer.nesting_depth(), 2);
}

#[test]
fn test_quoted_triple_triple_nesting() {
    let inner = qt("http://s", "http://p", "http://o");
    let mid = QuotedTriple::new(
        StarSubject::Quoted(Box::new(inner)),
        StarPredicate::NamedNode(iri("http://cert")),
        StarObject::NamedNode(iri("http://v")),
    );
    let outer = QuotedTriple::new(
        StarSubject::Quoted(Box::new(mid)),
        StarPredicate::NamedNode(iri("http://source")),
        StarObject::NamedNode(iri("http://paper")),
    );
    assert_eq!(outer.nesting_depth(), 3);
}

// ── is_pattern / variables ────────────────────────────────────────────

#[test]
fn test_quoted_triple_no_variables_is_not_pattern() {
    let qt = qt("http://s", "http://p", "http://o");
    assert!(!qt.is_pattern());
    assert!(qt.variables().is_empty());
}

#[test]
fn test_quoted_triple_with_variable_subject() {
    let qt_var = QuotedTriple::new(
        StarSubject::Variable(var("s")),
        StarPredicate::NamedNode(iri("http://p")),
        StarObject::NamedNode(iri("http://o")),
    );
    assert!(qt_var.is_pattern());
    let vars = qt_var.variables();
    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].as_str(), "s");
}

#[test]
fn test_quoted_triple_with_variable_predicate_and_object() {
    let qt_vars = QuotedTriple::new(
        StarSubject::NamedNode(iri("http://s")),
        StarPredicate::Variable(var("p")),
        StarObject::Variable(var("o")),
    );
    let vars = qt_vars.variables();
    assert_eq!(vars.len(), 2);
}

// ── Triple pattern conversion ─────────────────────────────────────────

#[test]
fn test_to_triple_pattern() {
    let quoted = qt("http://s", "http://p", "http://o");
    let tp = quoted.to_triple_pattern();
    assert!(matches!(tp.subject, Term::Iri(_)));
    assert!(matches!(tp.predicate, Term::Iri(_)));
    assert!(matches!(tp.object, Term::Iri(_)));
}

#[test]
fn test_from_triple_pattern() {
    let tp = TriplePattern::new(
        Term::Iri(iri("http://s")),
        Term::Iri(iri("http://p")),
        Term::Iri(iri("http://o")),
    );
    let qt = QuotedTriple::from_triple_pattern(&tp).unwrap();
    assert!(matches!(qt.subject, StarSubject::NamedNode(_)));
}

#[test]
fn test_round_trip_triple_pattern() {
    let original = qt("http://s", "http://p", "http://o");
    let tp = original.to_triple_pattern();
    let back = QuotedTriple::from_triple_pattern(&tp).unwrap();
    assert_eq!(original, back);
}

// ── StarPredicate / StarSubject / StarObject ──────────────────────────

#[test]
fn test_star_subject_from_term_iri() {
    let term = Term::Iri(iri("http://s"));
    let subject = StarSubject::from_term(&term).unwrap();
    assert!(matches!(subject, StarSubject::NamedNode(_)));
}

#[test]
fn test_star_subject_from_term_blank_node() {
    let term = Term::BlankNode("b0".to_string());
    let subject = StarSubject::from_term(&term).unwrap();
    assert!(matches!(subject, StarSubject::BlankNode(_)));
}

#[test]
fn test_star_subject_from_literal_fails() {
    let term = Term::Literal(Literal::new("x".into(), None, None));
    assert!(StarSubject::from_term(&term).is_err());
}

#[test]
fn test_star_object_from_literal() {
    let term = Term::Literal(Literal::new("hello".into(), None, None));
    let obj = StarObject::from_term(&term).unwrap();
    assert!(matches!(obj, StarObject::Literal(_)));
}

#[test]
fn test_star_predicate_from_iri() {
    let term = Term::Iri(iri("http://p"));
    let pred = StarPredicate::from_term(&term).unwrap();
    assert!(matches!(pred, StarPredicate::NamedNode(_)));
}

#[test]
fn test_star_predicate_from_variable() {
    let term = Term::Variable(var("p"));
    let pred = StarPredicate::from_term(&term).unwrap();
    assert!(matches!(pred, StarPredicate::Variable(_)));
}

// ── RdfStarStore operations ───────────────────────────────────────────

#[test]
fn test_store_assert_and_contains() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    assert!(!store.contains(&triple));
    store.assert_triple(triple.clone());
    assert!(store.contains(&triple));
    assert_eq!(store.len(), 1);
}

#[test]
fn test_store_retract() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.assert_triple(triple.clone());
    assert!(store.retract_triple(&triple));
    assert!(!store.contains(&triple));
    assert_eq!(store.len(), 0);
}

#[test]
fn test_store_add_annotation() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    let pred = iri("http://certainty");
    store.add_annotation(
        &triple,
        &pred,
        StarObject::Literal(Literal::new("0.9".into(), None, None)),
    );
    let entry = store.annotations(&triple).unwrap();
    assert!(entry.annotation(&pred).is_some());
}

#[test]
fn test_store_remove_annotation() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    let pred = iri("http://certainty");
    store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://high")));
    assert!(store.remove_annotation(&triple, &pred));
    let entry = store.annotations(&triple).unwrap();
    assert!(entry.annotation(&pred).is_none());
}

#[test]
fn test_store_multiple_triples() {
    let mut store = RdfStarStore::new();
    store.assert_triple(qt("http://s1", "http://p", "http://o1"));
    store.assert_triple(qt("http://s2", "http://p", "http://o2"));
    assert_eq!(store.len(), 2);
}

// ── StarOperator ──────────────────────────────────────────────────────

#[test]
fn test_assert_quoted_operator() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.apply_operator(StarOperator::AssertQuoted {
        triple: triple.clone(),
    });
    assert!(store.contains(&triple));
}

#[test]
fn test_retract_quoted_operator() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.assert_triple(triple.clone());
    store.apply_operator(StarOperator::RetractQuoted {
        triple: triple.clone(),
    });
    assert!(!store.contains(&triple));
}

#[test]
fn test_add_annotation_operator() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.apply_operator(StarOperator::AddAnnotation {
        triple: triple.clone(),
        annotations: vec![ann("http://cert", "http://high")],
    });
    assert!(store.annotations(&triple).is_some());
}

#[test]
fn test_remove_annotation_operator() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    let pred = iri("http://cert");
    store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://high")));
    store.apply_operator(StarOperator::RemoveAnnotation {
        triple: triple.clone(),
        predicate: pred.clone(),
    });
    let entry = store.annotations(&triple).unwrap();
    assert!(entry.annotation(&pred).is_none());
}

// ── FindAnnotations / pattern matching ───────────────────────────────

#[test]
fn test_find_annotations_exact_match() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    let cert = iri("http://certainty");
    store.add_annotation(&triple, &cert, StarObject::NamedNode(iri("http://high")));

    let pattern = StarPattern::new(
        triple.clone(),
        StarPredicate::NamedNode(cert),
        StarObject::NamedNode(iri("http://high")),
    );
    let results = store.find_annotations(&pattern);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_find_annotations_variable_predicate() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.add_annotation(
        &triple,
        &iri("http://cert"),
        StarObject::NamedNode(iri("http://high")),
    );

    let pattern = StarPattern::new(
        triple.clone(),
        StarPredicate::Variable(var("pred")),
        StarObject::Variable(var("obj")),
    );
    let results = store.find_annotations(&pattern);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_find_annotations_no_match() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.add_annotation(
        &triple,
        &iri("http://cert"),
        StarObject::NamedNode(iri("http://high")),
    );

    let other_triple = qt("http://other", "http://p", "http://o");
    let pattern = StarPattern::new(
        other_triple,
        StarPredicate::Variable(var("pred")),
        StarObject::Variable(var("obj")),
    );
    let results = store.find_annotations(&pattern);
    assert!(results.is_empty());
}

// ── Pattern matching / binding ─────────────────────────────────────

#[test]
fn test_bind_pattern_all_variables() {
    let triple = qt("http://s", "http://p", "http://o");
    let pattern = QuotedTriple::new(
        StarSubject::Variable(var("s")),
        StarPredicate::Variable(var("p")),
        StarObject::Variable(var("o")),
    );
    let mut binding = StarBinding::new();
    assert!(bind_pattern(&triple, &pattern, &mut binding));
    assert!(binding.contains_key("s"));
    assert!(binding.contains_key("p"));
    assert!(binding.contains_key("o"));
}

#[test]
fn test_bind_pattern_partial_variables() {
    let triple = qt("http://s", "http://p", "http://o");
    let pattern = QuotedTriple::new(
        StarSubject::NamedNode(iri("http://s")),
        StarPredicate::Variable(var("p")),
        StarObject::Variable(var("o")),
    );
    let mut binding = StarBinding::new();
    assert!(bind_pattern(&triple, &pattern, &mut binding));
    assert!(binding.contains_key("p"));
    assert!(binding.contains_key("o"));
}

#[test]
fn test_bind_pattern_mismatch() {
    let triple = qt("http://s", "http://p", "http://o");
    let pattern = QuotedTriple::new(
        StarSubject::NamedNode(iri("http://DIFFERENT")),
        StarPredicate::Variable(var("p")),
        StarObject::Variable(var("o")),
    );
    let mut binding = StarBinding::new();
    assert!(!bind_pattern(&triple, &pattern, &mut binding));
}

// ── CONSTRUCT (instantiation) ─────────────────────────────────────────

#[test]
fn test_instantiate_quoted_triple() {
    let template = QuotedTriple::new(
        StarSubject::Variable(var("s")),
        StarPredicate::NamedNode(iri("http://p")),
        StarObject::Variable(var("o")),
    );
    let mut binding = StarBinding::new();
    binding.insert("s".to_string(), StarObject::NamedNode(iri("http://alice")));
    binding.insert(
        "o".to_string(),
        StarObject::Literal(Literal::new("42".into(), None, None)),
    );

    let result = instantiate_quoted_triple(&template, &binding).unwrap();
    assert!(matches!(result.subject, StarSubject::NamedNode(n) if n.to_string().contains("alice")));
    assert!(matches!(result.object, StarObject::Literal(_)));
}

#[test]
fn test_instantiate_unbound_variable_fails() {
    let template = QuotedTriple::new(
        StarSubject::Variable(var("missing")),
        StarPredicate::NamedNode(iri("http://p")),
        StarObject::NamedNode(iri("http://o")),
    );
    let binding = StarBinding::new();
    assert!(instantiate_quoted_triple(&template, &binding).is_err());
}

// ── SPARQL-star builtins ─────────────────────────────────────────────

#[test]
fn test_is_triple_function() {
    let obj = StarObject::Quoted(Box::new(qt("http://s", "http://p", "http://o")));
    assert!(is_triple(&obj));
    let not_triple = StarObject::NamedNode(iri("http://x"));
    assert!(!is_triple(&not_triple));
}

#[test]
fn test_subject_of() {
    let triple = qt("http://alice", "http://p", "http://o");
    let s = subject_of(&triple);
    assert!(matches!(s, StarSubject::NamedNode(n) if n.to_string().contains("alice")));
}

#[test]
fn test_predicate_of() {
    let triple = qt("http://s", "http://predicate", "http://o");
    let p = predicate_of(&triple);
    assert!(matches!(p, StarPredicate::NamedNode(n) if n.to_string().contains("predicate")));
}

#[test]
fn test_object_of() {
    let triple = qt("http://s", "http://p", "http://target");
    let o = object_of(&triple);
    assert!(matches!(o, StarObject::NamedNode(n) if n.to_string().contains("target")));
}

#[test]
fn test_triple_fn_builtin() {
    let s = StarSubject::NamedNode(iri("http://s"));
    let p = StarPredicate::NamedNode(iri("http://p"));
    let o = StarObject::NamedNode(iri("http://o"));
    let qt = triple_fn(s, p, o);
    assert_eq!(qt.nesting_depth(), 1);
}

// ── Mixed standard + star patterns ───────────────────────────────────

#[test]
fn test_mixed_star_and_standard_patterns() {
    let mut store = RdfStarStore::new();
    // Add a plain triple annotation
    let t1 = qt("http://alice", "http://knows", "http://bob");
    store.add_annotation(
        &t1,
        &iri("http://since"),
        StarObject::Literal(Literal::new("2020".into(), None, None)),
    );
    // Add a second triple with different annotation
    let t2 = qt("http://bob", "http://knows", "http://carol");
    store.add_annotation(
        &t2,
        &iri("http://since"),
        StarObject::Literal(Literal::new("2021".into(), None, None)),
    );

    // Find all "knows" triples with "since" annotation
    let pattern_triple = QuotedTriple::new(
        StarSubject::Variable(var("s")),
        StarPredicate::NamedNode(iri("http://knows")),
        StarObject::Variable(var("o")),
    );
    let pattern = StarPattern::new(
        pattern_triple,
        StarPredicate::NamedNode(iri("http://since")),
        StarObject::Variable(var("when")),
    );
    let results = store.find_annotations(&pattern);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_star_pattern_variables() {
    let pattern = StarPattern::new(
        QuotedTriple::new(
            StarSubject::Variable(var("s")),
            StarPredicate::Variable(var("p")),
            StarObject::Variable(var("o")),
        ),
        StarPredicate::Variable(var("ap")),
        StarObject::Variable(var("ao")),
    );
    let vars = pattern.variables();
    assert_eq!(vars.len(), 5);
}

// ── Display / formatting ──────────────────────────────────────────────

#[test]
fn test_star_operator_display() {
    let triple = qt("http://s", "http://p", "http://o");
    let op = StarOperator::AssertQuoted { triple };
    assert!(op.to_string().contains("AssertQuoted"));
}

#[test]
fn test_quoted_triple_display_nested() {
    let inner = qt("http://s", "http://p", "http://o");
    let outer = QuotedTriple::new(
        StarSubject::Quoted(Box::new(inner)),
        StarPredicate::NamedNode(iri("http://cert")),
        StarObject::Literal(Literal::new("high".into(), None, None)),
    );
    let s = outer.to_string();
    assert!(s.contains("<<"));
    // Nested << inside outer <<
    assert!(s.matches("<<").count() >= 2);
}

// ── Annotation store operations ───────────────────────────────────────

#[test]
fn test_multiple_annotations_on_same_triple() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.add_annotation(
        &triple,
        &iri("http://cert"),
        StarObject::NamedNode(iri("http://high")),
    );
    store.add_annotation(
        &triple,
        &iri("http://source"),
        StarObject::NamedNode(iri("http://paper1")),
    );

    let entry = store.annotations(&triple).unwrap();
    assert_eq!(entry.annotations.len(), 2);
}

#[test]
fn test_annotation_overwrite() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    let pred = iri("http://cert");
    store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://low")));
    store.add_annotation(&triple, &pred, StarObject::NamedNode(iri("http://high")));

    let entry = store.annotations(&triple).unwrap();
    // Overwritten — still just 1 annotation for this predicate
    assert_eq!(entry.annotations.len(), 1);
    if let Some(StarObject::NamedNode(n)) = entry.annotation(&pred) {
        assert!(n.to_string().contains("high"));
    } else {
        panic!("expected NamedNode annotation");
    }
}

#[test]
fn test_store_iter() {
    let mut store = RdfStarStore::new();
    store.assert_triple(qt("http://s1", "http://p", "http://o1"));
    store.assert_triple(qt("http://s2", "http://p", "http://o2"));
    assert_eq!(store.iter().count(), 2);
}

#[test]
fn test_find_annotations_with_literal_object_value() {
    let mut store = RdfStarStore::new();
    let triple = qt_lit("http://s", "http://p", "42");
    store.add_annotation(
        &triple,
        &iri("http://source"),
        StarObject::NamedNode(iri("http://db")),
    );
    let pattern = StarPattern::new(
        triple.clone(),
        StarPredicate::Variable(var("pred")),
        StarObject::Variable(var("obj")),
    );
    let results = store.find_annotations(&pattern);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_apply_operator_find_annotations() {
    let mut store = RdfStarStore::new();
    let triple = qt("http://s", "http://p", "http://o");
    store.add_annotation(
        &triple,
        &iri("http://cert"),
        StarObject::NamedNode(iri("http://high")),
    );

    let pattern = StarPattern::new(
        triple.clone(),
        StarPredicate::NamedNode(iri("http://cert")),
        StarObject::Variable(var("v")),
    );
    let results = store.apply_operator(StarOperator::FindAnnotations { pattern });
    assert_eq!(results.len(), 1);
}
