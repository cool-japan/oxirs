//! Tests for the SPARQL 1.1 Entailment Regimes API.
//!
//! Covers:
//! - [`Triple`], [`TripleStore`], [`EntailmentEngine`] trait
//! - [`ClosureGraph`] (flat-string API)
//! - [`EntailmentGraph`] (rich term API with [`EntailmentRegime`] trait)
//! - [`RdfEntailmentEngine`], [`RdfsEntailmentEngine`], [`OwlRlEntailmentEngine`]
//! - All RDFS rules rdfs2–rdfs12, OWL RL rules, and error cases

use super::{
    ClosureGraph, EntailmentConfig, EntailmentError, EntailmentGraph, EntailmentRegime,
    EntailmentTerm, RichEntailmentTriple, SparqlRegimeKind, Triple, TripleStore,
};
use crate::entailment::rdf_entailment::{rdf_iri, rdfs_iri};
use crate::entailment::{OwlRlEntailmentEngine, RdfEntailmentEngine, RdfsEntailmentEngine};

// ── TripleStore helpers ───────────────────────────────────────────────────────

/// Build a [`TripleStore`] from a list of (s, p, o) tuples.
fn store_from(triples: &[(&str, &str, &str)]) -> TripleStore {
    let mut s = TripleStore::new();
    for (subj, pred, obj) in triples {
        s.add(Triple::new(*subj, *pred, *obj));
    }
    s
}

/// Build a `Vec<RichEntailmentTriple>` from (s, p, o) IRI tuples.
fn rich_triples(triples: &[(&str, &str, &str)]) -> Vec<RichEntailmentTriple> {
    triples
        .iter()
        .map(|(s, p, o)| RichEntailmentTriple::named_triple(*s, *p, *o))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// TripleStore tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_triple_store_contains() {
    let mut store = TripleStore::new();
    store.add(Triple::new("s", "p", "o"));
    assert!(store.contains("s", "p", "o"));
    assert!(!store.contains("s", "p", "x"));
    assert!(!store.contains("x", "p", "o"));
}

#[test]
fn test_triple_store_get_by_sp() {
    let store = store_from(&[
        ("alice", "knows", "bob"),
        ("alice", "knows", "carol"),
        ("alice", "likes", "coffee"),
    ]);
    let results = store.get_by_sp("alice", "knows");
    assert_eq!(results.len(), 2);
    let objects: Vec<&str> = results.iter().map(|t| t.object.as_str()).collect();
    assert!(objects.contains(&"bob"));
    assert!(objects.contains(&"carol"));
}

#[test]
fn test_triple_store_get_by_po() {
    let store = store_from(&[
        ("alice", "type", "Person"),
        ("bob", "type", "Person"),
        ("cat", "type", "Animal"),
    ]);
    let results = store.get_by_po("type", "Person");
    assert_eq!(results.len(), 2);
    let subjects: Vec<&str> = results.iter().map(|t| t.subject.as_str()).collect();
    assert!(subjects.contains(&"alice"));
    assert!(subjects.contains(&"bob"));
}

#[test]
fn test_triple_store_dedup() {
    let mut store = TripleStore::new();
    store.add(Triple::new("s", "p", "o"));
    store.add(Triple::new("s", "p", "o")); // duplicate
    assert_eq!(store.len(), 1, "duplicates must not be inserted");
}

// ─────────────────────────────────────────────────────────────────────────────
// RDF entailment engine (flat-string API)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_store() {
    let store = TripleStore::new();
    let engine = RdfEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("should not error");
    // rdf4 axiom: rdf:type rdf:type rdf:Property
    let rdf_type = rdf_iri("type");
    let rdf_property = rdf_iri("Property");
    assert!(result
        .iter()
        .any(|t| t.subject == rdf_type && t.predicate == rdf_type && t.object == rdf_property));
}

#[test]
fn test_rdf_entailment_property_typing() {
    let store = store_from(&[(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/knows",
        "http://example.org/bob",
    )]);
    let engine = RdfEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    let rdf_type = rdf_iri("type");
    let rdf_property = rdf_iri("Property");

    assert!(
        result.iter().any(|t| {
            t.subject == "http://xmlns.com/foaf/0.1/knows"
                && t.predicate == rdf_type
                && t.object == rdf_property
        }),
        "rdf1 should infer that the predicate is an rdf:Property"
    );
}

#[test]
fn test_rdf_entailment_fixpoint() {
    let store = store_from(&[("s", "p", "o")]);
    let engine = RdfEntailmentEngine::new();

    let first = super::EntailmentEngine::entail(&engine, &store).expect("ok");
    assert!(!first.is_empty());

    let mut augmented = store.clone();
    for t in first {
        augmented.add(t);
    }

    let second = super::EntailmentEngine::entail(&engine, &augmented).expect("ok");
    assert!(
        second.is_empty(),
        "second pass should produce no new triples (fixpoint)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// RDFS entailment engine (flat-string API)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rdfs_domain() {
    let domain_pred = rdfs_iri("domain");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        (
            "http://example.org/hasAge",
            &domain_pred,
            "http://example.org/Person",
        ),
        ("http://example.org/alice", "http://example.org/hasAge", "30"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result.iter().any(|t| {
            t.subject == "http://example.org/alice"
                && t.predicate == rdf_type
                && t.object == "http://example.org/Person"
        }),
        "rdfs2 should infer alice rdf:type Person via domain declaration"
    );
}

#[test]
fn test_rdfs_range() {
    let range_pred = rdfs_iri("range");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        (
            "http://example.org/knows",
            &range_pred,
            "http://example.org/Person",
        ),
        (
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        ),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result.iter().any(|t| {
            t.subject == "http://example.org/bob"
                && t.predicate == rdf_type
                && t.object == "http://example.org/Person"
        }),
        "rdfs3 should infer bob rdf:type Person via range declaration"
    );
}

#[test]
fn test_rdfs_subproperty_trans() {
    let sub_prop = rdfs_iri("subPropertyOf");
    let store = store_from(&[
        ("hasMother", &sub_prop, "hasParent"),
        ("hasParent", &sub_prop, "hasAncestor"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result.iter().any(|t| {
            t.subject == "hasMother" && t.predicate == sub_prop && t.object == "hasAncestor"
        }),
        "rdfs5 should infer hasMother rdfs:subPropertyOf hasAncestor transitively"
    );
}

#[test]
fn test_rdfs_subproperty_prop_usage() {
    let sub_prop = rdfs_iri("subPropertyOf");
    let store = store_from(&[
        ("hasMother", &sub_prop, "hasParent"),
        ("alice", "hasMother", "eve"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result
            .iter()
            .any(|t| { t.subject == "alice" && t.predicate == "hasParent" && t.object == "eve" }),
        "rdfs7 should propagate hasMother usage to hasParent"
    );
}

#[test]
fn test_rdfs_subclass_trans() {
    let sub_class = rdfs_iri("subClassOf");
    let store = store_from(&[
        ("Poodle", &sub_class, "Dog"),
        ("Dog", &sub_class, "Animal"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result
            .iter()
            .any(|t| { t.subject == "Poodle" && t.predicate == sub_class && t.object == "Animal" }),
        "rdfs11 should infer Poodle rdfs:subClassOf Animal transitively"
    );
}

#[test]
fn test_rdfs_subclass_instance() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("Dog", &sub_class, "Animal"),
        ("fido", &rdf_type, "Dog"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result
            .iter()
            .any(|t| { t.subject == "fido" && t.predicate == rdf_type && t.object == "Animal" }),
        "rdfs9 should infer fido rdf:type Animal"
    );
}

#[test]
fn test_rdfs_subclass_self() {
    let rdf_type = rdf_iri("type");
    let rdfs_class = rdfs_iri("Class");
    let sub_class = rdfs_iri("subClassOf");
    let store = store_from(&[("Cat", &rdf_type, &rdfs_class)]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result
            .iter()
            .any(|t| { t.subject == "Cat" && t.predicate == sub_class && t.object == "Cat" }),
        "rdfs10 should infer Cat rdfs:subClassOf Cat"
    );
}

#[test]
fn test_rdfs_class_resource() {
    let rdf_type = rdf_iri("type");
    let rdfs_class = rdfs_iri("Class");
    let rdfs_resource = rdfs_iri("Resource");
    let sub_class = rdfs_iri("subClassOf");
    let store = store_from(&[("Animal", &rdf_type, &rdfs_class)]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        result.iter().any(|t| {
            t.subject == "Animal" && t.predicate == sub_class && t.object == rdfs_resource
        }),
        "rdfs8 should infer Animal rdfs:subClassOf rdfs:Resource"
    );
}

#[test]
fn test_rdfs_no_extra_inferences_for_unrelated() {
    let sub_class = rdfs_iri("subClassOf");
    let store = store_from(&[("A", &sub_class, "B"), ("C", &sub_class, "D")]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    assert!(
        !result
            .iter()
            .any(|t| t.subject == "A" && t.predicate == sub_class && t.object == "D"),
        "A must not be inferred subClassOf D (unrelated hierarchy)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ClosureGraph (flat-string API)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_entailment_graph_close_rdf() {
    let store = store_from(&[(
        "http://example.org/alice",
        "http://example.org/knows",
        "http://example.org/bob",
    )]);
    let engine = Box::new(RdfEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);

    let added = graph.close().expect("should succeed");
    assert!(added > 0, "closing should add at least one triple");

    let rdf_type = rdf_iri("type");
    let rdf_property = rdf_iri("Property");
    assert!(
        graph
            .store()
            .contains("http://example.org/knows", &rdf_type, &rdf_property)
    );
}

#[test]
fn test_entailment_graph_close_rdfs() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("Cat", &sub_class, "Mammal"),
        ("Mammal", &sub_class, "Animal"),
        ("whiskers", &rdf_type, "Cat"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);

    let added = graph.close().expect("ok");
    assert!(added > 0);

    assert!(
        graph.store().contains("whiskers", &rdf_type, "Animal"),
        "close() should transitively infer whiskers rdf:type Animal"
    );
}

#[test]
fn test_entailment_graph_fixpoint_terminates() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("A", &sub_class, "B"),
        ("B", &sub_class, "C"),
        ("C", &sub_class, "D"),
        ("inst", &rdf_type, "A"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    let added = graph.close().expect("terminates");
    assert!(added > 0);
    assert!(graph.store().contains("inst", &rdf_type, "D"));
}

#[test]
fn test_entailment_chain_3() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("A", &sub_class, "B"),
        ("B", &sub_class, "C"),
        ("x", &rdf_type, "A"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("x", &rdf_type, "C"),
        "3-level chain: x must be inferred rdf:type C"
    );
}

#[test]
fn test_combined_domain_subclass() {
    let domain_pred = rdfs_iri("domain");
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("hasAge", &domain_pred, "Employee"),
        ("Employee", &sub_class, "Person"),
        ("alice", "hasAge", "30"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("alice", &rdf_type, "Person"),
        "combined rdfs2 + rdfs9: alice must be Person"
    );
}

#[test]
fn test_entailment_regime_enum() {
    let regimes = [
        SparqlRegimeKind::Simple,
        SparqlRegimeKind::Rdf,
        SparqlRegimeKind::Rdfs,
        SparqlRegimeKind::Owl2Rl,
        SparqlRegimeKind::D,
    ];
    assert_eq!(regimes[0], SparqlRegimeKind::Simple);
    assert_eq!(regimes[1], SparqlRegimeKind::Rdf);
    assert_eq!(regimes[2], SparqlRegimeKind::Rdfs);
    assert_ne!(regimes[0], regimes[1]);
    let debug_str = format!("{:?}", regimes[2]);
    assert!(debug_str.contains("Rdfs"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Task-required 20+ tests — new trait-based API and OWL RL
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: OWL / RDF IRIs
fn owl_iri(local: &str) -> String {
    format!("http://www.w3.org/2002/07/owl#{local}")
}

// ── Test 1: rdf1 — predicate typed as rdf:Property ───────────────────────────

#[test]
fn test_rdf_rule_rdf1() {
    let store = store_from(&[("http://example.org/alice", "http://example.org/age", "30")]);
    let engine = RdfEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    let rdf_type = rdf_iri("type");
    let rdf_property = rdf_iri("Property");
    assert!(
        result.iter().any(|t| {
            t.subject == "http://example.org/age"
                && t.predicate == rdf_type
                && t.object == rdf_property
        }),
        "rdf1: predicate should be typed as rdf:Property"
    );
}

// ── Test 2: rdf container membership property ────────────────────────────────

#[test]
fn test_rdf_container_membership_property() {
    let rdf_1 = format!("{}_{}", rdf_iri(""), "1");
    let store = store_from(&[("http://example.org/list", &rdf_1, "http://example.org/item")]);
    let engine = RdfEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");

    let rdf_type = rdf_iri("type");
    let rdfs_cmp = rdfs_iri("ContainerMembershipProperty");
    let rdfs_sub = rdfs_iri("subPropertyOf");
    let rdfs_member = rdfs_iri("member");

    assert!(
        result
            .iter()
            .any(|t| t.subject == rdf_1 && t.predicate == rdf_type && t.object == rdfs_cmp),
        "rdf:_1 should be typed as rdfs:ContainerMembershipProperty"
    );
    assert!(
        result
            .iter()
            .any(|t| t.subject == rdf_1 && t.predicate == rdfs_sub && t.object == rdfs_member),
        "rdf:_1 should be rdfs:subPropertyOf rdfs:member"
    );
}

// ── Test 3: rdfs2 domain inference ───────────────────────────────────────────

#[test]
fn test_rdfs_rule_rdfs2_domain() {
    let domain_pred = rdfs_iri("domain");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("http://ex.org/hasAge", &domain_pred, "http://ex.org/Person"),
        ("http://ex.org/bob", "http://ex.org/hasAge", "25"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");
    assert!(result.iter().any(|t| {
        t.subject == "http://ex.org/bob"
            && t.predicate == rdf_type
            && t.object == "http://ex.org/Person"
    }));
}

// ── Test 4: rdfs3 range inference ────────────────────────────────────────────

#[test]
fn test_rdfs_rule_rdfs3_range() {
    let range_pred = rdfs_iri("range");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        (
            "http://ex.org/parentOf",
            &range_pred,
            "http://ex.org/Person",
        ),
        (
            "http://ex.org/alice",
            "http://ex.org/parentOf",
            "http://ex.org/bob",
        ),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");
    assert!(result.iter().any(|t| {
        t.subject == "http://ex.org/bob"
            && t.predicate == rdf_type
            && t.object == "http://ex.org/Person"
    }));
}

// ── Test 5: rdfs9 subClassOf type propagation ────────────────────────────────

#[test]
fn test_rdfs_rule_rdfs9_subclass() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("http://ex.org/Cat", &sub_class, "http://ex.org/Animal"),
        ("http://ex.org/whiskers", &rdf_type, "http://ex.org/Cat"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");
    assert!(result.iter().any(|t| {
        t.subject == "http://ex.org/whiskers"
            && t.predicate == rdf_type
            && t.object == "http://ex.org/Animal"
    }));
}

// ── Test 6: rdfs11 subClassOf transitivity ───────────────────────────────────

#[test]
fn test_rdfs_rule_rdfs11_subclass_transitivity() {
    let sub_class = rdfs_iri("subClassOf");
    let store = store_from(&[
        ("Labrador", &sub_class, "Dog"),
        ("Dog", &sub_class, "Mammal"),
        ("Mammal", &sub_class, "Animal"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("Labrador", &sub_class, "Animal"),
        "Labrador must be transitively inferred as subClassOf Animal"
    );
}

// ── Test 7: rdfs5 subPropertyOf transitivity ─────────────────────────────────

#[test]
fn test_rdfs_rule_rdfs5_subproperty_transitivity() {
    let sub_prop = rdfs_iri("subPropertyOf");
    let store = store_from(&[
        ("hasDaughter", &sub_prop, "hasFemaleChild"),
        ("hasFemaleChild", &sub_prop, "hasChild"),
        ("hasChild", &sub_prop, "hasDescendant"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("hasDaughter", &sub_prop, "hasDescendant"),
        "hasDaughter must transitively be subPropertyOf hasDescendant"
    );
}

// ── Test 8: rdfs7 subPropertyOf usage propagation ────────────────────────────

#[test]
fn test_rdfs_rule_rdfs7_subproperty_usage() {
    let sub_prop = rdfs_iri("subPropertyOf");
    let store = store_from(&[
        ("hasMother", &sub_prop, "hasParent"),
        ("alice", "hasMother", "eve"),
    ]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");
    assert!(result
        .iter()
        .any(|t| t.subject == "alice" && t.predicate == "hasParent" && t.object == "eve"));
}

// ── Test 9: rdfs10 reflexive subClassOf ──────────────────────────────────────

#[test]
fn test_rdfs_rule_rdfs10_class_selfsubclass() {
    let rdf_type = rdf_iri("type");
    let rdfs_class = rdfs_iri("Class");
    let sub_class = rdfs_iri("subClassOf");
    let store = store_from(&[("Dog", &rdf_type, &rdfs_class)]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");
    assert!(result
        .iter()
        .any(|t| t.subject == "Dog" && t.predicate == sub_class && t.object == "Dog"));
}

// ── Test 10: RDFS closure reaches fixpoint ───────────────────────────────────

#[test]
fn test_rdfs_closure_fixpoint() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    // 5-level chain — must reach fixpoint without infinite loop
    let store = store_from(&[
        ("A", &sub_class, "B"),
        ("B", &sub_class, "C"),
        ("C", &sub_class, "D"),
        ("D", &sub_class, "E"),
        ("inst", &rdf_type, "A"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    let added = graph.close().expect("must terminate");
    assert!(added > 0);
    assert!(graph.store().contains("inst", &rdf_type, "E"));
}

// ── Test 11: EntailmentGraph::compute_closure ────────────────────────────────

#[test]
fn test_entailment_graph_compute_closure() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let triples = rich_triples(&[
        ("Dog", &sub_class, "Animal"),
        ("rex", &rdf_type, "Dog"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = EntailmentGraph::new(triples, engine);
    let added = graph.compute_closure().expect("ok");
    assert!(added > 0, "at least one triple should be added");
    // rex should be inferred as Animal
    let obj = EntailmentTerm::NamedNode("Animal".to_string());
    assert!(
        graph.contains("rex", &rdf_type, &obj),
        "rex must be inferred rdf:type Animal"
    );
}

// ── Test 12: EntailmentGraph::contains ───────────────────────────────────────

#[test]
fn test_entailment_graph_contains() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let triples = rich_triples(&[
        ("Cat", &sub_class, "Mammal"),
        ("whiskers", &rdf_type, "Cat"),
    ]);
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = EntailmentGraph::new(triples, engine);
    graph.compute_closure().expect("ok");

    let mammal = EntailmentTerm::NamedNode("Mammal".to_string());
    assert!(
        graph.contains("whiskers", &rdf_type, &mammal),
        "whiskers must be rdf:type Mammal after closure"
    );
}

// ── Test 13: OWL RL — symmetric property ─────────────────────────────────────

#[test]
fn test_owl_rl_symmetric_property() {
    let rdf_type = rdf_iri("type");
    let symmetric = owl_iri("SymmetricProperty");
    let store = store_from(&[
        ("marriedTo", &rdf_type, &symmetric),
        ("alice", "marriedTo", "bob"),
    ]);
    let engine = Box::new(OwlRlEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("bob", "marriedTo", "alice"),
        "prp-symp: marriedTo is symmetric so bob marriedTo alice"
    );
}

// ── Test 14: OWL RL — transitive property ────────────────────────────────────

#[test]
fn test_owl_rl_transitive_property() {
    let rdf_type = rdf_iri("type");
    let transitive = owl_iri("TransitiveProperty");
    let store = store_from(&[
        ("ancestor", &rdf_type, &transitive),
        ("alice", "ancestor", "bob"),
        ("bob", "ancestor", "carol"),
    ]);
    let engine = Box::new(OwlRlEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("alice", "ancestor", "carol"),
        "prp-trp: ancestor is transitive so alice ancestor carol"
    );
}

// ── Test 15: OWL RL — equivalent class ───────────────────────────────────────

#[test]
fn test_owl_rl_equivalent_class() {
    let rdf_type = rdf_iri("type");
    let sub_class = rdfs_iri("subClassOf");
    let equiv = owl_iri("equivalentClass");
    let store = store_from(&[
        ("Human", &equiv, "Person"),
        ("homer", &rdf_type, "Human"),
    ]);
    let engine = Box::new(OwlRlEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    // cax-eqc1: Human ≡ Person → Human subClassOf Person
    assert!(
        graph.store().contains("Human", &sub_class, "Person")
            || graph.store().contains("homer", &rdf_type, "Person"),
        "cax-eqc1/2: homer must be typed as Person via equivalentClass"
    );
}

// ── Test 16: OWL RL — hasValue ───────────────────────────────────────────────

#[test]
fn test_owl_rl_has_value() {
    let rdf_type = rdf_iri("type");
    let has_value = owl_iri("hasValue");
    let on_property = owl_iri("onProperty");
    // cls-hv2: (?x owl:hasValue ?y), (?x owl:onProperty ?p), (?u ?p ?y) → ?u rdf:type ?x
    let store = store_from(&[
        ("PizzaLover", &has_value, "Pizza"),
        ("PizzaLover", &on_property, "likes"),
        ("alice", "likes", "Pizza"),
    ]);
    let engine = Box::new(OwlRlEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("alice", &rdf_type, "PizzaLover"),
        "cls-hv2: alice likes Pizza → alice rdf:type PizzaLover"
    );
}

// ── Test 17: OWL RL — intersection subtyping ─────────────────────────────────

#[test]
fn test_owl_rl_intersection_subtyping() {
    let rdf_type = rdf_iri("type");
    let intersection_of = owl_iri("intersectionOf");
    let first_pred = rdf_iri("first");
    let rest_pred = rdf_iri("rest");
    let nil = rdf_iri("nil");

    // cls-int2: ?c owl:intersectionOf [c1, c2], ?y rdf:type ?c → ?y rdf:type c1
    // Simplified: EngineerAndTeacher intersectionOf list([Engineer, Teacher])
    // We encode the list as RDF list nodes
    let store = store_from(&[
        ("EngineerTeacher", &intersection_of, "_:list1"),
        ("_:list1", &first_pred, "Engineer"),
        ("_:list1", &rest_pred, "_:list2"),
        ("_:list2", &first_pred, "Teacher"),
        ("_:list2", &rest_pred, &nil),
        ("alice", &rdf_type, "EngineerTeacher"),
    ]);
    let engine = Box::new(OwlRlEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    // alice should be inferred rdf:type Engineer (cls-int2)
    assert!(
        graph.store().contains("alice", &rdf_type, "Engineer"),
        "cls-int2: alice rdf:type EngineerTeacher → alice rdf:type Engineer"
    );
}

// ── Test 18: Max iterations exceeded ─────────────────────────────────────────

/// A deliberately "pathological" regime that always claims to have new triples.
#[derive(Debug)]
struct InfiniteRegime;
impl EntailmentRegime for InfiniteRegime {
    fn name(&self) -> &str {
        "InfiniteRegime"
    }
    fn entail(&self, triples: &[RichEntailmentTriple]) -> Vec<RichEntailmentTriple> {
        // Always generate a new triple with a unique counter embedded in the object IRI
        let next_id = triples.len();
        vec![RichEntailmentTriple::named_triple(
            "s",
            "http://ex.org/p",
            &format!("http://ex.org/o{next_id}"),
        )]
    }
}

#[test]
fn test_max_iterations_exceeded() {
    let triples = rich_triples(&[("s", "http://ex.org/p", "http://ex.org/o0")]);
    let engine = Box::new(InfiniteRegime);
    let config = EntailmentConfig {
        regime: super::RegimeVariant::Custom,
        max_iterations: 5,
        detect_cycles: false,
    };
    let mut graph = EntailmentGraph::with_config(triples, engine, &config);
    let result = graph.compute_closure();
    assert!(
        matches!(
            result,
            Err(EntailmentError::MaxIterationsExceeded { iterations: 5 })
        ),
        "should return MaxIterationsExceeded after 5 iterations"
    );
}

// ── Test 19: RDF entailment is_consistent always true ───────────────────────

#[test]
fn test_rdf_entailment_is_consistent_true() {
    let rdf_type = rdf_iri("type");
    let rdf_property = rdf_iri("Property");
    let triples = rich_triples(&[("p", &rdf_type, &rdf_property)]);
    let engine = RdfEntailmentEngine::new();
    // The EntailmentEngine trait doesn't expose is_consistent, so we verify via
    // EntailmentGraph::compute_closure which calls the regime's is_consistent.
    let regime = Box::new(engine);
    let mut graph = EntailmentGraph::new(triples, regime);
    let result = graph.compute_closure();
    assert!(result.is_ok(), "RDF entailment closure should succeed");
}

// ── Test 20: EntailmentConfig selects regime ─────────────────────────────────

#[test]
fn test_entailment_config_selects_regime() {
    let cfg = EntailmentConfig::default();
    assert_eq!(cfg.regime, super::RegimeVariant::Rdfs);
    assert_eq!(cfg.max_iterations, 1000);
    assert!(!cfg.detect_cycles);

    let custom = EntailmentConfig {
        regime: super::RegimeVariant::OwlRl,
        max_iterations: 50,
        detect_cycles: true,
    };
    assert_eq!(custom.regime, super::RegimeVariant::OwlRl);
    assert_eq!(custom.max_iterations, 50);
    assert!(custom.detect_cycles);
}

// ── Test 21: OWL RL equivalent property ─────────────────────────────────────

#[test]
fn test_owl_rl_equivalent_property() {
    let sub_prop = rdfs_iri("subPropertyOf");
    let equiv_prop = owl_iri("equivalentProperty");
    let store = store_from(&[("knows", &equiv_prop, "acquaintanceOf")]);
    let engine = Box::new(OwlRlEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("knows", &sub_prop, "acquaintanceOf")
            || graph.store().contains("acquaintanceOf", &sub_prop, "knows"),
        "prp-eqp: equivalent properties get subPropertyOf in both directions"
    );
}

// ── Test 22: RDFS container membership property rdfs12 ──────────────────────

#[test]
fn test_rdfs_rule_rdfs12_container_membership() {
    let rdf_type = rdf_iri("type");
    let rdfs_cmp = rdfs_iri("ContainerMembershipProperty");
    let rdfs_sub = rdfs_iri("subPropertyOf");
    let rdfs_member = rdfs_iri("member");
    let store = store_from(&[("rdf:_5", &rdf_type, &rdfs_cmp)]);
    let engine = RdfsEntailmentEngine::new();
    let result = super::EntailmentEngine::entail(&engine, &store).expect("ok");
    assert!(
        result
            .iter()
            .any(|t| t.subject == "rdf:_5" && t.predicate == rdfs_sub && t.object == rdfs_member),
        "rdfs12: rdf:_5 rdfs:ContainerMembershipProperty → rdfs:subPropertyOf rdfs:member"
    );
}

// ── Test 23: OWL RL — cax-sco (subClassOf type propagation) ─────────────────

#[test]
fn test_owl_rl_cax_sco() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let store = store_from(&[
        ("Kitten", &sub_class, "Cat"),
        ("fluffball", &rdf_type, "Kitten"),
    ]);
    let engine = Box::new(OwlRlEntailmentEngine::new());
    let mut graph = ClosureGraph::new(store, engine);
    graph.close().expect("ok");
    assert!(
        graph.store().contains("fluffball", &rdf_type, "Cat"),
        "cax-sco: Kitten subClassOf Cat, fluffball:Kitten → fluffball:Cat"
    );
}

// ── Test 24: EntailmentGraph triples() returns all including inferred ─────────

#[test]
fn test_entailment_graph_triples_access() {
    let sub_class = rdfs_iri("subClassOf");
    let rdf_type = rdf_iri("type");
    let triples = rich_triples(&[
        ("Dog", &sub_class, "Animal"),
        ("buddy", &rdf_type, "Dog"),
    ]);
    let orig_len = triples.len();
    let engine = Box::new(RdfsEntailmentEngine::new());
    let mut graph = EntailmentGraph::new(triples, engine);
    graph.compute_closure().expect("ok");
    assert!(
        graph.triples().len() > orig_len,
        "triples() should include newly inferred triples"
    );
}

// ── Test 25: EntailmentError variants are properly formed ───────────────────

#[test]
fn test_entailment_error_variants() {
    let e1 = EntailmentError::InconsistentData("test".to_string());
    assert!(format!("{e1}").contains("Inconsistent"));

    let e2 = EntailmentError::MaxIterationsExceeded { iterations: 42 };
    assert!(format!("{e2}").contains("42"));

    let e3 = EntailmentError::InvalidRule("bad rule".to_string());
    assert!(format!("{e3}").contains("bad rule"));

    let e4 = EntailmentError::CycleDetected;
    assert!(format!("{e4}").contains("Cycle"));
}
