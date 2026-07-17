//! Regression tests for loading SHACL shapes directly from an RDF `Store`.
//!
//! Previously `ShapeParser::find_shape_nodes` unconditionally returned an empty
//! vector, so `Validator::load_shapes_from_store` reported success with zero
//! shapes loaded — making every subsequent validation trivially conform (a
//! silent false negative). These tests pin the real shape-node discovery.

use oxirs_core::{model::*, ConcreteStore};
use oxirs_shacl::Validator;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const SH_NODE_SHAPE: &str = "http://www.w3.org/ns/shacl#NodeShape";
const SH_TARGET_CLASS: &str = "http://www.w3.org/ns/shacl#targetClass";
const SH_PATH: &str = "http://www.w3.org/ns/shacl#path";
const SH_MIN_COUNT: &str = "http://www.w3.org/ns/shacl#minCount";
const SH_PROPERTY: &str = "http://www.w3.org/ns/shacl#property";

fn nn(iri: &str) -> NamedNode {
    NamedNode::new(iri).expect("valid IRI")
}

fn insert(store: &ConcreteStore, s: NamedNode, p: &str, o: Object) {
    store
        .insert_quad(Quad::new(s, nn(p), o, GraphName::DefaultGraph))
        .expect("insert quad");
}

#[test]
fn test_load_shapes_from_store_discovers_typed_node_shape() {
    let store = ConcreteStore::new().expect("store");

    let person_shape = nn("http://example.org/PersonShape");
    // ex:PersonShape a sh:NodeShape ; sh:targetClass ex:Person .
    insert(
        &store,
        person_shape.clone(),
        RDF_TYPE,
        Object::NamedNode(nn(SH_NODE_SHAPE)),
    );
    insert(
        &store,
        person_shape.clone(),
        SH_TARGET_CLASS,
        Object::NamedNode(nn("http://example.org/Person")),
    );

    let mut validator = Validator::new();
    let count = validator
        .load_shapes_from_store(&store, None)
        .expect("load shapes");

    assert!(
        count >= 1,
        "expected at least one shape discovered from the store, got {count}"
    );
}

#[test]
fn test_load_shapes_from_store_discovers_shape_via_target_only() {
    // A subject bearing sh:targetClass but no explicit rdf:type must still be
    // recognized as a shape.
    let store = ConcreteStore::new().expect("store");

    let shape = nn("http://example.org/UntypedShape");
    insert(
        &store,
        shape.clone(),
        SH_TARGET_CLASS,
        Object::NamedNode(nn("http://example.org/Thing")),
    );
    insert(
        &store,
        shape,
        SH_PATH,
        Object::NamedNode(nn("http://example.org/name")),
    );

    let mut validator = Validator::new();
    let count = validator
        .load_shapes_from_store(&store, None)
        .expect("load shapes");

    assert!(
        count >= 1,
        "expected a shape discovered via sh:targetClass/sh:path, got {count}"
    );
}

#[test]
fn test_load_shapes_from_store_with_property_shape() {
    let store = ConcreteStore::new().expect("store");

    let shape = nn("http://example.org/PropShape");
    let prop = nn("http://example.org/PropShape_name");
    insert(
        &store,
        shape.clone(),
        RDF_TYPE,
        Object::NamedNode(nn(SH_NODE_SHAPE)),
    );
    insert(&store, shape, SH_PROPERTY, Object::NamedNode(prop.clone()));
    insert(
        &store,
        prop.clone(),
        SH_PATH,
        Object::NamedNode(nn("http://example.org/name")),
    );
    insert(
        &store,
        prop,
        SH_MIN_COUNT,
        Object::Literal(Literal::new_simple_literal("1")),
    );

    let mut validator = Validator::new();
    let count = validator
        .load_shapes_from_store(&store, None)
        .expect("load shapes");

    assert!(
        count >= 1,
        "expected the node shape (and its property shape) to be discovered, got {count}"
    );
}

#[test]
fn test_load_shapes_from_empty_store_is_ok_and_zero() {
    let store = ConcreteStore::new().expect("store");
    let mut validator = Validator::new();
    let count = validator
        .load_shapes_from_store(&store, None)
        .expect("load shapes from empty store must not error");
    assert_eq!(count, 0);
}

#[test]
fn test_load_shapes_from_store_with_only_data_yields_zero() {
    // A non-empty store containing only instance data (no shapes) must load
    // zero shapes without erroring (the parser warns internally).
    let store = ConcreteStore::new().expect("store");
    insert(
        &store,
        nn("http://example.org/alice"),
        RDF_TYPE,
        Object::NamedNode(nn("http://example.org/Person")),
    );

    let mut validator = Validator::new();
    let count = validator
        .load_shapes_from_store(&store, None)
        .expect("load shapes");
    assert_eq!(count, 0);
}
