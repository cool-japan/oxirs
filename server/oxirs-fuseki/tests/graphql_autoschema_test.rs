//! Acceptance tests for the oxirs-gql schema auto-generation integration
//! (`oxirs_fuseki::graphql_autoschema` + `GraphQLService::auto_generated_sdl`).
//!
//! These prove the first integration step end to end: a fuseki dataset's RDF
//! vocabulary is introspected by `oxirs-gql`'s `SchemaGenerator` and rendered as
//! a GraphQL SDL, without disturbing the existing hand-written `/graphql`
//! catalog schema.

use std::sync::Arc;

use oxirs_core::model::{GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject, Triple};
use oxirs_core::Store as _;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDFS_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#Class";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";
const OWL_DATATYPE_PROPERTY: &str = "http://www.w3.org/2002/07/owl#DatatypeProperty";
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

const PERSON: &str = "http://example.org/Person";
const NAME: &str = "http://example.org/name";
const ALICE: &str = "http://example.org/alice";

fn node(iri: &str) -> NamedNode {
    NamedNode::new(iri).expect("valid IRI")
}

fn t(subject: &str, predicate: &str, object_iri: &str) -> Triple {
    Triple::new(
        Subject::NamedNode(node(subject)),
        Predicate::NamedNode(node(predicate)),
        Object::NamedNode(node(object_iri)),
    )
}

fn t_lit(subject: &str, predicate: &str, literal: &str) -> Triple {
    Triple::new(
        Subject::NamedNode(node(subject)),
        Predicate::NamedNode(node(predicate)),
        Object::Literal(Literal::new_simple_literal(literal)),
    )
}

/// A minimal ontology: a `Person` class with a datatype `name` property, plus
/// one instance. This is the shape `SchemaGenerator::extract_vocabulary_from_store`
/// looks for (`?class a rdfs:Class`, `?prop a owl:DatatypeProperty` with
/// domain/range).
fn person_ontology() -> Vec<Triple> {
    vec![
        t(PERSON, RDF_TYPE, RDFS_CLASS),
        t(NAME, RDF_TYPE, OWL_DATATYPE_PROPERTY),
        t(NAME, RDFS_DOMAIN, PERSON),
        t(NAME, RDFS_RANGE, XSD_STRING),
        t(ALICE, RDF_TYPE, PERSON),
        t_lit(ALICE, NAME, "Alice"),
    ]
}

#[test]
fn auto_schema_from_triples_renders_ontology_types() {
    let triples = person_ontology();
    let sdl = oxirs_fuseki::graphql_autoschema::generate_sdl_from_triples(triples.iter())
        .expect("SDL generation from triples must succeed");

    println!("--- generated SDL (from triples) ---\n{sdl}\n--- end SDL ---");

    assert!(!sdl.trim().is_empty(), "generated SDL must not be empty");
    assert!(
        sdl.contains("Person"),
        "the SDL must expose a GraphQL type for the Person class; got:\n{sdl}"
    );
}

#[tokio::test]
async fn auto_schema_for_dataset_via_service_and_store() {
    // Populate a real fuseki store with the same ontology, then drive the
    // integration through the public GraphQLService method (store CONSTRUCT →
    // oxirs-gql schema generation).
    let store = oxirs_fuseki::store::Store::new().expect("store must build");
    for triple in person_ontology() {
        let quad = Quad::new(
            triple.subject().clone(),
            triple.predicate().clone(),
            triple.object().clone(),
            GraphName::DefaultGraph,
        );
        store.insert_quad(quad).expect("insert quad");
    }

    let service = oxirs_fuseki::graphql_integration::GraphQLService::new(Arc::new(store));
    let sdl = service
        .auto_generated_sdl(None)
        .expect("auto_generated_sdl must succeed for the default dataset");

    println!("--- generated SDL (via service) ---\n{sdl}\n--- end SDL ---");

    assert!(
        !sdl.trim().is_empty(),
        "service-generated SDL must not be empty"
    );
    assert!(
        sdl.contains("Person"),
        "the service SDL must expose a GraphQL type for the Person class; got:\n{sdl}"
    );

    // The hand-written catalog schema must remain intact and distinct from the
    // auto-generated one (non-breaking integration).
    let catalog = service.sdl();
    assert!(
        catalog.contains("sparqlQuery")
            || catalog.contains("SparqlQuery")
            || catalog.contains("datasets"),
        "the existing hand-written catalog schema must be unchanged; got:\n{catalog}"
    );
}
