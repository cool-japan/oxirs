//! Dataset construction helpers for conformance tests
//!
//! Provides convenient functions for building RDF datasets for conformance testing.

// Re-export framework helpers for use within the conformance module
pub use super::framework::*;

/// Common IRI prefixes used in conformance tests
pub const EX: &str = "http://example.org/";
pub const FOAF: &str = "http://xmlns.com/foaf/0.1/";
pub const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
pub const OWL: &str = "http://www.w3.org/2002/07/owl#";
pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

/// Construct a full IRI from a prefix and local part
pub fn ex(local: &str) -> Term {
    iri(&format!("{EX}{local}"))
}

pub fn foaf(local: &str) -> Term {
    iri(&format!("{FOAF}{local}"))
}

pub fn rdf_type() -> Term {
    iri(&format!("{RDF}type"))
}

pub fn rdfs(local: &str) -> Term {
    iri(&format!("{RDFS}{local}"))
}

/// Build a dataset from a list of triple tuples
pub fn dataset_from_triples(triples: Vec<(Term, Term, Term)>) -> InMemoryDataset {
    let mut ds = InMemoryDataset::new();
    for (s, p, o) in triples {
        ds.add_triple(s, p, o);
    }
    ds
}

/// Standard person dataset used in many basic tests
///
/// Contains:
///   :alice foaf:name "Alice" .
///   :alice foaf:age 30 .
///   :alice rdf:type foaf:Person .
///   :bob foaf:name "Bob" .
///   :bob foaf:age 25 .
///   :bob rdf:type foaf:Person .
///   :charlie foaf:name "Charlie" .
///   :charlie foaf:age 35 .
///   :charlie rdf:type foaf:Person .
pub fn person_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("alice"), foaf("name"), str_lit("Alice")),
        (ex("alice"), iri(&format!("{FOAF}age")), int_lit(30)),
        (ex("alice"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("bob"), foaf("name"), str_lit("Bob")),
        (ex("bob"), iri(&format!("{FOAF}age")), int_lit(25)),
        (ex("bob"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("charlie"), foaf("name"), str_lit("Charlie")),
        (ex("charlie"), iri(&format!("{FOAF}age")), int_lit(35)),
        (ex("charlie"), rdf_type(), iri(&format!("{FOAF}Person"))),
    ])
}

/// Numeric dataset for aggregate/math tests
///
/// Contains:
///   :s1 :value 10 .
///   :s2 :value 20 .
///   :s3 :value 30 .
///   :s4 :value 15 .
///   :s5 :value 25 .
pub fn numeric_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("s1"), ex("value"), int_lit(10)),
        (ex("s2"), ex("value"), int_lit(20)),
        (ex("s3"), ex("value"), int_lit(30)),
        (ex("s4"), ex("value"), int_lit(15)),
        (ex("s5"), ex("value"), int_lit(25)),
    ])
}

/// Graph with hierarchy for property path tests
///
/// Contains (ancestor chain):
///   :a ex:child :b .
///   :b ex:child :c .
///   :c ex:child :d .
///   :d ex:child :e .
///   :a ex:knows :b .
///   :b ex:knows :c .
pub fn hierarchy_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("a"), ex("child"), ex("b")),
        (ex("b"), ex("child"), ex("c")),
        (ex("c"), ex("child"), ex("d")),
        (ex("d"), ex("child"), ex("e")),
        (ex("a"), ex("knows"), ex("b")),
        (ex("b"), ex("knows"), ex("c")),
    ])
}

/// Dataset with optional data for OPTIONAL tests
///
/// Contains:
///   :alice foaf:name "Alice" .
///   :alice foaf:mbox "alice@example.org" .
///   :bob foaf:name "Bob" .
///   (note: no mbox for bob)
///   :charlie foaf:name "Charlie" .
///   :charlie foaf:mbox "charlie@example.org" .
pub fn optional_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("alice"), foaf("name"), str_lit("Alice")),
        (ex("alice"), foaf("mbox"), str_lit("alice@example.org")),
        (ex("bob"), foaf("name"), str_lit("Bob")),
        (ex("charlie"), foaf("name"), str_lit("Charlie")),
        (ex("charlie"), foaf("mbox"), str_lit("charlie@example.org")),
    ])
}

/// Dataset for union tests
///
/// Contains both Person and Organization types
pub fn union_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("alice"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("alice"), foaf("name"), str_lit("Alice")),
        (ex("acme"), rdf_type(), ex("Organization")),
        (ex("acme"), ex("orgName"), str_lit("ACME Corp")),
        (ex("bob"), rdf_type(), iri(&format!("{FOAF}Person"))),
        (ex("bob"), foaf("name"), str_lit("Bob")),
    ])
}

/// Dataset for negation tests (NOT EXISTS / MINUS)
///
/// Contains:
///   :alice foaf:name "Alice" ; foaf:mbox "alice@example" .
///   :bob foaf:name "Bob" . (no email)
///   :charlie foaf:name "Charlie" ; foaf:mbox "charlie@example" .
///   :dave foaf:name "Dave" . (no email)
pub fn negation_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("alice"), foaf("name"), str_lit("Alice")),
        (ex("alice"), foaf("mbox"), str_lit("alice@example")),
        (ex("bob"), foaf("name"), str_lit("Bob")),
        (ex("charlie"), foaf("name"), str_lit("Charlie")),
        (ex("charlie"), foaf("mbox"), str_lit("charlie@example")),
        (ex("dave"), foaf("name"), str_lit("Dave")),
    ])
}

/// Dataset for GROUP BY tests with categories
///
/// Contains:
///   :item1 :category :A ; :price 10 .
///   :item2 :category :A ; :price 20 .
///   :item3 :category :B ; :price 15 .
///   :item4 :category :B ; :price 25 .
///   :item5 :category :A ; :price 30 .
pub fn group_by_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("item1"), ex("category"), ex("A")),
        (ex("item1"), ex("price"), int_lit(10)),
        (ex("item2"), ex("category"), ex("A")),
        (ex("item2"), ex("price"), int_lit(20)),
        (ex("item3"), ex("category"), ex("B")),
        (ex("item3"), ex("price"), int_lit(15)),
        (ex("item4"), ex("category"), ex("B")),
        (ex("item4"), ex("price"), int_lit(25)),
        (ex("item5"), ex("category"), ex("A")),
        (ex("item5"), ex("price"), int_lit(30)),
    ])
}

/// Dataset for string function tests
pub fn string_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("r1"), ex("label"), str_lit("Hello World")),
        (ex("r2"), ex("label"), str_lit("SPARQL Query")),
        (ex("r3"), ex("label"), str_lit("Semantic Web")),
        (ex("r4"), ex("label"), str_lit("RDF Graph")),
        (ex("r1"), ex("code"), str_lit("ABC123")),
        (ex("r2"), ex("code"), str_lit("XYZ789")),
    ])
}

/// Dataset for subquery tests
pub fn subquery_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("alice"), ex("score"), int_lit(85)),
        (ex("bob"), ex("score"), int_lit(92)),
        (ex("charlie"), ex("score"), int_lit(78)),
        (ex("dave"), ex("score"), int_lit(92)),
        (ex("alice"), foaf("name"), str_lit("Alice")),
        (ex("bob"), foaf("name"), str_lit("Bob")),
        (ex("charlie"), foaf("name"), str_lit("Charlie")),
        (ex("dave"), foaf("name"), str_lit("Dave")),
    ])
}

/// Dataset for VALUES clause tests
pub fn values_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("a"), ex("value"), int_lit(1)),
        (ex("b"), ex("value"), int_lit(2)),
        (ex("c"), ex("value"), int_lit(3)),
        (ex("d"), ex("value"), int_lit(4)),
        (ex("e"), ex("value"), int_lit(5)),
        (ex("a"), foaf("name"), str_lit("Resource A")),
        (ex("b"), foaf("name"), str_lit("Resource B")),
        (ex("c"), foaf("name"), str_lit("Resource C")),
    ])
}

/// Dataset with typed literals for type system tests
pub fn typed_dataset() -> InMemoryDataset {
    dataset_from_triples(vec![
        (ex("r1"), ex("intVal"), int_lit(42)),
        (ex("r1"), ex("strVal"), str_lit("hello")),
        (
            ex("r1"),
            ex("boolVal"),
            Term::Literal(Literal {
                value: "true".to_string(),
                language: None,
                datatype: Some(NamedNode::new_unchecked(
                    "http://www.w3.org/2001/XMLSchema#boolean",
                )),
            }),
        ),
        (
            ex("r1"),
            ex("decVal"),
            Term::Literal(Literal {
                value: "3.14".to_string(),
                language: None,
                datatype: Some(NamedNode::new_unchecked(
                    "http://www.w3.org/2001/XMLSchema#decimal",
                )),
            }),
        ),
        (
            ex("r2"),
            ex("intVal"),
            Term::Literal(Literal {
                value: "-7".to_string(),
                language: None,
                datatype: Some(NamedNode::new_unchecked(
                    "http://www.w3.org/2001/XMLSchema#integer",
                )),
            }),
        ),
    ])
}
