//! Integration tests for RDFS inference via the `OxiRSStore` public API.
//!
//! These tests exercise `apply_rdfs_inference` (the same logic that the WASM
//! `inferRdfs()` binding delegates to) through the store's `insert`/`contains`
//! interface, verifying each of the six core RDFS entailment rules.

use oxirs_wasm::inference::apply_rdfs_inference;
use oxirs_wasm::store::OxiRSStore;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDFS_SUB_CLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_SUB_PROPERTY_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
const RDFS_DOMAIN: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE: &str = "http://www.w3.org/2000/01/rdf-schema#range";

fn ex(local: &str) -> String {
    format!("http://example.org/{local}")
}

// ---------------------------------------------------------------------------
// rdfs11: subClassOf transitivity
// ---------------------------------------------------------------------------

#[test]
fn test_rdfs11_sub_class_of_transitivity() {
    // A subClassOf B, B subClassOf C  =>  A subClassOf C
    let mut store = OxiRSStore::new();
    store.insert(&ex("A"), RDFS_SUB_CLASS_OF, &ex("B"));
    store.insert(&ex("B"), RDFS_SUB_CLASS_OF, &ex("C"));

    let added = apply_rdfs_inference(&mut store);
    assert!(added > 0, "should derive at least one triple");
    assert!(
        store.contains(&ex("A"), RDFS_SUB_CLASS_OF, &ex("C")),
        "A subClassOf C must be inferred via transitivity (rdfs11)"
    );
}

#[test]
fn test_rdfs11_three_level_chain() {
    // A subClassOf B, B subClassOf C, C subClassOf D
    let mut store = OxiRSStore::new();
    store.insert(&ex("A"), RDFS_SUB_CLASS_OF, &ex("B"));
    store.insert(&ex("B"), RDFS_SUB_CLASS_OF, &ex("C"));
    store.insert(&ex("C"), RDFS_SUB_CLASS_OF, &ex("D"));

    apply_rdfs_inference(&mut store);
    assert!(store.contains(&ex("A"), RDFS_SUB_CLASS_OF, &ex("C")));
    assert!(store.contains(&ex("A"), RDFS_SUB_CLASS_OF, &ex("D")));
    assert!(store.contains(&ex("B"), RDFS_SUB_CLASS_OF, &ex("D")));
}

#[test]
fn test_rdfs11_no_self_loops() {
    let mut store = OxiRSStore::new();
    store.insert(&ex("A"), RDFS_SUB_CLASS_OF, &ex("B"));

    apply_rdfs_inference(&mut store);
    assert!(
        !store.contains(&ex("A"), RDFS_SUB_CLASS_OF, &ex("A")),
        "A must not be inferred as subClassOf itself"
    );
}

// ---------------------------------------------------------------------------
// rdfs9: type propagation through subClassOf
// ---------------------------------------------------------------------------

#[test]
fn test_rdfs9_type_propagation() {
    // x rdf:type A, A subClassOf B  =>  x rdf:type B
    let mut store = OxiRSStore::new();
    store.insert(&ex("x"), RDF_TYPE, &ex("A"));
    store.insert(&ex("A"), RDFS_SUB_CLASS_OF, &ex("B"));

    apply_rdfs_inference(&mut store);
    assert!(
        store.contains(&ex("x"), RDF_TYPE, &ex("B")),
        "x type B must be inferred via rdfs9"
    );
}

#[test]
fn test_rdfs9_type_propagation_chain() {
    // x type A, A subClassOf B, B subClassOf C  =>  x type B, x type C
    let mut store = OxiRSStore::new();
    store.insert(&ex("x"), RDF_TYPE, &ex("A"));
    store.insert(&ex("A"), RDFS_SUB_CLASS_OF, &ex("B"));
    store.insert(&ex("B"), RDFS_SUB_CLASS_OF, &ex("C"));

    apply_rdfs_inference(&mut store);
    assert!(store.contains(&ex("x"), RDF_TYPE, &ex("B")));
    assert!(store.contains(&ex("x"), RDF_TYPE, &ex("C")));
}

// ---------------------------------------------------------------------------
// rdfs5: subPropertyOf transitivity
// ---------------------------------------------------------------------------

#[test]
fn test_rdfs5_sub_property_of_transitivity() {
    // a subPropertyOf b, b subPropertyOf c  =>  a subPropertyOf c
    let mut store = OxiRSStore::new();
    store.insert(&ex("a"), RDFS_SUB_PROPERTY_OF, &ex("b"));
    store.insert(&ex("b"), RDFS_SUB_PROPERTY_OF, &ex("c"));

    apply_rdfs_inference(&mut store);
    assert!(
        store.contains(&ex("a"), RDFS_SUB_PROPERTY_OF, &ex("c")),
        "a subPropertyOf c must be inferred via rdfs5"
    );
}

// ---------------------------------------------------------------------------
// rdfs7: subPropertyOf usage propagation
// ---------------------------------------------------------------------------

#[test]
fn test_rdfs7_sub_property_usage_propagation() {
    // narrower subPropertyOf broader, cat narrower animal  =>  cat broader animal
    let mut store = OxiRSStore::new();
    store.insert(&ex("narrower"), RDFS_SUB_PROPERTY_OF, &ex("broader"));
    store.insert(&ex("cat"), &ex("narrower"), &ex("animal"));

    apply_rdfs_inference(&mut store);
    assert!(
        store.contains(&ex("cat"), &ex("broader"), &ex("animal")),
        "cat broader animal must be inferred via rdfs7"
    );
}

// ---------------------------------------------------------------------------
// rdfs2: domain inference
// ---------------------------------------------------------------------------

#[test]
fn test_rdfs2_domain_inference() {
    // knows rdfs:domain Person, Alice knows Bob  =>  Alice rdf:type Person
    let mut store = OxiRSStore::new();
    store.insert(&ex("knows"), RDFS_DOMAIN, &ex("Person"));
    store.insert(&ex("Alice"), &ex("knows"), &ex("Bob"));

    apply_rdfs_inference(&mut store);
    assert!(
        store.contains(&ex("Alice"), RDF_TYPE, &ex("Person")),
        "Alice type Person must be inferred via rdfs2 (domain)"
    );
}

// ---------------------------------------------------------------------------
// rdfs3: range inference
// ---------------------------------------------------------------------------

#[test]
fn test_rdfs3_range_inference() {
    // knows rdfs:range Agent, Alice knows Bob  =>  Bob rdf:type Agent
    let mut store = OxiRSStore::new();
    store.insert(&ex("knows"), RDFS_RANGE, &ex("Agent"));
    store.insert(&ex("Alice"), &ex("knows"), &ex("Bob"));

    apply_rdfs_inference(&mut store);
    assert!(
        store.contains(&ex("Bob"), RDF_TYPE, &ex("Agent")),
        "Bob type Agent must be inferred via rdfs3 (range)"
    );
}

// ---------------------------------------------------------------------------
// Fixed-point properties
// ---------------------------------------------------------------------------

#[test]
fn test_idempotent_fixed_point() {
    let mut store = OxiRSStore::new();
    store.insert(&ex("fido"), RDF_TYPE, &ex("Dog"));
    store.insert(&ex("Dog"), RDFS_SUB_CLASS_OF, &ex("Animal"));

    let added_first = apply_rdfs_inference(&mut store);
    let added_second = apply_rdfs_inference(&mut store);
    assert!(added_first > 0, "first pass should add triples");
    assert_eq!(added_second, 0, "second pass must be idempotent");
}

#[test]
fn test_cyclic_ontology_terminates() {
    // A subClassOf B, B subClassOf A — symmetric cycle must terminate
    let mut store = OxiRSStore::new();
    store.insert(&ex("A"), RDFS_SUB_CLASS_OF, &ex("B"));
    store.insert(&ex("B"), RDFS_SUB_CLASS_OF, &ex("A"));

    // This must not loop infinitely
    apply_rdfs_inference(&mut store);
    // Both directions should be present (transitivity closes with itself)
    assert!(store.contains(&ex("A"), RDFS_SUB_CLASS_OF, &ex("B")));
    assert!(store.contains(&ex("B"), RDFS_SUB_CLASS_OF, &ex("A")));
}

#[test]
fn test_empty_store_no_derivations() {
    let mut store = OxiRSStore::new();
    let added = apply_rdfs_inference(&mut store);
    assert_eq!(added, 0, "empty store should produce no inferences");
}

// ---------------------------------------------------------------------------
// Wasm API surface validation (Rust-side proxy)
// ---------------------------------------------------------------------------

#[test]
fn test_wasm_api_returns_u32() {
    // Validates that `apply_rdfs_inference` returns a u32 as expected by
    // the WASM binding which casts to f64 for JS.
    let mut store = OxiRSStore::new();
    store.insert(&ex("s"), RDFS_DOMAIN, &ex("C"));
    store.insert(&ex("x"), &ex("s"), &ex("o"));
    let added: u32 = apply_rdfs_inference(&mut store);
    assert!(added > 0);
    // The value must fit in an f64 without loss (JS Number)
    let as_f64 = added as f64;
    assert_eq!(as_f64 as u32, added);
}
