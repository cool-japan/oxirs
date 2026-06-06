//! Integration tests for the Jena Assembler vocabulary implementation.
//!
//! Tests cover:
//! - AssemblerConfig helper methods (is_empty, len, find_dataset)
//! - from_triples API (core graph-walking logic)
//! - from_turtle API (end-to-end Turtle parsing)

use oxirs_core::assembler::{from_turtle, AssemblerBuilder, StoreBackend};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn t(s: &str, p: &str, o: &str) -> (String, String, String) {
    (s.to_owned(), p.to_owned(), o.to_owned())
}

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const JA_RDF_DATASET: &str = "http://jena.hpl.hp.com/2005/11/Assembler#RDFDataset";
const JA_MEMORY_DATASET: &str = "http://jena.hpl.hp.com/2005/11/Assembler#MemoryDataset";
const JA_MEMORY_MODEL: &str = "http://jena.hpl.hp.com/2005/11/Assembler#MemoryModel";
const JA_NAMED_GRAPH: &str = "http://jena.hpl.hp.com/2005/11/Assembler#namedGraph";
const JA_GRAPH_NAME: &str = "http://jena.hpl.hp.com/2005/11/Assembler#graphName";
const JA_GRAPH: &str = "http://jena.hpl.hp.com/2005/11/Assembler#graph";
const JA_DEFAULT_GRAPH: &str = "http://jena.hpl.hp.com/2005/11/Assembler#defaultGraph";
const JA_CONTENT_URL: &str = "http://jena.hpl.hp.com/2005/11/Assembler#contentURL";
const TDB2_DATASET: &str = "http://jena.apache.org/2016/tdb#DatasetTDB2";
const TDB2_LOCATION: &str = "http://jena.apache.org/2016/tdb#location";

// ---------------------------------------------------------------------------
// AssemblerConfig helper method tests
// ---------------------------------------------------------------------------

#[test]
fn test_assembler_config_is_empty_when_no_datasets() {
    let config = AssemblerBuilder::from_triples(&[]).unwrap();
    assert!(config.is_empty());
    assert_eq!(config.len(), 0);
}

#[test]
fn test_assembler_config_find_dataset_found() {
    let triples = vec![t("http://example.org/ds", RDF_TYPE, JA_MEMORY_DATASET)];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    let found = config.find_dataset("http://example.org/ds");
    assert!(found.is_some(), "expected to find dataset by IRI");
    assert_eq!(found.unwrap().resource_iri, "http://example.org/ds");
}

#[test]
fn test_assembler_config_find_dataset_not_found() {
    let triples = vec![t("http://example.org/ds", RDF_TYPE, JA_MEMORY_DATASET)];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    let found = config.find_dataset("http://example.org/other");
    assert!(found.is_none(), "should not find non-existent dataset");
}

// ---------------------------------------------------------------------------
// from_triples tests
// ---------------------------------------------------------------------------

#[test]
fn test_in_memory_dataset_from_triples() {
    let triples = vec![t("http://example.org/mem", RDF_TYPE, JA_MEMORY_DATASET)];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    assert_eq!(config.len(), 1);
    let ds = config.find_dataset("http://example.org/mem").unwrap();
    assert_eq!(ds.backend, StoreBackend::InMemory);
    assert!(ds.named_graphs.is_empty());
    assert!(ds.default_graph.is_none());
}

#[test]
fn test_tdb2_dataset_from_triples() {
    let triples = vec![
        t("http://example.org/tdb", RDF_TYPE, TDB2_DATASET),
        t("http://example.org/tdb", TDB2_LOCATION, "\"/data/mydb\""),
    ];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    assert_eq!(config.len(), 1);
    let ds = config.find_dataset("http://example.org/tdb").unwrap();
    match &ds.backend {
        StoreBackend::Tdb2 { location } => {
            assert_eq!(location.to_str().unwrap(), "/data/mydb");
        }
        other => panic!("expected Tdb2, got {other:?}"),
    }
}

#[test]
fn test_named_graph_from_triples() {
    // ex:ds a ja:RDFDataset ;
    //     ja:namedGraph _:b0 .
    // _:b0 ja:graphName <http://example.org/g1> ;
    //      ja:graph     <http://example.org/model1> .
    // ex:model1 a ja:MemoryModel .
    let triples = vec![
        t("http://example.org/ds", RDF_TYPE, JA_RDF_DATASET),
        t("http://example.org/ds", JA_NAMED_GRAPH, "_:b0"),
        t("_:b0", JA_GRAPH_NAME, "http://example.org/g1"),
        t("_:b0", JA_GRAPH, "http://example.org/model1"),
        t("http://example.org/model1", RDF_TYPE, JA_MEMORY_MODEL),
    ];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    let ds = config.find_dataset("http://example.org/ds").unwrap();
    assert_eq!(ds.named_graphs.len(), 1);
    let ng = &ds.named_graphs[0];
    assert_eq!(
        ng.graph_name.as_deref(),
        Some("http://example.org/g1"),
        "graph name IRI should be preserved"
    );
    assert_eq!(ng.backend, StoreBackend::InMemory);
}

#[test]
fn test_default_graph_from_triples() {
    // ex:ds ja:defaultGraph ex:model1 .
    // ex:model1 a ja:MemoryModel .
    let triples = vec![
        t("http://example.org/ds", RDF_TYPE, JA_RDF_DATASET),
        t(
            "http://example.org/ds",
            JA_DEFAULT_GRAPH,
            "http://example.org/model1",
        ),
        t("http://example.org/model1", RDF_TYPE, JA_MEMORY_MODEL),
    ];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    let ds = config.find_dataset("http://example.org/ds").unwrap();
    assert!(ds.named_graphs.is_empty());
    let dg = ds.default_graph.as_ref().unwrap();
    assert!(dg.graph_name.is_none(), "default graph has no graph name");
    assert_eq!(dg.backend, StoreBackend::InMemory);
}

#[test]
fn test_content_url_from_triples() {
    // ex:ds a ja:MemoryDataset ;
    //     ja:namedGraph _:b0 .
    // _:b0 ja:graphName <http://example.org/g1> ;
    //      ja:graph <http://example.org/m1> .
    // ex:m1 ja:contentURL "http://example.org/data.ttl" .
    let triples = vec![
        t("http://example.org/ds", RDF_TYPE, JA_MEMORY_DATASET),
        t("http://example.org/ds", JA_NAMED_GRAPH, "_:b0"),
        t("_:b0", JA_GRAPH_NAME, "http://example.org/g1"),
        t("_:b0", JA_GRAPH, "http://example.org/m1"),
        t(
            "http://example.org/m1",
            JA_CONTENT_URL,
            "\"http://example.org/data.ttl\"",
        ),
    ];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    let ds = config.find_dataset("http://example.org/ds").unwrap();
    let ng = &ds.named_graphs[0];
    assert_eq!(ng.content_urls.len(), 1);
    assert_eq!(ng.content_urls[0], "http://example.org/data.ttl");
}

// ---------------------------------------------------------------------------
// from_turtle tests
// ---------------------------------------------------------------------------

#[test]
fn test_from_turtle_memory_dataset() {
    let ttl = r#"
        @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
        <http://example.org/ds> a ja:MemoryDataset .
    "#;
    let config = from_turtle(ttl).unwrap();
    assert_eq!(config.len(), 1);
    let ds = config.find_dataset("http://example.org/ds").unwrap();
    assert_eq!(ds.backend, StoreBackend::InMemory);
}

#[test]
fn test_from_turtle_tdb2_dataset_with_location() {
    let location = std::env::temp_dir()
        .join(format!("oxirs_assembler_test_tdb2_{}", std::process::id()))
        .display()
        .to_string();
    let ttl = format!(
        r#"
        @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
        @prefix tdb2: <http://jena.apache.org/2016/tdb#> .

        <http://example.org/tdb> a tdb2:DatasetTDB2 ;
            tdb2:location "{location}" .
    "#
    );
    let config = from_turtle(&ttl).unwrap();
    assert_eq!(config.len(), 1);
    let ds = config.find_dataset("http://example.org/tdb").unwrap();
    match &ds.backend {
        StoreBackend::Tdb2 { location: loc } => {
            assert_eq!(loc.to_str().unwrap(), location);
        }
        other => panic!("expected Tdb2, got {other:?}"),
    }
}

#[test]
fn test_from_turtle_named_graph() {
    let ttl = r#"
        @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .

        <http://example.org/ds> a ja:RDFDataset ;
            ja:namedGraph [
                ja:graphName <http://example.org/graph1> ;
                ja:graph <http://example.org/model1>
            ] .

        <http://example.org/model1> a ja:MemoryModel .
    "#;
    let config = from_turtle(ttl).unwrap();
    let ds = config.find_dataset("http://example.org/ds").unwrap();
    assert_eq!(ds.named_graphs.len(), 1);
    let ng = &ds.named_graphs[0];
    assert_eq!(ng.graph_name.as_deref(), Some("http://example.org/graph1"));
    assert_eq!(ng.backend, StoreBackend::InMemory);
}

#[test]
fn test_from_turtle_multiple_datasets() {
    let location = std::env::temp_dir()
        .join(format!("oxirs_assembler_test_multi_{}", std::process::id()))
        .display()
        .to_string();
    let ttl = format!(
        r#"
        @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
        @prefix tdb2: <http://jena.apache.org/2016/tdb#> .

        <http://example.org/ds1> a ja:MemoryDataset .
        <http://example.org/ds2> a tdb2:DatasetTDB2 ;
            tdb2:location "{location}" .
    "#
    );
    let config = from_turtle(&ttl).unwrap();
    assert_eq!(config.len(), 2, "two datasets should be found");
    assert!(config.find_dataset("http://example.org/ds1").is_some());
    assert!(config.find_dataset("http://example.org/ds2").is_some());
}

#[test]
fn test_from_turtle_empty_returns_empty_config() {
    let ttl = r#"
        @prefix ex: <http://example.org/> .
        # No dataset declarations — just a comment
    "#;
    let config = from_turtle(ttl).unwrap();
    assert!(config.is_empty());
}

#[test]
fn test_from_turtle_unknown_backend_preserved() {
    let ttl = r#"
        @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
        @prefix ex: <http://example.org/> .

        ex:ds a ex:CustomBackend .
    "#;
    // ex:CustomBackend is an unrecognised type. The assembler preserves it
    // as a DatasetConfig with backend: Unknown(_) so that callers handling
    // proprietary or future Jena extension backends do not lose the resource.
    let config = from_turtle(ttl).unwrap();
    assert_eq!(
        config.len(),
        1,
        "unrecognised type should still produce a dataset entry"
    );
    let ds = config.find_dataset("http://example.org/ds").unwrap();
    match &ds.backend {
        StoreBackend::Unknown(type_iri) => {
            assert_eq!(type_iri, "http://example.org/CustomBackend");
        }
        other => panic!("expected Unknown, got {other:?}"),
    }
}

#[test]
fn test_from_turtle_rdf_dataset_type() {
    // ja:RDFDataset should also be recognised
    let ttl = r#"
        @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
        <http://example.org/generic> a ja:RDFDataset .
    "#;
    let config = from_turtle(ttl).unwrap();
    assert_eq!(config.len(), 1);
    let ds = config.find_dataset("http://example.org/generic").unwrap();
    // ja:RDFDataset maps to InMemory (generic in-memory dataset)
    assert_eq!(ds.backend, StoreBackend::InMemory);
}

#[test]
fn test_from_triples_unknown_backend_round_trips() {
    // Subjects with unrecognised types produce DatasetConfig entries with
    // backend: Unknown(type_iri), preserving the IRI for callers that handle
    // proprietary or future extension backends.
    let triples = vec![t(
        "http://example.org/ds",
        RDF_TYPE,
        "http://example.org/FutureBackend",
    )];
    let config = AssemblerBuilder::from_triples(&triples).unwrap();
    assert_eq!(
        config.len(),
        1,
        "unknown rdf:type should still produce a dataset entry"
    );
    let ds = config.find_dataset("http://example.org/ds").unwrap();
    match &ds.backend {
        StoreBackend::Unknown(type_iri) => {
            assert_eq!(type_iri, "http://example.org/FutureBackend");
        }
        other => panic!("expected Unknown, got {other:?}"),
    }
}
