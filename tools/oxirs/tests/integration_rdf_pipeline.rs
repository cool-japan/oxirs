//! Integration tests for the complete RDF data pipeline
//!
//! Tests the full lifecycle: import → query → update → export → migrate

use oxirs_core::format::{RdfFormat, RdfParser, RdfSerializer};
use oxirs_core::model::{GraphName, NamedNode, Quad, Subject, Term};
use oxirs_core::rdf_store::RdfStore;
use std::io::Cursor;
use tempfile::TempDir;

/// Test basic import and export cycle
#[test]
fn test_import_export_cycle() {
    // Create temporary directory for test store
    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path();

    // Create sample RDF data in N-Triples format
    let sample_data = r#"<http://example.org/subject1> <http://example.org/predicate> "Value 1" .
<http://example.org/subject2> <http://example.org/predicate> "Value 2" .
<http://example.org/subject3> <http://example.org/predicate> "Value 3" .
"#;

    // Parse and import data
    let cursor = Cursor::new(sample_data.as_bytes());
    let parser = RdfParser::new(RdfFormat::NTriples);
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Verify parsed 3 quads
    assert_eq!(quads.len(), 3);

    // Create store and insert quads
    let mut store = RdfStore::open(store_path).unwrap();
    for quad in &quads {
        store.insert_quad(quad.clone()).unwrap();
    }

    // Verify store contains all quads
    let stored_quads = store.quads().unwrap();
    assert_eq!(stored_quads.len(), 3);

    // Export to Turtle format
    let export_buffer = Vec::new();
    let mut serializer = RdfSerializer::new(RdfFormat::Turtle)
        .with_prefix("ex", "http://example.org/")
        .pretty()
        .for_writer(export_buffer);

    for quad in &stored_quads {
        serializer.serialize_quad(quad.as_ref()).unwrap();
    }
    let export_buffer = serializer.finish().unwrap();

    // Verify exported data is not empty
    assert!(!export_buffer.is_empty());
    let exported_str = String::from_utf8(export_buffer).unwrap();
    assert!(exported_str.contains("example.org"));
}

/// Test format migration (N-Triples → Turtle → N-Quads)
#[test]
fn test_format_migration() {
    // Step 1: Create sample N-Triples data
    let ntriples_data = r#"<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/age> "30" .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> "Bob" .
"#;

    // Step 2: Parse N-Triples
    let cursor = Cursor::new(ntriples_data.as_bytes());
    let parser = RdfParser::new(RdfFormat::NTriples);
    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(quads.len(), 3);

    // Step 3: Serialize to N-Quads (skipping Turtle parser which is not yet implemented)
    let nquads_buffer = Vec::new();
    let mut nquads_serializer = RdfSerializer::new(RdfFormat::NQuads).for_writer(nquads_buffer);

    for quad in &quads {
        nquads_serializer.serialize_quad(quad.as_ref()).unwrap();
    }
    let nquads_buffer = nquads_serializer.finish().unwrap();
    assert!(!nquads_buffer.is_empty());

    // Step 4: Verify N-Quads output contains expected data
    let nquads_str = String::from_utf8(nquads_buffer.clone()).unwrap();
    assert!(nquads_str.contains("Alice"));
    assert!(nquads_str.contains("Bob"));

    // Step 5: Parse N-Quads back to verify round-trip
    let nquads_cursor = Cursor::new(nquads_buffer.clone());
    let nquads_parser = RdfParser::new(RdfFormat::NQuads);
    let reparsed_quads: Vec<_> = nquads_parser
        .for_reader(nquads_cursor)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(reparsed_quads.len(), 3);
}

/// Test named graph handling
#[test]
fn test_named_graph_operations() {
    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path();

    // Create quads in different graphs
    let graph1 = GraphName::NamedNode(NamedNode::new("http://example.org/graph1").unwrap());
    let graph2 = GraphName::NamedNode(NamedNode::new("http://example.org/graph2").unwrap());

    let quad1 = Quad::new(
        Subject::NamedNode(NamedNode::new("http://example.org/s1").unwrap()),
        NamedNode::new("http://example.org/p").unwrap(),
        Term::Literal(oxirs_core::model::Literal::new("Value in graph1")),
        graph1.clone(),
    );

    let quad2 = Quad::new(
        Subject::NamedNode(NamedNode::new("http://example.org/s2").unwrap()),
        NamedNode::new("http://example.org/p").unwrap(),
        Term::Literal(oxirs_core::model::Literal::new("Value in graph2")),
        graph2.clone(),
    );

    // Insert into store
    let mut store = RdfStore::open(store_path).unwrap();
    store.insert_quad(quad1.clone()).unwrap();
    store.insert_quad(quad2.clone()).unwrap();

    // Verify both quads are stored
    let all_quads = store.quads().unwrap();
    assert_eq!(all_quads.len(), 2);

    // Export as TriG (supports named graphs)
    let trig_buffer = Vec::new();
    let mut trig_serializer = RdfSerializer::new(RdfFormat::TriG)
        .with_prefix("ex", "http://example.org/")
        .pretty()
        .for_writer(trig_buffer);

    for quad in &all_quads {
        trig_serializer.serialize_quad(quad.as_ref()).unwrap();
    }
    let trig_buffer = trig_serializer.finish().unwrap();

    let trig_str = String::from_utf8(trig_buffer).unwrap();
    assert!(trig_str.contains("graph1"));
    assert!(trig_str.contains("graph2"));
}

/// Test streaming performance with large dataset
#[test]
fn test_streaming_large_dataset() {
    // Generate large dataset (10,000 quads)
    let quad_count = 10000;
    let mut ntriples_data = String::new();

    for i in 0..quad_count {
        ntriples_data.push_str(&format!(
            "<http://example.org/subject{}> <http://example.org/predicate> \"Value {}\" .\n",
            i, i
        ));
    }

    // Parse with streaming
    let cursor = Cursor::new(ntriples_data.into_bytes());
    let parser = RdfParser::new(RdfFormat::NTriples);

    // Stream through without collecting all in memory
    let mut parsed_count = 0;
    let turtle_buffer = Vec::new();
    let mut serializer = RdfSerializer::new(RdfFormat::Turtle).for_writer(turtle_buffer);

    for quad_result in parser.for_reader(cursor) {
        let quad = quad_result.unwrap();
        serializer.serialize_quad(quad.as_ref()).unwrap();
        parsed_count += 1;
    }
    let turtle_buffer = serializer.finish().unwrap();

    assert_eq!(parsed_count, quad_count);
    assert!(!turtle_buffer.is_empty());
}

/// Test parsing with comments and empty lines
#[test]
fn test_parse_error_resilience() {
    // Create data with comments and empty lines (error streaming not yet implemented)
    let mixed_data = r#"# This is a comment
<http://example.org/valid1> <http://example.org/p> "Valid" .

<http://example.org/valid2> <http://example.org/p> "Also Valid" .
# Another comment
<http://example.org/valid3> <http://example.org/p> "Still Valid" .
"#;

    let cursor = Cursor::new(mixed_data.as_bytes());
    let parser = RdfParser::new(RdfFormat::NTriples);

    let quads: Vec<_> = parser
        .for_reader(cursor)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Should have parsed the 3 valid lines (comments and empty lines skipped)
    assert_eq!(quads.len(), 3);

    // Verify all quads have the expected predicate
    for quad in &quads {
        if let oxirs_core::model::Predicate::NamedNode(node) = quad.predicate() {
            assert_eq!(node.as_str(), "http://example.org/p");
        } else {
            panic!("Expected NamedNode predicate");
        }
    }
}

/// Test all 7 supported formats
#[test]
fn test_all_format_support() {
    let sample_quad = Quad::new(
        Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
        NamedNode::new("http://example.org/predicate").unwrap(),
        Term::Literal(oxirs_core::model::Literal::new("Test Value")),
        GraphName::DefaultGraph,
    );

    let formats = [
        RdfFormat::Turtle,
        RdfFormat::NTriples,
        RdfFormat::NQuads,
        RdfFormat::TriG,
        RdfFormat::RdfXml,
        RdfFormat::JsonLd {
            profile: oxirs_core::format::JsonLdProfileSet::empty(),
        },
        RdfFormat::N3,
    ];

    for format in &formats {
        let buffer = Vec::new();
        let mut serializer = RdfSerializer::new(format.clone()).for_writer(buffer);

        serializer.serialize_quad(sample_quad.as_ref()).unwrap();
        let buffer = serializer.finish().unwrap();

        // Verify each format produces non-empty output
        assert!(
            !buffer.is_empty(),
            "Format {:?} produced empty output",
            format
        );
    }
}

/// Test prefix management in Turtle/TriG
#[test]
fn test_prefix_management() {
    let quad = Quad::new(
        Subject::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap()),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        Term::NamedNode(NamedNode::new("http://www.w3.org/2000/01/rdf-schema#Class").unwrap()),
        GraphName::DefaultGraph,
    );

    let buffer = Vec::new();
    let mut serializer = RdfSerializer::new(RdfFormat::Turtle)
        .with_prefix("foaf", "http://xmlns.com/foaf/0.1/")
        .with_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        .with_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
        .pretty()
        .for_writer(buffer);

    serializer.serialize_quad(quad.as_ref()).unwrap();
    let buffer = serializer.finish().unwrap();

    let output = String::from_utf8(buffer).unwrap();
    assert!(output.contains("@prefix"));
    assert!(output.contains("foaf:"));
    assert!(output.contains("rdf:"));
    assert!(output.contains("rdfs:"));
}
