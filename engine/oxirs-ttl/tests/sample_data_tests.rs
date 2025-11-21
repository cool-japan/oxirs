//! Integration tests using sample data files
//!
//! These tests verify that the sample RDF files in the data/ directory
//! can be successfully parsed by all applicable parsers.

use oxirs_ttl::nquads::NQuadsParser;
use oxirs_ttl::ntriples::NTriplesParser;
use oxirs_ttl::trig::TriGParser;
use oxirs_ttl::turtle::TurtleParser;
use oxirs_ttl::Parser;
use std::fs;
use std::io::Cursor;

#[test]
fn test_sample_turtle() {
    let content = fs::read_to_string("data/sample.ttl").expect("Failed to read data/sample.ttl");

    let parser = TurtleParser::new();
    let triples = parser
        .parse_document(&content)
        .expect("Failed to parse sample.ttl");

    // Verify we got a reasonable number of triples
    assert!(
        triples.len() >= 20,
        "Expected at least 20 triples, got {}",
        triples.len()
    );

    // Verify some expected content
    let has_alice = triples
        .iter()
        .any(|t| t.subject().to_string().contains("alice"));
    assert!(has_alice, "Should contain triples about Alice");

    let has_foaf_person = triples.iter().any(|t| {
        t.object().to_string().contains("foaf") && t.object().to_string().contains("Person")
    });
    assert!(has_foaf_person, "Should contain foaf:Person types");
}

#[test]
fn test_sample_turtle_streaming() {
    let content = fs::read_to_string("data/sample.ttl").expect("Failed to read data/sample.ttl");

    let parser = TurtleParser::new();
    let cursor = Cursor::new(content.clone());

    let mut count = 0;
    for result in parser.for_reader(cursor) {
        let _triple = result.expect("Failed to parse triple");
        count += 1;
    }

    assert!(
        count >= 20,
        "Expected at least 20 triples via streaming, got {}",
        count
    );
}

#[test]
fn test_sample_ntriples() {
    let content = fs::read_to_string("data/sample.nt").expect("Failed to read data/sample.nt");

    let parser = NTriplesParser::new();
    let triples = parser
        .parse(Cursor::new(content))
        .expect("Failed to parse sample.nt");

    // N-Triples file has fewer triples than Turtle version
    assert!(
        triples.len() >= 10,
        "Expected at least 10 triples, got {}",
        triples.len()
    );

    // Verify full IRIs (no prefixes) - check object since it's more likely to be IRI
    let has_full_iri = triples.iter().any(|t| {
        let obj_str = t.object().to_string();
        obj_str.starts_with("<http://") || obj_str.contains("http://example.org/")
    });
    assert!(has_full_iri, "N-Triples should use full IRIs");
}

#[test]
fn test_sample_ntriples_unicode() {
    let content = fs::read_to_string("data/sample.nt").expect("Failed to read data/sample.nt");

    let parser = NTriplesParser::new();
    let triples = parser
        .parse(Cursor::new(content))
        .expect("Failed to parse sample.nt");

    // Check for Japanese language-tagged literal
    let has_japanese = triples
        .iter()
        .any(|t| t.object().to_string().contains("@ja"));
    assert!(
        has_japanese,
        "Should contain Japanese language-tagged literal"
    );
}

#[test]
fn test_sample_trig() {
    let content = fs::read_to_string("data/sample.trig").expect("Failed to read data/sample.trig");

    let parser = TriGParser::new();
    let quads = parser
        .parse(Cursor::new(content))
        .expect("Failed to parse sample.trig");

    // TriG file has multiple named graphs
    assert!(
        quads.len() >= 15,
        "Expected at least 15 quads, got {}",
        quads.len()
    );

    // Verify we have named graphs
    let has_named_graph = quads.iter().any(|q| !q.graph_name().is_default_graph());
    assert!(has_named_graph, "TriG should have named graphs");

    // Verify specific graph exists
    let has_people_graph = quads
        .iter()
        .any(|q| q.graph_name().to_string().contains("people"));
    assert!(has_people_graph, "Should contain people graph");
}

#[test]
fn test_sample_nquads() {
    let content = fs::read_to_string("data/sample.nq").expect("Failed to read data/sample.nq");

    let parser = NQuadsParser::new();
    let quads = parser
        .parse(Cursor::new(content))
        .expect("Failed to parse sample.nq");

    // N-Quads file should have quads
    assert!(
        quads.len() >= 15,
        "Expected at least 15 quads, got {}",
        quads.len()
    );

    // Count quads with and without graph names
    let default_graph_count = quads
        .iter()
        .filter(|q| q.graph_name().is_default_graph())
        .count();
    let named_graph_count = quads
        .iter()
        .filter(|q| !q.graph_name().is_default_graph())
        .count();

    assert!(default_graph_count > 0, "Should have default graph quads");
    assert!(named_graph_count > 0, "Should have named graph quads");
}

#[test]
fn test_sample_file_consistency() {
    // Parse all files and verify they contain related data
    let ttl_content = fs::read_to_string("data/sample.ttl").unwrap();
    let nt_content = fs::read_to_string("data/sample.nt").unwrap();
    let trig_content = fs::read_to_string("data/sample.trig").unwrap();
    let nq_content = fs::read_to_string("data/sample.nq").unwrap();

    let ttl_triples = TurtleParser::new().parse_document(&ttl_content).unwrap();
    let nt_triples = NTriplesParser::new()
        .parse(Cursor::new(nt_content))
        .unwrap();
    let trig_quads = TriGParser::new().parse(Cursor::new(trig_content)).unwrap();
    let nq_quads = NQuadsParser::new().parse(Cursor::new(nq_content)).unwrap();

    // All files should contain data about Alice, Bob, or Charlie
    assert!(!ttl_triples.is_empty(), "Turtle file should not be empty");
    assert!(!nt_triples.is_empty(), "N-Triples file should not be empty");
    assert!(!trig_quads.is_empty(), "TriG file should not be empty");
    assert!(!nq_quads.is_empty(), "N-Quads file should not be empty");

    // Turtle has more triples due to syntactic sugar
    assert!(
        ttl_triples.len() > nt_triples.len(),
        "Turtle should have more triples than N-Triples"
    );

    // TriG and N-Quads should have similar counts (both quad formats)
    let trig_total = trig_quads.len();
    let nq_total = nq_quads.len();
    let diff = (trig_total as i32 - nq_total as i32).abs();
    assert!(
        diff < 10,
        "TriG and N-Quads should have similar quad counts (diff: {})",
        diff
    );
}

#[test]
fn test_sample_data_directory_exists() {
    // Verify data directory and all sample files exist
    assert!(fs::metadata("data").is_ok(), "data/ directory should exist");
    assert!(
        fs::metadata("data/README.md").is_ok(),
        "data/README.md should exist"
    );
    assert!(
        fs::metadata("data/sample.ttl").is_ok(),
        "data/sample.ttl should exist"
    );
    assert!(
        fs::metadata("data/sample.nt").is_ok(),
        "data/sample.nt should exist"
    );
    assert!(
        fs::metadata("data/sample.trig").is_ok(),
        "data/sample.trig should exist"
    );
    assert!(
        fs::metadata("data/sample.nq").is_ok(),
        "data/sample.nq should exist"
    );
}

#[test]
fn test_sample_files_not_empty() {
    // Verify all sample files have reasonable content
    let ttl = fs::read_to_string("data/sample.ttl").unwrap();
    let nt = fs::read_to_string("data/sample.nt").unwrap();
    let trig = fs::read_to_string("data/sample.trig").unwrap();
    let nq = fs::read_to_string("data/sample.nq").unwrap();

    assert!(ttl.len() > 1000, "sample.ttl should be > 1KB");
    assert!(nt.len() > 1000, "sample.nt should be > 1KB");
    assert!(trig.len() > 1000, "sample.trig should be > 1KB");
    assert!(nq.len() > 1000, "sample.nq should be > 1KB");
}
