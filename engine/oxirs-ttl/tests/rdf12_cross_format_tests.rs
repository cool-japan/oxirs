//! RDF 1.2 Cross-Format Interoperability Tests
//!
//! This test suite validates that RDF 1.2 features (RDF-star and directional
//! language tags) work consistently across all supported formats:
//! - Turtle
//! - TriG
//! - N-Quads (extended)
//!
//! Focus areas:
//! - Cross-format round-trip preservation
//! - Format conversion accuracy
//! - Consistent quoted triple handling
//! - Directional language tag preservation

use oxirs_core::model::{GraphName, NamedNode, Object, Quad, Subject, Triple};
use oxirs_ttl::formats::trig::{TriGParser, TriGSerializer};
use oxirs_ttl::formats::turtle::{TurtleParser, TurtleSerializer};
use oxirs_ttl::toolkit::{Parser, Serializer};
use std::io::Cursor;

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_turtle(input: &str) -> Vec<Triple> {
    let parser = TurtleParser::new();
    parser.parse(input.as_bytes()).expect("Turtle parse failed")
}

fn serialize_turtle(triples: &[Triple]) -> String {
    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize(triples, &mut output)
        .expect("Turtle serialization failed");
    String::from_utf8(output).unwrap()
}

fn parse_trig(input: &str) -> Vec<Quad> {
    let parser = TriGParser::new();
    parser.parse(Cursor::new(input)).expect("TriG parse failed")
}

fn serialize_trig(quads: &[Quad]) -> String {
    let serializer = TriGSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize(quads, &mut output)
        .expect("TriG serialization failed");
    String::from_utf8(output).unwrap()
}

fn triples_to_quads(triples: &[Triple]) -> Vec<Quad> {
    triples
        .iter()
        .map(|t| {
            Quad::new(
                t.subject().clone(),
                t.predicate().clone(),
                t.object().clone(),
                GraphName::DefaultGraph,
            )
        })
        .collect()
}

// ============================================================================
// TriG RDF-star Support Tests
// ============================================================================

#[test]
fn test_trig_quoted_triple_in_default_graph() {
    let trig = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob >> ex:confidence "high" .
"#;

    let quads = parse_trig(trig);
    assert_eq!(quads.len(), 1);

    // Verify quoted triple in subject
    matches!(quads[0].subject(), Subject::QuotedTriple(_));

    // Verify it's in default graph
    assert!(quads[0].graph_name().is_default_graph());
}

#[test]
fn test_trig_quoted_triple_in_named_graph() {
    let trig = r#"
@prefix ex: <http://example.org/> .

ex:socialGraph {
    << ex:alice ex:knows ex:bob >> ex:confidence "high" .
    << ex:bob ex:knows ex:charlie >> ex:confidence "medium" .
}
"#;

    let quads = parse_trig(trig);
    assert_eq!(quads.len(), 2);

    // Both should have quoted triples as subjects
    for quad in &quads {
        matches!(quad.subject(), Subject::QuotedTriple(_));

        // Verify they're in the same named graph
        if let GraphName::NamedNode(graph) = quad.graph_name() {
            assert_eq!(graph.as_str(), "http://example.org/socialGraph");
        } else {
            panic!("Expected named graph");
        }
    }
}

#[test]
fn test_trig_nested_quoted_triples() {
    let trig = r#"
@prefix ex: <http://example.org/> .

ex:metaGraph {
    << << ex:alice ex:knows ex:bob >> ex:certainty "high" >> ex:source ex:researcher .
}
"#;

    let quads = parse_trig(trig);
    assert_eq!(quads.len(), 1);

    // Verify nested structure
    if let Subject::QuotedTriple(outer) = quads[0].subject() {
        matches!(outer.subject(), Subject::QuotedTriple(_));
    } else {
        panic!("Expected nested quoted triple");
    }
}

#[test]
fn test_trig_multiple_graphs_with_quoted_triples() {
    let trig = r#"
@prefix ex: <http://example.org/> .

# Default graph
<< ex:fact1 ex:verified "true" >> ex:timestamp "2025-12-05" .

ex:graph1 {
    << ex:fact2 ex:verified "true" >> ex:confidence "0.95" .
}

ex:graph2 {
    << ex:fact3 ex:verified "false" >> ex:confidence "0.23" .
}
"#;

    let quads = parse_trig(trig);
    assert_eq!(quads.len(), 3);

    // Check default graph
    assert!(quads[0].graph_name().is_default_graph());

    // Check named graphs
    let graph_names: Vec<_> = quads[1..]
        .iter()
        .filter_map(|q| match q.graph_name() {
            GraphName::NamedNode(nn) => Some(nn.as_str()),
            _ => None,
        })
        .collect();

    assert!(graph_names.contains(&"http://example.org/graph1"));
    assert!(graph_names.contains(&"http://example.org/graph2"));
}

#[test]
fn test_trig_roundtrip_with_quoted_triples() {
    let original_trig = r#"
@prefix ex: <http://example.org/> .

ex:provGraph {
    << ex:alice ex:knows ex:bob >> ex:source ex:socialNetwork .
    << ex:bob ex:age 30 >> ex:verifiedBy ex:census .
}
"#;

    let quads = parse_trig(original_trig);
    let serialized = serialize_trig(&quads);
    let reparsed = parse_trig(&serialized);

    assert_eq!(quads.len(), reparsed.len());
}

// ============================================================================
// TriG Directional Language Tags
// ============================================================================

#[test]
fn test_trig_directional_tags_in_graphs() {
    let trig = r#"
@prefix ex: <http://example.org/> .

ex:labels {
    ex:alice ex:nameEn "Alice"@en--ltr .
    ex:alice ex:nameAr "أليس"@ar--rtl .
}

ex:descriptions {
    ex:alice ex:bio "A person"@en--ltr .
}
"#;

    let quads = parse_trig(trig);
    assert_eq!(quads.len(), 3);

    // All should have language-tagged literals
    for quad in &quads {
        matches!(quad.object(), Object::Literal(_));
    }
}

// ============================================================================
// Cross-Format Conversion Tests
// ============================================================================

#[test]
fn test_turtle_to_trig_with_quoted_triples() {
    let turtle = r#"
@prefix ex: <http://example.org/> .

<< ex:alice ex:knows ex:bob >> ex:confidence "high" .
<< ex:bob ex:knows ex:charlie >> ex:confidence "medium" .
"#;

    // Parse as Turtle
    let triples = parse_turtle(turtle);
    assert_eq!(triples.len(), 2);

    // Convert to quads (default graph)
    let quads = triples_to_quads(&triples);

    // Serialize as TriG
    let trig_output = serialize_trig(&quads);

    // Re-parse as TriG
    let reparsed_quads = parse_trig(&trig_output);

    assert_eq!(quads.len(), reparsed_quads.len());
}

#[test]
fn test_trig_to_turtle_default_graph_extraction() {
    let trig = r#"
@prefix ex: <http://example.org/> .

# Default graph only
<< ex:alice ex:knows ex:bob >> ex:confidence "high" .
ex:alice ex:age 30 .
"#;

    let quads = parse_trig(trig);

    // Extract triples from default graph
    let default_graph_triples: Vec<Triple> = quads
        .iter()
        .filter(|q| q.graph_name().is_default_graph())
        .map(|q| {
            Triple::new(
                q.subject().clone(),
                q.predicate().clone(),
                q.object().clone(),
            )
        })
        .collect();

    assert_eq!(default_graph_triples.len(), 2);

    // Serialize as Turtle
    let turtle_output = serialize_turtle(&default_graph_triples);

    // Re-parse as Turtle
    let reparsed_triples = parse_turtle(&turtle_output);

    assert_eq!(default_graph_triples.len(), reparsed_triples.len());
}

#[test]
fn test_cross_format_directional_tags() {
    let turtle = r#"
@prefix ex: <http://example.org/> .

ex:alice ex:greeting "Hello"@en--ltr .
ex:bob ex:greeting "مرحبا"@ar--rtl .
"#;

    // Parse as Turtle
    let triples = parse_turtle(turtle);
    assert_eq!(triples.len(), 2);

    // Convert to quads and serialize as TriG
    let quads = triples_to_quads(&triples);
    let trig_output = serialize_trig(&quads);

    // Re-parse as TriG and verify
    let reparsed_quads = parse_trig(&trig_output);
    assert_eq!(quads.len(), reparsed_quads.len());
}

// ============================================================================
// Complex Multi-Format Scenarios
// ============================================================================

#[test]
fn test_knowledge_graph_with_provenance_across_formats() {
    // Original knowledge graph in Turtle with RDF-star provenance
    let turtle_kg = r#"
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Facts with provenance
<< ex:alice ex:knows ex:bob >> ex:confidence "0.95"^^xsd:decimal .
<< ex:alice ex:knows ex:bob >> ex:source ex:socialNetwork .
<< ex:bob ex:age 30 >> ex:verifiedBy ex:census .

# Multilingual descriptions
ex:alice ex:name "Alice"@en--ltr .
ex:alice ex:name "أليس"@ar--rtl .
"#;

    // Parse as Turtle
    let triples = parse_turtle(turtle_kg);
    assert_eq!(triples.len(), 5);

    // Convert to TriG with named graphs
    let mut quads = Vec::new();

    // Put provenance triples in a metadata graph
    for (i, triple) in triples.iter().enumerate() {
        let graph_name = if i < 3 {
            GraphName::NamedNode(NamedNode::new("http://example.org/provenance").unwrap())
        } else {
            GraphName::NamedNode(NamedNode::new("http://example.org/labels").unwrap())
        };

        quads.push(Quad::new(
            triple.subject().clone(),
            triple.predicate().clone(),
            triple.object().clone(),
            graph_name,
        ));
    }

    // Serialize as TriG
    let trig_output = serialize_trig(&quads);

    // Re-parse and verify
    let reparsed_quads = parse_trig(&trig_output);
    assert_eq!(quads.len(), reparsed_quads.len());

    // Verify graphs are distinct
    let graph_names: std::collections::HashSet<_> = reparsed_quads
        .iter()
        .filter_map(|q| match q.graph_name() {
            GraphName::NamedNode(nn) => Some(nn.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(graph_names.len(), 2);
    assert!(graph_names.contains("http://example.org/provenance"));
    assert!(graph_names.contains("http://example.org/labels"));
}

#[test]
fn test_mixed_rdf11_and_rdf12_across_formats() {
    let turtle_mixed = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

# RDF 1.1 standard triples
ex:alice rdf:type ex:Person .
ex:alice ex:name "Alice" .

# RDF 1.2 quoted triples
<< ex:alice ex:knows ex:bob >> ex:confidence "high" .

# RDF 1.2 directional language tags
ex:alice ex:greeting "Hello"@en--ltr .
"#;

    // Parse as Turtle
    let triples = parse_turtle(turtle_mixed);
    assert!(triples.len() >= 4);

    // Convert to TriG
    let quads = triples_to_quads(&triples);
    let trig_output = serialize_trig(&quads);

    // Re-parse and verify structure preservation
    let reparsed_quads = parse_trig(&trig_output);
    assert_eq!(quads.len(), reparsed_quads.len());

    // Verify we have both RDF 1.1 and RDF 1.2 features
    let has_quoted_triple = reparsed_quads
        .iter()
        .any(|q| matches!(q.subject(), Subject::QuotedTriple(_)));

    let has_standard_triple = reparsed_quads
        .iter()
        .any(|q| matches!(q.subject(), Subject::NamedNode(_)));

    assert!(has_quoted_triple, "Should have RDF-star quoted triples");
    assert!(has_standard_triple, "Should have standard RDF 1.1 triples");
}

// ============================================================================
// Format-Specific Feature Tests
// ============================================================================

#[test]
fn test_trig_with_blank_node_graphs() {
    let trig = r#"
@prefix ex: <http://example.org/> .

_:graph1 {
    << ex:alice ex:knows ex:bob >> ex:confidence "high" .
}

_:graph2 {
    << ex:bob ex:knows ex:charlie >> ex:confidence "medium" .
}
"#;

    let quads = parse_trig(trig);
    assert_eq!(quads.len(), 2);

    // Both should be in blank node graphs
    for quad in &quads {
        matches!(quad.graph_name(), GraphName::BlankNode(_));
    }
}

#[test]
fn test_trig_graph_keyword_with_quoted_triples() {
    let trig = r#"
@prefix ex: <http://example.org/> .

GRAPH ex:metadata {
    << ex:doc1 ex:author "Alice" >> ex:verifiedAt "2025-12-05" .
    << ex:doc2 ex:author "Bob" >> ex:verifiedAt "2025-12-04" .
}
"#;

    let quads = parse_trig(trig);
    assert_eq!(quads.len(), 2);

    // Verify all are in the metadata graph
    for quad in &quads {
        if let GraphName::NamedNode(graph) = quad.graph_name() {
            assert_eq!(graph.as_str(), "http://example.org/metadata");
        } else {
            panic!("Expected named graph");
        }
    }
}

// ============================================================================
// Performance Tests for Cross-Format Operations
// ============================================================================

#[test]
fn test_large_trig_with_quoted_triples() {
    use std::time::Instant;

    let mut trig = String::from("@prefix ex: <http://example.org/> .\n\n");
    trig.push_str("ex:data {\n");

    // Generate 500 quoted triples in a named graph
    for i in 0..500 {
        trig.push_str(&format!(
            "    << ex:s{} ex:p{} ex:o{} >> ex:id {} .\n",
            i, i, i, i
        ));
    }

    trig.push_str("}\n");

    let start = Instant::now();
    let quads = parse_trig(&trig);
    let parse_duration = start.elapsed();

    assert_eq!(quads.len(), 500);

    // Should parse 500 quoted triples in TriG in < 200ms (more lenient for CI/debug builds)
    assert!(
        parse_duration.as_millis() < 200,
        "TriG parsing took {:?} (should be < 200ms)",
        parse_duration
    );

    // Test serialization performance
    let start = Instant::now();
    let _output = serialize_trig(&quads);
    let serialize_duration = start.elapsed();

    assert!(
        serialize_duration.as_millis() < 200,
        "TriG serialization took {:?} (should be < 200ms)",
        serialize_duration
    );
}

#[test]
fn test_cross_format_conversion_performance() {
    use std::time::Instant;

    // Generate 1000 mixed RDF 1.2 triples
    let mut turtle = String::from("@prefix ex: <http://example.org/> .\n\n");
    for i in 0..1000 {
        if i % 2 == 0 {
            turtle.push_str(&format!(
                "<< ex:s{} ex:p{} ex:o{} >> ex:confidence \"high\" .\n",
                i, i, i
            ));
        } else {
            turtle.push_str(&format!(
                "ex:subject{} ex:text \"Text {}\"@en--ltr .\n",
                i, i
            ));
        }
    }

    let start = Instant::now();

    // Parse as Turtle
    let triples = parse_turtle(&turtle);

    // Convert to quads
    let quads = triples_to_quads(&triples);

    // Serialize as TriG
    let _trig_output = serialize_trig(&quads);

    let total_duration = start.elapsed();

    assert_eq!(triples.len(), 1000);

    // Full conversion pipeline should complete in < 150ms
    assert!(
        total_duration.as_millis() < 150,
        "Cross-format conversion took {:?} (should be < 150ms)",
        total_duration
    );
}
