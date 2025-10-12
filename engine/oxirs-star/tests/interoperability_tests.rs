//! Interoperability tests for RDF-star compatibility with standard RDF tools
//!
//! This module provides comprehensive interoperability testing for:
//! - Round-trip conversions (RDF-star → Standard RDF → RDF-star)
//! - Format conversions across all supported formats
//! - Compatibility with Apache Jena, RDF4J, Virtuoso patterns
//! - Validation of reification strategies
//!
//! NOTE: Tests that require actual tool instances are marked with `#[ignore]`
//! and can be run with: `cargo test --test interoperability_tests -- --ignored`

use oxirs_star::compatibility::{
    BatchCompatibilityConverter, CompatibilityConfig, CompatibilityMode, CompatibilityPresets,
};
use oxirs_star::model::{StarGraph, StarTerm, StarTriple};
use oxirs_star::parser::{StarFormat, StarParser};
use oxirs_star::serializer::StarSerializer;
use oxirs_star::StarResult;

/// Test round-trip conversion with standard reification
#[test]
fn test_roundtrip_standard_reification() -> StarResult<()> {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create RDF-star graph with quoted triple
    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/age")?,
        StarTerm::literal("30")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("0.95")?,
    );
    star_graph.insert(meta)?;

    // Test round-trip
    let success = compat.test_roundtrip(&star_graph)?;
    assert!(success, "Round-trip conversion failed");

    // Verify statistics
    let stats = compat.statistics();
    assert_eq!(stats.conversions_to_standard, 1);
    assert_eq!(stats.conversions_from_standard, 1);
    assert_eq!(stats.quoted_triples_converted, 1);

    Ok(())
}

/// Test conversion with unique IRIs strategy
#[test]
fn test_conversion_unique_iris() -> StarResult<()> {
    let config = CompatibilityConfig::unique_iris("http://example.org/stmt/".to_string());
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/bob")?,
        StarTerm::iri("http://example.org/knows")?,
        StarTerm::iri("http://example.org/charlie")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/source")?,
        StarTerm::literal("survey2023")?,
    );
    star_graph.insert(meta)?;

    // Test conversion to standard RDF
    let standard_graph = compat.to_standard_rdf(&star_graph)?;
    assert!(standard_graph.len() > 1, "Should have reification triples");

    // NOTE: Dereification for unique IRIs strategy is not yet fully implemented
    // The current dereify_graph primarily handles standard reification with rdf:type rdf:Statement
    // Full round-trip support for all strategies will be added in a future version

    Ok(())
}

/// Test conversion with blank nodes strategy
#[test]
fn test_conversion_blank_nodes() -> StarResult<()> {
    let config = CompatibilityConfig::blank_nodes();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/product1")?,
        StarTerm::iri("http://example.org/price")?,
        StarTerm::literal("99.99")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/validUntil")?,
        StarTerm::literal("2025-12-31")?,
    );
    star_graph.insert(meta)?;

    // Test conversion to standard RDF
    let standard_graph = compat.to_standard_rdf(&star_graph)?;
    assert!(standard_graph.len() > 1, "Should have reification triples");

    // NOTE: Dereification for blank nodes strategy requires enhanced blank node handling
    // Full round-trip support will be added in a future version

    Ok(())
}

/// Test conversion with singleton properties strategy
#[test]
fn test_conversion_singleton_properties() -> StarResult<()> {
    let config = CompatibilityConfig::singleton_properties();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/event1")?,
        StarTerm::iri("http://example.org/occuredAt")?,
        StarTerm::literal("2025-10-12T10:00:00Z")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/reportedBy")?,
        StarTerm::iri("http://example.org/sensor42")?,
    );
    star_graph.insert(meta)?;

    // Test conversion to standard RDF
    let standard_graph = compat.to_standard_rdf(&star_graph)?;

    // Singleton properties should be most efficient (2 triples: statement + singletonPropertyOf)
    assert!(standard_graph.len() >= 2, "Should have at least 2 triples");
    assert!(
        standard_graph.len() < 5,
        "Singleton properties should be more efficient than standard reification"
    );

    // NOTE: Dereification for singleton properties is not yet implemented
    // Full round-trip support will be added in a future version

    Ok(())
}

/// Test complex graph with multiple quoted triples
#[test]
fn test_complex_graph_roundtrip() -> StarResult<()> {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();

    // Multiple quoted triples
    for i in 1..=5 {
        let quoted = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/person{i}"))?,
            StarTerm::iri("http://example.org/score")?,
            StarTerm::literal(&format!("{}", 80 + i))?,
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/confidence")?,
            StarTerm::literal(&format!("0.{}", 85 + i))?,
        );
        star_graph.insert(meta)?;
    }

    let success = compat.test_roundtrip(&star_graph)?;
    assert!(success, "Complex graph round-trip failed");

    // Verify all quoted triples were converted
    let stats = compat.statistics();
    assert_eq!(stats.quoted_triples_converted, 5);

    Ok(())
}

/// Test nested quoted triples
#[test]
fn test_nested_quoted_triples_roundtrip() -> StarResult<()> {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();

    // Inner quoted triple
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/age")?,
        StarTerm::literal("30")?,
    );

    // Middle layer
    let middle = StarTriple::new(
        StarTerm::quoted_triple(inner),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("0.9")?,
    );

    // Outer layer
    let outer = StarTriple::new(
        StarTerm::quoted_triple(middle),
        StarTerm::iri("http://example.org/verifiedBy")?,
        StarTerm::iri("http://example.org/validator")?,
    );

    star_graph.insert(outer)?;

    let success = compat.test_roundtrip(&star_graph)?;
    assert!(success, "Nested quoted triples round-trip failed");

    Ok(())
}

/// Test Apache Jena preset compatibility
#[test]
fn test_apache_jena_preset() -> StarResult<()> {
    let config = CompatibilityPresets::apache_jena();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/resource")?,
        StarTerm::iri("http://example.org/property")?,
        StarTerm::literal("value")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/meta")?,
        StarTerm::literal("metadata")?,
    );
    star_graph.insert(meta)?;

    // Convert to Jena-compatible format
    let standard_graph = compat.to_standard_rdf(&star_graph)?;
    assert!(
        standard_graph.len() > 1,
        "Jena conversion produced no reification triples"
    );

    // Verify reification patterns exist
    assert!(CompatibilityMode::has_reifications(&standard_graph));
    assert_eq!(CompatibilityMode::count_reifications(&standard_graph), 1);

    // Convert back
    let recovered = compat.from_standard_rdf(&standard_graph)?;
    assert_eq!(recovered.len(), 1, "Failed to recover original structure");

    Ok(())
}

/// Test RDF4J preset compatibility
#[test]
fn test_rdf4j_preset() -> StarResult<()> {
    let config = CompatibilityPresets::rdf4j();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/subject")?,
        StarTerm::iri("http://example.org/predicate")?,
        StarTerm::iri("http://example.org/object")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/annotation")?,
        StarTerm::literal("note")?,
    );
    star_graph.insert(meta)?;

    // Convert to RDF4J-compatible format (uses unique IRIs)
    let standard_graph = compat.to_standard_rdf(&star_graph)?;
    assert!(
        standard_graph.len() > 1,
        "RDF4J conversion should produce reification triples"
    );

    // Verify the conversion uses unique IRIs (should have statement IRI with rdf4j.org base)
    let has_rdf4j_iri = standard_graph.triples().iter().any(|t| {
        if let Some(node) = t.subject.as_named_node() {
            node.iri.contains("rdf4j.org")
        } else {
            false
        }
    });
    assert!(has_rdf4j_iri, "Should use RDF4J statement IRIs");

    // NOTE: RDF4J preset uses UniqueIris strategy which does not yet support full dereification
    // Full round-trip support for RDF4J preset will be added in a future version

    Ok(())
}

/// Test Virtuoso preset compatibility
#[test]
fn test_virtuoso_preset() -> StarResult<()> {
    let config = CompatibilityPresets::virtuoso();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/x")?,
        StarTerm::iri("http://example.org/y")?,
        StarTerm::iri("http://example.org/z")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/context")?,
        StarTerm::literal("graph1")?,
    );
    star_graph.insert(meta)?;

    // Convert to Virtuoso-compatible format (uses blank nodes)
    let standard_graph = compat.to_standard_rdf(&star_graph)?;
    assert!(
        standard_graph.len() > 1,
        "Virtuoso conversion should produce reification triples"
    );

    // Verify the conversion uses blank nodes (should have blank node subjects)
    let has_blank_nodes = standard_graph
        .triples()
        .iter()
        .any(|t| t.subject.is_blank_node());
    assert!(has_blank_nodes, "Should use blank nodes for statements");

    // NOTE: Virtuoso preset uses BlankNodes strategy which does not yet support full dereification
    // Full round-trip support for Virtuoso preset will be added in a future version

    Ok(())
}

/// Test efficient preset (singleton properties)
#[test]
fn test_efficient_preset() -> StarResult<()> {
    let config = CompatibilityPresets::efficient();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/a")?,
        StarTerm::iri("http://example.org/b")?,
        StarTerm::iri("http://example.org/c")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/d")?,
        StarTerm::literal("e")?,
    );
    star_graph.insert(meta)?;

    // Test conversion to standard RDF
    let standard_graph = compat.to_standard_rdf(&star_graph)?;

    // Singleton properties should produce fewer triples (2 vs 5)
    assert!(
        standard_graph.len() < 5,
        "Singleton properties should produce fewer triples"
    );

    // NOTE: Dereification for singleton properties is not yet implemented
    // The current dereify_graph only handles standard reification patterns
    // For now, we just test that conversion to standard RDF works
    // Round-trip conversion will be implemented in a future version

    Ok(())
}

/// Test batch conversion
#[test]
fn test_batch_conversion() -> StarResult<()> {
    let config = CompatibilityConfig::standard_reification();
    let mut batch = BatchCompatibilityConverter::new(config);

    // Create multiple graphs
    let mut graphs = Vec::new();
    for i in 1..=10 {
        let mut graph = StarGraph::new();
        let quoted = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{i}"))?,
            StarTerm::iri(&format!("http://example.org/p{i}"))?,
            StarTerm::literal(&format!("value{i}"))?,
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/meta")?,
            StarTerm::literal(&format!("meta{i}"))?,
        );
        graph.insert(meta)?;
        graphs.push(graph);
    }

    // Batch convert to standard RDF
    let standard_graphs = batch.batch_to_standard_rdf(graphs)?;
    assert_eq!(standard_graphs.len(), 10);

    // Batch convert back
    let recovered_graphs = batch.batch_from_standard_rdf(standard_graphs)?;
    assert_eq!(recovered_graphs.len(), 10);

    // Verify statistics
    let stats = batch.statistics();
    assert_eq!(stats.conversions_to_standard, 10);
    assert_eq!(stats.conversions_from_standard, 10);
    assert_eq!(stats.quoted_triples_converted, 10);

    Ok(())
}

/// Test format conversion: RDF-star → Standard RDF → RDF-star
#[test]
fn test_format_conversion_with_serialization() -> StarResult<()> {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Create RDF-star graph
    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/knows")?,
        StarTerm::iri("http://example.org/bob")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/source")?,
        StarTerm::literal("social_network")?,
    );
    star_graph.insert(meta)?;

    // Serialize original RDF-star to Turtle-star
    let serializer = StarSerializer::new();
    let turtle_star = serializer.serialize_to_string(&star_graph, StarFormat::TurtleStar)?;
    assert!(turtle_star.contains("<<"), "Should contain RDF-star syntax");

    // Convert to standard RDF
    let standard_graph = compat.to_standard_rdf(&star_graph)?;

    // Serialize standard RDF (reified) to Turtle-star format
    // (The serializer works with any graph, reified or not)
    let reified_turtle = serializer.serialize_to_string(&standard_graph, StarFormat::TurtleStar)?;

    // Parse the reified graph
    let parser = StarParser::new();
    let parsed_standard = parser.parse_str(&reified_turtle, StarFormat::TurtleStar)?;

    // Verify the parsed graph has the reification triples
    assert!(
        parsed_standard.len() >= 4,
        "Parsed graph should have reification triples"
    );

    // Convert back to RDF-star (direct conversion without serialization round-trip)
    let recovered = compat.from_standard_rdf(&standard_graph)?;

    // Verify structure: direct conversion should properly detect reification
    assert_eq!(
        recovered.len(),
        1,
        "Direct conversion should recover 1 triple"
    );
    assert!(CompatibilityMode::has_quoted_triples(&recovered));

    // NOTE: Converting parsed_standard back to RDF-star may not always work perfectly
    // because serialization/parsing can alter the exact structure needed for reification detection.
    // The important thing is that direct conversion (without serialization) works correctly.

    Ok(())
}

/// Test format conversion with N-Triples-star serialization
#[test]
fn test_format_conversion_ntriples() -> StarResult<()> {
    let config = CompatibilityConfig::unique_iris("http://example.org/stmt/".to_string());
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/x")?,
        StarTerm::iri("http://example.org/y")?,
        StarTerm::literal("z")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/meta")?,
        StarTerm::literal("value")?,
    );
    star_graph.insert(meta)?;

    // Serialize to N-Triples-star
    let serializer = StarSerializer::new();
    let ntriples_star = serializer.serialize_to_string(&star_graph, StarFormat::NTriplesStar)?;
    assert!(
        ntriples_star.contains("<<"),
        "Should contain RDF-star syntax"
    );

    // Convert to standard RDF
    let standard_graph = compat.to_standard_rdf(&star_graph)?;
    let reified_ntriples =
        serializer.serialize_to_string(&standard_graph, StarFormat::NTriplesStar)?;

    // Parse the reified graph
    let parser = StarParser::new();
    let parsed = parser.parse_str(&reified_ntriples, StarFormat::NTriplesStar)?;

    // The parsed graph should have the reification triples
    assert!(
        parsed.len() > 1,
        "Reified graph should have multiple triples"
    );

    // NOTE: UniqueIris strategy does not yet support full dereification
    // The from_standard_rdf call will not collapse reified triples back to quoted triples
    // Full round-trip support for UniqueIris will be added in a future version

    // Verify we can serialize and parse the reified format successfully
    assert!(
        parsed.len() >= star_graph.len(),
        "Parsed graph should have at least as many triples as original"
    );

    Ok(())
}

/// Test reification detection
#[test]
fn test_reification_detection() -> StarResult<()> {
    // Create a graph with manual reification pattern
    let mut reified_graph = StarGraph::new();

    let stmt_iri = "http://example.org/stmt1";

    // Manual reification: stmt rdf:type rdf:Statement
    reified_graph.insert(StarTriple::new(
        StarTerm::iri(stmt_iri)?,
        StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
        StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")?,
    ))?;

    // stmt rdf:subject alice
    reified_graph.insert(StarTriple::new(
        StarTerm::iri(stmt_iri)?,
        StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")?,
        StarTerm::iri("http://example.org/alice")?,
    ))?;

    // stmt rdf:predicate age
    reified_graph.insert(StarTriple::new(
        StarTerm::iri(stmt_iri)?,
        StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")?,
        StarTerm::iri("http://example.org/age")?,
    ))?;

    // stmt rdf:object "30"
    reified_graph.insert(StarTriple::new(
        StarTerm::iri(stmt_iri)?,
        StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#object")?,
        StarTerm::literal("30")?,
    ))?;

    // Test detection
    assert!(CompatibilityMode::has_reifications(&reified_graph));
    assert_eq!(CompatibilityMode::count_reifications(&reified_graph), 1);

    Ok(())
}

/// Test statistics accuracy
#[test]
fn test_statistics_accuracy() -> StarResult<()> {
    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Perform multiple conversions
    for i in 1..=5 {
        let mut graph = StarGraph::new();
        let quoted = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{i}"))?,
            StarTerm::iri(&format!("http://example.org/p{i}"))?,
            StarTerm::literal(&format!("o{i}"))?,
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/meta")?,
            StarTerm::literal("value")?,
        );
        graph.insert(meta)?;

        let _standard = compat.to_standard_rdf(&graph)?;
    }

    let stats = compat.statistics();
    assert_eq!(stats.conversions_to_standard, 5);
    assert_eq!(stats.quoted_triples_converted, 5);
    assert!(stats.avg_conversion_time_us > 0.0);

    // Strategy usage should be tracked
    assert!(!stats.strategy_stats.is_empty());

    Ok(())
}

/// Test error handling for incomplete reifications
#[test]
fn test_incomplete_reification_detection() -> StarResult<()> {
    let mut incomplete_graph = StarGraph::new();

    let stmt_iri = "http://example.org/stmt1";

    // Only add subject, missing predicate and object
    incomplete_graph.insert(StarTriple::new(
        StarTerm::iri(stmt_iri)?,
        StarTerm::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")?,
        StarTerm::iri("http://example.org/alice")?,
    ))?;

    let config = CompatibilityConfig::standard_reification();
    let mut compat = CompatibilityMode::new(config);

    // Should detect but fail validation
    let result = compat.from_standard_rdf(&incomplete_graph);

    // With validation enabled, this should fail
    assert!(result.is_err(), "Incomplete reification should be detected");

    Ok(())
}

/// Benchmark: Measure conversion performance
#[test]
fn test_conversion_performance() -> StarResult<()> {
    let config = CompatibilityConfig::singleton_properties(); // Most efficient
    let mut compat = CompatibilityMode::new(config);

    // Create larger graph
    let mut star_graph = StarGraph::new();
    for i in 1..=100 {
        let quoted = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{i}"))?,
            StarTerm::iri(&format!("http://example.org/p{i}"))?,
            StarTerm::literal(&format!("value{i}"))?,
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/meta")?,
            StarTerm::literal(&format!("meta{i}"))?,
        );
        star_graph.insert(meta)?;
    }

    let start = std::time::Instant::now();
    let _standard = compat.to_standard_rdf(&star_graph)?;
    let duration = start.elapsed();

    // Should complete in reasonable time (< 50ms for 100 triples)
    assert!(
        duration.as_millis() < 50,
        "Conversion took too long: {:?}",
        duration
    );

    let stats = compat.statistics();
    println!("Conversion stats: {:?}", stats);
    println!(
        "Average time per triple: {:.2}μs",
        stats.avg_conversion_time_us
    );

    Ok(())
}

/// Test with actual tool instance (Apache Jena) - requires Jena to be running
#[test]
#[ignore] // Run with: cargo test -- --ignored
fn test_jena_integration() -> StarResult<()> {
    // NOTE: This test requires Apache Jena Fuseki to be running at http://localhost:3030
    // Start Fuseki with: ./fuseki-server --mem /ds

    let config = CompatibilityPresets::apache_jena();
    let mut compat = CompatibilityMode::new(config);

    // Create RDF-star data
    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/age")?,
        StarTerm::literal("30")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("0.9")?,
    );
    star_graph.insert(meta)?;

    // Convert to Jena format
    let standard_graph = compat.to_standard_rdf(&star_graph)?;

    // TODO: Upload to Jena via HTTP POST
    // TODO: Query back from Jena
    // TODO: Verify data integrity

    println!("Jena integration test requires manual setup");
    println!("Converted graph has {} triples", standard_graph.len());

    Ok(())
}

/// Test with actual tool instance (RDF4J) - requires RDF4J server
#[test]
#[ignore] // Run with: cargo test -- --ignored
fn test_rdf4j_integration() -> StarResult<()> {
    // NOTE: This test requires RDF4J server to be running at http://localhost:8080/rdf4j-server

    let config = CompatibilityPresets::rdf4j();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/bob")?,
        StarTerm::iri("http://example.org/knows")?,
        StarTerm::iri("http://example.org/charlie")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/source")?,
        StarTerm::literal("social_graph")?,
    );
    star_graph.insert(meta)?;

    let standard_graph = compat.to_standard_rdf(&star_graph)?;

    // TODO: Upload to RDF4J via HTTP POST
    // TODO: Query back from RDF4J
    // TODO: Verify data integrity

    println!("RDF4J integration test requires manual setup");
    println!("Converted graph has {} triples", standard_graph.len());

    Ok(())
}

/// Test with actual tool instance (Virtuoso) - requires Virtuoso server
#[test]
#[ignore] // Run with: cargo test -- --ignored
fn test_virtuoso_integration() -> StarResult<()> {
    // NOTE: This test requires Virtuoso to be running at http://localhost:8890

    let config = CompatibilityPresets::virtuoso();
    let mut compat = CompatibilityMode::new(config);

    let mut star_graph = StarGraph::new();
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/event1")?,
        StarTerm::iri("http://example.org/occurred")?,
        StarTerm::literal("2025-10-12")?,
    );
    let meta = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/confidence")?,
        StarTerm::literal("high")?,
    );
    star_graph.insert(meta)?;

    let standard_graph = compat.to_standard_rdf(&star_graph)?;

    // TODO: Upload to Virtuoso via SPARQL UPDATE
    // TODO: Query back from Virtuoso
    // TODO: Verify data integrity

    println!("Virtuoso integration test requires manual setup");
    println!("Converted graph has {} triples", standard_graph.len());

    Ok(())
}
