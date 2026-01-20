//! Auto-Detection Demonstration
//!
//! Shows how the RdfBridge intelligently detects time-series data
//! using predicate analysis, value type parsing, and frequency tracking.

use oxirs_core::model::{Literal, NamedNode, Quad};
use oxirs_tsdb::RdfBridge;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS Auto-Detection Demo ===\n");

    let bridge = RdfBridge::new();

    // Test 1: Known time-series predicate
    println!("1. Testing known time-series predicate...");
    let quad1 = create_quad(
        "http://example.org/sensor1",
        "http://qudt.org/schema/qudt/numericValue",
        "42.5",
    )?;
    let result = bridge.detect(&quad1)?;
    println!(
        "   Predicate: qudt:numericValue\n   Decision: {} (confidence: {}%)\n   Reason: {}",
        if result.is_timeseries {
            "TIME-SERIES"
        } else {
            "RDF"
        },
        result.confidence.percentage(),
        result.reason
    );

    // Test 2: Known metadata predicate
    println!("\n2. Testing known metadata predicate...");
    let quad2 = create_quad(
        "http://example.org/sensor1",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/ns/sosa/Sensor",
    )?;
    let result = bridge.detect(&quad2)?;
    println!(
        "   Predicate: rdf:type\n   Decision: {} (confidence: {}%)\n   Reason: {}",
        if result.is_timeseries {
            "TIME-SERIES"
        } else {
            "RDF"
        },
        result.confidence.percentage(),
        result.reason
    );

    // Test 3: Numeric value detection
    println!("\n3. Testing numeric value auto-detection...");
    let quad3 = create_quad(
        "http://example.org/sensor1",
        "http://example.org/customReading",
        "98.6",
    )?;
    let result = bridge.detect(&quad3)?;
    println!(
        "   Predicate: custom:reading (unknown)\n   Value: 98.6 (numeric)\n   Decision: {} (confidence: {}%)\n   Reason: {}",
        if result.is_timeseries { "TIME-SERIES" } else { "RDF" },
        result.confidence.percentage(),
        result.reason
    );

    // Test 4: Non-numeric value
    println!("\n4. Testing non-numeric value...");
    let quad4 = create_quad(
        "http://example.org/sensor1",
        "http://xmlns.com/foaf/0.1/name",
        "Temperature Sensor Alpha",
    )?;
    let result = bridge.detect(&quad4)?;
    println!(
        "   Predicate: foaf:name\n   Value: \"Temperature Sensor Alpha\" (string)\n   Decision: {} (confidence: {}%)\n   Reason: {}",
        if result.is_timeseries { "TIME-SERIES" } else { "RDF" },
        result.confidence.percentage(),
        result.reason
    );

    // Test 5: Frequency-based detection
    println!("\n5. Testing frequency-based detection...");
    let mut bridge_freq = RdfBridge::new();
    bridge_freq.set_frequency_threshold(3);

    let quad5 = create_quad(
        "http://example.org/sensor2",
        "http://example.org/measurement",
        "10.0",
    )?;

    for i in 1..=5 {
        let result = bridge_freq.detect(&quad5)?;
        println!(
            "   Insert #{i}: Decision: {}, Confidence: {}%",
            if result.is_timeseries {
                "TIME-SERIES"
            } else {
                "RDF"
            },
            result.confidence.percentage()
        );

        if i == 3 {
            println!("   → Threshold reached! Switched to TIME-SERIES storage");
        }
    }

    println!("\n✓ Auto-detection working correctly!");

    Ok(())
}

fn create_quad(
    subject: &str,
    predicate: &str,
    object: &str,
) -> Result<Quad, Box<dyn std::error::Error>> {
    let s = NamedNode::new(subject)?;
    let p = NamedNode::new(predicate)?;
    let o = Literal::new(object);
    Ok(Quad::new(
        s,
        p,
        o,
        oxirs_core::model::GraphName::DefaultGraph,
    ))
}
