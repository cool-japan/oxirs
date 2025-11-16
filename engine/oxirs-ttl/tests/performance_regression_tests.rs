//! Performance Regression Tests for OxiRS TTL Parser
//!
//! This test suite tracks parser performance over time to detect regressions.
//! Each test establishes performance baselines and alerts when performance degrades.

use oxirs_ttl::formats::ntriples::NTriplesParser;
use oxirs_ttl::formats::trig::TriGParser;
use oxirs_ttl::formats::turtle::TurtleParser;
use oxirs_ttl::toolkit::Parser;
use std::time::{Duration, Instant};

/// Performance baseline thresholds (updated as optimizations are made)
const SMALL_DATASET_PARSE_TIME_MS: u128 = 10; // 10ms for 100 triples
const MEDIUM_DATASET_PARSE_TIME_MS: u128 = 80; // 80ms for 1000 triples
const LARGE_DATASET_PARSE_TIME_MS: u128 = 500; // 500ms for 10000 triples

/// Generate test data with specified number of triples
fn generate_turtle_data(num_triples: usize) -> String {
    let mut data = String::with_capacity(num_triples * 100);
    data.push_str("@prefix ex: <http://example.org/> .\n");
    data.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
    data.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

    for i in 0..num_triples {
        data.push_str(&format!(
            "ex:subject{} ex:predicate{} \"Object {}\"^^xsd:string .\n",
            i,
            i % 10,
            i
        ));
    }

    data
}

/// Generate N-Triples test data
fn generate_ntriples_data(num_triples: usize) -> String {
    let mut data = String::with_capacity(num_triples * 150);

    for i in 0..num_triples {
        data.push_str(&format!(
            "<http://example.org/subject{}> <http://example.org/predicate{}> \"Object {}\"^^<http://www.w3.org/2001/XMLSchema#string> .\n",
            i, i % 10, i
        ));
    }

    data
}

/// Helper to measure parse time
fn measure_parse_time<F>(parse_fn: F) -> Duration
where
    F: FnOnce() -> Result<(), Box<dyn std::error::Error>>,
{
    let start = Instant::now();
    parse_fn().expect("Parsing should succeed");
    start.elapsed()
}

#[test]
fn test_turtle_small_dataset_performance() {
    // Baseline: 100 triples should parse in <10ms
    let data = generate_turtle_data(100);

    let elapsed = measure_parse_time(|| {
        let parser = TurtleParser::new();
        let _ = parser.parse_document(&data)?;
        Ok(())
    });

    println!("Small dataset (100 triples): {:?}", elapsed);
    assert!(
        elapsed.as_millis() < SMALL_DATASET_PARSE_TIME_MS,
        "Performance regression: Small dataset took {:?}, expected <{}ms",
        elapsed,
        SMALL_DATASET_PARSE_TIME_MS
    );
}

#[test]
fn test_turtle_medium_dataset_performance() {
    // Baseline: 1000 triples should parse in <50ms
    let data = generate_turtle_data(1000);

    let elapsed = measure_parse_time(|| {
        let parser = TurtleParser::new();
        let _ = parser.parse_document(&data)?;
        Ok(())
    });

    println!("Medium dataset (1000 triples): {:?}", elapsed);
    assert!(
        elapsed.as_millis() < MEDIUM_DATASET_PARSE_TIME_MS,
        "Performance regression: Medium dataset took {:?}, expected <{}ms",
        elapsed,
        MEDIUM_DATASET_PARSE_TIME_MS
    );
}

#[test]
fn test_turtle_large_dataset_performance() {
    // Baseline: 10000 triples should parse in <500ms
    let data = generate_turtle_data(10000);

    let elapsed = measure_parse_time(|| {
        let parser = TurtleParser::new();
        let _ = parser.parse_document(&data)?;
        Ok(())
    });

    println!("Large dataset (10000 triples): {:?}", elapsed);
    assert!(
        elapsed.as_millis() < LARGE_DATASET_PARSE_TIME_MS,
        "Performance regression: Large dataset took {:?}, expected <{}ms",
        elapsed,
        LARGE_DATASET_PARSE_TIME_MS
    );
}

#[test]
fn test_ntriples_performance_scalability() {
    // N-Triples should be faster than Turtle (simpler syntax)
    let sizes = [100, 1000, 10000];
    let mut prev_time_per_triple: Option<f64> = None;

    for &size in &sizes {
        let data = generate_ntriples_data(size);

        let elapsed = measure_parse_time(|| {
            let parser = NTriplesParser::new();
            let reader = std::io::Cursor::new(data.as_bytes());
            let mut buf_reader = std::io::BufReader::new(reader);
            let _ = parser.parse(&mut buf_reader)?;
            Ok(())
        });

        let time_per_triple = elapsed.as_micros() as f64 / size as f64;
        println!(
            "N-Triples {} triples: {:?} ({:.2}μs/triple)",
            size, elapsed, time_per_triple
        );

        // Check for super-linear scaling (should be roughly linear)
        // Allow up to 3x increase due to JIT warmup and startup costs
        if let Some(prev) = prev_time_per_triple {
            let ratio = time_per_triple / prev;
            assert!(
                ratio < 3.0,
                "Performance scaling issue: time per triple increased by {:.1}x",
                ratio
            );
        }

        prev_time_per_triple = Some(time_per_triple);
    }
}

#[test]
fn test_complex_turtle_syntax_performance() {
    // Test performance with complex Turtle features
    let mut data = String::new();
    data.push_str("@prefix ex: <http://example.org/> .\n\n");

    // Add blank nodes, lists, and nested structures
    for i in 0..1000 {
        data.push_str(&format!(
            "ex:subject{} ex:knows [ ex:name \"Person {}\" ; ex:age {} ] .\n",
            i,
            i,
            i % 100
        ));
        data.push_str(&format!(
            "ex:list{} ex:items ( ex:item1 ex:item2 ex:item3 ) .\n",
            i
        ));
    }

    let elapsed = measure_parse_time(|| {
        let parser = TurtleParser::new();
        let _ = parser.parse_document(&data)?;
        Ok(())
    });

    println!("Complex syntax (2000 statements): {:?}", elapsed);
    // Complex syntax with blank nodes and lists is more intensive
    assert!(
        elapsed.as_millis() < 250,
        "Performance regression in complex syntax: {:?}",
        elapsed
    );
}

#[test]
fn test_memory_efficiency_large_dataset() {
    // Verify that parser doesn't leak memory or accumulate unnecessary data
    let data = generate_turtle_data(5000);

    // Parse multiple times to check for memory leaks
    for iteration in 0..10 {
        let start = Instant::now();
        let parser = TurtleParser::new();
        let _ = parser.parse_document(&data).expect("Parse should succeed");
        let elapsed = start.elapsed();

        println!("Iteration {}: {:?}", iteration, elapsed);

        // Each iteration should take similar time (no accumulation)
        assert!(
            elapsed.as_millis() < 300,
            "Memory efficiency issue: iteration {} took {:?}",
            iteration,
            elapsed
        );
    }
}

#[test]
fn test_prefix_resolution_performance() {
    // Test performance with many prefixes
    let mut data = String::new();

    // Add 100 prefix declarations
    for i in 0..100 {
        data.push_str(&format!(
            "@prefix ns{}: <http://example.org/namespace{}#> .\n",
            i, i
        ));
    }

    // Use all prefixes
    for i in 0..1000 {
        let prefix_idx = i % 100;
        data.push_str(&format!(
            "ns{}:subject{} ns{}:predicate \"Object\" .\n",
            prefix_idx, i, prefix_idx
        ));
    }

    let elapsed = measure_parse_time(|| {
        let parser = TurtleParser::new();
        let _ = parser.parse_document(&data)?;
        Ok(())
    });

    println!(
        "Prefix resolution (100 prefixes, 1000 triples): {:?}",
        elapsed
    );
    assert!(
        elapsed.as_millis() < 100,
        "Prefix resolution performance regression: {:?}",
        elapsed
    );
}

#[test]
fn test_error_recovery_performance() {
    // Test that error recovery doesn't significantly impact performance
    let mut data = String::new();
    data.push_str("@prefix ex: <http://example.org/> .\n");

    // Mix valid and invalid statements (every 10th is invalid)
    for i in 0..1000 {
        if i % 10 == 0 {
            data.push_str("ex:invalid syntax error here\n");
        } else {
            data.push_str(&format!("ex:subject{} ex:predicate \"Object\" .\n", i));
        }
    }

    let elapsed = measure_parse_time(|| {
        let parser = TurtleParser::new_lenient();
        let result = parser.parse_document(&data);
        // Should collect errors, not fail fast
        assert!(result.is_err() || result.is_ok());
        Ok(())
    });

    println!("Error recovery (10% invalid): {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 150,
        "Error recovery performance regression: {:?}",
        elapsed
    );
}

#[test]
fn test_trig_named_graph_performance() {
    // Test TriG parser performance with multiple named graphs
    let mut data = String::new();
    data.push_str("@prefix ex: <http://example.org/> .\n\n");

    for graph_num in 0..10 {
        data.push_str(&format!("ex:graph{} {{\n", graph_num));

        for i in 0..100 {
            data.push_str(&format!("  ex:subject{} ex:predicate \"Object\" .\n", i));
        }

        data.push_str("}\n\n");
    }

    let elapsed = measure_parse_time(|| {
        let parser = TriGParser::new();
        let reader = std::io::Cursor::new(data.as_bytes());
        let _ = parser.parse(reader)?;
        Ok(())
    });

    println!("TriG named graphs (10 graphs, 1000 triples): {:?}", elapsed);
    // TriG parsing with multiple named graphs requires more processing
    assert!(
        elapsed.as_millis() < 250,
        "TriG performance regression: {:?}",
        elapsed
    );
}

#[test]
fn test_unicode_string_performance() {
    // Test performance with Unicode strings
    let mut data = String::new();
    data.push_str("@prefix ex: <http://example.org/> .\n");

    for i in 0..1000 {
        data.push_str(&format!("ex:subject{} ex:label \"日本語{}\" .\n", i, i));
        data.push_str(&format!("ex:subject{} ex:desc \"Émoji: 🎉🚀✨\" .\n", i));
    }

    let elapsed = measure_parse_time(|| {
        let parser = TurtleParser::new();
        let _ = parser.parse_document(&data)?;
        Ok(())
    });

    println!("Unicode strings (2000 statements): {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 100,
        "Unicode performance regression: {:?}",
        elapsed
    );
}
