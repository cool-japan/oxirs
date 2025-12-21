//! Memory Leak Tests for Streaming Operations
//!
//! These tests verify that streaming parsers don't leak memory when processing
//! large files or when interrupted partway through parsing. Critical for
//! production environments handling multi-GB RDF files.

use oxirs_ttl::formats::turtle::TurtleParser;
use oxirs_ttl::nquads::NQuadsParser;
use oxirs_ttl::ntriples::NTriplesParser;
use oxirs_ttl::streaming::{StreamingConfig, StreamingParser};
use oxirs_ttl::trig::TriGParser;
use oxirs_ttl::Parser;
use std::io::Cursor;

/// Get current process memory usage in bytes (platform-specific)
#[cfg(target_os = "macos")]
fn get_memory_usage() -> usize {
    use std::process::Command;

    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("Failed to run ps command");

    let rss_kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<usize>()
        .unwrap_or(0);

    rss_kb * 1024 // Convert KB to bytes
}

#[cfg(target_os = "linux")]
fn get_memory_usage() -> usize {
    use std::fs;

    let status = fs::read_to_string("/proc/self/status").expect("Failed to read /proc/self/status");

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let rss_kb = parts[1].parse::<usize>().unwrap_or(0);
                return rss_kb * 1024; // Convert KB to bytes
            }
        }
    }

    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_memory_usage() -> usize {
    // Fallback for unsupported platforms
    0
}

/// Generate large RDF document for memory testing
fn generate_large_turtle(num_triples: usize) -> String {
    let mut doc = String::from(
        r#"
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"#,
    );

    for i in 0..num_triples {
        doc.push_str(&format!("ex:person{} foaf:name \"Person {}\" ;\n", i, i));
        doc.push_str(&format!("    foaf:age {} ;\n", (i % 100) + 20));
        doc.push_str(&format!("    ex:email \"person{}@example.org\" .\n\n", i));
    }

    doc
}

/// Generate large N-Triples document
fn generate_large_ntriples(num_triples: usize) -> String {
    let mut doc = String::new();

    for i in 0..num_triples {
        doc.push_str(&format!(
            "<http://example.org/person{}> <http://xmlns.com/foaf/0.1/name> \"Person {}\" .\n",
            i, i
        ));
        doc.push_str(&format!(
            "<http://example.org/person{}> <http://xmlns.com/foaf/0.1/age> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
            i,
            (i % 100) + 20
        ));
    }

    doc
}

#[test]
fn test_turtle_streaming_no_memory_leak() {
    // Skip if we can't measure memory
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    // Force garbage collection
    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Parse large document multiple times with streaming
    for iteration in 0..5 {
        let turtle = generate_large_turtle(1000); // 3000 triples per iteration
        let config = StreamingConfig::default().with_batch_size(100);
        let cursor = Cursor::new(turtle);
        let streaming = StreamingParser::with_config(cursor, config);

        for batch in streaming.batches() {
            assert!(batch.is_ok(), "Iteration {}: Batch should parse", iteration);
            // Immediately drop the batch to ensure memory is freed
            drop(batch);
        }
    }

    // Allow some time for OS to reclaim memory
    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);

    // Memory growth should be less than 10MB (accounting for fragmentation and OS overhead)
    let max_acceptable_growth = 10 * 1024 * 1024;

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected: grew by {} MB (from {} to {} bytes). Max acceptable: {} MB",
        memory_growth / (1024 * 1024),
        initial_memory,
        final_memory,
        max_acceptable_growth / (1024 * 1024)
    );
}

#[test]
fn test_ntriples_streaming_no_memory_leak() {
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Parse large N-Triples document multiple times
    for _ in 0..5 {
        let ntriples = generate_large_ntriples(2000); // 2000 triples per iteration
        let parser = NTriplesParser::new();
        let cursor = Cursor::new(ntriples);

        let triples: Result<Vec<_>, _> = parser.for_reader(cursor).collect();
        assert!(triples.is_ok());
        drop(triples);
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    let max_acceptable_growth = 10 * 1024 * 1024;

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected in N-Triples: grew by {} MB",
        memory_growth / (1024 * 1024)
    );
}

#[test]
fn test_interrupted_streaming_no_leak() {
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Simulate interrupted streaming (process only first batch)
    for _ in 0..10 {
        let turtle = generate_large_turtle(1000);
        let config = StreamingConfig::default().with_batch_size(50);
        let cursor = Cursor::new(turtle);
        let streaming = StreamingParser::with_config(cursor, config);

        let mut batches = streaming.batches();

        // Only process first batch, then drop iterator
        if let Some(batch) = batches.next() {
            assert!(batch.is_ok());
            drop(batch);
        }

        // Drop the rest without consuming
        drop(batches);
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    let max_acceptable_growth = 10 * 1024 * 1024;

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected in interrupted streaming: grew by {} MB",
        memory_growth / (1024 * 1024)
    );
}

#[test]
fn test_repeated_small_parses_no_leak() {
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Parse many small documents (simulates server handling many requests)
    for _ in 0..1000 {
        let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate ex:object .
        "#;

        let parser = TurtleParser::new();
        let result = parser.parse_document(turtle);
        assert!(result.is_ok());
        drop(result);
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    let max_acceptable_growth = 10 * 1024 * 1024; // Account for OS memory overhead

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected in repeated small parses: grew by {} MB",
        memory_growth / (1024 * 1024)
    );
}

#[test]
fn test_error_recovery_no_leak() {
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Parse many invalid documents (lenient mode)
    for i in 0..500 {
        let invalid_turtle = format!(
            r#"
@prefix ex: <http://example.org/> .
ex:subject{} ex:predicate "invalid literal
        "#,
            i
        );

        let parser = TurtleParser::new_lenient();
        let _ = parser.parse_document(&invalid_turtle); // Expect errors
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    let max_acceptable_growth = 10 * 1024 * 1024; // Account for OS memory overhead

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected in error recovery: grew by {} MB",
        memory_growth / (1024 * 1024)
    );
}

#[test]
fn test_buffer_pool_reuse() {
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Use the same parser instance to test buffer reuse
    let parser = TurtleParser::new();

    for _ in 0..100 {
        let turtle = generate_large_turtle(100); // 300 triples per iteration
        let result = parser.parse_document(&turtle);
        assert!(result.is_ok());
        drop(result);
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    let max_acceptable_growth = 10 * 1024 * 1024; // Account for OS memory overhead

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected in buffer pool reuse: grew by {} MB",
        memory_growth / (1024 * 1024)
    );
}

#[test]
fn test_trig_streaming_no_leak() {
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Generate large TriG document with multiple named graphs
    let generate_trig = |num_graphs: usize, triples_per_graph: usize| {
        let mut doc = String::from("@prefix ex: <http://example.org/> .\n\n");
        for g in 0..num_graphs {
            doc.push_str(&format!("ex:graph{} {{\n", g));
            for t in 0..triples_per_graph {
                doc.push_str(&format!(
                    "    ex:subject{} ex:predicate \"value{}\" .\n",
                    t, t
                ));
            }
            doc.push_str("}\n\n");
        }
        doc
    };

    for _ in 0..5 {
        let trig = generate_trig(10, 100); // 10 graphs * 100 triples = 1000 quads
        let parser = TriGParser::new();
        let cursor = Cursor::new(trig);

        let quads: Result<Vec<_>, _> = parser.for_reader(cursor).collect();
        assert!(quads.is_ok());
        drop(quads);
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    let max_acceptable_growth = 10 * 1024 * 1024;

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected in TriG streaming: grew by {} MB",
        memory_growth / (1024 * 1024)
    );
}

#[test]
fn test_nquads_streaming_no_leak() {
    if get_memory_usage() == 0 {
        eprintln!("Memory measurement not available on this platform, skipping test");
        return;
    }

    std::hint::black_box(vec![0u8; 1024 * 1024]);
    drop(vec![0u8; 1024 * 1024]);

    let initial_memory = get_memory_usage();

    // Generate large N-Quads document
    let generate_nquads = |num_quads: usize| {
        let mut doc = String::new();
        for i in 0..num_quads {
            doc.push_str(&format!(
                "<http://example.org/s{}> <http://example.org/p> \"value{}\" <http://example.org/g{}> .\n",
                i, i, i % 10
            ));
        }
        doc
    };

    for _ in 0..5 {
        let nquads = generate_nquads(2000);
        let parser = NQuadsParser::new();
        let cursor = Cursor::new(nquads);

        let quads: Result<Vec<_>, _> = parser.for_reader(cursor).collect();
        assert!(quads.is_ok());
        drop(quads);
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let final_memory = get_memory_usage();
    let memory_growth = final_memory.saturating_sub(initial_memory);
    let max_acceptable_growth = 10 * 1024 * 1024;

    assert!(
        memory_growth < max_acceptable_growth,
        "Memory leak detected in N-Quads streaming: grew by {} MB",
        memory_growth / (1024 * 1024)
    );
}
