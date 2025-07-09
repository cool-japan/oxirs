//! Comprehensive tests for memory-mapped RDF store

use anyhow::Result;
use oxirs_core::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject,
};
use oxirs_core::store::MmapStore;
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

#[test]
fn test_create_and_open_store() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path();

    // Create new store
    {
        let store = MmapStore::new(path)?;
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    // Open existing store
    {
        let store = MmapStore::open(path)?;
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    Ok(())
}

#[test]
fn test_add_and_persist_quads() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path();

    // Add quads and verify persistence using batch API for better performance
    {
        let store = MmapStore::new(path)?;

        // Create all quads first, then add them in a batch
        let mut quads = Vec::with_capacity(50);
        for i in 0..50 {
            let quad = Quad::new(
                Subject::NamedNode(NamedNode::new(&format!(
                    "http://example.org/subject/{}",
                    i
                ))?),
                Predicate::NamedNode(NamedNode::new("http://example.org/predicate")?),
                Object::Literal(Literal::new_simple_literal(&format!("value {}", i))),
                GraphName::DefaultGraph,
            );
            quads.push(quad);
        }

        // Use batch API for much better performance
        store.add_batch(&quads)?;
        store.flush()?;
        assert_eq!(store.len(), 50);
    }

    // Reopen and verify data
    {
        let store = MmapStore::open(path)?;
        assert_eq!(store.len(), 50);
    }

    Ok(())
}

#[test]
fn test_large_dataset() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    // Generate 1,000 quads and add them in batches for better performance
    let mut quads = Vec::with_capacity(100);

    for i in 0..1_000 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(&format!("http://example.org/s/{}", i))?),
            Predicate::NamedNode(NamedNode::new(&format!("http://example.org/p/{}", i % 10))?),
            if i % 2 == 0 {
                Object::Literal(Literal::new_simple_literal(&format!("literal {}", i)))
            } else {
                Object::NamedNode(NamedNode::new(&format!("http://example.org/o/{}", i))?)
            },
            if i % 3 == 0 {
                GraphName::NamedNode(NamedNode::new(&format!("http://example.org/g/{}", i % 5))?)
            } else {
                GraphName::DefaultGraph
            },
        );
        quads.push(quad);

        // Process in batches of 1000 for optimal performance
        if quads.len() >= 1000 {
            store.add_batch(&quads)?;
            quads.clear();
        }
    }

    // Add any remaining quads
    if !quads.is_empty() {
        store.add_batch(&quads)?;
    }

    store.flush()?;
    assert_eq!(store.len(), 1_000);

    // Verify stats
    let stats = store.stats();
    assert_eq!(stats.quad_count, 1_000);
    assert!(stats.data_size > 0);

    Ok(())
}

#[test]
fn test_blank_nodes() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    // Add quads with blank nodes using batch API for better performance
    let mut quads = Vec::with_capacity(50);
    for i in 0..50 {
        let quad = Quad::new(
            Subject::BlankNode(BlankNode::new(&format!("b{}", i))?),
            Predicate::NamedNode(NamedNode::new("http://example.org/hasValue")?),
            Object::BlankNode(BlankNode::new(&format!("b{}", i + 50))?),
            GraphName::DefaultGraph,
        );
        quads.push(quad);
    }

    store.add_batch(&quads)?;
    store.flush()?;
    assert_eq!(store.len(), 50);

    Ok(())
}

#[test]
fn test_literal_types() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    // Test different literal types
    let quads = vec![
        // Simple literal
        Quad::new(
            Subject::NamedNode(NamedNode::new("http://example.org/s1")?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::Literal(Literal::new_simple_literal("simple")),
            GraphName::DefaultGraph,
        ),
        // Language-tagged literal
        Quad::new(
            Subject::NamedNode(NamedNode::new("http://example.org/s2")?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::Literal(Literal::new_language_tagged_literal("hello", "en")?),
            GraphName::DefaultGraph,
        ),
        // Typed literal
        Quad::new(
            Subject::NamedNode(NamedNode::new("http://example.org/s3")?),
            Predicate::NamedNode(NamedNode::new("http://example.org/p")?),
            Object::Literal(Literal::new_typed(
                "42",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
            )),
            GraphName::DefaultGraph,
        ),
    ];

    store.add_batch(&quads)?;

    store.flush()?;
    assert_eq!(store.len(), 3);

    Ok(())
}

#[test]
fn test_named_graphs() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    // Add quads to different graphs using batch API
    let mut quads = Vec::with_capacity(100);
    let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
    for i in 0..100 {
        let graph_name = if i % 4 == 0 {
            GraphName::DefaultGraph
        } else {
            GraphName::NamedNode(NamedNode::new(&format!(
                "http://example.org/graph/{}",
                i % 4
            ))?)
        };

        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(&format!("http://example.org/s/{i}"))?),
            predicate.clone(),
            Object::Literal(Literal::new_simple_literal(&format!("{i}"))),
            graph_name,
        );
        quads.push(quad);
    }

    store.add_batch(&quads)?;
    store.flush()?;
    assert_eq!(store.len(), 100);

    Ok(())
}

#[test]
fn test_concurrent_reads() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = Arc::new(MmapStore::new(temp_dir.path())?);

    // Add some data using batch API for better performance
    let mut quads = Vec::with_capacity(1000);
    let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
    for i in 0..1000 {
        let quad = Quad::new_default_graph(
            Subject::NamedNode(NamedNode::new(&format!("http://example.org/s/{i}"))?),
            predicate.clone(),
            Object::Literal(Literal::new_simple_literal(&format!("{i}"))),
        );
        quads.push(quad);
    }
    store.add_batch(&quads)?;
    store.flush()?;

    // Concurrent reads
    let barrier = Arc::new(Barrier::new(10));
    let mut handles = vec![];

    for _ in 0..10 {
        let store_clone = Arc::clone(&store);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            // Each thread reads the store
            let count = store_clone.len();
            assert_eq!(count, 1000);

            // Get stats
            let stats = store_clone.stats();
            assert_eq!(stats.quad_count, 1000);
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

#[test]
fn test_append_only_safety() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    // Add initial data using batch API
    let mut initial_quads = Vec::with_capacity(50);
    let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
    for i in 0..50 {
        let quad = Quad::new_default_graph(
            Subject::NamedNode(NamedNode::new(&format!("http://example.org/s/{i}"))?),
            predicate.clone(),
            Object::Literal(Literal::new_simple_literal(&format!("{i}"))),
        );
        initial_quads.push(quad);
    }
    store.add_batch(&initial_quads)?;

    let initial_count = store.len();
    store.flush()?;

    // Add more data using batch API
    let mut additional_quads = Vec::with_capacity(50);
    for i in 50..100 {
        let quad = Quad::new_default_graph(
            Subject::NamedNode(NamedNode::new(&format!("http://example.org/s/{i}"))?),
            predicate.clone(),
            Object::Literal(Literal::new_simple_literal(&format!("{i}"))),
        );
        additional_quads.push(quad);
    }
    store.add_batch(&additional_quads)?;

    store.flush()?;

    // Verify append-only behavior
    assert_eq!(store.len(), initial_count + 50);
    assert_eq!(store.len(), 100);

    Ok(())
}

#[test]
fn test_very_large_dataset() -> Result<()> {
    if std::env::var("RUN_LARGE_TESTS").is_err() {
        eprintln!("Skipping large dataset test. Set RUN_LARGE_TESTS=1 to run.");
        return Ok(());
    }

    let temp_dir = TempDir::new()?;
    let store = MmapStore::new(temp_dir.path())?;

    // Add 1 million quads using efficient batching
    let batch_size = 10_000;
    for batch_start in (0..1_000_000).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size, 1_000_000);
        let mut quads = Vec::with_capacity(batch_end - batch_start);

        for i in batch_start..batch_end {
            let quad = Quad::new_default_graph(
                Subject::NamedNode(NamedNode::new(&format!("http://example.org/entity/{}", i))?),
                Predicate::NamedNode(NamedNode::new(&format!(
                    "http://example.org/property/{}",
                    i % 100
                ))?),
                Object::Literal(Literal::new_typed(
                    &format!("{}", i),
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
                )),
            );
            quads.push(quad);
        }

        store.add_batch(&quads)?;
        store.flush()?;
        println!("Progress: {} quads", batch_end);
    }

    store.flush()?;
    assert_eq!(store.len(), 1_000_000);

    let stats = store.stats();
    println!("Store stats: {:?}", stats);

    Ok(())
}

#[test]
fn test_recovery_after_crash() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = temp_dir.path();

    // Simulate adding data and partial flush
    {
        let store = MmapStore::new(path)?;

        // Add some data and flush using batch API
        let mut initial_quads = Vec::with_capacity(50);
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/p")?);
        for i in 0..50 {
            let quad = Quad::new_default_graph(
                Subject::NamedNode(NamedNode::new(&format!("http://example.org/s/{i}"))?),
                predicate.clone(),
                Object::Literal(Literal::new_simple_literal(&format!("{i}"))),
            );
            initial_quads.push(quad);
        }
        store.add_batch(&initial_quads)?;
        store.flush()?;

        // Add more data but don't flush (simulate crash) using batch API
        let mut crash_quads = Vec::with_capacity(10);
        for i in 50..60 {
            let quad = Quad::new_default_graph(
                Subject::NamedNode(NamedNode::new(&format!("http://example.org/s/{i}"))?),
                predicate.clone(),
                Object::Literal(Literal::new_simple_literal(&format!("{i}"))),
            );
            crash_quads.push(quad);
        }
        store.add_batch(&crash_quads)?;
        // Drop without flushing
    }

    // Reopen and verify only flushed data is present
    {
        let store = MmapStore::open(path)?;
        assert_eq!(store.len(), 50); // Only flushed data should be present
    }

    Ok(())
}
