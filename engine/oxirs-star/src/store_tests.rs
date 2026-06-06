//! Tests for the RDF-star store, kept as a sibling of `store.rs`.
//!
//! Compiled only under `#[cfg(test)]`. Hosted as a sibling file rather than an
//! inline module so that the production code in `store_core.rs` /
//! `store_query.rs` / `store_indexing.rs` stays under the 2000-line limit.

#![cfg(test)]

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::store::StarStore;
use crate::StarResult;

#[test]
fn test_store_creation() -> StarResult<()> {
    let store = StarStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    Ok(())
}

#[test]
fn test_basic_operations() -> StarResult<()> {
    let store = StarStore::new();

    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/knows")?,
        StarTerm::iri("http://example.org/bob")?,
    );

    // Insert
    store.insert(&triple)?;
    assert_eq!(store.len(), 1);
    assert!(store.contains(&triple));

    // Query
    let results = store.query_triples(
        Some(&StarTerm::iri("http://example.org/alice")?),
        None,
        None,
    )?;
    assert_eq!(results.len(), 1);

    // Remove
    assert!(store.remove(&triple)?);
    assert!(store.is_empty());
    Ok(())
}

#[test]
fn test_quoted_triple_operations() -> StarResult<()> {
    let store = StarStore::new();

    // Create a quoted triple
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/age")?,
        StarTerm::literal("25")?,
    );

    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("0.9")?,
    );

    store.insert(&outer)?;
    assert_eq!(store.len(), 1);

    // Find triples containing the quoted triple
    let containing = store.find_triples_containing_quoted(&inner);
    assert_eq!(containing.len(), 1);
    assert_eq!(containing[0], outer);
    Ok(())
}

#[test]
fn test_store_statistics() -> StarResult<()> {
    let store = StarStore::new();

    let regular = StarTriple::new(
        StarTerm::iri("http://example.org/s")?,
        StarTerm::iri("http://example.org/p")?,
        StarTerm::iri("http://example.org/o")?,
    );

    let quoted = StarTriple::new(
        StarTerm::quoted_triple(regular.clone()),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("high")?,
    );

    store.insert(&regular)?;
    store.insert(&quoted)?;

    let stats = store.statistics();
    assert_eq!(stats.quoted_triples_count, 1);
    assert_eq!(stats.max_nesting_encountered, 1);
    Ok(())
}

#[test]
fn test_btree_indexing_performance() -> StarResult<()> {
    let store = StarStore::new();

    // Create multiple quoted triples with different patterns
    let base_triple = StarTriple::new(
        StarTerm::iri("http://example.org/alice")?,
        StarTerm::iri("http://example.org/age")?,
        StarTerm::literal("25")?,
    );

    let quoted1 = StarTriple::new(
        StarTerm::quoted_triple(base_triple.clone()),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("0.9")?,
    );

    let quoted2 = StarTriple::new(
        StarTerm::iri("http://example.org/bob")?,
        StarTerm::iri("http://example.org/believes")?,
        StarTerm::quoted_triple(base_triple.clone()),
    );

    store.insert(&quoted1)?;
    store.insert(&quoted2)?;

    // Test pattern-based queries using the new B-tree indices
    let results = store.find_triples_by_quoted_pattern(
        Some(&StarTerm::iri("http://example.org/alice")?),
        None,
        None,
    );
    assert_eq!(results.len(), 2);

    // Test nesting depth queries
    let shallow_results = store.find_triples_by_nesting_depth(0, Some(0));
    assert_eq!(shallow_results.len(), 0); // No triples with depth 0

    let depth_1_results = store.find_triples_by_nesting_depth(1, Some(1));
    assert_eq!(depth_1_results.len(), 2); // Both quoted triples have depth 1
    Ok(())
}

#[test]
fn test_graph_import_export() -> StarResult<()> {
    let store = StarStore::new();
    let mut graph = StarGraph::new();

    let triple = StarTriple::new(
        StarTerm::iri("http://example.org/s")?,
        StarTerm::iri("http://example.org/p")?,
        StarTerm::iri("http://example.org/o")?,
    );

    graph.insert(triple.clone())?;
    store.from_graph(&graph)?;

    assert_eq!(store.len(), 1);
    assert!(store.contains(&triple));

    let exported = store.to_graph();
    assert_eq!(exported.len(), 1);
    assert!(exported.contains(&triple));
    Ok(())
}

#[test]
fn test_streaming_iterator() -> StarResult<()> {
    let store = StarStore::new();

    // Insert multiple triples
    for i in 0..100 {
        let triple = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{i}"))?,
            StarTerm::iri("http://example.org/p")?,
            StarTerm::iri(&format!("http://example.org/o{i}"))?,
        );
        store.insert(&triple)?;
    }

    // Test streaming iterator with different chunk sizes
    let chunk_sizes = vec![1, 10, 50, 100, 200];

    for chunk_size in chunk_sizes {
        let mut count = 0;
        for _triple in store.streaming_iter(chunk_size) {
            count += 1;
        }
        assert_eq!(
            count, 100,
            "Streaming iterator with chunk size {chunk_size} should return all triples"
        );
    }

    // Test that streaming iterator returns the same triples as regular iterator
    let regular_triples: Vec<_> = store.iter().collect();
    let streaming_triples: Vec<_> = store.streaming_iter(25).collect();

    assert_eq!(regular_triples.len(), streaming_triples.len());

    // Both iterators should contain the same triples (though possibly in different order)
    for triple in &regular_triples {
        assert!(streaming_triples.contains(triple));
    }
    Ok(())
}
