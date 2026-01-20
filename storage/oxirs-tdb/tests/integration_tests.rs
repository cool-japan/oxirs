//! Integration tests for oxirs-tdb storage engine
//! Tests the complete functionality of Phases 1-5 implementation

use oxirs_tdb::dictionary::Term;
use oxirs_tdb::error::Result;
use oxirs_tdb::{TdbConfig, TdbStore};
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn create_test_store() -> Result<(TdbStore, std::path::PathBuf)> {
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let temp_dir = env::temp_dir().join(format!("oxirs_tdb_test_{}", id));
    std::fs::create_dir_all(&temp_dir)?;
    let store = TdbStore::open(&temp_dir)?;
    Ok((store, temp_dir))
}

#[test]
fn test_basic_triple_operations() -> Result<()> {
    let (mut store, _temp_dir) = create_test_store()?;

    // Insert triples using string API
    store.insert(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/knows",
        "http://example.org/bob",
    )?;

    // Check count
    assert_eq!(store.count(), 1);
    assert_eq!(store.len()?, 1);
    assert!(!store.is_empty());

    // Check contains
    assert!(store.contains(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/knows",
        "http://example.org/bob"
    )?);

    // Delete
    let deleted = store.delete(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/knows",
        "http://example.org/bob",
    )?;
    assert!(deleted);
    assert_eq!(store.count(), 0);
    assert!(store.is_empty());

    Ok(())
}

#[test]
fn test_term_api() -> Result<()> {
    let (mut store, _temp_dir) = create_test_store()?;

    // Insert using Term API
    let alice = Term::iri("http://example.org/alice");
    let name = Term::iri("http://xmlns.com/foaf/0.1/name");
    let alice_name = Term::literal("Alice");

    store.insert_triple(&alice, &name, &alice_name)?;

    assert_eq!(store.count(), 1);

    Ok(())
}

#[test]
fn test_bulk_insert() -> Result<()> {
    let (mut store, _temp_dir) = create_test_store()?;

    // Prepare bulk triples
    let mut triples = Vec::new();
    for i in 0..10 {
        triples.push((
            Term::iri(format!("http://example.org/subject{}", i)),
            Term::iri("http://example.org/predicate"),
            Term::literal(format!("value{}", i)),
        ));
    }

    // Bulk insert
    store.insert_triples_bulk(&triples)?;

    assert_eq!(store.count(), 10);

    Ok(())
}

#[test]
fn test_bulk_insert_validation() -> Result<()> {
    let (mut store, _temp_dir) = create_test_store()?;

    // Try to insert invalid triple (literal as subject)
    let bad_triples = vec![(
        Term::literal("bad_subject"),
        Term::iri("http://example.org/pred"),
        Term::literal("value"),
    )];

    let result = store.insert_triples_bulk(&bad_triples);
    assert!(result.is_err());

    // Store should still be empty (transactional)
    assert_eq!(store.count(), 0);

    Ok(())
}

#[test]
fn test_configuration() -> Result<()> {
    let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let temp_dir = env::temp_dir().join(format!("oxirs_tdb_config_{}", id));
    std::fs::create_dir_all(&temp_dir)?;

    // Create with custom config
    let config = TdbConfig::new(&temp_dir)
        .with_buffer_pool_size(2000)
        .with_compression(true)
        .with_bloom_filters(true);

    let store = TdbStore::open_with_config(config)?;

    assert_eq!(store.config().buffer_pool_size, 2000);
    assert!(store.config().enable_compression);
    assert!(store.config().enable_bloom_filters);

    std::fs::remove_dir_all(&temp_dir).ok();

    Ok(())
}

#[test]
fn test_statistics() -> Result<()> {
    let (mut store, _temp_dir) = create_test_store()?;

    // Insert some data
    for i in 0..5 {
        store.insert(
            &format!("http://example.org/s{}", i),
            "http://example.org/p",
            &format!("http://example.org/o{}", i),
        )?;
    }

    let stats = store.stats();
    assert_eq!(stats.triple_count, 5);
    assert!(stats.dictionary_size > 0);
    // Bloom filter stats should be present if enabled
    assert!(stats.bloom_filter_stats.is_some());
    // Compression stats should be present if enabled
    assert!(stats.compression_stats.is_some());

    Ok(())
}

#[test]
fn test_transaction_manager() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Verify transaction manager is accessible
    let txn_manager = store.transaction_manager();
    let txn = txn_manager.begin()?;

    assert_eq!(txn.state(), oxirs_tdb::transaction::TxnState::Active);

    // Commit transaction
    txn.commit()?;
    assert_eq!(txn.state(), oxirs_tdb::transaction::TxnState::Committed);

    Ok(())
}
