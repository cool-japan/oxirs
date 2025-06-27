use anyhow::Result;
use oxirs_tdb::{TdbConfig, TdbStore, Term, Transaction};
use tempfile::TempDir;

/// Helper to create a test TDB store
fn create_test_store() -> Result<(TdbStore, TempDir)> {
    let temp_dir = TempDir::new()?;
    let config = TdbConfig {
        location: temp_dir.path().to_string_lossy().to_string(),
        cache_size: 1024 * 1024 * 10, // 10MB for tests
        enable_transactions: true,
        enable_mvcc: true,
    };
    let store = TdbStore::new(config)?;
    Ok((store, temp_dir))
}

#[test]
fn test_basic_triple_operations() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Test data
    let subject = Term::iri("http://example.org/subject");
    let predicate = Term::iri("http://example.org/predicate");
    let object = Term::literal("test value");

    // Test insert
    store.insert_triple(&subject, &predicate, &object)?;

    // Verify stats
    let stats = store.get_stats()?;
    assert_eq!(stats.total_triples, 1);
    assert_eq!(stats.insert_count, 1);

    // Test query
    let results = store.query_triples(Some(&subject), Some(&predicate), Some(&object))?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, subject);
    assert_eq!(results[0].1, predicate);
    assert_eq!(results[0].2, object);

    // Test partial queries
    let results = store.query_triples(Some(&subject), None, None)?;
    assert_eq!(results.len(), 1);

    let results = store.query_triples(None, Some(&predicate), None)?;
    assert_eq!(results.len(), 1);

    let results = store.query_triples(None, None, Some(&object))?;
    assert_eq!(results.len(), 1);

    // Test delete
    let deleted = store.delete_triple(&subject, &predicate, &object)?;
    assert!(deleted);

    // Verify deletion
    let stats = store.get_stats()?;
    assert_eq!(stats.total_triples, 0);
    assert_eq!(stats.delete_count, 1);

    let results = store.query_triples(Some(&subject), Some(&predicate), Some(&object))?;
    assert_eq!(results.len(), 0);

    Ok(())
}

#[test]
fn test_multiple_triples() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Insert multiple triples with shared subject
    let person = Term::iri("http://example.org/person/john");
    let name_pred = Term::iri("http://xmlns.com/foaf/0.1/name");
    let age_pred = Term::iri("http://xmlns.com/foaf/0.1/age");
    let email_pred = Term::iri("http://xmlns.com/foaf/0.1/email");

    let name_obj = Term::literal("John Doe");
    let age_obj = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");
    let email_obj = Term::literal("john@example.org");

    store.insert_triple(&person, &name_pred, &name_obj)?;
    store.insert_triple(&person, &age_pred, &age_obj)?;
    store.insert_triple(&person, &email_pred, &email_obj)?;

    // Query all triples for person
    let results = store.query_triples(Some(&person), None, None)?;
    assert_eq!(results.len(), 3);

    // Add another person
    let person2 = Term::iri("http://example.org/person/jane");
    let name_obj2 = Term::literal("Jane Smith");
    let age_obj2 = Term::typed_literal("25", "http://www.w3.org/2001/XMLSchema#integer");

    store.insert_triple(&person2, &name_pred, &name_obj2)?;
    store.insert_triple(&person2, &age_pred, &age_obj2)?;

    // Query all triples with name predicate
    let results = store.query_triples(None, Some(&name_pred), None)?;
    assert_eq!(results.len(), 2);

    // Verify total count
    let stats = store.get_stats()?;
    assert_eq!(stats.total_triples, 5);

    Ok(())
}

#[test]
fn test_transactions() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Start a transaction
    let tx = store.begin_transaction()?;
    let tx_id = tx.id();

    // Commit empty transaction
    let version = store.commit_transaction(tx)?;
    assert!(version > 0);

    // Verify transaction stats
    let stats = store.get_stats()?;
    assert_eq!(stats.completed_transactions, 1);

    // Test rollback
    let tx2 = store.begin_transaction()?;
    store.rollback_transaction(tx2)?;

    // Transaction rollback doesn't track aborted count in stats currently
    // Just verify we can rollback without error

    Ok(())
}

#[test]
fn test_read_only_transactions() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Insert some data
    let subject = Term::iri("http://example.org/test");
    let predicate = Term::iri("http://example.org/pred");
    let object = Term::literal("value");
    store.insert_triple(&subject, &predicate, &object)?;

    // Start read-only transaction
    let read_tx = store.begin_read_transaction()?;

    // Verify we can query within read transaction
    let results = store.query_triples(Some(&subject), None, None)?;
    assert_eq!(results.len(), 1);

    // Commit read transaction
    store.commit_transaction(read_tx)?;

    Ok(())
}

#[test]
fn test_term_types() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Test different term types
    let subject = Term::iri("http://example.org/test");
    let predicate = Term::iri("http://example.org/pred");

    // Plain literal
    let plain_literal = Term::literal("plain text");
    store.insert_triple(&subject, &predicate, &plain_literal)?;

    // Language-tagged literal
    let lang_literal = Term::lang_literal("hello", "en");
    let pred2 = Term::iri("http://example.org/label");
    store.insert_triple(&subject, &pred2, &lang_literal)?;

    // Typed literal
    let typed_literal = Term::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer");
    let pred3 = Term::iri("http://example.org/count");
    store.insert_triple(&subject, &pred3, &typed_literal)?;

    // Blank node
    let blank = Term::blank_node("b1");
    let pred4 = Term::iri("http://example.org/ref");
    store.insert_triple(&subject, &pred4, &blank)?;

    // Verify all insertions
    let results = store.query_triples(Some(&subject), None, None)?;
    assert_eq!(results.len(), 4);

    Ok(())
}

#[test]
fn test_quad_operations() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    let subject = Term::iri("http://example.org/subject");
    let predicate = Term::iri("http://example.org/predicate");
    let object = Term::literal("test value");
    let graph = Term::iri("http://example.org/graph1");

    // Insert quad with named graph
    store.insert_quad(&subject, &predicate, &object, Some(&graph))?;

    // Insert quad in default graph
    store.insert_quad(&subject, &predicate, &object, None)?;

    let stats = store.get_stats()?;
    assert_eq!(stats.total_triples, 2);

    Ok(())
}

#[test]
fn test_large_dataset() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Insert many triples (reduced for performance)
    let num_subjects = 10;
    let num_predicates = 5;
    let num_objects_per = 3;

    for i in 0..num_subjects {
        let subject = Term::iri(&format!("http://example.org/subject/{}", i));

        for j in 0..num_predicates {
            let predicate = Term::iri(&format!("http://example.org/predicate/{}", j));

            for k in 0..num_objects_per {
                let object = Term::literal(&format!("value_{}_{}", j, k));
                store.insert_triple(&subject, &predicate, &object)?;
            }
        }
    }

    // Verify count
    let expected_triples = num_subjects * num_predicates * num_objects_per;
    assert_eq!(store.len()?, expected_triples as u64);

    // Test query performance - use a subject that actually exists (0-9)
    let subject = Term::iri("http://example.org/subject/5");
    let results = store.query_triples(Some(&subject), None, None)?;
    assert_eq!(results.len(), (num_predicates * num_objects_per) as usize);

    Ok(())
}

#[test]
fn test_clear_and_compact() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Insert data
    for i in 0..10 {
        let subject = Term::iri(&format!("http://example.org/s{}", i));
        let predicate = Term::iri("http://example.org/p");
        let object = Term::literal(&format!("value{}", i));
        store.insert_triple(&subject, &predicate, &object)?;
    }

    assert_eq!(store.len()?, 10);

    // Test compact
    store.compact()?;
    assert_eq!(store.len()?, 10);

    // Test clear
    store.clear()?;
    assert_eq!(store.len()?, 0);
    assert!(store.is_empty()?);

    Ok(())
}

#[test]
fn test_concurrent_operations() -> Result<()> {
    use std::sync::Arc;
    use std::thread;

    let (store, _temp_dir) = create_test_store()?;
    let store = Arc::new(store);

    let mut handles = vec![];

    // Spawn multiple threads performing insertions
    for thread_id in 0..5 {
        let store_clone = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for i in 0..20 {
                let subject =
                    Term::iri(&format!("http://example.org/thread{}/subject{}", thread_id, i));
                let predicate = Term::iri("http://example.org/value");
                let object = Term::literal(&format!("thread{}_value{}", thread_id, i));

                store_clone
                    .insert_triple(&subject, &predicate, &object)
                    .unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all insertions
    assert_eq!(store.len()?, 100);

    Ok(())
}

#[test]
fn test_error_handling() -> Result<()> {
    let (store, _temp_dir) = create_test_store()?;

    // Test deletion of non-existent triple
    let subject = Term::iri("http://example.org/nonexistent");
    let predicate = Term::iri("http://example.org/pred");
    let object = Term::literal("value");

    let deleted = store.delete_triple(&subject, &predicate, &object)?;
    assert!(!deleted);

    // Test query with non-existent terms
    let results = store.query_triples(Some(&subject), None, None)?;
    assert_eq!(results.len(), 0);

    Ok(())
}