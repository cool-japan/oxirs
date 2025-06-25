//! # Basic OxiRS TDB Usage Example
//!
//! Demonstrates the core functionality of the oxirs-tdb storage engine,
//! including RDF triple storage, MVCC transactions, and querying.

use anyhow::Result;
use oxirs_tdb::{TdbConfig, TdbStore, Term, TripleStoreStats};

fn main() -> Result<()> {
    // Create TDB store configuration
    let config = TdbConfig {
        location: "./example_tdb".to_string(),
        cache_size: 1024 * 1024 * 50, // 50MB cache
        enable_transactions: true,
        enable_mvcc: true,
    };

    // Create TDB store
    let store = TdbStore::new(config)?;
    println!("✅ Created TDB store");

    // Create some RDF terms
    let person = Term::iri("http://example.org/person/john");
    let name_predicate = Term::iri("http://xmlns.com/foaf/0.1/name");
    let age_predicate = Term::iri("http://example.org/age");
    let type_predicate = Term::iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    let person_class = Term::iri("http://xmlns.com/foaf/0.1/Person");

    let john_name = Term::literal("John Doe");
    let john_age = Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer");

    println!("✅ Created RDF terms");

    // Insert triples without explicit transaction
    store.insert_triple(&person, &name_predicate, &john_name)?;
    store.insert_triple(&person, &age_predicate, &john_age)?;
    store.insert_triple(&person, &type_predicate, &person_class)?;

    println!("✅ Inserted triples");

    // Get statistics
    let stats = store.get_stats()?;
    println!("📊 Store statistics:");
    println!("   - Total triples: {}", stats.total_triples);
    println!("   - Insert operations: {}", stats.insert_count);
    println!(
        "   - Completed transactions: {}",
        stats.completed_transactions
    );

    // Query all triples for John
    println!("\n🔍 Querying all triples for John:");
    let results = store.query_triples(Some(&person), None, None)?;
    for (s, p, o) in &results {
        println!("   {} {} {}", s, p, o);
    }

    // Query with specific predicate
    println!("\n🔍 Querying John's name:");
    let name_results = store.query_triples(Some(&person), Some(&name_predicate), None)?;
    for (s, p, o) in &name_results {
        println!("   {} has name: {}", s, o);
    }

    // Demonstrate transaction usage
    println!("\n💼 Demonstrating transactions:");

    // Begin a transaction
    let tx = store.begin_transaction()?;
    println!("   Started transaction: {}", tx.id());

    // Insert additional data in transaction
    let hobby_predicate = Term::iri("http://example.org/hobby");
    let hobby_value = Term::literal("Programming");

    // Note: For now, transaction-level operations aren't fully integrated
    // This would be store.insert_triple_tx(&tx, &person, &hobby_predicate, &hobby_value)?;
    // For demonstration, we'll commit the empty transaction

    let version = store.commit_transaction(tx)?;
    println!("   Committed transaction at version: {}", version);

    // Add more people to demonstrate bulk operations
    println!("\n👥 Adding more people:");

    let alice = Term::iri("http://example.org/person/alice");
    let alice_name = Term::literal("Alice Smith");
    let alice_age = Term::typed_literal("28", "http://www.w3.org/2001/XMLSchema#integer");

    store.insert_triple(&alice, &name_predicate, &alice_name)?;
    store.insert_triple(&alice, &age_predicate, &alice_age)?;
    store.insert_triple(&alice, &type_predicate, &person_class)?;

    let bob = Term::iri("http://example.org/person/bob");
    let bob_name = Term::literal("Bob Johnson");
    let bob_age = Term::typed_literal("35", "http://www.w3.org/2001/XMLSchema#integer");

    store.insert_triple(&bob, &name_predicate, &bob_name)?;
    store.insert_triple(&bob, &age_predicate, &bob_age)?;
    store.insert_triple(&bob, &type_predicate, &person_class)?;

    // Query all people
    println!("\n🔍 All people in the database:");
    let people_results = store.query_triples(None, Some(&type_predicate), Some(&person_class))?;
    for (person, _, _) in &people_results {
        // Get their name
        if let Ok(name_results) = store.query_triples(Some(person), Some(&name_predicate), None) {
            if let Some((_, _, name)) = name_results.first() {
                println!("   Person: {} ({})", person, name);
            }
        }
    }

    // Final statistics
    let final_stats = store.get_stats()?;
    println!("\n📊 Final statistics:");
    println!("   - Total triples: {}", final_stats.total_triples);
    println!("   - Insert operations: {}", final_stats.insert_count);
    println!("   - Query operations: {}", final_stats.query_count);
    println!(
        "   - Completed transactions: {}",
        final_stats.completed_transactions
    );

    // Demonstrate store maintenance
    println!("\n🔧 Performing store maintenance:");
    store.compact()?;
    println!("   Store compacted successfully");

    // Test deletion
    println!("\n🗑️ Testing deletion:");
    let deleted = store.delete_triple(&bob, &name_predicate, &bob_name)?;
    if deleted {
        println!("   Deleted Bob's name triple");
    }

    let after_delete_stats = store.get_stats()?;
    println!(
        "   Triples after deletion: {}",
        after_delete_stats.total_triples
    );

    println!("\n🎉 Example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_basic_example() {
        let temp_dir = TempDir::new().unwrap();

        let config = TdbConfig {
            location: temp_dir.path().to_string_lossy().to_string(),
            cache_size: 1024 * 1024, // 1MB for test
            enable_transactions: true,
            enable_mvcc: true,
        };

        let store = TdbStore::new(config).unwrap();

        // Basic operations
        let subject = Term::iri("http://example.org/test");
        let predicate = Term::iri("http://example.org/predicate");
        let object = Term::literal("test value");

        store.insert_triple(&subject, &predicate, &object).unwrap();

        let stats = store.get_stats().unwrap();
        assert_eq!(stats.total_triples, 1);

        let results = store.query_triples(Some(&subject), None, None).unwrap();
        assert_eq!(results.len(), 1);

        let deleted = store.delete_triple(&subject, &predicate, &object).unwrap();
        assert!(deleted);

        let final_stats = store.get_stats().unwrap();
        assert_eq!(final_stats.total_triples, 0);
    }
}
