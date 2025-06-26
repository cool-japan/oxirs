//! Example demonstrating usage of MmapStore with large datasets
//!
//! This example shows how to:
//! - Create a memory-mapped RDF store
//! - Load a large dataset that doesn't fit in memory
//! - Perform efficient queries without loading the entire dataset
//! - Monitor memory usage and performance

use oxirs_core::store::{MmapStore, StoreStats};
use oxirs_core::model::{Subject, Predicate, Object, Quad, GraphName, NamedNode, Literal};
use anyhow::Result;
use std::time::{Instant, Duration};
use std::path::Path;

fn main() -> Result<()> {
    // Create a store in a temporary directory
    let store_path = Path::new("/tmp/oxirs_large_dataset");
    std::fs::create_dir_all(&store_path)?;
    
    println!("Creating memory-mapped RDF store at: {:?}", store_path);
    println!("This example will generate and query a large RDF dataset.");
    println!();
    
    // Create or open the store
    let store = if store_path.join("data.oxirs").exists() {
        println!("Opening existing store...");
        MmapStore::open(store_path)?
    } else {
        println!("Creating new store...");
        MmapStore::new(store_path)?
    };
    
    // Check if we need to populate the store
    if store.is_empty() {
        println!("Populating store with sample data...");
        populate_large_dataset(&store)?;
    } else {
        println!("Store already contains {} quads", store.len());
    }
    
    // Display store statistics
    print_store_stats(&store);
    
    // Demonstrate various query patterns
    demonstrate_queries(&store)?;
    
    // Demonstrate memory efficiency
    demonstrate_memory_efficiency(&store)?;
    
    println!("\nExample completed successfully!");
    println!("Store remains on disk at: {:?}", store_path);
    println!("You can re-run this example to see persistence in action.");
    
    Ok(())
}

/// Populate the store with a large dataset
fn populate_large_dataset(store: &MmapStore) -> Result<()> {
    let start = Instant::now();
    
    // Configuration
    const NUM_ENTITIES: usize = 10_000;
    const PROPERTIES_PER_ENTITY: usize = 20;
    const BATCH_SIZE: usize = 1000;
    
    println!("Generating {} entities with {} properties each...", NUM_ENTITIES, PROPERTIES_PER_ENTITY);
    println!("Total quads to generate: {}", NUM_ENTITIES * PROPERTIES_PER_ENTITY);
    
    let mut batch_count = 0;
    let mut total_quads = 0;
    
    // Common predicates
    let predicates = vec![
        NamedNode::new("http://example.org/name")?,
        NamedNode::new("http://example.org/age")?,
        NamedNode::new("http://example.org/email")?,
        NamedNode::new("http://example.org/phone")?,
        NamedNode::new("http://example.org/address")?,
        NamedNode::new("http://example.org/city")?,
        NamedNode::new("http://example.org/country")?,
        NamedNode::new("http://example.org/occupation")?,
        NamedNode::new("http://example.org/company")?,
        NamedNode::new("http://example.org/department")?,
        NamedNode::new("http://example.org/salary")?,
        NamedNode::new("http://example.org/startDate")?,
        NamedNode::new("http://example.org/manager")?,
        NamedNode::new("http://example.org/project")?,
        NamedNode::new("http://example.org/skill")?,
        NamedNode::new("http://example.org/certification")?,
        NamedNode::new("http://example.org/education")?,
        NamedNode::new("http://example.org/language")?,
        NamedNode::new("http://example.org/hobby")?,
        NamedNode::new("http://example.org/status")?,
    ];
    
    // Generate data in batches
    for entity_id in 0..NUM_ENTITIES {
        let subject = Subject::NamedNode(NamedNode::new(&format!("http://example.org/entity/{}", entity_id))?);
        
        // Add properties for this entity
        for (prop_idx, predicate) in predicates.iter().enumerate() {
            let object = match prop_idx {
                0 => Object::Literal(Literal::new_simple_literal(&format!("Entity {}", entity_id))),
                1 => Object::Literal(Literal::new_typed(&format!("{}", 20 + (entity_id % 50)), 
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?)),
                2 => Object::Literal(Literal::new_simple_literal(&format!("entity{}@example.org", entity_id))),
                10 => Object::Literal(Literal::new_typed(&format!("{}", 50000 + (entity_id * 1000)), 
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")?)),
                11 => Object::Literal(Literal::new_typed("2024-01-01", 
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#date")?)),
                12 => Object::NamedNode(NamedNode::new(&format!("http://example.org/entity/{}", entity_id / 10))?),
                17 => Object::Literal(Literal::new_language_tagged_literal("English", "en")?),
                _ => Object::Literal(Literal::new_simple_literal(&format!("Value {}_{}", entity_id, prop_idx))),
            };
            
            // Distribute across different graphs
            let graph = if entity_id % 100 == 0 {
                GraphName::DefaultGraph
            } else {
                GraphName::NamedNode(NamedNode::new(&format!("http://example.org/graph/{}", entity_id % 10))?)
            };
            
            let quad = Quad::new(
                subject.clone(),
                Predicate::NamedNode(predicate.clone()),
                object,
                graph,
            );
            
            store.add(&quad)?;
            total_quads += 1;
        }
        
        // Flush periodically
        if (entity_id + 1) % BATCH_SIZE == 0 {
            store.flush()?;
            batch_count += 1;
            
            let elapsed = start.elapsed();
            let rate = total_quads as f64 / elapsed.as_secs_f64();
            println!("  Processed {} entities ({} quads) - {:.0} quads/sec", 
                entity_id + 1, total_quads, rate);
        }
    }
    
    // Final flush
    store.flush()?;
    
    let elapsed = start.elapsed();
    println!("\nDataset generation completed in {:.2} seconds", elapsed.as_secs_f64());
    println!("Average rate: {:.0} quads/second", total_quads as f64 / elapsed.as_secs_f64());
    
    Ok(())
}

/// Print store statistics
fn print_store_stats(store: &MmapStore) {
    println!("\n=== Store Statistics ===");
    let stats = store.stats();
    println!("Total quads: {}", stats.quad_count);
    println!("Unique terms: {}", stats.term_count);
    println!("Data size: {} MB", stats.data_size / (1024 * 1024));
    println!("Index size: {} MB", stats.index_size / (1024 * 1024));
    println!("Term dictionary size: {} MB", stats.term_size / (1024 * 1024));
}

/// Demonstrate various query patterns
fn demonstrate_queries(store: &MmapStore) -> Result<()> {
    println!("\n=== Query Demonstrations ===");
    
    // Query 1: Find all properties of a specific entity
    {
        println!("\n1. Finding all properties of entity 1000:");
        let start = Instant::now();
        
        let subject = Subject::NamedNode(NamedNode::new("http://example.org/entity/1000")?);
        let quads: Vec<_> = store.quads_matching(
            Some(&subject),
            None,
            None,
            None,
        )?.collect::<Result<Vec<_>>>()?;
        
        let elapsed = start.elapsed();
        println!("  Found {} quads in {:?}", quads.len(), elapsed);
        
        // Show first few results
        for (i, quad) in quads.iter().take(5).enumerate() {
            println!("  [{}] {} -> {}", i, quad.predicate(), quad.object());
        }
        if quads.len() > 5 {
            println!("  ... and {} more", quads.len() - 5);
        }
    }
    
    // Query 2: Find all entities with a specific age
    {
        println!("\n2. Finding all entities aged 30:");
        let start = Instant::now();
        
        let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/age")?);
        let object = Object::Literal(Literal::new_typed("30", 
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?));
        
        let quads: Vec<_> = store.quads_matching(
            None,
            Some(&predicate),
            Some(&object),
            None,
        )?.collect::<Result<Vec<_>>>()?;
        
        let elapsed = start.elapsed();
        println!("  Found {} entities in {:?}", quads.len(), elapsed);
    }
    
    // Query 3: Count quads in a specific graph
    {
        println!("\n3. Counting quads in graph 5:");
        let start = Instant::now();
        
        let graph = GraphName::NamedNode(NamedNode::new("http://example.org/graph/5")?);
        let count = store.quads_matching(
            None,
            None,
            None,
            Some(&graph),
        )?.count();
        
        let elapsed = start.elapsed();
        println!("  Graph contains {} quads (counted in {:?})", count, elapsed);
    }
    
    // Query 4: Find managers and their direct reports
    {
        println!("\n4. Finding entities and their managers:");
        let start = Instant::now();
        
        let manager_pred = Predicate::NamedNode(NamedNode::new("http://example.org/manager")?);
        let mut manager_count = 0;
        
        // Find first 10 manager relationships
        for quad in store.quads_matching(None, Some(&manager_pred), None, None)?.take(10) {
            let quad = quad?;
            manager_count += 1;
            println!("  {} reports to {}", quad.subject(), quad.object());
        }
        
        let elapsed = start.elapsed();
        println!("  Found {} manager relationships (sampled 10) in {:?}", manager_count, elapsed);
    }
    
    Ok(())
}

/// Demonstrate memory efficiency
fn demonstrate_memory_efficiency(store: &MmapStore) -> Result<()> {
    println!("\n=== Memory Efficiency Demonstration ===");
    
    // Get current process memory usage (approximation)
    let stats = store.stats();
    let total_data = stats.data_size + stats.index_size + stats.term_size;
    
    println!("Total data on disk: {} MB", total_data / (1024 * 1024));
    println!("This data is accessed via memory mapping without loading it all into RAM.");
    
    // Demonstrate random access
    println!("\nPerforming 1000 random queries...");
    let start = Instant::now();
    
    for i in 0..1000 {
        let entity_id = i * 10;
        let subject = Subject::NamedNode(NamedNode::new(&format!("http://example.org/entity/{}", entity_id))?);
        
        // Just count the results
        let _count = store.quads_matching(Some(&subject), None, None, None)?.count();
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed.as_micros() as f64 / 1000.0;
    println!("Completed 1000 random queries in {:?}", elapsed);
    println!("Average query time: {:.2} microseconds", avg_time);
    
    Ok(())
}