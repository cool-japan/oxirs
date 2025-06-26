//! Example of using memory-mapped store for large RDF datasets

use oxirs_core::model::{NamedNode, BlankNode, Literal, Subject, Predicate, Object, Quad, GraphName};
use oxirs_core::store::{MmapStore, StoreStats};
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("Memory-Mapped RDF Store Example");
    println!("===============================\n");
    
    // Create a memory-mapped store
    let store_path = "/tmp/oxirs_mmap_example";
    std::fs::create_dir_all(store_path)?;
    
    let store = MmapStore::new(store_path)?;
    println!("Created memory-mapped store at: {}", store_path);
    
    // Example 1: Adding individual quads
    println!("\n1. Adding individual quads:");
    let start = Instant::now();
    
    for i in 0..10 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(&format!("http://example.org/person/{}", i))?),
            Predicate::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/name")?),
            Object::Literal(Literal::new_simple_literal(&format!("Person {}", i))),
            GraphName::DefaultGraph,
        );
        store.add(&quad)?;
    }
    
    store.flush()?;
    println!("   Added {} quads in {:?}", store.len(), start.elapsed());
    
    // Example 2: Batch loading for better performance
    println!("\n2. Batch loading example:");
    let start = Instant::now();
    let batch_size = 1000;
    let total_quads = 100_000;
    
    for batch in 0..(total_quads / batch_size) {
        for i in 0..batch_size {
            let id = batch * batch_size + i;
            
            // Create diverse data
            let quad = match id % 4 {
                0 => Quad::new(
                    Subject::NamedNode(NamedNode::new(&format!("http://dbpedia.org/resource/Entity_{}", id))?),
                    Predicate::NamedNode(NamedNode::new("http://www.w3.org/2000/01/rdf-schema#label")?),
                    Object::Literal(Literal::new_language_tagged_literal(&format!("Entity {}", id), "en")?),
                    GraphName::DefaultGraph,
                ),
                1 => Quad::new(
                    Subject::NamedNode(NamedNode::new(&format!("http://dbpedia.org/resource/Entity_{}", id))?),
                    Predicate::NamedNode(NamedNode::new("http://dbpedia.org/ontology/populationTotal")?),
                    Object::Literal(Literal::new_typed(
                        &format!("{}", id * 1000),
                        NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?
                    )),
                    GraphName::DefaultGraph,
                ),
                2 => Quad::new(
                    Subject::BlankNode(BlankNode::new(&format!("node{}", id))?),
                    Predicate::NamedNode(NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?),
                    Object::NamedNode(NamedNode::new("http://xmlns.com/foaf/0.1/Person")?),
                    GraphName::NamedNode(NamedNode::new("http://example.org/graph/people")?),
                ),
                _ => Quad::new(
                    Subject::NamedNode(NamedNode::new(&format!("http://example.org/doc/{}", id))?),
                    Predicate::NamedNode(NamedNode::new("http://purl.org/dc/terms/created")?),
                    Object::Literal(Literal::new_typed(
                        "2024-01-01T00:00:00Z",
                        NamedNode::new("http://www.w3.org/2001/XMLSchema#dateTime")?
                    )),
                    GraphName::NamedNode(NamedNode::new("http://example.org/graph/metadata")?),
                ),
            };
            
            store.add(&quad)?;
        }
        
        // Flush periodically for better performance
        store.flush()?;
        
        if (batch + 1) % 10 == 0 {
            println!("   Progress: {} quads loaded...", (batch + 1) * batch_size);
        }
    }
    
    let elapsed = start.elapsed();
    println!("   Loaded {} quads in {:?}", total_quads, elapsed);
    println!("   Rate: {:.0} quads/second", total_quads as f64 / elapsed.as_secs_f64());
    
    // Example 3: Store statistics
    println!("\n3. Store statistics:");
    let stats = store.stats();
    print_stats(&stats);
    
    // Example 4: Simulating incremental updates
    println!("\n4. Incremental updates:");
    let start = Instant::now();
    
    for i in 0..1000 {
        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new(&format!("http://example.org/update/{}", i))?),
            Predicate::NamedNode(NamedNode::new("http://example.org/timestamp")?),
            Object::Literal(Literal::new_typed(
                &format!("{}", start.elapsed().as_millis()),
                NamedNode::new("http://www.w3.org/2001/XMLSchema#long")?
            )),
            GraphName::NamedNode(NamedNode::new("http://example.org/graph/updates")?),
        );
        store.add(&quad)?;
        
        // Simulate periodic flushes
        if i % 100 == 99 {
            store.flush()?;
        }
    }
    
    store.flush()?;
    println!("   Added 1000 incremental updates in {:?}", start.elapsed());
    
    // Final statistics
    println!("\n5. Final store statistics:");
    let final_stats = store.stats();
    print_stats(&final_stats);
    
    println!("\n6. Memory-mapped store benefits:");
    println!("   - Data persisted to disk: {}", store_path);
    println!("   - Can handle datasets larger than RAM");
    println!("   - Crash-safe with append-only writes");
    println!("   - Supports concurrent readers");
    println!("   - Automatic OS-level caching");
    
    // Demonstrate reopening the store
    println!("\n7. Reopening the store:");
    drop(store); // Close the current store
    
    let reopened_store = MmapStore::open(store_path)?;
    println!("   Reopened store contains {} quads", reopened_store.len());
    
    println!("\nExample completed successfully!");
    
    Ok(())
}

fn print_stats(stats: &StoreStats) {
    println!("   Quad count: {}", stats.quad_count);
    println!("   Term count: {}", stats.term_count);
    println!("   Data size: {} MB", stats.data_size as f64 / 1_048_576.0);
    println!("   Index size: {} MB", stats.index_size as f64 / 1_048_576.0);
    println!("   Term size: {} MB", stats.term_size as f64 / 1_048_576.0);
    let total_size = stats.data_size + stats.index_size + stats.term_size;
    println!("   Total size: {} MB", total_size as f64 / 1_048_576.0);
}