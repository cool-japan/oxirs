//! Example demonstrating memory-mapped vector index for large datasets

use anyhow::Result;
use oxirs_vec::{
    Vector, VectorIndex, MemoryMappedVectorIndex, IndexConfig, IndexType, DistanceMetric
};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Memory-Mapped Vector Index Example");
    println!("==================================\n");

    // Create index configuration
    let config = IndexConfig {
        index_type: IndexType::Flat, // Memory-mapped index uses flat search
        distance_metric: DistanceMetric::Cosine,
        parallel: true,
        ..Default::default()
    };

    // Create a memory-mapped index file
    let index_path = "vectors_mmap.idx";
    let mut index = MemoryMappedVectorIndex::new(index_path, config)?;

    println!("1. Indexing vectors...");
    let start = Instant::now();

    // Index a large number of vectors
    let num_vectors = 10000;
    let dimensions = 128;

    for i in 0..num_vectors {
        // Generate a random vector for demonstration
        let mut values = Vec::with_capacity(dimensions);
        for j in 0..dimensions {
            // Create deterministic "random" values
            let value = ((i * j) as f32 % 100.0) / 100.0 - 0.5;
            values.push(value);
        }
        
        let vector = Vector::new(values);
        let uri = format!("http://example.org/vector/{}", i);
        
        index.insert(uri, vector)?;
        
        if (i + 1) % 1000 == 0 {
            println!("  Indexed {} vectors...", i + 1);
        }
    }

    let indexing_time = start.elapsed();
    println!("  Indexed {} vectors in {:?}\n", num_vectors, indexing_time);

    // Get index statistics
    let stats = index.stats();
    println!("2. Index Statistics:");
    println!("  - Vector count: {}", stats.vector_count);
    println!("  - Dimensions: {}", stats.dimensions);
    println!("  - File size: {} bytes", stats.file_size);
    println!("  - Memory usage: {} bytes\n", stats.memory_usage);

    // Perform similarity search
    println!("3. Performing similarity search...");

    // Create a query vector
    let mut query_values = Vec::with_capacity(dimensions);
    for j in 0..dimensions {
        query_values.push((j as f32 / dimensions as f32) - 0.5);
    }
    let query_vector = Vector::new(query_values);

    // Search for 10 nearest neighbors
    let k = 10;
    let search_start = Instant::now();
    let results = index.search_knn(&query_vector, k)?;
    let search_time = search_start.elapsed();

    println!("  Found {} nearest neighbors in {:?}:", results.len(), search_time);
    for (i, (uri, distance)) in results.iter().enumerate() {
        println!("    {}. {} (distance: {:.4})", i + 1, uri, distance);
    }
    println!();

    // Test persistence - drop and reload
    println!("4. Testing persistence...");
    drop(index);

    // Load existing index
    let loaded_index = MemoryMappedVectorIndex::load(index_path, config)?;
    let loaded_stats = loaded_index.stats();
    
    println!("  Loaded index with {} vectors", loaded_stats.vector_count);
    
    // Perform search on loaded index
    let search_start = Instant::now();
    let loaded_results = loaded_index.search_knn(&query_vector, k)?;
    let search_time = search_start.elapsed();
    
    println!("  Search on loaded index completed in {:?}", search_time);
    println!("  Results match: {}\n", results == loaded_results);

    // Threshold search example
    println!("5. Threshold search example...");
    let threshold = 0.3; // Find all vectors with distance < 0.3
    let threshold_results = loaded_index.search_threshold(&query_vector, threshold)?;
    
    println!("  Found {} vectors within distance threshold {}", 
             threshold_results.len(), threshold);
    if threshold_results.len() <= 20 {
        for (uri, distance) in &threshold_results {
            println!("    {} (distance: {:.4})", uri, distance);
        }
    } else {
        println!("    (showing first 5 results)");
        for (uri, distance) in threshold_results.iter().take(5) {
            println!("    {} (distance: {:.4})", uri, distance);
        }
    }

    // Demonstrate memory efficiency
    println!("\n6. Memory Efficiency Comparison:");
    let in_memory_size = num_vectors * dimensions * std::mem::size_of::<f32>();
    let mmap_memory = loaded_stats.memory_usage;
    let savings = (1.0 - (mmap_memory as f64 / in_memory_size as f64)) * 100.0;
    
    println!("  In-memory storage would use: {} bytes", in_memory_size);
    println!("  Memory-mapped index uses: {} bytes", mmap_memory);
    println!("  Memory savings: {:.1}%", savings);

    // Clean up
    std::fs::remove_file(index_path).ok();

    println!("\nExample completed successfully!");

    Ok(())
}