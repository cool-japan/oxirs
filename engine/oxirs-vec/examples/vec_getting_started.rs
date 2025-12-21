//! Getting Started with OxiRS Vec
//!
//! This example demonstrates basic vector search functionality:
//! - Creating a vector store
//! - Indexing vectors
//! - Similarity search
//! - Using HNSW index
//!
//! Run with: cargo run --example getting_started --no-default-features

use anyhow::Result;
use oxirs_vec::{
    hnsw::{HnswConfig, HnswIndex},
    Vector, VectorIndex, VectorStore,
};

fn main() -> Result<()> {
    println!("=== OxiRS Vec - Getting Started ===\n");

    // Example 1: Basic vector operations
    basic_vector_operations()?;

    // Example 2: Simple vector store
    simple_vector_store()?;

    // Example 3: HNSW index
    hnsw_index_example()?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Demonstrate basic vector operations
fn basic_vector_operations() -> Result<()> {
    println!("1. Basic Vector Operations");
    println!("   ---------------------");

    // Create vectors
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);

    println!("   Created vectors:");
    println!("     v1: {:?}", v1.as_f32());
    println!("     v2: {:?}", v2.as_f32());

    // Calculate similarity
    let similarity = v1.cosine_similarity(&v2)?;
    println!("   Cosine similarity: {:.4}", similarity);

    // Calculate distance
    let distance = v1.euclidean_distance(&v2)?;
    println!("   Euclidean distance: {:.4}", distance);

    // Vector arithmetic
    let sum = v1.add(&v2)?;
    println!("   v1 + v2 = {:?}", sum.as_f32());

    let diff = v2.subtract(&v1)?;
    println!("   v2 - v1 = {:?}", diff.as_f32());

    // Normalization
    let mut v3 = Vector::new(vec![3.0, 4.0, 0.0]);
    println!("   Before normalization: magnitude = {:.4}", v3.magnitude());
    v3.normalize();
    println!("   After normalization: magnitude = {:.4}", v3.magnitude());

    println!("   ✓ Vector operations complete\n");
    Ok(())
}

/// Demonstrate simple vector store
fn simple_vector_store() -> Result<()> {
    println!("2. Simple Vector Store");
    println!("   ------------------");

    let mut store = VectorStore::new();

    // Index some vectors
    println!("   Indexing vectors...");

    let doc1 = Vector::new(vec![1.0, 0.0, 0.0]);
    let doc2 = Vector::new(vec![0.0, 1.0, 0.0]);
    let doc3 = Vector::new(vec![0.0, 0.0, 1.0]);
    let doc4 = Vector::new(vec![0.7, 0.7, 0.0]); // Similar to doc1 and doc2

    store.index_vector("doc1".to_string(), doc1)?;
    store.index_vector("doc2".to_string(), doc2)?;
    store.index_vector("doc3".to_string(), doc3)?;
    store.index_vector("doc4".to_string(), doc4)?;

    println!("   Indexed 4 documents");

    // Search
    println!("   Searching for vectors similar to [1.0, 0.0, 0.0]...");
    let query = Vector::new(vec![1.0, 0.0, 0.0]);
    let results = store.similarity_search_vector(&query, 3)?;

    println!("   Top 3 results:");
    for (i, (id, score)) in results.iter().enumerate() {
        println!("     {}. {} (similarity: {:.4})", i + 1, id, score);
    }

    println!("   ✓ Vector store complete\n");
    Ok(())
}

/// Demonstrate HNSW index for efficient similarity search
fn hnsw_index_example() -> Result<()> {
    println!("3. HNSW Index (High-Performance Search)");
    println!("   -----------------------------------");

    // Configure HNSW for good balance of speed and accuracy
    let config = HnswConfig {
        m: 16,                     // Connections per layer
        m_l0: 32,                  // Connections for layer 0
        ml: 1.0 / (16.0_f64).ln(), // Level generation factor
        ef: 100,                   // Search quality
        ef_construction: 200,      // Build quality
        ..Default::default()
    };

    println!("   Creating HNSW index with:");
    println!("     - M (connections): {}", config.m);
    println!("     - ef_construction: {}", config.ef_construction);
    println!("     - ef (search): {}", config.ef);

    let mut index = HnswIndex::new(config)?;

    // Index 100 random vectors
    println!("   Indexing 100 vectors...");
    for i in 0..100 {
        // Create a simple pattern for demonstration
        let values: Vec<f32> = (0..128)
            .map(|j| {
                let phase = (i as f32 * 0.1) + (j as f32 * 0.01);
                phase.sin() * 0.5
            })
            .collect();

        let vector = Vector::new(values);
        index.insert(format!("vector_{:03}", i), vector)?;
    }

    println!("   Successfully indexed 100 vectors");

    // Search
    println!("   Performing similarity search...");

    let query_values: Vec<f32> = (0..128).map(|j| ((j as f32) * 0.02).cos() * 0.5).collect();
    let query = Vector::new(query_values);

    let results = index.search_knn(&query, 5)?;

    println!("   Top 5 most similar vectors:");
    for (i, (id, score)) in results.iter().enumerate() {
        println!("     {}. {} (similarity: {:.4})", i + 1, id, score);
    }

    println!("   ✓ HNSW index complete\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        assert!(basic_vector_operations().is_ok());
    }

    #[test]
    fn test_vector_store() {
        assert!(simple_vector_store().is_ok());
    }

    #[test]
    fn test_hnsw_index() {
        assert!(hnsw_index_example().is_ok());
    }
}
