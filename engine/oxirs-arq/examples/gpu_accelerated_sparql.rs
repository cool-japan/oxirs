//! GPU-Accelerated SPARQL Operations Example
//!
//! This example demonstrates the GPU-accelerated query engine for
//! high-performance vector similarity search in SPARQL queries.
//!
//! # Features Demonstrated
//!
//! - SIMD-accelerated vector similarity search
//! - Configuration options for different performance profiles
//! - Result caching and statistics tracking
//! - Batch processing for optimal throughput
//!
//! # Run Example
//!
//! ```bash
//! cargo run --example gpu_accelerated_sparql --features=parallel
//! ```

use oxirs_arq::gpu_accelerated_ops::{GpuConfig, GpuQueryEngine};
use scirs2_core::ndarray_ext::{array, Array1, Array2};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU-Accelerated SPARQL Operations Demo\n");
    println!("{}", "=".repeat(60));

    // Demo 1: Auto-detect configuration
    demo_auto_detect()?;

    // Demo 2: High-performance configuration
    demo_high_performance()?;

    // Demo 3: Low-memory configuration
    demo_low_memory()?;

    // Demo 4: Batch processing
    demo_batch_processing()?;

    // Demo 5: Cache effectiveness
    demo_caching()?;

    // Demo 6: Performance comparison
    demo_performance_comparison()?;

    println!("\n✅ All GPU operations demonstrations completed successfully!");
    Ok(())
}

fn demo_auto_detect() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demo 1: Auto-Detect Configuration");
    println!("{}", "-".repeat(60));

    let config = GpuConfig::auto_detect();
    let engine = GpuQueryEngine::new(config)?;

    println!("Device info: {}", engine.gpu_info().unwrap_or_default());
    println!("GPU available: {}", engine.is_gpu_available());

    // Create sample embeddings (simulating RDF entity embeddings)
    let embeddings = array![
        [1.0, 0.0, 0.0, 0.0],     // Entity 0
        [0.0, 1.0, 0.0, 0.0],     // Entity 1
        [0.0, 0.0, 1.0, 0.0],     // Entity 2
        [0.707, 0.707, 0.0, 0.0], // Entity 3 (similar to 0 and 1)
        [0.0, 0.0, 0.707, 0.707], // Entity 4 (similar to 2)
    ];

    // Query vector (looking for entities similar to Entity 0)
    let query = array![1.0, 0.0, 0.0, 0.0];

    // Find top-3 most similar entities
    let results = engine.vector_similarity_search(&embeddings, &query, 3)?;

    println!("\nQuery vector: {:?}", query);
    println!("Top-3 similar entities:");
    for (idx, (entity_id, similarity)) in results.iter().enumerate() {
        println!(
            "  {}. Entity {} (similarity: {:.4})",
            idx + 1,
            entity_id,
            similarity
        );
    }

    let stats = engine.stats();
    println!("\nStatistics:");
    println!("  Total operations: {}", stats.total_operations);
    println!("  Average time: {:.2}ms", stats.avg_time_ms());

    Ok(())
}

fn demo_high_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Demo 2: High-Performance Configuration");
    println!("{}", "-".repeat(60));

    let config = GpuConfig::high_performance();
    let engine = GpuQueryEngine::new(config)?;

    println!("Batch size: {}", engine.config().batch_size);
    println!("Caching enabled: {}", engine.config().enable_caching);

    // Larger dataset for performance testing
    let n_entities = 1000;
    let embedding_dim = 128;

    // Generate random embeddings (in practice, these would come from RDF data)
    let mut embeddings_vec = Vec::with_capacity(n_entities * embedding_dim);
    for i in 0..n_entities {
        for j in 0..embedding_dim {
            let val = ((i * 7 + j * 13) % 100) as f32 / 100.0;
            embeddings_vec.push(val);
        }
    }
    let embeddings = Array2::from_shape_vec((n_entities, embedding_dim), embeddings_vec)?;

    // Query vector
    let query_vec: Vec<f32> = (0..embedding_dim).map(|i| (i % 10) as f32 / 10.0).collect();
    let query = Array1::from_vec(query_vec);

    // Measure performance
    let start = Instant::now();
    let results = engine.vector_similarity_search(&embeddings, &query, 10)?;
    let duration = start.elapsed();

    println!(
        "\nProcessed {} entities in {:.2}ms",
        n_entities,
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Throughput: {:.0} entities/sec",
        n_entities as f64 / duration.as_secs_f64()
    );
    println!("\nTop-5 results:");
    for (idx, (entity_id, similarity)) in results.iter().take(5).enumerate() {
        println!(
            "  {}. Entity {} (similarity: {:.4})",
            idx + 1,
            entity_id,
            similarity
        );
    }

    Ok(())
}

fn demo_low_memory() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Demo 3: Low-Memory Configuration");
    println!("{}", "-".repeat(60));

    let config = GpuConfig::low_memory();
    let engine = GpuQueryEngine::new(config)?;

    println!("Batch size: {}", engine.config().batch_size);
    println!("Caching enabled: {}", engine.config().enable_caching);

    // Small dataset for memory-constrained environments
    let embeddings = array![[1.0, 0.0], [0.0, 1.0], [0.707, 0.707],];
    let query = array![1.0, 0.0];

    let results = engine.vector_similarity_search(&embeddings, &query, 2)?;

    println!("\nResults:");
    for (entity_id, similarity) in results {
        println!("  Entity {} (similarity: {:.4})", entity_id, similarity);
    }

    println!("\nThis configuration is optimized for:");
    println!("  - Embedded systems");
    println!("  - Resource-constrained containers");
    println!("  - Edge computing devices");

    Ok(())
}

fn demo_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📦 Demo 4: Batch Processing");
    println!("{}", "-".repeat(60));

    let config = GpuConfig::high_performance();
    let engine = GpuQueryEngine::new(config)?;

    // Dataset
    let embeddings = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ];

    // Multiple queries to process in batch
    let queries = [
        array![1.0, 0.0, 0.0],
        array![0.0, 1.0, 0.0],
        array![0.0, 0.0, 1.0],
    ];

    println!("Processing {} queries in batch...", queries.len());
    let start = Instant::now();

    let mut all_results = Vec::new();
    for (i, query) in queries.iter().enumerate() {
        let results = engine.vector_similarity_search(&embeddings, query, 2)?;
        all_results.push((i, results));
    }

    let duration = start.elapsed();

    println!(
        "Batch completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Average per query: {:.2}ms",
        duration.as_secs_f64() * 1000.0 / queries.len() as f64
    );

    for (query_id, results) in all_results {
        println!("\nQuery {}:", query_id);
        for (entity_id, similarity) in results {
            println!("  Entity {} (similarity: {:.4})", entity_id, similarity);
        }
    }

    Ok(())
}

fn demo_caching() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🗄️  Demo 5: Cache Effectiveness");
    println!("{}", "-".repeat(60));

    let config = GpuConfig::auto_detect();
    let engine = GpuQueryEngine::new(config)?;

    let embeddings = array![[1.0, 0.0], [0.0, 1.0], [0.707, 0.707],];
    let query = array![1.0, 0.0];

    // First query (cache miss)
    println!("Executing query 1 (cache miss)...");
    let start = Instant::now();
    let _ = engine.vector_similarity_search(&embeddings, &query, 2)?;
    let first_time = start.elapsed();

    // Second query (cache hit)
    println!("Executing query 2 (cache hit)...");
    let start = Instant::now();
    let _ = engine.vector_similarity_search(&embeddings, &query, 2)?;
    let second_time = start.elapsed();

    let stats = engine.stats();
    println!("\nCache Statistics:");
    println!("  Cache hits: {}", stats.cache_hits);
    println!("  Cache misses: {}", stats.cache_misses);
    println!("  Hit rate: {:.2}%", stats.cache_hit_rate());

    println!("\nTiming:");
    println!("  First query:  {:.2}ms", first_time.as_secs_f64() * 1000.0);
    println!(
        "  Second query: {:.2}ms",
        second_time.as_secs_f64() * 1000.0
    );
    if first_time > second_time {
        let speedup = first_time.as_secs_f64() / second_time.as_secs_f64();
        println!("  Speedup: {:.2}x", speedup);
    }

    // Clear cache and verify
    engine.clear_cache();
    println!("\nCache cleared!");

    let start = Instant::now();
    let _ = engine.vector_similarity_search(&embeddings, &query, 2)?;
    let third_time = start.elapsed();

    println!(
        "Query after cache clear: {:.2}ms",
        third_time.as_secs_f64() * 1000.0
    );

    Ok(())
}

fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Demo 6: Performance Comparison");
    println!("{}", "-".repeat(60));

    // Test with different configurations
    let configs = vec![
        ("Low Memory", GpuConfig::low_memory()),
        ("Auto-detect", GpuConfig::auto_detect()),
        ("High Performance", GpuConfig::high_performance()),
    ];

    // Dataset
    let n_entities = 500;
    let embedding_dim = 64;
    let mut embeddings_vec = Vec::with_capacity(n_entities * embedding_dim);
    for i in 0..n_entities {
        for j in 0..embedding_dim {
            embeddings_vec.push(((i + j) % 100) as f32 / 100.0);
        }
    }
    let embeddings = Array2::from_shape_vec((n_entities, embedding_dim), embeddings_vec)?;

    let query = Array1::from_vec(vec![0.5; embedding_dim]);

    println!(
        "\nTesting {} entities with dimension {}",
        n_entities, embedding_dim
    );
    println!(
        "\n{:<20} {:>15} {:>15}",
        "Configuration", "Time (ms)", "Throughput"
    );
    println!("{}", "-".repeat(52));

    for (name, config) in configs {
        let engine = GpuQueryEngine::new(config)?;

        // Warmup
        let _ = engine.vector_similarity_search(&embeddings, &query, 10)?;

        // Measure
        let start = Instant::now();
        let _ = engine.vector_similarity_search(&embeddings, &query, 10)?;
        let duration = start.elapsed();

        let time_ms = duration.as_secs_f64() * 1000.0;
        let throughput = n_entities as f64 / duration.as_secs_f64();

        println!(
            "{:<20} {:>12.2} ms {:>11.0} ent/s",
            name, time_ms, throughput
        );

        let stats = engine.stats();
        if stats.total_operations > 1 {
            println!("  └─ Cache hit rate: {:.1}%", stats.cache_hit_rate());
        }
    }

    println!("\n💡 Observations:");
    println!("  - High-performance config optimizes for throughput");
    println!("  - Low-memory config minimizes resource usage");
    println!("  - Cache significantly improves repeat query performance");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_compiles() {
        // Just ensure the example compiles
        assert!(true);
    }
}
