//! Example demonstrating Locality Sensitive Hashing (LSH) for approximate nearest neighbor search

use anyhow::Result;
use oxirs_vec::{LshConfig, LshFamily, LshIndex, Vector, VectorIndex};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Locality Sensitive Hashing (LSH) Example");
    println!("========================================\n");

    // Example 1: Random Projection LSH for Cosine Similarity
    println!("1. Random Projection LSH (Cosine Similarity)");
    random_projection_example()?;
    println!();

    // Example 2: MinHash LSH for Jaccard Similarity
    println!("2. MinHash LSH (Jaccard Similarity)");
    minhash_example()?;
    println!();

    // Example 3: Multi-probe LSH for improved recall
    println!("3. Multi-probe LSH");
    multiprobe_example()?;
    println!();

    // Example 4: Performance comparison
    println!("4. Performance Comparison");
    performance_comparison()?;

    Ok(())
}

fn random_projection_example() -> Result<()> {
    let config = LshConfig {
        num_tables: 10,
        num_hash_functions: 8,
        lsh_family: LshFamily::RandomProjection,
        seed: 42,
        multi_probe: false,
        num_probes: 0,
    };

    let mut index = LshIndex::new(config);

    // Create document vectors (simulating TF-IDF or embeddings)
    let documents = vec![
        ("doc1", vec![0.8, 0.2, 0.1, 0.0, 0.3]),
        ("doc2", vec![0.1, 0.9, 0.2, 0.1, 0.0]),
        ("doc3", vec![0.0, 0.1, 0.8, 0.9, 0.2]),
        ("doc4", vec![0.7, 0.3, 0.2, 0.1, 0.4]), // Similar to doc1
    ];

    // Index documents
    for (uri, values) in &documents {
        let vector = Vector::new(values.clone());
        index.insert(uri.to_string(), vector)?;
    }

    // Query with a vector similar to doc1
    let query = Vector::new(vec![0.9, 0.1, 0.2, 0.0, 0.2]);
    let results = index.search_knn(&query, 2)?;

    println!("  Query vector: [0.9, 0.1, 0.2, 0.0, 0.2]");
    println!("  Top 2 results:");
    for (uri, distance) in results {
        println!("    {uri} (distance: {distance:.4})");
    }

    // Get index statistics
    let stats = index.stats();
    println!(
        "  Index stats: {} vectors, {} tables, avg bucket size: {:.2}",
        stats.num_vectors, stats.num_tables, stats.avg_bucket_size
    );

    Ok(())
}

fn minhash_example() -> Result<()> {
    let config = LshConfig {
        num_tables: 5,
        num_hash_functions: 64,
        lsh_family: LshFamily::MinHash,
        seed: 42,
        multi_probe: false,
        num_probes: 0,
    };

    let mut index = LshIndex::new(config);

    // Create sparse binary vectors (e.g., document term sets)
    let mut doc1 = vec![0.0; 100];
    doc1[5] = 1.0;
    doc1[10] = 1.0;
    doc1[15] = 1.0;
    doc1[20] = 1.0;

    let mut doc2 = vec![0.0; 100];
    doc2[5] = 1.0;
    doc2[10] = 1.0;
    doc2[25] = 1.0;
    doc2[30] = 1.0; // 50% overlap

    let mut doc3 = vec![0.0; 100];
    doc3[50] = 1.0;
    doc3[55] = 1.0;
    doc3[60] = 1.0;
    doc3[65] = 1.0; // No overlap

    index.insert("doc1".to_string(), Vector::new(doc1.clone()))?;
    index.insert("doc2".to_string(), Vector::new(doc2))?;
    index.insert("doc3".to_string(), Vector::new(doc3))?;

    // Query with doc1
    let query = Vector::new(doc1);
    let results = index.search_knn(&query, 3)?;

    println!("  Query: Document with terms at positions [5, 10, 15, 20]");
    println!("  Results ordered by Jaccard similarity:");
    for (uri, distance) in results {
        let similarity = 1.0 - distance; // Convert distance to similarity
        println!("    {uri} (Jaccard similarity: {similarity:.4})");
    }

    Ok(())
}

fn multiprobe_example() -> Result<()> {
    // Create two indices - one with and one without multi-probe
    let config_standard = LshConfig {
        num_tables: 3,
        num_hash_functions: 4,
        lsh_family: LshFamily::RandomProjection,
        seed: 42,
        multi_probe: false,
        num_probes: 0,
    };

    let config_multiprobe = LshConfig {
        num_tables: 3,
        num_hash_functions: 4,
        lsh_family: LshFamily::RandomProjection,
        seed: 42,
        multi_probe: true,
        num_probes: 3,
    };

    let mut index_standard = LshIndex::new(config_standard);
    let mut index_multiprobe = LshIndex::new(config_multiprobe);

    // Create vectors in a circle
    let num_vectors = 20;
    for i in 0..num_vectors {
        let angle = i as f32 * 2.0 * std::f32::consts::PI / num_vectors as f32;
        let vector = Vector::new(vec![angle.cos(), angle.sin()]);
        let uri = format!("point_{i}");

        index_standard.insert(uri.clone(), vector.clone())?;
        index_multiprobe.insert(uri, vector)?;
    }

    // Query with a specific point
    let query = Vector::new(vec![1.0, 0.0]);

    let results_standard = index_standard.search_knn(&query, 5)?;
    let results_multiprobe = index_multiprobe.search_knn(&query, 5)?;

    println!("  Query point: [1.0, 0.0]");
    println!("  Standard LSH found {} neighbors", results_standard.len());
    println!(
        "  Multi-probe LSH found {} neighbors",
        results_multiprobe.len()
    );

    Ok(())
}

fn performance_comparison() -> Result<()> {
    let dimensions = 128;
    let num_vectors = 5000;
    let num_queries = 100;

    // Create LSH index
    let lsh_config = LshConfig {
        num_tables: 10,
        num_hash_functions: 8,
        lsh_family: LshFamily::RandomProjection,
        seed: 42,
        multi_probe: true,
        num_probes: 2,
    };
    let mut lsh_index = LshIndex::new(lsh_config);

    // Create brute force index for comparison
    use oxirs_vec::MemoryVectorIndex;
    let mut brute_force = MemoryVectorIndex::new();

    // Generate and index random vectors
    println!(
        "  Indexing {num_vectors} {dimensions}-dimensional vectors..."
    );
    let start = Instant::now();

    for i in 0..num_vectors {
        let mut values = Vec::with_capacity(dimensions);
        for j in 0..dimensions {
            // Deterministic "random" values
            let value = ((i * j + i) as f32 % 100.0) / 100.0 - 0.5;
            values.push(value);
        }
        let vector = Vector::new(values);
        let uri = format!("vec_{i}");

        lsh_index.insert(uri.clone(), vector.clone())?;
        brute_force.insert(uri, vector)?;
    }

    let indexing_time = start.elapsed();
    println!("  Indexing completed in {indexing_time:?}");

    // Run queries
    println!("  Running {num_queries} queries...");
    let mut lsh_times = Vec::new();
    let mut brute_times = Vec::new();
    let mut recall_scores = Vec::new();

    for q in 0..num_queries {
        // Generate query vector
        let mut query_values = Vec::with_capacity(dimensions);
        for j in 0..dimensions {
            let value = ((q * j * 7) as f32 % 100.0) / 100.0 - 0.5;
            query_values.push(value);
        }
        let query = Vector::new(query_values);

        // LSH search
        let lsh_start = Instant::now();
        let lsh_results = lsh_index.search_knn(&query, 10)?;
        lsh_times.push(lsh_start.elapsed());

        // Brute force search
        let brute_start = Instant::now();
        let brute_results = brute_force.search_knn(&query, 10)?;
        brute_times.push(brute_start.elapsed());

        // Calculate recall
        let lsh_set: std::collections::HashSet<_> =
            lsh_results.iter().map(|(uri, _)| uri).collect();
        let brute_set: std::collections::HashSet<_> =
            brute_results.iter().map(|(uri, _)| uri).collect();
        let intersection = lsh_set.intersection(&brute_set).count();
        let recall = intersection as f32 / brute_set.len() as f32;
        recall_scores.push(recall);
    }

    // Calculate statistics
    let avg_lsh_time = lsh_times.iter().sum::<std::time::Duration>() / num_queries as u32;
    let avg_brute_time = brute_times.iter().sum::<std::time::Duration>() / num_queries as u32;
    let avg_recall = recall_scores.iter().sum::<f32>() / num_queries as f32;
    let speedup = avg_brute_time.as_secs_f64() / avg_lsh_time.as_secs_f64();

    println!("\n  Results:");
    println!("    LSH average query time: {avg_lsh_time:?}");
    println!("    Brute force average query time: {avg_brute_time:?}");
    println!("    Speedup: {speedup:.2}x");
    println!("    Average recall@10: {:.2}%", avg_recall * 100.0);

    let lsh_stats = lsh_index.stats();
    println!("\n  LSH Index Statistics:");
    println!("    Number of tables: {}", lsh_stats.num_tables);
    println!("    Average bucket size: {:.2}", lsh_stats.avg_bucket_size);
    println!("    Memory usage: {} KB", lsh_stats.memory_usage / 1024);

    Ok(())
}
