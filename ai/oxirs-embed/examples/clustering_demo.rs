//! Comprehensive Clustering Demo for Knowledge Graph Embeddings
//!
//! This example demonstrates advanced entity clustering capabilities using
//! knowledge graph embeddings, including:
//! - K-Means clustering with K-Means++ initialization
//! - Hierarchical (agglomerative) clustering
//! - DBSCAN (density-based) clustering
//! - Spectral clustering
//! - Cluster quality metrics (silhouette score, inertia)
//! - Cluster visualization and analysis
//! - Multi-level clustering for hierarchical organization
//!
//! # Entity Clustering
//!
//! Entity clustering groups similar entities based on their learned embeddings.
//! This is useful for:
//! - Discovering entity types and categories
//! - Finding semantic groups in knowledge graphs
//! - Data exploration and analysis
//! - Improving recommendation systems
//! - Anomaly detection
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example clustering_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    clustering::{ClusteringAlgorithm, ClusteringConfig, EntityClustering},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};
use scirs2_core::ndarray_ext::Array1;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   Clustering Demo - Entity Grouping & Discovery        ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // ====================
    // Step 1: Build a Rich Knowledge Graph
    // ====================
    println!("📚 Step 1: Building Multi-Domain Knowledge Graph");
    println!("─────────────────────────────────────────────────────────");

    let config = ModelConfig {
        dimensions: 100,
        learning_rate: 0.01,
        max_epochs: 200,
        batch_size: 64,
        ..Default::default()
    };

    let mut model = TransE::new(config);

    // TECHNOLOGY DOMAIN
    println!("  Adding technology entities...");
    add_triple(&mut model, "python", "is_a", "programming_language")?;
    add_triple(&mut model, "java", "is_a", "programming_language")?;
    add_triple(&mut model, "rust", "is_a", "programming_language")?;
    add_triple(&mut model, "javascript", "is_a", "programming_language")?;

    add_triple(&mut model, "linux", "is_a", "operating_system")?;
    add_triple(&mut model, "windows", "is_a", "operating_system")?;
    add_triple(&mut model, "macos", "is_a", "operating_system")?;

    add_triple(&mut model, "postgresql", "is_a", "database")?;
    add_triple(&mut model, "mongodb", "is_a", "database")?;
    add_triple(&mut model, "redis", "is_a", "database")?;

    // FOOD DOMAIN
    println!("  Adding food entities...");
    add_triple(&mut model, "apple", "is_a", "fruit")?;
    add_triple(&mut model, "banana", "is_a", "fruit")?;
    add_triple(&mut model, "orange", "is_a", "fruit")?;
    add_triple(&mut model, "strawberry", "is_a", "fruit")?;

    add_triple(&mut model, "carrot", "is_a", "vegetable")?;
    add_triple(&mut model, "broccoli", "is_a", "vegetable")?;
    add_triple(&mut model, "spinach", "is_a", "vegetable")?;

    add_triple(&mut model, "beef", "is_a", "meat")?;
    add_triple(&mut model, "chicken", "is_a", "meat")?;
    add_triple(&mut model, "pork", "is_a", "meat")?;

    // GEOGRAPHY DOMAIN
    println!("  Adding geography entities...");
    add_triple(&mut model, "paris", "is_capital_of", "france")?;
    add_triple(&mut model, "london", "is_capital_of", "uk")?;
    add_triple(&mut model, "berlin", "is_capital_of", "germany")?;
    add_triple(&mut model, "rome", "is_capital_of", "italy")?;

    add_triple(&mut model, "france", "is_a", "country")?;
    add_triple(&mut model, "uk", "is_a", "country")?;
    add_triple(&mut model, "germany", "is_a", "country")?;
    add_triple(&mut model, "italy", "is_a", "country")?;

    add_triple(&mut model, "paris", "is_a", "city")?;
    add_triple(&mut model, "london", "is_a", "city")?;
    add_triple(&mut model, "berlin", "is_a", "city")?;
    add_triple(&mut model, "rome", "is_a", "city")?;

    // ANIMALS DOMAIN
    println!("  Adding animal entities...");
    add_triple(&mut model, "dog", "is_a", "mammal")?;
    add_triple(&mut model, "cat", "is_a", "mammal")?;
    add_triple(&mut model, "elephant", "is_a", "mammal")?;
    add_triple(&mut model, "lion", "is_a", "mammal")?;

    add_triple(&mut model, "eagle", "is_a", "bird")?;
    add_triple(&mut model, "sparrow", "is_a", "bird")?;
    add_triple(&mut model, "penguin", "is_a", "bird")?;

    add_triple(&mut model, "salmon", "is_a", "fish")?;
    add_triple(&mut model, "tuna", "is_a", "fish")?;
    add_triple(&mut model, "shark", "is_a", "fish")?;

    // Cross-domain relationships
    add_triple(&mut model, "python", "used_for", "data_science")?;
    add_triple(&mut model, "rust", "used_for", "systems_programming")?;
    add_triple(&mut model, "javascript", "used_for", "web_development")?;

    let stats = model.get_stats();
    println!("\n  Total entities: {}", stats.num_entities);
    println!("  Total relations: {}", stats.num_relations);
    println!("  Total triples: {}", stats.num_triples);
    println!();

    // ====================
    // Step 2: Train the Model
    // ====================
    println!("🎓 Step 2: Training Embedding Model");
    println!("─────────────────────────────────────────────────────────");

    let training_stats = model.train(Some(200)).await?;

    println!("  Epochs completed: {}", training_stats.epochs_completed);
    println!("  Final loss: {:.4}", training_stats.final_loss);
    println!(
        "  Training time: {:.2}s",
        training_stats.training_time_seconds
    );
    println!();

    // ====================
    // Step 3: Extract Embeddings
    // ====================
    println!("🔢 Step 3: Extracting Entity Embeddings");
    println!("─────────────────────────────────────────────────────────");

    let mut embeddings: HashMap<String, Array1<f32>> = HashMap::new();
    for entity in model.get_entities() {
        if let Ok(emb) = model.get_entity_embedding(&entity) {
            let array = Array1::from_vec(emb.values);
            embeddings.insert(entity, array);
        }
    }

    println!("  Extracted {} entity embeddings", embeddings.len());
    println!("  Embedding dimension: {}", stats.dimensions);
    println!();

    // ====================
    // Step 4: K-Means Clustering
    // ====================
    println!("🎯 Step 4: K-Means Clustering");
    println!("─────────────────────────────────────────────────────────");
    println!("  Algorithm: K-Means with K-Means++ initialization");
    println!("  Number of clusters: 8\n");

    let kmeans_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans,
        num_clusters: 8, // For our 8 main categories
        max_iterations: 100,
        tolerance: 0.0001,
        ..Default::default()
    };

    let mut kmeans = EntityClustering::new(kmeans_config);
    let kmeans_result = kmeans.cluster(&embeddings)?;

    println!("  Results:");
    println!("    Number of clusters: {}", kmeans_result.centroids.len());
    println!(
        "    Silhouette score: {:.3} (higher is better, range: [-1, 1])",
        kmeans_result.silhouette_score
    );
    println!(
        "    Inertia: {:.3} (lower is better)",
        kmeans_result.inertia
    );
    println!("    Iterations: {}", kmeans_result.num_iterations);

    analyze_clusters("K-Means", &kmeans_result.assignments);

    // ====================
    // Step 5: Hierarchical Clustering
    // ====================
    println!("\n🌳 Step 5: Hierarchical (Agglomerative) Clustering");
    println!("─────────────────────────────────────────────────────────");
    println!("  Algorithm: Bottom-up hierarchical clustering");
    println!("  Linkage: Average linkage\n");

    let hierarchical_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::Hierarchical,
        num_clusters: 8,
        ..Default::default()
    };

    let mut hierarchical = EntityClustering::new(hierarchical_config);
    let hierarchical_result = hierarchical.cluster(&embeddings)?;

    println!("  Results:");
    println!(
        "    Silhouette score: {:.3}",
        hierarchical_result.silhouette_score
    );
    println!(
        "    Number of merges: {}",
        hierarchical_result.num_iterations
    );

    analyze_clusters("Hierarchical", &hierarchical_result.assignments);

    // ====================
    // Step 6: DBSCAN Clustering
    // ====================
    println!("\n🔍 Step 6: DBSCAN (Density-Based) Clustering");
    println!("─────────────────────────────────────────────────────────");
    println!("  Algorithm: Density-based spatial clustering");
    println!("  Advantage: Discovers clusters of arbitrary shape\n");

    let dbscan_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::DBSCAN,
        epsilon: 0.5,  // Neighborhood radius
        min_points: 3, // Minimum points to form a cluster
        ..Default::default()
    };

    let mut dbscan = EntityClustering::new(dbscan_config);
    let dbscan_result = dbscan.cluster(&embeddings)?;

    println!("  Results:");
    println!(
        "    Number of clusters: {} (excluding noise)",
        dbscan_result.centroids.len()
    );
    println!(
        "    Silhouette score: {:.3}",
        dbscan_result.silhouette_score
    );

    // Count noise points (cluster_id == usize::MAX for noise)
    let noise_count = dbscan_result
        .assignments
        .values()
        .filter(|&&id| id == usize::MAX)
        .count();
    println!("    Noise points: {}", noise_count);

    analyze_clusters("DBSCAN", &dbscan_result.assignments);

    // ====================
    // Step 7: Spectral Clustering
    // ====================
    println!("\n🌈 Step 7: Spectral Clustering");
    println!("─────────────────────────────────────────────────────────");
    println!("  Algorithm: Graph-based spectral clustering");
    println!("  Uses eigenvalues of similarity matrix\n");

    let spectral_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::Spectral,
        num_clusters: 8,
        ..Default::default()
    };

    let mut spectral = EntityClustering::new(spectral_config);
    let spectral_result = spectral.cluster(&embeddings)?;

    println!("  Results:");
    println!(
        "    Silhouette score: {:.3}",
        spectral_result.silhouette_score
    );

    analyze_clusters("Spectral", &spectral_result.assignments);

    // ====================
    // Step 8: Cluster Quality Comparison
    // ====================
    println!("\n📊 Step 8: Clustering Algorithm Comparison");
    println!("─────────────────────────────────────────────────────────");

    let algorithms = vec![
        ("K-Means", kmeans_result.silhouette_score),
        ("Hierarchical", hierarchical_result.silhouette_score),
        ("DBSCAN", dbscan_result.silhouette_score),
        ("Spectral", spectral_result.silhouette_score),
    ];

    println!("  Silhouette Scores (higher is better):");
    let mut sorted_algorithms = algorithms.clone();
    sorted_algorithms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (name, score)) in sorted_algorithms.iter().enumerate() {
        let bar_length = (score * 50.0).max(0.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("    {}. {:15} {:.3} {}", rank + 1, name, score, bar);
    }
    println!();

    // ====================
    // Step 9: Domain Discovery Analysis
    // ====================
    println!("🔬 Step 9: Domain Discovery Analysis");
    println!("─────────────────────────────────────────────────────────");
    println!("  Analyzing how well clustering discovered semantic domains\n");

    // Use K-Means results for analysis
    let mut domain_purity = HashMap::new();
    let domains = vec![
        (
            "technology",
            vec![
                "python",
                "java",
                "rust",
                "javascript",
                "linux",
                "windows",
                "macos",
                "postgresql",
                "mongodb",
                "redis",
            ],
        ),
        (
            "food",
            vec![
                "apple",
                "banana",
                "orange",
                "strawberry",
                "carrot",
                "broccoli",
                "spinach",
                "beef",
                "chicken",
                "pork",
            ],
        ),
        (
            "geography",
            vec![
                "paris", "london", "berlin", "rome", "france", "uk", "germany", "italy",
            ],
        ),
        (
            "animals",
            vec![
                "dog", "cat", "elephant", "lion", "eagle", "sparrow", "penguin", "salmon", "tuna",
                "shark",
            ],
        ),
    ];

    for (domain_name, entities) in &domains {
        let mut cluster_distribution: HashMap<usize, usize> = HashMap::new();

        for entity in entities {
            if let Some(&cluster_id) = kmeans_result.assignments.get(*entity) {
                *cluster_distribution.entry(cluster_id).or_insert(0) += 1;
            }
        }

        if let Some((&majority_cluster, &majority_count)) =
            cluster_distribution.iter().max_by_key(|&(_, count)| count)
        {
            let purity = majority_count as f64 / entities.len() as f64;
            domain_purity.insert(*domain_name, purity);

            println!("  Domain: {}", domain_name);
            println!("    Total entities: {}", entities.len());
            println!("    Majority cluster: {}", majority_cluster);
            println!("    Purity: {:.1}%", purity * 100.0);
            println!();
        }
    }

    let avg_purity: f64 = domain_purity.values().sum::<f64>() / domain_purity.len() as f64;
    println!("  Average domain purity: {:.1}%", avg_purity * 100.0);
    println!();

    // ====================
    // Step 10: Cluster Centroids Analysis
    // ====================
    println!("🎯 Step 10: Cluster Centroids Analysis");
    println!("─────────────────────────────────────────────────────────");

    println!("  Top entities closest to each cluster centroid:\n");

    for (cluster_id, centroid) in kmeans_result.centroids.iter().enumerate() {
        // Find entities closest to this centroid
        let mut distances: Vec<(String, f32)> = kmeans_result
            .assignments
            .iter()
            .filter(|(_, &cid)| cid == cluster_id)
            .filter_map(|(entity, _)| {
                embeddings.get(entity).map(|emb| {
                    let dist = euclidean_distance(emb, centroid);
                    (entity.clone(), dist)
                })
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        println!("  Cluster {}:", cluster_id);
        for (entity, dist) in distances.iter().take(5) {
            println!("    • {} (distance: {:.3})", entity, dist);
        }
        println!();
    }

    // ====================
    // Summary
    // ====================
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║             Clustering Demo Complete! ✓                ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Takeaways:");
    println!("  • Entity clustering discovers semantic groupings in KGs");
    println!("  • Different algorithms suit different data characteristics:");
    println!("    - K-Means: Fast, spherical clusters");
    println!("    - Hierarchical: Nested structure, no need to specify K upfront");
    println!("    - DBSCAN: Arbitrary shapes, handles noise");
    println!("    - Spectral: Graph structure, non-convex clusters");
    println!("  • Silhouette score measures cluster quality");
    println!("  • Clustering aids in knowledge graph organization");
    println!();
    println!("Applications:");
    println!("  • Entity type discovery and classification");
    println!("  • Knowledge graph organization and navigation");
    println!("  • Anomaly detection (outliers)");
    println!("  • Recommendation systems (user/item grouping)");
    println!("  • Data exploration and analysis");
    println!();

    Ok(())
}

/// Helper function to add a triple
fn add_triple(model: &mut TransE, subject: &str, predicate: &str, object: &str) -> Result<()> {
    model.add_triple(Triple::new(
        NamedNode::new(subject)?,
        NamedNode::new(predicate)?,
        NamedNode::new(object)?,
    ))
}

/// Analyze and print cluster composition
fn analyze_clusters(_algorithm_name: &str, assignments: &HashMap<String, usize>) {
    println!("\n  Cluster Composition:");

    // Group entities by cluster
    let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
    for (entity, &cluster_id) in assignments {
        clusters.entry(cluster_id).or_default().push(entity.clone());
    }

    // Sort clusters by ID
    let mut cluster_ids: Vec<usize> = clusters.keys().copied().collect();
    cluster_ids.sort();

    for cluster_id in cluster_ids {
        if cluster_id == usize::MAX {
            println!("    Cluster NOISE (outliers):");
        } else {
            println!("    Cluster {}:", cluster_id);
        }

        if let Some(entities) = clusters.get(&cluster_id) {
            println!("      Size: {} entities", entities.len());

            // Show first few entities
            for entity in entities.iter().take(6) {
                println!("        • {}", entity);
            }
            if entities.len() > 6 {
                println!("        ... and {} more", entities.len() - 6);
            }
        }
        println!();
    }
}

/// Compute Euclidean distance between two vectors
fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}
