//! Advanced Embedding Visualization Demo
//!
//! This example demonstrates comprehensive visualization capabilities for knowledge graph
//! embeddings, including:
//! - Multiple dimensionality reduction techniques (PCA, t-SNE, Random Projection)
//! - 2D and 3D visualizations
//! - Cluster coloring and labeling
//! - Export to various formats (CSV, JSON)
//! - Interactive analysis of embedding space
//!
//! # Visualization Techniques
//!
//! ## PCA (Principal Component Analysis)
//! - Linear transformation preserving maximum variance
//! - Fast and deterministic
//! - Good for initial exploration
//!
//! ## t-SNE (t-Distributed Stochastic Neighbor Embedding)
//! - Non-linear technique preserving local structure
//! - Reveals clusters and patterns
//! - Slower but more insightful
//!
//! ## Random Projection
//! - Very fast approximate dimensionality reduction
//! - Good for large datasets
//! - Less accurate but sufficient for quick visualization
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example advanced_visualization_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    clustering::{ClusteringAlgorithm, ClusteringConfig, EntityClustering},
    visualization::{EmbeddingVisualizer, ReductionMethod, VisualizationConfig},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   Advanced Visualization Demo - Embedding Analysis     ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // ====================
    // Step 1: Build Multi-Domain Knowledge Graph
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

    // ANIMALS
    println!("  Adding animals domain...");
    add_triple(&mut model, "dog", "is_a", "mammal")?;
    add_triple(&mut model, "cat", "is_a", "mammal")?;
    add_triple(&mut model, "elephant", "is_a", "mammal")?;
    add_triple(&mut model, "whale", "is_a", "mammal")?;

    add_triple(&mut model, "eagle", "is_a", "bird")?;
    add_triple(&mut model, "sparrow", "is_a", "bird")?;
    add_triple(&mut model, "penguin", "is_a", "bird")?;

    add_triple(&mut model, "salmon", "is_a", "fish")?;
    add_triple(&mut model, "shark", "is_a", "fish")?;
    add_triple(&mut model, "tuna", "is_a", "fish")?;

    // COLORS
    println!("  Adding colors domain...");
    add_triple(&mut model, "red", "is_a", "color")?;
    add_triple(&mut model, "blue", "is_a", "color")?;
    add_triple(&mut model, "green", "is_a", "color")?;
    add_triple(&mut model, "yellow", "is_a", "color")?;

    add_triple(&mut model, "red", "similar_to", "orange")?;
    add_triple(&mut model, "blue", "similar_to", "cyan")?;
    add_triple(&mut model, "green", "similar_to", "lime")?;

    // FRUITS
    println!("  Adding fruits domain...");
    add_triple(&mut model, "apple", "is_a", "fruit")?;
    add_triple(&mut model, "banana", "is_a", "fruit")?;
    add_triple(&mut model, "orange", "is_a", "fruit")?;
    add_triple(&mut model, "strawberry", "is_a", "fruit")?;

    add_triple(&mut model, "apple", "has_color", "red")?;
    add_triple(&mut model, "banana", "has_color", "yellow")?;
    add_triple(&mut model, "orange", "has_color", "orange")?;

    // TECHNOLOGY
    println!("  Adding technology domain...");
    add_triple(&mut model, "python", "is_a", "language")?;
    add_triple(&mut model, "rust", "is_a", "language")?;
    add_triple(&mut model, "javascript", "is_a", "language")?;

    add_triple(&mut model, "linux", "is_a", "os")?;
    add_triple(&mut model, "windows", "is_a", "os")?;
    add_triple(&mut model, "macos", "is_a", "os")?;

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

    let mut embeddings = HashMap::new();
    for entity in model.get_entities() {
        if let Ok(emb) = model.get_entity_embedding(&entity) {
            let array = scirs2_core::ndarray_ext::Array1::from_vec(emb.values);
            embeddings.insert(entity, array);
        }
    }

    println!("  Extracted {} embeddings", embeddings.len());
    println!("  Embedding dimensions: {}", stats.dimensions);
    println!();

    // ====================
    // Step 4: PCA Visualization
    // ====================
    println!("📊 Step 4: PCA Visualization (2D)");
    println!("─────────────────────────────────────────────────────────");

    let pca_config = VisualizationConfig {
        method: ReductionMethod::PCA,
        target_dims: 2,
        ..Default::default()
    };

    let mut pca_viz = EmbeddingVisualizer::new(pca_config);
    let pca_result = pca_viz.visualize(&embeddings)?;

    if let Some(ref variance) = pca_result.explained_variance {
        println!("  Explained variance:");
        for (i, var) in variance.iter().enumerate() {
            println!("    PC{}: {:.2}%", i + 1, var * 100.0);
        }
        let total: f32 = variance.iter().sum();
        println!("  Total explained: {:.2}%", total * 100.0);
    }

    // Save PCA results to CSV
    save_to_csv("pca_2d_embeddings.csv", &pca_result)?;
    println!("\n  ✓ Saved to pca_2d_embeddings.csv");
    println!();

    // ====================
    // Step 5: t-SNE Visualization
    // ====================
    println!("🎯 Step 5: t-SNE Visualization (2D)");
    println!("─────────────────────────────────────────────────────────");
    println!("  (This may take a minute for iterative optimization...)");

    let tsne_config = VisualizationConfig {
        method: ReductionMethod::TSNE,
        target_dims: 2,
        tsne_perplexity: 10.0,
        tsne_learning_rate: 200.0,
        max_iterations: 500,
        ..Default::default()
    };

    let mut tsne_viz = EmbeddingVisualizer::new(tsne_config);
    let tsne_result = tsne_viz.visualize(&embeddings)?;

    if let Some(loss) = tsne_result.final_loss {
        println!("\n  Final t-SNE loss: {:.4}", loss);
    }

    save_to_csv("tsne_2d_embeddings.csv", &tsne_result)?;
    println!("  ✓ Saved to tsne_2d_embeddings.csv");
    println!();

    // ====================
    // Step 6: Random Projection (Fast)
    // ====================
    println!("⚡ Step 6: Random Projection Visualization (2D)");
    println!("─────────────────────────────────────────────────────────");

    let rp_config = VisualizationConfig {
        method: ReductionMethod::RandomProjection,
        target_dims: 2,
        ..Default::default()
    };

    let mut rp_viz = EmbeddingVisualizer::new(rp_config);
    let rp_result = rp_viz.visualize(&embeddings)?;

    save_to_csv("random_projection_2d_embeddings.csv", &rp_result)?;
    println!("  ✓ Saved to random_projection_2d_embeddings.csv");
    println!();

    // ====================
    // Step 7: 3D PCA Visualization
    // ====================
    println!("🌐 Step 7: 3D PCA Visualization");
    println!("─────────────────────────────────────────────────────────");

    let pca_3d_config = VisualizationConfig {
        method: ReductionMethod::PCA,
        target_dims: 3,
        ..Default::default()
    };

    let mut pca_3d_viz = EmbeddingVisualizer::new(pca_3d_config);
    let pca_3d_result = pca_3d_viz.visualize(&embeddings)?;

    if let Some(ref variance) = pca_3d_result.explained_variance {
        println!("  Explained variance (3D):");
        for (i, var) in variance.iter().enumerate() {
            println!("    PC{}: {:.2}%", i + 1, var * 100.0);
        }
    }

    save_to_csv("pca_3d_embeddings.csv", &pca_3d_result)?;
    println!("  ✓ Saved to pca_3d_embeddings.csv");
    println!();

    // ====================
    // Step 8: Cluster Analysis with Visualization
    // ====================
    println!("🗂️  Step 8: Cluster Analysis");
    println!("─────────────────────────────────────────────────────────");

    let cluster_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans,
        num_clusters: 5,
        max_iterations: 100,
        ..Default::default()
    };

    let mut clustering = EntityClustering::new(cluster_config);
    let cluster_result = clustering.cluster(&embeddings)?;

    println!("  Number of clusters: {}", cluster_result.centroids.len());
    println!("  Silhouette score: {:.3}", cluster_result.silhouette_score);

    // Show cluster assignments by domain
    println!("\n  Cluster assignments:");
    let mut clusters_by_id: HashMap<usize, Vec<String>> = HashMap::new();
    for (entity, &cluster_id) in &cluster_result.assignments {
        clusters_by_id
            .entry(cluster_id)
            .or_default()
            .push(entity.clone());
    }

    for (cluster_id, entities) in clusters_by_id.iter() {
        println!("    Cluster {}: {:?}", cluster_id, entities);
    }

    // Save clustered PCA visualization
    save_clustered_to_csv(
        "pca_2d_with_clusters.csv",
        &pca_result,
        &cluster_result.assignments,
    )?;
    println!("\n  ✓ Saved clustered visualization to pca_2d_with_clusters.csv");
    println!();

    // ====================
    // Step 9: Embedding Space Analysis
    // ====================
    println!("🔍 Step 9: Embedding Space Analysis");
    println!("─────────────────────────────────────────────────────────");

    // Find nearest neighbors for sample entities
    let sample_entities = vec!["dog", "python", "apple"];

    for entity in sample_entities {
        if pca_result.coordinates.contains_key(entity) {
            println!("\n  Nearest neighbors to '{}' in PCA space:", entity);

            let entity_coords = &pca_result.coordinates[entity];
            let mut distances: Vec<(String, f32)> = pca_result
                .coordinates
                .iter()
                .filter(|(e, _)| *e != entity)
                .map(|(e, coords)| {
                    let dist = euclidean_distance(entity_coords, coords);
                    (e.clone(), dist)
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (neighbor, dist) in distances.iter().take(3) {
                println!("    {} (distance: {:.3})", neighbor, dist);
            }
        }
    }

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║           Visualization Demo Completed!                ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    println!("📁 Generated Files:");
    println!("  • pca_2d_embeddings.csv - 2D PCA visualization");
    println!("  • tsne_2d_embeddings.csv - 2D t-SNE visualization");
    println!("  • random_projection_2d_embeddings.csv - Fast 2D projection");
    println!("  • pca_3d_embeddings.csv - 3D PCA visualization");
    println!("  • pca_2d_with_clusters.csv - Clustered 2D visualization");
    println!();

    println!("💡 Next Steps:");
    println!("  • Open CSV files in spreadsheet software");
    println!("  • Plot (x, y) coordinates to visualize embeddings");
    println!("  • Color by cluster_id to see semantic groupings");
    println!("  • Use 3D visualization tools for interactive exploration");
    println!();

    Ok(())
}

fn add_triple(model: &mut TransE, s: &str, p: &str, o: &str) -> Result<()> {
    model.add_triple(Triple::new(
        NamedNode::new(s)?,
        NamedNode::new(p)?,
        NamedNode::new(o)?,
    ))
}

fn save_to_csv(
    filename: &str,
    result: &oxirs_embed::visualization::VisualizationResult,
) -> Result<()> {
    let mut file = File::create(filename)?;

    // Write header
    if result.dimensions == 2 {
        writeln!(file, "entity,x,y")?;
    } else {
        writeln!(file, "entity,x,y,z")?;
    }

    // Write data
    for (entity, coords) in &result.coordinates {
        if coords.len() == 2 {
            writeln!(file, "{},{},{}", entity, coords[0], coords[1])?;
        } else if coords.len() == 3 {
            writeln!(file, "{},{},{},{}", entity, coords[0], coords[1], coords[2])?;
        }
    }

    Ok(())
}

fn save_clustered_to_csv(
    filename: &str,
    result: &oxirs_embed::visualization::VisualizationResult,
    clusters: &HashMap<String, usize>,
) -> Result<()> {
    let mut file = File::create(filename)?;

    // Write header
    writeln!(file, "entity,x,y,cluster_id")?;

    // Write data
    for (entity, coords) in &result.coordinates {
        let cluster_id = clusters.get(entity).unwrap_or(&0);
        writeln!(
            file,
            "{},{},{},{}",
            entity, coords[0], coords[1], cluster_id
        )?;
    }

    Ok(())
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
