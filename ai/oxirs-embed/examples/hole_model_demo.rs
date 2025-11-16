//! HolE (Holographic Embeddings) Model Demonstration
//!
//! This example demonstrates the usage of the HolE model for knowledge graph embeddings,
//! including training, link prediction, and entity clustering.
//!
//! # HolE Model
//!
//! HolE uses circular correlation to model entity and relation interactions.
//! It's particularly effective for:
//! - Capturing symmetric and asymmetric relations
//! - Handling complex relational patterns
//! - Efficient computation with circular correlation in Fourier domain
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example hole_model_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    clustering::{ClusteringAlgorithm, ClusteringConfig, EntityClustering},
    link_prediction::{LinkPredictionConfig, LinkPredictor},
    models::hole::{HoLE, HoLEConfig},
    EmbeddingModel, ModelConfig, NamedNode, Triple,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   HolE Model Demo - Knowledge Graph Embeddings         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Step 1: Create and configure HolE model
    println!("📊 Step 1: Configuring HolE Model");
    println!("─────────────────────────────────────────────────────────");

    let config = HoLEConfig {
        base: ModelConfig {
            dimensions: 100,
            learning_rate: 0.01,
            max_epochs: 100,
            batch_size: 50,
            ..Default::default()
        },
        regularization: 0.0001,
        margin: 1.0,
        num_negatives: 5,
        use_sigmoid: true,
    };

    println!("  Dimensions: {}", config.base.dimensions);
    println!("  Learning rate: {}", config.base.learning_rate);
    println!("  Max epochs: {}", config.base.max_epochs);
    println!("  Regularization: {}", config.regularization);
    println!("  Margin: {}", config.margin);
    println!();

    let mut model = HoLE::new(config);

    // Step 2: Add knowledge graph triples
    println!("📚 Step 2: Building Knowledge Graph");
    println!("─────────────────────────────────────────────────────────");

    // Geographic knowledge
    add_triple(&mut model, "paris", "capital_of", "france")?;
    add_triple(&mut model, "london", "capital_of", "uk")?;
    add_triple(&mut model, "berlin", "capital_of", "germany")?;
    add_triple(&mut model, "rome", "capital_of", "italy")?;

    add_triple(&mut model, "france", "in_continent", "europe")?;
    add_triple(&mut model, "uk", "in_continent", "europe")?;
    add_triple(&mut model, "germany", "in_continent", "europe")?;
    add_triple(&mut model, "italy", "in_continent", "europe")?;

    add_triple(&mut model, "paris", "located_in", "france")?;
    add_triple(&mut model, "london", "located_in", "uk")?;
    add_triple(&mut model, "berlin", "located_in", "germany")?;
    add_triple(&mut model, "rome", "located_in", "italy")?;

    // Cultural knowledge
    add_triple(&mut model, "france", "speaks", "french")?;
    add_triple(&mut model, "uk", "speaks", "english")?;
    add_triple(&mut model, "germany", "speaks", "german")?;
    add_triple(&mut model, "italy", "speaks", "italian")?;

    // Economic knowledge
    add_triple(&mut model, "france", "currency", "euro")?;
    add_triple(&mut model, "germany", "currency", "euro")?;
    add_triple(&mut model, "italy", "currency", "euro")?;
    add_triple(&mut model, "uk", "currency", "pound")?;

    let stats = model.get_stats();
    println!("  Total entities: {}", stats.num_entities);
    println!("  Total relations: {}", stats.num_relations);
    println!("  Total triples: {}", stats.num_triples);
    println!();

    // Step 3: Train the model
    println!("🎓 Step 3: Training HolE Model");
    println!("─────────────────────────────────────────────────────────");

    let training_stats = model.train(Some(100)).await?;

    println!("  Epochs completed: {}", training_stats.epochs_completed);
    println!("  Final loss: {:.4}", training_stats.final_loss);
    println!(
        "  Training time: {:.2}s",
        training_stats.training_time_seconds
    );
    println!(
        "  Convergence: {}",
        if training_stats.convergence_achieved {
            "✓ Achieved"
        } else {
            "✗ Not achieved"
        }
    );
    println!();

    // Step 4: Entity Clustering (before link prediction to keep model ownership)
    println!("🗂️  Step 4: Entity Clustering");
    println!("─────────────────────────────────────────────────────────");

    // Extract embeddings
    let mut embeddings = HashMap::new();
    for entity in model.get_entities() {
        if let Ok(emb) = model.get_entity_embedding(&entity) {
            let array = scirs2_core::ndarray_ext::Array1::from_vec(emb.values);
            embeddings.insert(entity, array);
        }
    }

    // Cluster entities
    let cluster_config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans,
        num_clusters: 4, // Countries, Cities, Languages, Currencies
        max_iterations: 50,
        ..Default::default()
    };

    let mut clustering = EntityClustering::new(cluster_config);
    let cluster_result = clustering.cluster(&embeddings)?;

    println!("  Number of clusters: {}", cluster_result.centroids.len());
    println!("  Silhouette score: {:.3}", cluster_result.silhouette_score);
    println!("  Inertia: {:.3}", cluster_result.inertia);
    println!("\n  Cluster assignments:");

    // Group entities by cluster
    let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
    for (entity, cluster_id) in cluster_result.assignments {
        clusters.entry(cluster_id).or_default().push(entity);
    }

    for (cluster_id, entities) in clusters.iter() {
        println!("    Cluster {}: {} entities", cluster_id, entities.len());
        for entity in entities.iter().take(5) {
            println!("      • {}", entity);
        }
        if entities.len() > 5 {
            println!("      ... and {} more", entities.len() - 5);
        }
    }
    println!();

    // Step 5: Embedding Analysis
    println!("📈 Step 5: Embedding Analysis");
    println!("─────────────────────────────────────────────────────────");

    // Compare entity similarity
    let paris_emb = model.get_entity_embedding("paris")?;
    let london_emb = model.get_entity_embedding("london")?;
    let euro_emb = model.get_entity_embedding("euro")?;

    let paris_london_sim = cosine_similarity(&paris_emb.values, &london_emb.values);
    let paris_euro_sim = cosine_similarity(&paris_emb.values, &euro_emb.values);

    println!("  Entity Similarities:");
    println!("    paris ↔ london: {:.3}", paris_london_sim);
    println!("    paris ↔ euro:   {:.3}", paris_euro_sim);
    println!();

    // Relation embeddings
    let capital_rel = model.get_relation_embedding("capital_of")?;
    let speaks_rel = model.get_relation_embedding("speaks")?;

    let rel_sim = cosine_similarity(&capital_rel.values, &speaks_rel.values);
    println!("  Relation Similarity:");
    println!("    capital_of ↔ speaks: {:.3}", rel_sim);
    println!();

    // Step 6: Link Prediction
    println!("🔮 Step 6: Link Prediction");
    println!("─────────────────────────────────────────────────────────");

    let pred_config = LinkPredictionConfig {
        top_k: 3,
        filter_known_triples: false,
        min_confidence: 0.0,
        ..Default::default()
    };

    let predictor = LinkPredictor::new(pred_config, model);

    // Predict capitals
    println!("\n  Query: (spain, capital_of, ?)");
    let city_candidates = vec![
        "madrid".to_string(),
        "paris".to_string(),
        "london".to_string(),
        "barcelona".to_string(),
    ];
    let predictions = predictor.predict_tail("spain", "capital_of", &city_candidates)?;

    for pred in &predictions {
        println!(
            "    → {} (score: {:.3}, confidence: {:.1}%, rank: {})",
            pred.predicted_id,
            pred.score,
            pred.confidence * 100.0,
            pred.rank
        );
    }

    // Predict countries by language
    println!("\n  Query: (?, speaks, french)");
    let country_candidates = vec![
        "france".to_string(),
        "germany".to_string(),
        "belgium".to_string(),
    ];
    let predictions = predictor.predict_head("speaks", "french", &country_candidates)?;

    for pred in &predictions {
        println!(
            "    → {} (score: {:.3}, confidence: {:.1}%, rank: {})",
            pred.predicted_id,
            pred.score,
            pred.confidence * 100.0,
            pred.rank
        );
    }

    // Predict relations
    println!("\n  Query: (paris, ?, france)");
    let relation_candidates = vec![
        "located_in".to_string(),
        "capital_of".to_string(),
        "speaks".to_string(),
    ];
    let predictions = predictor.predict_relation("paris", "france", &relation_candidates)?;

    for pred in &predictions {
        println!(
            "    → {} (score: {:.3}, confidence: {:.1}%, rank: {})",
            pred.predicted_id,
            pred.score,
            pred.confidence * 100.0,
            pred.rank
        );
    }
    println!();

    // Summary
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║                   Demo Complete! ✓                      ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Takeaways:");
    println!("  • HolE effectively learns geographic and cultural relationships");
    println!("  • Link prediction identifies plausible missing facts");
    println!("  • Entity clustering discovers semantic groups");
    println!("  • Embedding similarity reflects semantic relatedness");
    println!();

    Ok(())
}

/// Helper function to add a triple
fn add_triple(model: &mut HoLE, subject: &str, predicate: &str, object: &str) -> Result<()> {
    model.add_triple(Triple::new(
        NamedNode::new(subject)?,
        NamedNode::new(predicate)?,
        NamedNode::new(object)?,
    ))
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}
