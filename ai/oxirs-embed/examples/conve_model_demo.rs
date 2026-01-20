//! ConvE (Convolutional Embeddings) Model Demonstration
//!
//! This example demonstrates the usage of the ConvE model for knowledge graph embeddings,
//! including training, link prediction, and comparison with other models.
//!
//! # ConvE Model
//!
//! ConvE uses 2D convolutional neural networks to model entity and relation interactions.
//! It's particularly effective for:
//! - Learning complex interaction patterns through convolution
//! - Parameter-efficient representation with shared convolutional filters
//! - Capturing local and global features in embeddings
//! - Handling large-scale knowledge graphs
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example conve_model_demo --features conve
//! ```

use anyhow::Result;
use oxirs_embed::{
    clustering::{ClusteringAlgorithm, ClusteringConfig, EntityClustering},
    link_prediction::{LinkPredictionConfig, LinkPredictor},
    models::conve::{ConvE, ConvEConfig},
    EmbeddingModel, ModelConfig, NamedNode, Triple,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   ConvE Model Demo - Convolutional KG Embeddings       ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Step 1: Create and configure ConvE model
    println!("📊 Step 1: Configuring ConvE Model");
    println!("─────────────────────────────────────────────────────────");

    let config = ConvEConfig {
        base: ModelConfig {
            dimensions: 200, // Must be compatible with reshape_width
            learning_rate: 0.001,
            max_epochs: 100,
            batch_size: 32,
            ..Default::default()
        },
        reshape_width: 20, // 200 / 20 = 10 height
        num_filters: 32,
        kernel_size: 3,
        dropout_rate: 0.3,
        regularization: 0.0001,
        margin: 1.0,
        num_negatives: 5,
        use_batch_norm: true,
    };

    println!("  Dimensions: {}", config.base.dimensions);
    println!(
        "  Reshape: {}x{}",
        config.base.dimensions / config.reshape_width,
        config.reshape_width
    );
    println!("  Learning rate: {}", config.base.learning_rate);
    println!("  Max epochs: {}", config.base.max_epochs);
    println!("  Num filters: {}", config.num_filters);
    println!(
        "  Kernel size: {}x{}",
        config.kernel_size, config.kernel_size
    );
    println!("  Dropout rate: {}", config.dropout_rate);
    println!();

    let mut model = ConvE::new(config);

    // Step 2: Add knowledge graph triples
    println!("📚 Step 2: Building Family Relationships Knowledge Graph");
    println!("─────────────────────────────────────────────────────────");

    // Family relationships - more complex patterns for ConvE
    add_triple(&mut model, "john", "father_of", "mary")?;
    add_triple(&mut model, "john", "father_of", "alice")?;
    add_triple(&mut model, "susan", "mother_of", "mary")?;
    add_triple(&mut model, "susan", "mother_of", "alice")?;
    add_triple(&mut model, "john", "husband_of", "susan")?;
    add_triple(&mut model, "susan", "wife_of", "john")?;

    add_triple(&mut model, "bob", "father_of", "charlie")?;
    add_triple(&mut model, "bob", "father_of", "dave")?;
    add_triple(&mut model, "eve", "mother_of", "charlie")?;
    add_triple(&mut model, "eve", "mother_of", "dave")?;
    add_triple(&mut model, "bob", "husband_of", "eve")?;
    add_triple(&mut model, "eve", "wife_of", "bob")?;

    // Sibling relationships
    add_triple(&mut model, "mary", "sister_of", "alice")?;
    add_triple(&mut model, "alice", "sister_of", "mary")?;
    add_triple(&mut model, "charlie", "brother_of", "dave")?;
    add_triple(&mut model, "dave", "brother_of", "charlie")?;

    // Friend relationships
    add_triple(&mut model, "mary", "friend_of", "charlie")?;
    add_triple(&mut model, "charlie", "friend_of", "mary")?;
    add_triple(&mut model, "alice", "friend_of", "dave")?;
    add_triple(&mut model, "dave", "friend_of", "alice")?;

    // Work relationships
    add_triple(&mut model, "john", "works_with", "bob")?;
    add_triple(&mut model, "bob", "works_with", "john")?;
    add_triple(&mut model, "susan", "colleague_of", "eve")?;
    add_triple(&mut model, "eve", "colleague_of", "susan")?;

    // Professional attributes
    add_triple(&mut model, "john", "profession", "engineer")?;
    add_triple(&mut model, "bob", "profession", "engineer")?;
    add_triple(&mut model, "susan", "profession", "doctor")?;
    add_triple(&mut model, "eve", "profession", "doctor")?;

    // Location relationships
    add_triple(&mut model, "john", "lives_in", "boston")?;
    add_triple(&mut model, "susan", "lives_in", "boston")?;
    add_triple(&mut model, "mary", "lives_in", "boston")?;
    add_triple(&mut model, "alice", "lives_in", "boston")?;
    add_triple(&mut model, "bob", "lives_in", "seattle")?;
    add_triple(&mut model, "eve", "lives_in", "seattle")?;
    add_triple(&mut model, "charlie", "lives_in", "seattle")?;
    add_triple(&mut model, "dave", "lives_in", "seattle")?;

    let stats = model.get_stats();
    println!("  Total entities: {}", stats.num_entities);
    println!("  Total relations: {}", stats.num_relations);
    println!("  Total triples: {}", stats.num_triples);
    println!();

    // Step 3: Train the model
    println!("🎓 Step 3: Training ConvE Model");
    println!("─────────────────────────────────────────────────────────");
    println!("  ConvE applies 2D convolutions to learn interaction patterns...");

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

    // Step 4: Triple Scoring
    println!("📊 Step 4: Triple Scoring");
    println!("─────────────────────────────────────────────────────────");

    let test_triples = vec![
        ("john", "father_of", "mary"),
        ("mary", "sister_of", "alice"),
        ("john", "works_with", "bob"),
        ("susan", "profession", "doctor"),
    ];

    for (s, p, o) in test_triples {
        if let Ok(score) = model.score_triple(s, p, o) {
            println!("  ({}, {}, {}): {:.4}", s, p, o, score);
        }
    }
    println!();

    // Step 5: Entity Clustering
    println!("🗂️  Step 5: Entity Clustering");
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
        num_clusters: 5, // People, Professions, Cities, Family roles, etc.
        max_iterations: 50,
        ..Default::default()
    };

    let mut clustering = EntityClustering::new(cluster_config);
    let cluster_result = clustering.cluster(&embeddings)?;

    println!("  Number of clusters: {}", cluster_result.centroids.len());
    println!("  Silhouette score: {:.3}", cluster_result.silhouette_score);
    println!("  Inertia: {:.3}", cluster_result.inertia);
    println!("  Iterations: {}", cluster_result.num_iterations);
    println!("\n  Sample cluster assignments:");

    // Show sample cluster assignments
    let mut cluster_samples: HashMap<usize, Vec<String>> = HashMap::new();
    for (entity, &cluster_id) in cluster_result.assignments.iter().take(10) {
        cluster_samples
            .entry(cluster_id)
            .or_default()
            .push(entity.clone());
    }

    for (cluster_id, entities) in cluster_samples.iter() {
        println!("    Cluster {}: {:?}", cluster_id, entities);
    }
    println!();

    // Step 6: Link Prediction
    println!("🔮 Step 6: Link Prediction");
    println!("─────────────────────────────────────────────────────────");

    let pred_config = LinkPredictionConfig {
        top_k: 3,
        filter_known_triples: false,
        min_confidence: 0.0,
        parallel: true,
        batch_size: 50,
    };

    let predictor = LinkPredictor::new(pred_config, model);

    // Query 1: Who are John's children?
    println!("  Query 1: (john, father_of, ?)");
    let candidates = vec![
        "mary".to_string(),
        "alice".to_string(),
        "charlie".to_string(),
        "dave".to_string(),
    ];
    let predictions = predictor.predict_tail("john", "father_of", &candidates)?;
    print_predictions(&predictions);

    // Query 2: Who lives in Boston?
    println!("\n  Query 2: (?, lives_in, boston)");
    let person_candidates = vec![
        "john".to_string(),
        "susan".to_string(),
        "mary".to_string(),
        "alice".to_string(),
        "bob".to_string(),
        "eve".to_string(),
    ];
    let predictions = predictor.predict_head("lives_in", "boston", &person_candidates)?;
    print_predictions(&predictions);

    // Query 3: What is the relationship between Mary and Alice?
    println!("\n  Query 3: (mary, ?, alice)");
    let relation_candidates = vec![
        "sister_of".to_string(),
        "friend_of".to_string(),
        "mother_of".to_string(),
        "colleague_of".to_string(),
    ];
    let predictions = predictor.predict_relation("mary", "alice", &relation_candidates)?;
    print_predictions(&predictions);

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║              Demo Completed Successfully!              ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    println!("📖 Key Takeaways:");
    println!("  • ConvE uses 2D convolutions for expressive embeddings");
    println!("  • Effective for capturing complex interaction patterns");
    println!("  • Parameter-efficient through convolutional weight sharing");
    println!("  • Good performance on link prediction tasks");
    println!("  • Can cluster entities into meaningful semantic groups");
    println!();

    Ok(())
}

fn add_triple(model: &mut ConvE, s: &str, p: &str, o: &str) -> Result<()> {
    model.add_triple(Triple::new(
        NamedNode::new(s)?,
        NamedNode::new(p)?,
        NamedNode::new(o)?,
    ))
}

fn print_predictions(predictions: &[oxirs_embed::link_prediction::LinkPrediction]) {
    for (i, pred) in predictions.iter().enumerate() {
        println!(
            "    {}. {} (score: {:.4}, rank: {})",
            i + 1,
            pred.predicted_id,
            pred.score,
            pred.rank
        );
    }
}
