//! Vector Search Demo
//!
//! This example demonstrates high-performance vector search capabilities for knowledge
//! graph embeddings, including:
//! - Building searchable indexes from embeddings
//! - Semantic similarity search
//! - Multiple distance metrics (cosine, Euclidean, dot product)
//! - Batch search operations
//! - Radius-based filtering
//! - Real-time nearest neighbor queries
//!
//! # Vector Search Use Cases
//!
//! - **Semantic Search**: Find entities by meaning, not just keywords
//! - **Recommendation Systems**: Suggest similar items based on embeddings
//! - **Duplicate Detection**: Find near-duplicate entities in knowledge graphs
//! - **Anomaly Detection**: Identify outliers in embedding space
//! - **Query Expansion**: Discover related concepts for search queries
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example vector_search_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    vector_search::{DistanceMetric, SearchConfig, VectorSearchIndex},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   Vector Search Demo - Semantic Similarity Search      ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // ====================
    // Step 1: Build Product Knowledge Graph
    // ====================
    println!("📚 Step 1: Building E-Commerce Product Knowledge Graph");
    println!("─────────────────────────────────────────────────────────");

    let config = ModelConfig {
        dimensions: 128,
        learning_rate: 0.01,
        max_epochs: 200,
        ..Default::default()
    };

    let mut model = TransE::new(config);

    // Electronics
    println!("  Adding electronics products...");
    add_triple(&mut model, "iphone_14", "category", "smartphones")?;
    add_triple(&mut model, "samsung_galaxy", "category", "smartphones")?;
    add_triple(&mut model, "pixel_7", "category", "smartphones")?;

    add_triple(&mut model, "macbook_pro", "category", "laptops")?;
    add_triple(&mut model, "dell_xps", "category", "laptops")?;
    add_triple(&mut model, "thinkpad_x1", "category", "laptops")?;

    add_triple(&mut model, "airpods_pro", "category", "audio")?;
    add_triple(&mut model, "sony_wh1000", "category", "audio")?;
    add_triple(&mut model, "bose_qc45", "category", "audio")?;

    // Product features
    add_triple(&mut model, "iphone_14", "has_feature", "camera")?;
    add_triple(&mut model, "samsung_galaxy", "has_feature", "camera")?;
    add_triple(&mut model, "pixel_7", "has_feature", "camera")?;

    add_triple(&mut model, "macbook_pro", "has_feature", "high_performance")?;
    add_triple(&mut model, "dell_xps", "has_feature", "high_performance")?;

    add_triple(&mut model, "airpods_pro", "has_feature", "noise_canceling")?;
    add_triple(&mut model, "sony_wh1000", "has_feature", "noise_canceling")?;
    add_triple(&mut model, "bose_qc45", "has_feature", "noise_canceling")?;

    // Brand relationships
    add_triple(&mut model, "iphone_14", "brand", "apple")?;
    add_triple(&mut model, "macbook_pro", "brand", "apple")?;
    add_triple(&mut model, "airpods_pro", "brand", "apple")?;

    add_triple(&mut model, "samsung_galaxy", "brand", "samsung")?;
    add_triple(&mut model, "pixel_7", "brand", "google")?;
    add_triple(&mut model, "dell_xps", "brand", "dell")?;
    add_triple(&mut model, "thinkpad_x1", "brand", "lenovo")?;

    // Price ranges
    add_triple(&mut model, "iphone_14", "price_range", "premium")?;
    add_triple(&mut model, "macbook_pro", "price_range", "premium")?;
    add_triple(&mut model, "samsung_galaxy", "price_range", "premium")?;
    add_triple(&mut model, "dell_xps", "price_range", "premium")?;

    add_triple(&mut model, "pixel_7", "price_range", "mid_range")?;
    add_triple(&mut model, "thinkpad_x1", "price_range", "mid_range")?;

    // Complementary products
    add_triple(&mut model, "iphone_14", "works_with", "airpods_pro")?;
    add_triple(&mut model, "macbook_pro", "works_with", "airpods_pro")?;
    add_triple(&mut model, "samsung_galaxy", "works_with", "sony_wh1000")?;

    let stats = model.get_stats();
    println!("  Total entities: {}", stats.num_entities);
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
    println!("🔢 Step 3: Extracting Product Embeddings");
    println!("─────────────────────────────────────────────────────────");

    let mut embeddings = HashMap::new();
    for entity in model.get_entities() {
        if let Ok(emb) = model.get_entity_embedding(&entity) {
            let array = scirs2_core::ndarray_ext::Array1::from_vec(emb.values);
            embeddings.insert(entity, array);
        }
    }

    println!("  Extracted {} embeddings", embeddings.len());
    println!();

    // ====================
    // Step 4: Build Vector Search Index
    // ====================
    println!("🔍 Step 4: Building Vector Search Index");
    println!("─────────────────────────────────────────────────────────");

    let search_config = SearchConfig {
        metric: DistanceMetric::Cosine,
        use_approximate: false,
        parallel: true,
        normalize: true,
        ..Default::default()
    };

    let mut index = VectorSearchIndex::new(search_config);
    index.build(&embeddings)?;

    let index_stats = index.get_stats();
    println!("  Index built successfully!");
    println!("    Entities indexed: {}", index_stats.num_entities);
    println!("    Dimensions: {}", index_stats.dimensions);
    println!("    Distance metric: {:?}", index_stats.metric);
    println!();

    // ====================
    // Step 5: Semantic Product Search
    // ====================
    println!("🛍️  Step 5: Semantic Product Search");
    println!("─────────────────────────────────────────────────────────");

    // Query 1: Find similar products to iPhone 14
    println!("  Query 1: Products similar to 'iphone_14'");
    let iphone_embedding = embeddings["iphone_14"].to_vec();
    let results = index.search(&iphone_embedding, 5)?;

    for result in results {
        println!(
            "    {}. {} (similarity: {:.3}, distance: {:.3})",
            result.rank, result.entity_id, result.score, result.distance
        );
    }

    // Query 2: Find similar products to MacBook Pro
    println!("\n  Query 2: Products similar to 'macbook_pro'");
    let macbook_embedding = embeddings["macbook_pro"].to_vec();
    let results = index.search(&macbook_embedding, 5)?;

    for result in results {
        println!(
            "    {}. {} (similarity: {:.3}, distance: {:.3})",
            result.rank, result.entity_id, result.score, result.distance
        );
    }

    // Query 3: Find similar products to AirPods Pro
    println!("\n  Query 3: Products similar to 'airpods_pro'");
    let airpods_embedding = embeddings["airpods_pro"].to_vec();
    let results = index.search(&airpods_embedding, 5)?;

    for result in results {
        println!(
            "    {}. {} (similarity: {:.3}, distance: {:.3})",
            result.rank, result.entity_id, result.score, result.distance
        );
    }
    println!();

    // ====================
    // Step 6: Batch Search
    // ====================
    println!("⚡ Step 6: Batch Search (Multiple Queries)");
    println!("─────────────────────────────────────────────────────────");

    let queries = vec![
        embeddings["smartphones"].to_vec(),
        embeddings["laptops"].to_vec(),
        embeddings["audio"].to_vec(),
    ];

    let batch_results = index.batch_search(&queries, 3)?;

    println!("  Results for 'smartphones' category:");
    for result in &batch_results[0] {
        println!(
            "    • {} (similarity: {:.3})",
            result.entity_id, result.score
        );
    }

    println!("\n  Results for 'laptops' category:");
    for result in &batch_results[1] {
        println!(
            "    • {} (similarity: {:.3})",
            result.entity_id, result.score
        );
    }

    println!("\n  Results for 'audio' category:");
    for result in &batch_results[2] {
        println!(
            "    • {} (similarity: {:.3})",
            result.entity_id, result.score
        );
    }
    println!();

    // ====================
    // Step 7: Radius Search
    // ====================
    println!("🎯 Step 7: Radius-Based Search");
    println!("─────────────────────────────────────────────────────────");

    println!("  Finding all products within distance 0.3 of 'iphone_14':");
    let radius_results = index.radius_search(&iphone_embedding, 0.3)?;

    for result in radius_results {
        println!(
            "    • {} (distance: {:.3}, similarity: {:.3})",
            result.entity_id, result.distance, result.score
        );
    }
    println!();

    // ====================
    // Step 8: Compare Distance Metrics
    // ====================
    println!("📊 Step 8: Comparing Distance Metrics");
    println!("─────────────────────────────────────────────────────────");

    let metrics = vec![
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ];

    for metric in metrics {
        let config = SearchConfig {
            metric,
            normalize: metric == DistanceMetric::Cosine,
            ..Default::default()
        };

        let mut metric_index = VectorSearchIndex::new(config);
        metric_index.build(&embeddings)?;

        println!("\n  Using {:?} metric:", metric);
        let results = metric_index.search(&iphone_embedding, 3)?;

        for result in results {
            println!(
                "    • {} (score: {:.3}, distance: {:.3})",
                result.entity_id, result.score, result.distance
            );
        }
    }
    println!();

    // ====================
    // Step 9: Recommendation System Demo
    // ====================
    println!("💡 Step 9: Product Recommendation System");
    println!("─────────────────────────────────────────────────────────");

    println!("  User browsing history: ['iphone_14', 'macbook_pro']");
    println!("  Computing average embedding...");

    // Combine embeddings (simple average)
    let user_profile: Vec<f32> = (0..128)
        .map(|i| (embeddings["iphone_14"][i] + embeddings["macbook_pro"][i]) / 2.0)
        .collect();

    println!("  Recommended products based on browsing history:");
    let recommendations = index.search(&user_profile, 5)?;

    for result in recommendations {
        if result.entity_id != "iphone_14" && result.entity_id != "macbook_pro" {
            println!(
                "    • {} (relevance: {:.3})",
                result.entity_id, result.score
            );
        }
    }
    println!();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║           Vector Search Demo Completed!                ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    println!("💡 Key Capabilities:");
    println!("  • Fast semantic similarity search");
    println!("  • Multiple distance metrics supported");
    println!("  • Batch processing for efficiency");
    println!("  • Radius-based filtering");
    println!("  • Real-time recommendations");
    println!();

    println!("🚀 Performance Notes:");
    println!("  • Exact search: O(n) complexity, suitable for <100k entities");
    println!("  • Parallel processing enabled for multi-core systems");
    println!("  • Normalized vectors for cosine similarity");
    println!("  • Index built in memory for fast queries");
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
