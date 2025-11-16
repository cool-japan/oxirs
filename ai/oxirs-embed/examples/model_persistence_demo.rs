//! Model Persistence Demo
//!
//! This example demonstrates how to save and load trained embedding models,
//! enabling:
//! - Model checkpoint saving during training
//! - Loading pre-trained models for inference
//! - Model versioning and management
//! - Cross-session model reuse
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example model_persistence_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    link_prediction::{LinkPredictionConfig, LinkPredictor},
    models::hole::{HoLE, HoLEConfig},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};
use std::env::temp_dir;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║         Model Persistence Demo - Save & Load           ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Setup temporary directory for models
    let model_dir = temp_dir().join("oxirs_embed_models");
    std::fs::create_dir_all(&model_dir)?;

    // ====================
    // Demo 1: TransE Model Persistence
    // ====================
    println!("📦 Demo 1: TransE Model Persistence");
    println!("─────────────────────────────────────────────────────────");

    demonstrate_transe_persistence(&model_dir).await?;

    // ====================
    // Demo 2: HolE Model Persistence
    // ====================
    println!("\n📦 Demo 2: HolE Model Persistence");
    println!("─────────────────────────────────────────────────────────");

    demonstrate_hole_persistence(&model_dir).await?;

    // ====================
    // Demo 3: Model Checkpointing During Training
    // ====================
    println!("\n📦 Demo 3: Model Checkpointing During Training");
    println!("─────────────────────────────────────────────────────────");

    demonstrate_checkpointing(&model_dir).await?;

    // ====================
    // Demo 4: Pre-trained Model for Inference
    // ====================
    println!("\n📦 Demo 4: Using Pre-trained Models for Inference");
    println!("─────────────────────────────────────────────────────────");

    demonstrate_pretrained_inference(&model_dir).await?;

    // Cleanup
    println!("\n🧹 Cleaning up temporary files...");
    std::fs::remove_dir_all(&model_dir).ok();

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║          Model Persistence Demo Complete! ✓            ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Takeaways:");
    println!("  • Models can be saved and loaded for reuse");
    println!("  • Embeddings and model state are fully preserved");
    println!("  • Checkpointing enables resuming training");
    println!("  • Pre-trained models speed up deployment");
    println!();
    println!("Use Cases:");
    println!("  • Production deployment of trained models");
    println!("  • Model versioning and A/B testing");
    println!("  • Transfer learning from pre-trained models");
    println!("  • Collaborative model sharing");
    println!();

    Ok(())
}

/// Demonstrate TransE model persistence
async fn demonstrate_transe_persistence(model_dir: &Path) -> Result<()> {
    println!("  1. Training new TransE model...");

    let config = ModelConfig {
        dimensions: 100,
        learning_rate: 0.01,
        max_epochs: 50,
        ..Default::default()
    };

    let mut model = TransE::new(config);

    // Add knowledge graph triples
    add_sample_triples(&mut model)?;

    let stats = model.train(Some(50)).await?;
    println!("     Training complete: loss = {:.4}", stats.final_loss);

    // Get embedding before save
    let emb_before = model.get_entity_embedding("paris")?;
    let score_before = model.score_triple("paris", "capital_of", "france")?;

    println!("\n  2. Saving model to disk...");
    let model_path = model_dir.join("transe_model.bin");
    model.save(model_path.to_str().unwrap())?;

    let file_size = std::fs::metadata(&model_path)?.len();
    println!(
        "     Model saved: {} ({} KB)",
        model_path.display(),
        file_size / 1024
    );

    println!("\n  3. Loading model from disk...");
    let mut loaded_model = TransE::new(ModelConfig::default());
    loaded_model.load(model_path.to_str().unwrap())?;

    println!("     Model loaded successfully");
    println!("     Entities: {}", loaded_model.get_entities().len());
    println!("     Relations: {}", loaded_model.get_relations().len());
    println!("     Is trained: {}", loaded_model.is_trained());

    // Verify embeddings are preserved
    let emb_after = loaded_model.get_entity_embedding("paris")?;
    let score_after = loaded_model.score_triple("paris", "capital_of", "france")?;

    println!("\n  4. Verification:");
    println!(
        "     Embedding preservation: {}",
        if vectors_equal(&emb_before.values, &emb_after.values) {
            "✓ Identical"
        } else {
            "✗ Different"
        }
    );
    println!("     Score before save: {:.6}", score_before);
    println!("     Score after load:  {:.6}", score_after);
    println!("     Difference: {:.8}", (score_before - score_after).abs());

    Ok(())
}

/// Demonstrate HolE model persistence
async fn demonstrate_hole_persistence(model_dir: &Path) -> Result<()> {
    println!("  1. Training new HolE model...");

    let config = HoLEConfig {
        base: ModelConfig {
            dimensions: 80,
            learning_rate: 0.01,
            max_epochs: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut model = HoLE::new(config);

    // Add knowledge graph triples
    add_sample_triples(&mut model)?;

    let stats = model.train(Some(60)).await?;
    println!("     Training complete: loss = {:.4}", stats.final_loss);

    println!("\n  2. Saving HolE model...");
    let model_path = model_dir.join("hole_model.bin");
    model.save(model_path.to_str().unwrap())?;

    let file_size = std::fs::metadata(&model_path)?.len();
    println!("     Model saved: {} KB", file_size / 1024);

    println!("\n  3. Loading HolE model...");
    let mut loaded_model = HoLE::new(HoLEConfig::default());
    loaded_model.load(model_path.to_str().unwrap())?;

    println!("     Model loaded successfully");

    // Test predictions using LinkPredictor
    let pred_config = LinkPredictionConfig::default();
    let predictor = LinkPredictor::new(pred_config, loaded_model);

    let candidates = vec![
        "france".to_string(),
        "germany".to_string(),
        "uk".to_string(),
    ];

    let predictions = predictor.predict_tail("paris", "capital_of", &candidates)?;

    println!("\n  4. Prediction test (paris, capital_of, ?):");
    for pred in predictions.iter().take(3) {
        println!(
            "     → {} (score: {:.3}, rank: {})",
            pred.predicted_id, pred.score, pred.rank
        );
    }

    Ok(())
}

/// Demonstrate model checkpointing during training
async fn demonstrate_checkpointing(model_dir: &Path) -> Result<()> {
    println!("  1. Training with checkpoints...");

    let config = ModelConfig {
        dimensions: 64,
        learning_rate: 0.01,
        max_epochs: 100,
        ..Default::default()
    };

    let mut model = TransE::new(config);
    add_sample_triples(&mut model)?;

    // Train for 30 epochs and checkpoint
    let checkpoint1_path = model_dir.join("checkpoint_epoch_30.bin");
    model.train(Some(30)).await?;
    model.save(checkpoint1_path.to_str().unwrap())?;
    println!("     Checkpoint 1 saved at epoch 30");

    // Continue training for another 30 epochs
    let checkpoint2_path = model_dir.join("checkpoint_epoch_60.bin");
    model.train(Some(30)).await?;
    model.save(checkpoint2_path.to_str().unwrap())?;
    println!("     Checkpoint 2 saved at epoch 60");

    // Final training
    let final_path = model_dir.join("final_model.bin");
    model.train(Some(40)).await?;
    model.save(final_path.to_str().unwrap())?;
    println!("     Final model saved at epoch 100");

    println!("\n  2. Comparing checkpoints:");
    let mut checkpoint1 = TransE::new(ModelConfig::default());
    checkpoint1.load(checkpoint1_path.to_str().unwrap())?;

    let mut checkpoint2 = TransE::new(ModelConfig::default());
    checkpoint2.load(checkpoint2_path.to_str().unwrap())?;

    let mut final_model = TransE::new(ModelConfig::default());
    final_model.load(final_path.to_str().unwrap())?;

    let score1 = checkpoint1.score_triple("paris", "capital_of", "france")?;
    let score2 = checkpoint2.score_triple("paris", "capital_of", "france")?;
    let score_final = final_model.score_triple("paris", "capital_of", "france")?;

    println!("     Epoch 30 score:  {:.6}", score1);
    println!("     Epoch 60 score:  {:.6}", score2);
    println!("     Epoch 100 score: {:.6}", score_final);
    println!(
        "     Training improved score by: {:.6}",
        (score_final - score1).abs()
    );

    Ok(())
}

/// Demonstrate using pre-trained models for inference
async fn demonstrate_pretrained_inference(model_dir: &Path) -> Result<()> {
    println!("  1. Loading pre-trained model...");

    let model_path = model_dir.join("final_model.bin");
    let mut model = TransE::new(ModelConfig::default());
    model.load(model_path.to_str().unwrap())?;

    println!("     Pre-trained model loaded");
    println!("     Ready for inference without training");

    println!("\n  2. Performing inference tasks:");

    // Task 1: Entity similarity
    println!("\n     Task 1: Entity similarity");
    let paris_emb = model.get_entity_embedding("paris")?;
    let london_emb = model.get_entity_embedding("london")?;
    let berlin_emb = model.get_entity_embedding("berlin")?;

    let sim_paris_london = cosine_similarity(&paris_emb.values, &london_emb.values);
    let sim_paris_berlin = cosine_similarity(&paris_emb.values, &berlin_emb.values);

    println!("       Similarity(paris, london): {:.3}", sim_paris_london);
    println!("       Similarity(paris, berlin): {:.3}", sim_paris_berlin);

    // Task 2: Triple scoring
    println!("\n     Task 2: Triple scoring");
    let score1 = model.score_triple("paris", "capital_of", "france")?;
    let score2 = model.score_triple("berlin", "capital_of", "germany")?;
    let score3 = model.score_triple("paris", "capital_of", "germany")?; // Wrong triple

    println!("       (paris, capital_of, france):  {:.3} ✓", score1);
    println!("       (berlin, capital_of, germany): {:.3} ✓", score2);
    println!("       (paris, capital_of, germany):  {:.3} ✗", score3);

    // Task 3: Link prediction
    println!("\n     Task 3: Link prediction");

    let pred_config = LinkPredictionConfig::default();
    let predictor = LinkPredictor::new(pred_config, model);

    let candidates = vec![
        "france".to_string(),
        "germany".to_string(),
        "uk".to_string(),
        "italy".to_string(),
    ];

    let predictions = predictor.predict_tail("rome", "capital_of", &candidates)?;
    println!("       Query: (rome, capital_of, ?)");
    for pred in predictions.iter().take(2) {
        println!(
            "         → {} (score: {:.3})",
            pred.predicted_id, pred.score
        );
    }

    Ok(())
}

/// Add sample knowledge graph triples
fn add_sample_triples<M: EmbeddingModel>(model: &mut M) -> Result<()> {
    // European capitals
    model.add_triple(Triple::new(
        NamedNode::new("paris")?,
        NamedNode::new("capital_of")?,
        NamedNode::new("france")?,
    ))?;

    model.add_triple(Triple::new(
        NamedNode::new("london")?,
        NamedNode::new("capital_of")?,
        NamedNode::new("uk")?,
    ))?;

    model.add_triple(Triple::new(
        NamedNode::new("berlin")?,
        NamedNode::new("capital_of")?,
        NamedNode::new("germany")?,
    ))?;

    model.add_triple(Triple::new(
        NamedNode::new("rome")?,
        NamedNode::new("capital_of")?,
        NamedNode::new("italy")?,
    ))?;

    // Country locations
    model.add_triple(Triple::new(
        NamedNode::new("france")?,
        NamedNode::new("located_in")?,
        NamedNode::new("europe")?,
    ))?;

    model.add_triple(Triple::new(
        NamedNode::new("germany")?,
        NamedNode::new("located_in")?,
        NamedNode::new("europe")?,
    ))?;

    model.add_triple(Triple::new(
        NamedNode::new("uk")?,
        NamedNode::new("located_in")?,
        NamedNode::new("europe")?,
    ))?;

    model.add_triple(Triple::new(
        NamedNode::new("italy")?,
        NamedNode::new("located_in")?,
        NamedNode::new("europe")?,
    ))?;

    // City properties
    model.add_triple(Triple::new(
        NamedNode::new("paris")?,
        NamedNode::new("has_landmark")?,
        NamedNode::new("eiffel_tower")?,
    ))?;

    model.add_triple(Triple::new(
        NamedNode::new("london")?,
        NamedNode::new("has_landmark")?,
        NamedNode::new("big_ben")?,
    ))?;

    Ok(())
}

/// Check if two vectors are equal within tolerance
fn vectors_equal(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}
