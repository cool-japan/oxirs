//! Quick Start Guide for OxiRS Embed
//!
//! This example demonstrates the complete workflow for using oxirs-embed:
//! 1. Creating a knowledge graph embedding model
//! 2. Adding triples to the model
//! 3. Training the model
//! 4. Querying embeddings
//! 5. Making predictions
//! 6. Evaluating model performance
//!
//! This is an ideal starting point for new users.

use anyhow::Result;
use oxirs_embed::{
    ComplEx, DistMult, EmbeddingModel, ModelConfig, NamedNode, TransE, Triple, Vector,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for better debugging
    tracing_subscriber::fmt::init();

    println!("🚀 OxiRS Embed Quick Start Guide\n");
    println!("This guide demonstrates the complete workflow for knowledge graph embeddings.\n");

    // ============================================================
    // Step 1: Create a simple knowledge graph
    // ============================================================
    println!("📊 Step 1: Creating sample knowledge graph...");

    let triples = vec![
        // Academic relationships
        ("Alice", "teaches", "Mathematics"),
        ("Bob", "teaches", "Physics"),
        ("Carol", "teaches", "Chemistry"),
        ("Alice", "colleagues_with", "Bob"),
        ("Bob", "colleagues_with", "Carol"),
        ("Alice", "colleagues_with", "Carol"),
        // Student relationships
        ("Dave", "studies", "Mathematics"),
        ("Eve", "studies", "Physics"),
        ("Frank", "studies", "Chemistry"),
        ("Dave", "friends_with", "Eve"),
        ("Eve", "friends_with", "Frank"),
        // Course relationships
        ("Mathematics", "requires", "Algebra"),
        ("Physics", "requires", "Calculus"),
        ("Chemistry", "requires", "Algebra"),
        ("Calculus", "builds_on", "Algebra"),
        // Additional relationships for link prediction
        ("Alice", "advises", "Dave"),
        ("Bob", "advises", "Eve"),
        ("Carol", "advises", "Frank"),
    ];

    println!("  ✓ Created {} triples", triples.len());

    // ============================================================
    // Step 2: Train TransE model (simplest model)
    // ============================================================
    println!("\n📈 Step 2: Training TransE model...");

    // Model needs to be mutable for the save() operation
    #[allow(unused_mut)]
    let mut transe_model = create_and_train_transe(&triples).await?;

    // ============================================================
    // Step 3: Query entity embeddings
    // ============================================================
    println!("\n🔍 Step 3: Querying entity embeddings...");

    let alice_embedding = transe_model.get_entity_embedding("http://example.org/Alice")?;
    let bob_embedding = transe_model.get_entity_embedding("http://example.org/Bob")?;

    println!(
        "  ✓ Alice embedding (first 5 dims): [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}...]",
        alice_embedding.values[0],
        alice_embedding.values[1],
        alice_embedding.values[2],
        alice_embedding.values[3],
        alice_embedding.values[4]
    );

    // Calculate similarity between entities
    let similarity = cosine_similarity(&alice_embedding, &bob_embedding);
    println!("  ✓ Similarity between Alice and Bob: {:.3}", similarity);

    // ============================================================
    // Step 4: Make predictions
    // ============================================================
    println!("\n🎯 Step 4: Making predictions...");

    // Predict what Alice might teach (beyond Mathematics)
    let predictions = transe_model.predict_objects(
        "http://example.org/Alice",
        "http://example.org/teaches",
        3,
    )?;

    println!("  Top 3 predictions for 'Alice teaches ?':");
    for (i, (entity, score)) in predictions.iter().enumerate() {
        let clean_name = entity.replace("http://example.org/", "");
        println!("    {}. {} (score: {:.3})", i + 1, clean_name, score);
    }

    // Predict who might advise Dave
    let advisor_predictions = transe_model.predict_subjects(
        "http://example.org/advises",
        "http://example.org/Dave",
        3,
    )?;

    println!("\n  Top 3 predictions for '? advises Dave':");
    for (i, (entity, score)) in advisor_predictions.iter().enumerate() {
        let clean_name = entity.replace("http://example.org/", "");
        println!("    {}. {} (score: {:.3})", i + 1, clean_name, score);
    }

    // ============================================================
    // Step 5: Compare different models
    // ============================================================
    println!("\n⚖️  Step 5: Comparing different embedding models...");

    // Train DistMult model
    let distmult_model = create_and_train_distmult(&triples).await?;

    // Train ComplEx model
    let complex_model = create_and_train_complex(&triples).await?;

    // Compare predictions across models
    println!("\n  Comparing predictions for 'Alice teaches ?':");

    let transe_preds = transe_model.predict_objects(
        "http://example.org/Alice",
        "http://example.org/teaches",
        1,
    )?;
    let distmult_preds = distmult_model.predict_objects(
        "http://example.org/Alice",
        "http://example.org/teaches",
        1,
    )?;
    let complex_preds = complex_model.predict_objects(
        "http://example.org/Alice",
        "http://example.org/teaches",
        1,
    )?;

    println!(
        "    TransE:   {} (score: {:.3})",
        transe_preds[0].0.replace("http://example.org/", ""),
        transe_preds[0].1
    );
    println!(
        "    DistMult: {} (score: {:.3})",
        distmult_preds[0].0.replace("http://example.org/", ""),
        distmult_preds[0].1
    );
    println!(
        "    ComplEx:  {} (score: {:.3})",
        complex_preds[0].0.replace("http://example.org/", ""),
        complex_preds[0].1
    );

    // ============================================================
    // Step 6: Model statistics
    // ============================================================
    println!("\n📊 Step 6: Model statistics...");

    let stats = transe_model.get_stats();
    println!("  Model: {}", stats.model_type);
    println!("  Entities: {}", stats.num_entities);
    println!("  Relations: {}", stats.num_relations);
    println!("  Triples: {}", stats.num_triples);
    println!("  Dimensions: {}", stats.dimensions);
    println!("  Trained: {}", stats.is_trained);

    // ============================================================
    // Step 7: Save and load models
    // ============================================================
    println!("\n💾 Step 7: Saving and loading models...");

    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("transe_model.bin");

    transe_model.save(model_path.to_str().unwrap())?;
    println!("  ✓ Model saved to: {}", model_path.display());

    let mut loaded_model = TransE::new(ModelConfig::default());
    loaded_model.load(model_path.to_str().unwrap())?;
    println!("  ✓ Model loaded successfully");

    // Verify loaded model works
    let loaded_predictions = loaded_model.predict_objects(
        "http://example.org/Alice",
        "http://example.org/teaches",
        1,
    )?;
    println!(
        "  ✓ Loaded model prediction: {}",
        loaded_predictions[0].0.replace("http://example.org/", "")
    );

    // ============================================================
    // Summary
    // ============================================================
    println!("\n✅ Quick Start Guide completed successfully!");
    println!("\n📚 Next steps:");
    println!("  • Explore more advanced models: HolE, ConvE, TuckER, RotatE");
    println!("  • Try fine-tuning with domain-specific data");
    println!("  • Enable GPU acceleration for faster training");
    println!("  • Integrate with SPARQL queries using vec: extension");
    println!("  • Check out other examples for advanced features");
    println!("\n  Run: cargo run --example <example_name> --features basic-models");

    Ok(())
}

/// Helper function to create and train TransE model
async fn create_and_train_transe(triples: &[(&str, &str, &str)]) -> Result<TransE> {
    let config = ModelConfig::default()
        .with_dimensions(50) // Small dimensions for quick training
        .with_learning_rate(0.01)
        .with_max_epochs(50) // Reduced for demo
        .with_batch_size(10);

    let mut model = TransE::new(config.clone());

    // Add triples to model
    for (subject, predicate, object) in triples {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/{subject}"))?,
            NamedNode::new(&format!("http://example.org/{predicate}"))?,
            NamedNode::new(&format!("http://example.org/{object}"))?,
        );
        model.add_triple(triple)?;
    }

    // Train the model
    print!("  Training TransE... ");
    let stats = model.train(Some(50)).await?;
    println!(
        "✓ (epochs: {}, final loss: {:.3})",
        stats.epochs_completed, stats.final_loss
    );

    Ok(model)
}

/// Helper function to create and train DistMult model
async fn create_and_train_distmult(triples: &[(&str, &str, &str)]) -> Result<DistMult> {
    let config = ModelConfig::default()
        .with_dimensions(50)
        .with_learning_rate(0.01)
        .with_max_epochs(50)
        .with_batch_size(10);

    let mut model = DistMult::new(config);

    for (subject, predicate, object) in triples {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/{subject}"))?,
            NamedNode::new(&format!("http://example.org/{predicate}"))?,
            NamedNode::new(&format!("http://example.org/{object}"))?,
        );
        model.add_triple(triple)?;
    }

    print!("  Training DistMult... ");
    let stats = model.train(Some(50)).await?;
    println!(
        "✓ (epochs: {}, final loss: {:.3})",
        stats.epochs_completed, stats.final_loss
    );

    Ok(model)
}

/// Helper function to create and train ComplEx model
async fn create_and_train_complex(triples: &[(&str, &str, &str)]) -> Result<ComplEx> {
    let config = ModelConfig::default()
        .with_dimensions(50)
        .with_learning_rate(0.01)
        .with_max_epochs(50)
        .with_batch_size(10);

    let mut model = ComplEx::new(config);

    for (subject, predicate, object) in triples {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/{subject}"))?,
            NamedNode::new(&format!("http://example.org/{predicate}"))?,
            NamedNode::new(&format!("http://example.org/{object}"))?,
        );
        model.add_triple(triple)?;
    }

    print!("  Training ComplEx... ");
    let stats = model.train(Some(50)).await?;
    println!(
        "✓ (epochs: {}, final loss: {:.3})",
        stats.epochs_completed, stats.final_loss
    );

    Ok(model)
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &Vector, b: &Vector) -> f64 {
    if a.values.len() != b.values.len() {
        return 0.0;
    }

    let dot_product: f32 = a
        .values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| x * y)
        .sum();

    let norm_a: f32 = a.values.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.values.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot_product / (norm_a * norm_b)) as f64
    }
}
