//! Fine-Tuning Demo - Transfer Learning for Domain-Specific Knowledge Graphs
//!
//! This example demonstrates fine-tuning a pre-trained embedding model on domain-specific
//! data, showcasing transfer learning capabilities for knowledge graph embeddings.
//!
//! # Fine-Tuning Strategies
//!
//! 1. **Full Fine-Tuning**: Update all parameters
//! 2. **Freeze Entities**: Only update relation embeddings
//! 3. **Freeze Relations**: Only update entity embeddings
//! 4. **Partial Dimensions**: Fine-tune only top dimensions
//! 5. **Adapter-Based**: Add small adapter layers (parameter-efficient)
//! 6. **Discriminative**: Use different learning rates per layer
//!
//! # Use Cases
//!
//! - Adapting general medical knowledge to specific diseases
//! - Fine-tuning scientific embeddings for a specific domain
//! - Personalizing embeddings for specific applications
//! - Incremental learning with new knowledge
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example fine_tuning_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    fine_tuning::{FineTuningConfig, FineTuningManager, FineTuningStrategy},
    link_prediction::{LinkPredictionConfig, LinkPredictor},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Fine-Tuning Demo - Transfer Learning for Knowledge Graphs   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // Part 1: Pre-train on General Medical Knowledge
    // ========================================================================
    println!("📚 Part 1: Pre-training on General Medical Knowledge");
    println!("──────────────────────────────────────────────────────────────────");

    let pretrain_config = ModelConfig {
        dimensions: 100,
        learning_rate: 0.01,
        max_epochs: 50,
        batch_size: 32,
        ..Default::default()
    };

    let mut pretrained_model = TransE::new(pretrain_config);

    // General medical knowledge
    println!("  Adding general medical knowledge...");

    // Disease categories
    add_triple(&mut pretrained_model, "cancer", "is_a", "disease")?;
    add_triple(&mut pretrained_model, "diabetes", "is_a", "disease")?;
    add_triple(&mut pretrained_model, "hypertension", "is_a", "disease")?;
    add_triple(&mut pretrained_model, "asthma", "is_a", "disease")?;
    add_triple(&mut pretrained_model, "arthritis", "is_a", "disease")?;

    // Treatments
    add_triple(&mut pretrained_model, "chemotherapy", "treats", "cancer")?;
    add_triple(&mut pretrained_model, "insulin", "treats", "diabetes")?;
    add_triple(
        &mut pretrained_model,
        "beta_blocker",
        "treats",
        "hypertension",
    )?;
    add_triple(&mut pretrained_model, "inhaler", "treats", "asthma")?;

    // Drug categories
    add_triple(&mut pretrained_model, "aspirin", "is_a", "drug")?;
    add_triple(&mut pretrained_model, "metformin", "is_a", "drug")?;
    add_triple(&mut pretrained_model, "lisinopril", "is_a", "drug")?;

    // General relationships
    add_triple(
        &mut pretrained_model,
        "smoking",
        "risk_factor_for",
        "cancer",
    )?;
    add_triple(
        &mut pretrained_model,
        "obesity",
        "risk_factor_for",
        "diabetes",
    )?;
    add_triple(
        &mut pretrained_model,
        "stress",
        "risk_factor_for",
        "hypertension",
    )?;

    let stats = pretrained_model.get_stats();
    println!("  ✓ Entities: {}", stats.num_entities);
    println!("  ✓ Relations: {}", stats.num_relations);
    println!("  ✓ Triples: {}", stats.num_triples);
    println!();

    // Train pre-trained model
    println!("🎓 Training General Medical Model...");
    println!("──────────────────────────────────────────────────────────────────");

    let pretrain_stats = pretrained_model.train(Some(50)).await?;

    println!("  Pre-training Results:");
    println!("    • Epochs: {}", pretrain_stats.epochs_completed);
    println!("    • Final Loss: {:.4}", pretrain_stats.final_loss);
    println!("    • Time: {:.2}s", pretrain_stats.training_time_seconds);
    println!();

    // Test pre-trained model
    println!("🔍 Testing Pre-trained Model (Before Fine-Tuning)");
    println!("──────────────────────────────────────────────────────────────────");

    let pred_config = LinkPredictionConfig {
        top_k: 3,
        filter_known_triples: false,
        min_confidence: 0.0,
        ..Default::default()
    };

    let pretrain_predictor = LinkPredictor::new(pred_config.clone(), pretrained_model);

    println!("  Query: (new_drug, treats, ?)");
    let disease_candidates = vec![
        "cancer".to_string(),
        "diabetes".to_string(),
        "hypertension".to_string(),
        "alzheimers".to_string(), // Domain-specific disease (not in pre-training)
    ];

    let predictions = pretrain_predictor.predict_tail("new_drug", "treats", &disease_candidates)?;

    println!("  Pre-trained Predictions:");
    for pred in predictions.iter().take(3) {
        println!(
            "    → {} (score: {:.3}, confidence: {:.1}%)",
            pred.predicted_id,
            pred.score,
            pred.confidence * 100.0
        );
    }
    println!();

    // ========================================================================
    // Part 2: Create Domain-Specific Data (Alzheimer's Disease)
    // ========================================================================
    println!("🧠 Part 2: Preparing Domain-Specific Data (Alzheimer's)");
    println!("──────────────────────────────────────────────────────────────────");

    let mut domain_triples = Vec::new();

    // Alzheimer's specific knowledge
    println!("  Adding Alzheimer's-specific knowledge...");

    // Disease specifics
    domain_triples.push(Triple::new(
        NamedNode::new("alzheimers")?,
        NamedNode::new("is_a")?,
        NamedNode::new("neurodegenerative_disease")?,
    ));

    domain_triples.push(Triple::new(
        NamedNode::new("alzheimers")?,
        NamedNode::new("is_a")?,
        NamedNode::new("disease")?,
    ));

    // Biomarkers
    domain_triples.push(Triple::new(
        NamedNode::new("amyloid_beta")?,
        NamedNode::new("biomarker_for")?,
        NamedNode::new("alzheimers")?,
    ));

    domain_triples.push(Triple::new(
        NamedNode::new("tau_protein")?,
        NamedNode::new("biomarker_for")?,
        NamedNode::new("alzheimers")?,
    ));

    // Genes
    domain_triples.push(Triple::new(
        NamedNode::new("apoe4")?,
        NamedNode::new("associated_with")?,
        NamedNode::new("alzheimers")?,
    ));

    domain_triples.push(Triple::new(
        NamedNode::new("psen1")?,
        NamedNode::new("associated_with")?,
        NamedNode::new("alzheimers")?,
    ));

    // Treatments
    domain_triples.push(Triple::new(
        NamedNode::new("donepezil")?,
        NamedNode::new("treats")?,
        NamedNode::new("alzheimers")?,
    ));

    domain_triples.push(Triple::new(
        NamedNode::new("memantine")?,
        NamedNode::new("treats")?,
        NamedNode::new("alzheimers")?,
    ));

    // Risk factors
    domain_triples.push(Triple::new(
        NamedNode::new("age")?,
        NamedNode::new("risk_factor_for")?,
        NamedNode::new("alzheimers")?,
    ));

    domain_triples.push(Triple::new(
        NamedNode::new("family_history")?,
        NamedNode::new("risk_factor_for")?,
        NamedNode::new("alzheimers")?,
    ));

    println!("  ✓ Domain-specific triples: {}", domain_triples.len());
    println!();

    // ========================================================================
    // Part 3: Fine-Tuning Strategies Comparison
    // ========================================================================
    println!("⚡ Part 3: Comparing Fine-Tuning Strategies");
    println!("──────────────────────────────────────────────────────────────────");

    let strategies = vec![
        (
            "Full Fine-Tuning",
            FineTuningStrategy::FullFineTuning,
            0.001,
        ),
        (
            "Adapter-Based (Parameter-Efficient)",
            FineTuningStrategy::AdapterBased,
            0.001,
        ),
        (
            "Partial Dimensions (Top 30%)",
            FineTuningStrategy::PartialDimensions,
            0.001,
        ),
    ];

    for (strategy_name, strategy, lr) in strategies {
        println!("\n  Strategy: {}", strategy_name);
        println!("  ────────────────────────────────────────────────────────────");

        // Clone pre-trained model
        let mut model = pretrain_predictor.model().clone();

        // Configure fine-tuning
        let finetune_config = FineTuningConfig {
            strategy,
            learning_rate: lr,
            max_epochs: 30,
            regularization: 0.01,
            partial_dimensions_pct: 0.3,
            adapter_dim: 32,
            early_stopping_patience: 5,
            min_improvement: 0.001,
            validation_split: 0.2,
            use_distillation: false,
            distillation_temperature: 2.0,
            distillation_weight: 0.5,
        };

        let mut ft_manager = FineTuningManager::new(finetune_config);

        // Fine-tune
        let ft_result = ft_manager
            .fine_tune(&mut model, domain_triples.clone())
            .await?;

        println!("    Fine-Tuning Results:");
        println!("      • Epochs: {}", ft_result.epochs_completed);
        println!(
            "      • Training Loss: {:.4}",
            ft_result.final_training_loss
        );
        println!(
            "      • Validation Loss: {:.4}",
            ft_result.final_validation_loss
        );
        println!("      • Time: {:.2}s", ft_result.training_time_seconds);
        println!(
            "      • Parameters Updated: {}",
            ft_result.num_parameters_updated
        );
        println!(
            "      • Early Stopped: {}",
            if ft_result.early_stopped { "Yes" } else { "No" }
        );

        // Test fine-tuned model
        let ft_predictor = LinkPredictor::new(pred_config.clone(), model);

        println!("\n    Query: (new_drug, treats, ?) [After Fine-Tuning]");
        let predictions = ft_predictor.predict_tail("new_drug", "treats", &disease_candidates)?;

        for pred in predictions.iter().take(3) {
            let is_alzheimers = pred.predicted_id == "alzheimers";
            let marker = if is_alzheimers { " ⭐" } else { "" };
            println!(
                "      → {} (score: {:.3}, confidence: {:.1}%){}",
                pred.predicted_id,
                pred.score,
                pred.confidence * 100.0,
                marker
            );
        }
    }

    println!();

    // ========================================================================
    // Part 4: Knowledge Distillation (Advanced)
    // ========================================================================
    println!("🧪 Part 4: Fine-Tuning with Knowledge Distillation");
    println!("──────────────────────────────────────────────────────────────────");

    let mut model = pretrain_predictor.model().clone();

    let distill_config = FineTuningConfig {
        strategy: FineTuningStrategy::FullFineTuning,
        learning_rate: 0.001,
        max_epochs: 30,
        use_distillation: true, // Enable knowledge distillation
        distillation_temperature: 2.0,
        distillation_weight: 0.5, // Balance task loss and distillation loss
        validation_split: 0.2,
        early_stopping_patience: 5,
        ..Default::default()
    };

    let mut distill_manager = FineTuningManager::new(distill_config);

    println!("  Configuration:");
    println!("    • Knowledge Distillation: Enabled");
    println!("    • Distillation Temperature: 2.0");
    println!("    • Distillation Weight: 0.5");
    println!("    • Purpose: Prevent catastrophic forgetting");
    println!();

    let distill_result = distill_manager
        .fine_tune(&mut model, domain_triples.clone())
        .await?;

    println!("  Results:");
    println!("    • Epochs: {}", distill_result.epochs_completed);
    println!(
        "    • Training Loss: {:.4}",
        distill_result.final_training_loss
    );
    println!(
        "    • Validation Loss: {:.4}",
        distill_result.final_validation_loss
    );
    println!("    • Preserved pre-trained knowledge while learning new domain");
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                Fine-Tuning Demo Complete! ✓                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    println!("🎯 Key Takeaways:");
    println!("  • Pre-trained models provide strong general knowledge");
    println!("  • Fine-tuning adapts models to specific domains efficiently");
    println!("  • Adapter-based fine-tuning is parameter-efficient (fewer updates)");
    println!("  • Knowledge distillation prevents catastrophic forgetting");
    println!("  • Different strategies offer trade-offs between adaptation and preservation");
    println!();

    println!("💡 Production Applications:");
    println!("  • Domain-Specific Medical AI (Alzheimer's, Cancer, etc.)");
    println!("  • Personalized Recommendation Systems");
    println!("  • Incremental Learning with New Knowledge");
    println!("  • Multi-Tenant Systems (per-customer fine-tuning)");
    println!("  • Low-Resource Languages/Domains");
    println!();

    println!("📊 Strategy Comparison:");
    println!("  • Full Fine-Tuning: Best accuracy, most parameters updated");
    println!("  • Adapter-Based: Parameter-efficient (32x fewer parameters), good accuracy");
    println!("  • Partial Dimensions: Balance between Full and Adapter");
    println!("  • Knowledge Distillation: Prevents forgetting of pre-trained knowledge");
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
