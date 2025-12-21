//! # Model Selection Demonstration
//!
//! This example demonstrates the intelligent model selection and recommendation
//! capabilities of oxirs-embed, helping users choose the most appropriate
//! embedding model for their specific knowledge graph and use case.
//!
//! ## Features Demonstrated
//!
//! 1. **Dataset Characteristic Analysis**: Automatic inference from basic statistics
//! 2. **Model Recommendations**: Get ranked model suggestions based on use case
//! 3. **Model Comparison**: Compare multiple models on the same criteria
//! 4. **Resource Estimation**: Understand memory and time requirements
//! 5. **Use Case Matching**: Find models optimized for specific tasks
//!
//! ## Running this example
//!
//! ```bash
//! cargo run --example model_selection_demo
//! ```

use anyhow::Result;
use oxirs_embed::model_selection::{DatasetCharacteristics, ModelSelector, ModelType, UseCaseType};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> Result<()> {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== Model Selection Guidance Demonstration ===\n");

    // Create model selector
    let selector = ModelSelector::new();

    // Demo 1: Small biomedical knowledge graph
    info!("📊 SCENARIO 1: Small Biomedical Knowledge Graph");
    info!("================================================");
    demo_small_biomedical_kg(&selector)?;

    // Demo 2: Large general knowledge graph
    info!("\n📊 SCENARIO 2: Large General Knowledge Graph");
    info!("==============================================");
    demo_large_general_kg(&selector)?;

    // Demo 3: Sparse social network
    info!("\n📊 SCENARIO 3: Sparse Social Network");
    info!("=====================================");
    demo_sparse_social_network(&selector)?;

    // Demo 4: Complex hierarchical taxonomy
    info!("\n📊 SCENARIO 4: Complex Hierarchical Taxonomy");
    info!("=============================================");
    demo_hierarchical_taxonomy(&selector)?;

    // Demo 5: Model comparison
    info!("\n📊 SCENARIO 5: Model Comparison");
    info!("================================");
    demo_model_comparison(&selector)?;

    // Demo 6: Use case-specific recommendations
    info!("\n📊 SCENARIO 6: Use Case-Specific Recommendations");
    info!("=================================================");
    demo_use_case_recommendations(&selector)?;

    info!("\n=== Model Selection Demonstration Complete ===");

    Ok(())
}

/// Demo: Small biomedical knowledge graph
fn demo_small_biomedical_kg(selector: &ModelSelector) -> Result<()> {
    let characteristics = DatasetCharacteristics {
        num_entities: 1000,
        num_relations: 15,
        num_triples: 3000,
        avg_degree: 3.0,
        is_sparse: true,
        has_hierarchies: true,
        has_complex_relations: false,
        domain: Some("biomedical".to_string()),
    };

    info!(
        "Dataset: {} entities, {} relations, {} triples",
        characteristics.num_entities, characteristics.num_relations, characteristics.num_triples
    );
    info!("Density: {:.6}", characteristics.density());
    info!(
        "Memory estimate (128-dim): {:.2} MB\n",
        characteristics.estimated_memory_mb(128)
    );

    let recommendations =
        selector.recommend_models(&characteristics, UseCaseType::LinkPrediction)?;

    display_recommendations(&recommendations, 3);

    Ok(())
}

/// Demo: Large general knowledge graph
fn demo_large_general_kg(selector: &ModelSelector) -> Result<()> {
    let characteristics = DatasetCharacteristics {
        num_entities: 100000,
        num_relations: 200,
        num_triples: 500000,
        avg_degree: 5.0,
        is_sparse: true,
        has_hierarchies: true,
        has_complex_relations: true,
        domain: Some("general".to_string()),
    };

    info!(
        "Dataset: {} entities, {} relations, {} triples",
        characteristics.num_entities, characteristics.num_relations, characteristics.num_triples
    );
    info!("Density: {:.6}", characteristics.density());
    info!(
        "Memory estimate (256-dim): {:.2} MB\n",
        characteristics.estimated_memory_mb(256)
    );

    let recommendations = selector.recommend_models(&characteristics, UseCaseType::KGCompletion)?;

    display_recommendations(&recommendations, 3);

    Ok(())
}

/// Demo: Sparse social network
fn demo_sparse_social_network(selector: &ModelSelector) -> Result<()> {
    let characteristics = DatasetCharacteristics {
        num_entities: 50000,
        num_relations: 10,
        num_triples: 100000,
        avg_degree: 2.0,
        is_sparse: true,
        has_hierarchies: false,
        has_complex_relations: false,
        domain: Some("social".to_string()),
    };

    info!(
        "Dataset: {} entities, {} relations, {} triples",
        characteristics.num_entities, characteristics.num_relations, characteristics.num_triples
    );
    info!("Density: {:.6}", characteristics.density());
    info!(
        "Avg degree: {:.2} (very sparse)\n",
        characteristics.avg_degree
    );

    let recommendations =
        selector.recommend_models(&characteristics, UseCaseType::SimilaritySearch)?;

    display_recommendations(&recommendations, 3);

    Ok(())
}

/// Demo: Complex hierarchical taxonomy
fn demo_hierarchical_taxonomy(selector: &ModelSelector) -> Result<()> {
    let characteristics = DatasetCharacteristics {
        num_entities: 10000,
        num_relations: 50,
        num_triples: 30000,
        avg_degree: 3.0,
        is_sparse: false,
        has_hierarchies: true,
        has_complex_relations: true,
        domain: Some("taxonomy".to_string()),
    };

    info!(
        "Dataset: {} entities, {} relations, {} triples",
        characteristics.num_entities, characteristics.num_relations, characteristics.num_triples
    );
    info!("Characteristics: hierarchical structure, complex relations\n");

    let recommendations =
        selector.recommend_models(&characteristics, UseCaseType::EntityClassification)?;

    display_recommendations(&recommendations, 3);

    Ok(())
}

/// Demo: Model comparison
fn demo_model_comparison(selector: &ModelSelector) -> Result<()> {
    let characteristics = DatasetCharacteristics::infer(10000, 50, 50000);

    info!(
        "Comparing models for dataset: {} entities, {} triples\n",
        characteristics.num_entities, characteristics.num_triples
    );

    let models_to_compare = vec![
        ModelType::TransE,
        ModelType::DistMult,
        ModelType::ComplEx,
        ModelType::RotatE,
        ModelType::HolE,
    ];

    let comparison = selector.compare_models(&models_to_compare, &characteristics)?;

    info!("Model Comparison Results:");
    info!("├─────────────┬───────────┬───────┬──────────┬─────────────┬────────────┬──────────┐");
    info!("│ Model       │ Dimension │ Speed │ Accuracy │ Complexity  │ Train Time │ Memory   │");
    info!("├─────────────┼───────────┼───────┼──────────┼─────────────┼────────────┼──────────┤");

    for model_type in &models_to_compare {
        if let Some(entry) = comparison.models.get(model_type) {
            info!(
                "│ {:11} │ {:9} │ {:5} │ {:8} │ {:11} │ {:10} │ {:8} │",
                format!("{}", model_type),
                entry.recommended_dimensions,
                entry.speed,
                entry.accuracy,
                entry.complexity,
                format!("{}", entry.estimated_training_time),
                format!("{}", entry.memory_requirement)
            );
        }
    }
    info!("└─────────────┴───────────┴───────┴──────────┴─────────────┴────────────┴──────────┘");

    Ok(())
}

/// Demo: Use case-specific recommendations
fn demo_use_case_recommendations(selector: &ModelSelector) -> Result<()> {
    let characteristics = DatasetCharacteristics::infer(10000, 50, 50000);

    let use_cases = vec![
        UseCaseType::LinkPrediction,
        UseCaseType::EntityClassification,
        UseCaseType::QuestionAnswering,
        UseCaseType::SimilaritySearch,
    ];

    info!("Top model for each use case:\n");

    for use_case in use_cases {
        let recommendations = selector.recommend_models(&characteristics, use_case)?;

        if let Some(top) = recommendations.first() {
            info!("Use Case: {:?}", use_case);
            info!("  ✓ Top Model: {}", top.model_type);
            info!("  ✓ Score: {:.2}", top.suitability_score);
            info!("  ✓ Reasoning: {}", top.reasoning);
            info!("  ✓ Dimensions: {}", top.recommended_dimensions);
            info!("  ✓ Training Time: {}", top.estimated_training_time);
            info!("  ✓ Memory: {}\n", top.memory_requirement);
        }
    }

    Ok(())
}

/// Display model recommendations in a formatted way
fn display_recommendations(
    recommendations: &[oxirs_embed::model_selection::ModelRecommendation],
    top_n: usize,
) {
    info!(
        "Top {} Model Recommendations:\n",
        top_n.min(recommendations.len())
    );

    for (i, rec) in recommendations.iter().take(top_n).enumerate() {
        info!(
            "{}. {} (Score: {:.2})",
            i + 1,
            rec.model_type,
            rec.suitability_score
        );
        info!("   Reasoning: {}", rec.reasoning);
        info!("   Dimensions: {}", rec.recommended_dimensions);
        info!("   Training Time: {}", rec.estimated_training_time);
        info!("   Memory: {}", rec.memory_requirement);

        if !rec.pros.is_empty() {
            info!("   Pros:");
            for pro in &rec.pros {
                info!("     + {}", pro);
            }
        }

        if !rec.cons.is_empty() {
            info!("   Cons:");
            for con in &rec.cons {
                info!("     - {}", con);
            }
        }

        info!("");
    }
}
