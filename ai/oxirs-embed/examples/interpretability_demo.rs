//! Model Interpretability Demo
//!
//! This example demonstrates interpretability tools for understanding knowledge graph
//! embeddings, including:
//! - Similarity analysis to find related entities
//! - Feature importance analysis
//! - Nearest neighbors exploration
//! - Counterfactual explanations
//! - Embedding space diagnostics
//!
//! # Why Interpretability Matters
//!
//! Understanding *why* a model makes certain predictions is crucial for:
//! - Debugging model behavior
//! - Building trust in predictions
//! - Discovering unexpected patterns
//! - Improving model design
//! - Explaining results to stakeholders
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example interpretability_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    interpretability::{InterpretabilityAnalyzer, InterpretabilityConfig, InterpretationMethod},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   Model Interpretability Demo - Understanding Embeddings ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // ====================
    // Step 1: Build Knowledge Graph
    // ====================
    println!("📚 Step 1: Building Academic Knowledge Graph");
    println!("─────────────────────────────────────────────────────────");

    let config = ModelConfig {
        dimensions: 100,
        learning_rate: 0.01,
        max_epochs: 150,
        ..Default::default()
    };

    let mut model = TransE::new(config);

    // Universities and locations
    add_triple(&mut model, "stanford", "located_in", "california")?;
    add_triple(&mut model, "mit", "located_in", "massachusetts")?;
    add_triple(&mut model, "oxford", "located_in", "uk")?;
    add_triple(&mut model, "cambridge", "located_in", "uk")?;
    add_triple(&mut model, "harvard", "located_in", "massachusetts")?;

    // Research areas
    add_triple(&mut model, "alice", "researches", "machine_learning")?;
    add_triple(&mut model, "bob", "researches", "machine_learning")?;
    add_triple(&mut model, "charlie", "researches", "quantum_computing")?;
    add_triple(&mut model, "diana", "researches", "nlp")?;
    add_triple(&mut model, "eve", "researches", "robotics")?;
    add_triple(&mut model, "frank", "researches", "nlp")?;

    // Affiliations
    add_triple(&mut model, "alice", "affiliated_with", "stanford")?;
    add_triple(&mut model, "bob", "affiliated_with", "mit")?;
    add_triple(&mut model, "charlie", "affiliated_with", "oxford")?;
    add_triple(&mut model, "diana", "affiliated_with", "stanford")?;
    add_triple(&mut model, "eve", "affiliated_with", "mit")?;
    add_triple(&mut model, "frank", "affiliated_with", "cambridge")?;

    // Collaborations
    add_triple(&mut model, "alice", "collaborates_with", "diana")?;
    add_triple(&mut model, "diana", "collaborates_with", "alice")?;
    add_triple(&mut model, "bob", "collaborates_with", "eve")?;
    add_triple(&mut model, "eve", "collaborates_with", "bob")?;

    // Research area relationships
    add_triple(&mut model, "machine_learning", "related_to", "nlp")?;
    add_triple(&mut model, "nlp", "related_to", "machine_learning")?;
    add_triple(&mut model, "machine_learning", "related_to", "robotics")?;

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

    let training_stats = model.train(Some(150)).await?;

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
    println!();

    // ====================
    // Step 4: Similarity Analysis
    // ====================
    println!("🔍 Step 4: Similarity Analysis");
    println!("─────────────────────────────────────────────────────────");

    let similarity_config = InterpretabilityConfig {
        method: InterpretationMethod::SimilarityAnalysis,
        top_k: 5,
        similarity_threshold: 0.5,
        detailed: true,
    };

    let analyzer = InterpretabilityAnalyzer::new(similarity_config);

    // Analyze researchers
    println!("  Analyzing 'alice' (ML researcher at Stanford):");
    let alice_analysis = analyzer.similarity_analysis("alice", &embeddings)?;
    println!("    Most similar entities:");
    for (entity, score) in &alice_analysis.similar_entities {
        println!("      • {} (similarity: {:.3})", entity, score);
    }
    println!(
        "    Average similarity: {:.3}",
        alice_analysis.avg_similarity
    );

    println!("\n  Analyzing 'stanford' (University in California):");
    let stanford_analysis = analyzer.similarity_analysis("stanford", &embeddings)?;
    println!("    Most similar entities:");
    for (entity, score) in &stanford_analysis.similar_entities {
        println!("      • {} (similarity: {:.3})", entity, score);
    }

    println!("\n  Analyzing 'machine_learning' (Research area):");
    let ml_analysis = analyzer.similarity_analysis("machine_learning", &embeddings)?;
    println!("    Most similar entities:");
    for (entity, score) in &ml_analysis.similar_entities {
        println!("      • {} (similarity: {:.3})", entity, score);
    }
    println!();

    // ====================
    // Step 5: Feature Importance Analysis
    // ====================
    println!("📊 Step 5: Feature Importance Analysis");
    println!("─────────────────────────────────────────────────────────");

    let importance_config = InterpretabilityConfig {
        method: InterpretationMethod::FeatureImportance,
        top_k: 10,
        ..Default::default()
    };

    let importance_analyzer = InterpretabilityAnalyzer::new(importance_config);

    println!("  Top 10 important features for 'alice':");
    let alice_importance = importance_analyzer.feature_importance("alice", &embeddings)?;
    for (idx, score) in &alice_importance.important_features {
        println!("    Feature {}: importance = {:.4}", idx, score);
    }

    println!("\n  Feature statistics:");
    let feat_stats = &alice_importance.feature_stats;
    println!(
        "    Mean range: [{:.3}, {:.3}]",
        feat_stats
            .mean
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        feat_stats
            .mean
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "    Std range: [{:.3}, {:.3}]",
        feat_stats.std.iter().cloned().fold(f32::INFINITY, f32::min),
        feat_stats
            .std
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!();

    // ====================
    // Step 6: Nearest Neighbors Analysis
    // ====================
    println!("🎯 Step 6: Nearest Neighbors Analysis");
    println!("─────────────────────────────────────────────────────────");

    let neighbors_config = InterpretabilityConfig {
        method: InterpretationMethod::NearestNeighbors,
        top_k: 5,
        ..Default::default()
    };

    let neighbors_analyzer = InterpretabilityAnalyzer::new(neighbors_config);

    println!("  5 nearest neighbors to 'alice':");
    let alice_neighbors = neighbors_analyzer.nearest_neighbors_analysis("alice", &embeddings)?;
    for (neighbor, distance) in &alice_neighbors.neighbors {
        println!("    • {} (distance: {:.3})", neighbor, distance);
    }

    println!("\n  5 nearest neighbors to 'mit':");
    let mit_neighbors = neighbors_analyzer.nearest_neighbors_analysis("mit", &embeddings)?;
    for (neighbor, distance) in &mit_neighbors.neighbors {
        println!("    • {} (distance: {:.3})", neighbor, distance);
    }

    println!("\n  5 nearest neighbors to 'quantum_computing':");
    let qc_neighbors =
        neighbors_analyzer.nearest_neighbors_analysis("quantum_computing", &embeddings)?;
    for (neighbor, distance) in &qc_neighbors.neighbors {
        println!("    • {} (distance: {:.3})", neighbor, distance);
    }
    println!();

    // ====================
    // Step 7: Counterfactual Explanations
    // ====================
    println!("💡 Step 7: Counterfactual Explanations");
    println!("─────────────────────────────────────────────────────────");

    println!("  What would 'alice' need to change to be like 'bob'?");
    let counterfactual = analyzer.counterfactual_explanation("alice", "bob", &embeddings)?;
    println!("    Difficulty: {:.2}", counterfactual.difficulty);
    println!("    Required changes (top 5 dimensions):");
    for (i, (dim, from, to)) in counterfactual.required_changes.iter().take(5).enumerate() {
        let delta = to - from;
        println!(
            "      {}. Dimension {}: {:.3} → {:.3} (Δ{:.3})",
            i + 1,
            dim,
            from,
            to,
            delta
        );
    }

    println!("\n  What would 'stanford' need to change to be like 'mit'?");
    let stanford_to_mit = analyzer.counterfactual_explanation("stanford", "mit", &embeddings)?;
    println!("    Difficulty: {:.2}", stanford_to_mit.difficulty);
    println!("    Top 3 dimension changes:");
    for (i, (dim, from, to)) in stanford_to_mit.required_changes.iter().take(3).enumerate() {
        println!(
            "      {}. Dimension {}: {:.3} → {:.3}",
            i + 1,
            dim,
            from,
            to
        );
    }
    println!();

    // ====================
    // Step 8: Embedding Space Diagnostics
    // ====================
    println!("🔬 Step 8: Embedding Space Diagnostics");
    println!("─────────────────────────────────────────────────────────");

    // Compute overall statistics
    let all_similarities: Vec<f32> = embeddings
        .keys()
        .flat_map(|e1| {
            embeddings
                .keys()
                .filter(move |e2| e1 != *e2)
                .map(|e2| cosine_similarity(&embeddings[e1], &embeddings[e2]))
        })
        .collect();

    let avg_similarity: f32 = all_similarities.iter().sum::<f32>() / all_similarities.len() as f32;
    let max_similarity = all_similarities
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_similarity = all_similarities
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);

    println!("  Global embedding statistics:");
    println!("    Average pairwise similarity: {:.3}", avg_similarity);
    println!("    Max similarity: {:.3}", max_similarity);
    println!("    Min similarity: {:.3}", min_similarity);
    println!(
        "    Similarity range: {:.3}",
        max_similarity - min_similarity
    );

    // Find most and least similar pairs
    let mut pairs: Vec<(String, String, f32)> = Vec::new();
    for e1 in embeddings.keys() {
        for e2 in embeddings.keys() {
            if e1 < e2 {
                let sim = cosine_similarity(&embeddings[e1], &embeddings[e2]);
                pairs.push((e1.clone(), e2.clone(), sim));
            }
        }
    }

    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("\n  Most similar entity pairs:");
    for (e1, e2, sim) in pairs.iter().take(3) {
        println!("    • {} ↔ {} (similarity: {:.3})", e1, e2, sim);
    }

    println!("\n  Least similar entity pairs:");
    for (e1, e2, sim) in pairs.iter().rev().take(3) {
        println!("    • {} ↔ {} (similarity: {:.3})", e1, e2, sim);
    }
    println!();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║        Interpretability Analysis Completed!            ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    println!("💡 Key Insights:");
    println!("  • Similarity analysis reveals semantic relationships");
    println!("  • Feature importance shows which dimensions matter most");
    println!("  • Nearest neighbors help understand entity clustering");
    println!("  • Counterfactuals explain what would change predictions");
    println!("  • Diagnostics reveal overall embedding space quality");
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

// Helper function for cosine similarity
fn cosine_similarity(
    a: &scirs2_core::ndarray_ext::Array1<f32>,
    b: &scirs2_core::ndarray_ext::Array1<f32>,
) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}
