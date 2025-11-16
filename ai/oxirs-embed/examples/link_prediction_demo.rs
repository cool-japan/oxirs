//! Comprehensive Link Prediction Demo
//!
//! This example demonstrates advanced link prediction capabilities for knowledge graph
//! completion, including:
//! - Tail entity prediction (object prediction)
//! - Head entity prediction (subject prediction)
//! - Relation prediction
//! - Batch prediction for efficiency
//! - Evaluation metrics (MRR, Hits@K, Mean Rank)
//! - Filtered ranking to remove known triples
//! - Comparison across different embedding models
//!
//! # Knowledge Graph Completion
//!
//! Link prediction is a fundamental task in knowledge graph completion where we predict
//! missing links (triples) based on learned embeddings. Given partial information about
//! a triple, we predict the missing element.
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example link_prediction_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    link_prediction::{LinkPrediction, LinkPredictionConfig, LinkPredictor},
    models::hole::{HoLE, HoLEConfig},
    EmbeddingModel, ModelConfig, NamedNode, TransE, Triple,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║    Link Prediction Demo - Knowledge Graph Completion   ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // ====================
    // Step 1: Build Knowledge Graph
    // ====================
    println!("📚 Step 1: Building Academic Knowledge Graph");
    println!("─────────────────────────────────────────────────────────");

    let config = ModelConfig {
        dimensions: 128,
        learning_rate: 0.01,
        max_epochs: 150,
        batch_size: 32,
        ..Default::default()
    };

    let mut model = TransE::new(config);

    // Academic relationships
    // Authors and their affiliations
    add_triple(&mut model, "alice", "affiliated_with", "stanford")?;
    add_triple(&mut model, "bob", "affiliated_with", "mit")?;
    add_triple(&mut model, "charlie", "affiliated_with", "stanford")?;
    add_triple(&mut model, "diana", "affiliated_with", "oxford")?;
    add_triple(&mut model, "eve", "affiliated_with", "mit")?;

    // Authors and their research areas
    add_triple(&mut model, "alice", "researches", "machine_learning")?;
    add_triple(&mut model, "bob", "researches", "quantum_computing")?;
    add_triple(&mut model, "charlie", "researches", "machine_learning")?;
    add_triple(&mut model, "diana", "researches", "nlp")?;
    add_triple(&mut model, "eve", "researches", "robotics")?;

    // Collaboration relationships
    add_triple(&mut model, "alice", "collaborates_with", "charlie")?;
    add_triple(&mut model, "charlie", "collaborates_with", "alice")?;
    add_triple(&mut model, "bob", "collaborates_with", "eve")?;
    add_triple(&mut model, "eve", "collaborates_with", "bob")?;
    add_triple(&mut model, "alice", "collaborates_with", "diana")?;

    // Publication venues
    add_triple(&mut model, "alice", "publishes_in", "neurips")?;
    add_triple(&mut model, "charlie", "publishes_in", "neurips")?;
    add_triple(&mut model, "bob", "publishes_in", "quantum_journal")?;
    add_triple(&mut model, "diana", "publishes_in", "acl")?;
    add_triple(&mut model, "eve", "publishes_in", "icra")?;

    // Research area relationships
    add_triple(&mut model, "machine_learning", "related_to", "nlp")?;
    add_triple(&mut model, "nlp", "related_to", "machine_learning")?;
    add_triple(&mut model, "quantum_computing", "related_to", "robotics")?;

    // University locations
    add_triple(&mut model, "stanford", "located_in", "california")?;
    add_triple(&mut model, "mit", "located_in", "massachusetts")?;
    add_triple(&mut model, "oxford", "located_in", "uk")?;

    let stats = model.get_stats();
    println!("  Total entities: {}", stats.num_entities);
    println!("  Total relations: {}", stats.num_relations);
    println!("  Total triples: {}", stats.num_triples);
    println!();

    // ====================
    // Step 2: Train the Model
    // ====================
    println!("🎓 Step 2: Training TransE Model");
    println!("─────────────────────────────────────────────────────────");

    let training_stats = model.train(Some(150)).await?;

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

    // ====================
    // Step 3: Tail Entity Prediction (Object Prediction)
    // ====================
    println!("🔮 Step 3: Tail Entity Prediction");
    println!("─────────────────────────────────────────────────────────");
    println!("Task: Given (subject, relation, ?), predict the object\n");

    let pred_config = LinkPredictionConfig {
        top_k: 5,
        filter_known_triples: false,
        min_confidence: 0.0,
        parallel: true,
        batch_size: 100,
    };

    let predictor = LinkPredictor::new(pred_config.clone(), model);

    // Query 1: Who collaborates with Alice?
    println!("  Query 1: (alice, collaborates_with, ?)");
    let candidates = vec![
        "bob".to_string(),
        "charlie".to_string(),
        "diana".to_string(),
        "eve".to_string(),
    ];

    let predictions = predictor.predict_tail("alice", "collaborates_with", &candidates)?;
    print_predictions(&predictions);

    // Query 2: Where does Bob publish?
    println!("\n  Query 2: (bob, publishes_in, ?)");
    let venue_candidates = vec![
        "neurips".to_string(),
        "quantum_journal".to_string(),
        "acl".to_string(),
        "icra".to_string(),
    ];

    let predictions = predictor.predict_tail("bob", "publishes_in", &venue_candidates)?;
    print_predictions(&predictions);

    // Query 3: What does Diana research?
    println!("\n  Query 3: (diana, researches, ?)");
    let research_candidates = vec![
        "machine_learning".to_string(),
        "quantum_computing".to_string(),
        "nlp".to_string(),
        "robotics".to_string(),
    ];

    let predictions = predictor.predict_tail("diana", "researches", &research_candidates)?;
    print_predictions(&predictions);

    // ====================
    // Step 4: Head Entity Prediction (Subject Prediction)
    // ====================
    println!("\n🔮 Step 4: Head Entity Prediction");
    println!("─────────────────────────────────────────────────────────");
    println!("Task: Given (?, relation, object), predict the subject\n");

    // Query 4: Who is affiliated with Stanford?
    println!("  Query 4: (?, affiliated_with, stanford)");
    let person_candidates = vec![
        "alice".to_string(),
        "bob".to_string(),
        "charlie".to_string(),
        "diana".to_string(),
        "eve".to_string(),
    ];

    let predictions = predictor.predict_head("affiliated_with", "stanford", &person_candidates)?;
    print_predictions(&predictions);

    // Query 5: Who publishes in NeurIPS?
    println!("\n  Query 5: (?, publishes_in, neurips)");
    let predictions = predictor.predict_head("publishes_in", "neurips", &person_candidates)?;
    print_predictions(&predictions);

    // ====================
    // Step 5: Relation Prediction
    // ====================
    println!("\n🔮 Step 5: Relation Prediction");
    println!("─────────────────────────────────────────────────────────");
    println!("Task: Given (subject, ?, object), predict the relation\n");

    // Query 6: What is the relationship between Alice and Stanford?
    println!("  Query 6: (alice, ?, stanford)");
    let relation_candidates = vec![
        "affiliated_with".to_string(),
        "collaborates_with".to_string(),
        "located_in".to_string(),
        "researches".to_string(),
    ];

    let predictions = predictor.predict_relation("alice", "stanford", &relation_candidates)?;
    print_predictions(&predictions);

    // Query 7: What is the relationship between machine_learning and nlp?
    println!("\n  Query 7: (machine_learning, ?, nlp)");
    let area_relation_candidates = vec![
        "related_to".to_string(),
        "researches".to_string(),
        "publishes_in".to_string(),
    ];

    let predictions =
        predictor.predict_relation("machine_learning", "nlp", &area_relation_candidates)?;
    print_predictions(&predictions);

    // ====================
    // Step 6: Batch Prediction
    // ====================
    println!("\n⚡ Step 6: Batch Prediction (Efficient)");
    println!("─────────────────────────────────────────────────────────");
    println!("Processing multiple queries in parallel\n");

    let batch_queries = vec![
        ("alice".to_string(), "collaborates_with".to_string()),
        ("bob".to_string(), "collaborates_with".to_string()),
        ("charlie".to_string(), "collaborates_with".to_string()),
    ];

    let batch_results = predictor.predict_tails_batch(&batch_queries, &person_candidates)?;

    for (idx, (subject, relation)) in batch_queries.iter().enumerate() {
        println!("  Query: ({}, {}, ?)", subject, relation);
        if let Some(predictions) = batch_results.get(idx) {
            for pred in predictions.iter().take(3) {
                println!(
                    "    → {} (score: {:.3}, rank: {})",
                    pred.predicted_id, pred.score, pred.rank
                );
            }
        }
        println!();
    }

    // ====================
    // Step 7: Evaluation Metrics
    // ====================
    println!("📊 Step 7: Link Prediction Evaluation");
    println!("─────────────────────────────────────────────────────────");

    // Create test triples (withheld from training)
    let test_triples = vec![
        Triple::new(
            NamedNode::new("alice")?,
            NamedNode::new("collaborates_with")?,
            NamedNode::new("charlie")?,
        ),
        Triple::new(
            NamedNode::new("bob")?,
            NamedNode::new("publishes_in")?,
            NamedNode::new("quantum_journal")?,
        ),
        Triple::new(
            NamedNode::new("diana")?,
            NamedNode::new("researches")?,
            NamedNode::new("nlp")?,
        ),
    ];

    // Note: In a real scenario, you would train on a subset and test on held-out triples
    let all_entities = vec![
        "alice".to_string(),
        "bob".to_string(),
        "charlie".to_string(),
        "diana".to_string(),
        "eve".to_string(),
        "stanford".to_string(),
        "mit".to_string(),
        "oxford".to_string(),
        "machine_learning".to_string(),
        "quantum_computing".to_string(),
        "nlp".to_string(),
        "robotics".to_string(),
    ];

    let metrics = predictor.evaluate(&test_triples, &all_entities)?;

    println!("  Mean Reciprocal Rank (MRR): {:.4}", metrics.mrr);
    println!("  Mean Rank: {:.2}", metrics.mean_rank);
    println!("  Hits@1: {:.1}%", metrics.hits_at_1 * 100.0);
    println!("  Hits@3: {:.1}%", metrics.hits_at_3 * 100.0);
    println!("  Hits@5: {:.1}%", metrics.hits_at_5 * 100.0);
    println!("  Hits@10: {:.1}%", metrics.hits_at_10 * 100.0);
    println!();

    // ====================
    // Step 8: Model Comparison (Optional: if HolE feature is enabled)
    // ====================
    #[cfg(feature = "hole")]
    {
        println!("🔬 Step 8: Comparing with HolE Model");
        println!("─────────────────────────────────────────────────────────");

        let hole_config = HoLEConfig {
            base: ModelConfig {
                dimensions: 128,
                learning_rate: 0.01,
                max_epochs: 150,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut hole_model = HoLE::new(hole_config);

        // Add same triples
        for triple in &test_triples {
            hole_model.add_triple(triple.clone())?;
        }

        // Train HolE
        let hole_stats = hole_model.train(Some(150)).await?;
        println!("  HolE Final loss: {:.4}", hole_stats.final_loss);

        // Compare predictions
        let hole_predictor = LinkPredictor::new(pred_config, hole_model);

        let hole_predictions =
            hole_predictor.predict_tail("alice", "collaborates_with", &person_candidates)?;

        println!("\n  Comparison: (alice, collaborates_with, ?)");
        println!("  Top prediction scores:");
        for pred in hole_predictions.iter().take(3) {
            println!("    → {} (score: {:.3})", pred.predicted_id, pred.score);
        }
        println!();
    }

    // ====================
    // Summary
    // ====================
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║              Link Prediction Demo Complete! ✓          ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Takeaways:");
    println!("  • Link prediction enables knowledge graph completion");
    println!("  • Supports tail, head, and relation prediction");
    println!("  • Batch processing improves efficiency for multiple queries");
    println!("  • Standard metrics (MRR, Hits@K) enable model evaluation");
    println!("  • Different models can be compared on same data");
    println!();
    println!("Applications:");
    println!("  • Recommendation systems (user-item predictions)");
    println!("  • Drug discovery (drug-target predictions)");
    println!("  • Question answering (entity-relation-entity completion)");
    println!("  • Knowledge base curation (finding missing facts)");
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

/// Pretty print predictions
fn print_predictions(predictions: &[LinkPrediction]) {
    for pred in predictions {
        println!(
            "    → {} (score: {:.3}, confidence: {:.1}%, rank: {})",
            pred.predicted_id,
            pred.score,
            pred.confidence * 100.0,
            pred.rank
        );
    }
}
