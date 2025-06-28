//! Basic usage example for oxirs-embed
//!
//! This example demonstrates how to:
//! 1. Create and configure embedding models
//! 2. Load data and train models
//! 3. Generate embeddings and perform inference
//! 4. Evaluate model performance
//! 5. Integrate with other OxiRS components

use anyhow::Result;
use oxirs_embed::{NamedNode, Triple};
use oxirs_embed::{
    models::{ComplEx, DistMult, RotatE, TransE},
    EmbeddingModel, ModelConfig,
};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting OxiRS Embed basic usage example");

    // Example 1: Basic TransE model training
    basic_transe_example().await?;

    // Example 2: Model comparison
    model_comparison_example().await?;

    // Example 3: Advanced training with optimization (commented out due to missing types)
    // advanced_training_example().await?;

    // Example 4: Inference and similarity search (commented out due to missing types)
    // inference_example().await?;

    // Example 5: Integration with OxiRS ecosystem (commented out due to missing types)
    // integration_example().await?;

    // Example 6: Data loading and evaluation (commented out due to missing types)
    // evaluation_example().await?;

    println!("All examples completed successfully!");
    Ok(())
}

/// Example 1: Basic TransE model training
async fn basic_transe_example() -> Result<()> {
    println!("=== Example 1: Basic TransE Model ===");

    // Create model configuration
    let config = ModelConfig::default()
        .with_dimensions(50)
        .with_learning_rate(0.01)
        .with_max_epochs(100)
        .with_batch_size(32)
        .with_seed(42);

    // Create TransE model
    let mut model = TransE::new(config);

    // Add some sample triples
    let alice = NamedNode::new("http://example.org/alice")?;
    let bob = NamedNode::new("http://example.org/bob")?;
    let charlie = NamedNode::new("http://example.org/charlie")?;
    let knows = NamedNode::new("http://example.org/knows")?;
    let likes = NamedNode::new("http://example.org/likes")?;

    model.add_triple(Triple::new(alice.clone(), knows.clone(), bob.clone()))?;
    model.add_triple(Triple::new(bob.clone(), knows.clone(), charlie.clone()))?;
    model.add_triple(Triple::new(alice.clone(), likes.clone(), charlie.clone()))?;
    model.add_triple(Triple::new(charlie.clone(), likes.clone(), alice.clone()))?;

    println!(
        "Added {} triples to the model",
        model.get_stats().num_triples
    );

    // Train the model
    let start_time = Instant::now();
    let training_stats = model.train(Some(50)).await?;
    let training_time = start_time.elapsed();

    println!("Training completed in {:.2}s", training_time.as_secs_f64());
    println!("Final loss: {:.6}", training_stats.final_loss);
    println!("Epochs completed: {}", training_stats.epochs_completed);

    // Get embeddings
    let alice_embedding = model.get_entity_embedding("http://example.org/alice")?;
    let knows_embedding = model.get_relation_embedding("http://example.org/knows")?;

    println!("Alice embedding dimensions: {}", alice_embedding.dimensions);
    println!("Knows embedding dimensions: {}", knows_embedding.dimensions);

    // Score some triples
    let score = model.score_triple(
        "http://example.org/alice",
        "http://example.org/knows",
        "http://example.org/bob",
    )?;
    println!("Score for (alice, knows, bob): {:.6}", score);

    // Make predictions
    let predictions =
        model.predict_objects("http://example.org/alice", "http://example.org/knows", 5)?;
    println!("Top predictions for (alice, knows, ?): {:?}", predictions);

    Ok(())
}

/// Example 2: Comparing different embedding models
async fn model_comparison_example() -> Result<()> {
    println!("=== Example 2: Model Comparison ===");

    // Common configuration for all models
    let base_config = ModelConfig::default()
        .with_dimensions(32)
        .with_max_epochs(20)
        .with_seed(42);

    // Create different models
    let mut transe = TransE::new(base_config.clone());
    let mut distmult = DistMult::new(base_config.clone());
    let mut complex = ComplEx::new(base_config.clone());
    let mut rotate = RotatE::new(base_config);

    // Sample data
    let triples = vec![
        ("alice", "knows", "bob"),
        ("bob", "knows", "charlie"),
        ("alice", "likes", "charlie"),
        ("charlie", "likes", "alice"),
        ("bob", "friendOf", "alice"),
    ];

    // Add triples to all models
    for (s, p, o) in &triples {
        let subject = NamedNode::new(&format!("http://example.org/{}", s))?;
        let predicate = NamedNode::new(&format!("http://example.org/{}", p))?;
        let object = NamedNode::new(&format!("http://example.org/{}", o))?;
        let triple = Triple::new(subject, predicate, object);

        transe.add_triple(triple.clone())?;
        distmult.add_triple(triple.clone())?;
        complex.add_triple(triple.clone())?;
        rotate.add_triple(triple)?;
    }

    // Train all models and compare
    let models: Vec<(&str, &mut dyn EmbeddingModel)> = vec![
        ("TransE", &mut transe),
        ("DistMult", &mut distmult),
        ("ComplEx", &mut complex),
        ("RotatE", &mut rotate),
    ];

    for (name, model) in models {
        let start_time = Instant::now();
        let stats = model.train(Some(10)).await?;
        let training_time = start_time.elapsed();

        println!(
            "{} - Training time: {:.2}s, Final loss: {:.6}",
            name,
            training_time.as_secs_f64(),
            stats.final_loss
        );
    }

    Ok(())
}

/*
/// Example 3: Advanced training with optimizers (commented out due to missing types)
async fn advanced_training_example() -> Result<()> {
    println!("=== Example 3: Advanced Training ===");

    let config = ModelConfig::default().with_dimensions(64).with_seed(42);

    let mut model = TransE::new(config);

    // Add more complex dataset
    let entities = ["alice", "bob", "charlie", "david", "eve"];
    let relations = ["knows", "likes", "friendOf", "worksWith", "livesIn"];

    // Generate some synthetic data
    for i in 0..20 {
        let s = entities[i % entities.len()];
        let p = relations[i % relations.len()];
        let o = entities[(i + 1) % entities.len()];

        let subject = NamedNode::new(&format!("http://example.org/{}", s))?;
        let predicate = NamedNode::new(&format!("http://example.org/{}", p))?;
        let object = NamedNode::new(&format!("http://example.org/{}", o))?;

        model.add_triple(Triple::new(subject, predicate, object))?;
    }

    // Configure advanced training
    let training_config = TrainingConfig {
        max_epochs: 100,
        batch_size: 16,
        learning_rate: 0.01,
        validation_freq: 10,
        log_freq: 5,
        use_early_stopping: true,
        patience: 20,
        min_delta: 1e-6,
        ..Default::default()
    };

    let mut trainer = AdvancedTrainer::new(training_config).with_optimizer(OptimizerType::Adam {
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    });

    println!("Starting advanced training with Adam optimizer");
    let stats = trainer.train(&mut model).await?;

    println!("Advanced training completed:");
    println!("  Epochs: {}", stats.epochs_completed);
    println!("  Final loss: {:.6}", stats.final_loss);
    println!("  Training time: {:.2}s", stats.training_time_seconds);
    println!("  Converged: {}", stats.convergence_achieved);

    Ok(())
}
*/

/*
/// Example 4: High-performance inference (commented out due to missing types)
async fn inference_example() -> Result<()> {
    println!("=== Example 4: Inference and Similarity ===");

    let config = ModelConfig::default().with_dimensions(32).with_seed(42);

    let mut model = TransE::new(config);

    // Add training data
    let knowledge_base = vec![
        ("tokyo", "locatedIn", "japan"),
        ("osaka", "locatedIn", "japan"),
        ("paris", "locatedIn", "france"),
        ("london", "locatedIn", "uk"),
        ("japan", "hasCapital", "tokyo"),
        ("france", "hasCapital", "paris"),
        ("uk", "hasCapital", "london"),
    ];

    for (s, p, o) in knowledge_base {
        let subject = NamedNode::new(&format!("http://example.org/{}", s))?;
        let predicate = NamedNode::new(&format!("http://example.org/{}", p))?;
        let object = NamedNode::new(&format!("http://example.org/{}", o))?;

        model.add_triple(Triple::new(subject, predicate, object))?;
    }

    // Train the model
    model.train(Some(50)).await?;

    // Create inference engine with caching
    let inference_config = InferenceConfig {
        cache_size: 1000,
        enable_caching: true,
        batch_size: 10,
        ..Default::default()
    };

    let engine = InferenceEngine::new(Box::new(model), inference_config);

    // Warm up cache
    engine.warm_up_cache().await?;

    // Perform cached inference
    let tokyo_embedding = engine
        .get_entity_embedding("http://example.org/tokyo")
        .await?;
    let osaka_embedding = engine
        .get_entity_embedding("http://example.org/osaka")
        .await?;

    println!(
        "Tokyo embedding retrieved (dimensions: {})",
        tokyo_embedding.dimensions
    );
    println!(
        "Osaka embedding retrieved (dimensions: {})",
        osaka_embedding.dimensions
    );

    // Score triples with caching
    let score = engine
        .score_triple(
            "http://example.org/tokyo",
            "http://example.org/locatedIn",
            "http://example.org/japan",
        )
        .await?;

    println!("Score for (tokyo, locatedIn, japan): {:.6}", score);

    // Get cache statistics
    let cache_stats = engine.cache_stats()?;
    println!(
        "Cache stats: entity cache size: {}, relation cache size: {}",
        cache_stats.entity_cache_size, cache_stats.relation_cache_size
    );

    Ok(())
}
*/

/*
/// Example 5: Integration with OxiRS ecosystem (commented out due to missing types)
async fn integration_example() -> Result<()> {
    println!("=== Example 5: OxiRS Integration ===");

    let config = ModelConfig::default().with_dimensions(64).with_seed(42);

    let model = TransE::new(config);

    // Create integration service
    let integration_config = IntegrationConfig {
        auto_embed_new_triples: true,
        embedding_batch_size: 100,
        embedding_cache_size: 1000,
        ..Default::default()
    };

    let mut service = EmbeddingIntegrationService::new(Box::new(model), integration_config);

    // Start the service
    service.start().await?;

    // Process some triples
    let triples = vec![
        ("company_a", "hasEmployee", "person_1"),
        ("person_1", "worksIn", "department_ai"),
        ("department_ai", "partOf", "company_a"),
        ("person_1", "hasSkill", "machine_learning"),
    ];

    let mut triple_objects = Vec::new();
    for (s, p, o) in triples {
        let subject = NamedNode::new(&format!("http://example.org/{}", s))?;
        let predicate = NamedNode::new(&format!("http://example.org/{}", p))?;
        let object = NamedNode::new(&format!("http://example.org/{}", o))?;

        let triple = Triple::new(subject, predicate, object);
        triple_objects.push(triple);
    }

    // Process triples in batch
    service.process_triple_batch(&triple_objects).await?;

    // Train the integrated model
    service.train_model(Some(30)).await?;

    // Find similar entities
    let similar_entities = service
        .find_similar_entities("http://example.org/person_1", 3)
        .await?;

    println!("Entities similar to person_1: {:?}", similar_entities);

    // Get model statistics
    let stats = service.get_model_stats().await?;
    println!(
        "Model stats: {} entities, {} relations, {} triples",
        stats.num_entities, stats.num_relations, stats.num_triples
    );

    // Stop the service
    service.stop().await;

    Ok(())
}
*/

/*
/// Example 6: Data loading and evaluation (commented out due to missing types)
async fn evaluation_example() -> Result<()> {
    println!("=== Example 6: Evaluation ===");

    // Create synthetic dataset
    let train_triples = vec![
        ("person1", "knows", "person2"),
        ("person2", "knows", "person3"),
        ("person1", "likes", "activity1"),
        ("person3", "likes", "activity2"),
        ("activity1", "typeOf", "sport"),
        ("activity2", "typeOf", "art"),
    ];

    let test_triples = vec![
        ("person1", "knows", "person3"),
        ("person2", "likes", "activity1"),
    ];

    // Compute dataset statistics
    let train_triples_formatted: Vec<(String, String, String)> = train_triples
        .iter()
        .map(|(s, p, o)| (s.to_string(), p.to_string(), o.to_string()))
        .collect();
    let stats = compute_dataset_statistics(&train_triples_formatted);
    println!("Dataset statistics:");
    println!("  Triples: {}", stats.num_triples);
    println!("  Entities: {}", stats.num_entities);
    println!("  Relations: {}", stats.num_relations);
    println!("  Average degree: {:.2}", stats.avg_degree);
    println!("  Density: {:.6}", stats.density);

    // Create and train model
    let config = ModelConfig::default()
        .with_dimensions(32)
        .with_max_epochs(50)
        .with_seed(42);

    let mut model = TransE::new(config);

    // Add training data
    for (s, p, o) in train_triples {
        let subject = NamedNode::new(&format!("http://example.org/{}", s))?;
        let predicate = NamedNode::new(&format!("http://example.org/{}", p))?;
        let object = NamedNode::new(&format!("http://example.org/{}", o))?;

        model.add_triple(Triple::new(subject, predicate, object))?;
    }

    model.train(Some(30)).await?;

    // Create evaluation suite
    let test_triples_formatted: Vec<(String, String, String)> = test_triples
        .into_iter()
        .map(|(s, p, o)| {
            (
                format!("http://example.org/{}", s),
                format!("http://example.org/{}", p),
                format!("http://example.org/{}", o),
            )
        })
        .collect();

    let eval_config = EvaluationConfig {
        k_values: vec![1, 3, 5],
        use_filtered_ranking: true,
        parallel_evaluation: true,
        ..Default::default()
    };

    let mut eval_suite = EvaluationSuite::new(
        test_triples_formatted,
        vec![], // No validation triples for this example
    )
    .with_config(eval_config);

    // Generate negative samples
    eval_suite.generate_negative_samples(&model)?;

    // Run evaluation
    let eval_results = eval_suite.evaluate(&model)?;

    println!("Evaluation results:");
    println!("  Mean Rank: {:.2}", eval_results.mean_rank);
    println!(
        "  Mean Reciprocal Rank: {:.4}",
        eval_results.mean_reciprocal_rank
    );

    for (k, hits) in eval_results.hits_at_k {
        println!("  Hits@{}: {:.4}", k, hits);
    }

    println!(
        "  Evaluation time: {:.2}s",
        eval_results.evaluation_time_seconds
    );

    Ok(())
}
*/
