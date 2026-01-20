//! Distributed Training Demo
//!
//! This example demonstrates how to use the distributed training capabilities
//! of oxirs-embed to train knowledge graph embeddings across multiple workers.
//!
//! Run with:
//! ```bash
//! cargo run --example distributed_training_demo --features basic-models
//! ```

use anyhow::Result;
use chrono::Utc;
use oxirs_embed::{
    AggregationMethod,
    CommunicationBackend,
    DistributedEmbeddingTrainer,
    DistributedStrategy,
    DistributedTrainingConfig,
    EmbeddingModel, // Import the trait
    FaultToleranceConfig,
    ModelConfig,
    NamedNode,
    TransE,
    Triple,
    WorkerInfo,
    WorkerStatus,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== OxiRS Distributed Training Demo ===\n");

    // Step 1: Create a knowledge graph embedding model
    println!("1. Creating TransE model...");
    let model_config = ModelConfig::default()
        .with_dimensions(64)
        .with_learning_rate(0.01)
        .with_max_epochs(100)
        .with_batch_size(128);

    let mut model = TransE::new(model_config);

    // Add some sample triples
    println!("2. Adding sample knowledge graph triples...");
    let triples = vec![
        ("Alice", "knows", "Bob"),
        ("Bob", "worksFor", "Company"),
        ("Company", "locatedIn", "NewYork"),
        ("Alice", "livesIn", "NewYork"),
        ("Bob", "knows", "Charlie"),
        ("Charlie", "worksFor", "Company"),
        ("Alice", "friendOf", "Charlie"),
        ("Company", "hasEmployee", "Alice"),
        ("Company", "hasEmployee", "Bob"),
        ("NewYork", "isPartOf", "USA"),
    ];

    let num_triples = triples.len();
    for (s, p, o) in triples {
        let triple = Triple::new(
            NamedNode::new(&format!("http://example.org/{}", s))?,
            NamedNode::new(&format!("http://example.org/{}", p))?,
            NamedNode::new(&format!("http://example.org/{}", o))?,
        );
        model.add_triple(triple)?;
    }

    println!("   Added {} triples", num_triples);

    // Step 2: Configure distributed training
    println!("\n3. Configuring distributed training...");
    let distributed_config = DistributedTrainingConfig {
        strategy: DistributedStrategy::DataParallel {
            num_workers: 4,
            batch_size: 32,
        },
        aggregation: AggregationMethod::AllReduce,
        backend: CommunicationBackend::Tcp,
        fault_tolerance: FaultToleranceConfig {
            enable_checkpointing: true,
            checkpoint_frequency: 10,
            max_retries: 3,
            elastic_scaling: false,
            heartbeat_interval: 30,
            worker_timeout: 300,
        },
        gradient_compression: true,
        compression_ratio: 0.5,
        mixed_precision: true,
        gradient_clip: Some(1.0),
        warmup_epochs: 5,
        pipeline_parallelism: false,
        num_microbatches: 4,
    };

    println!("   Strategy: Data Parallel with 4 workers");
    println!("   Aggregation: AllReduce");
    println!("   Backend: TCP");
    println!("   Mixed Precision: Enabled");
    println!("   Gradient Compression: Enabled (50%)");

    // Step 3: Create distributed trainer
    println!("\n4. Creating distributed trainer...");
    let mut trainer = DistributedEmbeddingTrainer::new(model, distributed_config).await?;

    // Step 4: Register workers
    println!("\n5. Registering workers...");
    for i in 0..4 {
        let worker = WorkerInfo {
            worker_id: i,
            rank: i,
            address: format!("127.0.0.1:808{}", i),
            status: WorkerStatus::Idle,
            num_gpus: if i < 2 { 1 } else { 0 }, // First 2 workers have GPUs
            memory_gb: 16.0,
            last_heartbeat: Utc::now(),
        };
        trainer.register_worker(worker).await?;
        println!(
            "   Registered worker {} at 127.0.0.1:808{} (GPUs: {})",
            i,
            i,
            if i < 2 { 1 } else { 0 }
        );
    }

    // Step 5: Train the model in a distributed manner
    println!("\n6. Starting distributed training...");
    println!("   Training for 20 epochs across 4 workers...\n");

    let stats = trainer.train(20).await?;

    // Step 6: Display results
    println!("\n=== Training Results ===");
    println!("Total Epochs: {}", stats.total_epochs);
    println!("Final Loss: {:.6}", stats.final_loss);
    println!("Training Time: {:.2} seconds", stats.training_time);
    println!("Number of Workers: {}", stats.num_workers);
    println!("Throughput: {:.2} epochs/sec", stats.throughput);
    println!(
        "Communication Time: {:.2} seconds ({:.1}%)",
        stats.communication_time,
        100.0 * stats.communication_time / stats.training_time
    );
    println!(
        "Computation Time: {:.2} seconds ({:.1}%)",
        stats.computation_time,
        100.0 * stats.computation_time / stats.training_time
    );
    println!("Checkpoints Saved: {}", stats.num_checkpoints);
    println!("Worker Failures: {}", stats.num_failures);

    // Step 7: Show loss progression
    println!("\n=== Loss History ===");
    let loss_samples = 5;
    let step = stats.loss_history.len().max(1) / loss_samples.min(stats.loss_history.len());
    for (i, &loss) in stats.loss_history.iter().enumerate().step_by(step.max(1)) {
        println!("Epoch {:2}: {:.6}", i + 1, loss);
    }

    // Step 8: Test the trained model
    println!("\n=== Testing Trained Model ===");
    let model = trainer.model();

    // Predict objects for a given subject-predicate pair
    println!("\nPredicting who Alice knows:");
    let predictions =
        model.predict_objects("http://example.org/Alice", "http://example.org/knows", 3)?;

    for (i, (entity, score)) in predictions.iter().enumerate() {
        let name = entity.split('/').next_back().unwrap_or(entity);
        println!("  {}. {} (score: {:.4})", i + 1, name, score);
    }

    // Predict relations between two entities
    println!("\nPredicting relations between Alice and Company:");
    let relations =
        model.predict_relations("http://example.org/Alice", "http://example.org/Company", 3)?;

    for (i, (relation, score)) in relations.iter().enumerate() {
        let name = relation.split('/').next_back().unwrap_or(relation);
        println!("  {}. {} (score: {:.4})", i + 1, name, score);
    }

    // Step 9: Display training statistics
    let final_stats = trainer.get_stats().await;
    println!("\n=== Final Statistics ===");
    println!("Workers Utilized: {}", final_stats.num_workers);
    println!(
        "Average Communication Overhead: {:.1}%",
        100.0 * final_stats.communication_time
            / (final_stats.communication_time + final_stats.computation_time)
    );

    println!("\n=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("✓ Distributed training accelerates knowledge graph embedding");
    println!("✓ Multiple workers process data in parallel");
    println!("✓ Gradient aggregation ensures consistency");
    println!("✓ Fault tolerance provides reliability");
    println!("✓ Mixed precision reduces memory usage");

    Ok(())
}
