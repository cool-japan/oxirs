//! Distributed Training Demo
//!
//! This example demonstrates the distributed training capabilities of
//! oxirs-embed.  Two execution modes are available:
//!
//! - **default** (no flag): runs the legacy `DistributedEmbeddingTrainer`
//!   coordinator path with simulated worker registration and gradient
//!   aggregation.
//! - **`--distributed`**: drives the new parameter-server-style prototype
//!   (`ParameterServer` + `Worker` + `ModelShardManager`) with 4 workers on
//!   a small KG.  Use this to reproduce the unit/integration test path.
//!
//! Run with:
//! ```bash
//! cargo run --example distributed_training_demo --features basic-models
//! cargo run --example distributed_training_demo --features basic-models -- --distributed
//! ```

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use oxirs_embed::{
    distributed_training::{
        ModelShardManager, ParameterServer, ParameterServerConfig, ShardingStrategy, TripleSample,
        UpdateMode, Worker, WorkerConfig,
    },
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

    // Parse the `--distributed` flag explicitly — adding `clap` for one flag
    // is overkill, and the project policy prefers minimal dependencies.
    let distributed_mode = std::env::args().any(|a| a == "--distributed");

    if distributed_mode {
        return run_parameter_server_demo().await;
    }

    println!("=== OxiRS Distributed Training Demo ===\n");
    println!("(re-run with `-- --distributed` for the parameter-server prototype)\n");

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
    println!("[x] Distributed training accelerates knowledge graph embedding");
    println!("[x] Multiple workers process data in parallel");
    println!("[x] Gradient aggregation ensures consistency");
    println!("[x] Fault tolerance provides reliability");
    println!("[x] Mixed precision reduces memory usage");

    Ok(())
}

/// Parameter-server prototype demo invoked by the `--distributed` flag.
///
/// Boots a 4-shard parameter server and 4 workers, runs the TransE-shaped
/// inner loop on a tiny KG, and reports per-worker tail-loss alongside
/// server statistics.  Mirrors the integration test in
/// `tests/distributed_training.rs`.
async fn run_parameter_server_demo() -> Result<()> {
    println!("=== OxiRS Parameter-Server Prototype Demo ===\n");

    let num_entities = 16;
    let num_shards = 4;
    let num_workers = 4;

    let entity_ids: Vec<String> = (0..num_entities).map(|i| format!("e{i}")).collect();
    let relation_ids: Vec<String> = vec!["rel0".into(), "rel1".into()];

    let cfg = ParameterServerConfig {
        embedding_dim: 16,
        num_entities,
        num_relations: relation_ids.len(),
        num_shards,
        expected_workers: 1,
        update_mode: UpdateMode::Async,
        learning_rate: 0.05,
        max_staleness: 32,
        barrier_timeout: std::time::Duration::from_secs(2),
    };
    let mgr = ModelShardManager::new(num_shards, ShardingStrategy::EntityHash);
    let server = Arc::new(ParameterServer::new(
        cfg,
        entity_ids.clone(),
        relation_ids,
        mgr,
    )?);

    // Build the training KG.
    let mut samples = Vec::new();
    for i in 0..num_entities {
        let next = (i + 1) % num_entities;
        samples.push(TripleSample::new(
            format!("e{i}"),
            "rel0",
            format!("e{next}"),
        ));
    }
    for i in 0..num_entities {
        if i % 2 == 0 {
            let next = (i + 2) % num_entities;
            samples.push(TripleSample::new(
                format!("e{i}"),
                "rel1",
                format!("e{next}"),
            ));
        }
    }

    println!(
        "Configured {num_workers} workers, {num_shards} shards, {} samples\n",
        samples.len()
    );

    let mut workers = Vec::with_capacity(num_workers);
    for i in 0..num_workers {
        workers.push(Worker::new(
            WorkerConfig {
                worker_id: i as u32,
                max_steps: 20,
                margin: 1.0,
                l2_reg: 1e-4,
                seed: 1 + i as u64,
            },
            Arc::clone(&server),
            samples.clone(),
        ));
    }

    let losses = oxirs_embed::distributed_training::worker::run_workers(workers).await?;
    let stats = server.stats().await;
    let steps = server.shard_steps().await;

    println!("=== Per-worker Loss Summary ===");
    for l in &losses {
        println!(
            "  worker {:>2}: samples={:>5} mean_loss={:.4}",
            l.worker_id,
            l.samples,
            l.mean()
        );
    }
    println!();
    println!("=== Parameter Server Stats ===");
    println!("  total_pulls         : {}", stats.total_pulls);
    println!("  total_pushes        : {}", stats.total_pushes);
    println!("  barriers_completed  : {}", stats.barriers_completed);
    println!("  max_staleness_seen  : {}", stats.max_staleness_observed);
    println!("  per-shard steps     : {steps:?}");
    println!();
    println!("=== Demo Complete ===");
    Ok(())
}
