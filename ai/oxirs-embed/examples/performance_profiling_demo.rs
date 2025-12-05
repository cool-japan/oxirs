//! Performance Profiling Demonstration
//!
//! This example demonstrates how to use the performance profiler to track
//! and analyze embedding operation performance.

use anyhow::Result;
use oxirs_embed::{
    EmbeddingModel, ModelConfig, NamedNode, OperationType, PerformanceProfiler, TransE, Triple,
};
use std::thread;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║      OxiRS Embed: Performance Profiling Demonstration             ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    // Create a performance profiler
    let profiler = PerformanceProfiler::new();

    println!("1. Basic Operation Profiling");
    println!("─────────────────────────────────────────────────────────────────────\n");

    // Profile a training operation
    {
        let _timer = profiler.start_operation(OperationType::Training);
        thread::sleep(Duration::from_millis(100));
        // Timer auto-records when dropped
    }

    // Profile an inference operation
    {
        let _timer = profiler.start_operation(OperationType::Inference);
        thread::sleep(Duration::from_millis(50));
    }

    // Profile similarity computations
    for _ in 0..10 {
        let _timer = profiler.start_operation(OperationType::SimilarityComputation);
        thread::sleep(Duration::from_millis(10));
    }

    // Manual error recording
    profiler.record_operation(OperationType::Training, Duration::from_millis(200), true);

    println!("Recorded operations:");
    println!("  - 1 training operation (100ms)");
    println!("  - 1 inference operation (50ms)");
    println!("  - 10 similarity computations (10ms each)");
    println!("  - 1 failed training operation (200ms)\n");

    // Calculate percentiles
    profiler.calculate_percentiles(OperationType::SimilarityComputation);

    println!("2. Statistics for Individual Operation Types");
    println!("─────────────────────────────────────────────────────────────────────\n");

    if let Some(stats) = profiler.get_stats(OperationType::Training) {
        println!("Training Statistics:");
        println!("  Total operations: {}", stats.total_count);
        println!("  Success rate: {:.2}%", stats.success_rate());
        println!(
            "  Average duration: {:.2}ms",
            stats.average_duration.as_secs_f64() * 1000.0
        );
        println!("  Throughput: {:.2} ops/sec\n", stats.throughput());
    }

    if let Some(stats) = profiler.get_stats(OperationType::SimilarityComputation) {
        println!("Similarity Computation Statistics:");
        println!("  Total operations: {}", stats.total_count);
        println!(
            "  Min duration: {:.2}ms",
            stats.min_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  Max duration: {:.2}ms",
            stats.max_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  Average duration: {:.2}ms",
            stats.average_duration.as_secs_f64() * 1000.0
        );
        println!(
            "  P95 duration: {:.2}ms",
            stats.percentile_95.as_secs_f64() * 1000.0
        );
        println!(
            "  P99 duration: {:.2}ms\n",
            stats.percentile_99.as_secs_f64() * 1000.0
        );
    }

    println!("3. Real-World Example: Profile TransE Training");
    println!("─────────────────────────────────────────────────────────────────────\n");

    let profiler2 = PerformanceProfiler::new();

    // Create and train a TransE model
    let config = ModelConfig::default()
        .with_dimensions(64)
        .with_learning_rate(0.01)
        .with_max_epochs(10)
        .with_batch_size(100);

    let mut model = TransE::new(config);

    // Add training data with profiling
    {
        let _timer = profiler2.start_operation(OperationType::Custom("DataLoading".to_string()));

        for i in 0..50 {
            let triple = Triple::new(
                NamedNode::new(&format!("http://example.org/entity_{}", i))?,
                NamedNode::new("http://example.org/related_to")?,
                NamedNode::new(&format!("http://example.org/entity_{}", (i + 1) % 50))?,
            );
            model.add_triple(triple)?;
        }
    }

    println!("Added 50 training triples\n");

    // Profile training
    let training_stats = {
        let _timer = profiler2.start_operation(OperationType::Training);
        model.train(None).await?
    };

    println!("Training completed:");
    println!("  Epochs: {}", training_stats.epochs_completed);
    println!("  Final loss: {:.4}", training_stats.final_loss);
    println!(
        "  Training time: {:.2}s\n",
        training_stats.training_time_seconds
    );

    // Profile predictions
    for i in 0..5 {
        let _timer = profiler2.start_operation(OperationType::Prediction);
        let _predictions = model.predict_objects(
            &format!("http://example.org/entity_{}", i),
            "http://example.org/related_to",
            3,
        )?;
    }

    // Profile entity embedding retrieval
    for i in 0..10 {
        let _timer = profiler2.start_operation(OperationType::EntityEmbedding);
        let _embedding = model.get_entity_embedding(&format!("http://example.org/entity_{}", i))?;
    }

    println!("4. Comprehensive Performance Report");
    println!("─────────────────────────────────────────────────────────────────────\n");

    let report = profiler2.generate_report();
    println!("{}", report.summary());

    println!("\n5. Export to JSON");
    println!("─────────────────────────────────────────────────────────────────────\n");

    let json = profiler2.export_json()?;
    println!("JSON export (first 500 chars):");
    println!("{}\n", &json[..json.len().min(500)]);

    println!("6. Profiler Management");
    println!("─────────────────────────────────────────────────────────────────────\n");

    let mut profiler3 = PerformanceProfiler::new();

    // Disable profiling
    profiler3.disable();
    profiler3.record_operation(OperationType::Training, Duration::from_millis(100), false);
    println!("Disabled profiling - operations not recorded");
    println!("  Operations recorded: {}", profiler3.get_all_stats().len());

    // Re-enable and record
    profiler3.enable();
    profiler3.record_operation(OperationType::Training, Duration::from_millis(100), false);
    println!("Enabled profiling - operations recorded");
    println!(
        "  Operations recorded: {}\n",
        profiler3.get_all_stats().len()
    );

    // Reset statistics
    profiler3.reset();
    println!("Reset profiler");
    println!(
        "  Operations recorded: {}\n",
        profiler3.get_all_stats().len()
    );

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                    Demonstration Complete!                         ║");
    println!("╚════════════════════════════════════════════════════════════════════╝\n");

    println!("Key Takeaways:");
    println!("  • Use PerformanceProfiler to track embedding operation performance");
    println!("  • OperationTimer auto-records when dropped (RAII pattern)");
    println!("  • Calculate percentiles for detailed latency analysis");
    println!("  • Generate comprehensive reports with throughput and success rates");
    println!("  • Export statistics to JSON for external analysis");
    println!("  • Enable/disable profiling dynamically for production use\n");

    Ok(())
}
