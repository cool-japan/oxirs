//! # Performance Optimization Example
//!
//! This example demonstrates performance optimization techniques:
//! 1. Parallel batch processing
//! 2. Model caching strategies
//! 3. Performance profiling
//! 4. Metrics collection
//! 5. Memory-efficient operations
//!
//! Run with: `cargo run --example performance_optimization --release`

use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};
use oxirs_samm::performance::{BatchProcessor, ModelCache, PerformanceConfig};
use oxirs_samm::production::{
    init_production, LogLevel, MetricsCollector, OperationType, ProductionConfig,
};
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SAMM Performance Optimization Example ===\n");
    println!("Note: Run with --release flag for accurate performance measurements\n");

    // Initialize production monitoring
    let prod_config = ProductionConfig {
        log_level: LogLevel::Info,
        profiling_enabled: true,
        benchmarking_enabled: true,
        ..Default::default()
    };
    init_production(&prod_config)?;

    let metrics = MetricsCollector::global();

    // Step 1: Baseline - Sequential processing
    println!("Step 1: Baseline - Sequential processing...");
    let models = create_sample_models(100);

    let start = Instant::now();
    let mut results = Vec::new();
    for model in &models {
        metrics.record_operation(OperationType::Parse);
        let result = process_model(model);
        results.push(result);
    }
    let sequential_duration = start.elapsed();

    println!("  ✓ Processed {} models sequentially", models.len());
    println!(
        "    Time: {:.2}ms",
        sequential_duration.as_secs_f64() * 1000.0
    );
    println!(
        "    Average: {:.2}ms per model",
        sequential_duration.as_secs_f64() * 1000.0 / models.len() as f64
    );
    println!();

    // Step 2: Parallel batch processing
    println!("Step 2: Parallel batch processing with Rayon...");

    let perf_config = PerformanceConfig {
        parallel_processing: true,
        num_workers: num_cpus::get(),
        cache_size: 100,
        profiling_enabled: true,
        ..Default::default()
    };

    let processor = BatchProcessor::new(perf_config);

    let start = Instant::now();
    let parallel_results = processor
        .process_batch(models.clone(), |model| {
            metrics.record_operation(OperationType::Parse);
            Ok(process_model(model))
        })
        .await?;
    let parallel_duration = start.elapsed();

    println!("  ✓ Processed {} models in parallel", models.len());
    println!(
        "    Time: {:.2}ms",
        parallel_duration.as_secs_f64() * 1000.0
    );
    println!(
        "    Average: {:.2}ms per model",
        parallel_duration.as_secs_f64() * 1000.0 / models.len() as f64
    );

    let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!("    Speedup: {:.2}x faster", speedup);
    println!();

    // Step 3: Model caching
    println!("Step 3: Demonstrating model caching...");

    let cache = ModelCache::new(50);

    // First access - cache miss
    let start = Instant::now();
    for i in 0..10 {
        let key = format!("urn:samm:com.example:1.0.0#Model{}", i);
        let _ = cache.get(&key);
    }
    let miss_duration = start.elapsed();

    // Populate cache
    for i in 0..10 {
        let key = format!("urn:samm:com.example:1.0.0#Model{}", i);
        cache.put(key, Arc::new(format!("Model content {}", i)));
    }

    // Second access - cache hit
    let start = Instant::now();
    for i in 0..10 {
        let key = format!("urn:samm:com.example:1.0.0#Model{}", i);
        let _ = cache.get(&key);
    }
    let hit_duration = start.elapsed();

    println!("  Cache miss duration: {:.2}µs", miss_duration.as_micros());
    println!("  Cache hit duration: {:.2}µs", hit_duration.as_micros());
    println!(
        "  Speedup from caching: {:.2}x",
        miss_duration.as_secs_f64() / hit_duration.as_secs_f64()
    );
    println!();

    // Cache statistics
    let stats = cache.stats();
    println!("  Cache Statistics:");
    println!("    Size: {}", stats.size);
    println!("    Capacity: {}", stats.max_size);
    println!("    Hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!();

    // Step 4: Performance profiling
    println!("Step 4: Manual performance profiling...");

    let start = Instant::now();
    let models_for_prof = create_sample_models(50);
    let profiled_result: usize = models_for_prof.iter().map(|m| process_model(m)).sum();
    let prof_duration = start.elapsed();

    println!("  ✓ Profiled model processing");
    println!("    Result: {} total characters processed", profiled_result);
    println!(
        "    Duration: {:.2}ms",
        prof_duration.as_secs_f64() * 1000.0
    );
    println!();

    // Async profiling
    let start = Instant::now();
    let models_for_async = create_sample_models(50);
    let processor_async = BatchProcessor::new(PerformanceConfig {
        parallel_processing: true,
        num_workers: 4,
        ..Default::default()
    });

    let async_result = processor_async
        .process_batch(models_for_async, |model| Ok(process_model(model)))
        .await?;
    let async_duration = start.elapsed();

    println!("  ✓ Profiled async batch processing");
    println!("    Processed {} models", async_result.len());
    println!(
        "    Duration: {:.2}ms",
        async_duration.as_secs_f64() * 1000.0
    );
    println!();

    // Step 5: Memory-efficient operations
    println!("Step 5: Memory-efficient string operations...");

    let large_models = create_sample_models(1000);

    // Count lines efficiently using bytecount
    let start = Instant::now();
    let total_lines: usize = large_models
        .iter()
        .map(|m| bytecount::count(m.as_bytes(), b'\n'))
        .sum();
    let bytecount_duration = start.elapsed();

    println!("  ✓ Counted lines in {} models", large_models.len());
    println!("    Total lines: {}", total_lines);
    println!(
        "    Time: {:.2}ms (using bytecount)",
        bytecount_duration.as_secs_f64() * 1000.0
    );
    println!();

    // Step 6: Metrics collection and reporting
    println!("Step 6: Production metrics...");

    let snapshot = metrics.snapshot();
    println!("  Metrics Summary:");
    println!("    Total operations: {}", snapshot.operations_total);
    println!("    Parse operations: {}", snapshot.parse_operations);
    println!("    Codegen operations: {}", snapshot.codegen_operations);
    println!(
        "    Validation operations: {}",
        snapshot.validation_operations
    );
    println!("    Errors: {}", snapshot.errors_total);
    println!("    Error rate: {:.2}%", snapshot.error_rate() * 100.0);
    let throughput = snapshot.ops_per_second();
    if throughput > 0.0 {
        println!("    Throughput: {:.2} ops/sec", throughput);
    }
    println!();

    // Step 7: Configuration tuning recommendations
    println!("Step 7: Performance tuning recommendations...");
    println!();

    println!("  Recommendations based on system:");
    println!("    CPU cores: {}", num_cpus::get());
    println!(
        "    Recommended workers: {}",
        num_cpus::get().saturating_sub(1).max(1)
    );
    println!("    Recommended cache size: 100-1000 entries");
    println!("    Enable parallel processing: Yes (for > 10 models)");
    println!("    Enable profiling: Development only");
    println!();

    if speedup > 1.5 {
        println!("  ✓ Parallel processing provides significant speedup");
    } else {
        println!("  ℹ Overhead may outweigh benefits for small batches");
    }

    println!();

    // Summary
    println!("=== Performance Summary ===");
    println!("Optimization techniques demonstrated:");
    println!("  ✓ Parallel batch processing ({:.2}x speedup)", speedup);
    println!(
        "  ✓ Model caching (hit rate: {:.1}%)",
        stats.hit_rate * 100.0
    );
    println!("  ✓ Manual performance profiling");
    println!("  ✓ Memory-efficient operations");
    println!("  ✓ Production metrics collection");
    println!("  ✓ Configuration tuning");
    println!();
    println!("Key Takeaways:");
    println!("  • Use parallel processing for batches > 10 models");
    println!("  • Cache frequently accessed models");
    println!("  • Profile to identify bottlenecks");
    println!("  • Monitor production metrics");
    println!("  • Tune worker count based on CPU cores");

    Ok(())
}

/// Create sample models for testing
fn create_sample_models(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            format!(
                r#"@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .
@prefix samm-c: <urn:samm:org.eclipse.esmf.samm:characteristic:2.3.0#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:samm:com.example:1.0.0#> .

:Model{} a samm:Aspect ;
    samm:properties ( :property{} ) ;
    samm:preferredName "Model {}"@en .

:property{} a samm:Property ;
    samm:characteristic samm-c:Text .
"#,
                i, i, i, i
            )
        })
        .collect()
}

/// Simulate model processing
fn process_model(model: &str) -> usize {
    // Simulate some processing work
    model.chars().filter(|c| c.is_alphanumeric()).count()
}
