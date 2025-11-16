//! Memory Leak Tests for OxiRS SAMM
//!
//! Basic memory safety tests to ensure operations can be repeated without issues.

use oxirs_samm::metamodel::{Aspect, ModelElement};
use oxirs_samm::parser::ModelResolver;
use oxirs_samm::performance::{BatchProcessor, ModelCache, PerformanceConfig};
use oxirs_samm::production::{MetricsCollector, OperationType};
use std::sync::Arc;

/// Test that aspect creation and destruction works repeatedly
#[test]
fn test_aspect_repeated_creation() {
    // Create and drop many aspects - should not leak or panic
    for i in 0..1000 {
        let aspect = Aspect::new(format!("urn:samm:org.test:1.0.0#Aspect{}", i));
        assert!(!aspect.urn().is_empty());
        drop(aspect);
    }
}

/// Test that model cache handles many operations correctly
#[test]
fn test_model_cache_stress() {
    let cache = ModelCache::new(10);

    // Fill cache beyond capacity multiple times
    for i in 0..100 {
        let urn = format!("urn:samm:org.test:1.0.0#Model{}", i);
        let content = Arc::new(format!("Content for model {}", i));
        cache.put(urn, content);
    }

    // Cache should still be functional
    let stats = cache.stats();
    assert!(stats.size <= stats.max_size);

    // Clear cache
    cache.clear();
    let stats_after = cache.stats();
    assert_eq!(stats_after.size, 0);
}

/// Test that batch processor handles many batches
#[tokio::test]
async fn test_batch_processor_stress() {
    let config = PerformanceConfig {
        parallel_processing: true,
        chunk_size: 10,
        cache_size: 20,
        ..Default::default()
    };

    let processor = BatchProcessor::new(config);

    // Process many batches
    for batch_num in 0..10 {
        let models: Vec<String> = (0..50)
            .map(|i| format!("Model content {} - {}", batch_num, i))
            .collect();

        let results = processor
            .process_batch(models, |content| Ok(content.len()))
            .await;

        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 50);
    }
}

/// Test that model resolver handles many failed resolutions
#[tokio::test]
async fn test_model_resolver_stress() {
    let mut resolver = ModelResolver::new();

    // Add some remote bases
    for i in 0..5 {
        resolver.add_remote_base(format!("https://test-{}.example.com/models/", i));
    }

    // Attempt many URN resolutions (will fail but shouldn't panic)
    for i in 0..50 {
        let urn = format!("urn:samm:org.test:1.0.0#Model{}", i);
        let _ = resolver.load_element(&urn).await;
    }

    // Clear caches should work
    resolver.clear_cache();
    let stats = resolver.cache_stats();
    assert_eq!(stats.content_cache_size, 0);
}

/// Test that metrics collector handles extensive use
#[test]
fn test_metrics_collector_stress() {
    let metrics = MetricsCollector::global();

    // Record many operations
    for _ in 0..10000 {
        metrics.record_operation(OperationType::Parse);
        metrics.record_operation_with_duration(OperationType::CodeGeneration, 10.5);
        metrics.record_operation(OperationType::Validation);
        metrics.record_error();
    }

    // Get snapshots multiple times
    for _ in 0..100 {
        let snapshot = metrics.snapshot();
        assert!(snapshot.operations_total > 0);
    }
}

/// Test that string utilities handle large inputs
#[test]
fn test_string_utils_large_inputs() {
    use oxirs_samm::performance::string_utils;

    // Process large strings repeatedly
    for _ in 0..100 {
        let large_content = "line\n".repeat(10000);

        let line_count = string_utils::count_lines_efficient(&large_content);
        assert!(line_count > 0);

        let contains_result = string_utils::simd_contains(&large_content, "line");
        assert!(contains_result);

        let parts = string_utils::parallel_split(&large_content, '\n');
        assert!(!parts.is_empty());
    }
}

/// Test concurrent cache access
#[tokio::test]
async fn test_concurrent_cache_access() {
    let cache = Arc::new(ModelCache::new(50));
    let mut handles = vec![];

    for task_id in 0..10 {
        let cache_clone = Arc::clone(&cache);
        let handle = tokio::spawn(async move {
            for i in 0..100 {
                let urn = format!("urn:samm:org.test:1.0.0#Task{}Model{}", task_id, i);
                let content = Arc::new(format!("Content {} {}", task_id, i));
                cache_clone.put(urn.clone(), content);
                let _ = cache_clone.get(&urn);
            }
        });
        handles.push(handle);
    }

    // All tasks should complete successfully
    for handle in handles {
        handle.await.unwrap();
    }

    // Cache should be in valid state
    let stats = cache.stats();
    assert!(stats.size <= stats.max_size);
}

/// Stress test with batch processing
#[tokio::test]
async fn test_full_stress() {
    let config = PerformanceConfig {
        parallel_processing: true,
        cache_size: 100,
        ..Default::default()
    };

    let processor = BatchProcessor::new(config);

    // Multiple rounds of processing
    for round in 0..5 {
        let models: Vec<String> = (0..200)
            .map(|i| {
                let aspect =
                    Aspect::new(format!("urn:samm:org.test:1.0.0#StressTest{}{}", round, i));
                format!("Aspect {}", aspect.urn())
            })
            .collect();

        let results = processor
            .process_batch(models, |content| Ok(content.len()))
            .await;

        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 200);
    }
}

#[test]
fn test_operation_guard_tracking() {
    use oxirs_samm::production::MetricsCollector;

    let metrics = MetricsCollector::global();
    let initial_active = metrics.active_operations();

    // Create a scope with multiple active operations
    {
        let _guard1 = MetricsCollector::start_operation();
        assert_eq!(metrics.active_operations(), initial_active + 1);

        {
            let _guard2 = MetricsCollector::start_operation();
            assert_eq!(metrics.active_operations(), initial_active + 2);

            {
                let _guard3 = MetricsCollector::start_operation();
                assert_eq!(metrics.active_operations(), initial_active + 3);
            } // guard3 dropped
            assert_eq!(metrics.active_operations(), initial_active + 2);
        } // guard2 dropped
        assert_eq!(metrics.active_operations(), initial_active + 1);
    } // guard1 dropped

    assert_eq!(metrics.active_operations(), initial_active);
    println!("✅ Operation guard RAII tracking verified");
}

#[test]
fn test_operation_guard_panic_safety() {
    use oxirs_samm::production::MetricsCollector;
    use std::panic;

    let metrics = MetricsCollector::global();
    let initial_active = metrics.active_operations();

    // Test that guard is properly dropped even on panic
    let result = panic::catch_unwind(|| {
        let _guard = MetricsCollector::start_operation();
        assert_eq!(metrics.active_operations(), initial_active + 1);
        panic!("Intentional panic for testing");
    });

    assert!(result.is_err());
    // Guard should have been dropped despite panic
    assert_eq!(metrics.active_operations(), initial_active);
    println!("✅ Operation guard panic safety verified");
}

#[test]
fn test_source_location_display() {
    use oxirs_samm::error::SourceLocation;

    // Test all Display format variations
    let loc1 = SourceLocation::with_source(10, 25, "test.ttl".to_string());
    assert_eq!(loc1.to_string(), "test.ttl:10:25");

    let loc2 = SourceLocation::new(5, 15);
    assert_eq!(loc2.to_string(), "line 5:15");

    let loc3 = SourceLocation::from_source("example.ttl".to_string());
    assert_eq!(loc3.to_string(), "example.ttl");

    let loc4 = SourceLocation {
        line: Some(42),
        column: None,
        source: Some("file.ttl".to_string()),
    };
    assert_eq!(loc4.to_string(), "file.ttl:42");

    let loc5 = SourceLocation {
        line: Some(100),
        column: None,
        source: None,
    };
    assert_eq!(loc5.to_string(), "line 100");

    let loc6 = SourceLocation {
        line: None,
        column: None,
        source: None,
    };
    assert_eq!(loc6.to_string(), "unknown location");

    println!("✅ Source location display formats verified");
}
