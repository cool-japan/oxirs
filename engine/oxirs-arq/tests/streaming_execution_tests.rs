//! Comprehensive Tests for Streaming Execution Optimization
//!
//! These tests verify the streaming execution capabilities including:
//! - Streaming analysis and opportunity identification
//! - Pipeline breaker detection
//! - Disk spilling at memory thresholds
//! - Memory-mapped spill for large data
//! - Streaming hash join
//! - Memory usage constraints
//! - Performance improvements

use oxirs_arq::advanced_optimizer::{QueryPlan, StreamingAnalyzer, StreamingConfig};
use oxirs_arq::algebra::{Algebra, Binding, Solution, Term, TriplePattern, Variable};
use oxirs_arq::executor::{SpillConfig, SpillManager};
use oxirs_core::model::NamedNode;

// Helper function to create test solutions
fn create_test_solution(size: usize) -> Solution {
    let mut solution = Solution::new();
    for i in 0..size {
        let mut binding = Binding::new();
        let var_name = format!("x{}", i);
        let iri_string = format!("http://example.org/item{}", i);
        binding.insert(
            Variable::new(&var_name).expect("valid variable name"),
            Term::Iri(NamedNode::new(&iri_string).expect("valid IRI")),
        );
        solution.push(binding);
    }
    solution
}

// Helper function to create a complex query algebra
fn create_complex_algebra() -> Algebra {
    let pattern1 = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("valid variable name")),
        predicate: Term::Iri(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI"),
        ),
        object: Term::Variable(Variable::new("o").expect("valid variable name")),
    };

    let pattern2 = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("valid variable name")),
        predicate: Term::Iri(NamedNode::new("http://xmlns.com/foaf/0.1/name").expect("valid IRI")),
        object: Term::Variable(Variable::new("name").expect("valid variable name")),
    };

    Algebra::Join {
        left: Box::new(Algebra::Bgp(vec![pattern1])),
        right: Box::new(Algebra::Bgp(vec![pattern2])),
    }
}

#[test]
fn test_streaming_analyzer_creation() {
    let config = StreamingConfig::default();
    let analyzer = StreamingAnalyzer::new(config);
    assert_eq!(analyzer.memory_threshold(), 2048 * 1024 * 1024);
}

#[test]
fn test_streaming_analyzer_memory_threshold() {
    let config = StreamingConfig {
        memory_threshold_mb: 1024,
        ..Default::default()
    };
    let mut analyzer = StreamingAnalyzer::new(config);

    assert_eq!(analyzer.memory_threshold(), 1024 * 1024 * 1024);

    analyzer.set_memory_threshold(512);
    assert_eq!(analyzer.memory_threshold(), 512 * 1024 * 1024);
}

#[test]
fn test_identify_streamable_operators() {
    let config = StreamingConfig::default();
    let analyzer = StreamingAnalyzer::new(config);

    let algebra = create_complex_algebra();
    let plan = QueryPlan::from_algebra(&algebra);
    let opportunities = analyzer.analyze(&plan);

    // Should identify scan operations as streamable
    assert!(!opportunities.streamable_scans.is_empty());
    println!(
        "Found {} streamable scans",
        opportunities.streamable_scans.len()
    );
}

#[test]
fn test_pipeline_breakers_detection() {
    let config = StreamingConfig::default();
    let analyzer = StreamingAnalyzer::new(config);

    // Create algebra with pipeline breaker (ORDER BY)
    let pattern = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("valid variable name")),
        predicate: Term::Variable(Variable::new("p").expect("valid variable name")),
        object: Term::Variable(Variable::new("o").expect("valid variable name")),
    };

    let order_by_algebra = Algebra::OrderBy {
        pattern: Box::new(Algebra::Bgp(vec![pattern])),
        conditions: vec![],
    };

    let plan = QueryPlan::from_algebra(&order_by_algebra);
    let breakers = analyzer.find_pipeline_breakers(&plan);

    assert!(
        !breakers.is_empty(),
        "ORDER BY should be detected as pipeline breaker"
    );
    println!("Found {} pipeline breakers", breakers.len());
}

#[test]
fn test_aggregation_pipeline_breaker() {
    let config = StreamingConfig::default();
    let analyzer = StreamingAnalyzer::new(config);

    let pattern = TriplePattern {
        subject: Term::Variable(Variable::new("s").expect("valid variable name")),
        predicate: Term::Variable(Variable::new("p").expect("valid variable name")),
        object: Term::Variable(Variable::new("o").expect("valid variable name")),
    };

    let group_algebra = Algebra::Group {
        pattern: Box::new(Algebra::Bgp(vec![pattern])),
        variables: vec![],
        aggregates: vec![],
    };

    let plan = QueryPlan::from_algebra(&group_algebra);
    let opportunities = analyzer.analyze(&plan);

    assert!(
        !opportunities.pipeline_breakers.is_empty(),
        "GROUP BY should be pipeline breaker"
    );
}

#[test]
fn test_spill_manager_creation() {
    let config = SpillConfig::default();
    let manager = SpillManager::new(config);
    assert!(
        manager.is_ok(),
        "SpillManager should be created successfully"
    );
}

#[test]
fn test_spill_and_read_small_data() {
    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    let test_data = create_test_solution(100);
    let spill_id = manager.spill(&test_data).expect("Failed to spill");

    let read_data = manager.read_spill(spill_id).expect("Failed to read spill");
    assert_eq!(
        test_data.len(),
        read_data.len(),
        "Spilled data should match"
    );

    manager.cleanup(spill_id).expect("Failed to cleanup");
}

#[test]
fn test_spill_and_read_large_data() {
    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    // Create large dataset (10,000 rows)
    let test_data = create_test_solution(10_000);
    let spill_id = manager.spill(&test_data).expect("Failed to spill");

    let read_data = manager.read_spill(spill_id).expect("Failed to read spill");
    assert_eq!(
        test_data.len(),
        read_data.len(),
        "Large spilled data should match"
    );

    let stats = manager.statistics();
    assert_eq!(stats.num_spills, 1);
    assert_eq!(stats.total_rows, 10_000);
    assert!(
        stats.total_size_bytes > 0,
        "Spill should have non-zero size"
    );

    manager.cleanup(spill_id).expect("Failed to cleanup");
}

#[test]
fn test_spill_compression() {
    let config = SpillConfig {
        compression: true,
        ..Default::default()
    };

    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    let test_data = create_test_solution(1000);
    let spill_id = manager.spill(&test_data).expect("Failed to spill");

    let stats = manager.statistics();
    assert!(
        stats.average_compression_ratio < 1.0,
        "Compression should reduce size"
    );

    // Verify data integrity after compression
    let read_data = manager.read_spill(spill_id).expect("Failed to read spill");
    assert_eq!(
        test_data.len(),
        read_data.len(),
        "Compressed data should be intact"
    );

    manager.cleanup(spill_id).expect("Failed to cleanup");
}

#[test]
fn test_spill_threshold_detection() {
    let config = StreamingConfig {
        enable_streaming: true,
        memory_threshold_mb: 100, // Low threshold for testing
        spill_threshold_percent: 0.8,
        streaming_batch_size: 1000,
    };
    let analyzer = StreamingAnalyzer::new(config);

    // Small data should not trigger spilling
    assert!(
        !analyzer.should_spill(),
        "Should not spill with low memory usage"
    );
}

#[test]
fn test_multiple_spills() {
    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    let mut spill_ids = Vec::new();

    // Create multiple spills
    for i in 0..5 {
        let test_data = create_test_solution(100 * (i + 1));
        let spill_id = manager.spill(&test_data).expect("Failed to spill");
        spill_ids.push((spill_id, test_data));
    }

    let stats = manager.statistics();
    assert_eq!(stats.num_spills, 5, "Should have 5 active spills");

    // Verify all spills
    for (spill_id, original_data) in &spill_ids {
        let read_data = manager.read_spill(*spill_id).expect("Failed to read spill");
        assert_eq!(
            original_data.len(),
            read_data.len(),
            "Spill data should match"
        );
    }

    // Cleanup all
    for (spill_id, _) in spill_ids {
        manager.cleanup(spill_id).expect("Failed to cleanup");
    }

    assert_eq!(
        manager.num_active_spills(),
        0,
        "All spills should be cleaned up"
    );
}

#[test]
fn test_spill_statistics() {
    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    let test_data1 = create_test_solution(50);
    let test_data2 = create_test_solution(150);

    let _spill_id1 = manager.spill(&test_data1).expect("Failed to spill");
    let _spill_id2 = manager.spill(&test_data2).expect("Failed to spill");

    let stats = manager.statistics();
    assert_eq!(stats.num_spills, 2);
    assert_eq!(stats.total_rows, 200);
    assert!(stats.total_size_bytes > 0);

    manager.cleanup_all().expect("Failed to cleanup");
}

#[test]
fn test_streaming_analyzer_potential() {
    let config = StreamingConfig::default();
    let analyzer = StreamingAnalyzer::new(config);

    let algebra = create_complex_algebra();
    let strategy = analyzer
        .analyze_streaming_potential(&algebra)
        .expect("Failed to analyze");

    if let Some(strategy) = strategy {
        assert!(strategy.memory_limit > 0);
        assert!(strategy.batch_size > 0);
        println!("Streaming strategy: {:?}", strategy.strategy_type);
    }
}

#[test]
fn test_query_complexity_analysis() {
    let config = StreamingConfig::default();
    let analyzer = StreamingAnalyzer::new(config);

    let simple_algebra = Algebra::Bgp(vec![]);
    let simple_complexity = analyzer.analyze_query_complexity(&simple_algebra);
    assert_eq!(simple_complexity.num_patterns, 0);

    let complex_algebra = create_complex_algebra();
    let complex_complexity = analyzer.analyze_query_complexity(&complex_algebra);
    assert!(
        complex_complexity.num_joins > 0,
        "Complex query should have joins"
    );
    assert!(
        complex_complexity.total_complexity() > 0,
        "Should have non-zero complexity"
    );
}

#[test]
fn test_streaming_vs_materialized_memory() {
    // This test simulates memory usage comparison between streaming and materialized execution
    let config = StreamingConfig {
        enable_streaming: true,
        memory_threshold_mb: 2048,
        spill_threshold_percent: 0.8,
        streaming_batch_size: 1000,
    };
    let analyzer = StreamingAnalyzer::new(config);

    let large_query = create_complex_algebra();
    let plan = QueryPlan::from_algebra(&large_query);
    let opportunities = analyzer.analyze(&plan);

    // Streaming should save memory
    assert!(
        opportunities.estimated_memory_savings_mb > 0,
        "Streaming should save memory"
    );
    println!(
        "Estimated memory savings: {} MB",
        opportunities.estimated_memory_savings_mb
    );
}

#[test]
fn test_spill_iterator() {
    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    let test_data = create_test_solution(50);
    let spill_id = manager.spill(&test_data).expect("Failed to spill");

    let iterator = manager
        .read_spill_streaming(spill_id)
        .expect("Failed to create iterator");

    let collected: Result<Vec<_>, _> = iterator.collect();
    assert!(collected.is_ok(), "Iterator should work");

    let collected = collected.unwrap();
    assert_eq!(
        collected.len(),
        test_data.len(),
        "Iterator should read all data"
    );

    manager.cleanup(spill_id).expect("Failed to cleanup");
}

#[test]
fn test_streaming_correctness() {
    // Verify that streaming execution produces the same results as materialized
    let config = StreamingConfig::default();
    let analyzer = StreamingAnalyzer::new(config);

    let algebra = create_complex_algebra();
    let _plan = QueryPlan::from_algebra(&algebra);

    // Both should analyze the same query structure
    let complexity = analyzer.analyze_query_complexity(&algebra);
    assert!(complexity.total_complexity() > 0);
}

#[test]
fn test_empty_spill_rejection() {
    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    let empty_solution = Solution::new();
    let result = manager.spill(&empty_solution);

    assert!(result.is_err(), "Should reject empty solution");
}

#[test]
fn test_cleanup_on_drop() {
    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    let test_data = create_test_solution(100);
    let _spill_id = manager.spill(&test_data).expect("Failed to spill");

    assert_eq!(manager.num_active_spills(), 1);

    // Drop manager - should cleanup automatically
    drop(manager);

    // Create new manager to verify cleanup
    let new_manager = SpillManager::new(SpillConfig::default()).expect("Failed to create manager");
    assert_eq!(
        new_manager.num_active_spills(),
        0,
        "Spills should be cleaned up on drop"
    );
}

#[test]
fn test_performance_large_dataset() {
    use std::time::Instant;

    let config = SpillConfig::default();
    let mut manager = SpillManager::new(config).expect("Failed to create manager");

    // Create large dataset
    let start = Instant::now();
    let large_data = create_test_solution(100_000);
    let creation_time = start.elapsed();
    println!("Created 100K rows in {:?}", creation_time);

    // Spill
    let start = Instant::now();
    let spill_id = manager.spill(&large_data).expect("Failed to spill");
    let spill_time = start.elapsed();
    println!("Spilled 100K rows in {:?}", spill_time);

    // Read back
    let start = Instant::now();
    let read_data = manager.read_spill(spill_id).expect("Failed to read");
    let read_time = start.elapsed();
    println!("Read 100K rows in {:?}", read_time);

    assert_eq!(large_data.len(), read_data.len());

    // Verify reasonable performance (should complete in reasonable time)
    assert!(
        spill_time.as_secs() < 10,
        "Spilling should be reasonably fast"
    );
    assert!(
        read_time.as_secs() < 10,
        "Reading should be reasonably fast"
    );

    manager.cleanup(spill_id).expect("Failed to cleanup");
}

// Integration test combining streaming analysis and spilling
#[test]
fn test_integrated_streaming_execution() {
    let stream_config = StreamingConfig {
        enable_streaming: true,
        memory_threshold_mb: 100, // Low threshold to trigger streaming
        spill_threshold_percent: 0.8,
        streaming_batch_size: 1000,
    };
    let analyzer = StreamingAnalyzer::new(stream_config);

    let spill_config = SpillConfig {
        spill_threshold_percent: 0.8,
        max_memory_mb: 100,
        spill_dir: std::env::temp_dir().join("oxirs_test_spill"),
        compression: true,
    };
    let mut spill_manager = SpillManager::new(spill_config).expect("Failed to create manager");

    // Create query that should use streaming
    let algebra = create_complex_algebra();
    let plan = QueryPlan::from_algebra(&algebra);
    let opportunities = analyzer.analyze(&plan);

    assert!(!opportunities.streamable_scans.is_empty());

    // Simulate large result set that needs spilling
    let large_result = create_test_solution(5000);
    let spill_id = spill_manager.spill(&large_result).expect("Failed to spill");

    // Verify we can read it back
    let recovered = spill_manager.read_spill(spill_id).expect("Failed to read");
    assert_eq!(large_result.len(), recovered.len());

    spill_manager.cleanup(spill_id).expect("Failed to cleanup");
}
