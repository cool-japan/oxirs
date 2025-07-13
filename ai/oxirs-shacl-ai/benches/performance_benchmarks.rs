//! Performance benchmarks for oxirs-shacl-ai
//!
//! This module contains comprehensive benchmarks for measuring the performance
//! of AI-powered SHACL validation and shape generation components.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_core::model::{Literal, NamedNode, Term, Triple};
use oxirs_core::Store;
use oxirs_shacl_ai::insights::ValidationData;
use oxirs_shacl_ai::*;
use std::time::Duration;
use uuid;

/// Generate test RDF data for benchmarking
fn generate_benchmark_data(size: usize) -> Vec<Triple> {
    let mut triples = Vec::new();

    for i in 0..size {
        let subject = NamedNode::new(format!("http://example.org/person/{}", i)).unwrap();
        let name_predicate = NamedNode::new("http://example.org/name").unwrap();
        let age_predicate = NamedNode::new("http://example.org/age").unwrap();
        let email_predicate = NamedNode::new("http://example.org/email").unwrap();
        let type_predicate =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let person_class = NamedNode::new("http://example.org/Person").unwrap();

        // Add type triple
        triples.push(Triple::new(subject.clone(), type_predicate, person_class));

        // Add name triple
        let name_literal = Literal::new_simple_literal(format!("Person {}", i));
        triples.push(Triple::new(subject.clone(), name_predicate, name_literal));

        // Add age triple
        let age_literal = Literal::new_typed_literal(
            format!("{}", 20 + (i % 50)),
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        );
        triples.push(Triple::new(subject.clone(), age_predicate, age_literal));

        // Add email triple (for variety)
        let email_literal = Literal::new_simple_literal(format!("person{}@example.org", i));
        triples.push(Triple::new(subject, email_predicate, email_literal));
    }

    triples
}

/// Create a store with benchmark data
fn create_benchmark_store(size: usize) -> impl Store {
    let store = oxirs_core::rdf_store::RdfStore::new().expect("Failed to create store");
    let data = generate_benchmark_data(size);

    for triple in data {
        store
            .insert(&triple.into())
            .expect("Failed to insert benchmark data");
    }

    store
}

/// Benchmark shape learning performance
fn bench_shape_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_learning");
    group.measurement_time(Duration::from_secs(30));

    for size in [100, 500, 1000, 2000].iter() {
        let store = create_benchmark_store(*size);
        let mut learner = learning::ShapeLearner::new();

        group.bench_with_input(
            BenchmarkId::new("learn_shapes_from_store", size),
            size,
            |b, _| b.iter(|| black_box(learner.learn_shapes_from_store(&store, None))),
        );
    }

    group.finish();
}

/// Benchmark pattern discovery performance
fn bench_pattern_discovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_discovery");
    group.measurement_time(Duration::from_secs(30));

    for size in [100, 500, 1000, 2000].iter() {
        let store = create_benchmark_store(*size);
        let mut analyzer = patterns::PatternAnalyzer::new();

        group.bench_with_input(BenchmarkId::new("discover_patterns", size), size, |b, _| {
            let store2 = create_benchmark_store(*size / 2);
            b.iter(|| black_box(analyzer.discover_similar_patterns(&store, &store2)))
        });
    }

    group.finish();
}

/// Benchmark quality assessment performance
fn bench_quality_assessment(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_assessment");
    group.measurement_time(Duration::from_secs(30));

    for size in [100, 500, 1000, 2000].iter() {
        let store = create_benchmark_store(*size);
        let mut assessor = quality::QualityAssessor::new();

        // Create some dummy shapes for assessment
        let shapes = vec![]; // In practice, would have learned shapes

        group.bench_with_input(BenchmarkId::new("assess_quality", size), size, |b, _| {
            b.iter(|| black_box(assessor.assess_comprehensive_quality(&store, &shapes)))
        });
    }

    group.finish();
}

/// Benchmark validation prediction performance
fn bench_validation_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_prediction");
    group.measurement_time(Duration::from_secs(20));

    for size in [100, 500, 1000, 2000].iter() {
        let store = create_benchmark_store(*size);
        let mut predictor = prediction::ValidationPredictor::new();

        // Create some dummy shapes for prediction
        let shapes = vec![]; // In practice, would have learned shapes

        group.bench_with_input(BenchmarkId::new("predict_outcomes", size), size, |b, _| {
            let config = oxirs_shacl::ValidationConfig::default();
            b.iter(|| black_box(predictor.predict_validation_outcome(&store, &shapes, &config)))
        });
    }

    group.finish();
}

/// Benchmark optimization performance
fn bench_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization");
    group.measurement_time(Duration::from_secs(20));

    for size in [100, 500, 1000].iter() {
        let store = create_benchmark_store(*size);
        let mut optimizer = optimization::OptimizationEngine::new();

        // Create some dummy shapes for optimization
        let shapes = vec![]; // In practice, would have learned shapes

        group.bench_with_input(
            BenchmarkId::new("optimize_validation", size),
            size,
            |b, _| b.iter(|| black_box(optimizer.optimize_validation_strategy(&store, &shapes))),
        );
    }

    group.finish();
}

/// Benchmark analytics generation performance
fn bench_analytics(c: &mut Criterion) {
    let mut group = c.benchmark_group("analytics");
    group.measurement_time(Duration::from_secs(20));

    for size in [100, 500, 1000].iter() {
        let store = create_benchmark_store(*size);
        let mut analyzer = analytics::AnalyticsEngine::new();

        // Create some dummy validation reports
        let reports: Vec<oxirs_shacl::report::core::ValidationReport> = vec![]; // In practice, would have validation reports

        group.bench_with_input(
            BenchmarkId::new("generate_analytics", size),
            size,
            |b, _| {
                b.iter(|| {
                    let empty_shapes: Vec<oxirs_shacl::Shape> = Vec::new();
                    let empty_reports: Vec<oxirs_shacl::report::core::ValidationReport> =
                        Vec::new();
                    black_box(analyzer.generate_comprehensive_insights(
                        &store,
                        &empty_shapes,
                        &empty_reports,
                    ))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark insight generation performance
fn bench_insights(c: &mut Criterion) {
    let mut group = c.benchmark_group("insights");
    group.measurement_time(Duration::from_secs(15));

    for size in [100, 500, 1000].iter() {
        // Create dummy analytics data
        let analytics_data = analytics::ValidationInsights::default();
        let quality_report = quality::QualityReport::new();
        let patterns: Vec<patterns::Pattern> = vec![]; // Would have discovered patterns

        let insight_engine = insights::InsightGenerator::new();

        group.bench_with_input(BenchmarkId::new("generate_insights", size), size, |b, _| {
            b.iter(|| {
                let validation_data = insights::ValidationData {
                    total_validations: 100,
                    successful_validations: 95,
                    failed_validations: 5,
                    error_types: std::collections::HashMap::new(),
                    avg_validation_time: std::time::Duration::from_millis(10),
                    shapes_validated: vec![],
                    violations_by_shape: std::collections::HashMap::new(),
                    historical_success_rates: vec![],
                };
                black_box(insight_engine.generate_insights(&validation_data))
            })
        });
    }

    group.finish();
}

/// Benchmark memory usage and allocation patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("large_dataset_memory", |b| {
        b.iter(|| {
            let store = black_box(create_benchmark_store(5000));
            let assistant = ShaclAiAssistant::new();

            // This simulates memory allocation patterns for large datasets
            let _result = std::thread::scope(|_| {
                // Simulate concurrent operations that might stress memory
                black_box(store)
            });
        })
    });

    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("concurrent_pattern_discovery", |b| {
        let store = std::sync::Arc::new(create_benchmark_store(1000));

        b.iter(|| {
            let store_clone = store.clone();

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        let store_ref = store_clone.clone();
                        s.spawn(move || {
                            let mut analyzer = patterns::PatternAnalyzer::new();
                            let store2 = create_benchmark_store(500);
                            black_box(analyzer.discover_similar_patterns(&*store_ref, &store2))
                        })
                    })
                    .collect();

                for handle in handles {
                    let _ = handle.join();
                }
            });
        })
    });

    group.finish();
}

/// Benchmark data structure operations
fn bench_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_structures");

    group.bench_function("pattern_creation", |b| {
        b.iter(|| {
            let class = NamedNode::new("http://example.org/Person").unwrap();
            black_box(patterns::Pattern::ClassUsage {
                id: uuid::Uuid::new_v4().to_string(),
                class,
                instance_count: 100,
                support: 0.8,
                confidence: 0.9,
                pattern_type: patterns::PatternType::Structural,
            })
        })
    });

    group.bench_function("insight_creation", |b| {
        use std::collections::HashMap;

        b.iter(|| {
            black_box(insights::ValidationInsight {
                insight_type: insights::ValidationInsightType::LowSuccessRate,
                title: "Test Insight".to_string(),
                description: "Test description".to_string(),
                severity: analytics::InsightSeverity::High,
                confidence: 0.9,
                affected_shapes: vec![],
                recommendations: vec!["Fix issue".to_string()],
                supporting_data: HashMap::new(),
            })
        })
    });

    group.finish();
}

/// Benchmark configuration and setup operations
fn bench_configuration(c: &mut Criterion) {
    let mut group = c.benchmark_group("configuration");

    group.bench_function("assistant_creation", |b| {
        b.iter(|| {
            let config = ShaclAiConfig::default();
            black_box(ShaclAiAssistant::new())
        })
    });

    group.bench_function("config_serialization", |b| {
        let config = ShaclAiConfig::default();
        b.iter(|| black_box(serde_json::to_string(&config)))
    });

    group.bench_function("config_deserialization", |b| {
        let config = ShaclAiConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        b.iter(|| black_box(serde_json::from_str::<ShaclAiConfig>(&json)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_shape_learning,
    bench_pattern_discovery,
    bench_quality_assessment,
    bench_validation_prediction,
    bench_optimization,
    bench_analytics,
    bench_insights,
    bench_memory_usage,
    bench_concurrent_operations,
    bench_data_structures,
    bench_configuration
);

criterion_main!(benches);
