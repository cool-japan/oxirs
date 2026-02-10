//! ML Prediction Performance Benchmark
//!
//! Benchmarks for ML cost prediction system including:
//! 1. Prediction latency (target: < 5ms)
//! 2. Training throughput (target: > 1000 examples/sec)
//! 3. ML vs Heuristic accuracy comparison
//! 4. Feature extraction performance (target: < 1ms)
//! 5. Model save/load performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use oxirs_arq::advanced_optimizer::ml_predictor::{
    MLConfig, MLModelType, MLPredictor, QueryCharacteristics, TrainingExample,
};
use oxirs_arq::algebra::{Algebra, Expression, OrderCondition, Variable};

use std::time::SystemTime;

// ============================================================================
// Benchmark Utilities
// ============================================================================

/// Create a complex query with specified number of joins
fn create_complex_query(num_joins: usize) -> Algebra {
    let mut query = Algebra::Empty;

    for _ in 0..num_joins {
        query = Algebra::Join {
            left: Box::new(query),
            right: Box::new(Algebra::Empty),
        };
    }

    // Add filter and ordering
    query = Algebra::Filter {
        pattern: Box::new(query),
        condition: Expression::Variable(Variable::new("x").unwrap()),
    };

    query = Algebra::OrderBy {
        pattern: Box::new(query),
        conditions: vec![OrderCondition {
            expr: Expression::Variable(Variable::new("x").unwrap()),
            ascending: true,
        }],
    };

    query
}

/// Generate synthetic training dataset with clear linear relationships
fn generate_training_dataset(size: usize) -> Vec<TrainingExample> {
    let mut examples = Vec::with_capacity(size);

    for i in 0..size {
        let t = i as f64;

        let f0 = (t * 1.1) % 20.0 + 1.0;
        let f1 = (t * 1.3) % 15.0 + 1.0;
        let f2 = (t * 1.7) % 8.0 + 1.0;
        let f3 = (t * 2.1) % 5.0;
        let f4 = if (i % 4) == 0 { 1.0 } else { 0.0 };
        let f5 = if (i % 5) == 0 { 1.0 } else { 0.0 };
        let f6 = (t * 37.0) % 5000.0 + 1000.0;
        let f7 = f1.sqrt() + 1.0;
        let f8 = f1 / 3.0 + 0.5;
        let f9 = f1 * 0.7 + 1.0;
        let f10 = if f1 > 10.0 { 0.2 } else { 0.0 };
        let f11 = (f1 / 4.0).ceil();
        let f12 = f4 * f2 * 0.3;

        let features = vec![f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12];

        let cost = 20.0
            + f0 * 2.0
            + f1 * f1 * 1.5
            + f2 * 3.0
            + f3 * 5.0
            + f4 * 30.0
            + f5 * 10.0
            + f6.ln() * 2.0;

        examples.push(TrainingExample {
            features,
            target_cost: cost,
            actual_cost: cost,
            query_characteristics: QueryCharacteristics {
                triple_pattern_count: f0 as usize,
                join_count: f1 as usize,
                filter_count: f2 as usize,
                optional_count: f3 as usize,
                has_aggregation: f4 > 0.5,
                has_sorting: f5 > 0.5,
                estimated_cardinality: f6 as usize,
                complexity_score: cost / 10.0,
                query_graph_diameter: f7 as usize,
                avg_degree: f8,
                max_degree: f9 as usize,
            },
            timestamp: SystemTime::now(),
        });
    }

    examples
}

/// Create a trained predictor for benchmarking
fn create_trained_predictor(
    model_type: MLModelType,
    training_size: usize,
) -> MLPredictor {
    let config = MLConfig {
        model_type,
        min_examples_for_training: 50,
        feature_normalization: true,
        ..Default::default()
    };

    let mut predictor = MLPredictor::new(config).expect("Failed to create predictor");

    let training_data = generate_training_dataset(training_size);
    for example in training_data {
        predictor.add_training_example(example);
    }

    predictor
        .train_model()
        .expect("Failed to train predictor");

    predictor
}

// ============================================================================
// Benchmark 1: Prediction Latency (Target: < 5ms)
// ============================================================================

fn bench_prediction_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_prediction_latency");

    // Benchmark different query complexities
    for complexity in [1, 5, 10, 20] {
        let predictor = create_trained_predictor(MLModelType::Ridge, 200);
        let query = create_complex_query(complexity);

        group.bench_with_input(
            BenchmarkId::new("predict_cost", complexity),
            &complexity,
            |b, _| {
                let mut pred = predictor.clone();
                b.iter(|| {
                    let _ = pred.predict_cost(black_box(&query));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 2: Training Throughput (Target: > 1000 examples/sec)
// ============================================================================

fn bench_training_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_training_throughput");

    for dataset_size in [100, 500, 1000, 2000] {
        group.bench_with_input(
            BenchmarkId::new("train_model", dataset_size),
            &dataset_size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        let config = MLConfig {
                            model_type: MLModelType::Ridge,
                            min_examples_for_training: 50,
                            ..Default::default()
                        };
                        let mut predictor = MLPredictor::new(config).unwrap();
                        let training_data = generate_training_dataset(size);
                        for example in training_data {
                            predictor.add_training_example(example);
                        }
                        predictor
                    },
                    |mut predictor| {
                        let _ = predictor.train_model();
                    },
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 3: ML vs Heuristic Accuracy
// ============================================================================

fn bench_ml_vs_heuristic_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_vs_heuristic");

    let query = create_complex_query(5);

    // Benchmark heuristic cost estimation
    group.bench_function("heuristic_cost_estimate", |b| {
        let config = MLConfig {
            confidence_threshold: 1.0, // Force heuristic
            ..Default::default()
        };
        let mut predictor = MLPredictor::new(config).unwrap();

        b.iter(|| {
            let _ = predictor.predict_cost(black_box(&query));
        });
    });

    // Benchmark ML cost estimation
    group.bench_function("ml_cost_estimate", |b| {
        let mut predictor = create_trained_predictor(MLModelType::Ridge, 200);

        b.iter(|| {
            let _ = predictor.predict_cost(black_box(&query));
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 4: Feature Extraction (Target: < 1ms)
// ============================================================================

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    let config = MLConfig::default();
    let predictor = MLPredictor::new(config).unwrap();

    for complexity in [1, 5, 10, 20] {
        let query = create_complex_query(complexity);

        group.bench_with_input(
            BenchmarkId::new("extract_13_features", complexity),
            &complexity,
            |b, _| {
                b.iter(|| {
                    let _ = predictor.extract_features(black_box(&query));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 5: Model Persistence
// ============================================================================

fn bench_model_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_persistence");

    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("bench_ml_model.json");

    // Benchmark model saving
    group.bench_function("save_model", |b| {
        let predictor = create_trained_predictor(MLModelType::Ridge, 200);

        b.iter(|| {
            let _ = predictor.save_model(black_box(&model_path));
        });
    });

    // Create a model file for loading benchmark
    let predictor = create_trained_predictor(MLModelType::Ridge, 200);
    predictor
        .save_model(&model_path)
        .expect("Failed to save model");

    // Benchmark model loading
    group.bench_function("load_model", |b| {
        b.iter(|| {
            let _ = MLPredictor::load_model(black_box(&model_path));
        });
    });

    // Cleanup
    let _ = std::fs::remove_file(&model_path);

    group.finish();
}

// ============================================================================
// Benchmark 6: Online Learning Update
// ============================================================================

fn bench_online_learning_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_learning");

    let query = create_complex_query(5);

    group.bench_function("update_from_execution", |b| {
        let mut predictor = create_trained_predictor(MLModelType::Ridge, 100);

        b.iter(|| {
            let _ = predictor.update_from_execution(black_box(&query), black_box(100.0));
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 7: Batch Prediction
// ============================================================================

fn bench_batch_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_prediction");

    for batch_size in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("predict_batch", batch_size),
            &batch_size,
            |b, &size| {
                let mut predictor = create_trained_predictor(MLModelType::Ridge, 200);
                let queries: Vec<_> = (0..size).map(|i| create_complex_query(i % 10 + 1)).collect();

                b.iter(|| {
                    for query in &queries {
                        let _ = predictor.predict_cost(black_box(query));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 8: Feature Normalization
// ============================================================================

fn bench_feature_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_normalization");

    // Without normalization
    group.bench_function("without_normalization", |b| {
        let config = MLConfig {
            model_type: MLModelType::Ridge,
            feature_normalization: false,
            min_examples_for_training: 50,
            ..Default::default()
        };
        let mut predictor = MLPredictor::new(config).unwrap();
        let training_data = generate_training_dataset(100);
        for example in training_data {
            predictor.add_training_example(example);
        }

        b.iter(|| {
            let _ = predictor.train_model();
        });
    });

    // With normalization
    group.bench_function("with_normalization", |b| {
        let config = MLConfig {
            model_type: MLModelType::Ridge,
            feature_normalization: true,
            min_examples_for_training: 50,
            ..Default::default()
        };
        let mut predictor = MLPredictor::new(config).unwrap();
        let training_data = generate_training_dataset(100);
        for example in training_data {
            predictor.add_training_example(example);
        }

        b.iter(|| {
            let _ = predictor.train_model();
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 9: Model Type Comparison
// ============================================================================

fn bench_model_type_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_type_comparison");

    let query = create_complex_query(5);

    // Linear Regression
    group.bench_function("linear_regression", |b| {
        let mut predictor = create_trained_predictor(MLModelType::LinearRegression, 200);

        b.iter(|| {
            let _ = predictor.predict_cost(black_box(&query));
        });
    });

    // Ridge Regression
    group.bench_function("ridge_regression", |b| {
        let mut predictor = create_trained_predictor(MLModelType::Ridge, 200);

        b.iter(|| {
            let _ = predictor.predict_cost(black_box(&query));
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 10: Cache Performance
// ============================================================================

fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");

    let query = create_complex_query(5);

    // First prediction (not cached)
    group.bench_function("prediction_uncached", |b| {
        b.iter_with_setup(
            || create_trained_predictor(MLModelType::Ridge, 200),
            |mut predictor| {
                let _ = predictor.predict_cost(black_box(&query));
            },
        );
    });

    // Second prediction (cached)
    group.bench_function("prediction_cached", |b| {
        let mut predictor = create_trained_predictor(MLModelType::Ridge, 200);
        // Prime the cache
        let _ = predictor.predict_cost(&query);

        b.iter(|| {
            let _ = predictor.predict_cost(black_box(&query));
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_prediction_latency,
    bench_training_throughput,
    bench_ml_vs_heuristic_accuracy,
    bench_feature_extraction,
    bench_model_persistence,
    bench_online_learning_update,
    bench_batch_prediction,
    bench_feature_normalization,
    bench_model_type_comparison,
    bench_cache_performance,
);

criterion_main!(benches);
