//! Comprehensive benchmarks for Phase 3: Knowledge Graph Embeddings
//!
//! This benchmark suite validates the performance of:
//! - TransE, DistMult, ComplEx embedding models
//! - ML Pipeline infrastructure
//! - Model selection and hyperparameter tuning
//! - Cross-validation and evaluation metrics
//!
//! Run with: `cargo bench --bench embedding_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_star::kg_embeddings::{ComplEx, DistMult, EmbeddingConfig, EmbeddingModel, TransE};
use oxirs_star::ml_embedding_pipeline::{EmbeddingPipeline, PipelineConfig};
use oxirs_star::{model::NamedNode, StarTerm, StarTriple};

/// Create test triples for benchmarking
fn create_benchmark_triples(size: usize) -> Vec<StarTriple> {
    let mut triples = Vec::with_capacity(size);

    for i in 0..size {
        let subject_id = i % 100; // 100 entities
        let predicate_id = i % 10; // 10 relations
        let object_id = (i + 1) % 100;

        triples.push(StarTriple {
            subject: StarTerm::NamedNode(NamedNode {
                iri: format!("entity_{}", subject_id),
            }),
            predicate: StarTerm::NamedNode(NamedNode {
                iri: format!("relation_{}", predicate_id),
            }),
            object: StarTerm::NamedNode(NamedNode {
                iri: format!("entity_{}", object_id),
            }),
        });
    }

    triples
}

/// Benchmark TransE training
fn bench_transe_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("transe_training");

    for size in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let triples = create_benchmark_triples(*size);
        let config = EmbeddingConfig {
            embedding_dim: 64,
            learning_rate: 0.01,
            batch_size: 128,
            num_negative_samples: 5,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut model = TransE::new(config.clone());
                black_box(model.train(&triples, 10).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark DistMult training
fn bench_distmult_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("distmult_training");

    for size in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let triples = create_benchmark_triples(*size);
        let config = EmbeddingConfig {
            embedding_dim: 64,
            learning_rate: 0.01,
            batch_size: 128,
            num_negative_samples: 5,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut model = DistMult::new(config.clone());
                black_box(model.train(&triples, 10).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark ComplEx training
fn bench_complex_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_training");

    for size in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let triples = create_benchmark_triples(*size);
        let config = EmbeddingConfig {
            embedding_dim: 64,
            learning_rate: 0.01,
            batch_size: 128,
            num_negative_samples: 5,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut model = ComplEx::new(config.clone());
                black_box(model.train(&triples, 10).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark model comparison (same dataset, all three models)
fn bench_model_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_comparison");

    let triples = create_benchmark_triples(1000);
    let config = EmbeddingConfig {
        embedding_dim: 64,
        learning_rate: 0.01,
        batch_size: 128,
        num_negative_samples: 5,
        ..Default::default()
    };

    group.bench_function("transe", |b| {
        b.iter(|| {
            let mut model = TransE::new(config.clone());
            black_box(model.train(&triples, 10).unwrap())
        });
    });

    group.bench_function("distmult", |b| {
        b.iter(|| {
            let mut model = DistMult::new(config.clone());
            black_box(model.train(&triples, 10).unwrap())
        });
    });

    group.bench_function("complex", |b| {
        b.iter(|| {
            let mut model = ComplEx::new(config.clone());
            black_box(model.train(&triples, 10).unwrap())
        });
    });

    group.finish();
}

/// Benchmark embedding dimension impact
fn bench_embedding_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_dimensions");

    let triples = create_benchmark_triples(500);

    for dim in [32, 64, 128, 256].iter() {
        let config = EmbeddingConfig {
            embedding_dim: *dim,
            learning_rate: 0.01,
            batch_size: 128,
            num_negative_samples: 5,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("transe", dim), dim, |b, _| {
            b.iter(|| {
                let mut model = TransE::new(config.clone());
                black_box(model.train(&triples, 5).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark prediction performance
fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");

    let triples = create_benchmark_triples(500);
    let config = EmbeddingConfig {
        embedding_dim: 64,
        learning_rate: 0.01,
        batch_size: 128,
        num_negative_samples: 5,
        ..Default::default()
    };

    // Pre-train models
    let mut transe = TransE::new(config.clone());
    transe.train(&triples, 10).unwrap();

    let mut distmult = DistMult::new(config.clone());
    distmult.train(&triples, 10).unwrap();

    let mut complex = ComplEx::new(config);
    complex.train(&triples, 10).unwrap();

    // Benchmark predictions
    for k in [1, 3, 10].iter() {
        group.bench_with_input(BenchmarkId::new("transe", k), k, |b, &k| {
            b.iter(|| black_box(transe.predict_tail("entity_0", "relation_0", k).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("distmult", k), k, |b, &k| {
            b.iter(|| black_box(distmult.predict_tail("entity_0", "relation_0", k).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("complex", k), k, |b, &k| {
            b.iter(|| black_box(complex.predict_tail("entity_0", "relation_0", k).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark similarity computation
fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    let triples = create_benchmark_triples(500);
    let config = EmbeddingConfig {
        embedding_dim: 64,
        learning_rate: 0.01,
        batch_size: 128,
        num_negative_samples: 5,
        ..Default::default()
    };

    // Pre-train models
    let mut transe = TransE::new(config.clone());
    transe.train(&triples, 10).unwrap();

    let mut distmult = DistMult::new(config.clone());
    distmult.train(&triples, 10).unwrap();

    let mut complex = ComplEx::new(config);
    complex.train(&triples, 10).unwrap();

    // Benchmark similarity
    group.bench_function("transe", |b| {
        b.iter(|| black_box(transe.similarity("entity_0", "entity_1").unwrap()));
    });

    group.bench_function("distmult", |b| {
        b.iter(|| black_box(distmult.similarity("entity_0", "entity_1").unwrap()));
    });

    group.bench_function("complex", |b| {
        b.iter(|| black_box(complex.similarity("entity_0", "entity_1").unwrap()));
    });

    group.finish();
}

/// Benchmark batch size impact
fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sizes");

    let triples = create_benchmark_triples(1000);

    for batch_size in [32, 64, 128, 256].iter() {
        let config = EmbeddingConfig {
            embedding_dim: 64,
            learning_rate: 0.01,
            batch_size: *batch_size,
            num_negative_samples: 5,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let mut model = TransE::new(config.clone());
                    black_box(model.train(&triples, 5).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark negative sampling impact
fn bench_negative_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("negative_samples");

    let triples = create_benchmark_triples(500);

    for num_neg in [1, 5, 10, 20].iter() {
        let config = EmbeddingConfig {
            embedding_dim: 64,
            learning_rate: 0.01,
            batch_size: 128,
            num_negative_samples: *num_neg,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(num_neg), num_neg, |b, _| {
            b.iter(|| {
                let mut model = TransE::new(config.clone());
                black_box(model.train(&triples, 5).unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark ML Pipeline hyperparam generation
fn bench_pipeline_hyperparam_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_hyperparam");

    for budget in [10, 20, 50, 100].iter() {
        let config = PipelineConfig {
            search_budget: *budget,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::from_parameter(budget), budget, |b, _| {
            b.iter(|| {
                let pipeline = EmbeddingPipeline::new(config.clone());
                black_box(pipeline)
            });
        });
    }

    group.finish();
}

/// Benchmark embedding retrieval
fn bench_embedding_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_retrieval");

    let triples = create_benchmark_triples(500);
    let config = EmbeddingConfig {
        embedding_dim: 128,
        learning_rate: 0.01,
        batch_size: 128,
        num_negative_samples: 5,
        ..Default::default()
    };

    // Pre-train models
    let mut transe = TransE::new(config.clone());
    transe.train(&triples, 10).unwrap();

    let mut distmult = DistMult::new(config.clone());
    distmult.train(&triples, 10).unwrap();

    let mut complex = ComplEx::new(config);
    complex.train(&triples, 10).unwrap();

    // Benchmark retrieval
    group.bench_function("transe_64d", |b| {
        b.iter(|| black_box(transe.get_embedding("entity_0").unwrap()));
    });

    group.bench_function("distmult_64d", |b| {
        b.iter(|| black_box(distmult.get_embedding("entity_0").unwrap()));
    });

    group.bench_function("complex_128d", |b| {
        b.iter(|| black_box(complex.get_embedding("entity_0").unwrap()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transe_training,
    bench_distmult_training,
    bench_complex_training,
    bench_model_comparison,
    bench_embedding_dimensions,
    bench_prediction,
    bench_similarity,
    bench_batch_sizes,
    bench_negative_samples,
    bench_pipeline_hyperparam_generation,
    bench_embedding_retrieval,
);

criterion_main!(benches);
