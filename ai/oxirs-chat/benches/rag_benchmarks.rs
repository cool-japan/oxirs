//! Benchmarks for RAG System Performance
//!
//! Tests the performance of retrieval-augmented generation including
//! quantum-enhanced retrieval, consciousness-aware processing, and reasoning.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_chat::rag::{RagConfig, RagEngine, RetrievalConfig};
use oxirs_core::ConcreteStore;
use std::sync::Arc;
use std::time::Duration;

fn setup_rag_engine() -> RagEngine {
    let store = Arc::new(ConcreteStore::new().unwrap());
    let config = RagConfig {
        retrieval: RetrievalConfig {
            enable_quantum_enhancement: true,
            enable_consciousness_integration: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut engine = RagEngine::new(config, store);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        engine.initialize().await.unwrap();
    });

    engine
}

fn bench_basic_retrieval(c: &mut Criterion) {
    let mut engine = setup_rag_engine();

    c.bench_function("rag_basic_retrieval", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    engine
                        .retrieve("What are the properties of proteins?")
                        .await
                        .unwrap(),
                )
            })
        });
    });
}

fn bench_quantum_retrieval(c: &mut Criterion) {
    let mut engine = setup_rag_engine();

    c.bench_function("rag_quantum_retrieval", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    engine
                        .retrieve("Find genes related to cancer with high confidence")
                        .await
                        .unwrap(),
                )
            })
        });
    });
}

fn bench_retrieval_varying_query_length(c: &mut Criterion) {
    let mut engine = setup_rag_engine();
    let mut group = c.benchmark_group("rag_query_length");

    for query_words in [5, 10, 20, 50].iter() {
        let query = "protein ".repeat(*query_words);
        group.bench_with_input(BenchmarkId::from_parameter(query_words), &query, |b, q| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            b.iter(|| rt.block_on(async { black_box(engine.retrieve(q).await.unwrap()) }));
        });
    }
    group.finish();
}

fn bench_context_assembly(c: &mut Criterion) {
    let mut engine = setup_rag_engine();

    c.bench_function("rag_context_assembly", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    engine
                        .retrieve("Complex biomedical query about protein interactions")
                        .await
                        .unwrap(),
                )
            })
        });
    });
}

fn bench_entity_extraction(c: &mut Criterion) {
    let mut engine = setup_rag_engine();

    c.bench_function("rag_entity_extraction", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    engine
                        .retrieve("Find TP53 gene interactions with BRCA1 and MDM2")
                        .await
                        .unwrap(),
                )
            })
        });
    });
}

criterion_group! {
    name = rag_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));
    targets = bench_basic_retrieval,
        bench_quantum_retrieval,
        bench_retrieval_varying_query_length,
        bench_context_assembly,
        bench_entity_extraction
}

criterion_main!(rag_benches);
