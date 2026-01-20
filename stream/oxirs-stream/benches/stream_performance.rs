//! Performance Benchmarks for OxiRS Stream Processing
//!
//! Comprehensive benchmarks demonstrating highest possible performance
//! across all major features implemented in alpha.3

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_stream::event::EventMetadata;
use oxirs_stream::processing::{
    JoinConfig, JoinType, JoinWindowStrategy, Pattern, PatternMatcher, PipelineBuilder,
    SimdBatchConfig, SimdBatchProcessor, StreamJoiner,
};
use oxirs_stream::{
    BackpressureConfig, BackpressureController, DeadLetterQueue, DlqConfig, StreamEvent,
};
use tokio::runtime::Runtime;

// Helper to create test events
fn create_test_event(subject: &str, value: &str) -> StreamEvent {
    StreamEvent::TripleAdded {
        subject: subject.to_string(),
        predicate: "hasValue".to_string(),
        object: value.to_string(),
        graph: None,
        metadata: EventMetadata::default(),
    }
}

// Benchmark: Stream Operators Pipeline
fn bench_stream_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_operators");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let events: Vec<StreamEvent> = (0..size)
                .map(|i| create_test_event(&format!("subject_{}", i), &i.to_string()))
                .collect();

            b.iter(|| {
                let mut pipeline = PipelineBuilder::new()
                    .filter(|e| matches!(e, StreamEvent::TripleAdded { .. }))
                    .map(Ok)
                    .build();

                let rt = Runtime::new().unwrap();
                rt.block_on(async {
                    for event in &events {
                        black_box(pipeline.process(event.clone()).await).unwrap();
                    }
                });
            });
        });
    }

    group.finish();
}

// Benchmark: SIMD Batch Processing
fn bench_simd_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_processing");

    for batch_size in [128, 512, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let config = SimdBatchConfig {
                    batch_size,
                    auto_vectorize: true,
                    prefetch_distance: 64,
                    enable_parallel: true,
                };

                let mut processor = SimdBatchProcessor::new(config);

                let events: Vec<StreamEvent> = (1..=batch_size)
                    .map(|i| create_test_event(&format!("subject_{}", i), &i.to_string()))
                    .collect();

                b.iter(|| {
                    black_box(
                        processor
                            .process_batch(&events, |e| {
                                matches!(e, StreamEvent::TripleAdded { .. })
                            })
                            .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

// Benchmark: SIMD Aggregations
fn bench_simd_aggregations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_aggregations");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = SimdBatchConfig::default();
            let mut processor = SimdBatchProcessor::new(config);

            let events: Vec<StreamEvent> = (1..=size)
                .map(|i| create_test_event(&format!("subject_{}", i), &i.to_string()))
                .collect();

            b.iter(|| {
                black_box(processor.aggregate_batch(&events, "object").unwrap());
            });
        });
    }

    group.finish();
}

// Benchmark: Pattern Matching
fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");

    for size in [100, 500, 1_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut matcher = PatternMatcher::new(10000);

            // Register simple pattern
            let pattern = Pattern::Simple {
                name: "test_pattern".to_string(),
                predicate: "type:triple_added".to_string(),
            };
            matcher.register_pattern(pattern);

            let events: Vec<StreamEvent> = (0..size)
                .map(|i| create_test_event(&format!("subject_{}", i), &i.to_string()))
                .collect();

            b.iter(|| {
                for event in &events {
                    black_box(matcher.process_event(event.clone()).unwrap());
                }
            });
        });
    }

    group.finish();
}

// Benchmark: Stream Joins
fn bench_stream_joins(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_joins");

    for pairs in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*pairs as u64 * 2));

        group.bench_with_input(BenchmarkId::from_parameter(pairs), pairs, |b, &pairs| {
            let config = JoinConfig {
                join_type: JoinType::Inner,
                window_strategy: JoinWindowStrategy::Tumbling {
                    duration: chrono::Duration::seconds(60),
                },
                ..Default::default()
            };

            b.iter(|| {
                let rt = Runtime::new().unwrap();
                rt.block_on(async {
                    let joiner = StreamJoiner::new(config.clone());

                    for i in 0..pairs {
                        let left = create_test_event(&format!("subject_{}", i), &i.to_string());
                        let right = create_test_event(&format!("subject_{}", i), &i.to_string());

                        black_box(joiner.process_left(left).await.unwrap());
                        black_box(joiner.process_right(right).await.unwrap());
                    }
                });
            });
        });
    }

    group.finish();
}

// Benchmark: Backpressure Controller
fn bench_backpressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("backpressure");

    for size in [100, 500, 1_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = BackpressureConfig::default();

            b.iter(|| {
                let rt = Runtime::new().unwrap();
                rt.block_on(async {
                    let controller = BackpressureController::new(config.clone());

                    for i in 0..size {
                        let event = create_test_event(&format!("subject_{}", i), &i.to_string());
                        controller.offer(event).await.unwrap();
                        black_box(());
                    }
                });
            });
        });
    }

    group.finish();
}

// Benchmark: Dead Letter Queue Processing
fn bench_dlq_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("dlq_processing");

    for size in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let rt = Runtime::new().unwrap();
                rt.block_on(async {
                    let config = DlqConfig::default();
                    let dlq = DeadLetterQueue::new(config);

                    // Add events to retry queue
                    for i in 0..size {
                        let event = create_test_event(&format!("subject_{}", i), &i.to_string());
                        dlq.handle_failed_event(
                            event,
                            oxirs_stream::FailureReason::NetworkError,
                            "Test failure".to_string(),
                        )
                        .await
                        .unwrap();
                    }

                    // Wait for retry delay
                    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

                    // Process retries
                    let retry_fn = |_: StreamEvent| async { Ok(()) };
                    black_box(dlq.process_retries(retry_fn).await.unwrap());
                });
            });
        });
    }

    group.finish();
}

// Benchmark: End-to-End Pipeline Throughput
fn bench_e2e_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_throughput");
    group.sample_size(10); // Fewer samples for expensive benchmark

    for size in [1_000, 5_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let events: Vec<StreamEvent> = (0..size)
                .map(|i| create_test_event(&format!("subject_{}", i), &i.to_string()))
                .collect();

            b.iter(|| {
                let rt = Runtime::new().unwrap();
                rt.block_on(async {
                    // Full pipeline: operators + pattern matching + backpressure
                    let mut pipeline = PipelineBuilder::new()
                        .filter(|e| matches!(e, StreamEvent::TripleAdded { .. }))
                        .map(Ok)
                        .build();

                    let mut matcher = PatternMatcher::new(10000);
                    let pattern = Pattern::Simple {
                        name: "test".to_string(),
                        predicate: "type:triple_added".to_string(),
                    };
                    matcher.register_pattern(pattern);

                    let config = BackpressureConfig::default();
                    let controller = BackpressureController::new(config);

                    for event in &events {
                        // Pipeline processing
                        let processed = pipeline.process(event.clone()).await.unwrap();

                        // Pattern matching
                        if let Some(evt) = processed.first() {
                            matcher.process_event(evt.clone()).unwrap();
                        }

                        // Backpressure handling
                        controller.offer(event.clone()).await.unwrap();
                    }

                    black_box(matcher.stats());
                });
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_stream_operators,
    bench_simd_processing,
    bench_simd_aggregations,
    bench_pattern_matching,
    bench_stream_joins,
    bench_backpressure,
    bench_dlq_processing,
    bench_e2e_throughput
);

criterion_main!(benches);
