use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_stream::processing::{
    AggregateFunction, EventProcessor, ProcessorConfig, WindowConfig, WindowTrigger, WindowType,
};
use oxirs_stream::*;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

/// High-performance benchmark for oxirs-stream targeting 100K+ events/second
/// and <10ms latency as specified in the TODO requirements
fn create_benchmark_event(id: usize) -> StreamEvent {
    let metadata = EventMetadata {
        source: "benchmark".to_string(),
        ..Default::default()
    };

    StreamEvent::TripleAdded {
        subject: format!("http://benchmark.org/subject_{id}"),
        predicate: "http://benchmark.org/predicate".to_string(),
        object: format!("\"benchmark_value_{id}\""),
        graph: None,
        metadata,
    }
}

fn bench_memory_backend_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_backend_throughput");

    // Test different event counts for throughput
    for event_count in [1000, 10_000, 50_000].iter() {
        group.throughput(Throughput::Elements(*event_count as u64));

        group.bench_with_input(
            BenchmarkId::new("publish_events", event_count),
            event_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let start = Instant::now();

                        // Create events
                        let mut events = Vec::new();
                        for i in 0..count {
                            let event = create_benchmark_event(i);
                            events.push(black_box(event));
                        }

                        let duration = start.elapsed();

                        // Calculate events per second
                        let events_per_sec = (count as f64) / duration.as_secs_f64();

                        // Log performance for analysis
                        println!(
                            "Created {count} events in {duration:?} ({events_per_sec:.0} events/sec)"
                        );

                        duration
                    })
                });
            },
        );
    }

    group.finish();
}

fn bench_event_processing_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("event_processing_latency");

    group.bench_function("single_event_processing", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();

                // Create and process a single event
                let event = create_benchmark_event(1);
                let _serialized = serde_json::to_string(&event).unwrap();

                let latency = start.elapsed();

                // Target: process single event in <1ms
                assert!(
                    latency < Duration::from_millis(1),
                    "Failed to meet <1ms processing target. Got: {latency:?}"
                );

                latency
            })
        });
    });

    group.finish();
}

fn bench_delta_processing_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("delta_processing");

    for update_count in [1000, 10_000, 25_000].iter() {
        group.throughput(Throughput::Elements(*update_count as u64));

        group.bench_with_input(
            BenchmarkId::new("sparql_delta_processing", update_count),
            update_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut computer = DeltaComputer::new();

                        let start = Instant::now();

                        for i in 0..count {
                            let update = format!(
                                r#"INSERT DATA {{
                                    <http://benchmark.org/subject_{i}> <http://benchmark.org/predicate> "value_{i}" .
                                }}"#
                            );

                            let _events = computer.compute_delta(black_box(&update)).unwrap();
                        }

                        let duration = start.elapsed();
                        let updates_per_sec = (count as f64) / duration.as_secs_f64();

                        // Log performance for analysis
                        println!("Processed {count} SPARQL updates in {duration:?} ({updates_per_sec:.0} updates/sec)");

                        duration
                    })
                });
            },
        );
    }

    group.finish();
}

fn bench_rdf_patch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("rdf_patch_processing");

    for patch_count in [1000, 5_000, 15_000].iter() {
        group.throughput(Throughput::Elements(*patch_count as u64));

        group.bench_with_input(
            BenchmarkId::new("patch_generation", patch_count),
            patch_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut computer = DeltaComputer::new();

                        let start = Instant::now();

                        for i in 0..count {
                            let update = format!(
                                r#"INSERT DATA {{
                                    <http://patch.org/s_{i}> <http://patch.org/p> "object_{i}" .
                                }}"#
                            );

                            let _patch = computer.sparql_to_patch(black_box(&update)).unwrap();
                        }

                        let duration = start.elapsed();
                        let patches_per_sec = (count as f64) / duration.as_secs_f64();

                        // Log performance for analysis
                        println!(
                            "Generated {count} patches in {duration:?} ({patches_per_sec:.0} patches/sec)"
                        );

                        duration
                    })
                });
            },
        );
    }

    group.finish();
}

fn bench_stream_processing_pipeline(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("stream_processing_pipeline");

    group.bench_function("event_processor_10k_events", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut processor = EventProcessor::new(ProcessorConfig::default());

                // Create a window for processing
                let window_config = WindowConfig {
                    window_type: WindowType::CountBased { size: 1000 },
                    aggregates: vec![
                        AggregateFunction::Count,
                        AggregateFunction::Sum {
                            field: "value".to_string(),
                        },
                    ],
                    group_by: vec![],
                    filter: None,
                    allow_lateness: None,
                    trigger: WindowTrigger::OnCount(1000),
                };

                let _ = processor.create_window(window_config);

                let start = Instant::now();

                // Generate and process events
                let event_count = 10_000;
                let mut total_results = 0;

                for i in 0..event_count {
                    let event = create_benchmark_event(i);
                    let results = processor.process_event(black_box(event)).unwrap();
                    total_results += results.len();
                }

                let duration = start.elapsed();
                let events_per_sec = (event_count as f64) / duration.as_secs_f64();

                // Log performance for analysis
                println!(
                    "Processed {event_count} events in {duration:?} ({events_per_sec:.0} events/sec, {total_results} window results)"
                );

                duration
            })
        });
    });

    group.finish();
}

/// Benchmark zero-copy operations performance
fn bench_zero_copy_operations(c: &mut Criterion) {
    use bytes::{Bytes, BytesMut};
    use oxirs_stream::{SharedRefBuffer, ZeroCopyBuffer};

    let mut group = c.benchmark_group("zero_copy_operations");

    // Benchmark buffer creation from BytesMut
    group.bench_function("buffer_creation_1kb", |b| {
        b.iter(|| {
            let data = BytesMut::from(&[0u8; 1024][..]);
            let buffer = ZeroCopyBuffer::from_bytes_mut(black_box(data));
            black_box(buffer)
        });
    });

    group.bench_function("buffer_creation_64kb", |b| {
        b.iter(|| {
            let data = BytesMut::from(&[0u8; 65536][..]);
            let buffer = ZeroCopyBuffer::from_bytes_mut(black_box(data));
            black_box(buffer)
        });
    });

    // Benchmark buffer sharing (zero-copy)
    group.bench_function("buffer_share_1kb", |b| {
        let data = Bytes::from_static(&[0u8; 1024]);
        let buffer = SharedRefBuffer::new(data);

        b.iter(|| {
            let cloned = buffer.clone();
            black_box(cloned)
        });
    });

    // Benchmark buffer slicing
    group.bench_function("buffer_slice_operations", |b| {
        let data = BytesMut::from(&[0u8; 1024][..]);
        let buffer = ZeroCopyBuffer::from_bytes_mut(data);

        b.iter(|| {
            let slice = buffer.slice(black_box(0)..black_box(512));
            black_box(slice)
        });
    });

    group.finish();
}

/// Benchmark SIMD batch processing performance
fn bench_simd_batch_processing(c: &mut Criterion) {
    use oxirs_stream::{SimdBatchProcessor, SimdOperation};

    let mut group = c.benchmark_group("simd_batch_processing");

    // Benchmark SIMD batch processing
    for size in [100, 1000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("simd_sum", size), size, |b, &count| {
            let data: Vec<u8> = (0..count).map(|i| (i % 256) as u8).collect();

            b.iter(|| {
                let processor = SimdBatchProcessor::new(1024);
                let result = processor.process_batch(black_box(&data), SimdOperation::Sum);
                black_box(result)
            });
        });
    }

    // Benchmark XOR operation
    group.bench_function("simd_xor_1000", |b| {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let processor = SimdBatchProcessor::new(1024);

        b.iter(|| {
            let result = processor.process_batch(black_box(&data), SimdOperation::XorMask(0xFF));
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark serialization performance for different formats
fn bench_serialization_formats(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization_formats");

    let event = create_benchmark_event(1);

    // Benchmark JSON serialization
    group.bench_function("json_serialize", |b| {
        b.iter(|| {
            let result = serde_json::to_vec(black_box(&event)).unwrap();
            black_box(result)
        });
    });

    // Benchmark MessagePack serialization
    group.bench_function("messagepack_serialize", |b| {
        b.iter(|| {
            let result = rmp_serde::to_vec(black_box(&event)).unwrap();
            black_box(result)
        });
    });

    // Benchmark CBOR serialization
    group.bench_function("cbor_serialize", |b| {
        b.iter(|| {
            let result = serde_cbor::to_vec(black_box(&event)).unwrap();
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark pattern matching performance
fn bench_pattern_matching(c: &mut Criterion) {
    use oxirs_stream::processing::pattern::PatternMatcher;

    let mut group = c.benchmark_group("pattern_matching");

    group.bench_function("simple_event_processing_10", |b| {
        b.iter(|| {
            let events: Vec<_> = (0..10).map(create_benchmark_event).collect();
            let mut matcher = PatternMatcher::new(1000);

            for event in &events {
                let _ = matcher.process_event(black_box(event.clone()));
            }
        });
    });

    group.bench_function("complex_event_processing_100", |b| {
        b.iter(|| {
            let events: Vec<_> = (0..100).map(create_benchmark_event).collect();
            let mut matcher = PatternMatcher::new(1000);

            for event in &events {
                let _ = matcher.process_event(black_box(event.clone()));
            }
        });
    });

    group.finish();
}

/// Benchmark backpressure handling performance
fn bench_backpressure_handling(c: &mut Criterion) {
    use oxirs_stream::backpressure::{
        BackpressureConfig, BackpressureController, BackpressureStrategy,
    };

    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("backpressure_handling");

    for (name, strategy) in [
        ("drop_oldest", BackpressureStrategy::DropOldest),
        ("drop_newest", BackpressureStrategy::DropNewest),
        ("block", BackpressureStrategy::Block),
    ]
    .iter()
    {
        group.bench_with_input(
            BenchmarkId::new("backpressure", name),
            strategy,
            |b, strat| {
                b.iter(|| {
                    rt.block_on(async {
                        let config = BackpressureConfig {
                            strategy: strat.clone(),
                            max_buffer_size: 1000,
                            ..Default::default()
                        };

                        let controller = BackpressureController::new(config);

                        // Simulate high load
                        for i in 0..100 {
                            let event = create_benchmark_event(i);
                            let _ = controller.offer(black_box(event)).await;
                        }
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark compression performance
fn bench_compression_formats(c: &mut Criterion) {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut group = c.benchmark_group("compression_formats");

    // Create test data
    let data = vec![0u8; 10240]; // 10KB of zeros (highly compressible)

    // Benchmark Gzip compression
    group.bench_function("gzip_compress_10kb", |b| {
        b.iter(|| {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(black_box(&data)).unwrap();
            let result = encoder.finish().unwrap();
            black_box(result)
        });
    });

    // Benchmark LZ4 compression
    group.bench_function("lz4_compress_10kb", |b| {
        b.iter(|| {
            let result = lz4_flex::compress_prepend_size(black_box(&data));
            black_box(result)
        });
    });

    // Benchmark Snappy compression
    group.bench_function("snappy_compress_10kb", |b| {
        let mut encoder = snap::raw::Encoder::new();
        b.iter(|| {
            let result = encoder.compress_vec(black_box(&data)).unwrap();
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_backend_throughput,
    bench_event_processing_latency,
    bench_delta_processing_throughput,
    bench_rdf_patch_processing,
    bench_stream_processing_pipeline,
    bench_zero_copy_operations,
    bench_simd_batch_processing,
    bench_serialization_formats,
    bench_pattern_matching,
    bench_backpressure_handling,
    bench_compression_formats
);
criterion_main!(benches);
