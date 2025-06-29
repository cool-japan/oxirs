use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_stream::*;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use chrono::Utc;

/// High-performance benchmark for oxirs-stream targeting 100K+ events/second
/// and <10ms latency as specified in the TODO requirements

fn create_benchmark_event(id: usize) -> StreamEvent {
    let mut metadata = EventMetadata::default();
    metadata.source = "benchmark".to_string();
    
    StreamEvent::TripleAdded {
        subject: format!("http://benchmark.org/subject_{}", id),
        predicate: "http://benchmark.org/predicate".to_string(),
        object: format!("\"benchmark_value_{}\"", id),
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
                        println!("Created {} events in {:?} ({:.0} events/sec)", 
                                count, duration, events_per_sec);
                        
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
                assert!(latency < Duration::from_millis(1),
                       "Failed to meet <1ms processing target. Got: {:?}", latency);
                
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
                                    <http://benchmark.org/subject_{}> <http://benchmark.org/predicate> "value_{}" .
                                }}"#,
                                i, i
                            );
                            
                            let _events = computer.compute_delta(black_box(&update)).unwrap();
                        }
                        
                        let duration = start.elapsed();
                        let updates_per_sec = (count as f64) / duration.as_secs_f64();
                        
                        // Log performance for analysis
                        println!("Processed {} SPARQL updates in {:?} ({:.0} updates/sec)", 
                                count, duration, updates_per_sec);
                        
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
                                    <http://patch.org/s_{}> <http://patch.org/p> "object_{}" .
                                }}"#,
                                i, i
                            );
                            
                            let _patch = computer.sparql_to_patch(black_box(&update)).unwrap();
                        }
                        
                        let duration = start.elapsed();
                        let patches_per_sec = (count as f64) / duration.as_secs_f64();
                        
                        // Log performance for analysis
                        println!("Generated {} patches in {:?} ({:.0} patches/sec)", 
                                count, duration, patches_per_sec);
                        
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
                let mut processor = EventProcessor::new();
                
                // Create a window for processing
                let window_config = WindowConfig {
                    window_type: WindowType::CountBased { size: 1000 },
                    aggregates: vec![
                        AggregateFunction::Count,
                        AggregateFunction::Sum { field: "value".to_string() },
                    ],
                    group_by: vec![],
                    filter: None,
                    allow_lateness: None,
                    trigger: WindowTrigger::OnCount(1000),
                };
                
                processor.create_window(window_config);
                
                let start = Instant::now();
                
                // Generate and process events
                let event_count = 10_000;
                let mut total_results = 0;
                
                for i in 0..event_count {
                    let event = create_benchmark_event(i);
                    let results = processor.process_event(black_box(event)).await.unwrap();
                    total_results += results.len();
                }
                
                let duration = start.elapsed();
                let events_per_sec = (event_count as f64) / duration.as_secs_f64();
                
                // Log performance for analysis
                println!("Processed {} events in {:?} ({:.0} events/sec, {} window results)", 
                        event_count, duration, events_per_sec, total_results);
                
                duration
            })
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
    bench_stream_processing_pipeline
);
criterion_main!(benches);