//! Query profiler performance benchmarks
//!
//! Measures the overhead of query profiling to ensure it has minimal impact
//! on query execution performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_core::query::query_profiler::{ProfilerConfig, QueryProfiler};
use std::time::Duration;

fn profiler_overhead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiler_overhead");

    // Baseline: No profiling
    group.bench_function("no_profiling", |b| {
        b.iter(|| {
            // Simulate query execution
            let start = std::time::Instant::now();
            simulate_query_execution(100);
            black_box(start.elapsed());
        });
    });

    // With profiling enabled
    group.bench_function("with_profiling", |b| {
        let config = ProfilerConfig::default();
        let profiler = QueryProfiler::new(config);

        b.iter(|| {
            let mut session = profiler.start_session("SELECT * WHERE { ?s ?p ?o }");
            session.start_phase("parse");
            simulate_query_execution(10);
            session.end_phase("parse");

            session.start_phase("planning");
            simulate_query_execution(10);
            session.end_phase("planning");

            session.start_phase("execution");
            simulate_query_execution(80);
            session.record_triples_matched(100);
            session.record_results(10);
            session.end_phase("execution");

            black_box(session.finish());
        });
    });

    // With profiling and pattern tracking
    group.bench_function("with_pattern_tracking", |b| {
        let config = ProfilerConfig {
            profile_patterns: true,
            profile_indexes: true,
            profile_joins: true,
            ..Default::default()
        };
        let profiler = QueryProfiler::new(config);

        b.iter(|| {
            let mut session = profiler.start_session("SELECT * WHERE { ?s ?p ?o }");

            session.start_phase("execution");
            for _ in 0..10 {
                session.record_pattern("SPO".to_string());
                session.record_index_access("SPO_index".to_string());
                session.record_join();
            }
            simulate_query_execution(80);
            session.end_phase("execution");

            black_box(session.finish());
        });
    });

    group.finish();
}

fn profiler_history_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiler_history");

    // Test history management with varying sizes
    for history_size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(history_size),
            history_size,
            |b, &size| {
                let config = ProfilerConfig {
                    max_history: size,
                    ..Default::default()
                };
                let profiler = QueryProfiler::new(config);

                // Pre-fill history
                for i in 0..size {
                    let mut session = profiler.start_session(&format!("Query {}", i));
                    session.record_triples_matched(100);
                    let stats = session.finish();
                    profiler.record_query(format!("Query {}", i), stats, "SELECT".to_string());
                }

                b.iter(|| {
                    let mut session = profiler.start_session("New query");
                    session.record_triples_matched(100);
                    let stats = session.finish();
                    profiler.record_query("New query".to_string(), stats, "SELECT".to_string());
                });
            },
        );
    }

    group.finish();
}

fn profiler_statistics_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiler_statistics");

    // Test statistics computation with varying query counts
    for query_count in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(query_count),
            query_count,
            |b, &count| {
                let config = ProfilerConfig {
                    max_history: count,
                    ..Default::default()
                };
                let profiler = QueryProfiler::new(config);

                // Pre-fill with queries
                for i in 0..count {
                    let mut session = profiler.start_session(&format!("Query {}", i));
                    session.record_triples_matched((i * 100) as u64);
                    session.record_pattern("SPO".to_string());
                    session.record_index_access("SPO_index".to_string());
                    let stats = session.finish();
                    profiler.record_query(format!("Query {}", i), stats, "SELECT".to_string());
                }

                b.iter(|| {
                    black_box(profiler.get_statistics());
                });
            },
        );
    }

    group.finish();
}

fn profiler_concurrent_access_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiler_concurrent");

    group.bench_function("concurrent_sessions", |b| {
        let config = ProfilerConfig::default();
        let profiler = std::sync::Arc::new(QueryProfiler::new(config));

        b.iter(|| {
            let profiler_clone = profiler.clone();
            let handle = std::thread::spawn(move || {
                let mut session = profiler_clone.start_session("Thread query");
                session.record_triples_matched(100);
                session.finish()
            });

            let mut session = profiler.start_session("Main query");
            session.record_triples_matched(100);
            let main_stats = session.finish();

            let thread_stats = handle.join().unwrap();
            black_box((main_stats, thread_stats));
        });
    });

    group.finish();
}

fn profiler_memory_tracking_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiler_memory");

    // Without memory tracking
    group.bench_function("no_memory_tracking", |b| {
        let config = ProfilerConfig {
            track_memory: false,
            ..Default::default()
        };
        let profiler = QueryProfiler::new(config);

        b.iter(|| {
            let mut session = profiler.start_session("Query");
            session.record_triples_matched(100);
            black_box(session.finish());
        });
    });

    // With memory tracking
    group.bench_function("with_memory_tracking", |b| {
        let config = ProfilerConfig {
            track_memory: true,
            ..Default::default()
        };
        let profiler = QueryProfiler::new(config);

        b.iter(|| {
            let mut session = profiler.start_session("Query");
            session.record_triples_matched(100);
            black_box(session.finish());
        });
    });

    group.finish();
}

// Helper function to simulate query execution
fn simulate_query_execution(duration_us: u64) {
    let start = std::time::Instant::now();
    let mut counter = 0u64;
    while start.elapsed() < Duration::from_micros(duration_us) {
        // Busy wait to simulate work
        counter = counter.wrapping_add(1);
        black_box(counter);
    }
}

criterion_group!(
    benches,
    profiler_overhead_benchmark,
    profiler_history_benchmark,
    profiler_statistics_benchmark,
    profiler_concurrent_access_benchmark,
    profiler_memory_tracking_benchmark,
);
criterion_main!(benches);
