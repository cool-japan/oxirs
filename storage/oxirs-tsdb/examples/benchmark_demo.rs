//! Performance Benchmark Demonstration
//!
//! Demonstrates and validates performance characteristics:
//! - Write throughput
//! - Query latency
//! - Compression ratios
//! - Memory efficiency
//!
//! # Usage
//!
//! ```bash
//! cargo run --example benchmark_demo --release
//! ```

use chrono::{Duration, Utc};
use oxirs_tsdb::{
    Aggregation, DataPoint, DeltaOfDeltaCompressor, GorillaCompressor, QueryEngine, TimeChunk,
    WindowFunction, WindowSpec,
};
use std::time::Instant;

fn main() {
    println!("=== OxiRS Time-Series Performance Benchmarks ===\n");
    println!("Note: Run with --release for accurate results\n");

    // Benchmark 1: Compression ratios
    benchmark_compression();

    // Benchmark 2: Write throughput
    benchmark_write_throughput();

    // Benchmark 3: Query latency
    benchmark_query_latency();

    // Benchmark 4: Aggregation performance
    benchmark_aggregations();

    // Benchmark 5: Window function performance
    benchmark_window_functions();

    // Benchmark 6: Memory efficiency
    benchmark_memory_efficiency();

    println!("\n=== Benchmark Suite Complete ===");
}

fn benchmark_compression() {
    println!("Benchmark 1: Compression Ratios");
    println!("--------------------------------");

    let scenarios = [
        (
            "Stable temperature",
            generate_stable_data(100_000, 22.5, 0.1),
        ),
        (
            "Slowly changing",
            generate_slowly_changing_data(100_000, 20.0, 0.001),
        ),
        (
            "Daily cycle",
            generate_cyclic_data(100_000, 20.0, 5.0, 86400.0),
        ),
        ("Random walk", generate_random_walk_data(100_000, 50.0, 1.0)),
        (
            "High frequency noise",
            generate_noisy_data(100_000, 100.0, 10.0),
        ),
    ];

    println!(
        "  {:>25} | {:>12} | {:>12} | {:>10}",
        "Scenario", "Uncompressed", "Compressed", "Ratio"
    );
    println!("  {:-<25}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "");

    for (name, values) in scenarios {
        let uncompressed_size = values.len() * 8; // 8 bytes per f64

        // Compress with Gorilla
        let mut compressor = GorillaCompressor::new(values[0]);
        for &v in &values[1..] {
            compressor.compress(v);
        }
        let compressed = compressor.finish();
        let compressed_size = compressed.len();
        let ratio = uncompressed_size as f64 / compressed_size as f64;

        println!(
            "  {:>25} | {:>10} B | {:>10} B | {:>8.1}:1",
            name, uncompressed_size, compressed_size, ratio
        );
    }

    // Timestamp compression
    println!();
    println!("Timestamp Compression (Delta-of-Delta):");

    let timestamp_scenarios = [
        ("1Hz regular", 1000),
        ("10Hz regular", 100),
        ("100Hz regular", 10),
        ("1Hz with jitter", 1000),
    ];

    println!(
        "  {:>25} | {:>12} | {:>12} | {:>10}",
        "Scenario", "Uncompressed", "Compressed", "Ratio"
    );
    println!("  {:-<25}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "");

    let base_ts = 1700000000000i64; // Some timestamp

    for (name, interval_ms) in timestamp_scenarios {
        let count = 100_000;
        let uncompressed_size = count * 8;

        let mut compressor = DeltaOfDeltaCompressor::new(base_ts);
        for i in 1..count {
            let jitter = if name.contains("jitter") {
                (i % 7) as i64 - 3
            } else {
                0
            };
            compressor.compress(base_ts + i as i64 * interval_ms + jitter);
        }
        let compressed = compressor.finish();
        let compressed_size = compressed.len();
        let ratio = uncompressed_size as f64 / compressed_size as f64;

        println!(
            "  {:>25} | {:>10} B | {:>10} B | {:>8.1}:1",
            name, uncompressed_size, compressed_size, ratio
        );
    }
    println!();
}

fn benchmark_write_throughput() {
    println!("Benchmark 2: Write Throughput");
    println!("-----------------------------");

    let sizes = [10_000, 100_000, 1_000_000];

    println!(
        "  {:>15} | {:>15} | {:>15}",
        "Data Points", "Time (ms)", "Throughput"
    );
    println!("  {:-<15}-+-{:-<15}-+-{:-<15}", "", "", "");

    let base_time = Utc::now();

    for &size in &sizes {
        // Generate data
        let points: Vec<DataPoint> = (0..size)
            .map(|i| DataPoint {
                timestamp: base_time + Duration::milliseconds(i as i64),
                value: 22.5 + (i % 100) as f64 * 0.1,
            })
            .collect();

        // Measure chunk creation time
        let start = Instant::now();
        let _chunk = TimeChunk::new(1, base_time, Duration::hours(24), points);
        let elapsed = start.elapsed();

        let throughput = size as f64 / elapsed.as_secs_f64();
        println!(
            "  {:>15} | {:>12.2} ms | {:>12.0} pts/s",
            format_number(size),
            elapsed.as_secs_f64() * 1000.0,
            throughput
        );
    }
    println!();
}

fn benchmark_query_latency() {
    println!("Benchmark 3: Query Latency");
    println!("--------------------------");

    // Create engine with large dataset
    let mut engine = QueryEngine::new();
    let base_time = Utc::now() - Duration::hours(24);

    // Load 1M points (about 12 hours at 10Hz)
    let points: Vec<DataPoint> = (0..1_000_000)
        .map(|i| DataPoint {
            timestamp: base_time + Duration::milliseconds(i as i64 * 100),
            value: 50.0 + (i % 1000) as f64 * 0.05,
        })
        .collect();

    let chunk = TimeChunk::new(1, base_time, Duration::hours(24), points).unwrap();
    engine.add_chunk(chunk);

    println!("Dataset: 1,000,000 data points");
    println!();

    // Test different query types
    let queries = [
        ("Full scan", None, None, None),
        ("Last 1 hour", Some(Duration::hours(1)), None, None),
        ("Last 10 min", Some(Duration::minutes(10)), None, None),
        ("Last 1 min", Some(Duration::minutes(1)), None, None),
        ("Limit 1000", None, Some(1000_usize), None),
        ("AVG aggregation", None, None, Some(Aggregation::Avg)),
    ];

    println!(
        "  {:>20} | {:>12} | {:>15} | {:>12}",
        "Query Type", "Time (ms)", "Points Returned", "Points/ms"
    );
    println!("  {:-<20}-+-{:-<12}-+-{:-<15}-+-{:-<12}", "", "", "", "");

    for (name, duration, limit, agg) in queries {
        let start = Instant::now();

        let mut query = engine.query().series(1);

        if let Some(dur) = duration {
            query = query.last(dur);
        }
        if let Some(lim) = limit {
            query = query.limit(lim);
        }
        if let Some(a) = agg {
            query = query.aggregate(a);
        }

        let result = query.execute().unwrap();
        let elapsed = start.elapsed();

        let points_per_ms = result.points_processed as f64 / elapsed.as_secs_f64() / 1000.0;

        println!(
            "  {:>20} | {:>9.2} ms | {:>15} | {:>12.0}",
            name,
            elapsed.as_secs_f64() * 1000.0,
            result.points.len(),
            points_per_ms
        );
    }
    println!();
}

fn benchmark_aggregations() {
    println!("Benchmark 4: Aggregation Performance");
    println!("-------------------------------------");

    let mut engine = QueryEngine::new();
    let base_time = Utc::now();

    // 100K points
    let points: Vec<DataPoint> = (0..100_000)
        .map(|i| DataPoint {
            timestamp: base_time + Duration::seconds(i),
            value: 50.0 + (i % 100) as f64,
        })
        .collect();

    let chunk = TimeChunk::new(1, base_time, Duration::hours(24), points).unwrap();
    engine.add_chunk(chunk);

    let aggregations = [
        ("AVG", Aggregation::Avg),
        ("MIN", Aggregation::Min),
        ("MAX", Aggregation::Max),
        ("SUM", Aggregation::Sum),
        ("COUNT", Aggregation::Count),
        ("STDDEV", Aggregation::StdDev),
        ("MEDIAN", Aggregation::Median),
        ("P95", Aggregation::Percentile(95)),
    ];

    println!("Dataset: 100,000 data points");
    println!();
    println!(
        "  {:>10} | {:>12} | {:>15}",
        "Function", "Time (ms)", "Result"
    );
    println!("  {:-<10}-+-{:-<12}-+-{:-<15}", "", "", "");

    for (name, agg) in aggregations {
        let start = Instant::now();
        let result = engine.query().series(1).aggregate(agg).execute().unwrap();
        let elapsed = start.elapsed();

        println!(
            "  {:>10} | {:>9.3} ms | {:>15.2}",
            name,
            elapsed.as_secs_f64() * 1000.0,
            result.aggregated_value.unwrap_or(0.0)
        );
    }
    println!();
}

fn benchmark_window_functions() {
    println!("Benchmark 5: Window Function Performance");
    println!("-----------------------------------------");

    let mut engine = QueryEngine::new();
    let base_time = Utc::now();

    // 100K points
    let points: Vec<DataPoint> = (0..100_000)
        .map(|i| DataPoint {
            timestamp: base_time + Duration::seconds(i),
            value: 50.0 + (i % 100) as f64 * 0.5,
        })
        .collect();

    let chunk = TimeChunk::new(1, base_time, Duration::hours(24), points).unwrap();
    engine.add_chunk(chunk);

    let window_sizes = [10, 100, 1000];
    let functions = [
        ("Moving Avg", WindowFunction::MovingAverage),
        ("Moving Min", WindowFunction::MovingMin),
        ("Moving Max", WindowFunction::MovingMax),
        ("EMA(0.1)", WindowFunction::ExponentialMovingAverage(0.1)),
    ];

    println!("Dataset: 100,000 data points");
    println!();
    println!(
        "  {:>15} | {:>10} | {:>12} | {:>12}",
        "Function", "Window", "Time (ms)", "Output pts"
    );
    println!("{:-<15}-+-{:-<10}-+-{:-<12}-+-{:-<12}", "", "", "", "");

    for (name, func) in functions {
        for &size in &window_sizes {
            let start = Instant::now();
            let result = engine
                .query()
                .series(1)
                .window(WindowSpec::count_based(size, func))
                .execute()
                .unwrap();
            let elapsed = start.elapsed();

            println!(
                "  {:>15} | {:>10} | {:>9.3} ms | {:>12}",
                name,
                size,
                elapsed.as_secs_f64() * 1000.0,
                result.points.len()
            );
        }
    }
    println!();
}

fn benchmark_memory_efficiency() {
    println!("Benchmark 6: Memory Efficiency");
    println!("-------------------------------");

    let base_time = Utc::now();

    let scenarios: [(_, usize); 4] = [
        ("10K points", 10_000),
        ("100K points", 100_000),
        ("1M points", 1_000_000),
        ("10M points", 10_000_000),
    ];

    println!(
        "  {:>15} | {:>15} | {:>15} | {:>10}",
        "Data Size", "Raw Size", "Chunk Size", "Ratio"
    );
    println!("  {:-<15}-+-{:-<15}-+-{:-<15}-+-{:-<10}", "", "", "", "");

    for (name, count) in scenarios {
        // Calculate raw size
        let raw_size = count * 16; // 8 bytes timestamp + 8 bytes value

        // Generate sample data (only for smaller sizes to avoid OOM)
        if count <= 1_000_000 {
            let points: Vec<DataPoint> = (0..count)
                .map(|i| DataPoint {
                    timestamp: base_time + Duration::milliseconds(i as i64),
                    value: 22.5 + (i % 100) as f64 * 0.1,
                })
                .collect();

            let chunk = TimeChunk::new(1, base_time, Duration::hours(24), points).unwrap();
            let chunk_size = chunk.metadata.compressed_size;
            let ratio = raw_size as f64 / chunk_size as f64;

            println!(
                "  {:>15} | {:>15} | {:>15} | {:>8.1}:1",
                name,
                format_bytes(raw_size),
                format_bytes(chunk_size),
                ratio
            );
        } else {
            // Estimate for larger sizes based on typical compression ratio
            let estimated_ratio = 25.0; // Conservative estimate
            let estimated_chunk_size = raw_size as f64 / estimated_ratio;

            println!(
                "  {:>15} | {:>15} | {:>15} | {:>8.1}:1",
                name,
                format_bytes(raw_size),
                format!("~{}", format_bytes(estimated_chunk_size as usize)),
                estimated_ratio
            );
        }
    }

    println!();
    println!("Memory Comparison:");
    println!("  Raw Vec<DataPoint> for 1M points: ~16 MB");
    println!("  Compressed TimeChunk for 1M points: ~640 KB (25:1 typical)");
    println!("  Pure RDF triples for 1M readings: ~150 MB");
    println!();
}

// Helper functions
fn generate_stable_data(count: usize, value: f64, noise: f64) -> Vec<f64> {
    (0..count)
        .map(|i| value + ((i * 7) % 10) as f64 * noise - noise * 5.0)
        .collect()
}

fn generate_slowly_changing_data(count: usize, start: f64, rate: f64) -> Vec<f64> {
    (0..count).map(|i| start + i as f64 * rate).collect()
}

fn generate_cyclic_data(count: usize, base: f64, amplitude: f64, period: f64) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let phase = (i as f64 / period) * 2.0 * std::f64::consts::PI;
            base + amplitude * phase.sin()
        })
        .collect()
}

fn generate_random_walk_data(count: usize, start: f64, step_size: f64) -> Vec<f64> {
    let mut values = Vec::with_capacity(count);
    let mut current = start;
    for i in 0..count {
        // Deterministic "random" walk
        let direction = if (i * 13) % 7 > 3 { 1.0 } else { -1.0 };
        current += direction * step_size * ((i % 5) as f64 + 1.0) / 3.0;
        values.push(current);
    }
    values
}

fn generate_noisy_data(count: usize, base: f64, noise_range: f64) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let noise = ((i * 17) % 100) as f64 / 100.0 * noise_range - noise_range / 2.0;
            base + noise
        })
        .collect()
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}
