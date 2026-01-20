//! Benchmark result tracking and comparison system
//!
//! This module provides infrastructure to track benchmark results over time,
//! compare performance across versions, and detect performance regressions.
//!
//! # Usage
//!
//! Run benchmarks with tracking:
//! ```bash
//! cargo bench --bench benchmark_tracking
//! ```
//!
//! Compare results:
//! ```bash
//! cargo run --bin compare_benchmarks -- v0.1.0 v0.1.1
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_geosparql::functions::geometric_operations::*;
use oxirs_geosparql::functions::simple_features::*;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::SystemTime;

/// Benchmark result storage format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    pub timestamp: u64,
    pub version: String,
    pub rust_version: String,
    pub platform: String,
    pub benchmark_name: String,
    pub mean_time_ns: f64,
    pub std_dev_ns: f64,
    pub throughput_ops_per_sec: Option<f64>,
    pub sample_count: usize,
}

impl BenchmarkRecord {
    pub fn new(name: &str, mean_time_ns: f64, std_dev_ns: f64) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            platform: std::env::consts::OS.to_string(),
            benchmark_name: name.to_string(),
            mean_time_ns,
            std_dev_ns,
            throughput_ops_per_sec: None,
            sample_count: 100,
        }
    }

    pub fn with_throughput(mut self, ops_per_sec: f64) -> Self {
        self.throughput_ops_per_sec = Some(ops_per_sec);
        self
    }
}

/// Save benchmark results to JSON file
pub fn save_benchmark_results(records: &[BenchmarkRecord]) -> std::io::Result<()> {
    let results_dir = Path::new("target/benchmark_history");
    fs::create_dir_all(results_dir)?;

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let version = env!("CARGO_PKG_VERSION");
    let filename = format!("benchmarks_{}_{}json", version, timestamp);
    let filepath = results_dir.join(filename);

    let json = serde_json::to_string_pretty(&records)?;
    fs::write(filepath, json)?;

    // Also save to latest.json for easy comparison
    let latest_path = results_dir.join("latest.json");
    fs::write(latest_path, serde_json::to_string_pretty(&records)?)?;

    Ok(())
}

/// Load benchmark results from a specific version
pub fn load_benchmark_results(version: &str) -> std::io::Result<Vec<BenchmarkRecord>> {
    let results_dir = Path::new("target/benchmark_history");

    // Find the most recent file for this version
    let mut matching_files: Vec<_> = fs::read_dir(results_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("benchmarks_{}_", version))
        })
        .collect();

    matching_files.sort_by_key(|e| e.metadata().unwrap().modified().unwrap());

    if let Some(latest) = matching_files.last() {
        let content = fs::read_to_string(latest.path())?;
        let records: Vec<BenchmarkRecord> = serde_json::from_str(&content)?;
        Ok(records)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("No benchmark results found for version {}", version),
        ))
    }
}

// ============================================================================
// Core Benchmark Suite
// ============================================================================

fn bench_wkt_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("wkt_parsing");

    let test_cases = vec![
        ("point", "POINT(1 2)"),
        ("linestring", "LINESTRING(0 0, 1 1, 2 2, 3 3, 4 4)"),
        ("polygon", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
        (
            "complex_polygon",
            "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0), (20 20, 40 20, 40 40, 20 40, 20 20))",
        ),
    ];

    for (name, wkt) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), wkt, |b, wkt| {
            b.iter(|| {
                let _geom = Geometry::from_wkt(black_box(wkt)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_distance_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculations");
    group.throughput(Throughput::Elements(1));

    let p1 = Geometry::from_wkt("POINT(0 0)").unwrap();
    let p2 = Geometry::from_wkt("POINT(3 4)").unwrap();

    group.bench_function("point_to_point", |b| {
        b.iter(|| {
            let _dist = distance(black_box(&p1), black_box(&p2)).unwrap();
        });
    });

    let line = Geometry::from_wkt("LINESTRING(0 0, 10 0, 10 10, 0 10)").unwrap();

    group.bench_function("point_to_linestring", |b| {
        b.iter(|| {
            let _dist = distance(black_box(&p1), black_box(&line)).unwrap();
        });
    });

    group.finish();
}

fn bench_topological_relations(c: &mut Criterion) {
    let mut group = c.benchmark_group("topological_relations");

    let poly1 = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))").unwrap();
    let poly2 = Geometry::from_wkt("POLYGON((3 3, 8 3, 8 8, 3 8, 3 3))").unwrap();
    let point_inside = Geometry::from_wkt("POINT(2 2)").unwrap();

    group.bench_function("sf_intersects", |b| {
        b.iter(|| {
            let _result = sf_intersects(black_box(&poly1), black_box(&poly2)).unwrap();
        });
    });

    group.bench_function("sf_within", |b| {
        b.iter(|| {
            let _result = sf_within(black_box(&point_inside), black_box(&poly1)).unwrap();
        });
    });

    group.bench_function("sf_overlaps", |b| {
        b.iter(|| {
            let _result = sf_overlaps(black_box(&poly1), black_box(&poly2)).unwrap();
        });
    });

    group.finish();
}

fn bench_spatial_index_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_index");
    group.sample_size(50);

    // Create test geometries
    let geometries: Vec<Geometry> = (0..1000)
        .map(|i| {
            let x = (i % 100) as f64 * 0.1;
            let y = (i / 100) as f64 * 0.1;
            Geometry::from_wkt(&format!("POINT({} {})", x, y)).unwrap()
        })
        .collect();

    group.bench_function("insert_1000_points", |b| {
        b.iter(|| {
            let index = SpatialIndex::new();
            for geom in &geometries {
                let _ = index.insert(geom.clone());
            }
        });
    });

    // Bulk load benchmark
    group.bench_function("bulk_load_1000_points", |b| {
        b.iter(|| {
            let _index = SpatialIndex::bulk_load(black_box(geometries.clone())).unwrap();
        });
    });

    // Query benchmark
    let index = SpatialIndex::bulk_load(geometries.clone()).unwrap();

    group.bench_function("bbox_query_100_results", |b| {
        b.iter(|| {
            let _results = index.query_bbox(
                black_box(0.0),
                black_box(0.0),
                black_box(3.0),
                black_box(3.0),
            );
        });
    });

    group.bench_function("nearest_neighbor", |b| {
        b.iter(|| {
            let _result = index.nearest(black_box(5.0), black_box(5.0));
        });
    });

    group.finish();
}

#[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
fn bench_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_operations");
    group.sample_size(20);

    let point = Geometry::from_wkt("POINT(0 0)").unwrap();
    let line = Geometry::from_wkt("LINESTRING(0 0, 10 0, 10 10)").unwrap();
    let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();

    group.bench_function("buffer_point_1.0", |b| {
        b.iter(|| {
            let _buffered = buffer(black_box(&point), black_box(1.0)).unwrap();
        });
    });

    group.bench_function("buffer_linestring_1.0", |b| {
        b.iter(|| {
            let _buffered = buffer(black_box(&line), black_box(1.0)).unwrap();
        });
    });

    group.bench_function("buffer_polygon_1.0", |b| {
        b.iter(|| {
            let _buffered = buffer(black_box(&polygon), black_box(1.0)).unwrap();
        });
    });

    group.finish();
}

fn bench_set_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_operations");
    group.sample_size(50);

    let poly1 = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
    let poly2 = Geometry::from_wkt("POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))").unwrap();

    group.bench_function("union", |b| {
        b.iter(|| {
            let _result = union(black_box(&poly1), black_box(&poly2)).unwrap();
        });
    });

    group.bench_function("intersection", |b| {
        b.iter(|| {
            let _result = intersection(black_box(&poly1), black_box(&poly2)).unwrap();
        });
    });

    group.bench_function("difference", |b| {
        b.iter(|| {
            let _result = difference(black_box(&poly1), black_box(&poly2)).unwrap();
        });
    });

    group.bench_function("sym_difference", |b| {
        b.iter(|| {
            let _result = sym_difference(black_box(&poly1), black_box(&poly2)).unwrap();
        });
    });

    group.finish();
}

// Benchmark groups
criterion_group!(
    core_benches,
    bench_wkt_parsing,
    bench_distance_calculations,
    bench_topological_relations,
);

criterion_group!(index_benches, bench_spatial_index_operations,);

criterion_group!(geometry_benches, bench_set_operations,);

#[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
criterion_group!(buffer_benches, bench_buffer_operations,);

// Main benchmark runner
#[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
criterion_main!(
    core_benches,
    index_benches,
    geometry_benches,
    buffer_benches
);

#[cfg(not(any(feature = "geos-backend", feature = "rust-buffer")))]
criterion_main!(core_benches, index_benches, geometry_benches);
