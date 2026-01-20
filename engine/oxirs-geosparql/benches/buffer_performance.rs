//! Benchmark comparing Pure Rust buffer vs GEOS backend
//!
//! Run with: cargo bench --bench buffer_performance --features rust-buffer,geos-backend

use criterion::{criterion_group, criterion_main, Criterion};
use oxirs_geosparql::geometry::Geometry;

#[cfg(any(feature = "rust-buffer", feature = "geos-backend"))]
use criterion::{black_box, BenchmarkId};

#[cfg(feature = "rust-buffer")]
use oxirs_geosparql::functions::geometric_operations::buffer_rust;

#[cfg(feature = "geos-backend")]
use oxirs_geosparql::functions::geometric_operations::{buffer_with_params, BufferParams};

/// Create test polygons of various sizes
#[allow(dead_code)]
fn create_test_polygons() -> Vec<(String, Geometry)> {
    vec![
        (
            "Small Square (10x10)".to_string(),
            Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap(),
        ),
        (
            "Medium Square (100x100)".to_string(),
            Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))").unwrap(),
        ),
        (
            "Large Square (1000x1000)".to_string(),
            Geometry::from_wkt("POLYGON((0 0, 1000 0, 1000 1000, 0 1000, 0 0))").unwrap(),
        ),
        (
            "Polygon with Hole".to_string(),
            Geometry::from_wkt(
                "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0), (20 20, 80 20, 80 80, 20 80, 20 20))",
            )
            .unwrap(),
        ),
        (
            "Complex L-Shape".to_string(),
            Geometry::from_wkt("POLYGON((0 0, 50 0, 50 30, 30 30, 30 50, 0 50, 0 0))").unwrap(),
        ),
        (
            "MultiPolygon (3 squares)".to_string(),
            Geometry::from_wkt(
                "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), \
                 ((20 20, 30 20, 30 30, 20 30, 20 20)), \
                 ((40 40, 50 40, 50 50, 40 50, 40 40)))",
            )
            .unwrap(),
        ),
    ]
}

#[cfg(feature = "rust-buffer")]
fn bench_pure_rust_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pure Rust Buffer");

    for (name, geom) in create_test_polygons() {
        group.bench_with_input(
            BenchmarkId::new("Positive Buffer (2.0)", &name),
            &geom,
            |b, geom| {
                b.iter(|| {
                    buffer_rust(black_box(geom), black_box(2.0)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Negative Buffer (-2.0)", &name),
            &geom,
            |b, geom| {
                b.iter(|| {
                    buffer_rust(black_box(geom), black_box(-2.0)).unwrap();
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "geos-backend")]
fn bench_geos_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("GEOS Buffer");
    let params = BufferParams::default();

    for (name, geom) in create_test_polygons() {
        group.bench_with_input(
            BenchmarkId::new("Positive Buffer (2.0)", &name),
            &geom,
            |b, geom| {
                b.iter(|| {
                    buffer_with_params(black_box(geom), black_box(2.0), &params).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Negative Buffer (-2.0)", &name),
            &geom,
            |b, geom| {
                b.iter(|| {
                    buffer_with_params(black_box(geom), black_box(-2.0), &params).unwrap();
                });
            },
        );
    }

    group.finish();
}

#[cfg(all(feature = "rust-buffer", feature = "geos-backend"))]
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Backend Comparison");

    let medium_square = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))").unwrap();
    let params = BufferParams::default();

    group.bench_function("Pure Rust - Medium Square", |b| {
        b.iter(|| {
            buffer_rust(black_box(&medium_square), black_box(5.0)).unwrap();
        });
    });

    group.bench_function("GEOS - Medium Square", |b| {
        b.iter(|| {
            buffer_with_params(black_box(&medium_square), black_box(5.0), &params).unwrap();
        });
    });

    group.finish();
}

#[cfg(feature = "rust-buffer")]
fn bench_wkt_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("WKT Round-trip with Buffer");

    let polygon = Geometry::from_wkt("POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))").unwrap();

    group.bench_function("Pure Rust Buffer + WKT Round-trip", |b| {
        b.iter(|| {
            let buffered = buffer_rust(black_box(&polygon), black_box(3.0)).unwrap();
            let wkt = buffered.to_wkt();
            Geometry::from_wkt(&wkt).unwrap();
        });
    });

    group.finish();
}

#[cfg(not(any(feature = "rust-buffer", feature = "geos-backend")))]
fn bench_no_features(_c: &mut Criterion) {
    eprintln!("❌ No buffer features enabled!");
    eprintln!("Enable rust-buffer or geos-backend to run benchmarks.");
}

// Configure benchmark groups based on available features
#[cfg(all(feature = "rust-buffer", feature = "geos-backend"))]
criterion_group!(
    benches,
    bench_pure_rust_buffer,
    bench_geos_buffer,
    bench_comparison,
    bench_wkt_roundtrip
);

#[cfg(all(feature = "rust-buffer", not(feature = "geos-backend")))]
criterion_group!(benches, bench_pure_rust_buffer, bench_wkt_roundtrip);

#[cfg(all(not(feature = "rust-buffer"), feature = "geos-backend"))]
criterion_group!(benches, bench_geos_buffer);

#[cfg(not(any(feature = "rust-buffer", feature = "geos-backend")))]
criterion_group!(benches, bench_no_features);

criterion_main!(benches);
