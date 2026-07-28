//! Buffer benchmarks.
//!
//! Run with: cargo bench --bench buffer_performance
//!
//! Buffering is unconditional Pure Rust (`geo::algorithm::buffer`), so there is
//! nothing to feature-gate here any more. This benchmark used to compare a
//! `rust-buffer` straight-skeleton path against a GEOS C-library backend; both
//! are gone.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_geosparql::functions::geometric_operations::buffer;
use oxirs_geosparql::geometry::Geometry;
use std::hint::black_box;

/// Create test polygons of various sizes
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

/// Geometry types the old straight-skeleton buffer could not handle at all.
fn create_non_polygon_geometries() -> Vec<(String, Geometry)> {
    vec![
        (
            "Point".to_string(),
            Geometry::from_wkt("POINT(0 0)").unwrap(),
        ),
        (
            "LineString (3 vertices)".to_string(),
            Geometry::from_wkt("LINESTRING(0 0, 50 0, 50 50)").unwrap(),
        ),
        (
            "MultiPoint (4 points)".to_string(),
            Geometry::from_wkt("MULTIPOINT((0 0), (10 10), (20 0), (30 10))").unwrap(),
        ),
    ]
}

fn bench_polygon_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polygon Buffer");

    for (name, geom) in create_test_polygons() {
        group.bench_with_input(
            BenchmarkId::new("Positive Buffer (2.0)", &name),
            &geom,
            |b, geom| {
                b.iter(|| {
                    buffer(black_box(geom), black_box(2.0)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Negative Buffer (-2.0)", &name),
            &geom,
            |b, geom| {
                b.iter(|| {
                    buffer(black_box(geom), black_box(-2.0)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_non_polygon_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("Non-Polygon Buffer");

    for (name, geom) in create_non_polygon_geometries() {
        group.bench_with_input(
            BenchmarkId::new("Positive Buffer (2.0)", &name),
            &geom,
            |b, geom| {
                b.iter(|| {
                    buffer(black_box(geom), black_box(2.0)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_wkt_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("WKT Round-trip with Buffer");

    let polygon = Geometry::from_wkt("POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))").unwrap();

    group.bench_function("Buffer + WKT Round-trip", |b| {
        b.iter(|| {
            let buffered = buffer(black_box(&polygon), black_box(3.0)).unwrap();
            let wkt = buffered.to_wkt();
            Geometry::from_wkt(&wkt).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_polygon_buffer,
    bench_non_polygon_buffer,
    bench_wkt_roundtrip
);
criterion_main!(benches);
