//! Benchmarks for GeoSPARQL spatial operations
//!
//! These benchmarks measure the performance of:
//! - WKT parsing and serialization
//! - Topological relations (Simple Features)
//! - Geometric operations (distance, envelope, convex hull)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_geosparql::functions::geometric_operations::{convex_hull, distance, envelope};
use oxirs_geosparql::functions::simple_features::{
    sf_contains, sf_crosses, sf_disjoint, sf_equals, sf_intersects, sf_overlaps, sf_touches,
    sf_within,
};
use oxirs_geosparql::geometry::Geometry;

/// Benchmark WKT parsing for different geometry types
fn bench_wkt_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("wkt_parsing");

    let test_cases = vec![
        ("point", "POINT(1.5 2.5)"),
        ("linestring", "LINESTRING(0 0, 1 1, 2 2, 3 3, 4 4)"),
        (
            "polygon",
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))",
        ),
        ("multipoint", "MULTIPOINT((0 0), (1 1), (2 2), (3 3))"),
        (
            "multilinestring",
            "MULTILINESTRING((0 0, 1 1), (2 2, 3 3), (4 4, 5 5))",
        ),
        (
            "multipolygon",
            "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((10 10, 15 10, 15 15, 10 15, 10 10)))",
        ),
    ];

    for (name, wkt) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &wkt, |b, wkt| {
            b.iter(|| {
                let geom = Geometry::from_wkt(black_box(wkt)).unwrap();
                black_box(geom);
            });
        });
    }

    group.finish();
}

/// Benchmark WKT serialization for different geometry types
fn bench_wkt_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("wkt_serialization");

    let test_cases = vec![
        ("point", Geometry::from_wkt("POINT(1.5 2.5)").unwrap()),
        (
            "linestring",
            Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2, 3 3, 4 4)").unwrap(),
        ),
        (
            "polygon",
            Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap(),
        ),
        (
            "multipoint",
            Geometry::from_wkt("MULTIPOINT((0 0), (1 1), (2 2))").unwrap(),
        ),
    ];

    for (name, geom) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &geom, |b, geom| {
            b.iter(|| {
                let wkt = geom.to_wkt();
                black_box(wkt);
            });
        });
    }

    group.finish();
}

/// Benchmark topological relations (Simple Features)
fn bench_topological_relations(c: &mut Criterion) {
    let mut group = c.benchmark_group("topological_relations");

    // Create test geometries
    let polygon1 = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
    let polygon2 = Geometry::from_wkt("POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))").unwrap();
    let point_inside = Geometry::from_wkt("POINT(2 2)").unwrap();
    let point_outside = Geometry::from_wkt("POINT(20 20)").unwrap();
    let linestring = Geometry::from_wkt("LINESTRING(0 5, 20 5)").unwrap();

    group.bench_function("sf_equals", |b| {
        b.iter(|| {
            let result = sf_equals(black_box(&polygon1), black_box(&polygon1)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sf_disjoint", |b| {
        b.iter(|| {
            let result = sf_disjoint(black_box(&polygon1), black_box(&point_outside)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sf_intersects", |b| {
        b.iter(|| {
            let result = sf_intersects(black_box(&polygon1), black_box(&polygon2)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sf_touches", |b| {
        b.iter(|| {
            let result = sf_touches(black_box(&polygon1), black_box(&polygon2)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sf_crosses", |b| {
        b.iter(|| {
            let result = sf_crosses(black_box(&linestring), black_box(&polygon1)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sf_within", |b| {
        b.iter(|| {
            let result = sf_within(black_box(&point_inside), black_box(&polygon1)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sf_contains", |b| {
        b.iter(|| {
            let result = sf_contains(black_box(&polygon1), black_box(&point_inside)).unwrap();
            black_box(result);
        });
    });

    group.bench_function("sf_overlaps", |b| {
        b.iter(|| {
            let result = sf_overlaps(black_box(&polygon1), black_box(&polygon2)).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark geometric operations
fn bench_geometric_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_operations");

    let point1 = Geometry::from_wkt("POINT(0 0)").unwrap();
    let point2 = Geometry::from_wkt("POINT(3 4)").unwrap();
    let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
    let multipoint = Geometry::from_wkt("MULTIPOINT((0 0), (1 1), (2 2), (3 3), (4 4))").unwrap();

    group.bench_function("distance_points", |b| {
        b.iter(|| {
            let dist = distance(black_box(&point1), black_box(&point2)).unwrap();
            black_box(dist);
        });
    });

    group.bench_function("envelope_polygon", |b| {
        b.iter(|| {
            let env = envelope(black_box(&polygon)).unwrap();
            black_box(env);
        });
    });

    group.bench_function("convex_hull_multipoint", |b| {
        b.iter(|| {
            let hull = convex_hull(black_box(&multipoint)).unwrap();
            black_box(hull);
        });
    });

    group.finish();
}

/// Benchmark scaling with geometry complexity
fn bench_scaling_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_complexity");

    // Test with polygons of different sizes
    let sizes = vec![5, 10, 20, 50, 100];

    for size in sizes {
        // Create a regular polygon with 'size' vertices
        let mut coords = Vec::new();
        let radius = 10.0;
        for i in 0..size {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (size as f64);
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            coords.push(format!("{} {}", x, y));
        }
        // Close the ring
        let first_coord = coords[0].clone();
        coords.push(first_coord);

        let wkt = format!("POLYGON(({})))", coords.join(", "));
        let polygon = Geometry::from_wkt(&wkt).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("envelope_polygon", size),
            &polygon,
            |b, polygon| {
                b.iter(|| {
                    let env = envelope(black_box(polygon)).unwrap();
                    black_box(env);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("wkt_serialization", size),
            &polygon,
            |b, polygon| {
                b.iter(|| {
                    let wkt = polygon.to_wkt();
                    black_box(wkt);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CRS handling
fn bench_crs_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("crs_handling");

    let wkt_with_crs = "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1.5 2.5)";
    let wkt_without_crs = "POINT(1.5 2.5)";

    group.bench_function("parse_with_crs", |b| {
        b.iter(|| {
            let geom = Geometry::from_wkt(black_box(wkt_with_crs)).unwrap();
            black_box(geom);
        });
    });

    group.bench_function("parse_without_crs", |b| {
        b.iter(|| {
            let geom = Geometry::from_wkt(black_box(wkt_without_crs)).unwrap();
            black_box(geom);
        });
    });

    group.finish();
}

/// Benchmark WKT parsing with 3D coordinates (Z/M) to measure pre-allocation optimizations
fn bench_wkt_parsing_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("wkt_parsing_3d");

    // Pre-create large WKT strings to avoid temporary lifetime issues
    let large_linestring_z = format!(
        "LINESTRING Z({})",
        (0..100)
            .map(|i| format!("{} {} {}", i, i * 2, i * 3))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let large_polygon_z = format!(
        "POLYGON Z(({}))",
        (0..50)
            .map(|i| {
                let angle = (i as f64) * 2.0 * std::f64::consts::PI / 50.0;
                let x = 10.0 * angle.cos();
                let y = 10.0 * angle.sin();
                format!("{} {} 5", x, y)
            })
            .chain(std::iter::once("10 0 5".to_string())) // Close the ring
            .collect::<Vec<_>>()
            .join(", ")
    );

    let test_cases: Vec<(&str, &str)> = vec![
        ("point_z", "POINT Z(1.5 2.5 3.5)"),
        ("point_m", "POINT M(1.5 2.5 4.0)"),
        ("point_zm", "POINT ZM(1.5 2.5 3.5 4.0)"),
        (
            "linestring_z",
            "LINESTRING Z(0 0 0, 1 1 1, 2 2 2, 3 3 3, 4 4 4, 5 5 5, 6 6 6, 7 7 7, 8 8 8, 9 9 9)",
        ),
        (
            "polygon_z",
            "POLYGON Z((0 0 5, 10 0 5, 10 10 5, 0 10 5, 0 0 5), (2 2 5, 8 2 5, 8 8 5, 2 8 5, 2 2 5))",
        ),
        (
            "multipoint_z",
            "MULTIPOINT Z((0 0 1), (1 1 2), (2 2 3), (3 3 4), (4 4 5))",
        ),
        (
            "multilinestring_z",
            "MULTILINESTRING Z((0 0 0, 1 1 1, 2 2 2), (3 3 3, 4 4 4), (5 5 5, 6 6 6, 7 7 7))",
        ),
        (
            "multipolygon_z",
            "MULTIPOLYGON Z(((0 0 5, 5 0 5, 5 5 5, 0 5 5, 0 0 5)), ((10 10 10, 15 10 10, 15 15 10, 10 15 10, 10 10 10)))",
        ),
        ("large_linestring_z", &large_linestring_z),
        ("large_polygon_z", &large_polygon_z),
    ];

    for (name, wkt) in test_cases {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(name), &wkt, |b, wkt| {
            b.iter(|| {
                let geom = Geometry::from_wkt(black_box(wkt)).unwrap();
                black_box(geom);
            });
        });
    }

    group.finish();
}

/// Benchmark CRS extraction with lazy static regex
fn bench_crs_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("crs_extraction");

    let test_cases =
        vec![
        ("no_crs", "POINT(1 2)"),
        ("with_crs", "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1 2)"),
        (
            "with_crs_complex",
            "<http://www.opengis.net/def/crs/EPSG/0/4326> POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
        ),
    ];

    for (name, wkt) in test_cases {
        group.bench_with_input(BenchmarkId::from_parameter(name), &wkt, |b, wkt| {
            b.iter(|| {
                let geom = Geometry::from_wkt(black_box(wkt)).unwrap();
                black_box(geom);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_wkt_parsing,
    bench_wkt_serialization,
    bench_topological_relations,
    bench_geometric_operations,
    bench_scaling_complexity,
    bench_crs_handling,
    bench_wkt_parsing_3d,
    bench_crs_extraction,
);
criterion_main!(benches);
