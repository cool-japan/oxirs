//! Benchmarks for spatial index operations
//!
//! These benchmarks measure the performance of:
//! - R-tree insertion
//! - Bounding box queries
//! - Nearest neighbor queries
//! - Scaling with dataset size

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;

/// Benchmark R-tree insertions
fn bench_rtree_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtree_insert");

    // Create sample geometries
    let point = Geometry::from_wkt("POINT(1.5 2.5)").unwrap();
    let linestring = Geometry::from_wkt("LINESTRING(0 0, 10 10)").unwrap();
    let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();

    group.bench_function("insert_point", |b| {
        let index = SpatialIndex::new();
        b.iter(|| {
            let _id = index.insert(black_box(point.clone())).unwrap();
        });
    });

    group.bench_function("insert_linestring", |b| {
        let index = SpatialIndex::new();
        b.iter(|| {
            let _id = index.insert(black_box(linestring.clone())).unwrap();
        });
    });

    group.bench_function("insert_polygon", |b| {
        let index = SpatialIndex::new();
        b.iter(|| {
            let _id = index.insert(black_box(polygon.clone())).unwrap();
        });
    });

    group.finish();
}

/// Benchmark bounding box queries
fn bench_bbox_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbox_query");

    // Create indices with different sizes
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        let index = SpatialIndex::new();

        // Populate index with random points
        for i in 0..size {
            let x = (i % 100) as f64;
            let y = (i / 100) as f64;
            let wkt = format!("POINT({} {})", x, y);
            let geom = Geometry::from_wkt(&wkt).unwrap();
            index.insert(geom).unwrap();
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("query_10x10", size), &index, |b, index| {
            b.iter(|| {
                let results = index.query_bbox(
                    black_box(0.0),
                    black_box(0.0),
                    black_box(10.0),
                    black_box(10.0),
                );
                black_box(results);
            });
        });

        group.bench_with_input(BenchmarkId::new("query_50x50", size), &index, |b, index| {
            b.iter(|| {
                let results = index.query_bbox(
                    black_box(0.0),
                    black_box(0.0),
                    black_box(50.0),
                    black_box(50.0),
                );
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark nearest neighbor queries
fn bench_nearest_neighbor(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_neighbor");

    // Create indices with different sizes
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        let index = SpatialIndex::new();

        // Populate index with points in a grid
        for i in 0..size {
            let x = (i % 100) as f64;
            let y = (i / 100) as f64;
            let wkt = format!("POINT({} {})", x, y);
            let geom = Geometry::from_wkt(&wkt).unwrap();
            index.insert(geom).unwrap();
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("nearest", size), &index, |b, index| {
            b.iter(|| {
                let result = index.nearest(black_box(25.5), black_box(25.5));
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark within-distance queries
fn bench_within_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("within_distance");

    // Create indices with different sizes
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        let index = SpatialIndex::new();

        // Populate index with random points
        for i in 0..size {
            let x = (i % 100) as f64;
            let y = (i / 100) as f64;
            let wkt = format!("POINT({} {})", x, y);
            let geom = Geometry::from_wkt(&wkt).unwrap();
            index.insert(geom).unwrap();
        }

        group.throughput(Throughput::Elements(size as u64));

        // Query with small radius
        group.bench_with_input(BenchmarkId::new("radius_5", size), &index, |b, index| {
            b.iter(|| {
                let results =
                    index.query_within_distance(black_box(50.0), black_box(50.0), black_box(5.0));
                black_box(results);
            });
        });

        // Query with large radius
        group.bench_with_input(BenchmarkId::new("radius_50", size), &index, |b, index| {
            b.iter(|| {
                let results =
                    index.query_within_distance(black_box(50.0), black_box(50.0), black_box(50.0));
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark bulk insertion
fn bench_bulk_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_insert");

    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        // Pre-create geometries
        let geometries: Vec<_> = (0..size)
            .map(|i| {
                let x = (i % 100) as f64;
                let y = (i / 100) as f64;
                let wkt = format!("POINT({} {})", x, y);
                Geometry::from_wkt(&wkt).unwrap()
            })
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("insert_many", size),
            &geometries,
            |b, geometries| {
                b.iter(|| {
                    let index = SpatialIndex::new();
                    for geom in geometries {
                        index.insert(black_box(geom.clone())).unwrap();
                    }
                    black_box(index);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark index operations with polygons
fn bench_polygon_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("polygon_index");

    let sizes = vec![50, 100, 200];

    for size in sizes {
        let index = SpatialIndex::new();

        // Create non-overlapping polygons in a grid
        for i in 0..size {
            let x = ((i % 10) * 15) as f64;
            let y = ((i / 10) * 15) as f64;
            let wkt = format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                x,
                y,
                x + 10.0,
                y,
                x + 10.0,
                y + 10.0,
                x,
                y + 10.0,
                x,
                y
            );
            let geom = Geometry::from_wkt(&wkt).unwrap();
            index.insert(geom).unwrap();
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("query_polygons", size),
            &index,
            |b, index| {
                b.iter(|| {
                    let results = index.query_bbox(
                        black_box(0.0),
                        black_box(0.0),
                        black_box(50.0),
                        black_box(50.0),
                    );
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark index removal
fn bench_index_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_removal");

    let size = 1000;
    let index = SpatialIndex::new();

    // Populate index
    let mut ids = Vec::new();
    for i in 0..size {
        let x = (i % 100) as f64;
        let y = (i / 100) as f64;
        let wkt = format!("POINT({} {})", x, y);
        let geom = Geometry::from_wkt(&wkt).unwrap();
        let id = index.insert(geom).unwrap();
        ids.push(id);
    }

    group.bench_function("remove_single", |b| {
        b.iter(|| {
            let id = black_box(ids[500]);
            let _removed = index.remove(id).unwrap();
            // Re-insert to keep index size constant
            let geom = Geometry::from_wkt("POINT(50 50)").unwrap();
            index.insert(geom).unwrap();
        });
    });

    group.finish();
}

/// Benchmark index with mixed geometry types
fn bench_mixed_geometries(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_geometries");

    let index = SpatialIndex::new();

    // Insert mix of points, lines, and polygons
    for i in 0..300 {
        match i % 3 {
            0 => {
                // Point
                let wkt = format!("POINT({} {})", i as f64, i as f64);
                let geom = Geometry::from_wkt(&wkt).unwrap();
                index.insert(geom).unwrap();
            }
            1 => {
                // LineString
                let wkt = format!(
                    "LINESTRING({} {}, {} {})",
                    i as f64,
                    i as f64,
                    i as f64 + 10.0,
                    i as f64 + 10.0
                );
                let geom = Geometry::from_wkt(&wkt).unwrap();
                index.insert(geom).unwrap();
            }
            2 => {
                // Polygon
                let x = i as f64;
                let wkt = format!(
                    "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                    x,
                    x,
                    x + 5.0,
                    x,
                    x + 5.0,
                    x + 5.0,
                    x,
                    x + 5.0,
                    x,
                    x
                );
                let geom = Geometry::from_wkt(&wkt).unwrap();
                index.insert(geom).unwrap();
            }
            _ => unreachable!(),
        }
    }

    group.bench_function("query_mixed", |b| {
        b.iter(|| {
            let results = index.query_bbox(
                black_box(0.0),
                black_box(0.0),
                black_box(100.0),
                black_box(100.0),
            );
            black_box(results);
        });
    });

    group.bench_function("nearest_mixed", |b| {
        b.iter(|| {
            let result = index.nearest(black_box(150.0), black_box(150.0));
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rtree_insert,
    bench_bbox_query,
    bench_nearest_neighbor,
    bench_within_distance,
    bench_bulk_insert,
    bench_polygon_index,
    bench_index_removal,
    bench_mixed_geometries,
);
criterion_main!(benches);
