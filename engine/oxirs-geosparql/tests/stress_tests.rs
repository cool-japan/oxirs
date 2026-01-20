//! Stress tests for spatial index with large datasets
//!
//! These tests verify correctness and performance with realistic large-scale data.

use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point, Polygon};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::SpatialIndex;
use std::time::Instant;

/// Test bulk loading with 10,000 points
#[test]
fn test_bulk_load_10k_points() {
    let start = Instant::now();

    // Generate 10,000 random points in a grid pattern
    let mut geometries = Vec::with_capacity(10_000);
    for i in 0..100 {
        for j in 0..100 {
            let point = Point::new(i as f64, j as f64);
            geometries.push(Geometry::new(GeoGeometry::Point(point)));
        }
    }

    let index = SpatialIndex::bulk_load(geometries).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(index.len(), 10_000);
    println!("Bulk load 10k points: {:?}", elapsed);

    // Should complete in reasonable time (< 100ms on modern hardware)
    assert!(
        elapsed.as_millis() < 500,
        "Bulk load too slow: {:?}",
        elapsed
    );
}

/// Test batch insert vs individual insert performance
#[test]
fn test_batch_vs_individual_insert() {
    let geometries: Vec<_> = (0..1000)
        .map(|i| Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64))))
        .collect();

    // Test individual insert
    let index1 = SpatialIndex::new();
    let start1 = Instant::now();
    for geom in geometries.iter().cloned() {
        index1.insert(geom).unwrap();
    }
    let individual_time = start1.elapsed();

    // Test batch insert
    let index2 = SpatialIndex::new();
    let start2 = Instant::now();
    index2.insert_batch(geometries).unwrap();
    let batch_time = start2.elapsed();

    println!("Individual insert (1000 items): {:?}", individual_time);
    println!("Batch insert (1000 items): {:?}", batch_time);

    assert_eq!(index1.len(), 1000);
    assert_eq!(index2.len(), 1000);

    // Batch should be faster (but this is not guaranteed on all systems)
    println!(
        "Batch speedup: {:.2}x",
        individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64
    );
}

/// Test bbox query performance with large index
#[test]
fn test_bbox_query_large_index() {
    let index = SpatialIndex::new();

    // Insert 10,000 points in a grid
    for i in 0..100 {
        for j in 0..100 {
            index
                .insert(Geometry::new(GeoGeometry::Point(Point::new(
                    i as f64, j as f64,
                ))))
                .unwrap();
        }
    }

    // Query small bbox (should be fast)
    let start = Instant::now();
    let results = index.query_bbox(0.0, 0.0, 10.0, 10.0);
    let small_query_time = start.elapsed();

    // Should find 11x11 = 121 points (0-10 inclusive in both dimensions)
    assert_eq!(results.len(), 121);
    println!("Small bbox query (121 results): {:?}", small_query_time);

    // Query large bbox
    let start = Instant::now();
    let results = index.query_bbox(0.0, 0.0, 90.0, 90.0);
    let large_query_time = start.elapsed();

    // Should find 91x91 = 8281 points
    assert_eq!(results.len(), 8281);
    println!("Large bbox query (8281 results): {:?}", large_query_time);

    // Both queries should complete quickly (< 10ms)
    assert!(small_query_time.as_millis() < 10, "Small query too slow");
    assert!(large_query_time.as_millis() < 50, "Large query too slow");
}

/// Test nearest neighbor query with large index
#[test]
fn test_nearest_neighbor_large_index() {
    let index = SpatialIndex::new();

    // Insert 5,000 points in a grid
    for i in 0..100 {
        for j in 0..50 {
            index
                .insert(Geometry::new(GeoGeometry::Point(Point::new(
                    i as f64 * 2.0,
                    j as f64 * 2.0,
                ))))
                .unwrap();
        }
    }

    // Find nearest to various query points
    let test_points = [
        (0.5, 0.5),    // Near origin
        (50.0, 50.0),  // Middle
        (199.0, 99.0), // Near far corner
    ];

    for (x, y) in &test_points {
        let start = Instant::now();
        let result = index.nearest(*x, *y);
        let elapsed = start.elapsed();

        assert!(result.is_some(), "No nearest neighbor found");
        let (_nearest_geom, distance) = result.unwrap();

        println!(
            "Nearest to ({}, {}): distance = {:.2}, time = {:?}",
            x, y, distance, elapsed
        );

        // Should complete very quickly (< 1ms typically)
        assert!(
            elapsed.as_micros() < 5000,
            "Nearest neighbor query too slow"
        );

        // Distance should be reasonable (< 2.0 for grid with spacing 2.0)
        assert!(distance < 2.0, "Nearest neighbor too far: {}", distance);
    }
}

/// Test k-nearest neighbors with large index
#[test]
fn test_k_nearest_large_index() {
    let index = SpatialIndex::new();

    // Insert 1,000 points in a pseudo-random but deterministic pattern
    for i in 0..1000 {
        let x = ((i * 7) % 100) as f64 + (i % 10) as f64 / 10.0;
        let y = ((i * 13) % 100) as f64 + ((i / 10) % 10) as f64 / 10.0;
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(x, y))))
            .unwrap();
    }

    // Find 10 nearest neighbors
    let start = Instant::now();
    let nearest = index.nearest_k(50.0, 50.0, 10);
    let elapsed = start.elapsed();

    assert_eq!(nearest.len(), 10);
    println!("k=10 nearest query: {:?}", elapsed);

    // Verify results are sorted by distance
    for i in 0..nearest.len() - 1 {
        assert!(
            nearest[i].1 <= nearest[i + 1].1,
            "Results not sorted by distance: {} > {}",
            nearest[i].1,
            nearest[i + 1].1
        );
    }

    // Should complete quickly (< 5ms)
    assert!(elapsed.as_millis() < 10, "k-nearest query too slow");
}

/// Test distance query with large index
#[test]
fn test_distance_query_large_index() {
    let index = SpatialIndex::new();

    // Insert 5,000 points in a grid
    for i in 0..100 {
        for j in 0..50 {
            index
                .insert(Geometry::new(GeoGeometry::Point(Point::new(
                    i as f64, j as f64,
                ))))
                .unwrap();
        }
    }

    // Query for points within distance 10.0 from center
    let start = Instant::now();
    let results = index.query_within_distance(50.0, 25.0, 10.0);
    let elapsed = start.elapsed();

    println!(
        "Distance query (radius 10.0): {} results in {:?}",
        results.len(),
        elapsed
    );

    // All results should be within distance
    for (_geom, dist) in &results {
        assert!(dist <= &10.0, "Result outside distance threshold: {}", dist);
    }

    // Should have many results (roughly π * 10^2 ≈ 314 points)
    assert!(
        results.len() > 200 && results.len() < 400,
        "Unexpected result count: {}",
        results.len()
    );

    // Should complete quickly (< 10ms)
    assert!(elapsed.as_millis() < 20, "Distance query too slow");
}

/// Test with complex polygons
#[test]
fn test_large_polygon_index() {
    let index = SpatialIndex::new();

    // Create 100 polygons with varying complexity
    for i in 0..100 {
        let base_x = (i % 10) as f64 * 20.0;
        let base_y = (i / 10) as f64 * 20.0;

        // Create a square polygon
        let exterior = LineString::new(vec![
            Coord {
                x: base_x,
                y: base_y,
            },
            Coord {
                x: base_x + 15.0,
                y: base_y,
            },
            Coord {
                x: base_x + 15.0,
                y: base_y + 15.0,
            },
            Coord {
                x: base_x,
                y: base_y + 15.0,
            },
            Coord {
                x: base_x,
                y: base_y,
            },
        ]);

        let polygon = Polygon::new(exterior, vec![]);
        index
            .insert(Geometry::new(GeoGeometry::Polygon(polygon)))
            .unwrap();
    }

    assert_eq!(index.len(), 100);

    // Query overlapping bbox
    let start = Instant::now();
    let results = index.query_bbox(0.0, 0.0, 50.0, 50.0);
    let elapsed = start.elapsed();

    println!(
        "Polygon bbox query: {} results in {:?}",
        results.len(),
        elapsed
    );

    // Should find multiple polygons
    assert!(!results.is_empty());
    assert!(elapsed.as_millis() < 10, "Polygon query too slow");
}

/// Test concurrent access (read-heavy workload)
#[test]
#[cfg(feature = "parallel")]
fn test_concurrent_queries() {
    use rayon::prelude::*;

    let index = SpatialIndex::new();

    // Insert 1,000 points in a deterministic pattern
    for i in 0..1000 {
        let x = ((i * 7) % 100) as f64 + (i % 10) as f64 / 10.0;
        let y = ((i * 13) % 100) as f64 + ((i / 10) % 10) as f64 / 10.0;
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(x, y))))
            .unwrap();
    }

    // Perform 100 concurrent queries
    let start = Instant::now();
    let results: Vec<_> = (0..100)
        .into_par_iter()
        .map(|i| {
            let x = (i % 10) as f64 * 10.0;
            let y = (i / 10) as f64 * 10.0;
            index.query_bbox(x, y, x + 10.0, y + 10.0).len()
        })
        .collect();
    let elapsed = start.elapsed();

    println!("100 concurrent queries: {:?}", elapsed);

    // All queries should complete
    assert_eq!(results.len(), 100);

    // Should complete quickly with parallel processing
    assert!(elapsed.as_millis() < 100, "Concurrent queries too slow");
}

/// Test memory usage with large index
#[test]
fn test_memory_efficiency() {
    let index = SpatialIndex::new();

    // Insert 50,000 points
    let start = Instant::now();
    for i in 0..50_000 {
        let x = (i % 500) as f64;
        let y = (i / 500) as f64;
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(x, y))))
            .unwrap();
    }
    let insert_time = start.elapsed();

    println!("Inserted 50k points in {:?}", insert_time);
    assert_eq!(index.len(), 50_000);

    // Perform queries to verify index still works
    let results = index.query_bbox(0.0, 0.0, 100.0, 100.0);
    assert!(!results.is_empty());

    // Clear should free memory
    index.clear();
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

/// Test remove operation at scale
#[test]
fn test_bulk_remove() {
    let index = SpatialIndex::new();

    // Insert 1,000 points and track IDs
    let mut ids = Vec::new();
    for i in 0..1000 {
        let id = index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(
                i as f64, i as f64,
            ))))
            .unwrap();
        ids.push(id);
    }

    assert_eq!(index.len(), 1000);

    // Remove every other point
    let start = Instant::now();
    let mut removed_count = 0;
    for id in ids.iter().step_by(2) {
        if index.remove(*id).unwrap() {
            removed_count += 1;
        }
    }
    let elapsed = start.elapsed();

    println!("Removed {} points in {:?}", removed_count, elapsed);

    assert_eq!(removed_count, 500);
    assert_eq!(index.len(), 500);

    // Should complete in reasonable time
    assert!(elapsed.as_millis() < 200, "Bulk remove too slow");
}

/// Test edge case: empty queries
#[test]
fn test_empty_region_queries() {
    let index = SpatialIndex::new();

    // Insert points only in [0, 100] x [0, 100]
    for i in 0..100 {
        for j in 0..100 {
            index
                .insert(Geometry::new(GeoGeometry::Point(Point::new(
                    i as f64, j as f64,
                ))))
                .unwrap();
        }
    }

    // Query a region with no points
    let results = index.query_bbox(1000.0, 1000.0, 1100.0, 1100.0);
    assert_eq!(results.len(), 0, "Should find no points in empty region");

    // Distance query far from all points
    let results = index.query_within_distance(1000.0, 1000.0, 10.0);
    assert_eq!(results.len(), 0, "Should find no points far from all data");

    // Nearest neighbor should still find something
    let result = index.nearest(1000.0, 1000.0);
    assert!(
        result.is_some(),
        "Nearest should find closest point even when far"
    );
}

/// Test spatial locality - nearby queries should return nearby results
#[test]
fn test_spatial_locality() {
    let index = SpatialIndex::new();

    // Create clusters of points
    let clusters = [(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)];

    for (cx, cy) in &clusters {
        // Add 20 points around each cluster center
        for i in 0..20 {
            let angle = (i as f64) * std::f64::consts::PI * 2.0 / 20.0;
            let x = cx + 2.0 * angle.cos();
            let y = cy + 2.0 * angle.sin();
            index
                .insert(Geometry::new(GeoGeometry::Point(Point::new(x, y))))
                .unwrap();
        }
    }

    // Query near each cluster - should find mostly points from that cluster
    for (cx, cy) in &clusters {
        let results = index.query_within_distance(*cx, *cy, 5.0);

        // Should find all 20 points in the cluster
        assert!(
            results.len() >= 20,
            "Should find cluster points near ({}, {})",
            cx,
            cy
        );

        // All results should be close
        for (_geom, dist) in &results {
            assert!(dist <= &5.0, "Found point too far: {}", dist);
        }
    }
}
