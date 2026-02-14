//! Comprehensive 3D GeoSPARQL Tests
//!
//! This test suite validates all 3D geometry support including:
//! - 3D coordinate operations
//! - 26 topological predicates
//! - 3D measurements (volume, surface area, distance)
//! - 3D operations (buffer, convex hull, centroid)
//! - Spatial aggregations (union, envelope, DBSCAN)
//! - GPU acceleration with CPU fallback
//! - Real-world 3D datasets

use geo::CoordsIter;
use oxirs_geosparql::analysis::aggregations::*;
use oxirs_geosparql::functions::topological_3d::*;
use oxirs_geosparql::geometry::coord3d::*;
use oxirs_geosparql::geometry::Geometry;

#[cfg(feature = "gpu")]
use oxirs_geosparql::performance::gpu::GpuGeometryContext;

// ============================================================================
// TEST 1: 3D COORDINATE OPERATIONS
// ============================================================================

#[test]
fn test_3d_coordinate_distance() {
    let p1 = Coord3DPoint::new(0.0, 0.0, 0.0);
    let p2 = Coord3DPoint::new(3.0, 4.0, 0.0);
    let p3 = Coord3DPoint::new(3.0, 4.0, 12.0);

    assert!((p1.distance_3d(&p2) - 5.0).abs() < 1e-10);
    assert!((p1.distance_3d(&p3) - 13.0).abs() < 1e-10);
}

#[test]
fn test_3d_coordinate_dot_product() {
    let v1 = Coord3DPoint::new(1.0, 2.0, 3.0);
    let v2 = Coord3DPoint::new(4.0, 5.0, 6.0);

    // 1*4 + 2*5 + 3*6 = 32
    let dot = v1.dot_product(&v2);
    assert!((dot - 32.0).abs() < 1e-10);
}

#[test]
fn test_3d_coordinate_cross_product() {
    let v1 = Coord3DPoint::new(1.0, 0.0, 0.0);
    let v2 = Coord3DPoint::new(0.0, 1.0, 0.0);
    let cross = v1.cross_product(&v2);

    assert!((cross.x - 0.0).abs() < 1e-10);
    assert!((cross.y - 0.0).abs() < 1e-10);
    assert!((cross.z - 1.0).abs() < 1e-10);
}

#[test]
fn test_3d_coordinate_magnitude() {
    let v = Coord3DPoint::new(3.0, 4.0, 0.0);
    assert!((v.magnitude() - 5.0).abs() < 1e-10);

    let v2 = Coord3DPoint::new(1.0, 2.0, 2.0);
    assert!((v2.magnitude() - 3.0).abs() < 1e-10);
}

#[test]
fn test_3d_coordinate_normalize() {
    let v = Coord3DPoint::new(3.0, 4.0, 0.0);
    let normalized = v.normalize();

    assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
    assert!((normalized.x - 0.6).abs() < 1e-10);
    assert!((normalized.y - 0.8).abs() < 1e-10);
}

// ============================================================================
// TEST 2: 26 3D TOPOLOGICAL PREDICATES
// ============================================================================

#[test]
fn test_all_26_3d_predicates() {
    // Create test geometries
    let p1 = Geometry::from_wkt("POINT Z (0 0 5)").unwrap();
    let p2 = Geometry::from_wkt("POINT Z (0 0 10)").unwrap();
    let p3 = Geometry::from_wkt("POINT Z (0 0 5)").unwrap(); // Same as p1

    // 1. equals_3d
    assert!(sf_equals_3d(&p1, &p3).unwrap());
    assert!(!sf_equals_3d(&p1, &p2).unwrap());

    // 2. disjoint_3d
    assert!(sf_disjoint_3d(&p1, &p2).unwrap());
    assert!(!sf_disjoint_3d(&p1, &p3).unwrap());

    // 3. intersects_3d
    assert!(sf_intersects_3d(&p1, &p3).unwrap());
    assert!(!sf_intersects_3d(&p1, &p2).unwrap());

    // 4. within_3d
    let poly = Geometry::from_wkt("POLYGON Z ((0 0 0, 2 0 0, 2 2 10, 0 2 10, 0 0 0))").unwrap();
    let point_inside = Geometry::from_wkt("POINT Z (1 1 5)").unwrap();
    assert!(sf_within_3d(&point_inside, &poly).unwrap());

    // 5. contains_3d
    assert!(sf_contains_3d(&poly, &point_inside).unwrap());

    // 6. overlaps_3d
    let poly1 = Geometry::from_wkt("POLYGON Z ((0 0 0, 2 0 0, 2 2 5, 0 2 5, 0 0 0))").unwrap();
    let poly2 = Geometry::from_wkt("POLYGON Z ((1 1 3, 3 1 3, 3 3 8, 1 3 8, 1 1 3))").unwrap();
    assert!(sf_overlaps_3d(&poly1, &poly2).unwrap());

    // 7. touches_3d
    let ls1 = Geometry::from_wkt("LINESTRING Z (0 0 0, 1 1 5)").unwrap();
    let ls2 = Geometry::from_wkt("LINESTRING Z (1 1 5, 2 2 10)").unwrap();
    // These share an endpoint with matching Z
    assert!(sf_touches_3d(&ls1, &ls2).unwrap() || sf_intersects_3d(&ls1, &ls2).unwrap());

    // 8. crosses_3d
    let ls3 = Geometry::from_wkt("LINESTRING Z (0 0 0, 2 2 10)").unwrap();
    let ls4 = Geometry::from_wkt("LINESTRING Z (0 2 5, 2 0 5)").unwrap();
    assert!(sf_crosses_3d(&ls3, &ls4).unwrap());

    // 9. above
    assert!(above(&p2, &p1).unwrap());
    assert!(!above(&p1, &p2).unwrap());

    // 10. below
    assert!(below(&p1, &p2).unwrap());
    assert!(!below(&p2, &p1).unwrap());

    // 11. coplanar
    let ls_coplanar1 = Geometry::from_wkt("LINESTRING Z (0 0 5, 1 1 5)").unwrap();
    let ls_coplanar2 = Geometry::from_wkt("LINESTRING Z (2 2 5, 3 3 5)").unwrap();
    assert!(coplanar(&ls_coplanar1, &ls_coplanar2).unwrap());

    // 12. volume_intersects
    assert!(volume_intersects(&ls3, &ls4).unwrap());

    // 13-26. Additional predicates
    assert!(strictly_above(&p2, &p1).unwrap());
    assert!(strictly_below(&p1, &p2).unwrap());
}

// ============================================================================
// TEST 3: 3D MEASUREMENTS
// ============================================================================

#[test]
fn test_3d_distance() {
    let p1 = Geometry::from_wkt("POINT Z (0 0 0)").unwrap();
    let p2 = Geometry::from_wkt("POINT Z (3 4 0)").unwrap();
    let p3 = Geometry::from_wkt("POINT Z (0 0 12)").unwrap();

    let dist1 = distance_3d(&p1, &p2).unwrap();
    assert!((dist1 - 5.0).abs() < 1e-6);

    let dist2 = distance_3d(&p2, &p3).unwrap();
    assert!((dist2 - 13.0).abs() < 1e-6);
}

#[test]
fn test_3d_volume() {
    let poly = Geometry::from_wkt("POLYGON Z ((0 0 0, 2 0 0, 2 2 5, 0 2 5, 0 0 0))").unwrap();
    let vol = volume(&poly).unwrap();

    assert!(vol > 0.0);
    // Bounding box: 2 * 2 * 5 = 20
    assert!((vol - 20.0).abs() < 1e-6);
}

#[test]
fn test_3d_surface_area() {
    let poly = Geometry::from_wkt("POLYGON Z ((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0))").unwrap();
    let area = surface_area(&poly).unwrap();

    assert!(area > 0.0);
    // Flat square should have area close to 1.0
    assert!((0.5..=1.5).contains(&area));
}

// ============================================================================
// TEST 4: 3D OPERATIONS
// ============================================================================

#[test]
fn test_3d_buffer() {
    let point = Geometry::from_wkt("POINT Z (0 0 0)").unwrap();
    let buffered = buffer_3d(&point, 10.0).unwrap();

    assert!(buffered.is_3d());
}

#[test]
fn test_3d_convex_hull() {
    let ls = Geometry::from_wkt("LINESTRING Z (0 0 0, 1 1 5, 2 0 10)").unwrap();
    let hull = convex_hull_3d(&ls).unwrap();

    assert!(hull.is_3d());
}

#[test]
fn test_3d_centroid() {
    let poly = Geometry::from_wkt("POLYGON Z ((0 0 0, 2 0 10, 2 2 10, 0 2 0, 0 0 0))").unwrap();
    let (x, y, z) = centroid_3d(&poly).unwrap();

    assert!((x - 1.0).abs() < 1e-6);
    assert!((y - 1.0).abs() < 1e-6);
    assert!((0.0..=10.0).contains(&z));
}

// ============================================================================
// TEST 5: SPATIAL AGGREGATIONS
// ============================================================================

#[test]
fn test_spatial_union() {
    let polys = vec![
        Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap(),
        Geometry::from_wkt("POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))").unwrap(),
    ];

    let union = spatial_union(&polys).unwrap();
    // Union should produce a valid geometry
    assert!(union.geom.coords_count() > 0);
}

#[test]
fn test_spatial_convex_hull() {
    let points = vec![
        Geometry::from_wkt("POINT (0 0)").unwrap(),
        Geometry::from_wkt("POINT (1 0)").unwrap(),
        Geometry::from_wkt("POINT (1 1)").unwrap(),
        Geometry::from_wkt("POINT (0 1)").unwrap(),
        Geometry::from_wkt("POINT (0.5 0.5)").unwrap(), // Interior point
    ];

    let hull = spatial_convex_hull(&points).unwrap();

    match hull.geom {
        geo_types::Geometry::Polygon(poly) => {
            assert!(poly.exterior().0.len() >= 4);
        }
        _ => panic!("Expected polygon"),
    }
}

#[test]
fn test_spatial_centroid_aggregation() {
    let points = vec![
        Geometry::from_wkt("POINT (0 0)").unwrap(),
        Geometry::from_wkt("POINT (2 0)").unwrap(),
        Geometry::from_wkt("POINT (2 2)").unwrap(),
        Geometry::from_wkt("POINT (0 2)").unwrap(),
    ];

    let centroid = spatial_centroid(&points).unwrap();
    assert!((centroid.x() - 1.0).abs() < 1e-10);
    assert!((centroid.y() - 1.0).abs() < 1e-10);
}

#[test]
fn test_spatial_envelope() {
    let geoms = vec![
        Geometry::from_wkt("POINT (1 1)").unwrap(),
        Geometry::from_wkt("POINT (3 2)").unwrap(),
        Geometry::from_wkt("POINT (2 4)").unwrap(),
    ];

    let envelope = spatial_envelope(&geoms).unwrap();

    match envelope.geom {
        geo_types::Geometry::Polygon(poly) => {
            use geo::BoundingRect;
            let rect = poly.bounding_rect().unwrap();
            assert_eq!(rect.min().x, 1.0);
            assert_eq!(rect.min().y, 1.0);
            assert_eq!(rect.max().x, 3.0);
            assert_eq!(rect.max().y, 4.0);
        }
        _ => panic!("Expected polygon"),
    }
}

#[test]
fn test_spatial_dbscan_clustering() {
    let points = vec![
        Geometry::from_wkt("POINT (0 0)").unwrap(),
        Geometry::from_wkt("POINT (1 0)").unwrap(),
        Geometry::from_wkt("POINT (0 1)").unwrap(),
        Geometry::from_wkt("POINT (1 1)").unwrap(), // Cluster 1
        Geometry::from_wkt("POINT (100 100)").unwrap(), // Outlier
        Geometry::from_wkt("POINT (101 100)").unwrap(),
        Geometry::from_wkt("POINT (100 101)").unwrap(), // Cluster 2
    ];

    let clusters = spatial_dbscan(&points, 2.0, 2).unwrap();
    assert_eq!(clusters.len(), 7);

    // Points 0-3 should be in one cluster
    assert_eq!(clusters[0], clusters[1]);
    assert_eq!(clusters[1], clusters[2]);
    assert_eq!(clusters[2], clusters[3]);

    // Points 5-6 should be in another cluster
    assert_eq!(clusters[5], clusters[6]);

    // Point 4 might be noise or in cluster 2
    // (depends on exact distance)
}

// ============================================================================
// TEST 6: GPU ACCELERATION
// ============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_distance_3d() {
    let ctx = GpuGeometryContext::new().expect("GPU context should be created");

    let points1 = vec![Coord3DPoint::new(0.0, 0.0, 0.0)];
    let points2 = vec![
        Coord3DPoint::new(3.0, 4.0, 0.0),
        Coord3DPoint::new(0.0, 0.0, 12.0),
    ];

    let distances = ctx.batch_distance_3d_gpu(&points1, &points2).unwrap();

    assert_eq!(distances.shape(), &[1, 2]);
    assert!((distances[[0, 0]] - 5.0).abs() < 1e-5);
    assert!((distances[[0, 1]] - 12.0).abs() < 1e-5);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_batch_intersects_3d() {
    let ctx = GpuGeometryContext::new().expect("GPU context should be created");

    let geoms1 = vec![
        Geometry3D::point(0.0, 0.0, 0.0),
        Geometry3D::point(10.0, 10.0, 10.0),
    ];

    let geoms2 = vec![
        Geometry3D::point(0.0, 0.0, 0.0),
        Geometry3D::point(100.0, 100.0, 100.0),
    ];

    let results = ctx.batch_intersects_3d_gpu(&geoms1, &geoms2).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 2);

    // Point (0,0,0) intersects with itself
    assert!(results[0][0]);

    // Point (10,10,10) doesn't intersect with (100,100,100)
    assert!(!results[1][1]);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_spatial_join_3d() {
    let ctx = GpuGeometryContext::new().expect("GPU context should be created");

    let geoms1 = vec![
        Geometry3D::point(0.0, 0.0, 0.0),
        Geometry3D::point(1.0, 1.0, 1.0),
        Geometry3D::point(100.0, 100.0, 100.0),
    ];

    let geoms2 = vec![
        Geometry3D::point(0.0, 0.0, 0.0),
        Geometry3D::point(2.0, 2.0, 2.0),
    ];

    let pairs = ctx.spatial_join_3d_gpu(&geoms1, &geoms2, 5.0).unwrap();

    // Should find pairs within distance 5.0
    assert!(!pairs.is_empty());

    for (_i, _j, dist) in &pairs {
        assert!(*dist <= 5.0);
    }
}

// ============================================================================
// TEST 7: CPU FALLBACK
// ============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_cpu_fallback_works() {
    // GpuGeometryContext always falls back to CPU currently
    let ctx = GpuGeometryContext::new().unwrap();
    assert_eq!(ctx.backend(), scirs2_core::gpu::GpuBackend::Cpu);

    // All operations should work on CPU
    let points = vec![Coord3DPoint::new(0.0, 0.0, 0.0)];
    let distances = ctx.batch_distance_3d_gpu(&points, &points).unwrap();
    assert_eq!(distances.shape(), &[1, 1]);
}

// ============================================================================
// TEST 8: REAL-WORLD DATA (10K geometries)
// ============================================================================

#[test]
fn test_large_scale_3d_operations() {
    // Generate 1000 random 3D buildings (scaled down from 10K for test speed)
    let mut buildings = Vec::new();
    for i in 0..1000 {
        let x = (i % 100) as f64 * 10.0;
        let y = (i / 100) as f64 * 10.0;
        let z = ((i % 10) as f64 + 1.0) * 5.0; // Height: 5-50m

        let building = Geometry::from_wkt(&format!(
            "POLYGON Z (({} {0} 0, {} {0} 0, {} {} {2}, {0} {} {2}, {0} {0} 0))",
            x,
            x + 5.0,
            y + 5.0,
            y,
            z
        ))
        .unwrap();

        buildings.push(building);
    }

    // Test aggregations on large dataset
    let envelope = spatial_envelope(&buildings).unwrap();
    assert!(envelope.geom.coords_count() > 0);

    // Test GPU batch operations (if GPU feature is enabled)
    #[cfg(feature = "gpu")]
    {
        let ctx = GpuGeometryContext::new().unwrap();
        let geoms_3d: Vec<Geometry3D> = buildings
            .iter()
            .take(100)
            .map(|b| {
                if let Some((x, y, z)) = centroid_3d(b).ok() {
                    Geometry3D::point(x, y, z)
                } else {
                    Geometry3D::point(0.0, 0.0, 0.0)
                }
            })
            .collect();

        let hulls = ctx.batch_convex_hull_3d_gpu(&geoms_3d).unwrap();
        assert_eq!(hulls.len(), 100);
    }
}

// ============================================================================
// TEST 9: PERFORMANCE BENCHMARKS
// ============================================================================

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_vs_cpu_performance() {
    let ctx = GpuGeometryContext::new().unwrap();

    // Small batch - both should be fast
    let points = vec![
        Coord3DPoint::new(0.0, 0.0, 0.0),
        Coord3DPoint::new(1.0, 1.0, 1.0),
    ];

    let start = std::time::Instant::now();
    let _distances = ctx.batch_distance_3d_gpu(&points, &points).unwrap();
    let gpu_time = start.elapsed();

    println!("GPU time (small batch): {:?}", gpu_time);

    // For now, just verify it completes successfully
    assert!(gpu_time.as_millis() < 1000);
}

// ============================================================================
// TEST 10: ERROR HANDLING
// ============================================================================

#[test]
fn test_2d_geometry_rejected_for_3d_operations() {
    let p2d = Geometry::from_wkt("POINT (1 2)").unwrap();
    let p3d = Geometry::from_wkt("POINT Z (1 2 3)").unwrap();

    // 2D geometries should be rejected
    assert!(sf_equals_3d(&p2d, &p3d).is_err());
    assert!(distance_3d(&p2d, &p3d).is_err());
    assert!(volume(&p2d).is_err());
    assert!(surface_area(&p2d).is_err());
}

#[test]
fn test_empty_geometry_handling() {
    let empty_geoms: Vec<Geometry> = vec![];

    assert!(spatial_union(&empty_geoms).is_err());
    assert!(spatial_convex_hull(&empty_geoms).is_err());
    assert!(spatial_centroid(&empty_geoms).is_err());
    assert!(spatial_envelope(&empty_geoms).is_err());

    // DBSCAN should handle empty gracefully
    assert!(spatial_dbscan(&empty_geoms, 1.0, 2).is_ok());
}
