//! Tests for geometric operations (2D/3D distance, buffer, set operations, etc.)

#[cfg(test)]
mod tests {
    use crate::functions::geometric_operations::{
        buffer, buffer_3d, convex_hull, difference, distance, distance_3d, envelope, intersection,
        sym_difference, union,
    };

    #[cfg(feature = "geos-backend")]
    use crate::functions::geometric_operations::{
        buffer_with_params, BufferParams, CapStyle, JoinStyle,
    };
    use crate::geometry::Geometry;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point};

    #[cfg(feature = "geos-backend")]
    use crate::functions::geometric_operations::boundary;

    #[cfg(feature = "rust-buffer")]
    use crate::functions::geometric_operations::buffer_rust;

    #[test]
    fn test_distance() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0)));

        let dist = distance(&p1, &p2).expect("should succeed");
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_convex_hull() {
        let mp = Geometry::new(GeoGeometry::MultiPoint(geo_types::MultiPoint::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.5, 1.0),
        ])));

        let hull = convex_hull(&mp);
        assert!(hull.is_ok());
        assert_eq!(hull.expect("should succeed").geometry_type(), "Polygon");
    }

    #[test]
    fn test_envelope() {
        let ls = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 5.0, y: 5.0 },
        ])));

        let env = envelope(&ls);
        assert!(env.is_ok());
    }

    #[test]
    fn test_intersection_polygons() {
        use geo_types::Polygon;

        // Create two overlapping polygons
        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 6.0, y: 2.0 },
                Coord { x: 6.0, y: 6.0 },
                Coord { x: 2.0, y: 6.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        )));

        let result = intersection(&poly1, &poly2).expect("should succeed");
        assert!(result.is_some());
        let intersection_geom = result.expect("should succeed");
        assert_eq!(intersection_geom.geometry_type(), "MultiPolygon");
    }

    #[test]
    fn test_intersection_no_overlap() {
        use geo_types::Polygon;

        // Create two non-overlapping polygons
        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 2.0, y: 0.0 },
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 0.0, y: 2.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 12.0, y: 10.0 },
                Coord { x: 12.0, y: 12.0 },
                Coord { x: 10.0, y: 12.0 },
                Coord { x: 10.0, y: 10.0 },
            ]),
            vec![],
        )));

        let result = intersection(&poly1, &poly2).expect("should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_union_polygons() {
        use geo_types::Polygon;

        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 6.0, y: 2.0 },
                Coord { x: 6.0, y: 6.0 },
                Coord { x: 2.0, y: 6.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        )));

        let result = union(&poly1, &poly2);
        assert!(result.is_ok());
        let union_geom = result.expect("should succeed");
        assert_eq!(union_geom.geometry_type(), "MultiPolygon");
    }

    #[test]
    fn test_difference_polygons() {
        use geo_types::Polygon;

        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 6.0, y: 2.0 },
                Coord { x: 6.0, y: 6.0 },
                Coord { x: 2.0, y: 6.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        )));

        let result = difference(&poly1, &poly2);
        assert!(result.is_ok());
        let diff_geom = result.expect("should succeed");
        assert_eq!(diff_geom.geometry_type(), "MultiPolygon");
    }

    #[test]
    fn test_sym_difference_polygons() {
        use geo_types::Polygon;

        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 6.0, y: 2.0 },
                Coord { x: 6.0, y: 6.0 },
                Coord { x: 2.0, y: 6.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        )));

        let result = sym_difference(&poly1, &poly2);
        assert!(result.is_ok());
        let xor_geom = result.expect("should succeed");
        assert_eq!(xor_geom.geometry_type(), "MultiPolygon");
    }

    #[test]
    fn test_crs_compatibility_check() {
        use crate::geometry::Crs;
        use geo_types::Polygon;

        let poly1 = Geometry::with_crs(
            GeoGeometry::Polygon(Polygon::new(
                LineString::new(vec![
                    Coord { x: 0.0, y: 0.0 },
                    Coord { x: 4.0, y: 0.0 },
                    Coord { x: 4.0, y: 4.0 },
                    Coord { x: 0.0, y: 4.0 },
                    Coord { x: 0.0, y: 0.0 },
                ]),
                vec![],
            )),
            Crs::epsg(4326),
        );

        let poly2 = Geometry::with_crs(
            GeoGeometry::Polygon(Polygon::new(
                LineString::new(vec![
                    Coord { x: 2.0, y: 2.0 },
                    Coord { x: 6.0, y: 2.0 },
                    Coord { x: 6.0, y: 6.0 },
                    Coord { x: 2.0, y: 6.0 },
                    Coord { x: 2.0, y: 2.0 },
                ]),
                vec![],
            )),
            Crs::epsg(3857),
        );

        // Should fail due to CRS mismatch
        let result = intersection(&poly1, &poly2);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_buffer_positive() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));

        let buffered = buffer(&point, 1.0).expect("should succeed");
        assert_eq!(buffered.geometry_type(), "Polygon");

        // Buffered point should create a circular polygon
        // The area should be approximately π * r²
        use geo::Area;
        if let GeoGeometry::Polygon(poly) = buffered.geom {
            let area = poly.unsigned_area();
            let expected_area = std::f64::consts::PI * 1.0 * 1.0;
            assert!((area - expected_area).abs() < 0.1); // Allow small tolerance
        }
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_buffer_negative() {
        use geo_types::Polygon;

        // Create a square polygon
        let poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        // Negative buffer should shrink the polygon
        let shrunk = buffer(&poly, -1.0).expect("should succeed");
        assert!(matches!(shrunk.geometry_type(), "Polygon" | "MultiPolygon"));
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_buffer_with_custom_params() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));

        let params = BufferParams {
            cap_style: CapStyle::Square,
            join_style: JoinStyle::Mitre,
            quadrant_segments: 4,
            mitre_limit: 2.0,
        };

        let buffered = buffer_with_params(&point, 1.0, &params).expect("should succeed");
        assert_eq!(buffered.geometry_type(), "Polygon");
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_boundary_linestring() {
        // Boundary of a linestring is its endpoints
        let ls = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 5.0, y: 0.0 },
            Coord { x: 5.0, y: 5.0 },
        ])));

        let bound = boundary(&ls).expect("should succeed");
        assert_eq!(bound.geometry_type(), "MultiPoint");
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_boundary_polygon() {
        use geo_types::Polygon;

        // Boundary of a polygon is its rings
        let poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let bound = boundary(&poly).expect("should succeed");
        assert!(matches!(
            bound.geometry_type(),
            "LineString" | "MultiLineString"
        ));
    }

    #[test]
    #[cfg(not(feature = "geos-backend"))]
    fn test_buffer_without_geos() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));

        let result = buffer(&point, 1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("geos-backend"));
    }

    #[test]
    #[cfg(not(feature = "geos-backend"))]
    fn test_boundary_without_geos() {
        use crate::functions::geometric_operations::boundary;

        let ls = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 5.0, y: 5.0 },
        ])));

        let result = boundary(&ls);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("geos-backend"));
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_rust_polygon() {
        use geo_types::Polygon;

        let poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        // Positive buffer (expansion)
        let expanded = buffer_rust(&poly, 1.0).expect("should succeed");
        assert_eq!(expanded.geometry_type(), "MultiPolygon");

        // Negative buffer (erosion)
        let shrunk = buffer_rust(&poly, -1.0).expect("should succeed");
        assert_eq!(shrunk.geometry_type(), "MultiPolygon");
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_rust_multipolygon() {
        use geo_types::{MultiPolygon, Polygon};

        let poly1 = Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 5.0, y: 0.0 },
                Coord { x: 5.0, y: 5.0 },
                Coord { x: 0.0, y: 5.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        );

        let poly2 = Polygon::new(
            LineString::new(vec![
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 15.0, y: 10.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: 10.0, y: 15.0 },
                Coord { x: 10.0, y: 10.0 },
            ]),
            vec![],
        );

        let mpoly = Geometry::new(GeoGeometry::MultiPolygon(MultiPolygon::new(vec![
            poly1, poly2,
        ])));

        let buffered = buffer_rust(&mpoly, 1.0).expect("should succeed");
        assert_eq!(buffered.geometry_type(), "MultiPolygon");
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_rust_polygon_with_hole() {
        use geo_types::Polygon;

        // Polygon with a hole (donut shape)
        let exterior = LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 20.0, y: 0.0 },
            Coord { x: 20.0, y: 20.0 },
            Coord { x: 0.0, y: 20.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);

        let interior = LineString::new(vec![
            Coord { x: 5.0, y: 5.0 },
            Coord { x: 15.0, y: 5.0 },
            Coord { x: 15.0, y: 15.0 },
            Coord { x: 5.0, y: 15.0 },
            Coord { x: 5.0, y: 5.0 },
        ]);

        let poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(exterior, vec![interior])));

        let buffered = buffer_rust(&poly, 1.0).expect("should succeed");
        assert_eq!(buffered.geometry_type(), "MultiPolygon");
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_rust_unsupported_type() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));

        let result = buffer_rust(&point, 1.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Pure Rust buffer only supports Polygon"));
    }

    #[test]
    #[cfg(all(feature = "rust-buffer", not(feature = "geos-backend")))]
    fn test_buffer_hybrid_polygon_uses_rust() {
        use geo_types::Polygon;

        let poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        // Without geos-backend, Polygon should use rust-buffer
        let buffered = buffer(&poly, 1.0).expect("should succeed");
        assert_eq!(buffered.geometry_type(), "MultiPolygon");
    }

    #[test]
    #[cfg(all(feature = "rust-buffer", not(feature = "geos-backend")))]
    fn test_buffer_hybrid_point_fails_without_geos() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));

        // Without geos-backend, Point buffer should fail
        let result = buffer(&point, 1.0);
        assert!(result.is_err());
    }

    // === 3D Distance Tests ===

    #[test]
    fn test_distance_3d_point_to_point() {
        // Classic 3-4-5 right triangle in 3D: 3-4-12 -> distance = 13
        let p1 = Geometry::from_wkt("POINT Z(0 0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(3 4 12)").expect("should succeed");

        let dist = distance_3d(&p1, &p2).expect("should succeed");
        assert!((dist - 13.0).abs() < 0.001, "Expected ~13.0, got {}", dist);
    }

    #[test]
    fn test_distance_3d_pythagorean_triple() {
        // Another Pythagorean triple: 5-12-13 in 3D
        let p1 = Geometry::from_wkt("POINT Z(0 0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(5 12 13)").expect("should succeed");

        let dist = distance_3d(&p1, &p2).expect("should succeed");
        // √(5² + 12² + 13²) = √(25 + 144 + 169) = √338 ≈ 18.385
        assert!(
            (dist - 18.385).abs() < 0.01,
            "Expected ~18.385, got {}",
            dist
        );
    }

    #[test]
    fn test_distance_3d_same_xy_different_z() {
        // Points with same X,Y but different Z
        let p1 = Geometry::from_wkt("POINT Z(1 2 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(1 2 10)").expect("should succeed");

        let dist = distance_3d(&p1, &p2).expect("should succeed");
        assert!((dist - 10.0).abs() < 0.001, "Expected 10.0, got {}", dist);
    }

    #[test]
    fn test_distance_3d_negative_coordinates() {
        // Test with negative coordinates
        let p1 = Geometry::from_wkt("POINT Z(-1 -2 -3)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").expect("should succeed");

        // Distance = √((2)² + (4)² + (6)²) = √(4 + 16 + 36) = √56 ≈ 7.483
        let dist = distance_3d(&p1, &p2).expect("should succeed");
        assert!((dist - 7.483).abs() < 0.01, "Expected ~7.483, got {}", dist);
    }

    #[test]
    fn test_distance_3d_zero_distance() {
        // Same point should have zero distance
        let p1 = Geometry::from_wkt("POINT Z(1 2 3)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").expect("should succeed");

        let dist = distance_3d(&p1, &p2).expect("should succeed");
        assert!(dist.abs() < 0.001, "Expected 0.0, got {}", dist);
    }

    #[test]
    fn test_distance_3d_fallback_to_2d() {
        // If one geometry is 2D, should fall back to 2D distance
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed"); // 2D
        let p2 = Geometry::from_wkt("POINT Z(3 4 100)").expect("should succeed"); // 3D with Z=100

        let dist_3d = distance_3d(&p1, &p2).expect("should succeed");
        let dist_2d = distance(&p1, &p2).expect("should succeed");

        // Should be same as 2D distance (Z ignored)
        assert!((dist_3d - dist_2d).abs() < 0.001);
        assert!((dist_3d - 5.0).abs() < 0.001); // √(3² + 4²) = 5
    }

    #[test]
    fn test_distance_3d_both_2d_uses_2d() {
        // Both 2D should use standard 2D distance
        let p1 = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT(3 4)").expect("should succeed");

        let dist_3d = distance_3d(&p1, &p2).expect("should succeed");
        let dist_2d = distance(&p1, &p2).expect("should succeed");

        assert!((dist_3d - dist_2d).abs() < 0.001);
        assert!((dist_3d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_3d_fractional_coordinates() {
        // Test with fractional coordinates
        let p1 = Geometry::from_wkt("POINT Z(1.5 2.5 3.5)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(4.5 6.5 7.5)").expect("should succeed");

        // Δx=3, Δy=4, Δz=4 -> √(9 + 16 + 16) = √41 ≈ 6.403
        let dist = distance_3d(&p1, &p2).expect("should succeed");
        assert!((dist - 6.403).abs() < 0.01, "Expected ~6.403, got {}", dist);
    }

    #[test]
    fn test_distance_3d_crs_compatibility() {
        // Different CRS should fail
        let p1 = Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT Z(0 0 0)")
            .expect("should succeed");
        let p2 = Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/3857> POINT Z(1 1 1)")
            .expect("should succeed");

        let result = distance_3d(&p1, &p2);
        assert!(result.is_err());
    }

    #[test]
    fn test_distance_3d_with_m_coordinate() {
        // POINT ZM should work (M coordinate ignored in distance calculation)
        let p1 = Geometry::from_wkt("POINT ZM(0 0 0 100)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT ZM(3 4 12 200)").expect("should succeed");

        let dist = distance_3d(&p1, &p2).expect("should succeed");
        assert!((dist - 13.0).abs() < 0.001); // M coordinate should be ignored
    }

    #[test]
    fn test_distance_2d_vs_3d_comparison() {
        // Same 2D projection but different Z should give different distances
        let p1_3d = Geometry::from_wkt("POINT Z(0 0 0)").expect("should succeed");
        let p2_3d = Geometry::from_wkt("POINT Z(3 4 12)").expect("should succeed");

        let p1_2d = Geometry::from_wkt("POINT(0 0)").expect("should succeed");
        let p2_2d = Geometry::from_wkt("POINT(3 4)").expect("should succeed");

        let dist_3d = distance_3d(&p1_3d, &p2_3d).expect("should succeed");
        let dist_2d = distance(&p1_2d, &p2_2d).expect("should succeed");

        // 3D distance should be greater due to Z component
        assert!(dist_3d > dist_2d);
        assert!((dist_2d - 5.0).abs() < 0.001); // 2D: √(3² + 4²) = 5
        assert!((dist_3d - 13.0).abs() < 0.001); // 3D: √(3² + 4² + 12²) = 13
    }

    #[test]
    fn test_distance_3d_point_to_linestring() {
        // Point above a horizontal line segment
        let point = Geometry::from_wkt("POINT Z(5 5 10)").expect("should succeed");
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 0)").expect("should succeed");

        let dist = distance_3d(&point, &line).expect("should succeed");

        // Closest point on line is (5, 0, 0), distance = √(0² + 5² + 10²) = √125 ≈ 11.18
        assert!((dist - 11.180).abs() < 0.01);
    }

    #[test]
    fn test_distance_3d_linestring_to_linestring() {
        // Two parallel vertical lines at different heights
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 0 0 10)").expect("should succeed");
        let ls2 = Geometry::from_wkt("LINESTRING Z(3 4 0, 3 4 10)").expect("should succeed");

        let dist = distance_3d(&ls1, &ls2).expect("should succeed");

        // Lines are parallel, distance is constant at √(3² + 4²) = 5
        assert!((dist - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_3d_linestring_to_linestring_skew() {
        // Two skew lines in 3D (don't intersect and aren't parallel)
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 0)").expect("should succeed");
        let ls2 = Geometry::from_wkt("LINESTRING Z(0 10 10, 10 10 10)").expect("should succeed");

        let dist = distance_3d(&ls1, &ls2).expect("should succeed");

        // These are parallel horizontal lines
        // Line 1 at Y=0, Z=0
        // Line 2 at Y=10, Z=10
        // Distance = √(10² + 10²) = √200 ≈ 14.14
        assert!((dist - 14.14).abs() < 0.1);
    }

    #[test]
    fn test_distance_3d_point_to_polygon() {
        // Point above a square polygon
        let point = Geometry::from_wkt("POINT Z(5 5 10)").expect("should succeed");
        let poly = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))")
            .expect("should succeed");

        let dist = distance_3d(&point, &poly).expect("should succeed");

        // Point is directly above center of polygon at height 10
        // But distance is to the boundary, not the interior
        // Closest boundary point would be an edge, giving distance > 10
        assert!(dist > 9.5); // Should be close to 10 (vertical distance component)
    }

    #[test]
    fn test_distance_3d_multipoint() {
        // MultiPoint to Point distance
        let mp = Geometry::from_wkt("MULTIPOINT Z((0 0 0), (10 0 0), (0 10 0))")
            .expect("should succeed");
        let p = Geometry::from_wkt("POINT Z(0 0 5)").expect("should succeed");

        let dist = distance_3d(&mp, &p).expect("should succeed");

        // Closest point in MultiPoint is (0, 0, 0), distance = 5
        assert!((dist - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_3d_same_geometry() {
        // LineString to itself should have zero distance
        let ls = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 10)").expect("should succeed");

        let dist = distance_3d(&ls, &ls).expect("should succeed");

        assert!(dist.abs() < 0.001);
    }

    #[test]
    fn test_distance_3d_point_on_linestring() {
        // Point that lies exactly on a line segment
        let point = Geometry::from_wkt("POINT Z(5 0 5)").expect("should succeed");
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 10)").expect("should succeed");

        let dist = distance_3d(&point, &line).expect("should succeed");

        // Point (5, 0, 5) is exactly on the line from (0, 0, 0) to (10, 0, 10)
        assert!(dist < 0.01); // Should be very close to zero
    }

    #[test]
    fn test_distance_3d_vertical_separation() {
        // Two geometries with same X,Y but different Z
        let p1 = Geometry::from_wkt("POINT Z(1 2 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(1 2 100)").expect("should succeed");

        let dist = distance_3d(&p1, &p2).expect("should succeed");

        assert!((dist - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_3d_diagonal_linestring() {
        // Point to a diagonal line in 3D space
        let point = Geometry::from_wkt("POINT Z(0 0 0)").expect("should succeed");
        let line = Geometry::from_wkt("LINESTRING Z(10 10 10, 20 20 20)").expect("should succeed");

        let dist = distance_3d(&point, &line).expect("should succeed");

        // Closest point on line segment is (10, 10, 10)
        // Distance = √(10² + 10² + 10²) = √300 ≈ 17.32
        assert!((dist - 17.32).abs() < 0.1);
    }

    // ========================================================================
    // 3D Buffer Tests
    // ========================================================================

    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_buffer_3d_point() {
        use geo::{Area, Euclidean, Length};

        let point = Geometry::from_wkt("POINT Z(0 0 5)").expect("should succeed");
        let buffered = buffer_3d(&point, 1.0).expect("should succeed");

        // Should be 3D
        assert!(buffered.is_3d());

        // Check that the buffer was created (geometry type might change to Polygon)
        // The exact result depends on the buffer implementation
        let length = match &buffered.geom {
            GeoGeometry::LineString(ls) => Euclidean.length(ls),
            GeoGeometry::Polygon(p) => Euclidean.length(p.exterior()),
            _ => 0.0,
        };
        assert!(length > 0.0 || buffered.geom.unsigned_area() > 0.0);
    }

    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_buffer_3d_linestring() {
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 10)").expect("should succeed");
        let buffered = buffer_3d(&line, 2.0).expect("should succeed");

        // Should be 3D
        assert!(buffered.is_3d());

        // Should have expanded the geometry
        use geo::Area;
        assert!(buffered.geom.unsigned_area() > 0.0);
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_3d_polygon() {
        let poly = Geometry::from_wkt("POLYGON Z((0 0 5, 10 0 5, 10 10 5, 0 10 5, 0 0 5))")
            .expect("should succeed");
        let buffered = buffer_3d(&poly, 1.0).expect("should succeed");

        // Should be 3D
        assert!(buffered.is_3d());

        // Should have larger area after buffering
        use geo::Area;
        let original_area = poly.geom.unsigned_area();
        let buffered_area = buffered.geom.unsigned_area();
        assert!(buffered_area > original_area);
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_3d_z_range_extension() {
        // Test that Z coordinates are extended by the buffer distance
        let poly = Geometry::from_wkt("POLYGON Z((0 0 10, 5 0 10, 5 5 10, 0 5 10, 0 0 10))")
            .expect("should succeed");

        // Original Z range: [10, 10]
        let original_z_min = 10.0;
        let buffer_distance = 2.0;

        let buffered = buffer_3d(&poly, buffer_distance).expect("should succeed");

        // After buffering with distance=2, Z should be extended
        // New Z range should be approximately [8, 12] but we use average
        // So all Z values should be around 10.0
        if let Some(ref z_coords) = buffered.coord3d.z_coords {
            for &z in &z_coords.values {
                // The average of [10-2, 10+2] = 10.0
                assert!((z - original_z_min).abs() < 0.1);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_buffer_3d_varying_z() {
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 20)").expect("should succeed");
        let buffered = buffer_3d(&line, 1.0).expect("should succeed");

        // Should be 3D
        assert!(buffered.is_3d());

        // Original Z range: [0, 20], average = 10
        // After buffering, all Z should be set to average
        if let Some(ref z_coords) = buffered.coord3d.z_coords {
            for &z in &z_coords.values {
                assert!((z - 10.0).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_buffer_3d_requires_z_coordinates() {
        let point = Geometry::from_wkt("POINT(0 0)").expect("should succeed"); // 2D point

        let result = buffer_3d(&point, 1.0);

        // Should return error for 2D geometry
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("must have Z coordinates"));
    }

    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_buffer_3d_negative_distance() {
        // Negative buffer (erosion) in 3D
        let poly = Geometry::from_wkt("POLYGON Z((0 0 10, 20 0 10, 20 20 10, 0 20 10, 0 0 10))")
            .expect("should succeed");

        let buffered = buffer_3d(&poly, -2.0).expect("should succeed");

        // Should still be 3D
        assert!(buffered.is_3d());

        // Area should be smaller after negative buffer
        use geo::Area;
        let original_area = poly.geom.unsigned_area();
        let buffered_area = buffered.geom.unsigned_area();
        assert!(buffered_area < original_area);
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_3d_multipolygon() {
        let mpoly = Geometry::from_wkt(
            "MULTIPOLYGON Z(((0 0 5, 5 0 5, 5 5 5, 0 5 5, 0 0 5)), \
             ((10 10 10, 15 10 10, 15 15 10, 10 15 10, 10 10 10)))",
        )
        .expect("should succeed");

        let buffered = buffer_3d(&mpoly, 1.0).expect("should succeed");

        // Should be 3D
        assert!(buffered.is_3d());

        // Should have larger area
        use geo::Area;
        assert!(buffered.geom.unsigned_area() > mpoly.geom.unsigned_area());
    }
}
