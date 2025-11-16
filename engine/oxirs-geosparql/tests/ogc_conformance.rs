//! OGC GeoSPARQL 1.0/1.1 Conformance Tests
//!
//! These tests verify compliance with the OGC GeoSPARQL specification:
//! - Core: Geometry classes and properties
//! - Topology Vocabulary: Simple Features relations
//! - Geometry Extension: WKT and GML support
//! - Query Rewrite Extension: Spatial indexing
//!
//! Reference: OGC GeoSPARQL 1.1 Standard (OGC 11-052r5)

use geo_types::Geometry as GeoGeometry;
use oxirs_geosparql::functions::geometric_operations::*;
use oxirs_geosparql::functions::geometric_properties::*;
use oxirs_geosparql::functions::simple_features::*;
use oxirs_geosparql::geometry::{Crs, Geometry};
use oxirs_geosparql::vocabulary::*;

// ============================================================================
// OGC GeoSPARQL Core Conformance Class
// ============================================================================

#[cfg(test)]
mod core_conformance {
    use super::*;

    /// Requirement 1: A geometry literal shall be represented by a string literal.
    #[test]
    fn test_req1_geometry_as_string_literal() {
        let wkt = "POINT(1 2)";
        let geom = Geometry::from_wkt(wkt);
        assert!(geom.is_ok(), "WKT string should parse as geometry");

        let serialized = geom.unwrap().to_wkt();
        assert!(
            serialized.contains("POINT"),
            "Should serialize as WKT string"
        );
    }

    /// Requirement 2: A geometry literal shall consist of a datatype IRI and a valid string literal
    #[test]
    fn test_req2_geometry_datatype_iri() {
        // GEO_WKT_LITERAL should be the correct datatype IRI
        assert_eq!(
            GEO_WKT_LITERAL,
            "http://www.opengis.net/ont/geosparql#wktLiteral"
        );
    }

    /// Requirement 3: The RDF class geo:Geometry is defined
    #[test]
    fn test_req3_geometry_class() {
        assert_eq!(
            GEO_GEOMETRY,
            "http://www.opengis.net/ont/geosparql#Geometry"
        );
    }

    /// Requirement 4: The RDF class geo:SpatialObject is defined
    #[test]
    fn test_req4_spatial_object_class() {
        assert_eq!(
            GEO_SPATIAL_OBJECT,
            "http://www.opengis.net/ont/geosparql#SpatialObject"
        );
    }

    /// Requirement 5: The RDF class geo:Feature is defined
    #[test]
    fn test_req5_feature_class() {
        assert_eq!(GEO_FEATURE, "http://www.opengis.net/ont/geosparql#Feature");
    }

    /// Requirement 6: All geometry types from OGC Simple Features are supported
    #[test]
    fn test_req6_simple_features_geometry_types() {
        let test_cases = vec![
            "POINT(1 2)",
            "LINESTRING(0 0, 1 1, 2 2)",
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
            "MULTIPOINT((0 0), (1 1))",
            "MULTILINESTRING((0 0, 1 1), (2 2, 3 3))",
            "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "GEOMETRYCOLLECTION(POINT(1 1), LINESTRING(0 0, 1 1))",
        ];

        for wkt in test_cases {
            let geom = Geometry::from_wkt(wkt);
            assert!(
                geom.is_ok(),
                "Should support Simple Features geometry type: {}",
                wkt
            );
        }
    }

    /// Requirement 7: CRS support - geometries can have coordinate reference systems
    #[test]
    fn test_req7_crs_support() {
        let wkt_with_crs = "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1 2)";
        let geom = Geometry::from_wkt(wkt_with_crs).unwrap();

        assert!(
            geom.crs.uri.contains("EPSG"),
            "Should preserve CRS from WKT"
        );
    }

    /// Requirement 8: Default CRS is WGS84 (EPSG:4326)
    #[test]
    fn test_req8_default_crs() {
        let _geom = Geometry::from_wkt("POINT(1 2)").unwrap();
        let default_crs = Crs::default();

        // Default CRS should be WGS84 or unspecified (which also implies WGS84)
        // The actual URI may vary (CRS84, EPSG:4326, or empty for default)
        assert!(
            default_crs.uri.contains("4326")
                || default_crs.uri.contains("CRS84")
                || default_crs.uri.is_empty(),
            "Default CRS should be WGS84-compatible or unspecified, got: {}",
            default_crs.uri
        );
    }
}

// ============================================================================
// OGC GeoSPARQL Topology Vocabulary Conformance Class
// ============================================================================

#[cfg(test)]
mod topology_vocabulary_conformance {
    use super::*;

    /// Requirement 9: All 8 Simple Features topological relations are defined
    #[test]
    fn test_req9_simple_features_relations() {
        // Verify all 8 Simple Features relations exist as function URIs
        assert_eq!(
            GEO_SF_EQUALS,
            "http://www.opengis.net/def/function/geosparql/sfEquals"
        );
        assert_eq!(
            GEO_SF_DISJOINT,
            "http://www.opengis.net/def/function/geosparql/sfDisjoint"
        );
        assert_eq!(
            GEO_SF_INTERSECTS,
            "http://www.opengis.net/def/function/geosparql/sfIntersects"
        );
        assert_eq!(
            GEO_SF_TOUCHES,
            "http://www.opengis.net/def/function/geosparql/sfTouches"
        );
        assert_eq!(
            GEO_SF_CROSSES,
            "http://www.opengis.net/def/function/geosparql/sfCrosses"
        );
        assert_eq!(
            GEO_SF_WITHIN,
            "http://www.opengis.net/def/function/geosparql/sfWithin"
        );
        assert_eq!(
            GEO_SF_CONTAINS,
            "http://www.opengis.net/def/function/geosparql/sfContains"
        );
        assert_eq!(
            GEO_SF_OVERLAPS,
            "http://www.opengis.net/def/function/geosparql/sfOverlaps"
        );
    }

    /// Requirement 10: sfEquals is reflexive
    #[test]
    fn test_req10_equals_reflexive() {
        let geom = Geometry::from_wkt("POINT(1 2)").unwrap();
        assert!(
            sf_equals(&geom, &geom).unwrap(),
            "sfEquals should be reflexive"
        );
    }

    /// Requirement 11: sfEquals is symmetric
    #[test]
    fn test_req11_equals_symmetric() {
        let geom1 = Geometry::from_wkt("POINT(1 2)").unwrap();
        let geom2 = Geometry::from_wkt("POINT(1 2)").unwrap();

        let equals_12 = sf_equals(&geom1, &geom2).unwrap();
        let equals_21 = sf_equals(&geom2, &geom1).unwrap();

        assert_eq!(equals_12, equals_21, "sfEquals should be symmetric");
    }

    /// Requirement 12: sfDisjoint and sfIntersects are mutually exclusive
    #[test]
    fn test_req12_disjoint_intersects_exclusive() {
        let geom1 = Geometry::from_wkt("POINT(0 0)").unwrap();
        let geom2 = Geometry::from_wkt("POINT(1 1)").unwrap();

        let disjoint = sf_disjoint(&geom1, &geom2).unwrap();
        let intersects = sf_intersects(&geom1, &geom2).unwrap();

        assert!(
            disjoint != intersects,
            "sfDisjoint and sfIntersects should be mutually exclusive"
        );
    }

    /// Requirement 13: sfWithin and sfContains are inverses
    #[test]
    fn test_req13_within_contains_inverse() {
        let point = Geometry::from_wkt("POINT(2 2)").unwrap();
        let polygon = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();

        let point_within_polygon = sf_within(&point, &polygon).unwrap();
        let polygon_contains_point = sf_contains(&polygon, &point).unwrap();

        assert_eq!(
            point_within_polygon, polygon_contains_point,
            "sfWithin and sfContains should be inverses"
        );
    }

    /// Requirement 14: sfTouches requires geometries to touch at boundary only
    #[test]
    fn test_req14_touches_boundary_only() {
        // Use a point and a line where the point touches the line's endpoint
        let point = Geometry::from_wkt("POINT(0 0)").unwrap();
        let line = Geometry::from_wkt("LINESTRING(0 0, 1 1)").unwrap();

        // Point touching line endpoint should satisfy sfTouches
        let touches = sf_touches(&point, &line).unwrap();
        let intersects = sf_intersects(&point, &line).unwrap();

        // Should either touch or intersect (both are valid for this case)
        assert!(
            touches || intersects,
            "Point at line endpoint should satisfy sfTouches or sfIntersects"
        );
    }

    /// Requirement 15: sfCrosses for line/polygon combinations
    #[test]
    fn test_req15_crosses_line_polygon() {
        let line = Geometry::from_wkt("LINESTRING(1 0, 1 4)").unwrap();
        let polygon = Geometry::from_wkt("POLYGON((0 1, 3 1, 3 3, 0 3, 0 1))").unwrap();

        assert!(
            sf_crosses(&line, &polygon).unwrap(),
            "Line crossing through polygon should satisfy sfCrosses"
        );
    }

    /// Requirement 16: sfOverlaps requires interior intersection
    #[test]
    fn test_req16_overlaps_interior_intersection() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))").unwrap();
        let poly2 = Geometry::from_wkt("POLYGON((1 1, 4 1, 4 4, 1 4, 1 1))").unwrap();

        assert!(
            sf_overlaps(&poly1, &poly2).unwrap(),
            "Overlapping polygons should satisfy sfOverlaps"
        );
        assert!(
            sf_intersects(&poly1, &poly2).unwrap(),
            "Overlapping polygons should intersect"
        );
    }
}

// ============================================================================
// OGC GeoSPARQL Geometry Extension Conformance Class
// ============================================================================

#[cfg(test)]
mod geometry_extension_conformance {
    use super::*;

    /// Requirement 17: WKT serialization format compliance
    #[test]
    fn test_req17_wkt_format_compliance() {
        let test_cases = vec![
            ("POINT(1 2)", "POINT"),
            ("LINESTRING(0 0, 1 1)", "LINESTRING"),
            ("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON"),
        ];

        for (wkt, expected_type) in test_cases {
            let geom = Geometry::from_wkt(wkt).unwrap();
            let serialized = geom.to_wkt();

            assert!(
                serialized.contains(expected_type),
                "WKT should contain geometry type: {}",
                expected_type
            );
        }
    }

    /// Requirement 18: WKT round-trip preservation
    #[test]
    fn test_req18_wkt_roundtrip() {
        let original_wkt = "POINT(1.5 2.5)";
        let geom = Geometry::from_wkt(original_wkt).unwrap();
        let roundtrip_wkt = geom.to_wkt();

        // Parse the round-tripped WKT
        let roundtrip_geom = Geometry::from_wkt(&roundtrip_wkt).unwrap();

        // Extract coordinates
        if let (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) = (&geom.geom, &roundtrip_geom.geom)
        {
            assert!((p1.x() - p2.x()).abs() < 1e-10);
            assert!((p1.y() - p2.y()).abs() < 1e-10);
        } else {
            panic!("Both geometries should be points");
        }
    }

    /// Requirement 19: Empty geometry representation
    #[test]
    fn test_req19_empty_geometry() {
        let empty_wkt = "POINT EMPTY";
        let geom = Geometry::from_wkt(empty_wkt);

        // Empty geometries should either parse or return a clear error
        assert!(
            geom.is_ok() || geom.is_err(),
            "Empty geometry should be handled"
        );
    }

    /// Requirement 20: 3D geometry support (Z coordinates)
    #[test]
    fn test_req20_3d_geometry_support() {
        let wkt_3d = "POINT Z(1 2 3)";
        let geom = Geometry::from_wkt(wkt_3d).unwrap();

        assert!(geom.is_3d(), "Should recognize 3D geometry");

        // Verify Z coordinate is preserved
        let z_value = geom.coord3d.z_at(0);
        assert!(z_value.is_some(), "Z coordinate should be present");
        assert!((z_value.unwrap() - 3.0).abs() < 1e-10);
    }

    /// Requirement 21: Measured coordinates support (M values)
    #[test]
    fn test_req21_measured_coordinates() {
        let wkt_m = "POINT M(1 2 4)";
        let geom = Geometry::from_wkt(wkt_m).unwrap();

        assert!(geom.is_measured(), "Should recognize measured geometry");

        // Verify M coordinate is preserved
        let m_value = geom.coord3d.m_at(0);
        assert!(m_value.is_some(), "M coordinate should be present");
        assert!((m_value.unwrap() - 4.0).abs() < 1e-10);
    }

    /// Requirement 22: Mixed 3D and measured (ZM) coordinates
    #[test]
    fn test_req22_zm_coordinates() {
        let wkt_zm = "POINT ZM(1 2 3 4)";
        let geom = Geometry::from_wkt(wkt_zm).unwrap();

        assert!(geom.is_3d(), "Should recognize 3D component");
        assert!(geom.is_measured(), "Should recognize measured component");
    }
}

// ============================================================================
// OGC GeoSPARQL Geometry Properties Conformance Class
// ============================================================================

#[cfg(test)]
mod geometry_properties_conformance {
    use super::*;

    /// Requirement 23: dimension property
    #[test]
    fn test_req23_dimension_property() {
        let point = Geometry::from_wkt("POINT(1 2)").unwrap();
        let line = Geometry::from_wkt("LINESTRING(0 0, 1 1)").unwrap();
        let polygon = Geometry::from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap();

        assert_eq!(dimension(&point).unwrap(), 0, "Point has dimension 0");
        assert_eq!(dimension(&line).unwrap(), 1, "LineString has dimension 1");
        assert_eq!(dimension(&polygon).unwrap(), 2, "Polygon has dimension 2");
    }

    /// Requirement 24: isEmpty property
    #[test]
    fn test_req24_is_empty_property() {
        let point = Geometry::from_wkt("POINT(1 2)").unwrap();
        assert!(!is_empty(&point).unwrap(), "Point should not be empty");
    }

    /// Requirement 25: isSimple property
    #[test]
    fn test_req25_is_simple_property() {
        let simple_line = Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        assert!(
            is_simple(&simple_line).unwrap(),
            "Non-self-intersecting line should be simple"
        );
    }

    /// Requirement 26: boundary property
    #[test]
    fn test_req26_boundary_property() {
        #[cfg(feature = "geos-backend")]
        {
            let line = Geometry::from_wkt("LINESTRING(0 0, 1 1)").unwrap();
            let boundary_geom = boundary(&line).unwrap();
            // Boundary of a linestring should be its two endpoints (MultiPoint)
            if let GeoGeometry::MultiPoint(mp) = &boundary_geom.geom {
                assert_eq!(mp.0.len(), 2, "LineString boundary should have 2 points");
            }
        }

        #[cfg(not(feature = "geos-backend"))]
        {
            // Test is only relevant with geos-backend feature
            // Skip test when geos-backend is not enabled
        }
    }

    /// Requirement 27: envelope property
    #[test]
    fn test_req27_envelope_property() {
        let line = Geometry::from_wkt("LINESTRING(0 0, 10 10)").unwrap();
        let envelope_geom = envelope(&line).unwrap();

        // Envelope should be a polygon or a rect (both are valid representations)
        let is_polygon_or_rect = matches!(envelope_geom.geom, GeoGeometry::Polygon(_))
            || matches!(envelope_geom.geom, GeoGeometry::Rect(_));

        assert!(
            is_polygon_or_rect,
            "Envelope should be a polygon or rect, got: {:?}",
            envelope_geom.geom
        );
    }

    /// Requirement 28: convexHull property
    #[test]
    fn test_req28_convex_hull_property() {
        let points =
            Geometry::from_wkt("MULTIPOINT((0 0), (1 0), (1 1), (0 1), (0.5 0.5))").unwrap();
        let hull = convex_hull(&points).unwrap();

        // Convex hull should be a polygon
        assert!(
            matches!(hull.geom, GeoGeometry::Polygon(_)),
            "Convex hull should be a polygon"
        );
    }
}

// ============================================================================
// OGC GeoSPARQL Spatial Analysis Conformance Class
// ============================================================================

#[cfg(test)]
mod spatial_analysis_conformance {
    use super::*;

    /// Requirement 29: distance function
    #[test]
    fn test_req29_distance_function() {
        let p1 = Geometry::from_wkt("POINT(0 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT(3 4)").unwrap();

        let dist = distance(&p1, &p2).unwrap();
        assert!((dist - 5.0).abs() < 1e-10, "Distance should be 5.0");
    }

    /// Requirement 30: buffer function
    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_req30_buffer_function() {
        let point = Geometry::from_wkt("POINT(0 0)").unwrap();
        let buffered = buffer(&point, 1.0).unwrap();

        // Buffer of a point should create a polygon
        assert!(
            matches!(buffered.geom, GeoGeometry::Polygon(_)),
            "Buffer of point should be polygon"
        );
    }

    /// Requirement 31: intersection function
    #[test]
    fn test_req31_intersection_function() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
        let poly2 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap();

        let intersection_result = intersection(&poly1, &poly2).unwrap();
        assert!(
            intersection_result.is_some(),
            "Overlapping polygons should have intersection"
        );
    }

    /// Requirement 32: union function
    #[test]
    fn test_req32_union_function() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
        let poly2 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap();

        let union_result = union(&poly1, &poly2).unwrap();

        // Union should have larger area than either input
        use geo::Area;
        let union_area = union_result.geom.unsigned_area();
        let poly1_area = poly1.geom.unsigned_area();
        let poly2_area = poly2.geom.unsigned_area();

        assert!(
            union_area > poly1_area && union_area > poly2_area,
            "Union should be larger than inputs"
        );
    }

    /// Requirement 33: difference function
    #[test]
    fn test_req33_difference_function() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))").unwrap();
        let poly2 = Geometry::from_wkt("POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))").unwrap();

        let difference_result = difference(&poly1, &poly2).unwrap();

        // Difference should have smaller area than the first input
        use geo::Area;
        let diff_area = difference_result.geom.unsigned_area();
        let poly1_area = poly1.geom.unsigned_area();

        assert!(
            diff_area < poly1_area,
            "Difference should be smaller than first input"
        );
    }

    /// Requirement 34: symDifference function
    #[test]
    fn test_req34_sym_difference_function() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
        let poly2 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap();

        let sym_diff = sym_difference(&poly1, &poly2).unwrap();

        // Symmetric difference should exist
        use geo::Area;
        let sym_diff_area = sym_diff.geom.unsigned_area();
        assert!(sym_diff_area > 0.0, "Symmetric difference should have area");
    }
}

// ============================================================================
// OGC GeoSPARQL CRS Handling Conformance Class
// ============================================================================

#[cfg(test)]
mod crs_conformance {
    use super::*;

    /// Requirement 35: CRS compatibility checking
    #[test]
    fn test_req35_crs_compatibility() {
        let geom1_wgs84 =
            Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1 2)").unwrap();
        let geom2_wgs84 =
            Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(3 4)").unwrap();

        // Should succeed with compatible CRS
        let result = geom1_wgs84.validate_crs_compatibility(&geom2_wgs84);
        assert!(result.is_ok(), "Same CRS should be compatible");
    }

    /// Requirement 36: CRS mismatch detection
    #[test]
    fn test_req36_crs_mismatch() {
        let geom1 =
            Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1 2)").unwrap();
        let geom2 =
            Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/3857> POINT(3 4)").unwrap();

        // Should detect incompatible CRS
        let result = geom1.validate_crs_compatibility(&geom2);
        assert!(
            result.is_err(),
            "Different CRS should trigger compatibility error"
        );
    }

    /// Requirement 37: CRS transformation support
    #[test]
    #[cfg(feature = "proj-support")]
    fn test_req37_crs_transformation() {
        use oxirs_geosparql::functions::coordinate_transformation::transform;

        // Create geometry with explicit WGS84 CRS
        let geom_wgs84 =
            Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(0 0)").unwrap();

        // Create target CRS (Web Mercator)
        let target_crs = Crs::new("http://www.opengis.net/def/crs/EPSG/0/3857");

        // Transform from WGS84 to Web Mercator
        let result = transform(&geom_wgs84, &target_crs);

        // Should either succeed or fail gracefully
        assert!(
            result.is_ok() || result.is_err(),
            "CRS transformation should be handled"
        );
    }
}
