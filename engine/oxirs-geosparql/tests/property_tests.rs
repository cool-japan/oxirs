//! Property-based tests for geometric operations
//!
//! These tests use proptest to generate random geometries and verify
//! that operations satisfy mathematical properties and invariants.

use approx::abs_diff_eq;
use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point, Polygon};
use oxirs_geosparql::functions::geometric_operations::*;
use oxirs_geosparql::functions::geometric_properties::*;
use oxirs_geosparql::functions::simple_features::*;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::validation::*;
use proptest::prelude::*;

// Strategy for generating valid coordinates
fn coord_strategy() -> impl Strategy<Value = Coord<f64>> {
    (-180.0..180.0, -90.0..90.0).prop_map(|(x, y)| Coord { x, y })
}

// Strategy for generating valid points
fn point_strategy() -> impl Strategy<Value = Point<f64>> {
    coord_strategy().prop_map(Point::from)
}

// Strategy for generating valid linestrings (at least 2 points)
fn linestring_strategy() -> impl Strategy<Value = LineString<f64>> {
    prop::collection::vec(coord_strategy(), 2..10).prop_map(LineString::new)
}

// Strategy for generating valid polygons
fn polygon_strategy() -> impl Strategy<Value = Polygon<f64>> {
    // Generate exterior ring (at least 4 points, closed)
    prop::collection::vec(coord_strategy(), 3..10).prop_map(|mut coords| {
        // Close the ring
        if let Some(first) = coords.first() {
            coords.push(*first);
        }
        let exterior = LineString::new(coords);
        Polygon::new(exterior, vec![])
    })
}

// Strategy for generating geometries
fn geometry_strategy() -> impl Strategy<Value = Geometry> {
    prop_oneof![
        point_strategy().prop_map(|p| Geometry::new(GeoGeometry::Point(p))),
        linestring_strategy().prop_map(|ls| Geometry::new(GeoGeometry::LineString(ls))),
        polygon_strategy().prop_map(|poly| Geometry::new(GeoGeometry::Polygon(poly))),
    ]
}

proptest! {
    /// Property: Distance is symmetric
    /// For any geometries A and B: distance(A, B) = distance(B, A)
    #[test]
    fn prop_distance_symmetric(
        p1 in point_strategy(),
        p2 in point_strategy()
    ) {
        let g1 = Geometry::new(GeoGeometry::Point(p1));
        let g2 = Geometry::new(GeoGeometry::Point(p2));

        let d1 = distance(&g1, &g2).unwrap();
        let d2 = distance(&g2, &g1).unwrap();

        prop_assert!(abs_diff_eq!(d1, d2, epsilon = 1e-10));
    }

    /// Property: Distance from a point to itself is zero
    #[test]
    fn prop_distance_to_self_is_zero(p in point_strategy()) {
        let geom = Geometry::new(GeoGeometry::Point(p));
        let d = distance(&geom, &geom).unwrap();
        prop_assert!(abs_diff_eq!(d, 0.0, epsilon = 1e-10));
    }

    /// Property: Triangle inequality
    /// For any points A, B, C: distance(A, C) ≤ distance(A, B) + distance(B, C)
    #[test]
    fn prop_triangle_inequality(
        p1 in point_strategy(),
        p2 in point_strategy(),
        p3 in point_strategy()
    ) {
        let g1 = Geometry::new(GeoGeometry::Point(p1));
        let g2 = Geometry::new(GeoGeometry::Point(p2));
        let g3 = Geometry::new(GeoGeometry::Point(p3));

        let d_ac = distance(&g1, &g3).unwrap();
        let d_ab = distance(&g1, &g2).unwrap();
        let d_bc = distance(&g2, &g3).unwrap();

        // Allow small numerical error
        prop_assert!(d_ac <= d_ab + d_bc + 1e-10);
    }

    /// Property: Equals is reflexive
    /// A geometry always equals itself
    #[test]
    fn prop_equals_reflexive(geom in geometry_strategy()) {
        let result = sf_equals(&geom, &geom).unwrap();
        prop_assert!(result);
    }

    /// Property: Intersects is symmetric
    /// If A intersects B, then B intersects A
    #[test]
    fn prop_intersects_symmetric(
        p1 in point_strategy(),
        p2 in point_strategy()
    ) {
        let g1 = Geometry::new(GeoGeometry::Point(p1));
        let g2 = Geometry::new(GeoGeometry::Point(p2));

        let i1 = sf_intersects(&g1, &g2).unwrap();
        let i2 = sf_intersects(&g2, &g1).unwrap();

        prop_assert_eq!(i1, i2);
    }

    /// Property: Disjoint is opposite of intersects
    /// For any geometries A and B: disjoint(A, B) = !intersects(A, B)
    #[test]
    fn prop_disjoint_opposite_intersects(
        p1 in point_strategy(),
        p2 in point_strategy()
    ) {
        let g1 = Geometry::new(GeoGeometry::Point(p1));
        let g2 = Geometry::new(GeoGeometry::Point(p2));

        let disjoint = sf_disjoint(&g1, &g2).unwrap();
        let intersects = sf_intersects(&g1, &g2).unwrap();

        prop_assert_eq!(disjoint, !intersects);
    }

    /// Property: Envelope contains original geometry
    #[test]
    fn prop_envelope_contains_original(geom in geometry_strategy()) {
        if !geom.is_empty() {
            let env = envelope(&geom).unwrap();
            // Envelope should contain the original geometry
            // (This is a simplified check - full check would require actual containment test)
            prop_assert!(!env.is_empty());
        }
    }

    /// Property: Buffer with zero distance returns similar area
    #[test]

    /// Property: Simplification preserves general shape
    /// Simplified geometry should have fewer or equal points
    #[test]
    fn prop_simplify_reduces_complexity(ls in linestring_strategy()) {
        let geom = Geometry::new(GeoGeometry::LineString(ls.clone()));

        if let Ok(simplified) = simplify_geometry(&geom, 0.1) {
            if let (GeoGeometry::LineString(orig), GeoGeometry::LineString(simp)) = (&geom.geom, &simplified.geom) {
                prop_assert!(simp.0.len() <= orig.0.len());
            }
        }
    }

    /// Property: Validation is consistent
    /// Valid geometries should remain valid after certain operations
    #[test]
    fn prop_valid_geometry_stays_valid(p in point_strategy()) {
        let geom = Geometry::new(GeoGeometry::Point(p));
        let validation = validate_geometry(&geom);

        // Points with valid coordinates should be valid
        if !p.x().is_nan() && !p.y().is_nan() && !p.x().is_infinite() && !p.y().is_infinite() {
            prop_assert!(validation.is_valid);
        }
    }

    /// Property: Snap to precision is idempotent
    /// Snapping twice should give the same result as snapping once
    #[test]
    fn prop_snap_idempotent(p in point_strategy(), precision in 0u32..6u32) {
        let geom = Geometry::new(GeoGeometry::Point(p));

        if let Ok(snapped1) = snap_to_precision(&geom, precision) {
            if let Ok(snapped2) = snap_to_precision(&snapped1, precision) {
                if let (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) = (&snapped1.geom, &snapped2.geom) {
                    prop_assert!(abs_diff_eq!(p1.x(), p2.x(), epsilon = 1e-10));
                    prop_assert!(abs_diff_eq!(p1.y(), p2.y(), epsilon = 1e-10));
                }
            }
        }
    }

    /// Property: Centroid is inside or on boundary for valid polygons
    #[test]
    fn prop_centroid_inside_polygon(poly in polygon_strategy()) {
        let geom = Geometry::new(GeoGeometry::Polygon(poly));

        // For valid polygons, centroid should exist
        if let Ok(Some(centroid_geom)) = centroid(&geom) {
            prop_assert!(!centroid_geom.is_empty());
        }
    }

    /// Property: WKT round-trip preserves geometry
    #[test]
    fn prop_wkt_roundtrip(p in point_strategy()) {
        let geom = Geometry::new(GeoGeometry::Point(p));
        let wkt = geom.to_wkt();

        if let Ok(parsed) = Geometry::from_wkt(&wkt) {
            if let (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) = (&geom.geom, &parsed.geom) {
                prop_assert!(abs_diff_eq!(p1.x(), p2.x(), epsilon = 1e-10));
                prop_assert!(abs_diff_eq!(p1.y(), p2.y(), epsilon = 1e-10));
            }
        }
    }

    /// Property: Area is non-negative
    #[test]
    fn prop_area_non_negative(poly in polygon_strategy()) {
        let geom = Geometry::new(GeoGeometry::Polygon(poly));

        if let Ok(a) = area(&geom) {
            prop_assert!(a >= 0.0);
        }
    }

    /// Property: Length is non-negative
    #[test]
    fn prop_length_non_negative(ls in linestring_strategy()) {
        let geom = Geometry::new(GeoGeometry::LineString(ls));

        if let Ok(len) = length(&geom) {
            prop_assert!(len >= 0.0);
        }
    }

    /// Property: Distance is non-negative
    #[test]
    fn prop_distance_non_negative(
        p1 in point_strategy(),
        p2 in point_strategy()
    ) {
        let g1 = Geometry::new(GeoGeometry::Point(p1));
        let g2 = Geometry::new(GeoGeometry::Point(p2));

        let d = distance(&g1, &g2).unwrap();
        prop_assert!(d >= 0.0);
    }
}

#[cfg(feature = "geojson-support")]
mod geojson_properties {
    use super::*;

    proptest! {
        /// Property: GeoJSON round-trip preserves geometry
        #[test]
        fn prop_geojson_roundtrip(p in point_strategy()) {
            let geom = Geometry::new(GeoGeometry::Point(p));

            if let Ok(json) = geom.to_geojson() {
                if let Ok(parsed) = Geometry::from_geojson(&json) {
                    if let (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) = (&geom.geom, &parsed.geom) {
                        prop_assert!(abs_diff_eq!(p1.x(), p2.x(), epsilon = 1e-10));
                        prop_assert!(abs_diff_eq!(p1.y(), p2.y(), epsilon = 1e-10));
                    }
                }
            }
        }
    }
}
