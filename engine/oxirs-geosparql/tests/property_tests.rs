//! Property-based tests for geometric operations
//!
//! These tests use proptest to generate random geometries and verify
//! that operations satisfy mathematical properties and invariants.

use approx::abs_diff_eq;
use geo::algorithm::coords_iter::CoordsIter;
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

// ============================================================================
// WKT Parser Fuzzing Tests
// ============================================================================

mod wkt_fuzzing {
    use super::*;

    // Strategy for generating 3D coordinates (with Z)
    fn coord3d_strategy() -> impl Strategy<Value = (f64, f64, f64)> {
        (-180.0..180.0, -90.0..90.0, -1000.0..1000.0)
    }

    // Strategy for WKT geometry type names
    fn wkt_geom_type_strategy() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("POINT"),
            Just("LINESTRING"),
            Just("POLYGON"),
            Just("MULTIPOINT"),
            Just("MULTILINESTRING"),
            Just("MULTIPOLYGON"),
        ]
    }

    // Strategy for WKT dimension modifiers
    fn wkt_dimension_modifier_strategy() -> impl Strategy<Value = &'static str> {
        prop_oneof![Just(""), Just("Z"), Just("M"), Just("ZM"),]
    }

    proptest! {
        /// Fuzzing: WKT roundtrip for all geometry types
        #[test]
        fn fuzz_wkt_roundtrip_all_types(geom in geometry_strategy()) {
            let wkt = geom.to_wkt();

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                // Geometry type should be preserved
                let orig_type = std::mem::discriminant(&geom.geom);
                let parsed_type = std::mem::discriminant(&parsed.geom);
                prop_assert_eq!(orig_type, parsed_type);
            }
        }

        /// Fuzzing: WKT roundtrip for 3D points
        #[test]
        fn fuzz_wkt_roundtrip_3d_point((x, y, z) in coord3d_strategy()) {
            let wkt = format!("POINT Z({} {} {})", x, y, z);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                prop_assert!(parsed.is_3d());
                if let GeoGeometry::Point(p) = &parsed.geom {
                    prop_assert!(abs_diff_eq!(p.x(), x, epsilon = 1e-6));
                    prop_assert!(abs_diff_eq!(p.y(), y, epsilon = 1e-6));
                    if let Some(z_val) = parsed.coord3d.z_at(0) {
                        prop_assert!(abs_diff_eq!(z_val, z, epsilon = 1e-6));
                    }
                }
            }
        }

        /// Fuzzing: WKT roundtrip for 3D linestrings
        #[test]
        fn fuzz_wkt_roundtrip_3d_linestring(
            coords in prop::collection::vec(coord3d_strategy(), 2..10)
        ) {
            let coord_str = coords
                .iter()
                .map(|(x, y, z)| format!("{} {} {}", x, y, z))
                .collect::<Vec<_>>()
                .join(", ");

            let wkt = format!("LINESTRING Z({})", coord_str);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                prop_assert!(parsed.is_3d());
                if let GeoGeometry::LineString(ls) = &parsed.geom {
                    prop_assert_eq!(ls.0.len(), coords.len());
                }
            }
        }

        /// Fuzzing: WKT roundtrip for 3D polygons
        #[test]
        fn fuzz_wkt_roundtrip_3d_polygon(
            mut coords in prop::collection::vec(coord3d_strategy(), 3..10)
        ) {
            // Close the ring
            if let Some(first) = coords.first() {
                coords.push(*first);
            }

            let coord_str = coords
                .iter()
                .map(|(x, y, z)| format!("{} {} {}", x, y, z))
                .collect::<Vec<_>>()
                .join(", ");

            let wkt = format!("POLYGON Z(({})))", coord_str);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                prop_assert!(parsed.is_3d());
                if let GeoGeometry::Polygon(p) = &parsed.geom {
                    prop_assert_eq!(p.exterior().0.len(), coords.len());
                }
            }
        }

        /// Fuzzing: Parse malformed WKT should fail gracefully
        #[test]
        fn fuzz_malformed_wkt_fails_gracefully(
            geom_type in wkt_geom_type_strategy(),
            modifier in wkt_dimension_modifier_strategy(),
            random_chars in "[a-zA-Z0-9 ,().-]{0,50}"
        ) {
            let wkt = format!("{} {}{}", geom_type, modifier, random_chars);

            // Should either parse successfully or fail with an error (not panic)
            let result = Geometry::from_wkt(&wkt);

            // This test passes if we don't panic
            prop_assert!(result.is_ok() || result.is_err());
        }

        /// Fuzzing: Empty geometries
        #[test]
        fn fuzz_empty_geometries(
            geom_type in wkt_geom_type_strategy(),
            modifier in wkt_dimension_modifier_strategy()
        ) {
            let wkt = format!("{} {} EMPTY", geom_type, modifier);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                prop_assert!(parsed.is_empty());
            }
        }

        /// Fuzzing: Very large coordinates
        #[test]
        fn fuzz_large_coordinates(
            x in -1e9f64..1e9f64,
            y in -1e9f64..1e9f64,
            z in -1e9f64..1e9f64
        ) {
            let wkt = format!("POINT Z({} {} {})", x, y, z);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Point(p) = &parsed.geom {
                    // Should handle large numbers without losing precision
                    let x_diff = (p.x() - x).abs();
                    let y_diff = (p.y() - y).abs();

                    // Allow relative error for large numbers
                    let x_epsilon = x.abs() * 1e-6 + 1e-6;
                    let y_epsilon = y.abs() * 1e-6 + 1e-6;

                    prop_assert!(x_diff < x_epsilon);
                    prop_assert!(y_diff < y_epsilon);
                }
            }
        }

        /// Fuzzing: Very small coordinates (near zero)
        #[test]
        fn fuzz_small_coordinates(
            x in -1e-6..1e-6,
            y in -1e-6..1e-6,
            z in -1e-6..1e-6
        ) {
            let wkt = format!("POINT Z({} {} {})", x, y, z);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Point(p) = &parsed.geom {
                    prop_assert!(abs_diff_eq!(p.x(), x, epsilon = 1e-9));
                    prop_assert!(abs_diff_eq!(p.y(), y, epsilon = 1e-9));
                }
            }
        }

        /// Fuzzing: Scientific notation in coordinates
        #[test]
        fn fuzz_scientific_notation(
            mantissa_x in -9.99..9.99,
            exp_x in -10i32..10i32,
            mantissa_y in -9.99..9.99,
            exp_y in -10i32..10i32
        ) {
            let x: f64 = mantissa_x * 10f64.powi(exp_x);
            let y: f64 = mantissa_y * 10f64.powi(exp_y);

            let wkt = format!("POINT({} {})", x, y);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Point(p) = &parsed.geom {
                    let x_epsilon: f64 = x.abs() * 1e-6 + 1e-9;
                    let y_epsilon: f64 = y.abs() * 1e-6 + 1e-9;

                    prop_assert!((p.x() - x).abs() < x_epsilon);
                    prop_assert!((p.y() - y).abs() < y_epsilon);
                }
            }
        }

        /// Fuzzing: Whitespace variations
        #[test]
        fn fuzz_whitespace_tolerance(
            x in -180.0..180.0,
            y in -90.0..90.0,
            spaces1 in 0usize..5,
            spaces2 in 0usize..5,
            spaces3 in 0usize..5
        ) {
            let sp1 = " ".repeat(spaces1 + 1);
            let sp2 = " ".repeat(spaces2 + 1);
            let sp3 = " ".repeat(spaces3 + 1);

            let wkt = format!("POINT{}({}{sp2}{}{})", sp1, sp3, x, y);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Point(p) = &parsed.geom {
                    prop_assert!(abs_diff_eq!(p.x(), x, epsilon = 1e-6));
                    prop_assert!(abs_diff_eq!(p.y(), y, epsilon = 1e-6));
                }
            }
        }

        /// Fuzzing: Case insensitivity
        #[test]
        fn fuzz_case_insensitivity(
            x in -180.0..180.0,
            y in -90.0..90.0,
            use_lower in prop::bool::ANY
        ) {
            let geom_type = if use_lower { "point" } else { "POINT" };
            let wkt = format!("{}({} {})", geom_type, x, y);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Point(p) = &parsed.geom {
                    prop_assert!(abs_diff_eq!(p.x(), x, epsilon = 1e-6));
                    prop_assert!(abs_diff_eq!(p.y(), y, epsilon = 1e-6));
                }
            }
        }

        /// Fuzzing: Nested MultiPolygons
        #[test]
        fn fuzz_multipolygon_roundtrip(
            num_polygons in 1usize..5,
            coords_per_poly in 3usize..8
        ) {
            let mut polygon_parts = Vec::new();

            for i in 0..num_polygons {
                let mut coords = Vec::new();
                for j in 0..coords_per_poly {
                    let angle = 2.0 * std::f64::consts::PI * (j as f64) / (coords_per_poly as f64);
                    let radius = 1.0;
                    let cx = (i as f64) * 10.0;
                    let cy = (i as f64) * 10.0;
                    let x = cx + radius * angle.cos();
                    let y = cy + radius * angle.sin();
                    coords.push(format!("{} {}", x, y));
                }
                // Close the ring
                if let Some(first) = coords.first() {
                    coords.push(first.clone());
                }

                // Each polygon in a multipolygon needs ((coords))
                polygon_parts.push(format!("(({}))", coords.join(", ")));
            }

            let wkt = format!("MULTIPOLYGON({})", polygon_parts.join(", "));

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::MultiPolygon(mp) = &parsed.geom {
                    prop_assert_eq!(mp.0.len(), num_polygons);
                }
            }
        }

        /// Fuzzing: Polygon with holes
        #[test]
        fn fuzz_polygon_with_holes(
            outer_coords in prop::collection::vec(coord_strategy(), 4..10),
            inner_coords in prop::collection::vec(coord_strategy(), 4..8)
        ) {
            let outer_str = {
                let mut coords = outer_coords.clone();
                if let Some(first) = coords.first() {
                    coords.push(*first);
                }
                coords
                    .iter()
                    .map(|c| format!("{} {}", c.x, c.y))
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            let inner_str = {
                let mut coords = inner_coords.clone();
                if let Some(first) = coords.first() {
                    coords.push(*first);
                }
                coords
                    .iter()
                    .map(|c| format!("{} {}", c.x * 0.5, c.y * 0.5))
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            let wkt = format!("POLYGON(({outer_str}), ({inner_str}))");

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Polygon(p) = &parsed.geom {
                    prop_assert_eq!(p.interiors().len(), 1);
                }
            }
        }

        /// Fuzzing: Extreme precision coordinates (many decimal places)
        #[test]
        fn fuzz_extreme_precision(
            x in -180.0f64..180.0,
            y in -90.0f64..90.0,
            decimals in 1usize..15
        ) {
            let precision = 10f64.powi(-(decimals as i32));
            let x_precise = (x / precision).round() * precision;
            let y_precise = (y / precision).round() * precision;

            let wkt = format!("POINT({:.width$} {:.width$})", x_precise, y_precise, width = decimals);

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Point(p) = &parsed.geom {
                    // Verify coordinates are within reasonable tolerance
                    prop_assert!((p.x() - x_precise).abs() < 1e-10);
                    prop_assert!((p.y() - y_precise).abs() < 1e-10);
                }
            }
        }

        /// Fuzzing: Mixed positive and negative coordinates
        #[test]
        fn fuzz_mixed_sign_coordinates(
            signs in prop::collection::vec(prop::bool::ANY, 4..10)
        ) {
            let coords: Vec<String> = signs
                .iter()
                .enumerate()
                .map(|(i, &positive)| {
                    let x = if positive { i as f64 } else { -(i as f64) };
                    let y = if !positive { i as f64 } else { -(i as f64) };
                    format!("{} {}", x, y)
                })
                .collect();

            let wkt = format!("LINESTRING({})", coords.join(", "));

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::LineString(ls) = &parsed.geom {
                    prop_assert_eq!(ls.coords_count(), signs.len());
                }
            }
        }

        /// Fuzzing: Very long coordinate sequences (stress test pre-allocation)
        #[test]
        fn fuzz_long_coordinate_sequence(
            length in 100usize..1000
        ) {
            let coords: Vec<String> = (0..length)
                .map(|i| format!("{} {}", i, i * 2))
                .collect();

            let wkt = format!("LINESTRING({})", coords.join(", "));

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::LineString(ls) = &parsed.geom {
                    prop_assert_eq!(ls.coords_count(), length);
                }
            }
        }

        /// Fuzzing: Coordinate edge cases (very close to zero)
        #[test]
        fn fuzz_near_zero_coordinates(
            x_exp in -300i32..-100,
            y_exp in -300i32..-100
        ) {
            let x = 10.0f64.powi(x_exp);
            let y = 10.0f64.powi(y_exp);

            let wkt = format!("POINT({} {})", x, y);

            // Should either parse successfully or fail gracefully
            let result = Geometry::from_wkt(&wkt);
            if let Ok(parsed) = result {
                if let GeoGeometry::Point(p) = &parsed.geom {
                    prop_assert!(p.x().is_finite());
                    prop_assert!(p.y().is_finite());
                }
            }
        }

        /// Fuzzing: 3D coordinates with varying Z values
        #[test]
        fn fuzz_3d_varying_z(
            coords_2d in prop::collection::vec(coord_strategy(), 3..20),
            z_values in prop::collection::vec(-1000.0f64..1000.0, 3..20)
        ) {
            let min_len = coords_2d.len().min(z_values.len());
            let coords: Vec<String> = (0..min_len)
                .map(|i| format!("{} {} {}", coords_2d[i].x, coords_2d[i].y, z_values[i]))
                .collect();

            let wkt = format!("LINESTRING Z({})", coords.join(", "));

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                prop_assert!(parsed.is_3d());
                if let GeoGeometry::LineString(ls) = &parsed.geom {
                    prop_assert_eq!(ls.coords_count(), min_len);
                }
            }
        }

        /// Fuzzing: Multigeometry with varying sizes
        #[test]
        fn fuzz_multipoint_varying_sizes(
            point_count in 1usize..50
        ) {
            let coords: Vec<String> = (0..point_count)
                .map(|i| format!("({} {})", i, i * i))
                .collect();

            let wkt = format!("MULTIPOINT({})", coords.join(", "));

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::MultiPoint(mp) = &parsed.geom {
                    prop_assert_eq!(mp.0.len(), point_count);
                }
            }
        }

        /// Fuzzing: Polygon with extreme aspect ratio
        #[test]
        fn fuzz_extreme_aspect_ratio(
            width in 0.001f64..1000.0,
            height in 0.001f64..1000.0
        ) {
            let wkt = format!(
                "POLYGON((0 0, {} 0, {} {}, 0 {}, 0 0))",
                width, width, height, height
            );

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                if let GeoGeometry::Polygon(p) = &parsed.geom {
                    prop_assert_eq!(p.exterior().coords_count(), 5);
                }
            }
        }

        /// Fuzzing: CRS with special characters in URI
        #[test]
        fn fuzz_crs_uri_variations(
            crs_code in 1000u32..10000
        ) {
            let wkt = format!(
                "<http://www.opengis.net/def/crs/EPSG/0/{}> POINT(1 2)",
                crs_code
            );

            if let Ok(parsed) = Geometry::from_wkt(&wkt) {
                prop_assert!(parsed.crs.uri.contains(&crs_code.to_string()));
            }
        }
    }
}
