//! DE-9IM (Dimensionally Extended 9-Intersection Model) functions
//!
//! Implements OGC GeoSPARQL 1.1 `geof:relate`, which tests whether two geometries
//! satisfy a given DE-9IM intersection-matrix pattern.
//!
//! # Pattern Format
//!
//! A pattern is exactly 9 characters in row-major order:
//!
//! ```text
//! | II IB IE |
//! | BI BB BE |
//! | EI EB EE |
//! ```
//!
//! Each position may be:
//! - `*` — any dimension (always matches)
//! - `T` — dimension ≥ 0 (non-empty intersection)
//! - `F` — empty intersection
//! - `0` — point intersection (0-dimensional)
//! - `1` — line intersection (1-dimensional)
//! - `2` — area intersection (2-dimensional)

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::algorithm::relate::Relate;
use geo::coordinate_position::CoordPos;
use geo::dimensions::Dimensions;
use geo_types::Geometry as GeoGeometry;

/// DE-9IM relation index for accessing the intersection matrix.
///
/// The 9 positions are indexed as:
/// `[II, IB, IE, BI, BB, BE, EI, EB, EE]`
const DE9IM_POSITIONS: [(CoordPos, CoordPos); 9] = [
    (CoordPos::Inside, CoordPos::Inside),
    (CoordPos::Inside, CoordPos::OnBoundary),
    (CoordPos::Inside, CoordPos::Outside),
    (CoordPos::OnBoundary, CoordPos::Inside),
    (CoordPos::OnBoundary, CoordPos::OnBoundary),
    (CoordPos::OnBoundary, CoordPos::Outside),
    (CoordPos::Outside, CoordPos::Inside),
    (CoordPos::Outside, CoordPos::OnBoundary),
    (CoordPos::Outside, CoordPos::Outside),
];

/// Match a single DE-9IM pattern character against a computed `Dimensions` value.
fn pattern_matches_dim(ch: char, dim: Dimensions) -> bool {
    match ch {
        '*' => true,
        'T' => dim != Dimensions::Empty,
        'F' => dim == Dimensions::Empty,
        '0' => dim == Dimensions::ZeroDimensional,
        '1' => dim == Dimensions::OneDimensional,
        '2' => dim == Dimensions::TwoDimensional,
        _ => false,
    }
}

/// Validate a DE-9IM pattern string.
///
/// Returns `Err` if the pattern is not exactly 9 characters or contains illegal characters.
fn validate_pattern(pattern: &str) -> Result<()> {
    let chars: Vec<char> = pattern.chars().collect();
    if chars.len() != 9 {
        return Err(GeoSparqlError::InvalidParameter(format!(
            "DE-9IM pattern must be exactly 9 characters, got {}",
            chars.len()
        )));
    }
    for (i, ch) in chars.iter().enumerate() {
        if !matches!(ch, '*' | 'T' | 'F' | '0' | '1' | '2') {
            return Err(GeoSparqlError::InvalidParameter(format!(
                "DE-9IM pattern character at position {} is '{}'; must be one of: * T F 0 1 2",
                i, ch
            )));
        }
    }
    Ok(())
}

/// Implement `geof:relate(geomA, geomB, pattern)`.
///
/// Returns `true` if the spatial relationship between `geom1` and `geom2`
/// matches the given DE-9IM `pattern`, `false` otherwise.
///
/// # Errors
///
/// - `GeoSparqlError::InvalidParameter` if `pattern` is not a valid 9-char DE-9IM pattern.
/// - `GeoSparqlError::CrsIncompatibility` if the two geometries have different CRS.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::functions::de9im::relate;
/// use oxirs_geosparql::geometry::Geometry;
///
/// let a = Geometry::from_wkt("POLYGON((0 0,10 0,10 10,0 10,0 0))").expect("valid");
/// let b = Geometry::from_wkt("POLYGON((5 5,15 5,15 15,5 15,5 5))").expect("valid");
///
/// // sfIntersects ↔ "T*T***T**" (or any non-FF*FF****)
/// assert!(relate(&a, &b, "T*T***T**").expect("ok"));
///
/// // sfDisjoint ↔ "FF*FF****"
/// assert!(!relate(&a, &b, "FF*FF****").expect("ok"));
/// ```
pub fn relate(geom1: &Geometry, geom2: &Geometry, pattern: &str) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;
    validate_pattern(pattern)?;

    let pattern_chars: Vec<char> = pattern.chars().collect();
    let matrix = geom1.geom.relate(&geom2.geom);

    for (idx, (pos_a, pos_b)) in DE9IM_POSITIONS.iter().enumerate() {
        let dim = matrix.get(*pos_a, *pos_b);
        if !pattern_matches_dim(pattern_chars[idx], dim) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Return the raw DE-9IM `IntersectionMatrix` for two geometries as a 9-char string.
///
/// Each character is one of `F`, `0`, `1`, `2` encoding the dimension:
/// - `F` → empty
/// - `0` → zero-dimensional (point)
/// - `1` → one-dimensional (line)
/// - `2` → two-dimensional (area)
///
/// The positions follow the standard row-major ordering: `II IB IE BI BB BE EI EB EE`.
pub fn relate_matrix_string(geom1: &Geometry, geom2: &Geometry) -> Result<String> {
    geom1.validate_crs_compatibility(geom2)?;
    let matrix = geom1.geom.relate(&geom2.geom);
    let mut s = String::with_capacity(9);
    for (pos_a, pos_b) in &DE9IM_POSITIONS {
        let dim = matrix.get(*pos_a, *pos_b);
        s.push(match dim {
            Dimensions::Empty => 'F',
            Dimensions::ZeroDimensional => '0',
            Dimensions::OneDimensional => '1',
            Dimensions::TwoDimensional => '2',
        });
    }
    Ok(s)
}

/// Produce a geometry-level `boundary` in pure Rust, implementing OGC SFA §6.1.6.1.
///
/// Rules:
/// - `Point` / `MultiPoint` → empty `GeometryCollection`
/// - `LineString` (not closed) → `MultiPoint` of {start, end}
/// - `LineString` (closed, i.e. ring) → empty `GeometryCollection`
/// - `MultiLineString` → `MultiPoint` of endpoints with odd multiplicity (Mod-2 rule)
/// - `Polygon` → `MultiLineString` of exterior + all interior rings
/// - `MultiPolygon` → `MultiLineString` of all rings of all polygons
/// - `GeometryCollection` → empty (undefined by OGC SFA)
/// - `Line` / `Triangle` / `Rect` → treated as LineString / Polygon analogously
pub fn boundary(geom: &Geometry) -> Result<Geometry> {
    use geo_types::{GeometryCollection, LineString, MultiLineString, MultiPoint, Point};
    use std::collections::HashMap;

    let result_geom: GeoGeometry<f64> = match &geom.geom {
        GeoGeometry::Point(_) | GeoGeometry::MultiPoint(_) => {
            GeoGeometry::GeometryCollection(GeometryCollection(vec![]))
        }

        GeoGeometry::Line(l) => {
            // A Line has two distinct endpoints (if equal it would be degenerate)
            let start = Point::new(l.start.x, l.start.y);
            let end = Point::new(l.end.x, l.end.y);
            if (l.start.x - l.end.x).abs() < f64::EPSILON
                && (l.start.y - l.end.y).abs() < f64::EPSILON
            {
                // Closed line — empty boundary
                GeoGeometry::GeometryCollection(GeometryCollection(vec![]))
            } else {
                GeoGeometry::MultiPoint(MultiPoint(vec![start, end]))
            }
        }

        GeoGeometry::LineString(ls) => {
            if ls.0.is_empty() {
                GeoGeometry::GeometryCollection(GeometryCollection(vec![]))
            } else {
                let first = ls.0.first().copied();
                let last = ls.0.last().copied();
                match (first, last) {
                    (Some(f), Some(l))
                        if (f.x - l.x).abs() < f64::EPSILON && (f.y - l.y).abs() < f64::EPSILON =>
                    {
                        // Closed ring — boundary is empty
                        GeoGeometry::GeometryCollection(GeometryCollection(vec![]))
                    }
                    (Some(f), Some(l)) => {
                        let start = Point::new(f.x, f.y);
                        let end = Point::new(l.x, l.y);
                        GeoGeometry::MultiPoint(MultiPoint(vec![start, end]))
                    }
                    _ => GeoGeometry::GeometryCollection(GeometryCollection(vec![])),
                }
            }
        }

        GeoGeometry::MultiLineString(mls) => {
            // Mod-2 rule: a point is on the boundary iff it appears an odd number
            // of times as an endpoint across all component LineStrings.
            let mut endpoint_count: HashMap<(i64, i64), (Point<f64>, usize)> = HashMap::new();
            for ls in &mls.0 {
                if ls.0.is_empty() {
                    continue;
                }
                let endpoints: [Option<geo_types::Coord<f64>>; 2] =
                    [ls.0.first().copied(), ls.0.last().copied()];
                for c in endpoints.into_iter().flatten() {
                    // Use a quantised key to handle floating-point equality
                    let key = ((c.x * 1e9).round() as i64, (c.y * 1e9).round() as i64);
                    let entry = endpoint_count
                        .entry(key)
                        .or_insert_with(|| (Point::new(c.x, c.y), 0));
                    entry.1 += 1;
                }
            }
            let boundary_pts: Vec<Point<f64>> = endpoint_count
                .into_values()
                .filter_map(|(pt, count)| if count % 2 == 1 { Some(pt) } else { None })
                .collect();
            GeoGeometry::MultiPoint(MultiPoint(boundary_pts))
        }

        GeoGeometry::Polygon(p) => {
            let mut rings: Vec<LineString<f64>> = Vec::new();
            rings.push(p.exterior().clone());
            for interior in p.interiors() {
                rings.push(interior.clone());
            }
            GeoGeometry::MultiLineString(MultiLineString(rings))
        }

        GeoGeometry::MultiPolygon(mp) => {
            let mut rings: Vec<LineString<f64>> = Vec::new();
            for poly in &mp.0 {
                rings.push(poly.exterior().clone());
                for interior in poly.interiors() {
                    rings.push(interior.clone());
                }
            }
            GeoGeometry::MultiLineString(MultiLineString(rings))
        }

        GeoGeometry::Triangle(t) => {
            // Triangle boundary: its three edges as a LineString ring
            use geo_types::Coord;
            let ring = LineString(vec![
                Coord {
                    x: t.v1().x,
                    y: t.v1().y,
                },
                Coord {
                    x: t.v2().x,
                    y: t.v2().y,
                },
                Coord {
                    x: t.v3().x,
                    y: t.v3().y,
                },
                Coord {
                    x: t.v1().x,
                    y: t.v1().y,
                },
            ]);
            GeoGeometry::MultiLineString(MultiLineString(vec![ring]))
        }

        GeoGeometry::Rect(r) => {
            // Rect boundary: its exterior ring
            use geo_types::Coord;
            let min = r.min();
            let max = r.max();
            let ring = LineString(vec![
                Coord { x: min.x, y: min.y },
                Coord { x: max.x, y: min.y },
                Coord { x: max.x, y: max.y },
                Coord { x: min.x, y: max.y },
                Coord { x: min.x, y: min.y },
            ]);
            GeoGeometry::MultiLineString(MultiLineString(vec![ring]))
        }

        GeoGeometry::GeometryCollection(_) => {
            // OGC SFA: boundary of a geometry collection is undefined
            GeoGeometry::GeometryCollection(GeometryCollection(vec![]))
        }
    };

    Ok(Geometry::with_crs(result_geom, geom.crs.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, LineString, Point, Polygon};

    fn make_polygon(coords: &[(f64, f64)]) -> Geometry {
        use geo_types::Coord;
        let ring: Vec<Coord<f64>> = coords.iter().map(|&(x, y)| Coord { x, y }).collect();
        Geometry::new(GeoGeometry::Polygon(Polygon::new(LineString(ring), vec![])))
    }

    fn make_linestring(coords: &[(f64, f64)]) -> Geometry {
        use geo_types::Coord;
        let cs: Vec<Coord<f64>> = coords.iter().map(|&(x, y)| Coord { x, y }).collect();
        Geometry::new(GeoGeometry::LineString(LineString(cs)))
    }

    // ── validate_pattern ──────────────────────────────────────────────────────

    #[test]
    fn test_validate_pattern_ok() {
        assert!(validate_pattern("T*T***T**").is_ok());
        assert!(validate_pattern("FF*FF****").is_ok());
        assert!(validate_pattern("T*F**F***").is_ok());
        assert!(validate_pattern("012012012").is_ok());
    }

    #[test]
    fn test_validate_pattern_wrong_length() {
        assert!(validate_pattern("T*T").is_err());
        assert!(validate_pattern("T*T***T**X").is_err());
        assert!(validate_pattern("").is_err());
    }

    #[test]
    fn test_validate_pattern_bad_char() {
        assert!(validate_pattern("T*T***T*X").is_err());
        assert!(validate_pattern("T*T***T*3").is_err());
    }

    // ── relate (core) ────────────────────────────────────────────────────────

    #[test]
    fn test_relate_sf_disjoint_pattern() {
        // Two non-overlapping squares
        let a = make_polygon(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
        let b = make_polygon(&[
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
            (10.0, 10.0),
        ]);

        // sfDisjoint ↔ "FF*FF****"
        let disjoint = relate(&a, &b, "FF*FF****").expect("should succeed");
        assert!(disjoint, "disjoint squares should match FF*FF****");
    }

    #[test]
    fn test_relate_sf_intersects_pattern() {
        // Overlapping squares
        let a = make_polygon(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let b = make_polygon(&[
            (5.0, 5.0),
            (15.0, 5.0),
            (15.0, 15.0),
            (5.0, 15.0),
            (5.0, 5.0),
        ]);

        // sfIntersects ↔ NOT FF*FF**** → use T*T***T**
        let intersects = relate(&a, &b, "T*T***T**").expect("should succeed");
        assert!(intersects, "overlapping squares should match T*T***T**");
    }

    #[test]
    fn test_relate_sf_within_pattern() {
        // Small polygon inside larger polygon
        let outer = make_polygon(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let inner = make_polygon(&[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (2.0, 2.0)]);

        // sfWithin ↔ "T*F**F***"
        let within = relate(&inner, &outer, "T*F**F***").expect("should succeed");
        assert!(within, "inner polygon should be within outer polygon");

        // inner does NOT match within pattern against itself shifted
        let not_within = relate(&outer, &inner, "T*F**F***").expect("should succeed");
        assert!(
            !not_within,
            "outer polygon should not be within inner polygon"
        );
    }

    #[test]
    fn test_relate_sf_contains_pattern() {
        let outer = make_polygon(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let inner = make_polygon(&[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (2.0, 2.0)]);

        // sfContains ↔ "T*****FF*"
        let contains = relate(&outer, &inner, "T*****FF*").expect("should succeed");
        assert!(contains, "outer polygon should contain inner polygon");
    }

    #[test]
    fn test_relate_wildcard_matches_all() {
        let a = make_polygon(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
        let b = make_polygon(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
        let result = relate(&a, &b, "*********").expect("should succeed");
        assert!(result, "all-wildcard pattern should always match");
    }

    #[test]
    fn test_relate_returns_error_for_bad_pattern() {
        let a = make_polygon(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
        let b = make_polygon(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
        assert!(relate(&a, &b, "bad").is_err());
        assert!(relate(&a, &b, "T*T***T*X").is_err());
    }

    // ── relate_matrix_string ─────────────────────────────────────────────────

    #[test]
    fn test_relate_matrix_string_length() {
        let a = make_polygon(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let b = make_polygon(&[
            (5.0, 5.0),
            (15.0, 5.0),
            (15.0, 15.0),
            (5.0, 15.0),
            (5.0, 5.0),
        ]);
        let s = relate_matrix_string(&a, &b).expect("should succeed");
        assert_eq!(s.len(), 9);
        for ch in s.chars() {
            assert!(matches!(ch, 'F' | '0' | '1' | '2'));
        }
    }

    #[test]
    fn test_relate_matrix_string_disjoint() {
        let a = make_polygon(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
        let b = make_polygon(&[
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
            (10.0, 10.0),
        ]);
        let s = relate_matrix_string(&a, &b).expect("should succeed");
        // II must be F (interiors don't intersect)
        assert_eq!(
            &s[0..1],
            "F",
            "II dimension should be F for disjoint polygons"
        );
    }

    // ── boundary ─────────────────────────────────────────────────────────────

    #[test]
    fn test_boundary_point_is_empty() {
        let pt = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let b = boundary(&pt).expect("should succeed");
        if let GeoGeometry::GeometryCollection(gc) = b.geom {
            assert!(gc.0.is_empty(), "boundary of point should be empty");
        } else {
            panic!("boundary of point should be GeometryCollection");
        }
    }

    #[test]
    fn test_boundary_linestring_open() {
        let ls = make_linestring(&[(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]);
        let b = boundary(&ls).expect("should succeed");
        if let GeoGeometry::MultiPoint(mp) = b.geom {
            assert_eq!(
                mp.0.len(),
                2,
                "open LineString boundary should have 2 points"
            );
        } else {
            panic!("boundary of open LineString should be MultiPoint");
        }
    }

    #[test]
    fn test_boundary_linestring_closed_ring() {
        // Closed ring: start == end
        let ls = make_linestring(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
        let b = boundary(&ls).expect("should succeed");
        if let GeoGeometry::GeometryCollection(gc) = b.geom {
            assert!(
                gc.0.is_empty(),
                "closed LineString boundary should be empty"
            );
        } else {
            panic!("boundary of closed LineString should be empty GeometryCollection");
        }
    }

    #[test]
    fn test_boundary_polygon() {
        let poly = make_polygon(&[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let b = boundary(&poly).expect("should succeed");
        if let GeoGeometry::MultiLineString(mls) = b.geom {
            assert_eq!(mls.0.len(), 1, "simple polygon boundary should have 1 ring");
        } else {
            panic!("boundary of polygon should be MultiLineString");
        }
    }

    #[test]
    fn test_boundary_multilinestring_mod2() {
        use geo_types::{Coord, MultiLineString};
        // Two line strings sharing an endpoint at (5,5)
        // (0,0)-(5,5): endpoint (5,5) appears once
        // (5,5)-(10,0): endpoint (5,5) appears once → total 2 times → NOT on boundary (even)
        // So boundary = {(0,0), (10,0)} only
        let mls = GeoGeometry::MultiLineString(MultiLineString(vec![
            LineString(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 5.0, y: 5.0 }]),
            LineString(vec![Coord { x: 5.0, y: 5.0 }, Coord { x: 10.0, y: 0.0 }]),
        ]));
        let geom = Geometry::new(mls);
        let b = boundary(&geom).expect("should succeed");
        if let GeoGeometry::MultiPoint(mp) = b.geom {
            assert_eq!(
                mp.0.len(),
                2,
                "Mod-2 rule: shared interior endpoint not on boundary"
            );
        } else {
            panic!("boundary of MultiLineString should be MultiPoint");
        }
    }
}
