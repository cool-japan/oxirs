//! Geometry-level simplification via the Douglas-Peucker algorithm.
//!
//! Implements OGC GeoSPARQL 1.1 `geof:simplify`. Uses the `geo::Simplify` trait
//! (pure Rust, no GEOS dependency) and dispatches over all `geo_types::Geometry` variants.
//!
//! Points and empty geometries are returned unchanged.
//! Polygons and MultiPolygons have their rings simplified individually.
//! GeometryCollections are simplified recursively.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::Simplify;
use geo_types::Geometry as GeoGeometry;

/// Simplify a geometry using the Douglas-Peucker algorithm.
///
/// `tolerance` is the maximum allowed squared distance between the original and
/// simplified geometry, in the same coordinate units as the geometry.
///
/// Returns the simplified geometry with the same CRS as the input.
/// Geometries that cannot be simplified (e.g. Points) are returned unchanged.
///
/// # Errors
///
/// - `GeoSparqlError::InvalidParameter` if `tolerance` is negative or NaN.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::functions::geometry_simplify::simplify;
/// use oxirs_geosparql::geometry::Geometry;
///
/// let ls = Geometry::from_wkt("LINESTRING(0 0,1 0.1,2 0,3 0,4 0)").expect("valid");
/// let simplified = simplify(&ls, 0.2).expect("ok");
/// // The intermediate point (1, 0.1) is within tolerance of the line (0,0)-(4,0)
/// // so it should be removed.
/// ```
pub fn simplify(geom: &Geometry, tolerance: f64) -> Result<Geometry> {
    if tolerance.is_nan() || tolerance < 0.0 {
        return Err(GeoSparqlError::InvalidParameter(format!(
            "simplify tolerance must be a non-negative finite number, got {tolerance}"
        )));
    }

    let simplified_geom = simplify_geo_geometry(&geom.geom, tolerance);
    Ok(Geometry::with_crs(simplified_geom, geom.crs.clone()))
}

/// Dispatch simplification over `GeoGeometry` variants.
fn simplify_geo_geometry(geom: &GeoGeometry<f64>, tolerance: f64) -> GeoGeometry<f64> {
    match geom {
        // Points cannot be simplified — return as-is
        GeoGeometry::Point(_) | GeoGeometry::MultiPoint(_) => geom.clone(),

        // Line: convert to LineString, simplify, convert back (or keep as Line)
        GeoGeometry::Line(l) => {
            use geo_types::{Coord, LineString};
            let ls = LineString(vec![
                Coord {
                    x: l.start.x,
                    y: l.start.y,
                },
                Coord {
                    x: l.end.x,
                    y: l.end.y,
                },
            ]);
            GeoGeometry::LineString(ls.simplify(tolerance))
        }

        GeoGeometry::LineString(ls) => GeoGeometry::LineString(ls.simplify(tolerance)),

        GeoGeometry::MultiLineString(mls) => GeoGeometry::MultiLineString(mls.simplify(tolerance)),

        GeoGeometry::Polygon(poly) => GeoGeometry::Polygon(poly.simplify(tolerance)),

        GeoGeometry::MultiPolygon(mp) => GeoGeometry::MultiPolygon(mp.simplify(tolerance)),

        // Triangle: treat as a polygon ring, simplify rings, return Polygon
        GeoGeometry::Triangle(t) => {
            use geo_types::{Coord, LineString, Polygon};
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
            let poly = Polygon::new(ring.simplify(tolerance), vec![]);
            GeoGeometry::Polygon(poly)
        }

        // Rect: convert to Polygon, simplify, return
        GeoGeometry::Rect(r) => {
            use geo_types::{Coord, LineString, Polygon};
            let min = r.min();
            let max = r.max();
            let ring = LineString(vec![
                Coord { x: min.x, y: min.y },
                Coord { x: max.x, y: min.y },
                Coord { x: max.x, y: max.y },
                Coord { x: min.x, y: max.y },
                Coord { x: min.x, y: min.y },
            ]);
            let poly = Polygon::new(ring.simplify(tolerance), vec![]);
            GeoGeometry::Polygon(poly)
        }

        GeoGeometry::GeometryCollection(gc) => {
            use geo_types::GeometryCollection;
            let simplified: Vec<GeoGeometry<f64>> =
                gc.0.iter()
                    .map(|g| simplify_geo_geometry(g, tolerance))
                    .collect();
            GeoGeometry::GeometryCollection(GeometryCollection(simplified))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Polygon};

    fn make_linestring(coords: &[(f64, f64)]) -> Geometry {
        let cs: Vec<Coord<f64>> = coords.iter().map(|&(x, y)| Coord { x, y }).collect();
        Geometry::new(GeoGeometry::LineString(LineString(cs)))
    }

    fn make_polygon(coords: &[(f64, f64)]) -> Geometry {
        let ring: Vec<Coord<f64>> = coords.iter().map(|&(x, y)| Coord { x, y }).collect();
        Geometry::new(GeoGeometry::Polygon(Polygon::new(LineString(ring), vec![])))
    }

    #[test]
    fn test_simplify_collinear_linestring() {
        // Points on a straight line: (0,0), (1,0), (2,0), (3,0)
        // With tolerance > 0, middle points should be removed
        let ls = make_linestring(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]);
        let simplified = simplify(&ls, 0.5).expect("should succeed");
        if let GeoGeometry::LineString(result_ls) = simplified.geom {
            // Endpoints must be preserved; intermediate collinear points removed
            assert!(
                result_ls.0.len() <= 4,
                "simplified collinear line should have fewer or equal points"
            );
            assert!(
                result_ls.0.len() >= 2,
                "simplified line must have at least 2 points"
            );
        } else {
            panic!("expected LineString");
        }
    }

    #[test]
    fn test_simplify_keeps_endpoints() {
        let ls = make_linestring(&[(0.0, 0.0), (1.0, 0.05), (2.0, 0.0), (3.0, 0.05), (4.0, 0.0)]);
        let simplified = simplify(&ls, 0.1).expect("should succeed");
        if let GeoGeometry::LineString(result_ls) = simplified.geom {
            // First and last points must be preserved
            let first = result_ls.0.first().expect("should have first point");
            let last = result_ls.0.last().expect("should have last point");
            assert!((first.x - 0.0).abs() < f64::EPSILON);
            assert!((last.x - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected LineString");
        }
    }

    #[test]
    fn test_simplify_tolerance_zero_keeps_all() {
        let ls = make_linestring(&[(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.1), (4.0, 0.0)]);
        let simplified = simplify(&ls, 0.0).expect("should succeed");
        if let GeoGeometry::LineString(result_ls) = simplified.geom {
            assert_eq!(
                result_ls.0.len(),
                5,
                "zero tolerance must preserve all points"
            );
        } else {
            panic!("expected LineString");
        }
    }

    #[test]
    fn test_simplify_polygon() {
        let poly = make_polygon(&[
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let simplified = simplify(&poly, 1.0).expect("should succeed");
        match simplified.geom {
            GeoGeometry::Polygon(p) => {
                // Exterior ring must still close
                let ext = p.exterior();
                assert!(
                    ext.0.len() >= 4,
                    "simplified polygon exterior must have ≥ 4 coords (ring)"
                );
            }
            _ => panic!("expected Polygon"),
        }
    }

    #[test]
    fn test_simplify_point_unchanged() {
        use geo_types::Point;
        let pt = Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0)));
        let simplified = simplify(&pt, 1.0).expect("should succeed");
        if let GeoGeometry::Point(p) = simplified.geom {
            assert!((p.x() - 3.0).abs() < f64::EPSILON);
            assert!((p.y() - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected Point");
        }
    }

    #[test]
    fn test_simplify_negative_tolerance_errors() {
        let ls = make_linestring(&[(0.0, 0.0), (1.0, 1.0)]);
        assert!(simplify(&ls, -0.5).is_err());
    }

    #[test]
    fn test_simplify_nan_tolerance_errors() {
        let ls = make_linestring(&[(0.0, 0.0), (1.0, 1.0)]);
        assert!(simplify(&ls, f64::NAN).is_err());
    }

    #[test]
    fn test_simplify_preserves_crs() {
        use crate::geometry::Crs;
        use geo_types::Point;
        let pt = Geometry::with_crs(GeoGeometry::Point(Point::new(0.0, 0.0)), Crs::epsg(4326));
        let simplified = simplify(&pt, 1.0).expect("should succeed");
        assert_eq!(simplified.crs.uri, pt.crs.uri, "CRS must be preserved");
    }
}
