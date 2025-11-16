//! WKT (Well-Known Text) parser and serializer
//!
//! Converts between WKT strings and geometry objects.
//!
//! # Performance Optimizations
//!
//! This module includes several zero-copy and performance optimizations:
//! - Lazy static regex compilation for CRS extraction
//! - Pre-allocated vectors with known capacity
//! - Iterator-based coordinate extraction to avoid intermediate allocations
//! - Reduced cloning through borrowing

use crate::error::{GeoSparqlError, Result};
use crate::geometry::coord3d::{Coord3D, CoordDim, MCoords, ZCoords};
use crate::geometry::{Crs, Geometry};
use geo_types::Geometry as GeoGeometry;
use once_cell::sync::Lazy;
use regex::Regex;
use std::str::FromStr;

/// Lazy static regex for CRS extraction (compiled once, reused forever)
static CRS_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^<([^>]+)>\s+(.+)$").expect("Invalid CRS regex pattern"));

/// Parse a WKT string into a Geometry
pub fn parse_wkt(wkt_str: &str) -> Result<Geometry> {
    // Extract CRS if present (e.g., "<http://...> POINT(1 2)")
    let (crs, wkt_geom) = extract_crs(wkt_str)?;

    // Use the wkt crate for parsing
    let wkt_parsed: wkt::Wkt<f64> =
        wkt::Wkt::from_str(wkt_geom).map_err(|e| GeoSparqlError::InvalidWkt(e.to_string()))?;

    // Extract Z/M coordinates before converting to geo_types
    let coord3d = extract_3d_coords(&wkt_parsed)?;

    // Convert to geo_types geometry using try_into
    let geo_geom: GeoGeometry<f64> = wkt_parsed
        .try_into()
        .map_err(|_| GeoSparqlError::InvalidWkt("Failed to convert WKT to geometry".to_string()))?;

    Ok(Geometry::with_crs_and_coord3d(geo_geom, crs, coord3d))
}

/// Extract 3D coordinates (Z/M) from parsed WKT
fn extract_3d_coords(wkt: &wkt::Wkt<f64>) -> Result<Coord3D> {
    use wkt::Wkt;

    match wkt {
        Wkt::Point(point) => extract_point_3d(point.coord()),
        Wkt::LineString(ls) => extract_linestring_3d(ls.coords()),
        Wkt::Polygon(poly) => extract_polygon_3d(poly),
        Wkt::MultiPoint(mp) => extract_multipoint_3d(mp.points()),
        Wkt::MultiLineString(mls) => extract_multilinestring_3d(mls.line_strings()),
        Wkt::MultiPolygon(mpoly) => extract_multipolygon_3d(mpoly.polygons()),
        Wkt::GeometryCollection(gc) => extract_geometrycollection_3d(gc.geometries()),
    }
}

/// Extract 3D coordinates from a Point
fn extract_point_3d(coord: Option<&wkt::types::Coord<f64>>) -> Result<Coord3D> {
    match coord {
        Some(c) => {
            let has_z = c.z.is_some();
            let has_m = c.m.is_some();

            let dim = match (has_z, has_m) {
                (false, false) => CoordDim::XY,
                (true, false) => CoordDim::XYZ,
                (false, true) => CoordDim::XYM,
                (true, true) => CoordDim::XYZM,
            };

            let z_coords = c.z.map(|z| ZCoords::new(vec![z]));
            let m_coords = c.m.map(|m| MCoords::new(vec![m]));

            Ok(Coord3D {
                dim,
                z_coords,
                m_coords,
            })
        }
        None => Ok(Coord3D::default()), // Empty point
    }
}

/// Extract 3D coordinates from a LineString
///
/// # Performance
///
/// Pre-allocates vectors with known capacity to avoid reallocations
fn extract_linestring_3d(coords: &[wkt::types::Coord<f64>]) -> Result<Coord3D> {
    if coords.is_empty() {
        return Ok(Coord3D::default());
    }

    let has_z = coords[0].z.is_some();
    let has_m = coords[0].m.is_some();

    let dim = match (has_z, has_m) {
        (false, false) => CoordDim::XY,
        (true, false) => CoordDim::XYZ,
        (false, true) => CoordDim::XYM,
        (true, true) => CoordDim::XYZM,
    };

    // Pre-allocate with known capacity to avoid reallocations
    let z_coords = if has_z {
        let mut z_values = Vec::with_capacity(coords.len());
        z_values.extend(coords.iter().map(|c| c.z.unwrap_or(0.0)));
        Some(ZCoords::new(z_values))
    } else {
        None
    };

    let m_coords = if has_m {
        let mut m_values = Vec::with_capacity(coords.len());
        m_values.extend(coords.iter().map(|c| c.m.unwrap_or(0.0)));
        Some(MCoords::new(m_values))
    } else {
        None
    };

    Ok(Coord3D {
        dim,
        z_coords,
        m_coords,
    })
}

/// Extract 3D coordinates from a Polygon
///
/// # Performance
///
/// Pre-calculates total coordinate count and pre-allocates vector
/// to avoid multiple reallocations during ring concatenation
fn extract_polygon_3d(poly: &wkt::types::Polygon<f64>) -> Result<Coord3D> {
    // Handle empty polygon
    if poly.rings().is_empty() {
        return Ok(Coord3D::default());
    }

    // Calculate total coordinate count to pre-allocate vector
    let total_coords: usize = poly.rings().iter().map(|ring| ring.coords().len()).sum();
    let mut all_coords = Vec::with_capacity(total_coords);

    // Add exterior ring
    all_coords.extend_from_slice(poly.rings()[0].coords());

    // Add interior rings
    for ring in &poly.rings()[1..] {
        all_coords.extend_from_slice(ring.coords());
    }

    extract_linestring_3d(&all_coords)
}

/// Extract 3D coordinates from a MultiPoint
///
/// # Performance
///
/// Pre-allocates vector and uses references to avoid cloning coordinates
fn extract_multipoint_3d(points: &[wkt::types::Point<f64>]) -> Result<Coord3D> {
    // Pre-allocate with maximum possible size (all points present)
    let mut coords = Vec::with_capacity(points.len());

    // Use explicit loop to benefit from pre-allocation
    for point in points {
        if let Some(coord) = point.coord() {
            coords.push(*coord);
        }
    }

    extract_linestring_3d(&coords)
}

/// Extract 3D coordinates from a MultiLineString
///
/// # Performance
///
/// Pre-calculates total coordinate count for optimal allocation
fn extract_multilinestring_3d(linestrings: &[wkt::types::LineString<f64>]) -> Result<Coord3D> {
    // Calculate total coordinate count to pre-allocate vector
    let total_coords: usize = linestrings.iter().map(|ls| ls.coords().len()).sum();
    let mut all_coords = Vec::with_capacity(total_coords);

    for ls in linestrings {
        all_coords.extend_from_slice(ls.coords());
    }

    extract_linestring_3d(&all_coords)
}

/// Extract 3D coordinates from a MultiPolygon
///
/// # Performance
///
/// Pre-calculates total coordinate count across all polygons and rings
fn extract_multipolygon_3d(polygons: &[wkt::types::Polygon<f64>]) -> Result<Coord3D> {
    // Calculate total coordinate count to pre-allocate vector
    let total_coords: usize = polygons
        .iter()
        .flat_map(|poly| poly.rings())
        .map(|ring| ring.coords().len())
        .sum();
    let mut all_coords = Vec::with_capacity(total_coords);

    for poly in polygons {
        // Add exterior ring
        all_coords.extend_from_slice(poly.rings()[0].coords());

        // Add interior rings
        for ring in &poly.rings()[1..] {
            all_coords.extend_from_slice(ring.coords());
        }
    }

    extract_linestring_3d(&all_coords)
}

/// Extract 3D coordinates from a GeometryCollection
fn extract_geometrycollection_3d(geometries: &[wkt::Wkt<f64>]) -> Result<Coord3D> {
    use wkt::Wkt;

    // For GeometryCollection, we take the dimension from the first non-empty geometry
    for geom in geometries {
        let coord3d = match geom {
            Wkt::Point(point) => extract_point_3d(point.coord())?,
            Wkt::LineString(ls) => extract_linestring_3d(ls.coords())?,
            Wkt::Polygon(poly) => extract_polygon_3d(poly)?,
            Wkt::MultiPoint(mp) => extract_multipoint_3d(mp.points())?,
            Wkt::MultiLineString(mls) => extract_multilinestring_3d(mls.line_strings())?,
            Wkt::MultiPolygon(mpoly) => extract_multipolygon_3d(mpoly.polygons())?,
            Wkt::GeometryCollection(gc) => extract_geometrycollection_3d(gc.geometries())?,
        };

        if coord3d.dim != CoordDim::XY {
            return Ok(coord3d);
        }
    }

    Ok(Coord3D::default())
}

/// Extract CRS from WKT string (zero-copy using string slices)
///
/// # Performance
///
/// Uses a lazy static regex compiled once at startup, avoiding regex
/// compilation overhead on every call. Returns string slices to avoid
/// allocations.
fn extract_crs(wkt: &str) -> Result<(Crs, &str)> {
    let trimmed = wkt.trim();

    if let Some(caps) = CRS_REGEX.captures(trimmed) {
        let crs_uri = caps.get(1).unwrap().as_str();
        let wkt_geom = caps.get(2).unwrap().as_str();
        Ok((Crs::new(crs_uri), wkt_geom))
    } else {
        Ok((Crs::default(), wkt))
    }
}

/// Convert a Geometry (with 3D coordinates) to WKT string
pub fn geometry_to_wkt_with_3d(geometry: &Geometry) -> String {
    geometry_to_wkt_internal(&geometry.geom, &geometry.coord3d)
}

/// Convert a geo_types Geometry to WKT string (2D only, for backwards compatibility)
pub fn geometry_to_wkt(geom: &GeoGeometry<f64>) -> String {
    geometry_to_wkt_internal(geom, &Coord3D::default())
}

/// Internal function to convert geometry with optional 3D coordinates
fn geometry_to_wkt_internal(geom: &GeoGeometry<f64>, coord3d: &Coord3D) -> String {
    let modifier = coord3d.dim.to_wkt_modifier();
    let modifier_str = modifier.map(|m| format!(" {}", m)).unwrap_or_default();

    match geom {
        GeoGeometry::Point(p) => {
            if p.x().is_nan() || p.y().is_nan() {
                format!("POINT{} EMPTY", modifier_str)
            } else {
                let mut coords = format!("{} {}", p.x(), p.y());
                if let Some(z) = coord3d.z_at(0) {
                    coords.push_str(&format!(" {}", z));
                }
                if let Some(m) = coord3d.m_at(0) {
                    coords.push_str(&format!(" {}", m));
                }
                format!("POINT{}({})", modifier_str, coords)
            }
        }
        GeoGeometry::Line(l) => {
            let mut start_coords = format!("{} {}", l.start.x, l.start.y);
            let mut end_coords = format!("{} {}", l.end.x, l.end.y);

            if let Some(z) = coord3d.z_at(0) {
                start_coords.push_str(&format!(" {}", z));
            }
            if let Some(m) = coord3d.m_at(0) {
                start_coords.push_str(&format!(" {}", m));
            }
            if let Some(z) = coord3d.z_at(1) {
                end_coords.push_str(&format!(" {}", z));
            }
            if let Some(m) = coord3d.m_at(1) {
                end_coords.push_str(&format!(" {}", m));
            }

            format!(
                "LINESTRING{}({}, {})",
                modifier_str, start_coords, end_coords
            )
        }
        GeoGeometry::LineString(ls) => {
            if ls.0.is_empty() {
                format!("LINESTRING{} EMPTY", modifier_str)
            } else {
                let coords: Vec<String> =
                    ls.0.iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let mut s = format!("{} {}", c.x, c.y);
                            if let Some(z) = coord3d.z_at(i) {
                                s.push_str(&format!(" {}", z));
                            }
                            if let Some(m) = coord3d.m_at(i) {
                                s.push_str(&format!(" {}", m));
                            }
                            s
                        })
                        .collect();
                format!("LINESTRING{}({})", modifier_str, coords.join(", "))
            }
        }
        GeoGeometry::Polygon(poly) => {
            if poly.exterior().0.is_empty() {
                "POLYGON EMPTY".to_string()
            } else {
                let exterior: Vec<String> = poly
                    .exterior()
                    .0
                    .iter()
                    .map(|c| format!("{} {}", c.x, c.y))
                    .collect();

                let mut rings = vec![format!("({})", exterior.join(", "))];

                for interior in poly.interiors() {
                    let interior_coords: Vec<String> = interior
                        .0
                        .iter()
                        .map(|c| format!("{} {}", c.x, c.y))
                        .collect();
                    rings.push(format!("({})", interior_coords.join(", ")));
                }

                format!("POLYGON({})", rings.join(", "))
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            if mp.0.is_empty() {
                "MULTIPOINT EMPTY".to_string()
            } else {
                let points: Vec<String> =
                    mp.0.iter()
                        .map(|p| format!("({} {})", p.x(), p.y()))
                        .collect();
                format!("MULTIPOINT({})", points.join(", "))
            }
        }
        GeoGeometry::MultiLineString(mls) => {
            if mls.0.is_empty() {
                "MULTILINESTRING EMPTY".to_string()
            } else {
                let line_strings: Vec<String> = mls
                    .0
                    .iter()
                    .map(|ls| {
                        let coords: Vec<String> =
                            ls.0.iter().map(|c| format!("{} {}", c.x, c.y)).collect();
                        format!("({})", coords.join(", "))
                    })
                    .collect();
                format!("MULTILINESTRING({})", line_strings.join(", "))
            }
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            if mpoly.0.is_empty() {
                "MULTIPOLYGON EMPTY".to_string()
            } else {
                let polygons: Vec<String> = mpoly
                    .0
                    .iter()
                    .map(|poly| {
                        let exterior: Vec<String> = poly
                            .exterior()
                            .0
                            .iter()
                            .map(|c| format!("{} {}", c.x, c.y))
                            .collect();

                        let mut rings = vec![format!("({})", exterior.join(", "))];

                        for interior in poly.interiors() {
                            let interior_coords: Vec<String> = interior
                                .0
                                .iter()
                                .map(|c| format!("{} {}", c.x, c.y))
                                .collect();
                            rings.push(format!("({})", interior_coords.join(", ")));
                        }

                        format!("({})", rings.join(", "))
                    })
                    .collect();
                format!("MULTIPOLYGON({})", polygons.join(", "))
            }
        }
        GeoGeometry::GeometryCollection(gc) => {
            if gc.0.is_empty() {
                "GEOMETRYCOLLECTION EMPTY".to_string()
            } else {
                let geometries: Vec<String> = gc.0.iter().map(geometry_to_wkt).collect();
                format!("GEOMETRYCOLLECTION({})", geometries.join(", "))
            }
        }
        GeoGeometry::Triangle(t) => {
            format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}))",
                t.0.x, t.0.y, t.1.x, t.1.y, t.2.x, t.2.y, t.0.x, t.0.y
            )
        }
        GeoGeometry::Rect(r) => {
            let min = r.min();
            let max = r.max();
            format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                min.x, min.y, max.x, min.y, max.x, max.y, min.x, max.y, min.x, min.y
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Point};

    #[test]
    fn test_parse_point() {
        let geom = parse_wkt("POINT(1.0 2.0)").unwrap();
        assert_eq!(geom.geometry_type(), "Point");

        match &geom.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point"),
        }
    }

    #[test]
    fn test_parse_linestring() {
        let geom = parse_wkt("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        assert_eq!(geom.geometry_type(), "LineString");

        match &geom.geom {
            GeoGeometry::LineString(ls) => {
                assert_eq!(ls.0.len(), 3);
                assert_eq!(ls.0[0].x, 0.0);
                assert_eq!(ls.0[2].y, 2.0);
            }
            _ => panic!("Expected LineString"),
        }
    }

    #[test]
    fn test_parse_polygon() {
        let geom = parse_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
        assert_eq!(geom.geometry_type(), "Polygon");

        match &geom.geom {
            GeoGeometry::Polygon(p) => {
                assert_eq!(p.exterior().0.len(), 5);
            }
            _ => panic!("Expected Polygon"),
        }
    }

    #[test]
    fn test_parse_with_crs() {
        let geom =
            parse_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1.0 2.0)").unwrap();
        assert_eq!(geom.crs.uri, "http://www.opengis.net/def/crs/EPSG/0/4326");
    }

    #[test]
    fn test_geometry_to_wkt_point() {
        let point = GeoGeometry::Point(Point::new(1.0, 2.0));
        let wkt = geometry_to_wkt(&point);
        assert_eq!(wkt, "POINT(1 2)");
    }

    #[test]
    fn test_geometry_to_wkt_linestring() {
        let ls = GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
        ]));
        let wkt = geometry_to_wkt(&ls);
        assert_eq!(wkt, "LINESTRING(0 0, 1 1)");
    }

    #[test]
    fn test_roundtrip() {
        let original_wkt = "POINT(1.5 2.5)";
        let geom = parse_wkt(original_wkt).unwrap();
        let new_wkt = geom.to_wkt();
        let geom2 = parse_wkt(&new_wkt).unwrap();

        match (&geom.geom, &geom2.geom) {
            (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) => {
                assert_eq!(p1.x(), p2.x());
                assert_eq!(p1.y(), p2.y());
            }
            _ => panic!("Expected Points"),
        }
    }

    // === 3D Geometry Tests ===

    #[test]
    fn test_parse_point_z() {
        let geom = parse_wkt("POINT Z(1.0 2.0 3.0)").unwrap();
        assert_eq!(geom.geometry_type(), "Point");
        assert!(geom.is_3d());
        assert!(!geom.is_measured());

        // Check Z coordinate
        assert_eq!(geom.coord3d.dim, CoordDim::XYZ);
        assert_eq!(geom.coord3d.z_at(0), Some(3.0));
        assert_eq!(geom.coord3d.m_at(0), None);
    }

    #[test]
    fn test_parse_point_m() {
        let geom = parse_wkt("POINT M(1.0 2.0 100.0)").unwrap();
        assert_eq!(geom.geometry_type(), "Point");
        assert!(!geom.is_3d());
        assert!(geom.is_measured());

        // Check M coordinate
        assert_eq!(geom.coord3d.dim, CoordDim::XYM);
        assert_eq!(geom.coord3d.z_at(0), None);
        assert_eq!(geom.coord3d.m_at(0), Some(100.0));
    }

    #[test]
    fn test_parse_point_zm() {
        let geom = parse_wkt("POINT ZM(1.0 2.0 3.0 100.0)").unwrap();
        assert_eq!(geom.geometry_type(), "Point");
        assert!(geom.is_3d());
        assert!(geom.is_measured());

        // Check Z and M coordinates
        assert_eq!(geom.coord3d.dim, CoordDim::XYZM);
        assert_eq!(geom.coord3d.z_at(0), Some(3.0));
        assert_eq!(geom.coord3d.m_at(0), Some(100.0));
    }

    #[test]
    fn test_parse_linestring_z() {
        let geom = parse_wkt("LINESTRING Z(0 0 10, 1 1 20, 2 2 30)").unwrap();
        assert_eq!(geom.geometry_type(), "LineString");
        assert!(geom.is_3d());

        // Check Z coordinates
        assert_eq!(geom.coord3d.dim, CoordDim::XYZ);
        assert_eq!(geom.coord3d.z_at(0), Some(10.0));
        assert_eq!(geom.coord3d.z_at(1), Some(20.0));
        assert_eq!(geom.coord3d.z_at(2), Some(30.0));
    }

    #[test]
    fn test_parse_polygon_z() {
        let geom = parse_wkt("POLYGON Z((0 0 10, 4 0 20, 4 4 30, 0 4 40, 0 0 10))").unwrap();
        assert_eq!(geom.geometry_type(), "Polygon");
        assert!(geom.is_3d());

        // Check Z coordinates
        assert_eq!(geom.coord3d.dim, CoordDim::XYZ);
        assert_eq!(geom.coord3d.z_at(0), Some(10.0));
        assert_eq!(geom.coord3d.z_at(1), Some(20.0));
        assert_eq!(geom.coord3d.z_at(2), Some(30.0));
    }

    #[test]
    fn test_serialize_point_z() {
        let geom = parse_wkt("POINT Z(1.0 2.0 3.0)").unwrap();
        let wkt = geom.to_wkt();

        // Should include Z modifier and Z coordinate
        assert!(wkt.contains(" Z"));
        assert!(wkt.contains("1 2 3"));
    }

    #[test]
    fn test_serialize_point_m() {
        let geom = parse_wkt("POINT M(1.0 2.0 100.0)").unwrap();
        let wkt = geom.to_wkt();

        // Should include M modifier and M coordinate
        assert!(wkt.contains(" M"));
        assert!(wkt.contains("1 2 100"));
    }

    #[test]
    fn test_serialize_point_zm() {
        let geom = parse_wkt("POINT ZM(1.0 2.0 3.0 100.0)").unwrap();
        let wkt = geom.to_wkt();

        // Should include ZM modifier and both coordinates
        assert!(wkt.contains(" ZM"));
        assert!(wkt.contains("1 2 3 100"));
    }

    #[test]
    fn test_roundtrip_3d_point_z() {
        let original_wkt = "POINT Z(1.5 2.5 3.5)";
        let geom = parse_wkt(original_wkt).unwrap();
        let new_wkt = geom.to_wkt();
        let geom2 = parse_wkt(&new_wkt).unwrap();

        assert_eq!(geom.coord3d.dim, geom2.coord3d.dim);
        assert_eq!(geom.coord3d.z_at(0), geom2.coord3d.z_at(0));
    }

    #[test]
    fn test_roundtrip_3d_linestring_z() {
        let original_wkt = "LINESTRING Z(0 0 10, 1 1 20, 2 2 30)";
        let geom = parse_wkt(original_wkt).unwrap();
        let new_wkt = geom.to_wkt();
        let geom2 = parse_wkt(&new_wkt).unwrap();

        assert_eq!(geom.coord3d.dim, geom2.coord3d.dim);
        assert_eq!(geom.coord3d.z_at(0), geom2.coord3d.z_at(0));
        assert_eq!(geom.coord3d.z_at(1), geom2.coord3d.z_at(1));
        assert_eq!(geom.coord3d.z_at(2), geom2.coord3d.z_at(2));
    }

    #[test]
    fn test_2d_geometry_remains_2d() {
        let geom = parse_wkt("POINT(1.0 2.0)").unwrap();
        assert_eq!(geom.coord3d.dim, CoordDim::XY);
        assert!(!geom.is_3d());
        assert!(!geom.is_measured());
    }
}
