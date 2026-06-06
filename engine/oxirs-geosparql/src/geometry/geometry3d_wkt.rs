//! WKT parsing and serialization helpers for 3D geometry.
//!
//! This module contains the parsing functions that convert ISO/OGC WKT
//! strings (with optional `Z`/`ZM`/`M` modifiers) into [`Geometry3DEnum`]
//! values, the serialization helpers that turn coordinate sequences back
//! into WKT, and the Z-range utilities shared across the 3D geometry types.

use super::geometry3d_types::{Geometry3DEnum, LineString3D, LinearRing3D, Point3D, Polygon3D};
use crate::error::{GeoSparqlError, Result};

// ---------------------------------------------------------------------------
// WKT parsing helpers
// ---------------------------------------------------------------------------

/// Parse a 3D WKT string into a [`Geometry3DEnum`].
pub(crate) fn parse_wkt_3d(wkt: &str) -> Result<Geometry3DEnum> {
    // Strip optional SRID prefix: "SRID=4326;"
    let wkt = if let Some(_rest) = wkt.to_uppercase().strip_prefix("SRID=") {
        // find the semicolon
        if let Some(pos) = wkt.find(';') {
            wkt[pos + 1..].trim()
        } else {
            wkt
        }
    } else {
        wkt
    };

    let upper = wkt.to_uppercase();

    if upper.starts_with("POINT") {
        let inner = extract_inner(wkt, "POINT")?;
        let p = parse_point3d_coords(inner.trim())?;
        Ok(Geometry3DEnum::Point(p))
    } else if upper.starts_with("LINESTRING") {
        let inner = extract_inner(wkt, "LINESTRING")?;
        let pts = parse_coord_list_3d(inner.trim())?;
        Ok(Geometry3DEnum::LineString(LineString3D::new(pts)))
    } else if upper.starts_with("POLYGON") {
        let inner = extract_inner(wkt, "POLYGON")?;
        let poly = parse_polygon3d(inner.trim())?;
        Ok(Geometry3DEnum::Polygon(poly))
    } else if upper.starts_with("MULTIPOINT") {
        let inner = extract_inner(wkt, "MULTIPOINT")?;
        let pts = parse_multipoint3d(inner.trim())?;
        Ok(Geometry3DEnum::MultiPoint(pts))
    } else if upper.starts_with("MULTILINESTRING") {
        let inner = extract_inner(wkt, "MULTILINESTRING")?;
        let lines = parse_multilinestring3d(inner.trim())?;
        Ok(Geometry3DEnum::MultiLineString(lines))
    } else if upper.starts_with("MULTIPOLYGON") {
        let inner = extract_inner(wkt, "MULTIPOLYGON")?;
        let polys = parse_multipolygon3d(inner.trim())?;
        Ok(Geometry3DEnum::MultiPolygon(polys))
    } else if upper.starts_with("GEOMETRYCOLLECTION") {
        let inner = extract_inner(wkt, "GEOMETRYCOLLECTION")?;
        let geoms = parse_geometrycollection3d(inner.trim())?;
        Ok(Geometry3DEnum::GeometryCollection(geoms))
    } else {
        Err(GeoSparqlError::InvalidWkt(format!(
            "Unknown 3D geometry type in WKT: {}",
            &wkt[..wkt.len().min(30)]
        )))
    }
}

/// Strip the type prefix (including optional Z/ZM/M modifier) and outer parens.
fn extract_inner<'a>(wkt: &'a str, type_name: &str) -> Result<&'a str> {
    let rest = wkt[type_name.len()..].trim_start();

    // Strip optional Z, ZM, M modifier
    let rest = if rest.to_uppercase().starts_with("ZM") {
        rest[2..].trim_start()
    } else if rest.to_uppercase().starts_with('Z') || rest.to_uppercase().starts_with('M') {
        rest[1..].trim_start()
    } else {
        rest
    };

    // Strip EMPTY keyword
    if rest.to_uppercase().starts_with("EMPTY") {
        return Ok("");
    }

    // Strip outer parentheses
    let rest = rest.trim();
    if rest.starts_with('(') && rest.ends_with(')') {
        Ok(&rest[1..rest.len() - 1])
    } else {
        Err(GeoSparqlError::InvalidWkt(format!(
            "Expected parentheses in WKT after type name '{}', got: '{}'",
            type_name,
            &rest[..rest.len().min(40)]
        )))
    }
}

/// Parse a single 3D point from `"x y z"` format (no parens).
fn parse_point3d_coords(coords: &str) -> Result<Point3D> {
    let parts: Vec<&str> = coords.split_whitespace().collect();
    match parts.len() {
        2 => {
            let x = parse_f64(parts[0])?;
            let y = parse_f64(parts[1])?;
            Ok(Point3D::new(x, y, 0.0))
        }
        3 => {
            let x = parse_f64(parts[0])?;
            let y = parse_f64(parts[1])?;
            let z = parse_f64(parts[2])?;
            Ok(Point3D::new(x, y, z))
        }
        4 => {
            // XYZM: treat 4th value as M, ignore it for 3D point
            let x = parse_f64(parts[0])?;
            let y = parse_f64(parts[1])?;
            let z = parse_f64(parts[2])?;
            Ok(Point3D::new(x, y, z))
        }
        _ => Err(GeoSparqlError::InvalidWkt(format!(
            "Expected 2, 3 or 4 coordinates for point, got {} in: '{}'",
            parts.len(),
            coords
        ))),
    }
}

/// Parse a comma-separated list of 3D coordinate tuples.
fn parse_coord_list_3d(s: &str) -> Result<Vec<Point3D>> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|t| parse_point3d_coords(t.trim()))
        .collect()
}

/// Parse a polygon ring list `(ring1), (ring2), ...`.
fn parse_polygon3d(s: &str) -> Result<Polygon3D> {
    let rings = split_rings(s)?;
    if rings.is_empty() {
        return Err(GeoSparqlError::InvalidWkt(
            "Polygon must have at least one ring".to_string(),
        ));
    }
    let exterior = LinearRing3D::new(parse_coord_list_3d(rings[0])?);
    let holes = rings[1..]
        .iter()
        .map(|r| parse_coord_list_3d(r).map(LinearRing3D::new))
        .collect::<Result<Vec<_>>>()?;
    Ok(Polygon3D::with_holes(exterior, holes))
}

/// Parse MULTIPOINT: `(x y z), (x y z), ...`
fn parse_multipoint3d(s: &str) -> Result<Vec<Point3D>> {
    // Support both `(x y z), (x y z)` and `x y z, x y z`
    if s.contains('(') {
        split_rings(s)?
            .iter()
            .map(|r| parse_point3d_coords(r.trim()))
            .collect()
    } else {
        parse_coord_list_3d(s)
    }
}

/// Parse MULTILINESTRING: `(ls1_coords), (ls2_coords), ...`
fn parse_multilinestring3d(s: &str) -> Result<Vec<LineString3D>> {
    split_rings(s)?
        .iter()
        .map(|r| parse_coord_list_3d(r.trim()).map(LineString3D::new))
        .collect()
}

/// Parse MULTIPOLYGON: `((ext), (hole)), ((ext)), ...`
fn parse_multipolygon3d(s: &str) -> Result<Vec<Polygon3D>> {
    // Each polygon is wrapped in an extra pair of parens
    split_polygon_groups(s)?
        .iter()
        .map(|group| parse_polygon3d(group.trim()))
        .collect()
}

/// Parse GEOMETRYCOLLECTION: recursively parse each sub-geometry.
fn parse_geometrycollection3d(s: &str) -> Result<Vec<Geometry3DEnum>> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_geometries(s)?
        .iter()
        .map(|g| parse_wkt_3d(g.trim()))
        .collect()
}

// ---------------------------------------------------------------------------
// Utility: ring / group splitters
// ---------------------------------------------------------------------------

/// Split a polygon-like string into ring content strings (without parens).
///
/// Input: `(0 0 0, 1 0 0, 1 1 0, 0 0 0), (0.1 0.1 0, ...)`
/// Output: `["0 0 0, 1 0 0, 1 1 0, 0 0 0", "0.1 0.1 0, ..."]`
fn split_rings(s: &str) -> Result<Vec<&str>> {
    let mut rings = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = s.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => {
                if depth == 0 {
                    start = i + 1;
                }
                depth += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    rings.push(&s[start..i]);
                }
            }
            _ => {}
        }
    }

    if rings.is_empty() {
        // Try treating the whole string as a single flat coord list
        rings.push(s);
    }

    Ok(rings)
}

/// Split a MULTIPOLYGON content into individual polygon content strings.
fn split_polygon_groups(s: &str) -> Result<Vec<String>> {
    let mut groups = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;

    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => {
                if depth == 0 {
                    start = i + 1;
                }
                depth += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    groups.push(s[start..i].to_string());
                }
            }
            _ => {}
        }
    }
    Ok(groups)
}

/// Split a GEOMETRYCOLLECTION content into individual geometry WKT strings.
fn split_geometries(s: &str) -> Result<Vec<String>> {
    let mut geoms = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;

    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b',' if depth == 0 => {
                geoms.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    // Push last geometry
    let last = s[start..].trim();
    if !last.is_empty() {
        geoms.push(last.to_string());
    }
    Ok(geoms)
}

// ---------------------------------------------------------------------------
// Numeric parsing helper
// ---------------------------------------------------------------------------

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>().map_err(|_| {
        GeoSparqlError::InvalidWkt(format!("Cannot parse '{}' as floating-point number", s))
    })
}

// ---------------------------------------------------------------------------
// WKT serialization helpers
// ---------------------------------------------------------------------------

pub(crate) fn points_to_wkt_coords_3d(pts: &[Point3D]) -> String {
    pts.iter()
        .map(|p| format!("{} {} {}", p.x, p.y, p.z))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn points_to_wkt_coords_2d(pts: &[Point3D]) -> String {
    pts.iter()
        .map(|p| format!("{} {}", p.x, p.y))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn ring_to_wkt_3d(ring: &LinearRing3D) -> String {
    points_to_wkt_coords_3d(&ring.points)
}

pub(crate) fn ring_to_wkt_2d(ring: &LinearRing3D) -> String {
    points_to_wkt_coords_2d(&ring.points)
}

// ---------------------------------------------------------------------------
// Z-range utilities
// ---------------------------------------------------------------------------

pub(crate) fn z_range_of_points(pts: &[Point3D]) -> Option<(f64, f64)> {
    if pts.is_empty() {
        return None;
    }
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for p in pts {
        if p.z < min_z {
            min_z = p.z;
        }
        if p.z > max_z {
            max_z = p.z;
        }
    }
    Some((min_z, max_z))
}

pub(crate) fn merge_z_ranges(ranges: &[(f64, f64)]) -> Option<(f64, f64)> {
    if ranges.is_empty() {
        return None;
    }
    let min_z = ranges
        .iter()
        .map(|(min, _)| *min)
        .fold(f64::INFINITY, f64::min);
    let max_z = ranges
        .iter()
        .map(|(_, max)| *max)
        .fold(f64::NEG_INFINITY, f64::max);
    Some((min_z, max_z))
}
