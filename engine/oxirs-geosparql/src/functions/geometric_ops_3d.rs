//! 3D distance calculations for geometric operations.
//!
//! Implements distance_3d and all supporting helpers for Z-aware distance
//! between Points, LineStrings, Polygons, and general geometry collections.

use crate::error::Result;
use crate::geometry::Geometry;

/// Helper: Calculate 3D distance between two points
pub(crate) fn point_to_point_3d(
    p1: &geo_types::Point<f64>,
    p2: &geo_types::Point<f64>,
    geom1: &Geometry,
    geom2: &Geometry,
) -> Result<f64> {
    let dx = p2.x() - p1.x();
    let dy = p2.y() - p1.y();
    let z1 = geom1.coord3d.z_at(0).unwrap_or(0.0);
    let z2 = geom2.coord3d.z_at(0).unwrap_or(0.0);
    let dz = z2 - z1;
    Ok((dx * dx + dy * dy + dz * dz).sqrt())
}

/// Helper: Calculate 3D distance from point to linestring
pub(crate) fn point_to_linestring_3d(
    point: &geo_types::Point<f64>,
    linestring: &geo_types::LineString<f64>,
    point_geom: &Geometry,
    ls_geom: &Geometry,
) -> Result<f64> {
    let px = point.x();
    let py = point.y();
    let pz = point_geom.coord3d.z_at(0).unwrap_or(0.0);

    let mut min_dist = f64::MAX;

    // Calculate distance to each line segment
    for i in 0..linestring.0.len().saturating_sub(1) {
        let p1 = &linestring.0[i];
        let p2 = &linestring.0[i + 1];
        let z1 = ls_geom.coord3d.z_at(i).unwrap_or(0.0);
        let z2 = ls_geom.coord3d.z_at(i + 1).unwrap_or(0.0);

        // Find closest point on segment to the test point in 3D
        let dist = point_to_segment_3d(px, py, pz, p1.x, p1.y, z1, p2.x, p2.y, z2);
        min_dist = min_dist.min(dist);
    }

    Ok(min_dist)
}

/// Helper: Calculate 3D distance from point to line segment
#[allow(clippy::too_many_arguments)]
pub(crate) fn point_to_segment_3d(
    px: f64,
    py: f64,
    pz: f64,
    x1: f64,
    y1: f64,
    z1: f64,
    x2: f64,
    y2: f64,
    z2: f64,
) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dz = z2 - z1;
    let len_sq = dx * dx + dy * dy + dz * dz;

    if len_sq < 1e-10 {
        // Segment is a point
        let d_x = px - x1;
        let d_y = py - y1;
        let d_z = pz - z1;
        return (d_x * d_x + d_y * d_y + d_z * d_z).sqrt();
    }

    // Project point onto the line defined by the segment
    let t = ((px - x1) * dx + (py - y1) * dy + (pz - z1) * dz) / len_sq;
    let t = t.clamp(0.0, 1.0); // Clamp to segment

    // Find closest point on segment
    let closest_x = x1 + t * dx;
    let closest_y = y1 + t * dy;
    let closest_z = z1 + t * dz;

    // Calculate distance
    let d_x = px - closest_x;
    let d_y = py - closest_y;
    let d_z = pz - closest_z;
    (d_x * d_x + d_y * d_y + d_z * d_z).sqrt()
}

/// Helper: Calculate 3D distance between two linestrings
pub(crate) fn linestring_to_linestring_3d(
    ls1: &geo_types::LineString<f64>,
    ls2: &geo_types::LineString<f64>,
    geom1: &Geometry,
    geom2: &Geometry,
) -> Result<f64> {
    let mut min_dist = f64::MAX;

    // For each segment in ls1, find minimum distance to all segments in ls2
    for i in 0..ls1.0.len().saturating_sub(1) {
        let p1 = &ls1.0[i];
        let p2 = &ls1.0[i + 1];
        let z1 = geom1.coord3d.z_at(i).unwrap_or(0.0);
        let z2 = geom1.coord3d.z_at(i + 1).unwrap_or(0.0);

        for j in 0..ls2.0.len().saturating_sub(1) {
            let q1 = &ls2.0[j];
            let q2 = &ls2.0[j + 1];
            let w1 = geom2.coord3d.z_at(j).unwrap_or(0.0);
            let w2 = geom2.coord3d.z_at(j + 1).unwrap_or(0.0);

            // Calculate minimum distance between two segments
            let dist = segment_to_segment_3d(
                p1.x, p1.y, z1, p2.x, p2.y, z2, q1.x, q1.y, w1, q2.x, q2.y, w2,
            );
            min_dist = min_dist.min(dist);
        }
    }

    Ok(min_dist)
}

/// Helper: Calculate 3D distance between two line segments
#[allow(clippy::too_many_arguments)]
pub(crate) fn segment_to_segment_3d(
    x1: f64,
    y1: f64,
    z1: f64,
    x2: f64,
    y2: f64,
    z2: f64,
    x3: f64,
    y3: f64,
    z3: f64,
    x4: f64,
    y4: f64,
    z4: f64,
) -> f64 {
    // Simplified approach: check endpoints and their projections
    let mut min_dist = f64::MAX;

    // Distance from segment 1 endpoints to segment 2
    min_dist = min_dist.min(point_to_segment_3d(x1, y1, z1, x3, y3, z3, x4, y4, z4));
    min_dist = min_dist.min(point_to_segment_3d(x2, y2, z2, x3, y3, z3, x4, y4, z4));

    // Distance from segment 2 endpoints to segment 1
    min_dist = min_dist.min(point_to_segment_3d(x3, y3, z3, x1, y1, z1, x2, y2, z2));
    min_dist = min_dist.min(point_to_segment_3d(x4, y4, z4, x1, y1, z1, x2, y2, z2));

    min_dist
}

/// Helper: Calculate 3D distance from point to polygon
pub(crate) fn point_to_polygon_3d(
    point: &geo_types::Point<f64>,
    polygon: &geo_types::Polygon<f64>,
    point_geom: &Geometry,
    poly_geom: &Geometry,
) -> Result<f64> {
    let px = point.x();
    let py = point.y();
    let pz = point_geom.coord3d.z_at(0).unwrap_or(0.0);

    let mut min_dist = f64::MAX;
    let mut coord_idx = 0;

    // Check distance to exterior ring
    let exterior = polygon.exterior();
    for i in 0..exterior.0.len().saturating_sub(1) {
        let p1 = &exterior.0[i];
        let p2 = &exterior.0[i + 1];
        let z1 = poly_geom.coord3d.z_at(coord_idx).unwrap_or(0.0);
        let z2 = poly_geom.coord3d.z_at(coord_idx + 1).unwrap_or(0.0);

        let dist = point_to_segment_3d(px, py, pz, p1.x, p1.y, z1, p2.x, p2.y, z2);
        min_dist = min_dist.min(dist);
        coord_idx += 1;
    }

    // Check distance to interior rings (holes)
    for interior in polygon.interiors() {
        for i in 0..interior.0.len().saturating_sub(1) {
            let p1 = &interior.0[i];
            let p2 = &interior.0[i + 1];
            let z1 = poly_geom.coord3d.z_at(coord_idx).unwrap_or(0.0);
            let z2 = poly_geom.coord3d.z_at(coord_idx + 1).unwrap_or(0.0);

            let dist = point_to_segment_3d(px, py, pz, p1.x, p1.y, z1, p2.x, p2.y, z2);
            min_dist = min_dist.min(dist);
            coord_idx += 1;
        }
    }

    Ok(min_dist)
}

/// Helper: General 3D distance calculation for complex geometry types
pub(crate) fn compute_min_3d_distance_general(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    // Extract all coordinates from both geometries
    let coords1 = extract_all_coordinates(&geom1.geom, &geom1.coord3d);
    let coords2 = extract_all_coordinates(&geom2.geom, &geom2.coord3d);

    if coords1.is_empty() || coords2.is_empty() {
        return Ok(f64::MAX);
    }

    let mut min_dist = f64::MAX;

    // Compute minimum distance between all coordinate pairs
    for (x1, y1, z1) in &coords1 {
        for (x2, y2, z2) in &coords2 {
            let dx = x2 - x1;
            let dy = y2 - y1;
            let dz = z2 - z1;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            min_dist = min_dist.min(dist);
        }
    }

    Ok(min_dist)
}

/// Helper: Extract all coordinates from a geometry with their Z values
pub(crate) fn extract_all_coordinates(
    geom: &geo_types::Geometry<f64>,
    coord3d: &crate::geometry::coord3d::Coord3D,
) -> Vec<(f64, f64, f64)> {
    use geo_types::Geometry as GeoGeometry;

    let mut coords = Vec::new();
    let mut idx = 0;

    match geom {
        GeoGeometry::Point(p) => {
            let z = coord3d.z_at(idx).unwrap_or(0.0);
            coords.push((p.x(), p.y(), z));
        }
        GeoGeometry::Line(line) => {
            let z1 = coord3d.z_at(idx).unwrap_or(0.0);
            coords.push((line.start.x, line.start.y, z1));
            let z2 = coord3d.z_at(idx + 1).unwrap_or(0.0);
            coords.push((line.end.x, line.end.y, z2));
        }
        GeoGeometry::LineString(ls) => {
            for coord in &ls.0 {
                let z = coord3d.z_at(idx).unwrap_or(0.0);
                coords.push((coord.x, coord.y, z));
                idx += 1;
            }
        }
        GeoGeometry::Polygon(poly) => {
            for coord in poly.exterior().0.iter() {
                let z = coord3d.z_at(idx).unwrap_or(0.0);
                coords.push((coord.x, coord.y, z));
                idx += 1;
            }
            for interior in poly.interiors() {
                for coord in interior.0.iter() {
                    let z = coord3d.z_at(idx).unwrap_or(0.0);
                    coords.push((coord.x, coord.y, z));
                    idx += 1;
                }
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            for point in mp.0.iter() {
                let z = coord3d.z_at(idx).unwrap_or(0.0);
                coords.push((point.x(), point.y(), z));
                idx += 1;
            }
        }
        GeoGeometry::MultiLineString(mls) => {
            for ls in mls.0.iter() {
                for coord in &ls.0 {
                    let z = coord3d.z_at(idx).unwrap_or(0.0);
                    coords.push((coord.x, coord.y, z));
                    idx += 1;
                }
            }
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            for poly in mpoly.0.iter() {
                for coord in poly.exterior().0.iter() {
                    let z = coord3d.z_at(idx).unwrap_or(0.0);
                    coords.push((coord.x, coord.y, z));
                    idx += 1;
                }
                for interior in poly.interiors() {
                    for coord in interior.0.iter() {
                        let z = coord3d.z_at(idx).unwrap_or(0.0);
                        coords.push((coord.x, coord.y, z));
                        idx += 1;
                    }
                }
            }
        }
        GeoGeometry::GeometryCollection(gc) => {
            // Recursively extract from collection
            // Note: This is a simplified approach; coord3d indexing might need adjustment
            for sub_geom in gc.0.iter() {
                let sub_coords = extract_all_coordinates(sub_geom, coord3d);
                coords.extend(sub_coords);
            }
        }
        GeoGeometry::Rect(rect) => {
            coords.push((rect.min().x, rect.min().y, coord3d.z_at(0).unwrap_or(0.0)));
            coords.push((rect.max().x, rect.min().y, coord3d.z_at(1).unwrap_or(0.0)));
            coords.push((rect.max().x, rect.max().y, coord3d.z_at(2).unwrap_or(0.0)));
            coords.push((rect.min().x, rect.max().y, coord3d.z_at(3).unwrap_or(0.0)));
        }
        GeoGeometry::Triangle(tri) => {
            coords.push((tri.v1().x, tri.v1().y, coord3d.z_at(0).unwrap_or(0.0)));
            coords.push((tri.v2().x, tri.v2().y, coord3d.z_at(1).unwrap_or(0.0)));
            coords.push((tri.v3().x, tri.v3().y, coord3d.z_at(2).unwrap_or(0.0)));
        }
    }

    coords
}

/// Calculate the 3D distance between two geometries, including Z coordinates
///
/// If both geometries have Z coordinates, calculates the 3D Euclidean distance:
/// √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
///
/// If either geometry lacks Z coordinates, falls back to 2D distance.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::geometric_operations::distance_3d;
///
/// let p1 = Geometry::from_wkt("POINT Z(0 0 0)").expect("should succeed");
/// let p2 = Geometry::from_wkt("POINT Z(3 4 12)").expect("should succeed");
///
/// let dist = distance_3d(&p1, &p2).expect("should succeed");
/// assert!((dist - 13.0).abs() < 0.001); // 3-4-12 right triangle -> distance = 13
/// ```
pub fn distance_3d(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    use geo_types::Geometry as GeoGeometry;

    geom1.validate_crs_compatibility(geom2)?;

    // If either geometry doesn't have Z coordinates, fall back to 2D
    if !geom1.is_3d() || !geom2.is_3d() {
        return crate::functions::geometric_operations::distance(geom1, geom2);
    }

    // Calculate 3D distance based on geometry types
    match (&geom1.geom, &geom2.geom) {
        (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) => point_to_point_3d(p1, p2, geom1, geom2),
        (GeoGeometry::Point(p), GeoGeometry::LineString(ls)) => {
            point_to_linestring_3d(p, ls, geom1, geom2)
        }
        (GeoGeometry::LineString(ls), GeoGeometry::Point(p)) => {
            point_to_linestring_3d(p, ls, geom2, geom1)
        }
        (GeoGeometry::LineString(ls1), GeoGeometry::LineString(ls2)) => {
            linestring_to_linestring_3d(ls1, ls2, geom1, geom2)
        }
        (GeoGeometry::Point(p), GeoGeometry::Polygon(poly)) => {
            point_to_polygon_3d(p, poly, geom1, geom2)
        }
        (GeoGeometry::Polygon(poly), GeoGeometry::Point(p)) => {
            point_to_polygon_3d(p, poly, geom2, geom1)
        }
        _ => {
            // For other complex geometry types, compute minimum distance between all coordinates
            compute_min_3d_distance_general(geom1, geom2)
        }
    }
}
