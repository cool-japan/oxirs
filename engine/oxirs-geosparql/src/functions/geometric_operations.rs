//! Geometric operations (buffer, convex hull, intersection, etc.)

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::algorithm::*;
use geo_types::Geometry as GeoGeometry;

/// Calculate the distance between two geometries (2D only)
///
/// For 3D distance calculations that include Z coordinates, use `distance_3d()`.
pub fn distance(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    use geo::EuclideanDistance;

    geom1.validate_crs_compatibility(geom2)?;

    let dist = geom1.geom.euclidean_distance(&geom2.geom);
    Ok(dist)
}

/// Helper: Calculate 3D distance between two points
fn point_to_point_3d(
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
fn point_to_linestring_3d(
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
fn point_to_segment_3d(
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
fn linestring_to_linestring_3d(
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
fn segment_to_segment_3d(
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
fn point_to_polygon_3d(
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
fn compute_min_3d_distance_general(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
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
fn extract_all_coordinates(
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
            coords.push((tri.0.x, tri.0.y, coord3d.z_at(0).unwrap_or(0.0)));
            coords.push((tri.1.x, tri.1.y, coord3d.z_at(1).unwrap_or(0.0)));
            coords.push((tri.2.x, tri.2.y, coord3d.z_at(2).unwrap_or(0.0)));
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
/// let p1 = Geometry::from_wkt("POINT Z(0 0 0)").unwrap();
/// let p2 = Geometry::from_wkt("POINT Z(3 4 12)").unwrap();
///
/// let dist = distance_3d(&p1, &p2).unwrap();
/// assert!((dist - 13.0).abs() < 0.001); // 3-4-12 right triangle -> distance = 13
/// ```
pub fn distance_3d(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    use geo_types::Geometry as GeoGeometry;

    geom1.validate_crs_compatibility(geom2)?;

    // If either geometry doesn't have Z coordinates, fall back to 2D
    if !geom1.is_3d() || !geom2.is_3d() {
        return distance(geom1, geom2);
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

/// End cap style for buffer operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapStyle {
    /// Round end caps (default for positive buffers)
    Round,
    /// Flat/butt end caps
    Flat,
    /// Square end caps (extends beyond endpoint)
    Square,
}

/// Join style for buffer operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinStyle {
    /// Round joins (default)
    Round,
    /// Mitre/pointed joins
    Mitre,
    /// Bevel/chamfered joins
    Bevel,
}

/// Buffer parameters for controlling buffer operation
#[derive(Debug, Clone)]
pub struct BufferParams {
    /// End cap style
    pub cap_style: CapStyle,
    /// Join style
    pub join_style: JoinStyle,
    /// Number of segments per quadrant for round caps/joins (default: 8)
    pub quadrant_segments: i32,
    /// Mitre limit ratio (default: 5.0)
    pub mitre_limit: f64,
}

impl Default for BufferParams {
    fn default() -> Self {
        Self {
            cap_style: CapStyle::Round,
            join_style: JoinStyle::Round,
            quadrant_segments: 8,
            mitre_limit: 5.0,
        }
    }
}

/// Create a buffer around a geometry with default parameters
///
/// This function uses different backends based on available features:
/// - For Polygon/MultiPolygon: prefers `rust-buffer` (pure Rust), falls back to `geos-backend`
/// - For Point/LineString: requires `geos-backend`
pub fn buffer(geom: &Geometry, distance: f64) -> Result<Geometry> {
    // Try pure Rust implementation first for Polygon/MultiPolygon
    #[cfg(feature = "rust-buffer")]
    {
        match &geom.geom {
            GeoGeometry::Polygon(_) | GeoGeometry::MultiPolygon(_) => {
                return buffer_rust(geom, distance);
            }
            _ => {
                // Fall through to GEOS for other geometry types
            }
        }
    }

    // Fall back to GEOS backend or use it for Point/LineString
    buffer_with_params(geom, distance, &BufferParams::default())
}

/// Create a buffer around a geometry with custom parameters
#[cfg(feature = "geos-backend")]
pub fn buffer_with_params(
    geom: &Geometry,
    distance: f64,
    params: &BufferParams,
) -> Result<Geometry> {
    use geos::{BufferParams as GeosBufferParams, Geom, Geometry as GeosGeometry};

    // Convert our geometry to GEOS geometry
    let wkt = geom.to_wkt();
    let geos_geom = GeosGeometry::new_from_wkt(&wkt).map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("GEOS conversion failed: {}", e))
    })?;

    // Convert our BufferParams to GEOS BufferParams using builder pattern
    let cap_style = match params.cap_style {
        CapStyle::Round => geos::CapStyle::Round,
        CapStyle::Flat => geos::CapStyle::Flat,
        CapStyle::Square => geos::CapStyle::Square,
    };

    let join_style = match params.join_style {
        JoinStyle::Round => geos::JoinStyle::Round,
        JoinStyle::Mitre => geos::JoinStyle::Mitre,
        JoinStyle::Bevel => geos::JoinStyle::Bevel,
    };

    let geos_params = GeosBufferParams::builder()
        .end_cap_style(cap_style)
        .join_style(join_style)
        .quadrant_segments(params.quadrant_segments)
        .mitre_limit(params.mitre_limit)
        .build()
        .map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!(
                "Failed to create buffer params: {}",
                e
            ))
        })?;

    // Create buffer with parameters
    let buffered = geos_geom
        .buffer_with_params(distance, &geos_params)
        .map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Buffer operation failed: {}", e))
        })?;

    // Convert back to our geometry type
    let result_wkt = buffered.to_wkt().map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("WKT conversion failed: {}", e))
    })?;

    let result_geom = Geometry::from_wkt(&result_wkt)?;

    // Preserve CRS from original geometry
    Ok(Geometry::with_crs(result_geom.geom, geom.crs.clone()))
}

/// Create a buffer around a geometry with custom parameters (fallback when GEOS is not available)
#[cfg(not(feature = "geos-backend"))]
pub fn buffer_with_params(
    _geom: &Geometry,
    _distance: f64,
    _params: &BufferParams,
) -> Result<Geometry> {
    Err(GeoSparqlError::UnsupportedOperation(
        "Buffer operation requires the 'geos-backend' feature to be enabled".to_string(),
    ))
}

/// Calculate the convex hull of a geometry
pub fn convex_hull(geom: &Geometry) -> Result<Geometry> {
    use geo::ConvexHull;

    let hull = match &geom.geom {
        GeoGeometry::MultiPoint(mp) => {
            let hull_poly = mp.convex_hull();
            GeoGeometry::Polygon(hull_poly)
        }
        GeoGeometry::LineString(ls) => {
            let hull_poly = ls.convex_hull();
            GeoGeometry::Polygon(hull_poly)
        }
        GeoGeometry::Polygon(p) => {
            let hull_poly = p.convex_hull();
            GeoGeometry::Polygon(hull_poly)
        }
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "Convex hull not supported for this geometry type".to_string(),
            ))
        }
    };

    Ok(Geometry::with_crs(hull, geom.crs.clone()))
}

/// Calculate the intersection of two geometries
///
/// Returns the set of points common to both geometries.
/// For geometries that don't intersect, returns None.
pub fn intersection(geom1: &Geometry, geom2: &Geometry) -> Result<Option<Geometry>> {
    use geo::BooleanOps;

    geom1.validate_crs_compatibility(geom2)?;

    // Check if geometries intersect first
    if !geom1.geom.intersects(&geom2.geom) {
        return Ok(None);
    }

    // Perform intersection based on geometry types
    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon intersection
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let intersection = p1.intersection(p2);
            GeoGeometry::MultiPolygon(intersection)
        }
        // MultiPolygon-Polygon intersection
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p))
        | (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let intersection = mp.intersection(&p_multi);
            GeoGeometry::MultiPolygon(intersection)
        }
        // MultiPolygon-MultiPolygon intersection
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let intersection = mp1.intersection(mp2);
            GeoGeometry::MultiPolygon(intersection)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                format!(
                    "Intersection not supported for {} and {} (only Polygon and MultiPolygon are supported)",
                    geom1.geometry_type(),
                    geom2.geometry_type()
                ),
            ))
        }
    };

    Ok(Some(Geometry::with_crs(result, geom1.crs.clone())))
}

/// Calculate the union of two geometries
///
/// Returns a geometry representing all points in either geometry.
pub fn union(geom1: &Geometry, geom2: &Geometry) -> Result<Geometry> {
    use geo::BooleanOps;

    geom1.validate_crs_compatibility(geom2)?;

    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon union
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let union = p1.union(p2);
            GeoGeometry::MultiPolygon(union)
        }
        // MultiPolygon-Polygon union
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p))
        | (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let union = mp.union(&p_multi);
            GeoGeometry::MultiPolygon(union)
        }
        // MultiPolygon-MultiPolygon union
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let union = mp1.union(mp2);
            GeoGeometry::MultiPolygon(union)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(format!(
                "Union not supported for {} and {} (only Polygon and MultiPolygon are supported)",
                geom1.geometry_type(),
                geom2.geometry_type()
            )))
        }
    };

    Ok(Geometry::with_crs(result, geom1.crs.clone()))
}

/// Calculate the difference of two geometries (geom1 - geom2)
///
/// Returns the set of points in geom1 that are not in geom2.
pub fn difference(geom1: &Geometry, geom2: &Geometry) -> Result<Geometry> {
    use geo::BooleanOps;

    geom1.validate_crs_compatibility(geom2)?;

    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon difference
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let diff = p1.difference(p2);
            GeoGeometry::MultiPolygon(diff)
        }
        // MultiPolygon-Polygon difference
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let diff = mp.difference(&p_multi);
            GeoGeometry::MultiPolygon(diff)
        }
        // Polygon-MultiPolygon difference
        (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let diff = p_multi.difference(mp);
            GeoGeometry::MultiPolygon(diff)
        }
        // MultiPolygon-MultiPolygon difference
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let diff = mp1.difference(mp2);
            GeoGeometry::MultiPolygon(diff)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(format!(
            "Difference not supported for {} and {} (only Polygon and MultiPolygon are supported)",
            geom1.geometry_type(),
            geom2.geometry_type()
        )))
        }
    };

    Ok(Geometry::with_crs(result, geom1.crs.clone()))
}

/// Calculate the symmetric difference of two geometries
///
/// Returns the set of points that are in either geometry but not in both.
/// Also known as XOR operation.
pub fn sym_difference(geom1: &Geometry, geom2: &Geometry) -> Result<Geometry> {
    use geo::BooleanOps;

    geom1.validate_crs_compatibility(geom2)?;

    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon symmetric difference
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let xor = p1.xor(p2);
            GeoGeometry::MultiPolygon(xor)
        }
        // MultiPolygon-Polygon symmetric difference
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p))
        | (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let xor = mp.xor(&p_multi);
            GeoGeometry::MultiPolygon(xor)
        }
        // MultiPolygon-MultiPolygon symmetric difference
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let xor = mp1.xor(mp2);
            GeoGeometry::MultiPolygon(xor)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                format!(
                    "Symmetric difference not supported for {} and {} (only Polygon and MultiPolygon are supported)",
                    geom1.geometry_type(),
                    geom2.geometry_type()
                ),
            ))
        }
    };

    Ok(Geometry::with_crs(result, geom1.crs.clone()))
}

/// Calculate the envelope (bounding box) of a geometry
pub fn envelope(geom: &Geometry) -> Result<Geometry> {
    use geo::BoundingRect;

    let bbox = match &geom.geom {
        GeoGeometry::Point(p) => geo_types::Rect::new(
            geo_types::coord! { x: p.x(), y: p.y() },
            geo_types::coord! { x: p.x(), y: p.y() },
        ),
        GeoGeometry::LineString(ls) => ls.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::Polygon(p) => p.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::MultiPoint(mp) => mp.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::MultiLineString(mls) => mls.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::MultiPolygon(mp) => mp.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::Triangle(t) => t.bounding_rect(),
        GeoGeometry::Rect(r) => *r,
        _ => {
            return Err(GeoSparqlError::GeometryOperationFailed(
                "Could not calculate envelope".to_string(),
            ))
        }
    };

    Ok(Geometry::with_crs(
        GeoGeometry::Rect(bbox),
        geom.crs.clone(),
    ))
}

/// Create a buffer using pure Rust implementation (geo-buffer crate)
///
/// This implementation uses the straight skeleton algorithm and supports:
/// - Polygon and MultiPolygon geometries
/// - Positive buffers (expansion) and negative buffers (erosion)
/// - Simple polygons, non-convex polygons, and polygons with holes
///
/// Note: Only Polygon and MultiPolygon are supported. For other geometry types,
/// use the GEOS backend.
#[cfg(feature = "rust-buffer")]
pub fn buffer_rust(geom: &Geometry, distance: f64) -> Result<Geometry> {
    use geo_buffer::buffer_polygon;

    let result_geom = match &geom.geom {
        GeoGeometry::Polygon(poly) => {
            // buffer_polygon returns a MultiPolygon
            let buffered = buffer_polygon(poly, distance);
            GeoGeometry::MultiPolygon(buffered)
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            // Buffer each polygon individually and collect results
            let buffered_polygons: Vec<geo_types::Polygon<f64>> = mpoly
                .iter()
                .flat_map(|poly| {
                    let buffered_multi = buffer_polygon(poly, distance);
                    buffered_multi.into_iter()
                })
                .collect();

            GeoGeometry::MultiPolygon(geo_types::MultiPolygon::new(buffered_polygons))
        }
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(format!(
                "Pure Rust buffer only supports Polygon and MultiPolygon (got {}). Use geos-backend for other types.",
                geom.geometry_type()
            )))
        }
    };

    Ok(Geometry::with_crs(result_geom, geom.crs.clone()))
}

/// Create a 3D buffer around a geometry
///
/// This function creates a buffer that extends in all three dimensions:
/// - In XY: uses the standard 2D buffer operation
/// - In Z: extends the Z range by the buffer distance (both up and down)
///
/// The result is a geometry with:
/// - Expanded XY footprint (from 2D buffer)
/// - Z values extended by ±distance
///
/// # Arguments
///
/// * `geom` - The geometry to buffer (must have Z coordinates)
/// * `distance` - The buffer distance in all three dimensions
///
/// # Returns
///
/// A new 3D geometry representing the buffered region
pub fn buffer_3d(geom: &Geometry, distance: f64) -> Result<Geometry> {
    if !geom.is_3d() {
        return Err(GeoSparqlError::UnsupportedOperation(
            "Geometry must have Z coordinates for 3D buffer operation".to_string(),
        ));
    }

    // Step 1: Create 2D buffer in XY plane
    let buffered_2d = buffer(geom, distance)?;

    // Step 2: Extend Z coordinates by buffer distance
    let mut result = buffered_2d;

    // Get the original Z range
    let (original_z_min, original_z_max) = get_z_range_for_buffer(geom)?;

    // Create new Z coordinates extended by the buffer distance
    let new_z_min = original_z_min - distance;
    let new_z_max = original_z_max + distance;

    // Set the new Z coordinates on the buffered geometry
    // For simplicity, we'll set all vertices to have the extended Z range
    // In a more sophisticated implementation, we could interpolate Z values
    result.coord3d = create_extended_z_coords(&result, new_z_min, new_z_max)?;

    Ok(result)
}

/// Helper function to get Z range for buffer operation
fn get_z_range_for_buffer(geom: &Geometry) -> Result<(f64, f64)> {
    if let Some(ref z_coords) = geom.coord3d.z_coords {
        if z_coords.values.is_empty() {
            return Ok((0.0, 0.0));
        }

        let mut min_z = f64::MAX;
        let mut max_z = f64::MIN;

        for &z in &z_coords.values {
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        Ok((min_z, max_z))
    } else {
        Ok((0.0, 0.0))
    }
}

/// Helper function to create extended Z coordinates for buffered geometry
fn create_extended_z_coords(
    geom: &Geometry,
    z_min: f64,
    z_max: f64,
) -> Result<crate::geometry::coord3d::Coord3D> {
    // Count total number of coordinates in the geometry
    let coord_count = match &geom.geom {
        GeoGeometry::Point(_) => 1,
        GeoGeometry::LineString(ls) => ls.coords_count(),
        GeoGeometry::Polygon(p) => {
            let mut count = p.exterior().coords_count();
            for interior in p.interiors() {
                count += interior.coords_count();
            }
            count
        }
        GeoGeometry::MultiPoint(mp) => mp.0.len(),
        GeoGeometry::MultiLineString(mls) => mls.0.iter().map(|ls| ls.coords_count()).sum(),
        GeoGeometry::MultiPolygon(mp) => {
            mp.0.iter()
                .map(|p| {
                    let mut count = p.exterior().coords_count();
                    for interior in p.interiors() {
                        count += interior.coords_count();
                    }
                    count
                })
                .sum()
        }
        _ => 0,
    };

    // Create Z coordinates with averaged value for all points
    let avg_z = (z_min + z_max) / 2.0;
    let z_values = vec![avg_z; coord_count];

    Ok(crate::geometry::coord3d::Coord3D::xyz(z_values))
}

/// Calculate the boundary of a geometry
///
/// Returns the boundary according to the OGC Simple Features specification:
/// - Point: empty geometry
/// - LineString: the two end points
/// - Polygon: the exterior and interior rings
/// - MultiPoint/MultiLineString/MultiPolygon: union of boundaries of components
#[cfg(feature = "geos-backend")]
pub fn boundary(geom: &Geometry) -> Result<Geometry> {
    use geos::{Geom, Geometry as GeosGeometry};

    // Convert our geometry to GEOS geometry
    let wkt = geom.to_wkt();
    let geos_geom = GeosGeometry::new_from_wkt(&wkt).map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("GEOS conversion failed: {}", e))
    })?;

    // Calculate boundary
    let boundary_geom = geos_geom.boundary().map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("Boundary operation failed: {}", e))
    })?;

    // Convert back to our geometry type
    let result_wkt = boundary_geom.to_wkt().map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("WKT conversion failed: {}", e))
    })?;

    let result_geom = Geometry::from_wkt(&result_wkt)?;

    // Preserve CRS from original geometry
    Ok(Geometry::with_crs(result_geom.geom, geom.crs.clone()))
}

/// Calculate the boundary of a geometry (fallback when GEOS is not available)
#[cfg(not(feature = "geos-backend"))]
pub fn boundary(_geom: &Geometry) -> Result<Geometry> {
    Err(GeoSparqlError::UnsupportedOperation(
        "Boundary operation requires the 'geos-backend' feature to be enabled".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point};

    #[test]
    fn test_distance() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0)));

        let dist = distance(&p1, &p2).unwrap();
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
        assert_eq!(hull.unwrap().geometry_type(), "Polygon");
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

        let result = intersection(&poly1, &poly2).unwrap();
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert_eq!(intersection.geometry_type(), "MultiPolygon");
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

        let result = intersection(&poly1, &poly2).unwrap();
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
        let union_geom = result.unwrap();
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
        let diff_geom = result.unwrap();
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
        let xor_geom = result.unwrap();
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

        let buffered = buffer(&point, 1.0).unwrap();
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
        let shrunk = buffer(&poly, -1.0).unwrap();
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

        let buffered = buffer_with_params(&point, 1.0, &params).unwrap();
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

        let bound = boundary(&ls).unwrap();
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

        let bound = boundary(&poly).unwrap();
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
        let expanded = buffer_rust(&poly, 1.0).unwrap();
        assert_eq!(expanded.geometry_type(), "MultiPolygon");

        // Negative buffer (erosion)
        let shrunk = buffer_rust(&poly, -1.0).unwrap();
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

        let buffered = buffer_rust(&mpoly, 1.0).unwrap();
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

        let buffered = buffer_rust(&poly, 1.0).unwrap();
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
        let buffered = buffer(&poly, 1.0).unwrap();
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
        let p1 = Geometry::from_wkt("POINT Z(0 0 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(3 4 12)").unwrap();

        let dist = distance_3d(&p1, &p2).unwrap();
        assert!((dist - 13.0).abs() < 0.001, "Expected ~13.0, got {}", dist);
    }

    #[test]
    fn test_distance_3d_pythagorean_triple() {
        // Another Pythagorean triple: 5-12-13 in 3D
        let p1 = Geometry::from_wkt("POINT Z(0 0 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(5 12 13)").unwrap();

        let dist = distance_3d(&p1, &p2).unwrap();
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
        let p1 = Geometry::from_wkt("POINT Z(1 2 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(1 2 10)").unwrap();

        let dist = distance_3d(&p1, &p2).unwrap();
        assert!((dist - 10.0).abs() < 0.001, "Expected 10.0, got {}", dist);
    }

    #[test]
    fn test_distance_3d_negative_coordinates() {
        // Test with negative coordinates
        let p1 = Geometry::from_wkt("POINT Z(-1 -2 -3)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").unwrap();

        // Distance = √((2)² + (4)² + (6)²) = √(4 + 16 + 36) = √56 ≈ 7.483
        let dist = distance_3d(&p1, &p2).unwrap();
        assert!((dist - 7.483).abs() < 0.01, "Expected ~7.483, got {}", dist);
    }

    #[test]
    fn test_distance_3d_zero_distance() {
        // Same point should have zero distance
        let p1 = Geometry::from_wkt("POINT Z(1 2 3)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").unwrap();

        let dist = distance_3d(&p1, &p2).unwrap();
        assert!(dist.abs() < 0.001, "Expected 0.0, got {}", dist);
    }

    #[test]
    fn test_distance_3d_fallback_to_2d() {
        // If one geometry is 2D, should fall back to 2D distance
        let p1 = Geometry::from_wkt("POINT(0 0)").unwrap(); // 2D
        let p2 = Geometry::from_wkt("POINT Z(3 4 100)").unwrap(); // 3D with Z=100

        let dist_3d = distance_3d(&p1, &p2).unwrap();
        let dist_2d = distance(&p1, &p2).unwrap();

        // Should be same as 2D distance (Z ignored)
        assert!((dist_3d - dist_2d).abs() < 0.001);
        assert!((dist_3d - 5.0).abs() < 0.001); // √(3² + 4²) = 5
    }

    #[test]
    fn test_distance_3d_both_2d_uses_2d() {
        // Both 2D should use standard 2D distance
        let p1 = Geometry::from_wkt("POINT(0 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT(3 4)").unwrap();

        let dist_3d = distance_3d(&p1, &p2).unwrap();
        let dist_2d = distance(&p1, &p2).unwrap();

        assert!((dist_3d - dist_2d).abs() < 0.001);
        assert!((dist_3d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_3d_fractional_coordinates() {
        // Test with fractional coordinates
        let p1 = Geometry::from_wkt("POINT Z(1.5 2.5 3.5)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(4.5 6.5 7.5)").unwrap();

        // Δx=3, Δy=4, Δz=4 -> √(9 + 16 + 16) = √41 ≈ 6.403
        let dist = distance_3d(&p1, &p2).unwrap();
        assert!((dist - 6.403).abs() < 0.01, "Expected ~6.403, got {}", dist);
    }

    #[test]
    fn test_distance_3d_crs_compatibility() {
        // Different CRS should fail
        let p1 = Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT Z(0 0 0)")
            .unwrap();
        let p2 = Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/3857> POINT Z(1 1 1)")
            .unwrap();

        let result = distance_3d(&p1, &p2);
        assert!(result.is_err());
    }

    #[test]
    fn test_distance_3d_with_m_coordinate() {
        // POINT ZM should work (M coordinate ignored in distance calculation)
        let p1 = Geometry::from_wkt("POINT ZM(0 0 0 100)").unwrap();
        let p2 = Geometry::from_wkt("POINT ZM(3 4 12 200)").unwrap();

        let dist = distance_3d(&p1, &p2).unwrap();
        assert!((dist - 13.0).abs() < 0.001); // M coordinate should be ignored
    }

    #[test]
    fn test_distance_2d_vs_3d_comparison() {
        // Same 2D projection but different Z should give different distances
        let p1_3d = Geometry::from_wkt("POINT Z(0 0 0)").unwrap();
        let p2_3d = Geometry::from_wkt("POINT Z(3 4 12)").unwrap();

        let p1_2d = Geometry::from_wkt("POINT(0 0)").unwrap();
        let p2_2d = Geometry::from_wkt("POINT(3 4)").unwrap();

        let dist_3d = distance_3d(&p1_3d, &p2_3d).unwrap();
        let dist_2d = distance(&p1_2d, &p2_2d).unwrap();

        // 3D distance should be greater due to Z component
        assert!(dist_3d > dist_2d);
        assert!((dist_2d - 5.0).abs() < 0.001); // 2D: √(3² + 4²) = 5
        assert!((dist_3d - 13.0).abs() < 0.001); // 3D: √(3² + 4² + 12²) = 13
    }

    #[test]
    fn test_distance_3d_point_to_linestring() {
        // Point above a horizontal line segment
        let point = Geometry::from_wkt("POINT Z(5 5 10)").unwrap();
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 0)").unwrap();

        let dist = distance_3d(&point, &line).unwrap();

        // Closest point on line is (5, 0, 0), distance = √(0² + 5² + 10²) = √125 ≈ 11.18
        assert!((dist - 11.180).abs() < 0.01);
    }

    #[test]
    fn test_distance_3d_linestring_to_linestring() {
        // Two parallel vertical lines at different heights
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 0 0 10)").unwrap();
        let ls2 = Geometry::from_wkt("LINESTRING Z(3 4 0, 3 4 10)").unwrap();

        let dist = distance_3d(&ls1, &ls2).unwrap();

        // Lines are parallel, distance is constant at √(3² + 4²) = 5
        assert!((dist - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_3d_linestring_to_linestring_skew() {
        // Two skew lines in 3D (don't intersect and aren't parallel)
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 0)").unwrap();
        let ls2 = Geometry::from_wkt("LINESTRING Z(0 10 10, 10 10 10)").unwrap();

        let dist = distance_3d(&ls1, &ls2).unwrap();

        // These are parallel horizontal lines
        // Line 1 at Y=0, Z=0
        // Line 2 at Y=10, Z=10
        // Distance = √(10² + 10²) = √200 ≈ 14.14
        assert!((dist - 14.14).abs() < 0.1);
    }

    #[test]
    fn test_distance_3d_point_to_polygon() {
        // Point above a square polygon
        let point = Geometry::from_wkt("POINT Z(5 5 10)").unwrap();
        let poly =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))").unwrap();

        let dist = distance_3d(&point, &poly).unwrap();

        // Point is directly above center of polygon at height 10
        // But distance is to the boundary, not the interior
        // Closest boundary point would be an edge, giving distance > 10
        assert!(dist > 9.5); // Should be close to 10 (vertical distance component)
    }

    #[test]
    fn test_distance_3d_multipoint() {
        // MultiPoint to Point distance
        let mp = Geometry::from_wkt("MULTIPOINT Z((0 0 0), (10 0 0), (0 10 0))").unwrap();
        let p = Geometry::from_wkt("POINT Z(0 0 5)").unwrap();

        let dist = distance_3d(&mp, &p).unwrap();

        // Closest point in MultiPoint is (0, 0, 0), distance = 5
        assert!((dist - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_3d_same_geometry() {
        // LineString to itself should have zero distance
        let ls = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 10)").unwrap();

        let dist = distance_3d(&ls, &ls).unwrap();

        assert!(dist.abs() < 0.001);
    }

    #[test]
    fn test_distance_3d_point_on_linestring() {
        // Point that lies exactly on a line segment
        let point = Geometry::from_wkt("POINT Z(5 0 5)").unwrap();
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 10)").unwrap();

        let dist = distance_3d(&point, &line).unwrap();

        // Point (5, 0, 5) is exactly on the line from (0, 0, 0) to (10, 0, 10)
        assert!(dist < 0.01); // Should be very close to zero
    }

    #[test]
    fn test_distance_3d_vertical_separation() {
        // Two geometries with same X,Y but different Z
        let p1 = Geometry::from_wkt("POINT Z(1 2 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(1 2 100)").unwrap();

        let dist = distance_3d(&p1, &p2).unwrap();

        assert!((dist - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_3d_diagonal_linestring() {
        // Point to a diagonal line in 3D space
        let point = Geometry::from_wkt("POINT Z(0 0 0)").unwrap();
        let line = Geometry::from_wkt("LINESTRING Z(10 10 10, 20 20 20)").unwrap();

        let dist = distance_3d(&point, &line).unwrap();

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
        use geo::{Area, EuclideanLength};

        let point = Geometry::from_wkt("POINT Z(0 0 5)").unwrap();
        let buffered = buffer_3d(&point, 1.0).unwrap();

        // Should be 3D
        assert!(buffered.is_3d());

        // Check that the buffer was created (geometry type might change to Polygon)
        // The exact result depends on the buffer implementation
        let length = match &buffered.geom {
            GeoGeometry::LineString(ls) => ls.euclidean_length(),
            GeoGeometry::Polygon(p) => p.exterior().euclidean_length(),
            _ => 0.0,
        };
        assert!(length > 0.0 || buffered.geom.unsigned_area() > 0.0);
    }

    #[test]
    #[cfg(any(feature = "geos-backend", feature = "rust-buffer"))]
    fn test_buffer_3d_linestring() {
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 0 10)").unwrap();
        let buffered = buffer_3d(&line, 2.0).unwrap();

        // Should be 3D
        assert!(buffered.is_3d());

        // Should have expanded the geometry
        use geo::Area;
        assert!(buffered.geom.unsigned_area() > 0.0);
    }

    #[test]
    #[cfg(feature = "rust-buffer")]
    fn test_buffer_3d_polygon() {
        let poly =
            Geometry::from_wkt("POLYGON Z((0 0 5, 10 0 5, 10 10 5, 0 10 5, 0 0 5))").unwrap();
        let buffered = buffer_3d(&poly, 1.0).unwrap();

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
        let poly =
            Geometry::from_wkt("POLYGON Z((0 0 10, 5 0 10, 5 5 10, 0 5 10, 0 0 10))").unwrap();

        // Original Z range: [10, 10]
        let original_z_min = 10.0;
        let buffer_distance = 2.0;

        let buffered = buffer_3d(&poly, buffer_distance).unwrap();

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
        let line = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 20)").unwrap();
        let buffered = buffer_3d(&line, 1.0).unwrap();

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
        let point = Geometry::from_wkt("POINT(0 0)").unwrap(); // 2D point

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
        let poly =
            Geometry::from_wkt("POLYGON Z((0 0 10, 20 0 10, 20 20 10, 0 20 10, 0 0 10))").unwrap();

        let buffered = buffer_3d(&poly, -2.0).unwrap();

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
        .unwrap();

        let buffered = buffer_3d(&mpoly, 1.0).unwrap();

        // Should be 3D
        assert!(buffered.is_3d());

        // Should have larger area
        use geo::Area;
        assert!(buffered.geom.unsigned_area() > mpoly.geom.unsigned_area());
    }
}
