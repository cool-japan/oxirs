//! Geometric operations (buffer, convex hull, intersection, etc.)

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::algorithm::*;
use geo_types::Geometry as GeoGeometry;

/// Calculate the distance between two geometries
pub fn distance(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    use geo::{Distance, Euclidean};

    geom1.validate_crs_compatibility(geom2)?;

    let dist = Euclidean.distance(&geom1.geom, &geom2.geom);
    Ok(dist)
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
            let intersection = mp.intersection(p);
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
            let union = mp.union(p);
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
            let diff = mp.difference(p);
            GeoGeometry::MultiPolygon(diff)
        }
        // Polygon-MultiPolygon difference
        (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let diff = p.difference(mp);
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
            let xor = mp.xor(p);
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
}
