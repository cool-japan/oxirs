//! GEOS-backed buffer and boundary operations.
//!
//! These are the only two functions that touch the GEOS C FFI directly; the
//! Egenhofer/RCC8 predicates in this crate build on [`boundary`] plus the parent's
//! Pure-Rust helpers.

use geos::{BufferParams as GeosBufferParams, Geom, Geometry as GeosGeometry};
use oxirs_geosparql::error::{GeoSparqlError, Result};
use oxirs_geosparql::functions::geometric_operations::{BufferParams, CapStyle, JoinStyle};
use oxirs_geosparql::geometry::Geometry;

/// Create a buffer around a geometry with default parameters, using GEOS.
///
/// Unlike the parent's Pure-Rust `buffer_rust` (Polygon/MultiPolygon only), this
/// handles every geometry type (Point, LineString, MultiPoint, ...).
pub fn buffer(geom: &Geometry, distance: f64) -> Result<Geometry> {
    buffer_with_params(geom, distance, &BufferParams::default())
}

/// Create a buffer around a geometry with custom parameters, using GEOS.
pub fn buffer_with_params(
    geom: &Geometry,
    distance: f64,
    params: &BufferParams,
) -> Result<Geometry> {
    // Convert our geometry to a GEOS geometry.
    let wkt = geom.to_wkt();
    let geos_geom = GeosGeometry::new_from_wkt(&wkt).map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("GEOS conversion failed: {}", e))
    })?;

    // Map our BufferParams onto GEOS BufferParams via the builder.
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

    let buffered = geos_geom
        .buffer_with_params(distance, &geos_params)
        .map_err(|e| {
            GeoSparqlError::GeometryOperationFailed(format!("Buffer operation failed: {}", e))
        })?;

    // Convert back to our geometry type, preserving the original CRS.
    let result_wkt = buffered.to_wkt().map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("WKT conversion failed: {}", e))
    })?;

    let result_geom = Geometry::from_wkt(&result_wkt)?;
    Ok(Geometry::with_crs(result_geom.geom, geom.crs.clone()))
}

/// Calculate the boundary of a geometry, using GEOS.
///
/// Returns the boundary according to the OGC Simple Features specification:
/// - Point: empty geometry
/// - LineString: the two end points
/// - Polygon: the exterior and interior rings
/// - MultiPoint/MultiLineString/MultiPolygon: union of component boundaries
pub fn boundary(geom: &Geometry) -> Result<Geometry> {
    let wkt = geom.to_wkt();
    let geos_geom = GeosGeometry::new_from_wkt(&wkt).map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("GEOS conversion failed: {}", e))
    })?;

    let boundary_geom = geos_geom.boundary().map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("Boundary operation failed: {}", e))
    })?;

    let result_wkt = boundary_geom.to_wkt().map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!("WKT conversion failed: {}", e))
    })?;

    let result_geom = Geometry::from_wkt(&result_wkt)?;
    Ok(Geometry::with_crs(result_geom.geom, geom.crs.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::Geometry as GeoGeometry;

    #[test]
    fn test_buffer_positive_point() {
        let point = Geometry::from_wkt("POINT(0 0)").expect("valid point");
        let buffered = buffer(&point, 1.0).expect("geos buffer should succeed");
        assert_eq!(buffered.geometry_type(), "Polygon");

        use geo::Area;
        if let GeoGeometry::Polygon(poly) = buffered.geom {
            let area = poly.unsigned_area();
            let expected_area = std::f64::consts::PI;
            assert!((area - expected_area).abs() < 0.1);
        }
    }

    #[test]
    fn test_buffer_with_custom_params() {
        let point = Geometry::from_wkt("POINT(0 0)").expect("valid point");
        let params = BufferParams {
            cap_style: CapStyle::Square,
            join_style: JoinStyle::Mitre,
            quadrant_segments: 4,
            mitre_limit: 2.0,
        };
        let buffered =
            buffer_with_params(&point, 1.0, &params).expect("geos buffer should succeed");
        assert_eq!(buffered.geometry_type(), "Polygon");
    }

    #[test]
    fn test_boundary_linestring() {
        let ls = Geometry::from_wkt("LINESTRING(0 0, 5 0, 5 5)").expect("valid linestring");
        let bound = boundary(&ls).expect("geos boundary should succeed");
        assert_eq!(bound.geometry_type(), "MultiPoint");
    }

    #[test]
    fn test_boundary_polygon() {
        let poly =
            Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").expect("valid polygon");
        let bound = boundary(&poly).expect("geos boundary should succeed");
        assert!(matches!(
            bound.geometry_type(),
            "LineString" | "MultiLineString"
        ));
    }
}
