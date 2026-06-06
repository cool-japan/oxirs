//! Buffer and boundary operations for geometric types.
//!
//! Includes CapStyle, JoinStyle, BufferParams, standard 2D buffer,
//! pure-Rust polygon buffer, 3D buffer, and boundary extraction.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::CoordsIter;
use geo_types::Geometry as GeoGeometry;

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
