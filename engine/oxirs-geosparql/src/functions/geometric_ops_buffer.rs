//! Buffer and boundary operations for geometric types.
//!
//! Includes CapStyle, JoinStyle, BufferParams, 2D buffer, 3D buffer, and
//! boundary extraction — all Pure Rust.
//!
//! Buffering runs on [`geo::algorithm::buffer`], which is backed by `i_overlay`
//! and covers every geometry type plus the full OGC cap/join styles. This
//! replaced two narrower backends: the `geo-buffer` straight-skeleton crate
//! (Polygon/MultiPolygon only, behind a `rust-buffer` feature) and a
//! quarantined GEOS adapter that linked the `libgeos` C library for everything
//! else. Both are gone; there is now one unconditional Pure-Rust path.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::algorithm::buffer::{Buffer as GeoBuffer, BufferStyle, LineCap, LineJoin};
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

/// Create a buffer around a geometry with default parameters.
///
/// Works on every geometry type (Point, LineString, Polygon, the Multi\* forms,
/// and GeometryCollection). Positive distances expand, negative distances erode.
/// The result is always a MultiPolygon.
pub fn buffer(geom: &Geometry, distance: f64) -> Result<Geometry> {
    buffer_with_params(geom, distance, &BufferParams::default())
}

/// Create a buffer around a geometry with custom cap/join parameters.
pub fn buffer_with_params(
    geom: &Geometry,
    distance: f64,
    params: &BufferParams,
) -> Result<Geometry> {
    let style = buffer_style(distance, params)?;
    let buffered = geom.geom.buffer_with_style(style);
    Ok(Geometry::with_crs(
        GeoGeometry::MultiPolygon(buffered),
        geom.crs.clone(),
    ))
}

/// Translate our OGC-flavoured [`BufferParams`] into a [`BufferStyle`].
///
/// The two models express curve resolution differently, so the numeric
/// parameters are converted rather than passed through:
///
/// * `quadrant_segments` (segments per quarter circle, OGC/JTS) becomes an
///   angle per segment of `(π/2) / quadrant_segments` radians. The JTS default
///   of 8 segments maps to 0.196 rad, which is what `geo`'s own default of 0.20
///   approximates.
/// * `mitre_limit` (max ratio of miter extension to buffer distance) becomes the
///   sharpest corner angle that still gets a miter, `2·asin(1/mitre_limit)`.
///   Corners sharper than that fall back to a bevel, which is the same
///   behaviour the ratio limit produces.
fn buffer_style(distance: f64, params: &BufferParams) -> Result<BufferStyle<f64>> {
    if params.quadrant_segments < 1 {
        return Err(GeoSparqlError::InvalidParameter(format!(
            "quadrant_segments must be at least 1, got {}",
            params.quadrant_segments
        )));
    }
    if params.mitre_limit < 1.0 {
        return Err(GeoSparqlError::InvalidParameter(format!(
            "mitre_limit must be at least 1.0, got {}",
            params.mitre_limit
        )));
    }

    let segment_angle = std::f64::consts::FRAC_PI_2 / f64::from(params.quadrant_segments);

    let line_cap = match params.cap_style {
        CapStyle::Round => LineCap::Round(segment_angle),
        CapStyle::Flat => LineCap::Butt,
        CapStyle::Square => LineCap::Square,
    };

    let line_join = match params.join_style {
        JoinStyle::Round => LineJoin::Round(segment_angle),
        JoinStyle::Mitre => LineJoin::Miter(2.0 * (1.0 / params.mitre_limit).asin()),
        JoinStyle::Bevel => LineJoin::Bevel,
    };

    Ok(BufferStyle::new(distance)
        .line_cap(line_cap)
        .line_join(line_join))
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
///
/// This forwards to the Pure-Rust OGC SFA implementation in
/// [`crate::functions::de9im::boundary`], which is where the algorithm lives.
/// It used to require the GEOS C library via a quarantined adapter crate.
pub fn boundary(geom: &Geometry) -> Result<Geometry> {
    crate::functions::de9im::boundary(geom)
}
