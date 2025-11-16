//! 3D Buffer Operations
//!
//! This module provides advanced 3D buffering algorithms that extend geometries
//! in three dimensions with configurable parameters for vertical and horizontal
//! expansion, cap styles, and join styles.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{coord3d::Coord3D, Geometry};
use geo::CoordsIter;
use geo_types::{Coord, Geometry as GeoGeometry};
use serde::{Deserialize, Serialize};

/// Cap style for 3D buffer operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CapStyle3D {
    /// Spherical cap (rounded in all directions)
    #[default]
    Spherical,
    /// Cylindrical cap (rounded horizontally, flat vertically)
    Cylindrical,
    /// Flat cap (no rounding)
    Flat,
}

/// Join style for 3D edges
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum JoinStyle3D {
    /// Round join with 3D arc interpolation
    #[default]
    Round,
    /// Beveled join (straight connection)
    Bevel,
    /// Mitre join with 3D angle calculation
    Mitre,
}

/// Z-coordinate interpolation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ZInterpolationStrategy {
    /// Use average of min and max Z
    Average,
    /// Preserve original Z values and extend
    #[default]
    Preserve,
    /// Linear interpolation based on position
    Linear,
    /// Smooth interpolation using splines
    Smooth,
}

/// Parameters for 3D buffer operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferParams3D {
    /// Horizontal buffer distance (XY plane)
    pub horizontal_distance: f64,

    /// Vertical buffer distance (Z axis)
    pub vertical_distance: f64,

    /// Cap style for endpoints
    pub cap_style: CapStyle3D,

    /// Join style for corners/edges
    pub join_style: JoinStyle3D,

    /// Number of segments for rounded features
    pub quadrant_segments: i32,

    /// Mitre limit for mitre joins
    pub mitre_limit: f64,

    /// Z-coordinate interpolation strategy
    pub z_interpolation: ZInterpolationStrategy,
}

impl Default for BufferParams3D {
    fn default() -> Self {
        Self {
            horizontal_distance: 0.0,
            vertical_distance: 0.0,
            cap_style: CapStyle3D::default(),
            join_style: JoinStyle3D::default(),
            quadrant_segments: 8,
            mitre_limit: 5.0,
            z_interpolation: ZInterpolationStrategy::default(),
        }
    }
}

impl BufferParams3D {
    /// Create new buffer parameters with uniform distance in all directions
    pub fn uniform(distance: f64) -> Self {
        Self {
            horizontal_distance: distance,
            vertical_distance: distance,
            ..Default::default()
        }
    }

    /// Create new buffer parameters with separate horizontal and vertical distances
    pub fn anisotropic(horizontal: f64, vertical: f64) -> Self {
        Self {
            horizontal_distance: horizontal,
            vertical_distance: vertical,
            ..Default::default()
        }
    }

    /// Set cap style
    pub fn with_cap_style(mut self, cap_style: CapStyle3D) -> Self {
        self.cap_style = cap_style;
        self
    }

    /// Set join style
    pub fn with_join_style(mut self, join_style: JoinStyle3D) -> Self {
        self.join_style = join_style;
        self
    }

    /// Set quadrant segments
    pub fn with_quadrant_segments(mut self, segments: i32) -> Self {
        self.quadrant_segments = segments;
        self
    }

    /// Set Z interpolation strategy
    pub fn with_z_interpolation(mut self, strategy: ZInterpolationStrategy) -> Self {
        self.z_interpolation = strategy;
        self
    }
}

/// Create a 3D buffer around a geometry with uniform distance
///
/// This is a convenience function that creates a uniform 3D buffer.
/// For more control, use `buffer_3d_with_params`.
///
/// # Arguments
///
/// * `geom` - The geometry to buffer (must have Z coordinates)
/// * `distance` - The buffer distance in all three dimensions
///
/// # Returns
///
/// A new 3D geometry representing the buffered region
///
/// # Example
///
/// ```ignore
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::buffer_3d::buffer_3d;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a 3D point
/// let geom = Geometry::from_wkt("POINT Z (1 2 3)")?;
///
/// // Create a 3D buffer with uniform distance
/// let buffered = buffer_3d(&geom, 10.0)?;
///
/// assert!(buffered.is_3d());
/// # Ok(())
/// # }
/// ```
pub fn buffer_3d(geom: &Geometry, distance: f64) -> Result<Geometry> {
    let params = BufferParams3D::uniform(distance);
    buffer_3d_with_params(geom, &params)
}

/// Create a 3D buffer around a geometry with custom parameters
///
/// This function provides full control over the 3D buffering process, including
/// separate horizontal and vertical distances, cap styles, and Z interpolation.
///
/// # Arguments
///
/// * `geom` - The geometry to buffer (must have Z coordinates)
/// * `params` - Buffer parameters controlling the operation
///
/// # Returns
///
/// A new 3D geometry representing the buffered region
///
/// # Example
///
/// ```ignore
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::buffer_3d::{buffer_3d_with_params, BufferParams3D, CapStyle3D};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let geom = Geometry::from_wkt("POINT Z (1 2 3)")?;
///
/// // Create parameters with different horizontal and vertical distances
/// let params = BufferParams3D::anisotropic(10.0, 5.0)
///     .with_cap_style(CapStyle3D::Spherical);
///
/// let buffered = buffer_3d_with_params(&geom, &params)?;
/// # Ok(())
/// # }
/// ```
pub fn buffer_3d_with_params(geom: &Geometry, params: &BufferParams3D) -> Result<Geometry> {
    if !geom.is_3d() {
        return Err(GeoSparqlError::UnsupportedOperation(
            "Geometry must have Z coordinates for 3D buffer operation".to_string(),
        ));
    }

    // Step 1: Create 2D buffer in XY plane with horizontal distance
    let buffered_2d = buffer_2d_horizontal(geom, params)?;

    // Step 2: Apply vertical expansion to Z coordinates
    let mut result = buffered_2d;
    result.coord3d = apply_vertical_buffer(geom, &result, params)?;

    Ok(result)
}

/// Create a horizontal (XY plane) buffer using the existing 2D buffer implementation
fn buffer_2d_horizontal(geom: &Geometry, params: &BufferParams3D) -> Result<Geometry> {
    // Convert 3D cap/join styles to 2D equivalents
    use crate::functions::geometric_operations::{
        buffer_with_params, BufferParams, CapStyle, JoinStyle,
    };

    let cap_style_2d = match params.cap_style {
        CapStyle3D::Spherical | CapStyle3D::Cylindrical => CapStyle::Round,
        CapStyle3D::Flat => CapStyle::Flat,
    };

    let join_style_2d = match params.join_style {
        JoinStyle3D::Round => JoinStyle::Round,
        JoinStyle3D::Bevel => JoinStyle::Bevel,
        JoinStyle3D::Mitre => JoinStyle::Mitre,
    };

    let buffer_params_2d = BufferParams {
        cap_style: cap_style_2d,
        join_style: join_style_2d,
        quadrant_segments: params.quadrant_segments,
        mitre_limit: params.mitre_limit,
    };

    buffer_with_params(geom, params.horizontal_distance, &buffer_params_2d)
}

/// Apply vertical buffer expansion to Z coordinates
fn apply_vertical_buffer(
    original_geom: &Geometry,
    buffered_geom: &Geometry,
    params: &BufferParams3D,
) -> Result<Coord3D> {
    let (z_min, z_max) = get_z_range(original_geom)?;

    // Extend Z range by vertical distance
    let new_z_min = z_min - params.vertical_distance;
    let new_z_max = z_max + params.vertical_distance;

    // Create Z coordinates for the buffered geometry based on interpolation strategy
    let z_values = match params.z_interpolation {
        ZInterpolationStrategy::Average => {
            create_z_coords_average(buffered_geom, new_z_min, new_z_max)?
        }
        ZInterpolationStrategy::Preserve => {
            create_z_coords_preserve(original_geom, buffered_geom, new_z_min, new_z_max)?
        }
        ZInterpolationStrategy::Linear => {
            create_z_coords_linear(buffered_geom, new_z_min, new_z_max)?
        }
        ZInterpolationStrategy::Smooth => {
            create_z_coords_smooth(buffered_geom, new_z_min, new_z_max)?
        }
    };

    Ok(Coord3D::xyz(z_values))
}

/// Get Z coordinate range from geometry
fn get_z_range(geom: &Geometry) -> Result<(f64, f64)> {
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

/// Create Z coordinates using average strategy
fn create_z_coords_average(geom: &Geometry, z_min: f64, z_max: f64) -> Result<Vec<f64>> {
    let coord_count = count_coords(geom)?;
    let avg_z = (z_min + z_max) / 2.0;
    Ok(vec![avg_z; coord_count])
}

/// Create Z coordinates using preserve strategy
fn create_z_coords_preserve(
    original_geom: &Geometry,
    buffered_geom: &Geometry,
    z_min: f64,
    z_max: f64,
) -> Result<Vec<f64>> {
    let coord_count = count_coords(buffered_geom)?;

    // Get original Z values
    let original_z_values = if let Some(ref z_coords) = original_geom.coord3d.z_coords {
        &z_coords.values
    } else {
        return create_z_coords_average(buffered_geom, z_min, z_max);
    };

    if original_z_values.is_empty() {
        return create_z_coords_average(buffered_geom, z_min, z_max);
    }

    // For buffered geometry, we need to map new coordinates to nearest original coordinates
    // For simplicity, we'll use the average of original Z values for new coordinates
    let avg_original_z = original_z_values.iter().sum::<f64>() / original_z_values.len() as f64;

    Ok(vec![avg_original_z; coord_count])
}

/// Create Z coordinates using linear interpolation strategy
fn create_z_coords_linear(geom: &Geometry, z_min: f64, z_max: f64) -> Result<Vec<f64>> {
    let coords = extract_all_coords(geom)?;
    let coord_count = coords.len();

    if coord_count == 0 {
        return Ok(Vec::new());
    }

    // Find XY bounding box
    let (x_min, x_max, y_min, y_max) = compute_xy_bounds(&coords);

    // Interpolate Z based on position in XY plane
    let mut z_values = Vec::with_capacity(coord_count);

    for coord in &coords {
        // Normalize position to [0, 1]
        let t_x = if x_max > x_min {
            (coord.x - x_min) / (x_max - x_min)
        } else {
            0.5
        };

        let t_y = if y_max > y_min {
            (coord.y - y_min) / (y_max - y_min)
        } else {
            0.5
        };

        // Use combined position for Z interpolation
        let t = (t_x + t_y) / 2.0;
        let z = z_min + t * (z_max - z_min);
        z_values.push(z);
    }

    Ok(z_values)
}

/// Create Z coordinates using smooth interpolation strategy
fn create_z_coords_smooth(geom: &Geometry, z_min: f64, z_max: f64) -> Result<Vec<f64>> {
    let coords = extract_all_coords(geom)?;
    let coord_count = coords.len();

    if coord_count == 0 {
        return Ok(Vec::new());
    }

    // Find XY bounding box and centroid
    let (x_min, x_max, y_min, y_max) = compute_xy_bounds(&coords);
    let cx = (x_min + x_max) / 2.0;
    let cy = (y_min + y_max) / 2.0;
    let max_dist = ((x_max - x_min).powi(2) + (y_max - y_min).powi(2)).sqrt() / 2.0;

    // Smooth interpolation based on distance from centroid
    let mut z_values = Vec::with_capacity(coord_count);

    for coord in &coords {
        let dx = coord.x - cx;
        let dy = coord.y - cy;
        let dist = (dx * dx + dy * dy).sqrt();

        // Normalize distance and apply smooth step function
        let t = if max_dist > 0.0 {
            (dist / max_dist).min(1.0)
        } else {
            0.5
        };

        // Smoothstep function: 3t² - 2t³
        let smooth_t = 3.0 * t * t - 2.0 * t * t * t;

        let z = z_min + smooth_t * (z_max - z_min);
        z_values.push(z);
    }

    Ok(z_values)
}

/// Count total number of coordinates in a geometry
fn count_coords(geom: &Geometry) -> Result<usize> {
    let count = match &geom.geom {
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
    Ok(count)
}

/// Extract all coordinates from a geometry
fn extract_all_coords(geom: &Geometry) -> Result<Vec<Coord<f64>>> {
    let mut coords = Vec::new();

    match &geom.geom {
        GeoGeometry::Point(p) => {
            coords.push(p.0);
        }
        GeoGeometry::LineString(ls) => {
            coords.extend(ls.coords().cloned());
        }
        GeoGeometry::Polygon(p) => {
            coords.extend(p.exterior().coords().cloned());
            for interior in p.interiors() {
                coords.extend(interior.coords().cloned());
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            coords.extend(mp.0.iter().map(|p| p.0));
        }
        GeoGeometry::MultiLineString(mls) => {
            for ls in &mls.0 {
                coords.extend(ls.coords().cloned());
            }
        }
        GeoGeometry::MultiPolygon(mp) => {
            for p in &mp.0 {
                coords.extend(p.exterior().coords().cloned());
                for interior in p.interiors() {
                    coords.extend(interior.coords().cloned());
                }
            }
        }
        _ => {}
    }

    Ok(coords)
}

/// Compute XY bounding box
fn compute_xy_bounds(coords: &[Coord<f64>]) -> (f64, f64, f64, f64) {
    let mut x_min = f64::MAX;
    let mut x_max = f64::MIN;
    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;

    for coord in coords {
        x_min = x_min.min(coord.x);
        x_max = x_max.max(coord.x);
        y_min = y_min.min(coord.y);
        y_max = y_max.max(coord.y);
    }

    (x_min, x_max, y_min, y_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_buffer_params_3d_uniform() {
        let params = BufferParams3D::uniform(10.0);
        assert_eq!(params.horizontal_distance, 10.0);
        assert_eq!(params.vertical_distance, 10.0);
    }

    #[test]
    fn test_buffer_params_3d_anisotropic() {
        let params = BufferParams3D::anisotropic(10.0, 5.0);
        assert_eq!(params.horizontal_distance, 10.0);
        assert_eq!(params.vertical_distance, 5.0);
    }

    #[test]
    fn test_buffer_params_3d_builder() {
        let params = BufferParams3D::uniform(10.0)
            .with_cap_style(CapStyle3D::Cylindrical)
            .with_join_style(JoinStyle3D::Bevel)
            .with_quadrant_segments(16);

        assert_eq!(params.cap_style, CapStyle3D::Cylindrical);
        assert_eq!(params.join_style, JoinStyle3D::Bevel);
        assert_eq!(params.quadrant_segments, 16);
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_buffer_3d_point() {
        let geom = Geometry::from_wkt("POINT Z (1 2 3)").unwrap();
        let buffered = buffer_3d(&geom, 10.0).unwrap();

        assert!(buffered.is_3d());
        assert!(buffered.coord3d.has_z());
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_buffer_3d_with_params() {
        let geom = Geometry::from_wkt("POINT Z (1 2 3)").unwrap();
        let params = BufferParams3D::anisotropic(10.0, 5.0);
        let buffered = buffer_3d_with_params(&geom, &params).unwrap();

        assert!(buffered.is_3d());
    }

    #[test]
    fn test_buffer_3d_requires_3d_geometry() {
        let geom = Geometry::from_wkt("POINT (1 2)").unwrap();
        let result = buffer_3d(&geom, 10.0);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeoSparqlError::UnsupportedOperation(_)
        ));
    }

    #[test]
    fn test_get_z_range() {
        let geom = Geometry::from_wkt("LINESTRING Z (0 0 1, 1 1 5, 2 2 3)").unwrap();
        let (z_min, z_max) = get_z_range(&geom).unwrap();

        assert_relative_eq!(z_min, 1.0);
        assert_relative_eq!(z_max, 5.0);
    }

    #[test]
    fn test_count_coords_point() {
        let geom = Geometry::from_wkt("POINT Z (1 2 3)").unwrap();
        let count = count_coords(&geom).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_count_coords_linestring() {
        let geom = Geometry::from_wkt("LINESTRING Z (0 0 1, 1 1 2, 2 2 3)").unwrap();
        let count = count_coords(&geom).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_count_coords_polygon() {
        let geom = Geometry::from_wkt("POLYGON Z ((0 0 1, 1 0 2, 1 1 3, 0 1 4, 0 0 1))").unwrap();
        let count = count_coords(&geom).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_extract_all_coords() {
        let geom = Geometry::from_wkt("LINESTRING Z (0 0 1, 1 1 2, 2 2 3)").unwrap();
        let coords = extract_all_coords(&geom).unwrap();

        assert_eq!(coords.len(), 3);
        assert_relative_eq!(coords[0].x, 0.0);
        assert_relative_eq!(coords[1].x, 1.0);
        assert_relative_eq!(coords[2].x, 2.0);
    }

    #[test]
    fn test_compute_xy_bounds() {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 2.0, y: 3.0 },
            Coord { x: 1.0, y: 1.0 },
        ];

        let (x_min, x_max, y_min, y_max) = compute_xy_bounds(&coords);

        assert_relative_eq!(x_min, 0.0);
        assert_relative_eq!(x_max, 2.0);
        assert_relative_eq!(y_min, 0.0);
        assert_relative_eq!(y_max, 3.0);
    }

    #[test]
    fn test_create_z_coords_average() {
        let geom = Geometry::from_wkt("LINESTRING Z (0 0 1, 1 1 2, 2 2 3)").unwrap();
        let z_values = create_z_coords_average(&geom, 1.0, 5.0).unwrap();

        assert_eq!(z_values.len(), 3);
        for &z in &z_values {
            assert_relative_eq!(z, 3.0); // Average of 1.0 and 5.0
        }
    }

    #[test]
    fn test_z_interpolation_strategies() {
        // Use a linestring with 3 points where the middle point is closer to centroid
        let geom = Geometry::from_wkt("LINESTRING Z (0 0 0, 1 0 5, 2 0 10)").unwrap();

        // Average strategy - all values should be the same
        let z_avg = create_z_coords_average(&geom, 0.0, 10.0).unwrap();
        assert_eq!(z_avg.len(), 3);
        assert_relative_eq!(z_avg[0], 5.0);
        assert_relative_eq!(z_avg[1], 5.0);
        assert_relative_eq!(z_avg[2], 5.0);

        // Linear strategy - should vary based on position
        let z_linear = create_z_coords_linear(&geom, 0.0, 10.0).unwrap();
        assert_eq!(z_linear.len(), 3);
        // All points should be within expected range
        for &z in &z_linear {
            assert!((0.0..=10.0).contains(&z));
        }

        // Smooth strategy - should vary based on distance from centroid
        let z_smooth = create_z_coords_smooth(&geom, 0.0, 10.0).unwrap();
        assert_eq!(z_smooth.len(), 3);
        // All points should be within expected range
        for &z in &z_smooth {
            assert!((0.0..=10.0).contains(&z));
        }
        // The middle point should be closer to z_min (closer to centroid)
        // The endpoints should be closer to z_max (farther from centroid)
        assert!(z_smooth[1] < z_smooth[0] || z_smooth[1] < z_smooth[2]);
    }
}
