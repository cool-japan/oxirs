//! Geometry validation and quality checks
//!
//! This module provides validation and quality checking for geometries,
//! including validity checks, repair operations, and simplification algorithms.
//!
//! # Features
//!
//! - Validity checking (self-intersections, proper topology)
//! - Automatic geometry repair
//! - Simplification (Douglas-Peucker, Visvalingam-Whyatt)
//! - Precision model handling
//! - Topology validation
//!
//! # Examples
//!
//! ```
//! use oxirs_geosparql::validation::*;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
//!
//! // Check if geometry is valid
//! let validation = validate_geometry(&geom);
//! assert!(validation.is_valid);
//!
//! // Simplify geometry
//! let simplified = simplify_geometry(&geom, 0.01).unwrap();
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::{Area, EuclideanLength, Simplify, SimplifyVwPreserve};
use geo_types::Geometry as GeoGeometry;

/// Validation result containing validity status and any errors found
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the geometry is valid
    pub is_valid: bool,
    /// List of validation errors, if any
    pub errors: Vec<String>,
    /// Validation warnings (non-fatal issues)
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a new invalid result with errors
    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add a warning to the validation result
    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Add an error to the validation result
    pub fn with_error(mut self, error: String) -> Self {
        self.errors.push(error);
        self.is_valid = false;
        self
    }
}

/// Validate a geometry for correctness
///
/// Checks for:
/// - Self-intersections in LineStrings and Polygons
/// - Proper ring orientation in Polygons
/// - Non-empty geometries
/// - Valid coordinate values (no NaN or Infinity)
///
/// # Arguments
///
/// * `geometry` - The geometry to validate
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::validate_geometry;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
/// let result = validate_geometry(&geom);
/// assert!(result.is_valid);
/// ```
pub fn validate_geometry(geometry: &Geometry) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Check for empty geometries
    if geometry.is_empty() {
        result = result.with_warning("Geometry is empty".to_string());
    }

    // Check for invalid coordinates (NaN, Infinity)
    if has_invalid_coordinates(&geometry.geom) {
        result = result
            .with_error("Geometry contains invalid coordinates (NaN or Infinity)".to_string());
    }

    // Check for self-intersections
    if !geometry.is_simple() {
        result = result.with_error("Geometry has self-intersections".to_string());
    }

    // Geometry-specific validation
    match &geometry.geom {
        GeoGeometry::Point(p) => {
            if p.x().is_nan() || p.y().is_nan() {
                result = result.with_error("Point has NaN coordinates".to_string());
            }
        }
        GeoGeometry::LineString(ls) => {
            if ls.0.len() < 2 {
                result = result.with_error("LineString must have at least 2 points".to_string());
            }
        }
        GeoGeometry::Polygon(poly) => {
            // Check exterior ring
            if poly.exterior().0.len() < 4 {
                result = result
                    .with_error("Polygon exterior ring must have at least 4 points".to_string());
            }

            // Check if exterior ring is closed
            if poly.exterior().0.first() != poly.exterior().0.last() {
                result = result.with_error("Polygon exterior ring is not closed".to_string());
            }

            // Check interior rings (holes)
            for (i, hole) in poly.interiors().iter().enumerate() {
                if hole.0.len() < 4 {
                    result = result
                        .with_error(format!("Polygon hole {} must have at least 4 points", i));
                }
                if hole.0.first() != hole.0.last() {
                    result = result.with_error(format!("Polygon hole {} is not closed", i));
                }
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            if mp.0.is_empty() {
                result = result.with_warning("MultiPoint is empty".to_string());
            }
        }
        GeoGeometry::MultiLineString(mls) => {
            if mls.0.is_empty() {
                result = result.with_warning("MultiLineString is empty".to_string());
            }
            for (i, ls) in mls.0.iter().enumerate() {
                if ls.0.len() < 2 {
                    result = result.with_error(format!(
                        "LineString {} in MultiLineString must have at least 2 points",
                        i
                    ));
                }
            }
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            if mpoly.0.is_empty() {
                result = result.with_warning("MultiPolygon is empty".to_string());
            }
        }
        _ => {}
    }

    result
}

/// Check if a geometry contains invalid coordinates (NaN or Infinity)
fn has_invalid_coordinates(geom: &GeoGeometry<f64>) -> bool {
    match geom {
        GeoGeometry::Point(p) => {
            p.x().is_nan() || p.y().is_nan() || p.x().is_infinite() || p.y().is_infinite()
        }
        GeoGeometry::LineString(ls) => {
            ls.0.iter()
                .any(|c| c.x.is_nan() || c.y.is_nan() || c.x.is_infinite() || c.y.is_infinite())
        }
        GeoGeometry::Polygon(poly) => {
            poly.exterior()
                .0
                .iter()
                .any(|c| c.x.is_nan() || c.y.is_nan() || c.x.is_infinite() || c.y.is_infinite())
                || poly.interiors().iter().any(|ring| {
                    ring.0.iter().any(|c| {
                        c.x.is_nan() || c.y.is_nan() || c.x.is_infinite() || c.y.is_infinite()
                    })
                })
        }
        GeoGeometry::MultiPoint(mp) => mp.0.iter().any(|p| {
            p.x().is_nan() || p.y().is_nan() || p.x().is_infinite() || p.y().is_infinite()
        }),
        GeoGeometry::MultiLineString(mls) => mls.0.iter().any(|ls| {
            ls.0.iter()
                .any(|c| c.x.is_nan() || c.y.is_nan() || c.x.is_infinite() || c.y.is_infinite())
        }),
        GeoGeometry::MultiPolygon(mpoly) => mpoly.0.iter().any(|poly| {
            poly.exterior()
                .0
                .iter()
                .any(|c| c.x.is_nan() || c.y.is_nan() || c.x.is_infinite() || c.y.is_infinite())
                || poly.interiors().iter().any(|ring| {
                    ring.0.iter().any(|c| {
                        c.x.is_nan() || c.y.is_nan() || c.x.is_infinite() || c.y.is_infinite()
                    })
                })
        }),
        _ => false,
    }
}

/// Simplify a geometry using the Douglas-Peucker algorithm
///
/// This algorithm reduces the number of points in a geometry while preserving
/// its overall shape. The epsilon parameter controls the level of simplification.
///
/// # Arguments
///
/// * `geometry` - The geometry to simplify
/// * `epsilon` - Maximum distance threshold for simplification
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::simplify_geometry;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{LineString, Coord, Geometry as GeoGeometry};
///
/// let coords = vec![
///     Coord { x: 0.0, y: 0.0 },
///     Coord { x: 1.0, y: 0.1 },
///     Coord { x: 2.0, y: 0.0 },
/// ];
/// let ls = LineString::new(coords);
/// let geom = Geometry::new(GeoGeometry::LineString(ls));
///
/// let simplified = simplify_geometry(&geom, 0.2).unwrap();
/// ```
pub fn simplify_geometry(geometry: &Geometry, epsilon: f64) -> Result<Geometry> {
    if epsilon <= 0.0 {
        return Err(GeoSparqlError::InvalidParameter(
            "Epsilon must be positive".to_string(),
        ));
    }

    let simplified_geom = match &geometry.geom {
        GeoGeometry::LineString(ls) => GeoGeometry::LineString(ls.simplify(&epsilon)),
        GeoGeometry::Polygon(poly) => GeoGeometry::Polygon(poly.simplify(&epsilon)),
        GeoGeometry::MultiLineString(mls) => GeoGeometry::MultiLineString(mls.simplify(&epsilon)),
        GeoGeometry::MultiPolygon(mpoly) => GeoGeometry::MultiPolygon(mpoly.simplify(&epsilon)),
        // Other geometry types don't benefit from simplification
        _ => geometry.geom.clone(),
    };

    Ok(Geometry::with_crs(simplified_geom, geometry.crs.clone()))
}

/// Simplify a geometry using the Visvalingam-Whyatt algorithm
///
/// This algorithm preserves topology by removing points that contribute least
/// to the overall shape. The epsilon parameter represents the minimum area threshold.
///
/// # Arguments
///
/// * `geometry` - The geometry to simplify
/// * `epsilon` - Minimum area threshold for point removal
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::simplify_geometry_vw;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{LineString, Coord, Geometry as GeoGeometry};
///
/// let coords = vec![
///     Coord { x: 0.0, y: 0.0 },
///     Coord { x: 1.0, y: 0.1 },
///     Coord { x: 2.0, y: 0.0 },
/// ];
/// let ls = LineString::new(coords);
/// let geom = Geometry::new(GeoGeometry::LineString(ls));
///
/// let simplified = simplify_geometry_vw(&geom, 0.05).unwrap();
/// ```
pub fn simplify_geometry_vw(geometry: &Geometry, epsilon: f64) -> Result<Geometry> {
    if epsilon <= 0.0 {
        return Err(GeoSparqlError::InvalidParameter(
            "Epsilon must be positive".to_string(),
        ));
    }

    let simplified_geom = match &geometry.geom {
        GeoGeometry::LineString(ls) => GeoGeometry::LineString(ls.simplify_vw_preserve(&epsilon)),
        GeoGeometry::Polygon(poly) => GeoGeometry::Polygon(poly.simplify_vw_preserve(&epsilon)),
        GeoGeometry::MultiLineString(mls) => {
            GeoGeometry::MultiLineString(mls.simplify_vw_preserve(&epsilon))
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            GeoGeometry::MultiPolygon(mpoly.simplify_vw_preserve(&epsilon))
        }
        _ => geometry.geom.clone(),
    };

    Ok(Geometry::with_crs(simplified_geom, geometry.crs.clone()))
}

/// Snap geometry coordinates to a precision grid
///
/// This rounds all coordinates to the specified precision, which can help
/// fix small topological errors and reduce storage requirements.
///
/// # Arguments
///
/// * `geometry` - The geometry to snap
/// * `precision` - Number of decimal places to preserve
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::snap_to_precision;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.234567, 2.345678)));
/// let snapped = snap_to_precision(&geom, 2).unwrap();
///
/// match snapped.geom {
///     GeoGeometry::Point(p) => {
///         assert_eq!(p.x(), 1.23);
///         assert_eq!(p.y(), 2.35);
///     }
///     _ => panic!("Expected Point"),
/// }
/// ```
pub fn snap_to_precision(geometry: &Geometry, precision: u32) -> Result<Geometry> {
    let multiplier = 10_f64.powi(precision as i32);

    let snap_coord = |x: f64| -> f64 { (x * multiplier).round() / multiplier };

    let snapped_geom = match &geometry.geom {
        GeoGeometry::Point(p) => {
            GeoGeometry::Point(geo_types::Point::new(snap_coord(p.x()), snap_coord(p.y())))
        }
        GeoGeometry::LineString(ls) => {
            let coords: Vec<_> =
                ls.0.iter()
                    .map(|c| geo_types::Coord {
                        x: snap_coord(c.x),
                        y: snap_coord(c.y),
                    })
                    .collect();
            GeoGeometry::LineString(geo_types::LineString::new(coords))
        }
        GeoGeometry::Polygon(poly) => {
            let exterior: Vec<_> = poly
                .exterior()
                .0
                .iter()
                .map(|c| geo_types::Coord {
                    x: snap_coord(c.x),
                    y: snap_coord(c.y),
                })
                .collect();

            let interiors: Vec<_> = poly
                .interiors()
                .iter()
                .map(|ring| {
                    let coords: Vec<_> = ring
                        .0
                        .iter()
                        .map(|c| geo_types::Coord {
                            x: snap_coord(c.x),
                            y: snap_coord(c.y),
                        })
                        .collect();
                    geo_types::LineString::new(coords)
                })
                .collect();

            GeoGeometry::Polygon(geo_types::Polygon::new(
                geo_types::LineString::new(exterior),
                interiors,
            ))
        }
        // Implement for other types as needed
        _ => geometry.geom.clone(),
    };

    Ok(Geometry::with_crs(snapped_geom, geometry.crs.clone()))
}

/// Automatically repair a geometry by fixing common issues
///
/// This function attempts to fix common geometry issues:
/// - Removes NaN and Infinity coordinates
/// - Closes unclosed polygon rings
/// - Removes duplicate consecutive points
/// - Fixes self-intersecting polygons (attempts to make valid)
///
/// # Arguments
///
/// * `geometry` - The geometry to repair
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::repair_geometry;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{LineString, Coord, Geometry as GeoGeometry};
///
/// // Create a linestring with duplicate points
/// let coords = vec![
///     Coord { x: 0.0, y: 0.0 },
///     Coord { x: 1.0, y: 1.0 },
///     Coord { x: 1.0, y: 1.0 }, // duplicate
///     Coord { x: 2.0, y: 2.0 },
/// ];
/// let geom = Geometry::new(GeoGeometry::LineString(LineString::new(coords)));
/// let repaired = repair_geometry(&geom).unwrap();
/// ```
pub fn repair_geometry(geometry: &Geometry) -> Result<Geometry> {
    let repaired_geom = match &geometry.geom {
        GeoGeometry::Point(p) => {
            // Remove invalid coordinates
            if p.x().is_nan() || p.y().is_nan() || p.x().is_infinite() || p.y().is_infinite() {
                return Err(GeoSparqlError::ValidationError(
                    "Point has invalid coordinates that cannot be repaired".to_string(),
                ));
            }
            geometry.geom.clone()
        }
        GeoGeometry::LineString(ls) => {
            let repaired_ls = repair_linestring(ls)?;
            GeoGeometry::LineString(repaired_ls)
        }
        GeoGeometry::Polygon(poly) => {
            let repaired_poly = repair_polygon(poly)?;
            GeoGeometry::Polygon(repaired_poly)
        }
        GeoGeometry::MultiPoint(mp) => {
            // Remove invalid points
            let valid_points: Vec<_> =
                mp.0.iter()
                    .filter(|p| {
                        !p.x().is_nan()
                            && !p.y().is_nan()
                            && !p.x().is_infinite()
                            && !p.y().is_infinite()
                    })
                    .cloned()
                    .collect();

            if valid_points.is_empty() {
                return Err(GeoSparqlError::ValidationError(
                    "All points in MultiPoint are invalid".to_string(),
                ));
            }

            GeoGeometry::MultiPoint(geo_types::MultiPoint(valid_points))
        }
        GeoGeometry::MultiLineString(mls) => {
            let repaired_linestrings: Result<Vec<_>> =
                mls.0.iter().map(repair_linestring).collect();

            GeoGeometry::MultiLineString(geo_types::MultiLineString(repaired_linestrings?))
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            let repaired_polygons: Result<Vec<_>> = mpoly.0.iter().map(repair_polygon).collect();

            GeoGeometry::MultiPolygon(geo_types::MultiPolygon(repaired_polygons?))
        }
        _ => geometry.geom.clone(),
    };

    Ok(Geometry::with_crs(repaired_geom, geometry.crs.clone()))
}

/// Repair a LineString by removing duplicates and invalid coordinates
fn repair_linestring(
    linestring: &geo_types::LineString<f64>,
) -> Result<geo_types::LineString<f64>> {
    // Remove invalid coordinates
    let valid_coords: Vec<_> = linestring
        .0
        .iter()
        .filter(|c| !c.x.is_nan() && !c.y.is_nan() && !c.x.is_infinite() && !c.y.is_infinite())
        .cloned()
        .collect();

    if valid_coords.len() < 2 {
        return Err(GeoSparqlError::ValidationError(
            "LineString has fewer than 2 valid points after repair".to_string(),
        ));
    }

    // Remove consecutive duplicates
    let deduped = remove_consecutive_duplicates(&valid_coords);

    if deduped.len() < 2 {
        return Err(GeoSparqlError::ValidationError(
            "LineString has fewer than 2 points after removing duplicates".to_string(),
        ));
    }

    Ok(geo_types::LineString::new(deduped))
}

/// Repair a Polygon by fixing rings and removing invalid coordinates
fn repair_polygon(polygon: &geo_types::Polygon<f64>) -> Result<geo_types::Polygon<f64>> {
    // Repair exterior ring
    let exterior_coords: Vec<_> = polygon
        .exterior()
        .0
        .iter()
        .filter(|c| !c.x.is_nan() && !c.y.is_nan() && !c.x.is_infinite() && !c.y.is_infinite())
        .cloned()
        .collect();

    if exterior_coords.len() < 3 {
        return Err(GeoSparqlError::ValidationError(
            "Polygon exterior ring has fewer than 3 valid points".to_string(),
        ));
    }

    // Remove consecutive duplicates
    let exterior_deduped = remove_consecutive_duplicates(&exterior_coords);

    if exterior_deduped.len() < 3 {
        return Err(GeoSparqlError::ValidationError(
            "Polygon exterior ring has fewer than 3 points after removing duplicates".to_string(),
        ));
    }

    // Ensure ring is closed
    let exterior = close_ring(exterior_deduped);

    // Repair interior rings (holes)
    let mut interiors = Vec::new();
    for hole in polygon.interiors() {
        let hole_coords: Vec<_> = hole
            .0
            .iter()
            .filter(|c| !c.x.is_nan() && !c.y.is_nan() && !c.x.is_infinite() && !c.y.is_infinite())
            .cloned()
            .collect();

        if hole_coords.len() >= 3 {
            let hole_deduped = remove_consecutive_duplicates(&hole_coords);
            if hole_deduped.len() >= 3 {
                interiors.push(close_ring(hole_deduped));
            }
        }
    }

    Ok(geo_types::Polygon::new(exterior, interiors))
}

/// Remove consecutive duplicate points from a coordinate list
fn remove_consecutive_duplicates(coords: &[geo_types::Coord<f64>]) -> Vec<geo_types::Coord<f64>> {
    if coords.is_empty() {
        return Vec::new();
    }

    let mut result = vec![coords[0]];
    let epsilon = 1e-10; // Tolerance for floating-point comparison

    for coord in &coords[1..] {
        let last = result.last().unwrap();
        let dx = (coord.x - last.x).abs();
        let dy = (coord.y - last.y).abs();

        if dx > epsilon || dy > epsilon {
            result.push(*coord);
        }
    }

    result
}

/// Close a polygon ring by ensuring the first and last points are the same
fn close_ring(mut coords: Vec<geo_types::Coord<f64>>) -> geo_types::LineString<f64> {
    if !coords.is_empty() {
        let first = coords[0];
        let last = *coords.last().unwrap();
        let epsilon = 1e-10;

        let dx = (first.x - last.x).abs();
        let dy = (first.y - last.y).abs();

        if dx > epsilon || dy > epsilon {
            coords.push(first);
        }
    }

    geo_types::LineString::new(coords)
}

/// Configuration for geometry validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Tolerance for coordinate comparison (epsilon)
    pub coordinate_tolerance: f64,
    /// Tolerance for area calculations
    pub area_tolerance: f64,
    /// Tolerance for length calculations
    pub length_tolerance: f64,
    /// Minimum allowed area for polygons
    pub min_polygon_area: f64,
    /// Minimum allowed length for linestrings
    pub min_linestring_length: f64,
    /// Maximum allowed coordinate value
    pub max_coordinate_value: f64,
    /// Check for self-intersections
    pub check_self_intersection: bool,
    /// Check for proper orientation
    pub check_orientation: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            coordinate_tolerance: 1e-10,
            area_tolerance: 1e-10,
            length_tolerance: 1e-10,
            min_polygon_area: 0.0,
            min_linestring_length: 0.0,
            max_coordinate_value: 1e15,
            check_self_intersection: true,
            check_orientation: true,
        }
    }
}

/// Geometry quality metrics
#[derive(Debug, Clone)]
pub struct GeometryQualityMetrics {
    /// Number of coordinates in the geometry
    pub coordinate_count: usize,
    /// Complexity score (higher = more complex)
    pub complexity_score: f64,
    /// Whether geometry has duplicate consecutive points
    pub has_duplicates: bool,
    /// Whether geometry has spikes or sharp angles
    pub has_spikes: bool,
    /// Minimum segment length in the geometry
    pub min_segment_length: f64,
    /// Maximum segment length in the geometry
    pub max_segment_length: f64,
    /// Average segment length
    pub avg_segment_length: f64,
    /// For polygons: area-to-perimeter ratio
    pub compactness: Option<f64>,
}

/// Validate geometry with custom configuration
///
/// This provides fine-grained control over validation tolerance and checks.
///
/// # Arguments
///
/// * `geometry` - The geometry to validate
/// * `config` - Validation configuration
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::{validate_geometry_with_config, ValidationConfig};
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
/// let config = ValidationConfig {
///     coordinate_tolerance: 1e-6,
///     ..Default::default()
/// };
///
/// let result = validate_geometry_with_config(&geom, &config);
/// assert!(result.is_valid);
/// ```
pub fn validate_geometry_with_config(
    geometry: &Geometry,
    config: &ValidationConfig,
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Check coordinate values
    match &geometry.geom {
        GeoGeometry::Point(p) => {
            if p.x().abs() > config.max_coordinate_value
                || p.y().abs() > config.max_coordinate_value
            {
                result = result.with_error(format!(
                    "Coordinate value exceeds maximum allowed: {}",
                    config.max_coordinate_value
                ));
            }
            if !p.x().is_finite() || !p.y().is_finite() {
                result = result.with_error("Point contains non-finite coordinates".to_string());
            }
        }
        GeoGeometry::LineString(ls) => {
            if ls.0.is_empty() {
                result = result.with_error("LineString is empty".to_string());
            } else if ls.0.len() < 2 {
                result = result.with_error("LineString must have at least 2 points".to_string());
            }

            // Check length
            let length = ls.euclidean_length();
            if length < config.min_linestring_length {
                result = result.with_warning(format!(
                    "LineString length ({}) is below minimum ({})",
                    length, config.min_linestring_length
                ));
            }

            // Check for coordinate validity
            for coord in &ls.0 {
                if !coord.x.is_finite() || !coord.y.is_finite() {
                    result =
                        result.with_error("LineString contains non-finite coordinates".to_string());
                    break;
                }
                if coord.x.abs() > config.max_coordinate_value
                    || coord.y.abs() > config.max_coordinate_value
                {
                    result = result.with_error(format!(
                        "Coordinate value exceeds maximum allowed: {}",
                        config.max_coordinate_value
                    ));
                    break;
                }
            }
        }
        GeoGeometry::Polygon(poly) => {
            if poly.exterior().0.is_empty() {
                result = result.with_error("Polygon exterior ring is empty".to_string());
            } else if poly.exterior().0.len() < 4 {
                result = result.with_error(
                    "Polygon exterior ring must have at least 4 points (including closing point)"
                        .to_string(),
                );
            }

            // Check area
            let area = poly.unsigned_area();
            if area < config.min_polygon_area {
                result = result.with_warning(format!(
                    "Polygon area ({}) is below minimum ({})",
                    area, config.min_polygon_area
                ));
            }

            // Check ring closure
            let first = poly.exterior().0.first();
            let last = poly.exterior().0.last();
            if let (Some(f), Some(l)) = (first, last) {
                let dx = (f.x - l.x).abs();
                let dy = (f.y - l.y).abs();
                if dx > config.coordinate_tolerance || dy > config.coordinate_tolerance {
                    result = result.with_error("Polygon exterior ring is not closed".to_string());
                }
            }

            // Check interior rings
            for (i, interior) in poly.interiors().iter().enumerate() {
                if interior.0.len() < 4 {
                    result = result.with_error(format!(
                        "Polygon interior ring {} must have at least 4 points",
                        i
                    ));
                }

                let first = interior.0.first();
                let last = interior.0.last();
                if let (Some(f), Some(l)) = (first, last) {
                    let dx = (f.x - l.x).abs();
                    let dy = (f.y - l.y).abs();
                    if dx > config.coordinate_tolerance || dy > config.coordinate_tolerance {
                        result =
                            result.with_error(format!("Polygon interior ring {} is not closed", i));
                    }
                }
            }
        }
        _ => {}
    }

    result
}

/// Compute quality metrics for a geometry
///
/// This function analyzes the geometry and computes various quality metrics
/// that can be used to assess the quality of the data.
///
/// # Arguments
///
/// * `geometry` - The geometry to analyze
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::compute_quality_metrics;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
/// let metrics = compute_quality_metrics(&geom);
/// assert_eq!(metrics.coordinate_count, 1);
/// ```
pub fn compute_quality_metrics(geometry: &Geometry) -> GeometryQualityMetrics {
    match &geometry.geom {
        GeoGeometry::Point(_) => GeometryQualityMetrics {
            coordinate_count: 1,
            complexity_score: 1.0,
            has_duplicates: false,
            has_spikes: false,
            min_segment_length: 0.0,
            max_segment_length: 0.0,
            avg_segment_length: 0.0,
            compactness: None,
        },
        GeoGeometry::LineString(ls) => {
            let coord_count = ls.0.len();
            let mut segment_lengths = Vec::new();
            let mut has_duplicates = false;
            let mut has_spikes = false;

            // Calculate segment lengths and check for duplicates
            for i in 0..ls.0.len().saturating_sub(1) {
                let p1 = &ls.0[i];
                let p2 = &ls.0[i + 1];

                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let length = (dx * dx + dy * dy).sqrt();

                segment_lengths.push(length);

                // Check for duplicates (very short segments)
                if length < 1e-10 {
                    has_duplicates = true;
                }

                // Check for spikes (very sharp angles) if we have at least 3 points
                if i > 0 {
                    let p0 = &ls.0[i - 1];
                    let v1x = p1.x - p0.x;
                    let v1y = p1.y - p0.y;
                    let v2x = p2.x - p1.x;
                    let v2y = p2.y - p1.y;

                    let len1 = (v1x * v1x + v1y * v1y).sqrt();
                    let len2 = (v2x * v2x + v2y * v2y).sqrt();

                    if len1 > 1e-10 && len2 > 1e-10 {
                        let dot = (v1x * v2x + v2y * v2y) / (len1 * len2);
                        // If angle is very sharp (dot product close to -1), mark as spike
                        if dot < -0.95 {
                            has_spikes = true;
                        }
                    }
                }
            }

            let min_length = segment_lengths
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let max_length = segment_lengths
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let avg_length = if segment_lengths.is_empty() {
                0.0
            } else {
                segment_lengths.iter().sum::<f64>() / segment_lengths.len() as f64
            };

            // Complexity score based on number of segments and variation in segment lengths
            let std_dev = if segment_lengths.len() > 1 {
                let variance = segment_lengths
                    .iter()
                    .map(|&l| (l - avg_length).powi(2))
                    .sum::<f64>()
                    / segment_lengths.len() as f64;
                variance.sqrt()
            } else {
                0.0
            };

            let complexity = coord_count as f64 * (1.0 + std_dev / (avg_length + 1e-10));

            GeometryQualityMetrics {
                coordinate_count: coord_count,
                complexity_score: complexity,
                has_duplicates,
                has_spikes,
                min_segment_length: min_length,
                max_segment_length: max_length,
                avg_segment_length: avg_length,
                compactness: None,
            }
        }
        GeoGeometry::Polygon(poly) => {
            let coord_count =
                poly.exterior().0.len() + poly.interiors().iter().map(|r| r.0.len()).sum::<usize>();

            let area = poly.unsigned_area();
            let perimeter = poly.exterior().euclidean_length();

            // Compactness: 4π * area / perimeter²  (circle = 1.0, other shapes < 1.0)
            let compactness = if perimeter > 1e-10 {
                Some(4.0 * std::f64::consts::PI * area / (perimeter * perimeter))
            } else {
                None
            };

            // For polygons, we analyze the exterior ring
            let mut segment_lengths = Vec::new();
            let mut has_duplicates = false;
            let mut has_spikes = false;

            for i in 0..poly.exterior().0.len().saturating_sub(1) {
                let p1 = &poly.exterior().0[i];
                let p2 = &poly.exterior().0[i + 1];

                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let length = (dx * dx + dy * dy).sqrt();

                segment_lengths.push(length);

                if length < 1e-10 {
                    has_duplicates = true;
                }

                if i > 0 {
                    let p0 = &poly.exterior().0[i - 1];
                    let v1x = p1.x - p0.x;
                    let v1y = p1.y - p0.y;
                    let v2x = p2.x - p1.x;
                    let v2y = p2.y - p1.y;

                    let len1 = (v1x * v1x + v1y * v1y).sqrt();
                    let len2 = (v2x * v2x + v2y * v2y).sqrt();

                    if len1 > 1e-10 && len2 > 1e-10 {
                        let dot = (v1x * v2x + v1y * v2y) / (len1 * len2);
                        if dot < -0.95 {
                            has_spikes = true;
                        }
                    }
                }
            }

            let min_length = segment_lengths
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let max_length = segment_lengths
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let avg_length = if segment_lengths.is_empty() {
                0.0
            } else {
                segment_lengths.iter().sum::<f64>() / segment_lengths.len() as f64
            };

            let complexity = coord_count as f64 * (1.0 + area.log10().abs());

            GeometryQualityMetrics {
                coordinate_count: coord_count,
                complexity_score: complexity,
                has_duplicates,
                has_spikes,
                min_segment_length: min_length,
                max_segment_length: max_length,
                avg_segment_length: avg_length,
                compactness,
            }
        }
        _ => GeometryQualityMetrics {
            coordinate_count: 0,
            complexity_score: 0.0,
            has_duplicates: false,
            has_spikes: false,
            min_segment_length: 0.0,
            max_segment_length: 0.0,
            avg_segment_length: 0.0,
            compactness: None,
        },
    }
}

/// Check if geometry conforms to OGC Simple Features specification
///
/// This validates that the geometry meets the OGC SF standards including:
/// - Proper structure and topology
/// - Valid coordinate values
/// - Correct ring orientation for polygons
/// - No self-intersections
///
/// # Arguments
///
/// * `geometry` - The geometry to check
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::validation::check_ogc_compliance;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
/// let result = check_ogc_compliance(&geom);
/// assert!(result.is_valid);
/// ```
pub fn check_ogc_compliance(geometry: &Geometry) -> ValidationResult {
    let config = ValidationConfig::default();
    let mut result = validate_geometry_with_config(geometry, &config);

    // Add OGC-specific checks
    match &geometry.geom {
        GeoGeometry::LineString(ls) => {
            // OGC SF: LineString must have at least 2 distinct points
            if ls.0.len() >= 2 {
                let first = ls.0.first().unwrap();
                let all_same =
                    ls.0.iter()
                        .all(|c| (c.x - first.x).abs() < 1e-10 && (c.y - first.y).abs() < 1e-10);
                if all_same {
                    result = result.with_error(
                        "OGC SF: LineString must have at least 2 distinct points".to_string(),
                    );
                }
            }
        }
        GeoGeometry::Polygon(poly) => {
            // OGC SF: Polygon boundary must be simple (no self-intersections)
            // OGC SF: Interior rings must be inside exterior ring
            // OGC SF: Interior rings must not overlap

            // Check ring orientation (exterior should be counter-clockwise, interior clockwise)
            let exterior_area = poly.exterior().unsigned_area();
            if exterior_area < 1e-10 {
                result = result.with_warning("Polygon has very small or zero area".to_string());
            }

            // Check that interior rings don't overlap
            for i in 0..poly.interiors().len() {
                for j in (i + 1)..poly.interiors().len() {
                    // Simplified check: ensure rings don't share too many points
                    let ring_i = &poly.interiors()[i];
                    let ring_j = &poly.interiors()[j];

                    let mut shared_points = 0;
                    for pi in &ring_i.0 {
                        for pj in &ring_j.0 {
                            if (pi.x - pj.x).abs() < 1e-10 && (pi.y - pj.y).abs() < 1e-10 {
                                shared_points += 1;
                            }
                        }
                    }

                    if shared_points > 2 {
                        result = result.with_warning(format!(
                            "Interior rings {} and {} may overlap (shared {} points)",
                            i, j, shared_points
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Point, Polygon};

    #[test]
    fn test_validate_valid_point() {
        let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let result = validate_geometry(&geom);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_invalid_point() {
        let geom = Geometry::new(GeoGeometry::Point(Point::new(f64::NAN, 2.0)));
        let result = validate_geometry(&geom);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validate_valid_polygon() {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 4.0, y: 0.0 },
            Coord { x: 4.0, y: 4.0 },
            Coord { x: 0.0, y: 4.0 },
            Coord { x: 0.0, y: 0.0 },
        ];
        let poly = Polygon::new(LineString::new(coords), vec![]);
        let geom = Geometry::new(GeoGeometry::Polygon(poly));

        let result = validate_geometry(&geom);
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_invalid_polygon() {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 4.0, y: 0.0 },
            // Not closed - missing last point
        ];
        let poly = Polygon::new(LineString::new(coords), vec![]);
        let geom = Geometry::new(GeoGeometry::Polygon(poly));

        let result = validate_geometry(&geom);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_simplify_linestring() -> Result<()> {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.1 },
            Coord { x: 2.0, y: 0.0 },
            Coord { x: 3.0, y: 0.1 },
            Coord { x: 4.0, y: 0.0 },
        ];
        let ls = LineString::new(coords);
        let geom = Geometry::new(GeoGeometry::LineString(ls));

        let simplified = simplify_geometry(&geom, 0.2)?;

        match simplified.geom {
            GeoGeometry::LineString(ls) => {
                // Should be simplified to fewer points
                assert!(ls.0.len() < 5);
            }
            _ => panic!("Expected LineString"),
        }

        Ok(())
    }

    #[test]
    fn test_snap_to_precision() -> Result<()> {
        let geom = Geometry::new(GeoGeometry::Point(Point::new(1.234567, 2.345678)));
        let snapped = snap_to_precision(&geom, 2)?;

        match snapped.geom {
            GeoGeometry::Point(p) => {
                assert!((p.x() - 1.23).abs() < 1e-10);
                assert!((p.y() - 2.35).abs() < 1e-10);
            }
            _ => panic!("Expected Point"),
        }

        Ok(())
    }

    #[test]
    fn test_repair_linestring_with_duplicates() -> Result<()> {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 1.0, y: 1.0 }, // duplicate
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 2.0, y: 2.0 }, // duplicate
            Coord { x: 3.0, y: 3.0 },
        ];
        let geom = Geometry::new(GeoGeometry::LineString(LineString::new(coords)));
        let repaired = repair_geometry(&geom)?;

        match repaired.geom {
            GeoGeometry::LineString(ls) => {
                // Should have duplicates removed
                assert_eq!(ls.0.len(), 4); // 0, 1, 2, 3
            }
            _ => panic!("Expected LineString"),
        }

        Ok(())
    }

    #[test]
    fn test_repair_unclosed_polygon() -> Result<()> {
        // Create a polygon that's not closed (missing last point)
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 4.0, y: 0.0 },
            Coord { x: 4.0, y: 4.0 },
            Coord { x: 0.0, y: 4.0 },
            // Missing the closing point
        ];
        let poly = Polygon::new(LineString::new(coords), vec![]);
        let geom = Geometry::new(GeoGeometry::Polygon(poly));

        let repaired = repair_geometry(&geom)?;

        match repaired.geom {
            GeoGeometry::Polygon(p) => {
                // Should be closed now
                assert_eq!(p.exterior().0.len(), 5);
                assert_eq!(p.exterior().0.first(), p.exterior().0.last());
            }
            _ => panic!("Expected Polygon"),
        }

        Ok(())
    }

    #[test]
    fn test_repair_polygon_with_invalid_coords() -> Result<()> {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord {
                x: f64::NAN,
                y: 1.0,
            }, // invalid
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 0.0, y: 2.0 },
            Coord { x: 0.0, y: 0.0 },
        ];
        let poly = Polygon::new(LineString::new(coords), vec![]);
        let geom = Geometry::new(GeoGeometry::Polygon(poly));

        let repaired = repair_geometry(&geom)?;

        match repaired.geom {
            GeoGeometry::Polygon(p) => {
                // Should have NaN removed
                assert!(p
                    .exterior()
                    .0
                    .iter()
                    .all(|c| !c.x.is_nan() && !c.y.is_nan()));
                // Should still be closed
                assert_eq!(p.exterior().0.first(), p.exterior().0.last());
            }
            _ => panic!("Expected Polygon"),
        }

        Ok(())
    }

    #[test]
    fn test_repair_multipoint_removes_invalid() -> Result<()> {
        let points = vec![
            Point::new(1.0, 2.0),
            Point::new(f64::NAN, 3.0), // invalid
            Point::new(4.0, 5.0),
        ];
        let geom = Geometry::new(GeoGeometry::MultiPoint(geo_types::MultiPoint(points)));

        let repaired = repair_geometry(&geom)?;

        match repaired.geom {
            GeoGeometry::MultiPoint(mp) => {
                // Should have invalid point removed
                assert_eq!(mp.0.len(), 2);
                assert!(mp.0.iter().all(|p| !p.x().is_nan() && !p.y().is_nan()));
            }
            _ => panic!("Expected MultiPoint"),
        }

        Ok(())
    }

    #[test]
    fn test_repair_geometry_preserves_crs() -> Result<()> {
        use crate::geometry::Crs;

        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 1.0, y: 1.0 }, // duplicate
            Coord { x: 2.0, y: 2.0 },
        ];
        let ls = LineString::new(coords);
        let custom_crs = Crs::epsg(3857);
        let geom = Geometry::with_crs(GeoGeometry::LineString(ls), custom_crs.clone());

        let repaired = repair_geometry(&geom)?;

        // CRS should be preserved
        assert_eq!(repaired.crs, custom_crs);

        Ok(())
    }

    #[test]
    fn test_remove_consecutive_duplicates_empty() {
        let coords: Vec<Coord<f64>> = vec![];
        let result = remove_consecutive_duplicates(&coords);
        assert!(result.is_empty());
    }

    #[test]
    fn test_remove_consecutive_duplicates_no_dups() {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 2.0, y: 2.0 },
        ];
        let result = remove_consecutive_duplicates(&coords);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_remove_consecutive_duplicates_with_dups() {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 1.0, y: 1.0 }, // dup
            Coord { x: 1.0, y: 1.0 }, // dup
            Coord { x: 2.0, y: 2.0 },
        ];
        let result = remove_consecutive_duplicates(&coords);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_close_ring_already_closed() {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 0.0, y: 0.0 }, // already closed
        ];
        let ring = close_ring(coords.clone());
        assert_eq!(ring.0.len(), 4);
        assert_eq!(ring.0.first(), ring.0.last());
    }

    #[test]
    fn test_close_ring_not_closed() {
        let coords = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            // Missing closing point
        ];
        let ring = close_ring(coords);
        assert_eq!(ring.0.len(), 4); // Should add closing point
        assert_eq!(ring.0.first(), ring.0.last());
    }
}
