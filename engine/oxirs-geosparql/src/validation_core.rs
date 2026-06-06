//! Core validation types and basic geometry validity checks.
//!
//! This module contains the [`ValidationResult`] struct and the fundamental
//! validity check function [`validate_geometry`], along with the coordinate-
//! invalidity helper used across validation sub-modules.

use crate::geometry::Geometry;
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
        GeoGeometry::Point(p) if (p.x().is_nan() || p.y().is_nan()) => {
            result = result.with_error("Point has NaN coordinates".to_string());
        }
        GeoGeometry::LineString(ls) if ls.0.len() < 2 => {
            result = result.with_error("LineString must have at least 2 points".to_string());
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
        GeoGeometry::MultiPoint(mp) if mp.0.is_empty() => {
            result = result.with_warning("MultiPoint is empty".to_string());
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
        GeoGeometry::MultiPolygon(mpoly) if mpoly.0.is_empty() => {
            result = result.with_warning("MultiPolygon is empty".to_string());
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
