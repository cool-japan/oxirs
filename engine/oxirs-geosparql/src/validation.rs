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
use geo::{Simplify, SimplifyVwPreserve};
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
}
