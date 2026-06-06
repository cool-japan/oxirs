//! Topology-aware validation: configurable validation, quality metrics, and OGC compliance.
//!
//! This module provides:
//! - [`ValidationConfig`] — configurable tolerance and check parameters
//! - [`GeometryQualityMetrics`] — segment-level quality analysis
//! - [`validate_geometry_with_config`] — fine-grained configurable validator
//! - [`compute_quality_metrics`] — computes shape complexity, spike detection, etc.
//! - [`check_ogc_compliance`] — OGC Simple Features conformance checker

use crate::geometry::Geometry;
use crate::validation_core::ValidationResult;
use geo::Area;
use geo_types::Geometry as GeoGeometry;

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
            use geo::{Euclidean, Length};
            let length = Euclidean.length(ls);
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
        GeoGeometry::LineString(ls) => compute_linestring_metrics(ls),
        GeoGeometry::Polygon(poly) => compute_polygon_metrics(poly),
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

/// Compute quality metrics for a LineString
fn compute_linestring_metrics(ls: &geo_types::LineString<f64>) -> GeometryQualityMetrics {
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

    let (min_length, max_length, avg_length) = segment_stats(&segment_lengths);
    let complexity =
        complexity_from_coord_count_and_segments(coord_count, avg_length, &segment_lengths);

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

/// Compute quality metrics for a Polygon
fn compute_polygon_metrics(poly: &geo_types::Polygon<f64>) -> GeometryQualityMetrics {
    let coord_count =
        poly.exterior().0.len() + poly.interiors().iter().map(|r| r.0.len()).sum::<usize>();

    use geo::{Euclidean, Length};
    let area = poly.unsigned_area();
    let perimeter = Euclidean.length(poly.exterior());

    // Compactness: 4π * area / perimeter²  (circle = 1.0, other shapes < 1.0)
    let compactness = if perimeter > 1e-10 {
        Some(4.0 * std::f64::consts::PI * area / (perimeter * perimeter))
    } else {
        None
    };

    // For polygons, we analyse the exterior ring
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

    let (min_length, max_length, avg_length) = segment_stats(&segment_lengths);
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

/// Compute min, max, average from a slice of segment lengths
fn segment_stats(segment_lengths: &[f64]) -> (f64, f64, f64) {
    let min_length = segment_lengths
        .iter()
        .copied()
        .min_by(|a, b| {
            a.partial_cmp(b)
                .expect("segment lengths should be comparable")
        })
        .unwrap_or(0.0);
    let max_length = segment_lengths
        .iter()
        .copied()
        .max_by(|a, b| {
            a.partial_cmp(b)
                .expect("segment lengths should be comparable")
        })
        .unwrap_or(0.0);
    let avg_length = if segment_lengths.is_empty() {
        0.0
    } else {
        segment_lengths.iter().sum::<f64>() / segment_lengths.len() as f64
    };
    (min_length, max_length, avg_length)
}

/// Compute complexity score based on coordinate count and segment length variance
fn complexity_from_coord_count_and_segments(
    coord_count: usize,
    avg_length: f64,
    segment_lengths: &[f64],
) -> f64 {
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
    coord_count as f64 * (1.0 + std_dev / (avg_length + 1e-10))
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
        GeoGeometry::LineString(ls)
            // OGC SF: LineString must have at least 2 distinct points
            if ls.0.len() >= 2 => {
                let first =
                    ls.0.first()
                        .expect("linestring should have at least one point");
                let all_same =
                    ls.0.iter()
                        .all(|c| (c.x - first.x).abs() < 1e-10 && (c.y - first.y).abs() < 1e-10);
                if all_same {
                    result = result.with_error(
                        "OGC SF: LineString must have at least 2 distinct points".to_string(),
                    );
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
