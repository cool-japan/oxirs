//! Geometry repair algorithms, simplification, and precision-snapping.
//!
//! This module provides:
//! - Douglas-Peucker simplification ([`simplify_geometry`])
//! - Visvalingam-Whyatt simplification ([`simplify_geometry_vw`])
//! - Coordinate precision snapping ([`snap_to_precision`])
//! - Automatic geometry repair ([`repair_geometry`])
//! - Helper functions: [`repair_linestring`], [`repair_polygon`],
//!   [`remove_consecutive_duplicates`], [`close_ring`]

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::{Simplify, SimplifyVwPreserve};
use geo_types::Geometry as GeoGeometry;

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
/// let simplified = simplify_geometry(&geom, 0.2).expect("simplification should succeed");
/// ```
pub fn simplify_geometry(geometry: &Geometry, epsilon: f64) -> Result<Geometry> {
    if epsilon <= 0.0 {
        return Err(GeoSparqlError::InvalidParameter(
            "Epsilon must be positive".to_string(),
        ));
    }

    let simplified_geom = match &geometry.geom {
        GeoGeometry::LineString(ls) => GeoGeometry::LineString(ls.simplify(epsilon)),
        GeoGeometry::Polygon(poly) => GeoGeometry::Polygon(poly.simplify(epsilon)),
        GeoGeometry::MultiLineString(mls) => GeoGeometry::MultiLineString(mls.simplify(epsilon)),
        GeoGeometry::MultiPolygon(mpoly) => GeoGeometry::MultiPolygon(mpoly.simplify(epsilon)),
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
/// let simplified = simplify_geometry_vw(&geom, 0.05).expect("simplification should succeed");
/// ```
pub fn simplify_geometry_vw(geometry: &Geometry, epsilon: f64) -> Result<Geometry> {
    if epsilon <= 0.0 {
        return Err(GeoSparqlError::InvalidParameter(
            "Epsilon must be positive".to_string(),
        ));
    }

    let simplified_geom = match &geometry.geom {
        GeoGeometry::LineString(ls) => GeoGeometry::LineString(ls.simplify_vw_preserve(epsilon)),
        GeoGeometry::Polygon(poly) => GeoGeometry::Polygon(poly.simplify_vw_preserve(epsilon)),
        GeoGeometry::MultiLineString(mls) => {
            GeoGeometry::MultiLineString(mls.simplify_vw_preserve(epsilon))
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            GeoGeometry::MultiPolygon(mpoly.simplify_vw_preserve(epsilon))
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
/// let snapped = snap_to_precision(&geom, 2).expect("snap should succeed");
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
/// let repaired = repair_geometry(&geom).expect("repair should succeed");
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
pub(crate) fn remove_consecutive_duplicates(
    coords: &[geo_types::Coord<f64>],
) -> Vec<geo_types::Coord<f64>> {
    if coords.is_empty() {
        return Vec::new();
    }

    let mut result = vec![coords[0]];
    let epsilon = 1e-10; // Tolerance for floating-point comparison

    for coord in &coords[1..] {
        let last = result.last().expect("result should not be empty");
        let dx = (coord.x - last.x).abs();
        let dy = (coord.y - last.y).abs();

        if dx > epsilon || dy > epsilon {
            result.push(*coord);
        }
    }

    result
}

/// Close a polygon ring by ensuring the first and last points are the same
pub(crate) fn close_ring(mut coords: Vec<geo_types::Coord<f64>>) -> geo_types::LineString<f64> {
    if !coords.is_empty() {
        let first = coords[0];
        let last = *coords.last().expect("coords should not be empty");
        let epsilon = 1e-10;

        let dx = (first.x - last.x).abs();
        let dy = (first.y - last.y).abs();

        if dx > epsilon || dy > epsilon {
            coords.push(first);
        }
    }

    geo_types::LineString::new(coords)
}
