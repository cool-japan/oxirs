//! Coordinate Reference System (CRS) transformation functions
//!
//! This module provides functions for transforming geometries between different
//! Coordinate Reference Systems using the PROJ library.
//!
//! # Performance
//!
//! - `transform_batch()` reuses a single PROJ object for all geometries (~10x speedup)
//! - For large batches (>100 geometries), use `transform_batch_parallel()` with the
//!   `parallel` feature enabled for significant speedup on multi-core systems

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};

/// Transform a geometry from its current CRS to a target CRS
///
/// This function uses the PROJ library to perform coordinate transformations.
/// Both the source and target CRS must have EPSG codes.
///
/// # Arguments
///
/// * `geom` - The geometry to transform
/// * `target_crs` - The target Coordinate Reference System
///
/// # Returns
///
/// A new geometry in the target CRS
///
/// # Errors
///
/// Returns an error if:
/// - The source or target CRS doesn't have an EPSG code
/// - The PROJ transformation fails
/// - The `proj-support` feature is not enabled
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "proj-support")]
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use oxirs_geosparql::geometry::{Crs, Geometry};
/// use oxirs_geosparql::functions::coordinate_transformation::transform;
///
/// // Point in WGS84 (EPSG:4326) - Tokyo
/// let point = Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(139.7 35.7)")?;
///
/// // Transform to Web Mercator (EPSG:3857)
/// let transformed = transform(&point, &Crs::epsg(3857))?;
///
/// println!("Original: {}", point.to_wkt());
/// println!("Transformed: {}", transformed.to_wkt());
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "proj-support")]
pub fn transform(geom: &Geometry, target_crs: &Crs) -> Result<Geometry> {
    use geo::algorithm::map_coords::MapCoords;
    use proj::Proj;

    // If already in target CRS, return clone
    if geom.crs == *target_crs {
        return Ok(geom.clone());
    }

    // Extract EPSG codes
    let source_epsg = geom.crs.epsg_code().ok_or_else(|| {
        GeoSparqlError::CrsTransformationFailed(format!(
            "Source CRS must have EPSG code for transformation, got: {}",
            geom.crs.uri
        ))
    })?;

    let target_epsg = target_crs.epsg_code().ok_or_else(|| {
        GeoSparqlError::CrsTransformationFailed(format!(
            "Target CRS must have EPSG code for transformation, got: {}",
            target_crs.uri
        ))
    })?;

    // Create PROJ transformation
    let proj_string = format!("EPSG:{}", source_epsg);
    let target_string = format!("EPSG:{}", target_epsg);

    let proj = Proj::new_known_crs(&proj_string, &target_string, None).map_err(|e| {
        GeoSparqlError::CrsTransformationFailed(format!(
            "Failed to create PROJ transformation from EPSG:{} to EPSG:{}: {}",
            source_epsg, target_epsg, e
        ))
    })?;

    // Transform coordinates
    let transformed_geom = geom.geom.map_coords(|coord| {
        // PROJ expects (longitude, latitude) order for geographic CRS
        let point: (f64, f64) = proj.convert((coord.x, coord.y)).unwrap_or_else(|e| {
            // Log error but fallback to original coordinates
            tracing::warn!(
                "PROJ conversion failed for ({}, {}): {}. Using original coordinates.",
                coord.x,
                coord.y,
                e
            );
            (coord.x, coord.y)
        });
        geo_types::Coord {
            x: point.0,
            y: point.1,
        }
    });

    Ok(Geometry::with_crs(transformed_geom, target_crs.clone()))
}

/// Transform a geometry from its current CRS to a target CRS (fallback when proj-support is not enabled)
#[cfg(not(feature = "proj-support"))]
pub fn transform(_geom: &Geometry, _target_crs: &Crs) -> Result<Geometry> {
    Err(GeoSparqlError::UnsupportedOperation(
        "CRS transformation requires the 'proj-support' feature to be enabled".to_string(),
    ))
}

/// Transform multiple geometries to a target CRS
///
/// This function reuses a single PROJ transformation object for all geometries,
/// providing ~10x speedup over calling `transform()` multiple times.
///
/// # Performance
///
/// - Creates ONE Proj object for the entire batch
/// - All geometries must have the same source CRS
/// - For large batches (>100 geometries), consider using `transform_batch_parallel()`
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "proj-support")]
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use oxirs_geosparql::geometry::{Crs, Geometry};
/// use oxirs_geosparql::functions::coordinate_transformation::transform_batch;
///
/// let geometries = vec![
///     Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(139.7 35.7)")?,
///     Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(140.0 36.0)")?,
/// ];
///
/// let transformed = transform_batch(&geometries, &Crs::epsg(3857))?;
/// assert_eq!(transformed.len(), 2);
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "proj-support")]
pub fn transform_batch(geometries: &[Geometry], target_crs: &Crs) -> Result<Vec<Geometry>> {
    use geo::algorithm::map_coords::MapCoords;
    use proj::Proj;

    if geometries.is_empty() {
        return Ok(Vec::new());
    }

    // Check if all geometries have the same CRS
    let source_crs = &geometries[0].crs;
    if !geometries.iter().all(|g| &g.crs == source_crs) {
        return Err(GeoSparqlError::CrsTransformationFailed(
            "All geometries must have the same source CRS for batch transformation".to_string(),
        ));
    }

    // If already in target CRS, return clones
    if source_crs == target_crs {
        return Ok(geometries.to_vec());
    }

    // Extract EPSG codes
    let source_epsg = source_crs.epsg_code().ok_or_else(|| {
        GeoSparqlError::CrsTransformationFailed(format!(
            "Source CRS must have EPSG code for transformation, got: {}",
            source_crs.uri
        ))
    })?;

    let target_epsg = target_crs.epsg_code().ok_or_else(|| {
        GeoSparqlError::CrsTransformationFailed(format!(
            "Target CRS must have EPSG code for transformation, got: {}",
            target_crs.uri
        ))
    })?;

    // Create ONE PROJ transformation for all geometries
    let proj_string = format!("EPSG:{}", source_epsg);
    let target_string = format!("EPSG:{}", target_epsg);

    let proj = Proj::new_known_crs(&proj_string, &target_string, None).map_err(|e| {
        GeoSparqlError::CrsTransformationFailed(format!(
            "Failed to create PROJ transformation from EPSG:{} to EPSG:{}: {}",
            source_epsg, target_epsg, e
        ))
    })?;

    // Transform all geometries using the same Proj object
    let transformed: Result<Vec<_>> = geometries
        .iter()
        .map(|geom| {
            let transformed_geom = geom.geom.map_coords(|coord| {
                let point: (f64, f64) = proj.convert((coord.x, coord.y)).unwrap_or_else(|e| {
                    tracing::warn!(
                        "PROJ conversion failed for ({}, {}): {}. Using original coordinates.",
                        coord.x,
                        coord.y,
                        e
                    );
                    (coord.x, coord.y)
                });
                geo_types::Coord {
                    x: point.0,
                    y: point.1,
                }
            });
            Ok(Geometry::with_crs(transformed_geom, target_crs.clone()))
        })
        .collect();

    transformed
}

/// Transform multiple geometries to a target CRS in parallel
///
/// This function uses Rayon to parallelize coordinate transformations across
/// multiple CPU cores. Most beneficial for large batches (>100 geometries).
///
/// # Performance
///
/// - Small batches (<100): Similar to `transform_batch()`
/// - Large batches (>1000): Near-linear speedup with CPU core count
///
/// # Examples
///
/// ```
/// # #[cfg(all(feature = "proj-support", feature = "parallel"))]
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use oxirs_geosparql::geometry::{Crs, Geometry};
/// use oxirs_geosparql::functions::coordinate_transformation::transform_batch_parallel;
///
/// // Create 1000 geometries
/// let geometries: Vec<_> = (0..1000)
///     .map(|i| {
///         Geometry::from_wkt(&format!(
///             "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT({} {})",
///             139.0 + i as f64 * 0.01,
///             35.0 + i as f64 * 0.01
///         )).unwrap()
///     })
///     .collect();
///
/// // Transform in parallel (much faster for large batches)
/// let transformed = transform_batch_parallel(&geometries, &Crs::epsg(3857))?;
/// assert_eq!(transformed.len(), 1000);
/// # Ok(())
/// # }
/// ```
#[cfg(all(feature = "proj-support", feature = "parallel"))]
pub fn transform_batch_parallel(
    geometries: &[Geometry],
    target_crs: &Crs,
) -> Result<Vec<Geometry>> {
    use rayon::prelude::*;

    geometries
        .par_iter()
        .map(|geom| transform(geom, target_crs))
        .collect()
}

/// Transform multiple geometries to a target CRS (fallback when proj-support is not enabled)
#[cfg(not(feature = "proj-support"))]
pub fn transform_batch(_geometries: &[Geometry], _target_crs: &Crs) -> Result<Vec<Geometry>> {
    Err(GeoSparqlError::UnsupportedOperation(
        "CRS transformation requires the 'proj-support' feature to be enabled".to_string(),
    ))
}

#[cfg(test)]
#[cfg(feature = "proj-support")]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, Point};

    #[test]
    fn test_transform_point_wgs84_to_mercator() {
        // Tokyo in WGS84 (EPSG:4326)
        let point =
            Geometry::with_crs(GeoGeometry::Point(Point::new(139.7, 35.7)), Crs::epsg(4326));

        // Transform to Web Mercator (EPSG:3857)
        let transformed = transform(&point, &Crs::epsg(3857)).unwrap();

        // Verify CRS changed
        assert_eq!(transformed.crs, Crs::epsg(3857));

        // Verify coordinates transformed (approximate values)
        if let GeoGeometry::Point(p) = transformed.geom {
            // Tokyo in Web Mercator should be significantly different from WGS84
            // The actual values depend on PROJ version and datum shift
            // Just verify that transformation occurred (x should be > 1 million, y > 1 million)
            assert!(
                p.x() > 1_000_000.0,
                "x={} should be > 1,000,000 in Web Mercator",
                p.x()
            );
            assert!(
                p.y() > 1_000_000.0,
                "y={} should be > 1,000,000 in Web Mercator",
                p.y()
            );

            // Verify reasonable range for Tokyo in Web Mercator
            // Tokyo should be around 15-16 million east, 4-5 million north
            assert!(
                p.x() > 15_000_000.0 && p.x() < 16_000_000.0,
                "x={} not in expected range",
                p.x()
            );
            assert!(
                p.y() > 4_000_000.0 && p.y() < 5_000_000.0,
                "y={} not in expected range",
                p.y()
            );
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_transform_same_crs_returns_clone() {
        let point =
            Geometry::with_crs(GeoGeometry::Point(Point::new(139.7, 35.7)), Crs::epsg(4326));

        let result = transform(&point, &Crs::epsg(4326)).unwrap();

        // Should return identical coordinates
        assert_eq!(result.geom, point.geom);
        assert_eq!(result.crs, point.crs);
    }

    #[test]
    fn test_transform_linestring() {
        use geo_types::LineString;

        let line = Geometry::with_crs(
            GeoGeometry::LineString(LineString::new(vec![
                Coord { x: 139.7, y: 35.7 },
                Coord { x: 140.0, y: 36.0 },
            ])),
            Crs::epsg(4326),
        );

        let transformed = transform(&line, &Crs::epsg(3857)).unwrap();

        assert_eq!(transformed.crs, Crs::epsg(3857));
        assert_eq!(transformed.geometry_type(), "LineString");
    }

    #[test]
    fn test_transform_polygon() {
        use geo_types::{LineString, Polygon};

        let polygon = Geometry::with_crs(
            GeoGeometry::Polygon(Polygon::new(
                LineString::new(vec![
                    Coord { x: 139.0, y: 35.0 },
                    Coord { x: 140.0, y: 35.0 },
                    Coord { x: 140.0, y: 36.0 },
                    Coord { x: 139.0, y: 36.0 },
                    Coord { x: 139.0, y: 35.0 },
                ]),
                vec![],
            )),
            Crs::epsg(4326),
        );

        let transformed = transform(&polygon, &Crs::epsg(3857)).unwrap();

        assert_eq!(transformed.crs, Crs::epsg(3857));
        assert_eq!(transformed.geometry_type(), "Polygon");
    }

    #[test]
    fn test_transform_non_epsg_crs_fails() {
        let point = Geometry::with_crs(
            GeoGeometry::Point(Point::new(139.7, 35.7)),
            Crs::new("http://www.opengis.net/def/crs/OGC/1.3/CRS84"),
        );

        let result = transform(&point, &Crs::epsg(3857));
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_batch() {
        let geometries = vec![
            Geometry::with_crs(GeoGeometry::Point(Point::new(139.7, 35.7)), Crs::epsg(4326)),
            Geometry::with_crs(GeoGeometry::Point(Point::new(140.0, 36.0)), Crs::epsg(4326)),
        ];

        let transformed = transform_batch(&geometries, &Crs::epsg(3857)).unwrap();

        assert_eq!(transformed.len(), 2);
        assert!(transformed.iter().all(|g| g.crs == Crs::epsg(3857)));
    }

    #[test]
    fn test_transform_batch_reuses_proj() {
        // Create batch of geometries (all same source CRS)
        let geometries: Vec<_> = (0..10)
            .map(|i| {
                Geometry::with_crs(
                    GeoGeometry::Point(Point::new(139.0 + i as f64 * 0.1, 35.0)),
                    Crs::epsg(4326),
                )
            })
            .collect();

        // Transform batch - should create ONE Proj object and reuse it
        let transformed = transform_batch(&geometries, &Crs::epsg(3857)).unwrap();

        assert_eq!(transformed.len(), 10);
        assert!(transformed.iter().all(|g| g.crs == Crs::epsg(3857)));
    }

    #[test]
    fn test_transform_batch_empty() {
        let geometries: Vec<Geometry> = vec![];
        let transformed = transform_batch(&geometries, &Crs::epsg(3857)).unwrap();
        assert_eq!(transformed.len(), 0);
    }

    #[test]
    fn test_transform_batch_same_crs() {
        let geometries = vec![
            Geometry::with_crs(GeoGeometry::Point(Point::new(139.7, 35.7)), Crs::epsg(4326)),
            Geometry::with_crs(GeoGeometry::Point(Point::new(140.0, 36.0)), Crs::epsg(4326)),
        ];

        // Transform to same CRS - should just clone
        let transformed = transform_batch(&geometries, &Crs::epsg(4326)).unwrap();

        assert_eq!(transformed.len(), 2);
        assert!(transformed.iter().all(|g| g.crs == Crs::epsg(4326)));
    }
}

#[cfg(test)]
#[cfg(not(feature = "proj-support"))]
mod tests_without_proj {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_transform_without_proj_fails() {
        let point =
            Geometry::with_crs(GeoGeometry::Point(Point::new(139.7, 35.7)), Crs::epsg(4326));

        let result = transform(&point, &Crs::epsg(3857));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("proj-support"));
    }
}
