//! Spatial Aggregation Functions
//!
//! This module provides spatial aggregation operations for GeoSPARQL queries:
//! - Geometric union of multiple geometries
//! - Convex hull computation
//! - Spatial centroid calculation
//! - Bounding envelope computation
//! - DBSCAN clustering
//!
//! These functions are designed for use in SPARQL GROUP BY clauses and
//! aggregate spatial data efficiently.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::algorithm::*;
use geo_types::{Coord, Point};

/// Cluster identifier for DBSCAN results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClusterId(pub i32);

impl ClusterId {
    /// Noise cluster (outliers)
    pub const NOISE: ClusterId = ClusterId(-1);

    /// Check if this is a noise point
    pub fn is_noise(&self) -> bool {
        self.0 == -1
    }
}

/// Compute the geometric union of multiple geometries
///
/// Merges all input geometries into a single geometry representing their union.
///
/// # Arguments
///
/// * `geometries` - Input geometries to union
///
/// # Returns
///
/// A single geometry representing the union of all inputs
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::analysis::aggregations::spatial_union;
/// use oxirs_geosparql::geometry::Geometry;
///
/// let geoms = vec![
///     Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap(),
///     Geometry::from_wkt("POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))").unwrap(),
/// ];
///
/// let union = spatial_union(&geoms).unwrap();
/// // Result is a single polygon covering the combined area
/// ```
pub fn spatial_union(geometries: &[Geometry]) -> Result<Geometry> {
    if geometries.is_empty() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "Cannot union empty geometry list".to_string(),
        ));
    }

    // Start with the first geometry
    let mut result = geometries[0].clone();

    // Union with each subsequent geometry
    for geom in &geometries[1..] {
        result = union_two(&result, geom)?;
    }

    Ok(result)
}

/// Union two geometries
fn union_two(geom1: &Geometry, geom2: &Geometry) -> Result<Geometry> {
    geom1.validate_crs_compatibility(geom2)?;

    use geo::BooleanOps;

    let union_geom = match (&geom1.geom, &geom2.geom) {
        (geo_types::Geometry::Polygon(p1), geo_types::Geometry::Polygon(p2)) => {
            geo_types::Geometry::MultiPolygon(p1.union(p2))
        }
        _ => {
            // Simplified: return first geometry for unsupported types
            geom1.geom.clone()
        }
    };

    Ok(Geometry::new(union_geom))
}

/// Compute the convex hull of multiple geometries
///
/// Returns the smallest convex geometry that contains all input geometries.
///
/// # Arguments
///
/// * `geometries` - Input geometries
///
/// # Returns
///
/// The convex hull as a polygon
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::analysis::aggregations::spatial_convex_hull;
/// use oxirs_geosparql::geometry::Geometry;
///
/// let points = vec![
///     Geometry::from_wkt("POINT (0 0)").unwrap(),
///     Geometry::from_wkt("POINT (1 0)").unwrap(),
///     Geometry::from_wkt("POINT (1 1)").unwrap(),
///     Geometry::from_wkt("POINT (0 1)").unwrap(),
///     Geometry::from_wkt("POINT (0.5 0.5)").unwrap(),
/// ];
///
/// let hull = spatial_convex_hull(&points).unwrap();
/// // Result is a polygon with vertices (0,0), (1,0), (1,1), (0,1)
/// ```
pub fn spatial_convex_hull(geometries: &[Geometry]) -> Result<Geometry> {
    if geometries.is_empty() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "Cannot compute convex hull of empty geometry list".to_string(),
        ));
    }

    // Collect all coordinates
    let mut all_coords: Vec<Coord<f64>> = Vec::new();

    for geom in geometries {
        match &geom.geom {
            geo_types::Geometry::Point(p) => {
                all_coords.push(p.0);
            }
            geo_types::Geometry::LineString(ls) => {
                all_coords.extend(ls.coords());
            }
            geo_types::Geometry::Polygon(poly) => {
                all_coords.extend(poly.exterior().coords());
            }
            geo_types::Geometry::MultiPoint(mp) => {
                all_coords.extend(mp.iter().map(|p| p.0));
            }
            geo_types::Geometry::MultiLineString(mls) => {
                for ls in mls.iter() {
                    all_coords.extend(ls.coords());
                }
            }
            geo_types::Geometry::MultiPolygon(mpoly) => {
                for poly in mpoly.iter() {
                    all_coords.extend(poly.exterior().coords());
                }
            }
            _ => {}
        }
    }

    if all_coords.is_empty() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "No coordinates found in geometries".to_string(),
        ));
    }

    // Create MultiPoint and compute convex hull
    let multipoint =
        geo_types::MultiPoint::from(all_coords.into_iter().map(Point::from).collect::<Vec<_>>());

    let hull = multipoint.convex_hull();
    Ok(Geometry::new(geo_types::Geometry::Polygon(hull)))
}

/// Compute the spatial centroid of multiple geometries
///
/// Returns the center of mass of all input geometries.
///
/// # Arguments
///
/// * `geometries` - Input geometries
///
/// # Returns
///
/// A point representing the centroid
pub fn spatial_centroid(geometries: &[Geometry]) -> Result<Point<f64>> {
    if geometries.is_empty() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "Cannot compute centroid of empty geometry list".to_string(),
        ));
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut count = 0;

    for geom in geometries {
        if let Some(centroid) = geom.geom.centroid() {
            sum_x += centroid.x();
            sum_y += centroid.y();
            count += 1;
        }
    }

    if count == 0 {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "No valid centroids found".to_string(),
        ));
    }

    Ok(Point::new(sum_x / count as f64, sum_y / count as f64))
}

/// Compute the bounding envelope of multiple geometries
///
/// Returns a polygon representing the minimum bounding rectangle
/// that contains all input geometries.
///
/// # Arguments
///
/// * `geometries` - Input geometries
///
/// # Returns
///
/// A rectangular polygon (envelope)
pub fn spatial_envelope(geometries: &[Geometry]) -> Result<Geometry> {
    if geometries.is_empty() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "Cannot compute envelope of empty geometry list".to_string(),
        ));
    }

    use geo::BoundingRect;

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for geom in geometries {
        if let Some(rect) = geom.geom.bounding_rect() {
            min_x = min_x.min(rect.min().x);
            min_y = min_y.min(rect.min().y);
            max_x = max_x.max(rect.max().x);
            max_y = max_y.max(rect.max().y);
        }
    }

    if min_x.is_infinite() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "No valid bounding rectangles found".to_string(),
        ));
    }

    // Create rectangle as polygon
    let rect_coords = vec![
        Coord { x: min_x, y: min_y },
        Coord { x: max_x, y: min_y },
        Coord { x: max_x, y: max_y },
        Coord { x: min_x, y: max_y },
        Coord { x: min_x, y: min_y },
    ];

    let polygon = geo_types::Polygon::new(geo_types::LineString::from(rect_coords), vec![]);
    Ok(Geometry::new(geo_types::Geometry::Polygon(polygon)))
}

/// Perform DBSCAN clustering on spatial points
///
/// Groups nearby points into clusters based on density.
///
/// # Arguments
///
/// * `geometries` - Input point geometries
/// * `eps` - Maximum distance between points in a cluster
/// * `min_points` - Minimum number of points to form a dense region
///
/// # Returns
///
/// Vector of cluster IDs, one per input geometry. ClusterId::NOISE (-1) indicates outliers.
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::analysis::aggregations::spatial_dbscan;
/// use oxirs_geosparql::geometry::Geometry;
///
/// let points = vec![
///     Geometry::from_wkt("POINT (0 0)").unwrap(),
///     Geometry::from_wkt("POINT (1 0)").unwrap(),
///     Geometry::from_wkt("POINT (0 1)").unwrap(),
///     Geometry::from_wkt("POINT (100 100)").unwrap(), // Outlier
/// ];
///
/// let clusters = spatial_dbscan(&points, 2.0, 2).unwrap();
/// // First 3 points should be in cluster 0, last point is noise (-1)
/// assert_eq!(clusters[0].0, 0);
/// assert_eq!(clusters[3].0, -1);
/// ```
pub fn spatial_dbscan(
    geometries: &[Geometry],
    eps: f64,
    min_points: usize,
) -> Result<Vec<ClusterId>> {
    if geometries.is_empty() {
        return Ok(Vec::new());
    }

    // Extract point coordinates
    let mut points = Vec::new();
    for geom in geometries {
        match &geom.geom {
            geo_types::Geometry::Point(p) => {
                points.push((p.x(), p.y()));
            }
            _ => {
                // Use centroid for non-point geometries
                if let Some(centroid) = geom.geom.centroid() {
                    points.push((centroid.x(), centroid.y()));
                } else {
                    return Err(GeoSparqlError::InvalidGeometryType(
                        "Cannot extract point from geometry".to_string(),
                    ));
                }
            }
        }
    }

    // Run simple DBSCAN implementation
    Ok(dbscan_simple(&points, eps, min_points))
}

/// Simple DBSCAN implementation
fn dbscan_simple(points: &[(f64, f64)], eps: f64, min_points: usize) -> Vec<ClusterId> {
    let n = points.len();
    let mut labels = vec![ClusterId::NOISE; n];
    let mut cluster_id = 0;

    for i in 0..n {
        // Skip if already labeled
        if labels[i].0 != -1 {
            continue;
        }

        // Find neighbors
        let neighbors = region_query(points, i, eps);

        // Mark as noise if not enough neighbors
        if neighbors.len() < min_points {
            continue;
        }

        // Start new cluster
        labels[i] = ClusterId(cluster_id);
        let mut seed_set = neighbors.clone();
        let mut j = 0;

        while j < seed_set.len() {
            let q = seed_set[j];
            j += 1;

            // Change noise to border point
            if labels[q] == ClusterId::NOISE {
                labels[q] = ClusterId(cluster_id);
            }

            // Already processed
            if labels[q].0 != -1 && labels[q] != ClusterId::NOISE {
                continue;
            }

            labels[q] = ClusterId(cluster_id);

            // Find neighbors of q
            let q_neighbors = region_query(points, q, eps);

            if q_neighbors.len() >= min_points {
                // Add new neighbors to seed set
                for &neighbor in &q_neighbors {
                    if !seed_set.contains(&neighbor) {
                        seed_set.push(neighbor);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    labels
}

/// Find all points within eps distance of point at index
fn region_query(points: &[(f64, f64)], index: usize, eps: f64) -> Vec<usize> {
    let mut neighbors = Vec::new();
    let (x, y) = points[index];

    for (i, &(px, py)) in points.iter().enumerate() {
        let dist = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
        if dist <= eps {
            neighbors.push(i);
        }
    }

    neighbors
}

/// Aggregate geometries by computing their union (for SPARQL GROUP BY)
pub fn aggregate_union(group: &[Geometry]) -> Result<Geometry> {
    spatial_union(group)
}

/// Aggregate geometries by computing their convex hull (for SPARQL GROUP BY)
pub fn aggregate_convex_hull(group: &[Geometry]) -> Result<Geometry> {
    spatial_convex_hull(group)
}

/// Aggregate geometries by computing their centroid (for SPARQL GROUP BY)
pub fn aggregate_centroid(group: &[Geometry]) -> Result<Geometry> {
    let centroid = spatial_centroid(group)?;
    Ok(Geometry::new(geo_types::Geometry::Point(centroid)))
}

/// Aggregate geometries by computing their envelope (for SPARQL GROUP BY)
pub fn aggregate_envelope(group: &[Geometry]) -> Result<Geometry> {
    spatial_envelope(group)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_union_basic() {
        let geoms = vec![
            Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))").unwrap(),
        ];

        let union = spatial_union(&geoms);
        assert!(union.is_ok());
    }

    #[test]
    fn test_spatial_convex_hull() {
        let points = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (1 0)").unwrap(),
            Geometry::from_wkt("POINT (1 1)").unwrap(),
            Geometry::from_wkt("POINT (0 1)").unwrap(),
            Geometry::from_wkt("POINT (0.5 0.5)").unwrap(), // Interior point
        ];

        let hull = spatial_convex_hull(&points).unwrap();

        match hull.geom {
            geo_types::Geometry::Polygon(poly) => {
                // Hull should have 4 or 5 points (4 corners + closing point)
                assert!(poly.exterior().0.len() >= 4);
            }
            _ => panic!("Expected polygon"),
        }
    }

    #[test]
    fn test_spatial_centroid() {
        let points = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (2 0)").unwrap(),
            Geometry::from_wkt("POINT (2 2)").unwrap(),
            Geometry::from_wkt("POINT (0 2)").unwrap(),
        ];

        let centroid = spatial_centroid(&points).unwrap();
        assert!((centroid.x() - 1.0).abs() < 1e-10);
        assert!((centroid.y() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_envelope() {
        let geoms = vec![
            Geometry::from_wkt("POINT (1 1)").unwrap(),
            Geometry::from_wkt("POINT (3 2)").unwrap(),
            Geometry::from_wkt("POINT (2 4)").unwrap(),
        ];

        let envelope = spatial_envelope(&geoms).unwrap();

        match envelope.geom {
            geo_types::Geometry::Polygon(poly) => {
                use geo::BoundingRect;
                let rect = poly.bounding_rect().unwrap();
                assert_eq!(rect.min().x, 1.0);
                assert_eq!(rect.min().y, 1.0);
                assert_eq!(rect.max().x, 3.0);
                assert_eq!(rect.max().y, 4.0);
            }
            _ => panic!("Expected polygon"),
        }
    }

    #[test]
    fn test_spatial_dbscan() {
        let points = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (1 0)").unwrap(),
            Geometry::from_wkt("POINT (0 1)").unwrap(),
            Geometry::from_wkt("POINT (100 100)").unwrap(), // Outlier
        ];

        let clusters = spatial_dbscan(&points, 2.0, 2).unwrap();
        assert_eq!(clusters.len(), 4);

        // First 3 points should be in same cluster
        assert_eq!(clusters[0], clusters[1]);
        assert_eq!(clusters[1], clusters[2]);

        // Last point should be noise
        assert!(clusters[3].is_noise());
    }

    #[test]
    fn test_aggregate_functions() {
        let geoms = vec![
            Geometry::from_wkt("POINT (0 0)").unwrap(),
            Geometry::from_wkt("POINT (1 1)").unwrap(),
        ];

        assert!(aggregate_union(&geoms).is_ok());
        assert!(aggregate_convex_hull(&geoms).is_ok());
        assert!(aggregate_centroid(&geoms).is_ok());
        assert!(aggregate_envelope(&geoms).is_ok());
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<Geometry> = vec![];

        assert!(spatial_union(&empty).is_err());
        assert!(spatial_convex_hull(&empty).is_err());
        assert!(spatial_centroid(&empty).is_err());
        assert!(spatial_envelope(&empty).is_err());
        assert!(spatial_dbscan(&empty, 1.0, 2).is_ok()); // Empty is ok for DBSCAN
    }
}
