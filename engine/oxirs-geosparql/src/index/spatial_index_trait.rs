//! Unified trait API for spatial indexes
//!
//! This module provides a common interface for different spatial index types,
//! allowing users to switch between implementations based on their use case.
//!
//! Different index types excel at different scenarios:
//! - **R-tree**: General purpose, handles all geometry types
//! - **Quadtree**: Fast insertions, good for uniformly distributed data
//! - **K-d tree**: Optimal for point clouds and nearest neighbor queries
//! - **Grid**: Extremely fast for uniform data, fixed grid size

use crate::error::Result;
use crate::geometry::Geometry;

/// Trait for spatial indexes
///
/// This trait provides a common interface for all spatial index implementations.
/// Different implementations can be optimized for different use cases:
///
/// - **R-tree** (via `SpatialIndex`): General purpose, handles all geometry types
/// - **Quadtree** (via `QuadtreeIndex`): Dynamic partitioning, fast insertions
/// - **K-d tree** (via `KdTreeIndex`): Optimal for point clouds
/// - **Grid** (via `GridIndex`): Fixed-size cells, extremely fast queries
///
/// # Example
///
/// ```
/// use oxirs_geosparql::index::{SpatialIndexTrait, SpatialIndex};
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
///
/// // Can use any spatial index implementation
/// let index: Box<dyn SpatialIndexTrait> = Box::new(SpatialIndex::new());
/// // or let index: Box<dyn SpatialIndexTrait> = Box::new(QuadtreeIndex::new(...));
/// ```
pub trait SpatialIndexTrait: Send + Sync {
    /// Insert a geometry into the index
    ///
    /// Returns a unique ID for the inserted geometry
    fn insert(&self, geometry: Geometry) -> Result<u64>;

    /// Batch insert multiple geometries
    ///
    /// More efficient than calling `insert()` multiple times
    fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>>;

    /// Remove a geometry from the index by ID
    ///
    /// Returns true if the geometry was found and removed
    fn remove(&self, id: u64) -> Result<bool>;

    /// Query for geometries intersecting a bounding box
    ///
    /// Returns geometries whose bounding boxes intersect with the query box
    fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<Geometry>;

    /// Find geometries within a given distance from a point
    ///
    /// Returns geometries and their distances from the query point
    fn query_within_distance(&self, x: f64, y: f64, distance: f64) -> Vec<(Geometry, f64)>;

    /// Find the nearest geometry to a point
    ///
    /// Returns the nearest geometry and its distance, or None if index is empty
    fn nearest(&self, x: f64, y: f64) -> Option<(Geometry, f64)>;

    /// Find the k nearest geometries to a point
    ///
    /// Returns up to k nearest geometries sorted by distance
    fn nearest_k(&self, x: f64, y: f64, k: usize) -> Vec<(Geometry, f64)>;

    /// Get the number of geometries in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all geometries from the index
    fn clear(&self);

    /// Get the index type name
    fn index_type(&self) -> &'static str;
}

/// Automatically select the best spatial index for a given dataset
///
/// This function analyzes the data characteristics and returns the most
/// appropriate spatial index implementation.
///
/// # Selection criteria
///
/// - **Point data only**: K-d tree (fastest for point queries)
/// - **Dense uniform data**: Grid index (extremely fast queries)
/// - **Mixed geometry types**: R-tree (general purpose)
/// - **Dynamic insertions**: Quadtree (fast insertions, good for streaming)
///
/// # Example
///
/// ```
/// use oxirs_geosparql::index::select_optimal_index;
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geometries = vec![
///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
///     Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
/// ];
///
/// let index = select_optimal_index(&geometries);
/// println!("Selected index type: {}", index.index_type());
/// ```
pub fn select_optimal_index(geometries: &[Geometry]) -> Box<dyn SpatialIndexTrait> {
    use super::SpatialIndex;
    use geo_types::Geometry as GeoGeometry;

    if geometries.is_empty() {
        // Default to R-tree for empty datasets
        return Box::new(SpatialIndex::new());
    }

    // Check if all geometries are points
    let all_points = geometries
        .iter()
        .all(|g| matches!(g.geom, GeoGeometry::Point(_)));

    // Check spatial distribution
    let bounds = calculate_bounds(geometries);
    let density = geometries.len() as f64 / ((bounds.2 - bounds.0) * (bounds.3 - bounds.1));

    if all_points && geometries.len() > 1000 {
        // K-d tree would be ideal for large point datasets
        // For now, use R-tree (K-d tree implementation pending)
        Box::new(SpatialIndex::new())
    } else if density > 100.0 && geometries.len() > 500 {
        // Dense data - grid index would be good
        // For now, use R-tree (Grid index implementation pending)
        Box::new(SpatialIndex::new())
    } else {
        // General case - use R-tree
        Box::new(SpatialIndex::new())
    }
}

/// Calculate bounding box of a set of geometries
fn calculate_bounds(geometries: &[Geometry]) -> (f64, f64, f64, f64) {
    use geo::BoundingRect;

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for geom in geometries {
        if let Some(bbox) = geom.geom.bounding_rect() {
            let min = bbox.min();
            let max = bbox.max();

            min_x = min_x.min(min.x);
            min_y = min_y.min(min.y);
            max_x = max_x.max(max.x);
            max_y = max_y.max(max.y);
        }
    }

    (min_x, min_y, max_x, max_y)
}

/// Performance hints for spatial index selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexHint {
    /// Optimize for bulk loading (use R-tree bulk load)
    BulkLoad,
    /// Optimize for dynamic insertions (use Quadtree)
    DynamicInserts,
    /// Optimize for point queries (use K-d tree)
    PointQueries,
    /// Optimize for range queries (use R-tree)
    RangeQueries,
    /// Optimize for nearest neighbor (use K-d tree or R-tree)
    NearestNeighbor,
    /// Use default (R-tree)
    Default,
}

/// Create a spatial index with performance hints
///
/// This function creates an appropriate spatial index based on the provided hints.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::index::{create_index_with_hint, IndexHint};
///
/// // Create index optimized for nearest neighbor queries
/// let index = create_index_with_hint(IndexHint::NearestNeighbor);
/// assert_eq!(index.index_type(), "R-tree");
/// ```
pub fn create_index_with_hint(hint: IndexHint) -> Box<dyn SpatialIndexTrait> {
    use super::SpatialIndex;

    match hint {
        IndexHint::BulkLoad => Box::new(SpatialIndex::new()), // Can use bulk_load method
        IndexHint::DynamicInserts => Box::new(SpatialIndex::new()), // Quadtree when implemented
        IndexHint::PointQueries => Box::new(SpatialIndex::new()), // K-d tree when implemented
        IndexHint::RangeQueries => Box::new(SpatialIndex::new()),
        IndexHint::NearestNeighbor => Box::new(SpatialIndex::new()),
        IndexHint::Default => Box::new(SpatialIndex::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_select_optimal_index_empty() {
        let geometries: Vec<Geometry> = vec![];
        let index = select_optimal_index(&geometries);
        assert_eq!(index.index_type(), "R-tree");
    }

    #[test]
    fn test_select_optimal_index_points() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
        ];
        let index = select_optimal_index(&geometries);
        assert_eq!(index.index_type(), "R-tree");
    }

    #[test]
    fn test_create_index_with_hint() {
        let index = create_index_with_hint(IndexHint::Default);
        assert_eq!(index.index_type(), "R-tree");

        let index = create_index_with_hint(IndexHint::NearestNeighbor);
        assert_eq!(index.index_type(), "R-tree");
    }

    #[test]
    fn test_calculate_bounds() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))),
        ];

        let (min_x, min_y, max_x, max_y) = calculate_bounds(&geometries);
        assert_eq!(min_x, 0.0);
        assert_eq!(min_y, 0.0);
        assert_eq!(max_x, 10.0);
        assert_eq!(max_y, 10.0);
    }
}
