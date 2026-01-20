//! Geospatial indexing for RDF triple stores
//!
//! Provides GeoSPARQL-compatible spatial indexing using R*-tree data structure.
//! Supports efficient querying of spatial relationships:
//! - Distance queries (within radius)
//! - Containment queries (point in polygon)
//! - Intersection queries (bounding box overlaps)
//! - Nearest neighbor queries
//!
//! ## Features
//!
//! - **R*-tree index** - Optimized spatial index with low false positive rate
//! - **GeoSPARQL support** - Standard geospatial query language
//! - **WKT parsing** - Well-Known Text geometry format
//! - **GeoJSON support** - JSON-based geometry encoding
//! - **Efficient range queries** - O(log n) spatial lookups
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_tdb::index::spatial::{SpatialIndex, Point, BoundingBox};
//! use oxirs_tdb::dictionary::NodeId;
//!
//! let mut index = SpatialIndex::new();
//!
//! // Index a point
//! let point = Point::new(40.7128, -74.0060); // New York City
//! let node_id = NodeId::new(1);
//! index.insert(node_id, point.into());
//!
//! // Query nearby points (within 10km)
//! let center = Point::new(40.7589, -73.9851); // Times Square
//! let radius_meters = 10_000.0;
//! let nearby = index.within_distance(center, radius_meters);
//! ```

pub mod functions;
pub mod geometry;
pub mod rtree;

pub use functions::*;
pub use geometry::{BoundingBox, Geometry, LineString, Point, Polygon};
pub use rtree::SpatialIndex;

use crate::dictionary::NodeId;
use crate::error::{Result, TdbError};
use serde::{Deserialize, Serialize};

/// Spatial query types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialQuery {
    /// Find all geometries within distance of a point
    WithinDistance {
        /// Center point
        center: Point,
        /// Distance in meters
        distance: f64,
    },
    /// Find all geometries intersecting a bounding box
    IntersectsBBox {
        /// Bounding box
        bbox: BoundingBox,
    },
    /// Find geometries containing a point
    Contains {
        /// Point to test
        point: Point,
    },
    /// K nearest neighbors
    KNearestNeighbors {
        /// Query point
        point: Point,
        /// Number of neighbors
        k: usize,
    },
}

/// Spatial query result
#[derive(Debug, Clone)]
pub struct SpatialQueryResult {
    /// Node ID of the matching geometry
    pub node_id: NodeId,
    /// Geometry
    pub geometry: Geometry,
    /// Distance from query point (if applicable)
    pub distance: Option<f64>,
}

/// Spatial index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialStats {
    /// Total geometries indexed
    pub geometry_count: usize,
    /// Points count
    pub points_count: usize,
    /// Polygons count
    pub polygons_count: usize,
    /// LineStrings count
    pub linestrings_count: usize,
    /// R-tree depth
    pub rtree_depth: usize,
    /// R-tree node count
    pub rtree_node_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_module_exports() {
        // Test that all exports are accessible
        let _point = Point::new(0.0, 0.0);
        let _bbox = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
        let _index = SpatialIndex::new();
    }

    #[test]
    fn test_spatial_query_types() {
        let query = SpatialQuery::WithinDistance {
            center: Point::new(40.7128, -74.0060),
            distance: 1000.0,
        };
        assert!(matches!(query, SpatialQuery::WithinDistance { .. }));

        let query2 = SpatialQuery::IntersectsBBox {
            bbox: BoundingBox::new(0.0, 0.0, 1.0, 1.0),
        };
        assert!(matches!(query2, SpatialQuery::IntersectsBBox { .. }));
    }

    #[test]
    fn test_spatial_stats_creation() {
        let stats = SpatialStats {
            geometry_count: 100,
            points_count: 60,
            polygons_count: 30,
            linestrings_count: 10,
            rtree_depth: 4,
            rtree_node_count: 25,
        };

        assert_eq!(stats.geometry_count, 100);
        assert_eq!(stats.rtree_depth, 4);
    }
}
