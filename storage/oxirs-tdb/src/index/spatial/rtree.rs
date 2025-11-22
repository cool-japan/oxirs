//! R*-tree spatial index implementation
//!
//! Provides efficient spatial indexing using R*-tree data structure
//! optimized for 2D geospatial queries.

use super::{BoundingBox, Geometry, Point, SpatialQuery, SpatialQueryResult, SpatialStats};
use crate::dictionary::NodeId;
use crate::error::Result;
use rstar::{RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Spatial index entry
#[derive(Debug, Clone, PartialEq)]
struct SpatialEntry {
    /// Node ID
    node_id: NodeId,
    /// Geometry
    geometry: Geometry,
    /// Bounding box (for R-tree)
    bbox: BoundingBox,
}

impl RTreeObject for SpatialEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        let bbox = &self.bbox;
        AABB::from_corners([bbox.min_lon, bbox.min_lat], [bbox.max_lon, bbox.max_lat])
    }
}

/// Spatial index using R*-tree
///
/// Provides efficient geospatial querying with O(log n) lookup time.
pub struct SpatialIndex {
    /// R*-tree index
    rtree: RTree<SpatialEntry>,
    /// Geometry storage (NodeId -> Geometry)
    geometries: HashMap<NodeId, Geometry>,
    /// Statistics
    stats: IndexStats,
}

#[derive(Debug, Clone, Default)]
struct IndexStats {
    total_count: usize,
    points_count: usize,
    polygons_count: usize,
    linestrings_count: usize,
}

impl SpatialIndex {
    /// Create a new spatial index
    pub fn new() -> Self {
        Self {
            rtree: RTree::new(),
            geometries: HashMap::new(),
            stats: IndexStats::default(),
        }
    }

    /// Insert a geometry into the index
    pub fn insert(&mut self, node_id: NodeId, geometry: Geometry) {
        // Update statistics
        match &geometry {
            Geometry::Point(_) => self.stats.points_count += 1,
            Geometry::LineString(_) => self.stats.linestrings_count += 1,
            Geometry::Polygon(_) => self.stats.polygons_count += 1,
        }
        self.stats.total_count += 1;

        let bbox = geometry.bounding_box();

        // Store geometry
        self.geometries.insert(node_id, geometry.clone());

        // Insert into R-tree
        let entry = SpatialEntry {
            node_id,
            geometry,
            bbox,
        };
        self.rtree.insert(entry);
    }

    /// Remove a geometry from the index
    pub fn remove(&mut self, node_id: &NodeId) -> Option<Geometry> {
        // Remove from storage
        let geometry = self.geometries.remove(node_id)?;

        // Update statistics
        match &geometry {
            Geometry::Point(_) => {
                self.stats.points_count = self.stats.points_count.saturating_sub(1)
            }
            Geometry::LineString(_) => {
                self.stats.linestrings_count = self.stats.linestrings_count.saturating_sub(1)
            }
            Geometry::Polygon(_) => {
                self.stats.polygons_count = self.stats.polygons_count.saturating_sub(1)
            }
        }
        self.stats.total_count = self.stats.total_count.saturating_sub(1);

        // Remove from R-tree
        let bbox = geometry.bounding_box();
        let entry = SpatialEntry {
            node_id: *node_id,
            geometry: geometry.clone(),
            bbox,
        };
        self.rtree.remove(&entry);

        Some(geometry)
    }

    /// Get a geometry by NodeId
    pub fn get(&self, node_id: &NodeId) -> Option<&Geometry> {
        self.geometries.get(node_id)
    }

    /// Query geometries within distance of a point
    pub fn within_distance(&self, center: Point, distance_meters: f64) -> Vec<SpatialQueryResult> {
        let mut results = Vec::new();

        // Create a bounding box for initial filtering
        // Approximate: 1 degree ≈ 111 km at equator
        let degree_delta = distance_meters / 111_000.0;
        let bbox = BoundingBox::new(
            center.lat - degree_delta,
            center.lon - degree_delta,
            center.lat + degree_delta,
            center.lon + degree_delta,
        );

        // Query R-tree for candidates
        let candidates = self.intersects_bbox(&bbox);

        // Filter by exact distance
        for candidate in candidates {
            let distance = candidate.geometry.distance_to(&center);
            if distance <= distance_meters {
                results.push(SpatialQueryResult {
                    node_id: candidate.node_id,
                    geometry: candidate.geometry,
                    distance: Some(distance),
                });
            }
        }

        // Sort by distance
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Query geometries intersecting a bounding box
    pub fn intersects_bbox(&self, bbox: &BoundingBox) -> Vec<SpatialQueryResult> {
        let envelope =
            AABB::from_corners([bbox.min_lon, bbox.min_lat], [bbox.max_lon, bbox.max_lat]);

        self.rtree
            .locate_in_envelope_intersecting(&envelope)
            .map(|entry| SpatialQueryResult {
                node_id: entry.node_id,
                geometry: entry.geometry.clone(),
                distance: None,
            })
            .collect()
    }

    /// Query geometries containing a point
    pub fn contains_point(&self, point: &Point) -> Vec<SpatialQueryResult> {
        let envelope = AABB::from_corners([point.lon, point.lat], [point.lon, point.lat]);

        self.rtree
            .locate_in_envelope_intersecting(&envelope)
            .filter(|entry| entry.geometry.contains(point))
            .map(|entry| SpatialQueryResult {
                node_id: entry.node_id,
                geometry: entry.geometry.clone(),
                distance: None,
            })
            .collect()
    }

    /// K nearest neighbors query
    ///
    /// Note: This is a simplified implementation that scans all geometries
    /// and sorts by distance. For very large datasets, consider using
    /// a more sophisticated nearest neighbor algorithm.
    pub fn k_nearest_neighbors(&self, point: &Point, k: usize) -> Vec<SpatialQueryResult> {
        let mut results: Vec<SpatialQueryResult> = self
            .geometries
            .iter()
            .map(|(node_id, geometry)| {
                let distance = geometry.distance_to(point);
                SpatialQueryResult {
                    node_id: *node_id,
                    geometry: geometry.clone(),
                    distance: Some(distance),
                }
            })
            .collect();

        // Sort by distance
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top k
        results.into_iter().take(k).collect()
    }

    /// Execute a spatial query
    pub fn query(&self, query: &SpatialQuery) -> Vec<SpatialQueryResult> {
        match query {
            SpatialQuery::WithinDistance { center, distance } => {
                self.within_distance(*center, *distance)
            }
            SpatialQuery::IntersectsBBox { bbox } => self.intersects_bbox(bbox),
            SpatialQuery::Contains { point } => self.contains_point(point),
            SpatialQuery::KNearestNeighbors { point, k } => self.k_nearest_neighbors(point, *k),
        }
    }

    /// Get index statistics
    pub fn stats(&self) -> SpatialStats {
        SpatialStats {
            geometry_count: self.stats.total_count,
            points_count: self.stats.points_count,
            polygons_count: self.stats.polygons_count,
            linestrings_count: self.stats.linestrings_count,
            // R-tree depth and node count estimation
            rtree_depth: if self.stats.total_count > 0 {
                ((self.stats.total_count as f64).log2().ceil() as usize).max(1)
            } else {
                0
            },
            rtree_node_count: self.rtree.size(),
        }
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.rtree = RTree::new();
        self.geometries.clear();
        self.stats = IndexStats::default();
    }

    /// Get the number of geometries in the index
    pub fn len(&self) -> usize {
        self.stats.total_count
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.stats.total_count == 0
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_index_creation() {
        let index = SpatialIndex::new();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let mut index = SpatialIndex::new();
        let node_id = NodeId::new(1);
        let point = Point::new(40.7128, -74.0060);

        index.insert(node_id, Geometry::Point(point));

        assert_eq!(index.len(), 1);
        assert!(index.get(&node_id).is_some());
    }

    #[test]
    fn test_remove() {
        let mut index = SpatialIndex::new();
        let node_id = NodeId::new(1);
        let point = Point::new(40.7128, -74.0060);

        index.insert(node_id, Geometry::Point(point));
        assert_eq!(index.len(), 1);

        let removed = index.remove(&node_id);
        assert!(removed.is_some());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_within_distance() {
        let mut index = SpatialIndex::new();

        // New York City
        let nyc = Point::new(40.7128, -74.0060);
        index.insert(NodeId::new(1), Geometry::Point(nyc));

        // Times Square (nearby)
        let times_square = Point::new(40.7589, -73.9851);
        index.insert(NodeId::new(2), Geometry::Point(times_square));

        // Los Angeles (far away)
        let la = Point::new(34.0522, -118.2437);
        index.insert(NodeId::new(3), Geometry::Point(la));

        // Query within 10km of NYC
        let results = index.within_distance(nyc, 10_000.0);

        // Should find NYC and Times Square, but not LA
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_intersects_bbox() {
        let mut index = SpatialIndex::new();

        let p1 = Point::new(40.7128, -74.0060);
        let p2 = Point::new(40.7589, -73.9851);
        let p3 = Point::new(34.0522, -118.2437);

        index.insert(NodeId::new(1), Geometry::Point(p1));
        index.insert(NodeId::new(2), Geometry::Point(p2));
        index.insert(NodeId::new(3), Geometry::Point(p3));

        // Bounding box around NYC area
        let bbox = BoundingBox::new(40.0, -75.0, 41.0, -73.0);
        let results = index.intersects_bbox(&bbox);

        // Should find p1 and p2, but not p3
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let mut index = SpatialIndex::new();

        let p1 = Point::new(40.7128, -74.0060);
        let p2 = Point::new(40.7589, -73.9851);
        let p3 = Point::new(34.0522, -118.2437);

        index.insert(NodeId::new(1), Geometry::Point(p1));
        index.insert(NodeId::new(2), Geometry::Point(p2));
        index.insert(NodeId::new(3), Geometry::Point(p3));

        // Find 2 nearest neighbors to NYC
        let results = index.k_nearest_neighbors(&p1, 2);

        assert_eq!(results.len(), 2);
        // First should be NYC itself (distance ~0)
        assert!(results[0].distance.unwrap() < 1.0);
    }

    #[test]
    fn test_spatial_query_within_distance() {
        let mut index = SpatialIndex::new();

        let nyc = Point::new(40.7128, -74.0060);
        index.insert(NodeId::new(1), Geometry::Point(nyc));

        let query = SpatialQuery::WithinDistance {
            center: nyc,
            distance: 1000.0,
        };

        let results = index.query(&query);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut index = SpatialIndex::new();

        let p1 = Point::new(40.7128, -74.0060);
        index.insert(NodeId::new(1), Geometry::Point(p1));

        assert_eq!(index.len(), 1);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut index = SpatialIndex::new();

        let p1 = Point::new(40.7128, -74.0060);
        index.insert(NodeId::new(1), Geometry::Point(p1));

        let stats = index.stats();
        assert_eq!(stats.geometry_count, 1);
        assert_eq!(stats.points_count, 1);
        assert_eq!(stats.polygons_count, 0);
    }

    #[test]
    fn test_polygon_contains() {
        use super::super::Polygon;

        let mut index = SpatialIndex::new();

        // Create a square polygon
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let polygon = Polygon::new(square).unwrap();
        index.insert(NodeId::new(1), Geometry::Polygon(polygon));

        // Query for point inside polygon
        let inside_point = Point::new(0.5, 0.5);
        let results = index.contains_point(&inside_point);
        assert_eq!(results.len(), 1);

        // Query for point outside polygon
        let outside_point = Point::new(2.0, 2.0);
        let results = index.contains_point(&outside_point);
        assert_eq!(results.len(), 0);
    }
}
