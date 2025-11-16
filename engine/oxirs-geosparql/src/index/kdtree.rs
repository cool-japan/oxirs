//! K-d Tree spatial index implementation for point clouds
//!
//! K-d Tree provides optimal performance for point data with excellent
//! nearest neighbor query performance:
//!
//! - **Optimal for points**: Best-in-class for large point datasets
//! - **Fast nearest neighbor**: O(log n) average case
//! - **Efficient k-NN**: Superior to other structures for k-nearest
//! - **Memory efficient**: Binary tree structure
//!
//! # Performance Characteristics
//!
//! - **Build**: O(n log n) with median-based construction
//! - **Nearest neighbor**: O(log n) average, O(n) worst
//! - **k-NN query**: O(k log n) average case
//! - **Range query**: O(n^(1-1/k) + m) where m is result size
//!
//! # When to Use
//!
//! - Pure point data (no lines/polygons)
//! - Nearest neighbor queries are primary use case
//! - Large point clouds (>10K points)
//! - Static or infrequent updates
//!
//! # When NOT to Use
//!
//! - Mixed geometry types
//! - Frequent insertions (requires rebalancing)
//! - High-dimensional data (k > 10)
//! - Very small datasets (<100 points)
//!
//! # Example
//!
//! ```ignore
//! use oxirs_geosparql::index::kdtree::KdTree;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! // Best with bulk loading
//! let points = vec![
//!     Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
//!     Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
//! ];
//!
//! let index = KdTree::bulk_load(points);
//! let (nearest, dist) = index.nearest(1.5, 2.5).unwrap();
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use crate::index::SpatialIndexTrait;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// K-d Tree node
struct KdNode {
    /// Point coordinates
    point: [f64; 2],
    /// Geometry ID
    id: u64,
    /// Split dimension (0 = x, 1 = y)
    axis: usize,
    /// Left child (points with coord[axis] <= point[axis])
    left: Option<Box<KdNode>>,
    /// Right child (points with coord[axis] > point[axis])
    right: Option<Box<KdNode>>,
}

impl KdNode {
    fn new(point: [f64; 2], id: u64, axis: usize) -> Self {
        Self {
            point,
            id,
            axis,
            left: None,
            right: None,
        }
    }

    /// Find nearest neighbor to query point
    fn nearest(&self, query: [f64; 2], best: &mut Option<(u64, f64)>) {
        // Calculate distance to this point
        let dx = self.point[0] - query[0];
        let dy = self.point[1] - query[1];
        let dist_sq = dx * dx + dy * dy;

        // Update best if closer
        if best.is_none() || dist_sq < best.unwrap().1 {
            *best = Some((self.id, dist_sq));
        }

        // Determine which side of split to search first
        let diff = query[self.axis] - self.point[self.axis];
        let (first, second) = if diff <= 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };

        // Search near side
        if let Some(child) = first {
            child.nearest(query, best);
        }

        // Check if we need to search far side
        if let Some(best_dist) = best {
            if diff * diff < best_dist.1 {
                if let Some(child) = second {
                    child.nearest(query, best);
                }
            }
        }
    }

    /// Find k nearest neighbors
    fn nearest_k(&self, query: [f64; 2], k: usize, heap: &mut Vec<(u64, f64)>) {
        // Calculate distance to this point
        let dx = self.point[0] - query[0];
        let dy = self.point[1] - query[1];
        let dist_sq = dx * dx + dy * dy;

        // Add to heap
        if heap.len() < k {
            heap.push((self.id, dist_sq));
            heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        } else if dist_sq < heap[0].1 {
            heap[0] = (self.id, dist_sq);
            heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }

        // Determine search order
        let diff = query[self.axis] - self.point[self.axis];
        let (first, second) = if diff <= 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };

        // Search near side
        if let Some(child) = first {
            child.nearest_k(query, k, heap);
        }

        // Check far side if needed
        if heap.len() < k || diff * diff < heap[0].1 {
            if let Some(child) = second {
                child.nearest_k(query, k, heap);
            }
        }
    }

    /// Query range
    fn query_range(&self, min: [f64; 2], max: [f64; 2], results: &mut Vec<(u64, [f64; 2])>) {
        // Check if this point is in range
        if self.point[0] >= min[0]
            && self.point[0] <= max[0]
            && self.point[1] >= min[1]
            && self.point[1] <= max[1]
        {
            results.push((self.id, self.point));
        }

        // Search left if range overlaps
        if min[self.axis] <= self.point[self.axis] {
            if let Some(ref child) = self.left {
                child.query_range(min, max, results);
            }
        }

        // Search right if range overlaps
        if max[self.axis] >= self.point[self.axis] {
            if let Some(ref child) = self.right {
                child.query_range(min, max, results);
            }
        }
    }
}

/// K-d Tree spatial index optimized for point clouds
pub struct KdTree {
    /// Root node
    root: RwLock<Option<Box<KdNode>>>,
    /// ID to geometry mapping
    geometries: RwLock<std::collections::HashMap<u64, Geometry>>,
    /// Next ID counter
    next_id: AtomicU64,
}

impl KdTree {
    /// Create empty K-d tree
    pub fn new() -> Self {
        Self {
            root: RwLock::new(None),
            geometries: RwLock::new(std::collections::HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Build K-d tree from points using bulk loading (recommended)
    pub fn bulk_load(geometries: Vec<Geometry>) -> Self {
        let mut points = Vec::new();
        let mut id_map = std::collections::HashMap::new();
        let mut next_id = 1u64;

        for geom in geometries {
            if let Some(p) = Self::extract_point(&geom) {
                let id = next_id;
                next_id += 1;
                points.push(([p.x(), p.y()], id));
                id_map.insert(id, geom);
            }
        }

        let root = if !points.is_empty() {
            Some(Box::new(Self::build_tree(&mut points, 0)))
        } else {
            None
        };

        Self {
            root: RwLock::new(root),
            geometries: RwLock::new(id_map),
            next_id: AtomicU64::new(next_id),
        }
    }

    /// Build K-d tree recursively
    fn build_tree(points: &mut [([f64; 2], u64)], depth: usize) -> KdNode {
        let axis = depth % 2;

        // Sort by current axis
        points.sort_by(|a, b| a.0[axis].partial_cmp(&b.0[axis]).unwrap());

        // Find median
        let median = points.len() / 2;
        let (point, id) = points[median];

        let mut node = KdNode::new(point, id, axis);

        // Recursively build left and right subtrees
        if median > 0 {
            node.left = Some(Box::new(Self::build_tree(&mut points[..median], depth + 1)));
        }
        if median + 1 < points.len() {
            node.right = Some(Box::new(Self::build_tree(
                &mut points[median + 1..],
                depth + 1,
            )));
        }

        node
    }

    /// Extract point from geometry
    fn extract_point(geom: &Geometry) -> Option<geo_types::Point<f64>> {
        use geo_types::Geometry as GeoGeometry;

        match &geom.geom {
            GeoGeometry::Point(p) => Some(*p),
            _ => None,
        }
    }
}

impl Default for KdTree {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialIndexTrait for KdTree {
    fn insert(&self, _geometry: Geometry) -> Result<u64> {
        // K-d tree doesn't support efficient single insertions
        // Use bulk_load instead
        Err(GeoSparqlError::UnsupportedOperation(
            "K-d tree requires bulk loading. Use bulk_load() instead.".to_string(),
        ))
    }

    fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>> {
        // Rebuild tree with new geometries
        let mut all_geoms: Vec<_> = self.geometries.read().values().cloned().collect();
        all_geoms.extend(geometries);

        let new_tree = Self::bulk_load(all_geoms);

        *self.root.write() = new_tree.root.into_inner();
        *self.geometries.write() = new_tree.geometries.into_inner();
        self.next_id
            .store(new_tree.next_id.load(Ordering::SeqCst), Ordering::SeqCst);

        Ok(Vec::new())
    }

    fn remove(&self, id: u64) -> Result<bool> {
        let removed = self.geometries.write().remove(&id).is_some();

        if removed {
            // Rebuild tree without removed geometry
            let all_geoms: Vec<_> = self.geometries.read().values().cloned().collect();
            let new_tree = Self::bulk_load(all_geoms);

            *self.root.write() = new_tree.root.into_inner();
        }

        Ok(removed)
    }

    fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<Geometry> {
        let root = self.root.read();
        if let Some(ref node) = *root {
            let mut results = Vec::new();
            node.query_range([min_x, min_y], [max_x, max_y], &mut results);

            let geometries = self.geometries.read();
            results
                .into_iter()
                .filter_map(|(id, _p)| geometries.get(&id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    fn query_within_distance(&self, x: f64, y: f64, distance: f64) -> Vec<(Geometry, f64)> {
        let results = self.query_bbox(x - distance, y - distance, x + distance, y + distance);

        results
            .into_iter()
            .filter_map(|geom| {
                if let Some(p) = Self::extract_point(&geom) {
                    let dx = p.x() - x;
                    let dy = p.y() - y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= distance {
                        Some((geom, dist))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    fn nearest(&self, x: f64, y: f64) -> Option<(Geometry, f64)> {
        let root = self.root.read();
        if let Some(ref node) = *root {
            let mut best = None;
            node.nearest([x, y], &mut best);

            if let Some((id, dist_sq)) = best {
                let geometries = self.geometries.read();
                geometries.get(&id).map(|g| (g.clone(), dist_sq.sqrt()))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn nearest_k(&self, x: f64, y: f64, k: usize) -> Vec<(Geometry, f64)> {
        let root = self.root.read();
        if let Some(ref node) = *root {
            let mut heap = Vec::new();
            node.nearest_k([x, y], k, &mut heap);

            let geometries = self.geometries.read();
            let mut results: Vec<_> = heap
                .into_iter()
                .filter_map(|(id, dist_sq)| {
                    geometries.get(&id).map(|g| (g.clone(), dist_sq.sqrt()))
                })
                .collect();

            // Sort by distance (ascending order)
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results
        } else {
            Vec::new()
        }
    }

    fn len(&self) -> usize {
        self.geometries.read().len()
    }

    fn clear(&self) {
        *self.root.write() = None;
        self.geometries.write().clear();
    }

    fn index_type(&self) -> &'static str {
        "K-d Tree"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_kdtree_bulk_load() {
        let points = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
            Geometry::new(GeoGeometry::Point(Point::new(5.0, 6.0))),
        ];

        let index = KdTree::bulk_load(points);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_kdtree_nearest() {
        let points = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))),
            Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))),
        ];

        let index = KdTree::bulk_load(points);
        let (geom, dist) = index.nearest(1.0, 1.0).unwrap();

        match geom.geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), 0.0, epsilon = 0.001);
                assert_relative_eq!(p.y(), 0.0, epsilon = 0.001);
            }
            _ => panic!("Expected Point"),
        }
        assert_relative_eq!(dist, 1.414, epsilon = 0.01);
    }

    #[test]
    fn test_kdtree_nearest_k() {
        let points = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))),
        ];

        let index = KdTree::bulk_load(points);
        let results = index.nearest_k(0.0, 0.0, 2);

        assert_eq!(results.len(), 2);
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_kdtree_query_bbox() {
        let points = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))),
            Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))),
        ];

        let index = KdTree::bulk_load(points);
        let results = index.query_bbox(0.0, 0.0, 6.0, 6.0);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_kdtree_query_within_distance() {
        let points = vec![
            Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
            Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))),
        ];

        let index = KdTree::bulk_load(points);
        let results = index.query_within_distance(0.0, 0.0, 6.0);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_kdtree_clear() {
        let points = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
        ];

        let index = KdTree::bulk_load(points);
        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_kdtree_index_type() {
        let index = KdTree::new();
        assert_eq!(index.index_type(), "K-d Tree");
    }

    #[test]
    fn test_kdtree_insert_not_supported() {
        let index = KdTree::new();
        let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));

        let result = index.insert(geom);
        assert!(result.is_err());
    }
}
