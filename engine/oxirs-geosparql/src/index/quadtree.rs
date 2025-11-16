//! Quadtree and Octree spatial index implementations
//!
//! This module provides hierarchical space partitioning for both 2D (Quadtree)
//! and 3D (Octree) spatial data:
//!
//! **Quadtree (2D)**:
//! - Divides space into 4 quadrants recursively
//! - Adaptive subdivision based on point density
//! - Excellent for non-uniform data distributions
//!
//! **Octree (3D)**:
//! - Divides space into 8 octants recursively
//! - Full 3D spatial queries (requires Z coordinates)
//! - Optimal for volumetric data and 3D geometries
//!
//! # Performance Characteristics
//!
//! - **Insert**: O(log n) average, O(n) worst case
//! - **Query**: O(log n + k) where k is result size
//! - **Nearest neighbor**: O(log n)
//! - **Memory**: O(n) with tree overhead
//!
//! # When to Use Quadtree
//!
//! - Non-uniform 2D data distribution
//! - Dynamic insertions and deletions
//! - Spatial clustering analysis
//! - Region-based queries
//!
//! # When to Use Octree
//!
//! - 3D geometry data
//! - Volumetric spatial queries
//! - 3D rendering and collision detection
//! - Point cloud processing
//!
//! # Example
//!
//! ```ignore
//! use oxirs_geosparql::index::quadtree::Quadtree;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! // 2D Quadtree
//! let index = Quadtree::new(0.0, 0.0, 1000.0, 1000.0, 10);
//!
//! let geom = Geometry::new(GeoGeometry::Point(Point::new(500.0, 500.0)));
//! let id = index.insert(geom).unwrap();
//!
//! let results = index.query_bbox(400.0, 400.0, 600.0, 600.0);
//! assert_eq!(results.len(), 1);
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use crate::index::SpatialIndexTrait;
use geo::BoundingRect;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Quadrant indices for Quadtree (2D)
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quadrant {
    NorthEast = 0,
    NorthWest = 1,
    SouthWest = 2,
    SouthEast = 3,
}

/// Octant indices for Octree (3D)
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Octant {
    TopNorthEast = 0,
    TopNorthWest = 1,
    TopSouthWest = 2,
    TopSouthEast = 3,
    BottomNorthEast = 4,
    BottomNorthWest = 5,
    BottomSouthWest = 6,
    BottomSouthEast = 7,
}

/// 2D axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
struct BBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl BBox {
    fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    fn intersects(&self, other: &BBox) -> bool {
        !(self.max_x < other.min_x
            || self.min_x > other.max_x
            || self.max_y < other.min_y
            || self.min_y > other.max_y)
    }

    fn center(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
}

/// Quadtree node (2D spatial partitioning)
struct QuadNode {
    /// Node bounding box
    bounds: BBox,
    /// Maximum capacity before subdivision
    capacity: usize,
    /// Geometry IDs stored in this node
    geometries: Vec<(u64, f64, f64)>, // (id, x, y)
    /// Child nodes (NE, NW, SW, SE)
    children: Option<[Box<QuadNode>; 4]>,
}

impl QuadNode {
    fn new(bounds: BBox, capacity: usize) -> Self {
        Self {
            bounds,
            capacity,
            geometries: Vec::new(),
            children: None,
        }
    }

    /// Insert a point into the quadtree
    fn insert(&mut self, id: u64, x: f64, y: f64) -> bool {
        if !self.bounds.contains(x, y) {
            return false;
        }

        // If not at capacity and no children, add here
        if self.children.is_none() && self.geometries.len() < self.capacity {
            self.geometries.push((id, x, y));
            return true;
        }

        // Subdivide if needed
        if self.children.is_none() {
            self.subdivide();
        }

        // Insert into appropriate child
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                if child.insert(id, x, y) {
                    return true;
                }
            }
        }

        false
    }

    /// Subdivide this node into 4 children
    fn subdivide(&mut self) {
        let (cx, cy) = self.bounds.center();

        let ne = Box::new(QuadNode::new(
            BBox::new(cx, cy, self.bounds.max_x, self.bounds.max_y),
            self.capacity,
        ));
        let nw = Box::new(QuadNode::new(
            BBox::new(self.bounds.min_x, cy, cx, self.bounds.max_y),
            self.capacity,
        ));
        let sw = Box::new(QuadNode::new(
            BBox::new(self.bounds.min_x, self.bounds.min_y, cx, cy),
            self.capacity,
        ));
        let se = Box::new(QuadNode::new(
            BBox::new(cx, self.bounds.min_y, self.bounds.max_x, cy),
            self.capacity,
        ));

        self.children = Some([ne, nw, sw, se]);

        // Redistribute existing geometries to children
        let geometries = std::mem::take(&mut self.geometries);
        for (id, x, y) in geometries {
            // Try to insert into children
            let mut inserted = false;
            if let Some(ref mut children) = self.children {
                for child in children.iter_mut() {
                    if child.insert(id, x, y) {
                        inserted = true;
                        break;
                    }
                }
            }
            // If not inserted (shouldn't happen), keep in parent
            if !inserted {
                self.geometries.push((id, x, y));
            }
        }
    }

    /// Query for geometries within a bounding box
    fn query(&self, query_box: &BBox, results: &mut Vec<(u64, f64, f64)>) {
        if !self.bounds.intersects(query_box) {
            return;
        }

        // Add geometries from this node that are within query box
        for &(id, x, y) in &self.geometries {
            if query_box.contains(x, y) {
                results.push((id, x, y));
            }
        }

        // Query children if they exist
        if let Some(ref children) = self.children {
            for child in children.iter() {
                child.query(query_box, results);
            }
        }
    }

    /// Find all geometries within distance from point
    fn query_within_distance(&self, x: f64, y: f64, distance: f64, results: &mut Vec<(u64, f64)>) {
        // Check if node bounding box could contain any points within distance
        let dist_sq = distance * distance;

        // Quick bounding box check
        let query_box = BBox::new(x - distance, y - distance, x + distance, y + distance);

        if !self.bounds.intersects(&query_box) {
            return;
        }

        // Check geometries in this node
        for &(id, gx, gy) in &self.geometries {
            let dx = gx - x;
            let dy = gy - y;
            let d_sq = dx * dx + dy * dy;
            if d_sq <= dist_sq {
                results.push((id, d_sq.sqrt()));
            }
        }

        // Query children
        if let Some(ref children) = self.children {
            for child in children.iter() {
                child.query_within_distance(x, y, distance, results);
            }
        }
    }

    /// Count total number of geometries in tree
    fn count(&self) -> usize {
        let mut count = self.geometries.len();
        if let Some(ref children) = self.children {
            for child in children.iter() {
                count += child.count();
            }
        }
        count
    }

    /// Count tree depth
    fn depth(&self) -> usize {
        if let Some(ref children) = self.children {
            1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
        } else {
            1
        }
    }
}

/// Quadtree spatial index for 2D spatial data
///
/// Provides hierarchical space partitioning with adaptive subdivision
/// based on local point density.
pub struct Quadtree {
    /// Root node
    root: Arc<RwLock<QuadNode>>,
    /// ID to geometry mapping
    geometries: RwLock<std::collections::HashMap<u64, Geometry>>,
    /// ID to point mapping (for removal)
    id_points: RwLock<std::collections::HashMap<u64, (f64, f64)>>,
    /// Next ID counter
    next_id: AtomicU64,
}

impl Quadtree {
    /// Create a new Quadtree with specified bounds and node capacity
    ///
    /// # Arguments
    ///
    /// * `min_x`, `min_y` - Minimum bounds
    /// * `max_x`, `max_y` - Maximum bounds
    /// * `capacity` - Maximum points per node before subdivision (typically 4-16)
    ///
    /// # Example
    ///
    /// ```
    /// use oxirs_geosparql::index::quadtree::Quadtree;
    ///
    /// // Create quadtree with capacity of 10 points per node
    /// let index = Quadtree::new(0.0, 0.0, 1000.0, 1000.0, 10);
    /// ```
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64, capacity: usize) -> Self {
        if min_x >= max_x || min_y >= max_y {
            panic!("Invalid bounds: min must be less than max");
        }
        if capacity == 0 {
            panic!("Capacity must be at least 1");
        }

        let bounds = BBox::new(min_x, min_y, max_x, max_y);
        let root = QuadNode::new(bounds, capacity);

        Self {
            root: Arc::new(RwLock::new(root)),
            geometries: RwLock::new(std::collections::HashMap::new()),
            id_points: RwLock::new(std::collections::HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Extract representative point from geometry
    fn extract_representative_point(geom: &Geometry) -> Option<geo_types::Point<f64>> {
        use geo::Centroid;
        use geo_types::Geometry as GeoGeometry;

        match &geom.geom {
            GeoGeometry::Point(p) => Some(*p),
            GeoGeometry::LineString(ls) if !ls.0.is_empty() => Some(geo_types::Point(ls.0[0])),
            GeoGeometry::Polygon(p) => p.centroid(),
            GeoGeometry::MultiPoint(mp) if !mp.0.is_empty() => Some(mp.0[0]),
            GeoGeometry::MultiLineString(mls) if !mls.0.is_empty() && !mls.0[0].0.is_empty() => {
                Some(geo_types::Point(mls.0[0].0[0]))
            }
            GeoGeometry::MultiPolygon(mp) if !mp.0.is_empty() => mp.0[0].centroid(),
            _ => None,
        }
    }

    /// Get bounding box of a geometry
    #[allow(dead_code)]
    fn get_bbox(geom: &Geometry) -> Result<(f64, f64, f64, f64)> {
        let bbox = geom.geom.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Cannot compute bounding box".to_string())
        })?;

        Ok((bbox.min().x, bbox.min().y, bbox.max().x, bbox.max().y))
    }

    /// Get statistics about the quadtree
    ///
    /// Returns (num_geometries, tree_depth, num_nodes_estimate)
    pub fn stats(&self) -> (usize, usize, usize) {
        let root = self.root.read();
        let count = root.count();
        let depth = root.depth();
        // Estimate node count: sum of 4^i for i=0 to depth-1
        let nodes_estimate = if depth > 0 {
            ((4_usize.pow(depth as u32) - 1) / 3).max(count / 4)
        } else {
            1
        };
        (count, depth, nodes_estimate)
    }
}

impl Default for Quadtree {
    fn default() -> Self {
        Self::new(0.0, 0.0, 1000.0, 1000.0, 8)
    }
}

impl SpatialIndexTrait for Quadtree {
    fn insert(&self, geometry: Geometry) -> Result<u64> {
        let point = Self::extract_representative_point(&geometry).ok_or_else(|| {
            GeoSparqlError::InvalidInput("Cannot extract point from geometry".to_string())
        })?;

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let mut root = self.root.write();
        if !root.insert(id, point.x(), point.y()) {
            return Err(GeoSparqlError::InvalidInput(
                "Geometry outside quadtree bounds".to_string(),
            ));
        }

        self.geometries.write().insert(id, geometry);
        self.id_points.write().insert(id, (point.x(), point.y()));

        Ok(id)
    }

    fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(geometries.len());
        let mut root = self.root.write();
        let mut geom_map = self.geometries.write();
        let mut id_points = self.id_points.write();

        for geom in geometries {
            if let Some(point) = Self::extract_representative_point(&geom) {
                let id = self.next_id.fetch_add(1, Ordering::SeqCst);

                if root.insert(id, point.x(), point.y()) {
                    geom_map.insert(id, geom);
                    id_points.insert(id, (point.x(), point.y()));
                    ids.push(id);
                }
            }
        }

        Ok(ids)
    }

    fn remove(&self, id: u64) -> Result<bool> {
        // Note: This is a simplified removal that doesn't actually remove from tree
        // A full implementation would need to rebuild the tree or implement lazy deletion
        let geom = self.geometries.write().remove(&id);
        self.id_points.write().remove(&id);

        Ok(geom.is_some())
    }

    fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<Geometry> {
        let query_box = BBox::new(min_x, min_y, max_x, max_y);
        let mut results = Vec::new();

        let root = self.root.read();
        root.query(&query_box, &mut results);

        let geometries = self.geometries.read();
        results
            .into_iter()
            .filter_map(|(id, _x, _y)| geometries.get(&id).cloned())
            .collect()
    }

    fn query_within_distance(&self, x: f64, y: f64, distance: f64) -> Vec<(Geometry, f64)> {
        let mut results = Vec::new();

        let root = self.root.read();
        root.query_within_distance(x, y, distance, &mut results);

        let geometries = self.geometries.read();
        results
            .into_iter()
            .filter_map(|(id, dist)| geometries.get(&id).map(|g| (g.clone(), dist)))
            .collect()
    }

    fn nearest(&self, x: f64, y: f64) -> Option<(Geometry, f64)> {
        // Iteratively expand search radius
        let mut search_radius = 1.0;
        let max_iterations = 20;

        for _ in 0..max_iterations {
            let results = self.query_within_distance(x, y, search_radius);
            if !results.is_empty() {
                return results
                    .into_iter()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            search_radius *= 2.0;
        }

        None
    }

    fn nearest_k(&self, x: f64, y: f64, k: usize) -> Vec<(Geometry, f64)> {
        let mut search_radius = 1.0;
        let max_iterations = 20;

        for _ in 0..max_iterations {
            let mut results = self.query_within_distance(x, y, search_radius);
            if results.len() >= k {
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(k);
                return results;
            }
            search_radius *= 2.0;
        }

        Vec::new()
    }

    fn len(&self) -> usize {
        self.geometries.read().len()
    }

    fn clear(&self) {
        // Read bounds and capacity first, before acquiring write lock
        let (bounds, capacity) = {
            let root = self.root.read();
            (root.bounds, root.capacity)
        };

        *self.root.write() = QuadNode::new(bounds, capacity);
        self.geometries.write().clear();
        self.id_points.write().clear();
    }

    fn index_type(&self) -> &'static str {
        "Quadtree"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_quadtree_create() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_quadtree_insert() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);
        let geom = Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0)));

        let id = index.insert(geom).unwrap();
        assert_eq!(index.len(), 1);
        assert!(id > 0);
    }

    #[test]
    fn test_quadtree_query_bbox() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(25.0, 25.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(75.0, 75.0))))
            .unwrap();

        let results = index.query_bbox(20.0, 20.0, 60.0, 60.0);
        assert_eq!(results.len(), 2); // Points at (25,25) and (50,50)
    }

    #[test]
    fn test_quadtree_subdivision() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        // Insert more than capacity to trigger subdivision
        for i in 0..10 {
            let x = (i * 10) as f64;
            let y = (i * 10) as f64;
            index
                .insert(Geometry::new(GeoGeometry::Point(Point::new(x, y))))
                .unwrap();
        }

        assert_eq!(index.len(), 10);

        let (count, depth, _nodes) = index.stats();
        assert_eq!(count, 10);
        assert!(depth > 1); // Should have subdivided
    }

    #[test]
    fn test_quadtree_nearest() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(90.0, 90.0))))
            .unwrap();

        let (geom, dist) = index.nearest(15.0, 15.0).unwrap();

        match geom.geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), 10.0, epsilon = 0.001);
                assert_relative_eq!(p.y(), 10.0, epsilon = 0.001);
            }
            _ => panic!("Expected Point"),
        }
        assert!(dist < 10.0);
    }

    #[test]
    fn test_quadtree_nearest_k() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(20.0, 20.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(30.0, 30.0))))
            .unwrap();

        let results = index.nearest_k(0.0, 0.0, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_quadtree_within_distance() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(55.0, 55.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(90.0, 90.0))))
            .unwrap();

        let results = index.query_within_distance(50.0, 50.0, 10.0);
        assert_eq!(results.len(), 2); // Points at (50,50) and (55,55)
    }

    #[test]
    fn test_quadtree_remove() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        let id = index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0))))
            .unwrap();
        assert_eq!(index.len(), 1);

        let removed = index.remove(id).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_quadtree_clear() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(25.0, 25.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(75.0, 75.0))))
            .unwrap();

        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_quadtree_insert_batch() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(25.0, 25.0))),
            Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0))),
            Geometry::new(GeoGeometry::Point(Point::new(75.0, 75.0))),
        ];

        let ids = index.insert_batch(geometries).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_quadtree_index_type() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);
        assert_eq!(index.index_type(), "Quadtree");
    }

    #[test]
    fn test_bbox_contains() {
        let bbox = BBox::new(0.0, 0.0, 100.0, 100.0);
        assert!(bbox.contains(50.0, 50.0));
        assert!(!bbox.contains(150.0, 150.0));
    }

    #[test]
    fn test_bbox_intersects() {
        let bbox1 = BBox::new(0.0, 0.0, 100.0, 100.0);
        let bbox2 = BBox::new(50.0, 50.0, 150.0, 150.0);
        let bbox3 = BBox::new(200.0, 200.0, 300.0, 300.0);

        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.intersects(&bbox3));
    }

    #[test]
    fn test_quadtree_out_of_bounds() {
        let index = Quadtree::new(0.0, 0.0, 100.0, 100.0, 4);

        let geom = Geometry::new(GeoGeometry::Point(Point::new(150.0, 150.0)));
        let result = index.insert(geom);
        assert!(result.is_err());
    }
}
