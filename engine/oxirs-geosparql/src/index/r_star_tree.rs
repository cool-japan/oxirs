//! R*-tree spatial index implementation
//!
//! R*-tree is an improved variant of R-tree with better node splitting heuristics.
//! It provides 20-40% better query performance compared to standard R-tree through:
//!
//! - **Overlap minimization**: Splits minimize bounding box overlap
//! - **Coverage minimization**: Reduces total bounding box area
//! - **Forced reinsertion**: Improves tree structure on overflow
//! - **Margin minimization**: Better distribution of entries
//!
//! # Performance Characteristics
//!
//! - **Insert**: O(log n) with forced reinsertion overhead
//! - **Query**: O(log n + k) where k is result size (20-40% faster than R-tree)
//! - **Nearest neighbor**: O(log n) with better pruning
//! - **Memory**: Similar to R-tree
//!
//! # When to Use
//!
//! - General-purpose spatial queries
//! - When query performance is critical
//! - Mixed geometry types
//! - Moderate insertion rate (bulk load preferred)
//!
//! # Example
//!
//! ```ignore
//! use oxirs_geosparql::index::r_star_tree::RStarTree;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! let mut index = RStarTree::new();
//!
//! let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
//! let id = index.insert(geom).unwrap();
//!
//! // Query for geometries
//! let results = index.query_bbox(0.0, 0.0, 5.0, 5.0);
//! assert_eq!(results.len(), 1);
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use crate::index::SpatialIndexTrait;
use parking_lot::RwLock;
use rstar::{primitives::GeomWithData, RTree, AABB};
use std::sync::atomic::{AtomicU64, Ordering};

/// R*-tree spatial index with improved splitting heuristics
///
/// This implementation uses the `rstar` crate which implements the R*-tree algorithm.
/// R*-tree improves upon standard R-tree with better node splitting and insertion strategies.
pub struct RStarTree {
    /// Internal R*-tree structure
    tree: RwLock<RTree<GeomWithData<[f64; 2], u64>>>,
    /// ID to geometry mapping for retrieval
    geometries: RwLock<std::collections::HashMap<u64, Geometry>>,
    /// Next ID counter
    next_id: AtomicU64,
}

impl RStarTree {
    /// Create a new empty R*-tree index
    pub fn new() -> Self {
        Self {
            tree: RwLock::new(RTree::new()),
            geometries: RwLock::new(std::collections::HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Create an R*-tree with bulk loading for better initial structure
    ///
    /// Bulk loading creates a more balanced tree with better query performance.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use oxirs_geosparql::index::r_star_tree::RStarTree;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geometries = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
    /// ];
    ///
    /// let index = RStarTree::bulk_load(geometries);
    /// assert_eq!(index.len(), 2);
    /// ```
    pub fn bulk_load(geometries: Vec<Geometry>) -> Self {
        let mut id_map = std::collections::HashMap::new();
        let mut points = Vec::new();
        let mut next_id = 1u64;

        for geom in geometries {
            if let Some(point) = Self::extract_representative_point(&geom) {
                let id = next_id;
                next_id += 1;

                points.push(GeomWithData::new([point.x(), point.y()], id));
                id_map.insert(id, geom);
            }
        }

        Self {
            tree: RwLock::new(RTree::bulk_load(points)),
            geometries: RwLock::new(id_map),
            next_id: AtomicU64::new(next_id),
        }
    }

    /// Extract a representative point from a geometry for spatial indexing
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

    /// Get statistics about the R*-tree
    ///
    /// Returns (num_entries, tree_depth_estimate)
    pub fn stats(&self) -> (usize, usize) {
        let len = self.geometries.read().len();
        // Estimate depth based on size (R-tree typically has branching factor of ~10-50)
        let depth_estimate = if len > 0 {
            ((len as f64).ln() / 4.0_f64.ln()).ceil() as usize
        } else {
            0
        };
        (len, depth_estimate)
    }
}

impl Default for RStarTree {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialIndexTrait for RStarTree {
    fn insert(&self, geometry: Geometry) -> Result<u64> {
        let point = Self::extract_representative_point(&geometry).ok_or_else(|| {
            GeoSparqlError::InvalidInput("Cannot index empty geometry".to_string())
        })?;

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        self.tree
            .write()
            .insert(GeomWithData::new([point.x(), point.y()], id));
        self.geometries.write().insert(id, geometry);

        Ok(id)
    }

    fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(geometries.len());
        let mut points = Vec::with_capacity(geometries.len());
        let mut geom_map = self.geometries.write();

        for geom in geometries {
            if let Some(point) = Self::extract_representative_point(&geom) {
                let id = self.next_id.fetch_add(1, Ordering::SeqCst);
                points.push(GeomWithData::new([point.x(), point.y()], id));
                geom_map.insert(id, geom);
                ids.push(id);
            }
        }

        // Insert all points into tree
        let mut tree = self.tree.write();
        for point in points {
            tree.insert(point);
        }

        Ok(ids)
    }

    fn remove(&self, id: u64) -> Result<bool> {
        let geom = self.geometries.write().remove(&id);

        if let Some(geom) = geom {
            if let Some(point) = Self::extract_representative_point(&geom) {
                let removed = self
                    .tree
                    .write()
                    .remove(&GeomWithData::new([point.x(), point.y()], id));
                Ok(removed.is_some())
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<Geometry> {
        let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);
        let tree = self.tree.read();
        let geometries = self.geometries.read();

        tree.locate_in_envelope_intersecting(&envelope)
            .filter_map(|entry| geometries.get(&entry.data).cloned())
            .collect()
    }

    fn query_within_distance(&self, x: f64, y: f64, distance: f64) -> Vec<(Geometry, f64)> {
        let tree = self.tree.read();
        let geometries = self.geometries.read();

        tree.locate_within_distance([x, y], distance * distance)
            .filter_map(|entry| {
                geometries.get(&entry.data).map(|geom| {
                    let dist =
                        ((entry.geom()[0] - x).powi(2) + (entry.geom()[1] - y).powi(2)).sqrt();
                    (geom.clone(), dist)
                })
            })
            .collect()
    }

    fn nearest(&self, x: f64, y: f64) -> Option<(Geometry, f64)> {
        let tree = self.tree.read();
        let geometries = self.geometries.read();

        tree.nearest_neighbor(&[x, y]).and_then(|entry| {
            geometries.get(&entry.data).map(|geom| {
                let dist = ((entry.geom()[0] - x).powi(2) + (entry.geom()[1] - y).powi(2)).sqrt();
                (geom.clone(), dist)
            })
        })
    }

    fn nearest_k(&self, x: f64, y: f64, k: usize) -> Vec<(Geometry, f64)> {
        let tree = self.tree.read();
        let geometries = self.geometries.read();

        let mut results: Vec<_> = tree
            .nearest_neighbor_iter(&[x, y])
            .take(k)
            .filter_map(|entry| {
                geometries.get(&entry.data).map(|geom| {
                    let dist =
                        ((entry.geom()[0] - x).powi(2) + (entry.geom()[1] - y).powi(2)).sqrt();
                    (geom.clone(), dist)
                })
            })
            .collect();

        // Results are already sorted by distance from nearest_neighbor_iter
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.geometries.read().len()
    }

    fn clear(&self) {
        *self.tree.write() = RTree::new();
        self.geometries.write().clear();
    }

    fn index_type(&self) -> &'static str {
        "R*-tree"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use geo_types::{Geometry as GeoGeometry, LineString, Point, Polygon};

    #[test]
    fn test_rstar_tree_insert() {
        let index = RStarTree::new();
        let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));

        let id = index.insert(geom).unwrap();
        assert_eq!(index.len(), 1);
        assert!(id > 0);
    }

    #[test]
    fn test_rstar_tree_bulk_load() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
            Geometry::new(GeoGeometry::Point(Point::new(5.0, 6.0))),
        ];

        let index = RStarTree::bulk_load(geometries);
        assert_eq!(index.len(), 3);

        let (count, depth) = index.stats();
        assert_eq!(count, 3);
        assert!(depth > 0);
    }

    #[test]
    fn test_rstar_tree_query_bbox() {
        let index = RStarTree::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))))
            .unwrap();

        let results = index.query_bbox(0.0, 0.0, 6.0, 6.0);
        assert_eq!(results.len(), 2); // Points (1,1) and (5,5)
    }

    #[test]
    fn test_rstar_tree_nearest() {
        let index = RStarTree::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();

        let (geom, dist) = index.nearest(1.0, 1.0).unwrap();

        match geom.geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), 0.0, epsilon = 0.001);
                assert_relative_eq!(p.y(), 0.0, epsilon = 0.001);
            }
            _ => panic!("Expected Point"),
        }
        assert_relative_eq!(dist, 1.414, epsilon = 0.01); // sqrt(2)
    }

    #[test]
    fn test_rstar_tree_nearest_k() {
        let index = RStarTree::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))))
            .unwrap();

        let results = index.nearest_k(0.0, 0.0, 2);
        assert_eq!(results.len(), 2);

        // First result should be closest
        assert!(results[0].1 < results[1].1);
    }

    #[test]
    fn test_rstar_tree_remove() {
        let index = RStarTree::new();

        let id = index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))))
            .unwrap();
        assert_eq!(index.len(), 1);

        let removed = index.remove(id).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_rstar_tree_query_within_distance() {
        let index = RStarTree::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))))
            .unwrap();

        let results = index.query_within_distance(0.0, 0.0, 6.0);
        assert_eq!(results.len(), 2); // (0,0) and (3,4) are within distance 6
    }

    #[test]
    fn test_rstar_tree_linestring() {
        let index = RStarTree::new();

        let linestring = Geometry::new(GeoGeometry::LineString(LineString(vec![
            geo_types::Coord { x: 0.0, y: 0.0 },
            geo_types::Coord { x: 1.0, y: 1.0 },
        ])));

        index.insert(linestring).unwrap();
        assert_eq!(index.len(), 1);

        let results = index.query_bbox(-1.0, -1.0, 2.0, 2.0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_rstar_tree_polygon() {
        let index = RStarTree::new();

        let polygon = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString(vec![
                geo_types::Coord { x: 0.0, y: 0.0 },
                geo_types::Coord { x: 1.0, y: 0.0 },
                geo_types::Coord { x: 1.0, y: 1.0 },
                geo_types::Coord { x: 0.0, y: 1.0 },
                geo_types::Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        index.insert(polygon).unwrap();
        assert_eq!(index.len(), 1);

        // Query should find polygon by its centroid
        let results = index.query_bbox(0.0, 0.0, 1.0, 1.0);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_rstar_tree_clear() {
        let index = RStarTree::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))))
            .unwrap();

        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_rstar_tree_insert_batch() {
        let index = RStarTree::new();

        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0))),
            Geometry::new(GeoGeometry::Point(Point::new(5.0, 6.0))),
        ];

        let ids = index.insert_batch(geometries).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_rstar_tree_index_type() {
        let index = RStarTree::new();
        assert_eq!(index.index_type(), "R*-tree");
    }
}
