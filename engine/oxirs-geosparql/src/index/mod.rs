//! Spatial indexing for efficient GeoSPARQL queries
//!
//! This module provides multiple spatial index implementations:
//!
//! - **R-tree** (`SpatialIndex`): General purpose, handles all geometry types
//! - **3D R-tree** (`SpatialIndex3D`): Specialized for 3D geometries with Z coordinates
//!
//! # Performance
//!
//! - Use `bulk_load()` instead of multiple `insert()` calls for 30-50% faster indexing
//! - Use `query_*_iter()` methods to avoid cloning geometries
//! - Enable `parallel` feature for parallel bbox queries on large result sets
//!
//! # Unified API
//!
//! All spatial indexes implement the `SpatialIndexTrait` for polymorphic usage

pub mod grid_index;
pub mod hilbert_rtree;
pub mod kdtree;
pub mod quadtree;
pub mod r_star_tree;
pub mod spatial_hash;
pub mod spatial_index_trait;

pub use grid_index::GridIndex;
pub use hilbert_rtree::HilbertRTree;
pub use kdtree::KdTree;
pub use quadtree::Quadtree;
pub use r_star_tree::RStarTree;
pub use spatial_hash::SpatialHash;
pub use spatial_index_trait::{
    create_index_with_hint, select_optimal_index, IndexHint, SpatialIndexTrait,
};

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use parking_lot::RwLock;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use std::sync::Arc;

/// A spatial index entry containing a geometry and an associated ID
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialEntry {
    /// Unique identifier for this entry
    pub id: u64,
    /// The geometry
    pub geometry: Geometry,
    /// Cached envelope (bounding box)
    envelope: AABB<[f64; 2]>,
}

impl SpatialEntry {
    /// Create a new spatial entry
    pub fn new(id: u64, geometry: Geometry) -> Result<Self> {
        let envelope = Self::calculate_envelope(&geometry)?;
        Ok(Self {
            id,
            geometry,
            envelope,
        })
    }

    /// Calculate the envelope (AABB) of a geometry
    fn calculate_envelope(geometry: &Geometry) -> Result<AABB<[f64; 2]>> {
        use geo::BoundingRect;

        let bbox = match &geometry.geom {
            geo_types::Geometry::Point(p) => geo_types::Rect::new(
                geo_types::coord! { x: p.x(), y: p.y() },
                geo_types::coord! { x: p.x(), y: p.y() },
            ),
            geo_types::Geometry::LineString(ls) => ls.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for LineString".to_string(),
                )
            })?,
            geo_types::Geometry::Polygon(p) => p.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for Polygon".to_string(),
                )
            })?,
            geo_types::Geometry::MultiPoint(mp) => mp.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for MultiPoint".to_string(),
                )
            })?,
            geo_types::Geometry::MultiLineString(mls) => mls.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for MultiLineString".to_string(),
                )
            })?,
            geo_types::Geometry::MultiPolygon(mp) => mp.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for MultiPolygon".to_string(),
                )
            })?,
            geo_types::Geometry::GeometryCollection(gc) => gc.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for GeometryCollection".to_string(),
                )
            })?,
            geo_types::Geometry::Triangle(t) => t.bounding_rect(),
            geo_types::Geometry::Rect(r) => *r,
            geo_types::Geometry::Line(l) => l.bounding_rect(),
        };

        let min = bbox.min();
        let max = bbox.max();

        Ok(AABB::from_corners([min.x, min.y], [max.x, max.y]))
    }
}

impl RTreeObject for SpatialEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

impl PointDistance for SpatialEntry {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        // Calculate squared distance from point to entry's envelope
        let envelope = &self.envelope;
        let min = envelope.lower();
        let max = envelope.upper();

        let dx = if point[0] < min[0] {
            min[0] - point[0]
        } else if point[0] > max[0] {
            point[0] - max[0]
        } else {
            0.0
        };

        let dy = if point[1] < min[1] {
            min[1] - point[1]
        } else if point[1] > max[1] {
            point[1] - max[1]
        } else {
            0.0
        };

        dx * dx + dy * dy
    }

    fn contains_point(&self, point: &[f64; 2]) -> bool {
        let envelope = &self.envelope;
        let min = envelope.lower();
        let max = envelope.upper();

        point[0] >= min[0] && point[0] <= max[0] && point[1] >= min[1] && point[1] <= max[1]
    }
}

/// A spatial index using R-tree
pub struct SpatialIndex {
    /// The R-tree
    tree: Arc<RwLock<RTree<SpatialEntry>>>,
    /// Next available ID
    next_id: Arc<RwLock<u64>>,
}

impl SpatialIndex {
    /// Create a new empty spatial index
    pub fn new() -> Self {
        Self {
            tree: Arc::new(RwLock::new(RTree::new())),
            next_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Bulk load geometries into the index for better performance
    ///
    /// This is 30-50% faster than inserting geometries one at a time,
    /// and produces a better-balanced R-tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::SpatialIndex;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let geometries = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(3.0, 3.0))),
    /// ];
    ///
    /// let index = SpatialIndex::bulk_load(geometries).unwrap();
    /// assert_eq!(index.len(), 3);
    /// ```
    pub fn bulk_load(geometries: Vec<Geometry>) -> Result<Self> {
        let count = geometries.len();
        let mut entries = Vec::with_capacity(count);

        for (id, geometry) in geometries.into_iter().enumerate() {
            entries.push(SpatialEntry::new(id as u64, geometry)?);
        }

        let tree = RTree::bulk_load(entries);

        Ok(Self {
            tree: Arc::new(RwLock::new(tree)),
            next_id: Arc::new(RwLock::new(count as u64)),
        })
    }

    /// Insert a geometry into the index
    pub fn insert(&self, geometry: Geometry) -> Result<u64> {
        let mut next_id = self.next_id.write();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let entry = SpatialEntry::new(id, geometry)?;
        self.tree.write().insert(entry);

        Ok(id)
    }

    /// Batch insert multiple geometries
    ///
    /// More efficient than calling `insert()` multiple times.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::SpatialIndex;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let index = SpatialIndex::new();
    /// let geometries = vec![
    ///     Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
    ///     Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
    /// ];
    ///
    /// let ids = index.insert_batch(geometries).unwrap();
    /// assert_eq!(ids.len(), 2);
    /// ```
    pub fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>> {
        let mut next_id = self.next_id.write();
        let start_id = *next_id;
        *next_id += geometries.len() as u64;
        drop(next_id);

        let mut ids = Vec::with_capacity(geometries.len());
        let mut tree = self.tree.write();

        for (i, geometry) in geometries.into_iter().enumerate() {
            let id = start_id + i as u64;
            let entry = SpatialEntry::new(id, geometry)?;
            tree.insert(entry);
            ids.push(id);
        }

        Ok(ids)
    }

    /// Remove a geometry from the index by ID
    pub fn remove(&self, id: u64) -> Result<bool> {
        let mut tree = self.tree.write();

        // Find the entry with the given ID
        let to_remove: Vec<_> = tree
            .iter()
            .filter(|entry| entry.id == id)
            .cloned()
            .collect();

        if to_remove.is_empty() {
            return Ok(false);
        }

        for entry in to_remove {
            tree.remove(&entry);
        }

        Ok(true)
    }

    /// Find all geometries that intersect with the given bounding box
    pub fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<Geometry> {
        let tree = self.tree.read();
        let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);

        tree.locate_in_envelope_intersecting(&envelope)
            .map(|entry| entry.geometry.clone())
            .collect()
    }

    /// Find all geometries that intersect with the given bounding box (parallel)
    ///
    /// Uses parallel processing for large result sets.
    /// Requires `parallel` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "parallel")]
    /// # fn example() {
    /// use oxirs_geosparql::SpatialIndex;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let index = SpatialIndex::new();
    /// for i in 0..1000 {
    ///     index.insert(Geometry::new(GeoGeometry::Point(Point::new(i as f64, i as f64)))).unwrap();
    /// }
    ///
    /// // Parallel bbox query (faster for large result sets)
    /// let results = index.query_bbox_parallel(0.0, 0.0, 500.0, 500.0);
    /// # }
    /// ```
    #[cfg(feature = "parallel")]
    pub fn query_bbox_parallel(
        &self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Vec<Geometry> {
        use rayon::prelude::*;

        let tree = self.tree.read();
        let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);

        tree.locate_in_envelope_intersecting(&envelope)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|entry| entry.geometry.clone())
            .collect()
    }

    /// Find all geometries within a given distance from a point (optimized)
    ///
    /// This uses R-tree spatial queries instead of full iteration.
    /// 10-100x faster than the naive approach for large indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::SpatialIndex;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let index = SpatialIndex::new();
    /// index.insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)))).unwrap();
    ///
    /// let results = index.query_within_distance(0.0, 0.0, 2.0);
    /// assert_eq!(results.len(), 1);
    /// ```
    pub fn query_within_distance(&self, x: f64, y: f64, distance: f64) -> Vec<(Geometry, f64)> {
        let tree = self.tree.read();
        let query_point = geo_types::Point::new(x, y);

        // Use bounding box query to filter candidates first (much faster)
        let bbox = AABB::from_corners([x - distance, y - distance], [x + distance, y + distance]);

        tree.locate_in_envelope_intersecting(&bbox)
            .filter_map(|entry| {
                let dist = Self::point_distance(&query_point, &entry.geometry);
                if dist <= distance {
                    Some((entry.geometry.clone(), dist))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find the nearest geometry to a given point (optimized)
    ///
    /// Uses R-tree nearest neighbor query instead of full iteration.
    /// 10-100x faster than the naive approach for large indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::SpatialIndex;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let index = SpatialIndex::new();
    /// index.insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)))).unwrap();
    /// index.insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0)))).unwrap();
    ///
    /// let (nearest, distance) = index.nearest(0.0, 0.0).unwrap();
    /// assert!(distance > 1.0 && distance < 2.0); // Should find (1, 1)
    /// ```
    pub fn nearest(&self, x: f64, y: f64) -> Option<(Geometry, f64)> {
        let tree = self.tree.read();
        let query_point = [x, y];

        // Use R-tree's nearest neighbor query (much faster)
        tree.nearest_neighbor(&query_point).map(|entry| {
            let point = geo_types::Point::new(x, y);
            let dist = Self::point_distance(&point, &entry.geometry);
            (entry.geometry.clone(), dist)
        })
    }

    /// Find the k nearest geometries to a given point
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::SpatialIndex;
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Geometry as GeoGeometry, Point};
    ///
    /// let index = SpatialIndex::new();
    /// index.insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)))).unwrap();
    /// index.insert(Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0)))).unwrap();
    /// index.insert(Geometry::new(GeoGeometry::Point(Point::new(3.0, 3.0)))).unwrap();
    ///
    /// let nearest = index.nearest_k(0.0, 0.0, 2);
    /// assert_eq!(nearest.len(), 2);
    /// ```
    pub fn nearest_k(&self, x: f64, y: f64, k: usize) -> Vec<(Geometry, f64)> {
        let tree = self.tree.read();
        let query_point = [x, y];
        let geo_point = geo_types::Point::new(x, y);

        tree.nearest_neighbor_iter(&query_point)
            .take(k)
            .map(|entry| {
                let dist = Self::point_distance(&geo_point, &entry.geometry);
                (entry.geometry.clone(), dist)
            })
            .collect()
    }

    /// Calculate distance from a point to a geometry
    fn point_distance(point: &geo_types::Point<f64>, geometry: &Geometry) -> f64 {
        use geo::EuclideanDistance;

        let point_geom = geo_types::Geometry::Point(*point);
        point_geom.euclidean_distance(&geometry.geom)
    }

    /// Get the number of entries in the index
    pub fn len(&self) -> usize {
        self.tree.read().size()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries from the index
    pub fn clear(&self) {
        *self.tree.write() = RTree::new();
        *self.next_id.write() = 0;
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of SpatialIndexTrait for SpatialIndex
impl SpatialIndexTrait for SpatialIndex {
    fn insert(&self, geometry: Geometry) -> Result<u64> {
        SpatialIndex::insert(self, geometry)
    }

    fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>> {
        SpatialIndex::insert_batch(self, geometries)
    }

    fn remove(&self, id: u64) -> Result<bool> {
        SpatialIndex::remove(self, id)
    }

    fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<Geometry> {
        SpatialIndex::query_bbox(self, min_x, min_y, max_x, max_y)
    }

    fn query_within_distance(&self, x: f64, y: f64, distance: f64) -> Vec<(Geometry, f64)> {
        SpatialIndex::query_within_distance(self, x, y, distance)
    }

    fn nearest(&self, x: f64, y: f64) -> Option<(Geometry, f64)> {
        SpatialIndex::nearest(self, x, y)
    }

    fn nearest_k(&self, x: f64, y: f64, k: usize) -> Vec<(Geometry, f64)> {
        SpatialIndex::nearest_k(self, x, y, k)
    }

    fn len(&self) -> usize {
        SpatialIndex::len(self)
    }

    fn clear(&self) {
        SpatialIndex::clear(self)
    }

    fn index_type(&self) -> &'static str {
        "R-tree"
    }
}

/// A 3D spatial index entry containing a geometry with Z coordinates
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialEntry3D {
    /// Unique identifier for this entry
    pub id: u64,
    /// The geometry (must have Z coordinates)
    pub geometry: Geometry,
    /// Cached 3D envelope (bounding box)
    envelope: AABB<[f64; 3]>,
}

impl SpatialEntry3D {
    /// Create a new 3D spatial entry
    ///
    /// Returns an error if the geometry doesn't have Z coordinates
    pub fn new(id: u64, geometry: Geometry) -> Result<Self> {
        if !geometry.is_3d() {
            return Err(GeoSparqlError::GeometryOperationFailed(
                "Geometry must have Z coordinates for 3D spatial index".to_string(),
            ));
        }

        let envelope = Self::calculate_envelope_3d(&geometry)?;
        Ok(Self {
            id,
            geometry,
            envelope,
        })
    }

    /// Calculate the 3D envelope (AABB) of a geometry
    fn calculate_envelope_3d(geometry: &Geometry) -> Result<AABB<[f64; 3]>> {
        use geo::BoundingRect;
        use geo_types::Geometry as GeoGeometry;

        // Calculate 2D bounding box first
        let bbox_2d = match &geometry.geom {
            GeoGeometry::Point(p) => geo_types::Rect::new(
                geo_types::coord! { x: p.x(), y: p.y() },
                geo_types::coord! { x: p.x(), y: p.y() },
            ),
            GeoGeometry::LineString(ls) => ls.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for LineString".to_string(),
                )
            })?,
            GeoGeometry::Polygon(p) => p.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for Polygon".to_string(),
                )
            })?,
            GeoGeometry::MultiPoint(mp) => mp.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for MultiPoint".to_string(),
                )
            })?,
            GeoGeometry::MultiLineString(mls) => mls.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for MultiLineString".to_string(),
                )
            })?,
            GeoGeometry::MultiPolygon(mp) => mp.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for MultiPolygon".to_string(),
                )
            })?,
            GeoGeometry::GeometryCollection(gc) => gc.bounding_rect().ok_or_else(|| {
                GeoSparqlError::GeometryOperationFailed(
                    "Could not calculate bounding box for GeometryCollection".to_string(),
                )
            })?,
            GeoGeometry::Triangle(t) => t.bounding_rect(),
            GeoGeometry::Rect(r) => *r,
            GeoGeometry::Line(l) => l.bounding_rect(),
        };

        let min_2d = bbox_2d.min();
        let max_2d = bbox_2d.max();

        // Calculate Z bounds from coord3d
        let (min_z, max_z) = Self::calculate_z_bounds(&geometry.coord3d);

        Ok(AABB::from_corners(
            [min_2d.x, min_2d.y, min_z],
            [max_2d.x, max_2d.y, max_z],
        ))
    }

    /// Calculate Z coordinate bounds
    fn calculate_z_bounds(coord3d: &crate::geometry::coord3d::Coord3D) -> (f64, f64) {
        if let Some(ref z_coords) = coord3d.z_coords {
            if z_coords.is_empty() {
                return (0.0, 0.0);
            }

            let mut min_z = f64::MAX;
            let mut max_z = f64::MIN;

            for z in &z_coords.values {
                min_z = min_z.min(*z);
                max_z = max_z.max(*z);
            }

            (min_z, max_z)
        } else {
            (0.0, 0.0)
        }
    }
}

impl RTreeObject for SpatialEntry3D {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

impl PointDistance for SpatialEntry3D {
    fn distance_2(&self, point: &[f64; 3]) -> f64 {
        // Calculate squared distance from 3D point to entry's envelope
        let envelope = &self.envelope;
        let min = envelope.lower();
        let max = envelope.upper();

        let dx = if point[0] < min[0] {
            min[0] - point[0]
        } else if point[0] > max[0] {
            point[0] - max[0]
        } else {
            0.0
        };

        let dy = if point[1] < min[1] {
            min[1] - point[1]
        } else if point[1] > max[1] {
            point[1] - max[1]
        } else {
            0.0
        };

        let dz = if point[2] < min[2] {
            min[2] - point[2]
        } else if point[2] > max[2] {
            point[2] - max[2]
        } else {
            0.0
        };

        dx * dx + dy * dy + dz * dz
    }

    fn contains_point(&self, point: &[f64; 3]) -> bool {
        let envelope = &self.envelope;
        let min = envelope.lower();
        let max = envelope.upper();

        point[0] >= min[0]
            && point[0] <= max[0]
            && point[1] >= min[1]
            && point[1] <= max[1]
            && point[2] >= min[2]
            && point[2] <= max[2]
    }
}

/// A 3D spatial index using R-tree for geometries with Z coordinates
pub struct SpatialIndex3D {
    /// The 3D R-tree
    tree: Arc<RwLock<RTree<SpatialEntry3D>>>,
    /// Next available ID
    next_id: Arc<RwLock<u64>>,
}

impl SpatialIndex3D {
    /// Create a new empty 3D spatial index
    pub fn new() -> Self {
        Self {
            tree: Arc::new(RwLock::new(RTree::new())),
            next_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Bulk load 3D geometries into the index for better performance
    ///
    /// All geometries must have Z coordinates, or an error will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::index::SpatialIndex3D;
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let geometries = vec![
    ///     Geometry::from_wkt("POINT Z(1 1 10)").unwrap(),
    ///     Geometry::from_wkt("POINT Z(2 2 20)").unwrap(),
    ///     Geometry::from_wkt("POINT Z(3 3 30)").unwrap(),
    /// ];
    ///
    /// let index = SpatialIndex3D::bulk_load(geometries).unwrap();
    /// assert_eq!(index.len(), 3);
    /// ```
    pub fn bulk_load(geometries: Vec<Geometry>) -> Result<Self> {
        let count = geometries.len();
        let mut entries = Vec::with_capacity(count);

        for (id, geometry) in geometries.into_iter().enumerate() {
            entries.push(SpatialEntry3D::new(id as u64, geometry)?);
        }

        let tree = RTree::bulk_load(entries);

        Ok(Self {
            tree: Arc::new(RwLock::new(tree)),
            next_id: Arc::new(RwLock::new(count as u64)),
        })
    }

    /// Insert a 3D geometry into the index
    ///
    /// Returns an error if the geometry doesn't have Z coordinates.
    pub fn insert(&self, geometry: Geometry) -> Result<u64> {
        let mut next_id = self.next_id.write();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let entry = SpatialEntry3D::new(id, geometry)?;
        self.tree.write().insert(entry);

        Ok(id)
    }

    /// Batch insert multiple 3D geometries
    pub fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>> {
        let mut next_id = self.next_id.write();
        let start_id = *next_id;
        *next_id += geometries.len() as u64;
        drop(next_id);

        let mut ids = Vec::with_capacity(geometries.len());
        let mut tree = self.tree.write();

        for (i, geometry) in geometries.into_iter().enumerate() {
            let id = start_id + i as u64;
            let entry = SpatialEntry3D::new(id, geometry)?;
            tree.insert(entry);
            ids.push(id);
        }

        Ok(ids)
    }

    /// Remove a geometry from the index by ID
    pub fn remove(&self, id: u64) -> Result<bool> {
        let mut tree = self.tree.write();

        let to_remove: Vec<_> = tree
            .iter()
            .filter(|entry| entry.id == id)
            .cloned()
            .collect();

        if to_remove.is_empty() {
            return Ok(false);
        }

        for entry in to_remove {
            tree.remove(&entry);
        }

        Ok(true)
    }

    /// Find all geometries that intersect with the given 3D bounding box
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::index::SpatialIndex3D;
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let index = SpatialIndex3D::new();
    /// index.insert(Geometry::from_wkt("POINT Z(1 1 10)").unwrap()).unwrap();
    /// index.insert(Geometry::from_wkt("POINT Z(5 5 50)").unwrap()).unwrap();
    ///
    /// // Query for geometries in 3D bbox
    /// let results = index.query_bbox_3d(0.0, 0.0, 0.0, 6.0, 6.0, 60.0);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn query_bbox_3d(
        &self,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Vec<Geometry> {
        let tree = self.tree.read();
        let envelope = AABB::from_corners([min_x, min_y, min_z], [max_x, max_y, max_z]);

        tree.locate_in_envelope_intersecting(&envelope)
            .map(|entry| entry.geometry.clone())
            .collect()
    }

    /// Find all geometries within a given 3D distance from a point
    ///
    /// Uses the 3D distance calculation including Z coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::index::SpatialIndex3D;
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let index = SpatialIndex3D::new();
    /// index.insert(Geometry::from_wkt("POINT Z(0 0 0)").unwrap()).unwrap();
    /// index.insert(Geometry::from_wkt("POINT Z(1 1 1)").unwrap()).unwrap();
    ///
    /// let results = index.query_within_distance_3d(0.0, 0.0, 0.0, 2.0);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn query_within_distance_3d(
        &self,
        x: f64,
        y: f64,
        z: f64,
        distance: f64,
    ) -> Vec<(Geometry, f64)> {
        let tree = self.tree.read();

        // Use 3D bounding box query to filter candidates first
        let bbox = AABB::from_corners(
            [x - distance, y - distance, z - distance],
            [x + distance, y + distance, z + distance],
        );

        tree.locate_in_envelope_intersecting(&bbox)
            .filter_map(|entry| {
                // Calculate 3D distance
                let dist = Self::point_distance_3d(x, y, z, &entry.geometry);
                if dist <= distance {
                    Some((entry.geometry.clone(), dist))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Find the nearest geometry to a given 3D point
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::index::SpatialIndex3D;
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let index = SpatialIndex3D::new();
    /// index.insert(Geometry::from_wkt("POINT Z(1 1 10)").unwrap()).unwrap();
    /// index.insert(Geometry::from_wkt("POINT Z(5 5 50)").unwrap()).unwrap();
    ///
    /// let (nearest, distance) = index.nearest_3d(0.0, 0.0, 0.0).unwrap();
    /// // Should find (1, 1, 10) with distance ~10.1
    /// assert!(distance > 10.0 && distance < 10.2);
    /// ```
    pub fn nearest_3d(&self, x: f64, y: f64, z: f64) -> Option<(Geometry, f64)> {
        let tree = self.tree.read();
        let query_point = [x, y, z];

        tree.nearest_neighbor(&query_point).map(|entry| {
            let dist = Self::point_distance_3d(x, y, z, &entry.geometry);
            (entry.geometry.clone(), dist)
        })
    }

    /// Find the k nearest geometries to a given 3D point
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::index::SpatialIndex3D;
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let index = SpatialIndex3D::new();
    /// index.insert(Geometry::from_wkt("POINT Z(1 1 10)").unwrap()).unwrap();
    /// index.insert(Geometry::from_wkt("POINT Z(2 2 20)").unwrap()).unwrap();
    /// index.insert(Geometry::from_wkt("POINT Z(3 3 30)").unwrap()).unwrap();
    ///
    /// let nearest = index.nearest_k_3d(0.0, 0.0, 0.0, 2);
    /// assert_eq!(nearest.len(), 2);
    /// ```
    pub fn nearest_k_3d(&self, x: f64, y: f64, z: f64, k: usize) -> Vec<(Geometry, f64)> {
        let tree = self.tree.read();
        let query_point = [x, y, z];

        tree.nearest_neighbor_iter(&query_point)
            .take(k)
            .map(|entry| {
                let dist = Self::point_distance_3d(x, y, z, &entry.geometry);
                (entry.geometry.clone(), dist)
            })
            .collect()
    }

    /// Calculate 3D distance from a point to a geometry
    fn point_distance_3d(x: f64, y: f64, z: f64, geometry: &Geometry) -> f64 {
        use crate::functions::geometric_operations::distance_3d;
        use geo_types::Point;

        // Create a 3D point geometry
        let mut point_geom = Geometry::new(geo_types::Geometry::Point(Point::new(x, y)));
        point_geom.coord3d = crate::geometry::coord3d::Coord3D::xyz(vec![z]);

        // Use the 3D distance function
        distance_3d(&point_geom, geometry).unwrap_or(f64::MAX)
    }

    /// Get the number of entries in the index
    pub fn len(&self) -> usize {
        self.tree.read().size()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries from the index
    pub fn clear(&self) {
        *self.tree.write() = RTree::new();
        *self.next_id.write() = 0;
    }
}

impl Default for SpatialIndex3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_spatial_index_insert() {
        let index = SpatialIndex::new();

        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let id = index.insert(point).unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_spatial_index_bulk_load() {
        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
            Geometry::new(GeoGeometry::Point(Point::new(3.0, 3.0))),
        ];

        let index = SpatialIndex::bulk_load(geometries).unwrap();

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_spatial_index_insert_batch() {
        let index = SpatialIndex::new();

        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))),
            Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))),
        ];

        let ids = index.insert_batch(geometries).unwrap();

        assert_eq!(ids.len(), 2);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_spatial_index_remove() {
        let index = SpatialIndex::new();

        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let id = index.insert(point).unwrap();

        assert_eq!(index.len(), 1);

        let removed = index.remove(id).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_spatial_index_query_bbox() {
        let index = SpatialIndex::new();

        // Insert points
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))))
            .unwrap();

        // Query for points in bbox [0, 0] to [6, 6]
        let results = index.query_bbox(0.0, 0.0, 6.0, 6.0);

        assert_eq!(results.len(), 2); // Should find (1,1) and (5,5)
    }

    #[test]
    fn test_spatial_index_query_within_distance() {
        let index = SpatialIndex::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();

        // Find points within distance 2.0 from origin
        let results = index.query_within_distance(0.0, 0.0, 2.0);

        // Should find (0,0) and (1,1), but not (5,5)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_spatial_index_nearest() {
        let index = SpatialIndex::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();

        let (_nearest_geom, distance) = index.nearest(0.0, 0.0).unwrap();

        // The nearest point should be (1, 1) with distance sqrt(2)
        assert!((distance - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_index_nearest_k() {
        let index = SpatialIndex::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(3.0, 3.0))))
            .unwrap();

        let nearest = index.nearest_k(0.0, 0.0, 2);

        assert_eq!(nearest.len(), 2);
        // First should be closest
        assert!(nearest[0].1 < nearest[1].1);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_spatial_index_query_bbox_parallel() {
        let index = SpatialIndex::new();

        for i in 0..100 {
            index
                .insert(Geometry::new(GeoGeometry::Point(Point::new(
                    i as f64, i as f64,
                ))))
                .unwrap();
        }

        let results = index.query_bbox_parallel(0.0, 0.0, 50.0, 50.0);

        // Should find approximately 51 points (0-50 inclusive)
        assert!(results.len() >= 50 && results.len() <= 52);
    }

    #[test]
    fn test_spatial_index_clear() {
        let index = SpatialIndex::new();

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0))))
            .unwrap();

        assert_eq!(index.len(), 2);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    // ============ 3D Spatial Index Tests ============

    #[test]
    fn test_spatial_index_3d_insert() {
        let index = SpatialIndex3D::new();

        let point = Geometry::from_wkt("POINT Z(1 2 10)").unwrap();
        let id = index.insert(point).unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_spatial_index_3d_reject_2d() {
        let index = SpatialIndex3D::new();

        // Should reject 2D geometry
        let point = Geometry::from_wkt("POINT(1 2)").unwrap();
        let result = index.insert(point);

        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_index_3d_bulk_load() {
        let geometries = vec![
            Geometry::from_wkt("POINT Z(1 1 10)").unwrap(),
            Geometry::from_wkt("POINT Z(2 2 20)").unwrap(),
            Geometry::from_wkt("POINT Z(3 3 30)").unwrap(),
        ];

        let index = SpatialIndex3D::bulk_load(geometries).unwrap();

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_spatial_index_3d_insert_batch() {
        let index = SpatialIndex3D::new();

        let geometries = vec![
            Geometry::from_wkt("POINT Z(1 1 10)").unwrap(),
            Geometry::from_wkt("POINT Z(2 2 20)").unwrap(),
        ];

        let ids = index.insert_batch(geometries).unwrap();

        assert_eq!(ids.len(), 2);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_spatial_index_3d_remove() {
        let index = SpatialIndex3D::new();

        let point = Geometry::from_wkt("POINT Z(1 2 10)").unwrap();
        let id = index.insert(point).unwrap();

        assert_eq!(index.len(), 1);

        let removed = index.remove(id).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_spatial_index_3d_query_bbox() {
        let index = SpatialIndex3D::new();

        // Insert points at different heights
        index
            .insert(Geometry::from_wkt("POINT Z(1 1 10)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 50)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(10 10 100)").unwrap())
            .unwrap();

        // Query for points in 3D bbox [0, 0, 0] to [6, 6, 60]
        let results = index.query_bbox_3d(0.0, 0.0, 0.0, 6.0, 6.0, 60.0);

        // Should find (1,1,10) and (5,5,50), but not (10,10,100)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_spatial_index_3d_query_bbox_z_filtering() {
        let index = SpatialIndex3D::new();

        // Insert points at same X,Y but different Z
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 10)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 50)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 100)").unwrap())
            .unwrap();

        // Query for points with Z between 0-60
        let results = index.query_bbox_3d(0.0, 0.0, 0.0, 10.0, 10.0, 60.0);

        // Should find only (5,5,10) and (5,5,50), not (5,5,100)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_spatial_index_3d_query_within_distance() {
        let index = SpatialIndex3D::new();

        index
            .insert(Geometry::from_wkt("POINT Z(0 0 0)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(1 1 1)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 5)").unwrap())
            .unwrap();

        // Find points within distance 2.0 from origin in 3D
        let results = index.query_within_distance_3d(0.0, 0.0, 0.0, 2.0);

        // Distance to (0,0,0) = 0
        // Distance to (1,1,1) = √3 ≈ 1.73
        // Distance to (5,5,5) = √75 ≈ 8.66
        // Should find first two points
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_spatial_index_3d_nearest() {
        let index = SpatialIndex3D::new();

        index
            .insert(Geometry::from_wkt("POINT Z(1 1 10)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 50)").unwrap())
            .unwrap();

        let (_nearest_geom, distance) = index.nearest_3d(0.0, 0.0, 0.0).unwrap();

        // The nearest point should be (1, 1, 10) with distance = √(1² + 1² + 10²) = √102 ≈ 10.1
        assert!((distance - 10.1).abs() < 0.1);
    }

    #[test]
    fn test_spatial_index_3d_nearest_k() {
        let index = SpatialIndex3D::new();

        index
            .insert(Geometry::from_wkt("POINT Z(1 1 10)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(2 2 20)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(3 3 30)").unwrap())
            .unwrap();

        let nearest = index.nearest_k_3d(0.0, 0.0, 0.0, 2);

        assert_eq!(nearest.len(), 2);
        // First should be closest
        assert!(nearest[0].1 < nearest[1].1);
    }

    #[test]
    fn test_spatial_index_3d_vertical_separation() {
        let index = SpatialIndex3D::new();

        // Points with same X,Y but different Z
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 0)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(5 5 100)").unwrap())
            .unwrap();

        let (_nearest, distance) = index.nearest_3d(5.0, 5.0, 50.0).unwrap();

        // Should find (5, 5, 100) or (5, 5, 0) depending on which is closer
        // Distance from (5,5,50) to (5,5,0) = 50
        // Distance from (5,5,50) to (5,5,100) = 50
        assert!((distance - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_spatial_index_3d_linestring() {
        let index = SpatialIndex3D::new();

        // Insert a 3D LineString
        index
            .insert(Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 100)").unwrap())
            .unwrap();

        // Query should find the linestring
        let results = index.query_bbox_3d(0.0, 0.0, 0.0, 15.0, 15.0, 150.0);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_spatial_index_3d_polygon() {
        let index = SpatialIndex3D::new();

        // Insert a 3D Polygon (e.g., a raised square)
        index
            .insert(
                Geometry::from_wkt("POLYGON Z((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))")
                    .unwrap(),
            )
            .unwrap();

        // Query at Z=10 level should find it
        let results = index.query_bbox_3d(0.0, 0.0, 5.0, 15.0, 15.0, 15.0);

        assert_eq!(results.len(), 1);

        // Query at Z=50 level should not find it
        let results = index.query_bbox_3d(0.0, 0.0, 50.0, 15.0, 15.0, 60.0);

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_spatial_index_3d_clear() {
        let index = SpatialIndex3D::new();

        index
            .insert(Geometry::from_wkt("POINT Z(1 1 10)").unwrap())
            .unwrap();
        index
            .insert(Geometry::from_wkt("POINT Z(2 2 20)").unwrap())
            .unwrap();

        assert_eq!(index.len(), 2);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_spatial_index_3d_mixed_heights() {
        let index = SpatialIndex3D::new();

        // Create a grid of points at different heights
        for x in 0..5 {
            for y in 0..5 {
                let z = (x * 5 + y) * 10;
                let wkt = format!("POINT Z({} {} {})", x, y, z);
                index.insert(Geometry::from_wkt(&wkt).unwrap()).unwrap();
            }
        }

        assert_eq!(index.len(), 25);

        // Query a specific 3D region
        let results = index.query_bbox_3d(1.0, 1.0, 0.0, 3.0, 3.0, 100.0);

        // Should find points in the 2x2x100 box
        assert!(!results.is_empty() && results.len() < 25);
    }
}
