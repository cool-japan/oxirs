//! Spatial Hash index implementation
//!
//! Spatial Hash uses a fixed-size grid to partition space, providing extremely
//! fast queries for uniformly distributed data:
//!
//! - **O(1) insertion**: Direct hash-based placement
//! - **O(1) point queries**: Direct cell lookup
//! - **Very fast range queries**: Only check overlapping cells
//! - **Low memory overhead**: HashMap-based storage
//!
//! # How It Works
//!
//! 1. Space is divided into a uniform grid of cells
//! 2. Each geometry is assigned to one or more cells based on its bounding box
//! 3. Queries only examine geometries in relevant cells
//!
//! # Performance Characteristics
//!
//! - **Insert**: O(1) average case
//! - **Point query**: O(1) average case
//! - **Range query**: O(k) where k is geometries in overlapping cells
//! - **Memory**: O(n) where n is number of geometries
//!
//! # When to Use
//!
//! - Uniformly distributed data
//! - Known spatial extent
//! - Frequent point or small range queries
//! - Real-time insertions/deletions
//!
//! # When NOT to Use
//!
//! - Highly clustered data (many geometries per cell)
//! - Unknown spatial extent
//! - Large range queries spanning many cells
//! - Data with extreme coordinate ranges
//!
//! # Example
//!
//! ```ignore
//! use oxirs_geosparql::index::spatial_hash::SpatialHash;
//! use oxirs_geosparql::geometry::Geometry;
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! // Create hash with cell size 10.0
//! let index = SpatialHash::new(10.0);
//!
//! let geom = Geometry::new(GeoGeometry::Point(Point::new(15.0, 25.0)));
//! let id = index.insert(geom).unwrap();
//!
//! // Very fast point queries
//! let results = index.query_bbox(10.0, 20.0, 20.0, 30.0);
//! assert_eq!(results.len(), 1);
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use crate::index::SpatialIndexTrait;
use geo::BoundingRect;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cell key for spatial hash (grid coordinates)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey {
    x: i64,
    y: i64,
}

impl CellKey {
    fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }
}

/// Spatial Hash index using uniform grid partitioning
///
/// Provides O(1) insertions and very fast queries for uniformly distributed data.
pub struct SpatialHash {
    /// Cell size (width and height of each grid cell)
    cell_size: f64,
    /// Hash map from cell keys to geometry IDs
    cells: RwLock<HashMap<CellKey, Vec<u64>>>,
    /// ID to geometry mapping
    geometries: RwLock<HashMap<u64, Geometry>>,
    /// Next ID counter
    next_id: AtomicU64,
}

impl SpatialHash {
    /// Create a new spatial hash index with specified cell size
    ///
    /// # Arguments
    ///
    /// * `cell_size` - Size of each grid cell (should match typical query size)
    ///
    /// # Choosing Cell Size
    ///
    /// - **Too small**: Many cells per geometry, slower queries
    /// - **Too large**: Many geometries per cell, slower filtering
    /// - **Optimal**: Cell size ≈ average query range
    ///
    /// # Example
    ///
    /// ```
    /// use oxirs_geosparql::index::spatial_hash::SpatialHash;
    ///
    /// // For data in range 0-1000 with typical queries ~50 units
    /// let index = SpatialHash::new(50.0);
    /// ```
    pub fn new(cell_size: f64) -> Self {
        if cell_size <= 0.0 {
            panic!("Cell size must be positive");
        }

        Self {
            cell_size,
            cells: RwLock::new(HashMap::new()),
            geometries: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Get the cell key for a coordinate
    fn get_cell_key(&self, x: f64, y: f64) -> CellKey {
        CellKey::new(
            (x / self.cell_size).floor() as i64,
            (y / self.cell_size).floor() as i64,
        )
    }

    /// Get all cell keys that a bounding box overlaps
    fn get_overlapping_cells(
        &self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Vec<CellKey> {
        let min_cell = self.get_cell_key(min_x, min_y);
        let max_cell = self.get_cell_key(max_x, max_y);

        let mut cells = Vec::new();
        for x in min_cell.x..=max_cell.x {
            for y in min_cell.y..=max_cell.y {
                cells.push(CellKey::new(x, y));
            }
        }
        cells
    }

    /// Get bounding box of a geometry
    fn get_bbox(geom: &Geometry) -> Result<(f64, f64, f64, f64)> {
        let bbox = geom.geom.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Cannot compute bounding box".to_string())
        })?;

        Ok((bbox.min().x, bbox.min().y, bbox.max().x, bbox.max().y))
    }

    /// Get statistics about the spatial hash
    ///
    /// Returns (num_geometries, num_cells, avg_geometries_per_cell, max_geometries_per_cell)
    pub fn stats(&self) -> (usize, usize, f64, usize) {
        let geom_count = self.geometries.read().len();
        let cells = self.cells.read();
        let cell_count = cells.len();

        let max_per_cell = cells.values().map(|v| v.len()).max().unwrap_or(0);
        let avg_per_cell = if cell_count > 0 {
            geom_count as f64 / cell_count as f64
        } else {
            0.0
        };

        (geom_count, cell_count, avg_per_cell, max_per_cell)
    }

    /// Get the cell size
    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }
}

impl Default for SpatialHash {
    fn default() -> Self {
        Self::new(100.0) // Default cell size
    }
}

impl SpatialIndexTrait for SpatialHash {
    fn insert(&self, geometry: Geometry) -> Result<u64> {
        let (min_x, min_y, max_x, max_y) = Self::get_bbox(&geometry)?;
        let cells_to_insert = self.get_overlapping_cells(min_x, min_y, max_x, max_y);

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Insert into all overlapping cells
        let mut cells = self.cells.write();
        for cell_key in cells_to_insert {
            cells.entry(cell_key).or_default().push(id);
        }

        self.geometries.write().insert(id, geometry);

        Ok(id)
    }

    fn insert_batch(&self, geometries: Vec<Geometry>) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(geometries.len());
        let mut cells = self.cells.write();
        let mut geom_map = self.geometries.write();

        for geom in geometries {
            if let Ok((min_x, min_y, max_x, max_y)) = Self::get_bbox(&geom) {
                let id = self.next_id.fetch_add(1, Ordering::SeqCst);
                let cells_to_insert = self.get_overlapping_cells(min_x, min_y, max_x, max_y);

                for cell_key in cells_to_insert {
                    cells.entry(cell_key).or_default().push(id);
                }

                geom_map.insert(id, geom);
                ids.push(id);
            }
        }

        Ok(ids)
    }

    fn remove(&self, id: u64) -> Result<bool> {
        let geom = self.geometries.write().remove(&id);

        if let Some(geom) = geom {
            if let Ok((min_x, min_y, max_x, max_y)) = Self::get_bbox(&geom) {
                let cells_to_check = self.get_overlapping_cells(min_x, min_y, max_x, max_y);

                let mut cells = self.cells.write();
                for cell_key in cells_to_check {
                    if let Some(cell_geoms) = cells.get_mut(&cell_key) {
                        cell_geoms.retain(|&gid| gid != id);
                        if cell_geoms.is_empty() {
                            cells.remove(&cell_key);
                        }
                    }
                }

                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn query_bbox(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<Geometry> {
        let cells_to_check = self.get_overlapping_cells(min_x, min_y, max_x, max_y);
        let cells = self.cells.read();
        let geometries = self.geometries.read();

        let mut result_ids = std::collections::HashSet::new();

        for cell_key in cells_to_check {
            if let Some(cell_geoms) = cells.get(&cell_key) {
                result_ids.extend(cell_geoms.iter());
            }
        }

        // Filter by actual bounding box intersection
        result_ids
            .into_iter()
            .filter_map(|id| {
                geometries.get(id).and_then(|geom| {
                    if let Ok((gmin_x, gmin_y, gmax_x, gmax_y)) = Self::get_bbox(geom) {
                        // Check if bounding boxes actually intersect
                        if gmax_x >= min_x && gmin_x <= max_x && gmax_y >= min_y && gmin_y <= max_y
                        {
                            Some(geom.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    fn query_within_distance(&self, x: f64, y: f64, distance: f64) -> Vec<(Geometry, f64)> {
        // Query bounding box around point
        let results = self.query_bbox(x - distance, y - distance, x + distance, y + distance);

        // Filter by actual distance
        results
            .into_iter()
            .filter_map(|geom| {
                if let Some(point) = Self::extract_representative_point(&geom) {
                    let dx = point.x() - x;
                    let dy = point.y() - y;
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
        // Start with nearby cells and expand
        let mut search_radius = self.cell_size;
        let max_iterations = 10;

        for _ in 0..max_iterations {
            let results = self.query_within_distance(x, y, search_radius);
            if !results.is_empty() {
                // Find the closest
                return results
                    .into_iter()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            search_radius *= 2.0;
        }

        None
    }

    fn nearest_k(&self, x: f64, y: f64, k: usize) -> Vec<(Geometry, f64)> {
        let mut search_radius = self.cell_size;
        let max_iterations = 10;

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
        self.cells.write().clear();
        self.geometries.write().clear();
    }

    fn index_type(&self) -> &'static str {
        "Spatial Hash"
    }
}

impl SpatialHash {
    /// Extract a representative point from a geometry (for distance calculations)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use geo_types::{Geometry as GeoGeometry, LineString, Point, Polygon};

    #[test]
    fn test_spatial_hash_insert() {
        let index = SpatialHash::new(10.0);
        let geom = Geometry::new(GeoGeometry::Point(Point::new(15.0, 25.0)));

        let id = index.insert(geom).unwrap();
        assert_eq!(index.len(), 1);
        assert!(id > 0);
    }

    #[test]
    fn test_spatial_hash_cell_assignment() {
        let index = SpatialHash::new(10.0);

        // Point at (15, 25) should be in cell (1, 2)
        let cell = index.get_cell_key(15.0, 25.0);
        assert_eq!(cell.x, 1);
        assert_eq!(cell.y, 2);

        // Point at (0, 0) should be in cell (0, 0)
        let cell = index.get_cell_key(0.0, 0.0);
        assert_eq!(cell.x, 0);
        assert_eq!(cell.y, 0);
    }

    #[test]
    fn test_spatial_hash_query_bbox() {
        let index = SpatialHash::new(10.0);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(15.0, 15.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(25.0, 25.0))))
            .unwrap();

        // Query should find points in range
        let results = index.query_bbox(0.0, 0.0, 20.0, 20.0);
        assert_eq!(results.len(), 2); // (5,5) and (15,15)
    }

    #[test]
    fn test_spatial_hash_remove() {
        let index = SpatialHash::new(10.0);

        let id = index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(15.0, 25.0))))
            .unwrap();
        assert_eq!(index.len(), 1);

        let removed = index.remove(id).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_spatial_hash_nearest() {
        let index = SpatialHash::new(10.0);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0))))
            .unwrap();

        let (geom, dist) = index.nearest(5.0, 5.0).unwrap();

        match geom.geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), 0.0, epsilon = 0.001);
                assert_relative_eq!(p.y(), 0.0, epsilon = 0.001);
            }
            _ => panic!("Expected Point"),
        }
        assert!(dist < 10.0);
    }

    #[test]
    fn test_spatial_hash_nearest_k() {
        let index = SpatialHash::new(10.0);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(20.0, 20.0))))
            .unwrap();

        let results = index.nearest_k(0.0, 0.0, 2);
        assert_eq!(results.len(), 2);

        // Results should be sorted by distance
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_spatial_hash_within_distance() {
        let index = SpatialHash::new(10.0);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(50.0, 50.0))))
            .unwrap();

        let results = index.query_within_distance(0.0, 0.0, 10.0);
        assert_eq!(results.len(), 2); // (0,0) and (5,5)
    }

    #[test]
    fn test_spatial_hash_stats() {
        let index = SpatialHash::new(10.0);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(15.0, 15.0))))
            .unwrap();

        let (geom_count, cell_count, _avg, _max) = index.stats();
        assert_eq!(geom_count, 2);
        assert!(cell_count > 0);
    }

    #[test]
    fn test_spatial_hash_clear() {
        let index = SpatialHash::new(10.0);

        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))))
            .unwrap();
        index
            .insert(Geometry::new(GeoGeometry::Point(Point::new(15.0, 15.0))))
            .unwrap();

        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
        assert_eq!(index.stats().1, 0); // No cells
    }

    #[test]
    fn test_spatial_hash_overlapping_cells() {
        let index = SpatialHash::new(10.0);

        // Large geometry spanning multiple cells
        let polygon = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString(vec![
                geo_types::Coord { x: 0.0, y: 0.0 },
                geo_types::Coord { x: 25.0, y: 0.0 },
                geo_types::Coord { x: 25.0, y: 25.0 },
                geo_types::Coord { x: 0.0, y: 25.0 },
                geo_types::Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        index.insert(polygon).unwrap();

        // Should be found in multiple cells
        let cells = index.get_overlapping_cells(0.0, 0.0, 25.0, 25.0);
        assert!(cells.len() > 1);
    }

    #[test]
    fn test_spatial_hash_index_type() {
        let index = SpatialHash::new(10.0);
        assert_eq!(index.index_type(), "Spatial Hash");
    }

    #[test]
    fn test_spatial_hash_insert_batch() {
        let index = SpatialHash::new(10.0);

        let geometries = vec![
            Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0))),
            Geometry::new(GeoGeometry::Point(Point::new(15.0, 15.0))),
            Geometry::new(GeoGeometry::Point(Point::new(25.0, 25.0))),
        ];

        let ids = index.insert_batch(geometries).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(index.len(), 3);
    }
}
