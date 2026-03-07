//! Spatial grid index for fast bounding-box queries.
//!
//! Implements a uniform-grid spatial index dividing the world into an N×N grid.
//! Each inserted entry is placed into all cells its bounding box overlaps,
//! enabling fast O(k) bounding-box intersection queries.

use std::collections::HashMap;

/// An axis-aligned 2D bounding box
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Minimum X coordinate (west boundary).
    pub min_x: f64,
    /// Minimum Y coordinate (south boundary).
    pub min_y: f64,
    /// Maximum X coordinate (east boundary).
    pub max_x: f64,
    /// Maximum Y coordinate (north boundary).
    pub max_y: f64,
}

impl BoundingBox {
    /// Create a new bounding box, normalising so that min ≤ max
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x: min_x.min(max_x),
            min_y: min_y.min(max_y),
            max_x: min_x.max(max_x),
            max_y: min_y.max(max_y),
        }
    }

    /// True if this bbox has any overlap with `other` (including touching edges)
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// True if the point (x, y) lies inside or on the boundary of this bbox
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Centre of the bounding box
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }

    /// Area of the bounding box
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// Minimum Euclidean distance from this bbox to the point (x, y).
    /// Returns 0.0 if the point is inside or on the boundary.
    pub fn distance_to_point(&self, x: f64, y: f64) -> f64 {
        let dx = if x < self.min_x {
            self.min_x - x
        } else if x > self.max_x {
            x - self.max_x
        } else {
            0.0
        };
        let dy = if y < self.min_y {
            self.min_y - y
        } else if y > self.max_y {
            y - self.max_y
        } else {
            0.0
        };
        (dx * dx + dy * dy).sqrt()
    }
}

/// A single indexed spatial entry
#[derive(Debug, Clone)]
pub struct SpatialEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// Axis-aligned bounding box of the entry.
    pub bbox: BoundingBox,
    /// Arbitrary key-value metadata attached to this entry.
    pub data: HashMap<String, String>,
}

impl SpatialEntry {
    /// Create a new spatial entry
    pub fn new(id: impl Into<String>, bbox: BoundingBox) -> Self {
        Self {
            id: id.into(),
            bbox,
            data: HashMap::new(),
        }
    }

    /// Attach user data to this entry
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }
}

/// A single cell in the spatial grid
#[derive(Debug, Clone, Default)]
pub struct GridCell {
    /// IDs of entries that overlap this cell
    pub entries: Vec<String>,
}

/// Configuration for the spatial index
#[derive(Debug, Clone)]
pub struct SpatialIndexConfig {
    /// Number of cells per side (grid is grid_size × grid_size)
    pub grid_size: usize,
    /// World min corner
    pub world_min: (f64, f64),
    /// World max corner
    pub world_max: (f64, f64),
}

impl SpatialIndexConfig {
    /// Create a new config with default 16×16 grid covering the given world
    pub fn new(grid_size: usize, world_min: (f64, f64), world_max: (f64, f64)) -> Self {
        Self {
            grid_size,
            world_min,
            world_max,
        }
    }
}

impl Default for SpatialIndexConfig {
    fn default() -> Self {
        Self {
            grid_size: 16,
            world_min: (0.0, 0.0),
            world_max: (1.0, 1.0),
        }
    }
}

/// Uniform-grid spatial index
pub struct SpatialIndex {
    config: SpatialIndexConfig,
    entries: HashMap<String, SpatialEntry>,
    grid: Vec<Vec<GridCell>>,
}

impl SpatialIndex {
    /// Create a new spatial index with the given configuration
    pub fn new(config: SpatialIndexConfig) -> Self {
        let n = config.grid_size.max(1);
        let grid = (0..n)
            .map(|_| (0..n).map(|_| GridCell::default()).collect())
            .collect();
        Self {
            config,
            entries: HashMap::new(),
            grid,
        }
    }

    // ── Configuration helpers ─────────────────────────────────────────────

    fn cell_width(&self) -> f64 {
        (self.config.world_max.0 - self.config.world_min.0) / self.config.grid_size as f64
    }

    fn cell_height(&self) -> f64 {
        (self.config.world_max.1 - self.config.world_min.1) / self.config.grid_size as f64
    }

    fn coord_to_col(&self, x: f64) -> Option<usize> {
        let rel = x - self.config.world_min.0;
        let w = self.cell_width();
        if w <= 0.0 {
            return None;
        }
        // Explicit negative-rel check: casting a small negative float to isize
        // truncates toward zero (e.g. -0.1_f64 as isize == 0), so we must
        // handle the out-of-bounds case before the cast.
        if rel < 0.0 {
            return None;
        }
        let col = (rel / w) as isize;
        let n = self.config.grid_size as isize;
        if col >= n {
            None
        } else {
            Some(col as usize)
        }
    }

    fn coord_to_row(&self, y: f64) -> Option<usize> {
        let rel = y - self.config.world_min.1;
        let h = self.cell_height();
        if h <= 0.0 {
            return None;
        }
        // Explicit negative-rel check (see coord_to_col).
        if rel < 0.0 {
            return None;
        }
        let row = (rel / h) as isize;
        let n = self.config.grid_size as isize;
        if row >= n {
            None
        } else {
            Some(row as usize)
        }
    }

    /// Return the (col, row) of the cell that contains point (x, y), if in bounds.
    pub fn cell_for_point(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let col = self.coord_to_col(x)?;
        let row = self.coord_to_row(y)?;
        Some((col, row))
    }

    /// Return the cell range [col_min..=col_max], [row_min..=row_max] for a bbox
    fn bbox_cell_range(&self, bbox: &BoundingBox) -> Option<(usize, usize, usize, usize)> {
        let n = self.config.grid_size;
        let w = self.cell_width();
        let h = self.cell_height();
        if w <= 0.0 || h <= 0.0 {
            return None;
        }

        let col_min_f = (bbox.min_x - self.config.world_min.0) / w;
        let col_max_f = (bbox.max_x - self.config.world_min.0) / w;
        let row_min_f = (bbox.min_y - self.config.world_min.1) / h;
        let row_max_f = (bbox.max_y - self.config.world_min.1) / h;

        let col_min = (col_min_f.floor() as isize).max(0) as usize;
        let col_max = ((col_max_f.ceil() as isize) - 1).max(0).min(n as isize - 1) as usize;
        let row_min = (row_min_f.floor() as isize).max(0) as usize;
        let row_max = ((row_max_f.ceil() as isize) - 1).max(0).min(n as isize - 1) as usize;

        // bbox entirely outside world
        if col_min >= n || row_min >= n {
            return None;
        }

        Some((col_min, col_max.min(n - 1), row_min, row_max.min(n - 1)))
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Insert an entry into the index
    pub fn insert(&mut self, entry: SpatialEntry) {
        let id = entry.id.clone();
        let bbox = entry.bbox.clone();
        self.entries.insert(id.clone(), entry);

        if let Some((col_min, col_max, row_min, row_max)) = self.bbox_cell_range(&bbox) {
            for col in col_min..=col_max {
                for row in row_min..=row_max {
                    self.grid[col][row].entries.push(id.clone());
                }
            }
        }
    }

    /// Remove an entry by id. Returns true if it was present.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(entry) = self.entries.remove(id) {
            // Clean up grid cells
            if let Some((col_min, col_max, row_min, row_max)) = self.bbox_cell_range(&entry.bbox) {
                for col in col_min..=col_max {
                    for row in row_min..=row_max {
                        self.grid[col][row].entries.retain(|eid| eid != id);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Return all entries whose bounding box intersects the query bbox.
    /// Result may contain false positives from grid cells but is checked exactly.
    pub fn query_bbox(&self, bbox: &BoundingBox) -> Vec<&SpatialEntry> {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut result = Vec::new();

        if let Some((col_min, col_max, row_min, row_max)) = self.bbox_cell_range(bbox) {
            for col in col_min..=col_max {
                for row in row_min..=row_max {
                    for eid in &self.grid[col][row].entries {
                        if seen.insert(eid.as_str()) {
                            if let Some(entry) = self.entries.get(eid.as_str()) {
                                if entry.bbox.intersects(bbox) {
                                    result.push(entry);
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Return all entries whose bounding box contains the given point
    pub fn query_point(&self, x: f64, y: f64) -> Vec<&SpatialEntry> {
        let point_bbox = BoundingBox::new(x, y, x, y);
        self.query_bbox(&point_bbox)
            .into_iter()
            .filter(|e| e.bbox.contains_point(x, y))
            .collect()
    }

    /// Return the n closest entries to (x, y), sorted by minimum bbox distance (ascending).
    pub fn nearest_n(&self, x: f64, y: f64, n: usize) -> Vec<(&SpatialEntry, f64)> {
        if n == 0 {
            return Vec::new();
        }
        let mut all: Vec<(&SpatialEntry, f64)> = self
            .entries
            .values()
            .map(|e| (e, e.bbox.distance_to_point(x, y)))
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all.into_iter().take(n).collect()
    }

    /// Number of entries in the index
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// World bounding box derived from configuration
    pub fn world_bbox(&self) -> BoundingBox {
        BoundingBox::new(
            self.config.world_min.0,
            self.config.world_min.1,
            self.config.world_max.0,
            self.config.world_max.1,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn world_config() -> SpatialIndexConfig {
        SpatialIndexConfig::new(10, (0.0, 0.0), (100.0, 100.0))
    }

    fn make_entry(id: &str, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> SpatialEntry {
        SpatialEntry::new(id, BoundingBox::new(min_x, min_y, max_x, max_y))
    }

    // ── BoundingBox ────────────────────────────────────────────────────────

    #[test]
    fn test_bbox_new_normalises() {
        let b = BoundingBox::new(10.0, 20.0, 5.0, 15.0);
        assert_eq!(b.min_x, 5.0);
        assert_eq!(b.max_x, 10.0);
        assert_eq!(b.min_y, 15.0);
        assert_eq!(b.max_y, 20.0);
    }

    #[test]
    fn test_bbox_intersects_overlapping() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        assert!(a.intersects(&b));
    }

    #[test]
    fn test_bbox_intersects_touching_edge() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(10.0, 0.0, 20.0, 10.0);
        assert!(a.intersects(&b));
    }

    #[test]
    fn test_bbox_disjoint() {
        let a = BoundingBox::new(0.0, 0.0, 5.0, 5.0);
        let b = BoundingBox::new(10.0, 10.0, 20.0, 20.0);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_bbox_contains_point_inside() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(b.contains_point(5.0, 5.0));
    }

    #[test]
    fn test_bbox_contains_point_on_boundary() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(b.contains_point(0.0, 0.0));
        assert!(b.contains_point(10.0, 10.0));
    }

    #[test]
    fn test_bbox_contains_point_outside() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(!b.contains_point(11.0, 5.0));
    }

    #[test]
    fn test_bbox_center() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 6.0);
        let (cx, cy) = b.center();
        assert!((cx - 5.0).abs() < 1e-10);
        assert!((cy - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_area() {
        let b = BoundingBox::new(0.0, 0.0, 4.0, 5.0);
        assert!((b.area() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_area_point_is_zero() {
        let b = BoundingBox::new(1.0, 1.0, 1.0, 1.0);
        assert!((b.area()).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_point_inside() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.distance_to_point(5.0, 5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_point_outside_right() {
        let b = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let d = b.distance_to_point(13.0, 5.0);
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_point_corner() {
        let b = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
        let d = b.distance_to_point(2.0, 2.0);
        // distance from corner (1,1) to (2,2) = sqrt(2)
        assert!((d - (2.0_f64).sqrt()).abs() < 1e-10);
    }

    // ── SpatialIndex construction ──────────────────────────────────────────

    #[test]
    fn test_spatial_index_new_empty() {
        let idx = SpatialIndex::new(world_config());
        assert_eq!(idx.count(), 0);
    }

    #[test]
    fn test_spatial_index_world_bbox() {
        let idx = SpatialIndex::new(world_config());
        let wb = idx.world_bbox();
        assert_eq!(wb.min_x, 0.0);
        assert_eq!(wb.max_x, 100.0);
    }

    #[test]
    fn test_spatial_index_insert_increments_count() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 10.0, 10.0, 20.0, 20.0));
        assert_eq!(idx.count(), 1);
    }

    #[test]
    fn test_spatial_index_insert_multiple() {
        let mut idx = SpatialIndex::new(world_config());
        for i in 0..5 {
            let f = i as f64 * 10.0;
            idx.insert(make_entry(&format!("e{i}"), f, f, f + 5.0, f + 5.0));
        }
        assert_eq!(idx.count(), 5);
    }

    // ── insert / remove ────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_entry() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 10.0, 10.0, 20.0, 20.0));
        assert!(idx.remove("e1"));
        assert_eq!(idx.count(), 0);
    }

    #[test]
    fn test_remove_missing_returns_false() {
        let mut idx = SpatialIndex::new(world_config());
        assert!(!idx.remove("nope"));
    }

    #[test]
    fn test_remove_then_query_empty() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 10.0, 10.0, 20.0, 20.0));
        idx.remove("e1");
        let results = idx.query_bbox(&BoundingBox::new(5.0, 5.0, 25.0, 25.0));
        assert!(results.is_empty());
    }

    // ── query_bbox ────────────────────────────────────────────────────────

    #[test]
    fn test_query_bbox_finds_overlapping() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 10.0, 10.0, 30.0, 30.0));
        let results = idx.query_bbox(&BoundingBox::new(20.0, 20.0, 40.0, 40.0));
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "e1");
    }

    #[test]
    fn test_query_bbox_no_overlap() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 10.0, 10.0, 20.0, 20.0));
        let results = idx.query_bbox(&BoundingBox::new(50.0, 50.0, 60.0, 60.0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_bbox_multiple_results() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 0.0, 0.0, 50.0, 50.0));
        idx.insert(make_entry("e2", 10.0, 10.0, 30.0, 30.0));
        idx.insert(make_entry("e3", 60.0, 60.0, 90.0, 90.0));
        let results = idx.query_bbox(&BoundingBox::new(5.0, 5.0, 40.0, 40.0));
        assert_eq!(results.len(), 2);
    }

    // ── query_point ───────────────────────────────────────────────────────

    #[test]
    fn test_query_point_inside() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 0.0, 0.0, 50.0, 50.0));
        let results = idx.query_point(25.0, 25.0);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "e1");
    }

    #[test]
    fn test_query_point_outside() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 0.0, 0.0, 10.0, 10.0));
        let results = idx.query_point(50.0, 50.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_point_outside_world() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 0.0, 0.0, 10.0, 10.0));
        let results = idx.query_point(200.0, 200.0);
        assert!(results.is_empty());
    }

    // ── nearest_n ─────────────────────────────────────────────────────────

    #[test]
    fn test_nearest_n_ordering() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("far", 80.0, 80.0, 90.0, 90.0));
        idx.insert(make_entry("near", 1.0, 1.0, 5.0, 5.0));
        idx.insert(make_entry("mid", 40.0, 40.0, 50.0, 50.0));
        let results = idx.nearest_n(0.0, 0.0, 3);
        assert_eq!(results.len(), 3);
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
    }

    #[test]
    fn test_nearest_n_zero_returns_empty() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 0.0, 0.0, 10.0, 10.0));
        let results = idx.nearest_n(0.0, 0.0, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_nearest_n_fewer_than_n() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 0.0, 0.0, 10.0, 10.0));
        let results = idx.nearest_n(0.0, 0.0, 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_nearest_n_closest_is_first() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("close", 2.0, 2.0, 4.0, 4.0));
        idx.insert(make_entry("far", 90.0, 90.0, 95.0, 95.0));
        let results = idx.nearest_n(3.0, 3.0, 2);
        assert_eq!(results[0].0.id, "close");
    }

    // ── cell_for_point ────────────────────────────────────────────────────

    #[test]
    fn test_cell_for_point_inside_world() {
        let idx = SpatialIndex::new(world_config());
        let cell = idx.cell_for_point(50.0, 50.0);
        assert!(cell.is_some());
    }

    #[test]
    fn test_cell_for_point_outside_world_returns_none() {
        let idx = SpatialIndex::new(world_config());
        assert!(idx.cell_for_point(200.0, 200.0).is_none());
        assert!(idx.cell_for_point(-1.0, 50.0).is_none());
    }

    #[test]
    fn test_cell_for_point_origin() {
        let idx = SpatialIndex::new(world_config());
        let cell = idx.cell_for_point(0.0, 0.0);
        assert_eq!(cell, Some((0, 0)));
    }

    // ── SpatialEntry ──────────────────────────────────────────────────────

    #[test]
    fn test_spatial_entry_with_data() {
        let e = SpatialEntry::new("id", BoundingBox::new(0.0, 0.0, 1.0, 1.0))
            .with_data("type", "feature");
        assert_eq!(e.data["type"], "feature");
    }

    #[test]
    fn test_bbox_intersects_full_containment() {
        let outer = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let inner = BoundingBox::new(10.0, 10.0, 20.0, 20.0);
        assert!(outer.intersects(&inner));
        assert!(inner.intersects(&outer));
    }

    #[test]
    fn test_bbox_no_intersect_x_axis() {
        let a = BoundingBox::new(0.0, 0.0, 5.0, 10.0);
        let b = BoundingBox::new(6.0, 0.0, 11.0, 10.0);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_bbox_no_intersect_y_axis() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 5.0);
        let b = BoundingBox::new(0.0, 6.0, 10.0, 11.0);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_spatial_index_default_config() {
        let cfg = SpatialIndexConfig::default();
        assert_eq!(cfg.grid_size, 16);
    }

    #[test]
    fn test_query_bbox_returns_entry_reference() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("e1", 10.0, 10.0, 20.0, 20.0));
        let results = idx.query_bbox(&BoundingBox::new(0.0, 0.0, 50.0, 50.0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "e1");
    }

    #[test]
    fn test_world_bbox_min_max() {
        let cfg = SpatialIndexConfig::new(10, (-50.0, -50.0), (50.0, 50.0));
        let idx = SpatialIndex::new(cfg);
        let wb = idx.world_bbox();
        assert_eq!(wb.min_x, -50.0);
        assert_eq!(wb.max_x, 50.0);
    }

    #[test]
    fn test_insert_and_count_many() {
        let mut idx = SpatialIndex::new(world_config());
        for i in 0..20 {
            let f = (i % 10) as f64 * 9.0;
            idx.insert(make_entry(&format!("e{i}"), f, f, f + 5.0, f + 5.0));
        }
        assert_eq!(idx.count(), 20);
    }

    #[test]
    fn test_nearest_n_returns_distance_ascending() {
        let mut idx = SpatialIndex::new(world_config());
        idx.insert(make_entry("a", 1.0, 1.0, 2.0, 2.0));
        idx.insert(make_entry("b", 50.0, 50.0, 55.0, 55.0));
        idx.insert(make_entry("c", 90.0, 90.0, 95.0, 95.0));
        let results = idx.nearest_n(0.0, 0.0, 3);
        for window in results.windows(2) {
            assert!(window[0].1 <= window[1].1);
        }
    }

    #[test]
    fn test_remove_nonexistent_entry_no_panic() {
        let mut idx = SpatialIndex::new(world_config());
        let result = idx.remove("ghost");
        assert!(!result);
    }

    #[test]
    fn test_distance_to_point_left_of_bbox() {
        let b = BoundingBox::new(10.0, 0.0, 20.0, 10.0);
        let d = b.distance_to_point(5.0, 5.0);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_area_line_is_zero() {
        // A line (zero width or height) has area 0
        let b = BoundingBox::new(0.0, 0.0, 10.0, 0.0);
        assert!((b.area()).abs() < 1e-10);
    }
}
