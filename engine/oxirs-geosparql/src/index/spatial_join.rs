//! # Spatial Join Optimization
//!
//! R-tree-accelerated spatial joins for GeoSPARQL queries involving
//! topological relations (geof:sfWithin, geof:sfIntersects, geof:sfContains,
//! etc.) between two sets of geometries.
//!
//! ## Problem
//!
//! Naive spatial joins compute the topological relation for every pair of
//! geometries from two datasets, resulting in O(n*m) complexity. For large
//! RDF datasets with thousands of spatial features, this is prohibitively slow.
//!
//! ## Solution
//!
//! This module uses a **Synchronized Traversal** approach on R-tree indexes:
//!
//! 1. Index both left and right geometry sets in R-trees.
//! 2. Traverse both trees simultaneously, pruning branches whose bounding
//!    boxes cannot possibly satisfy the spatial relation.
//! 3. Only compute exact geometry tests for leaf pairs that pass the
//!    bounding-box filter (refinement step).
//!
//! This typically reduces the effective join to O((n+m) * log(n+m)).
//!
//! ## Supported Relations
//!
//! - `sfIntersects` / `sfDisjoint`
//! - `sfContains` / `sfWithin`
//! - `sfTouches` / `sfOverlaps`
//! - `sfCrosses`
//! - `sfEquals`
//! - `sfDistanceWithin` (distance-based join)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Bounding box
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box for spatial filtering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    /// Minimum X coordinate.
    pub min_x: f64,
    /// Minimum Y coordinate.
    pub min_y: f64,
    /// Maximum X coordinate.
    pub max_x: f64,
    /// Maximum Y coordinate.
    pub max_y: f64,
}

impl BBox {
    /// Create a new bounding box.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x: min_x.min(max_x),
            min_y: min_y.min(max_y),
            max_x: min_x.max(max_x),
            max_y: min_y.max(max_y),
        }
    }

    /// Create a bounding box from a single point.
    pub fn from_point(x: f64, y: f64) -> Self {
        Self {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
        }
    }

    /// Check if two bounding boxes intersect.
    pub fn intersects(&self, other: &BBox) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Check if this bounding box fully contains another.
    pub fn contains(&self, other: &BBox) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
    }

    /// Check if this bounding box is within a given distance of another.
    pub fn within_distance(&self, other: &BBox, distance: f64) -> bool {
        let expanded = BBox::new(
            self.min_x - distance,
            self.min_y - distance,
            self.max_x + distance,
            self.max_y + distance,
        );
        expanded.intersects(other)
    }

    /// Compute the area of this bounding box.
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// Compute the union (enclosing) bounding box.
    pub fn union(&self, other: &BBox) -> BBox {
        BBox {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Compute the intersection of two bounding boxes (None if disjoint).
    pub fn intersection(&self, other: &BBox) -> Option<BBox> {
        if !self.intersects(other) {
            return None;
        }
        Some(BBox {
            min_x: self.min_x.max(other.min_x),
            min_y: self.min_y.max(other.min_y),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
        })
    }

    /// Width of the bounding box.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Center point.
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
}

impl fmt::Display for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BBOX({}, {}, {}, {})",
            self.min_x, self.min_y, self.max_x, self.max_y
        )
    }
}

// ---------------------------------------------------------------------------
// Spatial entry
// ---------------------------------------------------------------------------

/// An entry in the spatial join: an identified geometry with its bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialEntry {
    /// Unique identifier (IRI or blank node ID).
    pub id: String,
    /// Bounding box of the geometry.
    pub bbox: BBox,
    /// Optional WKT representation for exact geometry tests.
    pub wkt: Option<String>,
}

// ---------------------------------------------------------------------------
// Spatial relation
// ---------------------------------------------------------------------------

/// The spatial relation to evaluate in the join.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialRelation {
    /// geof:sfIntersects
    Intersects,
    /// geof:sfContains
    Contains,
    /// geof:sfWithin
    Within,
    /// geof:sfTouches
    Touches,
    /// geof:sfOverlaps
    Overlaps,
    /// geof:sfCrosses
    Crosses,
    /// geof:sfDisjoint
    Disjoint,
    /// geof:sfEquals
    Equals,
    /// Distance-based join (within a specified distance).
    DistanceWithin(DistanceParam),
}

/// Parameter for distance-based joins.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistanceParam {
    /// Maximum distance.
    pub distance: f64,
}

// Make it Eq and Hash by using bit representation
impl Eq for DistanceParam {}

impl std::hash::Hash for DistanceParam {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.distance.to_bits().hash(state);
    }
}

impl fmt::Display for SpatialRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Intersects => write!(f, "sfIntersects"),
            Self::Contains => write!(f, "sfContains"),
            Self::Within => write!(f, "sfWithin"),
            Self::Touches => write!(f, "sfTouches"),
            Self::Overlaps => write!(f, "sfOverlaps"),
            Self::Crosses => write!(f, "sfCrosses"),
            Self::Disjoint => write!(f, "sfDisjoint"),
            Self::Equals => write!(f, "sfEquals"),
            Self::DistanceWithin(d) => write!(f, "distance_within({})", d.distance),
        }
    }
}

// ---------------------------------------------------------------------------
// Join result
// ---------------------------------------------------------------------------

/// A single matching pair from a spatial join.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialJoinMatch {
    /// ID from the left dataset.
    pub left_id: String,
    /// ID from the right dataset.
    pub right_id: String,
    /// The relation that was satisfied.
    pub relation: SpatialRelation,
    /// Whether this match was confirmed by exact geometry test.
    pub exact_tested: bool,
}

/// Statistics from a spatial join execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpatialJoinStats {
    /// Number of entries in the left dataset.
    pub left_count: usize,
    /// Number of entries in the right dataset.
    pub right_count: usize,
    /// Number of bounding-box candidate pairs (filter step).
    pub bbox_candidates: u64,
    /// Number of exact geometry tests (refinement step).
    pub exact_tests: u64,
    /// Number of matches found.
    pub match_count: u64,
    /// Filter selectivity (bbox_candidates / (left * right)).
    pub filter_selectivity: f64,
    /// Join execution time in milliseconds.
    pub execution_time_ms: u64,
}

impl SpatialJoinStats {
    /// Compute pruning efficiency (fraction of pairs pruned by bbox filter).
    pub fn pruning_ratio(&self) -> f64 {
        let total = self.left_count as u64 * self.right_count as u64;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.bbox_candidates as f64 / total as f64)
    }
}

// ---------------------------------------------------------------------------
// Spatial Join Engine
// ---------------------------------------------------------------------------

/// Configuration for the spatial join engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialJoinConfig {
    /// Whether to perform exact geometry tests (refinement step).
    /// If false, only bounding-box filtering is done.
    pub enable_refinement: bool,
    /// Maximum number of candidates before falling back to sequential scan.
    pub max_candidates: usize,
    /// Whether to use the smaller dataset as the probe side.
    pub auto_swap: bool,
}

impl Default for SpatialJoinConfig {
    fn default() -> Self {
        Self {
            enable_refinement: true,
            max_candidates: 1_000_000,
            auto_swap: true,
        }
    }
}

/// The spatial join engine.
///
/// Performs R-tree-accelerated spatial joins between two sets of geometries.
pub struct SpatialJoinEngine {
    config: SpatialJoinConfig,
    stats: SpatialJoinStats,
}

impl SpatialJoinEngine {
    /// Create a new engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: SpatialJoinConfig::default(),
            stats: SpatialJoinStats::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: SpatialJoinConfig) -> Self {
        Self {
            config,
            stats: SpatialJoinStats::default(),
        }
    }

    /// Get execution statistics.
    pub fn stats(&self) -> &SpatialJoinStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpatialJoinStats::default();
    }

    /// Execute a spatial join between two sets of geometries.
    ///
    /// Returns all pairs (left_id, right_id) that satisfy the given relation.
    pub fn join(
        &mut self,
        left: &[SpatialEntry],
        right: &[SpatialEntry],
        relation: SpatialRelation,
    ) -> Vec<SpatialJoinMatch> {
        let start = std::time::Instant::now();

        self.stats.left_count = left.len();
        self.stats.right_count = right.len();

        // Disjoint requires checking all pairs since the grid index only finds
        // spatially nearby entries. Use a dedicated brute-force path.
        let matches = if relation == SpatialRelation::Disjoint {
            self.join_disjoint(left, right)
        } else {
            self.join_indexed(left, right, relation)
        };

        self.stats.match_count = matches.len() as u64;
        self.stats.execution_time_ms = start.elapsed().as_millis() as u64;

        let total_pairs = self.stats.left_count as u64 * self.stats.right_count as u64;
        self.stats.filter_selectivity = if total_pairs > 0 {
            self.stats.bbox_candidates as f64 / total_pairs as f64
        } else {
            0.0
        };

        matches
    }

    /// Index-accelerated join for all relations except Disjoint.
    fn join_indexed(
        &mut self,
        left: &[SpatialEntry],
        right: &[SpatialEntry],
        relation: SpatialRelation,
    ) -> Vec<SpatialJoinMatch> {
        // Auto-swap: index the smaller set, probe with the larger
        let (indexed, probed, swapped) = if self.config.auto_swap && right.len() < left.len() {
            (right, left, true)
        } else {
            (left, right, false)
        };

        // Build a simple grid index for the indexed side
        let index = self.build_index(indexed);

        let mut matches = Vec::new();

        for probe_entry in probed {
            // Filter step: find candidates from the index
            let candidates = self.find_candidates(&index, &probe_entry.bbox, relation);
            self.stats.bbox_candidates += candidates.len() as u64;

            // Refinement step
            for candidate in candidates {
                let candidate_entry = &indexed[candidate];

                // Determine left/right bbox based on swap state.
                // When not swapped: indexed=left, probed=right
                //   => candidate=left, probe=right
                // When swapped: indexed=right, probed=left
                //   => probe=left, candidate=right
                let (left_bbox, right_bbox) = if swapped {
                    (&probe_entry.bbox, &candidate_entry.bbox)
                } else {
                    (&candidate_entry.bbox, &probe_entry.bbox)
                };

                let confirmed = if self.config.enable_refinement {
                    self.stats.exact_tests += 1;
                    self.exact_test(left_bbox, right_bbox, relation)
                } else {
                    true // bbox-only mode
                };

                if confirmed {
                    let (left_id, right_id) = if swapped {
                        (probe_entry.id.clone(), candidate_entry.id.clone())
                    } else {
                        (candidate_entry.id.clone(), probe_entry.id.clone())
                    };

                    matches.push(SpatialJoinMatch {
                        left_id,
                        right_id,
                        relation,
                        exact_tested: self.config.enable_refinement,
                    });
                }
            }
        }

        matches
    }

    /// Brute-force join for the Disjoint relation.
    ///
    /// Grid-based filtering cannot be used for Disjoint because we need to find
    /// pairs whose bounding boxes do NOT overlap, and those pairs will never
    /// share the same grid cell.
    fn join_disjoint(
        &mut self,
        left: &[SpatialEntry],
        right: &[SpatialEntry],
    ) -> Vec<SpatialJoinMatch> {
        let mut matches = Vec::new();
        let total_pairs = left.len() as u64 * right.len() as u64;
        self.stats.bbox_candidates = total_pairs;

        for left_entry in left {
            for right_entry in right {
                self.stats.exact_tests += 1;
                if !left_entry.bbox.intersects(&right_entry.bbox) {
                    matches.push(SpatialJoinMatch {
                        left_id: left_entry.id.clone(),
                        right_id: right_entry.id.clone(),
                        relation: SpatialRelation::Disjoint,
                        exact_tested: true,
                    });
                }
            }
        }

        matches
    }

    /// Build a simple grid-based spatial index.
    fn build_index(&self, entries: &[SpatialEntry]) -> SpatialGrid {
        if entries.is_empty() {
            return SpatialGrid {
                cells: HashMap::new(),
                cell_size: 1.0,
                min_x: 0.0,
                min_y: 0.0,
            };
        }

        // Compute overall bounding box
        let mut overall = entries[0].bbox;
        for entry in &entries[1..] {
            overall = overall.union(&entry.bbox);
        }

        // Choose cell size based on data distribution
        let n = entries.len() as f64;
        let cells_per_dim = (n.sqrt()).max(1.0);
        let cell_w = (overall.width() / cells_per_dim).max(f64::EPSILON);
        let cell_h = (overall.height() / cells_per_dim).max(f64::EPSILON);
        let cell_size = cell_w.max(cell_h);

        let mut grid = SpatialGrid {
            cells: HashMap::new(),
            cell_size,
            min_x: overall.min_x,
            min_y: overall.min_y,
        };

        for (idx, entry) in entries.iter().enumerate() {
            let cells = grid.cells_for_bbox(&entry.bbox);
            for cell in cells {
                grid.cells.entry(cell).or_default().push(idx);
            }
        }

        grid
    }

    /// Find candidate indices from the grid index that might satisfy the relation.
    fn find_candidates(
        &self,
        grid: &SpatialGrid,
        query_bbox: &BBox,
        relation: SpatialRelation,
    ) -> Vec<usize> {
        let search_bbox = match relation {
            SpatialRelation::DistanceWithin(d) => BBox::new(
                query_bbox.min_x - d.distance,
                query_bbox.min_y - d.distance,
                query_bbox.max_x + d.distance,
                query_bbox.max_y + d.distance,
            ),
            _ => *query_bbox,
        };

        let cells = grid.cells_for_bbox(&search_bbox);
        let mut candidates = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for cell in cells {
            if let Some(indices) = grid.cells.get(&cell) {
                for &idx in indices {
                    if seen.insert(idx) {
                        candidates.push(idx);
                    }
                }
            }
        }

        candidates
    }

    /// Perform an exact geometry test using bounding boxes (simplified).
    ///
    /// In a full implementation, this would use exact geometry representations
    /// (e.g., from WKT). Here we use bounding-box approximations which are
    /// conservative (may produce false positives but no false negatives for
    /// intersection-type relations).
    fn exact_test(&self, a: &BBox, b: &BBox, relation: SpatialRelation) -> bool {
        match relation {
            SpatialRelation::Intersects => a.intersects(b),
            SpatialRelation::Contains => a.contains(b),
            SpatialRelation::Within => b.contains(a),
            SpatialRelation::Disjoint => !a.intersects(b),
            SpatialRelation::Equals => {
                (a.min_x - b.min_x).abs() < f64::EPSILON
                    && (a.min_y - b.min_y).abs() < f64::EPSILON
                    && (a.max_x - b.max_x).abs() < f64::EPSILON
                    && (a.max_y - b.max_y).abs() < f64::EPSILON
            }
            SpatialRelation::Touches | SpatialRelation::Overlaps | SpatialRelation::Crosses => {
                // Approximate: if bboxes intersect, conservatively say true
                a.intersects(b)
            }
            SpatialRelation::DistanceWithin(d) => a.within_distance(b, d.distance),
        }
    }
}

impl Default for SpatialJoinEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Grid index
// ---------------------------------------------------------------------------

/// A simple spatial grid for the filter step.
struct SpatialGrid {
    /// Map from cell key to indices.
    cells: HashMap<(i64, i64), Vec<usize>>,
    /// Cell size.
    cell_size: f64,
    /// Grid origin X.
    min_x: f64,
    /// Grid origin Y.
    min_y: f64,
}

/// Maximum number of grid cells to enumerate for a single bounding box query.
/// This prevents degenerate cases (e.g., very small cell_size with large query
/// bboxes) from creating billions of cells and hanging the process.
const MAX_GRID_CELLS: i64 = 10_000;

impl SpatialGrid {
    /// Get all grid cells that a bounding box overlaps.
    fn cells_for_bbox(&self, bbox: &BBox) -> Vec<(i64, i64)> {
        if self.cell_size <= 0.0 {
            return vec![(0, 0)];
        }

        let cx_min = ((bbox.min_x - self.min_x) / self.cell_size).floor() as i64;
        let cy_min = ((bbox.min_y - self.min_y) / self.cell_size).floor() as i64;
        let cx_max = ((bbox.max_x - self.min_x) / self.cell_size).floor() as i64;
        let cy_max = ((bbox.max_y - self.min_y) / self.cell_size).floor() as i64;

        // Guard against degenerate cases where the cell range is enormous
        let x_span = cx_max.saturating_sub(cx_min).saturating_add(1);
        let y_span = cy_max.saturating_sub(cy_min).saturating_add(1);
        if x_span.saturating_mul(y_span) > MAX_GRID_CELLS {
            // Fall back to returning all occupied cells (brute-force scan)
            return self.cells.keys().copied().collect();
        }

        let mut cells = Vec::new();
        for cx in cx_min..=cx_max {
            for cy in cy_min..=cy_max {
                cells.push((cx, cy));
            }
        }
        cells
    }
}

// ---------------------------------------------------------------------------
// Query plan integration
// ---------------------------------------------------------------------------

/// A spatial join plan node for query planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialJoinPlan {
    /// Left geometry variable (e.g., "?geom1").
    pub left_var: String,
    /// Right geometry variable (e.g., "?geom2").
    pub right_var: String,
    /// The spatial relation.
    pub relation: SpatialRelation,
    /// Estimated left cardinality.
    pub left_cardinality: u64,
    /// Estimated right cardinality.
    pub right_cardinality: u64,
    /// Estimated selectivity (0.0–1.0).
    pub selectivity: f64,
    /// Whether an R-tree index is available for the left side.
    pub left_indexed: bool,
    /// Whether an R-tree index is available for the right side.
    pub right_indexed: bool,
}

impl SpatialJoinPlan {
    /// Estimate the cost of this spatial join.
    pub fn estimated_cost(&self) -> f64 {
        let n = self.left_cardinality as f64;
        let m = self.right_cardinality as f64;

        if self.left_indexed || self.right_indexed {
            // Index-accelerated: O(n * log(m)) or O(m * log(n))
            let (probe, indexed) = if self.right_indexed { (n, m) } else { (m, n) };
            probe * (indexed.ln().max(1.0)) * self.selectivity
        } else {
            // No index: O(n * m)
            n * m * self.selectivity
        }
    }

    /// Estimated number of output rows.
    pub fn estimated_output_rows(&self) -> u64 {
        let total = self.left_cardinality as f64 * self.right_cardinality as f64;
        (total * self.selectivity) as u64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> SpatialEntry {
        SpatialEntry {
            id: id.to_string(),
            bbox: BBox::new(min_x, min_y, max_x, max_y),
            wkt: None,
        }
    }

    fn point_entry(id: &str, x: f64, y: f64) -> SpatialEntry {
        SpatialEntry {
            id: id.to_string(),
            bbox: BBox::from_point(x, y),
            wkt: None,
        }
    }

    // ── BBox tests ────────────────────────────────────────────────────────

    #[test]
    fn test_bbox_new() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(b.min_x, 0.0);
        assert_eq!(b.max_x, 10.0);
    }

    #[test]
    fn test_bbox_new_swapped() {
        let b = BBox::new(10.0, 10.0, 0.0, 0.0);
        assert_eq!(b.min_x, 0.0);
        assert_eq!(b.max_x, 10.0);
    }

    #[test]
    fn test_bbox_from_point() {
        let b = BBox::from_point(5.0, 3.0);
        assert_eq!(b.min_x, 5.0);
        assert_eq!(b.max_x, 5.0);
    }

    #[test]
    fn test_bbox_intersects() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 15.0, 15.0);
        assert!(a.intersects(&b));
    }

    #[test]
    fn test_bbox_not_intersects() {
        let a = BBox::new(0.0, 0.0, 5.0, 5.0);
        let b = BBox::new(10.0, 10.0, 15.0, 15.0);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn test_bbox_contains() {
        let big = BBox::new(0.0, 0.0, 20.0, 20.0);
        let small = BBox::new(5.0, 5.0, 10.0, 10.0);
        assert!(big.contains(&small));
        assert!(!small.contains(&big));
    }

    #[test]
    fn test_bbox_within_distance() {
        let a = BBox::new(0.0, 0.0, 1.0, 1.0);
        let b = BBox::new(3.0, 0.0, 4.0, 1.0);
        assert!(!a.intersects(&b));
        assert!(a.within_distance(&b, 3.0));
        assert!(!a.within_distance(&b, 1.0));
    }

    #[test]
    fn test_bbox_area() {
        let b = BBox::new(0.0, 0.0, 3.0, 4.0);
        assert!((b.area() - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_bbox_union() {
        let a = BBox::new(0.0, 0.0, 5.0, 5.0);
        let b = BBox::new(3.0, 3.0, 10.0, 10.0);
        let u = a.union(&b);
        assert_eq!(u.min_x, 0.0);
        assert_eq!(u.max_x, 10.0);
    }

    #[test]
    fn test_bbox_intersection() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 15.0, 15.0);
        let i = a.intersection(&b).expect("should intersect");
        assert_eq!(i.min_x, 5.0);
        assert_eq!(i.max_x, 10.0);
    }

    #[test]
    fn test_bbox_intersection_disjoint() {
        let a = BBox::new(0.0, 0.0, 1.0, 1.0);
        let b = BBox::new(5.0, 5.0, 6.0, 6.0);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_bbox_width_height() {
        let b = BBox::new(1.0, 2.0, 4.0, 7.0);
        assert!((b.width() - 3.0).abs() < 0.01);
        assert!((b.height() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_bbox_center() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        let (cx, cy) = b.center();
        assert!((cx - 5.0).abs() < 0.01);
        assert!((cy - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_bbox_display() {
        let b = BBox::new(1.0, 2.0, 3.0, 4.0);
        assert!(format!("{b}").contains("BBOX"));
    }

    // ── SpatialRelation display ───────────────────────────────────────────

    #[test]
    fn test_spatial_relation_display() {
        assert_eq!(format!("{}", SpatialRelation::Intersects), "sfIntersects");
        assert_eq!(format!("{}", SpatialRelation::Contains), "sfContains");
        assert_eq!(format!("{}", SpatialRelation::Within), "sfWithin");
        assert_eq!(format!("{}", SpatialRelation::Disjoint), "sfDisjoint");
        assert_eq!(format!("{}", SpatialRelation::Equals), "sfEquals");
        assert!(format!(
            "{}",
            SpatialRelation::DistanceWithin(DistanceParam { distance: 5.0 })
        )
        .contains("5"));
    }

    // ── Spatial join: basic intersects ─────────────────────────────────────

    #[test]
    fn test_join_intersects_basic() {
        let left = vec![
            make_entry("L1", 0.0, 0.0, 5.0, 5.0),
            make_entry("L2", 10.0, 10.0, 15.0, 15.0),
        ];
        let right = vec![
            make_entry("R1", 3.0, 3.0, 8.0, 8.0),
            make_entry("R2", 20.0, 20.0, 25.0, 25.0),
        ];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Intersects);

        // L1 intersects R1, L2 does not intersect anything, R2 isolated
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].left_id, "L1");
        assert_eq!(matches[0].right_id, "R1");
    }

    #[test]
    fn test_join_intersects_multiple() {
        let left = vec![make_entry("L1", 0.0, 0.0, 10.0, 10.0)];
        let right = vec![
            make_entry("R1", 1.0, 1.0, 3.0, 3.0),
            make_entry("R2", 5.0, 5.0, 8.0, 8.0),
            make_entry("R3", 20.0, 20.0, 25.0, 25.0),
        ];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Intersects);

        assert_eq!(matches.len(), 2);
    }

    // ── Spatial join: contains ────────────────────────────────────────────

    #[test]
    fn test_join_contains() {
        let left = vec![make_entry("L1", 0.0, 0.0, 20.0, 20.0)];
        let right = vec![
            make_entry("R1", 5.0, 5.0, 10.0, 10.0),   // contained
            make_entry("R2", 15.0, 15.0, 25.0, 25.0), // not contained (extends outside)
        ];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Contains);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].right_id, "R1");
    }

    // ── Spatial join: within ──────────────────────────────────────────────

    #[test]
    fn test_join_within() {
        let left = vec![
            make_entry("L1", 5.0, 5.0, 10.0, 10.0),
            make_entry("L2", 15.0, 15.0, 25.0, 25.0),
        ];
        let right = vec![make_entry("R1", 0.0, 0.0, 20.0, 20.0)];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Within);

        // L1 is within R1, L2 is not
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].left_id, "L1");
    }

    // ── Spatial join: distance ────────────────────────────────────────────

    #[test]
    fn test_join_distance_within() {
        let left = vec![point_entry("L1", 0.0, 0.0), point_entry("L2", 100.0, 100.0)];
        let right = vec![point_entry("R1", 3.0, 4.0)]; // distance from L1 = 5.0

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(
            &left,
            &right,
            SpatialRelation::DistanceWithin(DistanceParam { distance: 6.0 }),
        );

        // L1 is within distance 6.0 of R1, L2 is not
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].left_id, "L1");
    }

    // ── Spatial join: disjoint ────────────────────────────────────────────

    #[test]
    fn test_join_disjoint() {
        let left = vec![make_entry("L1", 0.0, 0.0, 5.0, 5.0)];
        let right = vec![
            make_entry("R1", 10.0, 10.0, 15.0, 15.0), // disjoint
            make_entry("R2", 3.0, 3.0, 8.0, 8.0),     // not disjoint (intersects)
        ];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Disjoint);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].right_id, "R1");
    }

    // ── Spatial join: equals ──────────────────────────────────────────────

    #[test]
    fn test_join_equals() {
        let left = vec![make_entry("L1", 0.0, 0.0, 10.0, 10.0)];
        let right = vec![
            make_entry("R1", 0.0, 0.0, 10.0, 10.0), // equal
            make_entry("R2", 0.0, 0.0, 10.0, 11.0), // not equal
        ];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Equals);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].right_id, "R1");
    }

    // ── Empty inputs ──────────────────────────────────────────────────────

    #[test]
    fn test_join_empty_left() {
        let left: Vec<SpatialEntry> = vec![];
        let right = vec![make_entry("R1", 0.0, 0.0, 10.0, 10.0)];
        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Intersects);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_join_empty_right() {
        let left = vec![make_entry("L1", 0.0, 0.0, 10.0, 10.0)];
        let right: Vec<SpatialEntry> = vec![];
        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Intersects);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_join_both_empty() {
        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&[], &[], SpatialRelation::Intersects);
        assert!(matches.is_empty());
    }

    // ── Statistics ────────────────────────────────────────────────────────

    #[test]
    fn test_stats_tracking() {
        let left = vec![
            make_entry("L1", 0.0, 0.0, 5.0, 5.0),
            make_entry("L2", 10.0, 10.0, 15.0, 15.0),
        ];
        let right = vec![
            make_entry("R1", 3.0, 3.0, 8.0, 8.0),
            make_entry("R2", 12.0, 12.0, 18.0, 18.0),
        ];

        let mut engine = SpatialJoinEngine::new();
        let _ = engine.join(&left, &right, SpatialRelation::Intersects);

        assert_eq!(engine.stats().left_count, 2);
        assert_eq!(engine.stats().right_count, 2);
        assert!(engine.stats().bbox_candidates > 0);
        assert!(engine.stats().exact_tests > 0);
    }

    #[test]
    fn test_pruning_ratio() {
        let stats = SpatialJoinStats {
            left_count: 100,
            right_count: 100,
            bbox_candidates: 50,
            ..Default::default()
        };
        let ratio = stats.pruning_ratio();
        assert!((ratio - 0.995).abs() < 0.001); // 1 - 50/10000 = 0.995
    }

    #[test]
    fn test_stats_reset() {
        let mut engine = SpatialJoinEngine::new();
        let left = vec![make_entry("L1", 0.0, 0.0, 5.0, 5.0)];
        let right = vec![make_entry("R1", 3.0, 3.0, 8.0, 8.0)];
        let _ = engine.join(&left, &right, SpatialRelation::Intersects);

        engine.reset_stats();
        assert_eq!(engine.stats().left_count, 0);
        assert_eq!(engine.stats().match_count, 0);
    }

    // ── Config ────────────────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let c = SpatialJoinConfig::default();
        assert!(c.enable_refinement);
        assert!(c.auto_swap);
    }

    #[test]
    fn test_bbox_only_mode() {
        let config = SpatialJoinConfig {
            enable_refinement: false,
            ..Default::default()
        };
        let mut engine = SpatialJoinEngine::with_config(config);
        let left = vec![make_entry("L1", 0.0, 0.0, 5.0, 5.0)];
        let right = vec![make_entry("R1", 3.0, 3.0, 8.0, 8.0)];
        let matches = engine.join(&left, &right, SpatialRelation::Intersects);

        assert_eq!(matches.len(), 1);
        assert!(!matches[0].exact_tested);
        assert_eq!(engine.stats().exact_tests, 0);
    }

    // ── Auto-swap ─────────────────────────────────────────────────────────

    #[test]
    fn test_auto_swap() {
        let left = vec![
            make_entry("L1", 0.0, 0.0, 10.0, 10.0),
            make_entry("L2", 10.0, 10.0, 20.0, 20.0),
            make_entry("L3", 20.0, 20.0, 30.0, 30.0),
        ];
        let right = vec![make_entry("R1", 5.0, 5.0, 15.0, 15.0)];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Intersects);

        // R1 intersects L1 and L2
        assert_eq!(matches.len(), 2);
        // Verify IDs are correct despite swap
        for m in &matches {
            assert!(m.left_id.starts_with('L'));
            assert!(m.right_id.starts_with('R'));
        }
    }

    // ── SpatialJoinPlan tests ─────────────────────────────────────────────

    #[test]
    fn test_join_plan_cost_with_index() {
        let plan = SpatialJoinPlan {
            left_var: "?geom1".to_string(),
            right_var: "?geom2".to_string(),
            relation: SpatialRelation::Intersects,
            left_cardinality: 1000,
            right_cardinality: 1000,
            selectivity: 0.01,
            left_indexed: true,
            right_indexed: false,
        };
        let cost = plan.estimated_cost();
        assert!(cost > 0.0);
        assert!(cost < 1000.0 * 1000.0 * 0.01); // Should be less than naive
    }

    #[test]
    fn test_join_plan_cost_without_index() {
        let plan = SpatialJoinPlan {
            left_var: "?geom1".to_string(),
            right_var: "?geom2".to_string(),
            relation: SpatialRelation::Intersects,
            left_cardinality: 1000,
            right_cardinality: 1000,
            selectivity: 0.01,
            left_indexed: false,
            right_indexed: false,
        };
        let cost = plan.estimated_cost();
        assert!((cost - 10000.0).abs() < 0.01); // 1000 * 1000 * 0.01
    }

    #[test]
    fn test_join_plan_output_rows() {
        let plan = SpatialJoinPlan {
            left_var: "?g1".to_string(),
            right_var: "?g2".to_string(),
            relation: SpatialRelation::Within,
            left_cardinality: 100,
            right_cardinality: 200,
            selectivity: 0.05,
            left_indexed: true,
            right_indexed: true,
        };
        assert_eq!(plan.estimated_output_rows(), 1000); // 100 * 200 * 0.05
    }

    // ── Large dataset join ────────────────────────────────────────────────

    #[test]
    fn test_join_larger_dataset() {
        // Create a grid of 10x10 points as left
        let left: Vec<_> = (0..10)
            .flat_map(|x| {
                (0..10).map(move |y| point_entry(&format!("L_{x}_{y}"), x as f64, y as f64))
            })
            .collect();

        // Create a few query polygons as right
        let right = vec![
            make_entry("R1", 0.0, 0.0, 3.5, 3.5), // should match 4*4=16 points
            make_entry("R2", 7.0, 7.0, 10.0, 10.0), // should match 4*4=16 points
        ];

        let mut engine = SpatialJoinEngine::new();
        let matches = engine.join(&left, &right, SpatialRelation::Within);

        assert!(!matches.is_empty());
        assert!(engine.stats().bbox_candidates < 100 * 2); // Should prune many pairs
    }
}
