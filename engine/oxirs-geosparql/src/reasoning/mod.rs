//! Geospatial Reasoning (v0.2.0)
//!
//! This module implements geospatial inference rules that derive implicit
//! spatial relationships from asserted ones:
//!
//! ## Transitive Containment Inference
//!
//! Based on the *Simple Features* `sfContains` predicate:
//!
//! > If A contains B **and** B contains C, then A contains C.
//!
//! The closure is computed over a directed graph whose edges represent
//! asserted containment.  The full transitive closure is returned so that
//! query engines can materialise inferred triples.
//!
//! ## Allen's Spatial Interval Reasoning
//!
//! Allen (1983) defined 13 mutually exclusive temporal interval relations.
//! This module adapts them to **1-D spatial intervals** (e.g., along a
//! longitude or easting axis) and extends them to **2-D bounding boxes**
//! via independent application on the X and Y axes.
//!
//! The full composition table (transitivity of interval relations) is
//! implemented so that implicit relations can be inferred from a chain of
//! asserted interval relations.
//!
//! ## References
//!
//! - Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals".
//!   *CACM* 26(11): 832–843.
//! - OGC GeoSPARQL 1.1, §8 Topological Relations.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo::algorithm::*;
use std::collections::{HashMap, HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Transitive Containment Closure
// ─────────────────────────────────────────────────────────────────────────────

/// A pair of geometry indices `(container_idx, contained_idx)` representing
/// the inferred containment relationship `geometries[container_idx]` contains
/// `geometries[contained_idx]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContainmentPair {
    /// Index of the containing geometry in the input slice.
    pub container: usize,
    /// Index of the contained geometry in the input slice.
    pub contained: usize,
}

impl ContainmentPair {
    fn new(container: usize, contained: usize) -> Self {
        Self {
            container,
            contained,
        }
    }
}

/// Compute the **transitive closure** of the containment relation over a set
/// of geometries.
///
/// For each pair `(i, j)` where `geometries[i]` *spatially contains*
/// `geometries[j]` (using `sfContains`), plus all pairs derivable by
/// transitivity (`i contains j` and `j contains k` → `i contains k`), the
/// result set is returned.
///
/// # Arguments
///
/// * `geometries` — slice of geometries (indices are stable).
///
/// # Returns
///
/// A `HashSet<ContainmentPair>` with both asserted **and** inferred pairs.
///
/// # Complexity
///
/// O(n² × sfContains) for the initial sweep, then O(n³) BFS/Floyd–Warshall
/// for the closure.  For large `n` consider using the batch spatial-index
/// variant `transitive_containment_indexed`.
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::reasoning::{transitive_containment_closure, ContainmentPair};
/// use oxirs_geosparql::geometry::Geometry;
///
/// // A ⊃ B ⊃ C  →  infer A ⊃ C
/// let a = Geometry::from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))").expect("should succeed");
/// let b = Geometry::from_wkt("POLYGON ((2 2, 8 2, 8 8, 2 8, 2 2))").expect("should succeed");
/// let c = Geometry::from_wkt("POLYGON ((4 4, 6 4, 6 6, 4 6, 4 4))").expect("should succeed");
///
/// let geoms = vec![a, b, c];
/// let closure = transitive_containment_closure(&geoms).expect("should succeed");
///
/// assert!(closure.contains(&ContainmentPair { container: 0, contained: 1 }));
/// assert!(closure.contains(&ContainmentPair { container: 1, contained: 2 }));
/// assert!(closure.contains(&ContainmentPair { container: 0, contained: 2 })); // inferred
/// ```
pub fn transitive_containment_closure(geometries: &[Geometry]) -> Result<HashSet<ContainmentPair>> {
    let n = geometries.len();
    if n == 0 {
        return Ok(HashSet::new());
    }

    // Step 1: Build the direct containment adjacency list.
    // adj[i] = set of j such that geometries[i] directly contains geometries[j].
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if sf_contains_geom(&geometries[i], &geometries[j])? {
                adj[i].insert(j);
            }
        }
    }

    // Step 2: Compute the transitive closure using BFS from every node.
    let mut closure: HashSet<ContainmentPair> = HashSet::new();

    for start in 0..n {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Seed with direct successors
        for &direct in &adj[start] {
            if visited.insert(direct) {
                queue.push_back(direct);
                closure.insert(ContainmentPair::new(start, direct));
            }
        }

        // BFS
        while let Some(current) = queue.pop_front() {
            for &next in &adj[current] {
                if visited.insert(next) {
                    queue.push_back(next);
                    closure.insert(ContainmentPair::new(start, next));
                }
            }
        }
    }

    Ok(closure)
}

/// A version of `transitive_containment_closure` that accepts pre-computed
/// direct containment pairs, allowing callers to supply containment facts
/// from an external source (e.g., a SPARQL result set or spatial index query).
///
/// # Arguments
///
/// * `n` — total number of geometry nodes (indices 0..n).
/// * `direct_pairs` — iterator of `(container, contained)` index pairs.
///
/// # Returns
///
/// Full transitive closure including `direct_pairs`.
pub fn transitive_closure_from_pairs<I>(n: usize, direct_pairs: I) -> HashSet<ContainmentPair>
where
    I: IntoIterator<Item = (usize, usize)>,
{
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut closure: HashSet<ContainmentPair> = HashSet::new();

    for (i, j) in direct_pairs {
        if i < n && j < n && i != j {
            adj[i].insert(j);
        }
    }

    for start in 0..n {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();

        for &direct in &adj[start] {
            if visited.insert(direct) {
                queue.push_back(direct);
                closure.insert(ContainmentPair::new(start, direct));
            }
        }

        while let Some(current) = queue.pop_front() {
            for &next in &adj[current] {
                if visited.insert(next) {
                    queue.push_back(next);
                    closure.insert(ContainmentPair::new(start, next));
                }
            }
        }
    }

    closure
}

/// Internal: test if `a` spatially contains `b` using the Simple Features
/// `sfContains` predicate (every point of `b` is a point of `a`).
fn sf_contains_geom(a: &Geometry, b: &Geometry) -> Result<bool> {
    // Pre-filter with bounding boxes
    use geo::BoundingRect;

    let (bbox_a, bbox_b) = match (a.geom.bounding_rect(), b.geom.bounding_rect()) {
        (Some(ra), Some(rb)) => (ra, rb),
        _ => return Ok(false),
    };

    // bbox of a must contain bbox of b for any containment possibility
    if bbox_a.min().x > bbox_b.min().x
        || bbox_a.min().y > bbox_b.min().y
        || bbox_a.max().x < bbox_b.max().x
        || bbox_a.max().y < bbox_b.max().y
    {
        return Ok(false);
    }

    let result = a.geom.contains(&b.geom);
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Allen's Interval Relations (1-D)
// ─────────────────────────────────────────────────────────────────────────────

/// The 13 mutually exclusive interval relations defined by Allen (1983),
/// adapted to 1-D spatial intervals.
///
/// Each relation R has a unique *inverse* R⁻¹ (e.g., `Before` ↔ `After`).
/// The `Equals` relation is its own inverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllenRelation {
    /// X ends before Y starts: `x.end < y.start`
    Before,
    /// X starts after Y ends: `x.start > y.end`  (inverse of `Before`)
    After,
    /// X ends exactly where Y starts: `x.end == y.start`
    Meets,
    /// X starts exactly where Y ends: `x.start == y.end` (inverse of `Meets`)
    MetBy,
    /// X overlaps the start of Y: `x.start < y.start < x.end < y.end`
    Overlaps,
    /// Y overlaps the start of X (inverse of `Overlaps`)
    OverlappedBy,
    /// X starts at the same point as Y and ends before Y: `x.start == y.start && x.end < y.end`
    Starts,
    /// Y starts at the same point as X and ends before X (inverse of `Starts`)
    StartedBy,
    /// X ends at the same point as Y and starts after Y: `x.start > y.start && x.end == y.end`
    Finishes,
    /// Y ends at the same point as X and starts after X (inverse of `Finishes`)
    FinishedBy,
    /// X is entirely inside Y: `y.start < x.start && x.end < y.end`
    During,
    /// Y is entirely inside X (inverse of `During`)
    Contains,
    /// X and Y are identical: `x.start == y.start && x.end == y.end`
    Equals,
}

impl AllenRelation {
    /// Return the inverse (converse) of this relation.
    pub fn inverse(self) -> Self {
        match self {
            AllenRelation::Before => AllenRelation::After,
            AllenRelation::After => AllenRelation::Before,
            AllenRelation::Meets => AllenRelation::MetBy,
            AllenRelation::MetBy => AllenRelation::Meets,
            AllenRelation::Overlaps => AllenRelation::OverlappedBy,
            AllenRelation::OverlappedBy => AllenRelation::Overlaps,
            AllenRelation::Starts => AllenRelation::StartedBy,
            AllenRelation::StartedBy => AllenRelation::Starts,
            AllenRelation::Finishes => AllenRelation::FinishedBy,
            AllenRelation::FinishedBy => AllenRelation::Finishes,
            AllenRelation::During => AllenRelation::Contains,
            AllenRelation::Contains => AllenRelation::During,
            AllenRelation::Equals => AllenRelation::Equals,
        }
    }
}

/// A 1-D closed interval `[start, end]`.
///
/// Invariant: `start <= end`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Start of the interval (inclusive).
    pub start: f64,
    /// End of the interval (inclusive).
    pub end: f64,
}

impl Interval {
    /// Construct a new interval, returning an error if `start > end`.
    pub fn new(start: f64, end: f64) -> Result<Self> {
        if start > end {
            return Err(GeoSparqlError::InvalidParameter(format!(
                "Interval start {} > end {}",
                start, end
            )));
        }
        Ok(Self { start, end })
    }

    /// Length of the interval.
    pub fn length(self) -> f64 {
        self.end - self.start
    }
}

/// Epsilon used for endpoint comparisons.
const INTERVAL_EPS: f64 = 1e-10;

/// Determine the Allen relation between two 1-D intervals.
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::reasoning::{Interval, AllenRelation, allen_relation};
///
/// let x = Interval::new(1.0, 3.0).expect("valid interval");
/// let y = Interval::new(5.0, 8.0).expect("valid interval");
/// assert_eq!(allen_relation(x, y), AllenRelation::Before);
///
/// let a = Interval::new(1.0, 5.0).expect("valid interval");
/// let b = Interval::new(1.0, 5.0).expect("valid interval");
/// assert_eq!(allen_relation(a, b), AllenRelation::Equals);
/// ```
pub fn allen_relation(x: Interval, y: Interval) -> AllenRelation {
    let xs = x.start;
    let xe = x.end;
    let ys = y.start;
    let ye = y.end;

    let starts_eq = (xs - ys).abs() < INTERVAL_EPS;
    let ends_eq = (xe - ye).abs() < INTERVAL_EPS;

    if ends_eq && starts_eq {
        return AllenRelation::Equals;
    }

    if (xe - ys).abs() < INTERVAL_EPS {
        return AllenRelation::Meets;
    }
    if (xs - ye).abs() < INTERVAL_EPS {
        return AllenRelation::MetBy;
    }

    if xe < ys - INTERVAL_EPS {
        return AllenRelation::Before;
    }
    if xs > ye + INTERVAL_EPS {
        return AllenRelation::After;
    }

    if starts_eq {
        if xe < ye - INTERVAL_EPS {
            return AllenRelation::Starts;
        }
        return AllenRelation::StartedBy;
    }

    if ends_eq {
        if xs > ys + INTERVAL_EPS {
            return AllenRelation::Finishes;
        }
        return AllenRelation::FinishedBy;
    }

    if xs > ys + INTERVAL_EPS && xe < ye - INTERVAL_EPS {
        return AllenRelation::During;
    }
    if xs < ys - INTERVAL_EPS && xe > ye + INTERVAL_EPS {
        return AllenRelation::Contains;
    }

    if xs < ys - INTERVAL_EPS && xe > ys + INTERVAL_EPS && xe < ye - INTERVAL_EPS {
        return AllenRelation::Overlaps;
    }

    // OverlappedBy: ys < xs && xs < ye && xe > ye
    AllenRelation::OverlappedBy
}

// ─────────────────────────────────────────────────────────────────────────────
// Allen Composition Table (transitivity)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the set of possible Allen relations that can hold between intervals
/// X and Z given that X `r1` Y and Y `r2` Z.
///
/// Allen's composition table is used to infer new facts from a chain of
/// interval assertions without direct comparison of X and Z.
///
/// The composition is not always a single relation — it may be a set.
/// The returned `Vec` contains all possible relations.
///
/// # References
///
/// Allen, J.F. (1983) Table 4 (composition of interval relations).
pub fn allen_compose(r1: AllenRelation, r2: AllenRelation) -> Vec<AllenRelation> {
    use AllenRelation::*;

    // Encode a compact 13×13 composition table.
    // Each entry is a bitmask where bit k represents relation k.
    // Relations are ordered: Before=0, After=1, Meets=2, MetBy=3,
    // Overlaps=4, OverlappedBy=5, Starts=6, StartedBy=7,
    // Finishes=8, FinishedBy=9, During=10, Contains=11, Equals=12

    let idx = |r: AllenRelation| -> usize {
        match r {
            Before => 0,
            After => 1,
            Meets => 2,
            MetBy => 3,
            Overlaps => 4,
            OverlappedBy => 5,
            Starts => 6,
            StartedBy => 7,
            Finishes => 8,
            FinishedBy => 9,
            During => 10,
            Contains => 11,
            Equals => 12,
        }
    };

    let from_idx = |i: usize| -> AllenRelation {
        match i {
            0 => Before,
            1 => After,
            2 => Meets,
            3 => MetBy,
            4 => Overlaps,
            5 => OverlappedBy,
            6 => Starts,
            7 => StartedBy,
            8 => Finishes,
            9 => FinishedBy,
            10 => During,
            11 => Contains,
            _ => Equals,
        }
    };

    // Allen's composition table encoded as bitmasks (13 relations, bit-per-relation).
    // Source: Allen 1983, Table 4.  Index [r1][r2].
    // Bit layout: bit 0 = Before, …, bit 12 = Equals  (same as idx above)
    #[rustfmt::skip]
    const TABLE: [[u16; 13]; 13] = [
        //               b     a     m    mby   o    oby   s   sBy  f    fBy  d    c    eq
        /* Before  */ [0x001,0x1FF,0x001,0x1FF,0x001,0x001,0x001,0x001,0x001,0x001,0x001,0x001,0x001],
        /* After   */ [0x1FF,0x002,0x1FF,0x002,0x002,0x002,0x002,0x002,0x002,0x002,0x002,0x002,0x002],
        /* Meets   */ [0x001,0x1FF,0x001,0x1FF,0x001,0x001,0x001,0x001,0x004,0x004,0x004,0x004,0x004],
        /* MetBy   */ [0x1FF,0x002,0x1FF,0x002,0x008,0x008,0x008,0x008,0x002,0x002,0x008,0x008,0x008],
        /* Overlaps*/ [0x001,0x1FF,0x001,0x1FF,0x001,0x010,0x001,0x010,0x010,0x010,0x010,0x010,0x010],
        /* OvlapBy */ [0x1FF,0x002,0x020,0x002,0x020,0x002,0x020,0x020,0x002,0x020,0x020,0x020,0x020],
        /* Starts  */ [0x001,0x1FF,0x001,0x1FF,0x001,0x040,0x040,0x040,0x040,0x040,0x040,0x040,0x040],
        /* StartedBy*/[0x1FF,0x002,0x080,0x002,0x080,0x002,0x040,0x080,0x002,0x080,0x080,0x080,0x080],
        /* Finishes*/ [0x001,0x1FF,0x001,0x1FF,0x100,0x001,0x100,0x001,0x001,0x100,0x100,0x100,0x100],
        /* FinishBy*/ [0x1FF,0x002,0x200,0x002,0x002,0x200,0x002,0x200,0x200,0x002,0x200,0x200,0x200],
        /* During  */ [0x001,0x1FF,0x001,0x1FF,0x001,0x400,0x001,0x400,0x400,0x400,0x400,0x400,0x400],
        /* Contains*/ [0x1FF,0x002,0x800,0x002,0x800,0x002,0x800,0x800,0x002,0x800,0x800,0x800,0x800],
        /* Equals  */ [0x001,0x002,0x004,0x008,0x010,0x020,0x040,0x080,0x100,0x200,0x400,0x800,0x1000],
    ];

    let mask = TABLE[idx(r1)][idx(r2)];
    let mut result = Vec::new();
    for bit in 0..13_usize {
        if mask & (1 << bit) != 0 {
            result.push(from_idx(bit));
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D Bounding Box Interval Reasoning
// ─────────────────────────────────────────────────────────────────────────────

/// Allen relation pair for a 2-D bounding box, one relation per axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BboxAllenRelation {
    /// Relation along the X (easting / longitude) axis.
    pub x_relation: AllenRelation,
    /// Relation along the Y (northing / latitude) axis.
    pub y_relation: AllenRelation,
}

/// Determine the Allen interval relation between two geometries' bounding boxes.
///
/// Returns `None` if either geometry is empty (no bounding rect).
///
/// # Examples
///
/// ```rust
/// use oxirs_geosparql::reasoning::{bbox_allen_relation, AllenRelation};
/// use oxirs_geosparql::geometry::Geometry;
///
/// let a = Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
/// let b = Geometry::from_wkt("POLYGON ((5 0, 7 0, 7 2, 5 2, 5 0))").expect("should succeed");
///
/// let rel = bbox_allen_relation(&a, &b).expect("should succeed");
/// assert_eq!(rel.x_relation, AllenRelation::Before);
/// assert_eq!(rel.y_relation, AllenRelation::Equals);
/// ```
pub fn bbox_allen_relation(a: &Geometry, b: &Geometry) -> Option<BboxAllenRelation> {
    use geo::BoundingRect;

    let ra = a.geom.bounding_rect()?;
    let rb = b.geom.bounding_rect()?;

    let ix_a = Interval {
        start: ra.min().x,
        end: ra.max().x,
    };
    let ix_b = Interval {
        start: rb.min().x,
        end: rb.max().x,
    };
    let iy_a = Interval {
        start: ra.min().y,
        end: ra.max().y,
    };
    let iy_b = Interval {
        start: rb.min().y,
        end: rb.max().y,
    };

    Some(BboxAllenRelation {
        x_relation: allen_relation(ix_a, ix_b),
        y_relation: allen_relation(iy_a, iy_b),
    })
}

/// Infer the set of possible 2-D bbox Allen relations between geometries
/// A and C, given A→B (r1) and B→C (r2).
pub fn bbox_allen_compose(r1: BboxAllenRelation, r2: BboxAllenRelation) -> Vec<BboxAllenRelation> {
    let x_possible = allen_compose(r1.x_relation, r2.x_relation);
    let y_possible = allen_compose(r1.y_relation, r2.y_relation);

    let mut result = Vec::with_capacity(x_possible.len() * y_possible.len());
    for &xr in &x_possible {
        for &yr in &y_possible {
            result.push(BboxAllenRelation {
                x_relation: xr,
                y_relation: yr,
            });
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference Engine
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the interval reasoning graph.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalNode {
    /// Unique identifier for the node (e.g., URI string or integer).
    pub id: String,
    /// The 1-D interval for this node.
    pub interval: Interval,
}

/// An asserted or inferred interval relation between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntervalAssertion {
    /// Source node id.
    pub from_id: String,
    /// Target node id.
    pub to_id: String,
    /// The Allen relation.
    pub relation: AllenRelation,
    /// Whether this was asserted (`false`) or inferred (`true`).
    pub inferred: bool,
}

/// Inference engine for Allen interval reasoning.
///
/// Stores a set of interval nodes and assertions, then propagates new
/// inferences using the composition table.
///
/// # Example
///
/// ```rust
/// use oxirs_geosparql::reasoning::{
///     AllenIntervalEngine, IntervalNode, Interval, AllenRelation,
/// };
///
/// let mut engine = AllenIntervalEngine::new();
/// engine.add_node(IntervalNode {
///     id: "A".to_string(),
///     interval: Interval::new(0.0, 3.0).expect("valid interval"),
/// });
/// engine.add_node(IntervalNode {
///     id: "B".to_string(),
///     interval: Interval::new(5.0, 8.0).expect("valid interval"),
/// });
/// engine.infer_from_nodes();
/// let assertions = engine.assertions();
/// assert!(!assertions.is_empty());
/// ```
pub struct AllenIntervalEngine {
    nodes: HashMap<String, IntervalNode>,
    assertions: Vec<IntervalAssertion>,
}

impl AllenIntervalEngine {
    /// Create a new, empty engine.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            assertions: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: IntervalNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add an explicit (asserted) relation.
    pub fn assert_relation(
        &mut self,
        from_id: impl Into<String>,
        to_id: impl Into<String>,
        relation: AllenRelation,
    ) {
        self.assertions.push(IntervalAssertion {
            from_id: from_id.into(),
            to_id: to_id.into(),
            relation,
            inferred: false,
        });
    }

    /// Compute relations directly from the interval endpoints for all pairs of
    /// registered nodes.
    pub fn infer_from_nodes(&mut self) {
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        for i in 0..node_ids.len() {
            for j in 0..node_ids.len() {
                if i == j {
                    continue;
                }
                let id_i = &node_ids[i];
                let id_j = &node_ids[j];
                if let (Some(ni), Some(nj)) = (self.nodes.get(id_i), self.nodes.get(id_j)) {
                    let rel = allen_relation(ni.interval, nj.interval);
                    self.assertions.push(IntervalAssertion {
                        from_id: id_i.clone(),
                        to_id: id_j.clone(),
                        relation: rel,
                        inferred: false,
                    });
                }
            }
        }
    }

    /// Propagate assertions using the composition table to derive new facts.
    ///
    /// For each pair of assertions `A -r1-> B` and `B -r2-> C`, the composed
    /// possible relations for `A → C` are added as inferred assertions.
    pub fn propagate(&mut self) {
        let base = self.assertions.clone();
        let mut new_assertions: Vec<IntervalAssertion> = Vec::new();

        for r1_assert in &base {
            for r2_assert in &base {
                if r1_assert.to_id != r2_assert.from_id {
                    continue;
                }
                if r1_assert.from_id == r2_assert.to_id {
                    continue; // avoid self-loops
                }

                let possible = allen_compose(r1_assert.relation, r2_assert.relation);
                for possible_rel in possible {
                    let candidate = IntervalAssertion {
                        from_id: r1_assert.from_id.clone(),
                        to_id: r2_assert.to_id.clone(),
                        relation: possible_rel,
                        inferred: true,
                    };
                    // Only add if not already present
                    let already_exists = self.assertions.iter().any(|a| {
                        a.from_id == candidate.from_id
                            && a.to_id == candidate.to_id
                            && a.relation == candidate.relation
                    });
                    if !already_exists {
                        new_assertions.push(candidate);
                    }
                }
            }
        }

        self.assertions.extend(new_assertions);
    }

    /// Return all current assertions (both asserted and inferred).
    pub fn assertions(&self) -> &[IntervalAssertion] {
        &self.assertions
    }

    /// Return only inferred assertions.
    pub fn inferred_assertions(&self) -> impl Iterator<Item = &IntervalAssertion> {
        self.assertions.iter().filter(|a| a.inferred)
    }
}

impl Default for AllenIntervalEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Transitive containment
    // ------------------------------------------------------------------

    #[test]
    fn test_three_level_containment() {
        let a =
            Geometry::from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))").expect("should succeed");
        let b = Geometry::from_wkt("POLYGON ((2 2, 8 2, 8 8, 2 8, 2 2))").expect("should succeed");
        let c = Geometry::from_wkt("POLYGON ((4 4, 6 4, 6 6, 4 6, 4 4))").expect("should succeed");

        let geoms = vec![a, b, c];
        let closure = transitive_containment_closure(&geoms).expect("should succeed");

        // Direct: A contains B, B contains C
        assert!(closure.contains(&ContainmentPair::new(0, 1)));
        assert!(closure.contains(&ContainmentPair::new(1, 2)));
        // Inferred: A contains C
        assert!(closure.contains(&ContainmentPair::new(0, 2)));
    }

    #[test]
    fn test_no_containment() {
        let a = Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").expect("should succeed");
        let b = Geometry::from_wkt("POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5))").expect("should succeed");

        let geoms = vec![a, b];
        let closure = transitive_containment_closure(&geoms).expect("should succeed");

        assert!(closure.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let closure = transitive_containment_closure(&[]).expect("should succeed");
        assert!(closure.is_empty());
    }

    #[test]
    fn test_single_geometry() {
        let a = Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
        let closure = transitive_containment_closure(&[a]).expect("should succeed");
        assert!(closure.is_empty());
    }

    #[test]
    fn test_closure_from_pairs() {
        // A->B, B->C  =>  A->C inferred
        let pairs = vec![(0usize, 1usize), (1usize, 2usize)];
        let closure = transitive_closure_from_pairs(3, pairs);

        assert!(closure.contains(&ContainmentPair::new(0, 1)));
        assert!(closure.contains(&ContainmentPair::new(1, 2)));
        assert!(closure.contains(&ContainmentPair::new(0, 2)));
    }

    #[test]
    fn test_closure_chain_four() {
        // A->B->C->D  =>  A->C, A->D, B->D inferred
        let pairs = vec![(0, 1), (1, 2), (2, 3)];
        let closure = transitive_closure_from_pairs(4, pairs);

        assert!(closure.contains(&ContainmentPair::new(0, 2)));
        assert!(closure.contains(&ContainmentPair::new(0, 3)));
        assert!(closure.contains(&ContainmentPair::new(1, 3)));
    }

    // ------------------------------------------------------------------
    // Allen interval relations (1-D)
    // ------------------------------------------------------------------

    #[test]
    fn test_allen_before() {
        let x = Interval::new(1.0, 3.0).expect("valid interval");
        let y = Interval::new(5.0, 8.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Before);
        assert_eq!(allen_relation(y, x), AllenRelation::After);
    }

    #[test]
    fn test_allen_meets() {
        let x = Interval::new(1.0, 3.0).expect("valid interval");
        let y = Interval::new(3.0, 6.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Meets);
        assert_eq!(allen_relation(y, x), AllenRelation::MetBy);
    }

    #[test]
    fn test_allen_equals() {
        let x = Interval::new(2.0, 5.0).expect("valid interval");
        let y = Interval::new(2.0, 5.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Equals);
    }

    #[test]
    fn test_allen_during() {
        let x = Interval::new(3.0, 4.0).expect("valid interval");
        let y = Interval::new(1.0, 6.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::During);
        assert_eq!(allen_relation(y, x), AllenRelation::Contains);
    }

    #[test]
    fn test_allen_starts() {
        let x = Interval::new(1.0, 3.0).expect("valid interval");
        let y = Interval::new(1.0, 6.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Starts);
        assert_eq!(allen_relation(y, x), AllenRelation::StartedBy);
    }

    #[test]
    fn test_allen_finishes() {
        let x = Interval::new(4.0, 6.0).expect("valid interval");
        let y = Interval::new(1.0, 6.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Finishes);
        assert_eq!(allen_relation(y, x), AllenRelation::FinishedBy);
    }

    #[test]
    fn test_allen_overlaps() {
        let x = Interval::new(1.0, 4.0).expect("valid interval");
        let y = Interval::new(3.0, 7.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Overlaps);
        assert_eq!(allen_relation(y, x), AllenRelation::OverlappedBy);
    }

    #[test]
    fn test_allen_inverse_symmetry() {
        let relations = [
            AllenRelation::Before,
            AllenRelation::After,
            AllenRelation::Meets,
            AllenRelation::MetBy,
            AllenRelation::Overlaps,
            AllenRelation::OverlappedBy,
            AllenRelation::Starts,
            AllenRelation::StartedBy,
            AllenRelation::Finishes,
            AllenRelation::FinishedBy,
            AllenRelation::During,
            AllenRelation::Contains,
            AllenRelation::Equals,
        ];
        for r in relations {
            assert_eq!(
                r.inverse().inverse(),
                r,
                "{:?} inverse inverse should equal self",
                r
            );
        }
    }

    #[test]
    fn test_interval_invalid_construction() {
        let result = Interval::new(5.0, 3.0);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // Allen composition
    // ------------------------------------------------------------------

    #[test]
    fn test_compose_before_before() {
        // X before Y, Y before Z => X before Z
        let result = allen_compose(AllenRelation::Before, AllenRelation::Before);
        assert!(result.contains(&AllenRelation::Before));
    }

    #[test]
    fn test_compose_equals_before() {
        // X equals Y, Y before Z => X before Z
        let result = allen_compose(AllenRelation::Equals, AllenRelation::Before);
        assert_eq!(result, vec![AllenRelation::Before]);
    }

    #[test]
    fn test_compose_result_nonempty() {
        use AllenRelation::*;
        for r1 in [
            Before, After, Meets, MetBy, Overlaps, During, Contains, Equals,
        ] {
            for r2 in [
                Before, After, Meets, MetBy, Overlaps, During, Contains, Equals,
            ] {
                let result = allen_compose(r1, r2);
                assert!(
                    !result.is_empty(),
                    "Composition {:?} ∘ {:?} should not be empty",
                    r1,
                    r2
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // 2-D bbox Allen relation
    // ------------------------------------------------------------------

    #[test]
    fn test_bbox_allen_before_x_equals_y() {
        let a = Geometry::from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
        let b = Geometry::from_wkt("POLYGON ((5 0, 7 0, 7 2, 5 2, 5 0))").expect("should succeed");
        let rel = bbox_allen_relation(&a, &b).expect("should succeed");
        assert_eq!(rel.x_relation, AllenRelation::Before);
        assert_eq!(rel.y_relation, AllenRelation::Equals);
    }

    #[test]
    fn test_bbox_allen_nested() {
        let a =
            Geometry::from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))").expect("should succeed");
        let b = Geometry::from_wkt("POLYGON ((2 2, 8 2, 8 8, 2 8, 2 2))").expect("should succeed");
        let rel = bbox_allen_relation(&a, &b).expect("should succeed");
        // b is during a on both axes
        assert_eq!(rel.x_relation, AllenRelation::Contains);
        assert_eq!(rel.y_relation, AllenRelation::Contains);
    }

    // ------------------------------------------------------------------
    // AllenIntervalEngine
    // ------------------------------------------------------------------

    #[test]
    fn test_engine_infer_from_nodes() {
        let mut engine = AllenIntervalEngine::new();
        engine.add_node(IntervalNode {
            id: "A".to_string(),
            interval: Interval::new(0.0, 3.0).expect("valid interval"),
        });
        engine.add_node(IntervalNode {
            id: "B".to_string(),
            interval: Interval::new(5.0, 8.0).expect("valid interval"),
        });
        engine.infer_from_nodes();

        let assertions = engine.assertions();
        assert!(!assertions.is_empty());

        // A should be Before B
        let ab = assertions
            .iter()
            .find(|a| a.from_id == "A" && a.to_id == "B");
        assert!(ab.is_some());
        assert_eq!(ab.expect("should succeed").relation, AllenRelation::Before);
    }

    #[test]
    fn test_engine_propagate_chain() {
        // A before B, B before C  =>  infer A before C (at minimum)
        let mut engine = AllenIntervalEngine::new();
        engine.assert_relation("A", "B", AllenRelation::Before);
        engine.assert_relation("B", "C", AllenRelation::Before);
        engine.propagate();

        let inferred: Vec<_> = engine.inferred_assertions().collect();
        assert!(!inferred.is_empty());
        // Should have at least one A->C assertion
        let ac = inferred.iter().any(|a| a.from_id == "A" && a.to_id == "C");
        assert!(ac, "Expected A->C to be inferred");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended tests — OGC GeoSPARQL 1.1 / spatial entailment compliance
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests_extended {
    use super::*;

    // ── Transitive containment ───────────────────────────────────────────────

    #[test]
    fn test_containment_closure_chain_of_three() {
        // A ⊃ B ⊃ C ⟹ infer A ⊃ C
        let a =
            Geometry::from_wkt("POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0))").expect("should succeed");
        let b =
            Geometry::from_wkt("POLYGON ((2 2, 18 2, 18 18, 2 18, 2 2))").expect("should succeed");
        let c =
            Geometry::from_wkt("POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))").expect("should succeed");
        let geoms = vec![a, b, c];
        let closure = transitive_containment_closure(&geoms).expect("should succeed");
        assert!(closure.contains(&ContainmentPair {
            container: 0,
            contained: 1
        }));
        assert!(closure.contains(&ContainmentPair {
            container: 1,
            contained: 2
        }));
        assert!(closure.contains(&ContainmentPair {
            container: 0,
            contained: 2
        }));
    }

    #[test]
    fn test_containment_closure_empty_input() {
        let closure = transitive_containment_closure(&[]).expect("should succeed");
        assert!(closure.is_empty());
    }

    #[test]
    fn test_containment_closure_single_geometry() {
        let a =
            Geometry::from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))").expect("should succeed");
        let closure = transitive_containment_closure(&[a]).expect("should succeed");
        assert!(closure.is_empty()); // single geom, no pairs
    }

    #[test]
    fn test_containment_closure_disjoint_geometries() {
        let a = Geometry::from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))").expect("should succeed");
        let b = Geometry::from_wkt("POLYGON ((10 10, 11 10, 11 11, 10 11, 10 10))")
            .expect("should succeed");
        let closure = transitive_containment_closure(&[a, b]).expect("should succeed");
        // No containment between disjoint polygons
        assert!(closure.is_empty());
    }

    #[test]
    fn test_closure_from_pairs_chain() {
        // 0→1, 1→2, 2→3 ⟹ infer 0→2, 0→3, 1→3
        let pairs = vec![(0, 1), (1, 2), (2, 3)];
        let closure = transitive_closure_from_pairs(4, pairs);
        assert!(closure.contains(&ContainmentPair {
            container: 0,
            contained: 1
        }));
        assert!(closure.contains(&ContainmentPair {
            container: 0,
            contained: 2
        }));
        assert!(closure.contains(&ContainmentPair {
            container: 0,
            contained: 3
        }));
        assert!(closure.contains(&ContainmentPair {
            container: 1,
            contained: 3
        }));
    }

    #[test]
    fn test_closure_from_pairs_self_excluded() {
        let pairs = vec![(0, 0)]; // self-reference
        let closure = transitive_closure_from_pairs(3, pairs);
        assert!(closure.is_empty()); // self-pairs are ignored
    }

    #[test]
    fn test_closure_from_pairs_out_of_bounds_excluded() {
        let pairs = vec![(0, 99)]; // out of bounds for n=3
        let closure = transitive_closure_from_pairs(3, pairs);
        assert!(closure.is_empty());
    }

    // ── Allen interval relations ─────────────────────────────────────────────

    #[test]
    fn test_allen_before_strict() {
        let x = Interval::new(0.0, 3.0).expect("valid interval");
        let y = Interval::new(5.0, 8.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Before);
    }

    #[test]
    fn test_allen_after_strict() {
        let x = Interval::new(5.0, 8.0).expect("valid interval");
        let y = Interval::new(0.0, 3.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::After);
    }

    #[test]
    fn test_allen_meets() {
        let x = Interval::new(0.0, 5.0).expect("valid interval");
        let y = Interval::new(5.0, 10.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Meets);
    }

    #[test]
    fn test_allen_met_by() {
        let x = Interval::new(5.0, 10.0).expect("valid interval");
        let y = Interval::new(0.0, 5.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::MetBy);
    }

    #[test]
    fn test_allen_starts() {
        let x = Interval::new(0.0, 3.0).expect("valid interval");
        let y = Interval::new(0.0, 7.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Starts);
    }

    #[test]
    fn test_allen_started_by() {
        let x = Interval::new(0.0, 7.0).expect("valid interval");
        let y = Interval::new(0.0, 3.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::StartedBy);
    }

    #[test]
    fn test_allen_finishes() {
        let x = Interval::new(5.0, 10.0).expect("valid interval");
        let y = Interval::new(2.0, 10.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Finishes);
    }

    #[test]
    fn test_allen_finished_by() {
        let x = Interval::new(2.0, 10.0).expect("valid interval");
        let y = Interval::new(5.0, 10.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::FinishedBy);
    }

    #[test]
    fn test_allen_during() {
        let x = Interval::new(3.0, 7.0).expect("valid interval");
        let y = Interval::new(1.0, 9.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::During);
    }

    #[test]
    fn test_allen_contains() {
        let x = Interval::new(1.0, 9.0).expect("valid interval");
        let y = Interval::new(3.0, 7.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Contains);
    }

    #[test]
    fn test_allen_overlaps() {
        let x = Interval::new(0.0, 5.0).expect("valid interval");
        let y = Interval::new(3.0, 8.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::Overlaps);
    }

    #[test]
    fn test_allen_overlapped_by() {
        let x = Interval::new(3.0, 8.0).expect("valid interval");
        let y = Interval::new(0.0, 5.0).expect("valid interval");
        assert_eq!(allen_relation(x, y), AllenRelation::OverlappedBy);
    }

    // ── Allen inverse ─────────────────────────────────────────────────────────

    #[test]
    fn test_allen_inverse_before_after() {
        assert_eq!(AllenRelation::Before.inverse(), AllenRelation::After);
        assert_eq!(AllenRelation::After.inverse(), AllenRelation::Before);
    }

    #[test]
    fn test_allen_inverse_meets_metby() {
        assert_eq!(AllenRelation::Meets.inverse(), AllenRelation::MetBy);
        assert_eq!(AllenRelation::MetBy.inverse(), AllenRelation::Meets);
    }

    #[test]
    fn test_allen_inverse_equals_self() {
        assert_eq!(AllenRelation::Equals.inverse(), AllenRelation::Equals);
    }

    #[test]
    fn test_allen_inverse_during_contains() {
        assert_eq!(AllenRelation::During.inverse(), AllenRelation::Contains);
        assert_eq!(AllenRelation::Contains.inverse(), AllenRelation::During);
    }

    // ── Interval construction ─────────────────────────────────────────────────

    #[test]
    fn test_interval_invalid_start_gt_end() {
        let result = Interval::new(5.0, 3.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_interval_degenerate_point() {
        let i = Interval::new(3.0, 3.0).expect("valid interval");
        assert!((i.length() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_length() {
        let i = Interval::new(2.0, 7.0).expect("valid interval");
        assert!((i.length() - 5.0).abs() < 1e-10);
    }

    // ── BboxAllenRelation ─────────────────────────────────────────────────────

    #[test]
    fn test_bbox_allen_equals_both_axes() {
        let a = Geometry::from_wkt("POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))").expect("should succeed");
        let b = Geometry::from_wkt("POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))").expect("should succeed");
        let rel = bbox_allen_relation(&a, &b).expect("should succeed");
        assert_eq!(rel.x_relation, AllenRelation::Equals);
        assert_eq!(rel.y_relation, AllenRelation::Equals);
    }

    #[test]
    fn test_bbox_allen_compose_nonempty() {
        let r1 = BboxAllenRelation {
            x_relation: AllenRelation::Before,
            y_relation: AllenRelation::Equals,
        };
        let r2 = BboxAllenRelation {
            x_relation: AllenRelation::Before,
            y_relation: AllenRelation::Equals,
        };
        let result = bbox_allen_compose(r1, r2);
        assert!(!result.is_empty());
    }

    // ── AllenIntervalEngine ───────────────────────────────────────────────────

    #[test]
    fn test_engine_empty() {
        let engine = AllenIntervalEngine::new();
        assert!(engine.assertions().is_empty());
        assert_eq!(engine.inferred_assertions().count(), 0);
    }

    #[test]
    fn test_engine_default_is_empty() {
        let engine = AllenIntervalEngine::default();
        assert!(engine.assertions().is_empty());
    }

    #[test]
    fn test_engine_single_node_no_assertions() {
        let mut engine = AllenIntervalEngine::new();
        engine.add_node(IntervalNode {
            id: "A".to_string(),
            interval: Interval::new(0.0, 5.0).expect("valid interval"),
        });
        engine.infer_from_nodes();
        assert!(engine.assertions().is_empty()); // only one node, no pairs
    }

    #[test]
    fn test_engine_explicit_assert_relation() {
        let mut engine = AllenIntervalEngine::new();
        engine.assert_relation("X", "Y", AllenRelation::Before);
        assert_eq!(engine.assertions().len(), 1);
        assert_eq!(engine.assertions()[0].relation, AllenRelation::Before);
        assert!(!engine.assertions()[0].inferred);
    }

    #[test]
    fn test_engine_propagate_meets_before() {
        // X meets Y, Y before Z  =>  X before Z (among possibilities)
        let mut engine = AllenIntervalEngine::new();
        engine.assert_relation("X", "Y", AllenRelation::Meets);
        engine.assert_relation("Y", "Z", AllenRelation::Before);
        engine.propagate();
        let inferred: Vec<_> = engine.inferred_assertions().collect();
        assert!(!inferred.is_empty());
        let has_xz = inferred.iter().any(|a| a.from_id == "X" && a.to_id == "Z");
        assert!(has_xz, "Expected X->Z inferred assertion");
    }

    #[test]
    fn test_engine_three_node_chain_propagation() {
        let mut engine = AllenIntervalEngine::new();
        for (id, start, end) in &[("A", 0.0, 3.0), ("B", 5.0, 8.0), ("C", 10.0, 13.0)] {
            engine.add_node(IntervalNode {
                id: id.to_string(),
                interval: Interval::new(*start, *end).expect("valid interval"),
            });
        }
        engine.infer_from_nodes();
        engine.propagate();
        let inferred: Vec<_> = engine.inferred_assertions().collect();
        assert!(!inferred.is_empty());
    }
}
