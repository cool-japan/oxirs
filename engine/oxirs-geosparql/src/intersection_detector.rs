//! Geometric intersection detection for GeoSPARQL predicates.
//!
//! Implements pure-Rust 2-D computational geometry primitives:
//! * Point-in-polygon (ray-casting)
//! * Line–line intersection (general position + degenerate cases)
//! * Polygon–polygon overlap (bounding-box pre-check + detailed SAT-like)
//! * Containment (`within` / `contains`)
//! * Touches predicate (shared boundary, disjoint interiors)
//! * Crosses predicate (interior of A meets boundary of B)
//! * Segment–segment minimum distance
//! * Point-on-segment test (epsilon-tolerant)

/// Machine-epsilon tolerance for floating-point comparisons.
const EPSILON: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Primitive types
// ---------------------------------------------------------------------------

/// A 2-D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// The horizontal (east) coordinate.
    pub x: f64,
    /// The vertical (north) coordinate.
    pub y: f64,
}

impl Point {
    /// Construct a new `Point` from its `x` and `y` coordinates.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn dist_sq(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

/// A line segment from `start` to `end`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    /// The starting endpoint of the segment.
    pub start: Point,
    /// The ending endpoint of the segment.
    pub end: Point,
}

impl Segment {
    /// Construct a new `Segment` from its two endpoints.
    pub fn new(start: Point, end: Point) -> Self {
        Self { start, end }
    }

    /// Direction vector (dx, dy).
    fn dir(&self) -> (f64, f64) {
        (self.end.x - self.start.x, self.end.y - self.start.y)
    }

    /// Squared length.
    fn len_sq(&self) -> f64 {
        let (dx, dy) = self.dir();
        dx * dx + dy * dy
    }
}

/// A simple (non-self-intersecting) polygon defined by an ordered ring of vertices.
/// The ring is *not* required to repeat the first vertex at the end.
#[derive(Debug, Clone)]
pub struct Polygon {
    /// Exterior ring vertices in order (CW or CCW).
    pub vertices: Vec<Point>,
}

impl Polygon {
    /// Construct a polygon from a list of (x, y) pairs.
    pub fn from_coords(coords: impl IntoIterator<Item = (f64, f64)>) -> Self {
        let vertices = coords.into_iter().map(|(x, y)| Point::new(x, y)).collect();
        Self { vertices }
    }

    /// Iterate over edges as `Segment`s.
    fn edges(&self) -> impl Iterator<Item = Segment> + '_ {
        let n = self.vertices.len();
        (0..n).map(move |i| Segment::new(self.vertices[i], self.vertices[(i + 1) % n]))
    }

    /// Axis-aligned bounding box: (min_x, min_y, max_x, max_y).
    pub fn bbox(&self) -> Option<(f64, f64, f64, f64)> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min_x = self.vertices[0].x;
        let mut min_y = self.vertices[0].y;
        let mut max_x = min_x;
        let mut max_y = min_y;
        for v in &self.vertices[1..] {
            if v.x < min_x {
                min_x = v.x;
            }
            if v.y < min_y {
                min_y = v.y;
            }
            if v.x > max_x {
                max_x = v.x;
            }
            if v.y > max_y {
                max_y = v.y;
            }
        }
        Some((min_x, min_y, max_x, max_y))
    }
}

// ---------------------------------------------------------------------------
// Cross product helpers
// ---------------------------------------------------------------------------

/// Signed 2-D cross product of vectors `(b - a)` and `(c - a)`.
///
/// Positive: counter-clockwise turn; negative: clockwise; zero: collinear.
#[inline]
fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Return true when `v` is in the closed interval `[lo, hi]` (order-independent).
#[inline]
fn in_range(v: f64, lo: f64, hi: f64) -> bool {
    let (a, b) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    v >= a - EPSILON && v <= b + EPSILON
}

// ---------------------------------------------------------------------------
// 1. Point-on-segment test
// ---------------------------------------------------------------------------

/// Return `true` when `p` lies on the segment `seg` within `EPSILON`.
///
/// The check first requires the cross product of `(end - start, p - start)` to
/// be near zero (collinearity) then confirms `p` falls inside the bounding box
/// of the segment.
pub fn point_on_segment(p: Point, seg: Segment) -> bool {
    let cp = cross(seg.start, seg.end, p);
    if cp.abs() > EPSILON * (seg.len_sq().sqrt().max(1.0)) {
        return false;
    }
    in_range(p.x, seg.start.x, seg.end.x) && in_range(p.y, seg.start.y, seg.end.y)
}

// ---------------------------------------------------------------------------
// 2. Segment–segment intersection
// ---------------------------------------------------------------------------

/// The result of a segment-intersection test.
#[derive(Debug, Clone, PartialEq)]
pub enum SegmentIntersection {
    /// The segments do not intersect.
    None,
    /// The segments intersect at exactly one point.
    Point(Point),
    /// The segments are collinear and overlap on a sub-segment.
    Overlap(Segment),
}

/// Compute the intersection of two segments `ab` and `cd`.
///
/// Handles all degenerate cases (collinear overlap, endpoint touching).
pub fn segment_intersection(ab: Segment, cd: Segment) -> SegmentIntersection {
    let d1 = ab.dir();
    let d2 = cd.dir();
    let denom = d1.0 * d2.1 - d1.1 * d2.0;

    if denom.abs() < EPSILON {
        // Parallel or collinear
        let cp = cross(ab.start, ab.end, cd.start);
        if cp.abs() > EPSILON {
            return SegmentIntersection::None; // Parallel, not collinear
        }
        // Collinear: project onto axis with larger extent
        let use_x = (ab.end.x - ab.start.x).abs() > (ab.end.y - ab.start.y).abs();
        let (t_a0, t_a1, t_c0, t_c1) = if use_x {
            (ab.start.x, ab.end.x, cd.start.x, cd.end.x)
        } else {
            (ab.start.y, ab.end.y, cd.start.y, cd.end.y)
        };
        let (lo_a, hi_a) = if t_a0 <= t_a1 {
            (t_a0, t_a1)
        } else {
            (t_a1, t_a0)
        };
        let (lo_c, hi_c) = if t_c0 <= t_c1 {
            (t_c0, t_c1)
        } else {
            (t_c1, t_c0)
        };
        let lo = lo_a.max(lo_c);
        let hi = hi_a.min(hi_c);
        if lo > hi + EPSILON {
            return SegmentIntersection::None;
        }
        if (hi - lo).abs() < EPSILON {
            // Touch at a single point
            let t = if use_x {
                let len = (ab.end.x - ab.start.x).abs();
                if len < EPSILON {
                    0.0
                } else {
                    (lo - ab.start.x) / (ab.end.x - ab.start.x)
                }
            } else {
                let len = (ab.end.y - ab.start.y).abs();
                if len < EPSILON {
                    0.0
                } else {
                    (lo - ab.start.y) / (ab.end.y - ab.start.y)
                }
            };
            let pt = Point::new(ab.start.x + t * d1.0, ab.start.y + t * d1.1);
            return SegmentIntersection::Point(pt);
        }
        // True overlap
        let (p0, p1) = if use_x {
            let t0 = (lo - ab.start.x) / (ab.end.x - ab.start.x + EPSILON);
            let t1 = (hi - ab.start.x) / (ab.end.x - ab.start.x + EPSILON);
            (
                Point::new(ab.start.x + t0 * d1.0, ab.start.y + t0 * d1.1),
                Point::new(ab.start.x + t1 * d1.0, ab.start.y + t1 * d1.1),
            )
        } else {
            let t0 = (lo - ab.start.y) / (ab.end.y - ab.start.y + EPSILON);
            let t1 = (hi - ab.start.y) / (ab.end.y - ab.start.y + EPSILON);
            (
                Point::new(ab.start.x + t0 * d1.0, ab.start.y + t0 * d1.1),
                Point::new(ab.start.x + t1 * d1.0, ab.start.y + t1 * d1.1),
            )
        };
        return SegmentIntersection::Overlap(Segment::new(p0, p1));
    }

    let diff = (cd.start.x - ab.start.x, cd.start.y - ab.start.y);
    let t = (diff.0 * d2.1 - diff.1 * d2.0) / denom;
    let u = (diff.0 * d1.1 - diff.1 * d1.0) / denom;

    if (-EPSILON..=1.0 + EPSILON).contains(&t) && (-EPSILON..=1.0 + EPSILON).contains(&u) {
        let pt = Point::new(ab.start.x + t * d1.0, ab.start.y + t * d1.1);
        SegmentIntersection::Point(pt)
    } else {
        SegmentIntersection::None
    }
}

/// Return `true` if the two segments share at least one point.
pub fn segments_intersect(ab: Segment, cd: Segment) -> bool {
    segment_intersection(ab, cd) != SegmentIntersection::None
}

// ---------------------------------------------------------------------------
// 3. Point-in-polygon (ray-casting)
// ---------------------------------------------------------------------------

/// Return `true` when `p` is strictly inside `poly` or on its boundary.
pub fn point_in_polygon(p: Point, poly: &Polygon) -> bool {
    if poly.vertices.len() < 3 {
        return false;
    }
    // Boundary check first
    for edge in poly.edges() {
        if point_on_segment(p, edge) {
            return true;
        }
    }
    // Ray-casting (horizontal ray to the right)
    let mut inside = false;
    let n = poly.vertices.len();
    let mut j = n - 1;
    for i in 0..n {
        let vi = poly.vertices[i];
        let vj = poly.vertices[j];
        let cond1 = (vi.y > p.y) != (vj.y > p.y);
        let cond2 = p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x;
        if cond1 && cond2 {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ---------------------------------------------------------------------------
// 4. Polygon overlap / intersection
// ---------------------------------------------------------------------------

/// Return `true` when the bounding boxes of `a` and `b` overlap.
pub fn bbox_overlap(a: &Polygon, b: &Polygon) -> bool {
    match (a.bbox(), b.bbox()) {
        (Some((ax0, ay0, ax1, ay1)), Some((bx0, by0, bx1, by1))) => {
            ax0 <= bx1 + EPSILON
                && ax1 >= bx0 - EPSILON
                && ay0 <= by1 + EPSILON
                && ay1 >= by0 - EPSILON
        }
        _ => false,
    }
}

/// Return `true` when point `p` is *strictly* inside `poly` (not on the boundary).
fn point_strictly_inside(p: Point, poly: &Polygon) -> bool {
    if poly.vertices.len() < 3 {
        return false;
    }
    // First verify it is not on any boundary edge.
    for edge in poly.edges() {
        if point_on_segment(p, edge) {
            return false;
        }
    }
    // Ray-casting for interior test.
    let mut inside = false;
    let n = poly.vertices.len();
    let mut j = n - 1;
    for i in 0..n {
        let vi = poly.vertices[i];
        let vj = poly.vertices[j];
        let cond1 = (vi.y > p.y) != (vj.y > p.y);
        let cond2 = p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x;
        if cond1 && cond2 {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Return `true` when the interiors of `a` and `b` overlap (have a non-empty
/// common area).
///
/// Uses:
/// 1. Bounding-box pre-check.
/// 2. Any edge of A properly crosses (non-collinear, non-endpoint) an edge of B.
/// 3. Any vertex of A is strictly inside B (interior overlap without crossing).
/// 4. Any vertex of B is strictly inside A.
///
/// Boundary-only contact (shared edges or shared vertices) is NOT counted as
/// interior overlap; that is the "touches" relation.
pub fn polygons_overlap(a: &Polygon, b: &Polygon) -> bool {
    if !bbox_overlap(a, b) {
        return false;
    }
    // Check for proper edge crossings (non-collinear intersection at an interior
    // point of both edges — not just at shared endpoints).
    for ea in a.edges() {
        for eb in b.edges() {
            if let SegmentIntersection::Point(pt) = segment_intersection(ea, eb) {
                // Accept only intersections that are interior to at least one
                // of the two edges (not touching at a shared endpoint only).
                let at_ea_endpoint = pt.dist_sq(&ea.start) < EPSILON * EPSILON
                    || pt.dist_sq(&ea.end) < EPSILON * EPSILON;
                let at_eb_endpoint = pt.dist_sq(&eb.start) < EPSILON * EPSILON
                    || pt.dist_sq(&eb.end) < EPSILON * EPSILON;
                // If the intersection is at an endpoint of BOTH edges it is a
                // shared vertex: boundary contact only.  If it is interior to at
                // least one edge, the interiors genuinely overlap.
                if !(at_ea_endpoint && at_eb_endpoint) {
                    return true;
                }
            }
        }
    }
    // One polygon may be entirely inside the other — check with strict interior
    // test so that boundary-only contact is not mis-classified as overlap.
    for v in &a.vertices {
        if point_strictly_inside(*v, b) {
            return true;
        }
    }
    for v in &b.vertices {
        if point_strictly_inside(*v, a) {
            return true;
        }
    }
    // Fallback for the case where both polygons are congruent or concentric:
    // all vertices of each lie on the boundary of the other, yet the interiors
    // overlap.  The centroid of a simple polygon is guaranteed to be inside the
    // polygon for convex shapes and usually inside for non-convex shapes.  Use
    // the centroid of `a` as a probe point for `b` and vice-versa.
    if !a.vertices.is_empty() {
        let n = a.vertices.len() as f64;
        let cx = a.vertices.iter().map(|v| v.x).sum::<f64>() / n;
        let cy = a.vertices.iter().map(|v| v.y).sum::<f64>() / n;
        if point_strictly_inside(Point::new(cx, cy), b) {
            return true;
        }
    }
    if !b.vertices.is_empty() {
        let n = b.vertices.len() as f64;
        let cx = b.vertices.iter().map(|v| v.x).sum::<f64>() / n;
        let cy = b.vertices.iter().map(|v| v.y).sum::<f64>() / n;
        if point_strictly_inside(Point::new(cx, cy), a) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// 5. Containment
// ---------------------------------------------------------------------------

/// Return `true` when polygon `inner` is fully contained within polygon `outer`.
///
/// All vertices of `inner` must be inside `outer`, and no edge of `inner` must
/// cross any edge of `outer`.
pub fn polygon_contains_polygon(outer: &Polygon, inner: &Polygon) -> bool {
    if inner.vertices.is_empty() {
        return true; // Empty polygon is vacuously contained
    }
    for v in &inner.vertices {
        if !point_in_polygon(*v, outer) {
            return false;
        }
    }
    // No boundary crossing allowed for strict containment
    for ei in inner.edges() {
        for eo in outer.edges() {
            if let SegmentIntersection::Point(pt) = segment_intersection(ei, eo) {
                // Touching at a vertex of inner/outer is fine; interior crossing is not
                let on_inner_endpoint = pt.dist_sq(&ei.start) < EPSILON * EPSILON
                    || pt.dist_sq(&ei.end) < EPSILON * EPSILON;
                let on_outer_endpoint = pt.dist_sq(&eo.start) < EPSILON * EPSILON
                    || pt.dist_sq(&eo.end) < EPSILON * EPSILON;
                if !on_inner_endpoint && !on_outer_endpoint {
                    return false;
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// 6. Touches predicate
// ---------------------------------------------------------------------------

/// Return `true` when `a` and `b` touch: they share at least one boundary
/// point but their interiors are disjoint.
///
/// They must *not* have interior overlap, but at least one point of contact on
/// a boundary.
pub fn polygons_touch(a: &Polygon, b: &Polygon) -> bool {
    if polygons_overlap(a, b) {
        return false; // Interiors overlap → not "touches"
    }
    // Check for any edge pair with a shared point (boundary contact)
    for ea in a.edges() {
        for eb in b.edges() {
            if segment_intersection(ea, eb) != SegmentIntersection::None {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// 7. Crosses predicate (DE-9IM)
// ---------------------------------------------------------------------------

/// Return `true` when the interior of `a` properly intersects the boundary of
/// `b` (or vice-versa) but neither fully contains the other.
///
/// The DE-9IM "crosses" relation for polygons requires:
/// * The interiors have a non-empty intersection (overlap exists).
/// * Neither polygon is fully contained within the other.
/// * They are not merely touching on the boundary (disjoint interiors).
///
/// This is detected by:
/// 1. Verifying the polygons actually overlap (interiors share area).
/// 2. Verifying neither fully contains the other.
pub fn polygons_cross(a: &Polygon, b: &Polygon) -> bool {
    if !bbox_overlap(a, b) {
        return false;
    }
    // If one fully contains the other, the relation is `within`/`contains`, not `crosses`
    if polygon_contains_polygon(a, b) || polygon_contains_polygon(b, a) {
        return false;
    }
    // Must have genuine interior overlap (not just touching on boundary)
    polygons_overlap(a, b)
}

// ---------------------------------------------------------------------------
// 8. Segment–segment minimum distance
// ---------------------------------------------------------------------------

/// Return the minimum Euclidean distance between two segments.
///
/// If the segments intersect, the distance is 0.
pub fn segment_segment_distance(ab: Segment, cd: Segment) -> f64 {
    if segment_intersection(ab, cd) != SegmentIntersection::None {
        return 0.0;
    }
    // Minimum of the four point-to-segment distances
    let d1 = point_to_segment_dist(ab.start, cd);
    let d2 = point_to_segment_dist(ab.end, cd);
    let d3 = point_to_segment_dist(cd.start, ab);
    let d4 = point_to_segment_dist(cd.end, ab);
    d1.min(d2).min(d3).min(d4)
}

/// Compute the minimum distance from point `p` to segment `seg`.
pub fn point_to_segment_dist(p: Point, seg: Segment) -> f64 {
    let len_sq = seg.len_sq();
    if len_sq < EPSILON * EPSILON {
        // Degenerate segment (point)
        return (p.dist_sq(&seg.start)).sqrt();
    }
    let (dx, dy) = seg.dir();
    let t = ((p.x - seg.start.x) * dx + (p.y - seg.start.y) * dy) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);
    let closest = Point::new(seg.start.x + t_clamped * dx, seg.start.y + t_clamped * dy);
    (p.dist_sq(&closest)).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Point {
        Point::new(x, y)
    }

    fn seg(x0: f64, y0: f64, x1: f64, y1: f64) -> Segment {
        Segment::new(pt(x0, y0), pt(x1, y1))
    }

    fn square(cx: f64, cy: f64, half: f64) -> Polygon {
        Polygon::from_coords([
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ])
    }

    // -----------------------------------------------------------------------
    // point_on_segment
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_on_segment_midpoint() {
        let s = seg(0.0, 0.0, 4.0, 0.0);
        assert!(point_on_segment(pt(2.0, 0.0), s));
    }

    #[test]
    fn test_point_on_segment_endpoint() {
        let s = seg(1.0, 1.0, 5.0, 1.0);
        assert!(point_on_segment(pt(1.0, 1.0), s));
        assert!(point_on_segment(pt(5.0, 1.0), s));
    }

    #[test]
    fn test_point_not_on_segment() {
        let s = seg(0.0, 0.0, 4.0, 0.0);
        assert!(!point_on_segment(pt(2.0, 1.0), s));
    }

    #[test]
    fn test_point_on_segment_diagonal() {
        let s = seg(0.0, 0.0, 3.0, 3.0);
        assert!(point_on_segment(pt(1.5, 1.5), s));
        assert!(!point_on_segment(pt(1.5, 1.6), s));
    }

    #[test]
    fn test_point_on_segment_extension() {
        // Point on the line but outside the segment
        let s = seg(0.0, 0.0, 2.0, 0.0);
        assert!(!point_on_segment(pt(3.0, 0.0), s));
    }

    // -----------------------------------------------------------------------
    // segment_intersection
    // -----------------------------------------------------------------------

    #[test]
    fn test_segments_cross_at_origin() {
        let ab = seg(-1.0, 0.0, 1.0, 0.0);
        let cd = seg(0.0, -1.0, 0.0, 1.0);
        match segment_intersection(ab, cd) {
            SegmentIntersection::Point(p) => {
                assert!((p.x).abs() < 1e-9);
                assert!((p.y).abs() < 1e-9);
            }
            other => panic!("Expected Point, got {other:?}"),
        }
    }

    #[test]
    fn test_segments_parallel() {
        let ab = seg(0.0, 0.0, 4.0, 0.0);
        let cd = seg(0.0, 1.0, 4.0, 1.0);
        assert_eq!(segment_intersection(ab, cd), SegmentIntersection::None);
    }

    #[test]
    fn test_segments_collinear_overlap() {
        let ab = seg(0.0, 0.0, 4.0, 0.0);
        let cd = seg(2.0, 0.0, 6.0, 0.0);
        match segment_intersection(ab, cd) {
            SegmentIntersection::Overlap(_) => {}
            other => panic!("Expected Overlap, got {other:?}"),
        }
    }

    #[test]
    fn test_segments_collinear_touch() {
        // Touch at a single endpoint
        let ab = seg(0.0, 0.0, 2.0, 0.0);
        let cd = seg(2.0, 0.0, 4.0, 0.0);
        match segment_intersection(ab, cd) {
            SegmentIntersection::Point(p) => {
                assert!((p.x - 2.0).abs() < 1e-9);
            }
            other => panic!("Expected Point, got {other:?}"),
        }
    }

    #[test]
    fn test_segments_no_intersection() {
        let ab = seg(0.0, 0.0, 1.0, 0.0);
        let cd = seg(2.0, 0.0, 3.0, 0.0);
        assert_eq!(segment_intersection(ab, cd), SegmentIntersection::None);
    }

    #[test]
    fn test_segments_t_junction() {
        let ab = seg(0.0, 0.0, 4.0, 0.0);
        let cd = seg(2.0, -2.0, 2.0, 0.0);
        assert!(segments_intersect(ab, cd));
    }

    // -----------------------------------------------------------------------
    // point_in_polygon
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_inside_square() {
        let sq = square(0.0, 0.0, 2.0);
        assert!(point_in_polygon(pt(0.0, 0.0), &sq));
        assert!(point_in_polygon(pt(1.0, 1.0), &sq));
    }

    #[test]
    fn test_point_outside_square() {
        let sq = square(0.0, 0.0, 2.0);
        assert!(!point_in_polygon(pt(3.0, 0.0), &sq));
        assert!(!point_in_polygon(pt(0.0, 3.0), &sq));
    }

    #[test]
    fn test_point_on_boundary() {
        let sq = square(0.0, 0.0, 2.0);
        assert!(point_in_polygon(pt(2.0, 0.0), &sq));
    }

    #[test]
    fn test_point_in_polygon_triangle() {
        let tri = Polygon::from_coords([(0.0, 0.0), (4.0, 0.0), (2.0, 4.0)]);
        assert!(point_in_polygon(pt(2.0, 1.0), &tri));
        assert!(!point_in_polygon(pt(2.0, 5.0), &tri));
        assert!(!point_in_polygon(pt(-1.0, 0.0), &tri));
    }

    // -----------------------------------------------------------------------
    // bbox_overlap
    // -----------------------------------------------------------------------

    #[test]
    fn test_bbox_overlap_adjacent() {
        let a = square(0.0, 0.0, 1.0);
        let b = square(2.0, 0.0, 1.0);
        // Bboxes touch at x=1/x=1 — overlaps returns true (shared edge)
        assert!(bbox_overlap(&a, &b));
    }

    #[test]
    fn test_bbox_no_overlap() {
        let a = square(0.0, 0.0, 1.0);
        let b = square(5.0, 5.0, 1.0);
        assert!(!bbox_overlap(&a, &b));
    }

    // -----------------------------------------------------------------------
    // polygons_overlap
    // -----------------------------------------------------------------------

    #[test]
    fn test_polygons_overlap_crossing() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        assert!(polygons_overlap(&a, &b));
    }

    #[test]
    fn test_polygons_no_overlap() {
        let a = square(0.0, 0.0, 1.0);
        let b = square(5.0, 5.0, 1.0);
        assert!(!polygons_overlap(&a, &b));
    }

    #[test]
    fn test_polygons_overlap_containment() {
        let outer = square(0.0, 0.0, 3.0);
        let inner = square(0.0, 0.0, 1.0);
        // Inner is fully inside outer → interiors overlap
        assert!(polygons_overlap(&outer, &inner));
    }

    // -----------------------------------------------------------------------
    // polygon_contains_polygon
    // -----------------------------------------------------------------------

    #[test]
    fn test_containment_inner_inside_outer() {
        let outer = square(0.0, 0.0, 5.0);
        let inner = square(0.0, 0.0, 2.0);
        assert!(polygon_contains_polygon(&outer, &inner));
    }

    #[test]
    fn test_containment_not_contained() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(4.0, 0.0, 2.0);
        assert!(!polygon_contains_polygon(&a, &b));
    }

    #[test]
    fn test_containment_partial_overlap() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 0.0, 2.0); // Overlaps but not contained
        assert!(!polygon_contains_polygon(&a, &b));
    }

    // -----------------------------------------------------------------------
    // polygons_touch
    // -----------------------------------------------------------------------

    #[test]
    fn test_polygons_touch_shared_edge() {
        // Two adjacent squares sharing an edge
        let a = Polygon::from_coords([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]);
        let b = Polygon::from_coords([(2.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0)]);
        assert!(polygons_touch(&a, &b));
    }

    #[test]
    fn test_polygons_not_touch_disjoint() {
        let a = square(0.0, 0.0, 1.0);
        let b = square(5.0, 5.0, 1.0);
        assert!(!polygons_touch(&a, &b));
    }

    #[test]
    fn test_polygons_not_touch_overlap() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 1.0, 2.0);
        // They overlap → touch must be false
        assert!(!polygons_touch(&a, &b));
    }

    // -----------------------------------------------------------------------
    // polygons_cross
    // -----------------------------------------------------------------------

    #[test]
    fn test_polygons_cross_partial_overlap() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(1.0, 0.0, 2.0);
        // Edges properly cross interior-to-interior
        assert!(polygons_cross(&a, &b));
    }

    #[test]
    fn test_polygons_cross_containment_not_cross() {
        let outer = square(0.0, 0.0, 5.0);
        let inner = square(0.0, 0.0, 1.0);
        // Containment is not "crosses"
        assert!(!polygons_cross(&outer, &inner));
    }

    #[test]
    fn test_polygons_cross_disjoint_not_cross() {
        let a = square(0.0, 0.0, 1.0);
        let b = square(5.0, 5.0, 1.0);
        assert!(!polygons_cross(&a, &b));
    }

    // -----------------------------------------------------------------------
    // segment_segment_distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_segment_distance_parallel() {
        let ab = seg(0.0, 0.0, 4.0, 0.0);
        let cd = seg(0.0, 3.0, 4.0, 3.0);
        let dist = segment_segment_distance(ab, cd);
        assert!((dist - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_segment_distance_perpendicular_nooverlap() {
        let ab = seg(0.0, 0.0, 2.0, 0.0);
        let cd = seg(3.0, -1.0, 3.0, -3.0);
        // Closest points: (2,0) and (3,-1) → √2
        let dist = segment_segment_distance(ab, cd);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_segment_distance_intersecting_is_zero() {
        let ab = seg(-1.0, 0.0, 1.0, 0.0);
        let cd = seg(0.0, -1.0, 0.0, 1.0);
        let dist = segment_segment_distance(ab, cd);
        assert!(dist < 1e-9);
    }

    #[test]
    fn test_segment_distance_endpoint_to_segment() {
        let ab = seg(0.0, 0.0, 1.0, 0.0);
        let cd = seg(2.0, 0.0, 3.0, 0.0);
        let dist = segment_segment_distance(ab, cd);
        assert!((dist - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // point_to_segment_dist
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_to_segment_perpendicular() {
        let s = seg(0.0, 0.0, 4.0, 0.0);
        let d = point_to_segment_dist(pt(2.0, 3.0), s);
        assert!((d - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_to_segment_past_end() {
        let s = seg(0.0, 0.0, 4.0, 0.0);
        let d = point_to_segment_dist(pt(7.0, 0.0), s);
        assert!((d - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_to_segment_degenerate() {
        let s = seg(2.0, 2.0, 2.0, 2.0); // Point segment
        let d = point_to_segment_dist(pt(5.0, 6.0), s);
        assert!((d - 5.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Additional coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_struct() {
        let p = Point::new(3.0, 4.0);
        assert_eq!(p.x, 3.0);
        assert_eq!(p.y, 4.0);
    }

    #[test]
    fn test_polygon_from_coords_len() {
        let p = Polygon::from_coords([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        assert_eq!(p.vertices.len(), 4);
    }

    #[test]
    fn test_polygon_bbox_empty() {
        let p = Polygon { vertices: vec![] };
        assert!(p.bbox().is_none());
    }

    #[test]
    fn test_polygon_bbox_single_point() {
        let p = Polygon {
            vertices: vec![Point::new(3.0, 5.0)],
        };
        let bb = p.bbox().expect("bbox for single-point polygon");
        assert_eq!(bb, (3.0, 5.0, 3.0, 5.0));
    }

    #[test]
    fn test_point_in_polygon_concave() {
        // L-shaped concave polygon — point outside the notch
        let concave = Polygon::from_coords([
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (2.0, 4.0),
            (0.0, 4.0),
        ]);
        // Point in notch (upper-right) should be outside
        assert!(!point_in_polygon(pt(3.0, 3.0), &concave));
        // Point in body should be inside
        assert!(point_in_polygon(pt(1.0, 1.0), &concave));
    }

    #[test]
    fn test_polygons_overlap_identical() {
        let a = square(0.0, 0.0, 2.0);
        let b = square(0.0, 0.0, 2.0);
        // Identical polygons have overlapping interiors
        assert!(polygons_overlap(&a, &b));
    }

    #[test]
    fn test_containment_self_contained() {
        let a = square(0.0, 0.0, 2.0);
        // A contains itself
        assert!(polygon_contains_polygon(&a, &a));
    }

    #[test]
    fn test_segment_collinear_no_overlap() {
        // Collinear but no overlap
        let ab = seg(0.0, 0.0, 1.0, 0.0);
        let cd = seg(3.0, 0.0, 5.0, 0.0);
        assert_eq!(segment_intersection(ab, cd), SegmentIntersection::None);
    }

    #[test]
    fn test_cross_product_sign() {
        // CCW: A(0,0), B(1,0), C(0,1) → positive
        let result = super::cross(pt(0.0, 0.0), pt(1.0, 0.0), pt(0.0, 1.0));
        assert!(result > 0.0);
        // CW: A(0,0), B(0,1), C(1,0) → negative
        let result2 = super::cross(pt(0.0, 0.0), pt(0.0, 1.0), pt(1.0, 0.0));
        assert!(result2 < 0.0);
    }
}
