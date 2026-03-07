//! Spatial topology checking implementing a simplified DE-9IM model.
//!
//! Provides `contains`, `intersects`, `disjoint`, `touches`, `equals`,
//! `distance`, and related predicates for bounding-box, polygon, and
//! point geometries.
use std::f64;

// ── Primitive geometry types ──────────────────────────────────────────────────

/// A 2-D point.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Horizontal (longitude) coordinate.
    pub x: f64,
    /// Vertical (latitude) coordinate.
    pub y: f64,
}

/// An axis-aligned bounding box.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Minimum x (western edge).
    pub min_x: f64,
    /// Minimum y (southern edge).
    pub min_y: f64,
    /// Maximum x (eastern edge).
    pub max_x: f64,
    /// Maximum y (northern edge).
    pub max_y: f64,
}

/// A polygon with an exterior ring and optional holes.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    /// The outer boundary ring (counter-clockwise).
    pub exterior: Vec<Point>,
    /// Interior holes (clockwise winding for each hole).
    pub holes: Vec<Vec<Point>>,
}

/// A polyline.
#[derive(Debug, Clone, PartialEq)]
pub struct LineString {
    /// Ordered sequence of vertices.
    pub points: Vec<Point>,
}

/// Tagged union of supported geometry types.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// A single 2-D point.
    Point(Point),
    /// An axis-aligned bounding box.
    BoundingBox(BoundingBox),
    /// A polygon (with optional holes).
    Polygon(Polygon),
    /// A polyline.
    LineString(LineString),
}

/// A set of boolean DE-9IM–inspired topology flags.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TopologyRelation {
    /// `true` when `a` fully contains `b`.
    pub contains: bool,
    /// `true` when `a` is fully within `b`.
    pub within: bool,
    /// `true` when `a` and `b` share at least one point.
    pub intersects: bool,
    /// `true` when `a` and `b` share no points.
    pub disjoint: bool,
    /// `true` when `a` and `b` share boundary points only (no interior).
    pub touches: bool,
    /// `true` when `a` and `b` represent the same geometry.
    pub equals: bool,
    /// `true` when `a` and `b` intersect but neither contains the other.
    pub overlaps: bool,
    /// `true` when a line geometry crosses a polygon (or vice versa).
    pub crosses: bool,
}

// ── BoundingBox impl ──────────────────────────────────────────────────────────

impl BoundingBox {
    /// Create from individual coordinates (no ordering check – caller's responsibility).
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Compute the tight bounding box of a polygon.
    pub fn from_polygon(poly: &Polygon) -> Self {
        let xs = poly.exterior.iter().map(|p| p.x);
        let ys = poly.exterior.iter().map(|p| p.y);
        let min_x = xs.clone().fold(f64::INFINITY, f64::min);
        let max_x = xs.fold(f64::NEG_INFINITY, f64::max);
        let min_y = ys.clone().fold(f64::INFINITY, f64::min);
        let max_y = ys.fold(f64::NEG_INFINITY, f64::max);
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> Point {
        Point {
            x: (self.min_x + self.max_x) / 2.0,
            y: (self.min_y + self.max_y) / 2.0,
        }
    }
}

// ── TopologyChecker ───────────────────────────────────────────────────────────

/// Evaluates spatial topology relations between geometries.
pub struct TopologyChecker;

impl TopologyChecker {
    /// Compute all topology flags between `a` and `b`.
    pub fn relation(a: &Geometry, b: &Geometry) -> TopologyRelation {
        let contains = Self::contains(a, b);
        let within = Self::contains(b, a);
        let intersects = Self::intersects(a, b);
        let disjoint = !intersects;
        let equals = Self::equals(a, b);
        let touches = Self::touches(a, b);
        // overlaps: intersects but neither contains the other
        let overlaps = intersects && !contains && !within && !equals;
        // crosses: simplified — line crosses polygon / vice versa
        let crosses = Self::crosses_impl(a, b);
        TopologyRelation {
            contains,
            within,
            intersects,
            disjoint,
            touches,
            equals,
            overlaps,
            crosses,
        }
    }

    /// Returns `true` when `outer` fully contains `inner`.
    pub fn contains(outer: &Geometry, inner: &Geometry) -> bool {
        match (outer, inner) {
            (Geometry::BoundingBox(bb), Geometry::Point(p)) => Self::point_in_bbox(p, bb),
            (Geometry::BoundingBox(a), Geometry::BoundingBox(b)) => {
                a.min_x <= b.min_x && a.min_y <= b.min_y && a.max_x >= b.max_x && a.max_y >= b.max_y
            }
            (Geometry::Polygon(poly), Geometry::Point(p)) => Self::point_in_polygon(p, poly),
            (Geometry::BoundingBox(bb), Geometry::Polygon(poly)) => {
                poly.exterior.iter().all(|p| Self::point_in_bbox(p, bb))
            }
            (Geometry::Polygon(outer_p), Geometry::BoundingBox(bb)) => {
                // All corners of bb must be inside outer polygon
                let corners = bbox_corners(bb);
                corners.iter().all(|p| Self::point_in_polygon(p, outer_p))
            }
            _ => false,
        }
    }

    /// Returns `true` when the geometries share at least one common point.
    pub fn intersects(a: &Geometry, b: &Geometry) -> bool {
        match (a, b) {
            (Geometry::BoundingBox(ba), Geometry::BoundingBox(bb)) => Self::bbox_intersects(ba, bb),
            (Geometry::Point(p), Geometry::BoundingBox(bb)) => Self::point_in_bbox(p, bb),
            (Geometry::BoundingBox(bb), Geometry::Point(p)) => Self::point_in_bbox(p, bb),
            (Geometry::Point(p), Geometry::Polygon(poly)) => Self::point_in_polygon(p, poly),
            (Geometry::Polygon(poly), Geometry::Point(p)) => Self::point_in_polygon(p, poly),
            (Geometry::BoundingBox(bb), Geometry::Polygon(poly)) => {
                // BBox intersects polygon if any polygon vertex is in the bbox or
                // any bbox corner is in the polygon
                poly.exterior.iter().any(|p| Self::point_in_bbox(p, bb))
                    || bbox_corners(bb)
                        .iter()
                        .any(|p| Self::point_in_polygon(p, poly))
            }
            (Geometry::Polygon(poly), Geometry::BoundingBox(bb)) => Self::intersects(
                &Geometry::BoundingBox(bb.clone()),
                &Geometry::Polygon(poly.clone()),
            ),
            (Geometry::Polygon(pa), Geometry::Polygon(pb)) => {
                // Simplified: check if any vertex of pa is inside pb or vice versa
                pa.exterior.iter().any(|p| Self::point_in_polygon(p, pb))
                    || pb.exterior.iter().any(|p| Self::point_in_polygon(p, pa))
            }
            _ => false,
        }
    }

    /// Returns `true` when the geometries share no common point.
    pub fn disjoint(a: &Geometry, b: &Geometry) -> bool {
        !Self::intersects(a, b)
    }

    /// Returns `true` when the geometries touch but do not overlap (boundary only).
    /// For bounding boxes: they share an edge/corner but do not overlap interiors.
    pub fn touches(a: &Geometry, b: &Geometry) -> bool {
        match (a, b) {
            (Geometry::BoundingBox(ba), Geometry::BoundingBox(bb)) => bbox_touches(ba, bb),
            (Geometry::Point(p), Geometry::BoundingBox(bb)) => {
                // Point is exactly on the boundary of the bbox
                point_on_bbox_boundary(p, bb)
            }
            (Geometry::BoundingBox(bb), Geometry::Point(p)) => point_on_bbox_boundary(p, bb),
            _ => false,
        }
    }

    /// Returns `true` when the two geometries are equal.
    pub fn equals(a: &Geometry, b: &Geometry) -> bool {
        match (a, b) {
            (Geometry::Point(pa), Geometry::Point(pb)) => {
                (pa.x - pb.x).abs() < f64::EPSILON && (pa.y - pb.y).abs() < f64::EPSILON
            }
            (Geometry::BoundingBox(ba), Geometry::BoundingBox(bb)) => {
                (ba.min_x - bb.min_x).abs() < f64::EPSILON
                    && (ba.min_y - bb.min_y).abs() < f64::EPSILON
                    && (ba.max_x - bb.max_x).abs() < f64::EPSILON
                    && (ba.max_y - bb.max_y).abs() < f64::EPSILON
            }
            _ => false,
        }
    }

    /// Minimum Euclidean distance between two geometries.
    pub fn distance(a: &Geometry, b: &Geometry) -> f64 {
        match (a, b) {
            (Geometry::Point(pa), Geometry::Point(pb)) => point_distance(pa, pb),
            (Geometry::Point(p), Geometry::BoundingBox(bb)) => point_to_bbox_distance(p, bb),
            (Geometry::BoundingBox(bb), Geometry::Point(p)) => point_to_bbox_distance(p, bb),
            (Geometry::BoundingBox(ba), Geometry::BoundingBox(bb)) => {
                if Self::bbox_intersects(ba, bb) {
                    0.0
                } else {
                    bbox_to_bbox_distance(ba, bb)
                }
            }
            _ => 0.0,
        }
    }

    /// Returns `true` when the two bounding boxes intersect (share interior area).
    pub fn bbox_intersects(a: &BoundingBox, b: &BoundingBox) -> bool {
        a.min_x < b.max_x && a.max_x > b.min_x && a.min_y < b.max_y && a.max_y > b.min_y
    }

    /// Returns `true` when `p` lies inside or on the boundary of `bbox`.
    pub fn point_in_bbox(p: &Point, bbox: &BoundingBox) -> bool {
        p.x >= bbox.min_x && p.x <= bbox.max_x && p.y >= bbox.min_y && p.y <= bbox.max_y
    }

    /// Returns `true` when `p` is inside `poly` using the ray-casting algorithm.
    pub fn point_in_polygon(p: &Point, poly: &Polygon) -> bool {
        // Test exterior ring first
        if !ray_cast(p, &poly.exterior) {
            return false;
        }
        // Point must not be inside any hole
        for hole in &poly.holes {
            if ray_cast(p, hole) {
                return false;
            }
        }
        true
    }

    // ── Private helper ────────────────────────────────────────────────────────

    fn crosses_impl(a: &Geometry, b: &Geometry) -> bool {
        // Simplified: a LineString crosses a polygon when some of its vertices
        // are inside and some outside.
        match (a, b) {
            (Geometry::LineString(ls), Geometry::Polygon(poly)) => {
                let inside_count = ls
                    .points
                    .iter()
                    .filter(|p| Self::point_in_polygon(p, poly))
                    .count();
                let outside_count = ls.points.len() - inside_count;
                inside_count > 0 && outside_count > 0
            }
            (Geometry::Polygon(poly), Geometry::LineString(ls)) => Self::crosses_impl(
                &Geometry::LineString(ls.clone()),
                &Geometry::Polygon(poly.clone()),
            ),
            _ => false,
        }
    }
}

// ── Private free functions ────────────────────────────────────────────────────

/// Ray-casting algorithm: returns `true` when `p` is inside `ring`.
fn ray_cast(p: &Point, ring: &[Point]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let (mut xi, mut yi) = (ring[0].x, ring[0].y);
    for j in 1..=n {
        let (xj, yj) = (ring[j % n].x, ring[j % n].y);
        if ((yi > p.y) != (yj > p.y)) && (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        (xi, yi) = (xj, yj);
    }
    inside
}

fn point_distance(a: &Point, b: &Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn point_to_bbox_distance(p: &Point, bb: &BoundingBox) -> f64 {
    let dx = if p.x < bb.min_x {
        bb.min_x - p.x
    } else if p.x > bb.max_x {
        p.x - bb.max_x
    } else {
        0.0
    };
    let dy = if p.y < bb.min_y {
        bb.min_y - p.y
    } else if p.y > bb.max_y {
        p.y - bb.max_y
    } else {
        0.0
    };
    (dx * dx + dy * dy).sqrt()
}

fn bbox_to_bbox_distance(a: &BoundingBox, b: &BoundingBox) -> f64 {
    let dx = if a.max_x < b.min_x {
        b.min_x - a.max_x
    } else if b.max_x < a.min_x {
        a.min_x - b.max_x
    } else {
        0.0
    };
    let dy = if a.max_y < b.min_y {
        b.min_y - a.max_y
    } else if b.max_y < a.min_y {
        a.min_y - b.max_y
    } else {
        0.0
    };
    (dx * dx + dy * dy).sqrt()
}

/// Returns the four corners of a bounding box.
fn bbox_corners(bb: &BoundingBox) -> [Point; 4] {
    [
        Point {
            x: bb.min_x,
            y: bb.min_y,
        },
        Point {
            x: bb.max_x,
            y: bb.min_y,
        },
        Point {
            x: bb.max_x,
            y: bb.max_y,
        },
        Point {
            x: bb.min_x,
            y: bb.max_y,
        },
    ]
}

/// Bounding boxes touch (share boundary but do not overlap interiors).
fn bbox_touches(a: &BoundingBox, b: &BoundingBox) -> bool {
    // They touch if they share exactly one edge or corner but do not overlap
    let shared_x =
        (a.max_x - b.min_x).abs() < f64::EPSILON || (b.max_x - a.min_x).abs() < f64::EPSILON;
    let shared_y =
        (a.max_y - b.min_y).abs() < f64::EPSILON || (b.max_y - a.min_y).abs() < f64::EPSILON;
    // x-intervals overlap or touch
    let x_overlap = a.max_x >= b.min_x && b.max_x >= a.min_x;
    let y_overlap = a.max_y >= b.min_y && b.max_y >= a.min_y;
    // They do not penetrate (no interior overlap)
    let no_interior = !TopologyChecker::bbox_intersects(a, b);
    no_interior && x_overlap && y_overlap && (shared_x || shared_y)
}

/// Returns `true` when `p` lies exactly on the boundary of `bbox`.
fn point_on_bbox_boundary(p: &Point, bb: &BoundingBox) -> bool {
    if !TopologyChecker::point_in_bbox(p, bb) {
        return false;
    }
    let on_x = (p.x - bb.min_x).abs() < f64::EPSILON || (p.x - bb.max_x).abs() < f64::EPSILON;
    let on_y = (p.y - bb.min_y).abs() < f64::EPSILON || (p.y - bb.max_y).abs() < f64::EPSILON;
    on_x || on_y
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    fn bbox(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> BoundingBox {
        BoundingBox::new(min_x, min_y, max_x, max_y)
    }

    fn square_poly(cx: f64, cy: f64, half: f64) -> Polygon {
        Polygon {
            exterior: vec![
                pt(cx - half, cy - half),
                pt(cx + half, cy - half),
                pt(cx + half, cy + half),
                pt(cx - half, cy + half),
            ],
            holes: vec![],
        }
    }

    // ── bbox_intersects ───────────────────────────────────────────────────────

    #[test]
    fn test_bbox_intersects_overlapping() {
        let a = bbox(0.0, 0.0, 2.0, 2.0);
        let b = bbox(1.0, 1.0, 3.0, 3.0);
        assert!(TopologyChecker::bbox_intersects(&a, &b));
    }

    #[test]
    fn test_bbox_intersects_disjoint_x() {
        let a = bbox(0.0, 0.0, 1.0, 1.0);
        let b = bbox(2.0, 0.0, 3.0, 1.0);
        assert!(!TopologyChecker::bbox_intersects(&a, &b));
    }

    #[test]
    fn test_bbox_intersects_disjoint_y() {
        let a = bbox(0.0, 0.0, 1.0, 1.0);
        let b = bbox(0.0, 2.0, 1.0, 3.0);
        assert!(!TopologyChecker::bbox_intersects(&a, &b));
    }

    #[test]
    fn test_bbox_intersects_touching_edge_not_intersecting() {
        // Touching at x=1.0 edge: a.max_x == b.min_x → not a strict intersection
        let a = bbox(0.0, 0.0, 1.0, 1.0);
        let b = bbox(1.0, 0.0, 2.0, 1.0);
        assert!(!TopologyChecker::bbox_intersects(&a, &b));
    }

    #[test]
    fn test_bbox_intersects_contained() {
        let outer = bbox(0.0, 0.0, 10.0, 10.0);
        let inner = bbox(2.0, 2.0, 5.0, 5.0);
        assert!(TopologyChecker::bbox_intersects(&outer, &inner));
    }

    // ── point_in_bbox ─────────────────────────────────────────────────────────

    #[test]
    fn test_point_in_bbox_inside() {
        let bb = bbox(0.0, 0.0, 10.0, 10.0);
        assert!(TopologyChecker::point_in_bbox(&pt(5.0, 5.0), &bb));
    }

    #[test]
    fn test_point_in_bbox_on_boundary() {
        let bb = bbox(0.0, 0.0, 10.0, 10.0);
        assert!(TopologyChecker::point_in_bbox(&pt(0.0, 5.0), &bb));
    }

    #[test]
    fn test_point_in_bbox_outside() {
        let bb = bbox(0.0, 0.0, 5.0, 5.0);
        assert!(!TopologyChecker::point_in_bbox(&pt(6.0, 3.0), &bb));
    }

    // ── point_in_polygon ──────────────────────────────────────────────────────

    #[test]
    fn test_point_in_polygon_inside() {
        let poly = square_poly(5.0, 5.0, 3.0);
        assert!(TopologyChecker::point_in_polygon(&pt(5.0, 5.0), &poly));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let poly = square_poly(5.0, 5.0, 2.0);
        assert!(!TopologyChecker::point_in_polygon(&pt(10.0, 10.0), &poly));
    }

    #[test]
    fn test_point_in_polygon_hole_exclusion() {
        // Outer square minus inner square hole
        let mut poly = square_poly(5.0, 5.0, 4.0);
        let hole: Vec<Point> = vec![pt(4.0, 4.0), pt(6.0, 4.0), pt(6.0, 6.0), pt(4.0, 6.0)];
        poly.holes.push(hole);
        // Center is in the hole → not inside polygon
        assert!(!TopologyChecker::point_in_polygon(&pt(5.0, 5.0), &poly));
        // Outside the hole but inside outer
        assert!(TopologyChecker::point_in_polygon(&pt(8.0, 5.0), &poly));
    }

    // ── contains ──────────────────────────────────────────────────────────────

    #[test]
    fn test_bbox_contains_point() {
        let bb = Geometry::BoundingBox(bbox(0.0, 0.0, 10.0, 10.0));
        let p = Geometry::Point(pt(5.0, 5.0));
        assert!(TopologyChecker::contains(&bb, &p));
    }

    #[test]
    fn test_bbox_does_not_contain_outside_point() {
        let bb = Geometry::BoundingBox(bbox(0.0, 0.0, 5.0, 5.0));
        let p = Geometry::Point(pt(6.0, 3.0));
        assert!(!TopologyChecker::contains(&bb, &p));
    }

    #[test]
    fn test_bbox_contains_smaller_bbox() {
        let outer = Geometry::BoundingBox(bbox(0.0, 0.0, 10.0, 10.0));
        let inner = Geometry::BoundingBox(bbox(2.0, 2.0, 5.0, 5.0));
        assert!(TopologyChecker::contains(&outer, &inner));
    }

    #[test]
    fn test_bbox_does_not_contain_larger_bbox() {
        let small = Geometry::BoundingBox(bbox(2.0, 2.0, 5.0, 5.0));
        let large = Geometry::BoundingBox(bbox(0.0, 0.0, 10.0, 10.0));
        assert!(!TopologyChecker::contains(&small, &large));
    }

    // ── disjoint ──────────────────────────────────────────────────────────────

    #[test]
    fn test_disjoint_non_overlapping_bboxes() {
        let a = Geometry::BoundingBox(bbox(0.0, 0.0, 1.0, 1.0));
        let b = Geometry::BoundingBox(bbox(5.0, 5.0, 6.0, 6.0));
        assert!(TopologyChecker::disjoint(&a, &b));
    }

    #[test]
    fn test_not_disjoint_overlapping_bboxes() {
        let a = Geometry::BoundingBox(bbox(0.0, 0.0, 3.0, 3.0));
        let b = Geometry::BoundingBox(bbox(2.0, 2.0, 5.0, 5.0));
        assert!(!TopologyChecker::disjoint(&a, &b));
    }

    // ── BoundingBox methods ───────────────────────────────────────────────────

    #[test]
    fn test_bbox_area() {
        let bb = bbox(0.0, 0.0, 4.0, 5.0);
        assert!((bb.area() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bbox_area_zero() {
        let bb = bbox(1.0, 1.0, 1.0, 1.0);
        assert_eq!(bb.area(), 0.0);
    }

    #[test]
    fn test_bbox_center() {
        let bb = bbox(0.0, 0.0, 4.0, 2.0);
        let c = bb.center();
        assert!((c.x - 2.0).abs() < f64::EPSILON);
        assert!((c.y - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bbox_from_polygon() {
        let poly = square_poly(5.0, 5.0, 3.0);
        let bb = BoundingBox::from_polygon(&poly);
        assert!((bb.min_x - 2.0).abs() < f64::EPSILON);
        assert!((bb.min_y - 2.0).abs() < f64::EPSILON);
        assert!((bb.max_x - 8.0).abs() < f64::EPSILON);
        assert!((bb.max_y - 8.0).abs() < f64::EPSILON);
    }

    // ── distance ──────────────────────────────────────────────────────────────

    #[test]
    fn test_distance_point_to_point() {
        let a = Geometry::Point(pt(0.0, 0.0));
        let b = Geometry::Point(pt(3.0, 4.0));
        let d = TopologyChecker::distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_same_point_zero() {
        let a = Geometry::Point(pt(1.0, 1.0));
        let b = Geometry::Point(pt(1.0, 1.0));
        assert_eq!(TopologyChecker::distance(&a, &b), 0.0);
    }

    #[test]
    fn test_distance_point_outside_bbox() {
        let p = Geometry::Point(pt(0.0, 0.0));
        let bb = Geometry::BoundingBox(bbox(3.0, 4.0, 5.0, 6.0));
        let d = TopologyChecker::distance(&p, &bb);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_point_inside_bbox_is_zero() {
        let p = Geometry::Point(pt(5.0, 5.0));
        let bb = Geometry::BoundingBox(bbox(0.0, 0.0, 10.0, 10.0));
        assert_eq!(TopologyChecker::distance(&p, &bb), 0.0);
    }

    #[test]
    fn test_distance_disjoint_bboxes() {
        let a = Geometry::BoundingBox(bbox(0.0, 0.0, 1.0, 1.0));
        let b = Geometry::BoundingBox(bbox(4.0, 0.0, 5.0, 1.0));
        let d = TopologyChecker::distance(&a, &b);
        assert!((d - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_intersecting_bboxes_is_zero() {
        let a = Geometry::BoundingBox(bbox(0.0, 0.0, 3.0, 3.0));
        let b = Geometry::BoundingBox(bbox(2.0, 2.0, 5.0, 5.0));
        assert_eq!(TopologyChecker::distance(&a, &b), 0.0);
    }

    // ── equals ────────────────────────────────────────────────────────────────

    #[test]
    fn test_equals_same_point() {
        let a = Geometry::Point(pt(1.0, 2.0));
        let b = Geometry::Point(pt(1.0, 2.0));
        assert!(TopologyChecker::equals(&a, &b));
    }

    #[test]
    fn test_equals_different_point() {
        let a = Geometry::Point(pt(1.0, 2.0));
        let b = Geometry::Point(pt(1.0, 3.0));
        assert!(!TopologyChecker::equals(&a, &b));
    }

    #[test]
    fn test_equals_same_bbox() {
        let a = Geometry::BoundingBox(bbox(0.0, 0.0, 5.0, 5.0));
        let b = Geometry::BoundingBox(bbox(0.0, 0.0, 5.0, 5.0));
        assert!(TopologyChecker::equals(&a, &b));
    }

    // ── touches ───────────────────────────────────────────────────────────────

    #[test]
    fn test_touches_adjacent_bboxes() {
        let a = bbox(0.0, 0.0, 1.0, 1.0);
        let b = bbox(1.0, 0.0, 2.0, 1.0);
        assert!(TopologyChecker::touches(
            &Geometry::BoundingBox(a),
            &Geometry::BoundingBox(b)
        ));
    }

    #[test]
    fn test_touches_point_on_bbox_boundary() {
        let bb = Geometry::BoundingBox(bbox(0.0, 0.0, 5.0, 5.0));
        let p = Geometry::Point(pt(0.0, 3.0)); // on the left edge
        assert!(TopologyChecker::touches(&p, &bb));
    }

    // ── relation consistency ──────────────────────────────────────────────────

    #[test]
    fn test_relation_consistency_disjoint_implies_not_intersects() {
        let a = Geometry::BoundingBox(bbox(0.0, 0.0, 1.0, 1.0));
        let b = Geometry::BoundingBox(bbox(5.0, 5.0, 6.0, 6.0));
        let rel = TopologyChecker::relation(&a, &b);
        assert!(rel.disjoint);
        assert!(!rel.intersects);
    }

    #[test]
    fn test_relation_equals_implies_contains_and_within() {
        let a = Geometry::BoundingBox(bbox(0.0, 0.0, 5.0, 5.0));
        let b = Geometry::BoundingBox(bbox(0.0, 0.0, 5.0, 5.0));
        let rel = TopologyChecker::relation(&a, &b);
        assert!(rel.equals);
        assert!(rel.contains);
        assert!(rel.within);
        assert!(rel.intersects);
        assert!(!rel.disjoint);
    }
}
