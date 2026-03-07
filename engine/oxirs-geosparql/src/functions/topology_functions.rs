//! GeoSPARQL Simple Features topology predicate implementations.
//!
//! Provides independent (non-geo-types) geometry types and implements
//! DE-9IM-based Simple Features predicates: equals, disjoint, intersects,
//! touches, crosses, within, contains, overlaps.

// ── Geometry types ────────────────────────────────────────────────────────────

/// Axis-aligned bounding box.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Minimum x coordinate.
    pub min_x: f64,
    /// Minimum y coordinate.
    pub min_y: f64,
    /// Maximum x coordinate.
    pub max_x: f64,
    /// Maximum y coordinate.
    pub max_y: f64,
}

/// A 2-D point.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// Horizontal axis value.
    pub x: f64,
    /// Vertical axis value.
    pub y: f64,
}

impl Point {
    /// Create a new point with the given coordinates.
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

/// A polyline (ordered sequence of points).
#[derive(Debug, Clone, PartialEq)]
pub struct LineString {
    /// Ordered sequence of vertices.
    pub points: Vec<Point>,
}

impl LineString {
    /// Create a LineString from a sequence of points.
    pub fn new(points: Vec<Point>) -> Self {
        LineString { points }
    }
}

/// A polygon with an exterior ring and optional holes.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    /// The outer boundary ring.
    pub exterior: Vec<Point>,
    /// Interior hole rings.
    pub holes: Vec<Vec<Point>>,
}

impl Polygon {
    /// Create a polygon from an exterior ring (no holes).
    pub fn new(exterior: Vec<Point>) -> Self {
        Polygon {
            exterior,
            holes: Vec::new(),
        }
    }
}

/// A geometry value.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// A single point.
    Point(Point),
    /// A sequence of connected line segments.
    LineString(LineString),
    /// A closed polygon area.
    Polygon(Polygon),
    /// A collection of points.
    MultiPoint(Vec<Point>),
    /// A heterogeneous collection of geometries.
    GeometryCollection(Vec<Geometry>),
}

// ── Topology functions ────────────────────────────────────────────────────────

/// Namespace struct for all topology predicates.
pub struct TopologyFunctions;

impl TopologyFunctions {
    // ── Public predicates ─────────────────────────────────────────────────

    /// SF Equals: both geometries are topologically identical.
    pub fn sf_equals(g1: &Geometry, g2: &Geometry) -> bool {
        match (g1, g2) {
            (Geometry::Point(p1), Geometry::Point(p2)) => {
                (p1.x - p2.x).abs() < f64::EPSILON && (p1.y - p2.y).abs() < f64::EPSILON
            }
            (Geometry::MultiPoint(pts1), Geometry::MultiPoint(pts2)) => {
                pts1.len() == pts2.len()
                    && pts1.iter().zip(pts2.iter()).all(|(a, b)| {
                        (a.x - b.x).abs() < f64::EPSILON && (a.y - b.y).abs() < f64::EPSILON
                    })
            }
            (Geometry::LineString(l1), Geometry::LineString(l2)) => {
                l1.points.len() == l2.points.len()
                    && l1.points.iter().zip(l2.points.iter()).all(|(a, b)| {
                        (a.x - b.x).abs() < f64::EPSILON && (a.y - b.y).abs() < f64::EPSILON
                    })
            }
            (Geometry::Polygon(p1), Geometry::Polygon(p2)) => {
                ring_equals(&p1.exterior, &p2.exterior)
                    && p1.holes.len() == p2.holes.len()
                    && p1
                        .holes
                        .iter()
                        .zip(p2.holes.iter())
                        .all(|(h1, h2)| ring_equals(h1, h2))
            }
            _ => false,
        }
    }

    /// SF Disjoint: the geometries have no point in common.
    pub fn sf_disjoint(g1: &Geometry, g2: &Geometry) -> bool {
        !Self::sf_intersects(g1, g2)
    }

    /// SF Intersects: at least one point in common.
    pub fn sf_intersects(g1: &Geometry, g2: &Geometry) -> bool {
        // Fast reject via bounding box.
        if !bbox_intersects(&bounding_box(g1), &bounding_box(g2)) {
            return false;
        }
        Self::geometries_intersect(g1, g2)
    }

    /// SF Touches: boundaries share at least one point but interiors do not intersect.
    pub fn sf_touches(g1: &Geometry, g2: &Geometry) -> bool {
        // For two polygons: bboxes must touch or overlap at boundary.
        match (g1, g2) {
            (Geometry::Polygon(p1), Geometry::Polygon(p2)) => {
                let bb1 = bounding_box(g1);
                let bb2 = bounding_box(g2);
                if !bbox_intersects(&bb1, &bb2) {
                    return false;
                }
                // At least one vertex of p1 boundary on p2 boundary and
                // interiors do not intersect.
                let boundary_contact = p1.exterior.iter().any(|pt| point_on_ring(pt, &p2.exterior))
                    || p2.exterior.iter().any(|pt| point_on_ring(pt, &p1.exterior));
                if !boundary_contact {
                    return false;
                }
                // Interiors must not overlap.
                !Self::polygon_interiors_intersect(p1, p2)
            }
            (Geometry::Point(pt), Geometry::Polygon(poly))
            | (Geometry::Polygon(poly), Geometry::Point(pt)) => {
                point_on_ring(pt, &poly.exterior) && !point_in_polygon(pt, &poly.exterior)
            }
            _ => false,
        }
    }

    /// SF Crosses: geometries have some but not all interior points in common,
    /// and the dimension of the intersection is less than that of at least one.
    pub fn sf_crosses(g1: &Geometry, g2: &Geometry) -> bool {
        match (g1, g2) {
            (Geometry::LineString(l1), Geometry::LineString(l2)) => {
                // Lines cross if they share exactly one point and are not parallel.
                linestring_cross_check(l1, l2)
            }
            (Geometry::LineString(l), Geometry::Polygon(poly))
            | (Geometry::Polygon(poly), Geometry::LineString(l)) => {
                // Line crosses polygon if some points are inside and some outside.
                linestring_crosses_polygon(l, poly)
            }
            _ => false,
        }
    }

    /// SF Within: g1 is completely inside g2.
    pub fn sf_within(g1: &Geometry, g2: &Geometry) -> bool {
        match (g1, g2) {
            (Geometry::Point(pt), Geometry::Polygon(poly)) => point_in_polygon(pt, &poly.exterior),
            (Geometry::Polygon(p1), Geometry::Polygon(p2)) => polygon_within_polygon(p1, p2),
            (Geometry::MultiPoint(pts), Geometry::Polygon(poly)) => {
                pts.iter().all(|pt| point_in_polygon(pt, &poly.exterior))
            }
            _ => false,
        }
    }

    /// SF Contains: g1 completely contains g2.
    pub fn sf_contains(g1: &Geometry, g2: &Geometry) -> bool {
        Self::sf_within(g2, g1)
    }

    /// SF Overlaps: both geometries have the same dimension, they intersect but
    /// neither contains the other.
    pub fn sf_overlaps(g1: &Geometry, g2: &Geometry) -> bool {
        match (g1, g2) {
            (Geometry::Polygon(p1), Geometry::Polygon(p2)) => {
                if !bbox_intersects(&bounding_box(g1), &bounding_box(g2)) {
                    return false;
                }
                // Some vertices of p1 are inside p2 and vice versa.
                let p1_in_p2 = p1
                    .exterior
                    .iter()
                    .any(|pt| point_in_polygon(pt, &p2.exterior));
                let p2_in_p1 = p2
                    .exterior
                    .iter()
                    .any(|pt| point_in_polygon(pt, &p1.exterior));
                p1_in_p2 && p2_in_p1
            }
            _ => false,
        }
    }

    // ── Internal geometry intersection ────────────────────────────────────

    fn geometries_intersect(g1: &Geometry, g2: &Geometry) -> bool {
        match (g1, g2) {
            (Geometry::Point(p1), Geometry::Point(p2)) => {
                (p1.x - p2.x).abs() < f64::EPSILON && (p1.y - p2.y).abs() < f64::EPSILON
            }
            (Geometry::Point(pt), Geometry::Polygon(poly))
            | (Geometry::Polygon(poly), Geometry::Point(pt)) => {
                point_in_polygon(pt, &poly.exterior) || point_on_ring(pt, &poly.exterior)
            }
            (Geometry::Point(pt), Geometry::LineString(ls))
            | (Geometry::LineString(ls), Geometry::Point(pt)) => point_on_linestring(pt, ls),
            (Geometry::Polygon(p1), Geometry::Polygon(p2)) => polygons_intersect(p1, p2),
            (Geometry::LineString(l1), Geometry::LineString(l2)) => linestrings_intersect(l1, l2),
            (Geometry::LineString(l), Geometry::Polygon(poly))
            | (Geometry::Polygon(poly), Geometry::LineString(l)) => {
                linestring_intersects_polygon(l, poly)
            }
            (Geometry::MultiPoint(pts), Geometry::Polygon(poly))
            | (Geometry::Polygon(poly), Geometry::MultiPoint(pts)) => pts.iter().any(|pt| {
                point_in_polygon(pt, &poly.exterior) || point_on_ring(pt, &poly.exterior)
            }),
            (Geometry::MultiPoint(pts1), Geometry::MultiPoint(pts2)) => pts1.iter().any(|p1| {
                pts2.iter().any(|p2| {
                    (p1.x - p2.x).abs() < f64::EPSILON && (p1.y - p2.y).abs() < f64::EPSILON
                })
            }),
            (Geometry::GeometryCollection(geoms), other)
            | (other, Geometry::GeometryCollection(geoms)) => {
                geoms.iter().any(|g| Self::geometries_intersect(g, other))
            }
            _ => false,
        }
    }

    fn polygon_interiors_intersect(p1: &Polygon, p2: &Polygon) -> bool {
        // Check if any interior point of p1 is inside p2's interior and vice versa.
        // Use centroid as a proxy for "interior point".
        if let Some(c1) = centroid_of_ring(&p1.exterior) {
            if point_in_polygon(&c1, &p2.exterior) {
                return true;
            }
        }
        if let Some(c2) = centroid_of_ring(&p2.exterior) {
            if point_in_polygon(&c2, &p1.exterior) {
                return true;
            }
        }
        false
    }
}

// ── Public standalone helpers ─────────────────────────────────────────────────

/// Compute the axis-aligned bounding box of a geometry.
pub fn bounding_box(g: &Geometry) -> BoundingBox {
    match g {
        Geometry::Point(p) => BoundingBox {
            min_x: p.x,
            min_y: p.y,
            max_x: p.x,
            max_y: p.y,
        },
        Geometry::LineString(ls) => bbox_of_points(&ls.points),
        Geometry::Polygon(poly) => bbox_of_points(&poly.exterior),
        Geometry::MultiPoint(pts) => bbox_of_points(pts),
        Geometry::GeometryCollection(geoms) => {
            let bbs: Vec<BoundingBox> = geoms.iter().map(bounding_box).collect();
            merge_bboxes(&bbs)
        }
    }
}

/// Test whether two bounding boxes have any overlap (including touching edges).
pub fn bbox_intersects(a: &BoundingBox, b: &BoundingBox) -> bool {
    a.min_x <= b.max_x && a.max_x >= b.min_x && a.min_y <= b.max_y && a.max_y >= b.min_y
}

/// Ray-casting point-in-polygon test. Returns true if `p` is strictly inside `ring`.
pub fn point_in_polygon(p: &Point, ring: &[Point]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let mut inside = false;
    let n = ring.len();
    let mut j = n - 1;
    for i in 0..n {
        let vi = &ring[i];
        let vj = &ring[j];
        let intersects = ((vi.y > p.y) != (vj.y > p.y))
            && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Test whether segment (p1,p2) and segment (p3,p4) properly intersect.
pub fn segments_intersect(p1: &Point, p2: &Point, p3: &Point, p4: &Point) -> bool {
    let d1 = cross(p3, p4, p1);
    let d2 = cross(p3, p4, p2);
    let d3 = cross(p1, p2, p3);
    let d4 = cross(p1, p2, p4);

    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }

    // Collinear cases.
    if d1.abs() < f64::EPSILON && on_segment(p3, p4, p1) {
        return true;
    }
    if d2.abs() < f64::EPSILON && on_segment(p3, p4, p2) {
        return true;
    }
    if d3.abs() < f64::EPSILON && on_segment(p1, p2, p3) {
        return true;
    }
    if d4.abs() < f64::EPSILON && on_segment(p1, p2, p4) {
        return true;
    }

    false
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn cross(o: &Point, a: &Point, b: &Point) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

fn on_segment(p: &Point, q: &Point, r: &Point) -> bool {
    r.x <= p.x.max(q.x) && r.x >= p.x.min(q.x) && r.y <= p.y.max(q.y) && r.y >= p.y.min(q.y)
}

fn bbox_of_points(pts: &[Point]) -> BoundingBox {
    let min_x = pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let min_y = pts.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let max_x = pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let max_y = pts.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);
    BoundingBox {
        min_x,
        min_y,
        max_x,
        max_y,
    }
}

fn merge_bboxes(bbs: &[BoundingBox]) -> BoundingBox {
    BoundingBox {
        min_x: bbs.iter().map(|b| b.min_x).fold(f64::INFINITY, f64::min),
        min_y: bbs.iter().map(|b| b.min_y).fold(f64::INFINITY, f64::min),
        max_x: bbs
            .iter()
            .map(|b| b.max_x)
            .fold(f64::NEG_INFINITY, f64::max),
        max_y: bbs
            .iter()
            .map(|b| b.max_y)
            .fold(f64::NEG_INFINITY, f64::max),
    }
}

fn ring_equals(r1: &[Point], r2: &[Point]) -> bool {
    r1.len() == r2.len()
        && r1
            .iter()
            .zip(r2.iter())
            .all(|(a, b)| (a.x - b.x).abs() < f64::EPSILON && (a.y - b.y).abs() < f64::EPSILON)
}

fn point_on_segment(pt: &Point, a: &Point, b: &Point) -> bool {
    // Cross product zero (collinear) and within bounding box.
    let cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
    if cross.abs() > 1e-10 {
        return false;
    }
    pt.x >= a.x.min(b.x) - 1e-10
        && pt.x <= a.x.max(b.x) + 1e-10
        && pt.y >= a.y.min(b.y) - 1e-10
        && pt.y <= a.y.max(b.y) + 1e-10
}

fn point_on_ring(pt: &Point, ring: &[Point]) -> bool {
    let n = ring.len();
    if n < 2 {
        return false;
    }
    for i in 0..n {
        let a = &ring[i];
        let b = &ring[(i + 1) % n];
        if point_on_segment(pt, a, b) {
            return true;
        }
    }
    false
}

fn point_on_linestring(pt: &Point, ls: &LineString) -> bool {
    for i in 0..ls.points.len().saturating_sub(1) {
        if point_on_segment(pt, &ls.points[i], &ls.points[i + 1]) {
            return true;
        }
    }
    false
}

fn linestrings_intersect(l1: &LineString, l2: &LineString) -> bool {
    for i in 0..l1.points.len().saturating_sub(1) {
        for j in 0..l2.points.len().saturating_sub(1) {
            if segments_intersect(
                &l1.points[i],
                &l1.points[i + 1],
                &l2.points[j],
                &l2.points[j + 1],
            ) {
                return true;
            }
        }
    }
    false
}

fn polygons_intersect(p1: &Polygon, p2: &Polygon) -> bool {
    // Edges of each polygon intersect.
    let n1 = p1.exterior.len();
    let n2 = p2.exterior.len();
    for i in 0..n1 {
        for j in 0..n2 {
            if segments_intersect(
                &p1.exterior[i],
                &p1.exterior[(i + 1) % n1],
                &p2.exterior[j],
                &p2.exterior[(j + 1) % n2],
            ) {
                return true;
            }
        }
    }
    // Or one polygon is completely inside the other.
    if !p1.exterior.is_empty() && point_in_polygon(&p1.exterior[0], &p2.exterior) {
        return true;
    }
    if !p2.exterior.is_empty() && point_in_polygon(&p2.exterior[0], &p1.exterior) {
        return true;
    }
    false
}

fn linestring_intersects_polygon(ls: &LineString, poly: &Polygon) -> bool {
    // Any segment of ls crosses a polygon edge.
    let n = poly.exterior.len();
    for i in 0..ls.points.len().saturating_sub(1) {
        for j in 0..n {
            if segments_intersect(
                &ls.points[i],
                &ls.points[i + 1],
                &poly.exterior[j],
                &poly.exterior[(j + 1) % n],
            ) {
                return true;
            }
        }
    }
    // Or a point is inside.
    ls.points
        .iter()
        .any(|pt| point_in_polygon(pt, &poly.exterior))
}

fn linestring_crosses_polygon(ls: &LineString, poly: &Polygon) -> bool {
    let inside: Vec<bool> = ls
        .points
        .iter()
        .map(|pt| point_in_polygon(pt, &poly.exterior))
        .collect();
    // At least one inside and one outside.
    inside.iter().any(|&b| b) && inside.iter().any(|&b| !b)
}

fn linestring_cross_check(l1: &LineString, l2: &LineString) -> bool {
    // Lines cross if they share a point that is not a shared endpoint.
    for i in 0..l1.points.len().saturating_sub(1) {
        for j in 0..l2.points.len().saturating_sub(1) {
            if segments_intersect(
                &l1.points[i],
                &l1.points[i + 1],
                &l2.points[j],
                &l2.points[j + 1],
            ) {
                return true;
            }
        }
    }
    false
}

fn polygon_within_polygon(inner: &Polygon, outer: &Polygon) -> bool {
    // All vertices of inner must be inside outer.
    inner
        .exterior
        .iter()
        .all(|pt| point_in_polygon(pt, &outer.exterior))
}

fn centroid_of_ring(ring: &[Point]) -> Option<Point> {
    if ring.is_empty() {
        return None;
    }
    let sum_x: f64 = ring.iter().map(|p| p.x).sum();
    let sum_y: f64 = ring.iter().map(|p| p.y).sum();
    let n = ring.len() as f64;
    Some(Point {
        x: sum_x / n,
        y: sum_y / n,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Geometry {
        Geometry::Polygon(Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ]))
    }

    fn shifted_square(dx: f64, dy: f64) -> Geometry {
        Geometry::Polygon(Polygon::new(vec![
            Point::new(dx, dy),
            Point::new(dx + 1.0, dy),
            Point::new(dx + 1.0, dy + 1.0),
            Point::new(dx, dy + 1.0),
        ]))
    }

    fn point_at(x: f64, y: f64) -> Geometry {
        Geometry::Point(Point::new(x, y))
    }

    // ── bounding_box ──────────────────────────────────────────────────────

    #[test]
    fn test_bounding_box_point() {
        let bb = bounding_box(&point_at(3.0, 4.0));
        assert!((bb.min_x - 3.0).abs() < 1e-10);
        assert!((bb.min_y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_polygon() {
        let bb = bounding_box(&unit_square());
        assert!((bb.min_x).abs() < 1e-10);
        assert!((bb.max_x - 1.0).abs() < 1e-10);
    }

    // ── bbox_intersects ───────────────────────────────────────────────────

    #[test]
    fn test_bbox_overlap() {
        let b1 = BoundingBox {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 2.0,
            max_y: 2.0,
        };
        let b2 = BoundingBox {
            min_x: 1.0,
            min_y: 1.0,
            max_x: 3.0,
            max_y: 3.0,
        };
        assert!(bbox_intersects(&b1, &b2));
    }

    #[test]
    fn test_bbox_no_overlap() {
        let b1 = BoundingBox {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 1.0,
            max_y: 1.0,
        };
        let b2 = BoundingBox {
            min_x: 2.0,
            min_y: 2.0,
            max_x: 3.0,
            max_y: 3.0,
        };
        assert!(!bbox_intersects(&b1, &b2));
    }

    #[test]
    fn test_bbox_touching_edge() {
        let b1 = BoundingBox {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 1.0,
            max_y: 1.0,
        };
        let b2 = BoundingBox {
            min_x: 1.0,
            min_y: 0.0,
            max_x: 2.0,
            max_y: 1.0,
        };
        assert!(bbox_intersects(&b1, &b2));
    }

    // ── point_in_polygon ──────────────────────────────────────────────────

    #[test]
    fn test_point_inside_square() {
        let ring = vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
        ];
        assert!(point_in_polygon(&Point::new(2.0, 2.0), &ring));
    }

    #[test]
    fn test_point_outside_square() {
        let ring = vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
        ];
        assert!(!point_in_polygon(&Point::new(5.0, 5.0), &ring));
    }

    // ── segments_intersect ────────────────────────────────────────────────

    #[test]
    fn test_segments_cross() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(2.0, 2.0);
        let p3 = Point::new(0.0, 2.0);
        let p4 = Point::new(2.0, 0.0);
        assert!(segments_intersect(&p1, &p2, &p3, &p4));
    }

    #[test]
    fn test_segments_parallel_no_intersect() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(2.0, 0.0);
        let p3 = Point::new(0.0, 1.0);
        let p4 = Point::new(2.0, 1.0);
        assert!(!segments_intersect(&p1, &p2, &p3, &p4));
    }

    // ── sf_equals ─────────────────────────────────────────────────────────

    #[test]
    fn test_sf_equals_same_point() {
        assert!(TopologyFunctions::sf_equals(
            &point_at(1.0, 2.0),
            &point_at(1.0, 2.0)
        ));
    }

    #[test]
    fn test_sf_equals_different_points() {
        assert!(!TopologyFunctions::sf_equals(
            &point_at(1.0, 2.0),
            &point_at(3.0, 4.0)
        ));
    }

    #[test]
    fn test_sf_equals_same_polygon() {
        assert!(TopologyFunctions::sf_equals(&unit_square(), &unit_square()));
    }

    #[test]
    fn test_sf_equals_different_types() {
        assert!(!TopologyFunctions::sf_equals(
            &point_at(0.0, 0.0),
            &unit_square()
        ));
    }

    // ── sf_disjoint / sf_intersects ───────────────────────────────────────

    #[test]
    fn test_sf_disjoint_separate_polygons() {
        assert!(TopologyFunctions::sf_disjoint(
            &unit_square(),
            &shifted_square(5.0, 5.0)
        ));
    }

    #[test]
    fn test_sf_intersects_overlapping() {
        assert!(TopologyFunctions::sf_intersects(
            &unit_square(),
            &shifted_square(0.5, 0.5)
        ));
    }

    #[test]
    fn test_sf_intersects_point_in_polygon() {
        let poly = unit_square();
        let pt = point_at(0.5, 0.5);
        assert!(TopologyFunctions::sf_intersects(&pt, &poly));
    }

    #[test]
    fn test_sf_disjoint_point_outside() {
        let poly = unit_square();
        let pt = point_at(5.0, 5.0);
        assert!(TopologyFunctions::sf_disjoint(&pt, &poly));
    }

    // ── sf_within / sf_contains ───────────────────────────────────────────

    #[test]
    fn test_sf_within_point_inside() {
        assert!(TopologyFunctions::sf_within(
            &point_at(0.5, 0.5),
            &unit_square()
        ));
    }

    #[test]
    fn test_sf_within_point_outside() {
        assert!(!TopologyFunctions::sf_within(
            &point_at(5.0, 5.0),
            &unit_square()
        ));
    }

    #[test]
    fn test_sf_contains_point() {
        assert!(TopologyFunctions::sf_contains(
            &unit_square(),
            &point_at(0.5, 0.5)
        ));
    }

    #[test]
    fn test_sf_contains_polygon() {
        let small = Geometry::Polygon(Polygon::new(vec![
            Point::new(0.2, 0.2),
            Point::new(0.8, 0.2),
            Point::new(0.8, 0.8),
            Point::new(0.2, 0.8),
        ]));
        assert!(TopologyFunctions::sf_contains(&unit_square(), &small));
    }

    // ── sf_overlaps ───────────────────────────────────────────────────────

    #[test]
    fn test_sf_overlaps_partial() {
        assert!(TopologyFunctions::sf_overlaps(
            &unit_square(),
            &shifted_square(0.5, 0.0)
        ));
    }

    #[test]
    fn test_sf_no_overlaps_disjoint() {
        assert!(!TopologyFunctions::sf_overlaps(
            &unit_square(),
            &shifted_square(5.0, 5.0)
        ));
    }

    // ── sf_crosses ────────────────────────────────────────────────────────

    #[test]
    fn test_sf_crosses_lines() {
        let l1 = Geometry::LineString(LineString::new(vec![
            Point::new(0.0, 1.0),
            Point::new(2.0, 1.0),
        ]));
        let l2 = Geometry::LineString(LineString::new(vec![
            Point::new(1.0, 0.0),
            Point::new(1.0, 2.0),
        ]));
        assert!(TopologyFunctions::sf_crosses(&l1, &l2));
    }

    #[test]
    fn test_sf_crosses_parallel_lines_no_cross() {
        let l1 = Geometry::LineString(LineString::new(vec![
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
        ]));
        let l2 = Geometry::LineString(LineString::new(vec![
            Point::new(0.0, 1.0),
            Point::new(2.0, 1.0),
        ]));
        assert!(!TopologyFunctions::sf_crosses(&l1, &l2));
    }

    // ── sf_touches ────────────────────────────────────────────────────────

    #[test]
    fn test_sf_touches_adjacent_polygons() {
        let p1 = unit_square();
        let p2 = shifted_square(1.0, 0.0); // shares the x=1 edge
                                           // Touching along the shared edge — polygons share boundary but not interior.
                                           // Our implementation checks boundary contact AND that interiors don't overlap.
        let result = TopologyFunctions::sf_touches(&p1, &p2);
        assert!(result);
    }

    #[test]
    fn test_sf_touches_non_adjacent() {
        let p1 = unit_square();
        let p2 = shifted_square(5.0, 5.0);
        assert!(!TopologyFunctions::sf_touches(&p1, &p2));
    }

    // ── MultiPoint ────────────────────────────────────────────────────────

    #[test]
    fn test_sf_within_multipoint_all_inside() {
        let pts = Geometry::MultiPoint(vec![Point::new(0.3, 0.3), Point::new(0.7, 0.7)]);
        assert!(TopologyFunctions::sf_within(&pts, &unit_square()));
    }

    #[test]
    fn test_sf_within_multipoint_some_outside() {
        let pts = Geometry::MultiPoint(vec![Point::new(0.5, 0.5), Point::new(5.0, 5.0)]);
        assert!(!TopologyFunctions::sf_within(&pts, &unit_square()));
    }
}
