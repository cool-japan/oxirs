//! 2D convex hull computation using the Graham scan algorithm.
//!
//! Provides `ConvexHullBuilder` to collect points and compute the convex hull
//! with area (shoelace formula), perimeter (Euclidean), and centroid (vertex
//! average). Supports point-in-hull and containment checks.

// ── Point ─────────────────────────────────────────────────────────────────────

/// A 2-D point with double-precision coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Point2D {
    /// Horizontal coordinate.
    pub x: f64,
    /// Vertical coordinate.
    pub y: f64,
}

impl Point2D {
    /// Construct a new point.
    pub fn new(x: f64, y: f64) -> Self {
        Point2D { x, y }
    }

    /// Squared Euclidean distance to another point.
    fn dist_sq(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Euclidean distance to another point.
    pub fn distance(&self, other: &Point2D) -> f64 {
        self.dist_sq(other).sqrt()
    }
}

// ── Hull result ───────────────────────────────────────────────────────────────

/// Result of a convex hull computation.
#[derive(Debug, Clone)]
pub struct HullResult {
    /// Vertices of the convex hull in counter-clockwise order.
    pub hull: Vec<Point2D>,
    /// Area of the convex polygon (shoelace formula).
    pub area: f64,
    /// Perimeter of the convex polygon (sum of edge lengths).
    pub perimeter: f64,
    /// Centroid of the hull vertices (arithmetic mean of vertex coordinates).
    pub centroid: Point2D,
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors returned by hull computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HullError {
    /// Fewer than 3 distinct points were provided.
    InsufficientPoints(usize),
    /// All provided points are collinear (degenerate polygon).
    CollinearPoints,
    /// No points were provided at all.
    EmptyInput,
}

impl std::fmt::Display for HullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HullError::InsufficientPoints(n) => {
                write!(f, "need at least 3 points to form a hull, got {n}")
            }
            HullError::CollinearPoints => {
                write!(f, "all points are collinear — convex hull is degenerate")
            }
            HullError::EmptyInput => write!(f, "no points provided"),
        }
    }
}

impl std::error::Error for HullError {}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Accumulates 2-D points and computes their convex hull on demand.
pub struct ConvexHullBuilder {
    points: Vec<Point2D>,
}

impl ConvexHullBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        ConvexHullBuilder { points: Vec::new() }
    }

    /// Add a single point.
    pub fn add_point(&mut self, p: Point2D) {
        self.points.push(p);
    }

    /// Add multiple points.
    pub fn add_points(&mut self, pts: Vec<Point2D>) {
        self.points.extend(pts);
    }

    /// Compute the convex hull of the accumulated points.
    ///
    /// # Errors
    ///
    /// Returns [`HullError::EmptyInput`] when no points have been added,
    /// [`HullError::InsufficientPoints`] when fewer than 3 points exist, or
    /// [`HullError::CollinearPoints`] when all points are collinear.
    pub fn compute(&self) -> Result<HullResult, HullError> {
        if self.points.is_empty() {
            return Err(HullError::EmptyInput);
        }
        if self.points.len() < 3 {
            return Err(HullError::InsufficientPoints(self.points.len()));
        }

        let hull_pts = graham_scan(&self.points)?;

        if hull_pts.len() < 3 {
            return Err(HullError::CollinearPoints);
        }

        let area = shoelace_area(&hull_pts);
        let perimeter = compute_perimeter(&hull_pts);
        let centroid = vertex_centroid(&hull_pts);

        Ok(HullResult {
            hull: hull_pts,
            area,
            perimeter,
            centroid,
        })
    }

    /// Test whether a point `p` lies inside or on the boundary of the given
    /// convex hull (specified as an ordered sequence of vertices).
    ///
    /// Uses the sign-of-cross-product method: for a CCW hull a point is inside
    /// iff it is to the left of (or on) every edge.
    pub fn point_inside_hull(hull: &[Point2D], p: &Point2D) -> bool {
        let n = hull.len();
        if n < 3 {
            return false;
        }
        for i in 0..n {
            let a = &hull[i];
            let b = &hull[(i + 1) % n];
            let cp = cross_product(a, b, p);
            if cp < -1e-10 {
                return false;
            }
        }
        true
    }

    /// Return `true` iff every point in `points` is inside or on the hull.
    pub fn hull_contains_all(hull: &[Point2D], points: &[Point2D]) -> bool {
        points.iter().all(|p| Self::point_inside_hull(hull, p))
    }
}

impl Default for ConvexHullBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Signed cross product of vectors (a→b) and (a→c).
///
/// Positive ⇒ `c` is to the left of line a→b (CCW turn).
/// Negative ⇒ `c` is to the right (CW turn).
/// Zero     ⇒ collinear.
fn cross_product(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Sort points by polar angle around a pivot `origin`, breaking ties by
/// ascending distance.
fn polar_angle_sort(origin: &Point2D, points: &mut [Point2D]) {
    points.sort_by(|a, b| {
        let angle_a = (a.y - origin.y).atan2(a.x - origin.x);
        let angle_b = (b.y - origin.y).atan2(b.x - origin.x);
        match angle_a.partial_cmp(&angle_b) {
            Some(ord) if ord != std::cmp::Ordering::Equal => ord,
            _ => {
                let da = origin.dist_sq(a);
                let db = origin.dist_sq(b);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    });
}

/// Run Graham scan on `points` and return the CCW convex hull vertices.
///
/// # Errors
///
/// Returns [`HullError::CollinearPoints`] if the resulting hull has fewer than 3
/// vertices (i.e., all points are collinear).
fn graham_scan(points: &[Point2D]) -> Result<Vec<Point2D>, HullError> {
    // Find the bottom-most point (lowest y, leftmost x for ties).
    let pivot_idx = lowest_left_index(points);
    let pivot = points[pivot_idx].clone();

    // Collect all other points and sort by polar angle around the pivot.
    let mut rest: Vec<Point2D> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != pivot_idx)
        .map(|(_, p)| p.clone())
        .collect();

    polar_angle_sort(&pivot, &mut rest);

    // Remove collinear intermediate points (keep only the farthest from pivot).
    let rest = remove_collinear_intermediates(&pivot, rest);

    if rest.is_empty() {
        return Err(HullError::CollinearPoints);
    }

    // Build the hull stack.
    let mut stack: Vec<Point2D> = vec![pivot];
    stack.push(rest[0].clone());

    for p in rest.into_iter().skip(1) {
        // Pop while the last three points make a non-left turn.
        while stack.len() >= 2 {
            let n = stack.len();
            let cp = cross_product(&stack[n - 2], &stack[n - 1], &p);
            if cp > 1e-10 {
                // Genuine left turn — keep.
                break;
            }
            stack.pop();
        }
        stack.push(p);
    }

    if stack.len() < 3 {
        return Err(HullError::CollinearPoints);
    }

    Ok(stack)
}

/// Return the index of the lowest (then leftmost) point.
fn lowest_left_index(points: &[Point2D]) -> usize {
    let mut best = 0;
    for i in 1..points.len() {
        let p = &points[i];
        let b = &points[best];
        if p.y < b.y || (p.y == b.y && p.x < b.x) {
            best = i;
        }
    }
    best
}

/// Given points already sorted by polar angle around `pivot`, remove collinear
/// intermediate points so that only the farthest point in each collinear run
/// remains.
fn remove_collinear_intermediates(pivot: &Point2D, sorted: Vec<Point2D>) -> Vec<Point2D> {
    if sorted.is_empty() {
        return sorted;
    }

    let mut result: Vec<Point2D> = Vec::with_capacity(sorted.len());
    let mut i = 0;

    while i < sorted.len() {
        let mut j = i;
        // Advance j while points are collinear with pivot.
        while j + 1 < sorted.len() && cross_product(pivot, &sorted[j], &sorted[j + 1]).abs() < 1e-10
        {
            j += 1;
        }
        // Keep only the farthest point in the collinear run.
        let farthest = (i..=j)
            .max_by(|&a, &b| {
                pivot
                    .dist_sq(&sorted[a])
                    .partial_cmp(&pivot.dist_sq(&sorted[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(i);
        result.push(sorted[farthest].clone());
        i = j + 1;
    }
    result
}

/// Compute the signed area of a polygon via the shoelace formula.
/// Returns the absolute value.
fn shoelace_area(hull: &[Point2D]) -> f64 {
    let n = hull.len();
    let mut sum = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += hull[i].x * hull[j].y;
        sum -= hull[j].x * hull[i].y;
    }
    (sum / 2.0).abs()
}

/// Compute the perimeter of a polygon as the sum of edge lengths.
fn compute_perimeter(hull: &[Point2D]) -> f64 {
    let n = hull.len();
    (0..n).map(|i| hull[i].distance(&hull[(i + 1) % n])).sum()
}

/// Compute the centroid as the arithmetic mean of vertex coordinates.
fn vertex_centroid(hull: &[Point2D]) -> Point2D {
    let n = hull.len() as f64;
    let sum_x: f64 = hull.iter().map(|p| p.x).sum();
    let sum_y: f64 = hull.iter().map(|p| p.y).sum();
    Point2D::new(sum_x / n, sum_y / n)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-8
    }

    // ── Triangle ─────────────────────────────────────────────────────────────

    #[test]
    fn test_triangle_hull_size() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(4.0, 0.0), p(2.0, 3.0)]);
        let r = b.compute().expect("ok");
        assert_eq!(r.hull.len(), 3);
    }

    #[test]
    fn test_triangle_area() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(4.0, 0.0), p(0.0, 4.0)]);
        let r = b.compute().expect("ok");
        assert!(approx_eq(r.area, 8.0), "expected 8, got {}", r.area);
    }

    #[test]
    fn test_triangle_perimeter() {
        let mut b = ConvexHullBuilder::new();
        // Right triangle 3-4-5
        b.add_points(vec![p(0.0, 0.0), p(3.0, 0.0), p(0.0, 4.0)]);
        let r = b.compute().expect("ok");
        assert!(
            approx_eq(r.perimeter, 12.0),
            "expected 12, got {}",
            r.perimeter
        );
    }

    // ── Square ────────────────────────────────────────────────────────────────

    #[test]
    fn test_unit_square_hull_size() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]);
        let r = b.compute().expect("ok");
        assert_eq!(r.hull.len(), 4);
    }

    #[test]
    fn test_unit_square_area() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]);
        let r = b.compute().expect("ok");
        assert!(approx_eq(r.area, 1.0), "area={}", r.area);
    }

    #[test]
    fn test_unit_square_perimeter() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)]);
        let r = b.compute().expect("ok");
        assert!(approx_eq(r.perimeter, 4.0), "perimeter={}", r.perimeter);
    }

    #[test]
    fn test_square_centroid() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(2.0, 0.0), p(2.0, 2.0), p(0.0, 2.0)]);
        let r = b.compute().expect("ok");
        assert!(approx_eq(r.centroid.x, 1.0) && approx_eq(r.centroid.y, 1.0));
    }

    // ── Interior points dropped ───────────────────────────────────────────────

    #[test]
    fn test_interior_points_excluded() {
        let mut b = ConvexHullBuilder::new();
        // Square + interior point
        b.add_points(vec![
            p(0.0, 0.0),
            p(4.0, 0.0),
            p(4.0, 4.0),
            p(0.0, 4.0),
            p(2.0, 2.0), // interior
        ]);
        let r = b.compute().expect("ok");
        assert_eq!(r.hull.len(), 4);
    }

    // ── Convex polygon ─────────────────────────────────────────────────────

    #[test]
    fn test_pentagon_hull_size() {
        use std::f64::consts::PI;
        let mut b = ConvexHullBuilder::new();
        for i in 0..5 {
            let angle = 2.0 * PI * i as f64 / 5.0;
            b.add_point(p(angle.cos(), angle.sin()));
        }
        let r = b.compute().expect("ok");
        assert_eq!(r.hull.len(), 5);
    }

    #[test]
    fn test_hexagon_area_positive() {
        use std::f64::consts::PI;
        let mut b = ConvexHullBuilder::new();
        for i in 0..6 {
            let angle = 2.0 * PI * i as f64 / 6.0;
            b.add_point(p(angle.cos(), angle.sin()));
        }
        let r = b.compute().expect("ok");
        assert!(r.area > 0.0);
    }

    // ── Collinear / degenerate ─────────────────────────────────────────────

    #[test]
    fn test_collinear_three_points() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(1.0, 1.0), p(2.0, 2.0)]);
        let err = b.compute().expect_err("should fail");
        assert_eq!(err, HullError::CollinearPoints);
    }

    #[test]
    fn test_collinear_five_points() {
        let mut b = ConvexHullBuilder::new();
        for i in 0..5 {
            b.add_point(p(i as f64, i as f64));
        }
        let err = b.compute().expect_err("should fail");
        assert_eq!(err, HullError::CollinearPoints);
    }

    // ── Insufficient points ───────────────────────────────────────────────────

    #[test]
    fn test_empty_input_error() {
        let b = ConvexHullBuilder::new();
        let err = b.compute().expect_err("should fail");
        assert_eq!(err, HullError::EmptyInput);
    }

    #[test]
    fn test_single_point_error() {
        let mut b = ConvexHullBuilder::new();
        b.add_point(p(1.0, 2.0));
        let err = b.compute().expect_err("should fail");
        assert_eq!(err, HullError::InsufficientPoints(1));
    }

    #[test]
    fn test_two_points_error() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(1.0, 1.0)]);
        let err = b.compute().expect_err("should fail");
        assert_eq!(err, HullError::InsufficientPoints(2));
    }

    // ── Point-in-hull checks ──────────────────────────────────────────────────

    #[test]
    fn test_point_inside_square_hull() {
        let hull = vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)];
        assert!(ConvexHullBuilder::point_inside_hull(&hull, &p(2.0, 2.0)));
    }

    #[test]
    fn test_point_outside_square_hull() {
        let hull = vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)];
        assert!(!ConvexHullBuilder::point_inside_hull(&hull, &p(5.0, 5.0)));
    }

    #[test]
    fn test_point_on_hull_edge() {
        let hull = vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)];
        // On the bottom edge
        assert!(ConvexHullBuilder::point_inside_hull(&hull, &p(2.0, 0.0)));
    }

    #[test]
    fn test_point_at_hull_vertex() {
        let hull = vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)];
        assert!(ConvexHullBuilder::point_inside_hull(&hull, &p(0.0, 0.0)));
    }

    #[test]
    fn test_point_inside_triangle_hull() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(6.0, 0.0), p(3.0, 6.0)]);
        let r = b.compute().expect("ok");
        assert!(ConvexHullBuilder::point_inside_hull(&r.hull, &p(3.0, 2.0)));
    }

    #[test]
    fn test_point_outside_triangle_hull() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(6.0, 0.0), p(3.0, 6.0)]);
        let r = b.compute().expect("ok");
        assert!(!ConvexHullBuilder::point_inside_hull(&r.hull, &p(0.0, 6.0)));
    }

    // ── hull_contains_all ─────────────────────────────────────────────────────

    #[test]
    fn test_hull_contains_all_true() {
        let hull = vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)];
        let pts = vec![p(1.0, 1.0), p(2.0, 2.0), p(3.0, 3.0)];
        assert!(ConvexHullBuilder::hull_contains_all(&hull, &pts));
    }

    #[test]
    fn test_hull_contains_all_false() {
        let hull = vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)];
        let pts = vec![p(1.0, 1.0), p(5.0, 5.0)];
        assert!(!ConvexHullBuilder::hull_contains_all(&hull, &pts));
    }

    #[test]
    fn test_hull_contains_all_empty_points() {
        let hull = vec![p(0.0, 0.0), p(4.0, 0.0), p(4.0, 4.0), p(0.0, 4.0)];
        assert!(ConvexHullBuilder::hull_contains_all(&hull, &[]));
    }

    // ── add_points helper ─────────────────────────────────────────────────────

    #[test]
    fn test_add_points_batch() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(1.0, 0.0), p(0.5, 1.0)]);
        let r = b.compute().expect("ok");
        assert_eq!(r.hull.len(), 3);
    }

    #[test]
    fn test_add_point_single() {
        let mut b = ConvexHullBuilder::new();
        b.add_point(p(0.0, 0.0));
        b.add_point(p(1.0, 0.0));
        b.add_point(p(0.5, 1.0));
        let r = b.compute().expect("ok");
        assert_eq!(r.hull.len(), 3);
    }

    // ── Hull result fields ────────────────────────────────────────────────────

    #[test]
    fn test_hull_result_area_positive() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(3.0, 0.0), p(3.0, 3.0), p(0.0, 3.0)]);
        let r = b.compute().expect("ok");
        assert!(r.area > 0.0);
    }

    #[test]
    fn test_hull_result_perimeter_positive() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(3.0, 0.0), p(3.0, 3.0), p(0.0, 3.0)]);
        let r = b.compute().expect("ok");
        assert!(r.perimeter > 0.0);
    }

    #[test]
    fn test_cross_product_left_turn() {
        let a = p(0.0, 0.0);
        let b = p(1.0, 0.0);
        let c = p(1.0, 1.0);
        let cp = cross_product(&a, &b, &c);
        assert!(cp > 0.0, "expected positive, got {cp}");
    }

    #[test]
    fn test_cross_product_right_turn() {
        let a = p(0.0, 0.0);
        let b = p(1.0, 0.0);
        let c = p(1.0, -1.0);
        let cp = cross_product(&a, &b, &c);
        assert!(cp < 0.0, "expected negative, got {cp}");
    }

    #[test]
    fn test_cross_product_collinear() {
        let a = p(0.0, 0.0);
        let b = p(1.0, 1.0);
        let c = p(2.0, 2.0);
        let cp = cross_product(&a, &b, &c);
        assert!(cp.abs() < EPS, "expected ~0, got {cp}");
    }

    #[test]
    fn test_hull_error_display_empty() {
        let e = HullError::EmptyInput;
        assert!(e.to_string().contains("no points"));
    }

    #[test]
    fn test_hull_error_display_insufficient() {
        let e = HullError::InsufficientPoints(2);
        assert!(e.to_string().contains("2"));
    }

    #[test]
    fn test_hull_error_display_collinear() {
        let e = HullError::CollinearPoints;
        assert!(e.to_string().contains("collinear"));
    }

    #[test]
    fn test_point2d_distance() {
        let a = p(0.0, 0.0);
        let b = p(3.0, 4.0);
        assert!(approx_eq(a.distance(&b), 5.0));
    }

    #[test]
    fn test_default_constructor() {
        let b = ConvexHullBuilder::default();
        let err = b.compute().expect_err("empty");
        assert_eq!(err, HullError::EmptyInput);
    }

    #[test]
    fn test_large_square_area() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![
            p(0.0, 0.0),
            p(100.0, 0.0),
            p(100.0, 100.0),
            p(0.0, 100.0),
        ]);
        let r = b.compute().expect("ok");
        assert!(approx_eq(r.area, 10000.0));
    }

    #[test]
    fn test_point_inside_hull_too_small_hull() {
        // Hull with < 3 vertices always returns false.
        let hull = vec![p(0.0, 0.0), p(1.0, 0.0)];
        assert!(!ConvexHullBuilder::point_inside_hull(&hull, &p(0.5, 0.0)));
    }

    #[test]
    fn test_duplicate_points_handled() {
        let mut b = ConvexHullBuilder::new();
        // Repeat the same valid triangle 3× — hull should still have 3 verts.
        for _ in 0..3 {
            b.add_points(vec![p(0.0, 0.0), p(4.0, 0.0), p(2.0, 3.0)]);
        }
        let r = b.compute().expect("ok");
        assert_eq!(r.hull.len(), 3);
    }

    #[test]
    fn test_centroid_right_triangle() {
        let mut b = ConvexHullBuilder::new();
        b.add_points(vec![p(0.0, 0.0), p(3.0, 0.0), p(0.0, 3.0)]);
        let r = b.compute().expect("ok");
        assert!(approx_eq(r.centroid.x, 1.0));
        assert!(approx_eq(r.centroid.y, 1.0));
    }
}
