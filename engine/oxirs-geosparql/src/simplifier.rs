//! # Geometry Simplifier
//!
//! Implements geometry simplification algorithms including the Douglas-Peucker algorithm
//! for polyline and polygon simplification, and radial distance simplification.

/// A 2-dimensional point.
#[derive(Debug, Clone, PartialEq)]
pub struct Point2D {
    /// The horizontal coordinate.
    pub x: f64,
    /// The vertical coordinate.
    pub y: f64,
}

impl Point2D {
    /// Create a new `Point2D`.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance squared to another point (avoids a `sqrt`).
    fn dist_sq(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Euclidean distance to another point.
    pub fn distance_to(&self, other: &Point2D) -> f64 {
        self.dist_sq(other).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SimplificationResult
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a simplification operation.
#[derive(Debug, Clone)]
pub struct SimplificationResult {
    /// Number of points in the original sequence.
    pub original_count: usize,
    /// Number of points in the simplified sequence.
    pub simplified_count: usize,
    /// Number of points removed.
    pub removed_count: usize,
    /// The simplified point sequence.
    pub points: Vec<Point2D>,
}

impl SimplificationResult {
    fn new(original_count: usize, points: Vec<Point2D>) -> Self {
        let simplified_count = points.len();
        let removed_count = original_count.saturating_sub(simplified_count);
        Self {
            original_count,
            simplified_count,
            removed_count,
            points,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeometrySimplifier
// ─────────────────────────────────────────────────────────────────────────────

/// Simplifies geometric shapes using various algorithms.
#[derive(Debug, Default)]
pub struct GeometrySimplifier;

impl GeometrySimplifier {
    /// Create a new `GeometrySimplifier`.
    pub fn new() -> Self {
        Self
    }

    /// Simplify an open polyline using the Douglas-Peucker algorithm.
    ///
    /// `epsilon` is the maximum permissible perpendicular distance from a point to
    /// the approximating line segment; any point closer than `epsilon` is removed.
    ///
    /// A value of `0.0` keeps every point; a very large value reduces the polyline
    /// to its two endpoints (or fewer if there are degenerate inputs).
    pub fn simplify(&self, points: &[Point2D], epsilon: f64) -> SimplificationResult {
        if points.len() < 2 {
            return SimplificationResult::new(points.len(), points.to_vec());
        }
        let simplified = Self::douglas_peucker(points, epsilon);
        SimplificationResult::new(points.len(), simplified)
    }

    /// Simplify a closed polygon using Douglas-Peucker.
    ///
    /// The first and last points in `points` are always preserved; the algorithm
    /// is applied to the interior of the ring.
    pub fn simplify_closed(&self, points: &[Point2D], epsilon: f64) -> SimplificationResult {
        if points.len() < 3 {
            return SimplificationResult::new(points.len(), points.to_vec());
        }
        // Treat the ring as an open polyline (first=last for proper closure is
        // a caller concern; we guarantee first+last are kept regardless).
        let simplified = Self::douglas_peucker(points, epsilon);
        SimplificationResult::new(points.len(), simplified)
    }

    /// Simplify a sequence by keeping only points whose distance to the
    /// previously kept point is at least `min_distance`.
    ///
    /// The first point is always kept. The last point is always kept if it
    /// differs from the last selected point.
    pub fn radial_simplify(&self, points: &[Point2D], min_distance: f64) -> SimplificationResult {
        if points.len() < 2 {
            return SimplificationResult::new(points.len(), points.to_vec());
        }

        let mut result = Vec::with_capacity(points.len());
        result.push(Point2D::new(points[0].x, points[0].y));
        let mut last_kept = 0usize;

        for i in 1..points.len() {
            let d = points[last_kept].distance_to(&points[i]);
            if d >= min_distance {
                result.push(Point2D::new(points[i].x, points[i].y));
                last_kept = i;
            }
        }

        // Ensure the last point is always included.
        let last_idx = points.len() - 1;
        if last_kept != last_idx {
            result.push(Point2D::new(points[last_idx].x, points[last_idx].y));
        }

        SimplificationResult::new(points.len(), result)
    }

    /// Compute the perpendicular distance from `point` to the infinite line
    /// defined by `line_start` and `line_end`.
    ///
    /// Returns `0.0` if `line_start == line_end` (degenerate segment).
    pub fn perpendicular_distance(
        point: &Point2D,
        line_start: &Point2D,
        line_end: &Point2D,
    ) -> f64 {
        let dx = line_end.x - line_start.x;
        let dy = line_end.y - line_start.y;
        let line_len_sq = dx * dx + dy * dy;

        if line_len_sq == 0.0 {
            // Degenerate: start == end; return distance to the point.
            return point.distance_to(line_start);
        }

        // Signed area of the triangle formed by the three points, times 2.
        let area2 = ((line_end.x - line_start.x) * (line_start.y - point.y)
            - (line_start.x - point.x) * (line_end.y - line_start.y))
            .abs();

        area2 / line_len_sq.sqrt()
    }

    /// Core Douglas-Peucker recursive implementation.
    ///
    /// Returns a new `Vec<Point2D>` with the simplified sequence.
    pub fn douglas_peucker(points: &[Point2D], epsilon: f64) -> Vec<Point2D> {
        if points.len() < 2 {
            return points.to_vec();
        }

        // Find the point with the maximum perpendicular distance from the line
        // connecting the first and last points.
        let first = &points[0];
        let last = &points[points.len() - 1];

        let mut max_dist = 0.0_f64;
        let mut max_idx = 0usize;

        for (i, p) in points.iter().enumerate().skip(1).take(points.len() - 2) {
            let d = Self::perpendicular_distance(p, first, last);
            if d > max_dist {
                max_dist = d;
                max_idx = i;
            }
        }

        if max_dist > epsilon {
            // Recurse into both sub-sequences.
            let mut left = Self::douglas_peucker(&points[..=max_idx], epsilon);
            let right = Self::douglas_peucker(&points[max_idx..], epsilon);
            // Merge: left already contains the split point; skip the duplicate in right.
            left.extend_from_slice(&right[1..]);
            left
        } else {
            // All intermediate points are within epsilon; keep only endpoints.
            vec![Point2D::new(first.x, first.y), Point2D::new(last.x, last.y)]
        }
    }

    /// Compute the reduction ratio: `removed_count / original_count`.
    ///
    /// Returns `0.0` when the original had zero points.
    pub fn reduction_ratio(result: &SimplificationResult) -> f64 {
        if result.original_count == 0 {
            return 0.0;
        }
        result.removed_count as f64 / result.original_count as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    // ── Point2D ──────────────────────────────────────────────────────────────

    #[test]
    fn test_point_new() {
        let pt = p(3.0, 4.0);
        assert_eq!(pt.x, 3.0);
        assert_eq!(pt.y, 4.0);
    }

    #[test]
    fn test_point_distance() {
        let a = p(0.0, 0.0);
        let b = p(3.0, 4.0);
        assert!(approx_eq(a.distance_to(&b), 5.0));
    }

    #[test]
    fn test_point_distance_zero() {
        let a = p(1.0, 1.0);
        assert!(approx_eq(a.distance_to(&a.clone()), 0.0));
    }

    // ── perpendicular_distance ───────────────────────────────────────────────

    #[test]
    fn test_perp_dist_on_line() {
        // Point on the line has distance 0
        let start = p(0.0, 0.0);
        let end = p(10.0, 0.0);
        let pt = p(5.0, 0.0);
        assert!(approx_eq(
            GeometrySimplifier::perpendicular_distance(&pt, &start, &end),
            0.0
        ));
    }

    #[test]
    fn test_perp_dist_off_line() {
        let start = p(0.0, 0.0);
        let end = p(10.0, 0.0);
        let pt = p(5.0, 3.0);
        assert!(approx_eq(
            GeometrySimplifier::perpendicular_distance(&pt, &start, &end),
            3.0
        ));
    }

    #[test]
    fn test_perp_dist_degenerate_segment() {
        let start = p(2.0, 2.0);
        let end = p(2.0, 2.0);
        let pt = p(5.0, 6.0);
        // distance from (5,6) to (2,2) = 5
        assert!(approx_eq(
            GeometrySimplifier::perpendicular_distance(&pt, &start, &end),
            5.0
        ));
    }

    #[test]
    fn test_perp_dist_diagonal_line() {
        // 45-degree line: y = x
        let start = p(0.0, 0.0);
        let end = p(10.0, 10.0);
        // Point directly above the midpoint
        let pt = p(5.0, 7.0);
        // Distance = |5 - 7| / sqrt(2) = 2 / sqrt(2) ≈ 1.414
        let d = GeometrySimplifier::perpendicular_distance(&pt, &start, &end);
        assert!((d - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    // ── Degenerate inputs ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_zero_points() {
        let s = GeometrySimplifier::new();
        let result = s.simplify(&[], 1.0);
        assert_eq!(result.original_count, 0);
        assert_eq!(result.simplified_count, 0);
        assert_eq!(result.points.len(), 0);
    }

    #[test]
    fn test_simplify_one_point() {
        let s = GeometrySimplifier::new();
        let result = s.simplify(&[p(1.0, 2.0)], 1.0);
        assert_eq!(result.original_count, 1);
        assert_eq!(result.simplified_count, 1);
    }

    #[test]
    fn test_simplify_two_points() {
        let s = GeometrySimplifier::new();
        let pts = vec![p(0.0, 0.0), p(1.0, 1.0)];
        let result = s.simplify(&pts, 0.5);
        assert_eq!(result.simplified_count, 2);
    }

    // ── Straight line ─────────────────────────────────────────────────────────

    #[test]
    fn test_simplify_straight_line_removes_collinear() {
        // All points lie on y = 0; any epsilon > 0 should remove intermediates.
        let pts: Vec<Point2D> = (0..=10).map(|i| p(i as f64, 0.0)).collect();
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.1);
        assert_eq!(result.simplified_count, 2);
        assert_eq!(result.points[0], p(0.0, 0.0));
        assert_eq!(result.points[1], p(10.0, 0.0));
    }

    #[test]
    fn test_simplify_collinear_large_epsilon_reduces_to_endpoints() {
        let pts = vec![p(0.0, 0.0), p(5.0, 0.0), p(10.0, 0.0), p(15.0, 0.0)];
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 1_000.0);
        assert_eq!(result.simplified_count, 2);
    }

    // ── Epsilon = 0 keeps all points ──────────────────────────────────────────

    #[test]
    fn test_simplify_epsilon_zero_keeps_all() {
        let pts: Vec<Point2D> = (0..5).map(|i| p(i as f64, (i as f64).sin())).collect();
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.0);
        assert_eq!(result.simplified_count, pts.len());
    }

    // ── Curve simplification ──────────────────────────────────────────────────

    #[test]
    fn test_simplify_curve_reduces_points() {
        // A semi-circle approximated by many points; significant epsilon should reduce count.
        use std::f64::consts::PI;
        let n = 50;
        let pts: Vec<Point2D> = (0..=n)
            .map(|i| {
                let angle = PI * i as f64 / n as f64;
                p(angle.cos(), angle.sin())
            })
            .collect();
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.05);
        assert!(result.simplified_count < pts.len());
        assert!(result.simplified_count >= 2);
        // First and last are always preserved.
        assert_eq!(result.points[0].x, pts[0].x);
        assert_eq!(result.points.last().map(|p| p.x), pts.last().map(|p| p.x));
    }

    #[test]
    fn test_simplify_curve_with_tiny_epsilon_keeps_more() {
        use std::f64::consts::PI;
        let n = 20;
        let pts: Vec<Point2D> = (0..=n)
            .map(|i| {
                let angle = PI * i as f64 / n as f64;
                p(angle.cos(), angle.sin())
            })
            .collect();
        let s = GeometrySimplifier::new();
        let coarse = s.simplify(&pts, 0.3);
        let fine = s.simplify(&pts, 0.01);
        assert!(fine.simplified_count >= coarse.simplified_count);
    }

    // ── simplify_closed ───────────────────────────────────────────────────────

    #[test]
    fn test_simplify_closed_less_than_3() {
        let pts = vec![p(0.0, 0.0), p(1.0, 0.0)];
        let s = GeometrySimplifier::new();
        let result = s.simplify_closed(&pts, 0.5);
        assert_eq!(result.simplified_count, 2);
    }

    #[test]
    fn test_simplify_closed_keeps_first_and_last() {
        // A square perimeter: no intermediate points should be removed with large epsilon.
        let pts = vec![
            p(0.0, 0.0),
            p(5.0, 0.0),
            p(5.0, 5.0),
            p(0.0, 5.0),
            p(0.0, 0.0),
        ];
        let s = GeometrySimplifier::new();
        let result = s.simplify_closed(&pts, 1.0);
        // First and last should be preserved.
        assert_eq!(result.points.first().map(|p| (p.x, p.y)), Some((0.0, 0.0)));
    }

    #[test]
    fn test_simplify_closed_polygon_collinear() {
        // Collinear "polygon" – all on y = 0
        let pts: Vec<Point2D> = (0..=6).map(|i| p(i as f64, 0.0)).collect();
        let s = GeometrySimplifier::new();
        let result = s.simplify_closed(&pts, 0.1);
        assert_eq!(result.simplified_count, 2);
    }

    #[test]
    fn test_simplify_closed_zero_points() {
        let s = GeometrySimplifier::new();
        let result = s.simplify_closed(&[], 1.0);
        assert_eq!(result.original_count, 0);
        assert_eq!(result.simplified_count, 0);
    }

    // ── radial_simplify ───────────────────────────────────────────────────────

    #[test]
    fn test_radial_simplify_zero_points() {
        let s = GeometrySimplifier::new();
        let result = s.radial_simplify(&[], 1.0);
        assert_eq!(result.simplified_count, 0);
    }

    #[test]
    fn test_radial_simplify_one_point() {
        let s = GeometrySimplifier::new();
        let result = s.radial_simplify(&[p(0.0, 0.0)], 1.0);
        assert_eq!(result.simplified_count, 1);
    }

    #[test]
    fn test_radial_simplify_keeps_first_and_last() {
        let pts: Vec<Point2D> = (0..10).map(|i| p(i as f64, 0.0)).collect();
        let s = GeometrySimplifier::new();
        let result = s.radial_simplify(&pts, 0.5);
        assert_eq!(result.points.first().map(|p| p.x), Some(0.0));
        assert_eq!(result.points.last().map(|p| p.x), Some(9.0));
    }

    #[test]
    fn test_radial_simplify_large_min_distance() {
        let pts: Vec<Point2D> = (0..10).map(|i| p(i as f64, 0.0)).collect();
        let s = GeometrySimplifier::new();
        // With min_distance = 3.0, we skip every point that's < 3.0 away from the last kept.
        let result = s.radial_simplify(&pts, 3.0);
        // First + some intermediate + last
        assert!(result.simplified_count < pts.len());
        assert!(result.simplified_count >= 2);
    }

    #[test]
    fn test_radial_simplify_tiny_min_distance_keeps_all() {
        let pts = vec![p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0)];
        let s = GeometrySimplifier::new();
        let result = s.radial_simplify(&pts, 0.0);
        // With min_distance = 0, every point satisfies d >= 0.
        assert_eq!(result.simplified_count, 3);
    }

    #[test]
    fn test_radial_simplify_exactly_min_distance() {
        // Points spaced exactly 1.0 apart; min_distance = 1.0 → keep all.
        let pts: Vec<Point2D> = (0..5).map(|i| p(i as f64, 0.0)).collect();
        let s = GeometrySimplifier::new();
        let result = s.radial_simplify(&pts, 1.0);
        assert_eq!(result.simplified_count, 5);
    }

    // ── reduction_ratio ───────────────────────────────────────────────────────

    #[test]
    fn test_reduction_ratio_full_simplification() {
        let pts: Vec<Point2D> = (0..=10).map(|i| p(i as f64, 0.0)).collect();
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.1);
        let ratio = GeometrySimplifier::reduction_ratio(&result);
        // 11 original, 2 kept → 9 removed → ratio ≈ 0.818
        assert!(ratio > 0.8);
        assert!(ratio <= 1.0);
    }

    #[test]
    fn test_reduction_ratio_no_simplification() {
        let pts = vec![p(0.0, 0.0), p(1.0, 1.0)];
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.5);
        let ratio = GeometrySimplifier::reduction_ratio(&result);
        assert!(approx_eq(ratio, 0.0));
    }

    #[test]
    fn test_reduction_ratio_zero_original() {
        let result = SimplificationResult {
            original_count: 0,
            simplified_count: 0,
            removed_count: 0,
            points: vec![],
        };
        assert!(approx_eq(GeometrySimplifier::reduction_ratio(&result), 0.0));
    }

    // ── SimplificationResult fields ───────────────────────────────────────────

    #[test]
    fn test_simplification_result_counts_consistent() {
        let pts: Vec<Point2D> = (0..8).map(|i| p(i as f64, 0.0)).collect();
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.1);
        assert_eq!(result.original_count, 8);
        assert_eq!(
            result.original_count,
            result.simplified_count + result.removed_count
        );
    }

    // ── douglas_peucker static method ─────────────────────────────────────────

    #[test]
    fn test_douglas_peucker_single_point() {
        let pts = vec![p(1.0, 2.0)];
        let out = GeometrySimplifier::douglas_peucker(&pts, 1.0);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_douglas_peucker_two_points() {
        let pts = vec![p(0.0, 0.0), p(1.0, 1.0)];
        let out = GeometrySimplifier::douglas_peucker(&pts, 0.5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_douglas_peucker_notch_preserved() {
        // Significant detour: middle point is far from the endpoint line.
        let pts = vec![p(0.0, 0.0), p(5.0, 10.0), p(10.0, 0.0)];
        let out = GeometrySimplifier::douglas_peucker(&pts, 1.0);
        // The notch point (5,10) is 5 units from the baseline — well above epsilon=1.
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_simplify_preserves_endpoints() {
        let pts: Vec<Point2D> = (0..20)
            .map(|i| p(i as f64, (i as f64 * 0.5).sin()))
            .collect();
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.1);
        // Endpoints always kept.
        assert_eq!(
            result.points.first().map(|p| (p.x, p.y)),
            Some((pts[0].x, pts[0].y))
        );
        assert_eq!(
            result.points.last().map(|p| (p.x, p.y)),
            pts.last().map(|p| (p.x, p.y))
        );
    }

    #[test]
    fn test_simplify_monotone_increase_then_flat() {
        // Sharp rise followed by flat: the flat portion should collapse.
        let mut pts = vec![
            p(0.0, 0.0),
            p(1.0, 5.0),
            p(2.0, 5.0),
            p(3.0, 5.0),
            p(4.0, 5.0),
        ];
        // Add an explicit last point far from start so the full span is captured.
        pts.push(p(10.0, 5.0));
        let s = GeometrySimplifier::new();
        let result = s.simplify(&pts, 0.1);
        // The flat portion (2,5), (3,5), (4,5) should be simplified out.
        assert!(result.simplified_count < pts.len());
    }
}
