//! Bounding box (Envelope) operations for spatial queries.
//!
//! Provides a simple axis-aligned bounding box (AABB) type with intersection,
//! union, containment, area and WKT serialization.

// ---------------------------------------------------------------------------
// BoundingBox
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box (envelope) in 2-D space.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Minimum X (west) coordinate.
    pub min_x: f64,
    /// Minimum Y (south) coordinate.
    pub min_y: f64,
    /// Maximum X (east) coordinate.
    pub max_x: f64,
    /// Maximum Y (north) coordinate.
    pub max_y: f64,
}

impl BoundingBox {
    /// Create a new `BoundingBox`.  Returns `None` if `min_x > max_x` or
    /// `min_y > max_y`.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Option<Self> {
        if min_x > max_x || min_y > max_y {
            None
        } else {
            Some(BoundingBox {
                min_x,
                min_y,
                max_x,
                max_y,
            })
        }
    }

    /// Create a degenerate (point) bounding box.
    pub fn point(x: f64, y: f64) -> Self {
        BoundingBox {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
        }
    }

    /// Create the minimal bounding box that encompasses all given `points`.
    /// Returns `None` when the slice is empty.
    pub fn from_points(points: &[(f64, f64)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let (x0, y0) = points[0];
        let mut min_x = x0;
        let mut min_y = y0;
        let mut max_x = x0;
        let mut max_y = y0;
        for &(x, y) in points.iter().skip(1) {
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
        }
        Some(BoundingBox {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    /// Return `true` when the point `(x, y)` lies inside or on the boundary.
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Return `true` when `self` entirely contains `other` (inclusive).
    pub fn contains(&self, other: &BoundingBox) -> bool {
        other.min_x >= self.min_x
            && other.max_x <= self.max_x
            && other.min_y >= self.min_y
            && other.max_y <= self.max_y
    }

    /// Return `true` when `self` and `other` share any area or boundary.
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// Return the intersection of `self` and `other`, or `None` when they are
    /// disjoint.
    pub fn intersection(&self, other: &BoundingBox) -> Option<BoundingBox> {
        let min_x = self.min_x.max(other.min_x);
        let min_y = self.min_y.max(other.min_y);
        let max_x = self.max_x.min(other.max_x);
        let max_y = self.max_y.min(other.max_y);
        BoundingBox::new(min_x, min_y, max_x, max_y)
    }

    /// Return the smallest bounding box that contains both `self` and `other`.
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        BoundingBox {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Return a new bounding box expanded by `margin` on all four sides.
    pub fn expand_by(&self, margin: f64) -> BoundingBox {
        BoundingBox {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
        }
    }

    /// Compute the area of the bounding box.
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Width of the bounding box (extent along the x-axis).
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the bounding box (extent along the y-axis).
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Geometric centre of the bounding box.
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }

    /// Return `true` when width == 0 AND height == 0 (degenerate point box).
    pub fn is_point(&self) -> bool {
        (self.max_x - self.min_x).abs() < f64::EPSILON
            && (self.max_y - self.min_y).abs() < f64::EPSILON
    }

    /// Serialize as a closed WKT polygon (5 vertices; first == last).
    ///
    /// Format: `POLYGON ((minX minY, maxX minY, maxX maxY, minX maxY, minX minY))`
    pub fn to_wkt(&self) -> String {
        format!(
            "POLYGON (({minX} {minY}, {maxX} {minY}, {maxX} {maxY}, {minX} {maxY}, {minX} {minY}))",
            minX = self.min_x,
            minY = self.min_y,
            maxX = self.max_x,
            maxY = self.max_y,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> BoundingBox {
        BoundingBox::new(min_x, min_y, max_x, max_y).expect("should succeed")
    }

    // --- Construction ---
    #[test]
    fn test_new_valid() {
        let b = BoundingBox::new(0.0, 0.0, 1.0, 1.0);
        assert!(b.is_some());
    }

    #[test]
    fn test_new_invalid_x() {
        assert!(BoundingBox::new(2.0, 0.0, 1.0, 1.0).is_none());
    }

    #[test]
    fn test_new_invalid_y() {
        assert!(BoundingBox::new(0.0, 5.0, 1.0, 1.0).is_none());
    }

    #[test]
    fn test_new_degenerate_valid() {
        assert!(BoundingBox::new(0.0, 0.0, 0.0, 0.0).is_some());
    }

    #[test]
    fn test_point_constructor() {
        let p = BoundingBox::point(3.0, 4.0);
        assert_eq!(p.min_x, 3.0);
        assert_eq!(p.max_x, 3.0);
        assert!(p.is_point());
    }

    #[test]
    fn test_from_points_single() {
        let b = BoundingBox::from_points(&[(2.0, 3.0)]).expect("should succeed");
        assert!(b.is_point());
    }

    #[test]
    fn test_from_points_multiple() {
        let b = BoundingBox::from_points(&[(1.0, 1.0), (4.0, 2.0), (2.0, 5.0)])
            .expect("should succeed");
        assert_eq!(b.min_x, 1.0);
        assert_eq!(b.min_y, 1.0);
        assert_eq!(b.max_x, 4.0);
        assert_eq!(b.max_y, 5.0);
    }

    #[test]
    fn test_from_points_empty() {
        assert!(BoundingBox::from_points(&[]).is_none());
    }

    // --- contains_point ---
    #[test]
    fn test_contains_point_inside() {
        let b = bb(0.0, 0.0, 4.0, 4.0);
        assert!(b.contains_point(2.0, 2.0));
    }

    #[test]
    fn test_contains_point_on_boundary() {
        let b = bb(0.0, 0.0, 4.0, 4.0);
        assert!(b.contains_point(0.0, 0.0));
        assert!(b.contains_point(4.0, 4.0));
    }

    #[test]
    fn test_contains_point_outside() {
        let b = bb(0.0, 0.0, 4.0, 4.0);
        assert!(!b.contains_point(5.0, 2.0));
    }

    // --- contains ---
    #[test]
    fn test_contains_identical() {
        let b = bb(0.0, 0.0, 4.0, 4.0);
        assert!(b.contains(&b.clone()));
    }

    #[test]
    fn test_contains_smaller() {
        let outer = bb(0.0, 0.0, 10.0, 10.0);
        let inner = bb(2.0, 2.0, 5.0, 5.0);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_contains_partial_overlap() {
        let a = bb(0.0, 0.0, 4.0, 4.0);
        let b = bb(2.0, 2.0, 6.0, 6.0);
        assert!(!a.contains(&b));
    }

    // --- intersects ---
    #[test]
    fn test_intersects_overlapping() {
        let a = bb(0.0, 0.0, 4.0, 4.0);
        let b = bb(2.0, 2.0, 6.0, 6.0);
        assert!(a.intersects(&b));
    }

    #[test]
    fn test_intersects_touching_boundary() {
        let a = bb(0.0, 0.0, 2.0, 2.0);
        let b = bb(2.0, 0.0, 4.0, 2.0);
        assert!(a.intersects(&b));
    }

    #[test]
    fn test_intersects_disjoint() {
        let a = bb(0.0, 0.0, 1.0, 1.0);
        let b = bb(2.0, 2.0, 3.0, 3.0);
        assert!(!a.intersects(&b));
    }

    // --- intersection ---
    #[test]
    fn test_intersection_overlapping() {
        let a = bb(0.0, 0.0, 4.0, 4.0);
        let b = bb(2.0, 2.0, 6.0, 6.0);
        let i = a.intersection(&b).expect("should succeed");
        assert_eq!(i.min_x, 2.0);
        assert_eq!(i.min_y, 2.0);
        assert_eq!(i.max_x, 4.0);
        assert_eq!(i.max_y, 4.0);
    }

    #[test]
    fn test_intersection_disjoint_is_none() {
        let a = bb(0.0, 0.0, 1.0, 1.0);
        let b = bb(2.0, 2.0, 3.0, 3.0);
        assert!(a.intersection(&b).is_none());
    }

    // --- union ---
    #[test]
    fn test_union() {
        let a = bb(0.0, 0.0, 2.0, 2.0);
        let b = bb(3.0, 3.0, 5.0, 5.0);
        let u = a.union(&b);
        assert_eq!(u.min_x, 0.0);
        assert_eq!(u.min_y, 0.0);
        assert_eq!(u.max_x, 5.0);
        assert_eq!(u.max_y, 5.0);
    }

    // --- expand_by ---
    #[test]
    fn test_expand_by_positive() {
        let b = bb(1.0, 1.0, 3.0, 3.0);
        let e = b.expand_by(1.0);
        assert_eq!(e.min_x, 0.0);
        assert_eq!(e.min_y, 0.0);
        assert_eq!(e.max_x, 4.0);
        assert_eq!(e.max_y, 4.0);
    }

    #[test]
    fn test_expand_by_zero() {
        let b = bb(0.0, 0.0, 1.0, 1.0);
        let e = b.expand_by(0.0);
        assert_eq!(e, b);
    }

    // --- area ---
    #[test]
    fn test_area() {
        let b = bb(0.0, 0.0, 3.0, 4.0);
        assert!((b.area() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_area_point() {
        let b = BoundingBox::point(0.0, 0.0);
        assert!((b.area() - 0.0).abs() < 1e-9);
    }

    // --- width / height ---
    #[test]
    fn test_width_height() {
        let b = bb(1.0, 2.0, 4.0, 7.0);
        assert!((b.width() - 3.0).abs() < 1e-9);
        assert!((b.height() - 5.0).abs() < 1e-9);
    }

    // --- center ---
    #[test]
    fn test_center() {
        let b = bb(0.0, 0.0, 4.0, 6.0);
        let (cx, cy) = b.center();
        assert!((cx - 2.0).abs() < 1e-9);
        assert!((cy - 3.0).abs() < 1e-9);
    }

    // --- is_point ---
    #[test]
    fn test_is_point_true() {
        assert!(BoundingBox::point(1.0, 1.0).is_point());
    }

    #[test]
    fn test_is_point_false() {
        assert!(!bb(0.0, 0.0, 1.0, 1.0).is_point());
    }

    // --- to_wkt ---
    #[test]
    fn test_to_wkt_format() {
        let b = bb(0.0, 0.0, 1.0, 1.0);
        let wkt = b.to_wkt();
        assert!(wkt.starts_with("POLYGON (("));
        assert!(wkt.ends_with("))"));
    }

    #[test]
    fn test_to_wkt_closed_ring() {
        let b = bb(0.0, 0.0, 2.0, 3.0);
        let wkt = b.to_wkt();
        // Should contain the closing vertex identical to the first
        assert!(wkt.contains("0 0"));
        assert!(wkt.contains("2 0"));
        assert!(wkt.contains("2 3"));
        assert!(wkt.contains("0 3"));
    }

    // --- negative coordinates ---
    #[test]
    fn test_negative_coordinates() {
        let b = bb(-5.0, -3.0, -1.0, -1.0);
        assert!(b.contains_point(-3.0, -2.0));
        assert!(!b.contains_point(0.0, 0.0));
    }

    // --- from_points with negative values ---
    #[test]
    fn test_from_points_negative() {
        let b = BoundingBox::from_points(&[(-2.0, -2.0), (2.0, 2.0)]).expect("should succeed");
        assert_eq!(b.min_x, -2.0);
        assert_eq!(b.max_x, 2.0);
    }
}
