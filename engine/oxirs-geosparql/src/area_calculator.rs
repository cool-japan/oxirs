//! Area calculations for geographic polygons.
//!
//! Provides the shoelace formula (2D planar), spherical excess (unit sphere),
//! geodesic area on the WGS84 ellipsoid, multi-polygon support, self-intersection
//! detection, signed area / orientation, unit conversion, centroid computation,
//! and perimeter calculation.

use std::f64::consts::PI;

// ── WGS84 ellipsoid constants ────────────────────────────────────────────────

/// Semi-major axis (equatorial radius) in metres.
const WGS84_A: f64 = 6_378_137.0;
/// Semi-minor axis (polar radius) in metres.
const WGS84_B: f64 = 6_356_752.314_245;
/// Mean radius in metres (used for spherical approximation).
const MEAN_RADIUS_M: f64 = 6_371_008.8;

// ── Area unit ────────────────────────────────────────────────────────────────

/// Supported area units for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AreaUnit {
    /// Square metres (SI base).
    SquareMetres,
    /// Square kilometres.
    SquareKilometres,
    /// Acres.
    Acres,
    /// Hectares.
    Hectares,
    /// Square miles.
    SquareMiles,
    /// Square feet.
    SquareFeet,
}

/// Orientation (winding order) of a ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Counter-clockwise (positive signed area).
    CounterClockwise,
    /// Clockwise (negative signed area).
    Clockwise,
    /// Degenerate (zero area).
    Degenerate,
}

/// Statistics about a polygon area calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct AreaStats {
    /// Number of vertices in the exterior ring.
    pub exterior_vertices: usize,
    /// Number of holes.
    pub hole_count: usize,
    /// Total area in the requested unit.
    pub area: f64,
    /// Perimeter of the exterior ring.
    pub perimeter: f64,
}

/// A 2-D coordinate pair (x / longitude, y / latitude).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord {
    /// X / longitude.
    pub x: f64,
    /// Y / latitude.
    pub y: f64,
}

impl Coord {
    /// Create a new coordinate.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

// ── AreaCalculator ───────────────────────────────────────────────────────────

/// Stateless area-calculation utilities for geographic polygons.
pub struct AreaCalculator;

impl AreaCalculator {
    // ── Shoelace formula (2D planar) ─────────────────────────────────────────

    /// Compute the **signed** area of a planar polygon using the shoelace
    /// formula.  A positive result means counter-clockwise winding; negative
    /// means clockwise.
    ///
    /// The ring does **not** need to be explicitly closed (the last→first
    /// edge is handled automatically).
    pub fn signed_area(ring: &[Coord]) -> f64 {
        if ring.len() < 3 {
            return 0.0;
        }
        let n = ring.len();
        let mut sum = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            sum += ring[i].x * ring[j].y;
            sum -= ring[j].x * ring[i].y;
        }
        sum / 2.0
    }

    /// Absolute planar area (always non-negative).
    pub fn planar_area(ring: &[Coord]) -> f64 {
        Self::signed_area(ring).abs()
    }

    /// Determine winding orientation from the signed area.
    pub fn orientation(ring: &[Coord]) -> Orientation {
        let sa = Self::signed_area(ring);
        if sa > f64::EPSILON {
            Orientation::CounterClockwise
        } else if sa < -f64::EPSILON {
            Orientation::Clockwise
        } else {
            Orientation::Degenerate
        }
    }

    // ── Spherical excess (unit sphere) ───────────────────────────────────────

    /// Spherical excess area on a **unit sphere** for a spherical polygon
    /// whose vertices are given in **degrees** (longitude, latitude).
    ///
    /// Uses L'Huilier's theorem via half-angle tangent sums.
    pub fn spherical_excess_deg(ring: &[Coord]) -> f64 {
        if ring.len() < 3 {
            return 0.0;
        }
        let rad: Vec<(f64, f64)> = ring
            .iter()
            .map(|c| (c.x.to_radians(), c.y.to_radians()))
            .collect();
        Self::spherical_excess_rad(&rad)
    }

    /// Spherical excess on a unit sphere.  Vertices are (lon_rad, lat_rad).
    fn spherical_excess_rad(ring: &[(f64, f64)]) -> f64 {
        let n = ring.len();
        if n < 3 {
            return 0.0;
        }

        // Sum of interior angles via the haversine method.
        let mut angle_sum = 0.0;
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            let next = (i + 1) % n;

            let bearing_prev = Self::initial_bearing_rad(ring[i], ring[prev]);
            let bearing_next = Self::initial_bearing_rad(ring[i], ring[next]);

            let mut angle = bearing_next - bearing_prev;
            // Normalise to [0, 2π)
            while angle < 0.0 {
                angle += 2.0 * PI;
            }
            while angle >= 2.0 * PI {
                angle -= 2.0 * PI;
            }
            if angle > PI {
                angle = 2.0 * PI - angle;
            }
            angle_sum += angle;
        }

        // Spherical excess E = sum(angles) - (n - 2) * π
        let excess = angle_sum - (n as f64 - 2.0) * PI;
        excess.abs()
    }

    /// Initial bearing from p1 to p2 (both in radians).
    fn initial_bearing_rad(p1: (f64, f64), p2: (f64, f64)) -> f64 {
        let (lon1, lat1) = p1;
        let (lon2, lat2) = p2;
        let d_lon = lon2 - lon1;
        let y = d_lon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * d_lon.cos();
        y.atan2(x)
    }

    /// Spherical area in square metres (approximation using mean Earth radius).
    pub fn spherical_area_m2(ring: &[Coord]) -> f64 {
        Self::spherical_excess_deg(ring) * MEAN_RADIUS_M * MEAN_RADIUS_M
    }

    // ── Geodesic area on WGS84 ──────────────────────────────────────────────

    /// Geodesic area on the WGS84 ellipsoid (in square metres) using the
    /// trapezoidal integration approach over reduced latitude.
    ///
    /// Vertices are in (longitude, latitude) degrees.
    pub fn geodesic_area_m2(ring: &[Coord]) -> f64 {
        if ring.len() < 3 {
            return 0.0;
        }
        let n = ring.len();

        // Eccentricity squared
        let e2 = 1.0 - (WGS84_B * WGS84_B) / (WGS84_A * WGS84_A);
        let e = e2.sqrt();

        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let lon1 = ring[i].x.to_radians();
            let lat1 = ring[i].y.to_radians();
            let lon2 = ring[j].x.to_radians();
            let lat2 = ring[j].y.to_radians();

            area += (lon2 - lon1)
                * (2.0 + lat1.sin() + lat2.sin())
                * Self::authalic_factor(lat1, e)
                * Self::authalic_factor(lat2, e);
        }

        // The raw formula may give a negative value depending on winding.
        let authalic_r2 = WGS84_A * WGS84_A * Self::authalic_radius_factor(e);
        (area.abs() / 4.0) * authalic_r2
    }

    /// Helper: authalic latitude scaling factor.
    fn authalic_factor(lat: f64, e: f64) -> f64 {
        let sin_lat = lat.sin();
        let denom = 1.0 - e * e * sin_lat * sin_lat;
        if denom.abs() < f64::EPSILON {
            return 1.0;
        }
        1.0 / denom.sqrt()
    }

    /// Authalic radius factor R²/a² = (1 + (e'²/3) + (e'⁴/5) + ...) approximation.
    fn authalic_radius_factor(e: f64) -> f64 {
        let e2 = e * e;
        1.0 + e2 / 3.0 + e2 * e2 / 5.0 + e2 * e2 * e2 / 7.0
    }

    // ── Multi-polygon area ───────────────────────────────────────────────────

    /// Compute the planar area of a multi-polygon.
    ///
    /// Each element is `(exterior_ring, holes)`.  The total area is the sum of
    /// the exterior areas minus the hole areas.
    pub fn multi_polygon_area(parts: &[(Vec<Coord>, Vec<Vec<Coord>>)]) -> f64 {
        let mut total = 0.0;
        for (exterior, holes) in parts {
            total += Self::planar_area(exterior);
            for hole in holes {
                total -= Self::planar_area(hole);
            }
        }
        total.max(0.0)
    }

    /// Compute geodesic area of a multi-polygon (each part is exterior + holes).
    pub fn multi_polygon_geodesic_area(parts: &[(Vec<Coord>, Vec<Vec<Coord>>)]) -> f64 {
        let mut total = 0.0;
        for (exterior, holes) in parts {
            total += Self::geodesic_area_m2(exterior);
            for hole in holes {
                total -= Self::geodesic_area_m2(hole);
            }
        }
        total.max(0.0)
    }

    // ── Polygon with holes (planar) ──────────────────────────────────────────

    /// Planar area of a polygon with optional holes.
    pub fn polygon_area_with_holes(exterior: &[Coord], holes: &[Vec<Coord>]) -> f64 {
        let mut area = Self::planar_area(exterior);
        for hole in holes {
            area -= Self::planar_area(hole);
        }
        area.max(0.0)
    }

    // ── Self-intersection detection ──────────────────────────────────────────

    /// Check whether a polygon ring is simple (no self-intersections).
    ///
    /// Uses the O(n²) brute-force segment intersection test.  Adequate for
    /// moderate-sized polygons; for very large ones a sweep-line algorithm
    /// should be considered.
    pub fn is_simple_polygon(ring: &[Coord]) -> bool {
        if ring.len() < 4 {
            // A triangle cannot self-intersect.
            return true;
        }
        let n = ring.len();
        for i in 0..n {
            let i_next = (i + 1) % n;
            // Only test non-adjacent segments that are at least 2 apart.
            for j in (i + 2)..n {
                let j_next = (j + 1) % n;
                // Skip the wrap-around pair (first segment vs last segment share a vertex).
                if i == 0 && j_next == 0 {
                    continue;
                }
                if Self::segments_intersect(ring[i], ring[i_next], ring[j], ring[j_next]) {
                    return false;
                }
            }
        }
        true
    }

    /// Test whether two closed segments (p1-p2) and (p3-p4) properly intersect
    /// (i.e. cross each other, not just touch at an endpoint).
    fn segments_intersect(p1: Coord, p2: Coord, p3: Coord, p4: Coord) -> bool {
        let d1 = Self::cross_2d(p3, p4, p1);
        let d2 = Self::cross_2d(p3, p4, p2);
        let d3 = Self::cross_2d(p1, p2, p3);
        let d4 = Self::cross_2d(p1, p2, p4);

        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
            && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
        {
            return true;
        }

        // Collinear overlap check.
        if d1.abs() < f64::EPSILON && Self::on_segment(p3, p4, p1) {
            return true;
        }
        if d2.abs() < f64::EPSILON && Self::on_segment(p3, p4, p2) {
            return true;
        }
        if d3.abs() < f64::EPSILON && Self::on_segment(p1, p2, p3) {
            return true;
        }
        if d4.abs() < f64::EPSILON && Self::on_segment(p1, p2, p4) {
            return true;
        }

        false
    }

    /// 2D cross product: (b−a)×(c−a).
    fn cross_2d(a: Coord, b: Coord, c: Coord) -> f64 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    /// Is point `p` on the axis-aligned bounding box of segment `a`–`b`?
    fn on_segment(a: Coord, b: Coord, p: Coord) -> bool {
        p.x >= a.x.min(b.x) && p.x <= a.x.max(b.x) && p.y >= a.y.min(b.y) && p.y <= a.y.max(b.y)
    }

    // ── Unit conversion ──────────────────────────────────────────────────────

    /// Convert an area from square metres to the given unit.
    pub fn convert_area(area_m2: f64, unit: AreaUnit) -> f64 {
        match unit {
            AreaUnit::SquareMetres => area_m2,
            AreaUnit::SquareKilometres => area_m2 / 1_000_000.0,
            AreaUnit::Acres => area_m2 / 4046.8564224,
            AreaUnit::Hectares => area_m2 / 10_000.0,
            AreaUnit::SquareMiles => area_m2 / 2_589_988.110_336,
            AreaUnit::SquareFeet => area_m2 * 10.763_910_417,
        }
    }

    // ── Centroid ─────────────────────────────────────────────────────────────

    /// Centroid of a simple planar polygon.
    ///
    /// Returns `None` if the ring has fewer than 3 vertices or has zero area.
    pub fn centroid(ring: &[Coord]) -> Option<Coord> {
        if ring.len() < 3 {
            return None;
        }
        let n = ring.len();
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut a = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            let cross = ring[i].x * ring[j].y - ring[j].x * ring[i].y;
            cx += (ring[i].x + ring[j].x) * cross;
            cy += (ring[i].y + ring[j].y) * cross;
            a += cross;
        }

        if a.abs() < f64::EPSILON {
            return None;
        }
        let factor = 1.0 / (3.0 * a);
        Some(Coord::new(cx * factor, cy * factor))
    }

    // ── Perimeter ────────────────────────────────────────────────────────────

    /// Euclidean perimeter of a planar ring.
    pub fn perimeter(ring: &[Coord]) -> f64 {
        if ring.len() < 2 {
            return 0.0;
        }
        let n = ring.len();
        let mut total = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = ring[j].x - ring[i].x;
            let dy = ring[j].y - ring[i].y;
            total += (dx * dx + dy * dy).sqrt();
        }
        total
    }

    /// Geodesic perimeter (Haversine approximation) in metres.
    /// Vertices are (longitude, latitude) in degrees.
    pub fn geodesic_perimeter_m(ring: &[Coord]) -> f64 {
        if ring.len() < 2 {
            return 0.0;
        }
        let n = ring.len();
        let mut total = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            total += Self::haversine_m(ring[i], ring[j]);
        }
        total
    }

    /// Haversine distance between two points (degrees) in metres.
    fn haversine_m(a: Coord, b: Coord) -> f64 {
        let lat1 = a.y.to_radians();
        let lat2 = b.y.to_radians();
        let dlat = (b.y - a.y).to_radians();
        let dlon = (b.x - a.x).to_radians();

        let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
        2.0 * MEAN_RADIUS_M * h.sqrt().asin()
    }

    // ── Stats helper ─────────────────────────────────────────────────────────

    /// Compute area statistics for a polygon with optional holes.
    pub fn area_stats(exterior: &[Coord], holes: &[Vec<Coord>], unit: AreaUnit) -> AreaStats {
        let raw = Self::polygon_area_with_holes(exterior, holes);
        AreaStats {
            exterior_vertices: exterior.len(),
            hole_count: holes.len(),
            area: Self::convert_area(raw, unit),
            perimeter: Self::perimeter(exterior),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn unit_square() -> Vec<Coord> {
        vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 1.0),
        ]
    }

    fn unit_square_cw() -> Vec<Coord> {
        vec![
            Coord::new(0.0, 0.0),
            Coord::new(0.0, 1.0),
            Coord::new(1.0, 1.0),
            Coord::new(1.0, 0.0),
        ]
    }

    fn triangle() -> Vec<Coord> {
        vec![
            Coord::new(0.0, 0.0),
            Coord::new(4.0, 0.0),
            Coord::new(2.0, 3.0),
        ]
    }

    // ── Shoelace / planar area ───────────────────────────────────────────────

    #[test]
    fn test_planar_area_unit_square() {
        let area = AreaCalculator::planar_area(&unit_square());
        assert!(approx_eq(area, 1.0, 1e-10));
    }

    #[test]
    fn test_signed_area_ccw() {
        let sa = AreaCalculator::signed_area(&unit_square());
        assert!(sa > 0.0, "CCW ring should have positive signed area");
        assert!(approx_eq(sa, 1.0, 1e-10));
    }

    #[test]
    fn test_signed_area_cw() {
        let sa = AreaCalculator::signed_area(&unit_square_cw());
        assert!(sa < 0.0, "CW ring should have negative signed area");
        assert!(approx_eq(sa.abs(), 1.0, 1e-10));
    }

    #[test]
    fn test_planar_area_triangle() {
        let area = AreaCalculator::planar_area(&triangle());
        assert!(approx_eq(area, 6.0, 1e-10));
    }

    #[test]
    fn test_planar_area_empty() {
        let area = AreaCalculator::planar_area(&[]);
        assert!(approx_eq(area, 0.0, 1e-10));
    }

    #[test]
    fn test_planar_area_two_points() {
        let ring = vec![Coord::new(0.0, 0.0), Coord::new(1.0, 1.0)];
        let area = AreaCalculator::planar_area(&ring);
        assert!(approx_eq(area, 0.0, 1e-10));
    }

    #[test]
    fn test_planar_area_rectangle() {
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(5.0, 0.0),
            Coord::new(5.0, 3.0),
            Coord::new(0.0, 3.0),
        ];
        let area = AreaCalculator::planar_area(&ring);
        assert!(approx_eq(area, 15.0, 1e-10));
    }

    #[test]
    fn test_planar_area_l_shape() {
        // L-shaped polygon
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(3.0, 0.0),
            Coord::new(3.0, 1.0),
            Coord::new(1.0, 1.0),
            Coord::new(1.0, 3.0),
            Coord::new(0.0, 3.0),
        ];
        let area = AreaCalculator::planar_area(&ring);
        // Area = 3*1 + 1*2 = 5
        assert!(approx_eq(area, 5.0, 1e-10));
    }

    // ── Orientation ──────────────────────────────────────────────────────────

    #[test]
    fn test_orientation_ccw() {
        assert_eq!(
            AreaCalculator::orientation(&unit_square()),
            Orientation::CounterClockwise
        );
    }

    #[test]
    fn test_orientation_cw() {
        assert_eq!(
            AreaCalculator::orientation(&unit_square_cw()),
            Orientation::Clockwise
        );
    }

    #[test]
    fn test_orientation_degenerate() {
        let ring = vec![Coord::new(0.0, 0.0), Coord::new(1.0, 1.0)];
        assert_eq!(AreaCalculator::orientation(&ring), Orientation::Degenerate);
    }

    #[test]
    fn test_orientation_collinear() {
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(2.0, 0.0),
        ];
        assert_eq!(AreaCalculator::orientation(&ring), Orientation::Degenerate);
    }

    // ── Spherical excess ─────────────────────────────────────────────────────

    #[test]
    fn test_spherical_excess_small_polygon() {
        // Small polygon around (0,0) — should be non-zero.
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 1.0),
        ];
        let excess = AreaCalculator::spherical_excess_deg(&ring);
        assert!(excess > 0.0);
    }

    #[test]
    fn test_spherical_excess_empty() {
        assert!(approx_eq(
            AreaCalculator::spherical_excess_deg(&[]),
            0.0,
            1e-10
        ));
    }

    #[test]
    fn test_spherical_area_m2_positive() {
        let ring = vec![
            Coord::new(-1.0, -1.0),
            Coord::new(1.0, -1.0),
            Coord::new(1.0, 1.0),
            Coord::new(-1.0, 1.0),
        ];
        let area = AreaCalculator::spherical_area_m2(&ring);
        assert!(area > 0.0);
    }

    // ── Geodesic area ────────────────────────────────────────────────────────

    #[test]
    fn test_geodesic_area_positive() {
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 1.0),
        ];
        let area = AreaCalculator::geodesic_area_m2(&ring);
        assert!(area > 0.0);
    }

    #[test]
    fn test_geodesic_area_empty() {
        assert!(approx_eq(AreaCalculator::geodesic_area_m2(&[]), 0.0, 1e-10));
    }

    #[test]
    fn test_geodesic_area_two_points() {
        let ring = vec![Coord::new(0.0, 0.0), Coord::new(1.0, 1.0)];
        assert!(approx_eq(
            AreaCalculator::geodesic_area_m2(&ring),
            0.0,
            1e-10
        ));
    }

    #[test]
    fn test_geodesic_area_rough_order_of_magnitude() {
        // A 1° × 1° box near the equator ≈ 12 300 km²
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 1.0),
        ];
        let area_km2 = AreaCalculator::geodesic_area_m2(&ring) / 1e6;
        // Allow wide tolerance (order-of-magnitude check)
        assert!(area_km2 > 1_000.0 && area_km2 < 50_000.0);
    }

    // ── Multi-polygon ────────────────────────────────────────────────────────

    #[test]
    fn test_multi_polygon_area_two_squares() {
        let part1 = (unit_square(), vec![]);
        let part2 = (
            vec![
                Coord::new(10.0, 10.0),
                Coord::new(12.0, 10.0),
                Coord::new(12.0, 12.0),
                Coord::new(10.0, 12.0),
            ],
            vec![],
        );
        let area = AreaCalculator::multi_polygon_area(&[part1, part2]);
        assert!(approx_eq(area, 5.0, 1e-10)); // 1 + 4
    }

    #[test]
    fn test_multi_polygon_area_with_hole() {
        let outer = vec![
            Coord::new(0.0, 0.0),
            Coord::new(10.0, 0.0),
            Coord::new(10.0, 10.0),
            Coord::new(0.0, 10.0),
        ];
        let hole = vec![
            Coord::new(2.0, 2.0),
            Coord::new(8.0, 2.0),
            Coord::new(8.0, 8.0),
            Coord::new(2.0, 8.0),
        ];
        let area = AreaCalculator::multi_polygon_area(&[(outer, vec![hole])]);
        // 100 - 36 = 64
        assert!(approx_eq(area, 64.0, 1e-10));
    }

    #[test]
    fn test_multi_polygon_empty() {
        let area = AreaCalculator::multi_polygon_area(&[]);
        assert!(approx_eq(area, 0.0, 1e-10));
    }

    #[test]
    fn test_multi_polygon_geodesic_positive() {
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 1.0),
        ];
        let area = AreaCalculator::multi_polygon_geodesic_area(&[(ring, vec![])]);
        assert!(area > 0.0);
    }

    // ── Polygon with holes ───────────────────────────────────────────────────

    #[test]
    fn test_polygon_area_with_holes() {
        let outer = vec![
            Coord::new(0.0, 0.0),
            Coord::new(4.0, 0.0),
            Coord::new(4.0, 4.0),
            Coord::new(0.0, 4.0),
        ];
        let hole = vec![
            Coord::new(1.0, 1.0),
            Coord::new(3.0, 1.0),
            Coord::new(3.0, 3.0),
            Coord::new(1.0, 3.0),
        ];
        let area = AreaCalculator::polygon_area_with_holes(&outer, &[hole]);
        assert!(approx_eq(area, 12.0, 1e-10)); // 16 - 4
    }

    #[test]
    fn test_polygon_area_no_holes() {
        let area = AreaCalculator::polygon_area_with_holes(&unit_square(), &[]);
        assert!(approx_eq(area, 1.0, 1e-10));
    }

    // ── Self-intersection detection ──────────────────────────────────────────

    #[test]
    fn test_simple_polygon_square() {
        assert!(AreaCalculator::is_simple_polygon(&unit_square()));
    }

    #[test]
    fn test_simple_polygon_triangle() {
        assert!(AreaCalculator::is_simple_polygon(&triangle()));
    }

    #[test]
    fn test_self_intersecting_bowtie() {
        // Bowtie / figure-eight crosses itself.
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(2.0, 2.0),
            Coord::new(2.0, 0.0),
            Coord::new(0.0, 2.0),
        ];
        assert!(!AreaCalculator::is_simple_polygon(&ring));
    }

    #[test]
    fn test_simple_polygon_two_vertices() {
        // Fewer than 4 vertices is always considered simple.
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(0.5, 1.0),
        ];
        assert!(AreaCalculator::is_simple_polygon(&ring));
    }

    // ── Unit conversion ──────────────────────────────────────────────────────

    #[test]
    fn test_convert_m2_to_km2() {
        let km2 = AreaCalculator::convert_area(1_000_000.0, AreaUnit::SquareKilometres);
        assert!(approx_eq(km2, 1.0, 1e-10));
    }

    #[test]
    fn test_convert_m2_to_hectares() {
        let ha = AreaCalculator::convert_area(10_000.0, AreaUnit::Hectares);
        assert!(approx_eq(ha, 1.0, 1e-10));
    }

    #[test]
    fn test_convert_m2_to_acres() {
        let acres = AreaCalculator::convert_area(4_046.856_422_4, AreaUnit::Acres);
        assert!(approx_eq(acres, 1.0, 1e-6));
    }

    #[test]
    fn test_convert_identity_m2() {
        let m2 = AreaCalculator::convert_area(42.0, AreaUnit::SquareMetres);
        assert!(approx_eq(m2, 42.0, 1e-10));
    }

    #[test]
    fn test_convert_m2_to_sq_miles() {
        let mi2 = AreaCalculator::convert_area(2_589_988.110_336, AreaUnit::SquareMiles);
        assert!(approx_eq(mi2, 1.0, 1e-6));
    }

    #[test]
    fn test_convert_m2_to_sq_feet() {
        let sqft = AreaCalculator::convert_area(1.0, AreaUnit::SquareFeet);
        assert!(approx_eq(sqft, 10.763_910_417, 1e-6));
    }

    // ── Centroid ─────────────────────────────────────────────────────────────

    #[test]
    fn test_centroid_unit_square() {
        let c = AreaCalculator::centroid(&unit_square());
        assert!(c.is_some());
        let c = c.expect("centroid should exist");
        assert!(approx_eq(c.x, 0.5, 1e-10));
        assert!(approx_eq(c.y, 0.5, 1e-10));
    }

    #[test]
    fn test_centroid_triangle() {
        let c = AreaCalculator::centroid(&triangle());
        assert!(c.is_some());
        let c = c.expect("centroid should exist");
        assert!(approx_eq(c.x, 2.0, 1e-10));
        assert!(approx_eq(c.y, 1.0, 1e-10));
    }

    #[test]
    fn test_centroid_empty() {
        assert!(AreaCalculator::centroid(&[]).is_none());
    }

    #[test]
    fn test_centroid_degenerate() {
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(2.0, 0.0),
        ];
        assert!(AreaCalculator::centroid(&ring).is_none());
    }

    // ── Perimeter ────────────────────────────────────────────────────────────

    #[test]
    fn test_perimeter_unit_square() {
        let p = AreaCalculator::perimeter(&unit_square());
        assert!(approx_eq(p, 4.0, 1e-10));
    }

    #[test]
    fn test_perimeter_empty() {
        assert!(approx_eq(AreaCalculator::perimeter(&[]), 0.0, 1e-10));
    }

    #[test]
    fn test_perimeter_single_point() {
        assert!(approx_eq(
            AreaCalculator::perimeter(&[Coord::new(1.0, 2.0)]),
            0.0,
            1e-10
        ));
    }

    #[test]
    fn test_perimeter_triangle() {
        // 4 + 3 + 5 = 12 (right triangle with legs 3 and 4, hypotenuse 5)
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(4.0, 0.0),
            Coord::new(4.0, 3.0),
        ];
        let p = AreaCalculator::perimeter(&ring);
        assert!(approx_eq(p, 12.0, 1e-10));
    }

    #[test]
    fn test_geodesic_perimeter_positive() {
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 1.0),
        ];
        let p = AreaCalculator::geodesic_perimeter_m(&ring);
        assert!(p > 0.0);
        // Rough: 4 edges × ~111 km ≈ 444 km
        assert!(p > 100_000.0 && p < 1_000_000.0);
    }

    #[test]
    fn test_geodesic_perimeter_empty() {
        assert!(approx_eq(
            AreaCalculator::geodesic_perimeter_m(&[]),
            0.0,
            1e-10
        ));
    }

    // ── Area stats ───────────────────────────────────────────────────────────

    #[test]
    fn test_area_stats_basic() {
        let stats = AreaCalculator::area_stats(&unit_square(), &[], AreaUnit::SquareMetres);
        assert_eq!(stats.exterior_vertices, 4);
        assert_eq!(stats.hole_count, 0);
        assert!(approx_eq(stats.area, 1.0, 1e-10));
        assert!(approx_eq(stats.perimeter, 4.0, 1e-10));
    }

    #[test]
    fn test_area_stats_with_hole() {
        let outer = vec![
            Coord::new(0.0, 0.0),
            Coord::new(10.0, 0.0),
            Coord::new(10.0, 10.0),
            Coord::new(0.0, 10.0),
        ];
        let hole = vec![
            Coord::new(1.0, 1.0),
            Coord::new(2.0, 1.0),
            Coord::new(2.0, 2.0),
            Coord::new(1.0, 2.0),
        ];
        let stats = AreaCalculator::area_stats(&outer, &[hole], AreaUnit::SquareMetres);
        assert_eq!(stats.hole_count, 1);
        assert!(approx_eq(stats.area, 99.0, 1e-10));
    }

    // ── Haversine distance (internal, tested indirectly) ─────────────────────

    #[test]
    fn test_haversine_zero_distance() {
        let a = Coord::new(0.0, 0.0);
        let d = AreaCalculator::haversine_m(a, a);
        assert!(approx_eq(d, 0.0, 1e-6));
    }

    #[test]
    fn test_haversine_known_distance() {
        // London (−0.1278, 51.5074) → Paris (2.3522, 48.8566) ≈ 343 km
        let london = Coord::new(-0.1278, 51.5074);
        let paris = Coord::new(2.3522, 48.8566);
        let d = AreaCalculator::haversine_m(london, paris);
        assert!(d > 300_000.0 && d < 400_000.0);
    }
}
