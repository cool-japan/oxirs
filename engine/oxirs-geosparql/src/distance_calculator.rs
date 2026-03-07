//! Geodesic and Euclidean distance calculations for GeoSPARQL.
//!
//! Provides Haversine (great-circle), Vincenty (ellipsoidal), Euclidean (2D/3D),
//! bearing, midpoint, polyline distance, point-to-segment distance, and
//! bounding-box expansion operations.

use std::f64::consts::PI;

// ── WGS84 ellipsoid parameters ──────────────────────────────────────────────

/// Semi-major axis of the WGS84 ellipsoid in metres.
const WGS84_A: f64 = 6_378_137.0;
/// Semi-minor axis of the WGS84 ellipsoid in metres.
const WGS84_B: f64 = 6_356_752.314_245;
/// Flattening of the WGS84 ellipsoid.
const WGS84_F: f64 = 1.0 / 298.257_223_563;

// ── Geometry primitives ─────────────────────────────────────────────────────

/// A geographic coordinate in degrees (WGS84).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    /// Latitude in degrees (-90..90).
    pub lat: f64,
    /// Longitude in degrees (-180..180).
    pub lon: f64,
}

impl GeoPoint {
    /// Create a new geographic point.
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    fn lat_rad(self) -> f64 {
        self.lat.to_radians()
    }
    fn lon_rad(self) -> f64 {
        self.lon.to_radians()
    }
}

/// A 2D Cartesian point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A 3D Cartesian point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

impl Point3D {
    /// Create a new 3D point.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// An axis-aligned bounding box (geographic or projected).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Minimum x / longitude.
    pub min_x: f64,
    /// Minimum y / latitude.
    pub min_y: f64,
    /// Maximum x / longitude.
    pub max_x: f64,
    /// Maximum y / latitude.
    pub max_y: f64,
}

impl BBox {
    /// Create a new bounding box.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Width of the bounding box.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Centre point.
    pub fn centre(&self) -> Point2D {
        Point2D::new(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
}

/// Distance calculation errors.
#[derive(Debug)]
pub enum DistanceError {
    /// The Vincenty formula failed to converge.
    VincentyConvergence,
    /// An input coordinate is out of the valid range.
    OutOfRange(String),
    /// A polyline has fewer than two points.
    InsufficientPoints,
}

impl std::fmt::Display for DistanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VincentyConvergence => write!(f, "Vincenty formula failed to converge"),
            Self::OutOfRange(s) => write!(f, "Coordinate out of range: {s}"),
            Self::InsufficientPoints => write!(f, "Polyline requires at least two points"),
        }
    }
}

impl std::error::Error for DistanceError {}

// ── Haversine distance ──────────────────────────────────────────────────────

/// Compute the great-circle distance (Haversine formula) in metres.
///
/// Assumes a spherical Earth with mean radius 6,371,000 m.
pub fn haversine(a: GeoPoint, b: GeoPoint) -> f64 {
    const EARTH_R: f64 = 6_371_000.0;

    let d_lat = b.lat_rad() - a.lat_rad();
    let d_lon = b.lon_rad() - a.lon_rad();

    let half_dlat = (d_lat / 2.0).sin();
    let half_dlon = (d_lon / 2.0).sin();
    let h = half_dlat * half_dlat + a.lat_rad().cos() * b.lat_rad().cos() * half_dlon * half_dlon;

    2.0 * EARTH_R * h.sqrt().asin()
}

// ── Vincenty distance ───────────────────────────────────────────────────────

/// Compute the geodesic distance on the WGS84 ellipsoid using the Vincenty
/// inverse formula. Returns the distance in metres.
///
/// The iterative algorithm converges for most antipodal points (≤ 200 iterations).
pub fn vincenty(a: GeoPoint, b: GeoPoint) -> Result<f64, DistanceError> {
    if (a.lat - b.lat).abs() < 1e-12 && (a.lon - b.lon).abs() < 1e-12 {
        return Ok(0.0);
    }

    let u1 = ((1.0 - WGS84_F) * a.lat_rad().tan()).atan();
    let u2 = ((1.0 - WGS84_F) * b.lat_rad().tan()).atan();
    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let l = b.lon_rad() - a.lon_rad();
    let mut lambda = l;

    for _ in 0..200 {
        let sin_lam = lambda.sin();
        let cos_lam = lambda.cos();

        let sin_sigma = ((cos_u2 * sin_lam).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam).powi(2))
        .sqrt();

        if sin_sigma < 1e-12 {
            return Ok(0.0); // coincident points
        }

        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam;
        let sigma = sin_sigma.atan2(cos_sigma);

        let sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma;
        let cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        let cos_2sigma_m = if cos2_alpha.abs() < 1e-12 {
            0.0
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
        };

        let c = WGS84_F / 16.0 * cos2_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos2_alpha));
        let prev_lambda = lambda;
        lambda = l
            + (1.0 - c)
                * WGS84_F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

        if (lambda - prev_lambda).abs() < 1e-12 {
            // converged – compute distance
            let u_sq = cos2_alpha * (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);
            let aa =
                1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
            let bb = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
            let d_sigma = bb
                * sin_sigma
                * (cos_2sigma_m
                    + bb / 4.0
                        * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))
                            - bb / 6.0
                                * cos_2sigma_m
                                * (-3.0 + 4.0 * sin_sigma.powi(2))
                                * (-3.0 + 4.0 * cos_2sigma_m.powi(2))));

            return Ok(WGS84_B * aa * (sigma - d_sigma));
        }
    }

    Err(DistanceError::VincentyConvergence)
}

// ── Euclidean distances ─────────────────────────────────────────────────────

/// Euclidean distance in 2D.
pub fn euclidean_2d(a: Point2D, b: Point2D) -> f64 {
    ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt()
}

/// Euclidean distance in 3D.
pub fn euclidean_3d(a: Point3D, b: Point3D) -> f64 {
    ((b.x - a.x).powi(2) + (b.y - a.y).powi(2) + (b.z - a.z).powi(2)).sqrt()
}

/// Squared Euclidean distance in 2D (avoids the sqrt for comparison purposes).
pub fn euclidean_2d_sq(a: Point2D, b: Point2D) -> f64 {
    (b.x - a.x).powi(2) + (b.y - a.y).powi(2)
}

// ── Bearing ─────────────────────────────────────────────────────────────────

/// Compute the initial bearing (forward azimuth) from `a` to `b` in degrees [0, 360).
pub fn initial_bearing(a: GeoPoint, b: GeoPoint) -> f64 {
    let d_lon = b.lon_rad() - a.lon_rad();
    let y = d_lon.sin() * b.lat_rad().cos();
    let x =
        a.lat_rad().cos() * b.lat_rad().sin() - a.lat_rad().sin() * b.lat_rad().cos() * d_lon.cos();
    (y.atan2(x).to_degrees() + 360.0) % 360.0
}

/// Compute the final bearing from `a` to `b` in degrees [0, 360).
/// This is the bearing at the destination (the reverse of the initial bearing from b to a + 180°).
pub fn final_bearing(a: GeoPoint, b: GeoPoint) -> f64 {
    (initial_bearing(b, a) + 180.0) % 360.0
}

// ── Geographic midpoint ─────────────────────────────────────────────────────

/// Compute the geographic midpoint between two points on a sphere.
pub fn geographic_midpoint(a: GeoPoint, b: GeoPoint) -> GeoPoint {
    let d_lon = b.lon_rad() - a.lon_rad();
    let bx = b.lat_rad().cos() * d_lon.cos();
    let by = b.lat_rad().cos() * d_lon.sin();
    let lat = (a.lat_rad().sin() + b.lat_rad().sin())
        .atan2(((a.lat_rad().cos() + bx).powi(2) + by.powi(2)).sqrt());
    let lon = a.lon_rad() + by.atan2(a.lat_rad().cos() + bx);
    GeoPoint::new(lat.to_degrees(), lon.to_degrees())
}

/// Compute the geographic midpoint of a sequence of points (centroid on sphere).
pub fn geographic_centroid(points: &[GeoPoint]) -> Option<GeoPoint> {
    if points.is_empty() {
        return None;
    }
    let mut x = 0.0_f64;
    let mut y = 0.0_f64;
    let mut z = 0.0_f64;
    for p in points {
        let lat = p.lat_rad();
        let lon = p.lon_rad();
        x += lat.cos() * lon.cos();
        y += lat.cos() * lon.sin();
        z += lat.sin();
    }
    let n = points.len() as f64;
    x /= n;
    y /= n;
    z /= n;
    let lon = y.atan2(x).to_degrees();
    let hyp = (x * x + y * y).sqrt();
    let lat = z.atan2(hyp).to_degrees();
    Some(GeoPoint::new(lat, lon))
}

// ── Polyline distance ───────────────────────────────────────────────────────

/// Total Haversine distance along a geographic polyline (sum of segments).
pub fn polyline_distance_haversine(points: &[GeoPoint]) -> Result<f64, DistanceError> {
    if points.len() < 2 {
        return Err(DistanceError::InsufficientPoints);
    }
    let total = points.windows(2).map(|w| haversine(w[0], w[1])).sum();
    Ok(total)
}

/// Total Vincenty distance along a geographic polyline (sum of segments).
pub fn polyline_distance_vincenty(points: &[GeoPoint]) -> Result<f64, DistanceError> {
    if points.len() < 2 {
        return Err(DistanceError::InsufficientPoints);
    }
    let mut total = 0.0;
    for w in points.windows(2) {
        total += vincenty(w[0], w[1])?;
    }
    Ok(total)
}

/// Total Euclidean distance along a 2D polyline.
pub fn polyline_distance_2d(points: &[Point2D]) -> Result<f64, DistanceError> {
    if points.len() < 2 {
        return Err(DistanceError::InsufficientPoints);
    }
    let total = points.windows(2).map(|w| euclidean_2d(w[0], w[1])).sum();
    Ok(total)
}

// ── Point to segment distance ───────────────────────────────────────────────

/// Shortest Euclidean distance from a 2D point `p` to the line segment `(a, b)`.
pub fn point_to_segment_2d(p: Point2D, a: Point2D, b: Point2D) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;

    // degenerate segment (a == b)
    if len_sq < 1e-15 {
        return euclidean_2d(p, a);
    }

    // project p onto the line: t = dot(p-a, b-a) / |b-a|²
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);

    let proj = Point2D::new(a.x + t_clamped * dx, a.y + t_clamped * dy);
    euclidean_2d(p, proj)
}

/// Shortest Euclidean distance from a 2D point `p` to a polyline.
pub fn point_to_polyline_2d(p: Point2D, polyline: &[Point2D]) -> Result<f64, DistanceError> {
    if polyline.len() < 2 {
        return Err(DistanceError::InsufficientPoints);
    }
    let min_dist = polyline
        .windows(2)
        .map(|w| point_to_segment_2d(p, w[0], w[1]))
        .fold(f64::INFINITY, f64::min);
    Ok(min_dist)
}

/// Cross-track distance: shortest distance from a geographic point to the great-circle
/// arc defined by two other points (in metres, can be negative for left-side).
pub fn cross_track_distance(p: GeoPoint, arc_a: GeoPoint, arc_b: GeoPoint) -> f64 {
    const R: f64 = 6_371_000.0;
    let d_ap = haversine(arc_a, p) / R;
    let bearing_ap = initial_bearing(arc_a, p).to_radians();
    let bearing_ab = initial_bearing(arc_a, arc_b).to_radians();
    let xt = (d_ap.sin() * (bearing_ap - bearing_ab).sin()).asin();
    xt * R
}

// ── Bounding box expansion ──────────────────────────────────────────────────

/// Expand a bounding box by a uniform projected distance on all sides.
pub fn expand_bbox(bbox: BBox, distance: f64) -> BBox {
    BBox::new(
        bbox.min_x - distance,
        bbox.min_y - distance,
        bbox.max_x + distance,
        bbox.max_y + distance,
    )
}

/// Expand a geographic bounding box (lat/lon degrees) by a distance in metres.
///
/// Uses a rough conversion: 1° latitude ≈ 111,320 m, 1° longitude ≈ 111,320 * cos(lat) m.
pub fn expand_bbox_geodesic(bbox: BBox, distance_m: f64) -> BBox {
    const DEG_PER_METRE_LAT: f64 = 1.0 / 111_320.0;
    let mid_lat = ((bbox.min_y + bbox.max_y) / 2.0).to_radians();
    let cos_lat = mid_lat.cos().max(1e-10);
    let deg_per_metre_lon = DEG_PER_METRE_LAT / cos_lat;

    let d_lat = distance_m * DEG_PER_METRE_LAT;
    let d_lon = distance_m * deg_per_metre_lon;
    BBox::new(
        bbox.min_x - d_lon,
        bbox.min_y - d_lat,
        bbox.max_x + d_lon,
        bbox.max_y + d_lat,
    )
}

// ── Destination point given bearing and distance ────────────────────────────

/// Compute the destination point given a start point, initial bearing (degrees),
/// and great-circle distance (metres) on a sphere of radius R.
pub fn destination_point(start: GeoPoint, bearing_deg: f64, distance_m: f64) -> GeoPoint {
    const R: f64 = 6_371_000.0;
    let d = distance_m / R;
    let brng = bearing_deg.to_radians();
    let lat1 = start.lat_rad();
    let lon1 = start.lon_rad();
    let lat2 = (lat1.sin() * d.cos() + lat1.cos() * d.sin() * brng.cos()).asin();
    let lon2 = lon1 + (brng.sin() * d.sin() * lat1.cos()).atan2(d.cos() - lat1.sin() * lat2.sin());
    GeoPoint::new(lat2.to_degrees(), lon2.to_degrees())
}

// ── Rhumb-line distance ─────────────────────────────────────────────────────

/// Rhumb-line (loxodrome) distance in metres on a sphere.
pub fn rhumb_distance(a: GeoPoint, b: GeoPoint) -> f64 {
    const R: f64 = 6_371_000.0;
    let d_lat = b.lat_rad() - a.lat_rad();
    let d_lon = (b.lon_rad() - a.lon_rad()).abs();
    let d_psi = ((b.lat_rad() / 2.0 + PI / 4.0).tan() / (a.lat_rad() / 2.0 + PI / 4.0).tan()).ln();
    let q = if d_psi.abs() > 1e-12 {
        d_lat / d_psi
    } else {
        a.lat_rad().cos()
    };
    // Correct for crossing the 180° meridian
    let d_lon = if d_lon > PI { 2.0 * PI - d_lon } else { d_lon };

    (d_lat * d_lat + q * q * d_lon * d_lon).sqrt() * R
}

/// Rhumb-line bearing from `a` to `b` in degrees [0, 360).
pub fn rhumb_bearing(a: GeoPoint, b: GeoPoint) -> f64 {
    let d_lon = b.lon_rad() - a.lon_rad();
    let d_psi = ((b.lat_rad() / 2.0 + PI / 4.0).tan() / (a.lat_rad() / 2.0 + PI / 4.0).tan()).ln();
    (d_lon.atan2(d_psi).to_degrees() + 360.0) % 360.0
}

// ── Unit conversion helpers ─────────────────────────────────────────────────

/// Convert metres to kilometres.
pub fn metres_to_km(m: f64) -> f64 {
    m / 1000.0
}

/// Convert metres to statute miles.
pub fn metres_to_miles(m: f64) -> f64 {
    m / 1609.344
}

/// Convert metres to nautical miles.
pub fn metres_to_nautical_miles(m: f64) -> f64 {
    m / 1852.0
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1.0; // 1-metre tolerance for geodesic distances
    const EPSILON_DEG: f64 = 0.01; // 0.01-degree tolerance for angles

    // ── Haversine tests ─────────────────────────────────────────────────────

    #[test]
    fn test_haversine_same_point() {
        let p = GeoPoint::new(51.5074, -0.1278); // London
        let d = haversine(p, p);
        assert!(
            d.abs() < EPSILON,
            "Same point should give 0 distance, got {d}"
        );
    }

    #[test]
    fn test_haversine_london_to_paris() {
        let london = GeoPoint::new(51.5074, -0.1278);
        let paris = GeoPoint::new(48.8566, 2.3522);
        let d = haversine(london, paris);
        // Expected ~343 km
        assert!(
            (d - 343_557.0).abs() < 5000.0,
            "London→Paris ≈ 343 km, got {:.0} m",
            d
        );
    }

    #[test]
    fn test_haversine_antipodal() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 180.0);
        let d = haversine(a, b);
        let half_circum = PI * 6_371_000.0;
        assert!(
            (d - half_circum).abs() < 100.0,
            "Antipodal ≈ half circumference, got {d:.0}"
        );
    }

    #[test]
    fn test_haversine_equator() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 1.0);
        let d = haversine(a, b);
        // 1° on equator ≈ 111,195 m
        assert!(
            (d - 111_195.0).abs() < 500.0,
            "1° equator ≈ 111 km, got {d:.0} m"
        );
    }

    #[test]
    fn test_haversine_symmetry() {
        let a = GeoPoint::new(35.0, 139.0);
        let b = GeoPoint::new(-33.0, 151.0);
        let d1 = haversine(a, b);
        let d2 = haversine(b, a);
        assert!((d1 - d2).abs() < 0.01, "Haversine should be symmetric");
    }

    // ── Vincenty tests ──────────────────────────────────────────────────────

    #[test]
    fn test_vincenty_same_point() {
        let p = GeoPoint::new(48.8566, 2.3522);
        let d = vincenty(p, p).expect("Should succeed for same point");
        assert!(d < 0.01, "Same point Vincenty = 0, got {d}");
    }

    #[test]
    fn test_vincenty_london_paris() {
        let london = GeoPoint::new(51.5074, -0.1278);
        let paris = GeoPoint::new(48.8566, 2.3522);
        let d = vincenty(london, paris).expect("Should converge");
        assert!(
            (d - 343_557.0).abs() < 5000.0,
            "London→Paris Vincenty ≈ 343 km, got {d:.0}"
        );
    }

    #[test]
    fn test_vincenty_converges_long_distance() {
        let nyc = GeoPoint::new(40.7128, -74.0060);
        let tokyo = GeoPoint::new(35.6762, 139.6503);
        let d = vincenty(nyc, tokyo).expect("Should converge");
        // ~10,860 km
        assert!(
            d > 10_000_000.0 && d < 11_500_000.0,
            "NYC→Tokyo ≈ 10860 km, got {d:.0}"
        );
    }

    #[test]
    fn test_vincenty_vs_haversine() {
        let a = GeoPoint::new(52.2296, 21.0122); // Warsaw
        let b = GeoPoint::new(41.0082, 28.9784); // Istanbul
        let hav = haversine(a, b);
        let vin = vincenty(a, b).expect("Should converge");
        // Should agree within ~0.5%
        let diff_pct = ((hav - vin) / vin * 100.0).abs();
        assert!(
            diff_pct < 0.5,
            "Haversine and Vincenty should agree within 0.5%, diff = {diff_pct:.3}%"
        );
    }

    // ── Euclidean tests ─────────────────────────────────────────────────────

    #[test]
    fn test_euclidean_2d_zero() {
        let p = Point2D::new(3.0, 4.0);
        assert!((euclidean_2d(p, p)).abs() < 1e-15);
    }

    #[test]
    fn test_euclidean_2d_unit() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        let d = euclidean_2d(a, b);
        assert!((d - 5.0).abs() < 1e-10, "3-4-5 triangle, got {d}");
    }

    #[test]
    fn test_euclidean_3d_unit() {
        let a = Point3D::new(0.0, 0.0, 0.0);
        let b = Point3D::new(1.0, 2.0, 2.0);
        let d = euclidean_3d(a, b);
        assert!((d - 3.0).abs() < 1e-10, "1²+2²+2²=9, √9=3, got {d}");
    }

    #[test]
    fn test_euclidean_2d_sq() {
        let a = Point2D::new(1.0, 1.0);
        let b = Point2D::new(4.0, 5.0);
        let sq = euclidean_2d_sq(a, b);
        assert!((sq - 25.0).abs() < 1e-10, "9+16=25, got {sq}");
    }

    #[test]
    fn test_euclidean_3d_negative() {
        let a = Point3D::new(-1.0, -2.0, -3.0);
        let b = Point3D::new(1.0, 2.0, 3.0);
        let d = euclidean_3d(a, b);
        let expected = (4.0 + 16.0 + 36.0_f64).sqrt();
        assert!((d - expected).abs() < 1e-10);
    }

    // ── Bearing tests ───────────────────────────────────────────────────────

    #[test]
    fn test_bearing_north() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(1.0, 0.0);
        let brng = initial_bearing(a, b);
        assert!(
            (brng - 0.0).abs() < EPSILON_DEG || (brng - 360.0).abs() < EPSILON_DEG,
            "Due north bearing ≈ 0°, got {brng:.2}"
        );
    }

    #[test]
    fn test_bearing_east() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 1.0);
        let brng = initial_bearing(a, b);
        assert!(
            (brng - 90.0).abs() < EPSILON_DEG,
            "Due east bearing ≈ 90°, got {brng:.2}"
        );
    }

    #[test]
    fn test_bearing_south() {
        let a = GeoPoint::new(1.0, 0.0);
        let b = GeoPoint::new(0.0, 0.0);
        let brng = initial_bearing(a, b);
        assert!(
            (brng - 180.0).abs() < EPSILON_DEG,
            "Due south bearing ≈ 180°, got {brng:.2}"
        );
    }

    #[test]
    fn test_final_bearing() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 1.0);
        let fb = final_bearing(a, b);
        // On the equator, final bearing should also be ~90°
        assert!(
            (fb - 90.0).abs() < EPSILON_DEG,
            "Final bearing east on equator ≈ 90°, got {fb:.2}"
        );
    }

    // ── Midpoint tests ──────────────────────────────────────────────────────

    #[test]
    fn test_midpoint_equator() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 10.0);
        let mid = geographic_midpoint(a, b);
        assert!(
            mid.lat.abs() < 0.01,
            "Midpoint lat ≈ 0°, got {:.4}",
            mid.lat
        );
        assert!(
            (mid.lon - 5.0).abs() < 0.01,
            "Midpoint lon ≈ 5°, got {:.4}",
            mid.lon
        );
    }

    #[test]
    fn test_midpoint_same_point() {
        let p = GeoPoint::new(45.0, 90.0);
        let mid = geographic_midpoint(p, p);
        assert!((mid.lat - 45.0).abs() < 0.01);
        assert!((mid.lon - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_centroid_single() {
        let pts = vec![GeoPoint::new(10.0, 20.0)];
        let c = geographic_centroid(&pts).expect("Should return centroid");
        assert!((c.lat - 10.0).abs() < 0.01);
        assert!((c.lon - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_centroid_empty() {
        let pts: Vec<GeoPoint> = vec![];
        assert!(geographic_centroid(&pts).is_none());
    }

    #[test]
    fn test_centroid_symmetric() {
        let pts = vec![GeoPoint::new(0.0, -10.0), GeoPoint::new(0.0, 10.0)];
        let c = geographic_centroid(&pts).expect("Should return centroid");
        assert!(c.lat.abs() < 0.5, "Centroid lat ≈ 0°, got {:.4}", c.lat);
        assert!(c.lon.abs() < 0.5, "Centroid lon ≈ 0°, got {:.4}", c.lon);
    }

    // ── Polyline distance tests ─────────────────────────────────────────────

    #[test]
    fn test_polyline_haversine_two_points() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 1.0);
        let d = polyline_distance_haversine(&[a, b]).expect("Should succeed");
        assert!((d - haversine(a, b)).abs() < 0.01);
    }

    #[test]
    fn test_polyline_haversine_three_points() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 1.0);
        let c = GeoPoint::new(0.0, 2.0);
        let total = polyline_distance_haversine(&[a, b, c]).expect("Should succeed");
        let expected = haversine(a, b) + haversine(b, c);
        assert!((total - expected).abs() < 0.01);
    }

    #[test]
    fn test_polyline_insufficient_points() {
        let p = GeoPoint::new(0.0, 0.0);
        assert!(polyline_distance_haversine(&[p]).is_err());
        assert!(polyline_distance_haversine(&[]).is_err());
    }

    #[test]
    fn test_polyline_2d_rectangle() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(3.0, 4.0),
        ];
        let d = polyline_distance_2d(&pts).expect("ok");
        assert!((d - 7.0).abs() < 1e-10, "3+4=7, got {d}");
    }

    // ── Point to segment tests ──────────────────────────────────────────────

    #[test]
    fn test_point_to_segment_perpendicular() {
        let p = Point2D::new(1.0, 1.0);
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(2.0, 0.0);
        let d = point_to_segment_2d(p, a, b);
        assert!(
            (d - 1.0).abs() < 1e-10,
            "Perpendicular distance = 1, got {d}"
        );
    }

    #[test]
    fn test_point_to_segment_endpoint_a() {
        let p = Point2D::new(-1.0, 0.0);
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(2.0, 0.0);
        let d = point_to_segment_2d(p, a, b);
        assert!(
            (d - 1.0).abs() < 1e-10,
            "Nearest to A, distance = 1, got {d}"
        );
    }

    #[test]
    fn test_point_to_segment_endpoint_b() {
        let p = Point2D::new(3.0, 0.0);
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(2.0, 0.0);
        let d = point_to_segment_2d(p, a, b);
        assert!(
            (d - 1.0).abs() < 1e-10,
            "Nearest to B, distance = 1, got {d}"
        );
    }

    #[test]
    fn test_point_to_segment_degenerate() {
        let p = Point2D::new(3.0, 4.0);
        let a = Point2D::new(0.0, 0.0);
        let d = point_to_segment_2d(p, a, a);
        assert!((d - 5.0).abs() < 1e-10, "Degenerate segment, got {d}");
    }

    #[test]
    fn test_point_to_polyline() {
        let p = Point2D::new(1.0, 1.0);
        let poly = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 2.0),
        ];
        let d = point_to_polyline_2d(p, &poly).expect("ok");
        // Nearest to segment (0,0)-(2,0) at perpendicular distance 1
        assert!((d - 1.0).abs() < 1e-10, "Min distance = 1, got {d}");
    }

    // ── Bounding box expansion tests ────────────────────────────────────────

    #[test]
    fn test_expand_bbox_uniform() {
        let bbox = BBox::new(10.0, 20.0, 30.0, 40.0);
        let expanded = expand_bbox(bbox, 5.0);
        assert!((expanded.min_x - 5.0).abs() < 1e-10);
        assert!((expanded.min_y - 15.0).abs() < 1e-10);
        assert!((expanded.max_x - 35.0).abs() < 1e-10);
        assert!((expanded.max_y - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_expand_bbox_geodesic_equator() {
        let bbox = BBox::new(-1.0, -1.0, 1.0, 1.0);
        let expanded = expand_bbox_geodesic(bbox, 111_320.0); // ~1° at equator
                                                              // min_y should be approximately -2.0 (within tolerance)
        assert!(
            (expanded.min_y - (-2.0)).abs() < 0.1,
            "min_y ≈ -2.0, got {:.4}",
            expanded.min_y
        );
        assert!(
            (expanded.max_y - 2.0).abs() < 0.1,
            "max_y ≈ 2.0, got {:.4}",
            expanded.max_y
        );
    }

    #[test]
    fn test_bbox_centre() {
        let bbox = BBox::new(0.0, 0.0, 10.0, 20.0);
        let c = bbox.centre();
        assert!((c.x - 5.0).abs() < 1e-10);
        assert!((c.y - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_width_height() {
        let bbox = BBox::new(1.0, 2.0, 5.0, 8.0);
        assert!((bbox.width() - 4.0).abs() < 1e-10);
        assert!((bbox.height() - 6.0).abs() < 1e-10);
    }

    // ── Destination point tests ─────────────────────────────────────────────

    #[test]
    fn test_destination_north() {
        let start = GeoPoint::new(0.0, 0.0);
        let dest = destination_point(start, 0.0, 111_320.0);
        // ~1° north
        assert!(
            (dest.lat - 1.0).abs() < 0.1,
            "1° north ≈ lat 1.0, got {:.4}",
            dest.lat
        );
        assert!(dest.lon.abs() < 0.01);
    }

    #[test]
    fn test_destination_east() {
        let start = GeoPoint::new(0.0, 0.0);
        let dest = destination_point(start, 90.0, 111_320.0);
        assert!(start.lat.abs() < 0.1);
        assert!(
            (dest.lon - 1.0).abs() < 0.1,
            "1° east ≈ lon 1.0, got {:.4}",
            dest.lon
        );
    }

    // ── Rhumb line tests ────────────────────────────────────────────────────

    #[test]
    fn test_rhumb_same_point() {
        let p = GeoPoint::new(30.0, 40.0);
        let d = rhumb_distance(p, p);
        assert!(d < 1.0, "Same-point rhumb = 0, got {d}");
    }

    #[test]
    fn test_rhumb_equator() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 1.0);
        let d = rhumb_distance(a, b);
        // On equator, rhumb == great circle
        let h = haversine(a, b);
        assert!(
            (d - h).abs() < 100.0,
            "Equator rhumb ≈ haversine, rhumb={d:.0}, hav={h:.0}"
        );
    }

    #[test]
    fn test_rhumb_bearing_north() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(1.0, 0.0);
        let brng = rhumb_bearing(a, b);
        assert!(
            brng.abs() < 0.5 || (brng - 360.0).abs() < 0.5,
            "Due-north rhumb bearing ≈ 0°, got {brng:.2}"
        );
    }

    // ── Cross-track distance tests ──────────────────────────────────────────

    #[test]
    fn test_cross_track_on_arc() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 10.0);
        let p = GeoPoint::new(0.0, 5.0); // on the arc
        let xt = cross_track_distance(p, a, b);
        assert!(xt.abs() < 100.0, "On-arc cross-track ≈ 0, got {xt:.0}");
    }

    #[test]
    fn test_cross_track_offset() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 10.0);
        let p = GeoPoint::new(1.0, 5.0); // ~1° north of the arc
        let xt = cross_track_distance(p, a, b);
        // Should be approximately 111 km north
        assert!(
            xt.abs() > 100_000.0 && xt.abs() < 120_000.0,
            "1° offset ≈ 111 km, got {xt:.0}"
        );
    }

    // ── Unit conversion tests ───────────────────────────────────────────────

    #[test]
    fn test_metres_to_km() {
        assert!((metres_to_km(1000.0) - 1.0).abs() < 1e-10);
        assert!((metres_to_km(42_195.0) - 42.195).abs() < 1e-10);
    }

    #[test]
    fn test_metres_to_miles() {
        let m = metres_to_miles(1609.344);
        assert!((m - 1.0).abs() < 1e-6, "1609.344 m = 1 mile, got {m}");
    }

    #[test]
    fn test_metres_to_nautical_miles() {
        let nm = metres_to_nautical_miles(1852.0);
        assert!((nm - 1.0).abs() < 1e-6, "1852 m = 1 NM, got {nm}");
    }

    // ── Vincenty polyline tests ─────────────────────────────────────────────

    #[test]
    fn test_vincenty_polyline_two_points() {
        let a = GeoPoint::new(0.0, 0.0);
        let b = GeoPoint::new(0.0, 1.0);
        let poly_d = polyline_distance_vincenty(&[a, b]).expect("ok");
        let single_d = vincenty(a, b).expect("ok");
        assert!((poly_d - single_d).abs() < 0.01);
    }

    #[test]
    fn test_polyline_vincenty_insufficient() {
        assert!(polyline_distance_vincenty(&[GeoPoint::new(0.0, 0.0)]).is_err());
    }
}
