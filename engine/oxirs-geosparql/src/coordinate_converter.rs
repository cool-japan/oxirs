//! Coordinate system conversions for GeoSPARQL.
//!
//! Provides conversions between WGS84 geographic coordinates (EPSG:4326),
//! Web Mercator projected coordinates (EPSG:3857), and a simple 2D Cartesian
//! plane, along with Haversine great-circle distance computation.

use std::f64::consts::PI;

// ── Constants ─────────────────────────────────────────────────────────────────

/// WGS84 / Web Mercator equatorial Earth radius in metres.
const EARTH_RADIUS_M: f64 = 6_378_137.0;

// ── Crs ───────────────────────────────────────────────────────────────────────

/// Supported coordinate reference systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Crs {
    /// WGS84 geographic coordinates (EPSG:4326).
    ///
    /// `x` = longitude in degrees (−180 … +180),
    /// `y` = latitude  in degrees (−90  … +90).
    Wgs84,
    /// Web Mercator (EPSG:3857).
    ///
    /// `x` and `y` are easting and northing in metres, respectively.
    WebMercator,
    /// Simple 2D Cartesian plane (no geodetic datum).
    Cartesian,
}

// ── CoordPoint ────────────────────────────────────────────────────────────────

/// A coordinate point with CRS information.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordPoint {
    /// First axis: longitude (Wgs84), easting (WebMercator), or X (Cartesian).
    pub x: f64,
    /// Second axis: latitude (Wgs84), northing (WebMercator), or Y (Cartesian).
    pub y: f64,
    /// The coordinate reference system this point is expressed in.
    pub crs: Crs,
}

impl CoordPoint {
    /// Construct a new `CoordPoint` with explicit CRS.
    pub fn new(x: f64, y: f64, crs: Crs) -> Self {
        CoordPoint { x, y, crs }
    }

    /// Convenience constructor for a WGS84 lon/lat pair.
    pub fn wgs84(lon: f64, lat: f64) -> Self {
        CoordPoint {
            x: lon,
            y: lat,
            crs: Crs::Wgs84,
        }
    }

    /// Convenience constructor for a Web Mercator x/y pair (metres).
    pub fn web_mercator(x: f64, y: f64) -> Self {
        CoordPoint {
            x,
            y,
            crs: Crs::WebMercator,
        }
    }

    /// Convenience constructor for a 2D Cartesian point.
    pub fn cartesian(x: f64, y: f64) -> Self {
        CoordPoint {
            x,
            y,
            crs: Crs::Cartesian,
        }
    }
}

// ── CoordinateConverter ───────────────────────────────────────────────────────

/// Stateless converter between coordinate reference systems.
///
/// # Supported conversions
/// | From         | To           |
/// |--------------|--------------|
/// | WGS84        | WebMercator  |
/// | WebMercator  | WGS84        |
/// | Any CRS      | Same CRS     | ← identity
///
/// Conversions involving `Cartesian` as a *source* CRS (other than
/// identity) are not defined and return `Err`.
pub struct CoordinateConverter;

impl Default for CoordinateConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordinateConverter {
    /// Create a new `CoordinateConverter`.
    pub fn new() -> Self {
        CoordinateConverter
    }

    // ── Static projection formulas ──────────────────────────────────────────

    /// Convert WGS84 longitude/latitude (degrees) to Web Mercator x/y (metres).
    ///
    /// Formula (spherical Mercator on the WGS84 sphere of radius R):
    /// ```text
    /// x = lon × (π/180) × R
    /// y = ln(tan(π/4 + lat × π/360)) × R
    /// ```
    pub fn wgs84_to_web_mercator(lon: f64, lat: f64) -> CoordPoint {
        let x = lon * (PI / 180.0) * EARTH_RADIUS_M;
        let lat_rad = lat * PI / 360.0; // lat * π / 360 = lat_deg * (π/180) / 2
        let y = (PI / 4.0 + lat_rad).tan().ln() * EARTH_RADIUS_M;
        CoordPoint::web_mercator(x, y)
    }

    /// Convert Web Mercator x/y (metres) to WGS84 longitude/latitude (degrees).
    ///
    /// Formula (inverse spherical Mercator):
    /// ```text
    /// lon = x / R × (180/π)
    /// lat = atan(exp(y / R)) × (360/π) − 90
    /// ```
    pub fn web_mercator_to_wgs84(x: f64, y: f64) -> CoordPoint {
        let lon = x / EARTH_RADIUS_M * (180.0 / PI);
        let lat = (y / EARTH_RADIUS_M).exp().atan() * (360.0 / PI) - 90.0;
        CoordPoint::wgs84(lon, lat)
    }

    // ── Generic convert ────────────────────────────────────────────────────

    /// Convert `point` to the `target` CRS.
    ///
    /// Returns `Err` if the conversion is not supported (e.g. Cartesian → any).
    pub fn convert(&self, point: &CoordPoint, target: Crs) -> Result<CoordPoint, String> {
        if point.crs == target {
            return Ok(point.clone());
        }
        match (point.crs, target) {
            (Crs::Wgs84, Crs::WebMercator) => Ok(Self::wgs84_to_web_mercator(point.x, point.y)),
            (Crs::WebMercator, Crs::Wgs84) => Ok(Self::web_mercator_to_wgs84(point.x, point.y)),
            (Crs::Wgs84, Crs::Cartesian) => {
                // Treat as an identity-like pass-through with CRS re-label
                Ok(CoordPoint::cartesian(point.x, point.y))
            }
            (Crs::Cartesian, _) => Err(format!(
                "Conversion from Cartesian to {target:?} is not supported"
            )),
            (src, tgt) => Err(format!("Unsupported conversion from {src:?} to {tgt:?}")),
        }
    }

    // ── Haversine distance ─────────────────────────────────────────────────

    /// Compute the great-circle (Haversine) distance in metres between two
    /// WGS84 coordinate points.
    ///
    /// Both `p1` and `p2` must use [`Crs::Wgs84`]; otherwise `Err` is returned.
    ///
    /// # Formula
    /// ```text
    /// a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    /// c = 2 × atan2(√a, √(1−a))
    /// d = R × c
    /// ```
    pub fn haversine_distance(p1: &CoordPoint, p2: &CoordPoint) -> Result<f64, String> {
        if p1.crs != Crs::Wgs84 {
            return Err(format!("p1 must use Wgs84 CRS, got {:?}", p1.crs));
        }
        if p2.crs != Crs::Wgs84 {
            return Err(format!("p2 must use Wgs84 CRS, got {:?}", p2.crs));
        }
        let lat1 = p1.y.to_radians();
        let lat2 = p2.y.to_radians();
        let d_lat = (p2.y - p1.y).to_radians();
        let d_lon = (p2.x - p1.x).to_radians();

        let a = (d_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (d_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        Ok(EARTH_RADIUS_M * c)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn converter() -> CoordinateConverter {
        CoordinateConverter::new()
    }

    // ── CoordPoint constructors ────────────────────────────────────────────

    #[test]
    fn coord_point_new() {
        let p = CoordPoint::new(10.0, 20.0, Crs::Wgs84);
        assert_eq!(p.x, 10.0);
        assert_eq!(p.y, 20.0);
        assert_eq!(p.crs, Crs::Wgs84);
    }

    #[test]
    fn coord_point_wgs84_constructor() {
        let p = CoordPoint::wgs84(13.4050, 52.5200);
        assert_eq!(p.crs, Crs::Wgs84);
        assert_eq!(p.x, 13.4050);
        assert_eq!(p.y, 52.5200);
    }

    #[test]
    fn coord_point_web_mercator_constructor() {
        let p = CoordPoint::web_mercator(1_491_000.0, 6_890_000.0);
        assert_eq!(p.crs, Crs::WebMercator);
    }

    #[test]
    fn coord_point_cartesian_constructor() {
        let p = CoordPoint::cartesian(3.0, 4.0);
        assert_eq!(p.crs, Crs::Cartesian);
    }

    // ── wgs84_to_web_mercator ──────────────────────────────────────────────

    #[test]
    fn wgs84_to_web_mercator_origin() {
        // lon=0, lat=0 → x=0, y=0
        let p = CoordinateConverter::wgs84_to_web_mercator(0.0, 0.0);
        assert!(approx_eq(p.x, 0.0, 1e-6));
        assert!(approx_eq(p.y, 0.0, 1e-6));
        assert_eq!(p.crs, Crs::WebMercator);
    }

    #[test]
    fn wgs84_to_web_mercator_positive_quadrant() {
        // London: ~(−0.1276, 51.5074) → roughly (−14_208, 6_711_700)
        let p = CoordinateConverter::wgs84_to_web_mercator(-0.1276, 51.5074);
        assert!(approx_eq(p.x, -14_208.0, 200.0));
        assert!(approx_eq(p.y, 6_711_700.0, 1000.0));
    }

    #[test]
    fn wgs84_to_web_mercator_x_proportional_to_lon() {
        // x should be exactly lon × (π/180) × R
        let p = CoordinateConverter::wgs84_to_web_mercator(1.0, 0.0);
        let expected_x = (PI / 180.0) * EARTH_RADIUS_M;
        assert!(approx_eq(p.x, expected_x, 1e-3));
    }

    #[test]
    fn wgs84_to_web_mercator_negative_lon() {
        let p = CoordinateConverter::wgs84_to_web_mercator(-90.0, 0.0);
        let expected_x = -90.0 * (PI / 180.0) * EARTH_RADIUS_M;
        assert!(approx_eq(p.x, expected_x, 1e-3));
    }

    #[test]
    fn wgs84_to_web_mercator_crs_tag() {
        let p = CoordinateConverter::wgs84_to_web_mercator(10.0, 45.0);
        assert_eq!(p.crs, Crs::WebMercator);
    }

    // ── web_mercator_to_wgs84 ──────────────────────────────────────────────

    #[test]
    fn web_mercator_to_wgs84_origin() {
        let p = CoordinateConverter::web_mercator_to_wgs84(0.0, 0.0);
        assert!(approx_eq(p.x, 0.0, 1e-9));
        assert!(approx_eq(p.y, 0.0, 1e-9));
        assert_eq!(p.crs, Crs::Wgs84);
    }

    #[test]
    fn web_mercator_to_wgs84_crs_tag() {
        let p = CoordinateConverter::web_mercator_to_wgs84(1_000_000.0, 2_000_000.0);
        assert_eq!(p.crs, Crs::Wgs84);
    }

    // ── Round-trip WGS84 → WebMercator → WGS84 ────────────────────────────

    #[test]
    fn round_trip_origin() {
        let fwd = CoordinateConverter::wgs84_to_web_mercator(0.0, 0.0);
        let back = CoordinateConverter::web_mercator_to_wgs84(fwd.x, fwd.y);
        assert!(approx_eq(back.x, 0.0, 1e-6));
        assert!(approx_eq(back.y, 0.0, 1e-6));
    }

    #[test]
    fn round_trip_london() {
        let lon = -0.1276_f64;
        let lat = 51.5074_f64;
        let fwd = CoordinateConverter::wgs84_to_web_mercator(lon, lat);
        let back = CoordinateConverter::web_mercator_to_wgs84(fwd.x, fwd.y);
        assert!(approx_eq(back.x, lon, 1e-6));
        assert!(approx_eq(back.y, lat, 1e-6));
    }

    #[test]
    fn round_trip_new_york() {
        let lon = -74.006_f64;
        let lat = 40.7128_f64;
        let fwd = CoordinateConverter::wgs84_to_web_mercator(lon, lat);
        let back = CoordinateConverter::web_mercator_to_wgs84(fwd.x, fwd.y);
        assert!(approx_eq(back.x, lon, 1e-6));
        assert!(approx_eq(back.y, lat, 1e-6));
    }

    #[test]
    fn round_trip_tokyo() {
        let lon = 139.6917_f64;
        let lat = 35.6895_f64;
        let fwd = CoordinateConverter::wgs84_to_web_mercator(lon, lat);
        let back = CoordinateConverter::web_mercator_to_wgs84(fwd.x, fwd.y);
        assert!(approx_eq(back.x, lon, 1e-6));
        assert!(approx_eq(back.y, lat, 1e-6));
    }

    #[test]
    fn round_trip_southern_hemisphere() {
        let lon = 151.2093_f64;
        let lat = -33.8688_f64;
        let fwd = CoordinateConverter::wgs84_to_web_mercator(lon, lat);
        let back = CoordinateConverter::web_mercator_to_wgs84(fwd.x, fwd.y);
        assert!(approx_eq(back.x, lon, 1e-6));
        assert!(approx_eq(back.y, lat, 1e-6));
    }

    // ── convert (generic) ─────────────────────────────────────────────────

    #[test]
    fn convert_identity_wgs84() {
        let p = CoordPoint::wgs84(10.0, 20.0);
        let out = converter().convert(&p, Crs::Wgs84).expect("identity");
        assert_eq!(out, p);
    }

    #[test]
    fn convert_identity_web_mercator() {
        let p = CoordPoint::web_mercator(1_000_000.0, 2_000_000.0);
        let out = converter().convert(&p, Crs::WebMercator).expect("identity");
        assert_eq!(out, p);
    }

    #[test]
    fn convert_identity_cartesian() {
        let p = CoordPoint::cartesian(5.0, 7.0);
        let out = converter().convert(&p, Crs::Cartesian).expect("identity");
        assert_eq!(out, p);
    }

    #[test]
    fn convert_wgs84_to_web_mercator() {
        let p = CoordPoint::wgs84(0.0, 0.0);
        let out = converter().convert(&p, Crs::WebMercator).expect("ok");
        assert!(approx_eq(out.x, 0.0, 1e-6));
        assert!(approx_eq(out.y, 0.0, 1e-6));
        assert_eq!(out.crs, Crs::WebMercator);
    }

    #[test]
    fn convert_web_mercator_to_wgs84() {
        let p = CoordPoint::web_mercator(0.0, 0.0);
        let out = converter().convert(&p, Crs::Wgs84).expect("ok");
        assert!(approx_eq(out.x, 0.0, 1e-6));
        assert!(approx_eq(out.y, 0.0, 1e-6));
        assert_eq!(out.crs, Crs::Wgs84);
    }

    #[test]
    fn convert_cartesian_to_wgs84_returns_err() {
        let p = CoordPoint::cartesian(1.0, 2.0);
        let result = converter().convert(&p, Crs::Wgs84);
        assert!(result.is_err());
    }

    #[test]
    fn convert_cartesian_to_web_mercator_returns_err() {
        let p = CoordPoint::cartesian(1.0, 2.0);
        let result = converter().convert(&p, Crs::WebMercator);
        assert!(result.is_err());
    }

    // ── haversine_distance ─────────────────────────────────────────────────

    #[test]
    fn haversine_same_point_is_zero() {
        let p = CoordPoint::wgs84(10.0, 48.0);
        let d = CoordinateConverter::haversine_distance(&p, &p).expect("ok");
        assert!(approx_eq(d, 0.0, 1e-6));
    }

    #[test]
    fn haversine_london_paris() {
        // London: (−0.1276, 51.5074), Paris: (2.3522, 48.8566)
        // Actual great-circle ≈ 341 km
        let london = CoordPoint::wgs84(-0.1276, 51.5074);
        let paris = CoordPoint::wgs84(2.3522, 48.8566);
        let d = CoordinateConverter::haversine_distance(&london, &paris).expect("ok");
        assert!(approx_eq(d, 341_600.0, 5_000.0), "Got {d}");
    }

    #[test]
    fn haversine_equator_one_degree() {
        // Along equator, 1° ≈ π/180 × 6,378,137 ≈ 111,319 m
        let p1 = CoordPoint::wgs84(0.0, 0.0);
        let p2 = CoordPoint::wgs84(1.0, 0.0);
        let d = CoordinateConverter::haversine_distance(&p1, &p2).expect("ok");
        assert!(approx_eq(d, 111_319.0, 200.0), "Got {d}");
    }

    #[test]
    fn haversine_antipodal_points() {
        // Antipodal points are ~half circumference ≈ 20,015 km
        let p1 = CoordPoint::wgs84(0.0, 0.0);
        let p2 = CoordPoint::wgs84(180.0, 0.0);
        let d = CoordinateConverter::haversine_distance(&p1, &p2).expect("ok");
        assert!(approx_eq(d, 20_015_000.0, 50_000.0), "Got {d}");
    }

    #[test]
    fn haversine_symmetric() {
        let a = CoordPoint::wgs84(13.4050, 52.5200);
        let b = CoordPoint::wgs84(2.3522, 48.8566);
        let d1 = CoordinateConverter::haversine_distance(&a, &b).expect("ok");
        let d2 = CoordinateConverter::haversine_distance(&b, &a).expect("ok");
        assert!(approx_eq(d1, d2, 1e-6));
    }

    #[test]
    fn haversine_requires_wgs84_p1() {
        let p1 = CoordPoint::web_mercator(0.0, 0.0);
        let p2 = CoordPoint::wgs84(0.0, 0.0);
        let r = CoordinateConverter::haversine_distance(&p1, &p2);
        assert!(r.is_err());
    }

    #[test]
    fn haversine_requires_wgs84_p2() {
        let p1 = CoordPoint::wgs84(0.0, 0.0);
        let p2 = CoordPoint::cartesian(0.0, 0.0);
        let r = CoordinateConverter::haversine_distance(&p1, &p2);
        assert!(r.is_err());
    }

    // ── Default impl ──────────────────────────────────────────────────────

    #[test]
    fn coordinate_converter_default() {
        let _c: CoordinateConverter = Default::default();
    }

    // ── Boundary conditions ───────────────────────────────────────────────

    #[test]
    fn wgs84_max_lon() {
        // lon=180 → x = 180 × π/180 × R = π × R
        let p = CoordinateConverter::wgs84_to_web_mercator(180.0, 0.0);
        let expected = PI * EARTH_RADIUS_M;
        assert!(approx_eq(p.x, expected, 1e-3));
    }

    #[test]
    fn wgs84_negative_lat() {
        // lat<0 → y<0
        let p = CoordinateConverter::wgs84_to_web_mercator(0.0, -45.0);
        assert!(p.y < 0.0);
    }

    // ── Additional haversine tests ─────────────────────────────────────────

    #[test]
    fn haversine_new_york_los_angeles() {
        // New York: (−74.006, 40.7128), Los Angeles: (−118.2437, 34.0522)
        // Approximate great-circle ≈ 3,940 km
        let ny = CoordPoint::wgs84(-74.006, 40.7128);
        let la = CoordPoint::wgs84(-118.2437, 34.0522);
        let d = CoordinateConverter::haversine_distance(&ny, &la).expect("ok");
        assert!(approx_eq(d, 3_940_000.0, 50_000.0), "Got {d}");
    }

    #[test]
    fn haversine_north_pole_equator() {
        // North pole to equator ≈ quarter circumference ≈ 10,007 km
        let pole = CoordPoint::wgs84(0.0, 90.0);
        let equator = CoordPoint::wgs84(0.0, 0.0);
        let d = CoordinateConverter::haversine_distance(&pole, &equator).expect("ok");
        assert!(approx_eq(d, 10_007_000.0, 50_000.0), "Got {d}");
    }

    // ── Additional convert tests ───────────────────────────────────────────

    #[test]
    fn convert_web_mercator_to_wgs84_roundtrip_via_method() {
        let wgs = CoordPoint::wgs84(50.0, 30.0);
        let merc = converter().convert(&wgs, Crs::WebMercator).expect("ok");
        let back = converter().convert(&merc, Crs::Wgs84).expect("ok");
        assert!(approx_eq(back.x, 50.0, 1e-5));
        assert!(approx_eq(back.y, 30.0, 1e-5));
    }

    #[test]
    fn convert_wgs84_to_cartesian_produces_cartesian_crs() {
        let p = CoordPoint::wgs84(10.0, 20.0);
        let out = converter().convert(&p, Crs::Cartesian).expect("ok");
        assert_eq!(out.crs, Crs::Cartesian);
    }

    // ── wgs84_to_web_mercator known reference values ────────────────────────

    #[test]
    fn wgs84_to_web_mercator_positive_lat() {
        // lat=45 → y > 0
        let p = CoordinateConverter::wgs84_to_web_mercator(0.0, 45.0);
        assert!(p.y > 0.0);
    }

    #[test]
    fn wgs84_to_web_mercator_negative_lon_extreme() {
        let p = CoordinateConverter::wgs84_to_web_mercator(-180.0, 0.0);
        let expected = -PI * EARTH_RADIUS_M;
        assert!(approx_eq(p.x, expected, 1e-3));
    }

    // ── Crs equality ──────────────────────────────────────────────────────

    #[test]
    fn crs_equality() {
        assert_eq!(Crs::Wgs84, Crs::Wgs84);
        assert_ne!(Crs::Wgs84, Crs::WebMercator);
        assert_ne!(Crs::WebMercator, Crs::Cartesian);
    }

    #[test]
    fn crs_copy() {
        let c = Crs::WebMercator;
        let d = c;
        assert_eq!(c, d);
    }
}
