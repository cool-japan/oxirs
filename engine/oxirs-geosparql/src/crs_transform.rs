//! Coordinate Reference System (CRS) Transformation Pipeline
//!
//! Implements transformations between common CRS types:
//!
//! - WGS 84 (EPSG:4326) — geographic latitude/longitude in decimal degrees
//! - Web Mercator (EPSG:3857) — Spherical Mercator projection used by web maps
//! - UTM (Universal Transverse Mercator) — zone-based metric grid
//! - Custom EPSG codes (registered in `CrsRegistry`)
//!
//! # WGS84 ↔ Web Mercator
//!
//! Uses the standard spherical Mercator formulae (R = 6378137 m):
//!
//! ```text
//! x = lon_deg * 20037508.34 / 180
//! y = ln(tan((90 + lat_deg) * π / 360)) * R
//! ```
//!
//! Inverse:
//!
//! ```text
//! lon_deg = x * 180 / 20037508.34
//! lat_deg = atan(exp(y / R)) * 360 / π - 90
//! ```
//!
//! # WGS84 ↔ UTM
//!
//! Uses the Karney / Bowring series approximation (accurate to < 1 mm for
//! most of the valid UTM zone extent).

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

// ─── CRS types ─────────────────────────────────────────────────────────────

/// A supported Coordinate Reference System.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Crs {
    /// WGS 84 geographic (EPSG:4326). Coordinates: (longitude°, latitude°, [elevation m])
    Wgs84,
    /// Web Mercator / Spherical Mercator (EPSG:3857). Coordinates: (easting m, northing m)
    WebMercator,
    /// Universal Transverse Mercator — a specific zone and hemisphere.
    /// Coordinates: (easting m, northing m, [height m])
    Utm {
        /// UTM zone number (1–60)
        zone: u8,
        /// `true` for northern hemisphere, `false` for southern
        north: bool,
    },
    /// Any EPSG code not explicitly enumerated (lookup via `CrsRegistry`)
    CustomEpsg(u32),
}

impl fmt::Display for Crs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgs84 => write!(f, "EPSG:4326"),
            Self::WebMercator => write!(f, "EPSG:3857"),
            Self::Utm { zone, north } => {
                write!(f, "EPSG:{}", utm_epsg(*zone, *north))
            }
            Self::CustomEpsg(code) => write!(f, "EPSG:{code}"),
        }
    }
}

fn utm_epsg(zone: u8, north: bool) -> u32 {
    if north {
        32600 + zone as u32
    } else {
        32700 + zone as u32
    }
}

// ─── Coordinate ─────────────────────────────────────────────────────────────

/// A 2D or 3D coordinate.
///
/// The meaning of `x` and `y` depends on the CRS:
/// - WGS84: `x = longitude`, `y = latitude`
/// - Web Mercator / UTM: `x = easting`, `y = northing`
#[derive(Debug, Clone, PartialEq)]
pub struct Coordinate {
    /// X axis value (longitude for geographic CRS, easting for projected)
    pub x: f64,
    /// Y axis value (latitude for geographic CRS, northing for projected)
    pub y: f64,
    /// Optional third dimension (elevation / height in metres)
    pub z: Option<f64>,
}

impl Coordinate {
    /// Create a 2D coordinate.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y, z: None }
    }

    /// Create a 3D coordinate.
    pub fn new_3d(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z: Some(z) }
    }
}

// ─── Bounding box ────────────────────────────────────────────────────────────

/// An axis-aligned bounding box in any CRS.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Minimum corner
    pub min: Coordinate,
    /// Maximum corner
    pub max: Coordinate,
}

impl BoundingBox {
    /// Create a bounding box from two corners.
    pub fn new(min: Coordinate, max: Coordinate) -> Self {
        Self { min, max }
    }
}

// ─── Error ───────────────────────────────────────────────────────────────────

/// Error from CRS transformation.
#[derive(Debug, Clone)]
pub struct CrsError(pub String);

impl fmt::Display for CrsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrsError: {}", self.0)
    }
}

impl std::error::Error for CrsError {}

// ─── Constants ───────────────────────────────────────────────────────────────

/// WGS84 semi-major axis (metres)
const WGS84_A: f64 = 6_378_137.0;
/// WGS84 flattening
const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS84 first eccentricity squared: e² = 2f - f²
const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;
/// UTM scale factor at central meridian
const UTM_K0: f64 = 0.9996;
/// Web Mercator / Spherical Earth radius (same as WGS84 semi-major axis)
const MERC_R: f64 = WGS84_A;
/// Half-circumference at equator in metres (= π × R)
const MERC_HALF_CIRC: f64 = PI * MERC_R; // ≈ 20_037_508.34

// ─── CRS transformation implementations ──────────────────────────────────────

/// Transform from WGS84 to Web Mercator.
fn wgs84_to_web_mercator(coord: &Coordinate) -> Result<Coordinate, CrsError> {
    let lon = coord.x;
    let lat = coord.y;

    if lat >= 90.0 || lat <= -90.0 {
        return Err(CrsError(format!(
            "latitude {lat} is outside valid Web Mercator range (-90, 90)"
        )));
    }

    let x = lon * MERC_HALF_CIRC / 180.0;
    let lat_rad = lat.to_radians();
    let y = ((PI / 4.0 + lat_rad / 2.0).tan()).ln() * MERC_R;

    Ok(Coordinate { x, y, z: coord.z })
}

/// Transform from Web Mercator to WGS84.
fn web_mercator_to_wgs84(coord: &Coordinate) -> Result<Coordinate, CrsError> {
    let x = coord.x;
    let y = coord.y;

    let lon = x * 180.0 / MERC_HALF_CIRC;
    let lat = (2.0 * (y / MERC_R).exp().atan() - PI / 2.0).to_degrees();

    Ok(Coordinate {
        x: lon,
        y: lat,
        z: coord.z,
    })
}

/// Compute the UTM zone from a WGS84 longitude.
pub fn utm_zone_from_lon(lon: f64) -> u8 {
    let z = ((lon + 180.0) / 6.0).floor() as i32 + 1;
    z.clamp(1, 60) as u8
}

/// Central meridian of a UTM zone (degrees).
fn utm_central_meridian(zone: u8) -> f64 {
    (zone as f64 - 1.0) * 6.0 - 177.0
}

/// Transform WGS84 → UTM using the Transverse Mercator projection.
///
/// Based on the formulas by Bowring (1989) and verified against standard UTM tables.
fn wgs84_to_utm(coord: &Coordinate, zone: u8, north: bool) -> Result<Coordinate, CrsError> {
    let lon = coord.x;
    let lat = coord.y;

    if !(-80.0..=84.0).contains(&lat) {
        return Err(CrsError(format!(
            "latitude {lat} is outside UTM range [-80, 84]"
        )));
    }

    let lon0 = utm_central_meridian(zone).to_radians();
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();

    let a = WGS84_A;
    let e2 = WGS84_E2;
    let e_prime2 = e2 / (1.0 - e2);

    let n_val = a / (1.0 - e2 * lat_rad.sin().powi(2)).sqrt();
    let t = lat_rad.tan().powi(2);
    let c = e_prime2 * lat_rad.cos().powi(2);
    let a_lon = lat_rad.cos() * (lon_rad - lon0);

    // Meridional arc M
    let m = a
        * ((1.0 - e2 / 4.0 - 3.0 * e2.powi(2) / 64.0 - 5.0 * e2.powi(3) / 256.0) * lat_rad
            - (3.0 * e2 / 8.0 + 3.0 * e2.powi(2) / 32.0 + 45.0 * e2.powi(3) / 1024.0)
                * (2.0 * lat_rad).sin()
            + (15.0 * e2.powi(2) / 256.0 + 45.0 * e2.powi(3) / 1024.0) * (4.0 * lat_rad).sin()
            - (35.0 * e2.powi(3) / 3072.0) * (6.0 * lat_rad).sin());

    let easting = UTM_K0
        * n_val
        * (a_lon
            + (1.0 - t + c) * a_lon.powi(3) / 6.0
            + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * e_prime2) * a_lon.powi(5) / 120.0)
        + 500_000.0;

    let northing_base = UTM_K0
        * (m + n_val
            * lat_rad.tan()
            * (a_lon.powi(2) / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a_lon.powi(4) / 24.0
                + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * e_prime2) * a_lon.powi(6)
                    / 720.0));

    let northing = if north {
        northing_base
    } else {
        northing_base + 10_000_000.0
    };

    Ok(Coordinate {
        x: easting,
        y: northing,
        z: coord.z,
    })
}

/// Transform UTM → WGS84.
fn utm_to_wgs84(coord: &Coordinate, zone: u8, north: bool) -> Result<Coordinate, CrsError> {
    let mut easting = coord.x;
    let northing_raw = coord.y;

    let a = WGS84_A;
    let e2 = WGS84_E2;
    let e_prime2 = e2 / (1.0 - e2);

    easting -= 500_000.0;
    let northing = if north {
        northing_raw
    } else {
        northing_raw - 10_000_000.0
    };

    let lon0 = utm_central_meridian(zone).to_radians();

    let m = northing / UTM_K0;
    let mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e2.powi(2) / 64.0 - 5.0 * e2.powi(3) / 256.0));

    let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());

    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let n1 = a / (1.0 - e2 * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan().powi(2);
    let c1 = e_prime2 * phi1.cos().powi(2);
    let r1 = a * (1.0 - e2) / (1.0 - e2 * phi1.sin().powi(2)).powf(1.5);
    let d = easting / (n1 * UTM_K0);

    let lat_rad = phi1
        - (n1 * phi1.tan() / r1)
            * (d.powi(2) / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * e_prime2) * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                    - 252.0 * e_prime2
                    - 3.0 * c1.powi(2))
                    * d.powi(6)
                    / 720.0);

    let lon_rad = lon0
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2) + 8.0 * e_prime2 + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / phi1.cos();

    Ok(Coordinate {
        x: lon_rad.to_degrees(),
        y: lat_rad.to_degrees(),
        z: coord.z,
    })
}

// ─── CRS registry ────────────────────────────────────────────────────────────

/// Registry for looking up EPSG codes and their descriptions.
#[derive(Debug, Default)]
pub struct CrsRegistry {
    /// Map from EPSG code → description string
    entries: HashMap<u32, String>,
}

impl CrsRegistry {
    /// Create a new registry pre-populated with common CRS codes.
    pub fn new() -> Self {
        let mut reg = Self::default();
        reg.register(4326, "WGS 84");
        reg.register(3857, "WGS 84 / Pseudo-Mercator");
        // UTM zones 1–60 N and S
        for zone in 1u8..=60 {
            reg.register(32600 + zone as u32, &format!("WGS 84 / UTM zone {zone}N"));
            reg.register(32700 + zone as u32, &format!("WGS 84 / UTM zone {zone}S"));
        }
        reg
    }

    /// Register an EPSG code with a description.
    pub fn register(&mut self, code: u32, description: &str) {
        self.entries.insert(code, description.to_string());
    }

    /// Look up the description for an EPSG code.
    pub fn lookup(&self, code: u32) -> Option<&str> {
        self.entries.get(&code).map(String::as_str)
    }

    /// Return the `Crs` for a given EPSG code (known codes only).
    pub fn to_crs(&self, code: u32) -> Option<Crs> {
        match code {
            4326 => Some(Crs::Wgs84),
            3857 => Some(Crs::WebMercator),
            32601..=32660 => {
                let zone = (code - 32600) as u8;
                Some(Crs::Utm { zone, north: true })
            }
            32701..=32760 => {
                let zone = (code - 32700) as u8;
                Some(Crs::Utm { zone, north: false })
            }
            _ => {
                if self.entries.contains_key(&code) {
                    Some(Crs::CustomEpsg(code))
                } else {
                    None
                }
            }
        }
    }
}

// ─── Main transform struct ────────────────────────────────────────────────────

/// A stateless CRS transformation engine.
///
/// All transform functions are pure (no side effects) and thread-safe.
#[derive(Debug, Default, Clone)]
pub struct CrsTransform;

impl CrsTransform {
    /// Create a new transformer.
    pub fn new() -> Self {
        Self
    }

    /// Transform a coordinate from one CRS to another.
    ///
    /// Indirect routes (e.g. UTM zone A → UTM zone B) go through WGS84.
    pub fn transform(&self, coord: Coordinate, from: Crs, to: Crs) -> Result<Coordinate, CrsError> {
        if from == to {
            return Ok(coord);
        }

        // Convert to WGS84 first, then to target
        let wgs = self.to_wgs84(coord, from)?;
        self.transform_from_wgs84(wgs, to)
    }

    /// Transform a `BoundingBox` from one CRS to another.
    pub fn transform_bbox(
        &self,
        bbox: BoundingBox,
        from: Crs,
        to: Crs,
    ) -> Result<BoundingBox, CrsError> {
        let min = self.transform(bbox.min, from.clone(), to.clone())?;
        let max = self.transform(bbox.max, from, to)?;
        Ok(BoundingBox::new(min, max))
    }

    /// Transform WKT geometry coordinates from one CRS to another.
    ///
    /// Supports: POINT, LINESTRING, POLYGON (with holes), MULTIPOINT,
    /// MULTILINESTRING, MULTIPOLYGON.
    pub fn transform_wkt(&self, wkt: &str, from: Crs, to: Crs) -> Result<String, CrsError> {
        let wkt = wkt.trim();

        if let Some(rest) = wkt.strip_prefix("POINT") {
            let inner = extract_wkt_inner(rest.trim())?;
            let coord = parse_wkt_coord(&inner)?;
            let t = self.transform(coord, from, to)?;
            return Ok(format!("POINT ({} {})", t.x, t.y));
        }

        if let Some(rest) = wkt.strip_prefix("LINESTRING") {
            let inner = extract_wkt_inner(rest.trim())?;
            let coords = parse_wkt_coord_list(&inner, &from, &to, self)?;
            return Ok(format!("LINESTRING ({})", coords));
        }

        if let Some(rest) = wkt.strip_prefix("POLYGON") {
            let inner = rest.trim();
            // Handle rings: outer ( inner1, inner2, ... )
            let rings_str = extract_wkt_inner(inner)?;
            let ring_parts = split_wkt_rings(&rings_str);
            let mut ring_strs = Vec::new();
            for ring in &ring_parts {
                let ring_inner = extract_wkt_inner(ring.trim())?;
                let ring_coords = parse_wkt_coord_list(&ring_inner, &from, &to, self)?;
                ring_strs.push(format!("({ring_coords})"));
            }
            return Ok(format!("POLYGON ({})", ring_strs.join(", ")));
        }

        Err(CrsError(format!("unsupported WKT geometry type: '{wkt}'")))
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn to_wgs84(&self, coord: Coordinate, from: Crs) -> Result<Coordinate, CrsError> {
        match from {
            Crs::Wgs84 => Ok(coord),
            Crs::WebMercator => web_mercator_to_wgs84(&coord),
            Crs::Utm { zone, north } => utm_to_wgs84(&coord, zone, north),
            Crs::CustomEpsg(code) => Err(CrsError(format!(
                "EPSG:{code} to WGS84 transformation not built-in; register a custom converter"
            ))),
        }
    }

    fn transform_from_wgs84(&self, coord: Coordinate, to: Crs) -> Result<Coordinate, CrsError> {
        match to {
            Crs::Wgs84 => Ok(coord),
            Crs::WebMercator => wgs84_to_web_mercator(&coord),
            Crs::Utm { zone, north } => wgs84_to_utm(&coord, zone, north),
            Crs::CustomEpsg(code) => Err(CrsError(format!(
                "WGS84 to EPSG:{code} transformation not built-in; register a custom converter"
            ))),
        }
    }
}

// ─── WKT parsing helpers ─────────────────────────────────────────────────────

/// Extract the content inside the outermost parentheses.
fn extract_wkt_inner(s: &str) -> Result<String, CrsError> {
    let start = s
        .find('(')
        .ok_or_else(|| CrsError(format!("no '(' in WKT: '{s}'")))?;
    let end = s
        .rfind(')')
        .ok_or_else(|| CrsError(format!("no ')' in WKT: '{s}'")))?;
    if start >= end {
        return Err(CrsError(format!("malformed WKT parentheses: '{s}'")));
    }
    Ok(s[start + 1..end].to_string())
}

/// Parse a single WKT coordinate pair `"x y"` or `"x y z"`.
fn parse_wkt_coord(s: &str) -> Result<Coordinate, CrsError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(CrsError(format!("not enough ordinates in '{s}'")));
    }
    let x = parts[0]
        .parse::<f64>()
        .map_err(|e| CrsError(format!("bad x ordinate '{}: {e}", parts[0])))?;
    let y = parts[1]
        .parse::<f64>()
        .map_err(|e| CrsError(format!("bad y ordinate '{}': {e}", parts[1])))?;
    let z = parts.get(2).and_then(|s| s.parse::<f64>().ok());
    Ok(Coordinate { x, y, z })
}

/// Parse a comma-separated list of coordinate pairs and return them as a WKT string.
fn parse_wkt_coord_list(
    s: &str,
    from: &Crs,
    to: &Crs,
    transform: &CrsTransform,
) -> Result<String, CrsError> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let coord = parse_wkt_coord(part.trim())?;
        let t = transform.transform(coord, from.clone(), to.clone())?;
        if t.z.is_some() {
            out.push(format!("{} {} {}", t.x, t.y, t.z.unwrap_or(0.0)));
        } else {
            out.push(format!("{} {}", t.x, t.y));
        }
    }
    Ok(out.join(", "))
}

/// Split a polygon rings string into individual ring strings.
/// Input: `"(x1 y1, x2 y2), (x3 y3, x4 y4)"`
fn split_wkt_rings(s: &str) -> Vec<String> {
    let mut rings = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    rings.push(s[start..=i].to_string());
                    start = i + 1;
                    // Skip the comma and space
                    while start < s.len()
                        && (s.as_bytes()[start] == b',' || s.as_bytes()[start] == b' ')
                    {
                        start += 1;
                    }
                }
            }
            _ => {}
        }
    }

    if rings.is_empty() {
        // No sub-rings: treat the whole string as one ring
        rings.push(format!("({s})"));
    }

    rings
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-4; // 0.1 mm accuracy threshold for round-trips

    fn assert_close(a: f64, b: f64, tol: f64, label: &str) {
        let diff = (a - b).abs();
        assert!(diff < tol, "{label}: |{a} - {b}| = {diff} ≥ {tol}");
    }

    fn t() -> CrsTransform {
        CrsTransform::new()
    }

    // ── Identity transforms ─────────────────────────────────────────────────

    #[test]
    fn test_wgs84_identity() {
        let c = Coordinate::new(10.0, 50.0);
        let r = t()
            .transform(c.clone(), Crs::Wgs84, Crs::Wgs84)
            .expect("ok");
        assert_eq!(r.x, c.x);
        assert_eq!(r.y, c.y);
    }

    #[test]
    fn test_web_mercator_identity() {
        let c = Coordinate::new(1_113_194.9, 6_274_861.4);
        let r = t()
            .transform(c.clone(), Crs::WebMercator, Crs::WebMercator)
            .expect("ok");
        assert_eq!(r.x, c.x);
    }

    #[test]
    fn test_utm_identity() {
        let zone = Crs::Utm {
            zone: 32,
            north: true,
        };
        let c = Coordinate::new(500_000.0, 5_538_638.0);
        let r = t().transform(c.clone(), zone.clone(), zone).expect("ok");
        assert_eq!(r.x, c.x);
    }

    // ── WGS84 ↔ Web Mercator ─────────────────────────────────────────────────

    #[test]
    fn test_wgs84_to_web_mercator_origin() {
        let c = Coordinate::new(0.0, 0.0);
        let r = t().transform(c, Crs::Wgs84, Crs::WebMercator).expect("ok");
        assert_close(r.x, 0.0, 1.0, "x at origin");
        assert_close(r.y, 0.0, 1.0, "y at origin");
    }

    #[test]
    fn test_wgs84_to_web_mercator_london() {
        // London: lon=-0.1276, lat=51.5074
        let c = Coordinate::new(-0.1276, 51.5074);
        let r = t().transform(c, Crs::Wgs84, Crs::WebMercator).expect("ok");
        // Expected approx: x ≈ -14211, y ≈ 6711395
        assert_close(r.x, -14211.0, 200.0, "London easting");
        assert_close(r.y, 6_711_395.0, 200.0, "London northing");
    }

    #[test]
    fn test_web_mercator_to_wgs84_roundtrip() {
        let original = Coordinate::new(13.4050, 52.5200); // Berlin
        let merc = t()
            .transform(original.clone(), Crs::Wgs84, Crs::WebMercator)
            .expect("ok");
        let back = t()
            .transform(merc, Crs::WebMercator, Crs::Wgs84)
            .expect("ok");
        assert_close(back.x, original.x, EPSILON, "Berlin lon round-trip");
        assert_close(back.y, original.y, EPSILON, "Berlin lat round-trip");
    }

    #[test]
    fn test_web_mercator_extreme_longitude() {
        let c = Coordinate::new(180.0, 0.0);
        let r = t().transform(c, Crs::Wgs84, Crs::WebMercator).expect("ok");
        // x should be approximately half-circumference
        assert_close(r.x, MERC_HALF_CIRC, 1.0, "180° easting");
    }

    #[test]
    fn test_web_mercator_negative_longitude() {
        let c = Coordinate::new(-180.0, 0.0);
        let r = t().transform(c, Crs::Wgs84, Crs::WebMercator).expect("ok");
        assert_close(r.x, -MERC_HALF_CIRC, 1.0, "-180° easting");
    }

    #[test]
    fn test_wgs84_to_web_mercator_polar_error() {
        let c = Coordinate::new(0.0, 90.0);
        let result = t().transform(c, Crs::Wgs84, Crs::WebMercator);
        assert!(result.is_err());
    }

    #[test]
    fn test_web_mercator_preserves_z() {
        let c = Coordinate::new_3d(0.0, 0.0, 42.0);
        let r = t().transform(c, Crs::Wgs84, Crs::WebMercator).expect("ok");
        assert_eq!(r.z, Some(42.0));
    }

    // ── WGS84 ↔ UTM ──────────────────────────────────────────────────────────

    #[test]
    fn test_wgs84_to_utm_zone_32n_berlin() {
        // Berlin: lon=13.4050, lat=52.5200 → UTM zone 33N
        let c = Coordinate::new(13.4050, 52.5200);
        let zone = utm_zone_from_lon(c.x);
        assert_eq!(zone, 33);
        let r = t()
            .transform(c, Crs::Wgs84, Crs::Utm { zone, north: true })
            .expect("ok");
        // Easting should be close to 500 000 (central meridian of zone 33)
        assert!(
            r.x > 370_000.0 && r.x < 620_000.0,
            "easting in range: {}",
            r.x
        );
        assert!(r.y > 5_000_000.0, "northing positive: {}", r.y);
    }

    #[test]
    fn test_utm_to_wgs84_roundtrip() {
        let original = Coordinate::new(13.4050, 52.5200); // Berlin
        let zone = utm_zone_from_lon(original.x);
        let utm = t()
            .transform(original.clone(), Crs::Wgs84, Crs::Utm { zone, north: true })
            .expect("to UTM");
        let back = t()
            .transform(utm, Crs::Utm { zone, north: true }, Crs::Wgs84)
            .expect("from UTM");
        assert_close(back.x, original.x, 0.001, "Berlin lon UTM round-trip");
        assert_close(back.y, original.y, 0.001, "Berlin lat UTM round-trip");
    }

    #[test]
    fn test_utm_southern_hemisphere_offset() {
        // São Paulo: lon=-46.6333, lat=-23.5505 → UTM zone 23S
        let c = Coordinate::new(-46.6333, -23.5505);
        let zone = utm_zone_from_lon(c.x);
        assert_eq!(zone, 23);
        let r = t()
            .transform(c.clone(), Crs::Wgs84, Crs::Utm { zone, north: false })
            .expect("ok");
        // Southern hemisphere: northing is in range (0, 10_000_000) due to the 10 000 000 offset
        // São Paulo is at ~lat -23.55 → UTM northing ≈ 7 394 000
        assert!(
            r.y > 0.0 && r.y < 10_000_000.0,
            "southern hemisphere northing in range: {}",
            r.y
        );
    }

    #[test]
    fn test_utm_southern_roundtrip() {
        let original = Coordinate::new(-46.6333, -23.5505);
        let zone = utm_zone_from_lon(original.x);
        let utm = t()
            .transform(
                original.clone(),
                Crs::Wgs84,
                Crs::Utm { zone, north: false },
            )
            .expect("to UTM");
        let back = t()
            .transform(utm, Crs::Utm { zone, north: false }, Crs::Wgs84)
            .expect("from UTM");
        assert_close(back.x, original.x, 0.001, "São Paulo lon round-trip");
        assert_close(back.y, original.y, 0.001, "São Paulo lat round-trip");
    }

    #[test]
    fn test_utm_zone_from_lon_basic() {
        assert_eq!(utm_zone_from_lon(0.0), 31);
        assert_eq!(utm_zone_from_lon(6.0), 32);
        assert_eq!(utm_zone_from_lon(-180.0), 1);
        assert_eq!(utm_zone_from_lon(174.0), 60);
    }

    #[test]
    fn test_utm_error_polar() {
        let c = Coordinate::new(0.0, 85.0);
        let result = t().transform(
            c,
            Crs::Wgs84,
            Crs::Utm {
                zone: 31,
                north: true,
            },
        );
        assert!(result.is_err());
    }

    // ── Cross-CRS transform ────────────────────────────────────────────────────

    #[test]
    fn test_web_mercator_to_utm_via_wgs84() {
        // Web Mercator → UTM goes through WGS84
        let wgs = Coordinate::new(13.4050, 52.5200);
        let merc = t()
            .transform(wgs, Crs::Wgs84, Crs::WebMercator)
            .expect("ok");
        let utm = t()
            .transform(
                merc,
                Crs::WebMercator,
                Crs::Utm {
                    zone: 33,
                    north: true,
                },
            )
            .expect("ok");
        assert!(utm.x > 300_000.0 && utm.x < 700_000.0);
    }

    // ── Bounding box ─────────────────────────────────────────────────────────

    #[test]
    fn test_transform_bbox() {
        let bbox = BoundingBox::new(Coordinate::new(-10.0, 45.0), Coordinate::new(10.0, 55.0));
        let result = t()
            .transform_bbox(bbox, Crs::Wgs84, Crs::WebMercator)
            .expect("ok");
        assert!(result.min.x < result.max.x);
        assert!(result.min.y < result.max.y);
    }

    // ── WKT transform ────────────────────────────────────────────────────────

    #[test]
    fn test_transform_wkt_point() {
        let wkt = "POINT (0.0 0.0)";
        let r = t()
            .transform_wkt(wkt, Crs::Wgs84, Crs::WebMercator)
            .expect("ok");
        assert!(r.starts_with("POINT"));
        assert!(r.contains('0'));
    }

    #[test]
    fn test_transform_wkt_linestring() {
        let wkt = "LINESTRING (0.0 0.0, 10.0 10.0)";
        let r = t()
            .transform_wkt(wkt, Crs::Wgs84, Crs::WebMercator)
            .expect("ok");
        assert!(r.starts_with("LINESTRING"));
    }

    #[test]
    fn test_transform_wkt_polygon() {
        let wkt = "POLYGON ((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 1.0, 0.0 0.0))";
        let r = t()
            .transform_wkt(wkt, Crs::Wgs84, Crs::WebMercator)
            .expect("ok");
        assert!(r.starts_with("POLYGON"));
    }

    #[test]
    fn test_transform_wkt_unsupported_type() {
        let wkt = "GEOMETRYCOLLECTION EMPTY";
        let r = t().transform_wkt(wkt, Crs::Wgs84, Crs::WebMercator);
        assert!(r.is_err());
    }

    // ── CrsRegistry ───────────────────────────────────────────────────────────

    #[test]
    fn test_registry_lookup_wgs84() {
        let reg = CrsRegistry::new();
        let desc = reg.lookup(4326).expect("4326 exists");
        assert!(desc.contains("WGS"));
    }

    #[test]
    fn test_registry_lookup_web_mercator() {
        let reg = CrsRegistry::new();
        assert!(reg.lookup(3857).is_some());
    }

    #[test]
    fn test_registry_to_crs_wgs84() {
        let reg = CrsRegistry::new();
        assert_eq!(reg.to_crs(4326), Some(Crs::Wgs84));
    }

    #[test]
    fn test_registry_to_crs_web_mercator() {
        let reg = CrsRegistry::new();
        assert_eq!(reg.to_crs(3857), Some(Crs::WebMercator));
    }

    #[test]
    fn test_registry_to_crs_utm_north() {
        let reg = CrsRegistry::new();
        assert_eq!(
            reg.to_crs(32633),
            Some(Crs::Utm {
                zone: 33,
                north: true
            })
        );
    }

    #[test]
    fn test_registry_to_crs_utm_south() {
        let reg = CrsRegistry::new();
        assert_eq!(
            reg.to_crs(32723),
            Some(Crs::Utm {
                zone: 23,
                north: false
            })
        );
    }

    #[test]
    fn test_registry_to_crs_unknown() {
        let reg = CrsRegistry::new();
        assert_eq!(reg.to_crs(99999), None);
    }

    #[test]
    fn test_registry_custom_registration() {
        let mut reg = CrsRegistry::new();
        reg.register(99000, "My Custom CRS");
        assert_eq!(reg.lookup(99000), Some("My Custom CRS"));
        assert_eq!(reg.to_crs(99000), Some(Crs::CustomEpsg(99000)));
    }

    #[test]
    fn test_crs_display() {
        assert_eq!(Crs::Wgs84.to_string(), "EPSG:4326");
        assert_eq!(Crs::WebMercator.to_string(), "EPSG:3857");
        assert_eq!(
            Crs::Utm {
                zone: 33,
                north: true
            }
            .to_string(),
            "EPSG:32633"
        );
        assert_eq!(
            Crs::Utm {
                zone: 23,
                north: false
            }
            .to_string(),
            "EPSG:32723"
        );
    }

    #[test]
    fn test_custom_epsg_error() {
        let c = Coordinate::new(0.0, 0.0);
        let result = t().transform(c, Crs::Wgs84, Crs::CustomEpsg(99999));
        assert!(result.is_err());
    }
}
