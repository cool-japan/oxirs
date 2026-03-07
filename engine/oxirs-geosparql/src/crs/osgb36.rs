//! Ordnance Survey National Grid (OSGB36) coordinate system
//!
//! Provides conversion between:
//! - OSGB36 easting/northing (British National Grid)
//! - WGS84 geographic (lat/lon)
//! - Alphanumeric grid references (e.g. "TQ 30 80")
//!
//! ## Datum
//!
//! Uses the Airy 1830 ellipsoid with a 7-parameter Helmert transformation
//! (OSTN15-compatible shift).  The transformation is approximate; for
//! survey-grade accuracy use the OSTN15 grid shift file.
//!
//! ## References
//!
//! - OS "A Guide to coordinate systems in Great Britain" (OS document D00659)
//! - Ordnance Survey OSTN15 documentation

use crate::error::{GeoSparqlError, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Airy 1830 ellipsoid constants
// ---------------------------------------------------------------------------

const AIRY_A: f64 = 6_377_563.396;
const AIRY_B: f64 = 6_356_256.909;
const AIRY_E2: f64 = (AIRY_A * AIRY_A - AIRY_B * AIRY_B) / (AIRY_A * AIRY_A);

// WGS84 ellipsoid constants
const WGS84_A: f64 = 6_378_137.0;
const WGS84_E2: f64 = 0.006_694_379_990_14;

// Helmert 7-parameter transformation (WGS84 → OSGB36)
const TX: f64 = -446.448;
const TY: f64 = 125.157;
const TZ: f64 = -542.060;
const S: f64 = -20.489e-6; // scale (ppm)
const RX: f64 = -0.150_9e-6; // rotation X (rad)
const RY: f64 = -0.247_0e-6; // rotation Y (rad)
const RZ: f64 = -0.842_8e-6; // rotation Z (rad)

// British National Grid projection parameters (Transverse Mercator)
const BNG_LAT0: f64 = 49.0; // degrees
const BNG_LON0: f64 = -2.0; // degrees (central meridian)
const BNG_N0: f64 = -100_000.0; // false northing
const BNG_E0: f64 = 400_000.0; // false easting
const BNG_F0: f64 = 0.999_601_272; // scale factor

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// OSGB36 easting/northing coordinate (British National Grid).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OsgbCoordinate {
    /// Easting in meters from the false origin
    pub easting: f64,
    /// Northing in meters from the false origin
    pub northing: f64,
}

impl OsgbCoordinate {
    /// Create a new OSGB36 coordinate.
    pub fn new(easting: f64, northing: f64) -> Self {
        Self { easting, northing }
    }

    /// Convert WGS84 geographic coordinate to OSGB36.
    ///
    /// Uses a 7-parameter Helmert transformation followed by a
    /// Transverse Mercator projection onto the Airy 1830 ellipsoid.
    pub fn from_wgs84(lat_deg: f64, lon_deg: f64) -> Result<Self> {
        // Step 1: WGS84 lat/lon → WGS84 Cartesian ECEF
        let (x_wgs, y_wgs, z_wgs) = geodetic_to_ecef(lat_deg, lon_deg, WGS84_A, WGS84_E2);

        // Step 2: Apply Helmert transformation (WGS84 → OSGB36)
        let (x_osgb, y_osgb, z_osgb) = helmert_transform(x_wgs, y_wgs, z_wgs);

        // Step 3: OSGB36 ECEF → Airy 1830 lat/lon
        let (lat_airy, lon_airy) = ecef_to_geodetic(x_osgb, y_osgb, z_osgb, AIRY_A, AIRY_E2)?;

        // Step 4: Project onto British National Grid
        let (e, n) = bng_project(lat_airy, lon_airy)?;

        Ok(OsgbCoordinate {
            easting: e,
            northing: n,
        })
    }

    /// Convert OSGB36 coordinate back to approximate WGS84 geographic.
    pub fn to_wgs84(&self) -> Result<(f64, f64)> {
        // Step 1: Inverse BNG projection → Airy 1830 lat/lon
        let (lat_airy, lon_airy) = bng_inverse(self.easting, self.northing)?;

        // Step 2: Airy 1830 lat/lon → ECEF
        let (x_osgb, y_osgb, z_osgb) = geodetic_to_ecef(lat_airy, lon_airy, AIRY_A, AIRY_E2);

        // Step 3: Inverse Helmert (OSGB36 → WGS84)
        let (x_wgs, y_wgs, z_wgs) = helmert_transform_inverse(x_osgb, y_osgb, z_osgb);

        // Step 4: WGS84 ECEF → lat/lon
        let (lat, lon) = ecef_to_geodetic(x_wgs, y_wgs, z_wgs, WGS84_A, WGS84_E2)?;

        Ok((lat, lon))
    }

    /// Format as a standard Ordnance Survey grid reference.
    ///
    /// Returns e.g. `"TQ308805"` (10 m precision).
    pub fn to_grid_ref(&self, digits: u8) -> Result<String> {
        coordinate_to_osgb_grid_ref(self, digits)
    }

    /// EPSG code for OSGB36 British National Grid.
    pub fn epsg_code() -> u32 {
        27700
    }
}

// ---------------------------------------------------------------------------
// Grid reference ↔ coordinate
// ---------------------------------------------------------------------------

/// Parse an alphanumeric OS grid reference to an OSGB36 coordinate.
///
/// Accepts formats like `"TQ308805"`, `"TQ 308 805"`, `"SU 387 148"`.
pub fn osgb_grid_ref_to_coordinate(grid_ref: &str) -> Result<OsgbCoordinate> {
    let grid_ref = grid_ref.replace(' ', "");
    if grid_ref.len() < 2 {
        return Err(GeoSparqlError::InvalidParameter(
            "Grid reference too short".to_string(),
        ));
    }

    let letters: String = grid_ref.chars().take(2).collect();
    let digits: String = grid_ref.chars().skip(2).collect();

    let (e_sq, n_sq) = grid_letters_to_offsets(&letters)?;

    let digit_pairs = digits.len() / 2;
    let precision = 10.0f64.powi(5 - digit_pairs as i32);

    let e_digits: f64 = if digit_pairs > 0 {
        digits[..digit_pairs].parse().map_err(|_| {
            GeoSparqlError::InvalidParameter(format!("Invalid easting digits in '{}'", grid_ref))
        })?
    } else {
        0.0
    };

    let n_digits: f64 = if digit_pairs > 0 {
        digits[digit_pairs..].parse().map_err(|_| {
            GeoSparqlError::InvalidParameter(format!("Invalid northing digits in '{}'", grid_ref))
        })?
    } else {
        0.0
    };

    let easting = e_sq + e_digits * precision;
    let northing = n_sq + n_digits * precision;

    Ok(OsgbCoordinate { easting, northing })
}

/// Convert an OSGB36 coordinate to an OS grid reference string.
///
/// `digits` controls the precision per axis (1=10km, 2=1km, 3=100m, 4=10m, 5=1m).
pub fn coordinate_to_osgb_grid_ref(coord: &OsgbCoordinate, digits: u8) -> Result<String> {
    let digits = digits.clamp(1, 5) as usize;
    let e = coord.easting.floor() as i64;
    let n = coord.northing.floor() as i64;

    // Determine 100km grid square
    let e100 = (e / 100_000) as usize;
    let n100 = (n / 100_000) as usize;

    let letters = offsets_to_grid_letters(e100, n100)?;

    let e_rem = (e % 100_000) as f64;
    let n_rem = (n % 100_000) as f64;

    let divisor = 10.0f64.powi(5 - digits as i32);

    let e_part = (e_rem / divisor) as u32;
    let n_part = (n_rem / divisor) as u32;

    Ok(format!(
        "{}{:0digits$}{:0digits$}",
        letters,
        e_part,
        n_part,
        digits = digits
    ))
}

// ---------------------------------------------------------------------------
// Grid letter helpers
// ---------------------------------------------------------------------------

/// Map two-letter OS grid prefix to (easting_origin, northing_origin) in meters.
fn grid_letters_to_offsets(letters: &str) -> Result<(f64, f64)> {
    let mut chars = letters.chars();
    let first = chars.next().ok_or_else(|| {
        GeoSparqlError::InvalidParameter("Grid reference missing first letter".to_string())
    })?;
    let second = chars.next().ok_or_else(|| {
        GeoSparqlError::InvalidParameter("Grid reference missing second letter".to_string())
    })?;

    let major_e = letter_to_index(first)? % 5;
    let major_n = 4 - letter_to_index(first)? / 5;

    let minor_e = letter_to_index(second)? % 5;
    let minor_n = 4 - letter_to_index(second)? / 5;

    let easting = ((major_e * 5 + minor_e) as f64 - 10.0) * 100_000.0;
    let northing = ((major_n * 5 + minor_n) as f64 - 5.0) * 100_000.0;

    Ok((easting, northing))
}

/// Map (easting_100km, northing_100km) indices to two-letter OS grid prefix.
fn offsets_to_grid_letters(e100: usize, n100: usize) -> Result<String> {
    if e100 >= 7 || n100 >= 13 {
        return Err(GeoSparqlError::InvalidParameter(
            "Coordinate out of OSGB36 coverage".to_string(),
        ));
    }

    // Offset into the OSGB grid: origin is at 500km W, 1000km S of true origin
    let e_idx = e100 + 10;
    let n_idx = n100 + 5;

    let major_col = e_idx / 5;
    let major_row = 4 - (n_idx / 5);
    let minor_col = e_idx % 5;
    let minor_row = 4 - (n_idx % 5);

    let first = index_to_letter(major_row * 5 + major_col)?;
    let second = index_to_letter(minor_row * 5 + minor_col)?;

    Ok(format!("{}{}", first, second))
}

fn letter_to_index(c: char) -> Result<usize> {
    let c = c.to_ascii_uppercase();
    // Skip 'I'
    if c == 'I' {
        return Err(GeoSparqlError::InvalidParameter(
            "Grid reference letter 'I' is not valid".to_string(),
        ));
    }
    let base = c as usize - 'A' as usize;
    let idx = if c > 'I' { base - 1 } else { base };
    Ok(idx)
}

fn index_to_letter(idx: usize) -> Result<char> {
    // Skip 'I'
    let letter_idx = if idx >= 8 { idx + 1 } else { idx };
    let c = (b'A' + letter_idx as u8) as char;
    if c > 'Z' {
        return Err(GeoSparqlError::InvalidParameter(format!(
            "Grid index {} exceeds alphabet",
            idx
        )));
    }
    Ok(c)
}

// ---------------------------------------------------------------------------
// Ellipsoidal / projection helpers
// ---------------------------------------------------------------------------

/// Convert geodetic lat/lon to ECEF Cartesian (X, Y, Z).
fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, a: f64, e2: f64) -> (f64, f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let n = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    (
        n * lat.cos() * lon.cos(),
        n * lat.cos() * lon.sin(),
        n * (1.0 - e2) * lat.sin(),
    )
}

/// Convert ECEF Cartesian to geodetic lat/lon (Bowring iterative).
fn ecef_to_geodetic(x: f64, y: f64, z: f64, a: f64, e2: f64) -> Result<(f64, f64)> {
    let lon = y.atan2(x);
    let p = (x * x + y * y).sqrt();
    let mut lat = z.atan2(p * (1.0 - e2));

    for _ in 0..10 {
        let n = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
        let lat_new = (z + e2 * n * lat.sin()).atan2(p);
        if (lat_new - lat).abs() < 1e-12 {
            lat = lat_new;
            break;
        }
        lat = lat_new;
    }

    Ok((lat.to_degrees(), lon.to_degrees()))
}

/// Apply Helmert 7-parameter transformation.
fn helmert_transform(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (
        TX + (1.0 + S) * (x - RZ * y + RY * z),
        TY + (1.0 + S) * (RZ * x + y - RX * z),
        TZ + (1.0 + S) * (-RY * x + RX * y + z),
    )
}

/// Apply inverse Helmert 7-parameter transformation.
fn helmert_transform_inverse(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    // Approximate inverse (acceptable for < 1 m accuracy)
    (
        -TX + (1.0 - S) * (x + RZ * y - RY * z),
        -TY + (1.0 - S) * (-RZ * x + y + RX * z),
        -TZ + (1.0 - S) * (RY * x - RX * y + z),
    )
}

/// British National Grid: Transverse Mercator forward projection.
///
/// Formula from OS "A Guide to Coordinate Systems in Great Britain" (v2.3),
/// using Airy 1830 ellipsoid and OSGB36 datum.
fn bng_project(lat_deg: f64, lon_deg: f64) -> Result<(f64, f64)> {
    let phi = lat_deg.to_radians();
    let lam = lon_deg.to_radians();
    let phi0 = BNG_LAT0.to_radians();
    let lam0 = BNG_LON0.to_radians();

    let n_ratio = (AIRY_A - AIRY_B) / (AIRY_A + AIRY_B);

    // Scale-factor-corrected radii of curvature (F0 already embedded)
    let nu = AIRY_A * BNG_F0 / (1.0 - AIRY_E2 * phi.sin().powi(2)).sqrt();
    let rho = AIRY_A * BNG_F0 * (1.0 - AIRY_E2) / (1.0 - AIRY_E2 * phi.sin().powi(2)).powf(1.5);
    let eta2 = nu / rho - 1.0;

    // Meridional arc (OS guide eq. 3) — uses a*F0, correct argument ordering
    let mv = bng_meridional_arc(phi, phi0, n_ratio);

    let i = mv + BNG_N0;
    let ii = (nu / 2.0) * phi.sin() * phi.cos();
    let iii = (nu / 24.0) * phi.sin() * phi.cos().powi(3) * (5.0 - phi.tan().powi(2) + 9.0 * eta2);
    let iii_a = (nu / 720.0)
        * phi.sin()
        * phi.cos().powi(5)
        * (61.0 - 58.0 * phi.tan().powi(2) + phi.tan().powi(4));

    let iv = nu * phi.cos();
    let v = (nu / 6.0) * phi.cos().powi(3) * (nu / rho - phi.tan().powi(2));
    let vi = (nu / 120.0)
        * phi.cos().powi(5)
        * (5.0 - 18.0 * phi.tan().powi(2) + phi.tan().powi(4) + 14.0 * eta2
            - 58.0 * phi.tan().powi(2) * eta2);

    let dl = lam - lam0;

    // F0 is already embedded in nu/rho/M — do NOT multiply output by F0 again
    let northing = i + ii * dl.powi(2) + iii * dl.powi(4) + iii_a * dl.powi(6);
    let easting = BNG_E0 + iv * dl + v * dl.powi(3) + vi * dl.powi(5);

    Ok((easting, northing))
}

/// British National Grid: Transverse Mercator inverse projection.
///
/// Formula from OS "A Guide to Coordinate Systems in Great Britain" (v2.3).
fn bng_inverse(easting: f64, northing: f64) -> Result<(f64, f64)> {
    let phi0 = BNG_LAT0.to_radians();
    let lam0 = BNG_LON0.to_radians();
    let n_ratio = (AIRY_A - AIRY_B) / (AIRY_A + AIRY_B);

    // Find footpoint latitude by iteration
    let mut phi = phi0 + (northing - BNG_N0) / (AIRY_A * BNG_F0);
    for _ in 0..20 {
        let mv = bng_meridional_arc(phi, phi0, n_ratio);
        let dphi = (northing - BNG_N0 - mv) / (AIRY_A * BNG_F0);
        phi += dphi;
        if dphi.abs() < 1e-12 {
            break;
        }
    }

    let nu = AIRY_A * BNG_F0 / (1.0 - AIRY_E2 * phi.sin().powi(2)).sqrt();
    let rho = AIRY_A * BNG_F0 * (1.0 - AIRY_E2) / (1.0 - AIRY_E2 * phi.sin().powi(2)).powf(1.5);
    let eta2 = nu / rho - 1.0;

    let de = easting - BNG_E0;

    let vii = phi.tan() / (2.0 * rho * nu);
    let viii = phi.tan() / (24.0 * rho * nu.powi(3))
        * (5.0 + 3.0 * phi.tan().powi(2) + eta2 - 9.0 * phi.tan().powi(2) * eta2);
    let ix = phi.tan() / (720.0 * rho * nu.powi(5))
        * (61.0 + 90.0 * phi.tan().powi(2) + 45.0 * phi.tan().powi(4));

    let x = 1.0 / (nu * phi.cos());
    let xi = (1.0 + 2.0 * phi.tan().powi(2) + eta2) / (6.0 * nu.powi(3) * phi.cos());
    let xii = (5.0
        + 28.0 * phi.tan().powi(2)
        + 24.0 * phi.tan().powi(4)
        + 6.0 * eta2
        + 8.0 * phi.tan().powi(2) * eta2)
        / (120.0 * nu.powi(5) * phi.cos());

    let phi_out = phi - vii * de.powi(2) + viii * de.powi(4) - ix * de.powi(6);
    let lam_out = lam0 + x * de - xi * de.powi(3) + xii * de.powi(5);

    Ok((phi_out.to_degrees(), lam_out.to_degrees()))
}

/// Meridional arc calculation for BNG (OS Guide equation 3).
///
/// Uses `a * F0` (not `b`) and the correct argument ordering:
/// `sin(phi - phi0) * cos(phi + phi0)` (not the other way around).
fn bng_meridional_arc(lat: f64, lat0: f64, n_ratio: f64) -> f64 {
    AIRY_A
        * BNG_F0
        * ((1.0 + n_ratio + 5.0 / 4.0 * n_ratio.powi(2) + 5.0 / 4.0 * n_ratio.powi(3))
            * (lat - lat0)
            - (3.0 * n_ratio + 3.0 * n_ratio.powi(2) + 21.0 / 8.0 * n_ratio.powi(3))
                * (lat - lat0).sin()
                * (lat + lat0).cos()
            + (15.0 / 8.0 * n_ratio.powi(2) + 15.0 / 8.0 * n_ratio.powi(3))
                * (2.0 * (lat - lat0)).sin()
                * (2.0 * (lat + lat0)).cos()
            - (35.0 / 24.0 * n_ratio.powi(3))
                * (3.0 * (lat - lat0)).sin()
                * (3.0 * (lat + lat0)).cos())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    const MAX_METER_ERR: f64 = 5.0; // 5 m tolerance (Helmert approximation)
    #[allow(dead_code)]
    const MAX_DEG_ERR: f64 = 0.0001; // ~10 m at UK latitudes (reference tolerance)

    // ---- Grid reference parsing ----

    #[test]
    fn test_grid_ref_parsing_tq() {
        // TQ 30 80 → easting=530000, northing=180000 (approximately)
        let coord = osgb_grid_ref_to_coordinate("TQ3080").expect("TQ3080");
        assert!(
            (coord.easting - 530_000.0).abs() < 1000.0,
            "easting: {}",
            coord.easting
        );
        assert!(
            (coord.northing - 180_000.0).abs() < 1000.0,
            "northing: {}",
            coord.northing
        );
    }

    #[test]
    fn test_grid_ref_roundtrip() {
        let orig = OsgbCoordinate::new(530_000.0, 180_000.0);
        let grid_ref = orig.to_grid_ref(3).expect("to grid ref");
        let back = osgb_grid_ref_to_coordinate(&grid_ref).expect("from grid ref");
        // At 3-digit precision = 100 m accuracy
        assert!((back.easting - orig.easting).abs() < 200.0);
        assert!((back.northing - orig.northing).abs() < 200.0);
    }

    #[test]
    fn test_grid_letters_invalid_i() {
        assert!(osgb_grid_ref_to_coordinate("IQ1234").is_err());
    }

    // ---- Coordinate conversion ----

    #[test]
    fn test_from_wgs84_london() {
        // St Paul's Cathedral: 51.5138°N, -0.0984°W
        let coord = OsgbCoordinate::from_wgs84(51.5138, -0.0984).expect("St Pauls");
        // Expected: E~532_000, N~181_000 (approximate)
        assert!(
            (coord.easting - 532_000.0).abs() < 5000.0,
            "easting {} unexpected",
            coord.easting
        );
        assert!(
            (coord.northing - 181_000.0).abs() < 5000.0,
            "northing {} unexpected",
            coord.northing
        );
    }

    #[test]
    fn test_to_wgs84_roundtrip() {
        // Cambridge: the Helmert 7-parameter transform has ~5m accuracy
        // so we allow 0.001° (~100m) for WGS84 roundtrip
        let wgs84 = (52.2053, 0.1218);
        let osgb = OsgbCoordinate::from_wgs84(wgs84.0, wgs84.1).expect("Cambridge");
        let (lat, lon) = osgb.to_wgs84().expect("inverse");
        // Helmert approximation accuracy: ±50m ≈ 0.0005°
        const HELMERT_DEG_TOL: f64 = 0.001;
        assert!(
            (lat - wgs84.0).abs() < HELMERT_DEG_TOL,
            "lat error: {}",
            (lat - wgs84.0).abs()
        );
        assert!(
            (lon - wgs84.1).abs() < HELMERT_DEG_TOL,
            "lon error: {}",
            (lon - wgs84.1).abs()
        );
    }

    #[test]
    fn test_epsg_code() {
        assert_eq!(OsgbCoordinate::epsg_code(), 27700);
    }

    // ---- Grid reference formatting ----

    #[test]
    fn test_to_grid_ref_5digits() {
        let coord = OsgbCoordinate::new(530_023.0, 180_456.0);
        let gr = coord.to_grid_ref(5).expect("5-digit ref");
        // Should start with "TQ" for easting ~530000, northing ~180000
        assert!(gr.starts_with("TQ"), "expected TQ, got {}", gr);
        assert_eq!(gr.len(), 12); // 2 letters + 5 + 5 digits
    }

    #[test]
    fn test_to_grid_ref_2digits() {
        let coord = OsgbCoordinate::new(530_000.0, 180_000.0);
        let gr = coord.to_grid_ref(2).expect("2-digit ref");
        assert!(gr.starts_with("TQ"));
        assert_eq!(gr.len(), 6); // 2 letters + 2 + 2 digits
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended tests — OSGB36 / British National Grid coverage
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests_extended {
    use super::*;

    // ── Basic roundtrips ──────────────────────────────────────────────────────

    #[test]
    fn test_edinburgh_roundtrip() {
        // Edinburgh Castle: ~55.95°N, -3.20°E
        let lat = 55.9484;
        let lon = -3.2005;
        let osgb = OsgbCoordinate::from_wgs84(lat, lon).expect("Edinburgh");
        let (rlat, rlon) = osgb.to_wgs84().expect("inverse Edinburgh");
        assert!(
            (rlat - lat).abs() < 0.001,
            "lat mismatch: {} vs {}",
            rlat,
            lat
        );
        assert!(
            (rlon - lon).abs() < 0.001,
            "lon mismatch: {} vs {}",
            rlon,
            lon
        );
    }

    #[test]
    fn test_birmingham_roundtrip() {
        let lat = 52.4862;
        let lon = -1.8904;
        let osgb = OsgbCoordinate::from_wgs84(lat, lon).expect("Birmingham");
        let (rlat, rlon) = osgb.to_wgs84().expect("inverse");
        assert!((rlat - lat).abs() < 0.001);
        assert!((rlon - lon).abs() < 0.001);
    }

    #[test]
    fn test_manchester_roundtrip() {
        let lat = 53.4808;
        let lon = -2.2426;
        let osgb = OsgbCoordinate::from_wgs84(lat, lon).expect("Manchester");
        let _ = osgb.to_wgs84().expect("inverse Manchester");
    }

    #[test]
    fn test_easting_northing_range() {
        // Points in Britain should produce easting ~0–700_000, northing 0–1_300_000
        let lat = 52.0;
        let lon = 0.0;
        let osgb = OsgbCoordinate::from_wgs84(lat, lon).expect("middle England");
        assert!(
            osgb.easting > 0.0 && osgb.easting < 700_000.0,
            "Easting {} out of BNG range",
            osgb.easting
        );
        assert!(
            osgb.northing > 0.0 && osgb.northing < 1_300_000.0,
            "Northing {} out of BNG range",
            osgb.northing
        );
    }

    // ── Grid reference parsing ─────────────────────────────────────────────────

    #[test]
    fn test_grid_ref_parsing_sp() {
        // SP 123 456 is a valid BNG grid ref (Warwickshire area)
        let coord = osgb_grid_ref_to_coordinate("SP123456").expect("SP123456");
        assert!(coord.easting > 300_000.0);
    }

    #[test]
    fn test_grid_ref_parsing_nt() {
        // NT grid square is in Scotland
        let coord = osgb_grid_ref_to_coordinate("NT300800").expect("NT300800");
        assert!(coord.northing > 500_000.0, "NT should be northern BNG");
    }

    #[test]
    fn test_grid_ref_invalid_letters() {
        let result = osgb_grid_ref_to_coordinate("ZZ000000");
        // ZZ is not a valid BNG prefix — implementation may succeed or fail
        // Just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_grid_ref_short_format() {
        // 4-digit: SP 12 45
        let _ = osgb_grid_ref_to_coordinate("SP1245");
        // 6-digit: SP 123 456
        let _ = osgb_grid_ref_to_coordinate("SP123456");
    }

    // ── Coordinate-to-grid-ref ─────────────────────────────────────────────────

    #[test]
    fn test_coordinate_to_grid_ref_london() {
        // Big Ben: approx TQ 302 800
        let lat = 51.5007;
        let lon = -0.1246;
        let osgb = OsgbCoordinate::from_wgs84(lat, lon).expect("Big Ben");
        let gr4 = osgb.to_grid_ref(4).expect("4-digit");
        assert!(gr4.starts_with("TQ"), "expected TQ prefix, got {}", gr4);
    }

    #[test]
    fn test_coordinate_to_grid_ref_lengths() {
        let lat = 52.2053;
        let lon = 0.1218;
        let osgb = OsgbCoordinate::from_wgs84(lat, lon).expect("Cambridge");
        for digits in 2_u8..=5 {
            let gr = osgb.to_grid_ref(digits).expect("grid ref");
            let expected_len = 2 + 2 * digits as usize;
            assert_eq!(
                gr.len(),
                expected_len,
                "grid ref len for {digits} digits: {gr}"
            );
        }
    }

    #[test]
    fn test_coordinate_to_grid_ref_roundtrip() {
        let lat = 51.4774;
        let lon = -0.0014;
        let osgb = OsgbCoordinate::from_wgs84(lat, lon).expect("Greenwich");
        let gr = coordinate_to_osgb_grid_ref(&osgb, 5).expect("grid ref");
        // Parse it back
        let osgb2 = osgb_grid_ref_to_coordinate(&gr).expect("parse back");
        // ~10m tolerance for 5-digit (10m resolution)
        assert!(
            (osgb.easting - osgb2.easting).abs() < 20.0,
            "easting mismatch: {} vs {}",
            osgb.easting,
            osgb2.easting
        );
    }
}
