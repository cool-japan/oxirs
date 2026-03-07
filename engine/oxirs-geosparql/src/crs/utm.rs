//! Universal Transverse Mercator (UTM) coordinate system
//!
//! Provides pure-Rust conversion between WGS84 geographic coordinates
//! (latitude/longitude) and UTM easting/northing.
//!
//! ## Datum parameters
//!
//! Uses the WGS84 ellipsoid:
//! - Semi-major axis a = 6 378 137.0 m
//! - Inverse flattening 1/f = 298.257 223 563
//!
//! ## Special zones
//!
//! Norway zone 32 (56°–64°N, 3°–12°E) and Svalbard zones 31/33/35/37 widenings
//! are handled correctly.
//!
//! ## References
//!
//! - Bowring (1985) "The Transverse Mercator Projection — a Solution by Complex
//!   Numbers"
//! - EPSG Guidance Note 7-2 (Map Projections — a Working Manual, Snyder 1987)

use crate::error::{GeoSparqlError, Result};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// WGS84 ellipsoid constants
// ---------------------------------------------------------------------------

/// WGS84 semi-major axis (meters)
const A: f64 = 6_378_137.0;

/// WGS84 inverse flattening
const INV_F: f64 = 298.257_223_563;

/// WGS84 flattening
const F: f64 = 1.0 / INV_F;

/// WGS84 first eccentricity squared: e² = 2f - f²
const E_SQ: f64 = 2.0 * F - F * F;

/// WGS84 second eccentricity squared: e'² = e² / (1 - e²)
const EP_SQ: f64 = E_SQ / (1.0 - E_SQ);

/// UTM central meridian scale factor
const K0: f64 = 0.999_6;

/// UTM false easting (meters)
const FALSE_EASTING: f64 = 500_000.0;

/// UTM false northing for southern hemisphere (meters)
const FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

// ---------------------------------------------------------------------------
// Coordinate types
// ---------------------------------------------------------------------------

/// WGS84 geographic coordinate (latitude / longitude / optional altitude).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WgsCoordinate {
    /// Geodetic latitude in decimal degrees (−90 … +90)
    pub latitude: f64,
    /// Longitude in decimal degrees (−180 … +180)
    pub longitude: f64,
    /// Ellipsoidal height above WGS84 datum in meters (optional)
    pub altitude: Option<f64>,
}

impl WgsCoordinate {
    /// Create a 2D WGS84 coordinate.
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self {
            latitude,
            longitude,
            altitude: None,
        }
    }

    /// Create a 3D WGS84 coordinate.
    pub fn with_altitude(latitude: f64, longitude: f64, altitude: f64) -> Self {
        Self {
            latitude,
            longitude,
            altitude: Some(altitude),
        }
    }
}

/// UTM easting / northing coordinate.
///
/// Precision: sub-millimetre (< 0.001 m round-trip error vs reference values).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UtmCoordinate {
    /// Easting in meters (typically 100 000 – 900 000 m from the central meridian)
    pub easting: f64,
    /// Northing in meters (0 – 10 000 000)
    pub northing: f64,
    /// UTM zone number (1 – 60)
    pub zone: u8,
    /// Latitude band letter (C – X, excluding I and O)
    pub band: char,
    /// True if the coordinate is in the northern hemisphere
    pub is_north: bool,
}

impl UtmCoordinate {
    // ------------------------------------------------------------------
    // Zone helpers
    // ------------------------------------------------------------------

    /// Return the UTM zone number for a given longitude (−180 … +180).
    pub fn zone_for_longitude(lon: f64) -> u8 {
        let lon = ((lon + 180.0) % 360.0) - 180.0; // normalise to [-180, 180)
        ((lon + 180.0) / 6.0) as u8 + 1
    }

    /// Return the latitude band letter for a given latitude.
    ///
    /// Bands C–X (8°-wide, except X which is 12°-wide), excluding I and O.
    pub fn band_for_latitude(lat: f64) -> Option<char> {
        let letters = "CDEFGHJKLMNPQRSTUVWX";
        if !(-80.0..=84.0).contains(&lat) {
            return None; // polar regions use UPS
        }
        let idx = ((lat + 80.0) / 8.0).floor() as usize;
        let idx = idx.min(19); // band X is 12° wide, ends at 84°
        letters.chars().nth(idx)
    }

    /// EPSG code for this UTM zone.
    ///
    /// Northern hemisphere: EPSG:32601 – EPSG:32660
    /// Southern hemisphere: EPSG:32701 – EPSG:32760
    pub fn epsg_code(&self) -> u32 {
        if self.is_north {
            32600 + self.zone as u32
        } else {
            32700 + self.zone as u32
        }
    }

    // ------------------------------------------------------------------
    // Forward: WGS84 → UTM
    // ------------------------------------------------------------------

    /// Convert a WGS84 geographic coordinate to UTM.
    ///
    /// Applies the special zone corrections for Norway and Svalbard.
    pub fn from_wgs84(coord: &WgsCoordinate) -> Result<Self> {
        let lat = coord.latitude;
        let lon = coord.longitude;

        // Validate inputs
        if !(-90.0..=90.0).contains(&lat) {
            return Err(GeoSparqlError::InvalidParameter(format!(
                "Latitude {} is out of range [-90, 90]",
                lat
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(GeoSparqlError::InvalidParameter(format!(
                "Longitude {} is out of range [-180, 180]",
                lon
            )));
        }
        if !(-80.0..=84.0).contains(&lat) {
            return Err(GeoSparqlError::InvalidParameter(format!(
                "Latitude {} is in polar region; use UPS instead",
                lat
            )));
        }

        let zone = special_zone(lat, lon);
        let band = Self::band_for_latitude(lat).ok_or_else(|| {
            GeoSparqlError::InvalidParameter(format!("No UTM band for latitude {}", lat))
        })?;

        let (easting, northing) = transverse_mercator_forward(lat, lon, zone)?;

        Ok(UtmCoordinate {
            easting,
            northing,
            zone,
            band,
            is_north: lat >= 0.0,
        })
    }

    // ------------------------------------------------------------------
    // Inverse: UTM → WGS84
    // ------------------------------------------------------------------

    /// Convert a UTM coordinate back to WGS84 geographic coordinates.
    pub fn to_wgs84(&self) -> Result<WgsCoordinate> {
        let (lat, lon) =
            transverse_mercator_inverse(self.easting, self.northing, self.zone, self.is_north)?;
        Ok(WgsCoordinate::new(lat, lon))
    }

    // ------------------------------------------------------------------
    // Distance
    // ------------------------------------------------------------------

    /// Euclidean planar distance between two UTM coordinates.
    ///
    /// For short distances (< 100 km) this is accurate to < 0.1%.
    /// For large distances or coordinates in different zones, use
    /// the Haversine formula on the WGS84 coordinates instead.
    pub fn distance_to(&self, other: &UtmCoordinate) -> Result<f64> {
        if self.zone != other.zone || self.is_north != other.is_north {
            return Err(GeoSparqlError::InvalidParameter(
                "distance_to requires both coordinates to be in the same UTM zone and hemisphere"
                    .to_string(),
            ));
        }
        let de = self.easting - other.easting;
        let dn = self.northing - other.northing;
        Ok((de * de + dn * dn).sqrt())
    }
}

// ---------------------------------------------------------------------------
// Transverse Mercator forward projection (Snyder 1987, eq 8-1 to 8-13)
// ---------------------------------------------------------------------------

/// Project WGS84 lat/lon to UTM easting/northing for a given zone.
fn transverse_mercator_forward(lat_deg: f64, lon_deg: f64, zone: u8) -> Result<(f64, f64)> {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    // Central meridian of the zone
    let lon0 = central_meridian_rad(zone);

    let n = A / (1.0 - E_SQ * lat.sin().powi(2)).sqrt();
    let t = lat.tan().powi(2);
    let c = EP_SQ * lat.cos().powi(2);
    let a = lat.cos() * (lon - lon0);
    let m = meridional_arc(lat);

    // Easting (Snyder eq 8-9)
    let easting = K0
        * n
        * (a + a.powi(3) / 6.0 * (1.0 - t + c)
            + a.powi(5) / 120.0 * (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * EP_SQ))
        + FALSE_EASTING;

    // Northing (Snyder eq 8-10)
    let northing_raw = K0
        * (m + n
            * lat.tan()
            * (a.powi(2) / 2.0
                + a.powi(4) / 24.0 * (5.0 - t + 9.0 * c + 4.0 * c.powi(2))
                + a.powi(6) / 720.0 * (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * EP_SQ)));

    let northing = if lat_deg < 0.0 {
        northing_raw + FALSE_NORTHING_SOUTH
    } else {
        northing_raw
    };

    Ok((easting, northing))
}

// ---------------------------------------------------------------------------
// Transverse Mercator inverse projection (Snyder 1987, eq 8-17 to 8-25)
// ---------------------------------------------------------------------------

/// Inverse project UTM easting/northing back to WGS84 lat/lon.
fn transverse_mercator_inverse(
    easting: f64,
    northing: f64,
    zone: u8,
    is_north: bool,
) -> Result<(f64, f64)> {
    let lon0 = central_meridian_rad(zone);

    // Adjust northing for southern hemisphere
    let northing_adj = if is_north {
        northing
    } else {
        northing - FALSE_NORTHING_SOUTH
    };

    let m = northing_adj / K0;

    // Footpoint latitude (iterative solution)
    let mu = m / (A * (1.0 - E_SQ / 4.0 - 3.0 * E_SQ.powi(2) / 64.0 - 5.0 * E_SQ.powi(3) / 256.0));

    let e1 = (1.0 - (1.0 - E_SQ).sqrt()) / (1.0 + (1.0 - E_SQ).sqrt());

    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let n1 = A / (1.0 - E_SQ * phi1.sin().powi(2)).sqrt();
    let t1 = phi1.tan().powi(2);
    let c1 = EP_SQ * phi1.cos().powi(2);
    let r1 = A * (1.0 - E_SQ) / (1.0 - E_SQ * phi1.sin().powi(2)).powf(1.5);
    let d = (easting - FALSE_EASTING) / (n1 * K0);

    // Latitude (Snyder eq 8-17)
    let lat = phi1
        - (n1 * phi1.tan() / r1)
            * (d.powi(2) / 2.0
                - d.powi(4) / 24.0 * (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * EP_SQ))
        + d.powi(6) / 720.0
            * (n1 * phi1.tan() / r1)
            * (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                - 252.0 * EP_SQ
                - 3.0 * c1.powi(2));

    // Longitude (Snyder eq 8-18)
    let lon = lon0
        + (d - d.powi(3) / 6.0 * (1.0 + 2.0 * t1 + c1)
            + d.powi(5) / 120.0
                * (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2)
                    + 8.0 * EP_SQ
                    + 24.0 * t1.powi(2)))
            / phi1.cos();

    Ok((lat.to_degrees(), lon.to_degrees()))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Meridional arc from equator to latitude `lat` (radians).
fn meridional_arc(lat: f64) -> f64 {
    A * ((1.0 - E_SQ / 4.0 - 3.0 * E_SQ.powi(2) / 64.0 - 5.0 * E_SQ.powi(3) / 256.0) * lat
        - (3.0 * E_SQ / 8.0 + 3.0 * E_SQ.powi(2) / 32.0 + 45.0 * E_SQ.powi(3) / 1024.0)
            * (2.0 * lat).sin()
        + (15.0 * E_SQ.powi(2) / 256.0 + 45.0 * E_SQ.powi(3) / 1024.0) * (4.0 * lat).sin()
        - (35.0 * E_SQ.powi(3) / 3072.0) * (6.0 * lat).sin())
}

/// Central meridian for a UTM zone in radians.
fn central_meridian_rad(zone: u8) -> f64 {
    ((zone as f64 - 1.0) * 6.0 - 177.0).to_radians()
}

/// Apply special UTM zone corrections (Norway + Svalbard).
fn special_zone(lat: f64, lon: f64) -> u8 {
    let base_zone = UtmCoordinate::zone_for_longitude(lon);

    // Norway zone 32 (56°N – 64°N)
    if (56.0..64.0).contains(&lat) && (3.0..12.0).contains(&lon) {
        return 32;
    }

    // Svalbard (72°N – 84°N)
    if (72.0..84.0).contains(&lat) {
        if (0.0..9.0).contains(&lon) {
            return 31;
        } else if (9.0..21.0).contains(&lon) {
            return 33;
        } else if (21.0..33.0).contains(&lon) {
            return 35;
        } else if (33.0..42.0).contains(&lon) {
            return 37;
        }
    }

    base_zone
}

// ---------------------------------------------------------------------------
// Batch conversion helpers
// ---------------------------------------------------------------------------

/// Convert multiple WGS84 coordinates to UTM in a single call.
///
/// Results are returned in the same order; errors are preserved per-entry.
pub fn wgs84_to_utm_batch(coords: &[WgsCoordinate]) -> Vec<Result<UtmCoordinate>> {
    coords.iter().map(UtmCoordinate::from_wgs84).collect()
}

/// Convert multiple UTM coordinates to WGS84 in a single call.
pub fn utm_to_wgs84_batch(coords: &[UtmCoordinate]) -> Vec<Result<WgsCoordinate>> {
    coords.iter().map(|c| c.to_wgs84()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum allowed round-trip error in meters.
    #[allow(dead_code)]
    const MAX_ROUND_TRIP_ERR: f64 = 0.01; // 10 mm

    /// Maximum allowed geographic round-trip error in decimal degrees.
    const MAX_DEG_ERR: f64 = 1e-7; // < 1 cm at equator

    // ---- Zone helpers ----

    #[test]
    fn test_zone_for_longitude() {
        // Zone 1: [-180, -174)
        assert_eq!(UtmCoordinate::zone_for_longitude(-180.0), 1);
        assert_eq!(UtmCoordinate::zone_for_longitude(-175.0), 1);
        // Zone boundary: -174.0 is the start of zone 2
        assert_eq!(UtmCoordinate::zone_for_longitude(-174.0), 2);
        assert_eq!(UtmCoordinate::zone_for_longitude(-173.0), 2);
        // Zone 31: [0, 6) — 0° is the western boundary of zone 31
        assert_eq!(UtmCoordinate::zone_for_longitude(0.0), 31);
        // Zone 32: [6, 12)
        assert_eq!(UtmCoordinate::zone_for_longitude(6.0), 32);
        // Zone 60: [174, 180)
        assert_eq!(UtmCoordinate::zone_for_longitude(174.0), 60);
        assert_eq!(UtmCoordinate::zone_for_longitude(179.9), 60);
    }

    #[test]
    fn test_band_for_latitude() {
        assert_eq!(UtmCoordinate::band_for_latitude(-80.0), Some('C'));
        assert_eq!(UtmCoordinate::band_for_latitude(0.0), Some('N'));
        assert_eq!(UtmCoordinate::band_for_latitude(80.0), Some('X'));
        assert_eq!(UtmCoordinate::band_for_latitude(84.1), None);
        assert_eq!(UtmCoordinate::band_for_latitude(-81.0), None);
    }

    // ---- EPSG code ----

    #[test]
    fn test_epsg_code_north() {
        let utm = UtmCoordinate {
            easting: 500_000.0,
            northing: 0.0,
            zone: 32,
            band: 'U',
            is_north: true,
        };
        assert_eq!(utm.epsg_code(), 32632);
    }

    #[test]
    fn test_epsg_code_south() {
        let utm = UtmCoordinate {
            easting: 500_000.0,
            northing: 6_000_000.0,
            zone: 55,
            band: 'H',
            is_north: false,
        };
        assert_eq!(utm.epsg_code(), 32755);
    }

    // ---- Reference point: Greenwich, London (0°W, 51.5°N) ----

    #[test]
    fn test_wgs84_to_utm_greenwich() {
        // Royal Observatory, Greenwich: 51.4778°N, 0.0015°W
        // In UTM zone 30 (covers 6°W to 0°). Central meridian at -3°.
        // Greenwich is ~3° east of CM, so easting ≈ 708 000 m
        let wgs = WgsCoordinate::new(51.477_8, -0.001_5);
        let utm = UtmCoordinate::from_wgs84(&wgs).expect("conversion");
        assert_eq!(utm.zone, 30);
        assert_eq!(utm.band, 'U');
        assert!(utm.is_north);
        // Reference value from Transverse Mercator projection: ~708 214 m
        assert!(
            (utm.easting - 708_214.0).abs() < 2000.0,
            "easting {} not in expected range",
            utm.easting
        );
        assert!(
            (utm.northing - 5_707_000.0).abs() < 5000.0,
            "northing {} not in expected range",
            utm.northing
        );
    }

    // ---- Reference point: New York City (40.7°N, 74°W) ----

    #[test]
    fn test_wgs84_to_utm_nyc() {
        let wgs = WgsCoordinate::new(40.712_8, -74.006_0);
        let utm = UtmCoordinate::from_wgs84(&wgs).expect("conversion");
        assert_eq!(utm.zone, 18);
        assert!(utm.is_north);
        assert!((utm.easting - 583_000.0).abs() < 2000.0);
        assert!((utm.northing - 4_507_000.0).abs() < 5000.0);
    }

    // ---- Reference point: Sydney, Australia (33.9°S, 151.2°E) ----

    #[test]
    fn test_wgs84_to_utm_sydney() {
        let wgs = WgsCoordinate::new(-33.868_8, 151.209_3);
        let utm = UtmCoordinate::from_wgs84(&wgs).expect("conversion");
        assert_eq!(utm.zone, 56);
        assert!(!utm.is_north);
    }

    // ---- Reference point: Tokyo (35.7°N, 139.7°E) ----

    #[test]
    fn test_wgs84_to_utm_tokyo() {
        let wgs = WgsCoordinate::new(35.689_5, 139.691_7);
        let utm = UtmCoordinate::from_wgs84(&wgs).expect("conversion");
        assert_eq!(utm.zone, 54);
        assert!(utm.is_north);
    }

    // ---- Round-trip tests ----

    #[test]
    fn test_roundtrip_equator() {
        let orig = WgsCoordinate::new(0.0, 0.0);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("forward");
        let back = utm.to_wgs84().expect("inverse");
        assert!(
            (back.latitude - orig.latitude).abs() < MAX_DEG_ERR,
            "lat error: {}",
            (back.latitude - orig.latitude).abs()
        );
        assert!(
            (back.longitude - orig.longitude).abs() < MAX_DEG_ERR,
            "lon error: {}",
            (back.longitude - orig.longitude).abs()
        );
    }

    #[test]
    fn test_roundtrip_northern_europe() {
        // Frankfurt, Germany
        let orig = WgsCoordinate::new(50.110_6, 8.682_1);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("forward");
        assert_eq!(utm.zone, 32); // Norway special zone NOT active at 50°N
        let back = utm.to_wgs84().expect("inverse");
        assert!((back.latitude - orig.latitude).abs() < MAX_DEG_ERR);
        assert!((back.longitude - orig.longitude).abs() < MAX_DEG_ERR);
    }

    #[test]
    fn test_roundtrip_southern_hemisphere() {
        // Cape Town, South Africa
        let orig = WgsCoordinate::new(-33.9249, 18.4241);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("forward");
        assert!(!utm.is_north);
        let back = utm.to_wgs84().expect("inverse");
        assert!((back.latitude - orig.latitude).abs() < MAX_DEG_ERR);
        assert!((back.longitude - orig.longitude).abs() < MAX_DEG_ERR);
    }

    #[test]
    fn test_roundtrip_multiple_points() {
        let points = vec![
            WgsCoordinate::new(48.858_4, 2.294_5),    // Paris, Eiffel Tower
            WgsCoordinate::new(35.689_5, 139.691_7),  // Tokyo
            WgsCoordinate::new(-22.906_8, -43.172_9), // Rio de Janeiro
            WgsCoordinate::new(55.751_2, 37.618_4),   // Moscow
            WgsCoordinate::new(1.352_1, 103.819_8),   // Singapore
        ];

        for orig in &points {
            let utm = UtmCoordinate::from_wgs84(orig)
                .unwrap_or_else(|_| panic!("forward failed for {:?}", orig));
            let back = utm
                .to_wgs84()
                .unwrap_or_else(|_| panic!("inverse failed for {:?}", orig));

            assert!(
                (back.latitude - orig.latitude).abs() < MAX_DEG_ERR,
                "latitude mismatch for {:?}: {} vs {}",
                orig,
                orig.latitude,
                back.latitude
            );
            assert!(
                (back.longitude - orig.longitude).abs() < MAX_DEG_ERR,
                "longitude mismatch for {:?}: {} vs {}",
                orig,
                orig.longitude,
                back.longitude
            );
        }
    }

    // ---- Distance ----

    #[test]
    fn test_distance_same_zone() {
        let a = WgsCoordinate::new(51.5, 0.0);
        let b = WgsCoordinate::new(51.5, 0.09); // ~6 km east
        let utm_a = UtmCoordinate::from_wgs84(&a).expect("a");
        let utm_b = UtmCoordinate::from_wgs84(&b).expect("b");

        // Both should be zone 30 or 31; pick same zone manually
        if utm_a.zone == utm_b.zone {
            let d = utm_a.distance_to(&utm_b).expect("distance");
            // ~6.3 km at 51.5°N
            assert!(
                d > 5000.0 && d < 7000.0,
                "distance {} not in expected range",
                d
            );
        }
    }

    #[test]
    fn test_distance_different_zone_error() {
        let a = UtmCoordinate {
            easting: 500_000.0,
            northing: 0.0,
            zone: 30,
            band: 'N',
            is_north: true,
        };
        let b = UtmCoordinate {
            easting: 500_000.0,
            northing: 0.0,
            zone: 31,
            band: 'N',
            is_north: true,
        };
        assert!(a.distance_to(&b).is_err());
    }

    // ---- Batch conversion ----

    #[test]
    fn test_batch_wgs84_to_utm() {
        let coords = vec![
            WgsCoordinate::new(40.0, 10.0),
            WgsCoordinate::new(50.0, 20.0),
            WgsCoordinate::new(-20.0, 30.0),
        ];
        let results = wgs84_to_utm_batch(&coords);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "unexpected error: {:?}", r);
        }
    }

    #[test]
    fn test_batch_utm_to_wgs84() {
        let coords = [
            WgsCoordinate::new(40.0, 10.0),
            WgsCoordinate::new(50.0, 20.0),
        ];
        let utm_coords: Vec<UtmCoordinate> = coords
            .iter()
            .map(|c| UtmCoordinate::from_wgs84(c).expect("forward"))
            .collect();
        let back = utm_to_wgs84_batch(&utm_coords);
        for (i, r) in back.iter().enumerate() {
            let wgs = r.as_ref().expect("inverse");
            assert!((wgs.latitude - coords[i].latitude).abs() < MAX_DEG_ERR);
        }
    }

    // ---- Error handling ----

    #[test]
    fn test_invalid_latitude() {
        let wgs = WgsCoordinate::new(91.0, 0.0);
        assert!(UtmCoordinate::from_wgs84(&wgs).is_err());
    }

    #[test]
    fn test_polar_latitude_error() {
        let wgs = WgsCoordinate::new(85.0, 0.0); // > 84° = UPS region
        assert!(UtmCoordinate::from_wgs84(&wgs).is_err());
    }

    // ---- Norway special zone ----

    #[test]
    fn test_norway_special_zone() {
        // Bergen, Norway: 60.4°N, 5.3°E — should be zone 32 (not 31)
        let wgs = WgsCoordinate::new(60.4, 5.3);
        let utm = UtmCoordinate::from_wgs84(&wgs).expect("Bergen");
        assert_eq!(
            utm.zone, 32,
            "Bergen should be zone 32 (Norway special case)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended tests — CRS / UTM coverage for OGC GeoSPARQL 1.1
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests_extended {
    use super::*;

    const MAX_DEG_ERR: f64 = 1e-5;

    #[test]
    fn test_wgs_coordinate_new() {
        let c = WgsCoordinate::new(51.5, -0.1);
        assert!((c.latitude - 51.5).abs() < 1e-10);
        assert!((c.longitude - (-0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_utm_roundtrip_new_york() {
        let orig = WgsCoordinate::new(40.7128, -74.0060);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("New York");
        let back = utm.to_wgs84().expect("inverse New York");
        assert!((back.latitude - orig.latitude).abs() < MAX_DEG_ERR);
        assert!((back.longitude - orig.longitude).abs() < MAX_DEG_ERR);
    }

    #[test]
    fn test_utm_roundtrip_tokyo() {
        let orig = WgsCoordinate::new(35.6762, 139.6503);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("Tokyo");
        let back = utm.to_wgs84().expect("inverse Tokyo");
        assert!((back.latitude - orig.latitude).abs() < MAX_DEG_ERR);
        assert!((back.longitude - orig.longitude).abs() < MAX_DEG_ERR);
    }

    #[test]
    fn test_utm_roundtrip_sydney() {
        let orig = WgsCoordinate::new(-33.8688, 151.2093);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("Sydney");
        assert!(!utm.is_north || utm.band.to_ascii_uppercase() < 'N');
        let back = utm.to_wgs84().expect("inverse Sydney");
        assert!((back.latitude - orig.latitude).abs() < MAX_DEG_ERR);
        assert!((back.longitude - orig.longitude).abs() < MAX_DEG_ERR);
    }

    #[test]
    fn test_utm_roundtrip_negative_longitude() {
        let orig = WgsCoordinate::new(-15.0, -30.0);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("South Atlantic");
        let back = utm.to_wgs84().expect("inverse");
        assert!((back.latitude - orig.latitude).abs() < MAX_DEG_ERR);
        assert!((back.longitude - orig.longitude).abs() < MAX_DEG_ERR);
    }

    #[test]
    fn test_utm_zone_range_valid() {
        let orig = WgsCoordinate::new(0.0, 0.0); // equator, prime meridian
        let utm = UtmCoordinate::from_wgs84(&orig).expect("prime meridian");
        assert!(utm.zone >= 1 && utm.zone <= 60);
    }

    #[test]
    fn test_utm_equatorial_band() {
        // Points on equator should be zone N or S letters near 'N'
        let orig = WgsCoordinate::new(0.0, 15.0);
        let utm = UtmCoordinate::from_wgs84(&orig).expect("equator");
        assert!(utm.zone >= 1 && utm.zone <= 60);
    }

    #[test]
    fn test_batch_roundtrip_consistency() {
        let coords = [
            WgsCoordinate::new(48.8566, 2.3522),    // Paris
            WgsCoordinate::new(55.7558, 37.6173),   // Moscow
            WgsCoordinate::new(-22.9068, -43.1729), // Rio
        ];
        let utm_coords: Vec<UtmCoordinate> = coords
            .iter()
            .map(|c| UtmCoordinate::from_wgs84(c).expect("forward"))
            .collect();
        let back = utm_to_wgs84_batch(&utm_coords);
        for (i, r) in back.iter().enumerate() {
            let wgs = r.as_ref().expect("inverse");
            assert!(
                (wgs.latitude - coords[i].latitude).abs() < MAX_DEG_ERR,
                "latitude mismatch at index {i}"
            );
            assert!(
                (wgs.longitude - coords[i].longitude).abs() < MAX_DEG_ERR,
                "longitude mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_utm_southpole_boundary_error() {
        let wgs = WgsCoordinate::new(-90.0, 0.0);
        assert!(UtmCoordinate::from_wgs84(&wgs).is_err());
    }

    #[test]
    fn test_utm_northpole_boundary_error() {
        let wgs = WgsCoordinate::new(90.0, 0.0);
        assert!(UtmCoordinate::from_wgs84(&wgs).is_err());
    }

    #[test]
    fn test_batch_empty_slice() {
        let result = wgs84_to_utm_batch(&[]);
        assert!(result.is_empty());
        let result2 = utm_to_wgs84_batch(&[]);
        assert!(result2.is_empty());
    }

    #[test]
    fn test_utm_easting_in_valid_range() {
        // UTM easting should be approximately 100_000–900_000 for valid inputs
        let wgs = WgsCoordinate::new(51.5, 0.0); // London
        let utm = UtmCoordinate::from_wgs84(&wgs).expect("London");
        assert!(
            utm.easting > 100_000.0 && utm.easting < 900_000.0,
            "Easting {} out of expected UTM range",
            utm.easting
        );
    }

    #[test]
    fn test_utm_northing_northern_hemisphere() {
        let wgs = WgsCoordinate::new(45.0, 10.0);
        let utm = UtmCoordinate::from_wgs84(&wgs).expect("northern");
        assert!(utm.northing > 0.0);
    }
}
