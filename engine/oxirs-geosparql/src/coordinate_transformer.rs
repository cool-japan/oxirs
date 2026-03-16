//! Coordinate Reference System transformations: WGS84 ↔ UTM ↔ WebMercator.
//!
//! Provides forward and inverse WGS84 / UTM / WebMercator transformations,
//! UTM zone helpers, and basic geodetic utilities.

use std::f64::consts::PI;

/// A WGS84 latitude/longitude coordinate pair (degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatLon {
    /// Latitude in degrees, range -90..90.
    pub lat: f64,
    /// Longitude in degrees, range -180..180.
    pub lon: f64,
}

/// A UTM coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtmCoord {
    /// Easting in metres.
    pub easting: f64,
    /// Northing in metres.
    pub northing: f64,
    /// UTM zone number (1–60).
    pub zone_number: u8,
    /// UTM zone letter (C–X, excluding I and O).
    pub zone_letter: char,
}

/// A Web Mercator (EPSG:3857) coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WebMercator {
    /// Easting in metres (X axis).
    pub x: f64,
    /// Northing in metres (Y axis).
    pub y: f64,
}

/// Transformation errors.
#[derive(Debug)]
pub enum TransformError {
    /// Input coordinate is outside the valid range for this CRS.
    OutOfRange(String),
    /// The requested CRS is not supported.
    UnsupportedCrs(String),
    /// The input value is malformed or numerically invalid.
    InvalidInput(String),
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::OutOfRange(s) => write!(f, "Out of range: {s}"),
            TransformError::UnsupportedCrs(s) => write!(f, "Unsupported CRS: {s}"),
            TransformError::InvalidInput(s) => write!(f, "Invalid input: {s}"),
        }
    }
}

impl std::error::Error for TransformError {}

// WGS84 ellipsoid parameters
const WGS84_A: f64 = 6_378_137.0; // semi-major axis (meters)
const WGS84_F: f64 = 1.0 / 298.257_223_563; // flattening

/// CRS transformation utilities (WGS84, UTM, Web Mercator).
pub struct CoordinateTransformer;

impl CoordinateTransformer {
    /// WGS84 → UTM using the standard Transverse Mercator projection.
    pub fn wgs84_to_utm(ll: LatLon) -> Result<UtmCoord, TransformError> {
        if ll.lat < -80.0 || ll.lat > 84.0 {
            return Err(TransformError::OutOfRange(format!(
                "Latitude {:.4} is out of UTM range (-80..84)",
                ll.lat
            )));
        }
        let zone_number = Self::utm_zone_number(ll.lon);
        let zone_letter = Self::utm_zone_letter(ll.lat)?;

        let lat_rad = ll.lat.to_radians();
        let lon_rad = ll.lon.to_radians();
        let lon0_rad = ((zone_number as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();

        let b = WGS84_A * (1.0 - WGS84_F);
        let e2 = 1.0 - (b / WGS84_A).powi(2); // eccentricity squared
        let e_prime2 = e2 / (1.0 - e2);
        let n_val = WGS84_A / (1.0 - e2 * lat_rad.sin().powi(2)).sqrt();
        let t = lat_rad.tan().powi(2);
        let c = e_prime2 * lat_rad.cos().powi(2);
        let a_coeff = lat_rad.cos() * (lon_rad - lon0_rad);

        let m = meridional_arc(lat_rad, WGS84_A, e2);

        // Use the standard scale factor of 0.9996 for UTM
        let k0 = 0.9996;

        let x = k0
            * n_val
            * (a_coeff
                + (1.0 - t + c) * a_coeff.powi(3) / 6.0
                + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * e_prime2) * a_coeff.powi(5)
                    / 120.0)
            + 500_000.0; // false easting

        let y_raw = k0
            * (m + n_val
                * lat_rad.tan()
                * (a_coeff.powi(2) / 2.0
                    + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a_coeff.powi(4) / 24.0
                    + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * e_prime2)
                        * a_coeff.powi(6)
                        / 720.0));

        let y = if ll.lat < 0.0 {
            y_raw + 10_000_000.0 // false northing for southern hemisphere
        } else {
            y_raw
        };

        Ok(UtmCoord {
            easting: x,
            northing: y,
            zone_number,
            zone_letter,
        })
    }

    /// UTM → WGS84.
    pub fn utm_to_wgs84(utm: UtmCoord) -> Result<LatLon, TransformError> {
        let lon0_rad = ((utm.zone_number as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();

        let b = WGS84_A * (1.0 - WGS84_F);
        let e2 = 1.0 - (b / WGS84_A).powi(2);
        let e_prime2 = e2 / (1.0 - e2);
        let k0 = 0.9996;

        let x = utm.easting - 500_000.0;
        let y = if utm.zone_letter.is_ascii_uppercase() && (utm.zone_letter as u8) < b'N' {
            utm.northing - 10_000_000.0
        } else {
            utm.northing
        };

        let m = y / k0;
        let mu =
            m / (WGS84_A * (1.0 - e2 / 4.0 - 3.0 * e2.powi(2) / 64.0 - 5.0 * e2.powi(3) / 256.0));

        let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());
        let phi1_rad = mu
            + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
            + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
            + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
            + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

        let n1 = WGS84_A / (1.0 - e2 * phi1_rad.sin().powi(2)).sqrt();
        let t1 = phi1_rad.tan().powi(2);
        let c1 = e_prime2 * phi1_rad.cos().powi(2);
        let r1 = WGS84_A * (1.0 - e2) / (1.0 - e2 * phi1_rad.sin().powi(2)).powf(1.5);
        let d = x / (n1 * k0);

        let lat_rad = phi1_rad
            - (n1 * phi1_rad.tan() / r1)
                * (d.powi(2) / 2.0
                    - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * e_prime2) * d.powi(4)
                        / 24.0
                    + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                        - 252.0 * e_prime2
                        - 3.0 * c1.powi(2))
                        * d.powi(6)
                        / 720.0);

        let lon_rad = lon0_rad
            + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
                + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2)
                    + 8.0 * e_prime2
                    + 24.0 * t1.powi(2))
                    * d.powi(5)
                    / 120.0)
                / phi1_rad.cos();

        Ok(LatLon {
            lat: lat_rad.to_degrees(),
            lon: lon_rad.to_degrees(),
        })
    }

    /// WGS84 → Web Mercator (EPSG:3857).
    pub fn wgs84_to_web_mercator(ll: LatLon) -> Result<WebMercator, TransformError> {
        if ll.lat <= -90.0 || ll.lat >= 90.0 {
            return Err(TransformError::OutOfRange(format!(
                "Latitude {:.4} is out of Web Mercator range",
                ll.lat
            )));
        }
        let lat_rad = ll.lat.to_radians();
        let lon_rad = ll.lon.to_radians();
        let x = WGS84_A * lon_rad;
        let y = WGS84_A * ((PI / 4.0 + lat_rad / 2.0).tan()).ln();
        Ok(WebMercator { x, y })
    }

    /// Web Mercator → WGS84.
    pub fn web_mercator_to_wgs84(wm: WebMercator) -> Result<LatLon, TransformError> {
        let lon_rad = wm.x / WGS84_A;
        let lat_rad = 2.0 * (wm.y / WGS84_A).exp().atan() - PI / 2.0;
        Ok(LatLon {
            lat: lat_rad.to_degrees(),
            lon: lon_rad.to_degrees(),
        })
    }

    /// Haversine distance in meters between two WGS84 points.
    pub fn haversine_distance(a: LatLon, b: LatLon) -> f64 {
        let dlat = (b.lat - a.lat).to_radians();
        let dlon = (b.lon - a.lon).to_radians();
        let lat_a = a.lat.to_radians();
        let lat_b = b.lat.to_radians();

        let h = (dlat / 2.0).sin().powi(2) + lat_a.cos() * lat_b.cos() * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * h.sqrt().asin();
        WGS84_A * c
    }

    /// Get UTM zone number for a longitude.
    pub fn utm_zone_number(lon: f64) -> u8 {
        ((lon + 180.0) / 6.0).floor() as u8 + 1
    }

    /// Get UTM zone letter for a latitude.
    pub fn utm_zone_letter(lat: f64) -> Result<char, TransformError> {
        let letters = "CDEFGHJKLMNPQRSTUVWX";
        if !(-80.0_f64..=84.0_f64).contains(&lat) {
            return Err(TransformError::OutOfRange(format!(
                "Latitude {lat:.4} is out of UTM zone letter range"
            )));
        }
        // C starts at -80, each band is 8 degrees, X is 12 degrees wide
        let idx = ((lat + 80.0) / 8.0).floor() as usize;
        let idx = idx.min(letters.len() - 1);
        letters.chars().nth(idx).ok_or_else(|| {
            TransformError::OutOfRange("Latitude out of zone letter range".to_string())
        })
    }

    /// Parse EPSG code string → human-readable CRS name.
    pub fn crs_name(epsg_code: &str) -> Option<&'static str> {
        match epsg_code {
            "EPSG:4326" => Some("WGS 84"),
            "EPSG:4979" => Some("WGS 84 (3D)"),
            "EPSG:3857" | "EPSG:900913" => Some("WGS 84 / Web Mercator"),
            "EPSG:32601" => Some("WGS 84 / UTM zone 1N"),
            "EPSG:32610" => Some("WGS 84 / UTM zone 10N"),
            "EPSG:32632" => Some("WGS 84 / UTM zone 32N"),
            "EPSG:32633" => Some("WGS 84 / UTM zone 33N"),
            "EPSG:32634" => Some("WGS 84 / UTM zone 34N"),
            "EPSG:32636" => Some("WGS 84 / UTM zone 36N"),
            "EPSG:32637" => Some("WGS 84 / UTM zone 37N"),
            "EPSG:32701" => Some("WGS 84 / UTM zone 1S"),
            "EPSG:32732" => Some("WGS 84 / UTM zone 32S"),
            "EPSG:4269" => Some("NAD83"),
            "EPSG:4230" => Some("ED50"),
            "EPSG:4258" => Some("ETRS89"),
            "EPSG:25832" => Some("ETRS89 / UTM zone 32N"),
            _ => None,
        }
    }

    /// Convert decimal degrees to DMS string.
    pub fn to_dms(degrees: f64, is_lat: bool) -> String {
        let abs_deg = degrees.abs();
        let d = abs_deg.floor() as u32;
        let min_remainder = (abs_deg - d as f64) * 60.0;
        let m = min_remainder.floor() as u32;
        let s = (min_remainder - m as f64) * 60.0;
        let direction = if is_lat {
            if degrees >= 0.0 {
                'N'
            } else {
                'S'
            }
        } else {
            if degrees >= 0.0 {
                'E'
            } else {
                'W'
            }
        };
        format!("{d:02}°{m:02}'{s:.2}\"{direction}")
    }

    /// Parse a DMS string → decimal degrees.
    ///
    /// Accepts formats like: `51°30'00.00"N`, `13°24'35.42"E`, etc.
    pub fn from_dms(dms: &str) -> Result<f64, TransformError> {
        let s = dms.trim();
        // Extract direction letter if present
        let (s, direction) = if let Some(last) = s.chars().last() {
            if matches!(last, 'N' | 'S' | 'E' | 'W' | 'n' | 's' | 'e' | 'w') {
                let dir = last.to_ascii_uppercase();
                (&s[..s.len() - 1], Some(dir))
            } else {
                (s, None)
            }
        } else {
            return Err(TransformError::InvalidInput("empty DMS string".to_string()));
        };

        // Remove degree/minute/second symbols and split
        let cleaned: String = s
            .chars()
            .map(|c| {
                if matches!(c, '°' | '\'' | '"') {
                    ' '
                } else {
                    c
                }
            })
            .collect();
        let parts: Vec<f64> = cleaned
            .split_whitespace()
            .filter(|p| !p.is_empty())
            .map(|p| {
                p.parse::<f64>().map_err(|_| {
                    TransformError::InvalidInput(format!("Cannot parse '{p}' as number"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let decimal = match parts.len() {
            1 => parts[0],
            2 => parts[0] + parts[1] / 60.0,
            3 => parts[0] + parts[1] / 60.0 + parts[2] / 3600.0,
            n => {
                return Err(TransformError::InvalidInput(format!(
                    "Expected 1-3 DMS components, got {n}"
                )));
            }
        };

        let signed = match direction {
            Some('S') | Some('W') => -decimal,
            _ => decimal,
        };
        Ok(signed)
    }
}

/// Compute the meridional arc M for the WGS84 ellipsoid.
fn meridional_arc(lat: f64, a: f64, e2: f64) -> f64 {
    let _n = e2 / (1.0 - e2).sqrt(); // third flattening alternative term (not used in series)
                                     // Standard series expansion
    a * ((1.0 - e2 / 4.0 - 3.0 * e2.powi(2) / 64.0 - 5.0 * e2.powi(3) / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e2.powi(2) / 32.0 + 45.0 * e2.powi(3) / 1024.0)
            * (2.0 * lat).sin()
        + (15.0 * e2.powi(2) / 256.0 + 45.0 * e2.powi(3) / 1024.0) * (4.0 * lat).sin()
        - (35.0 * e2.powi(3) / 3072.0) * (6.0 * lat).sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL_DEG: f64 = 0.0001; // degree tolerance for round-trips
    const _TOL_M: f64 = 5.0; // meter tolerance for round-trips

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- WGS84 → UTM round-trips ----

    #[test]
    fn test_utm_round_trip_berlin() {
        let original = LatLon {
            lat: 52.52,
            lon: 13.405,
        };
        let utm = CoordinateTransformer::wgs84_to_utm(original).expect("should succeed");
        let recovered = CoordinateTransformer::utm_to_wgs84(utm).expect("should succeed");
        assert!(
            approx_eq(recovered.lat, original.lat, TOL_DEG),
            "lat diff: {}",
            (recovered.lat - original.lat).abs()
        );
        assert!(
            approx_eq(recovered.lon, original.lon, TOL_DEG),
            "lon diff: {}",
            (recovered.lon - original.lon).abs()
        );
    }

    #[test]
    fn test_utm_round_trip_london() {
        let original = LatLon {
            lat: 51.5074,
            lon: -0.1278,
        };
        let utm = CoordinateTransformer::wgs84_to_utm(original).expect("should succeed");
        let recovered = CoordinateTransformer::utm_to_wgs84(utm).expect("should succeed");
        assert!(approx_eq(recovered.lat, original.lat, TOL_DEG));
        assert!(approx_eq(recovered.lon, original.lon, TOL_DEG));
    }

    #[test]
    fn test_utm_round_trip_new_york() {
        let original = LatLon {
            lat: 40.7128,
            lon: -74.006,
        };
        let utm = CoordinateTransformer::wgs84_to_utm(original).expect("should succeed");
        let recovered = CoordinateTransformer::utm_to_wgs84(utm).expect("should succeed");
        assert!(approx_eq(recovered.lat, original.lat, TOL_DEG));
        assert!(approx_eq(recovered.lon, original.lon, TOL_DEG));
    }

    #[test]
    fn test_utm_round_trip_tokyo() {
        let original = LatLon {
            lat: 35.6762,
            lon: 139.6503,
        };
        let utm = CoordinateTransformer::wgs84_to_utm(original).expect("should succeed");
        let recovered = CoordinateTransformer::utm_to_wgs84(utm).expect("should succeed");
        assert!(approx_eq(recovered.lat, original.lat, TOL_DEG));
        assert!(approx_eq(recovered.lon, original.lon, TOL_DEG));
    }

    #[test]
    fn test_utm_round_trip_southern_hemisphere() {
        let original = LatLon {
            lat: -33.8688,
            lon: 151.2093,
        }; // Sydney
        let utm = CoordinateTransformer::wgs84_to_utm(original).expect("should succeed");
        let recovered = CoordinateTransformer::utm_to_wgs84(utm).expect("should succeed");
        assert!(approx_eq(recovered.lat, original.lat, TOL_DEG));
        assert!(approx_eq(recovered.lon, original.lon, TOL_DEG));
    }

    #[test]
    fn test_utm_zone_assignment_berlin() {
        let utm = CoordinateTransformer::wgs84_to_utm(LatLon {
            lat: 52.52,
            lon: 13.405,
        })
        .expect("should succeed");
        assert_eq!(utm.zone_number, 33);
    }

    #[test]
    fn test_utm_zone_assignment_new_york() {
        let utm = CoordinateTransformer::wgs84_to_utm(LatLon {
            lat: 40.7128,
            lon: -74.006,
        })
        .expect("should succeed");
        assert_eq!(utm.zone_number, 18);
    }

    // ---- WGS84 → Web Mercator round-trips ----

    #[test]
    fn test_web_mercator_round_trip_berlin() {
        let original = LatLon {
            lat: 52.52,
            lon: 13.405,
        };
        let wm = CoordinateTransformer::wgs84_to_web_mercator(original).expect("should succeed");
        let recovered = CoordinateTransformer::web_mercator_to_wgs84(wm).expect("should succeed");
        assert!(approx_eq(recovered.lat, original.lat, TOL_DEG));
        assert!(approx_eq(recovered.lon, original.lon, TOL_DEG));
    }

    #[test]
    fn test_web_mercator_round_trip_origin() {
        let original = LatLon { lat: 0.0, lon: 0.0 };
        let wm = CoordinateTransformer::wgs84_to_web_mercator(original).expect("should succeed");
        assert!(approx_eq(wm.x, 0.0, 1.0));
        assert!(approx_eq(wm.y, 0.0, 1.0));
        let recovered = CoordinateTransformer::web_mercator_to_wgs84(wm).expect("should succeed");
        assert!(approx_eq(recovered.lat, 0.0, TOL_DEG));
        assert!(approx_eq(recovered.lon, 0.0, TOL_DEG));
    }

    #[test]
    fn test_web_mercator_london() {
        let original = LatLon {
            lat: 51.5074,
            lon: -0.1278,
        };
        let wm = CoordinateTransformer::wgs84_to_web_mercator(original).expect("should succeed");
        let recovered = CoordinateTransformer::web_mercator_to_wgs84(wm).expect("should succeed");
        assert!(approx_eq(recovered.lat, original.lat, TOL_DEG));
        assert!(approx_eq(recovered.lon, original.lon, TOL_DEG));
    }

    // ---- Haversine distance ----

    #[test]
    fn test_haversine_same_point() {
        let p = LatLon {
            lat: 52.0,
            lon: 13.0,
        };
        let d = CoordinateTransformer::haversine_distance(p, p);
        assert!(d < 0.001);
    }

    #[test]
    fn test_haversine_one_degree_lat() {
        // 1 degree latitude ≈ 111.32 km
        let a = LatLon { lat: 0.0, lon: 0.0 };
        let b = LatLon { lat: 1.0, lon: 0.0 };
        let d = CoordinateTransformer::haversine_distance(a, b);
        assert!(approx_eq(d, 111_320.0, 200.0), "d = {d}");
    }

    #[test]
    fn test_haversine_london_paris() {
        // ~341 km
        let london = LatLon {
            lat: 51.5074,
            lon: -0.1278,
        };
        let paris = LatLon {
            lat: 48.8566,
            lon: 2.3522,
        };
        let d = CoordinateTransformer::haversine_distance(london, paris);
        assert!(d > 300_000.0 && d < 400_000.0, "d = {d}");
    }

    #[test]
    fn test_haversine_symmetry() {
        let a = LatLon {
            lat: 52.0,
            lon: 13.0,
        };
        let b = LatLon {
            lat: 48.0,
            lon: 2.0,
        };
        let d1 = CoordinateTransformer::haversine_distance(a, b);
        let d2 = CoordinateTransformer::haversine_distance(b, a);
        assert!(approx_eq(d1, d2, 0.001));
    }

    // ---- UTM zone number ----

    #[test]
    fn test_utm_zone_number_greenwich() {
        assert_eq!(CoordinateTransformer::utm_zone_number(0.0), 31);
    }

    #[test]
    fn test_utm_zone_number_berlin() {
        assert_eq!(CoordinateTransformer::utm_zone_number(13.405), 33);
    }

    #[test]
    fn test_utm_zone_number_new_york() {
        assert_eq!(CoordinateTransformer::utm_zone_number(-74.006), 18);
    }

    #[test]
    fn test_utm_zone_number_boundary() {
        // -180° → zone 1
        assert_eq!(CoordinateTransformer::utm_zone_number(-180.0), 1);
        // 180° → zone 61 (wraps, but we just compute)
        let z = CoordinateTransformer::utm_zone_number(179.9);
        assert_eq!(z, 60);
    }

    // ---- UTM zone letter ----

    #[test]
    fn test_utm_zone_letter_equator() {
        let letter = CoordinateTransformer::utm_zone_letter(0.0).expect("should succeed");
        assert!(letter.is_ascii_uppercase());
    }

    #[test]
    fn test_utm_zone_letter_berlin() {
        let letter = CoordinateTransformer::utm_zone_letter(52.52).expect("should succeed");
        assert_eq!(letter, 'U');
    }

    #[test]
    fn test_utm_zone_letter_out_of_range_high() {
        let result = CoordinateTransformer::utm_zone_letter(85.0);
        assert!(matches!(result, Err(TransformError::OutOfRange(_))));
    }

    #[test]
    fn test_utm_zone_letter_out_of_range_low() {
        let result = CoordinateTransformer::utm_zone_letter(-85.0);
        assert!(matches!(result, Err(TransformError::OutOfRange(_))));
    }

    #[test]
    fn test_utm_zone_letter_negative_lat() {
        let letter = CoordinateTransformer::utm_zone_letter(-33.8688).expect("should succeed"); // Sydney
        assert!(letter.is_ascii_uppercase());
        assert!((letter as u8) < b'N'); // Southern hemisphere < N
    }

    // ---- DMS conversion ----

    #[test]
    fn test_to_dms_positive_lat() {
        let dms = CoordinateTransformer::to_dms(51.5, true);
        assert!(dms.contains('N'));
        assert!(dms.contains('5'));
    }

    #[test]
    fn test_to_dms_negative_lat() {
        let dms = CoordinateTransformer::to_dms(-33.8688, true);
        assert!(dms.contains('S'));
    }

    #[test]
    fn test_to_dms_positive_lon() {
        let dms = CoordinateTransformer::to_dms(13.405, false);
        assert!(dms.contains('E'));
    }

    #[test]
    fn test_to_dms_negative_lon() {
        let dms = CoordinateTransformer::to_dms(-74.006, false);
        assert!(dms.contains('W'));
    }

    #[test]
    fn test_to_dms_zero() {
        let dms = CoordinateTransformer::to_dms(0.0, true);
        assert!(dms.contains('N'));
    }

    // ---- parse DMS ----

    #[test]
    fn test_from_dms_north() {
        let deg = CoordinateTransformer::from_dms("51°30'00.00\"N").expect("should succeed");
        assert!(approx_eq(deg, 51.5, 0.01));
    }

    #[test]
    fn test_from_dms_south() {
        let deg = CoordinateTransformer::from_dms("33°52'07.68\"S").expect("should succeed");
        assert!(deg < 0.0);
        assert!(approx_eq(deg.abs(), 33.87, 0.01));
    }

    #[test]
    fn test_from_dms_east() {
        let deg = CoordinateTransformer::from_dms("13°24'18.00\"E").expect("should succeed");
        assert!(deg > 0.0);
        assert!(approx_eq(deg, 13.405, 0.01));
    }

    #[test]
    fn test_from_dms_west() {
        let deg = CoordinateTransformer::from_dms("74°00'21.60\"W").expect("should succeed");
        assert!(deg < 0.0);
    }

    #[test]
    fn test_from_dms_round_trip() {
        let original = 52.52_f64;
        let dms = CoordinateTransformer::to_dms(original, true);
        let recovered = CoordinateTransformer::from_dms(&dms).expect("should succeed");
        assert!(approx_eq(recovered, original, 0.001));
    }

    // ---- CRS name lookup ----

    #[test]
    fn test_crs_name_wgs84() {
        assert_eq!(CoordinateTransformer::crs_name("EPSG:4326"), Some("WGS 84"));
    }

    #[test]
    fn test_crs_name_web_mercator() {
        assert_eq!(
            CoordinateTransformer::crs_name("EPSG:3857"),
            Some("WGS 84 / Web Mercator")
        );
    }

    #[test]
    fn test_crs_name_utm_32n() {
        assert_eq!(
            CoordinateTransformer::crs_name("EPSG:32632"),
            Some("WGS 84 / UTM zone 32N")
        );
    }

    #[test]
    fn test_crs_name_unknown() {
        assert_eq!(CoordinateTransformer::crs_name("EPSG:9999"), None);
    }

    #[test]
    fn test_crs_name_etrs89() {
        assert_eq!(CoordinateTransformer::crs_name("EPSG:4258"), Some("ETRS89"));
    }

    // ---- Out-of-range errors ----

    #[test]
    fn test_utm_out_of_range_high_lat() {
        let result = CoordinateTransformer::wgs84_to_utm(LatLon {
            lat: 85.0,
            lon: 0.0,
        });
        assert!(matches!(result, Err(TransformError::OutOfRange(_))));
    }

    #[test]
    fn test_utm_out_of_range_low_lat() {
        let result = CoordinateTransformer::wgs84_to_utm(LatLon {
            lat: -81.0,
            lon: 0.0,
        });
        assert!(matches!(result, Err(TransformError::OutOfRange(_))));
    }

    #[test]
    fn test_web_mercator_out_of_range() {
        let result = CoordinateTransformer::wgs84_to_web_mercator(LatLon {
            lat: 90.0,
            lon: 0.0,
        });
        assert!(matches!(result, Err(TransformError::OutOfRange(_))));
    }

    // ---- TransformError display ----

    #[test]
    fn test_transform_error_display() {
        let e = TransformError::OutOfRange("test".to_string());
        assert!(format!("{e}").contains("test"));
        let e2 = TransformError::UnsupportedCrs("EPSG:9".to_string());
        assert!(format!("{e2}").contains("EPSG:9"));
        let e3 = TransformError::InvalidInput("bad".to_string());
        assert!(format!("{e3}").contains("bad"));
    }
}
