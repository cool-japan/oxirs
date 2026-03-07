//! Coordinate Reference System (CRS) implementations
//!
//! This module provides pure-Rust coordinate reference system support:
//!
//! - [`utm`]: Universal Transverse Mercator (UTM) — all 60 zones
//! - [`osgb36`]: Ordnance Survey GB 1936 (OSGB36 / ETRS89-compatible)
//! - [`CrsKind`]: Named CRS enum (Wgs84, WebMercator, Utm, Custom)
//! - [`CrsTransformer`]: Pure-Rust coordinate transformations between CRS
//! - [`GeometryWithCrs`]: Geometry paired with its CRS

pub mod crs_literal;
pub mod osgb36;
pub mod utm;

pub use crs_literal::{
    encode_crs_wkt_literal, epsg_uri_for_kind, parse_crs_uri, parse_crs_wkt_literal,
    CrsGeometryTransformer, CrsLiteral, CRS84_URI, EPSG_URI_PREFIX, GEO_CRS, GEO_WKT_LITERAL,
};
pub use osgb36::{coordinate_to_osgb_grid_ref, osgb_grid_ref_to_coordinate, OsgbCoordinate};
pub use utm::{utm_to_wgs84_batch, wgs84_to_utm_batch, UtmCoordinate, WgsCoordinate};

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;

// ─────────────────────────────────────────────────────────────────────────────
// CrsKind — named coordinate reference systems
// ─────────────────────────────────────────────────────────────────────────────

/// Named coordinate reference system identifier.
///
/// Provides a strongly-typed alternative to raw URI strings for the most
/// common CRS used in GeoSPARQL work.
///
/// # Mapping to EPSG codes
///
/// | Variant | EPSG | Description |
/// |---------|------|-------------|
/// | `Wgs84` | 4326 | WGS84 geographic (lon/lat, degrees) |
/// | `WebMercator` | 3857 | Web/Pseudo-Mercator (metres) |
/// | `Utm { zone, hemisphere }` | 326xx / 327xx | Universal Transverse Mercator |
/// | `Custom { epsg }` | — | Any other EPSG code |
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CrsKind {
    /// WGS84 geographic coordinates — EPSG:4326.
    ///
    /// Coordinates are in decimal degrees: (longitude, latitude).
    Wgs84,

    /// Web Mercator / Pseudo-Mercator — EPSG:3857.
    ///
    /// Coordinates are in metres from the prime meridian / equator.
    /// Used by Google Maps, OpenStreetMap, and most web mapping services.
    WebMercator,

    /// Universal Transverse Mercator — EPSG:326xx (north) / 327xx (south).
    ///
    /// `zone` is the UTM zone number (1–60).
    /// `hemisphere` is `'N'` for northern or `'S'` for southern hemisphere.
    Utm {
        /// UTM zone number (1–60)
        zone: u8,
        /// Hemisphere: `'N'` (north) or `'S'` (south)
        hemisphere: char,
    },

    /// Any other EPSG-coded coordinate system.
    Custom {
        /// EPSG authority code
        epsg: u32,
    },
}

impl CrsKind {
    /// Return the EPSG code for this CRS, or `None` for unknown systems.
    pub fn epsg_code(&self) -> Option<u32> {
        match self {
            CrsKind::Wgs84 => Some(4326),
            CrsKind::WebMercator => Some(3857),
            CrsKind::Utm { zone, hemisphere } => {
                if *hemisphere == 'N' || *hemisphere == 'n' {
                    Some(32600 + *zone as u32)
                } else {
                    Some(32700 + *zone as u32)
                }
            }
            CrsKind::Custom { epsg } => Some(*epsg),
        }
    }

    /// Return the OGC URI for this CRS.
    pub fn to_uri(&self) -> String {
        match self.epsg_code() {
            Some(code) => format!("http://www.opengis.net/def/crs/EPSG/0/{code}"),
            None => "http://www.opengis.net/def/crs/OGC/1.3/CRS84".to_string(),
        }
    }

    /// Convert to the `Crs` URI type used by `Geometry`.
    pub fn to_geometry_crs(&self) -> crate::geometry::Crs {
        crate::geometry::Crs::new(self.to_uri())
    }

    /// Try to build a `CrsKind` from an EPSG code.
    ///
    /// Recognises well-known codes directly; all others become `Custom`.
    pub fn from_epsg(epsg: u32) -> Self {
        match epsg {
            4326 => CrsKind::Wgs84,
            3857 | 900913 => CrsKind::WebMercator,
            32601..=32660 => CrsKind::Utm {
                zone: (epsg - 32600) as u8,
                hemisphere: 'N',
            },
            32701..=32760 => CrsKind::Utm {
                zone: (epsg - 32700) as u8,
                hemisphere: 'S',
            },
            other => CrsKind::Custom { epsg: other },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CrsTransformer — coordinate projection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pure-Rust coordinate transformations between supported CRS.
///
/// All formulae use the WGS84 ellipsoid (a = 6 378 137 m, f = 1/298.257 223 563)
/// and are accurate to sub-millimetre level within the valid domain of each
/// projection.
///
/// # Supported transformations
///
/// | Method | From → To |
/// |--------|-----------|
/// | `wgs84_to_web_mercator` | WGS84 (deg) → Web Mercator (m) |
/// | `web_mercator_to_wgs84` | Web Mercator (m) → WGS84 (deg) |
/// | `wgs84_to_utm` | WGS84 (deg) → UTM easting/northing + zone |
/// | `utm_to_wgs84` | UTM easting/northing + zone → WGS84 (deg) |
pub struct CrsTransformer;

/// WGS84 semi-major axis (metres)
const A_WGS84: f64 = 6_378_137.0;

impl CrsTransformer {
    /// Project WGS84 geographic coordinates to Web Mercator (EPSG:3857).
    ///
    /// ## Arguments
    ///
    /// - `lon` — longitude in decimal degrees (−180 … +180)
    /// - `lat` — latitude in decimal degrees (−85.051129… … +85.051129…)
    ///
    /// ## Returns
    ///
    /// `(x, y)` in metres from the origin.
    ///
    /// ## Errors
    ///
    /// Returns an error if the latitude is outside the Web Mercator valid range
    /// (approximately ±85.051129°).
    ///
    /// ## Example
    ///
    /// ```
    /// use oxirs_geosparql::crs::CrsTransformer;
    ///
    /// // London (roughly)
    /// let (x, y) = CrsTransformer::wgs84_to_web_mercator(-0.1276, 51.5074).unwrap();
    /// assert!((x - (-14210.0)).abs() < 500.0);
    /// assert!((y - 6711790.0).abs() < 500.0);
    /// ```
    pub fn wgs84_to_web_mercator(lon: f64, lat: f64) -> Result<(f64, f64)> {
        const MAX_LAT: f64 = 85.051_128_779_806_59;

        if lat.abs() > MAX_LAT {
            return Err(GeoSparqlError::InvalidInput(format!(
                "Latitude {lat} is outside Web Mercator valid range (±{MAX_LAT}°)"
            )));
        }

        let x = A_WGS84 * lon.to_radians();
        let lat_rad = lat.to_radians();
        let y = A_WGS84 * ((std::f64::consts::FRAC_PI_4 + lat_rad / 2.0).tan().ln());

        Ok((x, y))
    }

    /// Convert Web Mercator (EPSG:3857) coordinates back to WGS84.
    ///
    /// ## Arguments
    ///
    /// - `x` — easting in metres
    /// - `y` — northing in metres
    ///
    /// ## Returns
    ///
    /// `(lon, lat)` in decimal degrees.
    ///
    /// ## Example
    ///
    /// ```
    /// use oxirs_geosparql::crs::CrsTransformer;
    ///
    /// let (lon, lat) = CrsTransformer::web_mercator_to_wgs84(0.0, 0.0).unwrap();
    /// assert!((lon - 0.0).abs() < 1e-10);
    /// assert!((lat - 0.0).abs() < 1e-10);
    /// ```
    pub fn web_mercator_to_wgs84(x: f64, y: f64) -> Result<(f64, f64)> {
        let lon = x.to_degrees() / A_WGS84;
        let lat_rad = 2.0 * (y / A_WGS84).exp().atan() - std::f64::consts::FRAC_PI_2;
        Ok((lon, lat_rad.to_degrees()))
    }

    /// Convert WGS84 geographic coordinates to UTM easting/northing.
    ///
    /// Delegates to the existing `crate::crs::utm::wgs84_to_utm_batch`
    /// implementation which handles all 60 zones plus the Norway/Svalbard
    /// special-zone exceptions.
    ///
    /// ## Arguments
    ///
    /// - `lon` — longitude in decimal degrees (−180 … +180)
    /// - `lat` — latitude in decimal degrees (−80 … +84)
    ///
    /// ## Returns
    ///
    /// `(easting_m, northing_m, zone, hemisphere)` where:
    /// - `easting_m` — UTM easting in metres (typically 100 000 – 900 000)
    /// - `northing_m` — UTM northing in metres (0 – 10 000 000)
    /// - `zone` — UTM zone number (1 – 60)
    /// - `hemisphere` — `'N'` or `'S'`
    ///
    /// ## Example
    ///
    /// ```
    /// use oxirs_geosparql::crs::CrsTransformer;
    ///
    /// // New York City (approximately)
    /// let (e, n, zone, hemi) = CrsTransformer::wgs84_to_utm(-74.006, 40.7128).unwrap();
    /// assert_eq!(zone, 18);
    /// assert_eq!(hemi, 'N');
    /// assert!((e - 583_960.0).abs() < 1_000.0, "easting ~ 584000 m");
    /// assert!((n - 4_507_523.0).abs() < 1_000.0, "northing ~ 4507523 m");
    /// ```
    pub fn wgs84_to_utm(lon: f64, lat: f64) -> Result<(f64, f64, u8, char)> {
        use crate::crs::utm::{wgs84_to_utm_batch, WgsCoordinate};

        let input = WgsCoordinate::new(lat, lon);
        let results = wgs84_to_utm_batch(&[input]);

        results
            .into_iter()
            .next()
            .and_then(|r| r.ok())
            .map(|utm| {
                let hemi = if utm.is_north { 'N' } else { 'S' };
                (utm.easting, utm.northing, utm.zone, hemi)
            })
            .ok_or_else(|| {
                GeoSparqlError::InvalidInput(format!("Cannot convert ({lon}, {lat}) to UTM"))
            })
    }

    /// Convert UTM easting/northing back to WGS84 geographic coordinates.
    ///
    /// ## Arguments
    ///
    /// - `easting` — UTM easting in metres
    /// - `northing` — UTM northing in metres
    /// - `zone` — UTM zone number (1–60)
    /// - `hemisphere` — `'N'` for northern, `'S'` for southern hemisphere
    ///
    /// ## Returns
    ///
    /// `(lon, lat)` in decimal degrees.
    pub fn utm_to_wgs84(
        easting: f64,
        northing: f64,
        zone: u8,
        hemisphere: char,
    ) -> Result<(f64, f64)> {
        use crate::crs::utm::{utm_to_wgs84_batch, UtmCoordinate};

        let is_north = hemisphere == 'N' || hemisphere == 'n';

        let input = UtmCoordinate {
            easting,
            northing,
            zone,
            band: if is_north { 'N' } else { 'S' },
            is_north,
        };

        let results = utm_to_wgs84_batch(&[input]);

        results
            .into_iter()
            .next()
            .and_then(|r| r.ok())
            .map(|wgs| (wgs.longitude, wgs.latitude))
            .ok_or_else(|| {
                GeoSparqlError::InvalidInput(format!(
                    "Cannot convert UTM ({easting}, {northing}, zone={zone}, {hemisphere}) to WGS84"
                ))
            })
    }

    /// Transform a `(lon, lat)` pair between two `CrsKind` instances.
    ///
    /// Currently supports:
    /// - WGS84 ↔ WebMercator
    /// - WGS84 → UTM
    /// - UTM → WGS84
    ///
    /// All other combinations return an error.
    pub fn transform(x: f64, y: f64, from: &CrsKind, to: &CrsKind) -> Result<(f64, f64)> {
        match (from, to) {
            (CrsKind::Wgs84, CrsKind::Wgs84) => Ok((x, y)),
            (CrsKind::WebMercator, CrsKind::WebMercator) => Ok((x, y)),

            (CrsKind::Wgs84, CrsKind::WebMercator) => Self::wgs84_to_web_mercator(x, y),
            (CrsKind::WebMercator, CrsKind::Wgs84) => Self::web_mercator_to_wgs84(x, y),

            (CrsKind::Wgs84, CrsKind::Utm { .. }) => {
                let (e, n, _, _) = Self::wgs84_to_utm(x, y)?;
                Ok((e, n))
            }
            (CrsKind::Utm { zone, hemisphere }, CrsKind::Wgs84) => {
                Self::utm_to_wgs84(x, y, *zone, *hemisphere)
            }

            (
                CrsKind::Utm {
                    zone: z1,
                    hemisphere: h1,
                },
                CrsKind::Utm {
                    zone: z2,
                    hemisphere: h2,
                },
            ) if z1 == z2 && h1 == h2 => Ok((x, y)),

            _ => Err(GeoSparqlError::UnsupportedOperation(format!(
                "CRS transformation from {from:?} to {to:?} is not supported"
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeometryWithCrs — geometry paired with a typed CRS
// ─────────────────────────────────────────────────────────────────────────────

/// A `Geometry` associated with a named `CrsKind`.
///
/// This type combines the `oxirs_geosparql::geometry::Geometry` wrapper
/// (which already embeds a CRS URI string) with the strongly-typed `CrsKind`
/// enum so that callers can work with named CRS variants rather than raw URI
/// strings.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::crs::{GeometryWithCrs, CrsKind};
/// use oxirs_geosparql::geometry::Geometry;
///
/// let point = Geometry::from_wkt("POINT(0 0)").unwrap();
/// let gwc = GeometryWithCrs::new(point, CrsKind::Wgs84);
///
/// assert_eq!(gwc.crs, CrsKind::Wgs84);
/// assert_eq!(gwc.geometry.geometry_type(), "Point");
/// ```
#[derive(Debug, Clone)]
pub struct GeometryWithCrs {
    /// The underlying geometry (coordinates in the CRS defined by `crs`)
    pub geometry: Geometry,
    /// The typed CRS for this geometry
    pub crs: CrsKind,
}

impl GeometryWithCrs {
    /// Create a new `GeometryWithCrs`.
    ///
    /// Also updates the geometry's embedded CRS URI to match `crs`.
    pub fn new(mut geometry: Geometry, crs: CrsKind) -> Self {
        geometry.crs = crs.to_geometry_crs();
        Self { geometry, crs }
    }

    /// Geometry type name (e.g., `"Point"`, `"Polygon"`)
    pub fn geometry_type(&self) -> &str {
        self.geometry.geometry_type()
    }

    /// Returns `true` if the geometry contains no coordinates.
    pub fn is_empty(&self) -> bool {
        self.geometry.is_empty()
    }

    /// Return the EPSG code of this geometry's CRS, if available.
    pub fn epsg_code(&self) -> Option<u32> {
        self.crs.epsg_code()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CrsKind ───────────────────────────────────────────────────────────────

    #[test]
    fn test_crs_kind_epsg_wgs84() {
        assert_eq!(CrsKind::Wgs84.epsg_code(), Some(4326));
    }

    #[test]
    fn test_crs_kind_epsg_web_mercator() {
        assert_eq!(CrsKind::WebMercator.epsg_code(), Some(3857));
    }

    #[test]
    fn test_crs_kind_epsg_utm_north() {
        let utm = CrsKind::Utm {
            zone: 32,
            hemisphere: 'N',
        };
        assert_eq!(utm.epsg_code(), Some(32632));
    }

    #[test]
    fn test_crs_kind_epsg_utm_south() {
        let utm = CrsKind::Utm {
            zone: 18,
            hemisphere: 'S',
        };
        assert_eq!(utm.epsg_code(), Some(32718));
    }

    #[test]
    fn test_crs_kind_custom() {
        let c = CrsKind::Custom { epsg: 27700 };
        assert_eq!(c.epsg_code(), Some(27700));
    }

    #[test]
    fn test_crs_kind_to_uri_wgs84() {
        let uri = CrsKind::Wgs84.to_uri();
        assert!(
            uri.contains("4326"),
            "expected EPSG:4326 in URI, got: {uri}"
        );
    }

    #[test]
    fn test_crs_kind_to_uri_web_mercator() {
        let uri = CrsKind::WebMercator.to_uri();
        assert!(
            uri.contains("3857"),
            "expected EPSG:3857 in URI, got: {uri}"
        );
    }

    #[test]
    fn test_crs_kind_from_epsg_wgs84() {
        assert_eq!(CrsKind::from_epsg(4326), CrsKind::Wgs84);
    }

    #[test]
    fn test_crs_kind_from_epsg_web_mercator() {
        assert_eq!(CrsKind::from_epsg(3857), CrsKind::WebMercator);
        assert_eq!(CrsKind::from_epsg(900913), CrsKind::WebMercator);
    }

    #[test]
    fn test_crs_kind_from_epsg_utm_north() {
        let c = CrsKind::from_epsg(32618);
        assert_eq!(
            c,
            CrsKind::Utm {
                zone: 18,
                hemisphere: 'N'
            }
        );
    }

    #[test]
    fn test_crs_kind_from_epsg_utm_south() {
        let c = CrsKind::from_epsg(32718);
        assert_eq!(
            c,
            CrsKind::Utm {
                zone: 18,
                hemisphere: 'S'
            }
        );
    }

    #[test]
    fn test_crs_kind_from_epsg_custom() {
        let c = CrsKind::from_epsg(27700);
        assert_eq!(c, CrsKind::Custom { epsg: 27700 });
    }

    // ── CrsTransformer: WGS84 ↔ WebMercator ──────────────────────────────────

    #[test]
    fn test_wgs84_to_web_mercator_origin() {
        let (x, y) = CrsTransformer::wgs84_to_web_mercator(0.0, 0.0).unwrap();
        assert!(x.abs() < 1e-6, "expected x≈0, got {x}");
        assert!(y.abs() < 1e-6, "expected y≈0, got {y}");
    }

    #[test]
    fn test_wgs84_to_web_mercator_greenwich_poles_rejected() {
        // Exact ±90° lat is outside valid range
        assert!(CrsTransformer::wgs84_to_web_mercator(0.0, 90.0).is_err());
        assert!(CrsTransformer::wgs84_to_web_mercator(0.0, -90.0).is_err());
    }

    #[test]
    fn test_web_mercator_to_wgs84_origin() {
        let (lon, lat) = CrsTransformer::web_mercator_to_wgs84(0.0, 0.0).unwrap();
        assert!(lon.abs() < 1e-10, "expected lon≈0, got {lon}");
        assert!(lat.abs() < 1e-10, "expected lat≈0, got {lat}");
    }

    #[test]
    fn test_wgs84_web_mercator_roundtrip() {
        let test_cases = [
            (0.0_f64, 0.0_f64),
            (51.5074, -0.1276), // London (lat, lon reversed for wkt but lon/lat for mercator)
            (-33.8688, 151.2093), // Sydney
            (40.7128, -74.006), // New York
        ];

        for (lat, lon) in test_cases {
            let (x, y) = CrsTransformer::wgs84_to_web_mercator(lon, lat).unwrap();
            let (lon2, lat2) = CrsTransformer::web_mercator_to_wgs84(x, y).unwrap();
            assert!(
                (lon2 - lon).abs() < 1e-8,
                "lon roundtrip failed for ({lon}, {lat}): got {lon2}"
            );
            assert!(
                (lat2 - lat).abs() < 1e-8,
                "lat roundtrip failed for ({lon}, {lat}): got {lat2}"
            );
        }
    }

    #[test]
    fn test_web_mercator_x_scale() {
        // At lon=1°, x = a * π/180 ≈ 111_319.49 m
        let (x, _) = CrsTransformer::wgs84_to_web_mercator(1.0, 0.0).unwrap();
        let expected = A_WGS84 * (1.0_f64).to_radians();
        assert!((x - expected).abs() < 1e-3, "expected {expected}, got {x}");
    }

    #[test]
    fn test_wgs84_to_web_mercator_london() {
        let (x, y) = CrsTransformer::wgs84_to_web_mercator(-0.1276, 51.5074).unwrap();
        // London Web Mercator: roughly (-14_211, 6_711_790)
        assert!((x - (-14_211.0)).abs() < 500.0, "x={x}");
        assert!((y - 6_711_790.0).abs() < 500.0, "y={y}");
    }

    // ── CrsTransformer: WGS84 → UTM ───────────────────────────────────────────

    #[test]
    fn test_wgs84_to_utm_new_york() {
        let (e, n, zone, hemi) = CrsTransformer::wgs84_to_utm(-74.006, 40.7128).unwrap();
        assert_eq!(zone, 18, "NYC should be zone 18");
        assert_eq!(hemi, 'N', "NYC is northern hemisphere");
        assert!((e - 583_960.0).abs() < 2_000.0, "easting={e}");
        assert!((n - 4_507_523.0).abs() < 2_000.0, "northing={n}");
    }

    #[test]
    fn test_wgs84_to_utm_london() {
        let (_, _, zone, hemi) = CrsTransformer::wgs84_to_utm(-0.1276, 51.5074).unwrap();
        assert_eq!(zone, 30, "London should be zone 30");
        assert_eq!(hemi, 'N');
    }

    #[test]
    fn test_wgs84_to_utm_sydney() {
        let (_, _, zone, hemi) = CrsTransformer::wgs84_to_utm(151.2093, -33.8688).unwrap();
        assert_eq!(zone, 56, "Sydney should be zone 56");
        assert_eq!(hemi, 'S', "Sydney is southern hemisphere");
    }

    #[test]
    fn test_wgs84_utm_roundtrip() {
        let test_cases = [
            (0.0_f64, 0.0_f64),
            (-74.006, 40.7128),
            (151.2093, -33.8688),
            (13.405, 52.52), // Berlin
        ];

        for (lon, lat) in test_cases {
            let (e, n, zone, hemi) = CrsTransformer::wgs84_to_utm(lon, lat).unwrap();
            let (lon2, lat2) = CrsTransformer::utm_to_wgs84(e, n, zone, hemi).unwrap();
            assert!(
                (lon2 - lon).abs() < 0.001,
                "lon roundtrip ({lon}, {lat}): got {lon2}"
            );
            assert!(
                (lat2 - lat).abs() < 0.001,
                "lat roundtrip ({lon}, {lat}): got {lat2}"
            );
        }
    }

    // ── CrsTransformer::transform ─────────────────────────────────────────────

    #[test]
    fn test_transform_identity_wgs84() {
        let (x, y) =
            CrsTransformer::transform(10.0, 20.0, &CrsKind::Wgs84, &CrsKind::Wgs84).unwrap();
        assert_eq!((x, y), (10.0, 20.0));
    }

    #[test]
    fn test_transform_wgs84_to_web_mercator() {
        let (x, y) =
            CrsTransformer::transform(0.0, 0.0, &CrsKind::Wgs84, &CrsKind::WebMercator).unwrap();
        assert!(x.abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    #[test]
    fn test_transform_web_mercator_to_wgs84() {
        let (lon, lat) =
            CrsTransformer::transform(0.0, 0.0, &CrsKind::WebMercator, &CrsKind::Wgs84).unwrap();
        assert!(lon.abs() < 1e-10);
        assert!(lat.abs() < 1e-10);
    }

    #[test]
    fn test_transform_wgs84_to_utm() {
        let utm18n = CrsKind::Utm {
            zone: 18,
            hemisphere: 'N',
        };
        let (e, n) = CrsTransformer::transform(-74.006, 40.7128, &CrsKind::Wgs84, &utm18n).unwrap();
        assert!((e - 583_960.0).abs() < 2_000.0, "easting={e}");
        assert!((n - 4_507_523.0).abs() < 2_000.0, "northing={n}");
    }

    #[test]
    fn test_transform_utm_to_wgs84() {
        let utm18n = CrsKind::Utm {
            zone: 18,
            hemisphere: 'N',
        };
        let (lon, lat) =
            CrsTransformer::transform(583_960.0, 4_507_523.0, &utm18n, &CrsKind::Wgs84).unwrap();
        assert!((lon - (-74.006)).abs() < 0.01, "lon={lon}");
        assert!((lat - 40.7128).abs() < 0.01, "lat={lat}");
    }

    #[test]
    fn test_transform_unsupported_fails() {
        let c1 = CrsKind::Custom { epsg: 27700 };
        let c2 = CrsKind::Custom { epsg: 4269 };
        assert!(CrsTransformer::transform(0.0, 0.0, &c1, &c2).is_err());
    }

    // ── GeometryWithCrs ───────────────────────────────────────────────────────

    #[test]
    fn test_geometry_with_crs_new() {
        let point = Geometry::from_wkt("POINT(10 20)").unwrap();
        let gwc = GeometryWithCrs::new(point, CrsKind::Wgs84);
        assert_eq!(gwc.crs, CrsKind::Wgs84);
        assert_eq!(gwc.geometry_type(), "Point");
    }

    #[test]
    fn test_geometry_with_crs_epsg_code() {
        let geom = Geometry::from_wkt("POINT(0 0)").unwrap();
        let gwc = GeometryWithCrs::new(geom, CrsKind::WebMercator);
        assert_eq!(gwc.epsg_code(), Some(3857));
    }

    #[test]
    fn test_geometry_with_crs_updates_embedded_crs() {
        let geom = Geometry::from_wkt("POINT(0 0)").unwrap();
        let gwc = GeometryWithCrs::new(geom, CrsKind::WebMercator);
        // The embedded CRS URI should contain the EPSG:3857 code
        assert!(
            gwc.geometry.crs.uri.contains("3857"),
            "CRS URI should contain 3857, got: {}",
            gwc.geometry.crs.uri
        );
    }

    #[test]
    fn test_geometry_with_crs_is_empty() {
        let geom = Geometry::from_wkt("POINT(5 5)").unwrap();
        let gwc = GeometryWithCrs::new(geom, CrsKind::Wgs84);
        assert!(!gwc.is_empty());
    }

    #[test]
    fn test_geometry_with_crs_polygon() {
        let poly = Geometry::from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))").unwrap();
        let gwc = GeometryWithCrs::new(
            poly,
            CrsKind::Utm {
                zone: 32,
                hemisphere: 'N',
            },
        );
        assert_eq!(gwc.geometry_type(), "Polygon");
        assert_eq!(
            gwc.crs,
            CrsKind::Utm {
                zone: 32,
                hemisphere: 'N'
            }
        );
        assert_eq!(gwc.epsg_code(), Some(32632));
    }
}
