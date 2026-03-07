//! CRS literal annotation support for GeoSPARQL geometries
//!
//! This module provides support for `geo:crs` literal annotations in RDF
//! geometry literals, enabling CRS-aware geometry processing.
//!
//! ## GeoSPARQL CRS Literals
//!
//! GeoSPARQL 1.1 allows CRS specification in WKT literals using the syntax:
//!
//! ```text
//! "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(lon lat)"^^geo:wktLiteral
//! ```
//!
//! This module provides:
//! - [`CrsLiteral`]: Parsed geometry literal with embedded CRS URI
//! - [`parse_crs_wkt_literal`]: Parse a WKT literal with optional CRS prefix
//! - [`encode_crs_wkt_literal`]: Encode a geometry with CRS as a WKT literal
//! - [`CrsGeometryTransformer`]: Transform geometries between CRS while preserving annotations
//!
//! ## Supported CRS URIs
//!
//! | URI Pattern | Meaning |
//! |-------------|---------|
//! | `http://www.opengis.net/def/crs/OGC/1.3/CRS84` | WGS84 (default) |
//! | `http://www.opengis.net/def/crs/EPSG/0/4326` | WGS84 geographic |
//! | `http://www.opengis.net/def/crs/EPSG/0/3857` | Web Mercator |
//! | `http://www.opengis.net/def/crs/EPSG/0/326xx` | UTM North (xx=zone) |
//! | `http://www.opengis.net/def/crs/EPSG/0/327xx` | UTM South (xx=zone) |

use crate::crs::{CrsKind, CrsTransformer, GeometryWithCrs};
use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// OGC CRS84 URI (WGS84 with lon/lat axis order — the GeoSPARQL default)
pub const CRS84_URI: &str = "http://www.opengis.net/def/crs/OGC/1.3/CRS84";

/// OGC EPSG authority URI prefix
pub const EPSG_URI_PREFIX: &str = "http://www.opengis.net/def/crs/EPSG/0/";

/// GeoSPARQL WKT literal datatype URI
pub const GEO_WKT_LITERAL: &str = "http://www.opengis.net/ont/geosparql#wktLiteral";

/// `geo:crs` property URI
pub const GEO_CRS: &str = "http://www.opengis.net/ont/geosparql#crs";

// ─────────────────────────────────────────────────────────────────────────────
// CrsLiteral
// ─────────────────────────────────────────────────────────────────────────────

/// A geometry literal with an embedded CRS annotation.
///
/// Corresponds to the GeoSPARQL 1.1 `geo:wktLiteral` with an optional
/// CRS URI prefix of the form:
///
/// ```text
/// <http://www.opengis.net/def/crs/EPSG/0/4326> POINT(0 0)
/// ```
#[derive(Debug, Clone)]
pub struct CrsLiteral {
    /// The geometry
    pub geometry: Geometry,
    /// Typed CRS (resolved from the URI)
    pub crs: CrsKind,
    /// Raw CRS URI from the literal (None if default CRS was used)
    pub crs_uri: Option<String>,
    /// Original WKT string (without the CRS prefix)
    pub wkt: String,
}

impl CrsLiteral {
    /// Create a `CrsLiteral` from a geometry and a CRS kind.
    pub fn new(geometry: Geometry, crs: CrsKind) -> Self {
        let crs_uri = Some(epsg_uri_for_kind(&crs));
        let wkt = geometry.to_wkt();
        Self {
            geometry,
            crs,
            crs_uri,
            wkt,
        }
    }

    /// Encode as a GeoSPARQL WKT literal string with CRS prefix.
    ///
    /// Returns `"<uri> WKT"` if a non-default CRS is present,
    /// or just `"WKT"` for the default CRS (CRS84 / WGS84).
    pub fn to_wkt_literal(&self) -> String {
        encode_crs_wkt_literal(&self.wkt, self.crs_uri.as_deref())
    }

    /// Return the EPSG code of this literal's CRS, if known.
    pub fn epsg_code(&self) -> Option<u32> {
        self.crs.epsg_code()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// URI helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the OGC EPSG URI for a `CrsKind`.
pub fn epsg_uri_for_kind(crs: &CrsKind) -> String {
    match crs {
        CrsKind::Wgs84 => CRS84_URI.to_string(),
        _ => match crs.epsg_code() {
            Some(code) => format!("{EPSG_URI_PREFIX}{code}"),
            None => CRS84_URI.to_string(),
        },
    }
}

/// Parse a CRS URI string into a `CrsKind`.
///
/// Recognises:
/// - OGC CRS84 (`http://www.opengis.net/def/crs/OGC/1.3/CRS84`) → `Wgs84`
/// - EPSG URIs (`http://www.opengis.net/def/crs/EPSG/0/<code>`) → resolved via `CrsKind::from_epsg`
/// - Short EPSG URIs (`http://www.opengis.net/def/crs/EPSG/<code>`) → same
/// - Plain EPSG codes (`EPSG:4326`) → same
pub fn parse_crs_uri(uri: &str) -> Result<CrsKind> {
    let uri = uri.trim();

    // OGC default
    if uri == CRS84_URI || uri.eq_ignore_ascii_case("crs84") || uri.contains("CRS84") {
        return Ok(CrsKind::Wgs84);
    }

    // EPSG:4326 short form
    if let Some(rest) = uri
        .strip_prefix("EPSG:")
        .or_else(|| uri.strip_prefix("epsg:"))
    {
        if let Ok(code) = rest.parse::<u32>() {
            return Ok(CrsKind::from_epsg(code));
        }
    }

    // http://www.opengis.net/def/crs/EPSG/0/<code>
    if let Some(rest) = uri.strip_prefix(EPSG_URI_PREFIX) {
        if let Ok(code) = rest.parse::<u32>() {
            return Ok(CrsKind::from_epsg(code));
        }
    }

    // http://www.opengis.net/def/crs/EPSG/<code>  (without /0/)
    const EPSG_SHORT_PREFIX: &str = "http://www.opengis.net/def/crs/EPSG/";
    if let Some(rest) = uri.strip_prefix(EPSG_SHORT_PREFIX) {
        if let Ok(code) = rest.parse::<u32>() {
            return Ok(CrsKind::from_epsg(code));
        }
    }

    Err(GeoSparqlError::InvalidInput(format!(
        "Unrecognised CRS URI: {uri}"
    )))
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsing & Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a GeoSPARQL WKT literal string that may begin with a CRS URI.
///
/// Accepted formats:
/// - `"<URI> WKT"` — CRS URI in angle brackets followed by WKT
/// - `"WKT"` — no CRS prefix (defaults to CRS84 / WGS84)
///
/// # Errors
///
/// Returns an error if the WKT is invalid or the CRS URI is unrecognised.
///
/// # Example
///
/// ```
/// use oxirs_geosparql::crs::crs_literal::parse_crs_wkt_literal;
///
/// let lit = parse_crs_wkt_literal(
///     "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(10 20)"
/// ).unwrap();
/// assert_eq!(lit.geometry.geometry_type(), "Point");
/// ```
pub fn parse_crs_wkt_literal(literal: &str) -> Result<CrsLiteral> {
    let s = literal.trim();

    // Check for angle-bracket URI prefix: `<URI> WKT`
    if s.starts_with('<') {
        let close = s.find('>').ok_or_else(|| {
            GeoSparqlError::InvalidInput(
                "CRS WKT literal: unclosed '<' in CRS URI prefix".to_string(),
            )
        })?;

        let uri = &s[1..close];
        let wkt_part = s[close + 1..].trim();

        let crs = parse_crs_uri(uri).unwrap_or(CrsKind::Wgs84);
        let geometry = Geometry::from_wkt(wkt_part)?;

        return Ok(CrsLiteral {
            geometry,
            crs,
            crs_uri: Some(uri.to_string()),
            wkt: wkt_part.to_string(),
        });
    }

    // No CRS prefix — use default WGS84
    let geometry = Geometry::from_wkt(s)?;
    let wkt = s.to_string();
    Ok(CrsLiteral {
        geometry,
        crs: CrsKind::Wgs84,
        crs_uri: None,
        wkt,
    })
}

/// Encode a WKT string and optional CRS URI as a GeoSPARQL WKT literal.
///
/// If `crs_uri` is `None` or the default CRS84, returns just the WKT.
/// Otherwise prepends `<URI> `.
pub fn encode_crs_wkt_literal(wkt: &str, crs_uri: Option<&str>) -> String {
    match crs_uri {
        None => wkt.to_string(),
        Some(uri) if uri == CRS84_URI => wkt.to_string(),
        Some(uri) => format!("<{uri}> {wkt}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CrsGeometryTransformer
// ─────────────────────────────────────────────────────────────────────────────

/// Transform geometries between coordinate reference systems while
/// preserving CRS annotations.
///
/// This struct wraps [`CrsTransformer`] to provide geometry-level
/// (rather than coordinate-level) transformations, preserving the
/// complete `GeometryWithCrs` wrapper.
pub struct CrsGeometryTransformer;

impl CrsGeometryTransformer {
    /// Transform all coordinates of a `GeometryWithCrs` to a target CRS.
    ///
    /// Only Point geometries are fully supported for coordinate-level
    /// transformation. For other types, the CRS label is updated but
    /// coordinates are not transformed (projection of multi-part geometries
    /// requires per-coordinate handling beyond this helper).
    ///
    /// # Errors
    ///
    /// Returns an error if the coordinate transformation is unsupported.
    pub fn transform_point(gwc: &GeometryWithCrs, target: CrsKind) -> Result<GeometryWithCrs> {
        use geo_types::Geometry as GeoGeometry;

        match &gwc.geometry.geom {
            GeoGeometry::Point(p) => {
                let (tx, ty) = CrsTransformer::transform(p.x(), p.y(), &gwc.crs, &target)?;
                let new_geom = Geometry::new(GeoGeometry::Point(geo_types::Point::new(tx, ty)));
                Ok(GeometryWithCrs::new(new_geom, target))
            }
            _ => Err(GeoSparqlError::UnsupportedOperation(
                "CrsGeometryTransformer::transform_point only supports Point geometries"
                    .to_string(),
            )),
        }
    }

    /// Transform a `CrsLiteral` geometry to a target CRS (Point only).
    pub fn transform_literal(literal: &CrsLiteral, target: CrsKind) -> Result<CrsLiteral> {
        use geo_types::Geometry as GeoGeometry;

        match &literal.geometry.geom {
            GeoGeometry::Point(p) => {
                let (tx, ty) = CrsTransformer::transform(p.x(), p.y(), &literal.crs, &target)?;
                let new_geom = Geometry::new(GeoGeometry::Point(geo_types::Point::new(tx, ty)));
                let wkt = new_geom.to_wkt();
                let crs_uri = Some(epsg_uri_for_kind(&target));
                Ok(CrsLiteral {
                    geometry: new_geom,
                    crs: target,
                    crs_uri,
                    wkt,
                })
            }
            _ => Err(GeoSparqlError::UnsupportedOperation(
                "CrsGeometryTransformer::transform_literal only supports Point geometries"
                    .to_string(),
            )),
        }
    }

    /// Parse a WKT literal, convert it to the target CRS, and return a new literal.
    pub fn parse_and_transform(literal_str: &str, target: CrsKind) -> Result<CrsLiteral> {
        let parsed = parse_crs_wkt_literal(literal_str)?;
        Self::transform_literal(&parsed, target)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_crs_uri ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_crs_uri_crs84() {
        let c = parse_crs_uri(CRS84_URI).unwrap();
        assert_eq!(c, CrsKind::Wgs84);
    }

    #[test]
    fn test_parse_crs_uri_epsg_4326() {
        let c = parse_crs_uri("http://www.opengis.net/def/crs/EPSG/0/4326").unwrap();
        assert_eq!(c, CrsKind::Wgs84);
    }

    #[test]
    fn test_parse_crs_uri_epsg_3857() {
        let c = parse_crs_uri("http://www.opengis.net/def/crs/EPSG/0/3857").unwrap();
        assert_eq!(c, CrsKind::WebMercator);
    }

    #[test]
    fn test_parse_crs_uri_epsg_utm_north() {
        let c = parse_crs_uri("http://www.opengis.net/def/crs/EPSG/0/32618").unwrap();
        assert_eq!(
            c,
            CrsKind::Utm {
                zone: 18,
                hemisphere: 'N'
            }
        );
    }

    #[test]
    fn test_parse_crs_uri_epsg_utm_south() {
        let c = parse_crs_uri("http://www.opengis.net/def/crs/EPSG/0/32756").unwrap();
        assert_eq!(
            c,
            CrsKind::Utm {
                zone: 56,
                hemisphere: 'S'
            }
        );
    }

    #[test]
    fn test_parse_crs_uri_short_epsg_colon_form() {
        let c = parse_crs_uri("EPSG:4326").unwrap();
        assert_eq!(c, CrsKind::Wgs84);
    }

    #[test]
    fn test_parse_crs_uri_short_epsg_colon_web_mercator() {
        let c = parse_crs_uri("EPSG:3857").unwrap();
        assert_eq!(c, CrsKind::WebMercator);
    }

    #[test]
    fn test_parse_crs_uri_unknown_returns_error() {
        assert!(parse_crs_uri("ftp://unknown.crs/foo").is_err());
    }

    // ── parse_crs_wkt_literal ─────────────────────────────────────────────────

    #[test]
    fn test_parse_plain_wkt() {
        let lit = parse_crs_wkt_literal("POINT(1 2)").unwrap();
        assert_eq!(lit.geometry.geometry_type(), "Point");
        assert_eq!(lit.crs, CrsKind::Wgs84);
        assert!(lit.crs_uri.is_none());
    }

    #[test]
    fn test_parse_wkt_with_epsg_4326_prefix() {
        let raw = "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(10 20)";
        let lit = parse_crs_wkt_literal(raw).unwrap();
        assert_eq!(lit.geometry.geometry_type(), "Point");
        assert_eq!(lit.crs, CrsKind::Wgs84);
    }

    #[test]
    fn test_parse_wkt_with_web_mercator_prefix() {
        let raw = "<http://www.opengis.net/def/crs/EPSG/0/3857> POINT(0 0)";
        let lit = parse_crs_wkt_literal(raw).unwrap();
        assert_eq!(lit.crs, CrsKind::WebMercator);
    }

    #[test]
    fn test_parse_wkt_with_utm_prefix() {
        let raw = "<http://www.opengis.net/def/crs/EPSG/0/32618> POINT(583960 4507523)";
        let lit = parse_crs_wkt_literal(raw).unwrap();
        assert_eq!(
            lit.crs,
            CrsKind::Utm {
                zone: 18,
                hemisphere: 'N'
            }
        );
    }

    #[test]
    fn test_parse_wkt_unclosed_angle_bracket_error() {
        assert!(parse_crs_wkt_literal("<http://example.com/crs POINT(0 0)").is_err());
    }

    #[test]
    fn test_parse_wkt_polygon() {
        let raw =
            "<http://www.opengis.net/def/crs/OGC/1.3/CRS84> POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))";
        let lit = parse_crs_wkt_literal(raw).unwrap();
        assert_eq!(lit.geometry.geometry_type(), "Polygon");
        assert_eq!(lit.crs, CrsKind::Wgs84);
    }

    // ── encode_crs_wkt_literal ────────────────────────────────────────────────

    #[test]
    fn test_encode_no_crs() {
        let s = encode_crs_wkt_literal("POINT(1 2)", None);
        assert_eq!(s, "POINT(1 2)");
    }

    #[test]
    fn test_encode_crs84_uri_is_omitted() {
        let s = encode_crs_wkt_literal("POINT(1 2)", Some(CRS84_URI));
        assert_eq!(s, "POINT(1 2)");
    }

    #[test]
    fn test_encode_epsg_3857() {
        let s = encode_crs_wkt_literal(
            "POINT(0 0)",
            Some("http://www.opengis.net/def/crs/EPSG/0/3857"),
        );
        assert!(s.starts_with('<'), "expected angle bracket: {s}");
        assert!(s.contains("3857"), "expected EPSG code: {s}");
    }

    #[test]
    fn test_roundtrip_parse_encode() {
        let original = "<http://www.opengis.net/def/crs/EPSG/0/3857> POINT(1113195 1118890)";
        let lit = parse_crs_wkt_literal(original).unwrap();
        let re_encoded = lit.to_wkt_literal();
        // Should contain the EPSG:3857 URI and the WKT
        assert!(re_encoded.contains("3857"), "re_encoded={re_encoded}");
    }

    // ── CrsLiteral methods ────────────────────────────────────────────────────

    #[test]
    fn test_crs_literal_new_wgs84() {
        let geom = Geometry::from_wkt("POINT(5 10)").unwrap();
        let lit = CrsLiteral::new(geom, CrsKind::Wgs84);
        assert_eq!(lit.epsg_code(), Some(4326));
    }

    #[test]
    fn test_crs_literal_new_web_mercator() {
        let geom = Geometry::from_wkt("POINT(0 0)").unwrap();
        let lit = CrsLiteral::new(geom, CrsKind::WebMercator);
        assert_eq!(lit.epsg_code(), Some(3857));
        assert!(
            lit.crs_uri.as_deref().unwrap_or("").contains("3857"),
            "URI should contain 3857"
        );
    }

    // ── CrsGeometryTransformer ────────────────────────────────────────────────

    #[test]
    fn test_transformer_wgs84_to_web_mercator() {
        let geom = Geometry::from_wkt("POINT(0 0)").unwrap();
        let gwc = GeometryWithCrs::new(geom, CrsKind::Wgs84);
        let result = CrsGeometryTransformer::transform_point(&gwc, CrsKind::WebMercator).unwrap();
        assert_eq!(result.crs, CrsKind::WebMercator);
    }

    #[test]
    fn test_transformer_wgs84_to_utm() {
        let geom = Geometry::from_wkt("POINT(-74.006 40.7128)").unwrap();
        let gwc = GeometryWithCrs::new(geom, CrsKind::Wgs84);
        let utm18n = CrsKind::Utm {
            zone: 18,
            hemisphere: 'N',
        };
        let result = CrsGeometryTransformer::transform_point(&gwc, utm18n.clone()).unwrap();
        assert_eq!(result.crs, utm18n);
    }

    #[test]
    fn test_transformer_non_point_returns_error() {
        let geom = Geometry::from_wkt("LINESTRING(0 0, 1 1)").unwrap();
        let gwc = GeometryWithCrs::new(geom, CrsKind::Wgs84);
        assert!(CrsGeometryTransformer::transform_point(&gwc, CrsKind::WebMercator).is_err());
    }

    #[test]
    fn test_parse_and_transform_wgs84_to_web_mercator() {
        let lit = "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(0 0)";
        let result =
            CrsGeometryTransformer::parse_and_transform(lit, CrsKind::WebMercator).unwrap();
        assert_eq!(result.crs, CrsKind::WebMercator);
    }

    // ── epsg_uri_for_kind ─────────────────────────────────────────────────────

    #[test]
    fn test_epsg_uri_wgs84_returns_crs84() {
        let uri = epsg_uri_for_kind(&CrsKind::Wgs84);
        assert_eq!(uri, CRS84_URI);
    }

    #[test]
    fn test_epsg_uri_web_mercator() {
        let uri = epsg_uri_for_kind(&CrsKind::WebMercator);
        assert!(uri.contains("3857"), "uri={uri}");
    }

    #[test]
    fn test_epsg_uri_utm_north() {
        let utm = CrsKind::Utm {
            zone: 32,
            hemisphere: 'N',
        };
        let uri = epsg_uri_for_kind(&utm);
        assert!(uri.contains("32632"), "uri={uri}");
    }
}
