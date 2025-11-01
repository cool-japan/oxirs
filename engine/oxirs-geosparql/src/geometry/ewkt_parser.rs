//! PostGIS EWKT (Extended Well-Known Text) Parser
//!
//! Implements parsing and serialization of PostGIS Extended Well-Known Text format.
//! EWKT extends WKT with SRID information in the format: SRID=4326;POINT(1 2)

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};
use regex::Regex;

/// Parse EWKT string into a Geometry
///
/// EWKT format: `SRID=4326;POINT(1 2)`
///
/// # Arguments
/// * `ewkt` - EWKT string
///
/// # Returns
/// Parsed geometry with CRS information
///
/// # Example
/// ```
/// use oxirs_geosparql::geometry::ewkt_parser::parse_ewkt;
///
/// let geom = parse_ewkt("SRID=4326;POINT(1 2)").unwrap();
/// assert_eq!(geom.crs.epsg_code(), Some(4326));
/// ```
pub fn parse_ewkt(ewkt: &str) -> Result<Geometry> {
    // Extract SRID if present (format: SRID=4326;GEOMETRY)
    let re = Regex::new(r"^SRID=(\d+);(.+)$").unwrap();

    if let Some(caps) = re.captures(ewkt.trim()) {
        let srid_str = caps.get(1).unwrap().as_str();
        let wkt_geom = caps.get(2).unwrap().as_str();

        let srid: u32 = srid_str
            .parse()
            .map_err(|_| GeoSparqlError::ParseError(format!("Invalid SRID: {}", srid_str)))?;

        // Parse the WKT part
        let mut geometry = crate::geometry::wkt_parser::parse_wkt(wkt_geom)?;

        // Set the CRS from SRID
        geometry.crs = Crs::epsg(srid);

        Ok(geometry)
    } else {
        // No SRID, parse as regular WKT
        crate::geometry::wkt_parser::parse_wkt(ewkt)
    }
}

/// Convert geometry to EWKT format
///
/// # Arguments
/// * `geometry` - Geometry to convert
///
/// # Returns
/// EWKT string representation
///
/// # Example
/// ```
/// use oxirs_geosparql::geometry::{Geometry, Crs, ewkt_parser::geometry_to_ewkt};
/// use geo_types::{Geometry as GeoGeometry, Point};
///
/// let geom = Geometry::with_crs(
///     GeoGeometry::Point(Point::new(1.0, 2.0)),
///     Crs::epsg(4326)
/// );
/// let ewkt = geometry_to_ewkt(&geom);
/// assert_eq!(ewkt, "SRID=4326;POINT(1 2)");
/// ```
pub fn geometry_to_ewkt(geometry: &Geometry) -> String {
    let wkt = crate::geometry::wkt_parser::geometry_to_wkt(&geometry.geom);

    // Add SRID prefix if CRS is not default
    if let Some(srid) = geometry.crs.epsg_code() {
        format!("SRID={};{}", srid, wkt)
    } else {
        wkt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point};

    #[test]
    fn test_parse_ewkt_point_with_srid() {
        let ewkt = "SRID=4326;POINT(1 2)";
        let geom = parse_ewkt(ewkt).unwrap();

        assert_eq!(geom.crs.epsg_code(), Some(4326));
        match geom.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_parse_ewkt_point_without_srid() {
        let ewkt = "POINT(1 2)";
        let geom = parse_ewkt(ewkt).unwrap();

        assert!(geom.crs.is_default());
        match geom.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_parse_ewkt_linestring() {
        let ewkt = "SRID=3857;LINESTRING(0 0, 1 1, 2 0)";
        let geom = parse_ewkt(ewkt).unwrap();

        assert_eq!(geom.crs.epsg_code(), Some(3857));
        match geom.geom {
            GeoGeometry::LineString(ls) => {
                assert_eq!(ls.0.len(), 3);
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_parse_ewkt_polygon() {
        let ewkt = "SRID=4326;POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))";
        let geom = parse_ewkt(ewkt).unwrap();

        assert_eq!(geom.crs.epsg_code(), Some(4326));
        match geom.geom {
            GeoGeometry::Polygon(poly) => {
                assert_eq!(poly.exterior().0.len(), 5);
            }
            _ => panic!("Expected Polygon geometry"),
        }
    }

    #[test]
    fn test_geometry_to_ewkt_point() {
        let geom = Geometry::with_crs(GeoGeometry::Point(Point::new(1.0, 2.0)), Crs::epsg(4326));

        let ewkt = geometry_to_ewkt(&geom);
        assert_eq!(ewkt, "SRID=4326;POINT(1 2)");
    }

    #[test]
    fn test_geometry_to_ewkt_without_srid() {
        let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));

        let ewkt = geometry_to_ewkt(&geom);
        assert_eq!(ewkt, "POINT(1 2)");
    }

    #[test]
    fn test_geometry_to_ewkt_linestring() {
        let ls = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 2.0, y: 0.0 },
        ]);
        let geom = Geometry::with_crs(GeoGeometry::LineString(ls), Crs::epsg(3857));

        let ewkt = geometry_to_ewkt(&geom);
        assert_eq!(ewkt, "SRID=3857;LINESTRING(0 0, 1 1, 2 0)");
    }

    #[test]
    fn test_ewkt_round_trip() {
        let original = "SRID=4326;POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))";
        let geom = parse_ewkt(original).unwrap();
        let ewkt = geometry_to_ewkt(&geom);

        // Parse again to verify
        let geom2 = parse_ewkt(&ewkt).unwrap();
        assert_eq!(geom.crs.epsg_code(), geom2.crs.epsg_code());
    }

    #[test]
    fn test_parse_invalid_srid() {
        let ewkt = "SRID=abc;POINT(1 2)";
        let result = parse_ewkt(ewkt);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ewkt_multipoint() {
        let ewkt = "SRID=4326;MULTIPOINT((1 2), (3 4))";
        let geom = parse_ewkt(ewkt).unwrap();

        assert_eq!(geom.crs.epsg_code(), Some(4326));
        match geom.geom {
            GeoGeometry::MultiPoint(mp) => {
                assert_eq!(mp.0.len(), 2);
            }
            _ => panic!("Expected MultiPoint geometry"),
        }
    }

    #[test]
    fn test_parse_ewkt_multipolygon() {
        let ewkt =
            "SRID=4326;MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))";
        let geom = parse_ewkt(ewkt).unwrap();

        assert_eq!(geom.crs.epsg_code(), Some(4326));
        match geom.geom {
            GeoGeometry::MultiPolygon(mpoly) => {
                assert_eq!(mpoly.0.len(), 2);
            }
            _ => panic!("Expected MultiPolygon geometry"),
        }
    }
}
