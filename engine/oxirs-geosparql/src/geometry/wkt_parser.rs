//! WKT (Well-Known Text) parser and serializer
//!
//! Converts between WKT strings and geometry objects.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};
use geo_types::Geometry as GeoGeometry;
use regex::Regex;
use std::str::FromStr;

/// Parse a WKT string into a Geometry
pub fn parse_wkt(wkt_str: &str) -> Result<Geometry> {
    // Extract CRS if present (e.g., "<http://...> POINT(1 2)")
    let (crs, wkt_geom) = extract_crs(wkt_str)?;

    // Use the wkt crate for parsing
    let wkt_parsed: wkt::Wkt<f64> =
        wkt::Wkt::from_str(wkt_geom).map_err(|e| GeoSparqlError::InvalidWkt(e.to_string()))?;

    // Convert to geo_types geometry using try_into
    let geo_geom: GeoGeometry<f64> = wkt_parsed
        .try_into()
        .map_err(|_| GeoSparqlError::InvalidWkt("Failed to convert WKT to geometry".to_string()))?;

    Ok(Geometry::with_crs(geo_geom, crs))
}

/// Extract CRS from WKT string if present
fn extract_crs(wkt: &str) -> Result<(Crs, &str)> {
    let re = Regex::new(r"^<([^>]+)>\s+(.+)$").unwrap();

    if let Some(caps) = re.captures(wkt.trim()) {
        let crs_uri = caps.get(1).unwrap().as_str();
        let wkt_geom = caps.get(2).unwrap().as_str();
        Ok((Crs::new(crs_uri), wkt_geom))
    } else {
        Ok((Crs::default(), wkt))
    }
}

// Removed - now using try_into() directly from wkt::Wkt to geo_types::Geometry

/// Convert a geo_types Geometry to WKT string
pub fn geometry_to_wkt(geom: &GeoGeometry<f64>) -> String {
    match geom {
        GeoGeometry::Point(p) => {
            if p.x().is_nan() || p.y().is_nan() {
                "POINT EMPTY".to_string()
            } else {
                format!("POINT({} {})", p.x(), p.y())
            }
        }
        GeoGeometry::Line(l) => {
            format!(
                "LINESTRING({} {}, {} {})",
                l.start.x, l.start.y, l.end.x, l.end.y
            )
        }
        GeoGeometry::LineString(ls) => {
            if ls.0.is_empty() {
                "LINESTRING EMPTY".to_string()
            } else {
                let coords: Vec<String> = ls.0.iter().map(|c| format!("{} {}", c.x, c.y)).collect();
                format!("LINESTRING({})", coords.join(", "))
            }
        }
        GeoGeometry::Polygon(poly) => {
            if poly.exterior().0.is_empty() {
                "POLYGON EMPTY".to_string()
            } else {
                let exterior: Vec<String> = poly
                    .exterior()
                    .0
                    .iter()
                    .map(|c| format!("{} {}", c.x, c.y))
                    .collect();

                let mut rings = vec![format!("({})", exterior.join(", "))];

                for interior in poly.interiors() {
                    let interior_coords: Vec<String> = interior
                        .0
                        .iter()
                        .map(|c| format!("{} {}", c.x, c.y))
                        .collect();
                    rings.push(format!("({})", interior_coords.join(", ")));
                }

                format!("POLYGON({})", rings.join(", "))
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            if mp.0.is_empty() {
                "MULTIPOINT EMPTY".to_string()
            } else {
                let points: Vec<String> =
                    mp.0.iter()
                        .map(|p| format!("({} {})", p.x(), p.y()))
                        .collect();
                format!("MULTIPOINT({})", points.join(", "))
            }
        }
        GeoGeometry::MultiLineString(mls) => {
            if mls.0.is_empty() {
                "MULTILINESTRING EMPTY".to_string()
            } else {
                let line_strings: Vec<String> = mls
                    .0
                    .iter()
                    .map(|ls| {
                        let coords: Vec<String> =
                            ls.0.iter().map(|c| format!("{} {}", c.x, c.y)).collect();
                        format!("({})", coords.join(", "))
                    })
                    .collect();
                format!("MULTILINESTRING({})", line_strings.join(", "))
            }
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            if mpoly.0.is_empty() {
                "MULTIPOLYGON EMPTY".to_string()
            } else {
                let polygons: Vec<String> = mpoly
                    .0
                    .iter()
                    .map(|poly| {
                        let exterior: Vec<String> = poly
                            .exterior()
                            .0
                            .iter()
                            .map(|c| format!("{} {}", c.x, c.y))
                            .collect();

                        let mut rings = vec![format!("({})", exterior.join(", "))];

                        for interior in poly.interiors() {
                            let interior_coords: Vec<String> = interior
                                .0
                                .iter()
                                .map(|c| format!("{} {}", c.x, c.y))
                                .collect();
                            rings.push(format!("({})", interior_coords.join(", ")));
                        }

                        format!("({})", rings.join(", "))
                    })
                    .collect();
                format!("MULTIPOLYGON({})", polygons.join(", "))
            }
        }
        GeoGeometry::GeometryCollection(gc) => {
            if gc.0.is_empty() {
                "GEOMETRYCOLLECTION EMPTY".to_string()
            } else {
                let geometries: Vec<String> = gc.0.iter().map(geometry_to_wkt).collect();
                format!("GEOMETRYCOLLECTION({})", geometries.join(", "))
            }
        }
        GeoGeometry::Triangle(t) => {
            format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}))",
                t.0.x, t.0.y, t.1.x, t.1.y, t.2.x, t.2.y, t.0.x, t.0.y
            )
        }
        GeoGeometry::Rect(r) => {
            let min = r.min();
            let max = r.max();
            format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                min.x, min.y, max.x, min.y, max.x, max.y, min.x, max.y, min.x, min.y
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Point};

    #[test]
    fn test_parse_point() {
        let geom = parse_wkt("POINT(1.0 2.0)").unwrap();
        assert_eq!(geom.geometry_type(), "Point");

        match &geom.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point"),
        }
    }

    #[test]
    fn test_parse_linestring() {
        let geom = parse_wkt("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        assert_eq!(geom.geometry_type(), "LineString");

        match &geom.geom {
            GeoGeometry::LineString(ls) => {
                assert_eq!(ls.0.len(), 3);
                assert_eq!(ls.0[0].x, 0.0);
                assert_eq!(ls.0[2].y, 2.0);
            }
            _ => panic!("Expected LineString"),
        }
    }

    #[test]
    fn test_parse_polygon() {
        let geom = parse_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
        assert_eq!(geom.geometry_type(), "Polygon");

        match &geom.geom {
            GeoGeometry::Polygon(p) => {
                assert_eq!(p.exterior().0.len(), 5);
            }
            _ => panic!("Expected Polygon"),
        }
    }

    #[test]
    fn test_parse_with_crs() {
        let geom =
            parse_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1.0 2.0)").unwrap();
        assert_eq!(geom.crs.uri, "http://www.opengis.net/def/crs/EPSG/0/4326");
    }

    #[test]
    fn test_geometry_to_wkt_point() {
        let point = GeoGeometry::Point(Point::new(1.0, 2.0));
        let wkt = geometry_to_wkt(&point);
        assert_eq!(wkt, "POINT(1 2)");
    }

    #[test]
    fn test_geometry_to_wkt_linestring() {
        let ls = GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
        ]));
        let wkt = geometry_to_wkt(&ls);
        assert_eq!(wkt, "LINESTRING(0 0, 1 1)");
    }

    #[test]
    fn test_roundtrip() {
        let original_wkt = "POINT(1.5 2.5)";
        let geom = parse_wkt(original_wkt).unwrap();
        let new_wkt = geom.to_wkt();
        let geom2 = parse_wkt(&new_wkt).unwrap();

        match (&geom.geom, &geom2.geom) {
            (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) => {
                assert_eq!(p1.x(), p2.x());
                assert_eq!(p1.y(), p2.y());
            }
            _ => panic!("Expected Points"),
        }
    }
}
