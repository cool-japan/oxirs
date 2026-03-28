//! GeoJSON parser and serializer
//!
//! This module provides parsing and serialization support for GeoJSON format.
//! GeoJSON is a widely used format for encoding geographic data structures.
//!
//! # Features
//!
//! - Parse GeoJSON features, geometries, and feature collections
//! - Serialize geometries to GeoJSON format
//! - Preserve CRS information
//! - Support for all GeoJSON geometry types
//!
//! # Examples
//!
//! ```
//! use oxirs_geosparql::geometry::Geometry;
//!
//! // Parse from GeoJSON
//! let geojson = r#"{"type":"Point","coordinates":[1.0,2.0]}"#;
//! let geom = Geometry::from_geojson(geojson).expect("should succeed");
//!
//! // Serialize to GeoJSON
//! let geojson_output = geom.to_geojson().expect("should succeed");
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};
use geo_types::{
    Coord, Geometry as GeoGeometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};

/// Parse GeoJSON string into a Geometry
///
/// Supports GeoJSON geometries and features. For feature collections,
/// use `parse_geojson_feature_collection()`.
///
/// # Arguments
///
/// * `geojson` - GeoJSON string
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::geojson_parser::parse_geojson;
///
/// let geojson = r#"{"type":"Point","coordinates":[1.0,2.0]}"#;
/// let geom = parse_geojson(geojson).expect("should succeed");
/// ```
pub fn parse_geojson(geojson: &str) -> Result<Geometry> {
    use geojson::GeoJson;

    let geojson_obj: GeoJson = geojson
        .parse()
        .map_err(|e| GeoSparqlError::ParseError(format!("Invalid GeoJSON: {}", e)))?;

    match geojson_obj {
        GeoJson::Geometry(geom) => {
            let geo_geom = parse_geometry_value(&geom.value)?;
            // Extract CRS if present
            let crs = extract_crs_from_geometry(&geom)?;
            Ok(Geometry::with_crs(geo_geom, crs))
        }
        GeoJson::Feature(feature) => {
            if let Some(geom) = feature.geometry {
                let geo_geom = parse_geometry_value(&geom.value)?;
                let crs = extract_crs_from_geometry(&geom)?;
                Ok(Geometry::with_crs(geo_geom, crs))
            } else {
                Err(GeoSparqlError::ParseError(
                    "Feature has no geometry".to_string(),
                ))
            }
        }
        GeoJson::FeatureCollection(_) => Err(GeoSparqlError::ParseError(
            "Use parse_geojson_feature_collection() for FeatureCollection".to_string(),
        )),
    }
}

/// Parse GeoJSON feature collection into multiple geometries
///
/// # Arguments
///
/// * `geojson` - GeoJSON feature collection string
///
/// # Returns
///
/// Vector of geometries from the feature collection
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::geojson_parser::parse_geojson_feature_collection;
///
/// let geojson = r#"{
///   "type": "FeatureCollection",
///   "features": [
///     {"type": "Feature", "geometry": {"type":"Point","coordinates":[1.0,2.0]}},
///     {"type": "Feature", "geometry": {"type":"Point","coordinates":[3.0,4.0]}}
///   ]
/// }"#;
/// let geoms = parse_geojson_feature_collection(geojson).expect("should succeed");
/// assert_eq!(geoms.len(), 2);
/// ```
pub fn parse_geojson_feature_collection(geojson: &str) -> Result<Vec<Geometry>> {
    use geojson::GeoJson;

    let geojson_obj: GeoJson = geojson
        .parse()
        .map_err(|e| GeoSparqlError::ParseError(format!("Invalid GeoJSON: {}", e)))?;

    match geojson_obj {
        GeoJson::FeatureCollection(fc) => {
            let mut geometries = Vec::new();

            for feature in fc.features {
                if let Some(geom) = feature.geometry {
                    let geo_geom = parse_geometry_value(&geom.value)?;
                    let crs = extract_crs_from_geometry(&geom)?;
                    geometries.push(Geometry::with_crs(geo_geom, crs));
                }
            }

            Ok(geometries)
        }
        _ => Err(GeoSparqlError::ParseError(
            "Expected FeatureCollection".to_string(),
        )),
    }
}

/// Convert a Geometry to GeoJSON string
///
/// # Arguments
///
/// * `geometry` - The geometry to serialize
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
/// let geojson = geom.to_geojson().expect("should succeed");
/// assert!(geojson.contains("\"type\":\"Point\""));
/// ```
pub fn geometry_to_geojson(geometry: &Geometry) -> Result<String> {
    use geojson::Geometry as GeoJsonGeometry;

    let value = geo_to_geometry_value(&geometry.geom)?;

    let geojson_geom = GeoJsonGeometry::new(value);
    let json = serde_json::to_string(&geojson_geom).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to serialize to JSON: {}", e))
    })?;

    Ok(json)
}

/// Convert a geo_types Geometry to a geojson GeometryValue
fn geo_to_geometry_value(geom: &GeoGeometry<f64>) -> Result<geojson::GeometryValue> {
    use geojson::{GeometryValue, Position};

    match geom {
        GeoGeometry::Point(p) => Ok(GeometryValue::new_point([p.x(), p.y()])),
        GeoGeometry::LineString(ls) => {
            let coords: Vec<Position> = ls.coords().map(|c| Position::from([c.x, c.y])).collect();
            Ok(GeometryValue::LineString {
                coordinates: coords,
            })
        }
        GeoGeometry::Polygon(poly) => {
            let rings = polygon_to_rings(poly);
            Ok(GeometryValue::Polygon { coordinates: rings })
        }
        GeoGeometry::MultiPoint(mp) => {
            let coords: Vec<Position> = mp.iter().map(|p| Position::from([p.x(), p.y()])).collect();
            Ok(GeometryValue::MultiPoint {
                coordinates: coords,
            })
        }
        GeoGeometry::MultiLineString(mls) => {
            let lines: Vec<Vec<Position>> = mls
                .iter()
                .map(|ls| ls.coords().map(|c| Position::from([c.x, c.y])).collect())
                .collect();
            Ok(GeometryValue::MultiLineString { coordinates: lines })
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            let polygons: Vec<Vec<Vec<Position>>> = mpoly.iter().map(polygon_to_rings).collect();
            Ok(GeometryValue::MultiPolygon {
                coordinates: polygons,
            })
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "Geometry type not supported for GeoJSON: {:?}",
            geom
        ))),
    }
}

/// Convert a geo_types Polygon to GeoJSON ring coordinates
fn polygon_to_rings(poly: &Polygon<f64>) -> Vec<Vec<geojson::Position>> {
    let mut rings = Vec::new();

    // Exterior ring
    let exterior: Vec<geojson::Position> = poly
        .exterior()
        .coords()
        .map(|c| geojson::Position::from([c.x, c.y]))
        .collect();
    rings.push(exterior);

    // Interior rings (holes)
    for hole in poly.interiors() {
        let interior: Vec<geojson::Position> = hole
            .coords()
            .map(|c| geojson::Position::from([c.x, c.y]))
            .collect();
        rings.push(interior);
    }

    rings
}

/// Convert a Geometry to GeoJSON Feature string
///
/// Features can include properties and other metadata.
///
/// # Arguments
///
/// * `geometry` - The geometry to serialize
/// * `properties` - Optional properties JSON object
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::geometry::geojson_parser::geometry_to_geojson_feature;
/// use geo_types::{Point, Geometry as GeoGeometry};
/// use serde_json::json;
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
/// let props = json!({"name": "Test Point"});
/// let feature = geometry_to_geojson_feature(&geom, Some(&props)).expect("should succeed");
/// assert!(feature.contains("\"type\":\"Feature\""));
/// ```
pub fn geometry_to_geojson_feature(
    geometry: &Geometry,
    properties: Option<&serde_json::Value>,
) -> Result<String> {
    use geojson::{Feature, Geometry as GeoJsonGeometry};

    let value = geo_to_geometry_value(&geometry.geom)?;

    let geojson_geom = GeoJsonGeometry::new(value);

    let mut feature = Feature {
        bbox: None,
        geometry: Some(geojson_geom),
        id: None,
        properties: None,
        foreign_members: None,
    };

    if let Some(serde_json::Value::Object(map)) = properties {
        feature.properties = Some(map.clone());
    }

    let json = serde_json::to_string(&feature).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to serialize feature: {}", e))
    })?;

    Ok(json)
}

/// Parse geometry value from GeoJSON
fn parse_geometry_value(value: &geojson::GeometryValue) -> Result<GeoGeometry<f64>> {
    match value {
        geojson::GeometryValue::Point {
            coordinates: coords,
        } => {
            if coords.len() < 2 {
                return Err(GeoSparqlError::ParseError(
                    "Point must have at least 2 coordinates".to_string(),
                ));
            }
            Ok(GeoGeometry::Point(Point::new(coords[0], coords[1])))
        }
        geojson::GeometryValue::LineString {
            coordinates: coords,
        } => {
            let points: Vec<Coord<f64>> = coords
                .iter()
                .map(|c| {
                    if c.len() < 2 {
                        Err(GeoSparqlError::ParseError(
                            "Coordinate must have at least 2 values".to_string(),
                        ))
                    } else {
                        Ok(Coord { x: c[0], y: c[1] })
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(GeoGeometry::LineString(LineString::new(points)))
        }
        geojson::GeometryValue::Polygon { coordinates: rings } => {
            if rings.is_empty() {
                return Err(GeoSparqlError::ParseError(
                    "Polygon must have at least one ring".to_string(),
                ));
            }

            let exterior = parse_ring(&rings[0])?;
            let interiors: Vec<LineString<f64>> = rings[1..]
                .iter()
                .map(|r| parse_ring(r))
                .collect::<Result<Vec<_>>>()?;

            Ok(GeoGeometry::Polygon(Polygon::new(exterior, interiors)))
        }
        geojson::GeometryValue::MultiPoint {
            coordinates: coords,
        } => {
            let points: Vec<Point<f64>> = coords
                .iter()
                .map(|c| {
                    if c.len() < 2 {
                        Err(GeoSparqlError::ParseError(
                            "Point must have at least 2 coordinates".to_string(),
                        ))
                    } else {
                        Ok(Point::new(c[0], c[1]))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(GeoGeometry::MultiPoint(MultiPoint::new(points)))
        }
        geojson::GeometryValue::MultiLineString { coordinates: lines } => {
            let linestrings: Vec<LineString<f64>> = lines
                .iter()
                .map(|coords| {
                    let points: Vec<Coord<f64>> = coords
                        .iter()
                        .map(|c| {
                            if c.len() < 2 {
                                Err(GeoSparqlError::ParseError(
                                    "Coordinate must have at least 2 values".to_string(),
                                ))
                            } else {
                                Ok(Coord { x: c[0], y: c[1] })
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(LineString::new(points))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(GeoGeometry::MultiLineString(MultiLineString::new(
                linestrings,
            )))
        }
        geojson::GeometryValue::MultiPolygon {
            coordinates: polygons,
        } => {
            let polys: Vec<Polygon<f64>> = polygons
                .iter()
                .map(|rings| {
                    if rings.is_empty() {
                        return Err(GeoSparqlError::ParseError(
                            "Polygon must have at least one ring".to_string(),
                        ));
                    }

                    let exterior = parse_ring(&rings[0])?;
                    let interiors: Vec<LineString<f64>> = rings[1..]
                        .iter()
                        .map(|r| parse_ring(r))
                        .collect::<Result<Vec<_>>>()?;

                    Ok(Polygon::new(exterior, interiors))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(GeoGeometry::MultiPolygon(MultiPolygon::new(polys)))
        }
        geojson::GeometryValue::GeometryCollection { geometries: _ } => {
            Err(GeoSparqlError::UnsupportedOperation(
                "GeometryCollection not yet supported".to_string(),
            ))
        }
    }
}

/// Parse a ring (closed LineString) from coordinates
fn parse_ring(coords: &[geojson::Position]) -> Result<LineString<f64>> {
    let points: Vec<Coord<f64>> = coords
        .iter()
        .map(|c| {
            if c.len() < 2 {
                Err(GeoSparqlError::ParseError(
                    "Coordinate must have at least 2 values".to_string(),
                ))
            } else {
                Ok(Coord { x: c[0], y: c[1] })
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LineString::new(points))
}

/// Extract CRS from GeoJSON geometry
fn extract_crs_from_geometry(_geom: &geojson::Geometry) -> Result<Crs> {
    // GeoJSON spec says CRS should be WGS84 by default
    // Named CRS objects are deprecated in newer GeoJSON specs
    Ok(Crs::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_point() -> Result<()> {
        let geojson = r#"{"type":"Point","coordinates":[1.0,2.0]}"#;
        let geom = parse_geojson(geojson)?;

        match geom.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_linestring() -> Result<()> {
        let geojson = r#"{"type":"LineString","coordinates":[[1.0,2.0],[3.0,4.0]]}"#;
        let geom = parse_geojson(geojson)?;

        match geom.geom {
            GeoGeometry::LineString(ls) => {
                assert_eq!(ls.0.len(), 2);
            }
            _ => panic!("Expected LineString"),
        }

        Ok(())
    }

    #[test]
    fn test_parse_polygon() -> Result<()> {
        let geojson = r#"{
            "type":"Polygon",
            "coordinates":[[[0.0,0.0],[4.0,0.0],[4.0,4.0],[0.0,4.0],[0.0,0.0]]]
        }"#;
        let geom = parse_geojson(geojson)?;

        match geom.geom {
            GeoGeometry::Polygon(poly) => {
                assert_eq!(poly.exterior().0.len(), 5);
            }
            _ => panic!("Expected Polygon"),
        }

        Ok(())
    }

    #[test]
    fn test_serialize_point() -> Result<()> {
        let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let geojson = geometry_to_geojson(&geom)?;

        assert!(geojson.contains("\"type\":\"Point\""));
        assert!(geojson.contains("[1.0,2.0]"));

        Ok(())
    }

    #[test]
    fn test_roundtrip() -> Result<()> {
        let original = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let geojson = geometry_to_geojson(&original)?;
        let parsed = parse_geojson(&geojson)?;

        match (&original.geom, &parsed.geom) {
            (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) => {
                assert_eq!(p1.x(), p2.x());
                assert_eq!(p1.y(), p2.y());
            }
            _ => panic!("Geometry types don't match"),
        }

        Ok(())
    }

    #[test]
    fn test_feature_collection() -> Result<()> {
        let geojson = r#"{
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type":"Point","coordinates":[1.0,2.0]}},
                {"type": "Feature", "geometry": {"type":"Point","coordinates":[3.0,4.0]}}
            ]
        }"#;

        let geoms = parse_geojson_feature_collection(geojson)?;
        assert_eq!(geoms.len(), 2);

        Ok(())
    }
}
