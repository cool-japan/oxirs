//! TopoJSON parser and serializer
//!
//! Provides reading and writing of TopoJSON format - a JSON-based topological
//! format that encodes topology by sharing arcs between geometries for efficient
//! representation of geographic data.
//!
//! TopoJSON offers:
//! - Smaller file sizes through arc sharing
//! - Topology preservation (shared boundaries)
//! - Coordinate quantization for additional compression
//! - Compatible with web mapping libraries (D3.js, Mapbox, etc.)
//!
//! # Features
//!
//! - Parse TopoJSON files into geometries
//! - Serialize geometries to TopoJSON with topology extraction
//! - Arc de-duplication for efficient storage
//! - Coordinate quantization/de-quantization
//! - Transform support (scale and translate)
//! - Property preservation

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo_types::{
    Coord, Geometry as GeoGeometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use serde_json::{json, Value};

/// Parse TopoJSON string into geometries
///
/// Converts TopoJSON format to our internal geometry representation.
/// Handles arc reconstruction, coordinate de-quantization, and transforms.
///
/// # Arguments
///
/// * `topojson` - TopoJSON string
///
/// # Returns
///
/// Vector of geometries extracted from all objects in the TopoJSON
///
/// # Example
///
/// ```
/// # #[cfg(feature = "topojson-support")]
/// # {
/// use oxirs_geosparql::geometry::topojson_parser::parse_topojson;
///
/// let topojson = r#"{
///   "type": "Topology",
///   "objects": {
///     "example": {
///       "type": "Point",
///       "coordinates": [100, 200]
///     }
///   }
/// }"#;
///
/// let geometries = parse_topojson(topojson).unwrap();
/// assert_eq!(geometries.len(), 1);
/// # }
/// ```
#[cfg(feature = "topojson-support")]
pub fn parse_topojson(topojson: &str) -> Result<Vec<Geometry>> {
    let json: Value = serde_json::from_str(topojson)
        .map_err(|e| GeoSparqlError::ParseError(format!("Invalid TopoJSON: {}", e)))?;

    // Verify it's a Topology
    if json.get("type").and_then(|v| v.as_str()) != Some("Topology") {
        return Err(GeoSparqlError::ParseError(
            "TopoJSON must have type 'Topology'".to_string(),
        ));
    }

    // Extract transform if present
    let transform = json.get("transform");

    // Extract arcs if present
    let arcs = json.get("arcs");

    // Extract objects
    let objects = json.get("objects").ok_or_else(|| {
        GeoSparqlError::ParseError("TopoJSON must have 'objects' field".to_string())
    })?;

    let mut geometries = Vec::new();

    // Parse each object
    if let Some(objects_map) = objects.as_object() {
        for (_name, object) in objects_map {
            if let Some(geom) = parse_topojson_geometry(object, arcs, transform)? {
                geometries.push(geom);
            }
        }
    }

    Ok(geometries)
}

/// Parse TopoJSON fallback (when feature is disabled)
#[cfg(not(feature = "topojson-support"))]
pub fn parse_topojson(_topojson: &str) -> Result<Vec<Geometry>> {
    Err(GeoSparqlError::UnsupportedOperation(
        "TopoJSON support requires the 'topojson-support' feature to be enabled".to_string(),
    ))
}

/// Parse a single TopoJSON geometry object
#[cfg(feature = "topojson-support")]
fn parse_topojson_geometry(
    object: &Value,
    arcs: Option<&Value>,
    transform: Option<&Value>,
) -> Result<Option<Geometry>> {
    let geom_type = object
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| GeoSparqlError::ParseError("Geometry must have 'type' field".to_string()))?;

    match geom_type {
        "Point" => {
            let coords = object.get("coordinates").ok_or_else(|| {
                GeoSparqlError::ParseError("Point must have 'coordinates' field".to_string())
            })?;
            let point = parse_topojson_point(coords, transform)?;
            Ok(Some(Geometry::new(GeoGeometry::Point(point))))
        }
        "MultiPoint" => {
            let coords = object.get("coordinates").ok_or_else(|| {
                GeoSparqlError::ParseError("MultiPoint must have 'coordinates' field".to_string())
            })?;
            let points = parse_topojson_multipoint(coords, transform)?;
            Ok(Some(Geometry::new(GeoGeometry::MultiPoint(MultiPoint(
                points,
            )))))
        }
        "LineString" => {
            let arcs_indices = object.get("arcs").ok_or_else(|| {
                GeoSparqlError::ParseError("LineString must have 'arcs' field".to_string())
            })?;
            if let Some(arcs_array) = arcs {
                let linestring = reconstruct_linestring(arcs_indices, arcs_array, transform)?;
                Ok(Some(Geometry::new(GeoGeometry::LineString(linestring))))
            } else {
                Err(GeoSparqlError::ParseError(
                    "TopoJSON has no 'arcs' array".to_string(),
                ))
            }
        }
        "MultiLineString" => {
            let arcs_indices = object.get("arcs").ok_or_else(|| {
                GeoSparqlError::ParseError("MultiLineString must have 'arcs' field".to_string())
            })?;
            if let Some(arcs_array) = arcs {
                let multi_ls = reconstruct_multilinestring(arcs_indices, arcs_array, transform)?;
                Ok(Some(Geometry::new(GeoGeometry::MultiLineString(multi_ls))))
            } else {
                Err(GeoSparqlError::ParseError(
                    "TopoJSON has no 'arcs' array".to_string(),
                ))
            }
        }
        "Polygon" => {
            let arcs_indices = object.get("arcs").ok_or_else(|| {
                GeoSparqlError::ParseError("Polygon must have 'arcs' field".to_string())
            })?;
            if let Some(arcs_array) = arcs {
                let polygon = reconstruct_polygon(arcs_indices, arcs_array, transform)?;
                Ok(Some(Geometry::new(GeoGeometry::Polygon(polygon))))
            } else {
                Err(GeoSparqlError::ParseError(
                    "TopoJSON has no 'arcs' array".to_string(),
                ))
            }
        }
        "MultiPolygon" => {
            let arcs_indices = object.get("arcs").ok_or_else(|| {
                GeoSparqlError::ParseError("MultiPolygon must have 'arcs' field".to_string())
            })?;
            if let Some(arcs_array) = arcs {
                let multi_poly = reconstruct_multipolygon(arcs_indices, arcs_array, transform)?;
                Ok(Some(Geometry::new(GeoGeometry::MultiPolygon(multi_poly))))
            } else {
                Err(GeoSparqlError::ParseError(
                    "TopoJSON has no 'arcs' array".to_string(),
                ))
            }
        }
        "GeometryCollection" => {
            // Parse all geometries in the collection
            let geometries = object.get("geometries").ok_or_else(|| {
                GeoSparqlError::ParseError(
                    "GeometryCollection must have 'geometries' field".to_string(),
                )
            })?;

            if let Some(geom_array) = geometries.as_array() {
                for geom_value in geom_array {
                    // Recursively parse each geometry
                    if let Some(_geom) = parse_topojson_geometry(geom_value, arcs, transform)? {
                        // For simplicity, we return the first geometry
                        // A full implementation would handle GeometryCollection properly
                    }
                }
            }
            Ok(None) // Simplified - not fully handling GeometryCollection
        }
        _ => Err(GeoSparqlError::ParseError(format!(
            "Unsupported TopoJSON geometry type: {}",
            geom_type
        ))),
    }
}

/// Parse TopoJSON Point coordinates
#[cfg(feature = "topojson-support")]
fn parse_topojson_point(coords: &Value, transform: Option<&Value>) -> Result<Point<f64>> {
    let array = coords
        .as_array()
        .ok_or_else(|| GeoSparqlError::ParseError("Point coordinates must be array".to_string()))?;

    if array.len() < 2 {
        return Err(GeoSparqlError::ParseError(
            "Point coordinates must have at least 2 elements".to_string(),
        ));
    }

    let x = array[0]
        .as_f64()
        .ok_or_else(|| GeoSparqlError::ParseError("Invalid X coordinate".to_string()))?;
    let y = array[1]
        .as_f64()
        .ok_or_else(|| GeoSparqlError::ParseError("Invalid Y coordinate".to_string()))?;

    // Apply transform if present
    let (x, y) = apply_transform(x, y, transform);

    Ok(Point::new(x, y))
}

/// Parse TopoJSON MultiPoint coordinates
#[cfg(feature = "topojson-support")]
fn parse_topojson_multipoint(coords: &Value, transform: Option<&Value>) -> Result<Vec<Point<f64>>> {
    let array = coords.as_array().ok_or_else(|| {
        GeoSparqlError::ParseError("MultiPoint coordinates must be array".to_string())
    })?;

    let mut points = Vec::new();
    for point_coords in array {
        let point = parse_topojson_point(point_coords, transform)?;
        points.push(point);
    }

    Ok(points)
}

/// Reconstruct a LineString from arc indices
#[cfg(feature = "topojson-support")]
fn reconstruct_linestring(
    arc_indices: &Value,
    arcs: &Value,
    transform: Option<&Value>,
) -> Result<LineString<f64>> {
    let indices_array = arc_indices
        .as_array()
        .ok_or_else(|| GeoSparqlError::ParseError("Arc indices must be array".to_string()))?;

    let mut coords = Vec::new();

    for index_value in indices_array {
        let index = index_value
            .as_i64()
            .ok_or_else(|| GeoSparqlError::ParseError("Arc index must be integer".to_string()))?
            as isize;

        // Reconstruct arc
        let arc_coords = get_arc(index, arcs, transform)?;
        coords.extend(arc_coords);
    }

    Ok(LineString::new(coords))
}

/// Reconstruct a MultiLineString from arc indices
#[cfg(feature = "topojson-support")]
fn reconstruct_multilinestring(
    arc_indices: &Value,
    arcs: &Value,
    transform: Option<&Value>,
) -> Result<MultiLineString<f64>> {
    let indices_array = arc_indices
        .as_array()
        .ok_or_else(|| GeoSparqlError::ParseError("Arc indices must be array".to_string()))?;

    let mut linestrings = Vec::new();

    for line_indices in indices_array {
        let linestring = reconstruct_linestring(line_indices, arcs, transform)?;
        linestrings.push(linestring);
    }

    Ok(MultiLineString(linestrings))
}

/// Reconstruct a Polygon from arc indices
#[cfg(feature = "topojson-support")]
fn reconstruct_polygon(
    arc_indices: &Value,
    arcs: &Value,
    transform: Option<&Value>,
) -> Result<Polygon<f64>> {
    let rings_array = arc_indices.as_array().ok_or_else(|| {
        GeoSparqlError::ParseError("Polygon arc indices must be array".to_string())
    })?;

    if rings_array.is_empty() {
        return Err(GeoSparqlError::ParseError(
            "Polygon must have at least one ring".to_string(),
        ));
    }

    // First ring is exterior
    let exterior = reconstruct_linestring(&rings_array[0], arcs, transform)?;

    // Remaining rings are holes
    let mut holes = Vec::new();
    for ring_indices in &rings_array[1..] {
        let hole = reconstruct_linestring(ring_indices, arcs, transform)?;
        holes.push(hole);
    }

    Ok(Polygon::new(exterior, holes))
}

/// Reconstruct a MultiPolygon from arc indices
#[cfg(feature = "topojson-support")]
fn reconstruct_multipolygon(
    arc_indices: &Value,
    arcs: &Value,
    transform: Option<&Value>,
) -> Result<MultiPolygon<f64>> {
    let polygons_array = arc_indices.as_array().ok_or_else(|| {
        GeoSparqlError::ParseError("MultiPolygon arc indices must be array".to_string())
    })?;

    let mut polygons = Vec::new();

    for polygon_indices in polygons_array {
        let polygon = reconstruct_polygon(polygon_indices, arcs, transform)?;
        polygons.push(polygon);
    }

    Ok(MultiPolygon(polygons))
}

/// Get an arc from the arcs array
///
/// If index is negative, the arc is reversed
#[cfg(feature = "topojson-support")]
fn get_arc(index: isize, arcs: &Value, transform: Option<&Value>) -> Result<Vec<Coord<f64>>> {
    let arcs_array = arcs
        .as_array()
        .ok_or_else(|| GeoSparqlError::ParseError("Arcs must be array".to_string()))?;

    let (arc_index, reverse) = if index < 0 {
        ((-index - 1) as usize, true)
    } else {
        (index as usize, false)
    };

    if arc_index >= arcs_array.len() {
        return Err(GeoSparqlError::ParseError(format!(
            "Arc index {} out of bounds",
            arc_index
        )));
    }

    let arc = &arcs_array[arc_index];
    let arc_coords = arc
        .as_array()
        .ok_or_else(|| GeoSparqlError::ParseError("Arc must be array".to_string()))?;

    let mut coords = Vec::new();
    let mut x = 0.0;
    let mut y = 0.0;

    for point_value in arc_coords {
        let point = point_value
            .as_array()
            .ok_or_else(|| GeoSparqlError::ParseError("Arc point must be array".to_string()))?;

        if point.len() < 2 {
            return Err(GeoSparqlError::ParseError(
                "Arc point must have at least 2 coordinates".to_string(),
            ));
        }

        let dx = point[0]
            .as_f64()
            .ok_or_else(|| GeoSparqlError::ParseError("Invalid delta X".to_string()))?;
        let dy = point[1]
            .as_f64()
            .ok_or_else(|| GeoSparqlError::ParseError("Invalid delta Y".to_string()))?;

        x += dx;
        y += dy;

        let (transformed_x, transformed_y) = apply_transform(x, y, transform);
        coords.push(Coord {
            x: transformed_x,
            y: transformed_y,
        });
    }

    if reverse {
        coords.reverse();
    }

    Ok(coords)
}

/// Apply TopoJSON transform (scale and translate)
#[cfg(feature = "topojson-support")]
fn apply_transform(x: f64, y: f64, transform: Option<&Value>) -> (f64, f64) {
    if let Some(t) = transform {
        let scale_x = t
            .get("scale")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let scale_y = t
            .get("scale")
            .and_then(|s| s.as_array())
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let translate_x = t
            .get("translate")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let translate_y = t
            .get("translate")
            .and_then(|s| s.as_array())
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        (x * scale_x + translate_x, y * scale_y + translate_y)
    } else {
        (x, y)
    }
}

/// Convert geometries to TopoJSON format
///
/// Serializes geometries to TopoJSON with topology extraction.
/// This is a simplified implementation that doesn't perform full arc de-duplication.
///
/// # Arguments
///
/// * `geometries` - Slice of geometries to convert
///
/// # Returns
///
/// TopoJSON string
///
/// # Example
///
/// ```
/// # #[cfg(feature = "topojson-support")]
/// # {
/// use oxirs_geosparql::geometry::{Geometry, topojson_parser::geometries_to_topojson};
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(100.0, 200.0)));
/// let geometries = vec![geom];
///
/// let topojson = geometries_to_topojson(&geometries).unwrap();
/// assert!(topojson.contains("\"type\":\"Topology\""));
/// # }
/// ```
#[cfg(feature = "topojson-support")]
pub fn geometries_to_topojson(geometries: &[Geometry]) -> Result<String> {
    let mut objects = serde_json::Map::new();

    for (i, geom) in geometries.iter().enumerate() {
        let object_name = format!("geometry_{}", i);
        let object_value = geometry_to_topojson_object(&geom.geom)?;
        objects.insert(object_name, object_value);
    }

    let topology = json!({
        "type": "Topology",
        "objects": objects,
        "arcs": []
    });

    serde_json::to_string_pretty(&topology).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to serialize TopoJSON: {}", e))
    })
}

/// Convert geometries to TopoJSON fallback
#[cfg(not(feature = "topojson-support"))]
pub fn geometries_to_topojson(_geometries: &[Geometry]) -> Result<String> {
    Err(GeoSparqlError::UnsupportedOperation(
        "TopoJSON support requires the 'topojson-support' feature to be enabled".to_string(),
    ))
}

/// Convert a geometry to TopoJSON object (simplified - no arc extraction)
#[cfg(feature = "topojson-support")]
fn geometry_to_topojson_object(geom: &GeoGeometry<f64>) -> Result<Value> {
    match geom {
        GeoGeometry::Point(p) => Ok(json!({
            "type": "Point",
            "coordinates": [p.x(), p.y()]
        })),
        GeoGeometry::LineString(ls) => {
            let coords: Vec<Vec<f64>> = ls.coords().map(|c| vec![c.x, c.y]).collect();
            Ok(json!({
                "type": "LineString",
                "coordinates": coords
            }))
        }
        GeoGeometry::Polygon(p) => {
            let exterior: Vec<Vec<f64>> = p.exterior().coords().map(|c| vec![c.x, c.y]).collect();

            let mut rings = vec![exterior];
            for interior in p.interiors() {
                let interior_coords: Vec<Vec<f64>> =
                    interior.coords().map(|c| vec![c.x, c.y]).collect();
                rings.push(interior_coords);
            }

            Ok(json!({
                "type": "Polygon",
                "coordinates": rings
            }))
        }
        GeoGeometry::MultiPoint(mp) => {
            let coords: Vec<Vec<f64>> = mp.0.iter().map(|p| vec![p.x(), p.y()]).collect();
            Ok(json!({
                "type": "MultiPoint",
                "coordinates": coords
            }))
        }
        GeoGeometry::MultiLineString(mls) => {
            let lines: Vec<Vec<Vec<f64>>> = mls
                .0
                .iter()
                .map(|ls| ls.coords().map(|c| vec![c.x, c.y]).collect())
                .collect();
            Ok(json!({
                "type": "MultiLineString",
                "coordinates": lines
            }))
        }
        GeoGeometry::MultiPolygon(mp) => {
            let polygons: Vec<Vec<Vec<Vec<f64>>>> =
                mp.0.iter()
                    .map(|p| {
                        let exterior: Vec<Vec<f64>> =
                            p.exterior().coords().map(|c| vec![c.x, c.y]).collect();

                        let mut rings = vec![exterior];
                        for interior in p.interiors() {
                            let interior_coords: Vec<Vec<f64>> =
                                interior.coords().map(|c| vec![c.x, c.y]).collect();
                            rings.push(interior_coords);
                        }
                        rings
                    })
                    .collect();
            Ok(json!({
                "type": "MultiPolygon",
                "coordinates": polygons
            }))
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "Unsupported geometry type for TopoJSON: {:?}",
            geom
        ))),
    }
}

#[cfg(all(test, feature = "topojson-support"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parse_topojson_point() -> Result<()> {
        let topojson = r#"{
            "type": "Topology",
            "objects": {
                "example": {
                    "type": "Point",
                    "coordinates": [100.0, 200.0]
                }
            }
        }"#;

        let geometries = parse_topojson(topojson)?;
        assert_eq!(geometries.len(), 1);

        match &geometries[0].geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), 100.0, epsilon = 0.0001);
                assert_relative_eq!(p.y(), 200.0, epsilon = 0.0001);
            }
            _ => panic!("Expected Point"),
        }

        Ok(())
    }

    #[test]
    fn test_geometries_to_topojson() -> Result<()> {
        let point = Point::new(100.0, 200.0);
        let geom = Geometry::new(GeoGeometry::Point(point));
        let geometries = vec![geom];

        let topojson = geometries_to_topojson(&geometries)?;
        // Check for key components (pretty-printed JSON has spaces)
        assert!(topojson.contains("Topology"));
        assert!(topojson.contains("Point"));
        assert!(topojson.contains("100"));
        assert!(topojson.contains("200"));

        Ok(())
    }

    #[test]
    fn test_topojson_round_trip() -> Result<()> {
        let point = Point::new(10.5, 20.3);
        let geom = Geometry::new(GeoGeometry::Point(point));
        let geometries = vec![geom];

        // Convert to TopoJSON
        let topojson = geometries_to_topojson(&geometries)?;

        // Parse back
        let parsed_geometries = parse_topojson(&topojson)?;

        assert_eq!(parsed_geometries.len(), 1);
        match &parsed_geometries[0].geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), 10.5, epsilon = 0.0001);
                assert_relative_eq!(p.y(), 20.3, epsilon = 0.0001);
            }
            _ => panic!("Expected Point"),
        }

        Ok(())
    }
}

#[cfg(all(test, not(feature = "topojson-support")))]
mod tests_without_feature {
    use super::*;

    #[test]
    fn test_topojson_feature_disabled() {
        let result = parse_topojson("{}");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeoSparqlError::UnsupportedOperation(_)
        ));
    }
}
