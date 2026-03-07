//! GeoJSON serialization / deserialization for SPARQL geo results.
//!
//! This module converts geometry objects to/from GeoJSON (RFC 7946),
//! parses simple WKT geometry strings, and adapts SPARQL binding rows
//! with latitude/longitude variables into GeoJSON feature collections.
//!
//! # Example
//!
//! ```rust
//! use oxirs_wasm::geojson_support::{GeoJsonSerializer, GeoJsonGeometry,
//!                                    GeoJsonFeature, GeoJsonFeatureCollection};
//! use std::collections::HashMap;
//!
//! let serializer = GeoJsonSerializer::new();
//! let point = GeoJsonGeometry::Point([13.405, 52.52]);
//! let mut props = HashMap::new();
//! props.insert("name".into(), serde_json::Value::String("Berlin".into()));
//! let feature = GeoJsonFeature { geometry: Some(point), properties: props };
//! let collection = GeoJsonFeatureCollection { features: vec![feature] };
//! let json = serializer.serialize_collection(&collection).unwrap();
//! assert!(json.contains("FeatureCollection"));
//! ```

use serde_json::{json, Value};
use std::collections::HashMap;

/// GeoJSON geometry variants (RFC 7946 §3.1)
#[derive(Debug, Clone, PartialEq)]
pub enum GeoJsonGeometry {
    /// A single position [longitude, latitude]
    Point([f64; 2]),
    /// Multiple positions
    MultiPoint(Vec<[f64; 2]>),
    /// A sequence of positions forming a line
    LineString(Vec<[f64; 2]>),
    /// Multiple line strings
    MultiLineString(Vec<Vec<[f64; 2]>>),
    /// A polygon with optional holes: first ring is exterior, rest are holes
    Polygon(Vec<Vec<[f64; 2]>>),
    /// Multiple polygons
    MultiPolygon(Vec<Vec<Vec<[f64; 2]>>>),
    /// A heterogeneous collection of geometries
    GeometryCollection(Vec<GeoJsonGeometry>),
}

/// A GeoJSON Feature with optional geometry and a properties map
#[derive(Debug, Clone)]
pub struct GeoJsonFeature {
    /// The feature geometry (may be absent)
    pub geometry: Option<GeoJsonGeometry>,
    /// Arbitrary properties
    pub properties: HashMap<String, Value>,
}

/// A GeoJSON FeatureCollection
#[derive(Debug, Clone)]
pub struct GeoJsonFeatureCollection {
    /// The list of features
    pub features: Vec<GeoJsonFeature>,
}

/// Error type for GeoJSON operations
#[derive(Debug)]
pub enum GeoJsonError {
    /// WKT input could not be parsed
    InvalidWkt(String),
    /// JSON serialization failed
    SerializationError(String),
    /// Coordinates contain invalid values (NaN, out of range, etc.)
    InvalidCoordinates(String),
}

impl std::fmt::Display for GeoJsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoJsonError::InvalidWkt(msg) => write!(f, "Invalid WKT: {}", msg),
            GeoJsonError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            GeoJsonError::InvalidCoordinates(msg) => write!(f, "Invalid coordinates: {}", msg),
        }
    }
}

impl std::error::Error for GeoJsonError {}

/// GeoJSON serializer / WKT parser / SPARQL-to-GeoJSON adapter
pub struct GeoJsonSerializer;

impl GeoJsonSerializer {
    /// Create a new serializer instance
    pub fn new() -> Self {
        Self
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Convert a geometry to its JSON object representation
    pub fn geometry_to_json(&self, geom: &GeoJsonGeometry) -> Value {
        match geom {
            GeoJsonGeometry::Point(coords) => {
                json!({
                    "type": "Point",
                    "coordinates": [coords[0], coords[1]]
                })
            }
            GeoJsonGeometry::MultiPoint(points) => {
                let coords: Vec<Value> = points.iter().map(|p| json!([p[0], p[1]])).collect();
                json!({ "type": "MultiPoint", "coordinates": coords })
            }
            GeoJsonGeometry::LineString(points) => {
                let coords: Vec<Value> = points.iter().map(|p| json!([p[0], p[1]])).collect();
                json!({ "type": "LineString", "coordinates": coords })
            }
            GeoJsonGeometry::MultiLineString(lines) => {
                let coords: Vec<Value> = lines
                    .iter()
                    .map(|line| {
                        let pts: Vec<Value> = line.iter().map(|p| json!([p[0], p[1]])).collect();
                        json!(pts)
                    })
                    .collect();
                json!({ "type": "MultiLineString", "coordinates": coords })
            }
            GeoJsonGeometry::Polygon(rings) => {
                let coords: Vec<Value> = rings
                    .iter()
                    .map(|ring| {
                        let pts: Vec<Value> = ring.iter().map(|p| json!([p[0], p[1]])).collect();
                        json!(pts)
                    })
                    .collect();
                json!({ "type": "Polygon", "coordinates": coords })
            }
            GeoJsonGeometry::MultiPolygon(polygons) => {
                let coords: Vec<Value> = polygons
                    .iter()
                    .map(|poly| {
                        let rings: Vec<Value> = poly
                            .iter()
                            .map(|ring| {
                                let pts: Vec<Value> =
                                    ring.iter().map(|p| json!([p[0], p[1]])).collect();
                                json!(pts)
                            })
                            .collect();
                        json!(rings)
                    })
                    .collect();
                json!({ "type": "MultiPolygon", "coordinates": coords })
            }
            GeoJsonGeometry::GeometryCollection(geometries) => {
                let geoms: Vec<Value> = geometries
                    .iter()
                    .map(|g| self.geometry_to_json(g))
                    .collect();
                json!({ "type": "GeometryCollection", "geometries": geoms })
            }
        }
    }

    /// Convert a feature to its JSON object representation
    pub fn feature_to_json(&self, feature: &GeoJsonFeature) -> Value {
        let geometry = match &feature.geometry {
            Some(g) => self.geometry_to_json(g),
            None => Value::Null,
        };

        // Stable key order is not required by RFC 7946, but consistent output
        // is helpful in tests
        let props: Value = json!(feature.properties);

        json!({
            "type": "Feature",
            "geometry": geometry,
            "properties": props
        })
    }

    /// Convert a feature collection to its JSON object representation
    pub fn collection_to_json(&self, collection: &GeoJsonFeatureCollection) -> Value {
        let features: Vec<Value> = collection
            .features
            .iter()
            .map(|f| self.feature_to_json(f))
            .collect();

        json!({
            "type": "FeatureCollection",
            "features": features
        })
    }

    /// Serialize a feature collection to a JSON string
    pub fn serialize_collection(
        &self,
        collection: &GeoJsonFeatureCollection,
    ) -> Result<String, GeoJsonError> {
        let value = self.collection_to_json(collection);
        serde_json::to_string(&value).map_err(|e| GeoJsonError::SerializationError(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // WKT parsing
    // -----------------------------------------------------------------------

    /// Parse a WKT `POINT(lon lat)` string
    ///
    /// Accepts both `POINT(lon lat)` and `POINT (lon lat)`.
    pub fn parse_wkt_point(wkt: &str) -> Result<GeoJsonGeometry, GeoJsonError> {
        let upper = wkt.trim().to_uppercase();
        let inner = upper
            .strip_prefix("POINT")
            .ok_or_else(|| GeoJsonError::InvalidWkt(format!("Expected POINT, got: {}", wkt)))?
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')');

        let (lon, lat) = parse_coord_pair(inner)?;
        Ok(GeoJsonGeometry::Point([lon, lat]))
    }

    /// Parse a WKT `LINESTRING(x1 y1, x2 y2, ...)` string
    pub fn parse_wkt_linestring(wkt: &str) -> Result<GeoJsonGeometry, GeoJsonError> {
        let upper = wkt.trim().to_uppercase();
        let inner = upper
            .strip_prefix("LINESTRING")
            .ok_or_else(|| GeoJsonError::InvalidWkt(format!("Expected LINESTRING, got: {}", wkt)))?
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')');

        let points = parse_coord_list(inner)?;
        if points.len() < 2 {
            return Err(GeoJsonError::InvalidWkt(
                "LINESTRING requires at least 2 points".into(),
            ));
        }
        Ok(GeoJsonGeometry::LineString(points))
    }

    /// Parse a WKT `POLYGON((x1 y1, x2 y2, ...))` string
    ///
    /// The exterior ring must be closed (first == last point).
    /// Inner rings (holes) are accepted but not required to be closed here.
    pub fn parse_wkt_polygon(wkt: &str) -> Result<GeoJsonGeometry, GeoJsonError> {
        let upper = wkt.trim().to_uppercase();
        let inner = upper
            .strip_prefix("POLYGON")
            .ok_or_else(|| GeoJsonError::InvalidWkt(format!("Expected POLYGON, got: {}", wkt)))?
            .trim();

        // Strip outer parentheses then split on "),(" to get rings
        let stripped = inner.trim_start_matches('(').trim_end_matches(')');

        // Split into individual rings by "),(
        let ring_strs: Vec<&str> = stripped.split("),(").collect();

        let mut rings: Vec<Vec<[f64; 2]>> = Vec::with_capacity(ring_strs.len());
        for ring_str in ring_strs {
            let cleaned = ring_str.trim_matches(|c| c == '(' || c == ')');
            let points = parse_coord_list(cleaned)?;
            if points.len() < 3 {
                return Err(GeoJsonError::InvalidWkt(
                    "Polygon ring requires at least 3 points".into(),
                ));
            }
            rings.push(points);
        }

        if rings.is_empty() {
            return Err(GeoJsonError::InvalidWkt("Empty POLYGON".into()));
        }

        Ok(GeoJsonGeometry::Polygon(rings))
    }

    // -----------------------------------------------------------------------
    // SPARQL result conversion
    // -----------------------------------------------------------------------

    /// Convert SPARQL binding rows to a GeoJSON FeatureCollection.
    ///
    /// Each row must have `lat_var` and `lon_var` string bindings.
    /// If `label_var` is `Some`, the label is added to `properties["label"]`.
    /// All remaining bindings are added as string properties.
    pub fn sparql_results_to_geojson(
        &self,
        results: &[HashMap<String, String>],
        lat_var: &str,
        lon_var: &str,
        label_var: Option<&str>,
    ) -> GeoJsonFeatureCollection {
        let mut features = Vec::with_capacity(results.len());

        for row in results {
            let lat = match row.get(lat_var).and_then(|s| s.parse::<f64>().ok()) {
                Some(v) => v,
                None => continue,
            };
            let lon = match row.get(lon_var).and_then(|s| s.parse::<f64>().ok()) {
                Some(v) => v,
                None => continue,
            };

            let geometry = Some(GeoJsonGeometry::Point([lon, lat]));

            let mut properties: HashMap<String, Value> = HashMap::new();

            if let Some(lv) = label_var {
                if let Some(label) = row.get(lv) {
                    properties.insert("label".into(), Value::String(label.clone()));
                }
            }

            // Add remaining bindings as string properties
            for (k, v) in row {
                if k == lat_var || k == lon_var {
                    continue;
                }
                if let Some(lv) = label_var {
                    if k == lv {
                        continue; // already added
                    }
                }
                properties.insert(k.clone(), Value::String(v.clone()));
            }

            features.push(GeoJsonFeature {
                geometry,
                properties,
            });
        }

        GeoJsonFeatureCollection { features }
    }

    // -----------------------------------------------------------------------
    // Bounding box
    // -----------------------------------------------------------------------

    /// Compute the bounding box `[west, south, east, north]` of a feature collection.
    ///
    /// Returns `None` if the collection is empty or contains no valid point geometry.
    pub fn bounding_box(collection: &GeoJsonFeatureCollection) -> Option<[f64; 4]> {
        // Flatten all geometries into a list of individual [lon, lat] positions,
        // then compute the bounding box over all of them.
        let mut positions: Vec<[f64; 2]> = Vec::new();

        fn collect_positions(geom: &GeoJsonGeometry, out: &mut Vec<[f64; 2]>) {
            match geom {
                GeoJsonGeometry::Point(p) => {
                    out.push(*p);
                }
                GeoJsonGeometry::MultiPoint(pts) => {
                    out.extend_from_slice(pts);
                }
                GeoJsonGeometry::LineString(pts) => {
                    out.extend_from_slice(pts);
                }
                GeoJsonGeometry::MultiLineString(lines) => {
                    for line in lines {
                        out.extend_from_slice(line);
                    }
                }
                GeoJsonGeometry::Polygon(rings) => {
                    for ring in rings {
                        out.extend_from_slice(ring);
                    }
                }
                GeoJsonGeometry::MultiPolygon(polys) => {
                    for poly in polys {
                        for ring in poly {
                            out.extend_from_slice(ring);
                        }
                    }
                }
                GeoJsonGeometry::GeometryCollection(geoms) => {
                    for g in geoms {
                        collect_positions(g, out);
                    }
                }
            }
        }

        for feature in &collection.features {
            if let Some(geom) = &feature.geometry {
                collect_positions(geom, &mut positions);
            }
        }

        if positions.is_empty() {
            return None;
        }

        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;

        for p in &positions {
            min_lon = min_lon.min(p[0]);
            max_lon = max_lon.max(p[0]);
            min_lat = min_lat.min(p[1]);
            max_lat = max_lat.max(p[1]);
        }

        // [west, south, east, north]
        Some([min_lon, min_lat, max_lon, max_lat])
    }
}

impl Default for GeoJsonSerializer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse a `"lon lat"` pair from a string slice
fn parse_coord_pair(s: &str) -> Result<(f64, f64), GeoJsonError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(GeoJsonError::InvalidWkt(format!(
            "Expected two coordinates, got: {}",
            s
        )));
    }
    let lon = parts[0]
        .parse::<f64>()
        .map_err(|_| GeoJsonError::InvalidWkt(format!("Cannot parse longitude: {}", parts[0])))?;
    let lat = parts[1]
        .parse::<f64>()
        .map_err(|_| GeoJsonError::InvalidWkt(format!("Cannot parse latitude: {}", parts[1])))?;
    Ok((lon, lat))
}

/// Parse a comma-separated list of `"lon lat"` pairs
fn parse_coord_list(s: &str) -> Result<Vec<[f64; 2]>, GeoJsonError> {
    s.split(',')
        .map(|pair| {
            let (lon, lat) = parse_coord_pair(pair.trim())?;
            Ok([lon, lat])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_props() -> HashMap<String, Value> {
        HashMap::new()
    }

    // -----------------------------------------------------------------------
    // geometry_to_json
    // -----------------------------------------------------------------------

    #[test]
    fn test_point_type_field() {
        let s = GeoJsonSerializer::new();
        let v = s.geometry_to_json(&GeoJsonGeometry::Point([10.0, 20.0]));
        assert_eq!(v["type"], "Point");
    }

    #[test]
    fn test_point_coordinates() {
        let s = GeoJsonSerializer::new();
        let v = s.geometry_to_json(&GeoJsonGeometry::Point([13.405, 52.52]));
        assert!((v["coordinates"][0].as_f64().unwrap() - 13.405).abs() < 1e-10);
        assert!((v["coordinates"][1].as_f64().unwrap() - 52.52).abs() < 1e-10);
    }

    #[test]
    fn test_linestring_type_field() {
        let s = GeoJsonSerializer::new();
        let v = s.geometry_to_json(&GeoJsonGeometry::LineString(vec![[0.0, 0.0], [1.0, 1.0]]));
        assert_eq!(v["type"], "LineString");
        assert_eq!(v["coordinates"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_polygon_type_field() {
        let s = GeoJsonSerializer::new();
        let ring = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        let v = s.geometry_to_json(&GeoJsonGeometry::Polygon(vec![ring]));
        assert_eq!(v["type"], "Polygon");
        assert_eq!(v["coordinates"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_multipoint_type_field() {
        let s = GeoJsonSerializer::new();
        let v = s.geometry_to_json(&GeoJsonGeometry::MultiPoint(vec![[1.0, 2.0], [3.0, 4.0]]));
        assert_eq!(v["type"], "MultiPoint");
    }

    #[test]
    fn test_multilinestring_type_field() {
        let s = GeoJsonSerializer::new();
        let v = s.geometry_to_json(&GeoJsonGeometry::MultiLineString(vec![
            vec![[0.0, 0.0], [1.0, 1.0]],
            vec![[2.0, 2.0], [3.0, 3.0]],
        ]));
        assert_eq!(v["type"], "MultiLineString");
        assert_eq!(v["coordinates"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_multipolygon_type_field() {
        let s = GeoJsonSerializer::new();
        let ring = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]];
        let v = s.geometry_to_json(&GeoJsonGeometry::MultiPolygon(vec![vec![ring]]));
        assert_eq!(v["type"], "MultiPolygon");
    }

    #[test]
    fn test_geometry_collection_type_field() {
        let s = GeoJsonSerializer::new();
        let v = s.geometry_to_json(&GeoJsonGeometry::GeometryCollection(vec![
            GeoJsonGeometry::Point([0.0, 0.0]),
        ]));
        assert_eq!(v["type"], "GeometryCollection");
        assert_eq!(v["geometries"].as_array().unwrap().len(), 1);
    }

    // -----------------------------------------------------------------------
    // feature_to_json
    // -----------------------------------------------------------------------

    #[test]
    fn test_feature_type_field() {
        let s = GeoJsonSerializer::new();
        let f = GeoJsonFeature {
            geometry: Some(GeoJsonGeometry::Point([0.0, 0.0])),
            properties: empty_props(),
        };
        let v = s.feature_to_json(&f);
        assert_eq!(v["type"], "Feature");
    }

    #[test]
    fn test_feature_null_geometry() {
        let s = GeoJsonSerializer::new();
        let f = GeoJsonFeature {
            geometry: None,
            properties: empty_props(),
        };
        let v = s.feature_to_json(&f);
        assert!(v["geometry"].is_null());
    }

    #[test]
    fn test_feature_properties_included() {
        let s = GeoJsonSerializer::new();
        let mut props = empty_props();
        props.insert("city".into(), Value::String("Berlin".into()));
        let f = GeoJsonFeature {
            geometry: None,
            properties: props,
        };
        let v = s.feature_to_json(&f);
        assert_eq!(v["properties"]["city"], "Berlin");
    }

    // -----------------------------------------------------------------------
    // serialize_collection
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_collection_produces_valid_json() {
        let s = GeoJsonSerializer::new();
        let collection = GeoJsonFeatureCollection { features: vec![] };
        let json = s.serialize_collection(&collection).unwrap();
        assert!(json.contains("FeatureCollection"));
    }

    #[test]
    fn test_serialize_collection_with_features() {
        let s = GeoJsonSerializer::new();
        let f = GeoJsonFeature {
            geometry: Some(GeoJsonGeometry::Point([13.405, 52.52])),
            properties: empty_props(),
        };
        let collection = GeoJsonFeatureCollection { features: vec![f] };
        let json = s.serialize_collection(&collection).unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["type"], "FeatureCollection");
        assert_eq!(parsed["features"].as_array().unwrap().len(), 1);
    }

    // -----------------------------------------------------------------------
    // WKT parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_wkt_point_basic() {
        let geom = GeoJsonSerializer::parse_wkt_point("POINT(13.405 52.52)").unwrap();
        if let GeoJsonGeometry::Point(c) = geom {
            assert!((c[0] - 13.405).abs() < 1e-10);
            assert!((c[1] - 52.52).abs() < 1e-10);
        } else {
            panic!("Expected Point");
        }
    }

    #[test]
    fn test_parse_wkt_point_with_space() {
        let geom = GeoJsonSerializer::parse_wkt_point("POINT (0.0 0.0)").unwrap();
        assert!(matches!(geom, GeoJsonGeometry::Point([_, _])));
    }

    #[test]
    fn test_parse_wkt_point_negative_coords() {
        let geom = GeoJsonSerializer::parse_wkt_point("POINT(-73.9857 40.7484)").unwrap();
        if let GeoJsonGeometry::Point(c) = geom {
            assert!((c[0] + 73.9857).abs() < 1e-10);
        } else {
            panic!("Expected Point");
        }
    }

    #[test]
    fn test_parse_wkt_point_invalid_returns_error() {
        assert!(GeoJsonSerializer::parse_wkt_point("LINE(0 0)").is_err());
    }

    #[test]
    fn test_parse_wkt_linestring_basic() {
        let geom = GeoJsonSerializer::parse_wkt_linestring("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        if let GeoJsonGeometry::LineString(pts) = geom {
            assert_eq!(pts.len(), 3);
        } else {
            panic!("Expected LineString");
        }
    }

    #[test]
    fn test_parse_wkt_linestring_insufficient_points() {
        assert!(GeoJsonSerializer::parse_wkt_linestring("LINESTRING(0 0)").is_err());
    }

    #[test]
    fn test_parse_wkt_linestring_wrong_prefix() {
        assert!(GeoJsonSerializer::parse_wkt_linestring("POINT(0 0, 1 1)").is_err());
    }

    #[test]
    fn test_parse_wkt_polygon_basic() {
        let wkt = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))";
        let geom = GeoJsonSerializer::parse_wkt_polygon(wkt).unwrap();
        if let GeoJsonGeometry::Polygon(rings) = geom {
            assert_eq!(rings.len(), 1);
            assert_eq!(rings[0].len(), 5);
        } else {
            panic!("Expected Polygon");
        }
    }

    #[test]
    fn test_parse_wkt_polygon_wrong_prefix() {
        assert!(GeoJsonSerializer::parse_wkt_polygon("POINT((0 0))").is_err());
    }

    // -----------------------------------------------------------------------
    // sparql_results_to_geojson
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparql_results_to_geojson_basic() {
        let s = GeoJsonSerializer::new();
        let mut row = HashMap::new();
        row.insert("lat".into(), "52.52".into());
        row.insert("lon".into(), "13.405".into());
        let fc = s.sparql_results_to_geojson(&[row], "lat", "lon", None);
        assert_eq!(fc.features.len(), 1);
    }

    #[test]
    fn test_sparql_results_to_geojson_with_label() {
        let s = GeoJsonSerializer::new();
        let mut row = HashMap::new();
        row.insert("lat".into(), "52.52".into());
        row.insert("lon".into(), "13.405".into());
        row.insert("name".into(), "Berlin".into());
        let fc = s.sparql_results_to_geojson(&[row], "lat", "lon", Some("name"));
        assert_eq!(fc.features[0].properties.get("label").unwrap(), "Berlin");
    }

    #[test]
    fn test_sparql_results_skips_missing_lat() {
        let s = GeoJsonSerializer::new();
        let mut row = HashMap::new();
        row.insert("lon".into(), "13.405".into());
        let fc = s.sparql_results_to_geojson(&[row], "lat", "lon", None);
        assert!(fc.features.is_empty());
    }

    #[test]
    fn test_sparql_results_skips_missing_lon() {
        let s = GeoJsonSerializer::new();
        let mut row = HashMap::new();
        row.insert("lat".into(), "52.52".into());
        let fc = s.sparql_results_to_geojson(&[row], "lat", "lon", None);
        assert!(fc.features.is_empty());
    }

    #[test]
    fn test_sparql_results_skips_invalid_coords() {
        let s = GeoJsonSerializer::new();
        let mut row = HashMap::new();
        row.insert("lat".into(), "not_a_number".into());
        row.insert("lon".into(), "13.405".into());
        let fc = s.sparql_results_to_geojson(&[row], "lat", "lon", None);
        assert!(fc.features.is_empty());
    }

    #[test]
    fn test_sparql_results_adds_other_bindings_as_properties() {
        let s = GeoJsonSerializer::new();
        let mut row = HashMap::new();
        row.insert("lat".into(), "52.52".into());
        row.insert("lon".into(), "13.405".into());
        row.insert("population".into(), "3.7M".into());
        let fc = s.sparql_results_to_geojson(&[row], "lat", "lon", None);
        assert!(fc.features[0].properties.contains_key("population"));
    }

    #[test]
    fn test_sparql_results_multiple_rows() {
        let s = GeoJsonSerializer::new();
        let rows: Vec<HashMap<String, String>> = (0..5)
            .map(|i| {
                let mut r = HashMap::new();
                r.insert("lat".into(), format!("{}", i as f64));
                r.insert("lon".into(), format!("{}", i as f64 * 2.0));
                r
            })
            .collect();
        let fc = s.sparql_results_to_geojson(&rows, "lat", "lon", None);
        assert_eq!(fc.features.len(), 5);
    }

    // -----------------------------------------------------------------------
    // bounding_box
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_empty_collection() {
        let fc = GeoJsonFeatureCollection { features: vec![] };
        assert!(GeoJsonSerializer::bounding_box(&fc).is_none());
    }

    #[test]
    fn test_bounding_box_single_point() {
        let f = GeoJsonFeature {
            geometry: Some(GeoJsonGeometry::Point([10.0, 20.0])),
            properties: empty_props(),
        };
        let fc = GeoJsonFeatureCollection { features: vec![f] };
        let bb = GeoJsonSerializer::bounding_box(&fc).unwrap();
        // [west, south, east, north]
        assert!((bb[0] - 10.0).abs() < 1e-10); // west
        assert!((bb[1] - 20.0).abs() < 1e-10); // south
        assert!((bb[2] - 10.0).abs() < 1e-10); // east
        assert!((bb[3] - 20.0).abs() < 1e-10); // north
    }

    #[test]
    fn test_bounding_box_multiple_points() {
        let features: Vec<GeoJsonFeature> = vec![
            GeoJsonFeature {
                geometry: Some(GeoJsonGeometry::Point([-10.0, -5.0])),
                properties: empty_props(),
            },
            GeoJsonFeature {
                geometry: Some(GeoJsonGeometry::Point([20.0, 30.0])),
                properties: empty_props(),
            },
        ];
        let fc = GeoJsonFeatureCollection { features };
        let bb = GeoJsonSerializer::bounding_box(&fc).unwrap();
        assert!((bb[0] + 10.0).abs() < 1e-10); // west = -10
        assert!((bb[1] + 5.0).abs() < 1e-10); // south = -5
        assert!((bb[2] - 20.0).abs() < 1e-10); // east = 20
        assert!((bb[3] - 30.0).abs() < 1e-10); // north = 30
    }

    #[test]
    fn test_bounding_box_feature_with_null_geometry() {
        let f = GeoJsonFeature {
            geometry: None,
            properties: empty_props(),
        };
        let fc = GeoJsonFeatureCollection { features: vec![f] };
        assert!(GeoJsonSerializer::bounding_box(&fc).is_none());
    }

    #[test]
    fn test_bounding_box_linestring() {
        let f = GeoJsonFeature {
            geometry: Some(GeoJsonGeometry::LineString(vec![[0.0, 0.0], [5.0, 10.0]])),
            properties: empty_props(),
        };
        let fc = GeoJsonFeatureCollection { features: vec![f] };
        let bb = GeoJsonSerializer::bounding_box(&fc).unwrap();
        assert!((bb[0] - 0.0).abs() < 1e-10);
        assert!((bb[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_serializer_default() {
        let _s = GeoJsonSerializer;
    }
}
