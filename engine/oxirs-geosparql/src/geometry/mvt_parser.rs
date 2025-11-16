//! MVT (Mapbox Vector Tiles) parser and serializer
//!
//! This module provides support for the Mapbox Vector Tiles format, which is
//! a binary format for efficiently encoding vector geometries for use in web maps.
//!
//! # Features
//!
//! - Encode geometries as MVT tiles
//! - Support for tile coordinates (z/x/y)
//! - Multiple layers per tile
//! - Feature properties support
//! - Efficient binary Protocol Buffers encoding
//!
//! # MVT Tile Coordinate System
//!
//! MVT uses a tile coordinate system where:
//! - `z` (zoom level): 0-22 typically
//! - `x` (tile column): 0 to 2^z - 1
//! - `y` (tile row): 0 to 2^z - 1
//! - Tile extent: typically 4096x4096 units
//!
//! # Examples
//!
//! ```rust
//! use oxirs_geosparql::geometry::{Geometry, mvt_parser::MvtTile};
//! use geo_types::{Point, Geometry as GeoGeometry};
//!
//! // Create a simple MVT tile
//! let mut tile = MvtTile::new(10, 511, 383); // z=10, x=511, y=383
//!
//! // Add a point geometry to the "places" layer
//! let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4, 37.8)));
//! tile.add_feature("places", point, None).unwrap();
//!
//! // Encode to MVT bytes
//! let mvt_bytes = tile.encode().unwrap();
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo_types::Geometry as GeoGeometry;
use std::collections::HashMap;

/// Mapbox Vector Tile representation
///
/// Represents a single MVT tile at a specific zoom level and tile coordinates.
/// A tile can contain multiple layers, each with multiple features.
#[derive(Debug, Clone)]
pub struct MvtTile {
    /// Zoom level (0-22 typically)
    pub zoom: u8,
    /// Tile X coordinate
    pub x: u32,
    /// Tile Y coordinate
    pub y: u32,
    /// Tile extent (default: 4096)
    pub extent: u32,
    /// Layers in this tile
    pub layers: Vec<MvtLayer>,
}

/// MVT Layer representation
///
/// A layer contains features of a specific type (e.g., "roads", "buildings", "water").
#[derive(Debug, Clone)]
pub struct MvtLayer {
    /// Layer name
    pub name: String,
    /// Layer version (default: 2 for MVT 2.x)
    pub version: u32,
    /// Extent (usually same as tile extent)
    pub extent: u32,
    /// Features in this layer
    pub features: Vec<MvtFeature>,
}

/// MVT Feature representation
///
/// A feature is a single geometry with optional properties.
#[derive(Debug, Clone)]
pub struct MvtFeature {
    /// Feature ID (optional)
    pub id: Option<u64>,
    /// Geometry
    pub geometry: Geometry,
    /// Feature properties (tags)
    pub properties: HashMap<String, MvtValue>,
}

/// MVT property value types
#[derive(Debug, Clone)]
pub enum MvtValue {
    /// String value
    String(String),
    /// Float value
    Float(f64),
    /// Double value
    Double(f64),
    /// Integer value
    Int(i64),
    /// Unsigned integer value
    UInt(u64),
    /// Signed integer value
    SInt(i64),
    /// Boolean value
    Bool(bool),
}

impl MvtTile {
    /// Create a new MVT tile
    ///
    /// # Arguments
    ///
    /// * `zoom` - Zoom level (0-22)
    /// * `x` - Tile X coordinate
    /// * `y` - Tile Y coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::geometry::mvt_parser::MvtTile;
    ///
    /// let tile = MvtTile::new(10, 511, 383);
    /// assert_eq!(tile.zoom, 10);
    /// assert_eq!(tile.extent, 4096);
    /// ```
    pub fn new(zoom: u8, x: u32, y: u32) -> Self {
        Self {
            zoom,
            x,
            y,
            extent: 4096, // Standard MVT extent
            layers: Vec::new(),
        }
    }

    /// Create a new MVT tile with custom extent
    ///
    /// # Arguments
    ///
    /// * `zoom` - Zoom level
    /// * `x` - Tile X coordinate
    /// * `y` - Tile Y coordinate
    /// * `extent` - Tile extent (typically 4096)
    pub fn new_with_extent(zoom: u8, x: u32, y: u32, extent: u32) -> Self {
        Self {
            zoom,
            x,
            y,
            extent,
            layers: Vec::new(),
        }
    }

    /// Add a feature to a layer
    ///
    /// If the layer doesn't exist, it will be created.
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer
    /// * `geometry` - Geometry to add
    /// * `properties` - Optional feature properties
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::geometry::{Geometry, mvt_parser::MvtTile};
    /// use geo_types::{Point, Geometry as GeoGeometry};
    /// use std::collections::HashMap;
    ///
    /// let mut tile = MvtTile::new(10, 511, 383);
    /// let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4, 37.8)));
    ///
    /// let mut props = HashMap::new();
    /// props.insert("name".to_string(), "San Francisco".to_string());
    ///
    /// tile.add_feature("cities", point, Some(props)).unwrap();
    /// ```
    pub fn add_feature(
        &mut self,
        layer_name: &str,
        geometry: Geometry,
        properties: Option<HashMap<String, String>>,
    ) -> Result<()> {
        // Find or create layer
        let layer = self
            .layers
            .iter_mut()
            .find(|l| l.name == layer_name)
            .map(|l| l as *mut MvtLayer);

        let layer = match layer {
            Some(l) => unsafe { &mut *l },
            None => {
                self.layers.push(MvtLayer {
                    name: layer_name.to_string(),
                    version: 2,
                    extent: self.extent,
                    features: Vec::new(),
                });
                self.layers.last_mut().unwrap()
            }
        };

        // Convert properties to MvtValue
        let mvt_properties = properties
            .map(|props| {
                props
                    .into_iter()
                    .map(|(k, v)| (k, MvtValue::String(v)))
                    .collect()
            })
            .unwrap_or_default();

        layer.features.push(MvtFeature {
            id: None,
            geometry,
            properties: mvt_properties,
        });

        Ok(())
    }

    /// Encode the tile to MVT bytes
    ///
    /// Returns the Protocol Buffers encoded MVT data.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::geometry::{Geometry, mvt_parser::MvtTile};
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let mut tile = MvtTile::new(10, 511, 383);
    /// let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4, 37.8)));
    /// tile.add_feature("places", point, None).unwrap();
    ///
    /// let bytes = tile.encode().unwrap();
    /// assert!(!bytes.is_empty());
    /// ```
    pub fn encode(&self) -> Result<Vec<u8>> {
        use mvt::Tile;

        let mut mvt_tile = Tile::new(self.extent);

        for layer in &self.layers {
            let mut mvt_layer = mvt_tile.create_layer(&layer.name);

            for feature in &layer.features {
                // Convert geometry to MVT geometry
                let geom_data = self.geometry_to_mvt(&feature.geometry)?;

                // Create feature
                let mut mvt_feature = mvt_layer.into_feature(geom_data);

                // Add properties
                for (key, value) in &feature.properties {
                    match value {
                        MvtValue::String(s) => mvt_feature.add_tag_string(key, s),
                        MvtValue::Int(i) => mvt_feature.add_tag_sint(key, *i),
                        MvtValue::UInt(u) => mvt_feature.add_tag_uint(key, *u),
                        MvtValue::Double(d) => mvt_feature.add_tag_double(key, *d),
                        MvtValue::Bool(b) => mvt_feature.add_tag_bool(key, *b),
                        _ => {}
                    }
                }

                if let Some(id) = feature.id {
                    mvt_feature.set_id(id);
                }

                mvt_layer = mvt_feature.into_layer();
            }

            mvt_tile.add_layer(mvt_layer).map_err(|e| {
                GeoSparqlError::SerializationError(format!("MVT layer error: {}", e))
            })?;
        }

        mvt_tile
            .to_bytes()
            .map_err(|e| GeoSparqlError::SerializationError(format!("MVT encoding error: {}", e)))
    }

    /// Convert a geometry to MVT GeomData
    fn geometry_to_mvt(&self, geometry: &Geometry) -> Result<mvt::GeomData> {
        use mvt::{GeomEncoder, GeomType};

        // Convert lat/lon to tile coordinates
        let tile_coords = self.latlon_to_tile_coords(geometry)?;

        match &geometry.geom {
            GeoGeometry::Point(_p) => {
                let (x, y) = tile_coords[0];
                let geom_data = GeomEncoder::new(GeomType::Point)
                    .point(x as f64, y as f64)
                    .map_err(|e| {
                        GeoSparqlError::SerializationError(format!("Point encoding error: {}", e))
                    })?
                    .encode()
                    .map_err(|e| {
                        GeoSparqlError::SerializationError(format!(
                            "Geometry encoding error: {}",
                            e
                        ))
                    })?;
                Ok(geom_data)
            }
            GeoGeometry::LineString(_ls) => {
                let mut encoder = GeomEncoder::new(GeomType::Linestring);
                for (x, y) in tile_coords {
                    encoder = encoder.point(x as f64, y as f64).map_err(|e| {
                        GeoSparqlError::SerializationError(format!("Point encoding error: {}", e))
                    })?;
                }
                let geom_data = encoder.encode().map_err(|e| {
                    GeoSparqlError::SerializationError(format!("Geometry encoding error: {}", e))
                })?;
                Ok(geom_data)
            }
            GeoGeometry::Polygon(_poly) => {
                let mut encoder = GeomEncoder::new(GeomType::Polygon);
                for (x, y) in tile_coords {
                    encoder = encoder.point(x as f64, y as f64).map_err(|e| {
                        GeoSparqlError::SerializationError(format!("Point encoding error: {}", e))
                    })?;
                }
                // Complete the polygon ring
                encoder.complete_geom().map_err(|e| {
                    GeoSparqlError::SerializationError(format!("Polygon completion error: {}", e))
                })?;

                let geom_data = encoder.encode().map_err(|e| {
                    GeoSparqlError::SerializationError(format!("Geometry encoding error: {}", e))
                })?;
                Ok(geom_data)
            }
            _ => Err(GeoSparqlError::UnsupportedOperation(
                "Geometry type not yet supported for MVT encoding".to_string(),
            )),
        }
    }

    /// Convert lat/lon coordinates to tile pixel coordinates
    fn latlon_to_tile_coords(&self, geometry: &Geometry) -> Result<Vec<(u32, u32)>> {
        let mut coords = Vec::new();

        match &geometry.geom {
            GeoGeometry::Point(p) => {
                let (x, y) = self.latlon_to_pixel(p.y(), p.x());
                coords.push((x, y));
            }
            GeoGeometry::LineString(ls) => {
                for coord in ls.coords() {
                    let (x, y) = self.latlon_to_pixel(coord.y, coord.x);
                    coords.push((x, y));
                }
            }
            GeoGeometry::Polygon(poly) => {
                for coord in poly.exterior().coords() {
                    let (x, y) = self.latlon_to_pixel(coord.y, coord.x);
                    coords.push((x, y));
                }
            }
            _ => {
                return Err(GeoSparqlError::UnsupportedOperation(
                    "Geometry type not supported".to_string(),
                ))
            }
        }

        Ok(coords)
    }

    /// Convert lat/lon to tile pixel coordinates
    ///
    /// Uses Web Mercator projection (EPSG:3857)
    fn latlon_to_pixel(&self, lat: f64, lon: f64) -> (u32, u32) {
        let tile_count = 2_u32.pow(self.zoom as u32);

        // Convert to normalized tile coordinates (0-1)
        let x_norm = (lon + 180.0) / 360.0;
        let y_norm = (1.0
            - (lat.to_radians().tan() + 1.0 / lat.to_radians().cos()).ln() / std::f64::consts::PI)
            / 2.0;

        // Get tile-relative coordinates
        let x_tile = x_norm * tile_count as f64 - self.x as f64;
        let y_tile = y_norm * tile_count as f64 - self.y as f64;

        // Convert to pixel coordinates within the tile
        let x_pixel = (x_tile * self.extent as f64).round() as u32;
        let y_pixel = (y_tile * self.extent as f64).round() as u32;

        (x_pixel, y_pixel)
    }

    /// Get tile bounds in lat/lon
    ///
    /// Returns (min_lon, min_lat, max_lon, max_lat)
    pub fn get_bounds(&self) -> (f64, f64, f64, f64) {
        let n = 2_f64.powi(self.zoom as i32);

        let lon_min = (self.x as f64) / n * 360.0 - 180.0;
        let lon_max = ((self.x + 1) as f64) / n * 360.0 - 180.0;

        let lat_min = ((std::f64::consts::PI * (1.0 - 2.0 * ((self.y + 1) as f64) / n)).sinh())
            .atan()
            .to_degrees();
        let lat_max = ((std::f64::consts::PI * (1.0 - 2.0 * (self.y as f64) / n)).sinh())
            .atan()
            .to_degrees();

        (lon_min, lat_min, lon_max, lat_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Point, Polygon};

    #[test]
    fn test_mvt_tile_creation() {
        let tile = MvtTile::new(10, 511, 383);
        assert_eq!(tile.zoom, 10);
        assert_eq!(tile.x, 511);
        assert_eq!(tile.y, 383);
        assert_eq!(tile.extent, 4096);
        assert_eq!(tile.layers.len(), 0);
    }

    #[test]
    fn test_mvt_add_point_feature() {
        let mut tile = MvtTile::new(10, 511, 383);
        let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4, 37.8)));

        tile.add_feature("places", point, None).unwrap();

        assert_eq!(tile.layers.len(), 1);
        assert_eq!(tile.layers[0].name, "places");
        assert_eq!(tile.layers[0].features.len(), 1);
    }

    #[test]
    fn test_mvt_add_feature_with_properties() {
        let mut tile = MvtTile::new(10, 511, 383);
        let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4, 37.8)));

        let mut props = HashMap::new();
        props.insert("name".to_string(), "San Francisco".to_string());
        props.insert("population".to_string(), "870000".to_string());

        tile.add_feature("cities", point, Some(props)).unwrap();

        assert_eq!(tile.layers[0].features[0].properties.len(), 2);
    }

    #[test]
    fn test_mvt_encode_point() {
        let mut tile = MvtTile::new(10, 511, 383);
        let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4, 37.8)));

        tile.add_feature("places", point, None).unwrap();

        let bytes = tile.encode().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_mvt_encode_linestring() {
        let mut tile = MvtTile::new(10, 511, 383);
        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: -122.4, y: 37.8 },
            Coord { x: -122.5, y: 37.9 },
        ])));

        tile.add_feature("roads", line, None).unwrap();

        let bytes = tile.encode().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_mvt_encode_polygon() {
        let mut tile = MvtTile::new(10, 511, 383);
        let polygon = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: -122.4, y: 37.8 },
                Coord { x: -122.5, y: 37.8 },
                Coord { x: -122.5, y: 37.9 },
                Coord { x: -122.4, y: 37.9 },
                Coord { x: -122.4, y: 37.8 },
            ]),
            vec![],
        )));

        tile.add_feature("buildings", polygon, None).unwrap();

        let bytes = tile.encode().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_mvt_multiple_layers() {
        let mut tile = MvtTile::new(10, 511, 383);

        let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4, 37.8)));
        tile.add_feature("places", point, None).unwrap();

        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: -122.4, y: 37.8 },
            Coord { x: -122.5, y: 37.9 },
        ])));
        tile.add_feature("roads", line, None).unwrap();

        assert_eq!(tile.layers.len(), 2);

        let bytes = tile.encode().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_mvt_tile_bounds() {
        let tile = MvtTile::new(10, 511, 383);
        let (min_lon, min_lat, max_lon, max_lat) = tile.get_bounds();

        // Check that bounds are reasonable for San Francisco area
        assert!(min_lon < max_lon);
        assert!(min_lat < max_lat);
        assert!(min_lon >= -180.0 && max_lon <= 180.0);
        assert!(min_lat >= -85.0 && max_lat <= 85.0);
    }
}
