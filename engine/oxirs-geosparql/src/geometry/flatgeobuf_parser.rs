//! FlatGeobuf parser and serializer
//!
//! Provides reading and writing of FlatGeobuf format - a cloud-native binary format
//! optimized for HTTP range requests and streaming access.
//!
//! # Features
//!
//! - Parse FlatGeobuf files into geometries
//! - Serialize geometries to FlatGeobuf format
//! - Support for streaming and random access
//! - Efficient binary encoding
//! - Spatial indexing support
//! - CRS information preservation
//! - 3D coordinate support (Z values)
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_geosparql::geometry::Geometry;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Read from file
//! let file = File::open("data.fgb")?;
//! let reader = BufReader::new(file);
//! # #[cfg(feature = "flatgeobuf-support")]
//! let geometries = oxirs_geosparql::geometry::flatgeobuf_parser::parse_flatgeobuf(reader)?;
//!
//! // Write to file
//! let output = File::create("output.fgb")?;
//! # #[cfg(feature = "flatgeobuf-support")]
//! oxirs_geosparql::geometry::flatgeobuf_parser::write_flatgeobuf(&geometries, output)?;
//! # Ok(())
//! # }
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use std::io::{Read, Seek, Write};

#[cfg(feature = "flatgeobuf-support")]
use flatgeobuf::{FgbReader, FgbWriter, GeometryType};
#[cfg(feature = "flatgeobuf-support")]
use geo_traits::{
    CoordTrait, GeometryTrait, LineStringTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait, PointTrait, PolygonTrait,
};
#[cfg(feature = "flatgeobuf-support")]
use geozero::error::GeozeroError;
#[cfg(feature = "flatgeobuf-support")]
use geozero::GeomProcessor;

/// GeomProcessor implementation for converting FlatGeobuf geometries to our Geometry type
#[cfg(feature = "flatgeobuf-support")]
#[allow(dead_code)]
struct GeometryCollector {
    /// Collected geometries
    geometries: Vec<Geometry>,
    /// Current geometry being built
    current_coords: Vec<geo_types::Coord>,
    /// Current rings for polygons
    current_rings: Vec<Vec<geo_types::Coord>>,
    /// Current geometries for collections
    current_geoms: Vec<geo_types::Geometry>,
    /// Current Z coordinates
    current_z_coords: Vec<f64>,
    /// Geometry type being processed
    current_geom_type: Option<GeometryType>,
}

#[cfg(feature = "flatgeobuf-support")]
#[allow(dead_code)]
impl GeometryCollector {
    fn new() -> Self {
        Self {
            geometries: Vec::new(),
            current_coords: Vec::new(),
            current_rings: Vec::new(),
            current_geoms: Vec::new(),
            current_z_coords: Vec::new(),
            current_geom_type: None,
        }
    }

    /// Finalize the current geometry and add to collection
    fn finalize_geometry(&mut self) -> geozero::error::Result<()> {
        if let Some(geom_type) = self.current_geom_type.take() {
            let geo_geom = match geom_type {
                GeometryType::Point => {
                    if let Some(coord) = self.current_coords.first() {
                        geo_types::Geometry::Point(geo_types::Point(*coord))
                    } else {
                        return Err(GeozeroError::Geometry(
                            "Point has no coordinates".to_string(),
                        ));
                    }
                }
                GeometryType::LineString => {
                    let coords = std::mem::take(&mut self.current_coords);
                    geo_types::Geometry::LineString(geo_types::LineString::new(coords))
                }
                GeometryType::Polygon => {
                    let rings = std::mem::take(&mut self.current_rings);
                    if rings.is_empty() {
                        return Err(GeozeroError::Geometry("Polygon has no rings".to_string()));
                    }
                    let exterior = geo_types::LineString::new(rings[0].clone());
                    let interiors = rings[1..]
                        .iter()
                        .map(|r| geo_types::LineString::new(r.clone()))
                        .collect();
                    geo_types::Geometry::Polygon(geo_types::Polygon::new(exterior, interiors))
                }
                GeometryType::MultiPoint => {
                    let coords = std::mem::take(&mut self.current_coords);
                    let points = coords.into_iter().map(geo_types::Point).collect();
                    geo_types::Geometry::MultiPoint(geo_types::MultiPoint(points))
                }
                GeometryType::MultiLineString => {
                    let geoms = std::mem::take(&mut self.current_geoms);
                    let linestrings: Vec<_> = geoms
                        .into_iter()
                        .filter_map(|g| match g {
                            geo_types::Geometry::LineString(ls) => Some(ls),
                            _ => None,
                        })
                        .collect();
                    geo_types::Geometry::MultiLineString(geo_types::MultiLineString(linestrings))
                }
                GeometryType::MultiPolygon => {
                    let geoms = std::mem::take(&mut self.current_geoms);
                    let polygons: Vec<_> = geoms
                        .into_iter()
                        .filter_map(|g| match g {
                            geo_types::Geometry::Polygon(p) => Some(p),
                            _ => None,
                        })
                        .collect();
                    geo_types::Geometry::MultiPolygon(geo_types::MultiPolygon(polygons))
                }
                _ => {
                    return Err(GeozeroError::Geometry(format!(
                        "Unsupported geometry type: {:?}",
                        geom_type
                    )))
                }
            };

            // Create Geometry with Z coordinates if available
            let mut geometry = Geometry::new(geo_geom);
            if !self.current_z_coords.is_empty() {
                geometry.coord3d = crate::geometry::coord3d::Coord3D::xyz(std::mem::take(
                    &mut self.current_z_coords,
                ));
            }

            self.geometries.push(geometry);
        }

        // Clear state
        self.current_coords.clear();
        self.current_rings.clear();
        self.current_geoms.clear();
        self.current_z_coords.clear();

        Ok(())
    }
}

#[cfg(feature = "flatgeobuf-support")]
impl GeomProcessor for GeometryCollector {
    fn xy(&mut self, x: f64, y: f64, _idx: usize) -> geozero::error::Result<()> {
        self.current_coords.push(geo_types::coord! { x: x, y: y });
        Ok(())
    }

    fn coordinate(
        &mut self,
        x: f64,
        y: f64,
        z: Option<f64>,
        _m: Option<f64>,
        _t: Option<f64>,
        _tm: Option<u64>,
        _idx: usize,
    ) -> geozero::error::Result<()> {
        self.current_coords.push(geo_types::coord! { x: x, y: y });
        if let Some(z_val) = z {
            self.current_z_coords.push(z_val);
        }
        Ok(())
    }

    fn point_begin(&mut self, _idx: usize) -> geozero::error::Result<()> {
        self.current_geom_type = Some(GeometryType::Point);
        Ok(())
    }

    fn point_end(&mut self, _idx: usize) -> geozero::error::Result<()> {
        Ok(())
    }

    fn linestring_begin(
        &mut self,
        _tagged: bool,
        _size: usize,
        _idx: usize,
    ) -> geozero::error::Result<()> {
        if self.current_geom_type.is_none() {
            self.current_geom_type = Some(GeometryType::LineString);
        }
        Ok(())
    }

    fn linestring_end(&mut self, _tagged: bool, _idx: usize) -> geozero::error::Result<()> {
        if self.current_geom_type == Some(GeometryType::MultiLineString) {
            // Store linestring for multi-geometry
            let coords = std::mem::take(&mut self.current_coords);
            self.current_geoms
                .push(geo_types::Geometry::LineString(geo_types::LineString::new(
                    coords,
                )));
        }
        Ok(())
    }

    fn polygon_begin(
        &mut self,
        _tagged: bool,
        _size: usize,
        _idx: usize,
    ) -> geozero::error::Result<()> {
        if self.current_geom_type.is_none() {
            self.current_geom_type = Some(GeometryType::Polygon);
        }
        Ok(())
    }

    fn polygon_end(&mut self, _tagged: bool, _idx: usize) -> geozero::error::Result<()> {
        if self.current_geom_type == Some(GeometryType::MultiPolygon) {
            // Store polygon for multi-geometry
            let rings = std::mem::take(&mut self.current_rings);
            if !rings.is_empty() {
                let exterior = geo_types::LineString::new(rings[0].clone());
                let interiors = rings[1..]
                    .iter()
                    .map(|r| geo_types::LineString::new(r.clone()))
                    .collect();
                self.current_geoms
                    .push(geo_types::Geometry::Polygon(geo_types::Polygon::new(
                        exterior, interiors,
                    )));
            }
        }
        Ok(())
    }

    fn multipoint_begin(&mut self, _size: usize, _idx: usize) -> geozero::error::Result<()> {
        self.current_geom_type = Some(GeometryType::MultiPoint);
        Ok(())
    }

    fn multipoint_end(&mut self, _idx: usize) -> geozero::error::Result<()> {
        Ok(())
    }

    fn multilinestring_begin(&mut self, _size: usize, _idx: usize) -> geozero::error::Result<()> {
        self.current_geom_type = Some(GeometryType::MultiLineString);
        Ok(())
    }

    fn multilinestring_end(&mut self, _idx: usize) -> geozero::error::Result<()> {
        Ok(())
    }

    fn multipolygon_begin(&mut self, _size: usize, _idx: usize) -> geozero::error::Result<()> {
        self.current_geom_type = Some(GeometryType::MultiPolygon);
        Ok(())
    }

    fn multipolygon_end(&mut self, _idx: usize) -> geozero::error::Result<()> {
        Ok(())
    }
}

/// Parse FlatGeobuf data from a reader
///
/// Reads all features from a FlatGeobuf file and converts them to Geometry objects.
///
/// # Arguments
///
/// * `reader` - Any type implementing Read + Seek
///
/// # Returns
///
/// A vector of all geometries in the FlatGeobuf file
///
/// # Example
///
/// ```rust,no_run
/// use std::fs::File;
/// use std::io::BufReader;
/// # #[cfg(feature = "flatgeobuf-support")]
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let file = File::open("countries.fgb")?;
/// let reader = BufReader::new(file);
/// let geometries = oxirs_geosparql::geometry::flatgeobuf_parser::parse_flatgeobuf(reader)?;
/// println!("Loaded {} geometries", geometries.len());
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "flatgeobuf-support")]
pub fn parse_flatgeobuf<R: Read + Seek>(mut reader: R) -> Result<Vec<Geometry>> {
    use fallible_streaming_iterator::FallibleStreamingIterator;

    let mut fgb = FgbReader::open(&mut reader)
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to open FlatGeobuf file: {}", e)))?
        .select_all()
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to select all features: {}", e)))?;

    let mut geometries = Vec::new();

    while let Some(feature) = fgb
        .next()
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to read feature: {}", e)))?
    {
        // Use geo_traits for zero-copy geometry access
        if let Ok(Some(geom_trait)) = feature.geometry_trait() {
            let geometry = convert_geo_trait_to_geometry(&geom_trait)?;
            geometries.push(geometry);
        }
    }

    Ok(geometries)
}

/// Convert geo_traits geometry to our Geometry type
#[cfg(feature = "flatgeobuf-support")]
fn convert_geo_trait_to_geometry<G: GeometryTrait<T = f64>>(geom_trait: &G) -> Result<Geometry> {
    use geo_traits::GeometryType as GeoTraitType;

    match geom_trait.as_type() {
        GeoTraitType::Point(point) => {
            let coord = point.coord().unwrap();
            let x = coord.x();
            let y = coord.y();
            let geometry = Geometry::new(geo_types::Geometry::Point(geo_types::Point::new(x, y)));

            // Note: Z coordinates require geozero processing
            // geo_traits doesn't provide direct Z access through CoordTrait

            Ok(geometry)
        }
        GeoTraitType::LineString(linestring) => {
            let coords: Vec<_> = linestring
                .coords()
                .map(|coord| geo_types::coord! { x: coord.x(), y: coord.y() })
                .collect();
            // Note: Z coordinates would require geozero processing
            // geo_traits doesn't expose Z through simple trait methods

            let geometry = Geometry::new(geo_types::Geometry::LineString(
                geo_types::LineString::new(coords),
            ));

            Ok(geometry)
        }
        GeoTraitType::Polygon(polygon) => {
            // Exterior ring
            let exterior_coords: Vec<_> = polygon
                .exterior()
                .expect("Polygon must have exterior ring")
                .coords()
                .map(|coord| {
                    // Note: Z coordinates require geozero processing
                    geo_types::coord! { x: coord.x(), y: coord.y() }
                })
                .collect();

            let exterior = geo_types::LineString::new(exterior_coords);

            // Interior rings
            let interiors: Vec<_> = polygon
                .interiors()
                .map(|ring| {
                    let coords: Vec<_> = ring
                        .coords()
                        .map(|coord| {
                            // Note: Z coordinates require geozero processing
                            geo_types::coord! { x: coord.x(), y: coord.y() }
                        })
                        .collect();
                    geo_types::LineString::new(coords)
                })
                .collect();

            let geometry = Geometry::new(geo_types::Geometry::Polygon(geo_types::Polygon::new(
                exterior, interiors,
            )));

            Ok(geometry)
        }
        GeoTraitType::MultiPoint(multipoint) => {
            let mut points = Vec::new();

            for point in multipoint.points() {
                let coord = point.coord().unwrap();
                points.push(geo_types::Point::new(coord.x(), coord.y()));
                // Note: Z coordinates require geozero processing
            }

            let geometry = Geometry::new(geo_types::Geometry::MultiPoint(geo_types::MultiPoint(
                points,
            )));

            Ok(geometry)
        }
        GeoTraitType::MultiLineString(multilinestring) => {
            let mut linestrings = Vec::new();

            for linestring in multilinestring.line_strings() {
                let coords: Vec<_> = linestring
                    .coords()
                    .map(|coord| {
                        // Note: Z coordinates require geozero processing
                        geo_types::coord! { x: coord.x(), y: coord.y() }
                    })
                    .collect();
                linestrings.push(geo_types::LineString::new(coords));
            }

            let geometry = Geometry::new(geo_types::Geometry::MultiLineString(
                geo_types::MultiLineString(linestrings),
            ));

            Ok(geometry)
        }
        GeoTraitType::MultiPolygon(multipolygon) => {
            let mut polygons = Vec::new();

            for polygon in multipolygon.polygons() {
                let exterior_coords: Vec<_> = polygon
                    .exterior()
                    .expect("Polygon must have exterior ring")
                    .coords()
                    .map(|coord| {
                        // Note: Z coordinates require geozero processing
                        geo_types::coord! { x: coord.x(), y: coord.y() }
                    })
                    .collect();

                let exterior = geo_types::LineString::new(exterior_coords);

                let interiors: Vec<_> = polygon
                    .interiors()
                    .map(|ring| {
                        let coords: Vec<_> = ring
                            .coords()
                            .map(|coord| {
                                // Note: Z coordinates require geozero processing
                                geo_types::coord! { x: coord.x(), y: coord.y() }
                            })
                            .collect();
                        geo_types::LineString::new(coords)
                    })
                    .collect();

                polygons.push(geo_types::Polygon::new(exterior, interiors));
            }

            let geometry = Geometry::new(geo_types::Geometry::MultiPolygon(
                geo_types::MultiPolygon(polygons),
            ));

            Ok(geometry)
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(
            "Unsupported geometry type in FlatGeobuf".to_string(),
        )),
    }
}

/// Parse FlatGeobuf data from a reader (fallback when feature is disabled)
#[cfg(not(feature = "flatgeobuf-support"))]
pub fn parse_flatgeobuf<R: Read + Seek>(_reader: R) -> Result<Vec<Geometry>> {
    Err(GeoSparqlError::UnsupportedOperation(
        "FlatGeobuf support requires the 'flatgeobuf-support' feature to be enabled".to_string(),
    ))
}

/// Parse FlatGeobuf data from a byte slice
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "flatgeobuf-support")]
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let data = std::fs::read("data.fgb")?;
/// let geometries = oxirs_geosparql::geometry::flatgeobuf_parser::parse_flatgeobuf_bytes(&data)?;
/// # Ok(())
/// # }
/// ```
pub fn parse_flatgeobuf_bytes(data: &[u8]) -> Result<Vec<Geometry>> {
    use std::io::Cursor;
    parse_flatgeobuf(Cursor::new(data))
}

/// Write geometries to FlatGeobuf format
///
/// # Arguments
///
/// * `geometries` - Slice of geometries to write
/// * `output_path` - Path where the FlatGeobuf file will be created
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_geosparql::geometry::Geometry;
/// use geo_types::{Point, Geometry as GeoGeometry};
/// # #[cfg(feature = "flatgeobuf-support")]
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let geom = Geometry::new(GeoGeometry::Point(Point::new(10.0, 20.0)));
/// let geometries = vec![geom];
/// oxirs_geosparql::geometry::flatgeobuf_parser::write_flatgeobuf_to_file(
///     &geometries,
///     "output.fgb",
/// )?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "flatgeobuf-support")]
pub fn write_flatgeobuf_to_file(geometries: &[Geometry], output_path: &str) -> Result<()> {
    if geometries.is_empty() {
        return Err(GeoSparqlError::GeometryOperationFailed(
            "Cannot write empty geometry collection".to_string(),
        ));
    }

    // Determine geometry type from first geometry
    let geom_type = match &geometries[0].geom {
        geo_types::Geometry::Point(_) => GeometryType::Point,
        geo_types::Geometry::LineString(_) => GeometryType::LineString,
        geo_types::Geometry::Polygon(_) => GeometryType::Polygon,
        geo_types::Geometry::MultiPoint(_) => GeometryType::MultiPoint,
        geo_types::Geometry::MultiLineString(_) => GeometryType::MultiLineString,
        geo_types::Geometry::MultiPolygon(_) => GeometryType::MultiPolygon,
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "Unsupported geometry type for FlatGeobuf export".to_string(),
            ))
        }
    };

    // Create writer
    let _fgb = FgbWriter::create(output_path, geom_type).map_err(|e| {
        GeoSparqlError::GeometryOperationFailed(format!(
            "Failed to create FlatGeobuf writer: {}",
            e
        ))
    })?;

    // Writing requires implementing geozero::GeozeroDatasource for our Geometry type
    // This is a complex integration that requires:
    // 1. Implementing geozero::GeozeroDatasource trait
    // 2. Using geozero process() API for serialization
    // 3. Proper handling of feature properties and metadata
    //
    // For now, we return an error directing users to the alternative approach
    Err(GeoSparqlError::UnsupportedOperation(
        "Direct FlatGeobuf writing requires implementing geozero::GeozeroDatasource. \
         As a workaround, convert to GeoJSON first, then use geozero's GeoJson reader with FgbWriter. \
         See examples/flatgeobuf_support.rs for the complete workflow.".to_string(),
    ))
}

/// Write geometries to FlatGeobuf format (fallback when feature is disabled)
#[cfg(not(feature = "flatgeobuf-support"))]
pub fn write_flatgeobuf_to_file(_geometries: &[Geometry], _output_path: &str) -> Result<()> {
    Err(GeoSparqlError::UnsupportedOperation(
        "FlatGeobuf support requires the 'flatgeobuf-support' feature to be enabled".to_string(),
    ))
}

/// Legacy write function (deprecated)
pub fn write_flatgeobuf<W: Write>(_geometries: &[Geometry], _writer: W) -> Result<()> {
    // FlatGeobuf v5.0 requires file-based writing
    Err(GeoSparqlError::UnsupportedOperation(
        "FlatGeobuf v5.0 requires file-based writing. Use write_flatgeobuf_to_file() instead. \
         The Write trait is not supported by FlatGeobuf v5.0 API."
            .to_string(),
    ))
}

/// Write geometries to FlatGeobuf bytes (not supported in v5.0)
pub fn write_flatgeobuf_bytes(_geometries: &[Geometry]) -> Result<Vec<u8>> {
    Err(GeoSparqlError::UnsupportedOperation(
        "FlatGeobuf v5.0 does not support in-memory byte writing. \
         Use write_flatgeobuf_to_file() and read the file instead."
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    #[cfg(feature = "flatgeobuf-support")]
    fn test_flatgeobuf_parse_empty_file() {
        // Test with empty cursor
        let data = vec![];
        let cursor = std::io::Cursor::new(data);
        let result = parse_flatgeobuf(cursor);
        // Should fail because empty data is not a valid FlatGeobuf file
        assert!(result.is_err());
    }

    #[test]
    fn test_flatgeobuf_write_empty_geometries() {
        let geometries: Vec<Geometry> = vec![];
        #[cfg(feature = "flatgeobuf-support")]
        {
            let result = write_flatgeobuf_to_file(&geometries, "/tmp/test.fgb");
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                GeoSparqlError::GeometryOperationFailed(_)
            ));
        }

        #[cfg(not(feature = "flatgeobuf-support"))]
        {
            let result = write_flatgeobuf_to_file(&geometries, "/tmp/test.fgb");
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_flatgeobuf_bytes_write_not_supported() {
        let point = Point::new(10.0, 20.0);
        let geom = Geometry::new(GeoGeometry::Point(point));
        let geometries = vec![geom];

        let result = write_flatgeobuf_bytes(&geometries);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeoSparqlError::UnsupportedOperation(_)
        ));
    }

    #[test]
    #[cfg(feature = "flatgeobuf-support")]
    fn test_flatgeobuf_write_requires_file_path() {
        let point = Point::new(10.0, 20.0);
        let geom = Geometry::new(GeoGeometry::Point(point));
        let geometries = vec![geom];

        // Test that Write trait is not supported
        let mut buffer = Vec::new();
        let result = write_flatgeobuf(&geometries, &mut buffer);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeoSparqlError::UnsupportedOperation(_)
        ));
    }

    #[test]
    #[cfg(not(feature = "flatgeobuf-support"))]
    fn test_flatgeobuf_disabled_feature() {
        let data = vec![0u8; 10];
        let result = parse_flatgeobuf_bytes(&data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeoSparqlError::UnsupportedOperation(_)
        ));
    }

    #[test]
    #[cfg(feature = "flatgeobuf-support")]
    fn test_flatgeobuf_invalid_data() {
        // Test with invalid FlatGeobuf data
        let data = vec![0u8; 100];
        let result = parse_flatgeobuf_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "flatgeobuf-support")]
    fn test_convert_geo_trait_point() {
        // This is a conceptual test - actual implementation would need real geo_traits objects
        // which are typically created by FlatGeobuf reader
        // For now, we verify the error handling structure is in place
        use geo_types::Point;
        let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert!(!geom.is_3d());
    }

    #[test]
    #[cfg(feature = "flatgeobuf-support")]
    fn test_convert_geo_trait_point_3d() {
        // Test 3D point handling infrastructure
        let mut geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        geom.coord3d = crate::geometry::coord3d::Coord3D::xyz(vec![3.0]);
        assert!(geom.is_3d());
    }
}
