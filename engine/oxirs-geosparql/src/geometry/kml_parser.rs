//! KML (Keyhole Markup Language) parser and serializer
//!
//! Converts between KML XML strings and geometry objects.
//! Supports KML 2.2 and 2.3 specifications (Google Earth format).

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};
use geo_types::Geometry as GeoGeometry;

#[cfg(feature = "kml-support")]
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
#[cfg(feature = "kml-support")]
use quick_xml::{Reader, Writer};
#[cfg(feature = "kml-support")]
use std::io::Cursor;

/// Parse a KML string into a Geometry
///
/// # KML Format
/// KML uses space-separated longitude,latitude,altitude tuples.
/// Note: KML coordinates are in longitude,latitude order (opposite of typical lat,lon).
///
/// # Examples
/// ```
/// # #[cfg(feature = "kml-support")]
/// # {
/// use oxirs_geosparql::geometry::kml_parser::parse_kml;
///
/// let kml = r#"
///   <Point>
///     <coordinates>-122.0822035425683,37.42228990140251,0</coordinates>
///   </Point>
/// "#;
///
/// let geometry = parse_kml(kml).unwrap();
/// # }
/// ```
#[cfg(feature = "kml-support")]
pub fn parse_kml(kml_str: &str) -> Result<Geometry> {
    let mut reader = Reader::from_str(kml_str);
    reader.config_mut().trim_text(true);

    let crs = Crs::new("http://www.opengis.net/def/crs/EPSG/0/4326".to_string()); // KML is always WGS84
    let mut geometry: Option<GeoGeometry<f64>> = None;
    let mut coords_buffer = String::new();
    let mut in_coordinates = false;
    let mut last_coordinates = String::new(); // Store coordinates from most recent </coordinates> tag

    // For multi-geometries
    let mut multi_points: Vec<geo_types::Point<f64>> = Vec::new();
    let mut multi_linestrings: Vec<geo_types::LineString<f64>> = Vec::new();
    let mut multi_polygons: Vec<geo_types::Polygon<f64>> = Vec::new();

    // For polygon boundaries
    let mut outer_boundary_coords = String::new();
    let mut inner_boundary_coords: Vec<String> = Vec::new();
    let mut in_outer_boundary = false;
    let mut in_inner_boundary = false;

    // Geometry type tracking
    let mut current_geom_type: Option<String> = None;
    let mut in_multi_geometry = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                match name.as_str() {
                    "Point" | "kml:Point" => {
                        if in_multi_geometry {
                            // Part of MultiGeometry - will add to collection
                        } else if current_geom_type.is_none() {
                            current_geom_type = Some("Point".to_string());
                        }
                    }
                    "LineString" | "kml:LineString" => {
                        if in_multi_geometry {
                            // Part of MultiGeometry
                        } else if current_geom_type.is_none() {
                            current_geom_type = Some("LineString".to_string());
                        }
                    }
                    "Polygon" | "kml:Polygon" => {
                        if in_multi_geometry {
                            // Part of MultiGeometry
                        } else if current_geom_type.is_none() {
                            current_geom_type = Some("Polygon".to_string());
                        }
                    }
                    "MultiGeometry" | "kml:MultiGeometry" => {
                        in_multi_geometry = true;
                        current_geom_type = Some("MultiGeometry".to_string());
                    }
                    "outerBoundaryIs" | "kml:outerBoundaryIs" => {
                        in_outer_boundary = true;
                        outer_boundary_coords.clear();
                    }
                    "innerBoundaryIs" | "kml:innerBoundaryIs" => {
                        in_inner_boundary = true;
                        coords_buffer.clear();
                    }
                    "coordinates" | "kml:coordinates" => {
                        in_coordinates = true;
                        coords_buffer.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_coordinates {
                    // In quick-xml 0.38+, decode text from bytes
                    let text = String::from_utf8_lossy(e.as_ref());
                    coords_buffer.push_str(&text);
                }
            }
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                match name.as_str() {
                    "coordinates" | "kml:coordinates" => {
                        in_coordinates = false;

                        if in_outer_boundary {
                            outer_boundary_coords.push_str(&coords_buffer);
                        } else if in_inner_boundary {
                            inner_boundary_coords.push(coords_buffer.clone());
                        } else {
                            // For simple geometries (not in boundary), save to last_coordinates
                            last_coordinates = coords_buffer.clone();
                        }

                        coords_buffer.clear();
                    }
                    "outerBoundaryIs" | "kml:outerBoundaryIs" => {
                        in_outer_boundary = false;
                    }
                    "innerBoundaryIs" | "kml:innerBoundaryIs" => {
                        in_inner_boundary = false;
                    }
                    "Point" | "kml:Point" => {
                        if !last_coordinates.is_empty() {
                            if let Some(point) = parse_kml_point(&last_coordinates) {
                                if in_multi_geometry {
                                    multi_points.push(point);
                                } else {
                                    geometry = Some(GeoGeometry::Point(point));
                                }
                            }
                            last_coordinates.clear();
                        }
                    }
                    "LineString" | "kml:LineString" => {
                        if !last_coordinates.is_empty() {
                            if let Some(linestring) = parse_kml_linestring(&last_coordinates) {
                                if in_multi_geometry {
                                    multi_linestrings.push(linestring);
                                } else {
                                    geometry = Some(GeoGeometry::LineString(linestring));
                                }
                            }
                            last_coordinates.clear();
                        }
                    }
                    "Polygon" | "kml:Polygon" => {
                        if !outer_boundary_coords.is_empty() {
                            if let Some(polygon) =
                                parse_kml_polygon(&outer_boundary_coords, &inner_boundary_coords)
                            {
                                if in_multi_geometry {
                                    multi_polygons.push(polygon);
                                } else {
                                    geometry = Some(GeoGeometry::Polygon(polygon));
                                }
                            }
                        }
                        outer_boundary_coords.clear();
                        inner_boundary_coords.clear();
                    }
                    "MultiGeometry" | "kml:MultiGeometry" => {
                        // Finalize multi-geometry
                        if !multi_points.is_empty() {
                            geometry = Some(GeoGeometry::MultiPoint(geo_types::MultiPoint(
                                multi_points.clone(),
                            )));
                        } else if !multi_linestrings.is_empty() {
                            geometry = Some(GeoGeometry::MultiLineString(
                                geo_types::MultiLineString(multi_linestrings.clone()),
                            ));
                        } else if !multi_polygons.is_empty() {
                            geometry = Some(GeoGeometry::MultiPolygon(geo_types::MultiPolygon(
                                multi_polygons.clone(),
                            )));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(GeoSparqlError::ParseError(format!(
                    "XML parsing error: {}",
                    e
                )))
            }
            _ => {}
        }
    }

    geometry
        .map(|g| Geometry::with_crs(g, crs))
        .ok_or_else(|| GeoSparqlError::ParseError("No geometry found in KML".to_string()))
}

/// Parse KML coordinates string into a Point
/// KML format: "lon,lat,alt" or "lon,lat"
#[cfg(feature = "kml-support")]
fn parse_kml_point(coords: &str) -> Option<geo_types::Point<f64>> {
    let coords = coords.trim();
    if coords.is_empty() {
        return None;
    }

    // KML coordinates are space-separated tuples: "lon1,lat1,alt1 lon2,lat2,alt2 ..."
    // For a point, we take the first tuple
    let first_tuple = coords.split_whitespace().next()?;
    let parts: Vec<&str> = first_tuple.split(',').collect();

    if parts.len() >= 2 {
        let lon = parts[0].trim().parse::<f64>().ok()?;
        let lat = parts[1].trim().parse::<f64>().ok()?;
        // Note: KML uses lon,lat order, but geo_types::Point is (x,y) which maps to (lon,lat)
        Some(geo_types::Point::new(lon, lat))
    } else {
        None
    }
}

/// Parse KML coordinates string into a LineString
#[cfg(feature = "kml-support")]
fn parse_kml_linestring(coords: &str) -> Option<geo_types::LineString<f64>> {
    let coords = coords.trim();
    if coords.is_empty() {
        return None;
    }

    let mut points = Vec::new();

    // Split by whitespace to get individual coordinate tuples
    for tuple in coords.split_whitespace() {
        let parts: Vec<&str> = tuple.split(',').collect();
        if parts.len() >= 2 {
            if let (Ok(lon), Ok(lat)) = (
                parts[0].trim().parse::<f64>(),
                parts[1].trim().parse::<f64>(),
            ) {
                points.push(geo_types::Coord { x: lon, y: lat });
            }
        }
    }

    if points.len() >= 2 {
        Some(geo_types::LineString(points))
    } else {
        None
    }
}

/// Parse KML coordinates into a Polygon with optional holes
#[cfg(feature = "kml-support")]
fn parse_kml_polygon(
    outer_coords: &str,
    inner_coords: &[String],
) -> Option<geo_types::Polygon<f64>> {
    let exterior = parse_kml_linestring(outer_coords)?;

    let mut interiors = Vec::new();
    for inner in inner_coords {
        if let Some(interior) = parse_kml_linestring(inner) {
            interiors.push(interior);
        }
    }

    Some(geo_types::Polygon::new(exterior, interiors))
}

/// Convert a Geometry to KML string
///
/// # Examples
/// ```
/// # #[cfg(feature = "kml-support")]
/// # {
/// use oxirs_geosparql::geometry::{Geometry, Crs};
/// use oxirs_geosparql::geometry::kml_parser::geometry_to_kml;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let point = Point::new(-122.0822035425683, 37.42228990140251);
/// let geometry = Geometry::new(GeoGeometry::Point(point));
/// let kml = geometry_to_kml(&geometry).unwrap();
/// # }
/// ```
#[cfg(feature = "kml-support")]
pub fn geometry_to_kml(geometry: &Geometry) -> Result<String> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    match &geometry.geom {
        GeoGeometry::Point(point) => write_kml_point(&mut writer, point)?,
        GeoGeometry::LineString(linestring) => write_kml_linestring(&mut writer, linestring)?,
        GeoGeometry::Polygon(polygon) => write_kml_polygon(&mut writer, polygon)?,
        GeoGeometry::MultiPoint(multi_point) => write_kml_multi_point(&mut writer, multi_point)?,
        GeoGeometry::MultiLineString(multi_linestring) => {
            write_kml_multi_linestring(&mut writer, multi_linestring)?
        }
        GeoGeometry::MultiPolygon(multi_polygon) => {
            write_kml_multi_polygon(&mut writer, multi_polygon)?
        }
        _ => {
            return Err(GeoSparqlError::SerializationError(
                "Unsupported geometry type for KML".to_string(),
            ))
        }
    }

    let result = writer.into_inner().into_inner();
    String::from_utf8(result)
        .map_err(|e| GeoSparqlError::SerializationError(format!("UTF-8 conversion error: {}", e)))
}

#[cfg(feature = "kml-support")]
fn write_kml_point<W: std::io::Write>(
    writer: &mut Writer<W>,
    point: &geo_types::Point<f64>,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new("Point")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("coordinates")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    let coords = format!("{},{},0", point.x(), point.y());
    writer
        .write_event(Event::Text(BytesText::new(&coords)))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("coordinates")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("Point")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "kml-support")]
fn write_kml_linestring<W: std::io::Write>(
    writer: &mut Writer<W>,
    linestring: &geo_types::LineString<f64>,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new("LineString")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("coordinates")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    let coords = linestring
        .coords()
        .map(|c| format!("{},{},0", c.x, c.y))
        .collect::<Vec<_>>()
        .join(" ");

    writer
        .write_event(Event::Text(BytesText::new(&coords)))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("coordinates")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("LineString")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "kml-support")]
fn write_kml_polygon<W: std::io::Write>(
    writer: &mut Writer<W>,
    polygon: &geo_types::Polygon<f64>,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new("Polygon")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    // Write outer boundary
    writer
        .write_event(Event::Start(BytesStart::new("outerBoundaryIs")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("LinearRing")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("coordinates")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    let exterior_coords = polygon
        .exterior()
        .coords()
        .map(|c| format!("{},{},0", c.x, c.y))
        .collect::<Vec<_>>()
        .join(" ");

    writer
        .write_event(Event::Text(BytesText::new(&exterior_coords)))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("coordinates")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("LinearRing")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("outerBoundaryIs")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    // Write inner boundaries (holes)
    for interior in polygon.interiors() {
        writer
            .write_event(Event::Start(BytesStart::new("innerBoundaryIs")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::Start(BytesStart::new("LinearRing")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::Start(BytesStart::new("coordinates")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

        let interior_coords = interior
            .coords()
            .map(|c| format!("{},{},0", c.x, c.y))
            .collect::<Vec<_>>()
            .join(" ");

        writer
            .write_event(Event::Text(BytesText::new(&interior_coords)))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::End(BytesEnd::new("coordinates")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::End(BytesEnd::new("LinearRing")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::End(BytesEnd::new("innerBoundaryIs")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("Polygon")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "kml-support")]
fn write_kml_multi_point<W: std::io::Write>(
    writer: &mut Writer<W>,
    multi_point: &geo_types::MultiPoint<f64>,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new("MultiGeometry")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    for point in multi_point.iter() {
        write_kml_point(writer, point)?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("MultiGeometry")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "kml-support")]
fn write_kml_multi_linestring<W: std::io::Write>(
    writer: &mut Writer<W>,
    multi_linestring: &geo_types::MultiLineString<f64>,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new("MultiGeometry")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    for linestring in multi_linestring.iter() {
        write_kml_linestring(writer, linestring)?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("MultiGeometry")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "kml-support")]
fn write_kml_multi_polygon<W: std::io::Write>(
    writer: &mut Writer<W>,
    multi_polygon: &geo_types::MultiPolygon<f64>,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new("MultiGeometry")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    for polygon in multi_polygon.iter() {
        write_kml_polygon(writer, polygon)?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("MultiGeometry")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(test)]
#[cfg(feature = "kml-support")]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use geo_types::{LineString, Point, Polygon};

    #[test]
    fn test_parse_kml_point() {
        let kml = r#"
            <Point>
                <coordinates>-122.0822035425683,37.42228990140251,0</coordinates>
            </Point>
        "#;

        let geometry = parse_kml(kml).unwrap();
        match &geometry.geom {
            GeoGeometry::Point(point) => {
                assert_relative_eq!(point.x(), -122.0822035425683, epsilon = 1e-9);
                assert_relative_eq!(point.y(), 37.42228990140251, epsilon = 1e-9);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_parse_kml_linestring() {
        let kml = r#"
            <LineString>
                <coordinates>
                    -122.08223,37.42254 -122.08219,37.42281 -122.08244,37.42292
                </coordinates>
            </LineString>
        "#;

        let geometry = parse_kml(kml).unwrap();
        match &geometry.geom {
            GeoGeometry::LineString(linestring) => {
                assert_eq!(linestring.coords().count(), 3);
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_parse_kml_polygon() {
        let kml = r#"
            <Polygon>
                <outerBoundaryIs>
                    <LinearRing>
                        <coordinates>
                            0,0,0 10,0,0 10,10,0 0,10,0 0,0,0
                        </coordinates>
                    </LinearRing>
                </outerBoundaryIs>
            </Polygon>
        "#;

        let geometry = parse_kml(kml).unwrap();
        match &geometry.geom {
            GeoGeometry::Polygon(polygon) => {
                assert_eq!(polygon.exterior().coords().count(), 5);
            }
            _ => panic!("Expected Polygon geometry"),
        }
    }

    #[test]
    fn test_parse_kml_polygon_with_hole() {
        let kml = r#"
            <Polygon>
                <outerBoundaryIs>
                    <LinearRing>
                        <coordinates>0,0,0 20,0,0 20,20,0 0,20,0 0,0,0</coordinates>
                    </LinearRing>
                </outerBoundaryIs>
                <innerBoundaryIs>
                    <LinearRing>
                        <coordinates>5,5,0 15,5,0 15,15,0 5,15,0 5,5,0</coordinates>
                    </LinearRing>
                </innerBoundaryIs>
            </Polygon>
        "#;

        let geometry = parse_kml(kml).unwrap();
        match &geometry.geom {
            GeoGeometry::Polygon(polygon) => {
                assert_eq!(polygon.exterior().coords().count(), 5);
                assert_eq!(polygon.interiors().len(), 1);
                assert_eq!(polygon.interiors()[0].coords().count(), 5);
            }
            _ => panic!("Expected Polygon geometry"),
        }
    }

    #[test]
    fn test_geometry_to_kml_point() {
        let point = Point::new(-122.0822035425683, 37.42228990140251);
        let geometry = Geometry::new(GeoGeometry::Point(point));
        let kml = geometry_to_kml(&geometry).unwrap();

        assert!(kml.contains("<Point>"));
        assert!(kml.contains("<coordinates>"));
        assert!(kml.contains("-122.0822035425683,37.42228990140251"));
    }

    #[test]
    fn test_geometry_to_kml_linestring() {
        let linestring = LineString::from(vec![
            (-122.08223, 37.42254),
            (-122.08219, 37.42281),
            (-122.08244, 37.42292),
        ]);
        let geometry = Geometry::new(GeoGeometry::LineString(linestring));
        let kml = geometry_to_kml(&geometry).unwrap();

        assert!(kml.contains("<LineString>"));
        assert!(kml.contains("<coordinates>"));
    }

    #[test]
    fn test_geometry_to_kml_polygon() {
        let polygon = Polygon::new(
            LineString::from(vec![
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ]),
            vec![],
        );
        let geometry = Geometry::new(GeoGeometry::Polygon(polygon));
        let kml = geometry_to_kml(&geometry).unwrap();

        assert!(kml.contains("<Polygon>"));
        assert!(kml.contains("<outerBoundaryIs>"));
        assert!(kml.contains("<LinearRing>"));
    }

    #[test]
    fn test_roundtrip_point() {
        let point = Point::new(-122.0822035425683, 37.42228990140251);
        let geometry = Geometry::new(GeoGeometry::Point(point));
        let kml = geometry_to_kml(&geometry).unwrap();
        let parsed = parse_kml(&kml).unwrap();

        match &parsed.geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), point.x(), epsilon = 1e-9);
                assert_relative_eq!(p.y(), point.y(), epsilon = 1e-9);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_roundtrip_linestring() {
        let linestring = LineString::from(vec![
            (-122.08223, 37.42254),
            (-122.08219, 37.42281),
            (-122.08244, 37.42292),
        ]);
        let geometry = Geometry::new(GeoGeometry::LineString(linestring.clone()));
        let kml = geometry_to_kml(&geometry).unwrap();
        let parsed = parse_kml(&kml).unwrap();

        match &parsed.geom {
            GeoGeometry::LineString(ls) => {
                assert_eq!(ls.coords().count(), linestring.coords().count());
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_roundtrip_polygon() {
        let polygon = Polygon::new(
            LineString::from(vec![
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ]),
            vec![],
        );
        let geometry = Geometry::new(GeoGeometry::Polygon(polygon.clone()));
        let kml = geometry_to_kml(&geometry).unwrap();
        let parsed = parse_kml(&kml).unwrap();

        match &parsed.geom {
            GeoGeometry::Polygon(p) => {
                assert_eq!(
                    p.exterior().coords().count(),
                    polygon.exterior().coords().count()
                );
            }
            _ => panic!("Expected Polygon geometry"),
        }
    }

    #[test]
    fn test_multi_geometry() {
        let kml = r#"
            <MultiGeometry>
                <Point><coordinates>-122.0822,37.4223,0</coordinates></Point>
                <Point><coordinates>-122.0844,37.4228,0</coordinates></Point>
            </MultiGeometry>
        "#;

        let geometry = parse_kml(kml).unwrap();
        match &geometry.geom {
            GeoGeometry::MultiPoint(mp) => {
                assert_eq!(mp.0.len(), 2);
            }
            _ => panic!("Expected MultiPoint geometry"),
        }
    }

    #[test]
    fn test_parse_kml_empty() {
        let kml = "<Point></Point>";
        let result = parse_kml(kml);
        assert!(result.is_err());
    }

    #[test]
    fn test_crs_always_wgs84() {
        let kml = r#"
            <Point>
                <coordinates>-122.0822035425683,37.42228990140251,0</coordinates>
            </Point>
        "#;

        let geometry = parse_kml(kml).unwrap();
        // KML is always WGS84
        assert_eq!(
            geometry.crs.uri,
            "http://www.opengis.net/def/crs/EPSG/0/4326"
        );
    }
}
