//! GML (Geography Markup Language) parser and serializer
//!
//! Converts between GML XML strings and geometry objects.
//! Supports GML 3.1.1 and GML 3.2.1 specifications.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};
use geo_types::Geometry as GeoGeometry;

#[cfg(feature = "gml-support")]
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
#[cfg(feature = "gml-support")]
use quick_xml::{Reader, Writer};
#[cfg(feature = "gml-support")]
use std::io::Cursor;

/// Parse a GML string into a Geometry
#[cfg(feature = "gml-support")]
pub fn parse_gml(gml_str: &str) -> Result<Geometry> {
    let mut reader = Reader::from_str(gml_str);
    reader.config_mut().trim_text(true);

    let mut crs = Crs::default();
    let mut geometry: Option<GeoGeometry<f64>> = None;
    let mut coords_buffer = String::new();
    let mut in_pos_list = false;
    let mut in_coordinates = false;

    // For multi-geometries
    let mut multi_points: Vec<geo_types::Point<f64>> = Vec::new();
    let mut multi_linestrings: Vec<geo_types::LineString<f64>> = Vec::new();
    let mut multi_polygons: Vec<geo_types::Polygon<f64>> = Vec::new();

    // For polygon with holes
    let mut exterior_coords = String::new();
    let mut interior_coords: Vec<String> = Vec::new();

    // Geometry type tracking
    let mut current_geom_type: Option<String> = None;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                // Extract CRS from srsName attribute
                if let Some(Ok(attr)) = e
                    .attributes()
                    .find(|a| a.as_ref().ok().map(|attr| attr.key.as_ref()) == Some(b"srsName"))
                {
                    let srs_value = String::from_utf8_lossy(&attr.value).to_string();
                    crs = Crs::new(srs_value);
                }

                match name.as_str() {
                    "Point" | "gml:Point" => {
                        if current_geom_type.is_none() {
                            current_geom_type = Some("Point".to_string());
                        }
                    }
                    "LineString" | "gml:LineString" => {
                        if current_geom_type.is_none() {
                            current_geom_type = Some("LineString".to_string());
                        }
                    }
                    "Polygon" | "gml:Polygon" => {
                        if current_geom_type.is_none() {
                            current_geom_type = Some("Polygon".to_string());
                        }
                    }
                    "MultiPoint" | "gml:MultiPoint" => {
                        current_geom_type = Some("MultiPoint".to_string());
                    }
                    "MultiLineString" | "gml:MultiLineString" | "MultiCurve" | "gml:MultiCurve" => {
                        current_geom_type = Some("MultiLineString".to_string());
                    }
                    "MultiPolygon" | "gml:MultiPolygon" | "MultiSurface" | "gml:MultiSurface" => {
                        current_geom_type = Some("MultiPolygon".to_string());
                    }
                    "exterior" | "gml:exterior" => {
                        exterior_coords.clear();
                    }
                    "interior" | "gml:interior" => {
                        coords_buffer.clear();
                    }
                    "posList" | "gml:posList" | "pos" | "gml:pos" => {
                        in_pos_list = true;
                        coords_buffer.clear();
                    }
                    "coordinates" | "gml:coordinates" => {
                        in_coordinates = true;
                        coords_buffer.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_pos_list || in_coordinates {
                    // In quick-xml 0.38+, decode text from bytes
                    let text = String::from_utf8_lossy(e.as_ref());
                    coords_buffer.push_str(&text);
                }
            }
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                match name.as_str() {
                    "Point" | "gml:Point" => {
                        if current_geom_type.as_deref() == Some("MultiPoint") {
                            // Part of MultiPoint
                            if let Ok(GeoGeometry::Point(p)) = parse_gml_point(&coords_buffer) {
                                multi_points.push(p);
                            }
                        } else {
                            // Standalone Point
                            geometry = Some(parse_gml_point(&coords_buffer)?);
                        }
                        coords_buffer.clear();
                    }
                    "LineString" | "gml:LineString" => {
                        if current_geom_type.as_deref() == Some("MultiLineString") {
                            // Part of MultiLineString
                            if let Ok(GeoGeometry::LineString(ls)) =
                                parse_gml_linestring(&coords_buffer)
                            {
                                multi_linestrings.push(ls);
                            }
                        } else {
                            // Standalone LineString
                            geometry = Some(parse_gml_linestring(&coords_buffer)?);
                        }
                        coords_buffer.clear();
                    }
                    "Polygon" | "gml:Polygon" => {
                        if current_geom_type.as_deref() == Some("MultiPolygon") {
                            // Part of MultiPolygon
                            let poly =
                                parse_gml_polygon_with_holes(&exterior_coords, &interior_coords)?;
                            if let GeoGeometry::Polygon(p) = poly {
                                multi_polygons.push(p);
                            }
                        } else {
                            // Standalone Polygon
                            geometry = Some(parse_gml_polygon_with_holes(
                                &exterior_coords,
                                &interior_coords,
                            )?);
                        }
                        exterior_coords.clear();
                        interior_coords.clear();
                    }
                    "MultiPoint" | "gml:MultiPoint" => {
                        geometry = Some(GeoGeometry::MultiPoint(geo_types::MultiPoint::new(
                            multi_points.clone(),
                        )));
                        multi_points.clear();
                    }
                    "MultiLineString" | "gml:MultiLineString" | "MultiCurve" | "gml:MultiCurve" => {
                        geometry = Some(GeoGeometry::MultiLineString(
                            geo_types::MultiLineString::new(multi_linestrings.clone()),
                        ));
                        multi_linestrings.clear();
                    }
                    "MultiPolygon" | "gml:MultiPolygon" | "MultiSurface" | "gml:MultiSurface" => {
                        geometry = Some(GeoGeometry::MultiPolygon(geo_types::MultiPolygon::new(
                            multi_polygons.clone(),
                        )));
                        multi_polygons.clear();
                    }
                    "exterior" | "gml:exterior" => {
                        exterior_coords = coords_buffer.clone();
                        coords_buffer.clear();
                    }
                    "interior" | "gml:interior" => {
                        interior_coords.push(coords_buffer.clone());
                        coords_buffer.clear();
                    }
                    "posList" | "gml:posList" | "pos" | "gml:pos" | "coordinates"
                    | "gml:coordinates" => {
                        in_pos_list = false;
                        in_coordinates = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(GeoSparqlError::InvalidGml(format!(
                    "XML parse error: {}",
                    e
                )))
            }
            _ => {}
        }
    }

    geometry
        .map(|geom| Geometry::with_crs(geom, crs))
        .ok_or_else(|| GeoSparqlError::InvalidGml("No geometry found in GML".to_string()))
}

#[cfg(feature = "gml-support")]
fn parse_gml_point(coords: &str) -> Result<GeoGeometry<f64>> {
    let numbers: Vec<f64> = coords
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    if numbers.len() >= 2 {
        Ok(GeoGeometry::Point(geo_types::Point::new(
            numbers[0], numbers[1],
        )))
    } else {
        Err(GeoSparqlError::InvalidGml(format!(
            "Invalid point coordinates: {}",
            coords
        )))
    }
}

#[cfg(feature = "gml-support")]
fn parse_gml_linestring(coords: &str) -> Result<GeoGeometry<f64>> {
    let numbers: Vec<f64> = coords
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    if numbers.len() < 4 || numbers.len() % 2 != 0 {
        return Err(GeoSparqlError::InvalidGml(format!(
            "Invalid linestring coordinates: {}",
            coords
        )));
    }

    let coords: Vec<geo_types::Coord<f64>> = numbers
        .chunks(2)
        .map(|pair| geo_types::Coord {
            x: pair[0],
            y: pair[1],
        })
        .collect();

    Ok(GeoGeometry::LineString(geo_types::LineString::new(coords)))
}

#[cfg(feature = "gml-support")]
fn parse_gml_polygon_with_holes(
    exterior_coords: &str,
    interior_coords_list: &[String],
) -> Result<GeoGeometry<f64>> {
    // Parse exterior ring
    let ext_numbers: Vec<f64> = exterior_coords
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    if ext_numbers.len() < 6 || ext_numbers.len() % 2 != 0 {
        // If no exterior coordinates, try to parse as simple polygon
        if exterior_coords.is_empty() && interior_coords_list.is_empty() {
            return Err(GeoSparqlError::InvalidGml(
                "No polygon coordinates found".to_string(),
            ));
        }
        if !exterior_coords.is_empty() {
            return Err(GeoSparqlError::InvalidGml(format!(
                "Invalid polygon exterior coordinates: {}",
                exterior_coords
            )));
        }
        return Err(GeoSparqlError::InvalidGml(
            "No exterior ring found for polygon".to_string(),
        ));
    }

    let ext_coords: Vec<geo_types::Coord<f64>> = ext_numbers
        .chunks(2)
        .map(|pair| geo_types::Coord {
            x: pair[0],
            y: pair[1],
        })
        .collect();

    let exterior_ring = geo_types::LineString::new(ext_coords);

    // Parse interior rings (holes)
    let mut interior_rings: Vec<geo_types::LineString<f64>> = Vec::new();
    for interior_coords in interior_coords_list {
        let int_numbers: Vec<f64> = interior_coords
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if int_numbers.len() >= 6 && int_numbers.len() % 2 == 0 {
            let int_coords: Vec<geo_types::Coord<f64>> = int_numbers
                .chunks(2)
                .map(|pair| geo_types::Coord {
                    x: pair[0],
                    y: pair[1],
                })
                .collect();

            interior_rings.push(geo_types::LineString::new(int_coords));
        }
    }

    Ok(GeoGeometry::Polygon(geo_types::Polygon::new(
        exterior_ring,
        interior_rings,
    )))
}

/// Convert a Geometry to GML string
#[cfg(feature = "gml-support")]
pub fn geometry_to_gml(geom: &Geometry) -> Result<String> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    match &geom.geom {
        GeoGeometry::Point(p) => {
            write_gml_point(&mut writer, p, &geom.crs)?;
        }
        GeoGeometry::LineString(ls) => {
            write_gml_linestring(&mut writer, ls, &geom.crs)?;
        }
        GeoGeometry::Polygon(poly) => {
            write_gml_polygon(&mut writer, poly, &geom.crs)?;
        }
        GeoGeometry::MultiPoint(mp) => {
            write_gml_multipoint(&mut writer, mp, &geom.crs)?;
        }
        GeoGeometry::MultiLineString(mls) => {
            write_gml_multilinestring(&mut writer, mls, &geom.crs)?;
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            write_gml_multipolygon(&mut writer, mpoly, &geom.crs)?;
        }
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(format!(
                "GML serialization not supported for {}",
                geom.geometry_type()
            )))
        }
    }

    let result = writer.into_inner().into_inner();
    String::from_utf8(result)
        .map_err(|e| GeoSparqlError::InvalidGml(format!("UTF-8 conversion error: {}", e)))
}

#[cfg(feature = "gml-support")]
fn write_gml_point(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    point: &geo_types::Point<f64>,
    crs: &Crs,
) -> Result<()> {
    let mut elem = BytesStart::new("gml:Point");
    if !crs.is_default() {
        elem.push_attribute(("srsName", crs.uri.as_str()));
    }
    writer
        .write_event(Event::Start(elem))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("gml:pos")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    let coords = format!("{} {}", point.x(), point.y());
    writer
        .write_event(Event::Text(BytesText::new(&coords)))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("gml:pos")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("gml:Point")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "gml-support")]
fn write_gml_linestring(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    linestring: &geo_types::LineString<f64>,
    crs: &Crs,
) -> Result<()> {
    let mut elem = BytesStart::new("gml:LineString");
    if !crs.is_default() {
        elem.push_attribute(("srsName", crs.uri.as_str()));
    }
    writer
        .write_event(Event::Start(elem))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("gml:posList")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    let coords: Vec<String> = linestring
        .coords()
        .map(|c| format!("{} {}", c.x, c.y))
        .collect();
    let coords_str = coords.join(" ");

    writer
        .write_event(Event::Text(BytesText::new(&coords_str)))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("gml:posList")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("gml:LineString")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "gml-support")]
fn write_gml_polygon(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    polygon: &geo_types::Polygon<f64>,
    crs: &Crs,
) -> Result<()> {
    let mut elem = BytesStart::new("gml:Polygon");
    if !crs.is_default() {
        elem.push_attribute(("srsName", crs.uri.as_str()));
    }
    writer
        .write_event(Event::Start(elem))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    // Exterior ring
    writer
        .write_event(Event::Start(BytesStart::new("gml:exterior")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("gml:LinearRing")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::Start(BytesStart::new("gml:posList")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    let coords: Vec<String> = polygon
        .exterior()
        .coords()
        .map(|c| format!("{} {}", c.x, c.y))
        .collect();
    let coords_str = coords.join(" ");

    writer
        .write_event(Event::Text(BytesText::new(&coords_str)))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("gml:posList")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("gml:LinearRing")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("gml:exterior")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    // Interior rings (holes)
    for interior in polygon.interiors() {
        writer
            .write_event(Event::Start(BytesStart::new("gml:interior")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::Start(BytesStart::new("gml:LinearRing")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::Start(BytesStart::new("gml:posList")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        let int_coords: Vec<String> = interior
            .coords()
            .map(|c| format!("{} {}", c.x, c.y))
            .collect();
        let int_coords_str = int_coords.join(" ");

        writer
            .write_event(Event::Text(BytesText::new(&int_coords_str)))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::End(BytesEnd::new("gml:posList")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::End(BytesEnd::new("gml:LinearRing")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::End(BytesEnd::new("gml:interior")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("gml:Polygon")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "gml-support")]
fn write_gml_multipoint(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    multipoint: &geo_types::MultiPoint<f64>,
    crs: &Crs,
) -> Result<()> {
    let mut elem = BytesStart::new("gml:MultiPoint");
    if !crs.is_default() {
        elem.push_attribute(("srsName", crs.uri.as_str()));
    }
    writer
        .write_event(Event::Start(elem))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    for point in multipoint.iter() {
        writer
            .write_event(Event::Start(BytesStart::new("gml:pointMember")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        write_gml_point(writer, point, &Crs::default())?; // Don't repeat CRS for members

        writer
            .write_event(Event::End(BytesEnd::new("gml:pointMember")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("gml:MultiPoint")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "gml-support")]
fn write_gml_multilinestring(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    multilinestring: &geo_types::MultiLineString<f64>,
    crs: &Crs,
) -> Result<()> {
    let mut elem = BytesStart::new("gml:MultiCurve");
    if !crs.is_default() {
        elem.push_attribute(("srsName", crs.uri.as_str()));
    }
    writer
        .write_event(Event::Start(elem))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    for linestring in multilinestring.iter() {
        writer
            .write_event(Event::Start(BytesStart::new("gml:curveMember")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        write_gml_linestring(writer, linestring, &Crs::default())?;

        writer
            .write_event(Event::End(BytesEnd::new("gml:curveMember")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("gml:MultiCurve")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "gml-support")]
fn write_gml_multipolygon(
    writer: &mut Writer<Cursor<Vec<u8>>>,
    multipolygon: &geo_types::MultiPolygon<f64>,
    crs: &Crs,
) -> Result<()> {
    let mut elem = BytesStart::new("gml:MultiSurface");
    if !crs.is_default() {
        elem.push_attribute(("srsName", crs.uri.as_str()));
    }
    writer
        .write_event(Event::Start(elem))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    for polygon in multipolygon.iter() {
        writer
            .write_event(Event::Start(BytesStart::new("gml:surfaceMember")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

        write_gml_polygon(writer, polygon, &Crs::default())?;

        writer
            .write_event(Event::End(BytesEnd::new("gml:surfaceMember")))
            .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("gml:MultiSurface")))
        .map_err(|e| GeoSparqlError::InvalidGml(format!("XML write error: {}", e)))?;

    Ok(())
}

// Stub implementations when gml-support feature is disabled
#[cfg(not(feature = "gml-support"))]
pub fn parse_gml(_gml_str: &str) -> Result<Geometry> {
    Err(GeoSparqlError::UnsupportedOperation(
        "GML support is not enabled. Enable the 'gml-support' feature".to_string(),
    ))
}

#[cfg(not(feature = "gml-support"))]
pub fn geometry_to_gml(_geom: &Geometry) -> Result<String> {
    Err(GeoSparqlError::UnsupportedOperation(
        "GML support is not enabled. Enable the 'gml-support' feature".to_string(),
    ))
}

#[cfg(all(test, feature = "gml-support"))]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gml_point() {
        let gml = r#"<gml:Point><gml:pos>1.0 2.0</gml:pos></gml:Point>"#;
        let geom = parse_gml(gml).unwrap();

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
    fn test_parse_gml_point_with_srs() {
        let gml = r#"<gml:Point srsName="http://www.opengis.net/def/crs/EPSG/0/4326"><gml:pos>1.0 2.0</gml:pos></gml:Point>"#;
        let geom = parse_gml(gml).unwrap();

        assert_eq!(geom.crs.uri, "http://www.opengis.net/def/crs/EPSG/0/4326");
    }

    #[test]
    fn test_roundtrip_point() {
        let original = Geometry::from_wkt("POINT(1.5 2.5)").unwrap();
        let gml = geometry_to_gml(&original).unwrap();
        let parsed = parse_gml(&gml).unwrap();

        match (&original.geom, &parsed.geom) {
            (GeoGeometry::Point(p1), GeoGeometry::Point(p2)) => {
                assert_eq!(p1.x(), p2.x());
                assert_eq!(p1.y(), p2.y());
            }
            _ => panic!("Expected Points"),
        }
    }

    #[test]
    fn test_parse_gml_linestring() {
        let gml = r#"<gml:LineString><gml:posList>0.0 0.0 1.0 1.0 2.0 2.0</gml:posList></gml:LineString>"#;
        let geom = parse_gml(gml).unwrap();

        assert_eq!(geom.geometry_type(), "LineString");
        match &geom.geom {
            GeoGeometry::LineString(ls) => {
                assert_eq!(ls.coords().count(), 3);
            }
            _ => panic!("Expected LineString"),
        }
    }

    #[test]
    fn test_roundtrip_linestring() {
        let original = Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        let gml = geometry_to_gml(&original).unwrap();
        let parsed = parse_gml(&gml).unwrap();

        assert_eq!(parsed.geometry_type(), "LineString");
    }

    #[test]
    fn test_parse_gml_polygon() {
        let gml = r#"
            <gml:Polygon>
                <gml:exterior>
                    <gml:LinearRing>
                        <gml:posList>0.0 0.0 4.0 0.0 4.0 4.0 0.0 4.0 0.0 0.0</gml:posList>
                    </gml:LinearRing>
                </gml:exterior>
            </gml:Polygon>
        "#;
        let geom = parse_gml(gml).unwrap();

        assert_eq!(geom.geometry_type(), "Polygon");
        match &geom.geom {
            GeoGeometry::Polygon(p) => {
                assert_eq!(p.exterior().coords().count(), 5);
                assert_eq!(p.interiors().len(), 0);
            }
            _ => panic!("Expected Polygon"),
        }
    }

    #[test]
    fn test_parse_gml_polygon_with_hole() {
        let gml = r#"
            <gml:Polygon>
                <gml:exterior>
                    <gml:LinearRing>
                        <gml:posList>0.0 0.0 10.0 0.0 10.0 10.0 0.0 10.0 0.0 0.0</gml:posList>
                    </gml:LinearRing>
                </gml:exterior>
                <gml:interior>
                    <gml:LinearRing>
                        <gml:posList>2.0 2.0 8.0 2.0 8.0 8.0 2.0 8.0 2.0 2.0</gml:posList>
                    </gml:LinearRing>
                </gml:interior>
            </gml:Polygon>
        "#;
        let geom = parse_gml(gml).unwrap();

        assert_eq!(geom.geometry_type(), "Polygon");
        match &geom.geom {
            GeoGeometry::Polygon(p) => {
                assert_eq!(p.exterior().coords().count(), 5);
                assert_eq!(p.interiors().len(), 1);
                assert_eq!(p.interiors()[0].coords().count(), 5);
            }
            _ => panic!("Expected Polygon"),
        }
    }

    #[test]
    fn test_roundtrip_polygon() {
        let original = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
        let gml = geometry_to_gml(&original).unwrap();
        let parsed = parse_gml(&gml).unwrap();

        assert_eq!(parsed.geometry_type(), "Polygon");
    }

    #[test]
    fn test_roundtrip_polygon_with_hole() {
        let original =
            Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))")
                .unwrap();
        let gml = geometry_to_gml(&original).unwrap();
        let parsed = parse_gml(&gml).unwrap();

        assert_eq!(parsed.geometry_type(), "Polygon");
        match &parsed.geom {
            GeoGeometry::Polygon(p) => {
                assert_eq!(p.interiors().len(), 1);
            }
            _ => panic!("Expected Polygon with hole"),
        }
    }

    #[test]
    fn test_parse_gml_multipoint() {
        let gml = r#"
            <gml:MultiPoint>
                <gml:pointMember>
                    <gml:Point><gml:pos>1.0 2.0</gml:pos></gml:Point>
                </gml:pointMember>
                <gml:pointMember>
                    <gml:Point><gml:pos>3.0 4.0</gml:pos></gml:Point>
                </gml:pointMember>
            </gml:MultiPoint>
        "#;
        let geom = parse_gml(gml).unwrap();

        assert_eq!(geom.geometry_type(), "MultiPoint");
        match &geom.geom {
            GeoGeometry::MultiPoint(mp) => {
                assert_eq!(mp.0.len(), 2);
            }
            _ => panic!("Expected MultiPoint"),
        }
    }

    #[test]
    fn test_roundtrip_multipoint() {
        let original = Geometry::from_wkt("MULTIPOINT((1 2), (3 4))").unwrap();
        let gml = geometry_to_gml(&original).unwrap();
        let parsed = parse_gml(&gml).unwrap();

        assert_eq!(parsed.geometry_type(), "MultiPoint");
    }

    #[test]
    fn test_parse_gml_multilinestring() {
        let gml = r#"
            <gml:MultiCurve>
                <gml:curveMember>
                    <gml:LineString><gml:posList>0.0 0.0 1.0 1.0</gml:posList></gml:LineString>
                </gml:curveMember>
                <gml:curveMember>
                    <gml:LineString><gml:posList>2.0 2.0 3.0 3.0</gml:posList></gml:LineString>
                </gml:curveMember>
            </gml:MultiCurve>
        "#;
        let geom = parse_gml(gml).unwrap();

        assert_eq!(geom.geometry_type(), "MultiLineString");
        match &geom.geom {
            GeoGeometry::MultiLineString(mls) => {
                assert_eq!(mls.0.len(), 2);
            }
            _ => panic!("Expected MultiLineString"),
        }
    }

    #[test]
    fn test_roundtrip_multilinestring() {
        let original = Geometry::from_wkt("MULTILINESTRING((0 0, 1 1), (2 2, 3 3))").unwrap();
        let gml = geometry_to_gml(&original).unwrap();
        let parsed = parse_gml(&gml).unwrap();

        assert_eq!(parsed.geometry_type(), "MultiLineString");
    }

    #[test]
    fn test_parse_gml_multipolygon() {
        let gml = r#"
            <gml:MultiSurface>
                <gml:surfaceMember>
                    <gml:Polygon>
                        <gml:exterior>
                            <gml:LinearRing>
                                <gml:posList>0.0 0.0 2.0 0.0 2.0 2.0 0.0 2.0 0.0 0.0</gml:posList>
                            </gml:LinearRing>
                        </gml:exterior>
                    </gml:Polygon>
                </gml:surfaceMember>
                <gml:surfaceMember>
                    <gml:Polygon>
                        <gml:exterior>
                            <gml:LinearRing>
                                <gml:posList>3.0 3.0 5.0 3.0 5.0 5.0 3.0 5.0 3.0 3.0</gml:posList>
                            </gml:LinearRing>
                        </gml:exterior>
                    </gml:Polygon>
                </gml:surfaceMember>
            </gml:MultiSurface>
        "#;
        let geom = parse_gml(gml).unwrap();

        assert_eq!(geom.geometry_type(), "MultiPolygon");
        match &geom.geom {
            GeoGeometry::MultiPolygon(mp) => {
                assert_eq!(mp.0.len(), 2);
            }
            _ => panic!("Expected MultiPolygon"),
        }
    }

    #[test]
    fn test_roundtrip_multipolygon() {
        let original = Geometry::from_wkt(
            "MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 3, 5 3, 5 5, 3 5, 3 3)))",
        )
        .unwrap();
        let gml = geometry_to_gml(&original).unwrap();
        let parsed = parse_gml(&gml).unwrap();

        assert_eq!(parsed.geometry_type(), "MultiPolygon");
    }

    #[test]
    fn test_gml_with_crs() {
        use crate::geometry::Crs;

        // Use EPSG:3857 (Web Mercator) which is not a default CRS
        let geom = Geometry::with_crs(
            GeoGeometry::Point(geo_types::Point::new(15548711.0, 4234937.0)),
            Crs::epsg(3857),
        );

        let gml = geometry_to_gml(&geom).unwrap();
        // Verify that GML contains srsName attribute for non-default CRS
        assert!(
            gml.contains("srsName"),
            "GML should contain srsName for non-default CRS: {}",
            gml
        );
        assert!(
            gml.contains("EPSG/0/3857"),
            "GML should contain EPSG code in srsName: {}",
            gml
        );

        // Parse back and verify CRS is preserved
        let parsed = parse_gml(&gml).unwrap();
        assert_eq!(parsed.crs.epsg_code(), Some(3857));
    }

    #[test]
    fn test_parse_gml_invalid_point() {
        let gml = r#"<gml:Point><gml:pos>1.0</gml:pos></gml:Point>"#;
        let result = parse_gml(gml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_gml_invalid_linestring() {
        let gml = r#"<gml:LineString><gml:posList>0.0 0.0 1.0</gml:posList></gml:LineString>"#;
        let result = parse_gml(gml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_gml_empty() {
        let gml = r#"<gml:Empty></gml:Empty>"#;
        let result = parse_gml(gml);
        assert!(result.is_err());
    }

    #[test]
    fn test_geometry_to_gml_unsupported() {
        use geo_types::GeometryCollection;

        let geom = Geometry::new(GeoGeometry::GeometryCollection(
            GeometryCollection::new_from(vec![]),
        ));

        let result = geometry_to_gml(&geom);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not supported"));
    }
}
