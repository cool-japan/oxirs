//! GPX (GPS Exchange Format) parser and serializer
//!
//! Converts between GPX XML strings and geometry objects.
//! Supports GPX 1.0 and GPX 1.1 specifications.
//!
//! GPX contains waypoints, tracks, and routes which map to Point and LineString geometries.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};
use geo_types::Geometry as GeoGeometry;

#[cfg(feature = "gpx-support")]
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
#[cfg(feature = "gpx-support")]
use quick_xml::{Reader, Writer};
#[cfg(feature = "gpx-support")]
use std::io::Cursor;

/// Parse a GPX string into a Geometry
///
/// # GPX Format
/// GPX uses lat/lon attributes on waypoints and track/route points.
/// This function extracts the first geometry found (waypoint, track, or route).
///
/// # Examples
/// ```
/// # #[cfg(feature = "gpx-support")]
/// # {
/// use oxirs_geosparql::geometry::gpx_parser::parse_gpx;
///
/// let gpx = r#"
///   <wpt lat="37.422" lon="-122.084">
///     <name>Mountain View</name>
///   </wpt>
/// "#;
///
/// let geometry = parse_gpx(gpx).unwrap();
/// # }
/// ```
#[cfg(feature = "gpx-support")]
pub fn parse_gpx(gpx_str: &str) -> Result<Geometry> {
    let mut reader = Reader::from_str(gpx_str);
    reader.config_mut().trim_text(true);

    let crs = Crs::new("http://www.opengis.net/def/crs/EPSG/0/4326".to_string()); // GPX is always WGS84
    let mut geometry: Option<GeoGeometry<f64>> = None;

    // Track/route building
    let mut current_track_points: Vec<geo_types::Coord<f64>> = Vec::new();
    let mut in_track_segment = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                match name.as_str() {
                    "wpt" | "gpx:wpt" => {
                        // Waypoint (point)
                        if let Some(point) = parse_gpx_point(&e) {
                            if geometry.is_none() {
                                geometry = Some(GeoGeometry::Point(point));
                            }
                        }
                    }
                    "trkseg" | "gpx:trkseg" => {
                        // Track segment start
                        in_track_segment = true;
                        current_track_points.clear();
                    }
                    "trkpt" | "gpx:trkpt" => {
                        // Track point
                        if in_track_segment {
                            if let Some(point) = parse_gpx_point(&e) {
                                current_track_points.push(geo_types::Coord {
                                    x: point.x(),
                                    y: point.y(),
                                });
                            }
                        }
                    }
                    "rtept" | "gpx:rtept" => {
                        // Route point (treat route as linestring)
                        if let Some(point) = parse_gpx_point(&e) {
                            current_track_points.push(geo_types::Coord {
                                x: point.x(),
                                y: point.y(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                match name.as_str() {
                    "trkseg" | "gpx:trkseg" => {
                        // Track segment end - create linestring
                        if !current_track_points.is_empty() && geometry.is_none() {
                            geometry = Some(GeoGeometry::LineString(geo_types::LineString(
                                current_track_points.clone(),
                            )));
                        }
                        in_track_segment = false;
                    }
                    "rte" | "gpx:rte" => {
                        // Route end - create linestring
                        if !current_track_points.is_empty() && geometry.is_none() {
                            geometry = Some(GeoGeometry::LineString(geo_types::LineString(
                                current_track_points.clone(),
                            )));
                        }
                        current_track_points.clear();
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
        .ok_or_else(|| GeoSparqlError::ParseError("No geometry found in GPX".to_string()))
}

/// Parse GPX point from element attributes (lat/lon)
#[cfg(feature = "gpx-support")]
fn parse_gpx_point(element: &BytesStart) -> Option<geo_types::Point<f64>> {
    let mut lat: Option<f64> = None;
    let mut lon: Option<f64> = None;

    for attr in element.attributes().flatten() {
        let key = String::from_utf8_lossy(attr.key.as_ref());
        let value = String::from_utf8_lossy(&attr.value);

        match key.as_ref() {
            "lat" => lat = value.parse::<f64>().ok(),
            "lon" => lon = value.parse::<f64>().ok(),
            _ => {}
        }
    }

    match (lat, lon) {
        (Some(lat_val), Some(lon_val)) => {
            // GPX uses lat/lon, but geo_types::Point is (x=lon, y=lat)
            Some(geo_types::Point::new(lon_val, lat_val))
        }
        _ => None,
    }
}

/// Convert a Geometry to GPX string
///
/// # Examples
/// ```
/// # #[cfg(feature = "gpx-support")]
/// # {
/// use oxirs_geosparql::geometry::{Geometry, Crs};
/// use oxirs_geosparql::geometry::gpx_parser::geometry_to_gpx;
/// use geo_types::{Point, Geometry as GeoGeometry};
///
/// let point = Point::new(-122.084, 37.422);
/// let geometry = Geometry::new(GeoGeometry::Point(point));
/// let gpx = geometry_to_gpx(&geometry, Some("My Location")).unwrap();
/// # }
/// ```
#[cfg(feature = "gpx-support")]
pub fn geometry_to_gpx(geometry: &Geometry, name: Option<&str>) -> Result<String> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    // Write GPX header
    let mut gpx_start = BytesStart::new("gpx");
    gpx_start.push_attribute(("version", "1.1"));
    gpx_start.push_attribute(("creator", "oxirs-geosparql"));
    gpx_start.push_attribute(("xmlns", "http://www.topografix.com/GPX/1/1"));
    writer
        .write_event(Event::Start(gpx_start))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    match &geometry.geom {
        GeoGeometry::Point(point) => write_gpx_waypoint(&mut writer, point, name)?,
        GeoGeometry::LineString(linestring) => write_gpx_track(&mut writer, linestring, name)?,
        GeoGeometry::MultiPoint(multi_point) => {
            for (i, point) in multi_point.iter().enumerate() {
                let point_name = name.map(|n| format!("{} {}", n, i + 1));
                write_gpx_waypoint(&mut writer, point, point_name.as_deref())?;
            }
        }
        _ => {
            return Err(GeoSparqlError::SerializationError(
                "Only Point, LineString, and MultiPoint are supported for GPX".to_string(),
            ))
        }
    }

    // Write GPX footer
    writer
        .write_event(Event::End(BytesEnd::new("gpx")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    let result = writer.into_inner().into_inner();
    String::from_utf8(result)
        .map_err(|e| GeoSparqlError::SerializationError(format!("UTF-8 conversion error: {}", e)))
}

#[cfg(feature = "gpx-support")]
fn write_gpx_waypoint<W: std::io::Write>(
    writer: &mut Writer<W>,
    point: &geo_types::Point<f64>,
    name: Option<&str>,
) -> Result<()> {
    let mut wpt_start = BytesStart::new("wpt");
    wpt_start.push_attribute(("lat", point.y().to_string().as_str()));
    wpt_start.push_attribute(("lon", point.x().to_string().as_str()));

    writer
        .write_event(Event::Start(wpt_start))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    if let Some(name_str) = name {
        writer
            .write_event(Event::Start(BytesStart::new("name")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
        writer
            .write_event(Event::Text(BytesText::new(name_str)))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
        writer
            .write_event(Event::End(BytesEnd::new("name")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("wpt")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(feature = "gpx-support")]
fn write_gpx_track<W: std::io::Write>(
    writer: &mut Writer<W>,
    linestring: &geo_types::LineString<f64>,
    name: Option<&str>,
) -> Result<()> {
    writer
        .write_event(Event::Start(BytesStart::new("trk")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    if let Some(name_str) = name {
        writer
            .write_event(Event::Start(BytesStart::new("name")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
        writer
            .write_event(Event::Text(BytesText::new(name_str)))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
        writer
            .write_event(Event::End(BytesEnd::new("name")))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::Start(BytesStart::new("trkseg")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    for coord in linestring.coords() {
        let mut trkpt_start = BytesStart::new("trkpt");
        trkpt_start.push_attribute(("lat", coord.y.to_string().as_str()));
        trkpt_start.push_attribute(("lon", coord.x.to_string().as_str()));

        writer
            .write_event(Event::Empty(trkpt_start))
            .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("trkseg")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    writer
        .write_event(Event::End(BytesEnd::new("trk")))
        .map_err(|e| GeoSparqlError::SerializationError(format!("XML write error: {}", e)))?;

    Ok(())
}

#[cfg(test)]
#[cfg(feature = "gpx-support")]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use geo_types::{LineString, Point};

    #[test]
    fn test_parse_gpx_waypoint() {
        let gpx = r#"
            <wpt lat="37.422" lon="-122.084">
                <name>Mountain View</name>
            </wpt>
        "#;

        let geometry = parse_gpx(gpx).unwrap();
        match &geometry.geom {
            GeoGeometry::Point(point) => {
                assert_relative_eq!(point.x(), -122.084, epsilon = 1e-6);
                assert_relative_eq!(point.y(), 37.422, epsilon = 1e-6);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_parse_gpx_track() {
        let gpx = r#"
            <trk>
                <name>Test Track</name>
                <trkseg>
                    <trkpt lat="37.422" lon="-122.084"/>
                    <trkpt lat="37.423" lon="-122.085"/>
                    <trkpt lat="37.424" lon="-122.086"/>
                </trkseg>
            </trk>
        "#;

        let geometry = parse_gpx(gpx).unwrap();
        match &geometry.geom {
            GeoGeometry::LineString(linestring) => {
                assert_eq!(linestring.coords().count(), 3);
                let coords: Vec<_> = linestring.coords().collect();
                assert_relative_eq!(coords[0].x, -122.084, epsilon = 1e-6);
                assert_relative_eq!(coords[0].y, 37.422, epsilon = 1e-6);
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_parse_gpx_route() {
        let gpx = r#"
            <rte>
                <name>Test Route</name>
                <rtept lat="37.422" lon="-122.084"/>
                <rtept lat="37.423" lon="-122.085"/>
            </rte>
        "#;

        let geometry = parse_gpx(gpx).unwrap();
        match &geometry.geom {
            GeoGeometry::LineString(linestring) => {
                assert_eq!(linestring.coords().count(), 2);
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_geometry_to_gpx_waypoint() {
        let point = Point::new(-122.084, 37.422);
        let geometry = Geometry::new(GeoGeometry::Point(point));
        let gpx = geometry_to_gpx(&geometry, Some("Test Point")).unwrap();

        assert!(gpx.contains("<gpx"));
        assert!(gpx.contains("<wpt"));
        assert!(gpx.contains("lat=\"37.422\""));
        assert!(gpx.contains("lon=\"-122.084\""));
        assert!(gpx.contains("<name>Test Point</name>"));
    }

    #[test]
    fn test_geometry_to_gpx_track() {
        let linestring = LineString::from(vec![
            (-122.084, 37.422),
            (-122.085, 37.423),
            (-122.086, 37.424),
        ]);
        let geometry = Geometry::new(GeoGeometry::LineString(linestring));
        let gpx = geometry_to_gpx(&geometry, Some("Test Track")).unwrap();

        assert!(gpx.contains("<trk>"));
        assert!(gpx.contains("<trkseg>"));
        assert!(gpx.contains("<trkpt"));
        assert!(gpx.contains("<name>Test Track</name>"));
    }

    #[test]
    fn test_roundtrip_waypoint() {
        let point = Point::new(-122.084, 37.422);
        let geometry = Geometry::new(GeoGeometry::Point(point));
        let gpx = geometry_to_gpx(&geometry, None).unwrap();
        let parsed = parse_gpx(&gpx).unwrap();

        match &parsed.geom {
            GeoGeometry::Point(p) => {
                assert_relative_eq!(p.x(), point.x(), epsilon = 1e-6);
                assert_relative_eq!(p.y(), point.y(), epsilon = 1e-6);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_roundtrip_track() {
        let linestring = LineString::from(vec![
            (-122.084, 37.422),
            (-122.085, 37.423),
            (-122.086, 37.424),
        ]);
        let geometry = Geometry::new(GeoGeometry::LineString(linestring.clone()));
        let gpx = geometry_to_gpx(&geometry, None).unwrap();
        let parsed = parse_gpx(&gpx).unwrap();

        match &parsed.geom {
            GeoGeometry::LineString(ls) => {
                assert_eq!(ls.coords().count(), linestring.coords().count());
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_crs_always_wgs84() {
        let gpx = r#"<wpt lat="37.422" lon="-122.084"/>"#;
        let geometry = parse_gpx(gpx).unwrap();
        assert_eq!(
            geometry.crs.uri,
            "http://www.opengis.net/def/crs/EPSG/0/4326"
        );
    }

    #[test]
    fn test_parse_gpx_empty() {
        let gpx = "<gpx></gpx>";
        let result = parse_gpx(gpx);
        assert!(result.is_err());
    }
}
