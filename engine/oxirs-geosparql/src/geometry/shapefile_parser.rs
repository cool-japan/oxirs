//! Shapefile (ESRI) Format Parser
//!
//! This module provides reading support for ESRI Shapefiles,
//! the most widely used vector data format in GIS applications.
//!
//! # Shapefile Format
//!
//! A shapefile consists of multiple files:
//! - .shp: Main file containing geometry
//! - .shx: Index file
//! - .dbf: Attribute database (dBASE format)
//! - .prj: Projection information (optional)
//!
//! # Supported Geometry Types
//!
//! - Point, PointM, PointZ
//! - MultiPoint, MultiPointM, MultiPointZ
//! - PolyLine (LineString), PolyLineM, PolyLineZ
//! - Polygon, PolygonM, PolygonZ
//!
//! # Coordinate Reference Systems
//!
//! The parser attempts to extract CRS information from the .prj file.
//! If no .prj file exists, it defaults to WGS84 (EPSG:4326).
//!
//! # Example
//!
//! ```no_run
//! use oxirs_geosparql::geometry::shapefile_parser::read_shapefile;
//!
//! let geometries = read_shapefile("data/cities.shp").unwrap();
//! for geometry in geometries {
//!     println!("Geometry: {}", geometry.to_wkt());
//! }
//! ```

use crate::error::{GeoSparqlError, Result};
use crate::geometry::{Crs, Geometry};
use geo_types::{
    Coord, Geometry as GeoGeometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use std::path::Path;

#[cfg(feature = "shapefile-support")]
use shapefile;

/// Read geometries from a shapefile
///
/// This function reads all geometries from a shapefile. The path should point
/// to the .shp file - the associated .shx and .dbf files will be automatically
/// located in the same directory.
///
/// # Arguments
///
/// * `path` - Path to the .shp file
///
/// # Returns
///
/// A vector of Geometry objects, one for each shape in the file
///
/// # Example
///
/// ```no_run
/// use oxirs_geosparql::geometry::shapefile_parser::read_shapefile;
///
/// let geometries = read_shapefile("data/cities.shp").unwrap();
/// println!("Read {} geometries", geometries.len());
/// ```
#[cfg(feature = "shapefile-support")]
pub fn read_shapefile<P: AsRef<Path>>(path: P) -> Result<Vec<Geometry>> {
    let path = path.as_ref();

    // Extract CRS from .prj file if it exists
    let crs = read_prj_file(path)?;

    // Use ShapeReader for reading shapes only (no attributes)
    let mut reader = shapefile::ShapeReader::from_path(path)
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to open shapefile: {}", e)))?;

    let mut geometries = Vec::new();

    // Iterate over shapes
    for shape_result in reader.iter_shapes() {
        let shape = shape_result
            .map_err(|e| GeoSparqlError::ParseError(format!("Failed to read shape: {}", e)))?;

        if let Some(geometry) = shape_to_geometry(shape, &crs)? {
            geometries.push(geometry);
        }
    }

    Ok(geometries)
}

/// Read CRS information from a .prj file
///
/// Attempts to read the projection file associated with a shapefile.
/// Returns WGS84 if no .prj file is found.
#[cfg(feature = "shapefile-support")]
fn read_prj_file<P: AsRef<Path>>(shp_path: P) -> Result<Crs> {
    let shp_path = shp_path.as_ref();
    let prj_path = shp_path.with_extension("prj");

    if prj_path.exists() {
        match std::fs::read_to_string(&prj_path) {
            Ok(wkt_crs) => {
                // Try to parse the WKT CRS and extract EPSG code if possible
                let crs_uri = parse_prj_to_uri(&wkt_crs);
                Ok(Crs::new(crs_uri))
            }
            Err(_) => {
                // If we can't read the .prj file, default to WGS84
                Ok(Crs::wgs84())
            }
        }
    } else {
        // No .prj file, default to WGS84
        Ok(Crs::wgs84())
    }
}

/// Parse a .prj file (WKT format) to a CRS URI
///
/// This is a simplified parser that looks for common EPSG codes in the WKT.
#[cfg(feature = "shapefile-support")]
fn parse_prj_to_uri(wkt: &str) -> String {
    // Look for AUTHORITY["EPSG","4326"] pattern
    // The pattern AUTHORITY["EPSG"," is 18 characters long
    if let Some(start) = wkt.find("AUTHORITY[\"EPSG\",\"") {
        let remaining = &wkt[start + 18..];
        if let Some(end) = remaining.find('\"') {
            let epsg_code = &remaining[..end];
            return format!("http://www.opengis.net/def/crs/EPSG/0/{}", epsg_code);
        }
    }

    // Check for common CRS names
    if wkt.contains("WGS 84") || wkt.contains("WGS84") {
        return "http://www.opengis.net/def/crs/EPSG/0/4326".to_string();
    }

    if wkt.contains("WGS_1984_Web_Mercator") || wkt.contains("Web Mercator") {
        return "http://www.opengis.net/def/crs/EPSG/0/3857".to_string();
    }

    // Default to WGS84
    "http://www.opengis.net/def/crs/EPSG/0/4326".to_string()
}

/// Convert a shapefile shape to a Geometry
#[cfg(feature = "shapefile-support")]
fn shape_to_geometry(shape: shapefile::Shape, crs: &Crs) -> Result<Option<Geometry>> {
    use shapefile::Shape as ShpShape;

    let geo_geom = match shape {
        // Point types
        ShpShape::Point(point) => {
            let pt = Point::new(point.x, point.y);
            GeoGeometry::Point(pt)
        }
        ShpShape::PointM(point) => {
            let pt = Point::new(point.x, point.y);
            GeoGeometry::Point(pt)
        }
        ShpShape::PointZ(point) => {
            // Note: geo-types doesn't support 3D coordinates natively,
            // so we drop the Z coordinate for now
            let pt = Point::new(point.x, point.y);
            GeoGeometry::Point(pt)
        }

        // MultiPoint types
        ShpShape::Multipoint(multipoint) => {
            let points: Vec<Point<f64>> = multipoint
                .points()
                .iter()
                .map(|p| Point::new(p.x, p.y))
                .collect();
            GeoGeometry::MultiPoint(MultiPoint(points))
        }
        ShpShape::MultipointM(multipoint) => {
            let points: Vec<Point<f64>> = multipoint
                .points()
                .iter()
                .map(|p| Point::new(p.x, p.y))
                .collect();
            GeoGeometry::MultiPoint(MultiPoint(points))
        }
        ShpShape::MultipointZ(multipoint) => {
            let points: Vec<Point<f64>> = multipoint
                .points()
                .iter()
                .map(|p| Point::new(p.x, p.y))
                .collect();
            GeoGeometry::MultiPoint(MultiPoint(points))
        }

        // PolyLine (LineString) types
        ShpShape::Polyline(polyline) => {
            let parts = polyline.parts();
            if parts.is_empty() {
                return Ok(None);
            } else if parts.len() == 1 {
                // Single LineString
                let coords: Vec<Coord<f64>> =
                    parts[0].iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                GeoGeometry::LineString(LineString(coords))
            } else {
                // MultiLineString
                let lines: Vec<LineString<f64>> = parts
                    .iter()
                    .map(|part| {
                        let coords: Vec<Coord<f64>> =
                            part.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                        LineString(coords)
                    })
                    .collect();
                GeoGeometry::MultiLineString(MultiLineString(lines))
            }
        }
        ShpShape::PolylineM(polyline) => {
            let parts = polyline.parts();
            if parts.is_empty() {
                return Ok(None);
            } else if parts.len() == 1 {
                let coords: Vec<Coord<f64>> =
                    parts[0].iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                GeoGeometry::LineString(LineString(coords))
            } else {
                let lines: Vec<LineString<f64>> = parts
                    .iter()
                    .map(|part| {
                        let coords: Vec<Coord<f64>> =
                            part.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                        LineString(coords)
                    })
                    .collect();
                GeoGeometry::MultiLineString(MultiLineString(lines))
            }
        }
        ShpShape::PolylineZ(polyline) => {
            let parts = polyline.parts();
            if parts.is_empty() {
                return Ok(None);
            } else if parts.len() == 1 {
                let coords: Vec<Coord<f64>> =
                    parts[0].iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                GeoGeometry::LineString(LineString(coords))
            } else {
                let lines: Vec<LineString<f64>> = parts
                    .iter()
                    .map(|part| {
                        let coords: Vec<Coord<f64>> =
                            part.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
                        LineString(coords)
                    })
                    .collect();
                GeoGeometry::MultiLineString(MultiLineString(lines))
            }
        }

        // Polygon types
        ShpShape::Polygon(polygon) => {
            let polygons = polygon_rings_to_polygons(polygon.rings())?;

            if polygons.is_empty() {
                return Ok(None);
            } else if polygons.len() == 1 {
                GeoGeometry::Polygon(polygons.into_iter().next().unwrap())
            } else {
                GeoGeometry::MultiPolygon(MultiPolygon(polygons))
            }
        }
        ShpShape::PolygonM(polygon) => {
            let polygons = polygon_rings_to_polygons_m(polygon.rings())?;

            if polygons.is_empty() {
                return Ok(None);
            } else if polygons.len() == 1 {
                GeoGeometry::Polygon(polygons.into_iter().next().unwrap())
            } else {
                GeoGeometry::MultiPolygon(MultiPolygon(polygons))
            }
        }
        ShpShape::PolygonZ(polygon) => {
            let polygons = polygon_rings_to_polygons_z(polygon.rings())?;

            if polygons.is_empty() {
                return Ok(None);
            } else if polygons.len() == 1 {
                GeoGeometry::Polygon(polygons.into_iter().next().unwrap())
            } else {
                GeoGeometry::MultiPolygon(MultiPolygon(polygons))
            }
        }

        // Null shape
        ShpShape::NullShape => return Ok(None),

        // Multipatch is not supported by geo-types
        ShpShape::Multipatch(_) => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "Multipatch geometry type is not supported".to_string(),
            ));
        }
    };

    Ok(Some(Geometry::with_crs(geo_geom, crs.clone())))
}

/// Convert shapefile polygon rings (Point type) to geo-types polygons
#[cfg(feature = "shapefile-support")]
fn polygon_rings_to_polygons(
    rings: &[shapefile::PolygonRing<shapefile::Point>],
) -> Result<Vec<Polygon<f64>>> {
    let mut polygons = Vec::new();
    let mut current_exterior: Option<LineString<f64>> = None;
    let mut current_holes: Vec<LineString<f64>> = Vec::new();

    for ring in rings {
        let coords: Vec<Coord<f64>> = ring
            .points()
            .iter()
            .map(|p| Coord { x: p.x, y: p.y })
            .collect();
        let linestring = LineString(coords);

        match ring {
            shapefile::PolygonRing::Outer(_) => {
                // This is an exterior ring
                // If we have a current polygon, save it
                if let Some(exterior) = current_exterior.take() {
                    polygons.push(Polygon::new(exterior, std::mem::take(&mut current_holes)));
                }
                // Start new polygon
                current_exterior = Some(linestring);
            }
            shapefile::PolygonRing::Inner(_) => {
                // This is a hole
                current_holes.push(linestring);
            }
        }
    }

    // Don't forget the last polygon
    if let Some(exterior) = current_exterior {
        polygons.push(Polygon::new(exterior, current_holes));
    }

    Ok(polygons)
}

/// Convert shapefile polygon rings (PointM type) to geo-types polygons
#[cfg(feature = "shapefile-support")]
fn polygon_rings_to_polygons_m(
    rings: &[shapefile::PolygonRing<shapefile::PointM>],
) -> Result<Vec<Polygon<f64>>> {
    let mut polygons = Vec::new();
    let mut current_exterior: Option<LineString<f64>> = None;
    let mut current_holes: Vec<LineString<f64>> = Vec::new();

    for ring in rings {
        let coords: Vec<Coord<f64>> = ring
            .points()
            .iter()
            .map(|p| Coord { x: p.x, y: p.y })
            .collect();
        let linestring = LineString(coords);

        match ring {
            shapefile::PolygonRing::Outer(_) => {
                if let Some(exterior) = current_exterior.take() {
                    polygons.push(Polygon::new(exterior, std::mem::take(&mut current_holes)));
                }
                current_exterior = Some(linestring);
            }
            shapefile::PolygonRing::Inner(_) => {
                current_holes.push(linestring);
            }
        }
    }

    if let Some(exterior) = current_exterior {
        polygons.push(Polygon::new(exterior, current_holes));
    }

    Ok(polygons)
}

/// Convert shapefile polygon rings (PointZ type) to geo-types polygons
#[cfg(feature = "shapefile-support")]
fn polygon_rings_to_polygons_z(
    rings: &[shapefile::PolygonRing<shapefile::PointZ>],
) -> Result<Vec<Polygon<f64>>> {
    let mut polygons = Vec::new();
    let mut current_exterior: Option<LineString<f64>> = None;
    let mut current_holes: Vec<LineString<f64>> = Vec::new();

    for ring in rings {
        let coords: Vec<Coord<f64>> = ring
            .points()
            .iter()
            .map(|p| Coord { x: p.x, y: p.y })
            .collect();
        let linestring = LineString(coords);

        match ring {
            shapefile::PolygonRing::Outer(_) => {
                if let Some(exterior) = current_exterior.take() {
                    polygons.push(Polygon::new(exterior, std::mem::take(&mut current_holes)));
                }
                current_exterior = Some(linestring);
            }
            shapefile::PolygonRing::Inner(_) => {
                current_holes.push(linestring);
            }
        }
    }

    if let Some(exterior) = current_exterior {
        polygons.push(Polygon::new(exterior, current_holes));
    }

    Ok(polygons)
}

#[cfg(all(test, feature = "shapefile-support"))]
mod tests {
    use super::*;

    #[test]
    fn test_parse_prj_to_uri_wgs84() {
        let wkt = r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],AUTHORITY["EPSG","4326"]]"#;
        let uri = parse_prj_to_uri(wkt);
        assert_eq!(uri, "http://www.opengis.net/def/crs/EPSG/0/4326");
    }

    #[test]
    fn test_parse_prj_to_uri_web_mercator() {
        let wkt = r#"PROJCS["WGS_1984_Web_Mercator",AUTHORITY["EPSG","3857"]]"#;
        let uri = parse_prj_to_uri(wkt);
        assert_eq!(uri, "http://www.opengis.net/def/crs/EPSG/0/3857");
    }
}
