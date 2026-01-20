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
//! let geometries = read_shapefile("data/cities.shp").expect("shapefile reading should succeed");
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
/// let geometries = read_shapefile("data/cities.shp").expect("shapefile reading should succeed");
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
                GeoGeometry::Polygon(
                    polygons
                        .into_iter()
                        .next()
                        .expect("polygon vec should have exactly 1 element"),
                )
            } else {
                GeoGeometry::MultiPolygon(MultiPolygon(polygons))
            }
        }
        ShpShape::PolygonM(polygon) => {
            let polygons = polygon_rings_to_polygons_m(polygon.rings())?;

            if polygons.is_empty() {
                return Ok(None);
            } else if polygons.len() == 1 {
                GeoGeometry::Polygon(
                    polygons
                        .into_iter()
                        .next()
                        .expect("polygon vec should have exactly 1 element"),
                )
            } else {
                GeoGeometry::MultiPolygon(MultiPolygon(polygons))
            }
        }
        ShpShape::PolygonZ(polygon) => {
            let polygons = polygon_rings_to_polygons_z(polygon.rings())?;

            if polygons.is_empty() {
                return Ok(None);
            } else if polygons.len() == 1 {
                GeoGeometry::Polygon(
                    polygons
                        .into_iter()
                        .next()
                        .expect("polygon vec should have exactly 1 element"),
                )
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

/// Shapefile writing support
///
/// NOTE: The current implementation of shapefile writing requires further
/// investigation of the shapefile crate's Writer API. The reading functionality
/// is fully implemented and tested. Writing will be completed in a future version
/// with proper API compatibility.
///
/// For now, users can:
/// 1. Convert geometries to other formats (WKT, GeoJSON, GML, GeoPackage)
/// 2. Use those formats for data exchange
/// 3. Wait for the complete writing implementation
///
/// TODO: Complete shapefile writing implementation with proper shapefile crate API usage
#[cfg(feature = "shapefile-support")]
pub fn write_shapefile<P: AsRef<Path>>(path: P, geometries: &[Geometry]) -> Result<()> {
    if geometries.is_empty() {
        return Err(GeoSparqlError::InvalidGeometryType(
            "Cannot write empty geometry collection".to_string(),
        ));
    }

    let path = path.as_ref();

    // Determine the geometry type from the first geometry
    let first_geom = &geometries[0];

    // Check that all geometries have the same type and CRS
    let expected_crs = &first_geom.crs;
    for geom in geometries.iter() {
        if geom.crs != *expected_crs {
            return Err(GeoSparqlError::CrsIncompatibility(
                expected_crs.uri.clone(),
                geom.crs.uri.clone(),
            ));
        }
    }

    // Create shapefile writer based on geometry type
    match &first_geom.geom {
        GeoGeometry::Point(_) => write_point_shapefile(path, geometries),
        GeoGeometry::MultiPoint(_) => write_multipoint_shapefile(path, geometries),
        GeoGeometry::LineString(_) | GeoGeometry::MultiLineString(_) => {
            write_polyline_shapefile(path, geometries)
        }
        GeoGeometry::Polygon(_) | GeoGeometry::MultiPolygon(_) => {
            write_polygon_shapefile(path, geometries)
        }
        GeoGeometry::Line(_) | GeoGeometry::Triangle(_) | GeoGeometry::Rect(_) => {
            Err(GeoSparqlError::UnsupportedOperation(format!(
                "Geometry type {:?} is not supported for shapefile writing",
                first_geom.geom
            )))
        }
        GeoGeometry::GeometryCollection(_) => Err(GeoSparqlError::UnsupportedOperation(
            "GeometryCollection is not supported for shapefile writing. \
             Write each geometry type to a separate shapefile."
                .to_string(),
        )),
    }?;

    // Write .prj file with CRS information
    write_prj_file(path, &first_geom.crs)?;

    Ok(())
}

/// Write Point geometries to shapefile
fn write_point_shapefile<P: AsRef<Path>>(path: P, geometries: &[Geometry]) -> Result<()> {
    use shapefile::{Point as ShpPoint, ShapeWriter};

    let mut writer = ShapeWriter::from_path(path.as_ref()).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to create shapefile writer: {}", e))
    })?;

    for geom in geometries {
        match &geom.geom {
            GeoGeometry::Point(point) => {
                let shp_point = ShpPoint::new(point.x(), point.y());
                writer.write_shape(&shp_point).map_err(|e| {
                    GeoSparqlError::SerializationError(format!("Failed to write point: {}", e))
                })?;
            }
            _ => {
                return Err(GeoSparqlError::InvalidGeometryType(
                    "Mixed geometry types not allowed in shapefile".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Write MultiPoint geometries to shapefile
fn write_multipoint_shapefile<P: AsRef<Path>>(path: P, geometries: &[Geometry]) -> Result<()> {
    use shapefile::{Multipoint, Point as ShpPoint, ShapeWriter};

    let mut writer = ShapeWriter::from_path(path.as_ref()).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to create shapefile writer: {}", e))
    })?;

    for geom in geometries {
        match &geom.geom {
            GeoGeometry::MultiPoint(multipoint) => {
                let points: Vec<ShpPoint> = multipoint
                    .0
                    .iter()
                    .map(|p| ShpPoint::new(p.x(), p.y()))
                    .collect();
                let shp_multipoint = Multipoint::new(points);
                writer.write_shape(&shp_multipoint).map_err(|e| {
                    GeoSparqlError::SerializationError(format!("Failed to write multipoint: {}", e))
                })?;
            }
            _ => {
                return Err(GeoSparqlError::InvalidGeometryType(
                    "Mixed geometry types not allowed in shapefile".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Write LineString/MultiLineString geometries to shapefile as Polyline
fn write_polyline_shapefile<P: AsRef<Path>>(path: P, geometries: &[Geometry]) -> Result<()> {
    use shapefile::{Point as ShpPoint, Polyline, ShapeWriter};

    let mut writer = ShapeWriter::from_path(path.as_ref()).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to create shapefile writer: {}", e))
    })?;

    for geom in geometries {
        match &geom.geom {
            GeoGeometry::LineString(linestring) => {
                let points: Vec<ShpPoint> = linestring
                    .0
                    .iter()
                    .map(|c| ShpPoint::new(c.x, c.y))
                    .collect();
                let polyline = Polyline::new(points);
                writer.write_shape(&polyline).map_err(|e| {
                    GeoSparqlError::SerializationError(format!("Failed to write linestring: {}", e))
                })?;
            }
            GeoGeometry::MultiLineString(multilinestring) => {
                // Flatten to single polyline with multiple parts
                let mut all_points = Vec::new();
                for linestring in &multilinestring.0 {
                    let points: Vec<ShpPoint> = linestring
                        .0
                        .iter()
                        .map(|c| ShpPoint::new(c.x, c.y))
                        .collect();
                    all_points.extend(points);
                }
                let polyline = Polyline::new(all_points);
                writer.write_shape(&polyline).map_err(|e| {
                    GeoSparqlError::SerializationError(format!(
                        "Failed to write multilinestring: {}",
                        e
                    ))
                })?;
            }
            _ => {
                return Err(GeoSparqlError::InvalidGeometryType(
                    "Mixed geometry types not allowed in shapefile".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Write Polygon/MultiPolygon geometries to shapefile
fn write_polygon_shapefile<P: AsRef<Path>>(path: P, geometries: &[Geometry]) -> Result<()> {
    use shapefile::{Point as ShpPoint, Polygon as ShpPolygon, PolygonRing, ShapeWriter};

    let mut writer = ShapeWriter::from_path(path.as_ref()).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to create shapefile writer: {}", e))
    })?;

    for geom in geometries {
        match &geom.geom {
            GeoGeometry::Polygon(polygon) => {
                // For now, only write simple polygons without holes
                // TODO: Support interior rings when shapefile crate API is clarified
                let exterior_points: Vec<ShpPoint> = polygon
                    .exterior()
                    .0
                    .iter()
                    .map(|c| ShpPoint::new(c.x, c.y))
                    .collect();
                let shp_polygon = ShpPolygon::new(PolygonRing::Outer(exterior_points));

                writer.write_shape(&shp_polygon).map_err(|e| {
                    GeoSparqlError::SerializationError(format!("Failed to write polygon: {}", e))
                })?;
            }
            GeoGeometry::MultiPolygon(multipolygon) => {
                // Write each polygon as a separate shape
                for polygon in &multipolygon.0 {
                    // For now, only write simple polygons without holes
                    let exterior_points: Vec<ShpPoint> = polygon
                        .exterior()
                        .0
                        .iter()
                        .map(|c| ShpPoint::new(c.x, c.y))
                        .collect();
                    let shp_polygon = ShpPolygon::new(PolygonRing::Outer(exterior_points));

                    writer.write_shape(&shp_polygon).map_err(|e| {
                        GeoSparqlError::SerializationError(format!(
                            "Failed to write multipolygon part: {}",
                            e
                        ))
                    })?;
                }
            }
            _ => {
                return Err(GeoSparqlError::InvalidGeometryType(
                    "Mixed geometry types not allowed in shapefile".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Write .prj file with CRS information
fn write_prj_file<P: AsRef<Path>>(shp_path: P, crs: &Crs) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let shp_path = shp_path.as_ref();
    let prj_path = shp_path.with_extension("prj");

    // Convert CRS URI to WKT format for .prj file
    let prj_wkt = uri_to_prj_wkt(&crs.uri);

    let mut file = File::create(prj_path).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to create .prj file: {}", e))
    })?;

    file.write_all(prj_wkt.as_bytes()).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to write .prj file: {}", e))
    })?;

    Ok(())
}

/// Convert CRS URI to WKT format for .prj file
fn uri_to_prj_wkt(uri: &str) -> String {
    // Common CRS URIs to WKT mappings
    if uri.contains("EPSG/0/4326") || uri.contains("CRS84") {
        // WGS 84
        r#"GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]"#.to_string()
    } else if uri.contains("EPSG/0/3857") || uri.contains("Web_Mercator") {
        // Web Mercator
        r#"PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]"#.to_string()
    } else {
        // Default to WGS 84 for unknown CRS
        r#"GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]"#.to_string()
    }
}

/// Non-functional stub when shapefile-support feature is disabled
#[cfg(not(feature = "shapefile-support"))]
pub fn write_shapefile<P: AsRef<Path>>(_path: P, _geometries: &[Geometry]) -> Result<()> {
    Err(GeoSparqlError::UnsupportedOperation(
        "Shapefile writing requires the 'shapefile-support' feature to be enabled".to_string(),
    ))
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

    #[test]
    fn test_write_point_shapefile() {
        use std::env::temp_dir;
        use std::fs;

        let temp_path = temp_dir().join("test_points.shp");

        // Create test geometries
        let geometries = vec![
            Geometry::from_wkt("POINT(1.0 2.0)").unwrap(),
            Geometry::from_wkt("POINT(3.0 4.0)").unwrap(),
            Geometry::from_wkt("POINT(5.0 6.0)").unwrap(),
        ];

        // Write shapefile
        let result = write_shapefile(&temp_path, &geometries);
        assert!(
            result.is_ok(),
            "Failed to write point shapefile: {:?}",
            result.err()
        );

        // Verify files were created
        assert!(temp_path.exists(), "Shapefile not created");
        assert!(
            temp_path.with_extension("shx").exists(),
            "Index file not created"
        );
        assert!(
            temp_path.with_extension("prj").exists(),
            "PRJ file not created"
        );

        // Clean up
        let _ = fs::remove_file(&temp_path);
        let _ = fs::remove_file(temp_path.with_extension("shx"));
        let _ = fs::remove_file(temp_path.with_extension("prj"));
    }

    #[test]
    fn test_write_polygon_shapefile() {
        use std::env::temp_dir;
        use std::fs;

        let temp_path = temp_dir().join("test_polygons.shp");

        // Create test geometries
        let geometries = vec![
            Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap(),
            Geometry::from_wkt("POLYGON((10 10, 14 10, 14 14, 10 14, 10 10))").unwrap(),
        ];

        // Write shapefile
        let result = write_shapefile(&temp_path, &geometries);
        assert!(
            result.is_ok(),
            "Failed to write polygon shapefile: {:?}",
            result.err()
        );

        // Verify files were created
        assert!(temp_path.exists(), "Shapefile not created");
        assert!(
            temp_path.with_extension("shx").exists(),
            "Index file not created"
        );
        assert!(
            temp_path.with_extension("prj").exists(),
            "PRJ file not created"
        );

        // Clean up
        let _ = fs::remove_file(&temp_path);
        let _ = fs::remove_file(temp_path.with_extension("shx"));
        let _ = fs::remove_file(temp_path.with_extension("prj"));
    }

    #[test]
    fn test_write_linestring_shapefile() {
        use std::env::temp_dir;
        use std::fs;

        let temp_path = temp_dir().join("test_linestrings.shp");

        // Create test geometries
        let geometries = vec![
            Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2)").unwrap(),
            Geometry::from_wkt("LINESTRING(3 3, 4 4, 5 5)").unwrap(),
        ];

        // Write shapefile
        let result = write_shapefile(&temp_path, &geometries);
        assert!(
            result.is_ok(),
            "Failed to write linestring shapefile: {:?}",
            result.err()
        );

        // Verify files were created
        assert!(temp_path.exists(), "Shapefile not created");

        // Clean up
        let _ = fs::remove_file(&temp_path);
        let _ = fs::remove_file(temp_path.with_extension("shx"));
        let _ = fs::remove_file(temp_path.with_extension("prj"));
    }

    #[test]
    fn test_write_empty_geometries_fails() {
        use std::env::temp_dir;

        let temp_path = temp_dir().join("test_empty.shp");
        let geometries: Vec<Geometry> = vec![];

        // Should fail with empty collection
        let result = write_shapefile(&temp_path, &geometries);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeoSparqlError::InvalidGeometryType(_)
        ));
    }

    #[test]
    fn test_write_mixed_crs_fails() {
        use std::env::temp_dir;

        let temp_path = temp_dir().join("test_mixed_crs.shp");

        let geom1 = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
        let mut geom2 = Geometry::from_wkt("POINT(3.0 4.0)").unwrap();

        // Set different CRS
        geom2.crs = Crs::new("http://www.opengis.net/def/crs/EPSG/0/3857");

        let geometries = vec![geom1, geom2];

        // Should fail with mixed CRS
        let result = write_shapefile(&temp_path, &geometries);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeoSparqlError::CrsIncompatibility(_, _)
        ));
    }

    #[test]
    fn test_uri_to_prj_wkt() {
        // Test WGS84
        let wkt = uri_to_prj_wkt("http://www.opengis.net/def/crs/EPSG/0/4326");
        assert!(wkt.contains("GCS_WGS_1984"));

        // Test Web Mercator
        let wkt = uri_to_prj_wkt("http://www.opengis.net/def/crs/EPSG/0/3857");
        assert!(wkt.contains("Web_Mercator"));

        // Test unknown CRS defaults to WGS84
        let wkt = uri_to_prj_wkt("http://example.com/unknown");
        assert!(wkt.contains("GCS_WGS_1984"));
    }
}
