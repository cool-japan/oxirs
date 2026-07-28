//! PostGIS EWKB (Extended Well-Known Binary) Parser
//!
//! Implements parsing and serialization of PostGIS Extended Well-Known Binary format.
//! EWKB extends WKB with SRID information and support for Z/M coordinates.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::coord3d::Coord3D;
use crate::geometry::{Crs, Geometry};
use geo::algorithm::coords_iter::CoordsIter;
use geo_types::{
    Coord, Geometry as GeoGeometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use std::io::{Cursor, Read, Write};

/// Byte order marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little endian (0x01)
    LittleEndian,
    /// Big endian (0x00)
    BigEndian,
}

impl ByteOrder {
    fn from_byte(byte: u8) -> Result<Self> {
        match byte {
            0x00 => Ok(ByteOrder::BigEndian),
            0x01 => Ok(ByteOrder::LittleEndian),
            _ => Err(GeoSparqlError::ParseError(format!(
                "Invalid byte order marker: {}",
                byte
            ))),
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            ByteOrder::LittleEndian => 0x01,
            ByteOrder::BigEndian => 0x00,
        }
    }
}

/// EWKB geometry type codes
#[allow(dead_code)]
mod type_codes {
    pub const POINT: u32 = 1;
    pub const LINESTRING: u32 = 2;
    pub const POLYGON: u32 = 3;
    pub const MULTIPOINT: u32 = 4;
    pub const MULTILINESTRING: u32 = 5;
    pub const MULTIPOLYGON: u32 = 6;
    pub const GEOMETRY_COLLECTION: u32 = 7;

    // SRID flag
    pub const SRID_FLAG: u32 = 0x20000000;

    // Z and M flags
    pub const Z_FLAG: u32 = 0x80000000;
    pub const M_FLAG: u32 = 0x40000000;
}

/// Parse EWKB binary data into a Geometry
///
/// # Arguments
/// * `ewkb` - EWKB binary data as byte slice
///
/// # Returns
/// Parsed geometry with CRS information
///
/// # Example
/// ```ignore
/// use oxirs_geosparql::geometry::ewkb_parser::parse_ewkb;
///
/// let ewkb_data = vec![0x01, 0x01, 0x00, 0x00, 0x20, 0xe6, 0x10, 0x00, 0x00, /* ... */];
/// let geometry = parse_ewkb(&ewkb_data).expect("should succeed");
/// ```
pub fn parse_ewkb(ewkb: &[u8]) -> Result<Geometry> {
    let mut cursor = Cursor::new(ewkb);
    parse_ewkb_geometry(&mut cursor)
}

fn parse_ewkb_geometry(cursor: &mut Cursor<&[u8]>) -> Result<Geometry> {
    // Read byte order
    let mut byte_order_byte = [0u8; 1];
    cursor
        .read_exact(&mut byte_order_byte)
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to read byte order: {}", e)))?;
    let byte_order = ByteOrder::from_byte(byte_order_byte[0])?;

    // Read geometry type (with flags)
    let geom_type_with_flags = read_u32(cursor, byte_order)?;

    // Extract SRID, Z and M flags
    let has_srid = (geom_type_with_flags & type_codes::SRID_FLAG) != 0;
    let has_z = (geom_type_with_flags & type_codes::Z_FLAG) != 0;
    let has_m = (geom_type_with_flags & type_codes::M_FLAG) != 0;

    // Extract base geometry type (remove flags)
    let geom_type = geom_type_with_flags & 0x000000FF;

    // Read SRID if present
    let crs = if has_srid {
        let srid = read_u32(cursor, byte_order)?;
        Crs::epsg(srid)
    } else {
        Crs::default()
    };

    // Parse geometry based on type, collecting any Z/M values in flat
    // `coords_iter()` order alongside the flattened 2D geometry.
    let mut z_values: Vec<f64> = Vec::new();
    let mut m_values: Vec<f64> = Vec::new();

    let geom = match geom_type {
        type_codes::POINT => {
            let point = parse_point(
                cursor,
                byte_order,
                has_z,
                has_m,
                &mut z_values,
                &mut m_values,
            )?;
            GeoGeometry::Point(point)
        }
        type_codes::LINESTRING => {
            let linestring = parse_linestring(
                cursor,
                byte_order,
                has_z,
                has_m,
                &mut z_values,
                &mut m_values,
            )?;
            GeoGeometry::LineString(linestring)
        }
        type_codes::POLYGON => {
            let polygon = parse_polygon(
                cursor,
                byte_order,
                has_z,
                has_m,
                &mut z_values,
                &mut m_values,
            )?;
            GeoGeometry::Polygon(polygon)
        }
        type_codes::MULTIPOINT => {
            let multipoint = parse_multipoint(
                cursor,
                byte_order,
                has_z,
                has_m,
                &mut z_values,
                &mut m_values,
            )?;
            GeoGeometry::MultiPoint(multipoint)
        }
        type_codes::MULTILINESTRING => {
            let multilinestring = parse_multilinestring(
                cursor,
                byte_order,
                has_z,
                has_m,
                &mut z_values,
                &mut m_values,
            )?;
            GeoGeometry::MultiLineString(multilinestring)
        }
        type_codes::MULTIPOLYGON => {
            let multipolygon = parse_multipolygon(
                cursor,
                byte_order,
                has_z,
                has_m,
                &mut z_values,
                &mut m_values,
            )?;
            GeoGeometry::MultiPolygon(multipolygon)
        }
        _ => {
            return Err(GeoSparqlError::ParseError(format!(
                "Unsupported geometry type: {}",
                geom_type
            )))
        }
    };

    let coord3d = match (has_z, has_m) {
        (false, false) => Coord3D::xy(),
        (true, false) => Coord3D::xyz(z_values),
        (false, true) => Coord3D::xym(m_values),
        (true, true) => Coord3D::xyzm(z_values, m_values),
    };

    Ok(Geometry::with_crs_and_coord3d(geom, crs, coord3d))
}

fn parse_point(
    cursor: &mut Cursor<&[u8]>,
    byte_order: ByteOrder,
    has_z: bool,
    has_m: bool,
    z_out: &mut Vec<f64>,
    m_out: &mut Vec<f64>,
) -> Result<Point<f64>> {
    let x = read_f64(cursor, byte_order)?;
    let y = read_f64(cursor, byte_order)?;
    if has_z {
        z_out.push(read_f64(cursor, byte_order)?);
    }
    if has_m {
        m_out.push(read_f64(cursor, byte_order)?);
    }
    Ok(Point::new(x, y))
}

fn parse_linestring(
    cursor: &mut Cursor<&[u8]>,
    byte_order: ByteOrder,
    has_z: bool,
    has_m: bool,
    z_out: &mut Vec<f64>,
    m_out: &mut Vec<f64>,
) -> Result<LineString<f64>> {
    let num_points = read_u32(cursor, byte_order)? as usize;
    let mut coords = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        let x = read_f64(cursor, byte_order)?;
        let y = read_f64(cursor, byte_order)?;
        if has_z {
            z_out.push(read_f64(cursor, byte_order)?);
        }
        if has_m {
            m_out.push(read_f64(cursor, byte_order)?);
        }
        coords.push(Coord { x, y });
    }

    Ok(LineString::from(coords))
}

fn parse_polygon(
    cursor: &mut Cursor<&[u8]>,
    byte_order: ByteOrder,
    has_z: bool,
    has_m: bool,
    z_out: &mut Vec<f64>,
    m_out: &mut Vec<f64>,
) -> Result<Polygon<f64>> {
    let num_rings = read_u32(cursor, byte_order)? as usize;

    if num_rings == 0 {
        let empty_ring: Vec<Coord<f64>> = vec![];
        return Ok(Polygon::new(LineString::from(empty_ring), vec![]));
    }

    // Read exterior ring
    let exterior = parse_linestring(cursor, byte_order, has_z, has_m, z_out, m_out)?;

    // Read interior rings (holes)
    let mut interiors = Vec::with_capacity(num_rings.saturating_sub(1));
    for _ in 1..num_rings {
        interiors.push(parse_linestring(
            cursor, byte_order, has_z, has_m, z_out, m_out,
        )?);
    }

    Ok(Polygon::new(exterior, interiors))
}

fn parse_multipoint(
    cursor: &mut Cursor<&[u8]>,
    byte_order: ByteOrder,
    has_z: bool,
    has_m: bool,
    z_out: &mut Vec<f64>,
    m_out: &mut Vec<f64>,
) -> Result<MultiPoint<f64>> {
    let num_points = read_u32(cursor, byte_order)? as usize;
    let mut points = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        // Each point has its own byte order and type marker; the
        // dimensionality (Z/M) is taken from the parent geometry's flags,
        // matching how this crate's own writer emits sub-geometry type
        // words (consistent flags at every level).
        let _sub_byte_order_byte = read_u8(cursor)?;
        let _sub_geom_type = read_u32(cursor, byte_order)?;

        let point = parse_point(cursor, byte_order, has_z, has_m, z_out, m_out)?;
        points.push(point);
    }

    Ok(MultiPoint::from(points))
}

fn parse_multilinestring(
    cursor: &mut Cursor<&[u8]>,
    byte_order: ByteOrder,
    has_z: bool,
    has_m: bool,
    z_out: &mut Vec<f64>,
    m_out: &mut Vec<f64>,
) -> Result<MultiLineString<f64>> {
    let num_linestrings = read_u32(cursor, byte_order)? as usize;
    let mut linestrings = Vec::with_capacity(num_linestrings);

    for _ in 0..num_linestrings {
        let _sub_byte_order_byte = read_u8(cursor)?;
        let _sub_geom_type = read_u32(cursor, byte_order)?;

        let linestring = parse_linestring(cursor, byte_order, has_z, has_m, z_out, m_out)?;
        linestrings.push(linestring);
    }

    Ok(MultiLineString(linestrings))
}

fn parse_multipolygon(
    cursor: &mut Cursor<&[u8]>,
    byte_order: ByteOrder,
    has_z: bool,
    has_m: bool,
    z_out: &mut Vec<f64>,
    m_out: &mut Vec<f64>,
) -> Result<MultiPolygon<f64>> {
    let num_polygons = read_u32(cursor, byte_order)? as usize;
    let mut polygons = Vec::with_capacity(num_polygons);

    for _ in 0..num_polygons {
        let _sub_byte_order_byte = read_u8(cursor)?;
        let _sub_geom_type = read_u32(cursor, byte_order)?;

        let polygon = parse_polygon(cursor, byte_order, has_z, has_m, z_out, m_out)?;
        polygons.push(polygon);
    }

    Ok(MultiPolygon(polygons))
}

/// Serialize geometry to EWKB binary format
///
/// # Arguments
/// * `geometry` - Geometry to serialize
///
/// # Returns
/// EWKB binary data as `Vec<u8>`
///
/// # Example
/// ```ignore
/// use oxirs_geosparql::geometry::{Geometry, ewkb_parser::geometry_to_ewkb};
///
/// let geom = Geometry::from_wkt("POINT(1 2)").expect("should succeed");
/// let ewkb = geometry_to_ewkb(&geom).expect("should succeed");
/// ```
pub fn geometry_to_ewkb(geometry: &Geometry) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let byte_order = ByteOrder::LittleEndian;

    write_ewkb_geometry(&mut buffer, geometry, byte_order)?;

    Ok(buffer)
}

/// Compute the Z/M flag bits (per EWKB spec: `0x80000000` for Z, `0x40000000`
/// for M) to OR into a geometry-type word, based on a `Coord3D`'s dimension.
fn zm_flags(coord3d: &Coord3D) -> u32 {
    let mut flags = 0u32;
    if coord3d.has_z() {
        flags |= type_codes::Z_FLAG;
    }
    if coord3d.has_m() {
        flags |= type_codes::M_FLAG;
    }
    flags
}

/// Write the Z and/or M value for the coordinate at flat index `idx` (in
/// `coords_iter()` order), if `coord3d` declares that dimension present.
///
/// Errors (rather than silently defaulting to 0.0) if `coord3d` declares Z
/// or M present but no value is available at `idx` -- this can only happen
/// if a `Geometry` was constructed with a `Coord3D` whose value count
/// doesn't match its actual coordinate count, which is a caller bug that
/// must surface loudly rather than silently truncate to 2D EWKB.
fn write_zm(
    buffer: &mut Vec<u8>,
    coord3d: &Coord3D,
    idx: usize,
    byte_order: ByteOrder,
) -> Result<()> {
    if coord3d.has_z() {
        let z = coord3d.z_at(idx).ok_or_else(|| {
            GeoSparqlError::SerializationError(format!(
                "Geometry declares Z coordinates but is missing a Z value at index {}",
                idx
            ))
        })?;
        write_f64(buffer, z, byte_order)?;
    }
    if coord3d.has_m() {
        let m = coord3d.m_at(idx).ok_or_else(|| {
            GeoSparqlError::SerializationError(format!(
                "Geometry declares M coordinates but is missing an M value at index {}",
                idx
            ))
        })?;
        write_f64(buffer, m, byte_order)?;
    }
    Ok(())
}

fn write_ewkb_geometry(
    buffer: &mut Vec<u8>,
    geometry: &Geometry,
    byte_order: ByteOrder,
) -> Result<()> {
    // Write byte order
    buffer.write_all(&[byte_order.to_byte()]).map_err(|e| {
        GeoSparqlError::SerializationError(format!("Failed to write byte order: {}", e))
    })?;

    // Determine geometry type and write with SRID flag if needed
    let base_type = match &geometry.geom {
        GeoGeometry::Point(_) => type_codes::POINT,
        GeoGeometry::Line(_) | GeoGeometry::LineString(_) => type_codes::LINESTRING,
        GeoGeometry::Polygon(_) => type_codes::POLYGON,
        GeoGeometry::MultiPoint(_) => type_codes::MULTIPOINT,
        GeoGeometry::MultiLineString(_) => type_codes::MULTILINESTRING,
        GeoGeometry::MultiPolygon(_) => type_codes::MULTIPOLYGON,
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "Geometry type not supported in EWKB".to_string(),
            ))
        }
    };

    // If the geometry carries Z and/or M coordinates, validate that the
    // coord3d side-channel has exactly as many values as the geometry has
    // coordinates *before* writing anything, so a mismatch fails loudly
    // with a clear error instead of silently truncating to 2D partway
    // through serialization.
    let coord3d = &geometry.coord3d;
    if coord3d.has_z() || coord3d.has_m() {
        let coord_count = geometry.geom.coords_count();
        coord3d
            .validate(coord_count)
            .map_err(GeoSparqlError::SerializationError)?;
    }

    // Add Z/M and SRID flags
    let mut geom_type = base_type | zm_flags(coord3d);
    let srid_opt = geometry.crs.epsg_code();
    let has_srid = srid_opt.is_some();
    if has_srid {
        geom_type |= type_codes::SRID_FLAG;
    }

    write_u32(buffer, geom_type, byte_order)?;

    // Write SRID if present
    if let Some(srid) = srid_opt {
        write_u32(buffer, srid, byte_order)?;
    }

    // Write geometry data. `idx` tracks the flat coordinate position (in
    // `coords_iter()` order) so Z/M values can be looked up from `coord3d`.
    let mut idx = 0usize;
    match &geometry.geom {
        GeoGeometry::Point(p) => write_point(buffer, p, byte_order, coord3d, &mut idx)?,
        GeoGeometry::Line(l) => {
            let ls = LineString::from(vec![l.start, l.end]);
            write_linestring(buffer, &ls, byte_order, coord3d, &mut idx)?;
        }
        GeoGeometry::LineString(ls) => write_linestring(buffer, ls, byte_order, coord3d, &mut idx)?,
        GeoGeometry::Polygon(poly) => write_polygon(buffer, poly, byte_order, coord3d, &mut idx)?,
        GeoGeometry::MultiPoint(mp) => write_multipoint(buffer, mp, byte_order, coord3d, &mut idx)?,
        GeoGeometry::MultiLineString(mls) => {
            write_multilinestring(buffer, mls, byte_order, coord3d, &mut idx)?
        }
        GeoGeometry::MultiPolygon(mpoly) => {
            write_multipolygon(buffer, mpoly, byte_order, coord3d, &mut idx)?
        }
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "Geometry type not supported in EWKB".to_string(),
            ))
        }
    }

    Ok(())
}

fn write_point(
    buffer: &mut Vec<u8>,
    point: &Point<f64>,
    byte_order: ByteOrder,
    coord3d: &Coord3D,
    idx: &mut usize,
) -> Result<()> {
    write_f64(buffer, point.x(), byte_order)?;
    write_f64(buffer, point.y(), byte_order)?;
    write_zm(buffer, coord3d, *idx, byte_order)?;
    *idx += 1;
    Ok(())
}

fn write_linestring(
    buffer: &mut Vec<u8>,
    linestring: &LineString<f64>,
    byte_order: ByteOrder,
    coord3d: &Coord3D,
    idx: &mut usize,
) -> Result<()> {
    write_u32(buffer, linestring.0.len() as u32, byte_order)?;
    for coord in &linestring.0 {
        write_f64(buffer, coord.x, byte_order)?;
        write_f64(buffer, coord.y, byte_order)?;
        write_zm(buffer, coord3d, *idx, byte_order)?;
        *idx += 1;
    }
    Ok(())
}

fn write_polygon(
    buffer: &mut Vec<u8>,
    polygon: &Polygon<f64>,
    byte_order: ByteOrder,
    coord3d: &Coord3D,
    idx: &mut usize,
) -> Result<()> {
    let num_rings = 1 + polygon.interiors().len();
    write_u32(buffer, num_rings as u32, byte_order)?;

    // Write exterior ring
    write_linestring(buffer, polygon.exterior(), byte_order, coord3d, idx)?;

    // Write interior rings
    for interior in polygon.interiors() {
        write_linestring(buffer, interior, byte_order, coord3d, idx)?;
    }

    Ok(())
}

fn write_multipoint(
    buffer: &mut Vec<u8>,
    multipoint: &MultiPoint<f64>,
    byte_order: ByteOrder,
    coord3d: &Coord3D,
    idx: &mut usize,
) -> Result<()> {
    write_u32(buffer, multipoint.0.len() as u32, byte_order)?;
    let sub_type = type_codes::POINT | zm_flags(coord3d);
    for point in &multipoint.0 {
        buffer.write_all(&[byte_order.to_byte()]).map_err(|e| {
            GeoSparqlError::SerializationError(format!("Failed to write byte order: {}", e))
        })?;
        write_u32(buffer, sub_type, byte_order)?;
        write_point(buffer, point, byte_order, coord3d, idx)?;
    }
    Ok(())
}

fn write_multilinestring(
    buffer: &mut Vec<u8>,
    multilinestring: &MultiLineString<f64>,
    byte_order: ByteOrder,
    coord3d: &Coord3D,
    idx: &mut usize,
) -> Result<()> {
    write_u32(buffer, multilinestring.0.len() as u32, byte_order)?;
    let sub_type = type_codes::LINESTRING | zm_flags(coord3d);
    for linestring in &multilinestring.0 {
        buffer.write_all(&[byte_order.to_byte()]).map_err(|e| {
            GeoSparqlError::SerializationError(format!("Failed to write byte order: {}", e))
        })?;
        write_u32(buffer, sub_type, byte_order)?;
        write_linestring(buffer, linestring, byte_order, coord3d, idx)?;
    }
    Ok(())
}

fn write_multipolygon(
    buffer: &mut Vec<u8>,
    multipolygon: &MultiPolygon<f64>,
    byte_order: ByteOrder,
    coord3d: &Coord3D,
    idx: &mut usize,
) -> Result<()> {
    write_u32(buffer, multipolygon.0.len() as u32, byte_order)?;
    let sub_type = type_codes::POLYGON | zm_flags(coord3d);
    for polygon in &multipolygon.0 {
        buffer.write_all(&[byte_order.to_byte()]).map_err(|e| {
            GeoSparqlError::SerializationError(format!("Failed to write byte order: {}", e))
        })?;
        write_u32(buffer, sub_type, byte_order)?;
        write_polygon(buffer, polygon, byte_order, coord3d, idx)?;
    }
    Ok(())
}

// Helper functions for reading binary data
fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to read u8: {}", e)))?;
    Ok(buf[0])
}

fn read_u32(cursor: &mut Cursor<&[u8]>, byte_order: ByteOrder) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to read u32: {}", e)))?;

    Ok(match byte_order {
        ByteOrder::LittleEndian => u32::from_le_bytes(buf),
        ByteOrder::BigEndian => u32::from_be_bytes(buf),
    })
}

fn read_f64(cursor: &mut Cursor<&[u8]>, byte_order: ByteOrder) -> Result<f64> {
    let mut buf = [0u8; 8];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| GeoSparqlError::ParseError(format!("Failed to read f64: {}", e)))?;

    Ok(match byte_order {
        ByteOrder::LittleEndian => f64::from_le_bytes(buf),
        ByteOrder::BigEndian => f64::from_be_bytes(buf),
    })
}

// Helper functions for writing binary data
fn write_u32(buffer: &mut Vec<u8>, value: u32, byte_order: ByteOrder) -> Result<()> {
    let bytes = match byte_order {
        ByteOrder::LittleEndian => value.to_le_bytes(),
        ByteOrder::BigEndian => value.to_be_bytes(),
    };
    buffer
        .write_all(&bytes)
        .map_err(|e| GeoSparqlError::SerializationError(format!("Failed to write u32: {}", e)))?;
    Ok(())
}

fn write_f64(buffer: &mut Vec<u8>, value: f64, byte_order: ByteOrder) -> Result<()> {
    let bytes = match byte_order {
        ByteOrder::LittleEndian => value.to_le_bytes(),
        ByteOrder::BigEndian => value.to_be_bytes(),
    };
    buffer
        .write_all(&bytes)
        .map_err(|e| GeoSparqlError::SerializationError(format!("Failed to write f64: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::point;

    #[test]
    fn test_point_round_trip() {
        let geom = Geometry::new(GeoGeometry::Point(point! { x: 1.0, y: 2.0 }));
        let ewkb = geometry_to_ewkb(&geom).expect("should succeed");
        let parsed = parse_ewkb(&ewkb).expect("should succeed");

        match parsed.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_point_with_srid() {
        let geom = Geometry::with_crs(
            GeoGeometry::Point(point! { x: 1.0, y: 2.0 }),
            Crs::epsg(4326),
        );
        let ewkb = geometry_to_ewkb(&geom).expect("should succeed");
        let parsed = parse_ewkb(&ewkb).expect("should succeed");

        assert_eq!(parsed.crs.epsg_code(), Some(4326));
        match parsed.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point geometry"),
        }
    }

    #[test]
    fn test_linestring_round_trip() {
        let ls = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 2.0, y: 0.0 },
        ]);
        let geom = Geometry::new(GeoGeometry::LineString(ls));
        let ewkb = geometry_to_ewkb(&geom).expect("should succeed");
        let parsed = parse_ewkb(&ewkb).expect("should succeed");

        match parsed.geom {
            GeoGeometry::LineString(parsed_ls) => {
                assert_eq!(parsed_ls.0.len(), 3);
                assert_eq!(parsed_ls.0[0].x, 0.0);
                assert_eq!(parsed_ls.0[1].y, 1.0);
            }
            _ => panic!("Expected LineString geometry"),
        }
    }

    #[test]
    fn test_polygon_round_trip() {
        let exterior = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 4.0, y: 0.0 },
            Coord { x: 4.0, y: 4.0 },
            Coord { x: 0.0, y: 4.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        let poly = Polygon::new(exterior, vec![]);
        let geom = Geometry::new(GeoGeometry::Polygon(poly));
        let ewkb = geometry_to_ewkb(&geom).expect("should succeed");
        let parsed = parse_ewkb(&ewkb).expect("should succeed");

        match parsed.geom {
            GeoGeometry::Polygon(parsed_poly) => {
                assert_eq!(parsed_poly.exterior().0.len(), 5);
                assert_eq!(parsed_poly.interiors().len(), 0);
            }
            _ => panic!("Expected Polygon geometry"),
        }
    }

    #[test]
    fn test_byte_order() {
        assert_eq!(
            ByteOrder::from_byte(0x00).expect("should succeed"),
            ByteOrder::BigEndian
        );
        assert_eq!(
            ByteOrder::from_byte(0x01).expect("should succeed"),
            ByteOrder::LittleEndian
        );
        assert!(ByteOrder::from_byte(0x02).is_err());
    }

    // ========================================================================
    // Regression tests: Z/M coordinates must round-trip through EWKB instead
    // of being silently dropped (see geometry/ewkb_parser.rs Z/M finding).
    // ========================================================================

    #[test]
    fn regression_ewkb_point_z_round_trip() {
        let geom = Geometry::from_wkt("POINT Z(1 2 3)").expect("parse POINT Z");
        let ewkb = geometry_to_ewkb(&geom).expect("should serialize 3D point");
        let parsed = parse_ewkb(&ewkb).expect("should parse 3D point");

        match &parsed.geom {
            GeoGeometry::Point(p) => {
                assert_eq!(p.x(), 1.0);
                assert_eq!(p.y(), 2.0);
            }
            _ => panic!("Expected Point geometry"),
        }
        assert!(parsed.coord3d.has_z());
        assert_eq!(parsed.coord3d.z_at(0), Some(3.0));
    }

    #[test]
    fn regression_ewkb_point_zm_round_trip() {
        let geom = Geometry::from_wkt("POINT ZM(1 2 3 4)").expect("parse POINT ZM");
        let ewkb = geometry_to_ewkb(&geom).expect("should serialize ZM point");
        let parsed = parse_ewkb(&ewkb).expect("should parse ZM point");

        assert!(parsed.coord3d.has_z());
        assert!(parsed.coord3d.has_m());
        assert_eq!(parsed.coord3d.z_at(0), Some(3.0));
        assert_eq!(parsed.coord3d.m_at(0), Some(4.0));
    }

    #[test]
    fn regression_ewkb_linestring_z_round_trip() {
        let geom =
            Geometry::from_wkt("LINESTRING Z(0 0 10, 1 1 20, 2 0 30)").expect("parse LINESTRING Z");
        let ewkb = geometry_to_ewkb(&geom).expect("should serialize 3D linestring");
        let parsed = parse_ewkb(&ewkb).expect("should parse 3D linestring");

        match &parsed.geom {
            GeoGeometry::LineString(ls) => assert_eq!(ls.0.len(), 3),
            _ => panic!("Expected LineString geometry"),
        }
        assert_eq!(parsed.coord3d.z_at(0), Some(10.0));
        assert_eq!(parsed.coord3d.z_at(1), Some(20.0));
        assert_eq!(parsed.coord3d.z_at(2), Some(30.0));
    }

    #[test]
    fn regression_ewkb_polygon_z_round_trip() {
        let geom = Geometry::from_wkt("POLYGON Z((0 0 1, 4 0 2, 4 4 3, 0 4 4, 0 0 1))")
            .expect("parse POLYGON Z");
        let ewkb = geometry_to_ewkb(&geom).expect("should serialize 3D polygon");
        let parsed = parse_ewkb(&ewkb).expect("should parse 3D polygon");

        match &parsed.geom {
            GeoGeometry::Polygon(poly) => assert_eq!(poly.exterior().0.len(), 5),
            _ => panic!("Expected Polygon geometry"),
        }
        assert!(parsed.coord3d.has_z());
        assert_eq!(parsed.coord3d.z_at(0), Some(1.0));
        assert_eq!(parsed.coord3d.z_at(2), Some(3.0));
    }

    #[test]
    fn regression_ewkb_2d_geometry_carries_no_zm_flags() {
        // A plain 2D point must not set the Z/M flag bits in the type word
        // (verifies we don't regress the existing 2D path).
        let geom = Geometry::new(GeoGeometry::Point(point! { x: 1.0, y: 2.0 }));
        let ewkb = geometry_to_ewkb(&geom).expect("should succeed");

        // byte[0] = byte order, byte[1..5] = little-endian type word
        let type_word = u32::from_le_bytes([ewkb[1], ewkb[2], ewkb[3], ewkb[4]]);
        assert_eq!(type_word & type_codes::Z_FLAG, 0);
        assert_eq!(type_word & type_codes::M_FLAG, 0);
    }

    #[test]
    fn regression_ewkb_z_value_count_mismatch_errors() {
        // A Geometry whose coord3d claims Z coordinates but doesn't actually
        // supply enough of them must fail loudly rather than silently
        // writing a truncated/garbled EWKB blob.
        let geom = Geometry::with_crs_and_coord3d(
            GeoGeometry::LineString(LineString::from(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 1.0, y: 1.0 },
            ])),
            Crs::default(),
            Coord3D::xyz(vec![1.0]), // only 1 Z value for 2 coordinates
        );

        let result = geometry_to_ewkb(&geom);
        assert!(result.is_err(), "mismatched Z count must be an error");
    }
}
