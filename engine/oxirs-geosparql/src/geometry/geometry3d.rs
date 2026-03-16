//! 3D Geometry Types and Operations
//!
//! Extends 2D geometry with Z (elevation) coordinate following:
//! - ISO 19107 Geographic Information — Spatial Schema
//! - OGC Simple Features for SQL specification
//! - WKT serialization as per OGC 06-103r4
//!
//! These types provide rich 3D operations including WKT roundtrip,
//! 3D bounding boxes, z-range queries, and flattening to 2D.

use crate::error::{GeoSparqlError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Point3D
// ---------------------------------------------------------------------------

/// A 3D point with X, Y, and Z (elevation) coordinates.
///
/// Coordinates follow the convention:
/// - X: longitude (or easting for projected CRS)
/// - Y: latitude (or northing for projected CRS)
/// - Z: elevation / altitude in meters above the reference ellipsoid
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    /// X coordinate (longitude or easting)
    pub x: f64,
    /// Y coordinate (latitude or northing)
    pub y: f64,
    /// Z coordinate (elevation in meters)
    pub z: f64,
    /// Optional EPSG SRID for the coordinate system
    pub srid: Option<u32>,
}

impl Point3D {
    /// Create a new 3D point without CRS information.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
            srid: None,
        }
    }

    /// Create a new 3D point with an EPSG SRID.
    pub fn with_srid(x: f64, y: f64, z: f64, srid: u32) -> Self {
        Self {
            x,
            y,
            z,
            srid: Some(srid),
        }
    }

    /// Euclidean distance in 3D space.
    ///
    /// Computed as `sqrt((dx² + dy² + dz²))`.
    pub fn distance_3d(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Projected (2D) distance ignoring Z.
    pub fn distance_2d(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Drop the Z coordinate and return `(x, y)`.
    pub fn to_2d(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// Midpoint between two 3D points.
    pub fn midpoint(&self, other: &Point3D) -> Point3D {
        Point3D::new(
            (self.x + other.x) / 2.0,
            (self.y + other.y) / 2.0,
            (self.z + other.z) / 2.0,
        )
    }

    /// Linear interpolation between two points at parameter `t` ∈ [0, 1].
    pub fn lerp(&self, other: &Point3D, t: f64) -> Point3D {
        Point3D::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z),
        )
    }
}

impl fmt::Display for Point3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {} {})", self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------
// LinearRing3D
// ---------------------------------------------------------------------------

/// A closed 3D ring (first point == last point).
///
/// Used as the exterior and interior rings of a [`Polygon3D`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearRing3D {
    /// Points forming the ring. Must have ≥ 4 points (first == last to close).
    pub points: Vec<Point3D>,
}

impl LinearRing3D {
    /// Create a new linear ring from a vector of points.
    ///
    /// If the last point differs from the first, the first point is appended to close the ring.
    pub fn new(mut points: Vec<Point3D>) -> Self {
        // Auto-close the ring
        if points.len() >= 2 {
            let first = points[0];
            let last = *points.last().expect("non-empty vector");
            if (first.x - last.x).abs() > f64::EPSILON
                || (first.y - last.y).abs() > f64::EPSILON
                || (first.z - last.z).abs() > f64::EPSILON
            {
                points.push(first);
            }
        }
        Self { points }
    }

    /// Check whether this ring is properly closed.
    pub fn is_closed(&self) -> bool {
        if self.points.len() < 2 {
            return false;
        }
        let first = &self.points[0];
        let last = self.points.last().expect("checked len");
        (first.x - last.x).abs() < f64::EPSILON
            && (first.y - last.y).abs() < f64::EPSILON
            && (first.z - last.z).abs() < f64::EPSILON
    }

    /// Signed projected area of the ring (ignoring Z).
    ///
    /// Positive = counter-clockwise, negative = clockwise.
    pub fn signed_area_2d(&self) -> f64 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..n - 1 {
            let p = &self.points[i];
            let q = &self.points[i + 1];
            area += p.x * q.y - q.x * p.y;
        }
        area / 2.0
    }

    /// 2D area (absolute value of signed area).
    pub fn area_2d(&self) -> f64 {
        self.signed_area_2d().abs()
    }

    /// Z-range of the ring.
    pub fn z_range(&self) -> Option<(f64, f64)> {
        z_range_of_points(&self.points)
    }
}

// ---------------------------------------------------------------------------
// LineString3D
// ---------------------------------------------------------------------------

/// A sequence of 3D points forming an open polyline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineString3D {
    /// Ordered sequence of 3D points.
    pub points: Vec<Point3D>,
}

impl LineString3D {
    /// Create a new line string from points.
    pub fn new(points: Vec<Point3D>) -> Self {
        Self { points }
    }

    /// 3D arc-length of the line string (sum of segment distances).
    pub fn length_3d(&self) -> f64 {
        self.points
            .windows(2)
            .map(|pair| pair[0].distance_3d(&pair[1]))
            .sum()
    }

    /// 2D projected length (ignoring Z).
    pub fn length_2d(&self) -> f64 {
        self.points
            .windows(2)
            .map(|pair| pair[0].distance_2d(&pair[1]))
            .sum()
    }

    /// Z-range of the line string.
    pub fn z_range(&self) -> Option<(f64, f64)> {
        z_range_of_points(&self.points)
    }

    /// Check whether the line string is closed (first == last point).
    pub fn is_closed(&self) -> bool {
        if self.points.len() < 2 {
            return false;
        }
        let first = &self.points[0];
        let last = self.points.last().expect("checked len");
        (first.x - last.x).abs() < f64::EPSILON
            && (first.y - last.y).abs() < f64::EPSILON
            && (first.z - last.z).abs() < f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// Polygon3D
// ---------------------------------------------------------------------------

/// A 3D polygon with an exterior ring and optional interior holes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon3D {
    /// Exterior ring (counter-clockwise winding in XY plane).
    pub exterior: LinearRing3D,
    /// Interior rings / holes (clockwise winding in XY plane).
    pub holes: Vec<LinearRing3D>,
}

impl Polygon3D {
    /// Create a simple polygon (no holes) from an exterior ring.
    pub fn new(exterior: LinearRing3D) -> Self {
        Self {
            exterior,
            holes: Vec::new(),
        }
    }

    /// Create a polygon with holes.
    pub fn with_holes(exterior: LinearRing3D, holes: Vec<LinearRing3D>) -> Self {
        Self { exterior, holes }
    }

    /// Projected 2D area (exterior minus holes).
    pub fn area_2d(&self) -> f64 {
        let ext_area = self.exterior.area_2d();
        let hole_area: f64 = self.holes.iter().map(|h| h.area_2d()).sum();
        ext_area - hole_area
    }

    /// Z-range of the polygon (exterior + holes combined).
    pub fn z_range(&self) -> Option<(f64, f64)> {
        let mut all_pts: Vec<Point3D> = self.exterior.points.clone();
        for hole in &self.holes {
            all_pts.extend_from_slice(&hole.points);
        }
        z_range_of_points(&all_pts)
    }
}

// ---------------------------------------------------------------------------
// BoundingBox3D
// ---------------------------------------------------------------------------

/// A 3D axis-aligned bounding box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox3D {
    /// Minimum X
    pub min_x: f64,
    /// Maximum X
    pub max_x: f64,
    /// Minimum Y
    pub min_y: f64,
    /// Maximum Y
    pub max_y: f64,
    /// Minimum Z
    pub min_z: f64,
    /// Maximum Z
    pub max_z: f64,
}

impl BoundingBox3D {
    /// Create a bounding box from min/max extents.
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64, min_z: f64, max_z: f64) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
        }
    }

    /// Check whether two 3D bounding boxes intersect (inclusive).
    pub fn intersects(&self, other: &BoundingBox3D) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }

    /// Check whether this bounding box fully contains another.
    pub fn contains_bbox(&self, other: &BoundingBox3D) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
            && self.min_z <= other.min_z
            && self.max_z >= other.max_z
    }

    /// Check whether a 3D point lies inside this bounding box.
    pub fn contains_point(&self, p: &Point3D) -> bool {
        p.x >= self.min_x
            && p.x <= self.max_x
            && p.y >= self.min_y
            && p.y <= self.max_y
            && p.z >= self.min_z
            && p.z <= self.max_z
    }

    /// Expand bounding box uniformly in all directions.
    pub fn expand_by(&self, delta: f64) -> Self {
        Self {
            min_x: self.min_x - delta,
            max_x: self.max_x + delta,
            min_y: self.min_y - delta,
            max_y: self.max_y + delta,
            min_z: self.min_z - delta,
            max_z: self.max_z + delta,
        }
    }

    /// Merge two bounding boxes into their union.
    pub fn union(&self, other: &BoundingBox3D) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            max_x: self.max_x.max(other.max_x),
            min_y: self.min_y.min(other.min_y),
            max_y: self.max_y.max(other.max_y),
            min_z: self.min_z.min(other.min_z),
            max_z: self.max_z.max(other.max_z),
        }
    }

    /// Volume of the 3D bounding box.
    pub fn volume(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> Point3D {
        Point3D::new(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )
    }

    /// Create from a single point (degenerate box).
    pub fn from_point(p: &Point3D) -> Self {
        Self::new(p.x, p.x, p.y, p.y, p.z, p.z)
    }
}

impl fmt::Display for BoundingBox3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BOX3D({} {} {}, {} {} {})",
            self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z
        )
    }
}

// ---------------------------------------------------------------------------
// Geometry3DEnum
// ---------------------------------------------------------------------------

/// Full 3D geometry type following OGC SF hierarchy.
///
/// This enum captures all geometry types that carry explicit Z coordinates.
/// For geometry that already exists in the codebase as [`crate::geometry::Geometry`]
/// (backed by `geo_types`), see `Geometry3DEnum::flatten_to_2d`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Geometry3DEnum {
    /// Single 3D point
    Point(Point3D),
    /// 3D polyline
    LineString(LineString3D),
    /// 3D polygon with optional holes
    Polygon(Polygon3D),
    /// Set of 3D points
    MultiPoint(Vec<Point3D>),
    /// Set of 3D polylines
    MultiLineString(Vec<LineString3D>),
    /// Set of 3D polygons
    MultiPolygon(Vec<Polygon3D>),
    /// Heterogeneous collection
    GeometryCollection(Vec<Geometry3DEnum>),
}

impl Geometry3DEnum {
    // ------------------------------------------------------------------
    // Bounding box / Z-range
    // ------------------------------------------------------------------

    /// Compute the Z-envelope: `(min_z, max_z)` over all coordinates.
    pub fn z_range(&self) -> Option<(f64, f64)> {
        match self {
            Geometry3DEnum::Point(p) => Some((p.z, p.z)),
            Geometry3DEnum::LineString(ls) => ls.z_range(),
            Geometry3DEnum::Polygon(poly) => poly.z_range(),
            Geometry3DEnum::MultiPoint(pts) => z_range_of_points(pts),
            Geometry3DEnum::MultiLineString(lines) => {
                let all: Vec<Point3D> = lines.iter().flat_map(|l| l.points.clone()).collect();
                z_range_of_points(&all)
            }
            Geometry3DEnum::MultiPolygon(polys) => {
                let ranges: Vec<(f64, f64)> = polys.iter().filter_map(|p| p.z_range()).collect();
                merge_z_ranges(&ranges)
            }
            Geometry3DEnum::GeometryCollection(geoms) => {
                let ranges: Vec<(f64, f64)> = geoms.iter().filter_map(|g| g.z_range()).collect();
                merge_z_ranges(&ranges)
            }
        }
    }

    /// Compute the 3D axis-aligned bounding box.
    pub fn bounding_box_3d(&self) -> Option<BoundingBox3D> {
        let pts = self.all_points();
        if pts.is_empty() {
            return None;
        }
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for p in &pts {
            if p.x < min_x {
                min_x = p.x;
            }
            if p.x > max_x {
                max_x = p.x;
            }
            if p.y < min_y {
                min_y = p.y;
            }
            if p.y > max_y {
                max_y = p.y;
            }
            if p.z < min_z {
                min_z = p.z;
            }
            if p.z > max_z {
                max_z = p.z;
            }
        }

        Some(BoundingBox3D::new(min_x, max_x, min_y, max_y, min_z, max_z))
    }

    // ------------------------------------------------------------------
    // WKT serialization
    // ------------------------------------------------------------------

    /// Serialize to ISO/OGC WKT with Z modifier (e.g. `POINT Z(1 2 3)`).
    pub fn to_wkt(&self) -> String {
        match self {
            Geometry3DEnum::Point(p) => format!("POINT Z({} {} {})", p.x, p.y, p.z),
            Geometry3DEnum::LineString(ls) => {
                let coords = points_to_wkt_coords_3d(&ls.points);
                format!("LINESTRING Z({})", coords)
            }
            Geometry3DEnum::Polygon(poly) => {
                let ext = ring_to_wkt_3d(&poly.exterior);
                let holes: Vec<String> = poly.holes.iter().map(ring_to_wkt_3d).collect();
                if holes.is_empty() {
                    format!("POLYGON Z(({}))", ext)
                } else {
                    format!("POLYGON Z(({}), {})", ext, holes.join(", "))
                }
            }
            Geometry3DEnum::MultiPoint(pts) => {
                let inner: Vec<String> = pts
                    .iter()
                    .map(|p| format!("({} {} {})", p.x, p.y, p.z))
                    .collect();
                format!("MULTIPOINT Z({})", inner.join(", "))
            }
            Geometry3DEnum::MultiLineString(lines) => {
                let inner: Vec<String> = lines
                    .iter()
                    .map(|l| format!("({})", points_to_wkt_coords_3d(&l.points)))
                    .collect();
                format!("MULTILINESTRING Z({})", inner.join(", "))
            }
            Geometry3DEnum::MultiPolygon(polys) => {
                let inner: Vec<String> = polys
                    .iter()
                    .map(|p| {
                        let ext = ring_to_wkt_3d(&p.exterior);
                        let holes: Vec<String> = p.holes.iter().map(ring_to_wkt_3d).collect();
                        if holes.is_empty() {
                            format!("(({})))", ext)
                        } else {
                            format!("(({}), {})", ext, holes.join(", "))
                        }
                    })
                    .collect();
                format!("MULTIPOLYGON Z({})", inner.join(", "))
            }
            Geometry3DEnum::GeometryCollection(geoms) => {
                let inner: Vec<String> = geoms.iter().map(|g| g.to_wkt()).collect();
                format!("GEOMETRYCOLLECTION Z({})", inner.join(", "))
            }
        }
    }

    /// Parse WKT with optional Z modifier.
    ///
    /// Accepts both `POINT Z(x y z)` and `POINT(x y z)` (implicit 3-tuple).
    pub fn from_wkt(wkt: &str) -> Result<Self> {
        parse_wkt_3d(wkt.trim())
    }

    // ------------------------------------------------------------------
    // 2D projection
    // ------------------------------------------------------------------

    /// Drop Z and return the 2D (x, y) coordinates as a flat list of
    /// `(x, y)` pairs. This is used internally by `flatten_to_2d`.
    fn all_points(&self) -> Vec<Point3D> {
        match self {
            Geometry3DEnum::Point(p) => vec![*p],
            Geometry3DEnum::LineString(ls) => ls.points.clone(),
            Geometry3DEnum::Polygon(poly) => {
                let mut pts = poly.exterior.points.clone();
                for hole in &poly.holes {
                    pts.extend_from_slice(&hole.points);
                }
                pts
            }
            Geometry3DEnum::MultiPoint(pts) => pts.clone(),
            Geometry3DEnum::MultiLineString(lines) => {
                lines.iter().flat_map(|l| l.points.clone()).collect()
            }
            Geometry3DEnum::MultiPolygon(polys) => polys
                .iter()
                .flat_map(|p| {
                    let mut pts = p.exterior.points.clone();
                    for hole in &p.holes {
                        pts.extend_from_slice(&hole.points);
                    }
                    pts
                })
                .collect(),
            Geometry3DEnum::GeometryCollection(geoms) => {
                geoms.iter().flat_map(|g| g.all_points()).collect()
            }
        }
    }

    /// Flatten to 2D WKT by stripping Z coordinates.
    ///
    /// Returns standard WKT without Z modifier.
    pub fn to_2d_wkt(&self) -> String {
        match self {
            Geometry3DEnum::Point(p) => format!("POINT({} {})", p.x, p.y),
            Geometry3DEnum::LineString(ls) => {
                let coords = points_to_wkt_coords_2d(&ls.points);
                format!("LINESTRING({})", coords)
            }
            Geometry3DEnum::Polygon(poly) => {
                let ext = ring_to_wkt_2d(&poly.exterior);
                let holes: Vec<String> = poly.holes.iter().map(ring_to_wkt_2d).collect();
                if holes.is_empty() {
                    format!("POLYGON(({}))", ext)
                } else {
                    format!("POLYGON(({}), {})", ext, holes.join(", "))
                }
            }
            Geometry3DEnum::MultiPoint(pts) => {
                let inner: Vec<String> = pts.iter().map(|p| format!("({} {})", p.x, p.y)).collect();
                format!("MULTIPOINT({})", inner.join(", "))
            }
            Geometry3DEnum::MultiLineString(lines) => {
                let inner: Vec<String> = lines
                    .iter()
                    .map(|l| format!("({})", points_to_wkt_coords_2d(&l.points)))
                    .collect();
                format!("MULTILINESTRING({})", inner.join(", "))
            }
            Geometry3DEnum::MultiPolygon(polys) => {
                let inner: Vec<String> = polys
                    .iter()
                    .map(|p| {
                        let ext = ring_to_wkt_2d(&p.exterior);
                        let holes: Vec<String> = p.holes.iter().map(ring_to_wkt_2d).collect();
                        if holes.is_empty() {
                            format!("(({})))", ext)
                        } else {
                            format!("(({}), {})", ext, holes.join(", "))
                        }
                    })
                    .collect();
                format!("MULTIPOLYGON({})", inner.join(", "))
            }
            Geometry3DEnum::GeometryCollection(geoms) => {
                let inner: Vec<String> = geoms.iter().map(|g| g.to_2d_wkt()).collect();
                format!("GEOMETRYCOLLECTION({})", inner.join(", "))
            }
        }
    }

    /// Count total number of vertices.
    pub fn num_points(&self) -> usize {
        self.all_points().len()
    }

    /// Geometry type name (uppercase, no Z modifier).
    pub fn geometry_type_name(&self) -> &'static str {
        match self {
            Geometry3DEnum::Point(_) => "Point",
            Geometry3DEnum::LineString(_) => "LineString",
            Geometry3DEnum::Polygon(_) => "Polygon",
            Geometry3DEnum::MultiPoint(_) => "MultiPoint",
            Geometry3DEnum::MultiLineString(_) => "MultiLineString",
            Geometry3DEnum::MultiPolygon(_) => "MultiPolygon",
            Geometry3DEnum::GeometryCollection(_) => "GeometryCollection",
        }
    }
}

impl fmt::Display for Geometry3DEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_wkt())
    }
}

// ---------------------------------------------------------------------------
// WKT parsing helpers
// ---------------------------------------------------------------------------

/// Parse a 3D WKT string into a [`Geometry3DEnum`].
fn parse_wkt_3d(wkt: &str) -> Result<Geometry3DEnum> {
    // Strip optional SRID prefix: "SRID=4326;"
    let wkt = if let Some(_rest) = wkt.to_uppercase().strip_prefix("SRID=") {
        // find the semicolon
        if let Some(pos) = wkt.find(';') {
            wkt[pos + 1..].trim()
        } else {
            wkt
        }
    } else {
        wkt
    };

    let upper = wkt.to_uppercase();

    if upper.starts_with("POINT") {
        let inner = extract_inner(wkt, "POINT")?;
        let p = parse_point3d_coords(inner.trim())?;
        Ok(Geometry3DEnum::Point(p))
    } else if upper.starts_with("LINESTRING") {
        let inner = extract_inner(wkt, "LINESTRING")?;
        let pts = parse_coord_list_3d(inner.trim())?;
        Ok(Geometry3DEnum::LineString(LineString3D::new(pts)))
    } else if upper.starts_with("POLYGON") {
        let inner = extract_inner(wkt, "POLYGON")?;
        let poly = parse_polygon3d(inner.trim())?;
        Ok(Geometry3DEnum::Polygon(poly))
    } else if upper.starts_with("MULTIPOINT") {
        let inner = extract_inner(wkt, "MULTIPOINT")?;
        let pts = parse_multipoint3d(inner.trim())?;
        Ok(Geometry3DEnum::MultiPoint(pts))
    } else if upper.starts_with("MULTILINESTRING") {
        let inner = extract_inner(wkt, "MULTILINESTRING")?;
        let lines = parse_multilinestring3d(inner.trim())?;
        Ok(Geometry3DEnum::MultiLineString(lines))
    } else if upper.starts_with("MULTIPOLYGON") {
        let inner = extract_inner(wkt, "MULTIPOLYGON")?;
        let polys = parse_multipolygon3d(inner.trim())?;
        Ok(Geometry3DEnum::MultiPolygon(polys))
    } else if upper.starts_with("GEOMETRYCOLLECTION") {
        let inner = extract_inner(wkt, "GEOMETRYCOLLECTION")?;
        let geoms = parse_geometrycollection3d(inner.trim())?;
        Ok(Geometry3DEnum::GeometryCollection(geoms))
    } else {
        Err(GeoSparqlError::InvalidWkt(format!(
            "Unknown 3D geometry type in WKT: {}",
            &wkt[..wkt.len().min(30)]
        )))
    }
}

/// Strip the type prefix (including optional Z/ZM/M modifier) and outer parens.
fn extract_inner<'a>(wkt: &'a str, type_name: &str) -> Result<&'a str> {
    let rest = wkt[type_name.len()..].trim_start();

    // Strip optional Z, ZM, M modifier
    let rest = if rest.to_uppercase().starts_with("ZM") {
        rest[2..].trim_start()
    } else if rest.to_uppercase().starts_with('Z') || rest.to_uppercase().starts_with('M') {
        rest[1..].trim_start()
    } else {
        rest
    };

    // Strip EMPTY keyword
    if rest.to_uppercase().starts_with("EMPTY") {
        return Ok("");
    }

    // Strip outer parentheses
    let rest = rest.trim();
    if rest.starts_with('(') && rest.ends_with(')') {
        Ok(&rest[1..rest.len() - 1])
    } else {
        Err(GeoSparqlError::InvalidWkt(format!(
            "Expected parentheses in WKT after type name '{}', got: '{}'",
            type_name,
            &rest[..rest.len().min(40)]
        )))
    }
}

/// Parse a single 3D point from `"x y z"` format (no parens).
fn parse_point3d_coords(coords: &str) -> Result<Point3D> {
    let parts: Vec<&str> = coords.split_whitespace().collect();
    match parts.len() {
        2 => {
            let x = parse_f64(parts[0])?;
            let y = parse_f64(parts[1])?;
            Ok(Point3D::new(x, y, 0.0))
        }
        3 => {
            let x = parse_f64(parts[0])?;
            let y = parse_f64(parts[1])?;
            let z = parse_f64(parts[2])?;
            Ok(Point3D::new(x, y, z))
        }
        4 => {
            // XYZM: treat 4th value as M, ignore it for 3D point
            let x = parse_f64(parts[0])?;
            let y = parse_f64(parts[1])?;
            let z = parse_f64(parts[2])?;
            Ok(Point3D::new(x, y, z))
        }
        _ => Err(GeoSparqlError::InvalidWkt(format!(
            "Expected 2, 3 or 4 coordinates for point, got {} in: '{}'",
            parts.len(),
            coords
        ))),
    }
}

/// Parse a comma-separated list of 3D coordinate tuples.
fn parse_coord_list_3d(s: &str) -> Result<Vec<Point3D>> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|t| parse_point3d_coords(t.trim()))
        .collect()
}

/// Parse a polygon ring list `(ring1), (ring2), ...`.
fn parse_polygon3d(s: &str) -> Result<Polygon3D> {
    let rings = split_rings(s)?;
    if rings.is_empty() {
        return Err(GeoSparqlError::InvalidWkt(
            "Polygon must have at least one ring".to_string(),
        ));
    }
    let exterior = LinearRing3D::new(parse_coord_list_3d(rings[0])?);
    let holes = rings[1..]
        .iter()
        .map(|r| parse_coord_list_3d(r).map(LinearRing3D::new))
        .collect::<Result<Vec<_>>>()?;
    Ok(Polygon3D::with_holes(exterior, holes))
}

/// Parse MULTIPOINT: `(x y z), (x y z), ...`
fn parse_multipoint3d(s: &str) -> Result<Vec<Point3D>> {
    // Support both `(x y z), (x y z)` and `x y z, x y z`
    if s.contains('(') {
        split_rings(s)?
            .iter()
            .map(|r| parse_point3d_coords(r.trim()))
            .collect()
    } else {
        parse_coord_list_3d(s)
    }
}

/// Parse MULTILINESTRING: `(ls1_coords), (ls2_coords), ...`
fn parse_multilinestring3d(s: &str) -> Result<Vec<LineString3D>> {
    split_rings(s)?
        .iter()
        .map(|r| parse_coord_list_3d(r.trim()).map(LineString3D::new))
        .collect()
}

/// Parse MULTIPOLYGON: `((ext), (hole)), ((ext)), ...`
fn parse_multipolygon3d(s: &str) -> Result<Vec<Polygon3D>> {
    // Each polygon is wrapped in an extra pair of parens
    split_polygon_groups(s)?
        .iter()
        .map(|group| parse_polygon3d(group.trim()))
        .collect()
}

/// Parse GEOMETRYCOLLECTION: recursively parse each sub-geometry.
fn parse_geometrycollection3d(s: &str) -> Result<Vec<Geometry3DEnum>> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_geometries(s)?
        .iter()
        .map(|g| parse_wkt_3d(g.trim()))
        .collect()
}

// ---------------------------------------------------------------------------
// Utility: ring / group splitters
// ---------------------------------------------------------------------------

/// Split a polygon-like string into ring content strings (without parens).
///
/// Input: `(0 0 0, 1 0 0, 1 1 0, 0 0 0), (0.1 0.1 0, ...)`
/// Output: `["0 0 0, 1 0 0, 1 1 0, 0 0 0", "0.1 0.1 0, ..."]`
fn split_rings(s: &str) -> Result<Vec<&str>> {
    let mut rings = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = s.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => {
                if depth == 0 {
                    start = i + 1;
                }
                depth += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    rings.push(&s[start..i]);
                }
            }
            _ => {}
        }
    }

    if rings.is_empty() {
        // Try treating the whole string as a single flat coord list
        rings.push(s);
    }

    Ok(rings)
}

/// Split a MULTIPOLYGON content into individual polygon content strings.
fn split_polygon_groups(s: &str) -> Result<Vec<String>> {
    let mut groups = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;

    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => {
                if depth == 0 {
                    start = i + 1;
                }
                depth += 1;
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    groups.push(s[start..i].to_string());
                }
            }
            _ => {}
        }
    }
    Ok(groups)
}

/// Split a GEOMETRYCOLLECTION content into individual geometry WKT strings.
fn split_geometries(s: &str) -> Result<Vec<String>> {
    let mut geoms = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;

    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b',' if depth == 0 => {
                geoms.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    // Push last geometry
    let last = s[start..].trim();
    if !last.is_empty() {
        geoms.push(last.to_string());
    }
    Ok(geoms)
}

// ---------------------------------------------------------------------------
// Numeric parsing helper
// ---------------------------------------------------------------------------

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>().map_err(|_| {
        GeoSparqlError::InvalidWkt(format!("Cannot parse '{}' as floating-point number", s))
    })
}

// ---------------------------------------------------------------------------
// WKT serialization helpers
// ---------------------------------------------------------------------------

fn points_to_wkt_coords_3d(pts: &[Point3D]) -> String {
    pts.iter()
        .map(|p| format!("{} {} {}", p.x, p.y, p.z))
        .collect::<Vec<_>>()
        .join(", ")
}

fn points_to_wkt_coords_2d(pts: &[Point3D]) -> String {
    pts.iter()
        .map(|p| format!("{} {}", p.x, p.y))
        .collect::<Vec<_>>()
        .join(", ")
}

fn ring_to_wkt_3d(ring: &LinearRing3D) -> String {
    points_to_wkt_coords_3d(&ring.points)
}

fn ring_to_wkt_2d(ring: &LinearRing3D) -> String {
    points_to_wkt_coords_2d(&ring.points)
}

// ---------------------------------------------------------------------------
// Z-range utilities
// ---------------------------------------------------------------------------

fn z_range_of_points(pts: &[Point3D]) -> Option<(f64, f64)> {
    if pts.is_empty() {
        return None;
    }
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for p in pts {
        if p.z < min_z {
            min_z = p.z;
        }
        if p.z > max_z {
            max_z = p.z;
        }
    }
    Some((min_z, max_z))
}

fn merge_z_ranges(ranges: &[(f64, f64)]) -> Option<(f64, f64)> {
    if ranges.is_empty() {
        return None;
    }
    let min_z = ranges
        .iter()
        .map(|(min, _)| *min)
        .fold(f64::INFINITY, f64::min);
    let max_z = ranges
        .iter()
        .map(|(_, max)| *max)
        .fold(f64::NEG_INFINITY, f64::max);
    Some((min_z, max_z))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Point3D ----

    #[test]
    fn test_point3d_new() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
        assert!(p.srid.is_none());
    }

    #[test]
    fn test_point3d_with_srid() {
        let p = Point3D::with_srid(10.0, 20.0, 100.0, 4326);
        assert_eq!(p.srid, Some(4326));
    }

    #[test]
    fn test_point3d_distance_3d() {
        let origin = Point3D::new(0.0, 0.0, 0.0);
        let p = Point3D::new(3.0, 4.0, 0.0);
        assert!((origin.distance_3d(&p) - 5.0).abs() < 1e-10);

        let q = Point3D::new(3.0, 4.0, 12.0);
        assert!((origin.distance_3d(&q) - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_to_2d() {
        let p = Point3D::new(1.5, 2.5, 99.0);
        assert_eq!(p.to_2d(), (1.5, 2.5));
    }

    #[test]
    fn test_point3d_midpoint() {
        let a = Point3D::new(0.0, 0.0, 0.0);
        let b = Point3D::new(2.0, 4.0, 6.0);
        let mid = a.midpoint(&b);
        assert!((mid.x - 1.0).abs() < 1e-10);
        assert!((mid.y - 2.0).abs() < 1e-10);
        assert!((mid.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_lerp() {
        let a = Point3D::new(0.0, 0.0, 0.0);
        let b = Point3D::new(10.0, 20.0, 30.0);
        let p = a.lerp(&b, 0.5);
        assert!((p.x - 5.0).abs() < 1e-10);
        assert!((p.y - 10.0).abs() < 1e-10);
        assert!((p.z - 15.0).abs() < 1e-10);
    }

    // ---- LinearRing3D ----

    #[test]
    fn test_linear_ring_auto_close() {
        let pts = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        let ring = LinearRing3D::new(pts);
        assert!(ring.is_closed());
        assert_eq!(ring.points.len(), 5); // auto-closed
    }

    #[test]
    fn test_linear_ring_already_closed() {
        let pts = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 0.0, 0.0),
        ];
        let ring = LinearRing3D::new(pts);
        assert!(ring.is_closed());
        assert_eq!(ring.points.len(), 3); // not duplicated
    }

    #[test]
    fn test_linear_ring_area() {
        // Unit square in XY plane
        let pts = vec![
            Point3D::new(0.0, 0.0, 5.0),
            Point3D::new(1.0, 0.0, 5.0),
            Point3D::new(1.0, 1.0, 5.0),
            Point3D::new(0.0, 1.0, 5.0),
        ];
        let ring = LinearRing3D::new(pts);
        assert!((ring.area_2d() - 1.0).abs() < 1e-10);
    }

    // ---- BoundingBox3D ----

    #[test]
    fn test_bbox3d_intersects() {
        let a = BoundingBox3D::new(0.0, 2.0, 0.0, 2.0, 0.0, 2.0);
        let b = BoundingBox3D::new(1.0, 3.0, 1.0, 3.0, 1.0, 3.0);
        assert!(a.intersects(&b));

        let c = BoundingBox3D::new(3.0, 5.0, 3.0, 5.0, 3.0, 5.0);
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_bbox3d_contains_point() {
        let bbox = BoundingBox3D::new(0.0, 10.0, 0.0, 10.0, 0.0, 10.0);
        let inside = Point3D::new(5.0, 5.0, 5.0);
        let outside = Point3D::new(15.0, 5.0, 5.0);
        assert!(bbox.contains_point(&inside));
        assert!(!bbox.contains_point(&outside));
    }

    #[test]
    fn test_bbox3d_expand_by() {
        let bbox = BoundingBox3D::new(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let expanded = bbox.expand_by(0.5);
        assert!((expanded.min_x - (-0.5)).abs() < 1e-10);
        assert!((expanded.max_x - 1.5).abs() < 1e-10);
        assert!((expanded.min_z - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_bbox3d_volume() {
        let bbox = BoundingBox3D::new(0.0, 3.0, 0.0, 4.0, 0.0, 5.0);
        assert!((bbox.volume() - 60.0).abs() < 1e-10);
    }

    // ---- WKT roundtrip ----

    #[test]
    fn test_wkt_point_z_roundtrip() {
        let wkt = "POINT Z(1.5 2.5 3.5)";
        let geom = Geometry3DEnum::from_wkt(wkt).expect("parse POINT Z");
        assert!(matches!(geom, Geometry3DEnum::Point(_)));
        let out = geom.to_wkt();
        let geom2 = Geometry3DEnum::from_wkt(&out).expect("reparse");
        if let (Geometry3DEnum::Point(p1), Geometry3DEnum::Point(p2)) = (&geom, &geom2) {
            assert!((p1.x - p2.x).abs() < 1e-10);
            assert!((p1.y - p2.y).abs() < 1e-10);
            assert!((p1.z - p2.z).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wkt_linestring_z_roundtrip() {
        let wkt = "LINESTRING Z(0 0 10, 1 1 20, 2 2 30)";
        let geom = Geometry3DEnum::from_wkt(wkt).expect("parse LINESTRING Z");
        let out = geom.to_wkt();
        let geom2 = Geometry3DEnum::from_wkt(&out).expect("reparse");
        if let (Geometry3DEnum::LineString(ls1), Geometry3DEnum::LineString(ls2)) = (&geom, &geom2)
        {
            assert_eq!(ls1.points.len(), ls2.points.len());
            for (a, b) in ls1.points.iter().zip(ls2.points.iter()) {
                assert!((a.z - b.z).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_wkt_polygon_z_roundtrip() {
        let wkt = "POLYGON Z((0 0 10, 4 0 20, 4 4 30, 0 4 40, 0 0 10))";
        let geom = Geometry3DEnum::from_wkt(wkt).expect("parse POLYGON Z");
        let out = geom.to_wkt();
        let geom2 = Geometry3DEnum::from_wkt(&out).expect("reparse");
        if let (Geometry3DEnum::Polygon(p1), Geometry3DEnum::Polygon(p2)) = (&geom, &geom2) {
            assert_eq!(p1.exterior.points.len(), p2.exterior.points.len());
        }
    }

    #[test]
    fn test_wkt_multipoint_z_roundtrip() {
        let wkt = "MULTIPOINT Z((1 2 3), (4 5 6))";
        let geom = Geometry3DEnum::from_wkt(wkt).expect("parse MULTIPOINT Z");
        let out = geom.to_wkt();
        let geom2 = Geometry3DEnum::from_wkt(&out).expect("reparse");
        if let (Geometry3DEnum::MultiPoint(pts1), Geometry3DEnum::MultiPoint(pts2)) =
            (&geom, &geom2)
        {
            assert_eq!(pts1.len(), pts2.len());
        }
    }

    // ---- z_range / bounding_box_3d ----

    #[test]
    fn test_z_range_point() {
        let g = Geometry3DEnum::Point(Point3D::new(1.0, 2.0, 42.0));
        assert_eq!(g.z_range(), Some((42.0, 42.0)));
    }

    #[test]
    fn test_z_range_linestring() {
        let ls = LineString3D::new(vec![
            Point3D::new(0.0, 0.0, -5.0),
            Point3D::new(1.0, 1.0, 100.0),
            Point3D::new(2.0, 2.0, 50.0),
        ]);
        let g = Geometry3DEnum::LineString(ls);
        assert_eq!(g.z_range(), Some((-5.0, 100.0)));
    }

    #[test]
    fn test_bounding_box_3d() {
        let ls = LineString3D::new(vec![
            Point3D::new(1.0, 2.0, 3.0),
            Point3D::new(5.0, 6.0, 7.0),
        ]);
        let g = Geometry3DEnum::LineString(ls);
        let bbox = g.bounding_box_3d().expect("has bbox");
        assert!((bbox.min_x - 1.0).abs() < 1e-10);
        assert!((bbox.max_z - 7.0).abs() < 1e-10);
    }

    // ---- 2D projection ----

    #[test]
    fn test_to_2d_wkt_point() {
        let g = Geometry3DEnum::Point(Point3D::new(1.0, 2.0, 999.0));
        let wkt = g.to_2d_wkt();
        assert_eq!(wkt, "POINT(1 2)");
    }

    #[test]
    fn test_display_impl() {
        let g = Geometry3DEnum::Point(Point3D::new(0.0, 0.0, 0.0));
        let s = format!("{}", g);
        assert!(s.starts_with("POINT Z"));
    }
}

// ---------------------------------------------------------------------------
// Additional tests (OGC GeoSPARQL 1.1 / 3D geometry coverage)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests_extended {
    use super::*;

    // ── Point3D ─────────────────────────────────────────────────────────────

    #[test]
    fn test_point3d_with_srid() {
        let p = Point3D::with_srid(1.0, 2.0, 3.0, 4326);
        assert_eq!(p.srid, Some(4326));
        assert!((p.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_distance_3d_identity() {
        let p = Point3D::new(3.0, 4.0, 5.0);
        assert!((p.distance_3d(&p)).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_distance_3d_known() {
        let a = Point3D::new(0.0, 0.0, 0.0);
        let b = Point3D::new(1.0, 2.0, 2.0);
        // sqrt(1+4+4) = 3
        assert!((a.distance_3d(&b) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_distance_2d_ignores_z() {
        let a = Point3D::new(0.0, 0.0, 0.0);
        let b = Point3D::new(3.0, 4.0, 999.0);
        // 2D distance = 5, not affected by z=999
        assert!((a.distance_2d(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_to_2d() {
        let p = Point3D::new(7.5, 8.5, 9.5);
        let (x, y) = p.to_2d();
        assert!((x - 7.5).abs() < 1e-10);
        assert!((y - 8.5).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_lerp_at_zero() {
        let a = Point3D::new(1.0, 2.0, 3.0);
        let b = Point3D::new(10.0, 20.0, 30.0);
        let p = a.lerp(&b, 0.0);
        assert!((p.x - 1.0).abs() < 1e-10);
        assert!((p.y - 2.0).abs() < 1e-10);
        assert!((p.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_lerp_at_one() {
        let a = Point3D::new(1.0, 2.0, 3.0);
        let b = Point3D::new(10.0, 20.0, 30.0);
        let p = a.lerp(&b, 1.0);
        assert!((p.x - 10.0).abs() < 1e-10);
        assert!((p.y - 20.0).abs() < 1e-10);
        assert!((p.z - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_display() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let s = format!("{}", p);
        assert!(s.contains("1") && s.contains("2") && s.contains("3"));
    }

    // ── LineString3D ─────────────────────────────────────────────────────────

    #[test]
    fn test_linestring3d_length_3d() {
        let ls = LineString3D::new(vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
        ]);
        assert!((ls.length_3d() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linestring3d_length_2d() {
        let ls = LineString3D::new(vec![
            Point3D::new(0.0, 0.0, 100.0),
            Point3D::new(3.0, 4.0, 200.0),
        ]);
        assert!((ls.length_2d() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linestring3d_empty_is_open() {
        let ls = LineString3D::new(vec![]);
        assert!(!ls.is_closed());
    }

    #[test]
    fn test_linestring3d_z_range_empty() {
        let ls = LineString3D::new(vec![]);
        assert!(ls.z_range().is_none());
    }

    // ── Polygon3D ────────────────────────────────────────────────────────────

    #[test]
    fn test_polygon3d_area_with_hole() {
        // 4×4 exterior minus 2×2 hole
        let exterior = LinearRing3D::new(vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(4.0, 0.0, 0.0),
            Point3D::new(4.0, 4.0, 0.0),
            Point3D::new(0.0, 4.0, 0.0),
        ]);
        let hole = LinearRing3D::new(vec![
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(3.0, 1.0, 0.0),
            Point3D::new(3.0, 3.0, 0.0),
            Point3D::new(1.0, 3.0, 0.0),
        ]);
        let poly = Polygon3D::with_holes(exterior, vec![hole]);
        let area = poly.area_2d();
        // exterior area = 16, hole area = 4, net = 12
        assert!((area - 12.0).abs() < 1e-8);
    }

    #[test]
    fn test_polygon3d_z_range() {
        let exterior = LinearRing3D::new(vec![
            Point3D::new(0.0, 0.0, 10.0),
            Point3D::new(1.0, 0.0, 20.0),
            Point3D::new(1.0, 1.0, 30.0),
            Point3D::new(0.0, 1.0, 40.0),
        ]);
        let poly = Polygon3D::new(exterior);
        let (min_z, max_z) = poly.z_range().expect("should succeed");
        assert!((min_z - 10.0).abs() < 1e-10);
        assert!((max_z - 40.0).abs() < 1e-10);
    }

    // ── BoundingBox3D extended ────────────────────────────────────────────────

    #[test]
    fn test_bbox3d_contains_bbox_true() {
        let outer = BoundingBox3D::new(0.0, 10.0, 0.0, 10.0, 0.0, 10.0);
        let inner = BoundingBox3D::new(2.0, 8.0, 2.0, 8.0, 2.0, 8.0);
        assert!(outer.contains_bbox(&inner));
        assert!(!inner.contains_bbox(&outer));
    }

    #[test]
    fn test_bbox3d_union() {
        let a = BoundingBox3D::new(0.0, 5.0, 0.0, 5.0, 0.0, 5.0);
        let b = BoundingBox3D::new(3.0, 8.0, 3.0, 8.0, 3.0, 8.0);
        let u = a.union(&b);
        assert!((u.min_x - 0.0).abs() < 1e-10);
        assert!((u.max_x - 8.0).abs() < 1e-10);
        assert!((u.max_z - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox3d_center() {
        let bbox = BoundingBox3D::new(0.0, 10.0, 0.0, 10.0, 0.0, 10.0);
        let c = bbox.center();
        assert!((c.x - 5.0).abs() < 1e-10);
        assert!((c.y - 5.0).abs() < 1e-10);
        assert!((c.z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox3d_from_point() {
        let p = Point3D::new(3.0, 4.0, 5.0);
        let bbox = BoundingBox3D::from_point(&p);
        assert!((bbox.min_x - 3.0).abs() < 1e-10);
        assert!((bbox.max_x - 3.0).abs() < 1e-10);
        assert!((bbox.volume() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox3d_display() {
        let bbox = BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let s = format!("{}", bbox);
        assert!(s.starts_with("BOX3D("));
    }

    // ── Geometry3DEnum WKT edge cases ─────────────────────────────────────────

    #[test]
    fn test_wkt_multilinestring_z_roundtrip() {
        let wkt = "MULTILINESTRING Z((0 0 1, 1 1 2), (3 3 3, 4 4 4))";
        let g = Geometry3DEnum::from_wkt(wkt).expect("parse");
        let out = g.to_wkt();
        let g2 = Geometry3DEnum::from_wkt(&out).expect("reparse");
        if let (Geometry3DEnum::MultiLineString(mls1), Geometry3DEnum::MultiLineString(mls2)) =
            (&g, &g2)
        {
            assert_eq!(mls1.len(), mls2.len());
            for (ls1, ls2) in mls1.iter().zip(mls2.iter()) {
                assert_eq!(ls1.points.len(), ls2.points.len());
            }
        } else {
            panic!("unexpected types");
        }
    }

    #[test]
    fn test_wkt_geometrycollection_z_roundtrip() {
        let wkt = "GEOMETRYCOLLECTION Z(POINT Z(1 2 3), LINESTRING Z(0 0 0, 1 1 1))";
        let g = Geometry3DEnum::from_wkt(wkt).expect("parse");
        let out = g.to_wkt();
        let g2 = Geometry3DEnum::from_wkt(&out).expect("reparse");
        if let (Geometry3DEnum::GeometryCollection(v1), Geometry3DEnum::GeometryCollection(v2)) =
            (&g, &g2)
        {
            assert_eq!(v1.len(), v2.len());
        } else {
            panic!("expected GeometryCollection");
        }
    }

    #[test]
    fn test_wkt_multipolygon_z_roundtrip() {
        let wkt = "MULTIPOLYGON Z(((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0)), ((2 2 1, 3 2 1, 3 3 1, 2 3 1, 2 2 1)))";
        let g = Geometry3DEnum::from_wkt(wkt).expect("parse");
        let out = g.to_wkt();
        let g2 = Geometry3DEnum::from_wkt(&out).expect("reparse");
        if let (Geometry3DEnum::MultiPolygon(mp1), Geometry3DEnum::MultiPolygon(mp2)) = (&g, &g2) {
            assert_eq!(mp1.len(), mp2.len());
        } else {
            panic!("expected MultiPolygon");
        }
    }

    #[test]
    fn test_wkt_invalid_type_error() {
        let result = Geometry3DEnum::from_wkt("TRIANGLE Z(0 0 0, 1 0 0, 0 1 0)");
        assert!(result.is_err());
    }

    #[test]
    fn test_wkt_srid_prefix_stripped() {
        let wkt = "SRID=4326;POINT Z(10 20 30)";
        let g = Geometry3DEnum::from_wkt(wkt).expect("parse with SRID prefix");
        assert!(matches!(g, Geometry3DEnum::Point(_)));
    }

    #[test]
    fn test_geometry_type_names() {
        let cases: &[(&str, &str)] = &[
            ("POINT Z(0 0 0)", "Point"),
            ("LINESTRING Z(0 0 0, 1 1 1)", "LineString"),
            ("POLYGON Z((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0))", "Polygon"),
            ("MULTIPOINT Z((0 0 0), (1 1 1))", "MultiPoint"),
        ];
        for (wkt, expected) in cases {
            let g = Geometry3DEnum::from_wkt(wkt).expect(wkt);
            assert_eq!(g.geometry_type_name(), *expected);
        }
    }

    #[test]
    fn test_num_points_collection() {
        let g = Geometry3DEnum::from_wkt(
            "GEOMETRYCOLLECTION Z(POINT Z(1 2 3), LINESTRING Z(0 0 0, 1 1 1))",
        )
        .expect("parse");
        assert_eq!(g.num_points(), 3); // 1 + 2
    }

    #[test]
    fn test_bounding_box_3d_polygon() {
        let wkt = "POLYGON Z((0 0 10, 5 0 20, 5 5 30, 0 5 40, 0 0 10))";
        let g = Geometry3DEnum::from_wkt(wkt).expect("parse");
        let bbox = g.bounding_box_3d().expect("bbox");
        assert!((bbox.min_x - 0.0).abs() < 1e-10);
        assert!((bbox.max_x - 5.0).abs() < 1e-10);
        assert!((bbox.min_z - 10.0).abs() < 1e-10);
        assert!((bbox.max_z - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_2d_wkt_linestring() {
        let g = Geometry3DEnum::from_wkt("LINESTRING Z(0 0 100, 1 1 200)").expect("parse");
        let wkt2d = g.to_2d_wkt();
        assert!(wkt2d.starts_with("LINESTRING"));
        assert!(!wkt2d.contains('Z'));
        assert!(!wkt2d.contains("100"));
    }

    #[test]
    fn test_z_range_multipoint() {
        let g = Geometry3DEnum::MultiPoint(vec![
            Point3D::new(0.0, 0.0, 5.0),
            Point3D::new(1.0, 1.0, 15.0),
            Point3D::new(2.0, 2.0, 10.0),
        ]);
        let (min_z, max_z) = g.z_range().expect("z_range");
        assert!((min_z - 5.0).abs() < 1e-10);
        assert!((max_z - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_z_range_multipolygon() {
        let p1 = Polygon3D::new(LinearRing3D::new(vec![
            Point3D::new(0.0, 0.0, 1.0),
            Point3D::new(1.0, 0.0, 2.0),
            Point3D::new(1.0, 1.0, 3.0),
            Point3D::new(0.0, 1.0, 4.0),
        ]));
        let p2 = Polygon3D::new(LinearRing3D::new(vec![
            Point3D::new(5.0, 5.0, 10.0),
            Point3D::new(6.0, 5.0, 20.0),
            Point3D::new(6.0, 6.0, 30.0),
            Point3D::new(5.0, 6.0, 40.0),
        ]));
        let g = Geometry3DEnum::MultiPolygon(vec![p1, p2]);
        let (min_z, max_z) = g.z_range().expect("z_range");
        assert!((min_z - 1.0).abs() < 1e-10);
        assert!((max_z - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometry_collection_bounding_box() {
        let g = Geometry3DEnum::GeometryCollection(vec![
            Geometry3DEnum::Point(Point3D::new(-5.0, -5.0, -5.0)),
            Geometry3DEnum::Point(Point3D::new(5.0, 5.0, 5.0)),
        ]);
        let bbox = g.bounding_box_3d().expect("bbox");
        assert!((bbox.min_x - (-5.0)).abs() < 1e-10);
        assert!((bbox.max_z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linestring3d_length_3d_diagonal() {
        // Diagonal with z component: sqrt(1+1+1) = sqrt(3) per segment, 3 segments
        let ls = LineString3D::new(vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 1.0),
            Point3D::new(2.0, 2.0, 2.0),
            Point3D::new(3.0, 3.0, 3.0),
        ]);
        let expected = 3.0 * (3.0_f64).sqrt();
        assert!((ls.length_3d() - expected).abs() < 1e-8);
    }
}
