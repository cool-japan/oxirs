//! 3D Geometry Types
//!
//! Defines the core 3D geometry value types (`Point3D`, `LinearRing3D`,
//! `LineString3D`, `Polygon3D`, `BoundingBox3D`, and the [`Geometry3DEnum`]
//! variant) along with their geometric operations: distances, areas,
//! bounding boxes, z-ranges, WKT (de)serialization, and 2D flattening.
//!
//! WKT parsing/serialization and Z-range helpers live in the sibling
//! [`super::geometry3d_wkt`] module.

use super::geometry3d_wkt::{
    merge_z_ranges, parse_wkt_3d, points_to_wkt_coords_2d, points_to_wkt_coords_3d, ring_to_wkt_2d,
    ring_to_wkt_3d, z_range_of_points,
};
use crate::error::Result;
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
