//! Geometric primitives for spatial indexing
//!
//! Wraps the `geo` crate types with RDF/GeoSPARQL semantics.

use geo::{
    Coord, Geometry as GeoGeometry, LineString as GeoLineString, Point as GeoPoint,
    Polygon as GeoPolygon,
};
use geojson::GeoJson;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use wkt::Wkt;

use crate::error::{Result, TdbError};

/// Unified geometry type for spatial indexing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Geometry {
    /// Point geometry (0-dimensional)
    Point(Point),
    /// LineString geometry (1-dimensional)
    LineString(LineString),
    /// Polygon geometry (2-dimensional)
    Polygon(Polygon),
}

impl Geometry {
    /// Get bounding box of this geometry
    pub fn bounding_box(&self) -> BoundingBox {
        match self {
            Geometry::Point(p) => p.bounding_box(),
            Geometry::LineString(ls) => ls.bounding_box(),
            Geometry::Polygon(poly) => poly.bounding_box(),
        }
    }

    /// Check if this geometry contains a point
    pub fn contains(&self, point: &Point) -> bool {
        match self {
            Geometry::Point(p) => p.equals(point),
            Geometry::LineString(_ls) => false, // LineStrings don't contain points
            Geometry::Polygon(poly) => poly.contains(point),
        }
    }

    /// Calculate distance to a point (in degrees)
    pub fn distance_to(&self, point: &Point) -> f64 {
        match self {
            Geometry::Point(p) => p.distance_to(point),
            Geometry::LineString(ls) => ls.distance_to(point),
            Geometry::Polygon(poly) => poly.distance_to(point),
        }
    }

    /// Parse from Well-Known Text (WKT)
    pub fn from_wkt(wkt_str: &str) -> Result<Self> {
        let wkt = Wkt::from_str(wkt_str)
            .map_err(|e| TdbError::InvalidInput(format!("Failed to parse WKT: {}", e)))?;

        // Convert WKT to geo types
        let geo_geom: GeoGeometry<f64> = wkt.try_into().map_err(|e| {
            TdbError::InvalidInput(format!("Failed to convert WKT to geometry: {:?}", e))
        })?;

        Self::from_geo(geo_geom)
    }

    /// Convert from geo::Geometry
    pub fn from_geo(geo: GeoGeometry<f64>) -> Result<Self> {
        match geo {
            GeoGeometry::Point(p) => Ok(Geometry::Point(Point::from_geo(p))),
            GeoGeometry::LineString(ls) => Ok(Geometry::LineString(LineString::from_geo(ls))),
            GeoGeometry::Polygon(poly) => Ok(Geometry::Polygon(Polygon::from_geo(poly))),
            _ => Err(TdbError::InvalidInput(
                "Unsupported geometry type".to_string(),
            )),
        }
    }

    /// Convert to geo::Geometry
    pub fn to_geo(&self) -> GeoGeometry<f64> {
        match self {
            Geometry::Point(p) => GeoGeometry::Point(p.to_geo()),
            Geometry::LineString(ls) => GeoGeometry::LineString(ls.to_geo()),
            Geometry::Polygon(poly) => GeoGeometry::Polygon(poly.to_geo()),
        }
    }

    /// Serialize to WKT
    pub fn to_wkt(&self) -> String {
        use wkt::ToWkt;
        self.to_geo().to_wkt().to_string()
    }
}

/// 2D Point (latitude, longitude)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Latitude (Y coordinate, -90 to 90)
    pub lat: f64,
    /// Longitude (X coordinate, -180 to 180)
    pub lon: f64,
}

impl Point {
    /// Create a new point
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Create from geo::Point
    pub fn from_geo(point: GeoPoint<f64>) -> Self {
        Self {
            lat: point.y(),
            lon: point.x(),
        }
    }

    /// Convert to geo::Point
    pub fn to_geo(&self) -> GeoPoint<f64> {
        GeoPoint::new(self.lon, self.lat)
    }

    /// Calculate great circle distance to another point (in meters)
    ///
    /// Uses the Haversine formula for accuracy
    pub fn distance_to(&self, other: &Point) -> f64 {
        use geo::{Distance, Haversine};
        let p1 = self.to_geo();
        let p2 = other.to_geo();
        Haversine::distance(p1, p2)
    }

    /// Check if two points are equal (with floating point tolerance)
    pub fn equals(&self, other: &Point) -> bool {
        const EPSILON: f64 = 1e-9;
        (self.lat - other.lat).abs() < EPSILON && (self.lon - other.lon).abs() < EPSILON
    }

    /// Get bounding box (point to itself)
    pub fn bounding_box(&self) -> BoundingBox {
        BoundingBox::new(self.lat, self.lon, self.lat, self.lon)
    }
}

impl From<Point> for Geometry {
    fn from(p: Point) -> Self {
        Geometry::Point(p)
    }
}

/// 2D Bounding Box
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Minimum latitude
    pub min_lat: f64,
    /// Minimum longitude
    pub min_lon: f64,
    /// Maximum latitude
    pub max_lat: f64,
    /// Maximum longitude
    pub max_lon: f64,
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Self {
        Self {
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        }
    }

    /// Create from two corner points
    pub fn from_points(p1: Point, p2: Point) -> Self {
        Self::new(
            p1.lat.min(p2.lat),
            p1.lon.min(p2.lon),
            p1.lat.max(p2.lat),
            p1.lon.max(p2.lon),
        )
    }

    /// Check if this bbox intersects another
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        !(self.max_lat < other.min_lat
            || self.min_lat > other.max_lat
            || self.max_lon < other.min_lon
            || self.min_lon > other.max_lon)
    }

    /// Check if this bbox contains a point
    pub fn contains_point(&self, point: &Point) -> bool {
        point.lat >= self.min_lat
            && point.lat <= self.max_lat
            && point.lon >= self.min_lon
            && point.lon <= self.max_lon
    }

    /// Expand bbox to include a point
    pub fn expand(&mut self, point: &Point) {
        self.min_lat = self.min_lat.min(point.lat);
        self.min_lon = self.min_lon.min(point.lon);
        self.max_lat = self.max_lat.max(point.lat);
        self.max_lon = self.max_lon.max(point.lon);
    }

    /// Get center point
    pub fn center(&self) -> Point {
        Point::new(
            (self.min_lat + self.max_lat) / 2.0,
            (self.min_lon + self.max_lon) / 2.0,
        )
    }

    /// Get area (in square degrees)
    pub fn area(&self) -> f64 {
        (self.max_lat - self.min_lat) * (self.max_lon - self.min_lon)
    }
}

/// LineString (connected line segments)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineString {
    /// Points defining the line
    pub points: Vec<Point>,
}

impl LineString {
    /// Create a new linestring
    pub fn new(points: Vec<Point>) -> Result<Self> {
        if points.len() < 2 {
            return Err(TdbError::InvalidInput(
                "LineString must have at least 2 points".to_string(),
            ));
        }
        Ok(Self { points })
    }

    /// Create from geo::LineString
    pub fn from_geo(ls: GeoLineString<f64>) -> Self {
        let points = ls.coords().map(|c| Point::new(c.y, c.x)).collect();
        Self { points }
    }

    /// Convert to geo::LineString
    pub fn to_geo(&self) -> GeoLineString<f64> {
        let coords: Vec<Coord<f64>> = self
            .points
            .iter()
            .map(|p| Coord { x: p.lon, y: p.lat })
            .collect();
        GeoLineString::new(coords)
    }

    /// Get bounding box
    pub fn bounding_box(&self) -> BoundingBox {
        let mut bbox = self.points[0].bounding_box();
        for point in &self.points[1..] {
            bbox.expand(point);
        }
        bbox
    }

    /// Calculate distance to a point
    pub fn distance_to(&self, point: &Point) -> f64 {
        use geo::{Distance, Euclidean};
        let ls = self.to_geo();
        let p = point.to_geo();
        Euclidean::distance(&ls, &p)
    }
}

impl From<LineString> for Geometry {
    fn from(ls: LineString) -> Self {
        Geometry::LineString(ls)
    }
}

/// Polygon (closed ring with optional holes)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon {
    /// Exterior ring
    pub exterior: Vec<Point>,
    /// Interior rings (holes)
    pub interiors: Vec<Vec<Point>>,
}

impl Polygon {
    /// Create a new polygon
    pub fn new(exterior: Vec<Point>) -> Result<Self> {
        if exterior.len() < 3 {
            return Err(TdbError::InvalidInput(
                "Polygon exterior must have at least 3 points".to_string(),
            ));
        }
        Ok(Self {
            exterior,
            interiors: Vec::new(),
        })
    }

    /// Add an interior ring (hole)
    pub fn add_interior(&mut self, interior: Vec<Point>) -> Result<()> {
        if interior.len() < 3 {
            return Err(TdbError::InvalidInput(
                "Polygon interior must have at least 3 points".to_string(),
            ));
        }
        self.interiors.push(interior);
        Ok(())
    }

    /// Create from geo::Polygon
    pub fn from_geo(poly: GeoPolygon<f64>) -> Self {
        let exterior: Vec<Point> = poly
            .exterior()
            .coords()
            .map(|c| Point::new(c.y, c.x))
            .collect();

        let interiors: Vec<Vec<Point>> = poly
            .interiors()
            .iter()
            .map(|ring| ring.coords().map(|c| Point::new(c.y, c.x)).collect())
            .collect();

        Self {
            exterior,
            interiors,
        }
    }

    /// Convert to geo::Polygon
    pub fn to_geo(&self) -> GeoPolygon<f64> {
        let exterior_coords: Vec<Coord<f64>> = self
            .exterior
            .iter()
            .map(|p| Coord { x: p.lon, y: p.lat })
            .collect();

        let exterior_line = GeoLineString::new(exterior_coords);

        let interior_lines: Vec<GeoLineString<f64>> = self
            .interiors
            .iter()
            .map(|ring| {
                let coords: Vec<Coord<f64>> =
                    ring.iter().map(|p| Coord { x: p.lon, y: p.lat }).collect();
                GeoLineString::new(coords)
            })
            .collect();

        GeoPolygon::new(exterior_line, interior_lines)
    }

    /// Get bounding box
    pub fn bounding_box(&self) -> BoundingBox {
        let mut bbox = self.exterior[0].bounding_box();
        for point in &self.exterior[1..] {
            bbox.expand(point);
        }
        bbox
    }

    /// Check if polygon contains a point
    pub fn contains(&self, point: &Point) -> bool {
        use geo::Contains;
        let poly = self.to_geo();
        let p = point.to_geo();
        poly.contains(&p)
    }

    /// Calculate distance to a point
    pub fn distance_to(&self, point: &Point) -> f64 {
        use geo::{Distance, Euclidean};
        let poly = self.to_geo();
        let p = point.to_geo();
        Euclidean::distance(&poly, &p)
    }
}

impl From<Polygon> for Geometry {
    fn from(poly: Polygon) -> Self {
        Geometry::Polygon(poly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point::new(40.7128, -74.0060);
        assert_eq!(p.lat, 40.7128);
        assert_eq!(p.lon, -74.0060);
    }

    #[test]
    fn test_point_distance() {
        let nyc = Point::new(40.7128, -74.0060);
        let la = Point::new(34.0522, -118.2437);

        let distance = nyc.distance_to(&la);
        // Distance between NYC and LA is approximately 3,944 km
        assert!(distance > 3_900_000.0 && distance < 4_000_000.0);
    }

    #[test]
    fn test_point_equality() {
        let p1 = Point::new(40.7128, -74.0060);
        let p2 = Point::new(40.7128, -74.0060);
        let p3 = Point::new(40.7129, -74.0060);

        assert!(p1.equals(&p2));
        assert!(!p1.equals(&p3));
    }

    #[test]
    fn test_bounding_box_creation() {
        let bbox = BoundingBox::new(40.0, -75.0, 41.0, -74.0);
        assert_eq!(bbox.min_lat, 40.0);
        assert_eq!(bbox.max_lon, -74.0);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(40.0, -75.0, 41.0, -74.0);
        let inside = Point::new(40.5, -74.5);
        let outside = Point::new(42.0, -74.5);

        assert!(bbox.contains_point(&inside));
        assert!(!bbox.contains_point(&outside));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let bbox1 = BoundingBox::new(40.0, -75.0, 41.0, -74.0);
        let bbox2 = BoundingBox::new(40.5, -74.5, 41.5, -73.5);
        let bbox3 = BoundingBox::new(42.0, -75.0, 43.0, -74.0);

        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.intersects(&bbox3));
    }

    #[test]
    fn test_bounding_box_expand() {
        let mut bbox = BoundingBox::new(40.0, -75.0, 41.0, -74.0);
        let point = Point::new(42.0, -73.0);

        bbox.expand(&point);
        assert_eq!(bbox.max_lat, 42.0);
        assert_eq!(bbox.max_lon, -73.0);
    }

    #[test]
    fn test_linestring_creation() {
        let points = vec![Point::new(40.7128, -74.0060), Point::new(40.7589, -73.9851)];
        let ls = LineString::new(points).unwrap();
        assert_eq!(ls.points.len(), 2);
    }

    #[test]
    fn test_linestring_minimum_points() {
        let points = vec![Point::new(40.7128, -74.0060)];
        let result = LineString::new(points);
        assert!(result.is_err());
    }

    #[test]
    fn test_polygon_creation() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0), // Close the ring
        ];
        let poly = Polygon::new(points).unwrap();
        assert_eq!(poly.exterior.len(), 5);
    }

    #[test]
    fn test_polygon_contains() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let poly = Polygon::new(points).unwrap();

        let inside = Point::new(0.5, 0.5);
        let outside = Point::new(2.0, 2.0);

        assert!(poly.contains(&inside));
        assert!(!poly.contains(&outside));
    }

    #[test]
    fn test_geometry_from_wkt_point() {
        let wkt = "POINT(40.7128 -74.0060)";
        let geom = Geometry::from_wkt(wkt).unwrap();
        assert!(matches!(geom, Geometry::Point(_)));
    }

    #[test]
    fn test_geometry_to_wkt() {
        let point = Point::new(40.7128, -74.0060);
        let geom = Geometry::Point(point);
        let wkt = geom.to_wkt();
        assert!(wkt.contains("POINT"));
    }

    #[test]
    fn test_geometry_bounding_box() {
        let point = Point::new(40.7128, -74.0060);
        let geom = Geometry::Point(point);
        let bbox = geom.bounding_box();
        assert_eq!(bbox.min_lat, 40.7128);
        assert_eq!(bbox.max_lat, 40.7128);
    }
}
