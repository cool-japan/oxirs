//! Geometry types and operations
//!
//! This module provides the core geometry types and operations for GeoSPARQL,
//! wrapping the `geo` crate and providing WKT/GML serialization support.

pub mod wkt_parser;

#[cfg(feature = "gml-support")]
pub mod gml_parser;

use crate::error::{GeoSparqlError, Result};
use geo_types::Geometry as GeoGeometry;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Spatial Reference System (CRS) identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Crs {
    /// The CRS URI (e.g., "http://www.opengis.net/def/crs/EPSG/0/4326")
    pub uri: String,
}

impl Crs {
    /// Create a new CRS from a URI
    pub fn new(uri: impl Into<String>) -> Self {
        Self { uri: uri.into() }
    }

    /// Create the default CRS (WGS84/CRS84)
    pub fn default_crs() -> Self {
        Self::new(crate::vocabulary::DEFAULT_CRS)
    }

    /// Create an EPSG CRS
    pub fn epsg(code: u32) -> Self {
        Self::new(format!("{}{}", crate::vocabulary::EPSG_PREFIX, code))
    }

    /// Check if this is the default CRS
    pub fn is_default(&self) -> bool {
        self.uri == crate::vocabulary::DEFAULT_CRS
            || self.uri == "http://www.opengis.net/def/crs/EPSG/0/4326"
    }

    /// Extract EPSG code from CRS URI if it's an EPSG CRS
    ///
    /// Returns Some(code) if this is an EPSG CRS, None otherwise.
    ///
    /// # Examples
    /// ```
    /// use oxirs_geosparql::geometry::Crs;
    ///
    /// let crs = Crs::epsg(4326);
    /// assert_eq!(crs.epsg_code(), Some(4326));
    ///
    /// let crs = Crs::new("http://www.opengis.net/def/crs/OGC/1.3/CRS84");
    /// assert_eq!(crs.epsg_code(), None);
    /// ```
    pub fn epsg_code(&self) -> Option<u32> {
        // Check if URI starts with EPSG prefix
        if let Some(code_str) = self.uri.strip_prefix(crate::vocabulary::EPSG_PREFIX) {
            code_str.parse().ok()
        } else {
            None
        }
    }
}

impl Default for Crs {
    fn default() -> Self {
        Self::default_crs()
    }
}

impl fmt::Display for Crs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.uri)
    }
}

/// A geometry with an associated Coordinate Reference System
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    /// The underlying geometry
    pub geom: GeoGeometry<f64>,
    /// The Coordinate Reference System
    pub crs: Crs,
}

impl Geometry {
    /// Create a new geometry with the default CRS
    pub fn new(geom: GeoGeometry<f64>) -> Self {
        Self {
            geom,
            crs: Crs::default(),
        }
    }

    /// Create a new geometry with a specific CRS
    pub fn with_crs(geom: GeoGeometry<f64>, crs: Crs) -> Self {
        Self { geom, crs }
    }

    /// Parse from WKT (Well-Known Text) format
    pub fn from_wkt(wkt: &str) -> Result<Self> {
        wkt_parser::parse_wkt(wkt)
    }

    /// Convert to WKT format
    pub fn to_wkt(&self) -> String {
        wkt_parser::geometry_to_wkt(&self.geom)
    }

    /// Parse from GML (Geography Markup Language) format
    ///
    /// Requires the `gml-support` feature to be enabled.
    #[cfg(feature = "gml-support")]
    pub fn from_gml(gml: &str) -> Result<Self> {
        gml_parser::parse_gml(gml)
    }

    /// Convert to GML format
    ///
    /// Requires the `gml-support` feature to be enabled.
    #[cfg(feature = "gml-support")]
    pub fn to_gml(&self) -> Result<String> {
        gml_parser::geometry_to_gml(self)
    }

    /// Get the dimension of the geometry (2 or 3)
    pub fn dimension(&self) -> u8 {
        // For now, we only support 2D geometries
        2
    }

    /// Get the coordinate dimension (same as dimension for Simple Features)
    pub fn coordinate_dimension(&self) -> u8 {
        self.dimension()
    }

    /// Get the spatial dimension (0, 1, or 2)
    pub fn spatial_dimension(&self) -> u8 {
        match &self.geom {
            GeoGeometry::Point(_) => 0,
            GeoGeometry::Line(_) | GeoGeometry::LineString(_) => 1,
            GeoGeometry::Polygon(_)
            | GeoGeometry::MultiPoint(_)
            | GeoGeometry::MultiLineString(_)
            | GeoGeometry::MultiPolygon(_)
            | GeoGeometry::GeometryCollection(_)
            | GeoGeometry::Triangle(_)
            | GeoGeometry::Rect(_) => 2,
        }
    }

    /// Check if the geometry is empty
    pub fn is_empty(&self) -> bool {
        use geo::algorithm::HasDimensions;
        match &self.geom {
            GeoGeometry::Point(p) => p.is_empty(),
            GeoGeometry::Line(l) => l.is_empty(),
            GeoGeometry::LineString(ls) => ls.is_empty(),
            GeoGeometry::Polygon(p) => p.is_empty(),
            GeoGeometry::MultiPoint(mp) => mp.is_empty(),
            GeoGeometry::MultiLineString(mls) => mls.is_empty(),
            GeoGeometry::MultiPolygon(mp) => mp.is_empty(),
            GeoGeometry::GeometryCollection(gc) => gc.is_empty(),
            GeoGeometry::Triangle(t) => t.is_empty(),
            GeoGeometry::Rect(r) => r.is_empty(),
        }
    }

    /// Check if the geometry is simple (no self-intersections)
    ///
    /// A geometry is simple if:
    /// - Point: always simple
    /// - LineString: no self-intersections (except at endpoints for closed lines)
    /// - Polygon: no self-intersections, properly formed
    /// - Multi*: all components are simple
    pub fn is_simple(&self) -> bool {
        use geo::algorithm::line_intersection::line_intersection;
        use geo::algorithm::HasDimensions;
        use geo::Line;

        match &self.geom {
            // Points are always simple
            GeoGeometry::Point(_) => true,

            // Line segment is always simple
            GeoGeometry::Line(_) => true,

            // LineString is simple if it doesn't self-intersect (except at endpoints)
            GeoGeometry::LineString(ls) => {
                if ls.is_empty() || ls.0.len() < 2 {
                    return true;
                }

                let coords = &ls.0;
                let is_closed = coords.first() == coords.last();

                // Check for self-intersections
                for i in 0..coords.len() - 1 {
                    let line1 = Line::new(coords[i], coords[i + 1]);

                    for j in (i + 2)..coords.len() - 1 {
                        // Don't check adjacent segments
                        if j == i + 1 {
                            continue;
                        }

                        // For closed lines, allow intersection at endpoints
                        if is_closed && i == 0 && j == coords.len() - 2 {
                            continue;
                        }

                        let line2 = Line::new(coords[j], coords[j + 1]);

                        // Check if lines intersect (not at endpoints)
                        if let Some(intersection) = line_intersection(line1, line2) {
                            use geo::LineIntersection;
                            match intersection {
                                LineIntersection::SinglePoint { intersection, .. } => {
                                    // Intersection is allowed only at shared endpoints
                                    let is_endpoint = intersection == coords[i]
                                        || intersection == coords[i + 1]
                                        || intersection == coords[j]
                                        || intersection == coords[j + 1];
                                    if !is_endpoint {
                                        return false;
                                    }
                                }
                                LineIntersection::Collinear { .. } => {
                                    // Overlapping segments means not simple
                                    return false;
                                }
                            }
                        }
                    }
                }
                true
            }

            // Polygon is simple if exterior and holes don't self-intersect
            GeoGeometry::Polygon(poly) => {
                if poly.is_empty() {
                    return true;
                }

                // Check exterior ring
                let exterior = Geometry::new(GeoGeometry::LineString(poly.exterior().clone()));
                if !exterior.is_simple() {
                    return false;
                }

                // Check each interior ring (holes)
                for hole in poly.interiors() {
                    let hole_geom = Geometry::new(GeoGeometry::LineString(hole.clone()));
                    if !hole_geom.is_simple() {
                        return false;
                    }
                }

                true
            }

            // Multi-geometries are simple if all components are simple
            GeoGeometry::MultiPoint(_) => true, // MultiPoints are always simple

            GeoGeometry::MultiLineString(mls) => {
                for ls in &mls.0 {
                    let geom = Geometry::new(GeoGeometry::LineString(ls.clone()));
                    if !geom.is_simple() {
                        return false;
                    }
                }
                true
            }

            GeoGeometry::MultiPolygon(mp) => {
                for poly in &mp.0 {
                    let geom = Geometry::new(GeoGeometry::Polygon(poly.clone()));
                    if !geom.is_simple() {
                        return false;
                    }
                }
                true
            }

            // GeometryCollection is simple if all components are simple
            GeoGeometry::GeometryCollection(gc) => {
                for geom in &gc.0 {
                    let geometry = Geometry::new(geom.clone());
                    if !geometry.is_simple() {
                        return false;
                    }
                }
                true
            }

            // Triangle and Rect are always simple (by definition)
            GeoGeometry::Triangle(_) | GeoGeometry::Rect(_) => true,
        }
    }

    /// Check if coordinates are 3D (has Z coordinate)
    pub fn is_3d(&self) -> bool {
        // Currently only supporting 2D
        false
    }

    /// Check if coordinates are measured (has M coordinate)
    pub fn is_measured(&self) -> bool {
        // Currently not supporting measured coordinates
        false
    }

    /// Get the geometry type name
    pub fn geometry_type(&self) -> &'static str {
        match &self.geom {
            GeoGeometry::Point(_) => "Point",
            GeoGeometry::Line(_) => "Line",
            GeoGeometry::LineString(_) => "LineString",
            GeoGeometry::Polygon(_) => "Polygon",
            GeoGeometry::MultiPoint(_) => "MultiPoint",
            GeoGeometry::MultiLineString(_) => "MultiLineString",
            GeoGeometry::MultiPolygon(_) => "MultiPolygon",
            GeoGeometry::GeometryCollection(_) => "GeometryCollection",
            GeoGeometry::Triangle(_) => "Triangle",
            GeoGeometry::Rect(_) => "Rect",
        }
    }

    /// Validate that two geometries have compatible CRS
    pub fn validate_crs_compatibility(&self, other: &Geometry) -> Result<()> {
        if self.crs != other.crs {
            return Err(GeoSparqlError::CrsMismatch {
                expected: self.crs.uri.clone(),
                found: other.crs.uri.clone(),
            });
        }
        Ok(())
    }
}

impl fmt::Display for Geometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (CRS: {})", self.to_wkt(), self.crs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Point};

    #[test]
    fn test_crs_creation() {
        let crs = Crs::default_crs();
        assert!(crs.is_default());

        let epsg_crs = Crs::epsg(4326);
        assert_eq!(epsg_crs.uri, "http://www.opengis.net/def/crs/EPSG/0/4326");
    }

    #[test]
    fn test_geometry_creation() {
        let point = Point::new(1.0, 2.0);
        let geom = Geometry::new(GeoGeometry::Point(point));

        assert_eq!(geom.geometry_type(), "Point");
        assert_eq!(geom.dimension(), 2);
        assert_eq!(geom.spatial_dimension(), 0);
        assert!(!geom.is_empty());
        assert!(!geom.is_3d());
        assert!(!geom.is_measured());
    }

    #[test]
    fn test_geometry_types() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert_eq!(point.geometry_type(), "Point");
        assert_eq!(point.spatial_dimension(), 0);

        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
        ])));
        assert_eq!(line.geometry_type(), "LineString");
        assert_eq!(line.spatial_dimension(), 1);
    }

    #[test]
    fn test_crs_compatibility() {
        let geom1 = Geometry::with_crs(GeoGeometry::Point(Point::new(1.0, 2.0)), Crs::epsg(4326));
        let geom2 = Geometry::with_crs(GeoGeometry::Point(Point::new(3.0, 4.0)), Crs::epsg(4326));
        let geom3 = Geometry::with_crs(GeoGeometry::Point(Point::new(5.0, 6.0)), Crs::epsg(3857));

        assert!(geom1.validate_crs_compatibility(&geom2).is_ok());
        assert!(geom1.validate_crs_compatibility(&geom3).is_err());
    }
}
