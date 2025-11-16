//! Geometry types and operations
//!
//! This module provides the core geometry types and operations for GeoSPARQL,
//! wrapping the `geo` crate and providing WKT/GML serialization support.

pub mod coord3d;
pub mod memory_pool;
pub mod wkt_parser;
pub mod zero_copy_wkt;

#[cfg(feature = "gml-support")]
pub mod gml_parser;

#[cfg(feature = "geojson-support")]
pub mod geojson_parser;

#[cfg(feature = "kml-support")]
pub mod kml_parser;

#[cfg(feature = "gpx-support")]
pub mod gpx_parser;

#[cfg(feature = "shapefile-support")]
pub mod shapefile_parser;

pub mod ewkb_parser;
pub mod ewkt_parser;
pub mod rdf_serialization;

#[cfg(feature = "geopackage")]
pub mod geopackage;

#[cfg(feature = "flatgeobuf-support")]
pub mod flatgeobuf_parser;

#[cfg(feature = "mvt-support")]
pub mod mvt_parser;

#[cfg(feature = "topojson-support")]
pub mod topojson_parser;

use crate::error::{GeoSparqlError, Result};
use coord3d::Coord3D;
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

    /// Create a WGS84 CRS (EPSG:4326)
    ///
    /// WGS84 is the most common coordinate system used by GPS and web mapping.
    pub fn wgs84() -> Self {
        Self::epsg(4326)
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
    /// The underlying geometry (X, Y coordinates)
    pub geom: GeoGeometry<f64>,
    /// The Coordinate Reference System
    pub crs: Crs,
    /// 3D coordinate metadata (Z and M values)
    pub coord3d: Coord3D,
}

impl Geometry {
    /// Create a new geometry with the default CRS
    pub fn new(geom: GeoGeometry<f64>) -> Self {
        Self {
            geom,
            crs: Crs::default(),
            coord3d: Coord3D::default(),
        }
    }

    /// Create a new geometry with a specific CRS
    pub fn with_crs(geom: GeoGeometry<f64>, crs: Crs) -> Self {
        Self {
            geom,
            crs,
            coord3d: Coord3D::default(),
        }
    }

    /// Create a new geometry with CRS and 3D coordinates
    pub fn with_crs_and_coord3d(geom: GeoGeometry<f64>, crs: Crs, coord3d: Coord3D) -> Self {
        Self { geom, crs, coord3d }
    }

    /// Parse from WKT (Well-Known Text) format
    pub fn from_wkt(wkt: &str) -> Result<Self> {
        wkt_parser::parse_wkt(wkt)
    }

    /// Convert to WKT format
    ///
    /// Includes Z/M coordinates if present.
    pub fn to_wkt(&self) -> String {
        wkt_parser::geometry_to_wkt_with_3d(self)
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

    /// Parse from GeoJSON format
    ///
    /// Requires the `geojson-support` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "geojson-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let geojson = r#"{"type":"Point","coordinates":[1.0,2.0]}"#;
    /// let geom = Geometry::from_geojson(geojson).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "geojson-support")]
    pub fn from_geojson(geojson: &str) -> Result<Self> {
        geojson_parser::parse_geojson(geojson)
    }

    /// Convert to GeoJSON format
    ///
    /// Requires the `geojson-support` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "geojson-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    /// let geojson = geom.to_geojson().unwrap();
    /// assert!(geojson.contains("\"type\":\"Point\""));
    /// # }
    /// ```
    #[cfg(feature = "geojson-support")]
    pub fn to_geojson(&self) -> Result<String> {
        geojson_parser::geometry_to_geojson(self)
    }

    /// Convert to GeoJSON Feature format
    ///
    /// Requires the `geojson-support` feature to be enabled.
    ///
    /// # Arguments
    ///
    /// * `properties` - Optional properties to include in the feature
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "geojson-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    /// use serde_json::json;
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    /// let props = json!({"name": "Test"});
    /// let feature = geom.to_geojson_feature(Some(&props)).unwrap();
    /// assert!(feature.contains("\"type\":\"Feature\""));
    /// # }
    /// ```
    #[cfg(feature = "geojson-support")]
    pub fn to_geojson_feature(&self, properties: Option<&serde_json::Value>) -> Result<String> {
        geojson_parser::geometry_to_geojson_feature(self, properties)
    }

    /// Parse from KML (Keyhole Markup Language) format
    ///
    /// Requires the `kml-support` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "kml-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let kml = r#"<Point><coordinates>-122.08,37.42,0</coordinates></Point>"#;
    /// let geom = Geometry::from_kml(kml).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "kml-support")]
    pub fn from_kml(kml: &str) -> Result<Self> {
        kml_parser::parse_kml(kml)
    }

    /// Convert to KML format
    ///
    /// Requires the `kml-support` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "gpx-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(-122.08, 37.42)));
    /// let kml = geom.to_kml().unwrap();
    /// assert!(kml.contains("<Point>"));
    /// # }
    /// ```
    #[cfg(feature = "kml-support")]
    pub fn to_kml(&self) -> Result<String> {
        kml_parser::geometry_to_kml(self)
    }

    /// Parse from GPX (GPS Exchange Format) format
    ///
    /// Requires the `gpx-support` feature to be enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "gpx-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let gpx = r#"<wpt lat="37.422" lon="-122.084"/>"#;
    /// let geom = Geometry::from_gpx(gpx).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "gpx-support")]
    pub fn from_gpx(gpx: &str) -> Result<Self> {
        gpx_parser::parse_gpx(gpx)
    }

    /// Convert to GPX format
    ///
    /// Requires the `gpx-support` feature to be enabled.
    ///
    /// # Arguments
    ///
    /// * `name` - Optional name for the waypoint/track
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "gpx-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(-122.084, 37.422)));
    /// let gpx = geom.to_gpx(Some("My Location")).unwrap();
    /// assert!(gpx.contains("<wpt"));
    /// # }
    /// ```
    #[cfg(feature = "gpx-support")]
    pub fn to_gpx(&self, name: Option<&str>) -> Result<String> {
        gpx_parser::geometry_to_gpx(self, name)
    }

    /// Read geometries from a shapefile
    ///
    /// Requires the `shapefile-support` feature to be enabled.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .shp file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "shapefile-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let geometries = Geometry::from_shapefile("data/cities.shp").unwrap();
    /// println!("Read {} geometries", geometries.len());
    /// # }
    /// ```
    #[cfg(feature = "shapefile-support")]
    pub fn from_shapefile<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Self>> {
        shapefile_parser::read_shapefile(path)
    }

    /// Parse from EWKB (Extended Well-Known Binary) format
    ///
    /// EWKB is PostGIS's extended binary format that includes SRID information.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// // EWKB for SRID=4326;POINT(1 2)
    /// let ewkb = vec![0x01, 0x01, 0x00, 0x00, 0x20, 0xe6, 0x10, 0x00, 0x00,
    ///                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
    ///                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40];
    /// let geom = Geometry::from_ewkb(&ewkb).unwrap();
    /// assert_eq!(geom.crs.epsg_code(), Some(4326));
    /// ```
    pub fn from_ewkb(ewkb: &[u8]) -> Result<Self> {
        ewkb_parser::parse_ewkb(ewkb)
    }

    /// Convert to EWKB (Extended Well-Known Binary) format
    ///
    /// EWKB is PostGIS's extended binary format that includes SRID information.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::geometry::{Geometry, Crs};
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geom = Geometry::with_crs(
    ///     GeoGeometry::Point(Point::new(1.0, 2.0)),
    ///     Crs::epsg(4326)
    /// );
    /// let ewkb = geom.to_ewkb().unwrap();
    /// assert!(!ewkb.is_empty());
    /// ```
    pub fn to_ewkb(&self) -> Result<Vec<u8>> {
        ewkb_parser::geometry_to_ewkb(self)
    }

    /// Parse from EWKT (Extended Well-Known Text) format
    ///
    /// EWKT is PostGIS's extended text format that includes SRID information.
    /// Format: `SRID=4326;POINT(1 2)`
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let geom = Geometry::from_ewkt("SRID=4326;POINT(1 2)").unwrap();
    /// assert_eq!(geom.crs.epsg_code(), Some(4326));
    /// ```
    pub fn from_ewkt(ewkt: &str) -> Result<Self> {
        ewkt_parser::parse_ewkt(ewkt)
    }

    /// Convert to EWKT (Extended Well-Known Text) format
    ///
    /// EWKT is PostGIS's extended text format that includes SRID information.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirs_geosparql::geometry::{Geometry, Crs};
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geom = Geometry::with_crs(
    ///     GeoGeometry::Point(Point::new(1.0, 2.0)),
    ///     Crs::epsg(4326)
    /// );
    /// let ewkt = geom.to_ewkt();
    /// assert_eq!(ewkt, "SRID=4326;POINT(1 2)");
    /// ```
    pub fn to_ewkt(&self) -> String {
        ewkt_parser::geometry_to_ewkt(self)
    }

    /// Parse geometries from FlatGeobuf format
    ///
    /// Requires the `flatgeobuf-support` feature to be enabled.
    ///
    /// FlatGeobuf is a modern cloud-native binary format optimized for
    /// streaming access and HTTP range requests.
    ///
    /// # Arguments
    ///
    /// * `reader` - Any type implementing Read (File, BufReader, etc.)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "flatgeobuf-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    /// use std::fs::File;
    /// use std::io::BufReader;
    ///
    /// let file = File::open("data.fgb").unwrap();
    /// let reader = BufReader::new(file);
    /// let geometries = Geometry::from_flatgeobuf(reader).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "flatgeobuf-support")]
    pub fn from_flatgeobuf<R: std::io::Read + std::io::Seek>(reader: R) -> Result<Vec<Self>> {
        flatgeobuf_parser::parse_flatgeobuf(reader)
    }

    /// Write geometries to FlatGeobuf format
    ///
    /// Requires the `flatgeobuf-support` feature to be enabled.
    ///
    /// # Arguments
    ///
    /// * `geometries` - Slice of geometries to write
    /// * `writer` - Any type implementing Write
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "flatgeobuf-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    /// use std::fs::File;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    /// let geometries = vec![geom];
    ///
    /// let file = File::create("output.fgb").unwrap();
    /// Geometry::to_flatgeobuf(&geometries, file).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "flatgeobuf-support")]
    pub fn to_flatgeobuf<W: std::io::Write>(geometries: &[Self], writer: W) -> Result<()> {
        flatgeobuf_parser::write_flatgeobuf(geometries, writer)
    }

    /// Parse geometries from TopoJSON format
    ///
    /// Requires the `topojson-support` feature to be enabled.
    ///
    /// TopoJSON is a topology-preserving JSON format that encodes topology
    /// by sharing arcs between geometries.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(feature = "topojson-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    ///
    /// let topojson = r#"{
    ///   "type": "Topology",
    ///   "objects": {
    ///     "example": {"type": "Point", "coordinates": [100, 200]}
    ///   }
    /// }"#;
    ///
    /// let geometries = Geometry::from_topojson(topojson).unwrap();
    /// # }
    /// ```
    #[cfg(feature = "topojson-support")]
    pub fn from_topojson(topojson: &str) -> Result<Vec<Self>> {
        topojson_parser::parse_topojson(topojson)
    }

    /// Convert geometries to TopoJSON format
    ///
    /// Requires the `topojson-support` feature to be enabled.
    ///
    /// # Arguments
    ///
    /// * `geometries` - Slice of geometries to convert
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "topojson-support")]
    /// # {
    /// use oxirs_geosparql::geometry::Geometry;
    /// use geo_types::{Point, Geometry as GeoGeometry};
    ///
    /// let geom = Geometry::new(GeoGeometry::Point(Point::new(100.0, 200.0)));
    /// let geometries = vec![geom];
    ///
    /// let topojson = Geometry::to_topojson(&geometries).unwrap();
    /// assert!(topojson.contains("\"type\":\"Topology\""));
    /// # }
    /// ```
    #[cfg(feature = "topojson-support")]
    pub fn to_topojson(geometries: &[Self]) -> Result<String> {
        topojson_parser::geometries_to_topojson(geometries)
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
        self.coord3d.has_z()
    }

    /// Check if coordinates are measured (has M coordinate)
    pub fn is_measured(&self) -> bool {
        self.coord3d.has_m()
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
