//! Geometry serialization to WKT, GeoJSON, and simplified GML 3.
//!
//! Supports Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon,
//! and GeometryCollection geometry types.

use std::fmt;

/// A two-dimensional coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoPoint {
    /// Horizontal axis (longitude or easting).
    pub x: f64,
    /// Vertical axis (latitude or northing).
    pub y: f64,
}

impl GeoPoint {
    /// Create a new point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Format as `"x y"` (WKT coordinate pair).
    fn wkt_pair(&self) -> String {
        format!("{} {}", self.x, self.y)
    }
}

/// A geometry value.
#[derive(Debug, Clone)]
pub enum GeoGeometry {
    /// A single point.
    Point(GeoPoint),
    /// A sequence of points forming a line.
    LineString(Vec<GeoPoint>),
    /// A polygon with an exterior ring and zero or more hole rings.
    Polygon {
        /// The exterior (outer) ring of the polygon.
        exterior: Vec<GeoPoint>,
        /// Zero or more interior (hole) rings.
        holes: Vec<Vec<GeoPoint>>,
    },
    /// A collection of points.
    MultiPoint(Vec<GeoPoint>),
    /// A collection of line strings.
    MultiLineString(Vec<Vec<GeoPoint>>),
    /// A collection of polygons (each: (exterior, holes)).
    MultiPolygon(Vec<(Vec<GeoPoint>, Vec<Vec<GeoPoint>>)>),
    /// A heterogeneous collection of geometries.
    GeometryCollection(Vec<GeoGeometry>),
}

/// Target serialization format.
#[derive(Debug, Clone, PartialEq)]
pub enum SerializationFormat {
    /// Well-Known Text (OGC WKT) format.
    Wkt,
    /// GeoJSON (RFC 7946) format.
    GeoJson,
    /// Geography Markup Language (GML) format.
    Gml,
}

/// Error type for geometry operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GeoError {
    /// The WKT/GML string could not be parsed.
    ParseError(String),
    /// The geometry type is not supported by the serializer.
    UnsupportedType(String),
}

impl fmt::Display for GeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeoError::ParseError(msg) => write!(f, "Geometry parse error: {msg}"),
            GeoError::UnsupportedType(t) => write!(f, "Unsupported geometry type: {t}"),
        }
    }
}

impl std::error::Error for GeoError {}

/// Serializes and deserializes geometries.
pub struct GeoSerializer;

impl GeoSerializer {
    // -------------------------------------------------------------------------
    // WKT serialization
    // -------------------------------------------------------------------------

    /// Serialize a geometry to WKT.
    pub fn to_wkt(geom: &GeoGeometry) -> String {
        match geom {
            GeoGeometry::Point(p) => format!("POINT ({} {})", p.x, p.y),

            GeoGeometry::LineString(pts) => {
                format!("LINESTRING ({})", Self::wkt_coords(pts))
            }

            GeoGeometry::Polygon { exterior, holes } => {
                let mut rings = vec![format!("({})", Self::wkt_coords(exterior))];
                for hole in holes {
                    rings.push(format!("({})", Self::wkt_coords(hole)));
                }
                format!("POLYGON ({})", rings.join(", "))
            }

            GeoGeometry::MultiPoint(pts) => {
                let inner: Vec<String> = pts.iter().map(|p| format!("({} {})", p.x, p.y)).collect();
                format!("MULTIPOINT ({})", inner.join(", "))
            }

            GeoGeometry::MultiLineString(lines) => {
                let inner: Vec<String> = lines
                    .iter()
                    .map(|pts| format!("({})", Self::wkt_coords(pts)))
                    .collect();
                format!("MULTILINESTRING ({})", inner.join(", "))
            }

            GeoGeometry::MultiPolygon(polys) => {
                let inner: Vec<String> = polys
                    .iter()
                    .map(|(exterior, holes)| {
                        let mut rings = vec![format!("({})", Self::wkt_coords(exterior))];
                        for hole in holes {
                            rings.push(format!("({})", Self::wkt_coords(hole)));
                        }
                        format!("({})", rings.join(", "))
                    })
                    .collect();
                format!("MULTIPOLYGON ({})", inner.join(", "))
            }

            GeoGeometry::GeometryCollection(geoms) => {
                let inner: Vec<String> = geoms.iter().map(Self::to_wkt).collect();
                format!("GEOMETRYCOLLECTION ({})", inner.join(", "))
            }
        }
    }

    fn wkt_coords(pts: &[GeoPoint]) -> String {
        pts.iter()
            .map(|p| p.wkt_pair())
            .collect::<Vec<_>>()
            .join(", ")
    }

    // -------------------------------------------------------------------------
    // WKT parsing
    // -------------------------------------------------------------------------

    /// Parse WKT into a `GeoGeometry`.
    ///
    /// Supports POINT, LINESTRING, POLYGON (no holes), and GEOMETRYCOLLECTION.
    pub fn from_wkt(s: &str) -> Result<GeoGeometry, GeoError> {
        let s = s.trim();
        let upper = s.to_ascii_uppercase();

        if upper.starts_with("GEOMETRYCOLLECTION") {
            Self::parse_geometry_collection(s)
        } else if upper.starts_with("MULTIPOLYGON") {
            Self::parse_multipolygon(s)
        } else if upper.starts_with("MULTILINESTRING") {
            Self::parse_multilinestring(s)
        } else if upper.starts_with("MULTIPOINT") {
            Self::parse_multipoint(s)
        } else if upper.starts_with("POLYGON") {
            Self::parse_polygon(s)
        } else if upper.starts_with("LINESTRING") {
            Self::parse_linestring(s)
        } else if upper.starts_with("POINT") {
            Self::parse_point(s)
        } else {
            Err(GeoError::ParseError(format!("Unknown WKT type: {s}")))
        }
    }

    /// Extract the content inside the outermost parentheses.
    fn inner(s: &str) -> Result<&str, GeoError> {
        let start = s
            .find('(')
            .ok_or_else(|| GeoError::ParseError(format!("Missing '(' in WKT: {s}")))?;
        let end = s
            .rfind(')')
            .ok_or_else(|| GeoError::ParseError(format!("Missing ')' in WKT: {s}")))?;
        if end <= start {
            return Err(GeoError::ParseError(format!("Empty parentheses in: {s}")));
        }
        Ok(&s[start + 1..end])
    }

    /// Parse "x y" into a GeoPoint.
    fn parse_pair(s: &str) -> Result<GeoPoint, GeoError> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(GeoError::ParseError(format!(
                "Invalid coordinate pair: '{s}'"
            )));
        }
        let x = parts[0]
            .parse::<f64>()
            .map_err(|_| GeoError::ParseError(format!("Invalid x: '{}'", parts[0])))?;
        let y = parts[1]
            .parse::<f64>()
            .map_err(|_| GeoError::ParseError(format!("Invalid y: '{}'", parts[1])))?;
        Ok(GeoPoint::new(x, y))
    }

    /// Parse a comma-separated list of "x y" pairs.
    fn parse_coord_list(s: &str) -> Result<Vec<GeoPoint>, GeoError> {
        s.split(',')
            .map(|part| Self::parse_pair(part.trim()))
            .collect()
    }

    fn parse_point(s: &str) -> Result<GeoGeometry, GeoError> {
        let inner = Self::inner(s)?;
        let p = Self::parse_pair(inner.trim())?;
        Ok(GeoGeometry::Point(p))
    }

    fn parse_linestring(s: &str) -> Result<GeoGeometry, GeoError> {
        let inner = Self::inner(s)?;
        let pts = Self::parse_coord_list(inner.trim())?;
        Ok(GeoGeometry::LineString(pts))
    }

    fn parse_polygon(s: &str) -> Result<GeoGeometry, GeoError> {
        // Split rings: each is wrapped in (...)
        let inner = Self::inner(s)?;
        let rings = Self::split_rings(inner)?;
        if rings.is_empty() {
            return Err(GeoError::ParseError("Polygon has no rings".to_string()));
        }
        let exterior = Self::parse_coord_list(rings[0].trim())?;
        let holes: Vec<Vec<GeoPoint>> = rings[1..]
            .iter()
            .map(|r| Self::parse_coord_list(r.trim()))
            .collect::<Result<_, _>>()?;
        Ok(GeoGeometry::Polygon { exterior, holes })
    }

    fn parse_multipoint(s: &str) -> Result<GeoGeometry, GeoError> {
        let inner = Self::inner(s)?;
        // Handle both MULTIPOINT (x y, x y) and MULTIPOINT ((x y), (x y))
        let inner = inner.trim();
        if inner.contains('(') {
            // Wrapped form
            let rings = Self::split_rings(inner)?;
            let pts: Vec<GeoPoint> = rings
                .iter()
                .map(|r| Self::parse_pair(r.trim()))
                .collect::<Result<_, _>>()?;
            Ok(GeoGeometry::MultiPoint(pts))
        } else {
            let pts = Self::parse_coord_list(inner)?;
            Ok(GeoGeometry::MultiPoint(pts))
        }
    }

    fn parse_multilinestring(s: &str) -> Result<GeoGeometry, GeoError> {
        let inner = Self::inner(s)?;
        let rings = Self::split_rings(inner.trim())?;
        let lines: Vec<Vec<GeoPoint>> = rings
            .iter()
            .map(|r| Self::parse_coord_list(r.trim()))
            .collect::<Result<_, _>>()?;
        Ok(GeoGeometry::MultiLineString(lines))
    }

    fn parse_multipolygon(s: &str) -> Result<GeoGeometry, GeoError> {
        // Format: MULTIPOLYGON (((x y, x y, ...)), ((x y, ...)))
        let outer_inner = Self::inner(s)?;
        let poly_strs = Self::split_nested_rings(outer_inner.trim())?;
        let polys: Vec<(Vec<GeoPoint>, Vec<Vec<GeoPoint>>)> = poly_strs
            .iter()
            .map(|ps| {
                let rings = Self::split_rings(ps.trim())?;
                if rings.is_empty() {
                    return Err(GeoError::ParseError(
                        "Empty polygon in MultiPolygon".to_string(),
                    ));
                }
                let exterior = Self::parse_coord_list(rings[0].trim())?;
                let holes: Vec<Vec<GeoPoint>> = rings[1..]
                    .iter()
                    .map(|r| Self::parse_coord_list(r.trim()))
                    .collect::<Result<_, _>>()?;
                Ok((exterior, holes))
            })
            .collect::<Result<_, _>>()?;
        Ok(GeoGeometry::MultiPolygon(polys))
    }

    fn parse_geometry_collection(s: &str) -> Result<GeoGeometry, GeoError> {
        let inner = Self::inner(s)?;
        let inner = inner.trim();
        if inner.is_empty() {
            return Ok(GeoGeometry::GeometryCollection(vec![]));
        }
        let parts = Self::split_top_level_geometries(inner)?;
        let geoms: Vec<GeoGeometry> = parts
            .iter()
            .map(|p| Self::from_wkt(p.trim()))
            .collect::<Result<_, _>>()?;
        Ok(GeoGeometry::GeometryCollection(geoms))
    }

    /// Split a string like "(x y, ...), (x y, ...)" into ["x y, ...", "x y, ..."].
    fn split_rings(s: &str) -> Result<Vec<String>, GeoError> {
        let mut rings = Vec::new();
        let mut depth = 0i32;
        let mut start = None;

        for (i, c) in s.char_indices() {
            match c {
                '(' => {
                    depth += 1;
                    if depth == 1 {
                        start = Some(i + 1);
                    }
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s_idx) = start {
                            rings.push(s[s_idx..i].to_string());
                        }
                        start = None;
                    }
                }
                _ => {}
            }
        }
        Ok(rings)
    }

    /// Split nested polygon strings from a MultiPolygon inner string.
    /// e.g., "((x y, ...)), ((x y, ...))" → ["(x y, ...)", "(x y, ...)"]
    fn split_nested_rings(s: &str) -> Result<Vec<String>, GeoError> {
        let mut result = Vec::new();
        let mut depth = 0i32;
        let mut start = None;

        for (i, c) in s.char_indices() {
            match c {
                '(' => {
                    depth += 1;
                    if depth == 1 {
                        start = Some(i + 1);
                    }
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s_idx) = start {
                            result.push(s[s_idx..i].to_string());
                        }
                        start = None;
                    }
                }
                _ => {}
            }
        }
        Ok(result)
    }

    /// Split top-level geometry strings for GEOMETRYCOLLECTION.
    fn split_top_level_geometries(s: &str) -> Result<Vec<String>, GeoError> {
        let mut result = Vec::new();
        let mut depth = 0i32;
        let mut seg_start = 0;

        for (i, c) in s.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    let part = s[seg_start..i].trim().to_string();
                    if !part.is_empty() {
                        result.push(part);
                    }
                    seg_start = i + 1;
                }
                _ => {}
            }
        }
        let last = s[seg_start..].trim().to_string();
        if !last.is_empty() {
            result.push(last);
        }
        Ok(result)
    }

    // -------------------------------------------------------------------------
    // GeoJSON serialization (hand-rolled)
    // -------------------------------------------------------------------------

    /// Serialize a geometry to a GeoJSON geometry object string.
    pub fn to_geojson(geom: &GeoGeometry) -> String {
        match geom {
            GeoGeometry::Point(p) => {
                format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, p.x, p.y)
            }

            GeoGeometry::LineString(pts) => {
                format!(
                    r#"{{"type":"LineString","coordinates":[{}]}}"#,
                    Self::geojson_coord_array(pts)
                )
            }

            GeoGeometry::Polygon { exterior, holes } => {
                let mut rings = vec![format!("[{}]", Self::geojson_coord_array(exterior))];
                for hole in holes {
                    rings.push(format!("[{}]", Self::geojson_coord_array(hole)));
                }
                format!(
                    r#"{{"type":"Polygon","coordinates":[{}]}}"#,
                    rings.join(",")
                )
            }

            GeoGeometry::MultiPoint(pts) => {
                format!(
                    r#"{{"type":"MultiPoint","coordinates":[{}]}}"#,
                    Self::geojson_coord_array(pts)
                )
            }

            GeoGeometry::MultiLineString(lines) => {
                let inner: Vec<String> = lines
                    .iter()
                    .map(|pts| format!("[{}]", Self::geojson_coord_array(pts)))
                    .collect();
                format!(
                    r#"{{"type":"MultiLineString","coordinates":[{}]}}"#,
                    inner.join(",")
                )
            }

            GeoGeometry::MultiPolygon(polys) => {
                let inner: Vec<String> = polys
                    .iter()
                    .map(|(exterior, holes)| {
                        let mut rings = vec![format!("[{}]", Self::geojson_coord_array(exterior))];
                        for hole in holes {
                            rings.push(format!("[{}]", Self::geojson_coord_array(hole)));
                        }
                        format!("[{}]", rings.join(","))
                    })
                    .collect();
                format!(
                    r#"{{"type":"MultiPolygon","coordinates":[{}]}}"#,
                    inner.join(",")
                )
            }

            GeoGeometry::GeometryCollection(geoms) => {
                let inner: Vec<String> = geoms.iter().map(Self::to_geojson).collect();
                format!(
                    r#"{{"type":"GeometryCollection","geometries":[{}]}}"#,
                    inner.join(",")
                )
            }
        }
    }

    fn geojson_coord_array(pts: &[GeoPoint]) -> String {
        pts.iter()
            .map(|p| format!("[{},{}]", p.x, p.y))
            .collect::<Vec<_>>()
            .join(",")
    }

    // -------------------------------------------------------------------------
    // GML 3 serialization (simplified)
    // -------------------------------------------------------------------------

    /// Serialize a geometry to simplified GML 3.
    pub fn to_gml(geom: &GeoGeometry) -> String {
        match geom {
            GeoGeometry::Point(p) => {
                format!("<gml:Point><gml:pos>{} {}</gml:pos></gml:Point>", p.x, p.y)
            }

            GeoGeometry::LineString(pts) => {
                format!(
                    "<gml:LineString><gml:posList>{}</gml:posList></gml:LineString>",
                    Self::gml_pos_list(pts)
                )
            }

            GeoGeometry::Polygon { exterior, holes } => {
                let mut inner = format!(
                    "<gml:exterior><gml:LinearRing><gml:posList>{}</gml:posList></gml:LinearRing></gml:exterior>",
                    Self::gml_pos_list(exterior)
                );
                for hole in holes {
                    inner.push_str(&format!(
                        "<gml:interior><gml:LinearRing><gml:posList>{}</gml:posList></gml:LinearRing></gml:interior>",
                        Self::gml_pos_list(hole)
                    ));
                }
                format!("<gml:Polygon>{inner}</gml:Polygon>")
            }

            GeoGeometry::MultiPoint(pts) => {
                let members: String = pts
                    .iter()
                    .map(|p| {
                        format!(
                            "<gml:pointMember><gml:Point><gml:pos>{} {}</gml:pos></gml:Point></gml:pointMember>",
                            p.x, p.y
                        )
                    })
                    .collect();
                format!("<gml:MultiPoint>{members}</gml:MultiPoint>")
            }

            GeoGeometry::MultiLineString(lines) => {
                let members: String = lines
                    .iter()
                    .map(|pts| {
                        format!(
                            "<gml:curveMember><gml:LineString><gml:posList>{}</gml:posList></gml:LineString></gml:curveMember>",
                            Self::gml_pos_list(pts)
                        )
                    })
                    .collect();
                format!("<gml:MultiCurve>{members}</gml:MultiCurve>")
            }

            GeoGeometry::MultiPolygon(polys) => {
                let members: String = polys
                    .iter()
                    .map(|(exterior, holes)| {
                        let sub = GeoGeometry::Polygon {
                            exterior: exterior.clone(),
                            holes: holes.clone(),
                        };
                        format!(
                            "<gml:surfaceMember>{}</gml:surfaceMember>",
                            Self::to_gml(&sub)
                        )
                    })
                    .collect();
                format!("<gml:MultiSurface>{members}</gml:MultiSurface>")
            }

            GeoGeometry::GeometryCollection(geoms) => {
                let members: String = geoms
                    .iter()
                    .map(|g| {
                        format!(
                            "<gml:geometryMember>{}</gml:geometryMember>",
                            Self::to_gml(g)
                        )
                    })
                    .collect();
                format!("<gml:MultiGeometry>{members}</gml:MultiGeometry>")
            }
        }
    }

    fn gml_pos_list(pts: &[GeoPoint]) -> String {
        pts.iter()
            .map(|p| format!("{} {}", p.x, p.y))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> GeoPoint {
        GeoPoint::new(x, y)
    }

    // ------ WKT: Point ------

    #[test]
    fn test_wkt_point() {
        let g = GeoGeometry::Point(pt(1.0, 2.0));
        assert_eq!(GeoSerializer::to_wkt(&g), "POINT (1 2)");
    }

    #[test]
    fn test_wkt_point_roundtrip() {
        let g = GeoGeometry::Point(pt(10.5, -3.75));
        let wkt = GeoSerializer::to_wkt(&g);
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::Point(p) = parsed {
            assert!((p.x - 10.5).abs() < 1e-9);
            assert!((p.y + 3.75).abs() < 1e-9);
        } else {
            panic!("Expected Point");
        }
    }

    // ------ WKT: LineString ------

    #[test]
    fn test_wkt_linestring() {
        let g = GeoGeometry::LineString(vec![pt(0.0, 0.0), pt(1.0, 1.0)]);
        assert_eq!(GeoSerializer::to_wkt(&g), "LINESTRING (0 0, 1 1)");
    }

    #[test]
    fn test_wkt_linestring_roundtrip() {
        let g = GeoGeometry::LineString(vec![pt(0.0, 0.0), pt(1.0, 2.0), pt(3.0, 4.0)]);
        let wkt = GeoSerializer::to_wkt(&g);
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::LineString(pts) = parsed {
            assert_eq!(pts.len(), 3);
            assert!((pts[2].x - 3.0).abs() < 1e-9);
        } else {
            panic!("Expected LineString");
        }
    }

    // ------ WKT: Polygon ------

    #[test]
    fn test_wkt_polygon_no_holes() {
        let g = GeoGeometry::Polygon {
            exterior: vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 0.0)],
            holes: vec![],
        };
        let wkt = GeoSerializer::to_wkt(&g);
        assert!(wkt.starts_with("POLYGON"));
        assert!(wkt.contains("0 0"));
    }

    #[test]
    fn test_wkt_polygon_roundtrip() {
        let g = GeoGeometry::Polygon {
            exterior: vec![pt(0.0, 0.0), pt(4.0, 0.0), pt(4.0, 4.0), pt(0.0, 0.0)],
            holes: vec![],
        };
        let wkt = GeoSerializer::to_wkt(&g);
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::Polygon { exterior, holes } = parsed {
            assert_eq!(exterior.len(), 4);
            assert!(holes.is_empty());
        } else {
            panic!("Expected Polygon");
        }
    }

    #[test]
    fn test_wkt_polygon_with_holes() {
        let hole = vec![pt(1.0, 1.0), pt(2.0, 1.0), pt(2.0, 2.0), pt(1.0, 1.0)];
        let g = GeoGeometry::Polygon {
            exterior: vec![pt(0.0, 0.0), pt(4.0, 0.0), pt(4.0, 4.0), pt(0.0, 0.0)],
            holes: vec![hole],
        };
        let wkt = GeoSerializer::to_wkt(&g);
        assert!(wkt.contains("POLYGON"));
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::Polygon { holes, .. } = parsed {
            assert_eq!(holes.len(), 1);
        } else {
            panic!("Expected Polygon");
        }
    }

    // ------ WKT: MultiPoint ------

    #[test]
    fn test_wkt_multipoint() {
        let g = GeoGeometry::MultiPoint(vec![pt(0.0, 0.0), pt(1.0, 1.0)]);
        let wkt = GeoSerializer::to_wkt(&g);
        assert!(wkt.starts_with("MULTIPOINT"));
    }

    #[test]
    fn test_wkt_multipoint_roundtrip() {
        let g = GeoGeometry::MultiPoint(vec![pt(1.0, 2.0), pt(3.0, 4.0)]);
        let wkt = GeoSerializer::to_wkt(&g);
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::MultiPoint(pts) = parsed {
            assert_eq!(pts.len(), 2);
        } else {
            panic!("Expected MultiPoint");
        }
    }

    // ------ WKT: MultiLineString ------

    #[test]
    fn test_wkt_multilinestring() {
        let g = GeoGeometry::MultiLineString(vec![
            vec![pt(0.0, 0.0), pt(1.0, 1.0)],
            vec![pt(2.0, 2.0), pt(3.0, 3.0)],
        ]);
        let wkt = GeoSerializer::to_wkt(&g);
        assert!(wkt.starts_with("MULTILINESTRING"));
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::MultiLineString(lines) = parsed {
            assert_eq!(lines.len(), 2);
        } else {
            panic!("Expected MultiLineString");
        }
    }

    // ------ WKT: MultiPolygon ------

    #[test]
    fn test_wkt_multipolygon() {
        let g = GeoGeometry::MultiPolygon(vec![
            (
                vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 0.0)],
                vec![],
            ),
            (
                vec![pt(2.0, 2.0), pt(3.0, 2.0), pt(3.0, 3.0), pt(2.0, 2.0)],
                vec![],
            ),
        ]);
        let wkt = GeoSerializer::to_wkt(&g);
        assert!(wkt.starts_with("MULTIPOLYGON"));
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::MultiPolygon(polys) = parsed {
            assert_eq!(polys.len(), 2);
        } else {
            panic!("Expected MultiPolygon");
        }
    }

    // ------ WKT: GeometryCollection ------

    #[test]
    fn test_wkt_geometry_collection_empty() {
        let g = GeoGeometry::GeometryCollection(vec![]);
        let wkt = GeoSerializer::to_wkt(&g);
        assert!(wkt.starts_with("GEOMETRYCOLLECTION"));
    }

    #[test]
    fn test_wkt_geometry_collection_roundtrip() {
        let g = GeoGeometry::GeometryCollection(vec![
            GeoGeometry::Point(pt(1.0, 2.0)),
            GeoGeometry::LineString(vec![pt(0.0, 0.0), pt(1.0, 1.0)]),
        ]);
        let wkt = GeoSerializer::to_wkt(&g);
        let parsed = GeoSerializer::from_wkt(&wkt).expect("parse");
        if let GeoGeometry::GeometryCollection(geoms) = parsed {
            assert_eq!(geoms.len(), 2);
        } else {
            panic!("Expected GeometryCollection");
        }
    }

    // ------ GeoJSON ------

    #[test]
    fn test_geojson_point_type() {
        let g = GeoGeometry::Point(pt(1.0, 2.0));
        let json = GeoSerializer::to_geojson(&g);
        assert!(json.contains(r#""type":"Point""#));
        assert!(json.contains("[1,2]"));
    }

    #[test]
    fn test_geojson_linestring_type() {
        let g = GeoGeometry::LineString(vec![pt(0.0, 0.0), pt(1.0, 1.0)]);
        let json = GeoSerializer::to_geojson(&g);
        assert!(json.contains(r#""type":"LineString""#));
        assert!(json.contains("coordinates"));
    }

    #[test]
    fn test_geojson_polygon_type() {
        let g = GeoGeometry::Polygon {
            exterior: vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 0.0)],
            holes: vec![],
        };
        let json = GeoSerializer::to_geojson(&g);
        assert!(json.contains(r#""type":"Polygon""#));
    }

    #[test]
    fn test_geojson_multipoint_type() {
        let g = GeoGeometry::MultiPoint(vec![pt(0.0, 0.0), pt(1.0, 1.0)]);
        let json = GeoSerializer::to_geojson(&g);
        assert!(json.contains(r#""type":"MultiPoint""#));
    }

    #[test]
    fn test_geojson_collection_type() {
        let g = GeoGeometry::GeometryCollection(vec![GeoGeometry::Point(pt(0.0, 0.0))]);
        let json = GeoSerializer::to_geojson(&g);
        assert!(json.contains(r#""type":"GeometryCollection""#));
        assert!(json.contains("geometries"));
    }

    // ------ GML ------

    #[test]
    fn test_gml_point_tags() {
        let g = GeoGeometry::Point(pt(1.0, 2.0));
        let gml = GeoSerializer::to_gml(&g);
        assert!(gml.contains("<gml:Point>"));
        assert!(gml.contains("<gml:pos>"));
        assert!(gml.contains("1 2"));
    }

    #[test]
    fn test_gml_linestring_tags() {
        let g = GeoGeometry::LineString(vec![pt(0.0, 0.0), pt(1.0, 1.0)]);
        let gml = GeoSerializer::to_gml(&g);
        assert!(gml.contains("<gml:LineString>"));
        assert!(gml.contains("<gml:posList>"));
    }

    #[test]
    fn test_gml_polygon_tags() {
        let g = GeoGeometry::Polygon {
            exterior: vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 0.0)],
            holes: vec![],
        };
        let gml = GeoSerializer::to_gml(&g);
        assert!(gml.contains("<gml:Polygon>"));
        assert!(gml.contains("<gml:exterior>"));
    }

    #[test]
    fn test_gml_multipoint_tags() {
        let g = GeoGeometry::MultiPoint(vec![pt(0.0, 0.0), pt(1.0, 1.0)]);
        let gml = GeoSerializer::to_gml(&g);
        assert!(gml.contains("<gml:MultiPoint>"));
        assert!(gml.contains("<gml:pointMember>"));
    }

    #[test]
    fn test_gml_geometry_collection_tags() {
        let g = GeoGeometry::GeometryCollection(vec![GeoGeometry::Point(pt(0.0, 0.0))]);
        let gml = GeoSerializer::to_gml(&g);
        assert!(gml.contains("<gml:MultiGeometry>"));
        assert!(gml.contains("<gml:geometryMember>"));
    }

    // ------ Error cases ------

    #[test]
    fn test_from_wkt_unknown_type_error() {
        let r = GeoSerializer::from_wkt("TRIANGLE (0 0, 1 0, 0 1)");
        assert!(r.is_err());
        if let Err(GeoError::ParseError(msg)) = r {
            assert!(msg.contains("TRIANGLE") || !msg.is_empty());
        }
    }

    #[test]
    fn test_from_wkt_invalid_coordinates() {
        let r = GeoSerializer::from_wkt("POINT (abc def)");
        assert!(r.is_err());
    }

    #[test]
    fn test_geo_error_display_parse() {
        let e = GeoError::ParseError("bad wkt".to_string());
        assert!(e.to_string().contains("bad wkt"));
    }

    #[test]
    fn test_geo_error_display_unsupported() {
        let e = GeoError::UnsupportedType("CURVE".to_string());
        assert!(e.to_string().contains("CURVE"));
    }
}
