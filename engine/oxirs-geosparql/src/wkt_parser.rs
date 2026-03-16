//! WKT (Well-Known Text) geometry parser and serializer.
//!
//! Implements the OGC WKT grammar for point, line-string, polygon,
//! multi-geometries, and geometry collections. Supports 2D and 3D coordinates.
use std::fmt;

/// Type alias for a polygon ring list: (exterior_ring, list_of_hole_rings).
pub type PolygonRings = (Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>);

/// All WKT geometry variants.
#[derive(Debug, Clone, PartialEq)]
pub enum WktGeometry {
    /// A single 2D or 3D point.
    Point {
        /// X (longitude / easting) coordinate.
        x: f64,
        /// Y (latitude / northing) coordinate.
        y: f64,
        /// Optional Z (elevation) coordinate.
        z: Option<f64>,
    },
    /// A sequence of connected 2D points.
    LineString {
        /// Ordered sequence of coordinate pairs.
        points: Vec<(f64, f64)>,
    },
    /// A polygon with an exterior ring and optional interior rings (holes).
    Polygon {
        /// Exterior boundary ring.
        exterior: Vec<(f64, f64)>,
        /// Interior rings (holes).
        holes: Vec<Vec<(f64, f64)>>,
    },
    /// A collection of independent points.
    MultiPoint {
        /// Individual points.
        points: Vec<(f64, f64)>,
    },
    /// A collection of line-strings.
    MultiLineString {
        /// Individual line-strings.
        lines: Vec<Vec<(f64, f64)>>,
    },
    /// A collection of polygons (exterior + holes per polygon).
    MultiPolygon {
        /// Individual polygons, each as `(exterior_ring, hole_rings)`.
        polygons: Vec<PolygonRings>,
    },
    /// A heterogeneous collection of geometries.
    GeometryCollection {
        /// Member geometries.
        geometries: Vec<WktGeometry>,
    },
}

/// Errors produced during WKT parsing.
#[derive(Debug, Clone)]
pub enum WktError {
    /// Generic syntax or structural parse error.
    ParseError(String),
    /// The geometry (or a component) contains no coordinates.
    EmptyGeometry,
    /// A coordinate value could not be parsed as a finite number.
    InvalidCoordinate(String),
}

impl fmt::Display for WktError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "WKT parse error: {msg}"),
            Self::EmptyGeometry => write!(f, "WKT geometry is empty"),
            Self::InvalidCoordinate(c) => write!(f, "Invalid coordinate: {c}"),
        }
    }
}

impl std::error::Error for WktError {}

/// Stateless WKT parser / serializer utilities.
pub struct WktParser;

impl WktParser {
    /// Parse a WKT string into a `WktGeometry`.
    pub fn parse(wkt: &str) -> Result<WktGeometry, WktError> {
        let input = wkt.trim();
        let upper = input.to_uppercase();

        if upper.starts_with("GEOMETRYCOLLECTION") {
            parse_geometry_collection(input)
        } else if upper.starts_with("MULTIPOLYGON") {
            parse_multi_polygon(input)
        } else if upper.starts_with("MULTILINESTRING") {
            parse_multi_line_string(input)
        } else if upper.starts_with("MULTIPOINT") {
            parse_multi_point(input)
        } else if upper.starts_with("POLYGON") {
            parse_polygon(input)
        } else if upper.starts_with("LINESTRING") {
            parse_line_string(input)
        } else if upper.starts_with("POINT") {
            parse_point(input)
        } else {
            Err(WktError::ParseError(format!(
                "Unknown geometry type in: {input}"
            )))
        }
    }

    /// Serialize a `WktGeometry` to its canonical WKT string.
    pub fn serialize(geom: &WktGeometry) -> String {
        match geom {
            WktGeometry::Point { x, y, z: None } => {
                format!("POINT ({x} {y})")
            }
            WktGeometry::Point {
                x,
                y,
                z: Some(z_val),
            } => {
                format!("POINT Z ({x} {y} {z_val})")
            }
            WktGeometry::LineString { points } => {
                format!("LINESTRING ({})", format_point_list(points))
            }
            WktGeometry::Polygon { exterior, holes } => {
                let mut rings = vec![format!("({})", format_point_list(exterior))];
                for hole in holes {
                    rings.push(format!("({})", format_point_list(hole)));
                }
                format!("POLYGON ({})", rings.join(", "))
            }
            WktGeometry::MultiPoint { points } => {
                format!("MULTIPOINT ({})", format_point_list(points))
            }
            WktGeometry::MultiLineString { lines } => {
                let parts: Vec<String> = lines
                    .iter()
                    .map(|l| format!("({})", format_point_list(l)))
                    .collect();
                format!("MULTILINESTRING ({})", parts.join(", "))
            }
            WktGeometry::MultiPolygon { polygons } => {
                let parts: Vec<String> = polygons
                    .iter()
                    .map(|(ext, holes)| {
                        let mut rings = vec![format!("({})", format_point_list(ext))];
                        for hole in holes {
                            rings.push(format!("({})", format_point_list(hole)));
                        }
                        format!("({})", rings.join(", "))
                    })
                    .collect();
                format!("MULTIPOLYGON ({})", parts.join(", "))
            }
            WktGeometry::GeometryCollection { geometries } => {
                let parts: Vec<String> = geometries.iter().map(Self::serialize).collect();
                format!("GEOMETRYCOLLECTION ({})", parts.join(", "))
            }
        }
    }

    /// Return the total number of coordinate pairs (2D) in the geometry.
    pub fn point_count(geom: &WktGeometry) -> usize {
        match geom {
            WktGeometry::Point { .. } => 1,
            WktGeometry::LineString { points } => points.len(),
            WktGeometry::Polygon { exterior, holes } => {
                exterior.len() + holes.iter().map(|h| h.len()).sum::<usize>()
            }
            WktGeometry::MultiPoint { points } => points.len(),
            WktGeometry::MultiLineString { lines } => lines.iter().map(|l| l.len()).sum(),
            WktGeometry::MultiPolygon { polygons } => polygons
                .iter()
                .map(|(ext, holes)| ext.len() + holes.iter().map(|h| h.len()).sum::<usize>())
                .sum(),
            WktGeometry::GeometryCollection { geometries } => {
                geometries.iter().map(Self::point_count).sum()
            }
        }
    }

    /// Compute the axis-aligned bounding box `(min_x, min_y, max_x, max_y)`.
    pub fn bounding_box(geom: &WktGeometry) -> Option<(f64, f64, f64, f64)> {
        let coords = collect_coords(geom);
        if coords.is_empty() {
            return None;
        }
        let min_x = coords.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
        let min_y = coords.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let max_x = coords.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max);
        let max_y = coords.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
        Some((min_x, min_y, max_x, max_y))
    }

    /// Return the static geometry type string.
    pub fn geometry_type(geom: &WktGeometry) -> &'static str {
        match geom {
            WktGeometry::Point { .. } => "POINT",
            WktGeometry::LineString { .. } => "LINESTRING",
            WktGeometry::Polygon { .. } => "POLYGON",
            WktGeometry::MultiPoint { .. } => "MULTIPOINT",
            WktGeometry::MultiLineString { .. } => "MULTILINESTRING",
            WktGeometry::MultiPolygon { .. } => "MULTIPOLYGON",
            WktGeometry::GeometryCollection { .. } => "GEOMETRYCOLLECTION",
        }
    }

    /// Return true if the geometry contains no coordinates.
    pub fn is_empty(geom: &WktGeometry) -> bool {
        Self::point_count(geom) == 0
    }

    /// Compute the centroid as the arithmetic mean of all coordinate pairs.
    pub fn centroid(geom: &WktGeometry) -> Option<(f64, f64)> {
        let coords = collect_coords(geom);
        if coords.is_empty() {
            return None;
        }
        let n = coords.len() as f64;
        let cx = coords.iter().map(|c| c.0).sum::<f64>() / n;
        let cy = coords.iter().map(|c| c.1).sum::<f64>() / n;
        Some((cx, cy))
    }
}

// ---- Serialization helpers ----

fn format_point_list(pts: &[(f64, f64)]) -> String {
    pts.iter()
        .map(|(x, y)| format!("{x} {y}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- Coordinate collection ----

fn collect_coords(geom: &WktGeometry) -> Vec<(f64, f64)> {
    match geom {
        WktGeometry::Point { x, y, .. } => vec![(*x, *y)],
        WktGeometry::LineString { points } => points.clone(),
        WktGeometry::Polygon { exterior, holes } => {
            let mut c = exterior.clone();
            for hole in holes {
                c.extend_from_slice(hole);
            }
            c
        }
        WktGeometry::MultiPoint { points } => points.clone(),
        WktGeometry::MultiLineString { lines } => lines.iter().flatten().copied().collect(),
        WktGeometry::MultiPolygon { polygons } => polygons
            .iter()
            .flat_map(|(ext, holes)| ext.iter().chain(holes.iter().flatten()).copied())
            .collect(),
        WktGeometry::GeometryCollection { geometries } => {
            geometries.iter().flat_map(collect_coords).collect()
        }
    }
}

// ---- Token/parsing helpers ----

/// Strip the geometry type keyword and return the content inside the outer parentheses.
fn strip_keyword_and_parens(input: &str) -> Result<&str, WktError> {
    let after_kw = input
        .find('(')
        .ok_or_else(|| WktError::ParseError(format!("Missing '(' in: {input}")))?;
    let inner = &input[after_kw..].trim_start();
    // Find the last matching close paren
    let content = inner
        .strip_prefix('(')
        .ok_or_else(|| WktError::ParseError("Expected '('".to_string()))?;
    // Remove trailing ')'
    let content = content
        .rfind(')')
        .map(|i| &content[..i])
        .ok_or_else(|| WktError::ParseError("Missing ')' to close geometry".to_string()))?;
    Ok(content)
}

/// Split a comma-separated list of coordinate pairs, each pair being two numbers.
fn parse_coord_list(list: &str) -> Result<Vec<(f64, f64)>, WktError> {
    if list.trim().is_empty() {
        return Ok(vec![]);
    }
    // Each comma separates coordinate PAIRS, but each pair has two numbers.
    // Strategy: collect all tokens and take pairs.
    let tokens: Vec<&str> = list.split_whitespace().collect();
    let mut result = Vec::new();
    let mut i = 0;
    while i + 1 < tokens.len() {
        // Remove trailing commas from tokens
        let raw_x = tokens[i].trim_end_matches(',');
        let raw_y = tokens[i + 1].trim_end_matches(',');
        let x = raw_x
            .parse::<f64>()
            .map_err(|_| WktError::InvalidCoordinate(raw_x.to_string()))?;
        let y = raw_y
            .parse::<f64>()
            .map_err(|_| WktError::InvalidCoordinate(raw_y.to_string()))?;
        result.push((x, y));
        i += 2;
    }
    Ok(result)
}

/// Split a ring list string by top-level commas (not inside parentheses).
fn split_rings(input: &str) -> Vec<&str> {
    let mut rings = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (idx, ch) in input.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                rings.push(input[start..idx].trim());
                start = idx + 1;
            }
            _ => {}
        }
    }
    let last = input[start..].trim();
    if !last.is_empty() {
        rings.push(last);
    }
    rings
}

/// Parse the content inside a ring: `(x1 y1, x2 y2, ...)`.
fn parse_ring(ring: &str) -> Result<Vec<(f64, f64)>, WktError> {
    let inner = ring
        .trim()
        .strip_prefix('(')
        .ok_or_else(|| WktError::ParseError(format!("Ring missing '(': {ring}")))?;
    let inner = inner
        .rfind(')')
        .map(|i| &inner[..i])
        .ok_or_else(|| WktError::ParseError("Ring missing ')'".to_string()))?;
    parse_coord_list(inner)
}

// ---- Individual geometry parsers ----

fn parse_point(input: &str) -> Result<WktGeometry, WktError> {
    let upper = input.to_uppercase();
    let is_3d = upper.contains("POINT Z");

    let after_point = if is_3d {
        input
            .to_uppercase()
            .find("POINT Z")
            .map(|i| &input[i + 7..])
            .unwrap_or(input)
    } else {
        input
            .to_uppercase()
            .find("POINT")
            .map(|i| &input[i + 5..])
            .unwrap_or(input)
    };

    let content = after_point
        .trim()
        .strip_prefix('(')
        .ok_or_else(|| WktError::ParseError("POINT missing '('".to_string()))?;
    let content = content
        .rfind(')')
        .map(|i| &content[..i])
        .ok_or_else(|| WktError::ParseError("POINT missing ')'".to_string()))?
        .trim();

    let tokens: Vec<&str> = content.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err(WktError::ParseError(format!(
            "POINT requires at least 2 coordinates, got: {content}"
        )));
    }
    let x = tokens[0]
        .parse::<f64>()
        .map_err(|_| WktError::InvalidCoordinate(tokens[0].to_string()))?;
    let y = tokens[1]
        .parse::<f64>()
        .map_err(|_| WktError::InvalidCoordinate(tokens[1].to_string()))?;
    let z = if is_3d && tokens.len() >= 3 {
        Some(
            tokens[2]
                .parse::<f64>()
                .map_err(|_| WktError::InvalidCoordinate(tokens[2].to_string()))?,
        )
    } else {
        None
    };
    Ok(WktGeometry::Point { x, y, z })
}

fn parse_line_string(input: &str) -> Result<WktGeometry, WktError> {
    let content = strip_keyword_and_parens(input)?;
    let points = parse_coord_list(content)?;
    Ok(WktGeometry::LineString { points })
}

fn parse_polygon(input: &str) -> Result<WktGeometry, WktError> {
    let content = strip_keyword_and_parens(input)?;
    let ring_strs = split_rings(content);
    if ring_strs.is_empty() {
        return Err(WktError::EmptyGeometry);
    }
    let exterior = parse_ring(ring_strs[0])?;
    let holes: Vec<Vec<(f64, f64)>> = ring_strs[1..]
        .iter()
        .map(|r| parse_ring(r))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WktGeometry::Polygon { exterior, holes })
}

fn parse_multi_point(input: &str) -> Result<WktGeometry, WktError> {
    let content = strip_keyword_and_parens(input)?;
    let points = parse_coord_list(content)?;
    Ok(WktGeometry::MultiPoint { points })
}

fn parse_multi_line_string(input: &str) -> Result<WktGeometry, WktError> {
    let content = strip_keyword_and_parens(input)?;
    let ring_strs = split_rings(content);
    let lines: Vec<Vec<(f64, f64)>> = ring_strs
        .iter()
        .map(|r| parse_ring(r))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WktGeometry::MultiLineString { lines })
}

fn parse_multi_polygon(input: &str) -> Result<WktGeometry, WktError> {
    let content = strip_keyword_and_parens(input)?;
    // Each polygon is wrapped in (...) — find top-level groups
    let poly_strs = split_rings(content);
    let polygons: Vec<PolygonRings> = poly_strs
        .iter()
        .map(|ps| {
            // ps looks like "((x y, ...), (x y, ...))"
            let inner = ps
                .trim()
                .strip_prefix('(')
                .ok_or_else(|| WktError::ParseError(format!("Expected '(' in polygon: {ps}")))?;
            let inner = inner
                .rfind(')')
                .map(|i| &inner[..i])
                .ok_or_else(|| WktError::ParseError("Missing ')' in polygon".to_string()))?;
            let ring_strs = split_rings(inner);
            if ring_strs.is_empty() {
                return Err(WktError::EmptyGeometry);
            }
            let exterior = parse_ring(ring_strs[0])?;
            let holes: Vec<Vec<(f64, f64)>> = ring_strs[1..]
                .iter()
                .map(|r| parse_ring(r))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((exterior, holes))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WktGeometry::MultiPolygon { polygons })
}

fn parse_geometry_collection(input: &str) -> Result<WktGeometry, WktError> {
    let content = strip_keyword_and_parens(input)?;
    // Split by top-level commas but keep type keywords intact
    let geom_strs = split_rings(content);
    let geometries: Vec<WktGeometry> = geom_strs
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| WktParser::parse(s))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WktGeometry::GeometryCollection { geometries })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // --- POINT ---

    #[test]
    fn test_parse_point_2d() {
        let g = WktParser::parse("POINT (1.0 2.0)").expect("should succeed");
        match g {
            WktGeometry::Point { x, y, z } => {
                assert!(approx(x, 1.0));
                assert!(approx(y, 2.0));
                assert!(z.is_none());
            }
            _ => panic!("Expected Point"),
        }
    }

    #[test]
    fn test_parse_point_3d() {
        let g = WktParser::parse("POINT Z (1.0 2.0 3.5)").expect("should succeed");
        match g {
            WktGeometry::Point { x, y, z } => {
                assert!(approx(x, 1.0));
                assert!(approx(y, 2.0));
                assert!(approx(z.expect("should succeed"), 3.5));
            }
            _ => panic!("Expected Point"),
        }
    }

    #[test]
    fn test_parse_point_negative_coords() {
        let g = WktParser::parse("POINT (-10.5 -20.3)").expect("should succeed");
        match g {
            WktGeometry::Point { x, y, .. } => {
                assert!(approx(x, -10.5));
                assert!(approx(y, -20.3));
            }
            _ => panic!("Expected Point"),
        }
    }

    // --- LINESTRING ---

    #[test]
    fn test_parse_linestring() {
        let g = WktParser::parse("LINESTRING (0 0, 1 1, 2 2)").expect("should succeed");
        match g {
            WktGeometry::LineString { points } => {
                assert_eq!(points.len(), 3);
                assert!(approx(points[0].0, 0.0));
                assert!(approx(points[2].1, 2.0));
            }
            _ => panic!("Expected LineString"),
        }
    }

    #[test]
    fn test_linestring_point_count() {
        let g = WktParser::parse("LINESTRING (0 0, 1 1, 2 2)").expect("should succeed");
        assert_eq!(WktParser::point_count(&g), 3);
    }

    // --- POLYGON ---

    #[test]
    fn test_parse_polygon_no_hole() {
        let g = WktParser::parse("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
        match &g {
            WktGeometry::Polygon { exterior, holes } => {
                assert_eq!(exterior.len(), 5);
                assert!(holes.is_empty());
            }
            _ => panic!("Expected Polygon"),
        }
    }

    #[test]
    fn test_parse_polygon_with_hole() {
        let g =
            WktParser::parse("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")
                .expect("should succeed");
        match &g {
            WktGeometry::Polygon { exterior, holes } => {
                assert_eq!(exterior.len(), 5);
                assert_eq!(holes.len(), 1);
                assert_eq!(holes[0].len(), 5);
            }
            _ => panic!("Expected Polygon"),
        }
    }

    // --- MULTIPOINT ---

    #[test]
    fn test_parse_multipoint() {
        let g = WktParser::parse("MULTIPOINT (1 2, 3 4, 5 6)").expect("should succeed");
        match g {
            WktGeometry::MultiPoint { points } => {
                assert_eq!(points.len(), 3);
            }
            _ => panic!("Expected MultiPoint"),
        }
    }

    #[test]
    fn test_multipoint_bounding_box() {
        let g = WktParser::parse("MULTIPOINT (1 2, 5 3, 3 7)").expect("should succeed");
        let bb = WktParser::bounding_box(&g).expect("should succeed");
        assert!(approx(bb.0, 1.0)); // min_x
        assert!(approx(bb.1, 2.0)); // min_y
        assert!(approx(bb.2, 5.0)); // max_x
        assert!(approx(bb.3, 7.0)); // max_y
    }

    // --- GEOMETRYCOLLECTION ---

    #[test]
    fn test_parse_geometry_collection() {
        let g = WktParser::parse("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))")
            .expect("should succeed");
        match &g {
            WktGeometry::GeometryCollection { geometries } => {
                assert_eq!(geometries.len(), 2);
            }
            _ => panic!("Expected GeometryCollection"),
        }
    }

    #[test]
    fn test_geometry_collection_point_count() {
        let g = WktParser::parse("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1, 2 2))")
            .expect("should succeed");
        // 1 from POINT + 3 from LINESTRING
        assert_eq!(WktParser::point_count(&g), 4);
    }

    // --- serialize round-trip ---

    #[test]
    fn test_serialize_point() {
        let g = WktGeometry::Point {
            x: 3.0,
            y: 4.0,
            z: None,
        };
        let s = WktParser::serialize(&g);
        assert!(s.contains("POINT"));
        assert!(s.contains("3"));
        assert!(s.contains("4"));
    }

    #[test]
    fn test_serialize_point_3d() {
        let g = WktGeometry::Point {
            x: 1.0,
            y: 2.0,
            z: Some(3.0),
        };
        let s = WktParser::serialize(&g);
        assert!(s.contains("POINT Z"));
    }

    #[test]
    fn test_serialize_linestring_round_trip() {
        let wkt = "LINESTRING (0 0, 1 1, 2 2)";
        let g = WktParser::parse(wkt).expect("should succeed");
        let s = WktParser::serialize(&g);
        assert!(s.starts_with("LINESTRING"));
        // Parse serialized back
        let g2 = WktParser::parse(&s).expect("should succeed");
        assert_eq!(WktParser::point_count(&g), WktParser::point_count(&g2));
    }

    #[test]
    fn test_serialize_polygon() {
        let wkt = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
        let g = WktParser::parse(wkt).expect("should succeed");
        let s = WktParser::serialize(&g);
        assert!(s.starts_with("POLYGON"));
    }

    // --- bounding_box ---

    #[test]
    fn test_bounding_box_point() {
        let g = WktGeometry::Point {
            x: 5.0,
            y: 7.0,
            z: None,
        };
        let bb = WktParser::bounding_box(&g).expect("should succeed");
        assert!(approx(bb.0, 5.0));
        assert!(approx(bb.1, 7.0));
        assert!(approx(bb.2, 5.0));
        assert!(approx(bb.3, 7.0));
    }

    #[test]
    fn test_bounding_box_linestring() {
        let g = WktParser::parse("LINESTRING (0 5, 3 2, 1 8)").expect("should succeed");
        let bb = WktParser::bounding_box(&g).expect("should succeed");
        assert!(approx(bb.0, 0.0));
        assert!(approx(bb.1, 2.0));
        assert!(approx(bb.2, 3.0));
        assert!(approx(bb.3, 8.0));
    }

    // --- centroid ---

    #[test]
    fn test_centroid_point() {
        let g = WktGeometry::Point {
            x: 2.0,
            y: 4.0,
            z: None,
        };
        let c = WktParser::centroid(&g).expect("should succeed");
        assert!(approx(c.0, 2.0));
        assert!(approx(c.1, 4.0));
    }

    #[test]
    fn test_centroid_linestring() {
        let g = WktParser::parse("LINESTRING (0 0, 2 0, 2 2, 0 2)").expect("should succeed");
        let c = WktParser::centroid(&g).expect("should succeed");
        assert!(approx(c.0, 1.0));
        assert!(approx(c.1, 1.0));
    }

    // --- geometry_type ---

    #[test]
    fn test_geometry_type_point() {
        let g = WktGeometry::Point {
            x: 0.0,
            y: 0.0,
            z: None,
        };
        assert_eq!(WktParser::geometry_type(&g), "POINT");
    }

    #[test]
    fn test_geometry_type_linestring() {
        let g = WktGeometry::LineString { points: vec![] };
        assert_eq!(WktParser::geometry_type(&g), "LINESTRING");
    }

    #[test]
    fn test_geometry_type_polygon() {
        let g = WktGeometry::Polygon {
            exterior: vec![],
            holes: vec![],
        };
        assert_eq!(WktParser::geometry_type(&g), "POLYGON");
    }

    #[test]
    fn test_geometry_type_multipoint() {
        let g = WktGeometry::MultiPoint { points: vec![] };
        assert_eq!(WktParser::geometry_type(&g), "MULTIPOINT");
    }

    #[test]
    fn test_geometry_type_multilinestring() {
        let g = WktGeometry::MultiLineString { lines: vec![] };
        assert_eq!(WktParser::geometry_type(&g), "MULTILINESTRING");
    }

    #[test]
    fn test_geometry_type_multipolygon() {
        let g = WktGeometry::MultiPolygon { polygons: vec![] };
        assert_eq!(WktParser::geometry_type(&g), "MULTIPOLYGON");
    }

    #[test]
    fn test_geometry_type_collection() {
        let g = WktGeometry::GeometryCollection { geometries: vec![] };
        assert_eq!(WktParser::geometry_type(&g), "GEOMETRYCOLLECTION");
    }

    // --- is_empty ---

    #[test]
    fn test_is_empty_linestring() {
        let g = WktGeometry::LineString { points: vec![] };
        assert!(WktParser::is_empty(&g));
    }

    #[test]
    fn test_is_not_empty_point() {
        let g = WktGeometry::Point {
            x: 1.0,
            y: 2.0,
            z: None,
        };
        assert!(!WktParser::is_empty(&g));
    }

    // --- error cases ---

    #[test]
    fn test_invalid_wkt_type() {
        let err = WktParser::parse("CIRCLE (1 2 5)");
        assert!(err.is_err());
    }

    #[test]
    fn test_invalid_coordinate_non_numeric() {
        let err = WktParser::parse("POINT (abc def)");
        assert!(err.is_err());
    }

    // --- MULTILINESTRING ---

    #[test]
    fn test_parse_multilinestring() {
        let g = WktParser::parse("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3, 4 4))")
            .expect("should succeed");
        match &g {
            WktGeometry::MultiLineString { lines } => {
                assert_eq!(lines.len(), 2);
                assert_eq!(lines[0].len(), 2);
                assert_eq!(lines[1].len(), 3);
            }
            _ => panic!("Expected MultiLineString"),
        }
    }

    // --- WktError display ---

    #[test]
    fn test_parse_error_display() {
        let e = WktError::ParseError("oops".to_string());
        assert!(format!("{e}").contains("oops"));
    }

    #[test]
    fn test_empty_geometry_display() {
        let e = WktError::EmptyGeometry;
        assert!(format!("{e}").contains("empty"));
    }

    #[test]
    fn test_invalid_coordinate_display() {
        let e = WktError::InvalidCoordinate("NaN".to_string());
        assert!(format!("{e}").contains("NaN"));
    }

    // --- point_count for polygon with hole ---

    #[test]
    fn test_point_count_polygon_with_hole() {
        let g =
            WktParser::parse("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))")
                .expect("should succeed");
        // 5 exterior + 5 hole
        assert_eq!(WktParser::point_count(&g), 10);
    }
}
