//! WKT (Well-Known Text) geometry serializer.
//!
//! Serialises geometry values to the OGC WKT format as required by
//! GeoSPARQL 1.1 (geo:wktLiteral).

// ── Geometry ──────────────────────────────────────────────────────────────────

/// An in-memory geometry representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// A single point.
    Point {
        /// Longitude / easting coordinate.
        x: f64,
        /// Latitude / northing coordinate.
        y: f64,
    },
    /// A sequence of connected points.
    LineString {
        /// Ordered sequence of (x, y) coordinate pairs.
        points: Vec<(f64, f64)>,
    },
    /// A polygon with an exterior ring and zero or more interior rings (holes).
    Polygon {
        /// Exterior boundary ring (closed coordinate sequence).
        exterior: Vec<(f64, f64)>,
        /// Interior rings (holes), each a closed coordinate sequence.
        interiors: Vec<Vec<(f64, f64)>>,
    },
    /// A set of disconnected points.
    MultiPoint {
        /// The individual point coordinates.
        points: Vec<(f64, f64)>,
    },
    /// A set of disconnected line strings.
    MultiLineString {
        /// Each inner `Vec` is one line string.
        lines: Vec<Vec<(f64, f64)>>,
    },
    /// A heterogeneous collection of geometries.
    GeometryCollection {
        /// The member geometries of this collection.
        geometries: Vec<Geometry>,
    },
}

// ── WktWriter ─────────────────────────────────────────────────────────────────

/// Serialises `Geometry` values to OGC Well-Known Text.
pub struct WktWriter;

impl WktWriter {
    // ── Dispatch ──────────────────────────────────────────────────────────────

    /// Write any geometry to its WKT string representation.
    pub fn write(geom: &Geometry) -> String {
        match geom {
            Geometry::Point { x, y } => Self::write_point(*x, *y),
            Geometry::LineString { points } => Self::write_linestring(points),
            Geometry::Polygon {
                exterior,
                interiors,
            } => Self::write_polygon(exterior, interiors),
            Geometry::MultiPoint { points } => Self::write_multi_point(points),
            Geometry::MultiLineString { lines } => Self::write_multi_linestring(lines),
            Geometry::GeometryCollection { geometries } => Self::write_collection(geometries),
        }
    }

    // ── Individual serializers ─────────────────────────────────────────────────

    /// Serialise a point: `POINT (x y)`.
    pub fn write_point(x: f64, y: f64) -> String {
        format!("POINT ({} {})", Self::fmt(x), Self::fmt(y))
    }

    /// Serialise a line string: `LINESTRING (x1 y1, x2 y2, ...)`.
    ///
    /// An empty line string is written as `LINESTRING EMPTY`.
    pub fn write_linestring(points: &[(f64, f64)]) -> String {
        if points.is_empty() {
            return "LINESTRING EMPTY".to_string();
        }
        format!("LINESTRING ({})", Self::coord_seq(points))
    }

    /// Serialise a polygon.
    ///
    /// `POLYGON ((ext_ring), (hole1), (hole2), ...)`
    ///
    /// An empty polygon is `POLYGON EMPTY`.
    pub fn write_polygon(exterior: &[(f64, f64)], interiors: &[Vec<(f64, f64)>]) -> String {
        if exterior.is_empty() {
            return "POLYGON EMPTY".to_string();
        }
        let mut parts: Vec<String> = vec![format!("({})", Self::coord_seq(exterior))];
        for ring in interiors {
            parts.push(format!("({})", Self::coord_seq(ring)));
        }
        format!("POLYGON ({})", parts.join(", "))
    }

    /// Serialise a multi-point: `MULTIPOINT ((x1 y1), (x2 y2), ...)`.
    ///
    /// Empty → `MULTIPOINT EMPTY`.
    pub fn write_multi_point(points: &[(f64, f64)]) -> String {
        if points.is_empty() {
            return "MULTIPOINT EMPTY".to_string();
        }
        let inner: Vec<String> = points
            .iter()
            .map(|(x, y)| format!("({} {})", Self::fmt(*x), Self::fmt(*y)))
            .collect();
        format!("MULTIPOINT ({})", inner.join(", "))
    }

    /// Serialise a multi-line string.
    pub fn write_multi_linestring(lines: &[Vec<(f64, f64)>]) -> String {
        if lines.is_empty() {
            return "MULTILINESTRING EMPTY".to_string();
        }
        let inner: Vec<String> = lines
            .iter()
            .map(|pts| format!("({})", Self::coord_seq(pts)))
            .collect();
        format!("MULTILINESTRING ({})", inner.join(", "))
    }

    /// Serialise a geometry collection: `GEOMETRYCOLLECTION (g1, g2, ...)`.
    pub fn write_collection(geometries: &[Geometry]) -> String {
        if geometries.is_empty() {
            return "GEOMETRYCOLLECTION EMPTY".to_string();
        }
        let inner: Vec<String> = geometries.iter().map(Self::write).collect();
        format!("GEOMETRYCOLLECTION ({})", inner.join(", "))
    }

    /// Format a floating-point coordinate with `decimals` decimal places.
    ///
    /// Trailing zeros are preserved for consistent output.
    pub fn precision(value: f64, decimals: usize) -> String {
        format!("{:.prec$}", value, prec = decimals)
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Format a coordinate value, removing unnecessary trailing zeros.
    fn fmt(v: f64) -> String {
        // Use enough decimal places to avoid rounding artefacts, then strip
        // trailing zeros after the decimal point.
        let s = format!("{v:.10}");
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        if s.is_empty() || s == "-" {
            "0".to_string()
        } else {
            s.to_string()
        }
    }

    /// Format a coordinate sequence as `x1 y1, x2 y2, ...`.
    fn coord_seq(pts: &[(f64, f64)]) -> String {
        pts.iter()
            .map(|(x, y)| format!("{} {}", Self::fmt(*x), Self::fmt(*y)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── write_point ───────────────────────────────────────────────────────────

    #[test]
    fn test_write_point_basic() {
        let wkt = WktWriter::write_point(1.0, 2.0);
        assert_eq!(wkt, "POINT (1 2)");
    }

    #[test]
    fn test_write_point_negative() {
        let wkt = WktWriter::write_point(-73.985_130, 40.758_896);
        assert!(wkt.starts_with("POINT ("));
        assert!(wkt.contains("-73.98513"));
    }

    #[test]
    fn test_write_point_zero() {
        let wkt = WktWriter::write_point(0.0, 0.0);
        assert_eq!(wkt, "POINT (0 0)");
    }

    #[test]
    fn test_write_point_fractional() {
        let wkt = WktWriter::write_point(1.5, 2.75);
        assert!(wkt.contains("1.5"));
        assert!(wkt.contains("2.75"));
    }

    #[test]
    fn test_write_point_geometry_dispatch() {
        let geom = Geometry::Point { x: 3.0, y: 4.0 };
        assert_eq!(WktWriter::write(&geom), WktWriter::write_point(3.0, 4.0));
    }

    // ── write_linestring ──────────────────────────────────────────────────────

    #[test]
    fn test_write_linestring_two_points() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0)];
        let wkt = WktWriter::write_linestring(&pts);
        assert_eq!(wkt, "LINESTRING (0 0, 1 1)");
    }

    #[test]
    fn test_write_linestring_empty() {
        let wkt = WktWriter::write_linestring(&[]);
        assert_eq!(wkt, "LINESTRING EMPTY");
    }

    #[test]
    fn test_write_linestring_three_points() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let wkt = WktWriter::write_linestring(&pts);
        assert!(wkt.starts_with("LINESTRING ("));
        assert!(wkt.contains("1 0"));
    }

    #[test]
    fn test_write_linestring_dispatch() {
        let pts = vec![(0.0, 0.0), (5.0, 5.0)];
        let geom = Geometry::LineString {
            points: pts.clone(),
        };
        assert_eq!(WktWriter::write(&geom), WktWriter::write_linestring(&pts));
    }

    // ── write_polygon ─────────────────────────────────────────────────────────

    #[test]
    fn test_write_polygon_no_holes() {
        let ext = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let wkt = WktWriter::write_polygon(&ext, &[]);
        assert!(wkt.starts_with("POLYGON ("));
        assert!(!wkt.contains("EMPTY"));
    }

    #[test]
    fn test_write_polygon_empty() {
        let wkt = WktWriter::write_polygon(&[], &[]);
        assert_eq!(wkt, "POLYGON EMPTY");
    }

    #[test]
    fn test_write_polygon_with_hole() {
        let ext = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)];
        let hole = vec![(2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 2.0)];
        let wkt = WktWriter::write_polygon(&ext, &[hole]);
        assert!(wkt.contains("(0 0"));
        assert!(wkt.contains("(2 2"));
    }

    #[test]
    fn test_write_polygon_dispatch() {
        let ext = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.0, 0.0)];
        let geom = Geometry::Polygon {
            exterior: ext.clone(),
            interiors: vec![],
        };
        assert_eq!(WktWriter::write(&geom), WktWriter::write_polygon(&ext, &[]));
    }

    // ── write_multi_point ─────────────────────────────────────────────────────

    #[test]
    fn test_write_multi_point_two() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0)];
        let wkt = WktWriter::write_multi_point(&pts);
        assert!(wkt.starts_with("MULTIPOINT ("));
        assert!(wkt.contains("(0 0)"));
        assert!(wkt.contains("(1 1)"));
    }

    #[test]
    fn test_write_multi_point_empty() {
        let wkt = WktWriter::write_multi_point(&[]);
        assert_eq!(wkt, "MULTIPOINT EMPTY");
    }

    #[test]
    fn test_write_multi_point_single() {
        let pts = vec![(5.0, 10.0)];
        let wkt = WktWriter::write_multi_point(&pts);
        assert!(wkt.contains("(5 10)"));
    }

    #[test]
    fn test_write_multi_point_dispatch() {
        let pts = vec![(1.0, 2.0), (3.0, 4.0)];
        let geom = Geometry::MultiPoint {
            points: pts.clone(),
        };
        assert_eq!(WktWriter::write(&geom), WktWriter::write_multi_point(&pts));
    }

    // ── write_multi_linestring ────────────────────────────────────────────────

    #[test]
    fn test_write_multi_linestring_two_lines() {
        let lines = vec![vec![(0.0, 0.0), (1.0, 1.0)], vec![(2.0, 2.0), (3.0, 3.0)]];
        let wkt = WktWriter::write_multi_linestring(&lines);
        assert!(wkt.starts_with("MULTILINESTRING ("));
        assert!(wkt.contains("0 0, 1 1"));
    }

    #[test]
    fn test_write_multi_linestring_empty() {
        let wkt = WktWriter::write_multi_linestring(&[]);
        assert_eq!(wkt, "MULTILINESTRING EMPTY");
    }

    #[test]
    fn test_write_multi_linestring_dispatch() {
        let lines = vec![vec![(0.0, 0.0), (1.0, 0.0)]];
        let geom = Geometry::MultiLineString {
            lines: lines.clone(),
        };
        assert_eq!(
            WktWriter::write(&geom),
            WktWriter::write_multi_linestring(&lines)
        );
    }

    // ── write_collection ──────────────────────────────────────────────────────

    #[test]
    fn test_write_collection_mixed() {
        let geoms = vec![
            Geometry::Point { x: 1.0, y: 2.0 },
            Geometry::LineString {
                points: vec![(0.0, 0.0), (1.0, 1.0)],
            },
        ];
        let wkt = WktWriter::write_collection(&geoms);
        assert!(wkt.starts_with("GEOMETRYCOLLECTION ("));
        assert!(wkt.contains("POINT"));
        assert!(wkt.contains("LINESTRING"));
    }

    #[test]
    fn test_write_collection_empty() {
        let wkt = WktWriter::write_collection(&[]);
        assert_eq!(wkt, "GEOMETRYCOLLECTION EMPTY");
    }

    #[test]
    fn test_write_collection_dispatch() {
        let geoms = vec![Geometry::Point { x: 0.0, y: 0.0 }];
        let outer = Geometry::GeometryCollection {
            geometries: geoms.clone(),
        };
        assert_eq!(
            WktWriter::write(&outer),
            WktWriter::write_collection(&geoms)
        );
    }

    #[test]
    fn test_write_collection_nested() {
        let inner_coll = Geometry::GeometryCollection {
            geometries: vec![Geometry::Point { x: 1.0, y: 1.0 }],
        };
        let outer = Geometry::GeometryCollection {
            geometries: vec![inner_coll],
        };
        let wkt = WktWriter::write(&outer);
        assert!(wkt.contains("GEOMETRYCOLLECTION"));
        assert!(wkt.contains("POINT"));
    }

    // ── precision ─────────────────────────────────────────────────────────────

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_precision_two_decimals() {
        let s = WktWriter::precision(3.14159, 2);
        assert_eq!(s, "3.14");
    }

    #[test]
    fn test_precision_zero_decimals() {
        let s = WktWriter::precision(42.9, 0);
        assert_eq!(s, "43");
    }

    #[test]
    fn test_precision_many_decimals() {
        let s = WktWriter::precision(1.0 / 3.0, 6);
        assert!(s.starts_with("0.333333"));
    }

    #[test]
    fn test_precision_negative() {
        let s = WktWriter::precision(-1.5, 1);
        assert_eq!(s, "-1.5");
    }

    // ── Formatting edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_fmt_strips_trailing_zeros() {
        // Internal fmt should not leave "1.0000000" for 1.0
        let wkt = WktWriter::write_point(1.0, 2.0);
        assert!(!wkt.contains("1.0000000"));
    }

    #[test]
    fn test_write_large_coordinates() {
        let wkt = WktWriter::write_point(1_000_000.0, -999_999.5);
        assert!(wkt.contains("1000000"));
        assert!(wkt.contains("-999999.5"));
    }

    #[test]
    fn test_write_polygon_multiple_holes() {
        let ext = vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 0.0)];
        let h1 = vec![(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 1.0)];
        let h2 = vec![(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 10.0)];
        let wkt = WktWriter::write_polygon(&ext, &[h1, h2]);
        // Should have three ring groups
        let open_parens = wkt.chars().filter(|&c| c == '(').count();
        assert!(open_parens >= 3);
    }

    #[test]
    fn test_geometry_clone() {
        let g = Geometry::Point { x: 1.0, y: 2.0 };
        let g2 = g.clone();
        assert_eq!(g, g2);
    }

    #[test]
    fn test_geometry_debug() {
        let g = Geometry::Point { x: 0.0, y: 0.0 };
        let s = format!("{g:?}");
        assert!(s.contains("Point"));
    }

    #[test]
    fn test_write_point_origin() {
        let wkt = WktWriter::write_point(0.0, 0.0);
        assert_eq!(wkt, "POINT (0 0)");
    }

    #[test]
    fn test_write_linestring_diagonal() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        let wkt = WktWriter::write_linestring(&points);
        assert!(wkt.starts_with("LINESTRING"));
        assert!(wkt.contains("0 0"));
        assert!(wkt.contains("1 1"));
    }

    #[test]
    fn test_write_multi_point_three() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let wkt = WktWriter::write_multi_point(&pts);
        assert!(wkt.starts_with("MULTIPOINT"));
        // Three coordinate pairs
        assert_eq!(wkt.matches(',').count(), 2);
    }

    #[test]
    fn test_write_collection_single_linestring() {
        let g = Geometry::GeometryCollection {
            geometries: vec![Geometry::LineString {
                points: vec![(0.0, 0.0), (5.0, 5.0)],
            }],
        };
        let wkt = WktWriter::write(&g);
        assert!(wkt.contains("LINESTRING"));
    }

    #[test]
    fn test_write_multi_linestring_single() {
        let lines = vec![vec![(0.0, 0.0), (1.0, 1.0)]];
        let wkt = WktWriter::write_multi_linestring(&lines);
        assert!(wkt.starts_with("MULTILINESTRING"));
    }

    #[test]
    fn test_write_polygon_exterior_only() {
        let ext = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let wkt = WktWriter::write_polygon(&ext, &[]);
        assert!(wkt.starts_with("POLYGON"));
        // Only one ring → exactly one group of coordinates in ((…))
        assert_eq!(wkt.matches("((").count(), 1);
    }

    #[test]
    fn test_write_collection_empty_geometries() {
        let wkt = WktWriter::write_collection(&[]);
        assert_eq!(wkt, "GEOMETRYCOLLECTION EMPTY");
    }

    #[test]
    fn test_write_linestring_comma_count() {
        let points = vec![(0.0, 0.0), (1.0, 2.0), (3.0, 4.0)];
        let wkt = WktWriter::write_linestring(&points);
        // Should have two commas separating three pairs
        assert_eq!(wkt.matches(',').count(), 2);
    }

    #[test]
    fn test_geometry_collection_variant_debug() {
        let g = Geometry::GeometryCollection { geometries: vec![] };
        let s = format!("{g:?}");
        assert!(s.contains("GeometryCollection"));
    }

    #[test]
    fn test_write_point_negative_coords() {
        let wkt = WktWriter::write_point(-10.5, -20.25);
        assert!(wkt.contains("-10.5"));
        assert!(wkt.contains("-20.25"));
    }

    #[test]
    fn test_precision_zero() {
        let s = WktWriter::precision(0.0, 3);
        assert_eq!(s, "0.000");
    }

    #[test]
    fn test_multipoint_empty() {
        let wkt = WktWriter::write_multi_point(&[]);
        assert_eq!(wkt, "MULTIPOINT EMPTY");
    }
}
