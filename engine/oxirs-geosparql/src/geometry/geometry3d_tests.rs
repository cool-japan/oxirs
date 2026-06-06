//! Tests for the 3D geometry types and operations.
//!
//! These cover construction, distance/area/volume computations, WKT
//! roundtrips, z-range queries, bounding boxes, and 2D projection across
//! all [`Geometry3DEnum`] variants.

#[cfg(test)]
mod tests {
    use crate::geometry::geometry3d::*;

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
    use crate::geometry::geometry3d::*;

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
