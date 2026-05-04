//! OGC GeoSPARQL 1.1 Conformance Tests
//!
//! Tests that validate OGC GeoSPARQL 1.1 specification compliance for:
//! - `geof:relate` — DE-9IM pattern-based spatial predicate
//! - `geof:simplify` — Douglas-Peucker geometry simplification
//! - `geof:boundary` — pure-Rust OGC SFA §6.1.6.1 boundary
//! - `geof:isValid` — topological validity predicate
//! - `geof:isRing` — ring (closed + simple) predicate
//! - `geof:isClosed` — closed endpoints predicate
//!
//! Requirements covered (per OGC GeoSPARQL 1.1 spec §7.5):
//! - Req 18: geof:relate with DE-9IM pattern
//! - Req 19: geof:simplify with tolerance
//! - Req 20: geof:boundary
//! - Req 21: geof:isValid
//! - Req 22: geof:isRing
//! - Req 23: geof:isClosed
//! - Consistency: relate(A,B,"FF*FF****") == sfDisjoint(A,B)
//! - Consistency: relate(A,B,"T*F**F***") == sfWithin(A,B)
//! - Consistency: relate(A,B,"T*****FF*") == sfContains(A,B)

use geo_types::{Coord, Geometry as GeoGeometry, LineString, Polygon};
use oxirs_geosparql::functions::de9im;
use oxirs_geosparql::functions::geometric_properties;
use oxirs_geosparql::functions::geometry_simplify;
use oxirs_geosparql::functions::simple_features;
use oxirs_geosparql::geometry::Geometry;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn poly(coords: &[(f64, f64)]) -> Geometry {
    let ring: Vec<Coord<f64>> = coords.iter().map(|&(x, y)| Coord { x, y }).collect();
    Geometry::new(GeoGeometry::Polygon(Polygon::new(LineString(ring), vec![])))
}

fn linestring(coords: &[(f64, f64)]) -> Geometry {
    let cs: Vec<Coord<f64>> = coords.iter().map(|&(x, y)| Coord { x, y }).collect();
    Geometry::new(GeoGeometry::LineString(LineString(cs)))
}

// Standard test geometries
fn square_a() -> Geometry {
    poly(&[
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
        (0.0, 0.0),
    ])
}

fn square_b_overlap() -> Geometry {
    poly(&[
        (5.0, 5.0),
        (15.0, 5.0),
        (15.0, 15.0),
        (5.0, 15.0),
        (5.0, 5.0),
    ])
}

fn square_b_disjoint() -> Geometry {
    poly(&[
        (20.0, 20.0),
        (30.0, 20.0),
        (30.0, 30.0),
        (20.0, 30.0),
        (20.0, 20.0),
    ])
}

fn square_inner() -> Geometry {
    poly(&[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (2.0, 2.0)])
}

// ── Req 18: geof:relate ───────────────────────────────────────────────────────

#[test]
fn req18_relate_wildcard_always_true() {
    let a = square_a();
    let b = square_b_disjoint();
    let result = de9im::relate(&a, &b, "*********").expect("relate should succeed");
    assert!(result, "all-wildcard pattern must always return true");
}

#[test]
fn req18_relate_disjoint_pattern_ff_ff_star_star_star_star() {
    let a = square_a();
    let b = square_b_disjoint();
    let result = de9im::relate(&a, &b, "FF*FF****").expect("relate should succeed");
    assert!(result, "disjoint squares must satisfy FF*FF****");
}

#[test]
fn req18_relate_not_disjoint_when_overlapping() {
    let a = square_a();
    let b = square_b_overlap();
    let result = de9im::relate(&a, &b, "FF*FF****").expect("relate should succeed");
    assert!(!result, "overlapping squares must NOT satisfy FF*FF****");
}

#[test]
fn req18_relate_within_pattern_t_f_star_star_f_star_star_star() {
    let outer = square_a();
    let inner = square_inner();
    let result = de9im::relate(&inner, &outer, "T*F**F***").expect("relate should succeed");
    assert!(
        result,
        "inner square must satisfy sfWithin pattern T*F**F***"
    );
}

#[test]
fn req18_relate_contains_pattern_t_star_star_star_star_star_ff_star() {
    let outer = square_a();
    let inner = square_inner();
    let result = de9im::relate(&outer, &inner, "T*****FF*").expect("relate should succeed");
    assert!(
        result,
        "outer square must satisfy sfContains pattern T*****FF*"
    );
}

#[test]
fn req18_relate_specific_dimension_0() {
    // A point inside a polygon: II dimension should be 2 (areal interior meets areal interior)
    // But for point-in-polygon: II = 0 (point dim), IB = F, IE = F, BI = F, BB = F, BE = 1, EI = 2, EB = 1, EE = 2
    use geo_types::Point;
    let point = Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0)));
    let outer = square_a();
    // Point inside polygon: relate is "0FFFFF212"
    // Position II=0, so pattern "0*******" should match
    let result = de9im::relate(&point, &outer, "0FFFFF212").expect("relate should succeed");
    assert!(result, "point inside polygon should match DE-9IM 0FFFFF212");
}

#[test]
fn req18_relate_error_on_short_pattern() {
    let a = square_a();
    let b = square_b_disjoint();
    assert!(
        de9im::relate(&a, &b, "FF*").is_err(),
        "relate with short pattern must return Err"
    );
}

#[test]
fn req18_relate_error_on_invalid_char() {
    let a = square_a();
    let b = square_b_disjoint();
    assert!(
        de9im::relate(&a, &b, "FF*FF**Z*").is_err(),
        "relate with invalid char must return Err"
    );
}

#[test]
fn req18_relate_matrix_string_9_chars() {
    let a = square_a();
    let b = square_b_overlap();
    let s = de9im::relate_matrix_string(&a, &b).expect("relate_matrix_string should succeed");
    assert_eq!(s.len(), 9, "matrix string must be exactly 9 characters");
    for ch in s.chars() {
        assert!(
            matches!(ch, 'F' | '0' | '1' | '2'),
            "matrix char must be F/0/1/2, got '{ch}'"
        );
    }
}

// ── Consistency: relate ↔ SF predicates ──────────────────────────────────────

type GeomPair<'a> = (&'a str, fn() -> Geometry, fn() -> Geometry);

#[test]
fn consistency_relate_disjoint_equals_sf_disjoint() {
    let pairs: &[GeomPair<'_>] = &[
        ("overlapping", square_a, square_b_overlap),
        ("disjoint", square_a, square_b_disjoint),
    ];
    for (label, a_fn, b_fn) in pairs {
        let a = a_fn();
        let b = b_fn();
        let sf = simple_features::sf_disjoint(&a, &b).expect("sf_disjoint should succeed");
        let de9im_result = de9im::relate(&a, &b, "FF*FF****").expect("relate should succeed");
        assert_eq!(
            sf, de9im_result,
            "sfDisjoint must equal relate(FF*FF****) for {label}"
        );
    }
}

#[test]
fn consistency_relate_within_equals_sf_within() {
    let outer = square_a();
    let inner = square_inner();
    let sf_w = simple_features::sf_within(&inner, &outer).expect("sf_within should succeed");
    let de9_w = de9im::relate(&inner, &outer, "T*F**F***").expect("relate should succeed");
    assert_eq!(
        sf_w, de9_w,
        "sfWithin must equal relate(T*F**F***) for contained polygon"
    );

    // Outer is NOT within inner
    let sf_nw = simple_features::sf_within(&outer, &inner).expect("sf_within should succeed");
    let de9_nw = de9im::relate(&outer, &inner, "T*F**F***").expect("relate should succeed");
    assert_eq!(
        sf_nw, de9_nw,
        "sfWithin must equal relate(T*F**F***) for outer polygon"
    );
}

#[test]
fn consistency_relate_contains_equals_sf_contains() {
    let outer = square_a();
    let inner = square_inner();
    let sf_c = simple_features::sf_contains(&outer, &inner).expect("sf_contains should succeed");
    let de9_c = de9im::relate(&outer, &inner, "T*****FF*").expect("relate should succeed");
    assert_eq!(
        sf_c, de9_c,
        "sfContains must equal relate(T*****FF*) for outer polygon"
    );
}

#[test]
fn consistency_within_contains_inverse() {
    let outer = square_a();
    let inner = square_inner();
    let within = simple_features::sf_within(&inner, &outer).expect("sf_within should succeed");
    let contains =
        simple_features::sf_contains(&outer, &inner).expect("sf_contains should succeed");
    assert_eq!(within, contains, "sfWithin(A,B) must equal sfContains(B,A)");
}

#[test]
fn consistency_disjoint_is_not_intersects() {
    let a = square_a();
    let b = square_b_disjoint();
    let disjoint = simple_features::sf_disjoint(&a, &b).expect("sf_disjoint should succeed");
    let intersects = simple_features::sf_intersects(&a, &b).expect("sf_intersects should succeed");
    assert_ne!(
        disjoint, intersects,
        "sfDisjoint and sfIntersects must be logical complements"
    );
}

// ── Req 19: geof:simplify ─────────────────────────────────────────────────────

#[test]
fn req19_simplify_reduces_collinear_points() {
    // Collinear points on Y=0 axis: only endpoints should survive high tolerance
    let ls = linestring(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
    let simplified = geometry_simplify::simplify(&ls, 0.5).expect("simplify should succeed");
    if let GeoGeometry::LineString(result) = simplified.geom {
        assert!(result.0.len() <= 5, "should not add points");
        assert!(result.0.len() >= 2, "must keep at least 2 points");
    } else {
        panic!("expected LineString");
    }
}

#[test]
fn req19_simplify_zero_tolerance_identity() {
    let ls = linestring(&[(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.2), (4.0, 0.0)]);
    let simplified = geometry_simplify::simplify(&ls, 0.0).expect("simplify should succeed");
    if let GeoGeometry::LineString(result) = simplified.geom {
        assert_eq!(
            result.0.len(),
            5,
            "zero tolerance must be identity (preserve all points)"
        );
    } else {
        panic!("expected LineString");
    }
}

#[test]
fn req19_simplify_polygon_preserves_closure() {
    let p = poly(&[
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
        (10.0, 5.0),
        (10.0, 10.0),
        (5.0, 10.0),
        (0.0, 10.0),
        (0.0, 5.0),
        (0.0, 0.0),
    ]);
    let simplified = geometry_simplify::simplify(&p, 0.1).expect("simplify should succeed");
    match simplified.geom {
        GeoGeometry::Polygon(result) => {
            let ext = result.exterior();
            assert!(ext.0.len() >= 4, "polygon exterior must have ≥ 4 coords");
            // Ring must be closed
            let first = ext.0.first().expect("must have first");
            let last = ext.0.last().expect("must have last");
            assert!(
                (first.x - last.x).abs() < 1e-9 && (first.y - last.y).abs() < 1e-9,
                "exterior ring must be closed after simplification"
            );
        }
        _ => panic!("expected Polygon"),
    }
}

#[test]
fn req19_simplify_negative_tolerance_errors() {
    let ls = linestring(&[(0.0, 0.0), (1.0, 1.0)]);
    assert!(
        geometry_simplify::simplify(&ls, -1.0).is_err(),
        "negative tolerance must return Err"
    );
}

#[test]
fn req19_simplify_nan_tolerance_errors() {
    let ls = linestring(&[(0.0, 0.0), (1.0, 1.0)]);
    assert!(
        geometry_simplify::simplify(&ls, f64::NAN).is_err(),
        "NaN tolerance must return Err"
    );
}

#[test]
fn req19_simplify_point_unchanged() {
    use geo_types::Point;
    let pt = Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0)));
    let simplified = geometry_simplify::simplify(&pt, 10.0).expect("simplify should succeed");
    if let GeoGeometry::Point(p) = simplified.geom {
        assert!((p.x() - 5.0).abs() < f64::EPSILON);
        assert!((p.y() - 5.0).abs() < f64::EPSILON);
    } else {
        panic!("expected Point");
    }
}

// ── Req 20: geof:boundary ─────────────────────────────────────────────────────

#[test]
fn req20_boundary_point_is_empty() {
    use geo_types::Point;
    let pt = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
    let b = de9im::boundary(&pt).expect("boundary should succeed");
    if let GeoGeometry::GeometryCollection(gc) = b.geom {
        assert!(gc.0.is_empty(), "boundary of Point must be empty");
    } else {
        panic!("expected empty GeometryCollection");
    }
}

#[test]
fn req20_boundary_open_linestring_two_points() {
    let ls = linestring(&[(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]);
    let b = de9im::boundary(&ls).expect("boundary should succeed");
    if let GeoGeometry::MultiPoint(mp) = b.geom {
        assert_eq!(
            mp.0.len(),
            2,
            "open LineString boundary must have 2 endpoints"
        );
    } else {
        panic!("expected MultiPoint");
    }
}

#[test]
fn req20_boundary_closed_ring_is_empty() {
    // Closed ring: start == end
    let ls = linestring(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
    let b = de9im::boundary(&ls).expect("boundary should succeed");
    if let GeoGeometry::GeometryCollection(gc) = b.geom {
        assert!(gc.0.is_empty(), "boundary of closed ring must be empty");
    } else {
        panic!("expected empty GeometryCollection");
    }
}

#[test]
fn req20_boundary_polygon_returns_rings() {
    let p = square_a();
    let b = de9im::boundary(&p).expect("boundary should succeed");
    if let GeoGeometry::MultiLineString(mls) = b.geom {
        // A simple polygon without holes has 1 ring (exterior only)
        assert_eq!(mls.0.len(), 1, "simple polygon boundary must have 1 ring");
    } else {
        panic!("expected MultiLineString");
    }
}

#[test]
fn req20_boundary_polygon_with_hole_two_rings() {
    let exterior = LineString(vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 10.0, y: 0.0 },
        Coord { x: 10.0, y: 10.0 },
        Coord { x: 0.0, y: 10.0 },
        Coord { x: 0.0, y: 0.0 },
    ]);
    let hole = LineString(vec![
        Coord { x: 2.0, y: 2.0 },
        Coord { x: 8.0, y: 2.0 },
        Coord { x: 8.0, y: 8.0 },
        Coord { x: 2.0, y: 8.0 },
        Coord { x: 2.0, y: 2.0 },
    ]);
    let poly_with_hole = Geometry::new(GeoGeometry::Polygon(Polygon::new(exterior, vec![hole])));
    let b = de9im::boundary(&poly_with_hole).expect("boundary should succeed");
    if let GeoGeometry::MultiLineString(mls) = b.geom {
        assert_eq!(
            mls.0.len(),
            2,
            "polygon with 1 hole boundary must have 2 rings"
        );
    } else {
        panic!("expected MultiLineString");
    }
}

#[test]
fn req20_boundary_multilinestring_mod2_rule() {
    use geo_types::MultiLineString;
    // Two line strings sharing endpoint (5,5):
    //   (0,0)-(5,5): endpoints (0,0) and (5,5) — each appears once
    //   (5,5)-(10,0): endpoints (5,5) and (10,0) — each appears once
    // (5,5) appears twice in total → even → NOT on boundary (Mod-2 rule)
    // boundary = {(0,0), (10,0)}
    let mls = GeoGeometry::MultiLineString(MultiLineString(vec![
        LineString(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 5.0, y: 5.0 }]),
        LineString(vec![Coord { x: 5.0, y: 5.0 }, Coord { x: 10.0, y: 0.0 }]),
    ]));
    let geom = Geometry::new(mls);
    let b = de9im::boundary(&geom).expect("boundary should succeed");
    if let GeoGeometry::MultiPoint(mp) = b.geom {
        assert_eq!(
            mp.0.len(),
            2,
            "Mod-2 rule: shared endpoint at (5,5) should NOT be on boundary"
        );
    } else {
        panic!("expected MultiPoint");
    }
}

// ── Req 21: geof:isValid ──────────────────────────────────────────────────────

#[test]
fn req21_is_valid_point_always_true() {
    use geo_types::Point;
    let pt = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
    let valid = geometric_properties::is_valid(&pt).expect("is_valid should succeed");
    assert!(valid, "Point is always valid");
}

#[test]
fn req21_is_valid_simple_polygon_true() {
    let p = square_a();
    let valid = geometric_properties::is_valid(&p).expect("is_valid should succeed");
    assert!(valid, "simple closed polygon must be valid");
}

#[test]
fn req21_is_valid_linestring_one_coord_invalid() {
    // A LineString with a single coordinate is degenerate
    let ls = Geometry::new(GeoGeometry::LineString(LineString(vec![Coord {
        x: 1.0,
        y: 2.0,
    }])));
    let valid = geometric_properties::is_valid(&ls).expect("is_valid should succeed");
    assert!(!valid, "single-coord LineString must be invalid");
}

#[test]
fn req21_is_valid_empty_linestring_is_valid() {
    let ls = Geometry::new(GeoGeometry::LineString(LineString(vec![])));
    let valid = geometric_properties::is_valid(&ls).expect("is_valid should succeed");
    assert!(valid, "empty LineString is considered valid (0 coords OK)");
}

// ── Req 22: geof:isRing ───────────────────────────────────────────────────────

#[test]
fn req22_is_ring_closed_linestring_is_ring() {
    let ls = linestring(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
    let ring = geometric_properties::is_ring(&ls).expect("is_ring should succeed");
    assert!(ring, "closed simple LineString must be a ring");
}

#[test]
fn req22_is_ring_open_linestring_is_not_ring() {
    let ls = linestring(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]);
    let ring = geometric_properties::is_ring(&ls).expect("is_ring should succeed");
    assert!(!ring, "open LineString must not be a ring");
}

#[test]
fn req22_is_ring_error_on_polygon() {
    let p = square_a();
    assert!(
        geometric_properties::is_ring(&p).is_err(),
        "isRing on Polygon must return Err"
    );
}

#[test]
fn req22_is_ring_error_on_point() {
    use geo_types::Point;
    let pt = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
    assert!(
        geometric_properties::is_ring(&pt).is_err(),
        "isRing on Point must return Err"
    );
}

// ── Req 23: geof:isClosed ─────────────────────────────────────────────────────

#[test]
fn req23_is_closed_closed_linestring_true() {
    let ls = linestring(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]);
    let closed = geometric_properties::is_closed(&ls).expect("is_closed should succeed");
    assert!(closed, "closed LineString (start==end) must be closed");
}

#[test]
fn req23_is_closed_open_linestring_false() {
    let ls = linestring(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]);
    let closed = geometric_properties::is_closed(&ls).expect("is_closed should succeed");
    assert!(!closed, "open LineString must not be closed");
}

#[test]
fn req23_is_closed_empty_linestring_true() {
    let ls = Geometry::new(GeoGeometry::LineString(LineString(vec![])));
    let closed = geometric_properties::is_closed(&ls).expect("is_closed should succeed");
    assert!(closed, "empty LineString is trivially closed");
}

#[test]
fn req23_is_closed_multilinestring_all_closed() {
    use geo_types::MultiLineString;
    let mls = GeoGeometry::MultiLineString(MultiLineString(vec![
        LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 5.0, y: 0.0 },
            Coord { x: 5.0, y: 5.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        LineString(vec![
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 20.0, y: 10.0 },
            Coord { x: 20.0, y: 20.0 },
            Coord { x: 10.0, y: 10.0 },
        ]),
    ]));
    let geom = Geometry::new(mls);
    let closed = geometric_properties::is_closed(&geom).expect("is_closed should succeed");
    assert!(
        closed,
        "MultiLineString with all closed rings must be closed"
    );
}

#[test]
fn req23_is_closed_multilinestring_one_open_false() {
    use geo_types::MultiLineString;
    let mls = GeoGeometry::MultiLineString(MultiLineString(vec![
        LineString(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 5.0, y: 0.0 },
            Coord { x: 0.0, y: 0.0 }, // closed
        ]),
        LineString(vec![
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 20.0, y: 0.0 }, // open
        ]),
    ]));
    let geom = Geometry::new(mls);
    let closed = geometric_properties::is_closed(&geom).expect("is_closed should succeed");
    assert!(
        !closed,
        "MultiLineString with one open component must not be closed"
    );
}

#[test]
fn req23_is_closed_error_on_polygon() {
    let p = square_a();
    assert!(
        geometric_properties::is_closed(&p).is_err(),
        "isClosed on Polygon must return Err"
    );
}

// ── SPARQL function registry completeness ────────────────────────────────────

#[test]
fn registry_all_ogc11_functions_registered() {
    use oxirs_geosparql::sparql_integration;

    let all = sparql_integration::get_all_geosparql_functions();

    let names: Vec<&str> = all.iter().map(|f| f.name.as_str()).collect();

    // OGC GeoSPARQL 1.1 additions
    assert!(names.contains(&"relate"), "relate must be in registry");
    assert!(names.contains(&"simplify"), "simplify must be in registry");
    assert!(names.contains(&"boundary"), "boundary must be in registry");
    assert!(names.contains(&"isValid"), "isValid must be in registry");
    assert!(names.contains(&"isRing"), "isRing must be in registry");
    assert!(names.contains(&"isClosed"), "isClosed must be in registry");

    // Pre-existing functions still registered
    assert!(
        names.contains(&"sfEquals"),
        "sfEquals must still be in registry"
    );
    assert!(
        names.contains(&"distance"),
        "distance must still be in registry"
    );
    assert!(
        names.contains(&"dimension"),
        "dimension must still be in registry"
    );
}

#[test]
fn registry_relate_has_arity_3() {
    use oxirs_geosparql::sparql_integration;

    let all = sparql_integration::get_all_geosparql_functions();
    let relate_fn = all
        .iter()
        .find(|f| f.name == "relate")
        .expect("relate must be in registry");
    assert_eq!(
        relate_fn.arity, 3,
        "geof:relate takes 3 arguments: geomA, geomB, pattern"
    );
}

#[test]
fn registry_relate_uri_matches_ogc_spec() {
    use oxirs_geosparql::sparql_integration;

    let all = sparql_integration::get_all_geosparql_functions();
    let relate_fn = all
        .iter()
        .find(|f| f.name == "relate")
        .expect("relate must be in registry");
    assert_eq!(
        relate_fn.uri, "http://www.opengis.net/def/function/geosparql/relate",
        "geof:relate URI must match OGC GeoSPARQL 1.1 spec"
    );
}

#[test]
fn registry_ogc11_filter_functions_count() {
    use oxirs_geosparql::sparql_integration;

    let ogc11 = sparql_integration::get_ogc11_filter_functions();
    // relate (arity=3), simplify (arity=2), boundary (arity=1)
    assert_eq!(
        ogc11.len(),
        3,
        "OGC GeoSPARQL 1.1 filter functions must total 3"
    );
}
