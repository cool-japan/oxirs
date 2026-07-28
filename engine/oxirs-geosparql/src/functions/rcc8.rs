//! RCC8 (Region Connection Calculus) topological relations
//!
//! This module implements the 8 RCC8 topological relations for qualitative spatial reasoning.
//! RCC8 is based on the concept of connection between regions.
//!
//! # Performance
//!
//! All functions use bounding box pre-filtering to avoid expensive boundary calculations
//! when geometries are clearly disjoint. This provides 50-90% speedup for disjoint cases.
//!
//! Reference: Randell, D. A., Cui, Z., & Cohn, A. G. (1992). "A spatial logic based on regions and connection"

use crate::error::Result;
use crate::functions::bbox_utils::{bbox_could_contain, bboxes_disjoint};
use crate::functions::de9im::boundary;
use crate::functions::geometric_operations::intersection;
use crate::functions::simple_features::{sf_disjoint, sf_equals};
use crate::geometry::Geometry;
use geo::{Contains, Intersects};

/// RCC8 Equals: regions are identical
///
/// Two regions are equal if they occupy exactly the same space.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_eq;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
///
/// assert!(rcc8_eq(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_eq(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // RCC8 equals is the same as Simple Features equals
    sf_equals(geom1, geom2)
}

/// RCC8 Disconnected: regions don't touch
///
/// Two regions are disconnected if they have no points in common.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_dc;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((5 5, 7 5, 7 7, 5 7, 5 5))").expect("should succeed");
///
/// assert!(rcc8_dc(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_dc(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // Disconnected is the same as disjoint
    sf_disjoint(geom1, geom2)
}

/// RCC8 Externally Connected: regions touch only at their boundaries
///
/// Two regions are externally connected if they share boundary points but their interiors don't overlap.
///
/// # Performance
///
/// Uses bbox pre-filtering. If bounding boxes are disjoint, returns false immediately
/// without expensive boundary calculations.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_ec;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))").expect("should succeed");
///
/// // These regions share only the boundary edge x=2.
/// assert!(rcc8_ec(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_ec(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, regions can't be externally connected.
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Boundaries must touch.
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_touch = b1.geom.intersects(&b2.geom);

    // Interiors must not overlap.
    let interiors_disjoint = match &intersection(geom1, geom2)? {
        None => true,
        Some(i) => i.is_empty(),
    };

    Ok(boundaries_touch && interiors_disjoint)
}

/// RCC8 Partially Overlapping: regions overlap but neither contains the other
///
/// Two regions partially overlap if their interiors intersect and neither is contained in the other.
///
/// # Performance
///
/// Uses bbox pre-filtering. If bounding boxes are disjoint, returns false immediately.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_po;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))").expect("should succeed");
///
/// assert!(rcc8_po(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_po(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, regions can't overlap
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Regions must intersect
    let intersects = geom1.geom.intersects(&geom2.geom);

    // Neither contains the other
    let not_contains = !geom1.geom.contains(&geom2.geom) && !geom2.geom.contains(&geom1.geom);

    Ok(intersects && not_contains)
}

/// RCC8 Tangential Proper Part: first region is a proper part of second, touching at boundary
///
/// geom1 is a tangential proper part of geom2 if geom1 is contained in geom2 and their boundaries touch.
///
/// # Performance
///
/// Uses bbox pre-filtering. If geom1's bbox is not within geom2's bbox, returns false immediately
/// without expensive boundary calculations.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_tpp;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
///
/// // poly1 is a proper part of poly2 and their boundaries touch.
/// assert!(rcc8_tpp(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_tpp(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if geom1's bbox is not within geom2's bbox, it can't be a proper part.
    if !bbox_could_contain(geom2, geom1) {
        return Ok(false);
    }

    // geom1 must be contained in geom2.
    let contained = geom2.geom.contains(&geom1.geom);

    // Boundaries must touch.
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_touch = b1.geom.intersects(&b2.geom);

    // Must be a proper part (not equal).
    let not_equal = !sf_equals(geom1, geom2)?;

    Ok(contained && boundaries_touch && not_equal)
}

/// RCC8 Tangential Proper Part Inverse: second region is a tangential proper part of first
///
/// geom2 is a tangential proper part of geom1 (inverse of TPP).
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_tppi;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
///
/// // poly2 is a tangential proper part of poly1 (the inverse relation).
/// assert!(rcc8_tppi(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_tppi(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    rcc8_tpp(geom2, geom1)
}

/// RCC8 Non-Tangential Proper Part: first region is inside second, not touching boundary
///
/// geom1 is a non-tangential proper part of geom2 if geom1 is contained in geom2's interior.
///
/// # Performance
///
/// Uses bbox pre-filtering. If geom1's bbox is not within geom2's bbox, returns false immediately
/// without expensive boundary calculations.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_ntpp;
///
/// let poly1 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
///
/// // poly1 lies entirely within poly2's interior, not touching its boundary.
/// assert!(rcc8_ntpp(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_ntpp(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if geom1's bbox is not within geom2's bbox, it can't be a proper part.
    if !bbox_could_contain(geom2, geom1) {
        return Ok(false);
    }

    // geom1 must be contained in geom2.
    let contained = geom2.geom.contains(&geom1.geom);

    // Boundaries must NOT touch.
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_disjoint = !b1.geom.intersects(&b2.geom);

    // Must be a proper part (not equal).
    let not_equal = !sf_equals(geom1, geom2)?;

    Ok(contained && boundaries_disjoint && not_equal)
}

/// RCC8 Non-Tangential Proper Part Inverse: second region is inside first, not touching boundary
///
/// geom2 is a non-tangential proper part of geom1 (inverse of NTPP).
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::rcc8::rcc8_ntppi;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
/// let poly2 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").expect("should succeed");
///
/// // poly2 is a non-tangential proper part of poly1 (the inverse relation).
/// assert!(rcc8_ntppi(&poly1, &poly2).expect("should succeed"));
/// ```
pub fn rcc8_ntppi(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    rcc8_ntpp(geom2, geom1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Polygon};

    #[test]
    fn test_rcc8_eq() {
        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = poly1.clone();
        assert!(rcc8_eq(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_rcc8_dc() {
        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 2.0, y: 0.0 },
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 0.0, y: 2.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 5.0, y: 5.0 },
                Coord { x: 7.0, y: 5.0 },
                Coord { x: 7.0, y: 7.0 },
                Coord { x: 5.0, y: 7.0 },
                Coord { x: 5.0, y: 5.0 },
            ]),
            vec![],
        )));

        assert!(rcc8_dc(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_rcc8_po() {
        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 3.0, y: 0.0 },
                Coord { x: 3.0, y: 3.0 },
                Coord { x: 0.0, y: 3.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 5.0, y: 2.0 },
                Coord { x: 5.0, y: 5.0 },
                Coord { x: 2.0, y: 5.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        )));

        assert!(rcc8_po(&poly1, &poly2).expect("should succeed"));
    }
}

/// Boundary-dependent relations. These used to require the GEOS C library and
/// so were only exercised in the (now deleted) `oxirs-geosparql-adapter-geos`
/// crate; they run on `geo`'s Pure-Rust boundary here.
#[cfg(test)]
mod tests_boundary_relations {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_rcc8_ec_shared_edge() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("valid");
        let poly2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))").expect("valid");
        assert!(rcc8_ec(&poly1, &poly2).expect("rcc8_ec should succeed"));
    }

    #[test]
    fn test_rcc8_ec_disjoint_is_false() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)));
        assert!(!rcc8_ec(&p1, &p2).expect("rcc8_ec should succeed"));
    }

    #[test]
    fn test_rcc8_tpp_and_tppi() {
        let inner = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("valid");
        let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        assert!(rcc8_tpp(&inner, &outer).expect("rcc8_tpp should succeed"));
        assert!(rcc8_tppi(&outer, &inner).expect("rcc8_tppi should succeed"));
    }

    #[test]
    fn test_rcc8_ntpp_and_ntppi() {
        let inner = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").expect("valid");
        let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        assert!(rcc8_ntpp(&inner, &outer).expect("rcc8_ntpp should succeed"));
        assert!(rcc8_ntppi(&outer, &inner).expect("rcc8_ntppi should succeed"));
    }

    #[test]
    fn test_tangential_and_non_tangential_are_exclusive() {
        // A shared boundary edge makes this tangential, so NTPP must be false.
        let inner = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("valid");
        let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        assert!(!rcc8_ntpp(&inner, &outer).expect("rcc8_ntpp should succeed"));
    }
}
