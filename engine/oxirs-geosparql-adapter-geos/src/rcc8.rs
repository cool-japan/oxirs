//! GEOS-backed RCC8 boundary-dependent relations.
//!
//! `EC`, `TPP`, `TPPi`, `NTPP` and `NTPPi` need a true geometric boundary, which the
//! parent only computes via GEOS. The other three RCC8 relations
//! (`EQ`/`DC`/`PO`) are Pure Rust and remain in
//! [`oxirs_geosparql::functions::rcc8`].

use crate::buffer::boundary;
use geo::{Contains, Intersects};
use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::bbox_utils::{bbox_could_contain, bboxes_disjoint};
use oxirs_geosparql::functions::geometric_operations::intersection;
use oxirs_geosparql::functions::simple_features::sf_equals;
use oxirs_geosparql::geometry::Geometry;

/// RCC8 Externally Connected: regions touch only at their boundaries.
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
    let int = intersection(geom1, geom2)?;
    let interiors_disjoint = match &int {
        None => true,
        Some(i) => i.is_empty(),
    };

    Ok(boundaries_touch && interiors_disjoint)
}

/// RCC8 Tangential Proper Part: geom1 is a proper part of geom2, touching at boundary.
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

/// RCC8 Tangential Proper Part Inverse: geom2 is a tangential proper part of geom1.
pub fn rcc8_tppi(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    rcc8_tpp(geom2, geom1)
}

/// RCC8 Non-Tangential Proper Part: geom1 is inside geom2's interior, not touching boundary.
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

/// RCC8 Non-Tangential Proper Part Inverse: geom2 is inside geom1's interior, not touching boundary.
pub fn rcc8_ntppi(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    rcc8_ntpp(geom2, geom1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rcc8_ec() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("valid");
        let poly2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))").expect("valid");
        assert!(rcc8_ec(&poly1, &poly2).expect("geos rcc8_ec should succeed"));
    }

    #[test]
    fn test_rcc8_tpp() {
        let inner = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("valid");
        let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        assert!(rcc8_tpp(&inner, &outer).expect("geos rcc8_tpp should succeed"));
    }

    #[test]
    fn test_rcc8_tppi() {
        let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        let inner = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("valid");
        assert!(rcc8_tppi(&outer, &inner).expect("geos rcc8_tppi should succeed"));
    }

    #[test]
    fn test_rcc8_ntpp() {
        let inner = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").expect("valid");
        let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        assert!(rcc8_ntpp(&inner, &outer).expect("geos rcc8_ntpp should succeed"));
    }

    #[test]
    fn test_rcc8_ntppi() {
        let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        let inner = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").expect("valid");
        assert!(rcc8_ntppi(&outer, &inner).expect("geos rcc8_ntppi should succeed"));
    }
}
