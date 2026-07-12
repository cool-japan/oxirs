//! GEOS-backed Egenhofer boundary-dependent relations.
//!
//! `ehMeet`, `ehInside` and `ehContains` need a true geometric boundary, which the
//! parent only computes via GEOS. The other five Egenhofer relations
//! (`ehEquals`/`ehDisjoint`/`ehOverlap`/`ehCovers`/`ehCoveredBy`) are Pure Rust and
//! remain in [`oxirs_geosparql::functions::egenhofer`].

use crate::buffer::boundary;
use geo::{Contains, Intersects};
use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::bbox_utils::{bbox_could_contain, bboxes_disjoint};
use oxirs_geosparql::geometry::Geometry;

/// Egenhofer Meet: geometries touch only at their boundaries.
///
/// Two geometries meet if their boundaries intersect but their interiors don't.
pub fn eh_meet(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries can't meet.
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Boundaries must intersect.
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_intersect = b1.geom.intersects(&b2.geom);

    // Interiors must NOT intersect.
    let interiors_disjoint = !geom1.geom.intersects(&geom2.geom)
        || (!geom1.geom.contains(&geom2.geom) && !geom2.geom.contains(&geom1.geom));

    Ok(boundaries_intersect && interiors_disjoint)
}

/// Egenhofer Inside: first geometry is inside the second (interior only).
///
/// geom1 is inside geom2 if geom1 is contained in geom2 and their boundaries do
/// not intersect.
pub fn eh_inside(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if geom1's bbox is not within geom2's bbox, it can't be inside.
    if !bbox_could_contain(geom2, geom1) {
        return Ok(false);
    }

    // geom1 must be contained in geom2.
    let contained = geom2.geom.contains(&geom1.geom);

    // geom1's boundary must not intersect geom2's boundary.
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_disjoint = !b1.geom.intersects(&b2.geom);

    Ok(contained && boundaries_disjoint)
}

/// Egenhofer Contains: first geometry contains the second (interior only).
///
/// Contains is the inverse of inside.
pub fn eh_contains(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    eh_inside(geom2, geom1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eh_meet() {
        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("valid");
        let poly2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))").expect("valid");
        assert!(eh_meet(&poly1, &poly2).expect("geos eh_meet should succeed"));
    }

    #[test]
    fn test_eh_inside() {
        let point = Geometry::from_wkt("POINT(2 2)").expect("valid");
        let poly = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        assert!(eh_inside(&point, &poly).expect("geos eh_inside should succeed"));
    }

    #[test]
    fn test_eh_contains() {
        let poly = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("valid");
        let point = Geometry::from_wkt("POINT(2 2)").expect("valid");
        assert!(eh_contains(&poly, &point).expect("geos eh_contains should succeed"));
    }
}
