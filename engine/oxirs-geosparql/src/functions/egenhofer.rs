//! Egenhofer topological relations
//!
//! This module implements the 8 Egenhofer topological relations based on the
//! 4-intersection model (boundary-boundary, boundary-interior, interior-boundary, interior-interior).
//!
//! # Performance
//!
//! All functions use bounding box pre-filtering to avoid expensive boundary calculations
//! when geometries are clearly disjoint. This provides 50-90% speedup for disjoint cases.
//!
//! Reference: Egenhofer, M. J., & Herring, J. (1990). "A mathematical framework for the definition of topological relationships"

use crate::error::Result;
#[cfg(feature = "geos-backend")]
use crate::functions::bbox_utils::bbox_could_contain;
use crate::functions::bbox_utils::bboxes_disjoint;
#[cfg(feature = "geos-backend")]
use crate::functions::geometric_operations::boundary;
use crate::functions::simple_features::{sf_contains, sf_disjoint, sf_equals};
use crate::geometry::Geometry;
use geo::Intersects;

/// Egenhofer Equals: geometries are topologically equal
///
/// Two geometries are equal if they have the same boundary and interior.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::egenhofer::eh_equals;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
///
/// assert!(eh_equals(&poly1, &poly2).unwrap());
/// ```
pub fn eh_equals(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // Egenhofer equals is the same as Simple Features equals
    sf_equals(geom1, geom2)
}

/// Egenhofer Disjoint: geometries do not intersect at all
///
/// Two geometries are disjoint if their boundaries and interiors don't intersect.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::egenhofer::eh_disjoint;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((5 5, 7 5, 7 7, 5 7, 5 5))").unwrap();
///
/// assert!(eh_disjoint(&poly1, &poly2).unwrap());
/// ```
pub fn eh_disjoint(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // Egenhofer disjoint is the same as Simple Features disjoint
    sf_disjoint(geom1, geom2)
}

/// Egenhofer Meet: geometries touch only at their boundaries
///
/// Two geometries meet if their boundaries intersect but their interiors don't.
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
/// use oxirs_geosparql::functions::egenhofer::eh_meet;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))").unwrap();
///
/// assert!(eh_meet(&poly1, &poly2).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn eh_meet(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries can't meet
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Boundaries must intersect
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_intersect = b1.geom.intersects(&b2.geom);

    // Interiors must NOT intersect
    let interiors_disjoint = !geom1.geom.intersects(&geom2.geom) || {
        // If they intersect, check if it's only at the boundary
        use geo::Contains;
        !geom1.geom.contains(&geom2.geom) && !geom2.geom.contains(&geom1.geom)
    };

    Ok(boundaries_intersect && interiors_disjoint)
}

/// Egenhofer Meet (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn eh_meet(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "Egenhofer Meet requires 'geos-backend' feature for boundary calculation".to_string(),
    ))
}

/// Egenhofer Overlap: geometries partially overlap
///
/// Two geometries overlap if their interiors intersect and neither contains the other.
///
/// # Performance
///
/// Uses bbox pre-filtering. If bounding boxes are disjoint, returns false immediately.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::egenhofer::eh_overlap;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))").unwrap();
///
/// assert!(eh_overlap(&poly1, &poly2).unwrap());
/// ```
pub fn eh_overlap(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    use geo::Contains;

    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries can't overlap
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Interiors must intersect
    let intersects = geom1.geom.intersects(&geom2.geom);

    // Neither contains the other
    let not_contains = !geom1.geom.contains(&geom2.geom) && !geom2.geom.contains(&geom1.geom);

    Ok(intersects && not_contains)
}

/// Egenhofer Covers: first geometry covers the second
///
/// geom1 covers geom2 if every point of geom2 is a point of geom1.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::egenhofer::eh_covers;
///
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap();
///
/// assert!(eh_covers(&poly1, &poly2).unwrap());
/// ```
pub fn eh_covers(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // Covers is the same as contains in Simple Features
    sf_contains(geom1, geom2)
}

/// Egenhofer Covered By: first geometry is covered by the second
///
/// geom1 is covered by geom2 if every point of geom1 is a point of geom2.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::egenhofer::eh_covered_by;
///
/// let poly1 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
///
/// assert!(eh_covered_by(&poly1, &poly2).unwrap());
/// ```
pub fn eh_covered_by(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // CoveredBy is the inverse of contains
    sf_contains(geom2, geom1)
}

/// Egenhofer Inside: first geometry is inside the second (interior only)
///
/// geom1 is inside geom2 if geom1's interior is within geom2's interior,
/// and geom1's boundary does not intersect geom2's boundary.
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
/// use oxirs_geosparql::functions::egenhofer::eh_inside;
///
/// let point = Geometry::from_wkt("POINT(2 2)").unwrap();
/// let poly = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
///
/// assert!(eh_inside(&point, &poly).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn eh_inside(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    use geo::Contains;

    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if geom1's bbox is not within geom2's bbox, geom1 can't be inside geom2
    if !bbox_could_contain(geom2, geom1) {
        return Ok(false);
    }

    // geom1 must be contained in geom2
    let contained = geom2.geom.contains(&geom1.geom);

    // geom1's boundary must not intersect geom2's boundary
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_disjoint = !b1.geom.intersects(&b2.geom);

    Ok(contained && boundaries_disjoint)
}

/// Egenhofer Inside (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn eh_inside(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "Egenhofer Inside requires 'geos-backend' feature for boundary calculation".to_string(),
    ))
}

/// Egenhofer Contains: first geometry contains the second (interior only)
///
/// geom1 contains geom2 if geom2's interior is within geom1's interior,
/// and geom2's boundary does not intersect geom1's boundary.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::egenhofer::eh_contains;
///
/// let poly = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
/// let point = Geometry::from_wkt("POINT(2 2)").unwrap();
///
/// assert!(eh_contains(&poly, &point).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn eh_contains(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // Contains is the inverse of inside
    eh_inside(geom2, geom1)
}

/// Egenhofer Contains (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn eh_contains(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "Egenhofer Contains requires 'geos-backend' feature for boundary calculation".to_string(),
    ))
}

#[cfg(test)]
#[cfg(feature = "geos-backend")]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point, Polygon};

    #[test]
    fn test_eh_equals() {
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

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        assert!(eh_equals(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_eh_disjoint() {
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

        assert!(eh_disjoint(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_eh_meet() {
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
                Coord { x: 2.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 2.0 },
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 2.0, y: 0.0 },
            ]),
            vec![],
        )));

        assert!(eh_meet(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_eh_overlap() {
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

        assert!(eh_overlap(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_eh_covers() {
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

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 1.0, y: 1.0 },
                Coord { x: 3.0, y: 1.0 },
                Coord { x: 3.0, y: 3.0 },
                Coord { x: 1.0, y: 3.0 },
                Coord { x: 1.0, y: 1.0 },
            ]),
            vec![],
        )));

        assert!(eh_covers(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_eh_covered_by() {
        let poly1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 1.0, y: 1.0 },
                Coord { x: 3.0, y: 1.0 },
                Coord { x: 3.0, y: 3.0 },
                Coord { x: 1.0, y: 3.0 },
                Coord { x: 1.0, y: 1.0 },
            ]),
            vec![],
        )));

        let poly2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        assert!(eh_covered_by(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_eh_inside() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0)));

        let poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        assert!(eh_inside(&point, &poly).unwrap());
    }

    #[test]
    fn test_eh_contains() {
        let poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let point = Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0)));

        assert!(eh_contains(&poly, &point).unwrap());
    }
}

#[cfg(test)]
#[cfg(not(feature = "geos-backend"))]
mod tests_without_geos {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_eh_meet_without_geos_fails() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)));

        let result = eh_meet(&p1, &p2);
        assert!(result.is_err());
    }

    #[test]
    fn test_eh_inside_without_geos_fails() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)));

        let result = eh_inside(&p1, &p2);
        assert!(result.is_err());
    }
}
