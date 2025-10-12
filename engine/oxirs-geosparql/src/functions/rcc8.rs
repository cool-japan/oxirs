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
#[cfg(feature = "geos-backend")]
use crate::functions::bbox_utils::bbox_could_contain;
use crate::functions::bbox_utils::bboxes_disjoint;
#[cfg(feature = "geos-backend")]
use crate::functions::geometric_operations::{boundary, intersection};
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
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
///
/// assert!(rcc8_eq(&poly1, &poly2).unwrap());
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
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((5 5, 7 5, 7 7, 5 7, 5 5))").unwrap();
///
/// assert!(rcc8_dc(&poly1, &poly2).unwrap());
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
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))").unwrap();
///
/// assert!(rcc8_ec(&poly1, &poly2).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn rcc8_ec(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, regions can't be externally connected
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Boundaries must touch
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_touch = b1.geom.intersects(&b2.geom);

    // Interiors must not overlap
    let int = intersection(geom1, geom2)?;
    let interiors_disjoint = int.is_none() || {
        if let Some(i) = int {
            i.is_empty()
        } else {
            true
        }
    };

    Ok(boundaries_touch && interiors_disjoint)
}

/// RCC8 Externally Connected (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn rcc8_ec(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "RCC8 Externally Connected requires 'geos-backend' feature for boundary calculation"
            .to_string(),
    ))
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
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))").unwrap();
///
/// assert!(rcc8_po(&poly1, &poly2).unwrap());
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
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
///
/// assert!(rcc8_tpp(&poly1, &poly2).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn rcc8_tpp(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if geom1's bbox is not within geom2's bbox, geom1 can't be a proper part
    if !bbox_could_contain(geom2, geom1) {
        return Ok(false);
    }

    // geom1 must be contained in geom2
    let contained = geom2.geom.contains(&geom1.geom);

    // Boundaries must touch
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_touch = b1.geom.intersects(&b2.geom);

    // Must be a proper part (not equal)
    let not_equal = !sf_equals(geom1, geom2)?;

    Ok(contained && boundaries_touch && not_equal)
}

/// RCC8 Tangential Proper Part (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn rcc8_tpp(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "RCC8 TPP requires 'geos-backend' feature for boundary calculation".to_string(),
    ))
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
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
///
/// assert!(rcc8_tppi(&poly1, &poly2).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn rcc8_tppi(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // TPPI is the inverse of TPP
    rcc8_tpp(geom2, geom1)
}

/// RCC8 Tangential Proper Part Inverse (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn rcc8_tppi(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "RCC8 TPPI requires 'geos-backend' feature for boundary calculation".to_string(),
    ))
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
/// let poly1 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
///
/// assert!(rcc8_ntpp(&poly1, &poly2).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn rcc8_ntpp(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if geom1's bbox is not within geom2's bbox, geom1 can't be a proper part
    if !bbox_could_contain(geom2, geom1) {
        return Ok(false);
    }

    // geom1 must be contained in geom2
    let contained = geom2.geom.contains(&geom1.geom);

    // Boundaries must NOT touch
    let b1 = boundary(geom1)?;
    let b2 = boundary(geom2)?;
    let boundaries_disjoint = !b1.geom.intersects(&b2.geom);

    // Must be a proper part (not equal)
    let not_equal = !sf_equals(geom1, geom2)?;

    Ok(contained && boundaries_disjoint && not_equal)
}

/// RCC8 Non-Tangential Proper Part (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn rcc8_ntpp(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "RCC8 NTPP requires 'geos-backend' feature for boundary calculation".to_string(),
    ))
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
/// let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
/// let poly2 = Geometry::from_wkt("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))").unwrap();
///
/// assert!(rcc8_ntppi(&poly1, &poly2).unwrap());
/// ```
#[cfg(feature = "geos-backend")]
pub fn rcc8_ntppi(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // NTPPI is the inverse of NTPP
    rcc8_ntpp(geom2, geom1)
}

/// RCC8 Non-Tangential Proper Part Inverse (fallback without GEOS)
#[cfg(not(feature = "geos-backend"))]
pub fn rcc8_ntppi(_geom1: &Geometry, _geom2: &Geometry) -> Result<bool> {
    Err(crate::error::GeoSparqlError::UnsupportedOperation(
        "RCC8 NTPPI requires 'geos-backend' feature for boundary calculation".to_string(),
    ))
}

#[cfg(test)]
#[cfg(feature = "geos-backend")]
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
        assert!(rcc8_eq(&poly1, &poly2).unwrap());
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

        assert!(rcc8_dc(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_rcc8_ec() {
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

        assert!(rcc8_ec(&poly1, &poly2).unwrap());
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

        assert!(rcc8_po(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_rcc8_tpp() {
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
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        assert!(rcc8_tpp(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_rcc8_tppi() {
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
                Coord { x: 2.0, y: 0.0 },
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 0.0, y: 2.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        assert!(rcc8_tppi(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_rcc8_ntpp() {
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

        assert!(rcc8_ntpp(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_rcc8_ntppi() {
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

        assert!(rcc8_ntppi(&poly1, &poly2).unwrap());
    }
}

#[cfg(test)]
#[cfg(not(feature = "geos-backend"))]
mod tests_without_geos {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_rcc8_ec_without_geos_fails() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)));

        let result = rcc8_ec(&p1, &p2);
        assert!(result.is_err());
    }
}
