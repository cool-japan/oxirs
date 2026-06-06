//! Set-theoretic geometric operations: intersection, union, difference,
//! symmetric difference, convex hull, and envelope.

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo_types::Geometry as GeoGeometry;

/// Calculate the convex hull of a geometry
pub fn convex_hull(geom: &Geometry) -> Result<Geometry> {
    use geo::ConvexHull;

    let hull = match &geom.geom {
        GeoGeometry::MultiPoint(mp) => {
            let hull_poly = mp.convex_hull();
            GeoGeometry::Polygon(hull_poly)
        }
        GeoGeometry::LineString(ls) => {
            let hull_poly = ls.convex_hull();
            GeoGeometry::Polygon(hull_poly)
        }
        GeoGeometry::Polygon(p) => {
            let hull_poly = p.convex_hull();
            GeoGeometry::Polygon(hull_poly)
        }
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "Convex hull not supported for this geometry type".to_string(),
            ))
        }
    };

    Ok(Geometry::with_crs(hull, geom.crs.clone()))
}

/// Calculate the intersection of two geometries
///
/// Returns the set of points common to both geometries.
/// For geometries that don't intersect, returns None.
pub fn intersection(geom1: &Geometry, geom2: &Geometry) -> Result<Option<Geometry>> {
    use geo::BooleanOps;
    use geo::Intersects;

    geom1.validate_crs_compatibility(geom2)?;

    // Check if geometries intersect first
    if !geom1.geom.intersects(&geom2.geom) {
        return Ok(None);
    }

    // Perform intersection based on geometry types
    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon intersection
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let intersection = p1.intersection(p2);
            GeoGeometry::MultiPolygon(intersection)
        }
        // MultiPolygon-Polygon intersection
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p))
        | (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let intersection = mp.intersection(&p_multi);
            GeoGeometry::MultiPolygon(intersection)
        }
        // MultiPolygon-MultiPolygon intersection
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let intersection = mp1.intersection(mp2);
            GeoGeometry::MultiPolygon(intersection)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                format!(
                    "Intersection not supported for {} and {} (only Polygon and MultiPolygon are supported)",
                    geom1.geometry_type(),
                    geom2.geometry_type()
                ),
            ))
        }
    };

    Ok(Some(Geometry::with_crs(result, geom1.crs.clone())))
}

/// Calculate the union of two geometries
///
/// Returns a geometry representing all points in either geometry.
pub fn union(geom1: &Geometry, geom2: &Geometry) -> Result<Geometry> {
    use geo::BooleanOps;

    geom1.validate_crs_compatibility(geom2)?;

    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon union
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let union = p1.union(p2);
            GeoGeometry::MultiPolygon(union)
        }
        // MultiPolygon-Polygon union
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p))
        | (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let union = mp.union(&p_multi);
            GeoGeometry::MultiPolygon(union)
        }
        // MultiPolygon-MultiPolygon union
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let union = mp1.union(mp2);
            GeoGeometry::MultiPolygon(union)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(format!(
                "Union not supported for {} and {} (only Polygon and MultiPolygon are supported)",
                geom1.geometry_type(),
                geom2.geometry_type()
            )))
        }
    };

    Ok(Geometry::with_crs(result, geom1.crs.clone()))
}

/// Calculate the difference of two geometries (geom1 - geom2)
///
/// Returns the set of points in geom1 that are not in geom2.
pub fn difference(geom1: &Geometry, geom2: &Geometry) -> Result<Geometry> {
    use geo::BooleanOps;

    geom1.validate_crs_compatibility(geom2)?;

    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon difference
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let diff = p1.difference(p2);
            GeoGeometry::MultiPolygon(diff)
        }
        // MultiPolygon-Polygon difference
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let diff = mp.difference(&p_multi);
            GeoGeometry::MultiPolygon(diff)
        }
        // Polygon-MultiPolygon difference
        (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let diff = p_multi.difference(mp);
            GeoGeometry::MultiPolygon(diff)
        }
        // MultiPolygon-MultiPolygon difference
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let diff = mp1.difference(mp2);
            GeoGeometry::MultiPolygon(diff)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(format!(
            "Difference not supported for {} and {} (only Polygon and MultiPolygon are supported)",
            geom1.geometry_type(),
            geom2.geometry_type()
        )))
        }
    };

    Ok(Geometry::with_crs(result, geom1.crs.clone()))
}

/// Calculate the symmetric difference of two geometries
///
/// Returns the set of points that are in either geometry but not in both.
/// Also known as XOR operation.
pub fn sym_difference(geom1: &Geometry, geom2: &Geometry) -> Result<Geometry> {
    use geo::BooleanOps;

    geom1.validate_crs_compatibility(geom2)?;

    // Note: BooleanOps trait only supports Polygon and MultiPolygon
    let result = match (&geom1.geom, &geom2.geom) {
        // Polygon-Polygon symmetric difference
        (GeoGeometry::Polygon(p1), GeoGeometry::Polygon(p2)) => {
            let xor = p1.xor(p2);
            GeoGeometry::MultiPolygon(xor)
        }
        // MultiPolygon-Polygon symmetric difference
        (GeoGeometry::MultiPolygon(mp), GeoGeometry::Polygon(p))
        | (GeoGeometry::Polygon(p), GeoGeometry::MultiPolygon(mp)) => {
            let p_multi = geo_types::MultiPolygon(vec![p.clone()]);
            let xor = mp.xor(&p_multi);
            GeoGeometry::MultiPolygon(xor)
        }
        // MultiPolygon-MultiPolygon symmetric difference
        (GeoGeometry::MultiPolygon(mp1), GeoGeometry::MultiPolygon(mp2)) => {
            let xor = mp1.xor(mp2);
            GeoGeometry::MultiPolygon(xor)
        }
        // Other combinations not yet supported
        _ => {
            return Err(GeoSparqlError::UnsupportedOperation(
                format!(
                    "Symmetric difference not supported for {} and {} (only Polygon and MultiPolygon are supported)",
                    geom1.geometry_type(),
                    geom2.geometry_type()
                ),
            ))
        }
    };

    Ok(Geometry::with_crs(result, geom1.crs.clone()))
}

/// Calculate the envelope (bounding box) of a geometry
pub fn envelope(geom: &Geometry) -> Result<Geometry> {
    use geo::BoundingRect;

    let bbox = match &geom.geom {
        GeoGeometry::Point(p) => geo_types::Rect::new(
            geo_types::coord! { x: p.x(), y: p.y() },
            geo_types::coord! { x: p.x(), y: p.y() },
        ),
        GeoGeometry::LineString(ls) => ls.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::Polygon(p) => p.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::MultiPoint(mp) => mp.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::MultiLineString(mls) => mls.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::MultiPolygon(mp) => mp.bounding_rect().ok_or_else(|| {
            GeoSparqlError::GeometryOperationFailed("Could not calculate envelope".to_string())
        })?,
        GeoGeometry::Triangle(t) => t.bounding_rect(),
        GeoGeometry::Rect(r) => *r,
        _ => {
            return Err(GeoSparqlError::GeometryOperationFailed(
                "Could not calculate envelope".to_string(),
            ))
        }
    };

    Ok(Geometry::with_crs(
        GeoGeometry::Rect(bbox),
        geom.crs.clone(),
    ))
}
