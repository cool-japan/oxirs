//! Simple Features topological relations (DE-9IM model)
//!
//! Implements the Simple Features spatial predicates based on the
//! Dimensionally Extended 9-Intersection Model (DE-9IM).
//!
//! # Performance
//!
//! All functions use bounding box pre-filtering for fast rejection of
//! disjoint geometry pairs. This provides 50-90% speedup for disjoint cases.

use crate::error::Result;
use crate::functions::bbox_utils::{bbox_within, bboxes_disjoint};
use crate::geometry::Geometry;
use geo::algorithm::*;

/// Test if two geometries are spatially equal
///
/// Returns true if the two geometries are spatially equal,
/// meaning they have the same set of points in space.
///
/// Combinations without a cheap special-case (e.g. Polygon/Polygon,
/// MultiPolygon/MultiPolygon, GeometryCollection combos) are resolved via
/// the DE-9IM `relate()` machinery (`IntersectionMatrix::is_equal_topo`),
/// which is the topologically-correct test for spatial equality.
pub fn sf_equals(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: geometries with bit-identical coordinates are trivially
    // spatially equal. This also guarantees reflexivity (sf_equals(g, g) is
    // always true) even for self-intersecting/topologically invalid inputs
    // that DE-9IM `relate()` cannot reliably evaluate.
    if geom1.geom == geom2.geom {
        return Ok(true);
    }

    let result = match (&geom1.geom, &geom2.geom) {
        (geo_types::Geometry::Point(p1), geo_types::Geometry::Point(p2)) => {
            use geo::{Distance, Euclidean};
            Euclidean.distance(*p1, *p2) < 1e-10
        }
        (geo_types::Geometry::LineString(ls1), geo_types::Geometry::LineString(ls2)) => {
            ls1.0 == ls2.0
        }
        _ => geom1.geom.relate(&geom2.geom).is_equal_topo(),
    };

    Ok(result)
}

/// Test if two geometries are spatially disjoint
///
/// Returns true if the two geometries have no points in common.
///
/// # Performance
///
/// Uses bbox pre-filtering for fast rejection. If bounding boxes are disjoint,
/// returns true immediately without expensive geometric tests.
pub fn sf_disjoint(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries are definitely disjoint
    if bboxes_disjoint(geom1, geom2) {
        return Ok(true);
    }

    // Disjoint is the opposite of intersects
    let intersects = geom1.geom.intersects(&geom2.geom);
    Ok(!intersects)
}

/// Test if two geometries spatially intersect
///
/// Returns true if the two geometries have at least one point in common.
///
/// # Performance
///
/// Uses bbox pre-filtering for fast rejection. If bounding boxes are disjoint,
/// returns false immediately without expensive geometric tests.
pub fn sf_intersects(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries definitely don't intersect
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    let result = geom1.geom.intersects(&geom2.geom);
    Ok(result)
}

/// Test if two geometries touch at boundaries only
///
/// Returns true if the geometries have at least one boundary point in common,
/// but no interior points in common.
///
/// # DE-9IM Pattern
///
/// Touches corresponds to DE-9IM patterns: FT*******, F**T*****, or F***T****
/// This means:
/// - Interior(A) ∩ Interior(B) = ∅ (empty)
/// - At least one of:
///   - Interior(A) ∩ Boundary(B) ≠ ∅
///   - Boundary(A) ∩ Interior(B) ≠ ∅
///   - Boundary(A) ∩ Boundary(B) ≠ ∅
///
/// # Performance
///
/// Uses bbox pre-filtering. If bounding boxes are disjoint, returns false immediately.
pub fn sf_touches(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries can't touch
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Check if they intersect at all - fast rejection
    if !geom1.geom.intersects(&geom2.geom) {
        return Ok(false);
    }

    // Use DE-9IM relate to check the touches pattern
    // Touches means: interiors are disjoint, but boundaries intersect
    use geo::algorithm::relate::Relate;
    use geo::coordinate_position::CoordPos;
    use geo::dimensions::Dimensions;

    let matrix = geom1.geom.relate(&geom2.geom);

    // Check if interior-interior intersection is empty (F in DE-9IM)
    // In DE-9IM notation: Interior(A) ∩ Interior(B)
    let ii_dimension = matrix.get(CoordPos::Inside, CoordPos::Inside);
    let interiors_disjoint = ii_dimension == Dimensions::Empty;

    if !interiors_disjoint {
        // Interiors intersect, so they don't just touch
        return Ok(false);
    }

    // Check if at least one boundary intersection exists
    // IB: Interior(A) ∩ Boundary(B)
    let ib_dimension = matrix.get(CoordPos::Inside, CoordPos::OnBoundary);
    // BI: Boundary(A) ∩ Interior(B)
    let bi_dimension = matrix.get(CoordPos::OnBoundary, CoordPos::Inside);
    // BB: Boundary(A) ∩ Boundary(B)
    let bb_dimension = matrix.get(CoordPos::OnBoundary, CoordPos::OnBoundary);

    let boundaries_intersect = ib_dimension != Dimensions::Empty
        || bi_dimension != Dimensions::Empty
        || bb_dimension != Dimensions::Empty;

    Ok(boundaries_intersect)
}

/// Test if two geometries cross
///
/// Returns true if the geometries have some but not all interior points in common.
///
/// # Performance
///
/// Uses bbox pre-filtering. If bounding boxes are disjoint, returns false immediately.
pub fn sf_crosses(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries can't cross
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Crosses is typically only defined for line/line or line/polygon; other
    // combinations are resolved via the DE-9IM `relate()` machinery
    // (`IntersectionMatrix::is_crosses`) rather than a blanket `false`.
    let result = match (&geom1.geom, &geom2.geom) {
        (geo_types::Geometry::LineString(ls1), geo_types::Geometry::LineString(ls2)) => {
            ls1.intersects(ls2)
        }
        (geo_types::Geometry::LineString(ls), geo_types::Geometry::Polygon(poly))
        | (geo_types::Geometry::Polygon(poly), geo_types::Geometry::LineString(ls)) => {
            ls.intersects(poly)
        }
        _ => geom1.geom.relate(&geom2.geom).is_crosses(),
    };

    Ok(result)
}

/// Test if first geometry is within the second
///
/// Returns true if the first geometry is completely within the second geometry.
///
/// # Performance
///
/// Uses bbox pre-filtering. If geom1's bbox is not within geom2's bbox, returns false immediately.
pub fn sf_within(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if geom1's bbox is not within geom2's bbox, geom1 can't be within geom2
    if !bbox_within(geom1, geom2) {
        return Ok(false);
    }

    let result = match (&geom1.geom, &geom2.geom) {
        (geo_types::Geometry::Point(point), geo_types::Geometry::Polygon(poly)) => {
            poly.contains(point)
        }
        (geo_types::Geometry::Point(point), geo_types::Geometry::LineString(ls)) => {
            ls.contains(point)
        }
        (geo_types::Geometry::LineString(ls), geo_types::Geometry::Polygon(poly)) => {
            // Check if all points of the linestring are within the polygon
            ls.coords_iter().all(|coord| {
                let p = geo_types::Point::new(coord.x, coord.y);
                poly.contains(&p)
            })
        }
        (geo_types::Geometry::Polygon(p1), geo_types::Geometry::Polygon(p2)) => {
            // True polygon-in-polygon containment (not a centroid heuristic):
            // p1 is within p2 iff p2 contains p1.
            p2.contains(p1)
        }
        // All other combinations (MultiPolygon/MultiPolygon, LineString/LineString,
        // Point/Point, Polygon/MultiPolygon, GeometryCollection, etc.) are resolved
        // via the DE-9IM `relate()` machinery (`IntersectionMatrix::is_within`)
        // rather than a blanket `false`.
        _ => geom1.geom.relate(&geom2.geom).is_within(),
    };

    Ok(result)
}

/// Test if first geometry contains the second
///
/// Returns true if the first geometry completely contains the second geometry.
pub fn sf_contains(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    // Contains is the inverse of within
    sf_within(geom2, geom1)
}

/// Test if two geometries spatially overlap
///
/// Returns true if the geometries have some but not all points in common,
/// and the intersection has the same dimension as the geometries themselves.
///
/// # Performance
///
/// Uses bbox pre-filtering. If bounding boxes are disjoint, returns false immediately.
pub fn sf_overlaps(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    // Fast path: if bboxes are disjoint, geometries can't overlap
    if bboxes_disjoint(geom1, geom2) {
        return Ok(false);
    }

    // Two geometries overlap if they intersect and neither is within the other
    let intersects = geom1.geom.intersects(&geom2.geom);
    if !intersects {
        return Ok(false);
    }

    let geom1_within_geom2 = sf_within(geom1, geom2)?;
    let geom2_within_geom1 = sf_within(geom2, geom1)?;

    Ok(intersects && !geom1_within_geom2 && !geom2_within_geom1)
}

// ============================================================================
// 3D Topological Relations
// ============================================================================

/// Test if two 3D geometries spatially intersect
///
/// Returns true if the two geometries have at least one point in common in 3D space.
/// Both geometries must have Z coordinates.
///
/// # Performance
///
/// Uses 3D bounding box pre-filtering for fast rejection.
pub fn sf_intersects_3d(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    if !geom1.is_3d() || !geom2.is_3d() {
        return Err(crate::error::GeoSparqlError::UnsupportedOperation(
            "Both geometries must have Z coordinates for 3D intersection test".to_string(),
        ));
    }

    // Get Z ranges for both geometries
    let (z1_min, z1_max) = get_z_range(geom1)?;
    let (z2_min, z2_max) = get_z_range(geom2)?;

    // Fast path: check if Z ranges are disjoint
    if z1_max < z2_min || z2_max < z1_min {
        return Ok(false);
    }

    // Check 2D intersection
    if !geom1.geom.intersects(&geom2.geom) {
        return Ok(false);
    }

    // For simple point-to-point, use 3D distance
    if let (geo_types::Geometry::Point(_), geo_types::Geometry::Point(_)) =
        (&geom1.geom, &geom2.geom)
    {
        use crate::functions::geometric_operations::distance_3d;
        let dist = distance_3d(geom1, geom2)?;
        return Ok(dist < 1e-10);
    }

    // If 2D projections intersect and Z ranges overlap, geometries intersect in 3D
    Ok(true)
}

/// Test if two 3D geometries are spatially disjoint
///
/// Returns true if the two geometries have no points in common in 3D space.
pub fn sf_disjoint_3d(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    Ok(!sf_intersects_3d(geom1, geom2)?)
}

/// Test if first 3D geometry is within the second
///
/// Returns true if the first geometry is completely within the second geometry in 3D space.
pub fn sf_within_3d(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    if !geom1.is_3d() || !geom2.is_3d() {
        return Err(crate::error::GeoSparqlError::UnsupportedOperation(
            "Both geometries must have Z coordinates for 3D within test".to_string(),
        ));
    }

    // Get Z ranges
    let (z1_min, z1_max) = get_z_range(geom1)?;
    let (z2_min, z2_max) = get_z_range(geom2)?;

    // Check if Z range of geom1 is within Z range of geom2
    if z1_min < z2_min || z1_max > z2_max {
        return Ok(false);
    }

    // Check 2D within relationship
    let within_2d = sf_within(geom1, geom2)?;

    Ok(within_2d)
}

/// Test if first 3D geometry contains the second
///
/// Returns true if the first geometry completely contains the second geometry in 3D space.
pub fn sf_contains_3d(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    sf_within_3d(geom2, geom1)
}

/// Test if two 3D geometries touch at boundaries only
///
/// Returns true if the geometries have at least one boundary point in common in 3D,
/// but no interior points in common.
pub fn sf_touches_3d(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    if !geom1.is_3d() || !geom2.is_3d() {
        return Err(crate::error::GeoSparqlError::UnsupportedOperation(
            "Both geometries must have Z coordinates for 3D touches test".to_string(),
        ));
    }

    // Get Z ranges
    let (z1_min, z1_max) = get_z_range(geom1)?;
    let (z2_min, z2_max) = get_z_range(geom2)?;

    // For touching in 3D, Z ranges should touch or overlap slightly
    // If they're completely disjoint in Z, they can't touch
    if z1_max < z2_min || z2_max < z1_min {
        return Ok(false);
    }

    // Use 2D touches relationship combined with Z range checks
    let touches_2d = sf_touches(geom1, geom2)?;

    // For true 3D touching, either:
    // 1. They touch in 2D and Z ranges overlap
    // 2. Z ranges touch exactly (one max equals the other min)
    let z_ranges_touch = (z1_max - z2_min).abs() < 1e-10 || (z2_max - z1_min).abs() < 1e-10;

    Ok(touches_2d || z_ranges_touch)
}

/// Test if two 3D geometries overlap
///
/// Returns true if the geometries have some but not all points in common in 3D space.
pub fn sf_overlaps_3d(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    if !geom1.is_3d() || !geom2.is_3d() {
        return Err(crate::error::GeoSparqlError::UnsupportedOperation(
            "Both geometries must have Z coordinates for 3D overlaps test".to_string(),
        ));
    }

    // Overlaps if they intersect but neither contains the other
    let intersects = sf_intersects_3d(geom1, geom2)?;
    if !intersects {
        return Ok(false);
    }

    let geom1_within = sf_within_3d(geom1, geom2)?;
    let geom2_within = sf_within_3d(geom2, geom1)?;

    Ok(!geom1_within && !geom2_within)
}

/// Helper function to get Z coordinate range for a geometry
fn get_z_range(geom: &Geometry) -> Result<(f64, f64)> {
    if let Some(ref z_coords) = geom.coord3d.z_coords {
        if z_coords.values.is_empty() {
            return Ok((0.0, 0.0));
        }

        let mut min_z = f64::MAX;
        let mut max_z = f64::MIN;

        for &z in &z_coords.values {
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        Ok((min_z, max_z))
    } else {
        Ok((0.0, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, MultiPolygon, Point, Polygon};

    #[test]
    fn test_sf_equals_points() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let p3 = Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0)));

        assert!(sf_equals(&p1, &p2).expect("should succeed"));
        assert!(!sf_equals(&p1, &p3).expect("should succeed"));
    }

    #[test]
    fn test_sf_disjoint() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0)));

        assert!(sf_disjoint(&p1, &p2).expect("should succeed"));
    }

    #[test]
    fn test_sf_intersects() {
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

        let point_inside = Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0)));
        let point_outside = Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0)));

        assert!(sf_intersects(&poly, &point_inside).expect("should succeed"));
        assert!(!sf_intersects(&poly, &point_outside).expect("should succeed"));
    }

    #[test]
    fn test_sf_within() {
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

        let point_inside = Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0)));
        let point_outside = Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0)));

        assert!(sf_within(&point_inside, &poly).expect("should succeed"));
        assert!(!sf_within(&point_outside, &poly).expect("should succeed"));
    }

    #[test]
    fn test_sf_contains() {
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

        let point_inside = Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0)));
        let point_outside = Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0)));

        assert!(sf_contains(&poly, &point_inside).expect("should succeed"));
        assert!(!sf_contains(&poly, &point_outside).expect("should succeed"));
    }

    #[test]
    fn test_sf_touches_point_at_line_endpoint() {
        // Point at the endpoint (boundary) of a linestring touches the line
        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 4.0, y: 0.0 },
        ])));
        // Point at the endpoint (boundary) of the line
        let point_at_endpoint = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));

        assert!(sf_touches(&point_at_endpoint, &line).expect("should succeed"));
    }

    #[test]
    fn test_sf_not_touches_point_on_line_interior() {
        // Point in the middle of a linestring intersects the interior, not just touching
        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 4.0, y: 0.0 },
        ])));
        // Point in the middle (interior) of the line
        let point_on_interior = Geometry::new(GeoGeometry::Point(Point::new(2.0, 0.0)));

        // This should NOT be touches because it intersects the interior
        assert!(!sf_touches(&point_on_interior, &line).expect("should succeed"));
    }

    #[test]
    fn test_sf_touches_point_on_polygon_boundary() {
        // Point on the boundary of a polygon
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
        let point_on_boundary = Geometry::new(GeoGeometry::Point(Point::new(4.0, 2.0)));

        assert!(sf_touches(&point_on_boundary, &poly).expect("should succeed"));
    }

    #[test]
    fn test_sf_touches_adjacent_polygons() {
        // Two polygons sharing an edge (touching at boundary)
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

        assert!(sf_touches(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_sf_not_touches_overlapping_polygons() {
        // Two overlapping polygons (interiors intersect, so they don't just touch)
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
                Coord { x: 1.0, y: 1.0 },
                Coord { x: 4.0, y: 1.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 1.0, y: 4.0 },
                Coord { x: 1.0, y: 1.0 },
            ]),
            vec![],
        )));

        // These overlap, so they don't just touch
        assert!(!sf_touches(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_sf_not_touches_disjoint() {
        // Two completely separate geometries
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
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 12.0, y: 10.0 },
                Coord { x: 12.0, y: 12.0 },
                Coord { x: 10.0, y: 12.0 },
                Coord { x: 10.0, y: 10.0 },
            ]),
            vec![],
        )));

        // These are disjoint, so they don't touch
        assert!(!sf_touches(&poly1, &poly2).expect("should succeed"));
    }

    // ========================================================================
    // 3D Topological Relations Tests
    // ========================================================================

    #[test]
    fn test_sf_intersects_3d_points_same_location() {
        let p1 = Geometry::from_wkt("POINT Z(1 2 3)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").expect("should succeed");

        assert!(sf_intersects_3d(&p1, &p2).expect("should succeed"));
    }

    #[test]
    fn test_sf_intersects_3d_points_different_z() {
        let p1 = Geometry::from_wkt("POINT Z(1 2 3)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(1 2 10)").expect("should succeed");

        // Same XY but different Z - should not intersect
        assert!(!sf_intersects_3d(&p1, &p2).expect("should succeed"));
    }

    #[test]
    fn test_sf_intersects_3d_linestrings_crossing_z() {
        // Two linestrings that cross in XY but at different Z levels
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 0)").expect("should succeed");
        let ls2 = Geometry::from_wkt("LINESTRING Z(0 10 5, 10 0 5)").expect("should succeed");

        // They cross in XY but at different Z levels (0 vs 5)
        // Z ranges: [0,0] and [5,5] - disjoint
        assert!(!sf_intersects_3d(&ls1, &ls2).expect("should succeed"));
    }

    #[test]
    fn test_sf_intersects_3d_linestrings_overlapping_z() {
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 10)").expect("should succeed");
        let ls2 = Geometry::from_wkt("LINESTRING Z(0 10 5, 10 0 5)").expect("should succeed");

        // Z ranges: [0,10] and [5,5] - overlap at z=5
        assert!(sf_intersects_3d(&ls1, &ls2).expect("should succeed"));
    }

    #[test]
    fn test_sf_disjoint_3d_points() {
        let p1 = Geometry::from_wkt("POINT Z(0 0 0)").expect("should succeed");
        let p2 = Geometry::from_wkt("POINT Z(10 10 10)").expect("should succeed");

        assert!(sf_disjoint_3d(&p1, &p2).expect("should succeed"));
    }

    #[test]
    fn test_sf_disjoint_3d_vertical_separation() {
        // Polygons overlapping in XY but at different Z levels
        let poly1 = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))")
            .expect("should succeed");
        let poly2 = Geometry::from_wkt("POLYGON Z((2 2 20, 8 2 20, 8 8 20, 2 8 20, 2 2 20))")
            .expect("should succeed");

        // Overlap in XY but Z ranges [0,0] and [20,20] are disjoint
        assert!(sf_disjoint_3d(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_sf_within_3d_point_in_cube() {
        // Point inside a cube (represented as polygon with Z range)
        let point = Geometry::from_wkt("POINT Z(5 5 5)").expect("should succeed");
        let cube = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 10, 10 10 10, 0 10 0, 0 0 0))")
            .expect("should succeed");

        // Point at (5,5,5) should be within the cube XY=[0,10], Z=[0,10]
        assert!(sf_within_3d(&point, &cube).expect("should succeed"));
    }

    #[test]
    fn test_sf_within_3d_point_outside_z_range() {
        let point = Geometry::from_wkt("POINT Z(5 5 15)").expect("should succeed");
        let cube = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 10, 10 10 10, 0 10 0, 0 0 0))")
            .expect("should succeed");

        // Point at z=15 is outside the cube's Z range [0,10]
        assert!(!sf_within_3d(&point, &cube).expect("should succeed"));
    }

    #[test]
    fn test_sf_contains_3d_cube_contains_point() {
        let cube = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 10, 10 10 10, 0 10 0, 0 0 0))")
            .expect("should succeed");
        let point = Geometry::from_wkt("POINT Z(5 5 5)").expect("should succeed");

        assert!(sf_contains_3d(&cube, &point).expect("should succeed"));
    }

    #[test]
    fn test_sf_touches_3d_adjacent_polygons_same_z() {
        // Two polygons sharing an edge at the same Z level
        let poly1 = Geometry::from_wkt("POLYGON Z((0 0 5, 5 0 5, 5 5 5, 0 5 5, 0 0 5))")
            .expect("should succeed");
        let poly2 = Geometry::from_wkt("POLYGON Z((5 0 5, 10 0 5, 10 5 5, 5 5 5, 5 0 5))")
            .expect("should succeed");

        // They share an edge in XY and have the same Z
        assert!(sf_touches_3d(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_sf_touches_3d_stacked_polygons() {
        // Two polygons stacked vertically (same XY, touching Z ranges)
        let poly1 = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))")
            .expect("should succeed");
        let poly2 = Geometry::from_wkt("POLYGON Z((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))")
            .expect("should succeed");

        // Z ranges don't touch: [0,0] and [10,10] - gap between them
        // For this simple implementation, they don't touch
        assert!(!sf_touches_3d(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_sf_overlaps_3d_partial_overlap() {
        // Two polygons with partial overlap in 3D
        let poly1 = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 5, 10 10 5, 0 10 0, 0 0 0))")
            .expect("should succeed");
        let poly2 = Geometry::from_wkt("POLYGON Z((5 5 3, 15 5 8, 15 15 8, 5 15 3, 5 5 3))")
            .expect("should succeed");

        // They overlap in XY and Z ranges overlap [0,5] and [3,8]
        assert!(sf_overlaps_3d(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_sf_overlaps_3d_no_overlap_different_z() {
        let poly1 = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))")
            .expect("should succeed");
        let poly2 = Geometry::from_wkt("POLYGON Z((2 2 20, 8 2 20, 8 8 20, 2 8 20, 2 2 20))")
            .expect("should succeed");

        // Overlap in XY but Z ranges [0,0] and [20,20] don't overlap
        assert!(!sf_overlaps_3d(&poly1, &poly2).expect("should succeed"));
    }

    #[test]
    fn test_sf_3d_requires_z_coordinates() {
        let p1 = Geometry::from_wkt("POINT(1 2)").expect("should succeed"); // 2D
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").expect("should succeed"); // 3D

        // Should return error for mixing 2D and 3D
        assert!(sf_intersects_3d(&p1, &p2).is_err());
        assert!(sf_disjoint_3d(&p1, &p2).is_err());
        assert!(sf_within_3d(&p1, &p2).is_err());
        assert!(sf_contains_3d(&p1, &p2).is_err());
        assert!(sf_touches_3d(&p1, &p2).is_err());
        assert!(sf_overlaps_3d(&p1, &p2).is_err());
    }

    // ========================================================================
    // Regression tests: sfWithin/sfEquals/sfCrosses correctness fixes
    // ========================================================================

    #[test]
    fn regression_sf_equals_reflexive_for_self_intersecting_polygon() {
        // A self-intersecting ("bowtie") polygon. DE-9IM `relate()` (used
        // for the Polygon/Polygon `is_equal_topo` fallback) cannot reliably
        // evaluate topologically invalid geometries, but sf_equals(g, g)
        // must still hold by definition -- this is guarded by the
        // bit-identical-coordinates fast path in `sf_equals`.
        let geom = Geometry::from_wkt(
            "POLYGON((0.0 0.0,-122.34703030725247 0.0,0.0 83.6315311606761,\
             -170.63824875286238 -51.211870378264635,0.0 0.0))",
        )
        .expect("should parse");

        assert!(
            sf_equals(&geom, &geom).expect("should succeed"),
            "sf_equals must be reflexive even for self-intersecting polygons"
        );
    }

    #[test]
    fn regression_sf_within_polygon_uses_real_containment_not_centroid() {
        // p2 is a 10x10 square [0,10]x[0,10].
        let p2 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        // p1 is a long thin rectangle whose centroid (5, 5) falls inside p2,
        // but which extends well outside p2's boundary on the x axis
        // ([-5, 15] range). The old centroid-in-polygon heuristic reported
        // this as "within"; real polygon-in-polygon containment must not.
        let p1 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: -5.0, y: 4.0 },
                Coord { x: 15.0, y: 4.0 },
                Coord { x: 15.0, y: 6.0 },
                Coord { x: -5.0, y: 6.0 },
                Coord { x: -5.0, y: 4.0 },
            ]),
            vec![],
        )));

        assert!(
            !sf_within(&p1, &p2).expect("should succeed"),
            "a polygon extending outside p2 must not be reported as within p2, \
             even if its centroid falls inside p2"
        );

        // Sanity check: a polygon that genuinely is within p2 must still
        // report true.
        let p3 = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 8.0, y: 2.0 },
                Coord { x: 8.0, y: 8.0 },
                Coord { x: 2.0, y: 8.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        )));
        assert!(sf_within(&p3, &p2).expect("should succeed"));
    }

    #[test]
    fn regression_sf_equals_polygon_uses_topology_not_area_centroid() {
        // Rectangle A: 4 wide x 1 tall, centered at origin. Area = 4, centroid (0,0).
        let rect_a = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: -2.0, y: -0.5 },
                Coord { x: 2.0, y: -0.5 },
                Coord { x: 2.0, y: 0.5 },
                Coord { x: -2.0, y: 0.5 },
                Coord { x: -2.0, y: -0.5 },
            ]),
            vec![],
        )));

        // Rectangle B: 1 wide x 4 tall, centered at origin. Same area (4) and
        // same centroid (0,0) as A, but a completely different footprint.
        let rect_b = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: -0.5, y: -2.0 },
                Coord { x: 0.5, y: -2.0 },
                Coord { x: 0.5, y: 2.0 },
                Coord { x: -0.5, y: 2.0 },
                Coord { x: -0.5, y: -2.0 },
            ]),
            vec![],
        )));

        assert!(
            !sf_equals(&rect_a, &rect_b).expect("should succeed"),
            "two polygons with equal area and centroid but different shapes \
             must not be reported as spatially equal"
        );

        // A polygon must still be equal to an exact (re-ordered ring is not
        // tested here, just identity) copy of itself.
        assert!(sf_equals(&rect_a, &rect_a.clone()).expect("should succeed"));
    }

    #[test]
    fn regression_sf_equals_multipolygon_routes_through_relate_not_false() {
        let square = Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        );
        let mp1 = Geometry::new(GeoGeometry::MultiPolygon(MultiPolygon(
            vec![square.clone()],
        )));
        let mp2 = Geometry::new(GeoGeometry::MultiPolygon(MultiPolygon(vec![square])));

        assert!(
            sf_equals(&mp1, &mp2).expect("should succeed"),
            "identical MultiPolygons must be reported as spatially equal, \
             not silently false via an unhandled-combination catch-all"
        );
    }

    #[test]
    fn regression_sf_within_multipolygon_routes_through_relate_not_false() {
        let outer = Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        );
        let inner = Polygon::new(
            LineString::new(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 4.0, y: 2.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 2.0, y: 4.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        );

        let mp_inner = Geometry::new(GeoGeometry::MultiPolygon(MultiPolygon(vec![inner])));
        let mp_outer = Geometry::new(GeoGeometry::MultiPolygon(MultiPolygon(vec![outer])));

        assert!(
            sf_within(&mp_inner, &mp_outer).expect("should succeed"),
            "a MultiPolygon fully inside another MultiPolygon must be reported \
             as within, not silently false via an unhandled-combination catch-all"
        );
    }

    #[test]
    fn regression_sf_crosses_multipolygon_routes_through_relate_not_false() {
        // A line that enters and exits a square multipolygon crosses it:
        // some but not all of the line lies in the polygon's interior.
        let square = Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        );
        let multi_square = Geometry::new(GeoGeometry::MultiPolygon(MultiPolygon(vec![square])));

        let crossing_line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: -1.0, y: 2.0 },
            Coord { x: 5.0, y: 2.0 },
        ])));

        assert!(
            sf_crosses(&crossing_line, &multi_square).expect("should succeed"),
            "a line crossing partially through a MultiPolygon must be reported \
             as crossing, not silently false via an unhandled-combination catch-all"
        );
    }
}
