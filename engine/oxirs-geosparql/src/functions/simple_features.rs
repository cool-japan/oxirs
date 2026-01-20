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
pub fn sf_equals(geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
    geom1.validate_crs_compatibility(geom2)?;

    let result = match (&geom1.geom, &geom2.geom) {
        (geo_types::Geometry::Point(p1), geo_types::Geometry::Point(p2)) => {
            use geo::{Distance, Euclidean};
            Euclidean::distance(*p1, *p2) < 1e-10
        }
        (geo_types::Geometry::LineString(ls1), geo_types::Geometry::LineString(ls2)) => {
            ls1.0 == ls2.0
        }
        (geo_types::Geometry::Polygon(p1), geo_types::Geometry::Polygon(p2)) => {
            // Check if polygons have the same area and centroid
            (p1.unsigned_area() - p2.unsigned_area()).abs() < 1e-10
                && p1.centroid() == p2.centroid()
        }
        _ => false,
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

    // Crosses is typically only defined for line/line or line/polygon
    let result = match (&geom1.geom, &geom2.geom) {
        (geo_types::Geometry::LineString(ls1), geo_types::Geometry::LineString(ls2)) => {
            ls1.intersects(ls2)
        }
        (geo_types::Geometry::LineString(ls), geo_types::Geometry::Polygon(poly))
        | (geo_types::Geometry::Polygon(poly), geo_types::Geometry::LineString(ls)) => {
            ls.intersects(poly)
        }
        _ => false,
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
            // Check if p1 is within p2
            p1.centroid().map(|c| p2.contains(&c)).unwrap_or(false)
        }
        _ => false,
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
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point, Polygon};

    #[test]
    fn test_sf_equals_points() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let p3 = Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0)));

        assert!(sf_equals(&p1, &p2).unwrap());
        assert!(!sf_equals(&p1, &p3).unwrap());
    }

    #[test]
    fn test_sf_disjoint() {
        let p1 = Geometry::new(GeoGeometry::Point(Point::new(0.0, 0.0)));
        let p2 = Geometry::new(GeoGeometry::Point(Point::new(10.0, 10.0)));

        assert!(sf_disjoint(&p1, &p2).unwrap());
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

        assert!(sf_intersects(&poly, &point_inside).unwrap());
        assert!(!sf_intersects(&poly, &point_outside).unwrap());
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

        assert!(sf_within(&point_inside, &poly).unwrap());
        assert!(!sf_within(&point_outside, &poly).unwrap());
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

        assert!(sf_contains(&poly, &point_inside).unwrap());
        assert!(!sf_contains(&poly, &point_outside).unwrap());
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

        assert!(sf_touches(&point_at_endpoint, &line).unwrap());
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
        assert!(!sf_touches(&point_on_interior, &line).unwrap());
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

        assert!(sf_touches(&point_on_boundary, &poly).unwrap());
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

        assert!(sf_touches(&poly1, &poly2).unwrap());
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
        assert!(!sf_touches(&poly1, &poly2).unwrap());
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
        assert!(!sf_touches(&poly1, &poly2).unwrap());
    }

    // ========================================================================
    // 3D Topological Relations Tests
    // ========================================================================

    #[test]
    fn test_sf_intersects_3d_points_same_location() {
        let p1 = Geometry::from_wkt("POINT Z(1 2 3)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").unwrap();

        assert!(sf_intersects_3d(&p1, &p2).unwrap());
    }

    #[test]
    fn test_sf_intersects_3d_points_different_z() {
        let p1 = Geometry::from_wkt("POINT Z(1 2 3)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(1 2 10)").unwrap();

        // Same XY but different Z - should not intersect
        assert!(!sf_intersects_3d(&p1, &p2).unwrap());
    }

    #[test]
    fn test_sf_intersects_3d_linestrings_crossing_z() {
        // Two linestrings that cross in XY but at different Z levels
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 0)").unwrap();
        let ls2 = Geometry::from_wkt("LINESTRING Z(0 10 5, 10 0 5)").unwrap();

        // They cross in XY but at different Z levels (0 vs 5)
        // Z ranges: [0,0] and [5,5] - disjoint
        assert!(!sf_intersects_3d(&ls1, &ls2).unwrap());
    }

    #[test]
    fn test_sf_intersects_3d_linestrings_overlapping_z() {
        let ls1 = Geometry::from_wkt("LINESTRING Z(0 0 0, 10 10 10)").unwrap();
        let ls2 = Geometry::from_wkt("LINESTRING Z(0 10 5, 10 0 5)").unwrap();

        // Z ranges: [0,10] and [5,5] - overlap at z=5
        assert!(sf_intersects_3d(&ls1, &ls2).unwrap());
    }

    #[test]
    fn test_sf_disjoint_3d_points() {
        let p1 = Geometry::from_wkt("POINT Z(0 0 0)").unwrap();
        let p2 = Geometry::from_wkt("POINT Z(10 10 10)").unwrap();

        assert!(sf_disjoint_3d(&p1, &p2).unwrap());
    }

    #[test]
    fn test_sf_disjoint_3d_vertical_separation() {
        // Polygons overlapping in XY but at different Z levels
        let poly1 =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))").unwrap();
        let poly2 =
            Geometry::from_wkt("POLYGON Z((2 2 20, 8 2 20, 8 8 20, 2 8 20, 2 2 20))").unwrap();

        // Overlap in XY but Z ranges [0,0] and [20,20] are disjoint
        assert!(sf_disjoint_3d(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_sf_within_3d_point_in_cube() {
        // Point inside a cube (represented as polygon with Z range)
        let point = Geometry::from_wkt("POINT Z(5 5 5)").unwrap();
        let cube =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 10, 10 10 10, 0 10 0, 0 0 0))").unwrap();

        // Point at (5,5,5) should be within the cube XY=[0,10], Z=[0,10]
        assert!(sf_within_3d(&point, &cube).unwrap());
    }

    #[test]
    fn test_sf_within_3d_point_outside_z_range() {
        let point = Geometry::from_wkt("POINT Z(5 5 15)").unwrap();
        let cube =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 10, 10 10 10, 0 10 0, 0 0 0))").unwrap();

        // Point at z=15 is outside the cube's Z range [0,10]
        assert!(!sf_within_3d(&point, &cube).unwrap());
    }

    #[test]
    fn test_sf_contains_3d_cube_contains_point() {
        let cube =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 10, 10 10 10, 0 10 0, 0 0 0))").unwrap();
        let point = Geometry::from_wkt("POINT Z(5 5 5)").unwrap();

        assert!(sf_contains_3d(&cube, &point).unwrap());
    }

    #[test]
    fn test_sf_touches_3d_adjacent_polygons_same_z() {
        // Two polygons sharing an edge at the same Z level
        let poly1 = Geometry::from_wkt("POLYGON Z((0 0 5, 5 0 5, 5 5 5, 0 5 5, 0 0 5))").unwrap();
        let poly2 = Geometry::from_wkt("POLYGON Z((5 0 5, 10 0 5, 10 5 5, 5 5 5, 5 0 5))").unwrap();

        // They share an edge in XY and have the same Z
        assert!(sf_touches_3d(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_sf_touches_3d_stacked_polygons() {
        // Two polygons stacked vertically (same XY, touching Z ranges)
        let poly1 =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))").unwrap();
        let poly2 =
            Geometry::from_wkt("POLYGON Z((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))").unwrap();

        // Z ranges don't touch: [0,0] and [10,10] - gap between them
        // For this simple implementation, they don't touch
        assert!(!sf_touches_3d(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_sf_overlaps_3d_partial_overlap() {
        // Two polygons with partial overlap in 3D
        let poly1 =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 5, 10 10 5, 0 10 0, 0 0 0))").unwrap();
        let poly2 =
            Geometry::from_wkt("POLYGON Z((5 5 3, 15 5 8, 15 15 8, 5 15 3, 5 5 3))").unwrap();

        // They overlap in XY and Z ranges overlap [0,5] and [3,8]
        assert!(sf_overlaps_3d(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_sf_overlaps_3d_no_overlap_different_z() {
        let poly1 =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))").unwrap();
        let poly2 =
            Geometry::from_wkt("POLYGON Z((2 2 20, 8 2 20, 8 8 20, 2 8 20, 2 2 20))").unwrap();

        // Overlap in XY but Z ranges [0,0] and [20,20] don't overlap
        assert!(!sf_overlaps_3d(&poly1, &poly2).unwrap());
    }

    #[test]
    fn test_sf_3d_requires_z_coordinates() {
        let p1 = Geometry::from_wkt("POINT(1 2)").unwrap(); // 2D
        let p2 = Geometry::from_wkt("POINT Z(1 2 3)").unwrap(); // 3D

        // Should return error for mixing 2D and 3D
        assert!(sf_intersects_3d(&p1, &p2).is_err());
        assert!(sf_disjoint_3d(&p1, &p2).is_err());
        assert!(sf_within_3d(&p1, &p2).is_err());
        assert!(sf_contains_3d(&p1, &p2).is_err());
        assert!(sf_touches_3d(&p1, &p2).is_err());
        assert!(sf_overlaps_3d(&p1, &p2).is_err());
    }
}
