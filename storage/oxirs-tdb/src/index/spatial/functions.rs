//! GeoSPARQL spatial functions
//!
//! Implements common GeoSPARQL spatial relationship functions

use super::{Geometry, Point};
use geo::{
    Area as GeoArea, BoundingRect, Centroid as GeoCentroid, Contains as GeoContains,
    Intersects as GeoIntersects,
};

/// Calculate the distance between two geometries (in meters)
///
/// Uses Haversine formula for point-to-point distance
pub fn distance(g1: &Geometry, g2: &Geometry) -> f64 {
    use geo::{Distance, Euclidean, Haversine};

    match (g1, g2) {
        (Geometry::Point(p1), Geometry::Point(p2)) => {
            let geo_p1 = p1.to_geo();
            let geo_p2 = p2.to_geo();
            Haversine::distance(geo_p1, geo_p2)
        }
        (Geometry::Point(p), other) | (other, Geometry::Point(p)) => other.distance_to(p),
        (g1, g2) => {
            // For non-point geometries, use Euclidean distance
            let geo_g1 = g1.to_geo();
            let geo_g2 = g2.to_geo();
            Euclidean::distance(&geo_g1, &geo_g2)
        }
    }
}

/// Check if geometry g1 contains geometry g2
///
/// GeoSPARQL: sfContains
pub fn contains(g1: &Geometry, g2: &Geometry) -> bool {
    if let Geometry::Point(p) = g2 {
        return g1.contains(p);
    }

    let geo_g1 = g1.to_geo();
    let geo_g2 = g2.to_geo();
    geo_g1.contains(&geo_g2)
}

/// Check if two geometries intersect
///
/// GeoSPARQL: sfIntersects
pub fn intersects(g1: &Geometry, g2: &Geometry) -> bool {
    let geo_g1 = g1.to_geo();
    let geo_g2 = g2.to_geo();
    geo_g1.intersects(&geo_g2)
}

/// Check if geometry g1 is within geometry g2
///
/// GeoSPARQL: sfWithin
pub fn within(g1: &Geometry, g2: &Geometry) -> bool {
    contains(g2, g1)
}

/// Check if two geometries touch (share a boundary but no interior)
///
/// GeoSPARQL: sfTouches
pub fn touches(g1: &Geometry, g2: &Geometry) -> bool {
    // Touches means they intersect but don't overlap
    intersects(g1, g2) && !contains(g1, g2) && !contains(g2, g1)
}

/// Check if two geometries are disjoint (don't intersect)
///
/// GeoSPARQL: sfDisjoint
pub fn disjoint(g1: &Geometry, g2: &Geometry) -> bool {
    !intersects(g1, g2)
}

/// Calculate the area of a geometry (in square meters)
///
/// GeoSPARQL: area
pub fn area(geometry: &Geometry) -> f64 {
    match geometry {
        Geometry::Point(_) => 0.0,
        Geometry::LineString(_) => 0.0,
        Geometry::Polygon(poly) => {
            let geo_poly = poly.to_geo();
            geo_poly.unsigned_area()
        }
    }
}

/// Calculate the centroid of a geometry
///
/// GeoSPARQL: centroid
pub fn centroid(geometry: &Geometry) -> Option<Point> {
    let geo_geom = geometry.to_geo();
    geo_geom.centroid().map(|c| Point::new(c.y(), c.x()))
}

/// Calculate the envelope (bounding box) of a geometry
///
/// GeoSPARQL: envelope
pub fn envelope(geometry: &Geometry) -> Option<super::BoundingBox> {
    let geo_geom = geometry.to_geo();
    geo_geom
        .bounding_rect()
        .map(|rect| super::BoundingBox::new(rect.min().y, rect.min().x, rect.max().y, rect.max().x))
}

/// Check if a geometry is simple (no self-intersections)
///
/// GeoSPARQL: isSimple
pub fn is_simple(geometry: &Geometry) -> bool {
    match geometry {
        Geometry::Point(_) => true,
        Geometry::LineString(_) => {
            // TODO: Implement proper self-intersection check
            true
        }
        Geometry::Polygon(_) => {
            // TODO: Implement proper self-intersection check
            true
        }
    }
}

/// Check if a geometry is valid
///
/// GeoSPARQL: isValid
pub fn is_valid(geometry: &Geometry) -> bool {
    match geometry {
        Geometry::Point(_) => true,
        Geometry::LineString(ls) => ls.points.len() >= 2,
        Geometry::Polygon(poly) => poly.exterior.len() >= 3,
    }
}

/// Check if two geometries are equal
///
/// GeoSPARQL: sfEquals
pub fn equals(g1: &Geometry, g2: &Geometry) -> bool {
    match (g1, g2) {
        (Geometry::Point(p1), Geometry::Point(p2)) => p1.equals(p2),
        _ => {
            // For complex geometries, check if they contain each other
            contains(g1, g2) && contains(g2, g1)
        }
    }
}

/// Calculate the buffer (expansion) of a geometry by a distance
///
/// GeoSPARQL: buffer
///
/// Note: This is a simplified implementation. Full buffer operation
/// requires more sophisticated geometric algorithms.
pub fn buffer(geometry: &Geometry, distance_meters: f64) -> super::BoundingBox {
    let bbox = geometry.bounding_box();

    // Approximate: 1 degree ≈ 111 km at equator
    let degree_delta = distance_meters / 111_000.0;

    super::BoundingBox::new(
        bbox.min_lat - degree_delta,
        bbox.min_lon - degree_delta,
        bbox.max_lat + degree_delta,
        bbox.max_lon + degree_delta,
    )
}

/// Check if geometry g1 overlaps geometry g2
///
/// GeoSPARQL: sfOverlaps
pub fn overlaps(g1: &Geometry, g2: &Geometry) -> bool {
    intersects(g1, g2) && !contains(g1, g2) && !contains(g2, g1)
}

/// Check if geometry g1 crosses geometry g2
///
/// GeoSPARQL: sfCrosses
pub fn crosses(g1: &Geometry, g2: &Geometry) -> bool {
    // Simplified: crosses means they intersect but neither contains the other
    intersects(g1, g2) && !contains(g1, g2) && !contains(g2, g1)
}

#[cfg(test)]
mod tests {
    use super::super::Polygon;
    use super::*;

    #[test]
    fn test_distance_points() {
        let p1 = Point::new(40.7128, -74.0060); // NYC
        let p2 = Point::new(40.7589, -73.9851); // Times Square

        let g1 = Geometry::Point(p1);
        let g2 = Geometry::Point(p2);

        let d = distance(&g1, &g2);
        // Distance should be around 5-6 km
        assert!(d > 5000.0 && d < 7000.0);
    }

    #[test]
    fn test_contains_point_in_polygon() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let polygon = Polygon::new(square).unwrap();
        let g1 = Geometry::Polygon(polygon);

        let inside_point = Point::new(0.5, 0.5);
        let g2 = Geometry::Point(inside_point);

        assert!(contains(&g1, &g2));
    }

    #[test]
    fn test_contains_point_outside_polygon() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let polygon = Polygon::new(square).unwrap();
        let g1 = Geometry::Polygon(polygon);

        let outside_point = Point::new(2.0, 2.0);
        let g2 = Geometry::Point(outside_point);

        assert!(!contains(&g1, &g2));
    }

    #[test]
    fn test_intersects_same_point() {
        let p = Point::new(40.7128, -74.0060);
        let g1 = Geometry::Point(p);
        let g2 = Geometry::Point(p);

        assert!(intersects(&g1, &g2));
    }

    #[test]
    fn test_within() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let polygon = Polygon::new(square).unwrap();
        let g1 = Geometry::Polygon(polygon);

        let inside_point = Point::new(0.5, 0.5);
        let g2 = Geometry::Point(inside_point);

        assert!(within(&g2, &g1));
    }

    #[test]
    fn test_disjoint() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(10.0, 10.0);
        let g1 = Geometry::Point(p1);
        let g2 = Geometry::Point(p2);

        assert!(disjoint(&g1, &g2));
    }

    #[test]
    fn test_area_point() {
        let p = Point::new(40.7128, -74.0060);
        let g = Geometry::Point(p);

        assert_eq!(area(&g), 0.0);
    }

    #[test]
    fn test_area_polygon() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let polygon = Polygon::new(square).unwrap();
        let g = Geometry::Polygon(polygon);

        let a = area(&g);
        assert!(a > 0.0);
    }

    #[test]
    fn test_centroid_point() {
        let p = Point::new(40.7128, -74.0060);
        let g = Geometry::Point(p);

        let c = centroid(&g).unwrap();
        assert!(c.equals(&p));
    }

    #[test]
    fn test_is_valid_point() {
        let p = Point::new(40.7128, -74.0060);
        let g = Geometry::Point(p);

        assert!(is_valid(&g));
    }

    #[test]
    fn test_is_valid_polygon() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let polygon = Polygon::new(square).unwrap();
        let g = Geometry::Polygon(polygon);

        assert!(is_valid(&g));
    }

    #[test]
    fn test_equals_same_point() {
        let p = Point::new(40.7128, -74.0060);
        let g1 = Geometry::Point(p);
        let g2 = Geometry::Point(p);

        assert!(equals(&g1, &g2));
    }

    #[test]
    fn test_buffer() {
        let p = Point::new(40.7128, -74.0060);
        let g = Geometry::Point(p);

        let buffered = buffer(&g, 1000.0);

        // Buffered bbox should be larger than original point
        assert!(buffered.max_lat > p.lat);
        assert!(buffered.min_lat < p.lat);
    }

    #[test]
    fn test_overlaps() {
        // Create two overlapping polygons
        let poly1 = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let poly2 = vec![
            Point::new(0.5, 0.5),
            Point::new(1.5, 0.5),
            Point::new(1.5, 1.5),
            Point::new(0.5, 1.5),
            Point::new(0.5, 0.5),
        ];

        let g1 = Geometry::Polygon(Polygon::new(poly1).unwrap());
        let g2 = Geometry::Polygon(Polygon::new(poly2).unwrap());

        assert!(overlaps(&g1, &g2));
    }
}
