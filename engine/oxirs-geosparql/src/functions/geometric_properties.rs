//! Geometric property functions

use crate::error::{GeoSparqlError, Result};
use crate::geometry::Geometry;
use geo_types::Geometry as GeoGeometry;

/// Get the SRID (Spatial Reference ID) from a geometry's CRS
pub fn get_srid(geom: &Geometry) -> Result<Option<String>> {
    // Extract EPSG code from CRS URI if present
    if let Some(epsg_code) = geom.crs.uri.strip_prefix(crate::vocabulary::EPSG_PREFIX) {
        Ok(Some(epsg_code.to_string()))
    } else {
        Ok(None)
    }
}

/// Get the dimension of a geometry (0=point, 1=line, 2=surface)
pub fn dimension(geom: &Geometry) -> Result<u8> {
    Ok(geom.spatial_dimension())
}

/// Get the coordinate dimension (2 or 3)
pub fn coordinate_dimension(geom: &Geometry) -> Result<u8> {
    Ok(geom.coordinate_dimension())
}

/// Get the spatial dimension
pub fn spatial_dimension(geom: &Geometry) -> Result<u8> {
    Ok(geom.spatial_dimension())
}

/// Check if a geometry is empty
pub fn is_empty(geom: &Geometry) -> Result<bool> {
    Ok(geom.is_empty())
}

/// Check if a geometry is simple (no self-intersections)
pub fn is_simple(geom: &Geometry) -> Result<bool> {
    Ok(geom.is_simple())
}

/// Check if a geometry has 3D coordinates
pub fn is_3d(geom: &Geometry) -> Result<bool> {
    Ok(geom.is_3d())
}

/// Check if a geometry has measured (M) coordinates
pub fn is_measured(geom: &Geometry) -> Result<bool> {
    Ok(geom.is_measured())
}

/// Calculate the area of a geometry
///
/// Returns the area for Polygon and MultiPolygon geometries.
/// For other geometry types, returns 0.0.
pub fn area(geom: &Geometry) -> Result<f64> {
    use geo::Area;

    let area = match &geom.geom {
        GeoGeometry::Polygon(p) => p.unsigned_area(),
        GeoGeometry::MultiPolygon(mp) => mp.unsigned_area(),
        GeoGeometry::Triangle(t) => t.unsigned_area(),
        GeoGeometry::Rect(r) => r.unsigned_area(),
        _ => 0.0,
    };

    Ok(area)
}

/// Calculate the signed area of a geometry
///
/// Returns the signed area (positive for counter-clockwise, negative for clockwise)
/// for Polygon and MultiPolygon geometries.
pub fn signed_area(geom: &Geometry) -> Result<f64> {
    use geo::Area;

    let area = match &geom.geom {
        GeoGeometry::Polygon(p) => p.signed_area(),
        GeoGeometry::MultiPolygon(mp) => mp.signed_area(),
        GeoGeometry::Triangle(t) => t.signed_area(),
        GeoGeometry::Rect(r) => r.signed_area(),
        _ => 0.0,
    };

    Ok(area)
}

/// Calculate the length/perimeter of a geometry
///
/// For LineString and MultiLineString: returns the total length
/// For Polygon and MultiPolygon: returns the perimeter
/// For other types: returns 0.0
pub fn length(geom: &Geometry) -> Result<f64> {
    use geo::{Euclidean, Length};

    let len = match &geom.geom {
        GeoGeometry::Line(l) => Euclidean.length(l),
        GeoGeometry::LineString(ls) => Euclidean.length(ls),
        GeoGeometry::MultiLineString(mls) => Euclidean.length(mls),
        GeoGeometry::Polygon(p) => {
            // Perimeter = exterior + all interiors
            let mut total = Euclidean.length(p.exterior());
            for interior in p.interiors() {
                total += Euclidean.length(interior);
            }
            total
        }
        GeoGeometry::MultiPolygon(mp) => {
            let mut total = 0.0;
            for poly in &mp.0 {
                total += Euclidean.length(poly.exterior());
                for interior in poly.interiors() {
                    total += Euclidean.length(interior);
                }
            }
            total
        }
        GeoGeometry::Triangle(t) => {
            // Calculate perimeter of triangle: sum of three sides
            let d1 = ((t.1.x - t.0.x).powi(2) + (t.1.y - t.0.y).powi(2)).sqrt();
            let d2 = ((t.2.x - t.1.x).powi(2) + (t.2.y - t.1.y).powi(2)).sqrt();
            let d3 = ((t.0.x - t.2.x).powi(2) + (t.0.y - t.2.y).powi(2)).sqrt();
            d1 + d2 + d3
        }
        GeoGeometry::Rect(r) => {
            let width = r.max().x - r.min().x;
            let height = r.max().y - r.min().y;
            2.0 * (width + height)
        }
        _ => 0.0,
    };

    Ok(len)
}

/// Calculate the centroid (center of mass) of a geometry
///
/// Returns None if the geometry is empty or the centroid cannot be calculated.
pub fn centroid(geom: &Geometry) -> Result<Option<Geometry>> {
    use geo::Centroid;

    let centroid_point = match &geom.geom {
        GeoGeometry::Point(p) => Some(*p),
        GeoGeometry::Line(l) => Some(l.centroid()),
        GeoGeometry::LineString(ls) => ls.centroid(),
        GeoGeometry::Polygon(p) => p.centroid(),
        GeoGeometry::MultiPoint(mp) => mp.centroid(),
        GeoGeometry::MultiLineString(mls) => mls.centroid(),
        GeoGeometry::MultiPolygon(mp) => mp.centroid(),
        GeoGeometry::Triangle(t) => Some(t.centroid()),
        GeoGeometry::Rect(r) => Some(r.centroid()),
        GeoGeometry::GeometryCollection(_) => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "Centroid not supported for GeometryCollection".to_string(),
            ))
        }
    };

    Ok(centroid_point.map(|p| Geometry::with_crs(GeoGeometry::Point(p), geom.crs.clone())))
}

/// Find a point guaranteed to be on the surface of a geometry
///
/// This is similar to centroid but guarantees the point is actually on/in the geometry.
/// For Point: returns the point itself
/// For LineString: returns the midpoint of the first segment
/// For Polygon: returns a point on the exterior ring (uses centroid if it's inside, otherwise finds a point on the boundary)
pub fn point_on_surface(geom: &Geometry) -> Result<Geometry> {
    use geo::{Centroid, Contains};
    use geo_types::Point;

    let point = match &geom.geom {
        GeoGeometry::Point(p) => *p,
        GeoGeometry::Line(l) => {
            // Midpoint of the line
            Point::new((l.start.x + l.end.x) / 2.0, (l.start.y + l.end.y) / 2.0)
        }
        GeoGeometry::LineString(ls) => {
            if ls.0.is_empty() {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot find point on empty LineString".to_string(),
                ));
            }
            // Return the first point
            Point::from(ls.0[0])
        }
        GeoGeometry::Polygon(p) => {
            // Try centroid first
            if let Some(c) = p.centroid() {
                if p.contains(&c) {
                    c
                } else {
                    // Centroid is outside, find a point on the exterior ring
                    if p.exterior().0.is_empty() {
                        return Err(GeoSparqlError::GeometryOperationFailed(
                            "Cannot find point on empty Polygon".to_string(),
                        ));
                    }
                    Point::from(p.exterior().0[0])
                }
            } else {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot calculate centroid for Polygon".to_string(),
                ));
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            if mp.0.is_empty() {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot find point on empty MultiPoint".to_string(),
                ));
            }
            mp.0[0]
        }
        GeoGeometry::MultiLineString(mls) => {
            if mls.0.is_empty() {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot find point on empty MultiLineString".to_string(),
                ));
            }
            if mls.0[0].0.is_empty() {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot find point on empty LineString in MultiLineString".to_string(),
                ));
            }
            Point::from(mls.0[0].0[0])
        }
        GeoGeometry::MultiPolygon(mp) => {
            if mp.0.is_empty() {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot find point on empty MultiPolygon".to_string(),
                ));
            }
            // Use the first polygon's logic
            let p = &mp.0[0];
            if let Some(c) = p.centroid() {
                if p.contains(&c) {
                    c
                } else {
                    if p.exterior().0.is_empty() {
                        return Err(GeoSparqlError::GeometryOperationFailed(
                            "Cannot find point on empty Polygon in MultiPolygon".to_string(),
                        ));
                    }
                    Point::from(p.exterior().0[0])
                }
            } else {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot calculate centroid for Polygon in MultiPolygon".to_string(),
                ));
            }
        }
        GeoGeometry::Triangle(t) => t.centroid(),
        GeoGeometry::Rect(r) => r.centroid(),
        GeoGeometry::GeometryCollection(_) => {
            return Err(GeoSparqlError::UnsupportedOperation(
                "pointOnSurface not supported for GeometryCollection".to_string(),
            ))
        }
    };

    Ok(Geometry::with_crs(
        GeoGeometry::Point(point),
        geom.crs.clone(),
    ))
}

/// Get the number of geometries in a geometry collection
///
/// For simple geometries (Point, LineString, Polygon): returns 1
/// For multi-geometries: returns the number of components
pub fn num_geometries(geom: &Geometry) -> Result<usize> {
    let count = match &geom.geom {
        GeoGeometry::Point(_)
        | GeoGeometry::Line(_)
        | GeoGeometry::LineString(_)
        | GeoGeometry::Polygon(_)
        | GeoGeometry::Triangle(_)
        | GeoGeometry::Rect(_) => 1,
        GeoGeometry::MultiPoint(mp) => mp.0.len(),
        GeoGeometry::MultiLineString(mls) => mls.0.len(),
        GeoGeometry::MultiPolygon(mp) => mp.0.len(),
        GeoGeometry::GeometryCollection(gc) => gc.0.len(),
    };

    Ok(count)
}

/// Extract the Nth geometry from a geometry collection (0-indexed)
///
/// For simple geometries: only index 0 is valid, returns the geometry itself
/// For multi-geometries: returns the Nth component
/// Returns None if the index is out of bounds
pub fn geometry_n(geom: &Geometry, n: usize) -> Result<Option<Geometry>> {
    let result = match &geom.geom {
        GeoGeometry::Point(_)
        | GeoGeometry::Line(_)
        | GeoGeometry::LineString(_)
        | GeoGeometry::Polygon(_)
        | GeoGeometry::Triangle(_)
        | GeoGeometry::Rect(_) => {
            if n == 0 {
                Some(geom.clone())
            } else {
                None
            }
        }
        GeoGeometry::MultiPoint(mp) => {
            mp.0.get(n)
                .map(|p| Geometry::with_crs(GeoGeometry::Point(*p), geom.crs.clone()))
        }
        GeoGeometry::MultiLineString(mls) => mls
            .0
            .get(n)
            .map(|ls| Geometry::with_crs(GeoGeometry::LineString(ls.clone()), geom.crs.clone())),
        GeoGeometry::MultiPolygon(mp) => {
            mp.0.get(n)
                .map(|p| Geometry::with_crs(GeoGeometry::Polygon(p.clone()), geom.crs.clone()))
        }
        GeoGeometry::GeometryCollection(gc) => {
            gc.0.get(n)
                .map(|g| Geometry::with_crs(g.clone(), geom.crs.clone()))
        }
    };

    Ok(result)
}

/// Get the start point of a LineString
///
/// Returns an error for non-LineString geometries or empty LineString
pub fn start_point(geom: &Geometry) -> Result<Geometry> {
    match &geom.geom {
        GeoGeometry::LineString(ls) => {
            if ls.0.is_empty() {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot get start point of empty LineString".to_string(),
                ));
            }
            let point = geo_types::Point::from(ls.0[0]);
            Ok(Geometry::with_crs(
                GeoGeometry::Point(point),
                geom.crs.clone(),
            ))
        }
        GeoGeometry::Line(l) => {
            let point = geo_types::Point::new(l.start.x, l.start.y);
            Ok(Geometry::with_crs(
                GeoGeometry::Point(point),
                geom.crs.clone(),
            ))
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "startPoint only supported for LineString (got {})",
            geom.geometry_type()
        ))),
    }
}

/// Get the end point of a LineString
///
/// Returns an error for non-LineString geometries or empty LineString
pub fn end_point(geom: &Geometry) -> Result<Geometry> {
    match &geom.geom {
        GeoGeometry::LineString(ls) => {
            if ls.0.is_empty() {
                return Err(GeoSparqlError::GeometryOperationFailed(
                    "Cannot get end point of empty LineString".to_string(),
                ));
            }
            let point = geo_types::Point::from(ls.0[ls.0.len() - 1]);
            Ok(Geometry::with_crs(
                GeoGeometry::Point(point),
                geom.crs.clone(),
            ))
        }
        GeoGeometry::Line(l) => {
            let point = geo_types::Point::new(l.end.x, l.end.y);
            Ok(Geometry::with_crs(
                GeoGeometry::Point(point),
                geom.crs.clone(),
            ))
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "endPoint only supported for LineString (got {})",
            geom.geometry_type()
        ))),
    }
}

/// Get the exterior ring of a Polygon
///
/// Returns an error for non-Polygon geometries
pub fn exterior_ring(geom: &Geometry) -> Result<Geometry> {
    match &geom.geom {
        GeoGeometry::Polygon(p) => {
            let ring = p.exterior().clone();
            Ok(Geometry::with_crs(
                GeoGeometry::LineString(ring),
                geom.crs.clone(),
            ))
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "exteriorRing only supported for Polygon (got {})",
            geom.geometry_type()
        ))),
    }
}

/// Get the number of interior rings (holes) in a Polygon
///
/// Returns an error for non-Polygon geometries
pub fn num_interior_rings(geom: &Geometry) -> Result<usize> {
    match &geom.geom {
        GeoGeometry::Polygon(p) => Ok(p.interiors().len()),
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "numInteriorRings only supported for Polygon (got {})",
            geom.geometry_type()
        ))),
    }
}

/// Get the Nth interior ring (hole) of a Polygon (0-indexed)
///
/// Returns None if the index is out of bounds
/// Returns an error for non-Polygon geometries
pub fn interior_ring_n(geom: &Geometry, n: usize) -> Result<Option<Geometry>> {
    match &geom.geom {
        GeoGeometry::Polygon(p) => {
            let ring = p
                .interiors()
                .get(n)
                .map(|r| Geometry::with_crs(GeoGeometry::LineString(r.clone()), geom.crs.clone()));
            Ok(ring)
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "interiorRingN only supported for Polygon (got {})",
            geom.geometry_type()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point, Polygon};

    #[test]
    fn test_dimension() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert_eq!(dimension(&point).unwrap(), 0);

        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
        ])));
        assert_eq!(dimension(&line).unwrap(), 1);
    }

    #[test]
    fn test_coordinate_dimension() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert_eq!(coordinate_dimension(&point).unwrap(), 2);
    }

    #[test]
    fn test_is_empty() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert!(!is_empty(&point).unwrap());
    }

    #[test]
    fn test_is_3d() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert!(!is_3d(&point).unwrap());
    }

    #[test]
    fn test_is_measured() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert!(!is_measured(&point).unwrap());
    }

    #[test]
    fn test_get_srid() {
        use crate::geometry::Crs;

        let point_with_epsg =
            Geometry::with_crs(GeoGeometry::Point(Point::new(1.0, 2.0)), Crs::epsg(4326));
        let srid = get_srid(&point_with_epsg).unwrap();
        assert_eq!(srid, Some("4326".to_string()));

        let point_default = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        let srid_default = get_srid(&point_default).unwrap();
        assert_eq!(srid_default, None);
    }

    #[test]
    fn test_area_polygon() {
        // Create a square polygon with area 100 (10x10)
        let square = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let area_result = area(&square).unwrap();
        assert!((area_result - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_with_hole() {
        // Square with a hole
        let exterior = LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);

        let hole = LineString::new(vec![
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 8.0, y: 2.0 },
            Coord { x: 8.0, y: 8.0 },
            Coord { x: 2.0, y: 8.0 },
            Coord { x: 2.0, y: 2.0 },
        ]);

        let poly_with_hole =
            Geometry::new(GeoGeometry::Polygon(Polygon::new(exterior, vec![hole])));

        let area_result = area(&poly_with_hole).unwrap();
        // Outer area (100) - hole area (36) = 64
        assert!((area_result - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_non_polygon() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert_eq!(area(&point).unwrap(), 0.0);

        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
        ])));
        assert_eq!(area(&line).unwrap(), 0.0);
    }

    #[test]
    fn test_signed_area() {
        // Counter-clockwise polygon should have positive area
        let ccw_polygon = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let signed = signed_area(&ccw_polygon).unwrap();
        assert!(signed > 0.0);
        assert!((signed.abs() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_length_linestring() {
        // Right triangle with sides 3, 4, 5
        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 3.0, y: 0.0 },
            Coord { x: 3.0, y: 4.0 },
        ])));

        let len = length(&line).unwrap();
        assert!((len - 7.0).abs() < 1e-10); // 3 + 4 = 7
    }

    #[test]
    fn test_length_polygon_perimeter() {
        // Square polygon with perimeter 40 (10 * 4)
        let square = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let len = length(&square).unwrap();
        assert!((len - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_polygon() {
        // Square with center at (5, 5)
        let square = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let centroid_geom = centroid(&square).unwrap().unwrap();
        if let GeoGeometry::Point(p) = centroid_geom.geom {
            assert!((p.x() - 5.0).abs() < 1e-10);
            assert!((p.y() - 5.0).abs() < 1e-10);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_centroid_point() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0)));
        let centroid_geom = centroid(&point).unwrap().unwrap();

        if let GeoGeometry::Point(p) = centroid_geom.geom {
            assert_eq!(p.x(), 3.0);
            assert_eq!(p.y(), 4.0);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_point_on_surface_point() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(3.0, 4.0)));
        let pos = point_on_surface(&point).unwrap();

        if let GeoGeometry::Point(p) = pos.geom {
            assert_eq!(p.x(), 3.0);
            assert_eq!(p.y(), 4.0);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_point_on_surface_linestring() {
        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 10.0 },
        ])));

        let pos = point_on_surface(&line).unwrap();
        if let GeoGeometry::Point(p) = pos.geom {
            // Should return the first point
            assert_eq!(p.x(), 0.0);
            assert_eq!(p.y(), 0.0);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_point_on_surface_polygon() {
        let square = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let pos = point_on_surface(&square).unwrap();
        if let GeoGeometry::Point(p) = pos.geom {
            // Centroid (5,5) should be inside the square
            assert!((p.x() - 5.0).abs() < 1e-10);
            assert!((p.y() - 5.0).abs() < 1e-10);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_num_geometries() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert_eq!(num_geometries(&point).unwrap(), 1);

        let multipoint = Geometry::new(GeoGeometry::MultiPoint(geo_types::MultiPoint::new(vec![
            Point::new(1.0, 2.0),
            Point::new(3.0, 4.0),
            Point::new(5.0, 6.0),
        ])));
        assert_eq!(num_geometries(&multipoint).unwrap(), 3);
    }

    #[test]
    fn test_geometry_n() {
        let multipoint = Geometry::new(GeoGeometry::MultiPoint(geo_types::MultiPoint::new(vec![
            Point::new(1.0, 2.0),
            Point::new(3.0, 4.0),
            Point::new(5.0, 6.0),
        ])));

        let geom0 = geometry_n(&multipoint, 0).unwrap().unwrap();
        if let GeoGeometry::Point(p) = geom0.geom {
            assert_eq!(p.x(), 1.0);
            assert_eq!(p.y(), 2.0);
        } else {
            panic!("Expected Point geometry");
        }

        let geom2 = geometry_n(&multipoint, 2).unwrap().unwrap();
        if let GeoGeometry::Point(p) = geom2.geom {
            assert_eq!(p.x(), 5.0);
            assert_eq!(p.y(), 6.0);
        } else {
            panic!("Expected Point geometry");
        }

        // Out of bounds
        assert!(geometry_n(&multipoint, 3).unwrap().is_none());
    }

    #[test]
    fn test_geometry_n_simple() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));

        // Index 0 should return the geometry itself
        let geom0 = geometry_n(&point, 0).unwrap().unwrap();
        assert_eq!(geom0.geometry_type(), "Point");

        // Index 1 should be out of bounds
        assert!(geometry_n(&point, 1).unwrap().is_none());
    }

    #[test]
    fn test_start_point() {
        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 1.0, y: 2.0 },
            Coord { x: 3.0, y: 4.0 },
            Coord { x: 5.0, y: 6.0 },
        ])));

        let start = start_point(&line).unwrap();
        if let GeoGeometry::Point(p) = start.geom {
            assert_eq!(p.x(), 1.0);
            assert_eq!(p.y(), 2.0);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_end_point() {
        let line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 1.0, y: 2.0 },
            Coord { x: 3.0, y: 4.0 },
            Coord { x: 5.0, y: 6.0 },
        ])));

        let end = end_point(&line).unwrap();
        if let GeoGeometry::Point(p) = end.geom {
            assert_eq!(p.x(), 5.0);
            assert_eq!(p.y(), 6.0);
        } else {
            panic!("Expected Point geometry");
        }
    }

    #[test]
    fn test_start_end_point_errors() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));

        // Should fail for non-LineString
        assert!(start_point(&point).is_err());
        assert!(end_point(&point).is_err());

        // Should fail for empty LineString
        let empty_line = Geometry::new(GeoGeometry::LineString(LineString::new(vec![])));
        assert!(start_point(&empty_line).is_err());
        assert!(end_point(&empty_line).is_err());
    }

    #[test]
    fn test_exterior_ring() {
        let square = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));

        let ring = exterior_ring(&square).unwrap();
        assert_eq!(ring.geometry_type(), "LineString");

        if let GeoGeometry::LineString(ls) = ring.geom {
            assert_eq!(ls.0.len(), 5); // 4 corners + closing point
        } else {
            panic!("Expected LineString geometry");
        }
    }

    #[test]
    fn test_exterior_ring_error() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert!(exterior_ring(&point).is_err());
    }

    #[test]
    fn test_num_interior_rings() {
        // Polygon without holes
        let simple_poly = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        )));
        assert_eq!(num_interior_rings(&simple_poly).unwrap(), 0);

        // Polygon with 2 holes
        let hole1 = LineString::new(vec![
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 3.0, y: 1.0 },
            Coord { x: 3.0, y: 3.0 },
            Coord { x: 1.0, y: 3.0 },
            Coord { x: 1.0, y: 1.0 },
        ]);
        let hole2 = LineString::new(vec![
            Coord { x: 5.0, y: 5.0 },
            Coord { x: 8.0, y: 5.0 },
            Coord { x: 8.0, y: 8.0 },
            Coord { x: 5.0, y: 8.0 },
            Coord { x: 5.0, y: 5.0 },
        ]);

        let poly_with_holes = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![hole1, hole2],
        )));
        assert_eq!(num_interior_rings(&poly_with_holes).unwrap(), 2);
    }

    #[test]
    fn test_interior_ring_n() {
        let hole = LineString::new(vec![
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 8.0, y: 2.0 },
            Coord { x: 8.0, y: 8.0 },
            Coord { x: 2.0, y: 8.0 },
            Coord { x: 2.0, y: 2.0 },
        ]);

        let poly_with_hole = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![hole],
        )));

        let ring0 = interior_ring_n(&poly_with_hole, 0).unwrap().unwrap();
        assert_eq!(ring0.geometry_type(), "LineString");

        // Out of bounds
        assert!(interior_ring_n(&poly_with_hole, 1).unwrap().is_none());
    }

    #[test]
    fn test_interior_ring_errors() {
        let point = Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0)));
        assert!(num_interior_rings(&point).is_err());
        assert!(interior_ring_n(&point, 0).is_err());
    }
}
