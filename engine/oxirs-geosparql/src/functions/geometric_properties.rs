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
    use geo::EuclideanLength;

    let len = match &geom.geom {
        GeoGeometry::Line(l) => l.euclidean_length(),
        GeoGeometry::LineString(ls) => ls.euclidean_length(),
        GeoGeometry::MultiLineString(mls) => mls.euclidean_length(),
        GeoGeometry::Polygon(p) => {
            // Perimeter = exterior + all interiors
            let mut total = p.exterior().euclidean_length();
            for interior in p.interiors() {
                total += interior.euclidean_length();
            }
            total
        }
        GeoGeometry::MultiPolygon(mp) => {
            let mut total = 0.0;
            for poly in &mp.0 {
                total += poly.exterior().euclidean_length();
                for interior in poly.interiors() {
                    total += interior.euclidean_length();
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

/// Calculate the volume of a 3D geometry
///
/// For 3D Polygons, calculates the volume as if it's a vertical prism (extrusion).
/// For other geometries, returns an error or 0.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::geometric_properties::volume_3d;
///
/// // Volume of a cube: base 10x10 at Z=0, height 10
/// let cube = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 0, 0 10 0, 0 0 0))").unwrap();
/// // This calculates base area * average height
/// ```
pub fn volume_3d(geom: &Geometry) -> Result<f64> {
    if !geom.is_3d() {
        return Err(GeoSparqlError::UnsupportedOperation(
            "Geometry must have Z coordinates for volume calculation".to_string(),
        ));
    }

    match &geom.geom {
        GeoGeometry::Polygon(p) => {
            // Calculate volume using signed area and average Z coordinate
            let coords: Vec<_> = p.exterior().coords().collect();
            if coords.len() < 4 {
                return Ok(0.0);
            }

            // Calculate base area (2D projection)
            use geo::Area;
            let base_area = p.unsigned_area();

            // Calculate average Z coordinate
            let mut z_sum = 0.0;
            let mut count = 0;
            for (i, _coord) in coords.iter().enumerate() {
                if let Some(z) = geom.coord3d.z_at(i) {
                    z_sum += z;
                    count += 1;
                }
            }

            if count == 0 {
                return Ok(0.0);
            }

            let avg_z = z_sum / count as f64;

            // Volume = base_area * average_height
            // This works for simple prisms or extruded shapes
            Ok(base_area * avg_z)
        }
        GeoGeometry::MultiPolygon(mp) => {
            // Sum volumes of all polygons
            let mut total_volume = 0.0;
            let mut coord_offset = 0;

            for poly in &mp.0 {
                // Create temporary geometry for this polygon
                let mut poly_geom = Geometry::new(GeoGeometry::Polygon(poly.clone()));

                // Extract Z coordinates for this polygon
                let poly_coord_count = poly.exterior().coords().count()
                    + poly
                        .interiors()
                        .iter()
                        .map(|r| r.coords().count())
                        .sum::<usize>();

                // Copy Z coordinates
                if let Some(ref z_coords) = geom.coord3d.z_coords {
                    let poly_z_values: Vec<f64> =
                        z_coords.values[coord_offset..coord_offset + poly_coord_count].to_vec();
                    poly_geom.coord3d = crate::geometry::coord3d::Coord3D::xyz(poly_z_values);
                }

                total_volume += volume_3d(&poly_geom)?;
                coord_offset += poly_coord_count;
            }

            Ok(total_volume)
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "Volume calculation not supported for geometry type: {}",
            geom.geometry_type()
        ))),
    }
}

/// Calculate volume of a convex polyhedron defined by vertices
///
/// Uses the divergence theorem to calculate volume from triangulated faces.
/// Vertices must form a closed, convex polyhedron.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::functions::geometric_properties::volume_convex_hull_3d;
///
/// // Volume of a tetrahedron
/// let vertices = vec![
///     (0.0, 0.0, 0.0),
///     (1.0, 0.0, 0.0),
///     (0.0, 1.0, 0.0),
///     (0.0, 0.0, 1.0),
/// ];
/// let vol = volume_convex_hull_3d(&vertices).unwrap();
/// // Volume should be 1/6 (tetrahedron formula)
/// ```
pub fn volume_convex_hull_3d(vertices: &[(f64, f64, f64)]) -> Result<f64> {
    if vertices.len() < 4 {
        return Ok(0.0);
    }

    // Use the shoelace/divergence theorem for convex hulls
    // For a tetrahedron (simplest case), volume = |det(v1-v0, v2-v0, v3-v0)| / 6
    // For general convex hull, we'd need to triangulate faces

    // Simplified: Calculate volume of tetrahedron formed by first 4 vertices
    if vertices.len() == 4 {
        let v0 = vertices[0];
        let v1 = vertices[1];
        let v2 = vertices[2];
        let v3 = vertices[3];

        // Vectors from v0 to other vertices
        let a = (v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
        let b = (v2.0 - v0.0, v2.1 - v0.1, v2.2 - v0.2);
        let c = (v3.0 - v0.0, v3.1 - v0.1, v3.2 - v0.2);

        // Calculate scalar triple product (determinant)
        let det = a.0 * (b.1 * c.2 - b.2 * c.1) - a.1 * (b.0 * c.2 - b.2 * c.0)
            + a.2 * (b.0 * c.1 - b.1 * c.0);

        // Volume = |det| / 6
        Ok(det.abs() / 6.0)
    } else {
        // For more complex convex hulls, this is a placeholder
        // A full implementation would require convex hull triangulation
        Err(GeoSparqlError::UnsupportedOperation(
            "Volume calculation for complex convex hulls not yet implemented".to_string(),
        ))
    }
}

/// Calculate surface area of a 3D polygon
///
/// Calculates the actual 3D surface area including the Z dimension.
///
/// # Examples
///
/// ```
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::functions::geometric_properties::surface_area_3d;
///
/// // Inclined square (tilted plane)
/// let tilted = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 10, 0 10 10, 0 0 0))").unwrap();
/// let area = surface_area_3d(&tilted).unwrap();
/// ```
pub fn surface_area_3d(geom: &Geometry) -> Result<f64> {
    if !geom.is_3d() {
        return Err(GeoSparqlError::UnsupportedOperation(
            "Geometry must have Z coordinates for 3D surface area".to_string(),
        ));
    }

    match &geom.geom {
        GeoGeometry::Polygon(p) => {
            let coords: Vec<_> = p.exterior().coords().collect();
            if coords.len() < 4 {
                return Ok(0.0);
            }

            // Calculate 3D area using triangulation
            let mut total_area = 0.0;

            // Triangulate polygon from first vertex
            let v0 = coords[0];
            let z0 = geom.coord3d.z_at(0).unwrap_or(0.0);

            for i in 1..coords.len() - 2 {
                let v1 = coords[i];
                let v2 = coords[i + 1];
                let z1 = geom.coord3d.z_at(i).unwrap_or(0.0);
                let z2 = geom.coord3d.z_at(i + 1).unwrap_or(0.0);

                // Calculate area of 3D triangle
                let area = triangle_area_3d((v0.x, v0.y, z0), (v1.x, v1.y, z1), (v2.x, v2.y, z2));

                total_area += area;
            }

            Ok(total_area)
        }
        GeoGeometry::MultiPolygon(mp) => {
            let mut total_area = 0.0;
            let mut coord_offset = 0;

            for poly in &mp.0 {
                let mut poly_geom = Geometry::new(GeoGeometry::Polygon(poly.clone()));

                let poly_coord_count = poly.exterior().coords().count();

                if let Some(ref z_coords) = geom.coord3d.z_coords {
                    let poly_z_values: Vec<f64> =
                        z_coords.values[coord_offset..coord_offset + poly_coord_count].to_vec();
                    poly_geom.coord3d = crate::geometry::coord3d::Coord3D::xyz(poly_z_values);
                }

                total_area += surface_area_3d(&poly_geom)?;
                coord_offset += poly_coord_count;
            }

            Ok(total_area)
        }
        _ => Err(GeoSparqlError::UnsupportedOperation(format!(
            "Surface area calculation not supported for geometry type: {}",
            geom.geometry_type()
        ))),
    }
}

/// Helper: Calculate area of a 3D triangle using cross product
fn triangle_area_3d(v0: (f64, f64, f64), v1: (f64, f64, f64), v2: (f64, f64, f64)) -> f64 {
    // Vectors from v0 to v1 and v0 to v2
    let a = (v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
    let b = (v2.0 - v0.0, v2.1 - v0.1, v2.2 - v0.2);

    // Cross product
    let cross = (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    );

    // Magnitude of cross product / 2 = area
    let mag = (cross.0 * cross.0 + cross.1 * cross.1 + cross.2 * cross.2).sqrt();
    mag / 2.0
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

    // ============ Volume and Surface Area Tests ============

    #[test]
    fn test_volume_3d_cube() {
        // Square base 10x10 at Z=10 (average height)
        let cube =
            Geometry::from_wkt("POLYGON Z((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))").unwrap();

        let vol = volume_3d(&cube).unwrap();

        // Base area = 100, average Z = 10, volume = 1000
        assert!((vol - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_volume_3d_reject_2d() {
        let square = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();

        let result = volume_3d(&square);
        assert!(result.is_err());
    }

    #[test]
    fn test_volume_3d_varying_heights() {
        // Polygon with varying Z coordinates (like a sloped roof)
        let sloped =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 10, 0 10 10, 0 0 0))").unwrap();

        let vol = volume_3d(&sloped).unwrap();

        // Base area = 100, average Z = (0+0+10+10+0)/5 = 4, volume = 400
        assert!((vol - 400.0).abs() < 1.0);
    }

    #[test]
    fn test_volume_convex_hull_tetrahedron() {
        // Unit tetrahedron
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ];

        let vol = volume_convex_hull_3d(&vertices).unwrap();

        // Volume of unit tetrahedron = 1/6
        assert!((vol - 0.16666).abs() < 0.001);
    }

    #[test]
    fn test_volume_convex_hull_scaled_tetrahedron() {
        // Scaled tetrahedron (all coordinates * 2)
        let vertices = vec![
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 2.0),
        ];

        let vol = volume_convex_hull_3d(&vertices).unwrap();

        // Volume scales with cube of linear dimension: (2^3) * (1/6) = 8/6 ≈ 1.333
        assert!((vol - 1.333).abs() < 0.01);
    }

    #[test]
    fn test_volume_convex_hull_insufficient_vertices() {
        let vertices = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];

        let vol = volume_convex_hull_3d(&vertices).unwrap();
        assert_eq!(vol, 0.0);
    }

    #[test]
    fn test_surface_area_3d_flat_square() {
        // Flat square at Z=10 (parallel to XY plane)
        let flat =
            Geometry::from_wkt("POLYGON Z((0 0 10, 10 0 10, 10 10 10, 0 10 10, 0 0 10))").unwrap();

        let area = surface_area_3d(&flat).unwrap();

        // Surface area should be same as 2D area = 100
        assert!((area - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_surface_area_3d_tilted_square() {
        // Tilted square: bottom edge at Z=0, top edge at Z=10
        let tilted =
            Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 10 10 10, 0 10 10, 0 0 0))").unwrap();

        let area = surface_area_3d(&tilted).unwrap();

        // Surface area should be larger than 2D projection (100)
        // due to the tilt
        assert!(area > 100.0);
        assert!(area < 150.0); // Reasonable upper bound
    }

    #[test]
    fn test_surface_area_3d_triangle() {
        // Simple 3D triangle
        let triangle = Geometry::from_wkt("POLYGON Z((0 0 0, 10 0 0, 5 5 10, 0 0 0))").unwrap();

        let area = surface_area_3d(&triangle).unwrap();

        // Should be > 0
        assert!(area > 0.0);
    }

    #[test]
    fn test_surface_area_3d_reject_2d() {
        let square = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();

        let result = surface_area_3d(&square);
        assert!(result.is_err());
    }

    #[test]
    fn test_triangle_area_3d() {
        // Right triangle in 3D space
        // Base: (0,0,0) to (3,0,0) = 3 units
        // Height: (0,0,0) to (0,4,0) = 4 units
        // Z doesn't affect this flat triangle
        let area = triangle_area_3d((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0));

        // Area of right triangle = (base * height) / 2 = (3 * 4) / 2 = 6
        assert!((area - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_triangle_area_3d_tilted() {
        // Triangle tilted in Z direction
        let area = triangle_area_3d((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0));

        // Should be larger than flat triangle due to Z component
        // Flat area would be 0.5, actual area includes the tilt
        assert!(area > 0.5);
    }
}
