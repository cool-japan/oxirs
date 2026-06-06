//! Geometric operations (buffer, convex hull, intersection, etc.)
//!
//! This module is a facade that re-exports all geometric operation functions
//! from their respective implementation modules.

use crate::error::Result;
use crate::geometry::Geometry;

// Re-export everything from sub-modules
pub use crate::functions::geometric_ops_3d::distance_3d;
pub use crate::functions::geometric_ops_buffer::{
    boundary, buffer, buffer_3d, buffer_with_params, BufferParams, CapStyle, JoinStyle,
};
pub use crate::functions::geometric_ops_set::{
    convex_hull, difference, envelope, intersection, sym_difference, union,
};

#[cfg(feature = "rust-buffer")]
pub use crate::functions::geometric_ops_buffer::buffer_rust;

/// Calculate the distance between two geometries (2D only)
///
/// For 3D distance calculations that include Z coordinates, use `distance_3d()`.
pub fn distance(geom1: &Geometry, geom2: &Geometry) -> Result<f64> {
    geom1.validate_crs_compatibility(geom2)?;

    let dist = geometry_euclidean_distance(&geom1.geom, &geom2.geom);
    Ok(dist)
}

/// Calculate Euclidean distance between two geo_types::Geometry instances
/// using the new Distance trait API with pattern matching
pub fn geometry_euclidean_distance(
    geom1: &geo_types::Geometry<f64>,
    geom2: &geo_types::Geometry<f64>,
) -> f64 {
    use geo::{Centroid, Distance, Euclidean};
    use geo_types::{Geometry as G, Point};

    // Helper to get representative point for complex geometries
    fn geometry_to_point(geom: &G<f64>) -> Option<Point<f64>> {
        match geom {
            G::Point(p) => Some(*p),
            G::Line(l) => {
                use geo::Centroid;
                Some(l.centroid())
            }
            G::LineString(ls) => ls.centroid(),
            G::Polygon(p) => p.centroid(),
            G::MultiPoint(mp) => mp.centroid(),
            G::MultiLineString(mls) => mls.centroid(),
            G::MultiPolygon(mp) => mp.centroid(),
            G::GeometryCollection(gc) => gc.centroid(),
            G::Rect(r) => Some(r.centroid()),
            G::Triangle(t) => Some(t.centroid()),
        }
    }

    // For Point-to-Point, use exact distance
    if let (G::Point(p1), G::Point(p2)) = (geom1, geom2) {
        return Euclidean.distance(*p1, *p2);
    }

    // For other combinations, use centroid-based distance
    // This is an approximation suitable for k-NN and similar operations
    match (geometry_to_point(geom1), geometry_to_point(geom2)) {
        (Some(p1), Some(p2)) => Euclidean.distance(p1, p2),
        _ => f64::INFINITY, // Empty geometries
    }
}
