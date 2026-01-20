//! GeoSPARQL filter and property functions
//!
//! This module implements the GeoSPARQL topological and geometric functions
//! for use in SPARQL queries.

pub mod bbox_utils;
pub mod buffer_3d;
pub mod coordinate_transformation;
pub mod egenhofer;
pub mod geometric_operations;
pub mod geometric_properties;
pub mod rcc8;
pub mod simple_features;
pub mod topological_3d;

#[cfg(feature = "proj-support")]
pub mod transformation_cache;

use crate::error::Result;
use crate::geometry::Geometry;

/// Trait for spatial relation predicates
pub trait SpatialRelation {
    /// Test the spatial relation between two geometries
    fn test(&self, geom1: &Geometry, geom2: &Geometry) -> Result<bool>;
}

/// Simple Features spatial relation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimpleFeatureRelation {
    /// Geometries are spatially equal
    Equals,
    /// Geometries are spatially disjoint
    Disjoint,
    /// Geometries spatially intersect
    Intersects,
    /// Geometries touch at boundaries only
    Touches,
    /// Geometries cross
    Crosses,
    /// First geometry is within the second
    Within,
    /// First geometry contains the second
    Contains,
    /// Geometries spatially overlap
    Overlaps,
}

impl SpatialRelation for SimpleFeatureRelation {
    fn test(&self, geom1: &Geometry, geom2: &Geometry) -> Result<bool> {
        // Validate CRS compatibility
        geom1.validate_crs_compatibility(geom2)?;

        match self {
            SimpleFeatureRelation::Equals => simple_features::sf_equals(geom1, geom2),
            SimpleFeatureRelation::Disjoint => simple_features::sf_disjoint(geom1, geom2),
            SimpleFeatureRelation::Intersects => simple_features::sf_intersects(geom1, geom2),
            SimpleFeatureRelation::Touches => simple_features::sf_touches(geom1, geom2),
            SimpleFeatureRelation::Crosses => simple_features::sf_crosses(geom1, geom2),
            SimpleFeatureRelation::Within => simple_features::sf_within(geom1, geom2),
            SimpleFeatureRelation::Contains => simple_features::sf_contains(geom1, geom2),
            SimpleFeatureRelation::Overlaps => simple_features::sf_overlaps(geom1, geom2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry as GeoGeometry, Point};

    #[test]
    fn test_spatial_relation_trait() {
        let point1 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)));
        let point2 = Geometry::new(GeoGeometry::Point(Point::new(2.0, 2.0)));

        let relation = SimpleFeatureRelation::Disjoint;
        let result = relation.test(&point1, &point2);
        assert!(result.is_ok());
    }
}
