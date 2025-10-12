//! # OxiRS GeoSPARQL
//!
//! GeoSPARQL implementation for spatial data and queries in RDF/SPARQL.
//!
//! This crate provides:
//! - GeoSPARQL vocabulary and datatypes
//! - WKT (Well-Known Text) geometry parsing and serialization
//! - Simple Features topological relations (sfEquals, sfContains, sfIntersects, etc.)
//! - Geometric operations (buffer, convex hull, intersection, etc.)
//! - Geometric properties (dimension, SRID, isEmpty, etc.)
//! - Spatial indexing with R-tree for efficient queries
//! - **SIMD-accelerated distance calculations** (2-4x speedup)
//! - **Parallel batch processing** for large datasets
//!
//! ## Example
//!
//! ```rust
//! use oxirs_geosparql::geometry::Geometry;
//! use oxirs_geosparql::functions::simple_features;
//!
//! // Parse WKT geometries
//! let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
//! let polygon = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
//!
//! // Test spatial relations
//! let contains = simple_features::sf_contains(&polygon, &point).unwrap();
//! assert!(contains);
//! ```
//!
//! ## GeoSPARQL Compliance
//!
//! This implementation follows the OGC GeoSPARQL 1.0/1.1 specification:
//! - Core: Geometry classes and properties
//! - Topology Vocabulary: Simple Features relations
//! - Geometry Extension: WKT and GML support
//! - Query Rewrite Extension: Spatial indexing
//!
//! ## Features
//!
//! - `wkt-support` (default): WKT parsing and serialization
//! - `geojson-support`: GeoJSON support
//! - `geos-backend`: Use GEOS library for geometric operations
//! - `proj-support`: Coordinate transformation support
//! - `parallel`: Parallel processing for large datasets

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod functions;
pub mod geometry;
pub mod index;
pub mod performance;
pub mod vocabulary;

// Re-export commonly used types
pub use error::{GeoSparqlError, Result};
pub use geometry::{Crs, Geometry};
pub use index::SpatialIndex;
pub use performance::BatchProcessor;

/// GeoSPARQL function registry
///
/// This registry contains all available GeoSPARQL filter functions (predicates)
/// and property functions (geometric operations).
pub struct GeoSparqlRegistry;

impl GeoSparqlRegistry {
    /// Get all available Simple Features topological relation functions
    ///
    /// Returns a list of (function_uri, function_name) tuples
    pub fn simple_features_functions() -> Vec<(&'static str, &'static str)> {
        vec![
            (vocabulary::GEO_SF_EQUALS, "sfEquals"),
            (vocabulary::GEO_SF_DISJOINT, "sfDisjoint"),
            (vocabulary::GEO_SF_INTERSECTS, "sfIntersects"),
            (vocabulary::GEO_SF_TOUCHES, "sfTouches"),
            (vocabulary::GEO_SF_CROSSES, "sfCrosses"),
            (vocabulary::GEO_SF_WITHIN, "sfWithin"),
            (vocabulary::GEO_SF_CONTAINS, "sfContains"),
            (vocabulary::GEO_SF_OVERLAPS, "sfOverlaps"),
        ]
    }

    /// Get all available Egenhofer topological relation functions
    ///
    /// Returns a list of (function_uri, function_name) tuples
    #[cfg(feature = "geos-backend")]
    pub fn egenhofer_functions() -> Vec<(&'static str, &'static str)> {
        vec![
            (vocabulary::GEO_EH_EQUALS, "ehEquals"),
            (vocabulary::GEO_EH_DISJOINT, "ehDisjoint"),
            (vocabulary::GEO_EH_MEET, "ehMeet"),
            (vocabulary::GEO_EH_OVERLAP, "ehOverlap"),
            (vocabulary::GEO_EH_COVERS, "ehCovers"),
            (vocabulary::GEO_EH_COVERED_BY, "ehCoveredBy"),
            (vocabulary::GEO_EH_INSIDE, "ehInside"),
            (vocabulary::GEO_EH_CONTAINS, "ehContains"),
        ]
    }

    /// Get all available RCC8 topological relation functions
    ///
    /// Returns a list of (function_uri, function_name) tuples
    #[cfg(feature = "geos-backend")]
    pub fn rcc8_functions() -> Vec<(&'static str, &'static str)> {
        vec![
            (vocabulary::GEO_RCC8_EQ, "rcc8eq"),
            (vocabulary::GEO_RCC8_DC, "rcc8dc"),
            (vocabulary::GEO_RCC8_EC, "rcc8ec"),
            (vocabulary::GEO_RCC8_PO, "rcc8po"),
            (vocabulary::GEO_RCC8_TPPI, "rcc8tppi"),
            (vocabulary::GEO_RCC8_TPP, "rcc8tpp"),
            (vocabulary::GEO_RCC8_NTPP, "rcc8ntpp"),
            (vocabulary::GEO_RCC8_NTPPI, "rcc8ntppi"),
        ]
    }

    /// Get all available geometric property functions
    ///
    /// Returns a list of (function_uri, function_name) tuples
    pub fn property_functions() -> Vec<(&'static str, &'static str)> {
        vec![
            (vocabulary::GEO_DIMENSION, "dimension"),
            (vocabulary::GEO_COORDINATE_DIMENSION, "coordinateDimension"),
            (vocabulary::GEO_SPATIAL_DIMENSION, "spatialDimension"),
            (vocabulary::GEO_IS_EMPTY, "isEmpty"),
            (vocabulary::GEO_IS_SIMPLE, "isSimple"),
            (vocabulary::GEO_AS_WKT, "asWKT"),
        ]
    }

    /// Get all available geometric operation functions
    ///
    /// Returns a list of (function_uri, function_name) tuples
    #[cfg(feature = "geos-backend")]
    pub fn operation_functions() -> Vec<(&'static str, &'static str)> {
        vec![
            (vocabulary::GEO_BUFFER, "buffer"),
            (vocabulary::GEO_CONVEX_HULL, "convexHull"),
            (vocabulary::GEO_INTERSECTION, "intersection"),
            (vocabulary::GEO_UNION, "union"),
            (vocabulary::GEO_DIFFERENCE, "difference"),
            (vocabulary::GEO_SYM_DIFFERENCE, "symDifference"),
            (vocabulary::GEO_ENVELOPE, "envelope"),
            (vocabulary::GEO_BOUNDARY, "boundary"),
        ]
    }

    /// Get all available distance functions
    ///
    /// Returns a list of (function_uri, function_name) tuples
    pub fn distance_functions() -> Vec<(&'static str, &'static str)> {
        vec![(vocabulary::GEO_DISTANCE, "distance")]
    }

    /// Get all GeoSPARQL extension functions (filter functions)
    ///
    /// Returns all boolean predicates that can be used in SPARQL FILTER clauses
    pub fn all_filter_functions() -> Vec<(&'static str, &'static str)> {
        #[cfg(not(feature = "geos-backend"))]
        {
            Self::simple_features_functions()
        }

        #[cfg(feature = "geos-backend")]
        {
            let mut functions = Self::simple_features_functions();
            functions.extend(Self::egenhofer_functions());
            functions.extend(Self::rcc8_functions());
            functions
        }
    }

    /// Get all GeoSPARQL property functions
    ///
    /// Returns all functions that compute geometric properties or perform operations
    pub fn all_property_functions() -> Vec<(&'static str, &'static str)> {
        let mut functions = Self::property_functions();
        functions.extend(Self::distance_functions());

        #[cfg(feature = "geos-backend")]
        {
            functions.extend(Self::operation_functions());
        }

        functions
    }
}

/// Type alias for GeoSPARQL function registry result
type GeoSparqlFunctions = (
    Vec<(&'static str, &'static str)>,
    Vec<(&'static str, &'static str)>,
);

/// Initialize GeoSPARQL support in a SPARQL engine
///
/// This function should be called to register GeoSPARQL functions
/// with the SPARQL query engine.
///
/// # Returns
///
/// Returns a tuple of (filter_functions, property_functions) where:
/// - filter_functions: Boolean predicates for FILTER clauses
/// - property_functions: Functions that compute properties or perform operations
pub fn init() -> GeoSparqlFunctions {
    tracing::info!("Initializing GeoSPARQL support");

    let filter_functions = GeoSparqlRegistry::all_filter_functions();
    let property_functions = GeoSparqlRegistry::all_property_functions();

    tracing::info!(
        "Registered {} GeoSPARQL filter functions",
        filter_functions.len()
    );
    tracing::info!(
        "Registered {} GeoSPARQL property functions",
        property_functions.len()
    );

    (filter_functions, property_functions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let (filter_functions, property_functions) = init();

        // Check that we have the expected number of functions
        #[cfg(feature = "geos-backend")]
        {
            assert_eq!(filter_functions.len(), 24); // 8 SF + 8 Egenhofer + 8 RCC8
            assert_eq!(property_functions.len(), 15); // 6 props + 1 distance + 8 ops
        }
        #[cfg(not(feature = "geos-backend"))]
        {
            assert_eq!(filter_functions.len(), 8); // 8 Simple Features functions
            assert_eq!(property_functions.len(), 7); // 6 properties + 1 distance
        }
    }

    #[test]
    fn test_simple_features_registry() {
        let functions = GeoSparqlRegistry::simple_features_functions();

        assert_eq!(functions.len(), 8);

        // Verify some key functions are present
        assert!(functions.contains(&(vocabulary::GEO_SF_EQUALS, "sfEquals")));
        assert!(functions.contains(&(vocabulary::GEO_SF_CONTAINS, "sfContains")));
        assert!(functions.contains(&(vocabulary::GEO_SF_INTERSECTS, "sfIntersects")));
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_egenhofer_registry() {
        let functions = GeoSparqlRegistry::egenhofer_functions();

        assert_eq!(functions.len(), 8);

        // Verify some key functions are present
        assert!(functions.contains(&(vocabulary::GEO_EH_EQUALS, "ehEquals")));
        assert!(functions.contains(&(vocabulary::GEO_EH_MEET, "ehMeet")));
        assert!(functions.contains(&(vocabulary::GEO_EH_CONTAINS, "ehContains")));
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_rcc8_registry() {
        let functions = GeoSparqlRegistry::rcc8_functions();

        assert_eq!(functions.len(), 8);

        // Verify some key functions are present
        assert!(functions.contains(&(vocabulary::GEO_RCC8_EQ, "rcc8eq")));
        assert!(functions.contains(&(vocabulary::GEO_RCC8_DC, "rcc8dc")));
        assert!(functions.contains(&(vocabulary::GEO_RCC8_PO, "rcc8po")));
    }

    #[test]
    fn test_property_functions_registry() {
        let functions = GeoSparqlRegistry::property_functions();

        assert_eq!(functions.len(), 6);

        // Verify some key properties are present
        assert!(functions.contains(&(vocabulary::GEO_DIMENSION, "dimension")));
        assert!(functions.contains(&(vocabulary::GEO_IS_EMPTY, "isEmpty")));
        assert!(functions.contains(&(vocabulary::GEO_IS_SIMPLE, "isSimple")));
    }

    #[test]
    #[cfg(feature = "geos-backend")]
    fn test_operation_functions_registry() {
        let functions = GeoSparqlRegistry::operation_functions();

        assert_eq!(functions.len(), 8);

        // Verify some key operations are present
        assert!(functions.contains(&(vocabulary::GEO_BUFFER, "buffer")));
        assert!(functions.contains(&(vocabulary::GEO_INTERSECTION, "intersection")));
        assert!(functions.contains(&(vocabulary::GEO_UNION, "union")));
    }

    #[test]
    fn test_distance_functions_registry() {
        let functions = GeoSparqlRegistry::distance_functions();

        assert_eq!(functions.len(), 1);
        assert!(functions.contains(&(vocabulary::GEO_DISTANCE, "distance")));
    }

    #[test]
    fn test_all_filter_functions() {
        let functions = GeoSparqlRegistry::all_filter_functions();

        // Should have at least the Simple Features functions
        assert!(functions.len() >= 8);

        // Verify all Simple Features functions are included
        assert!(functions.contains(&(vocabulary::GEO_SF_EQUALS, "sfEquals")));
        assert!(functions.contains(&(vocabulary::GEO_SF_DISJOINT, "sfDisjoint")));
        assert!(functions.contains(&(vocabulary::GEO_SF_INTERSECTS, "sfIntersects")));
        assert!(functions.contains(&(vocabulary::GEO_SF_TOUCHES, "sfTouches")));
        assert!(functions.contains(&(vocabulary::GEO_SF_CROSSES, "sfCrosses")));
        assert!(functions.contains(&(vocabulary::GEO_SF_WITHIN, "sfWithin")));
        assert!(functions.contains(&(vocabulary::GEO_SF_CONTAINS, "sfContains")));
        assert!(functions.contains(&(vocabulary::GEO_SF_OVERLAPS, "sfOverlaps")));
    }

    #[test]
    fn test_all_property_functions() {
        let functions = GeoSparqlRegistry::all_property_functions();

        // Should have at least the basic property functions + distance
        assert!(functions.len() >= 7);

        // Verify key property functions are included
        assert!(functions.contains(&(vocabulary::GEO_DIMENSION, "dimension")));
        assert!(functions.contains(&(vocabulary::GEO_DISTANCE, "distance")));
    }
}
