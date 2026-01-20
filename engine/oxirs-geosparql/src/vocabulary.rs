//! GeoSPARQL vocabulary and namespace definitions
//!
//! This module defines the standard GeoSPARQL vocabulary terms,
//! following the OGC GeoSPARQL 1.0/1.1 specification.

/// GeoSPARQL namespace prefix
pub const GEOSPARQL_PREFIX: &str = "geo";

/// GeoSPARQL namespace URI
pub const GEOSPARQL_NS: &str = "http://www.opengis.net/ont/geosparql#";

/// GeoSPARQL filter functions namespace URI
pub const GEOF_NS: &str = "http://www.opengis.net/def/function/geosparql/";

/// GeoSPARQL Simple Features namespace URI
pub const SF_NS: &str = "http://www.opengis.net/ont/sf#";

/// Well-Known Text namespace URI
pub const WKT_NS: &str = "http://www.opengis.net/ont/geosparql#wktLiteral";

/// Geography Markup Language namespace URI
pub const GML_NS: &str = "http://www.opengis.net/ont/geosparql#gmlLiteral";

/// Spatial Reference System URI prefix
pub const SRS_PREFIX: &str = "http://www.opengis.net/def/crs/";

/// Default Coordinate Reference System (WGS84)
pub const DEFAULT_CRS: &str = "http://www.opengis.net/def/crs/OGC/1.3/CRS84";

/// EPSG CRS prefix
pub const EPSG_PREFIX: &str = "http://www.opengis.net/def/crs/EPSG/0/";

// OGC GeoSPARQL Core Class and Datatype URIs

/// GeoSPARQL WKT Literal datatype
pub const GEO_WKT_LITERAL: &str = "http://www.opengis.net/ont/geosparql#wktLiteral";

/// GeoSPARQL GML Literal datatype
pub const GEO_GML_LITERAL: &str = "http://www.opengis.net/ont/geosparql#gmlLiteral";

/// GeoSPARQL Geometry class
pub const GEO_GEOMETRY: &str = "http://www.opengis.net/ont/geosparql#Geometry";

/// GeoSPARQL Feature class
pub const GEO_FEATURE: &str = "http://www.opengis.net/ont/geosparql#Feature";

/// GeoSPARQL SpatialObject class
pub const GEO_SPATIAL_OBJECT: &str = "http://www.opengis.net/ont/geosparql#SpatialObject";

// Full URI constants for commonly used functions (for function registration)

/// Simple Features equals relation (full URI)
pub const GEO_SF_EQUALS: &str = "http://www.opengis.net/def/function/geosparql/sfEquals";
/// Simple Features disjoint relation (full URI)
pub const GEO_SF_DISJOINT: &str = "http://www.opengis.net/def/function/geosparql/sfDisjoint";
/// Simple Features intersects relation (full URI)
pub const GEO_SF_INTERSECTS: &str = "http://www.opengis.net/def/function/geosparql/sfIntersects";
/// Simple Features touches relation (full URI)
pub const GEO_SF_TOUCHES: &str = "http://www.opengis.net/def/function/geosparql/sfTouches";
/// Simple Features crosses relation (full URI)
pub const GEO_SF_CROSSES: &str = "http://www.opengis.net/def/function/geosparql/sfCrosses";
/// Simple Features within relation (full URI)
pub const GEO_SF_WITHIN: &str = "http://www.opengis.net/def/function/geosparql/sfWithin";
/// Simple Features contains relation (full URI)
pub const GEO_SF_CONTAINS: &str = "http://www.opengis.net/def/function/geosparql/sfContains";
/// Simple Features overlaps relation (full URI)
pub const GEO_SF_OVERLAPS: &str = "http://www.opengis.net/def/function/geosparql/sfOverlaps";

/// Egenhofer equals relation (full URI)
pub const GEO_EH_EQUALS: &str = "http://www.opengis.net/def/function/geosparql/ehEquals";
/// Egenhofer disjoint relation (full URI)
pub const GEO_EH_DISJOINT: &str = "http://www.opengis.net/def/function/geosparql/ehDisjoint";
/// Egenhofer meet relation (full URI)
pub const GEO_EH_MEET: &str = "http://www.opengis.net/def/function/geosparql/ehMeet";
/// Egenhofer overlap relation (full URI)
pub const GEO_EH_OVERLAP: &str = "http://www.opengis.net/def/function/geosparql/ehOverlap";
/// Egenhofer covers relation (full URI)
pub const GEO_EH_COVERS: &str = "http://www.opengis.net/def/function/geosparql/ehCovers";
/// Egenhofer covered-by relation (full URI)
pub const GEO_EH_COVERED_BY: &str = "http://www.opengis.net/def/function/geosparql/ehCoveredBy";
/// Egenhofer inside relation (full URI)
pub const GEO_EH_INSIDE: &str = "http://www.opengis.net/def/function/geosparql/ehInside";
/// Egenhofer contains relation (full URI)
pub const GEO_EH_CONTAINS: &str = "http://www.opengis.net/def/function/geosparql/ehContains";

/// RCC8 equal relation (full URI)
pub const GEO_RCC8_EQ: &str = "http://www.opengis.net/def/function/geosparql/rcc8eq";
/// RCC8 disconnected relation (full URI)
pub const GEO_RCC8_DC: &str = "http://www.opengis.net/def/function/geosparql/rcc8dc";
/// RCC8 externally connected relation (full URI)
pub const GEO_RCC8_EC: &str = "http://www.opengis.net/def/function/geosparql/rcc8ec";
/// RCC8 partially overlapping relation (full URI)
pub const GEO_RCC8_PO: &str = "http://www.opengis.net/def/function/geosparql/rcc8po";
/// RCC8 tangential proper part inverse relation (full URI)
pub const GEO_RCC8_TPPI: &str = "http://www.opengis.net/def/function/geosparql/rcc8tppi";
/// RCC8 tangential proper part relation (full URI)
pub const GEO_RCC8_TPP: &str = "http://www.opengis.net/def/function/geosparql/rcc8tpp";
/// RCC8 non-tangential proper part relation (full URI)
pub const GEO_RCC8_NTPP: &str = "http://www.opengis.net/def/function/geosparql/rcc8ntpp";
/// RCC8 non-tangential proper part inverse relation (full URI)
pub const GEO_RCC8_NTPPI: &str = "http://www.opengis.net/def/function/geosparql/rcc8ntppi";

/// Distance function (full URI)
pub const GEO_DISTANCE: &str = "http://www.opengis.net/def/function/geosparql/distance";
/// Buffer function (full URI)
pub const GEO_BUFFER: &str = "http://www.opengis.net/def/function/geosparql/buffer";
/// Convex hull function (full URI)
pub const GEO_CONVEX_HULL: &str = "http://www.opengis.net/def/function/geosparql/convexHull";
/// Intersection function (full URI)
pub const GEO_INTERSECTION: &str = "http://www.opengis.net/def/function/geosparql/intersection";
/// Union function (full URI)
pub const GEO_UNION: &str = "http://www.opengis.net/def/function/geosparql/union";
/// Difference function (full URI)
pub const GEO_DIFFERENCE: &str = "http://www.opengis.net/def/function/geosparql/difference";
/// Symmetric difference function (full URI)
pub const GEO_SYM_DIFFERENCE: &str = "http://www.opengis.net/def/function/geosparql/symDifference";
/// Envelope function (full URI)
pub const GEO_ENVELOPE: &str = "http://www.opengis.net/def/function/geosparql/envelope";
/// Boundary function (full URI)
pub const GEO_BOUNDARY: &str = "http://www.opengis.net/def/function/geosparql/boundary";

/// Dimension property (full URI)
pub const GEO_DIMENSION: &str = "http://www.opengis.net/ont/geosparql#dimension";
/// Coordinate dimension property (full URI)
pub const GEO_COORDINATE_DIMENSION: &str =
    "http://www.opengis.net/ont/geosparql#coordinateDimension";
/// Spatial dimension property (full URI)
pub const GEO_SPATIAL_DIMENSION: &str = "http://www.opengis.net/ont/geosparql#spatialDimension";
/// Is empty property (full URI)
pub const GEO_IS_EMPTY: &str = "http://www.opengis.net/ont/geosparql#isEmpty";
/// Is simple property (full URI)
pub const GEO_IS_SIMPLE: &str = "http://www.opengis.net/ont/geosparql#isSimple";
/// As WKT property (full URI)
pub const GEO_AS_WKT: &str = "http://www.opengis.net/ont/geosparql#asWKT";

/// GeoSPARQL core vocabulary terms
pub mod core {
    use super::GEOSPARQL_NS;

    /// Returns the full URI for a GeoSPARQL term
    #[inline]
    pub fn term(local_name: &str) -> String {
        format!("{}{}", GEOSPARQL_NS, local_name)
    }

    // Classes
    /// GeoSPARQL Feature class
    pub const FEATURE: &str = "Feature";
    /// GeoSPARQL Geometry class
    pub const GEOMETRY: &str = "Geometry";
    /// GeoSPARQL SpatialObject class
    pub const SPATIAL_OBJECT: &str = "SpatialObject";

    // Properties
    /// Property relating a feature to its geometry
    pub const HAS_GEOMETRY: &str = "hasGeometry";
    /// Property relating a feature to its default geometry
    pub const HAS_DEFAULT_GEOMETRY: &str = "hasDefaultGeometry";
    /// Property for WKT serialization of a geometry
    pub const AS_WKT: &str = "asWKT";
    /// Property for GML serialization of a geometry
    pub const AS_GML: &str = "asGML";
    /// Property for the dimension of a geometry
    pub const DIMENSION: &str = "dimension";
    /// Property for the coordinate dimension of a geometry
    pub const COORDINATE_DIMENSION: &str = "coordinateDimension";
    /// Property for the spatial dimension of a geometry
    pub const SPATIAL_DIMENSION: &str = "spatialDimension";
    /// Property indicating if a geometry is empty
    pub const IS_EMPTY: &str = "isEmpty";
    /// Property indicating if a geometry is simple
    pub const IS_SIMPLE: &str = "isSimple";
    /// Property for geometry serialization
    pub const HAS_SERIALIZATION: &str = "hasSerialization";

    // Geometry types (Simple Features)
    /// Point geometry type
    pub const POINT: &str = "Point";
    /// LineString geometry type
    pub const LINE_STRING: &str = "LineString";
    /// Polygon geometry type
    pub const POLYGON: &str = "Polygon";
    /// MultiPoint geometry type
    pub const MULTI_POINT: &str = "MultiPoint";
    /// MultiLineString geometry type
    pub const MULTI_LINE_STRING: &str = "MultiLineString";
    /// MultiPolygon geometry type
    pub const MULTI_POLYGON: &str = "MultiPolygon";
    /// GeometryCollection geometry type
    pub const GEOMETRY_COLLECTION: &str = "GeometryCollection";
}

/// GeoSPARQL filter functions
pub mod functions {
    use super::GEOF_NS;

    /// Returns the full URI for a GeoSPARQL filter function
    #[inline]
    pub fn term(local_name: &str) -> String {
        format!("{}{}", GEOF_NS, local_name)
    }

    // Simple Features Topological Relations (DE-9IM)
    /// Simple Features equals relation - geometries are spatially equal
    pub const SF_EQUALS: &str = "sfEquals";
    /// Simple Features disjoint relation - geometries are spatially disjoint
    pub const SF_DISJOINT: &str = "sfDisjoint";
    /// Simple Features intersects relation - geometries spatially intersect
    pub const SF_INTERSECTS: &str = "sfIntersects";
    /// Simple Features touches relation - geometries touch at boundaries only
    pub const SF_TOUCHES: &str = "sfTouches";
    /// Simple Features crosses relation - geometries cross each other
    pub const SF_CROSSES: &str = "sfCrosses";
    /// Simple Features within relation - first geometry is within the second
    pub const SF_WITHIN: &str = "sfWithin";
    /// Simple Features contains relation - first geometry contains the second
    pub const SF_CONTAINS: &str = "sfContains";
    /// Simple Features overlaps relation - geometries spatially overlap
    pub const SF_OVERLAPS: &str = "sfOverlaps";

    // Egenhofer Topological Relations
    /// Egenhofer equals relation
    pub const EH_EQUALS: &str = "ehEquals";
    /// Egenhofer disjoint relation
    pub const EH_DISJOINT: &str = "ehDisjoint";
    /// Egenhofer meet relation
    pub const EH_MEET: &str = "ehMeet";
    /// Egenhofer overlap relation
    pub const EH_OVERLAP: &str = "ehOverlap";
    /// Egenhofer covers relation
    pub const EH_COVERS: &str = "ehCovers";
    /// Egenhofer covered-by relation
    pub const EH_COVERED_BY: &str = "ehCoveredBy";
    /// Egenhofer inside relation
    pub const EH_INSIDE: &str = "ehInside";
    /// Egenhofer contains relation
    pub const EH_CONTAINS: &str = "ehContains";

    // RCC8 Topological Relations
    /// RCC8 equal relation
    pub const RCC8_EQ: &str = "rcc8eq";
    /// RCC8 disconnected relation
    pub const RCC8_DC: &str = "rcc8dc";
    /// RCC8 externally connected relation
    pub const RCC8_EC: &str = "rcc8ec";
    /// RCC8 partially overlapping relation
    pub const RCC8_PO: &str = "rcc8po";
    /// RCC8 tangential proper part inverse relation
    pub const RCC8_TPPI: &str = "rcc8tppi";
    /// RCC8 tangential proper part relation
    pub const RCC8_TPP: &str = "rcc8tpp";
    /// RCC8 non-tangential proper part relation
    pub const RCC8_NTPP: &str = "rcc8ntpp";
    /// RCC8 non-tangential proper part inverse relation
    pub const RCC8_NTPPI: &str = "rcc8ntppi";

    // Non-topological Query Functions
    /// Calculate distance between two geometries
    pub const DISTANCE: &str = "distance";
    /// Create a buffer around a geometry
    pub const BUFFER: &str = "buffer";
    /// Calculate the convex hull of a geometry
    pub const CONVEX_HULL: &str = "convexHull";
    /// Calculate the intersection of two geometries
    pub const INTERSECTION: &str = "intersection";
    /// Calculate the union of two geometries
    pub const UNION: &str = "union";
    /// Calculate the difference of two geometries
    pub const DIFFERENCE: &str = "difference";
    /// Calculate the symmetric difference of two geometries
    pub const SYM_DIFFERENCE: &str = "symDifference";
    /// Calculate the envelope (bounding box) of a geometry
    pub const ENVELOPE: &str = "envelope";
    /// Calculate the boundary of a geometry
    pub const BOUNDARY: &str = "boundary";

    // Geometric Property Functions
    /// Get the SRID (Spatial Reference ID) of a geometry
    pub const GET_SRID: &str = "getSRID";
    /// Get the dimension of a geometry
    pub const DIMENSION: &str = "dimension";
    /// Get the coordinate dimension of a geometry
    pub const COORDINATE_DIMENSION: &str = "coordinateDimension";
    /// Get the spatial dimension of a geometry
    pub const SPATIAL_DIMENSION: &str = "spatialDimension";
    /// Check if a geometry is empty
    pub const IS_EMPTY: &str = "isEmpty";
    /// Check if a geometry is simple (no self-intersections)
    pub const IS_SIMPLE: &str = "isSimple";
    /// Check if coordinates are 3D (has Z coordinate)
    pub const IS_3D: &str = "is3D";
    /// Check if coordinates are measured (has M coordinate)
    pub const IS_MEASURED: &str = "isMeasured";

    // Common Functions
    /// Test the spatial relation using DE-9IM pattern
    pub const RELATE: &str = "relate";
    /// Convert geometry to WKT format
    pub const AS_WKT: &str = "asWKT";
    /// Convert geometry to GML format
    pub const AS_GML: &str = "asGML";
}

/// GeoSPARQL property functions (for use with FILTER)
pub mod properties {
    use super::GEOSPARQL_NS;

    /// Returns the full URI for a GeoSPARQL property function
    #[inline]
    pub fn term(local_name: &str) -> String {
        format!("{}{}", GEOSPARQL_NS, local_name)
    }

    // Simple Features Relations
    /// Simple Features equals relation property
    pub const SF_EQUALS: &str = "sfEquals";
    /// Simple Features disjoint relation property
    pub const SF_DISJOINT: &str = "sfDisjoint";
    /// Simple Features intersects relation property
    pub const SF_INTERSECTS: &str = "sfIntersects";
    /// Simple Features touches relation property
    pub const SF_TOUCHES: &str = "sfTouches";
    /// Simple Features crosses relation property
    pub const SF_CROSSES: &str = "sfCrosses";
    /// Simple Features within relation property
    pub const SF_WITHIN: &str = "sfWithin";
    /// Simple Features contains relation property
    pub const SF_CONTAINS: &str = "sfContains";
    /// Simple Features overlaps relation property
    pub const SF_OVERLAPS: &str = "sfOverlaps";

    // Egenhofer Topological Relations
    /// Egenhofer equals relation
    pub const EH_EQUALS: &str = "ehEquals";
    /// Egenhofer disjoint relation
    pub const EH_DISJOINT: &str = "ehDisjoint";
    /// Egenhofer meet relation
    pub const EH_MEET: &str = "ehMeet";
    /// Egenhofer overlap relation
    pub const EH_OVERLAP: &str = "ehOverlap";
    /// Egenhofer covers relation
    pub const EH_COVERS: &str = "ehCovers";
    /// Egenhofer covered-by relation
    pub const EH_COVERED_BY: &str = "ehCoveredBy";
    /// Egenhofer inside relation
    pub const EH_INSIDE: &str = "ehInside";
    /// Egenhofer contains relation
    pub const EH_CONTAINS: &str = "ehContains";

    // RCC8 Topological Relations
    /// RCC8 equal relation
    pub const RCC8_EQ: &str = "rcc8eq";
    /// RCC8 disconnected relation
    pub const RCC8_DC: &str = "rcc8dc";
    /// RCC8 externally connected relation
    pub const RCC8_EC: &str = "rcc8ec";
    /// RCC8 partially overlapping relation
    pub const RCC8_PO: &str = "rcc8po";
    /// RCC8 tangential proper part inverse relation
    pub const RCC8_TPPI: &str = "rcc8tppi";
    /// RCC8 tangential proper part relation
    pub const RCC8_TPP: &str = "rcc8tpp";
    /// RCC8 non-tangential proper part relation
    pub const RCC8_NTPP: &str = "rcc8ntpp";
    /// RCC8 non-tangential proper part inverse relation
    pub const RCC8_NTPPI: &str = "rcc8ntppi";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_uris() {
        assert_eq!(GEOSPARQL_NS, "http://www.opengis.net/ont/geosparql#");
        assert_eq!(GEOF_NS, "http://www.opengis.net/def/function/geosparql/");
        assert_eq!(SF_NS, "http://www.opengis.net/ont/sf#");
    }

    #[test]
    fn test_core_terms() {
        assert_eq!(
            core::term(core::FEATURE),
            "http://www.opengis.net/ont/geosparql#Feature"
        );
        assert_eq!(
            core::term(core::GEOMETRY),
            "http://www.opengis.net/ont/geosparql#Geometry"
        );
        assert_eq!(
            core::term(core::HAS_GEOMETRY),
            "http://www.opengis.net/ont/geosparql#hasGeometry"
        );
    }

    #[test]
    fn test_function_terms() {
        assert_eq!(
            functions::term(functions::SF_CONTAINS),
            "http://www.opengis.net/def/function/geosparql/sfContains"
        );
        assert_eq!(
            functions::term(functions::SF_INTERSECTS),
            "http://www.opengis.net/def/function/geosparql/sfIntersects"
        );
    }

    #[test]
    fn test_property_terms() {
        assert_eq!(
            properties::term(properties::SF_CONTAINS),
            "http://www.opengis.net/ont/geosparql#sfContains"
        );
        assert_eq!(
            properties::term(properties::SF_INTERSECTS),
            "http://www.opengis.net/ont/geosparql#sfIntersects"
        );
    }
}
