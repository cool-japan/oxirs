//! RDF serialization and deserialization for GeoSPARQL geometries
//!
//! This module provides functionality to serialize geometries to RDF formats
//! (Turtle, N-Triples, N-Quads) and deserialize from RDF.
//!
//! # GeoSPARQL RDF Representation
//!
//! In GeoSPARQL, geometries are represented as RDF literals with the following structure:
//!
//! ```turtle
//! @prefix geo: <http://www.opengis.net/ont/geosparql#> .
//! @prefix sf: <http://www.opengis.net/ont/sf#> .
//! @prefix geof: <http://www.opengis.net/def/function/geosparql/> .
//! @prefix uom: <http://www.opengis.net/def/uom/OGC/1.0/> .
//!
//! :feature1
//!     a geo:Feature ;
//!     geo:hasGeometry :feature1_geometry .
//!
//! :feature1_geometry
//!     a sf:Point ;
//!     geo:asWKT "POINT(1.0 2.0)"^^geo:wktLiteral ;
//!     geo:coordinateDimension 2 ;
//!     geo:dimension 0 ;
//!     geo:isEmpty false ;
//!     geo:isSimple true .
//! ```
//!
//! # Examples
//!
//! ## Turtle Serialization
//!
//! ```rust,ignore
//! use oxirs_geosparql::geometry::Geometry;
//! use oxirs_geosparql::geometry::rdf_serialization::to_turtle;
//!
//! let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
//! let turtle = to_turtle(&point, ":feature1_geometry").unwrap();
//!
//! assert!(turtle.contains("geo:asWKT"));
//! assert!(turtle.contains("POINT(1.0 2.0)"));
//! ```

use crate::error::Result;
use crate::geometry::Geometry;
use std::fmt::Write;

/// RDF serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfFormat {
    /// Turtle format (<https://www.w3.org/TR/turtle/>)
    Turtle,
    /// N-Triples format (<https://www.w3.org/TR/n-triples/>)
    NTriples,
    /// N-Quads format (<https://www.w3.org/TR/n-quads/>)
    NQuads,
}

/// Serialize a geometry to RDF Turtle format
///
/// # Arguments
///
/// * `geom` - The geometry to serialize
/// * `subject_uri` - The URI/identifier for this geometry resource
///
/// # Returns
///
/// A Turtle-formatted string representing the geometry as RDF triples
///
/// # Examples
///
/// ```rust,ignore
/// use oxirs_geosparql::geometry::Geometry;
/// use oxirs_geosparql::geometry::rdf_serialization::to_turtle;
///
/// let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
/// let turtle = to_turtle(&point, ":myPoint").unwrap();
///
/// assert!(turtle.contains("geo:asWKT"));
/// assert!(turtle.contains("sf:Point"));
/// ```
pub fn to_turtle(geom: &Geometry, subject_uri: &str) -> Result<String> {
    let mut output = String::new();

    // Write prefixes
    writeln!(
        output,
        "@prefix geo: <http://www.opengis.net/ont/geosparql#> ."
    )
    .unwrap();
    writeln!(output, "@prefix sf: <http://www.opengis.net/ont/sf#> .").unwrap();
    writeln!(output, "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .").unwrap();
    writeln!(output).unwrap();

    // Write geometry resource
    writeln!(output, "{} ", subject_uri).unwrap();
    writeln!(output, "    a {} ;", geometry_type_to_sf(geom)).unwrap();
    writeln!(
        output,
        "    geo:asWKT \"{}\"^^geo:wktLiteral ;",
        geom.to_wkt()
    )
    .unwrap();
    writeln!(
        output,
        "    geo:coordinateDimension {} ;",
        geom.coordinate_dimension()
    )
    .unwrap();
    writeln!(output, "    geo:dimension {} ;", geom.spatial_dimension()).unwrap();
    writeln!(output, "    geo:isEmpty {} ;", geom.is_empty()).unwrap();
    writeln!(output, "    geo:isSimple {} .", geom.is_simple()).unwrap();

    // Add CRS information if not default
    if !geom.crs.is_default() {
        writeln!(output, "    geo:hasSerialization \"{}\" .", geom.crs.uri).unwrap();
    }

    Ok(output)
}

/// Serialize a geometry to RDF N-Triples format
///
/// N-Triples is a line-based format where each line represents one RDF triple.
///
/// # Arguments
///
/// * `geom` - The geometry to serialize
/// * `subject_uri` - The full URI for this geometry resource (must be absolute)
///
/// # Returns
///
/// An N-Triples-formatted string representing the geometry as RDF triples
pub fn to_ntriples(geom: &Geometry, subject_uri: &str) -> Result<String> {
    let mut output = String::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let geo_as_wkt = "http://www.opengis.net/ont/geosparql#asWKT";
    let geo_coord_dim = "http://www.opengis.net/ont/geosparql#coordinateDimension";
    let geo_dimension = "http://www.opengis.net/ont/geosparql#dimension";
    let geo_is_empty = "http://www.opengis.net/ont/geosparql#isEmpty";
    let geo_is_simple = "http://www.opengis.net/ont/geosparql#isSimple";
    let geo_wkt_literal = "http://www.opengis.net/ont/geosparql#wktLiteral";
    let xsd_integer = "http://www.w3.org/2001/XMLSchema#integer";
    let xsd_boolean = "http://www.w3.org/2001/XMLSchema#boolean";

    let sf_type = geometry_type_to_sf_uri(geom);

    // Type triple
    writeln!(output, "<{}> <{}> <{}> .", subject_uri, rdf_type, sf_type).unwrap();

    // WKT literal triple
    writeln!(
        output,
        "<{}> <{}> \"{}\"^^<{}> .",
        subject_uri,
        geo_as_wkt,
        geom.to_wkt(),
        geo_wkt_literal
    )
    .unwrap();

    // Coordinate dimension triple
    writeln!(
        output,
        "<{}> <{}> \"{}\"^^<{}> .",
        subject_uri,
        geo_coord_dim,
        geom.coordinate_dimension(),
        xsd_integer
    )
    .unwrap();

    // Spatial dimension triple
    writeln!(
        output,
        "<{}> <{}> \"{}\"^^<{}> .",
        subject_uri,
        geo_dimension,
        geom.spatial_dimension(),
        xsd_integer
    )
    .unwrap();

    // isEmpty triple
    writeln!(
        output,
        "<{}> <{}> \"{}\"^^<{}> .",
        subject_uri,
        geo_is_empty,
        geom.is_empty(),
        xsd_boolean
    )
    .unwrap();

    // isSimple triple
    writeln!(
        output,
        "<{}> <{}> \"{}\"^^<{}> .",
        subject_uri,
        geo_is_simple,
        geom.is_simple(),
        xsd_boolean
    )
    .unwrap();

    Ok(output)
}

/// Serialize a geometry to RDF N-Quads format
///
/// N-Quads extends N-Triples by adding a fourth element (graph name) to each statement.
///
/// # Arguments
///
/// * `geom` - The geometry to serialize
/// * `subject_uri` - The full URI for this geometry resource
/// * `graph_uri` - The URI of the named graph (use None for default graph)
pub fn to_nquads(geom: &Geometry, subject_uri: &str, graph_uri: Option<&str>) -> Result<String> {
    let mut output = String::new();

    let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    let geo_as_wkt = "http://www.opengis.net/ont/geosparql#asWKT";
    let geo_wkt_literal = "http://www.opengis.net/ont/geosparql#wktLiteral";

    let sf_type = geometry_type_to_sf_uri(geom);
    let graph_suffix = match graph_uri {
        Some(g) => format!(" <{}>", g),
        None => String::new(),
    };

    // Type quad
    writeln!(
        output,
        "<{}> <{}> <{}> {} .",
        subject_uri, rdf_type, sf_type, graph_suffix
    )
    .unwrap();

    // WKT literal quad
    writeln!(
        output,
        "<{}> <{}> \"{}\"^^<{}> {} .",
        subject_uri,
        geo_as_wkt,
        geom.to_wkt(),
        geo_wkt_literal,
        graph_suffix
    )
    .unwrap();

    Ok(output)
}

/// Map geometry type to Simple Features ontology class (abbreviated)
fn geometry_type_to_sf(geom: &Geometry) -> &'static str {
    match &geom.geom {
        geo_types::Geometry::Point(_) => "sf:Point",
        geo_types::Geometry::Line(_) => "sf:LineString",
        geo_types::Geometry::LineString(_) => "sf:LineString",
        geo_types::Geometry::Polygon(_) => "sf:Polygon",
        geo_types::Geometry::MultiPoint(_) => "sf:MultiPoint",
        geo_types::Geometry::MultiLineString(_) => "sf:MultiLineString",
        geo_types::Geometry::MultiPolygon(_) => "sf:MultiPolygon",
        geo_types::Geometry::GeometryCollection(_) => "sf:GeometryCollection",
        _ => "sf:Geometry",
    }
}

/// Map geometry type to Simple Features ontology class (full URI)
fn geometry_type_to_sf_uri(geom: &Geometry) -> &'static str {
    match &geom.geom {
        geo_types::Geometry::Point(_) => "http://www.opengis.net/ont/sf#Point",
        geo_types::Geometry::Line(_) => "http://www.opengis.net/ont/sf#LineString",
        geo_types::Geometry::LineString(_) => "http://www.opengis.net/ont/sf#LineString",
        geo_types::Geometry::Polygon(_) => "http://www.opengis.net/ont/sf#Polygon",
        geo_types::Geometry::MultiPoint(_) => "http://www.opengis.net/ont/sf#MultiPoint",
        geo_types::Geometry::MultiLineString(_) => "http://www.opengis.net/ont/sf#MultiLineString",
        geo_types::Geometry::MultiPolygon(_) => "http://www.opengis.net/ont/sf#MultiPolygon",
        geo_types::Geometry::GeometryCollection(_) => {
            "http://www.opengis.net/ont/sf#GeometryCollection"
        }
        _ => "http://www.opengis.net/ont/sf#Geometry",
    }
}

/// Parse a WKT literal from RDF and create a Geometry
///
/// # Arguments
///
/// * `wkt_literal` - The WKT string (without the datatype suffix)
/// * `crs_uri` - Optional CRS URI for the geometry
///
/// # Examples
///
/// ```rust,ignore
/// use oxirs_geosparql::geometry::rdf_serialization::from_wkt_literal;
///
/// let geom = from_wkt_literal("POINT(1.0 2.0)", None).unwrap();
/// assert!(!geom.is_empty());
/// ```
pub fn from_wkt_literal(wkt_literal: &str, crs_uri: Option<String>) -> Result<Geometry> {
    let mut geom = Geometry::from_wkt(wkt_literal)?;

    if let Some(uri) = crs_uri {
        geom.crs = crate::geometry::Crs::new(uri);
    }

    Ok(geom)
}

/// Serialize to the specified RDF format
pub fn to_rdf(geom: &Geometry, subject_uri: &str, format: RdfFormat) -> Result<String> {
    match format {
        RdfFormat::Turtle => to_turtle(geom, subject_uri),
        RdfFormat::NTriples => to_ntriples(geom, subject_uri),
        RdfFormat::NQuads => to_nquads(geom, subject_uri, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_turtle_point() {
        let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
        let turtle = to_turtle(&point, ":myPoint").unwrap();

        assert!(turtle.contains("@prefix geo:"));
        assert!(turtle.contains("@prefix sf:"));
        assert!(turtle.contains(":myPoint"));
        assert!(turtle.contains("sf:Point"));
        assert!(turtle.contains("geo:asWKT"));
        assert!(turtle.contains("POINT") && turtle.contains("1") && turtle.contains("2"));
        assert!(turtle.contains("geo:coordinateDimension 2"));
        assert!(turtle.contains("geo:dimension 0"));
        assert!(turtle.contains("geo:isEmpty false"));
        assert!(turtle.contains("geo:isSimple true"));
    }

    #[test]
    fn test_to_turtle_linestring() {
        let line = Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        let turtle = to_turtle(&line, ":myLine").unwrap();

        assert!(turtle.contains("sf:LineString"));
        assert!(turtle.contains("LINESTRING(0 0, 1 1, 2 2)"));
        assert!(turtle.contains("geo:dimension 1"));
    }

    #[test]
    fn test_to_turtle_polygon() {
        let poly = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
        let turtle = to_turtle(&poly, ":myPolygon").unwrap();

        assert!(turtle.contains("sf:Polygon"));
        assert!(turtle.contains("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"));
        assert!(turtle.contains("geo:dimension 2"));
    }

    #[test]
    fn test_to_ntriples_point() {
        let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
        let ntriples = to_ntriples(&point, "http://example.org/geometry1").unwrap();

        assert!(ntriples.contains("<http://example.org/geometry1>"));
        assert!(ntriples.contains("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"));
        assert!(ntriples.contains("<http://www.opengis.net/ont/sf#Point>"));
        assert!(ntriples.contains("POINT") && ntriples.contains("1") && ntriples.contains("2"));
        assert!(ntriples.contains("^^<http://www.opengis.net/ont/geosparql#wktLiteral>"));

        // Should have exactly 6 triples (type, WKT, coordDim, dimension, isEmpty, isSimple)
        let line_count = ntriples.lines().filter(|l| !l.is_empty()).count();
        assert_eq!(line_count, 6);
    }

    #[test]
    fn test_to_nquads() {
        let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
        let nquads = to_nquads(
            &point,
            "http://example.org/geometry1",
            Some("http://example.org/graph1"),
        )
        .unwrap();

        assert!(nquads.contains("<http://example.org/geometry1>"));
        assert!(nquads.contains("<http://example.org/graph1>"));
        assert!(nquads.contains("<http://www.opengis.net/ont/sf#Point>"));
        assert!(nquads.contains("POINT") && nquads.contains("1") && nquads.contains("2"));

        // Each line should end with the graph URI
        for line in nquads.lines().filter(|l| !l.is_empty()) {
            assert!(line.contains("<http://example.org/graph1>"));
        }
    }

    #[test]
    fn test_to_nquads_default_graph() {
        let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
        let nquads = to_nquads(&point, "http://example.org/geometry1", None).unwrap();

        // Default graph - no graph URI at the end
        for line in nquads.lines().filter(|l| !l.is_empty()) {
            assert!(!line.contains("<http://example.org/graph"));
        }
    }

    #[test]
    fn test_from_wkt_literal() {
        let geom = from_wkt_literal("POINT(1.0 2.0)", None).unwrap();
        assert!(!geom.is_empty());
        assert_eq!(geom.spatial_dimension(), 0); // Point dimension
    }

    #[test]
    fn test_from_wkt_literal_with_crs() {
        let geom = from_wkt_literal(
            "POINT(1.0 2.0)",
            Some("http://www.opengis.net/def/crs/EPSG/0/4326".to_string()),
        )
        .unwrap();
        assert_eq!(geom.crs.uri, "http://www.opengis.net/def/crs/EPSG/0/4326");
    }

    #[test]
    fn test_to_rdf_formats() {
        let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();

        // Test Turtle format
        let turtle = to_rdf(&point, ":myPoint", RdfFormat::Turtle).unwrap();
        assert!(turtle.contains("@prefix geo:"));

        // Test N-Triples format
        let ntriples = to_rdf(&point, "http://example.org/geom1", RdfFormat::NTriples).unwrap();
        assert!(ntriples.contains("<http://example.org/geom1>"));

        // Test N-Quads format
        let nquads = to_rdf(&point, "http://example.org/geom1", RdfFormat::NQuads).unwrap();
        assert!(nquads.contains("<http://example.org/geom1>"));
    }

    #[test]
    fn test_multipoint_serialization() {
        let mp = Geometry::from_wkt("MULTIPOINT((1 1), (2 2), (3 3))").unwrap();
        let turtle = to_turtle(&mp, ":myMultiPoint").unwrap();

        assert!(turtle.contains("sf:MultiPoint"));
        assert!(turtle.contains("MULTIPOINT((1 1), (2 2), (3 3))"));
    }

    #[test]
    fn test_multipolygon_serialization() {
        let mp = Geometry::from_wkt(
            "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
        )
        .unwrap();
        let turtle = to_turtle(&mp, ":myMultiPoly").unwrap();

        assert!(turtle.contains("sf:MultiPolygon"));
        assert!(turtle.contains("geo:dimension 2"));
    }

    #[test]
    fn test_round_trip() {
        let original = Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        let turtle = to_turtle(&original, ":myLine").unwrap();

        // Extract WKT from turtle (simple parsing for test)
        let wkt_start = turtle.find("geo:asWKT \"").unwrap() + 11;
        let wkt_end = turtle[wkt_start..].find("\"^^geo:wktLiteral").unwrap();
        let wkt = &turtle[wkt_start..wkt_start + wkt_end];

        let parsed = from_wkt_literal(wkt, None).unwrap();
        assert_eq!(original.to_wkt(), parsed.to_wkt());
    }
}
