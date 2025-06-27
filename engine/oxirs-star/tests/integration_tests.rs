//! Integration tests for RDF-star parsers and serializers

use oxirs_star::{
    model::{StarGraph, StarTerm, StarTriple, StarQuad},
    parser::{StarParser, StarFormat},
    serializer::StarSerializer,
};
use tempfile::NamedTempFile;
use std::io::Write;

#[test]
fn test_turtle_star_round_trip() {
    let input = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:alice ex:says << ex:bob ex:age 30 >> .
<< ex:charlie ex:knows ex:david >> ex:since "2020" .
    "#;

    // Parse
    let parser = StarParser::new();
    let graph = parser.parse_str(input, StarFormat::TurtleStar).unwrap();
    
    assert_eq!(graph.len(), 2);
    
    // Serialize back
    let serializer = StarSerializer::new();
    let mut output = Vec::new();
    serializer.serialize(&graph, &mut output, StarFormat::TurtleStar).unwrap();
    
    let output_str = String::from_utf8(output).unwrap();
    
    // Parse the serialized output to verify
    let reparsed = parser.parse_str(&output_str, StarFormat::TurtleStar).unwrap();
    assert_eq!(reparsed.len(), graph.len());
}

#[test]
fn test_ntriples_star_round_trip() {
    let input = r#"<http://example.org/alice> <http://example.org/says> << <http://example.org/bob> <http://example.org/age> "30" >> .
<< <http://example.org/charlie> <http://example.org/knows> <http://example.org/david> >> <http://example.org/since> "2020" .
"#;

    // Parse
    let parser = StarParser::new();
    let graph = parser.parse_str(input, StarFormat::NTriplesStar).unwrap();
    
    assert_eq!(graph.len(), 2);
    
    // Serialize back
    let serializer = StarSerializer::new();
    let mut output = Vec::new();
    serializer.serialize(&graph, &mut output, StarFormat::NTriplesStar).unwrap();
    
    let output_str = String::from_utf8(output).unwrap();
    
    // Verify output contains the expected triples
    assert!(output_str.contains("<< <http://example.org/bob>"));
    assert!(output_str.contains("<< <http://example.org/charlie>"));
}

#[test]
fn test_trig_star_named_graphs() {
    let input = r#"
@prefix ex: <http://example.org/> .

{
    ex:alice ex:name "Alice" .
}

ex:graph1 {
    ex:bob ex:name "Bob" .
    << ex:bob ex:age 30 >> ex:confidence 0.9 .
}

ex:graph2 {
    ex:charlie ex:name "Charlie" .
}
    "#;

    // Parse
    let parser = StarParser::new();
    let graph = parser.parse_str(input, StarFormat::TrigStar).unwrap();
    
    // Check default graph
    assert!(graph.len() > 0); // Has default graph triples
    
    // Check named graphs
    let graph_names = graph.named_graph_names();
    assert!(graph_names.iter().any(|&name| name.contains("graph1")));
    assert!(graph_names.iter().any(|&name| name.contains("graph2")));
    
    // Serialize back
    let serializer = StarSerializer::new();
    let mut output = Vec::new();
    serializer.serialize(&graph, &mut output, StarFormat::TrigStar).unwrap();
    
    let output_str = String::from_utf8(output).unwrap();
    
    // Verify output contains graph blocks
    assert!(output_str.contains("graph1"));
    assert!(output_str.contains("graph2"));
}

#[test]
fn test_nquads_star_round_trip() {
    let input = r#"<http://example.org/alice> <http://example.org/name> "Alice" .
<http://example.org/bob> <http://example.org/name> "Bob" <http://example.org/graph1> .
<< <http://example.org/bob> <http://example.org/age> "30" >> <http://example.org/confidence> "0.9" <http://example.org/graph1> .
"#;

    // Parse
    let parser = StarParser::new();
    let graph = parser.parse_str(input, StarFormat::NQuadsStar).unwrap();
    
    assert_eq!(graph.quad_len(), 3);
    
    // Serialize back
    let serializer = StarSerializer::new();
    let mut output = Vec::new();
    serializer.serialize(&graph, &mut output, StarFormat::NQuadsStar).unwrap();
    
    let output_str = String::from_utf8(output).unwrap();
    
    // Parse the serialized output to verify
    let reparsed = parser.parse_str(&output_str, StarFormat::NQuadsStar).unwrap();
    assert_eq!(reparsed.quad_len(), graph.quad_len());
}

#[test]
fn test_deeply_nested_quoted_triples() {
    let input = r#"
@prefix ex: <http://example.org/> .

ex:alice ex:believes << ex:bob ex:claims << ex:charlie ex:says << ex:david ex:age 25 >> >> >> .
    "#;

    // Parse
    let parser = StarParser::new();
    let graph = parser.parse_str(input, StarFormat::TurtleStar).unwrap();
    
    assert_eq!(graph.len(), 1);
    
    // Check nesting depth
    assert_eq!(graph.max_nesting_depth(), 3);
    
    // Serialize to different formats
    let serializer = StarSerializer::new();
    
    // To N-Triples-star
    let mut nt_output = Vec::new();
    serializer.serialize(&graph, &mut nt_output, StarFormat::NTriplesStar).unwrap();
    let nt_str = String::from_utf8(nt_output).unwrap();
    
    // Count nested brackets
    let open_brackets = nt_str.matches("<<").count();
    let close_brackets = nt_str.matches(">>").count();
    assert_eq!(open_brackets, 3);
    assert_eq!(close_brackets, 3);
}

#[test]
fn test_error_recovery_mode() {
    let input = r#"
@prefix ex: <http://example.org/> .

ex:alice ex:knows ex:bob .
ex:charlie ex:knows  # Missing object - error
ex:david ex:knows ex:eve .
<< ex:frank ex:knows # Incomplete quoted triple
ex:grace ex:knows ex:henry .
    "#;

    // Parse in non-strict mode
    let mut parser = StarParser::new();
    parser.set_strict_mode(false);
    
    let result = parser.parse_str(input, StarFormat::TurtleStar);
    assert!(result.is_ok());
    
    let graph = result.unwrap();
    // Should have parsed at least the valid triples
    assert!(graph.len() >= 3);
}

#[test]
fn test_format_conversion() {
    // Create a graph with mixed content
    let mut graph = StarGraph::new();
    
    // Add regular triple
    let triple1 = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/knows").unwrap(),
        StarTerm::iri("http://example.org/bob").unwrap(),
    );
    graph.insert(triple1).unwrap();
    
    // Add quoted triple
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/charlie").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("30").unwrap(),
    );
    let triple2 = StarTriple::new(
        StarTerm::quoted_triple(inner),
        StarTerm::iri("http://example.org/confidence").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );
    graph.insert(triple2).unwrap();
    
    // Add quad with named graph
    let quad = StarQuad::new(
        StarTerm::iri("http://example.org/david").unwrap(),
        StarTerm::iri("http://example.org/name").unwrap(),
        StarTerm::literal("David").unwrap(),
        Some(StarTerm::iri("http://example.org/graph1").unwrap()),
    );
    graph.insert_quad(quad).unwrap();
    
    // Test conversion between formats
    let serializer = StarSerializer::new();
    let parser = StarParser::new();
    
    let formats = vec![
        StarFormat::TurtleStar,
        StarFormat::NTriplesStar,
        StarFormat::TrigStar,
        StarFormat::NQuadsStar,
    ];
    
    for format in formats {
        let mut output = Vec::new();
        serializer.serialize(&graph, &mut output, format).unwrap();
        
        let serialized = String::from_utf8(output).unwrap();
        let reparsed = parser.parse_str(&serialized, format).unwrap();
        
        // All formats should preserve the total number of triples
        assert_eq!(reparsed.total_len(), graph.total_len());
    }
}

#[test]
fn test_streaming_large_file() {
    // Create a temporary file with many triples
    let mut temp_file = NamedTempFile::new().unwrap();
    
    // Write many triples
    for i in 0..1000 {
        writeln!(
            temp_file,
            "<http://example.org/subject{}> <http://example.org/predicate> \"object{}\" .",
            i, i
        ).unwrap();
        
        // Add some quoted triples
        if i % 10 == 0 {
            writeln!(
                temp_file,
                "<< <http://example.org/s{}> <http://example.org/p> \"o{}\" >> <http://example.org/meta> \"metadata{}\" .",
                i, i, i
            ).unwrap();
        }
    }
    
    temp_file.flush().unwrap();
    
    // Parse the file
    let parser = StarParser::new();
    let file = std::fs::File::open(temp_file.path()).unwrap();
    let graph = parser.parse(file, StarFormat::NTriplesStar).unwrap();
    
    // Verify counts
    assert_eq!(graph.len(), 1100); // 1000 regular + 100 quoted
}

#[test]
fn test_prefix_compression() {
    let input = r#"
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:alice foaf:knows ex:bob ;
         foaf:name "Alice" ;
         rdf:type foaf:Person .

ex:bob foaf:knows ex:alice ;
       foaf:name "Bob" ;
       rdf:type foaf:Person .
    "#;

    // Parse
    let parser = StarParser::new();
    let graph = parser.parse_str(input, StarFormat::TurtleStar).unwrap();
    
    // Serialize with prefix compression
    let serializer = StarSerializer::new();
    let mut output = Vec::new();
    serializer.serialize(&graph, &mut output, StarFormat::TurtleStar).unwrap();
    
    let output_str = String::from_utf8(output).unwrap();
    
    // Verify prefixes are used
    assert!(output_str.contains("@prefix"));
    assert!(output_str.contains("ex:"));
    assert!(output_str.contains("foaf:"));
}