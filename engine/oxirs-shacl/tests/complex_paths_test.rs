//! Tests for complex property path parsing from RDF

use oxirs_core::{graph::Graph, model::*};
use oxirs_shacl::shapes::ShapeParser;
use oxirs_shacl::paths::PropertyPath;

#[test]
fn test_parse_complex_property_paths_from_graph() {
    // Create a graph with complex property paths
    let mut graph = Graph::new();
    
    // Add a shape with an inverse path
    graph.add_triple(Triple::new(
        NamedNode::new("http://example.org/InversePathShape").unwrap(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        NamedNode::new("http://www.w3.org/ns/shacl#PropertyShape").unwrap(),
    ));
    
    // The path is a blank node
    let path_blank = BlankNode::new("b1").unwrap();
    graph.add_triple(Triple::new(
        NamedNode::new("http://example.org/InversePathShape").unwrap(),
        NamedNode::new("http://www.w3.org/ns/shacl#path").unwrap(),
        path_blank.clone(),
    ));
    
    // The blank node has sh:inversePath
    graph.add_triple(Triple::new(
        path_blank.clone(),
        NamedNode::new("http://www.w3.org/ns/shacl#inversePath").unwrap(),
        NamedNode::new("http://example.org/knows").unwrap(),
    ));
    
    // Parse shapes from the graph
    let mut parser = ShapeParser::new();
    let shapes = parser.parse_shapes_from_graph(&graph).unwrap();
    
    assert_eq!(shapes.len(), 1);
    let shape = &shapes[0];
    
    // Check that the path is an inverse path
    assert!(shape.path.is_some());
    if let Some(path) = &shape.path {
        assert!(matches!(path, PropertyPath::Inverse(_)));
    }
}

#[test]
fn test_parse_alternative_paths_from_graph() {
    // Create a graph with alternative paths
    let mut graph = Graph::new();
    
    // Add a shape with alternative paths
    graph.add_triple(Triple::new(
        NamedNode::new("http://example.org/AlternativePathShape").unwrap(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        NamedNode::new("http://www.w3.org/ns/shacl#PropertyShape").unwrap(),
    ));
    
    // The path is a blank node
    let path_blank = BlankNode::new("b1").unwrap();
    graph.add_triple(Triple::new(
        NamedNode::new("http://example.org/AlternativePathShape").unwrap(),
        NamedNode::new("http://www.w3.org/ns/shacl#path").unwrap(),
        path_blank.clone(),
    ));
    
    // The blank node has sh:alternativePath pointing to a list
    let list_blank = BlankNode::new("list1").unwrap();
    graph.add_triple(Triple::new(
        path_blank.clone(),
        NamedNode::new("http://www.w3.org/ns/shacl#alternativePath").unwrap(),
        list_blank.clone(),
    ));
    
    // Build the RDF list
    graph.add_triple(Triple::new(
        list_blank.clone(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first").unwrap(),
        NamedNode::new("http://example.org/name").unwrap(),
    ));
    
    let list_rest = BlankNode::new("list2").unwrap();
    graph.add_triple(Triple::new(
        list_blank.clone(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest").unwrap(),
        list_rest.clone(),
    ));
    
    graph.add_triple(Triple::new(
        list_rest.clone(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first").unwrap(),
        NamedNode::new("http://example.org/label").unwrap(),
    ));
    
    graph.add_triple(Triple::new(
        list_rest.clone(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest").unwrap(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil").unwrap(),
    ));
    
    // Parse shapes from the graph
    let mut parser = ShapeParser::new();
    let shapes = parser.parse_shapes_from_graph(&graph).unwrap();
    
    assert_eq!(shapes.len(), 1);
    let shape = &shapes[0];
    
    // Check that the path is an alternative path
    assert!(shape.path.is_some());
    if let Some(path) = &shape.path {
        assert!(matches!(path, PropertyPath::Alternative(_)));
    }
}