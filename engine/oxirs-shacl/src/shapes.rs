//! SHACL shape parsing and representation
//!
//! This module handles parsing SHACL shapes from RDF data.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use oxirs_core::{
    graph::Graph,
    model::{BlankNode, Literal, NamedNode, RdfTerm, Term, Triple, Quad, GraphName},
    store::Store,
    OxirsError,
};

use crate::{
    constraints::*, paths::*, targets::*, Constraint, ConstraintComponentId, PropertyPath, Result,
    Severity, ShaclError, Shape, ShapeId, ShapeType, Target, SHACL_NS, SHACL_VOCAB,
};

/// SHACL shape parser for extracting shapes from RDF data
#[derive(Debug)]
pub struct ShapeParser {
    /// Cache for parsed shapes to avoid re-parsing
    shape_cache: HashMap<String, Shape>,

    /// Enable strict parsing mode
    strict_mode: bool,

    /// Maximum recursion depth for shape parsing
    max_depth: usize,
}

impl ShapeParser {
    /// Create a new shape parser
    pub fn new() -> Self {
        Self {
            shape_cache: HashMap::new(),
            strict_mode: false,
            max_depth: 50,
        }
    }

    /// Create a new parser in strict mode
    pub fn new_strict() -> Self {
        let mut parser = Self::new();
        parser.strict_mode = true;
        parser
    }

    /// Set maximum recursion depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Parse shapes from an RDF store
    pub fn parse_shapes_from_store(
        &mut self,
        store: &Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<Shape>> {
        let mut shapes = Vec::new();

        // Find all shape IRIs in the store
        let shape_iris = self.find_shape_iris_in_store(store, graph_name)?;

        // Parse each shape
        let mut visited = HashSet::new();
        for shape_iri in shape_iris {
            if !visited.contains(&shape_iri) {
                match self.parse_shape_from_store(store, &shape_iri, &mut visited, 0, graph_name) {
                    Ok(shape) => shapes.push(shape),
                    Err(e) => {
                        if self.strict_mode {
                            return Err(e);
                        } else {
                            tracing::warn!("Failed to parse shape {}: {}", shape_iri, e);
                        }
                    }
                }
            }
        }

        Ok(shapes)
    }

    /// Parse shapes from RDF data string
    pub fn parse_shapes_from_rdf(
        &mut self,
        rdf_data: &str,
        format: &str,
        base_iri: Option<&str>,
    ) -> Result<Vec<Shape>> {
        use oxirs_core::parser::{Parser, RdfFormat, ParserConfig};
        
        // Determine RDF format from string
        let rdf_format = match format.to_lowercase().as_str() {
            "turtle" | "ttl" => RdfFormat::Turtle,
            "ntriples" | "nt" => RdfFormat::NTriples,
            "trig" => RdfFormat::TriG,
            "nquads" | "nq" => RdfFormat::NQuads,
            "rdfxml" | "rdf" | "xml" => RdfFormat::RdfXml,
            "jsonld" | "json-ld" => RdfFormat::JsonLd,
            _ => return Err(ShaclError::ShapeParsing(
                format!("Unsupported RDF format: {}", format)
            )),
        };
        
        // Create parser with base IRI
        let mut parser_config = ParserConfig::default();
        if let Some(base) = base_iri {
            parser_config.base_iri = Some(base.to_string());
        }
        let parser = Parser::with_config(rdf_format, parser_config);
        
        // Parse RDF data to quads
        let quads = parser.parse_str_to_quads(rdf_data).map_err(|e| 
            ShaclError::ShapeParsing(format!("Failed to parse RDF: {}", e))
        )?;
        
        // Create a temporary graph from the quads
        let mut graph = Graph::new();
        for quad in quads {
            // Only use triples from the default graph for shapes
            if quad.is_default_graph() {
                graph.insert(quad.to_triple());
            }
        }
        
        // Parse shapes from the graph
        self.parse_shapes_from_graph(&graph)
    }

    /// Parse shapes from an RDF graph
    pub fn parse_shapes_from_graph(&mut self, graph: &Graph) -> Result<Vec<Shape>> {
        let mut shapes = Vec::new();
        let mut visited = HashSet::new();

        // Find all shape definitions in the graph
        let shape_iris = self.find_shape_iris(graph)?;

        for shape_iri in shape_iris {
            if !visited.contains(&shape_iri) {
                let shape = self.parse_shape(graph, &shape_iri, &mut visited, 0)?;
                shapes.push(shape);
            }
        }

        Ok(shapes)
    }

    /// Find all IRIs that represent shapes in a graph
    fn find_shape_iris(&self, graph: &Graph) -> Result<Vec<String>> {
        use oxirs_core::model::{Subject, Predicate, Object};
        
        let mut shape_iris = HashSet::new();
        
        // SHACL namespace
        let shacl_ns = "http://www.w3.org/ns/shacl#";
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid RDF type IRI: {}", e)))?;
        
        // Find explicit NodeShape and PropertyShape instances
        let shape_types = vec![
            NamedNode::new(&format!("{}NodeShape", shacl_ns))
                .map_err(|e| ShaclError::ShapeParsing(format!("Invalid NodeShape IRI: {}", e)))?,
            NamedNode::new(&format!("{}PropertyShape", shacl_ns))
                .map_err(|e| ShaclError::ShapeParsing(format!("Invalid PropertyShape IRI: {}", e)))?,
        ];
        
        for shape_type in shape_types {
            let triples = graph.query_triples(
                None,
                Some(&Predicate::NamedNode(rdf_type.clone())),
                Some(&Object::NamedNode(shape_type))
            );
            
            for triple in triples {
                if let Subject::NamedNode(shape_node) = triple.subject() {
                    shape_iris.insert(shape_node.as_str().to_string());
                }
            }
        }
        
        // Find shapes by properties that indicate shape-ness
        let shape_properties = vec![
            "targetClass", "targetNode", "targetObjectsOf", "targetSubjectsOf",
            "property", "path", "node", "class", "datatype", "minCount", "maxCount",
        ];
        
        for prop_name in shape_properties {
            let prop_iri = NamedNode::new(&format!("{}{}", shacl_ns, prop_name))
                .map_err(|e| ShaclError::ShapeParsing(format!("Invalid property IRI: {}", e)))?;
            
            let triples = graph.query_triples(
                None,
                Some(&Predicate::NamedNode(prop_iri)),
                None
            );
            
            for triple in triples {
                if let Subject::NamedNode(shape_node) = triple.subject() {
                    shape_iris.insert(shape_node.as_str().to_string());
                }
            }
        }
        
        tracing::info!("Discovered {} shape IRIs in graph", shape_iris.len());
        Ok(shape_iris.into_iter().collect())
    }

    /// Parse a single shape from a graph
    fn parse_shape(
        &mut self,
        graph: &Graph,
        shape_iri: &str,
        visited: &mut HashSet<String>,
        depth: usize,
    ) -> Result<Shape> {
        use oxirs_core::model::{Subject, Predicate, Object};
        
        if depth > self.max_depth {
            return Err(ShaclError::ShapeParsing(format!(
                "Maximum parsing depth {} exceeded for shape {}",
                self.max_depth, shape_iri
            )));
        }
        
        // Mark as visited
        visited.insert(shape_iri.to_string());
        
        // Check cache first
        if let Some(cached_shape) = self.shape_cache.get(shape_iri) {
            return Ok(cached_shape.clone());
        }
        
        let shape_node = NamedNode::new(shape_iri)
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid shape IRI {}: {}", shape_iri, e)))?;
        
        // Determine shape type
        let shape_type = self.determine_shape_type(graph, &shape_node)?;
        
        // Create the shape
        let mut shape = Shape::new(ShapeId::new(shape_iri), shape_type);
        
        // Parse targets
        self.parse_shape_targets_from_graph(graph, &shape_node, &mut shape)?;
        
        // Parse property path (for property shapes)
        if shape.shape_type == ShapeType::PropertyShape {
            self.parse_property_path_from_graph(graph, &shape_node, &mut shape)?;
        }
        
        // Parse constraints
        self.parse_shape_constraints_from_graph(graph, &shape_node, &mut shape)?;
        
        // Parse severity
        self.parse_shape_severity_from_graph(graph, &shape_node, &mut shape)?;
        
        // Parse message
        self.parse_shape_message_from_graph(graph, &shape_node, &mut shape)?;
        
        // Parse deactivated status
        self.parse_shape_deactivated_from_graph(graph, &shape_node, &mut shape)?;
        
        // Parse inheritance (sh:extends)
        self.parse_shape_extends_from_graph(graph, &shape_node, &mut shape)?;
        
        // Parse priority
        self.parse_shape_priority_from_graph(graph, &shape_node, &mut shape)?;
        
        // Parse metadata
        self.parse_shape_metadata_from_graph(graph, &shape_node, &mut shape)?;
        
        // Cache the parsed shape
        if self.shape_cache.len() < 1000 { // Limit cache size
            self.shape_cache.insert(shape_iri.to_string(), shape.clone());
        }
        
        Ok(shape)
    }
    
    /// Determine the type of a shape from the graph
    fn determine_shape_type(&self, graph: &Graph, shape_node: &NamedNode) -> Result<ShapeType> {
        use oxirs_core::model::{Subject, Predicate, Object};
        
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid RDF type IRI: {}", e)))?;
        
        let node_shape_type = NamedNode::new("http://www.w3.org/ns/shacl#NodeShape")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid NodeShape IRI: {}", e)))?;
        
        let property_shape_type = NamedNode::new("http://www.w3.org/ns/shacl#PropertyShape")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid PropertyShape IRI: {}", e)))?;
        
        // Check for explicit type declarations
        let type_triples = graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(rdf_type)),
            None
        );
        
        for triple in type_triples {
            if let Object::NamedNode(type_node) = triple.object() {
                if type_node == &property_shape_type {
                    return Ok(ShapeType::PropertyShape);
                } else if type_node == &node_shape_type {
                    return Ok(ShapeType::NodeShape);
                }
            }
        }
        
        // If no explicit type, check for property shape indicators
        let path_predicate = NamedNode::new("http://www.w3.org/ns/shacl#path")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid path IRI: {}", e)))?;
        
        let has_path = !graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(path_predicate)),
            None
        ).is_empty();
        
        if has_path {
            Ok(ShapeType::PropertyShape)
        } else {
            Ok(ShapeType::NodeShape)
        }
    }
    
    /// Parse shape targets from a graph
    fn parse_shape_targets_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate, Object};
        
        // Parse sh:targetClass
        let target_class_pred = NamedNode::new("http://www.w3.org/ns/shacl#targetClass")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid targetClass IRI: {}", e)))?;
        
        for triple in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(target_class_pred)),
            None
        ) {
            if let Object::NamedNode(class_node) = triple.object() {
                shape.add_target(Target::class(class_node.clone()));
            }
        }
        
        // Parse sh:targetNode
        let target_node_pred = NamedNode::new("http://www.w3.org/ns/shacl#targetNode")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid targetNode IRI: {}", e)))?;
        
        for triple in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(target_node_pred)),
            None
        ) {
            shape.add_target(Target::node(object_to_term(triple.object())?));
        }
        
        // Parse sh:targetObjectsOf
        let target_objects_of_pred = NamedNode::new("http://www.w3.org/ns/shacl#targetObjectsOf")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid targetObjectsOf IRI: {}", e)))?;
        
        for triple in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(target_objects_of_pred)),
            None
        ) {
            if let Object::NamedNode(prop_node) = triple.object() {
                shape.add_target(Target::objects_of(prop_node.clone()));
            }
        }
        
        // Parse sh:targetSubjectsOf
        let target_subjects_of_pred = NamedNode::new("http://www.w3.org/ns/shacl#targetSubjectsOf")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid targetSubjectsOf IRI: {}", e)))?;
        
        for triple in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(target_subjects_of_pred)),
            None
        ) {
            if let Object::NamedNode(prop_node) = triple.object() {
                shape.add_target(Target::subjects_of(prop_node.clone()));
            }
        }
        
        Ok(())
    }
    
    /// Parse property path from a graph
    fn parse_property_path_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate};
        
        let path_pred = NamedNode::new("http://www.w3.org/ns/shacl#path")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid path IRI: {}", e)))?;
        
        let path_triples = graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(path_pred)),
            None
        );
        
        if let Some(triple) = path_triples.first() {
            let path_term = object_to_term(triple.object())?;
            // Parse path directly from graph instead of using store
            let path = self.parse_property_path_from_term_with_graph(graph, &path_term)?;
            shape.path = Some(path);
        }
        
        Ok(())
    }
    
    /// Parse property path from an RDF term using a graph
    fn parse_property_path_from_term_with_graph(
        &self,
        graph: &Graph,
        path_term: &Term,
    ) -> Result<PropertyPath> {
        use crate::paths::PropertyPath;
        
        match path_term {
            Term::NamedNode(predicate_node) => Ok(PropertyPath::predicate(predicate_node.clone())),
            Term::BlankNode(blank_node) => {
                // Complex property path - need to analyze the structure
                self.parse_complex_property_path_from_graph(graph, blank_node)
            }
            _ => Err(ShaclError::PropertyPath(
                "Invalid property path term - must be a named node or blank node".to_string(),
            )),
        }
    }
    
    /// Parse complex property paths from graph
    fn parse_complex_property_path_from_graph(
        &self,
        graph: &Graph,
        blank_node: &BlankNode,
    ) -> Result<PropertyPath> {
        use crate::paths::PropertyPath;
        use oxirs_core::model::{Subject, Predicate};
        
        // Check for inverse path (sh:inversePath)
        let inverse_pred = NamedNode::new("http://www.w3.org/ns/shacl#inversePath")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid inversePath IRI: {}", e)))?;
        
        let inverse_triples = graph.query_triples(
            Some(&Subject::BlankNode(blank_node.clone())),
            Some(&Predicate::NamedNode(inverse_pred)),
            None
        );
        
        if let Some(triple) = inverse_triples.first() {
            let inner_term = object_to_term(triple.object())?;
            let inner_path = self.parse_property_path_from_term_with_graph(graph, &inner_term)?;
            return Ok(PropertyPath::inverse(inner_path));
        }
        
        // Check for alternative path (sh:alternativePath)
        let alt_pred = NamedNode::new("http://www.w3.org/ns/shacl#alternativePath")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid alternativePath IRI: {}", e)))?;
        
        let alt_triples = graph.query_triples(
            Some(&Subject::BlankNode(blank_node.clone())),
            Some(&Predicate::NamedNode(alt_pred)),
            None
        );
        
        if let Some(triple) = alt_triples.first() {
            if let Ok(Term::BlankNode(list_blank)) = object_to_term(triple.object()) {
                if let Some(paths) = self.parse_path_list_from_graph(graph, &list_blank)? {
                    return Ok(PropertyPath::alternative(paths));
                }
            }
        }
        
        // Check for zero-or-more path (sh:zeroOrMorePath)
        let zero_or_more_pred = NamedNode::new("http://www.w3.org/ns/shacl#zeroOrMorePath")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid zeroOrMorePath IRI: {}", e)))?;
        
        let zero_or_more_triples = graph.query_triples(
            Some(&Subject::BlankNode(blank_node.clone())),
            Some(&Predicate::NamedNode(zero_or_more_pred)),
            None
        );
        
        if let Some(triple) = zero_or_more_triples.first() {
            let inner_term = object_to_term(triple.object())?;
            let inner_path = self.parse_property_path_from_term_with_graph(graph, &inner_term)?;
            return Ok(PropertyPath::zero_or_more(inner_path));
        }
        
        // Check for one-or-more path (sh:oneOrMorePath)
        let one_or_more_pred = NamedNode::new("http://www.w3.org/ns/shacl#oneOrMorePath")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid oneOrMorePath IRI: {}", e)))?;
        
        let one_or_more_triples = graph.query_triples(
            Some(&Subject::BlankNode(blank_node.clone())),
            Some(&Predicate::NamedNode(one_or_more_pred)),
            None
        );
        
        if let Some(triple) = one_or_more_triples.first() {
            let inner_term = object_to_term(triple.object())?;
            let inner_path = self.parse_property_path_from_term_with_graph(graph, &inner_term)?;
            return Ok(PropertyPath::one_or_more(inner_path));
        }
        
        // Check for zero-or-one path (sh:zeroOrOnePath)
        let zero_or_one_pred = NamedNode::new("http://www.w3.org/ns/shacl#zeroOrOnePath")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid zeroOrOnePath IRI: {}", e)))?;
        
        let zero_or_one_triples = graph.query_triples(
            Some(&Subject::BlankNode(blank_node.clone())),
            Some(&Predicate::NamedNode(zero_or_one_pred)),
            None
        );
        
        if let Some(triple) = zero_or_one_triples.first() {
            let inner_term = object_to_term(triple.object())?;
            let inner_path = self.parse_property_path_from_term_with_graph(graph, &inner_term)?;
            return Ok(PropertyPath::zero_or_one(inner_path));
        }
        
        // Try to parse as a list (sequence path)
        if let Some(paths) = self.parse_path_list_from_graph(graph, blank_node)? {
            if paths.len() > 1 {
                return Ok(PropertyPath::sequence(paths));
            } else if paths.len() == 1 {
                return Ok(paths.into_iter().next().unwrap());
            }
        }
        
        Err(ShaclError::PropertyPath(format!(
            "Unable to parse complex property path from blank node: {}",
            blank_node.as_str()
        )))
    }
    
    /// Parse an RDF list of property paths from graph
    fn parse_path_list_from_graph(
        &self,
        graph: &Graph,
        list_node: &BlankNode,
    ) -> Result<Option<Vec<PropertyPath>>> {
        use oxirs_core::model::{Subject, Predicate};
        
        let mut paths = Vec::new();
        let mut current = Term::BlankNode(list_node.clone());
        
        let first_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid rdf:first IRI: {}", e)))?;
        let rest_pred = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid rdf:rest IRI: {}", e)))?;
        let nil = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
            .map_err(|e| ShaclError::PropertyPath(format!("Invalid rdf:nil IRI: {}", e)))?;
        
        loop {
            // Get the subject for the query
            let subject = match &current {
                Term::BlankNode(bn) => Subject::BlankNode(bn.clone()),
                Term::NamedNode(nn) => Subject::NamedNode(nn.clone()),
                _ => return Ok(None),
            };
            
            // Get the first element
            let first_triples = graph.query_triples(
                Some(&subject),
                Some(&Predicate::NamedNode(first_pred.clone())),
                None
            );
            
            if let Some(triple) = first_triples.first() {
                let first_term = object_to_term(triple.object())?;
                let path = self.parse_property_path_from_term_with_graph(graph, &first_term)?;
                paths.push(path);
                
                // Get the rest of the list
                let rest_triples = graph.query_triples(
                    Some(&subject),
                    Some(&Predicate::NamedNode(rest_pred.clone())),
                    None
                );
                
                if let Some(rest_triple) = rest_triples.first() {
                    let rest_term = object_to_term(rest_triple.object())?;
                    
                    // Check if we've reached rdf:nil
                    if let Term::NamedNode(nn) = &rest_term {
                        if nn == &nil {
                            break;
                        }
                    }
                    current = rest_term;
                } else {
                    break;
                }
            } else {
                // Not a valid list
                return Ok(None);
            }
        }
        
        if paths.is_empty() {
            Ok(None)
        } else {
            Ok(Some(paths))
        }
    }
    
    /// Parse shape constraints from a graph
    fn parse_shape_constraints_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use crate::constraints::*;
        use crate::{Constraint, ConstraintComponentId};
        use oxirs_core::model::{Subject, Predicate};
        
        let shacl_ns = "http://www.w3.org/ns/shacl#";
        
        // Parse sh:class constraint
        let class_pred = NamedNode::new(&format!("{}class", shacl_ns))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid class predicate: {}", e)))?;
        
        for triple in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(class_pred)),
            None
        ) {
            if let Ok(Term::NamedNode(class_node)) = object_to_term(triple.object()) {
                let constraint = Constraint::Class(ClassConstraint {
                    class_iri: class_node,
                });
                shape.add_constraint(
                    ConstraintComponentId::new("sh:ClassConstraintComponent"),
                    constraint
                );
            }
        }
        
        // Parse sh:datatype constraint
        let datatype_pred = NamedNode::new(&format!("{}datatype", shacl_ns))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid datatype predicate: {}", e)))?;
        
        for triple in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(datatype_pred)),
            None
        ) {
            if let Ok(Term::NamedNode(datatype_node)) = object_to_term(triple.object()) {
                let constraint = Constraint::Datatype(DatatypeConstraint {
                    datatype_iri: datatype_node,
                });
                shape.add_constraint(
                    ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
                    constraint
                );
            }
        }
        
        // Parse sh:qualifiedValueShape constraint
        let qualified_shape_pred = NamedNode::new(&format!("{}qualifiedValueShape", shacl_ns))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid qualifiedValueShape predicate: {}", e)))?;
        
        let qualified_min_pred = NamedNode::new(&format!("{}qualifiedMinCount", shacl_ns))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid qualifiedMinCount predicate: {}", e)))?;
            
        let qualified_max_pred = NamedNode::new(&format!("{}qualifiedMaxCount", shacl_ns))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid qualifiedMaxCount predicate: {}", e)))?;
            
        let qualified_disjoint_pred = NamedNode::new(&format!("{}qualifiedValueShapesDisjoint", shacl_ns))
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid qualifiedValueShapesDisjoint predicate: {}", e)))?;
        
        // Look for qualified value shape
        if let Some(triple) = graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(qualified_shape_pred)),
            None
        ).first() {
            if let Ok(Term::NamedNode(shape_ref)) = object_to_term(triple.object()) {
                let mut qualified_min_count = None;
                let mut qualified_max_count = None;
                let mut qualified_disjoint = false;
                
                // Get qualified min count
                if let Some(min_triple) = graph.query_triples(
                    Some(&Subject::NamedNode(shape_node.clone())),
                    Some(&Predicate::NamedNode(qualified_min_pred)),
                    None
                ).first() {
                    if let Ok(Term::Literal(literal)) = object_to_term(min_triple.object()) {
                        qualified_min_count = literal.value().parse::<u32>().ok();
                    }
                }
                
                // Get qualified max count
                if let Some(max_triple) = graph.query_triples(
                    Some(&Subject::NamedNode(shape_node.clone())),
                    Some(&Predicate::NamedNode(qualified_max_pred)),
                    None
                ).first() {
                    if let Ok(Term::Literal(literal)) = object_to_term(max_triple.object()) {
                        qualified_max_count = literal.value().parse::<u32>().ok();
                    }
                }
                
                // Get qualified disjoint
                if let Some(disjoint_triple) = graph.query_triples(
                    Some(&Subject::NamedNode(shape_node.clone())),
                    Some(&Predicate::NamedNode(qualified_disjoint_pred)),
                    None
                ).first() {
                    if let Ok(Term::Literal(literal)) = object_to_term(disjoint_triple.object()) {
                        qualified_disjoint = literal.value() == "true";
                    }
                }
                
                let constraint = Constraint::QualifiedValueShape(QualifiedValueShapeConstraint {
                    qualified_value_shape: ShapeId::new(shape_ref.as_str()),
                    qualified_min_count,
                    qualified_max_count,
                    qualified_value_shapes_disjoint: qualified_disjoint,
                });
                
                shape.add_constraint(
                    ConstraintComponentId::new("sh:QualifiedValueShapeConstraintComponent"),
                    constraint
                );
            }
        }
        
        // TODO: Add more constraint parsing as needed
        
        Ok(())
    }
    
    /// Parse shape severity from a graph
    fn parse_shape_severity_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate};
        
        let severity_pred = NamedNode::new("http://www.w3.org/ns/shacl#severity")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid severity predicate: {}", e)))?;
        
        if let Some(triple) = graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(severity_pred)),
            None
        ).first() {
            if let Ok(Term::NamedNode(severity_node)) = object_to_term(triple.object()) {
                shape.severity = match severity_node.as_str() {
                    "http://www.w3.org/ns/shacl#Violation" => Severity::Violation,
                    "http://www.w3.org/ns/shacl#Warning" => Severity::Warning,
                    "http://www.w3.org/ns/shacl#Info" => Severity::Info,
                    _ => Severity::Violation, // Default
                };
            }
        }
        
        Ok(())
    }
    
    /// Parse shape message from a graph
    fn parse_shape_message_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate};
        
        let message_pred = NamedNode::new("http://www.w3.org/ns/shacl#message")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid message predicate: {}", e)))?;
        
        for (i, triple) in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(message_pred)),
            None
        ).into_iter().enumerate() {
            if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                // Messages is an IndexMap, so we need to insert with a key
                shape.messages.insert(
                    format!("message_{}", i),
                    literal.value().to_string()
                );
            }
        }
        
        Ok(())
    }
    
    /// Parse shape deactivated status from a graph
    fn parse_shape_deactivated_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate};
        
        let deactivated_pred = NamedNode::new("http://www.w3.org/ns/shacl#deactivated")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid deactivated predicate: {}", e)))?;
        
        if let Some(triple) = graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(deactivated_pred)),
            None
        ).first() {
            if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                shape.deactivated = literal.value() == "true";
            }
        }
        
        Ok(())
    }
    
    /// Parse shape inheritance (sh:extends) from a graph
    fn parse_shape_extends_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate};
        
        let extends_pred = NamedNode::new("http://www.w3.org/ns/shacl#extends")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid extends predicate: {}", e)))?;
        
        for triple in graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(extends_pred)),
            None
        ) {
            if let Ok(Term::NamedNode(parent_node)) = object_to_term(triple.object()) {
                shape.extends(ShapeId::new(parent_node.as_str()));
            }
        }
        
        Ok(())
    }
    
    /// Parse shape priority from a graph
    fn parse_shape_priority_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate};
        
        // Using a custom predicate for priority since it's not standard SHACL
        let priority_pred = NamedNode::new("http://www.w3.org/ns/shacl#priority")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid priority predicate: {}", e)))?;
        
        if let Some(triple) = graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(priority_pred)),
            None
        ).first() {
            if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                if let Ok(priority) = literal.value().parse::<i32>() {
                    shape.with_priority(priority);
                }
            }
        }
        
        Ok(())
    }
    
    /// Parse shape metadata from a graph
    fn parse_shape_metadata_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        use oxirs_core::model::{Subject, Predicate};
        use crate::ShapeMetadata;
        
        // Common metadata predicates
        let dc_ns = "http://purl.org/dc/elements/1.1/";
        let dcterms_ns = "http://purl.org/dc/terms/";
        let rdfs_ns = "http://www.w3.org/2000/01/rdf-schema#";
        
        // Parse author (dc:creator or dcterms:creator)
        for creator_pred_iri in &[
            format!("{}creator", dc_ns),
            format!("{}creator", dcterms_ns),
        ] {
            if let Ok(creator_pred) = NamedNode::new(creator_pred_iri) {
                if let Some(triple) = graph.query_triples(
                    Some(&Subject::NamedNode(shape_node.clone())),
                    Some(&Predicate::NamedNode(creator_pred)),
                    None
                ).first() {
                    if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                        shape.metadata.author = Some(literal.value().to_string());
                    }
                }
            }
        }
        
        // Parse creation date (dcterms:created)
        if let Ok(created_pred) = NamedNode::new(&format!("{}created", dcterms_ns)) {
            if let Some(triple) = graph.query_triples(
                Some(&Subject::NamedNode(shape_node.clone())),
                Some(&Predicate::NamedNode(created_pred)),
                None
            ).first() {
                if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                    if let Ok(datetime) = chrono::DateTime::parse_from_rfc3339(literal.value()) {
                        shape.metadata.created = Some(datetime.with_timezone(&chrono::Utc));
                    }
                }
            }
        }
        
        // Parse modification date (dcterms:modified)
        if let Ok(modified_pred) = NamedNode::new(&format!("{}modified", dcterms_ns)) {
            if let Some(triple) = graph.query_triples(
                Some(&Subject::NamedNode(shape_node.clone())),
                Some(&Predicate::NamedNode(modified_pred)),
                None
            ).first() {
                if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                    if let Ok(datetime) = chrono::DateTime::parse_from_rfc3339(literal.value()) {
                        shape.metadata.modified = Some(datetime.with_timezone(&chrono::Utc));
                    }
                }
            }
        }
        
        // Parse version (dcterms:hasVersion)
        if let Ok(version_pred) = NamedNode::new(&format!("{}hasVersion", dcterms_ns)) {
            if let Some(triple) = graph.query_triples(
                Some(&Subject::NamedNode(shape_node.clone())),
                Some(&Predicate::NamedNode(version_pred)),
                None
            ).first() {
                if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                    shape.metadata.version = Some(literal.value().to_string());
                }
            }
        }
        
        // Parse license (dcterms:license)
        if let Ok(license_pred) = NamedNode::new(&format!("{}license", dcterms_ns)) {
            if let Some(triple) = graph.query_triples(
                Some(&Subject::NamedNode(shape_node.clone())),
                Some(&Predicate::NamedNode(license_pred)),
                None
            ).first() {
                if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                    shape.metadata.license = Some(literal.value().to_string());
                } else if let Ok(Term::NamedNode(license_node)) = object_to_term(triple.object()) {
                    shape.metadata.license = Some(license_node.as_str().to_string());
                }
            }
        }
        
        // Parse label and description for metadata too
        if let Ok(label_pred) = NamedNode::new(&format!("{}label", rdfs_ns)) {
            if let Some(triple) = graph.query_triples(
                Some(&Subject::NamedNode(shape_node.clone())),
                Some(&Predicate::NamedNode(label_pred)),
                None
            ).first() {
                if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                    shape.label = Some(literal.value().to_string());
                }
            }
        }
        
        if let Ok(description_pred) = NamedNode::new(&format!("{}comment", rdfs_ns)) {
            if let Some(triple) = graph.query_triples(
                Some(&Subject::NamedNode(shape_node.clone())),
                Some(&Predicate::NamedNode(description_pred)),
                None
            ).first() {
                if let Ok(Term::Literal(literal)) = object_to_term(triple.object()) {
                    shape.description = Some(literal.value().to_string());
                }
            }
        }
        
        // TODO: Parse tags and custom properties from additional metadata
        
        Ok(())
    }

    /// Find all IRIs that represent shapes in the store
    fn find_shape_iris_in_store(
        &self,
        store: &Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<String>> {
        let mut shape_iris = HashSet::new();

        // Query for explicit shape declarations
        let shape_type_queries = vec![
            // NodeShape instances
            self.create_type_query("http://www.w3.org/ns/shacl#NodeShape", graph_name),
            // PropertyShape instances
            self.create_type_query("http://www.w3.org/ns/shacl#PropertyShape", graph_name),
        ];

        for query in shape_type_queries {
            tracing::debug!("Shape discovery query: {}", query);
            match self.execute_shape_query(store, &query) {
                Ok(results) => {
                    if let oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    } = results
                    {
                        for binding in bindings {
                            if let Some(shape_iri) = binding.get("shape") {
                                if let Term::NamedNode(node) = shape_iri {
                                    shape_iris.insert(node.as_str().to_string());
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to execute shape type query: {}", e);
                    // Continue with other queries
                }
            }
        }

        // Query for resources with shape-indicating properties
        let property_queries = vec![
            self.create_property_query("targetClass", graph_name),
            self.create_property_query("targetNode", graph_name),
            self.create_property_query("targetObjectsOf", graph_name),
            self.create_property_query("targetSubjectsOf", graph_name),
            self.create_property_query("property", graph_name),
            self.create_property_query("path", graph_name),
        ];

        for query in property_queries {
            tracing::debug!("Shape property discovery query: {}", query);
            match self.execute_shape_query(store, &query) {
                Ok(results) => {
                    if let oxirs_core::query::QueryResult::Select {
                        variables: _,
                        bindings,
                    } = results
                    {
                        for binding in bindings {
                            if let Some(shape_iri) = binding.get("shape") {
                                if let Term::NamedNode(node) = shape_iri {
                                    shape_iris.insert(node.as_str().to_string());
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to execute shape property query: {}", e);
                    // Continue with other queries
                }
            }
        }

        tracing::info!("Discovered {} shape IRIs in store", shape_iris.len());
        Ok(shape_iris.into_iter().collect())
    }

    /// Create a SPARQL query to find instances of a specific type
    fn create_type_query(&self, rdf_type: &str, graph_name: Option<&str>) -> String {
        if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?shape WHERE {{
                    GRAPH <{}> {{
                        ?shape a <{}> .
                    }}
                }}
            "#,
                graph, rdf_type
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?shape WHERE {{
                    ?shape a <{}> .
                }}
            "#,
                rdf_type
            )
        }
    }

    /// Create a SPARQL query to find resources with a specific property
    fn create_property_query(&self, property: &str, graph_name: Option<&str>) -> String {
        let property_iri = format!("http://www.w3.org/ns/shacl#{}", property);
        if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?shape WHERE {{
                    GRAPH <{}> {{
                        ?shape <{}> ?value .
                    }}
                }}
            "#,
                graph, property_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?shape WHERE {{
                    ?shape <{}> ?value .
                }}
            "#,
                property_iri
            )
        }
    }

    /// Parse a single shape from the store
    fn parse_shape_from_store(
        &mut self,
        store: &Store,
        shape_iri: &str,
        visited: &mut HashSet<String>,
        depth: usize,
        graph_name: Option<&str>,
    ) -> Result<Shape> {
        if depth > self.max_depth {
            return Err(ShaclError::ShapeParsing(format!(
                "Maximum parsing depth {} exceeded for shape {}",
                self.max_depth, shape_iri
            )));
        }

        visited.insert(shape_iri.to_string());

        // Check cache first
        if let Some(cached_shape) = self.shape_cache.get(shape_iri) {
            return Ok(cached_shape.clone());
        }

        // Determine shape type
        let shape_type = self.determine_shape_type_from_store(store, shape_iri, graph_name)?;

        // Create shape with basic properties
        let shape_id = ShapeId::new(shape_iri.to_string());
        let mut shape = Shape::new(shape_id, shape_type);

        // Parse shape metadata
        self.parse_shape_metadata_from_store(store, shape_iri, &mut shape, graph_name)?;

        // Parse targets
        self.parse_shape_targets_from_store(store, shape_iri, &mut shape, graph_name)?;

        // Parse property path (for property shapes)
        if shape.is_property_shape() {
            self.parse_property_path_from_store(store, shape_iri, &mut shape, graph_name)?;
        }

        // Parse constraints
        self.parse_shape_constraints_from_store(store, shape_iri, &mut shape, graph_name)?;

        // Cache the parsed shape
        self.shape_cache
            .insert(shape_iri.to_string(), shape.clone());

        Ok(shape)
    }

    /// Determine the type of a shape (node or property) from store
    fn determine_shape_type_from_store(
        &self,
        store: &Store,
        shape_iri: &str,
        graph_name: Option<&str>,
    ) -> Result<ShapeType> {
        // Check for explicit PropertyShape type declaration
        let property_shape_query = if let Some(graph) = graph_name {
            format!(
                r#"
                ASK {{
                    GRAPH <{}> {{
                        <{}> a <http://www.w3.org/ns/shacl#PropertyShape> .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                ASK {{
                    <{}> a <http://www.w3.org/ns/shacl#PropertyShape> .
                }}
            "#,
                shape_iri
            )
        };

        // Execute ASK query to check for PropertyShape type
        match self.execute_shape_query(store, &property_shape_query) {
            Ok(result) => {
                if let oxirs_core::query::QueryResult::Ask(is_property_shape) = result {
                    if is_property_shape {
                        return Ok(ShapeType::PropertyShape);
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to check PropertyShape type: {}", e);
            }
        }

        // Check for sh:path property (indicates property shape)
        let path_query = if let Some(graph) = graph_name {
            format!(
                r#"
                ASK {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#path> ?path .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                ASK {{
                    <{}> <http://www.w3.org/ns/shacl#path> ?path .
                }}
            "#,
                shape_iri
            )
        };

        // Execute ASK query to check for path property
        match self.execute_shape_query(store, &path_query) {
            Ok(result) => {
                if let oxirs_core::query::QueryResult::Ask(has_path) = result {
                    if has_path {
                        return Ok(ShapeType::PropertyShape);
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to check path property: {}", e);
            }
        }

        // Default to NodeShape
        tracing::debug!("Shape {} defaulting to NodeShape", shape_iri);
        Ok(ShapeType::NodeShape)
    }

    /// Parse shape metadata (label, description, etc.)
    fn parse_shape_metadata(
        &self,
        graph: &Graph,
        shape_iri: &str,
        shape: &mut Shape,
    ) -> Result<()> {
        // TODO: Implement metadata parsing
        // This would extract:
        // - sh:label
        // - sh:description
        // - sh:group
        // - sh:order
        // - sh:deactivated
        // - sh:severity
        // - sh:message (with language tags)

        Ok(())
    }

    /// Parse shape targets
    fn parse_shape_targets(&self, graph: &Graph, shape_iri: &str, shape: &mut Shape) -> Result<()> {
        // TODO: Implement target parsing
        // This would extract:
        // - sh:targetClass
        // - sh:targetNode
        // - sh:targetObjectsOf
        // - sh:targetSubjectsOf
        // - sh:target (SPARQL-based)
        // - Implicit targets (shape IRI as class)

        Ok(())
    }

    /// Parse property path for property shapes
    fn parse_property_path(&self, graph: &Graph, shape_iri: &str, shape: &mut Shape) -> Result<()> {
        // TODO: Implement property path parsing
        // This would parse sh:path and create PropertyPath objects
        // Supporting all path types: predicate, inverse, sequence, alternative, etc.

        Ok(())
    }

    /// Parse shape constraints
    fn parse_shape_constraints(
        &self,
        graph: &Graph,
        shape_iri: &str,
        shape: &mut Shape,
    ) -> Result<()> {
        // TODO: Implement constraint parsing
        // This would extract all SHACL Core constraints:
        // - Value constraints (sh:class, sh:datatype, sh:nodeKind, etc.)
        // - Cardinality constraints (sh:minCount, sh:maxCount)
        // - Range constraints (sh:minInclusive, sh:maxInclusive, etc.)
        // - String constraints (sh:minLength, sh:maxLength, sh:pattern, etc.)
        // - Logical constraints (sh:and, sh:or, sh:not, sh:xone)
        // - Shape-based constraints (sh:node, sh:qualifiedValueShape)
        // - Closed constraints (sh:closed, sh:ignoredProperties)
        // - SPARQL constraints (sh:sparql)

        Ok(())
    }

    /// Parse a specific constraint type
    fn parse_constraint(
        &self,
        graph: &Graph,
        shape_iri: &str,
        constraint_property: &str,
    ) -> Result<Option<Constraint>> {
        // TODO: Implement specific constraint parsing based on property

        Ok(None)
    }

    /// Parse property path from RDF representation
    fn parse_property_path_from_rdf(
        &self,
        graph: &Graph,
        path_node: &Term,
    ) -> Result<PropertyPath> {
        // TODO: Implement property path parsing from RDF
        // This needs to handle:
        // - Simple predicates
        // - Inverse paths (sh:inversePath)
        // - Sequence paths (rdf:List)
        // - Alternative paths (sh:alternativePath)
        // - Kleene star paths (sh:zeroOrMorePath, sh:oneOrMorePath, sh:zeroOrOnePath)

        // For now, assume it's a simple predicate
        match path_node {
            Term::NamedNode(node) => Ok(PropertyPath::predicate(node.clone())),
            _ => Err(ShaclError::PropertyPath(
                "Invalid property path node".to_string(),
            )),
        }
    }

    /// Parse shape metadata from store
    fn parse_shape_metadata_from_store(
        &self,
        _store: &Store,
        _shape_iri: &str,
        _shape: &mut Shape,
        _graph_name: Option<&str>,
    ) -> Result<()> {
        // TODO: Implement metadata parsing from store
        Ok(())
    }

    /// Execute a shape parsing query using oxirs-core query engine
    fn execute_shape_query(
        &self,
        store: &Store,
        query: &str,
    ) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;

        let query_engine = QueryEngine::new();

        tracing::debug!("Executing shape parsing query: {}", query);

        let result = query_engine
            .query(query, store)
            .map_err(|e| ShaclError::ShapeParsing(format!("Shape parsing query failed: {}", e)))?;

        Ok(result)
    }

    /// Parse shape targets from store
    fn parse_shape_targets_from_store(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        use crate::targets::Target;

        // Parse sh:targetClass
        let target_class_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?class WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetClass> ?class .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?class WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetClass> ?class .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &target_class_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(class_term) = binding.get("class") {
                            if let Term::NamedNode(class_node) = class_term {
                                shape.add_target(Target::class(class_node.clone()));
                                tracing::debug!("Added target class: {}", class_node.as_str());
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse target classes: {}", e);
            }
        }

        // Parse sh:targetNode
        let target_node_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?node WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetNode> ?node .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?node WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetNode> ?node .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &target_node_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(node_term) = binding.get("node") {
                            shape.add_target(Target::node(node_term.clone()));
                            tracing::debug!("Added target node: {:?}", node_term);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse target nodes: {}", e);
            }
        }

        // Parse sh:targetObjectsOf
        let target_objects_of_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?property WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetObjectsOf> ?property .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?property WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetObjectsOf> ?property .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &target_objects_of_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(property_term) = binding.get("property") {
                            if let Term::NamedNode(property_node) = property_term {
                                shape.add_target(Target::objects_of(property_node.clone()));
                                tracing::debug!(
                                    "Added target objects of: {}",
                                    property_node.as_str()
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse target objects of: {}", e);
            }
        }

        // Parse sh:targetSubjectsOf
        let target_subjects_of_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?property WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetSubjectsOf> ?property .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?property WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetSubjectsOf> ?property .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &target_subjects_of_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(property_term) = binding.get("property") {
                            if let Term::NamedNode(property_node) = property_term {
                                shape.add_target(Target::subjects_of(property_node.clone()));
                                tracing::debug!(
                                    "Added target subjects of: {}",
                                    property_node.as_str()
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse target subjects of: {}", e);
            }
        }

        Ok(())
    }

    /// Parse property path from store
    fn parse_property_path_from_store(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        use crate::paths::PropertyPath;

        let path_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?path WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#path> ?path .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?path WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#path> ?path .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &path_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(path_term) = binding.get("path") {
                            // Parse the property path from the RDF term
                            match self.parse_property_path_from_term(store, path_term, graph_name) {
                                Ok(property_path) => {
                                    shape.path = Some(property_path);
                                    tracing::debug!("Parsed property path for shape {}", shape_iri);
                                    break; // Only take the first path found
                                }
                                Err(e) => {
                                    tracing::error!("Failed to parse property path: {}", e);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to query for property path: {}", e);
            }
        }

        Ok(())
    }

    /// Parse property path from an RDF term (supports complex paths)
    fn parse_property_path_from_term(
        &self,
        store: &Store,
        path_term: &Term,
        graph_name: Option<&str>,
    ) -> Result<PropertyPath> {
        use crate::paths::PropertyPath;

        match path_term {
            Term::NamedNode(predicate_node) => Ok(PropertyPath::predicate(predicate_node.clone())),
            Term::BlankNode(blank_node) => {
                // Complex property path - need to analyze the structure
                self.parse_complex_property_path(store, blank_node, graph_name)
            }
            _ => Err(ShaclError::PropertyPath(
                "Invalid property path term - must be a named node or blank node".to_string(),
            )),
        }
    }
    
    /// Parse complex property paths (sequences, alternatives, inverse paths, etc.)
    fn parse_complex_property_path(
        &self,
        store: &Store,
        blank_node: &BlankNode,
        graph_name: Option<&str>,
    ) -> Result<PropertyPath> {
        use crate::paths::PropertyPath;
        
        // Check for inverse path (sh:inversePath)
        if let Some(inverse_path) = self.get_property_value(
            store, 
            &Term::BlankNode(blank_node.clone()), 
            "http://www.w3.org/ns/shacl#inversePath",
            graph_name
        )? {
            let inner_path = self.parse_property_path_from_term(store, &inverse_path, graph_name)?;
            return Ok(PropertyPath::inverse(inner_path));
        }
        
        // Check for sequence path (represented as RDF list)
        if let Some(paths) = self.parse_path_list(store, blank_node, graph_name)? {
            if paths.len() > 1 {
                return Ok(PropertyPath::sequence(paths));
            } else if paths.len() == 1 {
                return Ok(paths.into_iter().next().unwrap());
            }
        }
        
        // Check for alternative path (sh:alternativePath)
        if let Some(alt_list_node) = self.get_property_value(
            store,
            &Term::BlankNode(blank_node.clone()),
            "http://www.w3.org/ns/shacl#alternativePath",
            graph_name
        )? {
            if let Term::BlankNode(list_blank) = alt_list_node {
                if let Some(paths) = self.parse_path_list(store, &list_blank, graph_name)? {
                    return Ok(PropertyPath::alternative(paths));
                }
            }
        }
        
        // Check for zero-or-more path (sh:zeroOrMorePath)
        if let Some(zero_or_more_path) = self.get_property_value(
            store,
            &Term::BlankNode(blank_node.clone()),
            "http://www.w3.org/ns/shacl#zeroOrMorePath",
            graph_name
        )? {
            let inner_path = self.parse_property_path_from_term(store, &zero_or_more_path, graph_name)?;
            return Ok(PropertyPath::zero_or_more(inner_path));
        }
        
        // Check for one-or-more path (sh:oneOrMorePath)
        if let Some(one_or_more_path) = self.get_property_value(
            store,
            &Term::BlankNode(blank_node.clone()),
            "http://www.w3.org/ns/shacl#oneOrMorePath",
            graph_name
        )? {
            let inner_path = self.parse_property_path_from_term(store, &one_or_more_path, graph_name)?;
            return Ok(PropertyPath::one_or_more(inner_path));
        }
        
        // Check for zero-or-one path (sh:zeroOrOnePath)
        if let Some(zero_or_one_path) = self.get_property_value(
            store,
            &Term::BlankNode(blank_node.clone()),
            "http://www.w3.org/ns/shacl#zeroOrOnePath",
            graph_name
        )? {
            let inner_path = self.parse_property_path_from_term(store, &zero_or_one_path, graph_name)?;
            return Ok(PropertyPath::zero_or_one(inner_path));
        }
        
        // If none of the above, it might be a direct RDF list (sequence path)
        // Try to parse as a list
        if self.is_rdf_list(store, blank_node, graph_name)? {
            if let Some(paths) = self.parse_path_list(store, blank_node, graph_name)? {
                if paths.len() > 1 {
                    return Ok(PropertyPath::sequence(paths));
                } else if paths.len() == 1 {
                    return Ok(paths.into_iter().next().unwrap());
                }
            }
        }
        
        Err(ShaclError::PropertyPath(format!(
            "Unable to parse complex property path from blank node: {}",
            blank_node.as_str()
        )))
    }
    
    /// Get a property value from the store
    fn get_property_value(
        &self,
        store: &Store,
        subject: &Term,
        predicate: &str,
        graph_name: Option<&str>,
    ) -> Result<Option<Term>> {
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT ?value WHERE {{
                    GRAPH <{}> {{
                        {} <{}> ?value .
                    }}
                }}
                LIMIT 1
                "#,
                graph, 
                format_term_for_sparql(subject)?,
                predicate
            )
        } else {
            format!(
                r#"
                SELECT ?value WHERE {{
                    {} <{}> ?value .
                }}
                LIMIT 1
                "#,
                format_term_for_sparql(subject)?,
                predicate
            )
        };
        
        match self.execute_shape_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    if let Some(binding) = bindings.first() {
                        return Ok(binding.get("value").cloned());
                    }
                }
                Ok(None)
            }
            Err(_) => Ok(None),
        }
    }
    
    /// Check if a blank node represents an RDF list
    fn is_rdf_list(
        &self,
        store: &Store,
        blank_node: &BlankNode,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        let first_value = self.get_property_value(
            store,
            &Term::BlankNode(blank_node.clone()),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#first",
            graph_name
        )?;
        
        Ok(first_value.is_some())
    }
    
    /// Parse an RDF list of property paths
    fn parse_path_list(
        &self,
        store: &Store,
        list_node: &BlankNode,
        graph_name: Option<&str>,
    ) -> Result<Option<Vec<PropertyPath>>> {
        let mut paths = Vec::new();
        let mut current = Term::BlankNode(list_node.clone());
        
        loop {
            // Get the first element
            if let Some(first) = self.get_property_value(
                store,
                &current,
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#first",
                graph_name
            )? {
                // Parse the path element
                let path = self.parse_property_path_from_term(store, &first, graph_name)?;
                paths.push(path);
                
                // Get the rest of the list
                if let Some(rest) = self.get_property_value(
                    store,
                    &current,
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest",
                    graph_name
                )? {
                    // Check if we've reached rdf:nil
                    if let Term::NamedNode(nn) = &rest {
                        if nn.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil" {
                            break;
                        }
                    }
                    current = rest;
                } else {
                    break;
                }
            } else {
                // Not a valid list
                return Ok(None);
            }
        }
        
        if paths.is_empty() {
            Ok(None)
        } else {
            Ok(Some(paths))
        }
    }

    /// Parse shape constraints from store
    fn parse_shape_constraints_from_store(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        use crate::constraints::*;
        use crate::{Constraint, ConstraintComponentId};

        // Parse sh:class constraint
        self.parse_class_constraint(store, shape_iri, shape, graph_name)?;

        // Parse sh:datatype constraint
        self.parse_datatype_constraint(store, shape_iri, shape, graph_name)?;

        // Parse cardinality constraints (sh:minCount, sh:maxCount)
        self.parse_cardinality_constraints(store, shape_iri, shape, graph_name)?;

        // Parse string constraints (sh:minLength, sh:maxLength, sh:pattern)
        self.parse_string_constraints(store, shape_iri, shape, graph_name)?;

        // Parse value constraints (sh:in, sh:hasValue)
        self.parse_value_constraints(store, shape_iri, shape, graph_name)?;

        // Parse range constraints (sh:minInclusive, sh:maxInclusive, etc.)
        self.parse_range_constraints(store, shape_iri, shape, graph_name)?;

        tracing::debug!(
            "Parsed {} constraints for shape {}",
            shape.constraints.len(),
            shape_iri
        );
        Ok(())
    }

    /// Parse sh:class constraint
    fn parse_class_constraint(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?class WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#class> ?class .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?class WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#class> ?class .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(class_term) = binding.get("class") {
                            if let Term::NamedNode(class_node) = class_term {
                                let constraint = Constraint::Class(ClassConstraint {
                                    class_iri: class_node.clone(),
                                });
                                shape.add_constraint(
                                    ConstraintComponentId::new("sh:ClassConstraintComponent"),
                                    constraint,
                                );
                                tracing::debug!("Added class constraint: {}", class_node.as_str());
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse class constraint: {}", e);
            }
        }

        Ok(())
    }

    /// Parse sh:datatype constraint
    fn parse_datatype_constraint(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        let query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?datatype WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#datatype> ?datatype .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?datatype WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#datatype> ?datatype .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(datatype_term) = binding.get("datatype") {
                            if let Term::NamedNode(datatype_node) = datatype_term {
                                let constraint = Constraint::Datatype(DatatypeConstraint {
                                    datatype_iri: datatype_node.clone(),
                                });
                                shape.add_constraint(
                                    ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
                                    constraint,
                                );
                                tracing::debug!(
                                    "Added datatype constraint: {}",
                                    datatype_node.as_str()
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse datatype constraint: {}", e);
            }
        }

        Ok(())
    }

    /// Parse cardinality constraints (sh:minCount, sh:maxCount)
    fn parse_cardinality_constraints(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        // Parse sh:minCount
        let min_count_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?minCount WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#minCount> ?minCount .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?minCount WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#minCount> ?minCount .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &min_count_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(min_count_term) = binding.get("minCount") {
                            if let Term::Literal(literal) = min_count_term {
                                if let Ok(min_count) = literal.as_str().parse::<u32>() {
                                    let constraint =
                                        Constraint::MinCount(MinCountConstraint { min_count });
                                    shape.add_constraint(
                                        ConstraintComponentId::new(
                                            "sh:MinCountConstraintComponent",
                                        ),
                                        constraint,
                                    );
                                    tracing::debug!("Added minCount constraint: {}", min_count);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse minCount constraint: {}", e);
            }
        }

        // Parse sh:maxCount
        let max_count_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?maxCount WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#maxCount> ?maxCount .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?maxCount WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#maxCount> ?maxCount .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &max_count_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(max_count_term) = binding.get("maxCount") {
                            if let Term::Literal(literal) = max_count_term {
                                if let Ok(max_count) = literal.as_str().parse::<u32>() {
                                    let constraint =
                                        Constraint::MaxCount(MaxCountConstraint { max_count });
                                    shape.add_constraint(
                                        ConstraintComponentId::new(
                                            "sh:MaxCountConstraintComponent",
                                        ),
                                        constraint,
                                    );
                                    tracing::debug!("Added maxCount constraint: {}", max_count);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse maxCount constraint: {}", e);
            }
        }

        Ok(())
    }

    /// Parse string constraints (sh:minLength, sh:maxLength, sh:pattern)
    fn parse_string_constraints(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        // Parse sh:minLength
        let min_length_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?minLength WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#minLength> ?minLength .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?minLength WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#minLength> ?minLength .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &min_length_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(min_length_term) = binding.get("minLength") {
                            if let Term::Literal(literal) = min_length_term {
                                if let Ok(min_length) = literal.as_str().parse::<u32>() {
                                    let constraint =
                                        Constraint::MinLength(MinLengthConstraint { min_length });
                                    shape.add_constraint(
                                        ConstraintComponentId::new(
                                            "sh:MinLengthConstraintComponent",
                                        ),
                                        constraint,
                                    );
                                    tracing::debug!("Added minLength constraint: {}", min_length);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse minLength constraint: {}", e);
            }
        }

        // Parse sh:maxLength
        let max_length_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?maxLength WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#maxLength> ?maxLength .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?maxLength WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#maxLength> ?maxLength .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &max_length_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(max_length_term) = binding.get("maxLength") {
                            if let Term::Literal(literal) = max_length_term {
                                if let Ok(max_length) = literal.as_str().parse::<u32>() {
                                    let constraint =
                                        Constraint::MaxLength(MaxLengthConstraint { max_length });
                                    shape.add_constraint(
                                        ConstraintComponentId::new(
                                            "sh:MaxLengthConstraintComponent",
                                        ),
                                        constraint,
                                    );
                                    tracing::debug!("Added maxLength constraint: {}", max_length);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse maxLength constraint: {}", e);
            }
        }

        Ok(())
    }

    /// Parse value constraints (sh:in, sh:hasValue) - basic implementation
    fn parse_value_constraints(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        // Parse sh:hasValue
        let has_value_query = if let Some(graph) = graph_name {
            format!(
                r#"
                SELECT DISTINCT ?value WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#hasValue> ?value .
                    }}
                }}
            "#,
                graph, shape_iri
            )
        } else {
            format!(
                r#"
                SELECT DISTINCT ?value WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#hasValue> ?value .
                }}
            "#,
                shape_iri
            )
        };

        match self.execute_shape_query(store, &has_value_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select {
                    variables: _,
                    bindings,
                } = results
                {
                    for binding in bindings {
                        if let Some(value_term) = binding.get("value") {
                            let constraint = Constraint::HasValue(HasValueConstraint {
                                value: value_term.clone(),
                            });
                            shape.add_constraint(
                                ConstraintComponentId::new("sh:HasValueConstraintComponent"),
                                constraint,
                            );
                            tracing::debug!("Added hasValue constraint: {}", value_term.as_str());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse hasValue constraint: {}", e);
            }
        }

        // TODO: Implement sh:in constraint (requires parsing RDF lists)

        Ok(())
    }

    /// Parse range constraints (sh:minInclusive, sh:maxInclusive, etc.) - basic implementation
    fn parse_range_constraints(
        &self,
        store: &Store,
        shape_iri: &str,
        shape: &mut Shape,
        graph_name: Option<&str>,
    ) -> Result<()> {
        // For now, just log that range constraints would be parsed here
        // TODO: Implement full range constraint parsing
        tracing::debug!(
            "Range constraint parsing not yet fully implemented for shape {}",
            shape_iri
        );
        Ok(())
    }

    /// Clear the shape cache
    pub fn clear_cache(&mut self) {
        self.shape_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> ShapeCacheStats {
        ShapeCacheStats {
            entries: self.shape_cache.len(),
            total_constraints: self
                .shape_cache
                .values()
                .map(|shape| shape.constraints.len())
                .sum(),
        }
    }
}

impl Default for ShapeParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about shape parsing cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeCacheStats {
    pub entries: usize,
    pub total_constraints: usize,
}

/// Shape validation context during parsing
#[derive(Debug, Clone)]
pub struct ShapeParsingContext {
    /// Current parsing depth
    pub depth: usize,

    /// Visited shape IRIs (for circular reference detection)
    pub visited: HashSet<String>,

    /// Parsing configuration
    pub config: ShapeParsingConfig,

    /// Performance statistics
    pub stats: ShapeParsingStats,
}

impl ShapeParsingContext {
    pub fn new(config: ShapeParsingConfig) -> Self {
        Self {
            depth: 0,
            visited: HashSet::new(),
            config,
            stats: ShapeParsingStats::default(),
        }
    }
}

/// Configuration for shape parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeParsingConfig {
    /// Enable strict parsing mode
    pub strict_mode: bool,

    /// Maximum recursion depth
    pub max_depth: usize,

    /// Follow implicit targets
    pub follow_implicit_targets: bool,

    /// Parse SPARQL constraints
    pub parse_sparql_constraints: bool,

    /// Validate constraints during parsing
    pub validate_constraints: bool,

    /// Cache parsed shapes
    pub cache_shapes: bool,
}

impl Default for ShapeParsingConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_depth: 50,
            follow_implicit_targets: true,
            parse_sparql_constraints: true,
            validate_constraints: true,
            cache_shapes: true,
        }
    }
}

/// Shape parsing performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeParsingStats {
    pub total_shapes_parsed: usize,
    pub total_constraints_parsed: usize,
    pub total_targets_parsed: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub parsing_errors: usize,
    pub avg_constraints_per_shape: f64,
}

impl ShapeParsingStats {
    pub fn record_shape_parsed(
        &mut self,
        constraints_count: usize,
        targets_count: usize,
        cache_hit: bool,
    ) {
        self.total_shapes_parsed += 1;
        self.total_constraints_parsed += constraints_count;
        self.total_targets_parsed += targets_count;

        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        self.avg_constraints_per_shape =
            self.total_constraints_parsed as f64 / self.total_shapes_parsed as f64;
    }

    pub fn record_parsing_error(&mut self) {
        self.parsing_errors += 1;
    }

    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_shapes_parsed == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_shapes_parsed as f64
        }
    }
}

/// Shape factory for creating shapes programmatically
#[derive(Debug)]
pub struct ShapeFactory;

impl ShapeFactory {
    /// Create a simple node shape with class constraint
    pub fn node_shape_with_class(shape_id: ShapeId, class_iri: NamedNode) -> Shape {
        let mut shape = Shape::node_shape(shape_id);
        shape.add_target(Target::class(class_iri.clone()));
        shape.add_constraint(
            ConstraintComponentId::new("sh:ClassConstraintComponent"),
            Constraint::Class(ClassConstraint { class_iri }),
        );
        shape
    }

    /// Create a property shape with basic constraints
    pub fn property_shape_with_constraints(
        shape_id: ShapeId,
        path: PropertyPath,
        constraints: Vec<(ConstraintComponentId, Constraint)>,
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);
        for (id, constraint) in constraints {
            shape.add_constraint(id, constraint);
        }
        shape
    }

    /// Create a string property shape with length constraints
    pub fn string_property_shape(
        shape_id: ShapeId,
        path: PropertyPath,
        min_length: Option<u32>,
        max_length: Option<u32>,
        pattern: Option<String>,
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);

        // Add datatype constraint for string
        let xsd_string = NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap();
        shape.add_constraint(
            ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
            Constraint::Datatype(DatatypeConstraint {
                datatype_iri: xsd_string,
            }),
        );

        // Add length constraints
        if let Some(min) = min_length {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MinLengthConstraintComponent"),
                Constraint::MinLength(MinLengthConstraint { min_length: min }),
            );
        }

        if let Some(max) = max_length {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MaxLengthConstraintComponent"),
                Constraint::MaxLength(MaxLengthConstraint { max_length: max }),
            );
        }

        // Add pattern constraint
        if let Some(pattern_str) = pattern {
            shape.add_constraint(
                ConstraintComponentId::new("sh:PatternConstraintComponent"),
                Constraint::Pattern(PatternConstraint {
                    pattern: pattern_str,
                    flags: None,
                    message: None,
                }),
            );
        }

        shape
    }

    /// Create a cardinality-constrained property shape
    pub fn cardinality_property_shape(
        shape_id: ShapeId,
        path: PropertyPath,
        min_count: Option<u32>,
        max_count: Option<u32>,
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);

        if let Some(min) = min_count {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MinCountConstraintComponent"),
                Constraint::MinCount(MinCountConstraint { min_count: min }),
            );
        }

        if let Some(max) = max_count {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MaxCountConstraintComponent"),
                Constraint::MaxCount(MaxCountConstraint { max_count: max }),
            );
        }

        shape
    }
}

/// Format a term for use in SPARQL queries
fn format_term_for_sparql(term: &Term) -> Result<String> {
    match term {
        Term::NamedNode(node) => Ok(format!("<{}>", node.as_str())),
        Term::BlankNode(node) => Ok(node.as_str().to_string()),
        Term::Literal(literal) => {
            // TODO: Proper literal formatting with datatype and language
            Ok(format!("\"{}\"", literal.as_str().replace('"', "\\\"")))
        }
        Term::Variable(var) => Ok(format!("?{}", var.name())),
        Term::QuotedTriple(_) => Err(ShaclError::PropertyPath(
            "Quoted triples not supported in property paths".to_string()
        )),
    }
}

/// Convert an RDF Object to a Term
fn object_to_term(object: &oxirs_core::model::Object) -> Result<Term> {
    use oxirs_core::model::Object;
    
    match object {
        Object::NamedNode(nn) => Ok(Term::NamedNode(nn.clone())),
        Object::BlankNode(bn) => Ok(Term::BlankNode(bn.clone())),
        Object::Literal(lit) => Ok(Term::Literal(lit.clone())),
        Object::Variable(var) => Ok(Term::Variable(var.clone())),
        Object::QuotedTriple(qt) => Ok(Term::QuotedTriple(qt.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_parser_creation() {
        let parser = ShapeParser::new();
        assert!(!parser.strict_mode);
        assert_eq!(parser.max_depth, 50);

        let strict_parser = ShapeParser::new_strict();
        assert!(strict_parser.strict_mode);
    }

    #[test]
    fn test_shape_factory_node_shape() {
        let shape_id = ShapeId::new("http://example.org/PersonShape");
        let person_class = NamedNode::new("http://example.org/Person").unwrap();

        let shape = ShapeFactory::node_shape_with_class(shape_id.clone(), person_class.clone());

        assert_eq!(shape.id, shape_id);
        assert!(shape.is_node_shape());
        assert_eq!(shape.targets.len(), 1);
        assert_eq!(shape.constraints.len(), 1);

        match &shape.targets[0] {
            Target::Class(class) => assert_eq!(class, &person_class),
            _ => panic!("Expected class target"),
        }
    }

    #[test]
    fn test_shape_factory_string_property() {
        let shape_id = ShapeId::new("http://example.org/NameShape");
        let name_path = PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap());

        let shape = ShapeFactory::string_property_shape(
            shape_id.clone(),
            name_path,
            Some(1),
            Some(100),
            Some("^[A-Za-z ]+$".to_string()),
        );

        assert_eq!(shape.id, shape_id);
        assert!(shape.is_property_shape());
        assert_eq!(shape.constraints.len(), 4); // datatype + minLength + maxLength + pattern
    }

    #[test]
    fn test_shape_factory_cardinality_property() {
        let shape_id = ShapeId::new("http://example.org/AgeShape");
        let age_path = PropertyPath::predicate(NamedNode::new("http://example.org/age").unwrap());

        let shape =
            ShapeFactory::cardinality_property_shape(shape_id.clone(), age_path, Some(1), Some(1));

        assert_eq!(shape.id, shape_id);
        assert!(shape.is_property_shape());
        assert_eq!(shape.constraints.len(), 2); // minCount + maxCount
    }

    #[test]
    fn test_parsing_config() {
        let config = ShapeParsingConfig::default();
        assert!(!config.strict_mode);
        assert_eq!(config.max_depth, 50);
        assert!(config.follow_implicit_targets);
        assert!(config.parse_sparql_constraints);
        assert!(config.validate_constraints);
        assert!(config.cache_shapes);
    }

    #[test]
    fn test_parsing_stats() {
        let mut stats = ShapeParsingStats::default();

        stats.record_shape_parsed(3, 2, false); // cache miss
        stats.record_shape_parsed(5, 1, true); // cache hit
        stats.record_parsing_error();

        assert_eq!(stats.total_shapes_parsed, 2);
        assert_eq!(stats.total_constraints_parsed, 8);
        assert_eq!(stats.total_targets_parsed, 3);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.parsing_errors, 1);
        assert_eq!(stats.avg_constraints_per_shape, 4.0);
        assert_eq!(stats.cache_hit_rate(), 0.5);
    }
}
