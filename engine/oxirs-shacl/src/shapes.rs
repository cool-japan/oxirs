//! SHACL shape parsing and representation
//! 
//! This module handles parsing SHACL shapes from RDF data.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{NamedNode, Term, Triple, BlankNode, Literal, RdfTerm},
    store::Store,
    graph::Graph,
    OxirsError,
};

use crate::{
    ShaclError, Result, Shape, ShapeId, ShapeType, PropertyPath, Target, Constraint,
    ConstraintComponentId, Severity, SHACL_VOCAB, SHACL_NS,
    constraints::*,
    targets::*,
    paths::*,
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
    pub fn parse_shapes_from_store(&mut self, store: &Store, graph_name: Option<&str>) -> Result<Vec<Shape>> {
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
    pub fn parse_shapes_from_rdf(&mut self, rdf_data: &str, format: &str, base_iri: Option<&str>) -> Result<Vec<Shape>> {
        // TODO: Implement shape parsing from RDF string
        // This would parse the RDF data and extract shape definitions
        
        Err(ShaclError::ShapeParsing("RDF string-based shape parsing not yet implemented".to_string()))
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
    fn find_shape_iris(&self, _graph: &Graph) -> Result<Vec<String>> {
        // TODO: Implement shape IRI discovery from graph
        Ok(Vec::new())
    }
    
    /// Parse a single shape from a graph
    fn parse_shape(&mut self, _graph: &Graph, _shape_iri: &str, _visited: &mut HashSet<String>, _depth: usize) -> Result<Shape> {
        // TODO: Implement shape parsing from graph
        Ok(Shape::new(
            ShapeId(_shape_iri.to_string()), 
            ShapeType::NodeShape
        ))
    }

    /// Find all IRIs that represent shapes in the store
    fn find_shape_iris_in_store(&self, store: &Store, graph_name: Option<&str>) -> Result<Vec<String>> {
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
                    if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
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
                    if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
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
            format!(r#"
                SELECT DISTINCT ?shape WHERE {{
                    GRAPH <{}> {{
                        ?shape a <{}> .
                    }}
                }}
            "#, graph, rdf_type)
        } else {
            format!(r#"
                SELECT DISTINCT ?shape WHERE {{
                    ?shape a <{}> .
                }}
            "#, rdf_type)
        }
    }
    
    /// Create a SPARQL query to find resources with a specific property
    fn create_property_query(&self, property: &str, graph_name: Option<&str>) -> String {
        let property_iri = format!("http://www.w3.org/ns/shacl#{}", property);
        if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?shape WHERE {{
                    GRAPH <{}> {{
                        ?shape <{}> ?value .
                    }}
                }}
            "#, graph, property_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?shape WHERE {{
                    ?shape <{}> ?value .
                }}
            "#, property_iri)
        }
    }
    
    /// Parse a single shape from the store
    fn parse_shape_from_store(&mut self, store: &Store, shape_iri: &str, visited: &mut HashSet<String>, depth: usize, graph_name: Option<&str>) -> Result<Shape> {
        if depth > self.max_depth {
            return Err(ShaclError::ShapeParsing(
                format!("Maximum parsing depth {} exceeded for shape {}", self.max_depth, shape_iri)
            ));
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
        self.shape_cache.insert(shape_iri.to_string(), shape.clone());
        
        Ok(shape)
    }
    
    /// Determine the type of a shape (node or property) from store
    fn determine_shape_type_from_store(&self, store: &Store, shape_iri: &str, graph_name: Option<&str>) -> Result<ShapeType> {
        // Check for explicit PropertyShape type declaration
        let property_shape_query = if let Some(graph) = graph_name {
            format!(r#"
                ASK {{
                    GRAPH <{}> {{
                        <{}> a <http://www.w3.org/ns/shacl#PropertyShape> .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                ASK {{
                    <{}> a <http://www.w3.org/ns/shacl#PropertyShape> .
                }}
            "#, shape_iri)
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
            format!(r#"
                ASK {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#path> ?path .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                ASK {{
                    <{}> <http://www.w3.org/ns/shacl#path> ?path .
                }}
            "#, shape_iri)
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
    fn parse_shape_metadata(&self, graph: &Graph, shape_iri: &str, shape: &mut Shape) -> Result<()> {
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
    fn parse_shape_constraints(&self, graph: &Graph, shape_iri: &str, shape: &mut Shape) -> Result<()> {
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
    fn parse_constraint(&self, graph: &Graph, shape_iri: &str, constraint_property: &str) -> Result<Option<Constraint>> {
        // TODO: Implement specific constraint parsing based on property
        
        Ok(None)
    }
    
    /// Parse property path from RDF representation
    fn parse_property_path_from_rdf(&self, graph: &Graph, path_node: &Term) -> Result<PropertyPath> {
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
            _ => Err(ShaclError::PropertyPath("Invalid property path node".to_string())),
        }
    }
    
    /// Parse shape metadata from store
    fn parse_shape_metadata_from_store(&self, _store: &Store, _shape_iri: &str, _shape: &mut Shape, _graph_name: Option<&str>) -> Result<()> {
        // TODO: Implement metadata parsing from store
        Ok(())
    }
    
    /// Execute a shape parsing query using oxirs-core query engine
    fn execute_shape_query(&self, store: &Store, query: &str) -> Result<oxirs_core::query::QueryResult> {
        use oxirs_core::query::QueryEngine;
        
        let query_engine = QueryEngine::new();
        
        tracing::debug!("Executing shape parsing query: {}", query);
        
        let result = query_engine.query(query, store)
            .map_err(|e| ShaclError::ShapeParsing(format!("Shape parsing query failed: {}", e)))?;
        
        Ok(result)
    }
    
    /// Parse shape targets from store
    fn parse_shape_targets_from_store(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        use crate::targets::Target;
        
        // Parse sh:targetClass
        let target_class_query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?class WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetClass> ?class .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?class WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetClass> ?class .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &target_class_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
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
            format!(r#"
                SELECT DISTINCT ?node WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetNode> ?node .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?node WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetNode> ?node .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &target_node_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
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
            format!(r#"
                SELECT DISTINCT ?property WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetObjectsOf> ?property .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?property WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetObjectsOf> ?property .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &target_objects_of_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(property_term) = binding.get("property") {
                            if let Term::NamedNode(property_node) = property_term {
                                shape.add_target(Target::objects_of(property_node.clone()));
                                tracing::debug!("Added target objects of: {}", property_node.as_str());
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
            format!(r#"
                SELECT DISTINCT ?property WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#targetSubjectsOf> ?property .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?property WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#targetSubjectsOf> ?property .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &target_subjects_of_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(property_term) = binding.get("property") {
                            if let Term::NamedNode(property_node) = property_term {
                                shape.add_target(Target::subjects_of(property_node.clone()));
                                tracing::debug!("Added target subjects of: {}", property_node.as_str());
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
    fn parse_property_path_from_store(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        use crate::paths::PropertyPath;
        
        let path_query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?path WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#path> ?path .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?path WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#path> ?path .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &path_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
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
    
    /// Parse property path from an RDF term (simple implementation)
    fn parse_property_path_from_term(&self, _store: &Store, path_term: &Term, _graph_name: Option<&str>) -> Result<PropertyPath> {
        use crate::paths::PropertyPath;
        
        // For now, implement simple predicate paths only
        // TODO: Implement complex property path parsing (inverse, sequence, alternative, etc.)
        match path_term {
            Term::NamedNode(predicate_node) => {
                Ok(PropertyPath::predicate(predicate_node.clone()))
            }
            Term::BlankNode(_) => {
                // This would be a complex property path (sequence, alternative, etc.)
                // For now, return an error
                Err(ShaclError::PropertyPath(
                    "Complex property paths (sequences, alternatives, etc.) not yet implemented".to_string()
                ))
            }
            _ => {
                Err(ShaclError::PropertyPath(
                    "Invalid property path term - must be a named node or blank node".to_string()
                ))
            }
        }
    }
    
    /// Parse shape constraints from store
    fn parse_shape_constraints_from_store(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        use crate::constraints::*;
        use crate::{ConstraintComponentId, Constraint};
        
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
        
        tracing::debug!("Parsed {} constraints for shape {}", shape.constraints.len(), shape_iri);
        Ok(())
    }
    
    /// Parse sh:class constraint
    fn parse_class_constraint(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        let query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?class WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#class> ?class .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?class WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#class> ?class .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(class_term) = binding.get("class") {
                            if let Term::NamedNode(class_node) = class_term {
                                let constraint = Constraint::Class(ClassConstraint { 
                                    class_iri: class_node.clone() 
                                });
                                shape.add_constraint(
                                    ConstraintComponentId::new("sh:ClassConstraintComponent"),
                                    constraint
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
    fn parse_datatype_constraint(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        let query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?datatype WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#datatype> ?datatype .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?datatype WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#datatype> ?datatype .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(datatype_term) = binding.get("datatype") {
                            if let Term::NamedNode(datatype_node) = datatype_term {
                                let constraint = Constraint::Datatype(DatatypeConstraint { 
                                    datatype_iri: datatype_node.clone() 
                                });
                                shape.add_constraint(
                                    ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
                                    constraint
                                );
                                tracing::debug!("Added datatype constraint: {}", datatype_node.as_str());
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
    fn parse_cardinality_constraints(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        // Parse sh:minCount
        let min_count_query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?minCount WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#minCount> ?minCount .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?minCount WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#minCount> ?minCount .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &min_count_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(min_count_term) = binding.get("minCount") {
                            if let Term::Literal(literal) = min_count_term {
                                if let Ok(min_count) = literal.as_str().parse::<u32>() {
                                    let constraint = Constraint::MinCount(MinCountConstraint { min_count });
                                    shape.add_constraint(
                                        ConstraintComponentId::new("sh:MinCountConstraintComponent"),
                                        constraint
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
            format!(r#"
                SELECT DISTINCT ?maxCount WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#maxCount> ?maxCount .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?maxCount WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#maxCount> ?maxCount .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &max_count_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(max_count_term) = binding.get("maxCount") {
                            if let Term::Literal(literal) = max_count_term {
                                if let Ok(max_count) = literal.as_str().parse::<u32>() {
                                    let constraint = Constraint::MaxCount(MaxCountConstraint { max_count });
                                    shape.add_constraint(
                                        ConstraintComponentId::new("sh:MaxCountConstraintComponent"),
                                        constraint
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
    fn parse_string_constraints(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        // Parse sh:minLength
        let min_length_query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?minLength WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#minLength> ?minLength .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?minLength WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#minLength> ?minLength .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &min_length_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(min_length_term) = binding.get("minLength") {
                            if let Term::Literal(literal) = min_length_term {
                                if let Ok(min_length) = literal.as_str().parse::<u32>() {
                                    let constraint = Constraint::MinLength(MinLengthConstraint { min_length });
                                    shape.add_constraint(
                                        ConstraintComponentId::new("sh:MinLengthConstraintComponent"),
                                        constraint
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
            format!(r#"
                SELECT DISTINCT ?maxLength WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#maxLength> ?maxLength .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?maxLength WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#maxLength> ?maxLength .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &max_length_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(max_length_term) = binding.get("maxLength") {
                            if let Term::Literal(literal) = max_length_term {
                                if let Ok(max_length) = literal.as_str().parse::<u32>() {
                                    let constraint = Constraint::MaxLength(MaxLengthConstraint { max_length });
                                    shape.add_constraint(
                                        ConstraintComponentId::new("sh:MaxLengthConstraintComponent"),
                                        constraint
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
    fn parse_value_constraints(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        // Parse sh:hasValue
        let has_value_query = if let Some(graph) = graph_name {
            format!(r#"
                SELECT DISTINCT ?value WHERE {{
                    GRAPH <{}> {{
                        <{}> <http://www.w3.org/ns/shacl#hasValue> ?value .
                    }}
                }}
            "#, graph, shape_iri)
        } else {
            format!(r#"
                SELECT DISTINCT ?value WHERE {{
                    <{}> <http://www.w3.org/ns/shacl#hasValue> ?value .
                }}
            "#, shape_iri)
        };
        
        match self.execute_shape_query(store, &has_value_query) {
            Ok(results) => {
                if let oxirs_core::query::QueryResult::Select { variables: _, bindings } = results {
                    for binding in bindings {
                        if let Some(value_term) = binding.get("value") {
                            let constraint = Constraint::HasValue(HasValueConstraint { 
                                value: value_term.clone() 
                            });
                            shape.add_constraint(
                                ConstraintComponentId::new("sh:HasValueConstraintComponent"),
                                constraint
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
    fn parse_range_constraints(&self, store: &Store, shape_iri: &str, shape: &mut Shape, graph_name: Option<&str>) -> Result<()> {
        // For now, just log that range constraints would be parsed here
        // TODO: Implement full range constraint parsing
        tracing::debug!("Range constraint parsing not yet fully implemented for shape {}", shape_iri);
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
            total_constraints: self.shape_cache.values()
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
    pub fn record_shape_parsed(&mut self, constraints_count: usize, targets_count: usize, cache_hit: bool) {
        self.total_shapes_parsed += 1;
        self.total_constraints_parsed += constraints_count;
        self.total_targets_parsed += targets_count;
        
        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        
        self.avg_constraints_per_shape = self.total_constraints_parsed as f64 / self.total_shapes_parsed as f64;
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
            Constraint::Class(ClassConstraint { class_iri })
        );
        shape
    }
    
    /// Create a property shape with basic constraints
    pub fn property_shape_with_constraints(
        shape_id: ShapeId, 
        path: PropertyPath, 
        constraints: Vec<(ConstraintComponentId, Constraint)>
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
        pattern: Option<String>
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);
        
        // Add datatype constraint for string
        let xsd_string = NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap();
        shape.add_constraint(
            ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
            Constraint::Datatype(DatatypeConstraint { datatype_iri: xsd_string })
        );
        
        // Add length constraints
        if let Some(min) = min_length {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MinLengthConstraintComponent"),
                Constraint::MinLength(MinLengthConstraint { min_length: min })
            );
        }
        
        if let Some(max) = max_length {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MaxLengthConstraintComponent"),
                Constraint::MaxLength(MaxLengthConstraint { max_length: max })
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
                })
            );
        }
        
        shape
    }
    
    /// Create a cardinality-constrained property shape
    pub fn cardinality_property_shape(
        shape_id: ShapeId,
        path: PropertyPath,
        min_count: Option<u32>,
        max_count: Option<u32>
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);
        
        if let Some(min) = min_count {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MinCountConstraintComponent"),
                Constraint::MinCount(MinCountConstraint { min_count: min })
            );
        }
        
        if let Some(max) = max_count {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MaxCountConstraintComponent"),
                Constraint::MaxCount(MaxCountConstraint { max_count: max })
            );
        }
        
        shape
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
            Some("^[A-Za-z ]+$".to_string())
        );
        
        assert_eq!(shape.id, shape_id);
        assert!(shape.is_property_shape());
        assert_eq!(shape.constraints.len(), 4); // datatype + minLength + maxLength + pattern
    }
    
    #[test]
    fn test_shape_factory_cardinality_property() {
        let shape_id = ShapeId::new("http://example.org/AgeShape");
        let age_path = PropertyPath::predicate(NamedNode::new("http://example.org/age").unwrap());
        
        let shape = ShapeFactory::cardinality_property_shape(
            shape_id.clone(),
            age_path,
            Some(1),
            Some(1)
        );
        
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
        stats.record_shape_parsed(5, 1, true);  // cache hit
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