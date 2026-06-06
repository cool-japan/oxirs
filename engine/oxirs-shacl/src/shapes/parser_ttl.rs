#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use oxirs_core::{
    graph::Graph,
    model::{BlankNode, NamedNode, Object, Predicate, Subject, Term},
    Store,
};

use crate::{
    paths::PropertyPath, targets::Target, Result, ShaclError, Shape, ShapeId, ShapeType,
    SHACL_VOCAB,
};

use super::parser_shaclc::{
    determine_shape_type, find_shape_iris, get_boolean_object, get_integer_object,
    get_named_node_object, get_string_object, get_string_with_language,
    parse_constraints_from_blank_node, parse_constraints_from_graph,
    parse_metadata_from_blank_node, parse_property_path_object, parse_severity,
};
use super::types::{ShapeParsingConfig, ShapeParsingStats};

/// SHACL shape parser for extracting shapes from RDF data
#[derive(Debug)]
pub struct ShapeParser {
    shape_cache: HashMap<String, Shape>,
    strict_mode: bool,
    max_depth: usize,
    stats: ShapeParsingStats,
    inline_property_shapes: Vec<Shape>,
    blank_node_counter: usize,
}

impl ShapeParser {
    pub fn new() -> Self {
        Self {
            shape_cache: HashMap::new(),
            strict_mode: false,
            max_depth: 50,
            stats: ShapeParsingStats::new(),
            inline_property_shapes: Vec::new(),
            blank_node_counter: 0,
        }
    }

    pub fn new_strict() -> Self {
        let mut parser = Self::new();
        parser.strict_mode = true;
        parser
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    pub fn with_config(mut self, config: ShapeParsingConfig) -> Self {
        self.max_depth = config.max_depth;
        self.strict_mode = config.strict_mode;
        self
    }

    pub fn parse_shapes_from_store(
        &mut self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<Shape>> {
        let start_time = std::time::Instant::now();
        let mut shapes = Vec::new();

        let shape_nodes = self.find_shape_nodes(store, graph_name)?;

        for shape_node in shape_nodes {
            match self.parse_shape_from_store(store, &shape_node, graph_name) {
                Ok(shape) => {
                    self.stats.update_shape_parsed(
                        shape.constraints.len(),
                        std::time::Duration::from_millis(1),
                    );
                    shapes.push(shape);
                }
                Err(e) => {
                    if self.strict_mode {
                        return Err(e);
                    } else {
                        tracing::warn!("Failed to parse shape {}: {}", shape_node, e);
                    }
                }
            }
        }

        let elapsed = start_time.elapsed();
        self.stats.parsing_time += elapsed;

        tracing::info!(
            "Parsed {} shapes in {:?} (total shapes parsed: {})",
            shapes.len(),
            elapsed,
            self.stats.total_shapes_parsed
        );

        Ok(shapes)
    }

    pub fn parse_shapes_from_rdf(
        &mut self,
        rdf_data: &str,
        format: &str,
        _base_iri: Option<&str>,
    ) -> Result<Vec<Shape>> {
        if format.to_lowercase() != "turtle" && format.to_lowercase() != "ttl" {
            return Err(ShaclError::ShapeParsing(format!(
                "RDF parsing currently only supports Turtle format, got: {format}"
            )));
        }

        use oxirs_core::format::TurtleParser;

        let parser = TurtleParser::new();
        tracing::debug!("Parsing Turtle data:\n{}", rdf_data);
        let triples = parser.parse_slice(rdf_data.as_bytes()).map_err(|e| {
            tracing::error!("Turtle parsing failed: {}", e);
            ShaclError::ShapeParsing(format!("Failed to parse Turtle data: {e}"))
        })?;
        tracing::debug!("Parsed {} triples", triples.len());

        let mut graph = Graph::new();
        for triple in triples {
            graph.insert(triple);
        }

        self.parse_shapes_from_graph(&graph)
    }

    pub fn parse_shapes_from_graph(&mut self, graph: &Graph) -> Result<Vec<Shape>> {
        self.inline_property_shapes.clear();

        let mut shapes = Vec::new();
        let mut visited = HashSet::new();

        let shape_iris = find_shape_iris(graph)?;

        for shape_iri in shape_iris {
            if !visited.contains(&shape_iri) {
                let shape = self.parse_shape(graph, &shape_iri, &mut visited, 0)?;
                shapes.push(shape);
            }
        }

        shapes.append(&mut self.inline_property_shapes);

        Ok(shapes)
    }

    fn find_shape_nodes(&self, _store: &dyn Store, _graph_name: Option<&str>) -> Result<Vec<Term>> {
        tracing::warn!("Shape node discovery not yet implemented in refactored parser");
        Ok(Vec::new())
    }

    fn parse_shape(
        &mut self,
        graph: &Graph,
        shape_iri: &str,
        visited: &mut HashSet<String>,
        depth: usize,
    ) -> Result<Shape> {
        if depth > self.max_depth {
            return Err(ShaclError::ShapeParsing(format!(
                "Maximum parsing depth {} exceeded for shape {}",
                self.max_depth, shape_iri
            )));
        }

        visited.insert(shape_iri.to_string());

        if let Some(cached_shape) = self.shape_cache.get(shape_iri) {
            return Ok(cached_shape.clone());
        }

        let shape_node = NamedNode::new(shape_iri)
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid shape IRI {shape_iri}: {e}")))?;

        let shape_type = determine_shape_type(graph, &shape_node)?;

        let mut shape = Shape::new(ShapeId::new(shape_iri), shape_type);

        if shape.shape_type == ShapeType::PropertyShape {
            self.parse_property_path_from_graph(graph, &shape_node, &mut shape)?;
        }

        self.parse_targets_from_graph(graph, &shape_node, &mut shape)?;
        parse_constraints_from_graph(graph, &shape_node, &mut shape)?;
        self.parse_property_shapes_from_graph(graph, &shape_node, &mut shape)?;
        self.parse_shape_metadata_from_graph(graph, &shape_node, &mut shape)?;

        if self.shape_cache.len() < 1000 {
            self.shape_cache
                .insert(shape_iri.to_string(), shape.clone());
        }

        Ok(shape)
    }

    fn parse_shape_from_store(
        &mut self,
        _store: &dyn Store,
        _shape_node: &Term,
        _graph_name: Option<&str>,
    ) -> Result<Shape> {
        Err(ShaclError::ShapeParsing(
            "Shape parsing not yet implemented in refactored parser".to_string(),
        ))
    }

    pub fn stats(&self) -> &ShapeParsingStats {
        &self.stats
    }

    pub fn clear_cache(&mut self) {
        self.shape_cache.clear();
    }

    pub fn cache_size(&self) -> usize {
        self.shape_cache.len()
    }

    fn parse_property_path_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        let path_prop = NamedNode::new("http://www.w3.org/ns/shacl#path")
            .map_err(|e| ShaclError::ShapeParsing(format!("Invalid path property IRI: {e}")))?;

        let path_triples = graph.query_triples(
            Some(&Subject::NamedNode(shape_node.clone())),
            Some(&Predicate::NamedNode(path_prop)),
            None,
        );

        if let Some(triple) = path_triples.into_iter().next() {
            let path = parse_property_path_object(graph, triple.object())?;
            shape.path = Some(path);
        }

        Ok(())
    }

    fn parse_targets_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        let shape_subject = Subject::NamedNode(shape_node.clone());

        if let Some(target_class) =
            get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.target_class)?
        {
            shape.add_target(Target::Class(target_class));
        }

        let target_node_triples = graph.query_triples(
            Some(&shape_subject),
            Some(&Predicate::NamedNode(SHACL_VOCAB.target_node.clone())),
            None,
        );
        for triple in target_node_triples {
            let target_term = match triple.object() {
                Object::NamedNode(node) => Term::NamedNode(node.clone()),
                Object::BlankNode(node) => Term::BlankNode(node.clone()),
                Object::Literal(lit) => Term::Literal(lit.clone()),
                Object::Variable(_) | Object::QuotedTriple(_) => continue,
            };
            shape.add_target(Target::Node(target_term));
        }

        if let Some(target_objects_of) =
            get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.target_objects_of)?
        {
            shape.add_target(Target::ObjectsOf(target_objects_of));
        }

        if let Some(target_subjects_of) =
            get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.target_subjects_of)?
        {
            shape.add_target(Target::SubjectsOf(target_subjects_of));
        }

        Ok(())
    }

    fn parse_shape_metadata_from_graph(
        &self,
        graph: &Graph,
        shape_node: &NamedNode,
        shape: &mut Shape,
    ) -> Result<()> {
        let shape_subject = Subject::NamedNode(shape_node.clone());

        if let Some(label) = get_string_object(graph, &shape_subject, &SHACL_VOCAB.label)? {
            shape.label = Some(label);
        }

        if let Some(description) =
            get_string_object(graph, &shape_subject, &SHACL_VOCAB.description)?
        {
            shape.description = Some(description);
        }

        if let Some(order) = get_integer_object(graph, &shape_subject, &SHACL_VOCAB.order)? {
            shape.order = Some(order as i32);
        }

        if let Some(deactivated) =
            get_boolean_object(graph, &shape_subject, &SHACL_VOCAB.deactivated)?
        {
            shape.deactivated = deactivated;
        }

        if let Some((message, lang_tag)) =
            get_string_with_language(graph, &shape_subject, &SHACL_VOCAB.message)?
        {
            shape.messages.insert(lang_tag, message);
        }

        if let Some(severity_iri) =
            get_named_node_object(graph, &shape_subject, &SHACL_VOCAB.severity)?
        {
            shape.severity = parse_severity(&severity_iri)?;
        }

        Ok(())
    }

    fn parse_property_shapes_from_graph(
        &mut self,
        graph: &Graph,
        shape_node: &NamedNode,
        parent_shape: &mut Shape,
    ) -> Result<()> {
        let shape_subject = Subject::NamedNode(shape_node.clone());

        let property_triples = graph.query_triples(
            Some(&shape_subject),
            Some(&Predicate::NamedNode(SHACL_VOCAB.property.clone())),
            None,
        );

        for triple in property_triples {
            match triple.object() {
                Object::BlankNode(blank_node) => {
                    let generated_iri = format!(
                        "{}#_property_{}",
                        shape_node.as_str(),
                        self.blank_node_counter
                    );
                    self.blank_node_counter += 1;

                    let property_shape =
                        self.parse_blank_node_property_shape(graph, blank_node, &generated_iri)?;

                    parent_shape
                        .property_shapes
                        .push(ShapeId::new(&generated_iri));

                    self.inline_property_shapes.push(property_shape);
                }
                Object::NamedNode(property_shape_node) => {
                    parent_shape
                        .property_shapes
                        .push(ShapeId::new(property_shape_node.as_str()));
                    tracing::debug!(
                        "Found named property shape reference: {}",
                        property_shape_node.as_str()
                    );
                }
                _ => {
                    tracing::warn!("Unexpected object type for sh:property");
                }
            }
        }

        Ok(())
    }

    fn parse_blank_node_property_shape(
        &mut self,
        graph: &Graph,
        blank_node: &BlankNode,
        generated_iri: &str,
    ) -> Result<Shape> {
        let blank_subject = Subject::BlankNode(blank_node.clone());

        let mut shape = Shape::new(ShapeId::new(generated_iri), ShapeType::PropertyShape);

        if let Some(path) = self.parse_property_path_for_blank_node(graph, &blank_subject)? {
            shape.path = Some(path);
        } else {
            return Err(ShaclError::ShapeParsing(format!(
                "PropertyShape {} is missing sh:path",
                generated_iri
            )));
        }

        parse_constraints_from_blank_node(graph, &blank_subject, &mut shape)?;
        parse_metadata_from_blank_node(graph, &blank_subject, &mut shape)?;

        tracing::debug!(
            "Parsed inline PropertyShape {} with {} constraints",
            generated_iri,
            shape.constraints.len()
        );

        Ok(shape)
    }

    fn parse_property_path_for_blank_node(
        &self,
        graph: &Graph,
        blank_subject: &Subject,
    ) -> Result<Option<PropertyPath>> {
        let path_triples = graph.query_triples(
            Some(blank_subject),
            Some(&Predicate::NamedNode(SHACL_VOCAB.path.clone())),
            None,
        );

        for triple in path_triples {
            match triple.object() {
                Object::NamedNode(path_node) => {
                    return Ok(Some(PropertyPath::Predicate(path_node.clone())));
                }
                Object::BlankNode(path_blank) => {
                    let path =
                        parse_property_path_object(graph, &Object::BlankNode(path_blank.clone()))?;
                    return Ok(Some(path));
                }
                _ => {}
            }
        }

        Ok(None)
    }
}

impl Default for ShapeParser {
    fn default() -> Self {
        Self::new()
    }
}
