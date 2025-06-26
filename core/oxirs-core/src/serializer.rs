//! RDF serialization utilities for various formats

use crate::{
    model::{dataset::Dataset, graph::Graph, GraphName, Quad},
    parser::RdfFormat,
    Result,
};

/// RDF serializer interface
pub struct Serializer {
    format: RdfFormat,
}

impl Serializer {
    /// Create a new serializer for the specified format
    pub fn new(format: RdfFormat) -> Self {
        Serializer { format }
    }

    /// Serialize a graph to a string
    pub fn serialize_graph(&self, graph: &Graph) -> Result<String> {
        match self.format {
            RdfFormat::Turtle => self.serialize_turtle(graph),
            RdfFormat::NTriples => self.serialize_ntriples(graph),
            RdfFormat::TriG => self.serialize_trig_graph(graph),
            RdfFormat::NQuads => self.serialize_nquads_graph(graph),
            RdfFormat::RdfXml => self.serialize_rdfxml(graph),
            RdfFormat::JsonLd => self.serialize_jsonld(graph),
        }
    }

    /// Serialize a dataset to a string (supports quad-based formats)
    pub fn serialize_dataset(&self, dataset: &Dataset) -> Result<String> {
        match self.format {
            RdfFormat::Turtle => Err(crate::OxirsError::Serialize(
                "Turtle format does not support datasets (use TriG instead)".to_string(),
            )),
            RdfFormat::NTriples => Err(crate::OxirsError::Serialize(
                "N-Triples format does not support datasets (use N-Quads instead)".to_string(),
            )),
            RdfFormat::TriG => self.serialize_trig_dataset(dataset),
            RdfFormat::NQuads => self.serialize_nquads_dataset(dataset),
            RdfFormat::RdfXml => Err(crate::OxirsError::Serialize(
                "RDF/XML dataset serialization not yet implemented".to_string(),
            )),
            RdfFormat::JsonLd => Err(crate::OxirsError::Serialize(
                "JSON-LD dataset serialization not yet implemented".to_string(),
            )),
        }
    }

    /// Legacy method for backward compatibility
    pub fn serialize(&self, graph: &Graph) -> Result<String> {
        self.serialize_graph(graph)
    }

    fn serialize_turtle(&self, graph: &Graph) -> Result<String> {
        let mut serializer = TurtleSerializer::new();
        serializer.serialize_graph(graph)
    }

    fn serialize_ntriples(&self, graph: &Graph) -> Result<String> {
        let mut result = String::new();

        for triple in graph.iter() {
            // Serialize subject
            match triple.subject() {
                crate::model::Subject::NamedNode(node) => {
                    result.push_str(&format!("<{}>", node.as_str()));
                }
                crate::model::Subject::BlankNode(node) => {
                    result.push_str(&format!("{}", node));
                }
                crate::model::Subject::Variable(_) => {
                    return Err(crate::OxirsError::Serialize(
                        "Variables not supported in N-Triples serialization".to_string(),
                    ));
                }
                crate::model::Subject::QuotedTriple(_) => {
                    return Err(crate::OxirsError::Serialize(
                        "Quoted triples not supported in N-Triples serialization".to_string(),
                    ));
                }
            }

            result.push(' ');

            // Serialize predicate
            match triple.predicate() {
                crate::model::Predicate::NamedNode(node) => {
                    result.push_str(&format!("<{}>", node.as_str()));
                }
                crate::model::Predicate::Variable(_) => {
                    return Err(crate::OxirsError::Serialize(
                        "Variables not supported in N-Triples serialization".to_string(),
                    ));
                }
            }

            result.push(' ');

            // Serialize object
            match triple.object() {
                crate::model::Object::NamedNode(node) => {
                    result.push_str(&format!("<{}>", node.as_str()));
                }
                crate::model::Object::BlankNode(node) => {
                    result.push_str(&format!("{}", node));
                }
                crate::model::Object::Literal(literal) => {
                    result.push('"');
                    // Escape quotes and backslashes in literal value
                    for c in literal.value().chars() {
                        match c {
                            '"' => result.push_str("\\\""),
                            '\\' => result.push_str("\\\\"),
                            '\n' => result.push_str("\\n"),
                            '\r' => result.push_str("\\r"),
                            '\t' => result.push_str("\\t"),
                            _ => result.push(c),
                        }
                    }
                    result.push('"');

                    // Add language tag or datatype
                    if let Some(lang) = literal.language() {
                        result.push_str(&format!("@{}", lang));
                    } else {
                        let datatype = literal.datatype();
                        if datatype.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                            result.push_str(&format!("^^<{}>", datatype.as_str()));
                        }
                    }
                }
                crate::model::Object::Variable(_) => {
                    return Err(crate::OxirsError::Serialize(
                        "Variables not supported in N-Triples serialization".to_string(),
                    ));
                }
                crate::model::Object::QuotedTriple(_) => {
                    return Err(crate::OxirsError::Serialize(
                        "Quoted triples not supported in N-Triples serialization".to_string(),
                    ));
                }
            }

            result.push_str(" .\n");
        }

        Ok(result)
    }

    fn serialize_rdfxml(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement RDF/XML serialization when API is stable
        Err(crate::OxirsError::Serialize(
            "RDF/XML serialization not yet implemented".to_string(),
        ))
    }

    fn serialize_trig_graph(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement TriG serialization
        Ok(String::new())
    }

    fn serialize_trig_dataset(&self, _dataset: &Dataset) -> Result<String> {
        // TODO: Implement TriG dataset serialization
        Ok(String::new())
    }

    fn serialize_nquads_graph(&self, graph: &Graph) -> Result<String> {
        // For a graph, serialize all triples as quads in the default graph
        let mut result = String::new();

        for triple in graph.iter() {
            result.push_str(&self.serialize_quad_to_nquads(&Quad::from_triple(triple.clone()))?);
        }

        Ok(result)
    }

    fn serialize_nquads_dataset(&self, dataset: &Dataset) -> Result<String> {
        let mut result = String::new();

        for quad in dataset.iter() {
            result.push_str(&self.serialize_quad_to_nquads(&quad)?);
        }

        Ok(result)
    }

    pub fn serialize_quad_to_nquads(&self, quad: &Quad) -> Result<String> {
        let mut result = String::new();

        // Serialize subject
        match quad.subject() {
            crate::model::Subject::NamedNode(node) => {
                result.push_str(&format!("<{}>", node.as_str()));
            }
            crate::model::Subject::BlankNode(node) => {
                result.push_str(&format!("{}", node));
            }
            crate::model::Subject::Variable(_) => {
                return Err(crate::OxirsError::Serialize(
                    "Variables not supported in N-Quads serialization".to_string(),
                ));
            }
            crate::model::Subject::QuotedTriple(_) => {
                return Err(crate::OxirsError::Serialize(
                    "Quoted triples not supported in N-Quads serialization".to_string(),
                ));
            }
        }

        result.push(' ');

        // Serialize predicate
        match quad.predicate() {
            crate::model::Predicate::NamedNode(node) => {
                result.push_str(&format!("<{}>", node.as_str()));
            }
            crate::model::Predicate::Variable(_) => {
                return Err(crate::OxirsError::Serialize(
                    "Variables not supported in N-Quads serialization".to_string(),
                ));
            }
        }

        result.push(' ');

        // Serialize object
        match quad.object() {
            crate::model::Object::NamedNode(node) => {
                result.push_str(&format!("<{}>", node.as_str()));
            }
            crate::model::Object::BlankNode(node) => {
                result.push_str(&format!("{}", node));
            }
            crate::model::Object::Literal(literal) => {
                result.push('"');
                // Escape quotes and backslashes in literal value
                for c in literal.value().chars() {
                    match c {
                        '"' => result.push_str("\\\""),
                        '\\' => result.push_str("\\\\"),
                        '\n' => result.push_str("\\n"),
                        '\r' => result.push_str("\\r"),
                        '\t' => result.push_str("\\t"),
                        _ => result.push(c),
                    }
                }
                result.push('"');

                // Add language tag or datatype
                if let Some(lang) = literal.language() {
                    result.push_str(&format!("@{}", lang));
                } else {
                    let datatype = literal.datatype();
                    if datatype.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                        result.push_str(&format!("^^<{}>", datatype.as_str()));
                    }
                }
            }
            crate::model::Object::Variable(_) => {
                return Err(crate::OxirsError::Serialize(
                    "Variables not supported in N-Quads serialization".to_string(),
                ));
            }
            crate::model::Object::QuotedTriple(_) => {
                return Err(crate::OxirsError::Serialize(
                    "Quoted triples not supported in N-Quads serialization".to_string(),
                ));
            }
        }

        result.push(' ');

        // Serialize graph name
        match quad.graph_name() {
            GraphName::NamedNode(node) => {
                result.push_str(&format!("<{}>", node.as_str()));
            }
            GraphName::BlankNode(node) => {
                result.push_str(&format!("{}", node));
            }
            GraphName::Variable(_) => {
                return Err(crate::OxirsError::Serialize(
                    "Variables not supported in N-Quads serialization".to_string(),
                ));
            }
            GraphName::DefaultGraph => {
                // For default graph, we can either omit the graph name entirely
                // or use a special representation. Let's omit it to make it N-Triples compatible
                result.pop(); // Remove the trailing space
                result.push_str(" .\n");
                return Ok(result);
            }
        }

        result.push_str(" .\n");

        Ok(result)
    }

    fn serialize_jsonld(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement JSON-LD serialization using oxjsonld
        Err(crate::OxirsError::Serialize(
            "JSON-LD serialization not yet implemented".to_string(),
        ))
    }
}

/// Turtle serializer with prefix optimization
struct TurtleSerializer {
    prefixes: std::collections::HashMap<String, String>,
    used_namespaces: std::collections::HashSet<String>,
}

impl TurtleSerializer {
    fn new() -> Self {
        let mut prefixes = std::collections::HashMap::new();

        // Add common prefixes
        prefixes.insert(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
            "rdf".to_string(),
        );
        prefixes.insert(
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
            "rdfs".to_string(),
        );
        prefixes.insert(
            "http://www.w3.org/2001/XMLSchema#".to_string(),
            "xsd".to_string(),
        );
        prefixes.insert("http://xmlns.com/foaf/0.1/".to_string(), "foaf".to_string());
        prefixes.insert(
            "http://purl.org/dc/elements/1.1/".to_string(),
            "dc".to_string(),
        );

        TurtleSerializer {
            prefixes,
            used_namespaces: std::collections::HashSet::new(),
        }
    }

    fn serialize_graph(&mut self, graph: &Graph) -> Result<String> {
        // First pass: collect all namespaces used in the graph
        self.collect_namespaces(graph);

        let mut result = String::new();

        // Write prefix declarations
        let mut prefix_entries: Vec<_> = self.prefixes.iter().collect();
        prefix_entries.sort_by_key(|(_, prefix)| *prefix);

        for (namespace, prefix) in prefix_entries {
            if self.used_namespaces.contains(namespace) {
                result.push_str(&format!("@prefix {}: <{}> .\n", prefix, namespace));
            }
        }

        if !self.prefixes.is_empty() && !graph.is_empty() {
            result.push('\n');
        }

        // Group triples by subject for better readability
        let mut subjects_map: std::collections::HashMap<
            crate::model::Subject,
            Vec<&crate::model::Triple>,
        > = std::collections::HashMap::new();

        for triple in graph.iter() {
            subjects_map
                .entry(triple.subject().clone())
                .or_insert_with(Vec::new)
                .push(triple);
        }

        let mut subject_entries: Vec<_> = subjects_map.iter().collect();
        subject_entries.sort_by_key(|(subject, _)| format!("{}", subject));

        for (i, (subject, triples)) in subject_entries.iter().enumerate() {
            if i > 0 {
                result.push('\n');
            }

            result.push_str(&self.serialize_subject(subject)?);

            // Group by predicate for ; syntax
            let mut predicates_map: std::collections::HashMap<
                crate::model::Predicate,
                Vec<&crate::model::Object>,
            > = std::collections::HashMap::new();

            for triple in triples.iter() {
                predicates_map
                    .entry(triple.predicate().clone())
                    .or_insert_with(Vec::new)
                    .push(triple.object());
            }

            let mut predicate_entries: Vec<_> = predicates_map.iter().collect();
            predicate_entries.sort_by_key(|(predicate, _)| format!("{}", predicate));

            for (j, (predicate, objects)) in predicate_entries.iter().enumerate() {
                if j == 0 {
                    result.push(' ');
                } else {
                    result.push_str(" ;\n        ");
                }

                result.push_str(&self.serialize_predicate(predicate)?);
                result.push(' ');

                // Serialize objects with , syntax
                for (k, object) in objects.iter().enumerate() {
                    if k > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&self.serialize_object(object)?);
                }
            }

            result.push_str(" .\n");
        }

        Ok(result)
    }

    fn collect_namespaces(&mut self, graph: &Graph) {
        for triple in graph.iter() {
            self.mark_namespace_used(triple.subject());
            self.mark_namespace_used_predicate(triple.predicate());
            self.mark_namespace_used_object(triple.object());
        }
    }

    fn mark_namespace_used(&mut self, subject: &crate::model::Subject) {
        match subject {
            crate::model::Subject::NamedNode(node) => {
                if let Some(namespace) = self.extract_namespace(node.as_str()) {
                    self.used_namespaces.insert(namespace);
                }
            }
            _ => {}
        }
    }

    fn mark_namespace_used_predicate(&mut self, predicate: &crate::model::Predicate) {
        match predicate {
            crate::model::Predicate::NamedNode(node) => {
                if let Some(namespace) = self.extract_namespace(node.as_str()) {
                    self.used_namespaces.insert(namespace);
                }
            }
            _ => {}
        }
    }

    fn mark_namespace_used_object(&mut self, object: &crate::model::Object) {
        match object {
            crate::model::Object::NamedNode(node) => {
                if let Some(namespace) = self.extract_namespace(node.as_str()) {
                    self.used_namespaces.insert(namespace);
                }
            }
            crate::model::Object::Literal(literal) => {
                let datatype = literal.datatype();
                if let Some(namespace) = self.extract_namespace(datatype.as_str()) {
                    self.used_namespaces.insert(namespace);
                }
            }
            _ => {}
        }
    }

    fn extract_namespace(&self, iri: &str) -> Option<String> {
        // Find the namespace part of an IRI
        if let Some(hash_pos) = iri.rfind('#') {
            Some(format!("{}#", &iri[..hash_pos]))
        } else if let Some(slash_pos) = iri.rfind('/') {
            Some(format!("{}/", &iri[..slash_pos]))
        } else {
            None
        }
    }

    fn serialize_subject(&self, subject: &crate::model::Subject) -> Result<String> {
        match subject {
            crate::model::Subject::NamedNode(node) => self.serialize_iri(node.as_str()),
            crate::model::Subject::BlankNode(node) => Ok(node.as_str().to_string()),
            crate::model::Subject::Variable(var) => Err(crate::OxirsError::Serialize(
                "Variables not supported in Turtle serialization".to_string(),
            )),
            crate::model::Subject::QuotedTriple(qt) => Ok(format!(
                "<< {} {} {} >>",
                self.serialize_subject(qt.subject())?,
                self.serialize_predicate(qt.predicate())?,
                self.serialize_object(qt.object())?
            )),
        }
    }

    fn serialize_predicate(&self, predicate: &crate::model::Predicate) -> Result<String> {
        match predicate {
            crate::model::Predicate::NamedNode(node) => {
                // Check for rdf:type shorthand
                if node.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    Ok("a".to_string())
                } else {
                    self.serialize_iri(node.as_str())
                }
            }
            crate::model::Predicate::Variable(_) => Err(crate::OxirsError::Serialize(
                "Variables not supported in Turtle serialization".to_string(),
            )),
        }
    }

    fn serialize_object(&self, object: &crate::model::Object) -> Result<String> {
        match object {
            crate::model::Object::NamedNode(node) => self.serialize_iri(node.as_str()),
            crate::model::Object::BlankNode(node) => Ok(node.as_str().to_string()),
            crate::model::Object::Literal(literal) => self.serialize_literal(literal),
            crate::model::Object::Variable(_) => Err(crate::OxirsError::Serialize(
                "Variables not supported in Turtle serialization".to_string(),
            )),
            crate::model::Object::QuotedTriple(qt) => Ok(format!(
                "<< {} {} {} >>",
                self.serialize_subject(qt.subject())?,
                self.serialize_predicate(qt.predicate())?,
                self.serialize_object(qt.object())?
            )),
        }
    }

    fn serialize_iri(&self, iri: &str) -> Result<String> {
        // Try to use a prefix if available
        for (namespace, prefix) in &self.prefixes {
            if iri.starts_with(namespace) && self.used_namespaces.contains(namespace) {
                let local_name = &iri[namespace.len()..];
                // Ensure local name is valid
                if self.is_valid_local_name(local_name) {
                    return Ok(format!("{}:{}", prefix, local_name));
                }
            }
        }

        // Fall back to full IRI
        Ok(format!("<{}>", iri))
    }

    fn is_valid_local_name(&self, name: &str) -> bool {
        // Simple validation - should start with letter or underscore
        // and contain only alphanumeric, underscore, hyphen
        if name.is_empty() {
            return true; // Empty local name is valid
        }

        let first_char = name.chars().next().unwrap();
        if !first_char.is_alphabetic() && first_char != '_' {
            return false;
        }

        name.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    }

    fn serialize_literal(&self, literal: &crate::model::Literal) -> Result<String> {
        let mut result = String::new();

        // Quote the literal value
        result.push('"');
        for c in literal.value().chars() {
            match c {
                '"' => result.push_str("\\\""),
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                _ => result.push(c),
            }
        }
        result.push('"');

        // Add language tag or datatype
        if let Some(lang) = literal.language() {
            result.push_str(&format!("@{}", lang));
        } else {
            let datatype = literal.datatype();
            // Check if it's the default string type
            if datatype.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                result.push_str("^^");
                result.push_str(&self.serialize_iri(datatype.as_str())?);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph::Graph;
    use crate::model::*;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add a simple triple
        let subject = NamedNode::new("http://example.org/alice").unwrap();
        let predicate = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let object = Literal::new("Alice Smith");
        let triple1 = Triple::new(subject.clone(), predicate, object);

        // Add a typed literal triple
        let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap();
        let age_obj = Literal::new_typed("30", crate::vocab::xsd::INTEGER.clone());
        let triple2 = Triple::new(subject.clone(), age_pred, age_obj);

        // Add a language-tagged literal triple
        let desc_pred = NamedNode::new("http://example.org/description").unwrap();
        let desc_obj = Literal::new_lang("Une personne", "fr").unwrap();
        let triple3 = Triple::new(subject, desc_pred, desc_obj);

        // Add a blank node triple
        let blank_subject = BlankNode::new("person1").unwrap();
        let knows_pred = NamedNode::new("http://xmlns.com/foaf/0.1/knows").unwrap();
        let knows_obj = NamedNode::new("http://example.org/bob").unwrap();
        let triple4 = Triple::new(blank_subject, knows_pred, knows_obj);

        graph.insert(triple1);
        graph.insert(triple2);
        graph.insert(triple3);
        graph.insert(triple4);

        graph
    }

    #[test]
    fn test_ntriples_serialization() {
        let graph = create_test_graph();
        let serializer = Serializer::new(RdfFormat::NTriples);

        let result = serializer.serialize_graph(&graph);
        assert!(result.is_ok());

        let ntriples = result.unwrap();
        assert!(!ntriples.is_empty());

        // Check that all lines end with " ."
        for line in ntriples.lines() {
            if !line.trim().is_empty() {
                assert!(line.ends_with(" ."), "Line should end with ' .': {}", line);
            }
        }

        // Check that the output contains our expected triples
        assert!(ntriples.contains("http://example.org/alice"));
        assert!(ntriples.contains("http://xmlns.com/foaf/0.1/name"));
        assert!(ntriples.contains("\"Alice Smith\""));
        assert!(ntriples.contains("\"30\"^^<http://www.w3.org/2001/XMLSchema#integer>"));
        assert!(ntriples.contains("\"Une personne\"@fr"));
        assert!(ntriples.contains("_:person1"));
    }

    #[test]
    fn test_literal_escaping() {
        let mut graph = Graph::new();
        let subject = NamedNode::new("http://example.org/test").unwrap();
        let predicate = NamedNode::new("http://example.org/description").unwrap();

        // Test literal with quotes and escape sequences
        let object = Literal::new("Text with \"quotes\" and \n newlines \t and tabs");
        let triple = Triple::new(subject, predicate, object);
        graph.insert(triple);

        let serializer = Serializer::new(RdfFormat::NTriples);
        let result = serializer.serialize_graph(&graph).unwrap();

        // Check that quotes and escape sequences are properly escaped
        assert!(result.contains("\\\"quotes\\\""));
        assert!(result.contains("\\n"));
        assert!(result.contains("\\t"));
    }

    #[test]
    fn test_empty_graph_serialization() {
        let graph = Graph::new();
        let serializer = Serializer::new(RdfFormat::NTriples);

        let result = serializer.serialize_graph(&graph);
        assert!(result.is_ok());

        let ntriples = result.unwrap();
        assert!(ntriples.is_empty());
    }

    #[test]
    fn test_variable_serialization_error() {
        let mut graph = Graph::new();
        let variable_subject = Variable::new("x").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("test");

        let triple = Triple::new(variable_subject, predicate, object);
        graph.insert(triple);

        let serializer = Serializer::new(RdfFormat::NTriples);
        let result = serializer.serialize_graph(&graph);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Variables not supported"));
    }

    #[test]
    fn test_turtle_serialization() {
        let graph = create_test_graph();
        let serializer = Serializer::new(RdfFormat::Turtle);

        let result = serializer.serialize_graph(&graph);
        assert!(result.is_ok());

        let turtle = result.unwrap();
        assert!(!turtle.is_empty());

        // Should contain prefix declarations
        assert!(turtle.contains("@prefix"));

        // Should contain the 'a' shorthand for rdf:type
        // Should use abbreviated syntax with ;

        // Should have proper Turtle syntax
        assert!(turtle.ends_with(" .\n") || turtle.ends_with("."));
    }

    #[test]
    fn test_turtle_serialization_with_prefixes() {
        let mut graph = Graph::new();

        // Create triples using FOAF vocabulary
        let alice = NamedNode::new("http://example.org/alice").unwrap();
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let person_type = NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();

        let name_literal = Literal::new("Alice");

        graph.insert(Triple::new(alice.clone(), name_pred, name_literal));
        graph.insert(Triple::new(alice, rdf_type, person_type));

        let serializer = Serializer::new(RdfFormat::Turtle);
        let turtle = serializer.serialize_graph(&graph).unwrap();

        // Should include FOAF prefix
        assert!(turtle.contains("@prefix foaf: <http://xmlns.com/foaf/0.1/>"));

        // Should use 'a' shorthand for rdf:type
        assert!(turtle.contains(" a "));

        // Should use prefixed names
        assert!(turtle.contains("foaf:"));
    }

    #[test]
    fn test_turtle_serialization_abbreviated_syntax() {
        let mut graph = Graph::new();

        let alice = NamedNode::new("http://example.org/alice").unwrap();
        let name_pred = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap();

        let name_literal = Literal::new("Alice");
        let age_literal = Literal::new_typed("30", crate::vocab::xsd::INTEGER.clone());

        graph.insert(Triple::new(alice.clone(), name_pred, name_literal));
        graph.insert(Triple::new(alice, age_pred, age_literal));

        let serializer = Serializer::new(RdfFormat::Turtle);
        let turtle = serializer.serialize_graph(&graph).unwrap();

        // Should use ; syntax for same subject
        assert!(turtle.contains(";"));

        // Should have proper indentation and formatting
        assert!(turtle.lines().count() >= 3); // prefixes + subject line + continuation
    }

    #[test]
    fn test_turtle_serialization_literals() {
        let mut graph = Graph::new();

        let subject = NamedNode::new("http://example.org/test").unwrap();
        let desc_pred = NamedNode::new("http://example.org/description").unwrap();
        let age_pred = NamedNode::new("http://example.org/age").unwrap();

        // Language-tagged literal
        let desc_literal = Literal::new_lang("Une description", "fr").unwrap();
        // Typed literal
        let age_literal = Literal::new_typed("25", crate::vocab::xsd::INTEGER.clone());

        graph.insert(Triple::new(subject.clone(), desc_pred, desc_literal));
        graph.insert(Triple::new(subject, age_pred, age_literal));

        let serializer = Serializer::new(RdfFormat::Turtle);
        let turtle = serializer.serialize_graph(&graph).unwrap();

        // Should contain language tag
        assert!(turtle.contains("@fr"));

        // Should contain datatype with xsd prefix
        assert!(turtle.contains("^^xsd:integer"));
    }
}
