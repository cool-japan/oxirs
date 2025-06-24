//! RDF serialization utilities for various formats

use crate::{Result, model::graph::Graph, parser::RdfFormat};

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
            RdfFormat::TriG => self.serialize_trig(graph),
            RdfFormat::NQuads => self.serialize_nquads(graph),
            RdfFormat::RdfXml => self.serialize_rdfxml(graph),
            RdfFormat::JsonLd => self.serialize_jsonld(graph),
        }
    }

    /// Legacy method for backward compatibility
    pub fn serialize(&self, graph: &Graph) -> Result<String> {
        self.serialize_graph(graph)
    }

    fn serialize_turtle(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement Turtle serialization
        Ok(String::new())
    }

    fn serialize_ntriples(&self, graph: &Graph) -> Result<String> {
        let mut result = String::new();
        
        for triple in graph.iter() {
            // Serialize subject
            match triple.subject() {
                crate::model::Subject::NamedNode(node) => {
                    result.push_str(&format!("<{}>", node.as_str()));
                },
                crate::model::Subject::BlankNode(node) => {
                    result.push_str(node.as_str());
                },
                crate::model::Subject::Variable(_) => {
                    return Err(crate::OxirsError::Serialization(
                        "Variables not supported in N-Triples serialization".to_string()
                    ));
                }
            }
            
            result.push(' ');
            
            // Serialize predicate
            match triple.predicate() {
                crate::model::Predicate::NamedNode(node) => {
                    result.push_str(&format!("<{}>", node.as_str()));
                },
                crate::model::Predicate::Variable(_) => {
                    return Err(crate::OxirsError::Serialization(
                        "Variables not supported in N-Triples serialization".to_string()
                    ));
                }
            }
            
            result.push(' ');
            
            // Serialize object
            match triple.object() {
                crate::model::Object::NamedNode(node) => {
                    result.push_str(&format!("<{}>", node.as_str()));
                },
                crate::model::Object::BlankNode(node) => {
                    result.push_str(node.as_str());
                },
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
                    } else if let Some(datatype) = literal.datatype() {
                        result.push_str(&format!("^^<{}>", datatype.as_str()));
                    }
                },
                crate::model::Object::Variable(_) => {
                    return Err(crate::OxirsError::Serialization(
                        "Variables not supported in N-Triples serialization".to_string()
                    ));
                }
            }
            
            result.push_str(" .\n");
        }
        
        Ok(result)
    }

    fn serialize_rdfxml(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement RDF/XML serialization
        Ok(String::new())
    }

    fn serialize_trig(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement TriG serialization
        Ok(String::new())
    }

    fn serialize_nquads(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement N-Quads serialization
        Ok(String::new())
    }

    fn serialize_jsonld(&self, _graph: &Graph) -> Result<String> {
        // TODO: Implement JSON-LD serialization
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;
    use crate::model::graph::Graph;
    
    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();
        
        // Add a simple triple
        let subject = NamedNode::new("http://example.org/alice").unwrap();
        let predicate = NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap();
        let object = Literal::new("Alice Smith");
        let triple1 = Triple::new(subject.clone(), predicate, object);
        
        // Add a typed literal triple
        let age_pred = NamedNode::new("http://xmlns.com/foaf/0.1/age").unwrap();
        let age_obj = Literal::new_typed("30", crate::model::literal::xsd::integer());
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
        assert!(result.unwrap_err().to_string().contains("Variables not supported"));
    }
}