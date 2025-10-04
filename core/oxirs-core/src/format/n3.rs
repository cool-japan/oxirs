//! N3 (Notation3) format serializer and parser
//!
//! N3 is a superset of Turtle that adds support for variables, rules, and formulae.
//! This implementation focuses on the Turtle-compatible subset for now.
//!
//! W3C Specification: https://w3c.github.io/N3/spec/

use super::error::FormatError;
use crate::model::{
    GraphName, Literal, NamedNode, ObjectRef,
    PredicateRef, Quad, QuadRef, SubjectRef,
};
use std::collections::HashMap;
use std::io::Write;

/// N3 serializer for writing RDF with Turtle-compatible syntax
#[derive(Debug, Clone)]
pub struct N3Serializer {
    /// Base IRI for relative IRI resolution
    base_iri: Option<String>,
    /// Prefix declarations for compact serialization
    prefixes: HashMap<String, String>,
    /// Pretty printing with indentation
    pretty: bool,
}

impl N3Serializer {
    /// Create a new N3 serializer
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add standard N3 prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );

        Self {
            base_iri: None,
            prefixes,
            pretty: false,
        }
    }

    /// Set the base IRI
    pub fn with_base_iri(mut self, base: &str) -> Self {
        self.base_iri = Some(base.to_string());
        self
    }

    /// Add a prefix mapping
    pub fn with_prefix(mut self, prefix: &str, iri: &str) -> Self {
        self.prefixes.insert(prefix.to_string(), iri.to_string());
        self
    }

    /// Enable pretty printing
    pub fn pretty(mut self) -> Self {
        self.pretty = true;
        self
    }

    /// Wrap this serializer for a specific writer
    pub fn for_writer<W: Write + 'static>(self, writer: W) -> N3Writer<W> {
        N3Writer {
            writer,
            serializer: self,
            buffer: Vec::new(),
        }
    }

    /// Serialize quads as N3 triples (only default graph)
    fn serialize_quads<W: Write>(&self, quads: &[Quad], writer: &mut W) -> Result<(), FormatError> {
        // Write prefix declarations
        for (prefix, namespace) in &self.prefixes {
            writeln!(writer, "@prefix {}: <{}> .", prefix, namespace).map_err(FormatError::from)?;
        }

        if !self.prefixes.is_empty() {
            writeln!(writer).map_err(FormatError::from)?;
        }

        // Write base if present
        if let Some(base) = &self.base_iri {
            writeln!(writer, "@base <{}> .", base).map_err(FormatError::from)?;
            writeln!(writer).map_err(FormatError::from)?;
        }

        // Serialize triples (only from default graph)
        for quad in quads {
            // N3 typically handles only default graph; named graphs would need special syntax
            if matches!(quad.graph_name(), GraphName::DefaultGraph) {
                self.serialize_triple(quad.as_ref(), writer)?;
                writeln!(writer, " .").map_err(FormatError::from)?;
            }
        }

        Ok(())
    }

    fn serialize_triple<W: Write>(
        &self,
        quad: QuadRef<'_>,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        self.write_subject(quad.subject(), writer)?;
        write!(writer, " ").map_err(FormatError::from)?;

        self.write_predicate(quad.predicate(), writer)?;
        write!(writer, " ").map_err(FormatError::from)?;

        self.write_object(quad.object(), writer)?;

        Ok(())
    }

    fn write_subject<W: Write>(
        &self,
        subject: SubjectRef<'_>,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        match subject {
            SubjectRef::NamedNode(node) => self.write_named_node(node, writer)?,
            SubjectRef::BlankNode(node) => {
                let id = node.as_str();
                let id = id.strip_prefix("_:").unwrap_or(id);
                write!(writer, "_:{}", id).map_err(FormatError::from)?;
            }
            SubjectRef::Variable(var) => {
                // N3 supports variables with ?variable syntax
                write!(writer, "?{}", var.name()).map_err(FormatError::from)?;
            }
        }
        Ok(())
    }

    fn write_predicate<W: Write>(
        &self,
        predicate: PredicateRef<'_>,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        match predicate {
            PredicateRef::NamedNode(node) => {
                // N3 uses 'a' for rdf:type
                if node.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    write!(writer, "a").map_err(FormatError::from)?;
                } else if node.as_str() == "http://www.w3.org/2002/07/owl#sameAs" {
                    // N3 uses '=' for owl:sameAs
                    write!(writer, "=").map_err(FormatError::from)?;
                } else {
                    self.write_named_node(node, writer)?;
                }
            }
            PredicateRef::Variable(var) => {
                write!(writer, "?{}", var.name()).map_err(FormatError::from)?;
            }
        }
        Ok(())
    }

    fn write_object<W: Write>(
        &self,
        object: ObjectRef<'_>,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        match object {
            ObjectRef::NamedNode(node) => self.write_named_node(node, writer)?,
            ObjectRef::BlankNode(node) => {
                let id = node.as_str();
                let id = id.strip_prefix("_:").unwrap_or(id);
                write!(writer, "_:{}", id).map_err(FormatError::from)?;
            }
            ObjectRef::Literal(literal) => self.write_literal(literal, writer)?,
            ObjectRef::Variable(var) => {
                write!(writer, "?{}", var.name()).map_err(FormatError::from)?;
            }
        }
        Ok(())
    }

    fn write_named_node<W: Write>(
        &self,
        node: &NamedNode,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        let iri = node.as_str();

        // Try to use a prefix
        for (prefix, namespace) in &self.prefixes {
            if let Some(local) = iri.strip_prefix(namespace) {
                write!(writer, "{}:{}", prefix, local).map_err(FormatError::from)?;
                return Ok(());
            }
        }

        // Use full IRI
        write!(writer, "<{}>", iri).map_err(FormatError::from)?;
        Ok(())
    }

    fn write_literal<W: Write>(
        &self,
        literal: &Literal,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        let value = literal.value();

        // N3 supports some numeric shortcuts
        if let Some(_datatype) = self.check_numeric_shortcut(literal) {
            write!(writer, "{}", value).map_err(FormatError::from)?;
            return Ok(());
        }

        // Regular string literal
        let escaped = self.escape_string(value);
        write!(writer, "\"{}\"", escaped).map_err(FormatError::from)?;

        if let Some(lang) = literal.language() {
            write!(writer, "@{}", lang).map_err(FormatError::from)?;
        } else {
            let datatype = literal.datatype();
            if datatype.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                write!(writer, "^^").map_err(FormatError::from)?;
                self.write_named_node(&datatype.into_owned(), writer)?;
            }
        }

        Ok(())
    }

    fn check_numeric_shortcut(&self, literal: &Literal) -> Option<String> {
        let datatype = literal.datatype();
        let value = literal.value();

        match datatype.as_str() {
            "http://www.w3.org/2001/XMLSchema#integer" if value.parse::<i64>().is_ok() => {
                Some("integer".to_string())
            }
            "http://www.w3.org/2001/XMLSchema#decimal" if value.parse::<f64>().is_ok() => {
                Some("decimal".to_string())
            }
            "http://www.w3.org/2001/XMLSchema#double" if value.parse::<f64>().is_ok() => {
                Some("double".to_string())
            }
            "http://www.w3.org/2001/XMLSchema#boolean" if value == "true" || value == "false" => {
                Some("boolean".to_string())
            }
            _ => None,
        }
    }

    fn escape_string(&self, s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '\\' => result.push_str("\\\\"),
                '\"' => result.push_str("\\\""),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                c if c.is_control() => {
                    result.push_str(&format!("\\u{:04X}", c as u32));
                }
                c => result.push(c),
            }
        }
        result
    }
}

impl Default for N3Serializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer wrapper for N3 serialization
pub struct N3Writer<W: Write> {
    writer: W,
    serializer: N3Serializer,
    buffer: Vec<Quad>,
}

impl<W: Write> N3Writer<W> {
    /// Serialize a single quad (buffered until finish)
    pub fn serialize_quad(&mut self, quad: QuadRef<'_>) -> Result<(), FormatError> {
        self.buffer.push(quad.into());
        Ok(())
    }

    /// Finish serialization and return the writer
    pub fn finish(mut self) -> Result<W, FormatError> {
        self.serializer
            .serialize_quads(&self.buffer, &mut self.writer)?;
        Ok(self.writer)
    }
}

/// Implement the QuadSerializer trait for integration with the format system
impl<W: Write> super::serializer::QuadSerializer<W> for N3Writer<W> {
    fn serialize_quad(&mut self, quad: QuadRef<'_>) -> super::serializer::QuadSerializeResult {
        N3Writer::serialize_quad(self, quad)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn finish(self: Box<Self>) -> super::error::SerializeResult<W> {
        N3Writer::finish(*self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Triple, Subject, Object};

    #[test]
    fn test_n3_serialize_triple() {
        let serializer = N3Serializer::new();
        let mut writer = Vec::new();

        let triple = Triple::new(
            Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Object::NamedNode(NamedNode::new("http://example.org/object").unwrap()),
        );

        let quads = vec![Quad::from(triple)];
        serializer.serialize_quads(&quads, &mut writer).unwrap();

        let output = String::from_utf8(writer).unwrap();
        assert!(output.contains("@prefix"));
        assert!(output.contains("<http://example.org/subject>"));
        assert!(output.contains("<http://example.org/predicate>"));
        assert!(output.contains("<http://example.org/object>"));
    }

    #[test]
    fn test_n3_rdf_type_abbreviation() {
        let serializer = N3Serializer::new();
        let mut writer = Vec::new();

        let triple = Triple::new(
            Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
            Object::NamedNode(NamedNode::new("http://example.org/Type").unwrap()),
        );

        let quads = vec![Quad::from(triple)];
        serializer.serialize_quads(&quads, &mut writer).unwrap();

        let output = String::from_utf8(writer).unwrap();
        assert!(output.contains(" a "));
    }
}
