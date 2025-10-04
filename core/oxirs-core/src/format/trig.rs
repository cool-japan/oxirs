//! TriG format serializer and parser
//!
//! TriG extends Turtle with support for named graphs, allowing multiple RDF graphs
//! to be serialized in a single document with graph-level organization.
//!
//! W3C Specification: https://www.w3.org/TR/trig/

use super::error::FormatError;
use crate::model::{
    GraphName, Literal, NamedNode, ObjectRef,
    PredicateRef, Quad, QuadRef, SubjectRef,
};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;

/// TriG serializer for writing RDF quads with named graph support
#[derive(Debug, Clone)]
pub struct TriGSerializer {
    /// Base IRI for relative IRI resolution
    base_iri: Option<String>,
    /// Prefix declarations for compact serialization
    prefixes: HashMap<String, String>,
    /// Pretty printing with indentation
    pretty: bool,
}

impl TriGSerializer {
    /// Create a new TriG serializer
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add common prefixes
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
    pub fn for_writer<W: Write + 'static>(self, writer: W) -> TriGWriter<W> {
        TriGWriter {
            writer,
            serializer: self,
            buffer: Vec::new(),
        }
    }

    /// Serialize quads grouped by graph
    fn serialize_quads<W: Write>(&self, quads: &[Quad], writer: &mut W) -> Result<(), FormatError> {
        // Write prefix declarations
        for (prefix, namespace) in &self.prefixes {
            writeln!(writer, "@prefix {}: <{}> .", prefix, namespace).map_err(FormatError::from)?;
        }

        if !self.prefixes.is_empty() {
            writeln!(writer).map_err(FormatError::from)?;
        }

        // Group quads by graph
        let grouped = self.group_quads_by_graph(quads);

        for (graph_name, graph_quads) in grouped {
            match graph_name {
                GraphName::DefaultGraph => {
                    // Serialize default graph triples directly
                    for quad in graph_quads {
                        self.serialize_triple(quad.as_ref(), writer)?;
                        writeln!(writer, " .").map_err(FormatError::from)?;
                    }
                }
                GraphName::NamedNode(node) => {
                    // Named graph
                    self.write_named_node(&node, writer)?;
                    writeln!(writer, " {{").map_err(FormatError::from)?;

                    for quad in graph_quads {
                        if self.pretty {
                            write!(writer, "    ").map_err(FormatError::from)?;
                        }
                        self.serialize_triple(quad.as_ref(), writer)?;
                        writeln!(writer, " .").map_err(FormatError::from)?;
                    }

                    writeln!(writer, "}}").map_err(FormatError::from)?;
                }
                GraphName::BlankNode(node) => {
                    // Blank node graph
                    let id = node.as_str();
                    let id = id.strip_prefix("_:").unwrap_or(id);
                    writeln!(writer, "_:{} {{", id).map_err(FormatError::from)?;

                    for quad in graph_quads {
                        if self.pretty {
                            write!(writer, "    ").map_err(FormatError::from)?;
                        }
                        self.serialize_triple(quad.as_ref(), writer)?;
                        writeln!(writer, " .").map_err(FormatError::from)?;
                    }

                    writeln!(writer, "}}").map_err(FormatError::from)?;
                }
                GraphName::Variable(_) => {
                    return Err(FormatError::InvalidData(
                        "Cannot serialize variable graph names".to_string(),
                    ));
                }
            }

            if self.pretty {
                writeln!(writer).map_err(FormatError::from)?;
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
                // Check for rdf:type abbreviation
                if node.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    write!(writer, "a").map_err(FormatError::from)?;
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

    fn group_quads_by_graph<'a>(&self, quads: &'a [Quad]) -> BTreeMap<GraphName, Vec<&'a Quad>> {
        let mut grouped = BTreeMap::new();

        for quad in quads {
            grouped
                .entry(quad.graph_name().clone())
                .or_insert_with(Vec::new)
                .push(quad);
        }

        grouped
    }
}

impl Default for TriGSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer wrapper for TriG serialization
pub struct TriGWriter<W: Write> {
    writer: W,
    serializer: TriGSerializer,
    buffer: Vec<Quad>,
}

impl<W: Write> TriGWriter<W> {
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
impl<W: Write> super::serializer::QuadSerializer<W> for TriGWriter<W> {
    fn serialize_quad(&mut self, quad: QuadRef<'_>) -> super::serializer::QuadSerializeResult {
        TriGWriter::serialize_quad(self, quad)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn finish(self: Box<Self>) -> super::error::SerializeResult<W> {
        TriGWriter::finish(*self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Triple, Subject, Object, Quad};

    #[test]
    fn test_trig_serialize_default_graph() {
        let serializer = TriGSerializer::new();
        let mut writer = Vec::new();

        let triple = Triple::new(
            Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Object::NamedNode(NamedNode::new("http://example.org/object").unwrap()),
        );

        let quads = vec![Quad::from(triple)];
        serializer.serialize_quads(&quads, &mut writer).unwrap();

        let output = String::from_utf8(writer).unwrap();
        assert!(output.contains("<http://example.org/subject>"));
        assert!(output.contains("<http://example.org/predicate>"));
        assert!(output.contains("<http://example.org/object>"));
    }

    #[test]
    fn test_trig_serialize_named_graph() {
        let serializer = TriGSerializer::new();
        let mut writer = Vec::new();

        let quad = Quad::new(
            Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Object::NamedNode(NamedNode::new("http://example.org/object").unwrap()),
            GraphName::NamedNode(NamedNode::new("http://example.org/graph").unwrap()),
        );

        let quads = vec![quad];
        serializer.serialize_quads(&quads, &mut writer).unwrap();

        let output = String::from_utf8(writer).unwrap();
        assert!(output.contains("<http://example.org/graph>"));
        assert!(output.contains("{"));
        assert!(output.contains("}"));
    }
}
