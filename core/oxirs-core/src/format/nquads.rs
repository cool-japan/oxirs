//! N-Quads format serializer and parser
//!
//! N-Quads is a line-based syntax for RDF quads (subject predicate object graph).
//! Each line contains one quad, with components separated by spaces and ending with a period.
//!
//! W3C Specification: https://www.w3.org/TR/n-quads/

use super::error::FormatError;
use crate::model::{
    GraphNameRef, Literal, ObjectRef,
    PredicateRef, QuadRef, SubjectRef,
};
use std::io::Write;

/// N-Quads serializer for writing RDF quads in line-based format
#[derive(Debug, Clone)]
pub struct NQuadsSerializer;

impl NQuadsSerializer {
    /// Create a new N-Quads serializer
    pub fn new() -> Self {
        Self
    }

    /// Wrap this serializer for a specific writer
    pub fn for_writer<W: Write + 'static>(self, writer: W) -> NQuadsWriter<W> {
        NQuadsWriter {
            writer,
            serializer: self,
        }
    }

    /// Serialize a single quad to N-Quads format
    fn serialize_quad<W: Write>(
        &self,
        quad: QuadRef<'_>,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        // Format: <subject> <predicate> <object> [<graph>] .
        self.write_subject(quad.subject(), writer)?;
        write!(writer, " ").map_err(FormatError::from)?;

        self.write_predicate(quad.predicate(), writer)?;
        write!(writer, " ").map_err(FormatError::from)?;

        self.write_object(quad.object(), writer)?;

        // Write graph name if not default graph
        if !matches!(quad.graph_name(), GraphNameRef::DefaultGraph) {
            write!(writer, " ").map_err(FormatError::from)?;
            self.write_graph_name(quad.graph_name(), writer)?;
        }

        writeln!(writer, " .").map_err(FormatError::from)?;
        Ok(())
    }

    fn write_subject<W: Write>(
        &self,
        subject: SubjectRef<'_>,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        match subject {
            SubjectRef::NamedNode(n) => {
                write!(writer, "<{}>", n.as_str()).map_err(FormatError::from)?;
            }
            SubjectRef::BlankNode(b) => {
                let id = b.as_str();
                let id = id.strip_prefix("_:").unwrap_or(id);
                write!(writer, "_:{}", id).map_err(FormatError::from)?;
            }
            SubjectRef::Variable(v) => {
                write!(writer, "?{}", v.name()).map_err(FormatError::from)?;
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
            PredicateRef::NamedNode(n) => {
                write!(writer, "<{}>", n.as_str()).map_err(FormatError::from)?;
            }
            PredicateRef::Variable(v) => {
                write!(writer, "?{}", v.name()).map_err(FormatError::from)?;
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
            ObjectRef::NamedNode(n) => {
                write!(writer, "<{}>", n.as_str()).map_err(FormatError::from)?;
            }
            ObjectRef::BlankNode(b) => {
                let id = b.as_str();
                let id = id.strip_prefix("_:").unwrap_or(id);
                write!(writer, "_:{}", id).map_err(FormatError::from)?;
            }
            ObjectRef::Literal(l) => {
                self.write_literal(l, writer)?;
            }
            ObjectRef::Variable(v) => {
                write!(writer, "?{}", v.name()).map_err(FormatError::from)?;
            }
        }
        Ok(())
    }

    fn write_graph_name<W: Write>(
        &self,
        graph: GraphNameRef<'_>,
        writer: &mut W,
    ) -> Result<(), FormatError> {
        match graph {
            GraphNameRef::NamedNode(n) => {
                write!(writer, "<{}>", n.as_str()).map_err(FormatError::from)?;
            }
            GraphNameRef::BlankNode(b) => {
                let id = b.as_str();
                let id = id.strip_prefix("_:").unwrap_or(id);
                write!(writer, "_:{}", id).map_err(FormatError::from)?;
            }
            GraphNameRef::Variable(v) => {
                write!(writer, "?{}", v.name()).map_err(FormatError::from)?;
            }
            GraphNameRef::DefaultGraph => {
                // Default graph should not be written in N-Quads
            }
        }
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
            // xsd:string is implicit, don't write it
            if datatype.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                write!(writer, "^^<{}>", datatype.as_str()).map_err(FormatError::from)?;
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
}

impl Default for NQuadsSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer wrapper for N-Quads serialization
pub struct NQuadsWriter<W: Write> {
    writer: W,
    serializer: NQuadsSerializer,
}

impl<W: Write> NQuadsWriter<W> {
    /// Serialize a single quad
    pub fn serialize_quad(&mut self, quad: QuadRef<'_>) -> Result<(), FormatError> {
        self.serializer.serialize_quad(quad, &mut self.writer)
    }

    /// Finish serialization and return the writer
    pub fn finish(self) -> Result<W, FormatError> {
        Ok(self.writer)
    }
}

/// Implement the QuadSerializer trait for integration with the format system
impl<W: Write> super::serializer::QuadSerializer<W> for NQuadsWriter<W> {
    fn serialize_quad(&mut self, quad: QuadRef<'_>) -> super::serializer::QuadSerializeResult {
        NQuadsWriter::serialize_quad(self, quad)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn finish(self: Box<Self>) -> super::error::SerializeResult<W> {
        Ok(self.writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Triple, Subject, Object, Quad};

    #[test]
    fn test_nquads_serialize_triple() {
        let serializer = NQuadsSerializer::new();
        let mut writer = Vec::new();

        let triple = Triple::new(
            Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Object::NamedNode(NamedNode::new("http://example.org/object").unwrap()),
        );

        let quad = Quad::from(triple);
        serializer
            .serialize_quad(quad.as_ref(), &mut writer)
            .unwrap();

        let output = String::from_utf8(writer).unwrap();
        assert_eq!(
            output,
            "<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .\n"
        );
    }

    #[test]
    fn test_nquads_serialize_literal() {
        let serializer = NQuadsSerializer::new();
        let mut writer = Vec::new();

        let triple = Triple::new(
            Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap()),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Object::Literal(Literal::new("Hello World")),
        );

        let quad = Quad::from(triple);
        serializer
            .serialize_quad(quad.as_ref(), &mut writer)
            .unwrap();

        let output = String::from_utf8(writer).unwrap();
        assert_eq!(
            output,
            "<http://example.org/subject> <http://example.org/predicate> \"Hello World\" .\n"
        );
    }
}
