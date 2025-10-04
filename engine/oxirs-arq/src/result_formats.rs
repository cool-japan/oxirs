//! Extended Result Format Support for SPARQL
//!
//! This module extends the basic result formats with additional serializers,
//! streaming capabilities, and custom format registration.

use crate::algebra::{Binding, Term, Variable};
use crate::results::{QueryResult, ResultFormat};
use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use std::io::Write;
use std::sync::Arc;

/// XML Result Serializer for SPARQL Results XML Format
pub struct XmlResultSerializer;

impl XmlResultSerializer {
    /// Serialize query result to W3C SPARQL Results XML format
    pub fn serialize<W: Write>(result: &QueryResult, writer: &mut W) -> Result<()> {
        match result {
            QueryResult::Boolean(value) => Self::serialize_boolean(*value, writer),
            QueryResult::Bindings {
                variables,
                solutions,
            } => Self::serialize_bindings(variables, solutions, writer),
            QueryResult::Graph(_) => {
                bail!("XML serialization not supported for graph results")
            }
        }
    }

    fn serialize_boolean<W: Write>(value: bool, writer: &mut W) -> Result<()> {
        writeln!(writer, "<?xml version=\"1.0\"?>")?;
        writeln!(
            writer,
            "<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">"
        )?;
        writeln!(writer, "  <head/>")?;
        writeln!(writer, "  <boolean>{value}</boolean>")?;
        writeln!(writer, "</sparql>")?;
        Ok(())
    }

    fn serialize_bindings<W: Write>(
        variables: &[Variable],
        solutions: &[Binding],
        writer: &mut W,
    ) -> Result<()> {
        writeln!(writer, "<?xml version=\"1.0\"?>")?;
        writeln!(
            writer,
            "<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">"
        )?;

        // Head section with variables
        writeln!(writer, "  <head>")?;
        for var in variables {
            let var_name = var.to_string().trim_start_matches('?').to_string();
            writeln!(writer, "    <variable name=\"{}\"/>", var_name)?;
        }
        writeln!(writer, "  </head>")?;

        // Results section
        writeln!(writer, "  <results>")?;
        for binding in solutions {
            writeln!(writer, "    <result>")?;
            for (var, term) in binding {
                Self::serialize_binding_term(writer, var, term)?;
            }
            writeln!(writer, "    </result>")?;
        }
        writeln!(writer, "  </results>")?;

        writeln!(writer, "</sparql>")?;
        Ok(())
    }

    fn serialize_binding_term<W: Write>(writer: &mut W, var: &Variable, term: &Term) -> Result<()> {
        let var_name = var.to_string().trim_start_matches('?').to_string();
        match term {
            Term::Iri(iri) => {
                // Strip angle brackets from IRI and escape XML
                let iri_str = iri.to_string();
                let iri_clean = iri_str.trim_start_matches('<').trim_end_matches('>');
                writeln!(
                    writer,
                    "      <binding name=\"{}\"><uri>{}</uri></binding>",
                    var_name,
                    Self::escape_xml(iri_clean)
                )?;
            }
            Term::Literal(lit) => {
                write!(writer, "      <binding name=\"{}\">", var_name)?;
                if let Some(lang) = &lit.language {
                    write!(
                        writer,
                        "<literal xml:lang=\"{}\">{}</literal>",
                        lang,
                        Self::escape_xml(&lit.value)
                    )?;
                } else if let Some(datatype) = &lit.datatype {
                    write!(
                        writer,
                        "<literal datatype=\"{}\">{}</literal>",
                        datatype,
                        Self::escape_xml(&lit.value)
                    )?;
                } else {
                    write!(
                        writer,
                        "<literal>{}</literal>",
                        Self::escape_xml(&lit.value)
                    )?;
                }
                writeln!(writer, "</binding>")?;
            }
            Term::BlankNode(id) => {
                writeln!(
                    writer,
                    "      <binding name=\"{}\"><bnode>{}</bnode></binding>",
                    var_name,
                    Self::escape_xml(id)
                )?;
            }
            Term::QuotedTriple(_) => {
                // RDF-star triple terms - serialize as string for now
                writeln!(
                    writer,
                    "      <binding name=\"{}\"><literal>{}</literal></binding>",
                    var_name,
                    Self::escape_xml(&term.to_string())
                )?;
            }
            Term::Variable(v) => {
                // Unbound variable in result - skip
                let _ = v;
            }
            Term::PropertyPath(_) => {
                // Property path - serialize as string
                writeln!(
                    writer,
                    "      <binding name=\"{}\"><literal>{}</literal></binding>",
                    var_name,
                    Self::escape_xml(&term.to_string())
                )?;
            }
        }
        Ok(())
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

/// Binary Result Serializer for efficient storage/transmission
pub struct BinaryResultSerializer;

impl BinaryResultSerializer {
    /// Serialize query result to compact binary format
    pub fn serialize<W: Write>(result: &QueryResult, writer: &mut W) -> Result<()> {
        match result {
            QueryResult::Boolean(value) => {
                // Type byte: 0x01 = boolean
                writer.write_all(&[0x01])?;
                writer.write_all(&[if *value { 1 } else { 0 }])?;
                Ok(())
            }
            QueryResult::Bindings {
                variables,
                solutions,
            } => {
                // Type byte: 0x02 = bindings
                writer.write_all(&[0x02])?;

                // Variable count (u32)
                let var_count = variables.len() as u32;
                writer.write_all(&var_count.to_le_bytes())?;

                // Variable names
                for var in variables {
                    let name = var.to_string();
                    let name_bytes = name.as_bytes();
                    writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
                    writer.write_all(name_bytes)?;
                }

                // Solution count (u32)
                let solution_count = solutions.len() as u32;
                writer.write_all(&solution_count.to_le_bytes())?;

                // Solutions
                for binding in solutions {
                    Self::serialize_binding(writer, binding, variables)?;
                }

                Ok(())
            }
            QueryResult::Graph(_) => {
                bail!("Binary serialization not supported for graph results")
            }
        }
    }

    fn serialize_binding<W: Write>(
        writer: &mut W,
        binding: &Binding,
        variables: &[Variable],
    ) -> Result<()> {
        // For each variable, write presence bit and value if present
        for var in variables {
            if let Some(term) = binding.get(var) {
                writer.write_all(&[1])?; // Present
                Self::serialize_term(writer, term)?;
            } else {
                writer.write_all(&[0])?; // Absent
            }
        }
        Ok(())
    }

    fn serialize_term<W: Write>(writer: &mut W, term: &Term) -> Result<()> {
        match term {
            Term::Iri(iri) => {
                writer.write_all(&[0x01])?; // IRI type
                let iri_str = iri.to_string();
                let iri_bytes = iri_str.as_bytes();
                writer.write_all(&(iri_bytes.len() as u32).to_le_bytes())?;
                writer.write_all(iri_bytes)?;
            }
            Term::Literal(lit) => {
                writer.write_all(&[0x02])?; // Literal type
                let value_bytes = lit.value.as_bytes();
                writer.write_all(&(value_bytes.len() as u32).to_le_bytes())?;
                writer.write_all(value_bytes)?;

                // Language tag (optional)
                if let Some(lang) = &lit.language {
                    writer.write_all(&[1])?; // Has language
                    let lang_bytes = lang.as_bytes();
                    writer.write_all(&(lang_bytes.len() as u16).to_le_bytes())?;
                    writer.write_all(lang_bytes)?;
                } else {
                    writer.write_all(&[0])?; // No language
                }

                // Datatype (optional)
                if let Some(dt) = &lit.datatype {
                    writer.write_all(&[1])?; // Has datatype
                    let dt_str = dt.to_string();
                    let dt_bytes = dt_str.as_bytes();
                    writer.write_all(&(dt_bytes.len() as u32).to_le_bytes())?;
                    writer.write_all(dt_bytes)?;
                } else {
                    writer.write_all(&[0])?; // No datatype
                }
            }
            Term::BlankNode(id) => {
                writer.write_all(&[0x03])?; // Blank node type
                let id_bytes = id.as_bytes();
                writer.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
                writer.write_all(id_bytes)?;
            }
            _ => {
                // Other term types - serialize as string
                writer.write_all(&[0xFF])?; // Generic type
                let term_str = term.to_string();
                let term_bytes = term_str.as_bytes();
                writer.write_all(&(term_bytes.len() as u32).to_le_bytes())?;
                writer.write_all(term_bytes)?;
            }
        }
        Ok(())
    }
}

/// Streaming result iterator for memory-efficient result processing
pub struct StreamingResultIterator {
    variables: Vec<Variable>,
    solutions: Vec<Binding>,
    position: usize,
    chunk_size: usize,
}

impl StreamingResultIterator {
    /// Create a new streaming iterator
    pub fn new(variables: Vec<Variable>, solutions: Vec<Binding>) -> Self {
        Self {
            variables,
            solutions,
            position: 0,
            chunk_size: 1000, // Default chunk size
        }
    }

    /// Create with custom chunk size
    pub fn with_chunk_size(
        variables: Vec<Variable>,
        solutions: Vec<Binding>,
        chunk_size: usize,
    ) -> Self {
        Self {
            variables,
            solutions,
            position: 0,
            chunk_size,
        }
    }

    /// Get variables
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Get next chunk of results
    pub fn next_chunk(&mut self) -> Option<&[Binding]> {
        if self.position >= self.solutions.len() {
            return None;
        }

        let end = (self.position + self.chunk_size).min(self.solutions.len());
        let chunk = &self.solutions[self.position..end];
        self.position = end;
        Some(chunk)
    }

    /// Check if there are more results
    pub fn has_more(&self) -> bool {
        self.position < self.solutions.len()
    }

    /// Get total number of solutions
    pub fn total_count(&self) -> usize {
        self.solutions.len()
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Convert to QueryResult (consumes iterator)
    pub fn into_query_result(self) -> QueryResult {
        QueryResult::Bindings {
            variables: self.variables,
            solutions: self.solutions,
        }
    }
}

impl Iterator for StreamingResultIterator {
    type Item = Binding;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.solutions.len() {
            None
        } else {
            let binding = self.solutions[self.position].clone();
            self.position += 1;
            Some(binding)
        }
    }
}

/// Custom format serializer trait
pub trait CustomFormatSerializer: Send + Sync {
    /// Get the format name/identifier
    fn format_name(&self) -> &str;

    /// Get the MIME type
    fn mime_type(&self) -> &str;

    /// Get the file extension
    fn extension(&self) -> &str;

    /// Serialize a query result (uses dynamic dispatch for Write trait)
    fn serialize(&self, result: &QueryResult, writer: &mut dyn Write) -> Result<()>;

    /// Check if this serializer supports the given result type
    fn supports(&self, result: &QueryResult) -> bool {
        match result {
            QueryResult::Boolean(_) => true,
            QueryResult::Bindings { .. } => true,
            QueryResult::Graph(_) => false,
        }
    }
}

/// Registry for custom result format serializers
pub struct FormatRegistry {
    formats: DashMap<String, Arc<dyn CustomFormatSerializer>>,
}

impl FormatRegistry {
    /// Create a new format registry
    pub fn new() -> Self {
        Self {
            formats: DashMap::new(),
        }
    }

    /// Register a custom format serializer
    pub fn register<F: CustomFormatSerializer + 'static>(&self, serializer: F) -> Result<()> {
        let name = serializer.format_name().to_string();
        self.formats.insert(name, Arc::new(serializer));
        Ok(())
    }

    /// Get a format serializer by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn CustomFormatSerializer>> {
        self.formats.get(name).map(|entry| Arc::clone(&*entry))
    }

    /// Check if a format is registered
    pub fn is_registered(&self, name: &str) -> bool {
        self.formats.contains_key(name)
    }

    /// Get all registered format names
    pub fn registered_formats(&self) -> Vec<String> {
        self.formats
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Serialize using a registered format
    pub fn serialize(
        &self,
        format_name: &str,
        result: &QueryResult,
        writer: &mut dyn Write,
    ) -> Result<()> {
        let serializer = self
            .get(format_name)
            .ok_or_else(|| anyhow!("Format not registered: {}", format_name))?;

        if !serializer.supports(result) {
            bail!(
                "Format {} does not support result type",
                serializer.format_name()
            );
        }

        serializer.serialize(result, writer)
    }
}

impl Default for FormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Format conversion utilities
pub struct FormatConverter;

impl FormatConverter {
    /// Convert between result formats
    pub fn convert(
        result: &QueryResult,
        from_format: ResultFormat,
        to_format: ResultFormat,
    ) -> Result<Vec<u8>> {
        // If same format, just serialize
        if from_format == to_format {
            let mut buffer = Vec::new();
            crate::results::ResultSerializer::serialize(result, to_format, &mut buffer)?;
            return Ok(buffer);
        }

        // Convert through intermediate representation (already have QueryResult)
        let mut buffer = Vec::new();
        crate::results::ResultSerializer::serialize(result, to_format, &mut buffer)?;
        Ok(buffer)
    }

    /// Detect format from content
    pub fn detect_format(content: &[u8]) -> Option<ResultFormat> {
        // Check for XML
        if content.starts_with(b"<?xml") || content.starts_with(b"<sparql") {
            return Some(ResultFormat::Xml);
        }

        // Check for JSON
        if content.starts_with(b"{") || content.starts_with(b"[") {
            return Some(ResultFormat::Json);
        }

        // Check for binary format marker
        if !content.is_empty() && (content[0] == 0x01 || content[0] == 0x02) {
            return Some(ResultFormat::Binary);
        }

        // Check for CSV/TSV by looking for delimiters
        if content.contains(&b',') {
            return Some(ResultFormat::Csv);
        }

        if content.contains(&b'\t') {
            return Some(ResultFormat::Tsv);
        }

        None
    }

    /// Validate format compatibility
    pub fn is_compatible(result: &QueryResult, format: ResultFormat) -> bool {
        match (result, format) {
            (QueryResult::Boolean(_), _) => true,
            (QueryResult::Bindings { .. }, _) => true,
            (QueryResult::Graph(_), ResultFormat::Json) => false,
            (QueryResult::Graph(_), ResultFormat::Xml) => false,
            (QueryResult::Graph(_), ResultFormat::Binary) => false,
            (QueryResult::Graph(_), _) => true, // CSV/TSV can serialize triples
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::{NamedNode, Variable};
    use std::collections::HashMap;

    fn create_test_variable(name: &str) -> Variable {
        Variable::new(name).unwrap()
    }

    fn create_test_iri(iri: &str) -> NamedNode {
        NamedNode::new(iri).unwrap()
    }

    #[test]
    fn test_xml_boolean_result() {
        let result = QueryResult::Boolean(true);
        let mut buffer = Vec::new();
        XmlResultSerializer::serialize(&result, &mut buffer).unwrap();

        let xml = String::from_utf8(buffer).unwrap();
        assert!(xml.contains("<boolean>true</boolean>"));
        assert!(xml.contains("<?xml version=\"1.0\"?>"));
    }

    #[test]
    fn test_xml_bindings_result() {
        let var = create_test_variable("x");
        let variables = vec![var.clone()];

        let mut binding = HashMap::new();
        binding.insert(var, Term::Iri(create_test_iri("http://example.org/alice")));

        let result = QueryResult::Bindings {
            variables,
            solutions: vec![binding],
        };

        let mut buffer = Vec::new();
        XmlResultSerializer::serialize(&result, &mut buffer).unwrap();

        let xml = String::from_utf8(buffer).unwrap();
        assert!(xml.contains("<variable name=\"x\"/>"));
        assert!(xml.contains("<uri>http://example.org/alice</uri>"));
        assert!(xml.contains("<result>"));
    }

    #[test]
    fn test_xml_literal_with_language() {
        let var = create_test_variable("name");
        let variables = vec![var.clone()];

        let mut binding = HashMap::new();
        binding.insert(
            var,
            Term::Literal(crate::algebra::Literal {
                value: "Alice".to_string(),
                language: Some("en".to_string()),
                datatype: None,
            }),
        );

        let result = QueryResult::Bindings {
            variables,
            solutions: vec![binding],
        };

        let mut buffer = Vec::new();
        XmlResultSerializer::serialize(&result, &mut buffer).unwrap();

        let xml = String::from_utf8(buffer).unwrap();
        assert!(xml.contains("xml:lang=\"en\""));
        assert!(xml.contains("Alice"));
    }

    #[test]
    fn test_xml_escape() {
        let escaped = XmlResultSerializer::escape_xml("<>&\"'");
        assert_eq!(escaped, "&lt;&gt;&amp;&quot;&apos;");
    }

    #[test]
    fn test_binary_boolean_result() {
        let result = QueryResult::Boolean(true);
        let mut buffer = Vec::new();
        BinaryResultSerializer::serialize(&result, &mut buffer).unwrap();

        assert_eq!(buffer[0], 0x01); // Boolean type
        assert_eq!(buffer[1], 1); // True
    }

    #[test]
    fn test_binary_bindings_result() {
        let var = create_test_variable("x");
        let variables = vec![var.clone()];

        let mut binding = HashMap::new();
        binding.insert(var, Term::Iri(create_test_iri("http://example.org/test")));

        let result = QueryResult::Bindings {
            variables,
            solutions: vec![binding],
        };

        let mut buffer = Vec::new();
        BinaryResultSerializer::serialize(&result, &mut buffer).unwrap();

        assert_eq!(buffer[0], 0x02); // Bindings type
                                     // Should have variable count, variable name, solution count, etc.
        assert!(buffer.len() > 10);
    }

    #[test]
    fn test_streaming_iterator() {
        let var = create_test_variable("x");
        let variables = vec![var.clone()];

        let mut solutions = Vec::new();
        for i in 0..5 {
            let mut binding = HashMap::new();
            binding.insert(
                var.clone(),
                Term::Iri(create_test_iri(&format!("http://example.org/{i}"))),
            );
            solutions.push(binding);
        }

        let mut iter = StreamingResultIterator::new(variables.clone(), solutions);

        assert_eq!(iter.total_count(), 5);
        assert_eq!(iter.position(), 0);

        let mut count = 0;
        for _ in &mut iter {
            count += 1;
        }
        assert_eq!(count, 5);
        assert!(!iter.has_more());
    }

    #[test]
    fn test_streaming_iterator_chunks() {
        let var = create_test_variable("x");
        let variables = vec![var.clone()];

        let mut solutions = Vec::new();
        for i in 0..10 {
            let mut binding = HashMap::new();
            binding.insert(
                var.clone(),
                Term::Iri(create_test_iri(&format!("http://example.org/{i}"))),
            );
            solutions.push(binding);
        }

        let mut iter = StreamingResultIterator::with_chunk_size(variables, solutions, 3);

        assert_eq!(iter.total_count(), 10);

        let chunk1 = iter.next_chunk().unwrap();
        assert_eq!(chunk1.len(), 3);

        let chunk2 = iter.next_chunk().unwrap();
        assert_eq!(chunk2.len(), 3);

        let chunk3 = iter.next_chunk().unwrap();
        assert_eq!(chunk3.len(), 3);

        let chunk4 = iter.next_chunk().unwrap();
        assert_eq!(chunk4.len(), 1); // Last chunk

        assert!(iter.next_chunk().is_none());
    }

    #[test]
    fn test_format_registry() {
        let registry = FormatRegistry::new();

        // Define a simple custom serializer
        struct TestSerializer;
        impl CustomFormatSerializer for TestSerializer {
            fn format_name(&self) -> &str {
                "test"
            }
            fn mime_type(&self) -> &str {
                "application/test"
            }
            fn extension(&self) -> &str {
                "test"
            }
            fn serialize(&self, _result: &QueryResult, writer: &mut dyn Write) -> Result<()> {
                writer.write_all(b"TEST")?;
                Ok(())
            }
        }

        registry.register(TestSerializer).unwrap();
        assert!(registry.is_registered("test"));

        let result = QueryResult::Boolean(true);
        let mut buffer = Vec::new();
        registry.serialize("test", &result, &mut buffer).unwrap();

        assert_eq!(buffer, b"TEST");
    }

    #[test]
    fn test_format_converter_detect() {
        let xml = b"<?xml version=\"1.0\"?>";
        assert_eq!(FormatConverter::detect_format(xml), Some(ResultFormat::Xml));

        let json = b"{\"head\":{}}";
        assert_eq!(
            FormatConverter::detect_format(json),
            Some(ResultFormat::Json)
        );

        let binary = b"\x01\x01";
        assert_eq!(
            FormatConverter::detect_format(binary),
            Some(ResultFormat::Binary)
        );
    }

    #[test]
    fn test_format_compatibility() {
        let boolean_result = QueryResult::Boolean(true);
        assert!(FormatConverter::is_compatible(
            &boolean_result,
            ResultFormat::Json
        ));
        assert!(FormatConverter::is_compatible(
            &boolean_result,
            ResultFormat::Xml
        ));

        let bindings_result = QueryResult::Bindings {
            variables: vec![],
            solutions: vec![],
        };
        assert!(FormatConverter::is_compatible(
            &bindings_result,
            ResultFormat::Csv
        ));
    }

    #[test]
    fn test_streaming_iterator_reset() {
        let var = create_test_variable("x");
        let variables = vec![var.clone()];

        let mut solutions = Vec::new();
        for i in 0..3 {
            let mut binding = HashMap::new();
            binding.insert(
                var.clone(),
                Term::Iri(create_test_iri(&format!("http://example.org/{i}"))),
            );
            solutions.push(binding);
        }

        let mut iter = StreamingResultIterator::new(variables, solutions);

        // Iterate through all
        let mut count = 0;
        for _ in &mut iter {
            count += 1;
        }
        assert_eq!(count, 3);

        // Reset and iterate again
        iter.reset();
        assert_eq!(iter.position(), 0);

        let mut count2 = 0;
        for _ in &mut iter {
            count2 += 1;
        }
        assert_eq!(count2, 3);
    }
}
