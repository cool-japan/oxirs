//! RDF/XML Format Parser and Serializer
//!
//! Extracted and adapted from OxiGraph oxrdfxml with OxiRS enhancements.
//! Based on W3C RDF/XML specification: <https://www.w3.org/TR/rdf-syntax-grammar/>

#![allow(dead_code)]

use super::error::{ParseResult, RdfParseError};
use super::serializer::{QuadSerializer, SerializeResult};
use crate::model::{QuadRef, Triple, TripleRef};
use std::collections::HashMap;
use std::io::{Read, Write};

/// RDF/XML parser implementation
#[derive(Debug, Clone)]
pub struct RdfXmlParser {
    base_iri: Option<String>,
    prefixes: HashMap<String, String>,
    lenient: bool,
}

impl RdfXmlParser {
    /// Create a new RDF/XML parser
    pub fn new() -> Self {
        Self {
            base_iri: None,
            prefixes: HashMap::new(),
            lenient: false,
        }
    }

    /// Set base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Add a namespace prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Enable lenient parsing (skip some validations)
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    /// Parse RDF/XML from a reader
    pub fn parse_reader<R: Read>(&self, reader: R) -> ParseResult<Vec<Triple>> {
        // Use the actual RDF/XML parser from oxirs-core::rdfxml
        let mut parser = crate::rdfxml::RdfXmlParser::new();

        // Apply base IRI if set
        if let Some(base) = &self.base_iri {
            parser = parser
                .with_base_iri(base.clone())
                .map_err(|e| RdfParseError::syntax(format!("Invalid base IRI: {e}")))?;
        }

        // Apply lenient mode if enabled
        if self.lenient {
            parser = parser.lenient();
        }

        // Parse triples from reader
        let mut triples = Vec::new();
        for result in parser.for_reader(reader) {
            let triple = result.map_err(|e| RdfParseError::syntax(format!("{e}")))?;
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Parse RDF/XML from a byte slice
    pub fn parse_slice(&self, slice: &[u8]) -> ParseResult<Vec<Triple>> {
        let content = std::str::from_utf8(slice)
            .map_err(|e| RdfParseError::syntax(format!("Invalid UTF-8: {e}")))?;
        self.parse_str(content)
    }

    /// Parse RDF/XML from a string
    pub fn parse_str(&self, input: &str) -> ParseResult<Vec<Triple>> {
        // Use parse_reader with a byte slice cursor
        self.parse_reader(input.as_bytes())
    }

    /// Get the prefixes
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Get the base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }

    /// Check if lenient parsing is enabled
    pub fn is_lenient(&self) -> bool {
        self.lenient
    }
}

impl Default for RdfXmlParser {
    fn default() -> Self {
        Self::new()
    }
}

/// RDF/XML serializer implementation
#[derive(Debug, Clone)]
pub struct RdfXmlSerializer {
    base_iri: Option<String>,
    prefixes: HashMap<String, String>,
    pretty: bool,
    xml_declaration: bool,
}

impl RdfXmlSerializer {
    /// Create a new RDF/XML serializer
    pub fn new() -> Self {
        Self {
            base_iri: None,
            prefixes: HashMap::new(),
            pretty: false,
            xml_declaration: true,
        }
    }

    /// Set base IRI for generating relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Add a namespace prefix
    pub fn with_prefix(mut self, prefix: impl Into<String>, iri: impl Into<String>) -> Self {
        self.prefixes.insert(prefix.into(), iri.into());
        self
    }

    /// Enable pretty formatting
    pub fn pretty(mut self) -> Self {
        self.pretty = true;
        self
    }

    /// Disable XML declaration
    pub fn without_xml_declaration(mut self) -> Self {
        self.xml_declaration = false;
        self
    }

    /// Create a writer-based serializer
    pub fn for_writer<W: Write>(self, writer: W) -> WriterRdfXmlSerializer<W> {
        WriterRdfXmlSerializer::new(writer, self)
    }

    /// Serialize triples to an RDF/XML string
    pub fn serialize_to_string(&self, triples: &[Triple]) -> SerializeResult<String> {
        let mut buffer = Vec::new();
        {
            let mut serializer = self.clone().for_writer(&mut buffer);
            for triple in triples {
                serializer.serialize_triple(triple.as_ref())?;
            }
            serializer.finish()?;
        }
        String::from_utf8(buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Get the prefixes
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Get the base IRI
    pub fn base_iri(&self) -> Option<&str> {
        self.base_iri.as_deref()
    }

    /// Check if pretty formatting is enabled
    pub fn is_pretty(&self) -> bool {
        self.pretty
    }

    /// Check if XML declaration is enabled
    pub fn has_xml_declaration(&self) -> bool {
        self.xml_declaration
    }
}

impl Default for RdfXmlSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer-based RDF/XML serializer
#[allow(dead_code)]
pub struct WriterRdfXmlSerializer<W: Write> {
    writer: W,
    config: RdfXmlSerializer,
    headers_written: bool,
    triples: Vec<Triple>,
}

impl<W: Write> WriterRdfXmlSerializer<W> {
    /// Create a new writer serializer
    pub fn new(writer: W, config: RdfXmlSerializer) -> Self {
        Self {
            writer,
            config,
            headers_written: false,
            triples: Vec::new(),
        }
    }

    /// Serialize a triple
    pub fn serialize_triple(&mut self, triple: TripleRef<'_>) -> SerializeResult<()> {
        // Collect triples for batch processing
        self.triples.push(triple.into_owned());
        Ok(())
    }

    /// Finish serialization and return the writer
    pub fn finish(mut self) -> SerializeResult<W> {
        self.write_document()?;
        Ok(self.writer)
    }

    /// Write the complete RDF/XML document
    fn write_document(&mut self) -> SerializeResult<()> {
        use std::collections::BTreeMap;

        // Write XML declaration
        if self.config.xml_declaration {
            writeln!(self.writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        }

        // Write RDF root element with namespaces
        write!(self.writer, "<rdf:RDF")?;

        // Add default RDF namespace
        write!(
            self.writer,
            " xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\""
        )?;

        // Add other namespace declarations
        for (prefix, iri) in &self.config.prefixes {
            write!(self.writer, " xmlns:{prefix}=\"{iri}\"")?;
        }

        // Add base IRI if present
        if let Some(base) = &self.config.base_iri {
            write!(self.writer, " xml:base=\"{base}\"")?;
        }

        writeln!(self.writer, ">")?;

        // Group triples by subject - clone to avoid borrow conflicts
        let mut grouped: BTreeMap<String, Vec<(crate::model::Predicate, crate::model::Object)>> =
            BTreeMap::new();
        for triple in &self.triples {
            let subject_key = format!("{}", triple.subject());
            grouped
                .entry(subject_key)
                .or_default()
                .push((triple.predicate().clone(), triple.object().clone()));
        }

        // Serialize each subject as rdf:Description
        let indent = if self.config.pretty { "  " } else { "" };
        let newline = if self.config.pretty { "\n" } else { "" };

        for (subject_idx, (subject_str, properties)) in grouped.iter().enumerate() {
            if subject_idx > 0 && self.config.pretty {
                writeln!(self.writer)?;
            }

            // Start rdf:Description
            write!(self.writer, "{indent}<rdf:Description")?;

            // Add subject as rdf:about or rdf:nodeID
            if let Some(stripped) = subject_str.strip_prefix("_:") {
                write!(self.writer, " rdf:nodeID=\"{}\"", stripped)?;
            } else {
                write!(self.writer, " rdf:about=\"{subject_str}\"")?;
            }

            if properties.is_empty() {
                writeln!(self.writer, "/>")?;
                continue;
            }

            write!(self.writer, ">{newline}")?;

            // Serialize properties
            for (pred, obj) in properties {
                self.write_property(pred, obj)?;
            }

            writeln!(self.writer, "{indent}</rdf:Description>")?;
        }

        // Close RDF root element
        writeln!(self.writer, "</rdf:RDF>")?;

        Ok(())
    }

    /// Write a single property triple
    fn write_property(
        &mut self,
        predicate: &crate::model::Predicate,
        object: &crate::model::Object,
    ) -> SerializeResult<()> {
        let indent = if self.config.pretty { "    " } else { "" };
        let newline = if self.config.pretty { "\n" } else { "" };

        let pred_str = match predicate {
            crate::model::Predicate::NamedNode(nn) => format!("{nn}"),
            crate::model::Predicate::Variable(v) => {
                // Variables in RDF/XML are not standard - use as-is
                format!("{v}")
            }
        };

        // Extract namespace and local name
        let (ns, local) = if let Some((n, l)) = namespaces::extract_namespace(&pred_str) {
            (n, l)
        } else {
            ("", pred_str.as_str())
        };

        // Find or generate prefix
        let prefix = if ns == "http://www.w3.org/1999/02/22-rdf-syntax-ns#" {
            "rdf"
        } else if let Some((p, _)) = self
            .config
            .prefixes
            .iter()
            .find(|(_, iri)| iri.as_str() == ns)
        {
            p.as_str()
        } else {
            "ns"
        };

        write!(self.writer, "{indent}<{prefix}:{local}")?;

        match object {
            crate::model::Object::NamedNode(nn) => {
                // Resource reference
                writeln!(self.writer, " rdf:resource=\"{nn}\"/>{newline}")?;
            }
            crate::model::Object::BlankNode(bn) => {
                // Blank node reference
                writeln!(
                    self.writer,
                    " rdf:nodeID=\"{}\"/>{newline}",
                    bn.as_str().trim_start_matches("_:")
                )?;
            }
            crate::model::Object::Literal(lit) => {
                // Literal value
                if let Some(lang) = lit.language() {
                    write!(self.writer, " xml:lang=\"{lang}\"")?;
                } else {
                    let dt = lit.datatype();
                    if dt.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                        write!(self.writer, " rdf:datatype=\"{dt}\"")?;
                    }
                }
                // Escape XML characters in literal value
                let value = lit
                    .value()
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;");
                writeln!(self.writer, ">{value}</{prefix}:{local}>{newline}")?;
            }
            crate::model::Object::Variable(v) => {
                // Variables in RDF/XML are not standard - serialize as comment
                writeln!(
                    self.writer,
                    "><!-- Variable: {v} --></{prefix}:{local}>{newline}"
                )?;
            }
            crate::model::Object::QuotedTriple(qt) => {
                // RDF-star quoted triples - serialize nested structure
                writeln!(
                    self.writer,
                    "><!-- QuotedTriple: {qt} --></{prefix}:{local}>{newline}"
                )?;
            }
        }

        Ok(())
    }
}

impl<W: Write> QuadSerializer<W> for WriterRdfXmlSerializer<W> {
    fn serialize_quad(&mut self, quad: QuadRef<'_>) -> SerializeResult<()> {
        // RDF/XML only supports default graph, so ignore named graphs
        if quad.graph_name().is_default_graph() {
            self.serialize_triple(quad.triple())
        } else {
            // Could log a warning here about ignoring named graph
            Ok(())
        }
    }

    fn finish(self: Box<Self>) -> SerializeResult<W> {
        (*self).finish()
    }
}

/// RDF/XML namespace utilities
pub mod namespaces {
    use super::*;

    /// Common RDF/XML namespace prefixes
    pub fn common_prefixes() -> HashMap<String, String> {
        let mut prefixes = HashMap::new();
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        prefixes
    }

    /// Extract namespace from IRI
    pub fn extract_namespace(iri: &str) -> Option<(&str, &str)> {
        // Simple namespace extraction
        if let Some(pos) = iri.rfind('#') {
            Some((&iri[..pos + 1], &iri[pos + 1..]))
        } else if let Some(pos) = iri.rfind('/') {
            Some((&iri[..pos + 1], &iri[pos + 1..]))
        } else {
            None
        }
    }

    /// Generate prefix for namespace
    ///
    /// Attempts to extract a meaningful prefix from the namespace URI by:
    /// 1. Looking for common vocabulary prefixes in the URI path
    /// 2. Extracting the last significant path component
    /// 3. Falling back to numbered prefixes (ns1, ns2, etc.)
    pub fn generate_prefix(namespace: &str, existing_prefixes: &HashMap<String, String>) -> String {
        // Try to extract a smart prefix from the namespace URI
        let candidate_prefix = extract_prefix_from_uri(namespace);

        // Check if the candidate prefix is available
        if !existing_prefixes.contains_key(&candidate_prefix)
            && !existing_prefixes.values().any(|v| v == namespace)
        {
            return candidate_prefix;
        }

        // If the candidate is taken, try with a number suffix
        let mut counter = 1;
        loop {
            let prefix = if candidate_prefix == "ns" {
                // Original candidate was generic, use ns1, ns2, etc.
                format!("ns{counter}")
            } else {
                // Append number to the meaningful prefix
                format!("{candidate_prefix}{counter}")
            };

            if !existing_prefixes.contains_key(&prefix) {
                return prefix;
            }
            counter += 1;
        }
    }

    /// Extract a meaningful prefix from a namespace URI
    fn extract_prefix_from_uri(namespace: &str) -> String {
        // Remove trailing # or /
        let trimmed = namespace.trim_end_matches('#').trim_end_matches('/');

        // Try to extract from path segments
        if let Some(last_segment) = trimmed.rsplit('/').next() {
            // Handle version numbers and dates in URIs
            let segment = if last_segment.chars().all(|c| c.is_numeric() || c == '.') {
                // Last segment is a version/date, try the previous one
                trimmed.rsplit('/').nth(1).unwrap_or(last_segment)
            } else {
                last_segment
            };

            // Clean up the segment to make a valid prefix
            let cleaned: String = segment.chars().filter(|c| c.is_alphanumeric()).collect();

            if !cleaned.is_empty() && cleaned.chars().next().unwrap().is_alphabetic() {
                return cleaned.to_lowercase();
            }
        }

        // Fallback to generic prefix
        "ns".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdfxml_parser_creation() {
        let parser = RdfXmlParser::new();
        assert!(parser.base_iri().is_none());
        assert!(parser.prefixes().is_empty());
        assert!(!parser.is_lenient());
    }

    #[test]
    fn test_rdfxml_parser_configuration() {
        let parser = RdfXmlParser::new()
            .with_base_iri("http://example.org/")
            .with_prefix("ex", "http://example.org/ns#")
            .lenient();

        assert_eq!(parser.base_iri(), Some("http://example.org/"));
        assert_eq!(
            parser.prefixes().get("ex"),
            Some(&"http://example.org/ns#".to_string())
        );
        assert!(parser.is_lenient());
    }

    #[test]
    fn test_rdfxml_serializer_creation() {
        let serializer = RdfXmlSerializer::new();
        assert!(serializer.base_iri().is_none());
        assert!(serializer.prefixes().is_empty());
        assert!(!serializer.is_pretty());
        assert!(serializer.has_xml_declaration());
    }

    #[test]
    fn test_rdfxml_serializer_configuration() {
        let serializer = RdfXmlSerializer::new()
            .with_base_iri("http://example.org/")
            .with_prefix("ex", "http://example.org/ns#")
            .pretty()
            .without_xml_declaration();

        assert_eq!(serializer.base_iri(), Some("http://example.org/"));
        assert_eq!(
            serializer.prefixes().get("ex"),
            Some(&"http://example.org/ns#".to_string())
        );
        assert!(serializer.is_pretty());
        assert!(!serializer.has_xml_declaration());
    }

    #[test]
    fn test_rdfxml_basic_validation() {
        let parser = RdfXmlParser::new();

        // Valid XML should pass basic validation
        let result = parser.parse_str("<?xml version=\"1.0\"?><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"></rdf:RDF>");
        assert!(result.is_ok());

        // Non-XML should fail
        let result = parser.parse_str("this is not xml");
        assert!(result.is_err());
    }

    #[test]
    fn test_namespace_extraction() {
        use super::namespaces::extract_namespace;

        assert_eq!(
            extract_namespace("http://example.org/ns#name"),
            Some(("http://example.org/ns#", "name"))
        );

        assert_eq!(
            extract_namespace("http://example.org/path/name"),
            Some(("http://example.org/path/", "name"))
        );

        assert_eq!(extract_namespace("simple"), None);
    }

    #[test]
    fn test_common_prefixes() {
        use super::namespaces::common_prefixes;

        let prefixes = common_prefixes();
        assert!(prefixes.contains_key("rdf"));
        assert!(prefixes.contains_key("rdfs"));
        assert!(prefixes.contains_key("owl"));
        assert!(prefixes.contains_key("xsd"));
    }
}
