//! Turtle serializer implementation
//!
//! Provides serialization of RDF triples to Turtle format with support for:
//! - Prefix declarations
//! - Base IRI declarations
//! - Abbreviated syntax (a for rdf:type)
//! - Auto-generation of prefixes from triple data
//! - RDF 1.2 features (quoted triples, directional language tags)
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::formats::turtle::TurtleSerializer;
//! use oxirs_ttl::toolkit::Serializer;
//! use oxirs_core::model::{Triple, Subject, Predicate, Object, NamedNode, Literal};
//!
//! let serializer = TurtleSerializer::new();
//! let subject = Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap());
//! let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/predicate").unwrap());
//! let object = Object::Literal(Literal::new_simple_literal("object"));
//! let triple = Triple::new(subject, predicate, object);
//!
//! let mut output = Vec::new();
//! serializer.serialize(&[triple], &mut output).unwrap();
//! ```

use crate::error::{TurtleParseError, TurtleResult};
use crate::toolkit::{FormattedWriter, SerializationConfig, Serializer};
#[cfg(feature = "rdf-12")]
#[allow(unused_imports)]
use oxirs_core::model::literal::BaseDirection;
use oxirs_core::model::{Object, Predicate, QuotedTriple, Subject, Triple};
use std::collections::HashMap;
use std::io::Write;

/// Turtle serializer for converting RDF triples to Turtle format
///
/// The serializer supports various configuration options including:
/// - Custom prefix declarations
/// - Base IRI settings
/// - Auto-generation of prefixes from triple data
/// - Pretty printing options
#[derive(Debug, Clone)]
pub struct TurtleSerializer {
    config: SerializationConfig,
}

impl Default for TurtleSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl TurtleSerializer {
    /// Create a new Turtle serializer
    pub fn new() -> Self {
        Self {
            config: SerializationConfig::default(),
        }
    }

    /// Create a Turtle serializer with custom configuration
    pub fn with_config(config: SerializationConfig) -> Self {
        Self { config }
    }

    /// Create a Turtle serializer with auto-generated prefixes from the triples
    pub fn with_auto_prefixes(triples: &[Triple]) -> Self {
        let prefixes = Self::auto_generate_prefixes(triples);
        let config = SerializationConfig::default().with_use_prefixes(true);

        let mut config_with_prefixes = config;
        config_with_prefixes.prefixes = prefixes;

        Self {
            config: config_with_prefixes,
        }
    }

    /// Auto-detect and generate common prefixes from a set of triples
    pub fn auto_generate_prefixes(triples: &[Triple]) -> HashMap<String, String> {
        let mut iri_counts: HashMap<String, usize> = HashMap::new();

        // Count IRI namespace occurrences
        for triple in triples {
            // Count subject namespace
            if let Subject::NamedNode(nn) = triple.subject() {
                if let Some(namespace) = Self::extract_namespace(nn.as_str()) {
                    *iri_counts.entry(namespace).or_insert(0) += 1;
                }
            }

            // Count predicate namespace
            if let Predicate::NamedNode(nn) = triple.predicate() {
                if let Some(namespace) = Self::extract_namespace(nn.as_str()) {
                    *iri_counts.entry(namespace).or_insert(0) += 1;
                }
            }

            // Count object namespace
            if let Object::NamedNode(nn) = triple.object() {
                if let Some(namespace) = Self::extract_namespace(nn.as_str()) {
                    *iri_counts.entry(namespace).or_insert(0) += 1;
                }
            }
        }

        // Generate prefixes for namespaces used more than once
        let mut prefixes = HashMap::new();
        let mut prefix_counter = 1;

        for (namespace, count) in iri_counts {
            if count > 1 {
                // Try to generate a meaningful prefix from the namespace
                let prefix = Self::suggest_prefix(&namespace, prefix_counter);
                prefixes.insert(prefix, namespace);
                prefix_counter += 1;
            }
        }

        // Add common well-known prefixes if they're used
        Self::add_well_known_prefixes(&mut prefixes, triples);

        prefixes
    }

    /// Extract namespace from an IRI (everything up to the last # or /)
    fn extract_namespace(iri: &str) -> Option<String> {
        // Find the last occurrence of # or /
        let last_separator = iri.rfind(['#', '/'])?;
        Some(iri[..=last_separator].to_string())
    }

    /// Suggest a prefix name based on the namespace
    fn suggest_prefix(namespace: &str, counter: usize) -> String {
        // Try to extract a meaningful part from the namespace
        if namespace.contains("example.org") {
            return "ex".to_string();
        } else if namespace.contains("w3.org/1999/02/22-rdf-syntax-ns#") {
            return "rdf".to_string();
        } else if namespace.contains("w3.org/2000/01/rdf-schema#") {
            return "rdfs".to_string();
        } else if namespace.contains("w3.org/2002/07/owl#") {
            return "owl".to_string();
        } else if namespace.contains("xmlns.com/foaf") {
            return "foaf".to_string();
        } else if namespace.contains("purl.org/dc") {
            return "dc".to_string();
        } else if namespace.contains("schema.org") {
            return "schema".to_string();
        }

        // Generic prefix
        format!("ns{counter}")
    }

    /// Add well-known prefixes if they're actually used in the triples
    fn add_well_known_prefixes(prefixes: &mut HashMap<String, String>, triples: &[Triple]) {
        let well_known = [
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
            ("xsd", "http://www.w3.org/2001/XMLSchema#"),
            ("owl", "http://www.w3.org/2002/07/owl#"),
        ];

        for (prefix, iri) in &well_known {
            // Check if this namespace is used
            let used = triples.iter().any(|t| Self::triple_uses_namespace(t, iri));

            if used && !prefixes.values().any(|v| v == iri) {
                prefixes.insert(prefix.to_string(), iri.to_string());
            }
        }
    }

    /// Check if a triple uses a specific namespace
    fn triple_uses_namespace(triple: &Triple, namespace: &str) -> bool {
        // Check subject
        if let Subject::NamedNode(nn) = triple.subject() {
            if nn.as_str().starts_with(namespace) {
                return true;
            }
        }

        // Check predicate
        if let Predicate::NamedNode(nn) = triple.predicate() {
            if nn.as_str().starts_with(namespace) {
                return true;
            }
        }

        // Check object
        if let Object::NamedNode(nn) = triple.object() {
            if nn.as_str().starts_with(namespace) {
                return true;
            }
        }

        false
    }
}

impl Serializer<Triple> for TurtleSerializer {
    fn serialize<W: Write>(&self, triples: &[Triple], writer: W) -> TurtleResult<()> {
        let mut formatted_writer = FormattedWriter::new(writer, self.config.clone());

        // Write prefix declarations
        for (prefix, iri) in &self.config.prefixes {
            formatted_writer
                .write_str(&format!("@prefix {prefix}: <{iri}> ."))
                .map_err(TurtleParseError::io)?;
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        // Write base declaration if present
        if let Some(ref base) = self.config.base_iri {
            formatted_writer
                .write_str(&format!("@base <{base}> ."))
                .map_err(TurtleParseError::io)?;
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        if !self.config.prefixes.is_empty() || self.config.base_iri.is_some() {
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        // Write triples
        for triple in triples {
            self.serialize_item_formatted(triple, &mut formatted_writer)?;
            formatted_writer
                .write_str(" .")
                .map_err(TurtleParseError::io)?;
            formatted_writer
                .write_newline()
                .map_err(TurtleParseError::io)?;
        }

        Ok(())
    }

    fn serialize_item<W: Write>(&self, triple: &Triple, writer: W) -> TurtleResult<()> {
        let mut formatted_writer = FormattedWriter::new(writer, self.config.clone());
        self.serialize_item_formatted(triple, &mut formatted_writer)
    }
}

impl TurtleSerializer {
    fn serialize_item_formatted<W: Write>(
        &self,
        triple: &Triple,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        // Serialize subject
        match triple.subject() {
            Subject::NamedNode(nn) => {
                let abbrev = writer.abbreviate_iri(nn.as_str());
                writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
            }
            Subject::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::QuotedTriple(qt) => {
                // RDF 1.2: << s p o >> syntax
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }

        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize predicate (check for rdf:type abbreviation)
        match triple.predicate() {
            Predicate::NamedNode(nn) => {
                if nn.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    writer.write_str("a").map_err(TurtleParseError::io)?;
                } else {
                    let abbrev = writer.abbreviate_iri(nn.as_str());
                    writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
                }
            }
            Predicate::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
        }

        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize object
        match triple.object() {
            Object::NamedNode(nn) => {
                let abbrev = writer.abbreviate_iri(nn.as_str());
                writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
            }
            Object::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::Literal(literal) => {
                let escaped = writer.escape_string(literal.value());
                writer.write_str(&escaped).map_err(TurtleParseError::io)?;

                if let Some(language) = literal.language() {
                    // Check for directional language tag (RDF 1.2)
                    #[cfg(feature = "rdf-12")]
                    if let Some(direction) = literal.direction() {
                        writer
                            .write_str(&format!("@{language}--{direction}"))
                            .map_err(TurtleParseError::io)?;
                    } else {
                        writer
                            .write_str(&format!("@{language}"))
                            .map_err(TurtleParseError::io)?;
                    }

                    #[cfg(not(feature = "rdf-12"))]
                    writer
                        .write_str(&format!("@{language}"))
                        .map_err(TurtleParseError::io)?;
                } else if literal.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    let datatype_abbrev = writer.abbreviate_iri(literal.datatype().as_str());
                    writer
                        .write_str(&format!("^^{datatype_abbrev}"))
                        .map_err(TurtleParseError::io)?;
                }
            }
            Object::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::QuotedTriple(qt) => {
                // RDF 1.2: << s p o >> syntax
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }

        Ok(())
    }

    /// Helper method to serialize a quoted triple (RDF 1.2 / RDF-star)
    fn serialize_quoted_triple<W: Write>(
        qt: &QuotedTriple,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        // Serialize inner subject
        match qt.subject() {
            Subject::NamedNode(nn) => {
                let abbrev = writer.abbreviate_iri(nn.as_str());
                writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
            }
            Subject::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Subject::QuotedTriple(inner_qt) => {
                // Nested quoted triple
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(inner_qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }

        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize inner predicate
        match qt.predicate() {
            Predicate::NamedNode(nn) => {
                if nn.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
                    writer.write_str("a").map_err(TurtleParseError::io)?;
                } else {
                    let abbrev = writer.abbreviate_iri(nn.as_str());
                    writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
                }
            }
            Predicate::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
        }

        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize inner object
        match qt.object() {
            Object::NamedNode(nn) => {
                let abbrev = writer.abbreviate_iri(nn.as_str());
                writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
            }
            Object::BlankNode(bn) => {
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::Literal(literal) => {
                let escaped = writer.escape_string(literal.value());
                writer.write_str(&escaped).map_err(TurtleParseError::io)?;

                if let Some(language) = literal.language() {
                    // Check for directional language tag (RDF 1.2)
                    #[cfg(feature = "rdf-12")]
                    if let Some(direction) = literal.direction() {
                        writer
                            .write_str(&format!("@{language}--{direction}"))
                            .map_err(TurtleParseError::io)?;
                    } else {
                        writer
                            .write_str(&format!("@{language}"))
                            .map_err(TurtleParseError::io)?;
                    }

                    #[cfg(not(feature = "rdf-12"))]
                    writer
                        .write_str(&format!("@{language}"))
                        .map_err(TurtleParseError::io)?;
                } else if literal.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    let datatype_abbrev = writer.abbreviate_iri(literal.datatype().as_str());
                    writer
                        .write_str(&format!("^^{datatype_abbrev}"))
                        .map_err(TurtleParseError::io)?;
                }
            }
            Object::Variable(var) => {
                writer
                    .write_str(&format!("?{}", var.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::QuotedTriple(inner_qt) => {
                // Nested quoted triple
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(inner_qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }

        Ok(())
    }
}
