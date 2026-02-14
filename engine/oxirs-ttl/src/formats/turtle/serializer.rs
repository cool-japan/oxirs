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
//! let subject = Subject::NamedNode(NamedNode::new("http://example.org/subject").expect("valid IRI"));
//! let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/predicate").expect("valid IRI"));
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

    /// Serialize triples with predicate grouping and object list optimization
    ///
    /// Groups triples by subject and predicate to use semicolon and comma syntax:
    /// - Same subject, different predicates: use `;`
    /// - Same subject and predicate, different objects: use `,`
    ///
    /// # Example
    ///
    /// Input triples:
    /// - `ex:alice ex:name "Alice"`
    /// - `ex:alice ex:age 30`
    /// - `ex:alice ex:knows ex:bob`
    /// - `ex:alice ex:knows ex:charlie`
    ///
    /// Output:
    /// ```turtle
    /// ex:alice ex:name "Alice" ;
    ///          ex:age 30 ;
    ///          ex:knows ex:bob, ex:charlie .
    /// ```
    pub fn serialize_optimized<W: Write>(&self, triples: &[Triple], writer: W) -> TurtleResult<()> {
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

        // Group triples by subject
        let grouped = self.group_triples_by_subject(triples);

        // Serialize each subject group
        for (idx, (subject, predicate_map)) in grouped.iter().enumerate() {
            if idx > 0 {
                formatted_writer
                    .write_newline()
                    .map_err(TurtleParseError::io)?;
            }

            self.serialize_subject_group(subject, predicate_map, &mut formatted_writer)?;
        }

        Ok(())
    }

    /// Group triples by subject, then by predicate
    fn group_triples_by_subject(
        &self,
        triples: &[Triple],
    ) -> Vec<(Subject, HashMap<Predicate, Vec<Object>>)> {
        use std::collections::HashMap;

        // First group by subject
        let mut subject_map: HashMap<String, Vec<&Triple>> = HashMap::new();

        for triple in triples {
            let subject_key = format!("{:?}", triple.subject());
            subject_map.entry(subject_key).or_default().push(triple);
        }

        // Then group by predicate within each subject
        let mut result = Vec::new();

        for triples_group in subject_map.values() {
            if triples_group.is_empty() {
                continue;
            }

            let subject = triples_group[0].subject().clone();
            let mut predicate_map: HashMap<String, Vec<Object>> = HashMap::new();

            for triple in triples_group {
                let predicate_key = format!("{:?}", triple.predicate());
                predicate_map
                    .entry(predicate_key)
                    .or_default()
                    .push(triple.object().clone());
            }

            // Convert predicate_map to HashMap<Predicate, Vec<Object>>
            let mut typed_predicate_map = HashMap::new();
            for triple in triples_group {
                let predicate = triple.predicate().clone();
                let predicate_key = format!("{:?}", &predicate);
                if let Some(objects) = predicate_map.get(&predicate_key) {
                    typed_predicate_map.insert(predicate, objects.clone());
                }
            }

            result.push((subject, typed_predicate_map));
        }

        result
    }

    /// Serialize a group of triples with the same subject
    fn serialize_subject_group<W: Write>(
        &self,
        subject: &Subject,
        predicate_map: &HashMap<Predicate, Vec<Object>>,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        // Serialize subject
        self.serialize_subject(subject, writer)?;
        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize predicate-object pairs
        let mut predicate_iter = predicate_map.iter().peekable();

        while let Some((predicate, objects)) = predicate_iter.next() {
            // Serialize predicate
            self.serialize_predicate(predicate, writer)?;
            writer.write_space().map_err(TurtleParseError::io)?;

            // Serialize objects (comma-separated if multiple)
            for (obj_idx, object) in objects.iter().enumerate() {
                if obj_idx > 0 {
                    writer.write_str(", ").map_err(TurtleParseError::io)?;
                }
                self.serialize_object(object, writer)?;
            }

            // Add semicolon if there are more predicates, otherwise add dot
            if predicate_iter.peek().is_some() {
                writer.write_str(" ;").map_err(TurtleParseError::io)?;
                writer.write_newline().map_err(TurtleParseError::io)?;

                // Indent for next predicate
                if self.config.pretty {
                    writer
                        .write_str("         ")
                        .map_err(TurtleParseError::io)?;
                }
            } else {
                writer.write_str(" .").map_err(TurtleParseError::io)?;
                writer.write_newline().map_err(TurtleParseError::io)?;
            }
        }

        Ok(())
    }

    /// Serialize a subject
    fn serialize_subject<W: Write>(
        &self,
        subject: &Subject,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        match subject {
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
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }
        Ok(())
    }

    /// Serialize a predicate
    fn serialize_predicate<W: Write>(
        &self,
        predicate: &Predicate,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        match predicate {
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
        Ok(())
    }

    /// Serialize an object (with optional blank node property list support)
    fn serialize_object<W: Write>(
        &self,
        object: &Object,
        writer: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        self.serialize_object_with_triples(object, writer, &[])
    }

    /// Serialize an object with access to all triples for blank node optimization
    fn serialize_object_with_triples<W: Write>(
        &self,
        object: &Object,
        writer: &mut FormattedWriter<W>,
        all_triples: &[Triple],
    ) -> TurtleResult<()> {
        match object {
            Object::NamedNode(nn) => {
                let abbrev = writer.abbreviate_iri(nn.as_str());
                writer.write_str(&abbrev).map_err(TurtleParseError::io)?;
            }
            Object::BlankNode(bn) => {
                // Priority 1: Check if this blank node is an RDF collection
                if !all_triples.is_empty() && self.is_collection_head(bn, all_triples) {
                    if let Some(items) = self.extract_collection_items(bn, all_triples) {
                        // Serialize as collection: (item1 item2 item3)
                        self.serialize_collection(&items, writer, all_triples)?;
                        return Ok(());
                    }
                }

                // Priority 2: Check if this blank node can be serialized as a property list
                if !all_triples.is_empty() {
                    let bn_triples = self.find_blank_node_properties(bn, all_triples);
                    if !bn_triples.is_empty() && self.is_blank_node_only_object(bn, all_triples) {
                        // Serialize as property list: [ prop1 val1 ; prop2 val2 ]
                        writer.write_str("[ ").map_err(TurtleParseError::io)?;
                        self.serialize_blank_node_properties(&bn_triples, writer, all_triples)?;
                        writer.write_str(" ]").map_err(TurtleParseError::io)?;
                        return Ok(());
                    }
                }

                // Default: serialize as labeled blank node
                writer
                    .write_str(&format!("_:{}", bn.as_str()))
                    .map_err(TurtleParseError::io)?;
            }
            Object::Literal(literal) => {
                let escaped = writer.escape_string(literal.value());
                writer.write_str(&escaped).map_err(TurtleParseError::io)?;

                if let Some(language) = literal.language() {
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
                writer.write_str("<< ").map_err(TurtleParseError::io)?;
                Self::serialize_quoted_triple(qt, writer)?;
                writer.write_str(" >>").map_err(TurtleParseError::io)?;
            }
        }
        Ok(())
    }

    /// Find all triples where the given blank node is the subject
    fn find_blank_node_properties<'a>(
        &self,
        bn: &oxirs_core::model::BlankNode,
        triples: &'a [Triple],
    ) -> Vec<&'a Triple> {
        triples
            .iter()
            .filter(|t| matches!(t.subject(), Subject::BlankNode(b) if b.as_str() == bn.as_str()))
            .collect()
    }

    /// Check if a blank node only appears as an object (safe to inline)
    fn is_blank_node_only_object(
        &self,
        bn: &oxirs_core::model::BlankNode,
        triples: &[Triple],
    ) -> bool {
        // Count how many times it appears as a subject
        let as_subject_count = triples
            .iter()
            .filter(|t| matches!(t.subject(), Subject::BlankNode(b) if b.as_str() == bn.as_str()))
            .count();

        // Count how many times it appears as an object
        let as_object_count = triples
            .iter()
            .filter(|t| matches!(t.object(), Object::BlankNode(b) if b.as_str() == bn.as_str()))
            .count();

        // It's safe to inline if:
        // 1. It has properties (as_subject_count > 0)
        // 2. It appears as an object exactly once (as_object_count == 1)
        // 3. No circular references
        as_subject_count > 0 && as_object_count == 1
    }

    /// Serialize the properties of a blank node as a property list
    fn serialize_blank_node_properties<W: Write>(
        &self,
        triples: &[&Triple],
        writer: &mut FormattedWriter<W>,
        all_triples: &[Triple],
    ) -> TurtleResult<()> {
        for (idx, triple) in triples.iter().enumerate() {
            if idx > 0 {
                writer.write_str(" ; ").map_err(TurtleParseError::io)?;
            }

            // Serialize predicate
            self.serialize_predicate(triple.predicate(), writer)?;
            writer.write_space().map_err(TurtleParseError::io)?;

            // Serialize object (recursively handle nested blank nodes)
            self.serialize_object_with_triples(triple.object(), writer, all_triples)?;
        }

        Ok(())
    }

    /// Serialize triples with full blank node optimization
    ///
    /// This method enhances `serialize_optimized` with blank node property list support:
    /// - Anonymous blank nodes: `[]`
    /// - Property lists: `[ ex:prop "value" ; ex:other "data" ]`
    /// - Nested blank nodes
    ///
    /// # Example
    ///
    /// Input triples:
    /// - `ex:alice ex:address _:b1`
    /// - `_:b1 ex:city "Wonderland"`
    /// - `_:b1 ex:zip "12345"`
    ///
    /// Output:
    /// ```turtle
    /// ex:alice ex:address [ ex:city "Wonderland" ; ex:zip "12345" ] .
    /// ```
    pub fn serialize_with_blank_node_optimization<W: Write>(
        &self,
        triples: &[Triple],
        writer: W,
    ) -> TurtleResult<()> {
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

        // Filter out triples where the subject is a blank node that can be inlined
        let mut inlineable_bns = std::collections::HashSet::new();
        for triple in triples {
            if let Object::BlankNode(bn) = triple.object() {
                if self.is_blank_node_only_object(bn, triples) {
                    inlineable_bns.insert(bn.as_str());
                }
            }
        }

        // Filter triples: exclude those with inlineable blank node subjects
        let main_triples: Vec<&Triple> = triples
            .iter()
            .filter(|t| {
                if let Subject::BlankNode(bn) = t.subject() {
                    !inlineable_bns.contains(bn.as_str())
                } else {
                    true
                }
            })
            .collect();

        // Group remaining triples by subject
        let grouped = self.group_triples_by_subject_refs(&main_triples);

        // Serialize each subject group
        for (idx, (subject, predicate_map)) in grouped.iter().enumerate() {
            if idx > 0 {
                formatted_writer
                    .write_newline()
                    .map_err(TurtleParseError::io)?;
            }

            self.serialize_subject_group_with_blanks(
                subject,
                predicate_map,
                &mut formatted_writer,
                triples,
            )?;
        }

        Ok(())
    }

    /// Group triples by subject (working with references)
    fn group_triples_by_subject_refs(
        &self,
        triples: &[&Triple],
    ) -> Vec<(Subject, HashMap<Predicate, Vec<Object>>)> {
        let mut subject_map: HashMap<String, Vec<&Triple>> = HashMap::new();

        for triple in triples {
            let subject_key = format!("{:?}", triple.subject());
            subject_map.entry(subject_key).or_default().push(triple);
        }

        let mut result = Vec::new();

        for triples_group in subject_map.values() {
            if triples_group.is_empty() {
                continue;
            }

            let subject = triples_group[0].subject().clone();
            let mut predicate_map: HashMap<String, Vec<Object>> = HashMap::new();

            for triple in triples_group {
                let predicate_key = format!("{:?}", triple.predicate());
                predicate_map
                    .entry(predicate_key)
                    .or_default()
                    .push(triple.object().clone());
            }

            // Convert predicate_map to HashMap<Predicate, Vec<Object>>
            let mut typed_predicate_map = HashMap::new();
            for triple in triples_group {
                let predicate = triple.predicate().clone();
                let predicate_key = format!("{:?}", &predicate);
                if let Some(objects) = predicate_map.get(&predicate_key) {
                    typed_predicate_map.insert(predicate, objects.clone());
                }
            }

            result.push((subject, typed_predicate_map));
        }

        result
    }

    /// Serialize a subject group with blank node inlining support
    fn serialize_subject_group_with_blanks<W: Write>(
        &self,
        subject: &Subject,
        predicate_map: &HashMap<Predicate, Vec<Object>>,
        writer: &mut FormattedWriter<W>,
        all_triples: &[Triple],
    ) -> TurtleResult<()> {
        // Serialize subject
        self.serialize_subject(subject, writer)?;
        writer.write_space().map_err(TurtleParseError::io)?;

        // Serialize predicate-object pairs
        let mut predicate_iter = predicate_map.iter().peekable();

        while let Some((predicate, objects)) = predicate_iter.next() {
            // Serialize predicate
            self.serialize_predicate(predicate, writer)?;
            writer.write_space().map_err(TurtleParseError::io)?;

            // Serialize objects (with blank node optimization)
            for (obj_idx, object) in objects.iter().enumerate() {
                if obj_idx > 0 {
                    writer.write_str(", ").map_err(TurtleParseError::io)?;
                }
                self.serialize_object_with_triples(object, writer, all_triples)?;
            }

            // Add semicolon if there are more predicates, otherwise add dot
            if predicate_iter.peek().is_some() {
                writer.write_str(" ;").map_err(TurtleParseError::io)?;
                writer.write_newline().map_err(TurtleParseError::io)?;

                // Indent for next predicate
                if self.config.pretty {
                    writer
                        .write_str("         ")
                        .map_err(TurtleParseError::io)?;
                }
            } else {
                writer.write_str(" .").map_err(TurtleParseError::io)?;
                writer.write_newline().map_err(TurtleParseError::io)?;
            }
        }

        Ok(())
    }

    /// Check if a blank node is the head of an RDF collection
    ///
    /// A blank node is a collection head if it has:
    /// - Exactly one `rdf:first` predicate
    /// - Exactly one `rdf:rest` predicate
    fn is_collection_head(&self, bn: &oxirs_core::model::BlankNode, triples: &[Triple]) -> bool {
        let rdf_first = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
        let rdf_rest = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";

        let mut has_first = false;
        let mut has_rest = false;

        for triple in triples {
            if let Subject::BlankNode(b) = triple.subject() {
                if b.as_str() == bn.as_str() {
                    if let Predicate::NamedNode(nn) = triple.predicate() {
                        if nn.as_str() == rdf_first {
                            has_first = true;
                        } else if nn.as_str() == rdf_rest {
                            has_rest = true;
                        }
                    }
                }
            }
        }

        has_first && has_rest
    }

    /// Extract all items from an RDF collection starting at the given blank node
    ///
    /// Follows the rdf:rest chain to collect all rdf:first values.
    /// Returns None if the structure is invalid or contains cycles.
    fn extract_collection_items(
        &self,
        bn: &oxirs_core::model::BlankNode,
        triples: &[Triple],
    ) -> Option<Vec<Object>> {
        use std::collections::HashSet;

        let rdf_first = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
        let rdf_rest = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
        let rdf_nil = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

        let mut items = Vec::new();
        let mut current_bn = bn.clone();
        let mut visited = HashSet::new();

        loop {
            // Check for cycles
            if !visited.insert(current_bn.as_str().to_string()) {
                return None; // Cycle detected
            }

            // Limit collection size to prevent infinite loops
            if items.len() > 1000 {
                return None;
            }

            // Find rdf:first and rdf:rest for current node
            let mut first_item = None;
            let mut rest_node = None;

            for triple in triples {
                if let Subject::BlankNode(b) = triple.subject() {
                    if b.as_str() == current_bn.as_str() {
                        if let Predicate::NamedNode(nn) = triple.predicate() {
                            if nn.as_str() == rdf_first {
                                first_item = Some(triple.object().clone());
                            } else if nn.as_str() == rdf_rest {
                                rest_node = Some(triple.object().clone());
                            }
                        }
                    }
                }
            }

            // Must have both rdf:first and rdf:rest
            let first = first_item?;
            let rest = rest_node?;

            items.push(first);

            // Check if we've reached the end (rdf:nil)
            match &rest {
                Object::NamedNode(nn) if nn.as_str() == rdf_nil => {
                    // End of collection
                    break;
                }
                Object::BlankNode(next_bn) => {
                    // Continue to next node
                    current_bn = next_bn.clone();
                }
                _ => {
                    // Invalid collection structure
                    return None;
                }
            }
        }

        Some(items)
    }

    /// Serialize an RDF collection using the compact `(item1 item2)` syntax
    fn serialize_collection<W: Write>(
        &self,
        items: &[Object],
        writer: &mut FormattedWriter<W>,
        all_triples: &[Triple],
    ) -> TurtleResult<()> {
        writer.write_str("(").map_err(TurtleParseError::io)?;

        for (idx, item) in items.iter().enumerate() {
            if idx > 0 {
                writer.write_space().map_err(TurtleParseError::io)?;
            }
            self.serialize_object_with_triples(item, writer, all_triples)?;
        }

        writer.write_str(")").map_err(TurtleParseError::io)?;
        Ok(())
    }
}
