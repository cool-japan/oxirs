//! JSON-LD-star serialization and parsing.
//!
//! This module implements serialization of RDF-star triples to JSON-LD 1.1
//! with embedded triple support, and the corresponding parser.
//!
//! # JSON-LD-star format
//!
//! Quoted (embedded) triples are represented using an `@id` object containing
//! the inner triple's components:
//!
//! ```json
//! {
//!   "@context": { "ex": "http://example.org/" },
//!   "@graph": [
//!     {
//!       "@id": {
//!         "@id": "ex:alice",
//!         "ex:age": [{ "@value": "30" }]
//!       },
//!       "ex:certainty": [{ "@value": "0.9" }]
//!     }
//!   ]
//! }
//! ```
//!
//! Annotated triples (where the asserted triple is also a subject) additionally
//! include an `@asserted` flag.

use serde_json::{json, Map, Value};
use std::collections::HashMap;

use crate::model::{Literal, NamedNode, StarTerm, StarTriple};
use crate::w3c_compliance::AnnotatedTriple;
use crate::{StarError, StarResult};

// ============================================================================
// Constants
// ============================================================================

const JSONLD_CONTEXT: &str = "@context";
const JSONLD_ID: &str = "@id";
const JSONLD_TYPE: &str = "@type";
const JSONLD_VALUE: &str = "@value";
const JSONLD_LANGUAGE: &str = "@language";
const JSONLD_GRAPH: &str = "@graph";

/// Special key used in JSON-LD-star to mark asserted/annotated triples.
const JSONLD_ANNOTATION: &str = "@annotation";

// ============================================================================
// JsonLdStarSerializer
// ============================================================================

/// Serializes RDF-star triples to JSON-LD 1.1 with embedded triple support.
///
/// # Example
///
/// ```
/// use oxirs_star::serialization::JsonLdStarSerializer;
/// use oxirs_star::model::{StarTerm, StarTriple};
///
/// let serializer = JsonLdStarSerializer::new();
/// let triple = StarTriple::new(
///     StarTerm::iri("http://example.org/alice").unwrap(),
///     StarTerm::iri("http://example.org/age").unwrap(),
///     StarTerm::literal("30").unwrap(),
/// );
/// let json = serializer.serialize(&[triple]);
/// assert!(json.is_object());
/// ```
pub struct JsonLdStarSerializer {
    /// Optional base IRI for the JSON-LD document.
    base_iri: Option<String>,
    /// Whether to include a `@context` section (default: true).
    include_context: bool,
    /// Whether to pretty-print (affects JSON formatting, not the Value itself).
    #[allow(dead_code)]
    pretty: bool,
}

impl Default for JsonLdStarSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonLdStarSerializer {
    /// Create a new serializer with default settings.
    pub fn new() -> Self {
        Self {
            base_iri: None,
            include_context: true,
            pretty: false,
        }
    }

    /// Set a base IRI for the JSON-LD document.
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Self {
        self.base_iri = Some(base_iri.into());
        self
    }

    /// Disable the `@context` section.
    pub fn without_context(mut self) -> Self {
        self.include_context = false;
        self
    }

    // ------------------------------------------------------------------
    // serialize
    // ------------------------------------------------------------------

    /// Serialize a slice of [`StarTriple`]s to a JSON-LD-star [`Value`].
    ///
    /// Returns a JSON-LD document with a `@graph` array.
    pub fn serialize(&self, triples: &[StarTriple]) -> Value {
        // Group triples by subject to build JSON-LD objects
        let grouped = group_by_subject(triples);
        let graph: Vec<Value> = grouped
            .into_iter()
            .map(|(subject, preds)| self.build_node_object(&subject, &preds))
            .collect();

        let mut doc = Map::new();

        if self.include_context {
            let mut ctx = Map::new();
            if let Some(ref base) = self.base_iri {
                ctx.insert("@base".to_string(), Value::String(base.clone()));
            }
            doc.insert(JSONLD_CONTEXT.to_string(), Value::Object(ctx));
        }

        doc.insert(JSONLD_GRAPH.to_string(), Value::Array(graph));
        Value::Object(doc)
    }

    // ------------------------------------------------------------------
    // serialize_annotated
    // ------------------------------------------------------------------

    /// Serialize a slice of [`AnnotatedTriple`]s, marking asserted triples with
    /// `@annotation: true` in the JSON-LD output.
    pub fn serialize_annotated(&self, triples: &[AnnotatedTriple]) -> Value {
        let raw: Vec<StarTriple> = triples.iter().map(|at| at.triple.clone()).collect();
        let grouped = group_by_subject(&raw);

        let graph: Vec<Value> = grouped
            .into_iter()
            .map(|(subject, preds)| {
                let mut node = self.build_node_object(&subject, &preds);
                // Mark as annotation if any triple in this node is asserted.
                // Compare using term_to_string (the same format used as map key).
                let is_asserted = triples
                    .iter()
                    .any(|at| at.is_asserted() && term_to_string(&at.triple.subject) == subject);
                if let Value::Object(ref mut obj) = node {
                    obj.insert(JSONLD_ANNOTATION.to_string(), Value::Bool(is_asserted));
                }
                node
            })
            .collect();

        let mut doc = Map::new();
        if self.include_context {
            doc.insert(JSONLD_CONTEXT.to_string(), json!({}));
        }
        doc.insert(JSONLD_GRAPH.to_string(), Value::Array(graph));
        Value::Object(doc)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn build_node_object(&self, subject_key: &str, predicates: &[(StarTerm, StarTerm)]) -> Value {
        let mut obj = Map::new();

        // @id: either a plain IRI string, or an embedded triple object for quoted subjects
        let id_value = if subject_key.starts_with("<<") {
            // Embedded triple — parse back from display form for completeness;
            // in practice subjects are tracked by their original StarTerm
            Value::String(subject_key.to_string())
        } else {
            Value::String(subject_key.to_string())
        };
        obj.insert(JSONLD_ID.to_string(), id_value);

        // Collect predicate-object pairs
        let mut pred_map: HashMap<String, Vec<Value>> = HashMap::new();
        for (predicate, object) in predicates {
            let pred_key = term_to_string(predicate);
            let obj_val = self.term_to_jsonld_value(object);
            pred_map.entry(pred_key).or_default().push(obj_val);
        }

        for (pred_key, values) in pred_map {
            obj.insert(pred_key, Value::Array(values));
        }

        Value::Object(obj)
    }

    fn term_to_jsonld_value(&self, term: &StarTerm) -> Value {
        match term {
            StarTerm::NamedNode(nn) => {
                json!({ JSONLD_ID: nn.iri })
            }
            StarTerm::BlankNode(bn) => {
                json!({ JSONLD_ID: format!("_:{}", bn.id) })
            }
            StarTerm::Literal(lit) => {
                let mut obj = Map::new();
                obj.insert(JSONLD_VALUE.to_string(), Value::String(lit.value.clone()));
                if let Some(ref lang) = lit.language {
                    obj.insert(JSONLD_LANGUAGE.to_string(), Value::String(lang.clone()));
                }
                if let Some(ref dt) = lit.datatype {
                    obj.insert(JSONLD_TYPE.to_string(), Value::String(dt.iri.clone()));
                }
                Value::Object(obj)
            }
            StarTerm::Variable(v) => {
                // Variables are serialized as a special string for diagnostics
                json!({ "@variable": v.name })
            }
            StarTerm::QuotedTriple(qt) => {
                // Embedded triple as @id object
                let mut inner = Map::new();
                inner.insert(
                    JSONLD_ID.to_string(),
                    Value::String(term_to_string(&qt.subject)),
                );
                let pred_key = term_to_string(&qt.predicate);
                let obj_val = self.term_to_jsonld_value(&qt.object);
                inner.insert(pred_key, Value::Array(vec![obj_val]));
                json!({ JSONLD_ID: Value::Object(inner) })
            }
        }
    }
}

// ============================================================================
// JsonLdStarParser
// ============================================================================

/// Parses JSON-LD-star documents back into RDF-star triples.
///
/// # Example
///
/// ```
/// use oxirs_star::serialization::JsonLdStarParser;
/// use serde_json::json;
///
/// let parser = JsonLdStarParser::new();
/// let doc = json!({
///     "@graph": [
///         {
///             "@id": "http://example.org/alice",
///             "http://example.org/age": [{ "@value": "30" }]
///         }
///     ]
/// });
/// let triples = parser.parse(&doc).unwrap();
/// assert_eq!(triples.len(), 1);
/// ```
pub struct JsonLdStarParser {
    /// Counter for generating blank node identifiers.
    blank_counter: std::sync::atomic::AtomicUsize,
}

impl Default for JsonLdStarParser {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonLdStarParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self {
            blank_counter: std::sync::atomic::AtomicUsize::new(1),
        }
    }

    /// Parse a JSON-LD-star document into a list of [`StarTriple`]s.
    pub fn parse(&self, json: &Value) -> StarResult<Vec<StarTriple>> {
        let mut triples = Vec::new();

        match json {
            Value::Object(obj) => {
                // Handle @graph array
                if let Some(Value::Array(graph)) = obj.get(JSONLD_GRAPH) {
                    for node in graph {
                        self.parse_node(node, &mut triples)?;
                    }
                } else {
                    // Top-level object treated as single node
                    self.parse_node(json, &mut triples)?;
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    self.parse_node(item, &mut triples)?;
                }
            }
            _ => {
                return Err(StarError::parse_error(
                    "JSON-LD-star document must be an object or array",
                ));
            }
        }

        Ok(triples)
    }

    fn parse_node(&self, node: &Value, triples: &mut Vec<StarTriple>) -> StarResult<()> {
        let obj = match node {
            Value::Object(o) => o,
            _ => return Ok(()),
        };

        // Extract subject
        let subject = self.extract_subject(obj)?;

        // Iterate over properties (skip JSON-LD keywords)
        for (key, value) in obj {
            if key.starts_with('@') {
                continue;
            }

            let predicate = StarTerm::iri(key)?;

            match value {
                Value::Array(values) => {
                    for v in values {
                        let object = self.parse_value(v)?;
                        triples.push(StarTriple::new(subject.clone(), predicate.clone(), object));
                    }
                }
                _ => {
                    let object = self.parse_value(value)?;
                    triples.push(StarTriple::new(subject.clone(), predicate.clone(), object));
                }
            }
        }

        Ok(())
    }

    fn extract_subject(&self, obj: &Map<String, Value>) -> StarResult<StarTerm> {
        match obj.get(JSONLD_ID) {
            Some(Value::String(iri)) => {
                if let Some(stripped) = iri.strip_prefix("_:") {
                    StarTerm::blank_node(stripped)
                } else {
                    StarTerm::iri(iri)
                }
            }
            Some(Value::Object(inner)) => {
                // Embedded triple as @id object
                self.parse_embedded_triple(inner)
            }
            Some(_) => Err(StarError::parse_error(
                "@id must be a string or embedded triple object",
            )),
            None => {
                // Generate blank node
                let id = self
                    .blank_counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                StarTerm::blank_node(&format!("b{id}"))
            }
        }
    }

    fn parse_embedded_triple(&self, inner: &Map<String, Value>) -> StarResult<StarTerm> {
        // Extract the inner subject (@id field)
        let inner_subject = match inner.get(JSONLD_ID) {
            Some(Value::String(iri)) => {
                if let Some(stripped) = iri.strip_prefix("_:") {
                    StarTerm::blank_node(stripped)?
                } else {
                    StarTerm::iri(iri)?
                }
            }
            _ => {
                return Err(StarError::parse_error(
                    "Embedded triple @id must have an @id string field",
                ));
            }
        };

        // Find the single non-keyword property as predicate
        let mut pred_iri: Option<&str> = None;
        let mut obj_val: Option<StarTerm> = None;

        for (key, value) in inner {
            if key.starts_with('@') {
                continue;
            }
            pred_iri = Some(key.as_str());
            // Value should be an array with one element
            let raw_val = match value {
                Value::Array(arr) if !arr.is_empty() => &arr[0],
                other => other,
            };
            obj_val = Some(self.parse_value(raw_val)?);
            break; // take first non-keyword property
        }

        let predicate = match pred_iri {
            Some(p) => StarTerm::iri(p)?,
            None => {
                return Err(StarError::parse_error(
                    "Embedded triple must have at least one predicate property",
                ));
            }
        };
        let object = match obj_val {
            Some(o) => o,
            None => {
                return Err(StarError::parse_error(
                    "Embedded triple is missing object value",
                ));
            }
        };

        let embedded = StarTriple::new(inner_subject, predicate, object);
        Ok(StarTerm::quoted_triple(embedded))
    }

    fn parse_value(&self, value: &Value) -> StarResult<StarTerm> {
        match value {
            Value::Object(obj) => {
                // Embedded triple (@id is an object)
                if let Some(Value::Object(inner_id)) = obj.get(JSONLD_ID) {
                    return self.parse_embedded_triple(inner_id);
                }

                // Named node: { "@id": "..." }
                if let Some(Value::String(iri)) = obj.get(JSONLD_ID) {
                    if let Some(stripped) = iri.strip_prefix("_:") {
                        return StarTerm::blank_node(stripped);
                    }
                    return StarTerm::iri(iri);
                }

                // Literal: { "@value": "...", "@language"?: "...", "@type"?: "..." }
                if let Some(Value::String(val)) = obj.get(JSONLD_VALUE) {
                    let language = obj
                        .get(JSONLD_LANGUAGE)
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    let datatype =
                        obj.get(JSONLD_TYPE)
                            .and_then(|v| v.as_str())
                            .map(|dt| NamedNode {
                                iri: dt.to_string(),
                            });

                    return Ok(StarTerm::Literal(Literal {
                        value: val.clone(),
                        language,
                        datatype,
                    }));
                }

                Err(StarError::parse_error(
                    "Cannot determine term type from JSON-LD object",
                ))
            }
            Value::String(s) => StarTerm::iri(s),
            Value::Bool(b) => {
                let value = if *b { "true" } else { "false" };
                Ok(StarTerm::Literal(Literal {
                    value: value.to_string(),
                    language: None,
                    datatype: Some(NamedNode {
                        iri: "http://www.w3.org/2001/XMLSchema#boolean".to_string(),
                    }),
                }))
            }
            Value::Number(n) => Ok(StarTerm::Literal(Literal {
                value: n.to_string(),
                language: None,
                datatype: Some(NamedNode {
                    iri: "http://www.w3.org/2001/XMLSchema#decimal".to_string(),
                }),
            })),
            Value::Null => Err(StarError::parse_error(
                "null values are not valid RDF terms",
            )),
            Value::Array(_) => Err(StarError::parse_error(
                "Unexpected array in value position; use outer array for multiple values",
            )),
        }
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Convert a [`StarTerm`] to its string representation for use as a map key.
fn term_to_string(term: &StarTerm) -> String {
    match term {
        StarTerm::NamedNode(nn) => nn.iri.clone(),
        StarTerm::BlankNode(bn) => format!("_:{}", bn.id),
        StarTerm::Literal(lit) => {
            let mut s = format!("\"{}\"", lit.value);
            if let Some(ref lang) = lit.language {
                s.push('@');
                s.push_str(lang);
            }
            if let Some(ref dt) = lit.datatype {
                s.push_str("^^<");
                s.push_str(&dt.iri);
                s.push('>');
            }
            s
        }
        StarTerm::Variable(v) => format!("?{}", v.name),
        StarTerm::QuotedTriple(qt) => format!(
            "<<{} {} {}>>",
            term_to_string(&qt.subject),
            term_to_string(&qt.predicate),
            term_to_string(&qt.object)
        ),
    }
}

/// Group triples by their subject's string representation.
fn group_by_subject(triples: &[StarTriple]) -> Vec<(String, Vec<(StarTerm, StarTerm)>)> {
    let mut map: Vec<(String, Vec<(StarTerm, StarTerm)>)> = Vec::new();

    for triple in triples {
        let key = term_to_string(&triple.subject);
        if let Some(entry) = map.iter_mut().find(|(k, _)| k == &key) {
            entry
                .1
                .push((triple.predicate.clone(), triple.object.clone()));
        } else {
            map.push((key, vec![(triple.predicate.clone(), triple.object.clone())]));
        }
    }

    map
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTriple;
    use crate::w3c_compliance::{AnnotatedTriple, AssertionStatus};
    use serde_json::json;

    fn iri(s: &str) -> StarTerm {
        StarTerm::iri(s).expect("iri")
    }
    fn lit(s: &str) -> StarTerm {
        StarTerm::literal(s).expect("lit")
    }

    fn simple_triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(iri(s), iri(p), iri(o))
    }

    // -------------------------------------------------------------------
    // JsonLdStarSerializer::serialize
    // -------------------------------------------------------------------

    #[test]
    fn test_serialize_empty_graph() {
        let ser = JsonLdStarSerializer::new();
        let doc = ser.serialize(&[]);
        assert!(doc.is_object());
        let graph = doc.get("@graph").expect("@graph key");
        assert_eq!(graph.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_serialize_single_triple() {
        let ser = JsonLdStarSerializer::new();
        let triple = simple_triple("http://alice", "http://age", "http://v");
        let doc = ser.serialize(&[triple]);
        let graph = doc["@graph"].as_array().unwrap();
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_serialize_includes_context() {
        let ser = JsonLdStarSerializer::new();
        let doc = ser.serialize(&[simple_triple("http://a", "http://p", "http://b")]);
        assert!(doc.get("@context").is_some());
    }

    #[test]
    fn test_serialize_without_context() {
        let ser = JsonLdStarSerializer::new().without_context();
        let doc = ser.serialize(&[simple_triple("http://a", "http://p", "http://b")]);
        assert!(doc.get("@context").is_none());
    }

    #[test]
    fn test_serialize_with_literal_object() {
        let ser = JsonLdStarSerializer::new();
        let triple = StarTriple::new(iri("http://alice"), iri("http://age"), lit("30"));
        let doc = ser.serialize(&[triple]);
        let graph = &doc["@graph"].as_array().unwrap()[0];
        // The age predicate should be present
        assert!(graph.get("http://age").is_some());
        let age_vals = graph["http://age"].as_array().unwrap();
        assert_eq!(age_vals[0]["@value"], json!("30"));
    }

    #[test]
    fn test_serialize_named_node_object() {
        let ser = JsonLdStarSerializer::new();
        let triple = StarTriple::new(iri("http://alice"), iri("http://knows"), iri("http://bob"));
        let doc = ser.serialize(&[triple]);
        let graph = &doc["@graph"].as_array().unwrap()[0];
        let knows_vals = graph["http://knows"].as_array().unwrap();
        assert_eq!(knows_vals[0]["@id"], json!("http://bob"));
    }

    #[test]
    fn test_serialize_blank_node_object() {
        let ser = JsonLdStarSerializer::new();
        let triple = StarTriple::new(
            iri("http://alice"),
            iri("http://hasAddress"),
            StarTerm::blank_node("addr1").unwrap(),
        );
        let doc = ser.serialize(&[triple]);
        let graph = &doc["@graph"].as_array().unwrap()[0];
        let addr_vals = graph["http://hasAddress"].as_array().unwrap();
        assert_eq!(addr_vals[0]["@id"], json!("_:addr1"));
    }

    #[test]
    fn test_serialize_quoted_triple_as_subject() {
        let ser = JsonLdStarSerializer::new();
        let inner = StarTriple::new(iri("http://alice"), iri("http://age"), lit("30"));
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://certainty"),
            lit("high"),
        );
        let doc = ser.serialize(&[outer]);
        let graph = doc["@graph"].as_array().unwrap();
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_serialize_multiple_triples_same_subject_merged() {
        let ser = JsonLdStarSerializer::new();
        let t1 = StarTriple::new(iri("http://alice"), iri("http://age"), lit("30"));
        let t2 = StarTriple::new(iri("http://alice"), iri("http://name"), lit("Alice"));
        let doc = ser.serialize(&[t1, t2]);
        let graph = doc["@graph"].as_array().unwrap();
        // Both triples share same subject → merged into one node
        assert_eq!(graph.len(), 1);
        let node = &graph[0];
        assert!(node.get("http://age").is_some());
        assert!(node.get("http://name").is_some());
    }

    #[test]
    fn test_serialize_language_tagged_literal() {
        let ser = JsonLdStarSerializer::new();
        let triple = StarTriple::new(
            iri("http://alice"),
            iri("http://name"),
            StarTerm::literal_with_language("Alice", "en").unwrap(),
        );
        let doc = ser.serialize(&[triple]);
        let graph = &doc["@graph"].as_array().unwrap()[0];
        let name_vals = graph["http://name"].as_array().unwrap();
        assert_eq!(name_vals[0]["@language"], json!("en"));
    }

    #[test]
    fn test_serialize_datatyped_literal() {
        let ser = JsonLdStarSerializer::new();
        let triple = StarTriple::new(
            iri("http://alice"),
            iri("http://age"),
            StarTerm::literal_with_datatype("30", "http://www.w3.org/2001/XMLSchema#integer")
                .unwrap(),
        );
        let doc = ser.serialize(&[triple]);
        let graph = &doc["@graph"].as_array().unwrap()[0];
        let age_vals = graph["http://age"].as_array().unwrap();
        assert_eq!(
            age_vals[0]["@type"],
            json!("http://www.w3.org/2001/XMLSchema#integer")
        );
    }

    // -------------------------------------------------------------------
    // JsonLdStarSerializer::serialize_annotated
    // -------------------------------------------------------------------

    #[test]
    fn test_serialize_annotated_includes_annotation_field() {
        let ser = JsonLdStarSerializer::new();
        let triple = simple_triple("http://alice", "http://age", "http://v");
        let annotated = AnnotatedTriple {
            triple,
            status: AssertionStatus::Asserted,
        };
        let doc = ser.serialize_annotated(&[annotated]);
        let graph = doc["@graph"].as_array().unwrap();
        assert_eq!(graph.len(), 1);
        assert!(graph[0].get("@annotation").is_some());
    }

    #[test]
    fn test_serialize_annotated_asserted_true() {
        let ser = JsonLdStarSerializer::new();
        let triple = simple_triple("http://alice", "http://p", "http://o");
        let annotated = AnnotatedTriple {
            triple,
            status: AssertionStatus::Asserted,
        };
        let doc = ser.serialize_annotated(&[annotated]);
        let node = &doc["@graph"].as_array().unwrap()[0];
        assert_eq!(node["@annotation"], json!(true));
    }

    #[test]
    fn test_serialize_annotated_unasserted_false() {
        let ser = JsonLdStarSerializer::new();
        let triple = simple_triple("http://alice", "http://p", "http://o");
        let annotated = AnnotatedTriple {
            triple,
            status: AssertionStatus::Unasserted,
        };
        let doc = ser.serialize_annotated(&[annotated]);
        let node = &doc["@graph"].as_array().unwrap()[0];
        assert_eq!(node["@annotation"], json!(false));
    }

    // -------------------------------------------------------------------
    // JsonLdStarParser::parse
    // -------------------------------------------------------------------

    #[test]
    fn test_parse_empty_graph() {
        let parser = JsonLdStarParser::new();
        let doc = json!({ "@graph": [] });
        let triples = parser.parse(&doc).unwrap();
        assert!(triples.is_empty());
    }

    #[test]
    fn test_parse_simple_triple() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/alice",
                "http://example.org/age": [{ "@value": "30" }]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].subject, iri("http://example.org/alice"));
    }

    #[test]
    fn test_parse_named_node_object() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/alice",
                "http://example.org/knows": [{ "@id": "http://example.org/bob" }]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].object, iri("http://example.org/bob"));
    }

    #[test]
    fn test_parse_literal_with_language() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/alice",
                "http://example.org/name": [{ "@value": "Alice", "@language": "en" }]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        if let StarTerm::Literal(lit) = &triples[0].object {
            assert_eq!(lit.language, Some("en".to_string()));
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_parse_literal_with_datatype() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/alice",
                "http://example.org/age": [{
                    "@value": "30",
                    "@type": "http://www.w3.org/2001/XMLSchema#integer"
                }]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        if let StarTerm::Literal(lit) = &triples[0].object {
            assert!(lit.datatype.is_some());
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_parse_multiple_predicates() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/alice",
                "http://example.org/age": [{ "@value": "30" }],
                "http://example.org/name": [{ "@value": "Alice" }]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_parse_multiple_values_for_predicate() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/alice",
                "http://example.org/knows": [
                    { "@id": "http://example.org/bob" },
                    { "@id": "http://example.org/charlie" }
                ]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_parse_blank_node_subject() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "_:b1",
                "http://example.org/p": [{ "@value": "v" }]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(triples[0].subject.is_blank_node());
    }

    #[test]
    fn test_parse_auto_blank_node_no_id() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "http://example.org/p": [{ "@value": "v" }]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(triples[0].subject.is_blank_node());
    }

    #[test]
    fn test_parse_null_returns_error() {
        let parser = JsonLdStarParser::new();
        assert!(parser.parse(&Value::Null).is_err());
    }

    #[test]
    fn test_parse_number_value() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/x",
                "http://example.org/count": [42]
            }]
        });
        // Numbers are parsed as xsd:decimal literals
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(triples[0].object.is_literal());
    }

    #[test]
    fn test_parse_boolean_value() {
        let parser = JsonLdStarParser::new();
        let doc = json!({
            "@graph": [{
                "@id": "http://example.org/x",
                "http://example.org/active": [true]
            }]
        });
        let triples = parser.parse(&doc).unwrap();
        assert_eq!(triples.len(), 1);
        if let StarTerm::Literal(lit) = &triples[0].object {
            assert_eq!(lit.value, "true");
        } else {
            panic!("Expected literal");
        }
    }
}
