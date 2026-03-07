//! JSON-LD 1.1 serialization, deserialization, and processing algorithms.
//!
//! Implements the W3C JSON-LD 1.1 specification:
//! <https://www.w3.org/TR/json-ld11/>
//!
//! Supports:
//! - Expansion (CURIE/prefix resolution to full IRIs)
//! - Compaction (IRI shortening with a context)
//! - Flattening (nested → flat @graph)
//! - Framing (reshape output to match a frame)
//! - RDF serialization/deserialization (JSON-LD ↔ N-Quads)
//! - Streaming writer with compact/pretty options

use serde_json::{json, Map, Value};
use std::collections::HashMap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during JSON-LD processing.
#[derive(Debug, Error)]
pub enum JsonLdError {
    /// A context entry is invalid.
    #[error("Invalid context: {0}")]
    InvalidContext(String),

    /// An IRI is syntactically invalid.
    #[error("Invalid IRI: {0}")]
    InvalidIri(String),

    /// Input document structure is unexpected.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// A required key is missing from the document.
    #[error("Missing key: {0}")]
    MissingKey(String),

    /// JSON serialization/deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Frame matching failed.
    #[error("Framing error: {0}")]
    Framing(String),
}

/// Convenience result alias.
pub type JsonLdResult<T> = Result<T, JsonLdError>;

// ─────────────────────────────────────────────────────────────────────────────
// Core data types
// ─────────────────────────────────────────────────────────────────────────────

/// A term in the JSON-LD internal representation (subject / predicate / object
/// / graph name of a quad).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonLdTerm {
    /// A full IRI, e.g. `"http://schema.org/name"`.
    Iri(String),
    /// A blank node, e.g. `"_:b0"`.
    BlankNode(String),
    /// An RDF literal value.
    Literal {
        /// The lexical string value.
        value: String,
        /// The datatype IRI (e.g. `xsd:string`).
        datatype: String,
        /// Optional BCP-47 language tag (only valid with `rdf:langString`).
        language: Option<String>,
    },
}

impl JsonLdTerm {
    /// Returns `true` when the term is an IRI node.
    pub fn is_iri(&self) -> bool {
        matches!(self, Self::Iri(_))
    }

    /// Returns `true` when the term is a blank node.
    pub fn is_blank_node(&self) -> bool {
        matches!(self, Self::BlankNode(_))
    }

    /// Returns `true` when the term is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal { .. })
    }

    /// Serialise to the N-Quads text representation.
    pub fn to_nquads_string(&self) -> String {
        match self {
            Self::Iri(iri) => format!("<{}>", iri),
            Self::BlankNode(id) => id.clone(),
            Self::Literal {
                value,
                datatype,
                language,
            } => {
                let escaped = value
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                if let Some(lang) = language {
                    format!("\"{}\"@{}", escaped, lang)
                } else {
                    format!("\"{}\"^^<{}>", escaped, datatype)
                }
            }
        }
    }
}

/// An RDF quad in the JSON-LD internal representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonLdQuad {
    /// Subject node (IRI or blank node).
    pub subject: JsonLdTerm,
    /// Predicate (always IRI).
    pub predicate: JsonLdTerm,
    /// Object (IRI, blank node, or literal).
    pub object: JsonLdTerm,
    /// Named graph (IRI or blank node), or `None` for the default graph.
    pub graph: Option<JsonLdTerm>,
}

impl JsonLdQuad {
    /// Create a new triple in the default graph.
    pub fn triple(subject: JsonLdTerm, predicate: JsonLdTerm, object: JsonLdTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph: None,
        }
    }

    /// Create a new quad in a named graph.
    pub fn named(
        subject: JsonLdTerm,
        predicate: JsonLdTerm,
        object: JsonLdTerm,
        graph: JsonLdTerm,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph: Some(graph),
        }
    }

    /// Serialise as an N-Quads line.
    pub fn to_nquads_line(&self) -> String {
        if let Some(g) = &self.graph {
            format!(
                "{} {} {} {} .",
                self.subject.to_nquads_string(),
                self.predicate.to_nquads_string(),
                self.object.to_nquads_string(),
                g.to_nquads_string()
            )
        } else {
            format!(
                "{} {} {} .",
                self.subject.to_nquads_string(),
                self.predicate.to_nquads_string(),
                self.object.to_nquads_string()
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Container type
// ─────────────────────────────────────────────────────────────────────────────

/// JSON-LD container types that affect how values are serialised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerType {
    /// `@list` — ordered collection.
    List,
    /// `@set` — unordered collection.
    Set,
    /// `@index` — index map keyed by an arbitrary string.
    Index,
    /// `@language` — language map.
    Language,
    /// `@id` — id map.
    Id,
    /// `@graph` — graph container.
    Graph,
}

impl ContainerType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "@list" => Some(Self::List),
            "@set" => Some(Self::Set),
            "@index" => Some(Self::Index),
            "@language" => Some(Self::Language),
            "@id" => Some(Self::Id),
            "@graph" => Some(Self::Graph),
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn as_str(&self) -> &'static str {
        match self {
            Self::List => "@list",
            Self::Set => "@set",
            Self::Index => "@index",
            Self::Language => "@language",
            Self::Id => "@id",
            Self::Graph => "@graph",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Term definition
// ─────────────────────────────────────────────────────────────────────────────

/// A single term definition inside a JSON-LD context.
#[derive(Debug, Clone)]
pub struct TermDefinition {
    /// The expanded IRI for this term.
    pub iri: String,
    /// Optional container type (e.g. `@list`).
    pub container: Option<ContainerType>,
    /// Optional default language for string values.
    pub language: Option<String>,
    /// Optional type coercion IRI (e.g. `@id`, `xsd:integer`).
    pub type_coercion: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON-LD Context
// ─────────────────────────────────────────────────────────────────────────────

/// Built-in prefix mappings that are always available.
fn builtin_prefixes() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "rdf".into(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#".into(),
    );
    m.insert(
        "rdfs".into(),
        "http://www.w3.org/2000/01/rdf-schema#".into(),
    );
    m.insert("owl".into(), "http://www.w3.org/2002/07/owl#".into());
    m.insert("xsd".into(), "http://www.w3.org/2001/XMLSchema#".into());
    m.insert("schema".into(), "http://schema.org/".into());
    m.insert("dc".into(), "http://purl.org/dc/elements/1.1/".into());
    m.insert("dcterms".into(), "http://purl.org/dc/terms/".into());
    m.insert("foaf".into(), "http://xmlns.com/foaf/0.1/".into());
    m.insert("skos".into(), "http://www.w3.org/2004/02/skos/core#".into());
    m
}

/// A parsed JSON-LD context object.
#[derive(Debug, Clone)]
pub struct JsonLdContext {
    /// The base IRI for resolving relative IRIs.
    pub base_iri: Option<String>,
    /// The default vocabulary IRI (`@vocab`).
    pub vocab: Option<String>,
    /// Prefix mappings (`prefix` → `namespace IRI`).
    pub prefixes: HashMap<String, String>,
    /// Term definitions.
    pub terms: HashMap<String, TermDefinition>,
    /// The default language (`@language`).
    pub default_language: Option<String>,
}

impl Default for JsonLdContext {
    fn default() -> Self {
        Self {
            base_iri: None,
            vocab: None,
            prefixes: builtin_prefixes(),
            terms: HashMap::new(),
            default_language: None,
        }
    }
}

impl JsonLdContext {
    /// Create an empty context (no built-in prefixes).
    pub fn empty() -> Self {
        Self {
            base_iri: None,
            vocab: None,
            prefixes: HashMap::new(),
            terms: HashMap::new(),
            default_language: None,
        }
    }

    /// Parse a JSON `@context` value and return a `JsonLdContext`.
    pub fn parse(context: &Value) -> JsonLdResult<Self> {
        let mut ctx = Self::default();
        match context {
            Value::Object(map) => {
                ctx.apply_object(map)?;
            }
            Value::Array(arr) => {
                for item in arr {
                    match item {
                        Value::Object(map) => ctx.apply_object(map)?,
                        Value::String(s) => {
                            // Remote context reference — treat as base IRI hint
                            ctx.base_iri = Some(s.clone());
                        }
                        Value::Null => {
                            // null resets context
                            ctx = Self::empty();
                        }
                        _ => {}
                    }
                }
            }
            Value::String(s) => {
                // Remote context reference
                ctx.base_iri = Some(s.clone());
            }
            Value::Null => {
                ctx = Self::empty();
            }
            _ => {
                return Err(JsonLdError::InvalidContext(
                    "context must be an object, array, string, or null".into(),
                ))
            }
        }
        Ok(ctx)
    }

    fn apply_object(&mut self, map: &Map<String, Value>) -> JsonLdResult<()> {
        // Process @base
        if let Some(base) = map.get("@base") {
            match base {
                Value::String(s) => self.base_iri = Some(s.clone()),
                Value::Null => self.base_iri = None,
                _ => return Err(JsonLdError::InvalidContext("@base must be a string".into())),
            }
        }
        // Process @vocab
        if let Some(vocab) = map.get("@vocab") {
            match vocab {
                Value::String(s) => self.vocab = Some(s.clone()),
                Value::Null => self.vocab = None,
                _ => {
                    return Err(JsonLdError::InvalidContext(
                        "@vocab must be a string".into(),
                    ))
                }
            }
        }
        // Process @language
        if let Some(lang) = map.get("@language") {
            match lang {
                Value::String(s) => self.default_language = Some(s.clone()),
                Value::Null => self.default_language = None,
                _ => {
                    return Err(JsonLdError::InvalidContext(
                        "@language must be a string".into(),
                    ))
                }
            }
        }
        // Process term definitions
        for (key, value) in map.iter() {
            if key.starts_with('@') {
                continue; // already handled keywords
            }
            match value {
                Value::String(iri_or_prefix) => {
                    // Could be a prefix mapping or a term mapping
                    if iri_or_prefix.ends_with('/') || iri_or_prefix.ends_with('#') {
                        self.prefixes.insert(key.clone(), iri_or_prefix.clone());
                    } else {
                        let expanded = self.expand_term(iri_or_prefix);
                        self.terms.insert(
                            key.clone(),
                            TermDefinition {
                                iri: expanded,
                                container: None,
                                language: None,
                                type_coercion: None,
                            },
                        );
                        // Also register as prefix if IRI-like
                        if iri_or_prefix.contains(':') && !iri_or_prefix.starts_with('@') {
                            self.prefixes.insert(key.clone(), iri_or_prefix.clone());
                        }
                    }
                }
                Value::Object(def_map) => {
                    let iri = if let Some(id_val) = def_map.get("@id") {
                        match id_val {
                            Value::String(s) => self.expand_term(s),
                            _ => {
                                return Err(JsonLdError::InvalidContext(format!(
                                    "@id in term '{}' must be a string",
                                    key
                                )))
                            }
                        }
                    } else {
                        // Use vocab or key as IRI
                        if let Some(vocab) = &self.vocab.clone() {
                            format!("{}{}", vocab, key)
                        } else {
                            key.clone()
                        }
                    };

                    let container = def_map
                        .get("@container")
                        .and_then(|v| v.as_str())
                        .and_then(ContainerType::from_str);

                    let language = def_map
                        .get("@language")
                        .and_then(|v| v.as_str())
                        .map(String::from);

                    let type_coercion = def_map
                        .get("@type")
                        .and_then(|v| v.as_str())
                        .map(|t| self.expand_term(t));

                    self.terms.insert(
                        key.clone(),
                        TermDefinition {
                            iri,
                            container,
                            language,
                            type_coercion,
                        },
                    );
                }
                Value::Null => {
                    self.terms.remove(key);
                    self.prefixes.remove(key);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Expand a term or CURIE to a full IRI.
    ///
    /// Resolution order follows JSON-LD 1.1 §6.4 (IRI expansion):
    /// 1. JSON-LD keywords (pass through)
    /// 2. Term definition lookup (exact term name)
    /// 3. CURIE expansion — `prefix:local` where prefix is registered
    ///    (must be checked BEFORE `is_absolute_iri` so that registered prefixes
    ///    such as `rdf`, `xsd`, `foaf`, `schema` are expanded and not treated as
    ///    opaque absolute IRIs)
    /// 4. Already absolute IRI (pass through unchanged)
    /// 5. `@vocab` relative IRI
    /// 6. `@base` relative IRI
    /// 7. Return unchanged
    pub fn expand_term(&self, term: &str) -> String {
        // 1. Keywords pass through
        if term.starts_with('@') {
            return term.to_string();
        }
        // 2. Term definition (exact key match)
        if let Some(def) = self.terms.get(term) {
            return def.iri.clone();
        }
        // 3. CURIE expansion: prefix:local
        //    Must be checked before is_absolute_iri so that registered short prefixes
        //    (rdf, xsd, foaf, schema, ex, …) are expanded rather than passed through.
        if let Some(colon_pos) = term.find(':') {
            let prefix = &term[..colon_pos];
            let local = &term[colon_pos + 1..];
            // Skip protocol-like schemes whose local part begins with "//"
            // (those are genuine absolute IRIs such as "http://…")
            if !local.starts_with("//") {
                if let Some(ns) = self.prefixes.get(prefix) {
                    return format!("{}{}", ns, local);
                }
            }
        }
        // 4. Already-absolute IRI (no registered prefix matched above)
        if is_absolute_iri(term) {
            return term.to_string();
        }
        // 5. @vocab relative
        if let Some(vocab) = &self.vocab {
            return format!("{}{}", vocab, term);
        }
        // 6. @base relative
        if let Some(base) = &self.base_iri {
            return format!("{}{}", base, term);
        }
        // 7. Return unchanged
        term.to_string()
    }

    /// Compact an absolute IRI to its shortest form using this context.
    ///
    /// Resolution order:
    /// 1. JSON-LD keywords (pass through)
    /// 2. Exact term match
    /// 3. `@vocab` prefix
    /// 4. Longest prefix match
    /// 5. Return full IRI
    pub fn compact_iri(&self, iri: &str) -> String {
        // Keywords pass through
        if iri.starts_with('@') {
            return iri.to_string();
        }
        // Exact term match
        for (term, def) in &self.terms {
            if def.iri == iri {
                return term.clone();
            }
        }
        // @vocab match
        if let Some(vocab) = &self.vocab {
            if let Some(local) = iri.strip_prefix(vocab.as_str()) {
                if !local.is_empty() && !local.contains('/') && !local.contains('#') {
                    return local.to_string();
                }
            }
        }
        // Longest prefix match
        let mut best: Option<(usize, String)> = None;
        for (prefix, ns) in &self.prefixes {
            if let Some(local) = iri.strip_prefix(ns.as_str()) {
                if local.is_empty() {
                    continue;
                }
                let len = ns.len();
                if best.as_ref().map_or(true, |(prev_len, _)| len > *prev_len) {
                    best = Some((len, format!("{}:{}", prefix, local)));
                }
            }
        }
        if let Some((_, compact)) = best {
            return compact;
        }
        iri.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────

fn is_absolute_iri(s: &str) -> bool {
    // Must contain "://" or end with ":" followed by no double slash
    if let Some(pos) = s.find(':') {
        let scheme = &s[..pos];
        // Valid scheme: letters, digits, +, -, .
        scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
            && !scheme.is_empty()
    } else {
        false
    }
}

/// XSD string datatype IRI.
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
/// XSD boolean datatype IRI.
const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";
/// XSD integer datatype IRI.
const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
/// XSD double datatype IRI.
const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";
/// RDF lang string datatype IRI.
const RDF_LANG_STRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

// ─────────────────────────────────────────────────────────────────────────────
// Blank node counter (thread-local for deterministic output in tests)
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};
static BLANK_NODE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_blank_node() -> String {
    format!("_:b{}", BLANK_NODE_COUNTER.fetch_add(1, Ordering::Relaxed))
}

// ─────────────────────────────────────────────────────────────────────────────
// Expansion algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Core JSON-LD 1.1 processing algorithms.
pub struct JsonLdProcessor;

impl JsonLdProcessor {
    /// **Expansion** — convert all CURIEs and context-relative terms to full IRIs.
    ///
    /// Input may be a compacted JSON-LD document (with or without inline `@context`).
    /// An external `context` may be provided in addition.
    ///
    /// Returns an expanded JSON-LD array (the spec always returns an array).
    pub fn expand(input: &Value, context: Option<&Value>) -> JsonLdResult<Value> {
        // Build initial context
        let mut ctx = JsonLdContext::default();
        if let Some(c) = context {
            let parsed = JsonLdContext::parse(c)?;
            ctx = parsed;
        }

        let result = expand_node(input, &mut ctx)?;
        // Expansion always returns an array
        Ok(match result {
            Value::Array(_) => result,
            Value::Null => Value::Array(vec![]),
            other => Value::Array(vec![other]),
        })
    }

    /// **Compaction** — shorten IRIs using the supplied context.
    ///
    /// Input should be an expanded JSON-LD document; the context is applied to
    /// produce a compacted representation.
    pub fn compact(input: &Value, context: &Value) -> JsonLdResult<Value> {
        let ctx = JsonLdContext::parse(context)?;

        // Expand first to normalise, then compact
        let expanded = Self::expand(input, None)?;
        let compacted = compact_node(&expanded, &ctx);

        // Wrap result with context
        let mut result = Map::new();
        result.insert("@context".into(), context.clone());
        match compacted {
            Value::Array(arr) if arr.len() == 1 => {
                if let Value::Object(obj) = &arr[0] {
                    for (k, v) in obj {
                        result.insert(k.clone(), v.clone());
                    }
                } else {
                    result.insert("@graph".into(), Value::Array(arr));
                }
            }
            Value::Array(arr) => {
                result.insert("@graph".into(), Value::Array(arr));
            }
            Value::Object(obj) => {
                for (k, v) in obj {
                    result.insert(k.clone(), v.clone());
                }
            }
            other => {
                result.insert("@value".into(), other);
            }
        }
        Ok(Value::Object(result))
    }

    /// **Flattening** — convert nested JSON-LD to a flat list of nodes in `@graph`.
    ///
    /// All blank nodes are renamed to canonical identifiers.
    pub fn flatten(input: &Value) -> JsonLdResult<Value> {
        let expanded = Self::expand(input, None)?;
        let mut node_map: HashMap<String, Map<String, Value>> = HashMap::new();
        let mut blank_map: HashMap<String, String> = HashMap::new();

        collect_nodes(&expanded, &mut node_map, &mut blank_map);

        let mut nodes: Vec<Value> = node_map.into_values().map(Value::Object).collect();
        nodes.sort_by_key(|n| {
            n.get("@id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        });

        Ok(json!({
            "@graph": nodes
        }))
    }

    /// **Framing** — reshape JSON-LD to match a supplied frame structure.
    ///
    /// Nodes matching the frame are selected and embedded according to the
    /// frame's shape.
    pub fn frame(input: &Value, frame: &Value) -> JsonLdResult<Value> {
        // First flatten to get canonical node list
        let flat = Self::flatten(input)?;
        let graph = flat
            .get("@graph")
            .and_then(|v| v.as_array())
            .ok_or_else(|| JsonLdError::Framing("flattened document has no @graph".into()))?;

        // Build a node index by @id
        let mut node_index: HashMap<String, &Value> = HashMap::new();
        for node in graph {
            if let Some(id) = node.get("@id").and_then(|v| v.as_str()) {
                node_index.insert(id.to_string(), node);
            }
        }

        let framed_nodes = apply_frame(graph, frame, &node_index)?;

        Ok(json!({
            "@graph": framed_nodes
        }))
    }

    /// **to_rdf** — convert JSON-LD to a list of RDF quads.
    pub fn to_rdf(input: &Value) -> JsonLdResult<Vec<JsonLdQuad>> {
        let expanded = Self::expand(input, None)?;
        let mut quads: Vec<JsonLdQuad> = Vec::new();
        let nodes = match &expanded {
            Value::Array(arr) => arr.as_slice(),
            _ => return Ok(quads),
        };
        for node in nodes {
            node_to_rdf(node, None, &mut quads)?;
        }
        Ok(quads)
    }

    /// **from_rdf** — convert RDF quads to a JSON-LD document.
    ///
    /// If a context is provided, the result is compacted with it.
    pub fn from_rdf(quads: &[JsonLdQuad], context: Option<&Value>) -> JsonLdResult<Value> {
        // Group quads by graph name
        let mut graph_map: HashMap<String, Vec<&JsonLdQuad>> = HashMap::new();
        for quad in quads {
            let graph_key = quad
                .graph
                .as_ref()
                .map(|g| g.to_nquads_string())
                .unwrap_or_else(|| "@default".into());
            graph_map.entry(graph_key).or_default().push(quad);
        }

        let mut all_nodes: Vec<Value> = Vec::new();

        for (graph_name, graph_quads) in &graph_map {
            // Build node objects from quads in this graph
            let mut node_map: HashMap<String, Map<String, Value>> = HashMap::new();
            for quad in graph_quads {
                let subj_key = quad.subject.to_nquads_string();
                let entry = node_map.entry(subj_key.clone()).or_insert_with(|| {
                    let mut m = Map::new();
                    m.insert("@id".into(), Value::String(subj_key.clone()));
                    m
                });

                let pred_iri = match &quad.predicate {
                    JsonLdTerm::Iri(iri) => iri.clone(),
                    _ => return Err(JsonLdError::InvalidIri("predicate must be an IRI".into())),
                };

                let obj_value = term_to_json_ld_value(&quad.object);
                let values = entry
                    .entry(pred_iri)
                    .or_insert_with(|| Value::Array(vec![]));
                if let Value::Array(arr) = values {
                    arr.push(obj_value);
                }
            }

            let nodes: Vec<Value> = node_map.into_values().map(Value::Object).collect();

            if graph_name == "@default" {
                all_nodes.extend(nodes);
            } else {
                // Named graph — wrap in @graph
                let graph_id = graph_name.trim_start_matches('<').trim_end_matches('>');
                all_nodes.push(json!({
                    "@id": graph_id,
                    "@graph": nodes
                }));
            }
        }

        let result = Value::Array(all_nodes);

        if let Some(ctx) = context {
            Self::compact(&result, ctx)
        } else {
            Ok(result)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Expansion helpers
// ─────────────────────────────────────────────────────────────────────────────

fn expand_node(value: &Value, ctx: &mut JsonLdContext) -> JsonLdResult<Value> {
    match value {
        Value::Array(arr) => {
            let mut result = Vec::with_capacity(arr.len());
            for item in arr {
                let expanded = expand_node(item, ctx)?;
                match expanded {
                    Value::Array(inner) => result.extend(inner),
                    Value::Null => {}
                    other => result.push(other),
                }
            }
            Ok(Value::Array(result))
        }
        Value::Object(map) => expand_object(map, ctx),
        Value::String(s) => {
            // Bare string in array — expand as @value
            Ok(json!({ "@value": s }))
        }
        Value::Bool(b) => Ok(json!({
            "@value": b,
            "@type": XSD_BOOLEAN
        })),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                Ok(json!({ "@value": n, "@type": XSD_INTEGER }))
            } else {
                Ok(json!({ "@value": n, "@type": XSD_DOUBLE }))
            }
        }
        Value::Null => Ok(Value::Null),
    }
}

fn expand_object(map: &Map<String, Value>, ctx: &mut JsonLdContext) -> JsonLdResult<Value> {
    // If the object has @context, process it first
    let mut local_ctx = ctx.clone();
    if let Some(inline_ctx) = map.get("@context") {
        let parsed = JsonLdContext::parse(inline_ctx)?;
        // Merge parsed context into local_ctx
        local_ctx.base_iri = parsed.base_iri.or(local_ctx.base_iri);
        local_ctx.vocab = parsed.vocab.or(local_ctx.vocab);
        for (k, v) in parsed.prefixes {
            local_ctx.prefixes.insert(k, v);
        }
        for (k, v) in parsed.terms {
            local_ctx.terms.insert(k, v);
        }
        if parsed.default_language.is_some() {
            local_ctx.default_language = parsed.default_language;
        }
    }

    let mut result = Map::new();

    for (key, value) in map {
        if key == "@context" {
            continue;
        }

        let expanded_key = if key.starts_with('@') {
            key.clone()
        } else {
            local_ctx.expand_term(key)
        };

        match expanded_key.as_str() {
            "@id" => {
                if let Value::String(s) = value {
                    let expanded_id = local_ctx.expand_term(s);
                    result.insert("@id".into(), Value::String(expanded_id));
                }
            }
            "@type" => {
                let expanded_type = match value {
                    Value::String(s) => Value::Array(vec![Value::String(local_ctx.expand_term(s))]),
                    Value::Array(arr) => Value::Array(
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .map(|s| Value::String(local_ctx.expand_term(s)))
                            .collect(),
                    ),
                    _ => value.clone(),
                };
                result.insert("@type".into(), expanded_type);
            }
            "@value" => {
                result.insert("@value".into(), value.clone());
            }
            "@language" => {
                if let Value::String(s) = value {
                    result.insert("@language".into(), Value::String(s.clone()));
                }
            }
            "@graph" => {
                let expanded_graph = expand_node(value, &mut local_ctx)?;
                let graph_arr = match expanded_graph {
                    Value::Array(a) => Value::Array(a),
                    other => Value::Array(vec![other]),
                };
                result.insert("@graph".into(), graph_arr);
            }
            "@list" => {
                let expanded_list = expand_node(value, &mut local_ctx)?;
                let list_arr = match expanded_list {
                    Value::Array(a) => Value::Array(a),
                    Value::Null => Value::Array(vec![]),
                    other => Value::Array(vec![other]),
                };
                result.insert("@list".into(), list_arr);
            }
            "@set" => {
                let expanded_set = expand_node(value, &mut local_ctx)?;
                let set_arr = match expanded_set {
                    Value::Array(a) => Value::Array(a),
                    Value::Null => Value::Array(vec![]),
                    other => Value::Array(vec![other]),
                };
                result.insert("@set".into(), set_arr);
            }
            _ if expanded_key.starts_with('@') => {
                // Unknown keyword — skip
            }
            _ => {
                // Regular property — expand value.
                // Per JSON-LD 1.1 § 1.4.15, all property values in expanded form
                // must be arrays.  Wrap single Value::Object / scalar results so
                // that consumers can always call `.as_array()` on the result.
                let expanded_value = expand_property_value(value, &expanded_key, &mut local_ctx)?;
                if !is_empty_array(&expanded_value) {
                    let array_value = match expanded_value {
                        Value::Array(_) => expanded_value,
                        other => Value::Array(vec![other]),
                    };
                    result.insert(expanded_key, array_value);
                }
            }
        }
    }

    if result.is_empty() {
        return Ok(Value::Null);
    }

    Ok(Value::Object(result))
}

fn expand_property_value(
    value: &Value,
    _property: &str,
    ctx: &mut JsonLdContext,
) -> JsonLdResult<Value> {
    match value {
        Value::Array(arr) => {
            let mut result = Vec::with_capacity(arr.len());
            for item in arr {
                let expanded = expand_property_value(item, _property, ctx)?;
                match expanded {
                    Value::Null => {}
                    Value::Array(inner) => result.extend(inner),
                    other => result.push(other),
                }
            }
            Ok(Value::Array(result))
        }
        Value::Object(map) => {
            // Check for @value object
            if map.contains_key("@value") {
                return expand_value_object(map, ctx);
            }
            expand_object(map, ctx)
        }
        Value::String(s) => {
            // String value — wrap as @value with optional language
            let mut obj = Map::new();
            obj.insert("@value".into(), Value::String(s.clone()));
            if let Some(lang) = &ctx.default_language {
                obj.insert("@language".into(), Value::String(lang.clone()));
                obj.insert("@type".into(), Value::String(RDF_LANG_STRING.into()));
            } else {
                obj.insert("@type".into(), Value::String(XSD_STRING.into()));
            }
            Ok(Value::Object(obj))
        }
        Value::Bool(b) => Ok(json!({
            "@value": b,
            "@type": XSD_BOOLEAN
        })),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                Ok(json!({ "@value": n, "@type": XSD_INTEGER }))
            } else {
                Ok(json!({ "@value": n, "@type": XSD_DOUBLE }))
            }
        }
        Value::Null => Ok(Value::Null),
    }
}

fn expand_value_object(map: &Map<String, Value>, ctx: &JsonLdContext) -> JsonLdResult<Value> {
    let mut result = Map::new();
    result.insert("@value".into(), map["@value"].clone());

    if let Some(lang) = map.get("@language") {
        result.insert("@language".into(), lang.clone());
        result.insert("@type".into(), Value::String(RDF_LANG_STRING.into()));
    } else if let Some(t) = map.get("@type") {
        if let Value::String(type_str) = t {
            let expanded_type = ctx.expand_term(type_str);
            result.insert("@type".into(), Value::String(expanded_type));
        }
    } else {
        result.insert("@type".into(), Value::String(XSD_STRING.into()));
    }

    Ok(Value::Object(result))
}

fn is_empty_array(v: &Value) -> bool {
    matches!(v, Value::Array(arr) if arr.is_empty())
}

// ─────────────────────────────────────────────────────────────────────────────
// Compaction helpers
// ─────────────────────────────────────────────────────────────────────────────

fn compact_node(value: &Value, ctx: &JsonLdContext) -> Value {
    match value {
        Value::Array(arr) => Value::Array(arr.iter().map(|v| compact_node(v, ctx)).collect()),
        Value::Object(map) => compact_object(map, ctx),
        other => other.clone(),
    }
}

fn compact_object(map: &Map<String, Value>, ctx: &JsonLdContext) -> Value {
    let mut result = Map::new();

    for (key, value) in map {
        let compact_key = match key.as_str() {
            "@id" | "@type" | "@value" | "@language" | "@graph" | "@list" | "@set" => key.clone(),
            _ => ctx.compact_iri(key),
        };

        let compact_value = match key.as_str() {
            "@id" => {
                if let Value::String(iri) = value {
                    Value::String(ctx.compact_iri(iri))
                } else {
                    value.clone()
                }
            }
            "@type" => match value {
                Value::Array(arr) => {
                    let compacted: Vec<Value> = arr
                        .iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| Value::String(ctx.compact_iri(s)))
                        .collect();
                    if compacted.len() == 1 {
                        compacted.into_iter().next().unwrap_or(Value::Null)
                    } else {
                        Value::Array(compacted)
                    }
                }
                Value::String(s) => Value::String(ctx.compact_iri(s)),
                _ => value.clone(),
            },
            "@value" => {
                // Compact @value objects
                if let Some(type_val) = map.get("@type") {
                    let type_str = type_val.as_str().unwrap_or("");
                    match type_str {
                        t if t == XSD_STRING => value.clone(),
                        t if t == XSD_INTEGER => value.clone(),
                        t if t == XSD_DOUBLE => value.clone(),
                        t if t == XSD_BOOLEAN => value.clone(),
                        _ => value.clone(),
                    }
                } else {
                    value.clone()
                }
            }
            _ => compact_node(value, ctx),
        };

        // Unwrap single-value arrays for predicates (JSON-LD compaction)
        let final_value = if key.as_str() != "@type" && key.as_str() != "@graph" {
            match &compact_value {
                Value::Array(arr) if arr.len() == 1 => arr[0].clone(),
                _ => compact_value,
            }
        } else {
            compact_value
        };

        result.insert(compact_key, final_value);
    }

    Value::Object(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Flattening helpers
// ─────────────────────────────────────────────────────────────────────────────

fn collect_nodes(
    value: &Value,
    node_map: &mut HashMap<String, Map<String, Value>>,
    blank_map: &mut HashMap<String, String>,
) {
    match value {
        Value::Array(arr) => {
            for item in arr {
                collect_nodes(item, node_map, blank_map);
            }
        }
        Value::Object(map) => {
            // Determine node ID
            let id = if let Some(id_val) = map.get("@id") {
                id_val.as_str().unwrap_or("").to_string()
            } else {
                next_blank_node()
            };

            // Ensure the node entry exists
            node_map.entry(id.clone()).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("@id".into(), Value::String(id.clone()));
                m
            });

            // Collect type values and flattened property values without holding
            // a mutable reference to node_map (needed for recursive flatten calls).
            let mut type_values: Vec<Value> = Vec::new();
            let mut prop_values: Vec<(String, Value)> = Vec::new();

            for (k, v) in map {
                if k == "@id" || k == "@context" {
                    continue;
                }
                if k == "@type" {
                    match v {
                        Value::Array(new_types) => type_values.extend(new_types.clone()),
                        _ => type_values.push(v.clone()),
                    }
                    continue;
                }
                // Flatten recursively — node_map is free here (no entry borrow held)
                let flat_v = flatten_value(v, node_map, blank_map);
                prop_values.push((k.clone(), flat_v));
            }

            // Now update node_map with collected values
            let entry = node_map.entry(id.clone()).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("@id".into(), Value::String(id.clone()));
                m
            });

            if !type_values.is_empty() {
                let existing = entry.entry("@type").or_insert_with(|| Value::Array(vec![]));
                if let Value::Array(types) = existing {
                    types.extend(type_values);
                }
            }

            for (k, flat_v) in prop_values {
                let existing = entry.entry(k).or_insert_with(|| Value::Array(vec![]));
                if let Value::Array(arr) = existing {
                    match flat_v {
                        Value::Array(inner) => arr.extend(inner),
                        Value::Null => {}
                        other => arr.push(other),
                    }
                }
            }
        }
        _ => {}
    }
}

fn flatten_value(
    value: &Value,
    node_map: &mut HashMap<String, Map<String, Value>>,
    blank_map: &mut HashMap<String, String>,
) -> Value {
    match value {
        Value::Array(arr) => {
            let flattened: Vec<Value> = arr
                .iter()
                .map(|v| flatten_value(v, node_map, blank_map))
                .filter(|v| !matches!(v, Value::Null))
                .collect();
            Value::Array(flattened)
        }
        Value::Object(map) => {
            // If it's a @value object, return as-is
            if map.contains_key("@value") {
                return Value::Object(map.clone());
            }
            // If it's a @list object, flatten list items
            if map.contains_key("@list") {
                let list_items = map["@list"].as_array().map(|a| a.as_slice()).unwrap_or(&[]);
                let flattened: Vec<Value> = list_items
                    .iter()
                    .map(|v| flatten_value(v, node_map, blank_map))
                    .collect();
                return json!({ "@list": flattened });
            }
            // Nested node — collect it and return an @id reference
            let id = if let Some(id_val) = map.get("@id") {
                id_val.as_str().unwrap_or("").to_string()
            } else {
                canonicalize_blank_node(&next_blank_node(), blank_map)
            };
            collect_nodes(&Value::Object(map.clone()), node_map, blank_map);
            json!({ "@id": id })
        }
        other => other.clone(),
    }
}

fn canonicalize_blank_node(id: &str, blank_map: &mut HashMap<String, String>) -> String {
    if let Some(canonical) = blank_map.get(id) {
        return canonical.clone();
    }
    let canonical = format!("_:c14n{}", blank_map.len());
    blank_map.insert(id.to_string(), canonical.clone());
    canonical
}

// ─────────────────────────────────────────────────────────────────────────────
// Framing helpers
// ─────────────────────────────────────────────────────────────────────────────

fn apply_frame<'a>(
    nodes: &'a [Value],
    frame: &Value,
    node_index: &HashMap<String, &'a Value>,
) -> JsonLdResult<Vec<Value>> {
    let frame_obj = match frame {
        Value::Object(m) => m,
        _ => return Err(JsonLdError::Framing("frame must be an object".into())),
    };

    let mut result = Vec::new();

    for node in nodes {
        if node_matches_frame(node, frame_obj) {
            let embedded = embed_node(node, frame_obj, node_index)?;
            result.push(embedded);
        }
    }

    Ok(result)
}

fn node_matches_frame(node: &Value, frame: &Map<String, Value>) -> bool {
    let node_obj = match node {
        Value::Object(m) => m,
        _ => return false,
    };

    // Check @type constraints
    if let Some(frame_type) = frame.get("@type") {
        let node_types: Vec<&str> = node_obj
            .get("@type")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let required_types: Vec<&str> = match frame_type {
            Value::Array(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
            Value::String(s) => vec![s.as_str()],
            _ => vec![],
        };

        if !required_types.is_empty() && !required_types.iter().any(|t| node_types.contains(t)) {
            return false;
        }
    }

    // Check @id constraints
    if let Some(Value::String(required_id)) = frame.get("@id") {
        let node_id = node_obj.get("@id").and_then(|v| v.as_str()).unwrap_or("");
        if node_id != required_id {
            return false;
        }
    }

    // Check property existence
    for (key, frame_value) in frame {
        if key.starts_with('@') {
            continue;
        }
        if let Value::Object(fv) = frame_value {
            // Empty object means property must exist
            if fv.is_empty() && !node_obj.contains_key(key) {
                return false;
            }
        }
    }

    true
}

fn embed_node(
    node: &Value,
    frame: &Map<String, Value>,
    node_index: &HashMap<String, &Value>,
) -> JsonLdResult<Value> {
    let node_obj = match node {
        Value::Object(m) => m.clone(),
        _ => return Ok(node.clone()),
    };

    let mut result = Map::new();

    // Copy @id
    if let Some(id) = node_obj.get("@id") {
        result.insert("@id".into(), id.clone());
    }

    // Copy @type
    if let Some(t) = node_obj.get("@type") {
        result.insert("@type".into(), t.clone());
    }

    // Process frame properties
    for (key, frame_prop) in frame {
        if key.starts_with('@') {
            continue;
        }
        if let Some(node_prop) = node_obj.get(key) {
            // If frame property has a nested frame, embed recursively
            let embedded_prop = match frame_prop {
                Value::Object(sub_frame) if !sub_frame.is_empty() => {
                    embed_property_values(node_prop, sub_frame, node_index)?
                }
                _ => node_prop.clone(),
            };
            result.insert(key.clone(), embedded_prop);
        }
    }

    // Also copy properties not in frame but in node
    for (key, value) in &node_obj {
        if !key.starts_with('@') && !result.contains_key(key) {
            result.insert(key.clone(), value.clone());
        }
    }

    Ok(Value::Object(result))
}

fn embed_property_values(
    values: &Value,
    sub_frame: &Map<String, Value>,
    node_index: &HashMap<String, &Value>,
) -> JsonLdResult<Value> {
    match values {
        Value::Array(arr) => {
            let embedded: JsonLdResult<Vec<Value>> = arr
                .iter()
                .map(|v| embed_single_value(v, sub_frame, node_index))
                .collect();
            Ok(Value::Array(embedded?))
        }
        _ => embed_single_value(values, sub_frame, node_index),
    }
}

fn embed_single_value(
    value: &Value,
    sub_frame: &Map<String, Value>,
    node_index: &HashMap<String, &Value>,
) -> JsonLdResult<Value> {
    // If it's an @id reference, resolve it from the index
    if let Value::Object(m) = value {
        if m.len() == 1 {
            if let Some(Value::String(id)) = m.get("@id") {
                if let Some(full_node) = node_index.get(id) {
                    return embed_node(full_node, sub_frame, node_index);
                }
            }
        }
        return embed_node(value, sub_frame, node_index);
    }
    Ok(value.clone())
}

// ─────────────────────────────────────────────────────────────────────────────
// to_rdf helpers
// ─────────────────────────────────────────────────────────────────────────────

fn node_to_rdf(
    node: &Value,
    graph_name: Option<&JsonLdTerm>,
    quads: &mut Vec<JsonLdQuad>,
) -> JsonLdResult<()> {
    let map = match node {
        Value::Object(m) => m,
        Value::Array(arr) => {
            for item in arr {
                node_to_rdf(item, graph_name, quads)?;
            }
            return Ok(());
        }
        _ => return Ok(()),
    };

    let subject = if let Some(id_val) = map.get("@id") {
        match id_val {
            Value::String(id) => {
                if id.starts_with("_:") {
                    JsonLdTerm::BlankNode(id.clone())
                } else {
                    JsonLdTerm::Iri(id.clone())
                }
            }
            _ => return Ok(()),
        }
    } else {
        JsonLdTerm::BlankNode(next_blank_node())
    };

    // @type → rdf:type triples
    if let Some(types_val) = map.get("@type") {
        let types = match types_val {
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .map(String::from)
                .collect::<Vec<_>>(),
            Value::String(s) => vec![s.clone()],
            _ => vec![],
        };
        for type_iri in types {
            let predicate =
                JsonLdTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into());
            let object = JsonLdTerm::Iri(type_iri);
            quads.push(JsonLdQuad {
                subject: subject.clone(),
                predicate,
                object,
                graph: graph_name.cloned(),
            });
        }
    }

    // @graph — named graph
    if let Some(graph_val) = map.get("@graph") {
        let nodes = match graph_val {
            Value::Array(arr) => arr.as_slice(),
            _ => return Ok(()),
        };
        for inner_node in nodes {
            node_to_rdf(inner_node, Some(&subject), quads)?;
        }
        return Ok(());
    }

    // Regular properties
    for (key, value) in map {
        if key.starts_with('@') {
            continue;
        }
        let predicate = JsonLdTerm::Iri(key.clone());
        let values = match value {
            Value::Array(arr) => arr.as_slice(),
            _ => std::slice::from_ref(value),
        };
        for v in values {
            let object = value_to_rdf_term(v)?;
            if let Some(obj) = object {
                quads.push(JsonLdQuad {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object: obj,
                    graph: graph_name.cloned(),
                });
            }
        }
    }

    Ok(())
}

fn value_to_rdf_term(value: &Value) -> JsonLdResult<Option<JsonLdTerm>> {
    match value {
        Value::Object(map) => {
            if let Some(v) = map.get("@value") {
                let lit_value = match v {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => b.to_string(),
                    Value::Number(n) => n.to_string(),
                    _ => return Ok(None),
                };
                let language = map
                    .get("@language")
                    .and_then(|l| l.as_str())
                    .map(String::from);
                let datatype = map
                    .get("@type")
                    .and_then(|t| t.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| XSD_STRING.into());
                Ok(Some(JsonLdTerm::Literal {
                    value: lit_value,
                    datatype,
                    language,
                }))
            } else if let Some(id_val) = map.get("@id") {
                match id_val {
                    Value::String(id) => {
                        if id.starts_with("_:") {
                            Ok(Some(JsonLdTerm::BlankNode(id.clone())))
                        } else {
                            Ok(Some(JsonLdTerm::Iri(id.clone())))
                        }
                    }
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        Value::String(s) => Ok(Some(JsonLdTerm::Literal {
            value: s.clone(),
            datatype: XSD_STRING.into(),
            language: None,
        })),
        Value::Bool(b) => Ok(Some(JsonLdTerm::Literal {
            value: b.to_string(),
            datatype: XSD_BOOLEAN.into(),
            language: None,
        })),
        Value::Number(n) => Ok(Some(JsonLdTerm::Literal {
            value: n.to_string(),
            datatype: if n.is_i64() {
                XSD_INTEGER.into()
            } else {
                XSD_DOUBLE.into()
            },
            language: None,
        })),
        Value::Null => Ok(None),
        Value::Array(_) => Ok(None),
    }
}

fn term_to_json_ld_value(term: &JsonLdTerm) -> Value {
    match term {
        JsonLdTerm::Iri(iri) => json!({ "@id": iri }),
        JsonLdTerm::BlankNode(id) => json!({ "@id": id }),
        JsonLdTerm::Literal {
            value,
            datatype,
            language,
        } => {
            if let Some(lang) = language {
                json!({ "@value": value, "@language": lang })
            } else {
                json!({ "@value": value, "@type": datatype })
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON-LD Writer
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight triple/quad holder for the writer.
#[derive(Debug, Clone)]
pub struct Triple {
    /// Subject IRI or blank node string.
    pub subject: String,
    /// Predicate IRI string.
    pub predicate: String,
    /// Object term.
    pub object: WriterObject,
}

/// A lightweight quad holder for the writer.
#[derive(Debug, Clone)]
pub struct Quad {
    /// Subject IRI or blank node string.
    pub subject: String,
    /// Predicate IRI string.
    pub predicate: String,
    /// Object term.
    pub object: WriterObject,
    /// Named graph IRI or blank node string.
    pub graph: Option<String>,
}

/// Object in a triple or quad for the writer.
#[derive(Debug, Clone)]
pub enum WriterObject {
    /// An IRI node.
    Iri(String),
    /// A blank node.
    BlankNode(String),
    /// A plain literal.
    Literal(String),
    /// A typed literal.
    TypedLiteral(String, String),
    /// A language-tagged literal.
    LangLiteral(String, String),
}

impl WriterObject {
    fn to_json_ld_value(&self, ctx: Option<&JsonLdContext>) -> Value {
        match self {
            Self::Iri(iri) => {
                let compacted = ctx
                    .map(|c| c.compact_iri(iri))
                    .unwrap_or_else(|| iri.clone());
                json!({ "@id": compacted })
            }
            Self::BlankNode(id) => json!({ "@id": id }),
            Self::Literal(s) => json!({ "@value": s }),
            Self::TypedLiteral(s, dt) => json!({ "@value": s, "@type": dt }),
            Self::LangLiteral(s, lang) => json!({ "@value": s, "@language": lang }),
        }
    }
}

/// JSON-LD 1.1 writer with configurable output options.
pub struct JsonLdWriter {
    /// Optional JSON-LD context to embed in the output.
    pub context: Option<Value>,
    /// Whether to compact IRIs using the context.
    pub compact: bool,
    /// Whether to pretty-print the JSON output.
    pub pretty: bool,
}

impl Default for JsonLdWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonLdWriter {
    /// Create a new writer with defaults (expanded, not pretty-printed).
    pub fn new() -> Self {
        Self {
            context: None,
            compact: false,
            pretty: false,
        }
    }

    /// Attach a JSON-LD context to the writer output.
    pub fn with_context(mut self, context: Value) -> Self {
        self.context = Some(context);
        self
    }

    /// Enable compaction of IRIs using the attached context.
    pub fn compact_mode(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Enable pretty-printing (indented JSON).
    pub fn pretty_print(mut self) -> Self {
        self.pretty = true;
        self
    }

    /// Write a collection of triples as a JSON-LD document string.
    pub fn write_triples(&self, triples: &[Triple]) -> JsonLdResult<String> {
        let ctx_parsed = self
            .context
            .as_ref()
            .map(JsonLdContext::parse)
            .transpose()?;
        let ctx_ref = if self.compact {
            ctx_parsed.as_ref()
        } else {
            None
        };

        // Group by subject
        let mut subject_map: HashMap<String, Map<String, Value>> = HashMap::new();
        for triple in triples {
            let subj_compacted = ctx_ref
                .map(|c| c.compact_iri(&triple.subject))
                .unwrap_or_else(|| triple.subject.clone());

            let entry = subject_map
                .entry(triple.subject.clone())
                .or_insert_with(|| {
                    let mut m = Map::new();
                    m.insert("@id".into(), Value::String(subj_compacted.clone()));
                    m
                });

            let pred_key = ctx_ref
                .map(|c| c.compact_iri(&triple.predicate))
                .unwrap_or_else(|| triple.predicate.clone());

            let obj_value = triple.object.to_json_ld_value(ctx_ref);
            let values = entry
                .entry(pred_key)
                .or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(arr) = values {
                arr.push(obj_value);
            }
        }

        let graph: Vec<Value> = subject_map.into_values().map(Value::Object).collect();
        let mut doc = Map::new();
        if let Some(ctx) = &self.context {
            doc.insert("@context".into(), ctx.clone());
        }
        doc.insert("@graph".into(), Value::Array(graph));

        self.serialize_json(&Value::Object(doc))
    }

    /// Write a collection of quads as a JSON-LD document string.
    pub fn write_quads(&self, quads: &[Quad]) -> JsonLdResult<String> {
        let ctx_parsed = self
            .context
            .as_ref()
            .map(JsonLdContext::parse)
            .transpose()?;
        let ctx_ref = if self.compact {
            ctx_parsed.as_ref()
        } else {
            None
        };

        // Group quads by graph, then by subject
        let mut graph_map: HashMap<String, HashMap<String, Map<String, Value>>> = HashMap::new();

        for quad in quads {
            let graph_key = quad.graph.clone().unwrap_or_else(|| "@default".into());
            let subj_compacted = ctx_ref
                .map(|c| c.compact_iri(&quad.subject))
                .unwrap_or_else(|| quad.subject.clone());

            let graph_entry = graph_map.entry(graph_key.clone()).or_default();
            let entry = graph_entry.entry(quad.subject.clone()).or_insert_with(|| {
                let mut m = Map::new();
                m.insert("@id".into(), Value::String(subj_compacted.clone()));
                m
            });

            let pred_key = ctx_ref
                .map(|c| c.compact_iri(&quad.predicate))
                .unwrap_or_else(|| quad.predicate.clone());

            let obj_value = quad.object.to_json_ld_value(ctx_ref);
            let values = entry
                .entry(pred_key)
                .or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(arr) = values {
                arr.push(obj_value);
            }
        }

        // Build document
        let default_nodes = graph_map
            .remove("@default")
            .map(|m| m.into_values().map(Value::Object).collect::<Vec<_>>())
            .unwrap_or_default();

        let mut named_graphs: Vec<Value> = graph_map
            .into_iter()
            .map(|(graph_id, node_map)| {
                let nodes: Vec<Value> = node_map.into_values().map(Value::Object).collect();
                let compact_graph_id = ctx_ref
                    .map(|c| c.compact_iri(&graph_id))
                    .unwrap_or(graph_id);
                json!({
                    "@id": compact_graph_id,
                    "@graph": nodes
                })
            })
            .collect();

        let mut all_nodes = default_nodes;
        all_nodes.append(&mut named_graphs);

        let mut doc = Map::new();
        if let Some(ctx) = &self.context {
            doc.insert("@context".into(), ctx.clone());
        }
        doc.insert("@graph".into(), Value::Array(all_nodes));

        self.serialize_json(&Value::Object(doc))
    }

    fn serialize_json(&self, value: &Value) -> JsonLdResult<String> {
        if self.pretty {
            serde_json::to_string_pretty(value).map_err(JsonLdError::Json)
        } else {
            serde_json::to_string(value).map_err(JsonLdError::Json)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
