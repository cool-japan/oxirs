//! JSON-LD context parsing, IRI expansion, and compaction.
//!
//! Exports: [`JsonLdContext`], [`TermDefinition`], [`ContainerType`],
//! [`JsonLdError`], [`JsonLdResult`], [`JsonLdTerm`], [`JsonLdQuad`],
//! XSD/RDF constants, and the [`next_blank_node`] utility.

use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
    pub(crate) fn from_str(s: &str) -> Option<Self> {
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
    pub(crate) fn as_str(&self) -> &'static str {
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
pub(crate) fn builtin_prefixes() -> HashMap<String, String> {
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

    pub(crate) fn apply_object(&mut self, map: &Map<String, Value>) -> JsonLdResult<()> {
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

/// Returns `true` if `s` looks like an absolute IRI (has a valid scheme prefix).
pub fn is_absolute_iri(s: &str) -> bool {
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
pub const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
/// XSD boolean datatype IRI.
pub const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";
/// XSD integer datatype IRI.
pub const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
/// XSD double datatype IRI.
pub const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";
/// RDF lang string datatype IRI.
pub const RDF_LANG_STRING: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString";

// ─────────────────────────────────────────────────────────────────────────────
// Blank node counter (thread-local for deterministic output in tests)
// ─────────────────────────────────────────────────────────────────────────────

static BLANK_NODE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn next_blank_node() -> String {
    format!("_:b{}", BLANK_NODE_COUNTER.fetch_add(1, Ordering::Relaxed))
}
