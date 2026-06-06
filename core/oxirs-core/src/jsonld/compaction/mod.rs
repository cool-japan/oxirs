//! JSON-LD 1.1 Compaction Algorithm
//!
//! This module implements the W3C JSON-LD 1.1 Compaction specification:
//! <https://www.w3.org/TR/json-ld11-api/#compaction-algorithm>
//!
//! Compaction is the inverse of Expansion: given an expanded JSON-LD document
//! and a context, produce a compact document that uses the prefixes, terms, and
//! language/type maps defined in the context.

pub mod algorithm;
pub mod context;
#[cfg(test)]
mod tests;

pub use algorithm::{compact_array, compact_node, compact_value};
pub use context::{compact_iri, create_compact_context, find_term};

use indexmap::IndexMap;
use std::collections::HashMap;
use thiserror::Error;

// ============================================================================
// JsonLdValue — the core JSON-LD value type
// ============================================================================

/// A JSON-LD value that can appear in an expanded or compact JSON-LD document.
///
/// Uses [`IndexMap`] for objects to maintain insertion order (required by JSON-LD spec).
#[derive(Debug, Clone, PartialEq)]
pub enum JsonLdValue {
    /// JSON null.
    Null,
    /// JSON boolean.
    Bool(bool),
    /// JSON number.
    Number(f64),
    /// JSON string.
    Str(String),
    /// JSON array.
    Array(Vec<JsonLdValue>),
    /// JSON object with insertion-ordered keys.
    Object(IndexMap<String, JsonLdValue>),
}

impl JsonLdValue {
    /// Returns `true` if this value is `Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns the string value if this is a `Str`, otherwise `None`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the object map if this is an `Object`, otherwise `None`.
    pub fn as_object(&self) -> Option<&IndexMap<String, JsonLdValue>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Returns the array if this is an `Array`, otherwise `None`.
    pub fn as_array(&self) -> Option<&[JsonLdValue]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Wraps a value in a single-element array.
    pub fn into_array_if_not(self) -> Vec<JsonLdValue> {
        match self {
            Self::Array(a) => a,
            other => vec![other],
        }
    }
}

// ============================================================================
// Context types
// ============================================================================

/// Container type for a JSON-LD term definition.
///
/// Corresponds to the `@container` keyword values defined in JSON-LD 1.1 §4.3.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContainerType {
    /// `@list` — the value is an ordered list.
    List,
    /// `@set` — the value is an unordered set (coerce to array).
    Set,
    /// `@language` — value is a language-keyed map.
    Language,
    /// `@index` — value is an index-keyed map.
    Index,
    /// `@id` — value is an id-keyed map.
    Id,
    /// `@type` — value is a type-keyed map.
    Type,
    /// `@graph` — value is a named graph.
    Graph,
}

/// Definition of a single term in the active context.
///
/// Corresponds to a JSON-LD term definition as specified in §4.1.
#[derive(Debug, Clone)]
pub struct TermDefinition {
    /// The expanded IRI mapping for this term.
    pub iri_mapping: Option<String>,
    /// Whether this term can be used as a prefix.
    pub prefix_flag: bool,
    /// Whether this term definition is protected.
    pub protected: bool,
    /// Whether this is a reverse property.
    pub reverse_property: bool,
    /// Container types for the term's values.
    pub container: Vec<ContainerType>,
    /// Default language for the term's values.
    pub language: Option<String>,
    /// Default text direction for the term's values.
    pub direction: Option<String>,
    /// Nest value for the term.
    pub nest: Option<String>,
    /// Type mapping for the term.
    pub type_mapping: Option<String>,
}

impl TermDefinition {
    /// Creates a minimal term definition with just an IRI mapping.
    pub fn simple(iri: impl Into<String>) -> Self {
        Self {
            iri_mapping: Some(iri.into()),
            prefix_flag: false,
            protected: false,
            reverse_property: false,
            container: Vec::new(),
            language: None,
            direction: None,
            nest: None,
            type_mapping: None,
        }
    }

    /// Creates a prefix term definition.
    pub fn prefix(iri: impl Into<String>) -> Self {
        Self {
            iri_mapping: Some(iri.into()),
            prefix_flag: true,
            protected: false,
            reverse_property: false,
            container: Vec::new(),
            language: None,
            direction: None,
            nest: None,
            type_mapping: None,
        }
    }
}

/// The active JSON-LD context used during compaction.
///
/// Corresponds to the W3C JSON-LD 1.1 active context data structure.
#[derive(Debug, Clone, Default)]
pub struct JsonLdContext {
    /// Term definitions keyed by compact term name.
    pub terms: HashMap<String, TermDefinition>,
    /// Optional vocabulary mapping (`@vocab`).
    pub vocab: Option<String>,
    /// Optional base IRI (`@base`).
    pub base: Option<String>,
    /// Optional default language (`@language`).
    pub language: Option<String>,
    /// Optional default text direction (`@direction`).
    pub direction: Option<String>,
}

impl JsonLdContext {
    /// Creates an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a simple prefix mapping (term → IRI prefix).
    pub fn add_prefix(&mut self, prefix: impl Into<String>, iri: impl Into<String>) {
        self.terms
            .insert(prefix.into(), TermDefinition::prefix(iri.into()));
    }

    /// Adds a term definition.
    pub fn add_term(&mut self, term: impl Into<String>, def: TermDefinition) {
        self.terms.insert(term.into(), def);
    }
}

// ============================================================================
// Options
// ============================================================================

/// Processing mode for JSON-LD algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessingMode {
    /// JSON-LD 1.0 processing.
    JsonLd10,
    /// JSON-LD 1.1 processing (default).
    #[default]
    JsonLd11,
}

/// Options controlling the compaction algorithm.
#[derive(Debug, Clone)]
pub struct CompactionOptions {
    /// If `true` (default), compact single-element arrays to a scalar value
    /// when the container does not require an array.
    pub compact_arrays: bool,
    /// Processing mode (JSON-LD 1.0 or 1.1).
    pub processing_mode: ProcessingMode,
    /// If `true`, sort keys in output objects for deterministic output.
    pub ordered: bool,
    /// Optional override for the base IRI.
    pub base: Option<String>,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            compact_arrays: true,
            processing_mode: ProcessingMode::JsonLd11,
            ordered: false,
            base: None,
        }
    }
}

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur during JSON-LD compaction.
#[derive(Debug, Error)]
pub enum CompactionError {
    /// The context was invalid.
    #[error("Invalid context: {0}")]
    InvalidContext(String),

    /// An IRI was invalid.
    #[error("Invalid IRI: {0}")]
    InvalidIri(String),

    /// A general processing error.
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Two terms conflict for the same compacted representation.
    #[error("Term collision: term '{term}' is already defined with a different mapping")]
    CollisionError {
        /// The conflicting term name.
        term: String,
    },

    /// An attempt was made to redefine a protected term.
    #[error("Protected term redefinition: cannot redefine protected term '{0}'")]
    ProtectedTermRedefinition(String),
}

// ============================================================================
// Main compaction entry point
// ============================================================================

/// Compact an expanded JSON-LD document using the given context.
///
/// This is the main entry point for the W3C JSON-LD 1.1 Compaction Algorithm.
/// See <https://www.w3.org/TR/json-ld11-api/#compaction-algorithm>.
///
/// # Arguments
///
/// * `input` — the expanded JSON-LD document (typically an array of node objects)
/// * `context` — the active context defining prefix mappings, vocabulary, etc.
/// * `options` — options controlling compaction behaviour
///
/// # Returns
///
/// A compact JSON-LD document as a [`JsonLdValue`].
pub fn compact(
    input: &JsonLdValue,
    context: &JsonLdContext,
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    // Step 1: If input is an array, compact each element.
    let compacted_input = match input {
        JsonLdValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let c = compact_element(item, context, None, options)?;
                if !c.is_null() {
                    out.push(c);
                }
            }
            if out.len() == 1 && options.compact_arrays {
                out.into_iter().next().unwrap_or(JsonLdValue::Null)
            } else {
                JsonLdValue::Array(out)
            }
        }
        other => compact_element(other, context, None, options)?,
    };

    // Step 2: Build the output document with the @context entry.
    let ctx_value = create_compact_context(context);
    let mut result: IndexMap<String, JsonLdValue> = IndexMap::new();

    if !ctx_value.is_null() {
        result.insert("@context".to_string(), ctx_value);
    }

    // Merge compacted input into the result object.
    match compacted_input {
        JsonLdValue::Object(map) => {
            for (k, v) in map {
                result.insert(k, v);
            }
        }
        JsonLdValue::Null => {}
        other => {
            // scalar — wrap in @graph
            result.insert("@graph".to_string(), other);
        }
    }

    Ok(JsonLdValue::Object(result))
}

/// Compact a single element (node, value, list, or scalar).
fn compact_element(
    value: &JsonLdValue,
    ctx: &JsonLdContext,
    active_property: Option<&str>,
    options: &CompactionOptions,
) -> Result<JsonLdValue, CompactionError> {
    match value {
        JsonLdValue::Object(map) => {
            // Check if this is a value object, list object, or node object.
            if map.contains_key("@value") {
                compact_value(ctx, active_property, value)
            } else if map.contains_key("@list") {
                let list_items = match map.get("@list") {
                    Some(JsonLdValue::Array(a)) => a.as_slice(),
                    _ => &[],
                };
                compact_array(ctx, active_property.unwrap_or("@list"), list_items, options)
            } else {
                compact_node(ctx, ctx, active_property, map, options)
            }
        }
        JsonLdValue::Array(items) => {
            compact_array(ctx, active_property.unwrap_or("@graph"), items, options)
        }
        other => Ok(other.clone()),
    }
}
