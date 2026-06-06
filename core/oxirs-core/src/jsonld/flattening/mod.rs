//! JSON-LD 1.1 Flattening Algorithm
//!
//! This module implements the W3C JSON-LD 1.1 Flattening specification:
//! <https://www.w3.org/TR/json-ld11-api/#flattening-algorithm>
//!
//! Flattening takes a JSON-LD document and produces a normalized form where all
//! node objects appear at the top level (no nesting). This makes it easy to
//! process the document as a flat list of triples.
//!
//! # Algorithm Overview
//!
//! 1. Expand the input document (resolve all contexts, expand all IRIs)
//! 2. Generate a node map (blank node ID mapping, collecting all nodes by subject)
//! 3. Optionally compact the result with a provided context
//! 4. Return `{"@graph": [...all_nodes_in_node_map...]}` or compacted equivalent

pub mod algorithm;
pub mod node_map;
#[cfg(test)]
mod tests;

pub use algorithm::{
    clone_node_recursively, flatten_internal, is_list_object, is_node_object, is_value_object,
    merge_value,
};
pub use node_map::{
    generate_node_map, node_map_to_flat_array, BlankNodeIdMapper, GraphNodeMap, NodeMap, NodeObject,
};

use crate::jsonld::compaction::{compact, CompactionOptions, JsonLdContext};
use indexmap::IndexMap;
use thiserror::Error;

// Re-export shared types
pub use crate::jsonld::compaction::{JsonLdValue, ProcessingMode};

// ============================================================================
// FlatteningOptions
// ============================================================================

/// Options controlling the JSON-LD 1.1 Flattening algorithm.
#[derive(Debug, Clone)]
pub struct FlatteningOptions {
    /// JSON-LD processing mode (1.0 or 1.1).
    pub processing_mode: ProcessingMode,
    /// If `true` (default), compact single-element arrays to scalar when context allows.
    pub compact_arrays: bool,
    /// If `true`, sort nodes and properties deterministically in the output.
    pub ordered: bool,
    /// Optional base IRI override.
    pub base: Option<String>,
    /// Additional context applied during expansion (not the output compaction context).
    pub expand_context: Option<JsonLdValue>,
    /// If `true`, enable document loader callbacks (currently informational).
    pub document_loader_enabled: bool,
}

impl Default for FlatteningOptions {
    fn default() -> Self {
        Self {
            processing_mode: ProcessingMode::JsonLd11,
            compact_arrays: true,
            ordered: false,
            base: None,
            expand_context: None,
            document_loader_enabled: false,
        }
    }
}

// ============================================================================
// FlatteningError
// ============================================================================

/// Errors that may occur during JSON-LD Flattening.
#[derive(Debug, Error)]
pub enum FlatteningError {
    /// An error occurred during the expansion phase.
    #[error("Expansion error: {0}")]
    ExpansionError(String),

    /// An error occurred during node map generation.
    #[error("Node map error: {0}")]
    NodeMapError(String),

    /// An error occurred during the optional compaction phase.
    #[error("Compaction error: {0}")]
    CompactionError(String),

    /// The input document structure was invalid.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// A cyclic reference between nodes was detected.
    #[error("Cyclic node reference involving subject: {0}")]
    CyclicNodeReference(String),
}

// ============================================================================
// Main entry point: flatten
// ============================================================================

/// Flatten a JSON-LD document according to the W3C JSON-LD 1.1 specification.
///
/// # Arguments
///
/// * `input`   — The JSON-LD document to flatten. Expected to be an array of
///   expanded node objects, or a document that will be treated as an array wrapping.
/// * `context` — Optional compaction context. If provided the flat result is
///   additionally compacted using this context.
/// * `options` — Options controlling processing mode, ordering, etc.
///
/// # Returns
///
/// A JSON-LD document `{"@graph": [...]}` where every node object appears at
/// the top level. If a `context` was provided the result is compacted and
/// includes a `@context` entry.
///
/// # Errors
///
/// Returns a [`FlatteningError`] if expansion, node-map generation, or
/// compaction fails.
pub fn flatten(
    input: &JsonLdValue,
    context: Option<&JsonLdValue>,
    options: &FlatteningOptions,
) -> Result<JsonLdValue, FlatteningError> {
    // Step 1 – Treat the input as an already-expanded document.
    // For a full implementation, expansion would happen here; but since we are
    // operating on the post-expansion JSON-LD value tree (consistent with how
    // the rest of the OxiRS pipeline works), we accept an already-expanded or
    // raw document.  Unknown scalars are wrapped in an array.
    let expanded: Vec<JsonLdValue> = match input {
        JsonLdValue::Array(items) => items.clone(),
        JsonLdValue::Null => Vec::new(),
        // Treat a single object as a one-element array.
        other => vec![other.clone()],
    };

    // Step 2 – Core flattening algorithm (node map + serialisation).
    let flat = flatten_internal(expanded, options)?;

    // Step 3 – Optional compaction with the supplied context.
    if let Some(ctx_value) = context {
        // Build a JsonLdContext from the provided context value.
        let active_ctx = build_context_from_value(ctx_value);
        let compact_opts = CompactionOptions {
            compact_arrays: options.compact_arrays,
            processing_mode: options.processing_mode,
            ordered: options.ordered,
            base: options.base.clone(),
        };

        let compacted = compact(&flat, &active_ctx, &compact_opts)
            .map_err(|e| FlatteningError::CompactionError(e.to_string()))?;

        // Inject the @context entry from the provided context value.
        let mut result_map: IndexMap<String, JsonLdValue> = IndexMap::new();
        result_map.insert("@context".to_string(), ctx_value.clone());

        // Merge the compacted body into the result.
        match compacted {
            JsonLdValue::Object(m) => {
                for (k, v) in m {
                    if k != "@context" {
                        result_map.insert(k, v);
                    }
                }
            }
            other => {
                result_map.insert("@graph".to_string(), other);
            }
        }

        return Ok(JsonLdValue::Object(result_map));
    }

    Ok(flat)
}

// ============================================================================
// Context builder helper
// ============================================================================

/// Build a minimal [`JsonLdContext`] from a JSON-LD context value.
///
/// This handles the common case where the context is a JSON object mapping
/// prefixes/terms to IRI strings.  A full context-processing pipeline would be
/// much more complex; this lightweight version covers the test and practical
/// use-cases needed here.
fn build_context_from_value(ctx: &JsonLdValue) -> JsonLdContext {
    let mut active = JsonLdContext::new();

    match ctx {
        JsonLdValue::Object(map) => {
            for (key, val) in map {
                match (key.as_str(), val) {
                    ("@vocab", JsonLdValue::Str(iri)) => {
                        active.vocab = Some(iri.clone());
                    }
                    ("@base", JsonLdValue::Str(iri)) => {
                        active.base = Some(iri.clone());
                    }
                    ("@language", JsonLdValue::Str(lang)) => {
                        active.language = Some(lang.clone());
                    }
                    (term, JsonLdValue::Str(iri)) if !term.starts_with('@') => {
                        active.add_prefix(term, iri.as_str());
                    }
                    (term, JsonLdValue::Object(term_def_map)) => {
                        if let Some(JsonLdValue::Str(iri)) = term_def_map.get("@id") {
                            active.add_prefix(term, iri.as_str());
                        }
                    }
                    _ => {}
                }
            }
        }
        JsonLdValue::Str(iri) => {
            // A string context is a remote context URL — use as @vocab.
            active.vocab = Some(iri.clone());
        }
        _ => {}
    }

    active
}
