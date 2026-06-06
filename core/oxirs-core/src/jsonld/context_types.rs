//! Core data types for JSON-LD context processing.
//!
//! Defines the JSON node representation ([`JsonNode`]), the active context
//! ([`JsonLdContext`]), term definitions ([`JsonLdTermDefinition`]), the
//! context processor ([`JsonLdContextProcessor`]) and the remote document
//! loading types ([`JsonLdLoadDocumentOptions`], [`JsonLdRemoteDocument`]).

use super::profile::JsonLdProcessingMode;
use oxiri::Iri;
use std::collections::HashMap;
use std::error::Error;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::{Arc, Mutex};

pub(super) type LoadDocumentCallback = dyn Fn(
        &str,
        &JsonLdLoadDocumentOptions,
    ) -> Result<JsonLdRemoteDocument, Box<dyn Error + Send + Sync>>
    + Send
    + Sync
    + UnwindSafe
    + RefUnwindSafe;

// Type alias for the complex remote context cache type
pub(super) type RemoteContextCache = Arc<Mutex<HashMap<String, (Option<Iri<String>>, JsonNode)>>>;

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum JsonNode {
    String(String),
    Number(String),
    Boolean(bool),
    Null,
    Array(Vec<JsonNode>),
    Object(HashMap<String, JsonNode>),
}

#[derive(Default, Clone)]
pub struct JsonLdContext {
    pub base_iri: Option<Iri<String>>,
    pub original_base_url: Option<Iri<String>>,
    pub vocabulary_mapping: Option<String>,
    pub default_language: Option<String>,
    pub default_direction: Option<&'static str>,
    pub term_definitions: HashMap<String, JsonLdTermDefinition>,
    pub previous_context: Option<Box<JsonLdContext>>,
}

impl JsonLdContext {
    pub fn new_empty(original_base_url: Option<Iri<String>>) -> Self {
        JsonLdContext {
            base_iri: original_base_url.clone(),
            original_base_url,
            vocabulary_mapping: None,
            default_language: None,
            default_direction: None,
            term_definitions: HashMap::new(),
            previous_context: None,
        }
    }
}

#[derive(Clone)]
pub struct JsonLdTermDefinition {
    // In the fields, None is unset Some(None) is set to null
    pub iri_mapping: Option<Option<String>>,
    pub prefix_flag: bool,
    pub protected: bool,
    pub reverse_property: bool,
    pub base_url: Option<Iri<String>>,
    pub context: Option<JsonNode>,
    pub container_mapping: &'static [&'static str],
    pub direction_mapping: Option<Option<&'static str>>,
    pub index_mapping: Option<String>,
    pub language_mapping: Option<Option<String>>,
    pub nest_value: Option<String>,
    pub type_mapping: Option<String>,
}

pub struct JsonLdContextProcessor {
    pub processing_mode: JsonLdProcessingMode,
    pub lenient: bool, // Custom option to ignore invalid base IRIs
    pub max_context_recursion: usize,
    pub remote_context_cache: RemoteContextCache,
    pub load_document_callback: Option<Arc<LoadDocumentCallback>>,
}

/// Used to pass various options to the LoadDocumentCallback.
pub struct JsonLdLoadDocumentOptions {
    /// One or more IRIs to use in the request as a profile parameter.
    pub request_profile: super::profile::JsonLdProfileSet,
}

/// Returned information about a remote JSON-LD document or context.
pub struct JsonLdRemoteDocument {
    /// The retrieved document
    pub document: Vec<u8>,
    /// The final URL of the loaded document. This is important to handle HTTP redirects properly
    pub document_url: String,
}
