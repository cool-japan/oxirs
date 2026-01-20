//! Prefix Management Handler
//!
//! Provides REST API for managing namespace prefixes.
//! Based on Apache Jena Fuseki's prefix management.
//!
//! GET /$/prefixes - List all prefixes
//! GET /$/prefixes/{prefix} - Get specific prefix URI
//! POST /$/prefixes - Add new prefix
//! PUT /$/prefixes/{prefix} - Update prefix URI
//! DELETE /$/prefixes/{prefix} - Delete prefix

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Prefix store with thread-safe access
#[derive(Clone)]
pub struct PrefixStore {
    /// Prefix to URI mappings
    prefixes: Arc<RwLock<HashMap<String, String>>>,
}

impl PrefixStore {
    /// Create new prefix store with well-known prefixes
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add well-known prefixes
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
        prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        prefixes.insert(
            "dc".to_string(),
            "http://purl.org/dc/elements/1.1/".to_string(),
        );
        prefixes.insert(
            "dcterms".to_string(),
            "http://purl.org/dc/terms/".to_string(),
        );
        prefixes.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );

        Self {
            prefixes: Arc::new(RwLock::new(prefixes)),
        }
    }

    /// Get all prefixes
    pub fn list(&self) -> Result<HashMap<String, String>, PrefixError> {
        self.prefixes
            .read()
            .map(|p| p.clone())
            .map_err(|e| PrefixError::Internal(format!("Lock error: {}", e)))
    }

    /// Get specific prefix URI
    pub fn get(&self, prefix: &str) -> Result<Option<String>, PrefixError> {
        self.prefixes
            .read()
            .map(|p| p.get(prefix).cloned())
            .map_err(|e| PrefixError::Internal(format!("Lock error: {}", e)))
    }

    /// Add or update prefix
    pub fn set(&self, prefix: String, uri: String) -> Result<(), PrefixError> {
        validate_prefix(&prefix)?;
        validate_uri(&uri)?;

        self.prefixes
            .write()
            .map(|mut p| {
                p.insert(prefix, uri);
            })
            .map_err(|e| PrefixError::Internal(format!("Lock error: {}", e)))
    }

    /// Delete prefix
    pub fn delete(&self, prefix: &str) -> Result<bool, PrefixError> {
        self.prefixes
            .write()
            .map(|mut p| p.remove(prefix).is_some())
            .map_err(|e| PrefixError::Internal(format!("Lock error: {}", e)))
    }

    /// Check if prefix exists
    pub fn exists(&self, prefix: &str) -> Result<bool, PrefixError> {
        self.prefixes
            .read()
            .map(|p| p.contains_key(prefix))
            .map_err(|e| PrefixError::Internal(format!("Lock error: {}", e)))
    }

    /// Expand prefixed name to full URI
    pub fn expand(&self, prefixed_name: &str) -> Result<Option<String>, PrefixError> {
        if let Some(colon_pos) = prefixed_name.find(':') {
            let prefix = &prefixed_name[..colon_pos];
            let local = &prefixed_name[colon_pos + 1..];

            self.get(prefix)
                .map(|uri_opt| uri_opt.map(|uri| format!("{}{}", uri, local)))
        } else {
            Ok(None)
        }
    }
}

impl Default for PrefixStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate prefix name
fn validate_prefix(prefix: &str) -> Result<(), PrefixError> {
    if prefix.is_empty() {
        return Err(PrefixError::BadRequest(
            "Prefix cannot be empty".to_string(),
        ));
    }

    // Check for valid characters (alphanumeric, -, _)
    if !prefix
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(PrefixError::BadRequest(format!(
            "Invalid prefix '{}': must contain only alphanumeric, -, _",
            prefix
        )));
    }

    // Must start with letter
    if !prefix.chars().next().is_some_and(|c| c.is_alphabetic()) {
        return Err(PrefixError::BadRequest(format!(
            "Invalid prefix '{}': must start with letter",
            prefix
        )));
    }

    Ok(())
}

/// Validate URI
fn validate_uri(uri: &str) -> Result<(), PrefixError> {
    if uri.is_empty() {
        return Err(PrefixError::BadRequest("URI cannot be empty".to_string()));
    }

    // Basic URI validation - must be absolute
    if !uri.starts_with("http://") && !uri.starts_with("https://") && !uri.starts_with("urn:") {
        return Err(PrefixError::BadRequest(format!(
            "Invalid URI '{}': must be absolute (http://, https://, or urn:)",
            uri
        )));
    }

    Ok(())
}

/// Prefix error types
#[derive(Debug, thiserror::Error)]
pub enum PrefixError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl PrefixError {
    fn status_code(&self) -> StatusCode {
        match self {
            PrefixError::NotFound(_) => StatusCode::NOT_FOUND,
            PrefixError::BadRequest(_) => StatusCode::BAD_REQUEST,
            PrefixError::Conflict(_) => StatusCode::CONFLICT,
            PrefixError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for PrefixError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.to_string();

        (
            status,
            Json(serde_json::json!({
                "error": message,
                "status": status.as_u16(),
            })),
        )
            .into_response()
    }
}

/// Prefix entry for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixEntry {
    pub prefix: String,
    pub uri: String,
}

/// Request body for adding/updating prefix
#[derive(Debug, Clone, Deserialize)]
pub struct PrefixRequest {
    pub prefix: String,
    pub uri: String,
}

/// List all prefixes
///
/// GET /$/prefixes
pub async fn list_prefixes(State(store): State<Arc<PrefixStore>>) -> Result<Response, PrefixError> {
    info!("List prefixes request");

    let prefixes = store.list()?;
    let entries: Vec<PrefixEntry> = prefixes
        .into_iter()
        .map(|(prefix, uri)| PrefixEntry { prefix, uri })
        .collect();

    debug!("Found {} prefixes", entries.len());

    Ok((StatusCode::OK, Json(entries)).into_response())
}

/// Get specific prefix
///
/// GET /$/prefixes/{prefix}
pub async fn get_prefix(
    Path(prefix): Path<String>,
    State(store): State<Arc<PrefixStore>>,
) -> Result<Response, PrefixError> {
    info!("Get prefix request: {}", prefix);

    match store.get(&prefix)? {
        Some(uri) => {
            debug!("Found prefix '{}' -> '{}'", prefix, uri);
            Ok((StatusCode::OK, Json(PrefixEntry { prefix, uri })).into_response())
        }
        None => {
            debug!("Prefix '{}' not found", prefix);
            Err(PrefixError::NotFound(format!(
                "Prefix '{}' not found",
                prefix
            )))
        }
    }
}

/// Add new prefix
///
/// POST /$/prefixes
/// Body: `{ "prefix": "ex", "uri": "http://example.org/" }`
pub async fn add_prefix(
    State(store): State<Arc<PrefixStore>>,
    Json(req): Json<PrefixRequest>,
) -> Result<Response, PrefixError> {
    info!("Add prefix request: {} -> {}", req.prefix, req.uri);

    // Check if already exists
    if store.exists(&req.prefix)? {
        return Err(PrefixError::Conflict(format!(
            "Prefix '{}' already exists",
            req.prefix
        )));
    }

    store.set(req.prefix.clone(), req.uri.clone())?;

    debug!("Added prefix '{}' -> '{}'", req.prefix, req.uri);

    Ok((
        StatusCode::CREATED,
        Json(PrefixEntry {
            prefix: req.prefix,
            uri: req.uri,
        }),
    )
        .into_response())
}

/// Update prefix
///
/// PUT /$/prefixes/{prefix}
/// Body: `{ "uri": "http://example.org/new/" }`
pub async fn update_prefix(
    Path(prefix): Path<String>,
    State(store): State<Arc<PrefixStore>>,
    Json(req): Json<serde_json::Value>,
) -> Result<Response, PrefixError> {
    info!("Update prefix request: {}", prefix);

    // Check if exists
    if !store.exists(&prefix)? {
        return Err(PrefixError::NotFound(format!(
            "Prefix '{}' not found",
            prefix
        )));
    }

    // Extract URI from request
    let uri = req
        .get("uri")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PrefixError::BadRequest("Missing 'uri' field".to_string()))?
        .to_string();

    store.set(prefix.clone(), uri.clone())?;

    debug!("Updated prefix '{}' -> '{}'", prefix, uri);

    Ok((StatusCode::OK, Json(PrefixEntry { prefix, uri })).into_response())
}

/// Delete prefix
///
/// DELETE /$/prefixes/{prefix}
pub async fn delete_prefix(
    Path(prefix): Path<String>,
    State(store): State<Arc<PrefixStore>>,
) -> Result<Response, PrefixError> {
    info!("Delete prefix request: {}", prefix);

    if store.delete(&prefix)? {
        debug!("Deleted prefix '{}'", prefix);
        Ok((StatusCode::NO_CONTENT, ()).into_response())
    } else {
        debug!("Prefix '{}' not found", prefix);
        Err(PrefixError::NotFound(format!(
            "Prefix '{}' not found",
            prefix
        )))
    }
}

/// Expand prefixed name
///
/// POST /$/prefixes/expand
/// Body: { "name": "rdf:type" }
pub async fn expand_prefix(
    State(store): State<Arc<PrefixStore>>,
    Json(req): Json<serde_json::Value>,
) -> Result<Response, PrefixError> {
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PrefixError::BadRequest("Missing 'name' field".to_string()))?;

    info!("Expand prefix request: {}", name);

    match store.expand(name)? {
        Some(uri) => {
            debug!("Expanded '{}' -> '{}'", name, uri);
            Ok((
                StatusCode::OK,
                Json(serde_json::json!({
                    "name": name,
                    "uri": uri,
                })),
            )
                .into_response())
        }
        None => Err(PrefixError::BadRequest(format!(
            "Cannot expand '{}': invalid format or unknown prefix",
            name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_store_new() {
        let store = PrefixStore::new();
        let prefixes = store.list().unwrap();

        // Should have well-known prefixes
        assert!(prefixes.contains_key("rdf"));
        assert!(prefixes.contains_key("rdfs"));
        assert!(prefixes.contains_key("owl"));
        assert!(prefixes.contains_key("xsd"));
    }

    #[test]
    fn test_prefix_store_set_get() {
        let store = PrefixStore::new();

        store
            .set("ex".to_string(), "http://example.org/".to_string())
            .unwrap();

        let uri = store.get("ex").unwrap();
        assert_eq!(uri, Some("http://example.org/".to_string()));
    }

    #[test]
    fn test_prefix_store_delete() {
        let store = PrefixStore::new();

        store
            .set("ex".to_string(), "http://example.org/".to_string())
            .unwrap();
        assert!(store.exists("ex").unwrap());

        let deleted = store.delete("ex").unwrap();
        assert!(deleted);
        assert!(!store.exists("ex").unwrap());
    }

    #[test]
    fn test_prefix_expand() {
        let store = PrefixStore::new();

        store
            .set("ex".to_string(), "http://example.org/".to_string())
            .unwrap();

        let expanded = store.expand("ex:foo").unwrap();
        assert_eq!(expanded, Some("http://example.org/foo".to_string()));

        let expanded_rdf = store.expand("rdf:type").unwrap();
        assert_eq!(
            expanded_rdf,
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
        );
    }

    #[test]
    fn test_validate_prefix() {
        assert!(validate_prefix("ex").is_ok());
        assert!(validate_prefix("foo-bar").is_ok());
        assert!(validate_prefix("foo_bar").is_ok());
        assert!(validate_prefix("foo123").is_ok());

        assert!(validate_prefix("").is_err());
        assert!(validate_prefix("123foo").is_err()); // Must start with letter
        assert!(validate_prefix("foo:bar").is_err()); // No colons
        assert!(validate_prefix("foo bar").is_err()); // No spaces
    }

    #[test]
    fn test_validate_uri() {
        assert!(validate_uri("http://example.org/").is_ok());
        assert!(validate_uri("https://example.org/").is_ok());
        assert!(validate_uri("urn:example:123").is_ok());

        assert!(validate_uri("").is_err());
        assert!(validate_uri("example.org").is_err()); // Not absolute
        assert!(validate_uri("//example.org").is_err()); // Not absolute
    }

    #[test]
    fn test_prefix_entry_serialization() {
        let entry = PrefixEntry {
            prefix: "ex".to_string(),
            uri: "http://example.org/".to_string(),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"prefix\""));
        assert!(json.contains("\"uri\""));
        assert!(json.contains("ex"));
        assert!(json.contains("http://example.org/"));
    }
}
