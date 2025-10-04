//! RDF Patch Handler
//!
//! Implements RDF Patch format for incremental updates to RDF graphs.
//! Based on Apache Jena's RDF Patch specification.
//!
//! POST /patch?graph={graph-uri}
//! Content-Type: application/rdf-patch
//! Body: RDF Patch operations
//!
//! RDF Patch Format:
//! - H (Header): Metadata
//! - PA (Prefix Add): Add prefix
//! - PD (Prefix Delete): Delete prefix
//! - A (Add): Add triple
//! - D (Delete): Delete triple
//! - TC (Transaction Commit): Commit changes
//! - TA (Transaction Abort): Abort changes

use axum::{
    body::Bytes,
    extract::{Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use oxirs_core::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Patch query parameters
#[derive(Debug, Clone, Deserialize)]
pub struct PatchParams {
    /// Target graph URI (default if not specified)
    pub graph: Option<String>,
}

impl PatchParams {
    /// Get target graph name
    pub fn graph_name(&self) -> oxirs_core::model::GraphName {
        match &self.graph {
            Some(uri) if uri != "default" => oxirs_core::model::NamedNode::new(uri)
                .map(oxirs_core::model::GraphName::NamedNode)
                .unwrap_or(oxirs_core::model::GraphName::DefaultGraph),
            _ => oxirs_core::model::GraphName::DefaultGraph,
        }
    }
}

/// Patch operation types
#[derive(Debug, Clone, PartialEq)]
pub enum PatchOperation {
    /// Header metadata
    Header { key: String, value: String },

    /// Add prefix declaration
    PrefixAdd { prefix: String, uri: String },

    /// Delete prefix declaration
    PrefixDelete { prefix: String },

    /// Add triple
    Add(oxirs_core::model::Triple),

    /// Delete triple
    Delete(oxirs_core::model::Triple),

    /// Transaction commit
    TransactionCommit,

    /// Transaction abort
    TransactionAbort,
}

/// RDF Patch document
#[derive(Debug, Clone)]
pub struct RdfPatch {
    /// Patch operations in order
    pub operations: Vec<PatchOperation>,

    /// Prefix mappings accumulated during parsing
    pub prefixes: HashMap<String, String>,

    /// Header metadata
    pub headers: HashMap<String, String>,
}

impl RdfPatch {
    /// Create empty patch
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            prefixes: HashMap::new(),
            headers: HashMap::new(),
        }
    }

    /// Parse RDF Patch from text
    pub fn parse(text: &str) -> Result<Self, PatchError> {
        let mut patch = RdfPatch::new();

        for (line_num, line) in text.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse operation
            let op = parse_operation(line, &patch.prefixes)
                .map_err(|e| PatchError::ParseError(format!("Line {}: {}", line_num + 1, e)))?;

            // Update state based on operation
            match &op {
                PatchOperation::Header { key, value } => {
                    patch.headers.insert(key.clone(), value.clone());
                }
                PatchOperation::PrefixAdd { prefix, uri } => {
                    patch.prefixes.insert(prefix.clone(), uri.clone());
                }
                PatchOperation::PrefixDelete { prefix } => {
                    patch.prefixes.remove(prefix);
                }
                _ => {}
            }

            patch.operations.push(op);
        }

        Ok(patch)
    }

    /// Apply patch to store
    pub fn apply<S: Store>(
        &self,
        store: &S,
        graph: &oxirs_core::model::GraphName,
    ) -> Result<PatchStats, PatchError> {
        let start = Instant::now();
        let mut stats = PatchStats::new();

        // Track transaction state
        let mut in_transaction = false;
        let mut transaction_ops = Vec::new();

        for op in &self.operations {
            match op {
                PatchOperation::Header { .. } => {
                    // Headers are metadata only
                    continue;
                }

                PatchOperation::PrefixAdd { .. } | PatchOperation::PrefixDelete { .. } => {
                    // Prefix operations are handled during parsing
                    stats.prefix_operations += 1;
                }

                PatchOperation::Add(triple) => {
                    if in_transaction {
                        transaction_ops.push(op.clone());
                    } else {
                        apply_add(store, triple, graph)?;
                        stats.triples_added += 1;
                    }
                }

                PatchOperation::Delete(triple) => {
                    if in_transaction {
                        transaction_ops.push(op.clone());
                    } else {
                        apply_delete(store, triple, graph)?;
                        stats.triples_deleted += 1;
                    }
                }

                PatchOperation::TransactionCommit => {
                    if !in_transaction {
                        return Err(PatchError::TransactionError(
                            "TC without transaction start".to_string(),
                        ));
                    }

                    // Apply all transaction operations
                    for tx_op in &transaction_ops {
                        match tx_op {
                            PatchOperation::Add(triple) => {
                                apply_add(store, triple, graph)?;
                                stats.triples_added += 1;
                            }
                            PatchOperation::Delete(triple) => {
                                apply_delete(store, triple, graph)?;
                                stats.triples_deleted += 1;
                            }
                            _ => {}
                        }
                    }

                    transaction_ops.clear();
                    in_transaction = false;
                    stats.transactions_committed += 1;
                }

                PatchOperation::TransactionAbort => {
                    if !in_transaction {
                        return Err(PatchError::TransactionError(
                            "TA without transaction start".to_string(),
                        ));
                    }

                    transaction_ops.clear();
                    in_transaction = false;
                    stats.transactions_aborted += 1;
                }
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        Ok(stats)
    }
}

/// Parse a single patch operation line
fn parse_operation(
    line: &str,
    prefixes: &HashMap<String, String>,
) -> Result<PatchOperation, String> {
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.is_empty() {
        return Err("Empty operation".to_string());
    }

    match parts[0] {
        "H" => {
            if parts.len() < 3 {
                return Err("H requires key and value".to_string());
            }
            Ok(PatchOperation::Header {
                key: parts[1].to_string(),
                value: parts[2..].join(" "),
            })
        }

        "PA" => {
            if parts.len() < 3 {
                return Err("PA requires prefix and URI".to_string());
            }
            let prefix = parts[1].trim_end_matches(':').to_string();
            let uri = parts[2].trim_matches('<').trim_matches('>').to_string();
            Ok(PatchOperation::PrefixAdd { prefix, uri })
        }

        "PD" => {
            if parts.len() < 2 {
                return Err("PD requires prefix".to_string());
            }
            let prefix = parts[1].trim_end_matches(':').to_string();
            Ok(PatchOperation::PrefixDelete { prefix })
        }

        "A" => {
            if parts.len() < 4 {
                return Err("A requires subject, predicate, object".to_string());
            }
            let triple = parse_triple(&parts[1..], prefixes)?;
            Ok(PatchOperation::Add(triple))
        }

        "D" => {
            if parts.len() < 4 {
                return Err("D requires subject, predicate, object".to_string());
            }
            let triple = parse_triple(&parts[1..], prefixes)?;
            Ok(PatchOperation::Delete(triple))
        }

        "TC" => Ok(PatchOperation::TransactionCommit),

        "TA" => Ok(PatchOperation::TransactionAbort),

        _ => Err(format!("Unknown operation: {}", parts[0])),
    }
}

/// Parse triple from parts
fn parse_triple(
    parts: &[&str],
    prefixes: &HashMap<String, String>,
) -> Result<oxirs_core::model::Triple, String> {
    if parts.len() < 3 {
        return Err("Triple requires at least 3 parts".to_string());
    }

    // Parse subject
    let subject = parse_subject(parts[0], prefixes)?;

    // Parse predicate
    let predicate = parse_predicate(parts[1], prefixes)?;

    // Parse object (may include multiple parts for literals)
    let object_parts = &parts[2..];
    let object = parse_object(object_parts, prefixes)?;

    Ok(oxirs_core::model::Triple::new(subject, predicate, object))
}

/// Parse subject from string
fn parse_subject(
    s: &str,
    prefixes: &HashMap<String, String>,
) -> Result<oxirs_core::model::Subject, String> {
    if s.starts_with('<') && s.ends_with('>') {
        // IRI
        let iri = s.trim_matches('<').trim_matches('>');
        oxirs_core::model::NamedNode::new(iri)
            .map(oxirs_core::model::Subject::NamedNode)
            .map_err(|e| format!("Invalid IRI: {}", e))
    } else if s.starts_with("_:") {
        // Blank node
        oxirs_core::model::BlankNode::new(s[2..].to_string())
            .map(oxirs_core::model::Subject::BlankNode)
            .map_err(|e| format!("Invalid blank node: {}", e))
    } else if s.contains(':') {
        // Prefixed name
        expand_prefixed_name(s, prefixes)
            .and_then(|iri| oxirs_core::model::NamedNode::new(&iri).map_err(|e| e.to_string()))
            .map(oxirs_core::model::Subject::NamedNode)
    } else {
        Err(format!("Invalid subject: {}", s))
    }
}

/// Parse predicate from string
fn parse_predicate(
    p: &str,
    prefixes: &HashMap<String, String>,
) -> Result<oxirs_core::model::Predicate, String> {
    if p.starts_with('<') && p.ends_with('>') {
        let iri = p.trim_matches('<').trim_matches('>');
        oxirs_core::model::NamedNode::new(iri)
            .map(oxirs_core::model::Predicate::NamedNode)
            .map_err(|e| format!("Invalid IRI: {}", e))
    } else if p.contains(':') {
        expand_prefixed_name(p, prefixes)
            .and_then(|iri| oxirs_core::model::NamedNode::new(&iri).map_err(|e| e.to_string()))
            .map(oxirs_core::model::Predicate::NamedNode)
    } else {
        Err(format!("Invalid predicate: {}", p))
    }
}

/// Parse object from parts
fn parse_object(
    parts: &[&str],
    prefixes: &HashMap<String, String>,
) -> Result<oxirs_core::model::Object, String> {
    if parts.is_empty() {
        return Err("Empty object".to_string());
    }

    let first = parts[0];

    if first.starts_with('"') {
        // Literal - may span multiple parts
        parse_literal(parts)
    } else if first.starts_with('<') && first.ends_with('>') {
        // IRI
        let iri = first.trim_matches('<').trim_matches('>');
        oxirs_core::model::NamedNode::new(iri)
            .map(oxirs_core::model::Object::NamedNode)
            .map_err(|e| format!("Invalid IRI: {}", e))
    } else if first.starts_with("_:") {
        // Blank node
        oxirs_core::model::BlankNode::new(first[2..].to_string())
            .map(oxirs_core::model::Object::BlankNode)
            .map_err(|e| format!("Invalid blank node: {}", e))
    } else if first.contains(':') {
        // Prefixed name
        expand_prefixed_name(first, prefixes)
            .and_then(|iri| oxirs_core::model::NamedNode::new(&iri).map_err(|e| e.to_string()))
            .map(oxirs_core::model::Object::NamedNode)
    } else {
        Err(format!("Invalid object: {}", first))
    }
}

/// Parse literal from parts
fn parse_literal(parts: &[&str]) -> Result<oxirs_core::model::Object, String> {
    // Join parts and extract literal value
    let joined = parts.join(" ");

    // Simple literal parsing (without language tags or datatypes for now)
    if joined.starts_with('"') {
        let value = joined.trim_matches('"').trim_end_matches(" .");
        Ok(oxirs_core::model::Object::Literal(
            oxirs_core::model::Literal::new_simple_literal(value),
        ))
    } else {
        Err("Invalid literal format".to_string())
    }
}

/// Expand prefixed name to full IRI
fn expand_prefixed_name(name: &str, prefixes: &HashMap<String, String>) -> Result<String, String> {
    if let Some(colon_pos) = name.find(':') {
        let prefix = &name[..colon_pos];
        let local = &name[colon_pos + 1..];

        if let Some(base_uri) = prefixes.get(prefix) {
            Ok(format!("{}{}", base_uri, local))
        } else {
            Err(format!("Unknown prefix: {}", prefix))
        }
    } else {
        Err(format!("Invalid prefixed name: {}", name))
    }
}

/// Apply add operation
fn apply_add<S: Store>(
    store: &S,
    triple: &oxirs_core::model::Triple,
    graph: &oxirs_core::model::GraphName,
) -> Result<(), PatchError> {
    let quad = oxirs_core::model::Quad::new(
        triple.subject().clone(),
        triple.predicate().clone(),
        triple.object().clone(),
        graph.clone(),
    );

    store
        .insert_quad(quad)
        .map_err(|e| PatchError::StoreError(e.to_string()))?;
    Ok(())
}

/// Apply delete operation
fn apply_delete<S: Store>(
    store: &S,
    triple: &oxirs_core::model::Triple,
    graph: &oxirs_core::model::GraphName,
) -> Result<(), PatchError> {
    let quad = oxirs_core::model::Quad::new(
        triple.subject().clone(),
        triple.predicate().clone(),
        triple.object().clone(),
        graph.clone(),
    );

    store
        .remove_quad(&quad)
        .map_err(|e| PatchError::StoreError(e.to_string()))?;
    Ok(())
}

/// Patch error types
#[derive(Debug, thiserror::Error)]
pub enum PatchError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Transaction error: {0}")]
    TransactionError(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl PatchError {
    fn status_code(&self) -> StatusCode {
        match self {
            PatchError::ParseError(_) => StatusCode::BAD_REQUEST,
            PatchError::StoreError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            PatchError::TransactionError(_) => StatusCode::BAD_REQUEST,
            PatchError::BadRequest(_) => StatusCode::BAD_REQUEST,
            PatchError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for PatchError {
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

/// Patch application statistics
#[derive(Debug, Clone, Serialize)]
pub struct PatchStats {
    /// Number of triples added
    pub triples_added: usize,

    /// Number of triples deleted
    pub triples_deleted: usize,

    /// Number of prefix operations
    pub prefix_operations: usize,

    /// Number of transactions committed
    pub transactions_committed: usize,

    /// Number of transactions aborted
    pub transactions_aborted: usize,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Target graph
    pub graph: String,
}

impl PatchStats {
    fn new() -> Self {
        Self {
            triples_added: 0,
            triples_deleted: 0,
            prefix_operations: 0,
            transactions_committed: 0,
            transactions_aborted: 0,
            duration_ms: 0,
            graph: String::new(),
        }
    }
}

/// Handle RDF Patch application
///
/// POST /patch?graph={graph-uri}
/// Content-Type: application/rdf-patch
/// Body: RDF Patch operations
pub async fn handle_patch<S: Store + Send + Sync + 'static>(
    Query(params): Query<PatchParams>,
    State(store): State<Arc<S>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, PatchError> {
    info!("RDF Patch request: graph={:?}", params.graph);

    // 1. Check Content-Type
    if let Some(content_type) = headers.get(header::CONTENT_TYPE) {
        if let Ok(ct) = content_type.to_str() {
            if !ct.contains("rdf-patch") && !ct.contains("text/plain") {
                warn!("Unexpected Content-Type: {}", ct);
            }
        }
    }

    // 2. Parse patch
    let text = std::str::from_utf8(&body)
        .map_err(|e| PatchError::ParseError(format!("UTF-8 error: {}", e)))?;

    let patch = RdfPatch::parse(text)?;
    info!("Parsed {} patch operations", patch.operations.len());

    // 3. Apply patch
    let graph = params.graph_name();
    let mut stats = patch.apply(store.as_ref(), &graph)?;
    stats.graph = params
        .graph
        .clone()
        .unwrap_or_else(|| "default".to_string());

    info!(
        "Patch applied: +{} -{} operations in {}ms",
        stats.triples_added, stats.triples_deleted, stats.duration_ms
    );

    // 4. Return statistics
    Ok((StatusCode::OK, Json(stats)).into_response())
}

/// Server-specific handler (works with AppState)
pub async fn handle_patch_server(
    Query(params): Query<PatchParams>,
    State(state): State<Arc<crate::server::AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match handle_patch(
        Query(params),
        State(Arc::new(state.store.clone())),
        headers,
        body,
    )
    .await
    {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_patch() {
        let patch_text = r#"
H id <urn:uuid:1234>
PA ex: <http://example.org/>
A <http://example.org/alice> <http://example.org/name> "Alice" .
D <http://example.org/bob> <http://example.org/age> "25" .
        "#;

        let patch = RdfPatch::parse(patch_text).unwrap();
        assert_eq!(patch.operations.len(), 4);
    }

    #[test]
    fn test_parse_with_prefixes() {
        let patch_text = r#"
PA ex: <http://example.org/>
A ex:alice ex:name "Alice" .
        "#;

        let patch = RdfPatch::parse(patch_text).unwrap();
        assert_eq!(
            patch.prefixes.get("ex"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_parse_transaction() {
        let patch_text = r#"
A <http://example.org/s1> <http://example.org/p1> "v1" .
A <http://example.org/s2> <http://example.org/p2> "v2" .
TC .
        "#;

        let patch = RdfPatch::parse(patch_text).unwrap();

        // Find TC operation
        let has_tc = patch
            .operations
            .iter()
            .any(|op| matches!(op, PatchOperation::TransactionCommit));
        assert!(has_tc);
    }
}
