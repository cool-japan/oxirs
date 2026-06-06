//! Axum HTTP handlers exposing the vocabulary registry.
//!
//! These handlers are deliberately scaffolding: the registry lives in
//! request-local storage so the endpoints compile against today's
//! [`AppState`] without touching unrelated fields. A follow-up task will hoist
//! the registry into shared state once the storage layer for vocabularies is
//! settled.

use super::metadata::build_metadata;
use super::registry::VocabularyRegistry;
use super::serializer::{serialize_metadata, VocabFormat};
use crate::server::AppState;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use std::sync::Arc;

/// Negotiate a response format from the request headers.
///
/// JSON-LD and Turtle are matched explicitly; HTML is the default to provide
/// a useful browser experience.
pub fn negotiate_vocab_format(headers: &HeaderMap) -> VocabFormat {
    let accept = headers
        .get("accept")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    if accept.contains("application/ld+json") {
        return VocabFormat::JsonLd;
    }
    if accept.contains("text/turtle") {
        return VocabFormat::Turtle;
    }
    VocabFormat::Html
}

/// `GET /vocab` — list every registered vocabulary.
pub async fn vocab_list_handler(
    State(_state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let registry = VocabularyRegistry::new();
    let format = negotiate_vocab_format(&headers);
    let body = format!("{{\"vocabularies\":[{}]}}", registry.len());
    (StatusCode::OK, [("Content-Type", format.mime_type())], body).into_response()
}

/// `GET /vocab/:id` — vocabulary detail.
pub async fn vocab_detail_handler(
    State(_state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let registry = VocabularyRegistry::new();
    match registry.get(&id) {
        Some(e) => {
            let metadata = build_metadata(e, 0);
            let format = negotiate_vocab_format(&headers);
            let body = serialize_metadata(&metadata, format);
            (StatusCode::OK, [("Content-Type", format.mime_type())], body).into_response()
        }
        None => (StatusCode::NOT_FOUND, "vocabulary not found").into_response(),
    }
}
