//! Linked Data Fragments — Triple Pattern Fragments handlers.
//!
//! The Triple Pattern Fragments (TPF) interface is a W3C Community proposal
//! that allows lightweight access to RDF data by querying individual triple
//! patterns with pagination. The TPF design distributes query workload between
//! client and server, enabling cheap server deployments.
//!
//! Reference: <https://linkeddatafragments.org/specification/triple-pattern-fragments/>

pub mod formats;
pub mod pagination;
pub mod response;
pub mod triple_pattern;

#[cfg(test)]
mod tests;

pub use formats::{negotiate_format, serialize_response, LdfFormat};
pub use pagination::{PaginationMetadata, PaginationParams};
pub use response::{build_tpf_response, ResponseTriple, TpfResponse};
pub use triple_pattern::{parse_tpf_query, TpfQuery};

use crate::server::AppState;
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    response::IntoResponse,
};
use std::sync::Arc;

/// Query parameters accepted by the TPF `GET /ldf` endpoint.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TpfQueryParams {
    /// Subject component of the triple pattern (IRI or blank node).
    pub subject: Option<String>,
    /// Predicate component of the triple pattern (IRI).
    pub predicate: Option<String>,
    /// Object component of the triple pattern (IRI, blank node, or literal).
    pub object: Option<String>,
    /// Requested page (1-indexed). Defaults to `1`.
    pub page: Option<usize>,
    /// Requested page size. Defaults to `DEFAULT_PAGE_SIZE`.
    pub page_size: Option<usize>,
}

/// `GET /ldf` — Triple Pattern Fragments handler.
///
/// Returns a fragment of triples that match the requested pattern. The
/// response format is negotiated through the `Accept` header.
pub async fn ldf_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<TpfQueryParams>,
) -> impl IntoResponse {
    let query = match parse_tpf_query(&params) {
        Ok(q) => q,
        Err(e) => return (axum::http::StatusCode::BAD_REQUEST, e).into_response(),
    };

    let pagination = PaginationParams::from_params(&params);
    let response = build_tpf_response(&query, &pagination, &state);
    let format = negotiate_format(&headers);
    let body = serialize_response(&response, format);

    (
        axum::http::StatusCode::OK,
        [("Content-Type", format.mime_type())],
        body,
    )
        .into_response()
}
