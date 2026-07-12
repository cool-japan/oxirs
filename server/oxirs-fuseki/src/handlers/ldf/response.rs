//! Triple Pattern Fragments response construction.
//!
//! A TPF response packages the matched triples together with pagination
//! metadata and the fragment's canonical URI. The fragment URI lets clients
//! refer to the exact slice they fetched and is also used as the subject of
//! the metadata triples emitted alongside the data.

use super::pagination::{PaginationMetadata, PaginationParams};
use super::triple_pattern::TpfQuery;
use crate::server::AppState;
use std::sync::Arc;

/// A single RDF triple in a TPF response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// A Triple Pattern Fragment response containing matched triples and
/// associated metadata.
#[derive(Debug, Clone)]
pub struct TpfResponse {
    pub query: TpfQuery,
    pub triples: Vec<ResponseTriple>,
    pub metadata: PaginationMetadata,
    /// Canonical URI of this fragment (suitable for `hydra:totalItems` etc.).
    pub fragment_uri: String,
}

/// Build a TPF response for the supplied query and pagination.
///
/// In a full implementation this would execute a triple-pattern match against
/// the server's RDF store and apply pagination. This scaffold returns an
/// empty result set; the structural integration with the rest of the server
/// is what matters for the v0.3.1 Track K27 delivery.
pub fn build_tpf_response(
    query: &TpfQuery,
    pagination: &PaginationParams,
    _state: &Arc<AppState>,
) -> TpfResponse {
    let triples: Vec<ResponseTriple> = Vec::new();
    let metadata = PaginationMetadata::new(pagination, 0);
    let fragment_uri = build_fragment_uri(query, pagination);
    TpfResponse {
        query: query.clone(),
        triples,
        metadata,
        fragment_uri,
    }
}

fn build_fragment_uri(query: &TpfQuery, pagination: &PaginationParams) -> String {
    let mut parts = Vec::new();
    if let Some(s) = &query.subject {
        parts.push(format!(
            "subject={}",
            oxirs_core::encoding::percent_encode(s)
        ));
    }
    if let Some(p) = &query.predicate {
        parts.push(format!(
            "predicate={}",
            oxirs_core::encoding::percent_encode(p)
        ));
    }
    if let Some(o) = &query.object {
        parts.push(format!(
            "object={}",
            oxirs_core::encoding::percent_encode(o)
        ));
    }
    parts.push(format!("page={}", pagination.page));
    format!("/ldf?{}", parts.join("&"))
}
