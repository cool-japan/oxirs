//! Server-specific GSP handlers that work with AppState

use super::{read, types::GspParams, write};
use crate::server::AppState;
use axum::{
    body::Bytes,
    extract::{Query, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

/// GET handler for AppState
pub async fn handle_gsp_get_server(
    Query(params): Query<GspParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    match read::handle_gsp_get(Query(params), State(Arc::new(state.store.clone())), headers).await {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

/// HEAD handler for AppState
pub async fn handle_gsp_head_server(
    Query(params): Query<GspParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    match read::handle_gsp_head(Query(params), State(Arc::new(state.store.clone())), headers).await
    {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

/// OPTIONS handler for AppState
pub async fn handle_gsp_options_server() -> Response {
    read::handle_gsp_options().await.into_response()
}

/// Reject a Graph Store Protocol write when the (default) dataset is read-only.
///
/// GSP `PUT`/`POST`/`DELETE` mutate the store just like SPARQL UPDATE, so a
/// read-only public deployment must block them too (HTTP 403). The default
/// dataset is keyed `"default"` in single-dataset mode.
fn gsp_read_only_guard(state: &AppState) -> Option<Response> {
    if state.is_dataset_read_only("default") {
        return Some(
            crate::error::FusekiError::forbidden(
                "Dataset is read-only; Graph Store Protocol writes are not permitted",
            )
            .into_response(),
        );
    }
    None
}

/// PUT handler for AppState
pub async fn handle_gsp_put_server(
    Query(params): Query<GspParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(resp) = gsp_read_only_guard(&state) {
        return resp;
    }
    match write::handle_gsp_put(
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

/// POST handler for AppState
pub async fn handle_gsp_post_server(
    Query(params): Query<GspParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(resp) = gsp_read_only_guard(&state) {
        return resp;
    }
    match write::handle_gsp_post(
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

/// DELETE handler for AppState
pub async fn handle_gsp_delete_server(
    Query(params): Query<GspParams>,
    State(state): State<Arc<AppState>>,
) -> Response {
    if let Some(resp) = gsp_read_only_guard(&state) {
        return resp;
    }
    match write::handle_gsp_delete(Query(params), State(Arc::new(state.store.clone()))).await {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}
