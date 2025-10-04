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

/// PUT handler for AppState
pub async fn handle_gsp_put_server(
    Query(params): Query<GspParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
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
    match write::handle_gsp_delete(Query(params), State(Arc::new(state.store.clone()))).await {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}
