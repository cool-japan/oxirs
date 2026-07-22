//! Server-specific GSP handlers that work with AppState

use super::{read, types::GspParams, write};
use crate::auth::{permissions::PermissionChecker, types::Permission, AuthUser};
use crate::server::AppState;
use axum::{
    body::Bytes,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
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
/// dataset is keyed `"default"` in single-dataset mode; see
/// `AppState::reject_if_read_only` for the shared resolution/guard logic.
fn gsp_read_only_guard(state: &AppState) -> Option<Response> {
    state
        .reject_if_read_only("default", "Graph Store Protocol writes")
        .err()
        .map(IntoResponse::into_response)
}

/// Authorization gate for Graph Store Protocol writes (PUT/POST/DELETE).
///
/// Enforced ONLY when authentication is actually configured (`auth_required`),
/// mirroring the router (which wires the RBAC layer solely when
/// `security.auth_required` is set) and the SPARQL UPDATE handler. With auth
/// disabled (the default) GSP writes serve anonymous callers and the sole
/// write-protection is the dataset `read_only` flag (`gsp_read_only_guard`).
///
/// GSP writes replace/merge/delete whole graphs, so — exactly like SPARQL
/// UPDATE — when auth IS enabled they must never be reachable by an
/// unauthenticated caller:
/// - no authenticated user → `401 Unauthorized`;
/// - a user lacking `Permission::GraphStore` → `403 Forbidden`.
///
/// Returns `Some(response)` when the request must be rejected, `None` when it
/// is authorized to proceed.
fn gsp_authz_guard(auth_required: bool, user: &Option<AuthUser>) -> Option<Response> {
    if !auth_required {
        return None;
    }
    match user {
        Some(AuthUser(u)) => {
            if PermissionChecker::has_permission(u, &Permission::GraphStore) {
                None
            } else {
                Some(
                    (
                        StatusCode::FORBIDDEN,
                        Json(serde_json::json!({
                            "error": "insufficient_permissions",
                            "message": "Graph Store Protocol write permission required"
                        })),
                    )
                        .into_response(),
                )
            }
        }
        None => Some(
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "error": "authentication_required",
                    "message": "Authentication required for Graph Store Protocol writes"
                })),
            )
                .into_response(),
        ),
    }
}

/// PUT handler for AppState
pub async fn handle_gsp_put_server(
    Query(params): Query<GspParams>,
    State(state): State<Arc<AppState>>,
    user: Option<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(resp) = gsp_authz_guard(state.config.security.auth_required, &user) {
        return resp;
    }
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
    user: Option<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(resp) = gsp_authz_guard(state.config.security.auth_required, &user) {
        return resp;
    }
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
    user: Option<AuthUser>,
) -> Response {
    if let Some(resp) = gsp_authz_guard(state.config.security.auth_required, &user) {
        return resp;
    }
    if let Some(resp) = gsp_read_only_guard(&state) {
        return resp;
    }
    match write::handle_gsp_delete(Query(params), State(Arc::new(state.store.clone()))).await {
        Ok(resp) => resp,
        Err(err) => err.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::types::User;

    fn user_with(perms: Vec<Permission>) -> AuthUser {
        AuthUser(User {
            username: "u".to_string(),
            roles: Vec::new(),
            email: None,
            full_name: None,
            last_login: None,
            permissions: perms,
        })
    }

    #[test]
    fn regression_gsp_write_requires_authz() {
        // With authentication enabled the write authorization is enforced.
        // Unauthenticated → 401.
        let resp = gsp_authz_guard(true, &None).expect("must reject");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Authenticated but lacking GraphStore → 403.
        let resp =
            gsp_authz_guard(true, &Some(user_with(vec![Permission::Read]))).expect("must reject");
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        // Authenticated with GraphStore → allowed (guard returns None).
        assert!(gsp_authz_guard(true, &Some(user_with(vec![Permission::GraphStore]))).is_none());
    }

    #[test]
    fn gsp_write_is_anonymous_when_auth_disabled() {
        // With authentication disabled (the default), the GSP write authz gate
        // is a no-op: anonymous callers pass it and write-protection is left to
        // the dataset `read_only` flag (`gsp_read_only_guard`).
        assert!(gsp_authz_guard(false, &None).is_none());
        assert!(gsp_authz_guard(false, &Some(user_with(vec![Permission::Read]))).is_none());
    }
}
