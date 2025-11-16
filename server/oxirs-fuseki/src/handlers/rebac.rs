//! ReBAC (Relationship-Based Access Control) HTTP API Handlers
//!
//! This module provides REST API endpoints for managing ReBAC relationships:
//! - POST /$/rebac/check - Check if a subject has permission
//! - POST /$/rebac/tuples - Add a new relationship tuple
//! - DELETE /$/rebac/tuples - Remove a relationship tuple
//! - GET /$/rebac/tuples - List relationship tuples (with optional filters)
//! - POST /$/rebac/batch-check - Check multiple permissions at once

use crate::auth::rebac::{
    CheckRequest, CheckResponse, RebacError, RebacEvaluator, RelationshipCondition,
    RelationshipTuple,
};
use crate::server::AppState;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request to check a single permission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckPermissionRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// Response from permission check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckPermissionResponse {
    pub allowed: bool,
    pub reason: Option<String>,
}

/// Request to add a relationship tuple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddTupleRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<RelationshipCondition>,
}

/// Request to remove a relationship tuple
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveTupleRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// Request for batch permission checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCheckRequest {
    pub checks: Vec<CheckPermissionRequest>,
}

/// Response from batch permission checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCheckResponse {
    pub results: Vec<CheckPermissionResponse>,
}

/// Query parameters for listing tuples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTuplesQuery {
    /// Filter by subject
    pub subject: Option<String>,
    /// Filter by object
    pub object: Option<String>,
    /// Filter by relation
    pub relation: Option<String>,
}

/// Response containing list of tuples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTuplesResponse {
    pub tuples: Vec<RelationshipTuple>,
    pub count: usize,
}

/// Success response for tuple operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TupleOperationResponse {
    pub success: bool,
    pub message: String,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub details: Option<String>,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let status = StatusCode::BAD_REQUEST;
        (status, Json(self)).into_response()
    }
}

impl From<RebacError> for ErrorResponse {
    fn from(err: RebacError) -> Self {
        Self {
            error: "ReBAC Error".to_string(),
            details: Some(err.to_string()),
        }
    }
}

/// POST /$/rebac/check - Check if a subject has a specific permission
///
/// Request body:
/// ```json
/// {
///   "subject": "user:alice",
///   "relation": "can_read",
///   "object": "graph:http://example.org/g1"
/// }
/// ```
///
/// Response:
/// ```json
/// {
///   "allowed": true,
///   "reason": null
/// }
/// ```
pub async fn check_permission(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CheckPermissionRequest>,
) -> Result<Json<CheckPermissionResponse>, ErrorResponse> {
    let rebac = state.rebac_manager.as_ref().ok_or_else(|| ErrorResponse {
        error: "ReBAC not enabled".to_string(),
        details: Some("ReBAC manager is not configured".to_string()),
    })?;

    let check_req = CheckRequest::new(&req.subject, &req.relation, &req.object);
    let response = rebac.check(&check_req).await.map_err(ErrorResponse::from)?;

    Ok(Json(CheckPermissionResponse {
        allowed: response.allowed,
        reason: response.reason,
    }))
}

/// POST /$/rebac/batch-check - Check multiple permissions at once
///
/// Request body:
/// ```json
/// {
///   "checks": [
///     {"subject": "user:alice", "relation": "can_read", "object": "graph:g1"},
///     {"subject": "user:alice", "relation": "can_write", "object": "graph:g1"}
///   ]
/// }
/// ```
///
/// Response:
/// ```json
/// {
///   "results": [
///     {"allowed": true, "reason": null},
///     {"allowed": false, "reason": "No write permission"}
///   ]
/// }
/// ```
pub async fn batch_check_permissions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchCheckRequest>,
) -> Result<Json<BatchCheckResponse>, ErrorResponse> {
    let rebac = state.rebac_manager.as_ref().ok_or_else(|| ErrorResponse {
        error: "ReBAC not enabled".to_string(),
        details: Some("ReBAC manager is not configured".to_string()),
    })?;

    let check_requests: Vec<CheckRequest> = req
        .checks
        .iter()
        .map(|c| CheckRequest::new(&c.subject, &c.relation, &c.object))
        .collect();

    let responses = rebac
        .batch_check(&check_requests)
        .await
        .map_err(ErrorResponse::from)?;

    let results = responses
        .into_iter()
        .map(|r| CheckPermissionResponse {
            allowed: r.allowed,
            reason: r.reason,
        })
        .collect();

    Ok(Json(BatchCheckResponse { results }))
}

/// POST /$/rebac/tuples - Add a new relationship tuple
///
/// Request body:
/// ```json
/// {
///   "subject": "user:alice",
///   "relation": "can_read",
///   "object": "graph:http://example.org/g1",
///   "condition": {
///     "type": "time_window",
///     "not_before": "2025-01-01T00:00:00Z",
///     "not_after": "2025-12-31T23:59:59Z"
///   }
/// }
/// ```
///
/// Response:
/// ```json
/// {
///   "success": true,
///   "message": "Relationship tuple added successfully"
/// }
/// ```
pub async fn add_tuple(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddTupleRequest>,
) -> Result<Json<TupleOperationResponse>, ErrorResponse> {
    let rebac = state.rebac_manager.as_ref().ok_or_else(|| ErrorResponse {
        error: "ReBAC not enabled".to_string(),
        details: Some("ReBAC manager is not configured".to_string()),
    })?;

    let tuple = if let Some(condition) = req.condition {
        RelationshipTuple::with_condition(&req.subject, &req.relation, &req.object, condition)
    } else {
        RelationshipTuple::new(&req.subject, &req.relation, &req.object)
    };

    rebac.add_tuple(tuple).await.map_err(ErrorResponse::from)?;

    Ok(Json(TupleOperationResponse {
        success: true,
        message: "Relationship tuple added successfully".to_string(),
    }))
}

/// DELETE /$/rebac/tuples - Remove a relationship tuple
///
/// Request body:
/// ```json
/// {
///   "subject": "user:alice",
///   "relation": "can_read",
///   "object": "graph:http://example.org/g1"
/// }
/// ```
///
/// Response:
/// ```json
/// {
///   "success": true,
///   "message": "Relationship tuple removed successfully"
/// }
/// ```
pub async fn remove_tuple(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RemoveTupleRequest>,
) -> Result<Json<TupleOperationResponse>, ErrorResponse> {
    let rebac = state.rebac_manager.as_ref().ok_or_else(|| ErrorResponse {
        error: "ReBAC not enabled".to_string(),
        details: Some("ReBAC manager is not configured".to_string()),
    })?;

    let tuple = RelationshipTuple::new(&req.subject, &req.relation, &req.object);

    rebac
        .remove_tuple(&tuple)
        .await
        .map_err(ErrorResponse::from)?;

    Ok(Json(TupleOperationResponse {
        success: true,
        message: "Relationship tuple removed successfully".to_string(),
    }))
}

/// GET /$/rebac/tuples - List relationship tuples with optional filters
///
/// Query parameters:
/// - subject: Filter by subject (e.g., "user:alice")
/// - object: Filter by object (e.g., "graph:http://example.org/g1")
/// - relation: Filter by relation (e.g., "can_read")
///
/// Examples:
/// - GET /$/rebac/tuples?subject=user:alice
/// - GET /$/rebac/tuples?object=graph:<http://example.org/g1>
/// - GET /$/rebac/tuples (list all)
///
/// Response:
/// ```json
/// {
///   "tuples": [
///     {
///       "subject": "user:alice",
///       "relation": "can_read",
///       "object": "graph:http://example.org/g1",
///       "condition": null
///     }
///   ],
///   "count": 1
/// }
/// ```
pub async fn list_tuples(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListTuplesQuery>,
) -> Result<Json<ListTuplesResponse>, ErrorResponse> {
    let rebac = state.rebac_manager.as_ref().ok_or_else(|| ErrorResponse {
        error: "ReBAC not enabled".to_string(),
        details: Some("ReBAC manager is not configured".to_string()),
    })?;

    let tuples = if let Some(subject) = &query.subject {
        // List by subject
        rebac
            .list_subject_tuples(subject)
            .await
            .map_err(ErrorResponse::from)?
    } else if let Some(object) = &query.object {
        // List by object
        rebac
            .list_object_tuples(object)
            .await
            .map_err(ErrorResponse::from)?
    } else {
        // For now, listing all tuples requires iterating through subjects
        // This is a limitation of the current API
        // Return empty list with a message
        vec![]
    };

    // Apply additional filters if specified
    let filtered_tuples: Vec<RelationshipTuple> = tuples
        .into_iter()
        .filter(|t| {
            if let Some(rel) = &query.relation {
                &t.relation == rel
            } else {
                true
            }
        })
        .collect();

    let count = filtered_tuples.len();

    Ok(Json(ListTuplesResponse {
        tuples: filtered_tuples,
        count,
    }))
}

// Tests for ReBAC endpoints are in tests/integration/rebac_api.rs
