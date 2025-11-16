//! ReBAC Relationship Management REST API
//!
//! This module provides REST API endpoints for managing ReBAC relationships,
//! allowing administrators to create, read, update, and delete relationship tuples.
//!
//! ## API Endpoints
//!
//! - `POST /api/auth/relationships` - Create new relationship
//! - `GET /api/auth/relationships` - List relationships (with filters)
//! - `DELETE /api/auth/relationships` - Delete relationship
//! - `POST /api/auth/relationships/check` - Check if relationship exists
//! - `GET /api/auth/subjects/:subject/relationships` - List subject's relationships
//! - `GET /api/auth/objects/:object/relationships` - List object's relationships
//! - `POST /api/auth/relationships/batch` - Batch create relationships
//! - `DELETE /api/auth/relationships/batch` - Batch delete relationships

use crate::auth::rebac::{
    CheckRequest, CheckResponse, RebacEvaluator, RelationshipCondition, RelationshipTuple,
};
use crate::auth::AuthUser;
use crate::error::{FusekiError, FusekiResult};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Request to create a new relationship tuple
#[derive(Debug, Deserialize, Serialize)]
pub struct CreateRelationshipRequest {
    /// Subject (e.g., "user:alice")
    pub subject: String,
    /// Relation (e.g., "can_read")
    pub relation: String,
    /// Object (e.g., "dataset:public")
    pub object: String,
    /// Optional condition
    pub condition: Option<RelationshipConditionDto>,
}

/// Request to delete a relationship tuple
#[derive(Debug, Deserialize, Serialize)]
pub struct DeleteRelationshipRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// Request to check if a relationship exists
#[derive(Debug, Deserialize, Serialize)]
pub struct CheckRelationshipRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// Batch create relationships request
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchCreateRequest {
    pub relationships: Vec<CreateRelationshipRequest>,
}

/// Batch delete relationships request
#[derive(Debug, Deserialize, Serialize)]
pub struct BatchDeleteRequest {
    pub relationships: Vec<DeleteRelationshipRequest>,
}

/// Query parameters for listing relationships
#[derive(Debug, Deserialize)]
pub struct ListRelationshipsQuery {
    /// Filter by subject
    pub subject: Option<String>,
    /// Filter by relation
    pub relation: Option<String>,
    /// Filter by object
    pub object: Option<String>,
    /// Pagination: offset
    #[serde(default)]
    pub offset: usize,
    /// Pagination: limit (max 100)
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

/// DTO for relationship condition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RelationshipConditionDto {
    #[serde(rename = "time_window")]
    TimeWindow {
        not_before: Option<DateTime<Utc>>,
        not_after: Option<DateTime<Utc>>,
    },
    #[serde(rename = "ip_address")]
    IpAddress { allowed_ips: Vec<String> },
    #[serde(rename = "attribute")]
    Attribute { key: String, value: String },
}

impl From<RelationshipConditionDto> for RelationshipCondition {
    fn from(dto: RelationshipConditionDto) -> Self {
        match dto {
            RelationshipConditionDto::TimeWindow {
                not_before,
                not_after,
            } => RelationshipCondition::TimeWindow {
                not_before,
                not_after,
            },
            RelationshipConditionDto::IpAddress { allowed_ips } => {
                RelationshipCondition::IpAddress { allowed_ips }
            }
            RelationshipConditionDto::Attribute { key, value } => {
                RelationshipCondition::Attribute { key, value }
            }
        }
    }
}

impl From<RelationshipCondition> for RelationshipConditionDto {
    fn from(cond: RelationshipCondition) -> Self {
        match cond {
            RelationshipCondition::TimeWindow {
                not_before,
                not_after,
            } => RelationshipConditionDto::TimeWindow {
                not_before,
                not_after,
            },
            RelationshipCondition::IpAddress { allowed_ips } => {
                RelationshipConditionDto::IpAddress { allowed_ips }
            }
            RelationshipCondition::Attribute { key, value } => {
                RelationshipConditionDto::Attribute { key, value }
            }
        }
    }
}

/// DTO for relationship tuple response
#[derive(Debug, Serialize)]
pub struct RelationshipDto {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub condition: Option<RelationshipConditionDto>,
}

impl From<RelationshipTuple> for RelationshipDto {
    fn from(tuple: RelationshipTuple) -> Self {
        Self {
            subject: tuple.subject,
            relation: tuple.relation,
            object: tuple.object,
            condition: tuple.condition.map(Into::into),
        }
    }
}

/// Response for list relationships
#[derive(Debug, Serialize)]
pub struct ListRelationshipsResponse {
    pub relationships: Vec<RelationshipDto>,
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
}

/// Response for check relationship
#[derive(Debug, Serialize)]
pub struct CheckRelationshipResponse {
    pub allowed: bool,
    pub reason: Option<String>,
}

/// Response for batch operations
#[derive(Debug, Serialize)]
pub struct BatchOperationResponse {
    pub success_count: usize,
    pub error_count: usize,
    pub errors: Vec<BatchOperationError>,
}

#[derive(Debug, Serialize)]
pub struct BatchOperationError {
    pub index: usize,
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub error: String,
}

/// Create a new relationship tuple
///
/// Requires admin permission
#[tracing::instrument(skip(rebac, user))]
pub async fn create_relationship(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Json(request): Json<CreateRelationshipRequest>,
) -> FusekiResult<impl IntoResponse> {
    // Check if user has admin permission
    if !user.0.roles.contains(&"admin".to_string()) {
        warn!(
            "Non-admin user {} attempted to create relationship",
            user.0.username
        );
        return Err(FusekiError::authorization("Admin permission required"));
    }

    debug!(
        "Creating relationship: {} --{}-> {}",
        request.subject, request.relation, request.object
    );

    let tuple = if let Some(condition_dto) = request.condition {
        RelationshipTuple::with_condition(
            request.subject.clone(),
            request.relation.clone(),
            request.object.clone(),
            condition_dto.into(),
        )
    } else {
        RelationshipTuple::new(
            request.subject.clone(),
            request.relation.clone(),
            request.object.clone(),
        )
    };

    rebac
        .add_tuple(tuple.clone())
        .await
        .map_err(|e| FusekiError::internal(format!("Failed to create relationship: {}", e)))?;

    info!(
        "Relationship created by {}: {} --{}-> {}",
        user.0.username, request.subject, request.relation, request.object
    );

    let response = RelationshipDto::from(tuple);
    Ok((StatusCode::CREATED, Json(response)))
}

/// Delete a relationship tuple
///
/// Requires admin permission
#[tracing::instrument(skip(rebac, user))]
pub async fn delete_relationship(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Json(request): Json<DeleteRelationshipRequest>,
) -> FusekiResult<impl IntoResponse> {
    // Check if user has admin permission
    if !user.0.roles.contains(&"admin".to_string()) {
        warn!(
            "Non-admin user {} attempted to delete relationship",
            user.0.username
        );
        return Err(FusekiError::authorization("Admin permission required"));
    }

    debug!(
        "Deleting relationship: {} --{}-> {}",
        request.subject, request.relation, request.object
    );

    let tuple = RelationshipTuple::new(
        request.subject.clone(),
        request.relation.clone(),
        request.object.clone(),
    );

    rebac
        .remove_tuple(&tuple)
        .await
        .map_err(|e| FusekiError::internal(format!("Failed to delete relationship: {}", e)))?;

    info!(
        "Relationship deleted by {}: {} --{}-> {}",
        user.0.username, request.subject, request.relation, request.object
    );

    Ok(StatusCode::NO_CONTENT)
}

/// Check if a relationship exists
#[tracing::instrument(skip(rebac))]
pub async fn check_relationship(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Json(request): Json<CheckRelationshipRequest>,
) -> FusekiResult<impl IntoResponse> {
    debug!(
        "Checking relationship: {} --{}-> {}",
        request.subject, request.relation, request.object
    );

    let check_request = CheckRequest::new(
        request.subject.clone(),
        request.relation.clone(),
        request.object.clone(),
    );

    let result = rebac
        .check(&check_request)
        .await
        .map_err(|e| FusekiError::internal(format!("Failed to check relationship: {}", e)))?;

    let response = CheckRelationshipResponse {
        allowed: result.allowed,
        reason: result.reason,
    };

    Ok(Json(response))
}

/// List relationships with optional filters
#[tracing::instrument(skip(rebac))]
pub async fn list_relationships(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Query(query): Query<ListRelationshipsQuery>,
) -> FusekiResult<impl IntoResponse> {
    // Check if user has admin or read permission
    if !user.0.roles.contains(&"admin".to_string()) && !user.0.roles.contains(&"viewer".to_string())
    {
        return Err(FusekiError::authorization(
            "Admin or viewer permission required",
        ));
    }

    let limit = query.limit.min(100); // Max 100 per request

    debug!(
        "Listing relationships: subject={:?}, relation={:?}, object={:?}, offset={}, limit={}",
        query.subject, query.relation, query.object, query.offset, limit
    );

    // Get all relationships based on filters
    let mut all_tuples = Vec::new();

    if let Some(subject) = &query.subject {
        let tuples = rebac
            .list_subject_tuples(subject)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to list subject tuples: {}", e)))?;
        all_tuples.extend(tuples);
    } else if let Some(object) = &query.object {
        let tuples = rebac
            .list_object_tuples(object)
            .await
            .map_err(|e| FusekiError::internal(format!("Failed to list object tuples: {}", e)))?;
        all_tuples.extend(tuples);
    } else {
        // If no filter, we'd need to get all tuples
        // For now, return error suggesting to use subject or object filter
        return Err(FusekiError::bad_request(
            "Must provide subject or object filter",
        ));
    }

    // Apply relation filter if specified
    if let Some(relation) = &query.relation {
        all_tuples.retain(|t| &t.relation == relation);
    }

    let total = all_tuples.len();

    // Apply pagination
    let paginated_tuples: Vec<_> = all_tuples
        .into_iter()
        .skip(query.offset)
        .take(limit)
        .map(RelationshipDto::from)
        .collect();

    let response = ListRelationshipsResponse {
        relationships: paginated_tuples,
        total,
        offset: query.offset,
        limit,
    };

    Ok(Json(response))
}

/// List all relationships for a specific subject
#[tracing::instrument(skip(rebac))]
pub async fn list_subject_relationships(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Path(subject): Path<String>,
) -> FusekiResult<impl IntoResponse> {
    // Users can view their own relationships, admins can view any
    let is_own = format!("user:{}", user.0.username) == subject;
    if !is_own && !user.0.roles.contains(&"admin".to_string()) {
        return Err(FusekiError::authorization(
            "Can only view own relationships unless admin",
        ));
    }

    debug!("Listing relationships for subject: {}", subject);

    let tuples = rebac
        .list_subject_tuples(&subject)
        .await
        .map_err(|e| FusekiError::internal(format!("Failed to list subject tuples: {}", e)))?;

    let relationships: Vec<_> = tuples.into_iter().map(RelationshipDto::from).collect();

    Ok(Json(relationships))
}

/// List all relationships for a specific object
#[tracing::instrument(skip(rebac))]
pub async fn list_object_relationships(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Path(object): Path<String>,
) -> FusekiResult<impl IntoResponse> {
    // Check if user has admin or viewer permission
    if !user.0.roles.contains(&"admin".to_string()) && !user.0.roles.contains(&"viewer".to_string())
    {
        return Err(FusekiError::authorization(
            "Admin or viewer permission required",
        ));
    }

    debug!("Listing relationships for object: {}", object);

    let tuples = rebac
        .list_object_tuples(&object)
        .await
        .map_err(|e| FusekiError::internal(format!("Failed to list object tuples: {}", e)))?;

    let relationships: Vec<_> = tuples.into_iter().map(RelationshipDto::from).collect();

    Ok(Json(relationships))
}

/// Batch create relationships
///
/// Requires admin permission
#[tracing::instrument(skip(rebac, user))]
pub async fn batch_create_relationships(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Json(request): Json<BatchCreateRequest>,
) -> FusekiResult<impl IntoResponse> {
    // Check if user has admin permission
    if !user.0.roles.contains(&"admin".to_string()) {
        warn!("Non-admin user {} attempted batch create", user.0.username);
        return Err(FusekiError::authorization("Admin permission required"));
    }

    info!(
        "Batch creating {} relationships",
        request.relationships.len()
    );

    let mut success_count = 0;
    let mut errors = Vec::new();

    for (index, rel_req) in request.relationships.into_iter().enumerate() {
        let tuple = if let Some(condition_dto) = rel_req.condition {
            RelationshipTuple::with_condition(
                rel_req.subject.clone(),
                rel_req.relation.clone(),
                rel_req.object.clone(),
                condition_dto.into(),
            )
        } else {
            RelationshipTuple::new(
                rel_req.subject.clone(),
                rel_req.relation.clone(),
                rel_req.object.clone(),
            )
        };

        match rebac.add_tuple(tuple).await {
            Ok(_) => success_count += 1,
            Err(e) => {
                errors.push(BatchOperationError {
                    index,
                    subject: rel_req.subject,
                    relation: rel_req.relation,
                    object: rel_req.object,
                    error: e.to_string(),
                });
            }
        }
    }

    let response = BatchOperationResponse {
        success_count,
        error_count: errors.len(),
        errors,
    };

    info!(
        "Batch create completed: {} success, {} errors",
        success_count, response.error_count
    );

    Ok(Json(response))
}

/// Batch delete relationships
///
/// Requires admin permission
#[tracing::instrument(skip(rebac, user))]
pub async fn batch_delete_relationships(
    State(rebac): State<Arc<dyn RebacEvaluator>>,
    user: AuthUser,
    Json(request): Json<BatchDeleteRequest>,
) -> FusekiResult<impl IntoResponse> {
    // Check if user has admin permission
    if !user.0.roles.contains(&"admin".to_string()) {
        warn!("Non-admin user {} attempted batch delete", user.0.username);
        return Err(FusekiError::authorization("Admin permission required"));
    }

    info!(
        "Batch deleting {} relationships",
        request.relationships.len()
    );

    let mut success_count = 0;
    let mut errors = Vec::new();

    for (index, rel_req) in request.relationships.into_iter().enumerate() {
        let tuple = RelationshipTuple::new(
            rel_req.subject.clone(),
            rel_req.relation.clone(),
            rel_req.object.clone(),
        );

        match rebac.remove_tuple(&tuple).await {
            Ok(_) => success_count += 1,
            Err(e) => {
                errors.push(BatchOperationError {
                    index,
                    subject: rel_req.subject,
                    relation: rel_req.relation,
                    object: rel_req.object,
                    error: e.to_string(),
                });
            }
        }
    }

    let response = BatchOperationResponse {
        success_count,
        error_count: errors.len(),
        errors,
    };

    info!(
        "Batch delete completed: {} success, {} errors",
        success_count, response.error_count
    );

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::rebac::InMemoryRebacManager;
    use crate::auth::types::{Permission, User};

    fn create_admin_user() -> AuthUser {
        AuthUser(User {
            username: "admin".to_string(),
            roles: vec!["admin".to_string()],
            email: Some("admin@example.com".to_string()),
            full_name: Some("Admin".to_string()),
            last_login: None,
            permissions: vec![Permission::Admin],
        })
    }

    fn create_regular_user() -> AuthUser {
        AuthUser(User {
            username: "alice".to_string(),
            roles: vec!["user".to_string()],
            email: Some("alice@example.com".to_string()),
            full_name: Some("Alice".to_string()),
            last_login: None,
            permissions: vec![],
        })
    }

    #[tokio::test]
    async fn test_create_relationship() {
        let rebac = Arc::new(InMemoryRebacManager::new()) as Arc<dyn RebacEvaluator>;
        let user = create_admin_user();

        let request = CreateRelationshipRequest {
            subject: "user:alice".to_string(),
            relation: "can_read".to_string(),
            object: "dataset:public".to_string(),
            condition: None,
        };

        let result = create_relationship(State(rebac.clone()), user, Json(request)).await;

        assert!(result.is_ok());

        // Verify relationship was created
        let check = CheckRequest::new("user:alice", "can_read", "dataset:public");
        let check_result = rebac.check(&check).await.unwrap();
        assert!(check_result.allowed);
    }

    #[tokio::test]
    async fn test_create_relationship_non_admin() {
        let rebac = Arc::new(InMemoryRebacManager::new()) as Arc<dyn RebacEvaluator>;
        let user = create_regular_user();

        let request = CreateRelationshipRequest {
            subject: "user:alice".to_string(),
            relation: "can_read".to_string(),
            object: "dataset:public".to_string(),
            condition: None,
        };

        let result = create_relationship(State(rebac), user, Json(request)).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_subject_relationships() {
        let rebac = Arc::new(InMemoryRebacManager::new()) as Arc<dyn RebacEvaluator>;

        // Add test relationship
        rebac
            .add_tuple(RelationshipTuple::new(
                "user:alice",
                "can_read",
                "dataset:public",
            ))
            .await
            .unwrap();

        let user = create_admin_user();

        let result =
            list_subject_relationships(State(rebac), user, Path("user:alice".to_string())).await;

        assert!(result.is_ok());
    }
}
