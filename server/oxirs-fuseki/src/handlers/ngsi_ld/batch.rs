//! NGSI-LD Batch Operations
//!
//! Implements batch entity operations for efficient bulk processing.

use super::entities::{validate_entity, EntityStore};
use super::types::{BatchError, BatchOperationResult, NgsiEntity, NgsiError, ProblemDetails};
use axum::{
    body::Bytes,
    extract::Query,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use std::sync::Arc;

/// Maximum entities per batch operation
const MAX_BATCH_SIZE: usize = 1000;

/// Batch query parameters
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchQueryParams {
    pub options: Option<String>,
}

impl BatchQueryParams {
    pub fn is_update(&self) -> bool {
        self.options
            .as_ref()
            .map(|o| o.contains("update"))
            .unwrap_or(false)
    }

    pub fn is_replace(&self) -> bool {
        self.options
            .as_ref()
            .map(|o| o.contains("replace"))
            .unwrap_or(false)
    }

    pub fn is_no_overwrite(&self) -> bool {
        self.options
            .as_ref()
            .map(|o| o.contains("noOverwrite"))
            .unwrap_or(false)
    }
}

/// Batch create entities
///
/// POST /ngsi-ld/v1/entityOperations/create
pub async fn batch_create_entities(
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    // Parse entities array
    let entities: Vec<NgsiEntity> = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid entities JSON: {}", e)))?;

    // Check batch size
    if entities.len() > MAX_BATCH_SIZE {
        return Err(NgsiError::InvalidRequest(format!(
            "Batch size {} exceeds maximum {}",
            entities.len(),
            MAX_BATCH_SIZE
        )));
    }

    let mut success = Vec::new();
    let mut errors = Vec::new();

    for mut entity in entities {
        // Validate entity
        if let Err(e) = validate_entity(&entity) {
            errors.push(BatchError {
                entity_id: entity.id.clone(),
                error: ProblemDetails {
                    error_type: e.error_type().to_string(),
                    title: e.to_string(),
                    status: e.status_code(),
                    detail: None,
                    instance: None,
                },
            });
            continue;
        }

        // Check if entity already exists
        match store.entity_exists(&entity.id).await {
            Ok(true) => {
                errors.push(BatchError {
                    entity_id: entity.id.clone(),
                    error: ProblemDetails {
                        error_type: "https://uri.etsi.org/ngsi-ld/errors/AlreadyExists".to_string(),
                        title: format!("Entity {} already exists", entity.id),
                        status: 409,
                        detail: None,
                        instance: None,
                    },
                });
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                errors.push(BatchError {
                    entity_id: entity.id.clone(),
                    error: ProblemDetails {
                        error_type: "https://uri.etsi.org/ngsi-ld/errors/InternalError".to_string(),
                        title: e.to_string(),
                        status: 500,
                        detail: None,
                        instance: None,
                    },
                });
                continue;
            }
        }

        // Set timestamps
        let now = Utc::now();
        entity.created_at = Some(now);
        entity.modified_at = Some(now);

        // Store entity
        match store.store_entity(&entity).await {
            Ok(()) => {
                success.push(entity.id.clone());
            }
            Err(e) => {
                errors.push(BatchError {
                    entity_id: entity.id.clone(),
                    error: ProblemDetails {
                        error_type: e.error_type().to_string(),
                        title: e.to_string(),
                        status: e.status_code(),
                        detail: None,
                        instance: None,
                    },
                });
            }
        }
    }

    // Determine response status
    let status = if errors.is_empty() {
        StatusCode::CREATED
    } else if success.is_empty() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::MULTI_STATUS
    };

    let result = BatchOperationResult { success, errors };

    Ok((status, [("Content-Type", "application/json")], Json(result)).into_response())
}

/// Batch upsert entities
///
/// POST /ngsi-ld/v1/entityOperations/upsert
pub async fn batch_upsert_entities(
    Query(params): Query<BatchQueryParams>,
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    // Parse entities array
    let entities: Vec<NgsiEntity> = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid entities JSON: {}", e)))?;

    // Check batch size
    if entities.len() > MAX_BATCH_SIZE {
        return Err(NgsiError::InvalidRequest(format!(
            "Batch size {} exceeds maximum {}",
            entities.len(),
            MAX_BATCH_SIZE
        )));
    }

    let is_update = params.is_update();
    let is_replace = params.is_replace();
    let no_overwrite = params.is_no_overwrite();

    let mut success = Vec::new();
    let mut errors = Vec::new();

    for mut entity in entities {
        // Validate entity
        if let Err(e) = validate_entity(&entity) {
            errors.push(BatchError {
                entity_id: entity.id.clone(),
                error: ProblemDetails {
                    error_type: e.error_type().to_string(),
                    title: e.to_string(),
                    status: e.status_code(),
                    detail: None,
                    instance: None,
                },
            });
            continue;
        }

        let now = Utc::now();

        match store.entity_exists(&entity.id).await {
            Ok(true) => {
                // Entity exists - update or replace
                if is_replace {
                    entity.modified_at = Some(now);
                    if let Err(e) = store.update_entity(&entity).await {
                        errors.push(BatchError {
                            entity_id: entity.id.clone(),
                            error: ProblemDetails {
                                error_type: e.error_type().to_string(),
                                title: e.to_string(),
                                status: e.status_code(),
                                detail: None,
                                instance: None,
                            },
                        });
                        continue;
                    }
                } else if is_update && !no_overwrite {
                    // Merge with existing
                    if let Ok(Some(mut existing)) = store.get_entity(&entity.id).await {
                        for (name, attr) in entity.properties {
                            existing.properties.insert(name, attr);
                        }
                        existing.modified_at = Some(now);
                        if let Err(e) = store.update_entity(&existing).await {
                            errors.push(BatchError {
                                entity_id: entity.id.clone(),
                                error: ProblemDetails {
                                    error_type: e.error_type().to_string(),
                                    title: e.to_string(),
                                    status: e.status_code(),
                                    detail: None,
                                    instance: None,
                                },
                            });
                            continue;
                        }
                    }
                } else if no_overwrite {
                    // Skip - entity exists and noOverwrite is set
                    continue;
                } else {
                    // Default: replace
                    entity.modified_at = Some(now);
                    if let Err(e) = store.update_entity(&entity).await {
                        errors.push(BatchError {
                            entity_id: entity.id.clone(),
                            error: ProblemDetails {
                                error_type: e.error_type().to_string(),
                                title: e.to_string(),
                                status: e.status_code(),
                                detail: None,
                                instance: None,
                            },
                        });
                        continue;
                    }
                }
                success.push(entity.id.clone());
            }
            Ok(false) => {
                // Entity doesn't exist - create
                entity.created_at = Some(now);
                entity.modified_at = Some(now);
                if let Err(e) = store.store_entity(&entity).await {
                    errors.push(BatchError {
                        entity_id: entity.id.clone(),
                        error: ProblemDetails {
                            error_type: e.error_type().to_string(),
                            title: e.to_string(),
                            status: e.status_code(),
                            detail: None,
                            instance: None,
                        },
                    });
                    continue;
                }
                success.push(entity.id.clone());
            }
            Err(e) => {
                errors.push(BatchError {
                    entity_id: entity.id.clone(),
                    error: ProblemDetails {
                        error_type: "https://uri.etsi.org/ngsi-ld/errors/InternalError".to_string(),
                        title: e.to_string(),
                        status: 500,
                        detail: None,
                        instance: None,
                    },
                });
            }
        }
    }

    let status = if errors.is_empty() {
        StatusCode::NO_CONTENT
    } else if success.is_empty() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::MULTI_STATUS
    };

    if errors.is_empty() {
        Ok(status.into_response())
    } else {
        let result = BatchOperationResult { success, errors };
        Ok((status, [("Content-Type", "application/json")], Json(result)).into_response())
    }
}

/// Batch update entities
///
/// POST /ngsi-ld/v1/entityOperations/update
pub async fn batch_update_entities(
    Query(params): Query<BatchQueryParams>,
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    // Parse entities array
    let entities: Vec<NgsiEntity> = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid entities JSON: {}", e)))?;

    // Check batch size
    if entities.len() > MAX_BATCH_SIZE {
        return Err(NgsiError::InvalidRequest(format!(
            "Batch size {} exceeds maximum {}",
            entities.len(),
            MAX_BATCH_SIZE
        )));
    }

    let no_overwrite = params.is_no_overwrite();

    let mut success = Vec::new();
    let mut errors = Vec::new();

    for entity in entities {
        let now = Utc::now();

        match store.get_entity(&entity.id).await {
            Ok(Some(mut existing)) => {
                // Merge attributes
                for (name, attr) in entity.properties {
                    if no_overwrite && existing.properties.contains_key(&name) {
                        continue;
                    }
                    existing.properties.insert(name, attr);
                }
                existing.modified_at = Some(now);

                if let Err(e) = store.update_entity(&existing).await {
                    errors.push(BatchError {
                        entity_id: entity.id.clone(),
                        error: ProblemDetails {
                            error_type: e.error_type().to_string(),
                            title: e.to_string(),
                            status: e.status_code(),
                            detail: None,
                            instance: None,
                        },
                    });
                    continue;
                }
                success.push(entity.id.clone());
            }
            Ok(None) => {
                errors.push(BatchError {
                    entity_id: entity.id.clone(),
                    error: ProblemDetails {
                        error_type: "https://uri.etsi.org/ngsi-ld/errors/ResourceNotFound"
                            .to_string(),
                        title: format!("Entity {} not found", entity.id),
                        status: 404,
                        detail: None,
                        instance: None,
                    },
                });
            }
            Err(e) => {
                errors.push(BatchError {
                    entity_id: entity.id.clone(),
                    error: ProblemDetails {
                        error_type: "https://uri.etsi.org/ngsi-ld/errors/InternalError".to_string(),
                        title: e.to_string(),
                        status: 500,
                        detail: None,
                        instance: None,
                    },
                });
            }
        }
    }

    let status = if errors.is_empty() {
        StatusCode::NO_CONTENT
    } else if success.is_empty() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::MULTI_STATUS
    };

    if errors.is_empty() {
        Ok(status.into_response())
    } else {
        let result = BatchOperationResult { success, errors };
        Ok((status, [("Content-Type", "application/json")], Json(result)).into_response())
    }
}

/// Batch delete entities
///
/// POST /ngsi-ld/v1/entityOperations/delete
pub async fn batch_delete_entities(
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    // Parse entity IDs array
    let entity_ids: Vec<String> = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid entity IDs JSON: {}", e)))?;

    // Check batch size
    if entity_ids.len() > MAX_BATCH_SIZE {
        return Err(NgsiError::InvalidRequest(format!(
            "Batch size {} exceeds maximum {}",
            entity_ids.len(),
            MAX_BATCH_SIZE
        )));
    }

    let mut success = Vec::new();
    let mut errors = Vec::new();

    for entity_id in entity_ids {
        match store.entity_exists(&entity_id).await {
            Ok(true) => {
                if let Err(e) = store.delete_entity(&entity_id).await {
                    errors.push(BatchError {
                        entity_id: entity_id.clone(),
                        error: ProblemDetails {
                            error_type: e.error_type().to_string(),
                            title: e.to_string(),
                            status: e.status_code(),
                            detail: None,
                            instance: None,
                        },
                    });
                    continue;
                }
                success.push(entity_id);
            }
            Ok(false) => {
                errors.push(BatchError {
                    entity_id: entity_id.clone(),
                    error: ProblemDetails {
                        error_type: "https://uri.etsi.org/ngsi-ld/errors/ResourceNotFound"
                            .to_string(),
                        title: format!("Entity {} not found", entity_id),
                        status: 404,
                        detail: None,
                        instance: None,
                    },
                });
            }
            Err(e) => {
                errors.push(BatchError {
                    entity_id: entity_id.clone(),
                    error: ProblemDetails {
                        error_type: "https://uri.etsi.org/ngsi-ld/errors/InternalError".to_string(),
                        title: e.to_string(),
                        status: 500,
                        detail: None,
                        instance: None,
                    },
                });
            }
        }
    }

    let status = if errors.is_empty() {
        StatusCode::NO_CONTENT
    } else if success.is_empty() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::MULTI_STATUS
    };

    if errors.is_empty() {
        Ok(status.into_response())
    } else {
        let result = BatchOperationResult { success, errors };
        Ok((status, [("Content-Type", "application/json")], Json(result)).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_query_params() {
        let params = BatchQueryParams {
            options: Some("update,noOverwrite".to_string()),
        };

        assert!(params.is_update());
        assert!(params.is_no_overwrite());
        assert!(!params.is_replace());
    }
}
