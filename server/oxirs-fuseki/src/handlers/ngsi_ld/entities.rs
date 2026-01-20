//! NGSI-LD Entity Operations
//!
//! Implements CRUD operations for NGSI-LD entities.

use super::content_neg::{NgsiContentNegotiator, NgsiFormat};
use super::converter::{NgsiRdfConverter, NgsiToRdf, RdfToNgsi};
use super::query::NgsiQueryTranslator;
use super::types::{
    NgsiAttribute, NgsiEntity, NgsiError, NgsiProperty, NgsiQueryParams, NgsiRelationship,
};
use axum::{
    body::Bytes,
    extract::{Path, Query},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;

/// Entity store trait for abstraction
#[async_trait::async_trait]
pub trait EntityStore: Send + Sync {
    /// Check if entity exists
    async fn entity_exists(&self, id: &str) -> Result<bool, NgsiError>;

    /// Get entity by ID
    async fn get_entity(&self, id: &str) -> Result<Option<NgsiEntity>, NgsiError>;

    /// Store entity
    async fn store_entity(&self, entity: &NgsiEntity) -> Result<(), NgsiError>;

    /// Update entity
    async fn update_entity(&self, entity: &NgsiEntity) -> Result<(), NgsiError>;

    /// Delete entity
    async fn delete_entity(&self, id: &str) -> Result<(), NgsiError>;

    /// Query entities
    async fn query_entities(&self, params: &NgsiQueryParams) -> Result<Vec<NgsiEntity>, NgsiError>;

    /// Count entities matching query
    async fn count_entities(&self, params: &NgsiQueryParams) -> Result<usize, NgsiError>;
}

/// Create a new entity
///
/// POST /ngsi-ld/v1/entities
pub async fn create_entity(
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Validate content type
    negotiator.validate_content_type(&headers)?;

    // Parse entity from JSON
    let mut entity: NgsiEntity = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid entity JSON: {}", e)))?;

    // Validate entity
    validate_entity(&entity)?;

    // Check if entity already exists
    if store.entity_exists(&entity.id).await? {
        return Err(NgsiError::AlreadyExists(format!(
            "Entity {} already exists",
            entity.id
        )));
    }

    // Set timestamps
    let now = Utc::now();
    entity.created_at = Some(now);
    entity.modified_at = Some(now);

    // Store entity
    store.store_entity(&entity).await?;

    // Return 201 Created with Location header
    Ok((
        StatusCode::CREATED,
        [
            ("Location", format!("/ngsi-ld/v1/entities/{}", entity.id)),
            ("Content-Type", "application/json".to_string()),
        ],
    )
        .into_response())
}

/// Get entity by ID
///
/// GET /ngsi-ld/v1/entities/:id
pub async fn get_entity(
    Path(entity_id): Path<String>,
    Query(params): Query<NgsiQueryParams>,
    headers: HeaderMap,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Negotiate response format
    let format = negotiator.negotiate_response(&headers)?;

    // Get entity
    let entity = store
        .get_entity(&entity_id)
        .await?
        .ok_or_else(|| NgsiError::NotFound(format!("Entity {} not found", entity_id)))?;

    // Apply attribute filter
    let entity = if let Some(ref attrs) = params.attrs {
        filter_entity_attrs(entity, attrs)
    } else {
        entity
    };

    // Format response
    format_entity_response(
        entity,
        format,
        params.is_key_values(),
        params.is_sys_attrs(),
    )
}

/// Query entities
///
/// GET /ngsi-ld/v1/entities
pub async fn query_entities(
    Query(params): Query<NgsiQueryParams>,
    headers: HeaderMap,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Negotiate response format
    let format = negotiator.negotiate_response(&headers)?;

    // Query entities
    let entities = store.query_entities(&params).await?;

    // Get count if requested
    let count = if params.is_count() {
        Some(store.count_entities(&params).await?)
    } else {
        None
    };

    // Apply attribute filter
    let entities: Vec<NgsiEntity> = if let Some(ref attrs) = params.attrs {
        entities
            .into_iter()
            .map(|e| filter_entity_attrs(e, attrs))
            .collect()
    } else {
        entities
    };

    // Format response
    format_entities_response(
        entities,
        format,
        params.is_key_values(),
        params.is_sys_attrs(),
        count,
    )
}

/// Update entity attributes
///
/// PATCH /ngsi-ld/v1/entities/:id/attrs
pub async fn update_entity_attrs(
    Path(entity_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Validate content type
    negotiator.validate_content_type(&headers)?;

    // Get existing entity
    let mut entity = store
        .get_entity(&entity_id)
        .await?
        .ok_or_else(|| NgsiError::NotFound(format!("Entity {} not found", entity_id)))?;

    // Parse patch
    let patch: HashMap<String, NgsiAttribute> = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid patch JSON: {}", e)))?;

    // Apply patch
    for (name, attr) in patch {
        // Skip special attributes
        if name == "@id" || name == "@type" || name == "@context" {
            continue;
        }
        entity.properties.insert(name, attr);
    }

    // Update timestamp
    entity.modified_at = Some(Utc::now());

    // Store updated entity
    store.update_entity(&entity).await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Append entity attributes
///
/// POST /ngsi-ld/v1/entities/:id/attrs
pub async fn append_entity_attrs(
    Path(entity_id): Path<String>,
    Query(params): Query<NgsiQueryParams>,
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Validate content type
    negotiator.validate_content_type(&headers)?;

    // Get existing entity
    let mut entity = store
        .get_entity(&entity_id)
        .await?
        .ok_or_else(|| NgsiError::NotFound(format!("Entity {} not found", entity_id)))?;

    // Parse new attributes
    let new_attrs: HashMap<String, NgsiAttribute> = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid attributes JSON: {}", e)))?;

    // Check for existing attributes (unless noOverwrite is specified)
    let no_overwrite = params
        .options
        .as_ref()
        .map(|o| o.contains("noOverwrite"))
        .unwrap_or(false);

    let mut updated_attrs = Vec::new();
    let mut not_updated_attrs = Vec::new();

    for (name, attr) in new_attrs {
        if name == "@id" || name == "@type" || name == "@context" {
            continue;
        }

        if entity.properties.contains_key(&name) && no_overwrite {
            not_updated_attrs.push(name);
        } else {
            entity.properties.insert(name.clone(), attr);
            updated_attrs.push(name);
        }
    }

    // Update timestamp
    entity.modified_at = Some(Utc::now());

    // Store updated entity
    store.update_entity(&entity).await?;

    if not_updated_attrs.is_empty() {
        Ok(StatusCode::NO_CONTENT.into_response())
    } else {
        // Return 207 Multi-Status with details
        let response = serde_json::json!({
            "updated": updated_attrs,
            "notUpdated": not_updated_attrs.iter().map(|attr| {
                serde_json::json!({
                    "attributeName": attr,
                    "reason": "attribute already exists"
                })
            }).collect::<Vec<_>>()
        });

        Ok((
            StatusCode::MULTI_STATUS,
            [("Content-Type", "application/json")],
            Json(response),
        )
            .into_response())
    }
}

/// Delete entity
///
/// DELETE /ngsi-ld/v1/entities/:id
pub async fn delete_entity(
    Path(entity_id): Path<String>,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    // Check if entity exists
    if !store.entity_exists(&entity_id).await? {
        return Err(NgsiError::NotFound(format!(
            "Entity {} not found",
            entity_id
        )));
    }

    // Delete entity
    store.delete_entity(&entity_id).await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Delete entity attribute
///
/// DELETE /ngsi-ld/v1/entities/:id/attrs/:attrId
pub async fn delete_entity_attr(
    Path((entity_id, attr_id)): Path<(String, String)>,
    Query(params): Query<NgsiQueryParams>,
    store: Arc<dyn EntityStore>,
) -> Result<Response, NgsiError> {
    // Get existing entity
    let mut entity = store
        .get_entity(&entity_id)
        .await?
        .ok_or_else(|| NgsiError::NotFound(format!("Entity {} not found", entity_id)))?;

    // Check if attribute exists
    if !entity.properties.contains_key(&attr_id) {
        return Err(NgsiError::NotFound(format!(
            "Attribute {} not found in entity {}",
            attr_id, entity_id
        )));
    }

    // Handle deleteAll and datasetId
    let delete_all = params.delete_all.unwrap_or(false);
    let dataset_id = params.dataset_id.as_ref();

    if delete_all || dataset_id.is_none() {
        entity.properties.remove(&attr_id);
    } else {
        // Would need to handle multi-attribute deletion by datasetId
        // For now, just remove the attribute
        entity.properties.remove(&attr_id);
    }

    // Update timestamp
    entity.modified_at = Some(Utc::now());

    // Store updated entity
    store.update_entity(&entity).await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Validate entity structure
pub fn validate_entity(entity: &NgsiEntity) -> Result<(), NgsiError> {
    // Check ID format
    if !entity.id.starts_with("urn:") && !entity.id.contains("://") {
        return Err(NgsiError::InvalidRequest(
            "Entity ID must be a URI".to_string(),
        ));
    }

    // Check type is present
    if entity.entity_type.primary().is_empty() {
        return Err(NgsiError::InvalidRequest(
            "Entity type is required".to_string(),
        ));
    }

    // Check property/relationship structure
    for (name, attr) in &entity.properties {
        if name.starts_with('@') {
            continue; // Skip JSON-LD keywords
        }

        match attr {
            NgsiAttribute::Property(prop) => {
                // Property must have a value
                if prop.value.is_null() {
                    return Err(NgsiError::InvalidRequest(format!(
                        "Property {} must have a value",
                        name
                    )));
                }
            }
            NgsiAttribute::Relationship(rel) => {
                // Relationship must have an object
                if rel.object.is_empty() {
                    return Err(NgsiError::InvalidRequest(format!(
                        "Relationship {} must have an object",
                        name
                    )));
                }
            }
            NgsiAttribute::GeoProperty(_) => {
                // GeoProperty validation handled by serde
            }
        }
    }

    Ok(())
}

/// Filter entity attributes
fn filter_entity_attrs(mut entity: NgsiEntity, attrs: &str) -> NgsiEntity {
    let attr_names: Vec<&str> = attrs.split(',').map(|s| s.trim()).collect();

    entity
        .properties
        .retain(|name, _| attr_names.contains(&name.as_str()));

    entity
}

/// Format single entity response
fn format_entity_response(
    entity: NgsiEntity,
    format: NgsiFormat,
    key_values: bool,
    sys_attrs: bool,
) -> Result<Response, NgsiError> {
    let mut response_entity = if key_values {
        simplify_entity(&entity)
    } else {
        serde_json::to_value(&entity)
            .map_err(|e| NgsiError::InternalError(format!("Failed to serialize entity: {}", e)))?
    };

    // Remove system attrs if not requested
    if !sys_attrs {
        if let Some(obj) = response_entity.as_object_mut() {
            obj.remove("createdAt");
            obj.remove("modifiedAt");
        }
    }

    // Add @context for JSON-LD format
    if format == NgsiFormat::JsonLd {
        if let Some(obj) = response_entity.as_object_mut() {
            obj.insert(
                "@context".to_string(),
                serde_json::json!(super::NGSI_LD_CORE_CONTEXT),
            );
        }
    }

    Ok((
        StatusCode::OK,
        [("Content-Type", format.mime_type())],
        Json(response_entity),
    )
        .into_response())
}

/// Format multiple entities response
fn format_entities_response(
    entities: Vec<NgsiEntity>,
    format: NgsiFormat,
    key_values: bool,
    sys_attrs: bool,
    count: Option<usize>,
) -> Result<Response, NgsiError> {
    let response_entities: Vec<serde_json::Value> = entities
        .into_iter()
        .map(|e| {
            let mut entity_json = if key_values {
                simplify_entity(&e)
            } else {
                serde_json::to_value(&e).unwrap_or_default()
            };

            if !sys_attrs {
                if let Some(obj) = entity_json.as_object_mut() {
                    obj.remove("createdAt");
                    obj.remove("modifiedAt");
                }
            }

            entity_json
        })
        .collect();

    let mut headers = vec![("Content-Type", format.mime_type().to_string())];

    if let Some(c) = count {
        headers.push(("NGSILD-Results-Count", c.to_string()));
    }

    use axum::http::header::{HeaderName, HeaderValue};
    let mut response = (StatusCode::OK, Json(response_entities)).into_response();
    for (key, value) in headers {
        if let (Ok(name), Ok(val)) = (key.parse::<HeaderName>(), value.parse::<HeaderValue>()) {
            response.headers_mut().insert(name, val);
        }
    }
    Ok(response)
}

/// Simplify entity to key-values format
fn simplify_entity(entity: &NgsiEntity) -> serde_json::Value {
    let mut simplified = serde_json::Map::new();

    simplified.insert("id".to_string(), serde_json::json!(entity.id));
    simplified.insert(
        "type".to_string(),
        serde_json::json!(entity.entity_type.primary()),
    );

    for (name, attr) in &entity.properties {
        let value = match attr {
            NgsiAttribute::Property(prop) => prop.value.clone(),
            NgsiAttribute::Relationship(rel) => serde_json::json!(rel.object),
            NgsiAttribute::GeoProperty(geo) => serde_json::to_value(&geo.value).unwrap_or_default(),
        };
        simplified.insert(name.clone(), value);
    }

    serde_json::Value::Object(simplified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::RwLock;

    /// In-memory entity store for testing
    struct TestEntityStore {
        entities: RwLock<HashMap<String, NgsiEntity>>,
    }

    impl TestEntityStore {
        fn new() -> Self {
            Self {
                entities: RwLock::new(HashMap::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl EntityStore for TestEntityStore {
        async fn entity_exists(&self, id: &str) -> Result<bool, NgsiError> {
            Ok(self.entities.read().await.contains_key(id))
        }

        async fn get_entity(&self, id: &str) -> Result<Option<NgsiEntity>, NgsiError> {
            Ok(self.entities.read().await.get(id).cloned())
        }

        async fn store_entity(&self, entity: &NgsiEntity) -> Result<(), NgsiError> {
            self.entities
                .write()
                .await
                .insert(entity.id.clone(), entity.clone());
            Ok(())
        }

        async fn update_entity(&self, entity: &NgsiEntity) -> Result<(), NgsiError> {
            self.entities
                .write()
                .await
                .insert(entity.id.clone(), entity.clone());
            Ok(())
        }

        async fn delete_entity(&self, id: &str) -> Result<(), NgsiError> {
            self.entities.write().await.remove(id);
            Ok(())
        }

        async fn query_entities(
            &self,
            params: &NgsiQueryParams,
        ) -> Result<Vec<NgsiEntity>, NgsiError> {
            let entities = self.entities.read().await;
            let mut result: Vec<NgsiEntity> = entities.values().cloned().collect();

            // Filter by type
            if let Some(ref entity_type) = params.entity_type {
                result.retain(|e| e.entity_type.primary() == entity_type);
            }

            // Filter by ID
            if let Some(ref id) = params.id {
                let ids: Vec<&str> = id.split(',').collect();
                result.retain(|e| ids.contains(&e.id.as_str()));
            }

            // Apply pagination
            let offset = params.offset.unwrap_or(0) as usize;
            let limit = params.limit.unwrap_or(100) as usize;

            Ok(result.into_iter().skip(offset).take(limit).collect())
        }

        async fn count_entities(&self, params: &NgsiQueryParams) -> Result<usize, NgsiError> {
            let entities = self.query_entities(params).await?;
            Ok(entities.len())
        }
    }

    #[test]
    fn test_validate_entity_valid() {
        let entity = NgsiEntity::new("urn:ngsi-ld:Vehicle:A123", "Vehicle")
            .with_property("speed", NgsiProperty::new(serde_json::json!(80)));

        assert!(validate_entity(&entity).is_ok());
    }

    #[test]
    fn test_validate_entity_invalid_id() {
        let entity = NgsiEntity::new("invalid-id", "Vehicle");
        assert!(validate_entity(&entity).is_err());
    }

    #[test]
    fn test_filter_entity_attrs() {
        let entity = NgsiEntity::new("urn:ngsi-ld:Vehicle:A123", "Vehicle")
            .with_property("speed", NgsiProperty::new(serde_json::json!(80)))
            .with_property("temperature", NgsiProperty::new(serde_json::json!(25)));

        let filtered = filter_entity_attrs(entity, "speed");

        assert!(filtered.properties.contains_key("speed"));
        assert!(!filtered.properties.contains_key("temperature"));
    }

    #[test]
    fn test_simplify_entity() {
        let entity = NgsiEntity::new("urn:ngsi-ld:Vehicle:A123", "Vehicle")
            .with_property("speed", NgsiProperty::new(serde_json::json!(80)));

        let simplified = simplify_entity(&entity);

        assert_eq!(simplified["id"], "urn:ngsi-ld:Vehicle:A123");
        assert_eq!(simplified["type"], "Vehicle");
        assert_eq!(simplified["speed"], 80);
    }

    #[tokio::test]
    async fn test_entity_store_operations() {
        let store = TestEntityStore::new();

        // Create entity
        let entity = NgsiEntity::new("urn:ngsi-ld:Vehicle:A123", "Vehicle")
            .with_property("speed", NgsiProperty::new(serde_json::json!(80)));

        store.store_entity(&entity).await.unwrap();

        // Check exists
        assert!(store
            .entity_exists("urn:ngsi-ld:Vehicle:A123")
            .await
            .unwrap());

        // Get entity
        let retrieved = store.get_entity("urn:ngsi-ld:Vehicle:A123").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "urn:ngsi-ld:Vehicle:A123");

        // Delete entity
        store
            .delete_entity("urn:ngsi-ld:Vehicle:A123")
            .await
            .unwrap();
        assert!(!store
            .entity_exists("urn:ngsi-ld:Vehicle:A123")
            .await
            .unwrap());
    }
}
