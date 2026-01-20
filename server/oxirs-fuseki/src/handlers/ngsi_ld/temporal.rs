//! NGSI-LD Temporal Entity Operations
//!
//! Implements temporal entity operations for time-series data.

use super::content_neg::{NgsiContentNegotiator, NgsiFormat};
use super::types::{
    NgsiAttribute, NgsiEntity, NgsiError, NgsiProperty, NgsiQueryParams, TemporalQuery,
    TimeRelation,
};
use axum::{
    body::Bytes,
    extract::{Path, Query},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Type alias for temporal entity storage: entity_id -> timestamped snapshots
type TemporalEntityStorage = RwLock<HashMap<String, Vec<(DateTime<Utc>, NgsiEntity)>>>;

/// Temporal entity representation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalEntity {
    /// Entity ID
    #[serde(rename = "@id")]
    pub id: String,

    /// Entity type
    #[serde(rename = "@type")]
    pub entity_type: String,

    /// Temporal properties (array of values with observedAt)
    #[serde(flatten)]
    pub temporal_properties: HashMap<String, Vec<TemporalValue>>,
}

/// Temporal value with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalValue {
    /// Type indicator
    #[serde(rename = "type")]
    pub value_type: String,

    /// Value
    pub value: serde_json::Value,

    /// Observation timestamp
    pub observed_at: DateTime<Utc>,

    /// Instance ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,

    /// Dataset ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
}

/// Temporal entity store trait
#[async_trait::async_trait]
pub trait TemporalEntityStore: Send + Sync {
    /// Get temporal entity
    async fn get_temporal_entity(
        &self,
        id: &str,
        temporal_q: Option<&TemporalQuery>,
    ) -> Result<Option<TemporalEntity>, NgsiError>;

    /// Store temporal entity instance
    async fn store_temporal_instance(&self, id: &str, entity: &NgsiEntity)
        -> Result<(), NgsiError>;

    /// Query temporal entities
    async fn query_temporal_entities(
        &self,
        params: &NgsiQueryParams,
        temporal_q: Option<&TemporalQuery>,
    ) -> Result<Vec<TemporalEntity>, NgsiError>;

    /// Delete temporal entity
    async fn delete_temporal_entity(&self, id: &str) -> Result<(), NgsiError>;

    /// Delete temporal attribute instances
    async fn delete_temporal_attribute(
        &self,
        id: &str,
        attr_name: &str,
        temporal_q: Option<&TemporalQuery>,
    ) -> Result<(), NgsiError>;
}

/// In-memory temporal entity store
pub struct InMemoryTemporalStore {
    entities: TemporalEntityStorage,
}

impl Default for InMemoryTemporalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryTemporalStore {
    pub fn new() -> Self {
        Self {
            entities: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait::async_trait]
impl TemporalEntityStore for InMemoryTemporalStore {
    async fn get_temporal_entity(
        &self,
        id: &str,
        temporal_q: Option<&TemporalQuery>,
    ) -> Result<Option<TemporalEntity>, NgsiError> {
        let entities = self.entities.read().await;

        let instances = match entities.get(id) {
            Some(i) => i,
            None => return Ok(None),
        };

        if instances.is_empty() {
            return Ok(None);
        }

        // Filter by temporal query
        let filtered_instances: Vec<_> = if let Some(tq) = temporal_q {
            instances
                .iter()
                .filter(|(ts, _)| match tq.timerel {
                    TimeRelation::Before => tq.time_at.map(|t| ts < &t).unwrap_or(true),
                    TimeRelation::After => tq.time_at.map(|t| ts > &t).unwrap_or(true),
                    TimeRelation::Between => {
                        let after_start = tq.time_at.map(|t| ts >= &t).unwrap_or(true);
                        let before_end = tq.end_time_at.map(|t| ts <= &t).unwrap_or(true);
                        after_start && before_end
                    }
                })
                .collect()
        } else {
            instances.iter().collect()
        };

        if filtered_instances.is_empty() {
            return Ok(None);
        }

        // Get entity type from first instance
        let entity_type = filtered_instances
            .first()
            .map(|(_, e)| e.entity_type.primary().to_string())
            .unwrap_or_default();

        // Aggregate temporal values
        let mut temporal_properties: HashMap<String, Vec<TemporalValue>> = HashMap::new();

        for (ts, entity) in filtered_instances {
            for (name, attr) in &entity.properties {
                let temporal_value = match attr {
                    NgsiAttribute::Property(prop) => TemporalValue {
                        value_type: "Property".to_string(),
                        value: prop.value.clone(),
                        observed_at: prop.observed_at.unwrap_or(*ts),
                        instance_id: prop.instance_id.clone(),
                        dataset_id: prop.dataset_id.clone(),
                    },
                    NgsiAttribute::Relationship(rel) => TemporalValue {
                        value_type: "Relationship".to_string(),
                        value: serde_json::json!(rel.object),
                        observed_at: rel.observed_at.unwrap_or(*ts),
                        instance_id: rel.instance_id.clone(),
                        dataset_id: rel.dataset_id.clone(),
                    },
                    NgsiAttribute::GeoProperty(geo) => TemporalValue {
                        value_type: "GeoProperty".to_string(),
                        value: serde_json::to_value(&geo.value).unwrap_or_default(),
                        observed_at: geo.observed_at.unwrap_or(*ts),
                        instance_id: geo.instance_id.clone(),
                        dataset_id: geo.dataset_id.clone(),
                    },
                };

                temporal_properties
                    .entry(name.clone())
                    .or_default()
                    .push(temporal_value);
            }
        }

        Ok(Some(TemporalEntity {
            id: id.to_string(),
            entity_type,
            temporal_properties,
        }))
    }

    async fn store_temporal_instance(
        &self,
        id: &str,
        entity: &NgsiEntity,
    ) -> Result<(), NgsiError> {
        let mut entities = self.entities.write().await;
        let timestamp = entity.modified_at.unwrap_or_else(Utc::now);

        entities
            .entry(id.to_string())
            .or_default()
            .push((timestamp, entity.clone()));

        Ok(())
    }

    async fn query_temporal_entities(
        &self,
        params: &NgsiQueryParams,
        temporal_q: Option<&TemporalQuery>,
    ) -> Result<Vec<TemporalEntity>, NgsiError> {
        let entities = self.entities.read().await;

        let mut results = Vec::new();

        for id in entities.keys() {
            // Filter by ID
            if let Some(ref filter_id) = params.id {
                if !filter_id.split(',').any(|i| i.trim() == id) {
                    continue;
                }
            }

            if let Some(entity) = self.get_temporal_entity(id, temporal_q).await? {
                // Filter by type
                if let Some(ref entity_type) = params.entity_type {
                    if entity.entity_type != *entity_type {
                        continue;
                    }
                }
                results.push(entity);
            }
        }

        // Apply pagination
        let offset = params.offset.unwrap_or(0) as usize;
        let limit = params.limit.unwrap_or(100) as usize;

        Ok(results.into_iter().skip(offset).take(limit).collect())
    }

    async fn delete_temporal_entity(&self, id: &str) -> Result<(), NgsiError> {
        let mut entities = self.entities.write().await;
        entities.remove(id);
        Ok(())
    }

    async fn delete_temporal_attribute(
        &self,
        id: &str,
        attr_name: &str,
        temporal_q: Option<&TemporalQuery>,
    ) -> Result<(), NgsiError> {
        let mut entities = self.entities.write().await;

        if let Some(instances) = entities.get_mut(id) {
            for (_, entity) in instances.iter_mut() {
                entity.properties.remove(attr_name);
            }

            // Remove empty entries if temporal query specified
            if temporal_q.is_some() {
                instances.retain(|(ts, _)| {
                    if let Some(tq) = temporal_q {
                        match tq.timerel {
                            TimeRelation::Before => !tq.time_at.map(|t| ts < &t).unwrap_or(false),
                            TimeRelation::After => !tq.time_at.map(|t| ts > &t).unwrap_or(false),
                            TimeRelation::Between => {
                                let in_range = tq.time_at.map(|t| ts >= &t).unwrap_or(true)
                                    && tq.end_time_at.map(|t| ts <= &t).unwrap_or(true);
                                !in_range
                            }
                        }
                    } else {
                        true
                    }
                });
            }
        }

        Ok(())
    }
}

/// Temporal query parameters
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalQueryParams {
    #[serde(flatten)]
    pub base: NgsiQueryParams,

    pub timerel: Option<String>,
    pub timeproperty: Option<String>,

    #[serde(rename = "timeAt")]
    pub time_at: Option<String>,

    #[serde(rename = "endTimeAt")]
    pub end_time_at: Option<String>,

    #[serde(rename = "lastN")]
    pub last_n: Option<u32>,

    pub aggr_method: Option<String>,
    pub aggr_period_duration: Option<String>,
}

impl TemporalQueryParams {
    pub fn to_temporal_query(&self) -> Option<TemporalQuery> {
        let timerel = match self.timerel.as_deref()? {
            "before" => TimeRelation::Before,
            "after" => TimeRelation::After,
            "between" => TimeRelation::Between,
            _ => return None,
        };

        let time_at = self
            .time_at
            .as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let end_time_at = self
            .end_time_at
            .as_ref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        Some(TemporalQuery {
            timerel,
            timeproperty: self.timeproperty.clone(),
            time_at,
            end_time_at,
        })
    }
}

/// Create temporal entity
///
/// POST /ngsi-ld/v1/temporal/entities
pub async fn create_temporal_entity(
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn TemporalEntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Validate content type
    negotiator.validate_content_type(&headers)?;

    // Parse entity
    let entity: NgsiEntity = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid entity JSON: {}", e)))?;

    // Store as temporal instance
    store.store_temporal_instance(&entity.id, &entity).await?;

    Ok((
        StatusCode::CREATED,
        [(
            "Location",
            format!("/ngsi-ld/v1/temporal/entities/{}", entity.id),
        )],
    )
        .into_response())
}

/// Get temporal entity
///
/// GET /ngsi-ld/v1/temporal/entities/:id
pub async fn get_temporal_entity(
    Path(entity_id): Path<String>,
    Query(params): Query<TemporalQueryParams>,
    headers: HeaderMap,
    store: Arc<dyn TemporalEntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();
    let format = negotiator.negotiate_response(&headers)?;

    let temporal_q = params.to_temporal_query();

    let entity = store
        .get_temporal_entity(&entity_id, temporal_q.as_ref())
        .await?
        .ok_or_else(|| NgsiError::NotFound(format!("Temporal entity {} not found", entity_id)))?;

    Ok((
        StatusCode::OK,
        [("Content-Type", format.mime_type())],
        Json(entity),
    )
        .into_response())
}

/// Query temporal entities
///
/// GET /ngsi-ld/v1/temporal/entities
pub async fn query_temporal_entities(
    Query(params): Query<TemporalQueryParams>,
    headers: HeaderMap,
    store: Arc<dyn TemporalEntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();
    let format = negotiator.negotiate_response(&headers)?;

    let temporal_q = params.to_temporal_query();

    let entities = store
        .query_temporal_entities(&params.base, temporal_q.as_ref())
        .await?;

    Ok((
        StatusCode::OK,
        [("Content-Type", format.mime_type())],
        Json(entities),
    )
        .into_response())
}

/// Delete temporal entity
///
/// DELETE /ngsi-ld/v1/temporal/entities/:id
pub async fn delete_temporal_entity(
    Path(entity_id): Path<String>,
    store: Arc<dyn TemporalEntityStore>,
) -> Result<Response, NgsiError> {
    store.delete_temporal_entity(&entity_id).await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Add temporal attribute instance
///
/// POST /ngsi-ld/v1/temporal/entities/:id/attrs
pub async fn add_temporal_attribute(
    Path(entity_id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
    store: Arc<dyn TemporalEntityStore>,
) -> Result<Response, NgsiError> {
    let negotiator = NgsiContentNegotiator::new();

    // Validate content type
    negotiator.validate_content_type(&headers)?;

    // Parse attributes
    let attrs: HashMap<String, NgsiAttribute> = serde_json::from_slice(&body)
        .map_err(|e| NgsiError::InvalidRequest(format!("Invalid attributes JSON: {}", e)))?;

    // Create a minimal entity with the new attributes
    let entity = NgsiEntity {
        id: entity_id.clone(),
        entity_type: super::types::NgsiType::Single("TemporalEntity".to_string()),
        context: None,
        scope: None,
        location: None,
        observation_space: None,
        operation_space: None,
        created_at: None,
        modified_at: Some(Utc::now()),
        properties: attrs,
    };

    store.store_temporal_instance(&entity_id, &entity).await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// Delete temporal attribute
///
/// DELETE /ngsi-ld/v1/temporal/entities/:id/attrs/:attrId
pub async fn delete_temporal_attribute(
    Path((entity_id, attr_id)): Path<(String, String)>,
    Query(params): Query<TemporalQueryParams>,
    store: Arc<dyn TemporalEntityStore>,
) -> Result<Response, NgsiError> {
    let temporal_q = params.to_temporal_query();

    store
        .delete_temporal_attribute(&entity_id, &attr_id, temporal_q.as_ref())
        .await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

#[cfg(test)]
mod tests {
    use super::super::types::NgsiProperty;
    use super::*;

    #[tokio::test]
    async fn test_temporal_store() {
        let store = InMemoryTemporalStore::new();

        // Create entity with temporal data
        let entity = NgsiEntity::new("urn:ngsi-ld:Sensor:S001", "TemperatureSensor").with_property(
            "temperature",
            NgsiProperty::with_observed_at(serde_json::json!(25.5), Utc::now()),
        );

        store
            .store_temporal_instance(&entity.id, &entity)
            .await
            .unwrap();

        // Retrieve temporal entity
        let temporal = store
            .get_temporal_entity("urn:ngsi-ld:Sensor:S001", None)
            .await
            .unwrap();

        assert!(temporal.is_some());
        let te = temporal.unwrap();
        assert_eq!(te.id, "urn:ngsi-ld:Sensor:S001");
        assert!(te.temporal_properties.contains_key("temperature"));
    }

    #[test]
    fn test_temporal_query_params() {
        let params = TemporalQueryParams {
            timerel: Some("after".to_string()),
            time_at: Some("2024-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };

        let tq = params.to_temporal_query();
        assert!(tq.is_some());

        let tq = tq.unwrap();
        assert!(matches!(tq.timerel, TimeRelation::After));
        assert!(tq.time_at.is_some());
    }
}
