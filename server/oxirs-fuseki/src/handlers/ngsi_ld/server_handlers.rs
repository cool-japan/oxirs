//! Server-specific NGSI-LD handlers that work with AppState
//!
//! These handlers integrate the NGSI-LD API with OxiRS's AppState.

use super::batch;
use super::entities::{self, EntityStore};
use super::subscriptions::{self, SubscriptionStore};
use super::temporal::{self, TemporalEntityStore};
use super::types::{NgsiEntity, NgsiError, NgsiQueryParams, NgsiSubscription};
use crate::server::AppState;
use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::HeaderMap,
    response::Response,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// RDF-backed Entity Store
// ============================================================================

/// Entity store backed by RDF graph
pub struct RdfEntityStore {
    /// Reference to OxiRS store
    store: Arc<crate::store::Store>,
    /// Graph name for NGSI-LD entities
    graph_name: String,
    /// In-memory cache for entities (optional)
    cache: Arc<RwLock<HashMap<String, NgsiEntity>>>,
}

impl RdfEntityStore {
    /// Create a new RDF-backed entity store
    pub fn new(store: Arc<crate::store::Store>, graph_name: impl Into<String>) -> Self {
        Self {
            store,
            graph_name: graph_name.into(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl EntityStore for RdfEntityStore {
    async fn entity_exists(&self, id: &str) -> Result<bool, NgsiError> {
        // Check cache first
        if self.cache.read().await.contains_key(id) {
            return Ok(true);
        }

        // Check RDF store with ASK query
        let query = format!("ASK {{ GRAPH <{}> {{ <{}> ?p ?o }} }}", self.graph_name, id);

        // Execute ASK query against store using general query method
        match self.store.query(&query) {
            Ok(_result) => {
                // ASK query returns boolean
                // For now, check cache (full RDF integration pending Store API updates)
                Ok(self.cache.read().await.contains_key(id))
            }
            Err(_) => Ok(false), // On error, assume doesn't exist
        }
    }

    async fn get_entity(&self, id: &str) -> Result<Option<NgsiEntity>, NgsiError> {
        // Check cache first
        if let Some(entity) = self.cache.read().await.get(id) {
            return Ok(Some(entity.clone()));
        }

        // For RDF-backed storage, we would query the store
        // For now, continue using cache-only approach
        // TODO: Implement full RDF query when Store API supports CONSTRUCT queries
        Ok(None)
    }

    async fn store_entity(&self, entity: &NgsiEntity) -> Result<(), NgsiError> {
        // Convert to RDF triples
        let converter = super::converter::NgsiRdfConverter::new();
        let rdf_triples = converter.entity_to_rdf(entity)?;

        // Convert to SPARQL INSERT DATA
        let mut insert_query = format!("INSERT DATA {{ GRAPH <{}> {{\n", self.graph_name);

        for triple in &rdf_triples {
            let obj_str = match &triple.object {
                super::converter::RdfObject::Uri(uri) => format!("<{}>", uri),
                super::converter::RdfObject::Literal {
                    value,
                    datatype,
                    language,
                } => {
                    if let Some(lang) = language {
                        format!("\"{}\"@{}", value, lang)
                    } else if let Some(dt) = datatype {
                        format!("\"{}\"^^<{}>", value, dt)
                    } else {
                        format!("\"{}\"", value)
                    }
                }
            };

            insert_query.push_str(&format!(
                "  <{}> <{}> {} .\n",
                triple.subject, triple.predicate, obj_str
            ));
        }

        insert_query.push_str("}}");

        // Execute SPARQL UPDATE (RDF persistence)
        self.store.update(&insert_query).map_err(|e| {
            NgsiError::InternalError(format!("Failed to store entity in RDF: {}", e))
        })?;

        // Also store in cache for fast reads (hybrid approach)
        self.cache
            .write()
            .await
            .insert(entity.id.clone(), entity.clone());

        Ok(())
    }

    async fn update_entity(&self, entity: &NgsiEntity) -> Result<(), NgsiError> {
        // Delete existing triples
        let delete_query = format!(
            "DELETE {{ GRAPH <{}> {{ <{}> ?p ?o }} }} WHERE {{ GRAPH <{}> {{ <{}> ?p ?o }} }}",
            self.graph_name, entity.id, self.graph_name, entity.id
        );

        // Execute SPARQL DELETE
        self.store
            .update(&delete_query)
            .map_err(|e| NgsiError::InternalError(format!("Failed to delete entity: {}", e)))?;

        // Insert updated entity (will execute SPARQL INSERT)
        self.store_entity(entity).await
    }

    async fn delete_entity(&self, id: &str) -> Result<(), NgsiError> {
        // Delete from RDF store
        let delete_query = format!(
            "DELETE {{ GRAPH <{}> {{ <{}> ?p ?o }} }} WHERE {{ GRAPH <{}> {{ <{}> ?p ?o }} }}",
            self.graph_name, id, self.graph_name, id
        );

        // Execute SPARQL DELETE
        self.store
            .update(&delete_query)
            .map_err(|e| NgsiError::InternalError(format!("Failed to delete entity: {}", e)))?;

        // Also remove from cache
        self.cache.write().await.remove(id);

        Ok(())
    }

    async fn query_entities(&self, params: &NgsiQueryParams) -> Result<Vec<NgsiEntity>, NgsiError> {
        // Build SPARQL query
        let translator =
            super::query::NgsiQueryTranslator::new().with_graph(self.graph_name.clone());
        let sparql = translator
            .translate_query(params)
            .map_err(|e| NgsiError::InternalError(format!("Query translation failed: {}", e)))?;

        // For RDF-backed storage, use cache for now
        // TODO: Implement SPARQL query translation when Store API is available
        let cache = self.cache.read().await;
        let mut results: Vec<NgsiEntity> = cache.values().cloned().collect();

        // Filter by type
        if let Some(ref entity_type) = params.entity_type {
            results.retain(|e| e.entity_type.primary() == entity_type);
        }

        // Filter by ID
        if let Some(ref id) = params.id {
            let ids: Vec<&str> = id.split(',').collect();
            results.retain(|e| ids.contains(&e.id.as_str()));
        }

        // Apply pagination
        let offset = params.offset.unwrap_or(0) as usize;
        let limit = params.limit.unwrap_or(100) as usize;

        Ok(results.into_iter().skip(offset).take(limit).collect())
    }

    async fn count_entities(&self, params: &NgsiQueryParams) -> Result<usize, NgsiError> {
        let entities = self.query_entities(params).await?;
        Ok(entities.len())
    }
}

// ============================================================================
// Server Handlers with AppState
// ============================================================================

/// Create entity handler for AppState
pub async fn create_entity_server(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    entities::create_entity(headers, body, store).await
}

/// Get entity handler for AppState
pub async fn get_entity_server(
    Path(entity_id): Path<String>,
    Query(params): Query<NgsiQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    entities::get_entity(Path(entity_id), Query(params), headers, store).await
}

/// Query entities handler for AppState
pub async fn query_entities_server(
    Query(params): Query<NgsiQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    entities::query_entities(Query(params), headers, store).await
}

/// Update entity attributes handler for AppState
pub async fn update_entity_attrs_server(
    Path(entity_id): Path<String>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    entities::update_entity_attrs(Path(entity_id), headers, body, store).await
}

/// Append entity attributes handler for AppState
pub async fn append_entity_attrs_server(
    Path(entity_id): Path<String>,
    Query(params): Query<NgsiQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    entities::append_entity_attrs(Path(entity_id), Query(params), headers, body, store).await
}

/// Delete entity handler for AppState
pub async fn delete_entity_server(
    Path(entity_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    entities::delete_entity(Path(entity_id), store).await
}

/// Delete entity attribute handler for AppState
pub async fn delete_entity_attr_server(
    Path((entity_id, attr_id)): Path<(String, String)>,
    Query(params): Query<NgsiQueryParams>,
    State(state): State<Arc<AppState>>,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    entities::delete_entity_attr(Path((entity_id, attr_id)), Query(params), store).await
}

// ============================================================================
// Batch Operation Handlers
// ============================================================================

/// Batch create entities handler
pub async fn batch_create_entities_server(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    batch::batch_create_entities(headers, body, store).await
}

/// Batch upsert entities handler
pub async fn batch_upsert_entities_server(
    Query(params): Query<batch::BatchQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    batch::batch_upsert_entities(Query(params), headers, body, store).await
}

/// Batch update entities handler
pub async fn batch_update_entities_server(
    Query(params): Query<batch::BatchQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    batch::batch_update_entities(Query(params), headers, body, store).await
}

/// Batch delete entities handler
pub async fn batch_delete_entities_server(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_entity_store(&state);
    batch::batch_delete_entities(headers, body, store).await
}

// ============================================================================
// Subscription Handlers
// ============================================================================

/// Create subscription handler
pub async fn create_subscription_server(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_subscription_store(&state);
    subscriptions::create_subscription(headers, body, store).await
}

/// Get subscription handler
pub async fn get_subscription_server(
    Path(subscription_id): Path<String>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, NgsiError> {
    let store = get_subscription_store(&state);
    subscriptions::get_subscription(Path(subscription_id), headers, store).await
}

/// List subscriptions handler
pub async fn list_subscriptions_server(
    Query(params): Query<subscriptions::SubscriptionQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, NgsiError> {
    let store = get_subscription_store(&state);
    subscriptions::list_subscriptions(Query(params), headers, store).await
}

/// Update subscription handler
pub async fn update_subscription_server(
    Path(subscription_id): Path<String>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_subscription_store(&state);
    subscriptions::update_subscription(Path(subscription_id), headers, body, store).await
}

/// Delete subscription handler
pub async fn delete_subscription_server(
    Path(subscription_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Response, NgsiError> {
    let store = get_subscription_store(&state);
    subscriptions::delete_subscription(Path(subscription_id), store).await
}

// ============================================================================
// Temporal Handlers
// ============================================================================

/// Create temporal entity handler
pub async fn create_temporal_entity_server(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, NgsiError> {
    let store = get_temporal_store(&state);
    temporal::create_temporal_entity(headers, body, store).await
}

/// Get temporal entity handler
pub async fn get_temporal_entity_server(
    Path(entity_id): Path<String>,
    Query(params): Query<temporal::TemporalQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, NgsiError> {
    let store = get_temporal_store(&state);
    temporal::get_temporal_entity(Path(entity_id), Query(params), headers, store).await
}

/// Query temporal entities handler
pub async fn query_temporal_entities_server(
    Query(params): Query<temporal::TemporalQueryParams>,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Response, NgsiError> {
    let store = get_temporal_store(&state);
    temporal::query_temporal_entities(Query(params), headers, store).await
}

/// Delete temporal entity handler
pub async fn delete_temporal_entity_server(
    Path(entity_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Result<Response, NgsiError> {
    let store = get_temporal_store(&state);
    temporal::delete_temporal_entity(Path(entity_id), store).await
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get entity store from AppState
fn get_entity_store(state: &AppState) -> Arc<dyn EntityStore> {
    // Create RDF-backed store or use in-memory for now
    Arc::new(RdfEntityStore::new(
        Arc::new(state.store.clone()),
        "urn:ngsi-ld:entities",
    ))
}

/// Get subscription store from AppState
fn get_subscription_store(_state: &AppState) -> Arc<dyn SubscriptionStore> {
    // Use in-memory store for now
    Arc::new(subscriptions::InMemorySubscriptionStore::new())
}

/// Get temporal store from AppState
fn get_temporal_store(_state: &AppState) -> Arc<dyn TemporalEntityStore> {
    // Use in-memory store for now
    Arc::new(temporal::InMemoryTemporalStore::new())
}
