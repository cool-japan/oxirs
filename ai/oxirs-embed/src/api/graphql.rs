//! GraphQL API implementation
//!
//! This module provides GraphQL schema and resolvers for embedding services.

#[cfg(feature = "graphql")]
use super::ApiState;
#[cfg(feature = "graphql")]
use async_graphql::{
    Context, EmptyMutation, EmptySubscription, Object, Result as GraphQLResult, Schema,
    SimpleObject,
};
#[cfg(feature = "graphql")]
use std::sync::Arc;
#[cfg(feature = "graphql")]
use uuid::Uuid;

/// GraphQL representation of model information
#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct ModelInfo {
    pub model_id: String,
    pub name: String,
    pub model_type: String,
    pub is_loaded: bool,
    pub is_trained: bool,
    pub num_entities: i32,
    pub num_relations: i32,
    pub num_triples: i32,
    pub dimensions: i32,
    pub created_at: String, // ISO 8601 timestamp
}

/// GraphQL representation of system health
#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct SystemHealth {
    pub status: String,
    pub models_loaded: i32,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub total_requests: i64,
}

/// GraphQL representation of cache statistics
#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct CacheStatistics {
    pub total_hits: i64,
    pub total_misses: i64,
    pub hit_rate: f64,
    pub memory_usage_bytes: i64,
    pub time_saved_seconds: f64,
}

/// GraphQL representation of prediction result
#[cfg(feature = "graphql")]
#[derive(SimpleObject)]
pub struct PredictionResult {
    pub entity: String,
    pub score: f64,
}

/// GraphQL Query root
#[cfg(feature = "graphql")]
pub struct Query;

#[cfg(feature = "graphql")]
#[Object]
impl Query {
    /// Get API version
    async fn version(&self) -> &str {
        "1.0.0"
    }

    /// Health check
    async fn health(&self) -> &str {
        "OK"
    }

    /// Get system health status
    async fn system_health(&self, ctx: &Context<'_>) -> GraphQLResult<SystemHealth> {
        let state = ctx.data::<Arc<ApiState>>()?;

        let models = state.models.read().await;
        let model_count = models.len() as i32;

        let cache_stats = state.cache_manager.get_stats();
        let cache_hit_rate = if cache_stats.total_hits + cache_stats.total_misses > 0 {
            cache_stats.total_hits as f64
                / (cache_stats.total_hits + cache_stats.total_misses) as f64
        } else {
            0.0
        };

        let memory_usage_mb =
            state.cache_manager.estimate_memory_usage() as f64 / (1024.0 * 1024.0);

        let status = if model_count > 0 && cache_hit_rate > 0.5 {
            "healthy"
        } else if model_count > 0 {
            "degraded"
        } else {
            "unhealthy"
        };

        Ok(SystemHealth {
            status: status.to_string(),
            models_loaded: model_count,
            cache_hit_rate,
            memory_usage_mb,
            total_requests: (cache_stats.total_hits + cache_stats.total_misses) as i64,
        })
    }

    /// Get cache statistics
    async fn cache_stats(&self, ctx: &Context<'_>) -> GraphQLResult<CacheStatistics> {
        let state = ctx.data::<Arc<ApiState>>()?;
        let cache_stats = state.cache_manager.get_stats();

        Ok(CacheStatistics {
            total_hits: cache_stats.total_hits as i64,
            total_misses: cache_stats.total_misses as i64,
            hit_rate: cache_stats.hit_rate,
            memory_usage_bytes: cache_stats.memory_usage_bytes as i64,
            time_saved_seconds: cache_stats.total_time_saved_seconds,
        })
    }

    /// List all models
    async fn models(&self, ctx: &Context<'_>) -> GraphQLResult<Vec<ModelInfo>> {
        let state = ctx.data::<Arc<ApiState>>()?;

        let registry_models = state.registry.list_models().await;
        let loaded_models = state.models.read().await;

        let mut model_list = Vec::new();
        for model_metadata in registry_models {
            let is_loaded = loaded_models.contains_key(&model_metadata.model_id);

            let (is_trained, stats) =
                if let Some(model) = loaded_models.get(&model_metadata.model_id) {
                    let stats = model.get_stats();
                    (model.is_trained(), stats)
                } else {
                    (false, Default::default())
                };

            model_list.push(ModelInfo {
                model_id: model_metadata.model_id.to_string(),
                name: model_metadata.name,
                model_type: model_metadata.model_type,
                is_loaded,
                is_trained,
                num_entities: stats.num_entities as i32,
                num_relations: stats.num_relations as i32,
                num_triples: stats.num_triples as i32,
                dimensions: stats.dimensions as i32,
                created_at: model_metadata.created_at.to_rfc3339(),
            });
        }

        Ok(model_list)
    }

    /// Get specific model information
    async fn model(&self, ctx: &Context<'_>, model_id: String) -> GraphQLResult<Option<ModelInfo>> {
        let state = ctx.data::<Arc<ApiState>>()?;

        let model_uuid = Uuid::parse_str(&model_id)
            .map_err(|_| async_graphql::Error::new("Invalid model ID format"))?;

        let model_metadata = match state.registry.get_model(model_uuid).await {
            Ok(metadata) => metadata,
            Err(_) => return Ok(None),
        };

        let loaded_models = state.models.read().await;
        let is_loaded = loaded_models.contains_key(&model_uuid);

        let (is_trained, stats) = if let Some(model) = loaded_models.get(&model_uuid) {
            let stats = model.get_stats();
            (model.is_trained(), stats)
        } else {
            (false, Default::default())
        };

        Ok(Some(ModelInfo {
            model_id: model_metadata.model_id.to_string(),
            name: model_metadata.name,
            model_type: model_metadata.model_type,
            is_loaded,
            is_trained,
            num_entities: stats.num_entities as i32,
            num_relations: stats.num_relations as i32,
            num_triples: stats.num_triples as i32,
            dimensions: stats.dimensions as i32,
            created_at: model_metadata.created_at.to_rfc3339(),
        }))
    }

    /// Predict objects for given subject and predicate
    async fn predict_objects(
        &self,
        ctx: &Context<'_>,
        subject: String,
        predicate: String,
        top_k: Option<i32>,
    ) -> GraphQLResult<Vec<PredictionResult>> {
        let state = ctx.data::<Arc<ApiState>>()?;

        // Get production model (simplified - would use the helper function)
        let models = state.models.read().await;
        let model = models
            .values()
            .next()
            .ok_or_else(|| async_graphql::Error::new("No models available"))?;

        if !model.is_trained() {
            return Err(async_graphql::Error::new("Model is not trained"));
        }

        let k = top_k.unwrap_or(10) as usize;
        let predictions = model
            .predict_objects(&subject, &predicate, k)
            .map_err(|e| async_graphql::Error::new(format!("Prediction failed: {}", e)))?;

        Ok(predictions
            .into_iter()
            .map(|(entity, score)| PredictionResult { entity, score })
            .collect())
    }

    /// Predict subjects for given predicate and object
    async fn predict_subjects(
        &self,
        ctx: &Context<'_>,
        predicate: String,
        object: String,
        top_k: Option<i32>,
    ) -> GraphQLResult<Vec<PredictionResult>> {
        let state = ctx.data::<Arc<ApiState>>()?;

        let models = state.models.read().await;
        let model = models
            .values()
            .next()
            .ok_or_else(|| async_graphql::Error::new("No models available"))?;

        if !model.is_trained() {
            return Err(async_graphql::Error::new("Model is not trained"));
        }

        let k = top_k.unwrap_or(10) as usize;
        let predictions = model
            .predict_subjects(&predicate, &object, k)
            .map_err(|e| async_graphql::Error::new(format!("Prediction failed: {}", e)))?;

        Ok(predictions
            .into_iter()
            .map(|(entity, score)| PredictionResult { entity, score })
            .collect())
    }

    /// Predict relations for given subject and object
    async fn predict_relations(
        &self,
        ctx: &Context<'_>,
        subject: String,
        object: String,
        top_k: Option<i32>,
    ) -> GraphQLResult<Vec<PredictionResult>> {
        let state = ctx.data::<Arc<ApiState>>()?;

        let models = state.models.read().await;
        let model = models
            .values()
            .next()
            .ok_or_else(|| async_graphql::Error::new("No models available"))?;

        if !model.is_trained() {
            return Err(async_graphql::Error::new("Model is not trained"));
        }

        let k = top_k.unwrap_or(10) as usize;
        let predictions = model
            .predict_relations(&subject, &object, k)
            .map_err(|e| async_graphql::Error::new(format!("Prediction failed: {}", e)))?;

        Ok(predictions
            .into_iter()
            .map(|(entity, score)| PredictionResult { entity, score })
            .collect())
    }

    /// Score a triple (subject, predicate, object)
    async fn score_triple(
        &self,
        ctx: &Context<'_>,
        subject: String,
        predicate: String,
        object: String,
    ) -> GraphQLResult<f64> {
        let state = ctx.data::<Arc<ApiState>>()?;

        let models = state.models.read().await;
        let model = models
            .values()
            .next()
            .ok_or_else(|| async_graphql::Error::new("No models available"))?;

        if !model.is_trained() {
            return Err(async_graphql::Error::new("Model is not trained"));
        }

        let score = model
            .score_triple(&subject, &predicate, &object)
            .map_err(|e| async_graphql::Error::new(format!("Scoring failed: {}", e)))?;

        Ok(score)
    }
}

/// Create GraphQL schema with API state
#[cfg(feature = "graphql")]
pub fn create_schema() -> Schema<Query, EmptyMutation, EmptySubscription> {
    Schema::build(Query, EmptyMutation, EmptySubscription).finish()
}

/// GraphQL handler for Axum
#[cfg(all(feature = "graphql", feature = "api-server"))]
pub async fn graphql_handler(
    schema: axum::extract::Extension<Schema<Query, EmptyMutation, EmptySubscription>>,
    state: axum::extract::Extension<Arc<ApiState>>,
    req: axum::extract::Json<async_graphql::Request>,
) -> axum::Json<async_graphql::Response> {
    let request = req.0.data(state.0.clone());
    let response = schema.execute(request).await;
    axum::Json(response)
}

/// GraphiQL playground handler
#[cfg(all(feature = "graphql", feature = "api-server"))]
pub async fn graphiql() -> impl axum::response::IntoResponse {
    axum::response::Html(
        async_graphql::http::GraphiQLSource::build()
            .endpoint("/graphql")
            .finish(),
    )
}
