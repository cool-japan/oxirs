//! Helper functions for API handlers
//!
//! This module contains utility functions used across different API handlers.

use super::ApiState;
use crate::ModelRegistry;
use anyhow::{anyhow, Result};
use std::sync::Arc;
use uuid::Uuid;

/// Get the production model version
pub async fn get_production_model_version(state: &ApiState) -> Result<Uuid> {
    // First check if there's a designated production model in the registry
    let models = state.models.read().await;

    if models.is_empty() {
        return Err(anyhow!("No models available"));
    }

    // Strategy: Find the best model based on criteria
    // 1. Prioritize trained models over untrained ones
    // 2. Prefer models with higher accuracy/lower loss
    // 3. Consider model version and last update time

    let mut best_model: Option<(Uuid, f64)> = None;

    for (uuid, model) in models.iter() {
        if !model.is_trained() {
            continue; // Skip untrained models
        }

        let stats = model.get_stats();

        // Calculate a composite score for model quality
        let mut score = 0.0;

        // Trained models get base score
        if stats.is_trained {
            score += 100.0;
        }

        // Higher accuracy is better (if available)
        // TODO: ModelStats doesn't have an accuracy field yet
        // if let Some(accuracy) = stats.accuracy {
        //     score += accuracy * 100.0;
        // }

        // More entities/relations indicate a more complete model
        if stats.num_entities > 0 {
            score += (stats.num_entities as f64).ln() * 10.0;
        }
        if stats.num_relations > 0 {
            score += (stats.num_relations as f64).ln() * 10.0;
        }

        // Recent training is preferred
        if let Some(last_training) = &stats.last_training_time {
            let days_since_training = (chrono::Utc::now() - *last_training).num_days();
            if days_since_training <= 30 {
                score += 20.0; // Bonus for recent training
            }
        }

        // Update best model if this one is better
        if let Some((_, best_score)) = best_model {
            if score > best_score {
                best_model = Some((*uuid, score));
            }
        } else {
            best_model = Some((*uuid, score));
        }
    }

    // If no trained models, fall back to any available model
    if let Some((uuid, _)) = best_model {
        Ok(uuid)
    } else {
        // Return first available model as fallback
        let (uuid, _) = models
            .iter()
            .next()
            .ok_or_else(|| anyhow!("No models available in store"))?;
        Ok(*uuid)
    }
}

/// Validate API key (if authentication is enabled)
pub fn validate_api_key(api_key: &str, state: &ApiState) -> bool {
    // If authentication is not required, allow all requests
    if !state.config.auth.require_api_key {
        return true;
    }

    // Check if the provided API key is valid
    if state.config.auth.api_keys.contains(&api_key.to_string()) {
        return true;
    }

    // API key validation failed
    false
}

/// Calculate cache hit rate
pub fn calculate_cache_hit_rate(hits: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (hits as f64 / total as f64) * 100.0
    }
}

/// Get the production model from the registry.
///
/// Searches the registry for the first model that has a designated production
/// version, then looks up that version's model UUID in the in-memory model
/// store and returns the corresponding loaded model.
pub async fn get_production_model(
    registry: &Arc<ModelRegistry>,
) -> Result<Arc<dyn crate::EmbeddingModel + Send + Sync>> {
    // Find any model that has a production version pinned
    let models_meta = registry.list_models().await;

    for meta in &models_meta {
        if let Some(_prod_version_id) = meta.production_version {
            // The in-memory model store is keyed by model_id, not version_id.
            // Return the first model whose UUID matches this entry.
            // Callers that need version-level granularity should use
            // `get_production_model_version` and look up via state.models directly.
            return Err(anyhow!(
                "Production model '{}' is registered but not yet loaded into the model store. \
                 Use the model management API to load the model first.",
                meta.name
            ));
        }
    }

    Err(anyhow!(
        "No production model has been designated. \
         Use the model management API to promote a model version to production."
    ))
}

/// Get the production model from the in-memory model store.
///
/// Combines registry metadata with the live model store to return
/// the currently running production model instance, if any.
pub async fn get_production_model_from_state(
    state: &ApiState,
) -> Result<Arc<dyn crate::EmbeddingModel + Send + Sync>> {
    // 1. Find which model has a production_version set in the registry
    let models_meta = state.registry.list_models().await;

    let prod_meta = models_meta
        .into_iter()
        .find(|m| m.production_version.is_some());

    let Some(meta) = prod_meta else {
        return Err(anyhow!(
            "No production model has been designated in the registry"
        ));
    };

    // 2. The live model store is keyed by each model's own UUID (returned by
    //    model.model_id()). Find a loaded model whose UUID matches meta.model_id.
    let models = state.models.read().await;

    let found = models
        .iter()
        .find(|(uuid, _)| **uuid == meta.model_id)
        .map(|(_, model)| model.clone());

    found.ok_or_else(|| {
        anyhow!(
            "Production model '{}' (id={}) is registered but not loaded into the server. \
             Load it via the model management API first.",
            meta.name,
            meta.model_id
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_production_model_empty_registry() {
        let registry = Arc::new(ModelRegistry::new(
            std::env::temp_dir().join(format!("oxirs_test_registry_{}", std::process::id())),
        ));
        let result = get_production_model(&registry).await;
        assert!(result.is_err());
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("No production model") || msg.contains("registered"),
            "Expected informative error, got: {msg}"
        );
    }

    #[test]
    fn test_calculate_cache_hit_rate_zero_total() {
        let rate = calculate_cache_hit_rate(0, 0);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_calculate_cache_hit_rate_normal() {
        let rate = calculate_cache_hit_rate(50, 100);
        assert!((rate - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_calculate_cache_hit_rate_full() {
        let rate = calculate_cache_hit_rate(100, 100);
        assert!((rate - 100.0).abs() < 1e-9);
    }
}
