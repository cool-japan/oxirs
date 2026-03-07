//! Helper functions for API handlers
//!
//! This module contains utility functions used across different API handlers.

use super::ApiState;
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
        score += (stats.num_entities as f64).ln() * 10.0;
        score += (stats.num_relations as f64).ln() * 10.0;

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
        let (uuid, _) = models.iter().next().expect("models should not be empty");
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

/// Get the production model (not just the version)
/// This is a stub that needs to be properly implemented based on the registry type
pub async fn get_production_model<T>(
    _registry: &T,
) -> Result<Arc<dyn crate::EmbeddingModel + Send + Sync>> {
    Err(anyhow!("get_production_model not yet implemented"))
}
