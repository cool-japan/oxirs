//! Helper functions for API handlers
//!
//! This module contains utility functions used across different API handlers.

use super::ApiState;
use anyhow::{anyhow, Result};
use uuid::Uuid;

/// Get the production model version
pub async fn get_production_model_version(state: &ApiState) -> Result<Uuid> {
    // TODO: Implement proper production model version retrieval
    // For now, return the first available model if any
    let models = state.models.read().await;
    if let Some((uuid, _)) = models.iter().next() {
        Ok(*uuid)
    } else {
        Err(anyhow!("No models available"))
    }
}

/// Validate API key (if authentication is enabled)
pub fn validate_api_key(_api_key: &str, _state: &ApiState) -> bool {
    // TODO: Implement proper API key validation
    true
}

/// Calculate cache hit rate
pub fn calculate_cache_hit_rate(hits: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (hits as f64 / total as f64) * 100.0
    }
}
