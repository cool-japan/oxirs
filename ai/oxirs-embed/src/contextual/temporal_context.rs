//! Temporal context handling

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Temporal context for time-aware embeddings
#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub timestamp: DateTime<Utc>,
    pub time_window: chrono::Duration,
    pub seasonal_factors: Vec<f32>,
    pub trend_indicators: Vec<f32>,
}

impl Default for TemporalContext {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            time_window: chrono::Duration::hours(24),
            seasonal_factors: Vec::new(),
            trend_indicators: Vec::new(),
        }
    }
}