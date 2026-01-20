//! Consciousness Performance Metrics
//!
//! Tracks performance and health metrics for consciousness processing.

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use super::super::*;

/// Consciousness performance metrics tracker
#[derive(Debug, Clone)]
pub struct ConsciousnessMetrics {
    processing_times: VecDeque<Duration>,
    accuracy_scores: VecDeque<f64>,
    health_scores: VecDeque<f64>,
}

impl Default for ConsciousnessMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessMetrics {
    pub fn new() -> Self {
        Self {
            processing_times: VecDeque::with_capacity(100),
            accuracy_scores: VecDeque::with_capacity(100),
            health_scores: VecDeque::with_capacity(100),
        }
    }

    pub fn update(
        &mut self,
        awareness: f64,
        _attention: &AttentionAllocation,
        memory: &MemoryIntegrationResult,
        emotion: &EmotionalResponse,
        processing_time: Duration,
    ) -> Result<()> {
        self.processing_times.push_back(processing_time);
        self.health_scores
            .push_back((awareness + memory.confidence + emotion.emotional_coherence) / 3.0);

        if self.processing_times.len() > 100 {
            self.processing_times.pop_front();
        }
        if self.health_scores.len() > 100 {
            self.health_scores.pop_front();
        }

        Ok(())
    }
}

/// Snapshot of consciousness state at a point in time
#[derive(Debug, Clone)]
pub struct ConsciousnessSnapshot {
    pub timestamp: DateTime<Utc>,
    pub awareness_level: f64,
    pub attention_weights: HashMap<String, f64>,
    pub emotional_state: EmotionalStateSnapshot,
    pub memory_pressure: f64,
    pub neural_activity: NeuralActivitySummary,
}
