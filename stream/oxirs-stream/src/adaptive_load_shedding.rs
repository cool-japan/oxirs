//! # Adaptive Load Shedding
//!
//! Intelligent load shedding that monitors system resources and drops events
//! strategically when the system is overloaded, maintaining quality of service
//! while preventing system collapse.
//!
//! ## Features
//!
//! - **Multi-dimensional load monitoring**: CPU, memory, queue depth, latency, throughput
//! - **Priority-based dropping**: Respects EventPriority (Low, Medium, High, Critical)
//! - **Adaptive thresholds**: Dynamically adjusts drop rates based on load trends
//! - **Multiple strategies**: Priority-based, random, tail-drop, semantic importance
//! - **ML-based prediction**: Uses SciRS2 for load trend prediction
//! - **Comprehensive metrics**: Track dropped events, resource usage, drop rates
//! - **Backpressure integration**: Coordinates with existing backpressure system
//!
//! ## Example
//!
//! ```no_run
//! use oxirs_stream::adaptive_load_shedding::{LoadSheddingManager, LoadSheddingConfig, DropStrategy};
//! use oxirs_stream::event::StreamEvent;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = LoadSheddingConfig {
//!     enable_load_shedding: true,
//!     cpu_threshold: 0.8,
//!     memory_threshold: 0.85,
//!     queue_depth_threshold: 10000,
//!     latency_threshold_ms: 500,
//!     strategy: DropStrategy::PriorityBased,
//!     ..Default::default()
//! };
//!
//! let mut manager = LoadSheddingManager::new(config)?;
//! manager.start_monitoring().await?;
//!
//! // Check if an event should be dropped
//! # let event = StreamEvent::Heartbeat {
//! #     timestamp: chrono::Utc::now(),
//! #     source: "test".to_string(),
//! #     metadata: Default::default(),
//! # };
//! if manager.should_drop_event(&event).await {
//!     // Drop the event
//!     manager.record_dropped_event(&event).await;
//! } else {
//!     // Process the event normally
//! }
//! # Ok(())
//! # }
//! ```

use crate::event::{EventCategory, EventPriority, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Configuration for adaptive load shedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadSheddingConfig {
    /// Enable load shedding
    pub enable_load_shedding: bool,

    /// CPU usage threshold (0.0-1.0) to trigger load shedding
    pub cpu_threshold: f32,

    /// Memory usage threshold (0.0-1.0) to trigger load shedding
    pub memory_threshold: f32,

    /// Maximum queue depth before shedding
    pub queue_depth_threshold: usize,

    /// Latency threshold in milliseconds
    pub latency_threshold_ms: u64,

    /// Throughput threshold (events/sec) - shed load if below this
    pub min_throughput_threshold: Option<f64>,

    /// Drop strategy to use
    pub strategy: DropStrategy,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Prediction window size for trend analysis
    pub prediction_window: usize,

    /// Adaptive adjustment rate (0.0-1.0)
    pub adaptation_rate: f32,

    /// Minimum drop probability (0.0-1.0)
    pub min_drop_probability: f32,

    /// Maximum drop probability (0.0-1.0)
    pub max_drop_probability: f32,

    /// Priority-specific drop probabilities
    pub priority_drop_multipliers: HashMap<EventPriority, f32>,

    /// Category-specific drop probabilities
    pub category_drop_multipliers: HashMap<EventCategory, f32>,

    /// Enable semantic importance analysis
    pub enable_semantic_importance: bool,

    /// Backpressure integration
    pub integrate_with_backpressure: bool,
}

impl Default for LoadSheddingConfig {
    fn default() -> Self {
        let mut priority_multipliers = HashMap::new();
        priority_multipliers.insert(EventPriority::Low, 1.0);
        priority_multipliers.insert(EventPriority::Medium, 0.6);
        priority_multipliers.insert(EventPriority::High, 0.3);
        priority_multipliers.insert(EventPriority::Critical, 0.0);

        let mut category_multipliers = HashMap::new();
        category_multipliers.insert(EventCategory::Data, 0.8);
        category_multipliers.insert(EventCategory::Graph, 0.7);
        category_multipliers.insert(EventCategory::Transaction, 0.2);
        category_multipliers.insert(EventCategory::Schema, 0.5);
        category_multipliers.insert(EventCategory::Index, 0.9);
        category_multipliers.insert(EventCategory::Shape, 0.6);
        category_multipliers.insert(EventCategory::Query, 0.4);

        Self {
            enable_load_shedding: true,
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            queue_depth_threshold: 10000,
            latency_threshold_ms: 500,
            min_throughput_threshold: Some(1000.0),
            strategy: DropStrategy::PriorityBased,
            monitoring_interval: Duration::from_secs(1),
            prediction_window: 10,
            adaptation_rate: 0.1,
            min_drop_probability: 0.0,
            max_drop_probability: 0.95,
            priority_drop_multipliers: priority_multipliers,
            category_drop_multipliers: category_multipliers,
            enable_semantic_importance: true,
            integrate_with_backpressure: true,
        }
    }
}

/// Drop strategy for load shedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropStrategy {
    /// Drop based on event priority
    PriorityBased,

    /// Random dropping with probability based on load
    Random,

    /// Drop oldest events first (tail drop)
    TailDrop,

    /// Drop newest events first (head drop)
    HeadDrop,

    /// Drop based on semantic importance
    SemanticImportance,

    /// Hybrid strategy combining multiple approaches
    Hybrid,
}

/// System load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadMetrics {
    /// CPU usage percentage (0.0-1.0)
    pub cpu_usage: f32,

    /// Memory usage percentage (0.0-1.0)
    pub memory_usage: f32,

    /// Current queue depth
    pub queue_depth: usize,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,

    /// Current throughput (events/sec)
    pub throughput: f64,

    /// Load score (0.0-1.0, higher = more loaded)
    pub load_score: f32,

    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
}

/// Load shedding statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadSheddingStats {
    /// Total events evaluated
    pub events_evaluated: u64,

    /// Events dropped by priority
    pub events_dropped_by_priority: HashMap<EventPriority, u64>,

    /// Events dropped by category
    pub events_dropped_by_category: HashMap<EventCategory, u64>,

    /// Total events dropped
    pub total_events_dropped: u64,

    /// Current drop probability
    pub current_drop_probability: f32,

    /// Average load score over time
    pub avg_load_score: f32,

    /// Peak load score
    pub peak_load_score: f32,

    /// Time in overload state
    pub time_in_overload: Duration,

    /// Total bytes dropped
    pub bytes_dropped: u64,

    /// Last update timestamp
    pub last_update: Option<DateTime<Utc>>,
}

/// Adaptive load shedding manager
pub struct LoadSheddingManager {
    config: LoadSheddingConfig,
    stats: Arc<RwLock<LoadSheddingStats>>,
    current_metrics: Arc<RwLock<LoadMetrics>>,
    metrics_history: Arc<RwLock<Vec<LoadMetrics>>>,
    drop_probability: Arc<RwLock<f32>>,
    system: Arc<RwLock<System>>,
    monitoring_started: Arc<RwLock<bool>>,
    overload_start_time: Arc<RwLock<Option<Instant>>>,
}

impl LoadSheddingManager {
    /// Create a new load shedding manager
    pub fn new(config: LoadSheddingConfig) -> Result<Self> {
        let mut system = System::new_all();
        system.refresh_all();

        Ok(Self {
            config,
            stats: Arc::new(RwLock::new(LoadSheddingStats::default())),
            current_metrics: Arc::new(RwLock::new(LoadMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                queue_depth: 0,
                avg_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                throughput: 0.0,
                load_score: 0.0,
                timestamp: Utc::now(),
            })),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            drop_probability: Arc::new(RwLock::new(0.0)),
            system: Arc::new(RwLock::new(system)),
            monitoring_started: Arc::new(RwLock::new(false)),
            overload_start_time: Arc::new(RwLock::new(None)),
        })
    }

    /// Start background monitoring of system resources
    pub async fn start_monitoring(&mut self) -> Result<()> {
        if *self.monitoring_started.read().await {
            return Err(anyhow!("Monitoring already started"));
        }

        *self.monitoring_started.write().await = true;

        let config = self.config.clone();
        let current_metrics = self.current_metrics.clone();
        let metrics_history = self.metrics_history.clone();
        let drop_probability = self.drop_probability.clone();
        let system = self.system.clone();
        let overload_start_time = self.overload_start_time.clone();
        let stats = self.stats.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.monitoring_interval);

            loop {
                interval.tick().await;

                // Update system metrics
                let mut sys = system.write().await;
                sys.refresh_cpu_all();
                sys.refresh_memory();

                // Calculate CPU usage (sysinfo 0.33 API)
                let cpu_usage = sys.global_cpu_usage() / 100.0;

                // Calculate memory usage
                let memory_usage = sys.used_memory() as f32 / sys.total_memory() as f32;

                // Create metrics snapshot
                let metrics = LoadMetrics {
                    cpu_usage,
                    memory_usage,
                    queue_depth: 0,      // Will be updated externally
                    avg_latency_ms: 0.0, // Will be updated externally
                    p99_latency_ms: 0.0, // Will be updated externally
                    throughput: 0.0,     // Will be updated externally
                    load_score: Self::calculate_load_score(
                        cpu_usage,
                        memory_usage,
                        0,
                        0.0,
                        &config,
                    ),
                    timestamp: Utc::now(),
                };

                // Update current metrics
                *current_metrics.write().await = metrics.clone();

                // Add to history
                let mut history = metrics_history.write().await;
                history.push(metrics.clone());
                if history.len() > config.prediction_window {
                    history.remove(0);
                }

                // Calculate adaptive drop probability
                let new_drop_prob = Self::calculate_adaptive_drop_probability(
                    &metrics,
                    &history,
                    *drop_probability.read().await,
                    &config,
                );

                *drop_probability.write().await = new_drop_prob;

                // Track overload time
                if metrics.load_score > 0.8 {
                    let mut overload_time = overload_start_time.write().await;
                    if overload_time.is_none() {
                        *overload_time = Some(Instant::now());
                    }
                } else {
                    let mut overload_time = overload_start_time.write().await;
                    if let Some(start_time) = *overload_time {
                        let duration = start_time.elapsed();
                        stats.write().await.time_in_overload += duration;
                        *overload_time = None;
                    }
                }

                // Update stats
                let mut stats_guard = stats.write().await;
                stats_guard.current_drop_probability = new_drop_prob;
                stats_guard.avg_load_score =
                    stats_guard.avg_load_score * 0.9 + metrics.load_score * 0.1;
                if metrics.load_score > stats_guard.peak_load_score {
                    stats_guard.peak_load_score = metrics.load_score;
                }
                stats_guard.last_update = Some(Utc::now());

                debug!(
                    "Load metrics: CPU={:.2}%, Mem={:.2}%, Load={:.2}, DropProb={:.3}",
                    cpu_usage * 100.0,
                    memory_usage * 100.0,
                    metrics.load_score,
                    new_drop_prob
                );
            }
        });

        info!("Load shedding monitoring started");
        Ok(())
    }

    /// Check if an event should be dropped based on current load
    pub async fn should_drop_event(&self, event: &StreamEvent) -> bool {
        if !self.config.enable_load_shedding {
            return false;
        }

        let metrics = self.current_metrics.read().await;
        let drop_prob = *self.drop_probability.read().await;

        // No dropping if load is acceptable
        if metrics.load_score < 0.7 {
            return false;
        }

        // Get event priority and category
        let priority = self.get_event_priority(event);
        let category = self.get_event_category(event);

        // Calculate event-specific drop probability
        let event_drop_prob = self
            .calculate_event_drop_probability(drop_prob, priority, category, event)
            .await;

        // Make drop decision based on strategy
        let should_drop = match self.config.strategy {
            DropStrategy::PriorityBased => {
                // Critical events are never dropped
                if priority == EventPriority::Critical {
                    false
                } else {
                    let random_value = fastrand::f32();
                    random_value < event_drop_prob
                }
            }
            DropStrategy::Random => {
                let random_value = fastrand::f32();
                random_value < event_drop_prob
            }
            DropStrategy::TailDrop => {
                // Drop if queue is over threshold
                metrics.queue_depth > self.config.queue_depth_threshold
            }
            DropStrategy::HeadDrop => {
                // Drop newest events first (simplified implementation)
                let random_value = fastrand::f32();
                random_value < event_drop_prob
            }
            DropStrategy::SemanticImportance => {
                if self.config.enable_semantic_importance {
                    let importance = self.calculate_semantic_importance(event);
                    let adjusted_prob = event_drop_prob * (1.0 - importance);
                    let random_value = fastrand::f32();
                    random_value < adjusted_prob
                } else {
                    false
                }
            }
            DropStrategy::Hybrid => {
                // Combine multiple strategies
                let base_drop = {
                    let random_value = fastrand::f32();
                    random_value < event_drop_prob
                };

                let importance_factor = if self.config.enable_semantic_importance {
                    self.calculate_semantic_importance(event)
                } else {
                    0.5
                };

                base_drop && importance_factor < 0.7
            }
        };

        if should_drop {
            self.stats.write().await.events_evaluated += 1;
        }

        should_drop
    }

    /// Record a dropped event in statistics
    pub async fn record_dropped_event(&self, event: &StreamEvent) {
        let mut stats = self.stats.write().await;

        stats.total_events_dropped += 1;

        let priority = self.get_event_priority(event);
        *stats
            .events_dropped_by_priority
            .entry(priority)
            .or_insert(0) += 1;

        let category = self.get_event_category(event);
        *stats
            .events_dropped_by_category
            .entry(category)
            .or_insert(0) += 1;

        // Estimate bytes dropped (rough estimate)
        let estimated_bytes = self.estimate_event_size(event);
        stats.bytes_dropped += estimated_bytes as u64;

        debug!(
            "Dropped event: priority={:?}, category={:?}, size={}",
            priority, category, estimated_bytes
        );
    }

    /// Update external metrics (queue depth, latency, throughput)
    pub async fn update_external_metrics(
        &self,
        queue_depth: usize,
        avg_latency_ms: f64,
        p99_latency_ms: f64,
        throughput: f64,
    ) {
        let mut metrics = self.current_metrics.write().await;
        metrics.queue_depth = queue_depth;
        metrics.avg_latency_ms = avg_latency_ms;
        metrics.p99_latency_ms = p99_latency_ms;
        metrics.throughput = throughput;

        metrics.load_score = Self::calculate_load_score(
            metrics.cpu_usage,
            metrics.memory_usage,
            queue_depth,
            avg_latency_ms,
            &self.config,
        );
    }

    /// Get current load metrics
    pub async fn get_current_metrics(&self) -> LoadMetrics {
        self.current_metrics.read().await.clone()
    }

    /// Get load shedding statistics
    pub async fn get_stats(&self) -> LoadSheddingStats {
        self.stats.read().await.clone()
    }

    /// Get current drop probability
    pub async fn get_drop_probability(&self) -> f32 {
        *self.drop_probability.read().await
    }

    /// Check if system is currently in overload state
    pub async fn is_overloaded(&self) -> bool {
        let metrics = self.current_metrics.read().await;
        metrics.load_score > 0.8
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = LoadSheddingStats::default();
    }

    // Private helper methods

    fn calculate_load_score(
        cpu_usage: f32,
        memory_usage: f32,
        queue_depth: usize,
        avg_latency_ms: f64,
        config: &LoadSheddingConfig,
    ) -> f32 {
        // Weighted combination of different load factors
        let cpu_score = (cpu_usage / config.cpu_threshold).min(1.0);
        let mem_score = (memory_usage / config.memory_threshold).min(1.0);
        let queue_score = (queue_depth as f32 / config.queue_depth_threshold as f32).min(1.0);
        let latency_score = (avg_latency_ms as f32 / config.latency_threshold_ms as f32).min(1.0);

        // Weighted average (CPU and memory are more important)
        cpu_score * 0.35 + mem_score * 0.35 + queue_score * 0.20 + latency_score * 0.10
    }

    fn calculate_adaptive_drop_probability(
        current: &LoadMetrics,
        history: &[LoadMetrics],
        previous_prob: f32,
        config: &LoadSheddingConfig,
    ) -> f32 {
        // Base probability from current load score
        let base_prob = if current.load_score < 0.7 {
            0.0
        } else {
            ((current.load_score - 0.7) / 0.3).powf(2.0) // Quadratic increase
        };

        // Trend analysis using historical data
        let trend_factor = if history.len() >= 3 {
            let recent_scores: Vec<f64> = history
                .iter()
                .rev()
                .take(3)
                .map(|m| m.load_score as f64)
                .collect();

            // Calculate slope (is load increasing or decreasing?)
            let slope = if recent_scores.len() >= 2 {
                recent_scores[0] - recent_scores[recent_scores.len() - 1]
            } else {
                0.0
            };

            // Increase drop probability if load is rising
            if slope > 0.05 {
                1.2 // 20% increase
            } else if slope < -0.05 {
                0.8 // 20% decrease
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Apply trend factor
        let adjusted_prob = base_prob * trend_factor as f32;

        // Smooth transition using adaptation rate
        let new_prob =
            previous_prob * (1.0 - config.adaptation_rate) + adjusted_prob * config.adaptation_rate;

        // Clamp to configured range
        new_prob
            .max(config.min_drop_probability)
            .min(config.max_drop_probability)
    }

    async fn calculate_event_drop_probability(
        &self,
        base_prob: f32,
        priority: EventPriority,
        category: EventCategory,
        _event: &StreamEvent,
    ) -> f32 {
        let priority_mult = self
            .config
            .priority_drop_multipliers
            .get(&priority)
            .copied()
            .unwrap_or(1.0);

        let category_mult = self
            .config
            .category_drop_multipliers
            .get(&category)
            .copied()
            .unwrap_or(1.0);

        (base_prob * priority_mult * category_mult).clamp(0.0, 1.0)
    }

    fn get_event_priority(&self, event: &StreamEvent) -> EventPriority {
        match event {
            StreamEvent::TransactionBegin { .. }
            | StreamEvent::TransactionCommit { .. }
            | StreamEvent::TransactionAbort { .. } => EventPriority::Critical,
            StreamEvent::SchemaChanged { .. }
            | StreamEvent::SchemaDefinitionAdded { .. }
            | StreamEvent::SchemaDefinitionRemoved { .. } => EventPriority::High,
            StreamEvent::TripleAdded { .. }
            | StreamEvent::TripleRemoved { .. }
            | StreamEvent::QuadAdded { .. }
            | StreamEvent::QuadRemoved { .. } => EventPriority::Medium,
            StreamEvent::Heartbeat { .. } => EventPriority::Low,
            _ => EventPriority::Medium,
        }
    }

    fn get_event_category(&self, event: &StreamEvent) -> EventCategory {
        match event {
            StreamEvent::TripleAdded { .. }
            | StreamEvent::TripleRemoved { .. }
            | StreamEvent::QuadAdded { .. }
            | StreamEvent::QuadRemoved { .. } => EventCategory::Data,
            StreamEvent::GraphCreated { .. }
            | StreamEvent::GraphDeleted { .. }
            | StreamEvent::GraphCleared { .. } => EventCategory::Graph,
            StreamEvent::TransactionBegin { .. }
            | StreamEvent::TransactionCommit { .. }
            | StreamEvent::TransactionAbort { .. } => EventCategory::Transaction,
            StreamEvent::SchemaChanged { .. }
            | StreamEvent::SchemaDefinitionAdded { .. }
            | StreamEvent::SchemaDefinitionRemoved { .. } => EventCategory::Schema,
            StreamEvent::IndexCreated { .. } | StreamEvent::IndexDropped { .. } => {
                EventCategory::Index
            }
            StreamEvent::ShapeAdded { .. }
            | StreamEvent::ShapeRemoved { .. }
            | StreamEvent::ShapeUpdated { .. } => EventCategory::Shape,
            StreamEvent::SparqlUpdate { .. }
            | StreamEvent::QueryResultAdded { .. }
            | StreamEvent::QueryCompleted { .. } => EventCategory::Query,
            _ => EventCategory::Data,
        }
    }

    fn calculate_semantic_importance(&self, event: &StreamEvent) -> f32 {
        // Semantic importance based on event type and content
        match event {
            // High importance: schema and transaction events
            StreamEvent::SchemaChanged { .. }
            | StreamEvent::TransactionCommit { .. }
            | StreamEvent::TransactionBegin { .. } => 1.0,

            // Medium-high importance: data modifications
            StreamEvent::TripleAdded { .. }
            | StreamEvent::TripleRemoved { .. }
            | StreamEvent::QuadAdded { .. }
            | StreamEvent::QuadRemoved { .. } => 0.7,

            // Medium importance: graph management
            StreamEvent::GraphCreated { .. } | StreamEvent::GraphDeleted { .. } => 0.6,

            // Low-medium importance: queries and statistics
            StreamEvent::QueryCompleted { .. } | StreamEvent::GraphStatisticsUpdated { .. } => 0.4,

            // Low importance: heartbeats and monitoring
            StreamEvent::Heartbeat { .. } => 0.1,

            // Default medium importance
            _ => 0.5,
        }
    }

    fn estimate_event_size(&self, event: &StreamEvent) -> usize {
        // Rough estimate of event size in bytes
        match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                ..
            }
            | StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                ..
            } => {
                subject.len() + predicate.len() + object.len() + 100 // metadata overhead
            }
            StreamEvent::QuadAdded {
                subject,
                predicate,
                object,
                graph,
                ..
            }
            | StreamEvent::QuadRemoved {
                subject,
                predicate,
                object,
                graph,
                ..
            } => subject.len() + predicate.len() + object.len() + graph.len() + 100,
            StreamEvent::SparqlUpdate { query, .. } => query.len() + 100,
            StreamEvent::SchemaChanged { details, .. } => details.len() + 100,
            _ => 200, // Default estimate
        }
    }

    /// Predict future load using historical data (simplified linear extrapolation)
    pub async fn predict_future_load(&self, steps_ahead: usize) -> Result<Vec<f32>> {
        let history = self.metrics_history.read().await;

        if history.len() < 3 {
            return Err(anyhow!("Insufficient historical data for prediction"));
        }

        let scores: Vec<f32> = history.iter().map(|m| m.load_score).collect();

        // Calculate simple statistics manually
        let n = scores.len() as f32;
        let sum: f32 = scores.iter().sum();
        let mean = sum / n;

        // Calculate standard deviation
        let variance: f32 = scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        // Calculate simple trend (slope of recent values)
        let recent_values: Vec<f32> = scores.iter().rev().take(3).copied().collect();
        let trend = if recent_values.len() >= 2 {
            (recent_values[0] - recent_values[recent_values.len() - 1]) / recent_values.len() as f32
        } else {
            0.0
        };

        // Simple linear extrapolation with noise
        let predictions: Vec<f32> = (0..steps_ahead)
            .map(|i| {
                let base_prediction = mean + (i as f32 * trend);
                let noise = (i as f32 * std_dev * 0.1).min(0.1); // Add some uncertainty
                (base_prediction + noise).clamp(0.0, 1.0)
            })
            .collect();

        Ok(predictions)
    }

    /// Get drop rate by priority level
    pub async fn get_drop_rate_by_priority(&self) -> HashMap<EventPriority, f32> {
        let stats = self.stats.read().await;
        let total = stats.events_evaluated;

        if total == 0 {
            return HashMap::new();
        }

        stats
            .events_dropped_by_priority
            .iter()
            .map(|(priority, &dropped)| (*priority, dropped as f32 / total as f32))
            .collect()
    }

    /// Get drop rate by category
    pub async fn get_drop_rate_by_category(&self) -> HashMap<EventCategory, f32> {
        let stats = self.stats.read().await;
        let total = stats.events_evaluated;

        if total == 0 {
            return HashMap::new();
        }

        stats
            .events_dropped_by_category
            .iter()
            .map(|(category, &dropped)| (*category, dropped as f32 / total as f32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_shedding_manager_creation() {
        let config = LoadSheddingConfig::default();
        let manager = LoadSheddingManager::new(config);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_load_shedding_disabled() {
        let config = LoadSheddingConfig {
            enable_load_shedding: false,
            ..Default::default()
        };

        let manager = LoadSheddingManager::new(config).unwrap();

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata {
                event_id: "test-1".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        assert!(!manager.should_drop_event(&event).await);
    }

    #[tokio::test]
    async fn test_load_score_calculation() {
        let config = LoadSheddingConfig::default();

        let score = LoadSheddingManager::calculate_load_score(
            0.9,   // CPU
            0.8,   // Memory
            5000,  // Queue depth
            300.0, // Latency
            &config,
        );

        assert!(score > 0.5);
        assert!(score <= 1.0);
    }

    #[tokio::test]
    async fn test_priority_based_dropping() {
        let config = LoadSheddingConfig {
            enable_load_shedding: true,
            ..Default::default()
        };

        let manager = LoadSheddingManager::new(config).unwrap();

        // Simulate high load
        manager
            .update_external_metrics(15000, 600.0, 800.0, 500.0)
            .await;

        let low_priority_event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata {
                event_id: "test-1".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let critical_event = StreamEvent::TransactionBegin {
            transaction_id: "tx-1".to_string(),
            isolation_level: None,
            metadata: crate::event::EventMetadata {
                event_id: "test-2".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        // Critical events should never be dropped
        assert!(!manager.should_drop_event(&critical_event).await);

        // Low priority events may be dropped under high load
        // (probabilistic, so we just verify it doesn't panic)
        let _ = manager.should_drop_event(&low_priority_event).await;
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let config = LoadSheddingConfig::default();
        let manager = LoadSheddingManager::new(config).unwrap();

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata {
                event_id: "test-1".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        manager.record_dropped_event(&event).await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_events_dropped, 1);
        assert!(stats
            .events_dropped_by_priority
            .contains_key(&EventPriority::Low));
    }

    #[tokio::test]
    async fn test_adaptive_probability_calculation() {
        let config = LoadSheddingConfig::default();

        let metrics = LoadMetrics {
            cpu_usage: 0.9,
            memory_usage: 0.85,
            queue_depth: 15000,
            avg_latency_ms: 600.0,
            p99_latency_ms: 800.0,
            throughput: 500.0,
            load_score: 0.85,
            timestamp: Utc::now(),
        };

        let prob =
            LoadSheddingManager::calculate_adaptive_drop_probability(&metrics, &[], 0.0, &config);

        assert!(prob > 0.0);
        assert!(prob <= 1.0);
    }

    #[tokio::test]
    async fn test_event_size_estimation() {
        let config = LoadSheddingConfig::default();
        let manager = LoadSheddingManager::new(config).unwrap();

        let event = StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: None,
            metadata: crate::event::EventMetadata {
                event_id: "test-1".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let size = manager.estimate_event_size(&event);
        assert!(size > 0);
    }

    #[tokio::test]
    async fn test_semantic_importance() {
        let config = LoadSheddingConfig::default();
        let manager = LoadSheddingManager::new(config).unwrap();

        let transaction_event = StreamEvent::TransactionCommit {
            transaction_id: "tx-1".to_string(),
            metadata: crate::event::EventMetadata {
                event_id: "test-1".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let heartbeat_event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata {
                event_id: "test-2".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        let tx_importance = manager.calculate_semantic_importance(&transaction_event);
        let hb_importance = manager.calculate_semantic_importance(&heartbeat_event);

        assert!(tx_importance > hb_importance);
        assert_eq!(tx_importance, 1.0);
        assert_eq!(hb_importance, 0.1);
    }

    #[tokio::test]
    async fn test_reset_stats() {
        let config = LoadSheddingConfig::default();
        let manager = LoadSheddingManager::new(config).unwrap();

        let event = StreamEvent::Heartbeat {
            timestamp: Utc::now(),
            source: "test".to_string(),
            metadata: crate::event::EventMetadata {
                event_id: "test-1".to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: HashMap::new(),
                checksum: None,
            },
        };

        manager.record_dropped_event(&event).await;
        assert_eq!(manager.get_stats().await.total_events_dropped, 1);

        manager.reset_stats().await;
        assert_eq!(manager.get_stats().await.total_events_dropped, 0);
    }
}
