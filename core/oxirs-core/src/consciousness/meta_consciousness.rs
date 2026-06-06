//! Meta-consciousness: self-awareness, component synchronization, and
//! cross-module communication.
//!
//! This module provides [`MetaConsciousness`], a component that tracks the
//! effectiveness of the individual consciousness subsystems, synchronizes
//! them, exchanges [`ConsciousnessMessage`]s between components, and produces
//! [`AdaptiveRecommendations`] for tuning consciousness parameters.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Meta-consciousness component for self-awareness and integration optimization
#[derive(Debug, Clone)]
pub struct MetaConsciousness {
    /// Self-awareness level (0.0 to 1.0)
    pub self_awareness: f64,
    /// Effectiveness tracking across consciousness components
    pub component_effectiveness: HashMap<String, f64>,
    /// Integration synchronization state
    pub sync_state: IntegrationSyncState,
    /// Performance history for adaptive learning
    pub performance_history: Vec<PerformanceMetric>,
    /// Cross-module communication channels
    pub communication_channels: Arc<RwLock<HashMap<String, ConsciousnessMessage>>>,
    /// Last synchronization time
    pub last_sync: std::time::Instant,
}

/// Integration synchronization state between consciousness components
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationSyncState {
    /// All components synchronized
    Synchronized,
    /// Components partially synchronized
    PartialSync,
    /// Synchronization in progress
    Synchronizing,
    /// Components need synchronization
    NeedsSync,
    /// Synchronization failed
    SyncFailed,
}

/// Performance metric for adaptive consciousness evolution
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    /// Timestamp of measurement
    pub timestamp: std::time::Instant,
    /// Query processing time improvement
    pub processing_improvement: f64,
    /// Accuracy improvement
    pub accuracy_improvement: f64,
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
    /// User satisfaction proxy
    pub satisfaction_proxy: f64,
}

/// Inter-component consciousness communication message
#[derive(Debug, Clone)]
pub struct ConsciousnessMessage {
    /// Source component
    pub source: String,
    /// Target component
    pub target: String,
    /// Message type
    pub message_type: MessageType,
    /// Message content
    pub content: String,
    /// Priority level
    pub priority: f64,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Types of consciousness communication messages
#[derive(Debug, Clone, PartialEq)]
pub enum MessageType {
    /// Emotional state change notification
    EmotionalStateChange,
    /// Quantum measurement result
    QuantumMeasurement,
    /// Dream insight discovery
    DreamInsight,
    /// Pattern recognition alert
    PatternAlert,
    /// Performance optimization suggestion
    OptimizationSuggestion,
    /// Synchronization request
    SyncRequest,
    /// Error or anomaly detected
    AnomalyDetection,
}

impl Default for MetaConsciousness {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaConsciousness {
    /// Create a new meta-consciousness component
    pub fn new() -> Self {
        Self {
            self_awareness: 0.3,
            component_effectiveness: HashMap::new(),
            sync_state: IntegrationSyncState::NeedsSync,
            performance_history: Vec::with_capacity(1000),
            communication_channels: Arc::new(RwLock::new(HashMap::new())),
            last_sync: std::time::Instant::now(),
        }
    }

    /// Update component effectiveness based on performance
    pub fn update_component_effectiveness(&mut self, component: &str, effectiveness: f64) {
        self.component_effectiveness
            .insert(component.to_string(), effectiveness);

        // Increase self-awareness as we learn about component effectiveness
        self.self_awareness = (self.self_awareness + 0.01).min(1.0);

        // Record performance metric
        let metric = PerformanceMetric {
            timestamp: std::time::Instant::now(),
            processing_improvement: effectiveness * 0.5,
            accuracy_improvement: effectiveness * 0.3,
            resource_efficiency: effectiveness * 0.4,
            satisfaction_proxy: effectiveness * 0.6,
        };

        self.performance_history.push(metric);

        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
    }

    /// Send a consciousness message between components
    pub fn send_message(&self, message: ConsciousnessMessage) -> Result<(), crate::OxirsError> {
        match self.communication_channels.write() {
            Ok(mut channels) => {
                let key = format!("{}_{}", message.source, message.target);
                channels.insert(key, message);
                Ok(())
            }
            _ => Err(crate::OxirsError::Query(
                "Failed to send consciousness message".to_string(),
            )),
        }
    }

    /// Receive consciousness messages for a component
    pub fn receive_messages(
        &self,
        component: &str,
    ) -> Result<Vec<ConsciousnessMessage>, crate::OxirsError> {
        match self.communication_channels.read() {
            Ok(channels) => {
                let messages: Vec<ConsciousnessMessage> = channels
                    .values()
                    .filter(|msg| msg.target == component)
                    .cloned()
                    .collect();
                Ok(messages)
            }
            _ => Err(crate::OxirsError::Query(
                "Failed to receive consciousness messages".to_string(),
            )),
        }
    }

    /// Synchronize all consciousness components
    pub fn synchronize_components(&mut self) -> Result<IntegrationSyncState, crate::OxirsError> {
        self.sync_state = IntegrationSyncState::Synchronizing;

        // Calculate overall effectiveness
        let overall_effectiveness: f64 = self.component_effectiveness.values().sum::<f64>()
            / self.component_effectiveness.len().max(1) as f64;

        // Update self-awareness based on overall effectiveness
        if overall_effectiveness > 0.8 {
            self.self_awareness = (self.self_awareness + 0.05).min(1.0);
            self.sync_state = IntegrationSyncState::Synchronized;
        } else if overall_effectiveness > 0.6 {
            self.sync_state = IntegrationSyncState::PartialSync;
        } else {
            self.sync_state = IntegrationSyncState::NeedsSync;
        }

        self.last_sync = std::time::Instant::now();
        Ok(self.sync_state.clone())
    }

    /// Calculate adaptive consciousness recommendations
    pub fn calculate_adaptive_recommendations(&self) -> AdaptiveRecommendations {
        let recent_performance: f64 = self
            .performance_history
            .iter()
            .rev()
            .take(10)
            .map(|p| {
                (p.processing_improvement + p.accuracy_improvement + p.resource_efficiency) / 3.0
            })
            .sum::<f64>()
            / 10.0;

        AdaptiveRecommendations {
            recommended_consciousness_level: self.self_awareness + recent_performance * 0.2,
            recommended_integration_level: if recent_performance > 0.7 { 0.9 } else { 0.6 },
            suggested_optimizations: self.generate_optimization_suggestions(),
            confidence: self.self_awareness * 0.8 + recent_performance * 0.2,
        }
    }

    /// Generate optimization suggestions based on performance history
    fn generate_optimization_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if let Some(avg_processing) = self.calculate_average_metric(|m| m.processing_improvement) {
            if avg_processing < 0.5 {
                suggestions.push("Increase quantum enhancement usage".to_string());
                suggestions.push("Optimize emotional learning parameters".to_string());
            }
        }

        if let Some(avg_accuracy) = self.calculate_average_metric(|m| m.accuracy_improvement) {
            if avg_accuracy < 0.6 {
                suggestions.push("Enable dream processing for pattern discovery".to_string());
                suggestions.push("Adjust intuitive planner sensitivity".to_string());
            }
        }

        if let Some(avg_efficiency) = self.calculate_average_metric(|m| m.resource_efficiency) {
            if avg_efficiency < 0.7 {
                suggestions.push("Balance consciousness levels for efficiency".to_string());
                suggestions.push("Optimize component synchronization frequency".to_string());
            }
        }

        suggestions
    }

    /// Calculate average for a specific metric
    fn calculate_average_metric<F>(&self, metric_extractor: F) -> Option<f64>
    where
        F: Fn(&PerformanceMetric) -> f64,
    {
        if self.performance_history.is_empty() {
            return None;
        }

        let sum: f64 = self.performance_history.iter().map(metric_extractor).sum();
        Some(sum / self.performance_history.len() as f64)
    }
}

/// Adaptive recommendations from meta-consciousness analysis
#[derive(Debug, Clone)]
pub struct AdaptiveRecommendations {
    /// Recommended consciousness level
    pub recommended_consciousness_level: f64,
    /// Recommended integration level
    pub recommended_integration_level: f64,
    /// Suggested optimizations
    pub suggested_optimizations: Vec<String>,
    /// Confidence in recommendations
    pub confidence: f64,
}
