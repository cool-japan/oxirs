// Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Query Anomaly Detection
//!
//! This module provides AI-powered anomaly detection for GraphQL queries,
//! identifying suspicious patterns, potential attacks, and unusual behavior.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Anomaly severity level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnomaly {
    /// Anomaly type
    pub anomaly_type: String,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Anomaly score (0.0 - 1.0, higher = more anomalous)
    pub score: f32,
    /// Description
    pub description: String,
    /// Affected query
    pub query_hash: String,
    /// Detected timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl QueryAnomaly {
    pub fn new(anomaly_type: String, severity: AnomalySeverity, score: f32) -> Self {
        Self {
            anomaly_type,
            severity,
            score,
            description: String::new(),
            query_hash: String::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Query features for anomaly detection
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    /// Query depth
    pub depth: usize,
    /// Number of fields
    pub field_count: usize,
    /// Complexity score
    pub complexity: f32,
    /// Execution time (ms)
    pub execution_time_ms: f64,
    /// Result size (bytes)
    pub result_size_bytes: usize,
}

/// Anomaly detection engine
pub struct AnomalyDetectionEngine {
    /// Baseline model for normal behavior
    baseline: Arc<RwLock<BaselineModel>>,
    /// Detection rules
    rules: Arc<RwLock<Vec<DetectionRule>>>,
    /// Detected anomalies
    anomalies: Arc<RwLock<Vec<QueryAnomaly>>>,
}

#[derive(Debug, Clone)]
pub struct BaselineModel {
    /// Mean values for each feature
    means: HashMap<String, f32>,
    /// Standard deviations
    std_devs: HashMap<String, f32>,
    /// Sample count
    sample_count: usize,
}

impl BaselineModel {
    pub fn new() -> Self {
        Self {
            means: HashMap::new(),
            std_devs: HashMap::new(),
            sample_count: 0,
        }
    }

    pub fn update(&mut self, features: &QueryFeatures) {
        self.sample_count += 1;
        self.update_stat("depth", features.depth as f32);
        self.update_stat("field_count", features.field_count as f32);
        self.update_stat("complexity", features.complexity);
        self.update_stat("execution_time", features.execution_time_ms as f32);
    }

    fn update_stat(&mut self, key: &str, value: f32) {
        let mean = self.means.entry(key.to_string()).or_insert(0.0);
        let old_mean = *mean;
        *mean = old_mean + (value - old_mean) / self.sample_count as f32;

        let std_dev = self.std_devs.entry(key.to_string()).or_insert(1.0);
        if self.sample_count > 1 {
            *std_dev = ((*std_dev).powi(2) + (value - old_mean) * (value - *mean)).sqrt();
        }
    }

    pub fn is_anomalous(&self, features: &QueryFeatures) -> f32 {
        let mut anomaly_score = 0.0;
        let mut count = 0;

        for (key, value) in [
            ("depth", features.depth as f32),
            ("field_count", features.field_count as f32),
            ("complexity", features.complexity),
        ] {
            if let (Some(&mean), Some(&std_dev)) = (self.means.get(key), self.std_devs.get(key)) {
                let z_score = ((value - mean) / std_dev.max(0.1)).abs();
                anomaly_score += z_score;
                count += 1;
            }
        }

        if count > 0 {
            (anomaly_score / count as f32).min(1.0)
        } else {
            0.0
        }
    }
}

impl Default for BaselineModel {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct DetectionRule {
    pub name: String,
    pub check: fn(&QueryFeatures) -> Option<QueryAnomaly>,
}

impl AnomalyDetectionEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            baseline: Arc::new(RwLock::new(BaselineModel::new())),
            rules: Arc::new(RwLock::new(Vec::new())),
            anomalies: Arc::new(RwLock::new(Vec::new())),
        };
        engine.init_default_rules();
        engine
    }

    fn init_default_rules(&mut self) {
        // Rules will be added via add_rule method
    }

    pub async fn add_rule(&self, rule: DetectionRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.push(rule);
        Ok(())
    }

    pub async fn detect(&self, features: QueryFeatures) -> Result<Vec<QueryAnomaly>> {
        let mut detected_anomalies = Vec::new();

        // Statistical anomaly detection
        let baseline = self.baseline.read().await;
        let anomaly_score = baseline.is_anomalous(&features);
        if anomaly_score > 0.8 {
            detected_anomalies.push(QueryAnomaly::new(
                "Statistical".to_string(),
                AnomalySeverity::High,
                anomaly_score,
            ));
        }

        // Rule-based detection
        let rules = self.rules.read().await;
        for rule in rules.iter() {
            if let Some(anomaly) = (rule.check)(&features) {
                detected_anomalies.push(anomaly);
            }
        }

        // Store anomalies
        let mut anomalies = self.anomalies.write().await;
        anomalies.extend(detected_anomalies.clone());

        Ok(detected_anomalies)
    }

    pub async fn train(&self, features: QueryFeatures) -> Result<()> {
        let mut baseline = self.baseline.write().await;
        baseline.update(&features);
        Ok(())
    }

    pub async fn get_anomalies(&self) -> Vec<QueryAnomaly> {
        let anomalies = self.anomalies.read().await;
        anomalies.clone()
    }
}

impl Default for AnomalyDetectionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_creation() {
        let anomaly = QueryAnomaly::new("Test".to_string(), AnomalySeverity::High, 0.9);
        assert_eq!(anomaly.severity, AnomalySeverity::High);
        assert_eq!(anomaly.score, 0.9);
    }

    #[test]
    fn test_baseline_model() {
        let mut model = BaselineModel::new();
        let features = QueryFeatures {
            depth: 3,
            field_count: 5,
            complexity: 10.0,
            execution_time_ms: 100.0,
            result_size_bytes: 1024,
        };
        model.update(&features);
        assert_eq!(model.sample_count, 1);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut model = BaselineModel::new();
        // Train with normal queries
        for _ in 0..10 {
            model.update(&QueryFeatures {
                depth: 3,
                field_count: 5,
                complexity: 10.0,
                execution_time_ms: 100.0,
                result_size_bytes: 1024,
            });
        }

        // Test with anomalous query
        let anomalous = QueryFeatures {
            depth: 20,
            field_count: 100,
            complexity: 1000.0,
            execution_time_ms: 5000.0,
            result_size_bytes: 1024000,
        };
        let score = model.is_anomalous(&anomalous);
        assert!(score > 0.5);
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = AnomalyDetectionEngine::new();
        let anomalies = engine.get_anomalies().await;
        assert!(anomalies.is_empty());
    }

    #[tokio::test]
    async fn test_train_and_detect() {
        let engine = AnomalyDetectionEngine::new();

        // Train
        for _ in 0..10 {
            engine
                .train(QueryFeatures {
                    depth: 3,
                    field_count: 5,
                    complexity: 10.0,
                    execution_time_ms: 100.0,
                    result_size_bytes: 1024,
                })
                .await
                .unwrap();
        }

        // Detect anomaly
        let result = engine
            .detect(QueryFeatures {
                depth: 20,
                field_count: 100,
                complexity: 1000.0,
                execution_time_ms: 5000.0,
                result_size_bytes: 1024000,
            })
            .await
            .unwrap();

        assert!(!result.is_empty());
    }
}
