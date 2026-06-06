//! Anomaly detection engine used by the system monitoring stack.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use uuid::Uuid;

use crate::system_monitoring_types::{AnomalyType, PerformanceMetric};
use crate::Result;

/// Advanced anomaly detection engine for intelligent monitoring
#[derive(Debug)]
pub struct AnomalyDetector {
    detection_models: HashMap<String, AnomalyModel>,
    baseline_profiles: HashMap<String, BaselineProfile>,
    detection_config: AnomalyDetectionConfig,
    anomaly_history: VecDeque<AnomalyEvent>,
}

/// Configuration for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    pub sensitivity_threshold: f64,
    pub learning_period_hours: u64,
    pub enable_statistical_detection: bool,
    pub enable_ml_detection: bool,
    pub enable_pattern_detection: bool,
    pub min_anomaly_confidence: f64,
    pub rolling_window_size: usize,
    pub enable_seasonal_decomposition: bool,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            sensitivity_threshold: 0.7,
            learning_period_hours: 24,
            enable_statistical_detection: true,
            enable_ml_detection: true,
            enable_pattern_detection: true,
            min_anomaly_confidence: 0.8,
            rolling_window_size: 50,
            enable_seasonal_decomposition: true,
        }
    }
}

/// Anomaly detection model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyModel {
    pub model_type: AnomalyModelType,
    pub training_data_size: usize,
    pub last_training: DateTime<Utc>,
    pub accuracy_score: f64,
    pub parameters: HashMap<String, f64>,
    pub feature_importance: HashMap<String, f64>,
}

/// Types of anomaly detection models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyModelType {
    Statistical,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    Autoencoder,
    TemporalPattern,
}

/// Baseline performance profile for normal operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineProfile {
    pub metric_name: String,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub sample_count: usize,
    pub last_updated: DateTime<Utc>,
    pub seasonal_patterns: Option<SeasonalPatternData>,
}

/// Seasonal pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPatternData {
    pub daily_patterns: Vec<f64>,
    pub weekly_patterns: Vec<f64>,
    pub monthly_patterns: Option<Vec<f64>>,
    pub trend_component: f64,
    pub seasonal_strength: f64,
}

/// Anomaly event detected by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub metric_name: String,
    pub observed_value: f64,
    pub expected_value: f64,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub anomaly_type: AnomalyType,
    pub detection_method: AnomalyModelType,
    pub impact_assessment: ImpactAssessment,
    pub related_metrics: Vec<String>,
    pub root_cause_hints: Vec<String>,
}

/// Impact assessment for anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub severity: AnomalySeverity,
    pub affected_components: Vec<String>,
    pub business_impact: BusinessImpact,
    pub estimated_user_impact: f64,
    pub recovery_time_estimate: Option<Duration>,
}

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Business impact categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusinessImpact {
    Negligible,
    Minor,
    Moderate,
    Significant,
    Severe,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self::with_config(AnomalyDetectionConfig::default())
    }

    pub fn with_config(config: AnomalyDetectionConfig) -> Self {
        Self {
            detection_models: HashMap::new(),
            baseline_profiles: HashMap::new(),
            detection_config: config,
            anomaly_history: VecDeque::new(),
        }
    }

    /// Analyze metrics for anomalies
    pub fn analyze_metrics(&mut self, metrics: &[PerformanceMetric]) -> Result<Vec<AnomalyEvent>> {
        let mut anomalies = Vec::new();

        for metric in metrics {
            if self.detection_config.enable_statistical_detection {
                if let Some(stat_anomaly) = self.detect_statistical_anomaly(metric)? {
                    anomalies.push(stat_anomaly);
                }
            }

            if self.detection_config.enable_pattern_detection {
                if let Some(pattern_anomaly) = self.detect_pattern_anomaly(metric)? {
                    anomalies.push(pattern_anomaly);
                }
            }

            if self.detection_config.enable_ml_detection {
                if let Some(ml_anomaly) = self.detect_ml_anomaly(metric)? {
                    anomalies.push(ml_anomaly);
                }
            }
        }

        for anomaly in &anomalies {
            self.anomaly_history.push_back(anomaly.clone());

            if self.anomaly_history.len() > 1000 {
                self.anomaly_history.pop_front();
            }
        }

        Ok(anomalies)
    }

    fn detect_statistical_anomaly(
        &mut self,
        metric: &PerformanceMetric,
    ) -> Result<Option<AnomalyEvent>> {
        let metric_name = "response_time_ms";
        let value = metric.response_time_ms;

        if !self.baseline_profiles.contains_key(metric_name) {
            self.initialize_baseline_profile(metric_name, value)?;
            return Ok(None);
        }

        let baseline = self
            .baseline_profiles
            .get(metric_name)
            .expect("baseline profile should exist after contains_key check");

        let z_score = (value - baseline.mean) / baseline.std_dev;
        let z_threshold = 3.0 * (1.0 - self.detection_config.sensitivity_threshold);

        if z_score.abs() > z_threshold {
            let anomaly_score = z_score.abs() / z_threshold;
            let confidence = (anomaly_score - 1.0).clamp(0.0, 1.0);

            if confidence >= self.detection_config.min_anomaly_confidence {
                let anomaly_type = if z_score > 0.0 {
                    AnomalyType::PositiveSpike
                } else {
                    AnomalyType::NegativeDip
                };

                let root_cause_hints = self.generate_root_cause_hints(metric_name, &anomaly_type);

                let event = AnomalyEvent {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp: metric.timestamp,
                    metric_name: metric_name.to_string(),
                    observed_value: value,
                    expected_value: baseline.mean,
                    anomaly_score,
                    confidence,
                    anomaly_type,
                    detection_method: AnomalyModelType::Statistical,
                    impact_assessment: self.assess_impact(metric_name, anomaly_score),
                    related_metrics: vec![
                        "cpu_usage_percent".to_string(),
                        "memory_usage_mb".to_string(),
                    ],
                    root_cause_hints,
                };

                return Ok(Some(event));
            }
        }

        Ok(None)
    }

    fn detect_pattern_anomaly(&self, metric: &PerformanceMetric) -> Result<Option<AnomalyEvent>> {
        let response_time = metric.response_time_ms;
        let cpu_usage = metric.cpu_usage_percent;

        if response_time > 1000.0 && cpu_usage < 20.0 {
            let anomaly_score = response_time / 1000.0 * (1.0 - cpu_usage / 100.0);
            let confidence = 0.8;

            if confidence >= self.detection_config.min_anomaly_confidence {
                let event = AnomalyEvent {
                    event_id: Uuid::new_v4().to_string(),
                    timestamp: metric.timestamp,
                    metric_name: "response_time_cpu_correlation".to_string(),
                    observed_value: response_time,
                    expected_value: cpu_usage * 10.0,
                    anomaly_score,
                    confidence,
                    anomaly_type: AnomalyType::CorrelationBreakdown,
                    detection_method: AnomalyModelType::TemporalPattern,
                    impact_assessment: self.assess_impact("correlation", anomaly_score),
                    related_metrics: vec![
                        "response_time_ms".to_string(),
                        "cpu_usage_percent".to_string(),
                    ],
                    root_cause_hints: vec![
                        "I/O bottleneck possible".to_string(),
                        "Database connection issues".to_string(),
                        "Network latency".to_string(),
                    ],
                };

                return Ok(Some(event));
            }
        }

        Ok(None)
    }

    fn detect_ml_anomaly(&self, _metric: &PerformanceMetric) -> Result<Option<AnomalyEvent>> {
        Ok(None)
    }

    fn initialize_baseline_profile(&mut self, metric_name: &str, initial_value: f64) -> Result<()> {
        let profile = BaselineProfile {
            metric_name: metric_name.to_string(),
            mean: initial_value,
            std_dev: 0.0,
            median: initial_value,
            percentile_95: initial_value,
            percentile_99: initial_value,
            min_value: initial_value,
            max_value: initial_value,
            sample_count: 1,
            last_updated: Utc::now(),
            seasonal_patterns: None,
        };

        self.baseline_profiles
            .insert(metric_name.to_string(), profile);
        Ok(())
    }

    fn assess_impact(&self, metric_name: &str, anomaly_score: f64) -> ImpactAssessment {
        let severity = if anomaly_score > 5.0 {
            AnomalySeverity::Critical
        } else if anomaly_score > 3.0 {
            AnomalySeverity::High
        } else if anomaly_score > 2.0 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        };

        let business_impact = match severity {
            AnomalySeverity::Critical => BusinessImpact::Severe,
            AnomalySeverity::High => BusinessImpact::Significant,
            AnomalySeverity::Medium => BusinessImpact::Moderate,
            AnomalySeverity::Low => BusinessImpact::Minor,
        };

        let affected_components = match metric_name {
            "response_time_ms" => vec!["web_server".to_string(), "database".to_string()],
            "cpu_usage_percent" => vec!["compute_engine".to_string()],
            "memory_usage_mb" => vec!["memory_manager".to_string()],
            _ => vec!["unknown".to_string()],
        };

        ImpactAssessment {
            severity,
            affected_components,
            business_impact,
            estimated_user_impact: anomaly_score / 10.0,
            recovery_time_estimate: Some(Duration::from_secs(300)),
        }
    }

    fn generate_root_cause_hints(
        &self,
        metric_name: &str,
        anomaly_type: &AnomalyType,
    ) -> Vec<String> {
        match (metric_name, anomaly_type) {
            ("response_time_ms", AnomalyType::PositiveSpike) => vec![
                "High database load".to_string(),
                "Network congestion".to_string(),
                "Resource contention".to_string(),
                "Cache miss spike".to_string(),
            ],
            ("cpu_usage_percent", AnomalyType::PositiveSpike) => vec![
                "CPU-intensive operation".to_string(),
                "Infinite loop or deadlock".to_string(),
                "Background process consuming CPU".to_string(),
            ],
            ("memory_usage_mb", AnomalyType::PositiveSpike) => vec![
                "Memory leak detected".to_string(),
                "Large dataset processing".to_string(),
                "Caching strategy ineffective".to_string(),
            ],
            _ => vec!["Unknown root cause".to_string()],
        }
    }

    pub fn get_anomaly_history(&self) -> &VecDeque<AnomalyEvent> {
        &self.anomaly_history
    }

    pub fn update_baselines(&mut self, metrics: &[PerformanceMetric]) -> Result<()> {
        for metric in metrics {
            self.update_baseline_for_metric("response_time_ms", metric.response_time_ms)?;
            self.update_baseline_for_metric("cpu_usage_percent", metric.cpu_usage_percent)?;
            self.update_baseline_for_metric("memory_usage_mb", metric.memory_usage_mb)?;
        }
        Ok(())
    }

    fn update_baseline_for_metric(&mut self, metric_name: &str, value: f64) -> Result<()> {
        if let Some(profile) = self.baseline_profiles.get_mut(metric_name) {
            let n = profile.sample_count as f64;
            let new_mean = (profile.mean * n + value) / (n + 1.0);
            let new_variance =
                ((n - 1.0) * profile.std_dev.powi(2) + (value - new_mean).powi(2)) / n;

            profile.mean = new_mean;
            profile.std_dev = new_variance.sqrt();
            profile.min_value = profile.min_value.min(value);
            profile.max_value = profile.max_value.max(value);
            profile.sample_count += 1;
            profile.last_updated = Utc::now();
        }
        Ok(())
    }

    /// Internal accessor for detection models (read-only).
    pub fn detection_models(&self) -> &HashMap<String, AnomalyModel> {
        &self.detection_models
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}
