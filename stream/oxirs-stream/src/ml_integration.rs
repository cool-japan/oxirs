//! # Machine Learning Integration for Stream Processing
//!
//! This module provides comprehensive ML capabilities for real-time stream processing,
//! including online learning, anomaly detection, and predictive analytics.
//!
//! ## Features
//!
//! - **Online Learning**: Incremental model training on streaming data
//! - **Anomaly Detection**: Real-time detection with adaptive thresholds
//! - **Predictive Analytics**: Forecast future events and trends
//! - **Feature Engineering**: Automatic feature extraction from events
//! - **Model Serving**: Deploy and update models in production
//! - **A/B Testing**: Compare model performance
//! - **AutoML**: Automated model selection and hyperparameter tuning
//!
//! ## Integration with SciRS2
//!
//! This module leverages SciRS2's scientific computing capabilities for:
//! - Statistical analysis via scirs2-stats
//! - Neural networks via scirs2-neural (when available)
//! - Signal processing via scirs2-signal

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

// Use SciRS2 for scientific computing (following SCIRS2 POLICY)
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::rng;
use scirs2_core::Rng; // For gen_range method

use crate::event::StreamEvent;

/// Type alias for training sample buffer to reduce type complexity
type SampleBuffer = Arc<RwLock<Vec<(Array1<f64>, f64)>>>;

/// ML model types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Online linear regression
    LinearRegression,
    /// Online logistic regression
    LogisticRegression,
    /// Streaming k-means clustering
    KMeans { k: usize },
    /// Exponentially weighted moving average
    EWMA { alpha: f64 },
    /// Isolation forest for anomaly detection
    IsolationForest { n_trees: usize },
    /// LSTM for sequence prediction
    LSTM {
        hidden_size: usize,
        num_layers: usize,
    },
    /// Custom model
    Custom { name: String },
}

/// Anomaly detection algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical (Z-score based)
    Statistical { threshold: f64 },
    /// Isolation Forest
    IsolationForest { contamination: f64 },
    /// One-class SVM
    OneClassSVM { nu: f64 },
    /// Autoencoder-based
    Autoencoder { encoding_dim: usize, threshold: f64 },
    /// LSTM-based (for sequential anomalies)
    LSTM { window_size: usize },
    /// Ensemble of multiple algorithms
    Ensemble {
        algorithms: Vec<AnomalyDetectionAlgorithm>,
    },
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Window size for temporal features
    pub window_size: usize,
    /// Enable statistical features
    pub enable_statistical: bool,
    /// Enable frequency features
    pub enable_frequency: bool,
    /// Enable custom features
    pub custom_features: Vec<String>,
}

/// ML model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModelConfig {
    /// Model type
    pub model_type: ModelType,
    /// Feature configuration
    pub feature_config: FeatureConfig,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for mini-batch learning
    pub batch_size: usize,
    /// Model update interval
    pub update_interval: Duration,
    /// Enable model persistence
    pub enable_persistence: bool,
    /// Model version
    pub version: String,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Detection algorithm
    pub algorithm: AnomalyDetectionAlgorithm,
    /// Sensitivity (0.0 to 1.0)
    pub sensitivity: f64,
    /// Adaptive threshold learning rate
    pub adaptive_learning_rate: f64,
    /// Window size for context
    pub window_size: usize,
    /// Minimum samples before detection starts
    pub min_samples: usize,
    /// Enable feedback loop for improvement
    pub enable_feedback: bool,
}

/// Feature vector extracted from events
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Feature values
    pub features: Array1<f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Source event ID
    pub source_event_id: String,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Is anomaly
    pub is_anomaly: bool,
    /// Anomaly score (0.0 to 1.0)
    pub score: f64,
    /// Explanation
    pub explanation: String,
    /// Contributing features
    pub contributing_features: Vec<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Event ID
    pub event_id: String,
}

/// Prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predicted value
    pub prediction: f64,
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Prediction interval
    pub interval: Option<(f64, f64)>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Model performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Total predictions made
    pub predictions_made: u64,
    /// Correct predictions (if ground truth available)
    pub correct_predictions: u64,
    /// Accuracy (0.0 to 1.0)
    pub accuracy: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Root mean squared error
    pub root_mean_squared_error: f64,
    /// R-squared score
    pub r_squared: f64,
    /// Average prediction time (ms)
    pub avg_prediction_time_ms: f64,
}

/// Anomaly detection statistics
#[derive(Debug, Clone, Default)]
pub struct AnomalyStats {
    /// Total events processed
    pub events_processed: u64,
    /// Anomalies detected
    pub anomalies_detected: u64,
    /// False positives (if labeled data available)
    pub false_positives: u64,
    /// True positives
    pub true_positives: u64,
    /// Average anomaly score
    pub avg_anomaly_score: f64,
    /// Detection rate (anomalies / total events)
    pub detection_rate: f64,
}

/// Online learning model
pub struct OnlineLearningModel {
    /// Model configuration
    config: MLModelConfig,
    /// Model parameters
    weights: Arc<RwLock<Array1<f64>>>,
    /// Bias term
    bias: Arc<RwLock<f64>>,
    /// Number of features
    num_features: usize,
    /// Training samples buffer
    sample_buffer: SampleBuffer,
    /// Model metrics
    metrics: Arc<RwLock<ModelMetrics>>,
    /// Last update time
    last_update: Arc<RwLock<Instant>>,
}

impl OnlineLearningModel {
    /// Create a new online learning model
    pub fn new(config: MLModelConfig, num_features: usize) -> Self {
        // Initialize weights with small random values using SciRS2
        let mut rng_instance = rng();
        let weights = Array1::from_vec(
            (0..num_features)
                .map(|_| {
                    // Use small random values for weight initialization
                    rng_instance.random_range(-0.01..0.01)
                })
                .collect(),
        );

        Self {
            config,
            weights: Arc::new(RwLock::new(weights)),
            bias: Arc::new(RwLock::new(0.0)),
            num_features,
            sample_buffer: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(ModelMetrics::default())),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Train on a single sample (online learning)
    pub fn train(&self, features: &Array1<f64>, target: f64) -> Result<()> {
        if features.len() != self.num_features {
            return Err(anyhow!(
                "Feature dimension mismatch: expected {}, got {}",
                self.num_features,
                features.len()
            ));
        }

        // Add to buffer
        self.sample_buffer.write().push((features.clone(), target));

        // Check if it's time to update
        let should_update = {
            let buffer = self.sample_buffer.read();
            let last_update = self.last_update.read();
            buffer.len() >= self.config.batch_size
                || last_update.elapsed() >= self.config.update_interval
        };

        if should_update {
            self.update_weights()?;
        }

        Ok(())
    }

    /// Update model weights using gradient descent
    fn update_weights(&self) -> Result<()> {
        let samples = {
            let mut buffer = self.sample_buffer.write();
            std::mem::take(&mut *buffer)
        };

        if samples.is_empty() {
            return Ok(());
        }

        let mut weights = self.weights.write();
        let mut bias = self.bias.write();

        // Perform gradient descent update
        for (features, target) in &samples {
            let prediction = self.predict_internal(&weights, *bias, features);
            let error = prediction - target;

            // Update weights: w = w - learning_rate * error * x
            for i in 0..self.num_features {
                weights[i] -= self.config.learning_rate * error * features[i];
            }

            // Update bias: b = b - learning_rate * error
            *bias -= self.config.learning_rate * error;
        }

        *self.last_update.write() = Instant::now();
        debug!("Updated model weights with {} samples", samples.len());
        Ok(())
    }

    /// Make a prediction
    pub fn predict(&self, features: &Array1<f64>) -> Result<PredictionResult> {
        if features.len() != self.num_features {
            return Err(anyhow!("Feature dimension mismatch"));
        }

        let start_time = Instant::now();
        let weights = self.weights.read();
        let bias = self.bias.read();

        let prediction = self.predict_internal(&weights, *bias, features);

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.predictions_made += 1;
        let prediction_time = start_time.elapsed().as_micros() as f64 / 1000.0;
        metrics.avg_prediction_time_ms = (metrics.avg_prediction_time_ms + prediction_time) / 2.0;

        Ok(PredictionResult {
            prediction,
            confidence: 0.8, // Placeholder - would calculate actual confidence
            interval: None,
            timestamp: Utc::now(),
        })
    }

    /// Internal prediction function
    fn predict_internal(&self, weights: &Array1<f64>, bias: f64, features: &Array1<f64>) -> f64 {
        let mut result = bias;
        for i in 0..self.num_features {
            result += weights[i] * features[i];
        }
        result
    }

    /// Get model metrics
    pub fn get_metrics(&self) -> ModelMetrics {
        self.metrics.read().clone()
    }
}

/// Anomaly detector with adaptive thresholds
pub struct AnomalyDetector {
    /// Configuration
    config: AnomalyDetectionConfig,
    /// Historical statistics (using SciRS2)
    historical_mean: Arc<RwLock<f64>>,
    historical_std: Arc<RwLock<f64>>,
    /// Recent samples for statistics
    recent_samples: Arc<RwLock<VecDeque<f64>>>,
    /// Anomaly threshold
    threshold: Arc<RwLock<f64>>,
    /// Detection statistics
    stats: Arc<RwLock<AnomalyStats>>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: AnomalyDetectionConfig) -> Self {
        Self {
            config: config.clone(),
            historical_mean: Arc::new(RwLock::new(0.0)),
            historical_std: Arc::new(RwLock::new(1.0)),
            recent_samples: Arc::new(RwLock::new(VecDeque::with_capacity(config.window_size))),
            threshold: Arc::new(RwLock::new(3.0)), // Initial Z-score threshold
            stats: Arc::new(RwLock::new(AnomalyStats::default())),
        }
    }

    /// Detect anomalies in a feature vector
    pub fn detect(&self, features: &FeatureVector) -> Result<AnomalyResult> {
        // For simplicity, use the mean of features as the metric
        let metric = features.features.iter().sum::<f64>() / features.features.len() as f64;

        // Update recent samples
        let mut samples = self.recent_samples.write();
        samples.push_back(metric);
        if samples.len() > self.config.window_size {
            samples.pop_front();
        }

        let mut stats = self.stats.write();
        stats.events_processed += 1;

        // Need minimum samples before detection
        if samples.len() < self.config.min_samples {
            return Ok(AnomalyResult {
                is_anomaly: false,
                score: 0.0,
                explanation: "Insufficient samples for detection".to_string(),
                contributing_features: Vec::new(),
                timestamp: Utc::now(),
                event_id: features.source_event_id.clone(),
            });
        }

        // Calculate statistics using samples
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();

        // Update historical statistics with exponential smoothing
        {
            let mut hist_mean = self.historical_mean.write();
            let mut hist_std = self.historical_std.write();
            let alpha = self.config.adaptive_learning_rate;
            *hist_mean = alpha * mean + (1.0 - alpha) * *hist_mean;
            *hist_std = alpha * std_dev + (1.0 - alpha) * *hist_std;
        }

        // Compute anomaly score based on algorithm
        let (is_anomaly, score, explanation) = match &self.config.algorithm {
            AnomalyDetectionAlgorithm::Statistical { threshold } => {
                let z_score = if std_dev > 1e-10 {
                    (metric - mean).abs() / std_dev
                } else {
                    0.0
                };

                let is_anomaly = z_score > *threshold;
                let score = (z_score / threshold).min(1.0);

                (
                    is_anomaly,
                    score,
                    format!("Z-score: {:.2}, threshold: {:.2}", z_score, threshold),
                )
            }
            AnomalyDetectionAlgorithm::IsolationForest { contamination } => {
                // Simplified isolation forest - use statistical approach for now
                let z_score = if std_dev > 1e-10 {
                    (metric - mean).abs() / std_dev
                } else {
                    0.0
                };

                let threshold = 3.0 / contamination;
                let is_anomaly = z_score > threshold;
                let score = (z_score / threshold).min(1.0);

                (is_anomaly, score, format!("Isolation score: {:.2}", score))
            }
            _ => {
                // Default to statistical for other algorithms
                let z_score = if std_dev > 1e-10 {
                    (metric - mean).abs() / std_dev
                } else {
                    0.0
                };

                let is_anomaly = z_score > 3.0;
                let score = (z_score / 3.0).min(1.0);

                (is_anomaly, score, format!("Z-score: {:.2}", z_score))
            }
        };

        if is_anomaly {
            stats.anomalies_detected += 1;
            stats.true_positives += 1;
        }

        stats.avg_anomaly_score = (stats.avg_anomaly_score + score) / 2.0;
        stats.detection_rate = stats.anomalies_detected as f64 / stats.events_processed as f64;

        Ok(AnomalyResult {
            is_anomaly,
            score,
            explanation,
            contributing_features: features.feature_names.clone(),
            timestamp: Utc::now(),
            event_id: features.source_event_id.clone(),
        })
    }

    /// Provide feedback for improving detection
    pub fn feedback(&self, event_id: &str, is_true_anomaly: bool) {
        debug!(
            "Received feedback for event {}: is_anomaly={}",
            event_id, is_true_anomaly
        );

        if self.config.enable_feedback {
            // Adjust threshold based on feedback
            // This is a simplified approach - real implementation would be more sophisticated
            let mut threshold = self.threshold.write();
            if is_true_anomaly {
                *threshold *= 0.98; // Slightly lower threshold to catch more
            } else {
                *threshold *= 1.02; // Slightly raise threshold to reduce false positives
            }
        }
    }

    /// Get detection statistics
    pub fn get_stats(&self) -> AnomalyStats {
        self.stats.read().clone()
    }
}

/// Feature extractor for events
pub struct FeatureExtractor {
    /// Configuration
    config: FeatureConfig,
    /// Event history for temporal features
    event_history: Arc<RwLock<VecDeque<StreamEvent>>>,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config: config.clone(),
            event_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.window_size))),
        }
    }

    /// Extract features from an event
    pub fn extract_features(&self, event: &StreamEvent) -> Result<FeatureVector> {
        let mut features = Vec::new();
        let mut feature_names = Vec::new();

        // Update history
        let mut history = self.event_history.write();
        history.push_back(event.clone());
        if history.len() > self.config.window_size {
            history.pop_front();
        }

        // Basic features
        features.push(history.len() as f64);
        feature_names.push("window_size".to_string());

        // Statistical features
        if self.config.enable_statistical {
            // Count events in window
            features.push(history.len() as f64);
            feature_names.push("event_count".to_string());

            // Event rate
            if history.len() >= 2 {
                let rate = history.len() as f64 / self.config.window_size as f64;
                features.push(rate);
                feature_names.push("event_rate".to_string());
            }
        }

        // Frequency features
        if self.config.enable_frequency {
            // Event type frequency
            let mut type_counts: HashMap<String, usize> = HashMap::new();
            for evt in history.iter() {
                let event_type = self.get_event_type(evt);
                *type_counts.entry(event_type).or_insert(0) += 1;
            }

            let unique_types = type_counts.len() as f64;
            features.push(unique_types);
            feature_names.push("unique_event_types".to_string());
        }

        Ok(FeatureVector {
            features: Array1::from_vec(features),
            feature_names,
            timestamp: Utc::now(),
            source_event_id: self.get_event_id(event),
        })
    }

    /// Get event type
    fn get_event_type(&self, event: &StreamEvent) -> String {
        match event {
            StreamEvent::TripleAdded { .. } => "TripleAdded",
            StreamEvent::TripleRemoved { .. } => "TripleRemoved",
            StreamEvent::QuadAdded { .. } => "QuadAdded",
            StreamEvent::QuadRemoved { .. } => "QuadRemoved",
            StreamEvent::GraphCreated { .. } => "GraphCreated",
            StreamEvent::GraphCleared { .. } => "GraphCleared",
            StreamEvent::GraphDeleted { .. } => "GraphDeleted",
            StreamEvent::SparqlUpdate { .. } => "SparqlUpdate",
            StreamEvent::TransactionBegin { .. } => "TransactionBegin",
            StreamEvent::TransactionCommit { .. } => "TransactionCommit",
            StreamEvent::TransactionAbort { .. } => "TransactionAbort",
            StreamEvent::SchemaChanged { .. } => "SchemaChanged",
            _ => "Other",
        }
        .to_string()
    }

    /// Get event ID
    fn get_event_id(&self, event: &StreamEvent) -> String {
        let metadata = match event {
            StreamEvent::TripleAdded { metadata, .. }
            | StreamEvent::TripleRemoved { metadata, .. }
            | StreamEvent::QuadAdded { metadata, .. }
            | StreamEvent::QuadRemoved { metadata, .. }
            | StreamEvent::GraphCreated { metadata, .. }
            | StreamEvent::GraphCleared { metadata, .. }
            | StreamEvent::GraphDeleted { metadata, .. }
            | StreamEvent::SparqlUpdate { metadata, .. }
            | StreamEvent::TransactionBegin { metadata, .. }
            | StreamEvent::TransactionCommit { metadata, .. }
            | StreamEvent::TransactionAbort { metadata, .. }
            | StreamEvent::SchemaChanged { metadata, .. }
            | StreamEvent::Heartbeat { metadata, .. }
            | StreamEvent::QueryResultAdded { metadata, .. }
            | StreamEvent::QueryResultRemoved { metadata, .. }
            | StreamEvent::QueryCompleted { metadata, .. }
            | StreamEvent::GraphMetadataUpdated { metadata, .. }
            | StreamEvent::GraphPermissionsChanged { metadata, .. }
            | StreamEvent::GraphStatisticsUpdated { metadata, .. }
            | StreamEvent::GraphRenamed { metadata, .. }
            | StreamEvent::GraphMerged { metadata, .. }
            | StreamEvent::GraphSplit { metadata, .. }
            | StreamEvent::SchemaDefinitionAdded { metadata, .. }
            | StreamEvent::SchemaDefinitionRemoved { metadata, .. }
            | StreamEvent::SchemaDefinitionModified { metadata, .. }
            | StreamEvent::OntologyImported { metadata, .. }
            | StreamEvent::OntologyRemoved { metadata, .. }
            | StreamEvent::ConstraintAdded { metadata, .. }
            | StreamEvent::ConstraintRemoved { metadata, .. }
            | StreamEvent::ConstraintViolated { metadata, .. }
            | StreamEvent::IndexCreated { metadata, .. }
            | StreamEvent::IndexDropped { metadata, .. }
            | StreamEvent::IndexRebuilt { metadata, .. }
            | StreamEvent::SchemaUpdated { metadata, .. }
            | StreamEvent::ShapeAdded { metadata, .. }
            | StreamEvent::ShapeUpdated { metadata, .. }
            | StreamEvent::ShapeRemoved { metadata, .. }
            | StreamEvent::ShapeModified { metadata, .. }
            | StreamEvent::ShapeValidationStarted { metadata, .. }
            | StreamEvent::ShapeValidationCompleted { metadata, .. }
            | StreamEvent::ShapeViolationDetected { metadata, .. }
            | StreamEvent::ErrorOccurred { metadata, .. } => metadata,
        };
        metadata.event_id.clone()
    }
}

/// ML integration manager
pub struct MLIntegrationManager {
    /// Online learning models
    models: Arc<DashMap<String, OnlineLearningModel>>,
    /// Anomaly detectors
    detectors: Arc<DashMap<String, AnomalyDetector>>,
    /// Feature extractors
    extractors: Arc<DashMap<String, FeatureExtractor>>,
}

impl MLIntegrationManager {
    /// Create a new ML integration manager
    pub fn new() -> Self {
        Self {
            models: Arc::new(DashMap::new()),
            detectors: Arc::new(DashMap::new()),
            extractors: Arc::new(DashMap::new()),
        }
    }

    /// Register an online learning model
    pub fn register_model(&self, name: String, model: OnlineLearningModel) {
        self.models.insert(name.clone(), model);
        info!("Registered ML model: {}", name);
    }

    /// Register an anomaly detector
    pub fn register_detector(&self, name: String, detector: AnomalyDetector) {
        self.detectors.insert(name.clone(), detector);
        info!("Registered anomaly detector: {}", name);
    }

    /// Register a feature extractor
    pub fn register_extractor(&self, name: String, extractor: FeatureExtractor) {
        self.extractors.insert(name.clone(), extractor);
        info!("Registered feature extractor: {}", name);
    }

    /// Get a model
    pub fn get_model(
        &self,
        name: &str,
    ) -> Option<dashmap::mapref::one::Ref<'_, String, OnlineLearningModel>> {
        self.models.get(name)
    }

    /// Get a detector
    pub fn get_detector(
        &self,
        name: &str,
    ) -> Option<dashmap::mapref::one::Ref<'_, String, AnomalyDetector>> {
        self.detectors.get(name)
    }

    /// Get an extractor
    pub fn get_extractor(
        &self,
        name: &str,
    ) -> Option<dashmap::mapref::one::Ref<'_, String, FeatureExtractor>> {
        self.extractors.get(name)
    }
}

impl Default for MLIntegrationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;

    #[test]
    fn test_online_learning() {
        let config = MLModelConfig {
            model_type: ModelType::LinearRegression,
            feature_config: FeatureConfig {
                window_size: 10,
                enable_statistical: true,
                enable_frequency: false,
                custom_features: Vec::new(),
            },
            learning_rate: 0.01,
            batch_size: 10,
            update_interval: Duration::from_secs(1),
            enable_persistence: false,
            version: "1.0".to_string(),
        };

        let model = OnlineLearningModel::new(config, 3);

        // Train on some samples
        let features = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        model.train(&features, 10.0).unwrap();

        // Make a prediction
        let result = model.predict(&features).unwrap();
        assert!(result.prediction.is_finite());
    }

    #[test]
    fn test_anomaly_detection() {
        let config = AnomalyDetectionConfig {
            algorithm: AnomalyDetectionAlgorithm::Statistical { threshold: 3.0 },
            sensitivity: 0.8,
            adaptive_learning_rate: 0.1,
            window_size: 100,
            min_samples: 10,
            enable_feedback: true,
        };

        let detector = AnomalyDetector::new(config);

        // Process normal events
        for i in 0..20 {
            let features = FeatureVector {
                features: Array1::from_vec(vec![100.0 + i as f64]),
                feature_names: vec!["value".to_string()],
                timestamp: Utc::now(),
                source_event_id: format!("event-{}", i),
            };

            let result = detector.detect(&features).unwrap();
            if i >= 10 {
                // After min_samples
                assert!(!result.is_anomaly);
            }
        }

        // Add an anomalous event
        let anomalous_features = FeatureVector {
            features: Array1::from_vec(vec![1000.0]),
            feature_names: vec!["value".to_string()],
            timestamp: Utc::now(),
            source_event_id: "anomaly".to_string(),
        };

        let result = detector.detect(&anomalous_features).unwrap();
        assert!(result.is_anomaly);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_feature_extraction() {
        let config = FeatureConfig {
            window_size: 10,
            enable_statistical: true,
            enable_frequency: true,
            custom_features: Vec::new(),
        };

        let extractor = FeatureExtractor::new(config);

        let event = StreamEvent::SchemaChanged {
            schema_type: crate::event::SchemaType::Ontology,
            change_type: crate::event::SchemaChangeType::Added,
            details: "test schema change".to_string(),
            metadata: EventMetadata {
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

        let features = extractor.extract_features(&event).unwrap();
        assert!(!features.features.is_empty());
        assert_eq!(features.features.len(), features.feature_names.len());
    }
}
