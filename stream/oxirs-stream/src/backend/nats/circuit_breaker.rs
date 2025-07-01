//! # NATS Circuit Breaker
//!
//! Advanced circuit breaker implementation with adaptive thresholds,
//! machine learning-based failure prediction, and intelligent recovery.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing recovery
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_seconds: u64,
    pub half_open_max_calls: u32,
    pub enable_adaptive_thresholds: bool,
    pub enable_ml_prediction: bool,
    pub window_size_seconds: u64,
    pub slow_call_threshold_ms: u64,
}

/// Circuit breaker metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerMetrics {
    pub state: CircuitState,
    pub failure_count: u64,
    pub success_count: u64,
    pub total_calls: u64,
    pub last_failure_time: Option<DateTime<Utc>>,
    pub last_state_change: DateTime<Utc>,
    pub success_rate: f64,
    pub average_response_time_ms: f64,
    pub rejected_calls: u64,
    pub half_open_calls: u32,
}

/// Machine learning features for failure prediction
#[derive(Debug, Clone)]
struct MLFeatures {
    recent_failure_rate: f64,
    response_time_trend: f64,
    error_pattern_score: f64,
    time_of_day_factor: f64,
    load_factor: f64,
}

/// Advanced circuit breaker with ML capabilities
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: RwLock<CircuitState>,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    rejected_calls: AtomicU64,
    half_open_calls: RwLock<u32>,
    last_failure_time: RwLock<Option<DateTime<Utc>>>,
    last_state_change: RwLock<DateTime<Utc>>,
    response_times: RwLock<Vec<(DateTime<Utc>, u64)>>,
    failure_patterns: RwLock<Vec<DateTime<Utc>>>,
    ml_model: RwLock<Option<SimpleMLModel>>,
}

/// Simple ML model for failure prediction
#[derive(Debug, Clone)]
struct SimpleMLModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    update_count: u64,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let ml_model = if config.enable_ml_prediction {
            Some(SimpleMLModel::new())
        } else {
            None
        };

        Self {
            config,
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            rejected_calls: AtomicU64::new(0),
            half_open_calls: RwLock::new(0),
            last_failure_time: RwLock::new(None),
            last_state_change: RwLock::new(Utc::now()),
            response_times: RwLock::new(Vec::new()),
            failure_patterns: RwLock::new(Vec::new()),
            ml_model: RwLock::new(ml_model),
        }
    }

    /// Check if call should be allowed
    pub async fn allow_call(&self) -> bool {
        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitState::Closed => {
                // Check if we should predict failure using ML
                if self.config.enable_ml_prediction {
                    if let Some(failure_probability) = self.predict_failure_probability().await {
                        if failure_probability > 0.8 {
                            warn!(
                                "ML model predicts high failure probability: {:.2}",
                                failure_probability
                            );
                            // Still allow call but monitor closely
                        }
                    }
                }
                true
            }
            CircuitState::Open => {
                // Check if timeout has elapsed
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    let timeout_duration = Duration::seconds(self.config.timeout_seconds as i64);
                    if Utc::now() - last_failure > timeout_duration {
                        self.transition_to_half_open().await;
                        true
                    } else {
                        self.rejected_calls.fetch_add(1, Ordering::Relaxed);
                        false
                    }
                } else {
                    // No last failure time, allow transition
                    self.transition_to_half_open().await;
                    true
                }
            }
            CircuitState::HalfOpen => {
                let mut half_open_calls = self.half_open_calls.write().await;
                if *half_open_calls < self.config.half_open_max_calls {
                    *half_open_calls += 1;
                    true
                } else {
                    self.rejected_calls.fetch_add(1, Ordering::Relaxed);
                    false
                }
            }
        }
    }

    /// Record successful call
    pub async fn record_success(&self, response_time_ms: u64) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        // Record response time
        let mut response_times = self.response_times.write().await;
        response_times.push((Utc::now(), response_time_ms));

        // Maintain window size
        let window_duration = Duration::seconds(self.config.window_size_seconds as i64);
        let cutoff_time = Utc::now() - window_duration;
        response_times.retain(|(time, _)| *time > cutoff_time);

        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitState::HalfOpen => {
                let successes = self.success_count.load(Ordering::Relaxed);
                if successes >= self.config.success_threshold as u64 {
                    self.transition_to_closed().await;
                }
            }
            CircuitState::Open => {
                // Should not happen, but handle gracefully
                warn!("Recorded success while circuit is open");
            }
            CircuitState::Closed => {
                // Normal operation, update ML model if enabled
                if self.config.enable_ml_prediction {
                    self.update_ml_model(false, response_time_ms).await;
                }
            }
        }

        debug!("Recorded success: {}ms response time", response_time_ms);
    }

    /// Record failed call
    pub async fn record_failure(&self, error_type: &str) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);

        let now = Utc::now();
        *self.last_failure_time.write().await = Some(now);

        // Record failure pattern
        let mut failure_patterns = self.failure_patterns.write().await;
        failure_patterns.push(now);

        // Maintain window size
        let window_duration = Duration::seconds(self.config.window_size_seconds as i64);
        let cutoff_time = now - window_duration;
        failure_patterns.retain(|time| *time > cutoff_time);

        // Update ML model if enabled
        if self.config.enable_ml_prediction {
            self.update_ml_model(true, 0).await;
        }

        let current_state = self.state.read().await.clone();

        match current_state {
            CircuitState::Closed => {
                let threshold = if self.config.enable_adaptive_thresholds {
                    self.calculate_adaptive_threshold().await
                } else {
                    self.config.failure_threshold
                };

                let failures = self.failure_count.load(Ordering::Relaxed);
                if failures >= threshold as u64 {
                    self.transition_to_open().await;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state opens the circuit
                self.transition_to_open().await;
            }
            CircuitState::Open => {
                // Already open, update failure time
            }
        }

        warn!(
            "Recorded failure: {} (state: {:?})",
            error_type, current_state
        );
    }

    /// Calculate adaptive threshold based on historical data
    async fn calculate_adaptive_threshold(&self) -> u32 {
        let failure_patterns = self.failure_patterns.read().await;
        let response_times = self.response_times.read().await;

        // Simple adaptive algorithm based on recent activity
        let base_threshold = self.config.failure_threshold;
        let recent_failures = failure_patterns.len();
        let recent_calls = response_times.len();

        if recent_calls == 0 {
            return base_threshold;
        }

        let failure_rate = recent_failures as f64 / recent_calls as f64;

        // Adjust threshold based on failure rate trends
        if failure_rate > 0.1 {
            // High failure rate, lower threshold
            std::cmp::max(1, base_threshold / 2)
        } else if failure_rate < 0.01 {
            // Low failure rate, higher threshold
            base_threshold * 2
        } else {
            base_threshold
        }
    }

    /// Predict failure probability using ML model
    async fn predict_failure_probability(&self) -> Option<f64> {
        let ml_model_guard = self.ml_model.read().await;
        if let Some(ref model) = *ml_model_guard {
            let features = self.extract_ml_features().await;
            Some(model.predict(&features))
        } else {
            None
        }
    }

    /// Extract ML features from current state
    async fn extract_ml_features(&self) -> MLFeatures {
        let failure_patterns = self.failure_patterns.read().await;
        let response_times = self.response_times.read().await;

        let now = Utc::now();
        let recent_window = Duration::minutes(5);
        let cutoff_time = now - recent_window;

        // Calculate recent failure rate
        let recent_failures = failure_patterns
            .iter()
            .filter(|&&t| t > cutoff_time)
            .count();
        let recent_calls = response_times
            .iter()
            .filter(|(t, _)| *t > cutoff_time)
            .count();
        let recent_failure_rate = if recent_calls > 0 {
            recent_failures as f64 / recent_calls as f64
        } else {
            0.0
        };

        // Calculate response time trend
        let recent_response_times: Vec<u64> = response_times
            .iter()
            .filter(|(t, _)| *t > cutoff_time)
            .map(|(_, rt)| *rt)
            .collect();

        let response_time_trend = if recent_response_times.len() >= 2 {
            let mid = recent_response_times.len() / 2;
            let first_half_avg: f64 = recent_response_times[..mid]
                .iter()
                .map(|&x| x as f64)
                .sum::<f64>()
                / mid as f64;
            let second_half_avg: f64 = recent_response_times[mid..]
                .iter()
                .map(|&x| x as f64)
                .sum::<f64>()
                / (recent_response_times.len() - mid) as f64;
            second_half_avg - first_half_avg
        } else {
            0.0
        };

        // Calculate error pattern score (simplified)
        let error_pattern_score = if failure_patterns.len() >= 3 {
            // Check for increasing frequency of failures
            let intervals: Vec<i64> = failure_patterns
                .windows(2)
                .map(|w| (w[1] - w[0]).num_seconds())
                .collect();

            if intervals.len() >= 2 {
                let avg_interval = intervals.iter().sum::<i64>() as f64 / intervals.len() as f64;
                1.0 / (1.0 + avg_interval / 60.0) // Normalize to 0-1 range
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Time of day factor (simplified)
        let hour = now.hour() as f64;
        let time_of_day_factor = if hour >= 9.0 && hour <= 17.0 {
            1.0
        } else {
            0.5
        };

        // Load factor based on recent call volume
        let load_factor = std::cmp::min(recent_calls, 100) as f64 / 100.0;

        MLFeatures {
            recent_failure_rate,
            response_time_trend,
            error_pattern_score,
            time_of_day_factor,
            load_factor,
        }
    }

    /// Update ML model with new observation
    async fn update_ml_model(&self, is_failure: bool, response_time_ms: u64) {
        let mut ml_model_guard = self.ml_model.write().await;
        if let Some(ref mut model) = *ml_model_guard {
            let features = self.extract_ml_features().await;
            let label = if is_failure { 1.0 } else { 0.0 };
            model.update(&features, label);
        }
    }

    /// Transition to closed state
    async fn transition_to_closed(&self) {
        *self.state.write().await = CircuitState::Closed;
        *self.last_state_change.write().await = Utc::now();
        *self.half_open_calls.write().await = 0;
        self.failure_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker transitioned to CLOSED");
    }

    /// Transition to open state
    async fn transition_to_open(&self) {
        *self.state.write().await = CircuitState::Open;
        *self.last_state_change.write().await = Utc::now();
        *self.half_open_calls.write().await = 0;
        warn!("Circuit breaker transitioned to OPEN");
    }

    /// Transition to half-open state
    async fn transition_to_half_open(&self) {
        *self.state.write().await = CircuitState::HalfOpen;
        *self.last_state_change.write().await = Utc::now();
        *self.half_open_calls.write().await = 0;
        self.success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker transitioned to HALF-OPEN");
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> CircuitBreakerMetrics {
        let state = self.state.read().await.clone();
        let failure_count = self.failure_count.load(Ordering::Relaxed);
        let success_count = self.success_count.load(Ordering::Relaxed);
        let total_calls = failure_count + success_count;
        let success_rate = if total_calls > 0 {
            success_count as f64 / total_calls as f64
        } else {
            0.0
        };

        let response_times = self.response_times.read().await;
        let average_response_time_ms = if !response_times.is_empty() {
            response_times.iter().map(|(_, rt)| *rt as f64).sum::<f64>()
                / response_times.len() as f64
        } else {
            0.0
        };

        CircuitBreakerMetrics {
            state,
            failure_count,
            success_count,
            total_calls,
            last_failure_time: *self.last_failure_time.read().await,
            last_state_change: *self.last_state_change.read().await,
            success_rate,
            average_response_time_ms,
            rejected_calls: self.rejected_calls.load(Ordering::Relaxed),
            half_open_calls: *self.half_open_calls.read().await,
        }
    }

    /// Reset circuit breaker state
    pub async fn reset(&self) {
        *self.state.write().await = CircuitState::Closed;
        *self.last_state_change.write().await = Utc::now();
        *self.half_open_calls.write().await = 0;
        *self.last_failure_time.write().await = None;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.rejected_calls.store(0, Ordering::Relaxed);
        self.response_times.write().await.clear();
        self.failure_patterns.write().await.clear();

        // Reset ML model
        if self.config.enable_ml_prediction {
            *self.ml_model.write().await = Some(SimpleMLModel::new());
        }

        info!("Circuit breaker reset to initial state");
    }
}

impl SimpleMLModel {
    fn new() -> Self {
        Self {
            weights: vec![0.0; 5], // Number of features
            bias: 0.0,
            learning_rate: 0.01,
            update_count: 0,
        }
    }

    fn predict(&self, features: &MLFeatures) -> f64 {
        let feature_vector = vec![
            features.recent_failure_rate,
            features.response_time_trend / 1000.0, // Normalize
            features.error_pattern_score,
            features.time_of_day_factor,
            features.load_factor,
        ];

        let weighted_sum: f64 = self
            .weights
            .iter()
            .zip(feature_vector.iter())
            .map(|(w, f)| w * f)
            .sum::<f64>()
            + self.bias;

        // Sigmoid activation
        1.0 / (1.0 + (-weighted_sum).exp())
    }

    fn update(&mut self, features: &MLFeatures, label: f64) {
        let feature_vector = vec![
            features.recent_failure_rate,
            features.response_time_trend / 1000.0,
            features.error_pattern_score,
            features.time_of_day_factor,
            features.load_factor,
        ];

        let prediction = self.predict(features);
        let error = label - prediction;

        // Update weights using gradient descent
        for (i, &feature) in feature_vector.iter().enumerate() {
            self.weights[i] += self.learning_rate * error * feature;
        }
        self.bias += self.learning_rate * error;

        self.update_count += 1;

        // Decay learning rate over time
        if self.update_count % 100 == 0 {
            self.learning_rate *= 0.95;
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_seconds: 30,
            half_open_max_calls: 3,
            enable_adaptive_thresholds: true,
            enable_ml_prediction: true,
            window_size_seconds: 300, // 5 minutes
            slow_call_threshold_ms: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_circuit_breaker_creation() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        assert!(cb.allow_call().await);

        let metrics = cb.get_metrics().await;
        assert_eq!(metrics.state, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            enable_adaptive_thresholds: false,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // First failure
        cb.record_failure("test_error").await;
        assert!(cb.allow_call().await);

        // Second failure should open circuit
        cb.record_failure("test_error").await;
        let metrics = cb.get_metrics().await;
        assert_eq!(metrics.state, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_success_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_seconds: 1,
            enable_adaptive_thresholds: false,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Trigger failure
        cb.record_failure("test_error").await;
        assert_eq!(cb.get_metrics().await.state, CircuitState::Open);

        // Wait for timeout
        sleep(TokioDuration::from_secs(2)).await;

        // Should transition to half-open
        assert!(cb.allow_call().await);
        assert_eq!(cb.get_metrics().await.state, CircuitState::HalfOpen);

        // Record successes
        cb.record_success(50).await;
        cb.record_success(60).await;

        // Should be closed now
        assert_eq!(cb.get_metrics().await.state, CircuitState::Closed);
    }
}
