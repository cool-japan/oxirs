//! Real-time Metrics Module
//!
//! Provides real-time monitoring and alerting for SHACL validation operations.

use super::{ValidationEvent, ValidationEventType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Real-time metrics collector and monitor
#[derive(Debug)]
pub struct RealTimeMetrics {
    // Atomic counters for thread-safe real-time updates
    total_validations: AtomicUsize,
    total_violations: AtomicUsize,
    active_validations: AtomicUsize,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    memory_usage: AtomicU64,

    // Recent events for sliding window calculations
    recent_events: VecDeque<TimestampedMetric>,
    window_size: Duration,

    // Alert thresholds and configurations
    alert_config: AlertConfiguration,
    active_alerts: Vec<ActiveAlert>,

    // Performance snapshots
    snapshots: VecDeque<MetricsSnapshot>,
    max_snapshots: usize,
}

impl RealTimeMetrics {
    /// Create new real-time metrics collector
    pub fn new() -> Self {
        Self {
            total_validations: AtomicUsize::new(0),
            total_violations: AtomicUsize::new(0),
            active_validations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            memory_usage: AtomicU64::new(0),
            recent_events: VecDeque::new(),
            window_size: Duration::from_secs(300), // 5 minutes
            alert_config: AlertConfiguration::default(),
            active_alerts: Vec::new(),
            snapshots: VecDeque::new(),
            max_snapshots: 1440, // 24 hours at 1-minute intervals
        }
    }

    /// Create with custom configuration
    pub fn with_config(alert_config: AlertConfiguration, window_size: Duration) -> Self {
        Self {
            total_validations: AtomicUsize::new(0),
            total_violations: AtomicUsize::new(0),
            active_validations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            memory_usage: AtomicU64::new(0),
            recent_events: VecDeque::new(),
            window_size,
            alert_config,
            active_alerts: Vec::new(),
            snapshots: VecDeque::new(),
            max_snapshots: 1440,
        }
    }

    /// Update metrics with a validation event
    pub fn update(&mut self, event: &ValidationEvent) {
        let now = Instant::now();

        // Update atomic counters
        match event.event_type {
            ValidationEventType::ValidationStarted => {
                self.total_validations.fetch_add(1, Ordering::Relaxed);
                self.active_validations.fetch_add(1, Ordering::Relaxed);
            }
            ValidationEventType::ValidationCompleted => {
                self.active_validations.fetch_sub(1, Ordering::Relaxed);
            }
            ValidationEventType::ViolationDetected => {
                self.total_violations.fetch_add(1, Ordering::Relaxed);
            }
            ValidationEventType::CacheHit => {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            ValidationEventType::CacheMiss => {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        // Update memory usage if provided
        if let Some(memory) = event.memory_usage {
            self.memory_usage.store(memory as u64, Ordering::Relaxed);
        }

        // Add to recent events for windowed calculations
        let metric = TimestampedMetric {
            timestamp: now,
            event_type: event.event_type.clone(),
            duration: event.duration,
            violation_count: event.violation_count.unwrap_or(0),
            target_count: event.target_count.unwrap_or(0),
        };

        self.recent_events.push_back(metric);

        // Clean old events outside the window
        let cutoff = now - self.window_size;
        while let Some(front) = self.recent_events.front() {
            if front.timestamp < cutoff {
                self.recent_events.pop_front();
            } else {
                break;
            }
        }

        // Check for alerts
        self.check_alerts();
    }

    /// Get current real-time metrics snapshot
    pub fn get_current_metrics(&self) -> RealTimeMetricsSnapshot {
        let now = Instant::now();
        let cutoff = now - self.window_size;

        // Calculate windowed metrics
        let recent_events: Vec<_> = self
            .recent_events
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .collect();

        let validations_per_minute = if !recent_events.is_empty() {
            let completed_validations = recent_events
                .iter()
                .filter(|e| e.event_type == ValidationEventType::ValidationCompleted)
                .count();
            (completed_validations as f64) * (60.0 / self.window_size.as_secs_f64())
        } else {
            0.0
        };

        let violations_per_minute = if !recent_events.is_empty() {
            let total_violations: usize = recent_events.iter().map(|e| e.violation_count).sum();
            (total_violations as f64) * (60.0 / self.window_size.as_secs_f64())
        } else {
            0.0
        };

        let average_validation_time = if !recent_events.is_empty() {
            let durations: Vec<Duration> =
                recent_events.iter().filter_map(|e| e.duration).collect();
            if !durations.is_empty() {
                durations.iter().sum::<Duration>() / durations.len() as u32
            } else {
                Duration::ZERO
            }
        } else {
            Duration::ZERO
        };

        let cache_hit_rate = {
            let hits = self.cache_hits.load(Ordering::Relaxed);
            let misses = self.cache_misses.load(Ordering::Relaxed);
            if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            }
        };

        RealTimeMetricsSnapshot {
            timestamp: Utc::now(),
            total_validations: self.total_validations.load(Ordering::Relaxed),
            total_violations: self.total_violations.load(Ordering::Relaxed),
            active_validations: self.active_validations.load(Ordering::Relaxed),
            validations_per_minute,
            violations_per_minute,
            average_validation_time,
            cache_hit_rate,
            memory_usage_bytes: self.memory_usage.load(Ordering::Relaxed),
            active_alerts: self.active_alerts.clone(),
        }
    }

    /// Take a metrics snapshot and store it
    pub fn take_snapshot(&mut self) {
        let snapshot = MetricsSnapshot {
            timestamp: Utc::now(),
            metrics: self.get_current_metrics(),
        };

        self.snapshots.push_back(snapshot);

        // Maintain size limit
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.pop_front();
        }
    }

    /// Get historical snapshots
    pub fn get_snapshots(&self, duration: Duration) -> Vec<&MetricsSnapshot> {
        let cutoff = Utc::now() - chrono::Duration::from_std(duration).unwrap_or_default();
        self.snapshots
            .iter()
            .filter(|s| s.timestamp >= cutoff)
            .collect()
    }

    /// Check for alert conditions
    fn check_alerts(&mut self) {
        let current_metrics = self.get_current_metrics();
        let now = Utc::now();

        // Clear expired alerts
        self.active_alerts.retain(|alert| {
            now.signed_duration_since(alert.triggered_at)
                .to_std()
                .unwrap_or_default()
                < self.alert_config.alert_duration
        });

        // Check high violation rate
        if current_metrics.violations_per_minute > self.alert_config.max_violations_per_minute {
            let alert_id = format!("high_violation_rate_{}", now.timestamp());
            if !self
                .active_alerts
                .iter()
                .any(|a| a.alert_type == AlertType::HighViolationRate)
            {
                self.active_alerts.push(ActiveAlert {
                    id: alert_id,
                    alert_type: AlertType::HighViolationRate,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "High violation rate detected: {:.2} violations/minute (threshold: {:.2})",
                        current_metrics.violations_per_minute,
                        self.alert_config.max_violations_per_minute
                    ),
                    triggered_at: now,
                    additional_data: HashMap::new(),
                });
            }
        }

        // Check high memory usage
        if current_metrics.memory_usage_bytes > self.alert_config.max_memory_usage_bytes {
            let alert_id = format!("high_memory_usage_{}", now.timestamp());
            if !self
                .active_alerts
                .iter()
                .any(|a| a.alert_type == AlertType::HighMemoryUsage)
            {
                self.active_alerts.push(ActiveAlert {
                    id: alert_id,
                    alert_type: AlertType::HighMemoryUsage,
                    severity: AlertSeverity::Critical,
                    message: format!(
                        "High memory usage detected: {:.2} MB (threshold: {:.2} MB)",
                        current_metrics.memory_usage_bytes as f64 / (1024.0 * 1024.0),
                        self.alert_config.max_memory_usage_bytes as f64 / (1024.0 * 1024.0)
                    ),
                    triggered_at: now,
                    additional_data: HashMap::new(),
                });
            }
        }

        // Check slow validation time
        if current_metrics.average_validation_time > self.alert_config.max_validation_time {
            let alert_id = format!("slow_validation_{}", now.timestamp());
            if !self
                .active_alerts
                .iter()
                .any(|a| a.alert_type == AlertType::SlowValidation)
            {
                self.active_alerts.push(ActiveAlert {
                    id: alert_id,
                    alert_type: AlertType::SlowValidation,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Slow validation detected: {:.2}s average (threshold: {:.2}s)",
                        current_metrics.average_validation_time.as_secs_f64(),
                        self.alert_config.max_validation_time.as_secs_f64()
                    ),
                    triggered_at: now,
                    additional_data: HashMap::new(),
                });
            }
        }

        // Check low cache hit rate
        if current_metrics.cache_hit_rate < self.alert_config.min_cache_hit_rate {
            let alert_id = format!("low_cache_hit_rate_{}", now.timestamp());
            if !self
                .active_alerts
                .iter()
                .any(|a| a.alert_type == AlertType::LowCacheHitRate)
            {
                self.active_alerts.push(ActiveAlert {
                    id: alert_id,
                    alert_type: AlertType::LowCacheHitRate,
                    severity: AlertSeverity::Info,
                    message: format!(
                        "Low cache hit rate detected: {:.2}% (threshold: {:.2}%)",
                        current_metrics.cache_hit_rate * 100.0,
                        self.alert_config.min_cache_hit_rate * 100.0
                    ),
                    triggered_at: now,
                    additional_data: HashMap::new(),
                });
            }
        }
    }

    /// Cleanup old data
    pub fn cleanup_before(&mut self, cutoff: DateTime<Utc>) {
        let cutoff_instant = Instant::now()
            - Duration::from_secs(Utc::now().signed_duration_since(cutoff).num_seconds() as u64);

        self.recent_events.retain(|e| e.timestamp >= cutoff_instant);
        self.snapshots.retain(|s| s.timestamp >= cutoff);
    }
}

impl Default for RealTimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Timestamped metric for windowed calculations
#[derive(Debug, Clone)]
struct TimestampedMetric {
    timestamp: Instant,
    event_type: ValidationEventType,
    duration: Option<Duration>,
    violation_count: usize,
    #[allow(dead_code)]
    target_count: usize,
}

/// Configuration for real-time alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfiguration {
    pub max_violations_per_minute: f64,
    pub max_memory_usage_bytes: u64,
    pub max_validation_time: Duration,
    pub min_cache_hit_rate: f64,
    pub alert_duration: Duration,
}

impl Default for AlertConfiguration {
    fn default() -> Self {
        Self {
            max_violations_per_minute: 100.0,
            max_memory_usage_bytes: 500 * 1024 * 1024, // 500MB
            max_validation_time: Duration::from_secs(5),
            min_cache_hit_rate: 0.8,                  // 80%
            alert_duration: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Real-time metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_validations: usize,
    pub total_violations: usize,
    pub active_validations: usize,
    pub validations_per_minute: f64,
    pub violations_per_minute: f64,
    pub average_validation_time: Duration,
    pub cache_hit_rate: f64,
    pub memory_usage_bytes: u64,
    pub active_alerts: Vec<ActiveAlert>,
}

/// Historical metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub metrics: RealTimeMetricsSnapshot,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: DateTime<Utc>,
    pub additional_data: HashMap<String, String>,
}

/// Types of alerts
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    HighViolationRate,
    HighMemoryUsage,
    SlowValidation,
    LowCacheHitRate,
    SystemError,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_time_metrics_creation() {
        let metrics = RealTimeMetrics::new();
        assert_eq!(metrics.total_validations.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.total_violations.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.active_validations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_alert_configuration_default() {
        let config = AlertConfiguration::default();
        assert_eq!(config.max_violations_per_minute, 100.0);
        assert_eq!(config.max_memory_usage_bytes, 500 * 1024 * 1024);
        assert_eq!(config.min_cache_hit_rate, 0.8);
    }

    #[test]
    fn test_metrics_update() {
        let mut metrics = RealTimeMetrics::new();

        let event = ValidationEvent {
            timestamp: Utc::now(),
            event_type: ValidationEventType::ValidationStarted,
            duration: None,
            shape_id: Some("test_shape".to_string()),
            constraint_id: None,
            target_count: None,
            violation_count: None,
            memory_usage: None,
            additional_metadata: HashMap::new(),
        };

        metrics.update(&event);

        assert_eq!(metrics.total_validations.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.active_validations.load(Ordering::Relaxed), 1);
    }
}
