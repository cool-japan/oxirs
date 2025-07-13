//! Performance Monitoring Module
//!
//! Provides comprehensive performance monitoring for SHACL validation operations,
//! including timing analysis, memory tracking, and throughput measurements.

use super::{ValidationEvent, ValidationEventType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// Performance monitoring system for validation operations
#[derive(Debug)]
pub struct PerformanceMonitor {
    events: VecDeque<PerformanceEvent>,
    current_sessions: HashMap<String, ValidationSession>,
    summaries: VecDeque<PerformanceSummary>,
    max_events: usize,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            current_sessions: HashMap::new(),
            summaries: VecDeque::new(),
            max_events: 10000,
        }
    }

    /// Create with custom capacity
    pub fn with_capacity(max_events: usize) -> Self {
        Self {
            events: VecDeque::new(),
            current_sessions: HashMap::new(),
            summaries: VecDeque::new(),
            max_events,
        }
    }

    /// Record a validation event
    pub fn record_event(&mut self, event: &ValidationEvent) {
        let perf_event = PerformanceEvent {
            timestamp: event.timestamp,
            event_type: event.event_type.clone(),
            duration: event.duration,
            memory_usage: event.memory_usage,
            shape_id: event.shape_id.clone(),
            constraint_id: event.constraint_id.clone(),
            target_count: event.target_count,
            violation_count: event.violation_count,
        };

        // Handle session tracking
        if let Some(shape_id) = &event.shape_id {
            match event.event_type {
                ValidationEventType::ValidationStarted => {
                    self.start_validation_session(shape_id.clone(), event.timestamp);
                }
                ValidationEventType::ValidationCompleted => {
                    if let Some(duration) = event.duration {
                        self.complete_validation_session(shape_id, duration, event.timestamp);
                    }
                }
                _ => {}
            }
        }

        self.events.push_back(perf_event);

        // Maintain size limit
        if self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Start tracking a validation session
    fn start_validation_session(&mut self, shape_id: String, timestamp: DateTime<Utc>) {
        let session = ValidationSession {
            shape_id: shape_id.clone(),
            start_time: timestamp,
            events: Vec::new(),
        };
        self.current_sessions.insert(shape_id, session);
    }

    /// Complete a validation session and generate summary
    fn complete_validation_session(
        &mut self,
        shape_id: &str,
        duration: Duration,
        timestamp: DateTime<Utc>,
    ) {
        if let Some(mut session) = self.current_sessions.remove(shape_id) {
            session.events = self
                .events
                .iter()
                .filter(|e| {
                    e.shape_id.as_ref().is_some_and(|id| id == shape_id)
                        && e.timestamp >= session.start_time
                        && e.timestamp <= timestamp
                })
                .cloned()
                .collect();

            // Generate summary for this session
            self.generate_session_summary(session, duration);
        }
    }

    /// Generate performance summary for a completed session
    fn generate_session_summary(&mut self, session: ValidationSession, total_duration: Duration) {
        let constraint_timings: HashMap<String, Duration> = session
            .events
            .iter()
            .filter_map(|e| {
                if let (Some(constraint_id), Some(duration)) = (&e.constraint_id, &e.duration) {
                    Some((constraint_id.clone(), *duration))
                } else {
                    None
                }
            })
            .collect();

        let total_violations: usize = session
            .events
            .iter()
            .filter_map(|e| e.violation_count)
            .sum();

        let peak_memory = session.events.iter().filter_map(|e| e.memory_usage).max();

        let cache_hits = session
            .events
            .iter()
            .filter(|e| e.event_type == ValidationEventType::CacheHit)
            .count();

        let cache_misses = session
            .events
            .iter()
            .filter(|e| e.event_type == ValidationEventType::CacheMiss)
            .count();

        let summary = PerformanceSummary {
            timestamp: session.start_time,
            shape_id: session.shape_id,
            total_duration,
            constraint_timings,
            total_violations,
            peak_memory_usage: peak_memory,
            cache_hit_ratio: if cache_hits + cache_misses > 0 {
                cache_hits as f64 / (cache_hits + cache_misses) as f64
            } else {
                0.0
            },
            throughput_items_per_second: session
                .events
                .iter()
                .filter_map(|e| e.target_count)
                .sum::<usize>() as f64
                / total_duration.as_secs_f64(),
            average_validation_time: total_duration,
        };

        self.summaries.push_back(summary);

        // Maintain size limit for summaries
        if self.summaries.len() > 1000 {
            self.summaries.pop_front();
        }
    }

    /// Get overall performance summary
    pub fn get_summary(&self) -> PerformanceSummary {
        if self.summaries.is_empty() {
            return PerformanceSummary::default();
        }

        let count = self.summaries.len();
        let total_duration: Duration = self.summaries.iter().map(|s| s.total_duration).sum();
        let average_duration = total_duration / count as u32;

        let total_violations: usize = self.summaries.iter().map(|s| s.total_violations).sum();
        let peak_memory = self
            .summaries
            .iter()
            .filter_map(|s| s.peak_memory_usage)
            .max();
        let average_cache_hit_ratio = self
            .summaries
            .iter()
            .map(|s| s.cache_hit_ratio)
            .sum::<f64>()
            / count as f64;
        let average_throughput = self
            .summaries
            .iter()
            .map(|s| s.throughput_items_per_second)
            .sum::<f64>()
            / count as f64;

        // Aggregate constraint timings
        let mut all_constraint_timings: HashMap<String, Vec<Duration>> = HashMap::new();
        for summary in &self.summaries {
            for (constraint_id, duration) in &summary.constraint_timings {
                all_constraint_timings
                    .entry(constraint_id.clone())
                    .or_default()
                    .push(*duration);
            }
        }

        let average_constraint_timings: HashMap<String, Duration> = all_constraint_timings
            .into_iter()
            .map(|(constraint_id, durations)| {
                let average = durations.iter().sum::<Duration>() / durations.len() as u32;
                (constraint_id, average)
            })
            .collect();

        PerformanceSummary {
            timestamp: Utc::now(),
            shape_id: "aggregate".to_string(),
            total_duration,
            constraint_timings: average_constraint_timings,
            total_violations,
            peak_memory_usage: peak_memory,
            cache_hit_ratio: average_cache_hit_ratio,
            throughput_items_per_second: average_throughput,
            average_validation_time: average_duration,
        }
    }

    /// Get recent performance events
    pub fn get_recent_events(&self, limit: usize) -> Vec<&PerformanceEvent> {
        self.events.iter().rev().take(limit).collect()
    }

    /// Get performance trend analysis
    pub fn get_performance_trend(&self, duration: Duration) -> PerformanceTrend {
        let cutoff = Utc::now() - chrono::Duration::from_std(duration).unwrap_or_default();
        let recent_summaries: Vec<&PerformanceSummary> = self
            .summaries
            .iter()
            .filter(|s| s.timestamp >= cutoff)
            .collect();

        if recent_summaries.is_empty() {
            return PerformanceTrend::default();
        }

        let validation_times: Vec<f64> = recent_summaries
            .iter()
            .map(|s| s.average_validation_time.as_secs_f64())
            .collect();

        let throughputs: Vec<f64> = recent_summaries
            .iter()
            .map(|s| s.throughput_items_per_second)
            .collect();

        let memory_usages: Vec<f64> = recent_summaries
            .iter()
            .filter_map(|s| s.peak_memory_usage.map(|m| m as f64))
            .collect();

        PerformanceTrend {
            validation_time_trend: calculate_trend(&validation_times),
            throughput_trend: calculate_trend(&throughputs),
            memory_usage_trend: calculate_trend(&memory_usages),
            sample_count: recent_summaries.len(),
            time_period: duration,
        }
    }

    /// Cleanup events before a certain timestamp
    pub fn cleanup_before(&mut self, cutoff: DateTime<Utc>) {
        self.events.retain(|e| e.timestamp >= cutoff);
        self.summaries.retain(|s| s.timestamp >= cutoff);
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual performance event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: ValidationEventType,
    pub duration: Option<Duration>,
    pub memory_usage: Option<usize>,
    pub shape_id: Option<String>,
    pub constraint_id: Option<String>,
    pub target_count: Option<usize>,
    pub violation_count: Option<usize>,
}

/// Validation session tracking
#[derive(Debug, Clone)]
struct ValidationSession {
    shape_id: String,
    start_time: DateTime<Utc>,
    events: Vec<PerformanceEvent>,
}

/// Performance summary for completed validations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub timestamp: DateTime<Utc>,
    pub shape_id: String,
    pub total_duration: Duration,
    pub constraint_timings: HashMap<String, Duration>,
    pub total_violations: usize,
    pub peak_memory_usage: Option<usize>,
    pub cache_hit_ratio: f64,
    pub throughput_items_per_second: f64,
    pub average_validation_time: Duration,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            shape_id: "unknown".to_string(),
            total_duration: Duration::ZERO,
            constraint_timings: HashMap::new(),
            total_violations: 0,
            peak_memory_usage: None,
            cache_hit_ratio: 0.0,
            throughput_items_per_second: 0.0,
            average_validation_time: Duration::ZERO,
        }
    }
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub validation_time_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub memory_usage_trend: TrendDirection,
    pub sample_count: usize,
    pub time_period: Duration,
}

impl Default for PerformanceTrend {
    fn default() -> Self {
        Self {
            validation_time_trend: TrendDirection::Stable,
            throughput_trend: TrendDirection::Stable,
            memory_usage_trend: TrendDirection::Stable,
            sample_count: 0,
            time_period: Duration::ZERO,
        }
    }
}

/// Direction of performance trends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    InsufficientData,
}

/// Calculate trend direction from a series of values
fn calculate_trend(values: &[f64]) -> TrendDirection {
    if values.len() < 3 {
        return TrendDirection::InsufficientData;
    }

    // Simple linear regression to determine trend
    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

    if slope.abs() <= 0.01 {
        TrendDirection::Stable
    } else if slope > 0.0 {
        TrendDirection::Degrading // For validation time and memory, increasing is degrading
    } else {
        TrendDirection::Improving
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert!(monitor.events.is_empty());
        assert!(monitor.current_sessions.is_empty());
    }

    #[test]
    fn test_trend_calculation() {
        // Improving trend (decreasing values)
        let improving = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert_eq!(calculate_trend(&improving), TrendDirection::Improving);

        // Degrading trend (increasing values)
        let degrading = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(calculate_trend(&degrading), TrendDirection::Degrading);

        // Stable trend
        let stable = vec![5.0, 5.1, 4.9, 5.0, 5.0];
        assert_eq!(calculate_trend(&stable), TrendDirection::Stable);

        // Insufficient data
        let insufficient = vec![1.0, 2.0];
        assert_eq!(
            calculate_trend(&insufficient),
            TrendDirection::InsufficientData
        );
    }
}
