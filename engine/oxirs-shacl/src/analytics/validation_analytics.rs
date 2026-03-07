//! Validation Analytics Module
//!
//! Provides comprehensive analytics for validation results, patterns, and trends.

use super::{ValidationEvent, ValidationEventType};
use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

/// Analytics for validation results and patterns
#[derive(Debug)]
pub struct ValidationAnalytics {
    events: VecDeque<ValidationEvent>,
    shape_statistics: HashMap<String, ShapeStatistics>,
    constraint_statistics: HashMap<String, ConstraintStatistics>,
    temporal_data: VecDeque<TemporalValidationData>,
    max_events: usize,
}

impl ValidationAnalytics {
    /// Create new validation analytics
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            shape_statistics: HashMap::new(),
            constraint_statistics: HashMap::new(),
            temporal_data: VecDeque::new(),
            max_events: 50000,
        }
    }

    /// Record a validation event
    pub fn record_event(&mut self, event: ValidationEvent) {
        // Update shape statistics
        if let Some(shape_id) = &event.shape_id {
            self.update_shape_statistics(shape_id, &event);
        }

        // Update constraint statistics
        if let Some(constraint_id) = &event.constraint_id {
            self.update_constraint_statistics(constraint_id, &event);
        }

        // Update temporal data
        self.update_temporal_data(&event);

        self.events.push_back(event);

        // Maintain size limit
        if self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Update shape-specific statistics
    fn update_shape_statistics(&mut self, shape_id: &str, event: &ValidationEvent) {
        let stats = self
            .shape_statistics
            .entry(shape_id.to_string())
            .or_insert_with(|| ShapeStatistics::new(shape_id.to_string()));

        match event.event_type {
            ValidationEventType::ValidationStarted => {
                stats.validation_count += 1;
                stats.last_validation = Some(event.timestamp);
            }
            ValidationEventType::ValidationCompleted => {
                if let Some(duration) = event.duration {
                    stats.total_validation_time += duration;
                    stats.update_avg_validation_time();
                }
                if let Some(violations) = event.violation_count {
                    stats.total_violations += violations;
                    stats.update_avg_violations();
                }
                if let Some(targets) = event.target_count {
                    stats.total_targets_validated += targets;
                }
            }
            ValidationEventType::ViolationDetected => {
                stats.violation_events += 1;
            }
            _ => {}
        }
    }

    /// Update constraint-specific statistics
    fn update_constraint_statistics(&mut self, constraint_id: &str, event: &ValidationEvent) {
        let stats = self
            .constraint_statistics
            .entry(constraint_id.to_string())
            .or_insert_with(|| ConstraintStatistics::new(constraint_id.to_string()));

        match event.event_type {
            ValidationEventType::ConstraintEvaluated => {
                stats.evaluation_count += 1;
                if let Some(duration) = event.duration {
                    stats.total_evaluation_time += duration;
                    stats.update_avg_evaluation_time();
                }
            }
            ValidationEventType::ViolationDetected => {
                stats.violation_count += 1;
                stats.update_violation_rate();
            }
            ValidationEventType::CacheHit => {
                stats.cache_hits += 1;
            }
            ValidationEventType::CacheMiss => {
                stats.cache_misses += 1;
            }
            _ => {}
        }
    }

    /// Update temporal validation data
    fn update_temporal_data(&mut self, event: &ValidationEvent) {
        // Group events by hour for temporal analysis
        let hour_timestamp = event
            .timestamp
            .with_minute(0)
            .expect("operation should succeed")
            .with_second(0)
            .expect("operation should succeed")
            .with_nanosecond(0)
            .expect("operation should succeed");

        // Find or create temporal data for this hour
        if let Some(data) = self
            .temporal_data
            .iter_mut()
            .find(|d| d.timestamp == hour_timestamp)
        {
            data.update_with_event(event);
        } else {
            let mut new_data = TemporalValidationData::new(hour_timestamp);
            new_data.update_with_event(event);
            self.temporal_data.push_back(new_data);

            // Keep only last 24 hours of data
            if self.temporal_data.len() > 24 {
                self.temporal_data.pop_front();
            }
        }
    }

    /// Get comprehensive validation summary
    pub fn get_summary(&self) -> ValidationSummary {
        let total_validations: usize = self
            .shape_statistics
            .values()
            .map(|s| s.validation_count)
            .sum();
        let total_violations: usize = self
            .shape_statistics
            .values()
            .map(|s| s.total_violations)
            .sum();
        let total_targets: usize = self
            .shape_statistics
            .values()
            .map(|s| s.total_targets_validated)
            .sum();

        let overall_success_rate = if total_targets > 0 {
            1.0 - (total_violations as f64 / total_targets as f64)
        } else {
            1.0
        };

        let most_violated_shapes = self.get_most_violated_shapes(5);
        let slowest_constraints = self.get_slowest_constraints(5);
        let validation_trends = self.get_validation_trends();

        ValidationSummary {
            timestamp: Utc::now(),
            total_validations,
            total_violations,
            total_targets_validated: total_targets,
            overall_success_rate,
            most_violated_shapes,
            slowest_constraints,
            validation_trends,
            shape_count: self.shape_statistics.len(),
            constraint_count: self.constraint_statistics.len(),
        }
    }

    /// Get shapes with the most violations
    fn get_most_violated_shapes(&self, limit: usize) -> Vec<ShapeViolationSummary> {
        let mut shapes: Vec<_> = self
            .shape_statistics
            .values()
            .map(|stats| ShapeViolationSummary {
                shape_id: stats.shape_id.clone(),
                total_violations: stats.total_violations,
                violation_rate: if stats.total_targets_validated > 0 {
                    stats.total_violations as f64 / stats.total_targets_validated as f64
                } else {
                    0.0
                },
                validation_count: stats.validation_count,
            })
            .collect();

        shapes.sort_by(|a, b| b.total_violations.cmp(&a.total_violations));
        shapes.truncate(limit);
        shapes
    }

    /// Get constraints with the slowest average evaluation time
    fn get_slowest_constraints(&self, limit: usize) -> Vec<ConstraintPerformanceSummary> {
        let mut constraints: Vec<_> = self
            .constraint_statistics
            .values()
            .map(|stats| ConstraintPerformanceSummary {
                constraint_id: stats.constraint_id.clone(),
                average_evaluation_time: stats.average_evaluation_time,
                total_evaluations: stats.evaluation_count,
                violation_rate: stats.violation_rate,
                cache_hit_rate: if stats.cache_hits + stats.cache_misses > 0 {
                    stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64
                } else {
                    0.0
                },
            })
            .collect();

        constraints.sort_by(|a, b| b.average_evaluation_time.cmp(&a.average_evaluation_time));
        constraints.truncate(limit);
        constraints
    }

    /// Get validation trends over time
    fn get_validation_trends(&self) -> ValidationTrends {
        if self.temporal_data.len() < 2 {
            return ValidationTrends::default();
        }

        let recent_hours: Vec<_> = self.temporal_data.iter().rev().take(6).collect();
        let earlier_hours: Vec<_> = if self.temporal_data.len() > 6 {
            self.temporal_data.iter().rev().skip(6).take(6).collect()
        } else {
            Vec::new()
        };

        let recent_avg_violations = if !recent_hours.is_empty() {
            recent_hours
                .iter()
                .map(|d| d.violation_count)
                .sum::<usize>() as f64
                / recent_hours.len() as f64
        } else {
            0.0
        };

        let earlier_avg_violations = if !earlier_hours.is_empty() {
            earlier_hours
                .iter()
                .map(|d| d.violation_count)
                .sum::<usize>() as f64
                / earlier_hours.len() as f64
        } else {
            recent_avg_violations
        };

        let violation_trend = if (recent_avg_violations - earlier_avg_violations).abs() < 0.1 {
            TrendDirection::Stable
        } else if recent_avg_violations > earlier_avg_violations {
            TrendDirection::Increasing
        } else {
            TrendDirection::Decreasing
        };

        let recent_avg_validations = if !recent_hours.is_empty() {
            recent_hours
                .iter()
                .map(|d| d.validation_count)
                .sum::<usize>() as f64
                / recent_hours.len() as f64
        } else {
            0.0
        };

        let earlier_avg_validations = if !earlier_hours.is_empty() {
            earlier_hours
                .iter()
                .map(|d| d.validation_count)
                .sum::<usize>() as f64
                / earlier_hours.len() as f64
        } else {
            recent_avg_validations
        };

        let validation_volume_trend =
            if (recent_avg_validations - earlier_avg_validations).abs() < 0.1 {
                TrendDirection::Stable
            } else if recent_avg_validations > earlier_avg_validations {
                TrendDirection::Increasing
            } else {
                TrendDirection::Decreasing
            };

        ValidationTrends {
            violation_trend,
            validation_volume_trend,
            data_points: self.temporal_data.len(),
        }
    }

    /// Cleanup old events
    pub fn cleanup_before(&mut self, cutoff: DateTime<Utc>) {
        self.events.retain(|e| e.timestamp >= cutoff);

        let cutoff_chrono = chrono::Duration::from_std(
            cutoff
                .signed_duration_since(Utc::now())
                .to_std()
                .unwrap_or_default(),
        )
        .unwrap_or_default();

        self.temporal_data
            .retain(|d| d.timestamp >= cutoff - cutoff_chrono);
    }
}

impl Default for ValidationAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for individual shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeStatistics {
    pub shape_id: String,
    pub validation_count: usize,
    pub total_violations: usize,
    pub total_targets_validated: usize,
    pub total_validation_time: Duration,
    pub average_validation_time: Duration,
    pub average_violations: f64,
    pub violation_events: usize,
    pub last_validation: Option<DateTime<Utc>>,
}

impl ShapeStatistics {
    fn new(shape_id: String) -> Self {
        Self {
            shape_id,
            validation_count: 0,
            total_violations: 0,
            total_targets_validated: 0,
            total_validation_time: Duration::ZERO,
            average_validation_time: Duration::ZERO,
            average_violations: 0.0,
            violation_events: 0,
            last_validation: None,
        }
    }

    fn update_avg_validation_time(&mut self) {
        if self.validation_count > 0 {
            self.average_validation_time =
                self.total_validation_time / self.validation_count as u32;
        }
    }

    fn update_avg_violations(&mut self) {
        if self.validation_count > 0 {
            self.average_violations = self.total_violations as f64 / self.validation_count as f64;
        }
    }
}

/// Statistics for individual constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintStatistics {
    pub constraint_id: String,
    pub evaluation_count: usize,
    pub violation_count: usize,
    pub total_evaluation_time: Duration,
    pub average_evaluation_time: Duration,
    pub violation_rate: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl ConstraintStatistics {
    fn new(constraint_id: String) -> Self {
        Self {
            constraint_id,
            evaluation_count: 0,
            violation_count: 0,
            total_evaluation_time: Duration::ZERO,
            average_evaluation_time: Duration::ZERO,
            violation_rate: 0.0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    fn update_avg_evaluation_time(&mut self) {
        if self.evaluation_count > 0 {
            self.average_evaluation_time =
                self.total_evaluation_time / self.evaluation_count as u32;
        }
    }

    fn update_violation_rate(&mut self) {
        if self.evaluation_count > 0 {
            self.violation_rate = self.violation_count as f64 / self.evaluation_count as f64;
        }
    }
}

/// Temporal validation data for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalValidationData {
    pub timestamp: DateTime<Utc>,
    pub validation_count: usize,
    pub violation_count: usize,
    pub target_count: usize,
    pub average_validation_time: Duration,
}

impl TemporalValidationData {
    fn new(timestamp: DateTime<Utc>) -> Self {
        Self {
            timestamp,
            validation_count: 0,
            violation_count: 0,
            target_count: 0,
            average_validation_time: Duration::ZERO,
        }
    }

    fn update_with_event(&mut self, event: &ValidationEvent) {
        if event.event_type == ValidationEventType::ValidationCompleted {
            self.validation_count += 1;
            if let Some(violations) = event.violation_count {
                self.violation_count += violations;
            }
            if let Some(targets) = event.target_count {
                self.target_count += targets;
            }
        }
    }
}

/// Summary of validation analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub timestamp: DateTime<Utc>,
    pub total_validations: usize,
    pub total_violations: usize,
    pub total_targets_validated: usize,
    pub overall_success_rate: f64,
    pub most_violated_shapes: Vec<ShapeViolationSummary>,
    pub slowest_constraints: Vec<ConstraintPerformanceSummary>,
    pub validation_trends: ValidationTrends,
    pub shape_count: usize,
    pub constraint_count: usize,
}

/// Shape violation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeViolationSummary {
    pub shape_id: String,
    pub total_violations: usize,
    pub violation_rate: f64,
    pub validation_count: usize,
}

/// Constraint performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintPerformanceSummary {
    pub constraint_id: String,
    pub average_evaluation_time: Duration,
    pub total_evaluations: usize,
    pub violation_rate: f64,
    pub cache_hit_rate: f64,
}

/// Validation trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTrends {
    pub violation_trend: TrendDirection,
    pub validation_volume_trend: TrendDirection,
    pub data_points: usize,
}

impl Default for ValidationTrends {
    fn default() -> Self {
        Self {
            violation_trend: TrendDirection::Stable,
            validation_volume_trend: TrendDirection::Stable,
            data_points: 0,
        }
    }
}

/// Trend direction enum
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_analytics_creation() {
        let analytics = ValidationAnalytics::new();
        assert!(analytics.events.is_empty());
        assert!(analytics.shape_statistics.is_empty());
        assert!(analytics.constraint_statistics.is_empty());
    }

    #[test]
    fn test_shape_statistics() {
        let mut stats = ShapeStatistics::new("test_shape".to_string());
        assert_eq!(stats.validation_count, 0);
        assert_eq!(stats.total_violations, 0);

        stats.validation_count = 5;
        stats.total_validation_time = Duration::from_secs(10);
        stats.update_avg_validation_time();

        assert_eq!(stats.average_validation_time, Duration::from_secs(2));
    }
}
