//! Pattern Detection Module
//!
//! This module provides advanced pattern detection capabilities for SHACL validation,
//! including failure pattern analysis, performance bottleneck detection, and data quality insights.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Pattern detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetectionConfig {
    /// Enable failure pattern detection
    pub enable_failure_patterns: bool,
    /// Enable performance pattern detection
    pub enable_performance_patterns: bool,
    /// Enable data quality pattern detection  
    pub enable_data_quality_patterns: bool,
    /// Minimum pattern occurrence threshold
    pub min_pattern_threshold: usize,
    /// Pattern detection window duration
    pub detection_window: Duration,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
}

impl Default for PatternDetectionConfig {
    fn default() -> Self {
        Self {
            enable_failure_patterns: true,
            enable_performance_patterns: true,
            enable_data_quality_patterns: true,
            min_pattern_threshold: 3,
            detection_window: Duration::from_secs(3600), // 1 hour
            enable_anomaly_detection: true,
        }
    }
}

/// Types of patterns that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Validation failure patterns
    ValidationFailure,
    /// Performance bottleneck patterns
    PerformanceBottleneck,
    /// Data quality issues
    DataQuality,
    /// Constraint usage patterns
    ConstraintUsage,
    /// Target selection patterns
    TargetSelection,
    /// Anomalous behavior
    Anomaly,
}

/// Detected pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern description
    pub description: String,
    /// Pattern severity (0.0 to 1.0)
    pub severity: f64,
    /// Number of occurrences
    pub occurrence_count: usize,
    /// First detected timestamp
    pub first_detected: DateTime<Utc>,
    /// Last detected timestamp
    pub last_detected: DateTime<Utc>,
    /// Pattern metadata
    pub metadata: HashMap<String, String>,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Pattern detection engine
#[derive(Debug)]
pub struct PatternDetector {
    config: PatternDetectionConfig,
    detected_patterns: HashMap<String, DetectedPattern>,
    failure_patterns: HashMap<String, usize>,
    performance_patterns: HashMap<String, Vec<Duration>>,
    data_quality_patterns: HashMap<String, usize>,
    last_cleanup: DateTime<Utc>,
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new(config: PatternDetectionConfig) -> Self {
        Self {
            config,
            detected_patterns: HashMap::new(),
            failure_patterns: HashMap::new(),
            performance_patterns: HashMap::new(),
            data_quality_patterns: HashMap::new(),
            last_cleanup: Utc::now(),
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(PatternDetectionConfig::default())
    }

    /// Record a validation failure for pattern analysis
    pub fn record_validation_failure(
        &mut self,
        shape_iri: &str,
        constraint_type: &str,
        failure_reason: &str,
    ) {
        if !self.config.enable_failure_patterns {
            return;
        }

        let pattern_key = format!("{shape_iri}:{constraint_type}:{failure_reason}");
        let new_count = {
            let count = self
                .failure_patterns
                .entry(pattern_key.clone())
                .or_insert(0);
            *count += 1;
            *count
        };

        if new_count >= self.config.min_pattern_threshold {
            self.create_or_update_pattern(
                PatternType::ValidationFailure,
                pattern_key,
                format!("Recurring validation failure in shape {shape_iri} for constraint {constraint_type} with reason: {failure_reason}"),
                0.7, // Medium-high severity
                new_count,
                vec![
                    "Review shape definition for potential issues".to_string(),
                    "Check data quality for this constraint".to_string(),
                    "Consider relaxing constraint if appropriate".to_string(),
                ],
            );
        }
    }

    /// Record performance data for pattern analysis
    pub fn record_performance_data(&mut self, operation: &str, duration: Duration) {
        if !self.config.enable_performance_patterns {
            return;
        }

        let (durations_len, avg_duration) = {
            let durations = self
                .performance_patterns
                .entry(operation.to_string())
                .or_default();
            durations.push(duration);

            // Keep only recent measurements
            durations.retain(|d| d.as_millis() > 0);

            let len = durations.len();
            let avg_duration = if len > 0 {
                Duration::from_millis(
                    durations.iter().map(|d| d.as_millis() as u64).sum::<u64>() / len as u64,
                )
            } else {
                Duration::from_millis(0)
            };
            (len, avg_duration)
        };

        // Detect slow operation patterns
        if durations_len >= self.config.min_pattern_threshold
            && avg_duration > Duration::from_millis(1000)
        {
            self.create_or_update_pattern(
                PatternType::PerformanceBottleneck,
                format!("slow_{operation}"),
                format!("Performance bottleneck detected in operation: {operation} (avg: {avg_duration:?})"),
                0.8, // High severity
                durations_len,
                vec![
                    "Optimize the operation implementation".to_string(),
                    "Consider caching strategies".to_string(),
                    "Review data structures and algorithms".to_string(),
                ],
            );
        }
    }

    /// Record data quality issues for pattern analysis
    pub fn record_data_quality_issue(&mut self, issue_type: &str, resource_iri: &str) {
        if !self.config.enable_data_quality_patterns {
            return;
        }

        let pattern_key = format!("{issue_type}:{resource_iri}");
        let new_count = {
            let count = self
                .data_quality_patterns
                .entry(pattern_key.clone())
                .or_insert(0);
            *count += 1;
            *count
        };

        if new_count >= self.config.min_pattern_threshold {
            self.create_or_update_pattern(
                PatternType::DataQuality,
                pattern_key,
                format!("Data quality issue detected: {issue_type} for resource {resource_iri}"),
                0.6, // Medium severity
                new_count,
                vec![
                    "Review data source quality".to_string(),
                    "Implement data validation rules".to_string(),
                    "Consider data cleansing processes".to_string(),
                ],
            );
        }
    }

    /// Detect anomalous behavior
    pub fn detect_anomalies(&mut self) -> Vec<DetectedPattern> {
        if !self.config.enable_anomaly_detection {
            return Vec::new();
        }

        let mut anomalies = Vec::new();

        // Detect sudden spikes in failure patterns
        for (pattern_key, count) in &self.failure_patterns {
            if *count > self.config.min_pattern_threshold * 5 {
                // 5x threshold
                let anomaly = DetectedPattern {
                    pattern_type: PatternType::Anomaly,
                    pattern_id: format!("anomaly_spike_{pattern_key}"),
                    description: format!("Anomalous spike in failures: {count} occurrences"),
                    severity: 0.9,
                    occurrence_count: *count,
                    first_detected: Utc::now(),
                    last_detected: Utc::now(),
                    metadata: HashMap::new(),
                    suggested_actions: vec![
                        "Investigate root cause immediately".to_string(),
                        "Check for data source changes".to_string(),
                        "Review recent system changes".to_string(),
                    ],
                };
                anomalies.push(anomaly);
            }
        }

        anomalies
    }

    /// Get all detected patterns
    pub fn get_detected_patterns(&self) -> Vec<&DetectedPattern> {
        self.detected_patterns.values().collect()
    }

    /// Get patterns by type
    pub fn get_patterns_by_type(&self, pattern_type: PatternType) -> Vec<&DetectedPattern> {
        self.detected_patterns
            .values()
            .filter(|p| p.pattern_type == pattern_type)
            .collect()
    }

    /// Clear old patterns based on detection window
    pub fn cleanup_old_patterns(&mut self) {
        let cutoff = Utc::now()
            - chrono::Duration::from_std(self.config.detection_window).unwrap_or_default();

        self.detected_patterns
            .retain(|_, pattern| pattern.last_detected > cutoff);

        // Also clean up raw data
        if Utc::now()
            .signed_duration_since(self.last_cleanup)
            .num_minutes()
            > 60
        {
            self.failure_patterns.clear();
            self.performance_patterns.clear();
            self.data_quality_patterns.clear();
            self.last_cleanup = Utc::now();
        }
    }

    /// Generate pattern analysis report
    pub fn generate_report(&self) -> PatternAnalysisReport {
        let patterns_by_type: HashMap<PatternType, Vec<&DetectedPattern>> = self
            .detected_patterns
            .values()
            .fold(HashMap::new(), |mut acc, pattern| {
                acc.entry(pattern.pattern_type.clone())
                    .or_default()
                    .push(pattern);
                acc
            });

        let high_severity_patterns = self
            .detected_patterns
            .values()
            .filter(|p| p.severity >= 0.8)
            .cloned()
            .collect();

        PatternAnalysisReport {
            total_patterns: self.detected_patterns.len(),
            patterns_by_type: patterns_by_type
                .into_iter()
                .map(|(k, v)| (k, v.len()))
                .collect(),
            high_severity_patterns,
            analysis_timestamp: Utc::now(),
            summary: self.generate_summary(),
        }
    }

    /// Create or update a detected pattern
    fn create_or_update_pattern(
        &mut self,
        pattern_type: PatternType,
        pattern_id: String,
        description: String,
        severity: f64,
        occurrence_count: usize,
        suggested_actions: Vec<String>,
    ) {
        let now = Utc::now();

        match self.detected_patterns.get_mut(&pattern_id) {
            Some(existing_pattern) => {
                existing_pattern.occurrence_count = occurrence_count;
                existing_pattern.last_detected = now;
                existing_pattern.severity = severity;
            }
            None => {
                let pattern = DetectedPattern {
                    pattern_type,
                    pattern_id: pattern_id.clone(),
                    description,
                    severity,
                    occurrence_count,
                    first_detected: now,
                    last_detected: now,
                    metadata: HashMap::new(),
                    suggested_actions,
                };
                self.detected_patterns.insert(pattern_id, pattern);
            }
        }
    }

    /// Generate summary of detected patterns
    fn generate_summary(&self) -> String {
        let total = self.detected_patterns.len();
        let high_severity = self
            .detected_patterns
            .values()
            .filter(|p| p.severity >= 0.8)
            .count();
        let medium_severity = self
            .detected_patterns
            .values()
            .filter(|p| p.severity >= 0.5 && p.severity < 0.8)
            .count();
        let low_severity = total - high_severity - medium_severity;

        format!(
            "Pattern Analysis Summary: {total} total patterns detected ({high_severity} high severity, {medium_severity} medium severity, {low_severity} low severity)"
        )
    }
}

/// Pattern analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysisReport {
    /// Total number of patterns detected
    pub total_patterns: usize,
    /// Patterns grouped by type
    pub patterns_by_type: HashMap<PatternType, usize>,
    /// High severity patterns requiring immediate attention
    pub high_severity_patterns: Vec<DetectedPattern>,
    /// Analysis timestamp
    pub analysis_timestamp: DateTime<Utc>,
    /// Summary description
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_detector_creation() {
        let detector = PatternDetector::with_default_config();
        assert_eq!(detector.detected_patterns.len(), 0);
    }

    #[test]
    fn test_validation_failure_pattern_detection() {
        let mut detector = PatternDetector::with_default_config();

        // Record multiple failures for the same pattern
        for _ in 0..5 {
            detector.record_validation_failure(
                "http://example.org/shape1",
                "minCount",
                "insufficient_values",
            );
        }

        let patterns = detector.get_detected_patterns();
        assert!(!patterns.is_empty());
        assert!(patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::ValidationFailure));
    }

    #[test]
    fn test_performance_pattern_detection() {
        let mut detector = PatternDetector::with_default_config();

        // Record multiple slow operations
        for _ in 0..5 {
            detector.record_performance_data("shape_validation", Duration::from_millis(2000));
        }

        let patterns = detector.get_detected_patterns();
        assert!(patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::PerformanceBottleneck));
    }

    #[test]
    fn test_data_quality_pattern_detection() {
        let mut detector = PatternDetector::with_default_config();

        // Record multiple data quality issues
        for _ in 0..5 {
            detector.record_data_quality_issue(
                "missing_required_property",
                "http://example.org/resource1",
            );
        }

        let patterns = detector.get_detected_patterns();
        assert!(patterns
            .iter()
            .any(|p| p.pattern_type == PatternType::DataQuality));
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = PatternDetector::with_default_config();

        // Create an anomalous spike in failures
        for _ in 0..20 {
            // Way above threshold
            detector.record_validation_failure(
                "http://example.org/shape1",
                "minCount",
                "insufficient_values",
            );
        }

        let anomalies = detector.detect_anomalies();
        assert!(!anomalies.is_empty());
        assert!(anomalies
            .iter()
            .any(|a| a.pattern_type == PatternType::Anomaly));
    }

    #[test]
    fn test_pattern_cleanup() {
        let mut detector = PatternDetector::with_default_config();

        // Add some patterns
        detector.record_validation_failure("shape1", "constraint1", "reason1");
        assert!(!detector.failure_patterns.is_empty());

        // Cleanup should remove old data
        detector.cleanup_old_patterns();
        // Since we just added the data, it shouldn't be cleaned up yet
        assert!(!detector.failure_patterns.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let mut detector = PatternDetector::with_default_config();

        // Add various patterns
        for _ in 0..5 {
            detector.record_validation_failure("shape1", "constraint1", "reason1");
            detector.record_performance_data("operation1", Duration::from_millis(2000));
            detector.record_data_quality_issue("issue1", "resource1");
        }

        let report = detector.generate_report();
        assert!(report.total_patterns > 0);
        assert!(!report.patterns_by_type.is_empty());
        assert!(!report.summary.is_empty());
    }

    #[test]
    fn test_patterns_by_type_filtering() {
        let mut detector = PatternDetector::with_default_config();

        // Add different types of patterns
        for _ in 0..5 {
            detector.record_validation_failure("shape1", "constraint1", "reason1");
            detector.record_performance_data("operation1", Duration::from_millis(2000));
        }

        let validation_patterns = detector.get_patterns_by_type(PatternType::ValidationFailure);
        let performance_patterns =
            detector.get_patterns_by_type(PatternType::PerformanceBottleneck);

        assert!(!validation_patterns.is_empty());
        assert!(!performance_patterns.is_empty());
    }
}
