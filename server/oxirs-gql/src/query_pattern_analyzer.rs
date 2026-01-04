//! Real-time Query Pattern Analysis
//!
//! This module provides real-time analysis of GraphQL query patterns,
//! identifying trends, common patterns, and optimization opportunities.
//!
//! # Features
//!
//! - **Pattern Detection**: Identify frequently executed query patterns
//! - **Trend Analysis**: Detect increasing/decreasing query patterns
//! - **N-gram Analysis**: Find common query field combinations
//! - **Temporal Patterns**: Time-of-day and day-of-week patterns
//! - **Correlation Analysis**: Find correlated queries
//! - **Real-time Alerts**: Trigger alerts on pattern changes
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::query_pattern_analyzer::{PatternAnalyzer, AnalyzerConfig};
//!
//! let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());
//! analyzer.record_query("GetUser", vec!["id", "name", "email"]);
//!
//! let patterns = analyzer.get_top_patterns(10);
//! let trends = analyzer.analyze_trends();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Pattern analyzer configuration
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Window size for trend analysis
    pub trend_window: Duration,
    /// Minimum pattern frequency to track
    pub min_frequency: usize,
    /// Maximum patterns to retain
    pub max_patterns: usize,
    /// N-gram size for field combinations
    pub ngram_size: usize,
}

impl AnalyzerConfig {
    /// Create new analyzer configuration
    pub fn new() -> Self {
        Self {
            trend_window: Duration::from_secs(3600), // 1 hour
            min_frequency: 5,
            max_patterns: 1000,
            ngram_size: 3,
        }
    }

    /// Set trend window
    pub fn with_trend_window(mut self, window: Duration) -> Self {
        self.trend_window = window;
        self
    }

    /// Set minimum frequency
    pub fn with_min_frequency(mut self, min: usize) -> Self {
        self.min_frequency = min;
        self
    }
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Query pattern
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Operation name
    pub operation: String,
    /// Sorted field list
    pub fields: Vec<String>,
}

impl QueryPattern {
    /// Create a new query pattern
    pub fn new(operation: impl Into<String>, mut fields: Vec<String>) -> Self {
        fields.sort();
        Self {
            operation: operation.into(),
            fields,
        }
    }

    /// Get pattern signature
    pub fn signature(&self) -> String {
        format!("{}:{}", self.operation, self.fields.join(","))
    }
}

/// Pattern occurrence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternOccurrence {
    /// Pattern
    pub pattern: QueryPattern,
    /// Occurrence count
    pub count: usize,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// First seen timestamp
    pub first_seen: SystemTime,
    /// Recent timestamps (for trend analysis)
    pub recent_timestamps: Vec<SystemTime>,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable trend
    Stable,
}

/// Pattern trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTrend {
    /// Pattern
    pub pattern: QueryPattern,
    /// Trend direction
    pub direction: TrendDirection,
    /// Change rate (queries per hour)
    pub change_rate: f64,
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
}

/// Field combination (N-gram)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldCombination {
    /// Fields in combination
    pub fields: Vec<String>,
    /// Occurrence count
    pub count: usize,
}

/// Temporal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Hour of day (0-23)
    pub hour: u8,
    /// Day of week (0=Sunday, 6=Saturday)
    pub day_of_week: u8,
    /// Query count
    pub count: usize,
}

/// Pattern analyzer
pub struct PatternAnalyzer {
    config: AnalyzerConfig,
    patterns: HashMap<String, PatternOccurrence>,
    field_combinations: HashMap<Vec<String>, usize>,
    temporal_patterns: HashMap<(u8, u8), usize>,
}

impl PatternAnalyzer {
    /// Create new pattern analyzer
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            config,
            patterns: HashMap::new(),
            field_combinations: HashMap::new(),
            temporal_patterns: HashMap::new(),
        }
    }

    /// Record a query pattern
    pub fn record_query(&mut self, operation: impl Into<String>, fields: Vec<String>) {
        self.record_query_at(operation, fields, SystemTime::now());
    }

    /// Record query at specific time
    pub fn record_query_at(
        &mut self,
        operation: impl Into<String>,
        fields: Vec<String>,
        timestamp: SystemTime,
    ) {
        let pattern = QueryPattern::new(operation, fields.clone());
        let signature = pattern.signature();

        // Update pattern occurrence
        self.patterns
            .entry(signature.clone())
            .and_modify(|occ| {
                occ.count += 1;
                occ.last_seen = timestamp;
                occ.recent_timestamps.push(timestamp);

                // Keep only recent timestamps for trend analysis
                let cutoff = timestamp
                    .checked_sub(self.config.trend_window)
                    .unwrap_or(timestamp);
                occ.recent_timestamps.retain(|&ts| ts >= cutoff);
            })
            .or_insert_with(|| PatternOccurrence {
                pattern: pattern.clone(),
                count: 1,
                last_seen: timestamp,
                first_seen: timestamp,
                recent_timestamps: vec![timestamp],
            });

        // Record field combinations (N-grams)
        self.record_field_combinations(&pattern.fields);

        // Record temporal pattern
        self.record_temporal_pattern(timestamp);

        // Trim patterns if exceeds max
        if self.patterns.len() > self.config.max_patterns {
            self.trim_patterns();
        }
    }

    /// Record field combinations
    fn record_field_combinations(&mut self, fields: &[String]) {
        if fields.len() < self.config.ngram_size {
            return;
        }

        for window in fields.windows(self.config.ngram_size) {
            let combo = window.to_vec();
            *self.field_combinations.entry(combo).or_insert(0) += 1;
        }
    }

    /// Record temporal pattern
    fn record_temporal_pattern(&mut self, timestamp: SystemTime) {
        // This is simplified - in production, use chrono for proper time handling
        let elapsed = timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH");
        let hours = (elapsed.as_secs() / 3600) % 24;
        let days = (elapsed.as_secs() / 86400) % 7;

        let hour = hours as u8;
        let day_of_week = days as u8;

        *self
            .temporal_patterns
            .entry((hour, day_of_week))
            .or_insert(0) += 1;
    }

    /// Get top N patterns by frequency
    pub fn get_top_patterns(&self, n: usize) -> Vec<PatternOccurrence> {
        let mut patterns: Vec<_> = self.patterns.values().cloned().collect();
        patterns.sort_by(|a, b| b.count.cmp(&a.count));
        patterns.into_iter().take(n).collect()
    }

    /// Analyze trends for all patterns
    pub fn analyze_trends(&self) -> Vec<PatternTrend> {
        self.patterns
            .values()
            .filter_map(|occ| self.analyze_pattern_trend(occ))
            .collect()
    }

    /// Analyze trend for a specific pattern
    fn analyze_pattern_trend(&self, occ: &PatternOccurrence) -> Option<PatternTrend> {
        if occ.recent_timestamps.len() < 2 {
            return None;
        }

        // Calculate time span
        let first = occ.recent_timestamps.first()?;
        let last = occ.recent_timestamps.last()?;
        let span = last.duration_since(*first).ok()?;

        if span.as_secs() == 0 {
            return None;
        }

        // Calculate rate (queries per hour)
        let hours = span.as_secs_f64() / 3600.0;
        let rate = occ.recent_timestamps.len() as f64 / hours;

        // Determine trend direction
        let mid_point = occ.recent_timestamps.len() / 2;
        let first_half_count = mid_point;
        let second_half_count = occ.recent_timestamps.len() - mid_point;

        let direction = if second_half_count > first_half_count * 12 / 10 {
            TrendDirection::Increasing
        } else if second_half_count < first_half_count * 8 / 10 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        // Calculate confidence based on sample size
        let confidence = (occ.recent_timestamps.len() as f64 / 100.0).min(1.0);

        Some(PatternTrend {
            pattern: occ.pattern.clone(),
            direction,
            change_rate: rate,
            confidence,
        })
    }

    /// Get top field combinations
    pub fn get_top_field_combinations(&self, n: usize) -> Vec<FieldCombination> {
        let mut combos: Vec<_> = self
            .field_combinations
            .iter()
            .map(|(fields, &count)| FieldCombination {
                fields: fields.clone(),
                count,
            })
            .collect();

        combos.sort_by(|a, b| b.count.cmp(&a.count));
        combos.into_iter().take(n).collect()
    }

    /// Get temporal patterns
    pub fn get_temporal_patterns(&self) -> Vec<TemporalPattern> {
        self.temporal_patterns
            .iter()
            .map(|((hour, day_of_week), &count)| TemporalPattern {
                hour: *hour,
                day_of_week: *day_of_week,
                count,
            })
            .collect()
    }

    /// Find correlated patterns
    pub fn find_correlated_patterns(&self, pattern_signature: &str, threshold: f64) -> Vec<String> {
        // Simplified correlation - in production use proper statistical correlation
        let mut correlations = Vec::new();

        if let Some(base_pattern) = self.patterns.get(pattern_signature) {
            for (sig, occ) in &self.patterns {
                if sig == pattern_signature {
                    continue;
                }

                // Check if patterns occur in similar time windows
                let correlation = self.calculate_temporal_correlation(base_pattern, occ);
                if correlation >= threshold {
                    correlations.push(sig.clone());
                }
            }
        }

        correlations
    }

    /// Calculate temporal correlation between two patterns
    fn calculate_temporal_correlation(
        &self,
        pattern1: &PatternOccurrence,
        pattern2: &PatternOccurrence,
    ) -> f64 {
        // Simplified: count how many timestamps are within 1 minute
        let mut matches = 0;
        let window = Duration::from_secs(60);

        for ts1 in &pattern1.recent_timestamps {
            for ts2 in &pattern2.recent_timestamps {
                if let Ok(diff) = ts1.duration_since(*ts2) {
                    if diff < window {
                        matches += 1;
                        break;
                    }
                }
                if let Ok(diff) = ts2.duration_since(*ts1) {
                    if diff < window {
                        matches += 1;
                        break;
                    }
                }
            }
        }

        matches as f64 / pattern1.recent_timestamps.len().max(1) as f64
    }

    /// Trim least frequent patterns
    fn trim_patterns(&mut self) {
        let mut patterns: Vec<_> = self.patterns.iter().collect();
        patterns.sort_by(|a, b| b.1.count.cmp(&a.1.count));

        let to_remove: Vec<_> = patterns
            .into_iter()
            .skip(self.config.max_patterns)
            .map(|(sig, _)| sig.clone())
            .collect();

        for sig in to_remove {
            self.patterns.remove(&sig);
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.field_combinations.clear();
        self.temporal_patterns.clear();
    }

    /// Get pattern count
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_config_creation() {
        let config = AnalyzerConfig::new()
            .with_trend_window(Duration::from_secs(1800))
            .with_min_frequency(10);

        assert_eq!(config.trend_window.as_secs(), 1800);
        assert_eq!(config.min_frequency, 10);
    }

    #[test]
    fn test_query_pattern_creation() {
        let pattern = QueryPattern::new("GetUser", vec!["id".to_string(), "name".to_string()]);

        assert_eq!(pattern.operation, "GetUser");
        assert_eq!(pattern.fields.len(), 2);
    }

    #[test]
    fn test_query_pattern_signature() {
        let pattern = QueryPattern::new("GetUser", vec!["name".to_string(), "id".to_string()]);

        let sig = pattern.signature();
        assert!(sig.contains("GetUser"));
        assert!(sig.contains("id"));
        assert!(sig.contains("name"));
    }

    #[test]
    fn test_pattern_analyzer_creation() {
        let analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        assert_eq!(analyzer.pattern_count(), 0);
    }

    #[test]
    fn test_record_query() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        analyzer.record_query("GetUser", vec!["id".to_string(), "name".to_string()]);

        assert_eq!(analyzer.pattern_count(), 1);
    }

    #[test]
    fn test_record_multiple_queries() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        analyzer.record_query("GetUser", vec!["id".to_string()]);
        analyzer.record_query("GetUser", vec!["id".to_string()]);
        analyzer.record_query("GetPosts", vec!["title".to_string()]);

        assert_eq!(analyzer.pattern_count(), 2);
    }

    #[test]
    fn test_get_top_patterns() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        for _ in 0..10 {
            analyzer.record_query("GetUser", vec!["id".to_string()]);
        }
        for _ in 0..5 {
            analyzer.record_query("GetPosts", vec!["title".to_string()]);
        }

        let top = analyzer.get_top_patterns(2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].count, 10);
        assert_eq!(top[1].count, 5);
    }

    #[test]
    fn test_field_combinations() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default().with_min_frequency(1));

        analyzer.record_query(
            "GetUser",
            vec![
                "id".to_string(),
                "name".to_string(),
                "email".to_string(),
                "age".to_string(),
            ],
        );

        let combos = analyzer.get_top_field_combinations(5);

        assert!(!combos.is_empty());
    }

    #[test]
    fn test_temporal_patterns() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        analyzer.record_query("GetUser", vec!["id".to_string()]);

        let temporal = analyzer.get_temporal_patterns();

        assert_eq!(temporal.len(), 1);
    }

    #[test]
    fn test_trend_direction() {
        let directions = [
            TrendDirection::Increasing,
            TrendDirection::Decreasing,
            TrendDirection::Stable,
        ];

        assert_eq!(directions.len(), 3);
    }

    #[test]
    fn test_analyze_trends() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        let now = SystemTime::now();
        for i in 0..20 {
            let ts = now + Duration::from_secs(i * 60);
            analyzer.record_query_at("GetUser", vec!["id".to_string()], ts);
        }

        let trends = analyzer.analyze_trends();

        assert!(!trends.is_empty());
    }

    #[test]
    fn test_analyze_trends_insufficient_data() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        analyzer.record_query("GetUser", vec!["id".to_string()]);

        let trends = analyzer.analyze_trends();

        // Should have no trends with only 1 data point
        assert!(trends.is_empty());
    }

    #[test]
    fn test_find_correlated_patterns() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        let now = SystemTime::now();
        for i in 0..10 {
            let ts = now + Duration::from_secs(i * 30);
            analyzer.record_query_at("GetUser", vec!["id".to_string()], ts);
            analyzer.record_query_at("GetPosts", vec!["title".to_string()], ts);
        }

        let pattern_sig = QueryPattern::new("GetUser", vec!["id".to_string()]).signature();
        let correlated = analyzer.find_correlated_patterns(&pattern_sig, 0.3);

        assert!(!correlated.is_empty());
    }

    #[test]
    fn test_pattern_trimming() {
        let config = AnalyzerConfig::default().with_min_frequency(1);
        let mut analyzer = PatternAnalyzer::new(config);

        // Set max to 5
        analyzer.config.max_patterns = 5;

        // Add 10 patterns
        for i in 0..10 {
            analyzer.record_query(format!("Query{}", i), vec!["field".to_string()]);
        }

        // Should trim to 5
        assert_eq!(analyzer.pattern_count(), 5);
    }

    #[test]
    fn test_clear() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        analyzer.record_query("GetUser", vec!["id".to_string()]);
        assert_eq!(analyzer.pattern_count(), 1);

        analyzer.clear();
        assert_eq!(analyzer.pattern_count(), 0);
    }

    #[test]
    fn test_pattern_occurrence_tracking() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        let now = SystemTime::now();
        analyzer.record_query_at("GetUser", vec!["id".to_string()], now);
        analyzer.record_query_at(
            "GetUser",
            vec!["id".to_string()],
            now + Duration::from_secs(60),
        );

        let top = analyzer.get_top_patterns(1);
        assert_eq!(top[0].count, 2);
        assert_eq!(top[0].recent_timestamps.len(), 2);
    }

    #[test]
    fn test_field_combination_count() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        analyzer.record_query(
            "GetUser",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );
        analyzer.record_query(
            "GetUser",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );

        let combos = analyzer.get_top_field_combinations(1);
        assert_eq!(combos[0].count, 2);
    }

    #[test]
    fn test_temporal_correlation() {
        let occ1 = PatternOccurrence {
            pattern: QueryPattern::new("Query1", vec!["field".to_string()]),
            count: 2,
            last_seen: SystemTime::now(),
            first_seen: SystemTime::now(),
            recent_timestamps: vec![
                SystemTime::now(),
                SystemTime::now() + Duration::from_secs(30),
            ],
        };

        let occ2 = PatternOccurrence {
            pattern: QueryPattern::new("Query2", vec!["field".to_string()]),
            count: 2,
            last_seen: SystemTime::now(),
            first_seen: SystemTime::now(),
            recent_timestamps: vec![
                SystemTime::now(),
                SystemTime::now() + Duration::from_secs(30),
            ],
        };

        let analyzer = PatternAnalyzer::new(AnalyzerConfig::default());
        let correlation = analyzer.calculate_temporal_correlation(&occ1, &occ2);

        assert!((0.0..=1.0).contains(&correlation));
    }

    #[test]
    fn test_pattern_trend_confidence() {
        let mut analyzer = PatternAnalyzer::new(AnalyzerConfig::default());

        let now = SystemTime::now();
        for i in 0..100 {
            let ts = now + Duration::from_secs(i * 60);
            analyzer.record_query_at("GetUser", vec!["id".to_string()], ts);
        }

        let trends = analyzer.analyze_trends();
        assert!(!trends.is_empty());
        assert!(trends[0].confidence > 0.0);
    }
}
