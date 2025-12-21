//! Prometheus Metric Cardinality Optimization
//!
//! Provides tools to manage and optimize Prometheus metric cardinality to prevent
//! memory issues and performance degradation.
//!
//! # Features
//!
//! - Label cardinality tracking and analysis
//! - Automatic label pruning and aggregation
//! - Configurable cardinality limits
//! - Dynamic label bucketing
//! - Label value allowlisting and denylisting
//! - Cardinality reporting and alerts
//! - Label normalization strategies

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Type alias for nested label values map: metric -> label -> value -> count
type LabelValuesMap = HashMap<String, HashMap<String, HashMap<String, usize>>>;

/// Configuration for cardinality optimization
#[derive(Debug, Clone)]
pub struct CardinalityConfig {
    /// Maximum unique label combinations per metric
    pub max_cardinality: usize,
    /// Maximum unique values per label
    pub max_label_values: usize,
    /// Enable automatic label pruning
    pub auto_prune: bool,
    /// Labels to exclude from cardinality limits
    pub exempt_labels: HashSet<String>,
    /// Enable label bucketing
    pub enable_bucketing: bool,
    /// Bucket size for numeric labels
    pub bucket_size: f64,
}

impl Default for CardinalityConfig {
    fn default() -> Self {
        Self {
            max_cardinality: 10_000,
            max_label_values: 1_000,
            auto_prune: true,
            exempt_labels: HashSet::new(),
            enable_bucketing: true,
            bucket_size: 100.0,
        }
    }
}

/// Label cardinality information
#[derive(Debug, Clone)]
pub struct LabelCardinality {
    pub label_name: String,
    pub unique_values: usize,
    pub total_occurrences: usize,
    pub sample_values: Vec<String>,
}

/// Metric cardinality information
#[derive(Debug, Clone)]
pub struct MetricCardinality {
    pub metric_name: String,
    pub total_series: usize,
    pub label_cardinalities: Vec<LabelCardinality>,
    pub exceeds_limit: bool,
}

/// Label value normalization strategy
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationStrategy {
    /// Keep original value
    None,
    /// Bucket numeric values
    Bucket { size: f64 },
    /// Truncate to prefix
    Prefix { length: usize },
    /// Hash long values
    Hash,
    /// Map to predefined categories
    Category { categories: Vec<String> },
    /// Custom transformation function name
    Custom(String),
}

/// Label allowlist/denylist
#[derive(Debug, Clone)]
pub struct LabelFilter {
    /// Allowed label values (if set, only these are allowed)
    pub allowlist: Option<HashSet<String>>,
    /// Denied label values
    pub denylist: HashSet<String>,
    /// Normalization strategy
    pub normalization: NormalizationStrategy,
}

/// Cardinality optimizer for Prometheus metrics
pub struct CardinalityOptimizer {
    config: CardinalityConfig,
    /// Track label value occurrences per metric and label
    label_values: Arc<RwLock<LabelValuesMap>>,
    /// Track total series count per metric
    series_counts: Arc<RwLock<HashMap<String, usize>>>,
    /// Label filters per metric/label combination
    filters: Arc<RwLock<HashMap<String, HashMap<String, LabelFilter>>>>,
    /// Cardinality warnings
    warnings: Arc<RwLock<Vec<String>>>,
    /// Last reset time
    last_reset: Arc<RwLock<Instant>>,
    /// Reset interval
    reset_interval: Duration,
}

impl CardinalityOptimizer {
    /// Create a new cardinality optimizer
    pub fn new(config: CardinalityConfig) -> Self {
        Self {
            config,
            label_values: Arc::new(RwLock::new(HashMap::new())),
            series_counts: Arc::new(RwLock::new(HashMap::new())),
            filters: Arc::new(RwLock::new(HashMap::new())),
            warnings: Arc::new(RwLock::new(Vec::new())),
            last_reset: Arc::new(RwLock::new(Instant::now())),
            reset_interval: Duration::from_secs(3600), // Reset hourly
        }
    }

    /// Record a metric with labels
    pub fn record_metric(
        &self,
        metric: &str,
        labels: &HashMap<String, String>,
    ) -> HashMap<String, String> {
        // Normalize labels
        let normalized_labels = self.normalize_labels(metric, labels);

        // Track cardinality - explicitly drop locks to avoid deadlock
        {
            let mut label_values = self.label_values.write().unwrap();
            let metric_labels = label_values.entry(metric.to_string()).or_default();

            for (label_name, label_value) in &normalized_labels {
                let label_map = metric_labels.entry(label_name.clone()).or_default();
                *label_map.entry(label_value.clone()).or_insert(0) += 1;
            }
        } // Drop label_values lock

        // Update series count
        {
            let mut series_counts = self.series_counts.write().unwrap();
            *series_counts.entry(metric.to_string()).or_insert(0) += 1;
        } // Drop series_counts lock

        // Check cardinality limits (acquires read locks internally)
        self.check_cardinality_limits(metric);

        // Auto-reset if needed
        self.check_auto_reset();

        normalized_labels
    }

    /// Normalize labels according to configured strategies
    pub fn normalize_labels(
        &self,
        metric: &str,
        labels: &HashMap<String, String>,
    ) -> HashMap<String, String> {
        let filters = self.filters.read().unwrap();
        let metric_filters = filters.get(metric);

        let mut normalized = HashMap::new();

        for (label_name, label_value) in labels {
            // Skip exempt labels
            if self.config.exempt_labels.contains(label_name) {
                normalized.insert(label_name.clone(), label_value.clone());
                continue;
            }

            // Apply filter if exists
            let filter = metric_filters.and_then(|f| f.get(label_name));

            let normalized_value = if let Some(filter) = filter {
                // Check denylist
                if filter.denylist.contains(label_value) {
                    continue; // Skip denied labels
                }

                // Check allowlist
                if let Some(ref allowlist) = filter.allowlist {
                    if !allowlist.contains(label_value) {
                        continue; // Skip non-allowed labels
                    }
                }

                // Apply normalization
                self.apply_normalization(label_value, &filter.normalization)
            } else {
                // Default normalization
                if self.config.enable_bucketing && Self::is_numeric(label_value) {
                    self.bucket_numeric(label_value, self.config.bucket_size)
                } else {
                    label_value.clone()
                }
            };

            normalized.insert(label_name.clone(), normalized_value);
        }

        normalized
    }

    /// Apply normalization strategy
    fn apply_normalization(&self, value: &str, strategy: &NormalizationStrategy) -> String {
        match strategy {
            NormalizationStrategy::None => value.to_string(),
            NormalizationStrategy::Bucket { size } => self.bucket_numeric(value, *size),
            NormalizationStrategy::Prefix { length } => value.chars().take(*length).collect(),
            NormalizationStrategy::Hash => {
                if value.len() > 32 {
                    format!("hash_{:x}", Self::simple_hash(value))
                } else {
                    value.to_string()
                }
            }
            NormalizationStrategy::Category { categories } => {
                // Find best matching category
                categories
                    .iter()
                    .find(|c| value.contains(c.as_str()))
                    .cloned()
                    .unwrap_or_else(|| "other".to_string())
            }
            NormalizationStrategy::Custom(_name) => {
                // Custom normalization would be implemented via callback
                value.to_string()
            }
        }
    }

    /// Bucket numeric values
    fn bucket_numeric(&self, value: &str, bucket_size: f64) -> String {
        if let Ok(num) = value.parse::<f64>() {
            let bucket = (num / bucket_size).floor() * bucket_size;
            format!("{}-{}", bucket, bucket + bucket_size)
        } else {
            value.to_string()
        }
    }

    /// Register a label filter for a metric
    pub fn register_filter(&self, metric: &str, label: &str, filter: LabelFilter) {
        let mut filters = self.filters.write().unwrap();
        filters
            .entry(metric.to_string())
            .or_default()
            .insert(label.to_string(), filter);
    }

    /// Get cardinality report for a metric
    pub fn get_metric_cardinality(&self, metric: &str) -> Option<MetricCardinality> {
        let label_values = self.label_values.read().unwrap();
        let series_counts = self.series_counts.read().unwrap();

        let metric_labels = label_values.get(metric)?;
        let total_series = *series_counts.get(metric).unwrap_or(&0);

        let mut label_cardinalities = Vec::new();

        for (label_name, values) in metric_labels {
            let unique_values = values.len();
            let total_occurrences: usize = values.values().sum();

            // Get sample values (top 5)
            let mut sample_values: Vec<_> = values.iter().collect();
            sample_values.sort_by(|a, b| b.1.cmp(a.1));
            let sample_values: Vec<String> = sample_values
                .iter()
                .take(5)
                .map(|(k, _)| k.to_string())
                .collect();

            label_cardinalities.push(LabelCardinality {
                label_name: label_name.clone(),
                unique_values,
                total_occurrences,
                sample_values,
            });
        }

        // Sort by cardinality (highest first)
        label_cardinalities.sort_by(|a, b| b.unique_values.cmp(&a.unique_values));

        let exceeds_limit = total_series > self.config.max_cardinality;

        Some(MetricCardinality {
            metric_name: metric.to_string(),
            total_series,
            label_cardinalities,
            exceeds_limit,
        })
    }

    /// Get all metrics with cardinality information
    pub fn get_all_cardinalities(&self) -> Vec<MetricCardinality> {
        let series_counts = self.series_counts.read().unwrap();
        let metrics: Vec<_> = series_counts.keys().cloned().collect();
        drop(series_counts);

        metrics
            .iter()
            .filter_map(|m| self.get_metric_cardinality(m))
            .collect()
    }

    /// Get metrics exceeding cardinality limits
    pub fn get_high_cardinality_metrics(&self) -> Vec<MetricCardinality> {
        self.get_all_cardinalities()
            .into_iter()
            .filter(|m| m.exceeds_limit)
            .collect()
    }

    /// Check cardinality limits and generate warnings
    fn check_cardinality_limits(&self, metric: &str) {
        let series_counts = self.series_counts.read().unwrap();
        let series_count = *series_counts.get(metric).unwrap_or(&0);
        drop(series_counts);

        if series_count > self.config.max_cardinality {
            let warning = format!(
                "Metric '{}' has {} series, exceeding limit of {}",
                metric, series_count, self.config.max_cardinality
            );
            let mut warnings = self.warnings.write().unwrap();
            warnings.push(warning);

            // Auto-prune if enabled
            if self.config.auto_prune {
                self.prune_metric(metric);
            }
        }

        // Check per-label cardinality
        let should_prune_labels = {
            let label_values = self.label_values.read().unwrap();
            let mut should_prune = false;
            if let Some(metric_labels) = label_values.get(metric) {
                for (label_name, values) in metric_labels {
                    if values.len() > self.config.max_label_values {
                        let warning = format!(
                            "Label '{}' on metric '{}' has {} unique values, exceeding limit of {}",
                            label_name,
                            metric,
                            values.len(),
                            self.config.max_label_values
                        );
                        let mut warnings = self.warnings.write().unwrap();
                        warnings.push(warning);
                        should_prune = true;
                    }
                }
            }
            should_prune
        }; // Drop read lock before pruning

        // Auto-prune labels if enabled and needed
        if should_prune_labels && self.config.auto_prune {
            self.prune_metric(metric);
        }
    }

    /// Prune least frequently used label values
    fn prune_metric(&self, metric: &str) {
        let mut label_values = self.label_values.write().unwrap();
        if let Some(metric_labels) = label_values.get_mut(metric) {
            for values in metric_labels.values_mut() {
                if values.len() > self.config.max_label_values {
                    // Keep only top N most frequent values
                    let mut sorted: Vec<_> = values.iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(a.1));

                    let to_keep: HashSet<_> = sorted
                        .iter()
                        .take(self.config.max_label_values)
                        .map(|(k, _)| k.to_string())
                        .collect();

                    values.retain(|k, _| to_keep.contains(k));
                }
            }
        }
    }

    /// Get all warnings
    pub fn get_warnings(&self) -> Vec<String> {
        self.warnings.read().unwrap().clone()
    }

    /// Clear warnings
    pub fn clear_warnings(&self) {
        self.warnings.write().unwrap().clear();
    }

    /// Reset all cardinality data
    pub fn reset(&self) {
        self.label_values.write().unwrap().clear();
        self.series_counts.write().unwrap().clear();
        self.warnings.write().unwrap().clear();
        *self.last_reset.write().unwrap() = Instant::now();
    }

    /// Check if auto-reset is needed
    fn check_auto_reset(&self) {
        let last_reset = self.last_reset.read().unwrap();
        if last_reset.elapsed() >= self.reset_interval {
            drop(last_reset);
            self.reset();
        }
    }

    /// Get total unique series across all metrics
    pub fn total_series(&self) -> usize {
        self.series_counts.read().unwrap().values().sum()
    }

    /// Get number of tracked metrics
    pub fn metric_count(&self) -> usize {
        self.series_counts.read().unwrap().len()
    }

    /// Generate cardinality report as text
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Prometheus Cardinality Report ===\n\n");

        let cardinalities = self.get_all_cardinalities();
        report.push_str(&format!("Total metrics: {}\n", cardinalities.len()));
        report.push_str(&format!("Total series: {}\n", self.total_series()));
        report.push_str(&format!(
            "Cardinality limit: {}\n\n",
            self.config.max_cardinality
        ));

        // High cardinality metrics
        let high_card = self.get_high_cardinality_metrics();
        if !high_card.is_empty() {
            report.push_str("⚠️  High Cardinality Metrics:\n");
            for metric in &high_card {
                report.push_str(&format!(
                    "  - {} ({} series)\n",
                    metric.metric_name, metric.total_series
                ));
            }
            report.push('\n');
        }

        // Top metrics by cardinality
        let mut sorted_metrics = cardinalities;
        sorted_metrics.sort_by(|a, b| b.total_series.cmp(&a.total_series));

        report.push_str("Top 10 Metrics by Cardinality:\n");
        for metric in sorted_metrics.iter().take(10) {
            report.push_str(&format!(
                "  {} - {} series\n",
                metric.metric_name, metric.total_series
            ));

            for label in metric.label_cardinalities.iter().take(3) {
                report.push_str(&format!(
                    "    {} - {} unique values\n",
                    label.label_name, label.unique_values
                ));
            }
        }

        // Warnings
        let warnings = self.get_warnings();
        if !warnings.is_empty() {
            report.push_str("\n⚠️  Warnings:\n");
            for warning in &warnings {
                report.push_str(&format!("  - {}\n", warning));
            }
        }

        report
    }

    // Helper methods

    fn is_numeric(s: &str) -> bool {
        s.parse::<f64>().is_ok()
    }

    fn simple_hash(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_metric_basic() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        let mut labels = HashMap::new();
        labels.insert("endpoint".to_string(), "/api/v1".to_string());

        optimizer.record_metric("http_requests_total", &labels);

        assert_eq!(optimizer.metric_count(), 1);
        assert_eq!(optimizer.total_series(), 1);
    }

    #[test]
    fn test_label_cardinality_tracking() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        for i in 0..10 {
            let mut labels = HashMap::new();
            labels.insert("endpoint".to_string(), format!("/api/v{}", i));
            optimizer.record_metric("http_requests_total", &labels);
        }

        let cardinality = optimizer.get_metric_cardinality("http_requests_total");
        assert!(cardinality.is_some());

        let card = cardinality.unwrap();
        assert_eq!(card.total_series, 10);
        assert_eq!(card.label_cardinalities.len(), 1);
        assert_eq!(card.label_cardinalities[0].unique_values, 10);
    }

    #[test]
    fn test_cardinality_limit_warning() {
        let config = CardinalityConfig {
            max_cardinality: 5,
            ..Default::default()
        };
        let optimizer = CardinalityOptimizer::new(config);

        for i in 0..10 {
            let mut labels = HashMap::new();
            labels.insert("id".to_string(), i.to_string());
            optimizer.record_metric("test_metric", &labels);
        }

        let warnings = optimizer.get_warnings();
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("exceeding limit"));
    }

    #[test]
    fn test_numeric_bucketing() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig {
            enable_bucketing: true,
            bucket_size: 100.0,
            ..Default::default()
        });

        let mut labels1 = HashMap::new();
        labels1.insert("response_time".to_string(), "150".to_string());

        let mut labels2 = HashMap::new();
        labels2.insert("response_time".to_string(), "175".to_string());

        let normalized1 = optimizer.normalize_labels("test", &labels1);
        let normalized2 = optimizer.normalize_labels("test", &labels2);

        // Both should be bucketed to same range
        assert_eq!(
            normalized1.get("response_time"),
            normalized2.get("response_time")
        );
        assert_eq!(normalized1.get("response_time").unwrap(), "100-200");
    }

    #[test]
    fn test_label_allowlist() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        let mut allowlist = HashSet::new();
        allowlist.insert("production".to_string());
        allowlist.insert("staging".to_string());

        optimizer.register_filter(
            "test_metric",
            "environment",
            LabelFilter {
                allowlist: Some(allowlist),
                denylist: HashSet::new(),
                normalization: NormalizationStrategy::None,
            },
        );

        let mut labels1 = HashMap::new();
        labels1.insert("environment".to_string(), "production".to_string());
        let normalized1 = optimizer.normalize_labels("test_metric", &labels1);
        assert!(normalized1.contains_key("environment"));

        let mut labels2 = HashMap::new();
        labels2.insert("environment".to_string(), "development".to_string());
        let normalized2 = optimizer.normalize_labels("test_metric", &labels2);
        assert!(!normalized2.contains_key("environment")); // Filtered out
    }

    #[test]
    fn test_label_denylist() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        let mut denylist = HashSet::new();
        denylist.insert("sensitive_value".to_string());

        optimizer.register_filter(
            "test_metric",
            "user_id",
            LabelFilter {
                allowlist: None,
                denylist,
                normalization: NormalizationStrategy::None,
            },
        );

        let mut labels = HashMap::new();
        labels.insert("user_id".to_string(), "sensitive_value".to_string());
        let normalized = optimizer.normalize_labels("test_metric", &labels);
        assert!(!normalized.contains_key("user_id")); // Denied
    }

    #[test]
    fn test_prefix_normalization() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        optimizer.register_filter(
            "test_metric",
            "transaction_id",
            LabelFilter {
                allowlist: None,
                denylist: HashSet::new(),
                normalization: NormalizationStrategy::Prefix { length: 8 },
            },
        );

        let mut labels = HashMap::new();
        labels.insert("transaction_id".to_string(), "1234567890abcdef".to_string());

        let normalized = optimizer.normalize_labels("test_metric", &labels);
        assert_eq!(normalized.get("transaction_id").unwrap(), "12345678");
    }

    #[test]
    fn test_hash_normalization() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        optimizer.register_filter(
            "test_metric",
            "long_value",
            LabelFilter {
                allowlist: None,
                denylist: HashSet::new(),
                normalization: NormalizationStrategy::Hash,
            },
        );

        let mut labels = HashMap::new();
        labels.insert(
            "long_value".to_string(),
            "this_is_a_very_long_value_that_should_be_hashed".to_string(),
        );

        let normalized = optimizer.normalize_labels("test_metric", &labels);
        let value = normalized.get("long_value").unwrap();
        assert!(value.starts_with("hash_"));
    }

    #[test]
    fn test_category_normalization() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        optimizer.register_filter(
            "test_metric",
            "endpoint",
            LabelFilter {
                allowlist: None,
                denylist: HashSet::new(),
                normalization: NormalizationStrategy::Category {
                    categories: vec!["api".to_string(), "web".to_string(), "admin".to_string()],
                },
            },
        );

        let mut labels1 = HashMap::new();
        labels1.insert("endpoint".to_string(), "/api/users/123".to_string());
        let normalized1 = optimizer.normalize_labels("test_metric", &labels1);
        assert_eq!(normalized1.get("endpoint").unwrap(), "api");

        let mut labels2 = HashMap::new();
        labels2.insert("endpoint".to_string(), "/web/dashboard".to_string());
        let normalized2 = optimizer.normalize_labels("test_metric", &labels2);
        assert_eq!(normalized2.get("endpoint").unwrap(), "web");

        let mut labels3 = HashMap::new();
        labels3.insert("endpoint".to_string(), "/unknown/path".to_string());
        let normalized3 = optimizer.normalize_labels("test_metric", &labels3);
        assert_eq!(normalized3.get("endpoint").unwrap(), "other");
    }

    #[test]
    fn test_exempt_labels() {
        let mut exempt = HashSet::new();
        exempt.insert("critical_label".to_string());

        let config = CardinalityConfig {
            exempt_labels: exempt,
            enable_bucketing: true,
            ..Default::default()
        };
        let optimizer = CardinalityOptimizer::new(config);

        let mut labels = HashMap::new();
        labels.insert("critical_label".to_string(), "123".to_string());

        let normalized = optimizer.normalize_labels("test", &labels);

        // Exempt labels should not be bucketed
        assert_eq!(normalized.get("critical_label").unwrap(), "123");
    }

    #[test]
    fn test_high_cardinality_detection() {
        let config = CardinalityConfig {
            max_cardinality: 5,
            auto_prune: false,
            ..Default::default()
        };
        let optimizer = CardinalityOptimizer::new(config);

        for i in 0..10 {
            let mut labels = HashMap::new();
            labels.insert("id".to_string(), i.to_string());
            optimizer.record_metric("test_metric", &labels);
        }

        let high_card = optimizer.get_high_cardinality_metrics();
        assert_eq!(high_card.len(), 1);
        assert_eq!(high_card[0].metric_name, "test_metric");
        assert!(high_card[0].exceeds_limit);
    }

    #[test]
    fn test_auto_pruning() {
        let config = CardinalityConfig {
            max_label_values: 3,
            auto_prune: true,
            ..Default::default()
        };
        let optimizer = CardinalityOptimizer::new(config);

        // Record same labels multiple times to establish frequency
        for _ in 0..10 {
            let mut labels = HashMap::new();
            labels.insert("frequent".to_string(), "value1".to_string());
            optimizer.record_metric("test", &labels);
        }

        for _ in 0..5 {
            let mut labels = HashMap::new();
            labels.insert("frequent".to_string(), "value2".to_string());
            optimizer.record_metric("test", &labels);
        }

        for _ in 0..2 {
            let mut labels = HashMap::new();
            labels.insert("frequent".to_string(), "value3".to_string());
            optimizer.record_metric("test", &labels);
        }

        // This should trigger pruning
        for i in 4..10 {
            let mut labels = HashMap::new();
            labels.insert("frequent".to_string(), format!("value{}", i));
            optimizer.record_metric("test", &labels);
        }

        let cardinality = optimizer.get_metric_cardinality("test").unwrap();
        let label_card = &cardinality.label_cardinalities[0];

        // Should have pruned to top 3 values
        assert!(label_card.unique_values <= 3);
    }

    #[test]
    fn test_generate_report() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        for i in 0..5 {
            let mut labels = HashMap::new();
            labels.insert("endpoint".to_string(), format!("/api/v{}", i));
            optimizer.record_metric("http_requests", &labels);
        }

        let report = optimizer.generate_report();
        assert!(report.contains("Prometheus Cardinality Report"));
        assert!(report.contains("Total metrics: 1"));
        assert!(report.contains("http_requests"));
    }

    #[test]
    fn test_clear_warnings() {
        let config = CardinalityConfig {
            max_cardinality: 2,
            ..Default::default()
        };
        let optimizer = CardinalityOptimizer::new(config);

        for i in 0..5 {
            let mut labels = HashMap::new();
            labels.insert("id".to_string(), i.to_string());
            optimizer.record_metric("test", &labels);
        }

        assert!(!optimizer.get_warnings().is_empty());

        optimizer.clear_warnings();
        assert!(optimizer.get_warnings().is_empty());
    }

    #[test]
    fn test_reset_clears_all_data() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        let mut labels = HashMap::new();
        labels.insert("test".to_string(), "value".to_string());
        optimizer.record_metric("metric", &labels);

        assert_eq!(optimizer.total_series(), 1);

        optimizer.reset();

        assert_eq!(optimizer.total_series(), 0);
        assert_eq!(optimizer.metric_count(), 0);
    }

    #[test]
    fn test_sample_values_in_cardinality_report() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        for i in 0..10 {
            let mut labels = HashMap::new();
            labels.insert("id".to_string(), format!("value{}", i));
            optimizer.record_metric("test", &labels);
        }

        let cardinality = optimizer.get_metric_cardinality("test").unwrap();
        assert_eq!(cardinality.label_cardinalities.len(), 1);

        let label_card = &cardinality.label_cardinalities[0];
        assert_eq!(label_card.sample_values.len(), 5); // Top 5 samples
    }

    #[test]
    fn test_multiple_metrics_tracking() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        let mut labels1 = HashMap::new();
        labels1.insert("type".to_string(), "a".to_string());
        optimizer.record_metric("metric1", &labels1);

        let mut labels2 = HashMap::new();
        labels2.insert("type".to_string(), "b".to_string());
        optimizer.record_metric("metric2", &labels2);

        assert_eq!(optimizer.metric_count(), 2);

        let all_cards = optimizer.get_all_cardinalities();
        assert_eq!(all_cards.len(), 2);
    }

    #[test]
    fn test_label_value_frequency_tracking() {
        let optimizer = CardinalityOptimizer::new(CardinalityConfig::default());

        // Record same label value multiple times
        for _ in 0..5 {
            let mut labels = HashMap::new();
            labels.insert("status".to_string(), "200".to_string());
            optimizer.record_metric("http_requests", &labels);
        }

        let cardinality = optimizer.get_metric_cardinality("http_requests").unwrap();
        let label_card = &cardinality.label_cardinalities[0];

        assert_eq!(label_card.total_occurrences, 5);
        assert_eq!(label_card.unique_values, 1);
    }
}
