//! Custom Business Metrics Integration
//!
//! Provides a flexible framework for defining and tracking custom business metrics
//! that are specific to GraphQL operations and RDF data.
//!
//! # Features
//!
//! - Custom metric definitions (counters, gauges, histograms, summaries)
//! - Metric registration and management
//! - Automatic Prometheus export
//! - Metric aggregation and computation
//! - Context-aware metric recording
//! - Tag-based metric organization
//! - Metric validation and constraints

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Custom metric types
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    /// Monotonically increasing counter
    Counter,
    /// Value that can go up or down
    Gauge,
    /// Distribution of values with buckets
    Histogram { buckets: Vec<f64> },
    /// Statistical summary with quantiles
    Summary { quantiles: Vec<f64> },
}

/// Metric value
#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(f64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Summary {
        count: u64,
        sum: f64,
        quantiles: HashMap<u64, f64>, // quantile (0-100) -> value
    },
}

/// Metric metadata
#[derive(Debug, Clone)]
pub struct MetricMetadata {
    pub name: String,
    pub description: String,
    pub metric_type: MetricType,
    pub unit: Option<String>,
    pub tags: HashMap<String, String>,
}

/// A recorded metric data point
#[derive(Debug, Clone)]
pub struct MetricDataPoint {
    pub metadata: MetricMetadata,
    pub value: MetricValue,
    pub timestamp: SystemTime,
}

/// Metric aggregation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationStrategy {
    Sum,
    Average,
    Min,
    Max,
    Count,
    P50,
    P95,
    P99,
}

/// Computed metric definition
#[derive(Debug, Clone)]
pub struct ComputedMetric {
    pub name: String,
    pub description: String,
    pub source_metrics: Vec<String>,
    pub aggregation: AggregationStrategy,
    pub filter: Option<MetricFilter>,
}

/// Filter for metrics
#[derive(Debug, Clone)]
pub struct MetricFilter {
    pub tags: HashMap<String, String>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub time_range: Option<(SystemTime, SystemTime)>,
}

/// Custom business metrics registry
pub struct CustomMetricsRegistry {
    metrics: Arc<RwLock<HashMap<String, Vec<MetricDataPoint>>>>,
    metadata: Arc<RwLock<HashMap<String, MetricMetadata>>>,
    computed: Arc<RwLock<Vec<ComputedMetric>>>,
    retention_period: Duration,
    last_cleanup: Arc<Mutex<Instant>>,
}

impl CustomMetricsRegistry {
    /// Create a new metrics registry
    pub fn new(retention_period: Duration) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            computed: Arc::new(RwLock::new(Vec::new())),
            retention_period,
            last_cleanup: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Register a new metric
    pub fn register_metric(&self, metadata: MetricMetadata) -> Result<(), String> {
        let mut meta_map = self.metadata.write().unwrap();

        if meta_map.contains_key(&metadata.name) {
            return Err(format!("Metric '{}' already registered", metadata.name));
        }

        // Validate metric name
        if !Self::is_valid_metric_name(&metadata.name) {
            return Err(format!("Invalid metric name: {}", metadata.name));
        }

        meta_map.insert(metadata.name.clone(), metadata);
        Ok(())
    }

    /// Register a computed metric
    pub fn register_computed_metric(&self, computed: ComputedMetric) -> Result<(), String> {
        // Validate source metrics exist
        let meta_map = self.metadata.read().unwrap();
        for source in &computed.source_metrics {
            if !meta_map.contains_key(source) {
                return Err(format!("Source metric '{}' not found", source));
            }
        }
        drop(meta_map);

        let mut computed_metrics = self.computed.write().unwrap();
        computed_metrics.push(computed);
        Ok(())
    }

    /// Record a metric value
    pub fn record(
        &self,
        name: &str,
        value: MetricValue,
        tags: HashMap<String, String>,
    ) -> Result<(), String> {
        let meta_map = self.metadata.read().unwrap();
        let metadata = meta_map
            .get(name)
            .ok_or_else(|| format!("Metric '{}' not registered", name))?
            .clone();
        drop(meta_map);

        // Validate metric type matches value type
        self.validate_metric_value(&metadata.metric_type, &value)?;

        let data_point = MetricDataPoint {
            metadata: MetricMetadata { tags, ..metadata },
            value,
            timestamp: SystemTime::now(),
        };

        let mut metrics = self.metrics.write().unwrap();
        metrics
            .entry(name.to_string())
            .or_default()
            .push(data_point);

        Ok(())
    }

    /// Increment a counter
    pub fn increment_counter(
        &self,
        name: &str,
        value: f64,
        tags: HashMap<String, String>,
    ) -> Result<(), String> {
        self.record(name, MetricValue::Counter(value), tags)
    }

    /// Set a gauge value
    pub fn set_gauge(
        &self,
        name: &str,
        value: f64,
        tags: HashMap<String, String>,
    ) -> Result<(), String> {
        self.record(name, MetricValue::Gauge(value), tags)
    }

    /// Record a histogram observation
    pub fn observe_histogram(
        &self,
        name: &str,
        value: f64,
        tags: HashMap<String, String>,
    ) -> Result<(), String> {
        let meta_map = self.metadata.read().unwrap();
        let metadata = meta_map
            .get(name)
            .ok_or_else(|| format!("Metric '{}' not registered", name))?;

        let buckets = if let MetricType::Histogram { buckets } = &metadata.metric_type {
            buckets.clone()
        } else {
            return Err(format!("Metric '{}' is not a histogram", name));
        };
        drop(meta_map);

        // Determine which bucket this value falls into
        let mut bucket_values = vec![0.0; buckets.len()];
        for (i, &bucket) in buckets.iter().enumerate() {
            if value <= bucket {
                bucket_values[i] = 1.0;
                break;
            }
        }

        self.record(name, MetricValue::Histogram(bucket_values), tags)
    }

    /// Get all data points for a metric
    pub fn get_metric_data(
        &self,
        name: &str,
        filter: Option<&MetricFilter>,
    ) -> Vec<MetricDataPoint> {
        let metrics = self.metrics.read().unwrap();
        let data_points = metrics.get(name).cloned().unwrap_or_default();

        if let Some(filter) = filter {
            data_points
                .into_iter()
                .filter(|dp| self.matches_filter(dp, filter))
                .collect()
        } else {
            data_points
        }
    }

    /// Compute aggregated value for a metric
    pub fn compute_aggregation(
        &self,
        name: &str,
        strategy: AggregationStrategy,
        filter: Option<&MetricFilter>,
    ) -> Option<f64> {
        let data_points = self.get_metric_data(name, filter);

        if data_points.is_empty() {
            return None;
        }

        let values: Vec<f64> = data_points
            .iter()
            .filter_map(|dp| match &dp.value {
                MetricValue::Counter(v) | MetricValue::Gauge(v) => Some(*v),
                _ => None,
            })
            .collect();

        if values.is_empty() {
            return None;
        }

        match strategy {
            AggregationStrategy::Sum => Some(values.iter().sum()),
            AggregationStrategy::Average => Some(values.iter().sum::<f64>() / values.len() as f64),
            AggregationStrategy::Min => values.iter().cloned().fold(f64::INFINITY, f64::min).into(),
            AggregationStrategy::Max => values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .into(),
            AggregationStrategy::Count => Some(values.len() as f64),
            AggregationStrategy::P50 => Self::percentile(&values, 50.0),
            AggregationStrategy::P95 => Self::percentile(&values, 95.0),
            AggregationStrategy::P99 => Self::percentile(&values, 99.0),
        }
    }

    /// Evaluate computed metrics
    pub fn evaluate_computed_metric(&self, name: &str) -> Option<f64> {
        let computed_metrics = self.computed.read().unwrap();
        let computed = computed_metrics.iter().find(|c| c.name == name)?;

        // Collect values from source metrics
        let mut all_values = Vec::new();
        for source_name in &computed.source_metrics {
            let data_points = self.get_metric_data(source_name, computed.filter.as_ref());
            for dp in data_points {
                if let MetricValue::Counter(v) | MetricValue::Gauge(v) = dp.value {
                    all_values.push(v);
                }
            }
        }

        if all_values.is_empty() {
            return None;
        }

        match computed.aggregation {
            AggregationStrategy::Sum => Some(all_values.iter().sum()),
            AggregationStrategy::Average => {
                Some(all_values.iter().sum::<f64>() / all_values.len() as f64)
            }
            AggregationStrategy::Min => all_values
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min)
                .into(),
            AggregationStrategy::Max => all_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .into(),
            AggregationStrategy::Count => Some(all_values.len() as f64),
            AggregationStrategy::P50 => Self::percentile(&all_values, 50.0),
            AggregationStrategy::P95 => Self::percentile(&all_values, 95.0),
            AggregationStrategy::P99 => Self::percentile(&all_values, 99.0),
        }
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();
        let metrics_map = self.metrics.read().unwrap();
        let metadata_map = self.metadata.read().unwrap();

        for (name, metadata) in metadata_map.iter() {
            // Write metric help and type
            output.push_str(&format!("# HELP {} {}\n", name, metadata.description));
            output.push_str(&format!(
                "# TYPE {} {}\n",
                name,
                Self::prometheus_type(&metadata.metric_type)
            ));

            // Write metric values
            if let Some(data_points) = metrics_map.get(name) {
                for dp in data_points {
                    let tags_str = Self::format_prometheus_tags(&dp.metadata.tags);
                    match &dp.value {
                        MetricValue::Counter(v) | MetricValue::Gauge(v) => {
                            output.push_str(&format!("{}{} {}\n", name, tags_str, v));
                        }
                        MetricValue::Histogram(buckets) => {
                            if let MetricType::Histogram {
                                buckets: bucket_defs,
                            } = &metadata.metric_type
                            {
                                for (i, &bucket_value) in buckets.iter().enumerate() {
                                    if i < bucket_defs.len() {
                                        let mut bucket_tags = dp.metadata.tags.clone();
                                        bucket_tags
                                            .insert("le".to_string(), bucket_defs[i].to_string());
                                        let bucket_tags_str =
                                            Self::format_prometheus_tags(&bucket_tags);
                                        output.push_str(&format!(
                                            "{}_bucket{} {}\n",
                                            name, bucket_tags_str, bucket_value
                                        ));
                                    }
                                }
                            }
                        }
                        MetricValue::Summary {
                            count,
                            sum,
                            quantiles,
                        } => {
                            output.push_str(&format!("{}_count{} {}\n", name, tags_str, count));
                            output.push_str(&format!("{}_sum{} {}\n", name, tags_str, sum));
                            for (q, v) in quantiles {
                                let mut q_tags = dp.metadata.tags.clone();
                                q_tags.insert("quantile".to_string(), format!("0.{:02}", q));
                                let q_tags_str = Self::format_prometheus_tags(&q_tags);
                                output.push_str(&format!("{}{} {}\n", name, q_tags_str, v));
                            }
                        }
                    }
                }
            }
            output.push('\n');
        }

        output
    }

    /// Get all registered metric names
    pub fn list_metrics(&self) -> Vec<String> {
        self.metadata.read().unwrap().keys().cloned().collect()
    }

    /// Get metric metadata
    pub fn get_metadata(&self, name: &str) -> Option<MetricMetadata> {
        self.metadata.read().unwrap().get(name).cloned()
    }

    /// Clean up old metric data based on retention period
    pub fn cleanup_old_data(&self) {
        let mut last_cleanup = self.last_cleanup.lock().unwrap();
        if last_cleanup.elapsed() < Duration::from_secs(60) {
            return; // Only cleanup every minute
        }

        *last_cleanup = Instant::now();
        drop(last_cleanup);

        let cutoff = SystemTime::now() - self.retention_period;
        let mut metrics = self.metrics.write().unwrap();

        for data_points in metrics.values_mut() {
            data_points.retain(|dp| dp.timestamp >= cutoff);
        }
    }

    /// Get total number of data points
    pub fn total_data_points(&self) -> usize {
        self.metrics.read().unwrap().values().map(|v| v.len()).sum()
    }

    // Helper methods

    fn is_valid_metric_name(name: &str) -> bool {
        !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_')
    }

    fn validate_metric_value(
        &self,
        metric_type: &MetricType,
        value: &MetricValue,
    ) -> Result<(), String> {
        match (metric_type, value) {
            (MetricType::Counter, MetricValue::Counter(_)) => Ok(()),
            (MetricType::Gauge, MetricValue::Gauge(_)) => Ok(()),
            (MetricType::Histogram { .. }, MetricValue::Histogram(_)) => Ok(()),
            (MetricType::Summary { .. }, MetricValue::Summary { .. }) => Ok(()),
            _ => Err("Metric type and value type mismatch".to_string()),
        }
    }

    fn matches_filter(&self, data_point: &MetricDataPoint, filter: &MetricFilter) -> bool {
        // Check tags
        for (key, value) in &filter.tags {
            if data_point.metadata.tags.get(key) != Some(value) {
                return false;
            }
        }

        // Check value range
        if let Some(min) = filter.min_value {
            if let MetricValue::Counter(v) | MetricValue::Gauge(v) = data_point.value {
                if v < min {
                    return false;
                }
            }
        }

        if let Some(max) = filter.max_value {
            if let MetricValue::Counter(v) | MetricValue::Gauge(v) = data_point.value {
                if v > max {
                    return false;
                }
            }
        }

        // Check time range
        if let Some((start, end)) = filter.time_range {
            if data_point.timestamp < start || data_point.timestamp > end {
                return false;
            }
        }

        true
    }

    fn percentile(values: &[f64], p: f64) -> Option<f64> {
        if values.is_empty() {
            return None;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use floor to match the lower value for percentiles (nearest-rank method)
        let index = (p / 100.0 * (sorted.len() - 1) as f64).floor() as usize;
        Some(sorted[index])
    }

    fn prometheus_type(metric_type: &MetricType) -> &str {
        match metric_type {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram { .. } => "histogram",
            MetricType::Summary { .. } => "summary",
        }
    }

    fn format_prometheus_tags(tags: &HashMap<String, String>) -> String {
        if tags.is_empty() {
            return String::new();
        }

        let tags_str: Vec<String> = tags
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect();
        format!("{{{}}}", tags_str.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_counter_metric() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        let metadata = MetricMetadata {
            name: "api_requests_total".to_string(),
            description: "Total API requests".to_string(),
            metric_type: MetricType::Counter,
            unit: Some("requests".to_string()),
            tags: HashMap::new(),
        };

        assert!(registry.register_metric(metadata).is_ok());
        assert_eq!(registry.list_metrics(), vec!["api_requests_total"]);
    }

    #[test]
    fn test_register_duplicate_metric_fails() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        let metadata = MetricMetadata {
            name: "test_metric".to_string(),
            description: "Test".to_string(),
            metric_type: MetricType::Counter,
            unit: None,
            tags: HashMap::new(),
        };

        assert!(registry.register_metric(metadata.clone()).is_ok());
        assert!(registry.register_metric(metadata).is_err());
    }

    #[test]
    fn test_increment_counter() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "requests".to_string(),
                description: "Request count".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        let mut tags = HashMap::new();
        tags.insert("endpoint".to_string(), "/api/v1".to_string());

        assert!(registry.increment_counter("requests", 1.0, tags).is_ok());

        let data = registry.get_metric_data("requests", None);
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_set_gauge() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "active_connections".to_string(),
                description: "Active connections".to_string(),
                metric_type: MetricType::Gauge,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        assert!(registry
            .set_gauge("active_connections", 42.0, HashMap::new())
            .is_ok());

        let data = registry.get_metric_data("active_connections", None);
        assert_eq!(data.len(), 1);

        if let MetricValue::Gauge(v) = data[0].value {
            assert_eq!(v, 42.0);
        } else {
            panic!("Expected gauge value");
        }
    }

    #[test]
    fn test_observe_histogram() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "request_duration".to_string(),
                description: "Request duration".to_string(),
                metric_type: MetricType::Histogram {
                    buckets: vec![0.1, 0.5, 1.0, 5.0],
                },
                unit: Some("seconds".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        assert!(registry
            .observe_histogram("request_duration", 0.3, HashMap::new())
            .is_ok());

        let data = registry.get_metric_data("request_duration", None);
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_compute_sum_aggregation() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "sales".to_string(),
                description: "Sales amount".to_string(),
                metric_type: MetricType::Counter,
                unit: Some("dollars".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .increment_counter("sales", 100.0, HashMap::new())
            .unwrap();
        registry
            .increment_counter("sales", 200.0, HashMap::new())
            .unwrap();
        registry
            .increment_counter("sales", 150.0, HashMap::new())
            .unwrap();

        let sum = registry.compute_aggregation("sales", AggregationStrategy::Sum, None);
        assert_eq!(sum, Some(450.0));
    }

    #[test]
    fn test_compute_average_aggregation() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "response_time".to_string(),
                description: "Response time".to_string(),
                metric_type: MetricType::Gauge,
                unit: Some("ms".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .set_gauge("response_time", 100.0, HashMap::new())
            .unwrap();
        registry
            .set_gauge("response_time", 200.0, HashMap::new())
            .unwrap();
        registry
            .set_gauge("response_time", 150.0, HashMap::new())
            .unwrap();

        let avg = registry.compute_aggregation("response_time", AggregationStrategy::Average, None);
        assert_eq!(avg, Some(150.0));
    }

    #[test]
    fn test_compute_percentile_aggregation() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "latency".to_string(),
                description: "Latency".to_string(),
                metric_type: MetricType::Gauge,
                unit: Some("ms".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        for i in 1..=100 {
            registry
                .set_gauge("latency", i as f64, HashMap::new())
                .unwrap();
        }

        let p50 = registry.compute_aggregation("latency", AggregationStrategy::P50, None);
        assert!(p50.is_some());
        assert!((p50.unwrap() - 50.0).abs() < 2.0); // Allow small error

        let p95 = registry.compute_aggregation("latency", AggregationStrategy::P95, None);
        assert!(p95.is_some());
        assert!((p95.unwrap() - 95.0).abs() < 2.0);
    }

    #[test]
    fn test_metric_filtering_by_tags() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "requests".to_string(),
                description: "Requests".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        let mut tags1 = HashMap::new();
        tags1.insert("endpoint".to_string(), "/api/v1".to_string());
        registry.increment_counter("requests", 10.0, tags1).unwrap();

        let mut tags2 = HashMap::new();
        tags2.insert("endpoint".to_string(), "/api/v2".to_string());
        registry.increment_counter("requests", 20.0, tags2).unwrap();

        let mut filter_tags = HashMap::new();
        filter_tags.insert("endpoint".to_string(), "/api/v1".to_string());

        let filter = MetricFilter {
            tags: filter_tags,
            min_value: None,
            max_value: None,
            time_range: None,
        };

        let filtered = registry.get_metric_data("requests", Some(&filter));
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_metric_filtering_by_value_range() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "temperature".to_string(),
                description: "Temperature".to_string(),
                metric_type: MetricType::Gauge,
                unit: Some("celsius".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .set_gauge("temperature", 10.0, HashMap::new())
            .unwrap();
        registry
            .set_gauge("temperature", 25.0, HashMap::new())
            .unwrap();
        registry
            .set_gauge("temperature", 40.0, HashMap::new())
            .unwrap();

        let filter = MetricFilter {
            tags: HashMap::new(),
            min_value: Some(20.0),
            max_value: Some(35.0),
            time_range: None,
        };

        let filtered = registry.get_metric_data("temperature", Some(&filter));
        assert_eq!(filtered.len(), 1);

        if let MetricValue::Gauge(v) = filtered[0].value {
            assert_eq!(v, 25.0);
        }
    }

    #[test]
    fn test_register_computed_metric() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "sales_region_a".to_string(),
                description: "Sales in region A".to_string(),
                metric_type: MetricType::Counter,
                unit: Some("dollars".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .register_metric(MetricMetadata {
                name: "sales_region_b".to_string(),
                description: "Sales in region B".to_string(),
                metric_type: MetricType::Counter,
                unit: Some("dollars".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        let computed = ComputedMetric {
            name: "total_sales".to_string(),
            description: "Total sales across all regions".to_string(),
            source_metrics: vec!["sales_region_a".to_string(), "sales_region_b".to_string()],
            aggregation: AggregationStrategy::Sum,
            filter: None,
        };

        assert!(registry.register_computed_metric(computed).is_ok());
    }

    #[test]
    fn test_evaluate_computed_metric() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "metric_a".to_string(),
                description: "Metric A".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .register_metric(MetricMetadata {
                name: "metric_b".to_string(),
                description: "Metric B".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .increment_counter("metric_a", 100.0, HashMap::new())
            .unwrap();
        registry
            .increment_counter("metric_b", 200.0, HashMap::new())
            .unwrap();

        let computed = ComputedMetric {
            name: "total".to_string(),
            description: "Total".to_string(),
            source_metrics: vec!["metric_a".to_string(), "metric_b".to_string()],
            aggregation: AggregationStrategy::Sum,
            filter: None,
        };

        registry.register_computed_metric(computed).unwrap();

        let total = registry.evaluate_computed_metric("total");
        assert_eq!(total, Some(300.0));
    }

    #[test]
    fn test_prometheus_export_counter() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "http_requests_total".to_string(),
                description: "Total HTTP requests".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        let mut tags = HashMap::new();
        tags.insert("method".to_string(), "GET".to_string());
        registry
            .increment_counter("http_requests_total", 42.0, tags)
            .unwrap();

        let output = registry.export_prometheus();
        assert!(output.contains("# HELP http_requests_total Total HTTP requests"));
        assert!(output.contains("# TYPE http_requests_total counter"));
        assert!(output.contains("http_requests_total{method=\"GET\"} 42"));
    }

    #[test]
    fn test_prometheus_export_gauge() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "memory_usage_bytes".to_string(),
                description: "Memory usage in bytes".to_string(),
                metric_type: MetricType::Gauge,
                unit: Some("bytes".to_string()),
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .set_gauge("memory_usage_bytes", 1024.0, HashMap::new())
            .unwrap();

        let output = registry.export_prometheus();
        assert!(output.contains("# TYPE memory_usage_bytes gauge"));
        assert!(output.contains("memory_usage_bytes 1024"));
    }

    #[test]
    fn test_invalid_metric_name_rejected() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        let metadata = MetricMetadata {
            name: "invalid-metric-name!".to_string(),
            description: "Invalid".to_string(),
            metric_type: MetricType::Counter,
            unit: None,
            tags: HashMap::new(),
        };

        assert!(registry.register_metric(metadata).is_err());
    }

    #[test]
    fn test_metric_type_value_mismatch_rejected() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "counter_metric".to_string(),
                description: "Counter".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        // Try to record a gauge value for a counter metric
        let result = registry.record("counter_metric", MetricValue::Gauge(42.0), HashMap::new());

        assert!(result.is_err());
    }

    #[test]
    fn test_data_point_count() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "test".to_string(),
                description: "Test".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        assert_eq!(registry.total_data_points(), 0);

        registry
            .increment_counter("test", 1.0, HashMap::new())
            .unwrap();
        assert_eq!(registry.total_data_points(), 1);

        registry
            .increment_counter("test", 1.0, HashMap::new())
            .unwrap();
        assert_eq!(registry.total_data_points(), 2);
    }

    #[test]
    fn test_get_metadata() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        let metadata = MetricMetadata {
            name: "test_metric".to_string(),
            description: "Test metric description".to_string(),
            metric_type: MetricType::Counter,
            unit: Some("items".to_string()),
            tags: HashMap::new(),
        };

        registry.register_metric(metadata.clone()).unwrap();

        let retrieved = registry.get_metadata("test_metric");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().description, "Test metric description");
    }

    #[test]
    fn test_cleanup_retains_recent_data() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(1));

        registry
            .register_metric(MetricMetadata {
                name: "test".to_string(),
                description: "Test".to_string(),
                metric_type: MetricType::Counter,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        registry
            .increment_counter("test", 1.0, HashMap::new())
            .unwrap();

        registry.cleanup_old_data();

        assert_eq!(registry.total_data_points(), 1);
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let p50 = CustomMetricsRegistry::percentile(&values, 50.0);
        assert_eq!(p50, Some(5.0));

        let p100 = CustomMetricsRegistry::percentile(&values, 100.0);
        assert_eq!(p100, Some(10.0));

        let p0 = CustomMetricsRegistry::percentile(&values, 0.0);
        assert_eq!(p0, Some(1.0));
    }

    #[test]
    fn test_min_max_aggregation() {
        let registry = CustomMetricsRegistry::new(Duration::from_secs(3600));

        registry
            .register_metric(MetricMetadata {
                name: "values".to_string(),
                description: "Values".to_string(),
                metric_type: MetricType::Gauge,
                unit: None,
                tags: HashMap::new(),
            })
            .unwrap();

        registry.set_gauge("values", 10.0, HashMap::new()).unwrap();
        registry.set_gauge("values", 50.0, HashMap::new()).unwrap();
        registry.set_gauge("values", 30.0, HashMap::new()).unwrap();

        let min = registry.compute_aggregation("values", AggregationStrategy::Min, None);
        assert_eq!(min, Some(10.0));

        let max = registry.compute_aggregation("values", AggregationStrategy::Max, None);
        assert_eq!(max, Some(50.0));
    }
}
