//! Query Performance Heatmaps
//!
//! This module provides performance heatmap generation for visualizing query
//! performance patterns over time and across different dimensions.
//!
//! # Features
//!
//! - **Time-based Heatmaps**: Visualize performance over time (hourly, daily)
//! - **Operation Heatmaps**: Performance by operation type/name
//! - **Field-level Heatmaps**: Per-field resolution performance
//! - **Percentile Heatmaps**: P50/P95/P99 performance distribution
//! - **Multiple Formats**: JSON, HTML, ASCII, CSV export
//! - **Configurable Buckets**: Time windows and performance buckets
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::performance_heatmap::{HeatmapGenerator, HeatmapConfig};
//!
//! let mut generator = HeatmapGenerator::new(HeatmapConfig::default());
//! generator.record_query("GetUser", 150);
//!
//! let heatmap = generator.generate_time_heatmap()?;
//! let html = heatmap.to_html();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Heatmap configuration
#[derive(Debug, Clone)]
pub struct HeatmapConfig {
    /// Time bucket size (e.g., 1 hour)
    pub time_bucket_size: Duration,
    /// Performance buckets (ms thresholds)
    pub performance_buckets: Vec<u64>,
    /// Maximum data points to retain
    pub max_data_points: usize,
    /// Percentiles to track
    pub percentiles: Vec<u8>,
}

impl HeatmapConfig {
    /// Create new heatmap configuration
    pub fn new() -> Self {
        Self {
            time_bucket_size: Duration::from_secs(3600), // 1 hour
            performance_buckets: vec![10, 50, 100, 250, 500, 1000, 2500, 5000],
            max_data_points: 10000,
            percentiles: vec![50, 95, 99],
        }
    }

    /// Set time bucket size
    pub fn with_time_bucket_size(mut self, duration: Duration) -> Self {
        self.time_bucket_size = duration;
        self
    }

    /// Set performance buckets
    pub fn with_performance_buckets(mut self, buckets: Vec<u64>) -> Self {
        self.performance_buckets = buckets;
        self
    }

    /// Set max data points
    pub fn with_max_data_points(mut self, max: usize) -> Self {
        self.max_data_points = max;
        self
    }
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Query performance data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Operation name
    pub operation: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Field name (if field-level)
    pub field: Option<String>,
}

/// Heatmap cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapCell {
    /// Time bucket start
    pub time_bucket: u64,
    /// Performance bucket index
    pub performance_bucket: usize,
    /// Count of queries in this cell
    pub count: usize,
    /// Average duration in this cell
    pub avg_duration_ms: f64,
}

/// Heatmap data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heatmap {
    /// Heatmap cells
    pub cells: Vec<HeatmapCell>,
    /// Time bucket size (seconds)
    pub time_bucket_size: u64,
    /// Performance buckets (ms)
    pub performance_buckets: Vec<u64>,
    /// Start time
    pub start_time: SystemTime,
    /// End time
    pub end_time: SystemTime,
}

impl Heatmap {
    /// Get maximum count across all cells
    pub fn max_count(&self) -> usize {
        self.cells.iter().map(|c| c.count).max().unwrap_or(0)
    }

    /// Get minimum count across all cells
    pub fn min_count(&self) -> usize {
        self.cells.iter().map(|c| c.count).min().unwrap_or(0)
    }

    /// Convert to JSON
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Convert to HTML heatmap visualization
    pub fn to_html(&self) -> String {
        let mut html = String::from("<html><head><style>\n");
        html.push_str(".heatmap { display: grid; gap: 2px; }\n");
        html.push_str(".cell { padding: 10px; text-align: center; border: 1px solid #ddd; }\n");
        html.push_str(".hot { background-color: #ff0000; color: white; }\n");
        html.push_str(".warm { background-color: #ffaa00; }\n");
        html.push_str(".cool { background-color: #00ff00; }\n");
        html.push_str(".cold { background-color: #e0e0e0; }\n");
        html.push_str("</style></head><body>\n");
        html.push_str("<h1>Performance Heatmap</h1>\n");
        html.push_str("<div class='heatmap'>\n");

        let max_count = self.max_count() as f64;

        for cell in &self.cells {
            let intensity = if max_count > 0.0 {
                cell.count as f64 / max_count
            } else {
                0.0
            };

            let class = if intensity > 0.75 {
                "hot"
            } else if intensity > 0.5 {
                "warm"
            } else if intensity > 0.25 {
                "cool"
            } else {
                "cold"
            };

            html.push_str(&format!(
                "<div class='cell {}'>Count: {}<br>Avg: {:.1}ms</div>\n",
                class, cell.count, cell.avg_duration_ms
            ));
        }

        html.push_str("</div></body></html>");
        html
    }

    /// Convert to ASCII art
    pub fn to_ascii(&self) -> String {
        let max_count = self.max_count();
        let mut output = String::from("Performance Heatmap:\n");

        for cell in &self.cells {
            let intensity = if max_count > 0 {
                (cell.count * 10 / max_count).min(10)
            } else {
                0
            };

            let bar = "█".repeat(intensity);
            output.push_str(&format!(
                "Bucket {}: {} ({} queries, avg {:.1}ms)\n",
                cell.performance_bucket, bar, cell.count, cell.avg_duration_ms
            ));
        }

        output
    }

    /// Convert to CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("time_bucket,performance_bucket,count,avg_duration_ms\n");

        for cell in &self.cells {
            csv.push_str(&format!(
                "{},{},{},{:.2}\n",
                cell.time_bucket, cell.performance_bucket, cell.count, cell.avg_duration_ms
            ));
        }

        csv
    }
}

/// Heatmap generator
pub struct HeatmapGenerator {
    config: HeatmapConfig,
    data_points: Vec<PerformanceDataPoint>,
}

impl HeatmapGenerator {
    /// Create new heatmap generator
    pub fn new(config: HeatmapConfig) -> Self {
        Self {
            config,
            data_points: Vec::new(),
        }
    }

    /// Record a query performance data point
    pub fn record_query(&mut self, operation: impl Into<String>, duration_ms: u64) {
        self.record_query_at(operation, duration_ms, SystemTime::now());
    }

    /// Record query at specific time
    pub fn record_query_at(
        &mut self,
        operation: impl Into<String>,
        duration_ms: u64,
        timestamp: SystemTime,
    ) {
        self.data_points.push(PerformanceDataPoint {
            timestamp,
            operation: operation.into(),
            duration_ms,
            field: None,
        });

        // Trim if exceeds max
        if self.data_points.len() > self.config.max_data_points {
            let excess = self.data_points.len() - self.config.max_data_points;
            self.data_points.drain(0..excess);
        }
    }

    /// Record field-level performance
    pub fn record_field(
        &mut self,
        operation: impl Into<String>,
        field: impl Into<String>,
        duration_ms: u64,
    ) {
        self.data_points.push(PerformanceDataPoint {
            timestamp: SystemTime::now(),
            operation: operation.into(),
            duration_ms,
            field: Some(field.into()),
        });

        if self.data_points.len() > self.config.max_data_points {
            let excess = self.data_points.len() - self.config.max_data_points;
            self.data_points.drain(0..excess);
        }
    }

    /// Generate time-based heatmap
    pub fn generate_time_heatmap(&self) -> Result<Heatmap, String> {
        if self.data_points.is_empty() {
            return Err("No data points available".to_string());
        }

        let start_time = self.data_points.iter().map(|p| p.timestamp).min().unwrap();
        let end_time = self.data_points.iter().map(|p| p.timestamp).max().unwrap();

        let mut cells_map: HashMap<(u64, usize), Vec<u64>> = HashMap::new();

        for point in &self.data_points {
            let time_bucket = self.get_time_bucket(point.timestamp, start_time);
            let perf_bucket = self.get_performance_bucket(point.duration_ms);

            cells_map
                .entry((time_bucket, perf_bucket))
                .or_default()
                .push(point.duration_ms);
        }

        let mut cells: Vec<HeatmapCell> = cells_map
            .into_iter()
            .map(|((time_bucket, perf_bucket), durations)| {
                let count = durations.len();
                let avg_duration_ms = durations.iter().sum::<u64>() as f64 / count as f64;

                HeatmapCell {
                    time_bucket,
                    performance_bucket: perf_bucket,
                    count,
                    avg_duration_ms,
                }
            })
            .collect();

        cells.sort_by_key(|c| (c.time_bucket, c.performance_bucket));

        Ok(Heatmap {
            cells,
            time_bucket_size: self.config.time_bucket_size.as_secs(),
            performance_buckets: self.config.performance_buckets.clone(),
            start_time,
            end_time,
        })
    }

    /// Generate operation-based heatmap
    pub fn generate_operation_heatmap(&self) -> HashMap<String, Vec<HeatmapCell>> {
        let mut operation_maps: HashMap<String, HashMap<usize, Vec<u64>>> = HashMap::new();

        for point in &self.data_points {
            let perf_bucket = self.get_performance_bucket(point.duration_ms);

            operation_maps
                .entry(point.operation.clone())
                .or_default()
                .entry(perf_bucket)
                .or_default()
                .push(point.duration_ms);
        }

        operation_maps
            .into_iter()
            .map(|(operation, buckets)| {
                let cells: Vec<HeatmapCell> = buckets
                    .into_iter()
                    .map(|(perf_bucket, durations)| {
                        let count = durations.len();
                        let avg_duration_ms = durations.iter().sum::<u64>() as f64 / count as f64;

                        HeatmapCell {
                            time_bucket: 0,
                            performance_bucket: perf_bucket,
                            count,
                            avg_duration_ms,
                        }
                    })
                    .collect();

                (operation, cells)
            })
            .collect()
    }

    /// Get time bucket for timestamp
    fn get_time_bucket(&self, timestamp: SystemTime, start_time: SystemTime) -> u64 {
        let elapsed = timestamp
            .duration_since(start_time)
            .unwrap_or_default()
            .as_secs();
        elapsed / self.config.time_bucket_size.as_secs()
    }

    /// Get performance bucket index
    fn get_performance_bucket(&self, duration_ms: u64) -> usize {
        self.config
            .performance_buckets
            .iter()
            .position(|&threshold| duration_ms <= threshold)
            .unwrap_or(self.config.performance_buckets.len())
    }

    /// Get percentile statistics
    pub fn get_percentile_stats(&self) -> HashMap<u8, u64> {
        if self.data_points.is_empty() {
            return HashMap::new();
        }

        let mut durations: Vec<u64> = self.data_points.iter().map(|p| p.duration_ms).collect();
        durations.sort_unstable();

        let mut stats = HashMap::new();
        for &percentile in &self.config.percentiles {
            let index = (durations.len() * percentile as usize / 100).min(durations.len() - 1);
            stats.insert(percentile, durations[index]);
        }

        stats
    }

    /// Clear all data points
    pub fn clear(&mut self) {
        self.data_points.clear();
    }

    /// Get data point count
    pub fn data_point_count(&self) -> usize {
        self.data_points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::UNIX_EPOCH;

    #[test]
    fn test_heatmap_config_creation() {
        let config = HeatmapConfig::new()
            .with_time_bucket_size(Duration::from_secs(1800))
            .with_performance_buckets(vec![100, 500, 1000])
            .with_max_data_points(5000);

        assert_eq!(config.time_bucket_size.as_secs(), 1800);
        assert_eq!(config.performance_buckets.len(), 3);
        assert_eq!(config.max_data_points, 5000);
    }

    #[test]
    fn test_heatmap_config_default() {
        let config = HeatmapConfig::default();

        assert_eq!(config.time_bucket_size.as_secs(), 3600);
        assert!(!config.performance_buckets.is_empty());
        assert_eq!(config.max_data_points, 10000);
    }

    #[test]
    fn test_performance_data_point() {
        let point = PerformanceDataPoint {
            timestamp: SystemTime::now(),
            operation: "GetUser".to_string(),
            duration_ms: 150,
            field: Some("user".to_string()),
        };

        assert_eq!(point.operation, "GetUser");
        assert_eq!(point.duration_ms, 150);
        assert_eq!(point.field, Some("user".to_string()));
    }

    #[test]
    fn test_heatmap_generator_creation() {
        let config = HeatmapConfig::default();
        let generator = HeatmapGenerator::new(config);

        assert_eq!(generator.data_point_count(), 0);
    }

    #[test]
    fn test_record_query() {
        let mut generator = HeatmapGenerator::new(HeatmapConfig::default());

        generator.record_query("GetUser", 100);
        generator.record_query("GetPosts", 250);

        assert_eq!(generator.data_point_count(), 2);
    }

    #[test]
    fn test_record_field() {
        let mut generator = HeatmapGenerator::new(HeatmapConfig::default());

        generator.record_field("GetUser", "profile", 50);

        assert_eq!(generator.data_point_count(), 1);
        assert!(generator.data_points[0].field.is_some());
    }

    #[test]
    fn test_max_data_points() {
        let config = HeatmapConfig::default().with_max_data_points(5);
        let mut generator = HeatmapGenerator::new(config);

        for i in 0..10 {
            generator.record_query(format!("Query{}", i), 100);
        }

        assert_eq!(generator.data_point_count(), 5);
    }

    #[test]
    fn test_generate_time_heatmap() {
        let mut generator = HeatmapGenerator::new(HeatmapConfig::default());

        let now = SystemTime::now();
        generator.record_query_at("Query1", 50, now);
        generator.record_query_at("Query2", 150, now);
        generator.record_query_at("Query3", 500, now);

        let heatmap = generator.generate_time_heatmap().unwrap();

        assert!(!heatmap.cells.is_empty());
        assert_eq!(heatmap.start_time, now);
    }

    #[test]
    fn test_generate_time_heatmap_empty() {
        let generator = HeatmapGenerator::new(HeatmapConfig::default());

        let result = generator.generate_time_heatmap();

        assert!(result.is_err());
    }

    #[test]
    fn test_generate_operation_heatmap() {
        let mut generator = HeatmapGenerator::new(HeatmapConfig::default());

        generator.record_query("GetUser", 100);
        generator.record_query("GetUser", 150);
        generator.record_query("GetPosts", 500);

        let op_heatmap = generator.generate_operation_heatmap();

        assert!(op_heatmap.contains_key("GetUser"));
        assert!(op_heatmap.contains_key("GetPosts"));
    }

    #[test]
    fn test_performance_bucket() {
        let config = HeatmapConfig::default().with_performance_buckets(vec![100, 500, 1000]);
        let generator = HeatmapGenerator::new(config);

        assert_eq!(generator.get_performance_bucket(50), 0);
        assert_eq!(generator.get_performance_bucket(250), 1);
        assert_eq!(generator.get_performance_bucket(750), 2);
        assert_eq!(generator.get_performance_bucket(2000), 3);
    }

    #[test]
    fn test_get_percentile_stats() {
        let mut generator = HeatmapGenerator::new(HeatmapConfig::default());

        for i in 1..=100 {
            generator.record_query("Query", i * 10);
        }

        let stats = generator.get_percentile_stats();

        assert!(stats.contains_key(&50));
        assert!(stats.contains_key(&95));
        assert!(stats.contains_key(&99));

        let p50 = stats.get(&50).unwrap();
        assert!(*p50 >= 400 && *p50 <= 600);
    }

    #[test]
    fn test_percentile_stats_empty() {
        let generator = HeatmapGenerator::new(HeatmapConfig::default());

        let stats = generator.get_percentile_stats();

        assert!(stats.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut generator = HeatmapGenerator::new(HeatmapConfig::default());

        generator.record_query("Query1", 100);
        generator.record_query("Query2", 200);

        assert_eq!(generator.data_point_count(), 2);

        generator.clear();

        assert_eq!(generator.data_point_count(), 0);
    }

    #[test]
    fn test_heatmap_max_min_count() {
        let cells = vec![
            HeatmapCell {
                time_bucket: 0,
                performance_bucket: 0,
                count: 10,
                avg_duration_ms: 100.0,
            },
            HeatmapCell {
                time_bucket: 1,
                performance_bucket: 1,
                count: 50,
                avg_duration_ms: 200.0,
            },
            HeatmapCell {
                time_bucket: 2,
                performance_bucket: 2,
                count: 25,
                avg_duration_ms: 300.0,
            },
        ];

        let heatmap = Heatmap {
            cells,
            time_bucket_size: 3600,
            performance_buckets: vec![100, 500, 1000],
            start_time: SystemTime::now(),
            end_time: SystemTime::now(),
        };

        assert_eq!(heatmap.max_count(), 50);
        assert_eq!(heatmap.min_count(), 10);
    }

    #[test]
    fn test_heatmap_to_json() {
        let heatmap = Heatmap {
            cells: vec![HeatmapCell {
                time_bucket: 0,
                performance_bucket: 0,
                count: 10,
                avg_duration_ms: 100.0,
            }],
            time_bucket_size: 3600,
            performance_buckets: vec![100, 500],
            start_time: SystemTime::UNIX_EPOCH,
            end_time: SystemTime::UNIX_EPOCH,
        };

        let json = heatmap.to_json().unwrap();

        assert!(json.contains("cells"));
        assert!(json.contains("time_bucket_size"));
    }

    #[test]
    fn test_heatmap_to_html() {
        let heatmap = Heatmap {
            cells: vec![HeatmapCell {
                time_bucket: 0,
                performance_bucket: 0,
                count: 10,
                avg_duration_ms: 100.0,
            }],
            time_bucket_size: 3600,
            performance_buckets: vec![100, 500],
            start_time: SystemTime::UNIX_EPOCH,
            end_time: SystemTime::UNIX_EPOCH,
        };

        let html = heatmap.to_html();

        assert!(html.contains("<html>"));
        assert!(html.contains("heatmap"));
        assert!(html.contains("Count: 10"));
    }

    #[test]
    fn test_heatmap_to_ascii() {
        let heatmap = Heatmap {
            cells: vec![HeatmapCell {
                time_bucket: 0,
                performance_bucket: 0,
                count: 10,
                avg_duration_ms: 100.0,
            }],
            time_bucket_size: 3600,
            performance_buckets: vec![100, 500],
            start_time: SystemTime::UNIX_EPOCH,
            end_time: SystemTime::UNIX_EPOCH,
        };

        let ascii = heatmap.to_ascii();

        assert!(ascii.contains("Performance Heatmap"));
        assert!(ascii.contains("Bucket"));
    }

    #[test]
    fn test_heatmap_to_csv() {
        let heatmap = Heatmap {
            cells: vec![HeatmapCell {
                time_bucket: 0,
                performance_bucket: 0,
                count: 10,
                avg_duration_ms: 100.0,
            }],
            time_bucket_size: 3600,
            performance_buckets: vec![100, 500],
            start_time: SystemTime::UNIX_EPOCH,
            end_time: SystemTime::UNIX_EPOCH,
        };

        let csv = heatmap.to_csv();

        assert!(csv.contains("time_bucket,performance_bucket,count,avg_duration_ms"));
        assert!(csv.contains("0,0,10,100"));
    }

    #[test]
    fn test_time_bucket_calculation() {
        let config = HeatmapConfig::default().with_time_bucket_size(Duration::from_secs(60));
        let generator = HeatmapGenerator::new(config);

        let start = UNIX_EPOCH;
        let t1 = start + Duration::from_secs(30);
        let t2 = start + Duration::from_secs(90);
        let t3 = start + Duration::from_secs(150);

        assert_eq!(generator.get_time_bucket(t1, start), 0);
        assert_eq!(generator.get_time_bucket(t2, start), 1);
        assert_eq!(generator.get_time_bucket(t3, start), 2);
    }
}
