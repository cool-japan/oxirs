//\! Report metadata and timing information

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Metadata about the validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Timestamp when the report was generated
    pub timestamp: u64,

    /// Version of the SHACL implementation
    pub shacl_version: String,

    /// Version of the validator
    pub validator_version: String,

    /// Validation duration
    pub validation_duration: Option<Duration>,

    /// Number of shapes validated
    pub shapes_count: Option<usize>,

    /// Number of data graph triples
    pub data_graph_size: Option<usize>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ReportMetadata {
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            shacl_version: "1.0".to_string(),
            validator_version: env!("CARGO_PKG_VERSION").to_string(),
            validation_duration: None,
            shapes_count: None,
            data_graph_size: None,
            metadata: HashMap::new(),
        }
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set validation duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.validation_duration = Some(duration);
        self
    }

    /// Set shapes count
    pub fn with_shapes_count(mut self, count: usize) -> Self {
        self.shapes_count = Some(count);
        self
    }

    /// Set data graph size
    pub fn with_data_graph_size(mut self, size: usize) -> Self {
        self.data_graph_size = Some(size);
        self
    }

    /// Get formatted timestamp
    pub fn formatted_timestamp(&self) -> String {
        if let Some(datetime) =
            SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(self.timestamp))
        {
            // In a real implementation, you'd use chrono or similar for proper formatting
            format!("{datetime:?}")
        } else {
            "Invalid timestamp".to_string()
        }
    }

    /// Get validation duration in human-readable format
    pub fn formatted_duration(&self) -> String {
        match self.validation_duration {
            Some(duration) => format!("{duration:.2?}"),
            None => "Unknown".to_string(),
        }
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> String {
        let mut summary = Vec::new();

        if let Some(duration) = self.validation_duration {
            summary.push(format!("Duration: {duration:.2?}"));
        }

        if let Some(shapes) = self.shapes_count {
            summary.push(format!("Shapes: {shapes}"));
        }

        if let Some(size) = self.data_graph_size {
            summary.push(format!("Triples: {size}"));
        }

        if let (Some(duration), Some(size)) = (self.validation_duration, self.data_graph_size) {
            let rate = size as f64 / duration.as_secs_f64();
            summary.push(format!("Rate: {rate:.0} triples/sec"));
        }

        summary.join(", ")
    }

    /// Check if performance data is available
    pub fn has_performance_data(&self) -> bool {
        self.validation_duration.is_some()
            || self.shapes_count.is_some()
            || self.data_graph_size.is_some()
    }

    /// Get validation efficiency metrics
    pub fn efficiency_metrics(&self) -> EfficiencyMetrics {
        EfficiencyMetrics {
            triples_per_second: self.calculate_triples_per_second(),
            shapes_per_second: self.calculate_shapes_per_second(),
            memory_efficiency: self.estimate_memory_efficiency(),
        }
    }

    fn calculate_triples_per_second(&self) -> Option<f64> {
        if let (Some(duration), Some(size)) = (self.validation_duration, self.data_graph_size) {
            if duration.as_secs_f64() > 0.0 {
                Some(size as f64 / duration.as_secs_f64())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn calculate_shapes_per_second(&self) -> Option<f64> {
        if let (Some(duration), Some(shapes)) = (self.validation_duration, self.shapes_count) {
            if duration.as_secs_f64() > 0.0 {
                Some(shapes as f64 / duration.as_secs_f64())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn estimate_memory_efficiency(&self) -> Option<f64> {
        // Simple heuristic based on data size and shapes count
        if let (Some(size), Some(shapes)) = (self.data_graph_size, self.shapes_count) {
            if shapes > 0 {
                Some(size as f64 / shapes as f64)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Efficiency metrics for validation performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Processing rate in triples per second
    pub triples_per_second: Option<f64>,
    /// Processing rate in shapes per second
    pub shapes_per_second: Option<f64>,
    /// Memory efficiency estimate (triples per shape)
    pub memory_efficiency: Option<f64>,
}

impl EfficiencyMetrics {
    /// Check if metrics are available
    pub fn has_metrics(&self) -> bool {
        self.triples_per_second.is_some()
            || self.shapes_per_second.is_some()
            || self.memory_efficiency.is_some()
    }

    /// Get a performance rating (0.0 to 1.0)
    pub fn performance_rating(&self) -> Option<f64> {
        // Simple rating based on throughput
        self.triples_per_second.map(|tps| (tps / 10000.0).min(1.0))
    }

    /// Format metrics for display
    pub fn format(&self) -> String {
        let mut parts = Vec::new();

        if let Some(tps) = self.triples_per_second {
            parts.push(format!("{tps:.0} triples/sec"));
        }

        if let Some(sps) = self.shapes_per_second {
            parts.push(format!("{sps:.1} shapes/sec"));
        }

        if let Some(eff) = self.memory_efficiency {
            parts.push(format!("{eff:.1} triples/shape"));
        }

        if parts.is_empty() {
            "No metrics available".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Timing helper for tracking validation performance
pub struct ValidationTimer {
    start_time: SystemTime,
    metadata: ReportMetadata,
}

impl ValidationTimer {
    /// Start timing validation
    pub fn start() -> Self {
        Self {
            start_time: SystemTime::now(),
            metadata: ReportMetadata::new(),
        }
    }

    /// Start with existing metadata
    pub fn start_with_metadata(metadata: ReportMetadata) -> Self {
        Self {
            start_time: SystemTime::now(),
            metadata,
        }
    }

    /// Stop timing and return metadata with duration
    pub fn stop(mut self) -> ReportMetadata {
        if let Ok(duration) = self.start_time.elapsed() {
            self.metadata.validation_duration = Some(duration);
        }
        self.metadata
    }

    /// Set shapes count
    pub fn with_shapes_count(mut self, count: usize) -> Self {
        self.metadata.shapes_count = Some(count);
        self
    }

    /// Set data graph size
    pub fn with_data_graph_size(mut self, size: usize) -> Self {
        self.metadata.data_graph_size = Some(size);
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.metadata.insert(key, value);
        self
    }

    /// Get elapsed time so far
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed().unwrap_or_default()
    }
}
