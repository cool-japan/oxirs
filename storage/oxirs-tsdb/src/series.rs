use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single time-series data point
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp of the data point
    pub timestamp: DateTime<Utc>,

    /// Numerical value
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(timestamp: DateTime<Utc>, value: f64) -> Self {
        Self { timestamp, value }
    }

    /// Create data point with current timestamp
    pub fn now(value: f64) -> Self {
        Self {
            timestamp: Utc::now(),
            value,
        }
    }
}

/// Time series descriptor (identifies a unique series)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SeriesDescriptor {
    /// Unique series identifier
    pub series_id: u64,

    /// RDF subject IRI
    pub subject: String,

    /// RDF predicate IRI
    pub predicate: String,

    /// Optional named graph IRI
    pub graph: Option<String>,
}

/// Metadata about a time series
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesMetadata {
    /// Unique series identifier (matches SeriesDescriptor)
    pub series_id: u64,

    /// Unit of measurement (QUDT URI)
    pub unit: Option<String>,

    /// Sampling rate (Hz) - not Eq/Hash due to f64
    pub sampling_rate: Option<f64>,

    /// Data type hint
    pub data_type: Option<String>,

    /// Human-readable description
    pub description: Option<String>,
}

impl SeriesDescriptor {
    /// Create a new series descriptor
    pub fn new(series_id: u64, subject: String, predicate: String) -> Self {
        Self {
            series_id,
            subject,
            predicate,
            graph: None,
        }
    }

    /// Set the named graph
    pub fn with_graph(mut self, graph: String) -> Self {
        self.graph = Some(graph);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point_creation() {
        let point = DataPoint::now(22.5);
        assert_eq!(point.value, 22.5);
    }

    #[test]
    fn test_series_descriptor() {
        let desc = SeriesDescriptor::new(
            1,
            "http://example.org/sensor1".to_string(),
            "http://example.org/temperature".to_string(),
        );
        assert_eq!(desc.series_id, 1);
        assert_eq!(desc.graph, None);
    }
}
