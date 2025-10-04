//! Dataset Statistics Handler
//!
//! Provides statistical information about RDF datasets.
//! Based on Apache Jena Fuseki's statistics endpoint.
//!
//! GET /$/stats - Get server-wide statistics
//! GET /$/stats/:dataset - Get specific dataset statistics

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tracing::{debug, info};

use crate::store::Store;
use oxirs_core::Store as CoreStore;

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    /// Dataset name
    pub dataset_name: String,

    /// Total number of triples in default graph
    pub triples_in_default_graph: u64,

    /// Total number of triples across all named graphs
    pub triples_in_named_graphs: u64,

    /// Total number of quads (including default graph)
    pub total_quads: u64,

    /// Number of named graphs
    pub named_graph_count: u64,

    /// List of named graph URIs
    pub named_graphs: Vec<String>,

    /// Estimated storage size in bytes
    pub storage_size_bytes: Option<u64>,

    /// Timestamp when statistics were collected
    pub collected_at: SystemTime,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl DatasetStatistics {
    /// Create empty statistics
    pub fn new(dataset_name: String) -> Self {
        Self {
            dataset_name,
            triples_in_default_graph: 0,
            triples_in_named_graphs: 0,
            total_quads: 0,
            named_graph_count: 0,
            named_graphs: Vec::new(),
            storage_size_bytes: None,
            collected_at: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Get total triples across all graphs
    pub fn total_triples(&self) -> u64 {
        self.triples_in_default_graph + self.triples_in_named_graphs
    }

    /// Format storage size as human-readable string
    pub fn storage_size_human(&self) -> String {
        match self.storage_size_bytes {
            Some(bytes) => format_bytes(bytes),
            None => "Unknown".to_string(),
        }
    }
}

/// Server-wide statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatistics {
    /// Total number of datasets
    pub dataset_count: u64,

    /// Statistics for each dataset
    pub datasets: Vec<DatasetStatistics>,

    /// Server uptime in seconds
    pub uptime_seconds: Option<u64>,

    /// Total storage size across all datasets
    pub total_storage_bytes: Option<u64>,

    /// Server version
    pub version: String,

    /// Timestamp when statistics were collected
    pub collected_at: SystemTime,
}

impl ServerStatistics {
    /// Create empty server statistics
    pub fn new() -> Self {
        Self {
            dataset_count: 0,
            datasets: Vec::new(),
            uptime_seconds: None,
            total_storage_bytes: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
            collected_at: SystemTime::now(),
        }
    }

    /// Get total triples across all datasets
    pub fn total_triples(&self) -> u64 {
        self.datasets.iter().map(|d| d.total_triples()).sum()
    }

    /// Get total named graphs across all datasets
    pub fn total_named_graphs(&self) -> u64 {
        self.datasets.iter().map(|d| d.named_graph_count).sum()
    }
}

impl Default for ServerStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics collector for gathering dataset metrics
pub struct StatisticsCollector;

impl StatisticsCollector {
    /// Collect statistics for a dataset
    pub fn collect_dataset_stats(
        dataset_name: &str,
        store: &Store,
    ) -> Result<DatasetStatistics, StatsError> {
        let mut stats = DatasetStatistics::new(dataset_name.to_string());

        // Count triples in default graph
        stats.triples_in_default_graph = Self::count_default_graph_triples(store)?;

        // Get named graphs and count triples
        let named_graphs = Self::get_named_graphs(store)?;
        stats.named_graph_count = named_graphs.len() as u64;
        stats.named_graphs = named_graphs.clone();

        // Count triples in all named graphs
        stats.triples_in_named_graphs = Self::count_named_graph_triples(store, &named_graphs)?;

        // Total quads
        stats.total_quads = stats.triples_in_default_graph + stats.triples_in_named_graphs;

        // Storage size (if available)
        stats.storage_size_bytes = Self::estimate_storage_size(store).ok();

        // Add metadata
        stats
            .metadata
            .insert("store_type".to_string(), "OxiRS".to_string());

        stats.collected_at = SystemTime::now();

        Ok(stats)
    }

    /// Count triples in default graph
    fn count_default_graph_triples(store: &Store) -> Result<u64, StatsError> {
        use oxirs_core::model::GraphName;

        // Query all triples in default graph
        let quads = store
            .find_quads(None, None, None, Some(&GraphName::DefaultGraph))
            .map_err(|e| StatsError::Internal(format!("Failed to count default graph: {}", e)))?;

        Ok(quads.len() as u64)
    }

    /// Get list of named graph URIs
    fn get_named_graphs(store: &Store) -> Result<Vec<String>, StatsError> {
        // Get all distinct graph names
        let all_quads = store
            .find_quads(None, None, None, None)
            .map_err(|e| StatsError::Internal(format!("Failed to get graphs: {}", e)))?;

        let mut graphs = std::collections::HashSet::new();
        for quad in all_quads {
            if let oxirs_core::model::GraphName::NamedNode(node) = quad.graph_name() {
                graphs.insert(node.as_str().to_string());
            }
        }

        Ok(graphs.into_iter().collect())
    }

    /// Count triples in all named graphs
    fn count_named_graph_triples(store: &Store, graphs: &[String]) -> Result<u64, StatsError> {
        let mut total = 0u64;

        for graph_uri in graphs {
            // Parse graph URI to NamedNode
            let named_node = oxirs_core::model::NamedNode::new(graph_uri.clone())
                .map_err(|e| StatsError::Internal(format!("Invalid graph URI: {}", e)))?;

            let graph_name = oxirs_core::model::GraphName::NamedNode(named_node);

            let quads = store
                .find_quads(None, None, None, Some(&graph_name))
                .map_err(|e| StatsError::Internal(format!("Failed to count graph: {}", e)))?;

            total += quads.len() as u64;
        }

        Ok(total)
    }

    /// Estimate storage size
    fn estimate_storage_size(store: &Store) -> Result<u64, StatsError> {
        // Get all quads
        let all_quads = store
            .find_quads(None, None, None, None)
            .map_err(|e| StatsError::Internal(format!("Failed to estimate size: {}", e)))?;

        // Rough estimate: 200 bytes per quad on average
        let estimated_bytes = all_quads.len() as u64 * 200;

        Ok(estimated_bytes)
    }

    /// Collect server-wide statistics
    pub fn collect_server_stats(
        datasets: &HashMap<String, Arc<Store>>,
        server_start_time: Option<SystemTime>,
    ) -> Result<ServerStatistics, StatsError> {
        let mut stats = ServerStatistics::new();

        stats.dataset_count = datasets.len() as u64;

        // Collect stats for each dataset
        for (name, store) in datasets {
            match Self::collect_dataset_stats(name, store) {
                Ok(dataset_stats) => {
                    stats.datasets.push(dataset_stats);
                }
                Err(e) => {
                    debug!("Failed to collect stats for dataset '{}': {}", name, e);
                    // Continue with other datasets
                }
            }
        }

        // Calculate total storage
        stats.total_storage_bytes = Some(
            stats
                .datasets
                .iter()
                .filter_map(|d| d.storage_size_bytes)
                .sum(),
        );

        // Calculate uptime
        if let Some(start_time) = server_start_time {
            if let Ok(elapsed) = SystemTime::now().duration_since(start_time) {
                stats.uptime_seconds = Some(elapsed.as_secs());
            }
        }

        stats.collected_at = SystemTime::now();

        Ok(stats)
    }
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Statistics error types
#[derive(Debug, thiserror::Error)]
pub enum StatsError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl StatsError {
    fn status_code(&self) -> StatusCode {
        match self {
            StatsError::NotFound(_) => StatusCode::NOT_FOUND,
            StatsError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for StatsError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = self.to_string();

        (
            status,
            Json(serde_json::json!({
                "error": message,
                "status": status.as_u16(),
            })),
        )
            .into_response()
    }
}

/// Get server-wide statistics
///
/// GET /$/stats
pub async fn get_server_stats(State(store): State<Arc<Store>>) -> Result<Response, StatsError> {
    info!("Get server statistics request");

    // For now, treat as single default dataset
    let mut datasets = HashMap::new();
    datasets.insert("default".to_string(), store.clone());

    let stats = StatisticsCollector::collect_server_stats(&datasets, None)?;

    debug!("Server statistics: {} datasets", stats.dataset_count);

    Ok((StatusCode::OK, Json(stats)).into_response())
}

/// Get specific dataset statistics
///
/// GET /$/stats/:dataset
pub async fn get_dataset_stats(
    Path(dataset_name): Path<String>,
    State(store): State<Arc<Store>>,
) -> Result<Response, StatsError> {
    info!("Get dataset statistics request: {}", dataset_name);

    // For now, only support "default" dataset
    if dataset_name != "default" {
        return Err(StatsError::NotFound(format!(
            "Dataset '{}' not found",
            dataset_name
        )));
    }

    let stats = StatisticsCollector::collect_dataset_stats(&dataset_name, &store)?;

    debug!(
        "Dataset '{}' statistics: {} triples",
        dataset_name,
        stats.total_triples()
    );

    Ok((StatusCode::OK, Json(stats)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_stats_creation() {
        let stats = DatasetStatistics::new("test-dataset".to_string());

        assert_eq!(stats.dataset_name, "test-dataset");
        assert_eq!(stats.triples_in_default_graph, 0);
        assert_eq!(stats.total_triples(), 0);
    }

    #[test]
    fn test_server_stats_creation() {
        let stats = ServerStatistics::new();

        assert_eq!(stats.dataset_count, 0);
        assert_eq!(stats.datasets.len(), 0);
        assert_eq!(stats.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0.00 B");
        assert_eq!(format_bytes(500), "500.00 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_total_triples_calculation() {
        let mut stats = DatasetStatistics::new("test".to_string());
        stats.triples_in_default_graph = 100;
        stats.triples_in_named_graphs = 200;

        assert_eq!(stats.total_triples(), 300);
    }

    #[test]
    fn test_server_stats_aggregation() {
        let mut server_stats = ServerStatistics::new();

        let mut dataset1 = DatasetStatistics::new("dataset1".to_string());
        dataset1.triples_in_default_graph = 100;
        dataset1.triples_in_named_graphs = 50;

        let mut dataset2 = DatasetStatistics::new("dataset2".to_string());
        dataset2.triples_in_default_graph = 200;
        dataset2.triples_in_named_graphs = 100;

        server_stats.datasets.push(dataset1);
        server_stats.datasets.push(dataset2);
        server_stats.dataset_count = 2;

        assert_eq!(server_stats.total_triples(), 450);
    }

    #[test]
    fn test_storage_size_human() {
        let mut stats = DatasetStatistics::new("test".to_string());

        stats.storage_size_bytes = None;
        assert_eq!(stats.storage_size_human(), "Unknown");

        stats.storage_size_bytes = Some(1024);
        assert_eq!(stats.storage_size_human(), "1.00 KB");

        stats.storage_size_bytes = Some(1048576);
        assert_eq!(stats.storage_size_human(), "1.00 MB");
    }

    #[test]
    fn test_named_graph_count() {
        let mut stats = DatasetStatistics::new("test".to_string());
        stats.named_graphs = vec![
            "http://example.org/graph1".to_string(),
            "http://example.org/graph2".to_string(),
            "http://example.org/graph3".to_string(),
        ];
        stats.named_graph_count = 3;

        assert_eq!(stats.named_graph_count, 3);
        assert_eq!(stats.named_graphs.len(), 3);
    }

    #[test]
    fn test_metadata_storage() {
        let mut stats = DatasetStatistics::new("test".to_string());
        stats
            .metadata
            .insert("key1".to_string(), "value1".to_string());
        stats
            .metadata
            .insert("key2".to_string(), "value2".to_string());

        assert_eq!(stats.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(stats.metadata.get("key2"), Some(&"value2".to_string()));
    }
}
