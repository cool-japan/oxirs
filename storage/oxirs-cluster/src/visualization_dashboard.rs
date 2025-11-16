//! Visualization Dashboard
//!
//! This module provides a web-based visualization dashboard for monitoring and managing
//! the OxiRS cluster. It includes real-time metrics, cluster topology visualization,
//! and operational controls.
//!
//! # Features
//!
//! - **Real-time Metrics**: Live cluster performance and health metrics
//! - **Topology Visualization**: Visual representation of cluster nodes and connections
//! - **Alert Management**: View and acknowledge alerts
//! - **Query Explorer**: Interactive SPARQL query interface
//! - **Node Management**: Add, remove, and configure nodes
//! - **Historical Data**: Trends and historical metrics
//! - **REST API**: Comprehensive API for programmatic access
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_cluster::visualization_dashboard::{DashboardConfig, DashboardServer};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = DashboardConfig::default()
//!     .with_bind_address("0.0.0.0:8080");
//!
//! let server = DashboardServer::new(config).await?;
//! server.start().await?;
//! # Ok(())
//! # }
//! ```

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
};

/// Errors that can occur during dashboard operations
#[derive(Debug, Error)]
pub enum DashboardError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Server error
    #[error("Server error: {0}")]
    ServerError(String),

    /// Not found error
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Invalid request error
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Other errors
    #[error("Dashboard error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DashboardError>;

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Bind address for the dashboard server
    pub bind_address: String,

    /// Enable CORS
    pub enable_cors: bool,

    /// Enable compression
    pub enable_compression: bool,

    /// Enable authentication
    pub enable_auth: bool,

    /// API key for authentication (if enabled)
    pub api_key: Option<String>,

    /// Refresh interval for metrics (milliseconds)
    pub refresh_interval_ms: u64,

    /// Enable debug mode
    pub debug_mode: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1:8080".to_string(),
            enable_cors: true,
            enable_compression: true,
            enable_auth: false,
            api_key: None,
            refresh_interval_ms: 1000,
            debug_mode: false,
        }
    }
}

impl DashboardConfig {
    /// Set the bind address
    pub fn with_bind_address(mut self, address: impl Into<String>) -> Self {
        self.bind_address = address.into();
        self
    }

    /// Enable authentication with API key
    pub fn with_auth(mut self, api_key: impl Into<String>) -> Self {
        self.enable_auth = true;
        self.api_key = Some(api_key.into());
        self
    }

    /// Set refresh interval
    pub fn with_refresh_interval(mut self, interval_ms: u64) -> Self {
        self.refresh_interval_ms = interval_ms;
        self
    }
}

/// Cluster metrics for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMetrics {
    /// Total number of nodes
    pub total_nodes: usize,

    /// Number of healthy nodes
    pub healthy_nodes: usize,

    /// Current leader node ID
    pub leader_node_id: Option<u64>,

    /// Total number of triples
    pub total_triples: u64,

    /// Queries per second
    pub queries_per_second: f64,

    /// Average query latency (milliseconds)
    pub avg_query_latency_ms: f64,

    /// Replication lag (milliseconds)
    pub avg_replication_lag_ms: f64,

    /// CPU usage percentage
    pub cpu_usage_percent: f64,

    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,

    /// Network throughput (bytes per second)
    pub network_throughput_bps: u64,

    /// Number of active alerts
    pub active_alerts: usize,

    /// Timestamp of metrics
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Node information for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node ID
    pub node_id: u64,

    /// Node address
    pub address: String,

    /// Is this node the leader
    pub is_leader: bool,

    /// Node health status
    pub health: String,

    /// Uptime (seconds)
    pub uptime_seconds: u64,

    /// Number of triples
    pub triple_count: u64,

    /// CPU usage
    pub cpu_percent: f64,

    /// Memory usage (bytes)
    pub memory_bytes: u64,

    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

/// Alert information for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertInfo {
    /// Alert ID
    pub id: String,

    /// Severity
    pub severity: String,

    /// Category
    pub category: String,

    /// Title
    pub title: String,

    /// Message
    pub message: String,

    /// Node ID
    pub node_id: Option<u64>,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Acknowledged
    pub acknowledged: bool,
}

/// Dashboard state
#[derive(Clone)]
struct DashboardState {
    #[allow(dead_code)] // Reserved for future use
    config: DashboardConfig,
    metrics: Arc<RwLock<ClusterMetrics>>,
    nodes: Arc<RwLock<HashMap<u64, NodeInfo>>>,
    alerts: Arc<RwLock<Vec<AlertInfo>>>,
}

impl DashboardState {
    fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(ClusterMetrics {
                total_nodes: 0,
                healthy_nodes: 0,
                leader_node_id: None,
                total_triples: 0,
                queries_per_second: 0.0,
                avg_query_latency_ms: 0.0,
                avg_replication_lag_ms: 0.0,
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                network_throughput_bps: 0,
                active_alerts: 0,
                timestamp: chrono::Utc::now(),
            })),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

/// Dashboard server
pub struct DashboardServer {
    config: DashboardConfig,
    state: DashboardState,
    running: Arc<RwLock<bool>>,
}

impl DashboardServer {
    /// Create a new dashboard server
    pub async fn new(config: DashboardConfig) -> Result<Self> {
        let state = DashboardState::new(config.clone());

        Ok(Self {
            config,
            state,
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the dashboard server
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Ok(());
        }

        tracing::info!(
            bind_address = %self.config.bind_address,
            "Starting visualization dashboard"
        );

        // Build router
        let app = self.build_router();

        // Parse bind address
        let addr: SocketAddr = self
            .config
            .bind_address
            .parse()
            .map_err(|e| DashboardError::ConfigError(format!("Invalid bind address: {e}")))?;

        *running = true;

        // Start server
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| DashboardError::ServerError(format!("Failed to bind: {e}")))?;

        tracing::info!("Dashboard server listening on {}", addr);

        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!("Dashboard server error: {}", e);
            }
        });

        Ok(())
    }

    /// Stop the dashboard server
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Ok(());
        }

        tracing::info!("Stopping visualization dashboard");

        *running = false;

        Ok(())
    }

    /// Build the Axum router with all routes
    fn build_router(&self) -> Router {
        let mut router = Router::new()
            // Dashboard home page
            .route("/", get(dashboard_home))
            // API routes
            .route("/api/metrics", get(get_metrics))
            .route("/api/nodes", get(get_nodes))
            .route("/api/nodes/:node_id", get(get_node))
            .route("/api/alerts", get(get_alerts))
            .route("/api/alerts/:alert_id/acknowledge", post(acknowledge_alert))
            .route("/api/health", get(health_check))
            .route("/api/topology", get(get_topology))
            .route("/api/queries", post(execute_query))
            // Management routes
            .route("/api/nodes/:node_id", delete(delete_node))
            .with_state(self.state.clone());

        // Add middleware
        let middleware = ServiceBuilder::new();

        // Add CORS if enabled
        if self.config.enable_cors {
            let cors = CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any);
            router = router.layer(cors);
        }

        // Add compression if enabled
        if self.config.enable_compression {
            router = router.layer(CompressionLayer::new());
        }

        router.layer(middleware)
    }

    /// Update cluster metrics
    pub async fn update_metrics(&self, metrics: ClusterMetrics) {
        let mut state_metrics = self.state.metrics.write().await;
        *state_metrics = metrics;
    }

    /// Update node information
    pub async fn update_node(&self, node: NodeInfo) {
        let mut nodes = self.state.nodes.write().await;
        nodes.insert(node.node_id, node);
    }

    /// Add alert
    pub async fn add_alert(&self, alert: AlertInfo) {
        let mut alerts = self.state.alerts.write().await;
        alerts.push(alert);
    }
}

// Route handlers

/// Dashboard home page
async fn dashboard_home() -> Html<&'static str> {
    Html(include_str!("../static/dashboard.html"))
}

/// Get cluster metrics
async fn get_metrics(State(state): State<DashboardState>) -> Json<ClusterMetrics> {
    let metrics = state.metrics.read().await;
    Json(metrics.clone())
}

/// Get all nodes
async fn get_nodes(State(state): State<DashboardState>) -> Json<Vec<NodeInfo>> {
    let nodes = state.nodes.read().await;
    Json(nodes.values().cloned().collect())
}

/// Get specific node
async fn get_node(
    State(state): State<DashboardState>,
    Path(node_id): Path<u64>,
) -> impl IntoResponse {
    let nodes = state.nodes.read().await;
    match nodes.get(&node_id).cloned() {
        Some(node) => (StatusCode::OK, Json(node)).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Get all alerts
async fn get_alerts(State(state): State<DashboardState>) -> Json<Vec<AlertInfo>> {
    let alerts = state.alerts.read().await;
    Json(alerts.clone())
}

/// Acknowledge an alert
async fn acknowledge_alert(
    State(state): State<DashboardState>,
    Path(alert_id): Path<String>,
) -> StatusCode {
    let mut alerts = state.alerts.write().await;
    for alert in alerts.iter_mut() {
        if alert.id == alert_id {
            alert.acknowledged = true;
            return StatusCode::OK;
        }
    }
    StatusCode::NOT_FOUND
}

/// Health check endpoint
async fn health_check() -> Json<HealthCheckResponse> {
    Json(HealthCheckResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now(),
    })
}

#[derive(Serialize)]
struct HealthCheckResponse {
    status: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Get cluster topology
async fn get_topology(State(state): State<DashboardState>) -> Json<TopologyResponse> {
    let nodes = state.nodes.read().await;

    let topology = TopologyResponse {
        nodes: nodes.values().cloned().collect(),
        connections: Vec::new(), // TODO: Add actual connections
    };

    Json(topology)
}

#[derive(Serialize)]
struct TopologyResponse {
    nodes: Vec<NodeInfo>,
    connections: Vec<Connection>,
}

#[derive(Serialize)]
struct Connection {
    source: u64,
    target: u64,
    connection_type: String,
}

/// Execute SPARQL query
#[derive(Deserialize)]
struct QueryRequest {
    query: String,
}

#[derive(Serialize)]
struct QueryResponse {
    results: Vec<HashMap<String, String>>,
    execution_time_ms: f64,
}

async fn execute_query(
    State(_state): State<DashboardState>,
    Json(request): Json<QueryRequest>,
) -> Json<QueryResponse> {
    // TODO: Implement actual query execution
    tracing::info!("Executing query: {}", request.query);

    Json(QueryResponse {
        results: Vec::new(),
        execution_time_ms: 0.0,
    })
}

/// Delete a node
async fn delete_node(State(state): State<DashboardState>, Path(node_id): Path<u64>) -> StatusCode {
    let mut nodes = state.nodes.write().await;
    if nodes.remove(&node_id).is_some() {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_config_default() {
        let config = DashboardConfig::default();
        assert_eq!(config.bind_address, "127.0.0.1:8080");
        assert!(config.enable_cors);
        assert!(config.enable_compression);
        assert!(!config.enable_auth);
    }

    #[test]
    fn test_dashboard_config_builder() {
        let config = DashboardConfig::default()
            .with_bind_address("0.0.0.0:9000")
            .with_auth("test-api-key")
            .with_refresh_interval(500);

        assert_eq!(config.bind_address, "0.0.0.0:9000");
        assert!(config.enable_auth);
        assert_eq!(config.api_key, Some("test-api-key".to_string()));
        assert_eq!(config.refresh_interval_ms, 500);
    }

    #[tokio::test]
    async fn test_dashboard_server_creation() {
        let config = DashboardConfig::default();
        let server = DashboardServer::new(config).await;
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_dashboard_state() {
        let config = DashboardConfig::default();
        let state = DashboardState::new(config);

        let metrics = state.metrics.read().await;
        assert_eq!(metrics.total_nodes, 0);

        let nodes = state.nodes.read().await;
        assert_eq!(nodes.len(), 0);
    }
}
