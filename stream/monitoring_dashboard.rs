//! # OxiRS Unified Monitoring Dashboard
//!
//! Comprehensive real-time monitoring dashboard for the entire OxiRS ecosystem,
//! providing visibility into stream performance, federation operations, and system health.
//!
//! ## Features
//! - **Real-time Metrics**: Live streaming of performance data
//! - **Multi-Component Monitoring**: Stream + Federation + Integration metrics
//! - **Interactive Dashboards**: Web-based interface with charts and graphs
//! - **Alerting System**: Configurable alerts and notifications
//! - **Historical Analytics**: Time-series data analysis and trends
//! - **Health Monitoring**: System health scores and diagnostics
//!
//! ## Dashboard Components
//! 1. **Stream Metrics**: Throughput, latency, error rates, backend performance
//! 2. **Federation Metrics**: Query performance, cache efficiency, service health
//! 3. **Integration Metrics**: Cross-system performance and data flow
//! 4. **System Metrics**: CPU, memory, network, disk usage
//! 5. **Alert Dashboard**: Active alerts, notifications, and incident tracking

use anyhow::{anyhow, Result};
use axum::{
    extract::{Query, State, WebSocketUpgrade, ws::WebSocket},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock, Semaphore};
use tokio::time::{interval, sleep};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Import OxiRS components for metrics collection
use oxirs_stream::{
    StreamManager, PerformanceMetrics as StreamMetrics, HealthStatus as StreamHealth,
    ConnectionPoolStats, BackendPerformance, EventSourcingStats, CQRSStats,
    SecurityMetrics, MultiRegionStats, TimeTravel,
};

/// Unified monitoring dashboard for the OxiRS ecosystem
pub struct OxiRSMonitoringDashboard {
    /// Shared application state
    state: Arc<DashboardState>,
    /// Web server configuration
    server_config: ServerConfig,
    /// Metrics collection configuration
    metrics_config: MetricsConfig,
}

/// Shared state across dashboard components
#[derive(Debug)]
pub struct DashboardState {
    /// Real-time metrics aggregator
    pub metrics_aggregator: Arc<RwLock<MetricsAggregator>>,
    /// Alert manager
    pub alert_manager: Arc<RwLock<AlertManager>>,
    /// Dashboard configuration
    pub config: Arc<RwLock<DashboardConfig>>,
    /// WebSocket broadcast channel for real-time updates
    pub broadcast_tx: broadcast::Sender<DashboardUpdate>,
    /// Historical data store
    pub historical_data: Arc<RwLock<HistoricalDataStore>>,
    /// Health monitor
    pub health_monitor: Arc<RwLock<SystemHealthMonitor>>,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub enable_cors: bool,
    pub static_files_dir: Option<String>,
    pub api_prefix: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            enable_cors: true,
            static_files_dir: Some("./dashboard/static".to_string()),
            api_prefix: "/api/v1".to_string(),
        }
    }
}

/// Metrics collection configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub collection_interval: Duration,
    pub retention_period: Duration,
    pub enable_detailed_metrics: bool,
    pub enable_profiling: bool,
    pub alert_thresholds: AlertThresholds,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(5),
            retention_period: Duration::from_secs(86400 * 7), // 7 days
            enable_detailed_metrics: true,
            enable_profiling: false,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub refresh_rate_ms: u64,
    pub max_data_points: usize,
    pub enable_real_time: bool,
    pub theme: DashboardTheme,
    pub enabled_panels: Vec<String>,
    pub custom_metrics: Vec<String>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: 1000,
            max_data_points: 1000,
            enable_real_time: true,
            theme: DashboardTheme::Dark,
            enabled_panels: vec![
                "stream_metrics".to_string(),
                "federation_metrics".to_string(),
                "system_health".to_string(),
                "alerts".to_string(),
            ],
            custom_metrics: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardTheme {
    Light,
    Dark,
    Auto,
}

/// Comprehensive metrics aggregator
#[derive(Debug, Default)]
pub struct MetricsAggregator {
    /// Stream component metrics
    pub stream_metrics: StreamComponentMetrics,
    /// Federation component metrics
    pub federation_metrics: FederationComponentMetrics,
    /// System-wide metrics
    pub system_metrics: SystemComponentMetrics,
    /// Integration metrics
    pub integration_metrics: IntegrationComponentMetrics,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StreamComponentMetrics {
    pub throughput_events_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub error_rate: f64,
    pub active_connections: u64,
    pub memory_usage_mb: f64,
    pub backend_performance: HashMap<String, BackendMetrics>,
    pub event_sourcing_stats: EventSourcingMetrics,
    pub cqrs_stats: CQRSMetrics,
    pub security_stats: SecurityMetrics,
    pub multi_region_stats: MultiRegionMetrics,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FederationComponentMetrics {
    pub query_rate_per_sec: f64,
    pub avg_query_time_ms: f64,
    pub cache_hit_rate: f64,
    pub service_availability: f64,
    pub active_services: u64,
    pub failed_services: u64,
    pub network_latency_ms: f64,
    pub federation_cache_size_mb: f64,
    pub auto_discovery_rate: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SystemComponentMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_throughput_mbps: f64,
    pub open_file_descriptors: u64,
    pub thread_count: u64,
    pub gc_pause_time_ms: f64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IntegrationComponentMetrics {
    pub stream_to_federation_latency_ms: f64,
    pub cross_component_error_rate: f64,
    pub data_consistency_score: f64,
    pub integration_throughput: f64,
    pub webhook_success_rate: f64,
    pub bridge_performance: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BackendMetrics {
    pub throughput: f64,
    pub latency_ms: f64,
    pub error_rate: f64,
    pub connection_count: u64,
    pub queue_depth: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EventSourcingMetrics {
    pub events_stored_per_sec: f64,
    pub replay_performance_ms: f64,
    pub storage_size_mb: f64,
    pub snapshot_frequency: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CQRSMetrics {
    pub command_rate_per_sec: f64,
    pub query_rate_per_sec: f64,
    pub command_latency_ms: f64,
    pub query_latency_ms: f64,
    pub read_model_update_latency_ms: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub auth_requests_per_sec: f64,
    pub auth_success_rate: f64,
    pub encryption_overhead_ms: f64,
    pub failed_auth_attempts: u64,
    pub active_sessions: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MultiRegionMetrics {
    pub replication_latency_ms: f64,
    pub sync_success_rate: f64,
    pub conflict_resolution_rate: f64,
    pub cross_region_throughput: f64,
    pub active_regions: u64,
}

/// Alert management system
#[derive(Debug, Default)]
pub struct AlertManager {
    pub active_alerts: Vec<Alert>,
    pub alert_history: Vec<Alert>,
    pub notification_channels: Vec<NotificationChannel>,
    pub alert_rules: Vec<AlertRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub level: AlertLevel,
    pub title: String,
    pub description: String,
    pub component: String,
    pub metric: String,
    pub current_value: f64,
    pub threshold: f64,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Fatal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub stream_max_latency_ms: f64,
    pub stream_min_throughput: f64,
    pub stream_max_error_rate: f64,
    pub federation_max_response_time_ms: f64,
    pub federation_min_cache_hit_rate: f64,
    pub system_max_cpu_percent: f64,
    pub system_max_memory_percent: f64,
    pub integration_max_latency_ms: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            stream_max_latency_ms: 100.0,
            stream_min_throughput: 1000.0,
            stream_max_error_rate: 0.05,
            federation_max_response_time_ms: 200.0,
            federation_min_cache_hit_rate: 0.8,
            system_max_cpu_percent: 80.0,
            system_max_memory_percent: 85.0,
            integration_max_latency_ms: 150.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub metric_path: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub duration_seconds: u64,
    pub level: AlertLevel,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub id: String,
    pub name: String,
    pub channel_type: NotificationChannelType,
    pub config: HashMap<String, String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    Slack,
    Webhook,
    PagerDuty,
    Discord,
}

/// Historical data storage
#[derive(Debug, Default)]
pub struct HistoricalDataStore {
    pub time_series_data: HashMap<String, Vec<TimeSeriesPoint>>,
    pub max_points_per_series: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

/// System health monitoring
#[derive(Debug, Default)]
pub struct SystemHealthMonitor {
    pub overall_health_score: f64,
    pub component_health: HashMap<String, ComponentHealth>,
    pub last_health_check: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub score: f64,
    pub status: HealthStatus,
    pub issues: Vec<String>,
    pub last_check: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Real-time dashboard updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardUpdate {
    MetricsUpdate(MetricsUpdatePayload),
    AlertUpdate(AlertUpdatePayload),
    HealthUpdate(HealthUpdatePayload),
    ConfigUpdate(DashboardConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsUpdatePayload {
    pub timestamp: DateTime<Utc>,
    pub stream_metrics: StreamComponentMetrics,
    pub federation_metrics: FederationComponentMetrics,
    pub system_metrics: SystemComponentMetrics,
    pub integration_metrics: IntegrationComponentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertUpdatePayload {
    pub new_alerts: Vec<Alert>,
    pub resolved_alerts: Vec<String>,
    pub active_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthUpdatePayload {
    pub overall_score: f64,
    pub component_health: HashMap<String, ComponentHealth>,
}

/// Query parameters for API endpoints
#[derive(Debug, Deserialize)]
pub struct TimeRangeQuery {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
    pub interval: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MetricsQuery {
    pub component: Option<String>,
    pub metric: Option<String>,
    pub aggregation: Option<String>,
}

impl OxiRSMonitoringDashboard {
    /// Create a new monitoring dashboard
    pub async fn new(
        server_config: ServerConfig,
        metrics_config: MetricsConfig,
    ) -> Result<Self> {
        info!("🚀 Initializing OxiRS Unified Monitoring Dashboard");

        let (broadcast_tx, _) = broadcast::channel(1000);

        let state = Arc::new(DashboardState {
            metrics_aggregator: Arc::new(RwLock::new(MetricsAggregator::default())),
            alert_manager: Arc::new(RwLock::new(AlertManager::default())),
            config: Arc::new(RwLock::new(DashboardConfig::default())),
            broadcast_tx: broadcast_tx.clone(),
            historical_data: Arc::new(RwLock::new(HistoricalDataStore {
                time_series_data: HashMap::new(),
                max_points_per_series: 10000,
            })),
            health_monitor: Arc::new(RwLock::new(SystemHealthMonitor::default())),
        });

        info!("✅ Dashboard state initialized");

        Ok(Self {
            state,
            server_config,
            metrics_config,
        })
    }

    /// Start the monitoring dashboard server
    pub async fn start(&self) -> Result<()> {
        info!("🎯 Starting OxiRS Monitoring Dashboard Server");

        // Start background tasks
        self.start_background_tasks().await?;

        // Create web server
        let app = self.create_web_server().await?;

        // Start server
        let addr = SocketAddr::from((
            self.server_config.host.parse::<std::net::IpAddr>()?,
            self.server_config.port,
        ));

        info!("🌐 Dashboard server starting on http://{}", addr);
        info!("📊 Access the dashboard at: http://{}/dashboard", addr);
        info!("🔧 API endpoints available at: http://{}{}", addr, self.server_config.api_prefix);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await?;

        Ok(())
    }

    /// Create the web server with all routes
    async fn create_web_server(&self) -> Result<Router> {
        let mut router = Router::new()
            // API routes
            .route(&format!("{}/metrics/current", self.server_config.api_prefix), get(Self::get_current_metrics))
            .route(&format!("{}/metrics/historical", self.server_config.api_prefix), get(Self::get_historical_metrics))
            .route(&format!("{}/alerts", self.server_config.api_prefix), get(Self::get_alerts))
            .route(&format!("{}/alerts", self.server_config.api_prefix), post(Self::create_alert))
            .route(&format!("{}/health", self.server_config.api_prefix), get(Self::get_health_status))
            .route(&format!("{}/config", self.server_config.api_prefix), get(Self::get_config))
            .route(&format!("{}/config", self.server_config.api_prefix), post(Self::update_config))
            .route(&format!("{}/ws", self.server_config.api_prefix), get(Self::websocket_handler))
            
            // Dashboard routes
            .route("/dashboard", get(Self::dashboard_index))
            .route("/dashboard/", get(Self::dashboard_index))
            
            // Health check
            .route("/health", get(|| async { "OK" }))
            
            // State sharing
            .with_state(self.state.clone());

        // Add CORS if enabled
        if self.server_config.enable_cors {
            router = router.layer(
                ServiceBuilder::new()
                    .layer(CorsLayer::permissive())
            );
        }

        // Add static file serving if configured
        if let Some(ref static_dir) = self.server_config.static_files_dir {
            router = router.nest_service("/static", ServeDir::new(static_dir));
        }

        Ok(router)
    }

    /// Start background monitoring tasks
    async fn start_background_tasks(&self) -> Result<()> {
        info!("🔄 Starting background monitoring tasks");

        // Start metrics collection task
        let state = self.state.clone();
        let collection_interval = self.metrics_config.collection_interval;
        tokio::spawn(async move {
            Self::metrics_collection_task(state, collection_interval).await;
        });

        // Start alert monitoring task
        let state = self.state.clone();
        let alert_thresholds = self.metrics_config.alert_thresholds.clone();
        tokio::spawn(async move {
            Self::alert_monitoring_task(state, alert_thresholds).await;
        });

        // Start health monitoring task
        let state = self.state.clone();
        tokio::spawn(async move {
            Self::health_monitoring_task(state).await;
        });

        // Start data cleanup task
        let state = self.state.clone();
        let retention_period = self.metrics_config.retention_period;
        tokio::spawn(async move {
            Self::data_cleanup_task(state, retention_period).await;
        });

        info!("✅ Background tasks started successfully");
        Ok(())
    }

    /// Metrics collection background task
    async fn metrics_collection_task(state: Arc<DashboardState>, interval: Duration) {
        let mut ticker = interval(interval);
        
        loop {
            ticker.tick().await;
            
            // Collect metrics from all components
            match Self::collect_all_metrics().await {
                Ok(metrics) => {
                    // Update metrics aggregator
                    {
                        let mut aggregator = state.metrics_aggregator.write().await;
                        *aggregator = metrics.clone();
                        aggregator.last_update = Utc::now();
                    }

                    // Store historical data
                    Self::store_historical_data(&state, &metrics).await;

                    // Broadcast update to connected clients
                    let update = DashboardUpdate::MetricsUpdate(MetricsUpdatePayload {
                        timestamp: Utc::now(),
                        stream_metrics: metrics.stream_metrics,
                        federation_metrics: metrics.federation_metrics,
                        system_metrics: metrics.system_metrics,
                        integration_metrics: metrics.integration_metrics,
                    });

                    let _ = state.broadcast_tx.send(update);
                }
                Err(e) => {
                    error!("Failed to collect metrics: {}", e);
                }
            }
        }
    }

    /// Alert monitoring background task
    async fn alert_monitoring_task(state: Arc<DashboardState>, thresholds: AlertThresholds) {
        let mut ticker = interval(Duration::from_secs(10));
        
        loop {
            ticker.tick().await;
            
            // Check for alert conditions
            let metrics = {
                let aggregator = state.metrics_aggregator.read().await;
                aggregator.clone()
            };

            let mut new_alerts = Vec::new();
            let mut resolved_alerts = Vec::new();

            // Check stream metrics alerts
            if metrics.stream_metrics.latency_p99_ms > thresholds.stream_max_latency_ms {
                new_alerts.push(Alert {
                    id: Uuid::new_v4().to_string(),
                    level: AlertLevel::Warning,
                    title: "High Stream Latency".to_string(),
                    description: format!("P99 latency ({:.2}ms) exceeds threshold ({:.2}ms)", 
                                       metrics.stream_metrics.latency_p99_ms, thresholds.stream_max_latency_ms),
                    component: "stream".to_string(),
                    metric: "latency_p99_ms".to_string(),
                    current_value: metrics.stream_metrics.latency_p99_ms,
                    threshold: thresholds.stream_max_latency_ms,
                    timestamp: Utc::now(),
                    resolved: false,
                    resolved_at: None,
                });
            }

            if metrics.stream_metrics.throughput_events_per_sec < thresholds.stream_min_throughput {
                new_alerts.push(Alert {
                    id: Uuid::new_v4().to_string(),
                    level: AlertLevel::Critical,
                    title: "Low Stream Throughput".to_string(),
                    description: format!("Throughput ({:.2} events/sec) below threshold ({:.2})", 
                                       metrics.stream_metrics.throughput_events_per_sec, thresholds.stream_min_throughput),
                    component: "stream".to_string(),
                    metric: "throughput_events_per_sec".to_string(),
                    current_value: metrics.stream_metrics.throughput_events_per_sec,
                    threshold: thresholds.stream_min_throughput,
                    timestamp: Utc::now(),
                    resolved: false,
                    resolved_at: None,
                });
            }

            // Check federation metrics alerts
            if metrics.federation_metrics.avg_query_time_ms > thresholds.federation_max_response_time_ms {
                new_alerts.push(Alert {
                    id: Uuid::new_v4().to_string(),
                    level: AlertLevel::Warning,
                    title: "High Federation Query Time".to_string(),
                    description: format!("Average query time ({:.2}ms) exceeds threshold ({:.2}ms)", 
                                       metrics.federation_metrics.avg_query_time_ms, thresholds.federation_max_response_time_ms),
                    component: "federation".to_string(),
                    metric: "avg_query_time_ms".to_string(),
                    current_value: metrics.federation_metrics.avg_query_time_ms,
                    threshold: thresholds.federation_max_response_time_ms,
                    timestamp: Utc::now(),
                    resolved: false,
                    resolved_at: None,
                });
            }

            // Update alert manager
            if !new_alerts.is_empty() || !resolved_alerts.is_empty() {
                {
                    let mut alert_manager = state.alert_manager.write().await;
                    alert_manager.active_alerts.extend(new_alerts.clone());
                    alert_manager.alert_history.extend(new_alerts.clone());
                    
                    // Remove resolved alerts
                    alert_manager.active_alerts.retain(|alert| !resolved_alerts.contains(&alert.id));
                }

                // Broadcast alert update
                let update = DashboardUpdate::AlertUpdate(AlertUpdatePayload {
                    new_alerts,
                    resolved_alerts,
                    active_count: {
                        let alert_manager = state.alert_manager.read().await;
                        alert_manager.active_alerts.len()
                    },
                });

                let _ = state.broadcast_tx.send(update);
            }
        }
    }

    /// Health monitoring background task
    async fn health_monitoring_task(state: Arc<DashboardState>) {
        let mut ticker = interval(Duration::from_secs(30));
        
        loop {
            ticker.tick().await;
            
            // Calculate overall health score
            let metrics = {
                let aggregator = state.metrics_aggregator.read().await;
                aggregator.clone()
            };

            let mut component_health = HashMap::new();
            
            // Stream component health
            let stream_health_score = Self::calculate_stream_health(&metrics.stream_metrics);
            component_health.insert("stream".to_string(), ComponentHealth {
                score: stream_health_score,
                status: if stream_health_score > 0.9 { HealthStatus::Healthy } 
                       else if stream_health_score > 0.7 { HealthStatus::Degraded } 
                       else { HealthStatus::Unhealthy },
                issues: Vec::new(),
                last_check: Utc::now(),
            });

            // Federation component health
            let federation_health_score = Self::calculate_federation_health(&metrics.federation_metrics);
            component_health.insert("federation".to_string(), ComponentHealth {
                score: federation_health_score,
                status: if federation_health_score > 0.9 { HealthStatus::Healthy } 
                       else if federation_health_score > 0.7 { HealthStatus::Degraded } 
                       else { HealthStatus::Unhealthy },
                issues: Vec::new(),
                last_check: Utc::now(),
            });

            // System component health
            let system_health_score = Self::calculate_system_health(&metrics.system_metrics);
            component_health.insert("system".to_string(), ComponentHealth {
                score: system_health_score,
                status: if system_health_score > 0.9 { HealthStatus::Healthy } 
                       else if system_health_score > 0.7 { HealthStatus::Degraded } 
                       else { HealthStatus::Unhealthy },
                issues: Vec::new(),
                last_check: Utc::now(),
            });

            let overall_health = (stream_health_score + federation_health_score + system_health_score) / 3.0;

            // Update health monitor
            {
                let mut health_monitor = state.health_monitor.write().await;
                health_monitor.overall_health_score = overall_health;
                health_monitor.component_health = component_health.clone();
                health_monitor.last_health_check = Utc::now();
            }

            // Broadcast health update
            let update = DashboardUpdate::HealthUpdate(HealthUpdatePayload {
                overall_score: overall_health,
                component_health,
            });

            let _ = state.broadcast_tx.send(update);
        }
    }

    /// Data cleanup background task
    async fn data_cleanup_task(state: Arc<DashboardState>, retention_period: Duration) {
        let mut ticker = interval(Duration::from_secs(3600)); // Run every hour
        
        loop {
            ticker.tick().await;
            
            let cutoff_time = Utc::now() - chrono::Duration::from_std(retention_period).unwrap();
            
            // Clean up historical data
            {
                let mut historical_data = state.historical_data.write().await;
                for series in historical_data.time_series_data.values_mut() {
                    series.retain(|point| point.timestamp > cutoff_time);
                }
            }

            // Clean up alert history
            {
                let mut alert_manager = state.alert_manager.write().await;
                alert_manager.alert_history.retain(|alert| alert.timestamp > cutoff_time);
            }

            debug!("Completed data cleanup for retention period: {:?}", retention_period);
        }
    }

    /// Collect metrics from all OxiRS components
    async fn collect_all_metrics() -> Result<MetricsAggregator> {
        // This would integrate with actual OxiRS components to collect real metrics
        // For now, we'll return simulated metrics
        
        Ok(MetricsAggregator {
            stream_metrics: StreamComponentMetrics {
                throughput_events_per_sec: 5000.0 + (rand::random::<f64>() * 1000.0),
                latency_p50_ms: 15.0 + (rand::random::<f64>() * 10.0),
                latency_p95_ms: 45.0 + (rand::random::<f64>() * 20.0),
                latency_p99_ms: 85.0 + (rand::random::<f64>() * 30.0),
                error_rate: 0.01 + (rand::random::<f64>() * 0.02),
                active_connections: 150 + (rand::random::<f64>() * 50.0) as u64,
                memory_usage_mb: 512.0 + (rand::random::<f64>() * 256.0),
                backend_performance: {
                    let mut backends = HashMap::new();
                    backends.insert("kafka".to_string(), BackendMetrics {
                        throughput: 8000.0,
                        latency_ms: 12.0,
                        error_rate: 0.005,
                        connection_count: 50,
                        queue_depth: 100,
                    });
                    backends.insert("nats".to_string(), BackendMetrics {
                        throughput: 12000.0,
                        latency_ms: 8.0,
                        error_rate: 0.002,
                        connection_count: 30,
                        queue_depth: 50,
                    });
                    backends
                },
                event_sourcing_stats: EventSourcingMetrics {
                    events_stored_per_sec: 3000.0,
                    replay_performance_ms: 150.0,
                    storage_size_mb: 2048.0,
                    snapshot_frequency: 0.1,
                },
                cqrs_stats: CQRSMetrics {
                    command_rate_per_sec: 800.0,
                    query_rate_per_sec: 2400.0,
                    command_latency_ms: 25.0,
                    query_latency_ms: 12.0,
                    read_model_update_latency_ms: 45.0,
                },
                security_stats: SecurityMetrics {
                    auth_requests_per_sec: 150.0,
                    auth_success_rate: 0.98,
                    encryption_overhead_ms: 2.5,
                    failed_auth_attempts: 5,
                    active_sessions: 320,
                },
                multi_region_stats: MultiRegionMetrics {
                    replication_latency_ms: 120.0,
                    sync_success_rate: 0.995,
                    conflict_resolution_rate: 0.02,
                    cross_region_throughput: 1500.0,
                    active_regions: 3,
                },
            },
            federation_metrics: FederationComponentMetrics {
                query_rate_per_sec: 200.0 + (rand::random::<f64>() * 100.0),
                avg_query_time_ms: 75.0 + (rand::random::<f64>() * 50.0),
                cache_hit_rate: 0.85 + (rand::random::<f64>() * 0.1),
                service_availability: 0.995 + (rand::random::<f64>() * 0.005),
                active_services: 25,
                failed_services: 1,
                network_latency_ms: 35.0 + (rand::random::<f64>() * 20.0),
                federation_cache_size_mb: 128.0 + (rand::random::<f64>() * 64.0),
                auto_discovery_rate: 5.0,
            },
            system_metrics: SystemComponentMetrics {
                cpu_usage_percent: 45.0 + (rand::random::<f64>() * 30.0),
                memory_usage_percent: 65.0 + (rand::random::<f64>() * 20.0),
                disk_usage_percent: 35.0 + (rand::random::<f64>() * 15.0),
                network_throughput_mbps: 850.0 + (rand::random::<f64>() * 200.0),
                open_file_descriptors: 1200 + (rand::random::<f64>() * 300.0) as u64,
                thread_count: 120 + (rand::random::<f64>() * 40.0) as u64,
                gc_pause_time_ms: 5.0 + (rand::random::<f64>() * 5.0),
                uptime_seconds: 86400, // 1 day
            },
            integration_metrics: IntegrationComponentMetrics {
                stream_to_federation_latency_ms: 25.0 + (rand::random::<f64>() * 15.0),
                cross_component_error_rate: 0.008 + (rand::random::<f64>() * 0.005),
                data_consistency_score: 0.99 + (rand::random::<f64>() * 0.01),
                integration_throughput: 1200.0 + (rand::random::<f64>() * 300.0),
                webhook_success_rate: 0.97 + (rand::random::<f64>() * 0.02),
                bridge_performance: 0.95 + (rand::random::<f64>() * 0.04),
            },
            last_update: Utc::now(),
        })
    }

    /// Store metrics data for historical analysis
    async fn store_historical_data(state: &Arc<DashboardState>, metrics: &MetricsAggregator) {
        let mut historical_data = state.historical_data.write().await;
        let timestamp = Utc::now();

        // Store stream metrics
        Self::add_time_series_point(&mut historical_data, "stream.throughput", 
                                   metrics.stream_metrics.throughput_events_per_sec, timestamp);
        Self::add_time_series_point(&mut historical_data, "stream.latency_p99", 
                                   metrics.stream_metrics.latency_p99_ms, timestamp);
        Self::add_time_series_point(&mut historical_data, "stream.error_rate", 
                                   metrics.stream_metrics.error_rate, timestamp);

        // Store federation metrics
        Self::add_time_series_point(&mut historical_data, "federation.query_rate", 
                                   metrics.federation_metrics.query_rate_per_sec, timestamp);
        Self::add_time_series_point(&mut historical_data, "federation.cache_hit_rate", 
                                   metrics.federation_metrics.cache_hit_rate, timestamp);

        // Store system metrics
        Self::add_time_series_point(&mut historical_data, "system.cpu_usage", 
                                   metrics.system_metrics.cpu_usage_percent, timestamp);
        Self::add_time_series_point(&mut historical_data, "system.memory_usage", 
                                   metrics.system_metrics.memory_usage_percent, timestamp);
    }

    fn add_time_series_point(
        historical_data: &mut HistoricalDataStore,
        metric_name: &str,
        value: f64,
        timestamp: DateTime<Utc>,
    ) {
        let series = historical_data.time_series_data
            .entry(metric_name.to_string())
            .or_insert_with(Vec::new);

        series.push(TimeSeriesPoint {
            timestamp,
            value,
            tags: HashMap::new(),
        });

        // Maintain max points limit
        if series.len() > historical_data.max_points_per_series {
            series.remove(0);
        }
    }

    /// Calculate component health scores
    fn calculate_stream_health(metrics: &StreamComponentMetrics) -> f64 {
        let latency_score = (200.0 - metrics.latency_p99_ms).max(0.0) / 200.0;
        let throughput_score = (metrics.throughput_events_per_sec / 10000.0).min(1.0);
        let error_score = (1.0 - metrics.error_rate * 20.0).max(0.0);
        
        (latency_score + throughput_score + error_score) / 3.0
    }

    fn calculate_federation_health(metrics: &FederationComponentMetrics) -> f64 {
        let response_time_score = (300.0 - metrics.avg_query_time_ms).max(0.0) / 300.0;
        let cache_score = metrics.cache_hit_rate;
        let availability_score = metrics.service_availability;
        
        (response_time_score + cache_score + availability_score) / 3.0
    }

    fn calculate_system_health(metrics: &SystemComponentMetrics) -> f64 {
        let cpu_score = (100.0 - metrics.cpu_usage_percent) / 100.0;
        let memory_score = (100.0 - metrics.memory_usage_percent) / 100.0;
        let disk_score = (100.0 - metrics.disk_usage_percent) / 100.0;
        
        (cpu_score + memory_score + disk_score) / 3.0
    }

    // API endpoint handlers
    async fn get_current_metrics(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
        let aggregator = state.metrics_aggregator.read().await;
        Json(aggregator.clone())
    }

    async fn get_historical_metrics(
        Query(query): Query<TimeRangeQuery>,
        State(state): State<Arc<DashboardState>>,
    ) -> impl IntoResponse {
        let historical_data = state.historical_data.read().await;
        
        let start_time = query.start.unwrap_or_else(|| Utc::now() - chrono::Duration::hours(1));
        let end_time = query.end.unwrap_or_else(|| Utc::now());

        let mut filtered_data = HashMap::new();
        for (metric_name, series) in &historical_data.time_series_data {
            let filtered_points: Vec<_> = series
                .iter()
                .filter(|point| point.timestamp >= start_time && point.timestamp <= end_time)
                .cloned()
                .collect();
            
            if !filtered_points.is_empty() {
                filtered_data.insert(metric_name.clone(), filtered_points);
            }
        }

        Json(filtered_data)
    }

    async fn get_alerts(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
        let alert_manager = state.alert_manager.read().await;
        Json(&alert_manager.active_alerts)
    }

    async fn create_alert(
        State(_state): State<Arc<DashboardState>>,
        Json(_alert): Json<Alert>,
    ) -> impl IntoResponse {
        // Implementation for creating custom alerts
        StatusCode::CREATED
    }

    async fn get_health_status(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
        let health_monitor = state.health_monitor.read().await;
        Json(health_monitor.clone())
    }

    async fn get_config(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
        let config = state.config.read().await;
        Json(config.clone())
    }

    async fn update_config(
        State(state): State<Arc<DashboardState>>,
        Json(new_config): Json<DashboardConfig>,
    ) -> impl IntoResponse {
        {
            let mut config = state.config.write().await;
            *config = new_config.clone();
        }

        // Broadcast config update
        let update = DashboardUpdate::ConfigUpdate(new_config);
        let _ = state.broadcast_tx.send(update);

        StatusCode::OK
    }

    async fn websocket_handler(
        ws: WebSocketUpgrade,
        State(state): State<Arc<DashboardState>>,
    ) -> Response {
        ws.on_upgrade(move |socket| Self::handle_websocket(socket, state))
    }

    async fn handle_websocket(socket: WebSocket, state: Arc<DashboardState>) {
        let mut rx = state.broadcast_tx.subscribe();
        let (mut sender, mut receiver) = socket.split();

        // Send initial data
        let initial_metrics = {
            let aggregator = state.metrics_aggregator.read().await;
            aggregator.clone()
        };

        let initial_update = DashboardUpdate::MetricsUpdate(MetricsUpdatePayload {
            timestamp: Utc::now(),
            stream_metrics: initial_metrics.stream_metrics,
            federation_metrics: initial_metrics.federation_metrics,
            system_metrics: initial_metrics.system_metrics,
            integration_metrics: initial_metrics.integration_metrics,
        });

        if let Ok(message) = serde_json::to_string(&initial_update) {
            let _ = sender.send(axum::extract::ws::Message::Text(message)).await;
        }

        // Handle real-time updates
        loop {
            tokio::select! {
                Ok(update) = rx.recv() => {
                    if let Ok(message) = serde_json::to_string(&update) {
                        if sender.send(axum::extract::ws::Message::Text(message)).await.is_err() {
                            break;
                        }
                    }
                }
                Some(msg) = receiver.next() => {
                    if let Ok(msg) = msg {
                        if matches!(msg, axum::extract::ws::Message::Close(_)) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }

    async fn dashboard_index() -> impl IntoResponse {
        Html(include_str!("dashboard_template.html"))
    }
}

/// Main function to start the monitoring dashboard
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("🚀 Starting OxiRS Unified Monitoring Dashboard");

    // Create dashboard configuration
    let server_config = ServerConfig {
        host: "0.0.0.0".to_string(),
        port: 8080,
        enable_cors: true,
        static_files_dir: Some("./dashboard/static".to_string()),
        api_prefix: "/api/v1".to_string(),
    };

    let metrics_config = MetricsConfig {
        collection_interval: Duration::from_secs(5),
        retention_period: Duration::from_secs(86400 * 7), // 7 days
        enable_detailed_metrics: true,
        enable_profiling: false,
        alert_thresholds: AlertThresholds::default(),
    };

    // Create and start dashboard
    let dashboard = OxiRSMonitoringDashboard::new(server_config, metrics_config).await?;
    dashboard.start().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let server_config = ServerConfig::default();
        let metrics_config = MetricsConfig::default();
        
        let dashboard = OxiRSMonitoringDashboard::new(server_config, metrics_config).await;
        assert!(dashboard.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let metrics = OxiRSMonitoringDashboard::collect_all_metrics().await;
        assert!(metrics.is_ok());
        
        let metrics = metrics.unwrap();
        assert!(metrics.stream_metrics.throughput_events_per_sec > 0.0);
        assert!(metrics.federation_metrics.query_rate_per_sec > 0.0);
        assert!(metrics.system_metrics.cpu_usage_percent >= 0.0);
    }

    #[tokio::test]
    async fn test_health_calculation() {
        let stream_metrics = StreamComponentMetrics {
            latency_p99_ms: 50.0,
            throughput_events_per_sec: 5000.0,
            error_rate: 0.01,
            ..Default::default()
        };

        let health_score = OxiRSMonitoringDashboard::calculate_stream_health(&stream_metrics);
        assert!(health_score > 0.0 && health_score <= 1.0);
    }
}