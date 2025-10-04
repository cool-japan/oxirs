//! Observability Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub struct ObservabilityConfig {
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
    /// Dashboards configuration
    pub dashboards: DashboardConfig,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            monitoring: MonitoringConfig::default(),
            logging: LoggingConfig::default(),
            alerting: AlertingConfig::default(),
            dashboards: DashboardConfig::default(),
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Prometheus configuration
    pub prometheus: PrometheusConfig,
    /// Service monitors
    pub service_monitors: Vec<ServiceMonitor>,
    /// Pod monitors
    pub pod_monitors: Vec<PodMonitor>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            prometheus: PrometheusConfig::default(),
            service_monitors: vec![
                ServiceMonitor {
                    name: "oxirs-stream-monitor".to_string(),
                    namespace: "oxirs".to_string(),
                    selector: LabelSelector {
                        match_labels: BTreeMap::from([
                            ("app".to_string(), "oxirs-stream".to_string()),
                        ]),
                    },
                    endpoints: vec![
                        ServiceMonitorEndpoint {
                            port: "metrics".to_string(),
                            path: "/metrics".to_string(),
                            interval: Duration::from_secs(30),
                        },
                    ],
                },
            ],
            pod_monitors: vec![],
        }
    }
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    /// Enable Prometheus
    pub enabled: bool,
    /// Retention period
    pub retention: Duration,
    /// Storage size
    pub storage_size: String,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention: Duration::from_secs(30 * 24 * 3600), // 30 days
            storage_size: "50Gi".to_string(),
            resources: ResourceRequirements {
                cpu: "2".to_string(),
                memory: "4Gi".to_string(),
            },
        }
    }
}

/// Service monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMonitor {
    pub name: String,
    pub namespace: String,
    pub selector: LabelSelector,
    pub endpoints: Vec<ServiceMonitorEndpoint>,
}

/// Service monitor endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMonitorEndpoint {
    pub port: String,
    pub path: String,
    pub interval: Duration,
}

/// Pod monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodMonitor {
    pub name: String,
    pub namespace: String,
    pub selector: LabelSelector,
    pub pod_metrics_endpoints: Vec<PodMetricsEndpoint>,
}

/// Pod metrics endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodMetricsEndpoint {
    pub port: String,
    pub path: String,
    pub interval: Duration,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log aggregation
    pub aggregation: LogAggregationConfig,
    /// Log forwarding
    pub forwarding: LogForwardingConfig,
    /// Log retention
    pub retention: LogRetentionConfig,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            aggregation: LogAggregationConfig::default(),
            forwarding: LogForwardingConfig::default(),
            retention: LogRetentionConfig::default(),
        }
    }
}

/// Log aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogAggregationConfig {
    /// Enable log aggregation
    pub enabled: bool,
    /// Aggregation provider
    pub provider: LogAggregationProvider,
    /// Buffer size
    pub buffer_size: usize,
    /// Flush interval
    pub flush_interval: Duration,
}

impl Default for LogAggregationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: LogAggregationProvider::Fluentd,
            buffer_size: 64 * 1024, // 64KB
            flush_interval: Duration::from_secs(10),
        }
    }
}

/// Log aggregation providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogAggregationProvider {
    Fluentd,
    Fluent_Bit,
    Logstash,
    Vector,
}

/// Log forwarding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogForwardingConfig {
    /// Enable log forwarding
    pub enabled: bool,
    /// Forwarding destinations
    pub destinations: Vec<LogDestination>,
}

impl Default for LogForwardingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            destinations: vec![
                LogDestination {
                    name: "elasticsearch".to_string(),
                    destination_type: LogDestinationType::Elasticsearch,
                    endpoint: "https://elasticsearch.example.com:9200".to_string(),
                    index: "oxirs-logs".to_string(),
                },
            ],
        }
    }
}

/// Log destination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogDestination {
    pub name: String,
    pub destination_type: LogDestinationType,
    pub endpoint: String,
    pub index: String,
}

/// Log destination types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogDestinationType {
    Elasticsearch,
    Splunk,
    CloudWatch,
    BigQuery,
    S3,
}

/// Log retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRetentionConfig {
    /// Retention period
    pub retention_period: Duration,
    /// Compression enabled
    pub compression: bool,
    /// Archive to cold storage
    pub cold_storage: bool,
}

impl Default for LogRetentionConfig {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(90 * 24 * 3600), // 90 days
            compression: true,
            cold_storage: true,
        }
    }
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Alert manager configuration
    pub alert_manager: AlertManagerConfig,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            alert_manager: AlertManagerConfig::default(),
            rules: vec![
                AlertRule {
                    name: "high-cpu-usage".to_string(),
                    expression: "rate(container_cpu_usage_seconds_total[5m]) > 0.8".to_string(),
                    duration: Duration::from_secs(300),
                    severity: AlertSeverity::Warning,
                    summary: "High CPU usage detected".to_string(),
                    description: "CPU usage is above 80% for more than 5 minutes".to_string(),
                },
                AlertRule {
                    name: "high-memory-usage".to_string(),
                    expression: "container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9".to_string(),
                    duration: Duration::from_secs(300),
                    severity: AlertSeverity::Critical,
                    summary: "High memory usage detected".to_string(),
                    description: "Memory usage is above 90% for more than 5 minutes".to_string(),
                },
            ],
            notification_channels: vec![
                NotificationChannel {
                    name: "slack".to_string(),
                    channel_type: NotificationChannelType::Slack,
                    webhook_url: Some("https://hooks.slack.com/services/...".to_string()),
                    email_addresses: None,
                },
            ],
        }
    }
}

/// Alert manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertManagerConfig {
    /// Enable alert manager
    pub enabled: bool,
    /// Storage size
    pub storage_size: String,
    /// Retention period
    pub retention: Duration,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            storage_size: "10Gi".to_string(),
            retention: Duration::from_secs(30 * 24 * 3600), // 30 days
            resources: ResourceRequirements {
                cpu: "500m".to_string(),
                memory: "1Gi".to_string(),
            },
        }
    }
}

/// Alert rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub expression: String,
    pub duration: Duration,
    pub severity: AlertSeverity,
    pub summary: String,
    pub description: String,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    pub name: String,
    pub channel_type: NotificationChannelType,
    pub webhook_url: Option<String>,
    pub email_addresses: Option<Vec<String>>,
}

/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Slack,
    Email,
    PagerDuty,
    Webhook,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Grafana configuration
    pub grafana: GrafanaConfig,
    /// Dashboard definitions
    pub dashboards: Vec<Dashboard>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            grafana: GrafanaConfig::default(),
            dashboards: vec![
                Dashboard {
                    name: "oxirs-overview".to_string(),
                    title: "OxiRS Stream Overview".to_string(),
                    description: "Overview dashboard for OxiRS Stream".to_string(),
                    panels: vec![
                        DashboardPanel {
                            title: "Events per Second".to_string(),
                            panel_type: PanelType::Graph,
                            query: "rate(oxirs_stream_events_total[1m])".to_string(),
                        },
                        DashboardPanel {
                            title: "Processing Latency".to_string(),
                            panel_type: PanelType::Graph,
                            query: "histogram_quantile(0.95, oxirs_stream_latency_seconds)".to_string(),
                        },
                    ],
                },
            ],
        }
    }
}

/// Grafana configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrafanaConfig {
    /// Enable Grafana
    pub enabled: bool,
    /// Admin user
    pub admin_user: String,
    /// Admin password
    pub admin_password: String,
    /// Persistence configuration
    pub persistence: PersistenceConfig,
    /// Resource requirements
    pub resources: ResourceRequirements,
}

impl Default for GrafanaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            admin_user: "admin".to_string(),
            admin_password: "admin".to_string(), // Should be changed in production
            persistence: PersistenceConfig {
                enabled: true,
                size: "10Gi".to_string(),
                storage_class: "default".to_string(),
            },
            resources: ResourceRequirements {
                cpu: "500m".to_string(),
                memory: "1Gi".to_string(),
            },
        }
    }
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub enabled: bool,
    pub size: String,
    pub storage_class: String,
}

/// Dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub name: String,
    pub title: String,
    pub description: String,
    pub panels: Vec<DashboardPanel>,
}

/// Dashboard panel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardPanel {
    pub title: String,
    pub panel_type: PanelType,
    pub query: String,
}

/// Panel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelType {
    Graph,
    SingleStat,
    Table,
    Heatmap,
    Logs,
}
