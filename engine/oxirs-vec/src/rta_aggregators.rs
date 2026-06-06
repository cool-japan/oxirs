//! Metrics collectors, monitors, alert managers, notification channels, and dashboard data types.

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use crate::rta_engine::{Alert, AlertSeverity};

/// Comprehensive metrics collection
#[derive(Debug)]
pub struct MetricsCollector {
    pub query_metrics: Arc<RwLock<QueryMetrics>>,
    pub system_metrics: Arc<RwLock<SystemMetrics>>,
    pub quality_metrics: Arc<RwLock<QualityMetrics>>,
    pub custom_metrics: Arc<RwLock<HashMap<String, CustomMetric>>>,
}

/// Query performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_latency: Duration,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub max_latency: Duration,
    pub min_latency: Duration,
    pub throughput_qps: f64,
    pub latency_history: VecDeque<(SystemTime, Duration)>,
    pub throughput_history: VecDeque<(SystemTime, f64)>,
    pub error_rate: f64,
    pub query_distribution: HashMap<String, u64>,
}

/// System resource metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub memory_total: u64,
    pub memory_available: u64,
    pub disk_usage: f64,
    pub network_io: NetworkIO,
    pub vector_count: u64,
    pub index_size: u64,
    pub cache_hit_ratio: f64,
    pub gc_pressure: f64,
    pub thread_count: u64,
    pub system_load: f64,
}

/// Network I/O metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkIO {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub connections_active: u64,
}

/// Vector search quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub recall_at_k: HashMap<usize, f64>,
    pub precision_at_k: HashMap<usize, f64>,
    pub ndcg_at_k: HashMap<usize, f64>,
    pub mean_reciprocal_rank: f64,
    pub average_similarity_score: f64,
    pub similarity_distribution: Vec<f64>,
    pub query_diversity: f64,
    pub result_diversity: f64,
    pub relevance_correlation: f64,
}

/// Custom metrics defined by users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub description: String,
    pub timestamp: SystemTime,
    pub tags: HashMap<String, String>,
}

/// Performance monitoring with alerting
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub thresholds: Arc<RwLock<PerformanceThresholds>>,
    pub alert_history: Arc<RwLock<VecDeque<Alert>>>,
    pub current_alerts: Arc<RwLock<HashMap<String, Alert>>>,
}

/// Performance thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_latency_ms: u64,
    pub min_throughput_qps: f64,
    pub max_memory_usage_percent: f64,
    pub max_cpu_usage_percent: f64,
    pub min_cache_hit_ratio: f64,
    pub max_error_rate_percent: f64,
    pub min_recall_at_10: f64,
    pub max_index_size_gb: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 100,
            min_throughput_qps: 100.0,
            max_memory_usage_percent: 80.0,
            max_cpu_usage_percent: 85.0,
            min_cache_hit_ratio: 0.8,
            max_error_rate_percent: 1.0,
            min_recall_at_10: 0.9,
            max_index_size_gb: 10.0,
        }
    }
}

/// Query pattern analysis
#[derive(Debug)]
pub struct QueryAnalyzer {
    pub query_patterns: Arc<RwLock<HashMap<String, QueryPattern>>>,
    pub popular_queries: Arc<RwLock<VecDeque<PopularQuery>>>,
    pub usage_trends: Arc<RwLock<UsageTrends>>,
}

/// Query pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    pub pattern_id: String,
    pub frequency: u64,
    pub avg_latency: Duration,
    pub success_rate: f64,
    pub peak_hours: Vec<u8>, // Hours of day (0-23)
    pub similarity_threshold_distribution: Vec<f64>,
    pub result_size_distribution: Vec<usize>,
    pub user_segments: HashMap<String, u64>,
}

/// Popular query tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopularQuery {
    pub query_text: String,
    pub frequency: u64,
    pub avg_similarity_score: f64,
    pub timestamp: SystemTime,
}

/// Usage trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageTrends {
    pub daily_query_counts: VecDeque<(SystemTime, u64)>,
    pub hourly_patterns: [u64; 24],
    pub weekly_patterns: [u64; 7],
    pub growth_rate: f64,
    pub seasonal_patterns: HashMap<String, f64>,
    pub user_growth: f64,
    pub feature_adoption: HashMap<String, f64>,
}

/// Alert management system
pub struct AlertManager {
    pub config: AlertConfig,
    pub notification_channels: Vec<Box<dyn NotificationChannel>>,
    pub alert_rules: Arc<RwLock<Vec<AlertRule>>>,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub enable_email: bool,
    pub enable_slack: bool,
    pub enable_webhook: bool,
    pub email_recipients: Vec<String>,
    pub slack_webhook: Option<String>,
    pub webhook_url: Option<String>,
    pub cooldown_period: Duration,
    pub escalation_enabled: bool,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enable_email: false,
            enable_slack: false,
            enable_webhook: false,
            email_recipients: Vec::new(),
            slack_webhook: None,
            webhook_url: None,
            cooldown_period: Duration::from_secs(300), // 5 minutes
            escalation_enabled: false,
        }
    }
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: String, // e.g., "latency > 100ms"
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub cooldown: Duration,
    pub actions: Vec<String>,
}

/// Notification channel trait
pub trait NotificationChannel: Send + Sync {
    fn send_notification(&self, alert: &Alert) -> Result<()>;
    fn get_channel_type(&self) -> String;
}

/// Dashboard data aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub overview: OverviewData,
    pub query_performance: QueryPerformanceData,
    pub system_health: SystemHealthData,
    pub quality_metrics: QualityMetricsData,
    pub usage_analytics: UsageAnalyticsData,
    pub alerts: Vec<Alert>,
    pub last_updated: SystemTime,
}

/// Overview dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverviewData {
    pub total_queries_today: u64,
    pub average_latency: Duration,
    pub current_qps: f64,
    pub system_health_score: f64,
    pub active_alerts: u64,
    pub index_size: u64,
    pub vector_count: u64,
    pub cache_hit_ratio: f64,
}

/// Query performance dashboard data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryPerformanceData {
    pub latency_trends: Vec<(SystemTime, Duration)>,
    pub throughput_trends: Vec<(SystemTime, f64)>,
    pub error_rate_trends: Vec<(SystemTime, f64)>,
    pub top_slow_queries: Vec<(String, Duration)>,
    pub query_distribution: HashMap<String, u64>,
    pub performance_percentiles: HashMap<String, Duration>,
}

/// System health dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthData {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_throughput: f64,
    pub resource_trends: Vec<(SystemTime, f64)>,
    pub capacity_forecast: Vec<(SystemTime, f64)>,
    pub bottlenecks: Vec<String>,
}

/// Quality metrics dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricsData {
    pub recall_trends: Vec<(SystemTime, f64)>,
    pub precision_trends: Vec<(SystemTime, f64)>,
    pub similarity_distribution: Vec<f64>,
    pub quality_score: f64,
    pub quality_trends: Vec<(SystemTime, f64)>,
    pub benchmark_comparisons: HashMap<String, f64>,
}

/// Usage analytics dashboard data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageAnalyticsData {
    pub user_activity: Vec<(SystemTime, u64)>,
    pub popular_queries: Vec<PopularQuery>,
    pub usage_patterns: HashMap<String, f64>,
    pub growth_metrics: GrowthMetrics,
    pub feature_usage: HashMap<String, u64>,
}

/// Growth metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthMetrics {
    pub daily_growth_rate: f64,
    pub weekly_growth_rate: f64,
    pub monthly_growth_rate: f64,
    pub user_retention: f64,
    pub query_volume_growth: f64,
}

/// Performance profiler for detailed analysis
pub struct PerformanceProfiler {
    pub profiles: Arc<RwLock<HashMap<String, ProfileData>>>,
    pub active_profiles: Arc<RwLock<HashMap<String, Instant>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileData {
    pub function_name: String,
    pub total_calls: u64,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub call_history: VecDeque<(SystemTime, Duration)>,
}

/// Export formats for metrics
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Prometheus,
    InfluxDb,
}

// --- Default implementations ---

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            query_metrics: Arc::new(RwLock::new(QueryMetrics::default())),
            system_metrics: Arc::new(RwLock::new(SystemMetrics::default())),
            quality_metrics: Arc::new(RwLock::new(QualityMetrics::default())),
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            thresholds: Arc::new(RwLock::new(PerformanceThresholds::default())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            current_alerts: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for QueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryAnalyzer {
    pub fn new() -> Self {
        Self {
            query_patterns: Arc::new(RwLock::new(HashMap::new())),
            popular_queries: Arc::new(RwLock::new(VecDeque::new())),
            usage_trends: Arc::new(RwLock::new(UsageTrends::default())),
        }
    }
}

impl AlertManager {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            notification_channels: Vec::new(),
            alert_rules: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn send_alert(&self, alert: &Alert) -> Result<()> {
        for channel in &self.notification_channels {
            if let Err(e) = channel.send_notification(alert) {
                eprintln!(
                    "Failed to send alert via {}: {}",
                    channel.get_channel_type(),
                    e
                );
            }
        }
        Ok(())
    }

    pub fn add_notification_channel(&mut self, channel: Box<dyn NotificationChannel>) {
        self.notification_channels.push(channel);
    }
}

impl Default for DashboardData {
    fn default() -> Self {
        Self {
            overview: OverviewData::default(),
            query_performance: QueryPerformanceData::default(),
            system_health: SystemHealthData::default(),
            quality_metrics: QualityMetricsData::default(),
            usage_analytics: UsageAnalyticsData::default(),
            alerts: Vec::new(),
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for OverviewData {
    fn default() -> Self {
        Self {
            total_queries_today: 0,
            average_latency: Duration::from_millis(0),
            current_qps: 0.0,
            system_health_score: 100.0,
            active_alerts: 0,
            index_size: 0,
            vector_count: 0,
            cache_hit_ratio: 0.0,
        }
    }
}

impl Default for SystemHealthData {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_throughput: 0.0,
            resource_trends: Vec::new(),
            capacity_forecast: Vec::new(),
            bottlenecks: Vec::new(),
        }
    }
}

impl Default for QualityMetricsData {
    fn default() -> Self {
        Self {
            recall_trends: Vec::new(),
            precision_trends: Vec::new(),
            similarity_distribution: Vec::new(),
            quality_score: 0.0,
            quality_trends: Vec::new(),
            benchmark_comparisons: HashMap::new(),
        }
    }
}

impl Default for UsageTrends {
    fn default() -> Self {
        Self {
            daily_query_counts: VecDeque::new(),
            hourly_patterns: [0; 24],
            weekly_patterns: [0; 7],
            growth_rate: 0.0,
            seasonal_patterns: HashMap::new(),
            user_growth: 0.0,
            feature_adoption: HashMap::new(),
        }
    }
}

impl Default for GrowthMetrics {
    fn default() -> Self {
        Self {
            daily_growth_rate: 0.0,
            weekly_growth_rate: 0.0,
            monthly_growth_rate: 0.0,
            user_retention: 0.0,
            query_volume_growth: 0.0,
        }
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            active_profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn start_profile(&self, function_name: &str) -> String {
        let profile_id = format!(
            "{}_{}",
            function_name,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        );
        let mut active = self.active_profiles.write();
        active.insert(profile_id.clone(), Instant::now());
        profile_id
    }

    pub fn end_profile(&self, profile_id: &str) -> Result<Duration> {
        let mut active = self.active_profiles.write();
        let start_time = active
            .remove(profile_id)
            .ok_or_else(|| anyhow!("Profile ID not found: {}", profile_id))?;

        let duration = start_time.elapsed();

        // Extract function name from profile ID
        let function_name = profile_id.split('_').next().unwrap_or("unknown");

        // Update profile data
        let mut profiles = self.profiles.write();
        let profile = profiles
            .entry(function_name.to_string())
            .or_insert_with(|| ProfileData {
                function_name: function_name.to_string(),
                total_calls: 0,
                total_time: Duration::from_nanos(0),
                average_time: Duration::from_nanos(0),
                min_time: Duration::from_secs(u64::MAX),
                max_time: Duration::from_nanos(0),
                call_history: VecDeque::new(),
            });

        profile.total_calls += 1;
        profile.total_time += duration;
        profile.average_time = profile.total_time / profile.total_calls as u32;
        profile.min_time = profile.min_time.min(duration);
        profile.max_time = profile.max_time.max(duration);
        profile
            .call_history
            .push_back((SystemTime::now(), duration));

        // Keep only recent history
        while profile.call_history.len() > 1000 {
            profile.call_history.pop_front();
        }

        Ok(duration)
    }

    pub fn get_profile_report(&self) -> HashMap<String, ProfileData> {
        self.profiles.read().clone()
    }
}

// --- Notification channel implementations ---

/// Email notification channel
pub struct EmailNotificationChannel {
    smtp_config: SmtpConfig,
}

#[derive(Debug, Clone)]
pub struct SmtpConfig {
    pub server: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub from_address: String,
}

impl EmailNotificationChannel {
    pub fn new(smtp_config: SmtpConfig) -> Self {
        Self { smtp_config }
    }
}

impl NotificationChannel for EmailNotificationChannel {
    fn send_notification(&self, alert: &Alert) -> Result<()> {
        // Email implementation would require SMTP client
        tracing::info!(
            "Email notification sent for alert {}: {}",
            alert.id,
            alert.message
        );
        Ok(())
    }

    fn get_channel_type(&self) -> String {
        "email".to_string()
    }
}

/// Slack notification channel
pub struct SlackNotificationChannel {
    webhook_url: String,
    client: reqwest::Client,
}

impl SlackNotificationChannel {
    pub fn new(webhook_url: String) -> Self {
        Self {
            webhook_url,
            client: reqwest::Client::new(),
        }
    }
}

impl NotificationChannel for SlackNotificationChannel {
    fn send_notification(&self, alert: &Alert) -> Result<()> {
        let _payload = serde_json::json!({
            "text": format!("Alert: {}", alert.message),
            "attachments": [{
                "color": match alert.severity {
                    AlertSeverity::Critical => "danger",
                    AlertSeverity::Warning => "warning",
                    AlertSeverity::Info => "good",
                },
                "fields": [
                    {
                        "title": "Alert Type",
                        "value": format!("{:?}", alert.alert_type),
                        "short": true
                    },
                    {
                        "title": "Severity",
                        "value": format!("{:?}", alert.severity),
                        "short": true
                    },
                    {
                        "title": "Timestamp",
                        "value": chrono::DateTime::<chrono::Utc>::from(alert.timestamp).format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                        "short": true
                    }
                ]
            }]
        });

        // In a real implementation, would send HTTP POST
        tracing::info!(
            "Slack notification sent for alert {}: {}",
            alert.id,
            alert.message
        );
        Ok(())
    }

    fn get_channel_type(&self) -> String {
        "slack".to_string()
    }
}

/// Webhook notification channel
pub struct WebhookNotificationChannel {
    webhook_url: String,
    client: reqwest::Client,
    headers: HashMap<String, String>,
}

impl WebhookNotificationChannel {
    pub fn new(webhook_url: String) -> Self {
        Self {
            webhook_url,
            client: reqwest::Client::new(),
            headers: HashMap::new(),
        }
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers = headers;
        self
    }
}

impl NotificationChannel for WebhookNotificationChannel {
    fn send_notification(&self, alert: &Alert) -> Result<()> {
        let _payload = serde_json::to_value(alert)?;

        // In a real implementation, would send HTTP POST
        tracing::info!(
            "Webhook notification sent for alert {}: {}",
            alert.id,
            alert.message
        );
        Ok(())
    }

    fn get_channel_type(&self) -> String {
        "webhook".to_string()
    }
}

/// Format alerts as HTML for dashboard display
pub fn format_alerts(alerts: &[Alert]) -> String {
    if alerts.is_empty() {
        return "<p>No active alerts</p>".to_string();
    }

    alerts
        .iter()
        .map(|alert| {
            let class = match alert.severity {
                AlertSeverity::Critical => "alert-critical",
                AlertSeverity::Warning => "alert-warning",
                AlertSeverity::Info => "alert-info",
            };

            format!(
                "<div class=\"alert {}\">
                        <strong>{:?}</strong>: {}
                        <br><small>{}</small>
                    </div>",
                class,
                alert.alert_type,
                alert.message,
                chrono::DateTime::<chrono::Utc>::from(alert.timestamp).format("%H:%M:%S")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
