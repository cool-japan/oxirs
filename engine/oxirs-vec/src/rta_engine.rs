//! Real-time analytics engine and core event/metric types.

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

use crate::rta_aggregators::{
    AlertManager, DashboardData, ExportFormat, MetricsCollector, OverviewData, PerformanceMonitor,
    QueryAnalyzer, QueryMetrics, SystemMetrics,
};

/// Real-time analytics engine for vector operations
pub struct VectorAnalyticsEngine {
    pub(crate) config: AnalyticsConfig,
    pub(crate) metrics_collector: Arc<MetricsCollector>,
    pub(crate) performance_monitor: Arc<PerformanceMonitor>,
    pub(crate) query_analyzer: Arc<QueryAnalyzer>,
    pub(crate) alert_manager: Arc<AlertManager>,
    pub(crate) dashboard_data: Arc<RwLock<DashboardData>>,
    pub(crate) event_broadcaster: broadcast::Sender<AnalyticsEvent>,
}

/// Configuration for analytics engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable real-time monitoring
    pub enable_real_time: bool,
    /// Metrics collection interval in seconds
    pub collection_interval: u64,
    /// Maximum number of metrics to retain in memory
    pub max_metrics_history: usize,
    /// Enable query pattern analysis
    pub enable_query_analysis: bool,
    /// Enable performance alerting
    pub enable_alerts: bool,
    /// Dashboard refresh interval in seconds
    pub dashboard_refresh_interval: u64,
    /// Enable detailed tracing
    pub enable_tracing: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Metrics retention period in days
    pub retention_days: u32,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_real_time: true,
            collection_interval: 1,
            max_metrics_history: 10000,
            enable_query_analysis: true,
            enable_alerts: true,
            dashboard_refresh_interval: 5,
            enable_tracing: true,
            enable_profiling: true,
            retention_days: 30,
        }
    }
}

/// Analytics event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsEvent {
    QueryExecuted {
        query_id: String,
        operation_type: String,
        duration: Duration,
        result_count: usize,
        success: bool,
        timestamp: SystemTime,
    },
    IndexUpdated {
        index_name: String,
        operation: String,
        vectors_affected: usize,
        timestamp: SystemTime,
    },
    PerformanceAlert {
        alert_type: AlertType,
        message: String,
        severity: AlertSeverity,
        timestamp: SystemTime,
    },
    SystemMetric {
        metric_name: String,
        value: f64,
        unit: String,
        timestamp: SystemTime,
    },
}

/// Alert types for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighLatency,
    LowThroughput,
    HighMemoryUsage,
    HighCpuUsage,
    QualityDegradation,
    IndexCorruption,
    SystemError,
    ResourceLimitReached,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
    pub resolved: bool,
    pub resolved_timestamp: Option<SystemTime>,
    pub metadata: HashMap<String, String>,
}

/// Analytics report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    pub report_id: String,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub query_metrics: QueryMetrics,
    pub system_metrics: SystemMetrics,
    pub quality_metrics: crate::rta_aggregators::QualityMetrics,
    pub alerts: Vec<Alert>,
    pub recommendations: Vec<String>,
    pub generated_at: SystemTime,
}

/// System information structure
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub memory_total: u64,
    pub disk_usage: f64,
    pub network_throughput: f64,
}

impl Clone for VectorAnalyticsEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_collector: Arc::clone(&self.metrics_collector),
            performance_monitor: Arc::clone(&self.performance_monitor),
            query_analyzer: Arc::clone(&self.query_analyzer),
            alert_manager: Arc::clone(&self.alert_manager),
            dashboard_data: Arc::clone(&self.dashboard_data),
            event_broadcaster: self.event_broadcaster.clone(),
        }
    }
}

impl VectorAnalyticsEngine {
    pub fn new(config: AnalyticsConfig) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);

        let metrics_collector = Arc::new(MetricsCollector::new());
        let performance_monitor = Arc::new(PerformanceMonitor::new());
        let query_analyzer = Arc::new(QueryAnalyzer::new());
        let alert_manager = Arc::new(AlertManager::new(
            crate::rta_aggregators::AlertConfig::default(),
        ));
        let dashboard_data = Arc::new(RwLock::new(DashboardData::default()));

        Self {
            config,
            metrics_collector,
            performance_monitor,
            query_analyzer,
            alert_manager,
            dashboard_data,
            event_broadcaster,
        }
    }

    /// Record a query execution for analytics
    pub fn record_query_execution(
        &self,
        query_id: String,
        operation_type: String,
        duration: Duration,
        result_count: usize,
        success: bool,
    ) -> Result<()> {
        // Update metrics
        {
            let mut metrics = self.metrics_collector.query_metrics.write();
            metrics.total_queries += 1;

            if success {
                metrics.successful_queries += 1;
            } else {
                metrics.failed_queries += 1;
            }

            // Update latency statistics
            self.update_latency_statistics(&mut metrics, duration);

            // Update query distribution
            *metrics
                .query_distribution
                .entry(operation_type.clone())
                .or_insert(0) += 1;

            // Update error rate
            metrics.error_rate =
                (metrics.failed_queries as f64) / (metrics.total_queries as f64) * 100.0;
        }

        // Check for alerts
        self.check_performance_alerts(duration, success)?;

        // Broadcast event
        let event = AnalyticsEvent::QueryExecuted {
            query_id,
            operation_type,
            duration,
            result_count,
            success,
            timestamp: SystemTime::now(),
        };

        let _ = self.event_broadcaster.send(event);

        Ok(())
    }

    pub(crate) fn update_latency_statistics(&self, metrics: &mut QueryMetrics, duration: Duration) {
        let timestamp = SystemTime::now();

        // Add to history
        metrics.latency_history.push_back((timestamp, duration));
        if metrics.latency_history.len() > self.config.max_metrics_history {
            metrics.latency_history.pop_front();
        }

        // Update running averages and percentiles
        let latencies: Vec<Duration> = metrics.latency_history.iter().map(|(_, d)| *d).collect();

        if !latencies.is_empty() {
            let mut sorted_latencies = latencies.clone();
            sorted_latencies.sort();

            let len = sorted_latencies.len();
            metrics.p50_latency = sorted_latencies[len / 2];
            metrics.p95_latency = sorted_latencies[(len as f64 * 0.95) as usize];
            metrics.p99_latency = sorted_latencies[(len as f64 * 0.99) as usize];
            metrics.max_latency = *sorted_latencies
                .last()
                .expect("sorted_latencies validated to be non-empty");
            metrics.min_latency = *sorted_latencies
                .first()
                .expect("collection validated to be non-empty");

            let total_duration: Duration = latencies.iter().sum();
            metrics.average_latency = total_duration / len as u32;
        }
    }

    pub(crate) fn check_performance_alerts(&self, duration: Duration, success: bool) -> Result<()> {
        let thresholds = self.performance_monitor.thresholds.read();

        // Check latency threshold
        if duration.as_millis() > thresholds.max_latency_ms as u128 {
            self.create_alert(
                AlertType::HighLatency,
                AlertSeverity::Warning,
                format!(
                    "Query latency {}ms exceeds threshold {}ms",
                    duration.as_millis(),
                    thresholds.max_latency_ms
                ),
            )?;
        }

        // Check error rate
        if !success {
            let metrics = self.metrics_collector.query_metrics.read();
            if metrics.error_rate > thresholds.max_error_rate_percent {
                self.create_alert(
                    AlertType::SystemError,
                    AlertSeverity::Critical,
                    format!(
                        "Error rate {:.2}% exceeds threshold {:.2}%",
                        metrics.error_rate, thresholds.max_error_rate_percent
                    ),
                )?;
            }
        }

        Ok(())
    }

    pub(crate) fn create_alert(
        &self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
    ) -> Result<()> {
        let alert_id = format!(
            "{:?}_{}",
            alert_type,
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
        );

        let alert = Alert {
            id: alert_id,
            alert_type,
            severity,
            message,
            timestamp: SystemTime::now(),
            resolved: false,
            resolved_timestamp: None,
            metadata: HashMap::new(),
        };

        // Store alert
        {
            let mut current_alerts = self.performance_monitor.current_alerts.write();
            current_alerts.insert(alert.id.clone(), alert.clone());

            let mut alert_history = self.performance_monitor.alert_history.write();
            alert_history.push_back(alert.clone());
            if alert_history.len() > self.config.max_metrics_history {
                alert_history.pop_front();
            }
        }

        // Send notifications
        self.alert_manager.send_alert(&alert)?;

        // Broadcast event
        let event = AnalyticsEvent::PerformanceAlert {
            alert_type: alert.alert_type.clone(),
            message: alert.message.clone(),
            severity: alert.severity.clone(),
            timestamp: alert.timestamp,
        };

        let _ = self.event_broadcaster.send(event);

        Ok(())
    }

    /// Record distributed query execution across multiple nodes
    pub fn record_distributed_query(
        &self,
        query_id: String,
        node_count: usize,
        total_duration: Duration,
        _federation_id: Option<String>,
        success: bool,
    ) -> Result<()> {
        // Update distributed query metrics
        {
            let mut metrics = self.metrics_collector.query_metrics.write();
            metrics.total_queries += 1;

            if success {
                metrics.successful_queries += 1;
            } else {
                metrics.failed_queries += 1;
            }

            // Update latency statistics for distributed queries
            self.update_latency_statistics(&mut metrics, total_duration);

            // Update distributed query distribution
            let operation_type = format!("distributed_query_{node_count}_nodes");
            *metrics
                .query_distribution
                .entry(operation_type)
                .or_insert(0) += 1;

            // Update error rate
            metrics.error_rate = if metrics.total_queries > 0 {
                metrics.failed_queries as f64 / metrics.total_queries as f64
            } else {
                0.0
            };
        }

        // Record distributed query event
        let event = AnalyticsEvent::QueryExecuted {
            query_id: query_id.clone(),
            operation_type: format!("distributed_query_{node_count}_nodes"),
            duration: total_duration,
            result_count: node_count,
            success,
            timestamp: SystemTime::now(),
        };

        let _ = self.event_broadcaster.send(event);

        // Generate alert if distributed query is taking too long
        if total_duration.as_millis() > 5000 {
            let message = format!(
                "Distributed query {} across {} nodes took {}ms",
                query_id,
                node_count,
                total_duration.as_millis()
            );

            self.create_alert(AlertType::HighLatency, AlertSeverity::Warning, message)?;
        }

        Ok(())
    }

    /// Update system metrics
    pub fn update_system_metrics(
        &self,
        cpu_usage: f64,
        memory_usage: f64,
        memory_total: u64,
    ) -> Result<()> {
        {
            let mut metrics = self.metrics_collector.system_metrics.write();
            metrics.cpu_usage = cpu_usage;
            metrics.memory_usage = memory_usage;
            metrics.memory_total = memory_total;
            metrics.memory_available =
                memory_total - (memory_total as f64 * memory_usage / 100.0) as u64;
        }

        // Check system alerts
        let thresholds = self.performance_monitor.thresholds.read();

        if cpu_usage > thresholds.max_cpu_usage_percent {
            self.create_alert(
                AlertType::HighCpuUsage,
                AlertSeverity::Warning,
                format!(
                    "CPU usage {:.2}% exceeds threshold {:.2}%",
                    cpu_usage, thresholds.max_cpu_usage_percent
                ),
            )?;
        }

        if memory_usage > thresholds.max_memory_usage_percent {
            self.create_alert(
                AlertType::HighMemoryUsage,
                AlertSeverity::Warning,
                format!(
                    "Memory usage {:.2}% exceeds threshold {:.2}%",
                    memory_usage, thresholds.max_memory_usage_percent
                ),
            )?;
        }

        Ok(())
    }

    /// Get current dashboard data
    pub fn get_dashboard_data(&self) -> DashboardData {
        self.dashboard_data.read().clone()
    }

    /// Subscribe to analytics events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<AnalyticsEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Generate analytics report
    pub fn generate_report(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<AnalyticsReport> {
        let query_metrics = self.metrics_collector.query_metrics.read().clone();
        let system_metrics = self.metrics_collector.system_metrics.read().clone();
        let quality_metrics = self.metrics_collector.quality_metrics.read().clone();

        Ok(AnalyticsReport {
            report_id: format!(
                "report_{}",
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
            ),
            start_time,
            end_time,
            query_metrics,
            system_metrics,
            quality_metrics,
            alerts: self.get_alerts_in_range(start_time, end_time)?,
            recommendations: self.generate_recommendations()?,
            generated_at: SystemTime::now(),
        })
    }

    fn get_alerts_in_range(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Result<Vec<Alert>> {
        let alert_history = self.performance_monitor.alert_history.read();
        Ok(alert_history
            .iter()
            .filter(|alert| alert.timestamp >= start_time && alert.timestamp <= end_time)
            .cloned()
            .collect())
    }

    fn generate_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();

        let query_metrics = self.metrics_collector.query_metrics.read();
        let system_metrics = self.metrics_collector.system_metrics.read();

        // Performance recommendations
        if query_metrics.average_latency.as_millis() > 50 {
            recommendations
                .push("Consider optimizing queries or adding more powerful hardware".to_string());
        }

        if system_metrics.memory_usage > 80.0 {
            recommendations.push(
                "Memory usage is high. Consider increasing memory or optimizing data structures"
                    .to_string(),
            );
        }

        if system_metrics.cache_hit_ratio < 0.8 {
            recommendations.push("Cache hit ratio is low. Consider increasing cache size or improving cache strategy".to_string());
        }

        Ok(recommendations)
    }

    /// Export metrics to external systems
    pub fn export_metrics(&self, format: ExportFormat, destination: &str) -> Result<()> {
        let metrics_data = self.collect_all_metrics()?;

        match format {
            ExportFormat::Json => self.export_as_json(&metrics_data, destination),
            ExportFormat::Csv => self.export_as_csv(&metrics_data, destination),
            ExportFormat::Prometheus => self.export_as_prometheus(&metrics_data, destination),
            ExportFormat::InfluxDb => self.export_as_influxdb(&metrics_data, destination),
        }
    }

    fn collect_all_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut all_metrics = HashMap::new();

        let query_metrics = self.metrics_collector.query_metrics.read();
        let system_metrics = self.metrics_collector.system_metrics.read();
        let quality_metrics = self.metrics_collector.quality_metrics.read();

        all_metrics.insert(
            "query_metrics".to_string(),
            serde_json::to_value(&*query_metrics)?,
        );
        all_metrics.insert(
            "system_metrics".to_string(),
            serde_json::to_value(&*system_metrics)?,
        );
        all_metrics.insert(
            "quality_metrics".to_string(),
            serde_json::to_value(&*quality_metrics)?,
        );

        Ok(all_metrics)
    }

    fn export_as_json(
        &self,
        metrics: &HashMap<String, serde_json::Value>,
        destination: &str,
    ) -> Result<()> {
        let json_data = serde_json::to_string_pretty(metrics)?;
        std::fs::write(destination, json_data)?;
        Ok(())
    }

    fn export_as_csv(
        &self,
        metrics: &HashMap<String, serde_json::Value>,
        destination: &str,
    ) -> Result<()> {
        let mut csv_content = String::new();
        csv_content.push_str("timestamp,metric_name,value,category\n");

        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S");

        // Export query metrics
        if let Some(query_metrics) = metrics.get("query_metrics") {
            if let Some(obj) = query_metrics.as_object() {
                for (key, value) in obj {
                    if let Some(num_val) = value.as_f64() {
                        csv_content.push_str(&format!("{timestamp},query_{key},{num_val},query\n"));
                    }
                }
            }
        }

        // Export system metrics
        if let Some(system_metrics) = metrics.get("system_metrics") {
            if let Some(obj) = system_metrics.as_object() {
                for (key, value) in obj {
                    if let Some(num_val) = value.as_f64() {
                        csv_content
                            .push_str(&format!("{timestamp},system_{key},{num_val},system\n"));
                    }
                }
            }
        }

        std::fs::write(destination, csv_content)?;
        Ok(())
    }

    fn export_as_prometheus(
        &self,
        metrics: &HashMap<String, serde_json::Value>,
        destination: &str,
    ) -> Result<()> {
        let mut prometheus_content = String::new();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();

        // Export query metrics
        if let Some(query_metrics) = metrics.get("query_metrics") {
            if let Some(obj) = query_metrics.as_object() {
                for (key, value) in obj {
                    if let Some(num_val) = value.as_f64() {
                        prometheus_content
                            .push_str(&format!("# HELP vector_query_{key} Query metric {key}\n"));
                        prometheus_content.push_str(&format!("# TYPE vector_query_{key} gauge\n"));
                        prometheus_content
                            .push_str(&format!("vector_query_{key} {num_val} {timestamp}\n"));
                    }
                }
            }
        }

        // Export system metrics
        if let Some(system_metrics) = metrics.get("system_metrics") {
            if let Some(obj) = system_metrics.as_object() {
                for (key, value) in obj {
                    if let Some(num_val) = value.as_f64() {
                        prometheus_content
                            .push_str(&format!("# HELP vector_system_{key} System metric {key}\n"));
                        prometheus_content.push_str(&format!("# TYPE vector_system_{key} gauge\n"));
                        prometheus_content
                            .push_str(&format!("vector_system_{key} {num_val} {timestamp}\n"));
                    }
                }
            }
        }

        std::fs::write(destination, prometheus_content)?;
        Ok(())
    }

    fn export_as_influxdb(
        &self,
        metrics: &HashMap<String, serde_json::Value>,
        destination: &str,
    ) -> Result<()> {
        let mut influxdb_content = String::new();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();

        // Export query metrics
        if let Some(query_metrics) = metrics.get("query_metrics") {
            if let Some(obj) = query_metrics.as_object() {
                for (key, value) in obj {
                    if let Some(num_val) = value.as_f64() {
                        influxdb_content.push_str(&format!(
                            "vector_query,type=query {key}={num_val} {timestamp}\n"
                        ));
                    }
                }
            }
        }

        // Export system metrics
        if let Some(system_metrics) = metrics.get("system_metrics") {
            if let Some(obj) = system_metrics.as_object() {
                for (key, value) in obj {
                    if let Some(num_val) = value.as_f64() {
                        influxdb_content.push_str(&format!(
                            "vector_system,type=system {key}={num_val} {timestamp}\n"
                        ));
                    }
                }
            }
        }

        std::fs::write(destination, influxdb_content)?;
        Ok(())
    }

    /// Start real-time dashboard update loop
    pub async fn start_dashboard_updates(&self) -> Result<()> {
        let dashboard_data = Arc::clone(&self.dashboard_data);
        let metrics_collector = Arc::clone(&self.metrics_collector);
        let performance_monitor = Arc::clone(&self.performance_monitor);
        let refresh_interval = Duration::from_secs(self.config.dashboard_refresh_interval);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(refresh_interval);

            loop {
                interval.tick().await;

                // Update dashboard data
                let updated_data =
                    Self::build_dashboard_data(&metrics_collector, &performance_monitor).await;

                {
                    let mut data = dashboard_data.write();
                    *data = updated_data;
                }
            }
        });

        Ok(())
    }

    async fn build_dashboard_data(
        metrics_collector: &MetricsCollector,
        performance_monitor: &PerformanceMonitor,
    ) -> DashboardData {
        use crate::rta_aggregators::{
            QualityMetricsData, QueryPerformanceData, SystemHealthData, UsageAnalyticsData,
        };

        let query_metrics = metrics_collector.query_metrics.read().clone();
        let system_metrics = metrics_collector.system_metrics.read().clone();
        let quality_metrics = metrics_collector.quality_metrics.read().clone();
        let current_alerts: Vec<Alert> = performance_monitor
            .current_alerts
            .read()
            .values()
            .cloned()
            .collect();

        // Calculate system health score
        let health_score = Self::calculate_health_score(&system_metrics, &query_metrics);

        // Calculate current QPS
        let current_qps = Self::calculate_current_qps(&query_metrics);

        DashboardData {
            overview: OverviewData {
                total_queries_today: query_metrics.total_queries,
                average_latency: query_metrics.average_latency,
                current_qps,
                system_health_score: health_score,
                active_alerts: current_alerts.len() as u64,
                index_size: system_metrics.index_size,
                vector_count: system_metrics.vector_count,
                cache_hit_ratio: system_metrics.cache_hit_ratio,
            },
            query_performance: QueryPerformanceData {
                latency_trends: query_metrics.latency_history.iter().cloned().collect(),
                throughput_trends: query_metrics.throughput_history.iter().cloned().collect(),
                error_rate_trends: vec![(SystemTime::now(), query_metrics.error_rate)],
                top_slow_queries: vec![], // Would be populated with actual slow queries
                query_distribution: query_metrics.query_distribution.clone(),
                performance_percentiles: {
                    let mut percentiles = HashMap::new();
                    percentiles.insert("p50".to_string(), query_metrics.p50_latency);
                    percentiles.insert("p95".to_string(), query_metrics.p95_latency);
                    percentiles.insert("p99".to_string(), query_metrics.p99_latency);
                    percentiles
                },
            },
            system_health: SystemHealthData {
                cpu_usage: system_metrics.cpu_usage,
                memory_usage: system_metrics.memory_usage,
                disk_usage: system_metrics.disk_usage,
                network_throughput: 0.0, // Would be calculated from network metrics
                resource_trends: vec![(SystemTime::now(), system_metrics.cpu_usage)],
                capacity_forecast: vec![], // Would be calculated with forecasting algorithm
                bottlenecks: Self::identify_bottlenecks(&system_metrics, &query_metrics),
            },
            quality_metrics: QualityMetricsData {
                recall_trends: vec![],
                precision_trends: vec![],
                similarity_distribution: quality_metrics.similarity_distribution.clone(),
                quality_score: quality_metrics.average_similarity_score,
                quality_trends: vec![(SystemTime::now(), quality_metrics.average_similarity_score)],
                benchmark_comparisons: HashMap::new(),
            },
            usage_analytics: UsageAnalyticsData {
                user_activity: vec![(SystemTime::now(), query_metrics.total_queries)],
                popular_queries: vec![], // Would be populated from query analyzer
                usage_patterns: HashMap::new(),
                growth_metrics: crate::rta_aggregators::GrowthMetrics::default(),
                feature_usage: HashMap::new(),
            },
            alerts: current_alerts,
            last_updated: SystemTime::now(),
        }
    }

    fn calculate_health_score(system_metrics: &SystemMetrics, query_metrics: &QueryMetrics) -> f64 {
        let mut score = 100.0;

        // Deduct points for high resource usage
        if system_metrics.cpu_usage > 80.0 {
            score -= (system_metrics.cpu_usage - 80.0) * 0.5;
        }
        if system_metrics.memory_usage > 80.0 {
            score -= (system_metrics.memory_usage - 80.0) * 0.5;
        }

        // Deduct points for high error rate
        if query_metrics.error_rate > 1.0 {
            score -= query_metrics.error_rate * 10.0;
        }

        // Deduct points for high latency
        if query_metrics.average_latency.as_millis() > 100 {
            score -= (query_metrics.average_latency.as_millis() as f64 - 100.0) * 0.1;
        }

        score.clamp(0.0, 100.0)
    }

    fn calculate_current_qps(query_metrics: &QueryMetrics) -> f64 {
        // Calculate QPS from recent query history
        if query_metrics.latency_history.len() < 2 {
            return 0.0;
        }

        let now = SystemTime::now();
        let one_second_ago = now - Duration::from_secs(1);

        let recent_queries = query_metrics
            .latency_history
            .iter()
            .filter(|(timestamp, _)| *timestamp >= one_second_ago)
            .count();

        recent_queries as f64
    }

    fn identify_bottlenecks(
        system_metrics: &SystemMetrics,
        query_metrics: &QueryMetrics,
    ) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        if system_metrics.cpu_usage > 90.0 {
            bottlenecks.push("High CPU usage".to_string());
        }

        if system_metrics.memory_usage > 90.0 {
            bottlenecks.push("High memory usage".to_string());
        }

        if query_metrics.average_latency.as_millis() > 500 {
            bottlenecks.push("High query latency".to_string());
        }

        if system_metrics.cache_hit_ratio < 0.7 {
            bottlenecks.push("Low cache hit ratio".to_string());
        }

        bottlenecks
    }

    /// Generate web dashboard HTML
    pub fn generate_dashboard_html(&self) -> Result<String> {
        use crate::rta_aggregators::format_alerts;

        let dashboard_data = self.get_dashboard_data();

        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>OxiRS Vector Search Analytics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .metric-value {{ font-weight: bold; color: #007acc; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .alert-critical {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
        .alert-warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .alert-info {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; }}
        .health-score {{ font-size: 2em; text-align: center; }}
        .health-good {{ color: #4caf50; }}
        .health-warning {{ color: #ff9800; }}
        .health-critical {{ color: #f44336; }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #555; margin-top: 0; }}
        .refresh-time {{ text-align: center; color: #888; font-size: 0.9em; }}
    </style>
    <script>
        function refreshPage() {{
            window.location.reload();
        }}
        setInterval(refreshPage, 30000); // Refresh every 30 seconds
    </script>
</head>
<body>
    <h1>OxiRS Vector Search Analytics Dashboard</h1>
    <p class="refresh-time">Last updated: {}</p>

    <div class="dashboard">
        <div class="card">
            <h2>System Health</h2>
            <div class="health-score {}">{:.1}%</div>
            <div class="metric">
                <span>Active Alerts:</span>
                <span class="metric-value">{}</span>
            </div>
        </div>

        <div class="card">
            <h2>Query Performance</h2>
            <div class="metric">
                <span>Total Queries Today:</span>
                <span class="metric-value">{}</span>
            </div>
            <div class="metric">
                <span>Average Latency:</span>
                <span class="metric-value">{}ms</span>
            </div>
            <div class="metric">
                <span>Current QPS:</span>
                <span class="metric-value">{:.1}</span>
            </div>
        </div>

        <div class="card">
            <h2>System Resources</h2>
            <div class="metric">
                <span>CPU Usage:</span>
                <span class="metric-value">{:.1}%</span>
            </div>
            <div class="metric">
                <span>Memory Usage:</span>
                <span class="metric-value">{:.1}%</span>
            </div>
            <div class="metric">
                <span>Cache Hit Ratio:</span>
                <span class="metric-value">{:.1}%</span>
            </div>
        </div>

        <div class="card">
            <h2>Vector Index</h2>
            <div class="metric">
                <span>Vector Count:</span>
                <span class="metric-value">{}</span>
            </div>
            <div class="metric">
                <span>Index Size:</span>
                <span class="metric-value">{} MB</span>
            </div>
        </div>

        <div class="card">
            <h2>Active Alerts</h2>
            {}
        </div>
    </div>
</body>
</html>
            "#,
            chrono::DateTime::<chrono::Utc>::from(dashboard_data.last_updated)
                .format("%Y-%m-%d %H:%M:%S UTC"),
            if dashboard_data.overview.system_health_score >= 80.0 {
                "health-good"
            } else if dashboard_data.overview.system_health_score >= 60.0 {
                "health-warning"
            } else {
                "health-critical"
            },
            dashboard_data.overview.system_health_score,
            dashboard_data.overview.active_alerts,
            dashboard_data.overview.total_queries_today,
            dashboard_data.overview.average_latency.as_millis(),
            dashboard_data.overview.current_qps,
            dashboard_data.system_health.cpu_usage,
            dashboard_data.system_health.memory_usage,
            dashboard_data.overview.cache_hit_ratio * 100.0,
            dashboard_data.overview.vector_count,
            dashboard_data.overview.index_size / (1024 * 1024), // Convert to MB
            format_alerts(&dashboard_data.alerts)
        );

        Ok(html)
    }

    /// Start comprehensive system monitoring
    pub async fn start_system_monitoring(&self) -> Result<()> {
        let analytics_engine = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                // Collect system metrics
                if let Ok(system_info) = Self::collect_system_info().await {
                    let _ = analytics_engine.update_system_metrics(
                        system_info.cpu_usage,
                        system_info.memory_usage,
                        system_info.memory_total,
                    );
                }
            }
        });

        Ok(())
    }

    async fn collect_system_info() -> Result<SystemInfo> {
        // In a real implementation, would use system monitoring library
        // For now, return mock data
        Ok(SystemInfo {
            cpu_usage: {
                #[allow(unused_imports)]
                use scirs2_core::random::{Random, Rng};
                let mut rng = Random::seed(42);
                45.0 + (rng.gen_range(0.0..1.0) * 20.0) // Mock: 45-65%
            },
            memory_usage: {
                #[allow(unused_imports)]
                use scirs2_core::random::{Random, Rng};
                let mut rng = Random::seed(42);
                60.0 + (rng.gen_range(0.0..1.0) * 20.0) // Mock: 60-80%
            },
            memory_total: 16 * 1024 * 1024 * 1024, // 16GB
            disk_usage: 30.0,
            network_throughput: 100.0,
        })
    }
}
