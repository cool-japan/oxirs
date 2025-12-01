//! Analytics Dashboard Backend API
//!
//! Provides comprehensive analytics and metrics endpoints for monitoring
//! chat system performance, user activity, and query patterns.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Dashboard analytics manager
pub struct DashboardAnalytics {
    /// Query performance metrics
    query_metrics: Arc<RwLock<QueryMetrics>>,
    /// User activity tracker
    user_activity: Arc<RwLock<UserActivityTracker>>,
    /// System health metrics
    system_health: Arc<RwLock<SystemHealthMetrics>>,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Metrics retention period (days)
    pub retention_days: u32,
    /// Enable real-time updates
    pub enable_realtime: bool,
    /// Aggregation interval (seconds)
    pub aggregation_interval_secs: u64,
    /// Maximum data points per chart
    pub max_data_points: usize,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            retention_days: 30,
            enable_realtime: true,
            aggregation_interval_secs: 300, // 5 minutes
            max_data_points: 100,
        }
    }
}

/// Query performance metrics
#[derive(Debug, Clone, Default)]
pub struct QueryMetrics {
    /// Total queries executed
    pub total_queries: u64,
    /// Successful queries
    pub successful_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Average response time (milliseconds)
    pub avg_response_time_ms: f64,
    /// P95 response time
    pub p95_response_time_ms: f64,
    /// P99 response time
    pub p99_response_time_ms: f64,
    /// Query history
    pub query_history: Vec<QueryRecord>,
}

/// Individual query record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRecord {
    pub query_id: String,
    pub query_type: QueryType,
    pub execution_time_ms: u64,
    pub result_count: usize,
    pub success: bool,
    pub timestamp: DateTime<Utc>,
    pub error: Option<String>,
}

/// Query type classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    NaturalLanguage,
    Sparql,
    VectorSearch,
    Hybrid,
}

/// User activity tracker
#[derive(Debug, Clone, Default)]
pub struct UserActivityTracker {
    /// Active users (last 24 hours)
    pub active_users_24h: u64,
    /// Total sessions
    pub total_sessions: u64,
    /// Average session duration (seconds)
    pub avg_session_duration_secs: f64,
    /// User activity timeline
    pub activity_timeline: Vec<ActivityDataPoint>,
    /// Top users by activity
    pub top_users: Vec<UserActivity>,
}

/// Activity data point for timeline charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityDataPoint {
    pub timestamp: DateTime<Utc>,
    pub active_users: u64,
    pub queries_per_minute: f64,
    pub avg_response_time_ms: f64,
}

/// User activity summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserActivity {
    pub user_id: String,
    pub query_count: u64,
    pub session_count: u64,
    pub total_time_secs: u64,
    pub last_active: DateTime<Utc>,
}

/// System health metrics
#[derive(Debug, Clone, Default)]
pub struct SystemHealthMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Active connections
    pub active_connections: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Error rate (per 1000 requests)
    pub error_rate: f64,
    /// Health timeline
    pub health_timeline: Vec<HealthDataPoint>,
}

/// Health data point for system monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDataPoint {
    pub timestamp: DateTime<Utc>,
    pub cpu_percent: f64,
    pub memory_mb: f64,
    pub active_connections: u64,
    pub requests_per_second: f64,
}

impl DashboardAnalytics {
    /// Create a new dashboard analytics manager
    pub fn new(config: DashboardConfig) -> Self {
        info!(
            "Initializing dashboard analytics with retention: {} days",
            config.retention_days
        );

        Self {
            query_metrics: Arc::new(RwLock::new(QueryMetrics::default())),
            user_activity: Arc::new(RwLock::new(UserActivityTracker::default())),
            system_health: Arc::new(RwLock::new(SystemHealthMetrics::default())),
        }
    }

    /// Get comprehensive dashboard overview
    pub async fn get_overview(&self) -> DashboardOverview {
        let query_metrics = self.query_metrics.read().await;
        let user_activity = self.user_activity.read().await;
        let system_health = self.system_health.read().await;

        DashboardOverview {
            total_queries: query_metrics.total_queries,
            successful_queries: query_metrics.successful_queries,
            failed_queries: query_metrics.failed_queries,
            avg_response_time_ms: query_metrics.avg_response_time_ms,
            active_users_24h: user_activity.active_users_24h,
            total_sessions: user_activity.total_sessions,
            cpu_usage_percent: system_health.cpu_usage_percent,
            memory_usage_mb: system_health.memory_usage_mb,
            cache_hit_rate: system_health.cache_hit_rate,
            error_rate: system_health.error_rate,
            timestamp: Utc::now(),
        }
    }

    /// Get query performance analytics
    pub async fn get_query_analytics(&self, time_range: TimeRange) -> QueryAnalytics {
        let metrics = self.query_metrics.read().await;

        // Filter queries by time range
        let filtered_queries: Vec<_> = metrics
            .query_history
            .iter()
            .filter(|q| time_range.contains(q.timestamp))
            .cloned()
            .collect();

        // Calculate statistics
        let total = filtered_queries.len() as u64;
        let successful = filtered_queries.iter().filter(|q| q.success).count() as u64;
        let failed = total - successful;

        let execution_times: Vec<f64> = filtered_queries
            .iter()
            .map(|q| q.execution_time_ms as f64)
            .collect();

        let avg_time = if !execution_times.is_empty() {
            execution_times.iter().sum::<f64>() / execution_times.len() as f64
        } else {
            0.0
        };

        // Query type distribution
        let mut type_distribution = HashMap::new();
        for query in &filtered_queries {
            *type_distribution.entry(query.query_type).or_insert(0) += 1;
        }

        QueryAnalytics {
            total_queries: total,
            successful_queries: successful,
            failed_queries: failed,
            avg_response_time_ms: avg_time,
            p95_response_time_ms: Self::calculate_percentile(&execution_times, 0.95),
            p99_response_time_ms: Self::calculate_percentile(&execution_times, 0.99),
            query_type_distribution: type_distribution,
            time_range,
        }
    }

    /// Get user activity analytics
    pub async fn get_user_analytics(&self, time_range: TimeRange) -> UserAnalytics {
        let activity = self.user_activity.read().await;

        // Filter activity by time range
        let filtered_timeline: Vec<_> = activity
            .activity_timeline
            .iter()
            .filter(|a| time_range.contains(a.timestamp))
            .cloned()
            .collect();

        UserAnalytics {
            active_users: activity.active_users_24h,
            total_sessions: activity.total_sessions,
            avg_session_duration_secs: activity.avg_session_duration_secs,
            activity_timeline: filtered_timeline,
            top_users: activity.top_users.clone(),
            time_range,
        }
    }

    /// Get system health analytics
    pub async fn get_health_analytics(&self, time_range: TimeRange) -> HealthAnalytics {
        let health = self.system_health.read().await;

        // Filter health data by time range
        let filtered_timeline: Vec<_> = health
            .health_timeline
            .iter()
            .filter(|h| time_range.contains(h.timestamp))
            .cloned()
            .collect();

        HealthAnalytics {
            current_cpu_percent: health.cpu_usage_percent,
            current_memory_mb: health.memory_usage_mb,
            active_connections: health.active_connections,
            cache_hit_rate: health.cache_hit_rate,
            error_rate: health.error_rate,
            health_timeline: filtered_timeline,
            time_range,
        }
    }

    /// Record a query execution
    pub async fn record_query(&self, record: QueryRecord) {
        let mut metrics = self.query_metrics.write().await;

        metrics.total_queries += 1;
        if record.success {
            metrics.successful_queries += 1;
        } else {
            metrics.failed_queries += 1;
        }

        // Update average response time
        let total_time = metrics.avg_response_time_ms * (metrics.total_queries - 1) as f64
            + record.execution_time_ms as f64;
        metrics.avg_response_time_ms = total_time / metrics.total_queries as f64;

        metrics.query_history.push(record);

        // Keep only recent queries (limit to 10,000)
        if metrics.query_history.len() > 10_000 {
            metrics.query_history.drain(0..1_000);
        }
    }

    /// Update user activity
    pub async fn update_user_activity(&self, user_id: String, query_count: u64) {
        let mut activity = self.user_activity.write().await;

        // Update or create user activity record
        if let Some(user) = activity.top_users.iter_mut().find(|u| u.user_id == user_id) {
            user.query_count += query_count;
            user.last_active = Utc::now();
        } else {
            activity.top_users.push(UserActivity {
                user_id,
                query_count,
                session_count: 1,
                total_time_secs: 0,
                last_active: Utc::now(),
            });
        }

        // Sort by query count and keep top 100
        activity
            .top_users
            .sort_by(|a, b| b.query_count.cmp(&a.query_count));
        activity.top_users.truncate(100);
    }

    /// Update system health metrics
    pub async fn update_health(&self, cpu_percent: f64, memory_mb: f64, connections: u64) {
        let mut health = self.system_health.write().await;

        health.cpu_usage_percent = cpu_percent;
        health.memory_usage_mb = memory_mb;
        health.active_connections = connections;

        // Add to timeline
        health.health_timeline.push(HealthDataPoint {
            timestamp: Utc::now(),
            cpu_percent,
            memory_mb,
            active_connections: connections,
            requests_per_second: 0.0, // TODO: Calculate from metrics
        });

        // Keep only recent data (last 24 hours at 5-minute intervals = 288 points)
        if health.health_timeline.len() > 288 {
            health.health_timeline.drain(0..100);
        }
    }

    /// Calculate percentile from sorted values
    fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile * sorted.len() as f64) as usize;
        sorted.get(index).copied().unwrap_or(0.0)
    }
}

/// Dashboard overview summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardOverview {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub avg_response_time_ms: f64,
    pub active_users_24h: u64,
    pub total_sessions: u64,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub timestamp: DateTime<Utc>,
}

/// Query analytics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalytics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub avg_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub query_type_distribution: HashMap<QueryType, u64>,
    pub time_range: TimeRange,
}

/// User analytics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAnalytics {
    pub active_users: u64,
    pub total_sessions: u64,
    pub avg_session_duration_secs: f64,
    pub activity_timeline: Vec<ActivityDataPoint>,
    pub top_users: Vec<UserActivity>,
    pub time_range: TimeRange,
}

/// Health analytics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAnalytics {
    pub current_cpu_percent: f64,
    pub current_memory_mb: f64,
    pub active_connections: u64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub health_timeline: Vec<HealthDataPoint>,
    pub time_range: TimeRange,
}

/// Time range for analytics queries
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl TimeRange {
    /// Create a time range for the last N hours
    pub fn last_hours(hours: i64) -> Self {
        let end = Utc::now();
        let start = end - Duration::hours(hours);
        Self { start, end }
    }

    /// Create a time range for the last N days
    pub fn last_days(days: i64) -> Self {
        let end = Utc::now();
        let start = end - Duration::days(days);
        Self { start, end }
    }

    /// Check if a timestamp is within this range
    pub fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.start && timestamp <= self.end
    }
}

/// Export format for analytics data
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Json,
    Csv,
    Excel,
}

impl DashboardAnalytics {
    /// Export analytics data in specified format
    pub async fn export_data(
        &self,
        format: ExportFormat,
        time_range: TimeRange,
    ) -> Result<Vec<u8>> {
        match format {
            ExportFormat::Json => self.export_json(time_range).await,
            ExportFormat::Csv => self.export_csv(time_range).await,
            ExportFormat::Excel => self.export_excel(time_range).await,
        }
    }

    async fn export_json(&self, time_range: TimeRange) -> Result<Vec<u8>> {
        let overview = self.get_overview().await;
        let query_analytics = self.get_query_analytics(time_range).await;
        let user_analytics = self.get_user_analytics(time_range).await;
        let health_analytics = self.get_health_analytics(time_range).await;

        let export_data = serde_json::json!({
            "overview": overview,
            "query_analytics": query_analytics,
            "user_analytics": user_analytics,
            "health_analytics": health_analytics,
        });

        Ok(serde_json::to_vec_pretty(&export_data)?)
    }

    async fn export_csv(&self, _time_range: TimeRange) -> Result<Vec<u8>> {
        // TODO: Implement CSV export
        Ok(b"CSV export not yet implemented".to_vec())
    }

    async fn export_excel(&self, _time_range: TimeRange) -> Result<Vec<u8>> {
        // TODO: Implement Excel export
        Ok(b"Excel export not yet implemented".to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = DashboardAnalytics::new(config);
        let overview = dashboard.get_overview().await;

        assert_eq!(overview.total_queries, 0);
        assert_eq!(overview.active_users_24h, 0);
    }

    #[tokio::test]
    async fn test_record_query() {
        let config = DashboardConfig::default();
        let dashboard = DashboardAnalytics::new(config);

        let record = QueryRecord {
            query_id: "test-query-1".to_string(),
            query_type: QueryType::NaturalLanguage,
            execution_time_ms: 150,
            result_count: 5,
            success: true,
            timestamp: Utc::now(),
            error: None,
        };

        dashboard.record_query(record).await;

        let overview = dashboard.get_overview().await;
        assert_eq!(overview.total_queries, 1);
        assert_eq!(overview.successful_queries, 1);
    }

    #[tokio::test]
    async fn test_time_range() {
        let now = Utc::now();
        let range = TimeRange {
            start: now - Duration::hours(24),
            end: now + Duration::hours(1), // Add buffer for test timing
        };

        assert!(range.contains(now));
        assert!(!range.contains(now - Duration::days(2)));
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p95 = DashboardAnalytics::calculate_percentile(&values, 0.95);
        assert!(p95 >= 9.0);
    }
}
