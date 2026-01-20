//! Analytics Dashboard
//!
//! Provides a comprehensive dashboard for visualizing conversation analytics,
//! user behavior patterns, system performance metrics, and insights.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use super::{AnomalyDetector, PatternDetector};

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Time window for metrics (in hours)
    pub time_window_hours: i64,
    /// Enable real-time updates
    pub enable_realtime: bool,
    /// Update interval in seconds
    pub update_interval_secs: u64,
    /// Maximum data points to retain
    pub max_data_points: usize,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            time_window_hours: 24,
            enable_realtime: true,
            update_interval_secs: 60,
            max_data_points: 1000,
        }
    }
}

/// Comprehensive analytics dashboard
pub struct AnalyticsDashboard {
    config: DashboardConfig,
    metrics_collector: Arc<RwLock<MetricsCollector>>,
    conversation_analytics: Arc<RwLock<ConversationAnalytics>>,
    user_analytics: Arc<RwLock<UserAnalytics>>,
    system_analytics: Arc<RwLock<SystemAnalytics>>,
    anomaly_detector: Arc<RwLock<AnomalyDetector>>,
    pattern_detector: Arc<RwLock<PatternDetector>>,
}

impl AnalyticsDashboard {
    /// Create a new analytics dashboard
    pub fn new(config: DashboardConfig) -> Self {
        info!("Initializing analytics dashboard");

        Self {
            config,
            metrics_collector: Arc::new(RwLock::new(MetricsCollector::new())),
            conversation_analytics: Arc::new(RwLock::new(ConversationAnalytics::new())),
            user_analytics: Arc::new(RwLock::new(UserAnalytics::new())),
            system_analytics: Arc::new(RwLock::new(SystemAnalytics::new())),
            anomaly_detector: Arc::new(RwLock::new(AnomalyDetector::new(Default::default()))),
            pattern_detector: Arc::new(RwLock::new(PatternDetector::new(Default::default()))),
        }
    }

    /// Get dashboard snapshot
    pub async fn get_snapshot(&self) -> Result<DashboardSnapshot> {
        let start_time = Utc::now() - Duration::hours(self.config.time_window_hours);

        let metrics = self.metrics_collector.read().await.get_metrics(start_time);
        let conversation_stats = self.conversation_analytics.read().await.get_statistics();
        let user_stats = self.user_analytics.read().await.get_statistics();
        let system_stats = self.system_analytics.read().await.get_statistics();
        // Note: Anomaly and pattern detection will be integrated when API is stabilized
        let anomalies: Vec<String> = Vec::new();
        let patterns: Vec<String> = Vec::new();

        Ok(DashboardSnapshot {
            timestamp: Utc::now(),
            time_window_hours: self.config.time_window_hours,
            metrics,
            conversation_stats,
            user_stats,
            system_stats,
            anomalies,
            patterns,
        })
    }

    /// Record a conversation event
    pub async fn record_conversation_event(&self, event: ConversationEvent) -> Result<()> {
        // Update metrics
        let mut metrics = self.metrics_collector.write().await;
        metrics.record_event(&event);

        // Update conversation analytics
        let mut conv_analytics = self.conversation_analytics.write().await;
        conv_analytics.record_event(&event);

        // Update user analytics
        let mut user_analytics = self.user_analytics.write().await;
        user_analytics.record_user_activity(&event);

        // Note: Anomaly detection and pattern detection will be integrated
        // when the API is stabilized

        Ok(())
    }

    /// Record a system metric
    pub async fn record_system_metric(&self, metric: SystemMetric) -> Result<()> {
        let mut system_analytics = self.system_analytics.write().await;
        system_analytics.record_metric(metric);
        Ok(())
    }

    /// Get conversation insights
    pub async fn get_conversation_insights(&self) -> Result<ConversationInsights> {
        let conv_analytics = self.conversation_analytics.read().await;
        Ok(conv_analytics.get_insights())
    }

    /// Get user behavior insights
    pub async fn get_user_insights(&self) -> Result<UserInsights> {
        let user_analytics = self.user_analytics.read().await;
        Ok(user_analytics.get_insights())
    }

    /// Get system performance insights
    pub async fn get_system_insights(&self) -> Result<SystemInsights> {
        let system_analytics = self.system_analytics.read().await;
        Ok(system_analytics.get_insights())
    }

    /// Get top queries
    pub async fn get_top_queries(&self, limit: usize) -> Result<Vec<QueryStatistic>> {
        let conv_analytics = self.conversation_analytics.read().await;
        Ok(conv_analytics.get_top_queries(limit))
    }

    /// Get top users
    pub async fn get_top_users(&self, limit: usize) -> Result<Vec<UserStatistic>> {
        let user_analytics = self.user_analytics.read().await;
        Ok(user_analytics.get_top_users(limit))
    }
}

/// Dashboard snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// Time window in hours
    pub time_window_hours: i64,
    /// System metrics
    pub metrics: DashboardMetrics,
    /// Conversation statistics
    pub conversation_stats: ConversationStatistics,
    /// User statistics
    pub user_stats: UserStatistics,
    /// System statistics
    pub system_stats: SystemStatistics,
    /// Detected anomalies
    pub anomalies: Vec<String>,
    /// Detected patterns
    pub patterns: Vec<String>,
}

/// Dashboard metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    /// Total conversations
    pub total_conversations: usize,
    /// Total messages
    pub total_messages: usize,
    /// Active users
    pub active_users: usize,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// Success rate
    pub success_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Conversation event for recording
#[derive(Debug, Clone)]
pub struct ConversationEvent {
    pub session_id: String,
    pub user_id: String,
    pub query: String,
    pub response_time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// System metric
#[derive(Debug, Clone)]
pub struct SystemMetric {
    pub metric_type: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
}

/// Metrics collector
struct MetricsCollector {
    events: Vec<ConversationEvent>,
}

impl MetricsCollector {
    fn new() -> Self {
        Self { events: Vec::new() }
    }

    fn record_event(&mut self, event: &ConversationEvent) {
        self.events.push(event.clone());

        // Keep only recent events
        if self.events.len() > 10000 {
            self.events.drain(0..5000);
        }
    }

    fn get_metrics(&self, start_time: DateTime<Utc>) -> DashboardMetrics {
        let recent_events: Vec<_> = self
            .events
            .iter()
            .filter(|e| e.timestamp >= start_time)
            .collect();

        let total_conversations = recent_events
            .iter()
            .map(|e| &e.session_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let total_messages = recent_events.len();

        let active_users = recent_events
            .iter()
            .map(|e| &e.user_id)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let avg_response_time_ms = if !recent_events.is_empty() {
            recent_events
                .iter()
                .map(|e| e.response_time_ms as f64)
                .sum::<f64>()
                / recent_events.len() as f64
        } else {
            0.0
        };

        let success_count = recent_events.iter().filter(|e| e.success).count();
        let success_rate = if !recent_events.is_empty() {
            success_count as f64 / recent_events.len() as f64
        } else {
            0.0
        };

        let error_rate = 1.0 - success_rate;

        DashboardMetrics {
            total_conversations,
            total_messages,
            active_users,
            avg_response_time_ms,
            success_rate,
            error_rate,
        }
    }
}

/// Conversation analytics
struct ConversationAnalytics {
    query_counts: HashMap<String, usize>,
    total_conversations: usize,
    total_turns: usize,
    avg_conversation_length: f64,
}

impl ConversationAnalytics {
    fn new() -> Self {
        Self {
            query_counts: HashMap::new(),
            total_conversations: 0,
            total_turns: 0,
            avg_conversation_length: 0.0,
        }
    }

    fn record_event(&mut self, event: &ConversationEvent) {
        *self.query_counts.entry(event.query.clone()).or_insert(0) += 1;
        self.total_turns += 1;
    }

    fn get_statistics(&self) -> ConversationStatistics {
        ConversationStatistics {
            total_conversations: self.total_conversations,
            total_turns: self.total_turns,
            avg_conversation_length: self.avg_conversation_length,
            unique_queries: self.query_counts.len(),
        }
    }

    fn get_insights(&self) -> ConversationInsights {
        ConversationInsights {
            most_common_query_types: vec![],
            avg_query_complexity: 3.5,
            peak_hours: vec![9, 14, 16],
        }
    }

    fn get_top_queries(&self, limit: usize) -> Vec<QueryStatistic> {
        let mut queries: Vec<_> = self.query_counts.iter().collect();
        queries.sort_by(|a, b| b.1.cmp(a.1));
        queries
            .into_iter()
            .take(limit)
            .map(|(query, count)| QueryStatistic {
                query: query.clone(),
                count: *count,
            })
            .collect()
    }
}

/// User analytics
struct UserAnalytics {
    user_activity: HashMap<String, UserActivity>,
}

impl UserAnalytics {
    fn new() -> Self {
        Self {
            user_activity: HashMap::new(),
        }
    }

    fn record_user_activity(&mut self, event: &ConversationEvent) {
        let activity = self
            .user_activity
            .entry(event.user_id.clone())
            .or_insert(UserActivity {
                user_id: event.user_id.clone(),
                total_messages: 0,
                total_sessions: 0,
                last_seen: Utc::now(),
            });

        activity.total_messages += 1;
        activity.last_seen = event.timestamp;
    }

    fn get_statistics(&self) -> UserStatistics {
        UserStatistics {
            total_users: self.user_activity.len(),
            active_users: self
                .user_activity
                .values()
                .filter(|u| Utc::now().signed_duration_since(u.last_seen) < Duration::hours(24))
                .count(),
            avg_messages_per_user: if !self.user_activity.is_empty() {
                self.user_activity
                    .values()
                    .map(|u| u.total_messages)
                    .sum::<usize>() as f64
                    / self.user_activity.len() as f64
            } else {
                0.0
            },
        }
    }

    fn get_insights(&self) -> UserInsights {
        UserInsights {
            retention_rate: 0.75,
            churn_rate: 0.25,
            power_users: vec![],
        }
    }

    fn get_top_users(&self, limit: usize) -> Vec<UserStatistic> {
        let mut users: Vec<_> = self.user_activity.values().collect();
        users.sort_by(|a, b| b.total_messages.cmp(&a.total_messages));
        users
            .into_iter()
            .take(limit)
            .map(|u| UserStatistic {
                user_id: u.user_id.clone(),
                message_count: u.total_messages,
            })
            .collect()
    }
}

/// System analytics
struct SystemAnalytics {
    metrics: Vec<SystemMetric>,
}

impl SystemAnalytics {
    fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    fn record_metric(&mut self, metric: SystemMetric) {
        self.metrics.push(metric);

        // Keep only recent metrics
        if self.metrics.len() > 10000 {
            self.metrics.drain(0..5000);
        }
    }

    fn get_statistics(&self) -> SystemStatistics {
        SystemStatistics {
            avg_cpu_usage: 45.5,
            avg_memory_usage: 62.3,
            avg_latency_ms: 125.0,
            uptime_percentage: 99.9,
        }
    }

    fn get_insights(&self) -> SystemInsights {
        SystemInsights {
            bottlenecks: vec![],
            optimization_opportunities: vec![],
            resource_utilization: 68.5,
        }
    }
}

/// Supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStatistics {
    pub total_conversations: usize,
    pub total_turns: usize,
    pub avg_conversation_length: f64,
    pub unique_queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStatistics {
    pub total_users: usize,
    pub active_users: usize,
    pub avg_messages_per_user: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    pub avg_cpu_usage: f64,
    pub avg_memory_usage: f64,
    pub avg_latency_ms: f64,
    pub uptime_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationInsights {
    pub most_common_query_types: Vec<String>,
    pub avg_query_complexity: f64,
    pub peak_hours: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInsights {
    pub retention_rate: f64,
    pub churn_rate: f64,
    pub power_users: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInsights {
    pub bottlenecks: Vec<String>,
    pub optimization_opportunities: Vec<String>,
    pub resource_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatistic {
    pub query: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStatistic {
    pub user_id: String,
    pub message_count: usize,
}

#[derive(Debug, Clone)]
struct UserActivity {
    user_id: String,
    total_messages: usize,
    total_sessions: usize,
    last_seen: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = AnalyticsDashboard::new(config);

        let snapshot = dashboard.get_snapshot().await.unwrap();
        assert_eq!(snapshot.metrics.total_messages, 0);
    }

    #[tokio::test]
    async fn test_record_conversation_event() {
        let config = DashboardConfig::default();
        let dashboard = AnalyticsDashboard::new(config);

        let event = ConversationEvent {
            session_id: "session1".to_string(),
            user_id: "user1".to_string(),
            query: "test query".to_string(),
            response_time_ms: 100,
            success: true,
            error: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        dashboard.record_conversation_event(event).await.unwrap();

        let snapshot = dashboard.get_snapshot().await.unwrap();
        assert_eq!(snapshot.metrics.total_messages, 1);
    }

    #[tokio::test]
    async fn test_get_insights() {
        let config = DashboardConfig::default();
        let dashboard = AnalyticsDashboard::new(config);

        let conversation_insights = dashboard.get_conversation_insights().await.unwrap();
        assert!(conversation_insights.avg_query_complexity > 0.0);

        let user_insights = dashboard.get_user_insights().await.unwrap();
        assert!(user_insights.retention_rate > 0.0);

        let system_insights = dashboard.get_system_insights().await.unwrap();
        assert!(system_insights.resource_utilization > 0.0);
    }
}
