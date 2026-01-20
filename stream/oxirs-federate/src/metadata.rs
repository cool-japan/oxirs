//! Extended metadata management for federated services
//!
//! This module provides comprehensive metadata structures for tracking service
//! capabilities, performance characteristics, and operational metrics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::service::ServiceMetadata;

/// Extended service metadata with comprehensive tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedServiceMetadata {
    /// Basic metadata (compatible with existing ServiceMetadata)
    pub basic: ServiceMetadata,

    /// Service Level Agreement information
    pub sla: ServiceSLA,

    /// Dataset statistics
    pub dataset_stats: DatasetStatistics,

    /// Query pattern examples
    pub query_patterns: Vec<QueryPattern>,

    /// Extended capability descriptions
    pub capability_details: HashMap<String, CapabilityDetail>,

    /// Health metrics history
    pub health_metrics: HealthMetrics,

    /// Service dependencies
    pub dependencies: Vec<ServiceDependency>,

    /// Known vocabularies supported by this service
    pub known_vocabularies: Option<Vec<String>>,

    /// Last metadata update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Service Level Agreement information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServiceSLA {
    /// Guaranteed uptime percentage (e.g., 99.9)
    pub uptime_guarantee: Option<f64>,

    /// Response time targets
    pub response_time_targets: ResponseTimeTargets,

    /// Rate limits
    pub rate_limits: RateLimitInfo,

    /// Maintenance windows
    pub maintenance_windows: Vec<MaintenanceWindow>,

    /// Support contact information
    pub support_contact: Option<String>,
}

/// Response time targets for different query types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeTargets {
    /// Target for simple queries (e.g., < 100ms)
    pub simple_query_p95: Duration,

    /// Target for medium complexity queries
    pub medium_query_p95: Duration,

    /// Target for complex queries
    pub complex_query_p95: Duration,

    /// Maximum allowed response time
    pub max_response_time: Duration,
}

/// Rate limit information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RateLimitInfo {
    /// Requests per minute limit
    pub requests_per_minute: Option<usize>,

    /// Requests per hour limit
    pub requests_per_hour: Option<usize>,

    /// Concurrent request limit
    pub concurrent_requests: Option<usize>,

    /// Burst allowance
    pub burst_size: Option<usize>,

    /// Query complexity limit
    pub max_query_complexity: Option<usize>,
}

/// Maintenance window information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    /// Day of week (0 = Sunday, 6 = Saturday)
    pub day_of_week: u8,

    /// Start hour (0-23)
    pub start_hour: u8,

    /// Duration
    pub duration: Duration,

    /// Timezone (e.g., "UTC", "US/Pacific")
    pub timezone: String,
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatasetStatistics {
    /// Total number of triples
    pub triple_count: Option<u64>,

    /// Total number of unique subjects
    pub subject_count: Option<u64>,

    /// Total number of unique predicates
    pub predicate_count: Option<u64>,

    /// Total number of unique objects
    pub object_count: Option<u64>,

    /// Dataset size in bytes
    pub size_bytes: Option<u64>,

    /// Last update timestamp
    pub last_modified: Option<DateTime<Utc>>,

    /// Named graphs available
    pub named_graphs: Vec<NamedGraphInfo>,

    /// Vocabulary/ontology URIs used
    pub vocabularies: Vec<String>,

    /// Language tags present in literals
    pub languages: Vec<String>,
}

/// Named graph information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedGraphInfo {
    /// Graph URI
    pub uri: String,

    /// Number of triples in this graph
    pub triple_count: Option<u64>,

    /// Description
    pub description: Option<String>,
}

/// Query pattern example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Pattern name/identifier
    pub name: String,

    /// Description of what this pattern does
    pub description: String,

    /// Example query
    pub example_query: String,

    /// Expected response time
    pub expected_response_time: Option<Duration>,

    /// Query complexity score
    pub complexity_score: Option<u32>,

    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Detailed capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityDetail {
    /// Capability name
    pub name: String,

    /// Detailed description
    pub description: String,

    /// Version or specification level supported
    pub version: Option<String>,

    /// Limitations or restrictions
    pub limitations: Vec<String>,

    /// Performance characteristics
    pub performance_notes: Option<String>,
}

/// Health metrics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Current health score (0-100)
    pub health_score: f64,

    /// Recent availability percentage
    pub availability_percentage: f64,

    /// Average response time over last period
    pub avg_response_time: Duration,

    /// Error rate (errors per 1000 requests)
    pub error_rate: f64,

    /// Recent health check results
    pub recent_checks: Vec<HealthCheckResult>,

    /// Performance trend (improving, stable, degrading)
    pub performance_trend: PerformanceTrend,
}

/// Individual health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Timestamp of the check
    pub timestamp: DateTime<Utc>,

    /// Was the check successful
    pub success: bool,

    /// Response time
    pub response_time: Option<Duration>,

    /// Error message if failed
    pub error_message: Option<String>,
}

/// Performance trend indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Service dependency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDependency {
    /// Dependency service ID or URL
    pub service_id: String,

    /// Type of dependency
    pub dependency_type: DependencyType,

    /// Is this a critical dependency
    pub is_critical: bool,

    /// Description
    pub description: Option<String>,
}

/// Type of service dependency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Data source dependency
    DataSource,
    /// Authentication provider
    Authentication,
    /// Schema/vocabulary provider
    Schema,
    /// Other service
    Other,
}

impl ExtendedServiceMetadata {
    /// Create new extended metadata from basic metadata
    pub fn from_basic(basic: ServiceMetadata) -> Self {
        Self {
            basic,
            sla: ServiceSLA::default(),
            dataset_stats: DatasetStatistics::default(),
            query_patterns: Vec::new(),
            capability_details: HashMap::new(),
            health_metrics: HealthMetrics::default(),
            dependencies: Vec::new(),
            known_vocabularies: None,
            last_updated: Utc::now(),
        }
    }

    /// Update health metrics with new check result
    pub fn update_health(&mut self, result: HealthCheckResult) {
        self.health_metrics.recent_checks.push(result);

        // Keep only last 100 checks
        if self.health_metrics.recent_checks.len() > 100 {
            self.health_metrics.recent_checks.remove(0);
        }

        // Recalculate metrics
        self.recalculate_health_metrics();
    }

    /// Recalculate health metrics from recent checks
    fn recalculate_health_metrics(&mut self) {
        let recent = &self.health_metrics.recent_checks;
        if recent.is_empty() {
            return;
        }

        let successful = recent.iter().filter(|c| c.success).count();
        let total = recent.len();

        self.health_metrics.availability_percentage = (successful as f64 / total as f64) * 100.0;

        let response_times: Vec<Duration> = recent.iter().filter_map(|c| c.response_time).collect();

        if !response_times.is_empty() {
            let total_millis: u64 = response_times.iter().map(|d| d.as_millis() as u64).sum();
            let avg_millis = total_millis / response_times.len() as u64;
            self.health_metrics.avg_response_time = Duration::from_millis(avg_millis);
        }

        let errors = total - successful;
        self.health_metrics.error_rate = (errors as f64 / total as f64) * 1000.0;

        // Calculate health score (simple weighted average)
        self.health_metrics.health_score = (self.health_metrics.availability_percentage * 0.5)
            + ((100.0 - self.health_metrics.error_rate.min(100.0)) * 0.3)
            + ((1000.0 / (self.health_metrics.avg_response_time.as_millis() as f64 + 1.0))
                .min(100.0)
                * 0.2);

        // Determine trend
        if recent.len() >= 10 {
            let recent_half = recent.len() / 2;
            let first_half_success = recent[..recent_half].iter().filter(|c| c.success).count();
            let second_half_success = recent[recent_half..].iter().filter(|c| c.success).count();

            if second_half_success > first_half_success + 2 {
                self.health_metrics.performance_trend = PerformanceTrend::Improving;
            } else if first_half_success > second_half_success + 2 {
                self.health_metrics.performance_trend = PerformanceTrend::Degrading;
            } else {
                self.health_metrics.performance_trend = PerformanceTrend::Stable;
            }
        }
    }
}

impl Default for ResponseTimeTargets {
    fn default() -> Self {
        Self {
            simple_query_p95: Duration::from_millis(100),
            medium_query_p95: Duration::from_millis(500),
            complex_query_p95: Duration::from_secs(2),
            max_response_time: Duration::from_secs(30),
        }
    }
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            health_score: 100.0,
            availability_percentage: 100.0,
            avg_response_time: Duration::from_millis(0),
            error_rate: 0.0,
            recent_checks: Vec::new(),
            performance_trend: PerformanceTrend::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_metadata_creation() {
        let basic = ServiceMetadata::default();
        let extended = ExtendedServiceMetadata::from_basic(basic);

        assert_eq!(extended.health_metrics.health_score, 100.0);
        assert!(extended.query_patterns.is_empty());
    }

    #[test]
    fn test_health_metrics_update() {
        let mut extended = ExtendedServiceMetadata::from_basic(ServiceMetadata::default());

        // Add successful check
        extended.update_health(HealthCheckResult {
            timestamp: Utc::now(),
            success: true,
            response_time: Some(Duration::from_millis(50)),
            error_message: None,
        });

        assert_eq!(extended.health_metrics.availability_percentage, 100.0);
        assert_eq!(extended.health_metrics.error_rate, 0.0);

        // Add failed check
        extended.update_health(HealthCheckResult {
            timestamp: Utc::now(),
            success: false,
            response_time: None,
            error_message: Some("Connection timeout".to_string()),
        });

        assert_eq!(extended.health_metrics.availability_percentage, 50.0);
        assert_eq!(extended.health_metrics.error_rate, 500.0);
    }
}
