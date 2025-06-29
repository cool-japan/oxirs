//! Type definitions for materialized views
//!
//! This module contains all the core data structures used in the materialized views system.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::{
    planner::planning::{FilterExpression, QueryComplexity, TriplePattern},
    FederatedService, ServiceCapability,
};

/// Configuration for materialized view management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializedViewConfig {
    pub max_views: usize,
    pub default_refresh_interval: Duration,
    pub enable_automatic_maintenance: bool,
    pub freshness_threshold_hours: u64,
    pub max_view_size_bytes: u64,
    pub enable_incremental_refresh: bool,
}

impl Default for MaterializedViewConfig {
    fn default() -> Self {
        Self {
            max_views: 100,
            default_refresh_interval: Duration::from_secs(3600), // 1 hour
            enable_automatic_maintenance: true,
            freshness_threshold_hours: 24,
            max_view_size_bytes: 1024 * 1024 * 1024, // 1GB
            enable_incremental_refresh: true,
        }
    }
}

/// Materialized view definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewDefinition {
    pub name: String,
    pub description: Option<String>,
    pub source_patterns: Vec<ServicePattern>,
    pub query: String,
    pub refresh_interval: Option<Duration>,
    pub supports_incremental: bool,
    pub partitioning_key: Option<String>,
    pub dependencies: Vec<String>,
}

impl ViewDefinition {
    /// Get query patterns from the view definition
    pub fn query_patterns(&self) -> Vec<TriplePattern> {
        // Extract patterns from the source patterns
        self.source_patterns
            .iter()
            .flat_map(|sp| sp.patterns.clone())
            .collect()
    }

    /// Get filter expressions from the view definition
    pub fn filters(&self) -> Vec<FilterExpression> {
        // Extract filters from source patterns
        self.source_patterns
            .iter()
            .flat_map(|sp| sp.filters.clone())
            .collect()
    }

    /// Check if the view can support the given query patterns
    pub fn supports_patterns(&self, query_patterns: &[TriplePattern]) -> bool {
        let view_patterns = self.query_patterns();
        query_patterns
            .iter()
            .all(|qp| view_patterns.iter().any(|vp| patterns_match(qp, vp)))
    }

    /// Estimate the freshness requirement for this view
    pub fn estimate_freshness_requirement(&self) -> Duration {
        self.refresh_interval.unwrap_or(Duration::from_secs(3600))
    }

    /// Get the complexity score for this view
    pub fn complexity_score(&self) -> f64 {
        let pattern_count = self.query_patterns().len();
        let filter_count = self.filters().len();
        let dependency_count = self.dependencies.len();

        (pattern_count * 2 + filter_count * 3 + dependency_count * 4) as f64
    }
}

/// Pattern matching helper function
fn patterns_match(query_pattern: &TriplePattern, view_pattern: &TriplePattern) -> bool {
    // Simple pattern matching - in practice this would be more sophisticated
    (query_pattern.subject.is_none() || query_pattern.subject == view_pattern.subject)
        && (query_pattern.predicate.is_none() || query_pattern.predicate == view_pattern.predicate)
        && (query_pattern.object.is_none() || query_pattern.object == view_pattern.object)
}

/// Service pattern for materialized views
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePattern {
    pub service_id: String,
    pub patterns: Vec<TriplePattern>,
    pub filters: Vec<FilterExpression>,
    pub estimated_selectivity: f64,
}

/// Materialized view instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializedView {
    pub id: String,
    pub definition: ViewDefinition,
    pub creation_time: DateTime<Utc>,
    pub last_refresh: Option<DateTime<Utc>>,
    pub size_bytes: u64,
    pub row_count: u64,
    pub is_stale: bool,
    pub refresh_in_progress: bool,
    pub error_count: u32,
    pub last_error: Option<String>,
    pub access_count: u64,
    pub last_access: Option<DateTime<Utc>>,
    pub data_location: ViewDataLocation,
}

/// Location of materialized view data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewDataLocation {
    Memory,
    Disk { path: String },
    Remote { url: String },
    Distributed { nodes: Vec<String> },
}

/// Statistics for a materialized view
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewStatistics {
    pub view_id: String,
    pub hit_count: u64,
    pub miss_count: u64,
    pub refresh_count: u64,
    pub avg_refresh_time: Duration,
    pub storage_efficiency: f64,
    pub query_coverage: f64,
    pub freshness_score: f64,
    pub cost_savings: f64,
    pub last_updated: DateTime<Utc>,
}

impl ViewStatistics {
    /// Create new statistics for a view
    pub fn new(view_id: String) -> Self {
        Self {
            view_id,
            hit_count: 0,
            miss_count: 0,
            refresh_count: 0,
            avg_refresh_time: Duration::from_secs(0),
            storage_efficiency: 0.0,
            query_coverage: 0.0,
            freshness_score: 1.0,
            cost_savings: 0.0,
            last_updated: Utc::now(),
        }
    }

    /// Calculate hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
        self.last_updated = Utc::now();
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.miss_count += 1;
        self.last_updated = Utc::now();
    }

    /// Record a refresh operation
    pub fn record_refresh(&mut self, duration: Duration) {
        self.refresh_count += 1;

        // Update rolling average
        if self.refresh_count == 1 {
            self.avg_refresh_time = duration;
        } else {
            let total_time = self.avg_refresh_time.as_secs_f64() * (self.refresh_count - 1) as f64
                + duration.as_secs_f64();
            self.avg_refresh_time = Duration::from_secs_f64(total_time / self.refresh_count as f64);
        }

        self.last_updated = Utc::now();
    }
}

/// Maintenance operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaintenanceOperation {
    Refresh,
    Cleanup,
    Optimize,
    Validate,
    Archive,
}

/// Maintenance schedule entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceSchedule {
    pub view_id: String,
    pub operation: MaintenanceOperation,
    pub scheduled_time: DateTime<Utc>,
    pub priority: MaintenancePriority,
    pub estimated_duration: Duration,
}

/// Priority levels for maintenance operations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaintenancePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Query optimization recommendation
#[derive(Debug, Clone)]
pub struct ViewRecommendation {
    pub view_id: String,
    pub reason: RecommendationReason,
    pub estimated_benefit: f64,
    pub implementation_cost: f64,
    pub confidence: f64,
}

/// Reasons for view recommendations
#[derive(Debug, Clone)]
pub enum RecommendationReason {
    HighQueryFrequency,
    ExpensiveJoins,
    SlowServiceResponse,
    DataLocalityBenefit,
    ReducedNetworkTraffic,
    ImprovedCacheHitRatio,
}

/// Change detection event
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    pub source_service: String,
    pub timestamp: DateTime<Utc>,
    pub change_type: ChangeType,
    pub affected_patterns: Vec<TriplePattern>,
    pub estimated_impact: f64,
}

/// Types of changes that can affect materialized views
#[derive(Debug, Clone)]
pub enum ChangeType {
    DataInsert,
    DataUpdate,
    DataDelete,
    SchemaChange,
    ServiceUnavailable,
}

/// Delta processing result
#[derive(Debug, Clone)]
pub struct DeltaResult {
    pub view_id: String,
    pub changes_applied: u64,
    pub processing_time: Duration,
    pub new_data_size: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupConfig {
    pub max_unused_days: u32,
    pub max_error_count: u32,
    pub min_hit_ratio: f64,
    pub max_size_bytes: u64,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            max_unused_days: 30,
            max_error_count: 10,
            min_hit_ratio: 0.1,
            max_size_bytes: 10 * 1024 * 1024 * 1024, // 10GB
        }
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub enable_syntax_validation: bool,
    pub enable_semantic_validation: bool,
    pub max_validation_time: Duration,
    pub strict_mode: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_syntax_validation: true,
            enable_semantic_validation: true,
            max_validation_time: Duration::from_secs(30),
            strict_mode: false,
        }
    }
}

/// Dependency relationship between views
#[derive(Debug, Clone)]
pub struct ViewDependency {
    pub dependent_view: String,
    pub dependency_view: String,
    pub dependency_type: DependencyType,
    pub strength: f64, // 0.0 to 1.0
}

/// Types of dependencies between materialized views
#[derive(Debug, Clone)]
pub enum DependencyType {
    DataDependency,
    TemporalDependency,
    ComputationalDependency,
    StorageDependency,
}

/// Pattern features for machine learning analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternFeatures {
    pub pattern_complexity: f64,
    pub selectivity: f64,
    pub join_complexity: f64,
    pub data_freshness_requirement: f64,
    pub access_frequency: f64,
    pub computational_cost: f64,
}

/// Temporal range for data coverage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRange {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub granularity: TemporalGranularity,
}

/// Granularity levels for temporal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalGranularity {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}
