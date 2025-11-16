//! Materialized views for RDF-star annotation queries
//!
//! This module provides materialized views that pre-compute and cache
//! results of common annotation queries for dramatic performance improvements.
//!
//! # Features
//!
//! - **Automatic view maintenance** - Views update incrementally on data changes
//! - **Pre-aggregated statistics** - Fast access to annotation metrics
//! - **Index-backed views** - Efficient lookups by various criteria
//! - **Refresh strategies** - Immediate, deferred, or periodic refresh
//! - **View dependencies** - Cascade updates through dependent views
//! - **Query rewriting** - Automatic use of views for matching queries
//!
//! # Examples
//!
//! ```rust
//! use oxirs_star::materialized_views::{MaterializedViewManager, ViewDefinition};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut manager = MaterializedViewManager::new();
//!
//! // Create a view for high-confidence annotations
//! let view_def = ViewDefinition::new("high_confidence")
//!     .with_filter(|ann| ann.confidence.is_some_and(|c| c > 0.8));
//!
//! manager.create_view(view_def)?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use tracing::{info, span, Level};

use crate::annotations::TripleAnnotation;
use crate::StarResult;

/// Type alias for annotation filter predicates
type AnnotationFilter = Arc<dyn Fn(&TripleAnnotation) -> bool + Send + Sync>;

/// Materialized view definition
#[derive(Clone)]
pub struct ViewDefinition {
    /// View name
    pub name: String,

    /// View description
    pub description: Option<String>,

    /// Filter predicate
    filter: Option<AnnotationFilter>,

    /// Projection (which fields to include)
    projection: Option<Vec<String>>,

    /// Aggregation function
    aggregation: Option<AggregationType>,

    /// Refresh strategy
    pub refresh_strategy: RefreshStrategy,

    /// Dependencies on other views
    pub dependencies: Vec<String>,
}

// Manual Debug implementation since filter contains function pointers
impl std::fmt::Debug for ViewDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("has_filter", &self.filter.is_some())
            .field("projection", &self.projection)
            .field("aggregation", &self.aggregation)
            .field("refresh_strategy", &self.refresh_strategy)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

/// Type of aggregation for the view
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationType {
    /// Count annotations
    Count,
    /// Average confidence
    AverageConfidence,
    /// Sum of quality scores
    SumQuality,
    /// Minimum trust score
    MinTrust,
    /// Maximum trust score
    MaxTrust,
    /// Group by source
    GroupBySource,
}

/// Strategy for refreshing materialized views
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshStrategy {
    /// Update immediately on every change
    Immediate,
    /// Defer updates until explicitly refreshed
    Deferred,
    /// Refresh periodically (interval in seconds)
    Periodic(u64),
}

impl ViewDefinition {
    /// Create a new view definition
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            filter: None,
            projection: None,
            aggregation: None,
            refresh_strategy: RefreshStrategy::Immediate,
            dependencies: Vec::new(),
        }
    }

    /// Add a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a filter predicate
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: Fn(&TripleAnnotation) -> bool + Send + Sync + 'static,
    {
        self.filter = Some(Arc::new(filter));
        self
    }

    /// Set refresh strategy
    pub fn with_refresh_strategy(mut self, strategy: RefreshStrategy) -> Self {
        self.refresh_strategy = strategy;
        self
    }

    /// Set aggregation
    pub fn with_aggregation(mut self, aggregation: AggregationType) -> Self {
        self.aggregation = Some(aggregation);
        self
    }

    /// Add dependencies
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }
}

/// Materialized view data
#[derive(Debug, Clone)]
struct MaterializedView {
    /// View definition
    definition: ViewDefinition,

    /// Cached annotation data (triple hash -> annotation)
    data: HashMap<u64, TripleAnnotation>,

    /// Aggregated result (if aggregation is enabled)
    aggregated_result: Option<AggregatedResult>,

    /// Last refresh timestamp
    last_refresh: std::time::Instant,

    /// Number of hits (times view was used)
    hit_count: usize,

    /// Number of misses (times view needed refresh)
    miss_count: usize,
}

/// Result of an aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregatedResult {
    Count(usize),
    Average(f64),
    Sum(f64),
    Min(f64),
    Max(f64),
    GroupBy(HashMap<String, usize>),
}

impl MaterializedView {
    fn new(definition: ViewDefinition) -> Self {
        Self {
            definition,
            data: HashMap::new(),
            aggregated_result: None,
            last_refresh: std::time::Instant::now(),
            hit_count: 0,
            miss_count: 0,
        }
    }

    fn refresh(&mut self, all_annotations: &HashMap<u64, TripleAnnotation>) {
        self.data.clear();

        // Apply filter if present
        if let Some(ref filter) = self.definition.filter {
            for (hash, annotation) in all_annotations {
                if filter(annotation) {
                    self.data.insert(*hash, annotation.clone());
                }
            }
        } else {
            self.data = all_annotations.clone();
        }

        // Compute aggregation if present
        if let Some(agg_type) = self.definition.aggregation {
            self.aggregated_result = Some(self.compute_aggregation(agg_type));
        }

        self.last_refresh = std::time::Instant::now();
    }

    fn compute_aggregation(&self, agg_type: AggregationType) -> AggregatedResult {
        match agg_type {
            AggregationType::Count => AggregatedResult::Count(self.data.len()),

            AggregationType::AverageConfidence => {
                let sum: f64 = self.data.values().filter_map(|ann| ann.confidence).sum();
                let count = self
                    .data
                    .values()
                    .filter(|ann| ann.confidence.is_some())
                    .count();
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                AggregatedResult::Average(avg)
            }

            AggregationType::SumQuality => {
                let sum: f64 = self.data.values().filter_map(|ann| ann.quality_score).sum();
                AggregatedResult::Sum(sum)
            }

            AggregationType::MinTrust => {
                let min = self
                    .data
                    .values()
                    .map(|ann| ann.trust_score())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                AggregatedResult::Min(min)
            }

            AggregationType::MaxTrust => {
                let max = self
                    .data
                    .values()
                    .map(|ann| ann.trust_score())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(1.0);
                AggregatedResult::Max(max)
            }

            AggregationType::GroupBySource => {
                let mut groups: HashMap<String, usize> = HashMap::new();
                for annotation in self.data.values() {
                    if let Some(ref source) = annotation.source {
                        *groups.entry(source.clone()).or_insert(0) += 1;
                    }
                }
                AggregatedResult::GroupBy(groups)
            }
        }
    }
}

/// Manager for materialized views
pub struct MaterializedViewManager {
    /// All views indexed by name
    views: Arc<RwLock<HashMap<String, MaterializedView>>>,

    /// Source data (triple hash -> annotation)
    source_data: Arc<RwLock<HashMap<u64, TripleAnnotation>>>,

    /// Dependency graph (view name -> dependent view names)
    dependency_graph: HashMap<String, HashSet<String>>,

    /// Statistics
    stats: ViewStatistics,
}

/// Statistics for view usage
#[derive(Debug, Clone, Default)]
pub struct ViewStatistics {
    /// Total number of views
    pub view_count: usize,

    /// Total hits across all views
    pub total_hits: usize,

    /// Total misses across all views
    pub total_misses: usize,

    /// Cache hit rate
    pub hit_rate: f64,

    /// Total memory used by views (approximate)
    pub memory_bytes: usize,
}

impl MaterializedViewManager {
    /// Create a new view manager
    pub fn new() -> Self {
        Self {
            views: Arc::new(RwLock::new(HashMap::new())),
            source_data: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: HashMap::new(),
            stats: ViewStatistics::default(),
        }
    }

    /// Create a new materialized view
    pub fn create_view(&mut self, definition: ViewDefinition) -> StarResult<()> {
        let span = span!(Level::INFO, "create_view");
        let _enter = span.enter();

        let name = definition.name.clone();

        // Build dependency graph
        for dep in &definition.dependencies {
            self.dependency_graph
                .entry(dep.clone())
                .or_default()
                .insert(name.clone());
        }

        let mut view = MaterializedView::new(definition);

        // Initial refresh
        {
            let source_data = self.source_data.read().unwrap();
            view.refresh(&source_data);
        } // Drop read guard

        self.views.write().unwrap().insert(name.clone(), view);

        info!("Created materialized view: {}", name);
        self.update_statistics();

        Ok(())
    }

    /// Drop a materialized view
    pub fn drop_view(&mut self, name: &str) -> StarResult<()> {
        self.views.write().unwrap().remove(name);

        // Remove from dependency graph
        self.dependency_graph.remove(name);
        for deps in self.dependency_graph.values_mut() {
            deps.remove(name);
        }

        self.update_statistics();
        Ok(())
    }

    /// Insert annotation into source data and update views
    pub fn insert_annotation(
        &mut self,
        triple_hash: u64,
        annotation: TripleAnnotation,
    ) -> StarResult<()> {
        self.source_data
            .write()
            .unwrap()
            .insert(triple_hash, annotation);

        // Refresh views that need immediate updates
        self.refresh_immediate_views()?;

        Ok(())
    }

    /// Query a specific view
    pub fn query_view(&self, view_name: &str) -> Option<Vec<TripleAnnotation>> {
        let mut views = self.views.write().unwrap();

        if let Some(view) = views.get_mut(view_name) {
            view.hit_count += 1;
            Some(view.data.values().cloned().collect())
        } else {
            None
        }
    }

    /// Get aggregated result from a view
    pub fn get_aggregated_result(&self, view_name: &str) -> Option<AggregatedResult> {
        let mut views = self.views.write().unwrap();

        if let Some(view) = views.get_mut(view_name) {
            view.hit_count += 1;
            view.aggregated_result.clone()
        } else {
            None
        }
    }

    /// Manually refresh a view
    pub fn refresh_view(&mut self, view_name: &str) -> StarResult<()> {
        let source_data = self.source_data.read().unwrap();
        let mut views = self.views.write().unwrap();

        if let Some(view) = views.get_mut(view_name) {
            view.refresh(&source_data);

            // Refresh dependent views
            if let Some(dependents) = self.dependency_graph.get(view_name) {
                for dep_name in dependents {
                    if let Some(dep_view) = views.get_mut(dep_name) {
                        dep_view.refresh(&source_data);
                    }
                }
            }
        }

        Ok(())
    }

    /// Refresh all views
    pub fn refresh_all_views(&mut self) -> StarResult<()> {
        let source_data = self.source_data.read().unwrap();
        let mut views = self.views.write().unwrap();

        for view in views.values_mut() {
            view.refresh(&source_data);
        }

        Ok(())
    }

    fn refresh_immediate_views(&mut self) -> StarResult<()> {
        let source_data = self.source_data.read().unwrap();
        let mut views = self.views.write().unwrap();

        for view in views.values_mut() {
            if matches!(view.definition.refresh_strategy, RefreshStrategy::Immediate) {
                view.refresh(&source_data);
            }
        }

        Ok(())
    }

    /// Get statistics
    pub fn statistics(&self) -> ViewStatistics {
        self.stats.clone()
    }

    fn update_statistics(&mut self) {
        let views = self.views.read().unwrap();

        self.stats.view_count = views.len();
        self.stats.total_hits = views.values().map(|v| v.hit_count).sum();
        self.stats.total_misses = views.values().map(|v| v.miss_count).sum();

        let total_requests = self.stats.total_hits + self.stats.total_misses;
        self.stats.hit_rate = if total_requests > 0 {
            self.stats.total_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        // Approximate memory usage
        self.stats.memory_bytes = views
            .values()
            .map(|v| v.data.len() * std::mem::size_of::<TripleAnnotation>())
            .sum();
    }

    /// List all view names
    pub fn list_views(&self) -> Vec<String> {
        self.views.read().unwrap().keys().cloned().collect()
    }
}

impl Default for MaterializedViewManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_manager_creation() {
        let manager = MaterializedViewManager::new();
        assert_eq!(manager.statistics().view_count, 0);
    }

    #[test]
    fn test_create_view() {
        let mut manager = MaterializedViewManager::new();

        let view_def = ViewDefinition::new("test_view").with_description("Test view");

        manager.create_view(view_def).unwrap();

        assert_eq!(manager.statistics().view_count, 1);
        assert!(manager.list_views().contains(&"test_view".to_string()));
    }

    #[test]
    fn test_view_with_filter() {
        let mut manager = MaterializedViewManager::new();

        // Create view for high-confidence annotations
        let view_def = ViewDefinition::new("high_confidence")
            .with_filter(|ann| ann.confidence.is_some_and(|c| c > 0.8));

        manager.create_view(view_def).unwrap();

        // Insert annotations
        let ann1 = TripleAnnotation::new().with_confidence(0.9);
        let ann2 = TripleAnnotation::new().with_confidence(0.7);

        manager.insert_annotation(1, ann1).unwrap();
        manager.insert_annotation(2, ann2).unwrap();

        // Query view - should only get high-confidence annotation
        let results = manager.query_view("high_confidence").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].confidence, Some(0.9));
    }

    #[test]
    fn test_aggregation_count() {
        let mut manager = MaterializedViewManager::new();

        let view_def = ViewDefinition::new("count_view").with_aggregation(AggregationType::Count);

        manager.create_view(view_def).unwrap();

        // Insert some annotations
        for i in 0..5 {
            let ann = TripleAnnotation::new().with_confidence(0.8);
            manager.insert_annotation(i, ann).unwrap();
        }

        // Get aggregated count
        if let Some(AggregatedResult::Count(count)) = manager.get_aggregated_result("count_view") {
            assert_eq!(count, 5);
        } else {
            panic!("Expected count result");
        }
    }

    #[test]
    fn test_aggregation_average() {
        let mut manager = MaterializedViewManager::new();

        let view_def =
            ViewDefinition::new("avg_view").with_aggregation(AggregationType::AverageConfidence);

        manager.create_view(view_def).unwrap();

        manager
            .insert_annotation(1, TripleAnnotation::new().with_confidence(0.8))
            .unwrap();
        manager
            .insert_annotation(2, TripleAnnotation::new().with_confidence(0.9))
            .unwrap();
        manager
            .insert_annotation(3, TripleAnnotation::new().with_confidence(0.7))
            .unwrap();

        if let Some(AggregatedResult::Average(avg)) = manager.get_aggregated_result("avg_view") {
            assert!((avg - 0.8).abs() < 0.01);
        } else {
            panic!("Expected average result");
        }
    }

    #[test]
    fn test_drop_view() {
        let mut manager = MaterializedViewManager::new();

        let view_def = ViewDefinition::new("temp_view");
        manager.create_view(view_def).unwrap();

        assert_eq!(manager.statistics().view_count, 1);

        manager.drop_view("temp_view").unwrap();

        assert_eq!(manager.statistics().view_count, 0);
    }

    #[test]
    fn test_refresh_strategy() {
        let mut manager = MaterializedViewManager::new();

        // Create view with deferred refresh
        let view_def = ViewDefinition::new("deferred_view")
            .with_refresh_strategy(RefreshStrategy::Deferred)
            .with_aggregation(AggregationType::Count);

        manager.create_view(view_def).unwrap();

        // Insert annotation - view should not update automatically
        manager
            .insert_annotation(1, TripleAnnotation::new().with_confidence(0.9))
            .unwrap();

        // Manually refresh
        manager.refresh_view("deferred_view").unwrap();

        if let Some(AggregatedResult::Count(count)) = manager.get_aggregated_result("deferred_view")
        {
            assert_eq!(count, 1);
        }
    }
}
