//! Enhanced GraphQL Subscription System
//!
//! This module provides advanced subscription features including intelligent filtering,
//! subscription grouping, performance optimizations, and comprehensive monitoring.

use crate::ast::{Document, Field, Selection, SelectionSet, Value};
use crate::execution::ExecutionContext;
use crate::performance::PerformanceTracker;
use crate::subscriptions::{SubscriptionEvent, SubscriptionMessage, ActiveSubscription};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock as AsyncRwLock};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

/// Advanced subscription configuration
#[derive(Debug, Clone)]
pub struct EnhancedSubscriptionConfig {
    pub enable_intelligent_filtering: bool,
    pub enable_subscription_grouping: bool,
    pub enable_query_analysis: bool,
    pub enable_performance_monitoring: bool,
    pub max_subscription_groups: usize,
    pub group_execution_batch_size: usize,
    pub filter_cache_size: usize,
    pub execution_timeout: Duration,
    pub subscription_priority_levels: usize,
}

impl Default for EnhancedSubscriptionConfig {
    fn default() -> Self {
        Self {
            enable_intelligent_filtering: true,
            enable_subscription_grouping: true,
            enable_query_analysis: true,
            enable_performance_monitoring: true,
            max_subscription_groups: 100,
            group_execution_batch_size: 10,
            filter_cache_size: 1000,
            execution_timeout: Duration::from_secs(30),
            subscription_priority_levels: 3,
        }
    }
}

/// Enhanced subscription with detailed metadata
#[derive(Debug, Clone)]
pub struct EnhancedSubscription {
    pub base: ActiveSubscription,
    pub query_analysis: QueryAnalysis,
    pub priority: SubscriptionPriority,
    pub filter_hash: u64,
    pub group_id: Option<String>,
    pub performance_metrics: SubscriptionMetrics,
    pub resource_dependencies: HashSet<String>,
    pub client_info: ClientInfo,
}

/// Query analysis for intelligent filtering
#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    pub fields: HashSet<String>,
    pub predicates: HashSet<String>,
    pub subjects: HashSet<String>,
    pub filters: Vec<String>,
    pub complexity_score: usize,
    pub depth: usize,
    pub estimated_cost: f64,
}

/// Subscription priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SubscriptionPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Client information for enhanced tracking
#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub subscription_quota: Option<usize>,
    pub rate_limit: Option<Duration>,
}

impl Default for ClientInfo {
    fn default() -> Self {
        Self {
            user_id: None,
            session_id: None,
            ip_address: None,
            user_agent: None,
            subscription_quota: None,
            rate_limit: None,
        }
    }
}

/// Subscription performance metrics
#[derive(Debug, Clone, Default)]
pub struct SubscriptionMetrics {
    pub execution_count: u64,
    pub total_execution_time: Duration,
    pub avg_execution_time: Duration,
    pub max_execution_time: Duration,
    pub error_count: u64,
    pub result_size_avg: usize,
    pub last_executed: Option<Instant>,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl SubscriptionMetrics {
    pub fn record_execution(&mut self, duration: Duration, result_size: usize, had_error: bool) {
        self.execution_count += 1;
        self.total_execution_time += duration;
        self.max_execution_time = self.max_execution_time.max(duration);
        
        if self.execution_count > 0 {
            self.avg_execution_time = self.total_execution_time / self.execution_count as u32;
        }
        
        if had_error {
            self.error_count += 1;
        }
        
        // Update rolling average for result size
        self.result_size_avg = (self.result_size_avg + result_size) / 2;
        self.last_executed = Some(Instant::now());
    }

    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Subscription group for batch execution
#[derive(Debug, Clone)]
pub struct SubscriptionGroup {
    pub id: String,
    pub query_hash: u64,
    pub subscription_ids: HashSet<String>,
    pub query_analysis: QueryAnalysis,
    pub last_executed: Option<Instant>,
    pub execution_count: u64,
    pub created_at: Instant,
}

impl SubscriptionGroup {
    pub fn new(query_hash: u64, analysis: QueryAnalysis) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            query_hash,
            subscription_ids: HashSet::new(),
            query_analysis: analysis,
            last_executed: None,
            execution_count: 0,
            created_at: Instant::now(),
        }
    }

    pub fn add_subscription(&mut self, subscription_id: String) {
        self.subscription_ids.insert(subscription_id);
    }

    pub fn remove_subscription(&mut self, subscription_id: &str) -> bool {
        self.subscription_ids.remove(subscription_id)
    }

    pub fn is_empty(&self) -> bool {
        self.subscription_ids.is_empty()
    }
}

/// Enhanced event filtering with intelligent matching
#[derive(Debug, Clone)]
pub struct IntelligentFilter {
    resource_patterns: HashMap<String, Vec<String>>,
    predicate_patterns: HashMap<String, Vec<String>>,
    cached_matches: Arc<RwLock<HashMap<u64, bool>>>,
    config: EnhancedSubscriptionConfig,
}

impl IntelligentFilter {
    pub fn new(config: EnhancedSubscriptionConfig) -> Self {
        Self {
            resource_patterns: HashMap::new(),
            predicate_patterns: HashMap::new(),
            cached_matches: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Analyze subscription query to extract filtering patterns
    pub fn analyze_subscription_query(&self, document: &Document) -> Result<QueryAnalysis> {
        let mut analysis = QueryAnalysis {
            fields: HashSet::new(),
            predicates: HashSet::new(),
            subjects: HashSet::new(),
            filters: Vec::new(),
            complexity_score: 0,
            depth: 0,
            estimated_cost: 0.0,
        };

        for definition in &document.definitions {
            if let crate::ast::Definition::Operation(operation) = definition {
                self.analyze_selection_set(&operation.selection_set, &mut analysis, 1)?;
            }
        }

        // Calculate estimated cost based on analysis
        analysis.estimated_cost = self.calculate_query_cost(&analysis);

        Ok(analysis)
    }

    fn analyze_selection_set(
        &self,
        selection_set: &SelectionSet,
        analysis: &mut QueryAnalysis,
        depth: usize,
    ) -> Result<()> {
        analysis.depth = analysis.depth.max(depth);

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    self.analyze_field(field, analysis, depth)?;
                    
                    if let Some(nested_set) = &field.selection_set {
                        self.analyze_selection_set(nested_set, analysis, depth + 1)?;
                    }
                }
                Selection::InlineFragment(fragment) => {
                    self.analyze_selection_set(&fragment.selection_set, analysis, depth)?;
                }
                Selection::FragmentSpread(_) => {
                    analysis.complexity_score += 5;
                }
            }
        }

        Ok(())
    }

    fn analyze_field(&self, field: &Field, analysis: &mut QueryAnalysis, _depth: usize) -> Result<()> {
        analysis.fields.insert(field.name.clone());
        analysis.complexity_score += 1;

        // Extract predicates and subjects from field arguments
        for arg in &field.arguments {
            match arg.name.as_str() {
                "subject" | "id" => {
                    if let Value::StringValue(subject) = &arg.value {
                        analysis.subjects.insert(subject.clone());
                    }
                }
                "predicate" | "property" => {
                    if let Value::StringValue(predicate) = &arg.value {
                        analysis.predicates.insert(predicate.clone());
                    }
                }
                "filter" | "where" => {
                    if let Value::StringValue(filter) = &arg.value {
                        analysis.filters.push(filter.clone());
                        analysis.complexity_score += 3;
                    }
                }
                _ => {}
            }
        }

        // Analyze field name for common RDF patterns
        self.extract_rdf_patterns(field, analysis);

        Ok(())
    }

    fn extract_rdf_patterns(&self, field: &Field, analysis: &mut QueryAnalysis) {
        match field.name.as_str() {
            "name" => analysis.predicates.insert("http://xmlns.com/foaf/0.1/name".to_string()),
            "email" => analysis.predicates.insert("http://xmlns.com/foaf/0.1/mbox".to_string()),
            "knows" => analysis.predicates.insert("http://xmlns.com/foaf/0.1/knows".to_string()),
            "type" => analysis.predicates.insert("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()),
            "label" => analysis.predicates.insert("http://www.w3.org/2000/01/rdf-schema#label".to_string()),
            _ => false,
        };
    }

    fn calculate_query_cost(&self, analysis: &QueryAnalysis) -> f64 {
        let base_cost = 1.0;
        let field_cost = analysis.fields.len() as f64 * 0.5;
        let predicate_cost = analysis.predicates.len() as f64 * 1.0;
        let subject_cost = analysis.subjects.len() as f64 * 0.3;
        let filter_cost = analysis.filters.len() as f64 * 2.0;
        let depth_cost = analysis.depth as f64 * 1.5;
        let complexity_cost = analysis.complexity_score as f64 * 0.1;

        base_cost + field_cost + predicate_cost + subject_cost + filter_cost + depth_cost + complexity_cost
    }

    /// Check if a subscription should be triggered by an event
    pub fn should_trigger_subscription(
        &self,
        subscription: &EnhancedSubscription,
        event: &SubscriptionEvent,
    ) -> bool {
        if !self.config.enable_intelligent_filtering {
            return true; // Fall back to triggering all subscriptions
        }

        // Generate cache key for this subscription-event pair
        let cache_key = self.generate_filter_cache_key(subscription, event);

        // Check cache first
        if let Ok(cache) = self.cached_matches.read() {
            if let Some(&result) = cache.get(&cache_key) {
                return result;
            }
        }

        // Perform actual filtering logic
        let should_trigger = self.evaluate_trigger_condition(subscription, event);

        // Cache the result
        if let Ok(mut cache) = self.cached_matches.write() {
            // Enforce cache size limit
            if cache.len() >= self.config.filter_cache_size {
                cache.clear(); // Simple cache eviction
            }
            cache.insert(cache_key, should_trigger);
        }

        should_trigger
    }

    fn evaluate_trigger_condition(
        &self,
        subscription: &EnhancedSubscription,
        event: &SubscriptionEvent,
    ) -> bool {
        match event {
            SubscriptionEvent::BulkChange => true, // Always trigger for bulk changes
            
            SubscriptionEvent::TripleAdded { subject, predicate, object: _ }
            | SubscriptionEvent::TripleRemoved { subject, predicate, object: _ } => {
                // Check if subscription depends on this subject or predicate
                subscription.query_analysis.subjects.contains(subject)
                    || subscription.query_analysis.predicates.contains(predicate)
                    || subscription.resource_dependencies.contains(subject)
                    || subscription.resource_dependencies.contains(predicate)
            }
            
            SubscriptionEvent::SubjectChanged { subject } => {
                subscription.query_analysis.subjects.contains(subject)
                    || subscription.resource_dependencies.contains(subject)
            }
            
            SubscriptionEvent::PredicateChanged { predicate } => {
                subscription.query_analysis.predicates.contains(predicate)
                    || subscription.resource_dependencies.contains(predicate)
            }
        }
    }

    fn generate_filter_cache_key(&self, subscription: &EnhancedSubscription, event: &SubscriptionEvent) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        subscription.filter_hash.hash(&mut hasher);
        format!("{:?}", event).hash(&mut hasher);
        hasher.finish()
    }
}

/// Enhanced subscription manager with advanced features
pub struct EnhancedSubscriptionManager {
    config: EnhancedSubscriptionConfig,
    subscriptions: Arc<RwLock<HashMap<String, EnhancedSubscription>>>,
    subscription_groups: Arc<RwLock<HashMap<String, SubscriptionGroup>>>,
    priority_queues: Arc<RwLock<BTreeMap<SubscriptionPriority, Vec<String>>>>,
    intelligent_filter: Arc<IntelligentFilter>,
    performance_tracker: Option<Arc<PerformanceTracker>>,
    metrics: Arc<RwLock<EnhancedSubscriptionMetrics>>,
    event_sender: broadcast::Sender<SubscriptionEvent>,
}

/// Enhanced subscription metrics
#[derive(Debug, Clone, Default)]
pub struct EnhancedSubscriptionMetrics {
    pub total_subscriptions: usize,
    pub subscriptions_by_priority: HashMap<SubscriptionPriority, usize>,
    pub total_groups: usize,
    pub avg_group_size: f64,
    pub filter_cache_hit_ratio: f64,
    pub total_events_processed: u64,
    pub events_filtered_out: u64,
    pub avg_execution_time_by_priority: HashMap<SubscriptionPriority, Duration>,
    pub subscription_distribution: HashMap<String, usize>, // by resource type
}

impl EnhancedSubscriptionManager {
    pub fn new(
        config: EnhancedSubscriptionConfig,
        performance_tracker: Option<Arc<PerformanceTracker>>,
    ) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        let intelligent_filter = Arc::new(IntelligentFilter::new(config.clone()));

        Self {
            config: config.clone(),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            subscription_groups: Arc::new(RwLock::new(HashMap::new())),
            priority_queues: Arc::new(RwLock::new(BTreeMap::new())),
            intelligent_filter,
            performance_tracker,
            metrics: Arc::new(RwLock::new(EnhancedSubscriptionMetrics::default())),
            event_sender,
        }
    }

    /// Add an enhanced subscription
    pub fn add_subscription(
        &self,
        base_subscription: ActiveSubscription,
        document: &Document,
        client_info: ClientInfo,
        priority: SubscriptionPriority,
    ) -> Result<()> {
        // Analyze the query
        let query_analysis = self.intelligent_filter.analyze_subscription_query(document)?;
        
        // Calculate filter hash for grouping
        let filter_hash = self.calculate_filter_hash(&query_analysis);
        
        // Extract resource dependencies
        let resource_dependencies = self.extract_resource_dependencies(&query_analysis);

        let enhanced_subscription = EnhancedSubscription {
            base: base_subscription.clone(),
            query_analysis: query_analysis.clone(),
            priority,
            filter_hash,
            group_id: None,
            performance_metrics: SubscriptionMetrics::default(),
            resource_dependencies,
            client_info,
        };

        // Add to appropriate group if grouping is enabled
        let group_id = if self.config.enable_subscription_grouping {
            self.add_to_group(&enhanced_subscription)?
        } else {
            None
        };

        // Update group_id in subscription
        let mut final_subscription = enhanced_subscription;
        final_subscription.group_id = group_id;

        // Add to subscriptions map
        {
            let mut subscriptions = self.subscriptions.write().expect("lock poisoned");
            subscriptions.insert(base_subscription.id.clone(), final_subscription);
        }

        // Add to priority queue
        {
            let mut priority_queues = self.priority_queues.write().expect("lock poisoned");
            priority_queues
                .entry(priority)
                .or_insert_with(Vec::new)
                .push(base_subscription.id.clone());
        }

        // Update metrics
        self.update_metrics_on_add(priority);

        info!(
            "Added enhanced subscription {} with priority {:?}",
            base_subscription.id, priority
        );

        Ok(())
    }

    /// Remove a subscription
    pub fn remove_subscription(&self, subscription_id: &str) -> Result<()> {
        let removed_subscription = {
            let mut subscriptions = self.subscriptions.write().expect("lock poisoned");
            subscriptions.remove(subscription_id)
        };

        if let Some(subscription) = removed_subscription {
            // Remove from group if applicable
            if let Some(group_id) = &subscription.group_id {
                self.remove_from_group(group_id, subscription_id)?;
            }

            // Remove from priority queue
            {
                let mut priority_queues = self.priority_queues.write().expect("lock poisoned");
                if let Some(queue) = priority_queues.get_mut(&subscription.priority) {
                    queue.retain(|id| id != subscription_id);
                }
            }

            // Update metrics
            self.update_metrics_on_remove(subscription.priority);

            info!("Removed enhanced subscription {}", subscription_id);
        }

        Ok(())
    }

    /// Process a subscription event with intelligent filtering
    pub async fn process_event(&self, event: SubscriptionEvent) -> Result<()> {
        let start_time = Instant::now();
        
        // Update event metrics
        {
            let mut metrics = self.metrics.write().expect("lock poisoned");
            metrics.total_events_processed += 1;
        }

        // Get all subscriptions that should be triggered
        let triggered_subscriptions = self.get_triggered_subscriptions(&event);
        
        if triggered_subscriptions.is_empty() {
            let mut metrics = self.metrics.write().expect("lock poisoned");
            metrics.events_filtered_out += 1;
            return Ok(());
        }

        // Group subscriptions by priority for execution
        let mut priority_groups: BTreeMap<SubscriptionPriority, Vec<String>> = BTreeMap::new();
        
        for subscription_id in triggered_subscriptions {
            let priority = {
                let subscriptions = self.subscriptions.read().expect("lock poisoned");
                subscriptions.get(&subscription_id)
                    .map(|s| s.priority)
                    .unwrap_or(SubscriptionPriority::Normal)
            };
            
            priority_groups
                .entry(priority)
                .or_insert_with(Vec::new)
                .push(subscription_id);
        }

        // Execute subscriptions by priority (highest first)
        for (priority, subscription_ids) in priority_groups.into_iter().rev() {
            self.execute_subscriptions_batch(subscription_ids, priority).await?;
        }

        // Record processing time
        if let Some(ref tracker) = self.performance_tracker {
            tracker.record_field_resolution(
                "subscription_event_processing",
                start_time.elapsed(),
                false,
            );
        }

        debug!(
            "Processed subscription event in {:?}",
            start_time.elapsed()
        );

        Ok(())
    }

    /// Get comprehensive metrics
    pub fn get_enhanced_metrics(&self) -> EnhancedSubscriptionMetrics {
        let mut metrics = self.metrics.read().expect("lock poisoned").clone();
        
        // Update real-time metrics
        let subscriptions = self.subscriptions.read().expect("lock poisoned");
        let groups = self.subscription_groups.read().expect("lock poisoned");
        
        metrics.total_subscriptions = subscriptions.len();
        metrics.total_groups = groups.len();
        
        if groups.len() > 0 {
            let total_subscriptions_in_groups: usize = groups
                .values()
                .map(|g| g.subscription_ids.len())
                .sum();
            metrics.avg_group_size = total_subscriptions_in_groups as f64 / groups.len() as f64;
        }

        // Calculate priority distribution
        metrics.subscriptions_by_priority.clear();
        for subscription in subscriptions.values() {
            *metrics.subscriptions_by_priority
                .entry(subscription.priority)
                .or_insert(0) += 1;
        }

        // Calculate resource distribution
        metrics.subscription_distribution.clear();
        for subscription in subscriptions.values() {
            for resource in &subscription.resource_dependencies {
                *metrics.subscription_distribution
                    .entry(resource.clone())
                    .or_insert(0) += 1;
            }
        }

        metrics
    }

    // Private helper methods

    fn calculate_filter_hash(&self, analysis: &QueryAnalysis) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Create a sorted representation for consistent hashing
        let mut fields: Vec<_> = analysis.fields.iter().collect();
        fields.sort();
        let mut predicates: Vec<_> = analysis.predicates.iter().collect();
        predicates.sort();
        let mut subjects: Vec<_> = analysis.subjects.iter().collect();
        subjects.sort();
        
        fields.hash(&mut hasher);
        predicates.hash(&mut hasher);
        subjects.hash(&mut hasher);
        analysis.filters.hash(&mut hasher);
        
        hasher.finish()
    }

    fn extract_resource_dependencies(&self, analysis: &QueryAnalysis) -> HashSet<String> {
        let mut dependencies = HashSet::new();
        
        // Add all predicates and subjects as dependencies
        dependencies.extend(analysis.predicates.iter().cloned());
        dependencies.extend(analysis.subjects.iter().cloned());
        
        // Add derived dependencies based on common patterns
        for field in &analysis.fields {
            match field.as_str() {
                "person" | "people" => {
                    dependencies.insert("http://xmlns.com/foaf/0.1/Person".to_string());
                }
                "organization" | "org" => {
                    dependencies.insert("http://xmlns.com/foaf/0.1/Organization".to_string());
                }
                _ => {}
            }
        }
        
        dependencies
    }

    fn add_to_group(&self, subscription: &EnhancedSubscription) -> Result<Option<String>> {
        let mut groups = self.subscription_groups.write().expect("lock poisoned");
        
        // Find existing group with same filter hash
        let existing_group = groups
            .values_mut()
            .find(|g| g.query_hash == subscription.filter_hash);
        
        if let Some(group) = existing_group {
            group.add_subscription(subscription.base.id.clone());
            Ok(Some(group.id.clone()))
        } else {
            // Create new group if under limit
            if groups.len() < self.config.max_subscription_groups {
                let mut new_group = SubscriptionGroup::new(
                    subscription.filter_hash,
                    subscription.query_analysis.clone(),
                );
                new_group.add_subscription(subscription.base.id.clone());
                let group_id = new_group.id.clone();
                groups.insert(group_id.clone(), new_group);
                Ok(Some(group_id))
            } else {
                Ok(None) // No grouping if limit exceeded
            }
        }
    }

    fn remove_from_group(&self, group_id: &str, subscription_id: &str) -> Result<()> {
        let mut groups = self.subscription_groups.write().expect("lock poisoned");
        
        if let Some(group) = groups.get_mut(group_id) {
            group.remove_subscription(subscription_id);
            
            // Remove group if empty
            if group.is_empty() {
                groups.remove(group_id);
            }
        }
        
        Ok(())
    }

    fn get_triggered_subscriptions(&self, event: &SubscriptionEvent) -> Vec<String> {
        let subscriptions = self.subscriptions.read().expect("lock poisoned");
        let mut triggered = Vec::new();
        
        for (id, subscription) in subscriptions.iter() {
            if self.intelligent_filter.should_trigger_subscription(subscription, event) {
                triggered.push(id.clone());
            }
        }
        
        triggered
    }

    async fn execute_subscriptions_batch(
        &self,
        subscription_ids: Vec<String>,
        priority: SubscriptionPriority,
    ) -> Result<()> {
        let batch_size = self.config.group_execution_batch_size;
        
        for chunk in subscription_ids.chunks(batch_size) {
            let tasks: Vec<_> = chunk
                .iter()
                .map(|id| self.execute_single_subscription(id.clone()))
                .collect();
            
            // Execute batch concurrently with timeout
            let timeout_duration = self.config.execution_timeout;
            
            match tokio::time::timeout(timeout_duration, futures_util::future::join_all(tasks)).await {
                Ok(results) => {
                    for (i, result) in results.into_iter().enumerate() {
                        if let Err(e) = result {
                            error!("Error executing subscription {}: {}", chunk[i], e);
                        }
                    }
                }
                Err(_) => {
                    warn!("Subscription batch execution timed out for priority {:?}", priority);
                }
            }
        }
        
        Ok(())
    }

    async fn execute_single_subscription(&self, subscription_id: String) -> Result<()> {
        // This would integrate with the actual subscription execution logic
        // For now, we'll just record metrics
        
        let start_time = Instant::now();
        
        // Simulate execution
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let execution_time = start_time.elapsed();
        
        // Update subscription metrics
        {
            let mut subscriptions = self.subscriptions.write().expect("lock poisoned");
            if let Some(subscription) = subscriptions.get_mut(&subscription_id) {
                subscription.performance_metrics.record_execution(
                    execution_time,
                    100, // simulated result size
                    false, // no error
                );
            }
        }
        
        Ok(())
    }

    fn update_metrics_on_add(&self, priority: SubscriptionPriority) {
        let mut metrics = self.metrics.write().expect("lock poisoned");
        *metrics.subscriptions_by_priority.entry(priority).or_insert(0) += 1;
    }

    fn update_metrics_on_remove(&self, priority: SubscriptionPriority) {
        let mut metrics = self.metrics.write().expect("lock poisoned");
        if let Some(count) = metrics.subscriptions_by_priority.get_mut(&priority) {
            *count = count.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    fn create_test_document() -> Document {
        Document {
            definitions: vec![Definition::Operation(OperationDefinition {
                operation_type: OperationType::Subscription,
                name: Some("TestSubscription".to_string()),
                variable_definitions: vec![],
                directives: vec![],
                selection_set: SelectionSet {
                    selections: vec![Selection::Field(Field {
                        alias: None,
                        name: "person".to_string(),
                        arguments: vec![],
                        directives: vec![],
                        selection_set: Some(SelectionSet {
                            selections: vec![
                                Selection::Field(Field {
                                    alias: None,
                                    name: "name".to_string(),
                                    arguments: vec![],
                                    directives: vec![],
                                    selection_set: None,
                                }),
                                Selection::Field(Field {
                                    alias: None,
                                    name: "email".to_string(),
                                    arguments: vec![],
                                    directives: vec![],
                                    selection_set: None,
                                }),
                            ],
                        }),
                    })],
                },
            })],
        }
    }

    #[test]
    fn test_query_analysis() {
        let config = EnhancedSubscriptionConfig::default();
        let filter = IntelligentFilter::new(config);
        let document = create_test_document();

        let analysis = filter.analyze_subscription_query(&document).unwrap();

        assert!(analysis.fields.contains("person"));
        assert!(analysis.fields.contains("name"));
        assert!(analysis.fields.contains("email"));
        assert!(analysis.predicates.contains("http://xmlns.com/foaf/0.1/name"));
        assert!(analysis.predicates.contains("http://xmlns.com/foaf/0.1/mbox"));
        assert!(analysis.depth >= 2);
        assert!(analysis.estimated_cost > 0.0);
    }

    #[test]
    fn test_subscription_metrics() {
        let mut metrics = SubscriptionMetrics::default();

        metrics.record_execution(Duration::from_millis(100), 50, false);
        metrics.record_execution(Duration::from_millis(200), 75, true);

        assert_eq!(metrics.execution_count, 2);
        assert_eq!(metrics.error_count, 1);
        assert_eq!(metrics.avg_execution_time, Duration::from_millis(150));
        assert_eq!(metrics.max_execution_time, Duration::from_millis(200));
        assert_eq!(metrics.result_size_avg, 62); // (50 + 75) / 2 rounded
    }

    #[tokio::test]
    async fn test_enhanced_subscription_manager() {
        let config = EnhancedSubscriptionConfig::default();
        let manager = EnhancedSubscriptionManager::new(config, None);

        let base_subscription = ActiveSubscription {
            id: "test_sub".to_string(),
            connection_id: "test_conn".to_string(),
            document: create_test_document(),
            context: ExecutionContext::new(),
            created_at: Instant::now(),
            last_execution: None,
            execution_count: 0,
        };

        let document = create_test_document();
        let client_info = ClientInfo::default();

        manager
            .add_subscription(
                base_subscription,
                &document,
                client_info,
                SubscriptionPriority::Normal,
            )
            .unwrap();

        let metrics = manager.get_enhanced_metrics();
        assert_eq!(metrics.total_subscriptions, 1);
        assert_eq!(metrics.subscriptions_by_priority[&SubscriptionPriority::Normal], 1);
    }
}