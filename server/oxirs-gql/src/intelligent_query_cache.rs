//! Intelligent Query Cache
//!
//! This module implements an AI-driven query caching system that learns from query patterns
//! and proactively caches frequently used queries for enhanced performance.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::ast::{Document, Value, Definition, OperationDefinition, FragmentDefinition, Selection};
use crate::distributed_cache::{CacheConfig, GraphQLQueryCache, DistributedCache};

/// Configuration for intelligent query caching
#[derive(Debug, Clone)]
pub struct IntelligentCacheConfig {
    pub max_cache_entries: usize,
    pub max_pattern_history: usize,
    pub prediction_confidence_threshold: f64,
    pub cache_ttl_seconds: u64,
    pub pattern_analysis_interval_seconds: u64,
    pub enable_predictive_caching: bool,
    pub enable_pattern_learning: bool,
    pub enable_usage_analytics: bool,
}

impl Default for IntelligentCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_entries: 10000,
            max_pattern_history: 1000,
            prediction_confidence_threshold: 0.75,
            cache_ttl_seconds: 3600, // 1 hour
            pattern_analysis_interval_seconds: 300, // 5 minutes
            enable_predictive_caching: true,
            enable_pattern_learning: true,
            enable_usage_analytics: true,
        }
    }
}

/// Query pattern for similarity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    pub query_type: String,
    pub field_count: usize,
    pub depth: usize,
    pub has_arguments: bool,
    pub has_fragments: bool,
    pub complexity_score: f64,
    pub timestamp: u64,
}

impl QueryPattern {
    pub fn from_document(doc: &Document) -> Self {
        let operations = Self::extract_operations(doc);
        let fragments = Self::extract_fragments(doc);
        
        let query_type = operations.first()
            .map(|op| format!("{:?}", op.operation_type).to_lowercase())
            .unwrap_or_else(|| "query".to_string());
        
        let field_count = Self::count_fields_from_operations(&operations);
        let depth = Self::calculate_depth_from_operations(&operations);
        let has_arguments = Self::has_arguments_from_operations(&operations);
        let has_fragments = !fragments.is_empty();
        let complexity_score = Self::calculate_complexity_from_operations(&operations);
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            query_type,
            field_count,
            depth,
            has_arguments,
            has_fragments,
            complexity_score,
            timestamp,
        }
    }
    
    fn extract_operations(doc: &Document) -> Vec<&OperationDefinition> {
        doc.definitions.iter().filter_map(|def| {
            if let Definition::Operation(op) = def {
                Some(op)
            } else {
                None
            }
        }).collect()
    }
    
    fn extract_fragments(doc: &Document) -> Vec<&FragmentDefinition> {
        doc.definitions.iter().filter_map(|def| {
            if let Definition::Fragment(frag) = def {
                Some(frag)
            } else {
                None
            }
        }).collect()
    }

    fn count_fields_from_operations(operations: &[&OperationDefinition]) -> usize {
        // Simplified field counting
        operations.iter().map(|op| {
            op.selection_set.selections.len()
        }).sum()
    }

    fn calculate_depth_from_operations(operations: &[&OperationDefinition]) -> usize {
        // Simplified depth calculation
        operations.iter().map(|op| {
            Self::selection_depth(&op.selection_set.selections)
        }).max().unwrap_or(1)
    }

    fn selection_depth(selections: &[crate::ast::Selection]) -> usize {
        selections.iter().map(|sel| {
            match sel {
                crate::ast::Selection::Field(field) => {
                    if let Some(ref selection_set) = field.selection_set {
                        1 + Self::selection_depth(&selection_set.selections)
                    } else {
                        1
                    }
                },
                crate::ast::Selection::InlineFragment(frag) => {
                    1 + Self::selection_depth(&frag.selection_set.selections)
                },
                crate::ast::Selection::FragmentSpread(_) => 1,
            }
        }).max().unwrap_or(1)
    }

    fn has_arguments_from_operations(operations: &[&OperationDefinition]) -> bool {
        operations.iter().any(|op| {
            Self::selection_has_args(&op.selection_set.selections)
        })
    }

    fn selection_has_args(selections: &[crate::ast::Selection]) -> bool {
        selections.iter().any(|sel| {
            match sel {
                crate::ast::Selection::Field(field) => {
                    !field.arguments.is_empty() ||
                    field.selection_set.as_ref()
                        .map(|ss| Self::selection_has_args(&ss.selections))
                        .unwrap_or(false)
                },
                crate::ast::Selection::InlineFragment(frag) => {
                    Self::selection_has_args(&frag.selection_set.selections)
                },
                crate::ast::Selection::FragmentSpread(_) => false,
            }
        })
    }

    fn calculate_complexity_from_operations(operations: &[&OperationDefinition]) -> f64 {
        let field_weight = Self::count_fields_from_operations(operations) as f64 * 1.0;
        let depth_weight = Self::calculate_depth_from_operations(operations) as f64 * 2.0;
        let args_weight = if Self::has_arguments_from_operations(operations) { 3.0 } else { 0.0 };
        
        field_weight + depth_weight + args_weight
    }

    /// Calculate similarity to another pattern (0.0 to 1.0)
    pub fn similarity(&self, other: &QueryPattern) -> f64 {
        let type_similarity = if self.query_type == other.query_type { 1.0 } else { 0.0 };
        let field_similarity = 1.0 - ((self.field_count as f64 - other.field_count as f64).abs() / 10.0).min(1.0);
        let depth_similarity = 1.0 - ((self.depth as f64 - other.depth as f64).abs() / 5.0).min(1.0);
        let args_similarity = if self.has_arguments == other.has_arguments { 1.0 } else { 0.5 };
        let fragments_similarity = if self.has_fragments == other.has_fragments { 1.0 } else { 0.5 };
        let complexity_similarity = 1.0 - ((self.complexity_score - other.complexity_score).abs() / 20.0).min(1.0);

        (type_similarity * 0.3 + field_similarity * 0.2 + depth_similarity * 0.2 + 
         args_similarity * 0.1 + fragments_similarity * 0.1 + complexity_similarity * 0.1)
    }
}

/// Query usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryUsageStats {
    pub hit_count: u64,
    pub last_access: u64,
    pub average_execution_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub pattern: QueryPattern,
}

impl QueryUsageStats {
    pub fn new(pattern: QueryPattern) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            hit_count: 1,
            last_access: timestamp,
            average_execution_time_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            pattern,
        }
    }

    pub fn update_access(&mut self, execution_time_ms: f64, was_cache_hit: bool) {
        self.hit_count += 1;
        self.last_access = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Update average execution time using exponential moving average
        let alpha = 0.1; // Smoothing factor
        self.average_execution_time_ms = alpha * execution_time_ms + 
                                       (1.0 - alpha) * self.average_execution_time_ms;

        if was_cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
    }

    pub fn cache_hit_ratio(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        }
    }

    pub fn access_frequency(&self) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let age_hours = ((now - self.pattern.timestamp) as f64) / 3600.0;
        if age_hours == 0.0 {
            self.hit_count as f64
        } else {
            self.hit_count as f64 / age_hours
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub value: Value,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub prediction_confidence: f64,
}

impl CacheEntry {
    pub fn new(value: Value, prediction_confidence: f64) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            prediction_confidence,
        }
    }

    pub fn access(&mut self) -> &Value {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        &self.value
    }

    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    pub fn access_score(&self) -> f64 {
        let recency_score = 1.0 / (1.0 + self.last_accessed.elapsed().as_secs() as f64 / 3600.0);
        let frequency_score = (self.access_count as f64).ln().max(1.0);
        let confidence_score = self.prediction_confidence;
        
        recency_score * 0.4 + frequency_score * 0.3 + confidence_score * 0.3
    }
}

/// Intelligent query cache with pattern learning and prediction
pub struct IntelligentQueryCache {
    config: IntelligentCacheConfig,
    cache_entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    usage_stats: Arc<RwLock<HashMap<String, QueryUsageStats>>>,
    pattern_history: Arc<RwLock<VecDeque<QueryPattern>>>,
    distributed_cache: Option<Arc<GraphQLQueryCache>>,
}

impl IntelligentQueryCache {
    pub fn new(config: IntelligentCacheConfig) -> Self {
        Self {
            config,
            cache_entries: Arc::new(RwLock::new(HashMap::new())),
            usage_stats: Arc::new(RwLock::new(HashMap::new())),
            pattern_history: Arc::new(RwLock::new(VecDeque::new())),
            distributed_cache: None,
        }
    }

    pub async fn with_distributed_cache(
        mut self,
        cache_config: CacheConfig,
    ) -> Result<Self> {
        let distributed_cache = Arc::new(GraphQLQueryCache::new(cache_config).await?);
        self.distributed_cache = Some(distributed_cache);
        Ok(self)
    }

    /// Generate cache key for a query
    pub fn generate_cache_key(&self, query: &str, variables: &HashMap<String, Value>) -> String {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        
        // Sort variables for consistent hashing
        let mut var_pairs: Vec<_> = variables.iter().collect();
        var_pairs.sort_by_key(|(k, _)| *k);
        
        for (key, value) in var_pairs {
            key.hash(&mut hasher);
            format!("{:?}", value).hash(&mut hasher);
        }
        
        format!("iqc:{:x}", hasher.finish())
    }

    /// Get cached query result
    pub async fn get(&self, query: &str, variables: &HashMap<String, Value>) -> Option<Value> {
        let cache_key = self.generate_cache_key(query, variables);
        let start_time = Instant::now();
        
        // Try local cache first
        {
            let mut cache = self.cache_entries.write().await;
            if let Some(entry) = cache.get_mut(&cache_key) {
                if !entry.is_expired(Duration::from_secs(self.config.cache_ttl_seconds)) {
                    let result = entry.access().clone();
                    
                    // Update usage stats
                    self.update_usage_stats(&cache_key, start_time.elapsed().as_millis() as f64, true).await;
                    
                    debug!("Cache hit for query key: {}", cache_key);
                    return Some(result);
                }
            }
        }

        // Try distributed cache if available
        if let Some(ref distributed_cache) = self.distributed_cache {
            if let Ok(Some(data)) = distributed_cache.raw_get(&cache_key).await {
                // Convert bytes back to Value
                if let Ok(value) = serde_json::from_slice::<Value>(&data) {
                    // Store in local cache for faster future access
                    self.store_local(&cache_key, value.clone(), 0.8).await;
                    
                    // Update usage stats
                    self.update_usage_stats(&cache_key, start_time.elapsed().as_millis() as f64, true).await;
                    
                    debug!("Distributed cache hit for query key: {}", cache_key);
                    return Some(value);
                }
            }
        }

        // Cache miss
        self.update_usage_stats(&cache_key, start_time.elapsed().as_millis() as f64, false).await;
        None
    }

    /// Store query result in cache
    pub async fn set(
        &self,
        query: &str,
        variables: &HashMap<String, Value>,
        result: Value,
        doc: &Document,
    ) -> Result<()> {
        let cache_key = self.generate_cache_key(query, variables);
        let pattern = QueryPattern::from_document(doc);
        
        // Analyze pattern and calculate prediction confidence
        let confidence = self.calculate_prediction_confidence(&pattern).await;
        
        // Store in local cache
        self.store_local(&cache_key, result.clone(), confidence).await;
        
        // Store in distributed cache if available
        if let Some(ref distributed_cache) = self.distributed_cache {
            if let Ok(data) = serde_json::to_vec(&result) {
                if let Err(e) = distributed_cache.raw_set(&cache_key, data, Some(Duration::from_secs(self.config.cache_ttl_seconds))).await {
                    warn!("Failed to store in distributed cache: {}", e);
                }
            }
        }

        // Record pattern for learning
        if self.config.enable_pattern_learning {
            self.record_pattern(pattern).await;
        }

        // Update usage stats
        {
            let mut stats = self.usage_stats.write().await;
            stats.entry(cache_key)
                .or_insert_with(|| QueryUsageStats::new(QueryPattern::from_document(doc)));
        }

        Ok(())
    }

    async fn store_local(&self, cache_key: &str, value: Value, confidence: f64) {
        let mut cache = self.cache_entries.write().await;
        
        // Evict expired entries and maintain size limit
        let ttl = Duration::from_secs(self.config.cache_ttl_seconds);
        cache.retain(|_, entry| !entry.is_expired(ttl));
        
        // If cache is full, evict least valuable entries
        while cache.len() >= self.config.max_cache_entries {
            if let Some(evict_key) = cache.iter()
                .min_by(|(_, a), (_, b)| a.access_score().partial_cmp(&b.access_score()).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.clone()) {
                cache.remove(&evict_key);
            } else {
                break;
            }
        }
        
        cache.insert(cache_key.to_string(), CacheEntry::new(value, confidence));
    }

    async fn calculate_prediction_confidence(&self, pattern: &QueryPattern) -> f64 {
        if !self.config.enable_predictive_caching {
            return 0.5; // Default confidence
        }

        let pattern_history = self.pattern_history.read().await;
        let similar_patterns: Vec<_> = pattern_history.iter()
            .map(|p| (p.similarity(pattern), p))
            .filter(|(sim, _)| *sim > 0.5)
            .collect();

        if similar_patterns.is_empty() {
            return 0.3; // Low confidence for new patterns
        }

        // Calculate confidence based on similarity and frequency of similar patterns
        let total_similarity: f64 = similar_patterns.iter().map(|(sim, _)| sim).sum();
        let avg_similarity = total_similarity / similar_patterns.len() as f64;
        
        // Higher confidence for patterns with more similar historical patterns
        let frequency_boost = (similar_patterns.len() as f64 / 10.0).min(0.3);
        
        (avg_similarity + frequency_boost).min(1.0)
    }

    async fn record_pattern(&self, pattern: QueryPattern) {
        let mut history = self.pattern_history.write().await;
        
        history.push_back(pattern);
        
        // Maintain history size limit
        while history.len() > self.config.max_pattern_history {
            history.pop_front();
        }
    }

    async fn update_usage_stats(&self, cache_key: &str, execution_time_ms: f64, was_cache_hit: bool) {
        if !self.config.enable_usage_analytics {
            return;
        }

        let mut stats = self.usage_stats.write().await;
        if let Some(stat) = stats.get_mut(cache_key) {
            stat.update_access(execution_time_ms, was_cache_hit);
        }
    }

    /// Get cache statistics
    pub async fn get_statistics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut stats = HashMap::new();
        
        let cache_entries = self.cache_entries.read().await;
        let usage_stats = self.usage_stats.read().await;
        let pattern_history = self.pattern_history.read().await;
        
        stats.insert("cache_size".to_string(), 
                     serde_json::Value::Number(cache_entries.len().into()));
        
        stats.insert("pattern_history_size".to_string(), 
                     serde_json::Value::Number(pattern_history.len().into()));
        
        stats.insert("usage_stats_size".to_string(), 
                     serde_json::Value::Number(usage_stats.len().into()));
        
        // Calculate overall cache hit ratio
        let total_hits: u64 = usage_stats.values().map(|s| s.cache_hits).sum();
        let total_misses: u64 = usage_stats.values().map(|s| s.cache_misses).sum();
        let hit_ratio = if total_hits + total_misses > 0 {
            total_hits as f64 / (total_hits + total_misses) as f64
        } else {
            0.0
        };
        
        stats.insert("overall_hit_ratio".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from_f64(hit_ratio).unwrap()));
        
        // Average execution time
        let avg_exec_time: f64 = usage_stats.values()
            .map(|s| s.average_execution_time_ms)
            .sum::<f64>() / usage_stats.len().max(1) as f64;
        
        stats.insert("average_execution_time_ms".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from_f64(avg_exec_time).unwrap()));
        
        Ok(stats)
    }

    /// Predict queries that should be pre-cached based on patterns
    pub async fn predict_queries(&self) -> Vec<(String, f64)> {
        if !self.config.enable_predictive_caching {
            return Vec::new();
        }

        let pattern_history = self.pattern_history.read().await;
        let usage_stats = self.usage_stats.read().await;
        
        // Find patterns that occur frequently and might be requested soon
        let mut predictions = Vec::new();
        
        for (cache_key, stats) in usage_stats.iter() {
            let frequency = stats.access_frequency();
            let recency_factor = {
                let hours_since_access = (SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() - stats.last_access) as f64 / 3600.0;
                1.0 / (1.0 + hours_since_access / 24.0) // Decay over days
            };
            
            let prediction_score = frequency * recency_factor * stats.cache_hit_ratio();
            
            if prediction_score > self.config.prediction_confidence_threshold {
                predictions.push((cache_key.clone(), prediction_score));
            }
        }
        
        // Sort by prediction score descending
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        info!("Generated {} query predictions", predictions.len());
        predictions
    }

    /// Clear expired entries from cache
    pub async fn cleanup_expired(&self) -> usize {
        let mut cache = self.cache_entries.write().await;
        let initial_size = cache.len();
        
        let ttl = Duration::from_secs(self.config.cache_ttl_seconds);
        cache.retain(|_, entry| !entry.is_expired(ttl));
        
        let removed = initial_size - cache.len();
        if removed > 0 {
            info!("Cleaned up {} expired cache entries", removed);
        }
        
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Document, Operation, OperationType, SelectionSet};

    #[tokio::test]
    async fn test_intelligent_cache_creation() {
        let config = IntelligentCacheConfig::default();
        let cache = IntelligentQueryCache::new(config);
        
        assert!(cache.cache_entries.read().await.is_empty());
        assert!(cache.usage_stats.read().await.is_empty());
        assert!(cache.pattern_history.read().await.is_empty());
    }

    #[tokio::test]
    async fn test_cache_key_generation() {
        let cache = IntelligentQueryCache::new(IntelligentCacheConfig::default());
        
        let query = "query { user { name } }";
        let variables = HashMap::new();
        
        let key1 = cache.generate_cache_key(query, &variables);
        let key2 = cache.generate_cache_key(query, &variables);
        
        assert_eq!(key1, key2);
        assert!(key1.starts_with("iqc:"));
    }

    #[tokio::test]
    async fn test_pattern_similarity() {
        let doc1 = Document {
            operations: {
                let mut ops = HashMap::new();
                ops.insert("query".to_string(), Operation {
                    operation_type: OperationType::Query,
                    name: None,
                    variable_definitions: Vec::new(),
                    directives: Vec::new(),
                    selection_set: SelectionSet { selections: Vec::new() },
                });
                ops
            },
            fragments: HashMap::new(),
        };
        
        let pattern1 = QueryPattern::from_document(&doc1);
        let pattern2 = QueryPattern::from_document(&doc1);
        
        let similarity = pattern1.similarity(&pattern2);
        assert!((similarity - 1.0).abs() < 0.001); // Should be identical
    }

    #[tokio::test]
    async fn test_cache_statistics() {
        let cache = IntelligentQueryCache::new(IntelligentCacheConfig::default());
        let stats = cache.get_statistics().await.unwrap();
        
        assert!(stats.contains_key("cache_size"));
        assert!(stats.contains_key("pattern_history_size"));
        assert!(stats.contains_key("overall_hit_ratio"));
    }
}