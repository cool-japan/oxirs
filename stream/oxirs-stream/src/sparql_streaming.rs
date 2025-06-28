//! # SPARQL Streaming Module
//!
//! This module provides continuous SPARQL query support with:
//! - Query registration and lifecycle management
//! - Result streaming with push notifications
//! - Query optimization for continuous execution
//! - Performance monitoring and statistics

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, oneshot, RwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::{
    store_integration::{RdfStore, QueryResult, Triple},
    StreamEvent, EventMetadata,
};

/// Continuous query manager
pub struct ContinuousQueryManager {
    /// Registered queries
    queries: Arc<RwLock<HashMap<String, RegisteredQuery>>>,
    /// RDF store connection
    store: Arc<dyn RdfStore>,
    /// Query execution engine
    executor: Arc<QueryExecutor>,
    /// Result dispatcher
    dispatcher: Arc<ResultDispatcher>,
    /// Configuration
    config: QueryManagerConfig,
    /// Statistics
    stats: Arc<RwLock<QueryManagerStats>>,
    /// Event notifier
    event_notifier: broadcast::Sender<QueryEvent>,
}

/// Registered continuous query
#[derive(Debug)]
struct RegisteredQuery {
    /// Query ID
    id: String,
    /// SPARQL query string
    query: String,
    /// Query metadata
    metadata: QueryMetadata,
    /// Query state
    state: QueryState,
    /// Result channel
    result_channel: QueryResultChannel,
    /// Statistics
    stats: QueryStatistics,
    /// Created timestamp
    created_at: Instant,
    /// Last execution
    last_execution: Option<Instant>,
}

/// Query metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    /// Query name
    pub name: Option<String>,
    /// Query description
    pub description: Option<String>,
    /// Query owner
    pub owner: Option<String>,
    /// Query tags
    pub tags: Vec<String>,
    /// Query parameters
    pub parameters: HashMap<String, String>,
    /// Execution interval (for polling)
    pub interval: Option<Duration>,
    /// Query timeout
    pub timeout: Duration,
    /// Result limit
    pub limit: Option<usize>,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
}

impl Default for QueryMetadata {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            owner: None,
            tags: Vec::new(),
            parameters: HashMap::new(),
            interval: Some(Duration::from_secs(60)),
            timeout: Duration::from_secs(30),
            limit: Some(1000),
            enable_caching: true,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

/// Query state
#[derive(Debug, Clone, PartialEq)]
enum QueryState {
    /// Query is active and running
    Active,
    /// Query is paused
    Paused,
    /// Query is stopped
    Stopped,
    /// Query has failed
    Failed { reason: String },
}

/// Query result channel
#[derive(Debug)]
pub enum QueryResultChannel {
    /// Direct channel to subscriber
    Direct(mpsc::Sender<QueryResultUpdate>),
    /// Broadcast channel for multiple subscribers
    Broadcast(broadcast::Sender<QueryResultUpdate>),
    /// Webhook delivery
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    /// Stream to topic
    Stream { topic: String },
}

/// Query result update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResultUpdate {
    /// Query ID
    pub query_id: String,
    /// Update timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Update type
    pub update_type: UpdateType,
    /// Result bindings
    pub bindings: Vec<HashMap<String, String>>,
    /// Changed triples (for CONSTRUCT queries)
    pub triples: Option<Vec<Triple>>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Update types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    /// Initial result set
    Initial,
    /// Incremental addition
    Added,
    /// Incremental removal
    Removed,
    /// Full refresh
    Refresh,
    /// Query error
    Error { message: String },
}

/// Query statistics
#[derive(Debug, Clone, Default)]
struct QueryStatistics {
    /// Total executions
    pub execution_count: u64,
    /// Successful executions
    pub success_count: u64,
    /// Failed executions
    pub failure_count: u64,
    /// Total results returned
    pub total_results: u64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Last execution time
    pub last_execution_time: Option<Duration>,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
}

/// Query manager configuration
#[derive(Debug, Clone)]
pub struct QueryManagerConfig {
    /// Maximum concurrent queries
    pub max_concurrent_queries: usize,
    /// Maximum queries per owner
    pub max_queries_per_owner: usize,
    /// Default query timeout
    pub default_timeout: Duration,
    /// Enable query optimization
    pub enable_optimization: bool,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Query execution thread pool size
    pub executor_threads: usize,
}

impl Default for QueryManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_queries: 1000,
            max_queries_per_owner: 100,
            default_timeout: Duration::from_secs(30),
            enable_optimization: true,
            enable_caching: true,
            cache_size_limit: 10000,
            executor_threads: 4,
        }
    }
}

/// Query manager statistics
#[derive(Debug, Clone, Default)]
pub struct QueryManagerStats {
    /// Total registered queries
    pub total_queries: usize,
    /// Active queries
    pub active_queries: usize,
    /// Total executions
    pub total_executions: u64,
    /// Failed executions
    pub failed_executions: u64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Cache statistics
    pub cache_hit_rate: f64,
    /// Current cache size
    pub cache_size: usize,
}

/// Query events for monitoring
#[derive(Debug, Clone)]
pub enum QueryEvent {
    /// Query registered
    QueryRegistered { id: String, query: String },
    /// Query started
    QueryStarted { id: String },
    /// Query stopped
    QueryStopped { id: String },
    /// Query failed
    QueryFailed { id: String, reason: String },
    /// Results delivered
    ResultsDelivered { id: String, count: usize },
}

/// Query executor
struct QueryExecutor {
    /// Execution thread pool
    pool: tokio::runtime::Handle,
    /// Query optimizer
    optimizer: Arc<QueryOptimizer>,
    /// Result cache
    cache: Arc<RwLock<ResultCache>>,
}

/// Query optimizer
struct QueryOptimizer {
    /// Optimization rules
    rules: Vec<OptimizationRule>,
    /// Query patterns
    patterns: HashMap<String, QueryPattern>,
}

/// Optimization rule
struct OptimizationRule {
    name: String,
    condition: Box<dyn Fn(&str) -> bool + Send + Sync>,
    transform: Box<dyn Fn(&str) -> String + Send + Sync>,
}

/// Query pattern for optimization
#[derive(Debug, Clone)]
struct QueryPattern {
    pattern: String,
    optimized: String,
    description: String,
}

/// Result cache
struct ResultCache {
    /// Cached results
    cache: HashMap<String, CachedResult>,
    /// Cache size
    size: usize,
    /// Size limit
    limit: usize,
}

/// Cached query result
#[derive(Debug, Clone)]
struct CachedResult {
    /// Result data
    data: QueryResult,
    /// Cache timestamp
    cached_at: Instant,
    /// TTL
    ttl: Duration,
    /// Access count
    access_count: u64,
}

/// Result dispatcher
struct ResultDispatcher {
    /// Dispatcher handle
    handle: tokio::runtime::Handle,
    /// Webhook client
    webhook_client: reqwest::Client,
    /// Retry configuration
    retry_config: RetryConfig,
}

/// Retry configuration
#[derive(Debug, Clone)]
struct RetryConfig {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
    exponential_backoff: bool,
}

impl ContinuousQueryManager {
    /// Create a new continuous query manager
    pub async fn new(
        store: Arc<dyn RdfStore>,
        config: QueryManagerConfig,
    ) -> Result<Self> {
        let (tx, _) = broadcast::channel(1000);
        
        let optimizer = Arc::new(QueryOptimizer::new());
        let cache = Arc::new(RwLock::new(ResultCache::new(config.cache_size_limit)));
        
        let executor = Arc::new(QueryExecutor {
            pool: tokio::runtime::Handle::current(),
            optimizer: optimizer.clone(),
            cache: cache.clone(),
        });
        
        let dispatcher = Arc::new(ResultDispatcher {
            handle: tokio::runtime::Handle::current(),
            webhook_client: reqwest::Client::new(),
            retry_config: RetryConfig {
                max_attempts: 3,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(10),
                exponential_backoff: true,
            },
        });
        
        Ok(Self {
            queries: Arc::new(RwLock::new(HashMap::new())),
            store,
            executor,
            dispatcher,
            config,
            stats: Arc::new(RwLock::new(QueryManagerStats::default())),
            event_notifier: tx,
        })
    }
    
    /// Register a continuous query
    pub async fn register_query(
        &self,
        query: String,
        metadata: QueryMetadata,
        channel: QueryResultChannel,
    ) -> Result<String> {
        // Validate query
        self.validate_query(&query)?;
        
        // Check limits
        let queries = self.queries.read().await;
        if queries.len() >= self.config.max_concurrent_queries {
            return Err(anyhow!("Maximum concurrent queries limit reached"));
        }
        
        if let Some(owner) = &metadata.owner {
            let owner_count = queries.values()
                .filter(|q| q.metadata.owner.as_ref() == Some(owner))
                .count();
            if owner_count >= self.config.max_queries_per_owner {
                return Err(anyhow!("Maximum queries per owner limit reached"));
            }
        }
        drop(queries);
        
        // Generate query ID
        let query_id = uuid::Uuid::new_v4().to_string();
        
        // Optimize query if enabled
        let optimized_query = if self.config.enable_optimization {
            self.executor.optimizer.optimize(&query).await
        } else {
            query.clone()
        };
        
        // Create registered query
        let registered_query = RegisteredQuery {
            id: query_id.clone(),
            query: optimized_query,
            metadata,
            state: QueryState::Active,
            result_channel: channel,
            stats: QueryStatistics::default(),
            created_at: Instant::now(),
            last_execution: None,
        };
        
        // Register query
        self.queries.write().await.insert(query_id.clone(), registered_query);
        
        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_queries += 1;
        stats.active_queries += 1;
        drop(stats);
        
        // Start query execution
        self.start_query_execution(&query_id).await?;
        
        // Notify
        let _ = self.event_notifier.send(QueryEvent::QueryRegistered {
            id: query_id.clone(),
            query,
        });
        
        info!("Registered continuous query: {}", query_id);
        Ok(query_id)
    }
    
    /// Unregister a query
    pub async fn unregister_query(&self, query_id: &str) -> Result<()> {
        let mut queries = self.queries.write().await;
        let query = queries.remove(query_id)
            .ok_or_else(|| anyhow!("Query not found"))?;
        
        // Update statistics
        self.stats.write().await.active_queries -= 1;
        
        // Notify
        let _ = self.event_notifier.send(QueryEvent::QueryStopped {
            id: query_id.to_string(),
        });
        
        info!("Unregistered query: {}", query_id);
        Ok(())
    }
    
    /// Pause a query
    pub async fn pause_query(&self, query_id: &str) -> Result<()> {
        let mut queries = self.queries.write().await;
        let query = queries.get_mut(query_id)
            .ok_or_else(|| anyhow!("Query not found"))?;
        
        query.state = QueryState::Paused;
        Ok(())
    }
    
    /// Resume a query
    pub async fn resume_query(&self, query_id: &str) -> Result<()> {
        let mut queries = self.queries.write().await;
        let query = queries.get_mut(query_id)
            .ok_or_else(|| anyhow!("Query not found"))?;
        
        if query.state == QueryState::Paused {
            query.state = QueryState::Active;
            drop(queries);
            self.start_query_execution(query_id).await?;
        }
        
        Ok(())
    }
    
    /// Validate a SPARQL query
    fn validate_query(&self, query: &str) -> Result<()> {
        // Basic validation - check for required keywords
        let query_lower = query.to_lowercase();
        
        if !query_lower.contains("select") && 
           !query_lower.contains("construct") && 
           !query_lower.contains("ask") && 
           !query_lower.contains("describe") {
            return Err(anyhow!("Invalid SPARQL query: missing query form"));
        }
        
        // Check for dangerous operations
        if query_lower.contains("drop") || 
           query_lower.contains("clear") ||
           query_lower.contains("delete") ||
           query_lower.contains("insert") {
            return Err(anyhow!("Continuous queries cannot contain update operations"));
        }
        
        Ok(())
    }
    
    /// Register a SPARQL subscription query with enhanced syntax
    pub async fn register_subscription(
        &self,
        query: String,
        metadata: QueryMetadata,
        channel: QueryResultChannel,
    ) -> Result<String> {
        // Parse subscription syntax extensions
        let enhanced_query = self.parse_subscription_syntax(&query)?;
        
        // Register as continuous query
        self.register_query(enhanced_query, metadata, channel).await
    }
    
    /// Parse SPARQL subscription syntax extensions
    fn parse_subscription_syntax(&self, query: &str) -> Result<String> {
        let mut enhanced_query = query.to_string();
        
        // Check for SUBSCRIBE keyword (custom extension)
        if enhanced_query.to_lowercase().contains("subscribe") {
            // Convert SUBSCRIBE to SELECT for standard SPARQL processing
            enhanced_query = enhanced_query.replace("SUBSCRIBE", "SELECT");
            enhanced_query = enhanced_query.replace("subscribe", "SELECT");
        }
        
        // Parse ON CHANGE clauses for change detection
        if enhanced_query.to_lowercase().contains("on change") {
            // Extract change detection patterns
            // This would be expanded to parse custom change detection syntax
            info!("Detected ON CHANGE clause in subscription query");
        }
        
        // Parse WINDOW clauses for temporal windows
        if enhanced_query.to_lowercase().contains("window") {
            // Extract windowing information
            info!("Detected WINDOW clause in subscription query");
        }
        
        Ok(enhanced_query)
    }
    
    /// Start query execution
    async fn start_query_execution(&self, query_id: &str) -> Result<()> {
        let queries = self.queries.clone();
        let store = self.store.clone();
        let executor = self.executor.clone();
        let dispatcher = self.dispatcher.clone();
        let stats = self.stats.clone();
        let event_notifier = self.event_notifier.clone();
        let query_id = query_id.to_string();
        
        tokio::spawn(async move {
            let query_data = {
                let queries_guard = queries.read().await;
                queries_guard.get(&query_id).map(|q| (
                    q.query.clone(),
                    q.metadata.clone(),
                    q.metadata.interval.unwrap_or(Duration::from_secs(60)),
                ))
            };
            
            if let Some((query, metadata, poll_interval)) = query_data {
                let mut interval = interval(poll_interval);
                let mut last_result_hash = None;
                
                loop {
                    interval.tick().await;
                    
                    // Check if query is still active
                    let state = {
                        let queries_guard = queries.read().await;
                        queries_guard.get(&query_id).map(|q| q.state.clone())
                    };
                    
                    match state {
                        Some(QueryState::Active) => {
                            // Execute query
                            let start_time = Instant::now();
                            
                            match Self::execute_query(
                                &store,
                                &executor,
                                &query,
                                &metadata,
                                last_result_hash.as_ref(),
                            ).await {
                                Ok((result, hash)) => {
                                    let execution_time = start_time.elapsed();
                                    
                                    // Update query statistics
                                    {
                                        let mut queries_guard = queries.write().await;
                                        if let Some(q) = queries_guard.get_mut(&query_id) {
                                            q.stats.execution_count += 1;
                                            q.stats.success_count += 1;
                                            q.stats.total_results += result.bindings.len() as u64;
                                            q.stats.last_execution_time = Some(execution_time);
                                            q.last_execution = Some(Instant::now());
                                            
                                            // Update average execution time
                                            let count = q.stats.execution_count as u32;
                                            q.stats.avg_execution_time = 
                                                (q.stats.avg_execution_time * (count - 1) + execution_time) / count;
                                        }
                                    }
                                    
                                    // Check if results changed
                                    if Some(&hash) != last_result_hash.as_ref() {
                                        // Create update
                                        let update = QueryResultUpdate {
                                            query_id: query_id.clone(),
                                            timestamp: chrono::Utc::now(),
                                            update_type: if last_result_hash.is_none() {
                                                UpdateType::Initial
                                            } else {
                                                UpdateType::Refresh
                                            },
                                            bindings: result.bindings.clone(),
                                            triples: None,
                                            metadata: HashMap::new(),
                                        };
                                        
                                        // Dispatch results
                                        if let Err(e) = Self::dispatch_results(
                                            &queries,
                                            &dispatcher,
                                            &query_id,
                                            update,
                                        ).await {
                                            error!("Failed to dispatch results for query {}: {}", query_id, e);
                                        } else {
                                            let _ = event_notifier.send(QueryEvent::ResultsDelivered {
                                                id: query_id.clone(),
                                                count: result.bindings.len(),
                                            });
                                        }
                                        
                                        last_result_hash = Some(hash);
                                    }
                                    
                                    // Update global statistics
                                    stats.write().await.total_executions += 1;
                                }
                                Err(e) => {
                                    error!("Query execution failed for {}: {}", query_id, e);
                                    
                                    // Update query statistics
                                    {
                                        let mut queries_guard = queries.write().await;
                                        if let Some(q) = queries_guard.get_mut(&query_id) {
                                            q.stats.execution_count += 1;
                                            q.stats.failure_count += 1;
                                        }
                                    }
                                    
                                    // Update global statistics
                                    stats.write().await.failed_executions += 1;
                                    
                                    // Send error update
                                    let update = QueryResultUpdate {
                                        query_id: query_id.clone(),
                                        timestamp: chrono::Utc::now(),
                                        update_type: UpdateType::Error {
                                            message: e.to_string(),
                                        },
                                        bindings: vec![],
                                        triples: None,
                                        metadata: HashMap::new(),
                                    };
                                    
                                    let _ = Self::dispatch_results(&queries, &dispatcher, &query_id, update).await;
                                    
                                    let _ = event_notifier.send(QueryEvent::QueryFailed {
                                        id: query_id.clone(),
                                        reason: e.to_string(),
                                    });
                                }
                            }
                        }
                        Some(QueryState::Paused) => {
                            // Skip execution
                            continue;
                        }
                        Some(QueryState::Stopped) | None => {
                            // Exit loop
                            break;
                        }
                        Some(QueryState::Failed { .. }) => {
                            // Exit loop
                            break;
                        }
                    }
                }
            }
        });
        
        let _ = self.event_notifier.send(QueryEvent::QueryStarted {
            id: query_id.to_string(),
        });
        
        Ok(())
    }
    
    /// Execute a query
    async fn execute_query(
        store: &Arc<dyn RdfStore>,
        executor: &Arc<QueryExecutor>,
        query: &str,
        metadata: &QueryMetadata,
        last_hash: Option<&String>,
    ) -> Result<(QueryResult, String)> {
        // Check cache if enabled
        if metadata.enable_caching {
            if let Some(cached) = executor.cache.read().await.get(query, metadata.cache_ttl) {
                return Ok((cached, Self::hash_result(&cached)));
            }
        }
        
        // Execute query with timeout
        let result = tokio::time::timeout(
            metadata.timeout,
            store.query(query)
        ).await
            .map_err(|_| anyhow!("Query timeout"))?
            .map_err(|e| anyhow!("Query execution failed: {}", e))?;
        
        // Apply limit if specified
        let result = if let Some(limit) = metadata.limit {
            QueryResult {
                bindings: result.bindings.into_iter().take(limit).collect(),
            }
        } else {
            result
        };
        
        // Cache result if enabled
        if metadata.enable_caching {
            executor.cache.write().await.put(query.to_string(), result.clone(), metadata.cache_ttl);
        }
        
        let hash = Self::hash_result(&result);
        Ok((result, hash))
    }
    
    /// Hash query result for change detection
    fn hash_result(result: &QueryResult) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for binding in &result.bindings {
            for (key, value) in binding {
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            }
        }
        
        hasher.finish().to_string()
    }
    
    /// Dispatch query results
    async fn dispatch_results(
        queries: &Arc<RwLock<HashMap<String, RegisteredQuery>>>,
        dispatcher: &Arc<ResultDispatcher>,
        query_id: &str,
        update: QueryResultUpdate,
    ) -> Result<()> {
        let channel = {
            let queries_guard = queries.read().await;
            queries_guard.get(query_id)
                .map(|q| &q.result_channel)
                .ok_or_else(|| anyhow!("Query not found"))?
        };
        
        match channel {
            QueryResultChannel::Direct(sender) => {
                sender.send(update).await
                    .map_err(|_| anyhow!("Failed to send to direct channel"))
            }
            QueryResultChannel::Broadcast(sender) => {
                sender.send(update)
                    .map_err(|_| anyhow!("Failed to broadcast results"))?;
                Ok(())
            }
            QueryResultChannel::Webhook { url, headers } => {
                dispatcher.send_webhook(url, headers, update).await
            }
            QueryResultChannel::Stream { topic } => {
                // Publish to stream topic (using internal stream producer)
                dispatcher.send_stream(topic, update).await
            }
        }
    }
    
    /// Get query status
    pub async fn get_query_status(&self, query_id: &str) -> Result<QueryStatus> {
        let queries = self.queries.read().await;
        let query = queries.get(query_id)
            .ok_or_else(|| anyhow!("Query not found"))?;
        
        Ok(QueryStatus {
            id: query.id.clone(),
            state: format!("{:?}", query.state),
            created_at: query.created_at.elapsed(),
            last_execution: query.last_execution.map(|t| t.elapsed()),
            execution_count: query.stats.execution_count,
            success_rate: if query.stats.execution_count > 0 {
                query.stats.success_count as f64 / query.stats.execution_count as f64
            } else {
                0.0
            },
            total_results: query.stats.total_results,
            avg_execution_time: query.stats.avg_execution_time,
        })
    }
    
    /// List all queries
    pub async fn list_queries(&self) -> Vec<QueryInfo> {
        let queries = self.queries.read().await;
        queries.values().map(|q| QueryInfo {
            id: q.id.clone(),
            name: q.metadata.name.clone(),
            owner: q.metadata.owner.clone(),
            state: format!("{:?}", q.state),
            created_at: q.created_at.elapsed(),
        }).collect()
    }
    
    /// Get manager statistics
    pub async fn get_stats(&self) -> QueryManagerStats {
        self.stats.read().await.clone()
    }
    
    /// Subscribe to query events
    pub fn subscribe(&self) -> broadcast::Receiver<QueryEvent> {
        self.event_notifier.subscribe()
    }
}

/// Query status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatus {
    pub id: String,
    pub state: String,
    pub created_at: Duration,
    pub last_execution: Option<Duration>,
    pub execution_count: u64,
    pub success_rate: f64,
    pub total_results: u64,
    pub avg_execution_time: Duration,
}

/// Query information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryInfo {
    pub id: String,
    pub name: Option<String>,
    pub owner: Option<String>,
    pub state: String,
    pub created_at: Duration,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    fn new() -> Self {
        let mut optimizer = Self {
            rules: Vec::new(),
            patterns: HashMap::new(),
        };
        
        // Add default optimization rules
        optimizer.add_default_rules();
        optimizer
    }
    
    /// Add default optimization rules
    fn add_default_rules(&mut self) {
        // Rule: Remove redundant DISTINCT
        self.rules.push(OptimizationRule {
            name: "remove-redundant-distinct".to_string(),
            condition: Box::new(|query| {
                query.contains("DISTINCT") && !query.contains("ORDER BY")
            }),
            transform: Box::new(|query| {
                // This would implement the actual transformation
                query.to_string()
            }),
        });
        
        // Rule: Optimize filter placement
        self.rules.push(OptimizationRule {
            name: "optimize-filter-placement".to_string(),
            condition: Box::new(|query| {
                query.contains("FILTER") && query.contains("OPTIONAL")
            }),
            transform: Box::new(|query| {
                // This would move filters before optionals when possible
                query.to_string()
            }),
        });
    }
    
    /// Optimize a query
    async fn optimize(&self, query: &str) -> String {
        let mut optimized = query.to_string();
        
        // Apply optimization rules
        for rule in &self.rules {
            if (rule.condition)(&optimized) {
                optimized = (rule.transform)(&optimized);
                debug!("Applied optimization rule: {}", rule.name);
            }
        }
        
        optimized
    }
}

impl ResultCache {
    /// Create a new result cache
    fn new(limit: usize) -> Self {
        Self {
            cache: HashMap::new(),
            size: 0,
            limit,
        }
    }
    
    /// Get cached result
    fn get(&self, query: &str, ttl: Duration) -> Option<QueryResult> {
        self.cache.get(query).and_then(|cached| {
            if cached.cached_at.elapsed() < cached.ttl {
                Some(cached.data.clone())
            } else {
                None
            }
        })
    }
    
    /// Put result in cache
    fn put(&mut self, query: String, result: QueryResult, ttl: Duration) {
        let size = result.bindings.len();
        
        // Evict old entries if needed
        while self.size + size > self.limit && !self.cache.is_empty() {
            // Simple LRU eviction - remove oldest
            if let Some((oldest_key, _)) = self.cache.iter()
                .min_by_key(|(_, v)| v.cached_at) {
                let key = oldest_key.clone();
                if let Some(removed) = self.cache.remove(&key) {
                    self.size -= removed.data.bindings.len();
                }
            }
        }
        
        self.cache.insert(query, CachedResult {
            data: result,
            cached_at: Instant::now(),
            ttl,
            access_count: 0,
        });
        
        self.size += size;
    }
}

impl ResultDispatcher {
    /// Create a stream producer for a specific topic
    async fn create_stream_producer_for_topic(&self, topic: &str) -> Result<crate::StreamProducer> {
        // Create a default stream configuration for this topic
        let config = crate::StreamConfig {
            backend: crate::StreamBackend::Memory {
                max_size: Some(10000),
                persistence: false,
            },
            topic: topic.to_string(),
            batch_size: 100,
            flush_interval_ms: 100,
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            enable_compression: false,
            compression_type: crate::CompressionType::None,
            retry_config: crate::RetryConfig::default(),
            circuit_breaker: crate::CircuitBreakerConfig::default(),
            security: crate::SecurityConfig::default(),
            performance: crate::PerformanceConfig::default(),
            monitoring: crate::MonitoringConfig::default(),
        };
        
        // Create and return the producer
        crate::StreamProducer::new(config).await
    }
    /// Send results via webhook
    async fn send_webhook(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        update: QueryResultUpdate,
    ) -> Result<()> {
        let mut request = self.webhook_client.post(url)
            .json(&update)
            .timeout(Duration::from_secs(30));
        
        // Add custom headers
        for (key, value) in headers {
            request = request.header(key, value);
        }
        
        // Send with retry
        let mut attempts = 0;
        let mut delay = self.retry_config.initial_delay;
        
        loop {
            attempts += 1;
            
            match request.try_clone().unwrap().send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(());
                    } else {
                        let status = response.status();
                        let body = response.text().await.unwrap_or_default();
                        
                        if attempts >= self.retry_config.max_attempts {
                            return Err(anyhow!("Webhook failed with status {}: {}", status, body));
                        }
                        
                        warn!("Webhook attempt {} failed with status {}", attempts, status);
                    }
                }
                Err(e) => {
                    if attempts >= self.retry_config.max_attempts {
                        return Err(anyhow!("Webhook failed after {} attempts: {}", attempts, e));
                    }
                    
                    warn!("Webhook attempt {} failed: {}", attempts, e);
                }
            }
            
            // Wait before retry
            tokio::time::sleep(delay).await;
            
            // Update delay for next attempt
            if self.retry_config.exponential_backoff {
                delay = (delay * 2).min(self.retry_config.max_delay);
            }
        }
    }
    
    /// Send results to stream topic
    async fn send_stream(
        &self,
        topic: &str,
        update: QueryResultUpdate,
    ) -> Result<()> {
        // Convert query result update to stream event
        let stream_event = match update.update_type {
            UpdateType::Added => StreamEvent::QueryResultAdded {
                query_id: update.query_id.clone(),
                result: crate::event::QueryResult {
                    query_id: update.query_id.clone(),
                    bindings: update.bindings.first().cloned().unwrap_or_default(),
                    execution_time: Duration::from_millis(0),
                },
                metadata: EventMetadata {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    source: "sparql-streaming".to_string(),
                    user: Some("query-engine".to_string()),
                    context: Some(update.query_id.clone()),
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: {
                        let mut props = std::collections::HashMap::new();
                        props.insert("topic".to_string(), topic.to_string());
                        props.insert("update_type".to_string(), "result_added".to_string());
                        props
                    },
                    checksum: None,
                },
            },
            UpdateType::Removed => StreamEvent::QueryResultRemoved {
                query_id: update.query_id.clone(),
                result: crate::event::QueryResult {
                    query_id: update.query_id.clone(),
                    bindings: update.bindings.first().cloned().unwrap_or_default(),
                    execution_time: Duration::from_millis(0),
                },
                metadata: EventMetadata {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    source: "sparql-streaming".to_string(),
                    user: Some("query-engine".to_string()),
                    context: Some(update.query_id.clone()),
                    caused_by: None,
                    version: "1.0".to_string(),
                    properties: {
                        let mut props = std::collections::HashMap::new();
                        props.insert("topic".to_string(), topic.to_string());
                        props.insert("update_type".to_string(), "result_removed".to_string());
                        props
                    },
                    checksum: None,
                },
            },
            UpdateType::Initial | UpdateType::Refresh => {
                // For initial and refresh updates, we just use QueryResultAdded
                StreamEvent::QueryResultAdded {
                    query_id: update.query_id.clone(),
                    result: crate::event::QueryResult {
                        query_id: update.query_id.clone(),
                        bindings: update.bindings.first().cloned().unwrap_or_default(),
                        execution_time: Duration::from_millis(0), // Default execution time
                    },
                    metadata: EventMetadata {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        timestamp: chrono::Utc::now(),
                        source: "sparql-streaming".to_string(),
                        user: Some("query-engine".to_string()),
                        context: Some(update.query_id.clone()),
                        caused_by: None,
                        version: "1.0".to_string(),
                        properties: {
                            let mut props = std::collections::HashMap::new();
                            props.insert("topic".to_string(), topic.to_string());
                            props.insert("update_type".to_string(), format!("{:?}", update.update_type).to_lowercase());
                            props
                        },
                        checksum: None,
                    },
                }
            },
            UpdateType::Error { message } => {
                // For errors, we'll just log and return Ok for now
                warn!("Query error in stream: {}", message);
                return Ok(());
            },
        };
        
        // Create a stream producer for the topic and publish the event
        match self.create_stream_producer_for_topic(topic).await {
            Ok(mut producer) => {
                match producer.publish(stream_event).await {
                    Ok(_) => {
                        info!("Successfully published query result to stream topic '{}'", topic);
                    }
                    Err(e) => {
                        error!("Failed to publish to stream topic '{}': {}", topic, e);
                        return Err(anyhow!("Stream publishing failed: {}", e));
                    }
                }
            }
            Err(e) => {
                error!("Failed to create stream producer for topic '{}': {}", topic, e);
                return Err(anyhow!("Stream producer creation failed: {}", e));
            }
        }
        
        Ok(())
    }
}

/// Create subscription channel for query results
pub fn create_subscription_channel() -> (mpsc::Sender<QueryResultUpdate>, mpsc::Receiver<QueryResultUpdate>) {
    mpsc::channel(100)
}

/// Create broadcast channel for query results
pub fn create_broadcast_channel() -> (broadcast::Sender<QueryResultUpdate>, broadcast::Receiver<QueryResultUpdate>) {
    broadcast::channel(100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store_integration::MockRdfStore;
    
    #[tokio::test]
    async fn test_query_registration() {
        let store = Arc::new(MockRdfStore {
            log_position: Arc::new(RwLock::new(0)),
            changes: Arc::new(RwLock::new(vec![])),
        });
        
        let manager = ContinuousQueryManager::new(store, QueryManagerConfig::default())
            .await
            .unwrap();
        
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10";
        let metadata = QueryMetadata::default();
        let (tx, _rx) = create_subscription_channel();
        let channel = QueryResultChannel::Direct(tx);
        
        let query_id = manager.register_query(query.to_string(), metadata, channel)
            .await
            .unwrap();
        
        assert!(!query_id.is_empty());
        
        // Check query is registered
        let queries = manager.list_queries().await;
        assert_eq!(queries.len(), 1);
        assert_eq!(queries[0].id, query_id);
    }
    
    #[tokio::test]
    async fn test_query_validation() {
        let store = Arc::new(MockRdfStore {
            log_position: Arc::new(RwLock::new(0)),
            changes: Arc::new(RwLock::new(vec![])),
        });
        
        let manager = ContinuousQueryManager::new(store, QueryManagerConfig::default())
            .await
            .unwrap();
        
        // Test invalid query
        let invalid_query = "DELETE WHERE { ?s ?p ?o }";
        let result = manager.validate_query(invalid_query);
        assert!(result.is_err());
        
        // Test valid query
        let valid_query = "SELECT ?s WHERE { ?s ?p ?o }";
        let result = manager.validate_query(valid_query);
        assert!(result.is_ok());
    }
}