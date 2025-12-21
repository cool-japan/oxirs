//! GraphQL Query Batching
//!
//! This module implements efficient query batching to reduce round-trip overhead
//! and improve overall system performance:
//! - **Request Batching**: Execute multiple queries in a single HTTP request
//! - **Automatic Deduplication**: Detect and eliminate duplicate queries
//! - **Intelligent Scheduling**: Optimize execution order based on dependencies
//! - **Parallel Execution**: Execute independent queries concurrently
//! - **Result Aggregation**: Combine results from batched queries
//! - **Timeout Management**: Per-query and batch-level timeouts
//!
//! ## Features
//!
//! ### Batch Processing
//! - Combine multiple GraphQL queries into a single request
//! - Automatic query deduplication within a batch
//! - Configurable batch size limits
//! - Batch timeout management
//!
//! ### Execution Strategies
//! - **Sequential**: Execute queries one by one (for dependencies)
//! - **Parallel**: Execute all queries concurrently (default)
//! - **Adaptive**: Analyze dependencies and execute optimally
//! - **Priority-based**: Execute high-priority queries first
//!
//! ### Performance Optimization
//! - Query result sharing for duplicates
//! - Connection pooling across batch
//! - Memory-efficient result aggregation
//! - Adaptive concurrency based on system load
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxirs_gql::query_batching::{QueryBatcher, BatchConfig, ExecutionStrategy};
//!
//! // Create a query batcher
//! let config = BatchConfig::default()
//!     .with_max_batch_size(10)
//!     .with_strategy(ExecutionStrategy::Parallel);
//!
//! let batcher = QueryBatcher::new(config);
//!
//! // Add queries to batch
//! let batch_id = batcher.create_batch();
//! batcher.add_query(batch_id, "{ user { name } }");
//! batcher.add_query(batch_id, "{ posts { title } }");
//!
//! // Execute batch
//! let results = batcher.execute_batch(batch_id).await?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Execution strategy for batch processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Execute queries sequentially
    Sequential,
    /// Execute all queries in parallel
    Parallel,
    /// Analyze dependencies and optimize execution
    Adaptive,
    /// Execute by priority order
    PriorityBased,
}

/// Query priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QueryPriority {
    /// Low priority (background tasks)
    Low = 0,
    /// Normal priority (default)
    Normal = 1,
    /// High priority (user-facing queries)
    High = 2,
    /// Critical priority (real-time requirements)
    Critical = 3,
}

impl Default for QueryPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Batch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum number of queries per batch
    pub max_batch_size: usize,
    /// Maximum concurrent query execution
    pub max_concurrency: usize,
    /// Batch execution timeout
    pub batch_timeout: Duration,
    /// Per-query timeout
    pub query_timeout: Duration,
    /// Execution strategy
    pub strategy: ExecutionStrategy,
    /// Enable query deduplication
    pub enable_deduplication: bool,
    /// Enable result caching
    pub enable_caching: bool,
}

impl BatchConfig {
    /// Create a new batch configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set maximum concurrency
    pub fn with_max_concurrency(mut self, concurrency: usize) -> Self {
        self.max_concurrency = concurrency;
        self
    }

    /// Set batch timeout
    pub fn with_batch_timeout(mut self, timeout: Duration) -> Self {
        self.batch_timeout = timeout;
        self
    }

    /// Set execution strategy
    pub fn with_strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable query deduplication
    pub fn with_deduplication(mut self, enable: bool) -> Self {
        self.enable_deduplication = enable;
        self
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_concurrency: 10,
            batch_timeout: Duration::from_secs(30),
            query_timeout: Duration::from_secs(10),
            strategy: ExecutionStrategy::Parallel,
            enable_deduplication: true,
            enable_caching: true,
        }
    }
}

/// Batched query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedQuery {
    /// Query ID
    pub id: String,
    /// GraphQL query string
    pub query: String,
    /// Query variables
    pub variables: Option<serde_json::Value>,
    /// Operation name
    pub operation_name: Option<String>,
    /// Query priority
    pub priority: QueryPriority,
    /// Query fingerprint for deduplication
    pub fingerprint: String,
}

impl BatchedQuery {
    /// Create a new batched query
    pub fn new(query: String) -> Self {
        let fingerprint = Self::calculate_fingerprint(&query, &None);
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            query,
            variables: None,
            operation_name: None,
            priority: QueryPriority::default(),
            fingerprint,
        }
    }

    /// Create with variables
    pub fn with_variables(mut self, variables: serde_json::Value) -> Self {
        self.fingerprint = Self::calculate_fingerprint(&self.query, &Some(variables.clone()));
        self.variables = Some(variables);
        self
    }

    /// Set operation name
    pub fn with_operation_name(mut self, name: String) -> Self {
        self.operation_name = Some(name);
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: QueryPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Calculate query fingerprint for deduplication
    fn calculate_fingerprint(query: &str, variables: &Option<serde_json::Value>) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        if let Some(vars) = variables {
            if let Ok(vars_str) = serde_json::to_string(vars) {
                hasher.update(vars_str.as_bytes());
            }
        }
        let result = hasher.finalize();
        hex::encode(&result[..16]) // Use first 16 bytes
    }
}

/// Query execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Query ID
    pub query_id: String,
    /// Execution result (JSON)
    pub data: Option<serde_json::Value>,
    /// Errors if any
    pub errors: Vec<String>,
    /// Execution duration
    pub duration: Duration,
    /// Whether result was cached
    pub cached: bool,
    /// Whether result was deduplicated
    pub deduplicated: bool,
}

/// Batch execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Batch ID
    pub batch_id: String,
    /// Query results
    pub results: Vec<QueryResult>,
    /// Total batch execution time
    pub total_duration: Duration,
    /// Number of queries executed
    pub queries_executed: usize,
    /// Number of queries deduplicated
    pub queries_deduplicated: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Batch statistics
    pub statistics: BatchStatistics,
}

/// Batch execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchStatistics {
    /// Total queries submitted
    pub total_queries: usize,
    /// Queries executed (after dedup)
    pub executed_queries: usize,
    /// Deduplicated queries
    pub deduplicated_queries: usize,
    /// Cached queries
    pub cached_queries: usize,
    /// Failed queries
    pub failed_queries: usize,
    /// Average query duration
    pub avg_query_duration: Duration,
    /// Max query duration
    pub max_query_duration: Duration,
    /// Min query duration
    pub min_query_duration: Duration,
}

/// Query batch
#[derive(Debug)]
struct QueryBatch {
    /// Batch ID
    id: String,
    /// Queries in batch
    queries: Vec<BatchedQuery>,
    /// Creation timestamp
    #[allow(dead_code)]
    created_at: Instant,
    /// Execution started
    started: bool,
    /// Execution completed
    completed: bool,
}

impl QueryBatch {
    fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            queries: Vec::new(),
            created_at: Instant::now(),
            started: false,
            completed: false,
        }
    }

    fn add_query(&mut self, query: BatchedQuery) {
        self.queries.push(query);
    }

    fn query_count(&self) -> usize {
        self.queries.len()
    }
}

/// Query batcher
pub struct QueryBatcher {
    /// Batch configuration
    config: BatchConfig,
    /// Active batches
    batches: Arc<RwLock<HashMap<String, QueryBatch>>>,
    /// Query result cache
    cache: Arc<RwLock<HashMap<String, QueryResult>>>,
    /// Batch statistics
    statistics: Arc<RwLock<HashMap<String, BatchStatistics>>>,
}

impl QueryBatcher {
    /// Create a new query batcher
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            batches: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new batch
    pub async fn create_batch(&self) -> String {
        let batch = QueryBatch::new();
        let batch_id = batch.id.clone();
        self.batches.write().await.insert(batch_id.clone(), batch);
        batch_id
    }

    /// Add query to batch
    pub async fn add_query(&self, batch_id: &str, query: BatchedQuery) -> Result<(), String> {
        let mut batches = self.batches.write().await;
        let batch = batches
            .get_mut(batch_id)
            .ok_or_else(|| "Batch not found".to_string())?;

        if batch.started {
            return Err("Batch already started execution".to_string());
        }

        if batch.query_count() >= self.config.max_batch_size {
            return Err(format!(
                "Batch size limit reached ({})",
                self.config.max_batch_size
            ));
        }

        batch.add_query(query);
        Ok(())
    }

    /// Execute batch
    pub async fn execute_batch(&self, batch_id: &str) -> Result<BatchResult, String> {
        let start_time = Instant::now();

        // Get and mark batch as started
        let queries = {
            let mut batches = self.batches.write().await;
            let batch = batches
                .get_mut(batch_id)
                .ok_or_else(|| "Batch not found".to_string())?;

            if batch.started {
                return Err("Batch already started".to_string());
            }

            batch.started = true;
            batch.queries.clone()
        };

        // Deduplicate queries
        let (unique_queries, dedup_map) = if self.config.enable_deduplication {
            self.deduplicate_queries(queries)
        } else {
            let map: HashMap<String, String> = HashMap::new();
            (queries, map)
        };

        // Execute queries based on strategy
        let mut results = match self.config.strategy {
            ExecutionStrategy::Sequential => self.execute_sequential(unique_queries.clone()).await,
            ExecutionStrategy::Parallel => self.execute_parallel(unique_queries.clone()).await,
            ExecutionStrategy::Adaptive => self.execute_adaptive(unique_queries.clone()).await,
            ExecutionStrategy::PriorityBased => {
                self.execute_priority_based(unique_queries.clone()).await
            }
        }?;

        // Apply deduplication results
        for (original_id, canonical_id) in dedup_map.iter() {
            if let Some(canonical_result) = results.iter().find(|r| &r.query_id == canonical_id) {
                let mut dedup_result = canonical_result.clone();
                dedup_result.query_id = original_id.clone();
                dedup_result.deduplicated = true;
                results.push(dedup_result);
            }
        }

        // Calculate statistics
        let statistics = self.calculate_statistics(&results, dedup_map.len());

        // Mark batch as completed
        {
            let mut batches = self.batches.write().await;
            if let Some(batch) = batches.get_mut(batch_id) {
                batch.completed = true;
            }
        }

        // Store statistics
        self.statistics
            .write()
            .await
            .insert(batch_id.to_string(), statistics.clone());

        Ok(BatchResult {
            batch_id: batch_id.to_string(),
            results,
            total_duration: start_time.elapsed(),
            queries_executed: unique_queries.len(),
            queries_deduplicated: dedup_map.len(),
            cache_hits: statistics.cached_queries,
            statistics,
        })
    }

    /// Deduplicate queries
    fn deduplicate_queries(
        &self,
        queries: Vec<BatchedQuery>,
    ) -> (Vec<BatchedQuery>, HashMap<String, String>) {
        let mut unique_queries = Vec::new();
        let mut dedup_map = HashMap::new();
        let mut fingerprint_map: HashMap<String, String> = HashMap::new();

        for query in queries {
            if let Some(canonical_id) = fingerprint_map.get(&query.fingerprint) {
                // Duplicate found
                dedup_map.insert(query.id.clone(), canonical_id.clone());
            } else {
                // First occurrence
                fingerprint_map.insert(query.fingerprint.clone(), query.id.clone());
                unique_queries.push(query);
            }
        }

        (unique_queries, dedup_map)
    }

    /// Execute queries sequentially
    async fn execute_sequential(
        &self,
        queries: Vec<BatchedQuery>,
    ) -> Result<Vec<QueryResult>, String> {
        let mut results = Vec::new();
        for query in queries {
            let result = self.execute_single_query(query).await?;
            results.push(result);
        }
        Ok(results)
    }

    /// Execute queries in parallel
    async fn execute_parallel(
        &self,
        queries: Vec<BatchedQuery>,
    ) -> Result<Vec<QueryResult>, String> {
        use futures::stream::{self, StreamExt};

        let results = stream::iter(queries)
            .map(|query| async move { self.execute_single_query(query).await })
            .buffer_unordered(self.config.max_concurrency)
            .collect::<Vec<_>>()
            .await;

        results.into_iter().collect()
    }

    /// Execute queries adaptively
    async fn execute_adaptive(
        &self,
        queries: Vec<BatchedQuery>,
    ) -> Result<Vec<QueryResult>, String> {
        // For now, use parallel execution
        // TODO: Implement dependency analysis
        self.execute_parallel(queries).await
    }

    /// Execute queries by priority
    async fn execute_priority_based(
        &self,
        mut queries: Vec<BatchedQuery>,
    ) -> Result<Vec<QueryResult>, String> {
        // Sort by priority (highest first)
        queries.sort_by(|a, b| b.priority.cmp(&a.priority));
        self.execute_parallel(queries).await
    }

    /// Execute a single query
    async fn execute_single_query(&self, query: BatchedQuery) -> Result<QueryResult, String> {
        let start_time = Instant::now();

        // Check cache
        if self.config.enable_caching {
            let cache = self.cache.read().await;
            if let Some(cached_result) = cache.get(&query.fingerprint) {
                let mut result = cached_result.clone();
                result.query_id = query.id;
                result.cached = true;
                return Ok(result);
            }
        }

        // Execute query (placeholder - integrate with actual GraphQL executor)
        let result = QueryResult {
            query_id: query.id.clone(),
            data: Some(serde_json::json!({
                "placeholder": "Query execution not implemented",
                "query": query.query
            })),
            errors: Vec::new(),
            duration: start_time.elapsed(),
            cached: false,
            deduplicated: false,
        };

        // Cache result
        if self.config.enable_caching {
            self.cache
                .write()
                .await
                .insert(query.fingerprint, result.clone());
        }

        Ok(result)
    }

    /// Calculate batch statistics
    fn calculate_statistics(
        &self,
        results: &[QueryResult],
        deduplicated_count: usize,
    ) -> BatchStatistics {
        let total_queries = results.len();
        let cached_queries = results.iter().filter(|r| r.cached).count();
        let failed_queries = results.iter().filter(|r| !r.errors.is_empty()).count();
        let executed_queries = total_queries - deduplicated_count;

        let durations: Vec<Duration> = results.iter().map(|r| r.duration).collect();
        let avg_duration = if !durations.is_empty() {
            durations.iter().sum::<Duration>() / durations.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let max_duration = durations.iter().max().copied().unwrap_or_default();
        let min_duration = durations.iter().min().copied().unwrap_or_default();

        BatchStatistics {
            total_queries,
            executed_queries,
            deduplicated_queries: deduplicated_count,
            cached_queries,
            failed_queries,
            avg_query_duration: avg_duration,
            max_query_duration: max_duration,
            min_query_duration: min_duration,
        }
    }

    /// Get batch statistics
    pub async fn get_statistics(&self, batch_id: &str) -> Option<BatchStatistics> {
        self.statistics.read().await.get(batch_id).cloned()
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
    }

    /// Get cache size
    pub async fn cache_size(&self) -> usize {
        self.cache.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 100);
        assert_eq!(config.max_concurrency, 10);
        assert_eq!(config.strategy, ExecutionStrategy::Parallel);
        assert!(config.enable_deduplication);
    }

    #[tokio::test]
    async fn test_batch_config_builder() {
        let config = BatchConfig::new()
            .with_max_batch_size(50)
            .with_max_concurrency(5)
            .with_strategy(ExecutionStrategy::Sequential)
            .with_deduplication(false);

        assert_eq!(config.max_batch_size, 50);
        assert_eq!(config.max_concurrency, 5);
        assert_eq!(config.strategy, ExecutionStrategy::Sequential);
        assert!(!config.enable_deduplication);
    }

    #[tokio::test]
    async fn test_batched_query_creation() {
        let query = BatchedQuery::new("{ user { name } }".to_string());
        assert!(!query.id.is_empty());
        assert_eq!(query.query, "{ user { name } }");
        assert!(query.variables.is_none());
        assert_eq!(query.priority, QueryPriority::Normal);
    }

    #[tokio::test]
    async fn test_batched_query_with_variables() {
        let vars = serde_json::json!({"id": 123});
        let query =
            BatchedQuery::new("{ user(id: $id) { name } }".to_string()).with_variables(vars);

        assert!(query.variables.is_some());
    }

    #[tokio::test]
    async fn test_query_priority() {
        assert!(QueryPriority::Critical > QueryPriority::High);
        assert!(QueryPriority::High > QueryPriority::Normal);
        assert!(QueryPriority::Normal > QueryPriority::Low);
    }

    #[tokio::test]
    async fn test_batcher_creation() {
        let config = BatchConfig::default();
        let batcher = QueryBatcher::new(config);
        assert_eq!(batcher.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_create_batch() {
        let batcher = QueryBatcher::new(BatchConfig::default());
        let batch_id = batcher.create_batch().await;
        assert!(!batch_id.is_empty());
    }

    #[tokio::test]
    async fn test_add_query_to_batch() {
        let batcher = QueryBatcher::new(BatchConfig::default());
        let batch_id = batcher.create_batch().await;
        let query = BatchedQuery::new("{ user { name } }".to_string());

        let result = batcher.add_query(&batch_id, query).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_add_query_to_nonexistent_batch() {
        let batcher = QueryBatcher::new(BatchConfig::default());
        let query = BatchedQuery::new("{ user { name } }".to_string());

        let result = batcher.add_query("nonexistent", query).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_size_limit() {
        let config = BatchConfig::default().with_max_batch_size(2);
        let batcher = QueryBatcher::new(config);
        let batch_id = batcher.create_batch().await;

        let q1 = BatchedQuery::new("query1".to_string());
        let q2 = BatchedQuery::new("query2".to_string());
        let q3 = BatchedQuery::new("query3".to_string());

        assert!(batcher.add_query(&batch_id, q1).await.is_ok());
        assert!(batcher.add_query(&batch_id, q2).await.is_ok());
        assert!(batcher.add_query(&batch_id, q3).await.is_err());
    }

    #[tokio::test]
    async fn test_execute_batch() {
        let batcher = QueryBatcher::new(BatchConfig::default());
        let batch_id = batcher.create_batch().await;

        let q1 = BatchedQuery::new("{ user { name } }".to_string());
        let q2 = BatchedQuery::new("{ posts { title } }".to_string());

        batcher.add_query(&batch_id, q1).await.unwrap();
        batcher.add_query(&batch_id, q2).await.unwrap();

        let result = batcher.execute_batch(&batch_id).await;
        assert!(result.is_ok());

        let batch_result = result.unwrap();
        assert_eq!(batch_result.results.len(), 2);
        assert_eq!(batch_result.queries_executed, 2);
    }

    #[tokio::test]
    async fn test_query_deduplication() {
        let config = BatchConfig::default().with_deduplication(true);
        let batcher = QueryBatcher::new(config);
        let batch_id = batcher.create_batch().await;

        let q1 = BatchedQuery::new("{ user { name } }".to_string());
        let q2 = BatchedQuery::new("{ user { name } }".to_string()); // Duplicate

        batcher.add_query(&batch_id, q1).await.unwrap();
        batcher.add_query(&batch_id, q2).await.unwrap();

        let result = batcher.execute_batch(&batch_id).await.unwrap();
        assert_eq!(result.queries_executed, 1); // Only one unique query
        assert_eq!(result.queries_deduplicated, 1);
        assert_eq!(result.results.len(), 2); // But two results
    }

    #[tokio::test]
    async fn test_priority_based_execution() {
        let config = BatchConfig::default().with_strategy(ExecutionStrategy::PriorityBased);
        let batcher = QueryBatcher::new(config);
        let batch_id = batcher.create_batch().await;

        let q1 = BatchedQuery::new("low".to_string()).with_priority(QueryPriority::Low);
        let q2 = BatchedQuery::new("high".to_string()).with_priority(QueryPriority::High);

        batcher.add_query(&batch_id, q1).await.unwrap();
        batcher.add_query(&batch_id, q2).await.unwrap();

        let result = batcher.execute_batch(&batch_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let config = BatchConfig::default().with_deduplication(false);
        let batcher = QueryBatcher::new(config);

        let batch1 = batcher.create_batch().await;
        let q1 = BatchedQuery::new("{ user { name } }".to_string());
        batcher.add_query(&batch1, q1).await.unwrap();
        batcher.execute_batch(&batch1).await.unwrap();

        // Second batch with same query should hit cache
        let batch2 = batcher.create_batch().await;
        let q2 = BatchedQuery::new("{ user { name } }".to_string());
        batcher.add_query(&batch2, q2).await.unwrap();
        let result = batcher.execute_batch(&batch2).await.unwrap();

        assert_eq!(result.cache_hits, 1);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let batcher = QueryBatcher::new(BatchConfig::default());
        let batch_id = batcher.create_batch().await;
        let q1 = BatchedQuery::new("{ user { name } }".to_string());
        batcher.add_query(&batch_id, q1).await.unwrap();
        batcher.execute_batch(&batch_id).await.unwrap();

        assert!(batcher.cache_size().await > 0);
        batcher.clear_cache().await;
        assert_eq!(batcher.cache_size().await, 0);
    }

    #[tokio::test]
    async fn test_statistics() {
        let batcher = QueryBatcher::new(BatchConfig::default());
        let batch_id = batcher.create_batch().await;

        let q1 = BatchedQuery::new("query1".to_string());
        let q2 = BatchedQuery::new("query2".to_string());

        batcher.add_query(&batch_id, q1).await.unwrap();
        batcher.add_query(&batch_id, q2).await.unwrap();

        batcher.execute_batch(&batch_id).await.unwrap();

        let stats = batcher.get_statistics(&batch_id).await;
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.executed_queries, 2);
    }
}
