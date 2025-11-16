//! Request Batching and Parallel Execution
//!
//! This module provides sophisticated batching and parallel execution for SPARQL queries:
//! - Automatic query batching for improved throughput
//! - Parallel execution of independent queries
//! - Query dependency resolution and ordering
//! - Batch optimization and result merging
//! - Adaptive batch sizing based on load

use crate::error::{FusekiError, FusekiResult};
use scirs2_core::metrics::{Counter, Histogram, Timer};
use scirs2_core::parallel_ops::{par_chunks, par_join};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// Batch execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable request batching
    pub enabled: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size before execution
    pub min_batch_size: usize,
    /// Maximum wait time for batch to fill (milliseconds)
    pub max_wait_time_ms: u64,
    /// Enable adaptive batch sizing
    pub adaptive_sizing: bool,
    /// Maximum parallel batches
    pub max_parallel_batches: usize,
    /// Enable query dependency analysis
    pub analyze_dependencies: bool,
    /// Maximum parallel queries within a batch
    pub max_parallel_queries: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            enabled: true,
            max_batch_size: 100,
            min_batch_size: 10,
            max_wait_time_ms: 100,
            adaptive_sizing: true,
            max_parallel_batches: 4,
            analyze_dependencies: true,
            max_parallel_queries: 20,
        }
    }
}

/// Query execution request for batching
#[derive(Debug, Clone)]
pub struct BatchQuery {
    pub id: String,
    pub dataset: String,
    pub query: String,
    pub user_id: Option<String>,
    pub submitted_at: Instant,
    pub timeout: Duration,
    pub metadata: HashMap<String, String>,
}

impl BatchQuery {
    pub fn new(query: String, dataset: String) -> Self {
        BatchQuery {
            id: Uuid::new_v4().to_string(),
            dataset,
            query,
            user_id: None,
            submitted_at: Instant::now(),
            timeout: Duration::from_secs(30),
            metadata: HashMap::new(),
        }
    }
}

/// Query result for a batched query
#[derive(Debug, Clone)]
pub struct BatchQueryResult {
    pub query_id: String,
    pub success: bool,
    pub result: Option<String>,
    pub error: Option<String>,
    pub execution_time: Duration,
}

/// Query batch for parallel execution
#[derive(Debug)]
pub struct QueryBatch {
    pub id: String,
    pub queries: Vec<BatchQuery>,
    pub created_at: Instant,
    pub dataset: String,
}

impl QueryBatch {
    pub fn new(dataset: String) -> Self {
        QueryBatch {
            id: Uuid::new_v4().to_string(),
            queries: Vec::new(),
            created_at: Instant::now(),
            dataset,
        }
    }

    pub fn add_query(&mut self, query: BatchQuery) {
        self.queries.push(query);
    }

    pub fn is_full(&self, max_size: usize) -> bool {
        self.queries.len() >= max_size
    }

    pub fn size(&self) -> usize {
        self.queries.len()
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Batch execution statistics
#[derive(Debug, Clone, Serialize)]
pub struct BatchStats {
    pub total_batches: u64,
    pub total_queries: u64,
    pub average_batch_size: f64,
    pub average_wait_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub parallel_efficiency: f64,
    pub queries_per_second: f64,
}

/// Batch executor for parallel query execution
pub struct BatchExecutor {
    config: BatchConfig,

    // Per-dataset batch queues
    dataset_batches: Arc<RwLock<HashMap<String, Arc<RwLock<QueryBatch>>>>>,

    // Pending queries waiting to be batched
    pending_queries: Arc<RwLock<VecDeque<BatchQuery>>>,

    // Active batches being executed
    active_batches: Arc<RwLock<HashMap<String, JoinHandle<()>>>>,

    // Result channels for completed queries
    result_channels: Arc<RwLock<HashMap<String, tokio::sync::oneshot::Sender<BatchQueryResult>>>>,

    // Statistics
    stats: Arc<RwLock<BatchStats>>,
    total_batches: Arc<AtomicU64>,
    total_queries: Arc<AtomicU64>,

    // Batch semaphore for limiting parallel batches
    batch_semaphore: Arc<Semaphore>,

    // Notify for new queries
    new_query_notify: Arc<Notify>,

    // Shutdown signal
    shutdown: Arc<tokio::sync::watch::Sender<bool>>,
}

impl BatchExecutor {
    /// Create a new batch executor
    pub fn new(config: BatchConfig) -> Arc<Self> {
        let batch_semaphore = Arc::new(Semaphore::new(config.max_parallel_batches));

        let (shutdown_tx, _) = tokio::sync::watch::channel(false);

        let executor = Arc::new(BatchExecutor {
            config,
            dataset_batches: Arc::new(RwLock::new(HashMap::new())),
            pending_queries: Arc::new(RwLock::new(VecDeque::new())),
            active_batches: Arc::new(RwLock::new(HashMap::new())),
            result_channels: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(BatchStats {
                total_batches: 0,
                total_queries: 0,
                average_batch_size: 0.0,
                average_wait_time_ms: 0.0,
                average_execution_time_ms: 0.0,
                parallel_efficiency: 1.0,
                queries_per_second: 0.0,
            })),
            total_batches: Arc::new(AtomicU64::new(0)),
            total_queries: Arc::new(AtomicU64::new(0)),
            batch_semaphore,
            new_query_notify: Arc::new(Notify::new()),
            shutdown: Arc::new(shutdown_tx),
        });

        // Start background batch processor
        executor.clone().start_batch_processor();

        info!(
            "Batch executor initialized with max {} queries per batch",
            executor.config.max_batch_size
        );

        executor
    }

    /// Submit a query for batched execution
    #[instrument(skip(self, query))]
    pub async fn submit_query(&self, query: BatchQuery) -> FusekiResult<BatchQueryResult> {
        if !self.config.enabled {
            // Execute immediately without batching
            return self.execute_single_query(query).await;
        }

        // Create result channel
        let (tx, rx) = tokio::sync::oneshot::channel();

        let query_id = query.id.clone();

        // Store result channel
        {
            let mut channels = self.result_channels.write().await;
            channels.insert(query_id.clone(), tx);
        }

        // Add to pending queue
        {
            let mut pending = self.pending_queries.write().await;
            pending.push_back(query);
        }

        // Notify batch processor
        self.new_query_notify.notify_one();

        self.total_queries.fetch_add(1, Ordering::Relaxed);

        // Wait for result
        let result = tokio::time::timeout(
            Duration::from_secs(60), // Global timeout
            rx,
        )
        .await
        .map_err(|_| FusekiError::request_timeout("Batch execution timeout"))?
        .map_err(|_| FusekiError::server_error("Result channel closed"))?;

        Ok(result)
    }

    /// Start background batch processor
    fn start_batch_processor(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut shutdown_rx = self.shutdown.subscribe();
            let mut ticker = tokio::time::interval(Duration::from_millis(10));

            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    _ = ticker.tick() => {
                        self.process_pending_queries().await;
                        self.check_batch_timeouts().await;
                    }
                    _ = self.new_query_notify.notified() => {
                        self.process_pending_queries().await;
                    }
                }
            }
        });
    }

    /// Process pending queries into batches
    async fn process_pending_queries(&self) {
        let mut pending = self.pending_queries.write().await;
        #[allow(unused_mut)]
        let mut batches = self.dataset_batches.write().await;

        while let Some(query) = pending.pop_front() {
            let dataset = query.dataset.clone();

            // Get or create batch for dataset
            let batch = batches
                .entry(dataset.clone())
                .or_insert_with(|| Arc::new(RwLock::new(QueryBatch::new(dataset.clone()))));

            let mut batch_lock = batch.write().await;

            // Add query to batch
            batch_lock.add_query(query);

            // Check if batch should be executed
            if batch_lock.is_full(self.config.max_batch_size)
                || (batch_lock.size() >= self.config.min_batch_size
                    && batch_lock.age() >= Duration::from_millis(self.config.max_wait_time_ms))
            {
                // Take the batch for execution
                let executing_batch =
                    std::mem::replace(&mut *batch_lock, QueryBatch::new(dataset.clone()));

                drop(batch_lock); // Release lock before execution

                // Execute batch directly
                self.execute_batch_impl(executing_batch).await;
            }
        }
    }

    /// Check for batches that have timed out
    async fn check_batch_timeouts(&self) {
        #[allow(unused_mut)]
        let mut batches = self.dataset_batches.write().await;
        let max_wait = Duration::from_millis(self.config.max_wait_time_ms);

        let datasets_to_execute: Vec<String> = batches
            .iter()
            .filter_map(|(dataset, batch_arc)| {
                // Use try_read to avoid deadlock
                if let Ok(batch) = batch_arc.try_read() {
                    if batch.size() >= self.config.min_batch_size && batch.age() >= max_wait {
                        return Some(dataset.clone());
                    }
                }
                None
            })
            .collect();

        for dataset in datasets_to_execute {
            if let Some(batch_arc) = batches.get(&dataset) {
                let mut batch_lock = batch_arc.write().await;

                if batch_lock.size() > 0 {
                    let executing_batch =
                        std::mem::replace(&mut *batch_lock, QueryBatch::new(dataset.clone()));

                    drop(batch_lock);

                    // Execute batch directly
                    self.execute_batch_impl(executing_batch).await;
                }
            }
        }
    }

    /// Execute a batch of queries in parallel
    #[instrument(skip(self, batch))]
    async fn execute_batch_impl(&self, batch: QueryBatch) {
        let batch_id = batch.id.clone();
        let batch_size = batch.size();

        debug!("Executing batch {} with {} queries", batch_id, batch_size);

        // Acquire batch execution permit
        let _permit = self.batch_semaphore.acquire().await.unwrap();

        self.total_batches.fetch_add(1, Ordering::Relaxed);

        let batch_start = Instant::now();

        // Execute queries in parallel using scirs2-core parallel ops
        let queries = batch.queries;
        let chunk_size = (batch_size / self.config.max_parallel_queries).max(1);

        let results: Vec<BatchQueryResult> = if self.config.analyze_dependencies {
            // Execute with dependency analysis
            self.execute_with_dependencies(queries).await
        } else {
            // Execute all queries in parallel
            self.execute_parallel(queries).await
        };

        let batch_duration = batch_start.elapsed();

        // Send results back to waiting queries
        for result in results {
            let query_id = result.query_id.clone();

            if let Some(tx) = self.result_channels.write().await.remove(&query_id) {
                let _ = tx.send(result);
            }
        }

        info!(
            "Batch {} completed in {:.2}ms ({} queries, {:.2} q/s)",
            batch_id,
            batch_duration.as_millis(),
            batch_size,
            (batch_size as f64) / batch_duration.as_secs_f64()
        );

        // Update statistics
        self.update_batch_stats(batch_size, batch_duration).await;
    }

    /// Execute queries in parallel without dependency analysis
    async fn execute_parallel(&self, queries: Vec<BatchQuery>) -> Vec<BatchQueryResult> {
        let tasks: Vec<_> = queries
            .into_iter()
            .map(|query| tokio::spawn(async move { Self::execute_query_impl(query).await }))
            .collect();

        let mut results = Vec::new();
        for task in tasks {
            if let Ok(result) = task.await {
                results.push(result);
            }
        }

        results
    }

    /// Execute queries with dependency analysis
    async fn execute_with_dependencies(&self, queries: Vec<BatchQuery>) -> Vec<BatchQueryResult> {
        if queries.is_empty() {
            return Vec::new();
        }

        // Build dependency graph
        let dependency_graph = self.build_dependency_graph(&queries);

        // Topologically sort queries based on dependencies
        let execution_order = self.topological_sort(&dependency_graph, queries.len());

        // Execute queries in dependency order, parallelizing independent ones
        self.execute_in_dependency_order(queries, execution_order, &dependency_graph)
            .await
    }

    /// Build dependency graph for queries
    fn build_dependency_graph(&self, queries: &[BatchQuery]) -> Vec<Vec<usize>> {
        let n = queries.len();
        let mut graph = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                // Check if query j depends on query i
                if self.has_dependency(&queries[i], &queries[j]) {
                    graph[i].push(j); // j depends on i
                }
                // Check if query i depends on query j
                else if self.has_dependency(&queries[j], &queries[i]) {
                    graph[j].push(i); // i depends on j
                }
            }
        }

        graph
    }

    /// Check if query2 depends on query1
    fn has_dependency(&self, query1: &BatchQuery, query2: &BatchQuery) -> bool {
        // Analyze query text to detect dependencies
        let q1_lower = query1.query.to_lowercase();
        let q2_lower = query2.query.to_lowercase();

        // If query1 is an UPDATE and query2 is a SELECT, there's a dependency
        if (q1_lower.contains("insert")
            || q1_lower.contains("delete")
            || q1_lower.contains("update"))
            && q2_lower.contains("select")
        {
            // Check if they reference the same graphs/subjects
            return self.queries_reference_same_data(&q1_lower, &q2_lower);
        }

        // If both are UPDATEs, they conflict and should be serialized
        if (q1_lower.contains("insert") || q1_lower.contains("delete"))
            && (q2_lower.contains("insert") || q2_lower.contains("delete"))
        {
            return self.queries_reference_same_data(&q1_lower, &q2_lower);
        }

        false
    }

    /// Check if queries reference the same data
    fn queries_reference_same_data(&self, query1: &str, query2: &str) -> bool {
        // Extract graph names from queries
        let graph1 = self.extract_graph_references(query1);
        let graph2 = self.extract_graph_references(query2);

        // If either references the default graph (empty), they might conflict
        if graph1.is_empty() || graph2.is_empty() {
            return true;
        }

        // Check for overlap in graph references
        for g1 in &graph1 {
            if graph2.contains(g1) {
                return true;
            }
        }

        false
    }

    /// Extract graph references from query
    fn extract_graph_references(&self, query: &str) -> Vec<String> {
        let mut graphs = Vec::new();

        // Simple regex-like pattern matching for GRAPH clauses
        let mut current = query;
        while let Some(pos) = current.find("graph") {
            let after = &current[pos + 5..];
            if let Some(start) = after.find('<') {
                if let Some(end) = after[start..].find('>') {
                    let graph_iri = after[start + 1..start + end].to_string();
                    graphs.push(graph_iri);
                    current = &after[start + end..];
                    continue;
                }
            }
            break;
        }

        graphs
    }

    /// Topologically sort queries
    fn topological_sort(&self, graph: &[Vec<usize>], n: usize) -> Vec<usize> {
        let mut in_degree = vec![0; n];
        let mut result = Vec::new();

        // Calculate in-degrees
        for edges in graph {
            for &dest in edges {
                in_degree[dest] += 1;
            }
        }

        // Queue of nodes with no incoming edges
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

        while !queue.is_empty() {
            // Process nodes in batches (these can run in parallel)
            let batch: Vec<usize> = std::mem::take(&mut queue);
            result.extend(&batch);

            // Reduce in-degrees for neighbors
            for &node in &batch {
                for &neighbor in &graph[node] {
                    in_degree[neighbor] -= 1;
                    if in_degree[neighbor] == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }

        // If we didn't process all nodes, there's a cycle - fall back to sequential
        if result.len() != n {
            (0..n).collect()
        } else {
            result
        }
    }

    /// Execute queries in dependency order
    async fn execute_in_dependency_order(
        &self,
        queries: Vec<BatchQuery>,
        execution_order: Vec<usize>,
        graph: &[Vec<usize>],
    ) -> Vec<BatchQueryResult> {
        let mut results = vec![None; queries.len()];
        let mut in_degree = vec![0; queries.len()];

        // Calculate in-degrees
        for edges in graph {
            for &dest in edges {
                in_degree[dest] += 1;
            }
        }

        // Process queries in batches based on dependency levels
        let mut completed = vec![false; queries.len()];
        let mut queue: Vec<usize> = (0..queries.len()).filter(|&i| in_degree[i] == 0).collect();

        while !queue.is_empty() {
            // Execute current batch in parallel
            let batch_queries: Vec<BatchQuery> =
                queue.iter().map(|&idx| queries[idx].clone()).collect();

            let batch_results = self.execute_parallel(batch_queries).await;

            // Store results
            for (i, &idx) in queue.iter().enumerate() {
                results[idx] = Some(batch_results[i].clone());
                completed[idx] = true;
            }

            // Update in-degrees and find next batch
            let current_batch = queue.clone();
            queue.clear();

            for &node in &current_batch {
                for &neighbor in &graph[node] {
                    in_degree[neighbor] -= 1;
                    if in_degree[neighbor] == 0 && !completed[neighbor] {
                        queue.push(neighbor);
                    }
                }
            }
        }

        results.into_iter().flatten().collect()
    }

    /// Execute a single query (implementation)
    async fn execute_query_impl(query: BatchQuery) -> BatchQueryResult {
        let start = Instant::now();

        // Simulate query execution
        // TODO: Replace with actual SPARQL execution
        tokio::time::sleep(Duration::from_millis(10)).await;

        let execution_time = start.elapsed();

        BatchQueryResult {
            query_id: query.id,
            success: true,
            result: Some("Query result".to_string()),
            error: None,
            execution_time,
        }
    }

    /// Execute a single query immediately
    async fn execute_single_query(&self, query: BatchQuery) -> FusekiResult<BatchQueryResult> {
        Ok(Self::execute_query_impl(query).await)
    }

    /// Update batch statistics
    async fn update_batch_stats(&self, batch_size: usize, duration: Duration) {
        let mut stats = self.stats.write().await;

        let total_batches = self.total_batches.load(Ordering::Relaxed);
        let total_queries = self.total_queries.load(Ordering::Relaxed);

        stats.total_batches = total_batches;
        stats.total_queries = total_queries;

        if total_batches > 0 {
            stats.average_batch_size = (total_queries as f64) / (total_batches as f64);
        }

        // Update execution time (exponential moving average)
        let alpha = 0.1;
        let new_exec_time = duration.as_millis() as f64;
        stats.average_execution_time_ms =
            alpha * new_exec_time + (1.0 - alpha) * stats.average_execution_time_ms;

        // Calculate parallel efficiency
        // Efficiency = (sequential_time / parallel_time) / num_parallel_queries
        let sequential_estimate = batch_size as f64 * 10.0; // Assume 10ms per query
        let parallel_time = duration.as_millis() as f64;
        if parallel_time > 0.0 {
            stats.parallel_efficiency = (sequential_estimate / parallel_time)
                / (batch_size as f64).min(self.config.max_parallel_queries as f64);
        }

        // Calculate throughput
        if duration.as_secs_f64() > 0.0 {
            stats.queries_per_second = (batch_size as f64) / duration.as_secs_f64();
        }
    }

    /// Get batch execution statistics
    pub async fn get_stats(&self) -> BatchStats {
        self.stats.read().await.clone()
    }

    /// Shutdown the executor
    pub async fn shutdown(&self) {
        info!("Shutting down batch executor");
        let _ = self.shutdown.send(true);

        // Wait for active batches to complete
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_executor_creation() {
        let config = BatchConfig::default();
        let executor = BatchExecutor::new(config);

        let stats = executor.get_stats().await;
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.total_queries, 0);
    }

    #[tokio::test]
    async fn test_single_query_execution() {
        let config = BatchConfig {
            enabled: false,
            ..Default::default()
        };
        let executor = BatchExecutor::new(config);

        let query = BatchQuery::new(
            "SELECT * WHERE { ?s ?p ?o }".to_string(),
            "test".to_string(),
        );

        let result = executor.submit_query(query).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_batched_execution() {
        let config = BatchConfig {
            enabled: true,
            max_batch_size: 5,
            min_batch_size: 2,
            max_wait_time_ms: 50,
            ..Default::default()
        };
        let executor = BatchExecutor::new(config);

        // Submit multiple queries
        let mut handles = Vec::new();
        for i in 0..5 {
            let executor = executor.clone();
            let handle = tokio::spawn(async move {
                let query = BatchQuery::new(
                    format!("SELECT * WHERE {{ ?s ?p ?o }} # Query {}", i),
                    "test".to_string(),
                );
                executor.submit_query(query).await
            });
            handles.push(handle);
        }

        // Wait for all queries
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        // Check stats
        let stats = executor.get_stats().await;
        assert!(stats.total_queries >= 5);
    }
}
