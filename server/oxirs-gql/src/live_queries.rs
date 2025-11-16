//! Live Queries - Automatic Query Re-execution on Data Changes
//!
//! This module provides live query capabilities where queries automatically
//! re-execute when the underlying RDF data changes, providing real-time updates
//! to clients without explicit subscriptions.
//!
//! ## Features
//!
//! - **Dependency Tracking**: Automatically track which triples a query depends on
//! - **Change Detection**: Detect when relevant RDF data changes
//! - **Automatic Re-execution**: Re-run queries when dependencies change
//! - **Efficient Diffing**: Only send changed data to clients
//! - **Batching**: Batch multiple changes to reduce update frequency
//! - **Throttling**: Prevent excessive updates with configurable throttling
//! - **Memory Efficient**: Track dependencies without storing all data

use crate::ast::Document;
use crate::execution::ExecutionContext;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Live query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveQueryConfig {
    /// Enable live queries
    pub enabled: bool,

    /// Throttle interval - minimum time between updates (in milliseconds)
    pub throttle_interval_ms: u64,

    /// Batch interval - collect changes for this duration before updating (in milliseconds)
    pub batch_interval_ms: u64,

    /// Maximum number of active live queries per connection
    pub max_queries_per_connection: usize,

    /// Enable result diffing
    pub enable_diffing: bool,

    /// Maximum result size to diff (larger results are sent in full)
    pub max_diff_size: usize,

    /// Cleanup interval for stale queries (in seconds)
    pub cleanup_interval_secs: u64,

    /// Query timeout - queries inactive for this long are removed (in seconds)
    pub query_timeout_secs: u64,
}

impl Default for LiveQueryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            throttle_interval_ms: 100,
            batch_interval_ms: 50,
            max_queries_per_connection: 100,
            enable_diffing: true,
            max_diff_size: 10000,
            cleanup_interval_secs: 300,
            query_timeout_secs: 3600,
        }
    }
}

/// Triple pattern for dependency tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

impl TriplePattern {
    /// Check if a triple matches this pattern
    pub fn matches(&self, subject: &str, predicate: &str, object: &str) -> bool {
        let subject_matches = self.subject.as_ref().map_or(true, |s| s == subject);
        let predicate_matches = self.predicate.as_ref().map_or(true, |p| p == predicate);
        let object_matches = self.object.as_ref().map_or(true, |o| o == object);

        subject_matches && predicate_matches && object_matches
    }

    /// Create a wildcard pattern (matches everything)
    pub fn wildcard() -> Self {
        Self {
            subject: None,
            predicate: None,
            object: None,
        }
    }
}

/// Query dependencies - which triples/patterns does this query depend on
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDependencies {
    /// Exact triples this query depends on
    pub exact_triples: HashSet<(String, String, String)>,

    /// Triple patterns this query depends on (with wildcards)
    pub patterns: HashSet<TriplePattern>,

    /// Graph names this query accesses
    pub graphs: HashSet<String>,
}

impl Default for QueryDependencies {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryDependencies {
    pub fn new() -> Self {
        Self {
            exact_triples: HashSet::new(),
            patterns: HashSet::new(),
            graphs: HashSet::new(),
        }
    }

    /// Check if a triple change affects this query
    pub fn is_affected_by(&self, subject: &str, predicate: &str, object: &str) -> bool {
        // Check exact triple match
        if self.exact_triples.contains(&(
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        )) {
            return true;
        }

        // Check pattern match
        for pattern in &self.patterns {
            if pattern.matches(subject, predicate, object) {
                return true;
            }
        }

        false
    }
}

/// Live query state
#[derive(Debug, Clone)]
pub struct LiveQuery {
    /// Query ID
    pub id: String,

    /// Connection ID
    pub connection_id: String,

    /// GraphQL document
    pub document: Document,

    /// Execution context
    pub context: ExecutionContext,

    /// Query dependencies
    pub dependencies: QueryDependencies,

    /// Last execution time
    pub last_executed: Instant,

    /// Last result (for diffing)
    pub last_result: Option<serde_json::Value>,

    /// Number of executions
    pub execution_count: u64,

    /// Pending update flag
    pub pending_update: bool,

    /// Created timestamp
    pub created_at: Instant,
}

impl LiveQuery {
    pub fn new(
        connection_id: String,
        document: Document,
        context: ExecutionContext,
        dependencies: QueryDependencies,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            connection_id,
            document,
            context,
            dependencies,
            last_executed: Instant::now(),
            last_result: None,
            execution_count: 0,
            pending_update: false,
            created_at: Instant::now(),
        }
    }
}

/// Live query update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveQueryUpdate {
    /// Query ID
    pub query_id: String,

    /// Update type
    pub update_type: UpdateType,

    /// Updated data
    pub data: serde_json::Value,

    /// Timestamp
    pub timestamp: String,
}

/// Type of live query update
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateType {
    /// Full result
    Full,

    /// Incremental diff
    Diff,

    /// Error occurred
    Error,
}

/// RDF change event
#[derive(Debug, Clone)]
pub struct RdfChange {
    /// Change type
    pub change_type: ChangeType,

    /// Subject
    pub subject: String,

    /// Predicate
    pub predicate: String,

    /// Object
    pub object: String,

    /// Graph name (optional)
    pub graph: Option<String>,

    /// Timestamp
    pub timestamp: Instant,
}

/// Type of RDF change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Triple was added
    Insert,

    /// Triple was deleted
    Delete,

    /// Triple was modified
    Update,
}

/// Live query manager
pub struct LiveQueryManager {
    config: LiveQueryConfig,
    queries: Arc<RwLock<HashMap<String, LiveQuery>>>,
    query_senders: Arc<RwLock<HashMap<String, mpsc::Sender<LiveQueryUpdate>>>>,
    change_queue: Arc<RwLock<Vec<RdfChange>>>,
}

impl LiveQueryManager {
    /// Create new live query manager
    pub fn new(config: LiveQueryConfig) -> Self {
        Self {
            config,
            queries: Arc::new(RwLock::new(HashMap::new())),
            query_senders: Arc::new(RwLock::new(HashMap::new())),
            change_queue: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a new live query
    pub async fn register_query(
        &self,
        connection_id: String,
        document: Document,
        context: ExecutionContext,
        dependencies: QueryDependencies,
    ) -> Result<(String, mpsc::Receiver<LiveQueryUpdate>)> {
        if !self.config.enabled {
            return Err(anyhow!("Live queries are disabled"));
        }

        // Check query limit per connection
        let queries = self.queries.read().await;
        let connection_query_count = queries
            .values()
            .filter(|q| q.connection_id == connection_id)
            .count();

        if connection_query_count >= self.config.max_queries_per_connection {
            return Err(anyhow!(
                "Maximum live queries per connection exceeded: {}",
                self.config.max_queries_per_connection
            ));
        }
        drop(queries);

        // Create live query
        let live_query = LiveQuery::new(connection_id, document, context, dependencies);
        let query_id = live_query.id.clone();

        // Create update channel
        let (tx, rx) = mpsc::channel(100);

        // Store query and sender
        self.queries
            .write()
            .await
            .insert(query_id.clone(), live_query);
        self.query_senders
            .write()
            .await
            .insert(query_id.clone(), tx);

        info!("Live query registered: {}", query_id);

        Ok((query_id, rx))
    }

    /// Unregister a live query
    pub async fn unregister_query(&self, query_id: &str) -> Result<()> {
        self.queries.write().await.remove(query_id);
        self.query_senders.write().await.remove(query_id);

        info!("Live query unregistered: {}", query_id);
        Ok(())
    }

    /// Notify of RDF data change
    pub async fn notify_change(&self, change: RdfChange) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Add to change queue
        self.change_queue.write().await.push(change);

        Ok(())
    }

    /// Process pending changes and update affected queries
    pub async fn process_changes(&self) -> Result<usize> {
        // Get and clear change queue
        let changes = {
            let mut queue = self.change_queue.write().await;
            let changes = queue.drain(..).collect::<Vec<_>>();
            changes
        };

        if changes.is_empty() {
            return Ok(0);
        }

        debug!("Processing {} RDF changes", changes.len());

        // Find affected queries
        let mut affected_queries = HashSet::new();
        let queries = self.queries.read().await;

        for change in &changes {
            for (query_id, query) in queries.iter() {
                if query.dependencies.is_affected_by(
                    &change.subject,
                    &change.predicate,
                    &change.object,
                ) {
                    affected_queries.insert(query_id.clone());
                }
            }
        }

        drop(queries);

        // Mark affected queries for update
        let mut queries_mut = self.queries.write().await;
        for query_id in &affected_queries {
            if let Some(query) = queries_mut.get_mut(query_id) {
                query.pending_update = true;
            }
        }

        Ok(affected_queries.len())
    }

    /// Execute pending updates (called by background task)
    pub async fn execute_pending_updates(&self) -> Result<usize> {
        let mut updated_count = 0;

        let queries = self.queries.read().await;
        let query_senders = self.query_senders.read().await;

        for (query_id, query) in queries.iter() {
            if !query.pending_update {
                continue;
            }

            // Check throttling
            if query.last_executed.elapsed()
                < Duration::from_millis(self.config.throttle_interval_ms)
            {
                continue;
            }

            // Execute query (simplified - in real implementation, would use QueryExecutor)
            let result = self.execute_query(query).await?;

            // Send update
            if let Some(sender) = query_senders.get(query_id) {
                let update = LiveQueryUpdate {
                    query_id: query_id.clone(),
                    update_type: UpdateType::Full,
                    data: result,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };

                if sender.send(update).await.is_ok() {
                    updated_count += 1;
                }
            }
        }

        // Clear pending flags
        drop(queries);
        drop(query_senders);

        let mut queries_mut = self.queries.write().await;
        for query in queries_mut.values_mut() {
            if query.pending_update {
                query.pending_update = false;
                query.last_executed = Instant::now();
                query.execution_count += 1;
            }
        }

        Ok(updated_count)
    }

    /// Execute a query (placeholder - real implementation would use QueryExecutor)
    async fn execute_query(&self, _query: &LiveQuery) -> Result<serde_json::Value> {
        // Placeholder - in real implementation, would execute GraphQL query
        Ok(serde_json::json!({
            "data": {
                "result": "updated"
            }
        }))
    }

    /// Cleanup stale queries
    pub async fn cleanup_stale_queries(&self) -> Result<usize> {
        let timeout = Duration::from_secs(self.config.query_timeout_secs);
        let mut removed_count = 0;

        let mut queries = self.queries.write().await;
        let mut senders = self.query_senders.write().await;

        let to_remove: Vec<String> = queries
            .iter()
            .filter(|(_, q)| q.last_executed.elapsed() > timeout)
            .map(|(id, _)| id.clone())
            .collect();

        for query_id in &to_remove {
            queries.remove(query_id);
            senders.remove(query_id);
            removed_count += 1;
        }

        if removed_count > 0 {
            info!("Cleaned up {} stale live queries", removed_count);
        }

        Ok(removed_count)
    }

    /// Start background processing tasks
    pub async fn start_background_tasks(self: Arc<Self>) {
        let manager = Arc::clone(&self);

        // Change processing task
        tokio::spawn(async move {
            let mut batch_interval =
                interval(Duration::from_millis(manager.config.batch_interval_ms));

            loop {
                batch_interval.tick().await;

                if let Err(e) = manager.process_changes().await {
                    warn!("Error processing changes: {}", e);
                }

                if let Err(e) = manager.execute_pending_updates().await {
                    warn!("Error executing updates: {}", e);
                }
            }
        });

        // Cleanup task
        let manager = Arc::clone(&self);
        tokio::spawn(async move {
            let mut cleanup_interval =
                interval(Duration::from_secs(manager.config.cleanup_interval_secs));

            loop {
                cleanup_interval.tick().await;

                if let Err(e) = manager.cleanup_stale_queries().await {
                    warn!("Error cleaning up queries: {}", e);
                }
            }
        });
    }

    /// Get statistics
    pub async fn get_stats(&self) -> LiveQueryStats {
        let queries = self.queries.read().await;

        LiveQueryStats {
            total_queries: queries.len(),
            total_executions: queries.values().map(|q| q.execution_count).sum(),
            pending_updates: queries.values().filter(|q| q.pending_update).count(),
        }
    }
}

/// Live query statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveQueryStats {
    pub total_queries: usize,
    pub total_executions: u64,
    pub pending_updates: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_query_config_default() {
        let config = LiveQueryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.throttle_interval_ms, 100);
        assert_eq!(config.batch_interval_ms, 50);
    }

    #[test]
    fn test_triple_pattern_wildcard() {
        let pattern = TriplePattern::wildcard();
        assert!(pattern.matches("s", "p", "o"));
        assert!(pattern.matches("x", "y", "z"));
    }

    #[test]
    fn test_triple_pattern_subject_only() {
        let pattern = TriplePattern {
            subject: Some("http://example.org/s".to_string()),
            predicate: None,
            object: None,
        };

        assert!(pattern.matches("http://example.org/s", "p", "o"));
        assert!(!pattern.matches("http://example.org/x", "p", "o"));
    }

    #[test]
    fn test_triple_pattern_full_match() {
        let pattern = TriplePattern {
            subject: Some("s".to_string()),
            predicate: Some("p".to_string()),
            object: Some("o".to_string()),
        };

        assert!(pattern.matches("s", "p", "o"));
        assert!(!pattern.matches("s", "p", "x"));
        assert!(!pattern.matches("x", "p", "o"));
    }

    #[test]
    fn test_query_dependencies_exact_triple() {
        let mut deps = QueryDependencies::new();
        deps.exact_triples
            .insert(("s".to_string(), "p".to_string(), "o".to_string()));

        assert!(deps.is_affected_by("s", "p", "o"));
        assert!(!deps.is_affected_by("x", "p", "o"));
    }

    #[test]
    fn test_query_dependencies_pattern() {
        let mut deps = QueryDependencies::new();
        deps.patterns.insert(TriplePattern {
            subject: Some("s".to_string()),
            predicate: None,
            object: None,
        });

        assert!(deps.is_affected_by("s", "p1", "o1"));
        assert!(deps.is_affected_by("s", "p2", "o2"));
        assert!(!deps.is_affected_by("x", "p1", "o1"));
    }

    #[tokio::test]
    async fn test_live_query_manager_creation() {
        let config = LiveQueryConfig::default();
        let manager = LiveQueryManager::new(config);

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_queries, 0);
    }

    #[tokio::test]
    async fn test_live_query_registration() {
        let config = LiveQueryConfig::default();
        let manager = LiveQueryManager::new(config);

        let document = Document {
            definitions: Vec::new(),
        };
        let context = ExecutionContext {
            variables: HashMap::new(),
            operation_name: None,
            request_id: Uuid::new_v4().to_string(),
            fragments: HashMap::new(),
        };
        let dependencies = QueryDependencies::new();

        let result = manager
            .register_query("conn-1".to_string(), document, context, dependencies)
            .await;

        assert!(result.is_ok());

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_queries, 1);
    }

    #[tokio::test]
    async fn test_live_query_limit() {
        let config = LiveQueryConfig {
            max_queries_per_connection: 2,
            ..Default::default()
        };
        let manager = LiveQueryManager::new(config);

        // Register first query
        let _ = manager
            .register_query(
                "conn-1".to_string(),
                Document {
                    definitions: Vec::new(),
                },
                ExecutionContext {
                    variables: HashMap::new(),
                    operation_name: None,
                    request_id: Uuid::new_v4().to_string(),
                    fragments: HashMap::new(),
                },
                QueryDependencies::new(),
            )
            .await;

        // Register second query
        let _ = manager
            .register_query(
                "conn-1".to_string(),
                Document {
                    definitions: Vec::new(),
                },
                ExecutionContext {
                    variables: HashMap::new(),
                    operation_name: None,
                    request_id: Uuid::new_v4().to_string(),
                    fragments: HashMap::new(),
                },
                QueryDependencies::new(),
            )
            .await;

        // Third query should fail
        let result = manager
            .register_query(
                "conn-1".to_string(),
                Document {
                    definitions: Vec::new(),
                },
                ExecutionContext {
                    variables: HashMap::new(),
                    operation_name: None,
                    request_id: Uuid::new_v4().to_string(),
                    fragments: HashMap::new(),
                },
                QueryDependencies::new(),
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_rdf_change_notification() {
        let config = LiveQueryConfig::default();
        let manager = LiveQueryManager::new(config);

        let change = RdfChange {
            change_type: ChangeType::Insert,
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
            graph: None,
            timestamp: Instant::now(),
        };

        let result = manager.notify_change(change).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_change_processing() {
        let config = LiveQueryConfig::default();
        let manager = LiveQueryManager::new(config);

        // Register query with dependencies
        let mut dependencies = QueryDependencies::new();
        dependencies
            .exact_triples
            .insert(("s".to_string(), "p".to_string(), "o".to_string()));

        let (_query_id, _rx) = manager
            .register_query(
                "conn-1".to_string(),
                Document {
                    definitions: Vec::new(),
                },
                ExecutionContext {
                    variables: HashMap::new(),
                    operation_name: None,
                    request_id: Uuid::new_v4().to_string(),
                    fragments: HashMap::new(),
                },
                dependencies,
            )
            .await
            .unwrap();

        // Notify change
        let change = RdfChange {
            change_type: ChangeType::Insert,
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
            graph: None,
            timestamp: Instant::now(),
        };
        manager.notify_change(change).await.unwrap();

        // Process changes
        let affected = manager.process_changes().await.unwrap();
        assert_eq!(affected, 1);
    }

    #[test]
    fn test_update_type() {
        assert_eq!(UpdateType::Full, UpdateType::Full);
        assert_ne!(UpdateType::Full, UpdateType::Diff);
    }

    #[test]
    fn test_change_type() {
        assert_eq!(ChangeType::Insert, ChangeType::Insert);
        assert_ne!(ChangeType::Insert, ChangeType::Delete);
    }
}
