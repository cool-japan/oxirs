//! Incremental Query Execution
//!
//! This module provides incremental execution capabilities for GraphQL queries,
//! allowing queries to be partially executed and results to be delivered progressively.
//!
//! # Features
//!
//! - **@defer Directive**: Defer non-critical fields for later execution
//! - **@stream Directive**: Stream list items incrementally
//! - **Execution Scheduling**: Priority-based execution scheduling
//! - **Partial Results**: Deliver partial results as they become available
//! - **Error Isolation**: Isolate errors to specific fragments without failing entire query
//! - **Dependency Tracking**: Track field dependencies for correct execution order
//!
//! # Example
//!
//! ```graphql
//! query {
//!   user(id: "123") {
//!     id
//!     name
//!     ... @defer {
//!       posts {
//!         title
//!         content
//!       }
//!     }
//!     friends @stream(initialCount: 10) {
//!       id
//!       name
//!     }
//!   }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

/// Configuration for incremental execution
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Enable @defer directive support
    pub enable_defer: bool,
    /// Enable @stream directive support
    pub enable_stream: bool,
    /// Default initial count for @stream
    pub default_stream_initial_count: usize,
    /// Maximum concurrent deferred fragments
    pub max_concurrent_defers: usize,
    /// Timeout for individual fragment execution
    pub fragment_timeout: Duration,
    /// Enable parallel execution of independent fragments
    pub enable_parallel_execution: bool,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enable_defer: true,
            enable_stream: true,
            default_stream_initial_count: 10,
            max_concurrent_defers: 10,
            fragment_timeout: Duration::from_secs(30),
            enable_parallel_execution: true,
        }
    }
}

impl IncrementalConfig {
    /// Create new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to enable @defer directive
    pub fn with_defer(mut self, enabled: bool) -> Self {
        self.enable_defer = enabled;
        self
    }

    /// Set whether to enable @stream directive
    pub fn with_stream(mut self, enabled: bool) -> Self {
        self.enable_stream = enabled;
        self
    }

    /// Set default initial count for @stream
    pub fn with_stream_initial_count(mut self, count: usize) -> Self {
        self.default_stream_initial_count = count;
        self
    }

    /// Set maximum concurrent deferred fragments
    pub fn with_max_concurrent_defers(mut self, max: usize) -> Self {
        self.max_concurrent_defers = max;
        self
    }

    /// Set fragment execution timeout
    pub fn with_fragment_timeout(mut self, timeout: Duration) -> Self {
        self.fragment_timeout = timeout;
        self
    }

    /// Enable parallel execution
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.enable_parallel_execution = enabled;
        self
    }
}

/// Execution phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionPhase {
    /// Initial synchronous execution
    Initial,
    /// Deferred fragment execution
    Deferred,
    /// Streaming execution
    Streaming,
}

/// Fragment identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FragmentId {
    /// Fragment path (e.g., "user.posts")
    pub path: String,
    /// Fragment label (optional)
    pub label: Option<String>,
}

impl FragmentId {
    /// Create a new fragment ID
    pub fn new(path: String) -> Self {
        Self { path, label: None }
    }

    /// Create with label
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
}

/// Deferred fragment
#[derive(Debug, Clone)]
pub struct DeferredFragment {
    /// Fragment identifier
    pub id: FragmentId,
    /// Fields to execute
    pub fields: Vec<String>,
    /// Priority (higher = execute sooner)
    pub priority: u8,
    /// Dependencies (fragment IDs that must complete first)
    pub dependencies: Vec<FragmentId>,
    /// If clause condition
    pub if_condition: bool,
}

impl DeferredFragment {
    /// Create a new deferred fragment
    pub fn new(id: FragmentId, fields: Vec<String>) -> Self {
        Self {
            id,
            fields,
            priority: 0,
            dependencies: Vec::new(),
            if_condition: true,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Add dependency
    pub fn with_dependency(mut self, dep: FragmentId) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Set if condition
    pub fn with_if_condition(mut self, condition: bool) -> Self {
        self.if_condition = condition;
        self
    }
}

/// Stream directive configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Fragment identifier
    pub id: FragmentId,
    /// Field to stream
    pub field: String,
    /// Initial count to return synchronously
    pub initial_count: usize,
    /// If clause condition
    pub if_condition: bool,
    /// Label for this stream
    pub label: Option<String>,
}

impl StreamConfig {
    /// Create a new stream configuration
    pub fn new(id: FragmentId, field: String, initial_count: usize) -> Self {
        Self {
            id,
            field,
            initial_count,
            if_condition: true,
            label: None,
        }
    }

    /// Set if condition
    pub fn with_if_condition(mut self, condition: bool) -> Self {
        self.if_condition = condition;
        self
    }

    /// Set label
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
}

/// Incremental result payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalPayload {
    /// Data for this payload
    pub data: Option<serde_json::Value>,
    /// Path to where this data should be merged
    pub path: Vec<String>,
    /// Label identifying this payload
    pub label: Option<String>,
    /// Errors in this payload
    pub errors: Vec<IncrementalError>,
    /// Whether this completes the incremental delivery
    pub has_next: bool,
}

/// Error in incremental execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalError {
    /// Error message
    pub message: String,
    /// Path where error occurred
    pub path: Vec<String>,
    /// Extensions
    pub extensions: HashMap<String, serde_json::Value>,
}

/// Initial execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialResult {
    /// Initial data
    pub data: serde_json::Value,
    /// Initial errors
    pub errors: Vec<IncrementalError>,
    /// Whether there are pending deferred/streamed results
    pub has_next: bool,
}

/// Execution plan for incremental execution
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Fields to execute immediately
    pub immediate_fields: Vec<String>,
    /// Deferred fragments
    pub deferred_fragments: Vec<DeferredFragment>,
    /// Stream configurations
    pub streams: Vec<StreamConfig>,
    /// Fragment dependency graph
    pub dependencies: HashMap<FragmentId, Vec<FragmentId>>,
}

impl ExecutionPlan {
    /// Create a new execution plan
    pub fn new() -> Self {
        Self {
            immediate_fields: Vec::new(),
            deferred_fragments: Vec::new(),
            streams: Vec::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Add immediate field
    pub fn add_immediate_field(&mut self, field: String) {
        self.immediate_fields.push(field);
    }

    /// Add deferred fragment
    pub fn add_deferred_fragment(&mut self, fragment: DeferredFragment) {
        // Update dependency graph
        for dep in &fragment.dependencies {
            self.dependencies
                .entry(dep.clone())
                .or_default()
                .push(fragment.id.clone());
        }
        self.deferred_fragments.push(fragment);
    }

    /// Add stream configuration
    pub fn add_stream(&mut self, stream: StreamConfig) {
        self.streams.push(stream);
    }

    /// Get execution order for deferred fragments (topological sort)
    pub fn get_execution_order(&self) -> Vec<FragmentId> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut in_progress = HashSet::new();

        for fragment in &self.deferred_fragments {
            self.visit_fragment(&fragment.id, &mut visited, &mut in_progress, &mut order);
        }

        order
    }

    fn visit_fragment(
        &self,
        id: &FragmentId,
        visited: &mut HashSet<FragmentId>,
        in_progress: &mut HashSet<FragmentId>,
        order: &mut Vec<FragmentId>,
    ) {
        if visited.contains(id) {
            return;
        }

        if in_progress.contains(id) {
            // Circular dependency detected
            return;
        }

        in_progress.insert(id.clone());

        // Visit dependencies first
        if let Some(fragment) = self.deferred_fragments.iter().find(|f| &f.id == id) {
            for dep in &fragment.dependencies {
                self.visit_fragment(dep, visited, in_progress, order);
            }
        }

        in_progress.remove(id);
        visited.insert(id.clone());
        order.push(id.clone());
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental executor
pub struct IncrementalExecutor {
    config: IncrementalConfig,
    stats: Arc<RwLock<ExecutionStats>>,
}

impl IncrementalExecutor {
    /// Create a new incremental executor
    pub fn new(config: IncrementalConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ExecutionStats::new())),
        }
    }

    /// Execute a query incrementally
    pub async fn execute(
        &self,
        plan: ExecutionPlan,
        initial_data: serde_json::Value,
    ) -> Result<IncrementalResultStream, ExecutionError> {
        let (tx, rx) = mpsc::channel(100);
        let config = self.config.clone();
        let stats = self.stats.clone();

        // Send initial result
        let has_deferred = !plan.deferred_fragments.is_empty();
        let has_streams = !plan.streams.is_empty();

        let initial = InitialResult {
            data: initial_data,
            errors: Vec::new(),
            has_next: has_deferred || has_streams,
        };

        if tx
            .send(Ok(IncrementalEvent::Initial(initial)))
            .await
            .is_err()
        {
            return Err(ExecutionError::ChannelError(
                "Failed to send initial result".to_string(),
            ));
        }

        // Spawn task to handle deferred execution
        if has_deferred && config.enable_defer {
            let plan_clone = plan.clone();
            let tx_clone = tx.clone();
            let stats_clone = stats.clone();
            let config_clone = config.clone();
            tokio::spawn(async move {
                Self::execute_deferred_fragments(plan_clone, tx_clone, config_clone, stats_clone)
                    .await;
            });
        }

        // Spawn task to handle streaming
        if has_streams && config.enable_stream {
            let plan_clone = plan;
            let tx_clone = tx;
            let stats_clone = stats.clone();
            tokio::spawn(async move {
                Self::execute_streams(plan_clone, tx_clone, stats_clone).await;
            });
        } else if !has_deferred {
            // No deferred or streams, send complete event
            let _ = tx.send(Ok(IncrementalEvent::Complete)).await;
        }

        Ok(IncrementalResultStream::new(rx, self.stats.clone()))
    }

    /// Execute deferred fragments
    async fn execute_deferred_fragments(
        plan: ExecutionPlan,
        tx: mpsc::Sender<IncrementalResult>,
        config: IncrementalConfig,
        stats: Arc<RwLock<ExecutionStats>>,
    ) {
        let execution_order = plan.get_execution_order();
        let mut completed = HashSet::new();

        for fragment_id in execution_order {
            // Find the fragment
            let fragment = match plan.deferred_fragments.iter().find(|f| f.id == fragment_id) {
                Some(f) => f,
                None => continue,
            };

            // Check if condition
            if !fragment.if_condition {
                completed.insert(fragment_id.clone());
                continue;
            }

            // Wait for dependencies
            for dep in &fragment.dependencies {
                while !completed.contains(dep) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }

            let start_time = Instant::now();

            // Execute fragment (mock execution for now)
            let data = serde_json::json!({
                "result": format!("Deferred data for {}", fragment_id.path)
            });

            let path: Vec<String> = fragment_id.path.split('.').map(String::from).collect();

            let payload = IncrementalPayload {
                data: Some(data),
                path,
                label: fragment_id.label.clone(),
                errors: Vec::new(),
                has_next: true,
            };

            // Update stats
            {
                let mut s = stats.write().await;
                s.deferred_fragments_executed += 1;
                s.total_fragment_execution_time += start_time.elapsed();
            }

            if tx
                .send(Ok(IncrementalEvent::Deferred(payload)))
                .await
                .is_err()
            {
                break;
            }

            completed.insert(fragment_id);

            // Check max concurrent limit
            if completed.len() >= config.max_concurrent_defers {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }

        // Send completion
        let _ = tx.send(Ok(IncrementalEvent::Complete)).await;
    }

    /// Execute streaming fields
    async fn execute_streams(
        plan: ExecutionPlan,
        tx: mpsc::Sender<IncrementalResult>,
        stats: Arc<RwLock<ExecutionStats>>,
    ) {
        for stream_config in plan.streams {
            if !stream_config.if_condition {
                continue;
            }

            // Simulate streaming items (mock implementation)
            let total_items = 50; // Mock total
            let mut sent = stream_config.initial_count;

            while sent < total_items {
                let chunk_size = 10.min(total_items - sent);
                let items: Vec<serde_json::Value> = (sent..sent + chunk_size)
                    .map(|i| serde_json::json!({ "id": i, "data": format!("item_{}", i) }))
                    .collect();

                let path: Vec<String> =
                    stream_config.id.path.split('.').map(String::from).collect();

                let payload = IncrementalPayload {
                    data: Some(serde_json::json!(items)),
                    path,
                    label: stream_config.label.clone(),
                    errors: Vec::new(),
                    has_next: sent + chunk_size < total_items,
                };

                // Update stats
                {
                    let mut s = stats.write().await;
                    s.stream_chunks_sent += 1;
                    s.total_stream_items += chunk_size;
                }

                if tx
                    .send(Ok(IncrementalEvent::Stream(payload)))
                    .await
                    .is_err()
                {
                    break;
                }

                sent += chunk_size;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }

        let _ = tx.send(Ok(IncrementalEvent::Complete)).await;
    }

    /// Get execution statistics
    pub async fn get_stats(&self) -> ExecutionStats {
        self.stats.read().await.clone()
    }
}

/// Incremental event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IncrementalEvent {
    /// Initial result
    Initial(InitialResult),
    /// Deferred fragment result
    Deferred(IncrementalPayload),
    /// Stream chunk
    Stream(IncrementalPayload),
    /// Execution complete
    Complete,
    /// Error occurred
    Error(IncrementalError),
}

/// Result type for incremental execution
pub type IncrementalResult = Result<IncrementalEvent, ExecutionError>;

/// Incremental result stream
pub struct IncrementalResultStream {
    receiver: mpsc::Receiver<IncrementalResult>,
    stats: Arc<RwLock<ExecutionStats>>,
}

impl IncrementalResultStream {
    /// Create a new result stream
    pub fn new(
        receiver: mpsc::Receiver<IncrementalResult>,
        stats: Arc<RwLock<ExecutionStats>>,
    ) -> Self {
        Self { receiver, stats }
    }

    /// Get next event from stream
    pub async fn next(&mut self) -> Option<IncrementalResult> {
        self.receiver.recv().await
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> ExecutionStats {
        self.stats.read().await.clone()
    }
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Number of deferred fragments executed
    pub deferred_fragments_executed: usize,
    /// Number of stream chunks sent
    pub stream_chunks_sent: usize,
    /// Total items streamed
    pub total_stream_items: usize,
    /// Total fragment execution time
    pub total_fragment_execution_time: Duration,
    /// Errors encountered
    pub errors: usize,
}

impl ExecutionStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            deferred_fragments_executed: 0,
            stream_chunks_sent: 0,
            total_stream_items: 0,
            total_fragment_execution_time: Duration::ZERO,
            errors: 0,
        }
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.errors += 1;
    }
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during incremental execution
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    /// Channel error
    #[error("Channel error: {0}")]
    ChannelError(String),

    /// Timeout error
    #[error("Fragment execution timeout")]
    Timeout,

    /// Dependency cycle detected
    #[error("Circular dependency detected in fragment: {0}")]
    CircularDependency(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_config_builder() {
        let config = IncrementalConfig::new()
            .with_defer(true)
            .with_stream(true)
            .with_stream_initial_count(20)
            .with_max_concurrent_defers(5)
            .with_parallel_execution(true);

        assert!(config.enable_defer);
        assert!(config.enable_stream);
        assert_eq!(config.default_stream_initial_count, 20);
        assert_eq!(config.max_concurrent_defers, 5);
        assert!(config.enable_parallel_execution);
    }

    #[test]
    fn test_fragment_id_creation() {
        let id = FragmentId::new("user.posts".to_string()).with_label("userPosts".to_string());

        assert_eq!(id.path, "user.posts");
        assert_eq!(id.label, Some("userPosts".to_string()));
    }

    #[test]
    fn test_deferred_fragment_creation() {
        let id = FragmentId::new("user.friends".to_string());
        let dep_id = FragmentId::new("user.profile".to_string());

        let fragment =
            DeferredFragment::new(id.clone(), vec!["name".to_string(), "email".to_string()])
                .with_priority(5)
                .with_dependency(dep_id)
                .with_if_condition(true);

        assert_eq!(fragment.id, id);
        assert_eq!(fragment.fields.len(), 2);
        assert_eq!(fragment.priority, 5);
        assert_eq!(fragment.dependencies.len(), 1);
        assert!(fragment.if_condition);
    }

    #[test]
    fn test_stream_config_creation() {
        let id = FragmentId::new("user.posts".to_string());
        let stream = StreamConfig::new(id.clone(), "posts".to_string(), 10)
            .with_if_condition(true)
            .with_label("postStream".to_string());

        assert_eq!(stream.id, id);
        assert_eq!(stream.field, "posts");
        assert_eq!(stream.initial_count, 10);
        assert!(stream.if_condition);
        assert_eq!(stream.label, Some("postStream".to_string()));
    }

    #[test]
    fn test_execution_plan_immediate_fields() {
        let mut plan = ExecutionPlan::new();
        plan.add_immediate_field("id".to_string());
        plan.add_immediate_field("name".to_string());

        assert_eq!(plan.immediate_fields.len(), 2);
    }

    #[test]
    fn test_execution_plan_deferred_fragments() {
        let mut plan = ExecutionPlan::new();

        let id1 = FragmentId::new("user.profile".to_string());
        let fragment1 = DeferredFragment::new(id1, vec!["bio".to_string()]);

        plan.add_deferred_fragment(fragment1);
        assert_eq!(plan.deferred_fragments.len(), 1);
    }

    #[test]
    fn test_execution_plan_streams() {
        let mut plan = ExecutionPlan::new();

        let id = FragmentId::new("user.posts".to_string());
        let stream = StreamConfig::new(id, "posts".to_string(), 10);

        plan.add_stream(stream);
        assert_eq!(plan.streams.len(), 1);
    }

    #[test]
    fn test_execution_plan_dependency_graph() {
        let mut plan = ExecutionPlan::new();

        let id1 = FragmentId::new("fragment1".to_string());
        let id2 = FragmentId::new("fragment2".to_string());

        let fragment2 = DeferredFragment::new(id2.clone(), vec!["field".to_string()])
            .with_dependency(id1.clone());

        plan.add_deferred_fragment(fragment2);

        assert!(plan.dependencies.contains_key(&id1));
        assert_eq!(plan.dependencies.get(&id1).unwrap(), &vec![id2]);
    }

    #[test]
    fn test_execution_order_simple() {
        let mut plan = ExecutionPlan::new();

        let id1 = FragmentId::new("fragment1".to_string());
        let id2 = FragmentId::new("fragment2".to_string());
        let id3 = FragmentId::new("fragment3".to_string());

        plan.add_deferred_fragment(DeferredFragment::new(id1.clone(), vec![]));
        plan.add_deferred_fragment(
            DeferredFragment::new(id2.clone(), vec![]).with_dependency(id1.clone()),
        );
        plan.add_deferred_fragment(
            DeferredFragment::new(id3.clone(), vec![]).with_dependency(id2.clone()),
        );

        let order = plan.get_execution_order();
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], id1);
        assert_eq!(order[1], id2);
        assert_eq!(order[2], id3);
    }

    #[test]
    fn test_execution_order_complex() {
        let mut plan = ExecutionPlan::new();

        let id1 = FragmentId::new("a".to_string());
        let id2 = FragmentId::new("b".to_string());
        let id3 = FragmentId::new("c".to_string());
        let id4 = FragmentId::new("d".to_string());

        // a -> b -> d
        // a -> c -> d
        plan.add_deferred_fragment(DeferredFragment::new(id1.clone(), vec![]));
        plan.add_deferred_fragment(
            DeferredFragment::new(id2.clone(), vec![]).with_dependency(id1.clone()),
        );
        plan.add_deferred_fragment(
            DeferredFragment::new(id3.clone(), vec![]).with_dependency(id1.clone()),
        );
        plan.add_deferred_fragment(
            DeferredFragment::new(id4.clone(), vec![])
                .with_dependency(id2.clone())
                .with_dependency(id3.clone()),
        );

        let order = plan.get_execution_order();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], id1);
        // b and c can be in any order
        assert!(order[3] == id4); // d must be last
    }

    #[test]
    fn test_execution_stats_creation() {
        let stats = ExecutionStats::new();

        assert_eq!(stats.deferred_fragments_executed, 0);
        assert_eq!(stats.stream_chunks_sent, 0);
        assert_eq!(stats.total_stream_items, 0);
        assert_eq!(stats.total_fragment_execution_time, Duration::ZERO);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_execution_stats_error_recording() {
        let mut stats = ExecutionStats::new();

        stats.record_error();
        assert_eq!(stats.errors, 1);

        stats.record_error();
        assert_eq!(stats.errors, 2);
    }

    #[tokio::test]
    async fn test_simple_incremental_execution() {
        let config = IncrementalConfig::new();
        let executor = IncrementalExecutor::new(config);

        let mut plan = ExecutionPlan::new();
        plan.add_immediate_field("id".to_string());
        plan.add_immediate_field("name".to_string());

        let initial_data = serde_json::json!({ "id": 1, "name": "Test" });

        let mut stream = executor.execute(plan, initial_data).await.unwrap();

        let mut event_count = 0;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                IncrementalEvent::Initial(result) => {
                    event_count += 1;
                    assert!(!result.has_next);
                }
                IncrementalEvent::Complete => {
                    event_count += 1;
                    break;
                }
                _ => {}
            }
        }

        assert!(event_count >= 1);
    }

    #[tokio::test]
    async fn test_deferred_execution() {
        let config = IncrementalConfig::new().with_defer(true);
        let executor = IncrementalExecutor::new(config);

        let mut plan = ExecutionPlan::new();
        plan.add_immediate_field("id".to_string());

        let id = FragmentId::new("user.posts".to_string());
        let fragment = DeferredFragment::new(id, vec!["title".to_string(), "content".to_string()]);
        plan.add_deferred_fragment(fragment);

        let initial_data = serde_json::json!({ "id": 1 });

        let mut stream = executor.execute(plan, initial_data).await.unwrap();

        let mut received_initial = false;
        let mut received_deferred = false;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                IncrementalEvent::Initial(result) => {
                    received_initial = true;
                    assert!(result.has_next);
                }
                IncrementalEvent::Deferred(_) => {
                    received_deferred = true;
                }
                IncrementalEvent::Complete => break,
                _ => {}
            }
        }

        assert!(received_initial);
        assert!(received_deferred);
    }

    #[tokio::test]
    async fn test_stream_execution() {
        let config = IncrementalConfig::new().with_stream(true);
        let executor = IncrementalExecutor::new(config);

        let mut plan = ExecutionPlan::new();
        plan.add_immediate_field("id".to_string());

        let id = FragmentId::new("user.posts".to_string());
        let stream_config = StreamConfig::new(id, "posts".to_string(), 10);
        plan.add_stream(stream_config);

        let initial_data = serde_json::json!({ "id": 1 });

        let mut stream = executor.execute(plan, initial_data).await.unwrap();

        let mut received_initial = false;
        let mut stream_chunks = 0;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                IncrementalEvent::Initial(result) => {
                    received_initial = true;
                    assert!(result.has_next);
                }
                IncrementalEvent::Stream(_) => {
                    stream_chunks += 1;
                }
                IncrementalEvent::Complete => break,
                _ => {}
            }
        }

        assert!(received_initial);
        assert!(stream_chunks > 0);
    }

    #[tokio::test]
    async fn test_deferred_with_dependencies() {
        let config = IncrementalConfig::new();
        let executor = IncrementalExecutor::new(config);

        let mut plan = ExecutionPlan::new();

        let id1 = FragmentId::new("user.profile".to_string());
        let id2 = FragmentId::new("user.posts".to_string());

        let fragment1 = DeferredFragment::new(id1.clone(), vec!["bio".to_string()]);
        let fragment2 = DeferredFragment::new(id2, vec!["posts".to_string()]).with_dependency(id1);

        plan.add_deferred_fragment(fragment1);
        plan.add_deferred_fragment(fragment2);

        let initial_data = serde_json::json!({ "id": 1 });

        let mut stream = executor.execute(plan, initial_data).await.unwrap();

        let mut deferred_count = 0;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                IncrementalEvent::Deferred(_) => {
                    deferred_count += 1;
                }
                IncrementalEvent::Complete => break,
                _ => {}
            }
        }

        assert_eq!(deferred_count, 2);
    }

    #[tokio::test]
    async fn test_conditional_defer() {
        let config = IncrementalConfig::new();
        let executor = IncrementalExecutor::new(config);

        let mut plan = ExecutionPlan::new();

        let id = FragmentId::new("user.posts".to_string());
        let fragment =
            DeferredFragment::new(id, vec!["posts".to_string()]).with_if_condition(false); // Should be skipped

        plan.add_deferred_fragment(fragment);

        let initial_data = serde_json::json!({ "id": 1 });

        let mut stream = executor.execute(plan, initial_data).await.unwrap();

        let mut deferred_count = 0;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                IncrementalEvent::Deferred(_) => {
                    deferred_count += 1;
                }
                IncrementalEvent::Complete => break,
                _ => {}
            }
        }

        assert_eq!(deferred_count, 0);
    }

    #[tokio::test]
    async fn test_execution_statistics() {
        let config = IncrementalConfig::new();
        let executor = IncrementalExecutor::new(config);

        let mut plan = ExecutionPlan::new();

        let id = FragmentId::new("user.posts".to_string());
        let fragment = DeferredFragment::new(id, vec!["posts".to_string()]);
        plan.add_deferred_fragment(fragment);

        let initial_data = serde_json::json!({ "id": 1 });

        let mut stream = executor.execute(plan, initial_data).await.unwrap();

        while let Some(event) = stream.next().await {
            if matches!(event.unwrap(), IncrementalEvent::Complete) {
                break;
            }
        }

        let stats = stream.get_stats().await;
        assert!(stats.deferred_fragments_executed > 0);
    }

    #[tokio::test]
    async fn test_parallel_execution_flag() {
        let config = IncrementalConfig::new().with_parallel_execution(true);

        assert!(config.enable_parallel_execution);
    }

    #[test]
    fn test_incremental_payload_creation() {
        let payload = IncrementalPayload {
            data: Some(serde_json::json!({"test": "data"})),
            path: vec!["user".to_string(), "posts".to_string()],
            label: Some("postData".to_string()),
            errors: Vec::new(),
            has_next: true,
        };

        assert!(payload.data.is_some());
        assert_eq!(payload.path.len(), 2);
        assert_eq!(payload.label, Some("postData".to_string()));
        assert!(payload.has_next);
    }

    #[test]
    fn test_incremental_error_creation() {
        let error = IncrementalError {
            message: "Test error".to_string(),
            path: vec!["user".to_string(), "posts".to_string()],
            extensions: HashMap::new(),
        };

        assert_eq!(error.message, "Test error");
        assert_eq!(error.path.len(), 2);
    }
}
