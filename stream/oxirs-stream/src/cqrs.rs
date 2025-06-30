//! # CQRS (Command Query Responsibility Segregation) Implementation
//!
//! This module provides a complete CQRS pattern implementation for OxiRS Stream,
//! separating command (write) and query (read) responsibilities. It integrates
//! with the event sourcing framework for eventual consistency and scalability.

use crate::event_sourcing::{EventStoreTrait, EventStream};
use crate::{EventMetadata, StreamEvent};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// CQRS configuration for the streaming system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CQRSConfig {
    /// Maximum command processing time before timeout
    pub command_timeout_ms: u64,
    /// Maximum query processing time before timeout
    pub query_timeout_ms: u64,
    /// Enable command validation
    pub enable_command_validation: bool,
    /// Enable query optimization
    pub enable_query_optimization: bool,
    /// Command retry configuration
    pub command_retry_config: RetryConfig,
    /// Query cache configuration
    pub query_cache_config: QueryCacheConfig,
    /// Eventual consistency window (max time for read models to catch up)
    pub consistency_window_ms: u64,
    /// Maximum concurrent commands
    pub max_concurrent_commands: usize,
    /// Maximum concurrent queries
    pub max_concurrent_queries: usize,
}

impl Default for CQRSConfig {
    fn default() -> Self {
        Self {
            command_timeout_ms: 30000,
            query_timeout_ms: 10000,
            enable_command_validation: true,
            enable_query_optimization: true,
            command_retry_config: RetryConfig::default(),
            query_cache_config: QueryCacheConfig::default(),
            consistency_window_ms: 5000,
            max_concurrent_commands: 1000,
            max_concurrent_queries: 10000,
        }
    }
}

/// Retry configuration for commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Query cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub ttl_seconds: u64,
    pub max_memory_mb: usize,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            ttl_seconds: 300,
            max_memory_mb: 512,
        }
    }
}

/// Base trait for all commands in the CQRS system
pub trait Command: Send + Sync + Clone + std::fmt::Debug {
    type AggregateId: Clone + std::fmt::Debug + Send + Sync;
    type EventType: Send + Sync + Clone;

    /// Get the unique identifier for this command
    fn command_id(&self) -> Uuid;

    /// Get the aggregate ID this command operates on
    fn aggregate_id(&self) -> Self::AggregateId;

    /// Validate the command before execution
    fn validate(&self) -> Result<()>;

    /// Get the expected version for optimistic concurrency control
    fn expected_version(&self) -> Option<u64>;
}

/// Base trait for all queries in the CQRS system
pub trait Query: Send + Sync + Clone + std::fmt::Debug {
    type Result: Send + Sync + Clone;

    /// Get the unique identifier for this query
    fn query_id(&self) -> Uuid;

    /// Validate the query before execution
    fn validate(&self) -> Result<()>;

    /// Get cache key for this query (if cacheable)
    fn cache_key(&self) -> Option<String>;

    /// Get query timeout in milliseconds
    fn timeout_ms(&self) -> Option<u64>;
}

/// Command handler trait for processing commands
#[async_trait::async_trait]
pub trait CommandHandler<C: Command>: Send + Sync {
    /// Handle the command and return events to be persisted
    async fn handle(&self, command: C) -> Result<Vec<StreamEvent>>;

    /// Validate the command (optional additional validation)
    async fn validate_command(&self, command: &C) -> Result<()> {
        command.validate()
    }
}

/// Query handler trait for processing queries
#[async_trait::async_trait]
pub trait QueryHandler<Q: Query>: Send + Sync {
    /// Handle the query and return the result
    async fn handle(&self, query: Q) -> Result<Q::Result>;

    /// Validate the query (optional additional validation)
    async fn validate_query(&self, query: &Q) -> Result<()> {
        query.validate()
    }
}

/// Read model projection trait for updating read models from events
#[async_trait::async_trait]
pub trait ReadModelProjection: Send + Sync {
    type Event: Send + Sync;

    /// Handle an event and update the read model
    async fn handle_event(&self, event: &Self::Event) -> Result<()>;

    /// Get the projection name for tracking progress
    fn projection_name(&self) -> &str;

    /// Reset the projection (for rebuilding)
    async fn reset(&self) -> Result<()>;
}

/// Command result containing execution metadata
#[derive(Debug, Clone)]
pub struct CommandResult {
    pub command_id: Uuid,
    pub aggregate_id: String,
    pub events_count: usize,
    pub execution_time: Duration,
    pub version: u64,
    pub timestamp: DateTime<Utc>,
}

/// Query result containing execution metadata
#[derive(Debug, Clone)]
pub struct QueryResult<T> {
    pub query_id: Uuid,
    pub result: T,
    pub execution_time: Duration,
    pub cache_hit: bool,
    pub timestamp: DateTime<Utc>,
}

/// CQRS command bus for handling commands
pub struct CommandBus {
    config: CQRSConfig,
    event_store: Arc<dyn EventStoreTrait>,
    handlers: Arc<RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>>,
    command_semaphore: Arc<tokio::sync::Semaphore>,
    metrics: Arc<RwLock<CommandBusMetrics>>,
    event_publisher: broadcast::Sender<StreamEvent>,
}

impl CommandBus {
    /// Create a new command bus
    pub fn new(config: CQRSConfig, event_store: Arc<dyn EventStoreTrait>) -> Self {
        let (event_publisher, _) = broadcast::channel(10000);

        Self {
            command_semaphore: Arc::new(tokio::sync::Semaphore::new(
                config.max_concurrent_commands,
            )),
            config,
            event_store,
            handlers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(CommandBusMetrics::default())),
            event_publisher,
        }
    }

    /// Register a command handler
    pub async fn register_handler<C, H>(&self, handler: H)
    where
        C: Command + 'static,
        H: CommandHandler<C> + 'static,
    {
        let type_name = std::any::type_name::<C>();
        let mut handlers = self.handlers.write().await;
        handlers.insert(type_name.to_string(), Box::new(handler));
        info!("Registered command handler for {}", type_name);
    }

    /// Execute a command
    pub async fn execute<C>(&self, command: C) -> Result<CommandResult>
    where
        C: Command + 'static,
    {
        let start_time = Instant::now();
        let command_id = command.command_id();

        debug!(
            "Executing command {} for aggregate {:?}",
            command_id,
            command.aggregate_id()
        );

        // Acquire semaphore for concurrency control
        let _permit = self.command_semaphore.acquire().await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.commands_received += 1;
            metrics.active_commands += 1;
        }

        let result = self.execute_with_retry(command).await;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.active_commands -= 1;
            match &result {
                Ok(_) => metrics.commands_succeeded += 1,
                Err(_) => metrics.commands_failed += 1,
            }
        }

        let execution_time = start_time.elapsed();
        debug!("Command {} executed in {:?}", command_id, execution_time);

        result
    }

    /// Execute command with retry logic
    async fn execute_with_retry<C>(&self, command: C) -> Result<CommandResult>
    where
        C: Command + 'static,
    {
        let mut attempt = 0;
        let mut delay = Duration::from_millis(self.config.command_retry_config.initial_delay_ms);

        loop {
            match self.execute_once(command.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) if attempt >= self.config.command_retry_config.max_retries => {
                    error!(
                        "Command {} failed after {} attempts: {}",
                        command.command_id(),
                        attempt + 1,
                        e
                    );
                    return Err(e);
                }
                Err(e) => {
                    warn!(
                        "Command {} failed on attempt {}: {}",
                        command.command_id(),
                        attempt + 1,
                        e
                    );
                    attempt += 1;

                    tokio::time::sleep(delay).await;
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64
                            * self.config.command_retry_config.backoff_multiplier)
                            as u64,
                    )
                    .min(Duration::from_millis(
                        self.config.command_retry_config.max_delay_ms,
                    ));
                }
            }
        }
    }

    /// Execute command once
    async fn execute_once<C>(&self, command: C) -> Result<CommandResult>
    where
        C: Command + 'static,
    {
        let start_time = Instant::now();
        let command_id = command.command_id();
        let aggregate_id = format!("{:?}", command.aggregate_id());

        // Validate command
        if self.config.enable_command_validation {
            command.validate()?;
        }

        // Get handler
        let type_name = std::any::type_name::<C>();
        let handlers = self.handlers.read().await;
        let handler = handlers
            .get(type_name)
            .ok_or_else(|| anyhow!("No handler registered for command type {}", type_name))?;

        // Downcast handler
        let handler = handler
            .downcast_ref::<Box<dyn CommandHandler<C>>>()
            .ok_or_else(|| anyhow!("Handler type mismatch for command {}", type_name))?;

        // Execute handler
        let events = handler
            .handle(command.clone())
            .await
            .map_err(|e| anyhow!("Command handler error: {}", e))?;

        // Store events
        let version = if events.is_empty() {
            0
        } else {
            self.event_store
                .append_events(&aggregate_id, &events, command.expected_version())
                .await?
        };

        // Publish events
        for event in &events {
            let _ = self.event_publisher.send(event.clone());
        }

        Ok(CommandResult {
            command_id,
            aggregate_id,
            events_count: events.len(),
            execution_time: start_time.elapsed(),
            version,
            timestamp: Utc::now(),
        })
    }

    /// Get command bus metrics
    pub async fn get_metrics(&self) -> CommandBusMetrics {
        self.metrics.read().await.clone()
    }

    /// Subscribe to events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<StreamEvent> {
        self.event_publisher.subscribe()
    }
}

/// CQRS query bus for handling queries
#[derive(Debug)]
pub struct QueryBus {
    config: CQRSConfig,
    handlers: Arc<RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>>,
    query_semaphore: Arc<tokio::sync::Semaphore>,
    cache: Arc<RwLock<QueryCache>>,
    metrics: Arc<RwLock<QueryBusMetrics>>,
}

impl QueryBus {
    /// Create a new query bus
    pub fn new(config: CQRSConfig) -> Self {
        Self {
            query_semaphore: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_queries)),
            cache: Arc::new(RwLock::new(QueryCache::new(
                config.query_cache_config.clone(),
            ))),
            config,
            handlers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(QueryBusMetrics::default())),
        }
    }

    /// Register a query handler
    pub async fn register_handler<Q, H>(&self, handler: H)
    where
        Q: Query + 'static,
        H: QueryHandler<Q> + 'static,
    {
        let type_name = std::any::type_name::<Q>();
        let mut handlers = self.handlers.write().await;
        handlers.insert(type_name.to_string(), Box::new(handler));
        info!("Registered query handler for {}", type_name);
    }

    /// Execute a query
    pub async fn execute<Q>(&self, query: Q) -> Result<QueryResult<Q::Result>>
    where
        Q: Query + 'static,
        Q::Result: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        let start_time = Instant::now();
        let query_id = query.query_id();

        debug!("Executing query {}", query_id);

        // Acquire semaphore for concurrency control
        let _permit = self.query_semaphore.acquire().await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.queries_received += 1;
            metrics.active_queries += 1;
        }

        // Check cache first
        let cache_hit = if self.config.query_cache_config.enabled {
            if let Some(cache_key) = query.cache_key() {
                let cache = self.cache.read().await;
                if let Some(cached_result) = cache.get::<Q::Result>(&cache_key) {
                    let mut metrics = self.metrics.write().await;
                    metrics.active_queries -= 1;
                    metrics.queries_succeeded += 1;
                    metrics.cache_hits += 1;

                    return Ok(QueryResult {
                        query_id,
                        result: cached_result,
                        execution_time: start_time.elapsed(),
                        cache_hit: true,
                        timestamp: Utc::now(),
                    });
                }
            }
            false
        } else {
            false
        };

        let result = self.execute_query_handler(query).await;

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.active_queries -= 1;
            match &result {
                Ok(_) => {
                    metrics.queries_succeeded += 1;
                    if !cache_hit {
                        metrics.cache_misses += 1;
                    }
                }
                Err(_) => metrics.queries_failed += 1,
            }
        }

        let execution_time = start_time.elapsed();
        debug!("Query {} executed in {:?}", query_id, execution_time);

        result.map(|r| QueryResult {
            query_id,
            result: r,
            execution_time,
            cache_hit,
            timestamp: Utc::now(),
        })
    }

    /// Execute query handler
    async fn execute_query_handler<Q>(&self, query: Q) -> Result<Q::Result>
    where
        Q: Query + 'static,
        Q::Result: Clone + Serialize + for<'de> Deserialize<'de>,
    {
        // Validate query
        query.validate()?;

        // Get handler
        let type_name = std::any::type_name::<Q>();
        let handlers = self.handlers.read().await;
        let handler = handlers
            .get(type_name)
            .ok_or_else(|| anyhow!("No handler registered for query type {}", type_name))?;

        // Downcast handler
        let handler = handler
            .downcast_ref::<Box<dyn QueryHandler<Q>>>()
            .ok_or_else(|| anyhow!("Handler type mismatch for query {}", type_name))?;

        // Execute handler with timeout
        let timeout =
            Duration::from_millis(query.timeout_ms().unwrap_or(self.config.query_timeout_ms));

        let result = tokio::time::timeout(timeout, handler.handle(query.clone()))
            .await
            .map_err(|_| anyhow!("Query timeout"))?
            .map_err(|e| anyhow!("Query handler error: {}", e))?;

        // Cache result if applicable
        if self.config.query_cache_config.enabled {
            if let Some(cache_key) = query.cache_key() {
                let mut cache = self.cache.write().await;
                cache.set(cache_key, result.clone());
            }
        }

        Ok(result)
    }

    /// Get query bus metrics
    pub async fn get_metrics(&self) -> QueryBusMetrics {
        self.metrics.read().await.clone()
    }

    /// Clear query cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

/// Read model manager for handling projections
pub struct ReadModelManager {
    projections: Arc<RwLock<HashMap<String, Box<dyn ReadModelProjection<Event = StreamEvent>>>>>,
    projection_positions: Arc<RwLock<HashMap<String, u64>>>,
    event_stream: Arc<dyn EventStream>,
    metrics: Arc<RwLock<ReadModelMetrics>>,
}

impl ReadModelManager {
    /// Create a new read model manager
    pub fn new(event_stream: Arc<dyn EventStream>) -> Self {
        Self {
            projections: Arc::new(RwLock::new(HashMap::new())),
            projection_positions: Arc::new(RwLock::new(HashMap::new())),
            event_stream,
            metrics: Arc::new(RwLock::new(ReadModelMetrics::default())),
        }
    }

    /// Register a read model projection
    pub async fn register_projection<P>(&self, projection: P)
    where
        P: ReadModelProjection<Event = StreamEvent> + 'static,
    {
        let name = projection.projection_name().to_string();
        let mut projections = self.projections.write().await;
        projections.insert(name.clone(), Box::new(projection));

        let mut positions = self.projection_positions.write().await;
        positions.entry(name.clone()).or_insert(0);

        info!("Registered read model projection: {}", name);
    }

    /// Process events for all projections
    pub async fn process_events(&self) -> Result<()> {
        let projections = self.projections.read().await;

        for (name, projection) in projections.iter() {
            if let Err(e) = self.process_projection(name, projection.as_ref()).await {
                error!("Error processing projection {}: {}", name, e);

                let mut metrics = self.metrics.write().await;
                *metrics.projection_errors.entry(name.clone()).or_insert(0) += 1;
            }
        }

        Ok(())
    }

    /// Process events for a specific projection
    async fn process_projection(
        &self,
        name: &str,
        projection: &dyn ReadModelProjection<Event = StreamEvent>,
    ) -> Result<()> {
        let position = {
            let positions = self.projection_positions.read().await;
            positions.get(name).copied().unwrap_or(0)
        };

        let events = self
            .event_stream
            .read_events_from_position(position, 1000)
            .await?;

        for stored_event in events {
            if let Err(e) = projection.handle_event(&stored_event.event_data).await {
                error!("Projection {} failed to handle event: {}", name, e);
                return Err(anyhow!("Projection error: {}", e));
            }

            // Update position
            let mut positions = self.projection_positions.write().await;
            *positions.entry(name.to_string()).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Reset a projection
    pub async fn reset_projection(&self, name: &str) -> Result<()> {
        let projections = self.projections.read().await;
        let projection = projections
            .get(name)
            .ok_or_else(|| anyhow!("Projection not found: {}", name))?;

        projection
            .reset()
            .await
            .map_err(|e| anyhow!("Failed to reset projection {}: {}", name, e))?;

        let mut positions = self.projection_positions.write().await;
        positions.insert(name.to_string(), 0);

        info!("Reset projection: {}", name);
        Ok(())
    }

    /// Get read model metrics
    pub async fn get_metrics(&self) -> ReadModelMetrics {
        self.metrics.read().await.clone()
    }
}

/// Query cache implementation
#[derive(Debug)]
struct QueryCache {
    config: QueryCacheConfig,
    entries: HashMap<String, CacheEntry>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    created_at: DateTime<Utc>,
    size_bytes: usize,
}

impl QueryCache {
    fn new(config: QueryCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
        }
    }

    fn get<T>(&self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        if !self.config.enabled {
            return None;
        }

        if let Some(entry) = self.entries.get(key) {
            let age = Utc::now().signed_duration_since(entry.created_at);
            if age.num_seconds() < self.config.ttl_seconds as i64 {
                if let Ok(value) = bincode::deserialize(&entry.data) {
                    return Some(value);
                }
            }
        }

        None
    }

    fn set<T>(&mut self, key: String, value: T)
    where
        T: Serialize,
    {
        if !self.config.enabled {
            return;
        }

        if let Ok(data) = bincode::serialize(&value) {
            let entry = CacheEntry {
                size_bytes: data.len(),
                data,
                created_at: Utc::now(),
            };

            self.entries.insert(key, entry);
            self.evict_if_needed();
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn evict_if_needed(&mut self) {
        // Remove expired entries
        let now = Utc::now();
        self.entries.retain(|_, entry| {
            let age = now.signed_duration_since(entry.created_at);
            age.num_seconds() < self.config.ttl_seconds as i64
        });

        // Remove entries if over limit
        if self.entries.len() > self.config.max_entries {
            let mut entries: Vec<_> = self
                .entries
                .iter()
                .map(|(k, v)| (k.clone(), v.created_at))
                .collect();
            entries.sort_by_key(|(_, created_at)| *created_at);

            let to_remove = self.entries.len() - self.config.max_entries;
            for (key, _) in entries.iter().take(to_remove) {
                self.entries.remove(key);
            }
        }

        // Check memory usage
        let total_size: usize = self.entries.values().map(|e| e.size_bytes).sum();
        let max_size = self.config.max_memory_mb * 1024 * 1024;

        if total_size > max_size {
            let mut entries: Vec<_> = self
                .entries
                .iter()
                .map(|(k, v)| (k.clone(), v.created_at, v.size_bytes))
                .collect();
            entries.sort_by_key(|(_, created_at, _)| *created_at);

            let mut current_size = total_size;
            for (key, _, size_bytes) in entries {
                if current_size <= max_size {
                    break;
                }
                current_size -= size_bytes;
                self.entries.remove(&key);
            }
        }
    }
}

/// Command bus metrics
#[derive(Debug, Clone, Default)]
pub struct CommandBusMetrics {
    pub commands_received: u64,
    pub commands_succeeded: u64,
    pub commands_failed: u64,
    pub active_commands: u64,
}

/// Query bus metrics
#[derive(Debug, Clone, Default)]
pub struct QueryBusMetrics {
    pub queries_received: u64,
    pub queries_succeeded: u64,
    pub queries_failed: u64,
    pub active_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Read model metrics
#[derive(Debug, Clone, Default)]
pub struct ReadModelMetrics {
    pub projection_errors: HashMap<String, u64>,
    pub events_processed: u64,
    pub projections_active: u64,
}

/// Complete CQRS system coordinator
pub struct CQRSSystem {
    pub command_bus: CommandBus,
    pub query_bus: QueryBus,
    pub read_model_manager: ReadModelManager,
    config: CQRSConfig,
}

impl CQRSSystem {
    /// Create a new CQRS system
    pub fn new(
        config: CQRSConfig,
        event_store: Arc<dyn EventStoreTrait>,
        event_stream: Arc<dyn EventStream>,
    ) -> Self {
        let command_bus = CommandBus::new(config.clone(), event_store);
        let query_bus = QueryBus::new(config.clone());
        let read_model_manager = ReadModelManager::new(event_stream);

        Self {
            command_bus,
            query_bus,
            read_model_manager,
            config,
        }
    }

    /// Start the CQRS system
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting CQRS system");

        // Start read model processing
        let system_clone = Arc::clone(&self);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(1000));
            loop {
                interval.tick().await;
                if let Err(e) = system_clone.read_model_manager.process_events().await {
                    error!("Error processing read model events: {}", e);
                }
            }
        });

        info!("CQRS system started successfully");
        Ok(())
    }

    /// Get system health status
    pub async fn health_check(&self) -> CQRSHealthStatus {
        let command_metrics = self.command_bus.get_metrics().await;
        let query_metrics = self.query_bus.get_metrics().await;
        let read_model_metrics = self.read_model_manager.get_metrics().await;

        CQRSHealthStatus {
            command_bus_healthy: command_metrics.active_commands
                < self.config.max_concurrent_commands as u64,
            query_bus_healthy: query_metrics.active_queries
                < self.config.max_concurrent_queries as u64,
            read_models_healthy: read_model_metrics.projection_errors.is_empty(),
            command_metrics,
            query_metrics,
            read_model_metrics,
        }
    }
}

/// CQRS system health status
#[derive(Debug, Clone)]
pub struct CQRSHealthStatus {
    pub command_bus_healthy: bool,
    pub query_bus_healthy: bool,
    pub read_models_healthy: bool,
    pub command_metrics: CommandBusMetrics,
    pub query_metrics: QueryBusMetrics,
    pub read_model_metrics: ReadModelMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestCommand {
        id: Uuid,
        aggregate_id: String,
        data: String,
    }

    impl Command for TestCommand {
        type AggregateId = String;
        type EventType = String;

        fn command_id(&self) -> Uuid {
            self.id
        }

        fn aggregate_id(&self) -> Self::AggregateId {
            self.aggregate_id.clone()
        }

        fn validate(&self) -> Result<()> {
            if self.data.is_empty() {
                return Err(anyhow!("Data cannot be empty"));
            }
            Ok(())
        }

        fn expected_version(&self) -> Option<u64> {
            None
        }
    }

    #[derive(Debug, Clone)]
    struct TestQuery {
        id: Uuid,
        filter: String,
    }

    impl Query for TestQuery {
        type Result = Vec<String>;

        fn query_id(&self) -> Uuid {
            self.id
        }

        fn validate(&self) -> Result<()> {
            Ok(())
        }

        fn cache_key(&self) -> Option<String> {
            Some(format!("test_query_{}", self.filter))
        }

        fn timeout_ms(&self) -> Option<u64> {
            Some(5000)
        }
    }

    #[tokio::test]
    async fn test_cqrs_config_defaults() {
        let config = CQRSConfig::default();
        assert_eq!(config.command_timeout_ms, 30000);
        assert_eq!(config.query_timeout_ms, 10000);
        assert!(config.enable_command_validation);
        assert!(config.enable_query_optimization);
    }

    #[tokio::test]
    async fn test_command_validation() {
        let valid_command = TestCommand {
            id: Uuid::new_v4(),
            aggregate_id: "test".to_string(),
            data: "valid data".to_string(),
        };

        let invalid_command = TestCommand {
            id: Uuid::new_v4(),
            aggregate_id: "test".to_string(),
            data: "".to_string(),
        };

        assert!(valid_command.validate().is_ok());
        assert!(invalid_command.validate().is_err());
    }

    #[tokio::test]
    async fn test_query_cache_key() {
        let query = TestQuery {
            id: Uuid::new_v4(),
            filter: "status=active".to_string(),
        };

        assert_eq!(
            query.cache_key(),
            Some("test_query_status=active".to_string())
        );
    }
}
