//! # Stateful Stream Processing
//!
//! State management for advanced stream processing operations.
//!
//! This module provides comprehensive state management capabilities for stateful
//! stream processing, including state stores, checkpointing, recovery, and
//! distributed state synchronization.

use crate::StreamEvent;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;
use tracing::{error, info};
use uuid::Uuid;

/// State store backend type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum StateBackend {
    /// In-memory state store (volatile)
    Memory,
    /// File-based persistent state
    File,
    /// RocksDB backend for large state
    RocksDB,
    /// Redis backend for distributed state
    Redis,
    /// Custom backend implementation
    Custom,
}

/// State value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum StateValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Binary(Vec<u8>),
    List(Vec<StateValue>),
    Map(HashMap<String, StateValue>),
    Counter(i64),
    Timestamp(DateTime<Utc>),
}

impl StateValue {
    /// Merge two state values (for aggregations)
    pub fn merge(&self, other: &StateValue) -> Result<StateValue> {
        match (self, other) {
            (StateValue::Integer(a), StateValue::Integer(b)) => Ok(StateValue::Integer(a + b)),
            (StateValue::Float(a), StateValue::Float(b)) => Ok(StateValue::Float(a + b)),
            (StateValue::Counter(a), StateValue::Counter(b)) => Ok(StateValue::Counter(a + b)),
            (StateValue::List(a), StateValue::List(b)) => {
                let mut merged = a.clone();
                merged.extend(b.clone());
                Ok(StateValue::List(merged))
            }
            (StateValue::Map(a), StateValue::Map(b)) => {
                let mut merged = a.clone();
                for (k, v) in b {
                    merged.insert(k.clone(), v.clone());
                }
                Ok(StateValue::Map(merged))
            }
            _ => Err(anyhow!("Cannot merge incompatible state value types")),
        }
    }
}

/// State store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateConfig {
    pub backend: StateBackend,
    pub checkpoint_interval: Duration,
    pub checkpoint_path: Option<PathBuf>,
    pub compaction_interval: Duration,
    pub ttl: Option<Duration>,
    pub max_size: Option<usize>,
    pub enable_changelog: bool,
    pub enable_metrics: bool,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            backend: StateBackend::Memory,
            checkpoint_interval: Duration::minutes(5),
            checkpoint_path: None,
            compaction_interval: Duration::hours(1),
            ttl: None,
            max_size: Some(1_000_000),
            enable_changelog: true,
            enable_metrics: true,
        }
    }
}

/// State operation for changelog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateOperation {
    pub timestamp: DateTime<Utc>,
    pub key: String,
    pub operation: StateOperationType,
    pub value: Option<StateValue>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateOperationType {
    Put,
    Delete,
    Merge,
    Clear,
}

/// State store statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateStatistics {
    pub total_keys: usize,
    pub total_size_bytes: usize,
    pub reads: u64,
    pub writes: u64,
    pub deletes: u64,
    pub checkpoints: u64,
    pub last_checkpoint: Option<DateTime<Utc>>,
    pub last_compaction: Option<DateTime<Utc>>,
}

/// State store trait for different backend implementations
#[async_trait::async_trait]
pub trait StateStore: Send + Sync {
    /// Get a value by key
    async fn get(&self, key: &str) -> Result<Option<StateValue>>;

    /// Put a value
    async fn put(&self, key: &str, value: StateValue) -> Result<()>;

    /// Delete a value
    async fn delete(&self, key: &str) -> Result<()>;

    /// Get multiple values
    async fn multi_get(&self, keys: &[String]) -> Result<HashMap<String, StateValue>>;

    /// Scan a key range
    async fn scan(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<(String, StateValue)>>;

    /// Clear all state
    async fn clear(&self) -> Result<()>;

    /// Create a checkpoint
    async fn checkpoint(&self) -> Result<String>;

    /// Restore from checkpoint
    async fn restore(&self, checkpoint_id: &str) -> Result<()>;

    /// Get statistics
    async fn statistics(&self) -> Result<StateStatistics>;
}

/// In-memory state store implementation
pub struct MemoryStateStore {
    data: Arc<RwLock<BTreeMap<String, StateValue>>>,
    changelog: Arc<RwLock<Vec<StateOperation>>>,
    statistics: Arc<RwLock<StateStatistics>>,
    config: StateConfig,
}

impl MemoryStateStore {
    pub fn new(config: StateConfig) -> Self {
        Self {
            data: Arc::new(RwLock::new(BTreeMap::new())),
            changelog: Arc::new(RwLock::new(Vec::new())),
            statistics: Arc::new(RwLock::new(StateStatistics::default())),
            config,
        }
    }

    async fn add_to_changelog(&self, operation: StateOperation) {
        if self.config.enable_changelog {
            self.changelog.write().await.push(operation);
        }
    }

    async fn apply_ttl(&self) {
        if let Some(ttl) = self.config.ttl {
            let cutoff = Utc::now() - ttl;
            let mut data = self.data.write().await;
            let keys_to_remove: Vec<String> = data
                .iter()
                .filter_map(|(k, v)| {
                    if let StateValue::Map(map) = v {
                        if let Some(StateValue::Timestamp(ts)) = map.get("_timestamp") {
                            if *ts < cutoff {
                                return Some(k.clone());
                            }
                        }
                    }
                    None
                })
                .collect();

            for key in keys_to_remove {
                data.remove(&key);
            }
        }
    }
}

#[async_trait::async_trait]
impl StateStore for MemoryStateStore {
    async fn get(&self, key: &str) -> Result<Option<StateValue>> {
        self.statistics.write().await.reads += 1;
        let data = self.data.read().await;
        Ok(data.get(key).cloned())
    }

    async fn put(&self, key: &str, value: StateValue) -> Result<()> {
        self.statistics.write().await.writes += 1;

        // Add timestamp for TTL
        let mut value_with_ts = value;
        if self.config.ttl.is_some() {
            if let StateValue::Map(ref mut map) = value_with_ts {
                map.insert("_timestamp".to_string(), StateValue::Timestamp(Utc::now()));
            }
        }

        self.data
            .write()
            .await
            .insert(key.to_string(), value_with_ts.clone());

        self.add_to_changelog(StateOperation {
            timestamp: Utc::now(),
            key: key.to_string(),
            operation: StateOperationType::Put,
            value: Some(value_with_ts),
            metadata: HashMap::new(),
        })
        .await;

        // Check size limit
        if let Some(max_size) = self.config.max_size {
            let data = self.data.read().await;
            if data.len() > max_size {
                drop(data);
                // Evict oldest entries (simple LRU approximation)
                let mut data = self.data.write().await;
                let to_remove = data.len() - max_size;
                let keys_to_remove: Vec<String> = data.keys().take(to_remove).cloned().collect();
                for key in keys_to_remove {
                    data.remove(&key);
                }
            }
        }

        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.statistics.write().await.deletes += 1;
        self.data.write().await.remove(key);

        self.add_to_changelog(StateOperation {
            timestamp: Utc::now(),
            key: key.to_string(),
            operation: StateOperationType::Delete,
            value: None,
            metadata: HashMap::new(),
        })
        .await;

        Ok(())
    }

    async fn multi_get(&self, keys: &[String]) -> Result<HashMap<String, StateValue>> {
        let mut stats = self.statistics.write().await;
        stats.reads += keys.len() as u64;
        drop(stats);

        let data = self.data.read().await;
        let mut result = HashMap::new();

        for key in keys {
            if let Some(value) = data.get(key) {
                result.insert(key.clone(), value.clone());
            }
        }

        Ok(result)
    }

    async fn scan(&self, prefix: &str, limit: Option<usize>) -> Result<Vec<(String, StateValue)>> {
        self.statistics.write().await.reads += 1;

        let data = self.data.read().await;
        let iter = data
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix));

        let result: Vec<(String, StateValue)> = match limit {
            Some(n) => iter.take(n).map(|(k, v)| (k.clone(), v.clone())).collect(),
            None => iter.map(|(k, v)| (k.clone(), v.clone())).collect(),
        };

        Ok(result)
    }

    async fn clear(&self) -> Result<()> {
        self.data.write().await.clear();

        self.add_to_changelog(StateOperation {
            timestamp: Utc::now(),
            key: String::new(),
            operation: StateOperationType::Clear,
            value: None,
            metadata: HashMap::new(),
        })
        .await;

        Ok(())
    }

    async fn checkpoint(&self) -> Result<String> {
        let checkpoint_id = Uuid::new_v4().to_string();

        if let Some(ref checkpoint_path) = self.config.checkpoint_path {
            let checkpoint_file = checkpoint_path.join(format!("{checkpoint_id}.checkpoint"));

            // Serialize state
            let data = self.data.read().await;
            let checkpoint_data = serde_json::to_vec(&*data)?;

            // Write to file
            let mut file = fs::File::create(&checkpoint_file).await?;
            file.write_all(&checkpoint_data).await?;
            file.sync_all().await?;

            info!(
                "Created checkpoint {} at {:?}",
                checkpoint_id, checkpoint_file
            );
        }

        let mut stats = self.statistics.write().await;
        stats.checkpoints += 1;
        stats.last_checkpoint = Some(Utc::now());

        Ok(checkpoint_id)
    }

    async fn restore(&self, checkpoint_id: &str) -> Result<()> {
        if let Some(ref checkpoint_path) = self.config.checkpoint_path {
            let checkpoint_file = checkpoint_path.join(format!("{checkpoint_id}.checkpoint"));

            // Read checkpoint file
            let mut file = fs::File::open(&checkpoint_file).await?;
            let mut checkpoint_data = Vec::new();
            file.read_to_end(&mut checkpoint_data).await?;

            // Deserialize and restore
            let restored_data: BTreeMap<String, StateValue> =
                serde_json::from_slice(&checkpoint_data)?;
            *self.data.write().await = restored_data;

            info!("Restored from checkpoint {}", checkpoint_id);
        } else {
            return Err(anyhow!("No checkpoint path configured"));
        }

        Ok(())
    }

    async fn statistics(&self) -> Result<StateStatistics> {
        self.apply_ttl().await;

        let mut stats = self.statistics.write().await.clone();
        let data = self.data.read().await;
        stats.total_keys = data.len();

        // Estimate size
        stats.total_size_bytes = data
            .values()
            .map(|v| serde_json::to_vec(v).map(|vec| vec.len()).unwrap_or(0))
            .sum();

        Ok(stats)
    }
}

/// State processor for managing stateful operations
pub struct StateProcessor {
    stores: HashMap<String, Arc<dyn StateStore>>,
    default_store: Arc<dyn StateStore>,
    config: StateConfig,
    checkpoint_task: Option<tokio::task::JoinHandle<()>>,
}

impl StateProcessor {
    pub fn new(config: StateConfig) -> Self {
        let default_store = Arc::new(MemoryStateStore::new(config.clone())) as Arc<dyn StateStore>;

        Self {
            stores: HashMap::new(),
            default_store: default_store.clone(),
            config,
            checkpoint_task: None,
        }
    }

    /// Start automatic checkpointing
    pub async fn start_checkpointing(&mut self) {
        let store = self.default_store.clone();
        let interval = self.config.checkpoint_interval;

        let task = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(
                interval
                    .to_std()
                    .expect("checkpoint interval should be valid std Duration"),
            );

            loop {
                interval_timer.tick().await;

                match store.checkpoint().await {
                    Ok(checkpoint_id) => {
                        info!("Automatic checkpoint created: {}", checkpoint_id);
                    }
                    Err(e) => {
                        error!("Failed to create checkpoint: {}", e);
                    }
                }
            }
        });

        self.checkpoint_task = Some(task);
    }

    /// Stop automatic checkpointing
    pub fn stop_checkpointing(&mut self) {
        if let Some(task) = self.checkpoint_task.take() {
            task.abort();
        }
    }

    /// Register a named state store
    pub fn register_store(&mut self, name: String, store: Arc<dyn StateStore>) {
        self.stores.insert(name, store);
    }

    /// Get a named state store
    pub fn get_store(&self, name: &str) -> Option<Arc<dyn StateStore>> {
        self.stores.get(name).cloned()
    }

    /// Get the default state store
    pub fn default_store(&self) -> Arc<dyn StateStore> {
        self.default_store.clone()
    }

    /// Process a stream event with state
    pub async fn process_with_state<F, R>(
        &self,
        event: &StreamEvent,
        state_key: &str,
        processor: F,
    ) -> Result<R>
    where
        F: FnOnce(&StreamEvent, Option<StateValue>) -> Result<(R, Option<StateValue>)>,
    {
        // Get current state
        let current_state = self.default_store.get(state_key).await?;

        // Process event with state
        let (result, new_state) = processor(event, current_state)?;

        // Update state if changed
        if let Some(state) = new_state {
            self.default_store.put(state_key, state).await?;
        }

        Ok(result)
    }
}

/// Builder for state processors
pub struct StateProcessorBuilder {
    config: StateConfig,
    stores: HashMap<String, Arc<dyn StateStore>>,
}

impl Default for StateProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StateProcessorBuilder {
    pub fn new() -> Self {
        Self {
            config: StateConfig::default(),
            stores: HashMap::new(),
        }
    }

    pub fn with_backend(mut self, backend: StateBackend) -> Self {
        self.config.backend = backend;
        self
    }

    pub fn with_checkpoint_interval(mut self, interval: Duration) -> Self {
        self.config.checkpoint_interval = interval;
        self
    }

    pub fn with_checkpoint_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.checkpoint_path = Some(path.into());
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.config.ttl = Some(ttl);
        self
    }

    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.config.max_size = Some(max_size);
        self
    }

    pub fn add_store(mut self, name: String, store: Arc<dyn StateStore>) -> Self {
        self.stores.insert(name, store);
        self
    }

    pub fn build(self) -> StateProcessor {
        let mut processor = StateProcessor::new(self.config);

        for (name, store) in self.stores {
            processor.register_store(name, store);
        }

        processor
    }
}

/// Helper functions for common state patterns
pub mod patterns {
    use super::*;

    /// Counter state pattern
    pub async fn increment_counter(
        store: &dyn StateStore,
        key: &str,
        increment: i64,
    ) -> Result<i64> {
        let current = store.get(key).await?;
        let new_value = match current {
            Some(StateValue::Counter(n)) => n + increment,
            _ => increment,
        };

        store.put(key, StateValue::Counter(new_value)).await?;
        Ok(new_value)
    }

    /// List accumulator pattern
    pub async fn append_to_list(
        store: &dyn StateStore,
        key: &str,
        value: StateValue,
    ) -> Result<()> {
        let current = store.get(key).await?;
        let mut list = match current {
            Some(StateValue::List(l)) => l,
            _ => Vec::new(),
        };

        list.push(value);
        store.put(key, StateValue::List(list)).await?;
        Ok(())
    }

    /// Map merger pattern
    pub async fn merge_map(
        store: &dyn StateStore,
        key: &str,
        updates: HashMap<String, StateValue>,
    ) -> Result<()> {
        let current = store.get(key).await?;
        let mut map = match current {
            Some(StateValue::Map(m)) => m,
            _ => HashMap::new(),
        };

        for (k, v) in updates {
            map.insert(k, v);
        }

        store.put(key, StateValue::Map(map)).await?;
        Ok(())
    }

    /// Time-based window state
    pub async fn update_time_window(
        store: &dyn StateStore,
        key: &str,
        value: StateValue,
        window_duration: Duration,
    ) -> Result<Vec<StateValue>> {
        let current = store.get(key).await?;
        let mut window_data = match current {
            Some(StateValue::List(l)) => l,
            _ => Vec::new(),
        };

        // Add new value with timestamp
        let mut value_with_time = HashMap::new();
        value_with_time.insert("value".to_string(), value);
        value_with_time.insert("timestamp".to_string(), StateValue::Timestamp(Utc::now()));
        window_data.push(StateValue::Map(value_with_time));

        // Remove expired values
        let cutoff = Utc::now() - window_duration;
        window_data.retain(|v| {
            if let StateValue::Map(m) = v {
                if let Some(StateValue::Timestamp(ts)) = m.get("timestamp") {
                    return *ts >= cutoff;
                }
            }
            false
        });

        store
            .put(key, StateValue::List(window_data.clone()))
            .await?;
        Ok(window_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventMetadata;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_memory_state_store() {
        let config = StateConfig::default();
        let store = MemoryStateStore::new(config);

        // Test put and get
        store
            .put("key1", StateValue::String("value1".to_string()))
            .await
            .unwrap();
        let value = store.get("key1").await.unwrap();
        assert!(matches!(value, Some(StateValue::String(s)) if s == "value1"));

        // Test delete
        store.delete("key1").await.unwrap();
        let value = store.get("key1").await.unwrap();
        assert!(value.is_none());

        // Test statistics
        let stats = store.statistics().await.unwrap();
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.deletes, 1);
    }

    #[tokio::test]
    async fn test_state_ttl() {
        let config = StateConfig {
            ttl: Some(Duration::milliseconds(100)),
            ..Default::default()
        };
        let store = MemoryStateStore::new(config);

        // Put value with TTL
        let mut map = HashMap::new();
        map.insert("data".to_string(), StateValue::String("test".to_string()));
        store.put("key1", StateValue::Map(map)).await.unwrap();

        // Value should exist immediately
        assert!(store.get("key1").await.unwrap().is_some());

        // Wait for TTL to expire
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Force TTL application by getting statistics
        let _ = store.statistics().await.unwrap();

        // Value should be gone
        assert!(store.get("key1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_checkpoint_restore() {
        let temp_dir = TempDir::new().unwrap();
        let config = StateConfig {
            checkpoint_path: Some(temp_dir.path().to_path_buf()),
            ..Default::default()
        };

        let store = MemoryStateStore::new(config.clone());

        // Add some data
        store
            .put("key1", StateValue::String("value1".to_string()))
            .await
            .unwrap();
        store.put("key2", StateValue::Integer(42)).await.unwrap();

        // Create checkpoint
        let checkpoint_id = store.checkpoint().await.unwrap();

        // Clear store
        store.clear().await.unwrap();
        assert!(store.get("key1").await.unwrap().is_none());

        // Restore from checkpoint
        store.restore(&checkpoint_id).await.unwrap();

        // Data should be restored
        let value1 = store.get("key1").await.unwrap();
        assert!(matches!(value1, Some(StateValue::String(s)) if s == "value1"));

        let value2 = store.get("key2").await.unwrap();
        assert!(matches!(value2, Some(StateValue::Integer(i)) if i == 42));
    }

    #[tokio::test]
    async fn test_state_processor() {
        let processor = StateProcessorBuilder::new()
            .with_backend(StateBackend::Memory)
            .build();

        let event = StreamEvent::TripleAdded {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: "http://example.org/o".to_string(),
            graph: None,
            metadata: EventMetadata::default(),
        };

        // Process event with state
        let result = processor
            .process_with_state(&event, "counter", |_event, state| {
                let count = match state {
                    Some(StateValue::Counter(n)) => n + 1,
                    _ => 1,
                };
                Ok((count, Some(StateValue::Counter(count))))
            })
            .await
            .unwrap();

        assert_eq!(result, 1);

        // Process again
        let result = processor
            .process_with_state(&event, "counter", |_event, state| {
                let count = match state {
                    Some(StateValue::Counter(n)) => n + 1,
                    _ => 1,
                };
                Ok((count, Some(StateValue::Counter(count))))
            })
            .await
            .unwrap();

        assert_eq!(result, 2);
    }

    #[tokio::test]
    async fn test_state_patterns() {
        let config = StateConfig::default();
        let store = MemoryStateStore::new(config);

        // Test counter pattern
        let count = patterns::increment_counter(&store, "counter1", 5)
            .await
            .unwrap();
        assert_eq!(count, 5);

        let count = patterns::increment_counter(&store, "counter1", 3)
            .await
            .unwrap();
        assert_eq!(count, 8);

        // Test list pattern
        patterns::append_to_list(&store, "list1", StateValue::String("item1".to_string()))
            .await
            .unwrap();
        patterns::append_to_list(&store, "list1", StateValue::String("item2".to_string()))
            .await
            .unwrap();

        let list = store.get("list1").await.unwrap();
        if let Some(StateValue::List(items)) = list {
            assert_eq!(items.len(), 2);
        } else {
            panic!("Expected list");
        }

        // Test map merger pattern
        let mut updates = HashMap::new();
        updates.insert(
            "field1".to_string(),
            StateValue::String("value1".to_string()),
        );
        updates.insert("field2".to_string(), StateValue::Integer(42));

        patterns::merge_map(&store, "map1", updates).await.unwrap();

        let map = store.get("map1").await.unwrap();
        if let Some(StateValue::Map(m)) = map {
            assert_eq!(m.len(), 2);
            assert!(matches!(m.get("field1"), Some(StateValue::String(s)) if s == "value1"));
            assert!(matches!(m.get("field2"), Some(StateValue::Integer(i)) if *i == 42));
        } else {
            panic!("Expected map");
        }
    }
}
