//! Schema caching with hot-reload support
//!
//! This module provides caching for generated GraphQL schemas with support for:
//! - File-based schema caching with hot-reload
//! - In-memory LRU caching
//! - Automatic invalidation on ontology changes
//! - Schema versioning

use crate::schema::{RdfVocabulary, SchemaGenerationConfig, SchemaGenerator};
use crate::types::Schema;
use anyhow::Result;
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Schema cache entry with metadata
#[derive(Debug, Clone)]
pub struct SchemaCacheEntry {
    pub schema: Schema,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub access_count: u64,
    pub version: u64,
    pub source_hash: String,
}

impl SchemaCacheEntry {
    pub fn new(schema: Schema, source_hash: String, version: u64) -> Self {
        let now = SystemTime::now();
        Self {
            schema,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            version,
            source_hash,
        }
    }

    pub fn access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now();
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed().unwrap_or(Duration::from_secs(0))
    }
}

/// Configuration for schema cache
#[derive(Debug, Clone)]
pub struct SchemaCacheConfig {
    /// Maximum number of schemas to cache in memory
    pub max_cache_size: usize,
    /// TTL for cached schemas
    pub ttl: Duration,
    /// Enable hot-reload from file system
    pub enable_hot_reload: bool,
    /// Directory to watch for schema changes
    pub watch_directory: Option<PathBuf>,
    /// Interval for checking file changes
    pub reload_check_interval: Duration,
    /// Enable cache statistics
    pub enable_statistics: bool,
}

impl Default for SchemaCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 100,
            ttl: Duration::from_secs(3600), // 1 hour
            enable_hot_reload: true,
            watch_directory: None,
            reload_check_interval: Duration::from_secs(5),
            enable_statistics: true,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub reloads: u64,
    pub total_schemas: u64,
    pub current_size: usize,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Schema cache with hot-reload support
pub struct SchemaCache {
    cache: Arc<RwLock<LruCache<String, SchemaCacheEntry>>>,
    config: SchemaCacheConfig,
    statistics: Arc<RwLock<CacheStatistics>>,
    file_watchers: Arc<RwLock<HashMap<PathBuf, SystemTime>>>,
    next_version: Arc<RwLock<u64>>,
}

impl SchemaCache {
    /// Create a new schema cache
    pub fn new(config: SchemaCacheConfig) -> Self {
        let cache_size = NonZeroUsize::new(config.max_cache_size)
            .unwrap_or(NonZeroUsize::new(100).expect("Default cache size should be valid"));

        Self {
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            config: config.clone(),
            statistics: Arc::new(RwLock::new(CacheStatistics::default())),
            file_watchers: Arc::new(RwLock::new(HashMap::new())),
            next_version: Arc::new(RwLock::new(1)),
        }
    }

    /// Get schema from cache
    pub async fn get(&self, key: &str) -> Option<Schema> {
        let mut cache = self.cache.write().await;

        if let Some(entry) = cache.get_mut(key) {
            // Check if entry is expired
            if entry.age() > self.config.ttl {
                debug!("Schema cache entry expired for key: {}", key);
                cache.pop(key);
                self.record_miss().await;
                return None;
            }

            entry.access();
            self.record_hit().await;
            debug!(
                "Schema cache hit for key: {} (version: {}, age: {:?})",
                key,
                entry.version,
                entry.age()
            );
            Some(entry.schema.clone())
        } else {
            self.record_miss().await;
            debug!("Schema cache miss for key: {}", key);
            None
        }
    }

    /// Put schema in cache
    pub async fn put(&self, key: String, schema: Schema, source_hash: String) {
        let mut cache = self.cache.write().await;
        let mut version_guard = self.next_version.write().await;
        let version = *version_guard;
        *version_guard += 1;

        let entry = SchemaCacheEntry::new(schema, source_hash, version);

        // Check if we're evicting an entry
        if cache.len() >= cache.cap().get() && !cache.contains(&key) {
            self.record_eviction().await;
        }

        cache.put(key.clone(), entry);
        info!("Cached schema for key: {} (version: {})", key, version);

        // Update statistics
        let mut stats = self.statistics.write().await;
        stats.total_schemas += 1;
        stats.current_size = cache.len();
    }

    /// Generate and cache schema from vocabulary
    pub async fn get_or_generate(
        &self,
        key: &str,
        vocabulary: RdfVocabulary,
        config: SchemaGenerationConfig,
    ) -> Result<Schema> {
        // Try to get from cache first
        if let Some(schema) = self.get(key).await {
            return Ok(schema);
        }

        // Generate schema
        let generator = SchemaGenerator::new()
            .with_config(config)
            .with_vocabulary(vocabulary.clone());
        let schema = generator.generate_schema()?;

        // Compute source hash (simple approach - in production use proper hashing)
        let source_hash = format!("{:?}", vocabulary).len().to_string();

        // Cache the schema
        self.put(key.to_string(), schema.clone(), source_hash).await;

        Ok(schema)
    }

    /// Invalidate cache entry
    pub async fn invalidate(&self, key: &str) -> bool {
        let mut cache = self.cache.write().await;
        if cache.pop(key).is_some() {
            info!("Invalidated schema cache for key: {}", key);

            let mut stats = self.statistics.write().await;
            stats.current_size = cache.len();
            true
        } else {
            false
        }
    }

    /// Clear all cache entries
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let count = cache.len();
        cache.clear();

        info!("Cleared schema cache ({} entries removed)", count);

        let mut stats = self.statistics.write().await;
        stats.current_size = 0;
    }

    /// Get cache statistics
    pub async fn statistics(&self) -> CacheStatistics {
        self.statistics.read().await.clone()
    }

    /// Start hot-reload watcher
    pub async fn start_hot_reload(&self) {
        if !self.config.enable_hot_reload {
            return;
        }

        let watch_dir = match &self.config.watch_directory {
            Some(dir) => dir.clone(),
            None => {
                warn!("Hot-reload enabled but no watch directory specified");
                return;
            }
        };

        info!(
            "Starting schema hot-reload watcher for directory: {:?}",
            watch_dir
        );

        let cache = Arc::clone(&self.cache);
        let file_watchers = Arc::clone(&self.file_watchers);
        let statistics = Arc::clone(&self.statistics);
        let check_interval = self.config.reload_check_interval;

        tokio::spawn(async move {
            let mut interval = interval(check_interval);

            loop {
                interval.tick().await;

                match Self::check_directory_changes(&watch_dir, &file_watchers, &cache, &statistics)
                    .await
                {
                    Ok(changed) => {
                        if changed > 0 {
                            info!("Hot-reload detected {} schema file changes", changed);
                        }
                    }
                    Err(e) => {
                        error!("Error checking directory changes: {}", e);
                    }
                }
            }
        });
    }

    /// Check directory for changes
    async fn check_directory_changes(
        dir: &Path,
        watchers: &Arc<RwLock<HashMap<PathBuf, SystemTime>>>,
        cache: &Arc<RwLock<LruCache<String, SchemaCacheEntry>>>,
        statistics: &Arc<RwLock<CacheStatistics>>,
    ) -> Result<usize> {
        let mut changed_count = 0;

        // Read directory entries
        let mut entries = fs::read_dir(dir).await?;

        let mut current_files = HashMap::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Only watch .ttl, .rdf, .owl files
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy();
                if !matches!(ext_str.as_ref(), "ttl" | "rdf" | "owl" | "nt") {
                    continue;
                }
            } else {
                continue;
            }

            // Get file modification time
            if let Ok(metadata) = fs::metadata(&path).await {
                if let Ok(modified) = metadata.modified() {
                    current_files.insert(path.clone(), modified);
                }
            }
        }

        let mut watchers_guard = watchers.write().await;

        // Check for changes or new files
        for (file_path, modified_time) in &current_files {
            match watchers_guard.get(file_path) {
                Some(cached_time) if cached_time != modified_time => {
                    // File was modified
                    debug!("Detected change in schema file: {:?}", file_path);

                    // Invalidate related cache entries
                    let mut cache_guard = cache.write().await;
                    cache_guard.clear(); // Simple approach: clear all cache

                    watchers_guard.insert(file_path.clone(), *modified_time);
                    changed_count += 1;

                    let mut stats = statistics.write().await;
                    stats.reloads += 1;
                }
                None => {
                    // New file detected
                    debug!("Detected new schema file: {:?}", file_path);
                    watchers_guard.insert(file_path.clone(), *modified_time);
                    changed_count += 1;
                }
                _ => {}
            }
        }

        // Remove deleted files from watchers
        let deleted: Vec<_> = watchers_guard
            .keys()
            .filter(|path| !current_files.contains_key(*path))
            .cloned()
            .collect();

        for path in deleted {
            debug!("Detected deletion of schema file: {:?}", path);
            watchers_guard.remove(&path);
            changed_count += 1;
        }

        Ok(changed_count)
    }

    /// Record cache hit
    async fn record_hit(&self) {
        if self.config.enable_statistics {
            let mut stats = self.statistics.write().await;
            stats.hits += 1;
        }
    }

    /// Record cache miss
    async fn record_miss(&self) {
        if self.config.enable_statistics {
            let mut stats = self.statistics.write().await;
            stats.misses += 1;
        }
    }

    /// Record cache eviction
    async fn record_eviction(&self) {
        if self.config.enable_statistics {
            let mut stats = self.statistics.write().await;
            stats.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_schema() -> Schema {
        Schema::new()
    }

    #[allow(dead_code)]
    fn create_test_vocabulary() -> RdfVocabulary {
        RdfVocabulary {
            classes: HashMap::new(),
            properties: HashMap::new(),
            namespaces: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_schema_cache_creation() {
        let config = SchemaCacheConfig::default();
        let cache = SchemaCache::new(config);

        let stats = cache.statistics().await;
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.current_size, 0);
    }

    #[tokio::test]
    async fn test_schema_cache_put_and_get() {
        let config = SchemaCacheConfig::default();
        let cache = SchemaCache::new(config);

        let schema = create_test_schema();
        let key = "test_schema".to_string();
        let hash = "test_hash".to_string();

        cache.put(key.clone(), schema.clone(), hash).await;

        let retrieved = cache.get(&key).await;
        assert!(retrieved.is_some());

        let stats = cache.statistics().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[tokio::test]
    async fn test_schema_cache_miss() {
        let config = SchemaCacheConfig::default();
        let cache = SchemaCache::new(config);

        let retrieved = cache.get("nonexistent").await;
        assert!(retrieved.is_none());

        let stats = cache.statistics().await;
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_schema_cache_invalidation() {
        let config = SchemaCacheConfig::default();
        let cache = SchemaCache::new(config);

        let schema = create_test_schema();
        let key = "test_schema".to_string();
        let hash = "test_hash".to_string();

        cache.put(key.clone(), schema, hash).await;
        assert!(cache.get(&key).await.is_some());

        cache.invalidate(&key).await;
        assert!(cache.get(&key).await.is_none());
    }

    #[tokio::test]
    async fn test_schema_cache_clear() {
        let config = SchemaCacheConfig::default();
        let cache = SchemaCache::new(config);

        for i in 0..5 {
            let schema = create_test_schema();
            let key = format!("schema_{}", i);
            let hash = format!("hash_{}", i);
            cache.put(key, schema, hash).await;
        }

        let stats = cache.statistics().await;
        assert_eq!(stats.current_size, 5);

        cache.clear().await;

        let stats = cache.statistics().await;
        assert_eq!(stats.current_size, 0);
    }

    #[tokio::test]
    async fn test_schema_cache_statistics() {
        let config = SchemaCacheConfig::default();
        let cache = SchemaCache::new(config);

        let schema = create_test_schema();
        cache
            .put("schema1".to_string(), schema.clone(), "hash1".to_string())
            .await;

        // Hit
        let _ = cache.get("schema1").await;

        // Miss
        let _ = cache.get("schema2").await;

        let stats = cache.statistics().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_schema_cache_ttl_expiration() {
        let config = SchemaCacheConfig {
            ttl: Duration::from_millis(100), // Very short TTL for testing
            ..Default::default()
        };

        let cache = SchemaCache::new(config);
        let schema = create_test_schema();

        cache
            .put("test".to_string(), schema, "hash".to_string())
            .await;

        // Should be in cache immediately
        assert!(cache.get("test").await.is_some());

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be expired now
        assert!(cache.get("test").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_entry_versioning() {
        let schema = create_test_schema();
        let entry1 = SchemaCacheEntry::new(schema.clone(), "hash1".to_string(), 1);
        let entry2 = SchemaCacheEntry::new(schema, "hash2".to_string(), 2);

        assert_eq!(entry1.version, 1);
        assert_eq!(entry2.version, 2);
        assert_ne!(entry1.source_hash, entry2.source_hash);
    }

    #[tokio::test]
    async fn test_cache_entry_access_tracking() {
        let schema = create_test_schema();
        let mut entry = SchemaCacheEntry::new(schema, "hash".to_string(), 1);

        assert_eq!(entry.access_count, 0);

        entry.access();
        assert_eq!(entry.access_count, 1);

        entry.access();
        assert_eq!(entry.access_count, 2);
    }
}
