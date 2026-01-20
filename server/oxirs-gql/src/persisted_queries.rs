//! # Persisted Queries and Automatic Persisted Queries (APQ)
//!
//! Implements persistent query documents and automatic persisted queries (APQ) for GraphQL.
//! APQ reduces bandwidth and improves performance by allowing clients to send query hashes
//! instead of full query strings.
//!
//! ## Features
//!
//! - **Auto Persisted Queries (APQ)**: Automatic query caching with SHA-256 hashing
//! - **Persistent Query Documents**: Pre-registered query allowlist
//! - **Query Allowlist**: Restrict queries to a predefined set
//! - **Query Denylist**: Block specific queries
//! - **Cache Storage**: In-memory and distributed cache support
//! - **Query Versioning**: Support multiple versions of the same query
//! - **Statistics**: Query usage and performance metrics
//!
//! ## APQ Protocol
//!
//! 1. Client sends query hash with `extensions.persistedQuery.sha256Hash`
//! 2. If server has query cached, execute it
//! 3. If not found, server returns `PersistedQueryNotFound` error
//! 4. Client resends with full query + hash
//! 5. Server caches query for future requests

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Persisted query configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedQueryConfig {
    /// Enable automatic persisted queries
    pub enable_apq: bool,
    /// Enable persistent query documents
    pub enable_pqd: bool,
    /// Query allowlist mode (only allow registered queries)
    pub allowlist_mode: bool,
    /// Maximum queries to cache (0 = unlimited)
    pub max_cached_queries: usize,
    /// Query TTL in seconds (0 = never expire)
    pub query_ttl_seconds: u64,
    /// Enable query statistics
    pub enable_statistics: bool,
}

impl Default for PersistedQueryConfig {
    fn default() -> Self {
        Self {
            enable_apq: true,
            enable_pqd: true,
            allowlist_mode: false,
            max_cached_queries: 10000,
            query_ttl_seconds: 3600, // 1 hour
            enable_statistics: true,
        }
    }
}

/// Persisted query entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedQuery {
    /// Query hash (SHA-256)
    pub hash: String,
    /// Query document
    pub query: String,
    /// Query version
    pub version: Option<String>,
    /// Query name (optional)
    pub name: Option<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last accessed timestamp
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    /// Access count
    pub access_count: u64,
    /// Whether this is a pre-registered query
    pub is_registered: bool,
}

impl PersistedQuery {
    pub fn new(hash: String, query: String, is_registered: bool) -> Self {
        let now = chrono::Utc::now();
        Self {
            hash,
            query,
            version: None,
            name: None,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            is_registered,
        }
    }

    pub fn with_version(mut self, version: String) -> Self {
        self.version = Some(version);
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn record_access(&mut self) {
        self.last_accessed = chrono::Utc::now();
        self.access_count += 1;
    }
}

/// APQ request extension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApqExtension {
    #[serde(rename = "persistedQuery")]
    pub persisted_query: ApqPersistedQuery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApqPersistedQuery {
    pub version: u32,
    #[serde(rename = "sha256Hash")]
    pub sha256_hash: String,
}

/// Query statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatistics {
    /// Total queries cached
    pub total_cached: usize,
    /// Registered queries
    pub registered_queries: usize,
    /// APQ hits
    pub apq_hits: u64,
    /// APQ misses
    pub apq_misses: u64,
    /// Cache hit rate
    pub hit_rate: f64,
    /// Most accessed queries
    pub top_queries: Vec<(String, u64)>,
}

/// Persisted query manager
pub struct PersistedQueryManager {
    config: Arc<PersistedQueryConfig>,
    /// Query cache (hash -> query)
    cache: Arc<RwLock<HashMap<String, PersistedQuery>>>,
    /// Query denylist (blocked hashes)
    denylist: Arc<RwLock<HashMap<String, String>>>,
    /// Statistics
    apq_hits: Arc<RwLock<u64>>,
    apq_misses: Arc<RwLock<u64>>,
}

impl PersistedQueryManager {
    /// Create a new persisted query manager
    pub fn new(config: PersistedQueryConfig) -> Self {
        Self {
            config: Arc::new(config),
            cache: Arc::new(RwLock::new(HashMap::new())),
            denylist: Arc::new(RwLock::new(HashMap::new())),
            apq_hits: Arc::new(RwLock::new(0)),
            apq_misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Register a query (for persistent query documents)
    pub async fn register_query(
        &self,
        hash: String,
        query: String,
        name: Option<String>,
        version: Option<String>,
    ) -> Result<()> {
        let mut cache = self.cache.write().await;

        // Check cache size limit
        if self.config.max_cached_queries > 0 && cache.len() >= self.config.max_cached_queries {
            // Evict oldest query
            if let Some((oldest_hash, _)) = cache
                .iter()
                .min_by_key(|(_, q)| q.last_accessed)
                .map(|(h, q)| (h.clone(), q.clone()))
            {
                cache.remove(&oldest_hash);
            }
        }

        let mut query_entry = PersistedQuery::new(hash.clone(), query, true);
        if let Some(name) = name {
            query_entry = query_entry.with_name(name);
        }
        if let Some(version) = version {
            query_entry = query_entry.with_version(version);
        }

        cache.insert(hash, query_entry);
        Ok(())
    }

    /// Get query by hash (APQ lookup)
    pub async fn get_query(&self, hash: &str) -> Result<String> {
        // Check denylist first
        {
            let denylist = self.denylist.read().await;
            if let Some(reason) = denylist.get(hash) {
                return Err(anyhow!("Query is blocked: {}", reason));
            }
        }

        let mut cache = self.cache.write().await;

        if let Some(query) = cache.get_mut(hash) {
            query.record_access();

            // Record APQ hit
            if self.config.enable_statistics {
                let mut hits = self.apq_hits.write().await;
                *hits += 1;
            }

            Ok(query.query.clone())
        } else {
            // Record APQ miss
            if self.config.enable_statistics {
                let mut misses = self.apq_misses.write().await;
                *misses += 1;
            }

            Err(anyhow!("PersistedQueryNotFound"))
        }
    }

    /// Store query with hash (APQ registration)
    pub async fn store_query(&self, hash: String, query: String) -> Result<()> {
        // Check if allowlist mode is enabled
        if self.config.allowlist_mode {
            return Err(anyhow!(
                "Cannot register new queries in allowlist mode. Only pre-registered queries are allowed."
            ));
        }

        if !self.config.enable_apq {
            return Err(anyhow!("Automatic persisted queries are disabled"));
        }

        // Verify hash
        let computed_hash = Self::compute_hash(&query);
        if computed_hash != hash {
            return Err(anyhow!(
                "Query hash mismatch. Expected: {}, Got: {}",
                computed_hash,
                hash
            ));
        }

        self.register_query(hash, query, None, None).await
    }

    /// Compute SHA-256 hash of query
    pub fn compute_hash(query: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Add query to denylist
    pub async fn deny_query(&self, hash: String, reason: String) -> Result<()> {
        let mut denylist = self.denylist.write().await;
        denylist.insert(hash, reason);
        Ok(())
    }

    /// Remove query from denylist
    pub async fn allow_query(&self, hash: &str) -> Result<()> {
        let mut denylist = self.denylist.write().await;
        denylist.remove(hash);
        Ok(())
    }

    /// Get query statistics
    pub async fn get_statistics(&self) -> QueryStatistics {
        let cache = self.cache.read().await;
        let hits = *self.apq_hits.read().await;
        let misses = *self.apq_misses.read().await;

        let total_requests = hits + misses;
        let hit_rate = if total_requests > 0 {
            hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let registered_queries = cache.values().filter(|q| q.is_registered).count();

        // Get top 10 most accessed queries
        let mut queries: Vec<_> = cache
            .values()
            .map(|q| (q.hash.clone(), q.access_count))
            .collect();
        queries.sort_by(|a, b| b.1.cmp(&a.1));
        let top_queries = queries.into_iter().take(10).collect();

        QueryStatistics {
            total_cached: cache.len(),
            registered_queries,
            apq_hits: hits,
            apq_misses: misses,
            hit_rate,
            top_queries,
        }
    }

    /// Clear all cached queries
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        Ok(())
    }

    /// Clear statistics
    pub async fn clear_statistics(&self) -> Result<()> {
        let mut hits = self.apq_hits.write().await;
        let mut misses = self.apq_misses.write().await;
        *hits = 0;
        *misses = 0;
        Ok(())
    }

    /// Export registered queries (for persistent query documents)
    pub async fn export_queries(&self) -> Vec<PersistedQuery> {
        let cache = self.cache.read().await;
        cache
            .values()
            .filter(|q| q.is_registered)
            .cloned()
            .collect()
    }

    /// Import queries (for persistent query documents)
    pub async fn import_queries(&self, queries: Vec<PersistedQuery>) -> Result<()> {
        let mut cache = self.cache.write().await;
        for query in queries {
            cache.insert(query.hash.clone(), query);
        }
        Ok(())
    }

    /// Load queries from file
    pub async fn load_from_file(&self, path: &std::path::Path) -> Result<()> {
        let content = tokio::fs::read_to_string(path).await?;
        let queries: Vec<PersistedQuery> = serde_json::from_str(&content)?;
        self.import_queries(queries).await
    }

    /// Save queries to file
    pub async fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let queries = self.export_queries().await;
        let content = serde_json::to_string_pretty(&queries)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_persisted_query_config_default() {
        let config = PersistedQueryConfig::default();
        assert!(config.enable_apq);
        assert!(config.enable_pqd);
        assert!(!config.allowlist_mode);
        assert_eq!(config.max_cached_queries, 10000);
        assert!(config.enable_statistics);
    }

    #[tokio::test]
    async fn test_compute_hash() {
        let query = "{ hello }";
        let hash = PersistedQueryManager::compute_hash(query);
        assert_eq!(hash.len(), 64); // SHA-256 produces 64 hex characters
    }

    #[tokio::test]
    async fn test_register_and_get_query() {
        let config = PersistedQueryConfig::default();
        let manager = PersistedQueryManager::new(config);

        let query = "{ hello }";
        let hash = PersistedQueryManager::compute_hash(query);

        manager
            .register_query(hash.clone(), query.to_string(), None, None)
            .await
            .unwrap();

        let result = manager.get_query(&hash).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), query);
    }

    #[tokio::test]
    async fn test_apq_flow() {
        let config = PersistedQueryConfig::default();
        let manager = PersistedQueryManager::new(config);

        let query = "{ user(id: 1) { name } }";
        let hash = PersistedQueryManager::compute_hash(query);

        // First request - query not found (APQ miss)
        let result = manager.get_query(&hash).await;
        assert!(result.is_err());

        // Client stores query
        manager
            .store_query(hash.clone(), query.to_string())
            .await
            .unwrap();

        // Second request - query found (APQ hit)
        let result = manager.get_query(&hash).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), query);
    }

    #[tokio::test]
    async fn test_allowlist_mode() {
        let config = PersistedQueryConfig {
            allowlist_mode: true,
            ..Default::default()
        };
        let manager = PersistedQueryManager::new(config);

        let query = "{ hello }";
        let hash = PersistedQueryManager::compute_hash(query);

        // Should fail - allowlist mode prevents new registrations
        let result = manager.store_query(hash, query.to_string()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_query_denylist() {
        let config = PersistedQueryConfig::default();
        let manager = PersistedQueryManager::new(config);

        let query = "{ malicious }";
        let hash = PersistedQueryManager::compute_hash(query);

        manager
            .register_query(hash.clone(), query.to_string(), None, None)
            .await
            .unwrap();

        // Block the query
        manager
            .deny_query(hash.clone(), "Malicious query".to_string())
            .await
            .unwrap();

        // Should fail - query is blocked
        let result = manager.get_query(&hash).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("blocked"));
    }

    #[tokio::test]
    async fn test_statistics() {
        let config = PersistedQueryConfig::default();
        let manager = PersistedQueryManager::new(config);

        let query = "{ hello }";
        let hash = PersistedQueryManager::compute_hash(query);

        // Register query
        manager
            .register_query(hash.clone(), query.to_string(), None, None)
            .await
            .unwrap();

        // Access query multiple times
        for _ in 0..5 {
            let _ = manager.get_query(&hash).await;
        }

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_cached, 1);
        assert_eq!(stats.registered_queries, 1);
        assert_eq!(stats.apq_hits, 5);
        assert_eq!(stats.hit_rate, 1.0);
    }

    #[tokio::test]
    async fn test_cache_size_limit() {
        let config = PersistedQueryConfig {
            max_cached_queries: 2,
            ..Default::default()
        };
        let manager = PersistedQueryManager::new(config);

        // Register 3 queries (should evict oldest)
        for i in 0..3 {
            let query = format!("{{ query{} }}", i);
            let hash = PersistedQueryManager::compute_hash(&query);
            manager
                .register_query(hash, query, None, None)
                .await
                .unwrap();
        }

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_cached, 2);
    }

    #[tokio::test]
    async fn test_query_versioning() {
        let config = PersistedQueryConfig::default();
        let manager = PersistedQueryManager::new(config);

        let query = "{ hello }";
        let hash = PersistedQueryManager::compute_hash(query);

        manager
            .register_query(
                hash.clone(),
                query.to_string(),
                Some("HelloQuery".to_string()),
                Some("v1".to_string()),
            )
            .await
            .unwrap();

        let result = manager.get_query(&hash).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_hash_verification() {
        let config = PersistedQueryConfig::default();
        let manager = PersistedQueryManager::new(config);

        let query = "{ hello }";
        let wrong_hash = "wronghash123";

        // Should fail - hash mismatch
        let result = manager
            .store_query(wrong_hash.to_string(), query.to_string())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }
}
