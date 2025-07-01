//! Storage virtualization with transparent migration
//!
//! This module provides a virtualized storage layer that can transparently
//! route requests to different storage backends and migrate data between them.

use crate::model::{Triple, TriplePattern};
use crate::storage::StorageEngine;
// Note: tiered::TieredStorageEngine temporarily disabled due to dependency conflicts
use crate::OxirsError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Virtual storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualStorageConfig {
    /// Base path for virtual storage metadata
    pub path: PathBuf,
    /// Backend configurations
    pub backends: Vec<BackendConfig>,
    /// Routing policy
    pub routing: RoutingPolicy,
    /// Migration policy
    pub migration: MigrationPolicy,
    /// Enable transparent caching
    pub caching: bool,
    /// Cache size in MB
    pub cache_size_mb: usize,
}

impl Default for VirtualStorageConfig {
    fn default() -> Self {
        VirtualStorageConfig {
            path: PathBuf::from("/var/oxirs/virtual"),
            backends: vec![BackendConfig {
                name: "primary".to_string(),
                backend_type: BackendType::Tiered,
                config: serde_json::json!({
                    "enable_tiering": true,
                    "enable_columnar": false,
                    "enable_temporal": false,
                    "compression": {
                        "Zstd": { "level": 3 }
                    },
                    "tiers": {
                        "hot_tier": {
                            "max_size_mb": 1024,
                            "eviction_policy": "Lru",
                            "ttl_seconds": 3600
                        },
                        "warm_tier": {
                            "path": "/tmp/oxirs_virtual_warm",
                            "max_size_gb": 10,
                            "promotion_threshold": 10,
                            "demotion_threshold_days": 7
                        },
                        "cold_tier": {
                            "path": "/tmp/oxirs_virtual_cold",
                            "max_size_tb": 1,
                            "compression_level": 9,
                            "archive_threshold_days": 90
                        },
                        "archive_tier": {
                            "backend": { "Local": "/tmp/oxirs_virtual_archive" },
                            "retention_years": 7,
                            "immutable": true
                        }
                    },
                    "cache_size_mb": 512
                }),
                weight: 1.0,
                read_only: false,
            }],
            routing: RoutingPolicy::default(),
            migration: MigrationPolicy::default(),
            caching: true,
            cache_size_mb: 1024,
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend name
    pub name: String,
    /// Backend type
    pub backend_type: BackendType,
    /// Backend-specific configuration
    pub config: serde_json::Value,
    /// Routing weight (for load balancing)
    pub weight: f64,
    /// Read-only flag
    pub read_only: bool,
}

/// Storage backend type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendType {
    Tiered,
    Columnar,
    Immutable,
    Temporal,
    Remote { endpoint: String },
    Cloud { provider: CloudProvider },
}

/// Cloud storage provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    AWS { bucket: String, region: String },
    GCP { bucket: String, project: String },
    Azure { container: String, account: String },
}

/// Routing policy for virtual storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPolicy {
    /// Read routing strategy
    pub read_strategy: ReadStrategy,
    /// Write routing strategy
    pub write_strategy: WriteStrategy,
    /// Query routing hints
    pub query_hints: HashMap<String, String>,
    /// Predicate-based routing rules
    pub predicate_rules: HashMap<String, String>,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        RoutingPolicy {
            read_strategy: ReadStrategy::FirstAvailable,
            write_strategy: WriteStrategy::All,
            query_hints: HashMap::new(),
            predicate_rules: HashMap::new(),
        }
    }
}

/// Read routing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadStrategy {
    /// Use first available backend
    FirstAvailable,
    /// Round-robin across backends
    RoundRobin,
    /// Weighted random selection
    WeightedRandom,
    /// Query all and merge results
    Broadcast,
    /// Use specific backend based on pattern
    PatternBased,
}

/// Write routing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WriteStrategy {
    /// Write to all backends
    All,
    /// Write to primary only
    PrimaryOnly,
    /// Write to N backends
    Quorum { n: usize },
    /// Write based on data characteristics
    Selective,
}

/// Migration policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPolicy {
    /// Enable automatic migration
    pub auto_migrate: bool,
    /// Migration trigger
    pub trigger: MigrationTrigger,
    /// Batch size for migration
    pub batch_size: usize,
    /// Rate limit (triples per second)
    pub rate_limit: Option<usize>,
    /// Migration rules
    pub rules: Vec<MigrationRule>,
}

impl Default for MigrationPolicy {
    fn default() -> Self {
        MigrationPolicy {
            auto_migrate: false,
            trigger: MigrationTrigger::Manual,
            batch_size: 10000,
            rate_limit: Some(100000),
            rules: Vec::new(),
        }
    }
}

/// Migration trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationTrigger {
    /// Manual migration only
    Manual,
    /// Storage threshold (percentage)
    StorageThreshold(f64),
    /// Time-based (hours)
    Periodic(u32),
    /// Cost-based optimization
    CostOptimization,
    /// Performance-based
    Performance,
}

/// Migration rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationRule {
    /// Rule name
    pub name: String,
    /// Source backend pattern
    pub source: String,
    /// Target backend
    pub target: String,
    /// Selection criteria
    pub criteria: SelectionCriteria,
    /// Priority
    pub priority: u32,
}

/// Selection criteria for migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionCriteria {
    /// Age-based (days)
    Age(u32),
    /// Access frequency
    AccessFrequency { threshold: u32 },
    /// Size-based (MB)
    Size(usize),
    /// Predicate pattern
    PredicatePattern(String),
    /// Custom function
    Custom(String),
}

/// Virtual storage engine
pub struct VirtualStorage {
    config: VirtualStorageConfig,
    /// Storage backends
    backends: Arc<RwLock<HashMap<String, Arc<dyn StorageEngine>>>>,
    /// Routing state
    routing_state: Arc<RwLock<RoutingState>>,
    /// Migration state
    migration_state: Arc<RwLock<MigrationState>>,
    /// Cache
    cache: Arc<RwLock<StorageCache>>,
    /// Statistics
    stats: Arc<RwLock<VirtualStorageStats>>,
}

/// Routing state
struct RoutingState {
    /// Round-robin counter
    round_robin_counter: usize,
    /// Backend health status
    backend_health: HashMap<String, BackendHealth>,
    /// Active migrations
    active_migrations: Vec<String>,
}

/// Backend health status
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendHealth {
    /// Is backend healthy
    healthy: bool,
    /// Last check time (timestamp in seconds)
    last_check: u64,
    /// Consecutive failures
    failure_count: u32,
    /// Average response time
    avg_response_time_ms: f64,
}

/// Migration state
struct MigrationState {
    /// Active migrations
    active_migrations: HashMap<String, MigrationJob>,
    /// Migration history
    history: Vec<MigrationRecord>,
    /// Migration coordinator
    coordinator: Option<MigrationCoordinator>,
}

/// Migration job
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MigrationJob {
    /// Job ID
    id: String,
    /// Source backend
    source: String,
    /// Target backend
    target: String,
    /// Progress
    progress: MigrationProgress,
    /// Start time (timestamp in seconds)
    start_time: u64,
    /// Status
    status: MigrationStatus,
}

/// Migration progress
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MigrationProgress {
    /// Total triples to migrate
    total_triples: u64,
    /// Triples migrated
    migrated_triples: u64,
    /// Triples failed
    failed_triples: u64,
    /// Current batch
    current_batch: u64,
    /// Estimated completion time (seconds)
    eta: Option<u64>,
}

/// Migration status
#[derive(Debug, Clone, Serialize, Deserialize)]
enum MigrationStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed(String),
    Cancelled,
}

/// Migration record
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MigrationRecord {
    /// Job ID
    job_id: String,
    /// Source backend
    source: String,
    /// Target backend
    target: String,
    /// Start time
    start_time: chrono::DateTime<chrono::Utc>,
    /// End time
    end_time: chrono::DateTime<chrono::Utc>,
    /// Triples migrated
    triples_migrated: u64,
    /// Status
    status: MigrationStatus,
}

/// Migration coordinator
struct MigrationCoordinator {
    /// Active workers
    workers: Vec<tokio::task::JoinHandle<()>>,
    /// Control channel
    control_tx: tokio::sync::mpsc::Sender<MigrationControl>,
}

/// Migration control messages
#[derive(Debug)]
enum MigrationControl {
    Pause,
    Resume,
    Cancel,
    UpdateRateLimit(usize),
}

/// Storage cache
struct StorageCache {
    /// Triple cache
    triple_cache: lru::LruCache<u64, Triple>,
    /// Query result cache
    query_cache: lru::LruCache<String, Vec<Triple>>,
    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Virtual storage statistics
#[derive(Debug, Default)]
struct VirtualStorageStats {
    /// Total operations
    total_operations: u64,
    /// Operations by backend
    backend_operations: HashMap<String, u64>,
    /// Migration statistics
    migration_stats: MigrationStats,
    /// Performance metrics
    performance: PerformanceMetrics,
}

/// Migration statistics
#[derive(Debug, Default)]
struct MigrationStats {
    /// Total migrations
    total_migrations: u64,
    /// Successful migrations
    successful_migrations: u64,
    /// Failed migrations
    failed_migrations: u64,
    /// Total triples migrated
    total_triples_migrated: u64,
    /// Total migration time
    total_migration_time_sec: u64,
}

/// Performance metrics
#[derive(Debug, Default)]
struct PerformanceMetrics {
    /// Average read latency
    avg_read_latency_ms: f64,
    /// Average write latency
    avg_write_latency_ms: f64,
    /// Query throughput
    query_throughput_qps: f64,
    /// Write throughput
    write_throughput_tps: f64,
}

impl VirtualStorage {
    /// Create new virtual storage
    pub async fn new(config: VirtualStorageConfig) -> Result<Self, OxirsError> {
        std::fs::create_dir_all(&config.path)?;

        let cache_size = (config.cache_size_mb * 1024 * 1024) / 1000; // Approximate entries

        Ok(VirtualStorage {
            config: config.clone(),
            backends: Arc::new(RwLock::new(HashMap::new())),
            routing_state: Arc::new(RwLock::new(RoutingState {
                round_robin_counter: 0,
                backend_health: HashMap::new(),
                active_migrations: Vec::new(),
            })),
            migration_state: Arc::new(RwLock::new(MigrationState {
                active_migrations: HashMap::new(),
                history: Vec::new(),
                coordinator: None,
            })),
            cache: Arc::new(RwLock::new(StorageCache {
                triple_cache: lru::LruCache::new(
                    std::num::NonZeroUsize::new(cache_size)
                        .unwrap_or(std::num::NonZeroUsize::new(10000).unwrap()),
                ),
                query_cache: lru::LruCache::new(std::num::NonZeroUsize::new(1000).unwrap()),
                stats: CacheStats::default(),
            })),
            stats: Arc::new(RwLock::new(VirtualStorageStats::default())),
        })
    }

    /// Initialize backends
    pub async fn initialize_backends(&self) -> Result<(), OxirsError> {
        for backend_config in &self.config.backends {
            let backend = self.create_backend(backend_config).await?;
            let mut backends = self.backends.write().await;
            backends.insert(backend_config.name.clone(), backend);

            // Initialize health status
            let mut routing = self.routing_state.write().await;
            routing.backend_health.insert(
                backend_config.name.clone(),
                BackendHealth {
                    healthy: true,
                    last_check: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    failure_count: 0,
                    avg_response_time_ms: 0.0,
                },
            );
        }

        // Start health monitoring
        self.start_health_monitoring();

        Ok(())
    }

    /// Start a migration job
    pub async fn start_migration(
        &self,
        source: &str,
        target: &str,
        criteria: SelectionCriteria,
    ) -> Result<String, OxirsError> {
        let job_id = uuid::Uuid::new_v4().to_string();

        let job = MigrationJob {
            id: job_id.clone(),
            source: source.to_string(),
            target: target.to_string(),
            progress: MigrationProgress {
                total_triples: 0,
                migrated_triples: 0,
                failed_triples: 0,
                current_batch: 0,
                eta: None,
            },
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            status: MigrationStatus::Pending,
        };

        let mut migration_state = self.migration_state.write().await;
        migration_state
            .active_migrations
            .insert(job_id.clone(), job);

        // Start migration worker
        self.spawn_migration_worker(
            job_id.clone(),
            source.to_string(),
            target.to_string(),
            criteria,
        )
        .await?;

        Ok(job_id)
    }

    /// Get migration status
    pub async fn get_migration_status(&self, job_id: &str) -> Result<MigrationJob, OxirsError> {
        let migration_state = self.migration_state.read().await;
        migration_state
            .active_migrations
            .get(job_id)
            .cloned()
            .ok_or_else(|| OxirsError::Store(format!("Migration job not found: {}", job_id)))
    }

    /// Create backend instance
    async fn create_backend(
        &self,
        config: &BackendConfig,
    ) -> Result<Arc<dyn StorageEngine>, OxirsError> {
        match &config.backend_type {
            BackendType::Tiered => {
                // Tiered backend temporarily disabled due to dependency conflicts
                Err(OxirsError::Store(
                    "Tiered backend temporarily disabled due to RocksDB dependency conflicts".to_string(),
                ))
            }
            BackendType::Columnar => {
                // Create columnar backend
                Err(OxirsError::Store(
                    "Columnar backend not yet integrated".to_string(),
                ))
            }
            BackendType::Immutable => {
                // Create immutable backend
                Err(OxirsError::Store(
                    "Immutable backend not yet integrated".to_string(),
                ))
            }
            BackendType::Temporal => {
                // Create temporal backend
                Err(OxirsError::Store(
                    "Temporal backend not yet integrated".to_string(),
                ))
            }
            BackendType::Remote { endpoint } => {
                // Create remote backend proxy
                Err(OxirsError::Store(format!(
                    "Remote backend not implemented: {}",
                    endpoint
                )))
            }
            BackendType::Cloud { provider } => {
                // Create cloud backend
                Err(OxirsError::Store(
                    "Cloud backend not implemented".to_string(),
                ))
            }
        }
    }

    /// Route read operation
    async fn route_read(&self) -> Result<Vec<String>, OxirsError> {
        let routing_state = self.routing_state.read().await;
        let backends = self.backends.read().await;

        match self.config.routing.read_strategy {
            ReadStrategy::FirstAvailable => {
                // Return first healthy backend
                for (name, health) in &routing_state.backend_health {
                    if health.healthy && backends.contains_key(name) {
                        return Ok(vec![name.clone()]);
                    }
                }
                Err(OxirsError::Store(
                    "No healthy backends available".to_string(),
                ))
            }
            ReadStrategy::RoundRobin => {
                // Round-robin selection
                let healthy_backends: Vec<_> = routing_state
                    .backend_health
                    .iter()
                    .filter(|(name, health)| health.healthy && backends.contains_key(*name))
                    .map(|(name, _)| name.clone())
                    .collect();

                if healthy_backends.is_empty() {
                    return Err(OxirsError::Store(
                        "No healthy backends available".to_string(),
                    ));
                }

                let index = routing_state.round_robin_counter % healthy_backends.len();
                Ok(vec![healthy_backends[index].clone()])
            }
            ReadStrategy::Broadcast => {
                // Query all healthy backends
                Ok(routing_state
                    .backend_health
                    .iter()
                    .filter(|(name, health)| health.healthy && backends.contains_key(*name))
                    .map(|(name, _)| name.clone())
                    .collect())
            }
            _ => Ok(vec!["primary".to_string()]), // Default to primary
        }
    }

    /// Route write operation
    async fn route_write(&self) -> Result<Vec<String>, OxirsError> {
        let routing_state = self.routing_state.read().await;
        let backends = self.backends.read().await;

        match self.config.routing.write_strategy {
            WriteStrategy::All => {
                // Write to all writable backends
                Ok(self
                    .config
                    .backends
                    .iter()
                    .filter(|b| !b.read_only)
                    .filter(|b| {
                        routing_state
                            .backend_health
                            .get(&b.name)
                            .map(|h| h.healthy)
                            .unwrap_or(false)
                    })
                    .filter(|b| backends.contains_key(&b.name))
                    .map(|b| b.name.clone())
                    .collect())
            }
            WriteStrategy::PrimaryOnly => Ok(vec!["primary".to_string()]),
            WriteStrategy::Quorum { n } => {
                // Select N backends
                let writable: Vec<_> = self
                    .config
                    .backends
                    .iter()
                    .filter(|b| !b.read_only)
                    .filter(|b| {
                        routing_state
                            .backend_health
                            .get(&b.name)
                            .map(|h| h.healthy)
                            .unwrap_or(false)
                    })
                    .filter(|b| backends.contains_key(&b.name))
                    .take(n)
                    .map(|b| b.name.clone())
                    .collect();

                if writable.len() < n {
                    return Err(OxirsError::Store(format!(
                        "Not enough healthy backends for quorum: need {}, have {}",
                        n,
                        writable.len()
                    )));
                }

                Ok(writable)
            }
            WriteStrategy::Selective => {
                // Selective routing based on data characteristics
                Ok(vec!["primary".to_string()]) // Simplified
            }
        }
    }

    /// Start health monitoring
    fn start_health_monitoring(&self) {
        let backends = self.backends.clone();
        let routing_state = self.routing_state.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Check health of each backend
                let backend_list = backends.read().await;
                for (name, backend) in backend_list.iter() {
                    let start = std::time::Instant::now();
                    let health_check = backend.stats().await;
                    let elapsed = start.elapsed();

                    let mut routing = routing_state.write().await;
                    if let Some(health) = routing.backend_health.get_mut(name) {
                        health.last_check = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        health.avg_response_time_ms = elapsed.as_millis() as f64;

                        if health_check.is_ok() {
                            health.healthy = true;
                            health.failure_count = 0;
                        } else {
                            health.failure_count += 1;
                            if health.failure_count >= 3 {
                                health.healthy = false;
                            }
                        }
                    }
                }
            }
        });
    }

    /// Spawn migration worker
    async fn spawn_migration_worker(
        &self,
        job_id: String,
        source: String,
        target: String,
        criteria: SelectionCriteria,
    ) -> Result<(), OxirsError> {
        let backends = self.backends.clone();
        let migration_state = self.migration_state.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            // Migration logic would go here
            // This is a simplified placeholder
            let mut state = migration_state.write().await;
            if let Some(job) = state.active_migrations.get_mut(&job_id) {
                job.status = MigrationStatus::Running;
            }

            // Simulate migration
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            // Update status
            let mut state = migration_state.write().await;
            if let Some(job) = state.active_migrations.get_mut(&job_id) {
                job.status = MigrationStatus::Completed;
                job.progress.migrated_triples = 1000; // Simulated
            }
        });

        Ok(())
    }
}

#[async_trait]
impl StorageEngine for VirtualStorage {
    async fn init(&mut self, config: super::StorageConfig) -> Result<(), OxirsError> {
        self.initialize_backends().await
    }

    async fn store_triple(&self, triple: &Triple) -> Result<(), OxirsError> {
        let start = std::time::Instant::now();

        // Route write
        let target_backends = self.route_write().await?;
        let backends = self.backends.read().await;

        let mut errors = Vec::new();
        for backend_name in &target_backends {
            if let Some(backend) = backends.get(backend_name) {
                if let Err(e) = backend.store_triple(triple).await {
                    errors.push((backend_name.clone(), e));
                }
            }
        }

        // Update cache
        if self.config.caching {
            let mut cache = self.cache.write().await;
            let hash = self.hash_triple(triple);
            cache.triple_cache.put(hash, triple.clone());
        }

        // Update statistics
        let elapsed = start.elapsed();
        let mut stats = self.stats.write().await;
        stats.total_operations += 1;
        stats.performance.avg_write_latency_ms = (stats.performance.avg_write_latency_ms
            * (stats.total_operations - 1) as f64
            + elapsed.as_millis() as f64)
            / stats.total_operations as f64;

        // Handle errors based on write strategy
        match self.config.routing.write_strategy {
            WriteStrategy::All => {
                if !errors.is_empty() {
                    return Err(OxirsError::Store(format!(
                        "Failed to write to backends: {:?}",
                        errors
                    )));
                }
            }
            WriteStrategy::Quorum { n } => {
                if target_backends.len() - errors.len() < n {
                    return Err(OxirsError::Store(format!(
                        "Quorum write failed: {} successes, needed {}",
                        target_backends.len() - errors.len(),
                        n
                    )));
                }
            }
            _ => {}
        }

        Ok(())
    }

    async fn store_triples(&self, triples: &[Triple]) -> Result<(), OxirsError> {
        // Batch operation - could be optimized
        for triple in triples {
            self.store_triple(triple).await?;
        }
        Ok(())
    }

    async fn query_triples(&self, pattern: &TriplePattern) -> Result<Vec<Triple>, OxirsError> {
        let start = std::time::Instant::now();

        // Check cache first
        if self.config.caching {
            let pattern_key = format!("{:?}", pattern);
            let mut cache = self.cache.write().await;
            if let Some(cached) = cache.query_cache.get(&pattern_key).cloned() {
                cache.stats.hits += 1;
                return Ok(cached);
            }
            cache.stats.misses += 1;
        }

        // Route read
        let target_backends = self.route_read().await?;
        let backends = self.backends.read().await;

        let mut all_results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for backend_name in &target_backends {
            if let Some(backend) = backends.get(backend_name) {
                match backend.query_triples(pattern).await {
                    Ok(results) => {
                        for triple in results {
                            let hash = self.hash_triple(&triple);
                            if seen.insert(hash) {
                                all_results.push(triple);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Query failed on backend {}: {}", backend_name, e);
                    }
                }
            }
        }

        // Update cache
        if self.config.caching && !all_results.is_empty() {
            let pattern_key = format!("{:?}", pattern);
            let mut cache = self.cache.write().await;
            cache.query_cache.put(pattern_key, all_results.clone());
        }

        // Update statistics
        let elapsed = start.elapsed();
        let mut stats = self.stats.write().await;
        stats.total_operations += 1;
        stats.performance.avg_read_latency_ms = (stats.performance.avg_read_latency_ms
            * (stats.total_operations - 1) as f64
            + elapsed.as_millis() as f64)
            / stats.total_operations as f64;

        Ok(all_results)
    }

    async fn delete_triples(&self, pattern: &TriplePattern) -> Result<usize, OxirsError> {
        // Route to all writable backends
        let target_backends = self.route_write().await?;
        let backends = self.backends.read().await;

        let mut total_deleted = 0;
        for backend_name in &target_backends {
            if let Some(backend) = backends.get(backend_name) {
                match backend.delete_triples(pattern).await {
                    Ok(count) => total_deleted = total_deleted.max(count),
                    Err(e) => {
                        tracing::warn!("Delete failed on backend {}: {}", backend_name, e);
                    }
                }
            }
        }

        // Invalidate cache
        if self.config.caching {
            let mut cache = self.cache.write().await;
            cache.query_cache.clear();
            // Could be more selective about cache invalidation
        }

        Ok(total_deleted)
    }

    async fn stats(&self) -> Result<super::StorageStats, OxirsError> {
        let backends = self.backends.read().await;
        let stats = self.stats.read().await;

        // Aggregate stats from all backends
        let mut total_triples = 0u64;
        let mut total_size = 0u64;

        for (_, backend) in backends.iter() {
            if let Ok(backend_stats) = backend.stats().await {
                total_triples += backend_stats.total_triples;
                total_size += backend_stats.total_size_bytes;
            }
        }

        Ok(super::StorageStats {
            total_triples,
            total_size_bytes: total_size,
            tier_stats: super::TierStats {
                hot: super::TierStat {
                    triple_count: 0,
                    size_bytes: 0,
                    hit_rate: 0.0,
                    avg_access_time_us: 0,
                },
                warm: super::TierStat {
                    triple_count: 0,
                    size_bytes: 0,
                    hit_rate: 0.0,
                    avg_access_time_us: 0,
                },
                cold: super::TierStat {
                    triple_count: 0,
                    size_bytes: 0,
                    hit_rate: 0.0,
                    avg_access_time_us: 0,
                },
                archive: super::TierStat {
                    triple_count: 0,
                    size_bytes: 0,
                    hit_rate: 0.0,
                    avg_access_time_us: 0,
                },
            },
            compression_ratio: 1.0,
            query_metrics: super::QueryMetrics {
                avg_query_time_ms: stats.performance.avg_read_latency_ms,
                p99_query_time_ms: stats.performance.avg_read_latency_ms * 2.0, // Estimate
                qps: stats.performance.query_throughput_qps,
                cache_hit_rate: if self.config.caching {
                    let cache = self.cache.read().await;
                    let total = cache.stats.hits + cache.stats.misses;
                    if total > 0 {
                        (cache.stats.hits as f64 / total as f64) * 100.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                },
            },
        })
    }

    async fn optimize(&self) -> Result<(), OxirsError> {
        let backends = self.backends.read().await;

        // Optimize all backends
        for (name, backend) in backends.iter() {
            if let Err(e) = backend.optimize().await {
                tracing::warn!("Optimization failed on backend {}: {}", name, e);
            }
        }

        // Clear cache to force refresh
        if self.config.caching {
            let mut cache = self.cache.write().await;
            cache.query_cache.clear();
        }

        Ok(())
    }

    async fn backup(&self, path: &Path) -> Result<(), OxirsError> {
        std::fs::create_dir_all(path)?;

        let backends = self.backends.read().await;

        // Backup each backend
        for (name, backend) in backends.iter() {
            let backend_path = path.join(name);
            backend.backup(&backend_path).await?;
        }

        // Save virtual storage metadata
        let metadata = VirtualStorageMetadata {
            config: self.config.clone(),
            migration_history: {
                let state = self.migration_state.read().await;
                state.history.clone()
            },
        };

        let metadata_path = path.join("virtual_metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| OxirsError::Serialize(e.to_string()))?;
        std::fs::write(metadata_path, metadata_json)?;

        Ok(())
    }

    async fn restore(&self, path: &Path) -> Result<(), OxirsError> {
        // Load metadata
        let metadata_path = path.join("virtual_metadata.json");
        let metadata_json = std::fs::read_to_string(metadata_path)?;
        let metadata: VirtualStorageMetadata =
            serde_json::from_str(&metadata_json).map_err(|e| OxirsError::Parse(e.to_string()))?;

        let backends = self.backends.read().await;

        // Restore each backend
        for (name, backend) in backends.iter() {
            let backend_path = path.join(name);
            if backend_path.exists() {
                backend.restore(&backend_path).await?;
            }
        }

        // Restore migration history
        let mut state = self.migration_state.write().await;
        state.history = metadata.migration_history;

        Ok(())
    }
}

/// Virtual storage metadata
#[derive(Debug, Serialize, Deserialize)]
struct VirtualStorageMetadata {
    config: VirtualStorageConfig,
    migration_history: Vec<MigrationRecord>,
}

impl VirtualStorage {
    /// Hash a triple for deduplication
    fn hash_triple(&self, triple: &Triple) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        triple.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[tokio::test]
    async fn test_virtual_storage() {
        let test_dir = format!(
            "/tmp/oxirs_virtual_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        let mut config = VirtualStorageConfig {
            path: PathBuf::from(&test_dir),
            ..Default::default()
        };

        // Update backend config with test-specific paths
        if let Some(backend) = config.backends.get_mut(0) {
            backend.config = serde_json::json!({
                "enable_tiering": true,
                "enable_columnar": false,
                "enable_temporal": false,
                "compression": {
                    "Zstd": { "level": 3 }
                },
                "tiers": {
                    "hot_tier": {
                        "max_size_mb": 1024,
                        "eviction_policy": "Lru",
                        "ttl_seconds": 3600
                    },
                    "warm_tier": {
                        "path": format!("{}/warm", test_dir),
                        "max_size_gb": 10,
                        "promotion_threshold": 10,
                        "demotion_threshold_days": 7
                    },
                    "cold_tier": {
                        "path": format!("{}/cold", test_dir),
                        "max_size_tb": 1,
                        "compression_level": 9,
                        "archive_threshold_days": 90
                    },
                    "archive_tier": {
                        "backend": { "Local": format!("{}/archive", test_dir) },
                        "retention_years": 7,
                        "immutable": true
                    }
                },
                "cache_size_mb": 512
            });
        }

        let storage = VirtualStorage::new(config).await.unwrap();
        storage.initialize_backends().await.unwrap();

        // Create test triple
        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            crate::model::Object::Literal(Literal::new("test")),
        );

        // Store triple
        storage.store_triple(&triple).await.unwrap();

        // Query triple
        let pattern = TriplePattern::new(None, None, None);
        let results = storage.query_triples(&pattern).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], triple);
    }
}
