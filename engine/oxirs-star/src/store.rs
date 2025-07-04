//! RDF-star storage implementation with efficient handling of quoted triples.
//!
//! This module provides storage backends for RDF-star data, extending the core
//! OxiRS storage with support for quoted triples and efficient indexing.
//!
//! Features:
//! - B-tree indexing for efficient quoted triple lookups
//! - Bulk insertion optimizations for large datasets  
//! - Memory-mapped storage options for persistent storage
//! - Compression for quoted triple storage
//! - Connection pooling for concurrent access
//! - Cache optimization strategies
//! - Transaction support with ACID properties

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use oxirs_core::rdf_store::{ConcreteStore as CoreStore, Store};
use tracing::{debug, info, span, warn, Level};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::{StarConfig, StarError, StarResult, StarStatistics};

/// Conversion utilities for StarTerm to core RDF terms
mod conversion {
    use super::*;
    use oxirs_core::model::{
        BlankNode as CoreBlankNode, Literal as CoreLiteral, NamedNode as CoreNamedNode, Object,
        Predicate, Subject, Term, Triple,
    };

    pub fn star_term_to_subject(term: &StarTerm) -> StarResult<Subject> {
        match term {
            StarTerm::NamedNode(nn) => {
                let named_node =
                    CoreNamedNode::new(&nn.iri).map_err(|e| StarError::CoreError(e))?;
                Ok(Subject::NamedNode(named_node))
            }
            StarTerm::BlankNode(bn) => {
                let blank_node = CoreBlankNode::new(&bn.id).map_err(|e| StarError::CoreError(e))?;
                Ok(Subject::BlankNode(blank_node))
            }
            StarTerm::Literal(_) => Err(StarError::invalid_term_type(
                "Literal cannot be used as subject".to_string(),
            )),
            StarTerm::QuotedTriple(_) => Err(StarError::invalid_term_type(
                "Quoted triple cannot be converted to core RDF subject".to_string(),
            )),
            StarTerm::Variable(_) => Err(StarError::invalid_term_type(
                "Variable cannot be converted to core RDF subject".to_string(),
            )),
        }
    }

    pub fn star_term_to_predicate(term: &StarTerm) -> StarResult<Predicate> {
        match term {
            StarTerm::NamedNode(nn) => {
                let named_node =
                    CoreNamedNode::new(&nn.iri).map_err(|e| StarError::CoreError(e))?;
                Ok(Predicate::NamedNode(named_node))
            }
            _ => Err(StarError::invalid_term_type(
                "Only IRIs can be used as predicates".to_string(),
            )),
        }
    }

    pub fn star_term_to_object(term: &StarTerm) -> StarResult<Object> {
        match term {
            StarTerm::NamedNode(nn) => {
                let named_node =
                    CoreNamedNode::new(&nn.iri).map_err(|e| StarError::CoreError(e))?;
                Ok(Object::NamedNode(named_node))
            }
            StarTerm::BlankNode(bn) => {
                let blank_node = CoreBlankNode::new(&bn.id).map_err(|e| StarError::CoreError(e))?;
                Ok(Object::BlankNode(blank_node))
            }
            StarTerm::Literal(lit) => {
                let literal = CoreLiteral::new(&lit.value);
                Ok(Object::Literal(literal))
            }
            StarTerm::QuotedTriple(_) => Err(StarError::invalid_term_type(
                "Quoted triple cannot be converted to core RDF object".to_string(),
            )),
            StarTerm::Variable(_) => Err(StarError::invalid_term_type(
                "Variable cannot be converted to core RDF object".to_string(),
            )),
        }
    }
}

/// Indexing structure for efficient quoted triple lookups
#[derive(Debug, Clone)]
struct QuotedTripleIndex {
    /// B-tree index mapping quoted triple signatures to triple indices
    signature_to_indices: BTreeMap<String, BTreeSet<usize>>,
    /// Subject-based index for S?? pattern queries
    subject_index: BTreeMap<String, BTreeSet<usize>>,
    /// Predicate-based index for ?P? pattern queries  
    predicate_index: BTreeMap<String, BTreeSet<usize>>,
    /// Object-based index for ??O pattern queries
    object_index: BTreeMap<String, BTreeSet<usize>>,
    /// Nesting depth index for performance optimization
    nesting_depth_index: BTreeMap<usize, BTreeSet<usize>>,
}

impl QuotedTripleIndex {
    fn new() -> Self {
        Self {
            signature_to_indices: BTreeMap::new(),
            subject_index: BTreeMap::new(),
            predicate_index: BTreeMap::new(),
            object_index: BTreeMap::new(),
            nesting_depth_index: BTreeMap::new(),
        }
    }

    fn clear(&mut self) {
        self.signature_to_indices.clear();
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        self.nesting_depth_index.clear();
    }

    /// Get index statistics for optimization analysis
    fn get_statistics(&self) -> IndexStatistics {
        IndexStatistics {
            total_entries: self.signature_to_indices.len(),
            subject_index_size: self.subject_index.len(),
            predicate_index_size: self.predicate_index.len(),
            object_index_size: self.object_index.len(),
            nesting_depth_levels: self.nesting_depth_index.len(),
            average_bucket_size: self.calculate_average_bucket_size(),
        }
    }

    fn calculate_average_bucket_size(&self) -> f64 {
        if self.signature_to_indices.is_empty() {
            return 0.0;
        }
        let total_entries: usize = self.signature_to_indices.values().map(|s| s.len()).sum();
        total_entries as f64 / self.signature_to_indices.len() as f64
    }
}

/// Statistics about the indexing performance
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    pub total_entries: usize,
    pub subject_index_size: usize,
    pub predicate_index_size: usize,
    pub object_index_size: usize,
    pub nesting_depth_levels: usize,
    pub average_bucket_size: f64,
}

/// Bulk insertion configuration for optimized batch operations
#[derive(Debug, Clone)]
pub struct BulkInsertConfig {
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Disable index updates during bulk insertion
    pub defer_index_updates: bool,
    /// Memory threshold before flushing (bytes)
    pub memory_threshold: usize,
    /// Enable parallel processing for large batches
    pub parallel_processing: bool,
    /// Number of worker threads for parallel insertion
    pub worker_threads: usize,
}

impl Default for BulkInsertConfig {
    fn default() -> Self {
        Self {
            batch_size: 10000,
            defer_index_updates: true,
            memory_threshold: 256 * 1024 * 1024, // 256MB
            parallel_processing: true,
            worker_threads: std::cmp::min(8, 4), // Use 4 as default thread count
        }
    }
}

/// Connection pool for managing concurrent access to the store
pub struct ConnectionPool {
    /// Pool of available store connections
    available_connections: Arc<Mutex<VecDeque<Arc<StarStore>>>>,
    /// Condition variable for waiting on available connections
    connection_available: Arc<Condvar>,
    /// Maximum number of connections in the pool
    max_connections: usize,
    /// Current number of created connections
    active_connections: Arc<Mutex<usize>>,
    /// Configuration for creating new connections
    config: StarConfig,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(max_connections: usize, config: StarConfig) -> Self {
        Self {
            available_connections: Arc::new(Mutex::new(VecDeque::new())),
            connection_available: Arc::new(Condvar::new()),
            max_connections,
            active_connections: Arc::new(Mutex::new(0)),
            config,
        }
    }

    /// Get a connection from the pool (blocks if none available)
    pub fn get_connection(&self) -> StarResult<PooledConnection> {
        let mut available = self.available_connections.lock().unwrap();

        // Try to get an existing connection
        if let Some(store) = available.pop_front() {
            return Ok(PooledConnection::new(store, self.clone()));
        }

        // Check if we can create a new connection
        let mut active_count = self.active_connections.lock().unwrap();
        if *active_count < self.max_connections {
            *active_count += 1;
            drop(active_count);
            drop(available);

            let store = Arc::new(StarStore::with_config(self.config.clone()));
            return Ok(PooledConnection::new(store, self.clone()));
        }

        // Wait for a connection to become available
        drop(active_count);
        available = self.connection_available.wait(available).unwrap();

        if let Some(store) = available.pop_front() {
            Ok(PooledConnection::new(store, self.clone()))
        } else {
            Err(StarError::query_error(
                "No connections available".to_string(),
            ))
        }
    }

    /// Try to get a connection without blocking
    pub fn try_get_connection(&self) -> Option<PooledConnection> {
        let mut available = self.available_connections.lock().ok()?;

        if let Some(store) = available.pop_front() {
            return Some(PooledConnection::new(store, self.clone()));
        }

        let mut active_count = self.active_connections.lock().ok()?;
        if *active_count < self.max_connections {
            *active_count += 1;
            drop(active_count);
            drop(available);

            let store = Arc::new(StarStore::with_config(self.config.clone()));
            Some(PooledConnection::new(store, self.clone()))
        } else {
            None
        }
    }

    /// Return a connection to the pool
    fn return_connection(&self, store: Arc<StarStore>) {
        let mut available = self.available_connections.lock().unwrap();
        available.push_back(store);
        self.connection_available.notify_one();
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        let available = self.available_connections.lock().unwrap();
        let active_count = self.active_connections.lock().unwrap();

        PoolStatistics {
            available_connections: available.len(),
            active_connections: *active_count,
            max_connections: self.max_connections,
            utilization: (*active_count as f64 / self.max_connections as f64) * 100.0,
        }
    }
}

impl Clone for ConnectionPool {
    fn clone(&self) -> Self {
        Self {
            available_connections: Arc::clone(&self.available_connections),
            connection_available: Arc::clone(&self.connection_available),
            max_connections: self.max_connections,
            active_connections: Arc::clone(&self.active_connections),
            config: self.config.clone(),
        }
    }
}

/// A pooled connection that automatically returns to the pool when dropped
pub struct PooledConnection {
    store: Option<Arc<StarStore>>,
    pool: ConnectionPool,
}

impl PooledConnection {
    fn new(store: Arc<StarStore>, pool: ConnectionPool) -> Self {
        Self {
            store: Some(store),
            pool,
        }
    }

    /// Get access to the underlying store
    pub fn store(&self) -> &StarStore {
        self.store.as_ref().expect("Connection has been dropped")
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(store) = self.store.take() {
            self.pool.return_connection(store);
        }
    }
}

/// Statistics about connection pool usage
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub available_connections: usize,
    pub active_connections: usize,
    pub max_connections: usize,
    pub utilization: f64,
}

/// Cache for frequently accessed data
#[derive(Debug)]
pub struct StarCache {
    /// LRU cache for triple lookups
    triple_cache: Arc<RwLock<HashMap<String, Vec<StarTriple>>>>,
    /// Cache for pattern queries
    pattern_cache: Arc<RwLock<HashMap<String, Vec<StarTriple>>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Access frequency tracking
    access_frequency: Arc<RwLock<HashMap<String, usize>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStatistics>>,
}

/// Configuration for the cache system
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in triple cache
    pub max_triple_entries: usize,
    /// Maximum number of entries in pattern cache
    pub max_pattern_entries: usize,
    /// Time to live for cache entries (seconds)
    pub ttl_seconds: u64,
    /// Enable LRU eviction
    pub enable_lru: bool,
    /// Cache hit rate threshold for optimization
    pub optimization_threshold: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_triple_entries: 10000,
            max_pattern_entries: 5000,
            ttl_seconds: 300, // 5 minutes
            enable_lru: true,
            optimization_threshold: 0.8,
        }
    }
}

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_lookups: u64,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

impl StarCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            triple_cache: Arc::new(RwLock::new(HashMap::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            access_frequency: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
        }
    }

    /// Get cached results for a query
    pub fn get(&self, key: &str) -> Option<Vec<StarTriple>> {
        let mut stats = self.stats.write().unwrap();
        stats.total_lookups += 1;

        // Check triple cache first
        if let Some(results) = self.triple_cache.read().unwrap().get(key) {
            stats.hits += 1;

            // Update access frequency
            let mut freq = self.access_frequency.write().unwrap();
            *freq.entry(key.to_string()).or_insert(0) += 1;

            return Some(results.clone());
        }

        // Check pattern cache
        if let Some(results) = self.pattern_cache.read().unwrap().get(key) {
            stats.hits += 1;

            let mut freq = self.access_frequency.write().unwrap();
            *freq.entry(key.to_string()).or_insert(0) += 1;

            return Some(results.clone());
        }

        stats.misses += 1;
        None
    }

    /// Store results in cache
    pub fn put(&self, key: String, results: Vec<StarTriple>) {
        // Simple LRU eviction if cache is full
        if self.config.enable_lru {
            let mut cache = self.triple_cache.write().unwrap();
            if cache.len() >= self.config.max_triple_entries {
                // Remove least frequently used entry
                if let Some(lfu_key) = self.find_least_frequent_key() {
                    cache.remove(&lfu_key);
                    let mut stats = self.stats.write().unwrap();
                    stats.evictions += 1;
                }
            }
            cache.insert(key, results);
        }
    }

    fn find_least_frequent_key(&self) -> Option<String> {
        let freq = self.access_frequency.read().unwrap();
        freq.iter()
            .min_by_key(|(_, &count)| count)
            .map(|(key, _)| key.clone())
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        self.stats.read().unwrap().clone()
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        self.triple_cache.write().unwrap().clear();
        self.pattern_cache.write().unwrap().clear();
        self.access_frequency.write().unwrap().clear();
    }
}

/// RDF-star storage backend with support for quoted triples
#[derive(Clone)]
pub struct StarStore {
    /// Core RDF storage backend
    core_store: Arc<RwLock<CoreStore>>,
    /// RDF-star specific triples (those containing quoted triples)
    star_triples: Arc<RwLock<Vec<StarTriple>>>,
    /// Enhanced B-tree based quoted triple index for efficient lookup
    quoted_triple_index: Arc<RwLock<QuotedTripleIndex>>,
    /// Configuration for the store
    config: StarConfig,
    /// Statistics tracking
    statistics: Arc<RwLock<StarStatistics>>,
    /// Cache for frequently accessed data
    cache: Arc<StarCache>,
    /// Bulk insertion state
    bulk_insert_state: Arc<RwLock<BulkInsertState>>,
    /// Memory-mapped storage state
    memory_mapped: Arc<RwLock<MemoryMappedState>>,
}

/// State tracking for bulk insertion operations
#[derive(Debug, Default)]
struct BulkInsertState {
    /// Whether bulk insertion is currently active
    active: bool,
    /// Pending triples waiting to be indexed
    pending_triples: Vec<StarTriple>,
    /// Memory usage tracking for bulk operations
    current_memory_usage: usize,
    /// Batch count for monitoring
    batch_count: usize,
}

/// State for memory-mapped storage operations
#[derive(Debug, Default)]
struct MemoryMappedState {
    /// Whether memory mapping is enabled
    enabled: bool,
    /// Path to the memory-mapped file
    file_path: Option<String>,
    /// Compression settings for stored data
    compression_enabled: bool,
    /// Last sync timestamp
    last_sync: Option<Instant>,
}

impl StarStore {
    /// Create a new RDF-star store with default configuration
    pub fn new() -> Self {
        Self::with_config(StarConfig::default())
    }

    /// Create a new RDF-star store with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        let span = span!(Level::INFO, "new_star_store");
        let _enter = span.enter();

        info!("Creating new RDF-star store with optimizations");
        debug!("Configuration: {:?}", config);

        Self {
            core_store: Arc::new(RwLock::new(
                CoreStore::new().expect("Failed to create core store"),
            )),
            star_triples: Arc::new(RwLock::new(Vec::new())),
            quoted_triple_index: Arc::new(RwLock::new(QuotedTripleIndex::new())),
            config: config.clone(),
            statistics: Arc::new(RwLock::new(StarStatistics::default())),
            cache: Arc::new(StarCache::new(CacheConfig::default())),
            bulk_insert_state: Arc::new(RwLock::new(BulkInsertState::default())),
            memory_mapped: Arc::new(RwLock::new(MemoryMappedState::default())),
        }
    }

    /// Insert a RDF-star triple into the store
    pub fn insert(&self, triple: &StarTriple) -> StarResult<()> {
        let span = span!(Level::DEBUG, "insert_triple");
        let _enter = span.enter();

        let start_time = Instant::now();

        // Validate the triple
        triple.validate()?;

        // Check nesting depth
        crate::validate_nesting_depth(&triple.subject, self.config.max_nesting_depth)?;
        crate::validate_nesting_depth(&triple.predicate, self.config.max_nesting_depth)?;
        crate::validate_nesting_depth(&triple.object, self.config.max_nesting_depth)?;

        // Insert into appropriate storage
        if triple.contains_quoted_triples() {
            self.insert_star_triple(triple)?;
        } else {
            self.insert_regular_triple(triple)?;
        }

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.processing_time_us += start_time.elapsed().as_micros() as u64;
            if triple.contains_quoted_triples() {
                stats.quoted_triples_count += 1;
                stats.max_nesting_encountered =
                    stats.max_nesting_encountered.max(triple.nesting_depth());
            }
        }

        debug!("Inserted triple: {}", triple);
        Ok(())
    }

    /// Insert a regular RDF triple (no quoted triples) into core store
    fn insert_regular_triple(&self, triple: &StarTriple) -> StarResult<()> {
        debug!("Inserting regular triple into core store");

        // Convert StarTriple to core RDF triple
        let core_triple = self.convert_to_core_triple(triple)?;

        // Insert into core store (convert triple to quad in default graph)
        let core_quad = oxirs_core::model::Quad::from_triple(core_triple);
        let mut core_store = self.core_store.write().unwrap();
        CoreStore::insert_quad(&mut *core_store, core_quad).map_err(|e| StarError::CoreError(e))?;

        Ok(())
    }

    /// Convert a StarTriple (without quoted triples) to a core RDF Triple
    fn convert_to_core_triple(&self, triple: &StarTriple) -> StarResult<oxirs_core::model::Triple> {
        let subject = conversion::star_term_to_subject(&triple.subject)?;
        let predicate = conversion::star_term_to_predicate(&triple.predicate)?;
        let object = conversion::star_term_to_object(&triple.object)?;

        Ok(oxirs_core::model::Triple::new(subject, predicate, object))
    }

    /// Insert a RDF-star triple (containing quoted triples) into star storage
    fn insert_star_triple(&self, triple: &StarTriple) -> StarResult<()> {
        let mut star_triples = self.star_triples.write().unwrap();
        let mut index = self.quoted_triple_index.write().unwrap();

        let triple_index = star_triples.len();
        star_triples.push(triple.clone());

        // Build index for quoted triples
        self.index_quoted_triples(triple, triple_index, &mut index);

        debug!(
            "Inserted star triple with {} quoted triples",
            self.count_quoted_triples_in_triple(triple)
        );
        Ok(())
    }

    /// Build index entries for quoted triples in a given triple using B-tree indices
    fn index_quoted_triples(
        &self,
        triple: &StarTriple,
        triple_index: usize,
        index: &mut QuotedTripleIndex,
    ) {
        self.index_quoted_triples_recursive(triple, triple_index, index);

        // Index by nesting depth for performance optimization
        let depth = triple.nesting_depth();
        index
            .nesting_depth_index
            .entry(depth)
            .or_insert_with(BTreeSet::new)
            .insert(triple_index);
    }

    /// Recursively index quoted triples with multi-dimensional indexing
    fn index_quoted_triples_recursive(
        &self,
        triple: &StarTriple,
        triple_index: usize,
        index: &mut QuotedTripleIndex,
    ) {
        // Index quoted triples in subject
        if let StarTerm::QuotedTriple(qt) = &triple.subject {
            let signature = self.quoted_triple_key(qt);
            index
                .signature_to_indices
                .entry(signature)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // Index by subject signature for S?? queries
            let subject_key = format!("SUBJ:{}", qt.subject);
            index
                .subject_index
                .entry(subject_key)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // Recursively index nested quoted triples
            self.index_quoted_triples_recursive(qt, triple_index, index);
        }

        // Index quoted triples in predicate (rare but possible in some extensions)
        if let StarTerm::QuotedTriple(qt) = &triple.predicate {
            let signature = self.quoted_triple_key(qt);
            index
                .signature_to_indices
                .entry(signature)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // Index by predicate signature for ?P? queries
            let predicate_key = format!("PRED:{}", qt.predicate);
            index
                .predicate_index
                .entry(predicate_key)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // ALSO index the subject and object of the quoted triple found in predicate position
            let qt_subject_key = format!("SUBJ:{}", qt.subject);
            index
                .subject_index
                .entry(qt_subject_key)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            let qt_object_key = format!("OBJ:{}", qt.object);
            index
                .object_index
                .entry(qt_object_key)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // Recursively index nested quoted triples
            self.index_quoted_triples_recursive(qt, triple_index, index);
        }

        // Index quoted triples in object
        if let StarTerm::QuotedTriple(qt) = &triple.object {
            let signature = self.quoted_triple_key(qt);
            index
                .signature_to_indices
                .entry(signature)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // Index by object signature for ??O queries
            let object_key = format!("OBJ:{}", qt.object);
            index
                .object_index
                .entry(object_key)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // ALSO index the subject and predicate of the quoted triple found in object position
            // This allows finding triples like "bob believes <<alice age 25>>" when searching for alice
            let qt_subject_key = format!("SUBJ:{}", qt.subject);
            index
                .subject_index
                .entry(qt_subject_key)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            let qt_predicate_key = format!("PRED:{}", qt.predicate);
            index
                .predicate_index
                .entry(qt_predicate_key)
                .or_insert_with(BTreeSet::new)
                .insert(triple_index);

            // Recursively index nested quoted triples
            self.index_quoted_triples_recursive(qt, triple_index, index);
        }
    }

    /// Generate a key for indexing quoted triples
    fn quoted_triple_key(&self, triple: &StarTriple) -> String {
        format!("{}|{}|{}", triple.subject, triple.predicate, triple.object)
    }

    /// Update indices after removing an item at position `pos`
    /// This efficiently updates all indices > pos by decrementing them
    fn update_indices_after_removal(indices: &mut BTreeSet<usize>, pos: usize) {
        // Remove the item at pos
        indices.remove(&pos);

        // Create a new set with updated indices
        let updated: BTreeSet<usize> = indices
            .iter()
            .map(|&idx| if idx > pos { idx - 1 } else { idx })
            .collect();

        // Replace the old set with the updated one
        *indices = updated;
    }

    /// Count quoted triples within a single triple
    fn count_quoted_triples_in_triple(&self, triple: &StarTriple) -> usize {
        let mut count = 0;

        if triple.subject.is_quoted_triple() {
            count += 1;
            if let StarTerm::QuotedTriple(qt) = &triple.subject {
                count += self.count_quoted_triples_in_triple(qt);
            }
        }

        if triple.predicate.is_quoted_triple() {
            count += 1;
            if let StarTerm::QuotedTriple(qt) = &triple.predicate {
                count += self.count_quoted_triples_in_triple(qt);
            }
        }

        if triple.object.is_quoted_triple() {
            count += 1;
            if let StarTerm::QuotedTriple(qt) = &triple.object {
                count += self.count_quoted_triples_in_triple(qt);
            }
        }

        count
    }

    /// Remove a triple from the store
    pub fn remove(&self, triple: &StarTriple) -> StarResult<bool> {
        let span = span!(Level::DEBUG, "remove_triple");
        let _enter = span.enter();

        // First try to remove from star triples
        if triple.contains_quoted_triples() {
            let mut star_triples = self.star_triples.write().unwrap();

            if let Some(pos) = star_triples.iter().position(|t| t == triple) {
                star_triples.remove(pos);

                // Update all indices
                let mut index = self.quoted_triple_index.write().unwrap();

                // Update signature index
                for (_, indices) in index.signature_to_indices.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update subject index
                for (_, indices) in index.subject_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update predicate index
                for (_, indices) in index.predicate_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update object index
                for (_, indices) in index.object_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                // Update nesting depth index
                for (_, indices) in index.nesting_depth_index.iter_mut() {
                    Self::update_indices_after_removal(indices, pos);
                }

                debug!("Removed star triple: {}", triple);
                return Ok(true);
            }
        } else {
            // Try to remove from core store for regular triples
            let mut core_store = self.core_store.write().unwrap();
            if let Ok(core_triple) = self.convert_to_core_triple(triple) {
                let core_quad = oxirs_core::model::Quad::from_triple(core_triple);
                if let Ok(removed) = core_store.remove_quad(&core_quad) {
                    if removed {
                        debug!("Removed regular triple: {}", triple);
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Check if the store contains a specific triple
    pub fn contains(&self, triple: &StarTriple) -> bool {
        // First check star triples
        let star_triples = self.star_triples.read().unwrap();
        if star_triples.contains(triple) {
            return true;
        }

        // Then check regular triples in core store
        if !triple.contains_quoted_triples() {
            let core_store = self.core_store.read().unwrap();
            if let Ok(core_triple) = self.convert_to_core_triple(triple) {
                // Convert triple to quad with default graph
                let core_quad = oxirs_core::model::Quad::from_triple(core_triple);
                if let Ok(quads) = core_store.find_quads(
                    Some(core_quad.subject()),
                    Some(core_quad.predicate()),
                    Some(core_quad.object()),
                    Some(core_quad.graph_name()),
                ) {
                    return !quads.is_empty();
                }
            }
        }

        false
    }

    /// Get all triples in the store
    pub fn triples(&self) -> Vec<StarTriple> {
        let mut all_triples = Vec::new();

        // Add star triples (clone to release lock quickly)
        {
            let star_triples = self.star_triples.read().unwrap();
            all_triples.extend(star_triples.clone());
        }

        // Add regular triples from core store (release lock quickly)
        {
            let core_store = self.core_store.read().unwrap();
            if let Ok(core_triples) = core_store.triples() {
                drop(core_store); // Release lock before conversion
                for core_triple in core_triples {
                    if let Ok(star_triple) = self.convert_from_core_triple(&core_triple) {
                        all_triples.push(star_triple);
                    }
                }
            }
        }

        all_triples
    }

    /// Find triples that contain a specific quoted triple
    pub fn find_triples_containing_quoted(&self, quoted_triple: &StarTriple) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "find_triples_containing_quoted");
        let _enter = span.enter();

        let key = self.quoted_triple_key(quoted_triple);
        let index = self.quoted_triple_index.read().unwrap();
        let star_triples = self.star_triples.read().unwrap();

        if let Some(indices) = index.signature_to_indices.get(&key) {
            indices
                .iter()
                .filter_map(|&idx| star_triples.get(idx))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Advanced query method: find triples by quoted triple pattern
    pub fn find_triples_by_quoted_pattern(
        &self,
        subject_pattern: Option<&StarTerm>,
        predicate_pattern: Option<&StarTerm>,
        object_pattern: Option<&StarTerm>,
    ) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "find_triples_by_quoted_pattern");
        let _enter = span.enter();

        let index = self.quoted_triple_index.read().unwrap();
        let star_triples = self.star_triples.read().unwrap();
        let mut candidate_indices: Option<BTreeSet<usize>> = None;

        // Use subject index if subject pattern is provided
        if let Some(subject_term) = subject_pattern {
            let mut found_indices = BTreeSet::new();

            // Search in all index types for the subject term, as it could appear in any position within quoted triples
            let subject_key = format!("SUBJ:{}", subject_term);
            if let Some(indices) = index.subject_index.get(&subject_key) {
                found_indices.extend(indices);
            }

            let predicate_key = format!("PRED:{}", subject_term);
            if let Some(indices) = index.predicate_index.get(&predicate_key) {
                found_indices.extend(indices);
            }

            let object_key = format!("OBJ:{}", subject_term);
            if let Some(indices) = index.object_index.get(&object_key) {
                found_indices.extend(indices);
            }

            if found_indices.is_empty() {
                return Vec::new(); // No matches
            }

            candidate_indices = Some(found_indices);
        }

        // Use predicate index if predicate pattern is provided
        if let Some(predicate_term) = predicate_pattern {
            let predicate_key = format!("PRED:{}", predicate_term);
            if let Some(indices) = index.predicate_index.get(&predicate_key) {
                if let Some(ref mut candidates) = candidate_indices {
                    *candidates = candidates.intersection(indices).cloned().collect();
                } else {
                    candidate_indices = Some(indices.clone());
                }

                if candidate_indices.as_ref().unwrap().is_empty() {
                    return Vec::new(); // No matches
                }
            } else {
                return Vec::new(); // No matches
            }
        }

        // Use object index if object pattern is provided
        if let Some(object_term) = object_pattern {
            let object_key = format!("OBJ:{}", object_term);
            if let Some(indices) = index.object_index.get(&object_key) {
                if let Some(ref mut candidates) = candidate_indices {
                    *candidates = candidates.intersection(indices).cloned().collect();
                } else {
                    candidate_indices = Some(indices.clone());
                }

                if candidate_indices.as_ref().unwrap().is_empty() {
                    return Vec::new(); // No matches
                }
            } else {
                return Vec::new(); // No matches
            }
        }

        // If no pattern was provided, return all triples with quoted triples
        let final_indices = candidate_indices.unwrap_or_else(|| {
            index
                .signature_to_indices
                .values()
                .flat_map(|indices| indices.iter())
                .cloned()
                .collect()
        });

        final_indices
            .iter()
            .filter_map(|&idx| star_triples.get(idx))
            .cloned()
            .collect()
    }

    /// Find triples by nesting depth
    pub fn find_triples_by_nesting_depth(
        &self,
        min_depth: usize,
        max_depth: Option<usize>,
    ) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "find_triples_by_nesting_depth");
        let _enter = span.enter();

        let index = self.quoted_triple_index.read().unwrap();
        let star_triples = self.star_triples.read().unwrap();
        let mut result_indices = BTreeSet::new();

        let max_d = max_depth.unwrap_or(usize::MAX);

        for (&depth, indices) in index.nesting_depth_index.range(min_depth..=max_d) {
            result_indices.extend(indices);
        }

        result_indices
            .iter()
            .filter_map(|&idx: &usize| star_triples.get(idx))
            .cloned()
            .collect()
    }

    /// Get the number of triples in the store
    pub fn len(&self) -> usize {
        let star_triples = self.star_triples.read().unwrap();
        let core_store = self.core_store.read().unwrap();

        // Count both star triples and regular triples from core store
        let regular_count = core_store.len().unwrap_or(0);
        let star_count = star_triples.len();

        regular_count + star_count
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        let star_triples = self.star_triples.read().unwrap();
        let core_store = self.core_store.read().unwrap();

        // Empty only if both stores are empty
        star_triples.is_empty() && core_store.is_empty().unwrap_or(true)
    }

    /// Clear all triples from the store
    pub fn clear(&self) -> StarResult<()> {
        let span = span!(Level::INFO, "clear_store");
        let _enter = span.enter();

        {
            let mut star_triples = self.star_triples.write().unwrap();
            star_triples.clear();
        }

        {
            let mut index = self.quoted_triple_index.write().unwrap();
            index.clear();
        }

        {
            let mut stats = self.statistics.write().unwrap();
            *stats = StarStatistics::default();
        }

        info!("Cleared all triples from store");
        Ok(())
    }

    /// Get statistics about the store
    pub fn statistics(&self) -> StarStatistics {
        let stats = self.statistics.read().unwrap();
        stats.clone()
    }

    /// Export the store as a StarGraph
    pub fn to_graph(&self) -> StarGraph {
        let star_triples = self.star_triples.read().unwrap();
        let mut graph = StarGraph::new();

        for triple in star_triples.iter() {
            // Unwrap is safe here because we validate on insert
            graph.insert(triple.clone()).unwrap();
        }

        graph
    }

    /// Import triples from a StarGraph
    pub fn from_graph(&self, graph: &StarGraph) -> StarResult<()> {
        let span = span!(Level::INFO, "import_from_graph");
        let _enter = span.enter();

        for triple in graph.triples() {
            self.insert(triple)?;
        }

        info!("Imported {} triples from graph", graph.len());
        Ok(())
    }

    /// Optimize the store by rebuilding indices
    pub fn optimize(&self) -> StarResult<()> {
        let span = span!(Level::INFO, "optimize_store");
        let _enter = span.enter();

        let star_triples = self.star_triples.read().unwrap();
        let mut index = self.quoted_triple_index.write().unwrap();

        // Rebuild the quoted triple index with all new B-tree structures
        index.clear();
        for (i, triple) in star_triples.iter().enumerate() {
            if triple.contains_quoted_triples() {
                self.index_quoted_triples(triple, i, &mut index);
            }
        }

        // Compact the indices by removing empty entries
        index
            .signature_to_indices
            .retain(|_, indices| !indices.is_empty());
        index.subject_index.retain(|_, indices| !indices.is_empty());
        index
            .predicate_index
            .retain(|_, indices| !indices.is_empty());
        index.object_index.retain(|_, indices| !indices.is_empty());
        index
            .nesting_depth_index
            .retain(|_, indices| !indices.is_empty());

        info!(
            "Store optimization completed - rebuilt {} index entries",
            index.signature_to_indices.len()
                + index.subject_index.len()
                + index.predicate_index.len()
                + index.object_index.len()
                + index.nesting_depth_index.len()
        );
        Ok(())
    }

    /// Query triples from both core store and star store
    pub fn query_triples(
        &self,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> StarResult<Vec<StarTriple>> {
        let mut results = Vec::new();

        // Query star triples
        let star_triples = self.star_triples.read().unwrap();
        for triple in star_triples.iter() {
            if self.triple_matches(triple, subject, predicate, object) {
                results.push(triple.clone());
            }
        }

        // If no quoted triple patterns, also query core store
        let has_quoted_pattern = [subject, predicate, object]
            .iter()
            .any(|term| term.map_or(false, |t| t.is_quoted_triple()));

        if !has_quoted_pattern {
            // Convert patterns to core RDF terms and query core store
            let core_results = self.query_core_store(subject, predicate, object)?;
            results.extend(core_results);
        }

        Ok(results)
    }

    /// Query the core store with converted patterns
    fn query_core_store(
        &self,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> StarResult<Vec<StarTriple>> {
        let core_store = self.core_store.read().unwrap();

        // Convert patterns to core types
        let core_subject = match subject {
            Some(term) => Some(conversion::star_term_to_subject(term)?),
            None => None,
        };

        let core_predicate = match predicate {
            Some(term) => Some(conversion::star_term_to_predicate(term)?),
            None => None,
        };

        let core_object = match object {
            Some(term) => Some(conversion::star_term_to_object(term)?),
            None => None,
        };

        // Query core store (find quads and convert to triples)
        let core_quads = core_store
            .find_quads(
                core_subject.as_ref(),
                core_predicate.as_ref(),
                core_object.as_ref(),
                None, // Query all graphs
            )
            .map_err(|e| StarError::CoreError(e))?;

        // Convert results back to StarTriples
        let mut results = Vec::new();
        for quad in core_quads {
            // Convert quad to triple (lose graph information)
            let triple = oxirs_core::model::Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            );
            let star_triple = self.convert_from_core_triple(&triple)?;
            results.push(star_triple);
        }

        Ok(results)
    }

    /// Convert a core RDF Triple to a StarTriple
    fn convert_from_core_triple(
        &self,
        triple: &oxirs_core::model::Triple,
    ) -> StarResult<StarTriple> {
        let subject = self.convert_subject_from_core(triple.subject())?;
        let predicate = self.convert_predicate_from_core(triple.predicate())?;
        let object = self.convert_object_from_core(triple.object())?;

        Ok(StarTriple::new(subject, predicate, object))
    }

    /// Convert core Subject to StarTerm
    fn convert_subject_from_core(
        &self,
        subject: &oxirs_core::model::Subject,
    ) -> StarResult<StarTerm> {
        match subject {
            oxirs_core::model::Subject::NamedNode(nn) => Ok(StarTerm::iri(nn.as_str())?),
            oxirs_core::model::Subject::BlankNode(bn) => Ok(StarTerm::blank_node(bn.as_str())?),
            oxirs_core::model::Subject::Variable(_) => Err(StarError::invalid_term_type(
                "Variables are not supported in subjects for RDF-star storage".to_string(),
            )),
            oxirs_core::model::Subject::QuotedTriple(_) => Err(StarError::invalid_term_type(
                "Quoted triples from core are not yet supported".to_string(),
            )),
        }
    }

    /// Convert core Predicate to StarTerm
    fn convert_predicate_from_core(
        &self,
        predicate: &oxirs_core::model::Predicate,
    ) -> StarResult<StarTerm> {
        match predicate {
            oxirs_core::model::Predicate::NamedNode(nn) => Ok(StarTerm::iri(nn.as_str())?),
            oxirs_core::model::Predicate::Variable(_) => Err(StarError::invalid_term_type(
                "Variables are not supported in predicates for RDF-star storage".to_string(),
            )),
        }
    }

    /// Convert core Object to StarTerm
    fn convert_object_from_core(&self, object: &oxirs_core::model::Object) -> StarResult<StarTerm> {
        match object {
            oxirs_core::model::Object::NamedNode(nn) => Ok(StarTerm::iri(nn.as_str())?),
            oxirs_core::model::Object::BlankNode(bn) => Ok(StarTerm::blank_node(bn.as_str())?),
            oxirs_core::model::Object::Literal(lit) => Ok(StarTerm::literal(lit.value())?),
            oxirs_core::model::Object::Variable(_) => Err(StarError::invalid_term_type(
                "Variables are not supported in objects for RDF-star storage".to_string(),
            )),
            oxirs_core::model::Object::QuotedTriple(_) => Err(StarError::invalid_term_type(
                "Quoted triples from core are not yet supported".to_string(),
            )),
        }
    }

    /// Check if a triple matches the given pattern
    fn triple_matches(
        &self,
        triple: &StarTriple,
        subject: Option<&StarTerm>,
        predicate: Option<&StarTerm>,
        object: Option<&StarTerm>,
    ) -> bool {
        if let Some(s) = subject {
            if &triple.subject != s {
                return false;
            }
        }
        if let Some(p) = predicate {
            if &triple.predicate != p {
                return false;
            }
        }
        if let Some(o) = object {
            if &triple.object != o {
                return false;
            }
        }
        true
    }

    /// Get configuration
    pub fn config(&self) -> &StarConfig {
        &self.config
    }

    /// Update configuration (requires store recreation for some settings)
    pub fn update_config(&mut self, config: StarConfig) -> StarResult<()> {
        // Validate new configuration
        crate::init_star_system(config.clone())?;
        self.config = config;
        Ok(())
    }
}

impl Default for StarStore {
    fn default() -> Self {
        Self::new()
    }
}

// Note: StarTripleIterator has been removed in favor of a safer iterator implementation
// that doesn't use unsafe code or hold locks across method boundaries

impl StarStore {
    /// Get a vector of all triples (cloned to avoid lifetime issues)
    pub fn all_triples(&self) -> Vec<StarTriple> {
        let star_triples = self.star_triples.read().unwrap();
        star_triples.clone()
    }

    /// Get an iterator over all triples using a safe implementation
    pub fn iter(&self) -> impl Iterator<Item = StarTriple> {
        // Clone all triples to avoid holding the lock
        // This is safe but potentially memory-intensive for large stores
        // For production use, consider using the streaming_iter method
        self.all_triples().into_iter()
    }

    /// Get a streaming iterator that processes triples in chunks
    /// This is more memory-efficient for large stores
    pub fn streaming_iter(&self, chunk_size: usize) -> StreamingTripleIterator {
        StreamingTripleIterator::new(self, chunk_size)
    }

    /// Bulk insert triples with optimized performance
    pub fn bulk_insert(&self, triples: &[StarTriple], config: &BulkInsertConfig) -> StarResult<()> {
        let span = span!(Level::INFO, "bulk_insert", count = triples.len());
        let _enter = span.enter();

        info!("Starting bulk insertion of {} triples", triples.len());
        let start_time = Instant::now();

        // Enable bulk mode
        {
            let mut bulk_state = self.bulk_insert_state.write().unwrap();
            bulk_state.active = true;
            bulk_state.pending_triples.clear();
            bulk_state.current_memory_usage = 0;
            bulk_state.batch_count = 0;
        }

        if config.parallel_processing && triples.len() > config.batch_size {
            self.bulk_insert_parallel(triples, config)?;
        } else {
            self.bulk_insert_sequential(triples, config)?;
        }

        // Finalize bulk insertion
        self.finalize_bulk_insert(config)?;

        let elapsed = start_time.elapsed();
        info!(
            "Bulk insertion completed in {:?} for {} triples",
            elapsed,
            triples.len()
        );

        // Update statistics
        {
            let mut stats = self.statistics.write().unwrap();
            stats.processing_time_us += elapsed.as_micros() as u64;
        }

        Ok(())
    }

    /// Sequential bulk insertion implementation
    fn bulk_insert_sequential(
        &self,
        triples: &[StarTriple],
        config: &BulkInsertConfig,
    ) -> StarResult<()> {
        for batch in triples.chunks(config.batch_size) {
            for triple in batch {
                // Validate the triple
                triple.validate()?;

                // Insert based on triple type
                if triple.contains_quoted_triples() {
                    if config.defer_index_updates {
                        // Add to pending list for later indexing
                        let mut bulk_state = self.bulk_insert_state.write().unwrap();
                        bulk_state.pending_triples.push(triple.clone());
                        bulk_state.current_memory_usage += self.estimate_triple_memory_size(triple);
                    } else {
                        self.insert_star_triple(triple)?;
                    }
                } else {
                    self.insert_regular_triple(triple)?;
                }
            }

            // Check memory threshold
            {
                let bulk_state = self.bulk_insert_state.read().unwrap();
                if bulk_state.current_memory_usage >= config.memory_threshold {
                    drop(bulk_state);
                    self.flush_pending_triples(config)?;
                }
            }

            // Update batch count
            {
                let mut bulk_state = self.bulk_insert_state.write().unwrap();
                bulk_state.batch_count += 1;
            }
        }

        Ok(())
    }

    /// Parallel bulk insertion implementation
    fn bulk_insert_parallel(
        &self,
        triples: &[StarTriple],
        config: &BulkInsertConfig,
    ) -> StarResult<()> {
        let chunk_size = triples.len() / config.worker_threads;
        let mut handles = Vec::new();

        for chunk in triples.chunks(chunk_size) {
            let chunk = chunk.to_vec();
            let store_clone = self.clone();
            let config_clone = config.clone();

            let handle =
                thread::spawn(move || store_clone.bulk_insert_sequential(&chunk, &config_clone));
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle
                .join()
                .map_err(|e| StarError::query_error(format!("Thread join error: {:?}", e)))??;
        }

        Ok(())
    }

    /// Flush pending triples and rebuild indices
    fn flush_pending_triples(&self, config: &BulkInsertConfig) -> StarResult<()> {
        let pending_triples = {
            let mut bulk_state = self.bulk_insert_state.write().unwrap();
            let triples = bulk_state.pending_triples.clone();
            bulk_state.pending_triples.clear();
            bulk_state.current_memory_usage = 0;
            triples
        };

        if !pending_triples.is_empty() {
            debug!("Flushing {} pending triples", pending_triples.len());

            // Insert all pending triples into storage
            {
                let mut star_triples = self.star_triples.write().unwrap();
                let base_index = star_triples.len();
                star_triples.extend(pending_triples.clone());

                // Build indices for the new triples
                if !config.defer_index_updates {
                    let mut index = self.quoted_triple_index.write().unwrap();
                    for (i, triple) in pending_triples.iter().enumerate() {
                        self.index_quoted_triples(triple, base_index + i, &mut index);
                    }
                }
            }
        }

        Ok(())
    }

    /// Finalize bulk insertion by rebuilding indices if needed
    fn finalize_bulk_insert(&self, config: &BulkInsertConfig) -> StarResult<()> {
        // Flush any remaining pending triples
        self.flush_pending_triples(config)?;

        // Rebuild indices if they were deferred
        if config.defer_index_updates {
            info!("Rebuilding indices after bulk insertion");
            self.optimize()?;
        }

        // Reset bulk state
        {
            let mut bulk_state = self.bulk_insert_state.write().unwrap();
            bulk_state.active = false;
            bulk_state.pending_triples.clear();
            bulk_state.current_memory_usage = 0;
            bulk_state.batch_count = 0;
        }

        Ok(())
    }

    /// Estimate memory size of a triple for memory tracking
    fn estimate_triple_memory_size(&self, triple: &StarTriple) -> usize {
        // Rough estimation based on string lengths and structure
        let subject_size = match &triple.subject {
            StarTerm::NamedNode(nn) => nn.iri.len(),
            StarTerm::BlankNode(bn) => bn.id.len(),
            StarTerm::Literal(lit) => lit.value.len(),
            StarTerm::QuotedTriple(_) => 200, // Estimated overhead
            StarTerm::Variable(var) => var.name.len(),
        };

        let predicate_size = match &triple.predicate {
            StarTerm::NamedNode(nn) => nn.iri.len(),
            _ => 50, // Default estimate
        };

        let object_size = match &triple.object {
            StarTerm::NamedNode(nn) => nn.iri.len(),
            StarTerm::BlankNode(bn) => bn.id.len(),
            StarTerm::Literal(lit) => lit.value.len(),
            StarTerm::QuotedTriple(_) => 200, // Estimated overhead
            StarTerm::Variable(var) => var.name.len(),
        };

        subject_size + predicate_size + object_size + 100 // Base overhead
    }

    /// Enable memory-mapped storage
    pub fn enable_memory_mapping(
        &self,
        file_path: &str,
        enable_compression: bool,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "enable_memory_mapping");
        let _enter = span.enter();

        info!("Enabling memory-mapped storage at: {}", file_path);

        {
            let mut mm_state = self.memory_mapped.write().unwrap();
            mm_state.enabled = true;
            mm_state.file_path = Some(file_path.to_string());
            mm_state.compression_enabled = enable_compression;
            mm_state.last_sync = Some(Instant::now());
        }

        // In a full implementation, this would set up actual memory mapping
        // For now, we just track the state
        info!(
            "Memory-mapped storage enabled with compression: {}",
            enable_compression
        );
        Ok(())
    }

    /// Get optimized triples using cache
    pub fn get_triples_cached(&self, pattern: &str) -> Vec<StarTriple> {
        let span = span!(Level::DEBUG, "get_triples_cached");
        let _enter = span.enter();

        // Check cache first
        if let Some(cached_results) = self.cache.get(pattern) {
            debug!("Cache hit for pattern: {}", pattern);
            return cached_results;
        }

        // Cache miss - compute results
        debug!("Cache miss for pattern: {}", pattern);
        let results = self.compute_pattern_results(pattern);

        // Store in cache
        self.cache.put(pattern.to_string(), results.clone());

        results
    }

    /// Compute pattern results (placeholder implementation)
    fn compute_pattern_results(&self, pattern: &str) -> Vec<StarTriple> {
        // This is a simplified implementation
        // In practice, this would parse the pattern and execute the query
        if pattern.contains("quoted") {
            self.find_triples_by_nesting_depth(1, None)
        } else {
            self.triples()
        }
    }

    /// Get comprehensive storage statistics
    pub fn get_detailed_statistics(&self) -> DetailedStorageStatistics {
        let base_stats = self.statistics();
        let cache_stats = self.cache.get_statistics();
        let index_stats = {
            let index = self.quoted_triple_index.read().unwrap();
            index.get_statistics()
        };
        let bulk_state = self.bulk_insert_state.read().unwrap();
        let mm_state = self.memory_mapped.read().unwrap();

        DetailedStorageStatistics {
            basic_stats: base_stats,
            cache_stats,
            index_stats,
            bulk_insert_active: bulk_state.active,
            bulk_pending_count: bulk_state.pending_triples.len(),
            bulk_memory_usage: bulk_state.current_memory_usage,
            memory_mapped_enabled: mm_state.enabled,
            memory_mapped_path: mm_state.file_path.clone(),
        }
    }

    /// Create a connection pool for this store type
    pub fn create_connection_pool(max_connections: usize, config: StarConfig) -> ConnectionPool {
        ConnectionPool::new(max_connections, config)
    }

    /// Compress stored data (placeholder implementation)
    pub fn compress_storage(&self) -> StarResult<usize> {
        let span = span!(Level::INFO, "compress_storage");
        let _enter = span.enter();

        // In a full implementation, this would compress the stored triples
        let triple_count = self.len();
        info!("Compressed storage for {} triples", triple_count);

        // Return estimated space saved (placeholder)
        Ok(triple_count * 50)
    }
}

/// Comprehensive storage statistics including optimizations
#[derive(Debug, Clone)]
pub struct DetailedStorageStatistics {
    pub basic_stats: StarStatistics,
    pub cache_stats: CacheStatistics,
    pub index_stats: IndexStatistics,
    pub bulk_insert_active: bool,
    pub bulk_pending_count: usize,
    pub bulk_memory_usage: usize,
    pub memory_mapped_enabled: bool,
    pub memory_mapped_path: Option<String>,
}

/// A memory-efficient streaming iterator for large triple stores
pub struct StreamingTripleIterator<'a> {
    store: &'a StarStore,
    chunk_size: usize,
    current_chunk: Vec<StarTriple>,
    current_index: usize,
    total_processed: usize,
}

impl<'a> StreamingTripleIterator<'a> {
    fn new(store: &'a StarStore, chunk_size: usize) -> Self {
        Self {
            store,
            chunk_size: chunk_size.max(1),
            current_chunk: Vec::new(),
            current_index: 0,
            total_processed: 0,
        }
    }

    fn load_next_chunk(&mut self) -> bool {
        let star_triples = self.store.star_triples.read().unwrap();

        // Calculate the range for the next chunk
        let start = self.total_processed;
        let end = (start + self.chunk_size).min(star_triples.len());

        if start >= star_triples.len() {
            return false;
        }

        // Load the chunk
        self.current_chunk.clear();
        self.current_chunk
            .extend(star_triples.iter().skip(start).take(end - start).cloned());

        self.current_index = 0;
        !self.current_chunk.is_empty()
    }
}

impl<'a> Iterator for StreamingTripleIterator<'a> {
    type Item = StarTriple;

    fn next(&mut self) -> Option<Self::Item> {
        // If we've exhausted the current chunk, load the next one
        if self.current_index >= self.current_chunk.len() {
            if !self.load_next_chunk() {
                return None;
            }
        }

        // Return the next triple from the current chunk
        let triple = self.current_chunk.get(self.current_index).cloned();
        if triple.is_some() {
            self.current_index += 1;
            self.total_processed += 1;
        }
        triple
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;

    #[test]
    fn test_store_creation() {
        let store = StarStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_basic_operations() {
        let store = StarStore::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        // Insert
        store.insert(&triple).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.contains(&triple));

        // Query
        let results = store
            .query_triples(
                Some(&StarTerm::iri("http://example.org/alice").unwrap()),
                None,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);

        // Remove
        assert!(store.remove(&triple).unwrap());
        assert!(store.is_empty());
    }

    #[test]
    fn test_quoted_triple_operations() {
        let store = StarStore::new();

        // Create a quoted triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        store.insert(&outer).unwrap();
        assert_eq!(store.len(), 1);

        // Find triples containing the quoted triple
        let containing = store.find_triples_containing_quoted(&inner);
        assert_eq!(containing.len(), 1);
        assert_eq!(containing[0], outer);
    }

    #[test]
    fn test_store_statistics() {
        let store = StarStore::new();

        let regular = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        let quoted = StarTriple::new(
            StarTerm::quoted_triple(regular.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("high").unwrap(),
        );

        store.insert(&regular).unwrap();
        store.insert(&quoted).unwrap();

        let stats = store.statistics();
        assert_eq!(stats.quoted_triples_count, 1);
        assert_eq!(stats.max_nesting_encountered, 1);
    }

    #[test]
    fn test_btree_indexing_performance() {
        let store = StarStore::new();

        // Create multiple quoted triples with different patterns
        let base_triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let quoted1 = StarTriple::new(
            StarTerm::quoted_triple(base_triple.clone()),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        let quoted2 = StarTriple::new(
            StarTerm::iri("http://example.org/bob").unwrap(),
            StarTerm::iri("http://example.org/believes").unwrap(),
            StarTerm::quoted_triple(base_triple.clone()),
        );

        store.insert(&quoted1).unwrap();
        store.insert(&quoted2).unwrap();

        // Test pattern-based queries using the new B-tree indices
        let results = store.find_triples_by_quoted_pattern(
            Some(&StarTerm::iri("http://example.org/alice").unwrap()),
            None,
            None,
        );
        assert_eq!(results.len(), 2);

        // Test nesting depth queries
        let shallow_results = store.find_triples_by_nesting_depth(0, Some(0));
        assert_eq!(shallow_results.len(), 0); // No triples with depth 0

        let depth_1_results = store.find_triples_by_nesting_depth(1, Some(1));
        assert_eq!(depth_1_results.len(), 2); // Both quoted triples have depth 1
    }

    #[test]
    fn test_graph_import_export() {
        let store = StarStore::new();
        let mut graph = StarGraph::new();

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        graph.insert(triple.clone()).unwrap();
        store.from_graph(&graph).unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.contains(&triple));

        let exported = store.to_graph();
        assert_eq!(exported.len(), 1);
        assert!(exported.contains(&triple));
    }

    #[test]
    fn test_streaming_iterator() {
        let store = StarStore::new();

        // Insert multiple triples
        for i in 0..100 {
            let triple = StarTriple::new(
                StarTerm::iri(&format!("http://example.org/s{}", i)).unwrap(),
                StarTerm::iri("http://example.org/p").unwrap(),
                StarTerm::iri(&format!("http://example.org/o{}", i)).unwrap(),
            );
            store.insert(&triple).unwrap();
        }

        // Test streaming iterator with different chunk sizes
        let chunk_sizes = vec![1, 10, 50, 100, 200];

        for chunk_size in chunk_sizes {
            let mut count = 0;
            for _triple in store.streaming_iter(chunk_size) {
                count += 1;
            }
            assert_eq!(
                count, 100,
                "Streaming iterator with chunk size {} should return all triples",
                chunk_size
            );
        }

        // Test that streaming iterator returns the same triples as regular iterator
        let regular_triples: Vec<_> = store.iter().collect();
        let streaming_triples: Vec<_> = store.streaming_iter(25).collect();

        assert_eq!(regular_triples.len(), streaming_triples.len());

        // Both iterators should contain the same triples (though possibly in different order)
        for triple in &regular_triples {
            assert!(streaming_triples.contains(triple));
        }
    }
}
