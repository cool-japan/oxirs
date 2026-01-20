//! Main HNSW index implementation

use crate::hnsw::query_cache::{QueryCache, QueryCacheConfig};
use crate::hnsw::{HnswConfig, HnswPerformanceStats, Node};
use crate::{Vector, VectorIndex};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::gpu::GpuAccelerator;

/// HNSW index implementation
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<Node>,
    uri_to_id: HashMap<String, usize>,
    entry_point: Option<usize>,
    level_multiplier: f64,
    rng_state: u64,
    /// Performance statistics
    stats: HnswPerformanceStats,
    /// Distance calculation count (for metrics)
    distance_calculations: AtomicU64,
    /// Query result cache for improved performance
    query_cache: Option<QueryCache>,
    /// GPU accelerator for CUDA-accelerated operations
    #[cfg(feature = "gpu")]
    gpu_accelerator: Option<Arc<GpuAccelerator>>,
    /// Multi-GPU accelerators for distributed computation
    #[cfg(feature = "gpu")]
    multi_gpu_accelerators: Vec<Arc<GpuAccelerator>>,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Result<Self> {
        // Initialize GPU accelerators if enabled
        #[cfg(feature = "gpu")]
        let (gpu_accelerator, multi_gpu_accelerators) = if config.enable_gpu {
            let gpu_config = config.gpu_config.clone().unwrap_or_default();

            if config.enable_multi_gpu && gpu_config.preferred_gpu_ids.len() > 1 {
                // Initialize multi-GPU setup
                let mut accelerators = Vec::new();
                for &gpu_id in &gpu_config.preferred_gpu_ids {
                    let mut gpu_conf = gpu_config.clone();
                    gpu_conf.device_id = gpu_id;
                    let accelerator = GpuAccelerator::new(gpu_conf)?;
                    accelerators.push(Arc::new(accelerator));
                }
                (None, accelerators)
            } else {
                // Single GPU setup
                let accelerator = GpuAccelerator::new(gpu_config)?;
                (Some(Arc::new(accelerator)), Vec::new())
            }
        } else {
            (None, Vec::new())
        };

        // Initialize query cache with default configuration
        let query_cache = Some(QueryCache::new(QueryCacheConfig::default()));

        Ok(Self {
            config,
            nodes: Vec::new(),
            uri_to_id: HashMap::new(),
            entry_point: None,
            level_multiplier: 1.0 / (2.0_f64).ln(),
            rng_state: 42, // Simple deterministic seed
            stats: HnswPerformanceStats::default(),
            distance_calculations: AtomicU64::new(0),
            query_cache,
            #[cfg(feature = "gpu")]
            gpu_accelerator,
            #[cfg(feature = "gpu")]
            multi_gpu_accelerators,
        })
    }

    /// Create a new HNSW index without GPU acceleration (for compatibility)
    pub fn new_cpu_only(config: HnswConfig) -> Self {
        let mut cpu_config = config;
        cpu_config.enable_gpu = false;
        cpu_config.enable_multi_gpu = false;

        // Initialize query cache with default configuration
        let query_cache = Some(QueryCache::new(QueryCacheConfig::default()));

        Self {
            config: cpu_config,
            nodes: Vec::new(),
            uri_to_id: HashMap::new(),
            entry_point: None,
            level_multiplier: 1.0 / (2.0_f64).ln(),
            rng_state: 42,
            stats: HnswPerformanceStats::default(),
            distance_calculations: AtomicU64::new(0),
            query_cache,
            #[cfg(feature = "gpu")]
            gpu_accelerator: None,
            #[cfg(feature = "gpu")]
            multi_gpu_accelerators: Vec::new(),
        }
    }

    /// Enable query result caching with custom configuration
    pub fn enable_query_cache(&mut self, config: QueryCacheConfig) {
        self.query_cache = Some(QueryCache::new(config));
    }

    /// Disable query result caching
    pub fn disable_query_cache(&mut self) {
        self.query_cache = None;
    }

    /// Get query cache statistics if caching is enabled
    pub fn get_query_cache_stats(&self) -> Option<crate::hnsw::query_cache::QueryCacheStats> {
        self.query_cache.as_ref().map(|cache| cache.get_stats())
    }

    /// Clear query cache if caching is enabled
    pub fn clear_query_cache(&self) {
        if let Some(ref cache) = self.query_cache {
            cache.clear();
        }
    }

    /// Get reference to query cache
    pub(crate) fn query_cache(&self) -> &Option<QueryCache> {
        &self.query_cache
    }

    /// Get the URI to ID mapping
    pub fn uri_to_id(&self) -> &HashMap<String, usize> {
        &self.uri_to_id
    }

    /// Get mutable URI to ID mapping
    pub fn uri_to_id_mut(&mut self) -> &mut HashMap<String, usize> {
        &mut self.uri_to_id
    }

    /// Get the nodes
    pub fn nodes(&self) -> &Vec<Node> {
        &self.nodes
    }

    /// Get mutable nodes
    pub fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    /// Get the entry point
    pub fn entry_point(&self) -> Option<usize> {
        self.entry_point
    }

    /// Set the entry point
    pub fn set_entry_point(&mut self, entry_point: Option<usize>) {
        self.entry_point = entry_point;
    }

    /// Get the configuration
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &HnswPerformanceStats {
        &self.stats
    }

    /// Check if GPU acceleration is available and enabled
    #[cfg(feature = "gpu")]
    pub fn is_gpu_available(&self) -> bool {
        self.config.enable_gpu
            && (self.gpu_accelerator.is_some() || !self.multi_gpu_accelerators.is_empty())
    }

    #[cfg(not(feature = "gpu"))]
    pub fn is_gpu_available(&self) -> bool {
        false
    }

    /// Get GPU performance statistics
    #[cfg(feature = "gpu")]
    pub fn get_gpu_stats(&self) -> Option<crate::gpu::GpuPerformanceStats> {
        if let Some(ref _accelerator) = self.gpu_accelerator {
            // Would need to implement stats retrieval in GpuAccelerator
            None // Placeholder
        } else {
            None
        }
    }

    /// Get reference to GPU accelerator
    #[cfg(feature = "gpu")]
    pub fn gpu_accelerator(&self) -> Option<&Arc<GpuAccelerator>> {
        self.gpu_accelerator.as_ref()
    }

    /// Get reference to multi-GPU accelerators
    #[cfg(feature = "gpu")]
    pub fn multi_gpu_accelerators(&self) -> &Vec<Arc<GpuAccelerator>> {
        &self.multi_gpu_accelerators
    }

    /// Get the number of nodes in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // Duplicate methods removed - already defined above

    /// Get mutable reference to stats
    pub fn stats_mut(&mut self) -> &mut HnswPerformanceStats {
        &mut self.stats
    }

    /// Get level multiplier
    pub fn level_multiplier(&self) -> f64 {
        self.level_multiplier
    }

    /// Get mutable reference to RNG state
    pub fn rng_state_mut(&mut self) -> &mut u64 {
        &mut self.rng_state
    }

    /// Get RNG state
    pub fn rng_state(&self) -> u64 {
        self.rng_state
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Use the add_vector implementation from construction module
        self.add_vector(uri, vector)
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        // Use the proper HNSW search algorithm from search.rs
        // This implements hierarchical navigable small world graph traversal
        HnswIndex::search_knn(self, query, k)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        // Use the proper HNSW range search algorithm from search.rs
        // This implements distance-based filtering with graph traversal
        HnswIndex::range_search(self, query, threshold)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        self.uri_to_id
            .get(uri)
            .and_then(|&id| self.nodes.get(id))
            .map(|node| &node.vector)
    }
}

impl HnswIndex {
    /// Remove a vector by its URI (not part of VectorIndex trait)
    pub fn remove(&mut self, uri: &str) -> Result<()> {
        // Implementation of vector removal from HNSW index

        // Find the node ID for the URI
        let node_id = if let Some(&id) = self.uri_to_id.get(uri) {
            id
        } else {
            return Err(anyhow::anyhow!("URI not found: {}", uri));
        };

        // Remove the node from all its connections at all levels
        if let Some(node) = self.nodes.get(node_id) {
            let node_connections = node.connections.clone();

            // Remove this node from all connected nodes
            for (level, connections) in node_connections.iter().enumerate() {
                for &connected_id in connections {
                    if let Some(connected_node) = self.nodes.get_mut(connected_id) {
                        connected_node.remove_connection(level, node_id);
                    }
                }
            }
        }

        // If this node was the entry point, find a new entry point
        if self.entry_point == Some(node_id) {
            self.entry_point = None;

            // Find a node with the highest level as the new entry point
            let mut highest_level = 0;
            let mut new_entry_point = None;

            for (id, node) in self.nodes.iter().enumerate() {
                if id != node_id && node.level() >= highest_level {
                    highest_level = node.level();
                    new_entry_point = Some(id);
                }
            }

            self.entry_point = new_entry_point;
        }

        // Remove the node from URI mapping
        self.uri_to_id.remove(uri);

        // Mark the node as removed (we don't actually remove it to avoid ID shifts)
        // In a production implementation, you might use a tombstone approach
        // or compact the index periodically
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.connections.clear();
            // We could add a "deleted" flag here if needed
        }

        // Update statistics
        self.stats
            .total_deletions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Update a vector by its URI (not part of VectorIndex trait)
    pub fn update(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Implementation of vector update in HNSW index
        // This is a simplified approach: remove and re-add the vector

        // Check if the URI exists
        if !self.uri_to_id.contains_key(&uri) {
            return Err(anyhow::anyhow!("URI not found: {}", uri));
        }

        // Store the current connections before removal for potential optimization
        let node_id = self.uri_to_id[&uri];
        let _old_connections = self.nodes.get(node_id).map(|node| node.connections.clone());

        // Remove the old vector
        self.remove(&uri)?;

        // Add the new vector with the same URI
        self.insert(uri.clone(), vector)?;

        // Update statistics
        self.stats
            .total_updates
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // In a more sophisticated implementation, we could:
        // 1. Check if the vector is similar enough to keep some connections
        // 2. Incrementally update the graph structure
        // 3. Use lazy updates to batch multiple updates

        Ok(())
    }

    /// Clear all vectors from the index (not part of VectorIndex trait)
    pub fn clear(&mut self) -> Result<()> {
        self.nodes.clear();
        self.uri_to_id.clear();
        self.entry_point = None;
        Ok(())
    }

    /// Get the number of vectors in the index (not part of VectorIndex trait)
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}
