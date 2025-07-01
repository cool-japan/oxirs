//! Main HNSW index implementation

use crate::hnsw::{HnswConfig, HnswPerformanceStats, Node};
use crate::{Vector, VectorIndex};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
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

        Ok(Self {
            config,
            nodes: Vec::new(),
            uri_to_id: HashMap::new(),
            entry_point: None,
            level_multiplier: 1.0 / (2.0_f64).ln(),
            rng_state: 42, // Simple deterministic seed
            stats: HnswPerformanceStats::default(),
            distance_calculations: AtomicU64::new(0),
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

        Self {
            config: cpu_config,
            nodes: Vec::new(),
            uri_to_id: HashMap::new(),
            entry_point: None,
            level_multiplier: 1.0 / (2.0_f64).ln(),
            rng_state: 42,
            stats: HnswPerformanceStats::default(),
            distance_calculations: AtomicU64::new(0),
            #[cfg(feature = "gpu")]
            gpu_accelerator: None,
            #[cfg(feature = "gpu")]
            multi_gpu_accelerators: Vec::new(),
        }
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
    pub fn is_gpu_enabled(&self) -> bool {
        self.config.enable_gpu
            && (self.gpu_accelerator.is_some() || !self.multi_gpu_accelerators.is_empty())
    }

    #[cfg(not(feature = "gpu"))]
    pub fn is_gpu_enabled(&self) -> bool {
        false
    }

    /// Get GPU performance statistics
    #[cfg(feature = "gpu")]
    pub fn get_gpu_stats(&self) -> Option<crate::gpu::GpuPerformanceStats> {
        if let Some(ref accelerator) = self.gpu_accelerator {
            // Would need to implement stats retrieval in GpuAccelerator
            None // Placeholder
        } else {
            None
        }
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
        // Placeholder implementation - will be moved to construction module
        todo!("Implementation moved to construction module")
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        // Placeholder implementation - will be moved to search module
        todo!("Implementation moved to search module")
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        // Placeholder implementation - will be moved to search module
        todo!("Implementation moved to search module")
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
        // Placeholder implementation
        todo!("Remove functionality not yet implemented")
    }

    /// Update a vector by its URI (not part of VectorIndex trait)
    pub fn update(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Placeholder implementation
        todo!("Update functionality not yet implemented")
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
