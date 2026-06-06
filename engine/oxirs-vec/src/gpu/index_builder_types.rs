//! Types, configuration, and data structures for GPU-accelerated HNSW index building.
//!
//! This module contains all configuration structs, enums, result types, and the
//! core HNSW graph structures used throughout the GPU index builder pipeline.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Cached computation results indexed by (query_dim, db_size)
pub type ComputationCache =
    std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<(usize, usize), Vec<Vec<f32>>>>>;

/// Configuration for GPU-accelerated HNSW index building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuIndexBuilderConfig {
    /// Target HNSW M parameter (max neighbors per layer)
    pub m: usize,
    /// HNSW ef_construction parameter
    pub ef_construction: usize,
    /// Number of layers in HNSW hierarchy
    pub num_layers: usize,
    /// GPU device to use for computation
    pub gpu_device_id: i32,
    /// Batch size for GPU distance computation
    pub batch_size: usize,
    /// Enable mixed-precision (FP16) for distance computation
    pub mixed_precision: bool,
    /// Enable tensor core acceleration
    pub tensor_cores: bool,
    /// Number of CUDA streams for pipelining
    pub num_streams: usize,
    /// Maximum vectors to hold in GPU memory at once
    pub gpu_memory_budget_mb: usize,
    /// Enable asynchronous memory transfers
    pub async_transfers: bool,
    /// Distance metric kernel to use
    pub distance_metric: GpuDistanceMetric,
}

impl Default for GpuIndexBuilderConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            num_layers: 4,
            gpu_device_id: 0,
            batch_size: 1024,
            mixed_precision: true,
            tensor_cores: true,
            num_streams: 4,
            gpu_memory_budget_mb: 4096,
            async_transfers: true,
            distance_metric: GpuDistanceMetric::Cosine,
        }
    }
}

/// Distance metric type for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDistanceMetric {
    /// Cosine similarity (most common for embeddings)
    Cosine,
    /// Euclidean / L2 distance
    Euclidean,
    /// Inner product / dot product
    InnerProduct,
    /// FP16 cosine (faster, slight precision loss)
    CosineF16,
    /// FP16 Euclidean (faster, slight precision loss)
    EuclideanF16,
}

/// Build statistics for the GPU index builder
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuIndexBuildStats {
    /// Total vectors indexed
    pub vectors_indexed: usize,
    /// Total build time in milliseconds
    pub build_time_ms: u64,
    /// Time spent on GPU distance computation (ms)
    pub gpu_compute_time_ms: u64,
    /// Time spent on data transfers (ms)
    pub transfer_time_ms: u64,
    /// Time spent on CPU-side graph assembly (ms)
    pub graph_assembly_time_ms: u64,
    /// Number of GPU batches processed
    pub batches_processed: usize,
    /// Peak GPU memory usage (bytes)
    pub peak_gpu_memory_bytes: usize,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization_pct: f32,
    /// Effective throughput (vectors/second)
    pub throughput_vps: f64,
    /// Whether mixed precision was used
    pub used_mixed_precision: bool,
    /// Whether tensor cores were used
    pub used_tensor_cores: bool,
}

/// A node in the HNSW graph
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Unique node ID
    pub id: usize,
    /// Vector data
    pub vector: Vec<f32>,
    /// Neighbor lists per layer: `neighbors[layer] = [node_ids]`
    pub neighbors: Vec<Vec<usize>>,
    /// Layer this node was assigned to (max layer index)
    pub max_layer: usize,
}

/// The assembled HNSW graph structure
#[derive(Debug)]
pub struct HnswGraph {
    /// All nodes in the graph
    pub nodes: Vec<HnswNode>,
    /// Entry point node ID
    pub entry_point: usize,
    /// Maximum layer in the graph
    pub max_layer: usize,
    /// Build configuration used
    pub config: GpuIndexBuilderConfig,
    /// Build statistics
    pub stats: GpuIndexBuildStats,
}

impl HnswGraph {
    /// Search the graph for k nearest neighbors
    pub fn search_knn(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(usize, f32)>> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        if query.len() != self.nodes[0].vector.len() {
            return Err(anyhow!(
                "Query dimension {} != index dimension {}",
                query.len(),
                self.nodes[0].vector.len()
            ));
        }

        // Greedy search from entry point, descending layers
        let entry = self.entry_point;
        let mut current_best = entry;
        let mut current_dist = self.compute_distance(query, &self.nodes[entry].vector);

        // Phase 1: descend from max_layer to layer 1
        for layer in (1..=self.max_layer).rev() {
            let mut improved = true;
            while improved {
                improved = false;
                if layer >= self.nodes[current_best].neighbors.len() {
                    break;
                }
                for &neighbor_id in &self.nodes[current_best].neighbors[layer] {
                    let neighbor_dist =
                        self.compute_distance(query, &self.nodes[neighbor_id].vector);
                    if neighbor_dist < current_dist {
                        current_dist = neighbor_dist;
                        current_best = neighbor_id;
                        improved = true;
                    }
                }
            }
        }

        // Phase 2: beam search at layer 0
        let search_ef = ef.max(k);
        let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(search_ef * 2);
        let mut visited: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(search_ef * 4);

        candidates.push((current_dist, current_best));
        visited.insert(current_best);

        let mut w: Vec<(f32, usize)> = vec![(current_dist, current_best)];
        let mut c_idx = 0;

        while c_idx < candidates.len() {
            let (c_dist, c_node) = candidates[c_idx];
            c_idx += 1;

            // If furthest in W is closer than current, stop
            if !w.is_empty() {
                let w_max = w.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
                if c_dist > w_max && w.len() >= search_ef {
                    break;
                }
            }

            if self.nodes[c_node].neighbors.is_empty() {
                continue;
            }
            for &neighbor_id in &self.nodes[c_node].neighbors[0] {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);
                let neighbor_dist = self.compute_distance(query, &self.nodes[neighbor_id].vector);

                let w_max = if !w.is_empty() {
                    w.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max)
                } else {
                    f32::INFINITY
                };

                if neighbor_dist < w_max || w.len() < search_ef {
                    candidates.push((neighbor_dist, neighbor_id));
                    w.push((neighbor_dist, neighbor_id));
                    if w.len() > search_ef {
                        // Remove furthest
                        if let Some(max_pos) = w
                            .iter()
                            .enumerate()
                            .max_by(|a, b| {
                                a.1 .0
                                    .partial_cmp(&b.1 .0)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                        {
                            w.remove(max_pos);
                        }
                    }
                }
            }
        }

        // Sort w by distance and return top k
        w.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        w.truncate(k);

        Ok(w.into_iter().map(|(dist, id)| (id, dist)).collect())
    }

    pub(crate) fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_metric {
            GpuDistanceMetric::Cosine | GpuDistanceMetric::CosineF16 => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - dot / (norm_a * norm_b)
                }
            }
            GpuDistanceMetric::Euclidean | GpuDistanceMetric::EuclideanF16 => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            GpuDistanceMetric::InnerProduct => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                -dot // Negate so lower = more similar
            }
        }
    }
}
