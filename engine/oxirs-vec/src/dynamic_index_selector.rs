//! Dynamic index selection for optimal query performance
//!
//! This module provides runtime index selection based on query characteristics,
//! automatically choosing the best index type (HNSW, NSG, IVF, PQ, etc.) for each query.
//!
//! # Features
//!
//! - **Automatic Strategy Selection**: Uses cost-based query planning
//! - **Multiple Index Support**: Maintains multiple indices for different use cases
//! - **Performance Learning**: Tracks actual performance to improve future selections
//! - **Adaptive Parameters**: Automatically tunes parameters based on query requirements
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_vec::dynamic_index_selector::{DynamicIndexSelector, IndexSelectorConfig};
//! use oxirs_vec::{Vector, VectorIndex};
//!
//! let config = IndexSelectorConfig::default();
//! let mut selector = DynamicIndexSelector::new(config).unwrap();
//!
//! // Add vectors - they'll be indexed in all configured indices
//! for i in 0..1000 {
//!     let vec = Vector::new(vec![i as f32, (i * 2) as f32]);
//!     selector.add(format!("vec_{}", i), vec).unwrap();
//! }
//!
//! // Build all indices
//! selector.build().unwrap();
//!
//! // Search - automatically selects best index
//! let query = Vector::new(vec![500.0, 1000.0]);
//! let results = selector.search_knn(&query, 10).unwrap();
//! ```

use crate::query_planning::*;
use crate::{hnsw::HnswIndex, ivf::IvfIndex, lsh::LshIndex, nsg::NsgIndex};
use crate::{Vector, VectorIndex};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Configuration for dynamic index selector
#[derive(Debug, Clone)]
pub struct IndexSelectorConfig {
    /// Enable HNSW index
    pub enable_hnsw: bool,
    /// Enable NSG index
    pub enable_nsg: bool,
    /// Enable IVF index
    pub enable_ivf: bool,
    /// Enable LSH index
    pub enable_lsh: bool,
    /// Minimum recall requirement (0.0 to 1.0)
    pub min_recall: f32,
    /// Maximum acceptable latency (milliseconds)
    pub max_latency_ms: f64,
    /// Enable performance learning
    pub enable_learning: bool,
    /// Build all indices immediately
    pub eager_build: bool,
}

impl Default for IndexSelectorConfig {
    fn default() -> Self {
        Self {
            enable_hnsw: true,
            enable_nsg: true,
            enable_ivf: true,
            enable_lsh: false, // LSH is less commonly used
            min_recall: 0.90,
            max_latency_ms: 100.0,
            enable_learning: true,
            eager_build: true,
        }
    }
}

/// Dynamic index selector with multiple index backends
pub struct DynamicIndexSelector {
    config: IndexSelectorConfig,
    hnsw_index: Option<HnswIndex>,
    nsg_index: Option<NsgIndex>,
    ivf_index: Option<IvfIndex>,
    lsh_index: Option<LshIndex>,
    query_planner: Arc<RwLock<QueryPlanner>>,
    data: Vec<(String, Vector)>,
    is_built: bool,
    performance_stats: Arc<RwLock<PerformanceStats>>,
}

/// Performance statistics for learning
#[derive(Debug, Clone, Default)]
struct PerformanceStats {
    strategy_latencies: HashMap<QueryStrategy, Vec<f64>>,
    strategy_recalls: HashMap<QueryStrategy, Vec<f32>>,
    total_queries: usize,
}

impl PerformanceStats {
    fn record(&mut self, strategy: QueryStrategy, latency_ms: f64, recall: f32) {
        self.strategy_latencies
            .entry(strategy)
            .or_default()
            .push(latency_ms);

        self.strategy_recalls
            .entry(strategy)
            .or_default()
            .push(recall);

        self.total_queries += 1;
    }

    fn avg_latency(&self, strategy: QueryStrategy) -> Option<f64> {
        self.strategy_latencies
            .get(&strategy)
            .and_then(|latencies| {
                if latencies.is_empty() {
                    None
                } else {
                    Some(latencies.iter().sum::<f64>() / latencies.len() as f64)
                }
            })
    }

    fn avg_recall(&self, strategy: QueryStrategy) -> Option<f32> {
        self.strategy_recalls.get(&strategy).and_then(|recalls| {
            if recalls.is_empty() {
                None
            } else {
                Some(recalls.iter().sum::<f32>() / recalls.len() as f32)
            }
        })
    }
}

impl DynamicIndexSelector {
    /// Create a new dynamic index selector
    pub fn new(config: IndexSelectorConfig) -> Result<Self> {
        // Determine available indices based on config
        let mut available_indices = Vec::new();
        if config.enable_hnsw {
            available_indices.push(QueryStrategy::HnswApproximate);
        }
        if config.enable_nsg {
            available_indices.push(QueryStrategy::NsgApproximate);
        }
        if config.enable_ivf {
            available_indices.push(QueryStrategy::IvfCoarse);
        }
        if config.enable_lsh {
            available_indices.push(QueryStrategy::LocalitySensitiveHashing);
        }

        if available_indices.is_empty() {
            return Err(anyhow::anyhow!("At least one index type must be enabled"));
        }

        // Create initial index statistics
        let index_stats = IndexStatistics {
            vector_count: 0,
            dimensions: 0,
            available_indices,
            avg_latencies: HashMap::new(),
            avg_recalls: HashMap::new(),
        };

        let cost_model = CostModel::default();
        let query_planner = Arc::new(RwLock::new(QueryPlanner::new(cost_model, index_stats)));

        Ok(Self {
            config,
            hnsw_index: None,
            nsg_index: None,
            ivf_index: None,
            lsh_index: None,
            query_planner,
            data: Vec::new(),
            is_built: false,
            performance_stats: Arc::new(RwLock::new(PerformanceStats::default())),
        })
    }

    /// Add a vector to all enabled indices
    pub fn add(&mut self, uri: String, vector: Vector) -> Result<()> {
        if self.is_built && self.config.eager_build {
            return Err(anyhow::anyhow!(
                "Cannot add vectors after indices are built in eager mode"
            ));
        }

        self.data.push((uri, vector));
        Ok(())
    }

    /// Build all enabled indices
    pub fn build(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Err(anyhow::anyhow!("No vectors to index"));
        }

        let dimensions = self.data[0].1.dimensions;
        let vector_count = self.data.len();

        info!(
            "Building dynamic index selector with {} vectors, {} dimensions",
            vector_count, dimensions
        );

        // Build HNSW index
        if self.config.enable_hnsw {
            debug!("Building HNSW index");
            let mut hnsw = HnswIndex::new(Default::default())?;
            for (uri, vec) in &self.data {
                hnsw.insert(uri.clone(), vec.clone())?;
            }
            self.hnsw_index = Some(hnsw);
        }

        // Build NSG index
        if self.config.enable_nsg {
            debug!("Building NSG index");
            let mut nsg = NsgIndex::new(Default::default())?;
            for (uri, vec) in &self.data {
                nsg.insert(uri.clone(), vec.clone())?;
            }
            nsg.build()?;
            self.nsg_index = Some(nsg);
        }

        // Build IVF index
        if self.config.enable_ivf {
            debug!("Building IVF index");
            let mut ivf = IvfIndex::new(Default::default())?;
            for (uri, vec) in &self.data {
                ivf.insert(uri.clone(), vec.clone())?;
            }
            // IVF trains clusters automatically during insertion
            self.ivf_index = Some(ivf);
        }

        // Build LSH index
        if self.config.enable_lsh {
            debug!("Building LSH index");
            let lsh = LshIndex::new(Default::default());
            let mut lsh_mut = lsh;
            for (uri, vec) in &self.data {
                lsh_mut.insert(uri.clone(), vec.clone())?;
            }
            self.lsh_index = Some(lsh_mut);
        }

        // Update query planner statistics
        let mut planner = self
            .query_planner
            .write()
            .expect("query_planner write lock should not be poisoned");
        planner.update_index_metadata(vector_count, dimensions);

        self.is_built = true;

        info!("Dynamic index selector built successfully");

        Ok(())
    }

    /// Search with automatic index selection
    pub fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        if !self.is_built {
            return Err(anyhow::anyhow!("Indices not built. Call build() first."));
        }

        // Create query characteristics
        let query_chars = QueryCharacteristics {
            k,
            dimensions: query.dimensions,
            min_recall: self.config.min_recall,
            max_latency_ms: self.config.max_latency_ms,
            query_type: VectorQueryType::Single,
        };

        // Get query plan
        let planner = self
            .query_planner
            .read()
            .expect("query_planner read lock should not be poisoned");
        let plan = planner.plan(&query_chars)?;
        drop(planner); // Release read lock

        debug!(
            "Selected strategy: {:?} (estimated cost: {:.2} µs, recall: {:.2})",
            plan.strategy, plan.estimated_cost_us, plan.estimated_recall
        );

        // Execute query using selected strategy
        let start = std::time::Instant::now();
        let results = self.execute_strategy(plan.strategy, query, k)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms

        // Record performance if learning is enabled
        if self.config.enable_learning {
            let mut stats = self
                .performance_stats
                .write()
                .expect("performance_stats write lock should not be poisoned");
            stats.record(plan.strategy, elapsed, plan.estimated_recall);
            drop(stats);

            // Update query planner with actual performance
            let mut planner = self
                .query_planner
                .write()
                .expect("query_planner write lock should not be poisoned");
            if let Some(avg_latency) = self
                .performance_stats
                .read()
                .expect("performance_stats read lock should not be poisoned")
                .avg_latency(plan.strategy)
            {
                planner.update_statistics(plan.strategy, avg_latency, plan.estimated_recall);
            }
        }

        Ok(results)
    }

    /// Execute query using specific strategy
    fn execute_strategy(
        &self,
        strategy: QueryStrategy,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        match strategy {
            QueryStrategy::HnswApproximate => {
                if let Some(ref index) = self.hnsw_index {
                    index.search_knn(query, k)
                } else {
                    Err(anyhow::anyhow!("HNSW index not available"))
                }
            }
            QueryStrategy::NsgApproximate => {
                if let Some(ref index) = self.nsg_index {
                    index.search_knn(query, k)
                } else {
                    Err(anyhow::anyhow!("NSG index not available"))
                }
            }
            QueryStrategy::IvfCoarse => {
                if let Some(ref index) = self.ivf_index {
                    index.search_knn(query, k)
                } else {
                    Err(anyhow::anyhow!("IVF index not available"))
                }
            }
            QueryStrategy::LocalitySensitiveHashing => {
                if let Some(ref index) = self.lsh_index {
                    index.search_knn(query, k)
                } else {
                    Err(anyhow::anyhow!("LSH index not available"))
                }
            }
            _ => Err(anyhow::anyhow!(
                "Strategy {:?} not supported by dynamic selector",
                strategy
            )),
        }
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        let perf_stats = self
            .performance_stats
            .read()
            .expect("performance_stats read lock should not be poisoned");

        stats.insert(
            "total_queries".to_string(),
            perf_stats.total_queries.to_string(),
        );
        stats.insert("vector_count".to_string(), self.data.len().to_string());
        stats.insert("is_built".to_string(), self.is_built.to_string());

        // Add per-strategy stats
        for strategy in &[
            QueryStrategy::HnswApproximate,
            QueryStrategy::NsgApproximate,
            QueryStrategy::IvfCoarse,
            QueryStrategy::LocalitySensitiveHashing,
        ] {
            if let Some(avg_lat) = perf_stats.avg_latency(*strategy) {
                stats.insert(
                    format!("{:?}_avg_latency_ms", strategy),
                    format!("{:.2}", avg_lat),
                );
            }
            if let Some(avg_rec) = perf_stats.avg_recall(*strategy) {
                stats.insert(
                    format!("{:?}_avg_recall", strategy),
                    format!("{:.2}", avg_rec),
                );
            }
        }

        stats
    }

    /// Check if indices are built
    pub fn is_built(&self) -> bool {
        self.is_built
    }

    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_selector_creation() {
        let config = IndexSelectorConfig::default();
        let selector = DynamicIndexSelector::new(config);
        assert!(selector.is_ok());
    }

    #[test]
    fn test_add_vectors() {
        let config = IndexSelectorConfig::default();
        let mut selector = DynamicIndexSelector::new(config).unwrap();

        for i in 0..10 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32]);
            selector.add(format!("vec_{}", i), vec).unwrap();
        }

        assert_eq!(selector.len(), 10);
    }

    #[test]
    fn test_build_and_search() {
        let config = IndexSelectorConfig {
            enable_hnsw: true,
            enable_nsg: true,
            enable_ivf: false, // Disable IVF to speed up test
            enable_lsh: false,
            ..Default::default()
        };
        let mut selector = DynamicIndexSelector::new(config).unwrap();

        // Add test vectors
        for i in 0..50 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            selector.add(format!("vec_{}", i), vec).unwrap();
        }

        // Build indices
        selector.build().unwrap();
        assert!(selector.is_built());

        // Search
        let query = Vector::new(vec![25.0, 50.0, 75.0]);
        let results = selector.search_knn(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted by similarity
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    #[test]
    fn test_performance_learning() {
        let config = IndexSelectorConfig {
            enable_hnsw: true,
            enable_nsg: true,
            enable_ivf: false, // Disable IVF to avoid training requirement
            enable_lsh: false,
            enable_learning: true,
            ..Default::default()
        };
        let mut selector = DynamicIndexSelector::new(config).unwrap();

        // Add vectors
        for i in 0..30 {
            let vec = Vector::new(vec![i as f32, (i * 2) as f32]);
            selector.add(format!("vec_{}", i), vec).unwrap();
        }

        selector.build().unwrap();

        // Perform multiple searches to build up statistics
        for _ in 0..5 {
            let query = Vector::new(vec![15.0, 30.0]);
            let _ = selector.search_knn(&query, 5);
        }

        // Check that statistics were recorded
        let stats = selector.get_stats();
        assert!(stats.contains_key("total_queries"));
        let total_queries: usize = stats.get("total_queries").unwrap().parse().unwrap();
        assert!(total_queries >= 5);
    }
}
