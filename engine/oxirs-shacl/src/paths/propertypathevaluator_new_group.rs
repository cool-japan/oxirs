//! # PropertyPathEvaluator - new_group Methods
//!
//! This module contains method implementations for `PropertyPathEvaluator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::prelude::*;

impl PropertyPathEvaluator {
    /// Create a new property path evaluator
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            query_plan_cache: HashMap::new(),
            max_depth: 50,
            max_intermediate_results: 10000,
            stats: PathEvaluationStats::default(),
            cache_config: PathCacheConfig::default(),
        }
    }
    /// Create a new evaluator with custom limits
    pub fn with_limits(max_depth: usize, max_intermediate_results: usize) -> Self {
        Self {
            cache: HashMap::new(),
            query_plan_cache: HashMap::new(),
            max_depth,
            max_intermediate_results,
            stats: PathEvaluationStats::default(),
            cache_config: PathCacheConfig::default(),
        }
    }
    /// Create a new evaluator with custom cache configuration
    pub fn with_cache_config(cache_config: PathCacheConfig) -> Self {
        Self {
            cache: HashMap::new(),
            query_plan_cache: HashMap::new(),
            max_depth: 50,
            max_intermediate_results: 10000,
            stats: PathEvaluationStats::default(),
            cache_config,
        }
    }
    /// Evaluate a property path from a starting node
    pub fn evaluate_path(
        &mut self,
        store: &dyn Store,
        start_node: &Term,
        path: &PropertyPath,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let cache_key = self.create_cache_key(start_node, path, graph_name);
        if let Some(cached_result) = self.cache.get_mut(&cache_key) {
            if cached_result.is_fresh(self.cache_config.max_cache_age) {
                cached_result.access();
                self.stats.cache_hits += 1;
                return Ok(cached_result.values.clone());
            } else {
                self.cache.remove(&cache_key);
            }
        }
        self.stats.cache_misses += 1;
        let result = if path.can_use_sparql_path() {
            match self.evaluate_path_with_sparql(store, start_node, path, graph_name) {
                Ok(results) => results,
                Err(e) => {
                    tracing::debug!(
                        "SPARQL property path failed, falling back to programmatic evaluation: {}",
                        e
                    );
                    self.evaluate_path_impl(store, start_node, path, graph_name, 0)?
                }
            }
        } else {
            self.evaluate_path_impl(store, start_node, path, graph_name, 0)?
        };
        if self.should_cache_result(&result) {
            let cached_result = CachedPathResult::new(result.clone());
            self.manage_cache_size(cache_key, cached_result);
        }
        self.stats.total_evaluations += 1;
        self.stats.total_values_found += result.len();
        if self.stats.total_evaluations > 0 {
            self.stats.avg_values_per_evaluation =
                self.stats.total_values_found as f64 / self.stats.total_evaluations as f64;
        }
        Ok(result)
    }
    /// Evaluate multiple paths from the same starting node
    pub fn evaluate_multiple_paths(
        &mut self,
        store: &dyn Store,
        start_node: &Term,
        paths: &[PropertyPath],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut all_values = HashSet::new();
        for path in paths {
            let values = self.evaluate_path(store, start_node, path, graph_name)?;
            all_values.extend(values);
        }
        Ok(all_values.into_iter().collect())
    }
}
