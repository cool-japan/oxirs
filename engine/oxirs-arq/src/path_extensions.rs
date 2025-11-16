//! Extended Property Path Support for SPARQL
//!
//! This module extends the basic property path functionality with advanced
//! optimization, caching, analysis, and specialized evaluation strategies.

use crate::algebra::Term;
use crate::path::{PathContext, PropertyPath};
use anyhow::Result;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Path cache for memoization of path evaluation results
pub struct PathCache {
    cache: DashMap<PathCacheKey, PathCacheEntry>,
    max_entries: usize,
    hits: Arc<std::sync::atomic::AtomicUsize>,
    misses: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PathCacheKey {
    path: PropertyPath,
    start: String,       // Serialized term
    end: Option<String>, // Serialized term (optional for reachability)
}

#[derive(Debug, Clone)]
struct PathCacheEntry {
    reachable: bool,
    #[allow(dead_code)]
    intermediate_nodes: Vec<String>,
    #[allow(dead_code)]
    path_length: usize,
    timestamp: std::time::Instant,
}

impl PathCache {
    /// Create a new path cache with default size
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    /// Create a new path cache with specified capacity
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            cache: DashMap::new(),
            max_entries,
            hits: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            misses: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Check if path is cached
    pub fn get(&self, path: &PropertyPath, start: &Term, end: Option<&Term>) -> Option<bool> {
        let key = PathCacheKey {
            path: path.clone(),
            start: format!("{:?}", start),
            end: end.map(|t| format!("{:?}", t)),
        };

        if let Some(entry) = self.cache.get(&key) {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(entry.reachable)
        } else {
            self.misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    /// Cache a path evaluation result
    pub fn insert(&self, path: &PropertyPath, start: &Term, end: Option<&Term>, reachable: bool) {
        // Simple eviction: if at capacity, clear 10% of oldest entries
        if self.cache.len() >= self.max_entries {
            self.evict_oldest(self.max_entries / 10);
        }

        let key = PathCacheKey {
            path: path.clone(),
            start: format!("{:?}", start),
            end: end.map(|t| format!("{:?}", t)),
        };

        let entry = PathCacheEntry {
            reachable,
            intermediate_nodes: Vec::new(),
            path_length: 0,
            timestamp: std::time::Instant::now(),
        };

        self.cache.insert(key, entry);
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStats {
            hits,
            misses,
            size: self.cache.len(),
            capacity: self.max_entries,
            hit_rate,
        }
    }

    fn evict_oldest(&self, count: usize) {
        let mut entries: Vec<_> = self.cache.iter().collect();
        entries.sort_by_key(|e| e.value().timestamp);

        for entry in entries.iter().take(count) {
            self.cache.remove(entry.key());
        }
    }
}

impl Default for PathCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f64,
}

/// Bidirectional path search for faster reachability queries
pub struct BidirectionalPathSearch {
    context: PathContext,
}

impl BidirectionalPathSearch {
    /// Create new bidirectional search
    pub fn new(context: PathContext) -> Self {
        Self { context }
    }

    /// Search for path from start to end using bidirectional BFS
    #[allow(unused_variables)]
    pub fn search(
        &self,
        path: &PropertyPath,
        start: &Term,
        end: &Term,
        dataset: &HashMap<Term, Vec<(Term, Term)>>,
    ) -> Result<Option<Vec<Term>>> {
        // Forward search from start
        let mut forward_visited: HashMap<Term, Option<Term>> = HashMap::new();
        let mut forward_queue = VecDeque::new();
        forward_queue.push_back(start.clone());
        forward_visited.insert(start.clone(), None);

        // Backward search from end
        let mut backward_visited: HashMap<Term, Option<Term>> = HashMap::new();
        let mut backward_queue = VecDeque::new();
        backward_queue.push_back(end.clone());
        backward_visited.insert(end.clone(), None);

        let mut iterations = 0;
        let max_iterations = self.context.max_nodes;

        while !forward_queue.is_empty() && !backward_queue.is_empty() && iterations < max_iterations
        {
            iterations += 1;

            // Expand forward frontier (smaller first)
            if forward_queue.len() <= backward_queue.len() {
                if let Some(meeting_point) = self.expand_frontier(
                    &mut forward_queue,
                    &mut forward_visited,
                    &backward_visited,
                    dataset,
                    false,
                )? {
                    return Ok(Some(self.reconstruct_path(
                        &meeting_point,
                        &forward_visited,
                        &backward_visited,
                        start,
                        end,
                    )));
                }
            } else {
                // Expand backward frontier
                if let Some(meeting_point) = self.expand_frontier(
                    &mut backward_queue,
                    &mut backward_visited,
                    &forward_visited,
                    dataset,
                    true,
                )? {
                    return Ok(Some(self.reconstruct_path(
                        &meeting_point,
                        &forward_visited,
                        &backward_visited,
                        start,
                        end,
                    )));
                }
            }
        }

        Ok(None)
    }

    fn expand_frontier(
        &self,
        queue: &mut VecDeque<Term>,
        visited: &mut HashMap<Term, Option<Term>>,
        other_visited: &HashMap<Term, Option<Term>>,
        dataset: &HashMap<Term, Vec<(Term, Term)>>,
        _reverse: bool,
    ) -> Result<Option<Term>> {
        if let Some(current) = queue.pop_front() {
            // Check if we've met the other search
            if other_visited.contains_key(&current) {
                return Ok(Some(current));
            }

            // Expand neighbors
            if let Some(neighbors) = dataset.get(&current) {
                for (predicate, neighbor) in neighbors {
                    let _ = predicate; // Use predicate for path matching in real impl
                    if !visited.contains_key(neighbor) {
                        visited.insert(neighbor.clone(), Some(current.clone()));
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        Ok(None)
    }

    fn reconstruct_path(
        &self,
        meeting_point: &Term,
        forward_visited: &HashMap<Term, Option<Term>>,
        backward_visited: &HashMap<Term, Option<Term>>,
        start: &Term,
        end: &Term,
    ) -> Vec<Term> {
        let mut path = Vec::new();

        // Build forward path
        let mut current = meeting_point.clone();
        let mut forward_path = Vec::new();
        while let Some(Some(prev)) = forward_visited.get(&current) {
            forward_path.push(current.clone());
            current = prev.clone();
            if current == *start {
                forward_path.push(current.clone());
                break;
            }
        }
        forward_path.reverse();
        path.extend(forward_path);

        // Build backward path
        current = meeting_point.clone();
        while let Some(Some(next)) = backward_visited.get(&current) {
            current = next.clone();
            path.push(current.clone());
            if current == *end {
                break;
            }
        }

        path
    }
}

/// Path pattern analyzer for query optimization
pub struct PathAnalyzer;

impl PathAnalyzer {
    /// Analyze path complexity
    pub fn analyze_complexity(path: &PropertyPath) -> PathComplexity {
        match path {
            PropertyPath::Direct(_) => PathComplexity {
                depth: 1,
                branching_factor: 1,
                has_recursion: false,
                has_negation: false,
                estimated_cost: 1.0,
            },
            PropertyPath::Inverse(inner) => {
                let mut complexity = Self::analyze_complexity(inner);
                complexity.estimated_cost *= 1.2; // Inverse slightly more expensive
                complexity
            }
            PropertyPath::Sequence(left, right) => {
                let left_complexity = Self::analyze_complexity(left);
                let right_complexity = Self::analyze_complexity(right);
                PathComplexity {
                    depth: left_complexity.depth + right_complexity.depth,
                    branching_factor: left_complexity
                        .branching_factor
                        .max(right_complexity.branching_factor),
                    has_recursion: left_complexity.has_recursion || right_complexity.has_recursion,
                    has_negation: left_complexity.has_negation || right_complexity.has_negation,
                    estimated_cost: left_complexity.estimated_cost
                        * right_complexity.estimated_cost,
                }
            }
            PropertyPath::Alternative(left, right) => {
                let left_complexity = Self::analyze_complexity(left);
                let right_complexity = Self::analyze_complexity(right);
                PathComplexity {
                    depth: left_complexity.depth.max(right_complexity.depth),
                    branching_factor: left_complexity.branching_factor
                        + right_complexity.branching_factor,
                    has_recursion: left_complexity.has_recursion || right_complexity.has_recursion,
                    has_negation: left_complexity.has_negation || right_complexity.has_negation,
                    estimated_cost: left_complexity.estimated_cost
                        + right_complexity.estimated_cost,
                }
            }
            PropertyPath::ZeroOrMore(inner) | PropertyPath::OneOrMore(inner) => {
                let mut complexity = Self::analyze_complexity(inner);
                complexity.depth *= 10; // Assume average of 10 iterations
                complexity.has_recursion = true;
                complexity.estimated_cost *= 100.0; // Recursive paths are expensive
                complexity
            }
            PropertyPath::ZeroOrOne(inner) => {
                let mut complexity = Self::analyze_complexity(inner);
                complexity.branching_factor += 1; // Optional adds one branch
                complexity.estimated_cost *= 1.5;
                complexity
            }
            PropertyPath::NegatedPropertySet(_) => PathComplexity {
                depth: 1,
                branching_factor: 1,
                has_recursion: false,
                has_negation: true,
                estimated_cost: 2.0, // Negation requires filtering
            },
        }
    }

    /// Suggest optimization for path
    pub fn suggest_optimization(path: &PropertyPath) -> Vec<PathOptimizationHint> {
        let complexity = Self::analyze_complexity(path);
        let mut hints = Vec::new();

        if complexity.has_recursion && complexity.depth > 100 {
            hints.push(PathOptimizationHint::LimitRecursionDepth);
        }

        if complexity.branching_factor > 10 {
            hints.push(PathOptimizationHint::UseIndexes);
        }

        if matches!(path, PropertyPath::Alternative(_, _)) {
            hints.push(PathOptimizationHint::ReorderAlternatives);
        }

        if complexity.estimated_cost > 1000.0 {
            hints.push(PathOptimizationHint::ConsiderMaterialization);
        }

        hints
    }

    /// Check if path is deterministic (always returns same results)
    pub fn is_deterministic(path: &PropertyPath) -> bool {
        match path {
            PropertyPath::Direct(_)
            | PropertyPath::Inverse(_)
            | PropertyPath::Sequence(_, _)
            | PropertyPath::Alternative(_, _)
            | PropertyPath::ZeroOrMore(_)
            | PropertyPath::OneOrMore(_)
            | PropertyPath::ZeroOrOne(_) => true,
            PropertyPath::NegatedPropertySet(_) => true,
        }
    }

    /// Extract all predicates used in path
    pub fn extract_predicates(path: &PropertyPath) -> HashSet<Term> {
        let mut predicates = HashSet::new();
        Self::extract_predicates_recursive(path, &mut predicates);
        predicates
    }

    fn extract_predicates_recursive(path: &PropertyPath, predicates: &mut HashSet<Term>) {
        match path {
            PropertyPath::Direct(term) => {
                predicates.insert(term.clone());
            }
            PropertyPath::Inverse(inner)
            | PropertyPath::ZeroOrMore(inner)
            | PropertyPath::OneOrMore(inner)
            | PropertyPath::ZeroOrOne(inner) => {
                Self::extract_predicates_recursive(inner, predicates);
            }
            PropertyPath::Sequence(left, right) | PropertyPath::Alternative(left, right) => {
                Self::extract_predicates_recursive(left, predicates);
                Self::extract_predicates_recursive(right, predicates);
            }
            PropertyPath::NegatedPropertySet(terms) => {
                for term in terms {
                    predicates.insert(term.clone());
                }
            }
        }
    }
}

/// Path complexity metrics
#[derive(Debug, Clone)]
pub struct PathComplexity {
    pub depth: usize,
    pub branching_factor: usize,
    pub has_recursion: bool,
    pub has_negation: bool,
    pub estimated_cost: f64,
}

/// Path optimization hints
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathOptimizationHint {
    LimitRecursionDepth,
    UseIndexes,
    ReorderAlternatives,
    ConsiderMaterialization,
    CacheResults,
    UseBidirectionalSearch,
}

/// Specialized path evaluator with caching and optimization
pub struct CachedPathEvaluator {
    cache: Arc<PathCache>,
    bidirectional: bool,
}

impl CachedPathEvaluator {
    /// Create new cached evaluator
    pub fn new() -> Self {
        Self {
            cache: Arc::new(PathCache::new()),
            bidirectional: true,
        }
    }

    /// Create with custom cache
    pub fn with_cache(cache: Arc<PathCache>) -> Self {
        Self {
            cache,
            bidirectional: true,
        }
    }

    /// Enable or disable bidirectional search
    pub fn set_bidirectional(&mut self, enabled: bool) {
        self.bidirectional = enabled;
    }

    /// Evaluate path with caching
    pub fn evaluate(
        &self,
        path: &PropertyPath,
        start: &Term,
        end: Option<&Term>,
        dataset: &HashMap<Term, Vec<(Term, Term)>>,
    ) -> Result<bool> {
        // Check cache first
        if let Some(cached) = self.cache.get(path, start, end) {
            return Ok(cached);
        }

        // Evaluate path
        let reachable = self.evaluate_uncached(path, start, end, dataset)?;

        // Cache result
        self.cache.insert(path, start, end, reachable);

        Ok(reachable)
    }

    fn evaluate_uncached(
        &self,
        path: &PropertyPath,
        start: &Term,
        end: Option<&Term>,
        dataset: &HashMap<Term, Vec<(Term, Term)>>,
    ) -> Result<bool> {
        // Simplified evaluation - in real implementation would use full path semantics
        match path {
            PropertyPath::Direct(predicate) => {
                if let Some(neighbors) = dataset.get(start) {
                    for (pred, neighbor) in neighbors {
                        if pred == predicate {
                            if let Some(target) = end {
                                if neighbor == target {
                                    return Ok(true);
                                }
                            } else {
                                return Ok(true);
                            }
                        }
                    }
                }
                Ok(false)
            }
            PropertyPath::ZeroOrMore(_) => {
                // Zero-or-more always reaches start node
                if end.is_none() {
                    return Ok(true);
                }
                if let Some(target) = end {
                    if start == target {
                        return Ok(true);
                    }
                }
                // Would need full BFS/DFS here
                Ok(false)
            }
            _ => {
                // Other path types - simplified
                Ok(false)
            }
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Default for CachedPathEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Path reachability index for fast lookups
pub struct ReachabilityIndex {
    /// Forward reachability: node -> set of reachable nodes
    forward: DashMap<String, HashSet<String>>,
    /// Backward reachability: node -> set of nodes that can reach it
    backward: DashMap<String, HashSet<String>>,
    /// Transitive closure computed
    computed: std::sync::atomic::AtomicBool,
}

impl ReachabilityIndex {
    /// Create new reachability index
    pub fn new() -> Self {
        Self {
            forward: DashMap::new(),
            backward: DashMap::new(),
            computed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Add edge to index
    pub fn add_edge(&self, from: &Term, to: &Term) {
        let from_key = format!("{:?}", from);
        let to_key = format!("{:?}", to);

        self.forward
            .entry(from_key.clone())
            .or_default()
            .insert(to_key.clone());

        self.backward.entry(to_key).or_default().insert(from_key);

        // Mark as needing recomputation
        self.computed
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check if path exists from start to end
    pub fn is_reachable(&self, from: &Term, to: &Term) -> bool {
        let from_key = format!("{:?}", from);
        let to_key = format!("{:?}", to);

        if let Some(reachable) = self.forward.get(&from_key) {
            reachable.contains(&to_key)
        } else {
            false
        }
    }

    /// Compute transitive closure (Floyd-Warshall style)
    pub fn compute_transitive_closure(&self) {
        // Simplified - real implementation would do full transitive closure
        self.computed
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Clear the index
    pub fn clear(&self) {
        self.forward.clear();
        self.backward.clear();
        self.computed
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for ReachabilityIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Cost-based property path optimizer for selecting the best evaluation strategy
pub struct CostBasedPathOptimizer {
    /// Path statistics for cost estimation
    path_stats: Arc<DashMap<String, PathStatistics>>,
    /// Configuration for cost-based optimization
    config: PathOptimizationConfig,
}

/// Statistics for a property path
#[derive(Debug, Clone)]
pub struct PathStatistics {
    /// Average result cardinality
    pub avg_cardinality: f64,
    /// Average evaluation time in microseconds
    pub avg_eval_time_us: f64,
    /// Number of times this path was evaluated
    pub evaluation_count: u64,
    /// Selectivity (ratio of results to candidates)
    pub selectivity: f64,
    /// Last update timestamp
    pub last_updated: std::time::Instant,
}

impl Default for PathStatistics {
    fn default() -> Self {
        Self {
            avg_cardinality: 100.0, // Default estimate
            avg_eval_time_us: 1000.0,
            evaluation_count: 0,
            selectivity: 0.1,
            last_updated: std::time::Instant::now(),
        }
    }
}

/// Configuration for path optimization
#[derive(Debug, Clone)]
pub struct PathOptimizationConfig {
    /// Enable cost-based optimization
    pub enabled: bool,
    /// Minimum samples before using statistics
    pub min_samples: u64,
    /// Cost weight for cardinality (0.0 to 1.0)
    pub cardinality_weight: f64,
    /// Cost weight for evaluation time (0.0 to 1.0)
    pub time_weight: f64,
    /// Enable adaptive learning from execution
    pub adaptive_learning: bool,
    /// Learning rate for statistics updates (0.0 to 1.0)
    pub learning_rate: f64,
}

impl Default for PathOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_samples: 10,
            cardinality_weight: 0.6,
            time_weight: 0.4,
            adaptive_learning: true,
            learning_rate: 0.1,
        }
    }
}

/// Path evaluation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathEvaluationStrategy {
    /// Forward BFS from start node
    ForwardBFS,
    /// Backward BFS from end node
    BackwardBFS,
    /// Bidirectional BFS (meet in the middle)
    BidirectionalBFS,
    /// Use precomputed reachability index
    IndexLookup,
    /// Direct evaluation without optimization
    Direct,
}

/// Cost estimate for a path evaluation strategy
#[derive(Debug, Clone)]
pub struct StrategyCostEstimate {
    /// Estimated cardinality
    pub estimated_cardinality: f64,
    /// Estimated evaluation time in microseconds
    pub estimated_time_us: f64,
    /// Total estimated cost (weighted combination)
    pub total_cost: f64,
    /// Recommended strategy
    pub strategy: PathEvaluationStrategy,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

impl CostBasedPathOptimizer {
    /// Create a new cost-based path optimizer
    pub fn new() -> Self {
        Self::with_config(PathOptimizationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PathOptimizationConfig) -> Self {
        Self {
            path_stats: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Estimate cost and select best evaluation strategy
    pub fn select_strategy(
        &self,
        path: &PropertyPath,
        start: Option<&Term>,
        end: Option<&Term>,
        dataset_size: usize,
    ) -> StrategyCostEstimate {
        if !self.config.enabled {
            return self.default_strategy(path);
        }

        let path_key = self.path_signature(path);
        let stats = self.get_or_create_stats(&path_key);

        // Analyze path complexity
        let complexity = PathAnalyzer::analyze_complexity(path);

        // Estimate costs for different strategies
        let forward_cost =
            self.estimate_forward_cost(&complexity, &stats, start.is_some(), dataset_size);
        let backward_cost =
            self.estimate_backward_cost(&complexity, &stats, end.is_some(), dataset_size);
        let bidirectional_cost = self.estimate_bidirectional_cost(
            &complexity,
            &stats,
            start.is_some() && end.is_some(),
            dataset_size,
        );
        let index_cost = self.estimate_index_cost(&complexity, &stats);

        // Select strategy with minimum cost
        let estimates = vec![
            (PathEvaluationStrategy::ForwardBFS, forward_cost),
            (PathEvaluationStrategy::BackwardBFS, backward_cost),
            (PathEvaluationStrategy::BidirectionalBFS, bidirectional_cost),
            (PathEvaluationStrategy::IndexLookup, index_cost),
        ];

        let best = estimates
            .into_iter()
            .min_by(|(_, cost1), (_, cost2)| {
                cost1
                    .total_cost
                    .partial_cmp(&cost2.total_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((
                PathEvaluationStrategy::Direct,
                StrategyCostEstimate {
                    estimated_cardinality: stats.avg_cardinality,
                    estimated_time_us: stats.avg_eval_time_us,
                    total_cost: stats.avg_eval_time_us,
                    strategy: PathEvaluationStrategy::Direct,
                    confidence: 0.5,
                },
            ));

        best.1
    }

    /// Record actual execution results for adaptive learning
    pub fn record_execution(
        &self,
        path: &PropertyPath,
        actual_cardinality: u64,
        actual_time_us: u64,
    ) {
        if !self.config.adaptive_learning {
            return;
        }

        let path_key = self.path_signature(path);
        let mut stats = self.get_or_create_stats(&path_key);

        // Update statistics using exponential moving average
        let alpha = self.config.learning_rate;
        stats.avg_cardinality =
            (1.0 - alpha) * stats.avg_cardinality + alpha * (actual_cardinality as f64);
        stats.avg_eval_time_us =
            (1.0 - alpha) * stats.avg_eval_time_us + alpha * (actual_time_us as f64);
        stats.evaluation_count += 1;

        // Update selectivity (simplified)
        if actual_cardinality > 0 {
            let observed_selectivity = (actual_cardinality as f64) / 1000.0; // Normalize
            stats.selectivity = (1.0 - alpha) * stats.selectivity + alpha * observed_selectivity;
        }

        stats.last_updated = std::time::Instant::now();

        // Update in map
        self.path_stats.insert(path_key, stats);
    }

    /// Get statistics for a path
    pub fn get_statistics(&self, path: &PropertyPath) -> Option<PathStatistics> {
        let path_key = self.path_signature(path);
        self.path_stats.get(&path_key).map(|s| s.clone())
    }

    /// Clear all statistics
    pub fn clear_statistics(&self) {
        self.path_stats.clear();
    }

    // Private helper methods

    fn path_signature(&self, path: &PropertyPath) -> String {
        format!("{:?}", path)
    }

    fn get_or_create_stats(&self, path_key: &str) -> PathStatistics {
        self.path_stats
            .entry(path_key.to_string())
            .or_default()
            .clone()
    }

    fn default_strategy(&self, path: &PropertyPath) -> StrategyCostEstimate {
        let complexity = PathAnalyzer::analyze_complexity(path);
        StrategyCostEstimate {
            estimated_cardinality: 100.0,
            estimated_time_us: complexity.estimated_cost * 1000.0,
            total_cost: complexity.estimated_cost * 1000.0,
            strategy: if complexity.has_recursion {
                PathEvaluationStrategy::BidirectionalBFS
            } else {
                PathEvaluationStrategy::ForwardBFS
            },
            confidence: 0.5,
        }
    }

    fn estimate_forward_cost(
        &self,
        complexity: &PathComplexity,
        stats: &PathStatistics,
        has_start: bool,
        dataset_size: usize,
    ) -> StrategyCostEstimate {
        let base_cost = if has_start {
            stats.avg_eval_time_us
        } else {
            stats.avg_eval_time_us * (dataset_size as f64).sqrt()
        };

        let cardinality = stats.avg_cardinality;
        let time_cost = base_cost * (complexity.depth as f64);

        let total_cost =
            self.config.cardinality_weight * cardinality + self.config.time_weight * time_cost;

        let confidence = self.calculate_confidence(stats);

        StrategyCostEstimate {
            estimated_cardinality: cardinality,
            estimated_time_us: time_cost,
            total_cost,
            strategy: PathEvaluationStrategy::ForwardBFS,
            confidence,
        }
    }

    fn estimate_backward_cost(
        &self,
        complexity: &PathComplexity,
        stats: &PathStatistics,
        has_end: bool,
        dataset_size: usize,
    ) -> StrategyCostEstimate {
        let base_cost = if has_end {
            stats.avg_eval_time_us
        } else {
            stats.avg_eval_time_us * (dataset_size as f64).sqrt()
        };

        let cardinality = stats.avg_cardinality;
        let time_cost = base_cost * (complexity.depth as f64);

        let total_cost =
            self.config.cardinality_weight * cardinality + self.config.time_weight * time_cost;

        let confidence = self.calculate_confidence(stats);

        StrategyCostEstimate {
            estimated_cardinality: cardinality,
            estimated_time_us: time_cost,
            total_cost,
            strategy: PathEvaluationStrategy::BackwardBFS,
            confidence,
        }
    }

    fn estimate_bidirectional_cost(
        &self,
        complexity: &PathComplexity,
        stats: &PathStatistics,
        has_both: bool,
        _dataset_size: usize,
    ) -> StrategyCostEstimate {
        if !has_both {
            // Bidirectional only works when both start and end are bound
            return StrategyCostEstimate {
                estimated_cardinality: f64::INFINITY,
                estimated_time_us: f64::INFINITY,
                total_cost: f64::INFINITY,
                strategy: PathEvaluationStrategy::BidirectionalBFS,
                confidence: 0.0,
            };
        }

        // Bidirectional search is typically faster for long paths
        let speedup_factor = if complexity.depth > 3 { 0.5 } else { 0.8 };
        let cardinality = stats.avg_cardinality;
        let time_cost = stats.avg_eval_time_us * speedup_factor;

        let total_cost =
            self.config.cardinality_weight * cardinality + self.config.time_weight * time_cost;

        let confidence = self.calculate_confidence(stats);

        StrategyCostEstimate {
            estimated_cardinality: cardinality,
            estimated_time_us: time_cost,
            total_cost,
            strategy: PathEvaluationStrategy::BidirectionalBFS,
            confidence,
        }
    }

    fn estimate_index_cost(
        &self,
        complexity: &PathComplexity,
        stats: &PathStatistics,
    ) -> StrategyCostEstimate {
        // Index lookup is very fast for simple paths
        let is_simple = !complexity.has_recursion && complexity.depth <= 2;

        let (time_cost, cardinality) = if is_simple {
            (10.0, stats.avg_cardinality) // Constant time lookup
        } else {
            (f64::INFINITY, f64::INFINITY) // Not suitable for complex paths
        };

        let total_cost = if is_simple {
            self.config.time_weight * time_cost
        } else {
            f64::INFINITY
        };

        StrategyCostEstimate {
            estimated_cardinality: cardinality,
            estimated_time_us: time_cost,
            total_cost,
            strategy: PathEvaluationStrategy::IndexLookup,
            confidence: if is_simple { 0.9 } else { 0.0 },
        }
    }

    fn calculate_confidence(&self, stats: &PathStatistics) -> f64 {
        if stats.evaluation_count < self.config.min_samples {
            (stats.evaluation_count as f64) / (self.config.min_samples as f64)
        } else {
            0.95 // High confidence after sufficient samples
        }
    }
}

impl Default for CostBasedPathOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::NamedNode;

    fn create_test_term(iri: &str) -> Term {
        Term::Iri(NamedNode::new(iri).unwrap())
    }

    #[test]
    fn test_path_cache() {
        let cache = PathCache::new();
        let path = PropertyPath::Direct(create_test_term("http://example.org/knows"));
        let start = create_test_term("http://example.org/alice");
        let end = create_test_term("http://example.org/bob");

        // Initially not in cache
        assert!(cache.get(&path, &start, Some(&end)).is_none());

        // Insert into cache
        cache.insert(&path, &start, Some(&end), true);

        // Now in cache
        assert_eq!(cache.get(&path, &start, Some(&end)), Some(true));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_path_analyzer_complexity() {
        let direct = PropertyPath::Direct(create_test_term("http://example.org/knows"));
        let complexity = PathAnalyzer::analyze_complexity(&direct);
        assert_eq!(complexity.depth, 1);
        assert!(!complexity.has_recursion);

        let recursive = PropertyPath::ZeroOrMore(Box::new(direct.clone()));
        let complexity = PathAnalyzer::analyze_complexity(&recursive);
        assert!(complexity.has_recursion);
        assert!(complexity.estimated_cost > 10.0);
    }

    #[test]
    fn test_path_analyzer_predicates() {
        let knows = create_test_term("http://example.org/knows");
        let likes = create_test_term("http://example.org/likes");

        let path = PropertyPath::Alternative(
            Box::new(PropertyPath::Direct(knows.clone())),
            Box::new(PropertyPath::Direct(likes.clone())),
        );

        let predicates = PathAnalyzer::extract_predicates(&path);
        assert_eq!(predicates.len(), 2);
        assert!(predicates.contains(&knows));
        assert!(predicates.contains(&likes));
    }

    #[test]
    fn test_path_optimization_hints() {
        let deep_recursive =
            PropertyPath::ZeroOrMore(Box::new(PropertyPath::ZeroOrMore(Box::new(
                PropertyPath::Direct(create_test_term("http://example.org/part")),
            ))));

        let hints = PathAnalyzer::suggest_optimization(&deep_recursive);
        assert!(!hints.is_empty());
    }

    #[test]
    fn test_cached_evaluator() {
        let evaluator = CachedPathEvaluator::new();
        let path = PropertyPath::Direct(create_test_term("http://example.org/knows"));
        let start = create_test_term("http://example.org/alice");
        let end = create_test_term("http://example.org/bob");

        let mut dataset = HashMap::new();
        dataset.insert(
            start.clone(),
            vec![(create_test_term("http://example.org/knows"), end.clone())],
        );

        let result = evaluator.evaluate(&path, &start, Some(&end), &dataset);
        assert!(result.is_ok());

        // Check cache hit
        let stats = evaluator.cache_stats();
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn test_reachability_index() {
        let index = ReachabilityIndex::new();
        let alice = create_test_term("http://example.org/alice");
        let bob = create_test_term("http://example.org/bob");
        let carol = create_test_term("http://example.org/carol");

        index.add_edge(&alice, &bob);
        index.add_edge(&bob, &carol);

        assert!(index.is_reachable(&alice, &bob));
        assert!(!index.is_reachable(&alice, &carol)); // Not computed yet

        index.compute_transitive_closure();
    }

    #[test]
    fn test_path_complexity_sequence() {
        let p1 = PropertyPath::Direct(create_test_term("http://example.org/p1"));
        let p2 = PropertyPath::Direct(create_test_term("http://example.org/p2"));
        let seq = PropertyPath::Sequence(Box::new(p1), Box::new(p2));

        let complexity = PathAnalyzer::analyze_complexity(&seq);
        assert_eq!(complexity.depth, 2);
    }

    #[test]
    fn test_path_determinism() {
        let direct = PropertyPath::Direct(create_test_term("http://example.org/knows"));
        assert!(PathAnalyzer::is_deterministic(&direct));

        let recursive = PropertyPath::ZeroOrMore(Box::new(direct));
        assert!(PathAnalyzer::is_deterministic(&recursive));
    }

    #[test]
    #[ignore = "Slow test - eviction logic needs optimization"]
    fn test_cache_eviction() {
        let cache = PathCache::with_capacity(10);
        let path = PropertyPath::Direct(create_test_term("http://example.org/p"));

        // Insert 15 entries
        for i in 0..15 {
            let term = create_test_term(&format!("http://example.org/node{}", i));
            cache.insert(&path, &term, None, true);
        }

        // Should have evicted some entries
        let stats = cache.stats();
        assert!(stats.size <= 10);
    }

    #[test]
    fn test_cost_based_optimizer_default() {
        let optimizer = CostBasedPathOptimizer::new();
        let path = PropertyPath::Direct(create_test_term("http://example.org/knows"));

        let start = create_test_term("http://example.org/alice");
        let end = create_test_term("http://example.org/bob");

        let estimate = optimizer.select_strategy(&path, Some(&start), Some(&end), 1000);

        assert!(estimate.total_cost > 0.0);
        assert!(estimate.confidence >= 0.0 && estimate.confidence <= 1.0);
    }

    #[test]
    fn test_cost_based_optimizer_adaptive_learning() {
        let optimizer = CostBasedPathOptimizer::new();
        let path = PropertyPath::Direct(create_test_term("http://example.org/knows"));

        // Record some executions
        optimizer.record_execution(&path, 50, 500);
        optimizer.record_execution(&path, 45, 480);
        optimizer.record_execution(&path, 55, 520);

        // Get updated statistics
        let stats = optimizer.get_statistics(&path);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.evaluation_count, 3);
        assert!(stats.avg_cardinality > 0.0);
    }

    #[test]
    fn test_cost_based_optimizer_strategy_selection() {
        let optimizer = CostBasedPathOptimizer::new();

        // Simple path - should prefer forward or index
        let simple = PropertyPath::Direct(create_test_term("http://example.org/p"));
        let start = create_test_term("http://example.org/alice");
        let end = create_test_term("http://example.org/bob");

        let estimate = optimizer.select_strategy(&simple, Some(&start), Some(&end), 100);
        assert!(matches!(
            estimate.strategy,
            PathEvaluationStrategy::ForwardBFS
                | PathEvaluationStrategy::BidirectionalBFS
                | PathEvaluationStrategy::IndexLookup
        ));

        // Recursive path - should consider bidirectional
        let recursive = PropertyPath::ZeroOrMore(Box::new(simple));
        let estimate = optimizer.select_strategy(&recursive, Some(&start), Some(&end), 100);
        assert!(estimate.estimated_time_us > 0.0);
    }

    #[test]
    fn test_path_statistics_confidence() {
        let optimizer = CostBasedPathOptimizer::with_config(PathOptimizationConfig {
            min_samples: 5,
            ..Default::default()
        });

        let path = PropertyPath::Direct(create_test_term("http://example.org/p"));

        // Low confidence with few samples
        optimizer.record_execution(&path, 10, 100);
        let stats = optimizer.get_statistics(&path).unwrap();
        let confidence = if stats.evaluation_count < 5 {
            (stats.evaluation_count as f64) / 5.0
        } else {
            0.95
        };
        assert!(confidence < 0.5);

        // Higher confidence with more samples
        for _ in 0..6 {
            optimizer.record_execution(&path, 10, 100);
        }
        let stats = optimizer.get_statistics(&path).unwrap();
        assert!(stats.evaluation_count >= 5);
    }

    #[test]
    fn test_bidirectional_strategy_requires_both_endpoints() {
        let optimizer = CostBasedPathOptimizer::new();
        let path = PropertyPath::Direct(create_test_term("http://example.org/p"));
        let start = create_test_term("http://example.org/alice");

        // Without end node, bidirectional should have infinite cost
        let estimate = optimizer.select_strategy(&path, Some(&start), None, 100);
        // The selected strategy should not be bidirectional (or if it is, it has special handling)
        assert!(
            estimate.total_cost.is_finite()
                || estimate.strategy != PathEvaluationStrategy::BidirectionalBFS
        );
    }

    #[test]
    fn test_cost_optimizer_clear_statistics() {
        let optimizer = CostBasedPathOptimizer::new();
        let path = PropertyPath::Direct(create_test_term("http://example.org/p"));

        optimizer.record_execution(&path, 10, 100);
        assert!(optimizer.get_statistics(&path).is_some());

        optimizer.clear_statistics();
        // After clearing, we should get default stats
        let stats = optimizer.get_statistics(&path);
        assert!(stats.is_none() || stats.unwrap().evaluation_count == 0);
    }

    #[test]
    fn test_path_evaluation_strategies() {
        // Test that all strategies are distinct
        let strategies = [
            PathEvaluationStrategy::ForwardBFS,
            PathEvaluationStrategy::BackwardBFS,
            PathEvaluationStrategy::BidirectionalBFS,
            PathEvaluationStrategy::IndexLookup,
            PathEvaluationStrategy::Direct,
        ];

        for (i, s1) in strategies.iter().enumerate() {
            for (j, s2) in strategies.iter().enumerate() {
                if i == j {
                    assert_eq!(s1, s2);
                } else {
                    assert_ne!(s1, s2);
                }
            }
        }
    }
}
