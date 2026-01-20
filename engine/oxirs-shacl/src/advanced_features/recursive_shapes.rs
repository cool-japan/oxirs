//! SHACL Advanced Features - Recursive Shape Definitions
//!
//! Support for recursive and mutually recursive shape definitions.
//! Handles circular shape references with proper cycle detection and validation.
//!
//! Examples of recursive shapes:
//! - Tree structures (nodes with children of the same type)
//! - Linked lists
//! - Graph structures
//! - Organizational hierarchies

use crate::{Result, ShaclError, Shape, ShapeId};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Recursive shape validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveValidationConfig {
    /// Maximum recursion depth allowed
    pub max_depth: usize,
    /// Whether to detect and break cycles
    pub detect_cycles: bool,
    /// Whether to cache validation results for visited nodes
    pub cache_results: bool,
    /// Strategy for handling recursive validation
    pub strategy: RecursionStrategy,
}

impl Default for RecursiveValidationConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            detect_cycles: true,
            cache_results: true,
            strategy: RecursionStrategy::DepthFirst,
        }
    }
}

/// Strategy for recursive validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursionStrategy {
    /// Depth-first traversal
    DepthFirst,
    /// Breadth-first traversal
    BreadthFirst,
    /// Optimized strategy with cycle breaking
    OptimizedCycleBreaking,
}

/// Recursive shape validator
pub struct RecursiveShapeValidator {
    /// Configuration
    config: RecursiveValidationConfig,
    /// Visited nodes tracking (for cycle detection)
    visited_nodes: HashSet<(Term, ShapeId)>,
    /// Validation result cache
    result_cache: HashMap<(Term, ShapeId), bool>,
    /// Current recursion depth
    current_depth: usize,
    /// Statistics
    stats: RecursionStats,
}

impl RecursiveShapeValidator {
    /// Create a new recursive shape validator
    pub fn new(config: RecursiveValidationConfig) -> Self {
        Self {
            config,
            visited_nodes: HashSet::new(),
            current_depth: 0,
            result_cache: HashMap::new(),
            stats: RecursionStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(RecursiveValidationConfig::default())
    }

    /// Validate a node against a potentially recursive shape
    pub fn validate_recursive(
        &mut self,
        focus_node: &Term,
        shape: &Shape,
        store: &dyn Store,
        shape_resolver: &dyn ShapeResolver,
    ) -> Result<RecursiveValidationResult> {
        // Check depth limit
        if self.current_depth >= self.config.max_depth {
            return Err(ShaclError::RecursionLimit(format!(
                "Maximum recursion depth exceeded: {}",
                self.config.max_depth
            )));
        }

        // Check cycle detection
        let node_shape_key = (focus_node.clone(), shape.id.clone());
        if self.config.detect_cycles && self.visited_nodes.contains(&node_shape_key) {
            // Found a cycle - handle based on strategy
            self.stats.cycles_detected += 1;
            return self.handle_cycle(focus_node, shape);
        }

        // Check cache
        if self.config.cache_results {
            if let Some(&cached_result) = self.result_cache.get(&node_shape_key) {
                self.stats.cache_hits += 1;
                return Ok(RecursiveValidationResult {
                    conforms: cached_result,
                    depth_reached: self.current_depth,
                    cycles_detected: false,
                    cached: true,
                });
            }
            self.stats.cache_misses += 1;
        }

        // Mark node as visited
        self.visited_nodes.insert(node_shape_key.clone());
        self.current_depth += 1;
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(self.current_depth);

        // Perform validation based on strategy
        let result = match self.config.strategy {
            RecursionStrategy::DepthFirst => {
                self.validate_depth_first(focus_node, shape, store, shape_resolver)
            }
            RecursionStrategy::BreadthFirst => {
                self.validate_breadth_first(focus_node, shape, store, shape_resolver)
            }
            RecursionStrategy::OptimizedCycleBreaking => {
                self.validate_optimized(focus_node, shape, store, shape_resolver)
            }
        };

        // Restore state
        self.current_depth -= 1;
        self.visited_nodes.remove(&node_shape_key);

        // Cache result
        if self.config.cache_results {
            if let Ok(ref validation_result) = result {
                self.result_cache
                    .insert(node_shape_key, validation_result.conforms);
            }
        }

        self.stats.total_validations += 1;
        result
    }

    /// Handle cycle detection
    fn handle_cycle(
        &mut self,
        _focus_node: &Term,
        _shape: &Shape,
    ) -> Result<RecursiveValidationResult> {
        // For now, treat cycles as valid (they've been validated before)
        // This prevents infinite recursion while still validating the structure
        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: self.current_depth,
            cycles_detected: true,
            cached: false,
        })
    }

    /// Depth-first validation
    fn validate_depth_first(
        &mut self,
        _focus_node: &Term,
        _shape: &Shape,
        _store: &dyn Store,
        _shape_resolver: &dyn ShapeResolver,
    ) -> Result<RecursiveValidationResult> {
        // TODO: Implement actual depth-first validation
        // This would recursively validate all nested shapes
        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: self.current_depth,
            cycles_detected: false,
            cached: false,
        })
    }

    /// Breadth-first validation
    fn validate_breadth_first(
        &mut self,
        _focus_node: &Term,
        _shape: &Shape,
        _store: &dyn Store,
        _shape_resolver: &dyn ShapeResolver,
    ) -> Result<RecursiveValidationResult> {
        // TODO: Implement breadth-first validation
        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: self.current_depth,
            cycles_detected: false,
            cached: false,
        })
    }

    /// Optimized validation with cycle breaking
    fn validate_optimized(
        &mut self,
        _focus_node: &Term,
        _shape: &Shape,
        _store: &dyn Store,
        _shape_resolver: &dyn ShapeResolver,
    ) -> Result<RecursiveValidationResult> {
        // TODO: Implement optimized validation
        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: self.current_depth,
            cycles_detected: false,
            cached: false,
        })
    }

    /// Clear all caches and state
    pub fn reset(&mut self) {
        self.visited_nodes.clear();
        self.result_cache.clear();
        self.current_depth = 0;
        self.stats = RecursionStats::default();
    }

    /// Get statistics
    pub fn stats(&self) -> &RecursionStats {
        &self.stats
    }

    /// Clear only the cache, keep statistics
    pub fn clear_cache(&mut self) {
        self.result_cache.clear();
        self.visited_nodes.clear();
        self.current_depth = 0;
    }
}

/// Trait for resolving shape references
pub trait ShapeResolver {
    /// Resolve a shape by ID
    fn resolve_shape(&self, shape_id: &ShapeId) -> Option<&Shape>;

    /// Get all shapes that reference a given shape
    fn get_referencing_shapes(&self, shape_id: &ShapeId) -> Vec<&Shape>;

    /// Detect circular dependencies
    fn detect_circular_dependencies(&self) -> Vec<Vec<ShapeId>>;
}

/// Result of recursive validation
#[derive(Debug, Clone)]
pub struct RecursiveValidationResult {
    /// Whether validation succeeded
    pub conforms: bool,
    /// Maximum depth reached during validation
    pub depth_reached: usize,
    /// Whether cycles were detected
    pub cycles_detected: bool,
    /// Whether result was from cache
    pub cached: bool,
}

/// Statistics for recursive validation
#[derive(Debug, Clone, Default)]
pub struct RecursionStats {
    /// Total number of validations performed
    pub total_validations: usize,
    /// Number of cycles detected
    pub cycles_detected: usize,
    /// Maximum depth reached
    pub max_depth_reached: usize,
    /// Cache hit count
    pub cache_hits: usize,
    /// Cache miss count
    pub cache_misses: usize,
}

impl RecursionStats {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Shape dependency graph analyzer
pub struct ShapeDependencyAnalyzer {
    /// Dependency graph: shape -> shapes it depends on
    dependencies: HashMap<ShapeId, HashSet<ShapeId>>,
}

impl ShapeDependencyAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, from: ShapeId, to: ShapeId) {
        self.dependencies.entry(from).or_default().insert(to);
    }

    /// Build dependency graph from shapes
    pub fn build_from_shapes(&mut self, shapes: &[Shape]) {
        for shape in shapes {
            // Analyze shape for dependencies
            let deps = self.extract_dependencies(shape);
            for dep in deps {
                self.add_dependency(shape.id.clone(), dep);
            }
        }
    }

    /// Extract dependencies from a shape
    fn extract_dependencies(&self, shape: &Shape) -> Vec<ShapeId> {
        let deps = Vec::new();

        // TODO: Analyze shape constraints and extract shape references
        // For now, return empty vector
        tracing::debug!(
            "Extracting dependencies for shape {}: found {}",
            shape.id,
            deps.len()
        );

        deps
    }

    /// Detect circular dependencies using Tarjan's algorithm
    pub fn detect_cycles(&self) -> Vec<Vec<ShapeId>> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut current_path = Vec::new();

        for shape_id in self.dependencies.keys() {
            if !visited.contains(shape_id) {
                self.detect_cycles_dfs(
                    shape_id,
                    &mut visited,
                    &mut rec_stack,
                    &mut current_path,
                    &mut cycles,
                );
            }
        }

        cycles
    }

    /// DFS for cycle detection
    fn detect_cycles_dfs(
        &self,
        node: &ShapeId,
        visited: &mut HashSet<ShapeId>,
        rec_stack: &mut HashSet<ShapeId>,
        current_path: &mut Vec<ShapeId>,
        cycles: &mut Vec<Vec<ShapeId>>,
    ) {
        visited.insert(node.clone());
        rec_stack.insert(node.clone());
        current_path.push(node.clone());

        if let Some(deps) = self.dependencies.get(node) {
            for dep in deps {
                if !visited.contains(dep) {
                    self.detect_cycles_dfs(dep, visited, rec_stack, current_path, cycles);
                } else if rec_stack.contains(dep) {
                    // Found a cycle
                    if let Some(pos) = current_path.iter().position(|id| id == dep) {
                        let cycle = current_path[pos..].to_vec();
                        cycles.push(cycle);
                    }
                }
            }
        }

        current_path.pop();
        rec_stack.remove(node);
    }

    /// Get topological sort of shapes (if no cycles)
    pub fn topological_sort(&self) -> Result<Vec<ShapeId>> {
        let cycles = self.detect_cycles();
        if !cycles.is_empty() {
            return Err(ShaclError::ShapeValidation(format!(
                "Cannot perform topological sort: {} circular dependencies found",
                cycles.len()
            )));
        }

        // Perform topological sort using DFS
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();

        for shape_id in self.dependencies.keys() {
            if !visited.contains(shape_id) {
                self.topological_sort_dfs(shape_id, &mut visited, &mut sorted);
            }
        }

        sorted.reverse();
        Ok(sorted)
    }

    /// DFS for topological sort
    fn topological_sort_dfs(
        &self,
        node: &ShapeId,
        visited: &mut HashSet<ShapeId>,
        sorted: &mut Vec<ShapeId>,
    ) {
        visited.insert(node.clone());

        if let Some(deps) = self.dependencies.get(node) {
            for dep in deps {
                if !visited.contains(dep) {
                    self.topological_sort_dfs(dep, visited, sorted);
                }
            }
        }

        sorted.push(node.clone());
    }

    /// Get dependency depth for a shape (longest path to a leaf)
    pub fn get_dependency_depth(&self, shape_id: &ShapeId) -> usize {
        let mut visited = HashSet::new();
        self.get_dependency_depth_dfs(shape_id, &mut visited)
    }

    /// DFS for dependency depth calculation
    fn get_dependency_depth_dfs(&self, node: &ShapeId, visited: &mut HashSet<ShapeId>) -> usize {
        if visited.contains(node) {
            return 0; // Cycle detected, return 0
        }

        visited.insert(node.clone());

        let max_depth = if let Some(deps) = self.dependencies.get(node) {
            deps.iter()
                .map(|dep| self.get_dependency_depth_dfs(dep, visited))
                .max()
                .unwrap_or(0)
        } else {
            0
        };

        visited.remove(node);

        max_depth + 1
    }
}

impl Default for ShapeDependencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_validator_creation() {
        let validator = RecursiveShapeValidator::default_config();
        assert_eq!(validator.current_depth, 0);
    }

    #[test]
    fn test_recursion_config() {
        let config = RecursiveValidationConfig {
            max_depth: 50,
            detect_cycles: true,
            cache_results: true,
            strategy: RecursionStrategy::DepthFirst,
        };
        assert_eq!(config.max_depth, 50);
    }

    #[test]
    fn test_dependency_analyzer() {
        let mut analyzer = ShapeDependencyAnalyzer::new();
        analyzer.add_dependency(ShapeId::new("shape1"), ShapeId::new("shape2"));
        analyzer.add_dependency(ShapeId::new("shape2"), ShapeId::new("shape3"));

        let depth = analyzer.get_dependency_depth(&ShapeId::new("shape1"));
        assert!(depth > 0);
    }

    #[test]
    fn test_cycle_detection() {
        let mut analyzer = ShapeDependencyAnalyzer::new();
        analyzer.add_dependency(ShapeId::new("shape1"), ShapeId::new("shape2"));
        analyzer.add_dependency(ShapeId::new("shape2"), ShapeId::new("shape1"));

        let cycles = analyzer.detect_cycles();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_recursion_stats() {
        let stats = RecursionStats {
            total_validations: 100,
            cache_hits: 60,
            cache_misses: 40,
            ..Default::default()
        };
        assert_eq!(stats.cache_hit_rate(), 0.6);
    }
}
