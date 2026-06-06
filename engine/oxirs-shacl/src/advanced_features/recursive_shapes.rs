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

use crate::{
    paths::PropertyPathEvaluator, validation::ValidationEngine, Constraint, PropertyPath, Result,
    ShaclError, Shape, ShapeId, ValidationConfig,
};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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

    /// Handle cycle detection.
    ///
    /// When a `(node, shape)` pair is encountered that is already on the active
    /// validation stack, we are inside a cycle in the shape graph (e.g. a node
    /// shape whose property shape points back to itself). Following SHACL's
    /// well-formedness assumption for recursive shapes, the back-edge is treated
    /// as conforming: the structure rooted here has already been (or is being)
    /// validated higher up the stack, so re-validating it would not add
    /// information and would not terminate. This is what breaks infinite
    /// recursion while still fully validating the finite portion of the graph.
    fn handle_cycle(
        &mut self,
        _focus_node: &Term,
        _shape: &Shape,
    ) -> Result<RecursiveValidationResult> {
        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: self.current_depth,
            cycles_detected: true,
            cached: false,
        })
    }

    /// Depth-first validation.
    ///
    /// Validates the focus node against the shape's *local* constraints, then
    /// immediately descends into each referenced child `(node, shape)` work item
    /// via [`Self::validate_recursive`] before moving on to the next sibling.
    /// This recursion-order (deep before wide) is what distinguishes it from the
    /// breadth-first strategy.
    fn validate_depth_first(
        &mut self,
        focus_node: &Term,
        shape: &Shape,
        store: &dyn Store,
        shape_resolver: &dyn ShapeResolver,
    ) -> Result<RecursiveValidationResult> {
        // 1. Local conformance (non-recursive constraints only).
        let local_ok = self.local_conformance(focus_node, shape, store)?;
        if !local_ok {
            return Ok(RecursiveValidationResult {
                conforms: false,
                depth_reached: self.current_depth,
                cycles_detected: false,
                cached: false,
            });
        }

        // 2. Recurse into each referenced child, depth-first.
        let children = self.recursive_children(focus_node, shape, store, shape_resolver)?;
        let mut cycles_detected = false;
        let mut deepest = self.current_depth;

        for (child_node, child_shape_id) in children {
            let Some(child_shape) = shape_resolver.resolve_shape(&child_shape_id) else {
                // Unresolvable reference: cannot prove conformance.
                return Ok(RecursiveValidationResult {
                    conforms: false,
                    depth_reached: deepest,
                    cycles_detected,
                    cached: false,
                });
            };
            let child_shape = child_shape.clone();
            let child_result =
                self.validate_recursive(&child_node, &child_shape, store, shape_resolver)?;
            cycles_detected |= child_result.cycles_detected;
            deepest = deepest.max(child_result.depth_reached);
            if !child_result.conforms {
                return Ok(RecursiveValidationResult {
                    conforms: false,
                    depth_reached: deepest,
                    cycles_detected,
                    cached: false,
                });
            }
        }

        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: deepest,
            cycles_detected,
            cached: false,
        })
    }

    /// Breadth-first validation.
    ///
    /// Validates the focus node's local constraints, then enqueues every directly
    /// referenced child `(node, shape)` work item and processes them level by
    /// level using an explicit FIFO queue. Each dequeued item is validated
    /// locally and its own children are appended to the back of the queue, so an
    /// entire level is processed before the next is reached — the opposite
    /// traversal order to the depth-first strategy. Cycle detection here uses the
    /// validator's `visited_nodes` set together with a queue-local seen-set so
    /// that revisiting a `(node, shape)` pair already enqueued in this traversal
    /// terminates instead of looping.
    fn validate_breadth_first(
        &mut self,
        focus_node: &Term,
        shape: &Shape,
        store: &dyn Store,
        shape_resolver: &dyn ShapeResolver,
    ) -> Result<RecursiveValidationResult> {
        let mut queue: VecDeque<(Term, ShapeId, usize)> = VecDeque::new();
        let mut seen: HashSet<(Term, ShapeId)> = HashSet::new();
        let mut cycles_detected = false;
        let mut deepest = self.current_depth;

        // Seed the queue with the root node/shape at the current depth.
        seen.insert((focus_node.clone(), shape.id.clone()));
        queue.push_back((focus_node.clone(), shape.id.clone(), self.current_depth));

        while let Some((node, shape_id, depth)) = queue.pop_front() {
            if depth > self.config.max_depth {
                return Err(ShaclError::RecursionLimit(format!(
                    "Maximum recursion depth exceeded: {}",
                    self.config.max_depth
                )));
            }
            deepest = deepest.max(depth);

            let Some(current_shape) = shape_resolver.resolve_shape(&shape_id) else {
                return Ok(RecursiveValidationResult {
                    conforms: false,
                    depth_reached: deepest,
                    cycles_detected,
                    cached: false,
                });
            };
            let current_shape = current_shape.clone();

            // Local conformance for this work item.
            if !self.local_conformance(&node, &current_shape, store)? {
                return Ok(RecursiveValidationResult {
                    conforms: false,
                    depth_reached: deepest,
                    cycles_detected,
                    cached: false,
                });
            }

            // Enqueue children one level deeper.
            let children = self.recursive_children(&node, &current_shape, store, shape_resolver)?;
            for (child_node, child_shape_id) in children {
                let key = (child_node.clone(), child_shape_id.clone());
                if seen.contains(&key) {
                    // Already scheduled/visited in this traversal => cycle edge.
                    cycles_detected = true;
                    continue;
                }
                seen.insert(key);
                queue.push_back((child_node, child_shape_id, depth + 1));
            }
        }

        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: deepest,
            cycles_detected,
            cached: false,
        })
    }

    /// Optimized validation with aggressive cache use and cycle breaking.
    ///
    /// Structurally this mirrors the depth-first traversal, but it consults
    /// [`Self::result_cache`] before descending into any child: if a
    /// `(node, shape)` result has already been computed in this run it is reused
    /// directly (incrementing the cache-hit statistic) instead of re-validating
    /// the whole subtree. The recursive descent it does perform still goes
    /// through [`Self::validate_recursive`], which also populates and reads the
    /// shared cache, so repeated substructures are validated at most once.
    fn validate_optimized(
        &mut self,
        focus_node: &Term,
        shape: &Shape,
        store: &dyn Store,
        shape_resolver: &dyn ShapeResolver,
    ) -> Result<RecursiveValidationResult> {
        let local_ok = self.local_conformance(focus_node, shape, store)?;
        if !local_ok {
            return Ok(RecursiveValidationResult {
                conforms: false,
                depth_reached: self.current_depth,
                cycles_detected: false,
                cached: false,
            });
        }

        let children = self.recursive_children(focus_node, shape, store, shape_resolver)?;
        let mut cycles_detected = false;
        let mut deepest = self.current_depth;

        for (child_node, child_shape_id) in children {
            let cache_key = (child_node.clone(), child_shape_id.clone());

            // Aggressive cache check at the call site before descending.
            if self.config.cache_results {
                if let Some(&cached) = self.result_cache.get(&cache_key) {
                    self.stats.cache_hits += 1;
                    if !cached {
                        return Ok(RecursiveValidationResult {
                            conforms: false,
                            depth_reached: deepest,
                            cycles_detected,
                            cached: true,
                        });
                    }
                    continue;
                }
            }

            let Some(child_shape) = shape_resolver.resolve_shape(&child_shape_id) else {
                return Ok(RecursiveValidationResult {
                    conforms: false,
                    depth_reached: deepest,
                    cycles_detected,
                    cached: false,
                });
            };
            let child_shape = child_shape.clone();
            let child_result =
                self.validate_recursive(&child_node, &child_shape, store, shape_resolver)?;
            cycles_detected |= child_result.cycles_detected;
            deepest = deepest.max(child_result.depth_reached);
            if !child_result.conforms {
                return Ok(RecursiveValidationResult {
                    conforms: false,
                    depth_reached: deepest,
                    cycles_detected,
                    cached: false,
                });
            }
        }

        Ok(RecursiveValidationResult {
            conforms: true,
            depth_reached: deepest,
            cycles_detected,
            cached: false,
        })
    }

    /// Validate a focus node against a shape's *local* constraints only.
    ///
    /// "Local" means everything except the shape-to-shape references that the
    /// recursive traversal handles itself (`sh:node`, linked `sh:property`
    /// shapes, and the logical `sh:and`/`sh:or`/`sh:xone`/`sh:not` combinators).
    /// Those references are stripped from a clone of the shape before it is run
    /// through a single-shape [`ValidationEngine`]; this guarantees the engine
    /// evaluates value/cardinality/string/range constraints without performing
    /// its own (non-cycle-aware) shape recursion, which would otherwise loop on
    /// cyclic shape graphs.
    fn local_conformance(
        &self,
        focus_node: &Term,
        shape: &Shape,
        store: &dyn Store,
    ) -> Result<bool> {
        let mut local_shape = shape.clone();
        // Strip references that the recursive traversal owns.
        local_shape.property_shapes.clear();
        local_shape.extends.clear();
        local_shape.constraints.retain(|_, constraint| {
            !matches!(
                constraint,
                Constraint::Node(_)
                    | Constraint::And(_)
                    | Constraint::Or(_)
                    | Constraint::Xone(_)
                    | Constraint::Not(_)
                    | Constraint::QualifiedValueShape(_)
            )
        });

        // If nothing local remains for a node shape, it trivially conforms.
        if local_shape.is_node_shape() && local_shape.constraints.is_empty() {
            return Ok(true);
        }

        let mut temp_shapes = indexmap::IndexMap::new();
        temp_shapes.insert(local_shape.id.clone(), local_shape.clone());

        let config = ValidationConfig::default();
        let mut validator = ValidationEngine::new(&temp_shapes, config);
        match validator.validate_node_against_shape(store, &local_shape, focus_node, None) {
            Ok(report) => Ok(report.conforms()),
            Err(e) => {
                tracing::warn!("Recursive local validation error: {e}");
                Ok(false)
            }
        }
    }

    /// Compute the recursive child work items for a `(focus_node, shape)` pair.
    ///
    /// Returns the `(node, shape_id)` pairs that must additionally conform for
    /// the focus node to conform to `shape`:
    ///
    /// * `sh:node S` on a node shape -> `(focus_node, S)` (same node, new shape).
    /// * A linked `sh:property` shape (or this shape if it is itself a property
    ///   shape) contributes, for each value reached via its `sh:path`, one item
    ///   `(value, S)` for every `sh:node S` declared inside that property shape.
    ///
    /// This is precisely the set of edges that make tree / linked-list / graph
    /// shapes recursive, and routing them through the validator's
    /// [`Self::validate_recursive`] is what makes cycle detection effective.
    fn recursive_children(
        &self,
        focus_node: &Term,
        shape: &Shape,
        store: &dyn Store,
        shape_resolver: &dyn ShapeResolver,
    ) -> Result<Vec<(Term, ShapeId)>> {
        let mut children = Vec::new();

        // sh:node references on this shape apply to the same focus node.
        for constraint in shape.constraints.values() {
            if let Constraint::Node(node_constraint) = constraint {
                children.push((focus_node.clone(), node_constraint.shape.clone()));
            }
        }

        // If this shape is itself a property shape, its path values feed any
        // nested sh:node references.
        if shape.is_property_shape() {
            if let Some(path) = &shape.path {
                self.collect_property_children(focus_node, shape, path, store, &mut children)?;
            }
        }

        // Linked property shapes (sh:property) on a node shape.
        for property_shape_id in &shape.property_shapes {
            if let Some(property_shape) = shape_resolver.resolve_shape(property_shape_id) {
                if let Some(path) = property_shape.path.clone() {
                    self.collect_property_children(
                        focus_node,
                        property_shape,
                        &path,
                        store,
                        &mut children,
                    )?;
                }
            }
        }

        Ok(children)
    }

    /// For a property shape with a resolved `path`, evaluate the path from
    /// `focus_node` and emit `(value, S)` child items for each `sh:node S`
    /// constraint declared on the property shape.
    fn collect_property_children(
        &self,
        focus_node: &Term,
        property_shape: &Shape,
        path: &PropertyPath,
        store: &dyn Store,
        children: &mut Vec<(Term, ShapeId)>,
    ) -> Result<()> {
        // Gather sh:node references declared on the property shape.
        let node_refs: Vec<ShapeId> = property_shape
            .constraints
            .values()
            .filter_map(|constraint| match constraint {
                Constraint::Node(node_constraint) => Some(node_constraint.shape.clone()),
                _ => None,
            })
            .collect();

        if node_refs.is_empty() {
            return Ok(());
        }

        let mut path_evaluator = PropertyPathEvaluator::new();
        let values = path_evaluator.evaluate_path(store, focus_node, path, None)?;

        for value in values {
            for shape_id in &node_refs {
                children.push((value.clone(), shape_id.clone()));
            }
        }

        Ok(())
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

    /// Extract dependencies from a shape.
    ///
    /// Collects every other [`ShapeId`] that `shape` references and therefore
    /// depends on: linked `sh:property` shapes, `sh:extends` parents, and shape
    /// references buried inside constraints (`sh:node`, `sh:not`,
    /// `sh:and`/`sh:or`/`sh:xone` lists, and `sh:qualifiedValueShape`).
    fn extract_dependencies(&self, shape: &Shape) -> Vec<ShapeId> {
        let mut deps: Vec<ShapeId> = Vec::new();

        // Linked property shapes and inheritance parents.
        deps.extend(shape.property_shapes.iter().cloned());
        deps.extend(shape.extends.iter().cloned());

        // Shape references inside constraints.
        for constraint in shape.constraints.values() {
            match constraint {
                Constraint::Node(c) => deps.push(c.shape.clone()),
                Constraint::Not(c) => deps.push(c.shape.clone()),
                Constraint::And(c) => deps.extend(c.shapes.iter().cloned()),
                Constraint::Or(c) => deps.extend(c.shapes.iter().cloned()),
                Constraint::Xone(c) => deps.extend(c.shapes.iter().cloned()),
                Constraint::QualifiedValueShape(c) => deps.push(c.shape.clone()),
                _ => {}
            }
        }

        // De-duplicate while preserving first-seen order.
        let mut seen = HashSet::new();
        deps.retain(|id| seen.insert(id.clone()));

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
