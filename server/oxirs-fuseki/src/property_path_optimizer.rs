//! Advanced property path optimization for SPARQL 1.2
//!
//! This module implements sophisticated optimization strategies for property paths
//! including path rewriting, cost-based optimization, and advanced execution strategies.

use crate::error::{FusekiError, FusekiResult};
use futures::future::BoxFuture;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Enhanced property path optimizer with advanced strategies
#[derive(Debug, Clone)]
pub struct AdvancedPropertyPathOptimizer {
    /// Cache for optimized paths
    pub path_cache: Arc<RwLock<HashMap<String, OptimizedPath>>>,
    /// Path execution statistics
    pub statistics: Arc<RwLock<PathStatistics>>,
    /// Cost model for path evaluation
    pub cost_model: CostModel,
    /// Path rewrite rules
    pub rewrite_rules: Vec<PathRewriteRule>,
    /// Index availability information
    pub index_info: Arc<RwLock<IndexInfo>>,
}

/// Cost model for property path evaluation
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Base cost per traversal step
    pub traversal_cost: f64,
    /// Cost multiplier for inverse traversal
    pub inverse_multiplier: f64,
    /// Cost multiplier for alternative paths
    pub alternative_multiplier: f64,
    /// Cost multiplier for repetition operators
    pub repetition_multiplier: f64,
    /// Cost reduction for indexed lookups
    pub index_reduction_factor: f64,
    /// Cost for join operations
    pub join_cost: f64,
    /// Memory cost factor
    pub memory_factor: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            traversal_cost: 10.0,
            inverse_multiplier: 1.5,
            alternative_multiplier: 1.2,
            repetition_multiplier: 3.0,
            index_reduction_factor: 0.1,
            join_cost: 20.0,
            memory_factor: 0.01,
        }
    }
}

/// Property path rewrite rule
#[derive(Debug, Clone)]
pub struct PathRewriteRule {
    pub name: String,
    pub pattern: PathPattern,
    pub rewrite: PathRewrite,
    pub conditions: Vec<RewriteCondition>,
    pub priority: i32,
}

/// Pattern for matching property paths
#[derive(Debug, Clone)]
pub enum PathPattern {
    /// Simple property
    Property(String),
    /// Sequence of properties
    Sequence(Vec<PathPattern>),
    /// Alternative paths
    Alternative(Vec<PathPattern>),
    /// Inverse path
    Inverse(Box<PathPattern>),
    /// Zero or more repetitions
    ZeroOrMore(Box<PathPattern>),
    /// One or more repetitions
    OneOrMore(Box<PathPattern>),
    /// Optional path
    Optional(Box<PathPattern>),
    /// Fixed repetitions
    Repetition {
        pattern: Box<PathPattern>,
        min: usize,
        max: Option<usize>,
    },
    /// Negated property set
    NegatedPropertySet(Vec<String>),
    /// Any pattern (wildcard)
    Any,
}

/// Path rewrite transformation
#[derive(Debug, Clone)]
pub enum PathRewrite {
    /// Replace with optimized pattern
    Replace(PathPattern),
    /// Use index lookup
    UseIndex(String),
    /// Split into subqueries
    SplitQuery(Vec<PathPattern>),
    /// Materialize intermediate results
    Materialize,
    /// Custom transformation
    Custom(String),
}

/// Conditions for applying rewrite rules
#[derive(Debug, Clone)]
pub enum RewriteCondition {
    /// Path length constraint
    PathLength {
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Cardinality estimate
    EstimatedCardinality { min: Option<u64>, max: Option<u64> },
    /// Index availability
    IndexAvailable(String),
    /// Memory constraint
    MemoryLimit(u64),
    /// Custom condition
    Custom(String),
}

/// Information about available indexes
#[derive(Debug, Clone, Default)]
pub struct IndexInfo {
    /// Property indexes (property -> subjects)
    pub property_indexes: HashSet<String>,
    /// Inverse property indexes (property -> objects)
    pub inverse_property_indexes: HashSet<String>,
    /// Path indexes (precomputed paths)
    pub path_indexes: HashMap<String, PathIndexInfo>,
    /// Type indexes
    pub type_indexes: HashSet<String>,
}

/// Information about a specific path index
#[derive(Debug, Clone)]
pub struct PathIndexInfo {
    pub path: String,
    pub max_length: usize,
    pub cardinality: u64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Enhanced path execution plan
#[derive(Debug, Clone)]
pub struct EnhancedPathExecutionPlan {
    pub strategy: PathExecutionStrategy,
    pub steps: Vec<EnhancedPathStep>,
    pub estimated_cost: f64,
    pub estimated_cardinality: u64,
    pub memory_requirements: u64,
    pub parallelizable: bool,
    pub optimization_hints: Vec<String>,
}

/// Advanced path execution strategies
#[derive(Debug, Clone)]
pub enum PathExecutionStrategy {
    /// Simple forward traversal
    ForwardTraversal,
    /// Backward traversal (from object to subject)
    BackwardTraversal,
    /// Bidirectional search meeting in the middle
    BidirectionalMeet { meet_point: Option<usize> },
    /// Use precomputed index
    IndexLookup { index_name: String },
    /// Use materialized view
    MaterializedView { view_name: String },
    /// Parallel execution of alternatives
    ParallelAlternatives,
    /// Breadth-first search for shortest paths
    BreadthFirst { max_depth: Option<usize> },
    /// Depth-first search with pruning
    DepthFirst {
        max_depth: Option<usize>,
        prune_threshold: f64,
    },
    /// Dynamic programming approach
    DynamicProgramming,
    /// Hybrid strategy combining multiple approaches
    Hybrid {
        strategies: Vec<PathExecutionStrategy>,
    },
}

/// Enhanced path execution step
#[derive(Debug, Clone)]
pub struct EnhancedPathStep {
    pub operation: PathOperation,
    pub estimated_cost: f64,
    pub estimated_selectivity: f64,
    pub can_use_index: bool,
    pub memory_usage: u64,
    pub dependencies: Vec<usize>, // Indices of dependent steps
}

/// Path operations
#[derive(Debug, Clone)]
pub enum PathOperation {
    /// Simple traversal
    Traverse {
        predicate: String,
        direction: TraversalDirection,
    },
    /// Join operation
    Join {
        left: Box<PathOperation>,
        right: Box<PathOperation>,
        join_type: JoinType,
    },
    /// Union of results
    Union(Vec<PathOperation>),
    /// Filter results
    Filter { condition: FilterCondition },
    /// Compute transitive closure
    TransitiveClosure {
        predicate: String,
        min_length: usize,
        max_length: Option<usize>,
    },
    /// Index scan
    IndexScan { index_name: String, pattern: String },
    /// Materialize intermediate results
    Materialize,
    /// Sort results
    Sort { key: String, ascending: bool },
    /// Limit results
    Limit(usize),
}

/// Types of joins
#[derive(Debug, Clone)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    Full,
    Semi,
    Anti,
}

/// Filter conditions
#[derive(Debug, Clone)]
pub enum FilterCondition {
    /// Type constraint
    TypeConstraint(String),
    /// Value constraint
    ValueConstraint { operator: String, value: String },
    /// Existence check
    Exists,
    /// Custom filter
    Custom(String),
}

/// Traversal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    Forward,  // Subject to Object
    Backward, // Object to Subject
    Both,     // Bidirectional
}

/// Path execution statistics
#[derive(Debug, Clone, Default)]
pub struct PathStatistics {
    pub total_executions: u64,
    pub average_execution_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub optimization_successes: u64,
    pub optimization_failures: u64,
    pub path_frequency: HashMap<String, u64>,
    pub strategy_effectiveness: HashMap<String, f64>,
}

impl AdvancedPropertyPathOptimizer {
    pub fn new() -> Self {
        Self {
            path_cache: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(PathStatistics::default())),
            cost_model: CostModel::default(),
            rewrite_rules: Self::create_default_rewrite_rules(),
            index_info: Arc::new(RwLock::new(IndexInfo::default())),
        }
    }

    /// Create default rewrite rules
    fn create_default_rewrite_rules() -> Vec<PathRewriteRule> {
        vec![
            // Optimize inverse of inverse
            PathRewriteRule {
                name: "double_inverse_elimination".to_string(),
                pattern: PathPattern::Inverse(Box::new(PathPattern::Inverse(Box::new(
                    PathPattern::Any,
                )))),
                rewrite: PathRewrite::Replace(PathPattern::Any),
                conditions: vec![],
                priority: 100,
            },
            // Optimize single property with repetition
            PathRewriteRule {
                name: "indexed_property_repetition".to_string(),
                pattern: PathPattern::ZeroOrMore(Box::new(PathPattern::Property("_".to_string()))),
                rewrite: PathRewrite::UseIndex("transitive_closure_index".to_string()),
                conditions: vec![RewriteCondition::IndexAvailable(
                    "transitive_closure_index".to_string(),
                )],
                priority: 90,
            },
            // Optimize alternatives with common prefix
            PathRewriteRule {
                name: "factor_common_prefix".to_string(),
                pattern: PathPattern::Alternative(vec![PathPattern::Any, PathPattern::Any]),
                rewrite: PathRewrite::Custom("factor_common_prefix".to_string()),
                conditions: vec![],
                priority: 80,
            },
        ]
    }

    /// Optimize a property path with advanced strategies
    pub async fn optimize_path(&self, path: &str) -> FusekiResult<OptimizedPath> {
        let start_time = std::time::Instant::now();

        // Check cache first
        if let Some(cached) = self.get_cached_path(path).await? {
            self.record_cache_hit().await;
            return Ok(cached);
        }

        self.record_cache_miss().await;

        // Parse the property path
        let parsed_path = self.parse_property_path(path)?;

        // Apply rewrite rules
        let rewritten_path = self.apply_rewrite_rules(&parsed_path).await?;

        // Analyze path characteristics
        let characteristics = self.analyze_path_characteristics(&rewritten_path).await?;

        // Choose optimal execution strategy
        let strategy = self
            .choose_optimal_strategy(&rewritten_path, &characteristics)
            .await?;

        // Create execution plan
        let execution_plan = self
            .create_enhanced_execution_plan(&rewritten_path, strategy)
            .await?;

        // Estimate cost and cardinality
        let cost_estimate = self.estimate_total_cost(&execution_plan).await?;
        let cardinality_estimate = self.estimate_result_cardinality(&rewritten_path).await?;

        let optimized = OptimizedPath {
            original_path: path.to_string(),
            optimized_form: self.path_to_string(&rewritten_path),
            estimated_cardinality: cardinality_estimate,
            execution_plan: PathExecutionPlan {
                strategy: self.strategy_to_legacy(&execution_plan.strategy),
                estimated_cost: cost_estimate,
                intermediate_steps: self.convert_to_legacy_steps(&execution_plan.steps),
            },
        };

        // Cache the result
        self.cache_optimized_path(path, &optimized).await?;

        // Record statistics
        let elapsed = start_time.elapsed();
        self.record_optimization_stats(path, elapsed.as_millis() as f64, true)
            .await;

        info!("Optimized property path '{}' in {:?}", path, elapsed);

        Ok(optimized)
    }

    /// Parse a property path string into internal representation
    fn parse_property_path(&self, path: &str) -> FusekiResult<PathPattern> {
        // This is a simplified parser - in production would use a proper parser
        let path = path.trim();

        // Handle simple cases
        if !path.contains(|c: char| {
            c == '/'
                || c == '|'
                || c == '*'
                || c == '+'
                || c == '?'
                || c == '^'
                || c == '!'
                || c == '('
                || c == ')'
        }) {
            return Ok(PathPattern::Property(path.to_string()));
        }

        // Handle inverse
        if path.starts_with('^') {
            let inner = self.parse_property_path(&path[1..])?;
            return Ok(PathPattern::Inverse(Box::new(inner)));
        }

        // Handle zero or more
        if path.ends_with('*') {
            let inner = self.parse_property_path(&path[..path.len() - 1])?;
            return Ok(PathPattern::ZeroOrMore(Box::new(inner)));
        }

        // Handle one or more
        if path.ends_with('+') {
            let inner = self.parse_property_path(&path[..path.len() - 1])?;
            return Ok(PathPattern::OneOrMore(Box::new(inner)));
        }

        // Handle optional
        if path.ends_with('?') {
            let inner = self.parse_property_path(&path[..path.len() - 1])?;
            return Ok(PathPattern::Optional(Box::new(inner)));
        }

        // Handle alternatives
        if path.contains('|') {
            let parts: Vec<_> = path
                .split('|')
                .map(|p| self.parse_property_path(p.trim()))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(PathPattern::Alternative(parts));
        }

        // Handle sequences
        if path.contains('/') {
            let parts: Vec<_> = path
                .split('/')
                .map(|p| self.parse_property_path(p.trim()))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(PathPattern::Sequence(parts));
        }

        // Default to property
        Ok(PathPattern::Property(path.to_string()))
    }

    /// Apply rewrite rules to optimize the path
    async fn apply_rewrite_rules(&self, path: &PathPattern) -> FusekiResult<PathPattern> {
        let mut current_path = path.clone();
        let mut applied_rules = Vec::new();

        // Sort rules by priority
        let mut rules = self.rewrite_rules.clone();
        rules.sort_by_key(|r| -r.priority);

        for rule in &rules {
            if self.pattern_matches(&current_path, &rule.pattern)
                && self.conditions_met(&current_path, &rule.conditions).await?
            {
                debug!("Applying rewrite rule: {}", rule.name);
                current_path = self.apply_rewrite(&current_path, &rule.rewrite)?;
                applied_rules.push(rule.name.clone());

                // Limit number of rewrites to prevent infinite loops
                if applied_rules.len() > 10 {
                    warn!("Rewrite limit reached, stopping optimization");
                    break;
                }
            }
        }

        if !applied_rules.is_empty() {
            debug!("Applied rewrite rules: {:?}", applied_rules);
        }

        Ok(current_path)
    }

    /// Check if a pattern matches
    fn pattern_matches(&self, path: &PathPattern, pattern: &PathPattern) -> bool {
        match (path, pattern) {
            (_, PathPattern::Any) => true,
            (PathPattern::Property(p1), PathPattern::Property(p2)) => p1 == p2 || p2 == "_",
            (PathPattern::Sequence(s1), PathPattern::Sequence(s2)) => {
                s1.len() == s2.len()
                    && s1
                        .iter()
                        .zip(s2.iter())
                        .all(|(p1, p2)| self.pattern_matches(p1, p2))
            }
            (PathPattern::Alternative(a1), PathPattern::Alternative(a2)) => {
                a1.len() == a2.len()
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(p1, p2)| self.pattern_matches(p1, p2))
            }
            (PathPattern::Inverse(i1), PathPattern::Inverse(i2)) => self.pattern_matches(i1, i2),
            (PathPattern::ZeroOrMore(z1), PathPattern::ZeroOrMore(z2)) => {
                self.pattern_matches(z1, z2)
            }
            (PathPattern::OneOrMore(o1), PathPattern::OneOrMore(o2)) => {
                self.pattern_matches(o1, o2)
            }
            (PathPattern::Optional(o1), PathPattern::Optional(o2)) => self.pattern_matches(o1, o2),
            _ => false,
        }
    }

    /// Check if rewrite conditions are met
    async fn conditions_met(
        &self,
        path: &PathPattern,
        conditions: &[RewriteCondition],
    ) -> FusekiResult<bool> {
        for condition in conditions {
            match condition {
                RewriteCondition::PathLength { min, max } => {
                    let length = self.estimate_path_length(path);
                    if let Some(min) = min {
                        if length < *min {
                            return Ok(false);
                        }
                    }
                    if let Some(max) = max {
                        if length > *max {
                            return Ok(false);
                        }
                    }
                }
                RewriteCondition::IndexAvailable(index) => {
                    let index_info = self.index_info.read().await;
                    if !index_info.property_indexes.contains(index)
                        && !index_info.path_indexes.contains_key(index)
                    {
                        return Ok(false);
                    }
                }
                RewriteCondition::EstimatedCardinality { min, max } => {
                    let cardinality = self.estimate_result_cardinality(path).await?;
                    if let Some(min) = min {
                        if cardinality < *min {
                            return Ok(false);
                        }
                    }
                    if let Some(max) = max {
                        if cardinality > *max {
                            return Ok(false);
                        }
                    }
                }
                _ => {} // Other conditions not implemented yet
            }
        }
        Ok(true)
    }

    /// Apply a rewrite transformation
    fn apply_rewrite(
        &self,
        path: &PathPattern,
        rewrite: &PathRewrite,
    ) -> FusekiResult<PathPattern> {
        match rewrite {
            PathRewrite::Replace(new_pattern) => Ok(new_pattern.clone()),
            PathRewrite::UseIndex(index_name) => {
                // Create a custom pattern that represents index usage
                Ok(PathPattern::Property(format!("INDEX:{}", index_name)))
            }
            PathRewrite::Materialize => {
                // Wrap the pattern to indicate materialization
                Ok(PathPattern::Property(format!(
                    "MATERIALIZE:{}",
                    self.path_to_string(path)
                )))
            }
            PathRewrite::Custom(transform) => {
                // Apply custom transformation
                match transform.as_str() {
                    "factor_common_prefix" => self.factor_common_prefix(path),
                    _ => Ok(path.clone()),
                }
            }
            _ => Ok(path.clone()),
        }
    }

    /// Factor out common prefixes in alternatives
    fn factor_common_prefix(&self, path: &PathPattern) -> FusekiResult<PathPattern> {
        if let PathPattern::Alternative(alts) = path {
            // Simplified implementation - would be more sophisticated in production
            Ok(path.clone())
        } else {
            Ok(path.clone())
        }
    }

    /// Analyze path characteristics for optimization
    async fn analyze_path_characteristics(
        &self,
        path: &PathPattern,
    ) -> FusekiResult<PathCharacteristics> {
        Ok(PathCharacteristics {
            estimated_length: self.estimate_path_length(path),
            has_cycles: self.detect_cycles(path),
            has_alternatives: self.has_alternatives(path),
            has_repetition: self.has_repetition(path),
            has_inverse: self.has_inverse(path),
            is_linear: self.is_linear_path(path),
            estimated_branching_factor: self.estimate_branching_factor(path).await?,
            can_use_index: self.can_use_index(path).await?,
        })
    }

    /// Choose optimal execution strategy based on path characteristics
    async fn choose_optimal_strategy(
        &self,
        path: &PathPattern,
        characteristics: &PathCharacteristics,
    ) -> FusekiResult<PathExecutionStrategy> {
        // If we can use an index, prefer that
        if characteristics.can_use_index {
            if let Ok(index_name) = self.find_best_index(path).await {
                return Ok(PathExecutionStrategy::IndexLookup { index_name });
            }
        }

        // For short paths without repetition, use simple traversal
        if characteristics.estimated_length <= 2 && !characteristics.has_repetition {
            return Ok(PathExecutionStrategy::ForwardTraversal);
        }

        // For paths with inverse and reasonable length, use bidirectional search
        if characteristics.has_inverse && characteristics.estimated_length > 3 {
            let meet_point = Some(characteristics.estimated_length / 2);
            return Ok(PathExecutionStrategy::BidirectionalMeet { meet_point });
        }

        // For paths with alternatives, consider parallel execution
        if characteristics.has_alternatives && !characteristics.has_repetition {
            return Ok(PathExecutionStrategy::ParallelAlternatives);
        }

        // For paths with repetition, use appropriate search strategy
        if characteristics.has_repetition {
            if characteristics.estimated_branching_factor > 10.0 {
                // High branching factor - use breadth-first with depth limit
                return Ok(PathExecutionStrategy::BreadthFirst {
                    max_depth: Some(10),
                });
            } else {
                // Low branching factor - depth-first might be more efficient
                return Ok(PathExecutionStrategy::DepthFirst {
                    max_depth: Some(20),
                    prune_threshold: 0.1,
                });
            }
        }

        // Default to forward traversal
        Ok(PathExecutionStrategy::ForwardTraversal)
    }

    /// Create enhanced execution plan
    async fn create_enhanced_execution_plan(
        &self,
        path: &PathPattern,
        strategy: PathExecutionStrategy,
    ) -> FusekiResult<EnhancedPathExecutionPlan> {
        let steps = self.decompose_path_into_operations(path, &strategy).await?;
        let estimated_cost = self.calculate_plan_cost(&steps, &strategy).await?;
        let estimated_cardinality = self.estimate_result_cardinality(path).await?;
        let memory_requirements = self.estimate_memory_requirements(&steps, estimated_cardinality);
        let parallelizable = self.is_parallelizable(&strategy, &steps);
        let optimization_hints = self.generate_optimization_hints(path, &strategy);

        Ok(EnhancedPathExecutionPlan {
            strategy,
            steps,
            estimated_cost,
            estimated_cardinality,
            memory_requirements,
            parallelizable,
            optimization_hints,
        })
    }

    /// Decompose path into executable operations
    fn decompose_path_into_operations<'a>(
        &'a self,
        path: &'a PathPattern,
        strategy: &'a PathExecutionStrategy,
    ) -> BoxFuture<'a, FusekiResult<Vec<EnhancedPathStep>>> {
        Box::pin(async move {
            let mut steps = Vec::new();

            match path {
                PathPattern::Property(prop) => {
                    steps.push(
                        self.create_traverse_step(prop, TraversalDirection::Forward)
                            .await?,
                    );
                }
                PathPattern::Sequence(seq) => {
                    for (i, p) in seq.iter().enumerate() {
                        let sub_steps = self.decompose_path_into_operations(p, strategy).await?;
                        let deps: Vec<usize> = if i > 0 { vec![steps.len() - 1] } else { vec![] };
                        for mut step in sub_steps {
                            step.dependencies = deps.clone();
                            steps.push(step);
                        }
                    }
                }
                PathPattern::Alternative(alts) => {
                    let mut alt_operations = Vec::new();
                    for alt in alts {
                        let sub_steps = self.decompose_path_into_operations(alt, strategy).await?;
                        if sub_steps.len() == 1 {
                            if let Some(step) = sub_steps.into_iter().next() {
                                alt_operations.push(step.operation);
                            }
                        }
                    }
                    if !alt_operations.is_empty() {
                        steps.push(EnhancedPathStep {
                            operation: PathOperation::Union(alt_operations),
                            estimated_cost: self.cost_model.alternative_multiplier
                                * self.cost_model.traversal_cost,
                            estimated_selectivity: 0.8,
                            can_use_index: false,
                            memory_usage: 1024 * 1024, // 1MB estimate
                            dependencies: vec![],
                        });
                    }
                }
                PathPattern::Inverse(inner) => {
                    let mut sub_steps =
                        self.decompose_path_into_operations(inner, strategy).await?;
                    for step in &mut sub_steps {
                        if let PathOperation::Traverse { direction, .. } = &mut step.operation {
                            *direction = match direction {
                                TraversalDirection::Forward => TraversalDirection::Backward,
                                TraversalDirection::Backward => TraversalDirection::Forward,
                                TraversalDirection::Both => TraversalDirection::Both,
                            };
                        }
                    }
                    steps.extend(sub_steps);
                }
                PathPattern::ZeroOrMore(inner) | PathPattern::OneOrMore(inner) => {
                    if let PathPattern::Property(prop) = inner.as_ref() {
                        let min_length = if matches!(path, PathPattern::OneOrMore(_)) {
                            1
                        } else {
                            0
                        };
                        steps.push(EnhancedPathStep {
                            operation: PathOperation::TransitiveClosure {
                                predicate: prop.clone(),
                                min_length,
                                max_length: None,
                            },
                            estimated_cost: self.cost_model.repetition_multiplier
                                * self.cost_model.traversal_cost,
                            estimated_selectivity: 0.3,
                            can_use_index: self.can_use_transitive_index(prop).await?,
                            memory_usage: 10 * 1024 * 1024, // 10MB estimate for transitive closure
                            dependencies: vec![],
                        });
                    }
                }
                _ => {
                    // Default to simple traversal
                    steps.push(
                        self.create_traverse_step("?", TraversalDirection::Forward)
                            .await?,
                    );
                }
            }

            Ok(steps)
        })
    }

    /// Create a traverse step
    async fn create_traverse_step(
        &self,
        predicate: &str,
        direction: TraversalDirection,
    ) -> FusekiResult<EnhancedPathStep> {
        let can_use_index = self.can_use_property_index(predicate, &direction).await?;
        let cost = if can_use_index {
            self.cost_model.traversal_cost * self.cost_model.index_reduction_factor
        } else {
            self.cost_model.traversal_cost
                * if direction == TraversalDirection::Backward {
                    self.cost_model.inverse_multiplier
                } else {
                    1.0
                }
        };

        Ok(EnhancedPathStep {
            operation: PathOperation::Traverse {
                predicate: predicate.to_string(),
                direction,
            },
            estimated_cost: cost,
            estimated_selectivity: 0.5,
            can_use_index,
            memory_usage: 1024 * 100, // 100KB estimate
            dependencies: vec![],
        })
    }

    /// Check if property index is available
    async fn can_use_property_index(
        &self,
        predicate: &str,
        direction: &TraversalDirection,
    ) -> FusekiResult<bool> {
        let index_info = self.index_info.read().await;
        Ok(match direction {
            TraversalDirection::Forward => index_info.property_indexes.contains(predicate),
            TraversalDirection::Backward => index_info.inverse_property_indexes.contains(predicate),
            TraversalDirection::Both => {
                index_info.property_indexes.contains(predicate)
                    && index_info.inverse_property_indexes.contains(predicate)
            }
        })
    }

    /// Check if transitive index is available
    async fn can_use_transitive_index(&self, predicate: &str) -> FusekiResult<bool> {
        let index_info = self.index_info.read().await;
        Ok(index_info
            .path_indexes
            .contains_key(&format!("{}+", predicate)))
    }

    /// Find best available index for path
    async fn find_best_index(&self, path: &PathPattern) -> FusekiResult<String> {
        let index_info = self.index_info.read().await;
        let path_str = self.path_to_string(path);

        // Check for exact path index match
        if let Some(path_index) = index_info.path_indexes.get(&path_str) {
            return Ok(path_str);
        }

        // Check for property index
        if let PathPattern::Property(prop) = path {
            if index_info.property_indexes.contains(prop) {
                return Ok(prop.clone());
            }
        }

        Err(FusekiError::internal("No suitable index found"))
    }

    /// Helper functions for path analysis
    fn estimate_path_length(&self, path: &PathPattern) -> usize {
        match path {
            PathPattern::Property(_) => 1,
            PathPattern::Sequence(seq) => seq.iter().map(|p| self.estimate_path_length(p)).sum(),
            PathPattern::Alternative(alts) => alts
                .iter()
                .map(|p| self.estimate_path_length(p))
                .max()
                .unwrap_or(0),
            PathPattern::Inverse(inner) => self.estimate_path_length(inner),
            PathPattern::ZeroOrMore(_) | PathPattern::OneOrMore(_) => 5, // Estimate
            PathPattern::Optional(inner) => self.estimate_path_length(inner),
            PathPattern::Repetition { pattern, min, max } => {
                let base_len = self.estimate_path_length(pattern);
                base_len * max.unwrap_or(*min + 2)
            }
            _ => 1,
        }
    }

    fn detect_cycles(&self, path: &PathPattern) -> bool {
        // Simplified cycle detection
        matches!(path, PathPattern::ZeroOrMore(_) | PathPattern::OneOrMore(_))
    }

    fn has_alternatives(&self, path: &PathPattern) -> bool {
        match path {
            PathPattern::Alternative(_) => true,
            PathPattern::Sequence(seq) => seq.iter().any(|p| self.has_alternatives(p)),
            PathPattern::Inverse(inner)
            | PathPattern::ZeroOrMore(inner)
            | PathPattern::OneOrMore(inner)
            | PathPattern::Optional(inner) => self.has_alternatives(inner),
            _ => false,
        }
    }

    fn has_repetition(&self, path: &PathPattern) -> bool {
        match path {
            PathPattern::ZeroOrMore(_)
            | PathPattern::OneOrMore(_)
            | PathPattern::Repetition { .. } => true,
            PathPattern::Sequence(seq) => seq.iter().any(|p| self.has_repetition(p)),
            PathPattern::Alternative(alts) => alts.iter().any(|p| self.has_repetition(p)),
            PathPattern::Inverse(inner) | PathPattern::Optional(inner) => {
                self.has_repetition(inner)
            }
            _ => false,
        }
    }

    fn has_inverse(&self, path: &PathPattern) -> bool {
        match path {
            PathPattern::Inverse(_) => true,
            PathPattern::Sequence(seq) => seq.iter().any(|p| self.has_inverse(p)),
            PathPattern::Alternative(alts) => alts.iter().any(|p| self.has_inverse(p)),
            PathPattern::ZeroOrMore(inner)
            | PathPattern::OneOrMore(inner)
            | PathPattern::Optional(inner) => self.has_inverse(inner),
            _ => false,
        }
    }

    fn is_linear_path(&self, path: &PathPattern) -> bool {
        match path {
            PathPattern::Property(_) => true,
            PathPattern::Sequence(seq) => seq.iter().all(|p| self.is_linear_path(p)),
            PathPattern::Inverse(inner) => self.is_linear_path(inner),
            _ => false,
        }
    }

    fn estimate_branching_factor<'a>(
        &'a self,
        path: &'a PathPattern,
    ) -> BoxFuture<'a, FusekiResult<f64>> {
        Box::pin(async move {
            // This would use actual statistics in production
            Ok(match path {
                PathPattern::Property(prop) => {
                    // Estimate based on property statistics
                    if prop.contains("type") || prop.contains("Type") {
                        5.0 // Type properties typically have lower branching
                    } else if prop.contains("subClassOf") || prop.contains("subPropertyOf") {
                        3.0 // Hierarchical properties
                    } else {
                        10.0 // Default estimate
                    }
                }
                PathPattern::Alternative(alts) => {
                    // Sum of branching factors for alternatives
                    let mut total = 0.0;
                    for alt in alts {
                        total += self.estimate_branching_factor(alt).await?;
                    }
                    total
                }
                _ => 10.0, // Default
            })
        })
    }

    async fn can_use_index(&self, path: &PathPattern) -> FusekiResult<bool> {
        match path {
            PathPattern::Property(prop) => {
                self.can_use_property_index(prop, &TraversalDirection::Forward)
                    .await
            }
            PathPattern::ZeroOrMore(inner) | PathPattern::OneOrMore(inner) => {
                if let PathPattern::Property(prop) = inner.as_ref() {
                    self.can_use_transitive_index(prop).await
                } else {
                    Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    fn estimate_result_cardinality<'a>(
        &'a self,
        path: &'a PathPattern,
    ) -> futures::future::BoxFuture<'a, FusekiResult<u64>> {
        Box::pin(async move {
            // This would use actual statistics in production
            Ok(match path {
                PathPattern::Property(_) => 1000,
                PathPattern::Sequence(seq) => {
                    // Multiply selectivities
                    let mut cardinality = 10000u64;
                    for _ in seq {
                        cardinality = (cardinality as f64 * 0.1) as u64;
                    }
                    cardinality.max(1)
                }
                PathPattern::Alternative(alts) => {
                    // Sum cardinalities
                    let mut total = 0u64;
                    for alt in alts {
                        total += self.estimate_result_cardinality(alt).await?;
                    }
                    total
                }
                PathPattern::ZeroOrMore(_) | PathPattern::OneOrMore(_) => 10000,
                _ => 1000,
            })
        })
    }

    async fn calculate_plan_cost(
        &self,
        steps: &[EnhancedPathStep],
        strategy: &PathExecutionStrategy,
    ) -> FusekiResult<f64> {
        let mut total_cost = 0.0;

        for step in steps {
            total_cost += step.estimated_cost;
        }

        // Apply strategy-specific cost adjustments
        match strategy {
            PathExecutionStrategy::IndexLookup { .. } => {
                total_cost *= self.cost_model.index_reduction_factor;
            }
            PathExecutionStrategy::ParallelAlternatives => {
                // Parallel execution reduces time but increases resource usage
                total_cost *= 0.7;
            }
            PathExecutionStrategy::BidirectionalMeet { .. } => {
                // Bidirectional can be faster for long paths
                total_cost *= 0.8;
            }
            _ => {}
        }

        Ok(total_cost)
    }

    fn estimate_memory_requirements(
        &self,
        steps: &[EnhancedPathStep],
        estimated_cardinality: u64,
    ) -> u64 {
        let mut total_memory = 0u64;

        for step in steps {
            total_memory += step.memory_usage;
        }

        // Add memory for result set
        total_memory += estimated_cardinality * 100; // 100 bytes per result estimate

        total_memory
    }

    /// Estimate total cost for execution plan
    async fn estimate_total_cost(&self, execution_plan: &EnhancedPathExecutionPlan) -> FusekiResult<f64> {
        let mut total_cost = 0.0;

        // Base cost from the execution plan
        total_cost += execution_plan.estimated_cost;

        // Add cost for each step
        for step in &execution_plan.steps {
            total_cost += step.estimated_cost;
        }

        // Apply strategy-specific multipliers
        match &execution_plan.strategy {
            PathExecutionStrategy::ForwardTraversal => {
                total_cost *= 1.0; // Baseline
            }
            PathExecutionStrategy::BackwardTraversal => {
                total_cost *= self.cost_model.inverse_multiplier;
            }
            PathExecutionStrategy::BidirectionalMeet { .. } => {
                total_cost *= 0.7; // Usually more efficient
            }
            PathExecutionStrategy::IndexLookup { .. } => {
                total_cost *= self.cost_model.index_reduction_factor;
            }
            PathExecutionStrategy::MaterializedView { .. } => {
                total_cost *= 0.1; // Very efficient for precomputed views
            }
            PathExecutionStrategy::ParallelAlternatives => {
                total_cost *= self.cost_model.alternative_multiplier;
            }
            PathExecutionStrategy::BreadthFirst { .. } | 
            PathExecutionStrategy::DepthFirst { .. } => {
                total_cost *= 1.5; // Search algorithms have overhead
            }
            PathExecutionStrategy::DynamicProgramming => {
                total_cost *= 2.0; // Higher initial cost but better for complex patterns
            }
            PathExecutionStrategy::Hybrid { strategies } => {
                // Average cost of strategies with some overhead
                total_cost *= 1.2 + (strategies.len() as f64 * 0.1);
            }
        }

        // Add memory cost factor
        let memory_cost = execution_plan.memory_requirements as f64 * self.cost_model.memory_factor;
        total_cost += memory_cost;

        Ok(total_cost)
    }

    fn is_parallelizable(
        &self,
        strategy: &PathExecutionStrategy,
        steps: &[EnhancedPathStep],
    ) -> bool {
        match strategy {
            PathExecutionStrategy::ParallelAlternatives => true,
            _ => {
                // Check if steps have no dependencies
                steps.iter().all(|s| s.dependencies.is_empty())
            }
        }
    }

    fn generate_optimization_hints(
        &self,
        path: &PathPattern,
        strategy: &PathExecutionStrategy,
    ) -> Vec<String> {
        let mut hints = Vec::new();

        match strategy {
            PathExecutionStrategy::IndexLookup { index_name } => {
                hints.push(format!("Using index: {}", index_name));
            }
            PathExecutionStrategy::BidirectionalMeet { meet_point } => {
                if let Some(point) = meet_point {
                    hints.push(format!("Bidirectional search meeting at depth {}", point));
                }
            }
            PathExecutionStrategy::ParallelAlternatives => {
                hints.push("Executing alternatives in parallel".to_string());
            }
            _ => {}
        }

        if self.has_repetition(path) {
            hints.push("Consider limiting depth for repetition operators".to_string());
        }

        if self.has_inverse(path) {
            hints.push("Inverse traversal may be slower without index".to_string());
        }

        hints
    }

    fn path_to_string(&self, path: &PathPattern) -> String {
        match path {
            PathPattern::Property(prop) => prop.clone(),
            PathPattern::Sequence(seq) => seq
                .iter()
                .map(|p| self.path_to_string(p))
                .collect::<Vec<_>>()
                .join("/"),
            PathPattern::Alternative(alts) => {
                format!(
                    "({})",
                    alts.iter()
                        .map(|p| self.path_to_string(p))
                        .collect::<Vec<_>>()
                        .join("|")
                )
            }
            PathPattern::Inverse(inner) => format!("^{}", self.path_to_string(inner)),
            PathPattern::ZeroOrMore(inner) => format!("{}*", self.path_to_string(inner)),
            PathPattern::OneOrMore(inner) => format!("{}+", self.path_to_string(inner)),
            PathPattern::Optional(inner) => format!("{}?", self.path_to_string(inner)),
            PathPattern::Repetition { pattern, min, max } => {
                if let Some(max) = max {
                    format!("{}{{{},{}}}", self.path_to_string(pattern), min, max)
                } else {
                    format!("{}{{{},}}", self.path_to_string(pattern), min)
                }
            }
            PathPattern::NegatedPropertySet(props) => {
                format!("![{}]", props.join(","))
            }
            PathPattern::Any => "*".to_string(),
        }
    }

    fn strategy_to_legacy(&self, strategy: &PathExecutionStrategy) -> PathStrategy {
        match strategy {
            PathExecutionStrategy::ForwardTraversal => PathStrategy::ForwardTraversal,
            PathExecutionStrategy::BackwardTraversal => PathStrategy::BackwardTraversal,
            PathExecutionStrategy::BidirectionalMeet { .. } => PathStrategy::BidirectionalMeet,
            PathExecutionStrategy::IndexLookup { .. } => PathStrategy::IndexLookup,
            PathExecutionStrategy::MaterializedView { .. } => PathStrategy::MaterializedView,
            _ => PathStrategy::ForwardTraversal,
        }
    }

    /// Convert local TraversalDirection to sparql::TraversalDirection
    fn convert_direction(
        &self,
        direction: TraversalDirection,
    ) -> crate::handlers::sparql::TraversalDirection {
        match direction {
            TraversalDirection::Forward => crate::handlers::sparql::TraversalDirection::Forward,
            TraversalDirection::Backward => crate::handlers::sparql::TraversalDirection::Backward,
            TraversalDirection::Both => crate::handlers::sparql::TraversalDirection::Both,
        }
    }

    fn convert_to_legacy_steps(&self, steps: &[EnhancedPathStep]) -> Vec<PathStep> {
        steps
            .iter()
            .map(|step| {
                let (operation, predicate, direction) = match &step.operation {
                    PathOperation::Traverse {
                        predicate,
                        direction,
                    } => (
                        "traverse".to_string(),
                        Some(predicate.clone()),
                        self.convert_direction(*direction),
                    ),
                    PathOperation::TransitiveClosure { predicate, .. } => (
                        "transitive_closure".to_string(),
                        Some(predicate.clone()),
                        crate::handlers::sparql::TraversalDirection::Forward,
                    ),
                    PathOperation::Union(_) => (
                        "union".to_string(),
                        None,
                        crate::handlers::sparql::TraversalDirection::Forward,
                    ),
                    _ => (
                        "unknown".to_string(),
                        None,
                        crate::handlers::sparql::TraversalDirection::Forward,
                    ),
                };

                PathStep {
                    operation,
                    predicate,
                    direction,
                    estimated_selectivity: step.estimated_selectivity,
                }
            })
            .collect()
    }

    // Cache management
    async fn get_cached_path(&self, path: &str) -> FusekiResult<Option<OptimizedPath>> {
        let cache = self.path_cache.read().await;
        Ok(cache.get(path).cloned())
    }

    async fn cache_optimized_path(
        &self,
        path: &str,
        optimized: &OptimizedPath,
    ) -> FusekiResult<()> {
        let mut cache = self.path_cache.write().await;

        // Implement LRU eviction if cache is too large
        if cache.len() > 1000 {
            // Simple eviction - remove oldest entries
            let to_remove: Vec<String> = cache.keys().take(100).cloned().collect();
            for key in to_remove {
                cache.remove(&key);
            }
        }

        cache.insert(path.to_string(), optimized.clone());
        Ok(())
    }

    // Statistics recording
    async fn record_cache_hit(&self) {
        let mut stats = self.statistics.write().await;
        stats.cache_hits += 1;
    }

    async fn record_cache_miss(&self) {
        let mut stats = self.statistics.write().await;
        stats.cache_misses += 1;
    }

    async fn record_optimization_stats(&self, path: &str, time_ms: f64, success: bool) {
        let mut stats = self.statistics.write().await;
        stats.total_executions += 1;

        // Update running average
        let n = stats.total_executions as f64;
        stats.average_execution_time_ms =
            (stats.average_execution_time_ms * (n - 1.0) + time_ms) / n;

        if success {
            stats.optimization_successes += 1;
        } else {
            stats.optimization_failures += 1;
        }

        // Track path frequency
        *stats.path_frequency.entry(path.to_string()).or_insert(0) += 1;
    }
}

/// Path characteristics for optimization decisions
#[derive(Debug, Clone)]
struct PathCharacteristics {
    estimated_length: usize,
    has_cycles: bool,
    has_alternatives: bool,
    has_repetition: bool,
    has_inverse: bool,
    is_linear: bool,
    estimated_branching_factor: f64,
    can_use_index: bool,
}

// Import legacy types for compatibility
use crate::handlers::sparql::{OptimizedPath, PathExecutionPlan, PathStep, PathStrategy};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_property_path_parsing() {
        let optimizer = AdvancedPropertyPathOptimizer::new();

        // Test simple property
        let path = optimizer.parse_property_path("foaf:knows").unwrap();
        matches!(path, PathPattern::Property(p) if p == "foaf:knows");

        // Test sequence
        let path = optimizer
            .parse_property_path("foaf:knows/foaf:name")
            .unwrap();
        matches!(path, PathPattern::Sequence(seq) if seq.len() == 2);

        // Test alternative
        let path = optimizer
            .parse_property_path("foaf:knows|foaf:member")
            .unwrap();
        matches!(path, PathPattern::Alternative(alts) if alts.len() == 2);

        // Test inverse
        let path = optimizer.parse_property_path("^foaf:knows").unwrap();
        matches!(path, PathPattern::Inverse(_));

        // Test zero or more
        let path = optimizer.parse_property_path("foaf:knows*").unwrap();
        matches!(path, PathPattern::ZeroOrMore(_));

        // Test one or more
        let path = optimizer.parse_property_path("foaf:knows+").unwrap();
        matches!(path, PathPattern::OneOrMore(_));
    }

    #[tokio::test]
    async fn test_path_optimization() {
        let optimizer = AdvancedPropertyPathOptimizer::new();

        // Add some indexes
        {
            let mut index_info = optimizer.index_info.write().await;
            index_info.property_indexes.insert("foaf:knows".to_string());
            index_info
                .property_indexes
                .insert("rdfs:subClassOf".to_string());
        }

        // Test optimization with index
        let result = optimizer.optimize_path("foaf:knows").await.unwrap();
        assert!(result.execution_plan.estimated_cost < 10.0); // Should be low due to index

        // Test optimization without index
        let result = optimizer.optimize_path("ex:unknownProperty").await.unwrap();
        assert!(result.execution_plan.estimated_cost >= 10.0); // Should be higher without index

        // Test transitive path optimization
        let result = optimizer.optimize_path("rdfs:subClassOf+").await.unwrap();
        assert!(result.optimized_form.contains("subClassOf")); // Should maintain the property
    }

    #[tokio::test]
    async fn test_rewrite_rules() {
        let optimizer = AdvancedPropertyPathOptimizer::new();

        // Test double inverse elimination
        let path = PathPattern::Inverse(Box::new(PathPattern::Inverse(Box::new(
            PathPattern::Property("test".to_string()),
        ))));
        let rewritten = optimizer.apply_rewrite_rules(&path).await.unwrap();
        matches!(rewritten, PathPattern::Property(p) if p == "test");
    }

    #[tokio::test]
    async fn test_strategy_selection() {
        let optimizer = AdvancedPropertyPathOptimizer::new();

        // Simple property should use forward traversal
        let path = PathPattern::Property("test".to_string());
        let chars = optimizer.analyze_path_characteristics(&path).await.unwrap();
        let strategy = optimizer
            .choose_optimal_strategy(&path, &chars)
            .await
            .unwrap();
        matches!(strategy, PathExecutionStrategy::ForwardTraversal);

        // Path with repetition should use breadth-first or depth-first
        let path = PathPattern::OneOrMore(Box::new(PathPattern::Property("test".to_string())));
        let chars = optimizer.analyze_path_characteristics(&path).await.unwrap();
        let strategy = optimizer
            .choose_optimal_strategy(&path, &chars)
            .await
            .unwrap();
        matches!(
            strategy,
            PathExecutionStrategy::BreadthFirst { .. } | PathExecutionStrategy::DepthFirst { .. }
        );
    }
}
