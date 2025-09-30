//! Pattern matching optimization for query execution
//!
//! This module provides specialized pattern matching optimization that
//! leverages multi-index strategies (SPO/POS/OSP) for efficient query execution.

#![allow(dead_code)]

use crate::indexing::IndexStats as BaseIndexStats;
use crate::model::pattern::{
    ObjectPattern, PredicatePattern, SubjectPattern, TriplePattern as ModelTriplePattern,
};
use crate::model::*;
use crate::query::algebra::{AlgebraTriplePattern, TermPattern as AlgebraTermPattern};
use crate::store::IndexedGraph;
use crate::OxirsError;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Extended index statistics for pattern optimization
#[derive(Debug, Default)]
pub struct IndexStats {
    /// Base statistics
    base_stats: Arc<BaseIndexStats>,
    /// Predicate occurrence counts
    pub predicate_counts: std::sync::RwLock<HashMap<String, usize>>,
    /// Total number of triples
    pub total_triples: std::sync::atomic::AtomicUsize,
    /// Subject cardinality estimates
    pub subject_cardinality: std::sync::RwLock<HashMap<String, usize>>,
    /// Object cardinality estimates  
    pub object_cardinality: std::sync::RwLock<HashMap<String, usize>>,
    /// Join selectivity cache
    pub join_selectivity_cache: std::sync::RwLock<HashMap<String, f64>>,
}

impl IndexStats {
    /// Create new index statistics
    pub fn new() -> Self {
        Self {
            base_stats: Arc::new(BaseIndexStats::new()),
            predicate_counts: std::sync::RwLock::new(HashMap::new()),
            total_triples: std::sync::atomic::AtomicUsize::new(0),
            subject_cardinality: std::sync::RwLock::new(HashMap::new()),
            object_cardinality: std::sync::RwLock::new(HashMap::new()),
            join_selectivity_cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Update predicate count
    pub fn update_predicate_count(&self, predicate: &str, count: usize) {
        if let Ok(mut counts) = self.predicate_counts.write() {
            counts.insert(predicate.to_string(), count);
        }
    }

    /// Set total triples
    pub fn set_total_triples(&self, count: usize) {
        self.total_triples
            .store(count, std::sync::atomic::Ordering::Relaxed);
    }

    /// Update subject cardinality estimate
    pub fn update_subject_cardinality(&self, predicate: &str, cardinality: usize) {
        if let Ok(mut card) = self.subject_cardinality.write() {
            card.insert(predicate.to_string(), cardinality);
        }
    }

    /// Update object cardinality estimate
    pub fn update_object_cardinality(&self, predicate: &str, cardinality: usize) {
        if let Ok(mut card) = self.object_cardinality.write() {
            card.insert(predicate.to_string(), cardinality);
        }
    }

    /// Cache join selectivity
    pub fn cache_join_selectivity(&self, pattern_pair: &str, selectivity: f64) {
        if let Ok(mut cache) = self.join_selectivity_cache.write() {
            cache.insert(pattern_pair.to_string(), selectivity);
        }
    }

    /// Get cached join selectivity
    pub fn get_join_selectivity(&self, pattern_pair: &str) -> Option<f64> {
        match self.join_selectivity_cache.read() {
            Ok(cache) => cache.get(pattern_pair).copied(),
            _ => None,
        }
    }
}

/// Pattern matching optimizer that selects optimal indexes
#[derive(Debug)]
pub struct PatternOptimizer {
    /// Index statistics for cost estimation
    index_stats: Arc<IndexStats>,
    /// Available index types
    available_indexes: Vec<IndexType>,
}

/// Types of indexes available for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// Subject-Predicate-Object index
    SPO,
    /// Predicate-Object-Subject index
    POS,
    /// Object-Subject-Predicate index
    OSP,
    /// Subject-Object-Predicate index (optional)
    SOP,
    /// Predicate-Subject-Object index (optional)
    PSO,
    /// Object-Predicate-Subject index (optional)
    OPS,
}

/// Variable position in a triple pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VarPosition {
    Subject,
    Predicate,
    Object,
}

/// Filter expression for evaluation optimization
#[derive(Debug, Clone)]
pub enum FilterExpression {
    /// Equality comparison
    Equals(Variable, Term),
    /// Less than comparison
    LessThan(Variable, Term),
    /// Greater than comparison
    GreaterThan(Variable, Term),
    /// String regex match
    Regex(Variable, String),
    /// In list
    In(Variable, Vec<Term>),
    /// Logical AND
    And(Box<FilterExpression>, Box<FilterExpression>),
    /// Logical OR
    Or(Box<FilterExpression>, Box<FilterExpression>),
    /// Logical NOT
    Not(Box<FilterExpression>),
}

/// Pattern matching strategy
#[derive(Debug, Clone)]
pub struct PatternStrategy {
    /// Index to use for this pattern
    pub index_type: IndexType,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Selectivity estimate (0.0 to 1.0)
    pub selectivity: f64,
    /// Variables that will be bound after executing this pattern
    pub bound_vars: HashSet<Variable>,
    /// Associated filter expressions that can be pushed down
    pub pushdown_filters: Vec<FilterExpression>,
}

/// Optimized pattern execution order
#[derive(Debug, Clone)]
pub struct OptimizedPatternPlan {
    /// Ordered list of patterns with their strategies
    pub patterns: Vec<(AlgebraTriplePattern, PatternStrategy)>,
    /// Total estimated cost
    pub total_cost: f64,
    /// Variables bound at each step
    pub binding_order: Vec<HashSet<Variable>>,
}

impl PatternOptimizer {
    /// Create a new pattern optimizer
    pub fn new(index_stats: Arc<IndexStats>) -> Self {
        Self {
            index_stats,
            available_indexes: vec![IndexType::SPO, IndexType::POS, IndexType::OSP],
        }
    }

    /// Optimize a set of triple patterns
    pub fn optimize_patterns(
        &self,
        patterns: &[AlgebraTriplePattern],
    ) -> Result<OptimizedPatternPlan, OxirsError> {
        if patterns.is_empty() {
            return Ok(OptimizedPatternPlan {
                patterns: Vec::new(),
                total_cost: 0.0,
                binding_order: Vec::new(),
            });
        }

        // Analyze each pattern individually
        let mut pattern_strategies: Vec<(usize, Vec<PatternStrategy>)> = patterns
            .iter()
            .enumerate()
            .map(|(idx, pattern)| {
                let strategies = self.analyze_pattern(pattern);
                (idx, strategies)
            })
            .collect();

        // Sort patterns by their best selectivity
        pattern_strategies.sort_by(|a, b| {
            let best_a =
                a.1.iter()
                    .map(|s| s.selectivity)
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap_or(1.0);
            let best_b =
                b.1.iter()
                    .map(|s| s.selectivity)
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap_or(1.0);
            best_a.partial_cmp(&best_b).unwrap()
        });

        // Build execution order considering variable bindings
        let mut ordered_patterns = Vec::new();
        let mut bound_vars = HashSet::new();
        let mut binding_order = Vec::new();
        let mut total_cost = 0.0;

        // First pattern - choose the most selective
        let (first_idx, first_strategies) = &pattern_strategies[0];
        let first_pattern = &patterns[*first_idx];
        let first_strategy = self.select_best_strategy(first_strategies, &bound_vars)?;

        ordered_patterns.push((first_pattern.clone(), first_strategy.clone()));
        bound_vars.extend(first_strategy.bound_vars.clone());
        binding_order.push(bound_vars.clone());
        total_cost += first_strategy.estimated_cost;

        // Process remaining patterns
        let mut remaining: Vec<_> = pattern_strategies[1..].to_vec();

        while !remaining.is_empty() {
            // Find pattern with best cost considering current bindings
            let (best_pos, best_idx, best_strategy) = remaining
                .iter()
                .enumerate()
                .map(|(pos, (idx, strategies))| {
                    let pattern = &patterns[*idx];
                    let strategy =
                        self.select_strategy_with_bindings(pattern, strategies, &bound_vars);
                    (pos, idx, strategy)
                })
                .min_by(|(_, _, a), (_, _, b)| {
                    a.estimated_cost.partial_cmp(&b.estimated_cost).unwrap()
                })
                .ok_or_else(|| OxirsError::Query("Failed to find next pattern".to_string()))?;

            let pattern = &patterns[*best_idx];
            ordered_patterns.push((pattern.clone(), best_strategy.clone()));
            bound_vars.extend(best_strategy.bound_vars.clone());
            binding_order.push(bound_vars.clone());
            total_cost += best_strategy.estimated_cost;

            remaining.remove(best_pos);
        }

        Ok(OptimizedPatternPlan {
            patterns: ordered_patterns,
            total_cost,
            binding_order,
        })
    }

    /// Analyze a single pattern and generate strategies
    pub fn analyze_pattern(&self, pattern: &AlgebraTriplePattern) -> Vec<PatternStrategy> {
        let mut strategies = Vec::new();

        // Analyze which components are bound vs variable
        let s_bound = !matches!(pattern.subject, AlgebraTermPattern::Variable(_));
        let p_bound = !matches!(pattern.predicate, AlgebraTermPattern::Variable(_));
        let o_bound = !matches!(pattern.object, AlgebraTermPattern::Variable(_));

        // Generate strategies for each index type
        for &index_type in &self.available_indexes {
            let (cost, selectivity) =
                self.estimate_index_cost(index_type, s_bound, p_bound, o_bound, pattern);

            let bound_vars = self.extract_bound_vars(pattern);

            strategies.push(PatternStrategy {
                index_type,
                estimated_cost: cost,
                selectivity,
                bound_vars,
                pushdown_filters: Vec::new(), // Will be populated by filter optimization
            });
        }

        strategies
    }

    /// Estimate cost of using a specific index
    fn estimate_index_cost(
        &self,
        index_type: IndexType,
        s_bound: bool,
        p_bound: bool,
        o_bound: bool,
        pattern: &AlgebraTriplePattern,
    ) -> (f64, f64) {
        // Base costs for different access patterns
        let base_cost = match index_type {
            IndexType::SPO => {
                if s_bound && p_bound && o_bound {
                    1.0 // Exact lookup
                } else if s_bound && p_bound {
                    10.0 // Range scan on O
                } else if s_bound {
                    100.0 // Range scan on P,O
                } else {
                    10000.0 // Full scan
                }
            }
            IndexType::POS => {
                if p_bound && o_bound && s_bound {
                    1.0
                } else if p_bound && o_bound {
                    10.0
                } else if p_bound {
                    100.0
                } else {
                    10000.0
                }
            }
            IndexType::OSP => {
                if o_bound && s_bound && p_bound {
                    1.0
                } else if o_bound && s_bound {
                    10.0
                } else if o_bound {
                    100.0
                } else {
                    10000.0
                }
            }
            _ => 10000.0, // Other indexes not implemented
        };

        // Adjust based on statistics
        let selectivity = self.estimate_selectivity(pattern);
        let adjusted_cost = base_cost * selectivity;

        (adjusted_cost, selectivity)
    }

    /// Estimate selectivity of a pattern using advanced statistics
    fn estimate_selectivity(&self, pattern: &AlgebraTriplePattern) -> f64 {
        // Base selectivity
        let mut selectivity: f64 = 1.0;
        let total_triples = self
            .index_stats
            .total_triples
            .load(std::sync::atomic::Ordering::Relaxed) as f64;

        if total_triples == 0.0 {
            return 0.001; // Default for empty dataset
        }

        // Subject selectivity
        match &pattern.subject {
            AlgebraTermPattern::NamedNode(_) | AlgebraTermPattern::BlankNode(_) => {
                // Bound subject - highly selective
                selectivity *= 0.001;
            }
            AlgebraTermPattern::Variable(_) => {
                // Variable subject - depends on predicate cardinality
                if let AlgebraTermPattern::NamedNode(pred) = &pattern.predicate {
                    if let Ok(card) = self.index_stats.subject_cardinality.read() {
                        if let Some(subj_card) = card.get(pred.as_str()) {
                            selectivity *= (*subj_card as f64) / total_triples;
                        }
                    }
                }
            }
            _ => {}
        }

        // Predicate selectivity (most important for triple stores)
        match &pattern.predicate {
            AlgebraTermPattern::NamedNode(pred) => {
                if let Ok(counts) = self.index_stats.predicate_counts.read() {
                    if let Some(pred_count) = counts.get(pred.as_str()) {
                        selectivity *= (*pred_count as f64) / total_triples;
                    } else {
                        // Unknown predicate - assume very low frequency
                        selectivity *= 0.001;
                    }
                }
            }
            AlgebraTermPattern::Variable(_) => {
                // Variable predicate - less selective
                selectivity *= 0.5;
            }
            _ => {}
        }

        // Object selectivity
        match &pattern.object {
            AlgebraTermPattern::Literal(_) => {
                // Literals are usually very selective
                selectivity *= 0.01;
            }
            AlgebraTermPattern::NamedNode(_) => {
                // Named nodes moderately selective
                selectivity *= 0.1;
            }
            AlgebraTermPattern::BlankNode(_) => {
                // Blank nodes moderately selective
                selectivity *= 0.1;
            }
            AlgebraTermPattern::Variable(_) => {
                // Variable object - depends on predicate object cardinality
                if let AlgebraTermPattern::NamedNode(pred) = &pattern.predicate {
                    if let Ok(card) = self.index_stats.object_cardinality.read() {
                        if let Some(obj_card) = card.get(pred.as_str()) {
                            selectivity *= (*obj_card as f64) / total_triples;
                        }
                    }
                }
            }
        }

        // Apply bounds and handle edge cases
        selectivity.clamp(0.00001, 1.0)
    }

    /// Extract variables that will be bound by this pattern
    fn extract_bound_vars(&self, pattern: &AlgebraTriplePattern) -> HashSet<Variable> {
        let mut vars = HashSet::new();

        if let AlgebraTermPattern::Variable(v) = &pattern.subject {
            vars.insert(v.clone());
        }
        if let AlgebraTermPattern::Variable(v) = &pattern.predicate {
            vars.insert(v.clone());
        }
        if let AlgebraTermPattern::Variable(v) = &pattern.object {
            vars.insert(v.clone());
        }

        vars
    }

    /// Select best strategy from options
    fn select_best_strategy(
        &self,
        strategies: &[PatternStrategy],
        _bound_vars: &HashSet<Variable>,
    ) -> Result<PatternStrategy, OxirsError> {
        strategies
            .iter()
            .min_by(|a, b| a.estimated_cost.partial_cmp(&b.estimated_cost).unwrap())
            .cloned()
            .ok_or_else(|| OxirsError::Query("No valid strategy found".to_string()))
    }

    /// Select strategy considering already bound variables
    fn select_strategy_with_bindings(
        &self,
        pattern: &AlgebraTriplePattern,
        strategies: &[PatternStrategy],
        bound_vars: &HashSet<Variable>,
    ) -> PatternStrategy {
        // Check which variables in pattern are already bound
        let s_bound = match &pattern.subject {
            AlgebraTermPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        };
        let p_bound = match &pattern.predicate {
            AlgebraTermPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        };
        let o_bound = match &pattern.object {
            AlgebraTermPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        };

        // Re-evaluate strategies with bound variables
        let mut best_strategy = strategies[0].clone();
        let mut best_cost = f64::MAX;

        for strategy in strategies {
            let (adjusted_cost, _) =
                self.estimate_index_cost(strategy.index_type, s_bound, p_bound, o_bound, pattern);

            if adjusted_cost < best_cost {
                best_cost = adjusted_cost;
                best_strategy = strategy.clone();
                best_strategy.estimated_cost = adjusted_cost;
            }
        }

        best_strategy
    }

    /// Estimate join selectivity between two patterns
    pub fn estimate_join_selectivity(
        &self,
        left: &AlgebraTriplePattern,
        right: &AlgebraTriplePattern,
    ) -> f64 {
        // Create cache key for this pattern pair
        let cache_key = format!("{left:?}|{right:?}");

        // Check cache first
        if let Some(cached) = self.index_stats.get_join_selectivity(&cache_key) {
            return cached;
        }

        // Find common variables
        let left_vars = self.extract_bound_vars(left);
        let right_vars = self.extract_bound_vars(right);
        let common_vars: HashSet<_> = left_vars.intersection(&right_vars).cloned().collect();

        let selectivity = if common_vars.is_empty() {
            // Cartesian product - very expensive
            1.0
        } else {
            // Estimate based on type of join variables
            let mut join_selectivity = 1.0;

            for var in common_vars.iter() {
                // Estimate selectivity based on variable position and pattern types
                let var_selectivity = self.estimate_variable_join_selectivity(var, left, right);
                join_selectivity *= var_selectivity;
            }

            // Apply correlation factor for multiple join variables
            if common_vars.len() > 1 {
                join_selectivity *= 0.1_f64.powf(common_vars.len() as f64 - 1.0);
            }

            join_selectivity
        };

        // Cache the result
        self.index_stats
            .cache_join_selectivity(&cache_key, selectivity);

        selectivity.clamp(0.00001, 1.0)
    }

    /// Estimate selectivity for a variable join
    fn estimate_variable_join_selectivity(
        &self,
        var: &Variable,
        left: &AlgebraTriplePattern,
        right: &AlgebraTriplePattern,
    ) -> f64 {
        // Find position of variable in each pattern
        let left_pos = self.find_variable_position(var, left);
        let right_pos = self.find_variable_position(var, right);

        match (left_pos, right_pos) {
            (Some(pos1), Some(pos2)) => {
                // Subject-subject joins are usually more selective than object-object
                match (pos1, pos2) {
                    (VarPosition::Subject, VarPosition::Subject) => 0.01, // Very selective
                    (VarPosition::Subject, VarPosition::Object) => 0.1,   // Moderately selective
                    (VarPosition::Object, VarPosition::Subject) => 0.1,   // Moderately selective
                    (VarPosition::Object, VarPosition::Object) => 0.2,    // Less selective
                    (VarPosition::Predicate, _) | (_, VarPosition::Predicate) => 0.05, // Predicate joins rare but selective
                }
            }
            _ => 1.0, // No actual join
        }
    }

    /// Find position of variable in pattern
    fn find_variable_position(
        &self,
        var: &Variable,
        pattern: &AlgebraTriplePattern,
    ) -> Option<VarPosition> {
        if let AlgebraTermPattern::Variable(v) = &pattern.subject {
            if v == var {
                return Some(VarPosition::Subject);
            }
        }
        if let AlgebraTermPattern::Variable(v) = &pattern.predicate {
            if v == var {
                return Some(VarPosition::Predicate);
            }
        }
        if let AlgebraTermPattern::Variable(v) = &pattern.object {
            if v == var {
                return Some(VarPosition::Object);
            }
        }
        None
    }

    /// Optimize filter expressions and determine pushdown opportunities
    pub fn optimize_filters(
        &self,
        patterns: &[AlgebraTriplePattern],
        filters: &[FilterExpression],
    ) -> Vec<(usize, Vec<FilterExpression>)> {
        let mut pushdown_map = Vec::new();

        for (pattern_idx, pattern) in patterns.iter().enumerate() {
            let mut pattern_filters = Vec::new();

            // Find filters that can be pushed down to this pattern
            for filter in filters {
                if self.can_pushdown_filter(filter, pattern) {
                    pattern_filters.push(filter.clone());
                }
            }

            if !pattern_filters.is_empty() {
                pushdown_map.push((pattern_idx, pattern_filters));
            }
        }

        pushdown_map
    }

    /// Check if filter can be pushed down to pattern
    fn can_pushdown_filter(
        &self,
        filter: &FilterExpression,
        pattern: &AlgebraTriplePattern,
    ) -> bool {
        match filter {
            FilterExpression::Equals(var, _)
            | FilterExpression::LessThan(var, _)
            | FilterExpression::GreaterThan(var, _)
            | FilterExpression::Regex(var, _)
            | FilterExpression::In(var, _) => {
                // Can push down if pattern binds this variable
                self.pattern_binds_variable(var, pattern)
            }
            FilterExpression::And(left, right) => {
                // Can push down if either side can be pushed down
                self.can_pushdown_filter(left, pattern) || self.can_pushdown_filter(right, pattern)
            }
            FilterExpression::Or(left, right) => {
                // Can only push down if both sides can be pushed down
                self.can_pushdown_filter(left, pattern) && self.can_pushdown_filter(right, pattern)
            }
            FilterExpression::Not(inner) => self.can_pushdown_filter(inner, pattern),
        }
    }

    /// Check if pattern binds variable
    fn pattern_binds_variable(&self, var: &Variable, pattern: &AlgebraTriplePattern) -> bool {
        matches!(&pattern.subject, AlgebraTermPattern::Variable(v) if v == var)
            || matches!(&pattern.predicate, AlgebraTermPattern::Variable(v) if v == var)
            || matches!(&pattern.object, AlgebraTermPattern::Variable(v) if v == var)
    }

    /// Estimate filter selectivity
    #[allow(clippy::only_used_in_recursion)]
    pub fn estimate_filter_selectivity(&self, filter: &FilterExpression) -> f64 {
        match filter {
            FilterExpression::Equals(_, _) => 0.1, // Equality is selective
            FilterExpression::LessThan(_, _) => 0.3, // Range filters moderately selective
            FilterExpression::GreaterThan(_, _) => 0.3,
            FilterExpression::Regex(_, _) => 0.2, // String matches moderately selective
            FilterExpression::In(_, values) => {
                // Selectivity depends on number of values
                (values.len() as f64 * 0.1).min(0.9)
            }
            FilterExpression::And(left, right) => {
                // AND is more selective
                self.estimate_filter_selectivity(left) * self.estimate_filter_selectivity(right)
            }
            FilterExpression::Or(left, right) => {
                // OR is less selective
                let left_sel = self.estimate_filter_selectivity(left);
                let right_sel = self.estimate_filter_selectivity(right);
                left_sel + right_sel - (left_sel * right_sel)
            }
            FilterExpression::Not(inner) => {
                // NOT inverts selectivity
                1.0 - self.estimate_filter_selectivity(inner)
            }
        }
    }

    /// Get optimal index type for a pattern execution
    pub fn get_optimal_index(
        &self,
        pattern: &ModelTriplePattern,
        bound_vars: &HashSet<Variable>,
    ) -> IndexType {
        // Check which components are bound
        let s_bound = pattern.subject.as_ref().is_some_and(|s| match s {
            SubjectPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        });

        let p_bound = pattern.predicate.as_ref().is_some_and(|p| match p {
            PredicatePattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        });

        let o_bound = pattern.object.as_ref().is_some_and(|o| match o {
            ObjectPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        });

        // Select index based on bound components
        match (s_bound, p_bound, o_bound) {
            (true, true, _) => IndexType::SPO,
            (true, false, true) => IndexType::SPO, // Could use SOP if available
            (false, true, true) => IndexType::POS,
            (true, false, false) => IndexType::SPO,
            (false, true, false) => IndexType::POS,
            (false, false, true) => IndexType::OSP,
            (false, false, false) => IndexType::SPO, // Full scan, any index works
        }
    }
}

/// Pattern execution engine that uses optimized strategies
pub struct PatternExecutor {
    /// The indexed graph to query
    graph: Arc<IndexedGraph>,
    /// Pattern optimizer
    optimizer: PatternOptimizer,
}

impl PatternExecutor {
    /// Create new pattern executor
    pub fn new(graph: Arc<IndexedGraph>, index_stats: Arc<IndexStats>) -> Self {
        Self {
            graph,
            optimizer: PatternOptimizer::new(index_stats),
        }
    }

    /// Execute an optimized pattern plan
    pub fn execute_plan(
        &self,
        plan: &OptimizedPatternPlan,
    ) -> Result<Vec<HashMap<Variable, Term>>, OxirsError> {
        let mut results = vec![HashMap::new()];

        for (pattern, strategy) in &plan.patterns {
            results = self.execute_pattern_with_strategy(pattern, strategy, results)?;
        }

        Ok(results)
    }

    /// Execute a single pattern with given strategy
    fn execute_pattern_with_strategy(
        &self,
        pattern: &AlgebraTriplePattern,
        _strategy: &PatternStrategy,
        bindings: Vec<HashMap<Variable, Term>>,
    ) -> Result<Vec<HashMap<Variable, Term>>, OxirsError> {
        let mut new_results = Vec::new();

        for binding in bindings {
            // Convert algebra pattern to model pattern with bindings
            let bound_pattern = self.bind_pattern(pattern, &binding)?;

            // Convert pattern types to concrete types for query
            let subject = bound_pattern
                .subject
                .as_ref()
                .and_then(|s| self.subject_pattern_to_subject(s));
            let predicate = bound_pattern
                .predicate
                .as_ref()
                .and_then(|p| self.predicate_pattern_to_predicate(p));
            let object = bound_pattern
                .object
                .as_ref()
                .and_then(|o| self.object_pattern_to_object(o));

            // Query using selected index
            let matches = self
                .graph
                .query(subject.as_ref(), predicate.as_ref(), object.as_ref());

            // Extend bindings with new matches
            for triple in matches {
                let mut new_binding = binding.clone();

                // Bind variables from matched triple
                if let AlgebraTermPattern::Variable(v) = &pattern.subject {
                    new_binding.insert(v.clone(), Term::from(triple.subject().clone()));
                }
                if let AlgebraTermPattern::Variable(v) = &pattern.predicate {
                    new_binding.insert(v.clone(), Term::from(triple.predicate().clone()));
                }
                if let AlgebraTermPattern::Variable(v) = &pattern.object {
                    new_binding.insert(v.clone(), Term::from(triple.object().clone()));
                }

                new_results.push(new_binding);
            }
        }

        Ok(new_results)
    }

    /// Bind variables in pattern using current bindings
    fn bind_pattern(
        &self,
        pattern: &AlgebraTriplePattern,
        bindings: &HashMap<Variable, Term>,
    ) -> Result<ModelTriplePattern, OxirsError> {
        let subject = match &pattern.subject {
            AlgebraTermPattern::Variable(v) => {
                if let Some(term) = bindings.get(v) {
                    Some(self.term_to_subject_pattern(term)?)
                } else {
                    None
                }
            }
            AlgebraTermPattern::NamedNode(n) => Some(SubjectPattern::NamedNode(n.clone())),
            AlgebraTermPattern::BlankNode(b) => Some(SubjectPattern::BlankNode(b.clone())),
            AlgebraTermPattern::Literal(_) => {
                return Err(OxirsError::Query("Literal cannot be subject".to_string()))
            }
        };

        let predicate = match &pattern.predicate {
            AlgebraTermPattern::Variable(v) => {
                if let Some(term) = bindings.get(v) {
                    Some(self.term_to_predicate_pattern(term)?)
                } else {
                    None
                }
            }
            AlgebraTermPattern::NamedNode(n) => Some(PredicatePattern::NamedNode(n.clone())),
            _ => return Err(OxirsError::Query("Invalid predicate pattern".to_string())),
        };

        let object = match &pattern.object {
            AlgebraTermPattern::Variable(v) => {
                if let Some(term) = bindings.get(v) {
                    Some(self.term_to_object_pattern(term)?)
                } else {
                    None
                }
            }
            AlgebraTermPattern::NamedNode(n) => Some(ObjectPattern::NamedNode(n.clone())),
            AlgebraTermPattern::BlankNode(b) => Some(ObjectPattern::BlankNode(b.clone())),
            AlgebraTermPattern::Literal(l) => Some(ObjectPattern::Literal(l.clone())),
        };

        Ok(ModelTriplePattern::new(subject, predicate, object))
    }

    /// Convert term to subject pattern
    fn term_to_subject_pattern(&self, term: &Term) -> Result<SubjectPattern, OxirsError> {
        match term {
            Term::NamedNode(n) => Ok(SubjectPattern::NamedNode(n.clone())),
            Term::BlankNode(b) => Ok(SubjectPattern::BlankNode(b.clone())),
            _ => Err(OxirsError::Query("Invalid term for subject".to_string())),
        }
    }

    /// Convert term to predicate pattern
    fn term_to_predicate_pattern(&self, term: &Term) -> Result<PredicatePattern, OxirsError> {
        match term {
            Term::NamedNode(n) => Ok(PredicatePattern::NamedNode(n.clone())),
            _ => Err(OxirsError::Query("Invalid term for predicate".to_string())),
        }
    }

    /// Convert term to object pattern
    fn term_to_object_pattern(&self, term: &Term) -> Result<ObjectPattern, OxirsError> {
        match term {
            Term::NamedNode(n) => Ok(ObjectPattern::NamedNode(n.clone())),
            Term::BlankNode(b) => Ok(ObjectPattern::BlankNode(b.clone())),
            Term::Literal(l) => Ok(ObjectPattern::Literal(l.clone())),
            _ => Err(OxirsError::Query(
                "Invalid term for object pattern".to_string(),
            )),
        }
    }

    /// Convert subject pattern to subject
    fn subject_pattern_to_subject(&self, pattern: &SubjectPattern) -> Option<Subject> {
        match pattern {
            SubjectPattern::NamedNode(n) => Some(Subject::NamedNode(n.clone())),
            SubjectPattern::BlankNode(b) => Some(Subject::BlankNode(b.clone())),
            SubjectPattern::Variable(_) => None,
        }
    }

    /// Convert predicate pattern to predicate
    fn predicate_pattern_to_predicate(&self, pattern: &PredicatePattern) -> Option<Predicate> {
        match pattern {
            PredicatePattern::NamedNode(n) => Some(Predicate::NamedNode(n.clone())),
            PredicatePattern::Variable(_) => None,
        }
    }

    /// Convert object pattern to object
    fn object_pattern_to_object(&self, pattern: &ObjectPattern) -> Option<Object> {
        match pattern {
            ObjectPattern::NamedNode(n) => Some(Object::NamedNode(n.clone())),
            ObjectPattern::BlankNode(b) => Some(Object::BlankNode(b.clone())),
            ObjectPattern::Literal(l) => Some(Object::Literal(l.clone())),
            ObjectPattern::Variable(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_optimizer_creation() {
        let stats = Arc::new(IndexStats::new());
        let optimizer = PatternOptimizer::new(stats);

        assert_eq!(optimizer.available_indexes.len(), 3);
    }

    #[test]
    fn test_index_selection() {
        let stats = Arc::new(IndexStats::new());
        let optimizer = PatternOptimizer::new(stats);

        // Pattern with bound subject
        let pattern = ModelTriplePattern::new(
            Some(SubjectPattern::NamedNode(
                NamedNode::new("http://example.org/s").unwrap(),
            )),
            None,
            None,
        );

        let bound_vars = HashSet::new();
        let index = optimizer.get_optimal_index(&pattern, &bound_vars);

        assert_eq!(index, IndexType::SPO);
    }

    #[test]
    fn test_selectivity_estimation() {
        let stats = Arc::new(IndexStats::new());
        let optimizer = PatternOptimizer::new(stats);

        let pattern = AlgebraTriplePattern::new(
            AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
            AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
            AlgebraTermPattern::Literal(Literal::new("test")),
        );

        let selectivity = optimizer.estimate_selectivity(&pattern);

        // Literal object should give low selectivity
        assert!(selectivity < 0.2);
    }

    #[test]
    fn test_pattern_optimization() {
        let stats = Arc::new(IndexStats::new());
        let optimizer = PatternOptimizer::new(stats);

        let patterns = vec![
            AlgebraTriplePattern::new(
                AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
                AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
                AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/Person").unwrap()),
            ),
            AlgebraTriplePattern::new(
                AlgebraTermPattern::Variable(Variable::new("s").unwrap()),
                AlgebraTermPattern::NamedNode(NamedNode::new("http://example.org/name").unwrap()),
                AlgebraTermPattern::Variable(Variable::new("name").unwrap()),
            ),
        ];

        let plan = optimizer.optimize_patterns(&patterns).unwrap();

        assert_eq!(plan.patterns.len(), 2);
        assert!(plan.total_cost > 0.0);
        assert_eq!(plan.binding_order.len(), 2);
    }
}
