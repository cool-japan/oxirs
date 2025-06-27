//! Pattern matching optimization for query execution
//!
//! This module provides specialized pattern matching optimization that
//! leverages multi-index strategies (SPO/POS/OSP) for efficient query execution.

use crate::indexing::IndexStats as BaseIndexStats;
use crate::model::*;
use crate::store::IndexedGraph;
use crate::query::algebra::{TriplePattern as AlgebraTriplePattern, TermPattern};
use crate::OxirsError;
use std::sync::Arc;
use std::collections::{HashMap, HashSet};

/// Extended index statistics for pattern optimization
#[derive(Debug, Default)]
pub struct IndexStats {
    /// Base statistics
    base_stats: Arc<BaseIndexStats>,
    /// Predicate occurrence counts
    pub predicate_counts: std::sync::RwLock<HashMap<String, usize>>,
    /// Total number of triples
    pub total_triples: std::sync::atomic::AtomicUsize,
}

impl IndexStats {
    /// Create new index statistics
    pub fn new() -> Self {
        Self {
            base_stats: Arc::new(BaseIndexStats::new()),
            predicate_counts: std::sync::RwLock::new(HashMap::new()),
            total_triples: std::sync::atomic::AtomicUsize::new(0),
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
        self.total_triples.store(count, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Pattern matching optimizer that selects optimal indexes
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
}

/// Optimized pattern execution order
#[derive(Debug)]
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
            available_indexes: vec![
                IndexType::SPO,
                IndexType::POS,
                IndexType::OSP,
            ],
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
            let best_a = a.1.iter()
                .map(|s| s.selectivity)
                .min_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap_or(1.0);
            let best_b = b.1.iter()
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
                    let strategy = self.select_strategy_with_bindings(
                        pattern,
                        strategies,
                        &bound_vars,
                    );
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
    fn analyze_pattern(&self, pattern: &AlgebraTriplePattern) -> Vec<PatternStrategy> {
        let mut strategies = Vec::new();

        // Analyze which components are bound vs variable
        let s_bound = !matches!(pattern.subject, TermPattern::Variable(_));
        let p_bound = !matches!(pattern.predicate, TermPattern::Variable(_));
        let o_bound = !matches!(pattern.object, TermPattern::Variable(_));

        // Generate strategies for each index type
        for &index_type in &self.available_indexes {
            let (cost, selectivity) = self.estimate_index_cost(
                index_type,
                s_bound,
                p_bound,
                o_bound,
                pattern,
            );

            let bound_vars = self.extract_bound_vars(pattern);

            strategies.push(PatternStrategy {
                index_type,
                estimated_cost: cost,
                selectivity,
                bound_vars,
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

    /// Estimate selectivity of a pattern
    fn estimate_selectivity(&self, pattern: &AlgebraTriplePattern) -> f64 {
        // Base selectivity
        let mut selectivity: f64 = 1.0;

        // Adjust based on predicate (most selective component usually)
        if let TermPattern::NamedNode(pred) = &pattern.predicate {
            if let Ok(counts) = self.index_stats.predicate_counts.read() {
                if let Some(pred_count) = counts.get(pred.as_str()) {
                    let total_triples = self.index_stats.total_triples.load(std::sync::atomic::Ordering::Relaxed) as f64;
                    if total_triples > 0.0 {
                        selectivity *= (*pred_count as f64) / total_triples;
                    }
                } else {
                    // Unknown predicate - assume low selectivity
                    selectivity *= 0.01;
                }
            }
        }

        // Literals in object position are usually selective
        if let TermPattern::Literal(_) = &pattern.object {
            selectivity *= 0.1;
        }

        selectivity.max(0.0001).min(1.0)
    }

    /// Extract variables that will be bound by this pattern
    fn extract_bound_vars(&self, pattern: &AlgebraTriplePattern) -> HashSet<Variable> {
        let mut vars = HashSet::new();

        if let TermPattern::Variable(v) = &pattern.subject {
            vars.insert(v.clone());
        }
        if let TermPattern::Variable(v) = &pattern.predicate {
            vars.insert(v.clone());
        }
        if let TermPattern::Variable(v) = &pattern.object {
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
            TermPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        };
        let p_bound = match &pattern.predicate {
            TermPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        };
        let o_bound = match &pattern.object {
            TermPattern::Variable(v) => bound_vars.contains(v),
            _ => true,
        };

        // Re-evaluate strategies with bound variables
        let mut best_strategy = strategies[0].clone();
        let mut best_cost = f64::MAX;

        for strategy in strategies {
            let (adjusted_cost, _) = self.estimate_index_cost(
                strategy.index_type,
                s_bound,
                p_bound,
                o_bound,
                pattern,
            );

            if adjusted_cost < best_cost {
                best_cost = adjusted_cost;
                best_strategy = strategy.clone();
                best_strategy.estimated_cost = adjusted_cost;
            }
        }

        best_strategy
    }

    /// Get optimal index type for a pattern execution
    pub fn get_optimal_index(
        &self,
        pattern: &TriplePattern,
        bound_vars: &HashSet<Variable>,
    ) -> IndexType {
        // Check which components are bound
        let s_bound = pattern.subject.as_ref().map_or(false, |s| {
            match s {
                SubjectPattern::Variable(v) => bound_vars.contains(v),
                _ => true,
            }
        });

        let p_bound = pattern.predicate.as_ref().map_or(false, |p| {
            match p {
                PredicatePattern::Variable(v) => bound_vars.contains(v),
                _ => true,
            }
        });

        let o_bound = pattern.object.as_ref().map_or(false, |o| {
            match o {
                ObjectPattern::Variable(v) => bound_vars.contains(v),
                _ => true,
            }
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
        strategy: &PatternStrategy,
        bindings: Vec<HashMap<Variable, Term>>,
    ) -> Result<Vec<HashMap<Variable, Term>>, OxirsError> {
        let mut new_results = Vec::new();

        for binding in bindings {
            // Convert algebra pattern to model pattern with bindings
            let bound_pattern = self.bind_pattern(pattern, &binding)?;
            
            // Convert pattern types to concrete types for query
            let subject = bound_pattern.subject.as_ref().and_then(|s| self.subject_pattern_to_subject(s));
            let predicate = bound_pattern.predicate.as_ref().and_then(|p| self.predicate_pattern_to_predicate(p));
            let object = bound_pattern.object.as_ref().and_then(|o| self.object_pattern_to_object(o));
            
            // Query using selected index
            let matches = self.graph.query(
                subject.as_ref(),
                predicate.as_ref(),
                object.as_ref(),
            );

            // Extend bindings with new matches
            for triple in matches {
                let mut new_binding = binding.clone();
                
                // Bind variables from matched triple
                if let TermPattern::Variable(v) = &pattern.subject {
                    new_binding.insert(v.clone(), Term::from(triple.subject().clone()));
                }
                if let TermPattern::Variable(v) = &pattern.predicate {
                    new_binding.insert(v.clone(), Term::from(triple.predicate().clone()));
                }
                if let TermPattern::Variable(v) = &pattern.object {
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
    ) -> Result<TriplePattern, OxirsError> {
        let subject = match &pattern.subject {
            TermPattern::Variable(v) => {
                if let Some(term) = bindings.get(v) {
                    Some(self.term_to_subject_pattern(term)?)
                } else {
                    None
                }
            }
            TermPattern::NamedNode(n) => Some(SubjectPattern::NamedNode(n.clone())),
            TermPattern::BlankNode(b) => Some(SubjectPattern::BlankNode(b.clone())),
            TermPattern::Literal(_) => {
                return Err(OxirsError::Query("Literal cannot be subject".to_string()))
            }
        };

        let predicate = match &pattern.predicate {
            TermPattern::Variable(v) => {
                if let Some(term) = bindings.get(v) {
                    Some(self.term_to_predicate_pattern(term)?)
                } else {
                    None
                }
            }
            TermPattern::NamedNode(n) => Some(PredicatePattern::NamedNode(n.clone())),
            _ => return Err(OxirsError::Query("Invalid predicate pattern".to_string())),
        };

        let object = match &pattern.object {
            TermPattern::Variable(v) => {
                if let Some(term) = bindings.get(v) {
                    Some(self.term_to_object_pattern(term)?)
                } else {
                    None
                }
            }
            TermPattern::NamedNode(n) => Some(ObjectPattern::NamedNode(n.clone())),
            TermPattern::BlankNode(b) => Some(ObjectPattern::BlankNode(b.clone())),
            TermPattern::Literal(l) => Some(ObjectPattern::Literal(l.clone())),
        };

        Ok(TriplePattern::new(subject, predicate, object))
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
            _ => Err(OxirsError::Query("Invalid term for object pattern".to_string())),
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
        let pattern = TriplePattern::new(
            Some(SubjectPattern::NamedNode(NamedNode::new("http://example.org/s").unwrap())),
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
        
        let pattern = AlgebraTriplePattern {
            subject: TermPattern::Variable(Variable::new("s").unwrap()),
            predicate: TermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
            object: TermPattern::Literal(Literal::new("test")),
        };
        
        let selectivity = optimizer.estimate_selectivity(&pattern);
        
        // Literal object should give low selectivity
        assert!(selectivity < 0.2);
    }

    #[test]
    fn test_pattern_optimization() {
        let stats = Arc::new(IndexStats::new());
        let optimizer = PatternOptimizer::new(stats);
        
        let patterns = vec![
            AlgebraTriplePattern {
                subject: TermPattern::Variable(Variable::new("s").unwrap()),
                predicate: TermPattern::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
                object: TermPattern::NamedNode(NamedNode::new("http://example.org/Person").unwrap()),
            },
            AlgebraTriplePattern {
                subject: TermPattern::Variable(Variable::new("s").unwrap()),
                predicate: TermPattern::NamedNode(NamedNode::new("http://example.org/name").unwrap()),
                object: TermPattern::Variable(Variable::new("name").unwrap()),
            },
        ];
        
        let plan = optimizer.optimize_patterns(&patterns).unwrap();
        
        assert_eq!(plan.patterns.len(), 2);
        assert!(plan.total_cost > 0.0);
        assert_eq!(plan.binding_order.len(), 2);
    }
}