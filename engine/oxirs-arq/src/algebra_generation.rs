//! Algebra Generation Module
//!
//! Provides bottom-up algebra construction, join ordering heuristics,
//! and query plan generation from parsed SPARQL queries.

use crate::algebra::{Algebra, Expression, TriplePattern, Variable};
use crate::query_analysis::{QueryAnalysis, QueryAnalyzer};
use crate::statistics_collector::StatisticsCollector;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// Algebra generation configuration
#[derive(Debug, Clone)]
pub struct AlgebraGenerationConfig {
    /// Enable join reordering during construction
    pub enable_join_reordering: bool,
    /// Enable filter pushdown during construction
    pub enable_filter_pushdown: bool,
    /// Enable projection pushdown during construction
    pub enable_projection_pushdown: bool,
    /// Maximum number of join reordering attempts
    pub max_join_reorder_attempts: usize,
    /// Cost threshold for considering reordering
    pub reorder_cost_threshold: f64,
}

impl Default for AlgebraGenerationConfig {
    fn default() -> Self {
        Self {
            enable_join_reordering: true,
            enable_filter_pushdown: true,
            enable_projection_pushdown: true,
            max_join_reorder_attempts: 1000,
            reorder_cost_threshold: 0.1,
        }
    }
}

/// Join ordering strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinOrderingStrategy {
    /// Left-deep joins (linear chain)
    LeftDeep,
    /// Right-deep joins
    RightDeep,
    /// Bushy joins (balanced tree)
    Bushy,
    /// Adaptive based on cardinality estimates
    Adaptive,
    /// Greedy ordering based on selectivity
    Greedy,
    /// Dynamic programming optimal ordering
    DynamicProgramming,
}

/// Join cost estimation
#[derive(Debug, Clone)]
pub struct JoinCost {
    /// Estimated cardinality of the result
    pub cardinality: usize,
    /// Estimated CPU cost
    pub cpu_cost: f64,
    /// Estimated I/O cost
    pub io_cost: f64,
    /// Estimated memory usage
    pub memory_cost: f64,
    /// Total cost (weighted combination)
    pub total_cost: f64,
}

impl JoinCost {
    pub fn new(cardinality: usize, cpu_cost: f64, io_cost: f64, memory_cost: f64) -> Self {
        let total_cost = cpu_cost + io_cost + (memory_cost * 0.1); // Weight memory lower
        Self {
            cardinality,
            cpu_cost,
            io_cost,
            memory_cost,
            total_cost,
        }
    }

    pub fn infinite() -> Self {
        Self {
            cardinality: usize::MAX,
            cpu_cost: f64::INFINITY,
            io_cost: f64::INFINITY,
            memory_cost: f64::INFINITY,
            total_cost: f64::INFINITY,
        }
    }
}

/// Join order candidate
#[derive(Debug, Clone)]
pub struct JoinCandidate {
    /// The algebra tree for this join order
    pub algebra: Algebra,
    /// Estimated cost of this join order
    pub cost: JoinCost,
    /// Variables available after this join
    pub available_variables: HashSet<Variable>,
}

/// Algebra generator
pub struct AlgebraGenerator {
    config: AlgebraGenerationConfig,
    analyzer: QueryAnalyzer,
    #[allow(dead_code)]
    statistics: Option<StatisticsCollector>,
}

impl AlgebraGenerator {
    /// Create a new algebra generator
    pub fn new(config: AlgebraGenerationConfig) -> Self {
        Self {
            config,
            analyzer: QueryAnalyzer::new(),
            statistics: None,
        }
    }

    /// Create generator with statistics collector
    pub fn with_statistics(
        config: AlgebraGenerationConfig,
        statistics: StatisticsCollector,
    ) -> Self {
        Self {
            config,
            analyzer: QueryAnalyzer::new(),
            statistics: Some(statistics),
        }
    }

    /// Generate algebra from a basic graph pattern
    pub fn generate_from_bgp(
        &self,
        patterns: Vec<TriplePattern>,
        filters: Vec<Expression>,
        projection: Option<Vec<Variable>>,
    ) -> Result<Algebra> {
        if patterns.is_empty() {
            return Err(anyhow!("Cannot generate algebra from empty BGP"));
        }

        // Step 1: Create initial algebra from patterns
        let mut algebra = if patterns.len() == 1 {
            Algebra::Bgp(vec![patterns[0].clone()])
        } else {
            Algebra::Bgp(patterns.clone())
        };

        // Step 2: Apply join reordering if enabled and beneficial
        if self.config.enable_join_reordering && patterns.len() > 1 {
            algebra = self.reorder_joins(algebra, &patterns)?;
        }

        // Step 3: Apply filters with pushdown if enabled
        if self.config.enable_filter_pushdown {
            algebra = self.apply_filters_with_pushdown(algebra, filters)?;
        } else {
            algebra = self.apply_filters(algebra, filters)?;
        }

        // Step 4: Apply projection with pushdown if enabled
        if let Some(vars) = projection {
            if self.config.enable_projection_pushdown {
                algebra = self.apply_projection_with_pushdown(algebra, vars)?;
            } else {
                algebra = Algebra::Project {
                    variables: vars,
                    pattern: Box::new(algebra),
                };
            }
        }

        Ok(algebra)
    }

    /// Generate optimized join order for multiple patterns
    pub fn generate_join_order(
        &self,
        patterns: Vec<TriplePattern>,
        strategy: JoinOrderingStrategy,
    ) -> Result<Algebra> {
        match strategy {
            JoinOrderingStrategy::LeftDeep => self.generate_left_deep_joins(patterns),
            JoinOrderingStrategy::RightDeep => self.generate_right_deep_joins(patterns),
            JoinOrderingStrategy::Bushy => self.generate_bushy_joins(patterns),
            JoinOrderingStrategy::Adaptive => self.generate_adaptive_joins(patterns),
            JoinOrderingStrategy::Greedy => self.generate_greedy_joins(patterns),
            JoinOrderingStrategy::DynamicProgramming => self.generate_dp_joins(patterns),
        }
    }

    /// Generate left-deep join tree
    fn generate_left_deep_joins(&self, patterns: Vec<TriplePattern>) -> Result<Algebra> {
        if patterns.is_empty() {
            return Err(anyhow!("No patterns to join"));
        }

        let mut result = Algebra::Bgp(vec![patterns[0].clone()]);

        for pattern in patterns.iter().skip(1) {
            result = Algebra::Join {
                left: Box::new(result),
                right: Box::new(Algebra::Bgp(vec![pattern.clone()])),
            };
        }

        Ok(result)
    }

    /// Generate right-deep join tree
    fn generate_right_deep_joins(&self, patterns: Vec<TriplePattern>) -> Result<Algebra> {
        if patterns.is_empty() {
            return Err(anyhow!("No patterns to join"));
        }

        let mut result = Algebra::Bgp(vec![patterns[patterns.len() - 1].clone()]);

        for pattern in patterns.iter().rev().skip(1) {
            result = Algebra::Join {
                left: Box::new(Algebra::Bgp(vec![pattern.clone()])),
                right: Box::new(result),
            };
        }

        Ok(result)
    }

    /// Generate bushy join tree (balanced)
    #[allow(clippy::only_used_in_recursion)]
    fn generate_bushy_joins(&self, patterns: Vec<TriplePattern>) -> Result<Algebra> {
        if patterns.is_empty() {
            return Err(anyhow!("No patterns to join"));
        }

        if patterns.len() == 1 {
            return Ok(Algebra::Bgp(vec![patterns[0].clone()]));
        }

        if patterns.len() == 2 {
            return Ok(Algebra::Join {
                left: Box::new(Algebra::Bgp(vec![patterns[0].clone()])),
                right: Box::new(Algebra::Bgp(vec![patterns[1].clone()])),
            });
        }

        // Split patterns and recursively build bushy tree
        let mid = patterns.len() / 2;
        let (left_patterns, right_patterns) = patterns.split_at(mid);

        let left_algebra = self.generate_bushy_joins(left_patterns.to_vec())?;
        let right_algebra = self.generate_bushy_joins(right_patterns.to_vec())?;

        Ok(Algebra::Join {
            left: Box::new(left_algebra),
            right: Box::new(right_algebra),
        })
    }

    /// Generate adaptive join order based on pattern characteristics
    fn generate_adaptive_joins(&self, patterns: Vec<TriplePattern>) -> Result<Algebra> {
        // Analyze patterns to choose the best strategy
        let join_variables = self.count_join_variables(&patterns);
        let pattern_selectivity = self.estimate_pattern_selectivity(&patterns);

        // Choose strategy based on characteristics
        let strategy = if patterns.len() <= 3 {
            JoinOrderingStrategy::LeftDeep
        } else if join_variables > patterns.len() / 2 {
            JoinOrderingStrategy::Bushy
        } else if pattern_selectivity.iter().fold(0.0, |acc, &s| acc + s)
            < patterns.len() as f64 * 0.1
        {
            JoinOrderingStrategy::Greedy
        } else {
            JoinOrderingStrategy::DynamicProgramming
        };

        self.generate_join_order(patterns, strategy)
    }

    /// Generate greedy join order based on selectivity
    fn generate_greedy_joins(&self, patterns: Vec<TriplePattern>) -> Result<Algebra> {
        if patterns.is_empty() {
            return Err(anyhow!("No patterns to join"));
        }

        let remaining_patterns = patterns;
        let mut candidates: Vec<JoinCandidate> = Vec::new();

        // Create initial candidates from individual patterns
        for pattern in &remaining_patterns {
            let algebra = Algebra::Bgp(vec![pattern.clone()]);
            let cost = self.estimate_cost(&algebra)?;
            let variables = self.extract_variables(&algebra);

            candidates.push(JoinCandidate {
                algebra,
                cost,
                available_variables: variables,
            });
        }

        // Greedily build join tree
        while candidates.len() > 1 {
            let mut best_join: Option<(usize, usize, JoinCandidate)> = None;
            let mut best_cost = JoinCost::infinite();

            // Find the best pair to join
            for i in 0..candidates.len() {
                for j in (i + 1)..candidates.len() {
                    if let Ok(join_candidate) =
                        self.create_join_candidate(&candidates[i], &candidates[j])
                    {
                        if join_candidate.cost.total_cost < best_cost.total_cost {
                            best_cost = join_candidate.cost.clone();
                            best_join = Some((i, j, join_candidate));
                        }
                    }
                }
            }

            // Apply the best join
            if let Some((i, j, join_candidate)) = best_join {
                // Remove the joined patterns (remove larger index first)
                let (larger, smaller) = if i > j { (i, j) } else { (j, i) };
                candidates.remove(larger);
                candidates.remove(smaller);
                candidates.push(join_candidate);
            } else {
                // No valid joins found, fall back to left-deep
                return self.generate_left_deep_joins(remaining_patterns);
            }
        }

        Ok(candidates
            .into_iter()
            .next()
            .expect("candidates validated to be non-empty")
            .algebra)
    }

    /// Generate optimal join order using dynamic programming
    fn generate_dp_joins(&self, patterns: Vec<TriplePattern>) -> Result<Algebra> {
        if patterns.is_empty() {
            return Err(anyhow!("No patterns to join"));
        }

        if patterns.len() == 1 {
            return Ok(Algebra::Bgp(vec![patterns[0].clone()]));
        }

        // Limit DP to reasonable size to avoid exponential explosion
        if patterns.len() > 10 {
            return self.generate_greedy_joins(patterns);
        }

        let n = patterns.len();
        let mut dp: HashMap<u64, JoinCandidate> = HashMap::new();

        // Initialize base cases (single patterns)
        for (i, pattern) in patterns.iter().enumerate() {
            let mask = 1u64 << i;
            let algebra = Algebra::Bgp(vec![pattern.clone()]);
            let cost = self.estimate_cost(&algebra)?;
            let variables = self.extract_variables(&algebra);

            dp.insert(
                mask,
                JoinCandidate {
                    algebra,
                    cost,
                    available_variables: variables,
                },
            );
        }

        // Fill DP table for all subset combinations
        for size in 2..=n {
            for mask in 1u64..(1u64 << n) {
                if mask.count_ones() != size as u32 {
                    continue;
                }

                let mut best_candidate: Option<JoinCandidate> = None;
                let mut best_cost = f64::INFINITY;

                // Try all ways to split this subset
                let mut submask = mask;
                while submask > 0 {
                    let complement = mask ^ submask;
                    if complement > 0 && submask < complement {
                        if let (Some(left), Some(right)) = (dp.get(&submask), dp.get(&complement)) {
                            if let Ok(candidate) = self.create_join_candidate(left, right) {
                                if candidate.cost.total_cost < best_cost {
                                    best_cost = candidate.cost.total_cost;
                                    best_candidate = Some(candidate);
                                }
                            }
                        }
                    }
                    submask = (submask - 1) & mask;
                }

                if let Some(candidate) = best_candidate {
                    dp.insert(mask, candidate);
                }
            }
        }

        // Return the optimal solution for all patterns
        let full_mask = (1u64 << n) - 1;
        dp.get(&full_mask)
            .map(|candidate| candidate.algebra.clone())
            .ok_or_else(|| anyhow!("Failed to generate optimal join order"))
    }

    /// Create a join candidate from two existing candidates
    fn create_join_candidate(
        &self,
        left: &JoinCandidate,
        right: &JoinCandidate,
    ) -> Result<JoinCandidate> {
        // Check if the join makes sense (shares variables or is a cartesian product)
        let shared_vars: HashSet<_> = left
            .available_variables
            .intersection(&right.available_variables)
            .cloned()
            .collect();

        let algebra = Algebra::Join {
            left: Box::new(left.algebra.clone()),
            right: Box::new(right.algebra.clone()),
        };

        let cost = self.estimate_join_cost(&left.cost, &right.cost, shared_vars.len())?;

        let mut available_variables = left.available_variables.clone();
        available_variables.extend(right.available_variables.iter().cloned());

        Ok(JoinCandidate {
            algebra,
            cost,
            available_variables,
        })
    }

    /// Reorder joins in existing algebra
    fn reorder_joins(&self, _algebra: Algebra, patterns: &[TriplePattern]) -> Result<Algebra> {
        // For now, use greedy reordering
        self.generate_greedy_joins(patterns.to_vec())
    }

    /// Apply filters with intelligent pushdown
    fn apply_filters_with_pushdown(
        &self,
        mut algebra: Algebra,
        filters: Vec<Expression>,
    ) -> Result<Algebra> {
        // Analyze each filter to determine optimal placement
        for filter in filters {
            let analysis = self.analyzer.analyze(&algebra)?;

            // Try to push filter down as far as possible
            algebra = self.push_filter_down(algebra, filter, &analysis)?;
        }

        Ok(algebra)
    }

    /// Apply filters without pushdown
    fn apply_filters(&self, mut algebra: Algebra, filters: Vec<Expression>) -> Result<Algebra> {
        for filter in filters {
            algebra = Algebra::Filter {
                condition: filter,
                pattern: Box::new(algebra),
            };
        }
        Ok(algebra)
    }

    /// Push a filter down to the optimal position
    #[allow(clippy::only_used_in_recursion)]
    fn push_filter_down(
        &self,
        algebra: Algebra,
        filter: Expression,
        analysis: &QueryAnalysis,
    ) -> Result<Algebra> {
        // Check if filter can be safely pushed down
        let mut filter_vars = HashSet::new();
        self.collect_filter_variables(&filter, &mut filter_vars);

        // Simple pushdown: if filter only uses variables from one side of a join, push it down
        match algebra {
            Algebra::Join { left, right } => {
                let left_vars = self.extract_variables(&left);
                let right_vars = self.extract_variables(&right);

                if filter_vars.is_subset(&left_vars) {
                    // Push to left side
                    let new_left = self.push_filter_down(*left, filter, analysis)?;
                    Ok(Algebra::Join {
                        left: Box::new(new_left),
                        right,
                    })
                } else if filter_vars.is_subset(&right_vars) {
                    // Push to right side
                    let new_right = self.push_filter_down(*right, filter, analysis)?;
                    Ok(Algebra::Join {
                        left,
                        right: Box::new(new_right),
                    })
                } else {
                    // Filter uses variables from both sides, keep at join level
                    Ok(Algebra::Filter {
                        condition: filter,
                        pattern: Box::new(Algebra::Join { left, right }),
                    })
                }
            }
            _ => {
                // For other algebra types, add filter at current level
                Ok(Algebra::Filter {
                    condition: filter,
                    pattern: Box::new(algebra),
                })
            }
        }
    }

    /// Apply projection with pushdown optimization
    fn apply_projection_with_pushdown(
        &self,
        algebra: Algebra,
        variables: Vec<Variable>,
    ) -> Result<Algebra> {
        // For now, simple implementation - just add projection at top level
        // A full implementation would push projections down and eliminate unused variables
        Ok(Algebra::Project {
            variables,
            pattern: Box::new(algebra),
        })
    }

    /// Estimate cost of an algebra expression
    fn estimate_cost(&self, algebra: &Algebra) -> Result<JoinCost> {
        match algebra {
            Algebra::Bgp(patterns) if patterns.len() == 1 => {
                // Base cost for a triple pattern
                Ok(JoinCost::new(1000, 1.0, 1.0, 0.1))
            }
            Algebra::Bgp(patterns) => {
                // Cost based on number of patterns
                let base_cost = patterns.len() as f64;
                Ok(JoinCost::new(
                    1000 * patterns.len(),
                    base_cost,
                    base_cost,
                    base_cost * 0.1,
                ))
            }
            Algebra::Join { left, right } => {
                let left_cost = self.estimate_cost(left)?;
                let right_cost = self.estimate_cost(right)?;
                self.estimate_join_cost(&left_cost, &right_cost, 1) // Assume 1 shared variable
            }
            _ => {
                // Default cost
                Ok(JoinCost::new(10000, 10.0, 10.0, 1.0))
            }
        }
    }

    /// Estimate cost of joining two relations
    fn estimate_join_cost(
        &self,
        left: &JoinCost,
        right: &JoinCost,
        shared_variables: usize,
    ) -> Result<JoinCost> {
        let selectivity = if shared_variables > 0 {
            0.1_f64.powi(shared_variables as i32)
        } else {
            1.0 // Cartesian product
        };

        let result_cardinality =
            ((left.cardinality as f64 * right.cardinality as f64 * selectivity) as usize).max(1);

        let cpu_cost =
            left.cpu_cost + right.cpu_cost + (left.cardinality + right.cardinality) as f64 * 0.001;
        let io_cost = left.io_cost + right.io_cost;
        let memory_cost = (left.cardinality + right.cardinality) as f64 * 0.0001;

        Ok(JoinCost::new(
            result_cardinality,
            cpu_cost,
            io_cost,
            memory_cost,
        ))
    }

    /// Count join variables between patterns
    fn count_join_variables(&self, patterns: &[TriplePattern]) -> usize {
        let mut all_vars: HashSet<Variable> = HashSet::new();
        let mut join_vars: HashSet<Variable> = HashSet::new();

        for pattern in patterns {
            let pattern_vars = self.extract_pattern_variables(pattern);
            for var in &pattern_vars {
                if all_vars.contains(var) {
                    join_vars.insert(var.clone());
                } else {
                    all_vars.insert(var.clone());
                }
            }
        }

        join_vars.len()
    }

    /// Estimate selectivity of patterns
    fn estimate_pattern_selectivity(&self, patterns: &[TriplePattern]) -> Vec<f64> {
        patterns
            .iter()
            .map(|pattern| {
                // Simple heuristic: more variables = higher selectivity
                let var_count = self.extract_pattern_variables(pattern).len();
                match var_count {
                    0 => 0.001, // All constants
                    1 => 0.01,  // One variable
                    2 => 0.1,   // Two variables
                    _ => 0.5,   // Three variables
                }
            })
            .collect()
    }

    /// Extract variables from algebra
    fn extract_variables(&self, algebra: &Algebra) -> HashSet<Variable> {
        let mut variables = HashSet::new();
        self.extract_variables_recursive(algebra, &mut variables);
        variables
    }

    fn extract_variables_recursive(&self, algebra: &Algebra, variables: &mut HashSet<Variable>) {
        match algebra {
            Algebra::Bgp(patterns) if patterns.len() == 1 => {
                variables.extend(self.extract_pattern_variables(&patterns[0]));
            }
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    variables.extend(self.extract_pattern_variables(pattern));
                }
            }
            Algebra::Join { left, right } => {
                self.extract_variables_recursive(left, variables);
                self.extract_variables_recursive(right, variables);
            }
            _ => {} // Handle other cases as needed
        }
    }

    /// Extract variables from a triple pattern
    fn extract_pattern_variables(&self, pattern: &TriplePattern) -> HashSet<Variable> {
        let mut variables = HashSet::new();

        if let crate::algebra::Term::Variable(var) = &pattern.subject {
            variables.insert(var.clone());
        }
        if let crate::algebra::Term::Variable(var) = &pattern.predicate {
            variables.insert(var.clone());
        }
        if let crate::algebra::Term::Variable(var) = &pattern.object {
            variables.insert(var.clone());
        }

        variables
    }

    /// Collect variables used in a filter expression
    #[allow(clippy::only_used_in_recursion)]
    fn collect_filter_variables(&self, expression: &Expression, variables: &mut HashSet<Variable>) {
        match expression {
            Expression::Variable(var) => {
                variables.insert(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                self.collect_filter_variables(left, variables);
                self.collect_filter_variables(right, variables);
            }
            Expression::Unary { operand, .. } => {
                self.collect_filter_variables(operand, variables);
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.collect_filter_variables(arg, variables);
                }
            }
            Expression::Conditional {
                condition,
                then_expr: if_true,
                else_expr: if_false,
            } => {
                self.collect_filter_variables(condition, variables);
                self.collect_filter_variables(if_true, variables);
                self.collect_filter_variables(if_false, variables);
            }
            Expression::Bound(var) => {
                variables.insert(var.clone());
            }
            _ => {} // Literals, constants
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Term;
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_left_deep_join_generation() {
        let generator = AlgebraGenerator::new(AlgebraGenerationConfig::default());

        let patterns = vec![
            TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p1")),
                object: Term::Variable(Variable::new("o1").unwrap()),
            },
            TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p2")),
                object: Term::Variable(Variable::new("o2").unwrap()),
            },
        ];

        let algebra = generator.generate_left_deep_joins(patterns).unwrap();

        // Should create a join with the first pattern on the left
        if let Algebra::Join { left, right } = algebra {
            assert!(matches!(*left.as_ref(), Algebra::Bgp(ref patterns) if patterns.len() == 1));
            assert!(matches!(*right.as_ref(), Algebra::Bgp(ref patterns) if patterns.len() == 1));
        } else {
            panic!("Expected join algebra");
        }
    }

    #[test]
    fn test_bushy_join_generation() {
        let generator = AlgebraGenerator::new(AlgebraGenerationConfig::default());

        let patterns = vec![
            TriplePattern {
                subject: Term::Variable(Variable::new("s1").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p1")),
                object: Term::Variable(Variable::new("o1").unwrap()),
            },
            TriplePattern {
                subject: Term::Variable(Variable::new("s2").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p2")),
                object: Term::Variable(Variable::new("o2").unwrap()),
            },
            TriplePattern {
                subject: Term::Variable(Variable::new("s3").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p3")),
                object: Term::Variable(Variable::new("o3").unwrap()),
            },
            TriplePattern {
                subject: Term::Variable(Variable::new("s4").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/p4")),
                object: Term::Variable(Variable::new("o4").unwrap()),
            },
        ];

        let algebra = generator.generate_bushy_joins(patterns).unwrap();

        // Should create a balanced tree structure
        if let Algebra::Join { left, right } = algebra {
            // Both sides should be joins or patterns
            assert!(
                matches!(*left.as_ref(), Algebra::Join { .. })
                    || matches!(*left.as_ref(), Algebra::Bgp(ref patterns) if !patterns.is_empty())
            );
            assert!(
                matches!(*right.as_ref(), Algebra::Join { .. })
                    || matches!(*right.as_ref(), Algebra::Bgp(ref patterns) if !patterns.is_empty())
            );
        } else {
            panic!("Expected join algebra");
        }
    }

    #[test]
    fn test_variable_extraction() {
        let generator = AlgebraGenerator::new(AlgebraGenerationConfig::default());

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/predicate")),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let variables = generator.extract_pattern_variables(&pattern);

        assert_eq!(variables.len(), 2);
        assert!(variables.contains(&Variable::new("s").unwrap()));
        assert!(variables.contains(&Variable::new("o").unwrap()));
    }

    #[test]
    fn test_cost_estimation() {
        let generator = AlgebraGenerator::new(AlgebraGenerationConfig::default());

        let pattern = TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://example.org/predicate")),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let algebra = Algebra::Bgp(vec![pattern]);
        let cost = generator.estimate_cost(&algebra).unwrap();

        assert!(cost.total_cost > 0.0);
        assert!(cost.cardinality > 0);
    }
}
