//! Advanced BGP (Basic Graph Pattern) Optimization
//!
//! This module provides index-aware BGP optimization with sophisticated
//! selectivity estimation for triple patterns.

use crate::algebra::{Algebra, Term, TriplePattern};
use crate::optimizer::{IndexStatistics, IndexType, Statistics};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// BGP optimization result
#[derive(Debug, Clone)]
pub struct OptimizedBGP {
    /// Reordered triple patterns
    pub patterns: Vec<TriplePattern>,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Selectivity information
    pub selectivity_info: SelectivityInfo,
    /// Recommended index usage
    pub index_plan: IndexUsagePlan,
}

/// Selectivity information for BGP
#[derive(Debug, Clone)]
pub struct SelectivityInfo {
    /// Pattern selectivities
    pub pattern_selectivity: Vec<PatternSelectivity>,
    /// Join selectivities
    pub join_selectivity: HashMap<(usize, usize), f64>,
    /// Overall BGP selectivity
    pub overall_selectivity: f64,
}

/// Selectivity information for a single pattern
#[derive(Debug, Clone)]
pub struct PatternSelectivity {
    /// Triple pattern
    pub pattern: TriplePattern,
    /// Estimated selectivity (0.0 to 1.0)
    pub selectivity: f64,
    /// Estimated cardinality
    pub cardinality: usize,
    /// Contributing factors
    pub factors: SelectivityFactors,
}

/// Factors contributing to selectivity
#[derive(Debug, Clone)]
pub struct SelectivityFactors {
    /// Subject selectivity
    pub subject_selectivity: f64,
    /// Predicate selectivity
    pub predicate_selectivity: f64,
    /// Object selectivity
    pub object_selectivity: f64,
    /// Index availability factor
    pub index_factor: f64,
    /// Data distribution factor
    pub distribution_factor: f64,
}

/// Index usage plan for BGP execution
#[derive(Debug, Clone)]
pub struct IndexUsagePlan {
    /// Index assignments per pattern
    pub pattern_indexes: Vec<IndexAssignment>,
    /// Join index opportunities
    pub join_indexes: Vec<JoinIndexOpportunity>,
}

/// Index assignment for a pattern
#[derive(Debug, Clone)]
pub struct IndexAssignment {
    /// Pattern index
    pub pattern_idx: usize,
    /// Recommended index type
    pub index_type: IndexType,
    /// Expected scan cost
    pub scan_cost: f64,
}

/// Join index opportunity
#[derive(Debug, Clone)]
pub struct JoinIndexOpportunity {
    /// Left pattern index
    pub left_pattern_idx: usize,
    /// Right pattern index
    pub right_pattern_idx: usize,
    /// Join variable
    pub join_var: String,
    /// Potential speedup factor
    pub speedup_factor: f64,
}

/// Advanced BGP optimizer
pub struct BGPOptimizer<'a> {
    statistics: &'a Statistics,
    index_stats: &'a IndexStatistics,
    adaptive_selector: AdaptiveIndexSelector,
}

/// Adaptive index selector for dynamic optimization
#[derive(Debug, Clone)]
pub struct AdaptiveIndexSelector {
    /// Query pattern frequency
    pattern_frequency: HashMap<String, usize>,
    /// Index effectiveness history
    index_effectiveness: HashMap<IndexType, f64>,
    /// Workload characteristics
    workload_characteristics: WorkloadCharacteristics,
}

/// Workload characteristics for adaptive optimization
#[derive(Debug, Clone, Default)]
pub struct WorkloadCharacteristics {
    /// Average query complexity (number of patterns)
    pub avg_query_complexity: f64,
    /// Predicate diversity
    pub predicate_diversity: f64,
    /// Join frequency patterns
    pub join_frequency: HashMap<String, usize>,
    /// Temporal query patterns
    pub temporal_access_patterns: HashMap<String, Vec<std::time::Instant>>,
}

impl<'a> BGPOptimizer<'a> {
    pub fn new(statistics: &'a Statistics, index_stats: &'a IndexStatistics) -> Self {
        Self {
            statistics,
            index_stats,
            adaptive_selector: AdaptiveIndexSelector::new(),
        }
    }

    /// Create optimizer with existing adaptive selector state
    pub fn with_adaptive_selector(
        statistics: &'a Statistics,
        index_stats: &'a IndexStatistics,
        adaptive_selector: AdaptiveIndexSelector,
    ) -> Self {
        Self {
            statistics,
            index_stats,
            adaptive_selector,
        }
    }

    /// Optimize a BGP with index awareness and selectivity estimation
    pub fn optimize_bgp(&self, patterns: Vec<TriplePattern>) -> Result<OptimizedBGP> {
        // Step 1: Calculate selectivity for each pattern
        let pattern_selectivities = self.calculate_pattern_selectivities(&patterns)?;

        // Step 2: Identify join variables and calculate join selectivities
        let join_selectivities =
            self.calculate_join_selectivities(&patterns, &pattern_selectivities)?;

        // Step 3: Determine optimal pattern ordering
        let ordered_patterns =
            self.order_patterns_optimally(&patterns, &pattern_selectivities, &join_selectivities)?;

        // Step 4: Create index usage plan
        let index_plan = self.create_index_usage_plan(&ordered_patterns, &pattern_selectivities)?;

        // Step 5: Calculate overall cost and selectivity
        let (overall_cost, overall_selectivity) = self.calculate_overall_metrics(
            &ordered_patterns,
            &pattern_selectivities,
            &join_selectivities,
            &index_plan,
        )?;

        Ok(OptimizedBGP {
            patterns: ordered_patterns,
            estimated_cost: overall_cost,
            selectivity_info: SelectivityInfo {
                pattern_selectivity: pattern_selectivities,
                join_selectivity: join_selectivities,
                overall_selectivity,
            },
            index_plan,
        })
    }

    /// Calculate selectivity for each pattern
    fn calculate_pattern_selectivities(
        &self,
        patterns: &[TriplePattern],
    ) -> Result<Vec<PatternSelectivity>> {
        patterns
            .iter()
            .map(|pattern| self.calculate_single_pattern_selectivity(pattern))
            .collect()
    }

    /// Calculate selectivity for a single pattern
    fn calculate_single_pattern_selectivity(
        &self,
        pattern: &TriplePattern,
    ) -> Result<PatternSelectivity> {
        // Calculate component selectivities
        let subject_sel =
            self.calculate_term_selectivity(&pattern.subject, TermPosition::Subject)?;
        let predicate_sel =
            self.calculate_term_selectivity(&pattern.predicate, TermPosition::Predicate)?;
        let object_sel = self.calculate_term_selectivity(&pattern.object, TermPosition::Object)?;

        // Calculate index factor
        let index_factor = self.calculate_index_factor(pattern)?;

        // Calculate data distribution factor
        let distribution_factor = self.calculate_distribution_factor(pattern)?;

        // Combine factors using advanced formula
        let combined_selectivity = self.combine_selectivity_factors(
            subject_sel,
            predicate_sel,
            object_sel,
            index_factor,
            distribution_factor,
        );

        // Estimate cardinality
        let total_triples = self
            .statistics
            .pattern_cardinality
            .values()
            .sum::<usize>()
            .max(1_000_000);
        let cardinality = (total_triples as f64 * combined_selectivity).ceil() as usize;

        Ok(PatternSelectivity {
            pattern: pattern.clone(),
            selectivity: combined_selectivity,
            cardinality,
            factors: SelectivityFactors {
                subject_selectivity: subject_sel,
                predicate_selectivity: predicate_sel,
                object_selectivity: object_sel,
                index_factor,
                distribution_factor,
            },
        })
    }

    /// Calculate selectivity for a term based on its position
    fn calculate_term_selectivity(&self, term: &Term, position: TermPosition) -> Result<f64> {
        match term {
            Term::Variable(_) => {
                // Variables have low selectivity
                match position {
                    TermPosition::Subject => Ok(0.1),
                    TermPosition::Predicate => Ok(0.01), // Variable predicates are rare and unselective
                    TermPosition::Object => Ok(0.2),
                }
            }
            Term::Iri(iri) => {
                // IRIs have selectivity based on statistics
                let cardinality = match position {
                    TermPosition::Subject => self.statistics.subject_cardinality.get(iri.as_str()),
                    TermPosition::Predicate => {
                        self.statistics.predicate_frequency.get(iri.as_str())
                    }
                    TermPosition::Object => self.statistics.object_cardinality.get(iri.as_str()),
                };

                if let Some(&card) = cardinality {
                    let total = self.get_total_count_for_position(position);
                    Ok((card as f64 / total as f64).min(1.0))
                } else {
                    // No statistics, use heuristic
                    Ok(match position {
                        TermPosition::Subject => 0.001,
                        TermPosition::Predicate => 0.01,
                        TermPosition::Object => 0.001,
                    })
                }
            }
            Term::Literal(_) => {
                // Literals are typically very selective
                Ok(0.0001)
            }
            Term::BlankNode(_) => {
                // Blank nodes have moderate selectivity
                Ok(0.01)
            }
        }
    }

    /// Calculate index availability factor
    fn calculate_index_factor(&self, pattern: &TriplePattern) -> Result<f64> {
        let mut best_index_factor = 1.0;

        // Check SPO index
        if self.is_bound(&pattern.subject)
            && self.is_bound(&pattern.predicate)
            && self.is_bound(&pattern.object)
        {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SubjectPredicate)
            {
                best_index_factor = 0.01; // Very efficient with full index
            }
        }
        // Check SP index
        else if self.is_bound(&pattern.subject) && self.is_bound(&pattern.predicate) {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SubjectPredicate)
            {
                best_index_factor = 0.1;
            }
        }
        // Check PO index
        else if self.is_bound(&pattern.predicate) && self.is_bound(&pattern.object) {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::PredicateObject)
            {
                best_index_factor = 0.15;
            }
        }
        // Check SO index
        else if self.is_bound(&pattern.subject) && self.is_bound(&pattern.object) {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SubjectObject)
            {
                best_index_factor = 0.2;
            }
        }
        // Single term indexes
        else if self.is_bound(&pattern.predicate) {
            best_index_factor = 0.5; // Predicate-only index is moderately efficient
        }

        Ok(best_index_factor)
    }

    /// Calculate data distribution factor
    fn calculate_distribution_factor(&self, pattern: &TriplePattern) -> Result<f64> {
        // Check for known patterns with skewed distributions
        if let Term::Iri(pred) = &pattern.predicate {
            // rdf:type often has skewed distribution
            if pred.as_str().ends_with("#type") || pred.as_str().ends_with("/type") {
                return Ok(0.8); // Less selective due to skew
            }
            // Labels and comments are usually unique
            if pred.as_str().ends_with("#label") || pred.as_str().ends_with("#comment") {
                return Ok(0.1); // More selective
            }
        }

        Ok(1.0) // Neutral factor
    }

    /// Combine selectivity factors using sophisticated formula
    fn combine_selectivity_factors(
        &self,
        subject_sel: f64,
        predicate_sel: f64,
        object_sel: f64,
        index_factor: f64,
        distribution_factor: f64,
    ) -> f64 {
        // Use independence assumption with correlation adjustment
        let base_selectivity = subject_sel * predicate_sel * object_sel;

        // Apply correlation factor (terms are not fully independent)
        let correlation_factor = 1.2; // Slight correlation penalty
        let adjusted_selectivity = (base_selectivity * correlation_factor).min(1.0);

        // Apply index and distribution factors
        (adjusted_selectivity * index_factor * distribution_factor).max(0.000001)
    }

    /// Calculate join selectivities between patterns
    fn calculate_join_selectivities(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
    ) -> Result<HashMap<(usize, usize), f64>> {
        let mut join_selectivities = HashMap::new();

        for i in 0..patterns.len() {
            for j in i + 1..patterns.len() {
                let join_vars = self.find_join_variables(&patterns[i], &patterns[j]);
                if !join_vars.is_empty() {
                    let selectivity = self.estimate_join_selectivity(
                        &patterns[i],
                        &patterns[j],
                        &join_vars,
                        &pattern_selectivities[i],
                        &pattern_selectivities[j],
                    )?;
                    join_selectivities.insert((i, j), selectivity);
                }
            }
        }

        Ok(join_selectivities)
    }

    /// Find join variables between two patterns
    fn find_join_variables(&self, p1: &TriplePattern, p2: &TriplePattern) -> Vec<String> {
        let vars1 = self.extract_variables(p1);
        let vars2 = self.extract_variables(p2);

        vars1.intersection(&vars2).cloned().collect()
    }

    /// Extract variables from a pattern
    fn extract_variables(&self, pattern: &TriplePattern) -> HashSet<String> {
        let mut vars = HashSet::new();

        if let Term::Variable(v) = &pattern.subject {
            vars.insert(v.to_string());
        }
        if let Term::Variable(v) = &pattern.predicate {
            vars.insert(v.to_string());
        }
        if let Term::Variable(v) = &pattern.object {
            vars.insert(v.to_string());
        }

        vars
    }

    /// Estimate join selectivity
    fn estimate_join_selectivity(
        &self,
        p1: &TriplePattern,
        p2: &TriplePattern,
        join_vars: &[String],
        sel1: &PatternSelectivity,
        sel2: &PatternSelectivity,
    ) -> Result<f64> {
        // Base selectivity from pattern selectivities
        let base_selectivity = (sel1.selectivity * sel2.selectivity).sqrt();

        // Adjust based on join variable position and type
        let mut position_factor = 1.0;
        for var in join_vars {
            if self.is_subject_variable(p1, var) && self.is_subject_variable(p2, var) {
                position_factor *= 0.1; // Subject-subject joins are selective
            } else if self.is_object_variable(p1, var) && self.is_object_variable(p2, var) {
                position_factor *= 0.2; // Object-object joins are moderately selective
            } else {
                position_factor *= 0.5; // Mixed position joins
            }
        }

        Ok((base_selectivity * position_factor).max(0.000001))
    }

    /// Order patterns optimally using dynamic programming approach
    fn order_patterns_optimally(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
        join_selectivities: &HashMap<(usize, usize), f64>,
    ) -> Result<Vec<TriplePattern>> {
        if patterns.len() <= 1 {
            return Ok(patterns.to_vec());
        }

        // Use greedy algorithm for now (can be replaced with DP for optimal solution)
        let mut remaining: HashSet<usize> = (0..patterns.len()).collect();
        let mut ordered = Vec::new();
        let mut ordered_indices = Vec::new();

        // Start with most selective pattern
        let start_idx = pattern_selectivities
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.selectivity.partial_cmp(&b.selectivity).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        ordered.push(patterns[start_idx].clone());
        ordered_indices.push(start_idx);
        remaining.remove(&start_idx);

        // Greedily add patterns with best join selectivity
        while !remaining.is_empty() {
            let mut best_next = None;
            let mut best_cost = f64::MAX;

            for &candidate in &remaining {
                let cost = self.calculate_join_cost(
                    candidate,
                    &ordered_indices,
                    pattern_selectivities,
                    join_selectivities,
                )?;

                if cost < best_cost {
                    best_cost = cost;
                    best_next = Some(candidate);
                }
            }

            if let Some(next_idx) = best_next {
                ordered.push(patterns[next_idx].clone());
                ordered_indices.push(next_idx);
                remaining.remove(&next_idx);
            } else {
                // No joins found, add remaining patterns by selectivity
                let mut remaining_vec: Vec<_> = remaining.into_iter().collect();
                remaining_vec.sort_by(|&a, &b| {
                    pattern_selectivities[a]
                        .selectivity
                        .partial_cmp(&pattern_selectivities[b].selectivity)
                        .unwrap()
                });

                for idx in remaining_vec {
                    ordered.push(patterns[idx].clone());
                }
                break;
            }
        }

        Ok(ordered)
    }

    /// Calculate cost of adding a pattern to current ordering
    fn calculate_join_cost(
        &self,
        candidate: usize,
        current_ordering: &[usize],
        pattern_selectivities: &[PatternSelectivity],
        join_selectivities: &HashMap<(usize, usize), f64>,
    ) -> Result<f64> {
        let mut min_cost = pattern_selectivities[candidate].selectivity;

        for &existing in current_ordering {
            let key = if existing < candidate {
                (existing, candidate)
            } else {
                (candidate, existing)
            };

            if let Some(&join_sel) = join_selectivities.get(&key) {
                min_cost = min_cost.min(join_sel);
            }
        }

        Ok(min_cost)
    }

    /// Create index usage plan
    fn create_index_usage_plan(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
    ) -> Result<IndexUsagePlan> {
        let mut pattern_indexes = Vec::new();
        let mut join_indexes = Vec::new();

        // Assign indexes to patterns
        for (idx, pattern) in patterns.iter().enumerate() {
            let best_index = self.select_best_index_for_pattern(pattern)?;
            let scan_cost = self.estimate_index_scan_cost(
                pattern,
                &best_index,
                pattern_selectivities[idx].cardinality,
            )?;

            pattern_indexes.push(IndexAssignment {
                pattern_idx: idx,
                index_type: best_index,
                scan_cost,
            });
        }

        // Identify join index opportunities
        for i in 0..patterns.len() {
            for j in i + 1..patterns.len() {
                let join_vars = self.find_join_variables(&patterns[i], &patterns[j]);
                if !join_vars.is_empty() {
                    for var in join_vars {
                        let speedup =
                            self.estimate_join_index_speedup(&patterns[i], &patterns[j], &var)?;
                        if speedup > 1.5 {
                            join_indexes.push(JoinIndexOpportunity {
                                left_pattern_idx: i,
                                right_pattern_idx: j,
                                join_var: var,
                                speedup_factor: speedup,
                            });
                        }
                    }
                }
            }
        }

        Ok(IndexUsagePlan {
            pattern_indexes,
            join_indexes,
        })
    }

    /// Select best index for a pattern
    fn select_best_index_for_pattern(&self, pattern: &TriplePattern) -> Result<IndexType> {
        let subject_bound = self.is_bound(&pattern.subject);
        let predicate_bound = self.is_bound(&pattern.predicate);
        let object_bound = self.is_bound(&pattern.object);

        // Priority order for index selection
        if subject_bound && predicate_bound {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SubjectPredicate)
            {
                return Ok(IndexType::SubjectPredicate);
            }
        }

        if predicate_bound && object_bound {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::PredicateObject)
            {
                return Ok(IndexType::PredicateObject);
            }
        }

        if subject_bound && object_bound {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SubjectObject)
            {
                return Ok(IndexType::SubjectObject);
            }
        }

        // Default to subject-predicate index
        Ok(IndexType::SubjectPredicate)
    }

    /// Estimate index scan cost
    fn estimate_index_scan_cost(
        &self,
        pattern: &TriplePattern,
        index_type: &IndexType,
        cardinality: usize,
    ) -> Result<f64> {
        let base_cost = cardinality as f64;

        let index_overhead = self
            .index_stats
            .index_access_cost
            .get(index_type)
            .copied()
            .unwrap_or(1.0);

        Ok(base_cost * index_overhead)
    }

    /// Estimate speedup from using join index
    fn estimate_join_index_speedup(
        &self,
        p1: &TriplePattern,
        p2: &TriplePattern,
        join_var: &str,
    ) -> Result<f64> {
        // Check if join variable is indexed in both patterns
        let p1_indexed = self.is_variable_indexed(p1, join_var);
        let p2_indexed = self.is_variable_indexed(p2, join_var);

        if p1_indexed && p2_indexed {
            Ok(10.0) // Significant speedup with indexed join
        } else if p1_indexed || p2_indexed {
            Ok(3.0) // Moderate speedup with one side indexed
        } else {
            Ok(1.0) // No speedup without indexes
        }
    }

    /// Calculate overall metrics
    fn calculate_overall_metrics(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
        join_selectivities: &HashMap<(usize, usize), f64>,
        index_plan: &IndexUsagePlan,
    ) -> Result<(f64, f64)> {
        let mut total_cost = 0.0;
        let mut cumulative_selectivity = 1.0;

        for (idx, assignment) in index_plan.pattern_indexes.iter().enumerate() {
            total_cost += assignment.scan_cost;
            cumulative_selectivity *= pattern_selectivities[idx].selectivity;

            // Apply join selectivity reductions
            if idx > 0 {
                for prev_idx in 0..idx {
                    let key = if prev_idx < idx {
                        (prev_idx, idx)
                    } else {
                        (idx, prev_idx)
                    };

                    if let Some(&join_sel) = join_selectivities.get(&key) {
                        cumulative_selectivity *= join_sel;
                    }
                }
            }
        }

        Ok((total_cost, cumulative_selectivity))
    }

    // Helper methods

    fn is_bound(&self, term: &Term) -> bool {
        !matches!(term, Term::Variable(_))
    }

    fn is_subject_variable(&self, pattern: &TriplePattern, var: &str) -> bool {
        matches!(&pattern.subject, Term::Variable(v) if v.to_string() == var)
    }

    fn is_object_variable(&self, pattern: &TriplePattern, var: &str) -> bool {
        matches!(&pattern.object, Term::Variable(v) if v.to_string() == var)
    }

    fn is_variable_indexed(&self, pattern: &TriplePattern, var: &str) -> bool {
        // Check if the variable position would benefit from available indexes
        if self.is_subject_variable(pattern, var) {
            self.index_stats
                .available_indexes
                .contains(&IndexType::SubjectPredicate)
                || self
                    .index_stats
                    .available_indexes
                    .contains(&IndexType::SubjectObject)
        } else if self.is_object_variable(pattern, var) {
            self.index_stats
                .available_indexes
                .contains(&IndexType::PredicateObject)
                || self
                    .index_stats
                    .available_indexes
                    .contains(&IndexType::SubjectObject)
        } else {
            false
        }
    }

    fn get_total_count_for_position(&self, position: TermPosition) -> usize {
        match position {
            TermPosition::Subject => self
                .statistics
                .subject_cardinality
                .values()
                .sum::<usize>()
                .max(100_000),
            TermPosition::Predicate => self
                .statistics
                .predicate_frequency
                .values()
                .sum::<usize>()
                .max(1_000),
            TermPosition::Object => self
                .statistics
                .object_cardinality
                .values()
                .sum::<usize>()
                .max(100_000),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum TermPosition {
    Subject,
    Predicate,
    Object,
}

impl AdaptiveIndexSelector {
    pub fn new() -> Self {
        Self {
            pattern_frequency: HashMap::new(),
            index_effectiveness: HashMap::new(),
            workload_characteristics: WorkloadCharacteristics::default(),
        }
    }

    /// Update selector with query execution feedback
    pub fn update_from_execution(
        &mut self,
        patterns: &[TriplePattern],
        index_used: IndexType,
        execution_time: std::time::Duration,
        result_count: usize,
    ) {
        // Update pattern frequency
        for pattern in patterns {
            let pattern_key = self.pattern_key(pattern);
            *self.pattern_frequency.entry(pattern_key).or_insert(0) += 1;
        }

        // Update index effectiveness
        let effectiveness = self.calculate_index_effectiveness(execution_time, result_count);
        self.index_effectiveness
            .entry(index_used)
            .and_modify(|e| *e = (*e + effectiveness) / 2.0)
            .or_insert(effectiveness);

        // Update workload characteristics
        self.update_workload_characteristics(patterns);
    }

    /// Calculate pattern key for frequency tracking
    fn pattern_key(&self, pattern: &TriplePattern) -> String {
        format!(
            "{:?}_{:?}_{:?}",
            self.term_type(&pattern.subject),
            self.term_type(&pattern.predicate),
            self.term_type(&pattern.object)
        )
    }

    /// Get term type for pattern analysis
    fn term_type(&self, term: &Term) -> &str {
        match term {
            Term::Variable(_) => "VAR",
            Term::Iri(_) => "IRI",
            Term::Literal(_) => "LIT",
            Term::BlankNode(_) => "BN",
        }
    }

    /// Calculate index effectiveness from execution results
    fn calculate_index_effectiveness(
        &self,
        execution_time: std::time::Duration,
        result_count: usize,
    ) -> f64 {
        let time_factor = 1.0 / (execution_time.as_millis() as f64 + 1.0);
        let result_factor = if result_count == 0 {
            0.1
        } else {
            (result_count as f64).ln()
        };
        time_factor * result_factor
    }

    /// Update workload characteristics
    fn update_workload_characteristics(&mut self, patterns: &[TriplePattern]) {
        let new_complexity = patterns.len() as f64;
        self.workload_characteristics.avg_query_complexity =
            (self.workload_characteristics.avg_query_complexity + new_complexity) / 2.0;
    }
}

impl Default for AdaptiveIndexSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Iri, Literal};

    fn create_test_statistics() -> Statistics {
        let mut stats = Statistics::new();

        // Add predicate frequencies
        stats.predicate_frequency.insert(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            10000,
        );
        stats
            .predicate_frequency
            .insert("http://xmlns.com/foaf/0.1/name".to_string(), 5000);
        stats
            .predicate_frequency
            .insert("http://xmlns.com/foaf/0.1/knows".to_string(), 2000);

        // Add subject cardinalities
        stats
            .subject_cardinality
            .insert("http://example.org/person/1".to_string(), 10);
        stats
            .subject_cardinality
            .insert("http://example.org/person/2".to_string(), 15);

        // Add available indexes
        stats
            .index_stats
            .available_indexes
            .insert(IndexType::SubjectPredicate);
        stats
            .index_stats
            .available_indexes
            .insert(IndexType::PredicateObject);

        stats
    }

    #[test]
    fn test_pattern_selectivity_calculation() {
        let stats = create_test_statistics();
        let optimizer = BGPOptimizer::new(&stats, &stats.index_stats);

        // Test pattern with bound predicate
        let pattern = TriplePattern {
            subject: Term::Variable("s".to_string()),
            predicate: Term::Iri(Iri(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()
            )),
            object: Term::Variable("o".to_string()),
        };

        let selectivity = optimizer
            .calculate_single_pattern_selectivity(&pattern)
            .unwrap();

        // Should have relatively low selectivity due to common predicate
        assert!(selectivity.selectivity < 0.1);
        assert!(selectivity.factors.predicate_selectivity > 0.0);
    }

    #[test]
    fn test_join_variable_detection() {
        let stats = create_test_statistics();
        let optimizer = BGPOptimizer::new(&stats, &stats.index_stats);

        let p1 = TriplePattern {
            subject: Term::Variable("x".to_string()),
            predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            object: Term::Variable("name".to_string()),
        };

        let p2 = TriplePattern {
            subject: Term::Variable("x".to_string()),
            predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
            object: Term::Variable("y".to_string()),
        };

        let join_vars = optimizer.find_join_variables(&p1, &p2);
        assert_eq!(join_vars.len(), 1);
        assert!(join_vars.contains(&"x".to_string()));
    }

    #[test]
    fn test_bgp_optimization() {
        let stats = create_test_statistics();
        let optimizer = BGPOptimizer::new(&stats, &stats.index_stats);

        let patterns = vec![
            // Less selective pattern
            TriplePattern {
                subject: Term::Variable("s".to_string()),
                predicate: Term::Iri(Iri(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()
                )),
                object: Term::Variable("o".to_string()),
            },
            // More selective pattern
            TriplePattern {
                subject: Term::Variable("s".to_string()),
                predicate: Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
                object: Term::Literal(Literal {
                    value: "John Doe".to_string(),
                    language: None,
                    datatype: None,
                }),
            },
        ];

        let optimized = optimizer.optimize_bgp(patterns.clone()).unwrap();

        // The more selective pattern should be ordered first
        assert_eq!(optimized.patterns.len(), 2);
        assert!(matches!(&optimized.patterns[0].object, Term::Literal(_)));
    }
}
