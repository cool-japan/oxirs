//! Advanced BGP (Basic Graph Pattern) Optimization
//!
//! This module provides index-aware BGP optimization with sophisticated
//! selectivity estimation for triple patterns.

use crate::algebra::{Algebra, Term, TriplePattern, Variable};
use crate::optimizer::{IndexPosition, IndexStatistics, IndexType, Statistics};
use anyhow::Result;
use oxirs_core::model::NamedNode;
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
    /// Multi-index intersection opportunities
    pub index_intersections: Vec<IndexIntersection>,
    /// Bloom filter recommendations
    pub bloom_filter_candidates: Vec<BloomFilterCandidate>,
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

/// Multi-index intersection for complex queries
#[derive(Debug, Clone)]
pub struct IndexIntersection {
    /// Pattern index for intersection
    pub pattern_idx: usize,
    /// Primary index type
    pub primary_index: IndexType,
    /// Secondary indexes for intersection
    pub secondary_indexes: Vec<IndexType>,
    /// Expected benefit from intersection
    pub selectivity_improvement: f64,
    /// Intersection algorithm to use
    pub intersection_algorithm: IntersectionAlgorithm,
}

/// Types of index intersections
#[derive(Debug, Clone)]
pub enum IntersectionType {
    /// Variable-based join intersection
    VariableJoin,
    /// Value-based intersection
    ValueIntersection,
    /// Spatial intersection
    SpatialIntersection,
    /// Temporal intersection
    TemporalIntersection,
}

/// Intersection algorithm types
#[derive(Debug, Clone)]
pub enum IntersectionAlgorithm {
    /// Bitmap intersection for dense results
    Bitmap,
    /// Hash-based intersection for sparse results
    Hash,
    /// Skip-list intersection for ordered indexes
    SkipList,
}

/// Bloom filter candidate for negative lookups
#[derive(Debug, Clone)]
pub struct BloomFilterCandidate {
    /// Pattern index
    pub pattern_idx: usize,
    /// Filter position (Subject, Predicate, Object)
    pub filter_position: TermPosition,
    /// Expected false positive rate
    pub false_positive_rate: f64,
    /// Memory footprint estimate (bytes)
    pub memory_footprint: usize,
    /// Expected performance gain
    pub performance_gain: f64,
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
            Term::QuotedTriple(_) => {
                // Quoted triples are rare and specific
                Ok(0.0001)
            }
            Term::PropertyPath(_) => {
                // Property paths have variable selectivity
                Ok(0.1)
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
            vars.insert(v.as_str().to_string());
        }
        if let Term::Variable(v) = &pattern.predicate {
            vars.insert(v.as_str().to_string());
        }
        if let Term::Variable(v) = &pattern.object {
            vars.insert(v.as_str().to_string());
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

        // Identify index intersection opportunities
        let index_intersections = self.identify_index_intersections(&patterns, &pattern_indexes)?;

        // Identify bloom filter candidates
        let bloom_filter_candidates =
            self.identify_bloom_filter_candidates(&patterns, pattern_selectivities)?;

        Ok(IndexUsagePlan {
            pattern_indexes,
            join_indexes,
            index_intersections,
            bloom_filter_candidates,
        })
    }

    /// Select best index for a pattern using advanced index types
    fn select_best_index_for_pattern(&self, pattern: &TriplePattern) -> Result<IndexType> {
        let subject_bound = self.is_bound(&pattern.subject);
        let predicate_bound = self.is_bound(&pattern.predicate);
        let object_bound = self.is_bound(&pattern.object);

        // Check for specialized index types first
        if let Some(specialized) = self.select_specialized_index(pattern)? {
            return Ok(specialized);
        }

        // Enhanced index selection with advanced types
        if subject_bound && predicate_bound && object_bound {
            // All three bound - try hash index for exact lookup
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::HashIndex(IndexPosition::FullTriple))
            {
                return Ok(IndexType::HashIndex(IndexPosition::FullTriple));
            }
        }

        if subject_bound && predicate_bound {
            // Try hash index for exact SP lookup
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::HashIndex(IndexPosition::SubjectPredicate))
            {
                return Ok(IndexType::HashIndex(IndexPosition::SubjectPredicate));
            }
            // Fall back to B+ tree index
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::BTreeIndex(IndexPosition::SubjectPredicate))
            {
                return Ok(IndexType::BTreeIndex(IndexPosition::SubjectPredicate));
            }
            // Traditional index
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SubjectPredicate)
            {
                return Ok(IndexType::SubjectPredicate);
            }
        }

        if predicate_bound && object_bound {
            // Check for bitmap index if predicate has low cardinality
            if self.is_low_cardinality_predicate(pattern) {
                if self
                    .index_stats
                    .available_indexes
                    .contains(&IndexType::BitmapIndex(IndexPosition::PredicateObject))
                {
                    return Ok(IndexType::BitmapIndex(IndexPosition::PredicateObject));
                }
            }
            // Hash index for exact PO lookup
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::HashIndex(IndexPosition::PredicateObject))
            {
                return Ok(IndexType::HashIndex(IndexPosition::PredicateObject));
            }
            // B+ tree index
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::BTreeIndex(IndexPosition::PredicateObject))
            {
                return Ok(IndexType::BTreeIndex(IndexPosition::PredicateObject));
            }
            // Traditional index
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::PredicateObject)
            {
                return Ok(IndexType::PredicateObject);
            }
        }

        if subject_bound && object_bound {
            // Hash index for exact SO lookup
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::HashIndex(IndexPosition::SubjectObject))
            {
                return Ok(IndexType::HashIndex(IndexPosition::SubjectObject));
            }
            // B+ tree index
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::BTreeIndex(IndexPosition::SubjectObject))
            {
                return Ok(IndexType::BTreeIndex(IndexPosition::SubjectObject));
            }
            // Traditional index
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SubjectObject)
            {
                return Ok(IndexType::SubjectObject);
            }
        }

        // Single bound term optimizations
        if predicate_bound {
            // Check for bitmap index for low cardinality predicates
            if self.is_low_cardinality_predicate(pattern) {
                if self
                    .index_stats
                    .available_indexes
                    .contains(&IndexType::BitmapIndex(IndexPosition::Predicate))
                {
                    return Ok(IndexType::BitmapIndex(IndexPosition::Predicate));
                }
            }
            // B+ tree for range queries
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::BTreeIndex(IndexPosition::Predicate))
            {
                return Ok(IndexType::BTreeIndex(IndexPosition::Predicate));
            }
        }

        // Default to subject-predicate B+ tree index
        if self
            .index_stats
            .available_indexes
            .contains(&IndexType::BTreeIndex(IndexPosition::SubjectPredicate))
        {
            Ok(IndexType::BTreeIndex(IndexPosition::SubjectPredicate))
        } else {
            Ok(IndexType::SubjectPredicate)
        }
    }

    /// Select specialized index types for specific pattern characteristics
    fn select_specialized_index(&self, pattern: &TriplePattern) -> Result<Option<IndexType>> {
        // Spatial index for geographic data
        if self.is_spatial_pattern(pattern) {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::SpatialRTree)
            {
                return Ok(Some(IndexType::SpatialRTree));
            }
        }

        // Temporal index for date/time data
        if self.is_temporal_pattern(pattern) {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::TemporalBTree)
            {
                return Ok(Some(IndexType::TemporalBTree));
            }
        }

        // Full-text index for text search
        if self.is_text_search_pattern(pattern) {
            if self
                .index_stats
                .available_indexes
                .contains(&IndexType::FullText)
            {
                return Ok(Some(IndexType::FullText));
            }
        }

        Ok(None)
    }

    /// Check if pattern involves low cardinality predicate suitable for bitmap index
    fn is_low_cardinality_predicate(&self, pattern: &TriplePattern) -> bool {
        if let Term::Iri(iri) = &pattern.predicate {
            if let Some(&frequency) = self.statistics.predicate_frequency.get(iri.as_str()) {
                // Use bitmap index if predicate appears in less than 1% of triples
                let total_triples = self
                    .statistics
                    .pattern_cardinality
                    .values()
                    .sum::<usize>()
                    .max(1);
                return frequency < total_triples / 100;
            }
        }
        false
    }

    /// Check if pattern involves spatial/geographic data
    fn is_spatial_pattern(&self, pattern: &TriplePattern) -> bool {
        if let Term::Iri(iri) = &pattern.predicate {
            let iri_str = iri.as_str();
            iri_str.contains("geo")
                || iri_str.contains("spatial")
                || iri_str.contains("latitude")
                || iri_str.contains("longitude")
                || iri_str.contains("wkt")
                || iri_str.contains("geometry")
        } else {
            false
        }
    }

    /// Check if pattern involves temporal/date-time data
    fn is_temporal_pattern(&self, pattern: &TriplePattern) -> bool {
        // Check predicate for temporal indicators
        if let Term::Iri(iri) = &pattern.predicate {
            let iri_str = iri.as_str();
            if iri_str.contains("date") || iri_str.contains("time") || iri_str.contains("temporal")
            {
                return true;
            }
        }

        // Check object for date/time literals
        if let Term::Literal(lit) = &pattern.object {
            if let Some(ref datatype) = lit.datatype {
                let dt_str = datatype.as_str();
                return dt_str.contains("date")
                    || dt_str.contains("time")
                    || dt_str.contains("temporal");
            }
        }

        false
    }

    /// Check if pattern involves text search
    fn is_text_search_pattern(&self, pattern: &TriplePattern) -> bool {
        if let Term::Iri(iri) = &pattern.predicate {
            let iri_str = iri.as_str();
            iri_str.contains("label")
                || iri_str.contains("comment")
                || iri_str.contains("description")
                || iri_str.contains("text")
        } else {
            false
        }
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

    /// Identify opportunities for index intersection
    fn identify_index_intersections(
        &self,
        patterns: &[TriplePattern],
        pattern_indexes: &[IndexAssignment],
    ) -> Result<Vec<IndexIntersection>> {
        let mut intersections = Vec::new();

        for (idx, pattern) in patterns.iter().enumerate() {
            let primary_index = pattern_indexes[idx].index_type.clone();
            let mut secondary_indexes = Vec::new();

            // Look for additional indexes that could be intersected
            let bound_positions = self.count_bound_positions(pattern);
            if bound_positions >= 2 {
                // Multi-dimensional pattern - candidate for intersection

                // Check available secondary indexes
                for index_type in &self.index_stats.available_indexes {
                    if *index_type != primary_index && self.is_index_applicable(pattern, index_type)
                    {
                        secondary_indexes.push(index_type.clone());
                    }
                }

                if !secondary_indexes.is_empty() {
                    let selectivity_improvement = self.estimate_intersection_benefit(
                        pattern,
                        &primary_index,
                        &secondary_indexes,
                    )?;

                    if selectivity_improvement > 0.3 {
                        // Significant improvement expected
                        let algorithm = self.recommend_intersection_algorithm(
                            &primary_index,
                            &secondary_indexes,
                            bound_positions,
                        );

                        intersections.push(IndexIntersection {
                            pattern_idx: idx,
                            primary_index,
                            secondary_indexes,
                            selectivity_improvement,
                            intersection_algorithm: algorithm,
                        });
                    }
                }
            }
        }

        Ok(intersections)
    }

    /// Identify candidates for bloom filter optimization
    fn identify_bloom_filter_candidates(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
    ) -> Result<Vec<BloomFilterCandidate>> {
        let mut candidates = Vec::new();

        for (idx, pattern) in patterns.iter().enumerate() {
            let selectivity = pattern_selectivities[idx].selectivity;

            // Bloom filters most effective for highly selective patterns
            if selectivity < 0.1 {
                // Check each position for bloom filter potential
                for position in [TermPosition::Subject, TermPosition::Object] {
                    if self.is_position_variable(pattern, position) {
                        let performance_gain =
                            self.estimate_bloom_filter_gain(pattern, position, selectivity)?;

                        if performance_gain > 1.5 {
                            let cardinality = self.get_total_count_for_position(position);
                            let false_positive_rate = 0.01; // 1% default
                            let memory_footprint =
                                self.estimate_bloom_filter_size(cardinality, false_positive_rate);

                            candidates.push(BloomFilterCandidate {
                                pattern_idx: idx,
                                filter_position: position,
                                false_positive_rate,
                                memory_footprint,
                                performance_gain,
                            });
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Count bound positions in a pattern
    fn count_bound_positions(&self, pattern: &TriplePattern) -> usize {
        let mut count = 0;
        if self.is_bound(&pattern.subject) {
            count += 1;
        }
        if self.is_bound(&pattern.predicate) {
            count += 1;
        }
        if self.is_bound(&pattern.object) {
            count += 1;
        }
        count
    }

    /// Check if an index is applicable to a pattern
    fn is_index_applicable(&self, pattern: &TriplePattern, index_type: &IndexType) -> bool {
        match index_type {
            IndexType::SubjectPredicate => {
                self.is_bound(&pattern.subject) || self.is_bound(&pattern.predicate)
            }
            IndexType::PredicateObject => {
                self.is_bound(&pattern.predicate) || self.is_bound(&pattern.object)
            }
            IndexType::SubjectObject => {
                self.is_bound(&pattern.subject) || self.is_bound(&pattern.object)
            }
            // RDF triple pattern permutation indices
            IndexType::SPO => true, // Supports all patterns
            IndexType::PSO => self.is_bound(&pattern.predicate) || self.is_bound(&pattern.subject),
            IndexType::OSP => self.is_bound(&pattern.object) || self.is_bound(&pattern.subject),
            IndexType::OPS => self.is_bound(&pattern.object) || self.is_bound(&pattern.predicate),
            IndexType::SOP => self.is_bound(&pattern.subject) || self.is_bound(&pattern.object),
            IndexType::POS => self.is_bound(&pattern.predicate) || self.is_bound(&pattern.object),
            // Simple index types
            IndexType::Hash => {
                // Hash indexes work best with bound terms
                self.is_bound(&pattern.subject)
                    && self.is_bound(&pattern.predicate)
                    && self.is_bound(&pattern.object)
            }
            IndexType::BTree => true, // B-tree supports range queries and existence
            IndexType::Bitmap => {
                // Bitmap indexes work well for low-cardinality predicates
                self.is_bound(&pattern.predicate)
            }
            IndexType::Bloom => {
                // Bloom filters are good for existence checks
                self.is_bound(&pattern.subject)
                    || self.is_bound(&pattern.predicate)
                    || self.is_bound(&pattern.object)
            }
            IndexType::FullText => {
                // Full-text indexes apply to literal values
                matches!(pattern.object, Term::Literal(_))
            }
            IndexType::Spatial => {
                // Spatial indexes apply to geographic data (specialized use case)
                false
            }
            IndexType::Temporal => {
                // Temporal indexes apply to date/time literals
                if let Term::Literal(lit) = &pattern.object {
                    lit.datatype.as_ref().map_or(false, |dt| {
                        let dt_str = dt.as_str();
                        dt_str.contains("date") || dt_str.contains("time")
                    })
                } else {
                    false
                }
            }
            IndexType::Custom(_) => {
                // Custom indexes require specialized logic
                false
            }
            IndexType::BTreeIndex(pos) => {
                // BTree indexes are efficient for range queries
                self.pattern_benefits_from_index(pattern, pos)
            }
            IndexType::HashIndex(pos) => {
                // Hash indexes are efficient for equality lookups
                self.pattern_benefits_from_index(pattern, pos)
            }
            IndexType::BitmapIndex(pos) => {
                // Bitmap indexes are efficient for low-cardinality data
                self.pattern_benefits_from_index(pattern, pos)
            }
            IndexType::SpatialRTree => {
                // Spatial R-tree indexes for geographic data
                false
            }
            IndexType::TemporalBTree => {
                // Temporal BTree indexes for time-series data
                if let Term::Literal(lit) = &pattern.object {
                    lit.datatype.as_ref().map_or(false, |dt| {
                        let dt_str = dt.as_str();
                        dt_str.contains("date") || dt_str.contains("time")
                    })
                } else {
                    false
                }
            }
            IndexType::MultiColumnBTree(positions) => {
                // Multi-column indexes benefit from multiple bound terms
                positions
                    .iter()
                    .any(|pos| self.pattern_benefits_from_index(pattern, pos))
            }
            IndexType::BloomFilter(pos) => {
                // Bloom filters are useful for membership testing
                self.pattern_benefits_from_index(pattern, pos)
            }
        }
    }

    /// Check if a pattern benefits from a specific index position
    fn pattern_benefits_from_index(&self, pattern: &TriplePattern, pos: &IndexPosition) -> bool {
        match pos {
            IndexPosition::Subject => self.is_bound(&pattern.subject),
            IndexPosition::Predicate => self.is_bound(&pattern.predicate),
            IndexPosition::Object => self.is_bound(&pattern.object),
            IndexPosition::SubjectPredicate => {
                self.is_bound(&pattern.subject) || self.is_bound(&pattern.predicate)
            }
            IndexPosition::PredicateObject => {
                self.is_bound(&pattern.predicate) || self.is_bound(&pattern.object)
            }
            IndexPosition::SubjectObject => {
                self.is_bound(&pattern.subject) || self.is_bound(&pattern.object)
            }
            IndexPosition::FullTriple => {
                self.is_bound(&pattern.subject)
                    && self.is_bound(&pattern.predicate)
                    && self.is_bound(&pattern.object)
            }
        }
    }

    /// Estimate benefit of index intersection
    fn estimate_intersection_benefit(
        &self,
        pattern: &TriplePattern,
        primary_index: &IndexType,
        secondary_indexes: &[IndexType],
    ) -> Result<f64> {
        let base_selectivity = 0.1; // Assumed selectivity without intersection
        let intersection_factor = 1.0 / (1.0 + secondary_indexes.len() as f64);
        Ok(1.0 - (base_selectivity * intersection_factor))
    }

    /// Recommend intersection algorithm based on characteristics
    fn recommend_intersection_algorithm(
        &self,
        primary_index: &IndexType,
        secondary_indexes: &[IndexType],
        bound_positions: usize,
    ) -> IntersectionAlgorithm {
        if secondary_indexes.len() > 2 {
            IntersectionAlgorithm::Bitmap // Best for multiple intersections
        } else if bound_positions == 3 {
            IntersectionAlgorithm::Hash // Good for highly selective queries
        } else {
            IntersectionAlgorithm::SkipList // Balanced approach
        }
    }

    /// Check if a position in pattern is a variable
    fn is_position_variable(&self, pattern: &TriplePattern, position: TermPosition) -> bool {
        match position {
            TermPosition::Subject => matches!(pattern.subject, Term::Variable(_)),
            TermPosition::Predicate => matches!(pattern.predicate, Term::Variable(_)),
            TermPosition::Object => matches!(pattern.object, Term::Variable(_)),
        }
    }

    /// Estimate performance gain from bloom filter
    fn estimate_bloom_filter_gain(
        &self,
        pattern: &TriplePattern,
        position: TermPosition,
        selectivity: f64,
    ) -> Result<f64> {
        // Bloom filters most effective for negative lookups
        let negative_lookup_ratio = 1.0 - selectivity;
        let access_cost_reduction = 0.9; // 90% reduction in unnecessary accesses
        Ok(1.0 + (negative_lookup_ratio * access_cost_reduction * 10.0))
    }

    /// Estimate bloom filter memory size
    fn estimate_bloom_filter_size(&self, cardinality: usize, false_positive_rate: f64) -> usize {
        // Approximate bloom filter size calculation
        let optimal_bits = (cardinality as f64
            * (-false_positive_rate.ln() / (2.0_f64.ln().powi(2))))
        .ceil() as usize;
        optimal_bits / 8 // Convert bits to bytes
    }

    // Enhanced index intersection methods for advanced optimization

    /// Identify intersections for OR conditions across patterns
    fn identify_or_condition_intersections(
        &self,
        patterns: &[TriplePattern],
        pattern_indexes: &[IndexAssignment],
    ) -> Result<Vec<IndexIntersection>> {
        let mut intersections = Vec::new();

        // Look for patterns that could benefit from index union (OR operations)
        for i in 0..patterns.len() {
            for j in i + 1..patterns.len() {
                if self.can_benefit_from_index_union(&patterns[i], &patterns[j]) {
                    let union_opportunity = self.create_index_union_intersection(
                        i,
                        j,
                        &pattern_indexes[i].index_type,
                        &pattern_indexes[j].index_type,
                        &patterns[i],
                        &patterns[j],
                    )?;

                    if let Some(intersection) = union_opportunity {
                        intersections.push(intersection);
                    }
                }
            }
        }

        Ok(intersections)
    }

    /// Identify dynamic index selections based on workload patterns
    fn identify_dynamic_index_selections(
        &self,
        patterns: &[TriplePattern],
        pattern_indexes: &[IndexAssignment],
    ) -> Result<Vec<IndexIntersection>> {
        let mut intersections = Vec::new();

        // Use adaptive selector to determine if alternative indexes would be better
        for (idx, pattern) in patterns.iter().enumerate() {
            let current_index = &pattern_indexes[idx].index_type;

            // Get alternative indexes recommended by adaptive selector
            let alternatives = self.get_adaptive_index_alternatives(pattern, current_index)?;

            if !alternatives.is_empty() {
                let selectivity_improvement =
                    self.estimate_adaptive_improvement(pattern, current_index, &alternatives)?;

                if selectivity_improvement > 0.2 {
                    let algorithm = IntersectionAlgorithm::Hash; // Use hash for dynamic selection

                    intersections.push(IndexIntersection {
                        pattern_idx: idx,
                        primary_index: current_index.clone(),
                        secondary_indexes: alternatives,
                        selectivity_improvement,
                        intersection_algorithm: algorithm,
                    });
                }
            }
        }

        Ok(intersections)
    }

    /// Enhanced index applicability check with advanced index types
    fn is_index_applicable_enhanced(
        &self,
        pattern: &TriplePattern,
        index_type: &IndexType,
    ) -> bool {
        match index_type {
            IndexType::BTreeIndex(position) => self.is_btree_applicable(pattern, position),
            IndexType::HashIndex(position) => self.is_hash_applicable(pattern, position),
            IndexType::BitmapIndex(position) => self.is_bitmap_applicable(pattern, position),
            IndexType::BloomFilter(position) => self.is_bloom_filter_applicable(pattern, position),
            IndexType::SpatialRTree => self.is_spatial_pattern(pattern),
            IndexType::TemporalBTree => self.is_temporal_pattern(pattern),
            IndexType::MultiColumnBTree(positions) => {
                self.is_multi_column_applicable(pattern, positions)
            }
            _ => self.is_index_applicable(pattern, index_type), // Fall back to original method
        }
    }

    /// Check if B+ tree index is applicable
    fn is_btree_applicable(
        &self,
        pattern: &TriplePattern,
        position: &crate::optimizer::IndexPosition,
    ) -> bool {
        use crate::optimizer::IndexPosition;
        match position {
            IndexPosition::Subject => self.is_bound(&pattern.subject),
            IndexPosition::Predicate => self.is_bound(&pattern.predicate),
            IndexPosition::Object => self.is_bound(&pattern.object),
            IndexPosition::SubjectPredicate => {
                self.is_bound(&pattern.subject) || self.is_bound(&pattern.predicate)
            }
            IndexPosition::PredicateObject => {
                self.is_bound(&pattern.predicate) || self.is_bound(&pattern.object)
            }
            IndexPosition::SubjectObject => {
                self.is_bound(&pattern.subject) || self.is_bound(&pattern.object)
            }
            IndexPosition::FullTriple => {
                self.is_bound(&pattern.subject)
                    && self.is_bound(&pattern.predicate)
                    && self.is_bound(&pattern.object)
            }
        }
    }

    /// Check if hash index is applicable (requires exact matches)
    fn is_hash_applicable(
        &self,
        pattern: &TriplePattern,
        position: &crate::optimizer::IndexPosition,
    ) -> bool {
        use crate::optimizer::IndexPosition;
        match position {
            IndexPosition::Subject => self.is_bound(&pattern.subject),
            IndexPosition::Predicate => self.is_bound(&pattern.predicate),
            IndexPosition::Object => self.is_bound(&pattern.object),
            IndexPosition::SubjectPredicate => {
                self.is_bound(&pattern.subject) && self.is_bound(&pattern.predicate)
            }
            IndexPosition::PredicateObject => {
                self.is_bound(&pattern.predicate) && self.is_bound(&pattern.object)
            }
            IndexPosition::SubjectObject => {
                self.is_bound(&pattern.subject) && self.is_bound(&pattern.object)
            }
            IndexPosition::FullTriple => {
                self.is_bound(&pattern.subject)
                    && self.is_bound(&pattern.predicate)
                    && self.is_bound(&pattern.object)
            }
        }
    }

    /// Check if bitmap index is applicable (good for low cardinality)
    fn is_bitmap_applicable(
        &self,
        pattern: &TriplePattern,
        position: &crate::optimizer::IndexPosition,
    ) -> bool {
        use crate::optimizer::IndexPosition;
        match position {
            IndexPosition::Predicate => {
                self.is_bound(&pattern.predicate) && self.is_low_cardinality_predicate(pattern)
            }
            IndexPosition::PredicateObject => {
                self.is_bound(&pattern.predicate) && self.is_low_cardinality_predicate(pattern)
            }
            _ => false, // Bitmap indexes most useful for predicates
        }
    }

    /// Check if bloom filter is applicable (good for negative lookups)
    fn is_bloom_filter_applicable(
        &self,
        pattern: &TriplePattern,
        position: &crate::optimizer::IndexPosition,
    ) -> bool {
        use crate::optimizer::IndexPosition;
        // Bloom filters are useful when we expect many negative lookups
        match position {
            IndexPosition::Subject => self.is_bound(&pattern.subject),
            IndexPosition::Object => self.is_bound(&pattern.object),
            _ => false,
        }
    }

    /// Check if multi-column B+ tree is applicable
    fn is_multi_column_applicable(
        &self,
        pattern: &TriplePattern,
        positions: &[crate::optimizer::IndexPosition],
    ) -> bool {
        positions
            .iter()
            .any(|pos| self.is_btree_applicable(pattern, pos))
    }

    /// Check if two patterns can benefit from index union
    fn can_benefit_from_index_union(&self, p1: &TriplePattern, p2: &TriplePattern) -> bool {
        // Patterns can benefit from union if they have similar structure but different constants
        self.have_similar_structure(p1, p2) && !self.have_same_constants(p1, p2)
    }

    /// Check if patterns have similar structure
    fn have_similar_structure(&self, p1: &TriplePattern, p2: &TriplePattern) -> bool {
        let p1_structure = (
            matches!(p1.subject, Term::Variable(_)),
            matches!(p1.predicate, Term::Variable(_)),
            matches!(p1.object, Term::Variable(_)),
        );
        let p2_structure = (
            matches!(p2.subject, Term::Variable(_)),
            matches!(p2.predicate, Term::Variable(_)),
            matches!(p2.object, Term::Variable(_)),
        );
        p1_structure == p2_structure
    }

    /// Check if patterns have the same constants
    fn have_same_constants(&self, p1: &TriplePattern, p2: &TriplePattern) -> bool {
        format!("{:?}", p1) == format!("{:?}", p2)
    }

    /// Create index union intersection for OR conditions
    fn create_index_union_intersection(
        &self,
        idx1: usize,
        idx2: usize,
        index1: &IndexType,
        index2: &IndexType,
        pattern1: &TriplePattern,
        pattern2: &TriplePattern,
    ) -> Result<Option<IndexIntersection>> {
        // Estimate benefit of union operation
        let union_benefit = self.estimate_union_benefit(pattern1, pattern2)?;

        if union_benefit > 0.25 {
            Ok(Some(IndexIntersection {
                pattern_idx: idx1, // Primary pattern
                primary_index: index1.clone(),
                secondary_indexes: vec![index2.clone()],
                selectivity_improvement: union_benefit,
                intersection_algorithm: IntersectionAlgorithm::Hash, // Good for unions
            }))
        } else {
            Ok(None)
        }
    }

    /// Get adaptive index alternatives based on workload patterns
    fn get_adaptive_index_alternatives(
        &self,
        pattern: &TriplePattern,
        current: &IndexType,
    ) -> Result<Vec<IndexType>> {
        use crate::optimizer::IndexPosition;
        let mut alternatives = Vec::new();

        // Check if workload characteristics suggest better alternatives
        let pattern_key = self.adaptive_selector.pattern_key(pattern);
        let pattern_frequency = self
            .adaptive_selector
            .pattern_frequency
            .get(&pattern_key)
            .copied()
            .unwrap_or(0);

        // For frequently accessed patterns, consider hash indexes
        if pattern_frequency > 100 {
            if !matches!(current, IndexType::HashIndex(_)) {
                if self.count_bound_positions(pattern) >= 2 {
                    alternatives.push(IndexType::HashIndex(IndexPosition::SubjectPredicate));
                }
            }
        }

        // For low cardinality patterns, consider bitmap indexes
        if self.is_low_cardinality_predicate(pattern) {
            if !matches!(current, IndexType::BitmapIndex(_)) {
                alternatives.push(IndexType::BitmapIndex(IndexPosition::Predicate));
            }
        }

        Ok(alternatives)
    }

    /// Enhanced intersection benefit estimation with sophisticated cost modeling
    fn estimate_intersection_benefit_enhanced(
        &self,
        pattern: &TriplePattern,
        primary_index: &IndexType,
        secondary_indexes: &[IndexType],
    ) -> Result<f64> {
        let base_selectivity = self.get_index_selectivity(primary_index);
        let mut combined_selectivity = base_selectivity;
        let mut intersection_cost = 0.0;

        // Calculate pattern cardinality for more accurate estimation
        let pattern_cardinality = self
            .statistics
            .pattern_cardinality
            .get(&format!("{:?}", pattern))
            .copied()
            .unwrap_or(10000);

        for secondary in secondary_indexes {
            let secondary_selectivity = self.get_index_selectivity(secondary);
            let intersection_overhead =
                self.calculate_intersection_overhead(primary_index, secondary);

            // Advanced combination rules based on index characteristics
            combined_selectivity = match (primary_index, secondary) {
                (IndexType::BTreeIndex(_), IndexType::HashIndex(_)) => {
                    // B-tree + Hash: Excellent for range + exact match
                    let synergy_factor = if pattern_cardinality > 100000 {
                        0.7
                    } else {
                        0.8
                    };
                    combined_selectivity * secondary_selectivity * synergy_factor
                }
                (IndexType::BitmapIndex(_), IndexType::BTreeIndex(_)) => {
                    // Bitmap + B-tree: Outstanding for categorical + range queries
                    let synergy_factor = if self.is_low_cardinality_predicate(pattern) {
                        0.6
                    } else {
                        0.9
                    };
                    combined_selectivity * secondary_selectivity * synergy_factor
                }
                (IndexType::BloomFilter(_), _) => {
                    // Bloom filters provide massive reduction for negative lookups
                    let negative_ratio = 1.0 - base_selectivity;
                    combined_selectivity * (0.05 + 0.95 * (1.0 - negative_ratio))
                }
                (IndexType::SpatialRTree, IndexType::BTreeIndex(_)) => {
                    // Spatial + B-tree: Great for geo + temporal queries
                    combined_selectivity * secondary_selectivity * 0.75
                }
                (IndexType::HashIndex(_), IndexType::BitmapIndex(_)) => {
                    // Hash + Bitmap: Good for exact match + categorical
                    combined_selectivity * secondary_selectivity * 0.85
                }
                (IndexType::MultiColumnBTree(_), _) => {
                    // Multi-column with any secondary: Diminishing returns
                    combined_selectivity * secondary_selectivity * 0.95
                }
                _ => {
                    // Standard independence assumption with correlation penalty
                    let correlation_factor = 1.1 + (secondary_indexes.len() as f64 * 0.05);
                    combined_selectivity * secondary_selectivity * correlation_factor
                }
            };

            intersection_cost += intersection_overhead;
        }

        // Account for intersection processing cost
        let cost_benefit_ratio = if intersection_cost > 100.0 {
            0.9 // High intersection cost reduces benefit
        } else if intersection_cost > 50.0 {
            0.95
        } else {
            1.0
        };

        let improvement = (base_selectivity - combined_selectivity) / base_selectivity;
        Ok(improvement * cost_benefit_ratio)
    }

    /// Calculate intersection overhead for different index combinations
    fn calculate_intersection_overhead(&self, primary: &IndexType, secondary: &IndexType) -> f64 {
        match (primary, secondary) {
            (IndexType::BTreeIndex(_), IndexType::HashIndex(_)) => 10.0, // Moderate overhead
            (IndexType::BitmapIndex(_), IndexType::BitmapIndex(_)) => 5.0, // Low overhead - bitwise ops
            (IndexType::BloomFilter(_), _) => 2.0,                         // Very low overhead
            (IndexType::SpatialRTree, _) => 25.0, // Higher overhead for spatial operations
            (IndexType::HashIndex(_), IndexType::HashIndex(_)) => 15.0, // Hash probe overhead
            _ => 12.0,                            // Default overhead
        }
    }

    /// Enhanced intersection algorithm recommendation
    fn recommend_intersection_algorithm_enhanced(
        &self,
        primary_index: &IndexType,
        secondary_indexes: &[IndexType],
        bound_positions: usize,
        pattern: &TriplePattern,
    ) -> IntersectionAlgorithm {
        // Bitmap intersection for bitmap indexes
        if matches!(primary_index, IndexType::BitmapIndex(_))
            || secondary_indexes
                .iter()
                .any(|idx| matches!(idx, IndexType::BitmapIndex(_)))
        {
            return IntersectionAlgorithm::Bitmap;
        }

        // Hash intersection for hash indexes or high selectivity
        if matches!(primary_index, IndexType::HashIndex(_))
            || secondary_indexes
                .iter()
                .any(|idx| matches!(idx, IndexType::HashIndex(_)))
        {
            return IntersectionAlgorithm::Hash;
        }

        // Skip list for B+ tree indexes (ordered data)
        if matches!(primary_index, IndexType::BTreeIndex(_))
            && secondary_indexes
                .iter()
                .all(|idx| matches!(idx, IndexType::BTreeIndex(_)))
        {
            return IntersectionAlgorithm::SkipList;
        }

        // Default based on complexity
        if secondary_indexes.len() > 2 {
            IntersectionAlgorithm::Bitmap
        } else if bound_positions == 3 {
            IntersectionAlgorithm::Hash
        } else {
            IntersectionAlgorithm::SkipList
        }
    }

    /// Get index selectivity for cost estimation
    fn get_index_selectivity(&self, index_type: &IndexType) -> f64 {
        self.index_stats
            .index_selectivity
            .get(index_type)
            .copied()
            .unwrap_or_else(|| {
                // Default selectivities for different index types
                match index_type {
                    IndexType::HashIndex(_) => 0.01,    // Very selective
                    IndexType::BTreeIndex(_) => 0.05,   // Good selectivity
                    IndexType::BitmapIndex(_) => 0.1,   // Moderate for low cardinality
                    IndexType::BloomFilter(_) => 0.001, // Excellent for negative lookups
                    IndexType::SpatialRTree => 0.02,    // Good for spatial queries
                    IndexType::TemporalBTree => 0.03,   // Good for temporal queries
                    _ => 0.1,                           // Default
                }
            })
    }

    /// Estimate union benefit for OR conditions
    fn estimate_union_benefit(&self, p1: &TriplePattern, p2: &TriplePattern) -> Result<f64> {
        let p1_selectivity =
            self.calculate_term_selectivity(&p1.predicate, TermPosition::Predicate)?;
        let p2_selectivity =
            self.calculate_term_selectivity(&p2.predicate, TermPosition::Predicate)?;

        // Union benefit is higher when individual selectivities are low
        let combined_selectivity = p1_selectivity + p2_selectivity;
        Ok((1.0 - combined_selectivity).max(0.0))
    }

    /// Estimate adaptive improvement from alternative indexes
    fn estimate_adaptive_improvement(
        &self,
        pattern: &TriplePattern,
        current_index: &IndexType,
        alternatives: &[IndexType],
    ) -> Result<f64> {
        let current_selectivity = self.get_index_selectivity(current_index);
        let mut best_alternative_selectivity = current_selectivity;

        for alternative in alternatives {
            let alt_selectivity = self.get_index_selectivity(alternative);
            if alt_selectivity < best_alternative_selectivity {
                best_alternative_selectivity = alt_selectivity;
            }
        }

        Ok((current_selectivity - best_alternative_selectivity) / current_selectivity)
    }

    /// Advanced index-aware streaming optimization for large datasets
    pub fn optimize_for_streaming(
        &self,
        patterns: &[TriplePattern],
        memory_limit: usize,
    ) -> Result<StreamingOptimizationPlan> {
        let pattern_selectivities = self.calculate_pattern_selectivities(patterns)?;
        let total_estimated_memory = self.estimate_total_memory_usage(&pattern_selectivities)?;

        let mut streaming_plan = StreamingOptimizationPlan {
            use_streaming: total_estimated_memory > memory_limit,
            streaming_patterns: Vec::new(),
            index_streaming_recommendations: Vec::new(),
            memory_estimation: total_estimated_memory,
            pipeline_breakers: Vec::new(),
            spill_candidates: Vec::new(),
        };

        if streaming_plan.use_streaming {
            // Identify patterns that should use streaming execution
            for (idx, pattern_sel) in pattern_selectivities.iter().enumerate() {
                let pattern_memory = self.estimate_pattern_memory_usage(pattern_sel)?;

                if pattern_memory > memory_limit / 4 {
                    streaming_plan.streaming_patterns.push(StreamingPattern {
                        pattern_index: idx,
                        pattern: pattern_sel.pattern.clone(),
                        estimated_memory: pattern_memory,
                        streaming_strategy: self.recommend_streaming_strategy(pattern_sel)?,
                    });
                }
            }

            // Generate index-specific streaming recommendations
            streaming_plan.index_streaming_recommendations =
                self.generate_index_streaming_recommendations(patterns, &pattern_selectivities)?;

            // Identify pipeline breaker locations
            streaming_plan.pipeline_breakers =
                self.identify_pipeline_breakers(patterns, &pattern_selectivities, memory_limit)?;

            // Identify spill candidates for memory-intensive operations
            streaming_plan.spill_candidates =
                self.identify_spill_candidates(patterns, &pattern_selectivities, memory_limit)?;
        }

        Ok(streaming_plan)
    }

    /// Estimate total memory usage for a set of patterns
    fn estimate_total_memory_usage(
        &self,
        pattern_selectivities: &[PatternSelectivity],
    ) -> Result<usize> {
        let mut total_memory = 0;
        let mut cumulative_cardinality = 1;

        for pattern_sel in pattern_selectivities {
            // Pattern evaluation memory
            let pattern_memory = pattern_sel.cardinality * 150; // Bytes per result
            total_memory += pattern_memory;

            // Join memory (hash table for smaller side)
            if cumulative_cardinality > 0 {
                let join_memory = cumulative_cardinality.min(pattern_sel.cardinality) * 200;
                total_memory += join_memory;
            }

            // Update cumulative cardinality for next join
            cumulative_cardinality = (cumulative_cardinality as f64
                * pattern_sel.cardinality as f64
                * pattern_sel.selectivity)
                .ceil() as usize;
        }

        Ok(total_memory)
    }

    /// Estimate memory usage for a single pattern
    fn estimate_pattern_memory_usage(&self, pattern_sel: &PatternSelectivity) -> Result<usize> {
        let base_memory = pattern_sel.cardinality * 150; // Base result memory
        let index_overhead = match self.recommend_best_index_for_pattern(&pattern_sel.pattern)? {
            IndexType::HashIndex(_) => 1.2,
            IndexType::BTreeIndex(_) => 1.1,
            IndexType::BitmapIndex(_) => 0.8,
            IndexType::BloomFilter(_) => 0.3,
            _ => 1.0,
        };

        Ok((base_memory as f64 * index_overhead) as usize)
    }

    /// Recommend streaming strategy for a pattern
    fn recommend_streaming_strategy(
        &self,
        pattern_sel: &PatternSelectivity,
    ) -> Result<StreamingStrategy> {
        let cardinality = pattern_sel.cardinality;
        let selectivity = pattern_sel.selectivity;

        if cardinality > 1_000_000 && selectivity < 0.01 {
            // Highly selective on large data - use index streaming
            Ok(StreamingStrategy::IndexedStreaming)
        } else if cardinality > 500_000 {
            // Large intermediate results - use batched streaming
            Ok(StreamingStrategy::BatchedStreaming { batch_size: 10_000 })
        } else if selectivity > 0.5 {
            // Low selectivity - use filtered streaming
            Ok(StreamingStrategy::FilteredStreaming)
        } else {
            // Default streaming
            Ok(StreamingStrategy::StandardStreaming)
        }
    }

    /// Generate index-specific streaming recommendations
    fn generate_index_streaming_recommendations(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
    ) -> Result<Vec<IndexStreamingRecommendation>> {
        let mut recommendations = Vec::new();

        for (idx, pattern) in patterns.iter().enumerate() {
            let best_index = self.recommend_best_index_for_pattern(pattern)?;
            let pattern_sel = &pattern_selectivities[idx];

            match best_index {
                IndexType::BTreeIndex(_) => {
                    // B-tree indexes support range scanning
                    if pattern_sel.cardinality > 100_000 {
                        recommendations.push(IndexStreamingRecommendation {
                            pattern_index: idx,
                            index_type: best_index,
                            streaming_mode: IndexStreamingMode::RangeScan {
                                batch_size: 10_000,
                                prefetch_size: 50_000,
                            },
                            memory_limit: pattern_sel.cardinality * 100,
                        });
                    }
                }
                IndexType::HashIndex(_) => {
                    // Hash indexes support batch lookup
                    if pattern_sel.cardinality > 50_000 {
                        recommendations.push(IndexStreamingRecommendation {
                            pattern_index: idx,
                            index_type: best_index,
                            streaming_mode: IndexStreamingMode::BatchLookup { batch_size: 5_000 },
                            memory_limit: pattern_sel.cardinality * 80,
                        });
                    }
                }
                IndexType::BitmapIndex(_) => {
                    // Bitmap indexes support bit-stream processing
                    recommendations.push(IndexStreamingRecommendation {
                        pattern_index: idx,
                        index_type: best_index,
                        streaming_mode: IndexStreamingMode::BitStream {
                            chunk_size: 1024 * 8, // 8KB chunks
                        },
                        memory_limit: pattern_sel.cardinality * 20,
                    });
                }
                _ => {} // No specific streaming recommendation
            }
        }

        Ok(recommendations)
    }

    /// Identify optimal pipeline breaker locations
    fn identify_pipeline_breakers(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
        memory_limit: usize,
    ) -> Result<Vec<PipelineBreaker>> {
        let mut breakers = Vec::new();
        let mut cumulative_memory = 0;

        for (idx, pattern_sel) in pattern_selectivities.iter().enumerate() {
            let pattern_memory = self.estimate_pattern_memory_usage(pattern_sel)?;
            cumulative_memory += pattern_memory;

            // Insert pipeline breaker if memory threshold exceeded
            if cumulative_memory > memory_limit / 2 {
                breakers.push(PipelineBreaker {
                    location: PipelineBreakerLocation::AfterPattern(idx.saturating_sub(1)),
                    estimated_memory_saving: cumulative_memory / 2,
                    spill_strategy: if pattern_sel.cardinality > 100_000 {
                        SpillStrategy::SortSpill
                    } else {
                        SpillStrategy::HashSpill
                    },
                });
                cumulative_memory = pattern_memory; // Reset after breaker
            }

            // Insert breaker before expensive joins
            if idx > 0 {
                let prev_cardinality = pattern_selectivities[idx - 1].cardinality;
                let join_cost = prev_cardinality * pattern_sel.cardinality;

                if join_cost > 10_000_000 {
                    // 10M tuple join threshold
                    breakers.push(PipelineBreaker {
                        location: PipelineBreakerLocation::BeforeJoin(idx - 1, idx),
                        estimated_memory_saving: join_cost / 10,
                        spill_strategy: SpillStrategy::HashSpill,
                    });
                }
            }
        }

        Ok(breakers)
    }

    /// Identify candidates for spilling to disk
    fn identify_spill_candidates(
        &self,
        patterns: &[TriplePattern],
        pattern_selectivities: &[PatternSelectivity],
        memory_limit: usize,
    ) -> Result<Vec<SpillCandidate>> {
        let mut candidates = Vec::new();

        for (idx, pattern_sel) in pattern_selectivities.iter().enumerate() {
            let pattern_memory = self.estimate_pattern_memory_usage(pattern_sel)?;

            // Large intermediate results are good spill candidates
            if pattern_memory > memory_limit / 8 {
                let spill_benefit =
                    self.calculate_spill_benefit(pattern_sel, pattern_memory, memory_limit);

                candidates.push(SpillCandidate {
                    pattern_index: idx,
                    operation: SpillOperation::IntermediateResults,
                    estimated_memory: pattern_memory,
                    spill_benefit,
                    spill_cost: self.estimate_spill_cost(pattern_memory),
                });
            }

            // Hash tables for joins are good spill candidates
            if idx > 0 {
                let join_memory = pattern_selectivities[idx - 1].cardinality * 200;
                if join_memory > memory_limit / 10 {
                    let spill_benefit =
                        self.calculate_spill_benefit(pattern_sel, join_memory, memory_limit);

                    candidates.push(SpillCandidate {
                        pattern_index: idx,
                        operation: SpillOperation::HashTable,
                        estimated_memory: join_memory,
                        spill_benefit,
                        spill_cost: self.estimate_spill_cost(join_memory),
                    });
                }
            }
        }

        // Sort by benefit/cost ratio
        candidates.sort_by(|a, b| {
            let ratio_a = a.spill_benefit / a.spill_cost;
            let ratio_b = b.spill_benefit / b.spill_cost;
            ratio_b.partial_cmp(&ratio_a).unwrap()
        });

        Ok(candidates)
    }

    /// Calculate benefit of spilling an operation
    fn calculate_spill_benefit(
        &self,
        pattern_sel: &PatternSelectivity,
        memory_usage: usize,
        memory_limit: usize,
    ) -> f64 {
        let memory_pressure = memory_usage as f64 / memory_limit as f64;
        let cardinality_factor = (pattern_sel.cardinality as f64).ln() / 10.0;
        let selectivity_factor = 1.0 - pattern_sel.selectivity;

        memory_pressure * cardinality_factor * selectivity_factor
    }

    /// Estimate cost of spilling (I/O overhead)
    fn estimate_spill_cost(&self, memory_usage: usize) -> f64 {
        // Assume 100 MB/s I/O throughput
        let io_time = memory_usage as f64 / (100.0 * 1024.0 * 1024.0);

        // Cost includes write + read overhead
        io_time * 2.0 + 0.1 // Base spill overhead
    }

    /// Recommend best index for a pattern (helper method)
    fn recommend_best_index_for_pattern(&self, pattern: &TriplePattern) -> Result<IndexType> {
        self.select_best_index_for_pattern(pattern)
    }
}

/// Streaming optimization plan for large datasets
#[derive(Debug, Clone)]
pub struct StreamingOptimizationPlan {
    /// Whether streaming execution is recommended
    pub use_streaming: bool,
    /// Patterns that should use streaming execution
    pub streaming_patterns: Vec<StreamingPattern>,
    /// Index-specific streaming recommendations
    pub index_streaming_recommendations: Vec<IndexStreamingRecommendation>,
    /// Total estimated memory usage
    pub memory_estimation: usize,
    /// Pipeline breaker locations
    pub pipeline_breakers: Vec<PipelineBreaker>,
    /// Spill candidates for memory management
    pub spill_candidates: Vec<SpillCandidate>,
}

/// Pattern-specific streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingPattern {
    pub pattern_index: usize,
    pub pattern: TriplePattern,
    pub estimated_memory: usize,
    pub streaming_strategy: StreamingStrategy,
}

/// Streaming strategies for different scenarios
#[derive(Debug, Clone)]
pub enum StreamingStrategy {
    /// Standard streaming for moderate datasets
    StandardStreaming,
    /// Index-based streaming for highly selective queries
    IndexedStreaming,
    /// Batched streaming for large intermediate results
    BatchedStreaming { batch_size: usize },
    /// Filtered streaming for low selectivity queries
    FilteredStreaming,
}

/// Index-specific streaming recommendation
#[derive(Debug, Clone)]
pub struct IndexStreamingRecommendation {
    pub pattern_index: usize,
    pub index_type: IndexType,
    pub streaming_mode: IndexStreamingMode,
    pub memory_limit: usize,
}

/// Index streaming modes
#[derive(Debug, Clone)]
pub enum IndexStreamingMode {
    /// Range scan for B-tree indexes
    RangeScan {
        batch_size: usize,
        prefetch_size: usize,
    },
    /// Batch lookup for hash indexes
    BatchLookup { batch_size: usize },
    /// Bit-stream processing for bitmap indexes
    BitStream { chunk_size: usize },
}

/// Pipeline breaker for memory management
#[derive(Debug, Clone)]
pub struct PipelineBreaker {
    pub location: PipelineBreakerLocation,
    pub estimated_memory_saving: usize,
    pub spill_strategy: SpillStrategy,
}

/// Pipeline breaker locations
#[derive(Debug, Clone)]
pub enum PipelineBreakerLocation {
    /// After processing a specific pattern
    AfterPattern(usize),
    /// Before joining two patterns
    BeforeJoin(usize, usize),
}

/// Spill strategies for different operations
#[derive(Debug, Clone)]
pub enum SpillStrategy {
    /// Hash-based spilling for joins
    HashSpill,
    /// Sort-based spilling for order-by operations
    SortSpill,
}

/// Spill candidate for memory-intensive operations
#[derive(Debug, Clone)]
pub struct SpillCandidate {
    pub pattern_index: usize,
    pub operation: SpillOperation,
    pub estimated_memory: usize,
    pub spill_benefit: f64,
    pub spill_cost: f64,
}

/// Operations that can be spilled to disk
#[derive(Debug, Clone)]
pub enum SpillOperation {
    /// Intermediate result sets
    IntermediateResults,
    /// Hash table for joins
    HashTable,
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
            Term::QuotedTriple(_) => "QT",
            Term::PropertyPath(_) => "PP",
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
    use crate::algebra::{Iri, Literal, Variable};

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
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            )),
            object: Term::Variable(Variable::new("o").unwrap()),
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
            subject: Term::Variable(Variable::new("x").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
            object: Term::Variable(Variable::new("name").unwrap()),
        };

        let p2 = TriplePattern {
            subject: Term::Variable(Variable::new("x").unwrap()),
            predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/knows")),
            object: Term::Variable(Variable::new("y").unwrap()),
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
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                )),
                object: Term::Variable(Variable::new("o").unwrap()),
            },
            // More selective pattern
            TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
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

// Include the optimizer submodule
pub mod optimizer;
