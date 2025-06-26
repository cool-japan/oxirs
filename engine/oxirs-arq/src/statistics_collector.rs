//! Advanced Statistics Collection for Query Optimization
//!
//! This module provides sophisticated statistics collection capabilities
//! for accurate cardinality estimation and query optimization.

use crate::algebra::{Algebra, Iri, Literal, Term, TriplePattern, Variable};
use crate::optimizer::{IndexStatistics, IndexType, Statistics};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

/// Histogram for value distribution analysis
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bucket boundaries (sorted)
    pub boundaries: Vec<String>,
    /// Count in each bucket
    pub frequencies: Vec<usize>,
    /// Total number of values
    pub total_count: usize,
    /// Number of distinct values
    pub distinct_count: usize,
    /// Most frequent values with their counts
    pub most_frequent: Vec<(String, usize)>,
}

impl Histogram {
    /// Create a new histogram with specified number of buckets
    pub fn new(num_buckets: usize) -> Self {
        Self {
            boundaries: Vec::with_capacity(num_buckets),
            frequencies: Vec::with_capacity(num_buckets),
            total_count: 0,
            distinct_count: 0,
            most_frequent: Vec::new(),
        }
    }

    /// Add a value to the histogram
    pub fn add_value(&mut self, value: &str) {
        self.total_count += 1;
        
        // Update bucket counts (simplified for now)
        let bucket = self.find_bucket(value);
        if bucket < self.frequencies.len() {
            self.frequencies[bucket] += 1;
        }
    }

    /// Find the bucket index for a value
    fn find_bucket(&self, value: &str) -> usize {
        self.boundaries.binary_search(&value.to_string())
            .unwrap_or_else(|pos| pos.saturating_sub(1))
    }

    /// Estimate selectivity for a value
    pub fn estimate_selectivity(&self, value: &str) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        // Check if it's a most frequent value
        for (freq_val, count) in &self.most_frequent {
            if freq_val == value {
                return *count as f64 / self.total_count as f64;
            }
        }

        // Otherwise use bucket estimation
        let bucket = self.find_bucket(value);
        if bucket < self.frequencies.len() {
            let bucket_freq = self.frequencies[bucket] as f64;
            let bucket_distinct = (self.distinct_count / self.boundaries.len()).max(1) as f64;
            bucket_freq / bucket_distinct / self.total_count as f64
        } else {
            1.0 / self.distinct_count.max(1) as f64
        }
    }

    /// Estimate range selectivity
    pub fn estimate_range_selectivity(&self, start: Option<&str>, end: Option<&str>) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let start_bucket = start.map(|s| self.find_bucket(s)).unwrap_or(0);
        let end_bucket = end.map(|e| self.find_bucket(e)).unwrap_or(self.frequencies.len());

        let count: usize = self.frequencies[start_bucket..=end_bucket.min(self.frequencies.len() - 1)]
            .iter()
            .sum();

        count as f64 / self.total_count as f64
    }
}

/// Correlation statistics between attributes
#[derive(Debug, Clone)]
pub struct CorrelationStats {
    /// Correlation coefficient between two attributes
    pub correlation: f64,
    /// Joint histogram for the attributes
    pub joint_distribution: HashMap<(String, String), usize>,
    /// Functional dependency strength (0.0 to 1.0)
    pub functional_dependency: f64,
}

/// Advanced statistics collector
pub struct StatisticsCollector {
    /// Current statistics
    stats: Statistics,
    /// Value distribution histograms per predicate
    histograms: HashMap<String, Histogram>,
    /// Correlation statistics between predicates
    correlations: HashMap<(String, String), CorrelationStats>,
    /// Sampling configuration
    sample_rate: f64,
    /// Maximum histogram buckets
    max_histogram_buckets: usize,
    /// Collection start time
    start_time: Instant,
}

impl StatisticsCollector {
    /// Create a new statistics collector
    pub fn new() -> Self {
        Self {
            stats: Statistics::new(),
            histograms: HashMap::new(),
            correlations: HashMap::new(),
            sample_rate: 0.1, // Sample 10% by default
            max_histogram_buckets: 100,
            start_time: Instant::now(),
        }
    }

    /// Configure sampling rate
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Configure histogram buckets
    pub fn with_histogram_buckets(mut self, buckets: usize) -> Self {
        self.max_histogram_buckets = buckets.max(10);
        self
    }

    /// Collect statistics from triple patterns
    pub fn collect_from_patterns(&mut self, patterns: &[TriplePattern]) -> Result<()> {
        // Count pattern occurrences
        for pattern in patterns {
            let pattern_key = format!("{}", pattern);
            *self.stats.pattern_cardinality.entry(pattern_key).or_insert(0) += 1;

            // Update term statistics
            self.update_term_statistics(&pattern.subject, TermPosition::Subject)?;
            self.update_term_statistics(&pattern.predicate, TermPosition::Predicate)?;
            self.update_term_statistics(&pattern.object, TermPosition::Object)?;

            // Update join variable statistics
            self.update_variable_statistics(pattern)?;
        }

        // Compute derived statistics
        self.compute_derived_statistics()?;

        Ok(())
    }

    /// Update statistics for a specific term
    fn update_term_statistics(&mut self, term: &Term, position: TermPosition) -> Result<()> {
        match term {
            Term::Iri(iri) => {
                match position {
                    TermPosition::Subject => {
                        *self.stats.subject_cardinality.entry(iri.0.clone()).or_insert(0) += 1;
                    }
                    TermPosition::Predicate => {
                        *self.stats.predicate_frequency.entry(iri.0.clone()).or_insert(0) += 1;
                        
                        // Create histogram for this predicate if needed
                        if !self.histograms.contains_key(&iri.0) {
                            self.histograms.insert(
                                iri.0.clone(),
                                Histogram::new(self.max_histogram_buckets),
                            );
                        }
                    }
                    TermPosition::Object => {
                        *self.stats.object_cardinality.entry(iri.0.clone()).or_insert(0) += 1;
                    }
                }
            }
            Term::Literal(lit) => {
                if position == TermPosition::Object {
                    // Update histogram for the associated predicate
                    // (In a real implementation, we'd track the predicate-object relationship)
                    for histogram in self.histograms.values_mut() {
                        if rand::random::<f64>() < self.sample_rate {
                            histogram.add_value(&lit.value);
                        }
                    }
                }
            }
            Term::Variable(var) => {
                // Track variable selectivity
                let current = self.stats.variable_selectivity.get(var).copied().unwrap_or(1.0);
                self.stats.variable_selectivity.insert(var.clone(), current * 0.9);
            }
            _ => {}
        }
        Ok(())
    }

    /// Update variable statistics
    fn update_variable_statistics(&mut self, pattern: &TriplePattern) -> Result<()> {
        let vars = self.extract_variables(pattern);
        
        for var in vars {
            // Update variable occurrence count
            let current = self.stats.variable_selectivity.get(&var).copied().unwrap_or(0.5);
            
            // Adjust selectivity based on position
            let position_factor = match (&pattern.subject, &pattern.predicate, &pattern.object) {
                (Term::Variable(v), _, _) if v == &var => 0.8, // Subject position
                (_, Term::Variable(v), _) if v == &var => 0.1, // Predicate position (rare)
                (_, _, Term::Variable(v)) if v == &var => 0.9, // Object position
                _ => 1.0,
            };
            
            self.stats.variable_selectivity.insert(
                var,
                (current * position_factor).clamp(0.001, 1.0),
            );
        }
        
        Ok(())
    }

    /// Extract variables from a pattern
    fn extract_variables(&self, pattern: &TriplePattern) -> HashSet<Variable> {
        let mut vars = HashSet::new();
        
        if let Term::Variable(v) = &pattern.subject {
            vars.insert(v.clone());
        }
        if let Term::Variable(v) = &pattern.predicate {
            vars.insert(v.clone());
        }
        if let Term::Variable(v) = &pattern.object {
            vars.insert(v.clone());
        }
        
        vars
    }

    /// Compute derived statistics
    fn compute_derived_statistics(&mut self) -> Result<()> {
        // Compute join selectivities
        self.compute_join_selectivities()?;
        
        // Update index statistics
        self.update_index_statistics()?;
        
        // Compute correlations
        self.compute_correlations()?;
        
        Ok(())
    }

    /// Compute join selectivities between patterns
    fn compute_join_selectivities(&mut self) -> Result<()> {
        // Analyze co-occurrence of predicates to estimate join selectivity
        let predicates: Vec<_> = self.stats.predicate_frequency.keys().cloned().collect();
        
        for i in 0..predicates.len() {
            for j in i + 1..predicates.len() {
                let pred1 = &predicates[i];
                let pred2 = &predicates[j];
                
                // Estimate selectivity based on frequencies
                let freq1 = self.stats.predicate_frequency.get(pred1).copied().unwrap_or(1) as f64;
                let freq2 = self.stats.predicate_frequency.get(pred2).copied().unwrap_or(1) as f64;
                let total = self.stats.predicate_frequency.values().sum::<usize>() as f64;
                
                let selectivity = (freq1.min(freq2) / total).sqrt();
                
                let join_key = format!("{}_{}", pred1, pred2);
                self.stats.join_selectivity.insert(join_key, selectivity);
            }
        }
        
        Ok(())
    }

    /// Update index statistics based on collected data
    fn update_index_statistics(&mut self) -> Result<()> {
        // Determine which indexes would be most beneficial
        let total_patterns = self.stats.pattern_cardinality.values().sum::<usize>() as f64;
        
        // Subject-Predicate index benefit
        let sp_benefit = self.stats.predicate_frequency.values()
            .filter(|&&freq| freq > 10)
            .count() as f64 / self.stats.predicate_frequency.len().max(1) as f64;
        
        self.stats.index_stats.index_selectivity.insert(
            IndexType::SubjectPredicate,
            sp_benefit,
        );
        
        // Predicate-Object index benefit
        let po_benefit = self.histograms.values()
            .map(|h| h.distinct_count as f64 / h.total_count.max(1) as f64)
            .sum::<f64>() / self.histograms.len().max(1) as f64;
        
        self.stats.index_stats.index_selectivity.insert(
            IndexType::PredicateObject,
            po_benefit,
        );
        
        // Update access costs based on selectivity
        for (index_type, selectivity) in &self.stats.index_stats.index_selectivity {
            let cost = (1.0 - selectivity) * 10.0 + 1.0;
            self.stats.index_stats.index_access_cost.insert(
                index_type.clone(),
                cost,
            );
        }
        
        Ok(())
    }

    /// Compute correlations between predicates
    fn compute_correlations(&mut self) -> Result<()> {
        // Simplified correlation computation
        // In practice, this would analyze co-occurrence patterns
        
        let predicates: Vec<_> = self.stats.predicate_frequency.keys().cloned().collect();
        
        for i in 0..predicates.len().min(20) {
            for j in i + 1..predicates.len().min(20) {
                let correlation = CorrelationStats {
                    correlation: rand::random::<f64>() * 0.5, // Placeholder
                    joint_distribution: HashMap::new(),
                    functional_dependency: 0.0,
                };
                
                self.correlations.insert(
                    (predicates[i].clone(), predicates[j].clone()),
                    correlation,
                );
            }
        }
        
        Ok(())
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> &Statistics {
        &self.stats
    }

    /// Get statistics with ownership transfer
    pub fn into_statistics(self) -> Statistics {
        self.stats
    }

    /// Get histogram for a predicate
    pub fn get_histogram(&self, predicate: &str) -> Option<&Histogram> {
        self.histograms.get(predicate)
    }

    /// Estimate cardinality for a triple pattern
    pub fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> usize {
        let pattern_key = format!("{}", pattern);
        
        // Use exact count if available
        if let Some(&count) = self.stats.pattern_cardinality.get(&pattern_key) {
            return count;
        }
        
        // Otherwise estimate based on term selectivities
        let base_cardinality = 1_000_000; // Default assumption
        let mut selectivity = 1.0;
        
        // Apply subject selectivity
        match &pattern.subject {
            Term::Iri(iri) => {
                if let Some(&count) = self.stats.subject_cardinality.get(&iri.0) {
                    selectivity *= count as f64 / base_cardinality as f64;
                } else {
                    selectivity *= 0.001;
                }
            }
            Term::Variable(_) => selectivity *= 0.5,
            Term::Literal(_) => selectivity *= 0.0001,
            _ => selectivity *= 0.01,
        }
        
        // Apply predicate selectivity
        match &pattern.predicate {
            Term::Iri(iri) => {
                if let Some(&freq) = self.stats.predicate_frequency.get(&iri.0) {
                    selectivity *= freq as f64 / base_cardinality as f64;
                } else {
                    selectivity *= 0.01;
                }
            }
            Term::Variable(_) => selectivity *= 0.1, // Variable predicates are rare
            _ => selectivity *= 0.001,
        }
        
        // Apply object selectivity
        match &pattern.object {
            Term::Iri(iri) => {
                if let Some(&count) = self.stats.object_cardinality.get(&iri.0) {
                    selectivity *= count as f64 / base_cardinality as f64;
                } else {
                    selectivity *= 0.001;
                }
            }
            Term::Variable(_) => selectivity *= 0.5,
            Term::Literal(_) => selectivity *= 0.0001,
            _ => selectivity *= 0.01,
        }
        
        (base_cardinality as f64 * selectivity).ceil() as usize
    }

    /// Estimate join cardinality
    pub fn estimate_join_cardinality(
        &self,
        left_cardinality: usize,
        right_cardinality: usize,
        join_vars: &[Variable],
    ) -> usize {
        if join_vars.is_empty() {
            // Cartesian product
            return left_cardinality * right_cardinality;
        }
        
        // Use variable selectivity to estimate join reduction
        let avg_selectivity: f64 = join_vars
            .iter()
            .map(|var| self.stats.variable_selectivity.get(var).copied().unwrap_or(0.1))
            .sum::<f64>() / join_vars.len() as f64;
        
        let join_factor = (avg_selectivity * join_vars.len() as f64).min(1.0);
        
        ((left_cardinality * right_cardinality) as f64 * join_factor).ceil() as usize
    }

    /// Get collection duration
    pub fn collection_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Term position in a triple
#[derive(Debug, Clone, Copy, PartialEq)]
enum TermPosition {
    Subject,
    Predicate,
    Object,
}

/// Dynamic statistics updater that learns from query execution
pub struct DynamicStatisticsUpdater {
    /// Historical execution records
    execution_history: Vec<QueryExecutionRecord>,
    /// Learning rate for statistics updates
    learning_rate: f64,
    /// Maximum history size
    max_history: usize,
}

/// Record of query execution for learning
#[derive(Debug, Clone)]
pub struct QueryExecutionRecord {
    /// Query algebra
    pub algebra: Algebra,
    /// Estimated cardinality
    pub estimated_cardinality: usize,
    /// Actual cardinality
    pub actual_cardinality: usize,
    /// Execution time
    pub execution_time: Duration,
    /// Timestamp
    pub timestamp: Instant,
}

impl DynamicStatisticsUpdater {
    /// Create a new dynamic updater
    pub fn new() -> Self {
        Self {
            execution_history: Vec::new(),
            learning_rate: 0.1,
            max_history: 1000,
        }
    }

    /// Update statistics based on execution feedback
    pub fn update_from_execution(
        &mut self,
        record: QueryExecutionRecord,
        stats: &mut Statistics,
    ) -> Result<()> {
        // Calculate estimation error
        let error_ratio = if record.estimated_cardinality > 0 {
            record.actual_cardinality as f64 / record.estimated_cardinality as f64
        } else {
            1.0
        };
        
        // Update pattern cardinalities
        self.update_pattern_statistics(&record.algebra, record.actual_cardinality, stats)?;
        
        // Update variable selectivities
        self.update_variable_selectivities(&record.algebra, error_ratio, stats)?;
        
        // Store execution record
        self.execution_history.push(record);
        if self.execution_history.len() > self.max_history {
            self.execution_history.remove(0);
        }
        
        Ok(())
    }

    /// Update pattern statistics based on actual results
    fn update_pattern_statistics(
        &self,
        algebra: &Algebra,
        actual_cardinality: usize,
        stats: &mut Statistics,
    ) -> Result<()> {
        match algebra {
            Algebra::Bgp(patterns) => {
                for pattern in patterns {
                    let pattern_key = format!("{}", pattern);
                    let current = stats.pattern_cardinality.get(&pattern_key).copied().unwrap_or(1000);
                    
                    // Exponential moving average update
                    let updated = (current as f64 * (1.0 - self.learning_rate)
                        + actual_cardinality as f64 * self.learning_rate / patterns.len() as f64)
                        .round() as usize;
                    
                    stats.pattern_cardinality.insert(pattern_key, updated);
                }
            }
            _ => {
                // Recursively process nested algebra
            }
        }
        
        Ok(())
    }

    /// Update variable selectivities based on estimation errors
    fn update_variable_selectivities(
        &self,
        algebra: &Algebra,
        error_ratio: f64,
        stats: &mut Statistics,
    ) -> Result<()> {
        let variables = algebra.variables();
        
        for var in variables {
            let current = stats.variable_selectivity.get(&var).copied().unwrap_or(0.1);
            
            // Adjust selectivity based on error
            let adjustment = if error_ratio > 1.0 {
                // We underestimated, increase selectivity
                1.0 + self.learning_rate * (error_ratio - 1.0).min(2.0)
            } else {
                // We overestimated, decrease selectivity
                1.0 - self.learning_rate * (1.0 - error_ratio).min(0.5)
            };
            
            let updated = (current * adjustment).clamp(0.001, 1.0);
            stats.variable_selectivity.insert(var, updated);
        }
        
        Ok(())
    }

    /// Analyze historical patterns for optimization opportunities
    pub fn analyze_patterns(&self) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();
        
        // Analyze frequently mis-estimated queries
        let mis_estimates: Vec<_> = self.execution_history
            .iter()
            .filter(|r| {
                let ratio = r.actual_cardinality as f64 / r.estimated_cardinality.max(1) as f64;
                ratio < 0.1 || ratio > 10.0
            })
            .collect();
        
        if mis_estimates.len() > 10 {
            hints.push(OptimizationHint::CollectMoreStatistics);
        }
        
        // Analyze slow queries
        let slow_queries: Vec<_> = self.execution_history
            .iter()
            .filter(|r| r.execution_time > Duration::from_secs(1))
            .collect();
        
        if slow_queries.len() > 5 {
            hints.push(OptimizationHint::ConsiderIndexing);
        }
        
        hints
    }
}

/// Optimization hints from statistics analysis
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    CollectMoreStatistics,
    ConsiderIndexing,
    UpdateHistograms,
    AnalyzeCorrelations,
}

// Helper function for tests
pub use rand;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_operations() {
        let mut histogram = Histogram::new(10);
        
        // Add some values
        for i in 0..100 {
            histogram.add_value(&format!("value_{}", i));
            histogram.total_count = i + 1;
        }
        histogram.distinct_count = 100;
        
        // Test selectivity estimation
        let selectivity = histogram.estimate_selectivity("value_50");
        assert!(selectivity > 0.0 && selectivity < 1.0);
    }

    #[test]
    fn test_statistics_collection() {
        let mut collector = StatisticsCollector::new();
        
        let patterns = vec![
            TriplePattern {
                subject: Term::Variable("s".to_string()),
                predicate: Term::Iri(Iri("http://example.org/type".to_string())),
                object: Term::Variable("o".to_string()),
            },
            TriplePattern {
                subject: Term::Variable("s".to_string()),
                predicate: Term::Iri(Iri("http://example.org/name".to_string())),
                object: Term::Literal(Literal {
                    value: "Test".to_string(),
                    language: None,
                    datatype: None,
                }),
            },
        ];
        
        collector.collect_from_patterns(&patterns).unwrap();
        
        let stats = collector.get_statistics();
        assert!(stats.predicate_frequency.contains_key("http://example.org/type"));
        assert!(stats.predicate_frequency.contains_key("http://example.org/name"));
        assert!(stats.variable_selectivity.contains_key("s"));
    }

    #[test]
    fn test_cardinality_estimation() {
        let collector = StatisticsCollector::new();
        
        let pattern = TriplePattern {
            subject: Term::Iri(Iri("http://example.org/subject".to_string())),
            predicate: Term::Iri(Iri("http://example.org/predicate".to_string())),
            object: Term::Variable("o".to_string()),
        };
        
        let cardinality = collector.estimate_pattern_cardinality(&pattern);
        assert!(cardinality > 0);
    }

    #[test]
    fn test_dynamic_updates() {
        let mut updater = DynamicStatisticsUpdater::new();
        let mut stats = Statistics::new();
        
        let record = QueryExecutionRecord {
            algebra: Algebra::Bgp(vec![
                TriplePattern {
                    subject: Term::Variable("s".to_string()),
                    predicate: Term::Iri(Iri("http://example.org/type".to_string())),
                    object: Term::Variable("o".to_string()),
                },
            ]),
            estimated_cardinality: 1000,
            actual_cardinality: 5000,
            execution_time: Duration::from_millis(100),
            timestamp: Instant::now(),
        };
        
        updater.update_from_execution(record, &mut stats).unwrap();
        
        // Check that statistics were updated
        assert!(!stats.pattern_cardinality.is_empty());
        assert!(!stats.variable_selectivity.is_empty());
    }
}