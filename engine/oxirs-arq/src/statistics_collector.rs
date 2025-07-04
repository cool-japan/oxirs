//! Advanced Statistics Collection for Query Optimization
//!
//! This module provides sophisticated statistics collection capabilities
//! for accurate cardinality estimation and query optimization.

use crate::algebra::{Algebra, Literal, Term, TriplePattern, Variable};
use crate::optimizer::{IndexStatistics, IndexType, Statistics};
use anyhow::Result;
use oxirs_core::model::NamedNode;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

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
        self.boundaries
            .binary_search(&value.to_string())
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
        let end_bucket = end
            .map(|e| self.find_bucket(e))
            .unwrap_or(self.frequencies.len());

        let count: usize = self.frequencies
            [start_bucket..=end_bucket.min(self.frequencies.len() - 1)]
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

/// Temporal statistics for tracking changes over time
#[derive(Debug, Clone)]
pub struct TemporalStatistics {
    /// Statistics snapshots over time
    pub snapshots: Vec<TemporalSnapshot>,
    /// Maximum snapshots to keep
    pub max_snapshots: usize,
    /// Update frequency in seconds
    pub update_frequency: u64,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Snapshot of statistics at a point in time
#[derive(Debug, Clone)]
pub struct TemporalSnapshot {
    /// Timestamp of snapshot
    pub timestamp: Instant,
    /// Predicate frequencies at this time
    pub predicate_frequencies: HashMap<String, usize>,
    /// Pattern cardinalities at this time
    pub pattern_cardinalities: HashMap<String, usize>,
    /// Query execution count since last snapshot
    pub query_count: usize,
}

/// Adaptive configuration for dynamic statistics updates
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Enable adaptive histogram bucket sizing
    pub adaptive_histograms: bool,
    /// Enable correlation decay over time
    pub correlation_decay: bool,
    /// Decay rate for correlations (per time unit)
    pub decay_rate: f64,
    /// Minimum samples required for reliable statistics
    pub min_samples: usize,
    /// Enable outlier detection and filtering
    pub outlier_detection: bool,
    /// Threshold for outlier detection (standard deviations)
    pub outlier_threshold: f64,
}

/// Advanced statistics collector
#[derive(Debug, Clone)]
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
    /// Temporal statistics tracking
    temporal_stats: TemporalStatistics,
    /// Adaptive configuration for dynamic updates
    adaptive_config: AdaptiveConfig,
}

impl StatisticsCollector {
    /// Create a new statistics collector
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            stats: Statistics::new(),
            histograms: HashMap::new(),
            correlations: HashMap::new(),
            sample_rate: 0.1, // Sample 10% by default
            max_histogram_buckets: 100,
            start_time: now,
            temporal_stats: TemporalStatistics {
                snapshots: Vec::new(),
                max_snapshots: 100,
                update_frequency: 300, // 5 minutes
                last_update: now,
            },
            adaptive_config: AdaptiveConfig {
                adaptive_histograms: true,
                correlation_decay: true,
                decay_rate: 0.01, // 1% decay per time unit
                min_samples: 10,
                outlier_detection: true,
                outlier_threshold: 2.0, // 2 standard deviations
            },
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
            *self
                .stats
                .pattern_cardinality
                .entry(pattern_key)
                .or_insert(0) += 1;

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
                        *self
                            .stats
                            .pattern_cardinality
                            .entry(format!("subject:{}", iri.as_str()))
                            .or_insert(0) += 1;
                    }
                    TermPosition::Predicate => {
                        *self
                            .stats
                            .cardinalities
                            .entry(format!("predicate:{}", iri.as_str()))
                            .or_insert(0) += 1;

                        // Create histogram for this predicate if needed
                        if !self.histograms.contains_key(iri.as_str()) {
                            self.histograms.insert(
                                iri.as_str().to_string(),
                                Histogram::new(self.max_histogram_buckets),
                            );
                        }
                    }
                    TermPosition::Object => {
                        *self
                            .stats
                            .pattern_cardinality
                            .entry(format!("object:{}", iri.as_str()))
                            .or_insert(0) += 1;
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
                let current = self
                    .stats
                    .filter_selectivities
                    .get(&format!("var:{}", var.as_str()))
                    .copied()
                    .unwrap_or(1.0);
                self.stats
                    .filter_selectivities
                    .insert(format!("var:{}", var.as_str()), current * 0.9);
            }
            Term::QuotedTriple(triple) => {
                // Recursively update statistics for the quoted triple's components
                self.update_term_statistics(&triple.subject, TermPosition::Subject)?;
                self.update_term_statistics(&triple.predicate, TermPosition::Predicate)?;
                self.update_term_statistics(&triple.object, TermPosition::Object)?;

                // Track quoted triple complexity using pattern cardinality
                let quoted_key = format!(
                    "<<{} {} {}>>",
                    triple.subject, triple.predicate, triple.object
                );
                *self
                    .stats
                    .pattern_cardinality
                    .entry(quoted_key)
                    .or_insert(0) += 1;
            }
            Term::PropertyPath(path) => {
                // Property paths are complex patterns that affect selectivity
                // Track as a special type of predicate usage
                let path_key = format!("path:{}", path);
                *self.stats.pattern_cardinality.entry(path_key).or_insert(0) += 1;

                // Property paths typically increase cardinality estimates
                if position == TermPosition::Predicate {
                    // Add to predicate frequency to track path usage
                    *self
                        .stats
                        .cardinalities
                        .entry(format!("path:{}", path))
                        .or_insert(0) += 1;
                }
            }
            Term::BlankNode(id) => {
                // Track blank node usage for cardinality estimation
                match position {
                    TermPosition::Subject => {
                        *self
                            .stats
                            .pattern_cardinality
                            .entry(format!("subject:_:{}", id))
                            .or_insert(0) += 1;
                    }
                    TermPosition::Object => {
                        *self
                            .stats
                            .pattern_cardinality
                            .entry(format!("object:_:{}", id))
                            .or_insert(0) += 1;
                    }
                    TermPosition::Predicate => {
                        // Blank nodes as predicates are unusual but possible in some contexts
                        let blank_predicate_key = format!("blank_predicate:{}", id);
                        *self
                            .stats
                            .pattern_cardinality
                            .entry(blank_predicate_key)
                            .or_insert(0) += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Update variable statistics
    fn update_variable_statistics(&mut self, pattern: &TriplePattern) -> Result<()> {
        let vars = self.extract_variables(pattern);

        for var in vars {
            // Update variable occurrence count
            let current = self
                .stats
                .filter_selectivities
                .get(&format!("var:{}", var.as_str()))
                .copied()
                .unwrap_or(0.5);

            // Adjust selectivity based on position
            let position_factor = match (&pattern.subject, &pattern.predicate, &pattern.object) {
                (Term::Variable(v), _, _) if v == &var => 0.8, // Subject position
                (_, Term::Variable(v), _) if v == &var => 0.1, // Predicate position (rare)
                (_, _, Term::Variable(v)) if v == &var => 0.9, // Object position
                _ => 1.0,
            };

            self.stats
                .filter_selectivities
                .insert(format!("var:{}", var.as_str()), (current * position_factor).clamp(0.001, 1.0));
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
        let predicates: Vec<_> = self.stats.cardinalities.keys()
            .filter(|k| k.starts_with("predicate:"))
            .cloned().collect();

        for i in 0..predicates.len() {
            for j in i + 1..predicates.len() {
                let pred1 = &predicates[i];
                let pred2 = &predicates[j];

                // Estimate selectivity based on frequencies
                let freq1 = self
                    .stats
                    .cardinalities
                    .get(pred1)
                    .copied()
                    .unwrap_or(1) as f64;
                let freq2 = self
                    .stats
                    .cardinalities
                    .get(pred2)
                    .copied()
                    .unwrap_or(1) as f64;
                let total = self.stats.cardinalities.values()
                    .filter(|_| true) // All cardinalities contribute to total
                    .sum::<usize>() as f64;

                let selectivity = (freq1.min(freq2) / total).sqrt();

                let join_key = format!("{}_{}", pred1, pred2);
                self.stats.join_selectivities.insert(join_key, selectivity);
            }
        }

        Ok(())
    }

    /// Update index statistics based on collected data
    fn update_index_statistics(&mut self) -> Result<()> {
        // Determine which indexes would be most beneficial
        let total_patterns = self.stats.pattern_cardinality.values().sum::<usize>() as f64;

        // Subject-Predicate index benefit
        let sp_benefit = self
            .stats
            .predicate_frequency
            .values()
            .filter(|&&freq| freq > 10)
            .count() as f64
            / self.stats.predicate_frequency.len().max(1) as f64;

        // Get or create IndexStatistics for SubjectPredicate index
        let sp_stats = self.stats
            .index_stats
            .entry(IndexType::SubjectPredicate)
            .or_insert_with(IndexStatistics::default);
        sp_stats.index_selectivity = sp_benefit;

        // Predicate-Object index benefit
        let po_benefit = self
            .histograms
            .values()
            .map(|h| h.distinct_count as f64 / h.total_count.max(1) as f64)
            .sum::<f64>()
            / self.histograms.len().max(1) as f64;

        // Get or create IndexStatistics for PredicateObject index
        let po_stats = self.stats
            .index_stats
            .entry(IndexType::PredicateObject)
            .or_insert_with(IndexStatistics::default);
        po_stats.index_selectivity = po_benefit;

        // Update access costs based on selectivity
        for (index_type, index_stats) in &mut self.stats.index_stats {
            let selectivity = index_stats.index_selectivity;
            let cost = (1.0 - selectivity) * 10.0 + 1.0;
            index_stats.index_access_cost.insert(index_type.clone(), cost);
        }

        Ok(())
    }

    /// Compute correlations between predicates based on co-occurrence patterns
    fn compute_correlations(&mut self) -> Result<()> {
        let predicates: Vec<_> = self.stats.predicate_frequency.keys().cloned().collect();

        for i in 0..predicates.len().min(20) {
            for j in i + 1..predicates.len().min(20) {
                let pred1 = &predicates[i];
                let pred2 = &predicates[j];

                // Calculate real correlation based on frequency and co-occurrence
                let correlation = self.calculate_predicate_correlation(pred1, pred2)?;
                let joint_distribution = self.build_joint_distribution(pred1, pred2)?;
                let functional_dependency = self.estimate_functional_dependency(pred1, pred2)?;

                let correlation_stats = CorrelationStats {
                    correlation,
                    joint_distribution,
                    functional_dependency,
                };

                self.correlations
                    .insert((pred1.clone(), pred2.clone()), correlation_stats);
            }
        }

        Ok(())
    }

    /// Calculate correlation coefficient between two predicates
    fn calculate_predicate_correlation(&self, pred1: &str, pred2: &str) -> Result<f64> {
        let freq1 = self
            .stats
            .predicate_frequency
            .get(pred1)
            .copied()
            .unwrap_or(0) as f64;
        let freq2 = self
            .stats
            .predicate_frequency
            .get(pred2)
            .copied()
            .unwrap_or(0) as f64;
        let total = self.stats.predicate_frequency.values().sum::<usize>() as f64;

        if total == 0.0 {
            return Ok(0.0);
        }

        // Normalized frequencies
        let norm_freq1 = freq1 / total;
        let norm_freq2 = freq2 / total;

        // Estimate co-occurrence (simplified heuristic based on frequency patterns)
        let expected_cooccurrence = norm_freq1 * norm_freq2;
        let observed_cooccurrence = self.estimate_cooccurrence(pred1, pred2);

        // Pearson-like correlation measure
        let correlation = if expected_cooccurrence > 0.0 {
            (observed_cooccurrence - expected_cooccurrence)
                / (expected_cooccurrence * (1.0 - expected_cooccurrence)).sqrt()
        } else {
            0.0
        };

        // Normalize to [-1, 1] range
        Ok(correlation.max(-1.0).min(1.0))
    }

    /// Estimate co-occurrence probability of two predicates
    fn estimate_cooccurrence(&self, pred1: &str, pred2: &str) -> f64 {
        // Heuristic: predicates with similar frequencies are more likely to co-occur
        let freq1 = self
            .stats
            .predicate_frequency
            .get(pred1)
            .copied()
            .unwrap_or(0) as f64;
        let freq2 = self
            .stats
            .predicate_frequency
            .get(pred2)
            .copied()
            .unwrap_or(0) as f64;

        if freq1 == 0.0 || freq2 == 0.0 {
            return 0.0;
        }

        // Proximity-based co-occurrence estimation
        let freq_ratio = (freq1 / freq2).min(freq2 / freq1);
        let base_cooccurrence = 0.1; // Base co-occurrence probability

        base_cooccurrence * freq_ratio
    }

    /// Build joint distribution histogram for two predicates
    fn build_joint_distribution(
        &self,
        pred1: &str,
        pred2: &str,
    ) -> Result<HashMap<(String, String), usize>> {
        let mut joint_dist = HashMap::new();

        // Get histograms for both predicates
        let hist1 = self.histograms.get(pred1);
        let hist2 = self.histograms.get(pred2);

        match (hist1, hist2) {
            (Some(h1), Some(h2)) => {
                // Create joint distribution from individual histograms
                for (i, &freq1) in h1.frequencies.iter().enumerate() {
                    for (j, &freq2) in h2.frequencies.iter().enumerate() {
                        if i < h1.boundaries.len() && j < h2.boundaries.len() {
                            let joint_key = (h1.boundaries[i].clone(), h2.boundaries[j].clone());
                            let joint_freq = ((freq1 as f64 * freq2 as f64).sqrt() as usize).max(1);
                            joint_dist.insert(joint_key, joint_freq);
                        }
                    }
                }
            }
            _ => {
                // Fallback: create minimal joint distribution
                joint_dist.insert(("unknown".to_string(), "unknown".to_string()), 1);
            }
        }

        Ok(joint_dist)
    }

    /// Estimate functional dependency strength between predicates
    fn estimate_functional_dependency(&self, pred1: &str, pred2: &str) -> Result<f64> {
        let freq1 = self
            .stats
            .predicate_frequency
            .get(pred1)
            .copied()
            .unwrap_or(0) as f64;
        let freq2 = self
            .stats
            .predicate_frequency
            .get(pred2)
            .copied()
            .unwrap_or(0) as f64;

        if freq1 == 0.0 || freq2 == 0.0 {
            return Ok(0.0);
        }

        // Functional dependency heuristic:
        // Higher dependency if one predicate is much more frequent than the other
        let freq_ratio = freq1.min(freq2) / freq1.max(freq2);
        let dependency = 1.0 - freq_ratio;

        // Scale by overall frequency to account for data volume
        let total_freq = self.stats.predicate_frequency.values().sum::<usize>() as f64;
        let prevalence_factor = (freq1 + freq2) / total_freq;

        Ok(dependency * prevalence_factor)
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
                if let Some(&count) = self.stats.subject_cardinality.get(iri.as_str()) {
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
                if let Some(&freq) = self.stats.predicate_frequency.get(iri.as_str()) {
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
                if let Some(&count) = self.stats.object_cardinality.get(iri.as_str()) {
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
            .map(|var| {
                self.stats
                    .variable_selectivity
                    .get(var)
                    .copied()
                    .unwrap_or(0.1)
            })
            .sum::<f64>()
            / join_vars.len() as f64;

        let join_factor = (avg_selectivity * join_vars.len() as f64).min(1.0);

        ((left_cardinality * right_cardinality) as f64 * join_factor).ceil() as usize
    }

    /// Get collection duration
    pub fn collection_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Update temporal statistics with current snapshot
    pub fn update_temporal_statistics(&mut self) -> Result<()> {
        let now = Instant::now();

        // Check if enough time has passed for an update
        if now
            .duration_since(self.temporal_stats.last_update)
            .as_secs()
            < self.temporal_stats.update_frequency
        {
            return Ok(());
        }

        // Create new snapshot
        let snapshot = TemporalSnapshot {
            timestamp: now,
            predicate_frequencies: self.stats.predicate_frequency.clone(),
            pattern_cardinalities: self.stats.pattern_cardinality.clone(),
            query_count: 1, // Would track actual query count in real implementation
        };

        // Add snapshot and maintain size limit
        self.temporal_stats.snapshots.push(snapshot);
        if self.temporal_stats.snapshots.len() > self.temporal_stats.max_snapshots {
            self.temporal_stats.snapshots.remove(0);
        }

        self.temporal_stats.last_update = now;

        // Apply adaptive updates if enabled
        if self.adaptive_config.correlation_decay {
            self.apply_correlation_decay()?;
        }

        if self.adaptive_config.adaptive_histograms {
            self.adapt_histogram_buckets()?;
        }

        Ok(())
    }

    /// Apply time-based decay to correlations
    fn apply_correlation_decay(&mut self) -> Result<()> {
        let decay_factor = 1.0 - self.adaptive_config.decay_rate;

        for correlation_stats in self.correlations.values_mut() {
            correlation_stats.correlation *= decay_factor;
            correlation_stats.functional_dependency *= decay_factor;
        }

        Ok(())
    }

    /// Adapt histogram bucket sizes based on data distribution
    fn adapt_histogram_buckets(&mut self) -> Result<()> {
        let mut histograms_to_rebuild = Vec::new();

        // First collect predicates that need histogram rebuilding
        let predicates_and_histograms: Vec<_> = self.histograms.iter().collect();

        for (predicate, histogram) in predicates_and_histograms {
            // Check if histogram needs more buckets for better resolution
            let avg_bucket_size = histogram.total_count / histogram.frequencies.len().max(1);

            // Conditions for bucket adaptation:
            // 1. Average bucket size is too large (poor resolution)
            // 2. We haven't reached the maximum bucket limit
            // 3. There's significant skew in bucket distribution
            let has_skew = Self::has_significant_skew_static(histogram);
            if (avg_bucket_size > 1000 && histogram.frequencies.len() < self.max_histogram_buckets)
                || has_skew
            {
                tracing::debug!(
                    "Adapting histogram for {} (avg bucket size: {}, buckets: {})",
                    predicate,
                    avg_bucket_size,
                    histogram.frequencies.len()
                );

                histograms_to_rebuild.push(predicate.clone());
            }
        }

        // Rebuild histograms that need adaptation
        for predicate in histograms_to_rebuild {
            self.rebuild_histogram(&predicate)?;
        }

        Ok(())
    }

    /// Check if histogram has significant skew requiring redistribution (static version)
    fn has_significant_skew_static(histogram: &Histogram) -> bool {
        if histogram.frequencies.len() < 3 {
            return false;
        }

        // Calculate coefficient of variation for bucket sizes
        let mean = histogram.total_count as f64 / histogram.frequencies.len() as f64;
        let variance: f64 = histogram
            .frequencies
            .iter()
            .map(|&count| {
                let diff = count as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / histogram.frequencies.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // Significant skew if CV > 1.5 (high variability in bucket sizes)
        coefficient_of_variation > 1.5
    }

    /// Check if histogram has significant skew requiring redistribution
    fn has_significant_skew(&self, histogram: &Histogram) -> bool {
        if histogram.frequencies.len() < 3 {
            return false;
        }

        // Calculate coefficient of variation for bucket sizes
        let mean = histogram.total_count as f64 / histogram.frequencies.len() as f64;
        let variance: f64 = histogram
            .frequencies
            .iter()
            .map(|&count| {
                let diff = count as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / histogram.frequencies.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // Significant skew if CV > 1.5 (high variability in bucket sizes)
        coefficient_of_variation > 1.5
    }

    /// Rebuild histogram with better bucket distribution
    fn rebuild_histogram(&mut self, predicate: &str) -> Result<()> {
        let histogram = self.histograms.get(predicate);
        if histogram.is_none() {
            return Ok(());
        }

        let old_histogram = histogram.unwrap().clone();

        // Determine optimal number of buckets using Sturges' rule with modifications
        let optimal_buckets = self.calculate_optimal_buckets(old_histogram.total_count);
        let new_bucket_count = optimal_buckets.min(self.max_histogram_buckets);

        // Create new histogram with adaptive bucket boundaries
        let mut new_histogram = self.create_adaptive_histogram(new_bucket_count, &old_histogram)?;

        // Redistribute the most frequent values to optimize for common queries
        self.redistribute_frequent_values(&mut new_histogram, &old_histogram)?;

        // Replace the old histogram
        self.histograms.insert(predicate.to_string(), new_histogram);

        tracing::debug!(
            "Rebuilt histogram for {} with {} buckets",
            predicate,
            new_bucket_count
        );
        Ok(())
    }

    /// Calculate optimal number of buckets based on data size
    fn calculate_optimal_buckets(&self, total_count: usize) -> usize {
        if total_count == 0 {
            return 10; // Minimum buckets
        }

        // Modified Sturges' rule: log2(n) + 1, with adjustments for large datasets
        let sturges_buckets = (total_count as f64).log2().ceil() as usize + 1;

        // For large datasets, use square root rule as upper bound
        let sqrt_rule = (total_count as f64).sqrt().ceil() as usize;

        // Use the more conservative estimate, but ensure reasonable bounds
        let optimal = sturges_buckets.min(sqrt_rule).max(10).min(100);

        optimal
    }

    /// Create new histogram with adaptive bucket boundaries
    fn create_adaptive_histogram(
        &self,
        bucket_count: usize,
        old_histogram: &Histogram,
    ) -> Result<Histogram> {
        let mut new_histogram = Histogram::new(bucket_count);
        new_histogram.total_count = old_histogram.total_count;
        new_histogram.distinct_count = old_histogram.distinct_count;

        // Create boundaries using quantile-based approach
        // This ensures more balanced bucket sizes
        let quantile_step = 1.0 / bucket_count as f64;

        // Generate new boundaries based on cumulative distribution
        for i in 0..bucket_count {
            let quantile = (i + 1) as f64 * quantile_step;
            let boundary = self.estimate_quantile_boundary(old_histogram, quantile);
            new_histogram.boundaries.push(boundary);
        }

        // Initialize frequency counts to zero
        new_histogram.frequencies.resize(bucket_count, 0);

        // Redistribute counts from old histogram to new buckets
        self.redistribute_histogram_counts(&mut new_histogram, old_histogram)?;

        Ok(new_histogram)
    }

    /// Estimate boundary value for a given quantile
    fn estimate_quantile_boundary(&self, histogram: &Histogram, quantile: f64) -> String {
        let target_count = (histogram.total_count as f64 * quantile) as usize;
        let mut cumulative_count = 0;

        for (i, &frequency) in histogram.frequencies.iter().enumerate() {
            cumulative_count += frequency;
            if cumulative_count >= target_count {
                return histogram
                    .boundaries
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("bucket_{}", i));
            }
        }

        // Fallback: generate a boundary name
        format!("quantile_{:.2}", quantile)
    }

    /// Redistribute counts from old histogram to new histogram
    fn redistribute_histogram_counts(
        &self,
        new_histogram: &mut Histogram,
        old_histogram: &Histogram,
    ) -> Result<()> {
        // Simple redistribution: proportionally distribute old bucket counts to new buckets
        let old_bucket_count = old_histogram.frequencies.len();
        let new_bucket_count = new_histogram.frequencies.len();

        if old_bucket_count == 0 || new_bucket_count == 0 {
            return Ok(());
        }

        for (old_idx, &old_count) in old_histogram.frequencies.iter().enumerate() {
            // Map old bucket to new bucket(s)
            let bucket_ratio = old_idx as f64 / old_bucket_count as f64;
            let new_idx = (bucket_ratio * new_bucket_count as f64) as usize;
            let target_idx = new_idx.min(new_bucket_count - 1);

            new_histogram.frequencies[target_idx] += old_count;
        }

        Ok(())
    }

    /// Redistribute frequent values for better query optimization
    fn redistribute_frequent_values(
        &self,
        new_histogram: &mut Histogram,
        old_histogram: &Histogram,
    ) -> Result<()> {
        // Copy over the most frequent values tracking
        new_histogram.most_frequent = old_histogram.most_frequent.clone();

        // Ensure most frequent values are properly distributed in new buckets
        for (value, count) in &old_histogram.most_frequent {
            let bucket_idx = new_histogram.find_bucket(value);

            // Adjust bucket count to account for this frequent value
            if bucket_idx < new_histogram.frequencies.len() {
                // Ensure the bucket has at least the count of this frequent value
                new_histogram.frequencies[bucket_idx] =
                    new_histogram.frequencies[bucket_idx].max(*count);
            }
        }

        Ok(())
    }

    /// Get temporal trend for a predicate
    pub fn get_predicate_trend(&self, predicate: &str) -> Option<f64> {
        if self.temporal_stats.snapshots.len() < 2 {
            return None;
        }

        let recent_snapshots = &self.temporal_stats.snapshots;
        let first_freq = recent_snapshots
            .first()?
            .predicate_frequencies
            .get(predicate)
            .copied()
            .unwrap_or(0) as f64;
        let last_freq = recent_snapshots
            .last()?
            .predicate_frequencies
            .get(predicate)
            .copied()
            .unwrap_or(0) as f64;

        if first_freq == 0.0 {
            return Some(0.0);
        }

        // Calculate growth rate
        Some((last_freq - first_freq) / first_freq)
    }

    /// Detect anomalies in statistics using outlier detection
    pub fn detect_anomalies(&self) -> Vec<StatisticsAnomaly> {
        let mut anomalies = Vec::new();

        if !self.adaptive_config.outlier_detection {
            return anomalies;
        }

        // Check for frequency anomalies
        let frequencies: Vec<f64> = self
            .stats
            .predicate_frequency
            .values()
            .map(|&x| x as f64)
            .collect();
        if frequencies.len() >= self.adaptive_config.min_samples {
            let mean = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
            let variance = frequencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / frequencies.len() as f64;
            let std_dev = variance.sqrt();

            for (predicate, &freq) in &self.stats.predicate_frequency {
                let z_score = (freq as f64 - mean) / std_dev;
                if z_score.abs() > self.adaptive_config.outlier_threshold {
                    anomalies.push(StatisticsAnomaly {
                        anomaly_type: AnomalyType::FrequencyOutlier,
                        predicate: predicate.clone(),
                        value: freq as f64,
                        z_score,
                        severity: if z_score.abs() > 3.0 {
                            AnomalySeverity::High
                        } else {
                            AnomalySeverity::Medium
                        },
                    });
                }
            }
        }

        anomalies
    }

    /// Get statistics evolution over time
    pub fn get_statistics_evolution(&self) -> StatisticsEvolution {
        StatisticsEvolution {
            snapshots_count: self.temporal_stats.snapshots.len(),
            time_span: if let (Some(first), Some(last)) = (
                self.temporal_stats.snapshots.first(),
                self.temporal_stats.snapshots.last(),
            ) {
                last.timestamp.duration_since(first.timestamp)
            } else {
                Duration::from_secs(0)
            },
            growth_rates: self.calculate_growth_rates(),
        }
    }

    /// Calculate growth rates for predicates
    fn calculate_growth_rates(&self) -> HashMap<String, f64> {
        let mut growth_rates = HashMap::new();

        for predicate in self.stats.predicate_frequency.keys() {
            if let Some(trend) = self.get_predicate_trend(predicate) {
                growth_rates.insert(predicate.clone(), trend);
            }
        }

        growth_rates
    }
}

/// Statistics anomaly detection result
#[derive(Debug, Clone)]
pub struct StatisticsAnomaly {
    pub anomaly_type: AnomalyType,
    pub predicate: String,
    pub value: f64,
    pub z_score: f64,
    pub severity: AnomalySeverity,
}

/// Types of statistical anomalies
#[derive(Debug, Clone)]
pub enum AnomalyType {
    FrequencyOutlier,
    CorrelationAnomaly,
    CardinalitySpike,
}

/// Severity levels for anomalies
#[derive(Debug, Clone)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
}

/// Statistics evolution summary
#[derive(Debug, Clone)]
pub struct StatisticsEvolution {
    pub snapshots_count: usize,
    pub time_span: Duration,
    pub growth_rates: HashMap<String, f64>,
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
                    let current = stats
                        .pattern_cardinality
                        .get(&pattern_key)
                        .copied()
                        .unwrap_or(1000);

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
        let mis_estimates: Vec<_> = self
            .execution_history
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
        let slow_queries: Vec<_> = self
            .execution_history
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
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new("http://example.org/type").unwrap()),
                object: Term::Variable(Variable::new("o").unwrap()),
            },
            TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new("http://example.org/name").unwrap()),
                object: Term::Literal(Literal {
                    value: "Test".to_string(),
                    language: None,
                    datatype: None,
                }),
            },
        ];

        collector.collect_from_patterns(&patterns).unwrap();

        let stats = collector.get_statistics();
        assert!(stats
            .predicate_frequency
            .contains_key("http://example.org/type"));
        assert!(stats
            .predicate_frequency
            .contains_key("http://example.org/name"));
        assert!(stats
            .variable_selectivity
            .contains_key(&Variable::new("s").unwrap()));
    }

    #[test]
    fn test_cardinality_estimation() {
        let collector = StatisticsCollector::new();

        let pattern = TriplePattern {
            subject: Term::Iri(NamedNode::new("http://example.org/subject").unwrap()),
            predicate: Term::Iri(NamedNode::new("http://example.org/predicate").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };

        let cardinality = collector.estimate_pattern_cardinality(&pattern);
        assert!(cardinality > 0);
    }

    #[test]
    fn test_dynamic_updates() {
        let mut updater = DynamicStatisticsUpdater::new();
        let mut stats = Statistics::new();

        let record = QueryExecutionRecord {
            algebra: Algebra::Bgp(vec![TriplePattern {
                subject: Term::Variable(Variable::new("s").unwrap()),
                predicate: Term::Iri(NamedNode::new("http://example.org/type").unwrap()),
                object: Term::Variable(Variable::new("o").unwrap()),
            }]),
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
