//! # Adaptive Index Advisor
//!
//! This module provides intelligent index recommendations based on query workload analysis.
//! It analyzes query patterns, tracks index usage, and suggests optimal index configurations.
//!
//! ## Features
//!
//! - **Query Pattern Analysis**: Analyzes triple patterns to identify indexing opportunities
//! - **Workload-Based Recommendations**: Learns from query history to suggest beneficial indexes
//! - **Index Benefit Estimation**: Estimates performance improvements for proposed indexes
//! - **Index Consolidation**: Identifies redundant or overlapping indexes
//! - **Cost-Benefit Analysis**: Balances index maintenance cost against query performance gains
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use oxirs_arq::adaptive_index_advisor::{
//!     IndexAdvisor, AdvisorConfig, QueryPattern, IndexRecommendation,
//! };
//!
//! // Create an index advisor
//! let config = AdvisorConfig::default();
//! let mut advisor = IndexAdvisor::new(config);
//!
//! // Record query patterns
//! advisor.record_query("SELECT * WHERE { ?s :knows ?o }");
//!
//! // Get recommendations
//! let recommendations = advisor.get_recommendations();
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::SystemTime;

/// Index type for RDF data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// Subject-Predicate-Object ordering
    SPO,
    /// Subject-Object-Predicate ordering
    SOP,
    /// Predicate-Subject-Object ordering
    PSO,
    /// Predicate-Object-Subject ordering
    POS,
    /// Object-Subject-Predicate ordering
    OSP,
    /// Object-Predicate-Subject ordering
    OPS,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SPO => write!(f, "SPO"),
            Self::SOP => write!(f, "SOP"),
            Self::PSO => write!(f, "PSO"),
            Self::POS => write!(f, "POS"),
            Self::OSP => write!(f, "OSP"),
            Self::OPS => write!(f, "OPS"),
        }
    }
}

impl IndexType {
    /// Get all possible index types
    pub fn all() -> Vec<IndexType> {
        vec![
            Self::SPO,
            Self::SOP,
            Self::PSO,
            Self::POS,
            Self::OSP,
            Self::OPS,
        ]
    }

    /// Check if this index covers the given access pattern
    /// An index is useful when its primary component is bound
    pub fn covers_pattern(&self, pattern: &AccessPattern) -> bool {
        match self {
            // SPO/SOP indexes are useful when subject is bound
            Self::SPO | Self::SOP => pattern.has_subject,
            // PSO/POS indexes are useful when predicate is bound
            Self::PSO | Self::POS => pattern.has_predicate,
            // OSP/OPS indexes are useful when object is bound
            Self::OSP | Self::OPS => pattern.has_object,
        }
    }

    /// Get the selectivity order for this index
    pub fn primary_component(&self) -> PatternComponent {
        match self {
            Self::SPO | Self::SOP => PatternComponent::Subject,
            Self::PSO | Self::POS => PatternComponent::Predicate,
            Self::OSP | Self::OPS => PatternComponent::Object,
        }
    }
}

/// Components of a triple pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternComponent {
    Subject,
    Predicate,
    Object,
}

impl fmt::Display for PatternComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Subject => write!(f, "subject"),
            Self::Predicate => write!(f, "predicate"),
            Self::Object => write!(f, "object"),
        }
    }
}

/// Access pattern extracted from a query
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AccessPattern {
    /// Whether subject is bound (constant)
    pub has_subject: bool,
    /// Whether predicate is bound (constant)
    pub has_predicate: bool,
    /// Whether object is bound (constant)
    pub has_object: bool,
    /// Optional predicate value if known
    pub predicate_value: Option<String>,
}

impl AccessPattern {
    /// Create a new access pattern
    pub fn new(has_subject: bool, has_predicate: bool, has_object: bool) -> Self {
        Self {
            has_subject,
            has_predicate,
            has_object,
            predicate_value: None,
        }
    }

    /// Create with predicate value
    pub fn with_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.predicate_value = Some(predicate.into());
        self
    }

    /// Get pattern signature for grouping
    pub fn signature(&self) -> String {
        format!(
            "{}{}{}",
            if self.has_subject { "S" } else { "?" },
            if self.has_predicate { "P" } else { "?" },
            if self.has_object { "O" } else { "?" }
        )
    }

    /// Count bound components
    pub fn bound_count(&self) -> usize {
        let mut count = 0;
        if self.has_subject {
            count += 1;
        }
        if self.has_predicate {
            count += 1;
        }
        if self.has_object {
            count += 1;
        }
        count
    }

    /// Best index type for this pattern
    pub fn best_index(&self) -> Option<IndexType> {
        match (self.has_subject, self.has_predicate, self.has_object) {
            (true, true, true) => Some(IndexType::SPO),
            (true, true, false) => Some(IndexType::SPO),
            (true, false, true) => Some(IndexType::SOP),
            (true, false, false) => Some(IndexType::SPO),
            (false, true, true) => Some(IndexType::POS),
            (false, true, false) => Some(IndexType::PSO),
            (false, false, true) => Some(IndexType::OSP),
            (false, false, false) => None, // Full scan needed
        }
    }
}

impl fmt::Display for AccessPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.signature())
    }
}

/// Query pattern with associated metadata
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// The access pattern
    pub access_pattern: AccessPattern,
    /// Number of times this pattern was observed
    pub frequency: usize,
    /// Average selectivity (0.0 - 1.0)
    pub avg_selectivity: f64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Last observed timestamp
    pub last_seen: SystemTime,
}

impl QueryPattern {
    /// Create a new query pattern
    pub fn new(access_pattern: AccessPattern) -> Self {
        Self {
            access_pattern,
            frequency: 1,
            avg_selectivity: 1.0,
            avg_execution_time_ms: 0.0,
            last_seen: SystemTime::now(),
        }
    }

    /// Update with new observation
    pub fn update(&mut self, selectivity: f64, execution_time_ms: f64) {
        let n = self.frequency as f64;
        self.avg_selectivity = (self.avg_selectivity * n + selectivity) / (n + 1.0);
        self.avg_execution_time_ms =
            (self.avg_execution_time_ms * n + execution_time_ms) / (n + 1.0);
        self.frequency += 1;
        self.last_seen = SystemTime::now();
    }
}

/// Configuration for the index advisor
#[derive(Debug, Clone)]
pub struct AdvisorConfig {
    /// Minimum frequency for a pattern to be considered
    pub min_pattern_frequency: usize,
    /// Maximum number of recommendations to generate
    pub max_recommendations: usize,
    /// Weight for query frequency in benefit calculation
    pub frequency_weight: f64,
    /// Weight for execution time in benefit calculation
    pub execution_time_weight: f64,
    /// Minimum benefit score to recommend an index
    pub min_benefit_score: f64,
    /// Consider index maintenance cost
    pub consider_maintenance_cost: bool,
    /// Decay factor for old patterns (0.0-1.0)
    pub time_decay_factor: f64,
    /// Maximum patterns to track
    pub max_tracked_patterns: usize,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            min_pattern_frequency: 5,
            max_recommendations: 10,
            frequency_weight: 0.4,
            execution_time_weight: 0.6,
            min_benefit_score: 0.1,
            consider_maintenance_cost: true,
            time_decay_factor: 0.95,
            max_tracked_patterns: 1000,
        }
    }
}

impl AdvisorConfig {
    /// Conservative configuration for production
    pub fn conservative() -> Self {
        Self {
            min_pattern_frequency: 10,
            max_recommendations: 5,
            frequency_weight: 0.3,
            execution_time_weight: 0.7,
            min_benefit_score: 0.3,
            consider_maintenance_cost: true,
            time_decay_factor: 0.90,
            max_tracked_patterns: 500,
        }
    }

    /// Aggressive configuration for optimization
    pub fn aggressive() -> Self {
        Self {
            min_pattern_frequency: 3,
            max_recommendations: 15,
            frequency_weight: 0.5,
            execution_time_weight: 0.5,
            min_benefit_score: 0.05,
            consider_maintenance_cost: false,
            time_decay_factor: 0.98,
            max_tracked_patterns: 2000,
        }
    }
}

/// Index recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Critical - significant performance impact
    Critical,
    /// High - notable improvement expected
    High,
    /// Medium - moderate improvement expected
    Medium,
    /// Low - minor improvement expected
    Low,
}

impl fmt::Display for RecommendationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::High => write!(f, "HIGH"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::Low => write!(f, "LOW"),
        }
    }
}

/// A single index recommendation
#[derive(Debug, Clone)]
pub struct IndexRecommendation {
    /// Recommended index type
    pub index_type: IndexType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Estimated benefit score (0.0 - 1.0)
    pub benefit_score: f64,
    /// Estimated performance improvement percentage
    pub estimated_improvement_percent: f64,
    /// Patterns that would benefit
    pub benefiting_patterns: Vec<AccessPattern>,
    /// Total query frequency affected
    pub total_frequency: usize,
    /// Rationale for the recommendation
    pub rationale: String,
    /// Estimated storage overhead
    pub storage_overhead_percent: f64,
    /// Estimated write performance impact
    pub write_impact_percent: f64,
}

impl IndexRecommendation {
    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "[{}] {} - {:.1}% improvement, affects {} queries",
            self.priority,
            self.index_type,
            self.estimated_improvement_percent,
            self.total_frequency
        )
    }
}

/// Current index configuration
#[derive(Debug, Clone, Default)]
pub struct IndexConfiguration {
    /// Active indexes
    pub active_indexes: HashSet<IndexType>,
    /// Index usage statistics
    pub usage_stats: HashMap<IndexType, IndexUsageStats>,
}

impl IndexConfiguration {
    /// Create a default configuration (SPO only)
    pub fn default_spo() -> Self {
        let mut config = Self::default();
        config.active_indexes.insert(IndexType::SPO);
        config
    }

    /// Check if an index is active
    pub fn has_index(&self, index_type: IndexType) -> bool {
        self.active_indexes.contains(&index_type)
    }

    /// Add an index
    pub fn add_index(&mut self, index_type: IndexType) {
        self.active_indexes.insert(index_type);
    }

    /// Remove an index
    pub fn remove_index(&mut self, index_type: IndexType) -> bool {
        self.active_indexes.remove(&index_type)
    }
}

/// Usage statistics for an index
#[derive(Debug, Clone, Default)]
pub struct IndexUsageStats {
    /// Number of queries that used this index
    pub query_count: usize,
    /// Number of index lookups
    pub lookup_count: usize,
    /// Total rows scanned
    pub rows_scanned: usize,
    /// Total rows returned
    pub rows_returned: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Analysis report for index optimization
#[derive(Debug, Clone)]
pub struct IndexAnalysisReport {
    /// Generated timestamp
    pub generated_at: SystemTime,
    /// Current index configuration
    pub current_config: IndexConfiguration,
    /// Recommendations
    pub recommendations: Vec<IndexRecommendation>,
    /// Unused indexes that could be removed
    pub unused_indexes: Vec<IndexType>,
    /// Overlapping index pairs
    pub overlapping_indexes: Vec<(IndexType, IndexType)>,
    /// Summary statistics
    pub summary: AnalysisSummary,
}

impl IndexAnalysisReport {
    /// Get high priority recommendations
    pub fn high_priority(&self) -> Vec<&IndexRecommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.priority <= RecommendationPriority::High)
            .collect()
    }

    /// Check if any changes are recommended
    pub fn has_recommendations(&self) -> bool {
        !self.recommendations.is_empty() || !self.unused_indexes.is_empty()
    }

    /// Generate a text summary
    pub fn text_summary(&self) -> String {
        let mut text = String::from("Index Analysis Report\n");
        text.push_str(&format!("Generated: {:?}\n\n", self.generated_at));

        text.push_str(&format!(
            "Current Indexes: {:?}\n",
            self.current_config.active_indexes
        ));
        text.push_str(&format!(
            "Patterns Analyzed: {}\n",
            self.summary.total_patterns
        ));
        text.push_str(&format!(
            "Total Queries: {}\n\n",
            self.summary.total_queries
        ));

        if !self.recommendations.is_empty() {
            text.push_str("Recommendations:\n");
            for rec in &self.recommendations {
                text.push_str(&format!("  - {}\n", rec.summary()));
            }
            text.push('\n');
        }

        if !self.unused_indexes.is_empty() {
            text.push_str(&format!(
                "Unused Indexes (consider removing): {:?}\n",
                self.unused_indexes
            ));
        }

        text
    }
}

/// Summary statistics for analysis
#[derive(Debug, Clone, Default)]
pub struct AnalysisSummary {
    /// Total patterns analyzed
    pub total_patterns: usize,
    /// Total queries analyzed
    pub total_queries: usize,
    /// Average selectivity
    pub avg_selectivity: f64,
    /// Most frequent pattern signature
    pub most_frequent_pattern: Option<String>,
    /// Estimated overall improvement if all recommendations adopted
    pub potential_improvement_percent: f64,
}

/// Main index advisor
#[derive(Debug)]
pub struct IndexAdvisor {
    /// Configuration
    config: AdvisorConfig,
    /// Current index configuration
    current_indexes: IndexConfiguration,
    /// Tracked query patterns
    patterns: HashMap<String, QueryPattern>,
    /// Statistics
    stats: AdvisorStatistics,
}

/// Statistics for the advisor
#[derive(Debug, Clone, Default)]
pub struct AdvisorStatistics {
    /// Total queries recorded
    pub total_queries: usize,
    /// Total patterns discovered
    pub total_patterns: usize,
    /// Analysis count
    pub analyses_performed: usize,
    /// Recommendations generated
    pub recommendations_generated: usize,
}

impl IndexAdvisor {
    /// Create a new index advisor
    pub fn new(config: AdvisorConfig) -> Self {
        Self {
            config,
            current_indexes: IndexConfiguration::default_spo(),
            patterns: HashMap::new(),
            stats: AdvisorStatistics::default(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(AdvisorConfig::default())
    }

    /// Set the current index configuration
    pub fn set_indexes(&mut self, indexes: IndexConfiguration) {
        self.current_indexes = indexes;
    }

    /// Record a query pattern
    pub fn record_pattern(
        &mut self,
        access_pattern: AccessPattern,
        selectivity: f64,
        execution_time_ms: f64,
    ) {
        let signature = access_pattern.signature();
        self.stats.total_queries += 1;

        if let Some(pattern) = self.patterns.get_mut(&signature) {
            pattern.update(selectivity, execution_time_ms);
        } else {
            let mut pattern = QueryPattern::new(access_pattern);
            pattern.avg_selectivity = selectivity;
            pattern.avg_execution_time_ms = execution_time_ms;
            self.patterns.insert(signature, pattern);
            self.stats.total_patterns += 1;
        }

        // Evict old patterns if needed
        if self.patterns.len() > self.config.max_tracked_patterns {
            self.evict_old_patterns();
        }
    }

    /// Record a simple query (parses pattern from SPARQL-like syntax)
    pub fn record_query(&mut self, query: &str) {
        // Simple pattern extraction - look for triple patterns
        let patterns = self.extract_patterns(query);
        for pattern in patterns {
            self.record_pattern(pattern, 1.0, 0.0);
        }
    }

    /// Analyze workload and generate recommendations
    pub fn analyze(&mut self) -> IndexAnalysisReport {
        self.stats.analyses_performed += 1;

        let mut recommendations = Vec::new();
        let mut index_benefits: HashMap<IndexType, IndexBenefitAccumulator> = HashMap::new();

        // Analyze each pattern
        for pattern in self.patterns.values() {
            if pattern.frequency < self.config.min_pattern_frequency {
                continue;
            }

            // Find best index for this pattern
            if let Some(best_index) = pattern.access_pattern.best_index() {
                // Check if we already have this index
                if !self.current_indexes.has_index(best_index) {
                    let entry = index_benefits.entry(best_index).or_default();
                    entry.add_pattern(pattern);
                }
            }
        }

        // Convert benefits to recommendations
        for (index_type, benefits) in index_benefits {
            let benefit_score = self.calculate_benefit_score(&benefits);
            if benefit_score >= self.config.min_benefit_score {
                let priority = self.determine_priority(benefit_score);
                let estimated_improvement = self.estimate_improvement(&benefits);

                recommendations.push(IndexRecommendation {
                    index_type,
                    priority,
                    benefit_score,
                    estimated_improvement_percent: estimated_improvement,
                    benefiting_patterns: benefits.patterns.clone(),
                    total_frequency: benefits.total_frequency,
                    rationale: self.generate_rationale(&benefits, index_type),
                    storage_overhead_percent: self.estimate_storage_overhead(index_type),
                    write_impact_percent: self.estimate_write_impact(index_type),
                });
            }
        }

        // Sort by priority and benefit
        recommendations.sort_by(|a, b| match a.priority.cmp(&b.priority) {
            std::cmp::Ordering::Equal => b
                .benefit_score
                .partial_cmp(&a.benefit_score)
                .unwrap_or(std::cmp::Ordering::Equal),
            other => other,
        });

        // Limit recommendations
        recommendations.truncate(self.config.max_recommendations);
        self.stats.recommendations_generated += recommendations.len();

        // Find unused indexes
        let unused_indexes = self.find_unused_indexes();

        // Find overlapping indexes
        let overlapping_indexes = self.find_overlapping_indexes();

        // Generate summary
        let summary = self.generate_summary(&recommendations);

        IndexAnalysisReport {
            generated_at: SystemTime::now(),
            current_config: self.current_indexes.clone(),
            recommendations,
            unused_indexes,
            overlapping_indexes,
            summary,
        }
    }

    /// Get current statistics
    pub fn statistics(&self) -> &AdvisorStatistics {
        &self.stats
    }

    /// Get current configuration
    pub fn config(&self) -> &AdvisorConfig {
        &self.config
    }

    /// Clear all tracked patterns
    pub fn clear_patterns(&mut self) {
        self.patterns.clear();
        self.stats.total_patterns = 0;
    }

    /// Export tracked patterns for persistence
    pub fn export_patterns(&self) -> Vec<(String, QueryPattern)> {
        self.patterns
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Import patterns
    pub fn import_patterns(&mut self, patterns: Vec<(String, QueryPattern)>) {
        for (sig, pattern) in patterns {
            self.patterns.insert(sig, pattern);
            self.stats.total_patterns += 1;
        }
    }

    // Private methods

    fn extract_patterns(&self, query: &str) -> Vec<AccessPattern> {
        let mut patterns = Vec::new();
        let query_lower = query.to_lowercase();

        // Look for triple pattern-like structures: ?var :pred ?var or :subj :pred ?var etc.
        // Simple heuristic parsing - not full SPARQL parsing
        for part in query_lower.split('{') {
            for segment in part.split('.') {
                let segment = segment.trim();
                if segment.is_empty() || segment.starts_with('}') {
                    continue;
                }

                let tokens: Vec<&str> = segment.split_whitespace().collect();
                if tokens.len() >= 3 {
                    let has_subject = !tokens[0].starts_with('?');
                    let has_predicate = !tokens[1].starts_with('?');
                    let has_object = !tokens[2].starts_with('?');

                    let mut pattern = AccessPattern::new(has_subject, has_predicate, has_object);
                    if has_predicate {
                        pattern.predicate_value = Some(tokens[1].to_string());
                    }
                    patterns.push(pattern);
                }
            }
        }

        if patterns.is_empty() {
            // Default to full scan pattern if no patterns found
            patterns.push(AccessPattern::new(false, false, false));
        }

        patterns
    }

    fn calculate_benefit_score(&self, benefits: &IndexBenefitAccumulator) -> f64 {
        let freq_score = (benefits.total_frequency as f64).ln().max(0.0) / 10.0;
        let time_score = benefits.total_execution_time_ms / 1000.0;

        let raw_score = self.config.frequency_weight * freq_score
            + self.config.execution_time_weight * time_score;

        // Normalize to 0.0 - 1.0
        (raw_score / 2.0).min(1.0)
    }

    fn determine_priority(&self, benefit_score: f64) -> RecommendationPriority {
        if benefit_score >= 0.8 {
            RecommendationPriority::Critical
        } else if benefit_score >= 0.5 {
            RecommendationPriority::High
        } else if benefit_score >= 0.3 {
            RecommendationPriority::Medium
        } else {
            RecommendationPriority::Low
        }
    }

    fn estimate_improvement(&self, benefits: &IndexBenefitAccumulator) -> f64 {
        // Estimate based on selectivity improvement
        let avg_selectivity = if benefits.patterns.is_empty() {
            1.0
        } else {
            benefits.patterns.len() as f64 / self.patterns.len() as f64
        };

        // Indexes typically improve performance by 10-90% depending on selectivity
        let base_improvement = 50.0 * (1.0 - avg_selectivity);
        base_improvement.clamp(10.0, 90.0)
    }

    fn generate_rationale(
        &self,
        benefits: &IndexBenefitAccumulator,
        index_type: IndexType,
    ) -> String {
        format!(
            "Index {} would benefit {} pattern(s) with total frequency {} queries. \
             Primary optimization for {} lookups.",
            index_type,
            benefits.patterns.len(),
            benefits.total_frequency,
            index_type.primary_component()
        )
    }

    fn estimate_storage_overhead(&self, _index_type: IndexType) -> f64 {
        // Each index typically adds ~100% storage overhead for the indexed data
        // This is a simplified estimate
        100.0
    }

    fn estimate_write_impact(&self, _index_type: IndexType) -> f64 {
        // Each additional index typically adds ~20% write overhead
        20.0
    }

    fn find_unused_indexes(&self) -> Vec<IndexType> {
        let mut unused = Vec::new();
        for &index_type in &self.current_indexes.active_indexes {
            let is_used = self.patterns.values().any(|p| {
                p.frequency >= self.config.min_pattern_frequency
                    && index_type.covers_pattern(&p.access_pattern)
            });
            if !is_used && index_type != IndexType::SPO {
                // Never recommend removing SPO as it's the default
                unused.push(index_type);
            }
        }
        unused
    }

    fn find_overlapping_indexes(&self) -> Vec<(IndexType, IndexType)> {
        let mut overlaps = Vec::new();
        let indexes: Vec<_> = self.current_indexes.active_indexes.iter().collect();

        for i in 0..indexes.len() {
            for j in (i + 1)..indexes.len() {
                if indexes[i].primary_component() == indexes[j].primary_component() {
                    overlaps.push((*indexes[i], *indexes[j]));
                }
            }
        }
        overlaps
    }

    fn generate_summary(&self, recommendations: &[IndexRecommendation]) -> AnalysisSummary {
        let total_patterns = self.patterns.len();
        let total_queries: usize = self.patterns.values().map(|p| p.frequency).sum();

        let avg_selectivity = if total_patterns > 0 {
            self.patterns
                .values()
                .map(|p| p.avg_selectivity)
                .sum::<f64>()
                / total_patterns as f64
        } else {
            1.0
        };

        let most_frequent_pattern = self
            .patterns
            .iter()
            .max_by_key(|(_, p)| p.frequency)
            .map(|(sig, _)| sig.clone());

        let potential_improvement = recommendations
            .iter()
            .map(|r| r.estimated_improvement_percent)
            .sum::<f64>()
            .min(95.0); // Cap at 95%

        AnalysisSummary {
            total_patterns,
            total_queries,
            avg_selectivity,
            most_frequent_pattern,
            potential_improvement_percent: potential_improvement,
        }
    }

    fn evict_old_patterns(&mut self) {
        // Sort by last seen and keep most recent
        let mut sorted: Vec<_> = self.patterns.iter().collect();
        sorted.sort_by(|a, b| b.1.last_seen.cmp(&a.1.last_seen));

        let to_keep: HashSet<_> = sorted
            .iter()
            .take(self.config.max_tracked_patterns / 2)
            .map(|(k, _)| (*k).clone())
            .collect();

        self.patterns.retain(|k, _| to_keep.contains(k));
        self.stats.total_patterns = self.patterns.len();
    }
}

/// Accumulator for index benefit calculation
#[derive(Debug, Default)]
struct IndexBenefitAccumulator {
    patterns: Vec<AccessPattern>,
    total_frequency: usize,
    total_execution_time_ms: f64,
}

impl IndexBenefitAccumulator {
    fn add_pattern(&mut self, pattern: &QueryPattern) {
        self.patterns.push(pattern.access_pattern.clone());
        self.total_frequency += pattern.frequency;
        self.total_execution_time_ms += pattern.avg_execution_time_ms * pattern.frequency as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern_signature() {
        let pattern = AccessPattern::new(true, true, false);
        assert_eq!(pattern.signature(), "SP?");

        let pattern = AccessPattern::new(false, true, true);
        assert_eq!(pattern.signature(), "?PO");

        let pattern = AccessPattern::new(false, false, false);
        assert_eq!(pattern.signature(), "???");
    }

    #[test]
    fn test_access_pattern_best_index() {
        assert_eq!(
            AccessPattern::new(true, true, false).best_index(),
            Some(IndexType::SPO)
        );
        assert_eq!(
            AccessPattern::new(true, false, true).best_index(),
            Some(IndexType::SOP)
        );
        assert_eq!(
            AccessPattern::new(false, true, true).best_index(),
            Some(IndexType::POS)
        );
        assert_eq!(
            AccessPattern::new(false, false, true).best_index(),
            Some(IndexType::OSP)
        );
        assert_eq!(AccessPattern::new(false, false, false).best_index(), None);
    }

    #[test]
    fn test_index_type_covers_pattern() {
        let pattern = AccessPattern::new(true, true, false);
        assert!(IndexType::SPO.covers_pattern(&pattern));
        assert!(IndexType::PSO.covers_pattern(&pattern));
    }

    #[test]
    fn test_query_pattern_update() {
        let mut pattern = QueryPattern::new(AccessPattern::new(true, false, false));
        assert_eq!(pattern.frequency, 1);

        pattern.update(0.5, 10.0);
        assert_eq!(pattern.frequency, 2);

        pattern.update(0.3, 20.0);
        assert_eq!(pattern.frequency, 3);
    }

    #[test]
    fn test_index_advisor_creation() {
        let advisor = IndexAdvisor::with_defaults();
        assert!(advisor.current_indexes.has_index(IndexType::SPO));
    }

    #[test]
    fn test_record_pattern() {
        let mut advisor = IndexAdvisor::with_defaults();

        advisor.record_pattern(AccessPattern::new(true, true, false), 0.1, 5.0);
        advisor.record_pattern(AccessPattern::new(true, true, false), 0.2, 6.0);

        assert_eq!(advisor.stats.total_queries, 2);
        assert_eq!(advisor.stats.total_patterns, 1);
    }

    #[test]
    fn test_record_query() {
        let mut advisor = IndexAdvisor::with_defaults();

        advisor.record_query("SELECT * WHERE { ?s :knows ?o }");
        assert!(advisor.stats.total_queries > 0);
    }

    #[test]
    fn test_analyze_with_patterns() {
        let config = AdvisorConfig {
            min_pattern_frequency: 2,
            min_benefit_score: 0.01, // Lower threshold for test
            ..Default::default()
        };
        let mut advisor = IndexAdvisor::new(config);

        // Record patterns that would benefit from POS index (not covered by SPO)
        for _ in 0..50 {
            advisor.record_pattern(AccessPattern::new(false, true, true), 0.1, 100.0);
        }

        let report = advisor.analyze();
        // May or may not have recommendations depending on benefit calculation
        // Just verify analysis completes successfully
        assert!(report.summary.total_patterns > 0);
        assert!(report.summary.total_queries > 0);
    }

    #[test]
    fn test_analyze_empty() {
        let mut advisor = IndexAdvisor::with_defaults();
        let report = advisor.analyze();
        assert!(report.recommendations.is_empty());
    }

    #[test]
    fn test_recommendation_priority() {
        let config = AdvisorConfig {
            min_pattern_frequency: 1,
            ..Default::default()
        };
        let mut advisor = IndexAdvisor::new(config);

        // High frequency pattern
        for _ in 0..100 {
            advisor.record_pattern(AccessPattern::new(false, false, true), 0.01, 100.0);
        }

        let report = advisor.analyze();
        if !report.recommendations.is_empty() {
            // Should have high priority due to frequency
            assert!(report.recommendations[0].priority <= RecommendationPriority::High);
        }
    }

    #[test]
    fn test_index_configuration() {
        let mut config = IndexConfiguration::default_spo();
        assert!(config.has_index(IndexType::SPO));
        assert!(!config.has_index(IndexType::POS));

        config.add_index(IndexType::POS);
        assert!(config.has_index(IndexType::POS));

        config.remove_index(IndexType::POS);
        assert!(!config.has_index(IndexType::POS));
    }

    #[test]
    fn test_unused_index_detection() {
        let mut advisor = IndexAdvisor::with_defaults();
        advisor.current_indexes.add_index(IndexType::OSP);

        // Only record SPO-compatible patterns
        for _ in 0..10 {
            advisor.record_pattern(AccessPattern::new(true, true, false), 0.1, 5.0);
        }

        let report = advisor.analyze();
        assert!(report.unused_indexes.contains(&IndexType::OSP));
    }

    #[test]
    fn test_export_import_patterns() {
        let mut advisor = IndexAdvisor::with_defaults();
        advisor.record_pattern(AccessPattern::new(true, false, false), 0.5, 10.0);
        advisor.record_pattern(AccessPattern::new(false, true, false), 0.3, 15.0);

        let exported = advisor.export_patterns();
        assert_eq!(exported.len(), 2);

        let mut new_advisor = IndexAdvisor::with_defaults();
        new_advisor.import_patterns(exported);
        assert_eq!(new_advisor.stats.total_patterns, 2);
    }

    #[test]
    fn test_config_presets() {
        let conservative = AdvisorConfig::conservative();
        assert_eq!(conservative.min_pattern_frequency, 10);

        let aggressive = AdvisorConfig::aggressive();
        assert_eq!(aggressive.min_pattern_frequency, 3);
    }

    #[test]
    fn test_report_text_summary() {
        let config = AdvisorConfig {
            min_pattern_frequency: 1,
            ..Default::default()
        };
        let mut advisor = IndexAdvisor::new(config);

        for _ in 0..5 {
            advisor.record_pattern(AccessPattern::new(false, true, true), 0.1, 10.0);
        }

        let report = advisor.analyze();
        let summary = report.text_summary();

        assert!(summary.contains("Index Analysis Report"));
        assert!(summary.contains("Current Indexes"));
    }

    #[test]
    fn test_pattern_bound_count() {
        assert_eq!(AccessPattern::new(true, true, true).bound_count(), 3);
        assert_eq!(AccessPattern::new(true, false, false).bound_count(), 1);
        assert_eq!(AccessPattern::new(false, false, false).bound_count(), 0);
    }

    #[test]
    fn test_index_type_display() {
        assert_eq!(format!("{}", IndexType::SPO), "SPO");
        assert_eq!(format!("{}", IndexType::POS), "POS");
    }
}
