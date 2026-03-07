//! Data analysis for AI orchestrator

pub use super::types::{DataCharacteristics, ModelPerformanceMetrics};

/// Statistical summary of a dataset
#[derive(Debug, Clone)]
pub struct DataStatistics {
    /// Total number of triples
    pub triple_count: usize,
    /// Total number of distinct subjects
    pub subject_count: usize,
    /// Total number of distinct predicates
    pub predicate_count: usize,
    /// Total number of distinct objects
    pub object_count: usize,
    /// Average triples per subject
    pub avg_triples_per_subject: f64,
    /// Maximum triples for a single subject
    pub max_triples_per_subject: usize,
    /// Fraction of nodes that are blank nodes
    pub blank_node_ratio: f64,
    /// Fraction of objects that are literals
    pub literal_ratio: f64,
    /// Depth of rdf:type hierarchy
    pub type_hierarchy_depth: u32,
    /// Number of distinct rdf:type values
    pub type_count: usize,
}

impl Default for DataStatistics {
    fn default() -> Self {
        Self {
            triple_count: 0,
            subject_count: 0,
            predicate_count: 0,
            object_count: 0,
            avg_triples_per_subject: 0.0,
            max_triples_per_subject: 0,
            blank_node_ratio: 0.0,
            literal_ratio: 0.0,
            type_hierarchy_depth: 0,
            type_count: 0,
        }
    }
}

impl DataStatistics {
    /// Create statistics from raw triple counts
    pub fn from_counts(
        triple_count: usize,
        subject_count: usize,
        predicate_count: usize,
        object_count: usize,
    ) -> Self {
        let avg_triples_per_subject = if subject_count > 0 {
            triple_count as f64 / subject_count as f64
        } else {
            0.0
        };

        Self {
            triple_count,
            subject_count,
            predicate_count,
            object_count,
            avg_triples_per_subject,
            max_triples_per_subject: 0,
            blank_node_ratio: 0.0,
            literal_ratio: 0.0,
            type_hierarchy_depth: 0,
            type_count: 0,
        }
    }

    /// Convert statistics into data characteristics for model selection
    pub fn to_characteristics(&self) -> DataCharacteristics {
        let complexity_score = {
            let predicate_diversity =
                self.predicate_count as f64 / (self.triple_count as f64 + 1.0);
            let structural_complexity = self.type_hierarchy_depth as f64 / 10.0;
            (predicate_diversity + structural_complexity) / 2.0
        };

        DataCharacteristics {
            graph_size: self.triple_count,
            complexity_score: complexity_score.min(1.0),
            sparsity_ratio: 1.0
                - (self.triple_count as f64
                    / ((self.subject_count * self.predicate_count) as f64 + 1.0))
                    .min(1.0),
            hierarchy_depth: self.type_hierarchy_depth,
            pattern_diversity: (self.predicate_count as f64 / 100.0).min(1.0),
            semantic_richness: (self.type_count as f64 / 50.0).min(1.0),
        }
    }

    /// Estimate complexity category
    pub fn complexity_category(&self) -> ComplexityCategory {
        match self.triple_count {
            0..=1_000 => ComplexityCategory::Small,
            1_001..=100_000 => ComplexityCategory::Medium,
            100_001..=10_000_000 => ComplexityCategory::Large,
            _ => ComplexityCategory::VeryLarge,
        }
    }
}

/// Complexity category for dataset size estimation
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityCategory {
    /// Fewer than 1,000 triples — fast models can be used
    Small,
    /// 1,000–100,000 triples — standard models
    Medium,
    /// 100,000–10,000,000 triples — streaming or distributed models
    Large,
    /// More than 10,000,000 triples — distributed only
    VeryLarge,
}

/// Analyzer that computes data characteristics from RDF statistics
pub struct DataAnalyzer {
    /// Minimum triple count before complexity analysis is meaningful
    min_triples_for_analysis: usize,
}

impl Default for DataAnalyzer {
    fn default() -> Self {
        Self {
            min_triples_for_analysis: 10,
        }
    }
}

impl DataAnalyzer {
    /// Create a new DataAnalyzer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyse the given statistics and return characteristics
    pub fn analyse(&self, stats: &DataStatistics) -> DataCharacteristics {
        if stats.triple_count < self.min_triples_for_analysis {
            return DataCharacteristics::default();
        }
        stats.to_characteristics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_statistics_from_counts() {
        let stats = DataStatistics::from_counts(100, 20, 10, 50);
        assert_eq!(stats.triple_count, 100);
        assert!((stats.avg_triples_per_subject - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_to_characteristics() {
        let stats = DataStatistics::from_counts(1000, 100, 20, 200);
        let chars = stats.to_characteristics();
        assert!(chars.complexity_score >= 0.0);
        assert!(chars.complexity_score <= 1.0);
        assert_eq!(chars.graph_size, 1000);
    }

    #[test]
    fn test_complexity_category_small() {
        let stats = DataStatistics {
            triple_count: 100,
            ..Default::default()
        };
        assert_eq!(stats.complexity_category(), ComplexityCategory::Small);
    }

    #[test]
    fn test_complexity_category_medium() {
        let stats = DataStatistics {
            triple_count: 50_000,
            ..Default::default()
        };
        assert_eq!(stats.complexity_category(), ComplexityCategory::Medium);
    }

    #[test]
    fn test_complexity_category_large() {
        let stats = DataStatistics {
            triple_count: 5_000_000,
            ..Default::default()
        };
        assert_eq!(stats.complexity_category(), ComplexityCategory::Large);
    }

    #[test]
    fn test_complexity_category_very_large() {
        let stats = DataStatistics {
            triple_count: 50_000_000,
            ..Default::default()
        };
        assert_eq!(stats.complexity_category(), ComplexityCategory::VeryLarge);
    }

    #[test]
    fn test_data_analyzer_default() {
        let analyzer = DataAnalyzer::new();
        let stats = DataStatistics::from_counts(5, 3, 2, 4);
        // Less than min_triples_for_analysis — returns default
        let chars = analyzer.analyse(&stats);
        assert_eq!(chars.graph_size, 0);
    }

    #[test]
    fn test_data_analyzer_sufficient_data() {
        let analyzer = DataAnalyzer::new();
        let stats = DataStatistics::from_counts(100, 20, 10, 50);
        let chars = analyzer.analyse(&stats);
        assert_eq!(chars.graph_size, 100);
    }

    #[test]
    fn test_default_statistics() {
        let stats = DataStatistics::default();
        assert_eq!(stats.triple_count, 0);
        assert!((stats.avg_triples_per_subject).abs() < 1e-9);
    }
}
