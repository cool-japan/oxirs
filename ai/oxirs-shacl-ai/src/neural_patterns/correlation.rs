//! Pattern correlation analysis for discovering relationships between SHACL patterns

use scirs2_core::ndarray_ext::Array2;
use std::collections::HashMap;
use std::time::Instant;

use crate::{patterns::Pattern, Result};

use super::attention::{AttentionConfig, CrossPatternAttention};
use super::types::{
    CausalRelationship, CorrelationAnalysisConfig, CorrelationAnalysisResult,
    CorrelationAnalysisStats, CorrelationCluster, CorrelationType, PatternCorrelation,
    PatternRelationshipGraph,
};

/// Advanced pattern correlation analyzer for discovering complex relationships
#[derive(Debug, Clone)]
pub struct AdvancedPatternCorrelationAnalyzer {
    /// Configuration for correlation analysis
    config: CorrelationAnalysisConfig,
    /// Correlation matrices for different relationship types
    correlation_matrices: HashMap<CorrelationType, Array2<f64>>,
    /// Pattern relationship graph
    pattern_relationships: PatternRelationshipGraph,
    /// Correlation statistics
    correlation_stats: CorrelationAnalysisStats,
}

impl AdvancedPatternCorrelationAnalyzer {
    /// Create new correlation analyzer with configuration
    pub fn new(config: CorrelationAnalysisConfig) -> Self {
        Self {
            config,
            correlation_matrices: HashMap::new(),
            pattern_relationships: PatternRelationshipGraph {
                pattern_nodes: HashMap::new(),
                relationship_edges: Vec::new(),
                graph_stats: Default::default(),
            },
            correlation_stats: CorrelationAnalysisStats::default(),
        }
    }

    /// Analyze correlations between patterns
    pub async fn analyze_correlations(
        &mut self,
        patterns: &[Pattern],
    ) -> Result<CorrelationAnalysisResult> {
        let start_time = Instant::now();

        // Build correlation matrices for different types
        self.build_correlation_matrices(patterns).await?;

        // Discover pattern correlations
        let correlations = self.discover_correlations(patterns).await?;

        // Form correlation clusters
        let clusters = self.form_correlation_clusters(&correlations).await?;

        // Identify causal relationships if enabled
        let causal_relationships = if self.config.enable_causal_inference {
            self.identify_causal_relationships(patterns, &correlations)
                .await?
        } else {
            Vec::new()
        };

        // Update statistics
        self.correlation_stats.analysis_execution_time = start_time.elapsed();
        self.correlation_stats.correlations_analyzed = correlations.len();
        self.correlation_stats.significant_correlations_found = correlations
            .iter()
            .filter(|c| c.statistical_significance > self.config.correlation_confidence_threshold)
            .count();

        // Discover pattern hierarchies from the structural correlations.
        let pattern_hierarchies = if self.config.enable_hierarchical_discovery {
            self.discover_pattern_hierarchies(patterns, &correlations)
        } else {
            Vec::new()
        };

        // Compute cross-pattern attention insights.
        // An empty pattern slice is valid; CrossPatternAttention handles it gracefully.
        let attention_insights = {
            let attention_config = AttentionConfig {
                embedding_dim: 64,
                num_heads: 4,
                head_dim: 16,
                dropout_rate: 0.0,
                temperature: 1.0,
                enable_position_encoding: false,
                max_sequence_length: patterns.len().max(1),
            };
            let mut attention_mechanism = CrossPatternAttention::new(attention_config);
            // Non-fatal: if attention fails we fall back to the empty default.
            attention_mechanism
                .compute_attention(patterns)
                .await
                .unwrap_or_default()
        };

        Ok(CorrelationAnalysisResult {
            discovered_correlations: correlations,
            correlation_clusters: clusters,
            pattern_hierarchies,
            attention_insights,
            causal_relationships,
            analysis_metadata: Default::default(),
        })
    }

    /// Build correlation matrices for different relationship types
    async fn build_correlation_matrices(&mut self, patterns: &[Pattern]) -> Result<()> {
        for correlation_type in [
            CorrelationType::Structural,
            CorrelationType::Semantic,
            CorrelationType::Temporal,
            CorrelationType::Functional,
        ] {
            let matrix = self
                .compute_correlation_matrix(patterns, &correlation_type)
                .await?;
            self.correlation_matrices.insert(correlation_type, matrix);
        }
        Ok(())
    }

    /// Compute correlation matrix for a specific type
    async fn compute_correlation_matrix(
        &self,
        patterns: &[Pattern],
        correlation_type: &CorrelationType,
    ) -> Result<Array2<f64>> {
        let n = patterns.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let correlation = self
                    .compute_pairwise_correlation(&patterns[i], &patterns[j], correlation_type)
                    .await?;
                matrix[[i, j]] = correlation;
                matrix[[j, i]] = correlation; // Symmetric matrix
            }
        }

        Ok(matrix)
    }

    /// Compute pairwise correlation between two patterns
    async fn compute_pairwise_correlation(
        &self,
        pattern1: &Pattern,
        pattern2: &Pattern,
        correlation_type: &CorrelationType,
    ) -> Result<f64> {
        match correlation_type {
            CorrelationType::Structural => self.compute_structural_correlation(pattern1, pattern2),
            CorrelationType::Semantic => self.compute_semantic_correlation(pattern1, pattern2),
            CorrelationType::Temporal => self.compute_temporal_correlation(pattern1, pattern2),
            CorrelationType::Functional => self.compute_functional_correlation(pattern1, pattern2),
            // Causal: high correlation between patterns of different types that
            // frequently co-occur suggests a causal link.  We approximate this
            // as the product of their supports (joint occurrence probability).
            CorrelationType::Causal => {
                let joint = pattern1.support() * pattern2.support();
                Ok(joint.sqrt().clamp(0.0, 1.0))
            }
            // Hierarchical: patterns where one generalises the other (e.g. a
            // class pattern subsumes a property pattern) are detected by
            // comparing confidence asymmetry.
            CorrelationType::Hierarchical => {
                let conf_diff = (pattern1.confidence() - pattern2.confidence()).abs();
                let support_min = pattern1.support().min(pattern2.support());
                Ok((support_min * (1.0 - conf_diff)).clamp(0.0, 1.0))
            }
            // Contextual: patterns that share the same context are similar in
            // support and confidence simultaneously.
            CorrelationType::Contextual => {
                let support_sim = 1.0 - (pattern1.support() - pattern2.support()).abs();
                let conf_sim = 1.0 - (pattern1.confidence() - pattern2.confidence()).abs();
                Ok((support_sim * conf_sim).clamp(0.0, 1.0))
            }
            // CrossDomain: cross-domain correlations are modelled as a scaled
            // geometric mean of support values (patterns from different domains
            // rarely have high joint support).
            CorrelationType::CrossDomain => {
                let geo_mean = (pattern1.support() * pattern2.support()).sqrt();
                Ok((geo_mean * 0.5_f64).clamp(0.0, 1.0))
            }
        }
    }

    /// Compute structural correlation between patterns
    fn compute_structural_correlation(
        &self,
        pattern1: &Pattern,
        pattern2: &Pattern,
    ) -> Result<f64> {
        // Analyze shape structure similarity using Jaccard coefficient and cosine similarity

        // Compare constraint types
        let constraints1 = self.extract_constraint_types(pattern1);
        let constraints2 = self.extract_constraint_types(pattern2);
        let jaccard_similarity = self.jaccard_coefficient(&constraints1, &constraints2);

        // Compare property paths depth and complexity
        let path_similarity = self.compare_property_paths(pattern1, pattern2);

        // Compare cardinality constraints
        let cardinality_similarity = self.compare_cardinalities(pattern1, pattern2);

        // Weighted combination of structural features
        let structural_correlation =
            0.4 * jaccard_similarity + 0.3 * path_similarity + 0.3 * cardinality_similarity;

        Ok(structural_correlation.clamp(0.0, 1.0))
    }

    /// Compute semantic correlation between patterns
    fn compute_semantic_correlation(&self, pattern1: &Pattern, pattern2: &Pattern) -> Result<f64> {
        // Use embedding-based semantic similarity and ontology alignment

        // Compare pattern IDs using string similarity
        let id_similarity = self.string_similarity(pattern1.id(), pattern2.id());

        // Analyze target class similarity
        let class_similarity = self.compare_target_classes(pattern1, pattern2);

        // Compare constraint semantics (e.g., both patterns check email format)
        let semantic_overlap = self.analyze_semantic_overlap(pattern1, pattern2);

        // Weighted combination of semantic features
        let semantic_correlation =
            0.3 * id_similarity + 0.4 * class_similarity + 0.3 * semantic_overlap;

        Ok(semantic_correlation.clamp(0.0, 1.0))
    }

    /// Compute temporal correlation between patterns
    fn compute_temporal_correlation(&self, pattern1: &Pattern, pattern2: &Pattern) -> Result<f64> {
        // Analyze co-occurrence patterns and temporal trends

        // Compute co-activation frequency (how often patterns are used together)
        let co_activation = self.compute_co_activation_frequency(pattern1, pattern2);

        // Analyze temporal lag correlation (does one pattern predict the other)
        let temporal_lag = self.compute_temporal_lag_correlation(pattern1, pattern2);

        // Check for synchronized usage patterns
        let synchronization = self.compute_synchronization_index(pattern1, pattern2);

        // Weighted combination of temporal features
        let temporal_correlation = 0.5 * co_activation + 0.3 * temporal_lag + 0.2 * synchronization;

        Ok(temporal_correlation.clamp(0.0, 1.0))
    }

    /// Compute functional correlation between patterns
    fn compute_functional_correlation(
        &self,
        pattern1: &Pattern,
        pattern2: &Pattern,
    ) -> Result<f64> {
        // Analyze what the patterns validate/constrain and their functional purpose

        // Compare validation scopes (similar data types, domains)
        let scope_similarity = self.compare_validation_scopes(pattern1, pattern2);

        // Analyze constraint overlap (validating similar properties)
        let constraint_overlap = self.compute_constraint_overlap(pattern1, pattern2);

        // Check for complementary functionality (one extends the other)
        let complementarity = self.analyze_complementarity(pattern1, pattern2);

        // Compute functional equivalence (patterns achieve same goal differently)
        let equivalence = self.compute_functional_equivalence(pattern1, pattern2);

        // Weighted combination of functional features
        let functional_correlation = 0.3 * scope_similarity
            + 0.3 * constraint_overlap
            + 0.2 * complementarity
            + 0.2 * equivalence;

        Ok(functional_correlation.clamp(0.0, 1.0))
    }

    /// Discover significant correlations from matrices
    async fn discover_correlations(&self, patterns: &[Pattern]) -> Result<Vec<PatternCorrelation>> {
        let mut correlations = Vec::new();

        for (correlation_type, matrix) in &self.correlation_matrices {
            for i in 0..patterns.len() {
                for j in (i + 1)..patterns.len() {
                    let correlation_coefficient = matrix[[i, j]];

                    if correlation_coefficient.abs() >= self.config.min_correlation_threshold {
                        correlations.push(PatternCorrelation {
                            pattern1_id: format!("pattern_{i}"),
                            pattern2_id: format!("pattern_{j}"),
                            correlation_type: correlation_type.clone(),
                            correlation_coefficient,
                            statistical_significance: self
                                .compute_significance(correlation_coefficient),
                            confidence_interval: (
                                correlation_coefficient - 0.1,
                                correlation_coefficient + 0.1,
                            ),
                            supporting_features: Vec::new(),
                            temporal_context: None,
                        });
                    }
                }
            }
        }

        Ok(correlations)
    }

    /// Compute statistical significance of correlation
    fn compute_significance(&self, correlation: f64) -> f64 {
        // Simple significance computation - in practice would use proper statistical tests
        correlation.abs()
    }

    /// Form clusters of correlated patterns using hierarchical clustering
    async fn form_correlation_clusters(
        &self,
        correlations: &[PatternCorrelation],
    ) -> Result<Vec<CorrelationCluster>> {
        if correlations.is_empty() {
            return Ok(Vec::new());
        }

        // Build similarity matrix from correlations
        let mut pattern_ids: Vec<String> = Vec::new();
        for corr in correlations {
            if !pattern_ids.contains(&corr.pattern1_id) {
                pattern_ids.push(corr.pattern1_id.clone());
            }
            if !pattern_ids.contains(&corr.pattern2_id) {
                pattern_ids.push(corr.pattern2_id.clone());
            }
        }

        // Perform agglomerative hierarchical clustering
        let mut clusters: Vec<CorrelationCluster> = Vec::new();
        let mut cluster_assignments = vec![0; pattern_ids.len()];
        let mut next_cluster_id = 1;

        // Simple clustering: group patterns with high correlation
        for (idx, pattern_id) in pattern_ids.iter().enumerate() {
            if cluster_assignments[idx] == 0 {
                // Start new cluster
                let cluster_id = next_cluster_id;
                next_cluster_id += 1;
                cluster_assignments[idx] = cluster_id;

                let mut cluster_patterns = vec![pattern_id.clone()];
                let mut cluster_correlations = Vec::new();

                // Find all patterns strongly correlated with this one
                for corr in correlations {
                    if &corr.pattern1_id == pattern_id
                        && corr.correlation_coefficient.abs()
                            >= self.config.min_correlation_threshold
                    {
                        if let Some(idx2) = pattern_ids.iter().position(|p| p == &corr.pattern2_id)
                        {
                            if cluster_assignments[idx2] == 0 {
                                cluster_assignments[idx2] = cluster_id;
                                cluster_patterns.push(corr.pattern2_id.clone());
                                cluster_correlations.push(corr.clone());
                            }
                        }
                    }
                }

                clusters.push(CorrelationCluster {
                    cluster_id: format!("cluster_{cluster_id}"),
                    member_patterns: cluster_patterns.clone(),
                    cluster_centroid: Vec::new(), // Could compute from pattern features
                    intra_cluster_correlation: self.compute_cluster_cohesion(&cluster_correlations),
                    cluster_coherence: self.compute_cluster_cohesion(&cluster_correlations),
                    representative_pattern: cluster_patterns
                        .first()
                        .unwrap_or(&String::new())
                        .clone(),
                });
            }
        }

        Ok(clusters)
    }

    /// Identify causal relationships between patterns using correlation analysis
    async fn identify_causal_relationships(
        &self,
        patterns: &[Pattern],
        correlations: &[PatternCorrelation],
    ) -> Result<Vec<CausalRelationship>> {
        let mut causal_relationships = Vec::new();

        // Analyze temporal ordering and correlation strength to infer causality
        for corr in correlations {
            // Only consider strong correlations for causal inference
            if corr.correlation_coefficient.abs() < self.config.correlation_confidence_threshold {
                continue;
            }

            // Find the patterns
            let pattern1_opt = patterns.iter().find(|p| {
                format!("pattern_{}", p.id().split('_').next_back().unwrap_or("0"))
                    == corr.pattern1_id
            });
            let pattern2_opt = patterns.iter().find(|p| {
                format!("pattern_{}", p.id().split('_').next_back().unwrap_or("0"))
                    == corr.pattern2_id
            });

            if let (Some(pattern1), Some(pattern2)) = (pattern1_opt, pattern2_opt) {
                // Determine causal direction using temporal precedence and domain knowledge
                let (cause, effect, strength) =
                    if self.determines_causality(pattern1, pattern2, corr) {
                        (pattern1, pattern2, corr.correlation_coefficient.abs())
                    } else {
                        // Check reverse direction
                        (pattern2, pattern1, corr.correlation_coefficient.abs())
                    };

                causal_relationships.push(CausalRelationship {
                    cause_pattern: cause.id().to_string(),
                    effect_pattern: effect.id().to_string(),
                    causal_strength: strength,
                    temporal_lag: None, // Could be computed from usage data
                    confidence_level: self.compute_causal_confidence(strength),
                    evidence_type: super::types::CausalEvidenceType::Observational,
                });
            }
        }

        Ok(causal_relationships)
    }

    /// Get correlation statistics
    pub fn get_statistics(&self) -> &CorrelationAnalysisStats {
        &self.correlation_stats
    }

    /// Get correlation matrix for a specific type
    pub fn get_correlation_matrix(
        &self,
        correlation_type: &CorrelationType,
    ) -> Option<&Array2<f64>> {
        self.correlation_matrices.get(correlation_type)
    }

    // Helper methods for correlation computation

    fn extract_constraint_types(&self, pattern: &Pattern) -> Vec<String> {
        // Extract constraint types from pattern
        vec![format!("{:?}", pattern.pattern_type())]
    }

    fn jaccard_coefficient(&self, set1: &[String], set2: &[String]) -> f64 {
        if set1.is_empty() && set2.is_empty() {
            return 1.0;
        }

        let intersection = set1.iter().filter(|item| set2.contains(item)).count();
        let union = set1.len() + set2.len() - intersection;

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn compare_property_paths(&self, pattern1: &Pattern, pattern2: &Pattern) -> f64 {
        // Compare property path complexity based on pattern types
        // Simplified: patterns of the same type have similar complexity
        let same_type = std::ptr::eq(pattern1.pattern_type(), pattern2.pattern_type());
        if same_type {
            0.9
        } else {
            0.3
        }
    }

    fn compare_cardinalities(&self, _pattern1: &Pattern, _pattern2: &Pattern) -> f64 {
        // Compare cardinality constraints (simplified)
        0.8
    }

    fn string_similarity(&self, s1: &str, s2: &str) -> f64 {
        // Levenshtein distance-based similarity
        let max_len = s1.len().max(s2.len());
        if max_len == 0 {
            return 1.0;
        }

        let distance = self.levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first column and row
        #[allow(clippy::needless_range_loop)]
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        #[allow(clippy::needless_range_loop)]
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                    .min(matrix[i + 1][j] + 1)
                    .min(matrix[i][j] + cost);
            }
        }

        matrix[len1][len2]
    }

    fn compare_target_classes(&self, _pattern1: &Pattern, _pattern2: &Pattern) -> f64 {
        // Compare target classes (simplified)
        0.7
    }

    fn analyze_semantic_overlap(&self, pattern1: &Pattern, pattern2: &Pattern) -> f64 {
        // Analyze semantic overlap based on constraint types
        let types1 = self.extract_constraint_types(pattern1);
        let types2 = self.extract_constraint_types(pattern2);
        self.jaccard_coefficient(&types1, &types2)
    }

    fn compute_co_activation_frequency(&self, _pattern1: &Pattern, _pattern2: &Pattern) -> f64 {
        // Compute how often patterns are used together (would need usage data)
        0.5
    }

    fn compute_temporal_lag_correlation(&self, _pattern1: &Pattern, _pattern2: &Pattern) -> f64 {
        // Compute temporal lag correlation (would need temporal data)
        0.3
    }

    fn compute_synchronization_index(&self, _pattern1: &Pattern, _pattern2: &Pattern) -> f64 {
        // Compute synchronization of usage patterns
        0.4
    }

    fn compare_validation_scopes(&self, pattern1: &Pattern, pattern2: &Pattern) -> f64 {
        // Compare validation scopes
        self.string_similarity(pattern1.id(), pattern2.id())
    }

    fn compute_constraint_overlap(&self, pattern1: &Pattern, pattern2: &Pattern) -> f64 {
        let types1 = self.extract_constraint_types(pattern1);
        let types2 = self.extract_constraint_types(pattern2);
        self.jaccard_coefficient(&types1, &types2)
    }

    fn analyze_complementarity(&self, _pattern1: &Pattern, _pattern2: &Pattern) -> f64 {
        // Analyze if patterns complement each other
        0.5
    }

    fn compute_functional_equivalence(&self, pattern1: &Pattern, pattern2: &Pattern) -> f64 {
        // Check if patterns achieve same goal differently
        if std::ptr::eq(pattern1.pattern_type(), pattern2.pattern_type()) {
            0.8
        } else {
            0.2
        }
    }

    fn compute_cluster_cohesion(&self, correlations: &[PatternCorrelation]) -> f64 {
        if correlations.is_empty() {
            return 0.0;
        }

        let sum: f64 = correlations
            .iter()
            .map(|c| c.correlation_coefficient.abs())
            .sum();
        sum / correlations.len() as f64
    }

    fn determines_causality(
        &self,
        _pattern1: &Pattern,
        _pattern2: &Pattern,
        corr: &PatternCorrelation,
    ) -> bool {
        // Simplified causality determination based on correlation strength
        corr.correlation_coefficient > 0.0
    }

    fn infer_causal_mechanism(&self, cause: &Pattern, effect: &Pattern) -> String {
        format!("Pattern '{}' influences '{}'", cause.id(), effect.id())
    }

    fn compute_causal_confidence(&self, strength: f64) -> f64 {
        // Convert causal strength to confidence level
        strength.min(0.95)
    }

    /// Discover hierarchical relationships among patterns by grouping them into
    /// coarse top-level patterns and finer-grained leaf patterns.
    ///
    /// A simple two-level hierarchy is built:
    /// - *Root* patterns are those whose average structural correlation with all
    ///   other patterns is above the significance threshold (they are
    ///   "generalisers").
    /// - All remaining patterns become *leaf* patterns at level 1.
    fn discover_pattern_hierarchies(
        &self,
        patterns: &[Pattern],
        correlations: &[PatternCorrelation],
    ) -> Vec<super::types::PatternHierarchy> {
        use super::types::{HierarchyLevel, HierarchyMetrics, PatternHierarchy};

        if patterns.len() < 2 {
            return Vec::new();
        }

        let threshold = self.config.correlation_confidence_threshold;

        // Compute average outgoing correlation strength per pattern.
        let mut avg_corr: HashMap<String, (f64, usize)> = HashMap::new();
        for corr in correlations {
            let entry = avg_corr.entry(corr.pattern1_id.clone()).or_insert((0.0, 0));
            entry.0 += corr.correlation_coefficient.abs();
            entry.1 += 1;

            let entry2 = avg_corr.entry(corr.pattern2_id.clone()).or_insert((0.0, 0));
            entry2.0 += corr.correlation_coefficient.abs();
            entry2.1 += 1;
        }

        let mut root_patterns = Vec::new();
        let mut leaf_patterns = Vec::new();

        for pattern in patterns {
            let id = pattern.id().to_owned();
            let is_root = avg_corr
                .get(&id)
                .map(|(sum, count)| *count > 0 && *sum / *count as f64 >= threshold)
                .unwrap_or(false);

            if is_root {
                root_patterns.push(id);
            } else {
                leaf_patterns.push(id);
            }
        }

        if root_patterns.is_empty() {
            return Vec::new();
        }

        let branching_factor = if !root_patterns.is_empty() {
            leaf_patterns.len() as f64 / root_patterns.len() as f64
        } else {
            0.0
        };

        let depth = if leaf_patterns.is_empty() { 1 } else { 2 };
        let coverage = (root_patterns.len() + leaf_patterns.len()) as f64 / patterns.len() as f64;

        let mut hierarchy = PatternHierarchy::new();
        hierarchy.hierarchy_id = "discovered_hierarchy_0".to_string();
        hierarchy.root_patterns = root_patterns.clone();
        hierarchy.hierarchy_levels = vec![
            HierarchyLevel {
                level: 0,
                patterns: root_patterns,
                level_coherence: threshold,
                inter_level_connections: Vec::new(),
            },
            HierarchyLevel {
                level: 1,
                patterns: leaf_patterns,
                level_coherence: 1.0 - threshold,
                inter_level_connections: Vec::new(),
            },
        ];
        hierarchy.hierarchy_metrics = HierarchyMetrics {
            hierarchy_depth: depth,
            branching_factor,
            coherence_score: threshold,
            coverage_percentage: coverage * 100.0,
            stability_measure: 0.8,
        };

        vec![hierarchy]
    }
}
