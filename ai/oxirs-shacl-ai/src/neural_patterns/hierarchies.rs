//! Pattern hierarchy discovery and analysis for SHACL shape relationships

use std::collections::{HashMap, HashSet};

use crate::{patterns::Pattern, Result};

use super::types::{
    CorrelationType, HierarchyLevel, HierarchyMetrics, PatternCorrelation, PatternHierarchy,
    PatternRelationshipGraph,
};

/// Pattern hierarchy analyzer for discovering structural relationships
#[derive(Debug)]
pub struct PatternHierarchyAnalyzer {
    /// Configuration for hierarchy analysis
    config: HierarchyAnalysisConfig,
    /// Discovered hierarchies
    hierarchies: Vec<PatternHierarchy>,
    /// Hierarchy building statistics
    statistics: HierarchyAnalysisStats,
}

/// Configuration for hierarchy analysis
#[derive(Debug, Clone)]
pub struct HierarchyAnalysisConfig {
    /// Maximum depth of hierarchies to discover
    pub max_hierarchy_depth: usize,
    /// Minimum confidence for parent-child relationships
    pub min_relationship_confidence: f64,
    /// Maximum branching factor at each level
    pub max_branching_factor: usize,
    /// Enable multi-level hierarchy discovery
    pub enable_multi_level: bool,
    /// Coherence threshold for hierarchy validation
    pub coherence_threshold: f64,
}

impl Default for HierarchyAnalysisConfig {
    fn default() -> Self {
        Self {
            max_hierarchy_depth: 6,
            min_relationship_confidence: 0.7,
            max_branching_factor: 10,
            enable_multi_level: true,
            coherence_threshold: 0.6,
        }
    }
}

/// Statistics for hierarchy analysis
#[derive(Debug, Clone, Default)]
pub struct HierarchyAnalysisStats {
    pub hierarchies_discovered: usize,
    pub total_levels_analyzed: usize,
    pub average_hierarchy_depth: f64,
    pub average_branching_factor: f64,
    pub hierarchy_coverage: f64,
    pub coherence_scores: Vec<f64>,
}

impl PatternHierarchyAnalyzer {
    /// Create new hierarchy analyzer
    pub fn new(config: HierarchyAnalysisConfig) -> Self {
        Self {
            config,
            hierarchies: Vec::new(),
            statistics: HierarchyAnalysisStats::default(),
        }
    }

    /// Discover pattern hierarchies from correlations
    pub async fn discover_hierarchies(
        &mut self,
        patterns: &[Pattern],
        correlations: &[PatternCorrelation],
        _relationship_graph: &PatternRelationshipGraph,
    ) -> Result<Vec<PatternHierarchy>> {
        // Build hierarchy structure from correlations
        let hierarchy_candidates = self
            .build_hierarchy_candidates(patterns, correlations)
            .await?;

        // Validate and refine hierarchies
        let validated_hierarchies = self.validate_hierarchies(hierarchy_candidates).await?;

        // Compute hierarchy metrics
        let hierarchies_with_metrics = self
            .compute_hierarchy_metrics(validated_hierarchies)
            .await?;

        // Update statistics
        self.update_statistics(&hierarchies_with_metrics);

        self.hierarchies = hierarchies_with_metrics.clone();
        Ok(hierarchies_with_metrics)
    }

    /// Build initial hierarchy candidates from correlations
    async fn build_hierarchy_candidates(
        &self,
        _patterns: &[Pattern],
        correlations: &[PatternCorrelation],
    ) -> Result<Vec<PatternHierarchy>> {
        let mut candidates = Vec::new();

        // Group correlations by hierarchical relationships
        let hierarchical_correlations: Vec<&PatternCorrelation> = correlations
            .iter()
            .filter(|c| c.correlation_type == CorrelationType::Hierarchical)
            .collect();

        // Build parent-child relationship map
        let parent_child_map = self.build_parent_child_map(&hierarchical_correlations)?;

        // Find root patterns (those with no parents)
        let root_patterns = self.find_root_patterns(&parent_child_map)?;

        // Build hierarchies starting from each root
        for root_pattern in root_patterns {
            let hierarchy = self
                .build_hierarchy_from_root(&root_pattern, &parent_child_map)
                .await?;
            candidates.push(hierarchy);
        }

        Ok(candidates)
    }

    /// Build parent-child relationship map from correlations
    fn build_parent_child_map(
        &self,
        correlations: &[&PatternCorrelation],
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut parent_child_map: HashMap<String, Vec<String>> = HashMap::new();

        for correlation in correlations {
            if correlation.correlation_coefficient >= self.config.min_relationship_confidence {
                // Determine parent-child direction based on correlation properties
                let (parent, child) = self.determine_parent_child_relationship(correlation)?;

                parent_child_map.entry(parent).or_default().push(child);
            }
        }

        Ok(parent_child_map)
    }

    /// Determine which pattern is parent and which is child.
    ///
    /// Heuristic: a *parent* pattern is one that is referenced (i.e. more
    /// general / higher in the hierarchy).  We use three signals in order:
    ///
    /// 1. **Correlation direction** – `Hierarchical` correlation type implies
    ///    the ordering already encoded in `(pattern1_id, pattern2_id)`.
    /// 2. **Confidence** – the higher-confidence endpoint is more likely to be
    ///    the general/parent concept.
    /// 3. **Lexicographic fallback** – purely for determinism when signals
    ///    are equal.
    fn determine_parent_child_relationship(
        &self,
        correlation: &PatternCorrelation,
    ) -> Result<(String, String)> {
        // Signal 1: explicit Hierarchical correlation preserves semantic direction.
        if matches!(correlation.correlation_type, CorrelationType::Hierarchical) {
            return Ok((
                correlation.pattern1_id.clone(),
                correlation.pattern2_id.clone(),
            ));
        }

        // Signal 2: the lower bound of the confidence interval indicates the
        // more stable/general (parent) side of the relationship.
        let (ci_lo, ci_hi) = correlation.confidence_interval;
        if (ci_hi - ci_lo).abs() > 1e-9 {
            // Wider interval → less certain → treat as child
            if ci_hi - ci_lo > 0.2 {
                // Asymmetry: p1 is child, p2 is parent
                return Ok((
                    correlation.pattern2_id.clone(),
                    correlation.pattern1_id.clone(),
                ));
            }
        }

        // Signal 3: lexicographic determinism
        if correlation.pattern1_id <= correlation.pattern2_id {
            Ok((
                correlation.pattern1_id.clone(),
                correlation.pattern2_id.clone(),
            ))
        } else {
            Ok((
                correlation.pattern2_id.clone(),
                correlation.pattern1_id.clone(),
            ))
        }
    }

    /// Find root patterns (those with no parents)
    fn find_root_patterns(
        &self,
        parent_child_map: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<String>> {
        let mut all_children: HashSet<String> = HashSet::new();
        let mut all_parents: HashSet<String> = HashSet::new();

        for (parent, children) in parent_child_map {
            all_parents.insert(parent.clone());
            for child in children {
                all_children.insert(child.clone());
            }
        }

        // Root patterns are those that are parents but not children
        let roots: Vec<String> = all_parents.difference(&all_children).cloned().collect();

        Ok(roots)
    }

    /// Build hierarchy starting from a root pattern
    async fn build_hierarchy_from_root(
        &self,
        root_pattern: &str,
        parent_child_map: &HashMap<String, Vec<String>>,
    ) -> Result<PatternHierarchy> {
        let hierarchy_id = format!("hierarchy_{root_pattern}");
        let mut hierarchy_levels = Vec::new();

        // Build levels using breadth-first traversal
        let mut current_level_patterns = vec![root_pattern.to_string()];
        let mut level = 0;

        while !current_level_patterns.is_empty() && level < self.config.max_hierarchy_depth {
            let mut next_level_patterns = Vec::new();

            // Find children of current level patterns
            for pattern in &current_level_patterns {
                if let Some(children) = parent_child_map.get(pattern) {
                    next_level_patterns.extend_from_slice(children);
                }
            }

            // Create hierarchy level
            let level_coherence = self
                .compute_level_coherence(&current_level_patterns)
                .await?;
            let inter_level_connections = self.compute_inter_level_connections(
                &current_level_patterns,
                &next_level_patterns,
                parent_child_map,
            )?;

            hierarchy_levels.push(HierarchyLevel {
                level,
                patterns: current_level_patterns.clone(),
                level_coherence,
                inter_level_connections,
            });

            current_level_patterns = next_level_patterns;
            level += 1;
        }

        Ok(PatternHierarchy {
            hierarchy_id,
            root_patterns: vec![root_pattern.to_string()],
            hierarchy_levels,
            hierarchy_metrics: HierarchyMetrics {
                hierarchy_depth: 0,
                branching_factor: 0.0,
                coherence_score: 0.0,
                coverage_percentage: 0.0,
                stability_measure: 0.0,
            },
        })
    }

    /// Compute coherence within a hierarchy level.
    ///
    /// Coherence measures how internally consistent the patterns at a given
    /// level are.  We approximate it by examining the proportion of pattern-id
    /// pairs that share a common prefix segment (e.g. "schema_", "temporal_"),
    /// which is a lightweight structural proxy for semantic similarity.
    ///
    /// Returns a value in [0, 1]: 1.0 means all patterns have matching
    /// prefixes; 0.0 means no shared structure.
    async fn compute_level_coherence(&self, patterns: &[String]) -> Result<f64> {
        let n = patterns.len();
        if n < 2 {
            // A singleton or empty level is trivially coherent
            return Ok(1.0);
        }

        // Extract the first "_"-delimited segment of each pattern id as its
        // "family" identifier, collected upfront to avoid lifetime issues.
        let families: Vec<&str> = patterns
            .iter()
            .map(|id| id.find('_').map(|pos| &id[..pos]).unwrap_or(id.as_str()))
            .collect();

        let mut shared_pairs = 0usize;
        let mut total_pairs = 0usize;
        for i in 0..n {
            for j in i + 1..n {
                total_pairs += 1;
                if families[i] == families[j] {
                    shared_pairs += 1;
                }
            }
        }

        let raw_coherence = shared_pairs as f64 / total_pairs as f64;

        // Blend with a size-based penalty: very large levels are harder to
        // maintain as coherent regardless of prefix matching.
        let size_factor =
            1.0 - ((n as f64 - 2.0) / (n as f64 + self.config.max_branching_factor as f64));
        let coherence = (raw_coherence * 0.7 + size_factor * 0.3).clamp(0.0, 1.0);

        Ok(coherence)
    }

    /// Compute connections between hierarchy levels
    fn compute_inter_level_connections(
        &self,
        current_level: &[String],
        next_level: &[String],
        parent_child_map: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<(String, String, f64)>> {
        let mut connections = Vec::new();

        for parent in current_level {
            if let Some(children) = parent_child_map.get(parent) {
                for child in children {
                    if next_level.contains(child) {
                        // Connection strength is inversely proportional to the
                        // edit distance between the two IDs (normalised to [0.5, 1]).
                        // Longer common prefix ⇒ stronger structural connection.
                        let common_prefix_len = parent
                            .chars()
                            .zip(child.chars())
                            .take_while(|(a, b)| a == b)
                            .count();
                        let max_len = parent.len().max(child.len()).max(1);
                        let prefix_ratio = common_prefix_len as f64 / max_len as f64;
                        // Map [0, 1] → [0.5, 1.0] so that even unrelated names still
                        // have a moderate connection strength due to the proven
                        // structural parent-child relationship.
                        let connection_strength = 0.5 + 0.5 * prefix_ratio;
                        connections.push((parent.clone(), child.clone(), connection_strength));
                    }
                }
            }
        }

        Ok(connections)
    }

    /// Validate discovered hierarchies
    async fn validate_hierarchies(
        &self,
        candidates: Vec<PatternHierarchy>,
    ) -> Result<Vec<PatternHierarchy>> {
        let mut validated = Vec::new();

        for hierarchy in candidates {
            if self.is_valid_hierarchy(&hierarchy).await? {
                validated.push(hierarchy);
            }
        }

        Ok(validated)
    }

    /// Check if a hierarchy meets validation criteria
    async fn is_valid_hierarchy(&self, hierarchy: &PatternHierarchy) -> Result<bool> {
        // Check minimum depth
        if hierarchy.hierarchy_levels.len() < 2 {
            return Ok(false);
        }

        // Check coherence threshold
        let average_coherence = hierarchy
            .hierarchy_levels
            .iter()
            .map(|level| level.level_coherence)
            .sum::<f64>()
            / hierarchy.hierarchy_levels.len() as f64;

        if average_coherence < self.config.coherence_threshold {
            return Ok(false);
        }

        // Check branching factor constraint
        for level in &hierarchy.hierarchy_levels {
            if level.patterns.len() > self.config.max_branching_factor {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Compute metrics for validated hierarchies
    async fn compute_hierarchy_metrics(
        &self,
        mut hierarchies: Vec<PatternHierarchy>,
    ) -> Result<Vec<PatternHierarchy>> {
        for hierarchy in &mut hierarchies {
            hierarchy.hierarchy_metrics = self.compute_metrics_for_hierarchy(hierarchy).await?;
        }

        Ok(hierarchies)
    }

    /// Compute metrics for a single hierarchy
    async fn compute_metrics_for_hierarchy(
        &self,
        hierarchy: &PatternHierarchy,
    ) -> Result<HierarchyMetrics> {
        let depth = hierarchy.hierarchy_levels.len();

        let branching_factor = if depth > 1 {
            hierarchy
                .hierarchy_levels
                .iter()
                .take(depth - 1) // Don't count leaf level
                .map(|level| level.patterns.len())
                .sum::<usize>() as f64
                / (depth - 1) as f64
        } else {
            0.0
        };

        let coherence_score = hierarchy
            .hierarchy_levels
            .iter()
            .map(|level| level.level_coherence)
            .sum::<f64>()
            / depth as f64;

        // Coverage: fraction of total known patterns captured by this hierarchy.
        // We estimate it as the ratio of patterns in the hierarchy to a
        // representative "universe" based on the deepest branching level.
        let total_patterns_in_hierarchy: usize = hierarchy
            .hierarchy_levels
            .iter()
            .map(|l| l.patterns.len())
            .sum();
        let max_level_size = hierarchy
            .hierarchy_levels
            .iter()
            .map(|l| l.patterns.len())
            .max()
            .unwrap_or(1);
        // A fully balanced tree of depth `depth` with branching `max_level_size`
        // would have up to max_level_size^depth leaf patterns.  Coverage is the
        // ratio of what we found to this theoretical maximum.
        let theoretical_max = (max_level_size as f64).powi(depth as i32).max(1.0);
        let coverage_percentage =
            (total_patterns_in_hierarchy as f64 / theoretical_max).clamp(0.0, 1.0);

        // Stability: reflects how uniform the level sizes are.
        // A perfectly balanced tree has stability 1.0; high variance → lower stability.
        let mean_level_size = total_patterns_in_hierarchy as f64 / depth as f64;
        let variance: f64 = hierarchy
            .hierarchy_levels
            .iter()
            .map(|l| {
                let diff = l.patterns.len() as f64 - mean_level_size;
                diff * diff
            })
            .sum::<f64>()
            / depth as f64;
        let stability_measure =
            (1.0 / (1.0 + variance.sqrt() / (mean_level_size + 1.0))).clamp(0.0, 1.0);

        Ok(HierarchyMetrics {
            hierarchy_depth: depth,
            branching_factor,
            coherence_score,
            coverage_percentage,
            stability_measure,
        })
    }

    /// Update analysis statistics
    fn update_statistics(&mut self, hierarchies: &[PatternHierarchy]) {
        self.statistics.hierarchies_discovered = hierarchies.len();
        self.statistics.total_levels_analyzed =
            hierarchies.iter().map(|h| h.hierarchy_levels.len()).sum();

        if !hierarchies.is_empty() {
            self.statistics.average_hierarchy_depth = hierarchies
                .iter()
                .map(|h| h.hierarchy_levels.len() as f64)
                .sum::<f64>()
                / hierarchies.len() as f64;

            self.statistics.average_branching_factor = hierarchies
                .iter()
                .map(|h| h.hierarchy_metrics.branching_factor)
                .sum::<f64>()
                / hierarchies.len() as f64;

            self.statistics.coherence_scores = hierarchies
                .iter()
                .map(|h| h.hierarchy_metrics.coherence_score)
                .collect();
        }
    }

    /// Get discovered hierarchies
    pub fn get_hierarchies(&self) -> &[PatternHierarchy] {
        &self.hierarchies
    }

    /// Get analysis statistics
    pub fn get_statistics(&self) -> &HierarchyAnalysisStats {
        &self.statistics
    }

    /// Find patterns at a specific hierarchy level
    pub fn find_patterns_at_level(&self, hierarchy_id: &str, level: usize) -> Option<Vec<String>> {
        for hierarchy in &self.hierarchies {
            if hierarchy.hierarchy_id == hierarchy_id && level < hierarchy.hierarchy_levels.len() {
                return Some(hierarchy.hierarchy_levels[level].patterns.clone());
            }
        }
        None
    }

    /// Get parent patterns for a given pattern
    pub fn get_parent_patterns(&self, pattern_id: &str) -> Vec<String> {
        let mut parents = Vec::new();

        for hierarchy in &self.hierarchies {
            for (level_idx, level) in hierarchy.hierarchy_levels.iter().enumerate() {
                if level.patterns.contains(&pattern_id.to_string()) && level_idx > 0 {
                    // Pattern found, look for parents in previous level
                    let parent_level = &hierarchy.hierarchy_levels[level_idx - 1];
                    for connection in &level.inter_level_connections {
                        if connection.1 == pattern_id {
                            parents.push(connection.0.clone());
                        }
                    }
                }
            }
        }

        parents
    }

    /// Get child patterns for a given pattern
    pub fn get_child_patterns(&self, pattern_id: &str) -> Vec<String> {
        let mut children = Vec::new();

        for hierarchy in &self.hierarchies {
            for (level_idx, level) in hierarchy.hierarchy_levels.iter().enumerate() {
                if level.patterns.contains(&pattern_id.to_string())
                    && level_idx < hierarchy.hierarchy_levels.len() - 1
                {
                    // Pattern found, look for children in next level
                    let child_level = &hierarchy.hierarchy_levels[level_idx + 1];
                    for connection in &child_level.inter_level_connections {
                        if connection.0 == pattern_id {
                            children.push(connection.1.clone());
                        }
                    }
                }
            }
        }

        children
    }
}
