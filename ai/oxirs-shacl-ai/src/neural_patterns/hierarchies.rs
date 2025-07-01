//! Pattern hierarchy discovery and analysis for SHACL shape relationships

use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::{patterns::Pattern, Result, ShaclAiError};

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
        relationship_graph: &PatternRelationshipGraph,
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
        patterns: &[Pattern],
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

                parent_child_map
                    .entry(parent)
                    .or_insert_with(Vec::new)
                    .push(child);
            }
        }

        Ok(parent_child_map)
    }

    /// Determine which pattern is parent and which is child
    fn determine_parent_child_relationship(
        &self,
        correlation: &PatternCorrelation,
    ) -> Result<(String, String)> {
        // TODO: Implement sophisticated parent-child determination logic
        // This could be based on pattern complexity, specificity, etc.

        // For now, use lexicographic ordering as a placeholder
        if correlation.pattern1_id < correlation.pattern2_id {
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
        let hierarchy_id = format!("hierarchy_{}", root_pattern);
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

    /// Compute coherence within a hierarchy level
    async fn compute_level_coherence(&self, patterns: &[String]) -> Result<f64> {
        // TODO: Implement proper coherence computation
        // This could measure semantic similarity, structural similarity, etc.
        Ok(0.8)
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
                        // TODO: Compute actual connection strength
                        let connection_strength = 0.9;
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

        // TODO: Implement proper coverage and stability measures
        let coverage_percentage = 0.85;
        let stability_measure = 0.9;

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
            if hierarchy.hierarchy_id == hierarchy_id {
                if level < hierarchy.hierarchy_levels.len() {
                    return Some(hierarchy.hierarchy_levels[level].patterns.clone());
                }
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
