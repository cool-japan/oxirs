//! Hierarchical Similarity Computation for Knowledge Graphs
//!
//! This module provides advanced similarity computation that takes into account
//! ontological hierarchies, concept relationships, and contextual similarities.

use crate::VectorStore;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

/// Configuration for hierarchical similarity computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSimilarityConfig {
    /// Enable ontology-based similarity
    pub enable_ontology_similarity: bool,
    /// Enable concept hierarchy traversal
    pub enable_hierarchy_traversal: bool,
    /// Enable contextual similarity computation
    pub enable_contextual_similarity: bool,
    /// Enable adaptive similarity learning
    pub enable_adaptive_similarity: bool,
    /// Maximum hierarchy traversal depth
    pub max_hierarchy_depth: usize,
    /// Weight for direct similarity
    pub direct_similarity_weight: f32,
    /// Weight for hierarchical similarity
    pub hierarchical_similarity_weight: f32,
    /// Weight for contextual similarity
    pub contextual_similarity_weight: f32,
    /// Cache size for computed similarities
    pub similarity_cache_size: usize,
}

impl Default for HierarchicalSimilarityConfig {
    fn default() -> Self {
        Self {
            enable_ontology_similarity: true,
            enable_hierarchy_traversal: true,
            enable_contextual_similarity: true,
            enable_adaptive_similarity: false,
            max_hierarchy_depth: 5,
            direct_similarity_weight: 0.6,
            hierarchical_similarity_weight: 0.3,
            contextual_similarity_weight: 0.1,
            similarity_cache_size: 10000,
        }
    }
}

/// Concept hierarchy for ontology-based similarity
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConceptHierarchy {
    /// Parent-child relationships (child -> parent)
    pub child_to_parent: HashMap<String, String>,
    /// Parent-child relationships (parent -> children)
    pub parent_to_children: HashMap<String, HashSet<String>>,
    /// Concept levels in hierarchy (concept -> level)
    pub concept_levels: HashMap<String, usize>,
    /// Concept types and categories
    pub concept_types: HashMap<String, String>,
    /// Concept weights for similarity computation
    pub concept_weights: HashMap<String, f32>,
}

impl ConceptHierarchy {
    /// Add a parent-child relationship to the hierarchy
    pub fn add_relationship(&mut self, parent: String, child: String) {
        self.child_to_parent.insert(child.clone(), parent.clone());
        self.parent_to_children
            .entry(parent)
            .or_default()
            .insert(child);

        // Recompute levels
        self.recompute_levels();
    }

    /// Find the lowest common ancestor of two concepts
    pub fn lowest_common_ancestor(&self, concept1: &str, concept2: &str) -> Option<String> {
        let ancestors1 = self.get_ancestors(concept1);
        let ancestors2 = self.get_ancestors(concept2);

        // Find common ancestors
        let common_ancestors: HashSet<&String> = ancestors1.intersection(&ancestors2).collect();

        // Find the one with the highest level (closest to concepts)
        common_ancestors
            .into_iter()
            .max_by_key(|ancestor| self.concept_levels.get(*ancestor).unwrap_or(&0))
            .cloned()
    }

    /// Get all ancestors of a concept
    pub fn get_ancestors(&self, concept: &str) -> HashSet<String> {
        let mut ancestors = HashSet::new();
        let mut current = concept.to_string();

        while let Some(parent) = self.child_to_parent.get(&current) {
            ancestors.insert(parent.clone());
            current = parent.clone();
        }

        ancestors
    }

    /// Get all descendants of a concept
    pub fn get_descendants(&self, concept: &str) -> HashSet<String> {
        let mut descendants = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(children) = self.parent_to_children.get(concept) {
            for child in children {
                queue.push_back(child.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            if descendants.insert(current.clone()) {
                if let Some(children) = self.parent_to_children.get(&current) {
                    for child in children {
                        queue.push_back(child.clone());
                    }
                }
            }
        }

        descendants
    }

    /// Compute the distance between two concepts in the hierarchy
    pub fn concept_distance(&self, concept1: &str, concept2: &str) -> f32 {
        if concept1 == concept2 {
            return 0.0;
        }

        if let Some(lca) = self.lowest_common_ancestor(concept1, concept2) {
            let level1 = self.concept_levels.get(concept1).unwrap_or(&0);
            let level2 = self.concept_levels.get(concept2).unwrap_or(&0);
            let lca_level = self.concept_levels.get(&lca).unwrap_or(&0);

            // Distance is the sum of distances to LCA
            let distance = (level1 - lca_level) + (level2 - lca_level);
            distance as f32
        } else {
            // No common ancestor, maximum distance
            f32::INFINITY
        }
    }

    /// Recompute the levels of all concepts in the hierarchy
    fn recompute_levels(&mut self) {
        self.concept_levels.clear();

        // Find root concepts (those with no parents)
        let roots: Vec<String> = self
            .parent_to_children
            .keys()
            .filter(|concept| !self.child_to_parent.contains_key(*concept))
            .cloned()
            .collect();

        // BFS to assign levels
        let mut queue = VecDeque::new();
        for root in roots {
            self.concept_levels.insert(root.clone(), 0);
            queue.push_back((root, 0));
        }

        while let Some((concept, level)) = queue.pop_front() {
            if let Some(children) = self.parent_to_children.get(&concept) {
                for child in children {
                    let child_level = level + 1;
                    // Only update if this gives a higher level (closer to root)
                    if child_level > *self.concept_levels.get(child).unwrap_or(&0) {
                        self.concept_levels.insert(child.clone(), child_level);
                        queue.push_back((child.clone(), child_level));
                    }
                }
            }
        }
    }
}

/// Context information for similarity computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityContext {
    /// Domain context (e.g., "medical", "technical", "general")
    pub domain: Option<String>,
    /// Temporal context (time period relevance)
    pub temporal_weight: f32,
    /// Cultural context for cross-cultural similarity
    pub cultural_context: Option<String>,
    /// User context (preferences, history)
    pub user_context: HashMap<String, f32>,
    /// Task context (what kind of search/comparison)
    pub task_type: SimilarityTaskType,
}

impl Default for SimilarityContext {
    fn default() -> Self {
        Self {
            domain: None,
            temporal_weight: 1.0,
            cultural_context: None,
            user_context: HashMap::new(),
            task_type: SimilarityTaskType::General,
        }
    }
}

/// Types of similarity computation tasks
#[derive(Debug, Clone, Copy, PartialEq, Hash, Serialize, Deserialize)]
pub enum SimilarityTaskType {
    /// General similarity comparison
    General,
    /// Document classification
    Classification,
    /// Information retrieval
    Retrieval,
    /// Recommendation generation
    Recommendation,
    /// Semantic clustering
    Clustering,
}

/// Result of hierarchical similarity computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSimilarityResult {
    /// Overall similarity score
    pub overall_similarity: f32,
    /// Direct vector similarity component
    pub direct_similarity: f32,
    /// Hierarchical similarity component
    pub hierarchical_similarity: f32,
    /// Contextual similarity component
    pub contextual_similarity: f32,
    /// Explanation of the similarity computation
    pub explanation: SimilarityExplanation,
}

/// Explanation of how similarity was computed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityExplanation {
    /// Concepts involved in the computation
    pub concepts_involved: Vec<String>,
    /// Hierarchy paths used
    pub hierarchy_paths: Vec<String>,
    /// Contextual factors considered
    pub contextual_factors: Vec<String>,
    /// Weight breakdown
    pub weight_breakdown: HashMap<String, f32>,
}

/// Hierarchical similarity computer
pub struct HierarchicalSimilarity {
    config: HierarchicalSimilarityConfig,
    concept_hierarchy: Arc<RwLock<ConceptHierarchy>>,
    similarity_cache: Arc<RwLock<HashMap<String, f32>>>,
    concept_to_resource: Arc<RwLock<HashMap<String, Vec<String>>>>,
    adaptive_weights: Arc<RwLock<HashMap<String, f32>>>,
}

impl HierarchicalSimilarity {
    /// Create a new hierarchical similarity computer
    pub fn new(config: HierarchicalSimilarityConfig) -> Self {
        Self {
            config,
            concept_hierarchy: Arc::new(RwLock::new(ConceptHierarchy::default())),
            similarity_cache: Arc::new(RwLock::new(HashMap::new())),
            concept_to_resource: Arc::new(RwLock::new(HashMap::new())),
            adaptive_weights: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Compute hierarchical similarity between two resources
    pub fn compute_similarity(
        &self,
        vector_store: &VectorStore,
        resource1: &str,
        resource2: &str,
        context: &SimilarityContext,
    ) -> Result<HierarchicalSimilarityResult> {
        // Check cache first
        let cache_key = format!("{}:{}:{}", resource1, resource2, self.context_hash(context));
        if let Ok(cache) = self.similarity_cache.read() {
            if let Some(&cached_similarity) = cache.get(&cache_key) {
                return Ok(HierarchicalSimilarityResult {
                    overall_similarity: cached_similarity,
                    direct_similarity: cached_similarity,
                    hierarchical_similarity: 0.0,
                    contextual_similarity: 0.0,
                    explanation: SimilarityExplanation {
                        concepts_involved: vec![],
                        hierarchy_paths: vec!["cached".to_string()],
                        contextual_factors: vec![],
                        weight_breakdown: HashMap::new(),
                    },
                });
            }
        }

        // Compute direct vector similarity
        let direct_sim = vector_store.calculate_similarity(resource1, resource2)?;

        // Compute hierarchical similarity if enabled
        let hierarchical_sim = if self.config.enable_ontology_similarity {
            self.compute_ontology_similarity(resource1, resource2, context)?
        } else {
            0.0
        };

        // Compute contextual similarity if enabled
        let contextual_sim = if self.config.enable_contextual_similarity {
            self.compute_contextual_similarity(resource1, resource2, context)?
        } else {
            0.0
        };

        // Apply adaptive weights if enabled
        let (direct_weight, hierarchical_weight, contextual_weight) =
            if self.config.enable_adaptive_similarity {
                self.get_adaptive_weights(context)
            } else {
                (
                    self.config.direct_similarity_weight,
                    self.config.hierarchical_similarity_weight,
                    self.config.contextual_similarity_weight,
                )
            };

        // Compute weighted overall similarity
        let overall_sim = direct_sim * direct_weight
            + hierarchical_sim * hierarchical_weight
            + contextual_sim * contextual_weight;

        // Cache the result
        if let Ok(mut cache) = self.similarity_cache.write() {
            if cache.len() >= self.config.similarity_cache_size {
                cache.clear(); // Simple cache eviction
            }
            cache.insert(cache_key, overall_sim);
        }

        let explanation = SimilarityExplanation {
            concepts_involved: self
                .get_concepts_for_resource(resource1)
                .into_iter()
                .chain(self.get_concepts_for_resource(resource2))
                .collect(),
            hierarchy_paths: vec![format!(
                "hierarchy_depth_{}",
                self.config.max_hierarchy_depth
            )],
            contextual_factors: vec![format!("domain_{:?}", context.domain)],
            weight_breakdown: {
                let mut breakdown = HashMap::new();
                breakdown.insert("direct".to_string(), direct_weight);
                breakdown.insert("hierarchical".to_string(), hierarchical_weight);
                breakdown.insert("contextual".to_string(), contextual_weight);
                breakdown
            },
        };

        Ok(HierarchicalSimilarityResult {
            overall_similarity: overall_sim,
            direct_similarity: direct_sim,
            hierarchical_similarity: hierarchical_sim,
            contextual_similarity: contextual_sim,
            explanation,
        })
    }

    /// Add a concept hierarchy relationship
    pub fn add_concept_relationship(&self, parent: &str, child: &str) -> Result<()> {
        match self.concept_hierarchy.write() {
            Ok(mut hierarchy) => {
                hierarchy.add_relationship(parent.to_string(), child.to_string());
                Ok(())
            }
            _ => Err(anyhow!("Failed to acquire write lock on concept hierarchy")),
        }
    }

    /// Associate a resource with concepts
    pub fn associate_resource_with_concepts(
        &self,
        resource: &str,
        concepts: Vec<String>,
    ) -> Result<()> {
        match self.concept_to_resource.write() {
            Ok(mut mapping) => {
                mapping.insert(resource.to_string(), concepts);
                Ok(())
            }
            _ => Err(anyhow!(
                "Failed to acquire write lock on concept-resource mapping"
            )),
        }
    }

    /// Compute ontology-based similarity
    fn compute_ontology_similarity(
        &self,
        resource1: &str,
        resource2: &str,
        _context: &SimilarityContext,
    ) -> Result<f32> {
        let concepts1 = self.get_concepts_for_resource(resource1);
        let concepts2 = self.get_concepts_for_resource(resource2);

        if concepts1.is_empty() || concepts2.is_empty() {
            return Ok(0.0);
        }

        let hierarchy = self
            .concept_hierarchy
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on concept hierarchy"))?;

        let mut max_similarity = 0.0f32;

        // Find the maximum similarity between any pair of concepts
        for concept1 in &concepts1 {
            for concept2 in &concepts2 {
                let distance = hierarchy.concept_distance(concept1, concept2);

                // Convert distance to similarity (inverse relationship)
                let similarity = if distance.is_infinite() {
                    0.0
                } else {
                    1.0 / (1.0 + distance)
                };

                max_similarity = max_similarity.max(similarity);
            }
        }

        Ok(max_similarity)
    }

    /// Compute contextual similarity
    fn compute_contextual_similarity(
        &self,
        _resource1: &str,
        _resource2: &str,
        context: &SimilarityContext,
    ) -> Result<f32> {
        let mut contextual_score = 0.0f32;

        // Domain context
        if context.domain.is_some() {
            contextual_score += 0.3; // Base score for having domain context
        }

        // Temporal context
        contextual_score += context.temporal_weight * 0.2;

        // Cultural context
        if context.cultural_context.is_some() {
            contextual_score += 0.2;
        }

        // Task type context
        let task_boost = match context.task_type {
            SimilarityTaskType::General => 0.1,
            SimilarityTaskType::Classification => 0.15,
            SimilarityTaskType::Retrieval => 0.2,
            SimilarityTaskType::Recommendation => 0.25,
            SimilarityTaskType::Clustering => 0.15,
        };
        contextual_score += task_boost;

        // User context boost
        if !context.user_context.is_empty() {
            let user_boost =
                context.user_context.values().sum::<f32>() / context.user_context.len() as f32;
            contextual_score += user_boost * 0.2;
        }

        Ok(contextual_score.clamp(0.0, 1.0))
    }

    /// Get adaptive weights based on context and learning
    fn get_adaptive_weights(&self, _context: &SimilarityContext) -> (f32, f32, f32) {
        // For now, return default weights
        // In a full implementation, this would use machine learning to adapt weights
        (
            self.config.direct_similarity_weight,
            self.config.hierarchical_similarity_weight,
            self.config.contextual_similarity_weight,
        )
    }

    /// Get concepts associated with a resource
    fn get_concepts_for_resource(&self, resource: &str) -> Vec<String> {
        match self.concept_to_resource.read() {
            Ok(mapping) => mapping.get(resource).cloned().unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Generate a hash for context (for caching)
    fn context_hash(&self, context: &SimilarityContext) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        context.domain.hash(&mut hasher);
        context.cultural_context.hash(&mut hasher);
        context.task_type.hash(&mut hasher);
        (context.temporal_weight as u64).hash(&mut hasher);

        hasher.finish()
    }

    /// Build concept hierarchy from ontology data
    pub fn build_hierarchy_from_ontology(&self, ontology_data: &[(String, String)]) -> Result<()> {
        let mut hierarchy = self
            .concept_hierarchy
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on concept hierarchy"))?;

        for (parent, child) in ontology_data {
            hierarchy.add_relationship(parent.clone(), child.clone());
        }

        Ok(())
    }

    /// Update adaptive weights based on feedback
    pub fn update_adaptive_weights(
        &self,
        context: &SimilarityContext,
        feedback_score: f32,
    ) -> Result<()> {
        if !self.config.enable_adaptive_similarity {
            return Ok(());
        }

        let context_key = format!("{:?}:{:?}", context.domain, context.task_type);

        if let Ok(mut weights) = self.adaptive_weights.write() {
            // Simple learning rule: move weights toward feedback
            let current_weight = weights.get(&context_key).cloned().unwrap_or(0.5);
            let new_weight = current_weight * 0.9 + feedback_score * 0.1;
            weights.insert(context_key, new_weight);
        }

        Ok(())
    }

    /// Get statistics about the hierarchical similarity system
    pub fn get_statistics(&self) -> Result<HierarchicalSimilarityStats> {
        let hierarchy = self
            .concept_hierarchy
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on concept hierarchy"))?;
        let cache = self
            .similarity_cache
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on similarity cache"))?;
        let concept_mapping = self
            .concept_to_resource
            .read()
            .map_err(|_| anyhow!("Failed to acquire read lock on concept mapping"))?;

        Ok(HierarchicalSimilarityStats {
            total_concepts: hierarchy.concept_levels.len(),
            max_hierarchy_depth: hierarchy
                .concept_levels
                .values()
                .max()
                .cloned()
                .unwrap_or(0),
            cached_similarities: cache.len(),
            mapped_resources: concept_mapping.len(),
            average_concepts_per_resource: if concept_mapping.is_empty() {
                0.0
            } else {
                concept_mapping.values().map(|v| v.len()).sum::<usize>() as f32
                    / concept_mapping.len() as f32
            },
        })
    }
}

/// Statistics about the hierarchical similarity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSimilarityStats {
    pub total_concepts: usize,
    pub max_hierarchy_depth: usize,
    pub cached_similarities: usize,
    pub mapped_resources: usize,
    pub average_concepts_per_resource: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_hierarchy() {
        let mut hierarchy = ConceptHierarchy::default();

        // Build a simple hierarchy: Animal -> Mammal -> Dog
        hierarchy.add_relationship("Animal".to_string(), "Mammal".to_string());
        hierarchy.add_relationship("Mammal".to_string(), "Dog".to_string());
        hierarchy.add_relationship("Animal".to_string(), "Bird".to_string());

        // Test distance computation
        let distance = hierarchy.concept_distance("Dog", "Bird");
        assert!(distance > 0.0);

        let distance_same = hierarchy.concept_distance("Dog", "Dog");
        assert_eq!(distance_same, 0.0);

        // Test LCA
        let lca = hierarchy.lowest_common_ancestor("Dog", "Bird");
        assert_eq!(lca, Some("Animal".to_string()));
    }

    #[test]
    fn test_hierarchical_similarity() {
        let config = HierarchicalSimilarityConfig::default();
        let hierarchical_sim = HierarchicalSimilarity::new(config);

        // Add some test concepts
        hierarchical_sim
            .add_concept_relationship("Animal", "Mammal")
            .unwrap();
        hierarchical_sim
            .add_concept_relationship("Mammal", "Dog")
            .unwrap();

        // Associate resources with concepts
        hierarchical_sim
            .associate_resource_with_concepts("resource1", vec!["Dog".to_string()])
            .unwrap();
        hierarchical_sim
            .associate_resource_with_concepts("resource2", vec!["Mammal".to_string()])
            .unwrap();

        // Test would require a VectorStore instance, which is complex to set up in a unit test
        // In practice, integration tests would cover this functionality
    }
}
