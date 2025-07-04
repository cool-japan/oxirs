//! Entity Resolution with Machine Learning
//!
//! This module provides entity resolution capabilities to identify and merge
//! duplicate entities across different data sources.

use crate::ai::AiConfig;
use crate::model::Triple;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Entity resolution module
pub struct EntityResolver {
    /// Configuration
    config: ResolutionConfig,

    /// Similarity calculator
    similarity_calculator: Box<dyn SimilarityCalculator>,

    /// Clustering algorithm
    clustering_algorithm: Box<dyn ClusteringAlgorithm>,

    /// Feature extractor
    feature_extractor: Box<dyn FeatureExtractor>,

    /// Blocking strategy
    blocking_strategy: Box<dyn BlockingStrategy>,
}

/// Entity resolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionConfig {
    /// Similarity threshold for entity matching
    pub similarity_threshold: f32,

    /// Clustering algorithm to use
    pub clustering_algorithm: ClusteringType,

    /// Features to use for similarity calculation
    pub features: Vec<FeatureType>,

    /// Blocking strategy
    pub blocking_strategy: BlockingType,

    /// Maximum cluster size
    pub max_cluster_size: usize,

    /// Enable machine learning similarity
    pub enable_ml_similarity: bool,

    /// Training data path (if using ML)
    pub training_data_path: Option<String>,
}

impl Default for ResolutionConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.8,
            clustering_algorithm: ClusteringType::HierarchicalClustering,
            features: vec![
                FeatureType::StringSimilarity,
                FeatureType::NumericSimilarity,
                FeatureType::StructuralSimilarity,
            ],
            blocking_strategy: BlockingType::SortedNeighborhood,
            max_cluster_size: 100,
            enable_ml_similarity: true,
            training_data_path: None,
        }
    }
}

/// Clustering algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringType {
    /// Hierarchical clustering
    HierarchicalClustering,

    /// Connected components
    ConnectedComponents,

    /// Correlation clustering
    CorrelationClustering,

    /// DBSCAN
    DBSCAN { eps: f32, min_samples: usize },

    /// Markov clustering
    MarkovClustering { inflation: f32 },
}

/// Feature types for entity similarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    /// String similarity features
    StringSimilarity,

    /// Numeric similarity features
    NumericSimilarity,

    /// Structural similarity (graph-based)
    StructuralSimilarity,

    /// Semantic similarity (embedding-based)
    SemanticSimilarity,

    /// Temporal similarity
    TemporalSimilarity,

    /// Contextual similarity
    ContextualSimilarity,
}

/// Blocking strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockingType {
    /// Standard blocking
    StandardBlocking,

    /// Sorted neighborhood method
    SortedNeighborhood,

    /// Locality-sensitive hashing
    LSH {
        num_hashes: usize,
        hash_length: usize,
    },

    /// Canopy clustering
    CanopyClustering { t1: f32, t2: f32 },

    /// Multi-pass blocking
    MultiPass(Vec<BlockingType>),
}

/// Entity cluster result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCluster {
    /// Cluster ID
    pub id: String,

    /// Entities in the cluster
    pub entities: Vec<EntityRecord>,

    /// Canonical entity (representative)
    pub canonical_entity: EntityRecord,

    /// Cluster confidence score
    pub confidence: f32,

    /// Cluster size
    pub size: usize,

    /// Merge decisions
    pub merge_decisions: Vec<MergeDecision>,
}

/// Entity record for resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Entity ID
    pub id: String,

    /// Entity URI
    pub uri: String,

    /// Attributes
    pub attributes: HashMap<String, String>,

    /// Associated triples
    pub triples: Vec<Triple>,

    /// Source information
    pub source: String,

    /// Quality score
    pub quality_score: f32,
}

/// Merge decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeDecision {
    /// Source entity
    pub source_entity: String,

    /// Target entity
    pub target_entity: String,

    /// Similarity score
    pub similarity: f32,

    /// Decision type
    pub decision: DecisionType,

    /// Confidence in decision
    pub confidence: f32,

    /// Features used
    pub features_used: Vec<FeatureType>,
}

/// Decision types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    Merge,
    NoMerge,
    Uncertain,
}

/// Similarity calculator trait
pub trait SimilarityCalculator: Send + Sync {
    /// Calculate similarity between two entities
    fn calculate_similarity(&self, entity1: &EntityRecord, entity2: &EntityRecord) -> Result<f32>;

    /// Get feature vector for entity
    fn get_feature_vector(&self, entity: &EntityRecord) -> Result<Vec<f32>>;
}

/// Clustering algorithm trait
pub trait ClusteringAlgorithm: Send + Sync {
    /// Cluster entities based on similarity
    fn cluster_entities(
        &self,
        entities: &[EntityRecord],
        similarity_matrix: &[Vec<f32>],
        threshold: f32,
    ) -> Result<Vec<EntityCluster>>;
}

/// Feature extractor trait
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from entity
    fn extract_features(&self, entity: &EntityRecord) -> Result<HashMap<String, f32>>;

    /// Get feature names
    fn feature_names(&self) -> Vec<String>;
}

/// Blocking strategy trait
pub trait BlockingStrategy: Send + Sync {
    /// Generate blocks of potentially matching entities
    fn generate_blocks(&self, entities: &[EntityRecord]) -> Result<Vec<Vec<usize>>>;

    /// Get blocking key for entity
    fn get_blocking_key(&self, entity: &EntityRecord) -> Result<String>;
}

impl EntityResolver {
    /// Create new entity resolver
    pub fn new(_config: &AiConfig) -> Result<Self> {
        let resolution_config = ResolutionConfig::default();

        // Create components
        let similarity_calculator = Box::new(DefaultSimilarityCalculator::new());
        let clustering_algorithm = Box::new(HierarchicalClusterer::new());
        let feature_extractor = Box::new(DefaultFeatureExtractor::new());
        let blocking_strategy = Box::new(SortedNeighborhoodBlocking::new());

        Ok(Self {
            config: resolution_config,
            similarity_calculator,
            clustering_algorithm,
            feature_extractor,
            blocking_strategy,
        })
    }

    /// Resolve entities from triples
    pub async fn resolve_entities(&self, triples: &[Triple]) -> Result<Vec<EntityCluster>> {
        // Step 1: Extract entity records from triples
        let entities = self.extract_entity_records(triples)?;

        // Step 2: Apply blocking strategy to reduce comparisons
        let blocks = self.blocking_strategy.generate_blocks(&entities)?;

        let mut all_clusters = Vec::new();

        // Step 3: Process each block separately
        for block in blocks {
            let block_entities: Vec<&EntityRecord> = block.iter().map(|&i| &entities[i]).collect();

            // Step 4: Calculate similarity matrix for block
            let similarity_matrix = self.calculate_similarity_matrix(&block_entities)?;

            // Step 5: Cluster entities
            let block_entities_owned: Vec<EntityRecord> =
                block_entities.into_iter().cloned().collect();
            let clusters = self.clustering_algorithm.cluster_entities(
                &block_entities_owned,
                &similarity_matrix,
                self.config.similarity_threshold,
            )?;

            all_clusters.extend(clusters);
        }

        // Step 6: Post-process clusters
        let final_clusters = self.post_process_clusters(all_clusters)?;

        Ok(final_clusters)
    }

    /// Extract entity records from triples
    fn extract_entity_records(&self, triples: &[Triple]) -> Result<Vec<EntityRecord>> {
        let mut entity_map: HashMap<String, EntityRecord> = HashMap::new();
        let entity_counter = std::cell::RefCell::new(0);

        for triple in triples {
            let subject_uri = triple.subject().to_string();
            let predicate_uri = triple.predicate().to_string();
            let object_string = triple.object().to_string();

            // Process subject entity
            let subject_entry = entity_map.entry(subject_uri.clone()).or_insert_with(|| {
                let id = {
                    let mut counter = entity_counter.borrow_mut();
                    *counter += 1;
                    *counter
                };
                EntityRecord {
                    id: format!("entity_{id}"),
                    uri: subject_uri.clone(),
                    attributes: HashMap::new(),
                    triples: Vec::new(),
                    source: "unknown".to_string(),
                    quality_score: 1.0,
                }
            });

            subject_entry.triples.push(triple.clone());
            subject_entry
                .attributes
                .insert(predicate_uri.clone(), object_string.clone());

            // Process object entity if it's not a literal
            if let crate::model::Object::NamedNode(node) = triple.object() {
                let object_uri = node.to_string();
                let object_entry = entity_map.entry(object_uri.clone()).or_insert_with(|| {
                    let id = {
                        let mut counter = entity_counter.borrow_mut();
                        *counter += 1;
                        *counter
                    };
                    EntityRecord {
                        id: format!("entity_{id}"),
                        uri: object_uri.clone(),
                        attributes: HashMap::new(),
                        triples: Vec::new(),
                        source: "unknown".to_string(),
                        quality_score: 1.0,
                    }
                });

                // Add reverse relation
                object_entry
                    .attributes
                    .insert(format!("{predicate_uri}^-1"), subject_uri.clone());
            }
        }

        Ok(entity_map.into_values().collect())
    }

    /// Calculate similarity matrix for entities
    fn calculate_similarity_matrix(&self, entities: &[&EntityRecord]) -> Result<Vec<Vec<f32>>> {
        let n = entities.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                if i == j {
                    matrix[i][j] = 1.0;
                } else {
                    let similarity = self
                        .similarity_calculator
                        .calculate_similarity(entities[i], entities[j])?;
                    matrix[i][j] = similarity;
                    matrix[j][i] = similarity;
                }
            }
        }

        Ok(matrix)
    }

    /// Post-process clusters
    fn post_process_clusters(&self, clusters: Vec<EntityCluster>) -> Result<Vec<EntityCluster>> {
        // TODO: Implement cluster post-processing
        // - Merge overlapping clusters
        // - Split large clusters
        // - Validate cluster quality

        Ok(clusters)
    }
}

/// Default similarity calculator
struct DefaultSimilarityCalculator;

impl DefaultSimilarityCalculator {
    fn new() -> Self {
        Self
    }

    fn string_similarity(&self, s1: &str, s2: &str) -> f32 {
        // Simplified Jaccard similarity
        let set1: HashSet<char> = s1.chars().collect();
        let set2: HashSet<char> = s2.chars().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn attribute_similarity(
        &self,
        attrs1: &HashMap<String, String>,
        attrs2: &HashMap<String, String>,
    ) -> f32 {
        let mut total_similarity = 0.0;
        let mut count = 0;

        for (key, value1) in attrs1 {
            if let Some(value2) = attrs2.get(key) {
                total_similarity += self.string_similarity(value1, value2);
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_similarity / count as f32
        }
    }
}

impl SimilarityCalculator for DefaultSimilarityCalculator {
    fn calculate_similarity(&self, entity1: &EntityRecord, entity2: &EntityRecord) -> Result<f32> {
        // Combine multiple similarity measures
        let uri_similarity = self.string_similarity(&entity1.uri, &entity2.uri);
        let attr_similarity = self.attribute_similarity(&entity1.attributes, &entity2.attributes);

        // Weighted combination
        let similarity = 0.3 * uri_similarity + 0.7 * attr_similarity;

        Ok(similarity)
    }

    fn get_feature_vector(&self, entity: &EntityRecord) -> Result<Vec<f32>> {
        // Extract simple features
        let mut features = Vec::new();

        // URI length
        features.push(entity.uri.len() as f32);

        // Number of attributes
        features.push(entity.attributes.len() as f32);

        // Number of triples
        features.push(entity.triples.len() as f32);

        // Quality score
        features.push(entity.quality_score);

        Ok(features)
    }
}

/// Hierarchical clustering implementation
struct HierarchicalClusterer;

impl HierarchicalClusterer {
    fn new() -> Self {
        Self
    }
}

impl ClusteringAlgorithm for HierarchicalClusterer {
    fn cluster_entities(
        &self,
        entities: &[EntityRecord],
        similarity_matrix: &[Vec<f32>],
        threshold: f32,
    ) -> Result<Vec<EntityCluster>> {
        let n = entities.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Simple clustering: group entities with similarity above threshold
        let mut clusters = Vec::new();
        let mut visited = vec![false; n];

        for i in 0..n {
            if visited[i] {
                continue;
            }

            let mut cluster_entities = vec![entities[i].clone()];
            visited[i] = true;

            for j in (i + 1)..n {
                if !visited[j] && similarity_matrix[i][j] >= threshold {
                    cluster_entities.push(entities[j].clone());
                    visited[j] = true;
                }
            }

            // Create cluster
            let canonical_entity = cluster_entities[0].clone(); // Simplified
            let cluster = EntityCluster {
                id: format!("cluster_{}", clusters.len()),
                entities: cluster_entities.clone(),
                canonical_entity,
                confidence: 0.8, // Simplified
                size: cluster_entities.len(),
                merge_decisions: Vec::new(), // TODO: Track merge decisions
            };

            clusters.push(cluster);
        }

        Ok(clusters)
    }
}

/// Default feature extractor
struct DefaultFeatureExtractor;

impl DefaultFeatureExtractor {
    fn new() -> Self {
        Self
    }
}

impl FeatureExtractor for DefaultFeatureExtractor {
    fn extract_features(&self, entity: &EntityRecord) -> Result<HashMap<String, f32>> {
        let mut features = HashMap::new();

        // Basic features
        features.insert("uri_length".to_string(), entity.uri.len() as f32);
        features.insert("num_attributes".to_string(), entity.attributes.len() as f32);
        features.insert("num_triples".to_string(), entity.triples.len() as f32);
        features.insert("quality_score".to_string(), entity.quality_score);

        // Attribute-based features
        for (key, value) in &entity.attributes {
            features.insert(format!("attr_{key}_length"), value.len() as f32);
        }

        Ok(features)
    }

    fn feature_names(&self) -> Vec<String> {
        vec![
            "uri_length".to_string(),
            "num_attributes".to_string(),
            "num_triples".to_string(),
            "quality_score".to_string(),
        ]
    }
}

/// Sorted neighborhood blocking
struct SortedNeighborhoodBlocking;

impl SortedNeighborhoodBlocking {
    fn new() -> Self {
        Self
    }
}

impl BlockingStrategy for SortedNeighborhoodBlocking {
    fn generate_blocks(&self, entities: &[EntityRecord]) -> Result<Vec<Vec<usize>>> {
        // Sort entities by blocking key and create windows
        let mut indexed_entities: Vec<(usize, String)> = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| (i, self.get_blocking_key(entity).unwrap_or_default()))
            .collect();

        indexed_entities.sort_by(|a, b| a.1.cmp(&b.1));

        // Create sliding windows
        let window_size = 10; // Configurable
        let mut blocks = Vec::new();

        for start in 0..entities.len() {
            if start + window_size <= entities.len() {
                let block: Vec<usize> = indexed_entities[start..start + window_size]
                    .iter()
                    .map(|(i, _)| *i)
                    .collect();
                blocks.push(block);
            }
        }

        if blocks.is_empty() {
            // Single block with all entities
            blocks.push((0..entities.len()).collect());
        }

        Ok(blocks)
    }

    fn get_blocking_key(&self, entity: &EntityRecord) -> Result<String> {
        // Use first few characters of URI as blocking key
        let key = entity
            .uri
            .chars()
            .take(10)
            .collect::<String>()
            .to_lowercase();
        Ok(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::AiConfig;
    use crate::model::{Literal, NamedNode};

    #[tokio::test]
    async fn test_entity_resolver_creation() {
        let config = AiConfig::default();
        let resolver = EntityResolver::new(&config);
        assert!(resolver.is_ok());
    }

    #[tokio::test]
    async fn test_entity_resolution() {
        let config = AiConfig::default();
        let resolver = EntityResolver::new(&config).unwrap();

        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/person1").unwrap(),
                NamedNode::new("http://example.org/name").unwrap(),
                Literal::new("John Smith"),
            ),
            Triple::new(
                NamedNode::new("http://example.org/person2").unwrap(),
                NamedNode::new("http://example.org/name").unwrap(),
                Literal::new("J. Smith"),
            ),
        ];

        let clusters = resolver.resolve_entities(&triples).await.unwrap();
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_similarity_calculation() {
        let calculator = DefaultSimilarityCalculator::new();

        let entity1 = EntityRecord {
            id: "1".to_string(),
            uri: "http://example.org/john".to_string(),
            attributes: [("name".to_string(), "John".to_string())]
                .iter()
                .cloned()
                .collect(),
            triples: Vec::new(),
            source: "source1".to_string(),
            quality_score: 1.0,
        };

        let entity2 = EntityRecord {
            id: "2".to_string(),
            uri: "http://example.org/john_smith".to_string(),
            attributes: [("name".to_string(), "John Smith".to_string())]
                .iter()
                .cloned()
                .collect(),
            triples: Vec::new(),
            source: "source2".to_string(),
            quality_score: 1.0,
        };

        let similarity = calculator.calculate_similarity(&entity1, &entity2).unwrap();
        assert!(similarity > 0.0);
        assert!(similarity <= 1.0);
    }
}
