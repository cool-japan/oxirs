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
    #[allow(dead_code)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        let mut processed_clusters = clusters;
        
        // Step 1: Merge overlapping clusters
        processed_clusters = self.merge_overlapping_clusters(processed_clusters)?;
        
        // Step 2: Split large clusters
        processed_clusters = self.split_large_clusters(processed_clusters)?;
        
        // Step 3: Validate cluster quality
        processed_clusters = self.validate_cluster_quality(processed_clusters)?;
        
        Ok(processed_clusters)
    }

    /// Merge clusters that have overlapping entities
    fn merge_overlapping_clusters(&self, clusters: Vec<EntityCluster>) -> Result<Vec<EntityCluster>> {
        let mut merged_clusters = Vec::new();
        let mut processed = vec![false; clusters.len()];
        
        for (i, cluster_a) in clusters.iter().enumerate() {
            if processed[i] {
                continue;
            }
            
            let mut merged_cluster = cluster_a.clone();
            processed[i] = true;
            
            // Find overlapping clusters
            for (j, cluster_b) in clusters.iter().enumerate().skip(i + 1) {
                if processed[j] {
                    continue;
                }
                
                // Check for entity overlap
                let overlap_count = cluster_a.entities.iter()
                    .filter(|entity| cluster_b.entities.contains(entity))
                    .count();
                
                let min_size = cluster_a.entities.len().min(cluster_b.entities.len());
                let overlap_ratio = overlap_count as f64 / min_size as f64;
                
                // Merge if overlap ratio exceeds threshold
                if overlap_ratio > 0.3 {
                    // Merge entities
                    for entity in &cluster_b.entities {
                        if !merged_cluster.entities.contains(entity) {
                            merged_cluster.entities.push(entity.clone());
                        }
                    }
                    
                    // Update cluster properties
                    merged_cluster.size = merged_cluster.entities.len();
                    merged_cluster.confidence = (merged_cluster.confidence + cluster_b.confidence) / 2.0;
                    
                    // Record merge decision
                    merged_cluster.merge_decisions.push(MergeDecision {
                        source_entity: cluster_b.id.clone(),
                        target_entity: merged_cluster.id.clone(),
                        similarity: overlap_ratio as f32,
                        decision: DecisionType::Merge,
                        confidence: overlap_ratio as f32,
                        features_used: vec![FeatureType::StructuralSimilarity],
                    });
                    
                    processed[j] = true;
                }
            }
            
            merged_clusters.push(merged_cluster);
        }
        
        Ok(merged_clusters)
    }

    /// Split clusters that are too large
    fn split_large_clusters(&self, clusters: Vec<EntityCluster>) -> Result<Vec<EntityCluster>> {
        let mut split_clusters = Vec::new();
        let max_cluster_size = 50; // Configurable threshold
        
        for cluster in clusters {
            if cluster.entities.len() <= max_cluster_size {
                split_clusters.push(cluster);
                continue;
            }
            
            // Split large cluster using similarity-based grouping
            let sub_clusters = self.split_cluster_by_similarity(&cluster, max_cluster_size)?;
            split_clusters.extend(sub_clusters);
        }
        
        Ok(split_clusters)
    }

    /// Split a cluster by similarity into smaller sub-clusters
    fn split_cluster_by_similarity(&self, cluster: &EntityCluster, max_size: usize) -> Result<Vec<EntityCluster>> {
        let mut sub_clusters = Vec::new();
        let mut remaining_entities = cluster.entities.clone();
        let mut cluster_id_counter = 0;
        
        while !remaining_entities.is_empty() {
            let mut current_cluster_entities = Vec::new();
            let seed_entity = remaining_entities.remove(0);
            current_cluster_entities.push(seed_entity.clone());
            
            // Add similar entities to current cluster
            let mut i = 0;
            while i < remaining_entities.len() && current_cluster_entities.len() < max_size {
                let entity = &remaining_entities[i];
                
                // Check similarity with entities in current cluster
                let mut max_similarity = 0.0;
                for cluster_entity in &current_cluster_entities {
                    let similarity = self.calculate_entity_similarity(entity, cluster_entity)?;
                    if similarity > max_similarity {
                        max_similarity = similarity;
                    }
                }
                
                // Add to cluster if similarity exceeds threshold
                if max_similarity > 0.7 {
                    current_cluster_entities.push(remaining_entities.remove(i));
                } else {
                    i += 1;
                }
            }
            
            // Create sub-cluster
            let canonical_entity = current_cluster_entities[0].clone();
            let sub_cluster = EntityCluster {
                id: format!("{}_split_{}", cluster.id, cluster_id_counter),
                entities: current_cluster_entities.clone(),
                canonical_entity,
                confidence: cluster.confidence * 0.9, // Slightly lower confidence for split clusters
                size: current_cluster_entities.len(),
                merge_decisions: vec![MergeDecision {
                    source_entity: cluster.id.clone(),
                    target_entity: format!("{}_split_{}", cluster.id, cluster_id_counter),
                    similarity: cluster.confidence,
                    decision: DecisionType::NoMerge,
                    confidence: cluster.confidence,
                    features_used: vec![FeatureType::StructuralSimilarity],
                }],
            };
            
            sub_clusters.push(sub_cluster);
            cluster_id_counter += 1;
        }
        
        Ok(sub_clusters)
    }

    /// Validate cluster quality and filter out low-quality clusters
    fn validate_cluster_quality(&self, clusters: Vec<EntityCluster>) -> Result<Vec<EntityCluster>> {
        let mut validated_clusters = Vec::new();
        
        for cluster in clusters {
            // Quality metrics
            let min_cluster_size = 2;
            let min_confidence = 0.5;
            
            // Check minimum size
            if cluster.entities.len() < min_cluster_size {
                continue;
            }
            
            // Check minimum confidence
            if cluster.confidence < min_confidence {
                continue;
            }
            
            // Calculate internal similarity
            let internal_similarity = self.calculate_cluster_internal_similarity(&cluster)?;
            if internal_similarity < 0.6 {
                continue;
            }
            
            validated_clusters.push(cluster);
        }
        
        Ok(validated_clusters)
    }

    /// Calculate internal similarity of a cluster
    fn calculate_cluster_internal_similarity(&self, cluster: &EntityCluster) -> Result<f64> {
        if cluster.entities.len() < 2 {
            return Ok(1.0);
        }
        
        let mut total_similarity = 0.0;
        let mut comparison_count = 0;
        
        for i in 0..cluster.entities.len() {
            for j in (i + 1)..cluster.entities.len() {
                let similarity = self.calculate_entity_similarity(&cluster.entities[i], &cluster.entities[j])?;
                total_similarity += similarity;
                comparison_count += 1;
            }
        }
        
        if comparison_count > 0 {
            Ok(total_similarity / comparison_count as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate similarity between two entities (helper method)
    fn calculate_entity_similarity(&self, entity1: &EntityRecord, entity2: &EntityRecord) -> Result<f64> {
        // Simple string similarity based on labels from attributes
        let label1 = entity1.attributes.get("label").unwrap_or(&entity1.uri).to_lowercase();
        let label2 = entity2.attributes.get("label").unwrap_or(&entity2.uri).to_lowercase();
        
        // Jaccard similarity on character n-grams
        let ngrams1: std::collections::HashSet<String> = self.generate_character_ngrams(&label1, 2);
        let ngrams2: std::collections::HashSet<String> = self.generate_character_ngrams(&label2, 2);
        
        let intersection = ngrams1.intersection(&ngrams2).count();
        let union = ngrams1.union(&ngrams2).count();
        
        if union > 0 {
            Ok(intersection as f64 / union as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Generate character n-grams
    fn generate_character_ngrams(&self, text: &str, n: usize) -> std::collections::HashSet<String> {
        let mut ngrams = std::collections::HashSet::new();
        let chars: Vec<char> = text.chars().collect();
        
        if chars.len() >= n {
            for i in 0..=(chars.len() - n) {
                let ngram: String = chars[i..i + n].iter().collect();
                ngrams.insert(ngram);
            }
        }
        
        ngrams
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
        let features = vec![
            // URI length
            entity.uri.len() as f32,
            // Number of attributes
            entity.attributes.len() as f32,
            // Number of triples
            entity.triples.len() as f32,
            // Quality score
            entity.quality_score,
        ];

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
                merge_decisions: vec![MergeDecision {
                    source_entity: "initial".to_string(),
                    target_entity: format!("cluster_{}", clusters.len()),
                    similarity: 0.8,
                    decision: DecisionType::Merge,
                    confidence: 0.8,
                    features_used: vec![FeatureType::StructuralSimilarity],
                }]
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

        indexed_entities.sort_by_key(|x| x.1.clone());

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
        let mut resolver = EntityResolver::new(&config).unwrap();
        
        // Use a lower similarity threshold for testing
        resolver.config.similarity_threshold = 0.3;

        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/person1").unwrap(),
                NamedNode::new("http://example.org/name").unwrap(),
                Literal::new("John Smith"),
            ),
            Triple::new(
                NamedNode::new("http://example.org/person2").unwrap(),
                NamedNode::new("http://example.org/name").unwrap(),
                Literal::new("John Smith"),
            ),
            Triple::new(
                NamedNode::new("http://example.org/person3").unwrap(),
                NamedNode::new("http://example.org/name").unwrap(),
                Literal::new("Jane Doe"),
            ),
            Triple::new(
                NamedNode::new("http://example.org/person4").unwrap(),
                NamedNode::new("http://example.org/name").unwrap(),
                Literal::new("Jane Doe"),
            ),
        ];

        let clusters = resolver.resolve_entities(&triples).await.unwrap();
        // Should create clusters for similar entities
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
