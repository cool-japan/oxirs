//! Enhanced RDF Content Processing with Entity and Relationship Embeddings
//!
//! This module provides advanced RDF content processing capabilities including:
//! - URI-based embeddings with context awareness
//! - Property aggregation and label integration
//! - Property path embeddings for complex relationships
//! - Subgraph embeddings for contextual understanding
//! - Multi-language support for international RDF data
//! - Temporal relationship encoding

use crate::{
    embeddings::{EmbeddableContent, EmbeddingManager, EmbeddingStrategy},
    kg_embeddings::{KGEmbeddingConfig, KGEmbeddingModel, Triple},
    similarity::SimilarityMetric,
    Vector, VectorError,
};
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Configuration for RDF content enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfContentConfig {
    /// Enable URI-based embeddings
    pub enable_uri_embeddings: bool,
    /// Enable property aggregation
    pub enable_property_aggregation: bool,
    /// Enable multi-language support
    pub enable_multi_language: bool,
    /// Enable temporal encoding
    pub enable_temporal_encoding: bool,
    /// Maximum property path length
    pub max_path_length: usize,
    /// Default language for text processing
    pub default_language: String,
    /// Subgraph context window size
    pub context_window_size: usize,
    /// Weight for different embedding components
    pub component_weights: ComponentWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentWeights {
    pub uri_weight: f32,
    pub label_weight: f32,
    pub description_weight: f32,
    pub property_weight: f32,
    pub context_weight: f32,
    pub temporal_weight: f32,
}

impl Default for ComponentWeights {
    fn default() -> Self {
        Self {
            uri_weight: 0.1,
            label_weight: 0.3,
            description_weight: 0.3,
            property_weight: 0.2,
            context_weight: 0.05,
            temporal_weight: 0.05,
        }
    }
}

impl Default for RdfContentConfig {
    fn default() -> Self {
        Self {
            enable_uri_embeddings: true,
            enable_property_aggregation: true,
            enable_multi_language: true,
            enable_temporal_encoding: false,
            max_path_length: 3,
            default_language: "en".to_string(),
            context_window_size: 5,
            component_weights: ComponentWeights::default(),
        }
    }
}

/// RDF Entity with enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfEntity {
    pub uri: String,
    pub labels: HashMap<String, String>, // language -> label
    pub descriptions: HashMap<String, String>, // language -> description
    pub properties: HashMap<String, Vec<RdfValue>>,
    pub types: Vec<String>,
    pub context: Option<RdfContext>,
    pub temporal_info: Option<TemporalInfo>,
}

/// RDF Value supporting various types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RdfValue {
    IRI(String),
    Literal(String, Option<String>), // value, datatype
    LangString(String, String),      // value, language
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Date(String),
    DateTime(String),
}

/// Context information for an entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfContext {
    pub graph_uri: Option<String>,
    pub neighbors: Vec<String>,                  // neighboring entity URIs
    pub subgraph_signature: Option<String>,      // hash of local subgraph structure
    pub semantic_distance: HashMap<String, f32>, // distance to key concepts
}

/// Temporal information for relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub created_at: Option<String>,
    pub modified_at: Option<String>,
    pub version: Option<String>,
}

/// Property path for relationship embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyPath {
    pub path: Vec<String>,             // sequence of properties
    pub direction: Vec<PathDirection>, // forward/backward for each step
    pub constraints: Vec<PathConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathDirection {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathConstraint {
    TypeFilter(String),
    PropertyFilter(String, RdfValue),
    LanguageFilter(String),
}

/// Enhanced RDF content processor
pub struct RdfContentProcessor {
    config: RdfContentConfig,
    embedding_manager: Arc<RwLock<EmbeddingManager>>,
    kg_embeddings: Option<Box<dyn KGEmbeddingModel>>,
    entity_cache: HashMap<String, Vector>,
    relationship_cache: HashMap<String, Vector>,
    property_aggregator: PropertyAggregator,
    multi_language_processor: MultiLanguageProcessor,
}

impl RdfContentProcessor {
    /// Create new RDF content processor
    pub fn new(config: RdfContentConfig, embedding_strategy: EmbeddingStrategy) -> Result<Self> {
        let embedding_manager = Arc::new(RwLock::new(EmbeddingManager::new(
            embedding_strategy,
            1000,
        )?));

        Ok(Self {
            config,
            embedding_manager,
            kg_embeddings: None,
            entity_cache: HashMap::new(),
            relationship_cache: HashMap::new(),
            property_aggregator: PropertyAggregator::new(),
            multi_language_processor: MultiLanguageProcessor::new(),
        })
    }

    /// Generate enhanced entity embedding
    pub fn generate_entity_embedding(&mut self, entity: &RdfEntity) -> Result<Vector> {
        // Check cache first
        if let Some(cached) = self.entity_cache.get(&entity.uri) {
            return Ok(cached.clone());
        }

        let mut embedding_components = Vec::new();
        let weights = &self.config.component_weights;

        // 1. URI-based embedding
        if self.config.enable_uri_embeddings {
            let uri_embedding = self.generate_uri_embedding(&entity.uri)?;
            embedding_components.push((uri_embedding, weights.uri_weight));
        }

        // 2. Label embeddings (multi-language support)
        if !entity.labels.is_empty() {
            let label_embedding = self.generate_label_embedding(&entity.labels)?;
            embedding_components.push((label_embedding, weights.label_weight));
        }

        // 3. Description embeddings
        if !entity.descriptions.is_empty() {
            let desc_embedding = self.generate_description_embedding(&entity.descriptions)?;
            embedding_components.push((desc_embedding, weights.description_weight));
        }

        // 4. Property aggregation embedding
        if self.config.enable_property_aggregation && !entity.properties.is_empty() {
            let prop_embedding = self
                .property_aggregator
                .aggregate_properties(&entity.properties)?;
            embedding_components.push((prop_embedding, weights.property_weight));
        }

        // 5. Context-aware embedding
        if let Some(context) = &entity.context {
            let context_embedding = self.generate_context_embedding(context)?;
            embedding_components.push((context_embedding, weights.context_weight));
        }

        // 6. Temporal embedding
        if self.config.enable_temporal_encoding {
            if let Some(temporal) = &entity.temporal_info {
                let temporal_embedding = self.generate_temporal_embedding(temporal)?;
                embedding_components.push((temporal_embedding, weights.temporal_weight));
            }
        }

        // Combine all embeddings using weighted average
        let final_embedding = self.combine_embeddings(embedding_components)?;

        // Cache the result
        self.entity_cache
            .insert(entity.uri.clone(), final_embedding.clone());

        Ok(final_embedding)
    }

    /// Generate property path embeddings
    pub fn generate_property_path_embedding(&mut self, path: &PropertyPath) -> Result<Vector> {
        let path_key = format!("{:?}", path);

        // Check cache first
        if let Some(cached) = self.relationship_cache.get(&path_key) {
            return Ok(cached.clone());
        }

        let mut path_embeddings = Vec::new();

        // Generate embeddings for each property in the path
        for (i, property) in path.path.iter().enumerate() {
            let mut prop_text = property.clone();

            // Add direction information
            match path.direction.get(i) {
                Some(PathDirection::Forward) => prop_text.push_str(" ->"),
                Some(PathDirection::Backward) => prop_text.push_str(" <-"),
                None => {}
            }

            // Add constraint information
            for constraint in &path.constraints {
                match constraint {
                    PathConstraint::TypeFilter(type_uri) => {
                        prop_text.push_str(&format!(" [type:{}]", type_uri));
                    }
                    PathConstraint::PropertyFilter(prop, value) => {
                        prop_text.push_str(&format!(" [{}:{:?}]", prop, value));
                    }
                    PathConstraint::LanguageFilter(lang) => {
                        prop_text.push_str(&format!(" [@{}]", lang));
                    }
                }
            }

            let content = EmbeddableContent::Text(prop_text);
            let prop_embedding = self.embedding_manager.write().get_embedding(&content)?;
            path_embeddings.push(prop_embedding);
        }

        // Combine path embeddings using sequence-aware method
        let path_embedding = self.combine_path_embeddings(path_embeddings)?;

        // Cache the result
        self.relationship_cache
            .insert(path_key, path_embedding.clone());

        Ok(path_embedding)
    }

    /// Generate subgraph embedding for contextual understanding
    pub fn generate_subgraph_embedding(&mut self, entities: &[RdfEntity]) -> Result<Vector> {
        if entities.is_empty() {
            return Err(anyhow!("Cannot generate embedding for empty subgraph"));
        }

        let mut entity_embeddings = Vec::new();

        // Generate embeddings for all entities in the subgraph
        for entity in entities {
            let embedding = self.generate_entity_embedding(entity)?;
            entity_embeddings.push(embedding);
        }

        // Use graph-aware aggregation method
        self.aggregate_subgraph_embeddings(entity_embeddings)
    }

    /// Generate URI-based embedding using structural decomposition
    fn generate_uri_embedding(&self, uri: &str) -> Result<Vector> {
        // Decompose URI into meaningful components
        let components = self.decompose_uri(uri);
        let text_content = components.join(" ");

        let content = EmbeddableContent::Text(text_content);
        self.embedding_manager.write().get_embedding(&content)
    }

    /// Generate multi-language label embedding
    fn generate_label_embedding(&self, labels: &HashMap<String, String>) -> Result<Vector> {
        let preferred_lang = &self.config.default_language;

        // Prefer the default language, fall back to others
        let label_text = if let Some(preferred_label) = labels.get(preferred_lang) {
            preferred_label.clone()
        } else if let Some((_, first_label)) = labels.iter().next() {
            first_label.clone()
        } else {
            return Err(anyhow!("No labels available"));
        };

        let content = EmbeddableContent::Text(label_text);
        self.embedding_manager.write().get_embedding(&content)
    }

    /// Generate description embedding
    fn generate_description_embedding(
        &self,
        descriptions: &HashMap<String, String>,
    ) -> Result<Vector> {
        let preferred_lang = &self.config.default_language;

        let desc_text = if let Some(preferred_desc) = descriptions.get(preferred_lang) {
            preferred_desc.clone()
        } else if let Some((_, first_desc)) = descriptions.iter().next() {
            first_desc.clone()
        } else {
            return Err(anyhow!("No descriptions available"));
        };

        let content = EmbeddableContent::Text(desc_text);
        self.embedding_manager.write().get_embedding(&content)
    }

    /// Generate context-aware embedding
    fn generate_context_embedding(&self, context: &RdfContext) -> Result<Vector> {
        let mut context_text = String::new();

        if let Some(graph_uri) = &context.graph_uri {
            context_text.push_str(&format!("graph:{} ", graph_uri));
        }

        // Add neighbor information
        if !context.neighbors.is_empty() {
            context_text.push_str("neighbors:");
            for neighbor in &context.neighbors {
                context_text.push_str(&format!(" {}", neighbor));
            }
        }

        if context_text.is_empty() {
            // Generate a zero vector if no context available
            return Ok(Vector::new(vec![0.0; 384])); // Default embedding size
        }

        let content = EmbeddableContent::Text(context_text);
        self.embedding_manager.write().get_embedding(&content)
    }

    /// Generate temporal embedding
    fn generate_temporal_embedding(&self, temporal: &TemporalInfo) -> Result<Vector> {
        let mut temporal_text = String::new();

        if let Some(valid_from) = &temporal.valid_from {
            temporal_text.push_str(&format!("from:{} ", valid_from));
        }

        if let Some(valid_to) = &temporal.valid_to {
            temporal_text.push_str(&format!("to:{} ", valid_to));
        }

        if let Some(created) = &temporal.created_at {
            temporal_text.push_str(&format!("created:{} ", created));
        }

        if temporal_text.is_empty() {
            // Generate a zero vector if no temporal info available
            return Ok(Vector::new(vec![0.0; 384])); // Default embedding size
        }

        let content = EmbeddableContent::Text(temporal_text);
        self.embedding_manager.write().get_embedding(&content)
    }

    /// Combine multiple embeddings using weighted average
    fn combine_embeddings(&self, embeddings: Vec<(Vector, f32)>) -> Result<Vector> {
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings to combine"));
        }

        let dimensions = embeddings[0].0.dimensions;
        let mut combined = vec![0.0; dimensions];
        let mut total_weight = 0.0;

        for (embedding, weight) in embeddings {
            if embedding.dimensions != dimensions {
                return Err(anyhow!("Dimension mismatch in embedding combination"));
            }

            let values = embedding.as_f32();
            for (i, value) in values.iter().enumerate() {
                combined[i] += value * weight;
            }
            total_weight += weight;
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            for value in &mut combined {
                *value /= total_weight;
            }
        }

        Ok(Vector::new(combined))
    }

    /// Combine path embeddings using sequence-aware method
    fn combine_path_embeddings(&self, embeddings: Vec<Vector>) -> Result<Vector> {
        if embeddings.is_empty() {
            return Err(anyhow!("No path embeddings to combine"));
        }

        let dimensions = embeddings[0].dimensions;
        let mut combined = vec![0.0; dimensions];

        // Use position-weighted combination
        for (i, embedding) in embeddings.iter().enumerate() {
            let position_weight = 1.0 / (i as f32 + 1.0); // Diminishing weights for later positions
            let values = embedding.as_f32();

            for (j, value) in values.iter().enumerate() {
                combined[j] += value * position_weight;
            }
        }

        // Normalize
        let total_positions = embeddings.len() as f32;
        for value in &mut combined {
            *value /= total_positions;
        }

        Ok(Vector::new(combined))
    }

    /// Aggregate subgraph embeddings using graph-aware method
    fn aggregate_subgraph_embeddings(&self, embeddings: Vec<Vector>) -> Result<Vector> {
        if embeddings.is_empty() {
            return Err(anyhow!("No subgraph embeddings to aggregate"));
        }

        // Use centroid calculation for now - could be enhanced with graph attention
        let dimensions = embeddings[0].dimensions;
        let mut centroid = vec![0.0; dimensions];

        for embedding in &embeddings {
            let values = embedding.as_f32();
            for (i, value) in values.iter().enumerate() {
                centroid[i] += value;
            }
        }

        let count = embeddings.len() as f32;
        for value in &mut centroid {
            *value /= count;
        }

        Ok(Vector::new(centroid))
    }

    /// Decompose URI into meaningful components
    fn decompose_uri(&self, uri: &str) -> Vec<String> {
        let mut components = Vec::new();

        // Extract domain
        if let Some(domain_start) = uri.find("://") {
            if let Some(domain_end) = uri[domain_start + 3..].find('/') {
                let domain = &uri[domain_start + 3..domain_start + 3 + domain_end];
                components.push(domain.to_string());
            }
        }

        // Extract path segments
        if let Some(path_start) = uri.rfind('/') {
            let fragment = &uri[path_start + 1..];
            if !fragment.is_empty() {
                // Split camelCase and snake_case
                components.extend(self.split_identifier(fragment));
            }
        }

        // Extract fragment
        if let Some(fragment_start) = uri.find('#') {
            let fragment = &uri[fragment_start + 1..];
            if !fragment.is_empty() {
                components.extend(self.split_identifier(fragment));
            }
        }

        components
    }

    /// Split identifier into words (camelCase, snake_case, etc.)
    fn split_identifier(&self, identifier: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut current_word = String::new();

        for ch in identifier.chars() {
            if ch.is_uppercase() && !current_word.is_empty() {
                words.push(current_word.to_lowercase());
                current_word = ch.to_string();
            } else if ch == '_' || ch == '-' {
                if !current_word.is_empty() {
                    words.push(current_word.to_lowercase());
                    current_word.clear();
                }
            } else {
                current_word.push(ch);
            }
        }

        if !current_word.is_empty() {
            words.push(current_word.to_lowercase());
        }

        words
    }

    /// Clear caches
    pub fn clear_cache(&mut self) {
        self.entity_cache.clear();
        self.relationship_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.entity_cache.len(), self.relationship_cache.len())
    }
}

/// Property aggregator for combining property embeddings
pub struct PropertyAggregator {
    aggregation_strategy: AggregationStrategy,
}

#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    Mean,
    WeightedMean,
    Attention,
    Concatenation,
}

impl PropertyAggregator {
    pub fn new() -> Self {
        Self {
            aggregation_strategy: AggregationStrategy::WeightedMean,
        }
    }

    pub fn aggregate_properties(
        &self,
        properties: &HashMap<String, Vec<RdfValue>>,
    ) -> Result<Vector> {
        let mut property_embeddings = Vec::new();

        for (property_uri, values) in properties {
            let mut property_text = property_uri.clone();

            // Aggregate values for this property
            for value in values {
                match value {
                    RdfValue::IRI(iri) => property_text.push_str(&format!(" {}", iri)),
                    RdfValue::Literal(lit, _) => property_text.push_str(&format!(" {}", lit)),
                    RdfValue::LangString(lit, _) => property_text.push_str(&format!(" {}", lit)),
                    RdfValue::Boolean(b) => property_text.push_str(&format!(" {}", b)),
                    RdfValue::Integer(i) => property_text.push_str(&format!(" {}", i)),
                    RdfValue::Float(f) => property_text.push_str(&format!(" {}", f)),
                    RdfValue::Date(d) | RdfValue::DateTime(d) => {
                        property_text.push_str(&format!(" {}", d))
                    }
                }
            }

            // For now, create a simple text-based embedding
            // In a real implementation, we'd use the embedding manager
            let embedding = self.create_simple_embedding(&property_text);
            property_embeddings.push(embedding);
        }

        if property_embeddings.is_empty() {
            return Ok(Vector::new(vec![0.0; 384])); // Default empty embedding
        }

        // Aggregate using the selected strategy
        match self.aggregation_strategy {
            AggregationStrategy::Mean => self.mean_aggregation(property_embeddings),
            AggregationStrategy::WeightedMean => {
                self.weighted_mean_aggregation(property_embeddings)
            }
            _ => self.mean_aggregation(property_embeddings), // Fallback to mean
        }
    }

    fn mean_aggregation(&self, embeddings: Vec<Vector>) -> Result<Vector> {
        if embeddings.is_empty() {
            return Err(anyhow!("No embeddings to aggregate"));
        }

        let dimensions = embeddings[0].dimensions;
        let mut mean = vec![0.0; dimensions];

        for embedding in &embeddings {
            let values = embedding.as_f32();
            for (i, value) in values.iter().enumerate() {
                mean[i] += value;
            }
        }

        let count = embeddings.len() as f32;
        for value in &mut mean {
            *value /= count;
        }

        Ok(Vector::new(mean))
    }

    fn weighted_mean_aggregation(&self, embeddings: Vec<Vector>) -> Result<Vector> {
        // For now, use equal weights (same as mean)
        // Could be enhanced with property importance weighting
        self.mean_aggregation(embeddings)
    }

    fn create_simple_embedding(&self, text: &str) -> Vector {
        // Simple hash-based embedding for fallback
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut values = Vec::with_capacity(384);
        let mut seed = hash;

        for _ in 0..384 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (seed as f32) / (u64::MAX as f32);
            values.push((normalized - 0.5) * 2.0);
        }

        Vector::new(values)
    }
}

impl Default for PropertyAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-language processor for handling international RDF data
pub struct MultiLanguageProcessor {
    language_weights: HashMap<String, f32>,
    fallback_language: String,
}

impl MultiLanguageProcessor {
    pub fn new() -> Self {
        let mut language_weights = HashMap::new();
        language_weights.insert("en".to_string(), 1.0);
        language_weights.insert("es".to_string(), 0.8);
        language_weights.insert("fr".to_string(), 0.8);
        language_weights.insert("de".to_string(), 0.8);
        language_weights.insert("zh".to_string(), 0.7);
        language_weights.insert("ja".to_string(), 0.7);

        Self {
            language_weights,
            fallback_language: "en".to_string(),
        }
    }

    pub fn get_preferred_text(&self, texts: &HashMap<String, String>) -> Option<String> {
        // Sort languages by weight in descending order
        let mut sorted_langs: Vec<_> = self.language_weights.iter().collect();
        sorted_langs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Try to find text in order of preference
        for (lang, _) in sorted_langs {
            if let Some(text) = texts.get(lang) {
                return Some(text.clone());
            }
        }

        // Fallback to any available text
        texts.values().next().cloned()
    }

    pub fn get_language_weight(&self, language: &str) -> f32 {
        self.language_weights.get(language).copied().unwrap_or(0.5)
    }
}

impl Default for MultiLanguageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingStrategy;

    #[test]
    fn test_rdf_entity_creation() {
        let mut labels = HashMap::new();
        labels.insert("en".to_string(), "Person".to_string());
        labels.insert("es".to_string(), "Persona".to_string());

        let entity = RdfEntity {
            uri: "http://example.org/Person".to_string(),
            labels,
            descriptions: HashMap::new(),
            properties: HashMap::new(),
            types: vec!["http://www.w3.org/2000/01/rdf-schema#Class".to_string()],
            context: None,
            temporal_info: None,
        };

        assert_eq!(entity.uri, "http://example.org/Person");
        assert_eq!(entity.labels.len(), 2);
    }

    #[test]
    fn test_property_path() {
        let path = PropertyPath {
            path: vec![
                "http://example.org/knows".to_string(),
                "http://example.org/worksAt".to_string(),
            ],
            direction: vec![PathDirection::Forward, PathDirection::Forward],
            constraints: vec![PathConstraint::TypeFilter(
                "http://example.org/Person".to_string(),
            )],
        };

        assert_eq!(path.path.len(), 2);
        assert_eq!(path.direction.len(), 2);
        assert_eq!(path.constraints.len(), 1);
    }

    #[test]
    fn test_uri_decomposition() {
        let config = RdfContentConfig::default();
        let processor = RdfContentProcessor::new(config, EmbeddingStrategy::TfIdf).unwrap();

        let components = processor.decompose_uri("http://example.org/ontology/PersonClass");
        assert!(components.contains(&"example.org".to_string()));
        assert!(components.contains(&"person".to_string()));
        assert!(components.contains(&"class".to_string()));
    }

    #[test]
    fn test_identifier_splitting() {
        let config = RdfContentConfig::default();
        let processor = RdfContentProcessor::new(config, EmbeddingStrategy::TfIdf).unwrap();

        let words = processor.split_identifier("PersonClass");
        assert_eq!(words, vec!["person", "class"]);

        let words = processor.split_identifier("person_class");
        assert_eq!(words, vec!["person", "class"]);

        let words = processor.split_identifier("person-class");
        assert_eq!(words, vec!["person", "class"]);
    }

    #[test]
    fn test_multi_language_processor() {
        let processor = MultiLanguageProcessor::new();

        let mut texts = HashMap::new();
        texts.insert("en".to_string(), "Hello".to_string());
        texts.insert("es".to_string(), "Hola".to_string());

        let preferred = processor.get_preferred_text(&texts);
        assert_eq!(preferred, Some("Hello".to_string())); // English should be preferred
    }
}
