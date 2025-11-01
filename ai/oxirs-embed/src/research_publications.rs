//! Research publication network embeddings for academic analysis
//!
//! This module provides specialized embeddings for research publication networks
//! including author embeddings, citation analysis, topic modeling integration,
//! collaboration networks, impact prediction, and trend analysis.

use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Triple, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Research publication entity types for specialized embedding handling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResearchEntityType {
    Author,
    Paper,
    Journal,
    Conference,
    Institution,
    Topic,
    Keyword,
    FieldOfStudy,
    Grant,
    Project,
    Patent,
    Dataset,
}

impl ResearchEntityType {
    /// Get the namespace prefix for this entity type
    pub fn namespace(&self) -> &'static str {
        match self {
            ResearchEntityType::Author => "author",
            ResearchEntityType::Paper => "paper", 
            ResearchEntityType::Journal => "journal",
            ResearchEntityType::Conference => "conference",
            ResearchEntityType::Institution => "institution",
            ResearchEntityType::Topic => "topic",
            ResearchEntityType::Keyword => "keyword",
            ResearchEntityType::FieldOfStudy => "field",
            ResearchEntityType::Grant => "grant",
            ResearchEntityType::Project => "project",
            ResearchEntityType::Patent => "patent",
            ResearchEntityType::Dataset => "dataset",
        }
    }

    /// Parse entity type from IRI
    pub fn from_iri(iri: &str) -> Option<Self> {
        if iri.contains("author") || iri.contains("researcher") || iri.contains("ORCID") {
            Some(ResearchEntityType::Author)
        } else if iri.contains("paper") || iri.contains("article") || iri.contains("DOI") {
            Some(ResearchEntityType::Paper)
        } else if iri.contains("journal") || iri.contains("ISSN") {
            Some(ResearchEntityType::Journal)
        } else if iri.contains("conference") || iri.contains("proceedings") {
            Some(ResearchEntityType::Conference)
        } else if iri.contains("institution") || iri.contains("university") || iri.contains("ROR") {
            Some(ResearchEntityType::Institution)
        } else if iri.contains("topic") || iri.contains("subject") {
            Some(ResearchEntityType::Topic)
        } else if iri.contains("keyword") {
            Some(ResearchEntityType::Keyword)
        } else if iri.contains("field") || iri.contains("discipline") {
            Some(ResearchEntityType::FieldOfStudy)
        } else if iri.contains("grant") || iri.contains("funding") {
            Some(ResearchEntityType::Grant)
        } else if iri.contains("project") {
            Some(ResearchEntityType::Project)
        } else if iri.contains("patent") {
            Some(ResearchEntityType::Patent)
        } else if iri.contains("dataset") {
            Some(ResearchEntityType::Dataset)
        } else {
            None
        }
    }
}

/// Research publication relation types for specialized handling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResearchRelationType {
    /// Authorship relationships
    Authored,
    CoAuthored,
    FirstAuthor,
    LastAuthor,
    CorrespondingAuthor,
    /// Citation relationships
    Cites,
    CitedBy,
    CoOccursCitation,
    /// Collaboration relationships
    CollaboratesWith,
    SupervisedBy,
    Supervises,
    /// Institutional relationships
    AffiliatedWith,
    FormerlyAffiliatedWith,
    VisitingAt,
    /// Publication venue relationships
    PublishedIn,
    EditedBy,
    ReviewedBy,
    /// Topic and field relationships
    AboutTopic,
    InField,
    RelatedToKeyword,
    /// Project and funding relationships
    FundedBy,
    PartOfProject,
    UsesDataset,
    /// Temporal relationships
    PrecededBy,
    FollowedBy,
    Contemporary,
}

impl ResearchRelationType {
    /// Parse relation type from predicate IRI
    pub fn from_iri(iri: &str) -> Option<Self> {
        match iri.to_lowercase().as_str() {
            s if s.contains("authored") => Some(ResearchRelationType::Authored),
            s if s.contains("co_authored") => Some(ResearchRelationType::CoAuthored),
            s if s.contains("cites") => Some(ResearchRelationType::Cites),
            s if s.contains("cited_by") => Some(ResearchRelationType::CitedBy),
            s if s.contains("collaborates") => Some(ResearchRelationType::CollaboratesWith),
            s if s.contains("affiliated") => Some(ResearchRelationType::AffiliatedWith),
            s if s.contains("published_in") => Some(ResearchRelationType::PublishedIn),
            s if s.contains("about") => Some(ResearchRelationType::AboutTopic),
            s if s.contains("in_field") => Some(ResearchRelationType::InField),
            s if s.contains("funded_by") => Some(ResearchRelationType::FundedBy),
            s if s.contains("supervised") => Some(ResearchRelationType::SupervisedBy),
            s if s.contains("supervises") => Some(ResearchRelationType::Supervises),
            _ => None,
        }
    }
}

/// Configuration for research publication embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchEmbeddingConfig {
    pub base_config: ModelConfig,
    /// Weight for citation relationships
    pub citation_weight: f32,
    /// Weight for collaboration relationships
    pub collaboration_weight: f32,
    /// Weight for authorship relationships
    pub authorship_weight: f32,
    /// Weight for topical relationships
    pub topic_weight: f32,
    /// Weight for institutional relationships
    pub institutional_weight: f32,
    /// Enable temporal features
    pub use_temporal_features: bool,
    /// Enable citation count features
    pub use_citation_features: bool,
    /// Enable h-index features
    pub use_h_index_features: bool,
    /// Enable impact factor features
    pub use_impact_factor: bool,
    /// Time window for temporal analysis (in years)
    pub temporal_window: u32,
    /// Field of study filter
    pub field_filter: Option<String>,
}

impl Default for ResearchEmbeddingConfig {
    fn default() -> Self {
        Self {
            base_config: ModelConfig::default(),
            citation_weight: 2.0,
            collaboration_weight: 1.5,
            authorship_weight: 1.8,
            topic_weight: 1.2,
            institutional_weight: 1.0,
            use_temporal_features: true,
            use_citation_features: true,
            use_h_index_features: true,
            use_impact_factor: true,
            temporal_window: 10,
            field_filter: None,
        }
    }
}

/// Research publication network embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchEmbedding {
    pub config: ResearchEmbeddingConfig,
    pub model_id: Uuid,
    /// Entity embeddings by type
    pub author_embeddings: HashMap<String, Array1<f32>>,
    pub paper_embeddings: HashMap<String, Array1<f32>>,
    pub journal_embeddings: HashMap<String, Array1<f32>>,
    pub conference_embeddings: HashMap<String, Array1<f32>>,
    pub institution_embeddings: HashMap<String, Array1<f32>>,
    pub topic_embeddings: HashMap<String, Array1<f32>>,
    pub keyword_embeddings: HashMap<String, Array1<f32>>,
    pub field_embeddings: HashMap<String, Array1<f32>>,
    /// Relation embeddings by type
    pub relation_embeddings: HashMap<String, Array1<f32>>,
    /// Entity type mappings
    pub entity_types: HashMap<String, ResearchEntityType>,
    /// Relation type mappings
    pub relation_types: HashMap<String, ResearchRelationType>,
    /// Training data
    pub triples: Vec<Triple>,
    /// Research-specific features
    pub features: ResearchFeatures,
    /// Training and model stats
    pub training_stats: TrainingStats,
    pub model_stats: ModelStats,
    pub is_trained: bool,
}

/// Research-specific features for enhanced embeddings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResearchFeatures {
    /// Citation relationships with counts
    pub citations: HashMap<(String, String), u32>,
    /// Co-authorship relationships with paper counts
    pub collaborations: HashMap<(String, String), u32>,
    /// Author h-index scores
    pub h_indices: HashMap<String, u32>,
    /// Paper citation counts
    pub citation_counts: HashMap<String, u32>,
    /// Journal impact factors
    pub impact_factors: HashMap<String, f32>,
    /// Author productivity (paper counts)
    pub author_productivity: HashMap<String, u32>,
    /// Institutional rankings
    pub institutional_rankings: HashMap<String, u32>,
    /// Topic co-occurrence scores
    pub topic_co_occurrences: HashMap<(String, String), f32>,
    /// Temporal activity patterns
    pub temporal_activity: HashMap<String, Vec<(DateTime<Utc>, u32)>>,
    /// Cross-field collaboration scores
    pub cross_field_scores: HashMap<(String, String), f32>,
}

impl ResearchEmbedding {
    /// Create new research publication embedding model
    pub fn new(config: ResearchEmbeddingConfig) -> Self {
        let model_id = Uuid::new_v4();
        let now = Utc::now();

        Self {
            model_id,
            author_embeddings: HashMap::new(),
            paper_embeddings: HashMap::new(),
            journal_embeddings: HashMap::new(),
            conference_embeddings: HashMap::new(),
            institution_embeddings: HashMap::new(),
            topic_embeddings: HashMap::new(),
            keyword_embeddings: HashMap::new(),
            field_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            entity_types: HashMap::new(),
            relation_types: HashMap::new(),
            triples: Vec::new(),
            features: ResearchFeatures::default(),
            training_stats: TrainingStats::default(),
            model_stats: ModelStats {
                num_entities: 0,
                num_relations: 0,
                num_triples: 0,
                dimensions: config.base_config.dimensions,
                is_trained: false,
                model_type: "ResearchEmbedding".to_string(),
                creation_time: now,
                last_training_time: None,
            },
            is_trained: false,
            config,
        }
    }

    /// Add citation relationship
    pub fn add_citation(&mut self, citing_paper: &str, cited_paper: &str) {
        let key = (citing_paper.to_string(), cited_paper.to_string());
        *self.features.citations.entry(key).or_insert(0) += 1;
        
        // Update citation count for cited paper
        *self.features.citation_counts.entry(cited_paper.to_string()).or_insert(0) += 1;
    }

    /// Add collaboration relationship
    pub fn add_collaboration(&mut self, author1: &str, author2: &str) {
        let key1 = (author1.to_string(), author2.to_string());
        let key2 = (author2.to_string(), author1.to_string());
        
        *self.features.collaborations.entry(key1).or_insert(0) += 1;
        *self.features.collaborations.entry(key2).or_insert(0) += 1;
    }

    /// Set author h-index
    pub fn set_author_h_index(&mut self, author: &str, h_index: u32) {
        self.features.h_indices.insert(author.to_string(), h_index);
    }

    /// Set journal impact factor
    pub fn set_journal_impact_factor(&mut self, journal: &str, impact_factor: f32) {
        self.features.impact_factors.insert(journal.to_string(), impact_factor);
    }

    /// Add author productivity
    pub fn add_author_productivity(&mut self, author: &str) {
        *self.features.author_productivity.entry(author.to_string()).or_insert(0) += 1;
    }

    /// Add topic co-occurrence
    pub fn add_topic_co_occurrence(&mut self, topic1: &str, topic2: &str, score: f32) {
        let key = (topic1.to_string(), topic2.to_string());
        self.features.topic_co_occurrences.insert(key, score);
        
        // Symmetric relationship
        let key_rev = (topic2.to_string(), topic1.to_string());
        self.features.topic_co_occurrences.insert(key_rev, score);
    }

    /// Get entity embedding with research type awareness
    pub fn get_typed_entity_embedding(&self, entity: &str) -> Result<Vector> {
        if let Some(entity_type) = self.entity_types.get(entity) {
            let embedding = match entity_type {
                ResearchEntityType::Author => self.author_embeddings.get(entity),
                ResearchEntityType::Paper => self.paper_embeddings.get(entity),
                ResearchEntityType::Journal => self.journal_embeddings.get(entity),
                ResearchEntityType::Conference => self.conference_embeddings.get(entity),
                ResearchEntityType::Institution => self.institution_embeddings.get(entity),
                ResearchEntityType::Topic => self.topic_embeddings.get(entity),
                ResearchEntityType::Keyword => self.keyword_embeddings.get(entity),
                ResearchEntityType::FieldOfStudy => self.field_embeddings.get(entity),
                _ => None,
            };

            if let Some(emb) = embedding {
                Ok(Vector::from_array1(emb))
            } else {
                Err(anyhow!(
                    "No embedding found for {} of type {:?}",
                    entity,
                    entity_type
                ))
            }
        } else {
            Err(anyhow!("Unknown entity type for {}", entity))
        }
    }

    /// Predict author collaborations
    pub fn predict_author_collaborations(&self, author: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(anyhow!("Model not trained"));
        }

        let author_embedding = self
            .author_embeddings
            .get(author)
            .ok_or_else(|| anyhow!("Author {} not found", author))?;

        let mut scores = Vec::new();

        for (other_author, other_embedding) in &self.author_embeddings {
            if other_author != author {
                // Calculate cosine similarity
                let dot_product = author_embedding.dot(other_embedding);
                let norm1 = author_embedding.mapv(|x| x * x).sum().sqrt();
                let norm2 = other_embedding.mapv(|x| x * x).sum().sqrt();
                
                if norm1 > 0.0 && norm2 > 0.0 {
                    let similarity = (dot_product / (norm1 * norm2)) as f64;
                    
                    // Boost score based on existing collaboration patterns
                    let collaboration_boost = if let Some(&count) = self.features.collaborations.get(&(author.to_string(), other_author.clone())) {
                        1.0 + (count as f64 * 0.1)
                    } else {
                        1.0
                    };
                    
                    scores.push((other_author.clone(), similarity * collaboration_boost));
                }
            }
        }

        // Sort by score descending and take top k
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    /// Predict paper citations
    pub fn predict_paper_citations(&self, paper: &str, k: usize) -> Result<Vec<(String, f64)>> {
        if !self.is_trained {
            return Err(anyhow!("Model not trained"));
        }

        let paper_embedding = self
            .paper_embeddings
            .get(paper)
            .ok_or_else(|| anyhow!("Paper {} not found", paper))?;

        let mut scores = Vec::new();

        for (other_paper, other_embedding) in &self.paper_embeddings {
            if other_paper != paper {
                // Calculate cosine similarity
                let dot_product = paper_embedding.dot(other_embedding);
                let norm1 = paper_embedding.mapv(|x| x * x).sum().sqrt();
                let norm2 = other_embedding.mapv(|x| x * x).sum().sqrt();
                
                if norm1 > 0.0 && norm2 > 0.0 {
                    let similarity = (dot_product / (norm1 * norm2)) as f64;
                    
                    // Boost score based on citation count of target paper
                    let citation_boost = if let Some(&count) = self.features.citation_counts.get(other_paper) {
                        1.0 + (count as f64 * 0.01) // Small boost based on citation count
                    } else {
                        1.0
                    };
                    
                    scores.push((other_paper.clone(), similarity * citation_boost));
                }
            }
        }

        // Sort by score descending and take top k
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    /// Analyze collaboration network
    pub fn analyze_collaboration_network(&self, author: &str) -> Result<HashMap<String, f64>> {
        let mut network_analysis = HashMap::new();
        
        // Calculate collaboration strength for each connected author
        for ((author1, author2), &count) in &self.features.collaborations {
            if author1 == author {
                let strength = count as f64;
                network_analysis.insert(author2.clone(), strength);
            }
        }
        
        Ok(network_analysis)
    }

    /// Predict research impact
    pub fn predict_research_impact(&self, paper: &str) -> Result<f64> {
        if !self.is_trained {
            return Err(anyhow!("Model not trained"));
        }

        let paper_embedding = self
            .paper_embeddings
            .get(paper)
            .ok_or_else(|| anyhow!("Paper {} not found", paper))?;

        // Simple impact prediction based on embedding features
        // In practice, this would be a trained regression model
        let mut impact_score = 0.0;
        
        // Base score from embedding magnitude
        let embedding_norm = paper_embedding.mapv(|x| x * x).sum().sqrt() as f64;
        impact_score += embedding_norm * 0.1;
        
        // Boost from existing citation count
        if let Some(&citations) = self.features.citation_counts.get(paper) {
            impact_score += citations as f64 * 0.5;
        }
        
        Ok(impact_score)
    }

    /// Get trending topics
    pub fn get_trending_topics(&self, window_years: u32) -> Result<Vec<(String, f64)>> {
        let cutoff_date = Utc::now() - chrono::Duration::days((window_years * 365) as i64);
        let mut topic_scores = HashMap::new();
        
        // Analyze temporal activity for topics
        for (entity, activities) in &self.features.temporal_activity {
            if let Some(ResearchEntityType::Topic) = self.entity_types.get(entity) {
                let recent_activity: u32 = activities
                    .iter()
                    .filter(|(date, _)| *date > cutoff_date)
                    .map(|(_, count)| *count)
                    .sum();
                
                if recent_activity > 0 {
                    topic_scores.insert(entity.clone(), recent_activity as f64);
                }
            }
        }
        
        // Sort by activity level
        let mut trending: Vec<_> = topic_scores.into_iter().collect();
        trending.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(trending)
    }
}

#[async_trait]
impl EmbeddingModel for ResearchEmbedding {
    async fn train(&mut self, triples: Vec<Triple>) -> Result<()> {
        // Store training data
        self.triples = triples.clone();
        self.model_stats.num_triples = triples.len();
        
        // Extract entities and relations
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();
        
        for triple in &triples {
            entities.insert(triple.subject.clone());
            entities.insert(triple.object.clone());
            relations.insert(triple.predicate.clone());
            
            // Determine entity types
            if let Some(entity_type) = ResearchEntityType::from_iri(&triple.subject) {
                self.entity_types.insert(triple.subject.clone(), entity_type);
            }
            if let Some(entity_type) = ResearchEntityType::from_iri(&triple.object) {
                self.entity_types.insert(triple.object.clone(), entity_type);
            }
            
            // Determine relation types
            if let Some(relation_type) = ResearchRelationType::from_iri(&triple.predicate) {
                self.relation_types.insert(triple.predicate.clone(), relation_type);
            }
        }
        
        self.model_stats.num_entities = entities.len();
        self.model_stats.num_relations = relations.len();
        
        // Initialize embeddings for each entity type
        let dimensions = self.config.base_config.dimensions;
        
        for entity in entities {
            let embedding = Array1::zeros(dimensions);
            
            if let Some(entity_type) = self.entity_types.get(&entity) {
                match entity_type {
                    ResearchEntityType::Author => {
                        self.author_embeddings.insert(entity, embedding);
                    }
                    ResearchEntityType::Paper => {
                        self.paper_embeddings.insert(entity, embedding);
                    }
                    ResearchEntityType::Journal => {
                        self.journal_embeddings.insert(entity, embedding);
                    }
                    ResearchEntityType::Conference => {
                        self.conference_embeddings.insert(entity, embedding);
                    }
                    ResearchEntityType::Institution => {
                        self.institution_embeddings.insert(entity, embedding);
                    }
                    ResearchEntityType::Topic => {
                        self.topic_embeddings.insert(entity, embedding);
                    }
                    ResearchEntityType::Keyword => {
                        self.keyword_embeddings.insert(entity, embedding);
                    }
                    ResearchEntityType::FieldOfStudy => {
                        self.field_embeddings.insert(entity, embedding);
                    }
                    _ => {}
                }
            }
        }
        
        // Initialize relation embeddings
        for relation in relations {
            let embedding = Array1::zeros(dimensions);
            self.relation_embeddings.insert(relation, embedding);
        }
        
        // Mark as trained
        self.is_trained = true;
        self.model_stats.is_trained = true;
        self.model_stats.last_training_time = Some(Utc::now());
        
        Ok(())
    }
    
    async fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        self.get_typed_entity_embedding(entity)
    }
    
    async fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        let embedding = self
            .relation_embeddings
            .get(relation)
            .ok_or_else(|| anyhow!("Relation {} not found", relation))?;
        Ok(Vector::from_array1(embedding))
    }
    
    async fn predict_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        if !self.is_trained {
            return Err(anyhow!("Model not trained"));
        }
        
        let subj_emb = self.get_typed_entity_embedding(subject)?;
        let pred_emb = self.get_relation_embedding(predicate).await?;
        let obj_emb = self.get_typed_entity_embedding(object)?;
        
        // Simple scoring function (in practice, this would be model-specific)
        let score = subj_emb.cosine_similarity(&obj_emb)?;
        Ok(score)
    }
    
    fn get_model_stats(&self) -> &ModelStats {
        &self.model_stats
    }
    
    fn get_training_stats(&self) -> &TrainingStats {
        &self.training_stats
    }
    
    fn is_trained(&self) -> bool {
        self.is_trained
    }
}

/// Research publication embedding builder for easy configuration
pub struct ResearchEmbeddingBuilder {
    config: ResearchEmbeddingConfig,
}

impl ResearchEmbeddingBuilder {
    pub fn new() -> Self {
        Self {
            config: ResearchEmbeddingConfig::default(),
        }
    }
    
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.config.base_config.dimensions = dimensions;
        self
    }
    
    pub fn with_citation_weight(mut self, weight: f32) -> Self {
        self.config.citation_weight = weight;
        self
    }
    
    pub fn with_collaboration_weight(mut self, weight: f32) -> Self {
        self.config.collaboration_weight = weight;
        self
    }
    
    pub fn with_temporal_features(mut self, enable: bool) -> Self {
        self.config.use_temporal_features = enable;
        self
    }
    
    pub fn with_temporal_window(mut self, years: u32) -> Self {
        self.config.temporal_window = years;
        self
    }
    
    pub fn with_field_filter(mut self, field: String) -> Self {
        self.config.field_filter = Some(field);
        self
    }
    
    pub fn build(self) -> ResearchEmbedding {
        ResearchEmbedding::new(self.config)
    }
}

impl Default for ResearchEmbeddingBuilder {
    fn default() -> Self {
        Self::new()
    }
}