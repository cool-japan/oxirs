//! Multi-modal embeddings and cross-modal alignment for unified representation learning
//!
//! This module provides advanced multi-modal integration capabilities for combining
//! text, knowledge graph, and other modalities into unified embedding spaces.

use crate::{EmbeddingModel, ModelConfig, ModelStats, TrainingStats, Vector};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Cross-modal alignment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    pub base_config: ModelConfig,
    /// Text embedding dimension
    pub text_dim: usize,
    /// Knowledge graph embedding dimension
    pub kg_dim: usize,
    /// Unified embedding dimension
    pub unified_dim: usize,
    /// Alignment objective type
    pub alignment_objective: AlignmentObjective,
    /// Contrastive learning parameters
    pub contrastive_config: ContrastiveConfig,
    /// Multi-task learning weights
    pub task_weights: HashMap<String, f32>,
    /// Cross-domain transfer settings
    pub cross_domain_config: CrossDomainConfig,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        let mut task_weights = HashMap::new();
        task_weights.insert("text_kg_alignment".to_string(), 1.0);
        task_weights.insert("entity_description".to_string(), 0.8);
        task_weights.insert("property_text".to_string(), 0.6);
        task_weights.insert("multilingual".to_string(), 0.4);

        Self {
            base_config: ModelConfig::default(),
            text_dim: 768,
            kg_dim: 128,
            unified_dim: 512,
            alignment_objective: AlignmentObjective::ContrastiveLearning,
            contrastive_config: ContrastiveConfig::default(),
            task_weights,
            cross_domain_config: CrossDomainConfig::default(),
        }
    }
}

/// Alignment objective types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentObjective {
    /// Contrastive learning for positive/negative pairs
    ContrastiveLearning,
    /// Mutual information maximization
    MutualInformation,
    /// Adversarial alignment with discriminator
    AdversarialAlignment,
    /// Multi-task learning with shared representations
    MultiTaskLearning,
    /// Self-supervised objectives
    SelfSupervised,
    /// Meta-learning for few-shot adaptation
    MetaLearning,
}

/// Contrastive learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    /// Temperature parameter for contrastive loss
    pub temperature: f32,
    /// Number of negative samples
    pub negative_samples: usize,
    /// Hard negative mining
    pub hard_negative_mining: bool,
    /// Margin for triplet loss
    pub margin: f32,
    /// Use InfoNCE loss
    pub use_info_nce: bool,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            negative_samples: 64,
            hard_negative_mining: true,
            margin: 0.2,
            use_info_nce: true,
        }
    }
}

/// Cross-domain transfer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainConfig {
    /// Enable domain adaptation
    pub enable_domain_adaptation: bool,
    /// Source domains for transfer learning
    pub source_domains: Vec<String>,
    /// Target domains
    pub target_domains: Vec<String>,
    /// Domain adversarial training
    pub domain_adversarial: bool,
    /// Gradual domain adaptation
    pub gradual_adaptation: bool,
}

impl Default for CrossDomainConfig {
    fn default() -> Self {
        Self {
            enable_domain_adaptation: true,
            source_domains: vec!["general".to_string(), "scientific".to_string()],
            target_domains: vec!["biomedical".to_string(), "legal".to_string()],
            domain_adversarial: false,
            gradual_adaptation: true,
        }
    }
}

/// Multi-modal embedding model for unified representation learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalEmbedding {
    pub config: CrossModalConfig,
    pub model_id: Uuid,
    /// Text embeddings cache
    pub text_embeddings: HashMap<String, Array1<f32>>,
    /// Knowledge graph embeddings cache
    pub kg_embeddings: HashMap<String, Array1<f32>>,
    /// Unified cross-modal embeddings
    pub unified_embeddings: HashMap<String, Array1<f32>>,
    /// Cross-modal alignment mappings
    pub text_kg_alignments: HashMap<String, String>,
    /// Entity descriptions for alignment
    pub entity_descriptions: HashMap<String, String>,
    /// Property-text mappings
    pub property_texts: HashMap<String, String>,
    /// Multi-language mappings
    pub multilingual_mappings: HashMap<String, Vec<String>>,
    /// Cross-domain mappings
    pub cross_domain_mappings: HashMap<String, String>,
    /// Training components
    pub text_encoder: TextEncoder,
    pub kg_encoder: KGEncoder,
    pub alignment_network: AlignmentNetwork,
    /// Training statistics
    pub training_stats: TrainingStats,
    pub model_stats: ModelStats,
    pub is_trained: bool,
}

/// Text encoder for multi-modal embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEncoder {
    /// Encoder type (BERT, RoBERTa, etc.)
    pub encoder_type: String,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Learned parameters (simplified representation)
    pub parameters: HashMap<String, Array2<f32>>,
}

impl TextEncoder {
    pub fn new(encoder_type: String, input_dim: usize, output_dim: usize) -> Self {
        let mut parameters = HashMap::new();

        // Initialize key transformation matrices
        parameters.insert(
            "projection".to_string(),
            Array2::from_shape_fn((output_dim, input_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        parameters.insert(
            "attention".to_string(),
            Array2::from_shape_fn((output_dim, output_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        Self {
            encoder_type,
            input_dim,
            output_dim,
            parameters,
        }
    }

    /// Encode text into embeddings
    pub fn encode(&self, text: &str) -> Result<Array1<f32>> {
        let input_features = self.extract_text_features(text);
        let projection = self.parameters.get("projection").unwrap();

        // Simple linear projection (in real implementation would be full transformer)
        let encoded = projection.dot(&input_features);

        // Apply layer normalization
        let mean = encoded.mean().unwrap_or(0.0);
        let var = encoded.var(0.0);
        let normalized = encoded.mapv(|x| (x - mean) / (var + 1e-8).sqrt());

        Ok(normalized)
    }

    /// Extract features from text (simplified)
    fn extract_text_features(&self, text: &str) -> Array1<f32> {
        let mut features = vec![0.0; self.input_dim];

        // Simple bag-of-words features (would be tokenization + embeddings in real implementation)
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if i < self.input_dim {
                features[i] = word.len() as f32 / 10.0; // Simple word length feature
            }
        }

        // Add sentence-level features
        if self.input_dim > words.len() {
            features[words.len()] = text.len() as f32 / 100.0; // Text length
            if self.input_dim > words.len() + 1 {
                features[words.len() + 1] = words.len() as f32 / 20.0; // Word count
            }
        }

        Array1::from_vec(features)
    }
}

/// Knowledge graph encoder for multi-modal embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KGEncoder {
    /// Encoder architecture (TransE, ComplEx, etc.)
    pub architecture: String,
    /// Entity embedding dimension
    pub entity_dim: usize,
    /// Relation embedding dimension
    pub relation_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Learned parameters
    pub parameters: HashMap<String, Array2<f32>>,
}

impl KGEncoder {
    pub fn new(
        architecture: String,
        entity_dim: usize,
        relation_dim: usize,
        output_dim: usize,
    ) -> Self {
        let mut parameters = HashMap::new();

        // Initialize transformation matrices
        parameters.insert(
            "entity_projection".to_string(),
            Array2::from_shape_fn((output_dim, entity_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        parameters.insert(
            "relation_projection".to_string(),
            Array2::from_shape_fn((output_dim, relation_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        Self {
            architecture,
            entity_dim,
            relation_dim,
            output_dim,
            parameters,
        }
    }

    /// Encode knowledge graph entity
    pub fn encode_entity(&self, entity_embedding: &Array1<f32>) -> Result<Array1<f32>> {
        let projection = self.parameters.get("entity_projection").unwrap();
        Ok(projection.dot(entity_embedding))
    }

    /// Encode knowledge graph relation
    pub fn encode_relation(&self, relation_embedding: &Array1<f32>) -> Result<Array1<f32>> {
        let projection = self.parameters.get("relation_projection").unwrap();
        Ok(projection.dot(relation_embedding))
    }

    /// Encode structured knowledge (entity + relations)
    pub fn encode_structured(
        &self,
        entity: &Array1<f32>,
        relations: &[Array1<f32>],
    ) -> Result<Array1<f32>> {
        let entity_encoded = self.encode_entity(entity)?;

        // Aggregate relation information
        let mut relation_agg = Array1::<f32>::zeros(self.output_dim);
        for relation in relations {
            let rel_encoded = self.encode_relation(relation)?;
            relation_agg = &relation_agg + &rel_encoded;
        }

        if !relations.is_empty() {
            relation_agg = relation_agg / relations.len() as f32;
        }

        // Combine entity and relation information
        Ok(&entity_encoded + &relation_agg)
    }
}

/// Alignment network for cross-modal learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentNetwork {
    /// Network architecture
    pub architecture: String,
    /// Input dimensions (text_dim, kg_dim)
    pub input_dims: (usize, usize),
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Network parameters
    pub parameters: HashMap<String, Array2<f32>>,
}

impl AlignmentNetwork {
    pub fn new(
        architecture: String,
        text_dim: usize,
        kg_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Self {
        let mut parameters = HashMap::new();

        // Text pathway
        parameters.insert(
            "text_hidden".to_string(),
            Array2::from_shape_fn((hidden_dim, text_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        parameters.insert(
            "text_output".to_string(),
            Array2::from_shape_fn((output_dim, hidden_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        // KG pathway
        parameters.insert(
            "kg_hidden".to_string(),
            Array2::from_shape_fn((hidden_dim, kg_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        parameters.insert(
            "kg_output".to_string(),
            Array2::from_shape_fn((output_dim, hidden_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        // Cross-modal attention
        parameters.insert(
            "cross_attention".to_string(),
            Array2::from_shape_fn((output_dim, output_dim), |(_, _)| {
                (rand::random::<f32>() - 0.5) * 0.1
            }),
        );

        Self {
            architecture,
            input_dims: (text_dim, kg_dim),
            hidden_dim,
            output_dim,
            parameters,
        }
    }

    /// Align text and KG embeddings
    pub fn align(
        &self,
        text_emb: &Array1<f32>,
        kg_emb: &Array1<f32>,
    ) -> Result<(Array1<f32>, f32)> {
        // Process text embedding
        let text_hidden_matrix = self.parameters.get("text_hidden").unwrap();
        let text_hidden = text_hidden_matrix.dot(text_emb);
        let text_hidden = text_hidden.mapv(|x| x.max(0.0)); // ReLU activation
        let text_output_matrix = self.parameters.get("text_output").unwrap();
        let text_output = text_output_matrix.dot(&text_hidden);

        // Process KG embedding
        let kg_hidden_matrix = self.parameters.get("kg_hidden").unwrap();
        let kg_hidden = kg_hidden_matrix.dot(kg_emb);
        let kg_hidden = kg_hidden.mapv(|x| x.max(0.0)); // ReLU activation
        let kg_output_matrix = self.parameters.get("kg_output").unwrap();
        let kg_output = kg_output_matrix.dot(&kg_hidden);

        // Cross-modal attention
        let attention_weights = self.compute_attention(&text_output, &kg_output)?;

        // Weighted combination
        let unified = &text_output * attention_weights + &kg_output * (1.0 - attention_weights);

        // Compute alignment score
        let alignment_score = self.compute_alignment_score(&text_output, &kg_output);

        Ok((unified, alignment_score))
    }

    /// Compute cross-modal attention weights
    fn compute_attention(&self, text_emb: &Array1<f32>, kg_emb: &Array1<f32>) -> Result<f32> {
        let attention_matrix = self.parameters.get("cross_attention").unwrap();
        let text_projected = attention_matrix.dot(text_emb);

        // Compute attention score
        let attention_score = text_projected.dot(kg_emb);
        let attention_weight = 1.0 / (1.0 + (-attention_score).exp()); // Sigmoid

        Ok(attention_weight)
    }

    /// Compute alignment score between modalities
    fn compute_alignment_score(&self, text_emb: &Array1<f32>, kg_emb: &Array1<f32>) -> f32 {
        // Cosine similarity
        let dot_product = text_emb.dot(kg_emb);
        let text_norm = text_emb.dot(text_emb).sqrt();
        let kg_norm = kg_emb.dot(kg_emb).sqrt();

        if text_norm > 0.0 && kg_norm > 0.0 {
            dot_product / (text_norm * kg_norm)
        } else {
            0.0
        }
    }
}

impl MultiModalEmbedding {
    /// Create new multi-modal embedding model
    pub fn new(config: CrossModalConfig) -> Self {
        let model_id = Uuid::from_u128(rand::random());
        let now = Utc::now();

        let text_encoder =
            TextEncoder::new("BERT".to_string(), config.text_dim, config.unified_dim);

        let kg_encoder = KGEncoder::new(
            "ComplEx".to_string(),
            config.kg_dim,
            config.kg_dim,
            config.unified_dim,
        );

        let alignment_network = AlignmentNetwork::new(
            "CrossModalAttention".to_string(),
            config.unified_dim,
            config.unified_dim,
            config.unified_dim / 2,
            config.unified_dim,
        );

        Self {
            model_id,
            text_embeddings: HashMap::new(),
            kg_embeddings: HashMap::new(),
            unified_embeddings: HashMap::new(),
            text_kg_alignments: HashMap::new(),
            entity_descriptions: HashMap::new(),
            property_texts: HashMap::new(),
            multilingual_mappings: HashMap::new(),
            cross_domain_mappings: HashMap::new(),
            text_encoder,
            kg_encoder,
            alignment_network,
            training_stats: TrainingStats::default(),
            model_stats: ModelStats {
                num_entities: 0,
                num_relations: 0,
                num_triples: 0,
                dimensions: config.unified_dim,
                is_trained: false,
                model_type: "MultiModalEmbedding".to_string(),
                creation_time: now,
                last_training_time: None,
            },
            is_trained: false,
            config,
        }
    }

    /// Add text-KG alignment pair
    pub fn add_text_kg_alignment(&mut self, text: &str, entity: &str) {
        self.text_kg_alignments
            .insert(text.to_string(), entity.to_string());
    }

    /// Add entity description
    pub fn add_entity_description(&mut self, entity: &str, description: &str) {
        self.entity_descriptions
            .insert(entity.to_string(), description.to_string());
    }

    /// Add property-text mapping
    pub fn add_property_text(&mut self, property: &str, text_description: &str) {
        self.property_texts
            .insert(property.to_string(), text_description.to_string());
    }

    /// Add multilingual mapping
    pub fn add_multilingual_mapping(&mut self, concept: &str, translations: Vec<String>) {
        self.multilingual_mappings
            .insert(concept.to_string(), translations);
    }

    /// Add cross-domain mapping
    pub fn add_cross_domain_mapping(&mut self, source_concept: &str, target_concept: &str) {
        self.cross_domain_mappings
            .insert(source_concept.to_string(), target_concept.to_string());
    }

    /// Generate unified embedding from text and KG
    pub async fn generate_unified_embedding(
        &mut self,
        text: &str,
        entity: &str,
    ) -> Result<Array1<f32>> {
        // Encode text
        let text_embedding = self.text_encoder.encode(text)?;

        // Get or create KG embedding (simplified - would use actual KG model)
        let kg_embedding_raw = self.get_or_create_kg_embedding(entity)?;

        // Encode KG embedding to unified dimension
        let kg_embedding = self.kg_encoder.encode_entity(&kg_embedding_raw)?;

        // Align modalities
        let (unified_embedding, alignment_score) = self
            .alignment_network
            .align(&text_embedding, &kg_embedding)?;

        // Cache embeddings - store raw KG embeddings to avoid dimension mismatch
        self.text_embeddings
            .insert(text.to_string(), text_embedding);
        self.kg_embeddings
            .insert(entity.to_string(), kg_embedding_raw); // Store raw, not encoded
        self.unified_embeddings
            .insert(format!("{}|{}", text, entity), unified_embedding.clone());

        println!(
            "Generated unified embedding with alignment score: {:.3}",
            alignment_score
        );

        Ok(unified_embedding)
    }

    /// Get or create KG embedding for entity
    fn get_or_create_kg_embedding(&self, entity: &str) -> Result<Array1<f32>> {
        if let Some(embedding) = self.kg_embeddings.get(entity) {
            Ok(embedding.clone())
        } else {
            // Create simple entity embedding based on name
            let mut embedding = vec![0.0; self.config.kg_dim];
            let entity_bytes = entity.as_bytes();

            for (i, &byte) in entity_bytes.iter().enumerate() {
                if i < self.config.kg_dim {
                    embedding[i] = (byte as f32 / 255.0 - 0.5) * 2.0;
                }
            }

            Ok(Array1::from_vec(embedding))
        }
    }

    /// Perform contrastive learning
    pub fn contrastive_loss(
        &self,
        positive_pairs: &[(String, String)],
        negative_pairs: &[(String, String)],
    ) -> Result<f32> {
        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();

        // Compute positive pair scores
        for (text, entity) in positive_pairs {
            if let (Some(text_emb), Some(kg_emb_raw)) = (
                self.text_embeddings.get(text),
                self.kg_embeddings.get(entity),
            ) {
                let kg_emb = self.kg_encoder.encode_entity(kg_emb_raw)?;
                let score = self
                    .alignment_network
                    .compute_alignment_score(text_emb, &kg_emb);
                positive_scores.push(score);
            }
        }

        // Compute negative pair scores
        for (text, entity) in negative_pairs {
            if let (Some(text_emb), Some(kg_emb_raw)) = (
                self.text_embeddings.get(text),
                self.kg_embeddings.get(entity),
            ) {
                let kg_emb = self.kg_encoder.encode_entity(kg_emb_raw)?;
                let score = self
                    .alignment_network
                    .compute_alignment_score(text_emb, &kg_emb);
                negative_scores.push(score);
            }
        }

        // Compute contrastive loss
        let temperature = self.config.contrastive_config.temperature;
        let mut loss = 0.0;

        for &pos_score in &positive_scores {
            let pos_exp = (pos_score / temperature).exp();
            let mut neg_sum = 0.0;

            for &neg_score in &negative_scores {
                neg_sum += (neg_score / temperature).exp();
            }

            if neg_sum > 0.0 {
                loss -= (pos_exp / (pos_exp + neg_sum)).ln();
            }
        }

        if !positive_scores.is_empty() {
            loss /= positive_scores.len() as f32;
        }

        Ok(loss)
    }

    /// Perform zero-shot learning
    pub async fn zero_shot_prediction(
        &self,
        text: &str,
        candidate_entities: &[String],
    ) -> Result<Vec<(String, f32)>> {
        let text_embedding = self.text_encoder.encode(text)?;
        let mut scores = Vec::new();

        for entity in candidate_entities {
            if let Some(kg_embedding_raw) = self.kg_embeddings.get(entity) {
                let kg_encoded = self.kg_encoder.encode_entity(kg_embedding_raw)?;
                let score = self
                    .alignment_network
                    .compute_alignment_score(&text_embedding, &kg_encoded);
                scores.push((entity.clone(), score));
            }
        }

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scores)
    }

    /// Cross-domain transfer
    pub async fn cross_domain_transfer(
        &mut self,
        source_domain: &str,
        target_domain: &str,
    ) -> Result<f32> {
        if !self.config.cross_domain_config.enable_domain_adaptation {
            return Ok(0.0);
        }

        // Find cross-domain mappings
        let mut transfer_pairs = Vec::new();
        for (source_concept, target_concept) in &self.cross_domain_mappings {
            if source_concept.contains(source_domain) && target_concept.contains(target_domain) {
                transfer_pairs.push((source_concept.clone(), target_concept.clone()));
            }
        }

        if transfer_pairs.is_empty() {
            return Ok(0.0);
        }

        // Compute domain adaptation loss
        let mut adaptation_loss = 0.0;
        for (source, target) in &transfer_pairs {
            if let (Some(source_emb), Some(target_emb)) = (
                self.unified_embeddings.get(source),
                self.unified_embeddings.get(target),
            ) {
                // L2 distance for domain alignment
                let diff = source_emb - target_emb;
                adaptation_loss += diff.dot(&diff).sqrt();
            }
        }

        adaptation_loss /= transfer_pairs.len() as f32;

        println!(
            "Cross-domain transfer loss ({} -> {}): {:.3}",
            source_domain, target_domain, adaptation_loss
        );

        Ok(adaptation_loss)
    }

    /// Multi-language alignment
    pub async fn multilingual_alignment(&self, concept: &str) -> Result<Vec<(String, f32)>> {
        if let Some(translations) = self.multilingual_mappings.get(concept) {
            let mut alignment_scores = Vec::new();

            if let Some(base_embedding) = self.unified_embeddings.get(concept) {
                for translation in translations {
                    if let Some(trans_embedding) = self.unified_embeddings.get(translation) {
                        let score = self
                            .alignment_network
                            .compute_alignment_score(base_embedding, trans_embedding);
                        alignment_scores.push((translation.clone(), score));
                    }
                }
            }

            // Sort by alignment score
            alignment_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            Ok(alignment_scores)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get multi-modal statistics
    pub fn get_multimodal_stats(&self) -> MultiModalStats {
        MultiModalStats {
            num_text_embeddings: self.text_embeddings.len(),
            num_kg_embeddings: self.kg_embeddings.len(),
            num_unified_embeddings: self.unified_embeddings.len(),
            num_alignments: self.text_kg_alignments.len(),
            num_entity_descriptions: self.entity_descriptions.len(),
            num_property_texts: self.property_texts.len(),
            num_multilingual_mappings: self.multilingual_mappings.len(),
            num_cross_domain_mappings: self.cross_domain_mappings.len(),
            text_dim: self.config.text_dim,
            kg_dim: self.config.kg_dim,
            unified_dim: self.config.unified_dim,
        }
    }
}

/// Multi-modal embedding statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalStats {
    pub num_text_embeddings: usize,
    pub num_kg_embeddings: usize,
    pub num_unified_embeddings: usize,
    pub num_alignments: usize,
    pub num_entity_descriptions: usize,
    pub num_property_texts: usize,
    pub num_multilingual_mappings: usize,
    pub num_cross_domain_mappings: usize,
    pub text_dim: usize,
    pub kg_dim: usize,
    pub unified_dim: usize,
}

#[async_trait]
impl EmbeddingModel for MultiModalEmbedding {
    fn config(&self) -> &ModelConfig {
        &self.config.base_config
    }

    fn model_id(&self) -> &Uuid {
        &self.model_id
    }

    fn model_type(&self) -> &'static str {
        "MultiModalEmbedding"
    }

    fn add_triple(&mut self, triple: crate::Triple) -> Result<()> {
        // Add triple components for multi-modal learning
        let subject = &triple.subject.iri;
        let predicate = &triple.predicate.iri;
        let object = &triple.object.iri;

        // Create alignment if description exists
        if let Some(description) = self.entity_descriptions.get(subject).cloned() {
            self.add_text_kg_alignment(&description, subject);
        }

        if let Some(description) = self.entity_descriptions.get(object).cloned() {
            self.add_text_kg_alignment(&description, object);
        }

        // Create property-text mapping if available
        if let Some(property_text) = self.property_texts.get(predicate).cloned() {
            self.add_text_kg_alignment(&property_text, predicate);
        }

        Ok(())
    }

    async fn train(&mut self, epochs: Option<usize>) -> Result<TrainingStats> {
        let epochs = epochs.unwrap_or(100);
        let start_time = std::time::Instant::now();
        let mut loss_history = Vec::new();

        // Training loop for multi-modal alignment
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Train on text-KG alignments
            let alignment_pairs: Vec<_> = self
                .text_kg_alignments
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (text, entity) in &alignment_pairs {
                // Generate embeddings and compute alignment loss
                if let Ok(unified) = self.generate_unified_embedding(text, entity).await {
                    // Simple reconstruction loss
                    let loss = unified.iter().map(|&x| x * x).sum::<f32>() / unified.len() as f32;
                    epoch_loss += loss;
                    num_batches += 1;
                }
            }

            // Add contrastive learning if we have negative samples
            if alignment_pairs.len() > 1 {
                let positive_pairs: Vec<_> = alignment_pairs
                    .iter()
                    .map(|(t, e)| (t.to_string(), e.to_string()))
                    .collect();

                // Create negative pairs by shuffling
                let mut negative_pairs = Vec::new();
                for i in 0..positive_pairs.len().min(10) {
                    let neg_entity = &positive_pairs[(i + 1) % positive_pairs.len()].1;
                    negative_pairs.push((positive_pairs[i].0.clone(), neg_entity.clone()));
                }

                if let Ok(contrastive_loss) =
                    self.contrastive_loss(&positive_pairs, &negative_pairs)
                {
                    epoch_loss += contrastive_loss;
                    num_batches += 1;
                }
            }

            if num_batches > 0 {
                epoch_loss /= num_batches as f32;
            }

            loss_history.push(epoch_loss as f64);

            if epoch % 10 == 0 {
                println!(
                    "Multi-modal training epoch {}: Loss = {:.6}",
                    epoch, epoch_loss
                );
            }

            // Early stopping
            if epoch_loss < 0.001 {
                break;
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();

        self.training_stats = TrainingStats {
            epochs_completed: epochs,
            final_loss: loss_history.last().copied().unwrap_or(0.0),
            training_time_seconds: training_time,
            convergence_achieved: loss_history.last().map_or(false, |&loss| loss < 0.001),
            loss_history,
        };

        self.is_trained = true;
        self.model_stats.is_trained = true;
        self.model_stats.last_training_time = Some(Utc::now());

        // Update statistics
        self.model_stats.num_entities = self.kg_embeddings.len();
        self.model_stats.num_relations = self.property_texts.len();
        self.model_stats.num_triples = self.text_kg_alignments.len();

        Ok(self.training_stats.clone())
    }

    fn get_entity_embedding(&self, entity: &str) -> Result<Vector> {
        if let Some(embedding) = self.unified_embeddings.get(entity) {
            Ok(Vector::from_array1(embedding))
        } else if let Some(embedding) = self.kg_embeddings.get(entity) {
            Ok(Vector::from_array1(embedding))
        } else {
            Err(anyhow!("Entity {} not found", entity))
        }
    }

    fn get_relation_embedding(&self, relation: &str) -> Result<Vector> {
        if let Some(embedding) = self.kg_embeddings.get(relation) {
            Ok(Vector::from_array1(embedding))
        } else {
            Err(anyhow!("Relation {} not found", relation))
        }
    }

    fn score_triple(&self, subject: &str, predicate: &str, object: &str) -> Result<f64> {
        let subject_emb = self.get_entity_embedding(subject)?;
        let predicate_emb = self.get_relation_embedding(predicate)?;
        let object_emb = self.get_entity_embedding(object)?;

        // Multi-modal scoring combines KG and text information
        let mut score = 0.0;
        for i in 0..subject_emb
            .dimensions
            .min(predicate_emb.dimensions)
            .min(object_emb.dimensions)
        {
            let diff = subject_emb.values[i] + predicate_emb.values[i] - object_emb.values[i];
            score += diff * diff;
        }

        // Convert to similarity score
        Ok(1.0 / (1.0 + score as f64))
    }

    fn predict_objects(
        &self,
        subject: &str,
        predicate: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let mut scores = Vec::new();

        for entity in self.kg_embeddings.keys() {
            if entity != subject {
                if let Ok(score) = self.score_triple(subject, predicate, entity) {
                    scores.push((entity.clone(), score));
                }
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_subjects(
        &self,
        predicate: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let mut scores = Vec::new();

        for entity in self.kg_embeddings.keys() {
            if entity != object {
                if let Ok(score) = self.score_triple(entity, predicate, object) {
                    scores.push((entity.clone(), score));
                }
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn predict_relations(
        &self,
        subject: &str,
        object: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>> {
        let mut scores = Vec::new();

        for relation in self.property_texts.keys() {
            if let Ok(score) = self.score_triple(subject, relation, object) {
                scores.push((relation.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn get_entities(&self) -> Vec<String> {
        self.kg_embeddings.keys().cloned().collect()
    }

    fn get_relations(&self) -> Vec<String> {
        self.property_texts.keys().cloned().collect()
    }

    fn get_stats(&self) -> ModelStats {
        self.model_stats.clone()
    }

    fn save(&self, _path: &str) -> Result<()> {
        // Implementation would serialize the multi-modal model
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<()> {
        // Implementation would deserialize the multi-modal model
        Ok(())
    }

    fn clear(&mut self) {
        self.text_embeddings.clear();
        self.kg_embeddings.clear();
        self.unified_embeddings.clear();
        self.text_kg_alignments.clear();
        self.entity_descriptions.clear();
        self.property_texts.clear();
        self.multilingual_mappings.clear();
        self.cross_domain_mappings.clear();
        self.is_trained = false;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    async fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for text in texts {
            if let Some(embedding) = self.text_embeddings.get(text) {
                embeddings.push(embedding.to_vec());
            } else {
                // Generate new text embedding
                let embedding = self.text_encoder.encode(text)?;
                embeddings.push(embedding.to_vec());
            }
        }

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_modal_config_default() {
        let config = CrossModalConfig::default();
        assert_eq!(config.text_dim, 768);
        assert_eq!(config.kg_dim, 128);
        assert_eq!(config.unified_dim, 512);
        assert_eq!(config.contrastive_config.temperature, 0.07);
    }

    #[test]
    fn test_multimodal_embedding_creation() {
        let config = CrossModalConfig::default();
        let model = MultiModalEmbedding::new(config);

        assert_eq!(model.model_type(), "MultiModalEmbedding");
        assert!(!model.is_trained());
        assert_eq!(model.text_embeddings.len(), 0);
        assert_eq!(model.kg_embeddings.len(), 0);
    }

    #[test]
    fn test_text_encoder() {
        let encoder = TextEncoder::new("BERT".to_string(), 768, 512);
        let embedding = encoder.encode("This is a test sentence").unwrap();
        assert_eq!(embedding.len(), 512);
    }

    #[test]
    fn test_kg_encoder() {
        let encoder = KGEncoder::new("ComplEx".to_string(), 128, 128, 512);
        let entity_emb = Array1::from_vec(vec![0.1; 128]);
        let encoded = encoder.encode_entity(&entity_emb).unwrap();
        assert_eq!(encoded.len(), 512);
    }

    #[test]
    fn test_alignment_network() {
        let network = AlignmentNetwork::new("CrossModalAttention".to_string(), 512, 512, 256, 512);
        let text_emb = Array1::from_vec(vec![0.1; 512]);
        let kg_emb = Array1::from_vec(vec![0.2; 512]);

        let (unified, score) = network.align(&text_emb, &kg_emb).unwrap();
        assert_eq!(unified.len(), 512);
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_multimodal_training() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        // Add some training data
        model.add_text_kg_alignment("This is a person", "http://example.org/Person");
        model.add_entity_description("http://example.org/Person", "A human being");
        model.add_property_text("http://example.org/knows", "knows relationship");

        let stats = model.train(Some(10)).await.unwrap();

        assert!(model.is_trained());
        assert_eq!(stats.epochs_completed, 10);
        assert!(stats.training_time_seconds > 0.0);
    }

    #[tokio::test]
    async fn test_unified_embedding_generation() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        let unified = model
            .generate_unified_embedding("A scientist working on AI", "http://example.org/Scientist")
            .await
            .unwrap();

        assert_eq!(unified.len(), 512); // unified_dim
        assert!(model
            .text_embeddings
            .contains_key("A scientist working on AI"));
        assert!(model
            .kg_embeddings
            .contains_key("http://example.org/Scientist"));
    }

    #[tokio::test]
    async fn test_zero_shot_prediction() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        // Add some KG embeddings
        let scientist_embedding = model.get_or_create_kg_embedding("scientist").unwrap();
        let doctor_embedding = model.get_or_create_kg_embedding("doctor").unwrap();
        let teacher_embedding = model.get_or_create_kg_embedding("teacher").unwrap();

        model
            .kg_embeddings
            .insert("scientist".to_string(), scientist_embedding);
        model
            .kg_embeddings
            .insert("doctor".to_string(), doctor_embedding);
        model
            .kg_embeddings
            .insert("teacher".to_string(), teacher_embedding);

        let candidates = vec![
            "scientist".to_string(),
            "doctor".to_string(),
            "teacher".to_string(),
        ];
        let predictions = model
            .zero_shot_prediction("A person who does research", &candidates)
            .await
            .unwrap();

        assert_eq!(predictions.len(), 3);
        assert!(predictions[0].1 >= predictions[1].1); // Scores should be sorted
    }

    #[test]
    fn test_contrastive_loss() {
        let config = CrossModalConfig::default();
        let mut model = MultiModalEmbedding::new(config);

        // Add some embeddings - text embeddings are 512-dim, kg embeddings are raw 128-dim
        model.text_embeddings.insert(
            "positive text".to_string(),
            Array1::from_vec(vec![1.0; 512]),
        );
        model.kg_embeddings.insert(
            "positive_entity".to_string(),
            Array1::from_vec(vec![1.0; 128]),
        );
        model.text_embeddings.insert(
            "negative text".to_string(),
            Array1::from_vec(vec![-1.0; 512]),
        );
        model.kg_embeddings.insert(
            "negative_entity".to_string(),
            Array1::from_vec(vec![-1.0; 128]),
        );

        let positive_pairs = vec![("positive text".to_string(), "positive_entity".to_string())];
        let negative_pairs = vec![("positive text".to_string(), "negative_entity".to_string())];

        let loss = model
            .contrastive_loss(&positive_pairs, &negative_pairs)
            .unwrap();
        assert!(loss >= 0.0);
    }
}
