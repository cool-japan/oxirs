//! Main multi-modal embedding model implementation

use super::config::CrossModalConfig;
use super::encoders::{AlignmentNetwork, KGEncoder, TextEncoder};
use super::learning::FewShotLearning;
use super::adaptation::RealTimeFinetuning;
use crate::{EmbeddingModel, ModelStats, TrainingStats, Vector, NamedNode};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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

impl MultiModalEmbedding {
    /// Create new multi-modal embedding model
    pub fn new(config: CrossModalConfig) -> Self {
        let model_id = Uuid::from_u128(rand::random());
        let now = Utc::now();

        let text_encoder = TextEncoder::new("BERT".to_string(), config.text_dim, config.text_dim);

        let kg_encoder = KGEncoder::new(
            "ComplEx".to_string(),
            config.kg_dim,
            config.kg_dim,
            config.kg_dim,
        );

        let alignment_network = AlignmentNetwork::new(
            "CrossModalAttention".to_string(),
            config.text_dim,
            config.kg_dim,
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
            training_stats: TrainingStats {
                epochs_completed: 0,
                final_loss: 0.0,
                training_time_seconds: 0.0,
                convergence_achieved: false,
                loss_history: Vec::new(),
            },
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
    pub fn get_or_create_kg_embedding(&self, entity: &str) -> Result<Array1<f32>> {
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

    /// Add few-shot learning capability
    pub fn with_few_shot_learning(mut self, _few_shot_config: FewShotLearning) -> Self {
        // Store few-shot learning configuration (would need to add field to struct)
        // For now, we'll just return self as the integration would require struct changes
        self
    }

    /// Perform few-shot learning task
    pub async fn few_shot_learn(
        &self,
        support_examples: &[(String, String, String)],
        query_examples: &[(String, String)],
    ) -> Result<Vec<(String, f32)>> {
        let mut few_shot_learner = FewShotLearning::default();
        few_shot_learner
            .few_shot_adapt(support_examples, query_examples, self)
            .await
    }

    /// Add real-time fine-tuning capability
    pub fn with_real_time_finetuning(mut self, _rt_config: RealTimeFinetuning) -> Self {
        // Store real-time fine-tuning configuration
        // For now, we'll just return self as the integration would require struct changes
        self
    }

    /// Update model with new example in real-time
    pub async fn real_time_update(&mut self, text: &str, entity: &str, label: &str) -> Result<f32> {
        let mut rt_finetuning = RealTimeFinetuning::default();
        rt_finetuning.add_example(text.to_string(), entity.to_string(), label.to_string());
        rt_finetuning.update_model(self).await
    }
}

#[async_trait]
impl EmbeddingModel for MultiModalEmbedding {
    fn config(&self) -> &crate::ModelConfig {
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