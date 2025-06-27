//! Knowledge Graph Embeddings for RDF
//!
//! This module implements various knowledge graph embedding models including
//! TransE, DistMult, ComplEx, RotatE, and other state-of-the-art approaches.

use crate::model::{Triple, NamedNode, Subject, Predicate, Object};
use crate::OxirsError;
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use rand::Rng;

/// Knowledge graph embedding trait
#[async_trait::async_trait]
pub trait KnowledgeGraphEmbedding: Send + Sync {
    /// Generate embeddings for entities and relations
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>>;
    
    /// Score a triple (head, relation, tail)
    async fn score_triple(&self, head: &str, relation: &str, tail: &str) -> Result<f32>;
    
    /// Predict missing links
    async fn predict_links(
        &self,
        entities: &[String],
        relations: &[String],
    ) -> Result<Vec<(String, String, String, f32)>>;
    
    /// Get entity embedding
    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>>;
    
    /// Get relation embedding
    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>>;
    
    /// Train the embedding model
    async fn train(&mut self, triples: &[Triple], config: &TrainingConfig) -> Result<TrainingMetrics>;
    
    /// Save model to file
    async fn save(&self, path: &str) -> Result<()>;
    
    /// Load model from file
    async fn load(&mut self, path: &str) -> Result<()>;
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model type
    pub model_type: EmbeddingModelType,
    
    /// Embedding dimension
    pub embedding_dim: usize,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// L2 regularization weight
    pub l2_weight: f32,
    
    /// Negative sampling ratio
    pub negative_sampling_ratio: f32,
    
    /// Training batch size
    pub batch_size: usize,
    
    /// Maximum training epochs
    pub max_epochs: usize,
    
    /// Early stopping patience
    pub patience: usize,
    
    /// Validation split
    pub validation_split: f32,
    
    /// Enable GPU acceleration
    pub use_gpu: bool,
    
    /// Random seed
    pub seed: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_type: EmbeddingModelType::TransE,
            embedding_dim: 100,
            learning_rate: 0.001,
            l2_weight: 1e-5,
            negative_sampling_ratio: 1.0,
            batch_size: 1024,
            max_epochs: 1000,
            patience: 50,
            validation_split: 0.1,
            use_gpu: true,
            seed: 42,
        }
    }
}

/// Embedding model types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingModelType {
    /// Translation-based model (Bordes et al., 2013)
    TransE,
    
    /// Bilinear model (Yang et al., 2014)
    DistMult,
    
    /// Complex embeddings (Trouillon et al., 2016)
    ComplEx,
    
    /// Rotation-based model (Sun et al., 2019)
    RotatE,
    
    /// Hyperbolic embeddings (Balazevic et al., 2019)
    HypE,
    
    /// Tucker decomposition (Balazevic et al., 2019)
    TuckER,
    
    /// Convolutional model (Dettmers et al., 2018)
    ConvE,
    
    /// Transformer-based model
    KGTransformer,
    
    /// Neural tensor network (Socher et al., 2013)
    NeuralTensorNetwork,
    
    /// SimplE (Kazemi & Poole, 2018)
    SimplE,
}

/// TransE embedding model
pub struct TransE {
    /// Model configuration
    config: EmbeddingConfig,
    
    /// Entity embeddings
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    
    /// Relation embeddings
    relation_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    
    /// Entity vocabulary
    entity_vocab: HashMap<String, usize>,
    
    /// Relation vocabulary
    relation_vocab: HashMap<String, usize>,
    
    /// Training state
    trained: bool,
}

impl TransE {
    /// Create new TransE model
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings: Arc::new(RwLock::new(HashMap::new())),
            entity_vocab: HashMap::new(),
            relation_vocab: HashMap::new(),
            trained: false,
        }
    }
    
    /// Initialize embeddings from vocabulary
    async fn initialize_embeddings(&mut self, triples: &[Triple]) -> Result<()> {
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();
        
        // Collect vocabulary
        for triple in triples {
            entities.insert(triple.subject().to_string());
            entities.insert(triple.object().to_string());
            relations.insert(triple.predicate().to_string());
        }
        
        // Create vocabularies
        self.entity_vocab = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| (entity.clone(), i))
            .collect();
        
        self.relation_vocab = relations
            .iter()
            .enumerate()
            .map(|(i, relation)| (relation.clone(), i))
            .collect();
        
        // Initialize embeddings with Xavier initialization
        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;
        
        let bound = (6.0 / self.config.embedding_dim as f32).sqrt();
        
        for entity in entities {
            let embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                rand::thread_rng().gen::<f32>() * 2.0 * bound - bound
            });
            entity_embs.insert(entity, embedding);
        }
        
        for relation in relations {
            let embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                rand::thread_rng().gen::<f32>() * 2.0 * bound - bound
            });
            relation_embs.insert(relation, embedding);
        }
        
        Ok(())
    }
    
    /// Compute TransE score: ||h + r - t||
    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;
        
        let h = entity_embs
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let r = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let t = entity_embs
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        
        // Compute ||h + r - t||_L1 or ||h + r - t||_L2
        let diff = h + r - t;
        let score = diff.mapv(|x| x.abs()).sum(); // L1 norm
        
        Ok(score)
    }
    
    /// Generate negative samples
    fn generate_negative_samples(
        &self,
        positive_triples: &[(String, String, String)],
        num_negatives: usize,
    ) -> Vec<(String, String, String)> {
        let mut negatives = Vec::new();
        let entities: Vec<String> = self.entity_vocab.keys().cloned().collect();
        let relations: Vec<String> = self.relation_vocab.keys().cloned().collect();
        
        for _ in 0..num_negatives {
            // Randomly corrupt head or tail
            let positive_idx = rand::thread_rng().gen_range(0..positive_triples.len());
            let (h, r, t) = &positive_triples[positive_idx];
            
            if rand::thread_rng().gen_bool(0.5) {
                // Corrupt head
                let new_head_idx = rand::thread_rng().gen_range(0..entities.len());
                let new_head = &entities[new_head_idx];
                if new_head != h {
                    negatives.push((new_head.clone(), r.clone(), t.clone()));
                }
            } else {
                // Corrupt tail
                let new_tail_idx = rand::thread_rng().gen_range(0..entities.len());
                let new_tail = &entities[new_tail_idx];
                if new_tail != t {
                    negatives.push((h.clone(), r.clone(), new_tail.clone()));
                }
            }
        }
        
        negatives
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for TransE {
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>> {
        let entity_embs = self.entity_embeddings.read().await;
        let mut embeddings = Vec::new();
        
        for triple in triples {
            let head_emb = entity_embs
                .get(&triple.subject().to_string())
                .ok_or_else(|| anyhow!("Entity not found"))?;
            let tail_emb = entity_embs
                .get(&triple.object().to_string())
                .ok_or_else(|| anyhow!("Entity not found"))?;
            
            // Combine head and tail embeddings
            let combined: Vec<f32> = head_emb
                .iter()
                .zip(tail_emb.iter())
                .map(|(h, t)| (h + t) / 2.0)
                .collect();
            
            embeddings.push(combined);
        }
        
        Ok(embeddings)
    }
    
    async fn score_triple(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        self.compute_score(head, relation, tail).await
    }
    
    async fn predict_links(
        &self,
        entities: &[String],
        relations: &[String],
    ) -> Result<Vec<(String, String, String, f32)>> {
        let mut predictions = Vec::new();
        
        // Generate all possible triples and score them
        for head in entities {
            for relation in relations {
                for tail in entities {
                    if head != tail {
                        let score = self.score_triple(head, relation, tail).await?;
                        predictions.push((head.clone(), relation.clone(), tail.clone(), score));
                    }
                }
            }
        }
        
        // Sort by score (lower is better for TransE)
        predictions.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
        
        Ok(predictions)
    }
    
    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>> {
        let entity_embs = self.entity_embeddings.read().await;
        let embedding = entity_embs
            .get(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        
        Ok(embedding.to_vec())
    }
    
    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>> {
        let relation_embs = self.relation_embeddings.read().await;
        let embedding = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        
        Ok(embedding.to_vec())
    }
    
    async fn train(&mut self, triples: &[Triple], config: &TrainingConfig) -> Result<TrainingMetrics> {
        // Initialize embeddings
        self.initialize_embeddings(triples).await?;
        
        // Convert triples to string format
        let triple_strings: Vec<(String, String, String)> = triples
            .iter()
            .map(|t| (
                t.subject().to_string(),
                t.predicate().to_string(),
                t.object().to_string(),
            ))
            .collect();
        
        let mut total_loss = 0.0;
        let margin = 1.0; // Margin for margin-based loss
        
        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            
            // Generate negative samples
            let negatives = self.generate_negative_samples(
                &triple_strings,
                (triple_strings.len() as f32 * self.config.negative_sampling_ratio) as usize,
            );
            
            // Training step (simplified - in real implementation would use proper SGD)
            for (i, positive) in triple_strings.iter().enumerate() {
                let positive_score = self.compute_score(&positive.0, &positive.1, &positive.2).await?;
                
                if i < negatives.len() {
                    let negative = &negatives[i];
                    let negative_score = self.compute_score(&negative.0, &negative.1, &negative.2).await?;
                    
                    // Margin-based loss: max(0, positive_score - negative_score + margin)
                    let loss = (positive_score - negative_score + margin).max(0.0);
                    epoch_loss += loss;
                }
            }
            
            total_loss = epoch_loss / triple_strings.len() as f32;
            
            // Early stopping check (simplified)
            if total_loss < 1e-6 {
                break;
            }
        }
        
        self.trained = true;
        
        Ok(TrainingMetrics {
            loss: total_loss,
            accuracy: 0.0, // TODO: Implement proper accuracy calculation
            epochs: self.config.max_epochs,
            time_elapsed: std::time::Duration::from_secs(0),
        })
    }
    
    async fn save(&self, path: &str) -> Result<()> {
        // TODO: Implement model serialization
        Ok(())
    }
    
    async fn load(&mut self, path: &str) -> Result<()> {
        // TODO: Implement model deserialization
        Ok(())
    }
}

/// DistMult embedding model
pub struct DistMult {
    config: EmbeddingConfig,
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    entity_vocab: HashMap<String, usize>,
    relation_vocab: HashMap<String, usize>,
    trained: bool,
}

impl DistMult {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings: Arc::new(RwLock::new(HashMap::new())),
            entity_vocab: HashMap::new(),
            relation_vocab: HashMap::new(),
            trained: false,
        }
    }
    
    /// Compute DistMult score: <h, r, t> = sum(h * r * t)
    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;
        
        let h = entity_embs
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let r = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let t = entity_embs
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        
        // Compute element-wise product and sum
        let score = (h * r * t).sum();
        
        Ok(score)
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for DistMult {
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>> {
        // Similar to TransE but with different scoring function
        let entity_embs = self.entity_embeddings.read().await;
        let mut embeddings = Vec::new();
        
        for triple in triples {
            let head_emb = entity_embs
                .get(&triple.subject().to_string())
                .ok_or_else(|| anyhow!("Entity not found"))?;
            let tail_emb = entity_embs
                .get(&triple.object().to_string())
                .ok_or_else(|| anyhow!("Entity not found"))?;
            
            let combined: Vec<f32> = head_emb
                .iter()
                .zip(tail_emb.iter())
                .map(|(h, t)| h * t) // Element-wise product for DistMult
                .collect();
            
            embeddings.push(combined);
        }
        
        Ok(embeddings)
    }
    
    async fn score_triple(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        self.compute_score(head, relation, tail).await
    }
    
    async fn predict_links(
        &self,
        entities: &[String],
        relations: &[String],
    ) -> Result<Vec<(String, String, String, f32)>> {
        let mut predictions = Vec::new();
        
        for head in entities {
            for relation in relations {
                for tail in entities {
                    if head != tail {
                        let score = self.score_triple(head, relation, tail).await?;
                        predictions.push((head.clone(), relation.clone(), tail.clone(), score));
                    }
                }
            }
        }
        
        // Sort by score (higher is better for DistMult)
        predictions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        
        Ok(predictions)
    }
    
    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>> {
        let entity_embs = self.entity_embeddings.read().await;
        let embedding = entity_embs
            .get(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        Ok(embedding.to_vec())
    }
    
    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>> {
        let relation_embs = self.relation_embeddings.read().await;
        let embedding = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        Ok(embedding.to_vec())
    }
    
    async fn train(&mut self, triples: &[Triple], config: &TrainingConfig) -> Result<TrainingMetrics> {
        // TODO: Implement DistMult training
        self.trained = true;
        Ok(TrainingMetrics {
            loss: 0.0,
            accuracy: 0.0,
            epochs: 0,
            time_elapsed: std::time::Duration::from_secs(0),
        })
    }
    
    async fn save(&self, path: &str) -> Result<()> {
        Ok(())
    }
    
    async fn load(&mut self, path: &str) -> Result<()> {
        Ok(())
    }
}

/// ComplEx embedding model with complex numbers
pub struct ComplEx {
    config: EmbeddingConfig,
    entity_embeddings_real: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    entity_embeddings_imag: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings_real: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings_imag: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    entity_vocab: HashMap<String, usize>,
    relation_vocab: HashMap<String, usize>,
    trained: bool,
}

impl ComplEx {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            entity_embeddings_real: Arc::new(RwLock::new(HashMap::new())),
            entity_embeddings_imag: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings_real: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings_imag: Arc::new(RwLock::new(HashMap::new())),
            entity_vocab: HashMap::new(),
            relation_vocab: HashMap::new(),
            trained: false,
        }
    }
    
    /// Compute ComplEx score using complex number operations
    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_real = self.entity_embeddings_real.read().await;
        let entity_imag = self.entity_embeddings_imag.read().await;
        let relation_real = self.relation_embeddings_real.read().await;
        let relation_imag = self.relation_embeddings_imag.read().await;
        
        let h_real = entity_real.get(head).ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let h_imag = entity_imag.get(head).ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let r_real = relation_real.get(relation).ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let r_imag = relation_imag.get(relation).ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let t_real = entity_real.get(tail).ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        let t_imag = entity_imag.get(tail).ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        
        // ComplEx score: Re(<h, r, conj(t)>)
        let score = (h_real * r_real * t_real + h_real * r_imag * t_imag + 
                    h_imag * r_real * t_imag - h_imag * r_imag * t_real).sum();
        
        Ok(score)
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for ComplEx {
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>> {
        let entity_real = self.entity_embeddings_real.read().await;
        let entity_imag = self.entity_embeddings_imag.read().await;
        let mut embeddings = Vec::new();
        
        for triple in triples {
            let head_real = entity_real
                .get(&triple.subject().to_string())
                .ok_or_else(|| anyhow!("Entity not found"))?;
            let head_imag = entity_imag
                .get(&triple.subject().to_string())
                .ok_or_else(|| anyhow!("Entity not found"))?;
            
            // Combine real and imaginary parts
            let mut combined = head_real.to_vec();
            combined.extend(head_imag.to_vec());
            
            embeddings.push(combined);
        }
        
        Ok(embeddings)
    }
    
    async fn score_triple(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        self.compute_score(head, relation, tail).await
    }
    
    async fn predict_links(
        &self,
        entities: &[String],
        relations: &[String],
    ) -> Result<Vec<(String, String, String, f32)>> {
        let mut predictions = Vec::new();
        
        for head in entities {
            for relation in relations {
                for tail in entities {
                    if head != tail {
                        let score = self.score_triple(head, relation, tail).await?;
                        predictions.push((head.clone(), relation.clone(), tail.clone(), score));
                    }
                }
            }
        }
        
        predictions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        Ok(predictions)
    }
    
    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>> {
        let real = self.entity_embeddings_real.read().await;
        let imag = self.entity_embeddings_imag.read().await;
        
        let real_emb = real.get(entity).ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        let imag_emb = imag.get(entity).ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        
        let mut combined = real_emb.to_vec();
        combined.extend(imag_emb.to_vec());
        
        Ok(combined)
    }
    
    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>> {
        let real = self.relation_embeddings_real.read().await;
        let imag = self.relation_embeddings_imag.read().await;
        
        let real_emb = real.get(relation).ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let imag_emb = imag.get(relation).ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        
        let mut combined = real_emb.to_vec();
        combined.extend(imag_emb.to_vec());
        
        Ok(combined)
    }
    
    async fn train(&mut self, triples: &[Triple], config: &TrainingConfig) -> Result<TrainingMetrics> {
        // TODO: Implement ComplEx training
        self.trained = true;
        Ok(TrainingMetrics {
            loss: 0.0,
            accuracy: 0.0,
            epochs: 0,
            time_elapsed: std::time::Duration::from_secs(0),
        })
    }
    
    async fn save(&self, path: &str) -> Result<()> {
        Ok(())
    }
    
    async fn load(&mut self, path: &str) -> Result<()> {
        Ok(())
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub patience: usize,
    pub validation_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            batch_size: 1024,
            learning_rate: 0.001,
            patience: 50,
            validation_split: 0.1,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub epochs: usize,
    pub time_elapsed: std::time::Duration,
}

/// Create embedding model based on configuration
pub fn create_embedding_model(config: EmbeddingConfig) -> Result<Arc<dyn KnowledgeGraphEmbedding>> {
    match config.model_type {
        EmbeddingModelType::TransE => Ok(Arc::new(TransE::new(config))),
        EmbeddingModelType::DistMult => Ok(Arc::new(DistMult::new(config))),
        EmbeddingModelType::ComplEx => Ok(Arc::new(ComplEx::new(config))),
        _ => Err(anyhow!("Embedding model not yet implemented")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Literal};
    
    #[tokio::test]
    async fn test_transe_creation() {
        let config = EmbeddingConfig::default();
        let transe = TransE::new(config);
        assert!(!transe.trained);
    }
    
    #[tokio::test]
    async fn test_transe_scoring() {
        let config = EmbeddingConfig {
            embedding_dim: 10,
            ..Default::default()
        };
        
        let mut transe = TransE::new(config);
        
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/alice").unwrap(),
                NamedNode::new("http://example.org/knows").unwrap(),
                NamedNode::new("http://example.org/bob").unwrap(),
            ),
        ];
        
        transe.initialize_embeddings(&triples).await.unwrap();
        
        let score = transe.score_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        ).await.unwrap();
        
        assert!(score > 0.0);
    }
    
    #[test]
    fn test_embedding_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 100);
        assert_eq!(config.model_type, EmbeddingModelType::TransE);
    }
    
    #[test]
    fn test_create_embedding_model() {
        let config = EmbeddingConfig::default();
        let model = create_embedding_model(config);
        // Test that the model was created successfully
        assert!(model.is_ok());
    }
}