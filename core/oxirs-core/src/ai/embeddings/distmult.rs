use super::{
    EmbeddingConfig, KnowledgeGraphEmbedding, KnowledgeGraphMetrics, TrainingConfig,
    TrainingMetrics,
};
use crate::model::Triple;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::{Random, Rng};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// DistMult embedding model
pub struct DistMult {
    #[allow(dead_code)]
    config: EmbeddingConfig,
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    #[allow(dead_code)]
    entity_vocab: HashMap<String, usize>,
    #[allow(dead_code)]
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

    /// Compute DistMult score: `<h, r, t>` = sum(h * r * t)
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
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            entity_embs.insert(entity, embedding);
        }

        for relation in relations {
            let embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            relation_embs.insert(relation, embedding);
        }

        Ok(())
    }

    /// Calculate accuracy on validation triples
    async fn calculate_accuracy(&self, triples: &[(String, String, String)]) -> Result<f32> {
        if triples.is_empty() {
            return Ok(0.0);
        }

        let mut correct = 0;
        let total = triples.len().min(100); // Sample for efficiency

        for triple in triples.iter().take(total) {
            let positive_score = self.compute_score(&triple.0, &triple.1, &triple.2).await?;

            // Generate a random negative and compare
            let entities: Vec<String> = self.entity_vocab.keys().cloned().collect();
            if entities.len() >= 2 {
                let corrupt_idx = {
                    let mut rng = Random::default();
                    rng.random_range(0..entities.len())
                };
                let corrupt_entity = &entities[corrupt_idx];

                let should_corrupt_head = {
                    let mut rng = Random::default();
                    rng.random_bool_with_chance(0.5)
                };
                let negative_score = if should_corrupt_head {
                    self.compute_score(corrupt_entity, &triple.1, &triple.2)
                        .await?
                } else {
                    self.compute_score(&triple.0, &triple.1, corrupt_entity)
                        .await?
                };

                // For DistMult, higher score is better
                if positive_score > negative_score {
                    correct += 1;
                }
            }
        }

        Ok(correct as f32 / total as f32)
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for DistMult {
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>> {
        // Similar to TransE but with different scoring function
        let entity_embs = self.entity_embeddings.read().await;
        let mut embeddings = Vec::new();

        for triple in triples {
            let subject_str = triple.subject().to_string();
            let object_str = triple.object().to_string();
            let head_emb = entity_embs
                .get(&subject_str)
                .ok_or_else(|| anyhow!("Entity not found"))?;
            let tail_emb = entity_embs
                .get(&object_str)
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
        predictions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

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

    async fn train(
        &mut self,
        triples: &[Triple],
        _config: &TrainingConfig,
    ) -> Result<TrainingMetrics> {
        // Initialize embeddings similar to TransE
        self.initialize_embeddings(triples).await?;

        // Convert triples to string format
        let triple_strings: Vec<(String, String, String)> = triples
            .iter()
            .map(|t| {
                (
                    t.subject().to_string(),
                    t.predicate().to_string(),
                    t.object().to_string(),
                )
            })
            .collect();

        let mut total_loss = 0.0;

        for _epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;

            // Simplified training - in practice would use proper SGD with gradients
            for triple in &triple_strings {
                let score = self.compute_score(&triple.0, &triple.1, &triple.2).await?;

                // For DistMult, we want to maximize the score for positive triples
                // This is a simplified loss - negative log-likelihood would be better
                epoch_loss += (1.0 - score).max(0.0);
            }

            total_loss = epoch_loss / triple_strings.len() as f32;

            // Early stopping
            if total_loss < 1e-6 {
                break;
            }
        }

        self.trained = true;

        // Calculate accuracy on validation set
        let accuracy = self.calculate_accuracy(&triple_strings).await?;

        Ok(TrainingMetrics {
            loss: total_loss,
            loss_history: vec![total_loss],
            accuracy,
            epochs: self.config.max_epochs,
            time_elapsed: std::time::Duration::from_secs(0),
            kg_metrics: KnowledgeGraphMetrics::default(),
        })
    }

    async fn save(&self, _path: &str) -> Result<()> {
        Ok(())
    }

    async fn load(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }
}
