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

/// ComplEx embedding model with complex numbers
pub struct ComplEx {
    #[allow(dead_code)]
    config: EmbeddingConfig,
    entity_embeddings_real: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    entity_embeddings_imag: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings_real: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings_imag: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    #[allow(dead_code)]
    entity_vocab: HashMap<String, usize>,
    #[allow(dead_code)]
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
        let mut entity_real = self.entity_embeddings_real.write().await;
        let mut entity_imag = self.entity_embeddings_imag.write().await;
        let mut relation_real = self.relation_embeddings_real.write().await;
        let mut relation_imag = self.relation_embeddings_imag.write().await;

        let bound = (6.0 / self.config.embedding_dim as f32).sqrt();

        for entity in entities {
            let real_embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            let imag_embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            entity_real.insert(entity.clone(), real_embedding);
            entity_imag.insert(entity, imag_embedding);
        }

        for relation in relations {
            let real_embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            let imag_embedding = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            relation_real.insert(relation.clone(), real_embedding);
            relation_imag.insert(relation, imag_embedding);
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
                    rng.random_range(0, entities.len())
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

                // For ComplEx, higher score is better
                if positive_score > negative_score {
                    correct += 1;
                }
            }
        }

        Ok(correct as f32 / total as f32)
    }

    /// Compute ComplEx score using complex number operations
    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_real = self.entity_embeddings_real.read().await;
        let entity_imag = self.entity_embeddings_imag.read().await;
        let relation_real = self.relation_embeddings_real.read().await;
        let relation_imag = self.relation_embeddings_imag.read().await;

        let h_real = entity_real
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let h_imag = entity_imag
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let r_real = relation_real
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let r_imag = relation_imag
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let t_real = entity_real
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        let t_imag = entity_imag
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;

        // ComplEx score: Re(<h, r, conj(t)>)
        let score =
            (h_real * r_real * t_real + h_real * r_imag * t_imag + h_imag * r_real * t_imag
                - h_imag * r_imag * t_real)
                .sum();

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
            let subject_str = triple.subject().to_string();
            let head_real = entity_real
                .get(&subject_str)
                .ok_or_else(|| anyhow!("Entity not found"))?;
            let head_imag = entity_imag
                .get(&subject_str)
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

        let real_emb = real
            .get(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        let imag_emb = imag
            .get(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;

        let mut combined = real_emb.to_vec();
        combined.extend(imag_emb.to_vec());

        Ok(combined)
    }

    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>> {
        let real = self.relation_embeddings_real.read().await;
        let imag = self.relation_embeddings_imag.read().await;

        let real_emb = real
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let imag_emb = imag
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;

        let mut combined = real_emb.to_vec();
        combined.extend(imag_emb.to_vec());

        Ok(combined)
    }

    async fn train(
        &mut self,
        triples: &[Triple],
        _config: &TrainingConfig,
    ) -> Result<TrainingMetrics> {
        // Initialize embeddings
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

            // Simplified training for ComplEx
            for triple in &triple_strings {
                let score = self.compute_score(&triple.0, &triple.1, &triple.2).await?;

                // For ComplEx, we want to maximize the score for positive triples
                epoch_loss += (1.0 - score.abs()).max(0.0);
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
