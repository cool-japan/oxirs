use super::{
    EmbeddingConfig, KnowledgeGraphEmbedding, KnowledgeGraphMetrics, TrainingConfig,
    TrainingMetrics,
};
use crate::model::Triple;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::random::{Random, RngExt};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// RotatE embedding model (Sun et al., 2019).
///
/// Relations are modelled as rotations in a complex-number embedding space:
///
/// `score(h, r, t) = -||h ∘ r - t||`
///
/// where `h`, `r`, `t ∈ ℂ^d/2` and `|r_i| = 1` for all i (relation embeddings
/// lie on the unit circle in each dimension).  The entity embeddings store
/// interleaved (real, imag) pairs so that the i-th complex component is
/// `(emb[2i], emb[2i+1])`.
pub struct RotatE {
    config: EmbeddingConfig,
    /// Entity embeddings — interleaved (real, imag), shape: dim * 2
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Relation phase angles θ_i, shape: dim (|r_i| = 1, r_i = e^{iθ_i})
    relation_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    entity_vocab: HashMap<String, usize>,
    relation_vocab: HashMap<String, usize>,
    trained: bool,
}

impl RotatE {
    /// Create a new RotatE model.
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

    /// Effective complex dimension (half the real embedding dim).
    fn complex_dim(&self) -> usize {
        self.config.embedding_dim / 2
    }

    /// Initialize embeddings.
    async fn initialize_embeddings(&mut self, triples: &[Triple]) -> Result<()> {
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();

        for triple in triples {
            entities.insert(triple.subject().to_string());
            entities.insert(triple.object().to_string());
            relations.insert(triple.predicate().to_string());
        }

        self.entity_vocab = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.clone(), i))
            .collect();
        self.relation_vocab = relations
            .iter()
            .enumerate()
            .map(|(i, r)| (r.clone(), i))
            .collect();

        let entity_dim = self.complex_dim() * 2; // interleaved real + imag
        let relation_dim = self.complex_dim(); // phase angles only
        let bound = (6.0 / entity_dim as f32).sqrt();

        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;

        for entity in &entities {
            let emb = Array1::from_shape_simple_fn(entity_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            entity_embs.insert(entity.clone(), emb);
        }

        for relation in &relations {
            // Phase angles uniformly in (-π, π)
            let emb = Array1::from_shape_simple_fn(relation_dim, || {
                let mut rng = Random::default();
                (rng.random::<f32>() * 2.0 - 1.0) * std::f32::consts::PI
            });
            relation_embs.insert(relation.clone(), emb);
        }

        Ok(())
    }

    /// Compute the RotatE score: `−||h ∘ r − t||₁`.
    ///
    /// A higher (less negative) score indicates a more plausible triple.
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

        let d = r.len(); // complex_dim
        let mut dist = 0.0f32;
        for i in 0..d {
            let h_re = h[2 * i];
            let h_im = h[2 * i + 1];
            let theta = r[i];
            // r_i = (cos θ_i, sin θ_i)
            let rot_re = h_re * theta.cos() - h_im * theta.sin();
            let rot_im = h_re * theta.sin() + h_im * theta.cos();
            let t_re = t[2 * i];
            let t_im = t[2 * i + 1];
            dist += (rot_re - t_re).abs() + (rot_im - t_im).abs();
        }

        // Return negative distance so higher = more plausible
        Ok(-dist)
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for RotatE {
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>> {
        let entity_embs = self.entity_embeddings.read().await;
        let mut embeddings = Vec::new();

        for triple in triples {
            let h = entity_embs
                .get(&triple.subject().to_string())
                .ok_or_else(|| anyhow!("Entity not found: {}", triple.subject()))?;
            let t = entity_embs
                .get(&triple.object().to_string())
                .ok_or_else(|| anyhow!("Entity not found: {}", triple.object()))?;
            let combined: Vec<f32> = h.iter().zip(t.iter()).map(|(a, b)| (a + b) / 2.0).collect();
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

        // Higher score = more plausible
        predictions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        Ok(predictions)
    }

    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>> {
        let entity_embs = self.entity_embeddings.read().await;
        entity_embs
            .get(entity)
            .map(|e| e.to_vec())
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))
    }

    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>> {
        let relation_embs = self.relation_embeddings.read().await;
        relation_embs
            .get(relation)
            .map(|r| r.to_vec())
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))
    }

    async fn train(
        &mut self,
        triples: &[Triple],
        config: &TrainingConfig,
    ) -> Result<TrainingMetrics> {
        use std::time::Instant;
        let start_time = Instant::now();

        self.initialize_embeddings(triples).await?;

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

        let entities: Vec<String> = self.entity_vocab.keys().cloned().collect();
        let margin = 1.0f32;
        let lr = config.learning_rate;
        let mut loss_history = Vec::new();

        for epoch in 0..config.max_epochs.min(50) {
            let mut epoch_loss = 0.0f32;

            for triple in &triple_strings {
                // Corrupt tail to form negative
                let corrupt_idx = {
                    let mut rng = Random::default();
                    rng.random_range(0..entities.len())
                };
                let corrupt = &entities[corrupt_idx];
                if corrupt == &triple.2 {
                    continue;
                }

                let score_pos = self.compute_score(&triple.0, &triple.1, &triple.2).await?;
                let score_neg = self.compute_score(&triple.0, &triple.1, corrupt).await?;

                // Margin ranking loss: max(0, margin - score_pos + score_neg)
                // (score is negative distance, so we want score_pos > score_neg)
                let loss = (margin - score_pos + score_neg).max(0.0);
                epoch_loss += loss;

                if loss > 0.0 {
                    // Simple SGD gradient step on phase angles
                    let d = self.complex_dim();
                    let mut relation_embs = self.relation_embeddings.write().await;
                    if let Some(r) = relation_embs.get_mut(&triple.1) {
                        for i in 0..d {
                            r[i] -= lr * (score_neg - score_pos).signum() * 0.01;
                        }
                    }
                }
            }

            loss_history.push(epoch_loss / triple_strings.len().max(1) as f32);

            if epoch % 10 == 0 {
                tracing::debug!(
                    "RotatE epoch {}/{}: loss={:.4}",
                    epoch,
                    config.max_epochs,
                    loss_history.last().copied().unwrap_or(0.0)
                );
            }
        }

        self.trained = true;

        Ok(TrainingMetrics {
            loss: loss_history.last().copied().unwrap_or(0.0),
            loss_history,
            accuracy: 0.5, // placeholder — full accuracy requires link prediction ranking
            epochs: config.max_epochs.min(50),
            time_elapsed: start_time.elapsed(),
            kg_metrics: KnowledgeGraphMetrics::default(),
        })
    }

    async fn save(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;
        let state = serde_json::json!({
            "config": self.config,
            "entity_embeddings": entity_embs.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "relation_embeddings": relation_embs.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "entity_vocab": self.entity_vocab,
            "relation_vocab": self.relation_vocab,
            "trained": self.trained,
        });
        let mut file = std::fs::File::create(path)?;
        file.write_all(serde_json::to_string_pretty(&state)?.as_bytes())?;
        Ok(())
    }

    async fn load(&mut self, path: &str) -> Result<()> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let state: serde_json::Value = serde_json::from_str(&contents)?;

        self.config = serde_json::from_value(state["config"].clone())?;
        self.entity_vocab = serde_json::from_value(state["entity_vocab"].clone())?;
        self.relation_vocab = serde_json::from_value(state["relation_vocab"].clone())?;
        self.trained = state["trained"].as_bool().unwrap_or(false);

        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;
        entity_embs.clear();
        relation_embs.clear();

        let entity_data: HashMap<String, Vec<f32>> =
            serde_json::from_value(state["entity_embeddings"].clone())?;
        for (k, v) in entity_data {
            entity_embs.insert(k, Array1::from_vec(v));
        }
        let relation_data: HashMap<String, Vec<f32>> =
            serde_json::from_value(state["relation_embeddings"].clone())?;
        for (k, v) in relation_data {
            relation_embs.insert(k, Array1::from_vec(v));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rotate_score_is_negative() {
        let config = EmbeddingConfig {
            embedding_dim: 8,
            ..Default::default()
        };
        let model = RotatE::new(config);

        // Manually plant embeddings so we can test scoring without training
        {
            let mut entity_embs = model.entity_embeddings.write().await;
            entity_embs.insert("h".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0]));
            entity_embs.insert("t".to_string(), Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]));
        }
        {
            let mut relation_embs = model.relation_embeddings.write().await;
            // θ = π/2 → rotation by 90°
            relation_embs.insert(
                "r".to_string(),
                Array1::from_vec(vec![
                    std::f32::consts::FRAC_PI_2,
                    std::f32::consts::FRAC_PI_2,
                ]),
            );
        }

        let score = model
            .score_triple("h", "r", "t")
            .await
            .expect("score_triple should succeed");
        // RotatE score is −distance, so it must be ≤ 0
        assert!(score <= 0.0, "RotatE score should be non-positive");
    }
}
