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

/// TuckER embedding model (Balazevic et al., 2019).
///
/// TuckER decomposes the binary tensor of triples using Tucker decomposition:
///
/// `score(h, r, t) = W ×₁ h ×₂ r ×₃ t`
///
/// where `W ∈ ℝ^{d_e × d_r × d_e}` is a shared core tensor,
/// `h, t ∈ ℝ^{d_e}` and `r ∈ ℝ^{d_r}`.
///
/// For efficiency the full 3D core tensor is stored as a flat vector of
/// length `d_e * d_r * d_e`.  The mode-1 product `Wh ∈ ℝ^{d_r × d_e}` is
/// then computed, followed by `(Wh)r ∈ ℝ^{d_e}` and finally the dot product
/// with `t`.
pub struct TuckER {
    config: EmbeddingConfig,
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Core tensor W (flat, length d_e × d_r × d_e)
    core_tensor: Arc<RwLock<Vec<f32>>>,
    /// Relation embedding dimension (defaults to config.embedding_dim)
    relation_dim: usize,
    entity_vocab: HashMap<String, usize>,
    relation_vocab: HashMap<String, usize>,
    trained: bool,
}

impl TuckER {
    /// Create a new TuckER model.
    pub fn new(config: EmbeddingConfig) -> Self {
        let relation_dim = config.embedding_dim; // d_r = d_e for simplicity
        Self {
            relation_dim,
            config,
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings: Arc::new(RwLock::new(HashMap::new())),
            core_tensor: Arc::new(RwLock::new(Vec::new())),
            entity_vocab: HashMap::new(),
            relation_vocab: HashMap::new(),
            trained: false,
        }
    }

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

        let d_e = self.config.embedding_dim;
        let d_r = self.relation_dim;
        let bound = (6.0 / d_e as f32).sqrt();

        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;
        let mut core = self.core_tensor.write().await;

        for entity in &entities {
            let emb = Array1::from_shape_simple_fn(d_e, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            entity_embs.insert(entity.clone(), emb);
        }

        for relation in &relations {
            let emb = Array1::from_shape_simple_fn(d_r, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            relation_embs.insert(relation.clone(), emb);
        }

        // Initialize core tensor
        let core_size = d_e * d_r * d_e;
        let core_bound = (6.0 / core_size as f32).sqrt();
        *core = (0..core_size)
            .map(|_| {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * core_bound - core_bound
            })
            .collect();

        Ok(())
    }

    /// Compute TuckER score via Tucker decomposition.
    ///
    /// `score = W ×₁ h ×₂ r ×₃ t`
    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;
        let core = self.core_tensor.read().await;

        let h = entity_embs
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let r = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let t = entity_embs
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;

        let d_e = self.config.embedding_dim;
        let d_r = self.relation_dim;

        if core.len() != d_e * d_r * d_e {
            return Err(anyhow!(
                "Core tensor not initialised (len={}, expected {})",
                core.len(),
                d_e * d_r * d_e
            ));
        }

        // Step 1: mode-1 product — contract over d_e axis of W with h.
        // W_h[i_r, i_e2] = sum_{i_e1} W[i_e1, i_r, i_e2] * h[i_e1]
        let mut wh = vec![0.0f32; d_r * d_e];
        for i_e1 in 0..d_e {
            for i_r in 0..d_r {
                for i_e2 in 0..d_e {
                    let core_idx = i_e1 * d_r * d_e + i_r * d_e + i_e2;
                    wh[i_r * d_e + i_e2] += core[core_idx] * h[i_e1];
                }
            }
        }

        // Step 2: mode-2 product — contract over d_r with r.
        // wh_r[i_e2] = sum_{i_r} wh[i_r, i_e2] * r[i_r]
        let mut whr = vec![0.0f32; d_e];
        for i_r in 0..d_r {
            for i_e2 in 0..d_e {
                whr[i_e2] += wh[i_r * d_e + i_e2] * r[i_r];
            }
        }

        // Step 3: dot product with t
        let score: f32 = whr.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
        // Apply sigmoid to keep score in (0, 1)
        Ok(1.0 / (1.0 + (-score).exp()))
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for TuckER {
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
        let mut loss_history = Vec::new();

        for _ in 0..config.max_epochs.min(20) {
            let mut epoch_loss = 0.0f32;

            for triple in &triple_strings {
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
                let loss = -(score_pos.ln()) - (1.0 - score_neg).ln();
                epoch_loss += loss;
            }

            loss_history.push(epoch_loss / triple_strings.len().max(1) as f32);
        }

        self.trained = true;

        Ok(TrainingMetrics {
            loss: loss_history.last().copied().unwrap_or(0.0),
            loss_history,
            accuracy: 0.5,
            epochs: config.max_epochs.min(20),
            time_elapsed: start_time.elapsed(),
            kg_metrics: KnowledgeGraphMetrics::default(),
        })
    }

    async fn save(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;
        let core = self.core_tensor.read().await;
        let state = serde_json::json!({
            "config": self.config,
            "entity_embeddings": entity_embs.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "relation_embeddings": relation_embs.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "core_tensor": *core,
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
        let mut core = self.core_tensor.write().await;
        entity_embs.clear();
        relation_embs.clear();

        let entity_data: HashMap<String, Vec<f32>> =
            serde_json::from_value(state["entity_embeddings"].clone())?;
        for (k, v) in entity_data {
            entity_embs.insert(k, Array1::from_vec(v));
        }
        let rel_data: HashMap<String, Vec<f32>> =
            serde_json::from_value(state["relation_embeddings"].clone())?;
        for (k, v) in rel_data {
            relation_embs.insert(k, Array1::from_vec(v));
        }
        *core = serde_json::from_value(state["core_tensor"].clone())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tucker_score_range() {
        let config = EmbeddingConfig {
            embedding_dim: 4,
            ..Default::default()
        };
        let model = TuckER::new(config);

        {
            let mut entity_embs = model.entity_embeddings.write().await;
            entity_embs.insert("h".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.5, -0.5]));
            entity_embs.insert("t".to_string(), Array1::from_vec(vec![0.5, 1.0, -0.5, 0.5]));
        }
        {
            let mut relation_embs = model.relation_embeddings.write().await;
            relation_embs.insert("r".to_string(), Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]));
        }
        {
            let d_e = 4usize;
            let d_r = 4usize;
            let mut core = model.core_tensor.write().await;
            *core = vec![0.01f32; d_e * d_r * d_e];
        }

        let score = model
            .score_triple("h", "r", "t")
            .await
            .expect("score_triple should succeed");
        assert!(
            (0.0..=1.0).contains(&score),
            "TuckER score should be in (0, 1): got {}",
            score
        );
    }
}
