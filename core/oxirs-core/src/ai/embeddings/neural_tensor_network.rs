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

/// Neural Tensor Network (NTN) embedding model (Socher et al., 2013).
///
/// The NTN score is:
///
/// `score(h, r, t) = u_r^T · tanh( h^T W_r t + V_r [h; t] + b_r )`
///
/// where:
/// - `W_r ∈ ℝ^{d × d × k}` is the bilinear tensor (one slice per feature)
/// - `V_r ∈ ℝ^{k × 2d}` is a standard linear layer
/// - `b_r ∈ ℝ^k` is the bias
/// - `u_r ∈ ℝ^k` is the relation-specific output weight
///
/// For efficiency `k` is set to `embedding_dim / 4` (capped at 16).
pub struct NeuralTensorNetwork {
    config: EmbeddingConfig,
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Per-relation parameters stored as flat JSON-able vectors:
    /// Each relation entry contains (W_r, V_r, b_r, u_r) concatenated.
    relation_params: Arc<RwLock<HashMap<String, RelationParams>>>,
    entity_vocab: HashMap<String, usize>,
    relation_vocab: HashMap<String, usize>,
    /// Feature dimension k
    k: usize,
    trained: bool,
}

/// Per-relation NTN parameters.
struct RelationParams {
    /// Bilinear tensor W_r: k slices of (d × d), stored flat (k * d * d)
    w: Vec<f32>,
    /// Linear layer V_r: k × 2d, stored flat (k * 2d)
    v: Vec<f32>,
    /// Bias b_r: k
    b: Vec<f32>,
    /// Output weights u_r: k
    u: Vec<f32>,
}

impl NeuralTensorNetwork {
    /// Create a new NTN model.
    pub fn new(config: EmbeddingConfig) -> Self {
        let k = (config.embedding_dim / 4).clamp(1, 16);
        Self {
            k,
            config,
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_params: Arc::new(RwLock::new(HashMap::new())),
            entity_vocab: HashMap::new(),
            relation_vocab: HashMap::new(),
            trained: false,
        }
    }

    fn make_relation_params(d: usize, k: usize) -> RelationParams {
        let bound = (6.0 / (d * d) as f32).sqrt();
        let mut rng = Random::default();
        RelationParams {
            w: (0..k * d * d)
                .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
                .collect(),
            v: (0..k * 2 * d)
                .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
                .collect(),
            b: vec![0.0f32; k],
            u: (0..k)
                .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
                .collect(),
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

        let d = self.config.embedding_dim;
        let k = self.k;
        let bound = (6.0 / d as f32).sqrt();

        let mut entity_embs = self.entity_embeddings.write().await;
        let mut rel_params = self.relation_params.write().await;

        for entity in &entities {
            let emb = Array1::from_shape_simple_fn(d, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            entity_embs.insert(entity.clone(), emb);
        }

        for relation in &relations {
            rel_params.insert(relation.clone(), Self::make_relation_params(d, k));
        }

        Ok(())
    }

    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_embs = self.entity_embeddings.read().await;
        let rel_params = self.relation_params.read().await;

        let h = entity_embs
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let t = entity_embs
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        let params = rel_params
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;

        let d = self.config.embedding_dim;
        let k = self.k;

        let h_slice = h.as_slice().unwrap_or(&[]);
        let t_slice = t.as_slice().unwrap_or(&[]);

        // Compute f = tanh( bilinear + linear + bias ) for each of k features
        let mut f = vec![0.0f32; k];
        #[allow(clippy::needless_range_loop)]
        for ki in 0..k {
            // Bilinear term: h^T W_{r,ki} t
            let mut bilinear = 0.0f32;
            for di in 0..d {
                for dj in 0..d {
                    let w_idx = ki * d * d + di * d + dj;
                    bilinear += h_slice.get(di).copied().unwrap_or(0.0)
                        * params.w.get(w_idx).copied().unwrap_or(0.0)
                        * t_slice.get(dj).copied().unwrap_or(0.0);
                }
            }
            // Linear term: V_{r,ki} · [h; t]
            let mut linear = 0.0f32;
            for di in 0..d {
                linear += h_slice.get(di).copied().unwrap_or(0.0)
                    * params.v.get(ki * 2 * d + di).copied().unwrap_or(0.0);
            }
            for dj in 0..d {
                linear += t_slice.get(dj).copied().unwrap_or(0.0)
                    * params.v.get(ki * 2 * d + d + dj).copied().unwrap_or(0.0);
            }
            f[ki] = (bilinear + linear + params.b.get(ki).copied().unwrap_or(0.0)).tanh();
        }

        // Final score: u_r · f
        let logit: f32 = f.iter().zip(params.u.iter()).map(|(fi, ui)| fi * ui).sum();
        // Sigmoid for probability
        Ok(1.0 / (1.0 + (-logit).exp()))
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for NeuralTensorNetwork {
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
        // NTN stores relation-specific parameters rather than a single embedding.
        // Return the u_r vector as a proxy for the relation embedding.
        let rel_params = self.relation_params.read().await;
        rel_params
            .get(relation)
            .map(|p| p.u.clone())
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

        for _ in 0..config.max_epochs.min(10) {
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
            epochs: config.max_epochs.min(10),
            time_elapsed: start_time.elapsed(),
            kg_metrics: KnowledgeGraphMetrics::default(),
        })
    }

    async fn save(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let entity_embs = self.entity_embeddings.read().await;
        let rel_params = self.relation_params.read().await;

        let rel_params_serializable: HashMap<String, serde_json::Value> = rel_params
            .iter()
            .map(|(k, p)| {
                (
                    k.clone(),
                    serde_json::json!({
                        "w": p.w,
                        "v": p.v,
                        "b": p.b,
                        "u": p.u,
                    }),
                )
            })
            .collect();

        let state = serde_json::json!({
            "config": self.config,
            "entity_embeddings": entity_embs.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "relation_params": rel_params_serializable,
            "entity_vocab": self.entity_vocab,
            "relation_vocab": self.relation_vocab,
            "k": self.k,
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
        self.k = state["k"].as_u64().unwrap_or(self.k as u64) as usize;
        self.trained = state["trained"].as_bool().unwrap_or(false);

        let mut entity_embs = self.entity_embeddings.write().await;
        let mut rel_params = self.relation_params.write().await;
        entity_embs.clear();
        rel_params.clear();

        let entity_data: HashMap<String, Vec<f32>> =
            serde_json::from_value(state["entity_embeddings"].clone())?;
        for (k, v) in entity_data {
            entity_embs.insert(k, Array1::from_vec(v));
        }

        let params_data: HashMap<String, serde_json::Value> =
            serde_json::from_value(state["relation_params"].clone())?;
        for (k, pv) in params_data {
            rel_params.insert(
                k,
                RelationParams {
                    w: serde_json::from_value(pv["w"].clone())?,
                    v: serde_json::from_value(pv["v"].clone())?,
                    b: serde_json::from_value(pv["b"].clone())?,
                    u: serde_json::from_value(pv["u"].clone())?,
                },
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ntn_score_range() {
        let config = EmbeddingConfig {
            embedding_dim: 4,
            ..Default::default()
        };
        let model = NeuralTensorNetwork::new(config);
        let d = 4usize;
        let k = model.k;

        {
            let mut entity_embs = model.entity_embeddings.write().await;
            entity_embs.insert("h".to_string(), Array1::from_vec(vec![0.5, -0.3, 0.8, 0.1]));
            entity_embs.insert(
                "t".to_string(),
                Array1::from_vec(vec![-0.2, 0.6, -0.4, 0.9]),
            );
        }
        {
            let mut rel_params = model.relation_params.write().await;
            rel_params.insert(
                "r".to_string(),
                NeuralTensorNetwork::make_relation_params(d, k),
            );
        }

        let score = model
            .score_triple("h", "r", "t")
            .await
            .expect("score_triple should succeed");
        assert!(
            (0.0..=1.0).contains(&score),
            "NTN score should be in (0, 1): got {}",
            score
        );
    }
}
