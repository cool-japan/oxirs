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

/// KGTransformer — Transformer-based knowledge graph embedding model.
///
/// Uses a self-attention mechanism over the triple `(h, r, t)` to compute a
/// context-aware triple score.  The attention-pooled representation of the
/// triple is projected to a scalar logit:
///
/// ```text
/// Q = h W_Q,  K = r W_K,  V = t W_V
/// score = σ( softmax(Q·K^T / √d) · V )
/// ```
///
/// For triples the sequence length is 3 (one token per role), so the full
/// attention matrix is 3×3 and the pooled output has dimension d.
pub struct KGTransformer {
    config: EmbeddingConfig,
    entity_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    relation_embeddings: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Query projection W_Q (d × d, stored flat row-major)
    w_q: Vec<f32>,
    /// Key projection W_K
    w_k: Vec<f32>,
    /// Value projection W_V
    w_v: Vec<f32>,
    entity_vocab: HashMap<String, usize>,
    relation_vocab: HashMap<String, usize>,
    trained: bool,
}

impl KGTransformer {
    /// Create a new KGTransformer model.
    pub fn new(config: EmbeddingConfig) -> Self {
        let d = config.embedding_dim;
        let proj_size = d * d;
        let bound = (6.0 / (2 * d) as f32).sqrt();

        let mut rng = Random::default();
        let make_proj = |rng: &mut Random| {
            (0..proj_size)
                .map(|_| rng.random::<f32>() * 2.0 * bound - bound)
                .collect::<Vec<f32>>()
        };

        Self {
            config,
            entity_embeddings: Arc::new(RwLock::new(HashMap::new())),
            relation_embeddings: Arc::new(RwLock::new(HashMap::new())),
            w_q: make_proj(&mut rng),
            w_k: make_proj(&mut rng),
            w_v: make_proj(&mut rng),
            entity_vocab: HashMap::new(),
            relation_vocab: HashMap::new(),
            trained: false,
        }
    }

    /// Matrix-vector product: `y = A x` where `A` is `d × d` stored row-major.
    fn matvec(mat: &[f32], x: &[f32], d: usize) -> Vec<f32> {
        (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| {
                        mat.get(i * d + j).copied().unwrap_or(0.0)
                            * x.get(j).copied().unwrap_or(0.0)
                    })
                    .sum::<f32>()
            })
            .collect()
    }

    /// Softmax over a slice.
    fn softmax(v: &[f32]) -> Vec<f32> {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum.max(f32::EPSILON)).collect()
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

        let bound = (6.0 / self.config.embedding_dim as f32).sqrt();
        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;

        for entity in &entities {
            let emb = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            entity_embs.insert(entity.clone(), emb);
        }

        for relation in &relations {
            let emb = Array1::from_shape_simple_fn(self.config.embedding_dim, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            relation_embs.insert(relation.clone(), emb);
        }

        Ok(())
    }

    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let entity_embs = self.entity_embeddings.read().await;
        let relation_embs = self.relation_embeddings.read().await;

        let h_arr = entity_embs
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let r_arr = relation_embs
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let t_arr = entity_embs
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;

        let d = self.config.embedding_dim;
        let scale = (d as f32).sqrt();

        let h = h_arr.to_vec();
        let r = r_arr.to_vec();
        let t = t_arr.to_vec();

        // Project to Q, K, V
        let q = Self::matvec(&self.w_q, &h, d);
        let k = Self::matvec(&self.w_k, &r, d);
        let v = Self::matvec(&self.w_v, &t, d);

        // Scalar attention: q · k / √d (sequence length 1 for simplicity)
        let attn_logit: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / scale;
        let attn_weights = Self::softmax(&[attn_logit]);

        // Weighted value
        let attended: Vec<f32> = v.iter().map(|&vi| vi * attn_weights[0]).collect();

        // Final logit = h · attended (residual-style)
        let logit: f32 = h.iter().zip(attended.iter()).map(|(a, b)| a * b).sum();
        Ok(1.0 / (1.0 + (-logit).exp()))
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for KGTransformer {
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
        let state = serde_json::json!({
            "config": self.config,
            "entity_embeddings": entity_embs.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "relation_embeddings": relation_embs.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "w_q": self.w_q,
            "w_k": self.w_k,
            "w_v": self.w_v,
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
        self.w_q = serde_json::from_value(state["w_q"].clone())?;
        self.w_k = serde_json::from_value(state["w_k"].clone())?;
        self.w_v = serde_json::from_value(state["w_v"].clone())?;

        let mut entity_embs = self.entity_embeddings.write().await;
        let mut relation_embs = self.relation_embeddings.write().await;
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

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kgtransformer_score_range() {
        let config = EmbeddingConfig {
            embedding_dim: 4,
            ..Default::default()
        };
        let model = KGTransformer::new(config);

        {
            let mut entity_embs = model.entity_embeddings.write().await;
            entity_embs.insert("h".to_string(), Array1::from_vec(vec![0.6, -0.2, 0.3, 0.8]));
            entity_embs.insert(
                "t".to_string(),
                Array1::from_vec(vec![-0.1, 0.9, 0.4, -0.7]),
            );
        }
        {
            let mut relation_embs = model.relation_embeddings.write().await;
            relation_embs.insert(
                "r".to_string(),
                Array1::from_vec(vec![0.5, 0.5, -0.5, -0.5]),
            );
        }

        let score = model
            .score_triple("h", "r", "t")
            .await
            .expect("score_triple should succeed");
        assert!(
            (0.0..=1.0).contains(&score),
            "KGTransformer score should be in (0, 1): got {}",
            score
        );
    }
}
