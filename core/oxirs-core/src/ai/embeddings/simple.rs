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

/// SimplE embedding model (Kazemi & Poole, 2018).
///
/// SimplE is a bilinear model that uses two embedding vectors per entity
/// (`h_e` and `t_e`) and two embedding vectors per relation (`r` and `r^{-1}`).
/// The score is the average of two CP-style dot products:
///
/// `score(h, r, t) = ( <h_h, r, t_t> + <t_h, r^{-1}, h_t> ) / 2`
///
/// where `<a, b, c>` = `sum_i a_i * b_i * c_i`.
pub struct SimplE {
    config: EmbeddingConfig,
    /// Head role embeddings h_e
    entity_head: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Tail role embeddings t_e
    entity_tail: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Forward relation embeddings r
    relation_forward: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    /// Inverse relation embeddings r^{-1}
    relation_inverse: Arc<RwLock<HashMap<String, Array1<f32>>>>,
    entity_vocab: HashMap<String, usize>,
    relation_vocab: HashMap<String, usize>,
    trained: bool,
}

impl SimplE {
    /// Create a new SimplE model.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            entity_head: Arc::new(RwLock::new(HashMap::new())),
            entity_tail: Arc::new(RwLock::new(HashMap::new())),
            relation_forward: Arc::new(RwLock::new(HashMap::new())),
            relation_inverse: Arc::new(RwLock::new(HashMap::new())),
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

        let d = self.config.embedding_dim;
        let bound = (6.0 / d as f32).sqrt();

        let mut e_head = self.entity_head.write().await;
        let mut e_tail = self.entity_tail.write().await;
        let mut r_fwd = self.relation_forward.write().await;
        let mut r_inv = self.relation_inverse.write().await;

        for entity in &entities {
            let emb_h = Array1::from_shape_simple_fn(d, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            let emb_t = Array1::from_shape_simple_fn(d, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            e_head.insert(entity.clone(), emb_h);
            e_tail.insert(entity.clone(), emb_t);
        }

        for relation in &relations {
            let fwd = Array1::from_shape_simple_fn(d, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            let inv = Array1::from_shape_simple_fn(d, || {
                let mut rng = Random::default();
                rng.random::<f32>() * 2.0 * bound - bound
            });
            r_fwd.insert(relation.clone(), fwd);
            r_inv.insert(relation.clone(), inv);
        }

        Ok(())
    }

    /// Compute `<a, b, c>` = element-wise product sum.
    fn triple_dot(a: &Array1<f32>, b: &Array1<f32>, c: &Array1<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((ai, bi), ci)| ai * bi * ci)
            .sum()
    }

    async fn compute_score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let e_head = self.entity_head.read().await;
        let e_tail = self.entity_tail.read().await;
        let r_fwd = self.relation_forward.read().await;
        let r_inv = self.relation_inverse.read().await;

        let h_head = e_head
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let h_tail = e_tail
            .get(head)
            .ok_or_else(|| anyhow!("Entity not found: {}", head))?;
        let t_head = e_head
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        let t_tail = e_tail
            .get(tail)
            .ok_or_else(|| anyhow!("Entity not found: {}", tail))?;
        let r = r_fwd
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;
        let r_i = r_inv
            .get(relation)
            .ok_or_else(|| anyhow!("Relation not found: {}", relation))?;

        let forward = Self::triple_dot(h_head, r, t_tail);
        let inverse = Self::triple_dot(t_head, r_i, h_tail);

        // Average the two CP scores; apply logistic to get (0, 1)
        let logit = (forward + inverse) / 2.0;
        Ok(1.0 / (1.0 + (-logit).exp()))
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphEmbedding for SimplE {
    async fn generate_embeddings(&self, triples: &[Triple]) -> Result<Vec<Vec<f32>>> {
        let e_head = self.entity_head.read().await;
        let e_tail = self.entity_tail.read().await;
        let mut embeddings = Vec::new();

        for triple in triples {
            let h_h = e_head
                .get(&triple.subject().to_string())
                .ok_or_else(|| anyhow!("Entity not found: {}", triple.subject()))?;
            let t_t = e_tail
                .get(&triple.object().to_string())
                .ok_or_else(|| anyhow!("Entity not found: {}", triple.object()))?;
            let combined: Vec<f32> = h_h
                .iter()
                .zip(t_t.iter())
                .map(|(a, b)| (a + b) / 2.0)
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

        predictions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        Ok(predictions)
    }

    async fn get_entity_embedding(&self, entity: &str) -> Result<Vec<f32>> {
        let e_head = self.entity_head.read().await;
        let e_tail = self.entity_tail.read().await;
        let h = e_head
            .get(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        let t = e_tail
            .get(entity)
            .ok_or_else(|| anyhow!("Entity not found: {}", entity))?;
        // Concatenate head and tail embeddings as the full representation
        let mut result = h.to_vec();
        result.extend_from_slice(t.as_slice().unwrap_or(&[]));
        Ok(result)
    }

    async fn get_relation_embedding(&self, relation: &str) -> Result<Vec<f32>> {
        let r_fwd = self.relation_forward.read().await;
        r_fwd
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

        for _ in 0..config.max_epochs.min(50) {
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
            epochs: config.max_epochs.min(50),
            time_elapsed: start_time.elapsed(),
            kg_metrics: KnowledgeGraphMetrics::default(),
        })
    }

    async fn save(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let e_head = self.entity_head.read().await;
        let e_tail = self.entity_tail.read().await;
        let r_fwd = self.relation_forward.read().await;
        let r_inv = self.relation_inverse.read().await;
        let state = serde_json::json!({
            "config": self.config,
            "entity_head": e_head.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "entity_tail": e_tail.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "relation_forward": r_fwd.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
            "relation_inverse": r_inv.iter().map(|(k, v)| (k, v.to_vec())).collect::<HashMap<_,_>>(),
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

        let mut e_head = self.entity_head.write().await;
        let mut e_tail = self.entity_tail.write().await;
        let mut r_fwd = self.relation_forward.write().await;
        let mut r_inv = self.relation_inverse.write().await;
        e_head.clear();
        e_tail.clear();
        r_fwd.clear();
        r_inv.clear();

        let load_embs = |key: &str,
                         map: &mut HashMap<String, Array1<f32>>,
                         data: &serde_json::Value|
         -> Result<()> {
            let emb_data: HashMap<String, Vec<f32>> = serde_json::from_value(data[key].clone())?;
            for (k, v) in emb_data {
                map.insert(k, Array1::from_vec(v));
            }
            Ok(())
        };

        load_embs("entity_head", &mut e_head, &state)?;
        load_embs("entity_tail", &mut e_tail, &state)?;
        load_embs("relation_forward", &mut r_fwd, &state)?;
        load_embs("relation_inverse", &mut r_inv, &state)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_score_range() {
        let config = EmbeddingConfig {
            embedding_dim: 4,
            ..Default::default()
        };
        let model = SimplE::new(config);

        {
            let mut e_head = model.entity_head.write().await;
            e_head.insert(
                "h".to_string(),
                Array1::from_vec(vec![1.0, -1.0, 0.5, -0.5]),
            );
            e_head.insert(
                "t".to_string(),
                Array1::from_vec(vec![-0.5, 0.5, 1.0, -1.0]),
            );
        }
        {
            let mut e_tail = model.entity_tail.write().await;
            e_tail.insert("h".to_string(), Array1::from_vec(vec![0.2, 0.3, -0.4, 0.1]));
            e_tail.insert(
                "t".to_string(),
                Array1::from_vec(vec![0.7, -0.2, 0.3, -0.8]),
            );
        }
        {
            let mut r_fwd = model.relation_forward.write().await;
            r_fwd.insert("r".to_string(), Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]));
        }
        {
            let mut r_inv = model.relation_inverse.write().await;
            r_inv.insert("r".to_string(), Array1::from_vec(vec![0.4, 0.3, 0.2, 0.1]));
        }

        let score = model
            .score_triple("h", "r", "t")
            .await
            .expect("score_triple should succeed");
        assert!(
            (0.0..=1.0).contains(&score),
            "SimplE score should be in (0, 1): got {}",
            score
        );
    }
}
