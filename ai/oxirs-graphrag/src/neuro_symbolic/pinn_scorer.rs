//! Physics-Informed Neural Network (PINN) entity scorer.
//!
//! Blends GNN cosine similarity (neural signal) with physics plausibility
//! (symbolic signal) via a simple damping parameter:
//!
//! ```text
//! combined = (1 − λ) · neural + λ · physics
//! ```
//!
//! where `λ ∈ [0, 1]` (`damping`): 0 = pure neural, 1 = pure physics.
//!
//! The neural score is the normalised cosine similarity `(cos+1)/2 ∈ [0,1]`.

use std::collections::HashMap;
use std::sync::Arc;

use scirs2_core::ndarray_ext::{Array1, Array2};

use super::physics_context::{PhysicsContext, PlausibilityScore};
use crate::gnn_encoder::{EntityEmbeddings, GnnError, GraphSageEncoder, KgGraph};

// ─── Public data structures ───────────────────────────────────────────────────

/// A knowledge-graph entity with an embedding index and physical properties.
#[derive(Debug, Clone)]
pub struct KgEntity {
    /// Stable identifier (e.g. URI or local name).
    pub id: String,
    /// Row index into the [`EntityEmbeddings`] matrix.
    pub embedding_idx: usize,
    /// Named scalar properties used by the physics context.
    pub properties: HashMap<String, f64>,
}

/// A scored entity after PINN evaluation.
#[derive(Debug, Clone)]
pub struct ScoredEntity {
    /// Identifier of the entity.
    pub entity_id: String,
    /// Normalised cosine similarity `(cos+1)/2 ∈ [0,1]`.
    pub neural_score: f64,
    /// Physics plausibility ∈ \[0,1\].
    pub physics_score: f64,
    /// Combined score `(1−λ)·neural + λ·physics`.
    pub combined_score: f64,
    /// Full plausibility breakdown from the physics context.
    pub plausibility: PlausibilityScore,
}

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by [`PinnEntityScorer`].
#[derive(Debug, thiserror::Error)]
pub enum PinnScorerError {
    #[error("GNN encoder error: {0}")]
    EncoderError(String),

    #[error("embedding index {idx} out of bounds (embeddings has {max} rows)")]
    EmbeddingIndexOutOfBounds { idx: usize, max: usize },

    #[error("entity list is empty")]
    EmptyEntityList,
}

impl From<GnnError> for PinnScorerError {
    fn from(e: GnnError) -> Self {
        Self::EncoderError(e.to_string())
    }
}

// ─── Scorer ───────────────────────────────────────────────────────────────────

/// PINN-driven entity scorer combining GNN embeddings with physics plausibility.
pub struct PinnEntityScorer {
    encoder: Arc<GraphSageEncoder>,
    physics_context: PhysicsContext,
    /// Damping λ: 0.0 = pure neural, 1.0 = pure physics.
    damping: f64,
}

impl PinnEntityScorer {
    /// Create a new scorer.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid `damping` values outside `[0, 1]` are silently clamped.
    pub fn new(
        encoder: Arc<GraphSageEncoder>,
        physics_context: PhysicsContext,
        damping: f64,
    ) -> Self {
        Self {
            encoder,
            physics_context,
            damping: damping.clamp(0.0, 1.0),
        }
    }

    /// Returns the configured damping coefficient λ.
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// Score a list of entities against pre-computed `embeddings`.
    ///
    /// The `query_embedding` is a 1-D vector in the same space as the GNN
    /// output embeddings; its cosine similarity to each entity embedding is
    /// used as the neural score.
    pub fn score_entities(
        &self,
        embeddings: &EntityEmbeddings,
        entities: &[KgEntity],
        query_embedding: &Array1<f64>,
    ) -> Result<Vec<ScoredEntity>, PinnScorerError> {
        if entities.is_empty() {
            return Err(PinnScorerError::EmptyEntityList);
        }

        let n_rows = embeddings.embeddings.nrows();
        let n_cols = embeddings.embeddings.ncols();

        let mut results = Vec::with_capacity(entities.len());

        for entity in entities {
            if entity.embedding_idx >= n_rows {
                return Err(PinnScorerError::EmbeddingIndexOutOfBounds {
                    idx: entity.embedding_idx,
                    max: n_rows,
                });
            }

            // Extract entity embedding row as Array1.
            let entity_row = extract_row(&embeddings.embeddings, entity.embedding_idx, n_cols);

            let neural_score = cosine_similarity_normalized(&entity_row, query_embedding);
            let plausibility = self.physics_context.plausibility_score(&entity.properties);
            let physics_score = plausibility.score;

            let combined_score = (1.0 - self.damping) * neural_score + self.damping * physics_score;

            results.push(ScoredEntity {
                entity_id: entity.id.clone(),
                neural_score,
                physics_score,
                combined_score,
                plausibility,
            });
        }

        Ok(results)
    }

    /// Encode the `kg` graph first, then score entities against `query_embedding`.
    ///
    /// Note: The [`HybridLlmHead`](crate::hybrid::HybridLlmHead) also encodes
    /// the same graph internally when `answer()` is called; the double-encode
    /// is accepted to keep the APIs independent.
    pub fn encode_and_score(
        &self,
        kg: &KgGraph,
        entities: &[KgEntity],
        query_embedding: &Array1<f64>,
    ) -> Result<Vec<ScoredEntity>, PinnScorerError> {
        let embeddings = self.encoder.encode(kg)?;
        self.score_entities(&embeddings, entities, query_embedding)
    }

    /// Like [`Self::encode_and_score`], but returns entities sorted by
    /// `combined_score` descending (highest first).
    pub fn rank(
        &self,
        kg: &KgGraph,
        entities: &[KgEntity],
        query_embedding: &Array1<f64>,
    ) -> Result<Vec<ScoredEntity>, PinnScorerError> {
        let mut scored = self.encode_and_score(kg, entities, query_embedding)?;
        scored.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(scored)
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Extract row `idx` from `mat` as a 1-D `Array1<f64>`.
fn extract_row(mat: &Array2<f64>, idx: usize, ncols: usize) -> Array1<f64> {
    let row_slice: Vec<f64> = (0..ncols).map(|j| mat[[idx, j]]).collect();
    Array1::from_vec(row_slice)
}

/// Normalised cosine similarity: `(dot(a,b) / (|a|·|b|) + 1) / 2 ∈ [0,1]`.
///
/// Returns 0.5 when either vector is (near-)zero (undefined cosine → neutral).
fn cosine_similarity_normalized(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if na < 1e-12 || nb < 1e-12 {
        return 0.5;
    }

    let cos = (dot / (na * nb)).clamp(-1.0, 1.0);
    (cos + 1.0) / 2.0
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array2;

    use crate::gnn_encoder::{GraphSageConfig, GraphSageEncoder};
    use crate::neuro_symbolic::physics_context::{PhysicsContext, PhysicsDomain};

    fn toy_encoder() -> Arc<GraphSageEncoder> {
        let config = GraphSageConfig {
            input_dim: 4,
            hidden_dim: 4,
            output_dim: 4,
            num_layers: 2,
            dropout: 0.0,
            k_neighbors: 2,
            learning_rate: 0.0,
        };
        Arc::new(GraphSageEncoder::new_with_seed(&config, 1).expect("encoder"))
    }

    fn toy_kg() -> KgGraph {
        KgGraph {
            num_nodes: 4,
            edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            node_features: Array2::zeros((4, 4)),
        }
    }

    fn thermal_ctx() -> PhysicsContext {
        PhysicsContext::new(PhysicsDomain::ThermalDiffusion {
            thermal_diffusivity: 1e-5,
        })
    }

    /// Build manual embeddings for unit-testing — bypasses GNN nondeterminism.
    fn manual_embeddings(rows: Vec<[f64; 4]>) -> EntityEmbeddings {
        let n = rows.len();
        let mut mat = Array2::zeros((n, 4));
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                mat[[i, j]] = v;
            }
        }
        EntityEmbeddings {
            embeddings: mat,
            node_ids: (0..n).map(|i| i.to_string()).collect(),
        }
    }

    #[test]
    fn test_cosine_normalized_same_vector() {
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let score = cosine_similarity_normalized(&v, &v);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "identical vectors → 1.0, got {score}"
        );
    }

    #[test]
    fn test_cosine_normalized_opposite_vectors() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0]);
        let score = cosine_similarity_normalized(&a, &b);
        assert!((score - 0.0).abs() < 1e-10, "opposite → 0.0, got {score}");
    }

    #[test]
    fn test_cosine_normalized_zero_vector() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0]);
        let score = cosine_similarity_normalized(&a, &b);
        assert!(
            (score - 0.5).abs() < 1e-10,
            "zero vector → 0.5, got {score}"
        );
    }

    #[test]
    fn test_cosine_normalized_range() {
        // Any pair must land in [0, 1].
        let pairs: &[([f64; 3], [f64; 3])] = &[
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]),
            ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0]),
        ];
        for (a, b) in pairs {
            let av = Array1::from_vec(a.to_vec());
            let bv = Array1::from_vec(b.to_vec());
            let s = cosine_similarity_normalized(&av, &bv);
            assert!(
                (0.0..=1.0).contains(&s),
                "score {s} out of [0,1] for {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn test_damping_zero_pure_neural() {
        let scorer = PinnEntityScorer::new(toy_encoder(), thermal_ctx(), 0.0);
        let embs = manual_embeddings(vec![[1.0, 0.0, 0.0, 0.0]]);
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let entities = vec![KgEntity {
            id: "e0".into(),
            embedding_idx: 0,
            properties: HashMap::new(),
        }];
        let results = scorer
            .score_entities(&embs, &entities, &query)
            .expect("score");
        let se = &results[0];
        assert!(
            (se.combined_score - se.neural_score).abs() < 1e-12,
            "damping=0 → combined == neural, got combined={} neural={}",
            se.combined_score,
            se.neural_score
        );
    }

    #[test]
    fn test_damping_one_pure_physics() {
        let scorer = PinnEntityScorer::new(toy_encoder(), thermal_ctx(), 1.0);
        let embs = manual_embeddings(vec![[1.0, 0.0, 0.0, 0.0]]);
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let mut props = HashMap::new();
        props.insert("time_s".to_string(), 1.0);
        props.insert("length_m".to_string(), (1e-5_f64).sqrt());
        let entities = vec![KgEntity {
            id: "e0".into(),
            embedding_idx: 0,
            properties: props,
        }];
        let results = scorer
            .score_entities(&embs, &entities, &query)
            .expect("score");
        let se = &results[0];
        assert!(
            (se.combined_score - se.physics_score).abs() < 1e-12,
            "damping=1 → combined == physics, got combined={} physics={}",
            se.combined_score,
            se.physics_score
        );
    }

    #[test]
    fn test_rank_descending_order() {
        let scorer = PinnEntityScorer::new(toy_encoder(), thermal_ctx(), 0.3);

        // Build manual embeddings: entity 0 = same as query, entity 1 = orthogonal
        let embs = manual_embeddings(vec![
            [1.0, 0.0, 0.0, 0.0], // will have high neural
            [0.0, 1.0, 0.0, 0.0], // orthogonal
        ]);
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let entities = vec![
            KgEntity {
                id: "e0".into(),
                embedding_idx: 0,
                properties: HashMap::new(),
            },
            KgEntity {
                id: "e1".into(),
                embedding_idx: 1,
                properties: HashMap::new(),
            },
        ];
        // Use score_entities directly to avoid GNN re-encode
        let mut results = scorer
            .score_entities(&embs, &entities, &query)
            .expect("score");
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Rank must be descending
        for w in results.windows(2) {
            assert!(
                w[0].combined_score >= w[1].combined_score,
                "not descending: {} > {}",
                w[0].combined_score,
                w[1].combined_score
            );
        }
    }

    #[test]
    fn test_empty_entity_list_errors() {
        let scorer = PinnEntityScorer::new(toy_encoder(), thermal_ctx(), 0.3);
        let embs = manual_embeddings(vec![[1.0, 0.0, 0.0, 0.0]]);
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let result = scorer.score_entities(&embs, &[], &query);
        assert!(matches!(result, Err(PinnScorerError::EmptyEntityList)));
    }

    #[test]
    fn test_out_of_bounds_embedding_errors() {
        let scorer = PinnEntityScorer::new(toy_encoder(), thermal_ctx(), 0.3);
        let embs = manual_embeddings(vec![[1.0, 0.0, 0.0, 0.0]]);
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let entities = vec![KgEntity {
            id: "e_bad".into(),
            embedding_idx: 99, // out of bounds
            properties: HashMap::new(),
        }];
        let result = scorer.score_entities(&embs, &entities, &query);
        assert!(matches!(
            result,
            Err(PinnScorerError::EmbeddingIndexOutOfBounds { idx: 99, max: 1 })
        ));
    }

    #[test]
    fn test_encode_and_score_returns_results() {
        let scorer = PinnEntityScorer::new(toy_encoder(), thermal_ctx(), 0.3);
        let kg = toy_kg();
        let query = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let entities: Vec<KgEntity> = (0..4)
            .map(|i| KgEntity {
                id: format!("e{i}"),
                embedding_idx: i,
                properties: HashMap::new(),
            })
            .collect();
        let results = scorer
            .encode_and_score(&kg, &entities, &query)
            .expect("score");
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!((0.0..=1.0).contains(&r.neural_score));
            assert!((0.0..=1.0).contains(&r.physics_score));
            assert!((0.0..=1.0).contains(&r.combined_score));
        }
    }

    #[test]
    fn test_rank_result_descending() {
        let scorer = PinnEntityScorer::new(toy_encoder(), thermal_ctx(), 0.3);
        let kg = toy_kg();
        let query = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let entities: Vec<KgEntity> = (0..4)
            .map(|i| KgEntity {
                id: format!("e{i}"),
                embedding_idx: i,
                properties: HashMap::new(),
            })
            .collect();
        let ranked = scorer.rank(&kg, &entities, &query).expect("rank");
        for w in ranked.windows(2) {
            assert!(
                w[0].combined_score >= w[1].combined_score,
                "rank not descending: {} then {}",
                w[0].combined_score,
                w[1].combined_score
            );
        }
    }
}
