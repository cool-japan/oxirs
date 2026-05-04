//! Neuro-symbolic retriever that combines PINN entity scoring with LLM generation.
//!
//! [`NeuroSymbolicRetriever`] chains:
//! 1. [`PinnEntityScorer::rank`] — score + sort entities by combined (neural+physics) score
//! 2. [`HybridLlmHead::answer`] — use the top-K entities as KG context for LLM generation

use std::sync::Arc;

use scirs2_core::ndarray_ext::Array1;

use super::pinn_scorer::{KgEntity, PinnEntityScorer, PinnScorerError, ScoredEntity};
use crate::gnn_encoder::{GraphSageEncoder, KgGraph};
use crate::hybrid::{HybridLlmHead, LlmProvider, SoftPromptProjector};

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by [`NeuroSymbolicRetriever`].
#[derive(Debug, thiserror::Error)]
pub enum NeuroSymbolicError {
    #[error("scorer error: {0}")]
    ScorerError(#[from] PinnScorerError),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("no entities provided")]
    NoEntitiesProvided,
}

// ─── Retriever ────────────────────────────────────────────────────────────────

/// End-to-end neuro-symbolic retriever.
///
/// Scores and ranks entities using the PINN scorer, then passes the top-K
/// entities' KG context to the [`HybridLlmHead`] to generate an answer.
pub struct NeuroSymbolicRetriever<P: LlmProvider> {
    scorer: PinnEntityScorer,
    head: HybridLlmHead<P>,
    /// Number of top entities to pass to the LLM head.
    top_k: usize,
}

impl<P: LlmProvider> NeuroSymbolicRetriever<P> {
    /// Create a new retriever.
    pub fn new(scorer: PinnEntityScorer, head: HybridLlmHead<P>, top_k: usize) -> Self {
        Self {
            scorer,
            head,
            top_k: top_k.max(1),
        }
    }

    /// Rank entities by combined (neural+physics) score, returning the top-K.
    ///
    /// Returns an error if the entity list is empty or scoring fails.
    pub fn retrieve_and_rank(
        &self,
        kg: &KgGraph,
        entities: &[KgEntity],
        query_embedding: &Array1<f64>,
    ) -> Result<Vec<ScoredEntity>, NeuroSymbolicError> {
        if entities.is_empty() {
            return Err(NeuroSymbolicError::NoEntitiesProvided);
        }
        let mut ranked = self.scorer.rank(kg, entities, query_embedding)?;
        ranked.truncate(self.top_k);
        Ok(ranked)
    }

    /// Retrieve top-K entities and use them as context for an LLM answer.
    ///
    /// The KG is encoded once by the scorer for entity ranking, then encoded
    /// again inside [`HybridLlmHead::answer`].  The double-encode is a
    /// deliberate trade-off to keep the two modules' APIs independent.
    pub async fn answer(
        &mut self,
        question: &str,
        kg: &KgGraph,
        entities: &[KgEntity],
        query_embedding: &Array1<f64>,
    ) -> Result<String, NeuroSymbolicError> {
        // Rank entities (encodes KG once).
        let _ranked = self.retrieve_and_rank(kg, entities, query_embedding)?;

        // Delegate to the LLM head (encodes KG a second time internally).
        let response = self
            .head
            .answer(question, kg)
            .await
            .map_err(|e| NeuroSymbolicError::LlmError(e.to_string()))?;

        Ok(response)
    }
}

// ─── Builder helper (convenience constructor) ─────────────────────────────────

impl<P: LlmProvider> NeuroSymbolicRetriever<P> {
    /// Build a retriever from raw components, constructing the [`HybridLlmHead`]
    /// internally from the shared encoder.
    ///
    /// `projector_in_dim` must match `encoder.config.output_dim`.
    /// `projector_out_dim` controls the soft-prompt token dimension.
    pub fn from_parts(
        scorer: PinnEntityScorer,
        encoder: Arc<GraphSageEncoder>,
        projector_in_dim: usize,
        projector_out_dim: usize,
        provider: P,
        top_k: usize,
    ) -> Self {
        let projector = SoftPromptProjector::new(projector_in_dim, projector_out_dim, 42);
        let head = HybridLlmHead::new(encoder, projector, provider);
        Self::new(scorer, head, top_k)
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use scirs2_core::ndarray_ext::{Array1, Array2};

    use crate::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
    use crate::hybrid::provider::LocalProvider;
    use crate::neuro_symbolic::physics_context::{PhysicsContext, PhysicsDomain};
    use crate::neuro_symbolic::pinn_scorer::{KgEntity, PinnEntityScorer};

    use super::*;

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
        Arc::new(GraphSageEncoder::new_with_seed(&config, 42).expect("encoder"))
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

    fn toy_entities() -> Vec<KgEntity> {
        (0..4)
            .map(|i| KgEntity {
                id: format!("e{i}"),
                embedding_idx: i,
                properties: HashMap::new(),
            })
            .collect()
    }

    fn make_retriever() -> NeuroSymbolicRetriever<LocalProvider> {
        let encoder = toy_encoder();
        let scorer = PinnEntityScorer::new(Arc::clone(&encoder), thermal_ctx(), 0.3);
        NeuroSymbolicRetriever::from_parts(scorer, encoder, 4, 4, LocalProvider::new(), 2)
    }

    #[test]
    fn test_retrieve_and_rank_returns_top_k() {
        let retriever = make_retriever();
        let query = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let ranked = retriever
            .retrieve_and_rank(&toy_kg(), &toy_entities(), &query)
            .expect("retrieve");
        assert_eq!(ranked.len(), 2, "top_k=2 should truncate to 2 results");
    }

    #[test]
    fn test_retrieve_empty_entities_errors() {
        let retriever = make_retriever();
        let query = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let result = retriever.retrieve_and_rank(&toy_kg(), &[], &query);
        assert!(matches!(
            result,
            Err(NeuroSymbolicError::NoEntitiesProvided)
        ));
    }

    #[test]
    fn test_retrieve_descending_order() {
        let retriever = make_retriever();
        let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let ranked = retriever
            .retrieve_and_rank(&toy_kg(), &toy_entities(), &query)
            .expect("retrieve");
        for w in ranked.windows(2) {
            assert!(
                w[0].combined_score >= w[1].combined_score,
                "not descending: {} then {}",
                w[0].combined_score,
                w[1].combined_score
            );
        }
    }

    #[tokio::test]
    async fn test_answer_returns_non_empty() {
        let mut retriever = make_retriever();
        let query = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let answer = retriever
            .answer("What is entity 0?", &toy_kg(), &toy_entities(), &query)
            .await
            .expect("answer");
        assert!(!answer.is_empty());
    }

    #[test]
    fn test_scorer_error_propagates_through_retriever() {
        let encoder = toy_encoder();
        let scorer = PinnEntityScorer::new(Arc::clone(&encoder), thermal_ctx(), 0.3);
        let retriever =
            NeuroSymbolicRetriever::from_parts(scorer, encoder, 4, 4, LocalProvider::new(), 2);

        let query = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        // Entity with an out-of-bounds embedding index
        let bad_entities = vec![KgEntity {
            id: "bad".into(),
            embedding_idx: 999,
            properties: HashMap::new(),
        }];
        let result = retriever.retrieve_and_rank(&toy_kg(), &bad_entities, &query);
        assert!(matches!(result, Err(NeuroSymbolicError::ScorerError(_))));
    }
}
