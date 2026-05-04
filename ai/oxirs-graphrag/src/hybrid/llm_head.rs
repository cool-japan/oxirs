//! Hybrid LLM head: frozen GNN encoder + learnable soft-prompt projector + LLM provider.

use std::sync::Arc;

use scirs2_core::ndarray_ext::Array2;

use super::provider::{CompletionRequest, LlmError, LlmProvider};
use super::soft_prompt::SoftPromptProjector;
use crate::gnn_encoder::{EntityEmbeddings, GraphSageEncoder, KgGraph};

/// A KGQA training example.
#[derive(Debug, Clone)]
pub struct KgqaExample {
    pub question: String,
    /// Expected answer string (used for cross-entropy-like loss approximation).
    pub answer: String,
    /// Key entities relevant to the question (used for entity lookup).
    pub entity_ids: Vec<usize>,
}

/// Training history for the LLM head projector.
#[derive(Debug, Default)]
pub struct LlmHeadHistory {
    pub epoch_losses: Vec<f64>,
}

/// Hybrid LLM head: frozen GNN encoder + learnable soft-prompt projector + LLM provider.
pub struct HybridLlmHead<P: LlmProvider> {
    encoder: Arc<GraphSageEncoder>,
    projector: SoftPromptProjector,
    provider: P,
    learning_rate: f64,
}

impl<P: LlmProvider> HybridLlmHead<P> {
    pub fn new(
        encoder: Arc<GraphSageEncoder>,
        projector: SoftPromptProjector,
        provider: P,
    ) -> Self {
        Self {
            encoder,
            projector,
            provider,
            learning_rate: 0.01,
        }
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Answer a question by retrieving entity embeddings from the KG,
    /// projecting them as a soft prompt, and querying the LLM.
    pub async fn answer(&mut self, question: &str, kg: &KgGraph) -> Result<String, LlmError> {
        // Encode the graph (encoder is frozen — no gradient update).
        let embeddings: EntityEmbeddings = self
            .encoder
            .encode(kg)
            .map_err(|e| LlmError::Provider(e.to_string()))?;

        // Project top-K entity embeddings to prompt tokens.
        let k = embeddings.embeddings.nrows().min(5);
        let dim = embeddings.embeddings.ncols();

        // Manual row extraction — avoids dependency on s! macro availability.
        let mut input_data = vec![0.0f64; k * dim];
        for i in 0..k {
            for j in 0..dim {
                input_data[i * dim + j] = embeddings.embeddings[[i, j]];
            }
        }
        let entity_2d = Array2::from_shape_vec((k, dim), input_data)
            .map_err(|e| LlmError::Provider(e.to_string()))?;

        let projected = self.projector.forward(&entity_2d);

        // Build soft prompt: summarise projected embeddings as text surrogates.
        let mut soft_prompt = String::new();
        for i in 0..k {
            let mean_val: f64 = if projected.ncols() == 0 {
                0.0
            } else {
                (0..projected.ncols())
                    .map(|j| projected[[i, j]])
                    .sum::<f64>()
                    / projected.ncols() as f64
            };
            soft_prompt.push_str(&format!("[entity_{i}:{:.3}] ", mean_val));
        }

        let prompt = format!("{soft_prompt}\nQuestion: {question}\nAnswer:");
        let response = self
            .provider
            .complete(&CompletionRequest {
                prompt,
                max_tokens: 128,
            })
            .await?;
        Ok(response.text)
    }

    /// Train the projector (GNN is frozen) on KGQA examples.
    ///
    /// Uses a surrogate loss: MSE toward unit vector (encourage discriminative projections),
    /// since we cannot compute real LLM log-probs without a gradient-capable model.
    pub fn train_projector(
        &mut self,
        kg: &KgGraph,
        examples: &[KgqaExample],
        epochs: usize,
    ) -> Result<LlmHeadHistory, String> {
        let mut history = LlmHeadHistory::default();
        let embeddings = self.encoder.encode(kg).map_err(|e| e.to_string())?;

        for _ in 0..epochs {
            let mut epoch_loss = 0.0;
            for ex in examples {
                let k = ex.entity_ids.len().min(embeddings.embeddings.nrows());
                if k == 0 {
                    continue;
                }

                let mut rows: Vec<usize> = ex
                    .entity_ids
                    .iter()
                    .copied()
                    .filter(|&id| id < embeddings.embeddings.nrows())
                    .take(k)
                    .collect();
                if rows.is_empty() {
                    rows.push(0);
                }

                let dim = embeddings.embeddings.ncols();
                let n = rows.len();
                let mut input_data = vec![0.0f64; n * dim];
                for (i, &row_idx) in rows.iter().enumerate() {
                    for j in 0..dim {
                        input_data[i * dim + j] = embeddings.embeddings[[row_idx, j]];
                    }
                }
                let input =
                    Array2::from_shape_vec((n, dim), input_data).map_err(|e| e.to_string())?;

                let projected = self.projector.forward(&input);
                let prompt_dim = projected.ncols();
                let mut loss = 0.0f64;
                let mut d_output: Array2<f64> = Array2::zeros((n, prompt_dim));
                for i in 0..n {
                    for j in 0..prompt_dim {
                        let target = if j == 0 { 1.0_f64 } else { 0.0 };
                        let diff = projected[[i, j]] - target;
                        loss += diff * diff;
                        d_output[[i, j]] = 2.0 * diff / (n * prompt_dim).max(1) as f64;
                    }
                }
                loss /= (n * prompt_dim).max(1) as f64;
                epoch_loss += loss;
                self.projector.backward(&d_output, self.learning_rate);
            }
            history
                .epoch_losses
                .push(epoch_loss / examples.len().max(1) as f64);
        }
        Ok(history)
    }

    /// Snapshot of the projector's current weight matrix (for inspection / tests).
    pub fn projector_weights_snapshot(&self) -> Array2<f64> {
        self.projector.weights_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn_encoder::{GraphSageConfig, GraphSageEncoder};
    use crate::hybrid::provider::LocalProvider;
    use std::sync::Arc;

    fn toy_kg() -> KgGraph {
        KgGraph {
            num_nodes: 4,
            edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            node_features: Array2::zeros((4, 8)),
        }
    }

    fn toy_config() -> GraphSageConfig {
        GraphSageConfig {
            input_dim: 8,
            hidden_dim: 8,
            output_dim: 8,
            num_layers: 1,
            dropout: 0.0,
            k_neighbors: 2,
            learning_rate: 0.0,
        }
    }

    fn make_head() -> HybridLlmHead<LocalProvider> {
        let encoder =
            Arc::new(GraphSageEncoder::new_with_seed(&toy_config(), 1).expect("construct encoder"));
        let projector = SoftPromptProjector::new(8, 8, 42);
        HybridLlmHead::new(encoder, projector, LocalProvider::new())
    }

    #[tokio::test]
    async fn test_answer_returns_non_empty() {
        let mut head = make_head();
        let answer = head
            .answer("Who is entity 1?", &toy_kg())
            .await
            .expect("should answer");
        assert!(!answer.is_empty());
    }

    #[test]
    fn test_train_projector_returns_history() {
        let mut head = make_head();
        let examples = vec![KgqaExample {
            question: "q".to_string(),
            answer: "a".to_string(),
            entity_ids: vec![0, 1],
        }];
        let history = head
            .train_projector(&toy_kg(), &examples, 3)
            .expect("train");
        assert_eq!(history.epoch_losses.len(), 3);
    }

    #[test]
    fn test_train_projector_loss_decreases() {
        let encoder =
            Arc::new(GraphSageEncoder::new_with_seed(&toy_config(), 2).expect("construct encoder"));
        let projector = SoftPromptProjector::new(8, 8, 99);
        let mut head =
            HybridLlmHead::new(encoder, projector, LocalProvider::new()).with_learning_rate(0.1);
        let examples = vec![
            KgqaExample {
                question: "q0".to_string(),
                answer: "a0".to_string(),
                entity_ids: vec![0],
            },
            KgqaExample {
                question: "q1".to_string(),
                answer: "a1".to_string(),
                entity_ids: vec![1],
            },
        ];
        let history = head
            .train_projector(&toy_kg(), &examples, 20)
            .expect("train");
        let first = history.epoch_losses[0];
        let last = *history.epoch_losses.last().expect("non-empty losses");
        assert!(last <= first, "loss should not increase: {first} → {last}");
    }
}
