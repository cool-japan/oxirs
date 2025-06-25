//! TuckER: Tucker Decomposition for Knowledge Graph Embeddings
//!
//! TuckER is a tensor factorization model that performs link prediction
//! using Tucker decomposition on the binary tensor representation of knowledge graphs.

use super::base::{EmbeddingModel, ModelConfig};
use crate::Error;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// TuckER model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuckERConfig {
    /// Entity embedding dimension
    pub entity_dim: usize,
    /// Relation embedding dimension  
    pub relation_dim: usize,
    /// Core tensor dimensions
    pub core_dims: (usize, usize, usize),
    /// Learning rate
    pub lr: f64,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for TuckERConfig {
    fn default() -> Self {
        Self {
            entity_dim: 200,
            relation_dim: 200,
            core_dims: (200, 200, 200),
            lr: 0.003,
            dropout: 0.3,
        }
    }
}

impl ModelConfig for TuckERConfig {
    fn embedding_dim(&self) -> usize {
        self.entity_dim
    }
}

/// TuckER model implementation
#[derive(Debug)]
pub struct TuckER {
    config: TuckERConfig,
    entity_embeddings: Option<Array2<f64>>,
    relation_embeddings: Option<Array2<f64>>,
    core_tensor: Option<Array3<f64>>,
}

impl TuckER {
    /// Create a new TuckER model
    pub fn new(config: TuckERConfig) -> Self {
        Self {
            config,
            entity_embeddings: None,
            relation_embeddings: None,
            core_tensor: None,
        }
    }

    /// Initialize model parameters
    pub fn init_params(&mut self, num_entities: usize, num_relations: usize) -> Result<(), Error> {
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).map_err(|e| Error::ModelError(e.to_string()))?;

        // Initialize entity embeddings
        let mut entity_emb = Array2::zeros((num_entities, self.config.entity_dim));
        for elem in entity_emb.iter_mut() {
            *elem = normal.sample(&mut rng);
        }
        self.entity_embeddings = Some(entity_emb);

        // Initialize relation embeddings
        let mut relation_emb = Array2::zeros((num_relations, self.config.relation_dim));
        for elem in relation_emb.iter_mut() {
            *elem = normal.sample(&mut rng);
        }
        self.relation_embeddings = Some(relation_emb);

        // Initialize core tensor
        let mut core = Array3::zeros(self.config.core_dims);
        for elem in core.iter_mut() {
            *elem = normal.sample(&mut rng);
        }
        self.core_tensor = Some(core);

        Ok(())
    }

    /// Compute score for a triple (head, relation, tail)
    pub fn score(&self, head: usize, relation: usize, tail: usize) -> Result<f64, Error> {
        let entity_emb = self
            .entity_embeddings
            .as_ref()
            .ok_or_else(|| Error::ModelError("Entity embeddings not initialized".to_string()))?;
        let relation_emb = self
            .relation_embeddings
            .as_ref()
            .ok_or_else(|| Error::ModelError("Relation embeddings not initialized".to_string()))?;
        let core = self
            .core_tensor
            .as_ref()
            .ok_or_else(|| Error::ModelError("Core tensor not initialized".to_string()))?;

        let h_emb = entity_emb.row(head);
        let r_emb = relation_emb.row(relation);
        let t_emb = entity_emb.row(tail);

        // Compute Tucker decomposition score
        // score = sum_i,j,k h_i * r_j * t_k * W_ijk
        let mut score = 0.0;
        for i in 0..self.config.core_dims.0 {
            for j in 0..self.config.core_dims.1 {
                for k in 0..self.config.core_dims.2 {
                    if i < h_emb.len() && j < r_emb.len() && k < t_emb.len() {
                        score += h_emb[i] * r_emb[j] * t_emb[k] * core[(i, j, k)];
                    }
                }
            }
        }

        Ok(score)
    }
}

impl EmbeddingModel for TuckER {
    type Config = TuckERConfig;

    fn new(config: Self::Config) -> Self {
        Self::new(config)
    }

    fn train(&mut self, _triples: &[(usize, usize, usize)]) -> Result<(), Error> {
        // Training implementation would go here
        // For now, just return success
        Ok(())
    }

    fn predict(&self, head: usize, relation: usize, tail: usize) -> Result<f64, Error> {
        self.score(head, relation, tail)
    }

    fn get_entity_embedding(&self, entity_id: usize) -> Result<Vec<f64>, Error> {
        let embeddings = self
            .entity_embeddings
            .as_ref()
            .ok_or_else(|| Error::ModelError("Entity embeddings not initialized".to_string()))?;

        if entity_id >= embeddings.nrows() {
            return Err(Error::ModelError(format!(
                "Entity ID {} out of bounds",
                entity_id
            )));
        }

        Ok(embeddings.row(entity_id).to_vec())
    }

    fn get_relation_embedding(&self, relation_id: usize) -> Result<Vec<f64>, Error> {
        let embeddings = self
            .relation_embeddings
            .as_ref()
            .ok_or_else(|| Error::ModelError("Relation embeddings not initialized".to_string()))?;

        if relation_id >= embeddings.nrows() {
            return Err(Error::ModelError(format!(
                "Relation ID {} out of bounds",
                relation_id
            )));
        }

        Ok(embeddings.row(relation_id).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tucker_creation() {
        let config = TuckERConfig::default();
        let tucker = TuckER::new(config);
        assert!(tucker.entity_embeddings.is_none());
        assert!(tucker.relation_embeddings.is_none());
        assert!(tucker.core_tensor.is_none());
    }

    #[test]
    fn test_tucker_init() {
        let config = TuckERConfig::default();
        let mut tucker = TuckER::new(config);
        tucker.init_params(100, 50).unwrap();

        assert!(tucker.entity_embeddings.is_some());
        assert!(tucker.relation_embeddings.is_some());
        assert!(tucker.core_tensor.is_some());
    }
}
