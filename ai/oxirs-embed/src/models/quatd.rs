//! QuatD: Quaternion-based Knowledge Graph Embeddings
//!
//! QuatD extends complex embeddings to use quaternions for modeling
//! both symmetry and anti-symmetry in knowledge graphs.

use crate::{EmbeddingModel, ModelConfig, EmbeddingError};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// QuatD model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuatDConfig {
    /// Embedding dimension (must be divisible by 4 for quaternions)
    pub embedding_dim: usize,
    /// Learning rate
    pub lr: f64,
    /// Regularization factor
    pub reg: f64,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for QuatDConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 200,
            lr: 0.001,
            reg: 0.01,
            dropout: 0.1,
        }
    }
}

impl ModelConfig for QuatDConfig {
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Quaternion representation
#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    pub w: f64, // real part
    pub x: f64, // i component
    pub y: f64, // j component
    pub z: f64, // k component
}

impl Quaternion {
    /// Create a new quaternion
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Quaternion multiplication
    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Quaternion conjugate
    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternion norm
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Inner product with another quaternion
    pub fn inner_product(&self, other: &Quaternion) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// QuatD model implementation
#[derive(Debug)]
pub struct QuatD {
    config: QuatDConfig,
    entity_embeddings: Option<Array2<f64>>,
    relation_embeddings: Option<Array2<f64>>,
}

impl QuatD {
    /// Create a new QuatD model
    pub fn new(config: QuatDConfig) -> Self {
        Self {
            config,
            entity_embeddings: None,
            relation_embeddings: None,
        }
    }

    /// Initialize model parameters
    pub fn init_params(&mut self, num_entities: usize, num_relations: usize) -> Result<(), Error> {
        if self.config.embedding_dim % 4 != 0 {
            return Err(Error::ModelError(
                "Embedding dimension must be divisible by 4 for quaternions".to_string(),
            ));
        }

        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1).map_err(|e| Error::ModelError(e.to_string()))?;

        // Initialize entity embeddings
        let mut entity_emb = Array2::zeros((num_entities, self.config.embedding_dim));
        for elem in entity_emb.iter_mut() {
            *elem = normal.sample(&mut rng);
        }
        self.entity_embeddings = Some(entity_emb);

        // Initialize relation embeddings
        let mut relation_emb = Array2::zeros((num_relations, self.config.embedding_dim));
        for elem in relation_emb.iter_mut() {
            *elem = normal.sample(&mut rng);
        }
        self.relation_embeddings = Some(relation_emb);

        Ok(())
    }

    /// Convert vector slice to quaternions
    fn vec_to_quaternions(&self, vec: &[f64]) -> Vec<Quaternion> {
        vec.chunks(4)
            .map(|chunk| {
                if chunk.len() == 4 {
                    Quaternion::new(chunk[0], chunk[1], chunk[2], chunk[3])
                } else {
                    // Pad with zeros if needed
                    let mut padded = [0.0; 4];
                    for (i, &val) in chunk.iter().enumerate() {
                        padded[i] = val;
                    }
                    Quaternion::new(padded[0], padded[1], padded[2], padded[3])
                }
            })
            .collect()
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

        let h_vec = entity_emb.row(head).to_vec();
        let r_vec = relation_emb.row(relation).to_vec();
        let t_vec = entity_emb.row(tail).to_vec();

        let h_quats = self.vec_to_quaternions(&h_vec);
        let r_quats = self.vec_to_quaternions(&r_vec);
        let t_quats = self.vec_to_quaternions(&t_vec);

        // QuatD scoring: Re(<h, r, conj(t)>)
        let mut score = 0.0;
        for ((h_q, r_q), t_q) in h_quats.iter().zip(r_quats.iter()).zip(t_quats.iter()) {
            let temp = h_q.multiply(r_q);
            let t_conj = t_q.conjugate();
            score += temp.inner_product(&t_conj);
        }

        Ok(score)
    }
}

impl EmbeddingModel for QuatD {
    type Config = QuatDConfig;

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
    fn test_quaternion_operations() {
        let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);

        let product = q1.multiply(&q2);
        assert!((product.w - (-60.0)).abs() < 1e-10);

        let conjugate = q1.conjugate();
        assert_eq!(conjugate.w, 1.0);
        assert_eq!(conjugate.x, -2.0);

        let norm = q1.norm();
        assert!((norm - (1.0 + 4.0 + 9.0 + 16.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_quatd_creation() {
        let config = QuatDConfig::default();
        let quatd = QuatD::new(config);
        assert!(quatd.entity_embeddings.is_none());
        assert!(quatd.relation_embeddings.is_none());
    }

    #[test]
    fn test_quatd_init() {
        let config = QuatDConfig::default();
        let mut quatd = QuatD::new(config);
        quatd.init_params(100, 50).unwrap();

        assert!(quatd.entity_embeddings.is_some());
        assert!(quatd.relation_embeddings.is_some());
    }

    #[test]
    fn test_quatd_invalid_dim() {
        let mut config = QuatDConfig::default();
        config.embedding_dim = 201; // Not divisible by 4
        let mut quatd = QuatD::new(config);

        let result = quatd.init_params(100, 50);
        assert!(result.is_err());
    }
}
