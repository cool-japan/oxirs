//! Shared serialization helpers for knowledge-graph embedding models.
//!
//! Embedding models keep their weights in `scirs2_core` ndarray matrices which
//! do not serialize directly through `oxicode`/`serde`. These helpers convert
//! those matrices to/from flat, shape-tagged vectors and provide a serializable
//! snapshot of the shared [`BaseModel`] state so that each model's `save`/`load`
//! implementation stays small and consistent.

use crate::models::base::BaseModel;
use crate::ModelConfig;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use scirs2_core::ndarray_ext::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Serializable, shape-tagged representation of a 2-D `f64` matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixF64 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl MatrixF64 {
    /// Capture a matrix into a serializable form (row-major).
    pub fn from_array(a: &Array2<f64>) -> Self {
        let (rows, cols) = a.dim();
        Self {
            rows,
            cols,
            data: a.iter().copied().collect(),
        }
    }

    /// Reconstruct the matrix, validating the element count against the shape.
    pub fn to_array(&self) -> Result<Array2<f64>> {
        if self.rows * self.cols != self.data.len() {
            return Err(anyhow!(
                "corrupt matrix payload: {}x{} != {} elements",
                self.rows,
                self.cols,
                self.data.len()
            ));
        }
        Array2::from_shape_vec((self.rows, self.cols), self.data.clone())
            .map_err(|e| anyhow!("failed to rebuild matrix: {}", e))
    }
}

/// Serializable, shape-tagged representation of a 2-D `f32` matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixF32 {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl MatrixF32 {
    pub fn from_array(a: &Array2<f32>) -> Self {
        let (rows, cols) = a.dim();
        Self {
            rows,
            cols,
            data: a.iter().copied().collect(),
        }
    }

    pub fn to_array(&self) -> Result<Array2<f32>> {
        if self.rows * self.cols != self.data.len() {
            return Err(anyhow!(
                "corrupt matrix payload: {}x{} != {} elements",
                self.rows,
                self.cols,
                self.data.len()
            ));
        }
        Array2::from_shape_vec((self.rows, self.cols), self.data.clone())
            .map_err(|e| anyhow!("failed to rebuild matrix: {}", e))
    }
}

/// Serializable, shape-tagged representation of a 1-D `f32` vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorF32 {
    pub data: Vec<f32>,
}

impl VectorF32 {
    pub fn from_array(a: &Array1<f32>) -> Self {
        Self {
            data: a.iter().copied().collect(),
        }
    }

    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.data.clone())
    }
}

/// Serializable, shape-tagged representation of a 3-D `f64` tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor3F64 {
    pub d0: usize,
    pub d1: usize,
    pub d2: usize,
    pub data: Vec<f64>,
}

impl Tensor3F64 {
    pub fn from_array(a: &Array3<f64>) -> Self {
        let (d0, d1, d2) = a.dim();
        Self {
            d0,
            d1,
            d2,
            data: a.iter().copied().collect(),
        }
    }

    pub fn to_array(&self) -> Result<Array3<f64>> {
        if self.d0 * self.d1 * self.d2 != self.data.len() {
            return Err(anyhow!(
                "corrupt tensor payload: {}x{}x{} != {} elements",
                self.d0,
                self.d1,
                self.d2,
                self.data.len()
            ));
        }
        Array3::from_shape_vec((self.d0, self.d1, self.d2), self.data.clone())
            .map_err(|e| anyhow!("failed to rebuild tensor: {}", e))
    }
}

/// Serializable snapshot of the shared [`BaseModel`] state.
///
/// `positive_triples` is intentionally omitted: it is a lookup set derived from
/// `triples` and is rebuilt on restore, keeping the payload compact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseModelSnapshot {
    pub config: ModelConfig,
    pub model_id: Uuid,
    pub entity_to_id: HashMap<String, usize>,
    pub id_to_entity: HashMap<usize, String>,
    pub relation_to_id: HashMap<String, usize>,
    pub id_to_relation: HashMap<usize, String>,
    pub triples: Vec<(usize, usize, usize)>,
    pub is_trained: bool,
    pub creation_time: DateTime<Utc>,
    pub last_training_time: Option<DateTime<Utc>>,
}

impl BaseModelSnapshot {
    /// Capture the persistable state of a base model.
    pub fn capture(base: &BaseModel) -> Self {
        Self {
            config: base.config.clone(),
            model_id: base.model_id,
            entity_to_id: base.entity_to_id.clone(),
            id_to_entity: base.id_to_entity.clone(),
            relation_to_id: base.relation_to_id.clone(),
            id_to_relation: base.id_to_relation.clone(),
            triples: base.triples.clone(),
            is_trained: base.is_trained,
            creation_time: base.creation_time,
            last_training_time: base.last_training_time,
        }
    }

    /// Restore a base model in-place from this snapshot, rebuilding the
    /// derived `positive_triples` lookup set.
    pub fn restore_into(self, base: &mut BaseModel) {
        base.config = self.config;
        base.model_id = self.model_id;
        base.entity_to_id = self.entity_to_id;
        base.id_to_entity = self.id_to_entity;
        base.relation_to_id = self.relation_to_id;
        base.id_to_relation = self.id_to_relation;
        base.positive_triples = self.triples.iter().copied().collect();
        base.triples = self.triples;
        base.is_trained = self.is_trained;
        base.creation_time = self.creation_time;
        base.last_training_time = self.last_training_time;
    }
}
