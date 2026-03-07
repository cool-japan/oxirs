//! # MockEmbeddingGenerator - Trait Implementations
//!
//! This module contains trait implementations for `MockEmbeddingGenerator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `AsAny`
//! - `EmbeddingGenerator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
use super::types::MockEmbeddingGenerator;
#[cfg(test)]
use crate::embeddings::{AsAny, EmbeddingGenerator};
#[cfg(test)]
use crate::{EmbeddableContent, EmbeddingConfig};
#[cfg(test)]
use std::hash::{Hash, Hasher};

#[cfg(test)]
impl Default for MockEmbeddingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl AsAny for MockEmbeddingGenerator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
impl EmbeddingGenerator for MockEmbeddingGenerator {
    fn generate(&self, content: &EmbeddableContent) -> anyhow::Result<crate::Vector> {
        let text = content.to_text();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        let mut embedding = Vec::with_capacity(self.config.dimensions);
        let mut seed = hash;
        for _ in 0..self.config.dimensions {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let value = (seed as f64 / u64::MAX as f64) as f32;
            embedding.push(value * 2.0 - 1.0);
        }
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for value in &mut embedding {
                *value /= magnitude;
            }
        }
        Ok(crate::Vector::new(embedding))
    }
    fn dimensions(&self) -> usize {
        self.config.dimensions
    }
    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}
