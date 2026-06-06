//! Flat (brute-force) vector index.

use crate::ai::vector_store::{
    compute_similarity, IndexStats, SimilarityMetric, VectorData, VectorIndex,
};
use anyhow::Result;
use dashmap::DashMap;
use std::cmp::Ordering;
use std::collections::HashMap;

pub struct FlatIndex {
    vectors: HashMap<String, Vec<f32>>,
    stats: IndexStats,
}

impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatIndex {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            stats: IndexStats {
                index_type: "Flat".to_string(),
                num_vectors: 0,
                build_time: std::time::Duration::from_secs(0),
                memory_usage: 0,
            },
        }
    }
}

#[async_trait::async_trait]
impl VectorIndex for FlatIndex {
    async fn build(&mut self, vectors: &DashMap<String, VectorData>) -> Result<()> {
        let start = std::time::Instant::now();

        self.vectors.clear();
        for entry in vectors.iter() {
            self.vectors
                .insert(entry.key().clone(), entry.value().vector.clone());
        }

        self.stats.num_vectors = self.vectors.len();
        self.stats.build_time = start.elapsed();
        self.stats.memory_usage = self.vectors.len()
            * self
                .vectors
                .values()
                .next()
                .map(|v| v.len() * 4)
                .unwrap_or(0);

        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<(String, f32)>> {
        let mut similarities = Vec::new();

        for (id, vector) in &self.vectors {
            let similarity = compute_similarity(query, vector, metric)?;
            similarities.push((id.clone(), similarity));
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        similarities.truncate(k);
        Ok(similarities)
    }

    async fn add(&mut self, id: String, vector: Vec<f32>) -> Result<()> {
        self.vectors.insert(id, vector);
        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    async fn remove(&mut self, id: &str) -> Result<()> {
        self.vectors.remove(id);
        self.stats.num_vectors = self.vectors.len();
        Ok(())
    }

    fn get_stats(&self) -> IndexStats {
        self.stats.clone()
    }
}
