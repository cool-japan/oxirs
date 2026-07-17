//! In-memory vector index implementations and the `VectorIndex` trait.

use anyhow::Result;
use std::collections::HashMap;

use crate::similarity;
use crate::Vector;
use crate::VectorId;

/// Vector index trait for efficient similarity search
pub trait VectorIndex: Send + Sync {
    /// Insert a vector with associated URI
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()>;

    /// Find k nearest neighbors
    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>>;

    /// Find all vectors within threshold similarity
    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>>;

    /// Get a vector by its URI
    fn get_vector(&self, uri: &str) -> Option<&Vector>;

    /// Add a vector with associated ID and metadata
    fn add_vector(
        &mut self,
        id: VectorId,
        vector: Vector,
        _metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        // Default implementation that delegates to insert
        self.insert(id, vector)
    }

    /// Update an existing vector
    fn update_vector(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        // Default implementation that delegates to insert
        self.insert(id, vector)
    }

    /// Update metadata for a vector
    fn update_metadata(&mut self, _id: VectorId, _metadata: HashMap<String, String>) -> Result<()> {
        // Default implementation (no-op)
        Ok(())
    }

    /// Remove a vector by its ID
    fn remove_vector(&mut self, _id: VectorId) -> Result<()> {
        // Default implementation (no-op)
        Ok(())
    }

    /// Iterate all stored (id, vector) pairs.
    ///
    /// The default returns an empty list; concrete index types that hold their
    /// vectors in memory (or can reconstruct them, e.g. via decoding quantized
    /// codes) should override this **and** [`VectorIndex::supports_enumeration`]
    /// so callers like `VectorStore::save_to_disk` can tell real emptiness
    /// apart from "this index type cannot enumerate its vectors".
    fn iter_vectors(&self) -> Vec<(String, Vector)> {
        Vec::new()
    }

    /// Whether [`VectorIndex::iter_vectors`] returns a real, complete
    /// enumeration of the vectors held by this index.
    ///
    /// Index types that override `iter_vectors` with a real implementation
    /// (e.g. [`MemoryVectorIndex`], `HnswIndex`, `IvfIndex`, `PQIndex`) must
    /// also override this to return `true`. Callers that need to persist or
    /// otherwise fully enumerate an index (e.g. `VectorStore::save_to_disk`)
    /// should check this flag and fail loudly instead of silently persisting
    /// an empty snapshot when it is `false`.
    fn supports_enumeration(&self) -> bool {
        false
    }
}

/// In-memory vector index implementation
pub struct MemoryVectorIndex {
    vectors: Vec<(String, Vector)>,
    similarity_config: similarity::SimilarityConfig,
}

impl MemoryVectorIndex {
    /// Create a new empty in-memory vector index with default similarity config.
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            similarity_config: similarity::SimilarityConfig::default(),
        }
    }

    /// Create a new in-memory vector index with a custom similarity configuration.
    pub fn with_similarity_config(config: similarity::SimilarityConfig) -> Self {
        Self {
            vectors: Vec::new(),
            similarity_config: config,
        }
    }
}

impl Default for MemoryVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for MemoryVectorIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        // Check if vector already exists and update it
        if let Some(pos) = self.vectors.iter().position(|(id, _)| id == &uri) {
            self.vectors[pos] = (uri, vector);
        } else {
            self.vectors.push((uri, vector));
        }
        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let metric = self.similarity_config.primary_metric;
        let query_f32 = query.as_f32();
        let mut similarities: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(uri, vec)| {
                let vec_f32 = vec.as_f32();
                let sim = metric.similarity(&query_f32, &vec_f32).unwrap_or(0.0);
                (uri.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);

        Ok(similarities)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let metric = self.similarity_config.primary_metric;
        let query_f32 = query.as_f32();
        let similarities: Vec<(String, f32)> = self
            .vectors
            .iter()
            .filter_map(|(uri, vec)| {
                let vec_f32 = vec.as_f32();
                let sim = metric.similarity(&query_f32, &vec_f32).unwrap_or(0.0);
                if sim >= threshold {
                    Some((uri.clone(), sim))
                } else {
                    None
                }
            })
            .collect();

        Ok(similarities)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        self.vectors.iter().find(|(u, _)| u == uri).map(|(_, v)| v)
    }

    fn update_vector(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        if let Some(pos) = self.vectors.iter().position(|(uri, _)| uri == &id) {
            self.vectors[pos] = (id, vector);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Vector with id '{}' not found", id))
        }
    }

    fn remove_vector(&mut self, id: VectorId) -> Result<()> {
        if let Some(pos) = self.vectors.iter().position(|(uri, _)| uri == &id) {
            self.vectors.remove(pos);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Vector with id '{}' not found", id))
        }
    }

    fn iter_vectors(&self) -> Vec<(String, Vector)> {
        self.vectors.clone()
    }

    fn supports_enumeration(&self) -> bool {
        true
    }
}
