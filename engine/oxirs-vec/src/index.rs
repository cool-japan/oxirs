//! Advanced vector indexing with HNSW and other efficient algorithms

use crate::Vector;

// Re-export VectorIndex trait for use by other modules
pub use crate::VectorIndex;
use anyhow::{anyhow, Result};
use oxirs_core::parallel::*;
use oxirs_core::Triple;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

#[cfg(feature = "hnsw")]
use hnsw_rs::prelude::*;

/// Type alias for filter functions
pub type FilterFunction = Box<dyn Fn(&str) -> bool>;
/// Type alias for filter functions with Send + Sync
pub type FilterFunctionSync = Box<dyn Fn(&str) -> bool + Send + Sync>;

/// Configuration for vector index
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Index type to use
    pub index_type: IndexType,
    /// Maximum number of connections for each node (for HNSW)
    pub max_connections: usize,
    /// Construction parameter (for HNSW)
    pub ef_construction: usize,
    /// Search parameter (for HNSW)
    pub ef_search: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// Whether to enable parallel operations
    pub parallel: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_type: IndexType::Hnsw,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            distance_metric: DistanceMetric::Cosine,
            parallel: true,
        }
    }
}

/// Available index types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IndexType {
    /// Hierarchical Navigable Small World
    Hnsw,
    /// Simple flat index (brute force)
    Flat,
    /// IVF (Inverted File) index
    Ivf,
    /// Product Quantization
    PQ,
}

/// Distance metrics supported
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Dot product (negative for max-heap behavior)
    DotProduct,
}

impl DistanceMetric {
    /// Calculate distance between two vectors
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        use oxirs_core::simd::SimdOps;

        match self {
            DistanceMetric::Cosine => f32::cosine_distance(a, b),
            DistanceMetric::Euclidean => f32::euclidean_distance(a, b),
            DistanceMetric::Manhattan => f32::manhattan_distance(a, b),
            DistanceMetric::DotProduct => -f32::dot(a, b), // Negative for max-heap
        }
    }

    /// Calculate distance between two Vector objects
    pub fn distance_vectors(&self, a: &Vector, b: &Vector) -> f32 {
        let a_f32 = a.as_f32();
        let b_f32 = b.as_f32();
        self.distance(&a_f32, &b_f32)
    }
}

/// Search result with distance/score
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub uri: String,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<HashMap<String, String>>,
}

impl Eq for SearchResult {}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Advanced vector index with multiple implementations
pub struct AdvancedVectorIndex {
    config: IndexConfig,
    vectors: Vec<(String, Vector)>,
    uri_to_id: HashMap<String, usize>,
    #[cfg(feature = "hnsw")]
    hnsw_index: Option<Hnsw<'static, f32, DistCosine>>,
    dimensions: Option<usize>,
}

impl AdvancedVectorIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            vectors: Vec::new(),
            uri_to_id: HashMap::new(),
            #[cfg(feature = "hnsw")]
            hnsw_index: None,
            dimensions: None,
        }
    }

    /// Build the index after adding all vectors
    pub fn build(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        match self.config.index_type {
            IndexType::Hnsw => {
                #[cfg(feature = "hnsw")]
                {
                    self.build_hnsw_index()?;
                }
                #[cfg(not(feature = "hnsw"))]
                {
                    return Err(anyhow!("HNSW feature not enabled"));
                }
            }
            IndexType::Flat => {
                // No special building needed for flat index
            }
            IndexType::Ivf | IndexType::PQ => {
                return Err(anyhow!("IVF and PQ indices not yet implemented"));
            }
        }

        Ok(())
    }

    #[cfg(feature = "hnsw")]
    fn build_hnsw_index(&mut self) -> Result<()> {
        if let Some(dimensions) = self.dimensions {
            let mut hnsw = Hnsw::<f32, DistCosine>::new(
                self.config.max_connections,
                self.vectors.len(),
                16, // layer factor
                self.config.ef_construction,
                DistCosine,
            );

            for (id, (_, vector)) in self.vectors.iter().enumerate() {
                let vector_f32 = vector.as_f32();
                hnsw.insert((&vector_f32, id));
            }

            self.hnsw_index = Some(hnsw);
        }

        Ok(())
    }

    /// Add metadata to a vector
    pub fn add_metadata(&mut self, _uri: &str, _metadata: HashMap<String, String>) -> Result<()> {
        // For now, we'll store metadata separately
        // In a full implementation, this would be integrated with the index
        Ok(())
    }

    /// Search with advanced parameters
    pub fn search_advanced(
        &self,
        query: &Vector,
        k: usize,
        ef: Option<usize>,
        filter: Option<FilterFunction>,
    ) -> Result<Vec<SearchResult>> {
        match self.config.index_type {
            IndexType::Hnsw => {
                #[cfg(feature = "hnsw")]
                {
                    self.search_hnsw(query, k, ef)
                }
                #[cfg(not(feature = "hnsw"))]
                {
                    let _ = ef;
                    self.search_flat(query, k, filter)
                }
            }
            _ => self.search_flat(query, k, filter),
        }
    }

    #[cfg(feature = "hnsw")]
    fn search_hnsw(
        &self,
        query: &Vector,
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        if let Some(ref hnsw) = self.hnsw_index {
            let search_ef = ef.unwrap_or(self.config.ef_search);
            let query_f32 = query.as_f32();
            let results = hnsw.search(&query_f32, k, search_ef);

            Ok(results
                .into_iter()
                .map(|result| SearchResult {
                    uri: self.vectors[result.d_id].0.clone(),
                    distance: result.distance,
                    score: 1.0 - result.distance, // Convert distance to similarity score
                    metadata: None,
                })
                .collect())
        } else {
            Err(anyhow!("HNSW index not built"))
        }
    }

    fn search_flat(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<FilterFunction>,
    ) -> Result<Vec<SearchResult>> {
        if self.config.parallel && self.vectors.len() > 1000 {
            // For parallel search, we need Send + Sync filter
            if filter.is_some() {
                // Fall back to sequential if filter is present but not Send + Sync
                self.search_flat_sequential(query, k, filter)
            } else {
                self.search_flat_parallel(query, k, None)
            }
        } else {
            self.search_flat_sequential(query, k, filter)
        }
    }

    fn search_flat_sequential(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<FilterFunction>,
    ) -> Result<Vec<SearchResult>> {
        let mut heap = BinaryHeap::new();

        for (uri, vector) in &self.vectors {
            if let Some(ref filter_fn) = filter {
                if !filter_fn(uri) {
                    continue;
                }
            }

            let distance = self.config.distance_metric.distance_vectors(query, vector);

            if heap.len() < k {
                heap.push(std::cmp::Reverse(SearchResult {
                    uri: uri.clone(),
                    distance,
                    score: 1.0 - distance, // Convert distance to similarity score
                    metadata: None,
                }));
            } else if let Some(std::cmp::Reverse(worst)) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(std::cmp::Reverse(SearchResult {
                        uri: uri.clone(),
                        distance,
                        score: 1.0 - distance, // Convert distance to similarity score
                        metadata: None,
                    }));
                }
            }
        }

        let mut results: Vec<SearchResult> = heap.into_iter().map(|r| r.0).collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        Ok(results)
    }

    fn search_flat_parallel(
        &self,
        query: &Vector,
        k: usize,
        filter: Option<FilterFunctionSync>,
    ) -> Result<Vec<SearchResult>> {
        // Split vectors into chunks for parallel processing
        let chunk_size = (self.vectors.len() / num_threads()).max(100);

        // Use Arc for thread-safe sharing of the filter
        let filter_arc = filter.map(Arc::new);

        // Process chunks in parallel and collect top-k from each
        let partial_results: Vec<Vec<SearchResult>> = self
            .vectors
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_heap = BinaryHeap::new();
                let filter_ref = filter_arc.as_ref();

                for (uri, vector) in chunk {
                    if let Some(filter_fn) = filter_ref {
                        if !filter_fn(uri) {
                            continue;
                        }
                    }

                    let distance = self.config.distance_metric.distance_vectors(query, vector);

                    if local_heap.len() < k {
                        local_heap.push(std::cmp::Reverse(SearchResult {
                            uri: uri.clone(),
                            distance,
                            score: 1.0 - distance, // Convert distance to similarity score
                            metadata: None,
                        }));
                    } else if let Some(std::cmp::Reverse(worst)) = local_heap.peek() {
                        if distance < worst.distance {
                            local_heap.pop();
                            local_heap.push(std::cmp::Reverse(SearchResult {
                                uri: uri.clone(),
                                distance,
                                score: 1.0 - distance, // Convert distance to similarity score
                                metadata: None,
                            }));
                        }
                    }
                }

                local_heap
                    .into_sorted_vec()
                    .into_iter()
                    .map(|r| r.0)
                    .collect()
            })
            .collect();

        // Merge results from all chunks
        let mut final_heap = BinaryHeap::new();
        for partial in partial_results {
            for result in partial {
                if final_heap.len() < k {
                    final_heap.push(std::cmp::Reverse(result));
                } else if let Some(std::cmp::Reverse(worst)) = final_heap.peek() {
                    if result.distance < worst.distance {
                        final_heap.pop();
                        final_heap.push(std::cmp::Reverse(result));
                    }
                }
            }
        }

        let mut results: Vec<SearchResult> = final_heap.into_iter().map(|r| r.0).collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        Ok(results)
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            num_vectors: self.vectors.len(),
            dimensions: self.dimensions.unwrap_or(0),
            index_type: self.config.index_type,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        let vector_memory = self.vectors.len()
            * (std::mem::size_of::<String>()
                + self.dimensions.unwrap_or(0) * std::mem::size_of::<f32>());

        let uri_map_memory =
            self.uri_to_id.len() * (std::mem::size_of::<String>() + std::mem::size_of::<usize>());

        vector_memory + uri_map_memory
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Add a vector with RDF triple and metadata (for compatibility with tests)
    pub fn add(
        &mut self,
        id: String,
        vector: Vec<f32>,
        _triple: Triple,
        _metadata: HashMap<String, String>,
    ) -> Result<()> {
        let vector_obj = Vector::new(vector);
        self.insert(id, vector_obj)
    }

    /// Search for nearest neighbors (for compatibility with tests)
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let query_vector = Vector::new(query.to_vec());
        let results = self.search_advanced(&query_vector, k, None, None)?;
        Ok(results)
    }
}

impl VectorIndex for AdvancedVectorIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        if let Some(dims) = self.dimensions {
            if vector.dimensions != dims {
                return Err(anyhow!(
                    "Vector dimensions ({}) don't match index dimensions ({})",
                    vector.dimensions,
                    dims
                ));
            }
        } else {
            self.dimensions = Some(vector.dimensions);
        }

        let id = self.vectors.len();
        self.uri_to_id.insert(uri.clone(), id);
        self.vectors.push((uri, vector));

        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let results = self.search_advanced(query, k, None, None)?;
        Ok(results.into_iter().map(|r| (r.uri, r.distance)).collect())
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let mut results = Vec::new();

        for (uri, vector) in &self.vectors {
            let distance = self.config.distance_metric.distance_vectors(query, vector);
            if distance <= threshold {
                results.push((uri.clone(), distance));
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        // For AdvancedVectorIndex, vectors are stored in the vectors field
        // regardless of the index type being used
        self.vectors.iter().find(|(u, _)| u == uri).map(|(_, v)| v)
    }
}

/// Index performance statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_vectors: usize,
    pub dimensions: usize,
    pub index_type: IndexType,
    pub memory_usage: usize,
}

/// Quantized vector index for memory efficiency
pub struct QuantizedVectorIndex {
    config: IndexConfig,
    quantized_vectors: Vec<Vec<u8>>,
    centroids: Vec<Vector>,
    uri_to_id: HashMap<String, usize>,
    dimensions: Option<usize>,
}

impl QuantizedVectorIndex {
    pub fn new(config: IndexConfig, num_centroids: usize) -> Self {
        Self {
            config,
            quantized_vectors: Vec::new(),
            centroids: Vec::with_capacity(num_centroids),
            uri_to_id: HashMap::new(),
            dimensions: None,
        }
    }

    /// Train quantization centroids using k-means
    pub fn train_quantization(&mut self, training_vectors: &[Vector]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(anyhow!("No training vectors provided"));
        }

        let dimensions = training_vectors[0].dimensions;
        self.dimensions = Some(dimensions);

        // Simple k-means clustering for centroids
        self.centroids = kmeans_clustering(training_vectors, self.centroids.capacity())?;

        Ok(())
    }

    fn quantize_vector(&self, vector: &Vector) -> Vec<u8> {
        let mut quantized = Vec::new();

        // Find nearest centroid for each dimension chunk
        let chunk_size = vector.dimensions / self.centroids.len().max(1);

        let vector_f32 = vector.as_f32();
        for chunk in vector_f32.chunks(chunk_size) {
            let mut best_centroid = 0u8;
            let mut best_distance = f32::INFINITY;

            for (i, centroid) in self.centroids.iter().enumerate() {
                let centroid_f32 = centroid.as_f32();
                let centroid_chunk = &centroid_f32[0..chunk.len().min(centroid.dimensions)];
                use oxirs_core::simd::SimdOps;
                let distance = f32::euclidean_distance(chunk, centroid_chunk);
                if distance < best_distance {
                    best_distance = distance;
                    best_centroid = i as u8;
                }
            }

            quantized.push(best_centroid);
        }

        quantized
    }
}

impl VectorIndex for QuantizedVectorIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        if self.centroids.is_empty() {
            return Err(anyhow!(
                "Quantization not trained. Call train_quantization first."
            ));
        }

        let id = self.quantized_vectors.len();
        self.uri_to_id.insert(uri.clone(), id);

        let quantized = self.quantize_vector(&vector);
        self.quantized_vectors.push(quantized);

        Ok(())
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        let query_quantized = self.quantize_vector(query);
        let mut results = Vec::new();

        for (uri, quantized) in self.uri_to_id.keys().zip(&self.quantized_vectors) {
            let distance = hamming_distance(&query_quantized, quantized);
            results.push((uri.clone(), distance));
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        let query_quantized = self.quantize_vector(query);
        let mut results = Vec::new();

        for (uri, quantized) in self.uri_to_id.keys().zip(&self.quantized_vectors) {
            let distance = hamming_distance(&query_quantized, quantized);
            if distance <= threshold {
                results.push((uri.clone(), distance));
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }

    fn get_vector(&self, _uri: &str) -> Option<&Vector> {
        // Quantized index doesn't store original vectors
        // Return None as we only have quantized representations
        None
    }
}

// Helper functions that don't have SIMD equivalents

fn hamming_distance(a: &[u8], b: &[u8]) -> f32 {
    a.iter().zip(b).filter(|(x, y)| x != y).count() as f32
}

// K-means clustering for quantization
fn kmeans_clustering(vectors: &[Vector], k: usize) -> Result<Vec<Vector>> {
    if vectors.is_empty() || k == 0 {
        return Ok(Vec::new());
    }

    let dimensions = vectors[0].dimensions;
    let mut centroids = Vec::with_capacity(k);

    // Initialize centroids randomly
    for i in 0..k {
        let idx = i % vectors.len();
        centroids.push(vectors[idx].clone());
    }

    // Simple k-means iterations
    for _ in 0..10 {
        let mut clusters: Vec<Vec<&Vector>> = vec![Vec::new(); k];

        // Assign vectors to nearest centroid
        for vector in vectors {
            let mut best_centroid = 0;
            let mut best_distance = f32::INFINITY;

            for (i, centroid) in centroids.iter().enumerate() {
                let vector_f32 = vector.as_f32();
                let centroid_f32 = centroid.as_f32();
                use oxirs_core::simd::SimdOps;
                let distance = f32::euclidean_distance(&vector_f32, &centroid_f32);
                if distance < best_distance {
                    best_distance = distance;
                    best_centroid = i;
                }
            }

            clusters[best_centroid].push(vector);
        }

        // Update centroids
        for (i, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let mut new_centroid = vec![0.0; dimensions];

                for vector in cluster {
                    let vector_f32 = vector.as_f32();
                    for (j, &value) in vector_f32.iter().enumerate() {
                        new_centroid[j] += value;
                    }
                }

                for value in &mut new_centroid {
                    *value /= cluster.len() as f32;
                }

                centroids[i] = Vector::new(new_centroid);
            }
        }
    }

    Ok(centroids)
}

/// Multi-index system that combines multiple index types
pub struct MultiIndex {
    indices: HashMap<String, Box<dyn VectorIndex>>,
    default_index: String,
}

impl MultiIndex {
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
            default_index: String::new(),
        }
    }

    pub fn add_index(&mut self, name: String, index: Box<dyn VectorIndex>) {
        if self.indices.is_empty() {
            self.default_index = name.clone();
        }
        self.indices.insert(name, index);
    }

    pub fn set_default(&mut self, name: &str) -> Result<()> {
        if self.indices.contains_key(name) {
            self.default_index = name.to_string();
            Ok(())
        } else {
            Err(anyhow!("Index '{}' not found", name))
        }
    }

    pub fn search_index(
        &self,
        index_name: &str,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        if let Some(index) = self.indices.get(index_name) {
            index.search_knn(query, k)
        } else {
            Err(anyhow!("Index '{}' not found", index_name))
        }
    }
}

impl Default for MultiIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex for MultiIndex {
    fn insert(&mut self, uri: String, vector: Vector) -> Result<()> {
        if let Some(index) = self.indices.get_mut(&self.default_index) {
            index.insert(uri, vector)
        } else {
            Err(anyhow!("No default index set"))
        }
    }

    fn search_knn(&self, query: &Vector, k: usize) -> Result<Vec<(String, f32)>> {
        if let Some(index) = self.indices.get(&self.default_index) {
            index.search_knn(query, k)
        } else {
            Err(anyhow!("No default index set"))
        }
    }

    fn search_threshold(&self, query: &Vector, threshold: f32) -> Result<Vec<(String, f32)>> {
        if let Some(index) = self.indices.get(&self.default_index) {
            index.search_threshold(query, threshold)
        } else {
            Err(anyhow!("No default index set"))
        }
    }

    fn get_vector(&self, uri: &str) -> Option<&Vector> {
        if let Some(index) = self.indices.get(&self.default_index) {
            index.get_vector(uri)
        } else {
            None
        }
    }
}
