//! Enhanced vector store with embedding management, advanced features, and persistence.

use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;

use crate::embeddings;
use crate::vector_index::{MemoryVectorIndex, VectorIndex};
use crate::{BatchSearchResult, Vector, VectorId, VectorStoreTrait};

/// Configuration for vector store
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VectorStoreConfig {
    pub auto_embed: bool,
    pub cache_embeddings: bool,
    pub similarity_threshold: f32,
    pub max_results: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            auto_embed: true,
            cache_embeddings: true,
            similarity_threshold: 0.7,
            max_results: 100,
        }
    }
}

/// Enhanced vector store with embedding management and advanced features
pub struct VectorStore {
    index: Box<dyn VectorIndex>,
    /// Embedding manager guarded by a mutex for interior mutability: `get_embedding`
    /// requires `&mut self` (it maintains an internal cache), but read-only search
    /// APIs (`similarity_search`, `threshold_search`, `advanced_search`) only have
    /// `&self`. Wrapping in a `Mutex` lets those paths compute *real* embeddings
    /// instead of silently falling back to a hash-based pseudo-vector.
    embedding_manager: Mutex<Option<embeddings::EmbeddingManager>>,
    config: VectorStoreConfig,
}

impl VectorStore {
    /// Create a new vector store with default memory index
    pub fn new() -> Self {
        Self {
            index: Box::new(MemoryVectorIndex::new()),
            embedding_manager: Mutex::new(None),
            config: VectorStoreConfig::default(),
        }
    }

    /// Create vector store with specific embedding strategy
    pub fn with_embedding_strategy(strategy: embeddings::EmbeddingStrategy) -> Result<Self> {
        let embedding_manager = embeddings::EmbeddingManager::new(strategy, 1000)?;

        Ok(Self {
            index: Box::new(MemoryVectorIndex::new()),
            embedding_manager: Mutex::new(Some(embedding_manager)),
            config: VectorStoreConfig::default(),
        })
    }

    /// Create vector store with custom index
    pub fn with_index(index: Box<dyn VectorIndex>) -> Self {
        Self {
            index,
            embedding_manager: Mutex::new(None),
            config: VectorStoreConfig::default(),
        }
    }

    /// Create vector store with custom index and embedding strategy
    pub fn with_index_and_embeddings(
        index: Box<dyn VectorIndex>,
        strategy: embeddings::EmbeddingStrategy,
    ) -> Result<Self> {
        let embedding_manager = embeddings::EmbeddingManager::new(strategy, 1000)?;

        Ok(Self {
            index,
            embedding_manager: Mutex::new(Some(embedding_manager)),
            config: VectorStoreConfig::default(),
        })
    }

    /// Set vector store configuration
    pub fn with_config(mut self, config: VectorStoreConfig) -> Self {
        self.config = config;
        self
    }

    /// Index a resource with automatic embedding generation
    pub fn index_resource(&mut self, uri: String, content: &str) -> Result<()> {
        let mut guard = self.embedding_manager.lock();
        if let Some(ref mut embedding_manager) = *guard {
            let embeddable_content = embeddings::EmbeddableContent::Text(content.to_string());
            let vector = embedding_manager.get_embedding(&embeddable_content)?;
            drop(guard);
            self.index.insert(uri, vector)
        } else {
            drop(guard);
            // No embedding manager configured: fall back to a hash-based vector.
            tracing::warn!(
                "index_resource: no embedding manager configured for '{}', using hash-based fallback vector",
                uri
            );
            let vector = self.generate_fallback_vector(content);
            self.index.insert(uri, vector)
        }
    }

    /// Index an RDF resource with structured content
    pub fn index_rdf_resource(
        &mut self,
        uri: String,
        label: Option<String>,
        description: Option<String>,
        properties: std::collections::HashMap<String, Vec<String>>,
    ) -> Result<()> {
        let mut guard = self.embedding_manager.lock();
        if let Some(ref mut embedding_manager) = *guard {
            let embeddable_content = embeddings::EmbeddableContent::RdfResource {
                uri: uri.clone(),
                label,
                description,
                properties,
            };
            let vector = embedding_manager.get_embedding(&embeddable_content)?;
            drop(guard);
            self.index.insert(uri, vector)
        } else {
            Err(anyhow::anyhow!(
                "Embedding manager required for RDF resource indexing"
            ))
        }
    }

    /// Index a pre-computed vector
    pub fn index_vector(&mut self, uri: String, vector: Vector) -> Result<()> {
        self.index.insert(uri, vector)
    }

    /// Compute a query vector for `text`, using the configured embedding manager
    /// when present (real embedding, including cache reuse) and only falling
    /// back to a hash-based pseudo-vector when no manager is configured at all.
    fn compute_query_vector(&self, text: &str) -> Result<Vector> {
        let mut guard = self.embedding_manager.lock();
        if let Some(ref mut embedding_manager) = *guard {
            let embeddable_content = embeddings::EmbeddableContent::Text(text.to_string());
            embedding_manager.get_embedding(&embeddable_content)
        } else {
            drop(guard);
            tracing::warn!(
                "similarity search: no embedding manager configured, using hash-based fallback vector for query"
            );
            Ok(self.generate_fallback_vector(text))
        }
    }

    /// Search for similar resources using text query
    pub fn similarity_search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        let query_vector = self.compute_query_vector(query)?;
        self.index.search_knn(&query_vector, limit)
    }

    /// Search for similar resources using a vector query
    pub fn similarity_search_vector(
        &self,
        query: &Vector,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        self.index.search_knn(query, limit)
    }

    /// Find resources within similarity threshold
    pub fn threshold_search(&self, query: &str, threshold: f32) -> Result<Vec<(String, f32)>> {
        let query_vector = self.compute_query_vector(query)?;
        self.index.search_threshold(&query_vector, threshold)
    }

    /// Advanced search with multiple options
    pub fn advanced_search(&self, options: SearchOptions) -> Result<Vec<(String, f32)>> {
        let query_vector = match options.query {
            SearchQuery::Text(text) => self.compute_query_vector(&text)?,
            SearchQuery::Vector(vector) => vector,
        };

        let results = match options.search_type {
            SearchType::KNN(k) => self.index.search_knn(&query_vector, k)?,
            SearchType::Threshold(threshold) => {
                self.index.search_threshold(&query_vector, threshold)?
            }
        };

        Ok(results)
    }

    fn generate_fallback_vector(&self, text: &str) -> Vector {
        // Simple hash-based vector generation for fallback
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut values = Vec::with_capacity(384); // Standard embedding size
        let mut seed = hash;

        for _ in 0..384 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let normalized = (seed as f32) / (u64::MAX as f32);
            values.push((normalized - 0.5) * 2.0); // Range: -1.0 to 1.0
        }

        Vector::new(values)
    }

    /// Get embedding manager statistics
    pub fn embedding_stats(&self) -> Option<(usize, usize)> {
        self.embedding_manager
            .lock()
            .as_ref()
            .map(|em| em.cache_stats())
    }

    /// Build vocabulary for TF-IDF embeddings
    pub fn build_vocabulary(&mut self, documents: &[String]) -> Result<()> {
        if let Some(ref mut embedding_manager) = *self.embedding_manager.lock() {
            embedding_manager.build_vocabulary(documents)
        } else {
            Ok(()) // No-op if no embedding manager
        }
    }

    /// Calculate similarity between two resources by their URIs
    pub fn calculate_similarity(&self, uri1: &str, uri2: &str) -> Result<f32> {
        // If the URIs are identical, return perfect similarity
        if uri1 == uri2 {
            return Ok(1.0);
        }

        // Get the vectors for both URIs
        let vector1 = self
            .index
            .get_vector(uri1)
            .ok_or_else(|| anyhow::anyhow!("Vector not found for URI: {}", uri1))?;

        let vector2 = self
            .index
            .get_vector(uri2)
            .ok_or_else(|| anyhow::anyhow!("Vector not found for URI: {}", uri2))?;

        // Calculate cosine similarity between the vectors
        vector1.cosine_similarity(vector2)
    }

    /// Get a vector by its ID (delegates to VectorIndex)
    pub fn get_vector(&self, id: &str) -> Option<&Vector> {
        self.index.get_vector(id)
    }

    /// Iterate all (id, vector) pairs stored in the underlying index.
    ///
    /// Only index types that override [`VectorIndex::iter_vectors`]
    /// (e.g. `MemoryVectorIndex`) return a non-empty list; other
    /// implementations return an empty `Vec` by default.
    pub fn iter_vectors(&self) -> Vec<(String, Vector)> {
        self.index.iter_vectors()
    }

    /// Index a vector with metadata (stub)
    pub fn index_vector_with_metadata(
        &mut self,
        uri: String,
        vector: Vector,
        _metadata: HashMap<String, String>,
    ) -> Result<()> {
        // For now, just delegate to index_vector, ignoring metadata
        // Future: Extend VectorIndex trait to support metadata
        self.index_vector(uri, vector)
    }

    /// Index a resource with metadata (stub)
    pub fn index_resource_with_metadata(
        &mut self,
        uri: String,
        content: &str,
        _metadata: HashMap<String, String>,
    ) -> Result<()> {
        // For now, just delegate to index_resource, ignoring metadata
        // Future: Store and utilize metadata
        self.index_resource(uri, content)
    }

    /// Search with additional parameters (stub)
    pub fn similarity_search_with_params(
        &self,
        query: &str,
        limit: usize,
        _params: HashMap<String, String>,
    ) -> Result<Vec<(String, f32)>> {
        // For now, just delegate to similarity_search, ignoring params
        // Future: Use params for filtering, threshold, etc.
        self.similarity_search(query, limit)
    }

    /// Vector search with additional parameters (stub)
    pub fn vector_search_with_params(
        &self,
        query: &Vector,
        limit: usize,
        _params: HashMap<String, String>,
    ) -> Result<Vec<(String, f32)>> {
        // For now, just delegate to similarity_search_vector, ignoring params
        // Future: Use params for filtering, distance metric selection, etc.
        self.similarity_search_vector(query, limit)
    }

    /// Get all vector IDs (stub)
    pub fn get_vector_ids(&self) -> Result<Vec<String>> {
        // VectorIndex trait doesn't provide this method yet
        // Future: Add to VectorIndex trait or track separately
        Ok(Vec::new())
    }

    /// Remove a vector by its URI (stub)
    pub fn remove_vector(&mut self, uri: &str) -> Result<()> {
        // Delegate to VectorIndex trait's remove_vector method
        self.index.remove_vector(uri.to_string())
    }

    /// Get store statistics (stub)
    pub fn get_statistics(&self) -> Result<HashMap<String, String>> {
        // Return basic statistics as a map
        // Future: Provide comprehensive stats from index
        let mut stats = HashMap::new();
        stats.insert("type".to_string(), "VectorStore".to_string());

        if let Some((cache_size, cache_capacity)) = self.embedding_stats() {
            stats.insert("embedding_cache_size".to_string(), cache_size.to_string());
            stats.insert(
                "embedding_cache_capacity".to_string(),
                cache_capacity.to_string(),
            );
        }

        Ok(stats)
    }

    /// Save store to disk.
    ///
    /// Creates `{path}/metadata.json` (config + vector count) and
    /// `{path}/vectors.json` (all `(id, Vector)` pairs).  The embedding
    /// manager (in-memory cache only) is **not** persisted; call
    /// `with_embedding_strategy` again after loading if needed.
    ///
    /// Only index types that report [`VectorIndex::supports_enumeration`] as
    /// `true` (e.g. `MemoryVectorIndex`, `HnswIndex`, `IvfIndex`, `PQIndex`)
    /// can be saved this way. Index types that cannot enumerate their vectors
    /// cause this to return an error rather than silently persisting an
    /// empty (and misleading) snapshot.
    pub fn save_to_disk(&self, path: &str) -> Result<()> {
        use anyhow::Context as _;

        if !self.index.supports_enumeration() {
            return Err(anyhow::anyhow!(
                "save_to_disk: the configured index type does not support full \
                 vector enumeration (VectorIndex::supports_enumeration() == false); \
                 refusing to write a silently-empty vectors.json. Use an index type \
                 that implements iter_vectors/supports_enumeration (e.g. \
                 MemoryVectorIndex, HnswIndex, IvfIndex, PQIndex)."
            ));
        }

        std::fs::create_dir_all(path)
            .with_context(|| format!("Failed to create directory: {}", path))?;

        // --- metadata ---
        let vectors = self.index.iter_vectors();
        let metadata = serde_json::json!({
            "config": self.config,
            "vector_count": vectors.len(),
            "index_type": "memory",
        });
        let metadata_path = std::path::Path::new(path).join("metadata.json");
        let metadata_str = serde_json::to_string_pretty(&metadata)
            .with_context(|| "Failed to serialize VectorStore metadata")?;
        std::fs::write(&metadata_path, metadata_str)
            .with_context(|| format!("Failed to write {}", metadata_path.display()))?;

        // --- vectors ---
        let vectors_path = std::path::Path::new(path).join("vectors.json");
        let vectors_str = serde_json::to_string_pretty(&vectors)
            .with_context(|| "Failed to serialize VectorStore vectors")?;
        std::fs::write(&vectors_path, vectors_str)
            .with_context(|| format!("Failed to write {}", vectors_path.display()))?;

        Ok(())
    }

    /// Load a store that was previously saved with [`VectorStore::save_to_disk`].
    ///
    /// Reconstructs a `VectorStore` backed by a fresh `MemoryVectorIndex` and
    /// re-inserts all vectors from the saved snapshot.  The embedding manager
    /// is not restored; create a new one with `with_embedding_strategy` if
    /// needed.
    pub fn load_from_disk(path: &str) -> Result<Self> {
        use anyhow::Context as _;

        // --- read metadata ---
        let metadata_path = std::path::Path::new(path).join("metadata.json");
        let metadata_str = std::fs::read_to_string(&metadata_path)
            .with_context(|| format!("Failed to read {}", metadata_path.display()))?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str)
            .with_context(|| "Failed to parse VectorStore metadata")?;

        let config: VectorStoreConfig = serde_json::from_value(metadata["config"].clone())
            .with_context(|| "Failed to deserialize VectorStoreConfig from metadata")?;

        // --- read vectors ---
        let vectors_path = std::path::Path::new(path).join("vectors.json");
        let vectors_str = std::fs::read_to_string(&vectors_path)
            .with_context(|| format!("Failed to read {}", vectors_path.display()))?;
        let entries: Vec<(String, Vector)> = serde_json::from_str(&vectors_str)
            .with_context(|| "Failed to deserialize VectorStore vectors")?;

        // --- reconstruct ---
        let mut store = Self {
            index: Box::new(MemoryVectorIndex::new()),
            embedding_manager: Mutex::new(None),
            config,
        };

        for (id, vector) in entries {
            store
                .index
                .insert(id.clone(), vector)
                .with_context(|| format!("Failed to re-insert vector '{}'", id))?;
        }

        Ok(store)
    }

    /// Optimize the underlying index (stub)
    pub fn optimize_index(&mut self) -> Result<()> {
        // Stub implementation - optimization not yet implemented
        // Future: Trigger index compaction, rebalancing, etc.
        Ok(())
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStoreTrait for VectorStore {
    fn insert_vector(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        self.index.insert(id, vector)
    }

    fn add_vector(&mut self, vector: Vector) -> Result<VectorId> {
        // Generate a unique ID for the vector
        let id = format!("vec_{}", uuid::Uuid::new_v4());
        self.index.insert(id.clone(), vector)?;
        Ok(id)
    }

    fn get_vector(&self, id: &VectorId) -> Result<Option<Vector>> {
        Ok(self.index.get_vector(id).cloned())
    }

    fn get_all_vector_ids(&self) -> Result<Vec<VectorId>> {
        // Delegates to the same iter_vectors() enumeration used by
        // save_to_disk/VectorStore::iter_vectors, so this only returns real
        // IDs for index types that support enumeration; other index types
        // return an empty list (consistent with `len()` below).
        Ok(self
            .index
            .iter_vectors()
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    fn search_similar(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>> {
        self.index.search_knn(query, k)
    }

    fn remove_vector(&mut self, id: &VectorId) -> Result<bool> {
        // Delegate to the underlying VectorIndex::remove_vector, matching the
        // crate's own inherent `VectorStore::remove_vector` method.
        match self.index.remove_vector(id.clone()) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn len(&self) -> usize {
        // Consistent with get_all_vector_ids(): counts vectors via the same
        // enumeration path so both agree for enumerable index types.
        self.index.iter_vectors().len()
    }
}

/// Search query types
#[derive(Debug, Clone)]
pub enum SearchQuery {
    Text(String),
    Vector(Vector),
}

/// Search operation types
#[derive(Debug, Clone)]
pub enum SearchType {
    KNN(usize),
    Threshold(f32),
}

/// Advanced search options
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: SearchQuery,
    pub search_type: SearchType,
}

/// Vector operation results with enhanced metadata
#[derive(Debug, Clone)]
pub struct VectorOperationResult {
    pub uri: String,
    pub similarity: f32,
    pub vector: Option<Vector>,
    pub metadata: Option<std::collections::HashMap<String, String>>,
    pub rank: usize,
}

/// Document batch processing utilities
pub struct DocumentBatchProcessor;

impl DocumentBatchProcessor {
    /// Process multiple documents in batch for efficient indexing
    pub fn batch_index(
        store: &mut VectorStore,
        documents: &[(String, String)], // (uri, content) pairs
    ) -> Result<Vec<Result<()>>> {
        let mut results = Vec::new();

        for (uri, content) in documents {
            let result = store.index_resource(uri.clone(), content);
            results.push(result);
        }

        Ok(results)
    }

    /// Process multiple queries in batch
    pub fn batch_search(
        store: &VectorStore,
        queries: &[String],
        limit: usize,
    ) -> Result<BatchSearchResult> {
        let mut results = Vec::new();

        for query in queries {
            let result = store.similarity_search(query, limit);
            results.push(result);
        }

        Ok(results)
    }
}
