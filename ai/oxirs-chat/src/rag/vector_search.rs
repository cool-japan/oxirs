//! Vector-based semantic search and document management
//!
//! Provides enhanced vector indexing and search capabilities for RAG retrieval.

use super::*;

/// Enhanced vector index with advanced search capabilities
pub struct EnhancedVectorIndex {
    embedding_manager: EmbeddingManager,
    index: AdvancedVectorIndex,
    document_mapping: HashMap<String, RagDocument>,
    triple_index: HashMap<String, Triple>,
}

impl EnhancedVectorIndex {
    /// Create a new enhanced vector index
    pub async fn new() -> Result<Self> {
        let embedding_manager = EmbeddingManager::new();
        let index_config = IndexConfig {
            index_type: IndexType::HNSW,
            distance_metric: DistanceMetric::Cosine,
            dimensions: 384, // Default for sentence transformers
            ..Default::default()
        };
        let index = AdvancedVectorIndex::new(index_config)?;

        Ok(Self {
            embedding_manager,
            index,
            document_mapping: HashMap::new(),
            triple_index: HashMap::new(),
        })
    }

    /// Add a document to the index
    pub async fn add_document(
        &mut self,
        id: String,
        content: String,
        triple: Option<Triple>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        // Generate embedding for the content
        let vector = self
            .embedding_manager
            .get_embedding(&EmbeddableContent::Text(content.clone()))?;

        // Add to vector index
        self.index.insert(id.clone(), vector.clone())?;

        // Store document mapping
        let document = RagDocument {
            id: id.clone(),
            content,
            triple: triple.clone(),
            metadata,
            embedding: Some(vector.as_f32()),
        };
        self.document_mapping.insert(id.clone(), document);

        // Store triple mapping if provided
        if let Some(triple) = triple {
            self.triple_index.insert(id, triple);
        }

        Ok(())
    }

    /// Search for similar documents using semantic similarity
    pub async fn search(&mut self, query: &str, limit: usize) -> Result<Vec<SearchDocument>> {
        // Generate embedding for query
        let query_vector = self
            .embedding_manager
            .get_embedding(&EmbeddableContent::Text(query.to_string()))?;

        // Search the vector index
        let search_results: Vec<VecSearchResult> =
            self.index.search(&query_vector.as_f32(), limit)?;

        // Convert to SearchDocument
        let mut documents = Vec::new();
        for result in search_results {
            if let Some(document) = self.document_mapping.get(&result.uri) {
                let search_doc = SearchDocument {
                    document: document.triple.clone().unwrap_or_else(|| {
                        // Create a default triple if none exists
                        Triple::new(
                            Subject::NamedNode(NamedNode::new_unchecked(&result.uri)),
                            NamedNode::new_unchecked("http://www.w3.org/2000/01/rdf-schema#label"),
                            Object::Literal(document.content.clone().into()),
                        )
                    }),
                    score: result.distance,
                };
                documents.push(search_doc);
            }
        }

        Ok(documents)
    }

    /// Get the number of documents in the index
    pub fn len(&self) -> usize {
        self.document_mapping.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.document_mapping.is_empty()
    }

    /// Get all indexed triples
    pub fn get_triples(&self) -> Vec<Triple> {
        self.triple_index.values().cloned().collect()
    }
}

/// Search document result
pub struct SearchDocument {
    pub document: Triple,
    pub score: f32,
}

/// RAG document with metadata and embeddings
#[derive(Debug, Clone)]
pub struct RagDocument {
    pub id: String,
    pub content: String,
    pub triple: Option<Triple>,
    pub metadata: HashMap<String, String>,
    pub embedding: Option<Vec<f32>>,
}

/// RAG index that wraps enhanced vector index
pub struct RagIndex {
    enhanced_index: EnhancedVectorIndex,
}

impl RagIndex {
    /// Create a new RAG index
    pub async fn new() -> Result<Self> {
        Ok(Self {
            enhanced_index: EnhancedVectorIndex::new().await?,
        })
    }

    /// Add document to index
    pub async fn add_document(
        &mut self,
        id: String,
        content: String,
        triple: Option<Triple>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        self.enhanced_index
            .add_document(id, content, triple, metadata)
            .await
    }

    /// Search for documents
    pub async fn search(&mut self, query: &str, limit: usize) -> Result<Vec<SearchDocument>> {
        self.enhanced_index.search(query, limit).await
    }

    /// Get number of documents
    pub fn len(&self) -> usize {
        self.enhanced_index.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.enhanced_index.is_empty()
    }
}
