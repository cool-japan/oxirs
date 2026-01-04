//! RDF term support integration with oxirs-core
//!
//! This module provides seamless integration between oxirs-vec's vector operations
//! and oxirs-core's RDF term system, enabling semantic vector search on RDF data.

use crate::{similarity::SimilarityMetric, Vector, VectorId, VectorStoreTrait};
use anyhow::{anyhow, Result};
use oxirs_core::model::{GraphName, Literal, NamedNode, Term};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Configuration for RDF-vector integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfVectorConfig {
    /// Enable automatic URI decomposition for embeddings
    pub uri_decomposition: bool,
    /// Include literal types in embeddings
    pub include_literal_types: bool,
    /// Enable graph context awareness
    pub graph_context: bool,
    /// Namespace prefix handling
    pub namespace_aware: bool,
    /// Default similarity metric for RDF term comparisons
    pub default_metric: SimilarityMetric,
    /// Cache size for term-to-vector mappings
    pub cache_size: usize,
}

impl Default for RdfVectorConfig {
    fn default() -> Self {
        Self {
            uri_decomposition: true,
            include_literal_types: true,
            graph_context: true,
            namespace_aware: true,
            default_metric: SimilarityMetric::Cosine,
            cache_size: 10000,
        }
    }
}

/// Mapping between RDF terms and vector identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfTermMapping {
    /// Original RDF term
    pub term: Term,
    /// Associated vector identifier
    pub vector_id: VectorId,
    /// Graph context (if applicable)
    pub graph_context: Option<GraphName>,
    /// Term metadata for enhanced processing
    pub metadata: RdfTermMetadata,
}

/// Metadata for RDF terms to enhance vector processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfTermMetadata {
    /// Term type for specialized processing
    pub term_type: RdfTermType,
    /// Namespace information
    pub namespace: Option<String>,
    /// Local name component
    pub local_name: Option<String>,
    /// Literal datatype (if applicable)
    pub datatype: Option<NamedNode>,
    /// Language tag (if applicable)
    pub language: Option<String>,
    /// Term complexity score for weighting
    pub complexity_score: f32,
}

/// RDF term type enumeration for processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RdfTermType {
    NamedNode,
    BlankNode,
    Literal,
    Variable,
    QuotedTriple,
}

/// Result of RDF-aware vector search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfVectorSearchResult {
    /// Matching RDF term
    pub term: Term,
    /// Similarity score
    pub score: f32,
    /// Vector identifier
    pub vector_id: VectorId,
    /// Graph context
    pub graph_context: Option<GraphName>,
    /// Search metadata
    pub metadata: SearchMetadata,
}

/// Search metadata for RDF vector results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetadata {
    /// Search algorithm used
    pub algorithm: String,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Term matching confidence
    pub confidence: f32,
    /// Explanation of result relevance
    pub explanation: Option<String>,
}

/// RDF-Vector integration engine
pub struct RdfVectorIntegration {
    /// Configuration
    config: RdfVectorConfig,
    /// Term to vector mappings
    term_mappings: Arc<RwLock<HashMap<TermHash, RdfTermMapping>>>,
    /// Vector to term reverse mappings
    vector_mappings: Arc<RwLock<HashMap<VectorId, RdfTermMapping>>>,
    /// Graph context cache
    graph_cache: Arc<RwLock<HashMap<GraphName, HashSet<VectorId>>>>,
    /// Namespace registry
    namespace_registry: Arc<RwLock<HashMap<String, String>>>,
    /// Vector store reference
    vector_store: Arc<RwLock<dyn VectorStoreTrait>>,
}

/// Hash wrapper for RDF terms to enable HashMap keys
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TermHash(u64);

impl TermHash {
    fn from_term(term: &Term) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        match term {
            Term::NamedNode(node) => {
                "NamedNode".hash(&mut hasher);
                node.as_str().hash(&mut hasher);
            }
            Term::BlankNode(node) => {
                "BlankNode".hash(&mut hasher);
                node.as_str().hash(&mut hasher);
            }
            Term::Literal(literal) => {
                "Literal".hash(&mut hasher);
                literal.value().hash(&mut hasher);
                if let Some(lang) = literal.language() {
                    lang.hash(&mut hasher);
                }
                literal.datatype().as_str().hash(&mut hasher);
            }
            Term::Variable(var) => {
                "Variable".hash(&mut hasher);
                var.as_str().hash(&mut hasher);
            }
            Term::QuotedTriple(_) => {
                "QuotedTriple".hash(&mut hasher);
                // Simplified hash for quoted triples
                "quoted_triple".hash(&mut hasher);
            }
        }

        TermHash(hasher.finish())
    }
}

impl RdfVectorIntegration {
    /// Create a new RDF-vector integration instance
    pub fn new(config: RdfVectorConfig, vector_store: Arc<RwLock<dyn VectorStoreTrait>>) -> Self {
        Self {
            config,
            term_mappings: Arc::new(RwLock::new(HashMap::new())),
            vector_mappings: Arc::new(RwLock::new(HashMap::new())),
            graph_cache: Arc::new(RwLock::new(HashMap::new())),
            namespace_registry: Arc::new(RwLock::new(HashMap::new())),
            vector_store,
        }
    }

    /// Register an RDF term with vector representation
    pub fn register_term(
        &self,
        term: Term,
        vector: Vector,
        graph_context: Option<GraphName>,
    ) -> Result<VectorId> {
        let vector_id = self
            .vector_store
            .write()
            .expect("lock poisoned")
            .add_vector(vector)?;
        let metadata = self.extract_term_metadata(&term)?;

        let mapping = RdfTermMapping {
            term: term.clone(),
            vector_id: vector_id.clone(),
            graph_context: graph_context.clone(),
            metadata,
        };

        let term_hash = TermHash::from_term(&term);

        // Update mappings
        {
            let mut term_mappings = self.term_mappings.write().expect("lock poisoned");
            term_mappings.insert(term_hash, mapping.clone());
        }

        {
            let mut vector_mappings = self.vector_mappings.write().expect("lock poisoned");
            vector_mappings.insert(vector_id.clone(), mapping);
        }

        // Update graph cache if applicable
        if let Some(graph) = graph_context {
            let mut graph_cache = self.graph_cache.write().expect("lock poisoned");
            graph_cache
                .entry(graph)
                .or_default()
                .insert(vector_id.clone());
        }

        Ok(vector_id)
    }

    /// Find similar RDF terms using vector similarity
    pub fn find_similar_terms(
        &self,
        query_term: &Term,
        limit: usize,
        threshold: Option<f32>,
        graph_context: Option<&GraphName>,
    ) -> Result<Vec<RdfVectorSearchResult>> {
        let start_time = std::time::Instant::now();

        // Get vector for query term
        let query_vector_id = self
            .get_vector_id(query_term)?
            .ok_or_else(|| anyhow!("Query term not found in vector store"))?;

        let query_vector = self
            .vector_store
            .read()
            .expect("lock poisoned")
            .get_vector(&query_vector_id)?
            .ok_or_else(|| anyhow!("Query vector not found"))?;

        // Filter by graph context if specified
        let candidate_vectors = if let Some(graph) = graph_context {
            let graph_cache = self.graph_cache.read().expect("lock poisoned");
            graph_cache
                .get(graph)
                .map(|set| set.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            // Use all vectors if no graph context specified
            self.vector_store
                .read()
                .expect("lock poisoned")
                .get_all_vector_ids()?
        };

        // Perform similarity search
        let mut results = Vec::new();
        for vector_id in candidate_vectors {
            if *vector_id == query_vector_id {
                continue; // Skip self
            }

            if let Ok(Some(vector)) = self
                .vector_store
                .read()
                .expect("lock poisoned")
                .get_vector(&vector_id)
            {
                let similarity = self.config.default_metric.compute(&query_vector, &vector)?;

                // Apply threshold filtering
                if let Some(thresh) = threshold {
                    if similarity < thresh {
                        continue;
                    }
                }

                // Get term mapping
                let vector_mappings = self.vector_mappings.read().expect("lock poisoned");
                if let Some(mapping) = vector_mappings.get(&vector_id) {
                    let processing_time = start_time.elapsed().as_micros() as u64;

                    results.push(RdfVectorSearchResult {
                        term: mapping.term.clone(),
                        score: similarity,
                        vector_id: vector_id.clone(),
                        graph_context: mapping.graph_context.clone(),
                        metadata: SearchMetadata {
                            algorithm: "vector_similarity".to_string(),
                            processing_time_us: processing_time,
                            confidence: self.calculate_confidence(similarity, &mapping.metadata),
                            explanation: self.generate_explanation(&mapping.metadata, similarity),
                        },
                    });
                }
            }
        }

        // Sort by similarity score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limit
        results.truncate(limit);

        Ok(results)
    }

    /// Search for terms by text content with RDF-aware processing
    pub fn search_by_text(
        &self,
        query_text: &str,
        limit: usize,
        threshold: Option<f32>,
        graph_context: Option<&GraphName>,
    ) -> Result<Vec<RdfVectorSearchResult>> {
        // Create a temporary literal term for text search
        let literal = Literal::new_simple_literal(query_text);
        let _query_term = Term::Literal(literal);

        // For text search, we would typically generate an embedding
        // This is a simplified version - in practice, you'd use an embedding model
        let query_vector = self.generate_text_embedding(query_text)?;

        // Register temporary term (optional - for caching)
        let temp_vector_id = self
            .vector_store
            .write()
            .expect("lock poisoned")
            .add_vector(query_vector.clone())?;

        // Perform similarity search against all terms
        let candidate_vectors = if let Some(graph) = graph_context {
            let graph_cache = self.graph_cache.read().expect("lock poisoned");
            graph_cache
                .get(graph)
                .map(|set| set.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        } else {
            self.vector_store
                .read()
                .expect("lock poisoned")
                .get_all_vector_ids()?
        };

        let mut results = Vec::new();
        let start_time = std::time::Instant::now();

        for vector_id in candidate_vectors {
            if let Ok(Some(vector)) = self
                .vector_store
                .read()
                .expect("lock poisoned")
                .get_vector(&vector_id)
            {
                let similarity = self.config.default_metric.compute(&query_vector, &vector)?;

                if let Some(thresh) = threshold {
                    if similarity < thresh {
                        continue;
                    }
                }

                let vector_mappings = self.vector_mappings.read().expect("lock poisoned");
                if let Some(mapping) = vector_mappings.get(&vector_id) {
                    let processing_time = start_time.elapsed().as_micros() as u64;

                    results.push(RdfVectorSearchResult {
                        term: mapping.term.clone(),
                        score: similarity,
                        vector_id: vector_id.clone(),
                        graph_context: mapping.graph_context.clone(),
                        metadata: SearchMetadata {
                            algorithm: "text_similarity".to_string(),
                            processing_time_us: processing_time,
                            confidence: self.calculate_confidence(similarity, &mapping.metadata),
                            explanation: Some(format!("Text similarity match: '{query_text}'")),
                        },
                    });
                }
            }
        }

        // Clean up temporary vector
        let _ = self
            .vector_store
            .write()
            .expect("lock poisoned")
            .remove_vector(&temp_vector_id);

        // Sort and limit results
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    /// Get vector ID for an RDF term
    pub fn get_vector_id(&self, term: &Term) -> Result<Option<VectorId>> {
        let term_hash = TermHash::from_term(term);
        let term_mappings = self.term_mappings.read().expect("lock poisoned");
        Ok(term_mappings
            .get(&term_hash)
            .map(|mapping| mapping.vector_id.clone()))
    }

    /// Get RDF term for a vector ID
    pub fn get_term(&self, vector_id: VectorId) -> Result<Option<Term>> {
        let vector_mappings = self.vector_mappings.read().expect("lock poisoned");
        Ok(vector_mappings
            .get(&vector_id)
            .map(|mapping| mapping.term.clone()))
    }

    /// Register a namespace prefix
    pub fn register_namespace(&self, prefix: String, uri: String) -> Result<()> {
        let mut registry = self.namespace_registry.write().expect("lock poisoned");
        registry.insert(prefix, uri);
        Ok(())
    }

    /// Extract metadata from RDF term
    fn extract_term_metadata(&self, term: &Term) -> Result<RdfTermMetadata> {
        match term {
            Term::NamedNode(node) => {
                let uri = node.as_str();
                let (namespace, local_name) = self.split_uri(uri);

                Ok(RdfTermMetadata {
                    term_type: RdfTermType::NamedNode,
                    namespace,
                    local_name,
                    datatype: None,
                    language: None,
                    complexity_score: self.calculate_uri_complexity(uri),
                })
            }
            Term::BlankNode(_) => {
                Ok(RdfTermMetadata {
                    term_type: RdfTermType::BlankNode,
                    namespace: None,
                    local_name: None,
                    datatype: None,
                    language: None,
                    complexity_score: 0.5, // Blank nodes have medium complexity
                })
            }
            Term::Literal(literal) => Ok(RdfTermMetadata {
                term_type: RdfTermType::Literal,
                namespace: None,
                local_name: None,
                datatype: Some(literal.datatype().into()),
                language: literal.language().map(|s| s.to_string()),
                complexity_score: self.calculate_literal_complexity(literal),
            }),
            Term::Variable(_) => {
                Ok(RdfTermMetadata {
                    term_type: RdfTermType::Variable,
                    namespace: None,
                    local_name: None,
                    datatype: None,
                    language: None,
                    complexity_score: 0.3, // Variables have low complexity
                })
            }
            Term::QuotedTriple(_) => {
                Ok(RdfTermMetadata {
                    term_type: RdfTermType::QuotedTriple,
                    namespace: None,
                    local_name: None,
                    datatype: None,
                    language: None,
                    complexity_score: 1.0, // Quoted triples have high complexity
                })
            }
        }
    }

    /// Split URI into namespace and local name
    fn split_uri(&self, uri: &str) -> (Option<String>, Option<String>) {
        // Simple URI splitting logic - can be enhanced
        if let Some(pos) = uri.rfind(&['#', '/'][..]) {
            let namespace = uri[..pos + 1].to_string();
            let local_name = uri[pos + 1..].to_string();
            (Some(namespace), Some(local_name))
        } else {
            (None, Some(uri.to_string()))
        }
    }

    /// Calculate URI complexity score
    fn calculate_uri_complexity(&self, uri: &str) -> f32 {
        let length_factor = (uri.len() as f32 / 100.0).min(1.0);
        let segment_count = uri.matches(&['/', '#'][..]).count() as f32 / 10.0;
        let query_params = if uri.contains('?') { 0.2 } else { 0.0 };

        (length_factor + segment_count + query_params).min(1.0)
    }

    /// Calculate literal complexity score
    fn calculate_literal_complexity(&self, literal: &Literal) -> f32 {
        let value_length = literal.value().len() as f32 / 200.0;
        let datatype_complexity =
            if literal.datatype().as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                0.3
            } else {
                0.7
            };
        let language_bonus = if literal.language().is_some() {
            0.2
        } else {
            0.0
        };

        (value_length + datatype_complexity + language_bonus).min(1.0)
    }

    /// Calculate confidence score for search results
    fn calculate_confidence(&self, similarity: f32, metadata: &RdfTermMetadata) -> f32 {
        let base_confidence = similarity;
        let complexity_bonus = metadata.complexity_score * 0.1;
        let type_bonus = match metadata.term_type {
            RdfTermType::NamedNode => 0.1,
            RdfTermType::Literal => 0.05,
            RdfTermType::BlankNode => 0.02,
            RdfTermType::Variable => 0.01,
            RdfTermType::QuotedTriple => 0.15,
        };

        (base_confidence + complexity_bonus + type_bonus).min(1.0)
    }

    /// Generate explanation for search results
    fn generate_explanation(&self, metadata: &RdfTermMetadata, similarity: f32) -> Option<String> {
        let term_type_str = match metadata.term_type {
            RdfTermType::NamedNode => "Named Node",
            RdfTermType::BlankNode => "Blank Node",
            RdfTermType::Literal => "Literal",
            RdfTermType::Variable => "Variable",
            RdfTermType::QuotedTriple => "Quoted Triple",
        };

        let mut explanation = format!(
            "{} with {:.2}% similarity",
            term_type_str,
            similarity * 100.0
        );

        if let Some(namespace) = &metadata.namespace {
            explanation.push_str(&format!(", namespace: {namespace}"));
        }

        if let Some(language) = &metadata.language {
            explanation.push_str(&format!(", language: {language}"));
        }

        Some(explanation)
    }

    /// Generate text embedding (placeholder implementation)
    fn generate_text_embedding(&self, text: &str) -> Result<Vector> {
        // This is a simplified implementation
        // In production, you would use a proper embedding model
        let words: Vec<&str> = text.split_whitespace().collect();
        let dimension = 384; // Standard sentence transformer dimension

        let mut vector_data = vec![0.0; dimension];

        // Simple word-based embedding generation
        for word in words.iter() {
            let word_hash = {
                use std::collections::hash_map::DefaultHasher;
                let mut hasher = DefaultHasher::new();
                word.hash(&mut hasher);
                hasher.finish()
            };

            // Distribute word influence across vector dimensions
            for j in 0..dimension {
                let index = (word_hash as usize + j) % dimension;
                vector_data[index] += 1.0 / (words.len() as f32);
            }
        }

        // Normalize vector
        let norm: f32 = vector_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector_data {
                *value /= norm;
            }
        }

        Ok(Vector::new(vector_data))
    }

    /// Get statistics about the RDF-vector integration
    pub fn get_statistics(&self) -> RdfIntegrationStats {
        let term_mappings = self.term_mappings.read().expect("lock poisoned");
        let graph_cache = self.graph_cache.read().expect("lock poisoned");
        let namespace_registry = self.namespace_registry.read().expect("lock poisoned");

        let mut type_counts = HashMap::new();
        for mapping in term_mappings.values() {
            *type_counts.entry(mapping.metadata.term_type).or_insert(0) += 1;
        }

        RdfIntegrationStats {
            total_terms: term_mappings.len(),
            total_graphs: graph_cache.len(),
            total_namespaces: namespace_registry.len(),
            type_distribution: type_counts,
            cache_hit_ratio: 0.95, // Placeholder
        }
    }
}

/// Statistics for RDF-vector integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfIntegrationStats {
    pub total_terms: usize,
    pub total_graphs: usize,
    pub total_namespaces: usize,
    pub type_distribution: HashMap<RdfTermType, usize>,
    pub cache_hit_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorStore;
    use oxirs_core::model::{NamedNode, Term};

    #[test]
    fn test_rdf_term_registration() {
        let config = RdfVectorConfig::default();
        let vector_store = Arc::new(RwLock::new(VectorStore::new()));
        let integration = RdfVectorIntegration::new(config, vector_store);

        let named_node = NamedNode::new("http://example.org/person").unwrap();
        let term = Term::NamedNode(named_node);
        let vector = Vector::new(vec![1.0, 0.0, 0.0]);

        let vector_id = integration
            .register_term(term.clone(), vector, None)
            .unwrap();

        assert!(integration.get_vector_id(&term).unwrap().is_some());
        assert_eq!(
            integration.get_vector_id(&term).unwrap().unwrap(),
            vector_id
        );
    }

    #[test]
    fn test_uri_splitting() {
        let config = RdfVectorConfig::default();
        let vector_store = Arc::new(RwLock::new(VectorStore::new()));
        let integration = RdfVectorIntegration::new(config, vector_store);

        let (namespace, local_name) = integration.split_uri("http://example.org/ontology#Person");
        assert_eq!(namespace, Some("http://example.org/ontology#".to_string()));
        assert_eq!(local_name, Some("Person".to_string()));
    }

    #[test]
    fn test_metadata_extraction() {
        let config = RdfVectorConfig::default();
        let vector_store = Arc::new(RwLock::new(VectorStore::new()));
        let integration = RdfVectorIntegration::new(config, vector_store);

        let literal = Literal::new_language_tagged_literal("Hello", "en").unwrap();
        let term = Term::Literal(literal);

        let metadata = integration.extract_term_metadata(&term).unwrap();
        assert_eq!(metadata.term_type, RdfTermType::Literal);
        assert_eq!(metadata.language, Some("en".to_string()));
    }
}
