//! Hybrid search manager coordinating keyword and semantic search

use super::config::{HybridSearchConfig, KeywordAlgorithm, SearchMode};
use super::fusion::RankFusion;
use super::keyword::{Bm25Scorer, KeywordSearcher, TfidfScorer};
use super::query_expansion::QueryExpander;
use super::types::{DocumentScore, HybridQuery, HybridResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Hybrid search manager
pub struct HybridSearchManager {
    /// Configuration
    config: HybridSearchConfig,
    /// Keyword searcher
    keyword_searcher: Arc<RwLock<Box<dyn KeywordSearcher>>>,
    /// Rank fusion
    fusion: RankFusion,
    /// Query expander
    query_expander: QueryExpander,
    /// Document vectors: doc_id -> vector
    doc_vectors: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    /// Document metadata
    doc_metadata: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
}

impl HybridSearchManager {
    /// Create a new hybrid search manager
    pub fn new(config: HybridSearchConfig) -> anyhow::Result<Self> {
        config.validate()?;

        let keyword_searcher: Box<dyn KeywordSearcher> = match config.keyword_algorithm {
            KeywordAlgorithm::Bm25 => Box::new(Bm25Scorer::new()),
            KeywordAlgorithm::Tfidf => Box::new(TfidfScorer::new()),
            KeywordAlgorithm::Combined => Box::new(Bm25Scorer::new()), // Default to BM25
        };

        let fusion = RankFusion::new(config.fusion_strategy);
        let query_expander = QueryExpander::new(config.max_expanded_terms);

        Ok(Self {
            config,
            keyword_searcher: Arc::new(RwLock::new(keyword_searcher)),
            fusion,
            query_expander,
            doc_vectors: Arc::new(RwLock::new(HashMap::new())),
            doc_metadata: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add a document to the index
    pub fn add_document(
        &self,
        doc_id: &str,
        content: &str,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> anyhow::Result<()> {
        // Add to keyword index
        self.keyword_searcher
            .write()
            .unwrap()
            .add_document(doc_id, content)?;

        // Store vector
        self.doc_vectors
            .write()
            .unwrap()
            .insert(doc_id.to_string(), vector);

        // Store metadata
        self.doc_metadata
            .write()
            .unwrap()
            .insert(doc_id.to_string(), metadata);

        Ok(())
    }

    /// Search with hybrid mode
    pub fn search(&self, query: HybridQuery) -> anyhow::Result<Vec<HybridResult>> {
        match self.config.mode {
            SearchMode::KeywordOnly => self.keyword_search(&query),
            SearchMode::SemanticOnly => self.semantic_search(&query),
            SearchMode::Hybrid => self.hybrid_search(&query),
            SearchMode::Adaptive => self.adaptive_search(&query),
        }
    }

    /// Keyword-only search
    fn keyword_search(&self, query: &HybridQuery) -> anyhow::Result<Vec<HybridResult>> {
        let query_text = if self.config.enable_query_expansion {
            let expanded = self.query_expander.expand(&query.query_text);
            expanded.join(" ")
        } else {
            query.query_text.clone()
        };

        let keyword_results = self
            .keyword_searcher
            .read()
            .unwrap()
            .search(&query_text, query.top_k)?;

        let results: Vec<HybridResult> = keyword_results
            .into_iter()
            .filter(|m| m.score >= self.config.min_keyword_score)
            .map(|m| {
                let metadata = self
                    .doc_metadata
                    .read()
                    .unwrap()
                    .get(&m.doc_id)
                    .cloned()
                    .unwrap_or_default();

                HybridResult {
                    doc_id: m.doc_id,
                    score: m.score,
                    score_breakdown: super::types::ScoreBreakdown {
                        keyword_score: m.score,
                        semantic_score: 0.0,
                        recency_score: 0.0,
                        keyword_rank: Some(0),
                        semantic_rank: None,
                    },
                    metadata,
                }
            })
            .collect();

        Ok(results)
    }

    /// Semantic-only search
    fn semantic_search(&self, query: &HybridQuery) -> anyhow::Result<Vec<HybridResult>> {
        let query_vector = query
            .query_vector
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Query vector required for semantic search"))?;

        let doc_vectors = self.doc_vectors.read().unwrap();
        let mut semantic_results: Vec<DocumentScore> = doc_vectors
            .iter()
            .map(|(doc_id, doc_vec)| {
                let similarity = Self::cosine_similarity(query_vector, doc_vec);
                DocumentScore {
                    doc_id: doc_id.clone(),
                    score: similarity,
                    rank: 0,
                }
            })
            .collect();

        semantic_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        semantic_results.truncate(query.top_k);

        // Update ranks
        for (rank, result) in semantic_results.iter_mut().enumerate() {
            result.rank = rank;
        }

        let results: Vec<HybridResult> = semantic_results
            .into_iter()
            .filter(|r| r.score >= self.config.min_semantic_score)
            .map(|r| {
                let metadata = self
                    .doc_metadata
                    .read()
                    .unwrap()
                    .get(&r.doc_id)
                    .cloned()
                    .unwrap_or_default();

                HybridResult {
                    doc_id: r.doc_id,
                    score: r.score,
                    score_breakdown: super::types::ScoreBreakdown {
                        keyword_score: 0.0,
                        semantic_score: r.score,
                        recency_score: 0.0,
                        keyword_rank: None,
                        semantic_rank: Some(r.rank),
                    },
                    metadata,
                }
            })
            .collect();

        Ok(results)
    }

    /// Full hybrid search
    fn hybrid_search(&self, query: &HybridQuery) -> anyhow::Result<Vec<HybridResult>> {
        // Get keyword results
        let query_text = if self.config.enable_query_expansion {
            let expanded = self.query_expander.expand(&query.query_text);
            expanded.join(" ")
        } else {
            query.query_text.clone()
        };

        let keyword_matches = self
            .keyword_searcher
            .read()
            .unwrap()
            .search(&query_text, query.top_k * 2)?; // Get more candidates

        let keyword_results: Vec<DocumentScore> = keyword_matches
            .into_iter()
            .enumerate()
            .filter(|(_, m)| m.score >= self.config.min_keyword_score)
            .map(|(rank, m)| DocumentScore {
                doc_id: m.doc_id,
                score: m.score,
                rank,
            })
            .collect();

        // Get semantic results
        let query_vector = query
            .query_vector
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Query vector required for hybrid search"))?;

        let doc_vectors = self.doc_vectors.read().unwrap();
        let mut semantic_results: Vec<DocumentScore> = doc_vectors
            .iter()
            .map(|(doc_id, doc_vec)| {
                let similarity = Self::cosine_similarity(query_vector, doc_vec);
                DocumentScore {
                    doc_id: doc_id.clone(),
                    score: similarity,
                    rank: 0,
                }
            })
            .filter(|r| r.score >= self.config.min_semantic_score)
            .collect();

        semantic_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        semantic_results.truncate(query.top_k * 2);

        for (rank, result) in semantic_results.iter_mut().enumerate() {
            result.rank = rank;
        }

        // Fuse results
        let mut fused_results = self
            .fusion
            .fuse(keyword_results, semantic_results, &query.weights);

        // Add metadata
        let metadata_map = self.doc_metadata.read().unwrap();
        for result in &mut fused_results {
            if let Some(metadata) = metadata_map.get(&result.doc_id) {
                result.metadata = metadata.clone();
            }
        }

        fused_results.truncate(query.top_k);

        Ok(fused_results)
    }

    /// Adaptive search (choose mode based on query)
    fn adaptive_search(&self, query: &HybridQuery) -> anyhow::Result<Vec<HybridResult>> {
        // Simple heuristic: if query is short, prefer keyword
        // if query is long or has complex semantics, prefer hybrid
        let word_count = query.query_text.split_whitespace().count();

        if word_count <= 2 && query.query_vector.is_none() {
            // Short query without vector -> keyword
            self.keyword_search(query)
        } else if query.query_vector.is_some() {
            // Has vector -> hybrid
            self.hybrid_search(query)
        } else {
            // Fallback to hybrid with generated vector
            self.keyword_search(query)
        }
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Get document count
    pub fn document_count(&self) -> usize {
        self.doc_vectors.read().unwrap().len()
    }

    /// Get configuration
    pub fn config(&self) -> &HybridSearchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_search::types::SearchWeights;

    #[test]
    fn test_manager_creation() {
        let config = HybridSearchConfig::default();
        let manager = HybridSearchManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_add_document() {
        let config = HybridSearchConfig::default();
        let manager = HybridSearchManager::new(config).unwrap();

        let vector = vec![0.1, 0.2, 0.3, 0.4];
        let metadata = HashMap::new();

        let result = manager.add_document("doc1", "test document", vector, metadata);
        assert!(result.is_ok());
        assert_eq!(manager.document_count(), 1);
    }

    #[test]
    fn test_keyword_only_search() {
        let config = HybridSearchConfig::keyword_only();
        let manager = HybridSearchManager::new(config).unwrap();

        manager
            .add_document("doc1", "machine learning", vec![0.1; 4], HashMap::new())
            .unwrap();
        manager
            .add_document("doc2", "deep learning", vec![0.2; 4], HashMap::new())
            .unwrap();

        let query = HybridQuery {
            query_text: "machine learning".to_string(),
            query_vector: None,
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_semantic_only_search() {
        let config = HybridSearchConfig::semantic_only();
        let manager = HybridSearchManager::new(config).unwrap();

        manager
            .add_document("doc1", "test1", vec![1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        manager
            .add_document("doc2", "test2", vec![0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();

        let query = HybridQuery {
            query_text: "test".to_string(),
            query_vector: Some(vec![0.9, 0.1, 0.0, 0.0]),
            top_k: 10,
            weights: SearchWeights::default(),
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "doc1"); // Closest vector
    }

    #[test]
    fn test_hybrid_search() {
        let config = HybridSearchConfig::balanced();
        let manager = HybridSearchManager::new(config).unwrap();

        manager
            .add_document(
                "doc1",
                "machine learning",
                vec![1.0, 0.0, 0.0, 0.0],
                HashMap::new(),
            )
            .unwrap();
        manager
            .add_document(
                "doc2",
                "deep learning",
                vec![0.0, 1.0, 0.0, 0.0],
                HashMap::new(),
            )
            .unwrap();

        let query = HybridQuery {
            query_text: "machine learning".to_string(),
            query_vector: Some(vec![0.9, 0.1, 0.0, 0.0]),
            top_k: 10,
            weights: SearchWeights {
                keyword_weight: 0.5,
                semantic_weight: 0.5,
                recency_weight: 0.0,
            },
            filters: HashMap::new(),
        };

        let results = manager.search(query).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = HybridSearchManager::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        let sim2 = HybridSearchManager::cosine_similarity(&c, &d);
        assert!((sim2 - 0.0).abs() < 0.001);
    }
}
