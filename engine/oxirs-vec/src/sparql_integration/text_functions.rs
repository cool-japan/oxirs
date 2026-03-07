//! SPARQL function bindings for Tantivy text search
//!
//! This module provides SPARQL functions that integrate Tantivy full-text search
//! capabilities into SPARQL queries.
//!
//! ## Available Functions
//!
//! - `vec:text_search(?lit, "query", limit, threshold)` - Basic text search with BM25
//! - `vec:phrase_search(?lit, "exact phrase", limit)` - Exact phrase matching
//! - `vec:fuzzy_search(?lit, "misspeled", distance, limit)` - Fuzzy matching with edit distance
//! - `vec:field_search(?lit, "title:AI description:neural", limit)` - Field-specific search
//!
//! ## Example SPARQL Query
//!
//! ```sparql
//! PREFIX vec: <http://oxirs.org/vec#>
//! PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
//!
//! SELECT ?doc ?label WHERE {
//!   ?doc rdfs:label ?label .
//!   FILTER(vec:text_search(?label, "machine learning", 10, 0.7))
//! }
//! ```

#[cfg(feature = "tantivy-search")]
use super::super::hybrid_search::tantivy_searcher::{RdfDocument, SearchResult, TantivySearcher};

use anyhow::{anyhow, Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// SPARQL text search function executor
#[cfg(feature = "tantivy-search")]
pub struct SparqlTextFunctions {
    searcher: Arc<RwLock<TantivySearcher>>,
}

#[cfg(feature = "tantivy-search")]
impl SparqlTextFunctions {
    /// Create a new SPARQL text functions executor
    pub fn new(searcher: TantivySearcher) -> Self {
        Self {
            searcher: Arc::new(RwLock::new(searcher)),
        }
    }

    /// Execute vec:text_search function
    ///
    /// Arguments:
    /// - literal: The RDF literal to search in
    /// - query: Text query string
    /// - limit: Maximum number of results (default: 10)
    /// - threshold: Minimum score threshold (default: 0.0)
    pub fn text_search(
        &self,
        _literal: &str,
        query: &str,
        limit: Option<usize>,
        threshold: Option<f32>,
    ) -> Result<Vec<SparqlSearchResult>> {
        let searcher = self.searcher.read();
        let limit = limit.unwrap_or(10);
        let threshold = threshold.unwrap_or(0.0);

        let results = searcher
            .text_search(query, limit, threshold)
            .context("Failed to execute text search")?;

        Ok(results.into_iter().map(SparqlSearchResult::from).collect())
    }

    /// Execute vec:phrase_search function
    ///
    /// Arguments:
    /// - literal: The RDF literal to search in
    /// - phrase: Exact phrase to match
    /// - limit: Maximum number of results (default: 10)
    pub fn phrase_search(
        &self,
        _literal: &str,
        phrase: &str,
        limit: Option<usize>,
    ) -> Result<Vec<SparqlSearchResult>> {
        let searcher = self.searcher.read();
        let limit = limit.unwrap_or(10);

        let results = searcher
            .phrase_search(phrase, limit)
            .context("Failed to execute phrase search")?;

        Ok(results.into_iter().map(SparqlSearchResult::from).collect())
    }

    /// Execute vec:fuzzy_search function
    ///
    /// Arguments:
    /// - literal: The RDF literal to search in
    /// - query: Query string (possibly misspelled)
    /// - distance: Maximum edit distance (default: 2)
    /// - limit: Maximum number of results (default: 10)
    pub fn fuzzy_search(
        &self,
        _literal: &str,
        query: &str,
        distance: Option<u8>,
        limit: Option<usize>,
    ) -> Result<Vec<SparqlSearchResult>> {
        let searcher = self.searcher.read();
        let distance = distance.unwrap_or(2);
        let limit = limit.unwrap_or(10);

        let results = searcher
            .fuzzy_search(query, distance, limit)
            .context("Failed to execute fuzzy search")?;

        Ok(results.into_iter().map(SparqlSearchResult::from).collect())
    }

    /// Execute vec:field_search function
    ///
    /// Arguments:
    /// - literal: The RDF literal to search in
    /// - field_query: Field-specific query string (e.g., "title:AI description:neural")
    /// - limit: Maximum number of results (default: 10)
    pub fn field_search(
        &self,
        _literal: &str,
        field_query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<SparqlSearchResult>> {
        let searcher = self.searcher.read();
        let limit = limit.unwrap_or(10);

        // Parse field query string
        let field_queries = Self::parse_field_query(field_query)?;

        let results = searcher
            .field_search(&field_queries, limit)
            .context("Failed to execute field search")?;

        Ok(results.into_iter().map(SparqlSearchResult::from).collect())
    }

    /// Parse field query string into field-value map
    ///
    /// Example: "title:AI description:neural" -> {"title": "AI", "description": "neural"}
    fn parse_field_query(query: &str) -> Result<HashMap<String, String>> {
        let mut field_queries = HashMap::new();

        // Split by whitespace, then by colon
        let parts: Vec<&str> = query.split_whitespace().collect();

        for part in parts {
            if let Some(colon_pos) = part.find(':') {
                let field = part[..colon_pos].to_string();
                let value = part[colon_pos + 1..].to_string();

                if !field.is_empty() && !value.is_empty() {
                    field_queries.insert(field, value);
                }
            }
        }

        if field_queries.is_empty() {
            return Err(anyhow!("Invalid field query format"));
        }

        Ok(field_queries)
    }

    /// Index RDF literals for searching
    pub fn index_literals(&mut self, literals: Vec<RdfLiteral>) -> Result<()> {
        let mut searcher = self.searcher.write();

        let docs: Vec<RdfDocument> = literals
            .into_iter()
            .map(|lit| RdfDocument {
                uri: lit.subject_uri,
                content: lit.value,
                language: lit.language,
                datatype: lit.datatype,
            })
            .collect();

        searcher
            .index_documents(&docs)
            .context("Failed to index RDF literals")?;

        Ok(())
    }

    /// Get search statistics
    pub fn get_stats(&self) -> SearchStats {
        let searcher = self.searcher.read();
        let index_stats = searcher.get_stats();

        SearchStats {
            total_indexed: index_stats.total_documents,
            heap_size_mb: index_stats.heap_size_mb,
        }
    }
}

#[cfg(not(feature = "tantivy-search"))]
/// Stub implementation when tantivy-search feature is disabled
pub struct SparqlTextFunctions;

#[cfg(not(feature = "tantivy-search"))]
impl SparqlTextFunctions {
    pub fn new(_searcher: ()) -> Self {
        Self
    }

    pub fn text_search(
        &self,
        _literal: &str,
        _query: &str,
        _limit: Option<usize>,
        _threshold: Option<f32>,
    ) -> Result<Vec<SparqlSearchResult>> {
        Err(anyhow!("Tantivy search feature is not enabled"))
    }

    pub fn phrase_search(
        &self,
        _literal: &str,
        _phrase: &str,
        _limit: Option<usize>,
    ) -> Result<Vec<SparqlSearchResult>> {
        Err(anyhow!("Tantivy search feature is not enabled"))
    }

    pub fn fuzzy_search(
        &self,
        _literal: &str,
        _query: &str,
        _distance: Option<u8>,
        _limit: Option<usize>,
    ) -> Result<Vec<SparqlSearchResult>> {
        Err(anyhow!("Tantivy search feature is not enabled"))
    }

    pub fn field_search(
        &self,
        _literal: &str,
        _field_query: &str,
        _limit: Option<usize>,
    ) -> Result<Vec<SparqlSearchResult>> {
        Err(anyhow!("Tantivy search feature is not enabled"))
    }
}

/// RDF literal for indexing
#[derive(Debug, Clone)]
pub struct RdfLiteral {
    /// Subject URI that has this literal
    pub subject_uri: String,
    /// Literal value
    pub value: String,
    /// Language tag (@en, @de, etc.)
    pub language: Option<String>,
    /// Datatype IRI (xsd:string, etc.)
    pub datatype: Option<String>,
}

/// Search result for SPARQL queries
#[derive(Debug, Clone)]
pub struct SparqlSearchResult {
    /// Resource URI
    pub uri: String,
    /// Relevance score (0.0 to 1.0)
    pub score: f32,
    /// Matched snippet
    pub snippet: Option<String>,
    /// Language tag
    pub language: Option<String>,
}

#[cfg(feature = "tantivy-search")]
impl From<SearchResult> for SparqlSearchResult {
    fn from(result: SearchResult) -> Self {
        Self {
            uri: result.uri,
            score: result.score,
            snippet: result.snippet,
            language: result.language,
        }
    }
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStats {
    /// Total documents indexed
    pub total_indexed: u64,
    /// Heap size in MB
    pub heap_size_mb: usize,
}

/// Helper function to register text search functions in SPARQL engine
///
/// This should be called during SPARQL engine initialization to make
/// text search functions available in queries.
#[cfg(feature = "tantivy-search")]
pub fn register_text_functions(
    function_registry: &mut HashMap<String, Box<dyn Fn(&[String]) -> Result<String>>>,
    text_functions: Arc<SparqlTextFunctions>,
) {
    // Register vec:text_search
    let text_funcs = text_functions.clone();
    function_registry.insert(
        "http://oxirs.org/vec#text_search".to_string(),
        Box::new(move |args: &[String]| -> Result<String> {
            if args.len() < 2 {
                return Err(anyhow!("text_search requires at least 2 arguments"));
            }

            let literal = &args[0];
            let query = &args[1];
            let limit = args.get(2).and_then(|s| s.parse().ok());
            let threshold = args.get(3).and_then(|s| s.parse().ok());

            let results = text_funcs.text_search(literal, query, limit, threshold)?;

            // Convert results to SPARQL result format
            let uris: Vec<String> = results.iter().map(|r| r.uri.clone()).collect();
            Ok(uris.join(" "))
        }),
    );

    // Register vec:phrase_search
    let text_funcs = text_functions.clone();
    function_registry.insert(
        "http://oxirs.org/vec#phrase_search".to_string(),
        Box::new(move |args: &[String]| -> Result<String> {
            if args.len() < 2 {
                return Err(anyhow!("phrase_search requires at least 2 arguments"));
            }

            let literal = &args[0];
            let phrase = &args[1];
            let limit = args.get(2).and_then(|s| s.parse().ok());

            let results = text_funcs.phrase_search(literal, phrase, limit)?;

            let uris: Vec<String> = results.iter().map(|r| r.uri.clone()).collect();
            Ok(uris.join(" "))
        }),
    );

    // Register vec:fuzzy_search
    let text_funcs = text_functions.clone();
    function_registry.insert(
        "http://oxirs.org/vec#fuzzy_search".to_string(),
        Box::new(move |args: &[String]| -> Result<String> {
            if args.len() < 2 {
                return Err(anyhow!("fuzzy_search requires at least 2 arguments"));
            }

            let literal = &args[0];
            let query = &args[1];
            let distance = args.get(2).and_then(|s| s.parse().ok());
            let limit = args.get(3).and_then(|s| s.parse().ok());

            let results = text_funcs.fuzzy_search(literal, query, distance, limit)?;

            let uris: Vec<String> = results.iter().map(|r| r.uri.clone()).collect();
            Ok(uris.join(" "))
        }),
    );

    // Register vec:field_search
    let text_funcs = text_functions;
    function_registry.insert(
        "http://oxirs.org/vec#field_search".to_string(),
        Box::new(move |args: &[String]| -> Result<String> {
            if args.len() < 2 {
                return Err(anyhow!("field_search requires at least 2 arguments"));
            }

            let literal = &args[0];
            let field_query = &args[1];
            let limit = args.get(2).and_then(|s| s.parse().ok());

            let results = text_funcs.field_search(literal, field_query, limit)?;

            let uris: Vec<String> = results.iter().map(|r| r.uri.clone()).collect();
            Ok(uris.join(" "))
        }),
    );
}

#[cfg(test)]
#[cfg(feature = "tantivy-search")]
mod tests {
    use super::*;
    use crate::hybrid_search::tantivy_searcher::{TantivyConfig, TantivySearcher};
    use std::env;

    fn create_test_searcher() -> TantivySearcher {
        let temp_dir =
            env::temp_dir().join(format!("tantivy_sparql_test_{}", uuid::Uuid::new_v4()));
        let config = TantivyConfig {
            index_path: temp_dir,
            heap_size_mb: 50,
            stemming: true,
            stopwords: true,
            fuzzy_distance: 2,
        };

        TantivySearcher::new(config).expect("Failed to create searcher")
    }

    #[test]
    fn test_sparql_text_search() -> Result<()> {
        let searcher = create_test_searcher();
        let mut text_funcs = SparqlTextFunctions::new(searcher);

        // Index some test literals
        let literals = vec![
            RdfLiteral {
                subject_uri: "http://example.org/paper1".to_string(),
                value: "Deep learning for natural language processing".to_string(),
                language: Some("en".to_string()),
                datatype: Some("xsd:string".to_string()),
            },
            RdfLiteral {
                subject_uri: "http://example.org/paper2".to_string(),
                value: "Machine learning algorithms and applications".to_string(),
                language: Some("en".to_string()),
                datatype: Some("xsd:string".to_string()),
            },
        ];

        text_funcs.index_literals(literals)?;

        // Test text search
        let results = text_funcs.text_search("", "learning", Some(10), Some(0.0))?;
        assert!(!results.is_empty(), "Should find matching documents");
        assert!(results.len() >= 1);

        Ok(())
    }

    #[test]
    fn test_sparql_phrase_search() -> Result<()> {
        let searcher = create_test_searcher();
        let mut text_funcs = SparqlTextFunctions::new(searcher);

        let literals = vec![RdfLiteral {
            subject_uri: "http://example.org/article1".to_string(),
            value: "The quick brown fox jumps over the lazy dog".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        }];

        text_funcs.index_literals(literals)?;

        // Test phrase search
        let results = text_funcs.phrase_search("", "brown fox", Some(10))?;
        assert_eq!(results.len(), 1, "Should find exact phrase");
        assert_eq!(results[0].uri, "http://example.org/article1");

        Ok(())
    }

    #[test]
    fn test_sparql_fuzzy_search() -> Result<()> {
        let searcher = create_test_searcher();
        let mut text_funcs = SparqlTextFunctions::new(searcher);

        let literals = vec![RdfLiteral {
            subject_uri: "http://example.org/doc1".to_string(),
            value: "Information retrieval systems".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        }];

        text_funcs.index_literals(literals)?;

        // Test fuzzy search with misspelling
        let results = text_funcs.fuzzy_search("", "retreival", Some(2), Some(10))?;
        assert!(
            !results.is_empty(),
            "Should find fuzzy match for misspelling"
        );

        Ok(())
    }

    #[test]
    fn test_field_query_parsing() -> Result<()> {
        let query = "title:AI description:neural author:Smith";
        let fields = SparqlTextFunctions::parse_field_query(query)?;

        assert_eq!(fields.len(), 3);
        assert_eq!(fields.get("title"), Some(&"AI".to_string()));
        assert_eq!(fields.get("description"), Some(&"neural".to_string()));
        assert_eq!(fields.get("author"), Some(&"Smith".to_string()));

        Ok(())
    }

    #[test]
    fn test_search_stats() -> Result<()> {
        let searcher = create_test_searcher();
        let mut text_funcs = SparqlTextFunctions::new(searcher);

        let literals = vec![RdfLiteral {
            subject_uri: "http://example.org/test".to_string(),
            value: "Test document".to_string(),
            language: None,
            datatype: None,
        }];

        text_funcs.index_literals(literals)?;

        let stats = text_funcs.get_stats();
        assert_eq!(stats.total_indexed, 1);
        assert_eq!(stats.heap_size_mb, 50);

        Ok(())
    }
}
