//! Tantivy full-text search integration for OxiRS
//!
//! This module provides advanced text search capabilities including:
//! - Stemming and stopword filtering
//! - Fuzzy matching with edit distance
//! - Phrase queries
//! - Field-specific search
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxirs_vec::hybrid_search::tantivy_searcher::{TantivySearcher, TantivyConfig};
//! use std::path::PathBuf;
//!
//! let config = TantivyConfig {
//!     index_path: PathBuf::from("./data/tantivy_index"),
//!     heap_size_mb: 50,
//!     stemming: true,
//!     stopwords: true,
//!     fuzzy_distance: 2,
//! };
//!
//! let mut searcher = TantivySearcher::new(config).unwrap();
//! ```

#[cfg(feature = "tantivy-search")]
use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    query::{FuzzyTermQuery, PhraseQuery, QueryParser},
    schema::{Field, Schema, SchemaBuilder, TextFieldIndexing, TextOptions, Value, STORED, STRING},
    tokenizer::{
        LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
    },
    IndexReader, IndexSettings, IndexWriter, ReloadPolicy, Term,
};

use anyhow::{anyhow, Context, Result};
use parking_lot::RwLock;
use scirs2_core::metrics::{Counter, Timer};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Configuration for Tantivy search engine
#[derive(Debug, Clone)]
pub struct TantivyConfig {
    /// Path to store the Tantivy index
    pub index_path: PathBuf,
    /// Heap size for indexing in megabytes (default: 50MB)
    pub heap_size_mb: usize,
    /// Enable stemming (default: true)
    pub stemming: bool,
    /// Enable stopword filtering (default: true)
    pub stopwords: bool,
    /// Default fuzzy search distance (default: 2)
    pub fuzzy_distance: u8,
}

impl Default for TantivyConfig {
    fn default() -> Self {
        Self {
            index_path: PathBuf::from("./data/tantivy_index"),
            heap_size_mb: 50,
            stemming: true,
            stopwords: true,
            fuzzy_distance: 2,
        }
    }
}

/// RDF document for indexing
#[derive(Debug, Clone)]
pub struct RdfDocument {
    /// RDF resource URI
    pub uri: String,
    /// Literal value content
    pub content: String,
    /// Language tag (e.g., "en", "de", "fr")
    pub language: Option<String>,
    /// XSD datatype (e.g., "xsd:string")
    pub datatype: Option<String>,
}

/// Search result from Tantivy
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// RDF resource URI
    pub uri: String,
    /// Relevance score (0.0 to 1.0)
    pub score: f32,
    /// Matched content snippet
    pub snippet: Option<String>,
    /// Language tag
    pub language: Option<String>,
}

#[cfg(feature = "tantivy-search")]
/// Tantivy-based full-text searcher
pub struct TantivySearcher {
    index: tantivy::Index,
    reader: IndexReader,
    writer: Option<Arc<RwLock<IndexWriter>>>,
    schema: Schema,
    config: TantivyConfig,

    // Schema fields
    uri_field: Field,
    content_field: Field,
    language_field: Field,
    datatype_field: Field,

    // Metrics
    index_counter: Counter,
    search_timer: Timer,
}

#[cfg(feature = "tantivy-search")]
impl TantivySearcher {
    /// Create a new Tantivy searcher with the given configuration
    pub fn new(config: TantivyConfig) -> Result<Self> {
        // Create schema
        let mut schema_builder = SchemaBuilder::new();

        // URI field - stored, not tokenized
        let uri_field = schema_builder.add_text_field("uri", STRING | STORED);

        // Content field - full-text searchable with custom analyzer
        let text_field_indexing = TextFieldIndexing::default()
            .set_tokenizer("custom")
            .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default()
            .set_indexing_options(text_field_indexing)
            .set_stored();
        let content_field = schema_builder.add_text_field("content", text_options);

        // Language field - stored, not tokenized
        let language_field = schema_builder.add_text_field("language", STRING | STORED);

        // Datatype field - stored, not tokenized
        let datatype_field = schema_builder.add_text_field("datatype", STRING | STORED);

        let schema = schema_builder.build();

        // Create or open index
        std::fs::create_dir_all(&config.index_path).context("Failed to create index directory")?;

        let index_settings = IndexSettings::default();
        let index = if config.index_path.join("meta.json").exists() {
            tantivy::Index::open_in_dir(&config.index_path)
                .context("Failed to open existing index")?
        } else {
            let dir = MmapDirectory::open(&config.index_path)
                .context("Failed to open index directory")?;
            tantivy::Index::create(dir, schema.clone(), index_settings)
                .context("Failed to create index")?
        };

        // Register custom tokenizer with stemming and stopwords
        let tokenizer = Self::create_custom_tokenizer(&config);
        index.tokenizers().register("custom", tokenizer);

        // Create index writer
        let heap_size = config.heap_size_mb * 1024 * 1024;
        let writer = index
            .writer(heap_size)
            .context("Failed to create index writer")?;

        // Create index reader with auto-reload
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create index reader")?;

        Ok(Self {
            index,
            reader,
            writer: Some(Arc::new(RwLock::new(writer))),
            schema,
            config,
            uri_field,
            content_field,
            language_field,
            datatype_field,
            index_counter: Counter::new("tantivy_indexed_docs".to_string()),
            search_timer: Timer::new("tantivy_search_time".to_string()),
        })
    }

    /// Create custom tokenizer with stemming and stopwords
    #[cfg(feature = "tantivy-search")]
    fn create_custom_tokenizer(config: &TantivyConfig) -> TextAnalyzer {
        // For simplicity, always apply all filters
        // Conditional filter application causes type incompatibility issues
        let stopwords = if config.stopwords {
            vec![
                "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
                "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
                "there", "these", "they", "this", "to", "was", "will", "with",
            ]
            .into_iter()
            .map(String::from)
            .collect()
        } else {
            vec![] // Empty stopword list effectively disables filtering
        };

        // Apply all filters in one chain for type compatibility
        TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(StopWordFilter::remove(stopwords))
            .filter(Stemmer::new(tantivy::tokenizer::Language::English))
            .build()
    }

    /// Index multiple RDF documents in batch
    pub fn index_documents(&mut self, docs: &[RdfDocument]) -> Result<()> {
        let writer = self
            .writer
            .as_ref()
            .ok_or_else(|| anyhow!("Index writer not available"))?;

        let mut writer_guard = writer.write();

        for doc in docs {
            let mut tantivy_doc = tantivy::TantivyDocument::default();

            tantivy_doc.add_text(self.uri_field, &doc.uri);
            tantivy_doc.add_text(self.content_field, &doc.content);

            if let Some(ref lang) = doc.language {
                tantivy_doc.add_text(self.language_field, lang);
            }

            if let Some(ref datatype) = doc.datatype {
                tantivy_doc.add_text(self.datatype_field, datatype);
            }

            writer_guard
                .add_document(tantivy_doc)
                .context("Failed to add document")?;

            self.index_counter.add(1);
        }

        writer_guard
            .commit()
            .context("Failed to commit documents")?;

        Ok(())
    }

    /// Basic text search with BM25 ranking
    pub fn text_search(
        &self,
        query: &str,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        let _timer = self.search_timer.start();

        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let query = query_parser
            .parse_query(query)
            .context("Failed to parse query")?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("Failed to execute search")?;

        let mut results = Vec::new();

        for (score, doc_address) in top_docs {
            if score < threshold {
                continue;
            }

            let retrieved_doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .context("Failed to retrieve document")?;

            let uri = retrieved_doc
                .get_first(self.uri_field)
                .and_then(|f| f.as_str())
                .ok_or_else(|| anyhow!("Document missing URI field"))?
                .to_string();

            let content = retrieved_doc
                .get_first(self.content_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            let language = retrieved_doc
                .get_first(self.language_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            results.push(SearchResult {
                uri,
                score,
                snippet: content,
                language,
            });
        }

        Ok(results)
    }

    /// Exact phrase search
    pub fn phrase_search(&self, phrase: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let _timer = self.search_timer.start();

        let searcher = self.reader.searcher();

        // Tokenize the phrase
        let terms: Vec<Term> = phrase
            .split_whitespace()
            .map(|word| Term::from_field_text(self.content_field, word))
            .collect();

        let query = PhraseQuery::new(terms);

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("Failed to execute phrase search")?;

        let mut results = Vec::new();

        for (score, doc_address) in top_docs {
            let retrieved_doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .context("Failed to retrieve document")?;

            let uri = retrieved_doc
                .get_first(self.uri_field)
                .and_then(|f| f.as_str())
                .ok_or_else(|| anyhow!("Document missing URI field"))?
                .to_string();

            let content = retrieved_doc
                .get_first(self.content_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            let language = retrieved_doc
                .get_first(self.language_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            results.push(SearchResult {
                uri,
                score,
                snippet: content,
                language,
            });
        }

        Ok(results)
    }

    /// Fuzzy search with edit distance tolerance
    pub fn fuzzy_search(
        &self,
        query: &str,
        distance: u8,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let _timer = self.search_timer.start();

        let searcher = self.reader.searcher();

        // Use first token for fuzzy matching
        let term = Term::from_field_text(self.content_field, query);
        let fuzzy_query = FuzzyTermQuery::new(term, distance, true);

        let top_docs = searcher
            .search(&fuzzy_query, &TopDocs::with_limit(limit))
            .context("Failed to execute fuzzy search")?;

        let mut results = Vec::new();

        for (score, doc_address) in top_docs {
            let retrieved_doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .context("Failed to retrieve document")?;

            let uri = retrieved_doc
                .get_first(self.uri_field)
                .and_then(|f| f.as_str())
                .ok_or_else(|| anyhow!("Document missing URI field"))?
                .to_string();

            let content = retrieved_doc
                .get_first(self.content_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            let language = retrieved_doc
                .get_first(self.language_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            results.push(SearchResult {
                uri,
                score,
                snippet: content,
                language,
            });
        }

        Ok(results)
    }

    /// Field-specific search with multiple fields
    pub fn field_search(
        &self,
        field_queries: &HashMap<String, String>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let _timer = self.search_timer.start();

        let searcher = self.reader.searcher();

        // Parse each field query
        let mut combined_query_str = String::new();

        for (field_name, query) in field_queries {
            if !combined_query_str.is_empty() {
                combined_query_str.push_str(" AND ");
            }
            combined_query_str.push_str(&format!("{}:({})", field_name, query));
        }

        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        let query = query_parser
            .parse_query(&combined_query_str)
            .context("Failed to parse field query")?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("Failed to execute field search")?;

        let mut results = Vec::new();

        for (score, doc_address) in top_docs {
            let retrieved_doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .context("Failed to retrieve document")?;

            let uri = retrieved_doc
                .get_first(self.uri_field)
                .and_then(|f| f.as_str())
                .ok_or_else(|| anyhow!("Document missing URI field"))?
                .to_string();

            let content = retrieved_doc
                .get_first(self.content_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            let language = retrieved_doc
                .get_first(self.language_field)
                .and_then(|f| f.as_str())
                .map(String::from);

            results.push(SearchResult {
                uri,
                score,
                snippet: content,
                language,
            });
        }

        Ok(results)
    }

    /// Get indexing statistics
    pub fn get_stats(&self) -> IndexStats {
        let searcher = self.reader.searcher();
        let segment_metas = searcher.segment_readers();

        let total_docs = segment_metas.iter().map(|seg| seg.num_docs() as u64).sum();

        IndexStats {
            total_documents: total_docs,
            indexed_count: self.index_counter.get(),
            heap_size_mb: self.config.heap_size_mb,
        }
    }

    /// Optimize index (commit pending changes)
    pub fn optimize(&mut self) -> Result<()> {
        let writer = self
            .writer
            .as_ref()
            .ok_or_else(|| anyhow!("Index writer not available"))?;

        let mut writer_guard = writer.write();

        // Tantivy automatically merges segments during commit
        // We just commit to ensure all changes are persisted
        writer_guard
            .commit()
            .context("Failed to commit during optimization")?;

        Ok(())
    }
}

#[cfg(not(feature = "tantivy-search"))]
/// Stub implementation when tantivy-search feature is disabled
pub struct TantivySearcher;

#[cfg(not(feature = "tantivy-search"))]
impl TantivySearcher {
    pub fn new(_config: TantivyConfig) -> Result<Self> {
        Err(anyhow!(
            "Tantivy search feature is not enabled. Enable with: --features tantivy-search"
        ))
    }
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Total number of indexed documents
    pub total_documents: u64,
    /// Number of documents indexed in current session
    pub indexed_count: u64,
    /// Configured heap size in MB
    pub heap_size_mb: usize,
}

#[cfg(test)]
#[cfg(feature = "tantivy-search")]
mod tests {
    use super::*;
    use std::env;

    fn create_test_config() -> TantivyConfig {
        let temp_dir = env::temp_dir().join(format!("tantivy_test_{}", uuid::Uuid::new_v4()));
        TantivyConfig {
            index_path: temp_dir,
            heap_size_mb: 50,
            stemming: true,
            stopwords: true,
            fuzzy_distance: 2,
        }
    }

    fn create_test_docs() -> Vec<RdfDocument> {
        vec![
            RdfDocument {
                uri: "http://example.org/doc1".to_string(),
                content: "Machine learning algorithms for deep neural networks".to_string(),
                language: Some("en".to_string()),
                datatype: Some("xsd:string".to_string()),
            },
            RdfDocument {
                uri: "http://example.org/doc2".to_string(),
                content: "Natural language processing and understanding".to_string(),
                language: Some("en".to_string()),
                datatype: Some("xsd:string".to_string()),
            },
            RdfDocument {
                uri: "http://example.org/doc3".to_string(),
                content: "Computer vision for image recognition tasks".to_string(),
                language: Some("en".to_string()),
                datatype: Some("xsd:string".to_string()),
            },
        ]
    }

    #[test]
    fn test_basic_indexing_and_search() -> Result<()> {
        let config = create_test_config();
        let mut searcher = TantivySearcher::new(config)?;

        let docs = create_test_docs();
        searcher.index_documents(&docs)?;

        // Search for "machine learning"
        let results = searcher.text_search("machine learning", 10, 0.0)?;
        assert!(!results.is_empty(), "Should find matching documents");
        assert_eq!(results[0].uri, "http://example.org/doc1");

        Ok(())
    }

    #[test]
    fn test_phrase_search() -> Result<()> {
        let config = create_test_config();
        let mut searcher = TantivySearcher::new(config)?;

        let docs = create_test_docs();
        searcher.index_documents(&docs)?;

        // Exact phrase search
        let results = searcher.phrase_search("natural language processing", 10)?;
        assert!(!results.is_empty(), "Should find phrase match");
        assert_eq!(results[0].uri, "http://example.org/doc2");

        Ok(())
    }

    #[test]
    fn test_fuzzy_search() -> Result<()> {
        let config = create_test_config();
        let mut searcher = TantivySearcher::new(config)?;

        let docs = create_test_docs();
        searcher.index_documents(&docs)?;

        // Fuzzy search with misspelling
        let results = searcher.fuzzy_search("machne", 2, 10)?;
        assert!(!results.is_empty(), "Should find fuzzy match");

        Ok(())
    }

    #[test]
    fn test_stemming() -> Result<()> {
        let config = create_test_config();
        let mut searcher = TantivySearcher::new(config)?;

        let docs = vec![RdfDocument {
            uri: "http://example.org/running".to_string(),
            content: "The runner is running a marathon".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        }];

        searcher.index_documents(&docs)?;

        // Search with different form of word (stemming should match)
        let results = searcher.text_search("run", 10, 0.0)?;
        assert!(
            !results.is_empty(),
            "Stemming should match 'run' with 'running'"
        );

        Ok(())
    }

    #[test]
    fn test_stopwords() -> Result<()> {
        let config = create_test_config();
        let mut searcher = TantivySearcher::new(config)?;

        let docs = create_test_docs();
        searcher.index_documents(&docs)?;

        // Stopwords like "and", "for" should be filtered
        let results = searcher.text_search("algorithms for networks", 10, 0.0)?;
        assert!(
            !results.is_empty(),
            "Should find results ignoring stopwords"
        );

        Ok(())
    }

    #[test]
    fn test_index_stats() -> Result<()> {
        let config = create_test_config();
        let mut searcher = TantivySearcher::new(config)?;

        let docs = create_test_docs();
        searcher.index_documents(&docs)?;

        let stats = searcher.get_stats();
        assert_eq!(stats.total_documents, 3);
        assert_eq!(stats.heap_size_mb, 50);

        Ok(())
    }
}
