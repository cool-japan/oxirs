//! Tantivy-backed full-text index for RDF literal search.
//!
//! Provides an in-memory tantivy index that stores (subject IRI, predicate IRI, literal text)
//! triples and supports full-text queries returning scored subject IRIs.

use parking_lot::Mutex;
use std::fmt;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, SchemaBuilder, STORED, STRING, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

/// Namespace for Jena text extension
pub const TEXT_NAMESPACE: &str = "http://jena.apache.org/text#";

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the text search index
#[derive(Debug)]
pub enum TextSearchError {
    /// Tantivy engine error
    IndexError(String),
    /// Invalid or malformed query string
    QueryError(String),
    /// Schema-related error (field not found, etc.)
    SchemaError(String),
}

impl fmt::Display for TextSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexError(msg) => write!(f, "text index error: {msg}"),
            Self::QueryError(msg) => write!(f, "text query error: {msg}"),
            Self::SchemaError(msg) => write!(f, "text schema error: {msg}"),
        }
    }
}

impl std::error::Error for TextSearchError {}

impl From<tantivy::TantivyError> for TextSearchError {
    fn from(e: tantivy::TantivyError) -> Self {
        Self::IndexError(e.to_string())
    }
}

impl From<tantivy::query::QueryParserError> for TextSearchError {
    fn from(e: tantivy::query::QueryParserError) -> Self {
        Self::QueryError(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// A single search result from the text index
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    /// The IRI of the RDF subject
    pub subject_iri: String,
    /// Tantivy BM25 relevance score
    pub score: f32,
    /// The matched literal value
    pub literal_value: String,
    /// The predicate IRI under which the literal was indexed
    pub predicate_iri: String,
}

// ---------------------------------------------------------------------------
// Schema field names
// ---------------------------------------------------------------------------

const FIELD_SUBJECT: &str = "subject";
const FIELD_LITERAL: &str = "literal";
const FIELD_PREDICATE: &str = "predicate";

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// In-memory tantivy full-text index for RDF literal triples.
///
/// Indexes `(subject IRI, predicate IRI, literal text)` triples, then exposes
/// full-text and predicate-filtered search returning ranked `TextSearchResult`s.
///
/// This index is thread-safe. Multiple threads may call `search`/`search_predicate`
/// concurrently; writes serialise through an internal `parking_lot::Mutex`.
pub struct TextSearchIndex {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    subject_field: Field,
    literal_field: Field,
    predicate_field: Field,
}

impl TextSearchIndex {
    /// Tantivy writer heap size in bytes (50 MB minimum for tantivy)
    const WRITER_HEAP_BYTES: usize = 50_000_000;

    /// Create a new in-memory text search index.
    pub fn new_in_memory() -> Result<Self, TextSearchError> {
        let (schema, subject_field, literal_field, predicate_field) = Self::build_schema();

        let index = Index::create_in_ram(schema);
        let writer: IndexWriter = index
            .writer(Self::WRITER_HEAP_BYTES)
            .map_err(TextSearchError::from)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(TextSearchError::from)?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            subject_field,
            literal_field,
            predicate_field,
        })
    }

    /// Index a `(subject IRI, predicate IRI, literal text)` triple.
    ///
    /// The write is buffered; call [`commit`](Self::commit) to make it visible
    /// to searchers.
    pub fn index_triple(
        &self,
        subject: &str,
        predicate: &str,
        literal: &str,
    ) -> Result<(), TextSearchError> {
        let mut doc = TantivyDocument::default();
        doc.add_text(self.subject_field, subject);
        doc.add_text(self.predicate_field, predicate);
        doc.add_text(self.literal_field, literal);

        self.writer
            .lock()
            .add_document(doc)
            .map_err(TextSearchError::from)?;
        Ok(())
    }

    /// Commit all pending writes and reload the reader so searches immediately
    /// reflect the new documents.
    pub fn commit(&self) -> Result<(), TextSearchError> {
        self.writer
            .lock()
            .commit()
            .map(|_| ())
            .map_err(TextSearchError::from)?;
        // Reload the reader so the committed segment is immediately visible.
        self.reader.reload().map_err(TextSearchError::from)
    }

    /// Full-text search across all indexed literals.
    ///
    /// Returns up to `max_results` results ranked by BM25 relevance score.
    pub fn search(
        &self,
        query_str: &str,
        max_results: usize,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        self.run_search(query_str, None, max_results)
    }

    /// Full-text search filtered to a specific predicate IRI.
    ///
    /// Only documents whose `predicate` field matches `predicate` exactly are
    /// returned.  Filtering is applied as a post-filter after BM25 ranking.
    pub fn search_predicate(
        &self,
        query_str: &str,
        predicate: &str,
        max_results: usize,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        self.run_search(query_str, Some(predicate), max_results)
    }

    /// Number of documents currently committed to the index.
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn build_schema() -> (Schema, Field, Field, Field) {
        let mut builder: SchemaBuilder = Schema::builder();
        // STRING = exact match, stored. TEXT = full-text indexed + stored.
        let subject_field = builder.add_text_field(FIELD_SUBJECT, STRING | STORED);
        let predicate_field = builder.add_text_field(FIELD_PREDICATE, STRING | STORED);
        let literal_field = builder.add_text_field(FIELD_LITERAL, TEXT | STORED);
        let schema = builder.build();
        (schema, subject_field, literal_field, predicate_field)
    }

    /// Core search: fetch candidates (with over-fetch when filtering), then
    /// apply optional predicate post-filter.
    fn run_search(
        &self,
        query_str: &str,
        predicate_filter: Option<&str>,
        max_results: usize,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        if query_str.is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.literal_field]);
        let query = query_parser
            .parse_query(query_str)
            .map_err(TextSearchError::from)?;

        // Over-fetch when filtering by predicate so we still get `max_results`
        // after post-filter removal.
        let fetch_limit = if predicate_filter.is_some() {
            max_results.saturating_mul(10).max(max_results + 50)
        } else {
            max_results
        };

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(fetch_limit).order_by_score())
            .map_err(TextSearchError::from)?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address).map_err(TextSearchError::from)?;

            let subject_iri = self.get_stored_str(&doc, self.subject_field);
            let literal_value = self.get_stored_str(&doc, self.literal_field);
            let predicate_iri = self.get_stored_str(&doc, self.predicate_field);

            // Apply predicate post-filter
            if let Some(pred) = predicate_filter {
                if predicate_iri != pred {
                    continue;
                }
            }

            results.push(TextSearchResult {
                subject_iri,
                score,
                literal_value,
                predicate_iri,
            });

            if results.len() >= max_results {
                break;
            }
        }

        Ok(results)
    }

    fn get_stored_str(&self, doc: &TantivyDocument, field: Field) -> String {
        use tantivy::schema::Value;
        doc.get_first(field)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }
}
