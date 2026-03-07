//! Full-Text Search Index for RDF Literals
//!
//! Provides full-text search capabilities over RDF literals for use via the
//! SPARQL `text:` service extension.
//!
//! Two backends:
//! - **Tantivy backend** (default): production-quality Lucene-equivalent engine
//! - **Simple inverted index fallback**: used only when the `tantivy` feature is
//!   unavailable (always compiled in as the `SimpleTextIndex` type for tests).

use crate::error::{FusekiError, FusekiResult};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

// ──────────────────────────────────────────────────────────────────────────────
// Shared types
// ──────────────────────────────────────────────────────────────────────────────

/// An RDF literal to be indexed for full-text search.
#[derive(Debug, Clone)]
pub struct IndexedLiteral {
    /// Subject IRI or blank node identifier
    pub subject: String,
    /// Predicate IRI
    pub predicate: String,
    /// The literal text value
    pub literal_value: String,
    /// Language tag (e.g. `"en"`)
    pub lang: Option<String>,
    /// Datatype IRI (e.g. `xsd:string`)
    pub datatype: Option<String>,
    /// Named graph IRI, or `None` for the default graph
    pub graph: Option<String>,
}

/// A single text-search hit.
#[derive(Debug, Clone)]
pub struct TextSearchHit {
    pub subject: String,
    pub predicate: String,
    pub literal_value: String,
    /// BM25 relevance score
    pub score: f32,
    pub graph: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Simple in-memory inverted index (always available)
// ──────────────────────────────────────────────────────────────────────────────

/// Internal document stored by `SimpleTextIndex`.
#[derive(Debug, Clone)]
struct Document {
    id: u32,
    literal: IndexedLiteral,
    /// Sorted list of tokens (may repeat) for BM25
    tokens: Vec<String>,
}

/// Simple in-memory inverted index for full-text search over RDF literals.
///
/// Uses BM25 scoring and AND-semantics for multi-term queries.
/// For production deployments with large datasets use `TantivyTextIndex`.
pub struct SimpleTextIndex {
    /// token → list of (doc_id, term_frequency)
    inverted: HashMap<String, Vec<(u32, u32)>>,
    /// All indexed documents
    documents: Vec<Document>,
    /// Next document ID
    next_id: u32,
    /// English stop words
    stop_words: HashSet<&'static str>,
    /// Average document length (tokens) – maintained incrementally
    avg_doc_len: f32,
}

impl SimpleTextIndex {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    /// English stop words used during tokenization.
    const STOP_WORDS: &'static [&'static str] = &[
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
        "he", "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "there",
        "they", "this", "to", "was", "were", "will", "with",
    ];

    pub fn new() -> Self {
        let stop_words: HashSet<&'static str> = Self::STOP_WORDS.iter().cloned().collect();
        SimpleTextIndex {
            inverted: HashMap::new(),
            documents: Vec::new(),
            next_id: 0,
            stop_words,
            avg_doc_len: 0.0,
        }
    }

    /// Index a literal and return its document ID.
    pub fn index(&mut self, literal: IndexedLiteral) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let tokens = Self::tokenize(&literal.literal_value);

        // Update inverted index
        let mut tf_counts: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *tf_counts.entry(token.clone()).or_insert(0) += 1;
        }
        for (token, tf) in &tf_counts {
            if !self.stop_words.contains(token.as_str()) {
                self.inverted
                    .entry(token.clone())
                    .or_default()
                    .push((id, *tf));
            }
        }

        // Update running average doc length
        let n = self.documents.len() as f32;
        self.avg_doc_len = (self.avg_doc_len * n + tokens.len() as f32) / (n + 1.0);

        self.documents.push(Document {
            id,
            literal,
            tokens,
        });
        id
    }

    /// Remove all indexed documents for a given subject URI.
    /// Returns the number of documents removed.
    pub fn remove_subject(&mut self, subject: &str) -> usize {
        let remove_ids: HashSet<u32> = self
            .documents
            .iter()
            .filter(|d| d.literal.subject == subject)
            .map(|d| d.id)
            .collect();

        if remove_ids.is_empty() {
            return 0;
        }

        let count = remove_ids.len();

        // Remove from documents list
        self.documents.retain(|d| !remove_ids.contains(&d.id));

        // Remove from inverted index
        for posting_list in self.inverted.values_mut() {
            posting_list.retain(|(id, _)| !remove_ids.contains(id));
        }
        self.inverted.retain(|_, list| !list.is_empty());

        // Recompute avg_doc_len
        if self.documents.is_empty() {
            self.avg_doc_len = 0.0;
        } else {
            let total: f32 = self.documents.iter().map(|d| d.tokens.len() as f32).sum();
            self.avg_doc_len = total / self.documents.len() as f32;
        }

        count
    }

    /// Search with AND-semantics: all query terms must appear in the document.
    /// Returns up to `limit` results ordered by BM25 score (descending).
    pub fn search(&self, query: &str, limit: usize) -> Vec<TextSearchHit> {
        let terms = Self::tokenize(query);
        if terms.is_empty() {
            return Vec::new();
        }

        // Collect candidate doc IDs: intersection of posting lists for each term
        let candidate_ids: HashSet<u32> = terms
            .iter()
            .filter_map(|t| self.inverted.get(t))
            .map(|list| list.iter().map(|(id, _)| *id).collect::<HashSet<u32>>())
            .reduce(|acc, set| acc.intersection(&set).cloned().collect())
            .unwrap_or_default();

        let mut scored: Vec<(u32, f32)> = candidate_ids
            .into_iter()
            .map(|id| {
                let score = self.bm25_score(id, &terms);
                (id, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored
            .into_iter()
            .filter_map(|(id, score)| {
                let doc = self.documents.iter().find(|d| d.id == id)?;
                Some(TextSearchHit {
                    subject: doc.literal.subject.clone(),
                    predicate: doc.literal.predicate.clone(),
                    literal_value: doc.literal.literal_value.clone(),
                    score,
                    graph: doc.literal.graph.clone(),
                })
            })
            .collect()
    }

    /// Phrase search: all tokens must appear in the document in adjacent order.
    /// Returns up to `limit` results ordered by BM25 score (descending).
    pub fn phrase_search(&self, phrase: &str, limit: usize) -> Vec<TextSearchHit> {
        let terms = Self::tokenize(phrase);
        if terms.is_empty() {
            return Vec::new();
        }

        // Filter candidate documents by phrase proximity
        let mut scored: Vec<(u32, f32)> = self
            .documents
            .iter()
            .filter(|doc| has_phrase(&doc.tokens, &terms))
            .map(|doc| {
                let score = self.bm25_score(doc.id, &terms);
                (doc.id, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored
            .into_iter()
            .filter_map(|(id, score)| {
                let doc = self.documents.iter().find(|d| d.id == id)?;
                Some(TextSearchHit {
                    subject: doc.literal.subject.clone(),
                    predicate: doc.literal.predicate.clone(),
                    literal_value: doc.literal.literal_value.clone(),
                    score,
                    graph: doc.literal.graph.clone(),
                })
            })
            .collect()
    }

    /// Tokenize text: lowercase, split on non-alphanumeric, drop stop words.
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty() && t.len() > 1)
            .map(|t| t.to_string())
            .collect()
    }

    /// BM25 score for a given document against the query terms.
    fn bm25_score(&self, doc_id: u32, query_terms: &[String]) -> f32 {
        let doc = match self.documents.iter().find(|d| d.id == doc_id) {
            Some(d) => d,
            None => return 0.0,
        };

        let doc_len = doc.tokens.len() as f32;
        let n = self.documents.len() as f32;
        let avg_dl = if self.avg_doc_len > 0.0 {
            self.avg_doc_len
        } else {
            1.0
        };

        query_terms
            .iter()
            .map(|term| {
                // Term frequency in this document
                let tf = doc.tokens.iter().filter(|t| *t == term).count() as f32;
                if tf == 0.0 {
                    return 0.0;
                }
                // Document frequency
                let df = self
                    .inverted
                    .get(term)
                    .map(|list| list.len() as f32)
                    .unwrap_or(0.0);
                if df == 0.0 {
                    return 0.0;
                }
                // IDF (smoothed)
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                // Normalised TF
                let tf_norm = (tf * (Self::K1 + 1.0))
                    / (tf + Self::K1 * (1.0 - Self::B + Self::B * doc_len / avg_dl));
                idf * tf_norm
            })
            .sum()
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn term_count(&self) -> usize {
        self.inverted.len()
    }
}

impl Default for SimpleTextIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` if `tokens` contains all elements of `phrase` in adjacent
/// sequential order (phrase match).
fn has_phrase(tokens: &[String], phrase: &[String]) -> bool {
    if phrase.is_empty() {
        return true;
    }
    if tokens.len() < phrase.len() {
        return false;
    }
    tokens.windows(phrase.len()).any(|window| window == phrase)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tantivy backend
// ──────────────────────────────────────────────────────────────────────────────

/// Tantivy-backed full-text search index for production use.
///
/// Provides high-performance, Lucene-compatible indexing and querying over
/// large RDF literal datasets.  The index is persisted to disk at `index_dir`.
pub struct TantivyTextIndex {
    index: tantivy::Index,
    writer: Arc<RwLock<tantivy::IndexWriter>>,
    reader: tantivy::IndexReader,
    schema: tantivy::schema::Schema,
    /// Field: subject IRI
    field_subject: tantivy::schema::Field,
    /// Field: predicate IRI
    field_predicate: tantivy::schema::Field,
    /// Field: literal text (full-text indexed)
    field_literal: tantivy::schema::Field,
    /// Field: graph IRI
    field_graph: tantivy::schema::Field,
    /// Field: language tag
    field_lang: tantivy::schema::Field,
}

impl TantivyTextIndex {
    /// Writer heap size (default 50 MB).
    const WRITER_HEAP_BYTES: usize = 50 * 1024 * 1024;

    /// Open (or create) a Tantivy index at the given directory.
    pub fn open(index_dir: PathBuf) -> FusekiResult<Self> {
        use tantivy::schema::{SchemaBuilder, STORED, STRING, TEXT};

        let mut schema_builder = SchemaBuilder::new();
        let field_subject = schema_builder.add_text_field("subject", STRING | STORED);
        let field_predicate = schema_builder.add_text_field("predicate", STRING | STORED);
        let field_literal = schema_builder.add_text_field("literal", TEXT | STORED);
        let field_graph = schema_builder.add_text_field("graph", STRING | STORED);
        let field_lang = schema_builder.add_text_field("lang", STRING | STORED);
        let schema = schema_builder.build();

        std::fs::create_dir_all(&index_dir).map_err(FusekiError::Io)?;

        let index = tantivy::Index::create_in_dir(&index_dir, schema.clone())
            .or_else(|_| tantivy::Index::open_in_dir(&index_dir))
            .map_err(|e| FusekiError::Internal {
                message: format!("Failed to open Tantivy index: {e}"),
            })?;

        let writer = index
            .writer(Self::WRITER_HEAP_BYTES)
            .map_err(|e| FusekiError::Internal {
                message: format!("Failed to create Tantivy writer: {e}"),
            })?;

        // Use Manual reload so that `commit()` can trigger an immediate reload,
        // making newly indexed documents visible to readers without any delay.
        // This is correct for both production (call commit() explicitly) and tests.
        let reader = index
            .reader_builder()
            .reload_policy(tantivy::ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| FusekiError::Internal {
                message: format!("Failed to create Tantivy reader: {e}"),
            })?;

        info!(path = %index_dir.display(), "Tantivy text index opened");

        Ok(TantivyTextIndex {
            index,
            writer: Arc::new(RwLock::new(writer)),
            reader,
            schema,
            field_subject,
            field_predicate,
            field_literal,
            field_graph,
            field_lang,
        })
    }

    /// Index a single RDF literal.  Changes are buffered; call `commit()` to persist.
    pub fn index(&self, literal: &IndexedLiteral) -> FusekiResult<()> {
        let mut doc = tantivy::TantivyDocument::default();
        doc.add_text(self.field_subject, &literal.subject);
        doc.add_text(self.field_predicate, &literal.predicate);
        doc.add_text(self.field_literal, &literal.literal_value);
        doc.add_text(self.field_graph, literal.graph.as_deref().unwrap_or(""));
        doc.add_text(self.field_lang, literal.lang.as_deref().unwrap_or(""));

        let writer = self.writer.write().map_err(|e| FusekiError::Internal {
            message: format!("Tantivy writer lock poisoned: {e}"),
        })?;
        writer
            .add_document(doc)
            .map_err(|e| FusekiError::Internal {
                message: format!("Failed to add Tantivy document: {e}"),
            })?;
        Ok(())
    }

    /// Commit buffered writes to the index and immediately reload the reader.
    ///
    /// Because we use `ReloadPolicy::Manual`, the reader will not pick up newly
    /// committed documents until `reload()` is called.  We do that here so that
    /// `search()` always reflects the latest committed state.
    pub fn commit(&self) -> FusekiResult<()> {
        {
            let mut writer = self.writer.write().map_err(|e| FusekiError::Internal {
                message: format!("Tantivy writer lock poisoned on commit: {e}"),
            })?;
            writer.commit().map_err(|e| FusekiError::Internal {
                message: format!("Tantivy commit failed: {e}"),
            })?;
        } // release writer lock before reloading reader
        self.reader.reload().map_err(|e| FusekiError::Internal {
            message: format!("Tantivy reader reload failed: {e}"),
        })?;
        debug!("Tantivy index committed and reader reloaded");
        Ok(())
    }

    /// Delete all documents for a given subject.
    pub fn remove_subject(&self, subject: &str) -> FusekiResult<()> {
        use tantivy::Term;

        let term = Term::from_field_text(self.field_subject, subject);
        let writer = self.writer.write().map_err(|e| FusekiError::Internal {
            message: format!("Tantivy writer lock poisoned on remove: {e}"),
        })?;
        writer.delete_term(term);
        Ok(())
    }

    /// Full-text search.  Returns up to `limit` hits ordered by relevance.
    pub fn search(&self, query_str: &str, limit: usize) -> FusekiResult<Vec<TextSearchHit>> {
        use tantivy::collector::TopDocs;
        use tantivy::query::QueryParser;

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.field_literal]);

        let query = query_parser
            .parse_query(query_str)
            .map_err(|e| FusekiError::Internal {
                message: format!("Tantivy query parse error: {e}"),
            })?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .map_err(|e| FusekiError::Internal {
                message: format!("Tantivy search error: {e}"),
            })?;

        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in top_docs {
            match searcher.doc(doc_addr) {
                Ok(doc) => {
                    let subject = get_field_str(&doc, self.field_subject);
                    let predicate = get_field_str(&doc, self.field_predicate);
                    let literal_value = get_field_str(&doc, self.field_literal);
                    let graph = {
                        let g = get_field_str(&doc, self.field_graph);
                        if g.is_empty() {
                            None
                        } else {
                            Some(g)
                        }
                    };
                    hits.push(TextSearchHit {
                        subject,
                        predicate,
                        literal_value,
                        score,
                        graph,
                    });
                }
                Err(e) => {
                    warn!("Failed to retrieve Tantivy document: {}", e);
                }
            }
        }
        Ok(hits)
    }

    /// Total number of documents in the index.
    pub fn document_count(&self) -> usize {
        self.reader.searcher().num_docs() as usize
    }
}

/// Extract a stored text field value from a Tantivy document.
fn get_field_str(doc: &tantivy::TantivyDocument, field: tantivy::schema::Field) -> String {
    use tantivy::schema::Value;
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

// ──────────────────────────────────────────────────────────────────────────────
// Unified facade
// ──────────────────────────────────────────────────────────────────────────────

/// Backend selector for the unified text index.
pub enum TextIndexBackend {
    /// In-memory simple inverted index (for small datasets / testing)
    Simple(SimpleTextIndex),
    /// Tantivy-backed disk-persistent index (for production)
    Tantivy(TantivyTextIndex),
}

/// Thread-safe wrapper that exposes a unified API regardless of the backend.
pub struct TextIndex {
    backend: Arc<RwLock<TextIndexBackend>>,
}

impl TextIndex {
    /// Create a new in-memory simple text index.
    pub fn new_simple() -> Self {
        TextIndex {
            backend: Arc::new(RwLock::new(
                TextIndexBackend::Simple(SimpleTextIndex::new()),
            )),
        }
    }

    /// Create a Tantivy-backed text index persisted at `index_dir`.
    pub fn new_tantivy(index_dir: PathBuf) -> FusekiResult<Self> {
        let tantivy_idx = TantivyTextIndex::open(index_dir)?;
        Ok(TextIndex {
            backend: Arc::new(RwLock::new(TextIndexBackend::Tantivy(tantivy_idx))),
        })
    }

    /// Index an RDF literal.
    pub fn index(&self, literal: IndexedLiteral) -> FusekiResult<()> {
        let mut backend = self.backend.write().map_err(|e| FusekiError::Internal {
            message: format!("TextIndex RwLock poisoned on index: {e}"),
        })?;
        match &mut *backend {
            TextIndexBackend::Simple(idx) => {
                idx.index(literal);
                Ok(())
            }
            TextIndexBackend::Tantivy(idx) => idx.index(&literal),
        }
    }

    /// Remove all literals for a subject.
    pub fn remove_subject(&self, subject: &str) -> FusekiResult<usize> {
        let mut backend = self.backend.write().map_err(|e| FusekiError::Internal {
            message: format!("TextIndex RwLock poisoned on remove: {e}"),
        })?;
        match &mut *backend {
            TextIndexBackend::Simple(idx) => Ok(idx.remove_subject(subject)),
            TextIndexBackend::Tantivy(idx) => {
                idx.remove_subject(subject)?;
                Ok(0) // Tantivy doesn't return a count synchronously
            }
        }
    }

    /// Full-text search.
    pub fn search(&self, query: &str, limit: usize) -> FusekiResult<Vec<TextSearchHit>> {
        let backend = self.backend.read().map_err(|e| FusekiError::Internal {
            message: format!("TextIndex RwLock poisoned on search: {e}"),
        })?;
        match &*backend {
            TextIndexBackend::Simple(idx) => Ok(idx.search(query, limit)),
            TextIndexBackend::Tantivy(idx) => idx.search(query, limit),
        }
    }

    /// Phrase search (Tantivy handles this natively via quoted queries).
    pub fn phrase_search(&self, phrase: &str, limit: usize) -> FusekiResult<Vec<TextSearchHit>> {
        let backend = self.backend.read().map_err(|e| FusekiError::Internal {
            message: format!("TextIndex RwLock poisoned on phrase_search: {e}"),
        })?;
        match &*backend {
            TextIndexBackend::Simple(idx) => Ok(idx.phrase_search(phrase, limit)),
            TextIndexBackend::Tantivy(idx) => {
                // Wrap in quotes for Tantivy phrase query
                let quoted = format!("\"{}\"", phrase.replace('"', ""));
                idx.search(&quoted, limit)
            }
        }
    }

    /// Commit pending writes (no-op for simple backend).
    pub fn commit(&self) -> FusekiResult<()> {
        let backend = self.backend.read().map_err(|e| FusekiError::Internal {
            message: format!("TextIndex RwLock poisoned on commit: {e}"),
        })?;
        match &*backend {
            TextIndexBackend::Simple(_) => Ok(()),
            TextIndexBackend::Tantivy(idx) => idx.commit(),
        }
    }

    /// Number of indexed documents.
    pub fn document_count(&self) -> usize {
        let backend = self.backend.read().unwrap_or_else(|e| e.into_inner());
        match &*backend {
            TextIndexBackend::Simple(idx) => idx.document_count(),
            TextIndexBackend::Tantivy(idx) => idx.document_count(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_literal(subject: &str, predicate: &str, text: &str) -> IndexedLiteral {
        IndexedLiteral {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            literal_value: text.to_string(),
            lang: Some("en".to_string()),
            datatype: None,
            graph: None,
        }
    }

    // ── SimpleTextIndex tests ──────────────────────────────────────────────

    #[test]
    fn test_simple_index_and_search() {
        let mut idx = SimpleTextIndex::new();
        idx.index(make_literal(
            "http://ex.org/doc1",
            "http://ex.org/title",
            "Rust programming language systems",
        ));
        idx.index(make_literal(
            "http://ex.org/doc2",
            "http://ex.org/title",
            "Python scripting programming language",
        ));

        let hits = idx.search("rust programming", 10);
        assert!(!hits.is_empty(), "Should find 'rust programming'");
        assert_eq!(hits[0].subject, "http://ex.org/doc1");
    }

    #[test]
    fn test_simple_bm25_ordering() {
        let mut idx = SimpleTextIndex::new();
        // doc1 mentions "database" once
        idx.index(make_literal(
            "http://ex.org/doc1",
            "http://ex.org/desc",
            "A database system",
        ));
        // doc2 mentions "database" three times
        idx.index(make_literal(
            "http://ex.org/doc2",
            "http://ex.org/desc",
            "database database database management",
        ));

        let hits = idx.search("database", 10);
        assert!(hits.len() == 2);
        // doc2 should rank higher due to higher TF
        assert_eq!(hits[0].subject, "http://ex.org/doc2");
    }

    #[test]
    fn test_simple_remove_subject() {
        let mut idx = SimpleTextIndex::new();
        idx.index(make_literal(
            "http://ex.org/s1",
            "http://ex.org/p",
            "Hello world",
        ));
        idx.index(make_literal(
            "http://ex.org/s2",
            "http://ex.org/p",
            "Hello Rust",
        ));

        let removed = idx.remove_subject("http://ex.org/s1");
        assert_eq!(removed, 1);

        let hits = idx.search("hello", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "http://ex.org/s2");
    }

    #[test]
    fn test_simple_phrase_search() {
        let mut idx = SimpleTextIndex::new();
        idx.index(make_literal(
            "http://ex.org/doc1",
            "http://ex.org/p",
            "semantic web technologies",
        ));
        idx.index(make_literal(
            "http://ex.org/doc2",
            "http://ex.org/p",
            "web semantic data technologies",
        ));

        let hits = idx.phrase_search("semantic web", 10);
        assert_eq!(hits.len(), 1, "Only doc1 has 'semantic web' in order");
        assert_eq!(hits[0].subject, "http://ex.org/doc1");
    }

    #[test]
    fn test_simple_and_semantics() {
        let mut idx = SimpleTextIndex::new();
        idx.index(make_literal(
            "http://ex.org/doc1",
            "http://ex.org/p",
            "apple orange banana",
        ));
        idx.index(make_literal(
            "http://ex.org/doc2",
            "http://ex.org/p",
            "apple mango kiwi",
        ));

        // AND semantics: both "apple" and "orange" must be present
        let hits = idx.search("apple orange", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "http://ex.org/doc1");
    }

    #[test]
    fn test_tokenization() {
        let tokens = SimpleTextIndex::tokenize("Hello, World! This is a TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Stop words should be present in tokenize output (filtering is done in index)
        // "is" and "a" are single chars or stop words, "this" should appear
        assert!(tokens.contains(&"this".to_string()));
    }

    #[test]
    fn test_empty_search() {
        let idx = SimpleTextIndex::new();
        let hits = idx.search("nonexistent", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_document_and_term_count() {
        let mut idx = SimpleTextIndex::new();
        assert_eq!(idx.document_count(), 0);
        idx.index(make_literal("s1", "p", "hello world rust"));
        assert_eq!(idx.document_count(), 1);
        assert!(idx.term_count() > 0);
    }

    // ── TextIndex facade tests ──────────────────────────────────────────────

    #[test]
    fn test_unified_index_simple_backend() {
        let idx = TextIndex::new_simple();
        idx.index(make_literal(
            "http://ex.org/s1",
            "http://ex.org/p",
            "knowledge graph reasoning",
        ))
        .unwrap();
        let hits = idx.search("knowledge graph", 10).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].subject, "http://ex.org/s1");
    }

    #[test]
    fn test_unified_remove_subject() {
        let idx = TextIndex::new_simple();
        idx.index(make_literal(
            "http://ex.org/s1",
            "http://ex.org/p",
            "sparql query language",
        ))
        .unwrap();
        idx.index(make_literal(
            "http://ex.org/s2",
            "http://ex.org/p",
            "sparql endpoint server",
        ))
        .unwrap();

        idx.remove_subject("http://ex.org/s1").unwrap();

        let hits = idx.search("sparql", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].subject, "http://ex.org/s2");
    }

    #[test]
    fn test_tantivy_index() {
        let dir = std::env::temp_dir().join(format!("oxirs_tantivy_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let idx = TextIndex::new_tantivy(dir.clone()).unwrap();
        idx.index(make_literal(
            "http://ex.org/s1",
            "http://ex.org/p",
            "tantivy full text search engine",
        ))
        .unwrap();
        idx.index(make_literal(
            "http://ex.org/s2",
            "http://ex.org/p",
            "sparql semantic web query",
        ))
        .unwrap();
        idx.commit().unwrap();

        let hits = idx.search("tantivy", 10).unwrap();
        assert!(!hits.is_empty(), "Should find 'tantivy' in Tantivy index");
        assert_eq!(hits[0].subject, "http://ex.org/s1");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tantivy_phrase_search() {
        let dir = std::env::temp_dir().join(format!("oxirs_tantivy_phrase_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let idx = TextIndex::new_tantivy(dir.clone()).unwrap();
        idx.index(make_literal(
            "http://ex.org/s1",
            "http://ex.org/p",
            "semantic web technologies",
        ))
        .unwrap();
        idx.index(make_literal(
            "http://ex.org/s2",
            "http://ex.org/p",
            "web semantic computing",
        ))
        .unwrap();
        idx.commit().unwrap();

        let hits = idx.phrase_search("semantic web", 10).unwrap();
        assert!(!hits.is_empty(), "Should find phrase 'semantic web'");
        assert_eq!(hits[0].subject, "http://ex.org/s1");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
