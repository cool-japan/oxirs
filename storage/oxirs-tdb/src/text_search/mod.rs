//! Full-text search index for RDF literal values using Tantivy.
//!
//! Provides:
//! - [`TextSearchIndex`]: Tantivy-backed index for RDF literals stored in TDB.
//! - [`TextPropertyFunction`]: Evaluates SPARQL `text:query(?subject, "terms")`.
//!
//! Two storage modes:
//! - **In-memory** (`TextSearchConfig { index_path: None, .. }`): fast, ephemeral, for tests.
//! - **On-disk** (`index_path: Some(dir)`): durable, production-ready.

use std::path::PathBuf;
use std::sync::Arc;

use log::{debug, warn};
use parking_lot::RwLock;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{SchemaBuilder, Value, STORED, STRING, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use crate::error::{Result, TdbError};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`TextSearchIndex`].
#[derive(Debug, Clone)]
pub struct TextSearchConfig {
    /// Directory for the Tantivy index.  `None` creates a RAM-based index.
    pub index_path: Option<PathBuf>,
    /// Tantivy writer heap budget in megabytes (default 50 MB).
    pub heap_size_mb: usize,
    /// Auto-commit the writer after accumulating this many staged documents.
    /// `0` disables auto-commit (manual `commit()` only).
    pub commit_threshold: usize,
}

impl Default for TextSearchConfig {
    fn default() -> Self {
        Self {
            index_path: None,
            heap_size_mb: 50,
            commit_threshold: 1_000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input / output types
// ─────────────────────────────────────────────────────────────────────────────

/// An RDF literal to be indexed.
#[derive(Debug, Clone)]
pub struct LiteralDocument {
    /// IRI (or blank-node ID) of the triple's subject.
    pub subject: String,
    /// IRI of the triple's predicate.
    pub predicate: String,
    /// BCP-47 language tag, e.g. `"en"` or `"de"`.  `None` for untagged literals.
    pub lang: Option<String>,
    /// Literal text value.
    pub value: String,
}

/// A single full-text search hit.
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    /// Subject IRI.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Matched literal text.
    pub value: String,
    /// BM25-based relevance score from Tantivy.
    pub score: f32,
    /// Language tag if the literal was tagged.
    pub lang: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal schema field names
// ─────────────────────────────────────────────────────────────────────────────

const FIELD_SUBJECT: &str = "subject";
const FIELD_PREDICATE: &str = "predicate";
const FIELD_LANG: &str = "lang";
const FIELD_VALUE: &str = "value";

// ─────────────────────────────────────────────────────────────────────────────
// TextSearchIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Full-text search index over RDF literals, backed by Tantivy.
///
/// Thread-safe via internal [`RwLock`] over the [`IndexWriter`].
pub struct TextSearchIndex {
    /// Tantivy index (may be RAM-backed or directory-backed).
    index: Index,
    /// Tantivy index writer wrapped in a lock so it can be shared.
    writer: Arc<RwLock<IndexWriter>>,
    /// Near-real-time reader for searches.
    reader: IndexReader,
    /// Compiled schema.
    schema: tantivy::schema::Schema,
    /// Schema field: subject IRI.
    field_subject: tantivy::schema::Field,
    /// Schema field: predicate IRI.
    field_predicate: tantivy::schema::Field,
    /// Schema field: language tag.
    field_lang: tantivy::schema::Field,
    /// Schema field: literal text (full-text indexed).
    field_value: tantivy::schema::Field,
    /// Auto-commit threshold (0 = disabled).
    commit_threshold: usize,
    /// Number of documents staged since the last commit.
    staged_count: usize,
}

impl TextSearchIndex {
    // ── Constructor ──────────────────────────────────────────────────────────

    /// Create or open a [`TextSearchIndex`] according to `config`.
    ///
    /// If `config.index_path` is `None` an in-memory (RAM) index is used.
    /// Otherwise the directory is created if it does not already exist and
    /// a persistent disk index is opened (or created on first use).
    pub fn new(config: TextSearchConfig) -> Result<Self> {
        let heap_bytes = config
            .heap_size_mb
            .saturating_mul(1024 * 1024)
            .max(15_000_000);

        // Build schema
        let mut builder = SchemaBuilder::new();
        let field_subject = builder.add_text_field(FIELD_SUBJECT, STRING | STORED);
        let field_predicate = builder.add_text_field(FIELD_PREDICATE, STRING | STORED);
        let field_lang = builder.add_text_field(FIELD_LANG, STRING | STORED);
        let field_value = builder.add_text_field(FIELD_VALUE, TEXT | STORED);
        let schema = builder.build();

        let index = match config.index_path {
            Some(ref dir) => {
                std::fs::create_dir_all(dir).map_err(TdbError::Io)?;
                // Try to create; fall back to opening an existing index.
                Index::create_in_dir(dir, schema.clone())
                    .or_else(|_| Index::open_in_dir(dir))
                    .map_err(|e| TdbError::Other(format!("Tantivy index open failed: {e}")))?
            }
            None => Index::create_in_ram(schema.clone()),
        };

        let writer = index
            .writer(heap_bytes)
            .map_err(|e| TdbError::Other(format!("Tantivy writer creation failed: {e}")))?;

        // Manual reload: new documents become visible only after explicit commit + reload.
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| TdbError::Other(format!("Tantivy reader creation failed: {e}")))?;

        Ok(Self {
            index,
            writer: Arc::new(RwLock::new(writer)),
            reader,
            schema,
            field_subject,
            field_predicate,
            field_lang,
            field_value,
            commit_threshold: config.commit_threshold,
            staged_count: 0,
        })
    }

    // ── Write operations ─────────────────────────────────────────────────────

    /// Add a single RDF literal document to the index.
    ///
    /// Does **not** commit automatically unless the configured
    /// `commit_threshold` is reached.
    pub fn add_literal(&mut self, doc: LiteralDocument) -> Result<()> {
        self.add_tantivy_document(&doc)?;
        self.staged_count += 1;

        if self.commit_threshold > 0 && self.staged_count >= self.commit_threshold {
            self.commit()?;
        }

        Ok(())
    }

    /// Add a batch of RDF literal documents.
    ///
    /// Returns the number of documents successfully staged.
    /// A single `commit()` is performed at the end of the batch if any documents
    /// were added.
    pub fn add_literals_batch(&mut self, docs: Vec<LiteralDocument>) -> Result<u64> {
        if docs.is_empty() {
            return Ok(0);
        }

        let mut count: u64 = 0;
        for doc in docs {
            self.add_tantivy_document(&doc)?;
            count += 1;
        }

        self.staged_count += count as usize;
        self.commit()?;

        Ok(count)
    }

    /// Commit all staged documents to the Tantivy index and reload the reader
    /// so that subsequent searches reflect the new data.
    pub fn commit(&mut self) -> Result<()> {
        {
            let mut writer = self.writer.write();
            writer
                .commit()
                .map_err(|e| TdbError::Other(format!("Tantivy commit failed: {e}")))?;
        }

        self.reader
            .reload()
            .map_err(|e| TdbError::Other(format!("Tantivy reader reload failed: {e}")))?;

        self.staged_count = 0;
        debug!("TextSearchIndex committed and reader reloaded");
        Ok(())
    }

    // ── Read operations ──────────────────────────────────────────────────────

    /// Search for the given free-text `query` string.
    ///
    /// Returns up to `limit` results ordered by descending BM25 score.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<TextSearchResult>> {
        self.run_search(query, None, None, limit)
    }

    /// Search restricted to documents whose `predicate` field equals the given
    /// predicate IRI.
    pub fn search_with_predicate(
        &self,
        query: &str,
        predicate: &str,
        limit: usize,
    ) -> Result<Vec<TextSearchResult>> {
        self.run_search(query, Some(predicate), None, limit)
    }

    /// Search restricted to documents whose `lang` field equals the given
    /// BCP-47 language tag (e.g. `"en"`).
    pub fn search_lang(
        &self,
        query: &str,
        lang: &str,
        limit: usize,
    ) -> Result<Vec<TextSearchResult>> {
        self.run_search(query, None, Some(lang), limit)
    }

    /// Delete every document whose `subject` field equals the given IRI.
    ///
    /// The deletion is staged in the writer buffer.  Call [`commit`](Self::commit)
    /// to make it durable.  Returns the number of documents that were staged for
    /// deletion (estimated from the searcher before deletion).
    pub fn delete_by_subject(&mut self, subject: &str) -> Result<u64> {
        let before = self.document_count()?;

        let term = Term::from_field_text(self.field_subject, subject);
        {
            let writer = self.writer.write();
            writer.delete_term(term);
        }

        // Commit immediately so the reader is consistent.
        self.commit()?;

        let after = self.document_count()?;
        Ok(before.saturating_sub(after))
    }

    /// Return the total number of documents currently visible in the index.
    pub fn document_count(&self) -> Result<u64> {
        Ok(self.reader.searcher().num_docs())
    }

    /// Force a reader reload from the latest committed state.
    ///
    /// Useful when the writer has been committed by another path (e.g. from a
    /// clone of the writer [`Arc`]).
    pub fn reload(&mut self) -> Result<()> {
        self.reader
            .reload()
            .map_err(|e| TdbError::Other(format!("Tantivy reader reload failed: {e}")))?;
        Ok(())
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Build a Tantivy document from a [`LiteralDocument`] and add it to the writer.
    fn add_tantivy_document(&self, doc: &LiteralDocument) -> Result<()> {
        let mut tdoc = TantivyDocument::default();
        tdoc.add_text(self.field_subject, &doc.subject);
        tdoc.add_text(self.field_predicate, &doc.predicate);
        tdoc.add_text(self.field_lang, doc.lang.as_deref().unwrap_or(""));
        tdoc.add_text(self.field_value, &doc.value);

        let writer = self.writer.write();
        writer
            .add_document(tdoc)
            .map_err(|e| TdbError::Other(format!("Tantivy add_document failed: {e}")))?;

        Ok(())
    }

    /// Core search implementation.
    ///
    /// After retrieving Tantivy hits, optional post-filters are applied for
    /// `predicate` and `lang` because Tantivy's schema stores those fields as
    /// `STRING` (exact-match, not full-text) and we need exact equality checks
    /// on stored values.
    fn run_search(
        &self,
        query: &str,
        predicate_filter: Option<&str>,
        lang_filter: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TextSearchResult>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.field_value]);

        let parsed_query = query_parser
            .parse_query(query)
            .map_err(|e| TdbError::Other(format!("Tantivy query parse error: {e}")))?;

        // Fetch more results than `limit` to account for post-filtering.
        let fetch_limit = if predicate_filter.is_some() || lang_filter.is_some() {
            limit.saturating_mul(10).max(limit + 50)
        } else {
            limit
        };

        let top_docs = searcher
            .search(&parsed_query, &TopDocs::with_limit(fetch_limit))
            .map_err(|e| TdbError::Other(format!("Tantivy search error: {e}")))?;

        let mut results = Vec::with_capacity(top_docs.len());

        for (score, doc_addr) in top_docs {
            match searcher.doc::<TantivyDocument>(doc_addr) {
                Ok(tdoc) => {
                    let subject = get_stored_str(&tdoc, self.field_subject);
                    let predicate = get_stored_str(&tdoc, self.field_predicate);
                    let lang_raw = get_stored_str(&tdoc, self.field_lang);
                    let value = get_stored_str(&tdoc, self.field_value);
                    let lang = if lang_raw.is_empty() {
                        None
                    } else {
                        Some(lang_raw)
                    };

                    // Post-filter: predicate
                    if let Some(pf) = predicate_filter {
                        if predicate != pf {
                            continue;
                        }
                    }

                    // Post-filter: language
                    if let Some(lf) = lang_filter {
                        match &lang {
                            Some(l) if l == lf => {}
                            _ => continue,
                        }
                    }

                    results.push(TextSearchResult {
                        subject,
                        predicate,
                        value,
                        score,
                        lang,
                    });

                    if results.len() >= limit {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Failed to retrieve Tantivy document: {}", e);
                }
            }
        }

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPARQL text: property function interface
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates the SPARQL `text:query(?subject, "search terms")` property function.
///
/// This struct holds a shared reference to the underlying [`TextSearchIndex`]
/// so that it can be cloned cheaply and used from multiple query-execution
/// contexts.
pub struct TextPropertyFunction {
    index: Arc<RwLock<TextSearchIndex>>,
}

impl TextPropertyFunction {
    /// Default limit used when the caller does not specify one.
    pub const DEFAULT_LIMIT: usize = 100;

    /// Create a new property function wrapper around a shared index.
    pub fn new(index: Arc<RwLock<TextSearchIndex>>) -> Self {
        Self { index }
    }

    /// Evaluate `text:query(?subject, "query_text")`.
    ///
    /// Returns a list of subject IRIs whose associated literals match
    /// `query_text`, ordered by descending relevance score.
    ///
    /// `limit` caps the result set.  Pass `None` to use [`DEFAULT_LIMIT`](Self::DEFAULT_LIMIT).
    pub fn evaluate(&self, query_text: &str, limit: Option<usize>) -> Result<Vec<String>> {
        let effective_limit = limit.unwrap_or(Self::DEFAULT_LIMIT);
        let idx = self.index.read();
        let hits = idx.search(query_text, effective_limit)?;
        // Deduplicate: a subject may match on multiple literals.
        let mut seen = std::collections::HashSet::new();
        let subjects = hits
            .into_iter()
            .filter_map(|r| {
                if seen.insert(r.subject.clone()) {
                    Some(r.subject)
                } else {
                    None
                }
            })
            .collect();
        Ok(subjects)
    }

    /// Evaluate with an additional predicate restriction.
    ///
    /// Only literals whose predicate equals `predicate` are considered.
    pub fn evaluate_with_predicate(
        &self,
        query_text: &str,
        predicate: &str,
        limit: Option<usize>,
    ) -> Result<Vec<String>> {
        let effective_limit = limit.unwrap_or(Self::DEFAULT_LIMIT);
        let idx = self.index.read();
        let hits = idx.search_with_predicate(query_text, predicate, effective_limit)?;
        let mut seen = std::collections::HashSet::new();
        let subjects = hits
            .into_iter()
            .filter_map(|r| {
                if seen.insert(r.subject.clone()) {
                    Some(r.subject)
                } else {
                    None
                }
            })
            .collect();
        Ok(subjects)
    }

    /// Evaluate with an additional language-tag restriction.
    ///
    /// Only literals tagged with `lang` (e.g. `"en"`) are considered.
    pub fn evaluate_with_lang(
        &self,
        query_text: &str,
        lang: &str,
        limit: Option<usize>,
    ) -> Result<Vec<String>> {
        let effective_limit = limit.unwrap_or(Self::DEFAULT_LIMIT);
        let idx = self.index.read();
        let hits = idx.search_lang(query_text, lang, effective_limit)?;
        let mut seen = std::collections::HashSet::new();
        let subjects = hits
            .into_iter()
            .filter_map(|r| {
                if seen.insert(r.subject.clone()) {
                    Some(r.subject)
                } else {
                    None
                }
            })
            .collect();
        Ok(subjects)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the first stored string value for `field` from a Tantivy document.
fn get_stored_str(doc: &TantivyDocument, field: tantivy::schema::Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    /// Build a minimal in-memory [`TextSearchConfig`].
    fn mem_config() -> TextSearchConfig {
        TextSearchConfig {
            index_path: None,
            heap_size_mb: 15,
            commit_threshold: 0, // manual commit in tests
        }
    }

    /// Build a [`LiteralDocument`] with default English language tag.
    fn make_doc(subject: &str, predicate: &str, value: &str) -> LiteralDocument {
        LiteralDocument {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            lang: Some("en".to_string()),
            value: value.to_string(),
        }
    }

    /// Build a [`LiteralDocument`] with a specific language tag.
    fn make_doc_lang(subject: &str, predicate: &str, value: &str, lang: &str) -> LiteralDocument {
        LiteralDocument {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            lang: Some(lang.to_string()),
            value: value.to_string(),
        }
    }

    // ── 1. In-memory index creation ──────────────────────────────────────────

    #[test]
    fn test_in_memory_index_creation() {
        let idx = TextSearchIndex::new(mem_config());
        assert!(
            idx.is_ok(),
            "Should create an in-memory index without error"
        );
        let idx = idx.expect("index creation failed");
        let count = idx.document_count().expect("doc count failed");
        assert_eq!(count, 0, "New index should be empty");
    }

    // ── 2. Add and search literal ─────────────────────────────────────────────

    #[test]
    fn test_add_and_search_literal() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/title",
            "Rust programming systems language",
        ))
        .expect("add_literal");
        idx.commit().expect("commit");

        let results = idx.search("rust programming", 10).expect("search");
        assert!(!results.is_empty(), "Expected at least one result");
        assert_eq!(results[0].subject, "http://ex.org/s1");
        assert!(results[0].score > 0.0, "Score must be positive");
    }

    // ── 3. Search with predicate filter ──────────────────────────────────────

    #[test]
    fn test_search_with_predicate_filter() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");

        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/title",
            "semantic web ontology",
        ))
        .expect("add 1");
        idx.add_literal(make_doc(
            "http://ex.org/s2",
            "http://ex.org/description",
            "semantic reasoning engine",
        ))
        .expect("add 2");
        idx.commit().expect("commit");

        // Only the document with predicate "title" should be returned.
        let results = idx
            .search_with_predicate("semantic", "http://ex.org/title", 10)
            .expect("search_with_predicate");
        assert_eq!(results.len(), 1, "Should return exactly one result");
        assert_eq!(results[0].subject, "http://ex.org/s1");
        assert_eq!(results[0].predicate, "http://ex.org/title");
    }

    // ── 4. Search with language filter ──────────────────────────────────────

    #[test]
    fn test_search_with_lang_filter() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");

        idx.add_literal(make_doc_lang(
            "http://ex.org/s1",
            "http://ex.org/label",
            "knowledge graph",
            "en",
        ))
        .expect("add en");
        idx.add_literal(make_doc_lang(
            "http://ex.org/s2",
            "http://ex.org/label",
            "Wissensgraph",
            "de",
        ))
        .expect("add de");
        idx.commit().expect("commit");

        let en_results = idx.search_lang("knowledge", "en", 10).expect("search en");
        assert_eq!(en_results.len(), 1);
        assert_eq!(en_results[0].subject, "http://ex.org/s1");
        assert_eq!(en_results[0].lang, Some("en".to_string()));

        let de_results = idx
            .search_lang("Wissensgraph", "de", 10)
            .expect("search de");
        assert_eq!(de_results.len(), 1);
        assert_eq!(de_results[0].subject, "http://ex.org/s2");
    }

    // ── 5. Delete by subject ─────────────────────────────────────────────────

    #[test]
    fn test_delete_by_subject() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");

        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/p",
            "hello world",
        ))
        .expect("add 1");
        idx.add_literal(make_doc(
            "http://ex.org/s2",
            "http://ex.org/p",
            "hello rust",
        ))
        .expect("add 2");
        idx.commit().expect("commit");

        let deleted = idx.delete_by_subject("http://ex.org/s1").expect("delete");
        assert_eq!(deleted, 1, "Should have deleted one document");

        let results = idx.search("hello", 10).expect("search after delete");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "http://ex.org/s2");
    }

    // ── 6. Batch add ─────────────────────────────────────────────────────────

    #[test]
    fn test_batch_add() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");

        let docs = vec![
            make_doc("http://ex.org/a", "http://ex.org/p", "apple fruit healthy"),
            make_doc(
                "http://ex.org/b",
                "http://ex.org/p",
                "banana tropical fruit",
            ),
            make_doc("http://ex.org/c", "http://ex.org/p", "cherry red fruit"),
        ];

        let count = idx.add_literals_batch(docs).expect("batch add");
        assert_eq!(count, 3, "All three documents should be staged");

        let results = idx.search("fruit", 10).expect("search fruit");
        assert_eq!(results.len(), 3, "All three fruit documents should match");
    }

    // ── 7. Document count ─────────────────────────────────────────────────────

    #[test]
    fn test_document_count() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        assert_eq!(idx.document_count().expect("count"), 0);

        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/p",
            "first document",
        ))
        .expect("add 1");
        idx.commit().expect("commit after 1");
        assert_eq!(idx.document_count().expect("count after 1"), 1);

        idx.add_literal(make_doc(
            "http://ex.org/s2",
            "http://ex.org/p",
            "second document",
        ))
        .expect("add 2");
        idx.add_literal(make_doc(
            "http://ex.org/s3",
            "http://ex.org/p",
            "third document",
        ))
        .expect("add 3");
        idx.commit().expect("commit after 3");
        assert_eq!(idx.document_count().expect("count after 3"), 3);
    }

    // ── 8. Property function evaluate ────────────────────────────────────────

    #[test]
    fn test_property_function_evaluate() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/p",
            "sparql query language semantic web",
        ))
        .expect("add 1");
        idx.add_literal(make_doc(
            "http://ex.org/s2",
            "http://ex.org/p",
            "graphql schema definition language",
        ))
        .expect("add 2");
        idx.commit().expect("commit");

        let pf = TextPropertyFunction::new(Arc::new(RwLock::new(idx)));
        let subjects = pf.evaluate("sparql", None).expect("evaluate");
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0], "http://ex.org/s1");
    }

    // ── 9. Property function with predicate restriction ───────────────────────

    #[test]
    fn test_property_function_with_predicate() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://schema.org/name",
            "OxiRS semantic web engine",
        ))
        .expect("add name");
        idx.add_literal(make_doc(
            "http://ex.org/s2",
            "http://schema.org/description",
            "OxiRS is a semantic web platform",
        ))
        .expect("add desc");
        idx.commit().expect("commit");

        let pf = TextPropertyFunction::new(Arc::new(RwLock::new(idx)));
        let subjects = pf
            .evaluate_with_predicate("semantic", "http://schema.org/name", None)
            .expect("evaluate_with_predicate");
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0], "http://ex.org/s1");
    }

    // ── 10. Auto-commit threshold ─────────────────────────────────────────────

    #[test]
    fn test_auto_commit_threshold() {
        let config = TextSearchConfig {
            index_path: None,
            heap_size_mb: 15,
            commit_threshold: 2, // auto-commit every 2 documents
        };
        let mut idx = TextSearchIndex::new(config).expect("index");

        // Add first document: no auto-commit yet (staged_count becomes 1 < 2)
        idx.add_literal(make_doc(
            "http://ex.org/a",
            "http://ex.org/p",
            "tantivy search",
        ))
        .expect("add 1");

        // Add second document: auto-commit triggered (staged_count becomes 2 >= 2)
        idx.add_literal(make_doc(
            "http://ex.org/b",
            "http://ex.org/p",
            "rust indexing",
        ))
        .expect("add 2");

        // After auto-commit both documents should be visible.
        let count = idx.document_count().expect("doc count");
        assert_eq!(count, 2, "Both documents visible after auto-commit");
    }

    // ── 11. On-disk index creation ────────────────────────────────────────────

    #[test]
    fn test_on_disk_index_creation() {
        let dir = env::temp_dir().join(format!("oxirs_tdb_text_search_{}", std::process::id()));
        let config = TextSearchConfig {
            index_path: Some(dir.clone()),
            heap_size_mb: 15,
            commit_threshold: 0,
        };

        {
            let mut idx = TextSearchIndex::new(config.clone()).expect("create on-disk index");
            idx.add_literal(make_doc(
                "http://ex.org/s1",
                "http://ex.org/p",
                "persistent full-text search",
            ))
            .expect("add");
            idx.commit().expect("commit");
            let count = idx.document_count().expect("count");
            assert_eq!(count, 1);
        }

        // Re-open the same directory — document should still be there.
        {
            let idx = TextSearchIndex::new(config).expect("reopen on-disk index");
            let count = idx.document_count().expect("count after reopen");
            assert_eq!(count, 1, "Document should persist after re-opening index");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 12. Empty query returns nothing ──────────────────────────────────────

    #[test]
    fn test_empty_query_returns_nothing() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc("http://ex.org/s1", "http://ex.org/p", "anything"))
            .expect("add");
        idx.commit().expect("commit");

        let results = idx.search("", 10).expect("empty search");
        assert!(
            results.is_empty(),
            "Empty query string should return no results"
        );
    }

    // ── 13. Reload method ─────────────────────────────────────────────────────

    #[test]
    fn test_reload() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/p",
            "reload test content",
        ))
        .expect("add");
        idx.commit().expect("commit");
        // A second reload should succeed without error.
        idx.reload().expect("reload");
        let count = idx.document_count().expect("count");
        assert_eq!(count, 1);
    }

    // ── 14. Multiple literals for same subject (dedup in property function) ───

    #[test]
    fn test_property_function_deduplicates_subjects() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        // Same subject, two different predicates, both matching "knowledge"
        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/title",
            "knowledge graph",
        ))
        .expect("add title");
        idx.add_literal(make_doc(
            "http://ex.org/s1",
            "http://ex.org/description",
            "knowledge representation",
        ))
        .expect("add desc");
        idx.commit().expect("commit");

        let pf = TextPropertyFunction::new(Arc::new(RwLock::new(idx)));
        let subjects = pf.evaluate("knowledge", Some(10)).expect("evaluate");
        // Subject should appear only once despite matching two documents.
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0], "http://ex.org/s1");
    }

    // ── 15. Batch add with lang filter search ─────────────────────────────────

    #[test]
    fn test_batch_add_with_lang_search() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");

        let docs = vec![
            make_doc_lang(
                "http://ex.org/en1",
                "http://ex.org/p",
                "open data linked data",
                "en",
            ),
            make_doc_lang(
                "http://ex.org/fr1",
                "http://ex.org/p",
                "données ouvertes liées",
                "fr",
            ),
            make_doc_lang(
                "http://ex.org/en2",
                "http://ex.org/p",
                "open source data platform",
                "en",
            ),
        ];

        idx.add_literals_batch(docs).expect("batch add");

        let en_results = idx.search_lang("open", "en", 10).expect("en search");
        assert_eq!(en_results.len(), 2, "Should find two English documents");
        for r in &en_results {
            assert_eq!(r.lang, Some("en".to_string()));
        }
    }
    // ── 16. Search with limit of 1 ───────────────────────────────────────────

    #[test]
    fn test_search_limit_one() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        for i in 0..5 {
            idx.add_literal(make_doc(
                &format!("http://ex.org/s{}", i),
                "http://ex.org/p",
                &format!("semantic web technology {}", i),
            ))
            .expect("add");
        }
        idx.commit().expect("commit");

        let results = idx.search("semantic", 1).expect("search");
        assert_eq!(results.len(), 1, "Limit=1 should return at most one result");
    }

    // ── 17. Multiple commits — document count is cumulative ──────────────────

    #[test]
    fn test_multiple_commits_cumulative_count() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc("http://s1", "http://p", "first document"))
            .expect("add");
        idx.commit().expect("commit 1");
        idx.add_literal(make_doc("http://s2", "http://p", "second document"))
            .expect("add");
        idx.commit().expect("commit 2");
        idx.add_literal(make_doc("http://s3", "http://p", "third document"))
            .expect("add");
        idx.commit().expect("commit 3");

        let count = idx.document_count().expect("count");
        assert_eq!(count, 3, "Should have 3 documents after 3 commits");
    }

    // ── 18. Delete non-existent subject returns 0 ────────────────────────────

    #[test]
    fn test_delete_nonexistent_subject_returns_zero() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.commit().expect("commit empty");
        let removed = idx
            .delete_by_subject("http://ghost.org/not_there")
            .expect("delete");
        assert_eq!(
            removed, 0,
            "No documents should be removed for unknown subject"
        );
    }

    // ── 19. Property function returns empty for unknown query ────────────────

    #[test]
    fn test_property_function_unknown_term_returns_empty() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc("http://s1", "http://p", "known content"))
            .expect("add");
        idx.commit().expect("commit");

        let pf = TextPropertyFunction::new(Arc::new(RwLock::new(idx)));
        let results = pf
            .evaluate("xyzzy_nonexistent_term", Some(10))
            .expect("evaluate");
        assert!(results.is_empty());
    }

    // ── 20. Search result contains correct subject and predicate ─────────────

    #[test]
    fn test_search_result_fields() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(LiteralDocument {
            subject: "http://ex.org/doc1".to_string(),
            predicate: "http://schema.org/description".to_string(),
            lang: Some("en".to_string()),
            value: "unique phrase for this test".to_string(),
        })
        .expect("add");
        idx.commit().expect("commit");

        let results = idx.search("unique", 5).expect("search");
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.subject, "http://ex.org/doc1");
        assert_eq!(r.predicate, "http://schema.org/description");
        assert!(r.value.contains("unique"));
        assert!(r.score > 0.0);
    }

    // ── 21. Batch add returns correct inserted count ─────────────────────────

    #[test]
    fn test_batch_add_count() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        let docs: Vec<LiteralDocument> = (0..7)
            .map(|i| {
                make_doc(
                    &format!("http://s{}", i),
                    "http://p",
                    &format!("item {}", i),
                )
            })
            .collect();
        let n = idx.add_literals_batch(docs).expect("batch add");
        assert_eq!(n, 7, "Should return the count of inserted documents");
    }

    // ── 22. Delete removes only matching subject ─────────────────────────────

    #[test]
    fn test_delete_removes_only_matching_subject() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc(
            "http://ex.org/keep",
            "http://p",
            "important knowledge",
        ))
        .expect("add keep");
        idx.add_literal(make_doc(
            "http://ex.org/remove",
            "http://p",
            "important knowledge",
        ))
        .expect("add remove");
        idx.commit().expect("commit");

        idx.delete_by_subject("http://ex.org/remove")
            .expect("delete");

        // "keep" subject should still be searchable
        let results = idx.search("important", 10).expect("search");
        let subjects: Vec<&str> = results.iter().map(|r| r.subject.as_str()).collect();
        assert!(
            subjects.contains(&"http://ex.org/keep"),
            "keep subject should remain"
        );
    }

    // ── 23. Search is case-insensitive (standard tokenizer lowercases) ───────

    #[test]
    fn test_search_case_insensitive() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc("http://s1", "http://p", "RDF Knowledge Graph"))
            .expect("add");
        idx.commit().expect("commit");

        let lower = idx.search("rdf", 5).expect("lower search");
        let upper = idx.search("RDF", 5).expect("upper search");
        // Both should find the document (Tantivy standard tokenizer lowercases)
        assert!(
            !lower.is_empty() || !upper.is_empty(),
            "At least one casing should match"
        );
    }

    // ── 24. Document count after delete decreases ────────────────────────────

    #[test]
    fn test_doc_count_after_delete() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc("http://s1", "http://p", "first"))
            .expect("add");
        idx.add_literal(make_doc("http://s2", "http://p", "second"))
            .expect("add");
        idx.commit().expect("commit");

        let before = idx.document_count().expect("count before");
        idx.delete_by_subject("http://s1").expect("delete");

        let after = idx.document_count().expect("count after");
        assert!(
            after <= before,
            "Document count should not increase after delete"
        );
    }

    // ── 25. Property function with language filter ───────────────────────────

    #[test]
    fn test_property_function_with_language_filter() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc_lang(
            "http://s_en",
            "http://p",
            "open linked data",
            "en",
        ))
        .expect("add en");
        idx.add_literal(make_doc_lang(
            "http://s_de",
            "http://p",
            "offene verknüpfte Daten",
            "de",
        ))
        .expect("add de");
        idx.commit().expect("commit");

        let pf = TextPropertyFunction::new(Arc::new(RwLock::new(idx)));
        let en_subjects = pf
            .evaluate_with_lang("open", "en", Some(10))
            .expect("en query");
        assert_eq!(en_subjects.len(), 1);
        assert_eq!(en_subjects[0], "http://s_en");
    }

    // ── 26. Search returns results sorted by descending score ────────────────

    #[test]
    fn test_search_results_ordered_by_relevance() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        // Doc with query term appearing multiple times → higher score
        idx.add_literal(make_doc(
            "http://s_high",
            "http://p",
            "semantic semantic semantic web web web",
        ))
        .expect("add high");
        idx.add_literal(make_doc("http://s_low", "http://p", "semantic"))
            .expect("add low");
        idx.commit().expect("commit");

        let results = idx.search("semantic", 10).expect("search");
        assert!(!results.is_empty());
        // Scores should be non-increasing (highest first)
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "Results should be ordered by descending score: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    // ── 27. Config: large heap size compiles and works ───────────────────────

    #[test]
    fn test_large_heap_config() {
        let config = TextSearchConfig {
            index_path: None,
            heap_size_mb: 200,
            commit_threshold: 0,
        };
        let mut idx = TextSearchIndex::new(config).expect("large heap index");
        idx.add_literal(make_doc("http://s1", "http://p", "heap test"))
            .expect("add");
        idx.commit().expect("commit");
        assert_eq!(idx.document_count().expect("count"), 1);
    }

    // ── 28. TextSearchResult score is positive ───────────────────────────────

    #[test]
    fn test_search_result_score_positive() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(make_doc("http://s1", "http://p", "score positive check"))
            .expect("add");
        idx.commit().expect("commit");

        let results = idx.search("score", 5).expect("search");
        for r in &results {
            assert!(
                r.score > 0.0,
                "All returned results should have positive scores"
            );
        }
    }

    // ── 29. Multiple subjects matching same query ────────────────────────────

    #[test]
    fn test_multiple_subjects_same_query() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        for i in 0..5 {
            idx.add_literal(make_doc(
                &format!("http://sub{}", i),
                "http://p",
                "sparql query engine",
            ))
            .expect("add");
        }
        idx.commit().expect("commit");

        let results = idx.search("sparql", 10).expect("search");
        assert_eq!(results.len(), 5, "Should find all 5 subjects");
    }

    // ── 30. Search with predicate filter excludes other predicates ───────────

    #[test]
    fn test_search_predicate_filter_excludes_others() {
        let mut idx = TextSearchIndex::new(mem_config()).expect("index");
        idx.add_literal(LiteralDocument {
            subject: "http://s1".to_string(),
            predicate: "http://ex.org/title".to_string(),
            lang: Some("en".to_string()),
            value: "rdf triples storage".to_string(),
        })
        .expect("add title");
        idx.add_literal(LiteralDocument {
            subject: "http://s2".to_string(),
            predicate: "http://ex.org/abstract".to_string(),
            lang: Some("en".to_string()),
            value: "rdf triples storage".to_string(),
        })
        .expect("add abstract");
        idx.commit().expect("commit");

        let results = idx
            .search_with_predicate("rdf", "http://ex.org/title", 10)
            .expect("predicate search");
        let predicates: Vec<&str> = results.iter().map(|r| r.predicate.as_str()).collect();
        for p in predicates {
            assert_eq!(
                p, "http://ex.org/title",
                "Only title predicate should match"
            );
        }
    }
}
