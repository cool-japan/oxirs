//! Keyword search implementations (BM25, TF-IDF)

use super::types::KeywordMatch;
use std::collections::HashMap;

/// Keyword search interface
pub trait KeywordSearcher: Send + Sync {
    /// Search for documents matching the query
    fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<KeywordMatch>>;

    /// Add a document to the index
    fn add_document(&mut self, doc_id: &str, content: &str) -> anyhow::Result<()>;

    /// Get document count
    fn document_count(&self) -> usize;
}

/// BM25 scorer for keyword search
pub struct Bm25Scorer {
    /// Document term frequencies: doc_id -> (term -> frequency)
    documents: HashMap<String, HashMap<String, usize>>,
    /// Document lengths
    doc_lengths: HashMap<String, usize>,
    /// Average document length
    avg_doc_length: f32,
    /// Document frequency: term -> number of documents containing term
    doc_frequency: HashMap<String, usize>,
    /// BM25 k1 parameter (term frequency saturation)
    k1: f32,
    /// BM25 b parameter (length normalization)
    b: f32,
}

impl Bm25Scorer {
    /// Create a new BM25 scorer
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_length: 0.0,
            doc_frequency: HashMap::new(),
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Tokenize text into terms
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Calculate term frequency map
    fn calculate_term_frequencies(&self, terms: &[String]) -> HashMap<String, usize> {
        let mut freqs = HashMap::new();
        for term in terms {
            *freqs.entry(term.clone()).or_insert(0) += 1;
        }
        freqs
    }

    /// Update average document length
    fn update_avg_doc_length(&mut self) {
        if !self.doc_lengths.is_empty() {
            let total: usize = self.doc_lengths.values().sum();
            self.avg_doc_length = total as f32 / self.doc_lengths.len() as f32;
        }
    }

    /// Calculate BM25 score for a term in a document
    fn bm25_score(&self, term: &str, doc_tf: usize, doc_length: usize) -> f32 {
        let n = self.documents.len() as f32;
        let df = self.doc_frequency.get(term).copied().unwrap_or(0) as f32;

        // IDF component
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        // TF component with saturation and length normalization
        let tf = doc_tf as f32;
        let norm_length = doc_length as f32 / self.avg_doc_length.max(1.0);
        let tf_component =
            (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * norm_length));

        idf * tf_component
    }
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self::new()
    }
}

impl KeywordSearcher for Bm25Scorer {
    fn add_document(&mut self, doc_id: &str, content: &str) -> anyhow::Result<()> {
        let terms = self.tokenize(content);
        let term_freqs = self.calculate_term_frequencies(&terms);

        // Update document frequency
        for term in term_freqs.keys() {
            *self.doc_frequency.entry(term.clone()).or_insert(0) += 1;
        }

        self.doc_lengths.insert(doc_id.to_string(), terms.len());
        self.documents.insert(doc_id.to_string(), term_freqs);
        self.update_avg_doc_length();

        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<KeywordMatch>> {
        let query_terms = self.tokenize(query);

        let mut results: Vec<KeywordMatch> = self
            .documents
            .iter()
            .map(|(doc_id, term_freqs)| {
                let doc_length = self.doc_lengths.get(doc_id).copied().unwrap_or(0);
                let mut score = 0.0;
                let mut matched_terms = Vec::new();

                for query_term in &query_terms {
                    if let Some(&tf) = term_freqs.get(query_term) {
                        score += self.bm25_score(query_term, tf, doc_length);
                        matched_terms.push(query_term.clone());
                    }
                }

                KeywordMatch {
                    doc_id: doc_id.clone(),
                    score,
                    matched_terms,
                    term_frequencies: term_freqs.clone(),
                }
            })
            .filter(|m| m.score > 0.0)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);

        Ok(results)
    }

    fn document_count(&self) -> usize {
        self.documents.len()
    }
}

/// TF-IDF scorer for keyword search
pub struct TfidfScorer {
    /// Document term frequencies
    documents: HashMap<String, HashMap<String, usize>>,
    /// Document lengths
    doc_lengths: HashMap<String, usize>,
    /// Document frequency
    doc_frequency: HashMap<String, usize>,
}

impl TfidfScorer {
    /// Create a new TF-IDF scorer
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_frequency: HashMap::new(),
        }
    }

    /// Tokenize text
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    /// Calculate term frequencies
    fn calculate_term_frequencies(&self, terms: &[String]) -> HashMap<String, usize> {
        let mut freqs = HashMap::new();
        for term in terms {
            *freqs.entry(term.clone()).or_insert(0) += 1;
        }
        freqs
    }

    /// Calculate TF-IDF score
    fn tfidf_score(&self, term: &str, tf: usize, doc_length: usize) -> f32 {
        let n = self.documents.len() as f32;
        let df = self.doc_frequency.get(term).copied().unwrap_or(0) as f32;

        // TF component (normalized by document length)
        let tf_normalized = tf as f32 / doc_length.max(1) as f32;

        // IDF component
        let idf = (n / (df + 1.0)).ln();

        tf_normalized * idf
    }
}

impl Default for TfidfScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl KeywordSearcher for TfidfScorer {
    fn add_document(&mut self, doc_id: &str, content: &str) -> anyhow::Result<()> {
        let terms = self.tokenize(content);
        let term_freqs = self.calculate_term_frequencies(&terms);

        // Update document frequency
        for term in term_freqs.keys() {
            *self.doc_frequency.entry(term.clone()).or_insert(0) += 1;
        }

        self.doc_lengths.insert(doc_id.to_string(), terms.len());
        self.documents.insert(doc_id.to_string(), term_freqs);

        Ok(())
    }

    fn search(&self, query: &str, top_k: usize) -> anyhow::Result<Vec<KeywordMatch>> {
        let query_terms = self.tokenize(query);

        let mut results: Vec<KeywordMatch> = self
            .documents
            .iter()
            .map(|(doc_id, term_freqs)| {
                let doc_length = self.doc_lengths.get(doc_id).copied().unwrap_or(0);
                let mut score = 0.0;
                let mut matched_terms = Vec::new();

                for query_term in &query_terms {
                    if let Some(&tf) = term_freqs.get(query_term) {
                        score += self.tfidf_score(query_term, tf, doc_length);
                        matched_terms.push(query_term.clone());
                    }
                }

                KeywordMatch {
                    doc_id: doc_id.clone(),
                    score,
                    matched_terms,
                    term_frequencies: term_freqs.clone(),
                }
            })
            .filter(|m| m.score > 0.0)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);

        Ok(results)
    }

    fn document_count(&self) -> usize {
        self.documents.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let mut bm25 = Bm25Scorer::new();

        bm25.add_document("doc1", "the quick brown fox").unwrap();
        bm25.add_document("doc2", "the lazy dog").unwrap();
        bm25.add_document("doc3", "quick brown dogs").unwrap();

        let results = bm25.search("quick brown", 2).unwrap();
        assert!(!results.is_empty());
        // doc3 should rank higher as it has both terms without common words
        assert!(results[0].doc_id == "doc3" || results[0].doc_id == "doc1");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tfidf_basic() {
        let mut tfidf = TfidfScorer::new();

        tfidf.add_document("doc1", "machine learning").unwrap();
        tfidf
            .add_document("doc2", "deep learning networks")
            .unwrap();
        tfidf
            .add_document("doc3", "natural language processing")
            .unwrap();

        let results = tfidf.search("machine learning", 2).unwrap();
        assert!(!results.is_empty());
        // doc1 should have the highest score for exact match
        assert_eq!(results[0].doc_id, "doc1");
    }

    #[test]
    fn test_tokenization() {
        let bm25 = Bm25Scorer::new();
        let tokens = bm25.tokenize("Hello, World! How are you?");
        assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
    }

    #[test]
    fn test_no_matches() {
        let mut bm25 = Bm25Scorer::new();
        bm25.add_document("doc1", "foo bar baz").unwrap();

        let results = bm25.search("xyz", 10).unwrap();
        assert!(results.is_empty());
    }
}
