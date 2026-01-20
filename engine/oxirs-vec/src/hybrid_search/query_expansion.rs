//! Query expansion for improved recall

use std::collections::{HashMap, HashSet};

/// Query expander for improving recall
pub struct QueryExpander {
    /// Synonym map: term -> synonyms
    synonyms: HashMap<String, Vec<String>>,
    /// Maximum expanded terms
    max_expanded_terms: usize,
}

impl QueryExpander {
    /// Create a new query expander
    pub fn new(max_expanded_terms: usize) -> Self {
        Self {
            synonyms: Self::build_default_synonyms(),
            max_expanded_terms,
        }
    }

    /// Build default synonym dictionary
    fn build_default_synonyms() -> HashMap<String, Vec<String>> {
        let mut synonyms = HashMap::new();

        // Common synonyms for search
        synonyms.insert(
            "search".to_string(),
            vec![
                "find".to_string(),
                "lookup".to_string(),
                "query".to_string(),
            ],
        );
        synonyms.insert(
            "find".to_string(),
            vec!["search".to_string(), "locate".to_string()],
        );
        synonyms.insert(
            "fast".to_string(),
            vec![
                "quick".to_string(),
                "rapid".to_string(),
                "speedy".to_string(),
            ],
        );
        synonyms.insert(
            "slow".to_string(),
            vec!["sluggish".to_string(), "gradual".to_string()],
        );
        synonyms.insert(
            "big".to_string(),
            vec![
                "large".to_string(),
                "huge".to_string(),
                "massive".to_string(),
            ],
        );
        synonyms.insert(
            "small".to_string(),
            vec![
                "tiny".to_string(),
                "little".to_string(),
                "compact".to_string(),
            ],
        );
        synonyms.insert(
            "good".to_string(),
            vec![
                "great".to_string(),
                "excellent".to_string(),
                "superb".to_string(),
            ],
        );
        synonyms.insert(
            "bad".to_string(),
            vec!["poor".to_string(), "terrible".to_string()],
        );

        synonyms
    }

    /// Add synonyms for a term
    pub fn add_synonyms(&mut self, term: &str, synonyms: Vec<String>) {
        self.synonyms.insert(term.to_string(), synonyms);
    }

    /// Expand a query with synonyms
    pub fn expand(&self, query: &str) -> Vec<String> {
        let original_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        let mut expanded = HashSet::new();

        // Add original terms
        for term in &original_terms {
            expanded.insert(term.clone());
        }

        // Add synonyms
        for term in &original_terms {
            if let Some(syns) = self.synonyms.get(term) {
                for syn in syns {
                    if expanded.len() < self.max_expanded_terms {
                        expanded.insert(syn.clone());
                    }
                }
            }
        }

        expanded.into_iter().collect()
    }

    /// Expand with co-occurrence based expansion
    pub fn expand_with_cooccurrence(
        &self,
        query: &str,
        cooccurrence_map: &HashMap<String, Vec<(String, f32)>>,
        threshold: f32,
    ) -> Vec<String> {
        let original_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        let mut expanded = HashSet::new();

        // Add original terms
        for term in &original_terms {
            expanded.insert(term.clone());
        }

        // Add co-occurring terms
        for term in &original_terms {
            if let Some(cooccurrences) = cooccurrence_map.get(term) {
                for (coterm, score) in cooccurrences {
                    if *score >= threshold && expanded.len() < self.max_expanded_terms {
                        expanded.insert(coterm.clone());
                    }
                }
            }
        }

        expanded.into_iter().collect()
    }

    /// Get synonym count
    pub fn synonym_count(&self) -> usize {
        self.synonyms.len()
    }
}

impl Default for QueryExpander {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_expansion() {
        let expander = QueryExpander::new(10);
        let expanded = expander.expand("fast search");

        assert!(expanded.contains(&"fast".to_string()));
        assert!(expanded.contains(&"search".to_string()));
        // Should have synonyms
        assert!(expanded.len() > 2);
    }

    #[test]
    fn test_max_expansion_limit() {
        let expander = QueryExpander::new(3);
        let expanded = expander.expand("fast search");

        assert!(expanded.len() <= 3);
    }

    #[test]
    fn test_custom_synonyms() {
        let mut expander = QueryExpander::new(10);
        expander.add_synonyms("ml", vec!["machine learning".to_string(), "ai".to_string()]);

        let expanded = expander.expand("ml");
        assert!(expanded.contains(&"ml".to_string()));
    }

    #[test]
    fn test_cooccurrence_expansion() {
        let expander = QueryExpander::new(10);
        let mut cooccurrence = HashMap::new();
        cooccurrence.insert(
            "machine".to_string(),
            vec![
                ("learning".to_string(), 0.9),
                ("intelligence".to_string(), 0.7),
                ("car".to_string(), 0.2),
            ],
        );

        let expanded = expander.expand_with_cooccurrence("machine", &cooccurrence, 0.5);

        assert!(expanded.contains(&"machine".to_string()));
        assert!(expanded.contains(&"learning".to_string()));
        assert!(expanded.contains(&"intelligence".to_string()));
        assert!(!expanded.contains(&"car".to_string())); // Below threshold
    }

    #[test]
    fn test_empty_query() {
        let expander = QueryExpander::new(10);
        let expanded = expander.expand("");
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_unknown_terms() {
        let expander = QueryExpander::new(10);
        let expanded = expander.expand("zzz xyz abc");

        // Should return original terms even without synonyms
        assert_eq!(expanded.len(), 3);
    }
}
