//! # Query Expansion using Graph Context and Synonyms
//!
//! Expands a query string with semantically related terms discovered from a
//! synonym lexicon and an in-memory RDF-like graph context.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Source from which an expansion term was derived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpansionSource {
    /// Explicitly registered synonym.
    Synonym,
    /// A broader/more general term.
    Hypernym,
    /// A narrower/more specific term.
    Hyponym,
    /// Entity related through any predicate in the graph context.
    RelatedEntity,
    /// Term that frequently appears near the query term in triples.
    CoOccurrence,
}

/// A single expansion term together with its relevance score and origin.
#[derive(Debug, Clone)]
pub struct ExpansionTerm {
    /// The expanded term.
    pub term: String,
    /// Relevance score in \[0, 1\].
    pub score: f64,
    /// Where this expansion came from.
    pub source: ExpansionSource,
}

/// The result of expanding a query.
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// The original query string.
    pub original: String,
    /// All expansion terms with scores and sources.
    pub expansions: Vec<ExpansionTerm>,
    /// Deduplicated, ordered list of terms to include in the expanded query.
    pub expanded_terms: Vec<String>,
}

/// Lightweight in-memory triple store used as graph context.
#[derive(Debug, Clone, Default)]
pub struct GraphContext {
    /// (subject, predicate, object) triples.
    pub triples: Vec<(String, String, String)>,
}

impl GraphContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the context.
    pub fn add_triple(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) {
        self.triples
            .push((subject.into(), predicate.into(), object.into()));
    }
}

// ---------------------------------------------------------------------------
// QueryExpander
// ---------------------------------------------------------------------------

/// Expands queries using a synonym lexicon and an RDF-like graph context.
pub struct QueryExpander {
    /// term → list of synonyms
    synonyms: HashMap<String, Vec<String>>,
    /// Graph context for entity-based expansions.
    context: GraphContext,
    /// Maximum number of expansion terms to return.
    max_expansions: usize,
}

impl QueryExpander {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new expander with the given expansion limit.
    pub fn new(max_expansions: usize) -> Self {
        Self {
            synonyms: HashMap::new(),
            context: GraphContext::new(),
            max_expansions: max_expansions.max(1),
        }
    }

    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    /// Register `synonym` as a synonym of `term` (bi-directional).
    pub fn add_synonym(&mut self, term: &str, synonym: &str) {
        self.synonyms
            .entry(term.to_lowercase())
            .or_default()
            .push(synonym.to_lowercase());
        // Also add the reverse mapping so expansion is symmetric
        self.synonyms
            .entry(synonym.to_lowercase())
            .or_default()
            .push(term.to_lowercase());
    }

    /// Replace the current graph context.
    pub fn set_context(&mut self, context: GraphContext) {
        self.context = context;
    }

    // ------------------------------------------------------------------
    // Scoring helper
    // ------------------------------------------------------------------

    /// Base relevance score assigned to terms from each source.
    pub fn score_expansion(source: &ExpansionSource) -> f64 {
        match source {
            ExpansionSource::Synonym => 0.9,
            ExpansionSource::Hypernym => 0.7,
            ExpansionSource::Hyponym => 0.6,
            ExpansionSource::RelatedEntity => 0.5,
            ExpansionSource::CoOccurrence => 0.4,
        }
    }

    // ------------------------------------------------------------------
    // Lookup helpers
    // ------------------------------------------------------------------

    /// Return all registered synonyms for `term` (case-insensitive).
    pub fn synonyms_for<'a>(&'a self, term: &str) -> Vec<&'a str> {
        self.synonyms
            .get(&term.to_lowercase())
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Return all entities connected to `term` via any predicate in the graph
    /// context (both as subject and as object).
    pub fn related_entities(&self, term: &str) -> Vec<String> {
        let term_lc = term.to_lowercase();
        let mut related: Vec<String> = Vec::new();

        for (s, _p, o) in &self.context.triples {
            if s.to_lowercase() == term_lc {
                related.push(o.clone());
            } else if o.to_lowercase() == term_lc {
                related.push(s.clone());
            }
        }

        // Deduplicate while preserving order
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        related.retain(|x| seen.insert(x.clone()));
        related
    }

    /// Return `(term, frequency)` pairs for terms that co-occur with `term` in
    /// the same triple (i.e. appear together as subject+object), sorted
    /// descending by frequency.
    pub fn co_occurring_terms(&self, term: &str) -> Vec<(String, usize)> {
        let term_lc = term.to_lowercase();
        let mut freq: HashMap<String, usize> = HashMap::new();

        for (s, _p, o) in &self.context.triples {
            let s_lc = s.to_lowercase();
            let o_lc = o.to_lowercase();
            if s_lc == term_lc {
                *freq.entry(o.clone()).or_insert(0) += 1;
            } else if o_lc == term_lc {
                *freq.entry(s.clone()).or_insert(0) += 1;
            }
        }

        let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        pairs
    }

    // ------------------------------------------------------------------
    // Main expansion
    // ------------------------------------------------------------------

    /// Expand `query` and return an `ExpandedQuery`.
    ///
    /// The algorithm:
    /// 1. Tokenise the query on whitespace.
    /// 2. For each token, gather synonyms, related entities, and co-occurring
    ///    terms from the graph context.
    /// 3. Rank all candidate expansions, deduplicate, and truncate to
    ///    `max_expansions`.
    pub fn expand(&self, query: &str) -> ExpandedQuery {
        let tokens: Vec<&str> = query.split_whitespace().collect();
        let mut all_expansions: Vec<ExpansionTerm> = Vec::new();
        let mut seen_terms: std::collections::HashSet<String> = std::collections::HashSet::new();

        for token in &tokens {
            let lc_token = token.to_lowercase();

            // Synonyms
            for syn in self.synonyms_for(token) {
                let key = syn.to_lowercase();
                if !seen_terms.contains(&key) && key != lc_token {
                    seen_terms.insert(key.clone());
                    all_expansions.push(ExpansionTerm {
                        term: syn.to_string(),
                        score: Self::score_expansion(&ExpansionSource::Synonym),
                        source: ExpansionSource::Synonym,
                    });
                }
            }

            // Related entities from graph context
            for entity in self.related_entities(token) {
                let key = entity.to_lowercase();
                if !seen_terms.contains(&key) && key != lc_token {
                    seen_terms.insert(key.clone());
                    all_expansions.push(ExpansionTerm {
                        term: entity,
                        score: Self::score_expansion(&ExpansionSource::RelatedEntity),
                        source: ExpansionSource::RelatedEntity,
                    });
                }
            }

            // Co-occurrence terms
            for (co_term, freq) in self.co_occurring_terms(token) {
                let key = co_term.to_lowercase();
                if !seen_terms.contains(&key) && key != lc_token {
                    seen_terms.insert(key.clone());
                    // Scale score by frequency (capped at 1.0)
                    let freq_bonus = (freq as f64 * 0.05).min(0.1);
                    let score =
                        (Self::score_expansion(&ExpansionSource::CoOccurrence) + freq_bonus)
                            .min(1.0);
                    all_expansions.push(ExpansionTerm {
                        term: co_term,
                        score,
                        source: ExpansionSource::CoOccurrence,
                    });
                }
            }
        }

        // Sort by score descending, then alphabetically for stability
        all_expansions.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.term.cmp(&b.term))
        });

        all_expansions.truncate(self.max_expansions);

        let expanded_terms: Vec<String> = all_expansions.iter().map(|e| e.term.clone()).collect();

        ExpandedQuery {
            original: query.to_string(),
            expansions: all_expansions,
            expanded_terms,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // GraphContext
    // -------------------------------------------------------------------------
    #[test]
    fn test_graph_context_new_empty() {
        let ctx = GraphContext::new();
        assert!(ctx.triples.is_empty());
    }

    #[test]
    fn test_graph_context_add_triple() {
        let mut ctx = GraphContext::new();
        ctx.add_triple("Battery", "type", "EnergyStorage");
        assert_eq!(ctx.triples.len(), 1);
        assert_eq!(ctx.triples[0].0, "Battery");
    }

    #[test]
    fn test_graph_context_multiple_triples() {
        let mut ctx = GraphContext::new();
        ctx.add_triple("A", "rel", "B");
        ctx.add_triple("B", "rel", "C");
        assert_eq!(ctx.triples.len(), 2);
    }

    // -------------------------------------------------------------------------
    // ExpansionSource scoring
    // -------------------------------------------------------------------------
    #[test]
    fn test_score_synonym_is_highest() {
        let syn = QueryExpander::score_expansion(&ExpansionSource::Synonym);
        let rel = QueryExpander::score_expansion(&ExpansionSource::RelatedEntity);
        assert!(syn > rel);
    }

    #[test]
    fn test_score_all_sources() {
        assert_eq!(QueryExpander::score_expansion(&ExpansionSource::Synonym), 0.9);
        assert_eq!(QueryExpander::score_expansion(&ExpansionSource::Hypernym), 0.7);
        assert_eq!(QueryExpander::score_expansion(&ExpansionSource::Hyponym), 0.6);
        assert_eq!(QueryExpander::score_expansion(&ExpansionSource::RelatedEntity), 0.5);
        assert_eq!(QueryExpander::score_expansion(&ExpansionSource::CoOccurrence), 0.4);
    }

    #[test]
    fn test_score_ordering() {
        let sources = [
            ExpansionSource::Synonym,
            ExpansionSource::Hypernym,
            ExpansionSource::Hyponym,
            ExpansionSource::RelatedEntity,
            ExpansionSource::CoOccurrence,
        ];
        for w in sources.windows(2) {
            assert!(
                QueryExpander::score_expansion(&w[0]) > QueryExpander::score_expansion(&w[1]),
                "{:?} should score higher than {:?}",
                w[0],
                w[1]
            );
        }
    }

    // -------------------------------------------------------------------------
    // QueryExpander::new
    // -------------------------------------------------------------------------
    #[test]
    fn test_new_stores_max_expansions() {
        let qe = QueryExpander::new(10);
        assert_eq!(qe.max_expansions, 10);
    }

    #[test]
    fn test_new_zero_clamps_to_one() {
        let qe = QueryExpander::new(0);
        assert_eq!(qe.max_expansions, 1);
    }

    // -------------------------------------------------------------------------
    // add_synonym / synonyms_for
    // -------------------------------------------------------------------------
    #[test]
    fn test_add_synonym_basic() {
        let mut qe = QueryExpander::new(10);
        qe.add_synonym("car", "automobile");
        let syns = qe.synonyms_for("car");
        assert!(syns.contains(&"automobile"));
    }

    #[test]
    fn test_add_synonym_bidirectional() {
        let mut qe = QueryExpander::new(10);
        qe.add_synonym("car", "vehicle");
        assert!(qe.synonyms_for("vehicle").contains(&"car"));
    }

    #[test]
    fn test_add_synonym_case_insensitive() {
        let mut qe = QueryExpander::new(10);
        qe.add_synonym("Car", "Automobile");
        let syns = qe.synonyms_for("car");
        assert!(syns.contains(&"automobile"));
    }

    #[test]
    fn test_synonyms_for_unknown_returns_empty() {
        let qe = QueryExpander::new(10);
        assert!(qe.synonyms_for("unknown").is_empty());
    }

    #[test]
    fn test_add_multiple_synonyms() {
        let mut qe = QueryExpander::new(10);
        qe.add_synonym("car", "auto");
        qe.add_synonym("car", "vehicle");
        qe.add_synonym("car", "automobile");
        assert_eq!(qe.synonyms_for("car").len(), 3);
    }

    // -------------------------------------------------------------------------
    // related_entities
    // -------------------------------------------------------------------------
    #[test]
    fn test_related_entities_as_subject() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("Battery", "hasComponent", "Anode");
        ctx.add_triple("Battery", "hasComponent", "Cathode");
        qe.set_context(ctx);
        let related = qe.related_entities("Battery");
        assert!(related.contains(&"Anode".to_string()));
        assert!(related.contains(&"Cathode".to_string()));
    }

    #[test]
    fn test_related_entities_as_object() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("Plant", "produces", "Oxygen");
        qe.set_context(ctx);
        let related = qe.related_entities("Oxygen");
        assert!(related.contains(&"Plant".to_string()));
    }

    #[test]
    fn test_related_entities_deduplication() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("A", "rel1", "B");
        ctx.add_triple("A", "rel2", "B");
        qe.set_context(ctx);
        let related = qe.related_entities("A");
        assert_eq!(related.iter().filter(|e| e.as_str() == "B").count(), 1);
    }

    #[test]
    fn test_related_entities_empty_context() {
        let qe = QueryExpander::new(10);
        assert!(qe.related_entities("anything").is_empty());
    }

    #[test]
    fn test_related_entities_case_insensitive() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("battery", "type", "LiIon");
        qe.set_context(ctx);
        let related = qe.related_entities("Battery");
        assert!(related.contains(&"LiIon".to_string()));
    }

    // -------------------------------------------------------------------------
    // co_occurring_terms
    // -------------------------------------------------------------------------
    #[test]
    fn test_co_occurring_terms_basic() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("A", "rel", "B");
        ctx.add_triple("A", "rel", "B");
        ctx.add_triple("A", "rel", "C");
        qe.set_context(ctx);
        let co = qe.co_occurring_terms("A");
        // B appears twice, C once
        assert_eq!(co[0].0, "B");
        assert_eq!(co[0].1, 2);
    }

    #[test]
    fn test_co_occurring_terms_empty() {
        let qe = QueryExpander::new(10);
        assert!(qe.co_occurring_terms("anything").is_empty());
    }

    #[test]
    fn test_co_occurring_terms_sorted_desc() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("X", "p", "Z");
        ctx.add_triple("X", "p", "Y");
        ctx.add_triple("X", "p", "Y");
        ctx.add_triple("X", "p", "Y");
        qe.set_context(ctx);
        let co = qe.co_occurring_terms("X");
        assert_eq!(co[0].0, "Y");
        assert_eq!(co[0].1, 3);
    }

    // -------------------------------------------------------------------------
    // expand
    // -------------------------------------------------------------------------
    #[test]
    fn test_expand_empty_query() {
        let qe = QueryExpander::new(10);
        let eq = qe.expand("");
        assert_eq!(eq.original, "");
        assert!(eq.expansions.is_empty());
    }

    #[test]
    fn test_expand_with_synonym() {
        let mut qe = QueryExpander::new(10);
        qe.add_synonym("cat", "feline");
        let eq = qe.expand("cat");
        assert!(eq.expanded_terms.contains(&"feline".to_string()));
    }

    #[test]
    fn test_expand_respects_max_expansions() {
        let mut qe = QueryExpander::new(2);
        qe.add_synonym("x", "a");
        qe.add_synonym("x", "b");
        qe.add_synonym("x", "c");
        let eq = qe.expand("x");
        assert!(eq.expansions.len() <= 2);
    }

    #[test]
    fn test_expand_no_duplicates() {
        let mut qe = QueryExpander::new(20);
        qe.add_synonym("dog", "hound");
        qe.add_synonym("dog", "hound"); // duplicate registration
        let eq = qe.expand("dog");
        let count = eq.expanded_terms.iter().filter(|t| t.as_str() == "hound").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_expand_original_preserved() {
        let qe = QueryExpander::new(10);
        let eq = qe.expand("find batteries");
        assert_eq!(eq.original, "find batteries");
    }

    #[test]
    fn test_expand_with_graph_context() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("Battery", "hasComponent", "Anode");
        qe.set_context(ctx);
        let eq = qe.expand("Battery");
        assert!(
            eq.expanded_terms.contains(&"Anode".to_string()),
            "expanded_terms: {:?}",
            eq.expanded_terms
        );
    }

    #[test]
    fn test_expand_combined_sources() {
        let mut qe = QueryExpander::new(20);
        qe.add_synonym("cell", "unit");
        let mut ctx = GraphContext::new();
        ctx.add_triple("cell", "partOf", "module");
        qe.set_context(ctx);
        let eq = qe.expand("cell");
        // Should have both synonym and related entity
        let sources: Vec<&ExpansionSource> = eq.expansions.iter().map(|e| &e.source).collect();
        assert!(sources.contains(&&ExpansionSource::Synonym));
        assert!(sources.contains(&&ExpansionSource::RelatedEntity));
    }

    #[test]
    fn test_expand_multi_token_query() {
        let mut qe = QueryExpander::new(20);
        qe.add_synonym("find", "search");
        qe.add_synonym("cell", "battery");
        let eq = qe.expand("find cell");
        assert!(eq.expanded_terms.contains(&"search".to_string()));
        assert!(eq.expanded_terms.contains(&"battery".to_string()));
    }

    #[test]
    fn test_expand_term_not_repeated_in_expansions() {
        let mut qe = QueryExpander::new(20);
        qe.add_synonym("car", "auto");
        let eq = qe.expand("car");
        // The original query token "car" should not appear as an expansion
        assert!(!eq.expanded_terms.contains(&"car".to_string()));
    }

    #[test]
    fn test_expand_sorted_by_score_descending() {
        let mut qe = QueryExpander::new(20);
        qe.add_synonym("item", "thing");
        let mut ctx = GraphContext::new();
        ctx.add_triple("item", "rel", "object");
        qe.set_context(ctx);
        let eq = qe.expand("item");
        // First expansion should have score >= last expansion
        if eq.expansions.len() >= 2 {
            let first_score = eq.expansions[0].score;
            let last_score = eq.expansions.last().map(|e| e.score).unwrap_or(0.0);
            assert!(first_score >= last_score);
        }
    }

    #[test]
    fn test_expand_co_occurrence_included() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("anode", "partOf", "cell");
        qe.set_context(ctx);
        let eq = qe.expand("anode");
        assert!(eq.expanded_terms.contains(&"cell".to_string()));
    }

    #[test]
    fn test_expansion_term_struct() {
        let t = ExpansionTerm {
            term: "test".to_string(),
            score: 0.8,
            source: ExpansionSource::Hypernym,
        };
        assert_eq!(t.term, "test");
        assert_eq!(t.score, 0.8);
        assert_eq!(t.source, ExpansionSource::Hypernym);
    }

    #[test]
    fn test_expanded_query_struct() {
        let eq = ExpandedQuery {
            original: "hello".to_string(),
            expansions: vec![],
            expanded_terms: vec!["world".to_string()],
        };
        assert_eq!(eq.original, "hello");
        assert_eq!(eq.expanded_terms.len(), 1);
    }

    #[test]
    fn test_set_context_replaces_previous() {
        let mut qe = QueryExpander::new(10);
        let mut ctx1 = GraphContext::new();
        ctx1.add_triple("A", "rel", "B");
        qe.set_context(ctx1);

        let ctx2 = GraphContext::new();
        qe.set_context(ctx2);
        assert!(qe.related_entities("A").is_empty());
    }

    #[test]
    fn test_co_occurring_from_both_positions() {
        let mut qe = QueryExpander::new(20);
        let mut ctx = GraphContext::new();
        ctx.add_triple("alpha", "to", "beta"); // alpha as subject
        ctx.add_triple("gamma", "to", "alpha"); // alpha as object
        qe.set_context(ctx);
        let co = qe.co_occurring_terms("alpha");
        let terms: Vec<&str> = co.iter().map(|(t, _)| t.as_str()).collect();
        assert!(terms.contains(&"beta"));
        assert!(terms.contains(&"gamma"));
    }

    #[test]
    fn test_expand_case_normalisation_in_expansion() {
        let mut qe = QueryExpander::new(10);
        qe.add_synonym("Battery", "accumulator");
        let eq = qe.expand("battery");
        // lowercase normalisation should still match
        assert!(
            eq.expanded_terms.contains(&"accumulator".to_string()),
            "expanded_terms: {:?}",
            eq.expanded_terms
        );
    }

    #[test]
    fn test_expansion_source_debug() {
        let s = format!("{:?}", ExpansionSource::Synonym);
        assert!(s.contains("Synonym"));
    }

    #[test]
    fn test_expansion_source_equality() {
        assert_eq!(ExpansionSource::Hypernym, ExpansionSource::Hypernym);
        assert_ne!(ExpansionSource::Hypernym, ExpansionSource::Hyponym);
    }

    #[test]
    fn test_graph_context_clone() {
        let mut ctx = GraphContext::new();
        ctx.add_triple("A", "b", "C");
        let ctx2 = ctx.clone();
        assert_eq!(ctx2.triples.len(), 1);
    }
}
