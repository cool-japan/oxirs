//! # In-Memory Triple Store for WASM
//!
//! A lightweight, indexed triple store suitable for WebAssembly environments.
//! Provides add / remove / pattern-matching / set operations / N-Triples
//! serialisation and parsing — all in pure Rust with no external dependencies
//! beyond the standard library.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::triple_store::{TripleStore, RdfTriple};
//!
//! let mut store = TripleStore::new();
//! store.add(RdfTriple::new(
//!     "<http://example.org/alice>",
//!     "<http://xmlns.com/foaf/0.1/knows>",
//!     "<http://example.org/bob>",
//! ));
//! assert_eq!(store.len(), 1);
//! ```

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// A single RDF triple represented as (subject, predicate, object) strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RdfTriple {
    /// Subject (IRI or blank-node identifier).
    pub subject: String,
    /// Predicate (IRI).
    pub predicate: String,
    /// Object (IRI, blank-node identifier, or literal).
    pub object: String,
}

impl RdfTriple {
    /// Create a new triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Serialise as a single N-Triples line (no trailing newline).
    pub fn to_ntriples_line(&self) -> String {
        format!("{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// Aggregate statistics about the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreStats {
    /// Total number of triples.
    pub triple_count: usize,
    /// Number of distinct subjects.
    pub unique_subjects: usize,
    /// Number of distinct predicates.
    pub unique_predicates: usize,
    /// Number of distinct objects.
    pub unique_objects: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// Failed to parse an N-Triples line.
    ParseError(String),
    /// Attempted to add a duplicate triple.
    DuplicateTriple,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            StoreError::DuplicateTriple => write!(f, "Triple already exists"),
        }
    }
}

impl std::error::Error for StoreError {}

// ─────────────────────────────────────────────────────────────────────────────
// TripleStore
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory triple store with three hash-based indexes.
///
/// Every triple is stored exactly once in `triples`; the three secondary
/// indexes (subject, predicate, object) map term → set of indices into
/// `triples`.
#[derive(Debug, Default)]
pub struct TripleStore {
    /// Canonical set of triples (order-preserving insertion vec).
    triples: Vec<RdfTriple>,
    /// Fast duplicate check.
    triple_set: HashSet<RdfTriple>,
    /// subject → positions in `triples`.
    subject_index: HashMap<String, Vec<usize>>,
    /// predicate → positions in `triples`.
    predicate_index: HashMap<String, Vec<usize>>,
    /// object → positions in `triples`.
    object_index: HashMap<String, Vec<usize>>,
}

impl TripleStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of triples in the store.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    // ── Add ──────────────────────────────────────────────────────────────

    /// Add a single triple.  Returns `true` if actually inserted (not a
    /// duplicate).
    pub fn add(&mut self, triple: RdfTriple) -> bool {
        if self.triple_set.contains(&triple) {
            return false;
        }
        let idx = self.triples.len();
        self.subject_index
            .entry(triple.subject.clone())
            .or_default()
            .push(idx);
        self.predicate_index
            .entry(triple.predicate.clone())
            .or_default()
            .push(idx);
        self.object_index
            .entry(triple.object.clone())
            .or_default()
            .push(idx);
        self.triple_set.insert(triple.clone());
        self.triples.push(triple);
        true
    }

    /// Bulk-add triples.  Returns the number of actually inserted triples.
    pub fn add_all(&mut self, triples: impl IntoIterator<Item = RdfTriple>) -> usize {
        let mut count = 0usize;
        for t in triples {
            if self.add(t) {
                count += 1;
            }
        }
        count
    }

    // ── Remove ───────────────────────────────────────────────────────────

    /// Remove a specific triple.  Returns `true` if found and removed.
    ///
    /// This performs a "logical delete" by rebuilding indexes, which is
    /// acceptable for small WASM datasets.
    pub fn remove(&mut self, triple: &RdfTriple) -> bool {
        if !self.triple_set.remove(triple) {
            return false;
        }
        self.triples.retain(|t| t != triple);
        self.rebuild_indexes();
        true
    }

    /// Bulk-remove triples.  Returns the number actually removed.
    pub fn remove_all(&mut self, triples: &[RdfTriple]) -> usize {
        let remove_set: HashSet<&RdfTriple> = triples.iter().collect();
        let before = self.triples.len();
        self.triple_set.retain(|t| !remove_set.contains(t));
        self.triples.retain(|t| !remove_set.contains(t));
        let count = before - self.triples.len();
        if count > 0 {
            self.rebuild_indexes();
        }
        count
    }

    /// Clear all triples.
    pub fn clear(&mut self) {
        self.triples.clear();
        self.triple_set.clear();
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
    }

    // ── Contains ─────────────────────────────────────────────────────────

    /// Check whether a specific triple exists.
    pub fn contains(&self, triple: &RdfTriple) -> bool {
        self.triple_set.contains(triple)
    }

    // ── Pattern matching ─────────────────────────────────────────────────

    /// Match triples by pattern.  `None` acts as a wildcard.
    pub fn match_pattern(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<&RdfTriple> {
        // Optimise: choose the most selective index.
        match (subject, predicate, object) {
            (Some(s), Some(p), Some(o)) => {
                // Exact lookup
                let target = RdfTriple::new(s, p, o);
                if self.triple_set.contains(&target) {
                    // Find the reference in self.triples
                    self.triples.iter().filter(|t| **t == target).collect()
                } else {
                    vec![]
                }
            }
            (Some(s), _, _) => {
                let indices = self.subject_index.get(s);
                self.filter_indices(indices, subject, predicate, object)
            }
            (_, Some(p), _) => {
                let indices = self.predicate_index.get(p);
                self.filter_indices(indices, subject, predicate, object)
            }
            (_, _, Some(o)) => {
                let indices = self.object_index.get(o);
                self.filter_indices(indices, subject, predicate, object)
            }
            (None, None, None) => self.triples.iter().collect(),
        }
    }

    /// Internal helper: filter indexed positions by the remaining pattern.
    fn filter_indices(
        &self,
        indices: Option<&Vec<usize>>,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<&RdfTriple> {
        let Some(indices) = indices else {
            return vec![];
        };
        indices
            .iter()
            .filter_map(|&i| self.triples.get(i))
            .filter(|t| {
                subject.map_or(true, |s| t.subject == s)
                    && predicate.map_or(true, |p| t.predicate == p)
                    && object.map_or(true, |o| t.object == o)
            })
            .collect()
    }

    // ── Iteration ────────────────────────────────────────────────────────

    /// Iterate over all triples.
    pub fn iter(&self) -> impl Iterator<Item = &RdfTriple> {
        self.triples.iter()
    }

    /// Collect all triples as a vector of references.
    pub fn all_triples(&self) -> Vec<&RdfTriple> {
        self.triples.iter().collect()
    }

    // ── Set operations ───────────────────────────────────────────────────

    /// Set union: return a new store containing triples from both stores.
    pub fn union(&self, other: &TripleStore) -> TripleStore {
        let mut result = TripleStore::new();
        for t in &self.triples {
            result.add(t.clone());
        }
        for t in &other.triples {
            result.add(t.clone());
        }
        result
    }

    /// Set intersection: return a new store containing only triples present
    /// in both stores.
    pub fn intersection(&self, other: &TripleStore) -> TripleStore {
        let mut result = TripleStore::new();
        for t in &self.triples {
            if other.contains(t) {
                result.add(t.clone());
            }
        }
        result
    }

    /// Set difference: return a new store containing triples in `self` but
    /// not in `other`.
    pub fn difference(&self, other: &TripleStore) -> TripleStore {
        let mut result = TripleStore::new();
        for t in &self.triples {
            if !other.contains(t) {
                result.add(t.clone());
            }
        }
        result
    }

    // ── Statistics ───────────────────────────────────────────────────────

    /// Compute aggregate statistics.
    pub fn stats(&self) -> StoreStats {
        StoreStats {
            triple_count: self.triples.len(),
            unique_subjects: self.subject_index.len(),
            unique_predicates: self.predicate_index.len(),
            unique_objects: self.object_index.len(),
        }
    }

    // ── N-Triples serialisation ──────────────────────────────────────────

    /// Serialise the store as an N-Triples string.
    pub fn to_ntriples(&self) -> String {
        let mut buf = String::new();
        for t in &self.triples {
            buf.push_str(&t.to_ntriples_line());
            buf.push('\n');
        }
        buf
    }

    /// Import triples from an N-Triples string (basic parser).
    ///
    /// Each non-empty, non-comment line is expected to have three
    /// whitespace-separated terms followed by a period:
    ///
    /// ```text
    /// <subject> <predicate> <object> .
    /// ```
    ///
    /// Returns the number of triples successfully imported.
    pub fn import_ntriples(&mut self, input: &str) -> Result<usize, StoreError> {
        let mut count = 0usize;
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let triple = parse_ntriples_line(trimmed)?;
            if self.add(triple) {
                count += 1;
            }
        }
        Ok(count)
    }

    // ── Internal ─────────────────────────────────────────────────────────

    /// Rebuild all three indexes from scratch.
    fn rebuild_indexes(&mut self) {
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        for (idx, t) in self.triples.iter().enumerate() {
            self.subject_index
                .entry(t.subject.clone())
                .or_default()
                .push(idx);
            self.predicate_index
                .entry(t.predicate.clone())
                .or_default()
                .push(idx);
            self.object_index
                .entry(t.object.clone())
                .or_default()
                .push(idx);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// N-Triples line parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a single N-Triples line into an `RdfTriple`.
///
/// Supports:
/// - `<IRI> <IRI> <IRI> .`
/// - `<IRI> <IRI> "literal" .`
/// - `<IRI> <IRI> "literal"^^<datatype> .`
/// - `<IRI> <IRI> "literal"@lang .`
/// - blank nodes: `_:label`
fn parse_ntriples_line(line: &str) -> Result<RdfTriple, StoreError> {
    let line = line.trim();
    // Strip trailing '.'
    let line = line
        .strip_suffix('.')
        .ok_or_else(|| StoreError::ParseError("Missing trailing '.'".to_string()))?
        .trim();

    let mut parts: Vec<String> = Vec::new();
    let mut chars = line.chars().peekable();

    while parts.len() < 3 {
        // Skip whitespace
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }

        if chars.peek().is_none() {
            break;
        }

        match chars.peek() {
            Some('<') => {
                // IRI
                let mut iri = String::new();
                for ch in chars.by_ref() {
                    iri.push(ch);
                    if ch == '>' {
                        break;
                    }
                }
                parts.push(iri);
            }
            Some('_') => {
                // Blank node
                let mut bnode = String::new();
                for ch in chars.by_ref() {
                    if ch.is_whitespace() {
                        break;
                    }
                    bnode.push(ch);
                }
                parts.push(bnode);
            }
            Some('"') => {
                // Literal
                let mut literal = String::new();
                literal.push('"');
                chars.next(); // consume opening quote
                let mut escaped = false;
                for ch in chars.by_ref() {
                    literal.push(ch);
                    if escaped {
                        escaped = false;
                        continue;
                    }
                    if ch == '\\' {
                        escaped = true;
                        continue;
                    }
                    if ch == '"' {
                        break;
                    }
                }
                // Check for datatype or lang tag
                match chars.peek() {
                    Some('^') => {
                        // ^^<datatype>
                        for ch in chars.by_ref() {
                            if ch == '>' {
                                literal.push(ch);
                                break;
                            }
                            literal.push(ch);
                        }
                    }
                    Some('@') => {
                        // @lang
                        for ch in chars.by_ref() {
                            if ch.is_whitespace() {
                                break;
                            }
                            literal.push(ch);
                        }
                    }
                    _ => {}
                }
                parts.push(literal);
            }
            Some(_) => {
                // Other token (unquoted)
                let mut token = String::new();
                for ch in chars.by_ref() {
                    if ch.is_whitespace() {
                        break;
                    }
                    token.push(ch);
                }
                parts.push(token);
            }
            None => break,
        }
    }

    if parts.len() < 3 {
        return Err(StoreError::ParseError(format!(
            "Expected 3 terms, found {}",
            parts.len()
        )));
    }

    Ok(RdfTriple::new(&parts[0], &parts[1], &parts[2]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn alice_knows_bob() -> RdfTriple {
        RdfTriple::new(
            "<http://example.org/alice>",
            "<http://xmlns.com/foaf/0.1/knows>",
            "<http://example.org/bob>",
        )
    }

    fn alice_name() -> RdfTriple {
        RdfTriple::new(
            "<http://example.org/alice>",
            "<http://xmlns.com/foaf/0.1/name>",
            "\"Alice\"",
        )
    }

    fn bob_name() -> RdfTriple {
        RdfTriple::new(
            "<http://example.org/bob>",
            "<http://xmlns.com/foaf/0.1/name>",
            "\"Bob\"",
        )
    }

    fn bob_knows_carol() -> RdfTriple {
        RdfTriple::new(
            "<http://example.org/bob>",
            "<http://xmlns.com/foaf/0.1/knows>",
            "<http://example.org/carol>",
        )
    }

    // ── Add ──────────────────────────────────────────────────────────────

    #[test]
    fn test_add_single_triple() {
        let mut store = TripleStore::new();
        assert!(store.add(alice_knows_bob()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_add_duplicate_returns_false() {
        let mut store = TripleStore::new();
        assert!(store.add(alice_knows_bob()));
        assert!(!store.add(alice_knows_bob()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_add_all_bulk() {
        let mut store = TripleStore::new();
        let count = store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_add_all_with_duplicates() {
        let mut store = TripleStore::new();
        store.add(alice_knows_bob());
        let count = store.add_all(vec![alice_knows_bob(), alice_name()]);
        assert_eq!(count, 1); // only alice_name is new
        assert_eq!(store.len(), 2);
    }

    // ── Remove ───────────────────────────────────────────────────────────

    #[test]
    fn test_remove_existing_triple() {
        let mut store = TripleStore::new();
        store.add(alice_knows_bob());
        assert!(store.remove(&alice_knows_bob()));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_triple() {
        let mut store = TripleStore::new();
        assert!(!store.remove(&alice_knows_bob()));
    }

    #[test]
    fn test_remove_all_bulk() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        let count = store.remove_all(&[alice_knows_bob(), bob_name()]);
        assert_eq!(count, 2);
        assert_eq!(store.len(), 1);
        assert!(store.contains(&alice_name()));
    }

    #[test]
    fn test_clear() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name()]);
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    // ── Contains ─────────────────────────────────────────────────────────

    #[test]
    fn test_contains_existing() {
        let mut store = TripleStore::new();
        store.add(alice_knows_bob());
        assert!(store.contains(&alice_knows_bob()));
    }

    #[test]
    fn test_contains_nonexistent() {
        let store = TripleStore::new();
        assert!(!store.contains(&alice_knows_bob()));
    }

    // ── Pattern matching ─────────────────────────────────────────────────

    #[test]
    fn test_match_all_wildcard() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name()]);
        let results = store.match_pattern(None, None, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_match_by_subject() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        let results = store.match_pattern(Some("<http://example.org/alice>"), None, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_match_by_predicate() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        let results = store.match_pattern(None, Some("<http://xmlns.com/foaf/0.1/name>"), None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_match_by_object() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        let results = store.match_pattern(None, None, Some("<http://example.org/bob>"));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_match_by_subject_and_predicate() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        let results = store.match_pattern(
            Some("<http://example.org/alice>"),
            Some("<http://xmlns.com/foaf/0.1/knows>"),
            None,
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_match_exact_triple() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name()]);
        let results = store.match_pattern(
            Some("<http://example.org/alice>"),
            Some("<http://xmlns.com/foaf/0.1/knows>"),
            Some("<http://example.org/bob>"),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_match_no_results() {
        let mut store = TripleStore::new();
        store.add(alice_knows_bob());
        let results = store.match_pattern(Some("<http://example.org/nonexistent>"), None, None);
        assert!(results.is_empty());
    }

    // ── Iteration ────────────────────────────────────────────────────────

    #[test]
    fn test_iter() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name()]);
        let collected: Vec<_> = store.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_all_triples() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name()]);
        assert_eq!(store.all_triples().len(), 2);
    }

    // ── Set operations ───────────────────────────────────────────────────

    #[test]
    fn test_union() {
        let mut a = TripleStore::new();
        a.add_all(vec![alice_knows_bob(), alice_name()]);
        let mut b = TripleStore::new();
        b.add_all(vec![alice_knows_bob(), bob_name()]);

        let u = a.union(&b);
        assert_eq!(u.len(), 3); // alice_knows_bob deduplicated
    }

    #[test]
    fn test_intersection() {
        let mut a = TripleStore::new();
        a.add_all(vec![alice_knows_bob(), alice_name()]);
        let mut b = TripleStore::new();
        b.add_all(vec![alice_knows_bob(), bob_name()]);

        let i = a.intersection(&b);
        assert_eq!(i.len(), 1);
        assert!(i.contains(&alice_knows_bob()));
    }

    #[test]
    fn test_difference() {
        let mut a = TripleStore::new();
        a.add_all(vec![alice_knows_bob(), alice_name()]);
        let mut b = TripleStore::new();
        b.add_all(vec![alice_knows_bob(), bob_name()]);

        let d = a.difference(&b);
        assert_eq!(d.len(), 1);
        assert!(d.contains(&alice_name()));
    }

    #[test]
    fn test_union_empty() {
        let a = TripleStore::new();
        let b = TripleStore::new();
        let u = a.union(&b);
        assert!(u.is_empty());
    }

    #[test]
    fn test_intersection_empty() {
        let mut a = TripleStore::new();
        a.add(alice_knows_bob());
        let b = TripleStore::new();
        let i = a.intersection(&b);
        assert!(i.is_empty());
    }

    #[test]
    fn test_difference_empty() {
        let mut a = TripleStore::new();
        a.add(alice_knows_bob());
        let d = a.difference(&a);
        assert!(d.is_empty());
    }

    // ── Statistics ───────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let store = TripleStore::new();
        let stats = store.stats();
        assert_eq!(stats.triple_count, 0);
        assert_eq!(stats.unique_subjects, 0);
        assert_eq!(stats.unique_predicates, 0);
        assert_eq!(stats.unique_objects, 0);
    }

    #[test]
    fn test_stats_populated() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        let stats = store.stats();
        assert_eq!(stats.triple_count, 3);
        assert_eq!(stats.unique_subjects, 2); // alice, bob
        assert_eq!(stats.unique_predicates, 2); // knows, name
        assert_eq!(stats.unique_objects, 3); // bob, "Alice", "Bob"
    }

    // ── N-Triples serialisation ──────────────────────────────────────────

    #[test]
    fn test_to_ntriples() {
        let mut store = TripleStore::new();
        store.add(alice_knows_bob());
        let nt = store.to_ntriples();
        assert!(
            nt.contains("<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .")
        );
    }

    #[test]
    fn test_to_ntriples_empty() {
        let store = TripleStore::new();
        assert_eq!(store.to_ntriples(), "");
    }

    #[test]
    fn test_triple_to_ntriples_line() {
        let t = alice_knows_bob();
        let line = t.to_ntriples_line();
        assert_eq!(
            line,
            "<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> ."
        );
    }

    // ── N-Triples import ─────────────────────────────────────────────────

    #[test]
    fn test_import_ntriples_basic() {
        let mut store = TripleStore::new();
        let input = "<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .\n";
        let count = store.import_ntriples(input).expect("import");
        assert_eq!(count, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_import_ntriples_multiple_lines() {
        let mut store = TripleStore::new();
        let input = "\
<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .
<http://example.org/bob> <http://xmlns.com/foaf/0.1/name> \"Bob\" .
";
        let count = store.import_ntriples(input).expect("import");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_import_ntriples_skip_comments() {
        let mut store = TripleStore::new();
        let input = "# This is a comment\n<http://example.org/a> <http://example.org/b> <http://example.org/c> .\n";
        let count = store.import_ntriples(input).expect("import");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_import_ntriples_skip_empty_lines() {
        let mut store = TripleStore::new();
        let input =
            "\n\n<http://example.org/a> <http://example.org/b> <http://example.org/c> .\n\n";
        let count = store.import_ntriples(input).expect("import");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_import_ntriples_literal_with_datatype() {
        let mut store = TripleStore::new();
        let input =
            "<http://example.org/a> <http://example.org/age> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n";
        let count = store.import_ntriples(input).expect("import");
        assert_eq!(count, 1);
        let triples = store.all_triples();
        assert!(triples[0].object.contains("^^"));
    }

    #[test]
    fn test_import_ntriples_literal_with_lang() {
        let mut store = TripleStore::new();
        let input = "<http://example.org/a> <http://example.org/name> \"Alice\"@en .\n";
        let count = store.import_ntriples(input).expect("import");
        assert_eq!(count, 1);
        let triples = store.all_triples();
        assert!(triples[0].object.contains("@en"));
    }

    #[test]
    fn test_import_ntriples_blank_node() {
        let mut store = TripleStore::new();
        let input = "_:b0 <http://example.org/name> \"Blank\" .\n";
        let count = store.import_ntriples(input).expect("import");
        assert_eq!(count, 1);
        assert_eq!(store.all_triples()[0].subject, "_:b0");
    }

    #[test]
    fn test_import_ntriples_error_missing_dot() {
        let mut store = TripleStore::new();
        let input = "<http://example.org/a> <http://example.org/b> <http://example.org/c>\n";
        let result = store.import_ntriples(input);
        assert!(result.is_err());
    }

    // ── Round-trip ───────────────────────────────────────────────────────

    #[test]
    fn test_ntriples_roundtrip() {
        let mut store1 = TripleStore::new();
        store1.add_all(vec![alice_knows_bob(), bob_name()]);
        let nt = store1.to_ntriples();

        let mut store2 = TripleStore::new();
        let count = store2.import_ntriples(&nt).expect("import");
        assert_eq!(count, 2);
        assert_eq!(store2.len(), 2);
    }

    // ── Index consistency after remove ───────────────────────────────────

    #[test]
    fn test_index_after_remove() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), alice_name(), bob_name()]);
        store.remove(&alice_knows_bob());

        // Subject index should still work
        let results = store.match_pattern(Some("<http://example.org/alice>"), None, None);
        assert_eq!(results.len(), 1); // only alice_name left
    }

    // ── Error display ────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let err = StoreError::ParseError("bad".to_string());
        assert!(err.to_string().contains("Parse error"));

        let err = StoreError::DuplicateTriple;
        assert!(err.to_string().contains("already exists"));
    }

    // ── is_empty ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_empty_true() {
        let store = TripleStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn test_is_empty_false() {
        let mut store = TripleStore::new();
        store.add(alice_knows_bob());
        assert!(!store.is_empty());
    }

    // ── Multiple predicates for same subject ─────────────────────────────

    #[test]
    fn test_subject_with_multiple_predicates() {
        let mut store = TripleStore::new();
        store.add_all(vec![
            alice_knows_bob(),
            alice_name(),
            RdfTriple::new(
                "<http://example.org/alice>",
                "<http://xmlns.com/foaf/0.1/age>",
                "\"30\"",
            ),
        ]);
        let results = store.match_pattern(Some("<http://example.org/alice>"), None, None);
        assert_eq!(results.len(), 3);
    }

    // ── Large bulk add ───────────────────────────────────────────────────

    #[test]
    fn test_bulk_add_100_triples() {
        let mut store = TripleStore::new();
        let triples: Vec<RdfTriple> = (0..100)
            .map(|i| {
                RdfTriple::new(
                    format!("<http://example.org/s{i}>"),
                    "<http://example.org/p>",
                    format!("<http://example.org/o{i}>"),
                )
            })
            .collect();
        let count = store.add_all(triples);
        assert_eq!(count, 100);
        assert_eq!(store.len(), 100);
    }

    // ── Match with predicate and object ──────────────────────────────────

    #[test]
    fn test_match_predicate_and_object() {
        let mut store = TripleStore::new();
        store.add_all(vec![alice_knows_bob(), bob_knows_carol()]);
        let results = store.match_pattern(
            None,
            Some("<http://xmlns.com/foaf/0.1/knows>"),
            Some("<http://example.org/bob>"),
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "<http://example.org/alice>");
    }
}
