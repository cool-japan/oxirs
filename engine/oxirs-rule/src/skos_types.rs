//! # SKOS Core Types and Vocabulary
//!
//! Vocabulary constants, error types, the in-memory [`Graph`] triple store and
//! basic concept-tree types shared across the SKOS reasoner and analyzer.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// SKOS vocabulary constants (mirrors oxirs-core::vocab::skos)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub(crate) const SKOS_NS: &str = "http://www.w3.org/2004/02/skos/core#";

pub(crate) const BROADER: &str = "http://www.w3.org/2004/02/skos/core#broader";
pub(crate) const NARROWER: &str = "http://www.w3.org/2004/02/skos/core#narrower";
pub(crate) const BROADER_TRANSITIVE: &str = "http://www.w3.org/2004/02/skos/core#broaderTransitive";
pub(crate) const NARROWER_TRANSITIVE: &str =
    "http://www.w3.org/2004/02/skos/core#narrowerTransitive";
pub(crate) const RELATED: &str = "http://www.w3.org/2004/02/skos/core#related";
pub(crate) const EXACT_MATCH: &str = "http://www.w3.org/2004/02/skos/core#exactMatch";
pub(crate) const CLOSE_MATCH: &str = "http://www.w3.org/2004/02/skos/core#closeMatch";
pub(crate) const BROAD_MATCH: &str = "http://www.w3.org/2004/02/skos/core#broadMatch";
pub(crate) const NARROW_MATCH: &str = "http://www.w3.org/2004/02/skos/core#narrowMatch";
pub(crate) const IN_SCHEME: &str = "http://www.w3.org/2004/02/skos/core#inScheme";
pub(crate) const HAS_TOP_CONCEPT: &str = "http://www.w3.org/2004/02/skos/core#hasTopConcept";
pub(crate) const TOP_CONCEPT_OF: &str = "http://www.w3.org/2004/02/skos/core#topConceptOf";
pub(crate) const PREF_LABEL: &str = "http://www.w3.org/2004/02/skos/core#prefLabel";
pub(crate) const ALT_LABEL: &str = "http://www.w3.org/2004/02/skos/core#altLabel";

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A triple using owned string values: (subject IRI, predicate IRI, object IRI or literal)
pub type Triple = (String, String, String);

/// Named node (IRI) string alias for clarity
pub type NamedNode = String;

/// Error type for SKOS reasoning operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum SkosError {
    #[error("Cycle detected in SKOS hierarchy involving concept: {0}")]
    CycleDetected(String),
    #[error("Invalid SKOS graph: {0}")]
    InvalidGraph(String),
    #[error("Maximum iteration limit ({0}) exceeded during entailment")]
    MaxIterationsExceeded(usize),
}

/// Result alias for SKOS operations
pub type SkosResult<T> = Result<T, SkosError>;

// ---------------------------------------------------------------------------
// Graph — a simple in-memory triple store for SKOS reasoning
// ---------------------------------------------------------------------------

/// A lightweight in-memory RDF graph sufficient for SKOS entailment.
///
/// Triples are stored as (subject, predicate, object) string tuples.
/// Subject and object are IRIs; literals are represented as plain strings
/// (for labels, the value is the lexical form, optionally with a `@lang` suffix).
#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub(crate) triples: HashSet<Triple>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            triples: HashSet::new(),
        }
    }

    /// Add a triple to the graph.  Returns `true` if the triple is new.
    pub fn add_triple(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> bool {
        self.triples
            .insert((subject.into(), predicate.into(), object.into()))
    }

    /// Test membership of a triple.
    pub fn contains(&self, subject: &str, predicate: &str, object: &str) -> bool {
        self.triples
            .contains(&(subject.to_owned(), predicate.to_owned(), object.to_owned()))
    }

    /// Iterate over all triples with the given predicate.
    pub fn triples_with_predicate<'a>(
        &'a self,
        predicate: &str,
    ) -> impl Iterator<Item = &'a Triple> {
        let predicate = predicate.to_string();
        self.triples.iter().filter(move |(_, p, _)| p == &predicate)
    }

    /// Iterate over all triples whose subject matches `subject`.
    pub fn triples_with_subject<'a>(&'a self, subject: &str) -> impl Iterator<Item = &'a Triple> {
        let subject = subject.to_string();
        self.triples.iter().filter(move |(s, _, _)| s == &subject)
    }

    /// Iterate over all triples whose object matches `object`.
    pub fn triples_with_object<'a>(&'a self, object: &str) -> impl Iterator<Item = &'a Triple> {
        let object = object.to_string();
        self.triples.iter().filter(move |(_, _, o)| o == &object)
    }

    /// Return all triples.
    pub fn triples(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }

    /// Number of triples in the graph.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Returns `true` if the graph has no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Merge another graph's triples into this one.
    pub fn merge(&mut self, other: &Graph) {
        for triple in &other.triples {
            self.triples.insert(triple.clone());
        }
    }

    /// Return an adjacency map for a specific predicate: subject → set of objects.
    pub fn adjacency_map(&self, predicate: &str) -> HashMap<String, HashSet<String>> {
        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        for (s, p, o) in &self.triples {
            if p == predicate {
                map.entry(s.clone()).or_default().insert(o.clone());
            }
        }
        map
    }
}

// ---------------------------------------------------------------------------
// label_matches helper
// ---------------------------------------------------------------------------

/// Internal helper: test whether an RDF literal object matches a label and
/// optional language tag.
///
/// Literals are stored as `"value"` or `"value"@lang`.
pub(crate) fn label_matches(literal: &str, label: &str, lang: Option<&str>) -> bool {
    match lang {
        None => {
            // Match if the literal value (stripped of language tag) equals label
            if let Some(at_pos) = literal.rfind('@') {
                &literal[..at_pos] == label
            } else {
                literal == label
            }
        }
        Some(expected_lang) => {
            if let Some(at_pos) = literal.rfind('@') {
                let val = &literal[..at_pos];
                let lit_lang = &literal[at_pos + 1..];
                val == label && lit_lang.eq_ignore_ascii_case(expected_lang)
            } else {
                // No language tag — only match if caller did not request specific lang
                literal == label
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConceptNode / ConceptTree
// ---------------------------------------------------------------------------

/// A tree node representing a concept and its narrower concepts (children).
#[derive(Debug, Clone)]
pub struct ConceptNode {
    /// The IRI of this concept
    pub iri: NamedNode,
    /// The preferred label (if any)
    pub pref_label: Option<String>,
    /// Narrower (child) concepts in the hierarchy
    pub children: Vec<ConceptNode>,
}

/// The root of a concept tree for a particular `ConceptScheme`.
#[derive(Debug, Clone)]
pub struct ConceptTree {
    /// Top-level concepts of the scheme (concepts without a broader concept within the scheme)
    pub root_concepts: Vec<ConceptNode>,
}
