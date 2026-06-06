//! Shared fixtures for SPARQL compliance integration tests.
//!
//! Each `tests/sparql_compliance*.rs` file is compiled as a separate cargo test
//! binary. To avoid duplicating the dataset and `Dataset` trait implementation,
//! all common code lives here and is included via `mod common;` in each test
//! binary.
//!
//! NOTE: Cargo treats `tests/common/mod.rs` as a non-binary helper module
//! (the `mod.rs` filename suppresses the implicit `[[test]]` discovery), so
//! this file is never compiled as a standalone test crate. Some helpers
//! defined here are intentionally only used by a subset of the test binaries;
//! `#[allow(dead_code)]` keeps the no-warnings policy satisfied across all
//! binaries.

#![allow(dead_code)]

use oxirs_arq::algebra::Term;
use oxirs_arq::{Literal, TriplePattern};
use oxirs_core::model::NamedNode;

/// Mock dataset for testing.
///
/// Implements the minimal `oxirs_arq::Dataset` surface needed to drive
/// `QueryExecutor::execute` over hand-built triple sets.
pub struct MockDataset {
    triples: Vec<(Term, Term, Term)>,
}

impl MockDataset {
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    pub fn add_triple(&mut self, s: Term, p: Term, o: Term) {
        self.triples.push((s, p, o));
    }
}

impl Default for MockDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl oxirs_arq::Dataset for MockDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> anyhow::Result<Vec<(Term, Term, Term)>> {
        let mut results = Vec::new();

        for (s, p, o) in &self.triples {
            let mut matches = true;

            // Check subject
            match &pattern.subject {
                Term::Variable(_) => {} // Variables match anything
                other => {
                    if other != s {
                        matches = false;
                    }
                }
            }

            // Check predicate
            if matches {
                match &pattern.predicate {
                    Term::Variable(_) => {} // Variables match anything
                    other => {
                        if other != p {
                            matches = false;
                        }
                    }
                }
            }

            // Check object
            if matches {
                match &pattern.object {
                    Term::Variable(_) => {} // Variables match anything
                    other => {
                        if other != o {
                            matches = false;
                        }
                    }
                }
            }

            if matches {
                results.push((s.clone(), p.clone(), o.clone()));
            }
        }

        Ok(results)
    }

    fn contains_triple(
        &self,
        subject: &Term,
        predicate: &Term,
        object: &Term,
    ) -> anyhow::Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(s, p, o)| s == subject && p == predicate && o == object))
    }

    fn subjects(&self) -> anyhow::Result<Vec<Term>> {
        let mut subjects: Vec<Term> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        subjects.sort();
        subjects.dedup();
        Ok(subjects)
    }

    fn predicates(&self) -> anyhow::Result<Vec<Term>> {
        let mut predicates: Vec<Term> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        predicates.sort();
        predicates.dedup();
        Ok(predicates)
    }

    fn objects(&self) -> anyhow::Result<Vec<Term>> {
        let mut objects: Vec<Term> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        objects.sort();
        objects.dedup();
        Ok(objects)
    }
}

/// Standard 4-triple `foaf` test dataset shared across compliance binaries:
///
/// - alice foaf:name "Alice"
/// - alice foaf:age  "30"^^xsd:integer
/// - bob   foaf:name "Bob"
/// - bob   foaf:age  "25"^^xsd:integer
pub fn create_test_dataset() -> MockDataset {
    let mut dataset = MockDataset::new();

    // Add some test data
    dataset.add_triple(
        Term::Iri(NamedNode::new_unchecked("http://example.org/alice")),
        Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
        Term::Literal(Literal {
            value: "Alice".to_string(),
            language: None,
            datatype: None,
        }),
    );

    dataset.add_triple(
        Term::Iri(NamedNode::new_unchecked("http://example.org/alice")),
        Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
        Term::Literal(Literal {
            value: "30".to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        }),
    );

    dataset.add_triple(
        Term::Iri(NamedNode::new_unchecked("http://example.org/bob")),
        Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/name")),
        Term::Literal(Literal {
            value: "Bob".to_string(),
            language: None,
            datatype: None,
        }),
    );

    dataset.add_triple(
        Term::Iri(NamedNode::new_unchecked("http://example.org/bob")),
        Term::Iri(NamedNode::new_unchecked("http://xmlns.com/foaf/0.1/age")),
        Term::Literal(Literal {
            value: "25".to_string(),
            language: None,
            datatype: Some(NamedNode::new_unchecked(
                "http://www.w3.org/2001/XMLSchema#integer",
            )),
        }),
    );

    dataset
}
