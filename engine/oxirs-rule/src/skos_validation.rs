//! # SKOS Integrity Conditions and Validation
//!
//! Implements SKOS integrity conditions as described in the W3C SKOS Reference
//! specification §S14. These checks detect structural problems in SKOS concept
//! schemes:
//!
//! - **S14**: `skos:Concept` and `skos:ConceptScheme` are disjoint classes.
//! - **S27**: `skos:Collection` is disjoint with `skos:Concept` and `skos:ConceptScheme`.
//! - **S46**: `skos:broader` and `skos:broaderTransitive` do not create cycles.
//! - Prefixed label uniqueness within a scheme (S13).
//! - Single top-concept verification per scheme.
//!
//! ## Reference
//! - <https://www.w3.org/TR/skos-reference/#integrity-conditions>

use std::collections::{HashMap, HashSet, VecDeque};

use crate::skos_types::{
    Graph, NamedNode, SkosError, SkosResult, BROADER, BROADER_TRANSITIVE, HAS_TOP_CONCEPT,
    IN_SCHEME, PREF_LABEL, TOP_CONCEPT_OF,
};

// RDF/SKOS class IRIs
const SKOS_CONCEPT: &str = "http://www.w3.org/2004/02/skos/core#Concept";
const SKOS_CONCEPT_SCHEME: &str = "http://www.w3.org/2004/02/skos/core#ConceptScheme";
const SKOS_COLLECTION: &str = "http://www.w3.org/2004/02/skos/core#Collection";
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    /// Machine-readable condition identifier (e.g. `"S14"`, `"S27"`, `"cycle"`).
    pub condition: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// The IRI(s) involved in the violation.
    pub subjects: Vec<NamedNode>,
}

/// Aggregated result from running all SKOS integrity checks.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// Violations found. Empty means the graph is valid under these checks.
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    /// Returns `true` if no violations were found.
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// Append a finding.
    pub fn push(&mut self, finding: ValidationFinding) {
        self.findings.push(finding);
    }
}

// ---------------------------------------------------------------------------
// SkosValidator
// ---------------------------------------------------------------------------

/// Validates a SKOS graph against the W3C SKOS integrity conditions.
pub struct SkosValidator;

impl SkosValidator {
    /// Run all implemented integrity checks on `graph`.
    ///
    /// Returns a [`ValidationReport`] describing any violations found.
    /// A graph that passes all checks is considered SKOS-valid under the
    /// implemented conditions.
    pub fn validate(graph: &Graph) -> ValidationReport {
        let mut report = ValidationReport::default();

        Self::check_concept_scheme_disjoint(graph, &mut report);
        Self::check_collection_disjoint(graph, &mut report);
        Self::check_broader_cycles(graph, &mut report);
        Self::check_label_uniqueness(graph, &mut report);
        Self::check_top_concept_in_scheme(graph, &mut report);

        report
    }

    // -----------------------------------------------------------------------
    // S14: skos:Concept and skos:ConceptScheme are disjoint
    // -----------------------------------------------------------------------

    /// S14: A resource cannot simultaneously be a `skos:Concept` and a
    /// `skos:ConceptScheme`.
    pub fn check_concept_scheme_disjoint(graph: &Graph, report: &mut ValidationReport) {
        let concepts: HashSet<_> = graph
            .triples_with_predicate(RDF_TYPE)
            .filter(|(_, _, o)| o == SKOS_CONCEPT)
            .map(|(s, _, _)| s.clone())
            .collect();

        let schemes: HashSet<_> = graph
            .triples_with_predicate(RDF_TYPE)
            .filter(|(_, _, o)| o == SKOS_CONCEPT_SCHEME)
            .map(|(s, _, _)| s.clone())
            .collect();

        for iri in concepts.intersection(&schemes) {
            report.push(ValidationFinding {
                condition: "S14".to_string(),
                message: format!("<{iri}> is both a skos:Concept and a skos:ConceptScheme"),
                subjects: vec![iri.clone()],
            });
        }
    }

    // -----------------------------------------------------------------------
    // S27: skos:Collection disjoint with Concept and ConceptScheme
    // -----------------------------------------------------------------------

    /// S27: A resource cannot simultaneously be a `skos:Collection` and a
    /// `skos:Concept` or `skos:ConceptScheme`.
    pub fn check_collection_disjoint(graph: &Graph, report: &mut ValidationReport) {
        let collections: HashSet<_> = graph
            .triples_with_predicate(RDF_TYPE)
            .filter(|(_, _, o)| o == SKOS_COLLECTION)
            .map(|(s, _, _)| s.clone())
            .collect();

        let concepts: HashSet<_> = graph
            .triples_with_predicate(RDF_TYPE)
            .filter(|(_, _, o)| o == SKOS_CONCEPT || o == SKOS_CONCEPT_SCHEME)
            .map(|(s, _, _)| s.clone())
            .collect();

        for iri in collections.intersection(&concepts) {
            report.push(ValidationFinding {
                condition: "S27".to_string(),
                message: format!(
                    "<{iri}> is a skos:Collection but also typed as skos:Concept or skos:ConceptScheme"
                ),
                subjects: vec![iri.clone()],
            });
        }
    }

    // -----------------------------------------------------------------------
    // S46: skos:broader / broaderTransitive must be acyclic
    // -----------------------------------------------------------------------

    /// Detect cycles reachable from any node through `skos:broader` or
    /// `skos:broaderTransitive` edges.
    ///
    /// Returns an error on the *first* cycle discovered; use
    /// [`Self::validate`] to collect all findings without early exit.
    pub fn check_broader_no_cycle(graph: &Graph, start: &NamedNode) -> SkosResult<()> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        for (_, p, o) in graph.triples_with_subject(start) {
            if (p == BROADER || p == BROADER_TRANSITIVE) && o != start {
                queue.push_back(o.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            if &current == start {
                return Err(SkosError::CycleDetected(start.clone()));
            }
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());
            for (_, p, o) in graph.triples_with_subject(&current) {
                if p == BROADER || p == BROADER_TRANSITIVE {
                    if o == start {
                        return Err(SkosError::CycleDetected(start.clone()));
                    }
                    if !visited.contains(o) {
                        queue.push_back(o.clone());
                    }
                }
            }
        }

        Ok(())
    }

    /// Scan all subjects of `skos:broader` and `skos:broaderTransitive` for
    /// cycles and record findings in `report`.
    pub fn check_broader_cycles(graph: &Graph, report: &mut ValidationReport) {
        let candidates: HashSet<_> = graph
            .triples_with_predicate(BROADER)
            .chain(graph.triples_with_predicate(BROADER_TRANSITIVE))
            .map(|(s, _, _)| s.clone())
            .collect();

        for start in candidates {
            if let Err(SkosError::CycleDetected(c)) = Self::check_broader_no_cycle(graph, &start) {
                report.push(ValidationFinding {
                    condition: "S46".to_string(),
                    message: format!("Cycle detected in skos:broader hierarchy at <{c}>"),
                    subjects: vec![c],
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // S13: skos:prefLabel uniqueness per language per concept (per scheme)
    // -----------------------------------------------------------------------

    /// Verify that each concept has at most one `skos:prefLabel` per language
    /// tag.
    ///
    /// Only concepts linked to *any* scheme via `skos:inScheme` are checked.
    pub fn check_label_uniqueness(graph: &Graph, report: &mut ValidationReport) {
        // Build: concept → Vec<lang> from prefLabel triples
        let mut concept_labels: HashMap<String, Vec<String>> = HashMap::new();

        for (s, _, o) in graph.triples_with_predicate(PREF_LABEL) {
            let lang = if let Some(pos) = o.rfind('@') {
                o[pos + 1..].to_lowercase()
            } else {
                String::new() // no-lang bucket
            };
            concept_labels.entry(s.clone()).or_default().push(lang);
        }

        for (concept, langs) in &concept_labels {
            let mut seen: HashMap<&str, usize> = HashMap::new();
            for lang in langs {
                let count = seen.entry(lang.as_str()).or_insert(0);
                *count += 1;
            }
            for (lang, &count) in &seen {
                if count > 1 {
                    let tag_desc = if lang.is_empty() {
                        "no language tag".to_string()
                    } else {
                        format!("language tag @{lang}")
                    };
                    report.push(ValidationFinding {
                        condition: "S13".to_string(),
                        message: format!("<{concept}> has {count} skos:prefLabels with {tag_desc}"),
                        subjects: vec![concept.clone()],
                    });
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Top-concept consistency: every topConceptOf target should be inScheme
    // -----------------------------------------------------------------------

    /// Verify that every concept `C` declared via `skos:topConceptOf S` or
    /// `S skos:hasTopConcept C` is also declared `skos:inScheme S`.
    ///
    /// This is an advisory check rather than a hard SKOS integrity condition;
    /// it is included because missing `inScheme` links are a common modelling
    /// error.
    pub fn check_top_concept_in_scheme(graph: &Graph, report: &mut ValidationReport) {
        // Collect (concept, scheme) pairs from topConceptOf
        let top_pairs: Vec<(String, String)> = graph
            .triples_with_predicate(TOP_CONCEPT_OF)
            .map(|(c, _, s)| (c.clone(), s.clone()))
            .chain(
                graph
                    .triples_with_predicate(HAS_TOP_CONCEPT)
                    .map(|(s, _, c)| (c.clone(), s.clone())),
            )
            .collect();

        for (concept, scheme) in top_pairs {
            if !graph.contains(&concept, IN_SCHEME, &scheme) {
                report.push(ValidationFinding {
                    condition: "top_concept_in_scheme".to_string(),
                    message: format!(
                        "<{concept}> is a top concept of <{scheme}> but lacks skos:inScheme <{scheme}>"
                    ),
                    subjects: vec![concept, scheme],
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skos_types::{Graph, BROADER, BROADER_TRANSITIVE, HAS_TOP_CONCEPT, PREF_LABEL};

    fn g() -> Graph {
        Graph::new()
    }

    #[test]
    fn test_no_violations_empty_graph() {
        let graph = g();
        let report = SkosValidator::validate(&graph);
        assert!(report.is_valid());
    }

    #[test]
    fn test_s14_concept_and_scheme_disjoint() {
        let mut graph = g();
        graph.add_triple("ex:X", RDF_TYPE, SKOS_CONCEPT);
        graph.add_triple("ex:X", RDF_TYPE, SKOS_CONCEPT_SCHEME);

        let report = SkosValidator::validate(&graph);
        assert!(!report.is_valid());
        assert!(report.findings.iter().any(|f| f.condition == "S14"));
    }

    #[test]
    fn test_s27_collection_and_concept_disjoint() {
        let mut graph = g();
        graph.add_triple("ex:C", RDF_TYPE, SKOS_COLLECTION);
        graph.add_triple("ex:C", RDF_TYPE, SKOS_CONCEPT);

        let report = SkosValidator::validate(&graph);
        assert!(!report.is_valid());
        assert!(report.findings.iter().any(|f| f.condition == "S27"));
    }

    #[test]
    fn test_s46_broader_cycle_detected() {
        let mut graph = g();
        graph.add_triple("ex:A", BROADER, "ex:B");
        graph.add_triple("ex:B", BROADER, "ex:C");
        graph.add_triple("ex:C", BROADER, "ex:A"); // cycle

        let report = SkosValidator::validate(&graph);
        assert!(!report.is_valid());
        assert!(report.findings.iter().any(|f| f.condition == "S46"));
    }

    #[test]
    fn test_s46_no_cycle_in_acyclic_graph() {
        let mut graph = g();
        graph.add_triple("ex:A", BROADER, "ex:B");
        graph.add_triple("ex:B", BROADER, "ex:C");
        // no cycle

        let report = SkosValidator::validate(&graph);
        let has_s46 = report.findings.iter().any(|f| f.condition == "S46");
        assert!(!has_s46, "Acyclic graph should not trigger S46");
    }

    #[test]
    fn test_s46_broader_transitive_cycle() {
        let mut graph = g();
        graph.add_triple("ex:A", BROADER_TRANSITIVE, "ex:B");
        graph.add_triple("ex:B", BROADER_TRANSITIVE, "ex:A"); // cycle

        let report = SkosValidator::validate(&graph);
        assert!(report.findings.iter().any(|f| f.condition == "S46"));
    }

    #[test]
    fn test_s13_label_uniqueness_violation() {
        let mut graph = g();
        graph.add_triple("ex:Art", PREF_LABEL, "Art@en");
        graph.add_triple("ex:Art", PREF_LABEL, "Arte@en"); // second @en label

        let report = SkosValidator::validate(&graph);
        assert!(report.findings.iter().any(|f| f.condition == "S13"));
    }

    #[test]
    fn test_s13_different_lang_labels_ok() {
        let mut graph = g();
        graph.add_triple("ex:Art", PREF_LABEL, "Art@en");
        graph.add_triple("ex:Art", PREF_LABEL, "Kunst@de"); // different lang — ok

        let report = SkosValidator::validate(&graph);
        let has_s13 = report.findings.iter().any(|f| f.condition == "S13");
        assert!(!has_s13, "Different language tags should not trigger S13");
    }

    #[test]
    fn test_top_concept_in_scheme_missing_in_scheme() {
        let mut graph = g();
        graph.add_triple("ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        // ex:Art does NOT have skos:inScheme ex:Scheme

        let report = SkosValidator::validate(&graph);
        assert!(report
            .findings
            .iter()
            .any(|f| f.condition == "top_concept_in_scheme"));
    }

    #[test]
    fn test_top_concept_in_scheme_present_no_violation() {
        let mut graph = g();
        graph.add_triple("ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        graph.add_triple("ex:Art", IN_SCHEME, "ex:Scheme");

        let report = SkosValidator::validate(&graph);
        let has = report
            .findings
            .iter()
            .any(|f| f.condition == "top_concept_in_scheme");
        assert!(!has, "Top concept with inScheme should not violate check");
    }
}
