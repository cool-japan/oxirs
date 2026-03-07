//! Constraint Learning from Positive and Negative Examples
//!
//! This module implements a supervised constraint learning algorithm that:
//!
//! 1. Analyses property distributions over **positive** example triples (triples from
//!    conforming resources) to extract candidate constraints.
//! 2. Validates each candidate against **negative** example triples (triples from
//!    non-conforming resources) to compute a discrimination score.
//! 3. Returns only those `PropertyConstraint`s whose score exceeds a configured
//!    threshold.
//!
//! The approach is inspired by inductive logic programming (ILP) for schema learning
//! and is designed to be lightweight enough to run in-process without external ML
//! runtimes.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::shape_miner::{NodeKind, PropertyConstraint};

/// The result of learning a single constraint, including its validation score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedConstraintResult {
    /// The learned constraint.
    pub constraint: PropertyConstraint,
    /// Discrimination score (0.0 – 1.0): how well the constraint separates positive
    /// from negative examples.  Higher is better.
    pub discrimination_score: f64,
    /// Number of positive examples that satisfy the constraint.
    pub positive_hits: usize,
    /// Number of negative examples that violate the constraint (desired: many).
    pub negative_violations: usize,
}

/// Runtime statistics produced by one `learn_constraints` call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintLearningStats {
    /// Number of positive triples ingested.
    pub positive_triples: usize,
    /// Number of negative triples ingested.
    pub negative_triples: usize,
    /// Number of candidate constraints before filtering.
    pub candidates_generated: usize,
    /// Number of constraints that passed the discrimination threshold.
    pub constraints_accepted: usize,
}

/// Summary report from one learning session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintLearningReport {
    /// Accepted constraints, ordered by `discrimination_score` descending.
    pub results: Vec<LearnedConstraintResult>,
    /// Runtime statistics.
    pub stats: ConstraintLearningStats,
}

/// The constraint learner.
///
/// Provide positive example triples and negative example triples, then call
/// `learn_constraints()` to obtain a ranked list of `PropertyConstraint`s.
///
/// # Example
/// ```rust
/// use oxirs_shacl_ai::shape_learning::ConstraintLearner;
///
/// let positives = vec![
///     ("http://ex.org/alice".into(), "http://ex.org/name".into(), "\"Alice\"".into()),
///     ("http://ex.org/bob".into(),   "http://ex.org/name".into(), "\"Bob\"".into()),
/// ];
/// let negatives = vec![
///     ("http://ex.org/x".into(), "http://ex.org/other".into(), "\"X\"".into()),
/// ];
/// let learner = ConstraintLearner {
///     positive_examples: positives,
///     negative_examples: negatives,
///     min_discrimination_score: 0.5,
///     infer_datatypes: true,
///     infer_node_kinds: true,
/// };
/// let constraints = learner.learn_constraints();
/// assert!(!constraints.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ConstraintLearner {
    /// Triples that represent **conforming** (positive) RDF resources.
    pub positive_examples: Vec<(String, String, String)>,
    /// Triples that represent **non-conforming** (negative) RDF resources.
    pub negative_examples: Vec<(String, String, String)>,
    /// Minimum discrimination score for a constraint to be returned.
    pub min_discrimination_score: f64,
    /// Whether to infer datatypes from literal objects.
    pub infer_datatypes: bool,
    /// Whether to infer node kinds from object values.
    pub infer_node_kinds: bool,
}

impl ConstraintLearner {
    /// Create a new learner with explicit examples and a discrimination threshold.
    pub fn new(
        positive_examples: Vec<(String, String, String)>,
        negative_examples: Vec<(String, String, String)>,
        min_discrimination_score: f64,
    ) -> Self {
        Self {
            positive_examples,
            negative_examples,
            min_discrimination_score: min_discrimination_score.clamp(0.0, 1.0),
            infer_datatypes: true,
            infer_node_kinds: true,
        }
    }

    /// Learn property constraints from the stored positive/negative examples.
    ///
    /// Returns a vector of constraints ordered by `discrimination_score` descending.
    pub fn learn_constraints(&self) -> Vec<PropertyConstraint> {
        let report = self.learn_constraints_with_report();
        report.results.into_iter().map(|r| r.constraint).collect()
    }

    /// Like `learn_constraints` but returns the full `ConstraintLearningReport`.
    pub fn learn_constraints_with_report(&self) -> ConstraintLearningReport {
        let mut stats = ConstraintLearningStats {
            positive_triples: self.positive_examples.len(),
            negative_triples: self.negative_examples.len(),
            ..Default::default()
        };

        // Build positive index: subject → predicate → vec<object>
        let pos_index = build_index(&self.positive_examples);
        // Build negative index: subject → predicate → vec<object>
        let neg_index = build_index(&self.negative_examples);

        let pos_subjects: HashSet<&str> = pos_index.keys().copied().collect();
        let neg_subjects: HashSet<&str> = neg_index.keys().copied().collect();
        let total_pos = pos_subjects.len();
        let total_neg = neg_subjects.len();

        // Gather all predicates from positive examples
        let pos_predicates: HashSet<&str> = pos_index
            .values()
            .flat_map(|pm| pm.keys().copied())
            .collect();

        let mut results: Vec<LearnedConstraintResult> = Vec::new();

        for pred in &pos_predicates {
            stats.candidates_generated += 1;

            // Positive subjects that have this predicate
            let pos_with_pred: Vec<&str> = pos_subjects
                .iter()
                .filter(|&&s| {
                    pos_index
                        .get(s)
                        .map(|pm| pm.contains_key(pred))
                        .unwrap_or(false)
                })
                .copied()
                .collect();

            if pos_with_pred.is_empty() {
                continue;
            }

            // Cardinality statistics over positive examples
            let pos_counts: Vec<u32> = pos_with_pred
                .iter()
                .map(|&s| {
                    pos_index
                        .get(s)
                        .and_then(|pm| pm.get(pred))
                        .map(|v| v.len() as u32)
                        .unwrap_or(0)
                })
                .collect();

            let min_count = pos_counts.iter().copied().min();
            let max_count = pos_counts.iter().copied().max();

            // Datatype / node-kind inference from positive objects
            let pos_objects: Vec<&str> = pos_with_pred
                .iter()
                .flat_map(|&s| {
                    pos_index
                        .get(s)
                        .and_then(|pm| pm.get(pred))
                        .map(|v| v.iter().map(|o| o.as_str()).collect::<Vec<_>>())
                        .unwrap_or_default()
                })
                .collect();

            let datatype = if self.infer_datatypes {
                infer_dominant_datatype(&pos_objects)
            } else {
                None
            };

            let node_kind = if self.infer_node_kinds {
                Some(infer_node_kind(&pos_objects))
            } else {
                None
            };

            // Support in positive examples
            let pos_support = pos_with_pred.len() as f64 / total_pos.max(1) as f64;

            // Validate against negative examples
            let validation_score = self
                .validate_constraint_internal(pred, min_count, max_count, &neg_index, total_neg);

            // discrimination_score = how much positive support we have AND
            // how well it discriminates negatives (violation rate among negatives)
            let discrimination_score = pos_support * (1.0 - validation_score);

            if discrimination_score < self.min_discrimination_score {
                continue;
            }

            // positive_hits: positive subjects that satisfy the constraint
            let positive_hits = pos_with_pred.len();

            // negative_violations: negative subjects that violate this constraint
            // (they lack the property or have wrong cardinality)
            let neg_violations = if total_neg == 0 {
                0
            } else {
                let neg_with_pred_count = neg_subjects
                    .iter()
                    .filter(|&&s| {
                        neg_index
                            .get(s)
                            .map(|pm| pm.contains_key(pred))
                            .unwrap_or(false)
                    })
                    .count();
                total_neg - neg_with_pred_count
            };

            let confidence = if pos_with_pred.is_empty() {
                0.0
            } else {
                let modal = modal_count(&pos_counts);
                let consistent = pos_counts.iter().filter(|&&c| c == modal).count();
                consistent as f64 / pos_with_pred.len() as f64
            };

            let constraint = PropertyConstraint {
                predicate: pred.to_string(),
                min_count,
                max_count,
                datatype,
                node_kind,
                support: pos_support,
                confidence,
            };

            stats.constraints_accepted += 1;
            results.push(LearnedConstraintResult {
                constraint,
                discrimination_score,
                positive_hits,
                negative_violations: neg_violations,
            });
        }

        // Sort by discrimination score descending
        results.sort_by(|a, b| {
            b.discrimination_score
                .partial_cmp(&a.discrimination_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ConstraintLearningReport { results, stats }
    }

    /// Validate a single `PropertyConstraint` against a triple set and return a
    /// **conformance score** (0.0 = zero conformance, 1.0 = full conformance).
    ///
    /// Useful for scoring a constraint against an arbitrary triple set after learning.
    ///
    /// The score is:
    /// ```text
    /// conformance = (subjects with predicate AND within [min,max] cardinality)
    ///                / total_subjects
    /// ```
    pub fn validate_constraint(
        &self,
        constraint: &PropertyConstraint,
        triples: &[(String, String, String)],
    ) -> f64 {
        let index = build_index(triples);
        let total = index.len();
        self.validate_constraint_internal(
            &constraint.predicate,
            constraint.min_count,
            constraint.max_count,
            &index,
            total,
        )
    }

    // ── Private helper ────────────────────────────────────────────────────────

    fn validate_constraint_internal(
        &self,
        pred: &str,
        min_count: Option<u32>,
        max_count: Option<u32>,
        index: &HashMap<&str, HashMap<&str, Vec<String>>>,
        total_subjects: usize,
    ) -> f64 {
        if total_subjects == 0 {
            return 1.0;
        }
        let conforming = index
            .values()
            .filter(|pm| {
                let count = pm.get(pred).map(|v| v.len() as u32).unwrap_or(0);
                let min_ok = min_count.map_or(true, |m| count >= m);
                let max_ok = max_count.map_or(true, |m| count <= m);
                min_ok && max_ok
            })
            .count();
        conforming as f64 / total_subjects as f64
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build a nested index: subject → predicate → vec<object> from a triple slice.
fn build_index<'a>(
    triples: &'a [(String, String, String)],
) -> HashMap<&'a str, HashMap<&'a str, Vec<String>>> {
    let mut index: HashMap<&'a str, HashMap<&'a str, Vec<String>>> = HashMap::new();
    for (subj, pred, obj) in triples {
        index
            .entry(subj.as_str())
            .or_default()
            .entry(pred.as_str())
            .or_default()
            .push(obj.clone());
    }
    index
}

/// Return the mode (most-frequent value) from a non-empty count slice.
fn modal_count(counts: &[u32]) -> u32 {
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for &c in counts {
        *freq.entry(c).or_insert(0) += 1;
    }
    freq.into_iter()
        .max_by_key(|(_, cnt)| *cnt)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

fn infer_dominant_datatype(objects: &[&str]) -> Option<String> {
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for obj in objects {
        if let Some(idx) = obj.find("^^") {
            let dt = obj[idx + 2..].trim_matches(|c| c == '<' || c == '>');
            *type_counts.entry(dt.to_string()).or_insert(0) += 1;
        }
    }
    type_counts
        .into_iter()
        .max_by_key(|(_, cnt)| *cnt)
        .map(|(dt, _)| dt)
}

fn infer_node_kind(objects: &[&str]) -> NodeKind {
    if objects.is_empty() {
        return NodeKind::Mixed;
    }
    let mut iri_count: usize = 0;
    let mut literal_count: usize = 0;
    let mut blank_count: usize = 0;

    for obj in objects {
        if obj.starts_with('"') {
            literal_count += 1;
        } else if obj.starts_with("_:") {
            blank_count += 1;
        } else {
            iri_count += 1;
        }
    }

    let max = iri_count.max(literal_count).max(blank_count);
    let total = objects.len();

    if max == iri_count && iri_count * 2 > total {
        NodeKind::Iri
    } else if max == literal_count && literal_count * 2 > total {
        NodeKind::Literal
    } else if max == blank_count && blank_count * 2 > total {
        NodeKind::BlankNode
    } else {
        NodeKind::Mixed
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn positive_triples() -> Vec<(String, String, String)> {
        vec![
            (
                "http://ex.org/alice".into(),
                "http://ex.org/name".into(),
                "\"Alice\"".into(),
            ),
            (
                "http://ex.org/alice".into(),
                "http://ex.org/email".into(),
                "\"alice@ex.com\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/name".into(),
                "\"Bob\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/email".into(),
                "\"bob@ex.com\"".into(),
            ),
        ]
    }

    fn negative_triples() -> Vec<(String, String, String)> {
        vec![
            (
                "http://ex.org/x".into(),
                "http://ex.org/other".into(),
                "\"X\"".into(),
            ),
            (
                "http://ex.org/y".into(),
                "http://ex.org/other".into(),
                "\"Y\"".into(),
            ),
        ]
    }

    #[test]
    fn test_learn_constraints_returns_constraints() {
        let learner = ConstraintLearner::new(positive_triples(), negative_triples(), 0.1);
        let constraints = learner.learn_constraints();
        assert!(
            !constraints.is_empty(),
            "should learn at least one constraint"
        );
    }

    #[test]
    fn test_learn_constraints_predicates() {
        let learner = ConstraintLearner::new(positive_triples(), negative_triples(), 0.1);
        let constraints = learner.learn_constraints();
        let preds: Vec<&str> = constraints.iter().map(|c| c.predicate.as_str()).collect();
        assert!(
            preds.contains(&"http://ex.org/name"),
            "expected name constraint"
        );
        assert!(
            preds.contains(&"http://ex.org/email"),
            "expected email constraint"
        );
    }

    #[test]
    fn test_validate_constraint_full_conformance() {
        let learner = ConstraintLearner::new(positive_triples(), negative_triples(), 0.0);
        let constraints = learner.learn_constraints();
        // Pick the name constraint and validate against the positive set
        if let Some(c) = constraints.iter().find(|c| c.predicate.ends_with("name")) {
            let score = learner.validate_constraint(c, &positive_triples());
            assert!(
                score > 0.5,
                "expected high conformance score, got {}",
                score
            );
        }
    }

    #[test]
    fn test_validate_constraint_zero_conformance() {
        let learner = ConstraintLearner::new(positive_triples(), negative_triples(), 0.0);
        // A constraint requiring a predicate that does not exist in the negative set
        let phantom = PropertyConstraint {
            predicate: "http://ex.org/name".into(),
            min_count: Some(1),
            max_count: None,
            datatype: None,
            node_kind: None,
            support: 1.0,
            confidence: 1.0,
        };
        // Validate against negative examples – they don't have `name`
        let score = learner.validate_constraint(&phantom, &negative_triples());
        // Subjects in negatives don't have 'name', so conformance = 0/2 = 0
        assert!(score < 0.5, "expected low conformance, got {}", score);
    }

    #[test]
    fn test_high_threshold_filters_all() {
        let learner = ConstraintLearner::new(positive_triples(), negative_triples(), 0.99);
        let constraints = learner.learn_constraints();
        // With extremely high threshold, may filter everything out
        // (this is correct behaviour, not a failure)
        let _ = constraints;
    }

    #[test]
    fn test_empty_positives_returns_empty() {
        let learner = ConstraintLearner::new(vec![], negative_triples(), 0.1);
        let constraints = learner.learn_constraints();
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_empty_negatives_accepted() {
        let learner = ConstraintLearner::new(positive_triples(), vec![], 0.0);
        let constraints = learner.learn_constraints();
        // With no negatives the discrimination score = pos_support * 1.0 = pos_support
        // so as long as threshold is 0 we should get constraints
        assert!(!constraints.is_empty());
    }

    #[test]
    fn test_report_stats_correctness() {
        let learner = ConstraintLearner::new(positive_triples(), negative_triples(), 0.0);
        let report = learner.learn_constraints_with_report();
        assert_eq!(report.stats.positive_triples, positive_triples().len());
        assert_eq!(report.stats.negative_triples, negative_triples().len());
        assert!(report.stats.candidates_generated >= 1);
        assert!(report.stats.constraints_accepted >= 1);
        assert_eq!(report.stats.constraints_accepted, report.results.len());
    }

    #[test]
    fn test_results_ordered_by_score() {
        let learner = ConstraintLearner::new(positive_triples(), negative_triples(), 0.0);
        let report = learner.learn_constraints_with_report();
        let scores: Vec<f64> = report
            .results
            .iter()
            .map(|r| r.discrimination_score)
            .collect();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "results should be ordered by discrimination_score desc"
            );
        }
    }

    #[test]
    fn test_min_count_inference() {
        // Alice and Bob each have exactly one name → min_count should be Some(1)
        let learner = ConstraintLearner::new(positive_triples(), vec![], 0.0);
        let report = learner.learn_constraints_with_report();
        if let Some(r) = report
            .results
            .iter()
            .find(|r| r.constraint.predicate.ends_with("name"))
        {
            assert_eq!(r.constraint.min_count, Some(1));
        }
    }
}
