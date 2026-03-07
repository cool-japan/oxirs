//! Graph Pattern Detection for SHACL Shape Discovery
//!
//! Analyses raw RDF triple data to identify recurring structural patterns such as
//! star-shaped property sets, functional properties, inverse-functional properties,
//! co-occurrence clusters, and hierarchical class structures.
//!
//! These patterns serve as evidence for generating high-quality SHACL shapes and
//! are complementary to the purely statistical approach in `shape_miner`.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// The kind of structural pattern detected in the RDF graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternKind {
    /// A subject with many outgoing property–value pairs (star topology).
    StarShaped,
    /// A predicate whose values are unique across all subjects (functional).
    FunctionalProperty,
    /// A predicate whose subjects are unique for each object (inverse-functional).
    InverseFunctionalProperty,
    /// Two predicates that frequently appear together on the same subject.
    CoOccurrence,
    /// A `rdfs:subClassOf` or `owl:equivalentClass` hierarchy link.
    ClassHierarchy,
    /// A property chain (`p1` then `p2` leads to a third value).
    PropertyChain,
    /// Many subjects sharing the exact same set of predicates.
    SharedPropertyProfile,
}

impl PatternKind {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            PatternKind::StarShaped => "star-shaped",
            PatternKind::FunctionalProperty => "functional-property",
            PatternKind::InverseFunctionalProperty => "inverse-functional-property",
            PatternKind::CoOccurrence => "co-occurrence",
            PatternKind::ClassHierarchy => "class-hierarchy",
            PatternKind::PropertyChain => "property-chain",
            PatternKind::SharedPropertyProfile => "shared-property-profile",
        }
    }
}

/// A concrete detected pattern instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Kind of pattern.
    pub kind: PatternKind,
    /// Primary entity involved (predicate URI, subject URI, class URI, etc.).
    pub primary_entity: String,
    /// Secondary entity involved (for binary patterns like co-occurrence / chain).
    pub secondary_entity: Option<String>,
    /// Statistical support: fraction of relevant candidates exhibiting this pattern.
    pub support: f64,
    /// Number of triples / subjects / pairs that constitute this observation.
    pub evidence_count: usize,
    /// Human-readable explanation.
    pub description: String,
}

/// A higher-level graph pattern grouping multiple `DetectedPattern`s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPattern {
    /// Unique identifier.
    pub id: String,
    /// Overall pattern kind (the dominant one).
    pub kind: PatternKind,
    /// Constituent detected patterns.
    pub patterns: Vec<DetectedPattern>,
    /// Aggregate support.
    pub support: f64,
    /// Associated class URI if applicable.
    pub associated_class: Option<String>,
}

/// Configuration for the pattern detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetectionConfig {
    /// Minimum support for a co-occurrence pattern.
    pub min_co_occurrence_support: f64,
    /// Minimum fraction of IFP values that must be unique.
    pub min_functional_uniqueness: f64,
    /// Minimum number of subjects sharing a property profile.
    pub min_shared_profile_count: usize,
    /// Whether to detect star-shaped patterns.
    pub detect_star_shaped: bool,
    /// Whether to detect functional / inverse-functional properties.
    pub detect_functional: bool,
    /// Whether to detect co-occurrence patterns.
    pub detect_co_occurrence: bool,
    /// Whether to detect class-hierarchy patterns.
    pub detect_class_hierarchy: bool,
    /// Whether to detect shared property profiles.
    pub detect_shared_profiles: bool,
}

impl Default for PatternDetectionConfig {
    fn default() -> Self {
        Self {
            min_co_occurrence_support: 0.5,
            min_functional_uniqueness: 0.95,
            min_shared_profile_count: 2,
            detect_star_shaped: true,
            detect_functional: true,
            detect_co_occurrence: true,
            detect_class_hierarchy: true,
            detect_shared_profiles: true,
        }
    }
}

/// Summary report from one pattern detection run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetectionReport {
    /// All detected patterns, grouped by kind.
    pub patterns: Vec<DetectedPattern>,
    /// High-level graph patterns.
    pub graph_patterns: Vec<GraphPattern>,
    /// Total distinct subjects analysed.
    pub subjects_analysed: usize,
    /// Total distinct predicates analysed.
    pub predicates_analysed: usize,
}

/// The pattern detection engine.
///
/// # Example
/// ```rust
/// use oxirs_shacl_ai::shape_learning::{PatternDetector, PatternDetectionConfig};
///
/// let detector = PatternDetector::new(PatternDetectionConfig::default());
/// let triples = vec![
///     ("http://ex.org/alice".into(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(), "http://ex.org/Person".into()),
///     ("http://ex.org/alice".into(), "http://ex.org/name".into(), "\"Alice\"".into()),
/// ];
/// let report = detector.detect(&triples);
/// assert!(report.subjects_analysed >= 1);
/// ```
#[derive(Debug, Clone)]
pub struct PatternDetector {
    config: PatternDetectionConfig,
}

impl PatternDetector {
    /// Create a detector with the given configuration.
    pub fn new(config: PatternDetectionConfig) -> Self {
        Self { config }
    }

    /// Create a detector with the default configuration.
    pub fn default_config() -> Self {
        Self::new(PatternDetectionConfig::default())
    }

    /// Access the current configuration.
    pub fn config(&self) -> &PatternDetectionConfig {
        &self.config
    }

    /// Run pattern detection on a triple set and return a full report.
    pub fn detect(&self, triples: &[(String, String, String)]) -> PatternDetectionReport {
        const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        const RDFS_SUB_CLASS: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
        const OWL_EQUIV: &str = "http://www.w3.org/2002/07/owl#equivalentClass";

        // Build indexes
        // subject → set of predicates
        let mut subj_preds: HashMap<&str, HashSet<&str>> = HashMap::new();
        // predicate → vec<(subject, object)>
        let mut pred_index: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
        // subject → set of classes
        let mut subj_classes: HashMap<&str, HashSet<&str>> = HashMap::new();

        for (subj, pred, obj) in triples {
            if pred == RDF_TYPE {
                subj_classes
                    .entry(subj.as_str())
                    .or_default()
                    .insert(obj.as_str());
            } else {
                subj_preds
                    .entry(subj.as_str())
                    .or_default()
                    .insert(pred.as_str());
            }
            pred_index
                .entry(pred.as_str())
                .or_default()
                .push((subj.as_str(), obj.as_str()));
        }

        let total_subjects = subj_preds.len().max(subj_classes.len());
        let total_predicates = pred_index.len();

        let mut detected: Vec<DetectedPattern> = Vec::new();

        if self.config.detect_star_shaped {
            detected.extend(self.detect_star_shaped(&subj_preds, total_subjects));
        }

        if self.config.detect_functional {
            detected.extend(self.detect_functional_properties(&pred_index, total_subjects));
        }

        if self.config.detect_co_occurrence {
            detected.extend(self.detect_co_occurrences(&subj_preds, total_subjects));
        }

        if self.config.detect_class_hierarchy {
            detected.extend(self.detect_class_hierarchy(&pred_index, RDFS_SUB_CLASS, OWL_EQUIV));
        }

        if self.config.detect_shared_profiles {
            detected.extend(self.detect_shared_profiles(&subj_preds));
        }

        // Build high-level graph patterns: group by kind
        let graph_patterns = self.build_graph_patterns(&detected);

        PatternDetectionReport {
            patterns: detected,
            graph_patterns,
            subjects_analysed: total_subjects,
            predicates_analysed: total_predicates,
        }
    }

    // ── Internal detection helpers ────────────────────────────────────────────

    fn detect_star_shaped(
        &self,
        subj_preds: &HashMap<&str, HashSet<&str>>,
        total_subjects: usize,
    ) -> Vec<DetectedPattern> {
        if total_subjects == 0 {
            return vec![];
        }
        // A subject is star-shaped if it has >= 3 distinct outgoing predicates.
        const STAR_MIN_PREDS: usize = 3;
        let star_subjects: Vec<&str> = subj_preds
            .iter()
            .filter(|(_, preds)| preds.len() >= STAR_MIN_PREDS)
            .map(|(s, _)| *s)
            .collect();

        if star_subjects.is_empty() {
            return vec![];
        }

        let support = star_subjects.len() as f64 / total_subjects as f64;
        vec![DetectedPattern {
            kind: PatternKind::StarShaped,
            primary_entity: format!("{} subjects", star_subjects.len()),
            secondary_entity: None,
            support,
            evidence_count: star_subjects.len(),
            description: format!(
                "{} subjects have >= {} outgoing properties (star-shaped topology)",
                star_subjects.len(),
                STAR_MIN_PREDS
            ),
        }]
    }

    fn detect_functional_properties(
        &self,
        pred_index: &HashMap<&str, Vec<(&str, &str)>>,
        total_subjects: usize,
    ) -> Vec<DetectedPattern> {
        if total_subjects == 0 {
            return vec![];
        }
        let mut result = Vec::new();
        for (pred, pairs) in pred_index {
            if pairs.is_empty() {
                continue;
            }
            // Functional: each subject appears at most once
            let mut subj_count: HashMap<&str, usize> = HashMap::new();
            for (subj, _) in pairs {
                *subj_count.entry(subj).or_insert(0) += 1;
            }
            let unique_subj = subj_count.values().filter(|&&c| c == 1).count();
            let uniqueness = unique_subj as f64 / subj_count.len() as f64;

            if uniqueness >= self.config.min_functional_uniqueness {
                let support = subj_count.len() as f64 / total_subjects as f64;
                result.push(DetectedPattern {
                    kind: PatternKind::FunctionalProperty,
                    primary_entity: pred.to_string(),
                    secondary_entity: None,
                    support,
                    evidence_count: pairs.len(),
                    description: format!(
                        "Property <{}> is functional (uniqueness {:.1}%)",
                        pred,
                        uniqueness * 100.0
                    ),
                });
            }

            // Inverse-functional: each object appears at most once
            let mut obj_count: HashMap<&str, usize> = HashMap::new();
            for (_, obj) in pairs {
                *obj_count.entry(obj).or_insert(0) += 1;
            }
            let unique_obj = obj_count.values().filter(|&&c| c == 1).count();
            let inv_uniqueness = unique_obj as f64 / obj_count.len() as f64;

            if inv_uniqueness >= self.config.min_functional_uniqueness {
                let support = obj_count.len() as f64 / total_subjects as f64;
                result.push(DetectedPattern {
                    kind: PatternKind::InverseFunctionalProperty,
                    primary_entity: pred.to_string(),
                    secondary_entity: None,
                    support,
                    evidence_count: pairs.len(),
                    description: format!(
                        "Property <{}> is inverse-functional (uniqueness {:.1}%)",
                        pred,
                        inv_uniqueness * 100.0
                    ),
                });
            }
        }
        result
    }

    fn detect_co_occurrences(
        &self,
        subj_preds: &HashMap<&str, HashSet<&str>>,
        total_subjects: usize,
    ) -> Vec<DetectedPattern> {
        if total_subjects == 0 || subj_preds.is_empty() {
            return vec![];
        }
        // Collect all predicates
        let all_preds: HashSet<&str> = subj_preds
            .values()
            .flat_map(|s| s.iter().copied())
            .collect();
        let pred_vec: Vec<&str> = all_preds.into_iter().collect();

        let mut result = Vec::new();

        // Only check pairs (O(n^2) but bounded by number of distinct predicates)
        for i in 0..pred_vec.len() {
            for j in (i + 1)..pred_vec.len() {
                let p1 = pred_vec[i];
                let p2 = pred_vec[j];
                let co_count = subj_preds
                    .values()
                    .filter(|preds| preds.contains(p1) && preds.contains(p2))
                    .count();

                let support = co_count as f64 / total_subjects as f64;
                if support >= self.config.min_co_occurrence_support {
                    result.push(DetectedPattern {
                        kind: PatternKind::CoOccurrence,
                        primary_entity: p1.to_string(),
                        secondary_entity: Some(p2.to_string()),
                        support,
                        evidence_count: co_count,
                        description: format!(
                            "Properties <{}> and <{}> co-occur in {:.1}% of subjects",
                            p1,
                            p2,
                            support * 100.0
                        ),
                    });
                }
            }
        }
        result
    }

    fn detect_class_hierarchy(
        &self,
        pred_index: &HashMap<&str, Vec<(&str, &str)>>,
        rdfs_subclass: &str,
        owl_equiv: &str,
    ) -> Vec<DetectedPattern> {
        let mut result = Vec::new();

        for &pred in &[rdfs_subclass, owl_equiv] {
            if let Some(pairs) = pred_index.get(pred) {
                for (subj, obj) in pairs {
                    result.push(DetectedPattern {
                        kind: PatternKind::ClassHierarchy,
                        primary_entity: subj.to_string(),
                        secondary_entity: Some(obj.to_string()),
                        support: 1.0,
                        evidence_count: 1,
                        description: format!(
                            "Class <{}> {} <{}>",
                            subj,
                            if pred == rdfs_subclass {
                                "is subclass of"
                            } else {
                                "is equivalent to"
                            },
                            obj
                        ),
                    });
                }
            }
        }
        result
    }

    fn detect_shared_profiles(
        &self,
        subj_preds: &HashMap<&str, HashSet<&str>>,
    ) -> Vec<DetectedPattern> {
        // Group subjects by their sorted predicate-set fingerprint
        let mut profile_counts: HashMap<Vec<&str>, usize> = HashMap::new();
        for preds in subj_preds.values() {
            let mut sorted: Vec<&str> = preds.iter().copied().collect();
            sorted.sort();
            *profile_counts.entry(sorted).or_insert(0) += 1;
        }

        let mut result = Vec::new();
        for (profile, count) in profile_counts {
            if count >= self.config.min_shared_profile_count {
                let support = count as f64 / subj_preds.len() as f64;
                let pred_list = profile.join(", ");
                result.push(DetectedPattern {
                    kind: PatternKind::SharedPropertyProfile,
                    primary_entity: format!("profile({} predicates)", profile.len()),
                    secondary_entity: None,
                    support,
                    evidence_count: count,
                    description: format!(
                        "{} subjects share the same property set: [{}]",
                        count, pred_list
                    ),
                });
            }
        }
        result
    }

    /// Group detected patterns into higher-level `GraphPattern` instances.
    fn build_graph_patterns(&self, patterns: &[DetectedPattern]) -> Vec<GraphPattern> {
        let mut by_kind: HashMap<String, Vec<DetectedPattern>> = HashMap::new();
        for p in patterns {
            by_kind
                .entry(p.kind.label().to_string())
                .or_default()
                .push(p.clone());
        }

        by_kind
            .into_iter()
            .map(|(label, pats)| {
                let avg_support = pats.iter().map(|p| p.support).sum::<f64>() / pats.len() as f64;
                let kind = pats[0].kind.clone();
                GraphPattern {
                    id: format!("gp-{}-{}", label, pats.len()),
                    kind,
                    support: avg_support,
                    associated_class: None,
                    patterns: pats,
                }
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rdf_type() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()
    }

    fn sample_triples() -> Vec<(String, String, String)> {
        vec![
            // Alice
            (
                "http://ex.org/alice".into(),
                rdf_type(),
                "http://ex.org/Person".into(),
            ),
            (
                "http://ex.org/alice".into(),
                "http://ex.org/name".into(),
                "\"Alice\"".into(),
            ),
            (
                "http://ex.org/alice".into(),
                "http://ex.org/age".into(),
                "\"30\"".into(),
            ),
            (
                "http://ex.org/alice".into(),
                "http://ex.org/email".into(),
                "\"a@x.com\"".into(),
            ),
            // Bob
            (
                "http://ex.org/bob".into(),
                rdf_type(),
                "http://ex.org/Person".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/name".into(),
                "\"Bob\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/age".into(),
                "\"25\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/email".into(),
                "\"b@x.com\"".into(),
            ),
        ]
    }

    #[test]
    fn test_detect_returns_subjects_count() {
        let detector = PatternDetector::default_config();
        let report = detector.detect(&sample_triples());
        assert!(report.subjects_analysed >= 2, "expected >=2 subjects");
    }

    #[test]
    fn test_shared_profile_detected() {
        let detector = PatternDetector::default_config();
        let report = detector.detect(&sample_triples());
        let has_shared = report
            .patterns
            .iter()
            .any(|p| p.kind == PatternKind::SharedPropertyProfile);
        assert!(has_shared, "expected a SharedPropertyProfile pattern");
    }

    #[test]
    fn test_co_occurrence_detected() {
        let config = PatternDetectionConfig {
            min_co_occurrence_support: 0.5,
            ..Default::default()
        };
        let detector = PatternDetector::new(config);
        let report = detector.detect(&sample_triples());
        let has_cooccur = report
            .patterns
            .iter()
            .any(|p| p.kind == PatternKind::CoOccurrence);
        assert!(has_cooccur, "expected a CoOccurrence pattern");
    }

    #[test]
    fn test_functional_property_detected() {
        // email is unique per person → functional
        let config = PatternDetectionConfig {
            min_functional_uniqueness: 0.9,
            ..Default::default()
        };
        let detector = PatternDetector::new(config);
        let report = detector.detect(&sample_triples());
        let func_preds: Vec<&str> = report
            .patterns
            .iter()
            .filter(|p| p.kind == PatternKind::FunctionalProperty)
            .map(|p| p.primary_entity.as_str())
            .collect();
        // name, age, email are each used once per subject ⇒ functional
        assert!(
            !func_preds.is_empty(),
            "expected at least one functional property"
        );
    }

    #[test]
    fn test_star_shaped_detected() {
        let detector = PatternDetector::default_config();
        let report = detector.detect(&sample_triples());
        let has_star = report
            .patterns
            .iter()
            .any(|p| p.kind == PatternKind::StarShaped);
        assert!(has_star, "expected star-shaped pattern");
    }

    #[test]
    fn test_class_hierarchy_detected() {
        let detector = PatternDetector::default_config();
        let triples = vec![(
            "http://ex.org/Employee".into(),
            "http://www.w3.org/2000/01/rdf-schema#subClassOf".into(),
            "http://ex.org/Person".into(),
        )];
        let report = detector.detect(&triples);
        let has_hier = report
            .patterns
            .iter()
            .any(|p| p.kind == PatternKind::ClassHierarchy);
        assert!(has_hier, "expected class hierarchy pattern");
    }

    #[test]
    fn test_empty_triples_returns_empty_report() {
        let detector = PatternDetector::default_config();
        let report = detector.detect(&[]);
        assert_eq!(report.subjects_analysed, 0);
        assert!(report.patterns.is_empty());
    }

    #[test]
    fn test_graph_patterns_built() {
        let detector = PatternDetector::default_config();
        let report = detector.detect(&sample_triples());
        // Each distinct kind should produce one GraphPattern
        assert!(!report.graph_patterns.is_empty());
    }
}
