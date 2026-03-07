//! Schema alignment between ontologies.
//!
//! Discovers mappings between classes and properties from two ontologies using
//! string similarity (normalised Levenshtein), structural similarity, and
//! configurable confidence thresholds.
//!
//! Provides:
//! - Class mapping discovery (find equivalent classes)
//! - Property mapping (find corresponding properties)
//! - Similarity scoring (string similarity, structural similarity)
//! - Alignment confidence computation
//! - Bidirectional mapping support
//! - 1:1 and 1:N mapping handling
//! - Alignment serialization (EDOAL-inspired format)
//! - Conflict detection in alignments

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Kind of ontology element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementKind {
    Class,
    Property,
}

/// A single ontology element (class or property).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OntologyElement {
    /// IRI / local name of the element.
    pub name: String,
    /// Kind: class or property.
    pub kind: ElementKind,
    /// Optional parent class or domain/range for structural context.
    pub parents: Vec<String>,
}

impl OntologyElement {
    /// Create a new class element.
    pub fn class(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: ElementKind::Class,
            parents: Vec::new(),
        }
    }

    /// Create a new property element.
    pub fn property(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: ElementKind::Property,
            parents: Vec::new(),
        }
    }

    /// Set the parent(s) for structural context.
    pub fn with_parents(mut self, parents: Vec<String>) -> Self {
        self.parents = parents;
        self
    }
}

/// A discovered mapping between two elements.
#[derive(Debug, Clone)]
pub struct AlignmentMapping {
    /// Source element.
    pub source: OntologyElement,
    /// Target element.
    pub target: OntologyElement,
    /// Confidence score in [0, 1].
    pub confidence: f64,
    /// How the similarity was determined.
    pub method: SimilarityMethod,
    /// Direction of the mapping.
    pub direction: MappingDirection,
}

/// The similarity method used to derive a mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMethod {
    /// Normalised Levenshtein string similarity on element names.
    StringSimilarity,
    /// Structural similarity based on parent hierarchy overlap.
    StructuralSimilarity,
    /// Combined string + structural.
    Combined,
}

/// Direction of a mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingDirection {
    /// Source → Target only.
    Forward,
    /// Target → Source only.
    Backward,
    /// Both directions.
    Bidirectional,
}

/// Mapping cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingCardinality {
    /// Each source maps to at most one target.
    OneToOne,
    /// Each source may map to multiple targets.
    OneToMany,
}

/// A conflict detected in the alignment.
#[derive(Debug, Clone)]
pub struct AlignmentConflict {
    /// The element that has conflicting mappings.
    pub element: String,
    /// The conflicting targets.
    pub conflicting_targets: Vec<String>,
    /// Description of the conflict.
    pub description: String,
}

/// Configuration for the schema aligner.
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Minimum confidence threshold for accepting a mapping.
    pub confidence_threshold: f64,
    /// Weight for string similarity (0–1).
    pub string_weight: f64,
    /// Weight for structural similarity (0–1).
    pub structural_weight: f64,
    /// Mapping cardinality.
    pub cardinality: MappingCardinality,
    /// Whether to produce bidirectional mappings.
    pub bidirectional: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            string_weight: 0.7,
            structural_weight: 0.3,
            cardinality: MappingCardinality::OneToOne,
            bidirectional: true,
        }
    }
}

/// Complete alignment result.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Discovered mappings.
    pub mappings: Vec<AlignmentMapping>,
    /// Detected conflicts.
    pub conflicts: Vec<AlignmentConflict>,
    /// Number of source elements considered.
    pub source_count: usize,
    /// Number of target elements considered.
    pub target_count: usize,
    /// Mean confidence across accepted mappings.
    pub mean_confidence: f64,
}

// ---------------------------------------------------------------------------
// SchemaAligner
// ---------------------------------------------------------------------------

/// Engine for discovering alignments between two ontology schemas.
pub struct SchemaAligner {
    config: AlignmentConfig,
    total_alignments: u64,
}

impl SchemaAligner {
    /// Create a new schema aligner with the given configuration.
    pub fn new(config: AlignmentConfig) -> Self {
        Self {
            config,
            total_alignments: 0,
        }
    }

    /// Discover alignments between `source` and `target` element sets.
    pub fn align(
        &mut self,
        source: &[OntologyElement],
        target: &[OntologyElement],
    ) -> AlignmentResult {
        let mut raw_mappings: Vec<AlignmentMapping> = Vec::new();

        for src in source {
            for tgt in target {
                // Only compare same-kind elements.
                if src.kind != tgt.kind {
                    continue;
                }

                let str_sim = normalized_levenshtein(&src.name, &tgt.name);
                let struct_sim = structural_similarity(src, tgt);
                let combined = self.config.string_weight * str_sim
                    + self.config.structural_weight * struct_sim;

                if combined >= self.config.confidence_threshold {
                    let method = if self.config.structural_weight == 0.0 {
                        SimilarityMethod::StringSimilarity
                    } else if self.config.string_weight == 0.0 {
                        SimilarityMethod::StructuralSimilarity
                    } else {
                        SimilarityMethod::Combined
                    };

                    let direction = if self.config.bidirectional {
                        MappingDirection::Bidirectional
                    } else {
                        MappingDirection::Forward
                    };

                    raw_mappings.push(AlignmentMapping {
                        source: src.clone(),
                        target: tgt.clone(),
                        confidence: combined,
                        method,
                        direction,
                    });
                }
            }
        }

        // Apply cardinality constraint.
        let mappings = match self.config.cardinality {
            MappingCardinality::OneToOne => self.enforce_one_to_one(raw_mappings),
            MappingCardinality::OneToMany => raw_mappings,
        };

        // Detect conflicts.
        let conflicts = self.detect_conflicts(&mappings);

        let mean_confidence = if mappings.is_empty() {
            0.0
        } else {
            mappings.iter().map(|m| m.confidence).sum::<f64>() / mappings.len() as f64
        };

        self.total_alignments += 1;

        AlignmentResult {
            mappings,
            conflicts,
            source_count: source.len(),
            target_count: target.len(),
            mean_confidence,
        }
    }

    /// Discover class mappings only (filter by ElementKind::Class).
    pub fn align_classes(
        &mut self,
        source: &[OntologyElement],
        target: &[OntologyElement],
    ) -> Vec<AlignmentMapping> {
        let src_classes: Vec<OntologyElement> = source
            .iter()
            .filter(|e| e.kind == ElementKind::Class)
            .cloned()
            .collect();
        let tgt_classes: Vec<OntologyElement> = target
            .iter()
            .filter(|e| e.kind == ElementKind::Class)
            .cloned()
            .collect();
        self.align(&src_classes, &tgt_classes).mappings
    }

    /// Discover property mappings only.
    pub fn align_properties(
        &mut self,
        source: &[OntologyElement],
        target: &[OntologyElement],
    ) -> Vec<AlignmentMapping> {
        let src_props: Vec<OntologyElement> = source
            .iter()
            .filter(|e| e.kind == ElementKind::Property)
            .cloned()
            .collect();
        let tgt_props: Vec<OntologyElement> = target
            .iter()
            .filter(|e| e.kind == ElementKind::Property)
            .cloned()
            .collect();
        self.align(&src_props, &tgt_props).mappings
    }

    /// Serialize the alignment result as EDOAL-inspired XML string.
    pub fn serialize_edoal(result: &AlignmentResult) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<Alignment>\n");
        xml.push_str(&format!(
            "  <level>0</level>\n  <type>{}</type>\n",
            if result
                .mappings
                .iter()
                .any(|m| m.source.kind == ElementKind::Class)
            {
                "class"
            } else {
                "property"
            }
        ));
        for mapping in &result.mappings {
            xml.push_str("  <map>\n");
            xml.push_str("    <Cell>\n");
            xml.push_str(&format!(
                "      <entity1 rdf:resource=\"{}\"/>\n",
                mapping.source.name
            ));
            xml.push_str(&format!(
                "      <entity2 rdf:resource=\"{}\"/>\n",
                mapping.target.name
            ));
            xml.push_str(&format!(
                "      <measure>{:.4}</measure>\n",
                mapping.confidence
            ));
            let rel = match mapping.direction {
                MappingDirection::Bidirectional => "=",
                MappingDirection::Forward => "&gt;",
                MappingDirection::Backward => "&lt;",
            };
            xml.push_str(&format!("      <relation>{rel}</relation>\n"));
            xml.push_str("    </Cell>\n");
            xml.push_str("  </map>\n");
        }
        xml.push_str("</Alignment>\n");
        xml
    }

    /// Total number of alignment operations performed.
    pub fn total_alignments(&self) -> u64 {
        self.total_alignments
    }

    // --- private helpers ---

    /// Keep only the highest-confidence mapping per source element.
    fn enforce_one_to_one(&self, mut mappings: Vec<AlignmentMapping>) -> Vec<AlignmentMapping> {
        // Sort by confidence descending.
        mappings.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut used_sources: HashSet<String> = HashSet::new();
        let mut used_targets: HashSet<String> = HashSet::new();
        let mut result = Vec::new();

        for m in mappings {
            if !used_sources.contains(&m.source.name) && !used_targets.contains(&m.target.name) {
                used_sources.insert(m.source.name.clone());
                used_targets.insert(m.target.name.clone());
                result.push(m);
            }
        }
        result
    }

    /// Detect conflicting mappings (same source mapped to multiple targets in 1:1 mode,
    /// or bidirectional inconsistencies).
    fn detect_conflicts(&self, mappings: &[AlignmentMapping]) -> Vec<AlignmentConflict> {
        let mut source_targets: HashMap<String, Vec<String>> = HashMap::new();
        for m in mappings {
            source_targets
                .entry(m.source.name.clone())
                .or_default()
                .push(m.target.name.clone());
        }

        let mut conflicts = Vec::new();
        for (src, targets) in &source_targets {
            if targets.len() > 1 {
                conflicts.push(AlignmentConflict {
                    element: src.clone(),
                    conflicting_targets: targets.clone(),
                    description: format!(
                        "Source '{}' maps to {} targets: {}",
                        src,
                        targets.len(),
                        targets.join(", ")
                    ),
                });
            }
        }
        conflicts
    }
}

// ---------------------------------------------------------------------------
// Free functions – similarity metrics
// ---------------------------------------------------------------------------

/// Normalised Levenshtein similarity in [0, 1].
///
/// Returns 1.0 for identical strings and 0.0 for completely different strings.
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein_distance(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

/// Standard Levenshtein edit distance.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Structural similarity based on overlap of parent elements.
///
/// Jaccard similarity of parent sets, or 0.0 if both have no parents.
pub fn structural_similarity(a: &OntologyElement, b: &OntologyElement) -> f64 {
    if a.parents.is_empty() && b.parents.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<&String> = a.parents.iter().collect();
    let set_b: HashSet<&String> = b.parents.iter().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_aligner() -> SchemaAligner {
        SchemaAligner::new(AlignmentConfig::default())
    }

    fn strict_aligner() -> SchemaAligner {
        SchemaAligner::new(AlignmentConfig {
            confidence_threshold: 0.9,
            ..AlignmentConfig::default()
        })
    }

    // --- normalized_levenshtein ---

    #[test]
    fn test_levenshtein_identical() {
        let sim = normalized_levenshtein("Person", "Person");
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_levenshtein_completely_different() {
        let sim = normalized_levenshtein("abc", "xyz");
        assert!((sim - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_levenshtein_partial_match() {
        let sim = normalized_levenshtein("Person", "Persons");
        // Distance = 1, max_len = 7, similarity = 6/7 ≈ 0.857
        assert!(sim > 0.85 && sim < 0.87);
    }

    #[test]
    fn test_levenshtein_empty_strings() {
        assert!((normalized_levenshtein("", "") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_levenshtein_one_empty() {
        assert!((normalized_levenshtein("abc", "") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_levenshtein_symmetric() {
        let ab = normalized_levenshtein("cat", "car");
        let ba = normalized_levenshtein("car", "cat");
        assert!((ab - ba).abs() < 1e-10);
    }

    // --- structural_similarity ---

    #[test]
    fn test_structural_sim_identical_parents() {
        let a = OntologyElement::class("A").with_parents(vec!["X".into(), "Y".into()]);
        let b = OntologyElement::class("B").with_parents(vec!["X".into(), "Y".into()]);
        assert!((structural_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_structural_sim_disjoint_parents() {
        let a = OntologyElement::class("A").with_parents(vec!["X".into()]);
        let b = OntologyElement::class("B").with_parents(vec!["Y".into()]);
        assert!((structural_similarity(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_structural_sim_no_parents() {
        let a = OntologyElement::class("A");
        let b = OntologyElement::class("B");
        assert!((structural_similarity(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_structural_sim_partial_overlap() {
        let a = OntologyElement::class("A").with_parents(vec!["X".into(), "Y".into()]);
        let b = OntologyElement::class("B").with_parents(vec!["Y".into(), "Z".into()]);
        // Jaccard: 1/3
        assert!((structural_similarity(&a, &b) - 1.0 / 3.0).abs() < 1e-10);
    }

    // --- SchemaAligner::align ---

    #[test]
    fn test_align_identical_classes() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("Person")];
        let target = vec![OntologyElement::class("Person")];
        let result = aligner.align(&source, &target);
        assert_eq!(result.mappings.len(), 1);
        // String sim = 1.0, structural sim = 0.0 (no parents).
        // confidence = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert!((result.mappings[0].confidence - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_align_no_match_below_threshold() {
        let mut aligner = strict_aligner();
        let source = vec![OntologyElement::class("Abc")];
        let target = vec![OntologyElement::class("Xyz")];
        let result = aligner.align(&source, &target);
        assert!(result.mappings.is_empty());
    }

    #[test]
    fn test_align_different_kinds_not_matched() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("Name")];
        let target = vec![OntologyElement::property("Name")];
        let result = aligner.align(&source, &target);
        assert!(
            result.mappings.is_empty(),
            "class should not match property"
        );
    }

    #[test]
    fn test_align_multiple_sources_and_targets() {
        let mut aligner = default_aligner();
        let source = vec![
            OntologyElement::class("Person"),
            OntologyElement::class("Organization"),
        ];
        let target = vec![
            OntologyElement::class("Person"),
            OntologyElement::class("Organisation"),
        ];
        let result = aligner.align(&source, &target);
        assert!(!result.mappings.is_empty());
    }

    #[test]
    fn test_align_result_source_target_counts() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("A"), OntologyElement::class("B")];
        let target = vec![OntologyElement::class("C")];
        let result = aligner.align(&source, &target);
        assert_eq!(result.source_count, 2);
        assert_eq!(result.target_count, 1);
    }

    #[test]
    fn test_align_mean_confidence_correct() {
        let mut aligner = SchemaAligner::new(AlignmentConfig {
            confidence_threshold: 0.0,
            ..AlignmentConfig::default()
        });
        let source = vec![OntologyElement::class("Person")];
        let target = vec![OntologyElement::class("Person")];
        let result = aligner.align(&source, &target);
        assert!(!result.mappings.is_empty());
        assert!((result.mean_confidence - result.mappings[0].confidence).abs() < 1e-10);
    }

    // --- one-to-one enforcement ---

    #[test]
    fn test_one_to_one_keeps_best_match() {
        let mut aligner = SchemaAligner::new(AlignmentConfig {
            confidence_threshold: 0.0,
            cardinality: MappingCardinality::OneToOne,
            ..AlignmentConfig::default()
        });
        let source = vec![OntologyElement::class("Cat")];
        let target = vec![
            OntologyElement::class("Cat"),
            OntologyElement::class("Car"), // similar but not exact
        ];
        let result = aligner.align(&source, &target);
        assert_eq!(result.mappings.len(), 1, "1:1 should keep only best");
        assert_eq!(result.mappings[0].target.name, "Cat");
    }

    #[test]
    fn test_one_to_many_keeps_all() {
        let mut aligner = SchemaAligner::new(AlignmentConfig {
            confidence_threshold: 0.0,
            cardinality: MappingCardinality::OneToMany,
            ..AlignmentConfig::default()
        });
        let source = vec![OntologyElement::class("Cat")];
        let target = vec![OntologyElement::class("Cat"), OntologyElement::class("Car")];
        let result = aligner.align(&source, &target);
        assert!(
            result.mappings.len() >= 2,
            "1:N should keep all above threshold"
        );
    }

    // --- bidirectional ---

    #[test]
    fn test_bidirectional_mapping_direction() {
        let mut aligner = SchemaAligner::new(AlignmentConfig {
            bidirectional: true,
            ..AlignmentConfig::default()
        });
        let source = vec![OntologyElement::class("Person")];
        let target = vec![OntologyElement::class("Person")];
        let result = aligner.align(&source, &target);
        assert_eq!(
            result.mappings[0].direction,
            MappingDirection::Bidirectional
        );
    }

    #[test]
    fn test_forward_only_mapping_direction() {
        let mut aligner = SchemaAligner::new(AlignmentConfig {
            bidirectional: false,
            ..AlignmentConfig::default()
        });
        let source = vec![OntologyElement::class("Person")];
        let target = vec![OntologyElement::class("Person")];
        let result = aligner.align(&source, &target);
        assert_eq!(result.mappings[0].direction, MappingDirection::Forward);
    }

    // --- align_classes / align_properties ---

    #[test]
    fn test_align_classes_filters_properties() {
        let mut aligner = default_aligner();
        let source = vec![
            OntologyElement::class("Person"),
            OntologyElement::property("name"),
        ];
        let target = vec![
            OntologyElement::class("Person"),
            OntologyElement::property("name"),
        ];
        let class_mappings = aligner.align_classes(&source, &target);
        assert_eq!(class_mappings.len(), 1);
        assert_eq!(class_mappings[0].source.kind, ElementKind::Class);
    }

    #[test]
    fn test_align_properties_filters_classes() {
        let mut aligner = default_aligner();
        let source = vec![
            OntologyElement::class("Person"),
            OntologyElement::property("name"),
        ];
        let target = vec![
            OntologyElement::class("Person"),
            OntologyElement::property("name"),
        ];
        let prop_mappings = aligner.align_properties(&source, &target);
        assert_eq!(prop_mappings.len(), 1);
        assert_eq!(prop_mappings[0].source.kind, ElementKind::Property);
    }

    // --- conflict detection ---

    #[test]
    fn test_no_conflicts_for_one_to_one() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("A")];
        let target = vec![OntologyElement::class("A")];
        let result = aligner.align(&source, &target);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn test_conflicts_detected_for_one_to_many() {
        let mut aligner = SchemaAligner::new(AlignmentConfig {
            confidence_threshold: 0.0,
            cardinality: MappingCardinality::OneToMany,
            ..AlignmentConfig::default()
        });
        let source = vec![OntologyElement::class("Cat")];
        let target = vec![OntologyElement::class("Cat"), OntologyElement::class("Car")];
        let result = aligner.align(&source, &target);
        // "Cat" maps to both "Cat" and "Car" -> conflict
        assert!(!result.conflicts.is_empty());
        assert_eq!(result.conflicts[0].conflicting_targets.len(), 2);
    }

    // --- serialization ---

    #[test]
    fn test_serialize_edoal_contains_xml_header() {
        let result = AlignmentResult {
            mappings: vec![],
            conflicts: vec![],
            source_count: 0,
            target_count: 0,
            mean_confidence: 0.0,
        };
        let xml = SchemaAligner::serialize_edoal(&result);
        assert!(xml.starts_with("<?xml"));
    }

    #[test]
    fn test_serialize_edoal_contains_mapping() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("Person")];
        let target = vec![OntologyElement::class("Person")];
        let result = aligner.align(&source, &target);
        let xml = SchemaAligner::serialize_edoal(&result);
        assert!(xml.contains("Person"), "EDOAL should contain element name");
        assert!(xml.contains("<measure>"), "EDOAL should contain measure");
    }

    #[test]
    fn test_serialize_edoal_relation_bidirectional() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("Person")];
        let target = vec![OntologyElement::class("Person")];
        let result = aligner.align(&source, &target);
        let xml = SchemaAligner::serialize_edoal(&result);
        assert!(xml.contains("<relation>=</relation>"));
    }

    // --- total_alignments tracking ---

    #[test]
    fn test_total_alignments_initially_zero() {
        let aligner = default_aligner();
        assert_eq!(aligner.total_alignments(), 0);
    }

    #[test]
    fn test_total_alignments_increments() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("A")];
        let target = vec![OntologyElement::class("B")];
        aligner.align(&source, &target);
        aligner.align(&source, &target);
        assert_eq!(aligner.total_alignments(), 2);
    }

    // --- empty inputs ---

    #[test]
    fn test_align_empty_source() {
        let mut aligner = default_aligner();
        let result = aligner.align(&[], &[OntologyElement::class("A")]);
        assert!(result.mappings.is_empty());
        assert_eq!(result.source_count, 0);
    }

    #[test]
    fn test_align_empty_target() {
        let mut aligner = default_aligner();
        let result = aligner.align(&[OntologyElement::class("A")], &[]);
        assert!(result.mappings.is_empty());
        assert_eq!(result.target_count, 0);
    }

    #[test]
    fn test_align_both_empty() {
        let mut aligner = default_aligner();
        let result = aligner.align(&[], &[]);
        assert!(result.mappings.is_empty());
        assert_eq!(result.mean_confidence, 0.0);
    }

    // --- structural weight only ---

    #[test]
    fn test_structural_only_alignment() {
        let mut aligner = SchemaAligner::new(AlignmentConfig {
            string_weight: 0.0,
            structural_weight: 1.0,
            confidence_threshold: 0.3,
            ..AlignmentConfig::default()
        });
        let source = vec![OntologyElement::class("Foo").with_parents(vec!["X".into()])];
        let target = vec![OntologyElement::class("Bar").with_parents(vec!["X".into()])];
        let result = aligner.align(&source, &target);
        assert_eq!(result.mappings.len(), 1);
        assert_eq!(
            result.mappings[0].method,
            SimilarityMethod::StructuralSimilarity
        );
    }

    // --- combined method ---

    #[test]
    fn test_combined_method_used() {
        let mut aligner = default_aligner();
        let source = vec![OntologyElement::class("Person").with_parents(vec!["Agent".into()])];
        let target = vec![OntologyElement::class("Person").with_parents(vec!["Agent".into()])];
        let result = aligner.align(&source, &target);
        assert_eq!(result.mappings[0].method, SimilarityMethod::Combined);
    }

    // --- OntologyElement constructors ---

    #[test]
    fn test_class_constructor() {
        let e = OntologyElement::class("Person");
        assert_eq!(e.kind, ElementKind::Class);
        assert_eq!(e.name, "Person");
        assert!(e.parents.is_empty());
    }

    #[test]
    fn test_property_constructor() {
        let e = OntologyElement::property("name");
        assert_eq!(e.kind, ElementKind::Property);
        assert_eq!(e.name, "name");
    }

    #[test]
    fn test_with_parents_builder() {
        let e = OntologyElement::class("Student").with_parents(vec!["Person".into()]);
        assert_eq!(e.parents, vec!["Person".to_string()]);
    }

    // --- AlignmentConfig default ---

    #[test]
    fn test_default_config_values() {
        let config = AlignmentConfig::default();
        assert!((config.confidence_threshold - 0.5).abs() < 1e-10);
        assert!((config.string_weight - 0.7).abs() < 1e-10);
        assert!((config.structural_weight - 0.3).abs() < 1e-10);
        assert!(config.bidirectional);
        assert_eq!(config.cardinality, MappingCardinality::OneToOne);
    }
}
