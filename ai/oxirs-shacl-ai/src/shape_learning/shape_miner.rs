//! Statistical Shape Mining from RDF Data
//!
//! Mines SHACL-compatible shapes from raw RDF triple data using frequency analysis
//! and statistical thresholds. For each detected class, it aggregates property
//! usage, cardinality distributions, datatype observations, and node-kind
//! distributions to produce `MinedShape` instances whose `support` and `confidence`
//! fields reflect how many class instances actually follow the discovered pattern.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Kind of RDF node that appears as an object value of a property.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    /// An IRI (named node) object.
    Iri,
    /// A plain or typed literal.
    Literal,
    /// A blank node.
    BlankNode,
    /// A mix of the above kinds was observed.
    Mixed,
}

/// A single property constraint derived from statistical analysis of RDF data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyConstraint {
    /// Predicate URI (string form) for this constraint.
    pub predicate: String,
    /// Minimum occurrence count (optional; `None` means no lower bound).
    pub min_count: Option<u32>,
    /// Maximum occurrence count (optional; `None` means no upper bound).
    pub max_count: Option<u32>,
    /// Most-common datatype URI observed for literal objects (if any).
    pub datatype: Option<String>,
    /// Predominant node kind for object values.
    pub node_kind: Option<NodeKind>,
    /// Support fraction: fraction of class instances having this property.
    pub support: f64,
    /// Confidence: how consistently instances honour the derived constraint.
    pub confidence: f64,
}

/// A shape mined from an RDF dataset, targeting one class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinedShape {
    /// The target class URI.
    pub target_class: String,
    /// Property constraints that make up this shape.
    pub properties: Vec<PropertyConstraint>,
    /// Fraction of class instances that satisfy all constraints in this shape.
    pub support: f64,
    /// Weighted average constraint confidence.
    pub confidence: f64,
    /// Number of instances of `target_class` observed in the data.
    pub instance_count: usize,
}

/// Configuration parameters for `ShapeMiner`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeMinerConfig {
    /// Minimum support threshold (0.0 – 1.0).
    /// Only shapes supported by at least this fraction of class instances are emitted.
    pub min_support: f64,
    /// Minimum confidence threshold (0.0 – 1.0).
    /// Only constraints with at least this confidence are included in shapes.
    pub min_confidence: f64,
    /// Maximum number of shapes to return (0 means unlimited).
    pub max_shapes: usize,
    /// Whether to attempt datatype inference from literal object values.
    pub infer_datatypes: bool,
    /// Whether to infer node-kind from object URI / literal / blank-node patterns.
    pub infer_node_kinds: bool,
}

impl Default for ShapeMinerConfig {
    fn default() -> Self {
        Self {
            min_support: 0.2,
            min_confidence: 0.7,
            max_shapes: 0,
            infer_datatypes: true,
            infer_node_kinds: true,
        }
    }
}

/// Runtime statistics collected by one `mine_shapes` call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeMiningStats {
    /// Total triples processed.
    pub triples_processed: usize,
    /// Number of distinct classes detected.
    pub classes_detected: usize,
    /// Number of shapes emitted (after threshold filtering).
    pub shapes_mined: usize,
    /// Number of property constraints that were discarded (below threshold).
    pub constraints_discarded: usize,
}

/// Summary report produced by one mining run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeMiningReport {
    /// Mined shapes ordered by `support` descending.
    pub shapes: Vec<MinedShape>,
    /// Runtime statistics.
    pub stats: ShapeMiningStats,
}

/// The main shape mining engine.
///
/// # Example
/// ```rust
/// use oxirs_shacl_ai::shape_learning::{ShapeMiner, ShapeMinerConfig};
///
/// let miner = ShapeMiner::new(0.2, 0.7);
/// let triples: Vec<(String, String, String)> = vec![
///     ("http://ex.org/alice".into(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into(), "http://ex.org/Person".into()),
///     ("http://ex.org/alice".into(), "http://ex.org/name".into(), "\"Alice\"".into()),
/// ];
/// let shapes = miner.mine_shapes(&triples);
/// assert!(!shapes.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct ShapeMiner {
    /// Effective configuration.
    pub config: ShapeMinerConfig,
}

impl ShapeMiner {
    /// Create a miner with explicit support and confidence thresholds.
    pub fn new(min_support: f64, min_confidence: f64) -> Self {
        Self {
            config: ShapeMinerConfig {
                min_support: min_support.clamp(0.0, 1.0),
                min_confidence: min_confidence.clamp(0.0, 1.0),
                ..ShapeMinerConfig::default()
            },
        }
    }

    /// Create a miner from a full `ShapeMinerConfig`.
    pub fn with_config(config: ShapeMinerConfig) -> Self {
        Self { config }
    }

    /// Detect all distinct RDF classes that appear as objects of `rdf:type` triples.
    ///
    /// Returns a deduplicated, sorted list of class URIs.
    pub fn detect_classes(&self, triples: &[(String, String, String)]) -> Vec<String> {
        const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let mut classes: HashSet<String> = HashSet::new();
        for (_, pred, obj) in triples {
            if pred == RDF_TYPE {
                classes.insert(obj.clone());
            }
        }
        let mut sorted: Vec<String> = classes.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Mine shapes from a flat list of RDF triples `(subject, predicate, object)`.
    ///
    /// The algorithm:
    /// 1. Detect all classes from `rdf:type` assertions.
    /// 2. For each class, collect its instances.
    /// 3. For each instance, gather predicate–object pairs (excluding `rdf:type`).
    /// 4. Compute per-predicate support and cardinality statistics.
    /// 5. Build `PropertyConstraint` entries that pass the configured thresholds.
    /// 6. Emit a `MinedShape` if it has at least one property constraint.
    pub fn mine_shapes(&self, triples: &[(String, String, String)]) -> Vec<MinedShape> {
        let report = self.mine_shapes_with_report(triples);
        report.shapes
    }

    /// Like `mine_shapes` but also returns runtime statistics.
    pub fn mine_shapes_with_report(
        &self,
        triples: &[(String, String, String)],
    ) -> ShapeMiningReport {
        const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

        let mut stats = ShapeMiningStats {
            triples_processed: triples.len(),
            ..ShapeMiningStats::default()
        };

        // Step 1: build subject → classes mapping
        let mut subject_classes: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (subj, pred, obj) in triples {
            if pred == RDF_TYPE {
                subject_classes
                    .entry(subj.as_str())
                    .or_default()
                    .insert(obj.as_str());
            }
        }

        // Collect distinct classes
        let classes: HashSet<&str> = subject_classes
            .values()
            .flat_map(|s| s.iter().copied())
            .collect();
        stats.classes_detected = classes.len();

        // Step 2: for each class, collect predicate data per instance
        let mut shapes: Vec<MinedShape> = Vec::new();

        for class_uri in &classes {
            // instances belonging to this class
            let instances: Vec<&str> = subject_classes
                .iter()
                .filter(|(_, cls_set)| cls_set.contains(class_uri))
                .map(|(subj, _)| *subj)
                .collect();

            let total_instances = instances.len();
            if total_instances == 0 {
                continue;
            }

            let instance_set: HashSet<&str> = instances.iter().copied().collect();

            // Gather per-instance per-predicate value lists (excluding rdf:type)
            // structure: predicate → instance → vec<object>
            let mut pred_instance_objects: HashMap<&str, HashMap<&str, Vec<&str>>> = HashMap::new();

            for (subj, pred, obj) in triples {
                if pred == RDF_TYPE {
                    continue;
                }
                if instance_set.contains(subj.as_str()) {
                    pred_instance_objects
                        .entry(pred.as_str())
                        .or_default()
                        .entry(subj.as_str())
                        .or_default()
                        .push(obj.as_str());
                }
            }

            let mut property_constraints: Vec<PropertyConstraint> = Vec::new();

            for (predicate, instance_obj_map) in &pred_instance_objects {
                let instances_with_prop = instance_obj_map.len();
                let prop_support = instances_with_prop as f64 / total_instances as f64;

                if prop_support < self.config.min_support {
                    stats.constraints_discarded += 1;
                    continue;
                }

                // Cardinality analysis
                let counts: Vec<u32> = instance_obj_map.values().map(|v| v.len() as u32).collect();
                let min_count = counts.iter().copied().min();
                let max_count = counts.iter().copied().max();

                // Datatype inference: look at literal objects of instances that have this property
                let all_objects: Vec<&str> = instance_obj_map
                    .values()
                    .flat_map(|v| v.iter().copied())
                    .collect();

                let datatype = if self.config.infer_datatypes {
                    infer_dominant_datatype(&all_objects)
                } else {
                    None
                };

                let node_kind = if self.config.infer_node_kinds {
                    Some(infer_node_kind(&all_objects))
                } else {
                    None
                };

                // Confidence: fraction of instances that have a consistent cardinality
                // We define consistency as having exactly the modal count.
                let modal_count = modal_value(&counts);
                let consistent = counts.iter().filter(|&&c| c == modal_count).count();
                let confidence = consistent as f64 / instances_with_prop as f64;

                if confidence < self.config.min_confidence {
                    stats.constraints_discarded += 1;
                    continue;
                }

                property_constraints.push(PropertyConstraint {
                    predicate: predicate.to_string(),
                    min_count,
                    max_count,
                    datatype,
                    node_kind,
                    support: prop_support,
                    confidence,
                });
            }

            if property_constraints.is_empty() {
                continue;
            }

            // Overall shape support: fraction of instances that have ALL required properties
            let required_preds: Vec<&str> = property_constraints
                .iter()
                .map(|pc| pc.predicate.as_str())
                .collect();

            let conforming_instances = instances
                .iter()
                .filter(|&&inst| {
                    required_preds.iter().all(|pred| {
                        pred_instance_objects
                            .get(pred)
                            .map(|m| m.contains_key(inst))
                            .unwrap_or(false)
                    })
                })
                .count();

            let shape_support = conforming_instances as f64 / total_instances as f64;

            // Overall confidence: weighted average of property confidences
            let shape_confidence = if property_constraints.is_empty() {
                0.0
            } else {
                property_constraints
                    .iter()
                    .map(|pc| pc.confidence)
                    .sum::<f64>()
                    / property_constraints.len() as f64
            };

            property_constraints.sort_by(|a, b| {
                b.support
                    .partial_cmp(&a.support)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            shapes.push(MinedShape {
                target_class: class_uri.to_string(),
                properties: property_constraints,
                support: shape_support,
                confidence: shape_confidence,
                instance_count: total_instances,
            });
        }

        // Sort by support descending
        shapes.sort_by(|a, b| {
            b.support
                .partial_cmp(&a.support)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply max_shapes cap
        if self.config.max_shapes > 0 && shapes.len() > self.config.max_shapes {
            shapes.truncate(self.config.max_shapes);
        }

        stats.shapes_mined = shapes.len();

        ShapeMiningReport { shapes, stats }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Guess the dominant XSD datatype from a slice of object strings.
///
/// Typed literals typically look like `"value"^^<xsd:...>` or `"value"^^xsd:...`.
/// We parse the suffix after `^^` and count occurrences.
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

/// Infer the dominant `NodeKind` from a set of object strings.
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
        } else if obj.starts_with("http://") || obj.starts_with("https://") || obj.contains(':') {
            iri_count += 1;
        } else {
            literal_count += 1;
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

/// Return the modal (most-frequent) value from a non-empty slice.
fn modal_value(values: &[u32]) -> u32 {
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for &v in values {
        *freq.entry(v).or_insert(0) += 1;
    }
    freq.into_iter()
        .max_by_key(|(_, cnt)| *cnt)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rdf_type() -> String {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()
    }

    fn person_class() -> String {
        "http://ex.org/Person".to_string()
    }

    fn sample_person_triples() -> Vec<(String, String, String)> {
        vec![
            // Alice – Person
            ("http://ex.org/alice".into(), rdf_type(), person_class()),
            (
                "http://ex.org/alice".into(),
                "http://ex.org/name".into(),
                "\"Alice\"".into(),
            ),
            (
                "http://ex.org/alice".into(),
                "http://ex.org/age".into(),
                "\"30\"^^<http://www.w3.org/2001/XMLSchema#integer>".into(),
            ),
            // Bob – Person
            ("http://ex.org/bob".into(), rdf_type(), person_class()),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/name".into(),
                "\"Bob\"".into(),
            ),
            (
                "http://ex.org/bob".into(),
                "http://ex.org/age".into(),
                "\"25\"^^<http://www.w3.org/2001/XMLSchema#integer>".into(),
            ),
        ]
    }

    #[test]
    fn test_detect_classes_basic() {
        let miner = ShapeMiner::new(0.1, 0.5);
        let triples = sample_person_triples();
        let classes = miner.detect_classes(&triples);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], person_class());
    }

    #[test]
    fn test_detect_classes_empty_input() {
        let miner = ShapeMiner::new(0.1, 0.5);
        let classes = miner.detect_classes(&[]);
        assert!(classes.is_empty());
    }

    #[test]
    fn test_mine_shapes_basic() {
        let miner = ShapeMiner::new(0.1, 0.5);
        let triples = sample_person_triples();
        let shapes = miner.mine_shapes(&triples);
        assert_eq!(shapes.len(), 1);
        let shape = &shapes[0];
        assert_eq!(shape.target_class, person_class());
        assert_eq!(shape.instance_count, 2);
        assert!(shape.support > 0.0);
        // Both name and age should appear
        assert!(shape
            .properties
            .iter()
            .any(|p| p.predicate.ends_with("name")));
        assert!(shape
            .properties
            .iter()
            .any(|p| p.predicate.ends_with("age")));
    }

    #[test]
    fn test_mine_shapes_support_threshold() {
        // With support=1.0, only predicates present in ALL instances are included.
        let miner = ShapeMiner::new(1.0, 0.0);
        let mut triples = sample_person_triples();
        // Add a predicate that only Alice has
        triples.push((
            "http://ex.org/alice".into(),
            "http://ex.org/email".into(),
            "\"alice@example.com\"".into(),
        ));
        let shapes = miner.mine_shapes(&triples);
        // shape should exist but NOT contain the email predicate
        assert_eq!(shapes.len(), 1);
        let shape = &shapes[0];
        assert!(!shape
            .properties
            .iter()
            .any(|p| p.predicate.ends_with("email")));
    }

    #[test]
    fn test_mine_shapes_no_type_triples() {
        let miner = ShapeMiner::new(0.1, 0.5);
        let triples = vec![(
            "http://ex.org/x".into(),
            "http://ex.org/foo".into(),
            "\"bar\"".into(),
        )];
        let shapes = miner.mine_shapes(&triples);
        assert!(shapes.is_empty());
    }

    #[test]
    fn test_mine_shapes_max_shapes_cap() {
        let config = ShapeMinerConfig {
            min_support: 0.0,
            min_confidence: 0.0,
            max_shapes: 1,
            infer_datatypes: false,
            infer_node_kinds: false,
        };
        let miner = ShapeMiner::with_config(config);
        let mut triples = sample_person_triples();
        // Add a second class
        triples.push((
            "http://ex.org/org1".into(),
            rdf_type(),
            "http://ex.org/Org".into(),
        ));
        triples.push((
            "http://ex.org/org1".into(),
            "http://ex.org/orgName".into(),
            "\"Acme\"".into(),
        ));
        let shapes = miner.mine_shapes(&triples);
        assert_eq!(shapes.len(), 1);
    }

    #[test]
    fn test_node_kind_inference_iri() {
        let objects = vec!["http://ex.org/a", "http://ex.org/b", "http://ex.org/c"];
        assert_eq!(infer_node_kind(&objects), NodeKind::Iri);
    }

    #[test]
    fn test_node_kind_inference_literal() {
        let objects = vec!["\"hello\"", "\"world\"", "\"foo\""];
        assert_eq!(infer_node_kind(&objects), NodeKind::Literal);
    }

    #[test]
    fn test_infer_dominant_datatype() {
        let objects = vec![
            "\"1\"^^<http://www.w3.org/2001/XMLSchema#integer>",
            "\"2\"^^<http://www.w3.org/2001/XMLSchema#integer>",
            "\"3\"^^<http://www.w3.org/2001/XMLSchema#string>",
        ];
        let dt = infer_dominant_datatype(&objects);
        // The function strips surrounding angle brackets, returning the bare URI.
        assert_eq!(
            dt,
            Some("http://www.w3.org/2001/XMLSchema#integer".to_string())
        );
    }

    #[test]
    fn test_mining_report_stats() {
        let miner = ShapeMiner::new(0.1, 0.5);
        let triples = sample_person_triples();
        let report = miner.mine_shapes_with_report(&triples);
        assert_eq!(report.stats.triples_processed, triples.len());
        assert!(report.stats.classes_detected >= 1);
        assert!(report.stats.shapes_mined >= 1);
    }
}
