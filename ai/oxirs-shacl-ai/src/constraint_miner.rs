//! # Constraint Mining from RDF Data
//!
//! Analyzes RDF data to automatically discover SHACL-like constraints including
//! cardinality bounds, value types, string patterns, and value ranges.
//!
//! ## Features
//!
//! - **Cardinality mining**: Discover min/max cardinality for properties
//! - **Type inference**: Infer expected datatypes for property values
//! - **Pattern detection**: Discover regex patterns from string values
//! - **Range analysis**: Compute min/max ranges for numeric properties
//! - **Class shape generation**: Generate complete shapes for RDF classes
//! - **Confidence scoring**: Each mined constraint includes a confidence score

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────
// Mined constraint types
// ─────────────────────────────────────────────

/// A mined constraint discovered from data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinedConstraint {
    /// The property (predicate IRI) this constraint applies to.
    pub property: String,
    /// The kind of constraint.
    pub kind: ConstraintKind,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f64,
    /// Number of data points supporting this constraint.
    pub support: usize,
    /// Human-readable description.
    pub description: String,
}

/// Kinds of constraints that can be mined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Minimum cardinality (sh:minCount).
    MinCardinality(usize),
    /// Maximum cardinality (sh:maxCount).
    MaxCardinality(usize),
    /// Exact cardinality (sh:minCount = sh:maxCount).
    ExactCardinality(usize),
    /// Expected datatype (sh:datatype).
    Datatype(String),
    /// String pattern (sh:pattern).
    Pattern(String),
    /// Minimum inclusive value (sh:minInclusive).
    MinInclusive(f64),
    /// Maximum inclusive value (sh:maxInclusive).
    MaxInclusive(f64),
    /// Minimum string length (sh:minLength).
    MinLength(usize),
    /// Maximum string length (sh:maxLength).
    MaxLength(usize),
    /// Value must be one of these (sh:in).
    ValueIn(Vec<String>),
    /// Value must be an IRI (sh:nodeKind sh:IRI).
    NodeKindIri,
    /// Value must be a literal (sh:nodeKind sh:Literal).
    NodeKindLiteral,
    /// Value must be a blank node (sh:nodeKind sh:BlankNode).
    NodeKindBlankNode,
}

/// A shape (collection of constraints) for an RDF class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinedShape {
    /// The RDF class IRI this shape targets.
    pub target_class: String,
    /// Constraints discovered for this class.
    pub constraints: Vec<MinedConstraint>,
    /// Number of instances analyzed.
    pub instance_count: usize,
    /// Properties discovered.
    pub properties: Vec<String>,
}

/// Configuration for the constraint miner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerConfig {
    /// Minimum confidence to include a constraint (default: 0.8).
    pub min_confidence: f64,
    /// Maximum number of distinct values before giving up on sh:in (default: 20).
    pub max_in_values: usize,
    /// Whether to mine string patterns (can be slow) (default: true).
    pub mine_patterns: bool,
    /// Whether to mine numeric ranges (default: true).
    pub mine_ranges: bool,
    /// Whether to mine cardinality (default: true).
    pub mine_cardinality: bool,
    /// Whether to mine datatypes (default: true).
    pub mine_datatypes: bool,
    /// Whether to mine node kinds (default: true).
    pub mine_node_kinds: bool,
}

impl Default for MinerConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            max_in_values: 20,
            mine_patterns: true,
            mine_ranges: true,
            mine_cardinality: true,
            mine_datatypes: true,
            mine_node_kinds: true,
        }
    }
}

// ─────────────────────────────────────────────
// RDF data representation (simplified for mining)
// ─────────────────────────────────────────────

/// An RDF value that can be an IRI, literal, or blank node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RdfValue {
    Iri(String),
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
    BlankNode(String),
}

impl RdfValue {
    pub fn iri(s: impl Into<String>) -> Self {
        RdfValue::Iri(s.into())
    }

    pub fn literal(value: impl Into<String>) -> Self {
        RdfValue::Literal {
            value: value.into(),
            datatype: None,
            language: None,
        }
    }

    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        RdfValue::Literal {
            value: value.into(),
            datatype: Some(datatype.into()),
            language: None,
        }
    }

    pub fn is_iri(&self) -> bool {
        matches!(self, RdfValue::Iri(_))
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, RdfValue::Literal { .. })
    }

    pub fn is_blank_node(&self) -> bool {
        matches!(self, RdfValue::BlankNode(_))
    }

    /// Get the string representation of the value.
    pub fn value_str(&self) -> &str {
        match self {
            RdfValue::Iri(s) => s,
            RdfValue::Literal { value, .. } => value,
            RdfValue::BlankNode(s) => s,
        }
    }

    /// Get the datatype if this is a typed literal.
    pub fn datatype(&self) -> Option<&str> {
        match self {
            RdfValue::Literal {
                datatype: Some(dt), ..
            } => Some(dt),
            _ => None,
        }
    }
}

/// An RDF triple for mining purposes.
#[derive(Debug, Clone)]
pub struct MiningTriple {
    /// Subject IRI or blank node ID.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object value.
    pub object: RdfValue,
}

impl MiningTriple {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>, object: RdfValue) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object,
        }
    }
}

// ─────────────────────────────────────────────
// ConstraintMiner
// ─────────────────────────────────────────────

/// Mines SHACL constraints from RDF data.
pub struct ConstraintMiner {
    config: MinerConfig,
}

impl ConstraintMiner {
    /// Create a new miner with default configuration.
    pub fn new() -> Self {
        Self {
            config: MinerConfig::default(),
        }
    }

    /// Create a new miner with the given configuration.
    pub fn with_config(config: MinerConfig) -> Self {
        Self { config }
    }

    /// Mine constraints for a specific property from a set of triples.
    pub fn mine_property(&self, property: &str, triples: &[MiningTriple]) -> Vec<MinedConstraint> {
        let relevant: Vec<_> = triples.iter().filter(|t| t.predicate == property).collect();

        if relevant.is_empty() {
            return Vec::new();
        }

        let mut constraints = Vec::new();

        // Group by subject to analyze cardinality
        let mut by_subject: HashMap<&str, Vec<&RdfValue>> = HashMap::new();
        for t in &relevant {
            by_subject
                .entry(t.subject.as_str())
                .or_default()
                .push(&t.object);
        }

        // Mine cardinality
        if self.config.mine_cardinality {
            constraints.extend(self.mine_cardinality(property, &by_subject));
        }

        // Collect all objects
        let objects: Vec<&RdfValue> = relevant.iter().map(|t| &t.object).collect();

        // Mine datatypes
        if self.config.mine_datatypes {
            constraints.extend(self.mine_datatypes(property, &objects));
        }

        // Mine node kinds
        if self.config.mine_node_kinds {
            constraints.extend(self.mine_node_kinds(property, &objects));
        }

        // Mine ranges (numeric)
        if self.config.mine_ranges {
            constraints.extend(self.mine_numeric_range(property, &objects));
        }

        // Mine string patterns
        if self.config.mine_patterns {
            constraints.extend(self.mine_string_patterns(property, &objects));
        }

        // Mine value enumeration
        constraints.extend(self.mine_value_in(property, &objects));

        // Filter by confidence
        constraints
            .into_iter()
            .filter(|c| c.confidence >= self.config.min_confidence)
            .collect()
    }

    /// Mine constraints for all properties found in the triples.
    pub fn mine_all(&self, triples: &[MiningTriple]) -> Vec<MinedConstraint> {
        let properties: HashSet<&str> = triples.iter().map(|t| t.predicate.as_str()).collect();
        let mut all_constraints = Vec::new();
        for property in properties {
            all_constraints.extend(self.mine_property(property, triples));
        }
        all_constraints
    }

    /// Mine a complete shape for instances of a given class.
    pub fn mine_shape(
        &self,
        class_iri: &str,
        type_predicate: &str,
        triples: &[MiningTriple],
    ) -> MinedShape {
        // Find instances of the class
        let instances: HashSet<&str> = triples
            .iter()
            .filter(|t| {
                t.predicate == type_predicate
                    && matches!(&t.object, RdfValue::Iri(iri) if iri == class_iri)
            })
            .map(|t| t.subject.as_str())
            .collect();

        // Get all triples for those instances
        let instance_triples: Vec<_> = triples
            .iter()
            .filter(|t| instances.contains(t.subject.as_str()) && t.predicate != type_predicate)
            .cloned()
            .collect();

        let properties: Vec<String> = instance_triples
            .iter()
            .map(|t| t.predicate.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let constraints = self.mine_all(&instance_triples);

        MinedShape {
            target_class: class_iri.to_string(),
            constraints,
            instance_count: instances.len(),
            properties,
        }
    }

    // ─── Internal mining methods ─────────────────────────

    fn mine_cardinality(
        &self,
        property: &str,
        by_subject: &HashMap<&str, Vec<&RdfValue>>,
    ) -> Vec<MinedConstraint> {
        let mut constraints = Vec::new();
        if by_subject.is_empty() {
            return constraints;
        }

        let counts: Vec<usize> = by_subject.values().map(|v| v.len()).collect();
        let min_count = counts.iter().copied().min().unwrap_or(0);
        let max_count = counts.iter().copied().max().unwrap_or(0);
        let total = counts.len();

        // MinCardinality
        if min_count > 0 {
            let support = counts.iter().filter(|&&c| c >= min_count).count();
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::MinCardinality(min_count),
                confidence: support as f64 / total as f64,
                support,
                description: format!(
                    "Property '{}' has minimum cardinality {}",
                    property, min_count
                ),
            });
        }

        // MaxCardinality
        {
            let support = counts.iter().filter(|&&c| c <= max_count).count();
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::MaxCardinality(max_count),
                confidence: support as f64 / total as f64,
                support,
                description: format!(
                    "Property '{}' has maximum cardinality {}",
                    property, max_count
                ),
            });
        }

        // ExactCardinality
        if min_count == max_count {
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::ExactCardinality(min_count),
                confidence: 1.0,
                support: total,
                description: format!(
                    "Property '{}' has exact cardinality {}",
                    property, min_count
                ),
            });
        }

        constraints
    }

    fn mine_datatypes(&self, property: &str, objects: &[&RdfValue]) -> Vec<MinedConstraint> {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut literal_count = 0usize;

        for obj in objects {
            if let RdfValue::Literal { datatype, .. } = obj {
                literal_count += 1;
                let dt = datatype
                    .as_deref()
                    .unwrap_or("http://www.w3.org/2001/XMLSchema#string");
                *type_counts.entry(dt.to_string()).or_insert(0) += 1;
            }
        }

        if literal_count == 0 {
            return Vec::new();
        }

        let mut constraints = Vec::new();

        // Find the dominant datatype
        if let Some((dt, count)) = type_counts.iter().max_by_key(|(_, v)| **v) {
            let confidence = *count as f64 / literal_count as f64;
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::Datatype(dt.clone()),
                confidence,
                support: *count,
                description: format!("Property '{}' values are of type '{}'", property, dt),
            });
        }

        constraints
    }

    fn mine_node_kinds(&self, property: &str, objects: &[&RdfValue]) -> Vec<MinedConstraint> {
        let total = objects.len();
        if total == 0 {
            return Vec::new();
        }

        let iri_count = objects.iter().filter(|o| o.is_iri()).count();
        let literal_count = objects.iter().filter(|o| o.is_literal()).count();
        let bnode_count = objects.iter().filter(|o| o.is_blank_node()).count();

        let mut constraints = Vec::new();
        let total_f = total as f64;

        if iri_count as f64 / total_f >= self.config.min_confidence {
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::NodeKindIri,
                confidence: iri_count as f64 / total_f,
                support: iri_count,
                description: format!("Property '{}' values are IRIs", property),
            });
        }

        if literal_count as f64 / total_f >= self.config.min_confidence {
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::NodeKindLiteral,
                confidence: literal_count as f64 / total_f,
                support: literal_count,
                description: format!("Property '{}' values are literals", property),
            });
        }

        if bnode_count as f64 / total_f >= self.config.min_confidence {
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::NodeKindBlankNode,
                confidence: bnode_count as f64 / total_f,
                support: bnode_count,
                description: format!("Property '{}' values are blank nodes", property),
            });
        }

        constraints
    }

    fn mine_numeric_range(&self, property: &str, objects: &[&RdfValue]) -> Vec<MinedConstraint> {
        let values: Vec<f64> = objects
            .iter()
            .filter_map(|o| {
                if let RdfValue::Literal { value, .. } = o {
                    value.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .collect();

        if values.is_empty() {
            return Vec::new();
        }

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let confidence = values.len() as f64 / objects.len() as f64;

        let mut constraints = Vec::new();

        constraints.push(MinedConstraint {
            property: property.to_string(),
            kind: ConstraintKind::MinInclusive(min_val),
            confidence,
            support: values.len(),
            description: format!("Property '{}' minimum value is {}", property, min_val),
        });

        constraints.push(MinedConstraint {
            property: property.to_string(),
            kind: ConstraintKind::MaxInclusive(max_val),
            confidence,
            support: values.len(),
            description: format!("Property '{}' maximum value is {}", property, max_val),
        });

        constraints
    }

    fn mine_string_patterns(&self, property: &str, objects: &[&RdfValue]) -> Vec<MinedConstraint> {
        let string_values: Vec<&str> = objects
            .iter()
            .filter_map(|o| {
                if let RdfValue::Literal { value, .. } = o {
                    Some(value.as_str())
                } else {
                    None
                }
            })
            .collect();

        if string_values.is_empty() {
            return Vec::new();
        }

        let mut constraints = Vec::new();

        // Min/Max length
        let lengths: Vec<usize> = string_values.iter().map(|s| s.len()).collect();
        let min_len = lengths.iter().copied().min().unwrap_or(0);
        let max_len = lengths.iter().copied().max().unwrap_or(0);
        let confidence = string_values.len() as f64 / objects.len() as f64;

        constraints.push(MinedConstraint {
            property: property.to_string(),
            kind: ConstraintKind::MinLength(min_len),
            confidence,
            support: string_values.len(),
            description: format!(
                "Property '{}' minimum string length is {}",
                property, min_len
            ),
        });

        constraints.push(MinedConstraint {
            property: property.to_string(),
            kind: ConstraintKind::MaxLength(max_len),
            confidence,
            support: string_values.len(),
            description: format!(
                "Property '{}' maximum string length is {}",
                property, max_len
            ),
        });

        // Detect common patterns: all-digits, email-like, URI-like
        let all_digits = string_values
            .iter()
            .all(|s| s.chars().all(|c| c.is_ascii_digit()));
        if all_digits && !string_values.is_empty() {
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::Pattern("^[0-9]+$".to_string()),
                confidence: 1.0,
                support: string_values.len(),
                description: format!("Property '{}' values are all-digit strings", property),
            });
        }

        let all_email_like = string_values
            .iter()
            .all(|s| s.contains('@') && s.contains('.'));
        if all_email_like && string_values.len() > 1 {
            constraints.push(MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::Pattern("^.+@.+\\..+$".to_string()),
                confidence: 1.0,
                support: string_values.len(),
                description: format!(
                    "Property '{}' values appear to be email addresses",
                    property
                ),
            });
        }

        constraints
    }

    fn mine_value_in(&self, property: &str, objects: &[&RdfValue]) -> Vec<MinedConstraint> {
        let values: HashSet<String> = objects.iter().map(|o| o.value_str().to_string()).collect();

        if values.len() <= self.config.max_in_values
            && !values.is_empty()
            && values.len() <= objects.len() / 2
        {
            let confidence = 1.0; // All observed values are in this set
            let mut sorted_values: Vec<String> = values.into_iter().collect();
            sorted_values.sort();
            return vec![MinedConstraint {
                property: property.to_string(),
                kind: ConstraintKind::ValueIn(sorted_values.clone()),
                confidence,
                support: objects.len(),
                description: format!(
                    "Property '{}' values are one of: {:?}",
                    property, sorted_values
                ),
            }];
        }

        Vec::new()
    }
}

impl Default for ConstraintMiner {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
    const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    fn sample_triples() -> Vec<MiningTriple> {
        vec![
            MiningTriple::new(
                "ex:alice",
                "ex:name",
                RdfValue::typed_literal("Alice", XSD_STRING),
            ),
            MiningTriple::new(
                "ex:alice",
                "ex:age",
                RdfValue::typed_literal("30", XSD_INTEGER),
            ),
            MiningTriple::new("ex:alice", "ex:knows", RdfValue::iri("ex:bob")),
            MiningTriple::new(
                "ex:bob",
                "ex:name",
                RdfValue::typed_literal("Bob", XSD_STRING),
            ),
            MiningTriple::new(
                "ex:bob",
                "ex:age",
                RdfValue::typed_literal("25", XSD_INTEGER),
            ),
            MiningTriple::new("ex:bob", "ex:knows", RdfValue::iri("ex:alice")),
            MiningTriple::new("ex:alice", RDF_TYPE, RdfValue::iri("ex:Person")),
            MiningTriple::new("ex:bob", RDF_TYPE, RdfValue::iri("ex:Person")),
        ]
    }

    // ═══ Config tests ════════════════════════════════════

    #[test]
    fn test_default_config() {
        let config = MinerConfig::default();
        assert!((config.min_confidence - 0.8).abs() < 1e-10);
        assert_eq!(config.max_in_values, 20);
        assert!(config.mine_patterns);
        assert!(config.mine_cardinality);
    }

    #[test]
    fn test_custom_config() {
        let config = MinerConfig {
            min_confidence: 0.5,
            max_in_values: 10,
            ..Default::default()
        };
        assert!((config.min_confidence - 0.5).abs() < 1e-10);
    }

    // ═══ RdfValue tests ══════════════════════════════════

    #[test]
    fn test_rdf_value_iri() {
        let v = RdfValue::iri("http://example.org/foo");
        assert!(v.is_iri());
        assert!(!v.is_literal());
        assert!(!v.is_blank_node());
    }

    #[test]
    fn test_rdf_value_literal() {
        let v = RdfValue::literal("hello");
        assert!(v.is_literal());
        assert!(!v.is_iri());
        assert_eq!(v.value_str(), "hello");
    }

    #[test]
    fn test_rdf_value_typed_literal() {
        let v = RdfValue::typed_literal("42", XSD_INTEGER);
        assert!(v.is_literal());
        assert_eq!(v.datatype(), Some(XSD_INTEGER));
    }

    #[test]
    fn test_rdf_value_blank_node() {
        let v = RdfValue::BlankNode("b0".to_string());
        assert!(v.is_blank_node());
    }

    // ═══ Cardinality mining tests ════════════════════════

    #[test]
    fn test_mine_cardinality_exact() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        let exact = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::ExactCardinality(1)));
        assert!(exact.is_some());
    }

    #[test]
    fn test_mine_cardinality_min() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        let min_card = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::MinCardinality(_)));
        assert!(min_card.is_some());
    }

    #[test]
    fn test_mine_cardinality_max() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        let max_card = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::MaxCardinality(_)));
        assert!(max_card.is_some());
    }

    // ═══ Datatype mining tests ═══════════════════════════

    #[test]
    fn test_mine_datatype_integer() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:age", &triples);
        let dt = constraints
            .iter()
            .find(|c| matches!(&c.kind, ConstraintKind::Datatype(d) if d == XSD_INTEGER));
        assert!(dt.is_some());
    }

    #[test]
    fn test_mine_datatype_string() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        let dt = constraints
            .iter()
            .find(|c| matches!(&c.kind, ConstraintKind::Datatype(d) if d == XSD_STRING));
        assert!(dt.is_some());
    }

    // ═══ Node kind mining tests ══════════════════════════

    #[test]
    fn test_mine_node_kind_iri() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:knows", &triples);
        let nk = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::NodeKindIri));
        assert!(nk.is_some());
    }

    #[test]
    fn test_mine_node_kind_literal() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        let nk = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::NodeKindLiteral));
        assert!(nk.is_some());
    }

    // ═══ Numeric range mining tests ══════════════════════

    #[test]
    fn test_mine_numeric_range() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:age", &triples);
        let min_inc = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::MinInclusive(v) if (v - 25.0).abs() < 1e-6));
        let max_inc = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::MaxInclusive(v) if (v - 30.0).abs() < 1e-6));
        assert!(min_inc.is_some());
        assert!(max_inc.is_some());
    }

    // ═══ String pattern mining tests ═════════════════════

    #[test]
    fn test_mine_digit_pattern() {
        let miner = ConstraintMiner::new();
        let triples = vec![
            MiningTriple::new("s1", "ex:code", RdfValue::literal("12345")),
            MiningTriple::new("s2", "ex:code", RdfValue::literal("67890")),
            MiningTriple::new("s3", "ex:code", RdfValue::literal("11111")),
        ];
        let constraints = miner.mine_property("ex:code", &triples);
        let pat = constraints
            .iter()
            .find(|c| matches!(&c.kind, ConstraintKind::Pattern(p) if p.contains("[0-9]")));
        assert!(pat.is_some());
    }

    #[test]
    fn test_mine_email_pattern() {
        let miner = ConstraintMiner::new();
        let triples = vec![
            MiningTriple::new("s1", "ex:email", RdfValue::literal("alice@example.com")),
            MiningTriple::new("s2", "ex:email", RdfValue::literal("bob@example.org")),
        ];
        let constraints = miner.mine_property("ex:email", &triples);
        let pat = constraints
            .iter()
            .find(|c| matches!(&c.kind, ConstraintKind::Pattern(_)));
        assert!(pat.is_some());
    }

    #[test]
    fn test_mine_string_length() {
        let miner = ConstraintMiner::new();
        let triples = vec![
            MiningTriple::new("s1", "ex:name", RdfValue::literal("Al")),
            MiningTriple::new("s2", "ex:name", RdfValue::literal("Alexander")),
        ];
        let constraints = miner.mine_property("ex:name", &triples);
        let min_len = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::MinLength(2)));
        let max_len = constraints
            .iter()
            .find(|c| matches!(c.kind, ConstraintKind::MaxLength(9)));
        assert!(min_len.is_some());
        assert!(max_len.is_some());
    }

    // ═══ Value enumeration tests ═════════════════════════

    #[test]
    fn test_mine_value_in() {
        let miner = ConstraintMiner::new();
        let triples = vec![
            MiningTriple::new("s1", "ex:status", RdfValue::literal("active")),
            MiningTriple::new("s2", "ex:status", RdfValue::literal("active")),
            MiningTriple::new("s3", "ex:status", RdfValue::literal("inactive")),
            MiningTriple::new("s4", "ex:status", RdfValue::literal("active")),
        ];
        let constraints = miner.mine_property("ex:status", &triples);
        let val_in = constraints
            .iter()
            .find(|c| matches!(&c.kind, ConstraintKind::ValueIn(_)));
        assert!(val_in.is_some());
    }

    // ═══ Mine all tests ══════════════════════════════════

    #[test]
    fn test_mine_all() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_all(&triples);
        assert!(!constraints.is_empty());
        // Should have constraints for multiple properties
        let properties: HashSet<&str> = constraints.iter().map(|c| c.property.as_str()).collect();
        assert!(properties.len() >= 2);
    }

    // ═══ Mine shape tests ════════════════════════════════

    #[test]
    fn test_mine_shape() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let shape = miner.mine_shape("ex:Person", RDF_TYPE, &triples);
        assert_eq!(shape.target_class, "ex:Person");
        assert_eq!(shape.instance_count, 2);
        assert!(!shape.constraints.is_empty());
        assert!(!shape.properties.is_empty());
    }

    #[test]
    fn test_mine_shape_no_instances() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let shape = miner.mine_shape("ex:Unknown", RDF_TYPE, &triples);
        assert_eq!(shape.instance_count, 0);
        assert!(shape.constraints.is_empty());
    }

    // ═══ Confidence filtering tests ══════════════════════

    #[test]
    fn test_confidence_filtering() {
        let config = MinerConfig {
            min_confidence: 0.99,
            ..Default::default()
        };
        let miner = ConstraintMiner::with_config(config);
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        // All returned constraints should have confidence >= 0.99
        assert!(constraints.iter().all(|c| c.confidence >= 0.99));
    }

    // ═══ Empty input tests ═══════════════════════════════

    #[test]
    fn test_mine_property_empty() {
        let miner = ConstraintMiner::new();
        let constraints = miner.mine_property("ex:foo", &[]);
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_mine_all_empty() {
        let miner = ConstraintMiner::new();
        let constraints = miner.mine_all(&[]);
        assert!(constraints.is_empty());
    }

    // ═══ Default impl test ═══════════════════════════════

    #[test]
    fn test_default_impl() {
        let miner = ConstraintMiner::default();
        let constraints = miner.mine_all(&[]);
        assert!(constraints.is_empty());
    }

    // ═══ Constraint description tests ════════════════════

    #[test]
    fn test_constraint_description() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:age", &triples);
        assert!(constraints.iter().all(|c| !c.description.is_empty()));
    }

    // ═══ Support count tests ═════════════════════════════

    #[test]
    fn test_support_count() {
        let miner = ConstraintMiner::new();
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:age", &triples);
        assert!(constraints.iter().all(|c| c.support > 0));
    }

    // ═══ Selective mining config tests ═══════════════════

    #[test]
    fn test_no_cardinality_mining() {
        let config = MinerConfig {
            mine_cardinality: false,
            min_confidence: 0.0,
            ..Default::default()
        };
        let miner = ConstraintMiner::with_config(config);
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        let has_cardinality = constraints.iter().any(|c| {
            matches!(
                c.kind,
                ConstraintKind::MinCardinality(_)
                    | ConstraintKind::MaxCardinality(_)
                    | ConstraintKind::ExactCardinality(_)
            )
        });
        assert!(!has_cardinality);
    }

    #[test]
    fn test_no_datatype_mining() {
        let config = MinerConfig {
            mine_datatypes: false,
            min_confidence: 0.0,
            ..Default::default()
        };
        let miner = ConstraintMiner::with_config(config);
        let triples = sample_triples();
        let constraints = miner.mine_property("ex:name", &triples);
        let has_datatype = constraints
            .iter()
            .any(|c| matches!(c.kind, ConstraintKind::Datatype(_)));
        assert!(!has_datatype);
    }
}
