//! LLM-based SHACL shape suggestion and violation explanation.
//!
//! This module provides:
//!
//! - [`ShapeSuggestionPrompt`] — builds structured prompts for LLM APIs.
//! - [`LlmShapeSuggester`] — mock/stub suggester that detects common RDF
//!   patterns and proposes SHACL constraints without a live LLM call.
//! - [`ShapeExplainer`] — generates natural-language explanations for why a
//!   SHACL constraint was violated.
//!
//! The design intentionally matches the interface of [`crate::llm`] but adds a
//! higher-level "suggestion" layer that operates directly on `GpuTriple`-style
//! compact triple representations and `ShapeRef` objects.
//!
//! ## Mock LLM policy
//!
//! `LlmShapeSuggester` uses deterministic pattern detection rather than a live
//! LLM API.  This keeps tests fully reproducible and removes network
//! dependencies.  Real implementations should replace `suggest_shapes` with an
//! async call to the provider of their choice.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::gpu::{GpuShapeRef, GpuTriple, ShapeViolation};

// ---------------------------------------------------------------------------
// LlmShapeProposal
// ---------------------------------------------------------------------------

/// A SHACL constraint proposal produced by (or simulated from) an LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmShapeProposal {
    /// Turtle/SPARQL snippets expressing the proposed SHACL constraints.
    pub proposed_constraints: Vec<String>,
    /// Model confidence in [0, 1].
    pub confidence: f64,
    /// Human-readable reasoning explaining why these constraints were proposed.
    pub reasoning: String,
}

impl LlmShapeProposal {
    /// Create a new proposal.
    pub fn new(
        proposed_constraints: Vec<String>,
        confidence: f64,
        reasoning: impl Into<String>,
    ) -> Self {
        let confidence = confidence.clamp(0.0, 1.0);
        Self {
            proposed_constraints,
            confidence,
            reasoning: reasoning.into(),
        }
    }

    /// Returns `true` if this proposal has high confidence (≥ 0.7).
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.7
    }

    /// Returns `true` if the proposal contains at least one constraint.
    pub fn has_constraints(&self) -> bool {
        !self.proposed_constraints.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ShapeSuggestionPrompt
// ---------------------------------------------------------------------------

/// Generates structured prompts for LLM-based SHACL shape suggestion.
///
/// Three prompt templates are provided:
///
/// | Template | Purpose |
/// |----------|---------|
/// | Analysis | Asks the LLM to analyse the graph structure |
/// | Constraint | Asks the LLM to propose specific SHACL constraints |
/// | Explanation | Asks the LLM to explain an existing shape or violation |
#[derive(Debug, Clone)]
pub struct ShapeSuggestionPrompt {
    /// System context injected at the beginning of every prompt.
    system_context: String,
    /// Maximum number of triples to include in the prompt (to stay within
    /// context window limits).
    max_triples_in_prompt: usize,
}

impl Default for ShapeSuggestionPrompt {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeSuggestionPrompt {
    /// Create a prompt generator with sensible defaults.
    pub fn new() -> Self {
        Self {
            system_context: "You are an expert in RDF, SHACL, and knowledge graph design.\
                             Your task is to analyse RDF triples and propose SHACL shape \
                             constraints that capture the implicit schema of the data."
                .to_owned(),
            max_triples_in_prompt: 50,
        }
    }

    /// Override the system context.
    pub fn with_system_context(mut self, ctx: impl Into<String>) -> Self {
        self.system_context = ctx.into();
        self
    }

    /// Override the maximum number of triples included in prompts.
    pub fn with_max_triples(mut self, n: usize) -> Self {
        self.max_triples_in_prompt = n;
        self
    }

    /// Generate the **analysis** prompt.
    ///
    /// Serialises a sample of `triples` and the IDs of any `existing_shapes`
    /// into a human-readable prompt that asks the LLM to identify structural
    /// patterns.
    pub fn generate_prompt(
        &self,
        graph_sample: &[GpuTriple],
        existing_shapes: &[GpuShapeRef],
    ) -> String {
        let triple_lines = self.format_triples(graph_sample);
        let shape_ids: Vec<&str> = existing_shapes
            .iter()
            .map(|s| s.shape_id.as_str())
            .collect();

        format!(
            "{}\n\n\
             ## Task: Analyse RDF Graph\n\n\
             ### Triples (compact form: subject_id predicate_id object_hash)\n\
             {}\n\n\
             ### Existing shapes\n\
             {}\n\n\
             Please identify implicit schema patterns in the triples above.",
            self.system_context,
            triple_lines,
            if shape_ids.is_empty() {
                "(none)".to_owned()
            } else {
                shape_ids.join(", ")
            }
        )
    }

    /// Generate the **constraint** prompt.
    ///
    /// Focuses the LLM on producing specific SHACL constraints in Turtle
    /// format.
    pub fn generate_constraint_prompt(
        &self,
        graph_sample: &[GpuTriple],
        existing_shapes: &[GpuShapeRef],
    ) -> String {
        let triple_lines = self.format_triples(graph_sample);
        let shape_ids: Vec<&str> = existing_shapes
            .iter()
            .map(|s| s.shape_id.as_str())
            .collect();

        format!(
            "{}\n\n\
             ## Task: Propose SHACL Constraints\n\n\
             ### Triples\n\
             {}\n\n\
             ### Existing shapes\n\
             {}\n\n\
             For each pattern you identify, write a SHACL NodeShape or \
             PropertyShape in Turtle syntax.  Include sh:minCount, sh:maxCount, \
             sh:datatype, and sh:nodeKind constraints where appropriate.",
            self.system_context,
            triple_lines,
            if shape_ids.is_empty() {
                "(none)".to_owned()
            } else {
                shape_ids.join(", ")
            }
        )
    }

    /// Generate the **explanation** prompt.
    ///
    /// Asks the LLM to explain a specific violation in natural language.
    pub fn generate_explanation_prompt(
        &self,
        violation: &ShapeViolation,
        shape: &GpuShapeRef,
    ) -> String {
        format!(
            "{}\n\n\
             ## Task: Explain SHACL Violation\n\n\
             ### Shape: {}\n\
             Constraint type byte: {}\n\
             Max count: {:?}\n\
             Min count: {:?}\n\
             Datatype hash: {:?}\n\n\
             ### Violation\n\
             Focus node id: {}\n\
             Message: {}\n\
             Severity: {}\n\n\
             Please explain in plain English why this violation occurred and \
             suggest how the data or the shape should be corrected.",
            self.system_context,
            shape.shape_id,
            shape.constraint_type,
            shape.max_count,
            shape.min_count,
            shape.datatype_hash,
            violation.focus_node_id,
            violation.message,
            violation.severity,
        )
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    fn format_triples(&self, triples: &[GpuTriple]) -> String {
        let slice = if triples.len() > self.max_triples_in_prompt {
            &triples[..self.max_triples_in_prompt]
        } else {
            triples
        };
        if slice.is_empty() {
            return "(empty)".to_owned();
        }
        slice
            .iter()
            .map(|t| format!("  {} {} {}", t.subject_id, t.predicate_id, t.object_hash))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---------------------------------------------------------------------------
// LlmShapeSuggester
// ---------------------------------------------------------------------------

/// Mock LLM interface that suggests SHACL shapes from RDF triple patterns.
///
/// Instead of making a live LLM API call this implementation detects common
/// structural patterns in the `GpuTriple` slice and returns canned but
/// semantically meaningful proposals.
///
/// ## Detected patterns
///
/// | Pattern | Proposed constraint |
/// |---------|---------------------|
/// | High property cardinality (≥ 3 triples share a predicate) | `sh:maxCount` |
/// | Uniform object hashes across subjects | `sh:datatype xsd:string` (simulated) |
/// | Single-valued properties (max 1 per subject) | `sh:maxCount 1` |
/// | Required properties (every subject has the predicate) | `sh:minCount 1` |
pub struct LlmShapeSuggester {
    /// Threshold above which a property is considered "high cardinality".
    high_cardinality_threshold: u32,
    /// Confidence multiplier applied per detected pattern.
    confidence_per_pattern: f64,
}

impl Default for LlmShapeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmShapeSuggester {
    /// Create a suggester with default thresholds.
    pub fn new() -> Self {
        Self {
            high_cardinality_threshold: 3,
            confidence_per_pattern: 0.25,
        }
    }

    /// Override the high-cardinality threshold.
    pub fn with_cardinality_threshold(mut self, t: u32) -> Self {
        self.high_cardinality_threshold = t;
        self
    }

    /// Suggest SHACL shapes for the given triple sample.
    ///
    /// Returns one [`LlmShapeProposal`] per detected pattern (possibly zero).
    pub fn suggest_shapes(&self, sample: &[GpuTriple]) -> Vec<LlmShapeProposal> {
        if sample.is_empty() {
            return Vec::new();
        }

        let mut proposals: Vec<LlmShapeProposal> = Vec::new();

        // --- pattern 1: cardinality per (subject, predicate) pair ----------
        let mut pair_counts: HashMap<(u64, u64), u32> = HashMap::new();
        for t in sample {
            *pair_counts
                .entry((t.subject_id, t.predicate_id))
                .or_insert(0) += 1;
        }

        let max_count = pair_counts.values().copied().max().unwrap_or(0);
        if max_count >= self.high_cardinality_threshold {
            proposals.push(LlmShapeProposal::new(
                vec![format!(
                    "sh:property [ sh:maxCount {} ] .",
                    max_count
                )],
                (self.confidence_per_pattern * 3.0).clamp(0.0, 1.0),
                format!(
                    "Detected high-cardinality property usage (max {max_count} values per subject). \
                     Proposing sh:maxCount to enforce an upper bound."
                ),
            ));
        }

        // --- pattern 2: uniform object hashes → string datatype hint -------
        let object_hashes: Vec<u64> = sample.iter().map(|t| t.object_hash).collect();
        let unique_hashes: std::collections::HashSet<u64> = object_hashes.iter().copied().collect();
        if unique_hashes.len() == 1 {
            proposals.push(LlmShapeProposal::new(
                vec!["sh:property [ sh:datatype xsd:string ] .".to_owned()],
                (self.confidence_per_pattern * 2.5).clamp(0.0, 1.0),
                "All observed object hashes are identical, suggesting a uniform datatype. \
                 Proposing sh:datatype xsd:string as a candidate."
                    .to_owned(),
            ));
        }

        // --- pattern 3: single-valued properties (max_count == 1) ----------
        let single_valued = pair_counts.values().all(|&c| c == 1);
        if single_valued && !pair_counts.is_empty() {
            proposals.push(LlmShapeProposal::new(
                vec!["sh:property [ sh:maxCount 1 ] .".to_owned()],
                (self.confidence_per_pattern * 2.0).clamp(0.0, 1.0),
                "Every (subject, predicate) pair appears exactly once, \
                 indicating a single-valued property.  Proposing sh:maxCount 1."
                    .to_owned(),
            ));
        }

        // --- pattern 4: required property (every distinct subject has it) --
        let distinct_subjects: std::collections::HashSet<u64> =
            sample.iter().map(|t| t.subject_id).collect();
        let distinct_predicates: std::collections::HashSet<u64> =
            sample.iter().map(|t| t.predicate_id).collect();

        // A predicate is "required" if every subject has at least one triple with it.
        for pred in &distinct_predicates {
            let subjects_with_pred: std::collections::HashSet<u64> = sample
                .iter()
                .filter(|t| t.predicate_id == *pred)
                .map(|t| t.subject_id)
                .collect();
            if subjects_with_pred == distinct_subjects {
                proposals.push(LlmShapeProposal::new(
                    vec![format!(
                        "sh:property [ sh:path <urn:predicate:{pred}> ; sh:minCount 1 ] ."
                    )],
                    (self.confidence_per_pattern * 3.5).clamp(0.0, 1.0),
                    format!(
                        "Predicate {pred} appears for every subject in the sample. \
                         Proposing sh:minCount 1 to capture this as a required property."
                    ),
                ));
            }
        }

        proposals
    }
}

// ---------------------------------------------------------------------------
// ShapeExplainer
// ---------------------------------------------------------------------------

/// Explains SHACL constraint violations in natural language.
///
/// Given a [`ShapeViolation`] and the [`GpuShapeRef`] that triggered it,
/// `ShapeExplainer` produces a human-readable explanation string that:
///
/// 1. Identifies the type of constraint that was violated.
/// 2. States what the data contained vs. what the shape expected.
/// 3. Suggests how to fix either the data or the shape.
pub struct ShapeExplainer;

impl ShapeExplainer {
    /// Create a new explainer.
    pub fn new() -> Self {
        Self
    }

    /// Generate a natural-language explanation for `violation` given the
    /// originating `shape`.
    pub fn explain_violation(&self, violation: &ShapeViolation, shape: &GpuShapeRef) -> String {
        let severity_label = match violation.severity {
            0 => "informational notice",
            1 => "warning",
            _ => "violation",
        };

        let constraint_description = self.describe_constraint(shape);
        let fix_suggestion = self.suggest_fix(violation, shape);

        format!(
            "SHACL {severity_label} on shape '{}': \
             Focus node (id={}) failed the {} constraint. \
             Details: {}. \
             Suggestion: {}",
            shape.shape_id,
            violation.focus_node_id,
            constraint_description,
            violation.message,
            fix_suggestion,
        )
    }

    /// Summarise what type of constraint a shape encodes.
    pub fn describe_constraint(&self, shape: &GpuShapeRef) -> String {
        let mut parts: Vec<String> = Vec::new();

        if let Some(dt) = shape.datatype_hash {
            parts.push(format!("sh:datatype (hash={dt})"));
        }
        if let Some(max) = shape.max_count {
            parts.push(format!("sh:maxCount {max}"));
        }
        if let Some(min) = shape.min_count {
            parts.push(format!("sh:minCount {min}"));
        }
        if parts.is_empty() {
            match shape.constraint_type {
                1 => parts.push("sh:nodeKind sh:IRI".to_owned()),
                2 => parts.push("sh:nodeKind sh:BlankNode".to_owned()),
                3 => parts.push("sh:nodeKind sh:Literal".to_owned()),
                _ => parts.push("generic SHACL constraint".to_owned()),
            }
        }

        parts.join(" + ")
    }

    /// Generate a short fix suggestion based on the violation.
    fn suggest_fix(&self, violation: &ShapeViolation, shape: &GpuShapeRef) -> String {
        if violation.message.contains("Datatype mismatch") {
            if let Some(expected) = shape.datatype_hash {
                return format!(
                    "Ensure that the object value for focus node {} has \
                     datatype hash {expected}.  Alternatively, relax the \
                     sh:datatype constraint in the shape if the data is correct.",
                    violation.focus_node_id
                );
            }
        }
        if violation.message.contains("maxCount") {
            if let Some(max) = shape.max_count {
                return format!(
                    "Remove extra property values for focus node {} so that \
                     at most {max} value(s) remain.  Alternatively, increase \
                     sh:maxCount in the shape.",
                    violation.focus_node_id
                );
            }
        }
        if violation.message.contains("minCount") {
            if let Some(min) = shape.min_count {
                return format!(
                    "Add at least {min} value(s) for focus node {}.  \
                     Alternatively, reduce or remove sh:minCount if the \
                     property is truly optional.",
                    violation.focus_node_id
                );
            }
        }
        format!(
            "Review the data for focus node {} against shape '{}' and \
             correct the non-conforming value.",
            violation.focus_node_id, shape.shape_id
        )
    }
}

impl Default for ShapeExplainer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuShapeRef, GpuTriple, ShapeViolation};

    // -----------------------------------------------------------------------
    // LlmShapeProposal tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proposal_high_confidence() {
        let p = LlmShapeProposal::new(vec!["sh:maxCount 1 .".into()], 0.9, "reason");
        assert!(p.is_high_confidence());
        assert!(p.has_constraints());
    }

    #[test]
    fn test_proposal_low_confidence() {
        let p = LlmShapeProposal::new(vec![], 0.3, "reason");
        assert!(!p.is_high_confidence());
        assert!(!p.has_constraints());
    }

    #[test]
    fn test_proposal_confidence_clamped() {
        let p = LlmShapeProposal::new(vec![], 2.0, "r");
        assert!((p.confidence - 1.0).abs() < 1e-9);
        let p2 = LlmShapeProposal::new(vec![], -1.0, "r");
        assert!((p2.confidence - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_proposal_serialization() {
        let p = LlmShapeProposal::new(
            vec!["sh:minCount 1 .".into()],
            0.75,
            "Required property detected",
        );
        let json = serde_json::to_string(&p).expect("serialize ok");
        let back: LlmShapeProposal = serde_json::from_str(&json).expect("deserialize ok");
        assert_eq!(back.proposed_constraints, p.proposed_constraints);
        assert!((back.confidence - 0.75).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // ShapeSuggestionPrompt tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prompt_contains_system_context() {
        let gen = ShapeSuggestionPrompt::new();
        let prompt = gen.generate_prompt(&[], &[]);
        assert!(prompt.contains("SHACL"));
    }

    #[test]
    fn test_prompt_empty_triples_shows_empty() {
        let gen = ShapeSuggestionPrompt::new();
        let prompt = gen.generate_prompt(&[], &[]);
        assert!(prompt.contains("(empty)"));
    }

    #[test]
    fn test_prompt_includes_triple_ids() {
        let gen = ShapeSuggestionPrompt::new();
        let triples = vec![GpuTriple::new(1, 2, 3)];
        let prompt = gen.generate_prompt(&triples, &[]);
        assert!(prompt.contains("1 2 3"));
    }

    #[test]
    fn test_prompt_includes_shape_ids() {
        let gen = ShapeSuggestionPrompt::new();
        let shapes = vec![GpuShapeRef::new("http://example.org/PersonShape")];
        let prompt = gen.generate_prompt(&[], &shapes);
        assert!(prompt.contains("http://example.org/PersonShape"));
    }

    #[test]
    fn test_constraint_prompt_mentions_shacl() {
        let gen = ShapeSuggestionPrompt::new();
        let prompt = gen.generate_constraint_prompt(&[], &[]);
        assert!(prompt.contains("SHACL"));
        assert!(prompt.contains("sh:minCount") || prompt.contains("sh:maxCount"));
    }

    #[test]
    fn test_explanation_prompt_contains_violation_info() {
        let gen = ShapeSuggestionPrompt::new();
        let violation = ShapeViolation::new(42, "MyShape", "Datatype mismatch", 2);
        let shape = GpuShapeRef::new("MyShape").with_datatype_hash(100);
        let prompt = gen.generate_explanation_prompt(&violation, &shape);
        assert!(prompt.contains("42"));
        assert!(prompt.contains("Datatype mismatch"));
        assert!(prompt.contains("MyShape"));
    }

    #[test]
    fn test_prompt_max_triples_truncated() {
        let gen = ShapeSuggestionPrompt::new().with_max_triples(2);
        let triples = (0..10u64)
            .map(|i| GpuTriple::new(i, i, i))
            .collect::<Vec<_>>();
        let prompt = gen.generate_prompt(&triples, &[]);
        // Only first 2 triples should appear → "9 9 9" should not be in prompt
        assert!(!prompt.contains("9 9 9"));
        assert!(prompt.contains("0 0 0"));
    }

    #[test]
    fn test_custom_system_context() {
        let gen = ShapeSuggestionPrompt::new().with_system_context("Custom context here.");
        let prompt = gen.generate_prompt(&[], &[]);
        assert!(prompt.contains("Custom context here."));
    }

    // -----------------------------------------------------------------------
    // LlmShapeSuggester tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_suggest_empty_returns_nothing() {
        let s = LlmShapeSuggester::new();
        assert!(s.suggest_shapes(&[]).is_empty());
    }

    #[test]
    fn test_suggest_high_cardinality_detected() {
        let s = LlmShapeSuggester::new().with_cardinality_threshold(2);
        // Subject 1, predicate 2 appears 3 times
        let triples = vec![
            GpuTriple::new(1, 2, 10),
            GpuTriple::new(1, 2, 11),
            GpuTriple::new(1, 2, 12),
        ];
        let proposals = s.suggest_shapes(&triples);
        let has_max_count = proposals.iter().any(|p| {
            p.proposed_constraints
                .iter()
                .any(|c| c.contains("maxCount"))
        });
        assert!(has_max_count, "expected a maxCount proposal");
    }

    #[test]
    fn test_suggest_single_valued_detected() {
        let s = LlmShapeSuggester::new();
        // Each (subject, predicate) pair appears once
        let triples = vec![
            GpuTriple::new(1, 10, 100),
            GpuTriple::new(2, 10, 200),
            GpuTriple::new(3, 10, 300),
        ];
        let proposals = s.suggest_shapes(&triples);
        let has_max1 = proposals.iter().any(|p| {
            p.proposed_constraints
                .iter()
                .any(|c| c.contains("maxCount 1"))
        });
        assert!(has_max1, "expected a maxCount 1 proposal");
    }

    #[test]
    fn test_suggest_required_property_detected() {
        let s = LlmShapeSuggester::new();
        // Predicate 10 present for every subject (1, 2, 3)
        let triples = vec![
            GpuTriple::new(1, 10, 100),
            GpuTriple::new(2, 10, 200),
            GpuTriple::new(3, 10, 300),
        ];
        let proposals = s.suggest_shapes(&triples);
        let has_min1 = proposals.iter().any(|p| {
            p.proposed_constraints
                .iter()
                .any(|c| c.contains("minCount 1"))
        });
        assert!(has_min1, "expected a minCount 1 proposal");
    }

    #[test]
    fn test_suggest_uniform_objects_detected() {
        let s = LlmShapeSuggester::new();
        // All triples have the same object hash
        let triples = vec![
            GpuTriple::new(1, 5, 777),
            GpuTriple::new(2, 5, 777),
            GpuTriple::new(3, 5, 777),
        ];
        let proposals = s.suggest_shapes(&triples);
        let has_datatype = proposals.iter().any(|p| {
            p.proposed_constraints
                .iter()
                .any(|c| c.contains("datatype"))
        });
        assert!(has_datatype, "expected a datatype proposal");
    }

    #[test]
    fn test_suggest_proposals_have_reasoning() {
        let s = LlmShapeSuggester::new().with_cardinality_threshold(2);
        let triples = vec![
            GpuTriple::new(1, 2, 10),
            GpuTriple::new(1, 2, 11),
            GpuTriple::new(1, 2, 12),
        ];
        let proposals = s.suggest_shapes(&triples);
        for p in &proposals {
            assert!(!p.reasoning.is_empty(), "proposal should have reasoning");
        }
    }

    // -----------------------------------------------------------------------
    // ShapeExplainer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_explain_datatype_violation() {
        let explainer = ShapeExplainer::new();
        let violation =
            ShapeViolation::new(1, "S", "Datatype mismatch: expected hash 42, got 99", 2);
        let shape = GpuShapeRef::new("S").with_datatype_hash(42);
        let explanation = explainer.explain_violation(&violation, &shape);
        assert!(
            explanation.contains("violation"),
            "should mention violation"
        );
        assert!(explanation.contains("42"), "should mention expected hash");
        assert!(explanation.contains("sh:datatype"), "constraint type shown");
    }

    #[test]
    fn test_explain_max_count_violation() {
        let explainer = ShapeExplainer::new();
        let violation = ShapeViolation::new(
            7,
            "MaxShape",
            "sh:maxCount violated: count 5 exceeds max 2",
            2,
        );
        let shape = GpuShapeRef::new("MaxShape").with_cardinality(0, 2);
        let explanation = explainer.explain_violation(&violation, &shape);
        assert!(explanation.contains("maxCount"));
        assert!(explanation.contains("7"));
    }

    #[test]
    fn test_explain_min_count_violation() {
        let explainer = ShapeExplainer::new();
        let violation =
            ShapeViolation::new(3, "MinShape", "sh:minCount violated: count 0 < min 1", 2);
        let shape = GpuShapeRef::new("MinShape").with_cardinality(1, 10);
        let explanation = explainer.explain_violation(&violation, &shape);
        assert!(explanation.contains("minCount"));
        assert!(explanation.contains("3"));
    }

    #[test]
    fn test_explain_info_severity_label() {
        let explainer = ShapeExplainer::new();
        let violation = ShapeViolation::new(1, "S", "advisory", 0);
        let shape = GpuShapeRef::new("S");
        let explanation = explainer.explain_violation(&violation, &shape);
        assert!(explanation.contains("informational notice"));
    }

    #[test]
    fn test_explain_warning_severity_label() {
        let explainer = ShapeExplainer::new();
        let violation = ShapeViolation::new(1, "S", "advisory", 1);
        let shape = GpuShapeRef::new("S");
        let explanation = explainer.explain_violation(&violation, &shape);
        assert!(explanation.contains("warning"));
    }

    #[test]
    fn test_describe_constraint_datatype_only() {
        let explainer = ShapeExplainer::new();
        let shape = GpuShapeRef::new("S").with_datatype_hash(123);
        let desc = explainer.describe_constraint(&shape);
        assert!(desc.contains("sh:datatype"));
        assert!(desc.contains("123"));
    }

    #[test]
    fn test_describe_constraint_cardinality() {
        let explainer = ShapeExplainer::new();
        let shape = GpuShapeRef::new("S").with_cardinality(2, 5);
        let desc = explainer.describe_constraint(&shape);
        assert!(desc.contains("sh:minCount 2"));
        assert!(desc.contains("sh:maxCount 5"));
    }

    #[test]
    fn test_describe_constraint_node_kind() {
        let explainer = ShapeExplainer::new();
        let shape = GpuShapeRef::new("S").with_constraint_type(1);
        let desc = explainer.describe_constraint(&shape);
        assert!(desc.contains("sh:nodeKind sh:IRI"));
    }

    #[test]
    fn test_explain_generic_violation() {
        let explainer = ShapeExplainer::new();
        let violation = ShapeViolation::new(99, "AnyShape", "some constraint failed", 2);
        let shape = GpuShapeRef::new("AnyShape");
        let explanation = explainer.explain_violation(&violation, &shape);
        assert!(explanation.contains("AnyShape"));
        assert!(explanation.contains("99"));
    }
}
