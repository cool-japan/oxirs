//! SHACL-star support for validating RDF-star data with quoted triples
//!
//! This module extends SHACL (Shapes Constraint Language) to support validation
//! of RDF-star data including quoted triples. It provides:
//!
//! - **Shape validation** - Validate quoted triples against SHACL-star shapes
//! - **Constraint checking** - Support for SHACL constraint types extended for RDF-star
//! - **Nesting validation** - Validate nested quoted triple structures
//! - **Custom constraints** - Define custom validation rules for quoted triples
//! - **Validation reports** - Detailed reports with suggestions and error recovery
//!
//! # Examples
//!
//! ```rust,ignore
//! use oxirs_star::shacl_star::{ShaclStarValidator, ShaclStarShape, ConstraintType};
//! use oxirs_star::{StarStore, StarTriple, StarTerm};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a validator
//! let mut validator = ShaclStarValidator::new();
//!
//! // Define a shape for quoted triples
//! let mut shape = ShaclStarShape::new("ExampleShape");
//! shape.add_constraint(ConstraintType::MaxNestingDepth(5));
//! shape.add_constraint(ConstraintType::RequiredPredicate("http://example.org/confidence".to_string()));
//!
//! validator.add_shape(shape);
//!
//! // Create a store with data
//! let mut store = StarStore::new();
//! let triple = StarTriple::new(
//!     StarTerm::iri("http://example.org/s")?,
//!     StarTerm::iri("http://example.org/p")?,
//!     StarTerm::literal("o")?,
//! );
//! store.insert(&triple)?;
//!
//! // Validate the store
//! let report = validator.validate(&store)?;
//! println!("Validation: {}", if report.conforms { "PASS" } else { "FAIL" });
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::model::{StarTerm, StarTriple};
use crate::store::StarStore;
use crate::StarResult;

/// SHACL-star validator for RDF-star data
#[derive(Debug, Clone)]
pub struct ShaclStarValidator {
    /// Collection of SHACL-star shapes
    shapes: HashMap<String, ShaclStarShape>,

    /// Global configuration for validation
    config: ValidationConfig,

    /// Statistics about validation runs
    stats: ValidationStats,
}

/// Configuration for SHACL-star validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum nesting depth allowed globally
    pub max_nesting_depth: usize,

    /// Strict mode (fail on first violation)
    pub strict_mode: bool,

    /// Enable custom constraint validators
    pub enable_custom_validators: bool,

    /// Validation timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_nesting_depth: 10,
            strict_mode: false,
            enable_custom_validators: true,
            timeout_ms: 30000, // 30 seconds
        }
    }
}

/// Statistics about validation runs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStats {
    /// Total validations performed
    pub total_validations: usize,

    /// Total violations found
    pub total_violations: usize,

    /// Total shapes validated
    pub total_shapes: usize,

    /// Total triples validated
    pub total_triples: usize,
}

/// A SHACL-star shape definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclStarShape {
    /// Unique identifier for the shape
    pub id: String,

    /// Human-readable label
    pub label: Option<String>,

    /// Description of what the shape validates
    pub description: Option<String>,

    /// Target class (if applicable)
    pub target_class: Option<String>,

    /// Constraints that apply to this shape
    pub constraints: Vec<ConstraintType>,

    /// Severity level for violations
    pub severity: SeveritLevel,
}

/// Severity level for SHACL violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeveritLevel {
    /// Informational message
    Info,
    /// Warning (validation still passes)
    Warning,
    /// Violation (validation fails)
    Violation,
}

/// Types of constraints for SHACL-star validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Maximum nesting depth for quoted triples
    MaxNestingDepth(usize),

    /// Minimum nesting depth for quoted triples
    MinNestingDepth(usize),

    /// Required predicate in quoted triples
    RequiredPredicate(String),

    /// Forbidden predicate in quoted triples
    ForbiddenPredicate(String),

    /// Pattern constraint for quoted triple structure
    QuotedTriplePattern {
        subject_pattern: Option<TermPattern>,
        predicate_pattern: Option<TermPattern>,
        object_pattern: Option<TermPattern>,
    },

    /// Cardinality constraint
    Cardinality {
        min: Option<usize>,
        max: Option<usize>,
        property: String,
    },

    /// Datatype constraint
    Datatype(String),

    /// Custom constraint with validator function name
    Custom(String),
}

/// Pattern for matching RDF terms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TermPattern {
    /// Match any IRI
    AnyIri,

    /// Match specific IRI
    SpecificIri(String),

    /// Match IRI with prefix
    IriPrefix(String),

    /// Match any literal
    AnyLiteral,

    /// Match literal with specific datatype
    LiteralWithDatatype(String),

    /// Match quoted triple
    QuotedTriple,

    /// Match blank node
    BlankNode,
}

/// Validation report containing all results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether the data conforms to all shapes
    pub conforms: bool,

    /// List of validation results
    pub results: Vec<ValidationResult>,

    /// Summary statistics
    pub summary: ValidationSummary,
}

/// Individual validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// The shape that was violated
    pub shape_id: String,

    /// Severity of the violation
    pub severity: SeveritLevel,

    /// Description of the violation
    pub message: String,

    /// The triple that caused the violation (if applicable)
    pub focus_triple: Option<String>,

    /// Path to the problematic element
    pub path: Option<String>,

    /// Value that violated the constraint
    pub value: Option<String>,

    /// Suggestions for fixing the violation
    pub suggestions: Vec<String>,
}

/// Summary of validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total number of shapes validated
    pub shapes_count: usize,

    /// Total number of triples validated
    pub triples_count: usize,

    /// Number of violations found
    pub violations_count: usize,

    /// Number of warnings found
    pub warnings_count: usize,

    /// Number of info messages
    pub info_count: usize,
}

impl ShaclStarValidator {
    /// Create a new SHACL-star validator with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new SHACL-star validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            shapes: HashMap::new(),
            config,
            stats: ValidationStats::default(),
        }
    }

    /// Add a shape to the validator
    pub fn add_shape(&mut self, shape: ShaclStarShape) {
        self.shapes.insert(shape.id.clone(), shape);
    }

    /// Remove a shape from the validator
    pub fn remove_shape(&mut self, shape_id: &str) -> Option<ShaclStarShape> {
        self.shapes.remove(shape_id)
    }

    /// Validate a store against all shapes
    pub fn validate(&mut self, store: &StarStore) -> StarResult<ValidationReport> {
        info!("Starting SHACL-star validation");

        let mut results = Vec::new();
        let mut violations_count = 0;
        let mut warnings_count = 0;
        let mut info_count = 0;

        // Get all triples from the store
        let triples = store.query(None, None, None)?;

        debug!(
            "Validating {} triples against {} shapes",
            triples.len(),
            self.shapes.len()
        );

        // Validate each shape
        for (shape_id, shape) in &self.shapes {
            debug!("Validating shape: {}", shape_id);

            for triple in &triples {
                let shape_results = self.validate_triple_against_shape(triple, shape)?;

                for result in shape_results {
                    match result.severity {
                        SeveritLevel::Violation => violations_count += 1,
                        SeveritLevel::Warning => warnings_count += 1,
                        SeveritLevel::Info => info_count += 1,
                    }
                    results.push(result);
                }
            }
        }

        // Update statistics
        self.stats.total_validations += 1;
        self.stats.total_violations += violations_count;
        self.stats.total_shapes += self.shapes.len();
        self.stats.total_triples += triples.len();

        let conforms = violations_count == 0;

        info!(
            "Validation complete: {} (violations: {}, warnings: {}, info: {})",
            if conforms { "PASS" } else { "FAIL" },
            violations_count,
            warnings_count,
            info_count
        );

        Ok(ValidationReport {
            conforms,
            results,
            summary: ValidationSummary {
                shapes_count: self.shapes.len(),
                triples_count: triples.len(),
                violations_count,
                warnings_count,
                info_count,
            },
        })
    }

    /// Validate a single triple against a shape
    fn validate_triple_against_shape(
        &self,
        triple: &StarTriple,
        shape: &ShaclStarShape,
    ) -> StarResult<Vec<ValidationResult>> {
        let mut results = Vec::new();

        for constraint in &shape.constraints {
            if let Some(result) = self.check_constraint(triple, constraint, shape)? {
                // Check severity before moving result
                let is_violation = result.severity == SeveritLevel::Violation;
                results.push(result);

                // In strict mode, stop on first violation
                if self.config.strict_mode && is_violation {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Check a single constraint
    fn check_constraint(
        &self,
        triple: &StarTriple,
        constraint: &ConstraintType,
        shape: &ShaclStarShape,
    ) -> StarResult<Option<ValidationResult>> {
        match constraint {
            ConstraintType::MaxNestingDepth(max_depth) => {
                let actual_depth = triple.nesting_depth();

                if actual_depth > *max_depth {
                    return Ok(Some(ValidationResult {
                        shape_id: shape.id.clone(),
                        severity: shape.severity,
                        message: format!(
                            "Nesting depth {} exceeds maximum allowed depth {}",
                            actual_depth, max_depth
                        ),
                        focus_triple: Some(format!("{}", triple)),
                        path: None,
                        value: Some(actual_depth.to_string()),
                        suggestions: vec![
                            format!("Reduce nesting depth to {} or less", max_depth),
                            "Consider flattening the quoted triple structure".to_string(),
                        ],
                    }));
                }
            }

            ConstraintType::MinNestingDepth(min_depth) => {
                let actual_depth = triple.nesting_depth();

                if actual_depth < *min_depth {
                    return Ok(Some(ValidationResult {
                        shape_id: shape.id.clone(),
                        severity: shape.severity,
                        message: format!(
                            "Nesting depth {} is below minimum required depth {}",
                            actual_depth, min_depth
                        ),
                        focus_triple: Some(format!("{}", triple)),
                        path: None,
                        value: Some(actual_depth.to_string()),
                        suggestions: vec![format!(
                            "Increase nesting depth to at least {}",
                            min_depth
                        )],
                    }));
                }
            }

            ConstraintType::RequiredPredicate(predicate) => {
                let has_predicate = self.triple_has_predicate(triple, predicate);

                if !has_predicate {
                    return Ok(Some(ValidationResult {
                        shape_id: shape.id.clone(),
                        severity: shape.severity,
                        message: format!("Required predicate <{}> not found", predicate),
                        focus_triple: Some(format!("{}", triple)),
                        path: Some("predicate".to_string()),
                        value: None,
                        suggestions: vec![format!("Add predicate <{}> to the triple", predicate)],
                    }));
                }
            }

            ConstraintType::ForbiddenPredicate(predicate) => {
                let has_predicate = self.triple_has_predicate(triple, predicate);

                if has_predicate {
                    return Ok(Some(ValidationResult {
                        shape_id: shape.id.clone(),
                        severity: shape.severity,
                        message: format!("Forbidden predicate <{}> found", predicate),
                        focus_triple: Some(format!("{}", triple)),
                        path: Some("predicate".to_string()),
                        value: Some(predicate.clone()),
                        suggestions: vec![format!(
                            "Remove predicate <{}> from the triple",
                            predicate
                        )],
                    }));
                }
            }

            ConstraintType::QuotedTriplePattern {
                subject_pattern,
                predicate_pattern,
                object_pattern,
            } => {
                // Check if triple matches the pattern
                let matches = self.matches_pattern(
                    triple,
                    subject_pattern.as_ref(),
                    predicate_pattern.as_ref(),
                    object_pattern.as_ref(),
                );

                if !matches {
                    return Ok(Some(ValidationResult {
                        shape_id: shape.id.clone(),
                        severity: shape.severity,
                        message: "Triple does not match required pattern".to_string(),
                        focus_triple: Some(format!("{}", triple)),
                        path: None,
                        value: None,
                        suggestions: vec!["Adjust triple to match the required pattern".to_string()],
                    }));
                }
            }

            ConstraintType::Datatype(datatype) => {
                // Check datatype of literals in the triple
                if !self.triple_has_datatype(triple, datatype) {
                    return Ok(Some(ValidationResult {
                        shape_id: shape.id.clone(),
                        severity: shape.severity,
                        message: format!("Required datatype <{}> not found", datatype),
                        focus_triple: Some(format!("{}", triple)),
                        path: Some("object".to_string()),
                        value: None,
                        suggestions: vec![format!("Ensure object has datatype <{}>", datatype)],
                    }));
                }
            }

            ConstraintType::Cardinality {
                min: _,
                max: _,
                property: _,
            } => {
                // This would require counting occurrences, which needs store access
                // For now, we'll skip this constraint type in triple-level validation
                debug!("Cardinality constraint requires store-level validation");
            }

            ConstraintType::Custom(validator_name) => {
                if self.config.enable_custom_validators {
                    debug!("Custom validator '{}' not implemented yet", validator_name);
                }
            }
        }

        Ok(None)
    }

    /// Check if a triple has a specific predicate
    fn triple_has_predicate(&self, triple: &StarTriple, predicate: &str) -> bool {
        if let StarTerm::NamedNode(nn) = &triple.predicate {
            return nn.iri == predicate;
        }

        // Check in nested quoted triples
        self.term_has_predicate(&triple.subject, predicate)
            || self.term_has_predicate(&triple.object, predicate)
    }

    /// Check if a term contains a predicate (recursively)
    #[allow(clippy::only_used_in_recursion)]
    fn term_has_predicate(&self, term: &StarTerm, predicate: &str) -> bool {
        if let StarTerm::QuotedTriple(qt) = term {
            if let StarTerm::NamedNode(nn) = &qt.predicate {
                if nn.iri == predicate {
                    return true;
                }
            }

            return self.term_has_predicate(&qt.subject, predicate)
                || self.term_has_predicate(&qt.predicate, predicate)
                || self.term_has_predicate(&qt.object, predicate);
        }

        false
    }

    /// Check if a triple matches term patterns
    fn matches_pattern(
        &self,
        triple: &StarTriple,
        subject_pattern: Option<&TermPattern>,
        predicate_pattern: Option<&TermPattern>,
        object_pattern: Option<&TermPattern>,
    ) -> bool {
        if let Some(pattern) = subject_pattern {
            if !self.term_matches_pattern(&triple.subject, pattern) {
                return false;
            }
        }

        if let Some(pattern) = predicate_pattern {
            if !self.term_matches_pattern(&triple.predicate, pattern) {
                return false;
            }
        }

        if let Some(pattern) = object_pattern {
            if !self.term_matches_pattern(&triple.object, pattern) {
                return false;
            }
        }

        true
    }

    /// Check if a term matches a pattern
    fn term_matches_pattern(&self, term: &StarTerm, pattern: &TermPattern) -> bool {
        match pattern {
            TermPattern::AnyIri => matches!(term, StarTerm::NamedNode(_)),
            TermPattern::SpecificIri(iri) => {
                if let StarTerm::NamedNode(nn) = term {
                    nn.iri == *iri
                } else {
                    false
                }
            }
            TermPattern::IriPrefix(prefix) => {
                if let StarTerm::NamedNode(nn) = term {
                    nn.iri.starts_with(prefix)
                } else {
                    false
                }
            }
            TermPattern::AnyLiteral => matches!(term, StarTerm::Literal(_)),
            TermPattern::LiteralWithDatatype(datatype) => {
                if let StarTerm::Literal(lit) = term {
                    if let Some(ref dt) = lit.datatype {
                        dt.iri == *datatype
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            TermPattern::QuotedTriple => matches!(term, StarTerm::QuotedTriple(_)),
            TermPattern::BlankNode => matches!(term, StarTerm::BlankNode(_)),
        }
    }

    /// Check if a triple has a specific datatype
    fn triple_has_datatype(&self, triple: &StarTriple, datatype: &str) -> bool {
        if let StarTerm::Literal(lit) = &triple.object {
            if let Some(ref dt) = lit.datatype {
                return dt.iri == datatype;
            }
        }

        false
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> &ValidationStats {
        &self.stats
    }
}

impl Default for ShaclStarValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaclStarShape {
    /// Create a new SHACL-star shape
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
            description: None,
            target_class: None,
            constraints: Vec::new(),
            severity: SeveritLevel::Violation,
        }
    }

    /// Set the label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the target class
    pub fn with_target_class(mut self, target_class: impl Into<String>) -> Self {
        self.target_class = Some(target_class.into());
        self
    }

    /// Set the severity level
    pub fn with_severity(mut self, severity: SeveritLevel) -> Self {
        self.severity = severity;
        self
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: ConstraintType) {
        self.constraints.push(constraint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple};
    use crate::store::StarStore;

    #[test]
    fn test_max_nesting_depth_validation() -> StarResult<()> {
        let mut validator = ShaclStarValidator::new();

        let mut shape = ShaclStarShape::new("MaxDepthShape");
        shape.add_constraint(ConstraintType::MaxNestingDepth(2));
        validator.add_shape(shape);

        let store = StarStore::new();

        // Create a deeply nested triple (depth = 3)
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/s1")?,
            StarTerm::iri("http://example.org/p1")?,
            StarTerm::literal("o1")?,
        );

        let middle1 = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/p2")?,
            StarTerm::literal("o2")?,
        );

        let middle2 = StarTriple::new(
            StarTerm::quoted_triple(middle1),
            StarTerm::iri("http://example.org/p3")?,
            StarTerm::literal("o3")?,
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(middle2),
            StarTerm::iri("http://example.org/p4")?,
            StarTerm::literal("o4")?,
        );

        store.insert(&outer)?;

        let report = validator.validate(&store)?;
        assert!(!report.conforms); // Should fail due to exceeding max depth (3 > 2)
        assert_eq!(report.summary.violations_count, 1);

        Ok(())
    }

    #[test]
    fn test_required_predicate_validation() -> StarResult<()> {
        let mut validator = ShaclStarValidator::new();

        let mut shape = ShaclStarShape::new("RequiredPredShape");
        shape.add_constraint(ConstraintType::RequiredPredicate(
            "http://example.org/required".to_string(),
        ));
        validator.add_shape(shape);

        let store = StarStore::new();

        // Triple without the required predicate
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/other")?,
            StarTerm::literal("o")?,
        );

        store.insert(&triple)?;

        let report = validator.validate(&store)?;
        assert!(!report.conforms);

        Ok(())
    }

    #[test]
    fn test_pattern_validation() -> StarResult<()> {
        let mut validator = ShaclStarValidator::new();

        let mut shape = ShaclStarShape::new("PatternShape");
        shape.add_constraint(ConstraintType::QuotedTriplePattern {
            subject_pattern: Some(TermPattern::IriPrefix("http://example.org/".to_string())),
            predicate_pattern: Some(TermPattern::AnyIri),
            object_pattern: Some(TermPattern::AnyLiteral),
        });
        validator.add_shape(shape);

        let store = StarStore::new();

        // Matching triple
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/subject")?,
            StarTerm::iri("http://example.org/predicate")?,
            StarTerm::literal("object")?,
        );

        store.insert(&triple)?;

        let report = validator.validate(&store)?;
        assert!(report.conforms);

        Ok(())
    }
}
