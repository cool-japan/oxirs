//! Comprehensive validation framework for RDF-star data
//!
//! This module provides extensive validation capabilities for RDF-star graphs,
//! including syntax validation, semantic validation, SHACL-star constraints,
//! custom rules, and performance validation.

use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::shacl_star::ShaclStarValidator;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use tracing::{debug, info};

/// Validation errors
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
}

/// Validation level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationLevel {
    /// Syntax validation only
    Syntax,
    /// Syntax + basic semantic checks
    Basic,
    /// Full semantic validation
    Semantic,
    /// Complete validation (syntax + semantics + constraints)
    Complete,
    /// Strict validation with all optional checks
    Strict,
}

/// Validation severity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Issue severity
    pub severity: ValidationSeverity,

    /// Issue category
    pub category: ValidationCategory,

    /// Issue message
    pub message: String,

    /// Location (triple index, line number, etc.)
    pub location: Option<String>,

    /// Suggestion for fixing
    pub suggestion: Option<String>,

    /// Related rule/constraint ID
    pub rule_id: Option<String>,
}

/// Validation category
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationCategory {
    Syntax,
    Semantics,
    Performance,
    DataQuality,
    Consistency,
    BestPractices,
    Security,
    Compliance,
}

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Validation level
    pub level: ValidationLevel,

    /// Maximum nesting depth allowed
    pub max_nesting_depth: usize,

    /// Maximum graph size (number of triples)
    pub max_graph_size: Option<usize>,

    /// Enable IRI validation
    pub validate_iris: bool,

    /// Enable literal datatype validation
    pub validate_datatypes: bool,

    /// Enable SHACL-star validation
    pub enable_shacl: bool,

    /// Custom validation rules
    pub custom_rules: Vec<CustomValidationRule>,

    /// Stop on first error
    pub fail_fast: bool,

    /// Maximum issues to collect
    pub max_issues: Option<usize>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            level: ValidationLevel::Complete,
            max_nesting_depth: 10,
            max_graph_size: None,
            validate_iris: true,
            validate_datatypes: true,
            enable_shacl: true,
            custom_rules: Vec::new(),
            fail_fast: false,
            max_issues: Some(100),
        }
    }
}

/// Custom validation rule
#[derive(Debug, Clone)]
pub struct CustomValidationRule {
    /// Rule ID
    pub id: String,

    /// Rule description
    pub description: String,

    /// Validation function
    pub validator: fn(&StarTriple) -> Option<String>,

    /// Severity if violated
    pub severity: ValidationSeverity,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Is valid
    pub is_valid: bool,

    /// Validation level used
    pub level: ValidationLevel,

    /// Issues found
    pub issues: Vec<ValidationIssue>,

    /// Statistics
    pub statistics: ValidationStatistics,

    /// Timestamp
    pub validated_at: DateTime<Utc>,

    /// Duration (milliseconds)
    pub duration_ms: u64,
}

impl ValidationResult {
    /// Get issues by severity
    pub fn get_issues_by_severity(&self, severity: ValidationSeverity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// Get issues by category
    pub fn get_issues_by_category(&self, category: ValidationCategory) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.category == category)
            .collect()
    }

    /// Get critical issues
    pub fn get_critical_issues(&self) -> Vec<&ValidationIssue> {
        self.get_issues_by_severity(ValidationSeverity::Critical)
    }

    /// Get errors
    pub fn get_errors(&self) -> Vec<&ValidationIssue> {
        self.get_issues_by_severity(ValidationSeverity::Error)
    }

    /// Get warnings
    pub fn get_warnings(&self) -> Vec<&ValidationIssue> {
        self.get_issues_by_severity(ValidationSeverity::Warning)
    }

    /// Has critical issues
    pub fn has_critical_issues(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == ValidationSeverity::Critical)
    }

    /// Has errors
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == ValidationSeverity::Error)
    }
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatistics {
    pub total_triples: usize,
    pub quoted_triples: usize,
    pub max_nesting_depth_found: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub blank_nodes: usize,
    pub literals: usize,
    pub iris: usize,
}

/// Comprehensive validator
pub struct RdfStarValidator {
    /// Configuration
    config: ValidationConfig,

    /// SHACL validator (if enabled)
    shacl_validator: Option<ShaclStarValidator>,
}

impl RdfStarValidator {
    /// Create a new validator
    pub fn new(config: ValidationConfig) -> Self {
        let shacl_validator = if config.enable_shacl {
            Some(ShaclStarValidator::new())
        } else {
            None
        };

        Self {
            config,
            shacl_validator,
        }
    }

    /// Validate a graph
    pub fn validate(&self, graph: &StarGraph) -> Result<ValidationResult, ValidationError> {
        let start = std::time::Instant::now();
        info!("Starting validation of graph with {} triples", graph.len());

        let mut issues = Vec::new();

        // Level-based validation
        match self.config.level {
            ValidationLevel::Syntax => {
                self.validate_syntax(graph, &mut issues)?;
            }
            ValidationLevel::Basic => {
                self.validate_syntax(graph, &mut issues)?;
                self.validate_basic_semantics(graph, &mut issues)?;
            }
            ValidationLevel::Semantic => {
                self.validate_syntax(graph, &mut issues)?;
                self.validate_basic_semantics(graph, &mut issues)?;
                self.validate_semantics(graph, &mut issues)?;
            }
            ValidationLevel::Complete | ValidationLevel::Strict => {
                self.validate_syntax(graph, &mut issues)?;
                self.validate_basic_semantics(graph, &mut issues)?;
                self.validate_semantics(graph, &mut issues)?;
                self.validate_constraints(graph, &mut issues)?;
                self.validate_performance(graph, &mut issues)?;

                if self.config.level == ValidationLevel::Strict {
                    self.validate_best_practices(graph, &mut issues)?;
                }
            }
        }

        // Custom rules
        self.apply_custom_rules(graph, &mut issues)?;

        // Compute statistics
        let statistics = self.compute_statistics(graph);

        let duration_ms = start.elapsed().as_millis() as u64;

        let is_valid = !issues.iter().any(|i| {
            matches!(
                i.severity,
                ValidationSeverity::Error | ValidationSeverity::Critical
            )
        });

        let result = ValidationResult {
            is_valid,
            level: self.config.level.clone(),
            issues,
            statistics,
            validated_at: Utc::now(),
            duration_ms,
        };

        info!(
            "Validation complete: {} issues found in {}ms",
            result.issues.len(),
            duration_ms
        );

        Ok(result)
    }

    /// Validate syntax
    fn validate_syntax(
        &self,
        graph: &StarGraph,
        issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), ValidationError> {
        debug!("Validating syntax");

        for (idx, triple) in graph.iter().enumerate() {
            // Check nesting depth
            let depth = self.get_nesting_depth(triple);
            if depth >= self.config.max_nesting_depth {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    category: ValidationCategory::Syntax,
                    message: format!(
                        "Nesting depth {} exceeds or equals maximum {}",
                        depth, self.config.max_nesting_depth
                    ),
                    location: Some(format!("triple {}", idx)),
                    suggestion: Some(
                        "Reduce nesting depth or increase max_nesting_depth".to_string(),
                    ),
                    rule_id: Some("MAX_NESTING_DEPTH".to_string()),
                });

                if self.config.fail_fast {
                    break;
                }
            }

            // Validate IRIs if enabled
            if self.config.validate_iris {
                self.validate_term_iris(&triple.subject, "subject", idx, issues);
                self.validate_term_iris(&triple.predicate, "predicate", idx, issues);
                self.validate_term_iris(&triple.object, "object", idx, issues);
            }

            if let Some(max_issues) = self.config.max_issues {
                if issues.len() >= max_issues {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Validate basic semantics
    fn validate_basic_semantics(
        &self,
        graph: &StarGraph,
        issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), ValidationError> {
        debug!("Validating basic semantics");

        // Check graph size
        if let Some(max_size) = self.config.max_graph_size {
            if graph.len() > max_size {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: ValidationCategory::Performance,
                    message: format!(
                        "Graph size {} exceeds recommended maximum {}",
                        graph.len(),
                        max_size
                    ),
                    location: None,
                    suggestion: Some(
                        "Consider splitting the graph or increasing max_graph_size".to_string(),
                    ),
                    rule_id: Some("MAX_GRAPH_SIZE".to_string()),
                });
            }
        }

        Ok(())
    }

    /// Validate semantics
    fn validate_semantics(
        &self,
        _graph: &StarGraph,
        _issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), ValidationError> {
        debug!("Validating semantics");

        // Check for common semantic issues
        for (idx, triple) in _graph.iter().enumerate() {
            // Predicate should be a NamedNode
            if !matches!(triple.predicate, StarTerm::NamedNode(_)) {
                _issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    category: ValidationCategory::Semantics,
                    message: "Predicate must be an IRI".to_string(),
                    location: Some(format!("triple {}", idx)),
                    suggestion: Some("Use an IRI for the predicate".to_string()),
                    rule_id: Some("PREDICATE_IRI".to_string()),
                });
            }
        }

        Ok(())
    }

    /// Validate constraints (SHACL-star)
    fn validate_constraints(
        &self,
        _graph: &StarGraph,
        _issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), ValidationError> {
        debug!("Validating constraints");

        if let Some(ref _validator) = self.shacl_validator {
            // SHACL validation would be integrated here
            debug!("SHACL-star validation enabled");
        }

        Ok(())
    }

    /// Validate performance characteristics
    fn validate_performance(
        &self,
        graph: &StarGraph,
        issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), ValidationError> {
        debug!("Validating performance characteristics");

        // Check for performance anti-patterns
        let blank_node_count = graph
            .iter()
            .filter(|t| matches!(t.subject, StarTerm::BlankNode(_)))
            .count();

        if blank_node_count > graph.len() / 2 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                category: ValidationCategory::Performance,
                message: format!(
                    "High blank node usage ({}/{} triples) may impact performance",
                    blank_node_count,
                    graph.len()
                ),
                location: None,
                suggestion: Some(
                    "Consider using IRIs instead of blank nodes where possible".to_string(),
                ),
                rule_id: Some("HIGH_BLANK_NODE_USAGE".to_string()),
            });
        }

        Ok(())
    }

    /// Validate best practices
    fn validate_best_practices(
        &self,
        graph: &StarGraph,
        issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), ValidationError> {
        debug!("Validating best practices");

        // Check for common best practice violations
        let mut predicate_usage: HashMap<String, usize> = HashMap::new();

        for triple in graph.iter() {
            if let StarTerm::NamedNode(nn) = &triple.predicate {
                *predicate_usage.entry(nn.iri.clone()).or_insert(0) += 1;
            }
        }

        // Check for very low predicate reuse
        let single_use_predicates: Vec<_> = predicate_usage
            .iter()
            .filter(|(_, &count)| count == 1)
            .collect();

        if single_use_predicates.len() > predicate_usage.len() / 2 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Info,
                category: ValidationCategory::BestPractices,
                message: format!(
                    "Many predicates ({}/{}) are used only once",
                    single_use_predicates.len(),
                    predicate_usage.len()
                ),
                location: None,
                suggestion: Some("Consider reusing predicates where appropriate".to_string()),
                rule_id: Some("LOW_PREDICATE_REUSE".to_string()),
            });
        }

        Ok(())
    }

    /// Apply custom validation rules
    fn apply_custom_rules(
        &self,
        graph: &StarGraph,
        issues: &mut Vec<ValidationIssue>,
    ) -> Result<(), ValidationError> {
        if self.config.custom_rules.is_empty() {
            return Ok(());
        }

        debug!("Applying {} custom rules", self.config.custom_rules.len());

        for (idx, triple) in graph.iter().enumerate() {
            for rule in &self.config.custom_rules {
                if let Some(violation_msg) = (rule.validator)(triple) {
                    issues.push(ValidationIssue {
                        severity: rule.severity.clone(),
                        category: ValidationCategory::Compliance,
                        message: violation_msg,
                        location: Some(format!("triple {}", idx)),
                        suggestion: None,
                        rule_id: Some(rule.id.clone()),
                    });

                    if self.config.fail_fast {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute validation statistics
    fn compute_statistics(&self, graph: &StarGraph) -> ValidationStatistics {
        let mut unique_subjects = HashSet::new();
        let mut unique_predicates = HashSet::new();
        let mut unique_objects = HashSet::new();
        let mut blank_nodes = 0;
        let mut literals = 0;
        let mut iris = 0;
        let mut quoted_triples = 0;
        let mut max_depth = 0;

        for triple in graph.iter() {
            // Count term types
            match &triple.subject {
                StarTerm::NamedNode(nn) => {
                    unique_subjects.insert(nn.iri.clone());
                    iris += 1;
                }
                StarTerm::BlankNode(bn) => {
                    unique_subjects.insert(format!("_:{}", bn.id));
                    blank_nodes += 1;
                }
                StarTerm::QuotedTriple(_) => {
                    quoted_triples += 1;
                }
                _ => {}
            }

            if let StarTerm::NamedNode(nn) = &triple.predicate {
                unique_predicates.insert(nn.iri.clone());
                iris += 1;
            }

            match &triple.object {
                StarTerm::NamedNode(nn) => {
                    unique_objects.insert(nn.iri.clone());
                    iris += 1;
                }
                StarTerm::Literal(lit) => {
                    unique_objects.insert(lit.value.clone());
                    literals += 1;
                }
                StarTerm::BlankNode(bn) => {
                    unique_objects.insert(format!("_:{}", bn.id));
                    blank_nodes += 1;
                }
                StarTerm::QuotedTriple(_) => {
                    quoted_triples += 1;
                }
                _ => {}
            }

            let depth = self.get_nesting_depth(triple);
            if depth > max_depth {
                max_depth = depth;
            }
        }

        ValidationStatistics {
            total_triples: graph.len(),
            quoted_triples,
            max_nesting_depth_found: max_depth,
            unique_subjects: unique_subjects.len(),
            unique_predicates: unique_predicates.len(),
            unique_objects: unique_objects.len(),
            blank_nodes,
            literals,
            iris,
        }
    }

    /// Get nesting depth of a triple
    fn get_nesting_depth(&self, triple: &StarTriple) -> usize {
        let subject_depth = self.get_term_depth(&triple.subject);
        let object_depth = self.get_term_depth(&triple.object);
        subject_depth.max(object_depth)
    }

    /// Get nesting depth of a term
    fn get_term_depth(&self, term: &StarTerm) -> usize {
        match term {
            StarTerm::QuotedTriple(qt) => 1 + self.get_nesting_depth(qt),
            _ => 0,
        }
    }

    /// Validate IRIs in a term
    fn validate_term_iris(
        &self,
        term: &StarTerm,
        position: &str,
        triple_idx: usize,
        issues: &mut Vec<ValidationIssue>,
    ) {
        if let StarTerm::NamedNode(nn) = term {
            if !self.is_valid_iri(&nn.iri) {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    category: ValidationCategory::Syntax,
                    message: format!("Invalid IRI in {}: {}", position, nn.iri),
                    location: Some(format!("triple {}", triple_idx)),
                    suggestion: Some("Ensure IRI is properly formatted".to_string()),
                    rule_id: Some("INVALID_IRI".to_string()),
                });
            }
        }

        // Recurse into quoted triples
        if let StarTerm::QuotedTriple(qt) = term {
            self.validate_term_iris(
                &qt.subject,
                &format!("{}->subject", position),
                triple_idx,
                issues,
            );
            self.validate_term_iris(
                &qt.predicate,
                &format!("{}->predicate", position),
                triple_idx,
                issues,
            );
            self.validate_term_iris(
                &qt.object,
                &format!("{}->object", position),
                triple_idx,
                issues,
            );
        }
    }

    /// Check if IRI is valid (basic check)
    fn is_valid_iri(&self, iri: &str) -> bool {
        // Basic IRI validation
        iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;

    #[test]
    fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = RdfStarValidator::new(config);

        assert!(validator.shacl_validator.is_some());
    }

    #[test]
    fn test_basic_validation() -> Result<(), Box<dyn std::error::Error>> {
        let config = ValidationConfig {
            level: ValidationLevel::Basic,
            ..Default::default()
        };

        let validator = RdfStarValidator::new(config);

        let mut graph = StarGraph::new();
        graph.insert(StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/p")?,
            StarTerm::literal("object")?,
        ))?;

        let result = validator.validate(&graph)?;

        assert!(result.is_valid);
        assert_eq!(result.statistics.total_triples, 1);

        Ok(())
    }

    #[test]
    fn test_nesting_depth_validation() -> Result<(), Box<dyn std::error::Error>> {
        let config = ValidationConfig {
            level: ValidationLevel::Syntax,
            max_nesting_depth: 2,
            ..Default::default()
        };

        let validator = RdfStarValidator::new(config);

        let mut graph = StarGraph::new();

        // Create nested quoted triple (depth 1)
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/s1")?,
            StarTerm::iri("http://example.org/p1")?,
            StarTerm::literal("obj1")?,
        );

        let middle = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/p2")?,
            StarTerm::literal("obj2")?,
        );

        let outer = StarTriple::new(
            StarTerm::quoted_triple(middle),
            StarTerm::iri("http://example.org/p3")?,
            StarTerm::literal("obj3")?,
        );

        graph.insert(outer)?;

        let result = validator.validate(&graph)?;

        // Should have error because depth is 2, which equals max
        assert!(!result.is_valid);
        assert!(result.has_errors());

        Ok(())
    }

    #[test]
    fn test_custom_validation_rule() -> Result<(), Box<dyn std::error::Error>> {
        let mut config = ValidationConfig::default();

        // Add custom rule: predicate must not be "http://example.org/forbidden"
        config.custom_rules.push(CustomValidationRule {
            id: "NO_FORBIDDEN_PREDICATE".to_string(),
            description: "Predicate must not be forbidden".to_string(),
            validator: |triple| {
                if let StarTerm::NamedNode(nn) = &triple.predicate {
                    if nn.iri == "http://example.org/forbidden" {
                        return Some("Forbidden predicate used".to_string());
                    }
                }
                None
            },
            severity: ValidationSeverity::Error,
        });

        let validator = RdfStarValidator::new(config);

        let mut graph = StarGraph::new();
        graph.insert(StarTriple::new(
            StarTerm::iri("http://example.org/s")?,
            StarTerm::iri("http://example.org/forbidden")?,
            StarTerm::literal("obj")?,
        ))?;

        let result = validator.validate(&graph)?;

        assert!(!result.is_valid);
        assert_eq!(result.get_errors().len(), 1);

        Ok(())
    }
}
