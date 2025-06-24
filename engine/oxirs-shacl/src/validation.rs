//! SHACL validation engine implementation
//! 
//! This module implements the core validation engine that orchestrates SHACL validation.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{NamedNode, Term, Triple, BlankNode, Literal, RdfTerm},
    store::Store,
    OxirsError,
};

use crate::{
    ShaclError, Result, Shape, ShapeId, PropertyPath, Target, Constraint, 
    ConstraintComponentId, Severity, ValidationConfig, ValidationReport,
    constraints::*,
    targets::*,
    paths::*,
    sparql::*,
    report::*,
};

/// Core SHACL validation engine
#[derive(Debug)]
pub struct ValidationEngine<'a> {
    /// Reference to shapes to validate against
    shapes: &'a IndexMap<ShapeId, Shape>,
    
    /// Validation configuration
    config: ValidationConfig,
    
    /// Target selector for finding focus nodes
    target_selector: TargetSelector,
    
    /// Property path evaluator
    path_evaluator: PropertyPathEvaluator,
    
    /// SPARQL constraint executor
    sparql_executor: SparqlConstraintExecutor,
    
    /// Validation statistics
    stats: ValidationStats,
}

impl<'a> ValidationEngine<'a> {
    /// Create a new validation engine
    pub fn new(shapes: &'a IndexMap<ShapeId, Shape>, config: ValidationConfig) -> Self {
        Self {
            shapes,
            config,
            target_selector: TargetSelector::new(),
            path_evaluator: PropertyPathEvaluator::new(),
            sparql_executor: SparqlConstraintExecutor::new(),
            stats: ValidationStats::default(),
        }
    }
    
    /// Validate all data in a store against all loaded shapes
    pub fn validate_store(&mut self, store: &Store) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let mut report = ValidationReport::new();
        
        // Validate each active shape
        for (shape_id, shape) in self.shapes {
            if shape.is_active() {
                let shape_result = self.validate_shape(store, shape, None)?;
                report.merge_result(shape_result);
                
                // Check if we should stop early
                if self.config.fail_fast && !report.conforms() {
                    break;
                }
                
                // Check violation limit
                if self.config.max_violations > 0 && report.violation_count() >= self.config.max_violations {
                    break;
                }
            }
        }
        
        // Update statistics
        self.stats.total_validations += 1;
        self.stats.total_validation_time += start_time.elapsed();
        self.stats.last_validation_time = Some(start_time.elapsed());
        
        Ok(report)
    }
    
    /// Validate specific nodes against a specific shape
    pub fn validate_nodes(&mut self, store: &Store, shape: &Shape, nodes: &[Term]) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let mut report = ValidationReport::new();
        
        for node in nodes {
            let node_result = self.validate_node_against_shape(store, shape, node, None)?;
            report.merge_result(node_result);
            
            // Check early termination conditions
            if self.config.fail_fast && !report.conforms() {
                break;
            }
            
            if self.config.max_violations > 0 && report.violation_count() >= self.config.max_violations {
                break;
            }
        }
        
        // Update statistics
        self.stats.total_node_validations += nodes.len();
        self.stats.total_validation_time += start_time.elapsed();
        
        Ok(report)
    }
    
    /// Validate a shape against its targets
    fn validate_shape(&mut self, store: &Store, shape: &Shape, graph_name: Option<&str>) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        
        // If no explicit targets, this might be an implicit target shape
        if shape.targets.is_empty() && shape.is_node_shape() {
            // Try using the shape IRI as an implicit class target
            let implicit_target = Target::implicit(
                NamedNode::new(shape.id.as_str())
                    .map_err(|e| ShaclError::TargetSelection(format!("Invalid shape IRI: {}", e)))?
            );
            let target_nodes = self.target_selector.select_targets(store, &implicit_target, graph_name)?;
            
            for node in target_nodes {
                let node_result = self.validate_node_against_shape(store, shape, &node, graph_name)?;
                report.merge_result(node_result);
                
                if self.should_stop_validation(&report) {
                    break;
                }
            }
        } else {
            // Validate against explicit targets
            for target in &shape.targets {
                let target_nodes = self.target_selector.select_targets(store, target, graph_name)?;
                
                for node in target_nodes {
                    let node_result = self.validate_node_against_shape(store, shape, &node, graph_name)?;
                    report.merge_result(node_result);
                    
                    if self.should_stop_validation(&report) {
                        break;
                    }
                }
                
                if self.should_stop_validation(&report) {
                    break;
                }
            }
        }
        
        Ok(report)
    }
    
    /// Validate a specific node against a shape
    fn validate_node_against_shape(&mut self, store: &Store, shape: &Shape, focus_node: &Term, graph_name: Option<&str>) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        
        if shape.is_node_shape() {
            // For node shapes, validate constraints directly against the focus node
            let values = vec![focus_node.clone()];
            let constraint_results = self.validate_constraints(store, shape, focus_node, None, &values, graph_name)?;
            
            for result in constraint_results {
                if let Some(violation) = result {
                    report.add_violation(violation);
                }
            }
        } else if shape.is_property_shape() {
            // For property shapes, evaluate the property path first
            if let Some(path) = &shape.path {
                let values = self.path_evaluator.evaluate_path(store, focus_node, path, graph_name)?;
                let constraint_results = self.validate_constraints(store, shape, focus_node, Some(path), &values, graph_name)?;
                
                for result in constraint_results {
                    if let Some(violation) = result {
                        report.add_violation(violation);
                    }
                }
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Property shape must have a property path".to_string()
                ));
            }
        }
        
        Ok(report)
    }
    
    /// Validate all constraints for a shape
    fn validate_constraints(
        &mut self, 
        store: &Store, 
        shape: &Shape, 
        focus_node: &Term, 
        path: Option<&PropertyPath>,
        values: &[Term],
        graph_name: Option<&str>
    ) -> Result<Vec<Option<ValidationViolation>>> {
        let mut results = Vec::new();
        
        for (component_id, constraint) in &shape.constraints {
            let context = ConstraintContext::new(focus_node.clone(), shape.id.clone())
                .with_values(values.to_vec());
            
            let constraint_result = self.validate_constraint(store, constraint, &context, path, graph_name)?;
            
            match constraint_result {
                ConstraintEvaluationResult::Satisfied => {
                    results.push(None);
                }
                ConstraintEvaluationResult::Violated { violating_value, message, details } => {
                    let violation = ValidationViolation {
                        focus_node: focus_node.clone(),
                        source_shape: shape.id.clone(),
                        source_constraint_component: component_id.clone(),
                        result_path: path.cloned(),
                        value: violating_value,
                        result_message: message.or_else(|| constraint.message().map(|s| s.to_string())),
                        result_severity: constraint.severity().unwrap_or_else(|| shape.severity.clone()),
                        details,
                    };
                    results.push(Some(violation));
                }
                ConstraintEvaluationResult::Error { message, details } => {
                    return Err(ShaclError::ConstraintValidation(
                        format!("Constraint evaluation error for {}: {}", component_id.as_str(), message)
                    ));
                }
            }
        }
        
        Ok(results)
    }
    
    /// Validate a single constraint
    fn validate_constraint(
        &mut self,
        store: &Store,
        constraint: &Constraint,
        context: &ConstraintContext,
        path: Option<&PropertyPath>,
        graph_name: Option<&str>
    ) -> Result<ConstraintEvaluationResult> {
        match constraint {
            Constraint::Class(c) => self.validate_class_constraint(store, c, context, graph_name),
            Constraint::Datatype(c) => self.validate_datatype_constraint(c, context),
            Constraint::NodeKind(c) => self.validate_node_kind_constraint(c, context),
            Constraint::MinCount(c) => self.validate_min_count_constraint(c, context),
            Constraint::MaxCount(c) => self.validate_max_count_constraint(c, context),
            Constraint::MinLength(c) => self.validate_min_length_constraint(c, context),
            Constraint::MaxLength(c) => self.validate_max_length_constraint(c, context),
            Constraint::Pattern(c) => self.validate_pattern_constraint(c, context),
            Constraint::In(c) => self.validate_in_constraint(c, context),
            Constraint::HasValue(c) => self.validate_has_value_constraint(c, context),
            Constraint::Sparql(c) => self.validate_sparql_constraint(store, c, context, graph_name),
            // TODO: Implement remaining constraint types
            _ => Ok(ConstraintEvaluationResult::satisfied()),
        }
    }
    
    /// Validate class constraint
    fn validate_class_constraint(&self, store: &Store, constraint: &ClassConstraint, context: &ConstraintContext, graph_name: Option<&str>) -> Result<ConstraintEvaluationResult> {
        // TODO: Implement class constraint validation
        // Check if all values are instances of the specified class
        Ok(ConstraintEvaluationResult::satisfied())
    }
    
    /// Validate datatype constraint
    fn validate_datatype_constraint(&self, constraint: &DatatypeConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                // TODO: Check if literal has the required datatype
                // For now, just check if it's a literal
                continue;
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!("Value must be a literal with datatype {}", constraint.datatype_iri.as_str()))
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
    
    /// Validate node kind constraint
    fn validate_node_kind_constraint(&self, constraint: &NodeKindConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            let matches = match &constraint.node_kind {
                NodeKind::Iri => value.is_named_node(),
                NodeKind::BlankNode => value.is_blank_node(),
                NodeKind::Literal => value.is_literal(),
                NodeKind::BlankNodeOrIri => value.is_blank_node() || value.is_named_node(),
                NodeKind::BlankNodeOrLiteral => value.is_blank_node() || value.is_literal(),
                NodeKind::IriOrLiteral => value.is_named_node() || value.is_literal(),
            };
            
            if !matches {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some(format!("Value does not match required node kind: {:?}", constraint.node_kind))
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
    
    /// Validate minimum count constraint
    fn validate_min_count_constraint(&self, constraint: &MinCountConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        let actual_count = context.values.len() as u32;
        if actual_count < constraint.min_count {
            Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!("Minimum count {} not satisfied, found {}", constraint.min_count, actual_count))
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }
    
    /// Validate maximum count constraint
    fn validate_max_count_constraint(&self, constraint: &MaxCountConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        let actual_count = context.values.len() as u32;
        if actual_count > constraint.max_count {
            Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!("Maximum count {} exceeded, found {}", constraint.max_count, actual_count))
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }
    
    /// Validate minimum length constraint
    fn validate_min_length_constraint(&self, constraint: &MinLengthConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                let length = literal.as_str().chars().count() as u32;
                if length < constraint.min_length {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!("Minimum length {} not satisfied, found {}", constraint.min_length, length))
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
    
    /// Validate maximum length constraint
    fn validate_max_length_constraint(&self, constraint: &MaxLengthConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                let length = literal.as_str().chars().count() as u32;
                if length > constraint.max_length {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!("Maximum length {} exceeded, found {}", constraint.max_length, length))
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
    
    /// Validate pattern constraint
    fn validate_pattern_constraint(&self, constraint: &PatternConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        use regex::{Regex, RegexBuilder};
        
        // Build regex with flags
        let regex = if let Some(flags) = &constraint.flags {
            let case_insensitive = flags.contains('i');
            let multi_line = flags.contains('m');
            let dot_matches_new_line = flags.contains('s');
            
            RegexBuilder::new(&constraint.pattern)
                .case_insensitive(case_insensitive)
                .multi_line(multi_line)
                .dot_matches_new_line(dot_matches_new_line)
                .build()
                .map_err(|e| ShaclError::ConstraintValidation(format!("Invalid regex: {}", e)))?
        } else {
            Regex::new(&constraint.pattern)
                .map_err(|e| ShaclError::ConstraintValidation(format!("Invalid regex: {}", e)))?
        };
        
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !regex.is_match(literal.as_str()) {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        constraint.message.clone().or_else(|| 
                            Some(format!("Value does not match pattern: {}", constraint.pattern))
                        )
                    ));
                }
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
    
    /// Validate in constraint
    fn validate_in_constraint(&self, constraint: &InConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if !constraint.values.contains(value) {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value is not in the allowed list".to_string())
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
    
    /// Validate has value constraint
    fn validate_has_value_constraint(&self, constraint: &HasValueConstraint, context: &ConstraintContext) -> Result<ConstraintEvaluationResult> {
        if !context.values.contains(&constraint.value) {
            Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!("Required value {} not found", constraint.value.as_str()))
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }
    
    /// Validate SPARQL constraint
    fn validate_sparql_constraint(&mut self, store: &Store, constraint: &SparqlConstraint, context: &ConstraintContext, graph_name: Option<&str>) -> Result<ConstraintEvaluationResult> {
        let bindings = SparqlBindings::new()
            .with_this(context.focus_node.clone())
            .with_current_shape(Term::NamedNode(
                NamedNode::new(context.shape_id.as_str())
                    .map_err(|e| ShaclError::SparqlExecution(format!("Invalid shape IRI: {}", e)))?
            ));
        
        let result = self.sparql_executor.execute_constraint(store, constraint, &bindings, graph_name)?;
        
        if result.is_violation() {
            Ok(ConstraintEvaluationResult::violated(
                None,
                constraint.message.clone().or_else(|| 
                    Some("SPARQL constraint violation".to_string())
                )
            ))
        } else {
            Ok(ConstraintEvaluationResult::satisfied())
        }
    }
    
    /// Check if validation should stop early
    fn should_stop_validation(&self, report: &ValidationReport) -> bool {
        (self.config.fail_fast && !report.conforms()) ||
        (self.config.max_violations > 0 && report.violation_count() >= self.config.max_violations)
    }
    
    /// Get validation statistics
    pub fn get_statistics(&self) -> &ValidationStats {
        &self.stats
    }
    
    /// Clear internal caches
    pub fn clear_caches(&mut self) {
        self.target_selector.clear_cache();
        self.path_evaluator.clear_cache();
    }
}

/// Validation performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_validations: usize,
    pub total_node_validations: usize,
    pub total_constraint_evaluations: usize,
    pub total_validation_time: Duration,
    pub last_validation_time: Option<Duration>,
    pub avg_validation_time: Duration,
    pub constraint_evaluation_times: HashMap<String, Duration>,
}

impl ValidationStats {
    pub fn record_constraint_evaluation(&mut self, constraint_type: String, duration: Duration) {
        self.total_constraint_evaluations += 1;
        *self.constraint_evaluation_times.entry(constraint_type).or_insert(Duration::ZERO) += duration;
        
        if self.total_validations > 0 {
            self.avg_validation_time = self.total_validation_time / self.total_validations as u32;
        }
    }
    
    pub fn get_avg_constraint_time(&self, constraint_type: &str) -> Option<Duration> {
        self.constraint_evaluation_times.get(constraint_type)
            .map(|total| *total / self.total_constraint_evaluations as u32)
    }
}

/// Validation violation with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// The focus node where the violation occurred
    pub focus_node: Term,
    
    /// The source shape that was violated
    pub source_shape: ShapeId,
    
    /// The constraint component that was violated
    pub source_constraint_component: ConstraintComponentId,
    
    /// The property path where the violation occurred (for property shapes)
    pub result_path: Option<PropertyPath>,
    
    /// The specific value that caused the violation
    pub value: Option<Term>,
    
    /// Human-readable message describing the violation
    pub result_message: Option<String>,
    
    /// Severity of the violation
    pub result_severity: Severity,
    
    /// Additional details about the violation
    pub details: HashMap<String, String>,
}

impl ValidationViolation {
    pub fn new(
        focus_node: Term,
        source_shape: ShapeId,
        source_constraint_component: ConstraintComponentId,
        result_severity: Severity
    ) -> Self {
        Self {
            focus_node,
            source_shape,
            source_constraint_component,
            result_path: None,
            value: None,
            result_message: None,
            result_severity,
            details: HashMap::new(),
        }
    }
    
    pub fn with_path(mut self, path: PropertyPath) -> Self {
        self.result_path = Some(path);
        self
    }
    
    pub fn with_value(mut self, value: Term) -> Self {
        self.value = Some(value);
        self
    }
    
    pub fn with_message(mut self, message: String) -> Self {
        self.result_message = Some(message);
        self
    }
    
    pub fn with_detail(mut self, key: String, value: String) -> Self {
        self.details.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, ShapeType, PropertyPath};
    
    #[test]
    fn test_validation_engine_creation() {
        let shapes = IndexMap::new();
        let config = ValidationConfig::default();
        let engine = ValidationEngine::new(&shapes, config.clone());
        
        assert_eq!(engine.shapes.len(), 0);
        assert_eq!(engine.config.max_violations, config.max_violations);
    }
    
    #[test]
    fn test_validation_violation() {
        let focus_node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
        let shape_id = ShapeId::new("http://example.org/PersonShape");
        let component_id = ConstraintComponentId::new("sh:ClassConstraintComponent");
        
        let violation = ValidationViolation::new(
            focus_node.clone(),
            shape_id.clone(),
            component_id.clone(),
            Severity::Violation
        )
        .with_message("Test violation".to_string())
        .with_detail("test_key".to_string(), "test_value".to_string());
        
        assert_eq!(violation.focus_node, focus_node);
        assert_eq!(violation.source_shape, shape_id);
        assert_eq!(violation.source_constraint_component, component_id);
        assert_eq!(violation.result_severity, Severity::Violation);
        assert_eq!(violation.result_message, Some("Test violation".to_string()));
        assert_eq!(violation.details.get("test_key"), Some(&"test_value".to_string()));
    }
    
    #[test]
    fn test_validation_stats() {
        let mut stats = ValidationStats::default();
        
        let duration = Duration::from_millis(100);
        stats.record_constraint_evaluation("ClassConstraint".to_string(), duration);
        stats.record_constraint_evaluation("ClassConstraint".to_string(), duration);
        
        assert_eq!(stats.total_constraint_evaluations, 2);
        assert_eq!(
            stats.constraint_evaluation_times.get("ClassConstraint"),
            Some(&Duration::from_millis(200))
        );
    }
    
    #[test]
    fn test_node_kind_validation() {
        let engine = create_test_engine();
        let constraint = NodeKindConstraint { node_kind: NodeKind::Iri };
        
        // Test with IRI - should pass
        let iri_context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape")
        ).with_values(vec![Term::NamedNode(NamedNode::new("http://example.org/value").unwrap())]);
        
        let result = engine.validate_node_kind_constraint(&constraint, &iri_context).unwrap();
        assert!(result.is_satisfied());
        
        // Test with literal - should fail
        let literal_context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape")
        ).with_values(vec![Term::Literal(Literal::new("test"))]);
        
        let result = engine.validate_node_kind_constraint(&constraint, &literal_context).unwrap();
        assert!(result.is_violated());
    }
    
    #[test]
    fn test_cardinality_validation() {
        let engine = create_test_engine();
        
        // Test min count constraint
        let min_constraint = MinCountConstraint { min_count: 2 };
        let context_insufficient = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape")
        ).with_values(vec![Term::Literal(Literal::new("value1"))]);
        
        let result = engine.validate_min_count_constraint(&min_constraint, &context_insufficient).unwrap();
        assert!(result.is_violated());
        
        let context_sufficient = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("test_shape")
        ).with_values(vec![
            Term::Literal(Literal::new("value1")),
            Term::Literal(Literal::new("value2")),
        ]);
        
        let result = engine.validate_min_count_constraint(&min_constraint, &context_sufficient).unwrap();
        assert!(result.is_satisfied());
    }
    
    fn create_test_engine() -> ValidationEngine<'static> {
        let shapes = Box::leak(Box::new(IndexMap::new()));
        let config = ValidationConfig::default();
        ValidationEngine::new(shapes, config)
    }
}