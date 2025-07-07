//! Comprehensive integration tests for intelligent error processing
//!
//! This module tests the newly implemented intelligent error processing functionality
//! including error classification, root cause analysis, and repair suggestions.

use oxirs_core::{model::*, rdf_store::RdfStore};
use oxirs_shacl::{
    constraints::*, validation::ValidationViolation, ConstraintComponentId, PropertyPath, Severity,
    Shape, ShapeId, ShapeType, Target, ValidationConfig, ValidationReport,
};
use oxirs_shacl_ai::{
    error_handling::{ErrorType, IntelligentErrorHandler, RepairType},
    Result, ShaclAiAssistant,
};
use std::collections::HashMap;

/// Create a mock store with test data
fn create_test_store() -> RdfStore {
    let mut store = RdfStore::new().unwrap();

    // Add test triples
    let subject = NamedNode::new("http://example.org/person1").unwrap();
    let name_predicate = NamedNode::new("http://example.org/name").unwrap();
    let age_predicate = NamedNode::new("http://example.org/age").unwrap();
    let type_predicate = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
    let person_class = NamedNode::new("http://example.org/Person").unwrap();

    // Add complete data using Quad (with default graph)
    store
        .insert(&Quad::new(
            subject.clone(),
            type_predicate,
            person_class,
            GraphName::DefaultGraph,
        ))
        .unwrap();
    store
        .insert(&Quad::new(
            subject.clone(),
            name_predicate,
            Literal::new_simple_literal("John Doe"),
            GraphName::DefaultGraph,
        ))
        .unwrap();

    // Missing age property to trigger validation error
    store
}

/// Create test shapes for validation
fn create_test_shapes() -> Vec<Shape> {
    let mut shapes = Vec::new();

    // Create a person shape with required properties
    let person_shape_id = ShapeId::from("http://example.org/PersonShape");
    let mut person_shape = Shape::new(person_shape_id.clone(), ShapeType::NodeShape);

    // Add target for Person class
    person_shape.add_target(Target::Class(
        NamedNode::new("http://example.org/Person").unwrap(),
    ));

    // Add constraints directly to the shape (simplified approach)
    // In practice, you would use property shapes, but for testing we'll use a simpler approach

    shapes.push(person_shape);
    shapes
}

/// Create a mock validation report with test violations
fn create_test_validation_report() -> ValidationReport {
    let mut report = ValidationReport::new();
    report.conforms = false;

    // Add missing property violation
    let violation = ValidationViolation::new(
        Term::NamedNode(NamedNode::new("http://example.org/person1").unwrap()),
        ShapeId::from("http://example.org/PersonShape"),
        ConstraintComponentId::from("MinCount"),
        Severity::Violation,
    )
    .with_path(PropertyPath::predicate(
        NamedNode::new("http://example.org/age").unwrap(),
    ))
    .with_message("Required property missing: age".to_string());

    report.violations.push(violation);

    // Add datatype violation
    let datatype_violation = ValidationViolation::new(
        Term::NamedNode(NamedNode::new("http://example.org/person2").unwrap()),
        ShapeId::from("http://example.org/PersonShape"),
        ConstraintComponentId::from("Datatype"),
        Severity::Violation,
    )
    .with_path(PropertyPath::predicate(
        NamedNode::new("http://example.org/age").unwrap(),
    ))
    .with_message("Value does not match expected datatype: integer".to_string());

    report.violations.push(datatype_violation);

    report
}

#[test]
fn test_intelligent_error_handler_creation() {
    let handler = IntelligentErrorHandler::new();
    assert!(handler.config().enable_ml_classification);
}

#[test]
fn test_error_classification() -> Result<()> {
    let handler = IntelligentErrorHandler::new();
    let store = create_test_store();
    let shapes = create_test_shapes();
    let validation_report = create_test_validation_report();

    let analysis = handler.process_validation_errors(&validation_report, &store, &shapes)?;

    // Verify error classification
    assert!(!matches!(
        analysis.error_classification,
        ErrorType::Other(_)
    ));

    // Verify root cause analysis
    assert!(!analysis.root_cause_analysis.is_empty());
    assert!(analysis
        .root_cause_analysis
        .iter()
        .any(|cause| cause.contains("Missing required properties")));

    // Verify confidence score
    assert!(analysis.confidence_score > 0.0 && analysis.confidence_score <= 1.0);

    Ok(())
}

#[test]
fn test_repair_suggestions_generation() -> Result<()> {
    let handler = IntelligentErrorHandler::new();
    let store = create_test_store();
    let shapes = create_test_shapes();
    let validation_report = create_test_validation_report();

    let analysis = handler.process_validation_errors(&validation_report, &store, &shapes)?;

    // Verify repair suggestions are generated
    assert!(!analysis.fix_suggestions.is_empty());

    // Check for data correction suggestion (for missing properties)
    let has_data_correction = analysis
        .fix_suggestions
        .iter()
        .any(|suggestion| matches!(suggestion.repair_type, RepairType::DataCorrection));
    assert!(has_data_correction);

    // Check for datatype conversion suggestion
    let has_datatype_conversion = analysis
        .fix_suggestions
        .iter()
        .any(|suggestion| matches!(suggestion.repair_type, RepairType::DataTypeConversion));
    assert!(has_datatype_conversion);

    // Verify suggestions have reasonable confidence scores
    for suggestion in &analysis.fix_suggestions {
        assert!(suggestion.confidence > 0.0 && suggestion.confidence <= 1.0);
        assert!(suggestion.effort_estimate > 0.0 && suggestion.effort_estimate <= 1.0);
        assert!(suggestion.success_probability > 0.0 && suggestion.success_probability <= 1.0);
    }

    Ok(())
}

#[test]
fn test_similar_cases_finding() -> Result<()> {
    let handler = IntelligentErrorHandler::new();
    let store = create_test_store();
    let shapes = create_test_shapes();
    let validation_report = create_test_validation_report();

    let analysis = handler.process_validation_errors(&validation_report, &store, &shapes)?;

    // Verify similar cases are found when violations exist
    if !validation_report.violations.is_empty() {
        assert!(!analysis.similar_cases.is_empty());
    }

    Ok(())
}

#[test]
fn test_confidence_score_calculation() -> Result<()> {
    let handler = IntelligentErrorHandler::new();
    let store = create_test_store();
    let shapes = create_test_shapes();

    // Test with no violations
    let empty_report = ValidationReport::new();
    let analysis_empty = handler.process_validation_errors(&empty_report, &store, &shapes)?;

    // Test with violations
    let validation_report = create_test_validation_report();
    let analysis_violations =
        handler.process_validation_errors(&validation_report, &store, &shapes)?;

    // Confidence should be higher with more data (violations)
    assert!(analysis_violations.confidence_score >= analysis_empty.confidence_score);

    // Both should be valid confidence scores
    assert!(analysis_empty.confidence_score >= 0.0 && analysis_empty.confidence_score <= 1.0);
    assert!(
        analysis_violations.confidence_score >= 0.0 && analysis_violations.confidence_score <= 1.0
    );

    Ok(())
}

#[test]
fn test_end_to_end_error_processing() -> Result<()> {
    let mut assistant = ShaclAiAssistant::new();
    let store = create_test_store();
    let shapes = create_test_shapes();
    let validation_report = create_test_validation_report();

    let analysis = assistant.process_validation_errors(&validation_report, &store, &shapes)?;

    // Verify the full pipeline works
    assert!(!matches!(
        analysis.error_classification,
        ErrorType::Other(_)
    ));
    assert!(!analysis.root_cause_analysis.is_empty());
    assert!(!analysis.fix_suggestions.is_empty());
    assert!(analysis.confidence_score > 0.0);

    // Verify the analysis provides actionable insights
    let has_actionable_suggestions = analysis
        .fix_suggestions
        .iter()
        .any(|suggestion| suggestion.automated || suggestion.confidence > 0.7);
    assert!(has_actionable_suggestions);

    Ok(())
}

#[test]
fn test_error_classification_patterns() -> Result<()> {
    let handler = IntelligentErrorHandler::new();
    let store = create_test_store();
    let shapes = create_test_shapes();

    // Create different types of violations to test classification
    let mut cardinality_report = ValidationReport::new();
    cardinality_report.conforms = false;
    cardinality_report.violations.push(
        ValidationViolation::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::from("http://example.org/TestShape"),
            ConstraintComponentId::from("MinCount"),
            Severity::Violation,
        )
        .with_message("Cardinality violation: minimum count not met".to_string()),
    );

    let cardinality_analysis =
        handler.process_validation_errors(&cardinality_report, &store, &shapes)?;
    assert!(matches!(
        cardinality_analysis.error_classification,
        ErrorType::CardinalityError
    ));

    // Test datatype classification
    let mut datatype_report = ValidationReport::new();
    datatype_report.conforms = false;
    datatype_report.violations.push(
        ValidationViolation::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::from("http://example.org/TestShape"),
            ConstraintComponentId::from("Datatype"),
            Severity::Violation,
        )
        .with_message("Datatype mismatch: expected integer".to_string()),
    );

    let datatype_analysis = handler.process_validation_errors(&datatype_report, &store, &shapes)?;
    assert!(matches!(
        datatype_analysis.error_classification,
        ErrorType::DataTypeError
    ));

    Ok(())
}
