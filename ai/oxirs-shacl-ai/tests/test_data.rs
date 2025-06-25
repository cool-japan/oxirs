//! Test data utilities for oxirs-shacl-ai integration tests
//!
//! This module provides common test data, fixtures, and utilities
//! used across various integration tests for the AI-powered SHACL system.

use oxirs_core::model::{Literal, NamedNode, Term, Triple};
use oxirs_core::store::Store;
use oxirs_shacl_ai::*;
use std::collections::HashMap;

/// Sample RDF graph for testing
pub fn create_sample_graph() -> Store {
    let mut store = Store::new();

    // Add some sample triples
    let person = NamedNode::new("http://example.org/Person").unwrap();
    let name = NamedNode::new("http://example.org/name").unwrap();
    let age = NamedNode::new("http://example.org/age").unwrap();
    let email = NamedNode::new("http://example.org/email").unwrap();

    let john = NamedNode::new("http://example.org/john").unwrap();
    let jane = NamedNode::new("http://example.org/jane").unwrap();
    let bob = NamedNode::new("http://example.org/bob").unwrap();

    // John's data
    store.insert(Triple::new(
        john.clone().into(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .unwrap()
            .into(),
        person.clone().into(),
    ));
    store.insert(Triple::new(
        john.clone().into(),
        name.clone().into(),
        Literal::new_simple_literal("John Doe").into(),
    ));
    store.insert(Triple::new(
        john.clone().into(),
        age.clone().into(),
        Literal::new_typed_literal("30", oxirs_core::vocab::xsd::INTEGER).into(),
    ));
    store.insert(Triple::new(
        john.clone().into(),
        email.clone().into(),
        Literal::new_simple_literal("john@example.org").into(),
    ));

    // Jane's data
    store.insert(Triple::new(
        jane.clone().into(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .unwrap()
            .into(),
        person.clone().into(),
    ));
    store.insert(Triple::new(
        jane.clone().into(),
        name.clone().into(),
        Literal::new_simple_literal("Jane Smith").into(),
    ));
    store.insert(Triple::new(
        jane.clone().into(),
        age.clone().into(),
        Literal::new_typed_literal("25", oxirs_core::vocab::xsd::INTEGER).into(),
    ));
    store.insert(Triple::new(
        jane.clone().into(),
        email.clone().into(),
        Literal::new_simple_literal("jane@example.org").into(),
    ));

    // Bob's data (incomplete for testing)
    store.insert(Triple::new(
        bob.clone().into(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .unwrap()
            .into(),
        person.clone().into(),
    ));
    store.insert(Triple::new(
        bob.clone().into(),
        name.clone().into(),
        Literal::new_simple_literal("Bob Johnson").into(),
    ));
    // Note: Bob is missing age and email for testing validation

    store
}

/// Create a sample SHACL shape for testing
pub fn create_person_shape() -> Shape {
    use oxirs_shacl_ai::shape::*;

    let mut shape = Shape::new("http://example.org/PersonShape".to_string());
    shape.set_target_class("http://example.org/Person".to_string());

    // Add property constraints
    let mut name_constraint = PropertyConstraint::new("http://example.org/name".to_string());
    name_constraint.min_count = Some(1);
    name_constraint.max_count = Some(1);
    name_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#string".to_string());
    shape.add_property_constraint(name_constraint);

    let mut age_constraint = PropertyConstraint::new("http://example.org/age".to_string());
    age_constraint.min_count = Some(1);
    age_constraint.max_count = Some(1);
    age_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#integer".to_string());
    age_constraint.min_inclusive = Some(0.0);
    age_constraint.max_inclusive = Some(150.0);
    shape.add_property_constraint(age_constraint);

    let mut email_constraint = PropertyConstraint::new("http://example.org/email".to_string());
    email_constraint.min_count = Some(1);
    email_constraint.max_count = Some(1);
    email_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#string".to_string());
    email_constraint.pattern = Some(r"^[^\s@]+@[^\s@]+\.[^\s@]+$".to_string());
    shape.add_property_constraint(email_constraint);

    shape
}

/// Create complex test graph with various patterns
pub fn create_complex_graph() -> Store {
    let mut store = Store::new();

    // Organizations
    let org = NamedNode::new("http://example.org/Organization").unwrap();
    let org_name = NamedNode::new("http://example.org/orgName").unwrap();
    let founded = NamedNode::new("http://example.org/founded").unwrap();
    let employee_count = NamedNode::new("http://example.org/employeeCount").unwrap();

    let acme = NamedNode::new("http://example.org/acme").unwrap();
    store.insert(Triple::new(
        acme.clone().into(),
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .unwrap()
            .into(),
        org.clone().into(),
    ));
    store.insert(Triple::new(
        acme.clone().into(),
        org_name.clone().into(),
        Literal::new_simple_literal("ACME Corporation").into(),
    ));
    store.insert(Triple::new(
        acme.clone().into(),
        founded.clone().into(),
        Literal::new_typed_literal("1990", oxirs_core::vocab::xsd::INTEGER).into(),
    ));
    store.insert(Triple::new(
        acme.clone().into(),
        employee_count.clone().into(),
        Literal::new_typed_literal("1000", oxirs_core::vocab::xsd::INTEGER).into(),
    ));

    // Add employees with relationships
    let works_for = NamedNode::new("http://example.org/worksFor").unwrap();
    let person = NamedNode::new("http://example.org/Person").unwrap();
    let name = NamedNode::new("http://example.org/name").unwrap();

    for i in 1..=20 {
        let employee = NamedNode::new(&format!("http://example.org/employee{}", i)).unwrap();
        store.insert(Triple::new(
            employee.clone().into(),
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                .unwrap()
                .into(),
            person.clone().into(),
        ));
        store.insert(Triple::new(
            employee.clone().into(),
            name.clone().into(),
            Literal::new_simple_literal(&format!("Employee {}", i)).into(),
        ));
        store.insert(Triple::new(
            employee.clone().into(),
            works_for.clone().into(),
            acme.clone().into(),
        ));
    }

    store
}

/// Create sample prediction features for testing
pub fn create_sample_features() -> HashMap<String, f64> {
    let mut features = HashMap::new();
    features.insert("graph_size".to_string(), 1000.0);
    features.insert("processing_complexity".to_string(), 75.0);
    features.insert("memory_estimate".to_string(), 250.0);
    features.insert("cpu_complexity".to_string(), 60.0);
    features.insert("unique_predicates".to_string(), 15.0);
    features.insert("unique_classes".to_string(), 5.0);
    features.insert("success_rate".to_string(), 0.85);
    features.insert("constraint_count".to_string(), 20.0);
    features.insert("node_shape_ratio".to_string(), 0.1);
    features.insert("property_path_complexity".to_string(), 2.5);
    features
}

/// Create validation config for testing
pub fn create_test_validation_config() -> ValidationConfig {
    ValidationConfig {
        strict_mode: false,
        max_violations: Some(100),
        early_termination: true,
        parallel_validation: true,
        cache_results: true,
        report_details: true,
        validate_closed_shapes: false,
        severity_threshold: ValidationSeverity::Info,
        custom_functions: HashMap::new(),
        timeout_seconds: Some(30),
    }
}

/// Create prediction config for testing
pub fn create_test_prediction_config() -> PredictionConfig {
    PredictionConfig {
        enable_prediction: true,
        enable_performance_prediction: true,
        enable_error_anticipation: true,
        enable_bottleneck_prediction: true,
        min_confidence_threshold: 0.6,
        max_cache_size: 500,
        cache_ttl_minutes: 15,
        prediction_horizon_minutes: 30,
        learning_enabled: true,
        feedback_integration: true,
        model_update_frequency: ModelUpdateFrequency::AfterValidation,
        advanced_features: AdvancedPredictionFeatures {
            ensemble_prediction: true,
            uncertainty_quantification: true,
            feature_importance_analysis: true,
            adaptive_thresholds: true,
        },
    }
}

/// Create analytics config for testing
pub fn create_test_analytics_config() -> AnalyticsConfig {
    AnalyticsConfig {
        enable_analytics: true,
        track_patterns: true,
        track_performance: true,
        track_errors: true,
        generate_insights: true,
        store_historical_data: true,
        data_retention_days: 90,
        aggregation_window_minutes: 60,
        anomaly_detection: true,
        export_metrics: true,
    }
}

/// Helper function to create test shapes with various constraint types
pub fn create_test_shapes() -> Vec<Shape> {
    vec![
        create_person_shape(),
        create_organization_shape(),
        create_product_shape(),
    ]
}

/// Create organization shape for testing
pub fn create_organization_shape() -> Shape {
    use oxirs_shacl_ai::shape::*;

    let mut shape = Shape::new("http://example.org/OrganizationShape".to_string());
    shape.set_target_class("http://example.org/Organization".to_string());

    // Organization name constraint
    let mut name_constraint = PropertyConstraint::new("http://example.org/orgName".to_string());
    name_constraint.min_count = Some(1);
    name_constraint.max_count = Some(1);
    name_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#string".to_string());
    name_constraint.min_length = Some(1);
    name_constraint.max_length = Some(100);
    shape.add_property_constraint(name_constraint);

    // Founded year constraint
    let mut founded_constraint = PropertyConstraint::new("http://example.org/founded".to_string());
    founded_constraint.min_count = Some(1);
    founded_constraint.max_count = Some(1);
    founded_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#integer".to_string());
    founded_constraint.min_inclusive = Some(1800.0);
    founded_constraint.max_inclusive = Some(2030.0);
    shape.add_property_constraint(founded_constraint);

    shape
}

/// Create product shape for testing
pub fn create_product_shape() -> Shape {
    use oxirs_shacl_ai::shape::*;

    let mut shape = Shape::new("http://example.org/ProductShape".to_string());
    shape.set_target_class("http://example.org/Product".to_string());

    // Product name constraint
    let mut name_constraint = PropertyConstraint::new("http://example.org/productName".to_string());
    name_constraint.min_count = Some(1);
    name_constraint.max_count = Some(1);
    name_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#string".to_string());
    shape.add_property_constraint(name_constraint);

    // Price constraint
    let mut price_constraint = PropertyConstraint::new("http://example.org/price".to_string());
    price_constraint.min_count = Some(1);
    price_constraint.max_count = Some(1);
    price_constraint.datatype = Some("http://www.w3.org/2001/XMLSchema#decimal".to_string());
    price_constraint.min_inclusive = Some(0.0);
    shape.add_property_constraint(price_constraint);

    shape
}

/// Mock validation outcome for testing
pub fn create_mock_validation_outcome(
    conformant: bool,
    violation_count: usize,
) -> ValidationOutcome {
    ValidationOutcome {
        is_conformant: conformant,
        violation_count,
        warning_count: if conformant { 0 } else { violation_count / 2 },
        processing_time: std::time::Duration::from_millis(100),
        validated_nodes: 50,
        shapes_evaluated: 3,
        constraints_checked: 15,
        errors: if conformant {
            Vec::new()
        } else {
            create_mock_validation_errors(violation_count)
        },
        metadata: ValidationMetadata {
            timestamp: chrono::Utc::now(),
            validator_version: "1.0.0".to_string(),
            configuration_hash: "abc123".to_string(),
        },
    }
}

/// Create mock validation errors for testing
pub fn create_mock_validation_errors(count: usize) -> Vec<ValidationError> {
    (0..count)
        .map(|i| ValidationError {
            message: format!("Mock validation error {}", i),
            severity: if i % 3 == 0 {
                ValidationSeverity::Error
            } else {
                ValidationSeverity::Warning
            },
            focus_node: Some(format!("http://example.org/node{}", i)),
            result_path: Some(format!("http://example.org/property{}", i % 3)),
            constraint_component: format!(
                "http://www.w3.org/ns/shacl#{}Constraint",
                if i % 2 == 0 { "MinCount" } else { "Datatype" }
            ),
            source_shape: format!("http://example.org/Shape{}", i % 2),
            detail: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_graph_creation() {
        let store = create_sample_graph();
        // Basic smoke test - ensure the store is not empty
        assert!(!store.is_empty());
    }

    #[test]
    fn test_complex_graph_creation() {
        let store = create_complex_graph();
        assert!(!store.is_empty());
    }

    #[test]
    fn test_sample_features() {
        let features = create_sample_features();
        assert!(features.contains_key("graph_size"));
        assert!(features.contains_key("processing_complexity"));
        assert_eq!(features.len(), 10);
    }

    #[test]
    fn test_shape_creation() {
        let shapes = create_test_shapes();
        assert_eq!(shapes.len(), 3);

        let person_shape = &shapes[0];
        assert!(!person_shape.get_constraints().is_empty());
    }
}
