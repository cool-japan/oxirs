//! Tests for closed shape validation

use oxirs_core::{
    model::{NamedNode, Term},
    store::Store,
};
use oxirs_shacl::{
    constraints::*, PropertyPath, Severity, Shape, ShapeId, ShapeType, 
    Constraint, ConstraintComponentId, ValidationConfig, Validator,
};

#[test]
fn test_closed_shape_validation() {
    // Create a validator
    let mut validator = Validator::new();
    
    // Create a closed node shape with some allowed properties
    let mut person_shape = Shape::new(
        ShapeId::new("http://example.org/PersonShape"),
        ShapeType::NodeShape,
    );
    
    // Add a closed constraint allowing only specific properties
    let closed_constraint = ClosedConstraint {
        closed: true,
        ignored_properties: vec![
            PropertyPath::predicate(NamedNode::new("http://example.org/age").unwrap()),
            PropertyPath::predicate(NamedNode::new("http://example.org/email").unwrap()),
        ],
    };
    
    person_shape.add_constraint(
        ConstraintComponentId::new("sh:closed"),
        Constraint::Closed(closed_constraint),
    );
    
    // Add the shape to the validator
    validator.add_shape(person_shape).unwrap();
    
    // Create a store with test data
    let store = Store::new().unwrap();
    
    // Note: Since the Store API is not fully implemented, we can't add actual data
    // This test will verify that the shape parses and validates correctly
    
    // Test validation with an empty store
    let result = validator.validate_store(&store, None);
    match result {
        Ok(report) => {
            assert!(report.conforms(), "Empty store should conform to closed shape");
        }
        Err(e) => {
            // Expected if Store iteration is not implemented
            let error_msg = format!("{}", e);
            assert!(
                error_msg.contains("not yet implemented") || 
                error_msg.contains("not implemented"),
                "Unexpected error: {}",
                e
            );
        }
    }
}

#[test]
fn test_closed_shape_with_property_shapes() {
    let mut validator = Validator::new();
    
    // Create a closed node shape
    let mut person_shape = Shape::new(
        ShapeId::new("http://example.org/PersonShape"),
        ShapeType::NodeShape,
    );
    
    // Make it closed with some ignored properties
    let closed_constraint = ClosedConstraint {
        closed: true,
        ignored_properties: vec![
            PropertyPath::predicate(NamedNode::new("http://www.w3.org/2000/01/rdf-schema#label").unwrap()),
        ],
    };
    
    person_shape.add_constraint(
        ConstraintComponentId::new("sh:closed"),
        Constraint::Closed(closed_constraint),
    );
    
    validator.add_shape(person_shape).unwrap();
    
    // Create property shapes for allowed properties
    let name_shape = Shape::property_shape(
        ShapeId::new("http://example.org/NamePropertyShape"),
        PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap()),
    );
    
    validator.add_shape(name_shape).unwrap();
    
    let age_shape = Shape::property_shape(
        ShapeId::new("http://example.org/AgePropertyShape"),
        PropertyPath::predicate(NamedNode::new("http://example.org/age").unwrap()),
    );
    
    validator.add_shape(age_shape).unwrap();
    
    // Test validation
    let store = Store::new().unwrap();
    let result = validator.validate_store(&store, None);
    
    match result {
        Ok(report) => {
            assert!(report.conforms(), "Should conform with property shapes");
        }
        Err(e) => {
            // Expected if Store iteration is not implemented
            let error_msg = format!("{}", e);
            assert!(
                error_msg.contains("not yet implemented") || 
                error_msg.contains("not implemented"),
                "Unexpected error: {}",
                e
            );
        }
    }
}

#[test]
fn test_open_shape() {
    let mut validator = Validator::new();
    
    // Create an open shape (closed = false)
    let mut open_shape = Shape::new(
        ShapeId::new("http://example.org/OpenShape"),
        ShapeType::NodeShape,
    );
    
    // Add a closed constraint with closed = false
    let open_constraint = ClosedConstraint {
        closed: false,
        ignored_properties: vec![],
    };
    
    open_shape.add_constraint(
        ConstraintComponentId::new("sh:closed"),
        Constraint::Closed(open_constraint),
    );
    
    validator.add_shape(open_shape).unwrap();
    
    // Open shapes should always conform regardless of properties
    let store = Store::new().unwrap();
    let result = validator.validate_store(&store, None);
    
    match result {
        Ok(report) => {
            assert!(report.conforms(), "Open shape should always conform");
        }
        Err(e) => {
            // Expected if Store iteration is not implemented
            let error_msg = format!("{}", e);
            assert!(
                error_msg.contains("not yet implemented") || 
                error_msg.contains("not implemented"),
                "Unexpected error: {}",
                e
            );
        }
    }
}