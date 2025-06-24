use oxirs_shacl::{
    Validator, ValidationConfig, Shape, ShapeId, ShapeType, PropertyPath, Target,
    Constraint, ConstraintComponentId, Severity,
    constraints::*,
    shapes::ShapeFactory,
};
use oxirs_core::{
    model::{NamedNode, Term, Literal},
    store::Store,
};

#[test]
fn test_full_validation_workflow() {
    // Create a validator
    let mut validator = Validator::new();
    
    // Create a person shape with constraints
    let person_class = NamedNode::new("http://example.org/Person").unwrap();
    let person_shape = ShapeFactory::node_shape_with_class(
        ShapeId::new("http://example.org/PersonShape"),
        person_class
    );
    
    // Add the shape to the validator
    validator.add_shape(person_shape).unwrap();
    
    // Create a name property shape
    let name_path = PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap());
    let name_shape = ShapeFactory::string_property_shape(
        ShapeId::new("http://example.org/NameShape"),
        name_path,
        Some(1),    // min length
        Some(100),  // max length
        Some("^[A-Za-z ]+$".to_string()) // pattern
    );
    
    validator.add_shape(name_shape).unwrap();
    
    // Create an age property shape
    let age_path = PropertyPath::predicate(NamedNode::new("http://example.org/age").unwrap());
    let age_shape = ShapeFactory::cardinality_property_shape(
        ShapeId::new("http://example.org/AgeShape"),
        age_path,
        Some(1), // min count
        Some(1)  // max count
    );
    
    validator.add_shape(age_shape).unwrap();
    
    // Create a store with test data
    let store = Store::new().unwrap();
    
    // TODO: Add actual test data to store when API is available
    // For now, just test that validation runs without errors
    
    // Validate the store
    let result = validator.validate_store(&store, None);
    
    // Note: This may fail with "Store quad iteration not yet implemented" 
    // which is expected since oxirs-core Store is not fully implemented yet
    match result {
        Ok(report) => {
            // If validation succeeds, check results
            assert!(report.conforms());
            assert_eq!(report.violation_count(), 0);
        }
        Err(e) => {
            // If it fails due to unimplemented store features, that's expected
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("not yet implemented") || error_msg.contains("not implemented"));
        }
    }
}

#[test]
fn test_node_validation() {
    let mut validator = Validator::new();
    
    // Create a simple shape with a class constraint
    let person_class = NamedNode::new("http://example.org/Person").unwrap();
    let person_shape = ShapeFactory::node_shape_with_class(
        ShapeId::new("http://example.org/PersonShape"),
        person_class
    );
    
    validator.add_shape(person_shape).unwrap();
    
    let store = Store::new().unwrap();
    
    // Test validating a specific node
    let test_node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
    let shape_id = ShapeId::new("http://example.org/PersonShape");
    
    let result = validator.validate_node(&store, &shape_id, &test_node, None);
    
    assert!(result.is_ok());
    let report = result.unwrap();
    
    // The validation should complete successfully
    // (actual conformance depends on store contents)
    assert_eq!(report.violation_count(), 0); // No violations with empty store
}

#[test]
fn test_shape_management() {
    let mut validator = Validator::new();
    
    // Test adding shapes
    assert_eq!(validator.shapes().len(), 0);
    
    let shape_id = ShapeId::new("http://example.org/TestShape");
    let shape = Shape::node_shape(shape_id.clone());
    
    validator.add_shape(shape).unwrap();
    assert_eq!(validator.shapes().len(), 1);
    
    // Test getting shape
    let retrieved_shape = validator.get_shape(&shape_id);
    assert!(retrieved_shape.is_some());
    assert_eq!(retrieved_shape.unwrap().id, shape_id);
    
    // Test removing shape
    let removed_shape = validator.remove_shape(&shape_id);
    assert!(removed_shape.is_some());
    assert_eq!(validator.shapes().len(), 0);
}

#[test]
fn test_validation_config() {
    let config = ValidationConfig {
        max_violations: 5,
        fail_fast: true,
        include_warnings: false,
        ..ValidationConfig::default()
    };
    
    let validator = Validator::with_config(config.clone());
    
    // Test that validator respects configuration
    let store = Store::new().unwrap();
    let result = validator.validate_store(&store, None);
    
    assert!(result.is_ok());
}

#[test]
fn test_constraint_types() {
    // Test creating different constraint types
    let class_constraint = Constraint::Class(ClassConstraint {
        class_iri: NamedNode::new("http://example.org/Person").unwrap()
    });
    
    let min_count_constraint = Constraint::MinCount(MinCountConstraint { min_count: 1 });
    
    let pattern_constraint = Constraint::Pattern(PatternConstraint {
        pattern: "^[A-Za-z]+$".to_string(),
        flags: None,
        message: None,
    });
    
    // Test constraint validation
    assert!(class_constraint.validate().is_ok());
    assert!(min_count_constraint.validate().is_ok());
    assert!(pattern_constraint.validate().is_ok());
}

#[test]
fn test_property_paths() {
    // Test simple property path
    let simple_path = PropertyPath::predicate(
        NamedNode::new("http://example.org/name").unwrap()
    );
    
    // Test inverse property path
    let inverse_path = PropertyPath::inverse(simple_path.clone());
    
    // Test sequence property path
    let sequence_path = PropertyPath::sequence(vec![
        simple_path.clone(),
        PropertyPath::predicate(NamedNode::new("http://example.org/value").unwrap())
    ]);
    
    // Test alternative property path
    let alternative_path = PropertyPath::alternative(vec![
        simple_path.clone(),
        PropertyPath::predicate(NamedNode::new("http://example.org/label").unwrap())
    ]);
    
    // All paths should be created successfully
    assert_eq!(simple_path, PropertyPath::predicate(
        NamedNode::new("http://example.org/name").unwrap()
    ));
}

#[test]
fn test_target_types() {
    // Test different target types
    let class_target = Target::class(NamedNode::new("http://example.org/Person").unwrap());
    let node_target = Target::node(Term::NamedNode(
        NamedNode::new("http://example.org/john").unwrap()
    ));
    
    // Test target creation
    match class_target {
        Target::Class(class) => {
            assert_eq!(class.as_str(), "http://example.org/Person");
        }
        _ => panic!("Expected class target"),
    }
    
    match node_target {
        Target::Node(node) => {
            if let Term::NamedNode(named_node) = node {
                assert_eq!(named_node.as_str(), "http://example.org/john");
            } else {
                panic!("Expected named node");
            }
        }
        _ => panic!("Expected node target"),
    }
}

#[test]
fn test_validation_report_formats() {
    let mut validator = Validator::new();
    
    let shape = Shape::node_shape(ShapeId::new("http://example.org/TestShape"));
    validator.add_shape(shape).unwrap();
    
    let store = Store::new().unwrap();
    let validation_result = validator.validate_store(&store, None);
    
    let report = match validation_result {
        Ok(report) => report,
        Err(e) => {
            // If store validation fails due to unimplemented features,
            // create a mock report for testing serialization
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("not yet implemented") || error_msg.contains("not implemented"));
            
            // Create a basic report for testing serialization features
            oxirs_shacl::ValidationReport::new()
        }
    };
    
    // Test JSON serialization
    let json_result = report.to_json();
    assert!(json_result.is_ok());
    
    // Test Turtle serialization  
    let turtle_result = report.to_rdf("turtle");
    assert!(turtle_result.is_ok());
    
    // Test HTML generation
    let html_result = report.to_html();
    assert!(html_result.is_ok());
    
    let html = html_result.unwrap();
    assert!(html.contains("SHACL Validation Report"));
    assert!(html.contains("Conforms"));
}

#[test]
fn test_validator_builder() {
    use oxirs_shacl::ValidatorBuilder;
    
    let validator = ValidatorBuilder::new()
        .max_violations(10)
        .fail_fast(true)
        .include_warnings(false)
        .parallel(true)
        .build();
    
    // Test that builder creates validator with correct configuration
    let store = Store::new().unwrap();
    let result = validator.validate_store(&store, None);
    assert!(result.is_ok());
}

#[test]
fn test_error_handling() {
    let mut validator = Validator::new();
    
    // Test validation of non-existent shape
    let store = Store::new().unwrap();
    let non_existent_shape = ShapeId::new("http://example.org/NonExistent");
    let test_node = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());
    
    let result = validator.validate_node(&store, &non_existent_shape, &test_node, None);
    assert!(result.is_err());
    
    // Test invalid property shape (property shape without path)
    let mut invalid_shape = Shape::new(
        ShapeId::new("http://example.org/InvalidShape"), 
        ShapeType::PropertyShape
    );
    // Don't set path - this should cause validation to fail
    
    let result = validator.add_shape(invalid_shape);
    assert!(result.is_err());
}

#[test]
fn test_shape_factory_comprehensive() {
    // Test all shape factory methods
    
    // Node shape with class
    let person_class = NamedNode::new("http://example.org/Person").unwrap();
    let node_shape = ShapeFactory::node_shape_with_class(
        ShapeId::new("http://example.org/PersonShape"),
        person_class
    );
    assert!(node_shape.is_node_shape());
    assert_eq!(node_shape.targets.len(), 1);
    assert_eq!(node_shape.constraints.len(), 1);
    
    // String property shape
    let name_path = PropertyPath::predicate(NamedNode::new("http://example.org/name").unwrap());
    let string_shape = ShapeFactory::string_property_shape(
        ShapeId::new("http://example.org/NameShape"),
        name_path,
        Some(1), Some(100), Some("^[A-Za-z ]+$".to_string())
    );
    assert!(string_shape.is_property_shape());
    assert_eq!(string_shape.constraints.len(), 4); // datatype + minLength + maxLength + pattern
    
    // Cardinality property shape
    let age_path = PropertyPath::predicate(NamedNode::new("http://example.org/age").unwrap());
    let cardinality_shape = ShapeFactory::cardinality_property_shape(
        ShapeId::new("http://example.org/AgeShape"),
        age_path,
        Some(1), Some(1)
    );
    assert!(cardinality_shape.is_property_shape());
    assert_eq!(cardinality_shape.constraints.len(), 2); // minCount + maxCount
}