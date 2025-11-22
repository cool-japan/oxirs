//! Integration tests for the interactive constraint tester
//!
//! Tests the core functionality of the constraint testing tool

use std::fs;
use std::path::PathBuf;

use oxirs_core::NamedNode;
use oxirs_shacl::{constraints::Constraint, ConstraintComponentId, Shape, ShapeId, ShapeType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Testing session structure (mirrored from bin/constraint_tester.rs)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestingSession {
    shapes: HashMap<String, Shape>,
    test_data: Vec<String>,
    history: Vec<String>,
    name: String,
    last_results: Option<String>,
}

impl TestingSession {
    fn new(name: String) -> Self {
        Self {
            shapes: HashMap::new(),
            test_data: Vec::new(),
            history: Vec::new(),
            name,
            last_results: None,
        }
    }

    fn save(&self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load(path: &PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let session: TestingSession = serde_json::from_str(&json)?;
        Ok(session)
    }
}

#[test]
fn test_session_creation() {
    let session = TestingSession::new("test-session".to_string());

    assert_eq!(session.name, "test-session");
    assert!(session.shapes.is_empty());
    assert!(session.test_data.is_empty());
    assert!(session.history.is_empty());
    assert!(session.last_results.is_none());
}

#[test]
fn test_session_save_and_load() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir();
    let session_file = temp_dir.join("test_session.json");

    // Create a session with data
    let mut session = TestingSession::new("save-test".to_string());
    session.history.push("create person node".to_string());
    session.history.push("add person minCount 1".to_string());

    // Add a shape
    let shape_id = ShapeId::new("http://example.org/shapes#Person");
    let mut shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

    use oxirs_shacl::constraints::cardinality_constraints::MinCountConstraint;
    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    session.shapes.insert("person".to_string(), shape);

    // Save session
    session.save(&session_file)?;

    // Load session
    let loaded_session = TestingSession::load(&session_file)?;

    // Verify
    assert_eq!(loaded_session.name, "save-test");
    assert_eq!(loaded_session.history.len(), 2);
    assert_eq!(loaded_session.shapes.len(), 1);
    assert!(loaded_session.shapes.contains_key("person"));

    // Cleanup
    fs::remove_file(session_file)?;

    Ok(())
}

#[test]
fn test_shape_creation_workflow() {
    let mut session = TestingSession::new("workflow-test".to_string());

    // Create a node shape
    let shape_id = ShapeId::new("http://example.org/shapes#EmailShape");
    let mut shape = Shape::new(shape_id.clone(), ShapeType::NodeShape);

    // Add constraints
    use oxirs_shacl::constraints::cardinality_constraints::{
        MaxCountConstraint, MinCountConstraint,
    };
    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    shape.add_constraint(
        ConstraintComponentId::new("sh:maxCount"),
        Constraint::MaxCount(MaxCountConstraint { max_count: 1 }),
    );

    use oxirs_shacl::constraints::string_constraints::PatternConstraint;
    shape.add_constraint(
        ConstraintComponentId::new("sh:pattern"),
        Constraint::Pattern(PatternConstraint {
            pattern: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
            flags: None,
            message: None,
        }),
    );

    session.shapes.insert("email".to_string(), shape);

    // Verify shape
    let email_shape = session.shapes.get("email").unwrap();
    assert_eq!(email_shape.constraints.len(), 3);
    assert!(email_shape.is_node_shape());
}

#[test]
fn test_constraint_types() {
    let shape_id = ShapeId::new("http://example.org/shapes#TestShape");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    // Test different constraint types
    use oxirs_shacl::constraints::cardinality_constraints::{
        MaxCountConstraint, MinCountConstraint,
    };
    use oxirs_shacl::constraints::string_constraints::{MaxLengthConstraint, MinLengthConstraint};
    use oxirs_shacl::constraints::value_constraints::{ClassConstraint, DatatypeConstraint};

    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    shape.add_constraint(
        ConstraintComponentId::new("sh:maxCount"),
        Constraint::MaxCount(MaxCountConstraint { max_count: 10 }),
    );

    shape.add_constraint(
        ConstraintComponentId::new("sh:minLength"),
        Constraint::MinLength(MinLengthConstraint { min_length: 5 }),
    );

    shape.add_constraint(
        ConstraintComponentId::new("sh:maxLength"),
        Constraint::MaxLength(MaxLengthConstraint { max_length: 100 }),
    );

    let datatype = NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap();
    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: datatype.clone(),
        }),
    );

    let class_node = NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap();
    shape.add_constraint(
        ConstraintComponentId::new("sh:class"),
        Constraint::Class(ClassConstraint {
            class_iri: class_node,
        }),
    );

    assert_eq!(shape.constraints.len(), 6);
}

#[test]
fn test_property_shape_creation() {
    use oxirs_shacl::paths::PropertyPath;

    let shape_id = ShapeId::new("http://example.org/shapes#NameShape");
    let property_path =
        PropertyPath::predicate(NamedNode::new("http://xmlns.com/foaf/0.1/name").unwrap());

    let mut shape = Shape::property_shape(shape_id, property_path);

    use oxirs_shacl::constraints::cardinality_constraints::MinCountConstraint;
    use oxirs_shacl::constraints::value_constraints::DatatypeConstraint;

    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
        }),
    );

    assert!(shape.is_property_shape());
    assert!(shape.path.is_some());
    assert_eq!(shape.constraints.len(), 2);
}

#[test]
fn test_session_history_tracking() {
    let mut session = TestingSession::new("history-test".to_string());

    // Simulate command history
    let commands = vec![
        "create person node",
        "add person minCount 1",
        "add person class http://xmlns.com/foaf/0.1/Person",
        "test person",
        "validate",
    ];

    for cmd in commands {
        session.history.push(cmd.to_string());
    }

    assert_eq!(session.history.len(), 5);
    assert_eq!(session.history[0], "create person node");
    assert_eq!(session.history[4], "validate");
}

#[test]
fn test_multiple_shapes_in_session() {
    let mut session = TestingSession::new("multi-shape-test".to_string());

    // Create multiple shapes
    for (name, shape_type) in [
        ("person", ShapeType::NodeShape),
        ("email", ShapeType::PropertyShape),
    ] {
        let shape_id = ShapeId::new(format!("http://example.org/shapes#{}", name));
        let shape = Shape::new(shape_id, shape_type);
        session.shapes.insert(name.to_string(), shape);
    }

    assert_eq!(session.shapes.len(), 2);
    assert!(session.shapes.contains_key("person"));
    assert!(session.shapes.contains_key("email"));
}

#[test]
fn test_constraint_pattern_validation() {
    use oxirs_shacl::constraints::string_constraints::PatternConstraint;

    // Test various regex patterns
    let patterns = vec![
        (r"^[A-Z].*", "Starts with uppercase"),
        (r"^\d{3}-\d{4}$", "Phone number format"),
        (r"^[^@]+@[^@]+\.[^@]+$", "Email format"),
        (r"^https?://.*", "URL format"),
    ];

    for (pattern, _description) in patterns {
        let constraint = Constraint::Pattern(PatternConstraint {
            pattern: pattern.to_string(),
            flags: None,
            message: None,
        });

        // Verify constraint can be created
        match constraint {
            Constraint::Pattern(p) => {
                assert_eq!(p.pattern, pattern);
            }
            _ => panic!("Expected Pattern constraint"),
        }
    }
}

#[test]
fn test_shape_creation_integration() {
    // Create a basic person shape
    let shape_id = ShapeId::new("http://example.org/shapes#Person");
    let _shape = Shape::new(shape_id, ShapeType::NodeShape);

    // Shape creation should work with tester
    assert!(_shape.is_node_shape(), "Shape creation works");
}

#[test]
fn test_session_serialization() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = TestingSession::new("serialize-test".to_string());

    // Add complex data
    let shape_id = ShapeId::new("http://example.org/shapes#Complex");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    use oxirs_shacl::constraints::cardinality_constraints::MinCountConstraint;
    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    session.shapes.insert("complex".to_string(), shape);
    session.history.push("test command".to_string());

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&session)?;

    // Deserialize
    let deserialized: TestingSession = serde_json::from_str(&json)?;

    assert_eq!(deserialized.name, "serialize-test");
    assert_eq!(deserialized.shapes.len(), 1);
    assert_eq!(deserialized.history.len(), 1);

    Ok(())
}

#[test]
fn test_session_data_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir();
    let test_file = temp_dir.join("persistence_test.json");

    // Create and save session
    let mut session1 = TestingSession::new("persist-test".to_string());
    session1
        .test_data
        .push("@prefix ex: <http://example.org/> .".to_string());
    session1.test_data.push("ex:john a ex:Person .".to_string());
    session1.save(&test_file)?;

    // Load and verify
    let session2 = TestingSession::load(&test_file)?;
    assert_eq!(session2.test_data.len(), 2);
    assert_eq!(session2.test_data[0], "@prefix ex: <http://example.org/> .");

    // Cleanup
    fs::remove_file(test_file)?;

    Ok(())
}

#[test]
fn test_constraint_builder_workflow() {
    // Simulate building constraints through interactive commands
    let mut session = TestingSession::new("builder-test".to_string());

    let shape_id = ShapeId::new("http://example.org/shapes#Product");
    let mut shape = Shape::new(shape_id, ShapeType::NodeShape);

    // Build a product validation shape
    use oxirs_shacl::constraints::cardinality_constraints::MinCountConstraint;
    use oxirs_shacl::constraints::value_constraints::DatatypeConstraint;

    shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    let datatype_decimal = NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").unwrap();
    shape.add_constraint(
        ConstraintComponentId::new("sh:datatype"),
        Constraint::Datatype(DatatypeConstraint {
            datatype_iri: datatype_decimal,
        }),
    );

    session.shapes.insert("product_price".to_string(), shape);

    assert!(session.shapes.contains_key("product_price"));
    let product_shape = session.shapes.get("product_price").unwrap();
    assert_eq!(product_shape.constraints.len(), 2);
}
