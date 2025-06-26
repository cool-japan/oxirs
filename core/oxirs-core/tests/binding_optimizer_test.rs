//! Tests for variable binding optimization

use oxirs_core::model::*;
use oxirs_core::query::{BindingSet, BindingOptimizer, BindingIterator, Constraint, TermType};
use oxirs_core::query::binding_optimizer::{
    TermBinding, BindingMetadata, ValueConstraintType, RelationType,
};
use std::collections::{HashMap, HashSet};

#[test]
fn test_binding_set_creation() {
    let mut bindings = BindingSet::new();
    
    // Add variables
    let var_person = Variable::new("person").unwrap();
    let var_name = Variable::new("name").unwrap();
    
    bindings.add_variable(var_person.clone());
    bindings.add_variable(var_name.clone());
    
    assert_eq!(bindings.variables.len(), 2);
    assert!(!bindings.is_bound(&var_person));
    assert!(!bindings.is_bound(&var_name));
}

#[test]
fn test_simple_binding() {
    let mut bindings = BindingSet::new();
    let var = Variable::new("x").unwrap();
    let term = Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    
    // Bind variable
    bindings.bind(
        var.clone(),
        term.clone(),
        BindingMetadata {
            source_pattern_id: 0,
            confidence: 1.0,
            is_fixed: false,
        },
    ).unwrap();
    
    // Check binding
    assert!(bindings.is_bound(&var));
    assert_eq!(bindings.get(&var), Some(&term));
}

#[test]
fn test_type_constraints() {
    let mut bindings = BindingSet::new();
    let var = Variable::new("value").unwrap();
    
    // Add constraint: value must be a literal
    let mut allowed_types = HashSet::new();
    allowed_types.insert(TermType::Literal);
    allowed_types.insert(TermType::NumericLiteral);
    allowed_types.insert(TermType::StringLiteral);
    
    bindings.add_constraint(Constraint::TypeConstraint {
        variable: var.clone(),
        allowed_types,
    });
    
    // Try to bind to a named node (should fail)
    let named_node = Term::NamedNode(NamedNode::new("http://example.org/resource").unwrap());
    let result = bindings.bind(var.clone(), named_node, BindingMetadata::default());
    assert!(result.is_err());
    
    // Try to bind to a literal (should succeed)
    let literal = Term::Literal(Literal::new("Hello"));
    let result = bindings.bind(var.clone(), literal, BindingMetadata::default());
    assert!(result.is_ok());
}

#[test]
fn test_numeric_range_constraint() {
    let mut bindings = BindingSet::new();
    let var = Variable::new("age").unwrap();
    
    // Add numeric range constraint: 0 <= age <= 120
    bindings.add_constraint(Constraint::ValueConstraint {
        variable: var.clone(),
        constraint: ValueConstraintType::NumericRange {
            min: 0.0,
            max: 120.0,
        },
    });
    
    // Valid ages
    for age in &["0", "25", "50", "100", "120"] {
        let term = Term::Literal(Literal::new(*age));
        assert!(
            bindings.bind(var.clone(), term, BindingMetadata::default()).is_ok(),
            "Age {} should be valid", age
        );
    }
    
    // Invalid ages
    for age in &["-10", "150", "1000"] {
        let term = Term::Literal(Literal::new(*age));
        assert!(
            bindings.bind(var.clone(), term, BindingMetadata::default()).is_err(),
            "Age {} should be invalid", age
        );
    }
}

#[test]
fn test_string_pattern_constraint() {
    let mut bindings = BindingSet::new();
    let var = Variable::new("email").unwrap();
    
    // Add email pattern constraint
    let email_regex = regex::Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
    bindings.add_constraint(Constraint::ValueConstraint {
        variable: var.clone(),
        constraint: ValueConstraintType::StringPattern(email_regex),
    });
    
    // Valid email
    let valid_email = Term::Literal(Literal::new("alice@example.com"));
    assert!(bindings.bind(var.clone(), valid_email, BindingMetadata::default()).is_ok());
    
    // Invalid email
    let invalid_email = Term::Literal(Literal::new("not-an-email"));
    assert!(bindings.bind(var.clone(), invalid_email, BindingMetadata::default()).is_err());
}

#[test]
fn test_binding_merge() {
    let mut bindings1 = BindingSet::new();
    let mut bindings2 = BindingSet::new();
    
    let var_x = Variable::new("x").unwrap();
    let var_y = Variable::new("y").unwrap();
    let var_z = Variable::new("z").unwrap();
    
    // Bind x in first set
    bindings1.bind(
        var_x.clone(),
        Term::NamedNode(NamedNode::new("http://example.org/x").unwrap()),
        BindingMetadata::default(),
    ).unwrap();
    
    // Bind y and z in second set
    bindings2.bind(
        var_y.clone(),
        Term::NamedNode(NamedNode::new("http://example.org/y").unwrap()),
        BindingMetadata::default(),
    ).unwrap();
    bindings2.bind(
        var_z.clone(),
        Term::NamedNode(NamedNode::new("http://example.org/z").unwrap()),
        BindingMetadata::default(),
    ).unwrap();
    
    // Merge
    bindings1.merge(&bindings2).unwrap();
    
    // Check all bindings exist
    assert!(bindings1.is_bound(&var_x));
    assert!(bindings1.is_bound(&var_y));
    assert!(bindings1.is_bound(&var_z));
    assert_eq!(bindings1.variables.len(), 3);
}

#[test]
fn test_binding_conflict_detection() {
    let mut bindings1 = BindingSet::new();
    let mut bindings2 = BindingSet::new();
    
    let var = Variable::new("x").unwrap();
    
    // Bind to different values
    bindings1.bind(
        var.clone(),
        Term::NamedNode(NamedNode::new("http://example.org/value1").unwrap()),
        BindingMetadata::default(),
    ).unwrap();
    
    bindings2.bind(
        var.clone(),
        Term::NamedNode(NamedNode::new("http://example.org/value2").unwrap()),
        BindingMetadata::default(),
    ).unwrap();
    
    // Merge should fail due to conflict
    assert!(bindings1.merge(&bindings2).is_err());
}

#[test]
fn test_unbound_variables() {
    let mut bindings = BindingSet::with_variables(vec![
        Variable::new("x").unwrap(),
        Variable::new("y").unwrap(),
        Variable::new("z").unwrap(),
    ]);
    
    // Initially all unbound
    assert_eq!(bindings.unbound_variables().len(), 3);
    
    // Bind x
    bindings.bind(
        Variable::new("x").unwrap(),
        Term::NamedNode(NamedNode::new("http://example.org/x").unwrap()),
        BindingMetadata::default(),
    ).unwrap();
    
    // Now 2 unbound
    let unbound = bindings.unbound_variables();
    assert_eq!(unbound.len(), 2);
    assert!(unbound.contains(&&Variable::new("y").unwrap()));
    assert!(unbound.contains(&&Variable::new("z").unwrap()));
}

#[test]
fn test_binding_optimizer_caching() {
    let mut optimizer = BindingOptimizer::new();
    
    let vars = vec![
        Variable::new("x").unwrap(),
        Variable::new("y").unwrap(),
    ];
    let constraints = vec![
        Constraint::TypeConstraint {
            variable: Variable::new("x").unwrap(),
            allowed_types: vec![TermType::NamedNode].into_iter().collect(),
        },
    ];
    
    // First call - cache miss
    let bindings1 = optimizer.optimize_bindings(vars.clone(), constraints.clone());
    
    // Second call - cache hit
    let bindings2 = optimizer.optimize_bindings(vars.clone(), constraints.clone());
    
    // Should be the same instance (Arc)
    assert!(std::sync::Arc::ptr_eq(&bindings1, &bindings2));
    
    // Check stats
    let stats = optimizer.stats();
    assert!(stats.contains("Cache hits: 1"));
    assert!(stats.contains("Cache misses: 1"));
}

#[test]
fn test_binding_iterator() {
    let base = vec![HashMap::new()];
    let vars = vec![Variable::new("x").unwrap(), Variable::new("y").unwrap()];
    
    let mut possible_values = HashMap::new();
    possible_values.insert(
        Variable::new("x").unwrap(),
        vec![
            Term::NamedNode(NamedNode::new("http://example.org/a").unwrap()),
            Term::NamedNode(NamedNode::new("http://example.org/b").unwrap()),
        ],
    );
    possible_values.insert(
        Variable::new("y").unwrap(),
        vec![
            Term::Literal(Literal::new("1")),
            Term::Literal(Literal::new("2")),
        ],
    );
    
    let mut iterator = BindingIterator::new(base, vars, possible_values, vec![]);
    
    // Should generate 4 combinations (2 x 2)
    let mut count = 0;
    while let Some(binding) = iterator.next_valid() {
        assert_eq!(binding.len(), 2);
        assert!(binding.contains_key(&Variable::new("x").unwrap()));
        assert!(binding.contains_key(&Variable::new("y").unwrap()));
        count += 1;
    }
    assert_eq!(count, 4);
}

#[test]
fn test_relationship_constraints() {
    let base = vec![HashMap::new()];
    let vars = vec![Variable::new("x").unwrap(), Variable::new("y").unwrap()];
    
    let mut possible_values = HashMap::new();
    possible_values.insert(
        Variable::new("x").unwrap(),
        vec![
            Term::Literal(Literal::new("5")),
            Term::Literal(Literal::new("10")),
            Term::Literal(Literal::new("15")),
        ],
    );
    possible_values.insert(
        Variable::new("y").unwrap(),
        vec![
            Term::Literal(Literal::new("8")),
            Term::Literal(Literal::new("12")),
        ],
    );
    
    // Add constraint: x < y
    let constraints = vec![
        Constraint::RelationshipConstraint {
            left: Variable::new("x").unwrap(),
            right: Variable::new("y").unwrap(),
            relation: RelationType::LessThan,
        },
    ];
    
    let mut iterator = BindingIterator::new(base, vars, possible_values, constraints);
    
    // Collect valid bindings
    let mut valid_bindings = Vec::new();
    while let Some(binding) = iterator.next_valid() {
        valid_bindings.push(binding);
    }
    
    // Should have: (5,8), (5,12), (10,12)
    assert_eq!(valid_bindings.len(), 3);
    
    for binding in &valid_bindings {
        let x_val = binding.get(&Variable::new("x").unwrap()).unwrap();
        let y_val = binding.get(&Variable::new("y").unwrap()).unwrap();
        
        if let (Term::Literal(x_lit), Term::Literal(y_lit)) = (x_val, y_val) {
            let x_num: f64 = x_lit.value().parse().unwrap();
            let y_num: f64 = y_lit.value().parse().unwrap();
            assert!(x_num < y_num);
        }
    }
}

#[test]
fn test_apply_bindings_to_pattern() {
    use oxirs_core::query::algebra::TermPattern;
    
    let mut bindings = BindingSet::new();
    let var = Variable::new("person").unwrap();
    let value = Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    
    bindings.bind(var.clone(), value.clone(), BindingMetadata::default()).unwrap();
    
    // Apply to variable pattern
    let var_pattern = TermPattern::Variable(var);
    let bound_pattern = bindings.apply_to_pattern(&var_pattern);
    
    match bound_pattern {
        TermPattern::NamedNode(n) => {
            assert_eq!(n.as_str(), "http://example.org/alice");
        }
        _ => panic!("Expected NamedNode pattern"),
    }
    
    // Apply to non-variable pattern (should remain unchanged)
    let literal_pattern = TermPattern::Literal(Literal::new("test"));
    let unchanged = bindings.apply_to_pattern(&literal_pattern);
    assert!(matches!(unchanged, TermPattern::Literal(_)));
}