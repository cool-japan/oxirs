//! Shape factory for creating shapes programmatically

use oxirs_core::model::NamedNode;

use crate::{
    constraints::{
        cardinality_constraints::{MaxCountConstraint, MinCountConstraint},
        string_constraints::{MaxLengthConstraint, MinLengthConstraint, PatternConstraint},
        value_constraints::{ClassConstraint, DatatypeConstraint},
    },
    paths::PropertyPath,
    targets::Target,
    Constraint, ConstraintComponentId, Shape, ShapeId,
};

/// Shape factory for creating shapes programmatically
#[derive(Debug)]
pub struct ShapeFactory;

impl ShapeFactory {
    /// Create a simple node shape with class constraint
    pub fn node_shape_with_class(shape_id: ShapeId, class_iri: NamedNode) -> Shape {
        let mut shape = Shape::node_shape(shape_id);
        shape.add_target(Target::class(class_iri.clone()));
        shape.add_constraint(
            ConstraintComponentId::new("sh:ClassConstraintComponent"),
            Constraint::Class(ClassConstraint { class_iri }),
        );
        shape
    }

    /// Create a property shape with basic constraints
    pub fn property_shape_with_constraints(
        shape_id: ShapeId,
        path: PropertyPath,
        constraints: Vec<(ConstraintComponentId, Constraint)>,
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);
        for (id, constraint) in constraints {
            shape.add_constraint(id, constraint);
        }
        shape
    }

    /// Create a string property shape with length constraints
    pub fn string_property_shape(
        shape_id: ShapeId,
        path: PropertyPath,
        min_length: Option<u32>,
        max_length: Option<u32>,
        pattern: Option<String>,
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);

        // Add datatype constraint for string
        let xsd_string = NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap();
        shape.add_constraint(
            ConstraintComponentId::new("sh:DatatypeConstraintComponent"),
            Constraint::Datatype(DatatypeConstraint {
                datatype_iri: xsd_string,
            }),
        );

        // Add length constraints
        if let Some(min) = min_length {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MinLengthConstraintComponent"),
                Constraint::MinLength(MinLengthConstraint { min_length: min }),
            );
        }

        if let Some(max) = max_length {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MaxLengthConstraintComponent"),
                Constraint::MaxLength(MaxLengthConstraint { max_length: max }),
            );
        }

        // Add pattern constraint
        if let Some(pattern_str) = pattern {
            shape.add_constraint(
                ConstraintComponentId::new("sh:PatternConstraintComponent"),
                Constraint::Pattern(PatternConstraint {
                    pattern: pattern_str,
                    flags: None,
                    message: None,
                }),
            );
        }

        shape
    }

    /// Create a cardinality-constrained property shape
    pub fn cardinality_property_shape(
        shape_id: ShapeId,
        path: PropertyPath,
        min_count: Option<u32>,
        max_count: Option<u32>,
    ) -> Shape {
        let mut shape = Shape::property_shape(shape_id, path);

        if let Some(min) = min_count {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MinCountConstraintComponent"),
                Constraint::MinCount(MinCountConstraint { min_count: min }),
            );
        }

        if let Some(max) = max_count {
            shape.add_constraint(
                ConstraintComponentId::new("sh:MaxCountConstraintComponent"),
                Constraint::MaxCount(MaxCountConstraint { max_count: max }),
            );
        }

        shape
    }
}